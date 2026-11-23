#ifndef ENTT_CUDA_REGISTRY_HPP
#define ENTT_CUDA_REGISTRY_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <cuda_runtime_api.h>
#include <entt/core/type_info.hpp>
#include <entt/entity/entity.hpp>
#include <entt/entity/registry.hpp>
#include <entt/entity/storage.hpp>
#include "contract.hpp"
#include "device.hpp"
#include "storage.hpp"

namespace entt {

template<typename Type>
class cuda_storage;

template<typename... Type>
struct cuda_view;

struct cuda_registry;

class cuda_runtime_view;

template<typename Owned, typename Get = get_t<>, typename Exclude = exclude_t<>>
struct cuda_group;

/**
 * @brief CUDA query result. Snapshot descriptors and dispatch on each call.
 * @tparam Type Required component types, possibly const-qualified.
 */
template<typename... Type>
struct cuda_view {
    cuda_registry *registry{};
    std::tuple<cuda_storage<std::remove_const_t<Type>> *...> storages{};
    std::vector<internal::cuda_pool_base *> excluded_pools{};

    template<typename Func>
    cudaError_t each(Func fn, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr) const;

    /*! @brief Checks whether an entity matches every required and no excluded pool. */
    [[nodiscard]] bool contains(const entity value) const {
        const auto required = std::apply([&](auto *...component) { return (component->contains(value) && ...); }, storages);
        const auto excluded = std::none_of(excluded_pools.begin(), excluded_pools.end(), [&](const auto *pool) { return pool->contains(value); });
        return required && excluded;
    }

    /*! @brief Returns the number of candidate entities for this query. */
    [[nodiscard]] std::size_t size_hint() const {
        if constexpr(sizeof...(Type) == 0u) {
            return registry ? static_cast<std::size_t>(registry->identity.free_list()) : 0u;
        } else {
            return std::apply([](auto *...component) { return std::min({component->size()...}); }, storages);
        }
    }

    /*! @brief Re-binds the query to a registry after the owner has been moved. */
    void rebind(cuda_registry &other) noexcept {
        registry = &other;
    }

    /*! @brief Iterates matching entities on the host, without component values. */
    template<typename Func>
    void each_entity(Func fn) const {
        if constexpr(sizeof...(Type) == 0u) {
            const auto count = registry ? static_cast<std::size_t>(registry->identity.free_list()) : 0u;
            const auto *entities = registry ? registry->identity.data() : nullptr;

            for(std::size_t pos = 0u; pos < count; ++pos) {
                if(contains(entities[pos])) {
                    fn(entities[pos]);
                }
            }
        } else {
            const auto *candidate = smallest_pool();

            for(std::size_t pos = 0u; pos < candidate->size(); ++pos) {
                if(contains(candidate->data()[pos])) {
                    fn(candidate->data()[pos]);
                }
            }
        }
    }

private:
    [[nodiscard]] const internal::cuda_pool_base *smallest_pool() const {
        const internal::cuda_pool_base *result = nullptr;
        std::apply([&](auto *...component) {
            const internal::cuda_pool_base *pools[] = {static_cast<const internal::cuda_pool_base *>(component)...};
            result = *std::min_element(std::begin(pools), std::end(pools), [](const auto *lhs, const auto *rhs) { return lhs->size() < rhs->size(); });
        },
                   storages);
        return result;
    }

    template<typename Func, std::size_t... Index>
    cudaError_t dispatch(Func fn, cudaStream_t stream, cudaEvent_t done, std::index_sequence<Index...>) const;
};

/**
 * @brief EnTT-style registry with host-owned identity and device-owned values.
 *
 * Structural operations enqueue their device updates on the caller's stream and
 * return submission status. Host metadata then describes the latest submitted
 * structure. `create` is CPU-only; no separate publication step exists.
 */
struct cuda_registry {
    template<typename...>
    friend struct cuda_view;

    friend class cuda_runtime_view;

    template<typename, typename, typename>
    friend struct cuda_group;

    /*! @brief Reused upstream identity storage (generation + free list). */
    using identity_storage = basic_storage<entity>;
    /*! @brief Pool container keyed by type hash. */
    using pool_container = std::unordered_map<id_type, std::unique_ptr<internal::cuda_pool_base>>;

    /*! @brief Underlying entity identifier. */
    using entity_type = entity;
    /*! @brief Underlying version type. */
    using version_type = entt_traits<entity>::version_type;
    /*! @brief Unsigned integer type. */
    using size_type = std::size_t;
    /*! @brief Allocator type, used by host metadata and the context. */
    using allocator_type = std::allocator<entity_type>;
    /*! @brief Type-erased pool interface used by erase_if predicates. */
    using common_type = internal::cuda_pool_base;
    /*! @brief General purpose host context. */
    using context = internal::registry_context<allocator_type>;
    /*! @brief Mutable pool enumeration. */
    using iterable = internal::cuda_pool_range<pool_container::iterator, internal::cuda_pool_base>;
    /*! @brief Constant pool enumeration. */
    using const_iterable = internal::cuda_pool_range<pool_container::const_iterator, const internal::cuda_pool_base>;

    /*! @brief Typed storage type, preserving constness. */
    template<typename Type>
    using storage_for_type = cuda_storage<std::remove_const_t<Type>>;

    identity_storage identity;
    pool_container pools;
    std::unordered_set<id_type> owned_types;
    internal::cuda_retirement retirement;
    context vars{allocator_type{}};

    entity *d_identity{};
    std::size_t identity_cap{};
    entity *d_list{};
    std::size_t list_cap{};

    mutable cudaError_t fault{cudaSuccess};

    cuda_registry() = default;
    cuda_registry(const cuda_registry &) = delete;
    cuda_registry(cuda_registry &&other) noexcept;
    ~cuda_registry();

    cuda_registry &operator=(const cuda_registry &) = delete;
    cuda_registry &operator=(cuda_registry &&other) noexcept;

    /*! @brief Creates an identifier; CPU-only, no device submission. */
    [[nodiscard]] entity create();

    /*! @brief Creates the suggested identifier if unused, else a fresh one. */
    [[nodiscard]] entity create(entity hint);

    /*! @brief Assigns each element in a range an identifier. */
    template<typename It>
    void create(It first, It last);

    /*! @brief Destroys an entity, removing every component and recycling its ID. */
    cudaError_t destroy(entity value, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    /*! @brief Destroys an entity and sets the recycled version. */
    cudaError_t destroy(entity value, version_type version, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    /*! @brief Destroys every entity in a range. */
    template<typename It>
    cudaError_t destroy(It first, It last, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    /**
     * @brief Emplaces using the default stream without a completion event.
     * @note A leading native stream or nullptr is reserved for stream selection.
     */
    template<typename Type, typename... Args>
    requires (!internal::cuda_leading_handle<cudaStream_t, Args...> && internal::cuda_component_constructible<Type, Args...>)
    cudaError_t emplace(const entity value, Args &&...args) {
        return emplace_impl<Type>(value, nullptr, nullptr, std::forward<Args>(args)...);
    }

    /**
     * @brief Emplaces using a native stream without a completion event.
     * @note A leading native event or nullptr after the stream selects done.
     */
    template<typename Type, typename... Args>
    requires (!internal::cuda_leading_handle<cudaEvent_t, Args...> && internal::cuda_component_constructible<Type, Args...>)
    cudaError_t emplace(const entity value, const cudaStream_t stream, Args &&...args) {
        return emplace_impl<Type>(value, stream, nullptr, std::forward<Args>(args)...);
    }

    /** @brief Emplaces using a native stream and an optional completion event. */
    template<typename Type, typename... Args>
    requires internal::cuda_component_constructible<Type, Args...>
    cudaError_t emplace(const entity value, const cudaStream_t stream, const cudaEvent_t done, Args &&...args) {
        return emplace_impl<Type>(value, stream, done, std::forward<Args>(args)...);
    }

    /*! @brief Assigns or replaces the given element for an entity. */
    template<typename Type, typename... Args>
    requires (!internal::cuda_leading_handle<cudaStream_t, Args...> && internal::cuda_component_constructible<Type, Args...>)
    cudaError_t emplace_or_replace(const entity value, Args &&...args) {
        return emplace_or_replace_impl<Type>(value, nullptr, nullptr, std::forward<Args>(args)...);
    }

    /*! @brief Assigns or replaces the given element on a native stream. */
    template<typename Type, typename... Args>
    requires (!internal::cuda_leading_handle<cudaEvent_t, Args...> && internal::cuda_component_constructible<Type, Args...>)
    cudaError_t emplace_or_replace(const entity value, const cudaStream_t stream, Args &&...args) {
        return emplace_or_replace_impl<Type>(value, stream, nullptr, std::forward<Args>(args)...);
    }

    /*! @brief Assigns or replaces the given element on a stream and event. */
    template<typename Type, typename... Args>
    requires internal::cuda_component_constructible<Type, Args...>
    cudaError_t emplace_or_replace(const entity value, const cudaStream_t stream, const cudaEvent_t done, Args &&...args) {
        return emplace_or_replace_impl<Type>(value, stream, done, std::forward<Args>(args)...);
    }

    /*! @brief Emplaces only if absent; the caller accesses via get/try_get. */
    template<typename Type, typename... Args>
    requires (!internal::cuda_leading_handle<cudaStream_t, Args...> && internal::cuda_component_constructible<Type, Args...>)
    cudaError_t get_or_emplace(const entity value, Args &&...args) {
        return get_or_emplace_impl<Type>(value, nullptr, nullptr, std::forward<Args>(args)...);
    }

    /*! @brief Emplaces only if absent on a native stream. */
    template<typename Type, typename... Args>
    requires (!internal::cuda_leading_handle<cudaEvent_t, Args...> && internal::cuda_component_constructible<Type, Args...>)
    cudaError_t get_or_emplace(const entity value, const cudaStream_t stream, Args &&...args) {
        return get_or_emplace_impl<Type>(value, stream, nullptr, std::forward<Args>(args)...);
    }

    /*! @brief Emplaces only if absent on a stream and event. */
    template<typename Type, typename... Args>
    requires internal::cuda_component_constructible<Type, Args...>
    cudaError_t get_or_emplace(const entity value, const cudaStream_t stream, const cudaEvent_t done, Args &&...args) {
        return get_or_emplace_impl<Type>(value, stream, done, std::forward<Args>(args)...);
    }

    /*! @brief Replaces an existing element with a freshly constructed value. */
    template<typename Type, typename... Args>
    requires (!internal::cuda_leading_handle<cudaStream_t, Args...> && internal::cuda_component_constructible<Type, Args...>)
    cudaError_t replace(const entity value, Args &&...args) {
        return replace_impl<Type>(value, nullptr, nullptr, std::forward<Args>(args)...);
    }

    /*! @brief Replaces an existing element on a native stream. */
    template<typename Type, typename... Args>
    requires (!internal::cuda_leading_handle<cudaEvent_t, Args...> && internal::cuda_component_constructible<Type, Args...>)
    cudaError_t replace(const entity value, const cudaStream_t stream, Args &&...args) {
        return replace_impl<Type>(value, stream, nullptr, std::forward<Args>(args)...);
    }

    /*! @brief Replaces an existing element on a stream and event. */
    template<typename Type, typename... Args>
    requires internal::cuda_component_constructible<Type, Args...>
    cudaError_t replace(const entity value, const cudaStream_t stream, const cudaEvent_t done, Args &&...args) {
        return replace_impl<Type>(value, stream, done, std::forward<Args>(args)...);
    }

    /*! @brief Applies ordered device-callable functions to one component. */
    template<typename Type, typename... Func>
    cudaError_t patch(const entity value, Func &&...func) {
        return patch_impl<Type>(value, nullptr, nullptr, std::forward<Func>(func)...);
    }

    /*! @brief Applies ordered device-callable functions on a native stream. */
    template<typename Type, typename... Func>
    cudaError_t patch(const entity value, const cudaStream_t stream, Func &&...func) {
        return patch_impl<Type>(value, stream, nullptr, std::forward<Func>(func)...);
    }

    /*! @brief Applies ordered device-callable functions on a stream and event. */
    template<typename Type, typename... Func>
    cudaError_t patch(const entity value, const cudaStream_t stream, const cudaEvent_t done, Func &&...func) {
        return patch_impl<Type>(value, stream, done, std::forward<Func>(func)...);
    }

    /*! @brief Assigns each entity in a range the given element. */
    template<typename Type, typename It>
    cudaError_t insert(It first, It last, const Type &value, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    /*! @brief Assigns each entity in a range one of the given elements. */
    template<typename Type, typename EIt, typename CIt>
    requires std::is_same_v<typename std::iterator_traits<CIt>::value_type, Type>
    cudaError_t insert(EIt first, EIt last, CIt from, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    /*! @brief Removes the given elements from an entity (tolerant). */
    template<typename Type, typename... Other>
    cudaError_t remove(entity value, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    /*! @brief Removes the given elements from every entity in a range (tolerant). */
    template<typename Type, typename... Other, typename It>
    cudaError_t remove(It first, It last, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    /*! @brief Erases the given elements from an entity (strict). */
    template<typename Type, typename... Other>
    cudaError_t erase(entity value, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    /*! @brief Erases the given elements from every entity in a range (strict). */
    template<typename Type, typename... Other, typename It>
    cudaError_t erase(It first, It last, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    /*! @brief Erases elements satisfying a predicate, receiving pool id and metadata. */
    template<typename Func>
    cudaError_t erase_if(entity value, Func func, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    /*! @brief Clears the given pools, or the whole registry for an empty pack. */
    template<typename... Type>
    cudaError_t clear(cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    template<typename Type>
    requires (internal::cuda_component<Type> && !std::is_empty_v<Type>)
    cudaError_t get(entity value, Type *pinned_host_dst, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    template<typename Type>
    requires (internal::cuda_component<Type> && !std::is_empty_v<Type>)
    cudaError_t get(entity value, Type *pinned_host_dst, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr) const;

    template<typename Type>
    requires (internal::cuda_component<Type> && !std::is_empty_v<Type>)
    cudaError_t assign(entity value, const Type *pinned_host_src, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    /*! @brief Returns non-owning device pointers to the given elements, or null. */
    template<typename... Type>
    [[nodiscard]] auto try_get(entity value) noexcept;

    /*! @brief Returns non-owning const device pointers, or null. */
    template<typename... Type>
    [[nodiscard]] auto try_get(entity value) const noexcept;

    /*! @brief Checks if an identifier refers to a valid entity. */
    [[nodiscard]] bool valid(const entity value) const noexcept {
        return static_cast<std::size_t>(identity.find(value).index()) < identity.free_list();
    }

    /*! @brief Returns the version of an identifier, or the tombstone version. */
    [[nodiscard]] version_type current(const entity value) const noexcept {
        return identity.current(value);
    }

    /*! @brief Checks if an entity has no elements assigned. */
    [[nodiscard]] bool orphan(const entity value) const noexcept {
        for(const auto &[key, pool]: pools) {
            if(pool->contains(value)) {
                return false;
            }
        }

        return true;
    }

    /*! @brief Checks if an entity is part of all the given pools. */
    template<typename... Type>
    [[nodiscard]] bool all_of(const entity value) const noexcept {
        return (has<Type>(value) && ...);
    }

    /*! @brief Checks if an entity is part of at least one given pool. */
    template<typename... Type>
    [[nodiscard]] bool any_of(const entity value) const noexcept {
        return (has<Type>(value) || ...);
    }

    template<typename... Type, typename... Exclude>
    [[nodiscard]] cuda_view<Type...> view(exclude_t<Exclude...> = exclude_t{});

    /*! @brief Returns a runtime view bound to this registry. */
    [[nodiscard]] cuda_runtime_view runtime_view() noexcept;

    /*! @brief Returns a group for the given owned/observed/excluded elements. */
    template<typename... Owned, typename... Get, typename... Exclude>
    [[nodiscard]] cuda_group<owned_t<Owned...>, get_t<Get...>, exclude_t<Exclude...>> group(get_t<Get...> = get_t{}, exclude_t<Exclude...> = exclude_t{});

    /*! @brief Returns a group for the given elements without creating pools. */
    template<typename... Owned, typename... Get, typename... Exclude>
    [[nodiscard]] cuda_group<owned_t<>, get_t<const Owned..., const Get...>, exclude_t<const Exclude...>> group_if_exists(get_t<Get...> = get_t{}, exclude_t<Exclude...> = exclude_t{}) const;

    /*! @brief Checks whether the given elements are owned by a group. */
    template<typename... Type>
    [[nodiscard]] bool owned() const noexcept {
        return (owned_types.contains(type_hash<std::remove_const_t<Type>>::value()) || ...);
    }

    /*! @brief Removes tombstones; a no-op for swap-and-pop pools. */
    template<typename... Type>
    cudaError_t compact(cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    template<typename Type>
    requires internal::cuda_component<Type>
    [[nodiscard]] cuda_storage<Type> &storage(const id_type id = type_hash<Type>::value()) {
        return pool<Type>(id);
    }

    template<typename Type>
    requires internal::cuda_component<Type>
    [[nodiscard]] const cuda_storage<Type> *storage(const id_type id = type_hash<Type>::value()) const {
        return find_pool<Type>(id);
    }

    /*! @brief Returns a type-erased pool by id, or null if absent. */
    [[nodiscard]] common_type *storage(const id_type id) noexcept {
        const auto it = pools.find(id);
        return (it != pools.end()) ? it->second.get() : nullptr;
    }

    /*! @brief Returns a type-erased pool by id, or null if absent. */
    [[nodiscard]] const common_type *storage(const id_type id) const noexcept {
        const auto it = pools.find(id);
        return (it != pools.end()) ? it->second.get() : nullptr;
    }

    /*! @brief Enumerates component pools, excluding the entity storage. */
    [[nodiscard]] iterable storage() noexcept {
        return iterable{pools.begin(), pools.end()};
    }

    /*! @brief Enumerates component pools, excluding the entity storage. */
    [[nodiscard]] const_iterable storage() const noexcept {
        return const_iterable{pools.begin(), pools.end()};
    }

    /*! @brief Removes a component pool, retiring its device allocations. */
    cudaError_t reset(const id_type id, cudaStream_t stream = nullptr);

    /*! @brief Returns the context object. */
    [[nodiscard]] context &ctx() noexcept {
        return vars;
    }

    /*! @brief Returns the context object. */
    [[nodiscard]] const context &ctx() const noexcept {
        return vars;
    }

private:
    cudaError_t fail(const cudaError_t err) const noexcept {
        if(fault == cudaSuccess) {
            fault = err;
        }

        return err;
    }

    static cudaError_t record_done(cudaEvent_t done, cudaStream_t stream) noexcept {
        return (done != nullptr) ? cudaEventRecord(done, stream) : cudaSuccess;
    }

    template<typename Type>
    [[nodiscard]] cuda_storage<Type> &pool(const id_type id = type_hash<Type>::value());

    template<typename Type>
    [[nodiscard]] const cuda_storage<Type> *find_pool(const id_type id = type_hash<Type>::value()) const noexcept {
        const auto it = pools.find(id);
        return (it != pools.end() && it->second->type() == type_hash<Type>::value()) ? static_cast<const cuda_storage<Type> *>(it->second.get()) : nullptr;
    }

    template<typename Type>
    [[nodiscard]] bool has(const entity value) const noexcept {
        const auto *component = find_pool<Type>();
        return component && component->contains(value);
    }

    cudaError_t upload_identity(std::size_t &count, const entity *&out, cudaStream_t stream);

    cudaError_t upload_list(const entity *src, std::size_t count, const entity *&out, cudaStream_t stream);

    template<typename Type, typename... Args>
    requires internal::cuda_component_constructible<Type, Args...>
    cudaError_t emplace_impl(entity value, cudaStream_t stream, cudaEvent_t done, Args &&...args);

    template<typename Type, typename... Args>
    requires internal::cuda_component_constructible<Type, Args...>
    cudaError_t emplace_or_replace_impl(entity value, cudaStream_t stream, cudaEvent_t done, Args &&...args);

    template<typename Type, typename... Args>
    requires internal::cuda_component_constructible<Type, Args...>
    cudaError_t get_or_emplace_impl(entity value, cudaStream_t stream, cudaEvent_t done, Args &&...args);

    template<typename Type, typename... Args>
    requires internal::cuda_component_constructible<Type, Args...>
    cudaError_t replace_impl(entity value, cudaStream_t stream, cudaEvent_t done, Args &&...args);

    template<typename Type, typename... Func>
    cudaError_t patch_impl(entity value, cudaStream_t stream, cudaEvent_t done, Func &&...func);
};

inline cuda_registry::cuda_registry(cuda_registry &&other) noexcept
    : identity{std::move(other.identity)},
      pools{std::move(other.pools)},
      owned_types{std::move(other.owned_types)},
      retirement{std::move(other.retirement)},
      vars{std::move(other.vars)},
      d_identity{std::exchange(other.d_identity, nullptr)},
      identity_cap{std::exchange(other.identity_cap, 0u)},
      d_list{std::exchange(other.d_list, nullptr)},
      list_cap{std::exchange(other.list_cap, 0u)},
      fault{other.fault} {
    for(auto &&[key, pool]: pools) {
        pool->rebind_retirement(retirement);
    }
}

inline cuda_registry &cuda_registry::operator=(cuda_registry &&other) noexcept {
    if(this != &other) {
        retirement.drain();
        cudaDeviceSynchronize();

        for(auto &&[key, pool]: pools) {
            pool->release();
        }

        if(d_identity != nullptr) {
            cudaFree(d_identity);
        }

        identity = std::move(other.identity);
        pools = std::move(other.pools);
        owned_types = std::move(other.owned_types);
        retirement = std::move(other.retirement);
        vars = std::move(other.vars);
        d_identity = std::exchange(other.d_identity, nullptr);
        identity_cap = std::exchange(other.identity_cap, 0u);
        d_list = std::exchange(other.d_list, nullptr);
        list_cap = std::exchange(other.list_cap, 0u);
        fault = other.fault;

        for(auto &&[key, pool]: pools) {
            pool->rebind_retirement(retirement);
        }
    }

    return *this;
}

inline cuda_registry::~cuda_registry() {
    retirement.drain();
    // Conservative: wait for all library-owned work before releasing buffers.
    cudaDeviceSynchronize();

    for(auto &&[key, pool]: pools) {
        pool->release();
    }

    if(d_identity != nullptr) {
        cudaFree(d_identity);
        d_identity = nullptr;
    }

    if(d_list != nullptr) {
        cudaFree(d_list);
        d_list = nullptr;
    }
}

[[nodiscard]] inline entity cuda_registry::create() {
    if(fault != cudaSuccess) {
        throw std::logic_error("entt::cuda_registry is in a fault state");
    }

    constexpr auto max_index = static_cast<std::size_t>(entt_traits<entity>::to_entity(entt::null)) + 1u;

    if(identity.free_list() == identity.size() && identity.size() >= max_index) {
        throw std::length_error("No more entities available");
    }

    return identity.generate();
}

[[nodiscard]] inline entity cuda_registry::create(const entity hint) {
    if(fault != cudaSuccess) {
        throw std::logic_error("entt::cuda_registry is in a fault state");
    }

    constexpr auto max_index = static_cast<std::size_t>(entt_traits<entity>::to_entity(entt::null)) + 1u;

    if(identity.free_list() == identity.size() && identity.size() >= max_index) {
        throw std::length_error("No more entities available");
    }

    return identity.generate(hint);
}

template<typename It>
inline void cuda_registry::create(It first, It last) {
    if(fault != cudaSuccess) {
        throw std::logic_error("entt::cuda_registry is in a fault state");
    }

    identity.generate(std::move(first), std::move(last));
}

inline cudaError_t cuda_registry::destroy(const entity value, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if(!valid(value)) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err = cudaSuccess;

    for(auto &&[key, pool]: pools) {
        if(const auto curr = pool->enqueue_remove(value, stream); curr != cudaSuccess && err == cudaSuccess) {
            err = curr;
        }
    }

    if(err != cudaSuccess) {
        return fail(err);
    }

    identity.erase(value);

    if(const auto curr = record_done(done, stream); curr != cudaSuccess) {
        return fail(curr);
    }

    return cudaSuccess;
}

inline cudaError_t cuda_registry::destroy(const entity value, const version_type version, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = destroy(value, stream); err != cudaSuccess) {
        return err;
    }

    const auto elem = entt_traits<entity>::construct(entt_traits<entity>::to_entity(value), version);
    identity.bump((elem == entt::tombstone) ? entt_traits<entity>::next(elem) : elem);

    if(const auto curr = record_done(done, stream); curr != cudaSuccess) {
        return fail(curr);
    }

    return cudaSuccess;
}

template<typename It>
inline cudaError_t cuda_registry::destroy(It first, It last, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    std::vector<entity> range(first, last);

    for(const auto value: range) {
        if(!valid(value)) {
            return cudaErrorInvalidValue;
        }
    }

    cudaError_t err = cudaSuccess;

    for(const auto value: range) {
        for(auto &&[key, pool]: pools) {
            if(const auto curr = pool->enqueue_remove(value, stream); curr != cudaSuccess && err == cudaSuccess) {
                err = curr;
            }
        }

        identity.erase(value);
    }

    if(err != cudaSuccess) {
        return fail(err);
    }

    if(const auto curr = record_done(done, stream); curr != cudaSuccess) {
        return fail(curr);
    }

    return cudaSuccess;
}

template<typename Type>
[[nodiscard]] inline cuda_storage<Type> &cuda_registry::pool(const id_type id) {
    static_assert(!std::is_const_v<Type>, "Pool types are unqualified and non-const");

    if(const auto it = pools.find(id); it != pools.end()) {
        ENTT_ASSERT(it->second->type() == type_hash<Type>::value(), "Unexpected type");
        return static_cast<cuda_storage<Type> &>(*it->second);
    }

    auto ptr = std::make_unique<cuda_storage<Type>>(retirement);
    auto *raw = ptr.get();
    pools.emplace(id, std::move(ptr));
    return *raw;
}

inline cudaError_t cuda_registry::reset(const id_type id, cudaStream_t stream) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if(const auto it = pools.find(id); it != pools.end()) {
        it->second->reset(stream);
        pools.erase(it);
    }

    return cudaSuccess;
}

inline cudaError_t cuda_registry::upload_identity(std::size_t &count, const entity *&out, cudaStream_t stream) {
    count = identity.free_list();
    out = nullptr;

    if(count == 0u) {
        return cudaSuccess;
    }

    if(count > identity_cap) {
        entity *next{};

        if(cudaMalloc(reinterpret_cast<void **>(&next), count * sizeof(entity)) != cudaSuccess) {
            return cudaErrorMemoryAllocation;
        }

        retirement.retire(d_identity, stream);
        d_identity = next;
        identity_cap = count;
    }

    out = d_identity;
    return cudaMemcpyAsync(d_identity, identity.data(), count * sizeof(entity), cudaMemcpyHostToDevice, stream);
}

inline cudaError_t cuda_registry::upload_list(const entity *src, const std::size_t count, const entity *&out, cudaStream_t stream) {
    out = nullptr;

    if(count == 0u) {
        return cudaSuccess;
    }

    if(count > list_cap) {
        entity *next{};

        if(cudaMalloc(reinterpret_cast<void **>(&next), count * sizeof(entity)) != cudaSuccess) {
            return cudaErrorMemoryAllocation;
        }

        retirement.retire(d_list, stream);
        d_list = next;
        list_cap = count;
    }

    out = d_list;
    return cudaMemcpyAsync(d_list, src, count * sizeof(entity), cudaMemcpyHostToDevice, stream);
}

template<typename Type, typename... Args>
requires internal::cuda_component_constructible<Type, Args...>
inline cudaError_t cuda_registry::emplace_impl(const entity value, cudaStream_t stream, cudaEvent_t done, Args &&...args) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if(!valid(value)) {
        return cudaErrorInvalidValue;
    }

    auto &component = pool<Type>();

    if(component.contains(value)) {
        return cudaErrorInvalidValue;
    }

    if(const auto err = component.emplace(value, stream, std::forward<Args>(args)...); err != cudaSuccess) {
        return fail(err);
    }

    if(const auto err = record_done(done, stream); err != cudaSuccess) {
        return fail(err);
    }

    return cudaSuccess;
}

template<typename Type, typename... Args>
requires internal::cuda_component_constructible<Type, Args...>
inline cudaError_t cuda_registry::emplace_or_replace_impl(const entity value, cudaStream_t stream, cudaEvent_t done, Args &&...args) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if(!valid(value)) {
        return cudaErrorInvalidValue;
    }

    auto &component = pool<Type>();

    if(const auto err = (component.contains(value) ? component.replace(value, stream, std::forward<Args>(args)...) : component.emplace(value, stream, std::forward<Args>(args)...)); err != cudaSuccess) {
        return fail(err);
    }

    if(const auto err = record_done(done, stream); err != cudaSuccess) {
        return fail(err);
    }

    return cudaSuccess;
}

template<typename Type, typename... Args>
requires internal::cuda_component_constructible<Type, Args...>
inline cudaError_t cuda_registry::get_or_emplace_impl(const entity value, cudaStream_t stream, cudaEvent_t done, Args &&...args) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if(!valid(value)) {
        return cudaErrorInvalidValue;
    }

    auto &component = pool<Type>();

    if(!component.contains(value)) {
        if(const auto err = component.emplace(value, stream, std::forward<Args>(args)...); err != cudaSuccess) {
            return fail(err);
        }
    }

    if(const auto err = record_done(done, stream); err != cudaSuccess) {
        return fail(err);
    }

    return cudaSuccess;
}

template<typename Type, typename... Args>
requires internal::cuda_component_constructible<Type, Args...>
inline cudaError_t cuda_registry::replace_impl(const entity value, cudaStream_t stream, cudaEvent_t done, Args &&...args) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if(!valid(value)) {
        return cudaErrorInvalidValue;
    }

    auto &component = pool<Type>();

    if(!component.contains(value)) {
        return cudaErrorInvalidValue;
    }

    if(const auto err = component.replace(value, stream, std::forward<Args>(args)...); err != cudaSuccess) {
        return fail(err);
    }

    if(const auto err = record_done(done, stream); err != cudaSuccess) {
        return fail(err);
    }

    return cudaSuccess;
}

template<typename Type, typename It>
inline cudaError_t cuda_registry::insert(It first, It last, const Type &value, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    std::vector<entity> range(first, last);

    for(const auto entity: range) {
        if(!valid(entity)) {
            return cudaErrorInvalidValue;
        }
    }

    auto &component = pool<Type>();

    for(const auto entity: range) {
        if(component.contains(entity)) {
            return cudaErrorInvalidValue;
        }
    }

    for(const auto entity: range) {
        if(const auto err = component.emplace(entity, stream, value); err != cudaSuccess) {
            return fail(err);
        }
    }

    if(const auto err = record_done(done, stream); err != cudaSuccess) {
        return fail(err);
    }

    return cudaSuccess;
}

template<typename Type, typename EIt, typename CIt>
requires std::is_same_v<typename std::iterator_traits<CIt>::value_type, Type>
inline cudaError_t cuda_registry::insert(EIt first, EIt last, CIt from, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    std::vector<entity> range(first, last);

    for(const auto entity: range) {
        if(!valid(entity)) {
            return cudaErrorInvalidValue;
        }
    }

    auto &component = pool<Type>();

    for(const auto entity: range) {
        if(component.contains(entity)) {
            return cudaErrorInvalidValue;
        }
    }

    for(const auto entity: range) {
        if(const auto err = component.emplace(entity, stream, *from); err != cudaSuccess) {
            return fail(err);
        }

        ++from;
    }

    if(const auto err = record_done(done, stream); err != cudaSuccess) {
        return fail(err);
    }

    return cudaSuccess;
}

template<typename Type, typename... Other>
inline cudaError_t cuda_registry::remove(const entity value, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if(!valid(value)) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err = cudaSuccess;
    const auto enqueue = [&](auto &component) {
        if(const auto curr = component.enqueue_remove(value, stream); curr != cudaSuccess && err == cudaSuccess) {
            err = curr;
        }
    };

    enqueue(pool<std::remove_const_t<Type>>());
    ((enqueue(pool<std::remove_const_t<Other>>())), ...);

    if(err != cudaSuccess) {
        return fail(err);
    }

    if(const auto curr = record_done(done, stream); curr != cudaSuccess) {
        return fail(curr);
    }

    return cudaSuccess;
}

template<typename Type, typename... Other, typename It>
inline cudaError_t cuda_registry::remove(It first, It last, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    std::vector<entity> range(first, last);

    for(const auto entity: range) {
        if(!valid(entity)) {
            return cudaErrorInvalidValue;
        }
    }

    cudaError_t err = cudaSuccess;
    const auto enqueue = [&](auto &component, const entity entity) {
        if(const auto curr = component.enqueue_remove(entity, stream); curr != cudaSuccess && err == cudaSuccess) {
            err = curr;
        }
    };

    for(const auto entity: range) {
        enqueue(pool<std::remove_const_t<Type>>(), entity);
        ((enqueue(pool<std::remove_const_t<Other>>(), entity)), ...);
    }

    if(err != cudaSuccess) {
        return fail(err);
    }

    if(const auto curr = record_done(done, stream); curr != cudaSuccess) {
        return fail(curr);
    }

    return cudaSuccess;
}

template<typename Type, typename... Other>
inline cudaError_t cuda_registry::erase(const entity value, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if(!valid(value)) {
        return cudaErrorInvalidValue;
    }

    if(!(has<Type>(value) && (has<Other>(value) && ...))) {
        return cudaErrorInvalidValue;
    }

    return remove<Type, Other...>(value, stream, done);
}

template<typename Type, typename... Other, typename It>
inline cudaError_t cuda_registry::erase(It first, It last, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    std::vector<entity> range(first, last);

    for(const auto entity: range) {
        if(!valid(entity) || !(has<Type>(entity) && (has<Other>(entity) && ...))) {
            return cudaErrorInvalidValue;
        }
    }

    return remove<Type, Other...>(range.begin(), range.end(), stream, done);
}

template<typename Func>
inline cudaError_t cuda_registry::erase_if(const entity value, Func func, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if(!valid(value)) {
        return cudaErrorInvalidValue;
    }

    cudaError_t err = cudaSuccess;

    for(auto &&[id, pool]: pools) {
        if(pool->contains(value) && func(id, std::as_const(*pool))) {
            if(const auto curr = pool->enqueue_remove(value, stream); curr != cudaSuccess && err == cudaSuccess) {
                err = curr;
            }
        }
    }

    if(err != cudaSuccess) {
        return fail(err);
    }

    if(const auto curr = record_done(done, stream); curr != cudaSuccess) {
        return fail(curr);
    }

    return cudaSuccess;
}

template<typename... Type>
inline cudaError_t cuda_registry::clear(cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if constexpr(sizeof...(Type) == 0u) {
        for(auto &&[key, pool]: pools) {
            pool->clear();
        }

        std::vector<entity> alive(identity.data(), identity.data() + identity.free_list());

        for(const auto entity: alive) {
            identity.erase(entity);
        }
    } else {
        (pool<Type>().clear(), ...);
    }

    if(const auto err = record_done(done, stream); err != cudaSuccess) {
        return fail(err);
    }

    return cudaSuccess;
}

template<typename Type>
requires (internal::cuda_component<Type> && !std::is_empty_v<Type>)
inline cudaError_t cuda_registry::get(const entity value, Type *pinned_host_dst, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if(!valid(value) || pinned_host_dst == nullptr) {
        return cudaErrorInvalidValue;
    }

    auto &component = pool<Type>();

    if(!component.contains(value)) {
        return cudaErrorInvalidValue;
    }

    if(const auto err = component.enqueue_get(value, pinned_host_dst, stream); err != cudaSuccess) {
        return fail(err);
    }

    if(const auto err = record_done(done, stream); err != cudaSuccess) {
        return fail(err);
    }

    return cudaSuccess;
}

template<typename Type>
requires (internal::cuda_component<Type> && !std::is_empty_v<Type>)
inline cudaError_t cuda_registry::get(const entity value, Type *pinned_host_dst, cudaStream_t stream, cudaEvent_t done) const {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if(!valid(value) || pinned_host_dst == nullptr) {
        return cudaErrorInvalidValue;
    }

    const auto *component = find_pool<Type>();

    if(component == nullptr || !component->contains(value)) {
        return cudaErrorInvalidValue;
    }

    if(const auto err = component->enqueue_get(value, pinned_host_dst, stream); err != cudaSuccess) {
        return fail(err);
    }

    if(const auto err = record_done(done, stream); err != cudaSuccess) {
        return fail(err);
    }

    return cudaSuccess;
}

template<typename Type>
requires (internal::cuda_component<Type> && !std::is_empty_v<Type>)
inline cudaError_t cuda_registry::assign(const entity value, const Type *pinned_host_src, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if(!valid(value) || pinned_host_src == nullptr) {
        return cudaErrorInvalidValue;
    }

    auto &component = pool<Type>();

    if(!component.contains(value)) {
        return cudaErrorInvalidValue;
    }

    if(const auto err = component.enqueue_assign(value, pinned_host_src, stream); err != cudaSuccess) {
        return fail(err);
    }

    if(const auto err = record_done(done, stream); err != cudaSuccess) {
        return fail(err);
    }

    return cudaSuccess;
}

template<typename... Type>
[[nodiscard]] inline auto cuda_registry::try_get(const entity value) noexcept {
    if constexpr(sizeof...(Type) == 1u) {
        using element = std::tuple_element_t<0, std::tuple<Type...>>;
        auto &component = pool<std::remove_const_t<element>>();
        return component.template try_get<element>(value);
    } else {
        return std::make_tuple(try_get<Type>(value)...);
    }
}

template<typename... Type>
[[nodiscard]] inline auto cuda_registry::try_get(const entity value) const noexcept {
    if constexpr(sizeof...(Type) == 1u) {
        using element = std::tuple_element_t<0, std::tuple<Type...>>;
        const auto *component = find_pool<std::remove_const_t<element>>();
        return component ? component->template try_get<const element>(value) : nullptr;
    } else {
        return std::make_tuple(try_get<Type>(value)...);
    }
}

template<typename... Type, typename... Exclude>
[[nodiscard]] inline cuda_view<Type...> cuda_registry::view(exclude_t<Exclude...>) {
    cuda_view<Type...> result;
    result.registry = this;
    result.storages = std::make_tuple(static_cast<cuda_storage<std::remove_const_t<Type>> *>(&pool<std::remove_const_t<Type>>())...);
    result.excluded_pools = {static_cast<internal::cuda_pool_base *>(&pool<std::remove_const_t<Exclude>>())...};
    return result;
}

/**
 * @brief Generic runtime view dispatching entity identifiers only.
 *
 * Required and excluded pools are added at runtime. The device callback receives
 * only the entity identifier; there is no runtime device compilation.
 */
class cuda_runtime_view {
    cuda_registry *registry{};
    std::vector<internal::cuda_pool_base *> pools;
    std::vector<internal::cuda_pool_base *> filter;

public:
    cuda_runtime_view() = default;
    explicit cuda_runtime_view(cuda_registry *owner) noexcept
        : registry{owner} {}

    /*! @brief Adds the default-named pool of the given type, keeping the smallest first. */
    template<typename Type>
    cuda_runtime_view &iterate() {
        auto *component = static_cast<internal::cuda_pool_base *>(&registry->storage<Type>());

        if(pools.empty() || component->size() >= pools.front()->size()) {
            pools.push_back(component);
        } else {
            pools.push_back(std::exchange(pools.front(), component));
        }

        return *this;
    }

    /*! @brief Adds an opaque pool, keeping the smallest first. */
    cuda_runtime_view &iterate(internal::cuda_pool_base &base) {
        if(pools.empty() || base.size() >= pools.front()->size()) {
            pools.push_back(&base);
        } else {
            pools.push_back(std::exchange(pools.front(), &base));
        }

        return *this;
    }

    /*! @brief Adds the default-named pool of the given type as a filter. */
    template<typename Type>
    cuda_runtime_view &exclude() {
        filter.push_back(&registry->storage<Type>());
        return *this;
    }

    /*! @brief Adds an opaque pool as a filter. */
    cuda_runtime_view &exclude(internal::cuda_pool_base &base) {
        filter.push_back(&base);
        return *this;
    }

    /*! @brief Clears the view. */
    void clear() noexcept {
        pools.clear();
        filter.clear();
    }

    /*! @brief Estimates the number of entities iterated by the view. */
    [[nodiscard]] std::size_t size_hint() const noexcept {
        return pools.empty() ? 0u : pools.front()->size();
    }

    /*! @brief Checks if the view contains an entity. */
    [[nodiscard]] bool contains(const entity value) const noexcept {
        return std::all_of(pools.begin(), pools.end(), [&](const auto *pool) { return pool->contains(value); })
               && std::none_of(filter.begin(), filter.end(), [&](const auto *pool) { return pool && pool->contains(value); });
    }

    /*! @brief Checks whether the view is initialized. */
    [[nodiscard]] explicit operator bool() const noexcept {
        return !(pools.empty() && filter.empty());
    }

    template<typename Func>
    cudaError_t each(Func fn, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr) const;
};

[[nodiscard]] inline cuda_runtime_view cuda_registry::runtime_view() noexcept {
    return cuda_runtime_view{this};
}

namespace internal {

/*! @brief Computes the entities matching a set of required and excluded pools. */
inline std::vector<entity> group_members(const std::vector<cuda_pool_base *> &required, const std::vector<cuda_pool_base *> &excluded,
                                         const entity *alive, const std::size_t alive_count) {
    std::vector<entity> members;

    if(required.empty()) {
        for(std::size_t pos = 0u; pos < alive_count; ++pos) {
            const entity value = alive[pos];
            const auto rejected = std::any_of(excluded.begin(), excluded.end(), [&](const auto *pool) { return pool && pool->contains(value); });

            if(!rejected) {
                members.push_back(value);
            }
        }

        return members;
    }

    if(std::any_of(required.begin(), required.end(), [](const auto *pool) { return pool == nullptr; })) {
        return members;
    }

    const auto *candidate = *std::min_element(required.begin(), required.end(), [](const auto *lhs, const auto *rhs) { return lhs->size() < rhs->size(); });

    for(std::size_t pos = 0u; pos < candidate->size(); ++pos) {
        const entity value = candidate->data()[pos];
        bool valid = true;

        for(const auto *pool: required) {
            if(!pool->contains(value)) {
                valid = false;
                break;
            }
        }

        if(valid) {
            for(const auto *pool: excluded) {
                if(pool->contains(value)) {
                    valid = false;
                    break;
                }
            }
        }

        if(valid) {
            members.push_back(value);
        }
    }

    return members;
}

/*! @brief Arranges a storage so matching entities occupy [0, matching.size()). */
template<typename Storage>
void arrange_storage(Storage &storage, const std::vector<entity> &matching, cudaStream_t stream) {
    for(std::size_t pos = 0u; pos < matching.size(); ++pos) {
        const auto idx = storage.index(matching[pos]);

        if(idx != pos) {
            storage.swap_elements(idx, pos, stream);
        }
    }
}

} // namespace internal

/**
 * @brief Non-owning group: a cached list of matching entities.
 * @tparam Get Observed component types, possibly const-qualified.
 * @tparam Exclude Excluded component types.
 */
template<typename... Get, typename... Exclude>
struct cuda_group<owned_t<>, get_t<Get...>, exclude_t<Exclude...>> {
    cuda_registry *registry{};
    std::tuple<cuda_storage<std::remove_const_t<Get>> *...> get_storages{};
    std::vector<internal::cuda_pool_base *> excluded_pools{};
    std::vector<entity> members{};

    cuda_group() = default;

    cuda_group(cuda_registry *owner, std::tuple<cuda_storage<std::remove_const_t<Get>> *...> get, std::vector<internal::cuda_pool_base *> exclude)
        : registry{owner},
          get_storages{get},
          excluded_pools{std::move(exclude)} {
        std::vector<internal::cuda_pool_base *> required;
        std::apply([&](auto *...component) { (required.push_back(static_cast<internal::cuda_pool_base *>(component)), ...); }, get_storages);
        members = internal::group_members(required, excluded_pools, registry->identity.data(), registry->identity.free_list());
    }

    /*! @brief Returns the number of entities in the group. */
    [[nodiscard]] std::size_t size() const noexcept {
        return members.size();
    }

    /*! @brief Checks whether the group is empty. */
    [[nodiscard]] bool empty() const noexcept {
        return members.empty();
    }

    /*! @brief Checks if the group contains an entity. */
    [[nodiscard]] bool contains(const entity value) const noexcept {
        return std::find(members.begin(), members.end(), value) != members.end();
    }

    template<typename Func>
    cudaError_t each(Func fn, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr) const;
};

/**
 * @brief Owning group: owned pools are arranged so members are at the front.
 * @tparam Owned Owned component types.
 * @tparam Get Observed component types, possibly const-qualified.
 * @tparam Exclude Excluded component types.
 */
template<typename... Owned, typename... Get, typename... Exclude>
struct cuda_group<owned_t<Owned...>, get_t<Get...>, exclude_t<Exclude...>> {
    cuda_registry *registry{};
    std::tuple<cuda_storage<std::remove_const_t<Owned>> *...> owned_storages{};
    std::tuple<cuda_storage<std::remove_const_t<Get>> *...> get_storages{};
    std::vector<internal::cuda_pool_base *> excluded_pools{};
    std::size_t length{};

    cuda_group() = default;

    cuda_group(cuda_registry *owner, std::tuple<cuda_storage<std::remove_const_t<Owned>> *...> owned,
               std::tuple<cuda_storage<std::remove_const_t<Get>> *...> get, std::vector<internal::cuda_pool_base *> exclude, cudaStream_t stream)
        : registry{owner},
          owned_storages{owned},
          get_storages{get},
          excluded_pools{std::move(exclude)} {
        std::vector<internal::cuda_pool_base *> required;
        std::apply([&](auto *...component) { (required.push_back(static_cast<internal::cuda_pool_base *>(component)), ...); }, owned_storages);
        std::apply([&](auto *...component) { (required.push_back(static_cast<internal::cuda_pool_base *>(component)), ...); }, get_storages);
        const auto matching = internal::group_members(required, excluded_pools, registry->identity.data(), registry->identity.free_list());
        length = matching.size();

        // Arrange every owned pool so matching entities occupy [0, length) in order.
        std::apply([&](auto *...component) {
            (internal::arrange_storage(*component, matching, stream), ...);
        },
                   owned_storages);
    }

    /*! @brief Returns the number of entities in the group. */
    [[nodiscard]] std::size_t size() const noexcept {
        return length;
    }

    /*! @brief Checks whether the group is empty. */
    [[nodiscard]] bool empty() const noexcept {
        return length == 0u;
    }

    /*! @brief Checks if the group contains an entity. */
    [[nodiscard]] bool contains(const entity value) const noexcept {
        if constexpr(sizeof...(Owned) == 0u) {
            return false;
        } else {
            auto *leading = std::get<0>(owned_storages);
            return leading->contains(value) && leading->index(value) < length;
        }
    }

    template<typename Func>
    cudaError_t each(Func fn, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr) const;
};

template<typename... Owned, typename... Get, typename... Exclude>
inline cuda_group<owned_t<Owned...>, get_t<Get...>, exclude_t<Exclude...>> cuda_registry::group(get_t<Get...>, exclude_t<Exclude...>) {
    if constexpr(sizeof...(Owned) != 0u) {
        ENTT_ASSERT(!(owned_types.contains(type_hash<std::remove_const_t<Owned>>::value()) || ...), "Conflicting groups");
    }

    auto owned = std::make_tuple(static_cast<cuda_storage<std::remove_const_t<Owned>> *>(&pool<std::remove_const_t<Owned>>())...);
    auto get = std::make_tuple(static_cast<cuda_storage<std::remove_const_t<Get>> *>(&pool<std::remove_const_t<Get>>())...);
    std::vector<internal::cuda_pool_base *> exclude{static_cast<internal::cuda_pool_base *>(&pool<std::remove_const_t<Exclude>>())...};

    if constexpr(sizeof...(Owned) != 0u) {
        (owned_types.insert(type_hash<std::remove_const_t<Owned>>::value()), ...);
    }

    if constexpr(sizeof...(Owned) == 0u) {
        return cuda_group<owned_t<>, get_t<Get...>, exclude_t<Exclude...>>{this, get, std::move(exclude)};
    } else {
        return cuda_group<owned_t<Owned...>, get_t<Get...>, exclude_t<Exclude...>>{this, owned, get, std::move(exclude), nullptr};
    }
}

template<typename... Owned, typename... Get, typename... Exclude>
inline cuda_group<owned_t<>, get_t<const Owned..., const Get...>, exclude_t<const Exclude...>> cuda_registry::group_if_exists(get_t<Get...>, exclude_t<Exclude...>) const {
    auto get = std::make_tuple(static_cast<cuda_storage<std::remove_const_t<Owned>> *>(const_cast<cuda_storage<std::remove_const_t<Owned>> *>(find_pool<std::remove_const_t<Owned>>()))...,
                               static_cast<cuda_storage<std::remove_const_t<Get>> *>(const_cast<cuda_storage<std::remove_const_t<Get>> *>(find_pool<std::remove_const_t<Get>>()))...);
    std::vector<internal::cuda_pool_base *> exclude{const_cast<internal::cuda_pool_base *>(static_cast<const internal::cuda_pool_base *>(find_pool<std::remove_const_t<Exclude>>()))...};

    return cuda_group<owned_t<>, get_t<const Owned..., const Get...>, exclude_t<const Exclude...>>{const_cast<cuda_registry *>(this), get, std::move(exclude)};
}

template<typename... Type>
inline cudaError_t cuda_registry::compact(cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    // Swap-and-pop pools have no tombstones, so compact is a no-op.
    if(const auto err = record_done(done, stream); err != cudaSuccess) {
        return fail(err);
    }

    return cudaSuccess;
}

#ifdef __CUDACC__

template<typename... Get, typename... Exclude>
template<typename Func>
cudaError_t cuda_group<owned_t<>, get_t<Get...>, exclude_t<Exclude...>>::each(Func fn, cudaStream_t stream, cudaEvent_t done) const {
    return [&]<std::size_t... Index>(std::index_sequence<Index...>) {
        if(const auto err = registry->fault; err != cudaSuccess) {
            return err;
        }

        const entity *candidate{};

        if(const auto err = registry->upload_list(members.data(), members.size(), candidate, stream); err != cudaSuccess) {
            return registry->fail(err);
        }

        using types = std::tuple<Get...>;
        auto reqs = std::make_tuple(internal::cuda_dispatch_pool<std::tuple_element_t<Index, types>>{std::get<Index>(get_storages)->pages(), std::get<Index>(get_storages)->template payload_device<std::tuple_element_t<Index, types>>()}...);

        internal::cuda_exclude_params excluded{};
        excluded.count = std::min(excluded_pools.size(), internal::cuda_max_exclude);

        for(std::size_t pos = 0u; pos < excluded.count; ++pos) {
            excluded.pages[pos] = excluded_pools[pos]->pages();
        }

        if(members.size() != 0u) {
            const auto block = 256u;
            const auto grid = static_cast<unsigned>((members.size() + block - 1u) / block);
            internal::cuda_view_each_kernel<<<grid, block, 0, stream>>>(fn, members.size(), candidate, reqs, excluded);

            if(const auto err = cudaGetLastError(); err != cudaSuccess) {
                return registry->fail(err);
            }
        }

        if(done != nullptr) {
            if(const auto err = cudaEventRecord(done, stream); err != cudaSuccess) {
                return registry->fail(err);
            }
        }

        return cudaSuccess;
    }(std::index_sequence_for<Get...>{});
}

template<typename... Owned, typename... Get, typename... Exclude>
template<typename Func>
cudaError_t cuda_group<owned_t<Owned...>, get_t<Get...>, exclude_t<Exclude...>>::each(Func fn, cudaStream_t stream, cudaEvent_t done) const {
    return [&]<std::size_t... O, std::size_t... G>(std::index_sequence<O...>, std::index_sequence<G...>) {
        if(const auto err = registry->fault; err != cudaSuccess) {
            return err;
        }

        const auto *leading = std::get<0>(owned_storages);
        const entity *candidate = leading->packed_device();

        using owned_types = std::tuple<Owned...>;
        using get_types = std::tuple<Get...>;

        auto reqs = std::tuple_cat(
            std::make_tuple(internal::cuda_dispatch_pool<std::tuple_element_t<O, owned_types>>{std::get<O>(owned_storages)->pages(), std::get<O>(owned_storages)->template payload_device<std::tuple_element_t<O, owned_types>>()}...),
            std::make_tuple(internal::cuda_dispatch_pool<std::tuple_element_t<G, get_types>>{std::get<G>(get_storages)->pages(), std::get<G>(get_storages)->template payload_device<std::tuple_element_t<G, get_types>>()}...));

        internal::cuda_exclude_params excluded{};
        excluded.count = std::min(excluded_pools.size(), internal::cuda_max_exclude);

        for(std::size_t pos = 0u; pos < excluded.count; ++pos) {
            excluded.pages[pos] = excluded_pools[pos]->pages();
        }

        if(length != 0u) {
            const auto block = 256u;
            const auto grid = static_cast<unsigned>((length + block - 1u) / block);
            internal::cuda_view_each_kernel<<<grid, block, 0, stream>>>(fn, length, candidate, reqs, excluded);

            if(const auto err = cudaGetLastError(); err != cudaSuccess) {
                return registry->fail(err);
            }
        }

        if(done != nullptr) {
            if(const auto err = cudaEventRecord(done, stream); err != cudaSuccess) {
                return registry->fail(err);
            }
        }

        return cudaSuccess;
    }(std::index_sequence_for<Owned...>{}, std::index_sequence_for<Get...>{});
}

template<typename Type, typename... Func>
inline cudaError_t cuda_registry::patch_impl(const entity value, cudaStream_t stream, cudaEvent_t done, Func &&...func) {
    if(const auto err = fault; err != cudaSuccess) {
        return err;
    }

    if(!valid(value)) {
        return cudaErrorInvalidValue;
    }

    auto &component = pool<Type>();

    if(!component.contains(value)) {
        return cudaErrorInvalidValue;
    }

    if constexpr(!std::is_empty_v<Type>) {
        internal::cuda_patch_kernel<<<1, 1, 0, stream>>>(component.pages(), component.template payload_device<Type>(), value, func...);

        if(const auto err = cudaGetLastError(); err != cudaSuccess) {
            return fail(err);
        }
    }

    if(const auto err = record_done(done, stream); err != cudaSuccess) {
        return fail(err);
    }

    return cudaSuccess;
}

template<typename... Type>
template<typename Func>
cudaError_t cuda_view<Type...>::each(Func fn, cudaStream_t stream, cudaEvent_t done) const {
    return dispatch(fn, stream, done, std::index_sequence_for<Type...>{});
}

template<typename... Type>
template<typename Func, std::size_t... Index>
cudaError_t cuda_view<Type...>::dispatch(Func fn, cudaStream_t stream, cudaEvent_t done, std::index_sequence<Index...>) const {
    if(const auto err = registry->fault; err != cudaSuccess) {
        return err;
    }

    using types = std::tuple<Type...>;

    std::size_t count{};
    const entity *candidate{};

    if constexpr(sizeof...(Type) == 0u) {
        if(const auto err = registry->upload_identity(count, candidate, stream); err != cudaSuccess) {
            return registry->fail(err);
        }
    } else {
        std::array<internal::cuda_pool_base *, sizeof...(Type)> bases{static_cast<internal::cuda_pool_base *>(std::get<Index>(storages))...};
        std::size_t chosen = 0u;

        for(std::size_t pos = 1u; pos < sizeof...(Type); ++pos) {
            if(bases[pos]->size() < bases[chosen]->size()) {
                chosen = pos;
            }
        }

        count = bases[chosen]->size();
        candidate = bases[chosen]->packed_device();
    }

    auto reqs = std::make_tuple(internal::cuda_dispatch_pool<std::tuple_element_t<Index, types>>{std::get<Index>(storages)->pages(), std::get<Index>(storages)->template payload_device<std::tuple_element_t<Index, types>>()}...);

    internal::cuda_exclude_params excluded_params{};
    excluded_params.count = std::min(excluded_pools.size(), internal::cuda_max_exclude);

    for(std::size_t pos = 0u; pos < excluded_params.count; ++pos) {
        excluded_params.pages[pos] = excluded_pools[pos]->pages();
    }

    if(count != 0u) {
        const auto block = 256u;
        const auto grid = static_cast<unsigned>((count + block - 1u) / block);
        internal::cuda_view_each_kernel<<<grid, block, 0, stream>>>(fn, count, candidate, reqs, excluded_params);

        if(const auto err = cudaGetLastError(); err != cudaSuccess) {
            return registry->fail(err);
        }
    }

    if(done != nullptr) {
        if(const auto err = cudaEventRecord(done, stream); err != cudaSuccess) {
            return registry->fail(err);
        }
    }

    return cudaSuccess;
}

template<typename Func>
cudaError_t cuda_runtime_view::each(Func fn, cudaStream_t stream, cudaEvent_t done) const {
    if(registry == nullptr) {
        return cudaErrorInvalidValue;
    }

    if(const auto err = registry->fault; err != cudaSuccess) {
        return err;
    }

    std::size_t count{};
    const entity *candidate{};

    if(pools.empty()) {
        if(const auto err = registry->upload_identity(count, candidate, stream); err != cudaSuccess) {
            return registry->fail(err);
        }
    } else {
        count = pools.front()->size();
        candidate = pools.front()->packed_device();
    }

    internal::cuda_runtime_params params{};
    params.required_count = std::min(pools.size(), internal::cuda_max_exclude);

    for(std::size_t pos = 0u; pos < params.required_count; ++pos) {
        params.required[pos] = pools[pos]->pages();
    }

    params.excluded_count = std::min(filter.size(), internal::cuda_max_exclude);

    for(std::size_t pos = 0u; pos < params.excluded_count; ++pos) {
        params.excluded[pos] = filter[pos]->pages();
    }

    if(count != 0u) {
        const auto block = 256u;
        const auto grid = static_cast<unsigned>((count + block - 1u) / block);
        internal::cuda_runtime_each_kernel<<<grid, block, 0, stream>>>(fn, count, candidate, params);

        if(const auto err = cudaGetLastError(); err != cudaSuccess) {
            return registry->fail(err);
        }
    }

    if(done != nullptr) {
        if(const auto err = cudaEventRecord(done, stream); err != cudaSuccess) {
            return registry->fail(err);
        }
    }

    return cudaSuccess;
}

#endif // __CUDACC__

} // namespace entt

#endif
