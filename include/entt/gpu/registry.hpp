#ifndef ENTT_GPU_REGISTRY_HPP
#define ENTT_GPU_REGISTRY_HPP

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
#include <entt/core/type_info.hpp>
#include <entt/entity/entity.hpp>
#include <entt/entity/registry.hpp>
#include <entt/entity/storage.hpp>
#include <entt/signal/sigh.hpp>
#include "contract.hpp"
#include "device.hpp"
#include "storage.hpp"

namespace entt {

template<typename Type, typename Entity, typename Backend>
class gpu_storage;

template<typename Entity, typename Backend, typename... Type>
struct gpu_view;

template<typename Entity, typename Allocator, typename Backend>
struct basic_gpu_registry;

template<typename Entity, typename Backend>
class gpu_runtime_view;

template<typename Entity, typename Backend, typename Owned = owned_t<>, typename Get = get_t<>, typename Exclude = exclude_t<>>
struct gpu_group;

/**
 * @brief GPU query result. Snapshot descriptors and dispatch on each call.
 * @tparam Entity Underlying entity identifier.
 * @tparam Backend Device backend providing native handles and runtime calls.
 * @tparam Type Required component types, possibly const-qualified.
 */
template<typename Entity, typename Backend, typename... Type>
struct gpu_view {
    basic_gpu_registry<Entity, std::allocator<Entity>, Backend> *registry{};
    std::tuple<gpu_storage<std::remove_const_t<Type>, Entity, Backend> *...> storages{};
    std::vector<internal::gpu_pool_base<Entity, Backend> *> excluded_pools{};

    template<typename Func>
    typename Backend::error_type each(Func fn, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr) const;

    /*! @brief Checks whether an Entity matches every required and no excluded pool. */
    [[nodiscard]] bool contains(const Entity value) const {
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
    void rebind(basic_gpu_registry<Entity, std::allocator<Entity>, Backend> &other) noexcept {
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
    [[nodiscard]] const internal::gpu_pool_base<Entity, Backend> *smallest_pool() const {
        const internal::gpu_pool_base<Entity, Backend> *result = nullptr;
        std::apply([&](auto *...component) {
            const internal::gpu_pool_base<Entity, Backend> *pools[] = {static_cast<const internal::gpu_pool_base<Entity, Backend> *>(component)...};
            result = *std::min_element(std::begin(pools), std::end(pools), [](const auto *lhs, const auto *rhs) { return lhs->size() < rhs->size(); });
        },
                   storages);
        return result;
    }

    template<typename Func, std::size_t... Index>
    typename Backend::error_type dispatch(Func fn, typename Backend::stream_type stream, typename Backend::event_type done, std::index_sequence<Index...>) const;
};

/**
 * @brief EnTT-style registry with host-owned identity and device-owned values.
 *
 * Structural operations enqueue their device updates on the caller's stream and
 * return submission status. Host metadata then describes the latest submitted
 * structure. `create` is CPU-only; no separate publication step exists.
 */
template<typename Entity, typename Allocator, typename Backend>
struct basic_gpu_registry {
    template<typename, typename, typename...>
    friend struct gpu_view;

    template<typename, typename>
    friend class gpu_runtime_view;

    template<typename, typename, typename, typename, typename>
    friend struct gpu_group;

    template<typename>
    friend class gpu_snapshot;

    template<typename>
    friend class gpu_snapshot_loader;

    /*! @brief Reused upstream identity storage (generation + free list). */
    using identity_storage = basic_storage<Entity, Entity, Allocator>;
    /*! @brief Pool container keyed by type hash. */
    using pool_container = std::unordered_map<id_type, std::unique_ptr<internal::gpu_pool_base<Entity, Backend>>>;
    /*! @brief Signal type for logical submission notifications. */
    using signal_type = sigh<void(basic_gpu_registry &, Entity, typename Backend::stream_type)>;
    /*! @brief Per-pool construct/update/destroy signals. */
    struct gpu_pool_signals {
        signal_type construct;
        signal_type update;
        signal_type destroy;
    };

    /*! @brief Underlying Entity identifier. */
    using entity_type = Entity;
    /*! @brief Underlying version type. */
    using version_type = entt_traits<Entity>::version_type;
    /*! @brief Unsigned integer type. */
    using size_type = std::size_t;
    /*! @brief Allocator type, used by host metadata and the context. */
    using allocator_type = Allocator;
    /*! @brief Type-erased pool interface used by erase_if predicates. */
    using common_type = internal::gpu_pool_base<Entity, Backend>;
    /*! @brief General purpose host context. */
    using context = internal::registry_context<allocator_type>;
    /*! @brief Mutable pool enumeration. */
    using iterable = internal::gpu_pool_range<typename pool_container::iterator, internal::gpu_pool_base<Entity, Backend>>;
    /*! @brief Constant pool enumeration. */
    using const_iterable = internal::gpu_pool_range<typename pool_container::const_iterator, const internal::gpu_pool_base<Entity, Backend>>;

    /*! @brief Typed storage type, preserving constness. */
    template<typename Type>
    using storage_for_type = gpu_storage<std::remove_const_t<Type>, Entity, Backend>;

    identity_storage identity;
    pool_container pools;
    std::unordered_map<id_type, gpu_pool_signals> signals;
    std::unordered_set<id_type> owned_types;
    internal::gpu_retirement<Backend> retirement;
    context vars{allocator_type{}};

    Entity *d_identity{};
    std::size_t identity_cap{};
    Entity *d_list{};
    std::size_t list_cap{};

    mutable typename Backend::error_type fault{Backend::success};

    basic_gpu_registry()
        : basic_gpu_registry{allocator_type{}} {}

    explicit basic_gpu_registry(const allocator_type &alloc)
        : identity{alloc},
          vars{alloc} {}

    basic_gpu_registry(const size_type count, const allocator_type &alloc = allocator_type{})
        : basic_gpu_registry{alloc} {
        pools.reserve(count);
    }

    basic_gpu_registry(const basic_gpu_registry &) = delete;
    basic_gpu_registry(basic_gpu_registry &&other) noexcept;
    ~basic_gpu_registry();

    basic_gpu_registry &operator=(const basic_gpu_registry &) = delete;
    basic_gpu_registry &operator=(basic_gpu_registry &&other) noexcept;

    /*! @brief Returns the associated allocator. */
    [[nodiscard]] allocator_type get_allocator() const noexcept {
        return identity.get_allocator();
    }

    /*! @brief Exchanges the contents with those of another registry. */
    void swap(basic_gpu_registry &other) noexcept {
        identity.swap(other.identity);
        pools.swap(other.pools);
        signals.swap(other.signals);
        owned_types.swap(other.owned_types);
        std::swap(retirement, other.retirement);
        std::swap(vars, other.vars);
        std::swap(d_identity, other.d_identity);
        std::swap(identity_cap, other.identity_cap);
        std::swap(d_list, other.d_list);
        std::swap(list_cap, other.list_cap);
        std::swap(fault, other.fault);

        for(auto &&[key, pool]: pools) {
            pool->rebind_retirement(retirement);
        }

        for(auto &&[key, pool]: other.pools) {
            pool->rebind_retirement(other.retirement);
        }
    }

    /*! @brief Creates an identifier; CPU-only, no device submission. */
    [[nodiscard]] Entity create();

    /*! @brief Creates the suggested identifier if unused, else a fresh one. */
    [[nodiscard]] Entity create(Entity hint);

    /*! @brief Assigns each element in a range an identifier. */
    template<typename It>
    void create(It first, It last);

    /*! @brief Destroys an Entity, removing every component and recycling its ID. */
    typename Backend::error_type destroy(Entity value, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /*! @brief Destroys an Entity and sets the recycled version. */
    typename Backend::error_type destroy(Entity value, version_type version, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /*! @brief Destroys every Entity in a range. */
    template<typename It>
    typename Backend::error_type destroy(It first, It last, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /**
     * @brief Emplaces using the default stream without a completion event.
     * @note A leading native stream or nullptr is reserved for stream selection.
     */
    template<typename Type, typename... Args>
    requires (!internal::gpu_leading_handle<typename Backend::stream_type, Args...> && internal::gpu_component_constructible<Type, Args...>)
    typename Backend::error_type emplace(const Entity value, Args &&...args) {
        return emplace_impl<Type>(value, nullptr, nullptr, std::forward<Args>(args)...);
    }

    /**
     * @brief Emplaces using a native stream without a completion event.
     * @note A leading native event or nullptr after the stream selects done.
     */
    template<typename Type, typename... Args>
    requires (!internal::gpu_leading_handle<typename Backend::event_type, Args...> && internal::gpu_component_constructible<Type, Args...>)
    typename Backend::error_type emplace(const Entity value, const typename Backend::stream_type stream, Args &&...args) {
        return emplace_impl<Type>(value, stream, nullptr, std::forward<Args>(args)...);
    }

    /** @brief Emplaces using a native stream and an optional completion event. */
    template<typename Type, typename... Args>
    requires internal::gpu_component_constructible<Type, Args...>
    typename Backend::error_type emplace(const Entity value, const typename Backend::stream_type stream, const typename Backend::event_type done, Args &&...args) {
        return emplace_impl<Type>(value, stream, done, std::forward<Args>(args)...);
    }

    /*! @brief Assigns or replaces the given element for an Entity. */
    template<typename Type, typename... Args>
    requires (!internal::gpu_leading_handle<typename Backend::stream_type, Args...> && internal::gpu_component_constructible<Type, Args...>)
    typename Backend::error_type emplace_or_replace(const Entity value, Args &&...args) {
        return emplace_or_replace_impl<Type>(value, nullptr, nullptr, std::forward<Args>(args)...);
    }

    /*! @brief Assigns or replaces the given element on a native stream. */
    template<typename Type, typename... Args>
    requires (!internal::gpu_leading_handle<typename Backend::event_type, Args...> && internal::gpu_component_constructible<Type, Args...>)
    typename Backend::error_type emplace_or_replace(const Entity value, const typename Backend::stream_type stream, Args &&...args) {
        return emplace_or_replace_impl<Type>(value, stream, nullptr, std::forward<Args>(args)...);
    }

    /*! @brief Assigns or replaces the given element on a stream and event. */
    template<typename Type, typename... Args>
    requires internal::gpu_component_constructible<Type, Args...>
    typename Backend::error_type emplace_or_replace(const Entity value, const typename Backend::stream_type stream, const typename Backend::event_type done, Args &&...args) {
        return emplace_or_replace_impl<Type>(value, stream, done, std::forward<Args>(args)...);
    }

    /*! @brief Emplaces only if absent; the caller accesses via get/try_get. */
    template<typename Type, typename... Args>
    requires (!internal::gpu_leading_handle<typename Backend::stream_type, Args...> && internal::gpu_component_constructible<Type, Args...>)
    typename Backend::error_type get_or_emplace(const Entity value, Args &&...args) {
        return get_or_emplace_impl<Type>(value, nullptr, nullptr, std::forward<Args>(args)...);
    }

    /*! @brief Emplaces only if absent on a native stream. */
    template<typename Type, typename... Args>
    requires (!internal::gpu_leading_handle<typename Backend::event_type, Args...> && internal::gpu_component_constructible<Type, Args...>)
    typename Backend::error_type get_or_emplace(const Entity value, const typename Backend::stream_type stream, Args &&...args) {
        return get_or_emplace_impl<Type>(value, stream, nullptr, std::forward<Args>(args)...);
    }

    /*! @brief Emplaces only if absent on a stream and event. */
    template<typename Type, typename... Args>
    requires internal::gpu_component_constructible<Type, Args...>
    typename Backend::error_type get_or_emplace(const Entity value, const typename Backend::stream_type stream, const typename Backend::event_type done, Args &&...args) {
        return get_or_emplace_impl<Type>(value, stream, done, std::forward<Args>(args)...);
    }

    /*! @brief Replaces an existing element with a freshly constructed value. */
    template<typename Type, typename... Args>
    requires (!internal::gpu_leading_handle<typename Backend::stream_type, Args...> && internal::gpu_component_constructible<Type, Args...>)
    typename Backend::error_type replace(const Entity value, Args &&...args) {
        return replace_impl<Type>(value, nullptr, nullptr, std::forward<Args>(args)...);
    }

    /*! @brief Replaces an existing element on a native stream. */
    template<typename Type, typename... Args>
    requires (!internal::gpu_leading_handle<typename Backend::event_type, Args...> && internal::gpu_component_constructible<Type, Args...>)
    typename Backend::error_type replace(const Entity value, const typename Backend::stream_type stream, Args &&...args) {
        return replace_impl<Type>(value, stream, nullptr, std::forward<Args>(args)...);
    }

    /*! @brief Replaces an existing element on a stream and event. */
    template<typename Type, typename... Args>
    requires internal::gpu_component_constructible<Type, Args...>
    typename Backend::error_type replace(const Entity value, const typename Backend::stream_type stream, const typename Backend::event_type done, Args &&...args) {
        return replace_impl<Type>(value, stream, done, std::forward<Args>(args)...);
    }

    /*! @brief Applies ordered device-callable functions to one component. */
    template<typename Type, typename... Func>
    typename Backend::error_type patch(const Entity value, Func &&...func) {
        return patch_impl<Type>(value, nullptr, nullptr, std::forward<Func>(func)...);
    }

    /*! @brief Applies ordered device-callable functions on a native stream. */
    template<typename Type, typename... Func>
    typename Backend::error_type patch(const Entity value, const typename Backend::stream_type stream, Func &&...func) {
        return patch_impl<Type>(value, stream, nullptr, std::forward<Func>(func)...);
    }

    /*! @brief Applies ordered device-callable functions on a stream and event. */
    template<typename Type, typename... Func>
    typename Backend::error_type patch(const Entity value, const typename Backend::stream_type stream, const typename Backend::event_type done, Func &&...func) {
        return patch_impl<Type>(value, stream, done, std::forward<Func>(func)...);
    }

    /*! @brief Assigns each Entity in a range the given element. */
    template<typename Type, typename It>
    typename Backend::error_type insert(It first, It last, const Type &value, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /*! @brief Assigns each Entity in a range one of the given elements. */
    template<typename Type, typename EIt, typename CIt>
    requires std::is_same_v<typename std::iterator_traits<CIt>::value_type, Type>
    typename Backend::error_type insert(EIt first, EIt last, CIt from, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /*! @brief Removes the given elements from an Entity (tolerant). */
    template<typename Type, typename... Other>
    typename Backend::error_type remove(Entity value, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /*! @brief Removes the given elements from every Entity in a range (tolerant). */
    template<typename Type, typename... Other, typename It>
    typename Backend::error_type remove(It first, It last, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /*! @brief Erases the given elements from an Entity (strict). */
    template<typename Type, typename... Other>
    typename Backend::error_type erase(Entity value, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /*! @brief Erases the given elements from every Entity in a range (strict). */
    template<typename Type, typename... Other, typename It>
    typename Backend::error_type erase(It first, It last, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /*! @brief Erases elements satisfying a predicate, receiving pool id and metadata. */
    template<typename Func>
    typename Backend::error_type erase_if(Entity value, Func func, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /*! @brief Clears the given pools, or the whole registry for an empty pack. */
    template<typename... Type>
    typename Backend::error_type clear(typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    template<typename Type>
    requires (internal::gpu_component<Type> && !std::is_empty_v<Type>)
    typename Backend::error_type get(Entity value, Type *pinned_host_dst, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    template<typename Type>
    requires (internal::gpu_component<Type> && !std::is_empty_v<Type>)
    typename Backend::error_type get(Entity value, Type *pinned_host_dst, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr) const;

    template<typename Type>
    requires (internal::gpu_component<Type> && !std::is_empty_v<Type>)
    typename Backend::error_type assign(Entity value, const Type *pinned_host_src, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /*! @brief Returns non-owning device pointers to the given elements, or null. */
    template<typename... Type>
    [[nodiscard]] auto try_get(Entity value) noexcept;

    /*! @brief Returns non-owning const device pointers, or null. */
    template<typename... Type>
    [[nodiscard]] auto try_get(Entity value) const noexcept;

    /*! @brief Checks if an identifier refers to a valid Entity. */
    [[nodiscard]] bool valid(const Entity value) const noexcept {
        return static_cast<std::size_t>(identity.find(value).index()) < identity.free_list();
    }

    /*! @brief Returns the version of an identifier, or the tombstone version. */
    [[nodiscard]] version_type current(const Entity value) const noexcept {
        return identity.current(value);
    }

    /*! @brief Checks if an Entity has no elements assigned. */
    [[nodiscard]] bool orphan(const Entity value) const noexcept {
        for(const auto &[key, pool]: pools) {
            if(pool->contains(value)) {
                return false;
            }
        }

        return true;
    }

    /*! @brief Checks if an Entity is part of all the given pools. */
    template<typename... Type>
    [[nodiscard]] bool all_of(const Entity value) const noexcept {
        return (has<Type>(value) && ...);
    }

    /*! @brief Checks if an Entity is part of at least one given pool. */
    template<typename... Type>
    [[nodiscard]] bool any_of(const Entity value) const noexcept {
        return (has<Type>(value) || ...);
    }

    template<typename... Type, typename... Exclude>
    [[nodiscard]] gpu_view<Entity, Backend, Type...> view(exclude_t<Exclude...> = exclude_t{});

    /*! @brief Returns a runtime view bound to this registry. */
    [[nodiscard]] gpu_runtime_view<Entity, Backend> runtime_view() noexcept;

    /*! @brief Returns a group for the given owned/observed/excluded elements. */
    template<typename... Owned, typename... Get, typename... Exclude>
    [[nodiscard]] gpu_group<Entity, Backend, owned_t<Owned...>, get_t<Get...>, exclude_t<Exclude...>> group(get_t<Get...> = get_t{}, exclude_t<Exclude...> = exclude_t{});

    /*! @brief Returns a group for the given elements without creating pools. */
    template<typename... Owned, typename... Get, typename... Exclude>
    [[nodiscard]] gpu_group<Entity, Backend, owned_t<>, get_t<const Owned..., const Get...>, exclude_t<const Exclude...>> group_if_exists(get_t<Get...> = get_t{}, exclude_t<Exclude...> = exclude_t{}) const;

    /*! @brief Checks whether the given elements are owned by a group. */
    template<typename... Type>
    [[nodiscard]] bool owned() const noexcept {
        return (owned_types.contains(type_hash<std::remove_const_t<Type>>::value()) || ...);
    }

    /*! @brief Removes tombstones; a no-op for swap-and-pop pools. */
    template<typename... Type>
    typename Backend::error_type compact(typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /*! @brief Sorts a pool by an Entity or value comparator. */
    template<typename Type, typename Compare>
    typename Backend::error_type sort(Compare compare, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /*! @brief Aligns a pool's order to another pool's order. */
    template<typename To, typename From>
    typename Backend::error_type sort(typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr);

    /*! @brief Returns a sink notified after a component is constructed. */
    template<typename Type>
    [[nodiscard]] auto on_construct(const id_type id = type_hash<Type>::value()) {
        return sink{signals[id].construct};
    }

    /*! @brief Returns a sink notified after a component is updated. */
    template<typename Type>
    [[nodiscard]] auto on_update(const id_type id = type_hash<Type>::value()) {
        return sink{signals[id].update};
    }

    /*! @brief Returns a sink notified before a component is destroyed. */
    template<typename Type>
    [[nodiscard]] auto on_destroy(const id_type id = type_hash<Type>::value()) {
        return sink{signals[id].destroy};
    }

    template<typename Type>
    requires internal::gpu_component<Type>
    [[nodiscard]] gpu_storage<Type, Entity, Backend> &storage(const id_type id = type_hash<Type>::value()) {
        return pool<Type>(id);
    }

    template<typename Type>
    requires internal::gpu_component<Type>
    [[nodiscard]] const gpu_storage<Type, Entity, Backend> *storage(const id_type id = type_hash<Type>::value()) const {
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

    /*! @brief Enumerates component pools, excluding the Entity storage. */
    [[nodiscard]] iterable storage() noexcept {
        return iterable{pools.begin(), pools.end()};
    }

    /*! @brief Enumerates component pools, excluding the Entity storage. */
    [[nodiscard]] const_iterable storage() const noexcept {
        return const_iterable{pools.begin(), pools.end()};
    }

    /*! @brief Removes a component pool, retiring its device allocations. */
    typename Backend::error_type reset(const id_type id, typename Backend::stream_type stream = nullptr);

    /*! @brief Returns the context object. */
    [[nodiscard]] context &ctx() noexcept {
        return vars;
    }

    /*! @brief Returns the context object. */
    [[nodiscard]] const context &ctx() const noexcept {
        return vars;
    }

private:
    typename Backend::error_type fail(const typename Backend::error_type err) const noexcept {
        if(fault == Backend::success) {
            fault = err;
        }

        return err;
    }

    static typename Backend::error_type record_done(typename Backend::event_type done, typename Backend::stream_type stream) noexcept {
        return (done != nullptr) ? Backend::event_record(done, stream) : Backend::success;
    }

    void publish_construct(const id_type id, const Entity value, typename Backend::stream_type stream) {
        if(const auto it = signals.find(id); it != signals.end()) {
            it->second.construct.publish(*this, value, stream);
        }
    }

    void publish_update(const id_type id, const Entity value, typename Backend::stream_type stream) {
        if(const auto it = signals.find(id); it != signals.end()) {
            it->second.update.publish(*this, value, stream);
        }
    }

    void publish_destroy(const id_type id, const Entity value, typename Backend::stream_type stream) {
        if(const auto it = signals.find(id); it != signals.end()) {
            it->second.destroy.publish(*this, value, stream);
        }
    }

    template<typename Type>
    [[nodiscard]] gpu_storage<Type, Entity, Backend> &pool(const id_type id = type_hash<Type>::value());

    template<typename Type>
    [[nodiscard]] const gpu_storage<Type, Entity, Backend> *find_pool(const id_type id = type_hash<Type>::value()) const noexcept {
        const auto it = pools.find(id);
        return (it != pools.end() && it->second->type() == type_hash<Type>::value()) ? static_cast<const gpu_storage<Type, Entity, Backend> *>(it->second.get()) : nullptr;
    }

    template<typename Type>
    [[nodiscard]] bool has(const Entity value) const noexcept {
        const auto *component = find_pool<Type>();
        return component && component->contains(value);
    }

    typename Backend::error_type upload_identity(std::size_t &count, const Entity *&out, typename Backend::stream_type stream);

    typename Backend::error_type upload_list(const Entity *src, std::size_t count, const Entity *&out, typename Backend::stream_type stream);

    template<typename Type, typename... Args>
    requires internal::gpu_component_constructible<Type, Args...>
    typename Backend::error_type emplace_impl(Entity value, typename Backend::stream_type stream, typename Backend::event_type done, Args &&...args);

    template<typename Type, typename... Args>
    requires internal::gpu_component_constructible<Type, Args...>
    typename Backend::error_type emplace_or_replace_impl(Entity value, typename Backend::stream_type stream, typename Backend::event_type done, Args &&...args);

    template<typename Type, typename... Args>
    requires internal::gpu_component_constructible<Type, Args...>
    typename Backend::error_type get_or_emplace_impl(Entity value, typename Backend::stream_type stream, typename Backend::event_type done, Args &&...args);

    template<typename Type, typename... Args>
    requires internal::gpu_component_constructible<Type, Args...>
    typename Backend::error_type replace_impl(Entity value, typename Backend::stream_type stream, typename Backend::event_type done, Args &&...args);

    template<typename Type, typename... Func>
    typename Backend::error_type patch_impl(Entity value, typename Backend::stream_type stream, typename Backend::event_type done, Func &&...func);
};

template<typename Entity, typename Allocator, typename Backend>
inline basic_gpu_registry<Entity, Allocator, Backend>::basic_gpu_registry(basic_gpu_registry<Entity, Allocator, Backend> &&other) noexcept
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

template<typename Entity, typename Allocator, typename Backend>
inline basic_gpu_registry<Entity, Allocator, Backend> &basic_gpu_registry<Entity, Allocator, Backend>::operator=(basic_gpu_registry<Entity, Allocator, Backend> &&other) noexcept {
    if(this != &other) {
        retirement.drain();
        Backend::device_synchronize();

        for(auto &&[key, pool]: pools) {
            pool->release();
        }

        if(d_identity != nullptr) {
            Backend::free(d_identity);
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

template<typename Entity, typename Allocator, typename Backend>
inline basic_gpu_registry<Entity, Allocator, Backend>::~basic_gpu_registry<Entity, Allocator, Backend>() {
    retirement.drain();
    // Conservative: wait for all library-owned work before releasing buffers.
    Backend::device_synchronize();

    for(auto &&[key, pool]: pools) {
        pool->release();
    }

    if(d_identity != nullptr) {
        Backend::free(d_identity);
        d_identity = nullptr;
    }

    if(d_list != nullptr) {
        Backend::free(d_list);
        d_list = nullptr;
    }
}

template<typename Entity, typename Allocator, typename Backend>
[[nodiscard]] inline Entity basic_gpu_registry<Entity, Allocator, Backend>::create() {
    if(fault != Backend::success) {
        throw std::logic_error("entt::basic_gpu_registry<Entity, Allocator, Backend> is in a fault state");
    }

    constexpr auto max_index = static_cast<std::size_t>(entt_traits<Entity>::to_entity(entt::null)) + 1u;

    if(identity.free_list() == identity.size() && identity.size() >= max_index) {
        throw std::length_error("No more entities available");
    }

    return identity.generate();
}

template<typename Entity, typename Allocator, typename Backend>
[[nodiscard]] inline Entity basic_gpu_registry<Entity, Allocator, Backend>::create(const Entity hint) {
    if(fault != Backend::success) {
        throw std::logic_error("entt::basic_gpu_registry<Entity, Allocator, Backend> is in a fault state");
    }

    constexpr auto max_index = static_cast<std::size_t>(entt_traits<Entity>::to_entity(entt::null)) + 1u;

    if(identity.free_list() == identity.size() && identity.size() >= max_index) {
        throw std::length_error("No more entities available");
    }

    return identity.generate(hint);
}

template<typename Entity, typename Allocator, typename Backend>
template<typename It>
inline void basic_gpu_registry<Entity, Allocator, Backend>::create(It first, It last) {
    if(fault != Backend::success) {
        throw std::logic_error("entt::basic_gpu_registry<Entity, Allocator, Backend> is in a fault state");
    }

    identity.generate(std::move(first), std::move(last));
}

template<typename Entity, typename Allocator, typename Backend>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::destroy(const Entity value, typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(!valid(value)) {
        return Backend::invalid_value;
    }

    typename Backend::error_type err = Backend::success;

    for(auto &&[key, pool]: pools) {
        if(pool->contains(value)) {
            publish_destroy(key, value, stream);
        }

        if(const auto curr = pool->enqueue_remove(value, stream); curr != Backend::success && err == Backend::success) {
            err = curr;
        }
    }

    if(err != Backend::success) {
        return fail(err);
    }

    identity.erase(value);

    if(const auto curr = record_done(done, stream); curr != Backend::success) {
        return fail(curr);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::destroy(const Entity value, const version_type version, typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = destroy(value, stream); err != Backend::success) {
        return err;
    }

    const auto elem = entt_traits<Entity>::construct(entt_traits<Entity>::to_entity(value), version);
    identity.bump((elem == entt::tombstone) ? entt_traits<Entity>::next(elem) : elem);

    if(const auto curr = record_done(done, stream); curr != Backend::success) {
        return fail(curr);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename It>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::destroy(It first, It last, typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    std::vector<Entity> range(first, last);

    for(const auto value: range) {
        if(!valid(value)) {
            return Backend::invalid_value;
        }
    }

    typename Backend::error_type err = Backend::success;

    for(const auto value: range) {
        for(auto &&[key, pool]: pools) {
            if(pool->contains(value)) {
                publish_destroy(key, value, stream);
            }

            if(const auto curr = pool->enqueue_remove(value, stream); curr != Backend::success && err == Backend::success) {
                err = curr;
            }
        }

        identity.erase(value);
    }

    if(err != Backend::success) {
        return fail(err);
    }

    if(const auto curr = record_done(done, stream); curr != Backend::success) {
        return fail(curr);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type>
[[nodiscard]] inline gpu_storage<Type, Entity, Backend> &basic_gpu_registry<Entity, Allocator, Backend>::pool(const id_type id) {
    static_assert(!std::is_const_v<Type>, "Pool types are unqualified and non-const");

    if(const auto it = pools.find(id); it != pools.end()) {
        ENTT_ASSERT(it->second->type() == type_hash<Type>::value(), "Unexpected type");
        return static_cast<gpu_storage<Type, Entity, Backend> &>(*it->second);
    }

    auto ptr = std::make_unique<gpu_storage<Type, Entity, Backend>>(retirement);
    auto *raw = ptr.get();
    pools.emplace(id, std::move(ptr));
    return *raw;
}

template<typename Entity, typename Allocator, typename Backend>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::reset(const id_type id, typename Backend::stream_type stream) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(const auto it = pools.find(id); it != pools.end()) {
        it->second->reset(stream);
        pools.erase(it);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::upload_identity(std::size_t &count, const Entity *&out, typename Backend::stream_type stream) {
    count = identity.free_list();
    out = nullptr;

    if(count == 0u) {
        return Backend::success;
    }

    if(count > identity_cap) {
        Entity *next{};

        if(Backend::malloc(reinterpret_cast<void **>(&next), count * sizeof(Entity)) != Backend::success) {
            return Backend::out_of_memory;
        }

        retirement.retire(d_identity, stream);
        d_identity = next;
        identity_cap = count;
    }

    out = d_identity;
    return Backend::memcpy_async(d_identity, identity.data(), count * sizeof(Entity), Backend::host_to_device, stream);
}

template<typename Entity, typename Allocator, typename Backend>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::upload_list(const Entity *src, const std::size_t count, const Entity *&out, typename Backend::stream_type stream) {
    out = nullptr;

    if(count == 0u) {
        return Backend::success;
    }

    if(count > list_cap) {
        Entity *next{};

        if(Backend::malloc(reinterpret_cast<void **>(&next), count * sizeof(Entity)) != Backend::success) {
            return Backend::out_of_memory;
        }

        retirement.retire(d_list, stream);
        d_list = next;
        list_cap = count;
    }

    out = d_list;
    return Backend::memcpy_async(d_list, src, count * sizeof(Entity), Backend::host_to_device, stream);
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type, typename... Args>
requires internal::gpu_component_constructible<Type, Args...>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::emplace_impl(const Entity value, typename Backend::stream_type stream, typename Backend::event_type done, Args &&...args) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(!valid(value)) {
        return Backend::invalid_value;
    }

    auto &component = pool<Type>();

    if(component.contains(value)) {
        return Backend::invalid_value;
    }

    if(const auto err = component.emplace(value, stream, std::forward<Args>(args)...); err != Backend::success) {
        return fail(err);
    }

    publish_construct(type_hash<Type>::value(), value, stream);

    if(const auto err = record_done(done, stream); err != Backend::success) {
        return fail(err);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type, typename... Args>
requires internal::gpu_component_constructible<Type, Args...>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::emplace_or_replace_impl(const Entity value, typename Backend::stream_type stream, typename Backend::event_type done, Args &&...args) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(!valid(value)) {
        return Backend::invalid_value;
    }

    auto &component = pool<Type>();
    const bool present = component.contains(value);

    if(const auto err = (present ? component.replace(value, stream, std::forward<Args>(args)...) : component.emplace(value, stream, std::forward<Args>(args)...)); err != Backend::success) {
        return fail(err);
    }

    if(present) {
        publish_update(type_hash<Type>::value(), value, stream);
    } else {
        publish_construct(type_hash<Type>::value(), value, stream);
    }

    if(const auto err = record_done(done, stream); err != Backend::success) {
        return fail(err);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type, typename... Args>
requires internal::gpu_component_constructible<Type, Args...>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::get_or_emplace_impl(const Entity value, typename Backend::stream_type stream, typename Backend::event_type done, Args &&...args) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(!valid(value)) {
        return Backend::invalid_value;
    }

    auto &component = pool<Type>();

    if(!component.contains(value)) {
        if(const auto err = component.emplace(value, stream, std::forward<Args>(args)...); err != Backend::success) {
            return fail(err);
        }

        publish_construct(type_hash<Type>::value(), value, stream);
    }

    if(const auto err = record_done(done, stream); err != Backend::success) {
        return fail(err);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type, typename... Args>
requires internal::gpu_component_constructible<Type, Args...>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::replace_impl(const Entity value, typename Backend::stream_type stream, typename Backend::event_type done, Args &&...args) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(!valid(value)) {
        return Backend::invalid_value;
    }

    auto &component = pool<Type>();

    if(!component.contains(value)) {
        return Backend::invalid_value;
    }

    if(const auto err = component.replace(value, stream, std::forward<Args>(args)...); err != Backend::success) {
        return fail(err);
    }

    publish_update(type_hash<Type>::value(), value, stream);

    if(const auto err = record_done(done, stream); err != Backend::success) {
        return fail(err);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type, typename It>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::insert(It first, It last, const Type &value, typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    std::vector<Entity> range(first, last);

    for(const auto Entity: range) {
        if(!valid(Entity)) {
            return Backend::invalid_value;
        }
    }

    auto &component = pool<Type>();

    for(const auto Entity: range) {
        if(component.contains(Entity)) {
            return Backend::invalid_value;
        }
    }

    for(const auto Entity: range) {
        if(const auto err = component.emplace(Entity, stream, value); err != Backend::success) {
            return fail(err);
        }

        publish_construct(type_hash<Type>::value(), Entity, stream);
    }

    if(const auto err = record_done(done, stream); err != Backend::success) {
        return fail(err);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type, typename EIt, typename CIt>
requires std::is_same_v<typename std::iterator_traits<CIt>::value_type, Type>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::insert(EIt first, EIt last, CIt from, typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    std::vector<Entity> range(first, last);

    for(const auto Entity: range) {
        if(!valid(Entity)) {
            return Backend::invalid_value;
        }
    }

    auto &component = pool<Type>();

    for(const auto Entity: range) {
        if(component.contains(Entity)) {
            return Backend::invalid_value;
        }
    }

    for(const auto Entity: range) {
        if(const auto err = component.emplace(Entity, stream, *from); err != Backend::success) {
            return fail(err);
        }

        publish_construct(type_hash<Type>::value(), Entity, stream);
        ++from;
    }

    if(const auto err = record_done(done, stream); err != Backend::success) {
        return fail(err);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type, typename... Other>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::remove(const Entity value, typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(!valid(value)) {
        return Backend::invalid_value;
    }

    typename Backend::error_type err = Backend::success;
    const auto remove_one = [&](auto &component, const id_type id) {
        if(component.contains(value)) {
            publish_destroy(id, value, stream);
        }

        if(const auto curr = component.enqueue_remove(value, stream); curr != Backend::success && err == Backend::success) {
            err = curr;
        }
    };

    remove_one(pool<std::remove_const_t<Type>>(), type_hash<std::remove_const_t<Type>>::value());
    ((remove_one(pool<std::remove_const_t<Other>>(), type_hash<std::remove_const_t<Other>>::value())), ...);

    if(err != Backend::success) {
        return fail(err);
    }

    if(const auto curr = record_done(done, stream); curr != Backend::success) {
        return fail(curr);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type, typename... Other, typename It>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::remove(It first, It last, typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    std::vector<Entity> range(first, last);

    for(const auto Entity: range) {
        if(!valid(Entity)) {
            return Backend::invalid_value;
        }
    }

    typename Backend::error_type err = Backend::success;
    const auto remove_one = [&](auto &component, const Entity Entity, const id_type id) {
        if(component.contains(Entity)) {
            publish_destroy(id, Entity, stream);
        }

        if(const auto curr = component.enqueue_remove(Entity, stream); curr != Backend::success && err == Backend::success) {
            err = curr;
        }
    };

    for(const auto Entity: range) {
        remove_one(pool<std::remove_const_t<Type>>(), Entity, type_hash<std::remove_const_t<Type>>::value());
        ((remove_one(pool<std::remove_const_t<Other>>(), Entity, type_hash<std::remove_const_t<Other>>::value())), ...);
    }

    if(err != Backend::success) {
        return fail(err);
    }

    if(const auto curr = record_done(done, stream); curr != Backend::success) {
        return fail(curr);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type, typename... Other>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::erase(const Entity value, typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(!valid(value)) {
        return Backend::invalid_value;
    }

    if(!(has<Type>(value) && (has<Other>(value) && ...))) {
        return Backend::invalid_value;
    }

    return remove<Type, Other...>(value, stream, done);
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type, typename... Other, typename It>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::erase(It first, It last, typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    std::vector<Entity> range(first, last);

    for(const auto Entity: range) {
        if(!valid(Entity) || !(has<Type>(Entity) && (has<Other>(Entity) && ...))) {
            return Backend::invalid_value;
        }
    }

    return remove<Type, Other...>(range.begin(), range.end(), stream, done);
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Func>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::erase_if(const Entity value, Func func, typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(!valid(value)) {
        return Backend::invalid_value;
    }

    typename Backend::error_type err = Backend::success;

    for(auto &&[id, pool]: pools) {
        if(pool->contains(value) && func(id, std::as_const(*pool))) {
            if(const auto curr = pool->enqueue_remove(value, stream); curr != Backend::success && err == Backend::success) {
                err = curr;
            }
        }
    }

    if(err != Backend::success) {
        return fail(err);
    }

    if(const auto curr = record_done(done, stream); curr != Backend::success) {
        return fail(curr);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename... Type>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::clear(typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if constexpr(sizeof...(Type) == 0u) {
        for(auto &&[key, pool]: pools) {
            pool->clear();
        }

        std::vector<Entity> alive(identity.data(), identity.data() + identity.free_list());

        for(const auto Entity: alive) {
            identity.erase(Entity);
        }
    } else {
        (pool<Type>().clear(), ...);
    }

    if(const auto err = record_done(done, stream); err != Backend::success) {
        return fail(err);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type>
requires (internal::gpu_component<Type> && !std::is_empty_v<Type>)
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::get(const Entity value, Type *pinned_host_dst, typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(!valid(value) || pinned_host_dst == nullptr) {
        return Backend::invalid_value;
    }

    auto &component = pool<Type>();

    if(!component.contains(value)) {
        return Backend::invalid_value;
    }

    if(const auto err = component.enqueue_get(value, pinned_host_dst, stream); err != Backend::success) {
        return fail(err);
    }

    if(const auto err = record_done(done, stream); err != Backend::success) {
        return fail(err);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type>
requires (internal::gpu_component<Type> && !std::is_empty_v<Type>)
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::get(const Entity value, Type *pinned_host_dst, typename Backend::stream_type stream, typename Backend::event_type done) const {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(!valid(value) || pinned_host_dst == nullptr) {
        return Backend::invalid_value;
    }

    const auto *component = find_pool<Type>();

    if(component == nullptr || !component->contains(value)) {
        return Backend::invalid_value;
    }

    if(const auto err = component->enqueue_get(value, pinned_host_dst, stream); err != Backend::success) {
        return fail(err);
    }

    if(const auto err = record_done(done, stream); err != Backend::success) {
        return fail(err);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type>
requires (internal::gpu_component<Type> && !std::is_empty_v<Type>)
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::assign(const Entity value, const Type *pinned_host_src, typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(!valid(value) || pinned_host_src == nullptr) {
        return Backend::invalid_value;
    }

    auto &component = pool<Type>();

    if(!component.contains(value)) {
        return Backend::invalid_value;
    }

    if(const auto err = component.enqueue_assign(value, pinned_host_src, stream); err != Backend::success) {
        return fail(err);
    }

    publish_update(type_hash<Type>::value(), value, stream);

    if(const auto err = record_done(done, stream); err != Backend::success) {
        return fail(err);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename... Type>
[[nodiscard]] inline auto basic_gpu_registry<Entity, Allocator, Backend>::try_get(const Entity value) noexcept {
    if constexpr(sizeof...(Type) == 1u) {
        using element = std::tuple_element_t<0, std::tuple<Type...>>;
        auto &component = pool<std::remove_const_t<element>>();
        return component.template try_get<element>(value);
    } else {
        return std::make_tuple(try_get<Type>(value)...);
    }
}

template<typename Entity, typename Allocator, typename Backend>
template<typename... Type>
[[nodiscard]] inline auto basic_gpu_registry<Entity, Allocator, Backend>::try_get(const Entity value) const noexcept {
    if constexpr(sizeof...(Type) == 1u) {
        using element = std::tuple_element_t<0, std::tuple<Type...>>;
        const auto *component = find_pool<std::remove_const_t<element>>();
        return component ? component->template try_get<const element>(value) : nullptr;
    } else {
        return std::make_tuple(try_get<Type>(value)...);
    }
}

template<typename Entity, typename Allocator, typename Backend>
template<typename... Type, typename... Exclude>
[[nodiscard]] inline gpu_view<Entity, Backend, Type...> basic_gpu_registry<Entity, Allocator, Backend>::view(exclude_t<Exclude...>) {
    gpu_view<Entity, Backend, Type...> result;
    result.registry = this;
    result.storages = std::make_tuple(static_cast<gpu_storage<std::remove_const_t<Type>, Entity, Backend> *>(&pool<std::remove_const_t<Type>>())...);
    result.excluded_pools = {static_cast<internal::gpu_pool_base<Entity, Backend> *>(&pool<std::remove_const_t<Exclude>>())...};
    return result;
}

/**
 * @brief Generic runtime view dispatching Entity identifiers only.
 *
 * Required and excluded pools are added at runtime. The device callback receives
 * only the Entity identifier; there is no runtime device compilation.
 * @tparam Entity Underlying entity identifier.
 */
template<typename Entity, typename Backend>
class gpu_runtime_view {
    basic_gpu_registry<Entity, std::allocator<Entity>, Backend> *registry{};
    std::vector<internal::gpu_pool_base<Entity, Backend> *> pools;
    std::vector<internal::gpu_pool_base<Entity, Backend> *> filter;

public:
    gpu_runtime_view() = default;
    explicit gpu_runtime_view(basic_gpu_registry<Entity, std::allocator<Entity>, Backend> *owner) noexcept
        : registry{owner} {}

    /*! @brief Adds the default-named pool of the given type, keeping the smallest first. */
    template<typename Type>
    gpu_runtime_view &iterate() {
        auto *component = static_cast<internal::gpu_pool_base<Entity, Backend> *>(&registry->storage<Type>());

        if(pools.empty() || component->size() >= pools.front()->size()) {
            pools.push_back(component);
        } else {
            pools.push_back(std::exchange(pools.front(), component));
        }

        return *this;
    }

    /*! @brief Adds an opaque pool, keeping the smallest first. */
    gpu_runtime_view &iterate(internal::gpu_pool_base<Entity, Backend> &base) {
        if(pools.empty() || base.size() >= pools.front()->size()) {
            pools.push_back(&base);
        } else {
            pools.push_back(std::exchange(pools.front(), &base));
        }

        return *this;
    }

    /*! @brief Adds the default-named pool of the given type as a filter. */
    template<typename Type>
    gpu_runtime_view &exclude() {
        filter.push_back(&registry->storage<Type>());
        return *this;
    }

    /*! @brief Adds an opaque pool as a filter. */
    gpu_runtime_view &exclude(internal::gpu_pool_base<Entity, Backend> &base) {
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

    /*! @brief Checks if the view contains an Entity. */
    [[nodiscard]] bool contains(const Entity value) const noexcept {
        return std::all_of(pools.begin(), pools.end(), [&](const auto *pool) { return pool->contains(value); })
               && std::none_of(filter.begin(), filter.end(), [&](const auto *pool) { return pool && pool->contains(value); });
    }

    /*! @brief Checks whether the view is initialized. */
    [[nodiscard]] explicit operator bool() const noexcept {
        return !(pools.empty() && filter.empty());
    }

    template<typename Func>
    typename Backend::error_type each(Func fn, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr) const;
};

template<typename Entity, typename Allocator, typename Backend>
[[nodiscard]] inline gpu_runtime_view<Entity, Backend> basic_gpu_registry<Entity, Allocator, Backend>::runtime_view() noexcept {
    return gpu_runtime_view{this};
}

namespace internal {

/*! @brief Computes the entities matching a set of required and excluded pools. */
template<typename Entity, typename Backend>
inline std::vector<Entity> group_members(const std::vector<gpu_pool_base<Entity, Backend> *> &required, const std::vector<gpu_pool_base<Entity, Backend> *> &excluded,
                                         const Entity *alive, const std::size_t alive_count) {
    std::vector<Entity> members;

    if(required.empty()) {
        for(std::size_t pos = 0u; pos < alive_count; ++pos) {
            const Entity value = alive[pos];
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
        const Entity value = candidate->data()[pos];
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
template<typename Entity, typename Storage, typename Stream>
void arrange_storage(Storage &storage, const std::vector<Entity> &matching, Stream stream) {
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
template<typename Entity, typename Backend, typename... Get, typename... Exclude>
struct gpu_group<Entity, Backend, owned_t<>, get_t<Get...>, exclude_t<Exclude...>> {
    basic_gpu_registry<Entity, std::allocator<Entity>, Backend> *registry{};
    std::tuple<gpu_storage<std::remove_const_t<Get>, Entity, Backend> *...> get_storages{};
    std::vector<internal::gpu_pool_base<Entity, Backend> *> excluded_pools{};
    std::vector<Entity> members{};

    gpu_group() = default;

    gpu_group(basic_gpu_registry<Entity, std::allocator<Entity>, Backend> *owner, std::tuple<gpu_storage<std::remove_const_t<Get>, Entity, Backend> *...> get, std::vector<internal::gpu_pool_base<Entity, Backend> *> exclude)
        : registry{owner},
          get_storages{get},
          excluded_pools{std::move(exclude)} {
        std::vector<internal::gpu_pool_base<Entity, Backend> *> required;
        std::apply([&](auto *...component) { (required.push_back(static_cast<internal::gpu_pool_base<Entity, Backend> *>(component)), ...); }, get_storages);
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

    /*! @brief Checks if the group contains an Entity. */
    [[nodiscard]] bool contains(const Entity value) const noexcept {
        return std::find(members.begin(), members.end(), value) != members.end();
    }

    template<typename Func>
    typename Backend::error_type each(Func fn, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr) const;
};

/**
 * @brief Owning group: owned pools are arranged so members are at the front.
 * @tparam Owned Owned component types.
 * @tparam Get Observed component types, possibly const-qualified.
 * @tparam Exclude Excluded component types.
 */
template<typename Entity, typename Backend, typename... Owned, typename... Get, typename... Exclude>
struct gpu_group<Entity, Backend, owned_t<Owned...>, get_t<Get...>, exclude_t<Exclude...>> {
    basic_gpu_registry<Entity, std::allocator<Entity>, Backend> *registry{};
    std::tuple<gpu_storage<std::remove_const_t<Owned>, Entity, Backend> *...> owned_storages{};
    std::tuple<gpu_storage<std::remove_const_t<Get>, Entity, Backend> *...> get_storages{};
    std::vector<internal::gpu_pool_base<Entity, Backend> *> excluded_pools{};
    std::size_t length{};

    gpu_group() = default;

    gpu_group(basic_gpu_registry<Entity, std::allocator<Entity>, Backend> *owner, std::tuple<gpu_storage<std::remove_const_t<Owned>, Entity, Backend> *...> owned,
              std::tuple<gpu_storage<std::remove_const_t<Get>, Entity, Backend> *...> get, std::vector<internal::gpu_pool_base<Entity, Backend> *> exclude, typename Backend::stream_type stream)
        : registry{owner},
          owned_storages{owned},
          get_storages{get},
          excluded_pools{std::move(exclude)} {
        std::vector<internal::gpu_pool_base<Entity, Backend> *> required;
        std::apply([&](auto *...component) { (required.push_back(static_cast<internal::gpu_pool_base<Entity, Backend> *>(component)), ...); }, owned_storages);
        std::apply([&](auto *...component) { (required.push_back(static_cast<internal::gpu_pool_base<Entity, Backend> *>(component)), ...); }, get_storages);
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

    /*! @brief Checks if the group contains an Entity. */
    [[nodiscard]] bool contains(const Entity value) const noexcept {
        if constexpr(sizeof...(Owned) == 0u) {
            return false;
        } else {
            auto *leading = std::get<0>(owned_storages);
            return leading->contains(value) && leading->index(value) < length;
        }
    }

    template<typename Func>
    typename Backend::error_type each(Func fn, typename Backend::stream_type stream = nullptr, typename Backend::event_type done = nullptr) const;
};

template<typename Entity, typename Allocator, typename Backend>
template<typename... Owned, typename... Get, typename... Exclude>
inline gpu_group<Entity, Backend, owned_t<Owned...>, get_t<Get...>, exclude_t<Exclude...>> basic_gpu_registry<Entity, Allocator, Backend>::group(get_t<Get...>, exclude_t<Exclude...>) {
    if constexpr(sizeof...(Owned) != 0u) {
        ENTT_ASSERT(!(owned_types.contains(type_hash<std::remove_const_t<Owned>>::value()) || ...), "Conflicting groups");
    }

    auto owned = std::make_tuple(static_cast<gpu_storage<std::remove_const_t<Owned>, Entity, Backend> *>(&pool<std::remove_const_t<Owned>>())...);
    auto get = std::make_tuple(static_cast<gpu_storage<std::remove_const_t<Get>, Entity, Backend> *>(&pool<std::remove_const_t<Get>>())...);
    std::vector<internal::gpu_pool_base<Entity, Backend> *> exclude{static_cast<internal::gpu_pool_base<Entity, Backend> *>(&pool<std::remove_const_t<Exclude>>())...};

    if constexpr(sizeof...(Owned) != 0u) {
        (owned_types.insert(type_hash<std::remove_const_t<Owned>>::value()), ...);
    }

    if constexpr(sizeof...(Owned) == 0u) {
        return gpu_group<Entity, Backend, owned_t<>, get_t<Get...>, exclude_t<Exclude...>>{this, get, std::move(exclude)};
    } else {
        return gpu_group<Entity, Backend, owned_t<Owned...>, get_t<Get...>, exclude_t<Exclude...>>{this, owned, get, std::move(exclude), nullptr};
    }
}

template<typename Entity, typename Allocator, typename Backend>
template<typename... Owned, typename... Get, typename... Exclude>
inline gpu_group<Entity, Backend, owned_t<>, get_t<const Owned..., const Get...>, exclude_t<const Exclude...>> basic_gpu_registry<Entity, Allocator, Backend>::group_if_exists(get_t<Get...>, exclude_t<Exclude...>) const {
    auto get = std::make_tuple(static_cast<gpu_storage<std::remove_const_t<Owned>, Entity, Backend> *>(const_cast<gpu_storage<std::remove_const_t<Owned>, Entity, Backend> *>(find_pool<std::remove_const_t<Owned>>()))...,
                               static_cast<gpu_storage<std::remove_const_t<Get>, Entity, Backend> *>(const_cast<gpu_storage<std::remove_const_t<Get>, Entity, Backend> *>(find_pool<std::remove_const_t<Get>>()))...);
    std::vector<internal::gpu_pool_base<Entity, Backend> *> exclude{const_cast<internal::gpu_pool_base<Entity, Backend> *>(static_cast<const internal::gpu_pool_base<Entity, Backend> *>(find_pool<std::remove_const_t<Exclude>>()))...};

    return gpu_group<Entity, Backend, owned_t<>, get_t<const Owned..., const Get...>, exclude_t<const Exclude...>>{const_cast<basic_gpu_registry<Entity, std::allocator<Entity>, Backend> *>(this), get, std::move(exclude)};
}

template<typename Entity, typename Allocator, typename Backend>
template<typename... Type>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::compact(typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    // Swap-and-pop pools have no tombstones, so compact is a no-op.
    if(const auto err = record_done(done, stream); err != Backend::success) {
        return fail(err);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type, typename Compare>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::sort(Compare compare, typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(owned<Type>()) {
        return Backend::invalid_value;
    }

    auto &component = pool<Type>();
    typename Backend::error_type err;

    if constexpr(!std::is_empty_v<Type> && std::is_invocable_v<Compare, const Type &, const Type &>) {
        err = component.sort_values(std::move(compare), stream);
    } else {
        err = component.sort_entities(std::move(compare), stream);
    }

    if(err != Backend::success) {
        return fail(err);
    }

    if(const auto curr = record_done(done, stream); curr != Backend::success) {
        return fail(curr);
    }

    return Backend::success;
}

template<typename Entity, typename Allocator, typename Backend>
template<typename To, typename From>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::sort(typename Backend::stream_type stream, typename Backend::event_type done) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(owned<To>()) {
        return Backend::invalid_value;
    }

    auto &to_pool = pool<To>();
    const auto *from_pool = find_pool<From>();

    std::vector<std::size_t> perm(to_pool.size());
    std::size_t next = 0u;

    if(from_pool != nullptr) {
        for(std::size_t pos = 0u; pos < from_pool->size(); ++pos) {
            const Entity value = from_pool->data()[pos];

            if(to_pool.contains(value)) {
                perm[next++] = to_pool.index(value);
            }
        }
    }

    for(std::size_t pos = 0u; pos < to_pool.size(); ++pos) {
        const Entity value = to_pool.data()[pos];

        if(from_pool == nullptr || !from_pool->contains(value)) {
            perm[next++] = pos;
        }
    }

    if(const auto err = to_pool.reorder(perm, stream); err != Backend::success) {
        return fail(err);
    }

    if(const auto curr = record_done(done, stream); curr != Backend::success) {
        return fail(curr);
    }

    return Backend::success;
}

/**
 * @brief Utility to serialize a registry (or selected pools) to an archive.
 *
 * Component values are downloaded synchronously before serialization; identity
 * and values are serialized, never process-local pointers.
 */
template<typename Registry>
class gpu_snapshot {
    const Registry *registry;

public:
    explicit gpu_snapshot(const Registry &owner) noexcept
        : registry{&owner} {}

    template<typename Type, typename Archive>
    const gpu_snapshot &get(Archive &archive, const id_type id = type_hash<Type>::value()) const {
        if constexpr(std::is_same_v<Type, typename Registry::entity_type>) {
            const auto alive = registry->identity.free_list();
            archive(static_cast<id_type>(alive));

            for(std::size_t pos = 0u; pos < alive; ++pos) {
                archive(registry->identity.data()[pos]);
            }
        } else if(const auto *storage = registry->template storage<Type>(id); storage != nullptr) {
            const auto size = storage->size();
            archive(static_cast<id_type>(size));

            std::vector<std::remove_const_t<Type>> values(size);
            storage->download(values.data());

            for(std::size_t pos = 0u; pos < size; ++pos) {
                archive(storage->data()[pos]);

                if constexpr(!std::is_empty_v<Type>) {
                    archive(std::as_const(values)[pos]);
                }
            }
        } else {
            archive(id_type{0});
        }

        return *this;
    }
};

/**
 * @brief Utility to restore a snapshot into an empty registry.
 */
template<typename Registry>
class gpu_snapshot_loader {
    Registry *registry;

public:
    explicit gpu_snapshot_loader(Registry &owner) noexcept
        : registry{&owner} {}

    template<typename Type, typename Archive>
    gpu_snapshot_loader &get(Archive &archive, const id_type id = type_hash<Type>::value()) {
        using entity_type = typename Registry::entity_type;
        id_type length{};
        archive(length);

        if constexpr(std::is_same_v<Type, entity_type>) {
            for(id_type pos = 0u; pos < length; ++pos) {
                entity_type entt{};
                archive(entt);
                registry->identity.generate(entt);
            }
        } else {
            auto &storage = registry->template storage<Type>(id);

            while(length--) {
                entity_type entt{};
                archive(entt);

                if(entt != entt::null) {
                    if(!registry->identity.contains(entt)) {
                        registry->identity.generate(entt);
                    }

                    if constexpr(std::is_empty_v<Type>) {
                        storage.emplace(entt, nullptr);
                    } else {
                        std::remove_const_t<Type> value{};
                        archive(value);
                        storage.emplace(entt, nullptr, value);
                    }
                }
            }
        }

        return *this;
    }
};

#ifdef ENTT_DEVICE_CODE

template<typename Entity, typename Backend, typename... Get, typename... Exclude>
template<typename Func>
typename Backend::error_type gpu_group<Entity, Backend, owned_t<>, get_t<Get...>, exclude_t<Exclude...>>::each(Func fn, typename Backend::stream_type stream, typename Backend::event_type done) const {
    return [&]<std::size_t... Index>(std::index_sequence<Index...>) {
        if(const auto err = registry->fault; err != Backend::success) {
            return err;
        }

        const Entity *candidate{};

        if(const auto err = registry->upload_list(members.data(), members.size(), candidate, stream); err != Backend::success) {
            return registry->fail(err);
        }

        using types = std::tuple<Get...>;
        auto reqs = std::make_tuple(internal::gpu_dispatch_pool<std::tuple_element_t<Index, types>, Entity>{std::get<Index>(get_storages)->pages(), std::get<Index>(get_storages)->template payload_device<std::tuple_element_t<Index, types>>()}...);

        internal::gpu_exclude_params<Entity> excluded{};
        excluded.count = std::min(excluded_pools.size(), internal::gpu_max_exclude);

        for(std::size_t pos = 0u; pos < excluded.count; ++pos) {
            excluded.pages[pos] = excluded_pools[pos]->pages();
        }

        if(members.size() != 0u) {
            const auto block = 256u;
            const auto grid = static_cast<unsigned>((members.size() + block - 1u) / block);
            internal::gpu_view_each_kernel<<<grid, block, 0, stream>>>(fn, members.size(), candidate, reqs, excluded);

            if(const auto err = Backend::get_last_error(); err != Backend::success) {
                return registry->fail(err);
            }
        }

        if(done != nullptr) {
            if(const auto err = Backend::event_record(done, stream); err != Backend::success) {
                return registry->fail(err);
            }
        }

        return Backend::success;
    }(std::index_sequence_for<Get...>{});
}

template<typename Entity, typename Backend, typename... Owned, typename... Get, typename... Exclude>
template<typename Func>
typename Backend::error_type gpu_group<Entity, Backend, owned_t<Owned...>, get_t<Get...>, exclude_t<Exclude...>>::each(Func fn, typename Backend::stream_type stream, typename Backend::event_type done) const {
    return [&]<std::size_t... O, std::size_t... G>(std::index_sequence<O...>, std::index_sequence<G...>) {
        if(const auto err = registry->fault; err != Backend::success) {
            return err;
        }

        const auto *leading = std::get<0>(owned_storages);
        const Entity *candidate = leading->packed_device();

        using owned_types = std::tuple<Owned...>;
        using get_types = std::tuple<Get...>;

        auto reqs = std::tuple_cat(
            std::make_tuple(internal::gpu_dispatch_pool<std::tuple_element_t<O, owned_types>, Entity>{std::get<O>(owned_storages)->pages(), std::get<O>(owned_storages)->template payload_device<std::tuple_element_t<O, owned_types>>()}...),
            std::make_tuple(internal::gpu_dispatch_pool<std::tuple_element_t<G, get_types>, Entity>{std::get<G>(get_storages)->pages(), std::get<G>(get_storages)->template payload_device<std::tuple_element_t<G, get_types>>()}...));

        internal::gpu_exclude_params<Entity> excluded{};
        excluded.count = std::min(excluded_pools.size(), internal::gpu_max_exclude);

        for(std::size_t pos = 0u; pos < excluded.count; ++pos) {
            excluded.pages[pos] = excluded_pools[pos]->pages();
        }

        if(length != 0u) {
            const auto block = 256u;
            const auto grid = static_cast<unsigned>((length + block - 1u) / block);
            internal::gpu_view_each_kernel<<<grid, block, 0, stream>>>(fn, length, candidate, reqs, excluded);

            if(const auto err = Backend::get_last_error(); err != Backend::success) {
                return registry->fail(err);
            }
        }

        if(done != nullptr) {
            if(const auto err = Backend::event_record(done, stream); err != Backend::success) {
                return registry->fail(err);
            }
        }

        return Backend::success;
    }(std::index_sequence_for<Owned...>{}, std::index_sequence_for<Get...>{});
}

template<typename Entity, typename Allocator, typename Backend>
template<typename Type, typename... Func>
inline typename Backend::error_type basic_gpu_registry<Entity, Allocator, Backend>::patch_impl(const Entity value, typename Backend::stream_type stream, typename Backend::event_type done, Func &&...func) {
    if(const auto err = fault; err != Backend::success) {
        return err;
    }

    if(!valid(value)) {
        return Backend::invalid_value;
    }

    auto &component = pool<Type>();

    if(!component.contains(value)) {
        return Backend::invalid_value;
    }

    if constexpr(!std::is_empty_v<Type>) {
        internal::gpu_patch_kernel<<<1, 1, 0, stream>>>(component.pages(), component.template payload_device<Type>(), value, func...);

        if(const auto err = Backend::get_last_error(); err != Backend::success) {
            return fail(err);
        }
    }

    publish_update(type_hash<Type>::value(), value, stream);

    if(const auto err = record_done(done, stream); err != Backend::success) {
        return fail(err);
    }

    return Backend::success;
}

template<typename Entity, typename Backend, typename... Type>
template<typename Func>
typename Backend::error_type gpu_view<Entity, Backend, Type...>::each(Func fn, typename Backend::stream_type stream, typename Backend::event_type done) const {
    return dispatch(fn, stream, done, std::index_sequence_for<Type...>{});
}

template<typename Entity, typename Backend, typename... Type>
template<typename Func, std::size_t... Index>
typename Backend::error_type gpu_view<Entity, Backend, Type...>::dispatch(Func fn, typename Backend::stream_type stream, typename Backend::event_type done, std::index_sequence<Index...>) const {
    if(const auto err = registry->fault; err != Backend::success) {
        return err;
    }

    using types = std::tuple<Type...>;

    std::size_t count{};
    const Entity *candidate{};

    if constexpr(sizeof...(Type) == 0u) {
        if(const auto err = registry->upload_identity(count, candidate, stream); err != Backend::success) {
            return registry->fail(err);
        }
    } else {
        std::array<internal::gpu_pool_base<Entity, Backend> *, sizeof...(Type)> bases{static_cast<internal::gpu_pool_base<Entity, Backend> *>(std::get<Index>(storages))...};
        std::size_t chosen = 0u;

        for(std::size_t pos = 1u; pos < sizeof...(Type); ++pos) {
            if(bases[pos]->size() < bases[chosen]->size()) {
                chosen = pos;
            }
        }

        count = bases[chosen]->size();
        candidate = bases[chosen]->packed_device();
    }

    auto reqs = std::make_tuple(internal::gpu_dispatch_pool<std::tuple_element_t<Index, types>, Entity>{std::get<Index>(storages)->pages(), std::get<Index>(storages)->template payload_device<std::tuple_element_t<Index, types>>()}...);

    internal::gpu_exclude_params<Entity> excluded_params{};
    excluded_params.count = std::min(excluded_pools.size(), internal::gpu_max_exclude);

    for(std::size_t pos = 0u; pos < excluded_params.count; ++pos) {
        excluded_params.pages[pos] = excluded_pools[pos]->pages();
    }

    if(count != 0u) {
        const auto block = 256u;
        const auto grid = static_cast<unsigned>((count + block - 1u) / block);
        internal::gpu_view_each_kernel<<<grid, block, 0, stream>>>(fn, count, candidate, reqs, excluded_params);

        if(const auto err = Backend::get_last_error(); err != Backend::success) {
            return registry->fail(err);
        }
    }

    if(done != nullptr) {
        if(const auto err = Backend::event_record(done, stream); err != Backend::success) {
            return registry->fail(err);
        }
    }

    return Backend::success;
}

template<typename Entity, typename Backend>
template<typename Func>
typename Backend::error_type gpu_runtime_view<Entity, Backend>::each(Func fn, typename Backend::stream_type stream, typename Backend::event_type done) const {
    if(registry == nullptr) {
        return Backend::invalid_value;
    }

    if(const auto err = registry->fault; err != Backend::success) {
        return err;
    }

    std::size_t count{};
    const Entity *candidate{};

    if(pools.empty()) {
        if(const auto err = registry->upload_identity(count, candidate, stream); err != Backend::success) {
            return registry->fail(err);
        }
    } else {
        count = pools.front()->size();
        candidate = pools.front()->packed_device();
    }

    internal::gpu_runtime_params<Entity> params{};
    params.required_count = std::min(pools.size(), internal::gpu_max_exclude);

    for(std::size_t pos = 0u; pos < params.required_count; ++pos) {
        params.required[pos] = pools[pos]->pages();
    }

    params.excluded_count = std::min(filter.size(), internal::gpu_max_exclude);

    for(std::size_t pos = 0u; pos < params.excluded_count; ++pos) {
        params.excluded[pos] = filter[pos]->pages();
    }

    if(count != 0u) {
        const auto block = 256u;
        const auto grid = static_cast<unsigned>((count + block - 1u) / block);
        internal::gpu_runtime_each_kernel<<<grid, block, 0, stream>>>(fn, count, candidate, params);

        if(const auto err = Backend::get_last_error(); err != Backend::success) {
            return registry->fail(err);
        }
    }

    if(done != nullptr) {
        if(const auto err = Backend::event_record(done, stream); err != Backend::success) {
            return registry->fail(err);
        }
    }

    return Backend::success;
}

#endif // ENTT_DEVICE_CODE

} // namespace entt

#endif
