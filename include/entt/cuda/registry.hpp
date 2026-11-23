#ifndef ENTT_CUDA_REGISTRY_HPP
#define ENTT_CUDA_REGISTRY_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include <cuda_runtime_api.h>
#include <entt/core/type_info.hpp>
#include <entt/entity/entity.hpp>
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

private:
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

    /*! @brief Reused upstream identity storage (generation + free list). */
    using identity_storage = basic_storage<entity>;
    /*! @brief Pool container keyed by type hash. */
    using pool_container = std::unordered_map<id_type, std::unique_ptr<internal::cuda_pool_base>>;

    identity_storage identity;
    pool_container pools;
    internal::cuda_retirement retirement;

    entity *d_identity{};
    std::size_t identity_cap{};

    cudaError_t fault_{cudaSuccess};

    cuda_registry() = default;
    cuda_registry(const cuda_registry &) = delete;
    cuda_registry(cuda_registry &&other) noexcept;
    ~cuda_registry();

    cuda_registry &operator=(const cuda_registry &) = delete;
    cuda_registry &operator=(cuda_registry &&other) noexcept;

    /*! @brief Creates an identifier; CPU-only, no device submission. */
    [[nodiscard]] entity create();

    /*! @brief Destroys an entity, removing every component and recycling its ID. */
    cudaError_t destroy(entity value, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

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

    template<typename Type>
    requires internal::cuda_component<std::remove_const_t<Type>>
    cudaError_t remove(entity value, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    template<typename Type>
    requires (internal::cuda_component<Type> && !std::is_empty_v<Type>)
    cudaError_t get(entity value, Type *pinned_host_dst, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    template<typename Type>
    requires (internal::cuda_component<Type> && !std::is_empty_v<Type>)
    cudaError_t assign(entity value, const Type *pinned_host_src, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr);

    template<typename... Type, typename... Exclude>
    [[nodiscard]] cuda_view<Type...> view(exclude_t<Exclude...> = exclude_t{});

    template<typename Type>
    requires internal::cuda_component<Type>
    [[nodiscard]] cuda_storage<Type> &storage() {
        return pool<Type>();
    }

private:
    [[nodiscard]] bool valid(const entity value) const noexcept {
        return static_cast<std::size_t>(identity.find(value).index()) < identity.free_list();
    }

    [[nodiscard]] cudaError_t fault() const noexcept {
        return fault_;
    }

    cudaError_t fail(const cudaError_t err) noexcept {
        if(fault_ == cudaSuccess) {
            fault_ = err;
        }

        return err;
    }

    static cudaError_t record_done(cudaEvent_t done, cudaStream_t stream) noexcept {
        return (done != nullptr) ? cudaEventRecord(done, stream) : cudaSuccess;
    }

    template<typename Type>
    [[nodiscard]] cuda_storage<Type> &pool();

    cudaError_t upload_identity(std::size_t &count, const entity *&out, cudaStream_t stream);

    template<typename Type, typename... Args>
    requires internal::cuda_component_constructible<Type, Args...>
    cudaError_t emplace_impl(entity value, cudaStream_t stream, cudaEvent_t done, Args &&...args);
};

cuda_registry::cuda_registry(cuda_registry &&other) noexcept
    : identity{std::move(other.identity)},
      pools{std::move(other.pools)},
      retirement{std::move(other.retirement)},
      d_identity{std::exchange(other.d_identity, nullptr)},
      identity_cap{std::exchange(other.identity_cap, 0u)},
      fault_{other.fault_} {
    for(auto &&[key, pool]: pools) {
        pool->rebind_retirement(retirement);
    }
}

cuda_registry &cuda_registry::operator=(cuda_registry &&other) noexcept {
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
        retirement = std::move(other.retirement);
        d_identity = std::exchange(other.d_identity, nullptr);
        identity_cap = std::exchange(other.identity_cap, 0u);
        fault_ = other.fault_;

        for(auto &&[key, pool]: pools) {
            pool->rebind_retirement(retirement);
        }
    }

    return *this;
}

cuda_registry::~cuda_registry() {
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
}

[[nodiscard]] inline entity cuda_registry::create() {
    if(fault_ != cudaSuccess) {
        throw std::logic_error("entt::cuda_registry is in a fault state");
    }

    constexpr auto max_index = static_cast<std::size_t>(entt_traits<entity>::to_entity(entt::null)) + 1u;

    if(identity.free_list() == identity.size() && identity.size() >= max_index) {
        throw std::length_error("No more entities available");
    }

    return identity.generate();
}

template<typename Type>
[[nodiscard]] inline cuda_storage<Type> &cuda_registry::pool() {
    static_assert(!std::is_const_v<Type>, "Pool types are unqualified and non-const");
    const auto key = type_hash<Type>::value();

    if(const auto it = pools.find(key); it != pools.end()) {
        return static_cast<cuda_storage<Type> &>(*it->second);
    }

    auto ptr = std::make_unique<cuda_storage<Type>>(retirement);
    auto *raw = ptr.get();
    pools.emplace(key, std::move(ptr));
    return *raw;
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

template<typename Type, typename... Args>
requires internal::cuda_component_constructible<Type, Args...>
inline cudaError_t cuda_registry::emplace_impl(const entity value, cudaStream_t stream, cudaEvent_t done, Args &&...args) {
    if(const auto err = fault_; err != cudaSuccess) {
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

template<typename Type>
requires internal::cuda_component<std::remove_const_t<Type>>
inline cudaError_t cuda_registry::remove(const entity value, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault_; err != cudaSuccess) {
        return err;
    }

    if(!valid(value)) {
        return cudaErrorInvalidValue;
    }

    auto &component = pool<std::remove_const_t<Type>>();

    if(const auto err = component.enqueue_remove(value, stream); err != cudaSuccess) {
        return fail(err);
    }

    if(const auto err = record_done(done, stream); err != cudaSuccess) {
        return fail(err);
    }

    return cudaSuccess;
}

template<typename Type>
requires (internal::cuda_component<Type> && !std::is_empty_v<Type>)
inline cudaError_t cuda_registry::get(const entity value, Type *pinned_host_dst, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault_; err != cudaSuccess) {
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
inline cudaError_t cuda_registry::assign(const entity value, const Type *pinned_host_src, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault_; err != cudaSuccess) {
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

template<typename... Type, typename... Exclude>
[[nodiscard]] inline cuda_view<Type...> cuda_registry::view(exclude_t<Exclude...>) {
    cuda_view<Type...> result;
    result.registry = this;
    result.storages = std::make_tuple(static_cast<cuda_storage<std::remove_const_t<Type>> *>(&pool<std::remove_const_t<Type>>())...);
    result.excluded_pools = {static_cast<internal::cuda_pool_base *>(&pool<std::remove_const_t<Exclude>>())...};
    return result;
}

inline cudaError_t cuda_registry::destroy(const entity value, cudaStream_t stream, cudaEvent_t done) {
    if(const auto err = fault_; err != cudaSuccess) {
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

#ifdef __CUDACC__

template<typename... Type>
template<typename Func>
cudaError_t cuda_view<Type...>::each(Func fn, cudaStream_t stream, cudaEvent_t done) const {
    return dispatch(fn, stream, done, std::index_sequence_for<Type...>{});
}

template<typename... Type>
template<typename Func, std::size_t... Index>
cudaError_t cuda_view<Type...>::dispatch(Func fn, cudaStream_t stream, cudaEvent_t done, std::index_sequence<Index...>) const {
    if(const auto err = registry->fault(); err != cudaSuccess) {
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

#endif // __CUDACC__

} // namespace entt

#endif
