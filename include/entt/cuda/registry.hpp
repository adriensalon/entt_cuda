#ifndef ENTT_CUDA_REGISTRY_HPP
#define ENTT_CUDA_REGISTRY_HPP

#include <utility>
#include <cuda_runtime_api.h>
#include <entt/entity/fwd.hpp>
#include "contract.hpp"

namespace entt {

template<typename Type>
class cuda_storage;

/**
 * @brief CUDA query contract. Dispatch definitions are delivered in phase 1.
 * @tparam Type Required component types, possibly const-qualified.
 */
template<typename... Type>
struct cuda_view {
    template<typename Func>
    cudaError_t each(Func fn, cudaStream_t stream = nullptr, cudaEvent_t done = nullptr) const;
};

/**
 * @brief Phase-0 registry API contract, not yet a runtime implementation.
 *
 * The declarations and emplacement forwarding are available for compilation
 * checks. Construction, storage, transfers and dispatch require phase 1.
 */
struct cuda_registry {
    cuda_registry();
    cuda_registry(const cuda_registry &) = delete;
    cuda_registry(cuda_registry &&) noexcept;
    ~cuda_registry();

    cuda_registry &operator=(const cuda_registry &) = delete;
    cuda_registry &operator=(cuda_registry &&) noexcept;

    [[nodiscard]] entity create();

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
    [[nodiscard]] cuda_storage<Type> &storage();

private:
    template<typename Type, typename... Args>
    requires internal::cuda_component_constructible<Type, Args...>
    cudaError_t emplace_impl(entity value, cudaStream_t stream, cudaEvent_t done, Args &&...args);
};

} // namespace entt

#endif
