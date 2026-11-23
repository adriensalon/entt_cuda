#ifndef ENTT_ENTT_CUDA_HPP
#define ENTT_ENTT_CUDA_HPP

// IWYU pragma: begin_exports
#include "backend/cuda.hpp"
#include "gpu/registry.hpp"
// IWYU pragma: end_exports

namespace entt {

/*! @brief CUDA spelling of the generic registry bound to the CUDA backend. */
template<typename Entity = entity, typename Allocator = std::allocator<Entity>>
using basic_cuda_registry = basic_gpu_registry<Entity, Allocator, cuda_backend>;

/*! @brief Default CUDA registry. */
using cuda_registry = basic_cuda_registry<>;

/*! @brief CUDA storage bound to the CUDA backend. */
template<typename Type, typename Entity = entity>
using cuda_storage = gpu_storage<Type, Entity, cuda_backend>;

/*! @brief CUDA view bound to the CUDA backend. */
template<typename Entity = entity, typename... Type>
using cuda_view = gpu_view<Entity, cuda_backend, Type...>;

/*! @brief CUDA runtime view bound to the CUDA backend. */
template<typename Entity = entity>
using cuda_runtime_view = gpu_runtime_view<Entity, cuda_backend>;

/*! @brief CUDA group bound to the CUDA backend. */
template<typename Entity = entity, typename Owned = owned_t<>, typename Get = get_t<>, typename Exclude = exclude_t<>>
using cuda_group = gpu_group<Entity, cuda_backend, Owned, Get, Exclude>;

/*! @brief CUDA snapshot utility over any registry. */
template<typename Registry>
using cuda_snapshot = gpu_snapshot<Registry>;

/*! @brief CUDA snapshot loader utility over any registry. */
template<typename Registry>
using cuda_snapshot_loader = gpu_snapshot_loader<Registry>;

} // namespace entt

#endif
