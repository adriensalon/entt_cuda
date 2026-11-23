#ifndef ENTT_ENTT_HIP_HPP
#define ENTT_ENTT_HIP_HPP

// IWYU pragma: begin_exports
#include "backend/hip.hpp"
#include "gpu/registry.hpp"
// IWYU pragma: end_exports

namespace entt {

/*! @brief HIP spelling of the generic registry bound to the HIP backend. */
template<typename Entity = entity, typename Allocator = std::allocator<Entity>>
using basic_hip_registry = basic_gpu_registry<Entity, Allocator, hip_backend>;

/*! @brief Default HIP registry. */
using hip_registry = basic_hip_registry<>;

/*! @brief HIP storage bound to the HIP backend. */
template<typename Type, typename Entity = entity>
using hip_storage = gpu_storage<Type, Entity, hip_backend>;

/*! @brief HIP view bound to the HIP backend. */
template<typename Entity = entity, typename... Type>
using hip_view = gpu_view<Entity, hip_backend, Type...>;

/*! @brief HIP runtime view bound to the HIP backend. */
template<typename Entity = entity>
using hip_runtime_view = gpu_runtime_view<Entity, hip_backend>;

/*! @brief HIP group bound to the HIP backend. */
template<typename Entity = entity, typename Owned = owned_t<>, typename Get = get_t<>, typename Exclude = exclude_t<>>
using hip_group = gpu_group<Entity, hip_backend, Owned, Get, Exclude>;

/*! @brief HIP snapshot utility over any registry. */
template<typename Registry>
using hip_snapshot = gpu_snapshot<Registry>;

/*! @brief HIP snapshot loader utility over any registry. */
template<typename Registry>
using hip_snapshot_loader = gpu_snapshot_loader<Registry>;

} // namespace entt

#endif
