#ifndef ENTT_GPU_DEVICE_HPP
#define ENTT_GPU_DEVICE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>
#include <entt/entity/entity.hpp>
#include "attributes.hpp"
#include "contract.hpp"

namespace entt {
namespace internal {

/*! @brief Public EnTT entity traits (to_entity, to_version, combine, page size). */
template<typename Entity>
using entity_traits = ::entt::entt_traits<Entity>;

/*! @brief Page size of the device sparse entity-index to dense-index table. */
template<typename Entity>
inline constexpr std::size_t gpu_sparse_page = entity_traits<Entity>::page_size;

/*! @brief Upper bound on the number of sparse pages (one per page index). */
template<typename Entity>
inline constexpr std::size_t gpu_sparse_page_count = (entity_traits<Entity>::entity_mask / gpu_sparse_page<Entity>)+1u;

/*! @brief Upper bound on the number of excluded pools a view can carry. */
inline constexpr std::size_t gpu_max_exclude = 8u;

/*! @brief Raw null sparse cell, the all-ones identifier. */
template<typename Entity>
inline constexpr Entity gpu_null_entity = static_cast<Entity>(entity_traits<Entity>::to_integral(entt::null));

/*! @brief Combines a dense index and an entity into a sparse cell value. */
template<typename Entity>
[[nodiscard]] constexpr Entity gpu_combine(const std::size_t pos, const Entity value) noexcept {
    return entity_traits<Entity>::combine(static_cast<typename entity_traits<Entity>::entity_type>(pos), entity_traits<Entity>::to_integral(value));
}

/*! @brief Returns the sentinel used for the "no member" result of a resolve. */
[[nodiscard]] constexpr std::size_t gpu_npos() noexcept {
    return static_cast<std::size_t>(-1);
}

/**
 * @brief Dispatch descriptor for one required pool.
 * @tparam Type Component type, possibly const-qualified.
 * @tparam Entity Underlying entity identifier.
 */
template<typename Type, typename Entity = entity>
struct gpu_dispatch_pool {
    using element_type = std::remove_const_t<Type>;
    static constexpr bool has_value = !std::is_empty_v<element_type>;

    const Entity *const *pages;
    Type *payload;
};

/*! @brief Dispatch descriptor for one excluded pool (membership only). */
template<typename Entity = entity>
struct gpu_dispatch_exclude {
    const Entity *const *pages;
};

/*! @brief Kernel-parameter bundle carrying a bounded list of exclusions. */
template<typename Entity = entity>
struct gpu_exclude_params {
    const Entity *const *pages[gpu_max_exclude];
    std::size_t count;
};

/*! @brief Kernel-parameter bundle for runtime views (required + excluded pools). */
template<typename Entity = entity>
struct gpu_runtime_params {
    const Entity *const *required[gpu_max_exclude];
    std::size_t required_count;
    const Entity *const *excluded[gpu_max_exclude];
    std::size_t excluded_count;
};

#ifdef ENTT_DEVICE_CODE

/*! @brief Resolves the dense index of an entity in a device sparse table. */
template<typename Entity>
ENTT_DEVICE [[nodiscard]] inline std::size_t gpu_sparse_resolve(const Entity *const *pages, const Entity value) noexcept {
    const auto raw = static_cast<std::size_t>(entity_traits<Entity>::to_entity(value));
    const Entity *page = pages[raw / gpu_sparse_page<Entity>];

    if(page == nullptr) {
        return gpu_npos();
    }

    const Entity cell = page[raw % gpu_sparse_page<Entity>];

    if(cell == gpu_null_entity<Entity> || entity_traits<Entity>::to_version(cell) != entity_traits<Entity>::to_version(value)) {
        return gpu_npos();
    }

    return static_cast<std::size_t>(entity_traits<Entity>::to_entity(cell));
}

/*! @brief Checks whether an entity is a member of a device sparse table. */
template<typename Entity>
ENTT_DEVICE [[nodiscard]] inline bool gpu_sparse_contains(const Entity *const *pages, const Entity value) noexcept {
    return gpu_sparse_resolve(pages, value) != gpu_npos();
}

/*! @brief Produces the callback argument tuple for one required pool. */
template<typename Type, typename Entity>
ENTT_DEVICE auto gpu_ref_arg(const gpu_dispatch_pool<Type, Entity> &pool, const std::size_t idx) {
    if constexpr(gpu_dispatch_pool<Type, Entity>::has_value) {
        return std::forward_as_tuple(pool.payload[idx]);
    } else {
        return std::tuple<>{};
    }
}

/*! @brief Invokes a callback with either `(entity, components...)` or `(components...)`. */
template<typename Entity, typename Func, typename RefTuple, std::size_t... Index>
ENTT_DEVICE void gpu_call(Func &fn, const Entity value, RefTuple &refs, std::index_sequence<Index...>) {
    if constexpr(std::is_invocable_v<Func &, Entity, std::tuple_element_t<Index, RefTuple>...>) {
        fn(value, std::get<Index>(refs)...);
    } else {
        fn(std::get<Index>(refs)...);
    }
}

/*! @brief Resolves membership and invokes the callback for a single candidate. */
template<typename Entity, typename Func, typename... Req, std::size_t... Index>
ENTT_DEVICE void gpu_each_entry(Func &fn, const Entity value, const std::tuple<gpu_dispatch_pool<Req, Entity>...> &pools,
                                const gpu_exclude_params<Entity> &excluded, std::index_sequence<Index...>) {
    std::array<std::size_t, sizeof...(Req)> idx{};
    bool valid = true;
    ((idx[Index] = gpu_sparse_resolve(std::get<Index>(pools).pages, value), valid = valid && (idx[Index] != gpu_npos())), ...);

    if(!valid) {
        return;
    }

    for(std::size_t pos = 0u; pos < excluded.count; ++pos) {
        if(gpu_sparse_contains(excluded.pages[pos], value)) {
            return;
        }
    }

    auto refs = std::tuple_cat(gpu_ref_arg(std::get<Index>(pools), idx[Index])...);
    using refs_type = std::remove_reference_t<decltype(refs)>;
    gpu_call(fn, value, refs, std::make_index_sequence<std::tuple_size_v<refs_type>>{});
}

/*! @brief One-thread-per-candidate dispatch kernel. */
template<typename Entity, typename Func, typename... Req>
ENTT_GLOBAL void gpu_view_each_kernel(Func fn, const std::size_t count, const Entity *candidate,
                                      std::tuple<gpu_dispatch_pool<Req, Entity>...> pools, gpu_exclude_params<Entity> excluded) {
    const auto pos = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if(pos >= count) {
        return;
    }

    gpu_each_entry(fn, candidate[pos], pools, excluded, std::index_sequence_for<Req...>{});
}

/*! @brief Applies ordered device-callable functions to one component. */
template<typename Entity, typename Type, typename... Func>
ENTT_GLOBAL void gpu_patch_kernel(const Entity *const *pages, Type *payload, const Entity value, Func... funcs) {
    const auto idx = gpu_sparse_resolve(pages, value);

    if(idx == gpu_npos()) {
        return;
    }

    (funcs(payload[idx]), ...);
}

/*! @brief One-thread-per-candidate dispatch kernel for runtime views. */
template<typename Entity, typename Func>
ENTT_GLOBAL void gpu_runtime_each_kernel(Func fn, const std::size_t count, const Entity *candidate, gpu_runtime_params<Entity> params) {
    const auto pos = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if(pos >= count) {
        return;
    }

    const Entity value = candidate[pos];

    for(std::size_t idx = 0u; idx < params.required_count; ++idx) {
        if(!gpu_sparse_contains(params.required[idx], value)) {
            return;
        }
    }

    for(std::size_t idx = 0u; idx < params.excluded_count; ++idx) {
        if(gpu_sparse_contains(params.excluded[idx], value)) {
            return;
        }
    }

    fn(value);
}

#endif // ENTT_DEVICE_CODE

} // namespace internal
} // namespace entt

#endif
