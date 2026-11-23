#ifndef ENTT_CUDA_DEVICE_HPP
#define ENTT_CUDA_DEVICE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>
#include <entt/entity/entity.hpp>
#include "contract.hpp"

namespace entt {

/*! @cond ENTT_INTERNAL */
namespace internal {

/*! @brief Public EnTT entity traits (to_entity, to_version, combine, page size). */
using entity_traits = ::entt::entt_traits<entity>;

/*! @brief Page size of the device sparse entity-index to dense-index table. */
inline constexpr std::size_t cuda_sparse_page = entity_traits::page_size;

/*! @brief Upper bound on the number of sparse pages (one per page index). */
inline constexpr std::size_t cuda_sparse_page_count = (entity_traits::entity_mask / cuda_sparse_page) + 1u;

/*! @brief Upper bound on the number of excluded pools a view can carry. */
inline constexpr std::size_t cuda_max_exclude = 8u;

/*! @brief Raw null sparse cell, the all-ones identifier. */
inline constexpr entity cuda_null_entity = static_cast<entity>(entity_traits::to_integral(entt::null));

/*! @brief Combines a dense index and an entity into a sparse cell value. */
[[nodiscard]] constexpr entity cuda_combine(const std::size_t pos, const entity value) noexcept {
    return entity_traits::combine(static_cast<entity_traits::entity_type>(pos), entity_traits::to_integral(value));
}

/*! @brief Returns the sentinel used for the "no member" result of a resolve. */
[[nodiscard]] constexpr std::size_t cuda_npos() noexcept {
    return static_cast<std::size_t>(-1);
}

/**
 * @brief Dispatch descriptor for one required pool.
 * @tparam Type Component type, possibly const-qualified.
 */
template<typename Type>
struct cuda_dispatch_pool {
    using element_type = std::remove_const_t<Type>;
    static constexpr bool has_value = !std::is_empty_v<element_type>;

    const entity *const *pages;
    Type *payload;
};

/*! @brief Dispatch descriptor for one excluded pool (membership only). */
struct cuda_dispatch_exclude {
    const entity *const *pages;
};

/*! @brief Kernel-parameter bundle carrying a bounded list of exclusions. */
struct cuda_exclude_params {
    const entity *const *pages[cuda_max_exclude];
    std::size_t count;
};

#ifdef __CUDACC__

/*! @brief Resolves the dense index of an entity in a device sparse table. */
__device__ [[nodiscard]] inline std::size_t cuda_sparse_resolve(const entity *const *pages, const entity value) noexcept {
    const auto raw = static_cast<std::size_t>(entity_traits::to_entity(value));
    const entity *page = pages[raw / cuda_sparse_page];

    if(page == nullptr) {
        return cuda_npos();
    }

    const entity cell = page[raw % cuda_sparse_page];

    if(cell == cuda_null_entity || entity_traits::to_version(cell) != entity_traits::to_version(value)) {
        return cuda_npos();
    }

    return static_cast<std::size_t>(entity_traits::to_entity(cell));
}

/*! @brief Checks whether an entity is a member of a device sparse table. */
__device__ [[nodiscard]] inline bool cuda_sparse_contains(const entity *const *pages, const entity value) noexcept {
    return cuda_sparse_resolve(pages, value) != cuda_npos();
}

/*! @brief Produces the callback argument tuple for one required pool. */
template<typename Type>
__device__ auto cuda_ref_arg(const cuda_dispatch_pool<Type> &pool, const std::size_t idx) {
    if constexpr(cuda_dispatch_pool<Type>::has_value) {
        return std::forward_as_tuple(pool.payload[idx]);
    } else {
        return std::tuple<>{};
    }
}

/*! @brief Invokes a callback with either `(entity, components...)` or `(components...)`. */
template<typename Func, typename RefTuple, std::size_t... Index>
__device__ void cuda_call(Func &fn, const entity value, RefTuple &refs, std::index_sequence<Index...>) {
    if constexpr(std::is_invocable_v<Func &, entity, std::tuple_element_t<Index, RefTuple>...>) {
        fn(value, std::get<Index>(refs)...);
    } else {
        fn(std::get<Index>(refs)...);
    }
}

/*! @brief Resolves membership and invokes the callback for a single candidate. */
template<typename Func, typename... Req, std::size_t... Index>
__device__ void cuda_each_entry(Func &fn, const entity value, const std::tuple<cuda_dispatch_pool<Req>...> &pools,
                                const cuda_exclude_params &excluded, std::index_sequence<Index...>) {
    std::array<std::size_t, sizeof...(Req)> idx{};
    bool valid = true;
    ((idx[Index] = cuda_sparse_resolve(std::get<Index>(pools).pages, value), valid = valid && (idx[Index] != cuda_npos())), ...);

    if(!valid) {
        return;
    }

    for(std::size_t pos = 0u; pos < excluded.count; ++pos) {
        if(cuda_sparse_contains(excluded.pages[pos], value)) {
            return;
        }
    }

    auto refs = std::tuple_cat(cuda_ref_arg(std::get<Index>(pools), idx[Index])...);
    using refs_type = std::remove_reference_t<decltype(refs)>;
    cuda_call(fn, value, refs, std::make_index_sequence<std::tuple_size_v<refs_type>>{});
}

/*! @brief One-thread-per-candidate dispatch kernel. */
template<typename Func, typename... Req>
__global__ void cuda_view_each_kernel(Func fn, const std::size_t count, const entity *candidate,
                                      std::tuple<cuda_dispatch_pool<Req>...> pools, cuda_exclude_params excluded) {
    const auto pos = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;

    if(pos >= count) {
        return;
    }

    cuda_each_entry(fn, candidate[pos], pools, excluded, std::index_sequence_for<Req...>{});
}

/*! @brief Applies ordered device-callable functions to one component. */
template<typename Type, typename... Func>
__global__ void cuda_patch_kernel(const entity *const *pages, Type *payload, const entity value, Func... funcs) {
    const auto idx = cuda_sparse_resolve(pages, value);

    if(idx == cuda_npos()) {
        return;
    }

    (funcs(payload[idx]), ...);
}

#endif // __CUDACC__

} // namespace internal
/*! @endcond */

} // namespace entt

#endif
