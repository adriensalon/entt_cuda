#ifndef ENTT_CUDA_CONTRACT_HPP
#define ENTT_CUDA_CONTRACT_HPP

#include <cstddef>
#include <type_traits>
#include <utility>
#include <cuda_runtime_api.h>
#include <entt/entity/component.hpp>
#include <entt/entity/entity.hpp>

namespace entt {

/*! @cond ENTT_INTERNAL */
namespace internal {

template<typename Type>
concept cuda_component = std::is_object_v<Type>
                         && !std::is_array_v<Type>
                         && std::is_same_v<Type, std::remove_cvref_t<Type>>
                         && !std::is_same_v<Type, entity>
                         && std::is_trivially_copyable_v<Type>
                         && std::is_trivially_destructible_v<Type>
                         && !component_traits<Type>::in_place_delete
                         && (component_traits<Type>::page_size == (std::is_empty_v<Type> ? 0u : ENTT_PACKED_PAGE));

template<typename Type, typename... Args>
concept cuda_component_constructible = cuda_component<Type> && requires(Args &&...args) {
    Type{std::forward<Args>(args)...};
};

template<typename Handle, typename... Args>
inline constexpr bool cuda_leading_handle = false;

template<typename Handle, typename First, typename... Other>
inline constexpr bool cuda_leading_handle<Handle, First, Other...> = std::is_same_v<std::remove_cvref_t<First>, Handle> || std::is_null_pointer_v<std::remove_cvref_t<First>>;

} // namespace internal
/*! @endcond */

} // namespace entt

#endif
