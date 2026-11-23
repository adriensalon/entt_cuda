#include <concepts>
#include <type_traits>
#include <utility>
#include <cuda_runtime.h>
#include <entt/entt_cuda.hpp>

namespace {

struct position {
    float x, y, z;
};
struct velocity {
    float x, y, z;
};
struct tag {};
struct stream_value {
    cudaStream_t value;
};
struct event_value {
    cudaEvent_t value;
};
struct pointer_value {
    int *value;
};
struct pointer_constructor {
    explicit pointer_constructor(int *arg): value{arg} {}
    int *value;
};
struct stream_constructor {
    explicit stream_constructor(cudaStream_t arg): value{arg} {}
    cudaStream_t value;
};
struct converted_stream {
    operator cudaStream_t() const {
        return nullptr;
    }
};
struct converted_value {
    converted_stream value;
};
struct nontrivial {
    ~nontrivial() {}
    int value;
};
struct stable {
    [[maybe_unused]] static constexpr bool in_place_delete = true;
    int value;
};
struct custom_page {
    static constexpr std::size_t page_size = 8;
    int value;
};

template<typename Type, typename... Args>
concept can_emplace = requires(entt::cuda_registry &registry, entt::entity entity, Args &&...args) {
    { registry.emplace<Type>(entity, std::forward<Args>(args)...) } -> std::same_as<cudaError_t>;
};

template<typename Type>
concept can_transfer = requires(entt::cuda_registry &registry, entt::entity entity, Type *value) {
    { registry.get<Type>(entity, value) } -> std::same_as<cudaError_t>;
    { registry.assign<Type>(entity, value) } -> std::same_as<cudaError_t>;
};

static_assert(can_emplace<position>);
static_assert(can_emplace<position, float, float, float>);
static_assert(can_emplace<position, cudaStream_t, float, float, float>);
static_assert(can_emplace<position, cudaStream_t, cudaEvent_t, float, float, float>);
static_assert(can_emplace<position, std::nullptr_t, std::nullptr_t, float, float, float>);
static_assert(can_emplace<tag, std::nullptr_t>);
static_assert(can_emplace<tag, std::nullptr_t, std::nullptr_t>);
static_assert(can_emplace<converted_value, converted_stream>);
static_assert(stable::in_place_delete);
static_assert(can_emplace<int, int>);
static_assert(!can_emplace<stream_constructor, cudaStream_t>);
static_assert(can_emplace<stream_constructor, cudaStream_t, cudaEvent_t, cudaStream_t>);
static_assert(!can_emplace<pointer_constructor, std::nullptr_t>);
static_assert(!can_emplace<pointer_constructor, cudaStream_t, std::nullptr_t>);
static_assert(can_emplace<pointer_constructor, cudaStream_t, cudaEvent_t, std::nullptr_t>);
static_assert(!can_emplace<position, int *>);
static_assert(!can_emplace<const position>);
static_assert(!can_emplace<volatile position>);
static_assert(!can_emplace<position &>);
static_assert(!can_emplace<nontrivial>);
static_assert(!can_emplace<stable>);
static_assert(!can_emplace<custom_page>);
static_assert(!can_emplace<entt::entity>);
static_assert(can_transfer<position>);
static_assert(!can_transfer<tag>);
static_assert(!can_transfer<const position>);
static_assert(!std::is_copy_constructible_v<entt::cuda_registry>);
static_assert(!std::is_copy_assignable_v<entt::cuda_registry>);
static_assert(std::is_nothrow_move_constructible_v<entt::cuda_registry>);
static_assert(std::is_nothrow_move_assignable_v<entt::cuda_registry>);

} // namespace

// This object target intentionally does not link: runtime definitions start in
// phase 1. Calls instantiate forwarding bodies and check the actual public API.
void api_contract(entt::cuda_registry &registry, const entt::entity entity, cudaStream_t stream, cudaEvent_t done, position *pinned) {
    const cudaStream_t converted_handle = converted_stream{};
    registry.emplace<int>(entity, 0);
    registry.emplace<int>(entity, stream, 0);
    registry.emplace<position>(entity, converted_handle);
    registry.emplace<position>(entity);
    registry.emplace<position>(entity, 1.f, 2.f, 3.f);
    registry.emplace<position>(entity, position{1.f, 2.f, 3.f});
    registry.emplace<position>(entity, nullptr);
    registry.emplace<position>(entity, stream);
    registry.emplace<position>(entity, stream, 1.f, 2.f, 3.f);
    registry.emplace<position>(entity, stream, done, 1.f, 2.f, 3.f);
    registry.emplace<position>(entity, nullptr, done, position{});
    registry.emplace<position>(entity, stream, nullptr, position{});
    registry.emplace<position>(entity, nullptr, nullptr, 1.f, 2.f, 3.f);
    registry.emplace<stream_value>(entity, stream_value{stream});
    registry.emplace<stream_value>(entity, stream, done, stream);
    registry.emplace<event_value>(entity, done);
    registry.emplace<event_value>(entity, stream, done, done);
    registry.emplace<stream_constructor>(entity, stream_constructor{stream});
    registry.emplace<stream_constructor>(entity, stream, done, stream);
    registry.emplace<pointer_value>(entity, pointer_value{nullptr});
    registry.emplace<pointer_value>(entity, stream, done, nullptr);
    registry.emplace<pointer_constructor>(entity, pointer_constructor{nullptr});
    registry.emplace<pointer_constructor>(entity, stream, done, nullptr);
    registry.emplace<converted_value>(entity, converted_stream{});
    registry.emplace<tag>(entity, nullptr, nullptr);
    registry.remove<tag>(entity, stream, done);
    registry.get<position>(entity, pinned, stream, done);
    registry.assign<position>(entity, pinned, stream, done);
    registry.destroy(entity, stream, done);
    registry.view<position, const velocity>().each(
        [dt = 0.016f] __device__(position &pos, const velocity &vel) { pos.x += dt * vel.x; }, stream, done);
    registry.view<position>(entt::exclude<tag>).each([] __device__(entt::entity, position &pos) { pos.y += 1.f; });
    registry.view<tag>().each([] __device__() {});
    registry.view<>().each([] __device__(entt::entity) {}, nullptr, done);
    registry.view<>(entt::exclude<tag>).each([] __device__(entt::entity) {});
    [[maybe_unused]] auto &pool = registry.storage<position>();
    [[maybe_unused]] const auto created = registry.create();
}
