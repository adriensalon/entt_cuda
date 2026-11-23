#include <cstdio>
#include <type_traits>
#include <entt/entity/registry.hpp>
#include <entt/entt_cuda.hpp>

static_assert(entt::internal::gpu_leading_handle<cudaStream_t, cudaStream_t &>);
static_assert(entt::internal::gpu_leading_handle<cudaStream_t, const cudaStream_t &>);
static_assert(entt::internal::gpu_leading_handle<cudaStream_t, std::nullptr_t, int>);
static_assert(!entt::internal::gpu_leading_handle<cudaStream_t>);
static_assert(!entt::internal::gpu_leading_handle<cudaStream_t, cudaEvent_t>);
static_assert(!entt::internal::gpu_leading_handle<cudaStream_t, int>);
static_assert(entt::internal::gpu_leading_handle<cudaEvent_t, cudaEvent_t>);
static_assert(!entt::internal::gpu_leading_handle<cudaEvent_t, cudaStream_t>);

int main() {
    // Reference expectations adapted from upstream entity/registry.cpp. These
    // check the chosen dependency, not a CUDA registry implementation.
    entt::registry reference;
    const auto old = reference.create();

    if(!reference.valid(old) || !reference.all_of<>(old) || reference.any_of<>(old)) {
        return 1;
    }

    reference.emplace<int>(old, 42);
    reference.destroy(old);
    const auto recycled = reference.create();

    if(reference.valid(old) || !reference.valid(recycled)
       || entt::to_entity(old) != entt::to_entity(recycled)
       || entt::to_version(old) == entt::to_version(recycled)
       || reference.all_of<int>(recycled)) {
        return 1;
    }

    std::puts("Pinned EnTT identity and empty-pack reference contracts passed.");
    return 0;
}
