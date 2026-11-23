#include <cstdio>
#include <cstdlib>
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

bool check(const cudaError_t err, const char *what) {
    if(err != cudaSuccess) {
        std::fprintf(stderr, "%s: status %d (%s)\n", what, static_cast<int>(err), cudaGetErrorString(err));
        return false;
    }

    return true;
}

} // namespace

int main() {
    int devices{};
    const auto available = cudaGetDeviceCount(&devices);

    if(available == cudaErrorNoDevice || available == cudaErrorInsufficientDriver || (available == cudaSuccess && devices == 0)) {
        std::puts("No usable CUDA device; skipping.");
        return 77;
    }

    if(available != cudaSuccess) {
        return 1;
    }

    cudaStream_t stream{};
    cudaEvent_t done{};

    if(!check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create")
       || !check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "event create")) {
        return 1;
    }

    entt::cuda_registry registry;

    const auto entity = registry.create();

    // Default-named pool via emplace.
    if(!check(registry.emplace<position>(entity, stream, position{1.f, 2.f, 3.f}), "emplace default")) {
        return 1;
    }

    check(cudaStreamSynchronize(stream), "sync");

    // Typed named storage is a distinct pool from the default name.
    const auto named = entt::type_hash<position>::value() ^ 0x5A5A5A5Au;

    if(registry.storage<position>(named).size() != 0u || !registry.storage<position>(named).empty()) {
        std::fprintf(stderr, "named storage should start empty\n");
        return 1;
    }

    if(!check(registry.storage<position>(named).emplace(entity, stream, position{4.f, 5.f, 6.f}), "emplace named")) {
        return 1;
    }

    check(cudaStreamSynchronize(stream), "sync named");

    // The default and named pools track membership independently.
    if(!registry.storage<position>().contains(entity) || !registry.storage<position>(named).contains(entity)) {
        std::fprintf(stderr, "default and named pools should both contain the entity\n");
        return 1;
    }

    // Erased lookup by id.
    if(registry.storage(entt::type_hash<position>::value()) == nullptr || registry.storage(named) == nullptr) {
        std::fprintf(stderr, "erased storage lookup should find both pools\n");
        return 1;
    }

    if(registry.storage(named + 1u) != nullptr) {
        std::fprintf(stderr, "erased storage lookup of a missing pool should be null\n");
        return 1;
    }

    // Const typed/erased lookups.
    const entt::cuda_registry &cdevice = registry;

    if(cdevice.storage<position>(named) == nullptr || cdevice.storage<velocity>() != nullptr || cdevice.storage(named) == nullptr) {
        std::fprintf(stderr, "const storage lookups mismatch\n");
        return 1;
    }

    // Enumeration yields every pool (excluding entity storage) with its id.
    std::size_t seen = 0u;
    bool saw_default = false;
    bool saw_named = false;

    for(const auto &[id, pool]: registry.storage()) {
        if(id == entt::type_hash<position>::value()) {
            saw_default = true;
        }

        if(id == named) {
            saw_named = true;
        }

        if(pool.type() == entt::type_hash<position>::value()) {
            ++seen;
        }
    }

    if(!saw_default || !saw_named || seen < 2u) {
        std::fprintf(stderr, "pool enumeration mismatch\n");
        return 1;
    }

    // reset removes the named pool and retires its device allocations.
    if(!check(registry.reset(named, stream), "reset")) {
        return 1;
    }

    if(registry.storage(named) != nullptr || registry.storage<position>(named).size() != 0u) {
        std::fprintf(stderr, "reset should remove the named pool\n");
        return 1;
    }

    check(cudaStreamSynchronize(stream), "sync reset");

    // View contains/size_hint mirror host membership.
    if(!check(registry.emplace<tag>(entity, stream), "emplace tag")) {
        return 1;
    }

    const auto other = registry.create();

    if(!check(registry.emplace<velocity>(other, stream, velocity{1.f, 1.f, 1.f}), "emplace velocity")) {
        return 1;
    }

    check(cudaStreamSynchronize(stream), "sync view setup");

    const auto position_view = registry.view<position>();
    const auto mixed_view = registry.view<position, tag>();
    const auto excluded_view = registry.view<position>(entt::exclude<tag>);
    const auto all_view = registry.view<>();

    if(!position_view.contains(entity) || position_view.contains(other)) {
        std::fprintf(stderr, "view<position>.contains mismatch\n");
        return 1;
    }

    if(!mixed_view.contains(entity) || mixed_view.contains(other)) {
        std::fprintf(stderr, "view<position, tag>.contains mismatch\n");
        return 1;
    }

    if(excluded_view.contains(entity)) {
        std::fprintf(stderr, "view<position>(exclude<tag>).contains should reject the tagged entity\n");
        return 1;
    }

    if(position_view.size_hint() != registry.storage<position>().size()
       || mixed_view.size_hint() != std::min(registry.storage<position>().size(), registry.storage<tag>().size())
       || all_view.size_hint() != 2u) {
        std::fprintf(stderr, "view size_hint mismatch\n");
        return 1;
    }

    // Host entity iteration yields matching entity identifiers only.
    std::size_t matched = 0u;
    position_view.each_entity([&](entt::entity) { ++matched; });

    if(matched != registry.storage<position>().size()) {
        std::fprintf(stderr, "host entity iteration count mismatch\n");
        return 1;
    }

    // rebind re-points the query after the registry has been moved.
    auto rebindable = registry.view<position>();
    entt::cuda_registry moved{std::move(registry)};
    rebindable.rebind(moved);

    if(!rebindable.contains(entity) || rebindable.contains(other)) {
        std::fprintf(stderr, "rebind should preserve membership\n");
        return 1;
    }

    std::puts("registry storage tests passed.");
    cudaEventDestroy(done);
    cudaStreamDestroy(stream);
    return 0;
}
