#include <cstdio>
#include <cstdlib>
#include <vector>
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

    // Use the default stream so the owning-group arrangement is ordered before dispatch.
    entt::cuda_registry registry;

    position *pinned{};

    if(!check(cudaMallocHost(reinterpret_cast<void **>(&pinned), sizeof(position)), "pinned alloc")) {
        return 1;
    }

    constexpr std::size_t shared = 20u;
    constexpr std::size_t tagged = 5u;
    constexpr std::size_t pos_only = 5u;

    std::vector<entt::entity> shared_entities;

    for(std::size_t i = 0u; i < shared; ++i) {
        const auto entity = registry.create();
        shared_entities.push_back(entity);

        if(!check(registry.emplace<position>(entity, position{static_cast<float>(i), 0.f, 0.f}), "emplace position")
           || !check(registry.emplace<velocity>(entity, velocity{1.f, 1.f, 1.f}), "emplace velocity")) {
            return 1;
        }
    }

    for(std::size_t i = 0u; i < tagged; ++i) {
        if(!check(registry.emplace<tag>(shared_entities[i]), "emplace tag")) {
            return 1;
        }
    }

    for(std::size_t i = 0u; i < pos_only; ++i) {
        const auto entity = registry.create();

        if(!check(registry.emplace<position>(entity, position{100.f, 100.f, 100.f}), "emplace position only")) {
            return 1;
        }
    }

    check(cudaStreamSynchronize(nullptr), "sync setup");

    // Non-owning group: position + velocity, excluding tag.
    auto non_owning = registry.group<>(entt::get<position, velocity>, entt::exclude<tag>);

    if(non_owning.size() != shared - tagged || non_owning.empty() || non_owning.contains(shared_entities[0]) || !non_owning.contains(shared_entities[tagged])) {
        std::fprintf(stderr, "non-owning group membership mismatch\n");
        return 1;
    }

    // Owning group: position and velocity are owned and arranged.
    auto owning = registry.group<position, velocity>();

    if(owning.size() != shared || owning.empty()) {
        std::fprintf(stderr, "owning group size mismatch\n");
        return 1;
    }

    if(!owning.contains(shared_entities[0]) || !registry.owned<position>() || !registry.owned<velocity>() || registry.owned<tag>()) {
        std::fprintf(stderr, "owning group ownership mismatch\n");
        return 1;
    }

    // CPU/GPU slot agreement: values survive the arrangement.
    if(!check(registry.get<position>(shared_entities[10], pinned), "get after arrange")
       || !check(cudaStreamSynchronize(nullptr), "sync get")) {
        return 1;
    }

    if(pinned->x != 10.f) {
        std::fprintf(stderr, "arranged value mismatch: %f\n", pinned->x);
        return 1;
    }

    // Owning-group dispatch modifies owned components in place.
    if(!check(owning.each([] __device__(position &pos, velocity &) { pos.x += 1.f; }), "owning each")
       || !check(cudaStreamSynchronize(nullptr), "sync owning each")
       || !check(registry.get<position>(shared_entities[0], pinned), "get after each")
       || !check(cudaStreamSynchronize(nullptr), "sync get after each")) {
        return 1;
    }

    if(pinned->x != 1.f) {
        std::fprintf(stderr, "owning each did not update the value: %f\n", pinned->x);
        return 1;
    }

    // Non-owning dispatch (observed components), excluding tagged entities.
    if(!check(non_owning.each([] __device__(position &pos, velocity &) { pos.x += 2.f; }), "non-owning each")
       || !check(cudaStreamSynchronize(nullptr), "sync non-owning each")) {
        return 1;
    }

    // group_if_exists performs a const, non-creating lookup.
    const entt::cuda_registry &cdevice = registry;
    auto lookup = cdevice.group_if_exists<position, velocity>(entt::get<>, entt::exclude<tag>);

    if(lookup.size() != shared - tagged) {
        std::fprintf(stderr, "group_if_exists size mismatch\n");
        return 1;
    }

    // compact is a no-op for swap-and-pop pools.
    if(!check(registry.compact<position>(), "compact")) {
        return 1;
    }

    std::puts("registry group tests passed.");
    check(cudaFreeHost(pinned), "pinned free");
    return 0;
}
