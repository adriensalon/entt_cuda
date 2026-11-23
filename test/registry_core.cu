#include <array>
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
struct unused {
    int value;
};

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

    position *pinned{};

    if(!check(cudaMallocHost(reinterpret_cast<void **>(&pinned), sizeof(position)), "pinned alloc")) {
        return 1;
    }

    entt::cuda_registry registry;

    // Enough entities to cross the sparse-page boundary at entity index 4096 and
    // to force repeated device growth/retirement.
    constexpr std::size_t entity_count = 5000u;
    std::vector<entt::entity> entities;
    entities.reserve(entity_count);

    for(std::size_t i = 0; i < entity_count; ++i) {
        const auto entity = registry.create();
        entities.push_back(entity);

        if(!check(registry.emplace<position>(entity, stream, position{static_cast<float>(i), 0.f, 0.f}), "emplace position")
           || !check(registry.emplace<velocity>(entity, stream, velocity{1.f, 2.f, 3.f}), "emplace velocity")) {
            return 1;
        }
    }

    // Core dispatch: pos.x += vel.x * dt on a single stream.
    if(!check(registry.view<position, const velocity>().each(
                  [dt = 0.5f] __device__(position &pos, const velocity &vel) { pos.x += vel.x * dt; }, stream, done),
              "each")) {
        return 1;
    }

    check(cudaEventSynchronize(done), "sync each");

    for(const std::size_t i: std::array<std::size_t, 5>{0u, 10u, 4095u, 4096u, entity_count - 1u}) {
        if(!check(registry.get<position>(entities[i], pinned, stream, done), "get")) {
            return 1;
        }

        check(cudaEventSynchronize(done), "sync get");

        if(pinned->x != static_cast<float>(i) + 0.5f) {
            std::fprintf(stderr, "value mismatch at %zu: got %f\n", i, pinned->x);
            return 1;
        }
    }

    // Swap-and-pop: removing a middle component relocates the last value.
    const auto removed = entities[entity_count / 2u];

    if(!check(registry.remove<position>(removed, stream, done), "remove")) {
        return 1;
    }

    check(cudaEventSynchronize(done), "sync remove");

    if(registry.get<position>(removed, pinned, stream, done) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "get of a removed component should fail\n");
        return 1;
    }

    for(std::size_t i = 0; i < entity_count; ++i) {
        if(i == entity_count / 2u) {
            continue;
        }

        if(!check(registry.get<position>(entities[i], pinned, stream, done), "get after remove")) {
            return 1;
        }

        check(cudaEventSynchronize(done), "sync get after remove");

        if(pinned->x != static_cast<float>(i) + 0.5f) {
            std::fprintf(stderr, "relocation mismatch at %zu: got %f\n", i, pinned->x);
            return 1;
        }
    }

    // Tolerant removal of an absent component succeeds.
    if(!check(registry.remove<position>(removed, stream, done), "tolerant remove")) {
        return 1;
    }

    // assign/get round trip.
    const position uploaded{42.f, 1.f, 2.f};

    if(!check(registry.assign<position>(entities[0], &uploaded, stream, done), "assign")
       || !check(cudaEventSynchronize(done), "sync assign")
       || !check(registry.get<position>(entities[0], pinned, stream, done), "get after assign")
       || !check(cudaEventSynchronize(done), "sync get after assign")) {
        return 1;
    }

    if(pinned->x != 42.f || pinned->y != 1.f || pinned->z != 2.f) {
        std::fprintf(stderr, "assign/get mismatch\n");
        return 1;
    }

    // Device-callable patch applies ordered functions to one component.
    if(!check(registry.patch<position>(entities[0], stream, done, [] __device__(position &pos) { pos.y += 10.f; }), "patch")
       || !check(cudaEventSynchronize(done), "sync patch")
       || !check(registry.get<position>(entities[0], pinned, stream, done), "get after patch")
       || !check(cudaEventSynchronize(done), "sync get after patch")) {
        return 1;
    }

    if(pinned->y != 11.f) {
        std::fprintf(stderr, "patch mismatch\n");
        return 1;
    }

    // Tags, exclusions and callback forms.
    constexpr std::size_t tag_count = 10u;

    for(std::size_t i = 0; i < tag_count; ++i) {
        if(!check(registry.emplace<tag>(entities[i], stream), "emplace tag")) {
            return 1;
        }
    }

    check(cudaStreamSynchronize(stream), "sync tags");

    int *counter{};

    if(!check(cudaMalloc(reinterpret_cast<void **>(&counter), sizeof(int)), "counter alloc")) {
        return 1;
    }

    // Tag-only view with an empty callback.
    if(!check(registry.view<tag>().each([] __device__() {}, stream, done), "tag view")) {
        return 1;
    }

    check(cudaEventSynchronize(done), "sync tag view");

    // Excluded view: positions without the tag component.
    check(cudaMemset(counter, 0, sizeof(int)), "counter memset");

    if(!check(registry.view<position>(entt::exclude<tag>).each([counter] __device__(position &) { atomicAdd(counter, 1); }, stream, done),
              "excluded view")) {
        return 1;
    }

    check(cudaEventSynchronize(done), "sync excluded view");

    int host_count{};
    check(cudaMemcpy(&host_count, counter, sizeof(int), cudaMemcpyDeviceToHost), "counter copy");

    // One of the entity_count positions was removed; tag_count of the rest are tagged.
    if(host_count != static_cast<int>(entity_count - 1u - tag_count)) {
        std::fprintf(stderr, "exclusion count mismatch: got %d\n", host_count);
        return 1;
    }

    // Entity-only dispatch and an empty (missing) required pool.
    if(!check(registry.view<>().each([] __device__(entt::entity) {}, stream, done), "entity view")) {
        return 1;
    }

    check(cudaEventSynchronize(done), "sync entity view");

    if(!check(registry.view<unused>().each([] __device__(unused &) {}, stream, done), "missing pool view")) {
        return 1;
    }

    check(cudaEventSynchronize(done), "sync missing pool view");

    // Destroy recycles the identifier with a bumped version.
    const auto victim = entities[1];
    const auto victim_entity = entt::to_entity(victim);
    const auto victim_version = entt::to_version(victim);

    if(!check(registry.destroy(victim, stream, done), "destroy")) {
        return 1;
    }

    check(cudaEventSynchronize(done), "sync destroy");

    const auto recycled = registry.create();

    if(entt::to_entity(recycled) != victim_entity || entt::to_version(recycled) == victim_version) {
        std::fprintf(stderr, "destroy/recycle version mismatch\n");
        return 1;
    }

    // Cross-stream handoff and move construction.
    {
        entt::cuda_registry source;

        const auto entity = source.create();

        if(!check(source.emplace<position>(entity, stream, position{1.f, 2.f, 3.f}), "cross emplace")) {
            return 1;
        }

        cudaStream_t other{};
        cudaEvent_t handoff{};
        cudaEvent_t done2{};

        if(!check(cudaStreamCreateWithFlags(&other, cudaStreamNonBlocking), "other stream")
           || !check(cudaEventCreateWithFlags(&handoff, cudaEventDisableTiming), "handoff event")
           || !check(cudaEventCreateWithFlags(&done2, cudaEventDisableTiming), "done2 event")) {
            return 1;
        }

        // Publish on `stream`, consume on `other`.
        if(!check(cudaEventRecord(handoff, stream), "record handoff")
           || !check(cudaStreamWaitEvent(other, handoff, 0), "wait handoff")
           || !check(source.view<position>().each([] __device__(position &pos) { pos.x += 5.f; }, other, done2), "cross each")
           || !check(cudaEventSynchronize(done2), "sync cross each")) {
            return 1;
        }

        // Move transfers ownership and outstanding bookkeeping.
        entt::cuda_registry moved{std::move(source)};

        if(!check(moved.get<position>(entity, pinned, stream, done), "get after move")
           || !check(cudaEventSynchronize(done), "sync get after move")) {
            return 1;
        }

        if(pinned->x != 6.f) {
            std::fprintf(stderr, "cross-stream/move mismatch: %f\n", pinned->x);
            return 1;
        }

        // The moved registry still grows and retires device buffers correctly.
        const auto extra = moved.create();

        if(!check(moved.emplace<position>(extra, stream, position{9.f, 9.f, 9.f}), "emplace after move")) {
            return 1;
        }

        check(cudaStreamSynchronize(stream), "sync after move");

        cudaEventDestroy(done2);
        cudaEventDestroy(handoff);
        cudaStreamDestroy(other);
    }

    std::puts("cuda registry core tests passed.");
    check(cudaFree(counter), "counter free");
    check(cudaFreeHost(pinned), "pinned free");
    cudaEventDestroy(done);
    cudaStreamDestroy(stream);
    return 0;
}
