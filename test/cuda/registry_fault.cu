#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <cuda_runtime.h>
#include <entt/entt_cuda.hpp>

namespace {

struct position {
    float x, y, z;
};
struct velocity {
    float x, y, z;
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
    const auto entity = registry.create();
    const auto healthy = registry.create();

    if(!check(registry.emplace<position>(entity, stream, position{1.f, 2.f, 3.f}), "emplace")) {
        return 1;
    }

    check(cudaStreamSynchronize(stream), "sync");

    // A never-created identifier is invalid but does not poison the registry.
    const auto bogus = entt::entity{12345};

    if(registry.destroy(bogus, stream, done) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "destroy of an invalid entity should be rejected\n");
        return 1;
    }

    // Validation rejections return cudaErrorInvalidValue without faulting.
    if(registry.emplace<position>(entity, stream, position{9.f, 9.f, 9.f}) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "duplicate emplace should be rejected\n");
        return 1;
    }

    if(registry.get<position>(entity, nullptr, stream, done) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "null destination should be rejected\n");
        return 1;
    }

    if(registry.get<position>(healthy, pinned, stream, done) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "get of a missing component should be rejected\n");
        return 1;
    }

    if(registry.assign<position>(entity, nullptr, stream, done) != cudaErrorInvalidValue) {
        std::fprintf(stderr, "null source should be rejected\n");
        return 1;
    }

    // The registry is still healthy after those rejections.
    if(!check(registry.emplace<velocity>(entity, stream, velocity{4.f, 5.f, 6.f}), "emplace after rejections")) {
        return 1;
    }

    check(cudaStreamSynchronize(stream), "sync after rejections");

    // A submission on a stale stream handle fails and enters the fault state.
    cudaStream_t stale{};

    if(!check(cudaStreamCreateWithFlags(&stale, cudaStreamNonBlocking), "stale stream create")
       || !check(cudaStreamDestroy(stale), "stale stream destroy")) {
        return 1;
    }

    const auto fault = registry.emplace<position>(healthy, stale, position{0.f, 0.f, 0.f});

    if(fault == cudaSuccess) {
        std::fprintf(stderr, "stale-stream submission unexpectedly succeeded\n");
        return 1;
    }

    // The fault is sticky: every status-returning operation reports it. Each of
    // these would return cudaSuccess on a healthy registry.
    if(registry.emplace<position>(healthy, stream, position{1.f, 1.f, 1.f}) != fault
       || registry.get<position>(entity, pinned, stream, done) != fault
       || registry.remove<position>(entity, stream, done) != fault
       || registry.destroy(entity, stream, done) != fault
       || registry.view<position>().each([] __device__(position &) {}, stream, done) != fault) {
        std::fprintf(stderr, "fault state is not sticky across operations\n");
        return 1;
    }

    // create() throws in the fault state.
    bool threw = false;

    try {
        (void)registry.create();
    } catch(const std::logic_error &) {
        threw = true;
    }

    if(!threw) {
        std::fprintf(stderr, "create() did not throw in the fault state\n");
        return 1;
    }

    // Teardown must remain safe in the fault state (destructor runs here).
    std::puts("fault state tests passed.");
    check(cudaFreeHost(pinned), "pinned free");
    cudaEventDestroy(done);
    cudaStreamDestroy(stream);
    return 0;
}
