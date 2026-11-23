#include <cstdio>
#include <entt/entt_hip.hpp>
#include <hip/hip_runtime.h>

namespace {

struct position {
    float x, y, z;
};

struct velocity {
    float x, y, z;
};

bool check(const hipError_t result) {
    if(result != hipSuccess) {
        const auto *message = hipGetErrorString(result);
        std::fprintf(stderr, "HIP status %d: %s\n", static_cast<int>(result), message ? message : "error text unavailable");
        return false;
    }

    return true;
}

} // namespace

int main() {
    int devices{};
    const auto available = hipGetDeviceCount(&devices);

    if(available == hipErrorNoDevice || available == hipErrorInsufficientDriver || (available == hipSuccess && devices == 0)) {
        std::printf("No usable HIP device (status %d, device count %d).\n", static_cast<int>(available), devices);
#ifdef ENTT_CUDA_REQUIRE_GPU
        return 1;
#else
        return 77;
#endif
    }

    if(!check(available)) {
        return 1;
    }

    hipStream_t stream{};
    hipEvent_t done{};

    if(!check(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking))
       || !check(hipEventCreateWithFlags(&done, hipEventDisableTiming))) {
        return 1;
    }

    // Same-stream emplace -> each -> get round trip, mirroring registry_core.cu.
    entt::hip_registry registry;
    const auto entity = registry.create();

    const bool submitted = check(registry.emplace<position>(entity, stream, position{0.f, 0.f, 0.f}))
                           && check(registry.emplace<velocity>(entity, stream, velocity{1.f, 2.f, 3.f}))
                           && check(registry.view<position, const velocity>().each(
                               [dt = 0.5f] __device__(position &p, const velocity &v) {
                                   p.x += v.x * dt;
                                   p.y += v.y * dt;
                                   p.z += v.z * dt;
                               },
                               stream));

    if(!submitted) {
        return 1;
    }

    position *result = nullptr;

    if(!check(hipHostMalloc(reinterpret_cast<void **>(&result), sizeof(position), hipHostMallocDefault))
       || !check(registry.get<position>(entity, result, stream, done))
       || !check(hipEventSynchronize(done))) {
        return 1;
    }

    bool success = (result->x == 0.5f) && (result->y == 1.0f) && (result->z == 1.5f);

    result->x = 42.f;

    if(success && (!check(registry.assign<position>(entity, result, stream, done)) || !check(hipEventSynchronize(done)))) {
        success = false;
    }

    if(!check(hipHostFree(result))
       || !check(registry.destroy(entity, stream, done))
       || !check(hipEventSynchronize(done))
       || !check(hipEventDestroy(done))
       || !check(hipStreamDestroy(stream))) {
        success = false;
    }

    std::puts(success ? "HIP registry core round trip passed." : "HIP registry core round trip failed.");
    return success ? 0 : 1;
}
