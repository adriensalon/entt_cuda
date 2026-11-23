#include <cstdio>
#include <entt/entt_hip.hpp>
#include <hip/hip_runtime.h>

namespace {

template<typename Func>
__global__ void invoke(Func fn, int *value) {
    fn(*value);
}

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

    hipStream_t upload{};
    hipStream_t download{};
    hipEvent_t ready{};
    hipEvent_t done{};
    int *host{};
    int *device{};

    const auto cleanup = [&] {
        // Test cleanup may block; registry submission APIs will not.
        if(upload) { hipStreamSynchronize(upload); }
        if(download) { hipStreamSynchronize(download); }
        if(device) { hipFree(device); }
        if(host) { hipHostFree(host); }
        if(done) { hipEventDestroy(done); }
        if(ready) { hipEventDestroy(ready); }
        if(download) { hipStreamDestroy(download); }
        if(upload) { hipStreamDestroy(upload); }
    };

    if(!check(hipStreamCreateWithFlags(&upload, hipStreamNonBlocking))
       || !check(hipStreamCreateWithFlags(&download, hipStreamNonBlocking))
       || !check(hipEventCreateWithFlags(&ready, hipEventDisableTiming))
       || !check(hipEventCreateWithFlags(&done, hipEventDisableTiming))
       || !check(hipHostMalloc(reinterpret_cast<void **>(&host), sizeof(int), hipHostMallocDefault))
       || !check(hipMalloc(reinterpret_cast<void **>(&device), sizeof(int)))) {
        cleanup();
        return 1;
    }

    *host = 7;
    bool success = check(hipMemcpyAsync(device, host, sizeof(int), hipMemcpyHostToDevice, upload));

    if(success) {
        invoke<<<1, 1, 0, upload>>>([increment = 5] __device__(int &value) { value += increment; }, device);
        success = check(hipGetLastError())
                  && check(hipEventRecord(ready, upload))
                  && check(hipStreamWaitEvent(download, ready, 0))
                  && check(hipMemcpyAsync(host, device, sizeof(int), hipMemcpyDeviceToHost, download))
                  && check(hipEventRecord(done, download))
                  && check(hipEventSynchronize(done));
    }

    success = success && (*host == 12);
    cleanup();
    return success ? 0 : 1;
}
