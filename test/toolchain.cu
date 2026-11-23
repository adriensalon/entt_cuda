#include <cstdio>
#include <cuda_runtime.h>
#include <entt/entt_cuda.hpp>

namespace {

template<typename Func>
__global__ void invoke(Func fn, int *value) {
    fn(*value);
}

bool check(const cudaError_t result) {
    if(result != cudaSuccess) {
        const auto *message = cudaGetErrorString(result);
        std::fprintf(stderr, "CUDA status %d: %s\n", static_cast<int>(result), message ? message : "error text unavailable");
        return false;
    }

    return true;
}

} // namespace

int main() {
    int devices{};
    const auto available = cudaGetDeviceCount(&devices);

    if(available == cudaErrorNoDevice || available == cudaErrorInsufficientDriver || (available == cudaSuccess && devices == 0)) {
        std::printf("No usable CUDA device (status %d, device count %d).\n", static_cast<int>(available), devices);
#ifdef ENTT_CUDA_REQUIRE_GPU
        return 1;
#else
        return 77;
#endif
    }

    if(!check(available)) {
        return 1;
    }

    cudaStream_t upload{};
    cudaStream_t download{};
    cudaEvent_t ready{};
    cudaEvent_t done{};
    int *host{};
    int *device{};

    const auto cleanup = [&] {
        // Test cleanup may block; registry submission APIs will not.
        if(upload) { cudaStreamSynchronize(upload); }
        if(download) { cudaStreamSynchronize(download); }
        if(device) { cudaFree(device); }
        if(host) { cudaFreeHost(host); }
        if(done) { cudaEventDestroy(done); }
        if(ready) { cudaEventDestroy(ready); }
        if(download) { cudaStreamDestroy(download); }
        if(upload) { cudaStreamDestroy(upload); }
    };

    if(!check(cudaStreamCreateWithFlags(&upload, cudaStreamNonBlocking))
       || !check(cudaStreamCreateWithFlags(&download, cudaStreamNonBlocking))
       || !check(cudaEventCreateWithFlags(&ready, cudaEventDisableTiming))
       || !check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming))
       || !check(cudaMallocHost(reinterpret_cast<void **>(&host), sizeof(int)))
       || !check(cudaMalloc(reinterpret_cast<void **>(&device), sizeof(int)))) {
        cleanup();
        return 1;
    }

    *host = 7;
    bool success = check(cudaMemcpyAsync(device, host, sizeof(int), cudaMemcpyHostToDevice, upload));

    if(success) {
        invoke<<<1, 1, 0, upload>>>([increment = 5] __device__(int &value) { value += increment; }, device);
        success = check(cudaGetLastError())
                  && check(cudaEventRecord(ready, upload))
                  && check(cudaStreamWaitEvent(download, ready, 0))
                  && check(cudaMemcpyAsync(host, device, sizeof(int), cudaMemcpyDeviceToHost, download))
                  && check(cudaEventRecord(done, download))
                  && check(cudaEventSynchronize(done));
    }

    success = success && (*host == 12);
    cleanup();
    return success ? 0 : 1;
}
