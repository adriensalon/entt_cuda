# entt_cuda

An EnTT style CUDA registry with host owned identity/structure and GPU authoritative
component values. Execute `__device__` systems from EnTT like views with capture by value lambdas, without any codegen.
The compatibility reference is EnTT revision
`1dcd7aab6548786f7672ff578639188bd2f0d837` (declaring 4.1.0). The submodule is
unchanged.

## Requirements

- CMake 3.28 or newer
- A C++20 host compiler (tested with MSVC)
- NVIDIA CUDA Toolkit 13.2 or newer `nvcc`
- A CUDA capable GPU (Turing `sm_75` or newer for the CUDA 13 toolchain)
- The pinned EnTT submodule, checked out with `git submodule update --init external/entt`

## Usage

Clone, build and run tests:

```powershell
git clone https://github.com/adriensalon/entt_cuda.git entt_cuda
cd entt_cuda
git submodule update --init external/entt
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

Consumers add the project with `add_subdirectory(...)`, link `EnTT::Cuda`, and include
`<entt/entt_cuda.hpp>`. Device callbacks require CUDA translation units (the target
provides `--extended-lambda`)

```cpp
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

void check(const cudaError_t err) {
    if(err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %d: %s\n", static_cast<int>(err), cudaGetErrorString(err));
        std::abort();
    }
}

struct position { float x, y, z; };
struct velocity { float x, y, z; };

entt::cuda_registry registry;
const auto e = registry.create();
check(registry.emplace<position>(e, stream, position{0, 0, 0}));
check(registry.emplace<velocity>(e, stream, velocity{1, 2, 3}));
check(registry.view<position, const velocity>().each(
    [dt = 0.016f] __device__ (position &p, const velocity &v) {
        p.x += v.x * dt;
        p.y += v.y * dt;
        p.z += v.z * dt;
    }, stream));

position *result = nullptr;
check(cudaMallocHost(reinterpret_cast<void **>(&result), sizeof(position)));
check(registry.get<position>(e, result, stream, done));
check(cudaEventSynchronize(done));
result->x = 42.f;
check(registry.assign<position>(e, result, stream, done));
check(cudaEventSynchronize(done));
check(cudaFreeHost(result));
check(registry.destroy(e, stream, done));
check(cudaEventSynchronize(done));
```

## HIP backend (experimental)

The generic core in `include/entt/gpu/` is backend-agnostic; `include/entt/backend/`
supplies `cuda_backend` and `hip_backend`, and `<entt/entt_hip.hpp>` exposes the
`hip_registry`/`hip_view`/... aliases over the same registry. A ROCm/hipcc
toolchain enables the optional `EnTT::Hip` interface target (detected via
`check_language(HIP)`). The HIP backend is written and clang-format clean but
not yet runtime-verified on this workspace, which has no AMD toolchain; see
[docs/backend-portability.md](docs/backend-portability.md).
