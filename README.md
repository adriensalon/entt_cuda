# entt_gpu

An EnTT style CUDA or HIP registry with host owned identity/structure and GPU authoritative component values. Execute `__device__` systems from EnTT like views with capture by value lambdas, without any codegen.
The compatibility reference is EnTT revision
`1dcd7aab6548786f7672ff578639188bd2f0d837` (declaring 4.1.0). The submodule is
unchanged.

## Requirements

- CMake 3.28 or newer
- C++20 host compiler
- NVIDIA CUDA Toolkit 13.2 or newer
- or AMD HIP SDK 6.x or newer for AMD GPUs (7.2 or newer for NVIDIA GPUs)
- The pinned EnTT submodule

`EnTT::Gpu` resolves to whichever backend is available, configuration fails if
neither is.

## Usage

Clone, build and run tests:

```powershell
git clone https://github.com/adriensalon/entt_gpu.git entt_gpu
cd entt_gpu
git submodule update --init external/entt
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

Link to a CMake target:

```cmake
add_subdirectory(path/to/entt_gpu)
target_link_libraries(my_target PRIVATE EnTT::Gpu)
```

Use as a 'regular' EnTT registry with CUDA:

```cpp
#include <cstdio>
#include <cstdlib>
#include <entt/entt_cuda.hpp>

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

// we have systems and lambdas at home :)
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

`EnTT::Gpu` is a portable umbrella target: it aliases `EnTT::Cuda` when that
backend is usable, else `EnTT::Hip`, and errors if neither is available.

