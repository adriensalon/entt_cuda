# entt_cuda

An EnTT-style CUDA registry with host-owned identity/structure and GPU-authoritative
component values.

**Current status: phase 1 (asynchronous core) complete.** Host identity, device
pools/sparse pages, emplacement, swap-and-pop removal, transfers, and typed device
dispatch (`view<T...>().each`) work end-to-end, with safe allocation retirement,
fault tracking and teardown. Acceptance is covered by `registry_core.cu`
(dispatch/relocation/tags/callbacks/handoffs/moves), `registry_fault.cu` (sticky
fault state) and `registry_parity.cpp` (randomized parity against the CPU EnTT
reference), all clean under Compute Sanitizer. Later gates (registry breadth,
groups, order, reactivity, persistence) remain pending.

The compatibility reference is EnTT revision
`1dcd7aab6548786f7672ff578639188bd2f0d837` (declaring 4.1.0). The submodule is
unchanged.

## Build on Windows

Use CMake 3.28+, Ninja, Visual Studio 2022's x64 C++ tools, and NVIDIA CUDA 13.2+.
The checked combination is nvcc 13.2.51 with MSVC 19.44.35229, in C++20 mode.
The provided bootstrap script downloads NVIDIA's compiler/runtime/header packages,
verifies archive SHA-256 values from the versioned NVIDIA manifest, and expands
them under ignored `.dev/`. It does not install a driver or change system CUDA.
Original package licenses remain under `.dev/cuda-downloads/`.

```powershell
git submodule update --init external/entt
./tools/bootstrap-cuda.ps1
```

Then use an x64 VS 2022 developer shell. Set the compiler/toolkit explicitly so a
system CUDA installation cannot override it. This machine's GTX 1650 uses
`75-real`; select the architecture for your GPU when building elsewhere.

```powershell
$enttCudaToolkit = (Resolve-Path .dev/cuda-13.2.0).Path
$env:PATH = "$enttCudaToolkit/bin;$env:PATH"
cmake -S . -B build/cuda-13.2 -G Ninja `
    -DCMAKE_BUILD_TYPE=Debug `
    -DCMAKE_CUDA_ARCHITECTURES=75-real `
    -DCMAKE_CUDA_COMPILER:FILEPATH="$enttCudaToolkit/bin/nvcc.exe" `
    -DCUDAToolkit_ROOT:PATH="$enttCudaToolkit"
cmake --build build/cuda-13.2
ctest --test-dir build/cuda-13.2 --output-on-failure
```

If Ninja on `PATH` cannot run, pass `-DCMAKE_MAKE_PROGRAM:FILEPATH=.../ninja.exe`
using the copy bundled under Visual Studio's
`Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/` directory.

`ENTT_CUDA_BUILD_TESTING=OFF` disables local test targets. For acceptance or GPU CI,
configure with `-DENTT_CUDA_REQUIRE_GPU=ON` to make an unusable device a failure.
By default the runtime test reports a CTest skip for an absent device or
insufficient driver; all other CUDA failures fail the test. Keep the selected
toolkit's `bin` on `PATH` when running executables linked to its shared runtime.

Consumers use `add_subdirectory(...)`, link `EnTT::Cuda`, and include
`<entt/entt_cuda.hpp>`. Device callbacks require CUDA translation units. Installation
and exported-package support are deferred.

## Validation and toolchain limits

- The CUDA object target compiles the actual three core emplacement overloads,
  aggregates, native/null handles, handle-like constructors, invalid component
  rejection, transfer signatures, exclusions and callback signatures.
- A second CUDA source instantiates the reused upstream host identity storage,
  type IDs, host context/reference alias and signal facilities. The CPU EnTT
  registry reference is instantiated in a normal C++ test.
- `registry_core.cu` exercises the implemented registry: `emplace -> each -> get`
  on one stream, GPU values surviving growth and swap-and-pop relocation across a
  sparse-page boundary, `assign`/`get` round trips, tolerant removal, tags,
  exclusions, entity-only and empty required pools, both callback forms, destroy
  with version recycling, cross-stream handoff and move construction. It reports
  clean under Compute Sanitizer memcheck (0 errors, 0 leaked allocations).
- `registry_fault.cu` verifies the sticky fault state: validation rejections
  return `cudaErrorInvalidValue` without faulting, a stale-stream submission
  faults the registry, every later status-returning operation reports the saved
  error, `create()` throws `std::logic_error`, and teardown stays safe.
- `registry_parity.cpp` drives 3000 randomized create/destroy/emplace/remove/
  assign steps against a CPU `entt::registry` reference and compares membership
  and downloaded values bit-exactly. It lives in a host translation unit because
  nvcc 13.2 rejects the upstream CPU registry's `basic_any` SBO union.
- `toolchain.cu` keeps the standalone extended-lambda and pinned transfer smoke
  test as a driver/runtime canary.

On this workspace, the CUDA 13.2 Debug build, host test and GPU runtime test all
pass. The GPU test previously skipped under driver 532.09 (CUDA driver API 12.1),
which could not run the CUDA 13 runtime; driver 621.24 (CUDA user-mode 13.5) now
runs the same workspace-local CUDA 13.2 toolkit (nvcc 13.2.51) on the GTX 1650.
This closes phase 0. NVIDIA documents CUDA 13's driver requirements in its
[compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

nvcc 12.4.131 and 12.9.86 both reject `core/hashed_string.hpp:122` in the pinned
EnTT source. nvcc 13.2 parses that header and compiles the reused facilities above,
but instantiating the full CPU registry in a CUDA translation unit hits a separate
`basic_any` SBO-union initialization diagnostic. Keep the CPU reference in host
translation units. The relevant upstream files are preserved; no compiler checks
are bypassed and no compatibility headers shadow the submodule.
