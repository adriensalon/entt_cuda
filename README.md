# entt_cuda

An EnTT-style CUDA registry with host-owned identity/structure and GPU-authoritative
component values. See [AGENTS.md](AGENTS.md) for the roadmap.

**Current status: phase-0 contracts/build foundation.** The core declarations and
emplacement forwarding compile, but registry runtime definitions start in phase 1.
Constructing/linking a `cuda_registry` is not supported yet. Phase 0 remains open
until the selected toolchain's GPU test runs successfully.

The compatibility reference is EnTT revision
`1dcd7aab6548786f7672ff578639188bd2f0d837` (declaring 4.1.0). The submodule is
unchanged. [Contracts](docs/contracts.md) define argument placement, allocation,
failure, ownership and ordering. [API coverage](docs/api-coverage.md) records every
public registry declaration/const overload/deleted operation and later ECS gates.

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
and exported-package support are deferred; there is no usable registry runtime
to package yet.

## Validation and toolchain limits

- The CUDA object target compiles the actual three core emplacement overloads,
  aggregates, native/null handles, handle-like constructors, invalid component
  rejection, transfer signatures, exclusions and callback signatures. It does
  not link unfinished registry operations or validate kernel invocation yet.
- A second CUDA source instantiates the reused upstream host identity storage,
  type IDs, host context/reference alias and signal facilities. The CPU EnTT
  registry reference is instantiated in a normal C++ test.
- The runtime smoke test launches an extended device lambda and checks pinned
  upload/download with two nonblocking streams and native event handoff. It
  does not use an implemented CUDA registry.

On this workspace, the CUDA 13.2 Debug build and host test pass. Its GPU test
skips: driver 532.09 reports CUDA driver API 12.1 and cannot run the CUDA 13
runtime. The same standalone kernel/transfer smoke test compiled with the
installed CUDA 12.4 toolkit ran successfully on the GTX 1650; this does not
validate the chosen CUDA 13.2 runtime or a CUDA registry. A compatible driver
and a passing strict GPU test are required to close phase 0. NVIDIA documents
CUDA 13's driver requirements in its
[compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

nvcc 12.4.131 and 12.9.86 both reject `core/hashed_string.hpp:122` in the pinned
EnTT source. nvcc 13.2 parses that header and compiles the reused facilities above,
but instantiating the full CPU registry in a CUDA translation unit hits a separate
`basic_any` SBO-union initialization diagnostic. Keep the CPU reference in host
translation units. The relevant upstream files are preserved; no compiler checks
are bypassed and no compatibility headers shadow the submodule.
