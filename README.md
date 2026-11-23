# entt_cuda

An EnTT-style CUDA registry with host-owned identity/structure and GPU-authoritative
component values.

**Current status: phase 3 (query/storage breadth) complete.** Phase 1 delivered
the asynchronous core and phase 2 the registry breadth. Phase 3 adds named pools
(`storage<T>(id)`), nullable type-erased lookup (`storage(id)`), pool enumeration
(`storage()`), stream-aware `reset(id)` with allocation retirement,
`storage_for_type<T>`, typed view `contains`/`size_hint`/host `each_entity`/`rebind`,
and a runtime view dispatching entity identifiers without runtime device
compilation. Remaining gates (groups, order, reactivity, persistence, allocator
customization) stay pending.

The compatibility reference is EnTT revision
`1dcd7aab6548786f7672ff578639188bd2f0d837` (declaring 4.1.0). The submodule is
unchanged.

## Build on Windows

Use CMake 3.28+, Visual Studio 2022/2026's x64 C++ tools, and an NVIDIA CUDA 13.x
toolkit installed system-wide. No workspace-local bootstrap is required: point
CMake at the installed toolkit and build the Visual Studio solution. This
machine's GTX 1650 uses compute capability 75.

```powershell
git submodule update --init external/entt
cmake -S . -B build -G "Visual Studio 18 2026" -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --config Debug
ctest --test-dir build -C Debug --output-on-failure
```

If more than one toolkit is installed, `find_package(CUDAToolkit)` selects the
latest by default; override with `-DCUDAToolkit_ROOT=...` and make sure the
matching versioned `CUDA_PATH_Vx_y` variable is set for the MSBuild CUDA rules.

`ENTT_CUDA_BUILD_TESTING=OFF` disables local test targets. For acceptance or GPU CI,
configure with `-DENTT_CUDA_REQUIRE_GPU=ON` to make an unusable device a failure.
By default a runtime test reports a CTest skip for an absent device or
insufficient driver; all other CUDA failures fail the test. Keep the selected
toolkit's `bin` on `PATH` when running executables linked to its shared runtime.

Consumers use `add_subdirectory(...)`, link `EnTT::Cuda`, and include
`<entt/entt_cuda.hpp>`. Device callbacks require CUDA translation units. Installation
and exported-package support are deferred.

## Formatting

`include/` and `test/` follow the EnTT style via the root `.clang-format` (which
mirrors `external/entt/.clang-format`). Check or rewrite with the local helper,
which also normalizes clang-format's CUDA extended-lambda artifact so `Type &name`
is preserved:

```powershell
python tools/check-format.py           # report differences
python tools/check-format.py --fix     # rewrite in place
```

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
  and downloaded values bit-exactly.
- `registry_breadth.cpp` exercises the phase-2 surface: hinted/ranged creation,
  versioned/ranged destruction, `all_of`/`any_of`/`orphan`/`current`, const
  access, `try_get` device pointers (single and tuple), `insert`,
  `emplace_or_replace`, `get_or_emplace`, `replace`, multi-type/ranged `remove`,
  strict `erase`/`erase_if`, `clear`, and the host context. `registry_core.cu`
  additionally covers device-callable `patch`.
- `registry_storage.cpp` exercises the phase-3 surface: named pools, type-erased
  lookup, pool enumeration, stream-aware `reset`, typed view `contains`/`size_hint`,
  host entity iteration and rebinding. `registry_core.cu` additionally covers the
  runtime view dispatching entity identifiers.
- `toolchain.cu` keeps the standalone extended-lambda and pinned transfer smoke
  test as a driver/runtime canary.

On this workspace the CUDA 13.4 Debug build and all six tests pass, and the
registry core and breadth tests report clean under Compute Sanitizer memcheck
(0 errors, 0 leaked allocations). The GPU test previously skipped under driver
532.09 (CUDA driver API 12.1); driver 621.24 now runs CUDA 13 runtimes on the
GTX 1650. NVIDIA documents CUDA 13's driver requirements in its
[compatibility guide](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html).

nvcc 12.4 rejects `core/hashed_string.hpp:122` in the pinned EnTT source. nvcc
13.x parses that header, but instantiating the full CPU registry in a CUDA
translation unit hits a separate `basic_any` SBO-union initialization diagnostic,
so the CPU EnTT reference lives in host translation units. The relevant upstream
files are preserved; no compiler checks are bypassed and no compatibility headers
shadow the submodule.
