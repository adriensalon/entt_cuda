#ifndef ENTT_GPU_ATTRIBUTES_HPP
#define ENTT_GPU_ATTRIBUTES_HPP

// Compiler attribute shims shared by every device backend. ENTT_DEVICE and
// ENTT_GLOBAL annotate device helper and kernel declarations respectively;
// ENTT_DEVICE_CODE guards kernel definitions so host-only translation units
// never see the annotations. These depend only on the device compiler, not on
// a specific backend, so they live in the generic layer rather than a backend
// header. HIP contributes the same definitions under __HIPCC__.
#if defined(__CUDACC__) || defined(__HIPCC__)
#    define ENTT_DEVICE __device__
#    define ENTT_GLOBAL __global__
#    define ENTT_DEVICE_CODE 1
#endif

#endif
