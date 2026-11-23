#ifndef ENTT_BACKEND_CUDA_HPP
#define ENTT_BACKEND_CUDA_HPP

#include <cstddef>
#include <cuda_runtime_api.h>

namespace entt {

/**
 * @brief CUDA backend trait: native handle/error types, error constants and
 * static wrappers over every runtime call the generic core performs.
 *
 * This is the portability seam. A HIP backend provides the same members over
 * the hip* runtime, so the core never spells a native type, call or copy
 * direction. It is consumed only by the implementation headers for now; it
 * becomes the public backend parameter when the core is generalized.
 */
struct cuda_backend {
    using stream_type = cudaStream_t;
    using event_type = cudaEvent_t;
    using error_type = cudaError_t;

    enum class copy_kind {
        host_to_device,
        device_to_host,
        device_to_device
    };

    static constexpr error_type success = cudaSuccess;
    static constexpr error_type invalid_value = cudaErrorInvalidValue;
    static constexpr error_type out_of_memory = cudaErrorMemoryAllocation;
    static constexpr unsigned event_disable_timing = cudaEventDisableTiming;

    static constexpr copy_kind host_to_device = copy_kind::host_to_device;
    static constexpr copy_kind device_to_host = copy_kind::device_to_host;
    static constexpr copy_kind device_to_device = copy_kind::device_to_device;

    static error_type malloc(void **ptr, const std::size_t size) noexcept {
        return cudaMalloc(ptr, size);
    }

    static error_type free(void *ptr) noexcept {
        return cudaFree(ptr);
    }

    static error_type memset_async(void *ptr, const int value, const std::size_t count, stream_type stream) noexcept {
        return cudaMemsetAsync(ptr, value, count, stream);
    }

    static error_type memcpy(void *dst, const void *src, const std::size_t count, const copy_kind kind) noexcept {
        return cudaMemcpy(dst, src, count, to_native(kind));
    }

    static error_type memcpy_async(void *dst, const void *src, const std::size_t count, const copy_kind kind, stream_type stream) noexcept {
        return cudaMemcpyAsync(dst, src, count, to_native(kind), stream);
    }

    static error_type device_synchronize() noexcept {
        return cudaDeviceSynchronize();
    }

    static error_type get_last_error() noexcept {
        return cudaGetLastError();
    }

    static error_type event_create(event_type *event, const unsigned flags) noexcept {
        return cudaEventCreateWithFlags(event, flags);
    }

    static error_type event_record(event_type event, stream_type stream) noexcept {
        return cudaEventRecord(event, stream);
    }

    static error_type event_query(event_type event) noexcept {
        return cudaEventQuery(event);
    }

    static error_type event_synchronize(event_type event) noexcept {
        return cudaEventSynchronize(event);
    }

    static error_type event_destroy(event_type event) noexcept {
        return cudaEventDestroy(event);
    }

private:
    static cudaMemcpyKind to_native(const copy_kind kind) noexcept {
        switch(kind) {
        case copy_kind::host_to_device:
            return cudaMemcpyHostToDevice;
        case copy_kind::device_to_host:
            return cudaMemcpyDeviceToHost;
        case copy_kind::device_to_device:
            return cudaMemcpyDeviceToDevice;
        }

        // The switch is exhaustive over copy_kind; the fallback satisfies
        // compilers that cannot prove it.
        return cudaMemcpyDeviceToDevice;
    }
};

} // namespace entt

#endif
