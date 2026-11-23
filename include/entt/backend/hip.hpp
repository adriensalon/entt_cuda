#ifndef ENTT_BACKEND_HIP_HPP
#define ENTT_BACKEND_HIP_HPP

#include <cstddef>
#include <hip/hip_runtime.h>

namespace entt {

/**
 * @brief HIP backend trait: native handle/error types, error constants and
 * static wrappers over every runtime call the generic core performs.
 *
 * This is the AMD counterpart of cuda_backend. It exposes the same member
 * surface over the hip* runtime, so the generic core is instantiated verbatim.
 * Note the error-constant spelling difference: HIP reports allocation failure
 * as hipErrorOutOfMemory rather than cudaErrorMemoryAllocation.
 */
struct hip_backend {
    using stream_type = hipStream_t;
    using event_type = hipEvent_t;
    using error_type = hipError_t;

    enum class copy_kind {
        host_to_device,
        device_to_host,
        device_to_device
    };

    static constexpr error_type success = hipSuccess;
    static constexpr error_type invalid_value = hipErrorInvalidValue;
    static constexpr error_type out_of_memory = hipErrorOutOfMemory;
    static constexpr unsigned event_disable_timing = hipEventDisableTiming;

    static constexpr copy_kind host_to_device = copy_kind::host_to_device;
    static constexpr copy_kind device_to_host = copy_kind::device_to_host;
    static constexpr copy_kind device_to_device = copy_kind::device_to_device;

    static error_type malloc(void **ptr, const std::size_t size) noexcept {
        return hipMalloc(ptr, size);
    }

    static error_type free(void *ptr) noexcept {
        return hipFree(ptr);
    }

    static error_type memset_async(void *ptr, const int value, const std::size_t count, stream_type stream) noexcept {
        return hipMemsetAsync(ptr, value, count, stream);
    }

    static error_type memcpy(void *dst, const void *src, const std::size_t count, const copy_kind kind) noexcept {
        return hipMemcpy(dst, src, count, to_native(kind));
    }

    static error_type memcpy_async(void *dst, const void *src, const std::size_t count, const copy_kind kind, stream_type stream) noexcept {
        return hipMemcpyAsync(dst, src, count, to_native(kind), stream);
    }

    static error_type device_synchronize() noexcept {
        return hipDeviceSynchronize();
    }

    static error_type get_last_error() noexcept {
        return hipGetLastError();
    }

    static error_type event_create(event_type *event, const unsigned flags) noexcept {
        return hipEventCreateWithFlags(event, flags);
    }

    static error_type event_record(event_type event, stream_type stream) noexcept {
        return hipEventRecord(event, stream);
    }

    static error_type event_query(event_type event) noexcept {
        return hipEventQuery(event);
    }

    static error_type event_synchronize(event_type event) noexcept {
        return hipEventSynchronize(event);
    }

    static error_type event_destroy(event_type event) noexcept {
        return hipEventDestroy(event);
    }

private:
    static hipMemcpyKind to_native(const copy_kind kind) noexcept {
        switch(kind) {
        case copy_kind::host_to_device:
            return hipMemcpyHostToDevice;
        case copy_kind::device_to_host:
            return hipMemcpyDeviceToHost;
        case copy_kind::device_to_device:
            return hipMemcpyDeviceToDevice;
        }

        // The switch is exhaustive over copy_kind; the fallback satisfies
        // compilers that cannot prove it.
        return hipMemcpyDeviceToDevice;
    }
};

} // namespace entt

#endif
