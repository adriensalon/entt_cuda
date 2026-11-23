#ifndef ENTT_CUDA_STORAGE_HPP
#define ENTT_CUDA_STORAGE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
#include <cuda_runtime_api.h>
#include <entt/core/type_info.hpp>
#include <entt/entity/entity.hpp>
#include "contract.hpp"
#include "device.hpp"

namespace entt {

/*! @cond ENTT_INTERNAL */
namespace internal {

/**
 * @brief Tracks device buffers whose reuse is ordered by a native event.
 *
 * Retired allocations stay alive until the stream on which they were retired
 * reaches the recorded event. The registry reclaims them opportunistically and
 * drains the remainder during teardown.
 */
class cuda_retirement {
    struct entry {
        void *ptr;
        cudaEvent_t event;
    };

    std::vector<entry> retired;

public:
    cuda_retirement() = default;
    cuda_retirement(const cuda_retirement &) = delete;
    cuda_retirement(cuda_retirement &&) noexcept = default;
    cuda_retirement &operator=(const cuda_retirement &) = delete;
    cuda_retirement &operator=(cuda_retirement &&) noexcept = default;

    /*! @brief Retires a device allocation after the stream reaches this point. */
    void retire(void *ptr, cudaStream_t stream) {
        if(ptr == nullptr) {
            return;
        }

        cudaEvent_t event{};

        if(cudaEventCreateWithFlags(&event, cudaEventDisableTiming) == cudaSuccess) {
            cudaEventRecord(event, stream);
        }

        // A null event keeps the allocation until drain, a conservative fallback.
        retired.push_back(entry{ptr, event});
    }

    /*! @brief Frees retired allocations whose events have completed. */
    void reclaim() noexcept {
        for(auto it = retired.begin(); it != retired.end();) {
            if(it->event != nullptr && cudaEventQuery(it->event) == cudaSuccess) {
                cudaFree(it->ptr);
                cudaEventDestroy(it->event);
                it = retired.erase(it);
            } else {
                ++it;
            }
        }
    }

    /*! @brief Synchronizes every event and frees every retired allocation. */
    void drain() noexcept {
        for(auto &&elem: retired) {
            if(elem.event != nullptr) {
                cudaEventSynchronize(elem.event);
                cudaEventDestroy(elem.event);
            }

            if(elem.ptr != nullptr) {
                cudaFree(elem.ptr);
            }
        }

        retired.clear();
    }
};

/**
 * @brief Type-erased view of a component pool for the registry.
 *
 * The registry owns pools through this interface so that `destroy` can remove an
 * Entity from every pool without knowing each component type.
 * @tparam Entity Underlying entity identifier.
 */
template<typename Entity = entity>
struct cuda_pool_base {
    virtual ~cuda_pool_base() = default;

    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] virtual bool empty() const noexcept = 0;
    [[nodiscard]] virtual std::size_t capacity() const noexcept = 0;
    [[nodiscard]] virtual bool contains(const Entity value) const noexcept = 0;

    /*! @brief Returns the type hash of the stored component type. */
    [[nodiscard]] virtual id_type type() const noexcept = 0;

    /*! @brief Enqueues a tolerant swap-and-pop removal; a no-op if absent. */
    virtual cudaError_t enqueue_remove(const Entity value, cudaStream_t stream) = 0;

    /*! @brief Clears every member; host-only, no device submission. */
    virtual void clear() noexcept = 0;

    /*! @brief Retires every live device allocation after the stream completes. */
    virtual void reset(cudaStream_t stream) noexcept = 0;

    /*! @brief Re-points internal bookkeeping after the registry is moved. */
    virtual void rebind_retirement(cuda_retirement &retire) noexcept = 0;

    /*! @brief Frees every live device allocation (called after teardown drain). */
    virtual void release() noexcept = 0;

    [[nodiscard]] virtual const Entity *packed_device() const noexcept = 0;
    [[nodiscard]] virtual const Entity *const *pages() const noexcept = 0;
    [[nodiscard]] virtual const Entity *data() const noexcept = 0;
};

/*! @brief Forward iterator over a pool map yielding (id, pool reference) pairs. */
template<typename It, typename Pool>
class cuda_pool_iterator {
    It it;

public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::pair<id_type, Pool &>;
    using reference = value_type;
    using pointer = void;
    using difference_type = std::ptrdiff_t;

    cuda_pool_iterator() = default;
    explicit cuda_pool_iterator(It curr) noexcept
        : it{curr} {}

    [[nodiscard]] value_type operator*() const noexcept {
        return {it->first, *it->second};
    }

    cuda_pool_iterator &operator++() noexcept {
        ++it;
        return *this;
    }

    cuda_pool_iterator operator++(int) noexcept {
        cuda_pool_iterator copy = *this;
        ++(*this);
        return copy;
    }

    [[nodiscard]] bool operator==(const cuda_pool_iterator &other) const noexcept {
        return it == other.it;
    }

    [[nodiscard]] bool operator!=(const cuda_pool_iterator &other) const noexcept {
        return it != other.it;
    }
};

/*! @brief A lightweight range over a pool map. */
template<typename It, typename Pool>
class cuda_pool_range {
    It first;
    It last;

public:
    cuda_pool_range(It begin, It end) noexcept
        : first{begin},
          last{end} {}

    [[nodiscard]] cuda_pool_iterator<It, Pool> begin() const noexcept {
        return cuda_pool_iterator<It, Pool>{first};
    }

    [[nodiscard]] cuda_pool_iterator<It, Pool> end() const noexcept {
        return cuda_pool_iterator<It, Pool>{last};
    }
};

} // namespace internal
/*! @endcond */

/**
 * @brief A single component pool with CPU-owned membership and device-owned values.
 * @tparam Type Component type, unqualified and non-const.
 * @tparam Entity Underlying entity identifier.
 */
template<typename Type, typename Entity = entity>
class cuda_storage final: public internal::cuda_pool_base<Entity> {
    using traits_type = entt_traits<Entity>;
    static constexpr bool has_value = !std::is_empty_v<Type>;

    std::vector<Entity> packed;       // host dense entities, in dense order
    std::vector<Entity> sparse;       // host flat sparse: sparse[to_entity(e)] = combine(pos, e) | null
    Entity *d_packed{};               // device dense entities
    Type *d_payload{};                // device dense component values (null for tags)
    Type *d_scratch{};                // device scratch for value swaps
    Entity **d_pages{};               // device sparse page directory
    std::vector<Entity *> pages_host; // host mirror of d_pages (device page pointers)
    std::size_t count{};
    std::size_t reserved{};

    internal::cuda_retirement *retirement;

    [[nodiscard]] std::size_t index_of(const Entity value) const noexcept {
        return static_cast<std::size_t>(traits_type::to_entity(sparse[static_cast<std::size_t>(traits_type::to_entity(value))]));
    }

    void ensure_host_sparse(const std::size_t raw) {
        if(raw >= sparse.size()) {
            sparse.resize(raw + 1u, internal::cuda_null_entity<Entity>);
        }
    }

    cudaError_t ensure_page(const std::size_t page, cudaStream_t stream) {
        if(d_pages == nullptr) {
            void *directory{};

            if(cudaMalloc(&directory, internal::cuda_sparse_page_count<Entity> * sizeof(Entity *)) != cudaSuccess) {
                return cudaErrorMemoryAllocation;
            }

            d_pages = static_cast<Entity **>(directory);
            pages_host.assign(internal::cuda_sparse_page_count<Entity>, nullptr);
            cudaMemsetAsync(d_pages, 0, internal::cuda_sparse_page_count<Entity> * sizeof(Entity *), stream);
        }

        if(pages_host[page] == nullptr) {
            void *raw{};

            if(cudaMalloc(&raw, internal::cuda_sparse_page<Entity> * sizeof(Entity)) != cudaSuccess) {
                return cudaErrorMemoryAllocation;
            }

            auto *page_ptr = static_cast<Entity *>(raw);
            pages_host[page] = page_ptr;
            // Every byte set to 0xFF makes each cell the all-ones null identifier.
            cudaMemsetAsync(page_ptr, 0xFF, internal::cuda_sparse_page<Entity> * sizeof(Entity), stream);
            cudaMemcpyAsync(&d_pages[page], &pages_host[page], sizeof(Entity *), cudaMemcpyHostToDevice, stream);
        }

        return cudaSuccess;
    }

    cudaError_t grow(const std::size_t required, cudaStream_t stream) {
        if(required <= reserved) {
            return cudaSuccess;
        }

        std::size_t next = (reserved == 0u) ? 1u : reserved;

        while(next < required) {
            next *= 2u;
        }

        void *raw_packed{};
        void *raw_payload{};

        if(cudaMalloc(&raw_packed, next * sizeof(Entity)) != cudaSuccess) {
            return cudaErrorMemoryAllocation;
        }

        if constexpr(has_value) {
            if(cudaMalloc(&raw_payload, next * sizeof(Type)) != cudaSuccess) {
                cudaFree(raw_packed);
                return cudaErrorMemoryAllocation;
            }
        }

        auto *next_packed = static_cast<Entity *>(raw_packed);
        auto *next_payload = static_cast<Type *>(raw_payload);

        if(count != 0u) {
            cudaMemcpyAsync(next_packed, d_packed, count * sizeof(Entity), cudaMemcpyDeviceToDevice, stream);

            if constexpr(has_value) {
                cudaMemcpyAsync(next_payload, d_payload, count * sizeof(Type), cudaMemcpyDeviceToDevice, stream);
            }
        }

        retirement->retire(d_packed, stream);

        if constexpr(has_value) {
            retirement->retire(d_payload, stream);
        }

        d_packed = next_packed;
        d_payload = next_payload;
        reserved = next;

        return cudaSuccess;
    }

    cudaError_t write_sparse_cell(const Entity value, const Entity cell, cudaStream_t stream) {
        const auto raw = static_cast<std::size_t>(traits_type::to_entity(value));
        const auto page = raw / internal::cuda_sparse_page<Entity>;
        const auto offset = raw % internal::cuda_sparse_page<Entity>;

        if(const auto err = ensure_page(page, stream); err != cudaSuccess) {
            return err;
        }

        return cudaMemcpyAsync(pages_host[page] + offset, &cell, sizeof(Entity), cudaMemcpyHostToDevice, stream);
    }

public:
    explicit cuda_storage(internal::cuda_retirement &other) noexcept
        : retirement{&other} {}

    cuda_storage(const cuda_storage &) = delete;
    cuda_storage(cuda_storage &&) = delete;
    cuda_storage &operator=(const cuda_storage &) = delete;
    cuda_storage &operator=(cuda_storage &&) = delete;

    ~cuda_storage() override {
        release();
    }

    void rebind_retirement(internal::cuda_retirement &other) noexcept override {
        retirement = &other;
    }

    [[nodiscard]] id_type type() const noexcept override {
        return type_hash<Type>::value();
    }

    [[nodiscard]] std::size_t size() const noexcept override {
        return count;
    }

    [[nodiscard]] bool empty() const noexcept override {
        return count == 0u;
    }

    [[nodiscard]] std::size_t capacity() const noexcept override {
        return reserved;
    }

    [[nodiscard]] bool contains(const Entity value) const noexcept override {
        const auto raw = static_cast<std::size_t>(traits_type::to_entity(value));
        return raw < sparse.size() && sparse[raw] != internal::cuda_null_entity<Entity> && traits_type::to_version(sparse[raw]) == traits_type::to_version(value);
    }

    [[nodiscard]] const Entity *packed_device() const noexcept override {
        return d_packed;
    }

    [[nodiscard]] const Entity *data() const noexcept override {
        return packed.data();
    }

    [[nodiscard]] const Entity *const *pages() const noexcept override {
        return d_pages;
    }

    /*! @brief Returns the typed device payload pointer, preserving constness. */
    template<typename Other>
    [[nodiscard]] Other *payload_device() noexcept {
        if constexpr(has_value) {
            return static_cast<Other *>(d_payload);
        } else {
            return nullptr;
        }
    }

    /*! @brief Returns the typed device pointer for an Entity, or null if absent. */
    template<typename Other>
    [[nodiscard]] Other *try_get(const Entity value) const noexcept {
        if constexpr(has_value) {
            return contains(value) ? static_cast<Other *>(d_payload + index_of(value)) : nullptr;
        } else {
            return nullptr;
        }
    }

    /*! @brief Returns the dense position of an Entity (valid members only). */
    [[nodiscard]] std::size_t index(const Entity value) const noexcept {
        return index_of(value);
    }

    /*! @brief Downloads every component value into host memory (synchronous). */
    void download(Type *dst) const {
        if constexpr(has_value) {
            if(count != 0u) {
                cudaMemcpy(dst, d_payload, count * sizeof(Type), cudaMemcpyDeviceToHost);
            }
        }
    }

    /*! @brief Clears every member; host-only, no device submission. */
    void clear() noexcept override {
        packed.clear();
        sparse.clear();
        count = 0u;
    }

    /*! @brief Retires every live device allocation after the stream completes. */
    void reset(cudaStream_t stream) noexcept override {
        retirement->retire(d_packed, stream);

        if constexpr(has_value) {
            retirement->retire(d_payload, stream);
            retirement->retire(d_scratch, stream);
        }

        for(auto *page: pages_host) {
            retirement->retire(page, stream);
        }

        retirement->retire(d_pages, stream);

        d_packed = nullptr;
        d_payload = nullptr;
        d_scratch = nullptr;
        pages_host.clear();
        d_pages = nullptr;
        packed.clear();
        sparse.clear();
        count = 0u;
    }

    /*! @brief Constructs the component on the host and enqueues its upload. */
    template<typename... Args>
    cudaError_t emplace(const Entity value, cudaStream_t stream, Args &&...args) {
        // Construction precedes any logical mutation; exceptions propagate.
        Type staged{std::forward<Args>(args)...};

        const auto pos = count;

        if(const auto err = grow(pos + 1u, stream); err != cudaSuccess) {
            return err;
        }

        const auto raw = static_cast<std::size_t>(traits_type::to_entity(value));
        const auto page = raw / internal::cuda_sparse_page<Entity>;

        if(const auto err = ensure_page(page, stream); err != cudaSuccess) {
            return err;
        }

        // Commit host metadata only after every allocation has succeeded.
        const auto cell = internal::cuda_combine<Entity>(pos, value);
        ensure_host_sparse(raw);
        sparse[raw] = cell;
        packed.push_back(value);
        count = pos + 1u;

        cudaError_t err = cudaSuccess;

        if constexpr(has_value) {
            err = cudaMemcpyAsync(d_payload + pos, std::addressof(staged), sizeof(Type), cudaMemcpyHostToDevice, stream);
        }

        if(err == cudaSuccess) {
            err = cudaMemcpyAsync(d_packed + pos, &value, sizeof(Entity), cudaMemcpyHostToDevice, stream);
        }

        if(err == cudaSuccess) {
            err = write_sparse_cell(value, cell, stream);
        }

        return err;
    }

    /*! @brief Enqueues a device-to-host download of the component value. */
    cudaError_t enqueue_get(const Entity value, Type *dst, cudaStream_t stream) const {
        const auto pos = index_of(value);
        return cudaMemcpyAsync(dst, d_payload + pos, sizeof(Type), cudaMemcpyDeviceToHost, stream);
    }

    /*! @brief Enqueues a host-to-device upload of the component value. */
    cudaError_t enqueue_assign(const Entity value, const Type *src, cudaStream_t stream) {
        const auto pos = index_of(value);
        return cudaMemcpyAsync(d_payload + pos, src, sizeof(Type), cudaMemcpyHostToDevice, stream);
    }

    /*! @brief Constructs a replacement value on the host and uploads it in place. */
    template<typename... Args>
    cudaError_t replace(const Entity value, cudaStream_t stream, Args &&...args) {
        if constexpr(has_value) {
            Type staged{std::forward<Args>(args)...};
            const auto pos = index_of(value);
            return cudaMemcpyAsync(d_payload + pos, std::addressof(staged), sizeof(Type), cudaMemcpyHostToDevice, stream);
        } else {
            return cudaSuccess;
        }
    }

    /*! @brief Swaps two dense slots, reordering host and device state together. */
    cudaError_t swap_elements(const std::size_t lhs, const std::size_t rhs, cudaStream_t stream) {
        if(lhs == rhs) {
            return cudaSuccess;
        }

        const Entity lhs_entity = packed[lhs];
        const Entity rhs_entity = packed[rhs];

        packed[lhs] = rhs_entity;
        packed[rhs] = lhs_entity;
        sparse[static_cast<std::size_t>(traits_type::to_entity(lhs_entity))] = internal::cuda_combine<Entity>(rhs, lhs_entity);
        sparse[static_cast<std::size_t>(traits_type::to_entity(rhs_entity))] = internal::cuda_combine<Entity>(lhs, rhs_entity);

        cudaError_t err = cudaSuccess;

        if constexpr(has_value) {
            if(d_scratch == nullptr && cudaMalloc(reinterpret_cast<void **>(&d_scratch), sizeof(Type)) != cudaSuccess) {
                return cudaErrorMemoryAllocation;
            }

            err = cudaMemcpyAsync(d_scratch, d_payload + lhs, sizeof(Type), cudaMemcpyDeviceToDevice, stream);
        }

        if(err == cudaSuccess) {
            err = cudaMemcpyAsync(d_packed + lhs, &rhs_entity, sizeof(Entity), cudaMemcpyHostToDevice, stream);
        }

        if(err == cudaSuccess) {
            err = cudaMemcpyAsync(d_packed + rhs, &lhs_entity, sizeof(Entity), cudaMemcpyHostToDevice, stream);
        }

        if constexpr(has_value) {
            if(err == cudaSuccess) {
                err = cudaMemcpyAsync(d_payload + lhs, d_payload + rhs, sizeof(Type), cudaMemcpyDeviceToDevice, stream);
            }

            if(err == cudaSuccess) {
                err = cudaMemcpyAsync(d_payload + rhs, d_scratch, sizeof(Type), cudaMemcpyDeviceToDevice, stream);
            }
        }

        if(err == cudaSuccess) {
            err = write_sparse_cell(lhs_entity, internal::cuda_combine<Entity>(rhs, lhs_entity), stream);
        }

        if(err == cudaSuccess) {
            err = write_sparse_cell(rhs_entity, internal::cuda_combine<Entity>(lhs, rhs_entity), stream);
        }

        return err;
    }

    /*! @brief Reorders the dense slots by a permutation, mirroring the device. */
    cudaError_t reorder(const std::vector<std::size_t> &perm, cudaStream_t stream) {
        const auto n = count;

        std::vector<Type> old_values;

        if constexpr(has_value) {
            old_values.resize(n);

            if(n != 0u) {
                cudaMemcpy(old_values.data(), d_payload, n * sizeof(Type), cudaMemcpyDeviceToHost);
            }
        }

        std::vector<Entity> ordered(n);

        for(std::size_t pos = 0u; pos < n; ++pos) {
            ordered[pos] = packed[perm[pos]];
        }

        packed = std::move(ordered);

        for(std::size_t pos = 0u; pos < n; ++pos) {
            sparse[static_cast<std::size_t>(traits_type::to_entity(packed[pos]))] = internal::cuda_combine<Entity>(pos, packed[pos]);
        }

        cudaError_t err = cudaSuccess;

        if(n != 0u) {
            err = cudaMemcpyAsync(d_packed, packed.data(), n * sizeof(Entity), cudaMemcpyHostToDevice, stream);

            if constexpr(has_value) {
                std::vector<Type> values(n);

                for(std::size_t pos = 0u; pos < n; ++pos) {
                    values[pos] = old_values[perm[pos]];
                }

                if(err == cudaSuccess) {
                    err = cudaMemcpyAsync(d_payload, values.data(), n * sizeof(Type), cudaMemcpyHostToDevice, stream);
                }
            }

            for(std::size_t pos = 0u; pos < n && err == cudaSuccess; ++pos) {
                err = write_sparse_cell(packed[pos], internal::cuda_combine<Entity>(pos, packed[pos]), stream);
            }
        }

        return err;
    }

    /*! @brief Sorts the dense entities by an Entity comparator. */
    template<typename Compare>
    cudaError_t sort_entities(Compare compare, cudaStream_t stream) {
        std::vector<std::size_t> perm(count);
        std::iota(perm.begin(), perm.end(), 0u);
        std::sort(perm.begin(), perm.end(), [&](const std::size_t lhs, const std::size_t rhs) { return compare(packed[lhs], packed[rhs]); });
        return reorder(perm, stream);
    }

    /*! @brief Sorts the dense entities by a value comparator, downloading values. */
    template<typename Compare>
    cudaError_t sort_values(Compare compare, cudaStream_t stream) {
        std::vector<Type> values(count);

        if(count != 0u) {
            cudaMemcpy(values.data(), d_payload, count * sizeof(Type), cudaMemcpyDeviceToHost);
        }

        std::vector<std::size_t> perm(count);
        std::iota(perm.begin(), perm.end(), 0u);
        std::sort(perm.begin(), perm.end(), [&](const std::size_t lhs, const std::size_t rhs) { return compare(values[lhs], values[rhs]); });
        return reorder(perm, stream);
    }

    cudaError_t enqueue_remove(const Entity value, cudaStream_t stream) override {
        if(!contains(value)) {
            return cudaSuccess;
        }
        const auto pos = index_of(value);
        const auto last_pos = count - 1u;
        const Entity last = packed[last_pos];

        if(pos != last_pos) {
            packed[pos] = last;
            sparse[static_cast<std::size_t>(traits_type::to_entity(last))] = internal::cuda_combine<Entity>(pos, last);
        }

        sparse[static_cast<std::size_t>(traits_type::to_entity(value))] = internal::cuda_null_entity<Entity>;
        packed.pop_back();
        count = last_pos;

        cudaError_t err = cudaSuccess;

        if(pos != last_pos) {
            if constexpr(has_value) {
                err = cudaMemcpyAsync(d_payload + pos, d_payload + last_pos, sizeof(Type), cudaMemcpyDeviceToDevice, stream);
            }

            if(err == cudaSuccess) {
                err = cudaMemcpyAsync(d_packed + pos, &last, sizeof(Entity), cudaMemcpyHostToDevice, stream);
            }

            if(err == cudaSuccess) {
                err = write_sparse_cell(last, internal::cuda_combine<Entity>(pos, last), stream);
            }
        }

        if(err == cudaSuccess) {
            err = write_sparse_cell(value, internal::cuda_null_entity<Entity>, stream);
        }

        return err;
    }

    void release() noexcept override {
        if(d_packed != nullptr) {
            cudaFree(d_packed);
            d_packed = nullptr;
        }

        if constexpr(has_value) {
            if(d_payload != nullptr) {
                cudaFree(d_payload);
                d_payload = nullptr;
            }

            if(d_scratch != nullptr) {
                cudaFree(d_scratch);
                d_scratch = nullptr;
            }
        }

        for(auto *page: pages_host) {
            if(page != nullptr) {
                cudaFree(page);
            }
        }

        pages_host.clear();

        if(d_pages != nullptr) {
            cudaFree(d_pages);
            d_pages = nullptr;
        }
    }
};

} // namespace entt

#endif
