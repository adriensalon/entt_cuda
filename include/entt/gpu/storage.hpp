#ifndef ENTT_GPU_STORAGE_HPP
#define ENTT_GPU_STORAGE_HPP

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
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
template<typename Backend>
class gpu_retirement {
    struct entry {
        void *ptr;
        typename Backend::event_type event;
    };

    std::vector<entry> retired;

public:
    gpu_retirement() = default;
    gpu_retirement(const gpu_retirement &) = delete;
    gpu_retirement(gpu_retirement &&) noexcept = default;
    gpu_retirement &operator=(const gpu_retirement &) = delete;
    gpu_retirement &operator=(gpu_retirement &&) noexcept = default;

    /*! @brief Retires a device allocation after the stream reaches this point. */
    void retire(void *ptr, typename Backend::stream_type stream) {
        if(ptr == nullptr) {
            return;
        }

        typename Backend::event_type event{};

        if(Backend::event_create(&event, Backend::event_disable_timing) == Backend::success) {
            Backend::event_record(event, stream);
        }

        // A null event keeps the allocation until drain, a conservative fallback.
        retired.push_back(entry{ptr, event});
    }

    /*! @brief Frees retired allocations whose events have completed. */
    void reclaim() noexcept {
        for(auto it = retired.begin(); it != retired.end();) {
            if(it->event != nullptr && Backend::event_query(it->event) == Backend::success) {
                Backend::free(it->ptr);
                Backend::event_destroy(it->event);
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
                Backend::event_synchronize(elem.event);
                Backend::event_destroy(elem.event);
            }

            if(elem.ptr != nullptr) {
                Backend::free(elem.ptr);
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
template<typename Entity, typename Backend>
struct gpu_pool_base {
    virtual ~gpu_pool_base() = default;

    [[nodiscard]] virtual std::size_t size() const noexcept = 0;
    [[nodiscard]] virtual bool empty() const noexcept = 0;
    [[nodiscard]] virtual std::size_t capacity() const noexcept = 0;
    [[nodiscard]] virtual bool contains(const Entity value) const noexcept = 0;

    /*! @brief Returns the type hash of the stored component type. */
    [[nodiscard]] virtual id_type type() const noexcept = 0;

    /*! @brief Enqueues a tolerant swap-and-pop removal; a no-op if absent. */
    virtual typename Backend::error_type enqueue_remove(const Entity value, typename Backend::stream_type stream) = 0;

    /*! @brief Clears every member; host-only, no device submission. */
    virtual void clear() noexcept = 0;

    /*! @brief Retires every live device allocation after the stream completes. */
    virtual void reset(typename Backend::stream_type stream) noexcept = 0;

    /*! @brief Re-points internal bookkeeping after the registry is moved. */
    virtual void rebind_retirement(gpu_retirement<Backend> &retire) noexcept = 0;

    /*! @brief Frees every live device allocation (called after teardown drain). */
    virtual void release() noexcept = 0;

    [[nodiscard]] virtual const Entity *packed_device() const noexcept = 0;
    [[nodiscard]] virtual const Entity *const *pages() const noexcept = 0;
    [[nodiscard]] virtual const Entity *data() const noexcept = 0;
};

/*! @brief Forward iterator over a pool map yielding (id, pool reference) pairs. */
template<typename It, typename Pool>
class gpu_pool_iterator {
    It it;

public:
    using iterator_category = std::forward_iterator_tag;
    using value_type = std::pair<id_type, Pool &>;
    using reference = value_type;
    using pointer = void;
    using difference_type = std::ptrdiff_t;

    gpu_pool_iterator() = default;
    explicit gpu_pool_iterator(It curr) noexcept
        : it{curr} {}

    [[nodiscard]] value_type operator*() const noexcept {
        return {it->first, *it->second};
    }

    gpu_pool_iterator &operator++() noexcept {
        ++it;
        return *this;
    }

    gpu_pool_iterator operator++(int) noexcept {
        gpu_pool_iterator copy = *this;
        ++(*this);
        return copy;
    }

    [[nodiscard]] bool operator==(const gpu_pool_iterator &other) const noexcept {
        return it == other.it;
    }

    [[nodiscard]] bool operator!=(const gpu_pool_iterator &other) const noexcept {
        return it != other.it;
    }
};

/*! @brief A lightweight range over a pool map. */
template<typename It, typename Pool>
class gpu_pool_range {
    It first;
    It last;

public:
    gpu_pool_range(It begin, It end) noexcept
        : first{begin},
          last{end} {}

    [[nodiscard]] gpu_pool_iterator<It, Pool> begin() const noexcept {
        return gpu_pool_iterator<It, Pool>{first};
    }

    [[nodiscard]] gpu_pool_iterator<It, Pool> end() const noexcept {
        return gpu_pool_iterator<It, Pool>{last};
    }
};

} // namespace internal
/*! @endcond */

/**
 * @brief A single component pool with CPU-owned membership and device-owned values.
 * @tparam Type Component type, unqualified and non-const.
 * @tparam Entity Underlying entity identifier.
 */
template<typename Type, typename Entity, typename Backend>
class gpu_storage final: public internal::gpu_pool_base<Entity, Backend> {
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

    internal::gpu_retirement<Backend> *retirement;

    [[nodiscard]] std::size_t index_of(const Entity value) const noexcept {
        return static_cast<std::size_t>(traits_type::to_entity(sparse[static_cast<std::size_t>(traits_type::to_entity(value))]));
    }

    void ensure_host_sparse(const std::size_t raw) {
        if(raw >= sparse.size()) {
            sparse.resize(raw + 1u, internal::gpu_null_entity<Entity>);
        }
    }

    typename Backend::error_type ensure_page(const std::size_t page, typename Backend::stream_type stream) {
        if(d_pages == nullptr) {
            void *directory{};

            if(Backend::malloc(&directory, internal::gpu_sparse_page_count<Entity> * sizeof(Entity *)) != Backend::success) {
                return Backend::out_of_memory;
            }

            d_pages = static_cast<Entity **>(directory);
            pages_host.assign(internal::gpu_sparse_page_count<Entity>, nullptr);
            Backend::memset_async(d_pages, 0, internal::gpu_sparse_page_count<Entity> * sizeof(Entity *), stream);
        }

        if(pages_host[page] == nullptr) {
            void *raw{};

            if(Backend::malloc(&raw, internal::gpu_sparse_page<Entity> * sizeof(Entity)) != Backend::success) {
                return Backend::out_of_memory;
            }

            auto *page_ptr = static_cast<Entity *>(raw);
            pages_host[page] = page_ptr;
            // Every byte set to 0xFF makes each cell the all-ones null identifier.
            Backend::memset_async(page_ptr, 0xFF, internal::gpu_sparse_page<Entity> * sizeof(Entity), stream);
            Backend::memcpy_async(&d_pages[page], &pages_host[page], sizeof(Entity *), Backend::host_to_device, stream);
        }

        return Backend::success;
    }

    typename Backend::error_type grow(const std::size_t required, typename Backend::stream_type stream) {
        if(required <= reserved) {
            return Backend::success;
        }

        std::size_t next = (reserved == 0u) ? 1u : reserved;

        while(next < required) {
            next *= 2u;
        }

        void *raw_packed{};
        void *raw_payload{};

        if(Backend::malloc(&raw_packed, next * sizeof(Entity)) != Backend::success) {
            return Backend::out_of_memory;
        }

        if constexpr(has_value) {
            if(Backend::malloc(&raw_payload, next * sizeof(Type)) != Backend::success) {
                Backend::free(raw_packed);
                return Backend::out_of_memory;
            }
        }

        auto *next_packed = static_cast<Entity *>(raw_packed);
        auto *next_payload = static_cast<Type *>(raw_payload);

        if(count != 0u) {
            Backend::memcpy_async(next_packed, d_packed, count * sizeof(Entity), Backend::device_to_device, stream);

            if constexpr(has_value) {
                Backend::memcpy_async(next_payload, d_payload, count * sizeof(Type), Backend::device_to_device, stream);
            }
        }

        retirement->retire(d_packed, stream);

        if constexpr(has_value) {
            retirement->retire(d_payload, stream);
        }

        d_packed = next_packed;
        d_payload = next_payload;
        reserved = next;

        return Backend::success;
    }

    typename Backend::error_type write_sparse_cell(const Entity value, const Entity cell, typename Backend::stream_type stream) {
        const auto raw = static_cast<std::size_t>(traits_type::to_entity(value));
        const auto page = raw / internal::gpu_sparse_page<Entity>;
        const auto offset = raw % internal::gpu_sparse_page<Entity>;

        if(const auto err = ensure_page(page, stream); err != Backend::success) {
            return err;
        }

        return Backend::memcpy_async(pages_host[page] + offset, &cell, sizeof(Entity), Backend::host_to_device, stream);
    }

public:
    explicit gpu_storage(internal::gpu_retirement<Backend> &other) noexcept
        : retirement{&other} {}

    gpu_storage(const gpu_storage &) = delete;
    gpu_storage(gpu_storage &&) = delete;
    gpu_storage &operator=(const gpu_storage &) = delete;
    gpu_storage &operator=(gpu_storage &&) = delete;

    ~gpu_storage() override {
        release();
    }

    void rebind_retirement(internal::gpu_retirement<Backend> &other) noexcept override {
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
        return raw < sparse.size() && sparse[raw] != internal::gpu_null_entity<Entity> && traits_type::to_version(sparse[raw]) == traits_type::to_version(value);
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
                Backend::memcpy(dst, d_payload, count * sizeof(Type), Backend::device_to_host);
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
    void reset(typename Backend::stream_type stream) noexcept override {
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
    typename Backend::error_type emplace(const Entity value, typename Backend::stream_type stream, Args &&...args) {
        // Construction precedes any logical mutation; exceptions propagate.
        Type staged{std::forward<Args>(args)...};

        const auto pos = count;

        if(const auto err = grow(pos + 1u, stream); err != Backend::success) {
            return err;
        }

        const auto raw = static_cast<std::size_t>(traits_type::to_entity(value));
        const auto page = raw / internal::gpu_sparse_page<Entity>;

        if(const auto err = ensure_page(page, stream); err != Backend::success) {
            return err;
        }

        // Commit host metadata only after every allocation has succeeded.
        const auto cell = internal::gpu_combine<Entity>(pos, value);
        ensure_host_sparse(raw);
        sparse[raw] = cell;
        packed.push_back(value);
        count = pos + 1u;

        typename Backend::error_type err = Backend::success;

        if constexpr(has_value) {
            err = Backend::memcpy_async(d_payload + pos, std::addressof(staged), sizeof(Type), Backend::host_to_device, stream);
        }

        if(err == Backend::success) {
            err = Backend::memcpy_async(d_packed + pos, &value, sizeof(Entity), Backend::host_to_device, stream);
        }

        if(err == Backend::success) {
            err = write_sparse_cell(value, cell, stream);
        }

        return err;
    }

    /*! @brief Enqueues a device-to-host download of the component value. */
    typename Backend::error_type enqueue_get(const Entity value, Type *dst, typename Backend::stream_type stream) const {
        const auto pos = index_of(value);
        return Backend::memcpy_async(dst, d_payload + pos, sizeof(Type), Backend::device_to_host, stream);
    }

    /*! @brief Enqueues a host-to-device upload of the component value. */
    typename Backend::error_type enqueue_assign(const Entity value, const Type *src, typename Backend::stream_type stream) {
        const auto pos = index_of(value);
        return Backend::memcpy_async(d_payload + pos, src, sizeof(Type), Backend::host_to_device, stream);
    }

    /*! @brief Constructs a replacement value on the host and uploads it in place. */
    template<typename... Args>
    typename Backend::error_type replace(const Entity value, typename Backend::stream_type stream, Args &&...args) {
        if constexpr(has_value) {
            Type staged{std::forward<Args>(args)...};
            const auto pos = index_of(value);
            return Backend::memcpy_async(d_payload + pos, std::addressof(staged), sizeof(Type), Backend::host_to_device, stream);
        } else {
            return Backend::success;
        }
    }

    /*! @brief Swaps two dense slots, reordering host and device state together. */
    typename Backend::error_type swap_elements(const std::size_t lhs, const std::size_t rhs, typename Backend::stream_type stream) {
        if(lhs == rhs) {
            return Backend::success;
        }

        const Entity lhs_entity = packed[lhs];
        const Entity rhs_entity = packed[rhs];

        packed[lhs] = rhs_entity;
        packed[rhs] = lhs_entity;
        sparse[static_cast<std::size_t>(traits_type::to_entity(lhs_entity))] = internal::gpu_combine<Entity>(rhs, lhs_entity);
        sparse[static_cast<std::size_t>(traits_type::to_entity(rhs_entity))] = internal::gpu_combine<Entity>(lhs, rhs_entity);

        typename Backend::error_type err = Backend::success;

        if constexpr(has_value) {
            if(d_scratch == nullptr && Backend::malloc(reinterpret_cast<void **>(&d_scratch), sizeof(Type)) != Backend::success) {
                return Backend::out_of_memory;
            }

            err = Backend::memcpy_async(d_scratch, d_payload + lhs, sizeof(Type), Backend::device_to_device, stream);
        }

        if(err == Backend::success) {
            err = Backend::memcpy_async(d_packed + lhs, &rhs_entity, sizeof(Entity), Backend::host_to_device, stream);
        }

        if(err == Backend::success) {
            err = Backend::memcpy_async(d_packed + rhs, &lhs_entity, sizeof(Entity), Backend::host_to_device, stream);
        }

        if constexpr(has_value) {
            if(err == Backend::success) {
                err = Backend::memcpy_async(d_payload + lhs, d_payload + rhs, sizeof(Type), Backend::device_to_device, stream);
            }

            if(err == Backend::success) {
                err = Backend::memcpy_async(d_payload + rhs, d_scratch, sizeof(Type), Backend::device_to_device, stream);
            }
        }

        if(err == Backend::success) {
            err = write_sparse_cell(lhs_entity, internal::gpu_combine<Entity>(rhs, lhs_entity), stream);
        }

        if(err == Backend::success) {
            err = write_sparse_cell(rhs_entity, internal::gpu_combine<Entity>(lhs, rhs_entity), stream);
        }

        return err;
    }

    /*! @brief Reorders the dense slots by a permutation, mirroring the device. */
    typename Backend::error_type reorder(const std::vector<std::size_t> &perm, typename Backend::stream_type stream) {
        const auto n = count;

        std::vector<Type> old_values;

        if constexpr(has_value) {
            old_values.resize(n);

            if(n != 0u) {
                Backend::memcpy(old_values.data(), d_payload, n * sizeof(Type), Backend::device_to_host);
            }
        }

        std::vector<Entity> ordered(n);

        for(std::size_t pos = 0u; pos < n; ++pos) {
            ordered[pos] = packed[perm[pos]];
        }

        packed = std::move(ordered);

        for(std::size_t pos = 0u; pos < n; ++pos) {
            sparse[static_cast<std::size_t>(traits_type::to_entity(packed[pos]))] = internal::gpu_combine<Entity>(pos, packed[pos]);
        }

        typename Backend::error_type err = Backend::success;

        if(n != 0u) {
            err = Backend::memcpy_async(d_packed, packed.data(), n * sizeof(Entity), Backend::host_to_device, stream);

            if constexpr(has_value) {
                std::vector<Type> values(n);

                for(std::size_t pos = 0u; pos < n; ++pos) {
                    values[pos] = old_values[perm[pos]];
                }

                if(err == Backend::success) {
                    err = Backend::memcpy_async(d_payload, values.data(), n * sizeof(Type), Backend::host_to_device, stream);
                }
            }

            for(std::size_t pos = 0u; pos < n && err == Backend::success; ++pos) {
                err = write_sparse_cell(packed[pos], internal::gpu_combine<Entity>(pos, packed[pos]), stream);
            }
        }

        return err;
    }

    /*! @brief Sorts the dense entities by an Entity comparator. */
    template<typename Compare>
    typename Backend::error_type sort_entities(Compare compare, typename Backend::stream_type stream) {
        std::vector<std::size_t> perm(count);
        std::iota(perm.begin(), perm.end(), 0u);
        std::sort(perm.begin(), perm.end(), [&](const std::size_t lhs, const std::size_t rhs) { return compare(packed[lhs], packed[rhs]); });
        return reorder(perm, stream);
    }

    /*! @brief Sorts the dense entities by a value comparator, downloading values. */
    template<typename Compare>
    typename Backend::error_type sort_values(Compare compare, typename Backend::stream_type stream) {
        std::vector<Type> values(count);

        if(count != 0u) {
            Backend::memcpy(values.data(), d_payload, count * sizeof(Type), Backend::device_to_host);
        }

        std::vector<std::size_t> perm(count);
        std::iota(perm.begin(), perm.end(), 0u);
        std::sort(perm.begin(), perm.end(), [&](const std::size_t lhs, const std::size_t rhs) { return compare(values[lhs], values[rhs]); });
        return reorder(perm, stream);
    }

    typename Backend::error_type enqueue_remove(const Entity value, typename Backend::stream_type stream) override {
        if(!contains(value)) {
            return Backend::success;
        }
        const auto pos = index_of(value);
        const auto last_pos = count - 1u;
        const Entity last = packed[last_pos];

        if(pos != last_pos) {
            packed[pos] = last;
            sparse[static_cast<std::size_t>(traits_type::to_entity(last))] = internal::gpu_combine<Entity>(pos, last);
        }

        sparse[static_cast<std::size_t>(traits_type::to_entity(value))] = internal::gpu_null_entity<Entity>;
        packed.pop_back();
        count = last_pos;

        typename Backend::error_type err = Backend::success;

        if(pos != last_pos) {
            if constexpr(has_value) {
                err = Backend::memcpy_async(d_payload + pos, d_payload + last_pos, sizeof(Type), Backend::device_to_device, stream);
            }

            if(err == Backend::success) {
                err = Backend::memcpy_async(d_packed + pos, &last, sizeof(Entity), Backend::host_to_device, stream);
            }

            if(err == Backend::success) {
                err = write_sparse_cell(last, internal::gpu_combine<Entity>(pos, last), stream);
            }
        }

        if(err == Backend::success) {
            err = write_sparse_cell(value, internal::gpu_null_entity<Entity>, stream);
        }

        return err;
    }

    void release() noexcept override {
        if(d_packed != nullptr) {
            Backend::free(d_packed);
            d_packed = nullptr;
        }

        if constexpr(has_value) {
            if(d_payload != nullptr) {
                Backend::free(d_payload);
                d_payload = nullptr;
            }

            if(d_scratch != nullptr) {
                Backend::free(d_scratch);
                d_scratch = nullptr;
            }
        }

        for(auto *page: pages_host) {
            if(page != nullptr) {
                Backend::free(page);
            }
        }

        pages_host.clear();

        if(d_pages != nullptr) {
            Backend::free(d_pages);
            d_pages = nullptr;
        }
    }
};

} // namespace entt

#endif
