#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>
#include <entt/entt_cuda.hpp>

namespace {

struct position {
    float x, y, z;
};
struct velocity {
    float x, y, z;
};
struct tag {};

bool check(const cudaError_t err, const char *what) {
    if(err != cudaSuccess) {
        std::fprintf(stderr, "%s: status %d (%s)\n", what, static_cast<int>(err), cudaGetErrorString(err));
        return false;
    }

    return true;
}

std::size_t construct_count{};
std::size_t update_count{};
std::size_t destroy_count{};
bool destroy_before_removal = true;
cudaStream_t last_stream{};

void on_construct(entt::cuda_registry &, entt::entity, cudaStream_t stream) {
    ++construct_count;
    last_stream = stream;
}

void on_update(entt::cuda_registry &, entt::entity, cudaStream_t stream) {
    ++update_count;
    last_stream = stream;
}

void on_destroy(entt::cuda_registry &reg, entt::entity entity, cudaStream_t) {
    ++destroy_count;
    destroy_before_removal = destroy_before_removal && reg.all_of<position>(entity);
}

/*! @brief A trivial in-memory archive: write on const args, read on mutable ones. */
struct byte_archive {
    std::vector<char> data;
    std::size_t cursor{};

    template<typename Type>
    void operator()(const Type &value) {
        const auto *bytes = reinterpret_cast<const char *>(&value);
        data.insert(data.end(), bytes, bytes + sizeof(Type));
    }

    template<typename Type>
    void operator()(Type &value) {
        std::memcpy(&value, data.data() + cursor, sizeof(Type));
        cursor += sizeof(Type);
    }
};

} // namespace

int main() {
    int devices{};
    const auto available = cudaGetDeviceCount(&devices);

    if(available == cudaErrorNoDevice || available == cudaErrorInsufficientDriver || (available == cudaSuccess && devices == 0)) {
        std::puts("No usable CUDA device; skipping.");
        return 77;
    }

    if(available != cudaSuccess) {
        return 1;
    }

    cudaStream_t stream{};

    if(!check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create")) {
        return 1;
    }

    entt::cuda_registry registry;

    registry.on_construct<position>().connect<&on_construct>();
    registry.on_update<position>().connect<&on_update>();
    registry.on_destroy<position>().connect<&on_destroy>();

    const auto entity = registry.create();

    // construct fires after emplace, carrying the triggering stream.
    if(!check(registry.emplace<position>(entity, stream, position{1.f, 2.f, 3.f}), "emplace")
       || construct_count != 1u || last_stream != stream) {
        std::fprintf(stderr, "construct signal mismatch\n");
        return 1;
    }

    // update fires after assign and replace.
    position uploaded{4.f, 5.f, 6.f};

    if(!check(registry.assign<position>(entity, &uploaded, stream), "assign")
       || !check(registry.replace<position>(entity, stream, position{7.f, 8.f, 9.f}), "replace")
       || update_count != 2u || last_stream != stream) {
        std::fprintf(stderr, "update signal mismatch\n");
        return 1;
    }

    // destroy fires before removal.
    if(!check(registry.remove<position>(entity, stream), "remove")
       || destroy_count != 1u || !destroy_before_removal || registry.all_of<position>(entity)) {
        std::fprintf(stderr, "destroy signal mismatch\n");
        return 1;
    }

    check(cudaStreamSynchronize(stream), "sync signals");

    // Snapshot round-trip.
    {
        entt::cuda_registry source;
        std::vector<entt::entity> entities;

        for(std::size_t i = 0u; i < 8u; ++i) {
            const auto e = source.create();
            entities.push_back(e);

            if(!check(source.emplace<position>(e, position{static_cast<float>(i), static_cast<float>(i + 1u), 0.f}), "snapshot emplace")
               || !check(source.emplace<velocity>(e, velocity{1.f, 1.f, 1.f}), "snapshot velocity")) {
                return 1;
            }
        }

        for(std::size_t i = 0u; i < 3u; ++i) {
            if(!check(source.emplace<tag>(entities[i]), "snapshot tag")) {
                return 1;
            }
        }

        check(cudaStreamSynchronize(nullptr), "sync snapshot source");

        byte_archive archive;
        entt::cuda_snapshot{source}.get<entt::entity>(archive).get<position>(archive).get<velocity>(archive).get<tag>(archive);

        entt::cuda_registry target;
        entt::cuda_snapshot_loader{target}.get<entt::entity>(archive).get<position>(archive).get<velocity>(archive).get<tag>(archive);

        check(cudaStreamSynchronize(nullptr), "sync snapshot target");

        position *pinned{};

        if(!check(cudaMallocHost(reinterpret_cast<void **>(&pinned), sizeof(position)), "pinned")) {
            return 1;
        }

        for(std::size_t i = 0u; i < 8u; ++i) {
            const auto e = entities[i];

            if(!target.valid(e) || !target.all_of<position, velocity>(e) || target.all_of<tag>(e) != (i < 3u)) {
                std::fprintf(stderr, "snapshot membership mismatch at %zu\n", i);
                return 1;
            }

            if(!check(target.get<position>(e, pinned), "snapshot get")
               || !check(cudaStreamSynchronize(nullptr), "sync snapshot get")) {
                return 1;
            }

            if(pinned->x != static_cast<float>(i) || pinned->y != static_cast<float>(i + 1u)) {
                std::fprintf(stderr, "snapshot value mismatch at %zu\n", i);
                return 1;
            }
        }

        check(cudaFreeHost(pinned), "pinned free");
    }

    std::puts("registry reactivity tests passed.");
    cudaStreamDestroy(stream);
    return 0;
}
