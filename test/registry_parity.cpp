#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <entt/entity/registry.hpp>
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

} // namespace

// The CPU EnTT reference must live in a host translation unit: nvcc 13.2 rejects
// upstream basic_any's nonzero-SBO union when the full registry binds pools.
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
    cudaEvent_t done{};

    if(!check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "stream create")
       || !check(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "event create")) {
        return 1;
    }

    position *pinned{};
    velocity *pinned_vel{};

    if(!check(cudaMallocHost(reinterpret_cast<void **>(&pinned), sizeof(position)), "pinned position")
       || !check(cudaMallocHost(reinterpret_cast<void **>(&pinned_vel), sizeof(velocity)), "pinned velocity")) {
        return 1;
    }

    entt::registry reference;
    entt::cuda_registry device;
    std::mt19937 rng{0xC0FFEEu};

    std::vector<entt::entity> entities;

    const auto verify = [&]() -> bool {
        for(const auto entity: entities) {
            if(reference.all_of<position>(entity) != device.storage<position>().contains(entity)
               || reference.all_of<velocity>(entity) != device.storage<velocity>().contains(entity)
               || reference.all_of<tag>(entity) != device.storage<tag>().contains(entity)) {
                std::fprintf(stderr, "membership mismatch\n");
                return false;
            }

            if(reference.all_of<position>(entity)) {
                if(!check(device.get<position>(entity, pinned, stream, done), "get position")
                   || !check(cudaEventSynchronize(done), "sync position")) {
                    return false;
                }

                const auto expected = reference.get<position>(entity);

                if(pinned->x != expected.x || pinned->y != expected.y || pinned->z != expected.z) {
                    std::fprintf(stderr, "position value mismatch\n");
                    return false;
                }
            }

            if(reference.all_of<velocity>(entity)) {
                if(!check(device.get<velocity>(entity, pinned_vel, stream, done), "get velocity")
                   || !check(cudaEventSynchronize(done), "sync velocity")) {
                    return false;
                }

                const auto expected = reference.get<velocity>(entity);

                if(pinned_vel->x != expected.x || pinned_vel->y != expected.y || pinned_vel->z != expected.z) {
                    std::fprintf(stderr, "velocity value mismatch\n");
                    return false;
                }
            }
        }

        return true;
    };

    constexpr std::size_t steps = 3000u;

    for(std::size_t step = 0u; step < steps; ++step) {
        const auto roll = rng() % 100u;

        if(roll < 35u) {
            // create: both registries must yield identical identifiers
            const auto lhs = device.create();
            const auto rhs = reference.create();

            if(lhs != rhs) {
                std::fprintf(stderr, "create mismatch\n");
                return 1;
            }

            entities.push_back(lhs);
        } else if(roll < 45u && !entities.empty()) {
            // destroy a random entity
            const auto idx = rng() % entities.size();
            const auto entity = entities[idx];

            if(!check(device.destroy(entity, stream, done), "destroy")) {
                return 1;
            }

            reference.destroy(entity);
            entities[idx] = entities.back();
            entities.pop_back();
        } else if(roll < 65u && !entities.empty()) {
            // emplace position
            const auto entity = entities[rng() % entities.size()];

            if(!reference.all_of<position>(entity)) {
                const position value{static_cast<float>(rng() % 1000), static_cast<float>(rng() % 1000), static_cast<float>(rng() % 1000)};

                if(!check(device.emplace<position>(entity, stream, value), "emplace position")) {
                    return 1;
                }

                reference.emplace<position>(entity, value);
            }
        } else if(roll < 75u && !entities.empty()) {
            // remove position (tolerant)
            const auto entity = entities[rng() % entities.size()];

            if(!check(device.remove<position>(entity, stream, done), "remove position")) {
                return 1;
            }

            reference.remove<position>(entity);
        } else if(roll < 85u && !entities.empty()) {
            // emplace velocity
            const auto entity = entities[rng() % entities.size()];

            if(!reference.all_of<velocity>(entity)) {
                const velocity value{static_cast<float>(rng() % 1000), static_cast<float>(rng() % 1000), static_cast<float>(rng() % 1000)};

                if(!check(device.emplace<velocity>(entity, stream, value), "emplace velocity")) {
                    return 1;
                }

                reference.emplace<velocity>(entity, value);
            }
        } else if(roll < 92u && !entities.empty()) {
            // emplace/remove tag
            const auto entity = entities[rng() % entities.size()];

            if(reference.all_of<tag>(entity)) {
                if(!check(device.remove<tag>(entity, stream, done), "remove tag")) {
                    return 1;
                }

                reference.remove<tag>(entity);
            } else {
                if(!check(device.emplace<tag>(entity, stream), "emplace tag")) {
                    return 1;
                }

                reference.emplace<tag>(entity);
            }
        } else if(!entities.empty()) {
            // assign a fresh position value to an entity that owns one
            const auto entity = entities[rng() % entities.size()];

            if(reference.all_of<position>(entity)) {
                const position value{static_cast<float>(rng() % 1000), static_cast<float>(rng() % 1000), static_cast<float>(rng() % 1000)};

                if(!check(device.assign<position>(entity, &value, stream, done), "assign position")) {
                    return 1;
                }

                reference.replace<position>(entity, value);
            }
        }

        if((step + 1u) % 250u == 0u) {
            if(!check(cudaStreamSynchronize(stream), "periodic sync")) {
                return 1;
            }

            if(!verify()) {
                return 1;
            }
        }
    }

    if(!check(cudaStreamSynchronize(stream), "final sync")) {
        return 1;
    }

    if(!verify()) {
        return 1;
    }

    std::printf("randomized parity passed over %zu live entities after %zu steps.\n", entities.size(), steps);
    check(cudaFreeHost(pinned), "pinned position free");
    check(cudaFreeHost(pinned_vel), "pinned velocity free");
    cudaEventDestroy(done);
    cudaStreamDestroy(stream);
    return 0;
}
