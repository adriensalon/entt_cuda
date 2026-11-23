#include <cstdio>
#include <cstdlib>
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

// The CPU EnTT reference is used for identity/version/context semantics that are
// independent of the device. It must live in a host translation unit.
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

    if(!check(cudaMallocHost(reinterpret_cast<void **>(&pinned), sizeof(position)), "pinned alloc")) {
        return 1;
    }

    entt::registry reference;
    entt::cuda_registry device;

    // Identity: create, valid, current, hinted and ranged creation.
    const auto lhs = device.create();
    const auto rhs = reference.create();

    if(lhs != rhs || !device.valid(lhs) || !device.all_of<>(lhs) || device.any_of<>(lhs) || !device.orphan(lhs)) {
        std::fprintf(stderr, "create/valid/orphan/all_of/any_of mismatch\n");
        return 1;
    }

    const auto hinted = device.create(lhs);
    const auto hinted_ref = reference.create(lhs);

    if(hinted != hinted_ref) {
        std::fprintf(stderr, "create(hint) mismatch\n");
        return 1;
    }

    std::vector<entt::entity> range(3u);
    device.create(range.begin(), range.end());
    std::vector<entt::entity> ref_range(3u);
    reference.create(ref_range.begin(), ref_range.end());

    if(range != ref_range) {
        std::fprintf(stderr, "create(range) mismatch\n");
        return 1;
    }

    // current tracks the version.
    if(device.current(lhs) != reference.current(lhs)) {
        std::fprintf(stderr, "current mismatch\n");
        return 1;
    }

    // emplace + all_of/any_of/try_get.
    if(!check(device.emplace<position>(lhs, stream, position{1.f, 2.f, 3.f}), "emplace")
       || !check(device.emplace<tag>(lhs, stream), "emplace tag")) {
        return 1;
    }

    check(cudaStreamSynchronize(stream), "sync");

    if(!device.all_of<position, tag>(lhs) || !device.any_of<position>(lhs) || device.any_of<velocity>(lhs) || device.orphan(lhs)) {
        std::fprintf(stderr, "membership mismatch after emplace\n");
        return 1;
    }

    // try_get returns the device pointer; verify it reads back the value.
    if(auto *ptr = device.try_get<position>(lhs); ptr == nullptr) {
        std::fprintf(stderr, "try_get returned null for a present component\n");
        return 1;
    } else {
        check(cudaMemcpy(pinned, ptr, sizeof(position), cudaMemcpyDeviceToHost), "try_get copy");

        if(pinned->x != 1.f || pinned->y != 2.f || pinned->z != 3.f) {
            std::fprintf(stderr, "try_get value mismatch\n");
            return 1;
        }
    }

    if(device.try_get<velocity>(lhs) != nullptr || device.try_get<tag>(lhs) != nullptr) {
        std::fprintf(stderr, "try_get should return null for absent/tag components\n");
        return 1;
    }

    // Multi-type try_get yields a tuple of pointers.
    const auto tuple = device.try_get<position, tag>(lhs);

    if(std::get<0>(tuple) == nullptr || std::get<1>(tuple) != nullptr) {
        std::fprintf(stderr, "multi-type try_get mismatch\n");
        return 1;
    }

    // const try_get yields const pointers.
    const entt::cuda_registry &cdevice = device;
    const auto *const_ptr = cdevice.try_get<position>(lhs);

    if(const_ptr == nullptr || !check(cdevice.get<position>(lhs, pinned, stream, done), "const get")
       || !check(cudaEventSynchronize(done), "sync const get")) {
        std::fprintf(stderr, "const try_get/get mismatch\n");
        return 1;
    }

    // emplace_or_replace / get_or_emplace / replace.
    if(!check(device.emplace_or_replace<position>(lhs, stream, position{9.f, 9.f, 9.f}), "emplace_or_replace")
       || !check(device.get_or_emplace<velocity>(lhs, stream, velocity{4.f, 5.f, 6.f}), "get_or_emplace")
       || !check(device.replace<position>(lhs, stream, position{7.f, 7.f, 7.f}), "replace")
       || !check(cudaStreamSynchronize(stream), "sync replacements")) {
        return 1;
    }

    if(!check(device.get<position>(lhs, pinned, stream, done), "get after replace")
       || !check(cudaEventSynchronize(done), "sync get after replace")) {
        return 1;
    }

    if(pinned->x != 7.f || pinned->y != 7.f || pinned->z != 7.f) {
        std::fprintf(stderr, "replace/emplace_or_replace value mismatch\n");
        return 1;
    }

    if(!device.all_of<velocity>(lhs)) {
        std::fprintf(stderr, "get_or_emplace did not emplace\n");
        return 1;
    }

    // get_or_emplace is a no-op when present.
    if(!check(device.get_or_emplace<position>(lhs, stream, position{0.f, 0.f, 0.f}), "get_or_emplace present")
       || !check(cudaStreamSynchronize(stream), "sync get_or_emplace")
       || !check(device.get<position>(lhs, pinned, stream, done), "get after get_or_emplace")
       || !check(cudaEventSynchronize(done), "sync")
       || pinned->x != 7.f) {
        std::fprintf(stderr, "get_or_emplace should be a no-op when present\n");
        return 1;
    }

    // insert (repeated value and component range).
    const auto a = device.create();
    const auto b = device.create();
    const auto c = device.create();
    std::vector<entt::entity> trio{a, b, c};

    if(!check(device.insert<position>(trio.begin(), trio.end(), position{1.f, 1.f, 1.f}, stream, done), "insert value")
       || !check(cudaEventSynchronize(done), "sync insert")) {
        return 1;
    }

    if(!device.all_of<position>(a) || !device.all_of<position>(b) || !device.all_of<position>(c)) {
        std::fprintf(stderr, "insert did not assign every entity\n");
        return 1;
    }

    // remove multi-type + tolerant single type.
    if(!check(device.remove<position, velocity>(lhs, stream, done), "remove multi")
       || !check(device.remove<position>(lhs, stream, done), "remove absent (tolerant)")
       || !check(cudaStreamSynchronize(stream), "sync remove")) {
        return 1;
    }

    if(device.all_of<position>(lhs) || device.all_of<velocity>(lhs) || !device.all_of<tag>(lhs)) {
        std::fprintf(stderr, "remove did not drop the requested components\n");
        return 1;
    }

    // erase is strict.
    if(!check(device.emplace<position>(lhs, stream, position{2.f, 2.f, 2.f}), "emplace for erase")
       || device.erase<velocity>(lhs, stream, done) != cudaErrorInvalidValue
       || !check(device.erase<position>(lhs, stream, done), "erase")
       || !check(cudaStreamSynchronize(stream), "sync erase")) {
        std::fprintf(stderr, "erase strictness mismatch\n");
        return 1;
    }

    if(device.all_of<position>(lhs)) {
        std::fprintf(stderr, "erase did not remove the component\n");
        return 1;
    }

    // erase_if removes pools matching a predicate.
    if(!check(device.emplace<position>(lhs, stream, position{3.f, 3.f, 3.f}), "emplace for erase_if")
       || !check(device.emplace<velocity>(lhs, stream, velocity{4.f, 4.f, 4.f}), "emplace for erase_if")
       || !check(device.erase_if(lhs, [](const entt::id_type id, const entt::cuda_registry::common_type &) { return id == entt::type_hash<position>::value(); }, stream, done), "erase_if") || !check(cudaStreamSynchronize(stream), "sync erase_if")) {
        return 1;
    }

    if(device.all_of<position>(lhs) || !device.all_of<velocity>(lhs)) {
        std::fprintf(stderr, "erase_if removed the wrong pool\n");
        return 1;
    }

    // clear<T> retains entities; clear<> destroys them.
    if(!check(device.emplace<position>(lhs, stream, position{5.f, 5.f, 5.f}), "emplace for clear")
       || !check(device.clear<position>(stream, done), "clear<T>")
       || !check(cudaStreamSynchronize(stream), "sync clear<T>")) {
        return 1;
    }

    if(device.all_of<position>(lhs) || !device.valid(lhs)) {
        std::fprintf(stderr, "clear<T> should retain entities but drop the pool\n");
        return 1;
    }

    const auto alive_before = device.storage<position>().size();
    (void)alive_before;

    // destroy with a requested version recycles the identifier.
    const auto victim = device.create();
    const auto victim_index = entt::to_entity(victim);

    if(!check(device.destroy(victim, entt::to_version(victim) + 1u, stream, done), "destroy(version)")
       || !check(cudaEventSynchronize(done), "sync destroy(version)")) {
        return 1;
    }

    const auto recycled = device.create();

    if(entt::to_entity(recycled) != victim_index || entt::to_version(recycled) != (entt::to_version(victim) + 1u)) {
        std::fprintf(stderr, "destroy(version) did not set the requested version\n");
        return 1;
    }

    // ranged destroy.
    std::vector<entt::entity> doomed{a, b, c};
    device.emplace<position>(a, stream, position{1.f, 1.f, 1.f});

    if(!check(device.destroy(doomed.begin(), doomed.end(), stream, done), "destroy(range)")
       || !check(cudaEventSynchronize(done), "sync destroy(range)")) {
        return 1;
    }

    if(device.valid(a) || device.valid(b) || device.valid(c)) {
        std::fprintf(stderr, "destroy(range) did not destroy every entity\n");
        return 1;
    }

    // whole-registry clear destroys live entities but keeps pools/context.
    if(!check(device.clear<>(stream, done), "clear<>")
       || !check(cudaStreamSynchronize(stream), "sync clear<>")) {
        return 1;
    }

    // ctx() host context.
    device.ctx().emplace<int>(42);
    device.ctx().emplace_as<double>(7u, 3.5);

    if(!device.ctx().contains<int>() || device.ctx().get<int>() != 42 || device.ctx().get<double>(7u) != 3.5) {
        std::fprintf(stderr, "context emplace/get/contains mismatch\n");
        return 1;
    }

    device.ctx().insert_or_assign<int>(99);
    int &alias = device.ctx().get<int>();

    if(alias != 99) {
        std::fprintf(stderr, "context insert_or_assign mismatch\n");
        return 1;
    }

    if(!device.ctx().erase<int>() || device.ctx().contains<int>()) {
        std::fprintf(stderr, "context erase mismatch\n");
        return 1;
    }

    std::puts("registry breadth tests passed.");
    check(cudaFreeHost(pinned), "pinned free");
    cudaEventDestroy(done);
    cudaStreamDestroy(stream);
    return 0;
}
