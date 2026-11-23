#include <memory>
#include <entt/core/type_info.hpp>
#include <entt/entity/registry.hpp>
#include <entt/entity/storage.hpp>
#include <entt/entt_cuda.hpp>
#include <entt/signal/sigh.hpp>

// Check the upstream host facilities that the CUDA implementation will reuse.
// The full CPU registry/reference runs in host_contract.cpp: nvcc 13.2 rejects
// upstream basic_any's nonzero-sized SBO union when the CPU registry binds pools.
void dependency_contract() {
    entt::basic_storage<entt::entity> identities;
    const auto entity = identities.generate();
    [[maybe_unused]] const auto generation = identities.current(entity);
    identities.erase(entity);
    [[maybe_unused]] const auto recycled = identities.generate();
    [[maybe_unused]] const auto hash = entt::type_hash<int>::value();
    [[maybe_unused]] const auto &type = entt::type_id<int>();

    entt::registry::context context{std::allocator<entt::entity>{}};
    context.emplace<int>(42);
    context.erase<int>();
    int value{};
    context.emplace<int &>(value);
    [[maybe_unused]] auto *alias = context.find<int>();

    entt::sigh<void(int)> signal;
    [[maybe_unused]] entt::sink sink{signal};
    signal.publish(value);
}
