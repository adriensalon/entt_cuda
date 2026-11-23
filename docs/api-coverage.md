# Pinned ECS API audit

Reference: `external/entt` at
`1dcd7aab6548786f7672ff578639188bd2f0d837`, declaring 4.1.0. Line numbers below
refer to `src/entt/entity/registry.hpp` at that revision. Audit scope is its entire
public `basic_registry` section and public context declarations, plus the ECS
owners listed in `AGENTS.md`. Private implementation helpers and unrelated EnTT
modules are excluded. The upstream submodule is unchanged.

**Contract** means declarations/compile checks only, with no runtime definition.
**Deferred** means neither an implemented CUDA adaptation nor a public promise
beyond the roadmap. **Implemented** means a runtime definition exists and is
covered by a runtime test. Deleted copy operations and emplacement argument
forwarding are implemented. The phase-1 asynchronous core (identity, pools,
emplacement, removal, transfers and dispatch) is implemented; remaining rows
stay contract/deferred. Phases are delivery gates, not estimates.

## Registry declarations

| Upstream declaration(s) and line | CUDA adaptation and current status | Gate |
| --- | --- | --- |
| `allocator_type` 266; `entity_type` 268; `version_type` 270; `size_type` 272 | Deferred: host allocator, entity traits and size aliases. | 2, 6 |
| `common_type` 274; `context` 276 | Deferred: erased CUDA metadata interface and upstream host context. | 2-3 |
| `iterable` 278; `const_iterable` 280 | Deferred: mutable/const pool enumeration excluding entity storage. | 3 |
| `storage_for_type<T>` 287 | Deferred: normalized typed CUDA storage, preserving constness. | 2-3, 6 |
| Default constructor 290 | Implemented: default-constructs host identity and an empty pool map; no device submission. | 1 |
| Explicit allocator constructor 297 | Deferred: host allocator customization. | 2, 6 |
| `(size_type count, allocator = {})` constructor 305 | Deferred: count reserves pool-map entries, not entity/component slots. | 2, 6 |
| Copy constructor 315; copy assignment 336 | Implemented as deleted operations; compile checks reject copying. | 0 |
| Move constructor 321; move assignment 343 | Implemented: `noexcept` transfer of identity, pools, retirement and device buffers; pool retirement bookkeeping is re-bound to the new owner. | 1-2 |
| Destructor 330 | Implemented: blocking, nonthrowing teardown drains retired work and frees live device buffers. | 1 |
| `swap` 352; `get_allocator() const` 368 | Deferred: transfer ownership/bookkeeping; report host allocator. | 2, 6 |
| `storage()` mutable 380 and const 385 | Deferred: enumerate component pool IDs/interfaces only. | 3 |
| `storage(id)` mutable 394 and const 403 | Deferred: nullable erased lookup, no pool creation. | 3 |
| `storage<T>(id = type_hash<T>)` mutable 415 and const 426 | Implemented for mutable default-name access only. Named and nullable const access deferred; entity storage ignores names. | 1-3 |
| `reset(id)` 435 | Deferred: native stream/event extension, status return, safe retirement; reject entity storage. | 3 |
| `valid(e) const` 445; `current(e) const` 455 | Deferred: CPU full-generation validation/version lookup. | 2 |
| `create()` 463 | Implemented: CPU-only generation with free-list recycling; throws `std::bad_alloc`/`std::length_error` per contracts.md. | 1 |
| `create(hint)` 476; output-iterator `create(first,last)` 490 | Deferred: upstream null/tombstone, generation/hint and output-range semantics. | 2 |
| `destroy(e)` 504 | Implemented: native stream/event parameters, `cudaError_t` instead of version; removes every component then recycles the ID. | 1 |
| `destroy(e,version)` 525; input-iterator `destroy(first,last)` 539 | Deferred: status returns and native stream/event placement, retained range inputs. | 2 |
| `emplace<T>(e,args...)` 566 | Implemented: all three core stream/event overloads forward to runtime staged upload, growth and sparse/dense device mirroring. Aggregate/handle/null cases compile-tested. | 0-1 |
| `insert<T>(first,last,value = {})` 582 | Deferred: repeated/default value staging and validation. | 2 |
| `insert<T>(first,last,components_first)` constrained to matching value type, 601 | Deferred: copy parallel inputs before return; caller supplies enough values. | 2 |
| `emplace_or_replace<T>` 619 | Deferred: status-returning replace-or-create. | 2 |
| `patch<T>(e,funcs...)` 645 | Deferred: device-callable ordered functions; empty pack is notification-only; immutable entity identity. | 2, 5 |
| `replace<T>` 665 | Deferred: staged replacement, no host reference result. | 2 |
| `remove<T,Other...>(e)` 677 | Implemented for scalar type only, including const-type normalization; tolerant swap-and-pop removal and status return. | 1-2 |
| `remove<T,Other...>(first,last)` 694 | Deferred: retained scalar/range/multi-type submission. | 2 |
| `erase<T,Other...>(e)` 731 and `(first,last)` 747 | Deferred: strict validation/removal; status return. | 2 |
| `erase_if(e,func)` 783 | Deferred: predicate sees pool ID and const host metadata; no device values. | 2 |
| `compact<T...>()` 797, including empty pack | Deferred: device relocation/map repair; compact pools succeed as no-ops. | 4 |
| `all_of<T...>(e) const` 814; `any_of<T...>(e) const` 831 | Deferred: const-normalized host membership; true/false respectively for empty packs, as checked in the CPU reference test. | 2 |
| `get<T...>(e) const` 847 and mutable 857 | Implemented for mutable scalar pinned download. Const and multi-type/tuple destinations deferred; validate all destinations before copies, one final event. Host component reference overloads deliberately unavailable. | 1-2 |
| `get_or_emplace<T>` 881 | Deferred: ensure-only status return; separate download/device lookup for access. | 2 |
| `try_get<T...>(e) const` 898 and mutable 909 | Deferred: nullable device pointers/tuples with const propagation; no host references, downloads or tag value pointers. | 2 |
| `clear<T...>()` 922, including empty pack | Deferred: selected pools retain entities; whole-registry clear destroys live IDs and retains pools/context. | 2 |
| `orphan(e) const` 940 | Deferred: includes every named pool. | 2 |
| `on_construct<T>(id)` 964; `on_update<T>(id)` 988; `on_destroy<T>(id)` 1012 | Deferred: default/named/entity sinks, logical submission notifications with triggering stream. | 5 |
| `view<T,Other...>(exclude_t<U...> = {}) const` 1025 and mutable 1034 | Implemented for mutable queries with requested const types and exclusions, including entity-only/exclude-only dispatch and both callback forms. Const-registry overload deferred. Empty required packs are a deliberate CUDA extension; excluded pools are bounded to a documented compile-time maximum. | 1-3 |
| `group<Owned...>(get_t<Get...> = {},exclude_t<U...> = {})` 1047 | Deferred: non-owning first, then partial/full ownership and device permutations. | 4 |
| `group_if_exists<Owned...>(...) const` 1071 | Deferred: nullable lookup without group/pool creation. | 4 |
| `owned<T...>() const` 1089 | Deferred: host ownership queries/conflict checks. | 4 |
| `sort<T>(compare,Sort algo = {},args...)` 1127 | Deferred: CPU entity permutation/GPU gathers and comparator/algorithm customization. Value sorting requires completed download or separate device order. | 4 |
| `sort<To,From>()` 1153 | Deferred: align common entities, preserve CPU/GPU slot agreement, reject owned pools. | 4 |
| `ctx()` mutable 1163 and const 1168 | Deferred: reuse complete upstream host context. Context is not automatically device data. | 2 |

There are no additional public/deleted registry declarations at this pin.
`basic_cuda_registry<Entity, Allocator>` customization is deferred to phase 6;
the default spelling remains `cuda_registry`.

## Host context declarations

The context is `internal::registry_context<Allocator>` in the pinned registry
header. All entries are deferred to phase 2 through reuse, rather than a partial
replacement.

| Declaration(s) and line | Required preservation |
| --- | --- |
| Explicit allocator constructor 137; `clear() noexcept` 140 | Host ownership and allocator behavior. |
| `emplace_as<T>(id,args...)` 145; `emplace<T>(args...)` 150 | Named/default construction, duplicate behavior and reference aliases. |
| `insert_or_assign(id,Type&&)` 155; `insert_or_assign(Type&&)` 160 | Both named/default forms and forwarding/reference behavior. |
| `erase<T>(id = type_id<T>)` 165 | Type-checked named erase. |
| `get<T>(id)` const 171 and mutable 176 | Host references, including const/reference aliases. |
| `find<T>(id)` const 181 and mutable 187 | Nullable pointers and exact type checks. |
| `contains<T>(id)` const 193 | Named/type-aware membership. |

## Other ECS owners

These facilities remain on their upstream owner. They are not invented registry
methods. This is the planned owner-level checklist; exhaustive declaration audits
for each owner must accompany its delivery gate.

| Owner/reference headers and upstream test files | Planned scope; current status | Gate |
| --- | --- | --- |
| `entity.hpp`, `component.hpp`; `entity.cpp`, `component.cpp` | Reuse host IDs, null/tombstone and component traits. Traits participate in phase-0 constraints; generation behavior checked only on the CPU reference. Device identity/traits diagnostics still need runtime acceptance. | 0-2, 4 |
| `sparse_set.hpp`, `storage.hpp`, `storage_utility.hpp`; `sparse_set.cpp`, `storage.cpp`, `storage_entity.cpp`, `storage_no_instance.cpp`, `storage_utility.cpp` | Deferred: host metadata iteration, contains/size/empty/capacity, native reserve/shrink, device descriptors and sparse pages; later standalone/void/tag pools, entity free lists/start_from, stable paged storage, in-place deletion/tombstones, raw access and compaction. | 1-4 |
| `view.hpp`, `runtime_view.hpp`; `view.cpp`, `runtime_view.cpp` | Only core typed signatures declared. Deferred: device each, contains/find, size_hint/exact compact single-pool size, use, refresh/rebinding, composition, host entity iteration, entity/exclude-only snapshots and runtime filters with captured typed descriptors. No runtime device compilation. | 1-3 |
| `group.hpp`; `group.cpp` | Deferred: non-/partial-/full-owning membership, device each, metadata iteration, aligned permutations, sorting, ownership conflicts and stable-storage restrictions. Dispatch order stays unspecified. | 4 |
| `mixin.hpp`, `reactive_mixin.hpp`, signal facilities; `sigh_mixin.cpp`, `reactive_mixin.cpp` | Deferred: sinks, auto-binding, connection/dependency helpers, explicit update tracking/reactive views; define assign notifications and failure/reentrancy rules. Kernel writes are not automatically observed. | 5 |
| `snapshot.hpp`; `snapshot.cpp` | Deferred: full/selected/named identity and completed pinned-value archives, versions/free lists, empty restore, orphan cleanup, continuous mapping, caller-defined entity-reference remapping. Never serialize process-local pointers. | 5 |
| `handle.hpp`, `helper.hpp`, `organizer.hpp`; `handle.cpp`, `helper.cpp`, `organizer.cpp` | Deferred: mutable/const forwarding handles, device component-to-entity lookup, invoke/dependency adaptations and optional native-CUDA organizer descriptions. | 5-6 |

## Explicit adaptations and exclusions

- `assign` is a CUDA upload extension. There is no upstream registry method to
  audit. Its signature is declared and its runtime upload is implemented in phase
  1; notification behavior is phase 5. No redundant transfer aliases are added.
- Native `cudaError_t` replaces values/references/counts from enqueueing upstream
  operations. Host metadata queries keep ordinary result types. Native
  stream/event parameters follow the core placement convention.
- Invalid operations that upstream treats as assertions/undefined behavior become
  defined validation failures where required by `AGENTS.md`.
- Initial contiguous allocation is not upstream paged pointer stability. Custom
  page sizes and in-place deletion are explicitly rejected until phase 4.
- Component values have no host reference iteration. Sorting cannot promise
  callback execution order. Const queries cannot create missing pools.
- GPU-origin structural commands, multiple devices, nontrivial device lifecycles
  and automatic device change detection require later separate designs.

## Verification recorded at phase 0/1

`api_contract.cu` compiles the actual core overloads and both callback signature
forms into an object; it links against the runtime registry. `host_contract.cpp`
checks dependency identity and empty-pack expectations adapted from upstream
registry tests; it is not a CUDA-registry parity test. `dependency_contract.cu`
checks C++20 parsing and instantiation of the reused upstream identity storage,
type IDs, zero-SBO context/reference aliases and signals under nvcc 13.2. Full CPU
registry instantiation remains in the host translation unit because its nonzero
SBO `basic_any` union triggers an nvcc diagnostic. `toolchain.cu` checks an
extended device lambda, pinned transfers and native nonblocking-stream/event
handoff; it passes on this workspace's GTX 1650 with driver 621.24 and the
workspace-local CUDA 13.2 toolkit. `registry_core.cu` exercises the phase-1 core
(`emplace -> each -> get`, relocation across growth and sparse-page boundaries,
tolerant removal, tags/exclusions, both callback forms, entity-only and empty
required pools, destroy with version recycling, cross-stream handoff and move
construction) and reports clean under Compute Sanitizer memcheck. `registry_fault.cu`
checks the sticky fault state, and `registry_parity.cpp` compares 3000 randomized
edit/query steps against the CPU EnTT reference with bit-exact agreement.
