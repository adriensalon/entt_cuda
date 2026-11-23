# Phase-0 contracts

This records decisions for the implementation gates in `AGENTS.md`. The current
headers provide a compile-only core API contract. They do not yet provide a
constructible, linkable CUDA registry. Runtime definitions, device storage and
submission bookkeeping start in phase 1. No placeholder returns `cudaSuccess`.

## Dependency and language

The compatibility reference is the unchanged submodule at
`1dcd7aab6548786f7672ff578639188bd2f0d837`. Its version header declares 4.1.0;
this is a revision pin, not a claim that it is a released 4.1.0 tag. CMake checks
the commit when Git metadata is available, and checks normalized registry and
version-header hashes even in source archives. The gitlink pins the other ECS
headers. Public declaration coverage is recorded in [api-coverage.md](api-coverage.md).

Both host and CUDA translation units require C++20. `EnTT::Cuda` supplies the
include paths, upstream `EnTT::EnTT`, native CUDA runtime linkage and
`--extended-lambda` for NVIDIA CUDA compilation. Kernel and callback definitions
will remain visible to the CUDA compiler. Host metadata APIs remain callable
from ordinary C++ translation units. No public execution wrappers are planned.

The selected baseline is NVIDIA nvcc 13.2+; the checked host compiler is MSVC
19.44.35229. Earlier local nvcc 12.4/12.9 builds reject the pinned hashed-string
header. The upstream facilities reused here compile with nvcc 13.2, while full
CPU-registry instantiation stays in host translation units because its nonzero
SBO union triggers a separate diagnostic. See README.md for reproduction and
the currently unmet CUDA 13 runtime driver requirement. The compile and host
checks pass; this GPU runtime validation gap keeps phase 0 open.

## Emplacement argument placement

The three public overloads have the core argument order from `AGENTS.md`.
Constraints reserve a leading `cudaStream_t` or `nullptr` for the stream, then a
leading `cudaEvent_t` or `nullptr` for the completion event. The reservation uses
the exact native type after removing cv/reference qualifiers. A user-defined
conversion to a handle does not reserve an argument. Integer zero is an ordinary
constructor argument; use `nullptr` for an omitted handle.

| Call suffix after the entity | Stream | Event | Component initializer |
| --- | --- | --- | --- |
| no arguments | default | none | `T{}` |
| `1.f, 2.f, 3.f` | default | none | `T{1.f, 2.f, 3.f}` |
| `nullptr` | default | none | `T{}` |
| `stream, T{...}` | `stream` | none | the supplied value |
| `stream, done, args...` | `stream` | `done` | `T{args...}` |
| `nullptr, done, args...` | default | `done` | `T{args...}` |
| `stream, nullptr, args...` | `stream` | none | `T{args...}` |
| `nullptr, nullptr, args...` | default | none | `T{args...}` |

This necessarily reserves ambiguous constructor arguments. For a component that
contains or accepts a stream, use `emplace<T>(e, T{stream})` or
`emplace<T>(e, stream, done, stream)`. Similarly, pass `T{nullptr}` for a null
pointer payload, or place the payload after explicit stream and event arguments.
An event supplied without a preceding stream is a constructor argument, not a
completion event. These choices avoid component-dependent interpretation of
`nullptr`. Construction uses braces, including aggregate initialization and its
usual narrowing restrictions.

The phase-0 object test instantiates actual forwarding bodies for these cases.
Runtime leaf operations are intentionally undefined until phase 1. Query calls
are compile-only signature checks; device callback invocation and const-reference
rejection must be tested against the actual phase-1 dispatch kernel.

## Component and raw-access policy

Initially components must be unqualified, non-array object types with trivial
copying and destruction. Registry entity identity cannot be emplaced as a
component. Stable/in-place deletion and custom component page sizes are rejected
until their phase-4 implementation. The default EnTT traits are reused; an empty
tag has membership only. `get` and `assign` reject tags at compile time. Standalone
void storage is deferred to phase 3. These restrictions are checked by named
concepts so compiler diagnostics identify the failed requirement.

Trivial copying cannot prove that a type is safe on the device. Callers remain
responsible for device-compatible representation, constructors used on the host,
and the accessibility/lifetime of embedded pointers. A trivially copyable host
pointer does not become device-accessible. Nontrivial host resources are rejected.
Device callback captures are by value and device-safe; callbacks cannot change
registry structure. Arbitrary device writes produce no automatic notifications.

Host metadata owns entity IDs, generations, alive/free lists, dense entity order,
sparse host lookup and capacity/submission bookkeeping. The device owns component
values, dense IDs and paged sparse indices. Membership must check sparse bounds
and compare the complete generation-bearing entity with the dense ID. Growth and
swap-and-pop preserve current device values with device-to-device relocation.

Later `try_get` returns nullable device pointers, individually nullable tuples for
multiple components, and const pointers for const access. It does not download.
Tags have no value pointers. Storage exposes metadata/descriptors, not host
component references. All descriptors and raw pointers are non-owning and may be
invalidated by structural changes. Reacquire after such changes.

## Allocation and errors

Default construction initializes host ownership only. Device affinity binds to
the current CUDA device at the first device operation; later submissions must
use that device. Calls from the host must be serialized or externally locked,
including completion polling and teardown. Multi-GPU use is outside this design.

`create()` performs no device submission. It returns a full generational ID,
preferring the host free list. It throws `std::bad_alloc` on host allocation
failure and `std::length_error` on entity-index exhaustion or a host-container
size limit. Reserved null/tombstone identities are never returned. Host capacity
and free-list updates must be prepared before committing an ID, so failure
consumes no identifier. Exhaustion needs an explicit check; upstream debug-only
assertions are insufficient. Hinted creation later follows the same policy.

Status-returning APIs translate host allocation failures, including capacity
reservation failure, and CUDA allocation failure to `cudaErrorMemoryAllocation`.
Other exceptions from user component constructors propagate before submission;
the registry remains unchanged. Components are constructed and upload staging
is owned before logical mutation. Invalid/stale entities, duplicate emplacement,
missing transfer targets and invalid buffers return `cudaErrorInvalidValue`
before enqueueing. A valid entity with an absent component can be removed
successfully. Pinned transfer buffers must be non-null, correctly aligned and
large enough for the value; their pinned-host attributes are checked before a
copy is submitted. Callers supply valid accessible object storage.

Device allocation should use native stream-ordered allocation when supported.
Blocking allocation fallback, if needed, is confined to documented structural
preparation/reserve paths. `each`, `get` and `assign` must not introduce hidden
host synchronization, including allocation/free paths. Pinned staging and private
event reuse must be prepared accordingly. Capacity growth may allocate; a
preallocated steady-state path must stay asynchronous. Initial contiguous device
growth provides no pointer-stability guarantee. CUDA Graph capture/replay is
deferred to phase 6, even if individual CUDA calls happen to permit capture.

## Submission and fault state

1. Validate all inputs, stage values and prepare host/device resource ownership.
   A preflight failure leaves logical metadata unchanged and does not poison an
   otherwise healthy registry.
2. Enqueue the work on the native caller stream, using the default stream for
   `nullptr`. No separate publication call exists. Retain every resource already
   referenced by submitted work, including during an error path.
3. Record private completion/lifetime bookkeeping covering all library uses,
   independently of the optional caller event. Old descriptors/allocations remain
   alive until every recorded library use is complete. Polling may query events
   but cannot block these asynchronous APIs.
4. Record caller-owned `done`, if supplied, after all operation work. Successful
   no-op removals and empty dispatches still record it. Commit host logical
   metadata only after all required submission steps succeed. `cudaSuccess`
   describes submission, not execution completion.

Once submission has begun, an enqueue or event-record failure can expose earlier
asynchronous errors and cannot promise rollback. Retain the old metadata and all
potentially referenced allocations/staging, save the first failure and enter a
sticky fault state. Metadata queries expose that frozen host state, which is no
longer a claim about completed device structure. Later status-returning
operations return the saved error without enqueueing. `create` and other
non-status mutations throw `std::logic_error` in this state. Teardown followed by
recreation is the recovery path; there is no public fault-reset shortcut.
Execution errors observed only by the caller are likewise grounds for teardown;
the registry cannot automatically observe errors in arbitrary external kernels.

Caller events are never owned/destroyed by the registry. Event recording can
fail after work was submitted. Streams/events used for outstanding work must
outlive the registry or completion of its teardown. A failed private-event record
requires retaining the stream and resources for conservative teardown rather
than freeing them early. Error injection tests are a phase-1 acceptance item.

## Ordering, views and teardown

Same-stream work observes previous submissions. Callers use native events and
`cudaStreamWaitEvent` for cross-stream dependencies, including growth, relocation,
removal and recycled IDs. Private retirement tracking keeps memory alive; it does
not establish data-race or structural ordering for the caller. A `done` event
covers only work ordered before it on its stream. External kernels must finish
their last ordered use before an affected structural change or teardown.

Views retain reusable query recipes and resolve current descriptors/count at
each submission. They choose the smallest required pool unless overridden;
missing required/excluded pools respectively produce no matches/no restriction.
Const registries do not create pools. Each match receives component references
with the requested constness, optionally preceded by immutable entity identity.
Tags supply no callback values. No host component iteration or per-entity
downloads occur. Entity-only queries can upload a bulk alive-ID snapshot once
their asynchronous staging path is available. Callback order is unspecified.

Caller pinned destinations and upload sources remain alive until ordered
completion; sources remain unchanged. Internal staging is retained independently
of caller `done`. Destruction is blocking: it drains all library-owned work,
including retained work on fault paths, before releasing allocations and staging.
It does not wait for unordered external uses. Teardown uses the owning device
and restores the caller's previously selected device when possible. The
destructor is nonthrowing; a CUDA context/device failure may prevent reclamation,
and uncertain resources must never be freed before their last possible use.
Destroy the registry before caller stream/event handles. Moves transfer all
ownership and outstanding bookkeeping; existing views/handles must be reacquired.

## Acceptance boundaries

Phase 0 validates header integration, overload resolution, restrictions, the
pinned CPU reference, native CUDA execution and extended lambdas. It does not
validate a registry implementation. Phase 1 must add same-stream
`emplace -> each -> get`, GPU-written value relocation, generation/sparse-page
boundaries, empty/tag/excluded queries, callback forms, cross-stream handoffs,
no-event staging/retirement, teardown and fault injection. Randomized comparisons
and Compute Sanitizer apply when those operations exist. Later-phase APIs remain
explicitly deferred in the coverage inventory.
