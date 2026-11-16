# Shared HOST region: phase F preparation

This is source preparation, not an executed probe or a selected allocation.
Run only after the bounded prompt-reading phase E, in the coordinator's planned
idle GPU window. The shared engine source and serving defaults are unchanged.

The authority is the free-time mover row in [AGENTS.md](../../AGENTS.md): one
HOST region directly reachable by CPU, 5090, 3090 and 4080; no CPU payload copy
on a swap's path; CPU re-sorting only in quiet time; chunked work on both helpers
under a shared budget; and a spare 3090 slot. The older fixed cadence and
one-card-per-step rules were superseded. This document describes implementation
consequences; it does not create another decision list.

## Bounded allocation gate

The standalone [probe](../../tools/mm3/bench/shared_host_probe.cu) owns one private
allocation and 8 MiB of device scratch per card. The [ladder driver](../../tools/mm3/probe_shared_host.py)
requires `--max-mib`, capped at 4096. Its ladder is 64, 256, 1024 and 4096 MiB up
to that ceiling, adding the exact ceiling when necessary. Only one region exists
at a time. Every attempt has a new process and CUDA contexts, with a 180-second
process watchdog and a default 600-second overall driver budget.

1. Try HOST VMM first: `CU_MEM_ALLOCATION_TYPE_PINNED`, location `HOST`, no
   exported handle. Reserve and map one range, grant CPU and all three GPU
   locations read/write permission, then re-query every final permission. Log
   advertised HOST-VMM capability separately from the actual grants. A helper's
   missing capability bit does not silently skip its real permission attempt.
2. If a VMM allocation or access grant is refused, the default driver explicitly
   tries `cudaHostAllocPortable | cudaHostAllocMapped` from the smallest size.
   This is the same kind of canonical shared region, not an incoming/outgoing
   staging pair. Each GPU's mapped pointer is recorded; equal virtual addresses
   are not assumed. Neither managed memory, partial registration, nor a hidden
   pageable remainder is introduced.
3. Any actual payload error, CUDA execution error, timeout or memory-reserve
   refusal stops the run. It is not relabeled as an unsupported allocation and
   hidden by trying another backend. All failed logs stay alongside successes.

HOST VMM permits explicit access descriptors per location. The normal pinned
alternative is documented as portable across CUDA contexts and mapped into the
device address space; its device alias may differ from the CPU address.
These API descriptions are prerequisites, not proof of support on this PC.
[NVIDIA VMM API](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__VA.html),
[NVIDIA pinned HOST API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY.html).

The existing `native_host_reserve_snapshot()` supplies Linux/cgroup and Windows
physical/commit receipts before allocation, after allocation, after checks and
after release. The allocation retains the established 8 GiB effective reserve.
The driver records read-only GPU power limits and free/used memory before and
after the ladder. It never sets a power limit, clock, driver flag or serving
configuration. GPU identities are recorded by name, UUID and PCI location; use
the coordinator's validated UUID ordering if setting `CUDA_VISIBLE_DEVICES`.

The correctness population is deliberately stronger than a first-page touch:

- CPU initializes every word; every GPU reads the entire requested region.
- Each GPU writes every word, followed by an exact CPU check. The next GPU reads
  the preceding GPU's full payload; all GPUs check the final payload.
- Each card performs H2D and D2H DMA directly against the region at its beginning,
  middle and end, using only bounded device scratch. Another GPU checks the D2H
  result through its own mapped alias; the window is restored and checked.
- Both helpers receive disjoint write submissions to the same allocation. CPU
  and 5090 verify both slices after retirement. This checks shared visibility;
  it does not establish physical overlap or concurrent-copy bandwidth.

Kernels operate in 8 MiB chunks. Every GPU operation is checked and retired
before a dependent CPU read or reuse. A failure with uncertain GPU access exits
without freeing leased storage. There is no model record, weight format change,
model request or performance adoption in this utility.

A passing 4 GiB run means “4 GiB tested successfully,” not “the mapping limit is
4 GiB.” A failing larger size bounds only the tested condition and allocation
route. The 3090's usable limit, competition with the resident engine, helper
transfer bandwidth, CPU read bandwidth and actual pass cost remain distinct
measurements. A successful empty-engine mapping must later be budgeted and
checked in the intended model allocation before integration is qualified.

## Coordinator build and run

No build or execution was performed during preparation. The coordinator may add
a manual `EXCLUDE_FROM_ALL` target with this equivalent definition in its next
allowed build; no CMake or engine source was changed by this preparation:

```cmake
add_executable(mm3_shared_host_probe EXCLUDE_FROM_ALL
    tools/mm3/bench/shared_host_probe.cu src/mm3/mm3_native_host.cpp)
target_include_directories(mm3_shared_host_probe PRIVATE src/mm3)
target_link_libraries(mm3_shared_host_probe PRIVATE
    CUDA::cudart CUDA::cuda_driver Threads::Threads)
set_target_properties(mm3_shared_host_probe PROPERTIES CUDA_ARCHITECTURES "86;89;120")
```

After E, preserve the current serving state, stop serving for the approved probe
gap, hold the existing GPU/quiet ownership, and bind the build's source and binary
hashes to the run. The driver does not stop serving or acquire that ownership.
For example, with an explicitly chosen 4 GiB ceiling:

```sh
python3 tools/mm3/probe_shared_host.py \
  --binary build-local/mm3_shared_host_probe \
  --max-mib 4096 \
  --out /mnt/wsl/transfer/minimax-wsl/runs/campaign-resume-20261116/placement/shared-region/measurement
```

The output directory must be new or empty. Optional `--cuda-visible-devices`
takes the coordinator's UUID list. `--backend vmm` or `--backend pinned` permits a
named follow-up; default `auto` uses the ordered alternative described above.
Do not automatically expand the ceiling or repeat a successful ladder.

## Minimal integration after the probe

Current storage is selective: `NativeHostBank` grants CPU and 5090 access;
`Layer::host_blob` points at canonical HOST records. The 3090 owns exclusive
records, while the 4080 holds copies whose canonical HOST records remain.
Today's `helper_live` copies HOST→incoming staging and outgoing staging→HOST on
the CPU, and overwrites the victim's 3090 slot. `third_set()` also stages through
the CPU. None of those paths constitutes the approved direct-region mover.

The first integration should make the following ownership transitions explicit,
using the successful allocation kind and its measured size. Do not implement a
large mover before that result exists.

**Canonical region and leases.** Introduce a shared-region frame table with
stable logical record IDs, byte offsets, generations, original scale metadata,
canonical-owner state, and CPU/primary/helper reader leases. Device aliases are
derived from the frame offset, never serialized as pointers. A record transferred
into the region becomes canonically owned there after its former readers retire;
the old HOST frame then becomes free. The region replaces those backing frames
in the memory ledger. It is not an additional full-copy bank or a second staging
pair. Partition the canonical HOST capacity between the ordinary CPU/5090 bank
and this region at the model load: allocating the old full bank plus an equally
counted shared subset would not implement that accounting. Existing contiguous
bank mappings cannot be assumed to release arbitrary freed record pages.
Include mapping-granularity padding and at least one free region frame for a
3090 victim landing in the startup memory ledger.

**3090 exchange without CPU payload copies.** Reserve one physical GPU slot
inside the existing 3090 budget and expose the reduced ready-record capacity.
While A remains canonical in a shared HOST frame, upload A in chunks to the spare
3090 slot. Concurrently download GPU-owned victim B in chunks directly into the
free shared HOST frame. Distinct streams may overlap these directions when the
measured link and shared memory budget permit it. Keep B's old slot readable until
the required reader leases end. After both payloads complete and readers retire,
publish A on the 3090 and B in HOST, retire A's old HOST readers, and rotate the
freed A HOST frame and old B GPU slot into the free lists. No whole-record CPU
copy is on this path. Small ownership/scale metadata updates are explicit.

This needs a new direct lease operation in `NativeHelper`; changing pointers
passed to existing `live_begin()` is insufficient because it always invokes the
CPU copier and uses the victim's slot. The current record-version/graph identity
checks must cover the new spare-slot publication. Report the one-record capacity
cost separately: a control with the same reduced capacity isolates the mover,
while the normal full-capacity control accounts for the deployment's lost slot.

**4080 copy-cache replacement.** Canonical incoming records stay in the shared
HOST region, and the 4080 replaces only its copy. After old compute readers
retire, remove the victim's ready bit, upload the new record directly in chunks,
then publish the new identity/version. No victim download or CPU payload copy is
needed. Retain the chosen tier ordering and keep incoming/outgoing identities
leased against primary/3090 placement and region re-sorting. A record whose only
canonical source is outside the direct region is not eligible for a swap yet.

**Quiet-time re-sorting.** A separate bounded CPU job prepares future region
contents only while the CPU lane and shared memory budget are free. It reserves
source/destination generations and publishes a new canonical pointer only after
the copy and all displaced-reader leases retire. It must never be awaited by a
pending helper swap; that swap chooses a ready region record or defers. Fixed
urgent host-copy escapes are not carried into this mechanism. Re-sort bytes and
time are charged to background work, even if they occur before an admitted swap.
Exchanging a private-bank record with a region record can require two background
copies through free canonical frames. Reserve those frames independently from
any already-admitted GPU victim landing, and do not hide this CPU traffic merely
because the subsequent GPU swap has no CPU copy on its path.

**Both helpers and one shared budget.** Extend the repaired generation guard to
both helper movers and cover every 5090 demand, early, promotion and exchange
stream. Combine the actual CPU-idle state, each helper's compute/result/link state
and the shared HOST traffic budget. Both helpers may have chunks in flight; no
one-card-per-step rule returns. An admitted chunk can cross a foreground boundary,
so count its accepted bytes and completion separately from permission. Never call
revocation preemption or claim zero physical overlap from admission counters.
Compare an incoming record's expected pass benefit against displaced work,
transfer cost and uncertainty; “keep placement” remains an option. Original router
demand supplies admission demand. A margin/minimum-stay guard grows with the
delivered swap rate. The planner chooses 3090 versus 5090; 4080 keeps its tiers.

**Restore and failure handling.** Before placement/reference restore, close new
admissions, settle or cancel both helper leases and region re-sorting, and join
all CPU, prompt, primary, queued-helper and transfer readers. Restore canonical
identities, helper rosters, spare/free slots, region membership, demand learning,
churn ages and budget state from logical IDs. Clear transient grants and reset
per-run counters. Extend reference signatures/versioning to bind allocation kind,
region capacity, usable helper capacities and mover policy. Old incompatible
references are refused, not silently coerced. A failed or uncertain retirement
retains the complete allocation/lease owner and forbids further publication.

The next gate after allocation is a small production-API lifetime fixture for
these transitions, then a bounded forced-rate cost check and the approved
multi-turn cache-benefit comparison. Judge each pass from one common origin and
then warmed token time. Mapping success alone completes none of those steps.
