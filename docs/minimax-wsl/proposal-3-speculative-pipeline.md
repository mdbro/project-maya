# Proposal 3 reference: grouped verification and stage runtime

The [central plan](README.md) owns current requirements, status and completion.
Its [parked-work table](README.md#parked-work) governs layer-pipeline work, and its
[deferred base comparison](README.md#runtime-base-decision-remains-open) governs
Strata. Token speculation and layer pipelining are separate mechanisms.

## Implementation map

| Area | Source | Role |
|---|---|---|
| Strata MiniMax request/window ownership | [runtime entry](../../../strata/src/program/minimax_runtime/main.cpp), [session](../../../strata/src/program/minimax_runtime/session.hpp), [window](../../../strata/src/program/minimax_runtime/window.hpp), [serial commit](../../../strata/src/program/minimax_runtime/serial_commit.hpp) | Existing integration for request, speculative and committed state. |
| Strata model/stage adapters | [model stage](../../../strata/src/program/minimax_runtime/model_stage.hpp), [model request](../../../strata/src/program/minimax_runtime/model_request.hpp), [model](../../../strata/src/minimax/mm3_model.cu), [stage acceptance](../../../strata/src/minimax/stage_accept.hpp) | Adapted MiniMax ownership and stage machinery; the source tree is distinct from its pinned donor snapshot. |
| Provenance | [source record](../../../strata/src/minimax/source-provenance.json), [pinned Maya donor](../../../strata/reference/maya-minimax/README.md) | Original source identities; donor presence alone is not an implemented port. |
| Original Strata stage runtime | [generate.cpp](../../../strata/src/program/generate.cpp), [MULTI_GPU.md](../../../strata/docs/MULTI_GPU.md) | Stage workers, speculative/committed frontiers, restore and buffer lifetimes. |
| Maya DSpark integration | [DSpark](../../src/mm3/mm3_dspark.cu), [interface](../../src/mm3/mm3_dspark.hpp), [serving loop](../../src/mm3/mm3_main.cpp), [sampler](../../src/mm3/mm3_sample.hpp) | Model/drafter machinery and current verifier integration points. The [central contract](README.md#expert-mixing-and-initial-routingverification-settings) owns acceptance semantics. |
| Window/CPU donor | [Basalt AVX-512](../../../basalt/src/kernels/cpu/nvfp4_avx512.cpp), [draft policy](../../../basalt/src/spec/draft_policy.cpp), [suffix drafter](../../../basalt/src/spec/suffix_drafter.cpp) | Grouped decode and cost-policy references; Qwen priors are not MiniMax measurements. |
| Transfer donor | [staged device transfer](../../../Infernix-glm5-local/src/core/staged_device_transfer.cpp) | Host staging and handoff reference; its historical held experiments remain unrelated to MiniMax work. |

## DSpark and grouped service

The official five-layer DSpark uses target taps from layers **3, 17, 31, 45 and 59**.
Its seven-position block cannot consume taps from future target positions that
have not executed. Grouped verification can share loaded/decoded expert records
across candidate columns; its value depends on complete time and missing payload
per committed token. A larger expert union can erase the benefit of wider windows.

DSpark confidence estimates schedule verification; they are not target acceptance
probabilities or the actual proposal distribution. The existing confidence head
was trained for the original acceptance rule. Observed accepted lengths and full
round costs can guide useful widths under lenience without making head retraining
a prerequisite. The Fast-HSD source and all acceptance/rejection requirements are
linked from the central verifier contract, rather than repeated here.

## Historical stage design

A layer stage executes attention and dense work as well as experts, unlike an
expert helper. A contiguous split moves activations at stage boundaries; it is not
per-projection tensor parallelism. A plain split serializes a token across devices.
The tested pipeline instead used already available DSpark candidates: a 3090 prefix
could work on the next segment while the primary verified the current segment.
Rejection invalidated dependent work and required safe frontier/buffer retirement.

MiniMax's target has attention KV and sparse index state, unlike Qwen's recurrent
state. Dense weights, KV, scratch, handoff buffers and expert residents all affect
stage capacity. Fewer helper-owned layers can free expert capacity but also move
service and memory pressure to the primary. CPU/DRAM contention and discarded work
prevent treating potential overlap as free.

These mechanisms and the [measured pipeline outcomes](evidence.md#base-comparisons)
remain technical evidence. They do not activate a split, placement, drafter or
window sweep. The [central synchronization rules](README.md#synchronization-and-buffer-retirement)
govern any active shared state paths.

<a id="execution-design"></a>
<a id="existing-code-to-reuse"></a>
<a id="fast-hsd-acceptance"></a>
<a id="why-it-could-help-and-what-could-defeat-it"></a>
<a id="implementation-sequence"></a>
<a id="resident-test-and-memory-design"></a>
<a id="discriminating-experiments-and-stopping-decisions"></a>
<a id="exit-evidence"></a>

The [original proposal](history-2026-11-03-pre-consolidation.md#snapshot-proposal-3)
preserves the retired smaller-prefix/helper gates, calibration requirements,
serial/pipeline and drafter-placement sequences, memory alternatives and exit
checklists. Current work follows only the central restart and completion contract.
