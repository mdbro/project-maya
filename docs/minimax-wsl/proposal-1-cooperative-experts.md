# Proposal 1 reference: cooperative expert service

The [central plan](README.md) owns requirements, status, restart order and
completion. This document maps reusable implementation and donor code; it does
not schedule Infernix work. See the central [base assignments](README.md#dedicated-bases)
and [deferred comparison](README.md#runtime-base-decision-remains-open).

## Implementation map

| Component | Source | Role |
|---|---|---|
| Official expert records | [record.cpp](../../../Infernix-minimax/src/ops/minimax_nvfp4_ffn/record.cpp) | Shared MiniMax NVFP4 record/layout support. |
| CPU and GPU expert service | [fast_cpu.cpp](../../../Infernix-minimax/src/ops/minimax_nvfp4_ffn/fast_cpu.cpp), [fast_cuda.cu](../../../Infernix-minimax/src/ops/minimax_nvfp4_ffn/fast_cuda.cu), [native_fp4.cu](../../../Infernix-minimax/src/ops/minimax_nvfp4_ffn/native_fp4.cu) | Complete MiniMax FFN routes; these are adaptations/new work in this effort, not an unchanged GLM executor. |
| Maya integration of shared service | [build sources](../../CMakeLists.txt), [primary expert adapter](../../src/mm3/mm3_native_experts.cu), [CPU adapter](../../src/mm3/mm3_native_cpu.cpp), [helper](../../src/mm3/mm3_native_helper.cu) | The exercised cooperative model path. Source sharing does not make the runtime bases interchangeable. |
| Infernix model/request ownership | [model compute](../../../Infernix-minimax/src/models/minimax/model_compute.cpp), [program](../../../Infernix-minimax/src/models/minimax/program.cpp), [context](../../../Infernix-minimax/src/models/minimax/context.cpp), [Engine instance](../../../Infernix-minimax/src/runtime/engine/minimax_instance.cpp) | Existing MiniMax integration retained for the later base comparison. |
| Infernix serving | [protocol](../../../Infernix-minimax/src/serve/minimax_protocol.cpp), [frontend](../../../Infernix-minimax/src/models/minimax/frontend.cpp) | Model-facing request and serving adapters. Source presence is not an observed serving qualification. |

## Model and service details

The target has 60 layers: three initial dense MLPs and 57 MoE layers, each with
128 experts and top-4 routing plus a shared expert. Hidden/intermediate widths are
6144/3072; attention has 64Q/4KV heads, Gemma-style norms, partial RoPE and sparse
index selection in all layers. Its SwiGLU-OAI uses alpha 1.702, limit 7 and an
up-branch offset. The inherited GLM clamped FFN and H<=4096/protocol assumptions
were not directly applicable.

Attention, shared experts and the output head stay on the 5090 in the cooperative
path. Routed service combines primary residents/staging, CPU work and helper
experts. Useful overlap depends on joint completion time, including packing,
conversion, transport and publication, rather than GPU utilization or isolated
dot-product speed. Packed blocks can be decoded once for several columns;
wide-matrix performance alone does not characterize interactive decode.

The payload calculation and coverage tradeoff are recorded in
[evidence](evidence.md#weight-geometry-and-source-provenance). Memory/arithmetic
rules are owned by the [selected-engine contract](README.md#selected-engine),
residency by the [approved design](README.md#adaptive-residency), and lifetime/failure
handling by [synchronization](README.md#synchronization-and-buffer-retirement).

## Donor map

| Donor | Source and relevant mechanism | Qualification |
|---|---|---|
| Local Infernix GLM runtime | [helper ownership/refill](../../../Infernix-minimax/src/models/glm5_next/execution/helper.h), [offloaded MoE](../../../Infernix-minimax/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu), [peer worker](../../../Infernix-minimax/src/ops/offloaded_sparse_moe/cpu/peer_executor.cpp), [residency](../../../Infernix-minimax/src/runtime/expert_offload/expert_residency.cpp) | Persistent worker/direct publication and lifecycle references; GLM geometry and arithmetic contracts are distinct. |
| WSL host allocation | [SelectiveHostBuffer](../../../Infernix-minimax/src/core/host_memory.h) and [implementation](../../../Infernix-minimax/src/core/host_memory.cpp) | CPU/primary accessibility and bounded helper refill primitives; historical allocation tests do not validate arbitrary MiniMax bank layouts. |
| Original Maya DeepSeek | [helper scheduler](../../src/ds4/ds4_model.cu), [mapped request kernels](../../src/ds4/ds4_kernels.cu) | `helper_window`, `helper_loop`, `helper_enqueue`, `helper_answer`, `hr_loop`: ordered whole-window jobs, mapped handoff and separate recent-slot refill. |
| Basalt | [NVFP4 AVX-512 dots](../../../basalt/src/kernels/cpu/nvfp4_avx512.cpp) | Multi-column packed decode; Blackwell-specific routes do not automatically apply to Ampere. |
| Bloomery | [qdot](https://github.com/midagedev/bloomery/blob/main/crates/qdot/src/lib.rs), [MiniMax body](https://github.com/midagedev/bloomery/blob/main/crates/gpu-minimaxm3/src/body.rs), [tier](https://github.com/midagedev/bloomery/blob/main/crates/gpu-minimaxm3/src/tier.rs) | CPU/layout/model references; source snapshots and limits are in the evidence archive. |
| Maya MiniMax | [model](../../src/mm3/mm3_model.cu), [kernels](../../src/mm3/mm3_kernels.cu), [DSpark](../../src/mm3/mm3_dspark.cu), [frontend](../../serve/frontend.py) | Shared model mathematics, tokenizer/template and serving semantics. |

The Windows MiniMax helper was a rough DeepSeek-derived port. Its measured
queue/fetch/allocation results remain [historical evidence](evidence.md#earlier-evidence-index),
not a reason to select its wrapper. Original DeepSeek and Infernix scheduling
mechanisms differ; their names do not establish which executes faster.

## Historical design

<a id="execution-design"></a>
<a id="existing-code-to-reuse"></a>
<a id="substantial-work-required"></a>
<a id="cheapest-useful-sequence"></a>
<a id="what-would-make-it-win-or-lose"></a>
<a id="exit-evidence"></a>

The [original proposal](history-2026-11-03-pre-consolidation.md#snapshot-proposal-1)
preserves the old kernel/port/calibration sequence, experiment tables and exit
checklist. Those instructions are retired. Current work follows only the central
[restart order](README.md#mandatory-restart-order) and
[completion contract](README.md#execution-and-completion).
