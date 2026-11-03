# Proposal 2 reference: Maya routing and residency

The [central plan](README.md) owns current requirements and work. The exact
[routing/mixing/lenience contract](README.md#expert-mixing-and-initial-routingverification-settings),
[adaptive design](README.md#adaptive-residency) and
[measurement protocol](README.md#short-experiments-with-resident-weights) are defined
there once. This document provides the implementation and donor map.

## Implementation map

| Area | Source | Role |
|---|---|---|
| Model, native loading and grouped dispatch | [mm3_model.cu](../../src/mm3/mm3_model.cu), [native host](../../src/mm3/mm3_native_host.cpp), [native experts](../../src/mm3/mm3_native_experts.cu) | Target composition, canonical weights and CPU/primary/helper execution. |
| Router | [resident router](../../src/mm3/mm3_resident_router.cu), [model kernels](../../src/mm3/mm3_kernels.cu) | MiniMax score/ranking domains, selected IDs and mixture weighting; Qwen softmax assumptions are not interchangeable with this router. |
| Demand and admission | [original-demand scoring](../../src/mm3/mm3_native_demand.inc), [adaptive primary](../../src/mm3/mm3_native_adaptive.inc) | Demand history and existing cross-layer primary exchanges. The observed reuse gap is in [evidence](evidence.md#adaptive-residency-evidence). |
| Helper residency | [live helper](../../src/mm3/mm3_native_helper_live.inc), [helper implementation](../../src/mm3/mm3_native_helper.cu) | Live replacement, execution and ownership coordination. |
| Publication and frame ownership | [slot publication](../../src/mm3/mm3_native_slot_publish.inc), [owner window](../../src/mm3/mm3_native_owner_window.inc), [placement](../../src/mm3/mm3_native_placement.inc) | Completion visibility, active readers and physical record ownership. |
| Drafting and verification | [serving loop](../../src/mm3/mm3_main.cpp), [sampler](../../src/mm3/mm3_sample.hpp), [DSpark](../../src/mm3/mm3_dspark.cu) | Existing sampled/cutoff branches, target/drafter state and residual draw integration points. |
| Frontend and controls | [frontend](../../serve/frontend.py), [launch](native-server.json), [runtime](native-runtime.json) | Public request/parser path and the recorded starting configuration; the central status table identifies pending defaults. |

The cooperative path retains attention, shared experts and the head on the primary
GPU; CPU and staging service remain available for misses. Helper residency can save
transfers yet overload the last-finishing branch, so a combined hit counter does
not explain its complete service cost. A device-cost adjustment is justified by
observed imbalance, not assumed superior to the starting readiness treatment.

## Routing donors

| Donor | Inspected source/mechanism | Adaptation boundary |
|---|---|---|
| Strata bounded substitution | [native_router.cu](https://github.com/Niko1221/Strata/blob/main/src/kernels/cuda/native_router.cu), `route_resident_k` / `native_route_resident` | Warp-wide resident search and rank controls; Qwen's 512/top-10 softmax geometry differs from MiniMax. |
| Strata cache prior | [route_prior.cu](https://github.com/Niko1221/Strata/blob/main/src/kernels/cuda/route_prior.cu), [verify.cpp](https://github.com/Niko1221/Strata/blob/main/src/core/verify.cpp) | Residency changes selection while original scores supply mixture weights. This and bounded substitution were alternative mechanisms, not automatically stacked bonuses. |
| FOMOE | [car.c](https://github.com/pmerolla/fomoe/blob/main/src/car.c), [active callers/backfill](https://github.com/pmerolla/fomoe/blob/main/src/gpu_kernels.hip) | Ratio candidates, distinct-ID selection, budgets, protected layers, optional damping and original-demand recovery. Its CPU policy and RAM/NVMe tiers differ from the selected CUDA execution. |
| BigMoeOnEdge | [RouterHook::apply_substitute](https://github.com/Helldez/BigMoeOnEdge/blob/cb78bfe0f1218ab350a1b7bcd3945c45eaf94595/core/src/moe/router_hook.cpp), [measurements](https://github.com/Helldez/BigMoeOnEdge/blob/cb78bfe0f1218ab350a1b7bcd3945c45eaf94595/docs/cache-aware-substitution.md) | Score-range-scaled resident reranking; its RAM/flash cache and host callback are reference mechanisms, not a local speed result. |

The [cache-conditional routing paper](https://arxiv.org/html/2412.00099v2) motivates
the mechanism; it does not establish MiniMax quality or speed. The
[earlier evidence](evidence.md#earlier-evidence-index) retains Strata's withdrawn
stronger-prior default, FOMOE's drift/backfill findings and BigMoeOnEdge's warning
that fluent prose can conceal worse task performance.

## Residency and helper donors

The [upstream GLM pool and local retention/refill evidence](evidence.md#adaptive-residency-evidence)
explain lifecycle and transfer mechanisms behind the approved design. The
[cooperative source map](proposal-1-cooperative-experts.md#donor-map) locates the
original DeepSeek window/mapped-handoff and Infernix worker/refill code.
Source reuse does not require reproducing every donor's architecture.

Existing EXL3 profiles and Windows allocation/calibration observations are
historical inputs. Matching layer/expert identities can make a profile a startup
hint; measured EXL3 coverage does not transfer to the larger NVFP4 representation.
The central [initial-placement rule](README.md#adaptive-residency) owns profile use.

## Historical design

<a id="execution-design"></a>
<a id="existing-implementations-to-reuse"></a>
<a id="minimax-specific-routing-rules"></a>
<a id="cache-and-placement-behavior"></a>
<a id="implementation-sequence"></a>
<a id="fast-decisions-and-rollback"></a>
<a id="exit-evidence"></a>

The [original proposal](history-2026-11-03-pre-consolidation.md#snapshot-proposal-2)
preserves lower-two/top-eight substitution screening, mandatory helper refresh,
calibrated fixed-bank comparisons and old exit instructions. They are retired;
the approved cache-prior/fully-adaptive design and current restart order govern.
