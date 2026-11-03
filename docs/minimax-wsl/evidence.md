# MiniMax evidence and provenance

The [central plan](README.md) owns requirements, status and work orders. This file
records observations at their original scope; it is not a second tracker.
No model, benchmark or build was run for this documentation consolidation.

## Current reference

The engineering reference is the recorded Maya cooperative configuration on two
short coding tasks, not a high-effort mixed-workload or lenience measurement:

| Part | Recorded setting |
|---|---|
| Execution | CPU/5090/3090 complete-expert cooperation, native FP4 on the primary, live helper replenishment and worker dispatch |
| Residency | 438 calibrated primary residents plus eight working/staging slots; 741 helper records |
| Routing | Ready-GPU ranking bonus 0.05; original scores determine mixture weights |
| Speculation | Official DSpark, grouped historical SpecCascade/Fast-HSD cutoff; window 8, rho 0.2 |
| Other policies | Layer pipeline, next-layer predictor and lossless HOST codec off |
| Target sampling | Temperature 1, top-p 0.95 |

[Held M-110](/mnt/wsl/transfer/minimax-wsl/runs/cr-20261102/held-v2/summary.json) measured **379 scored emitted tokens /
25.944107 seconds = 14.608327 tok/s**, pooling four 128-token requests after separate
priming and a completed-round suffix cutoff. Its two coding tasks pooled 17.9921
and 12.2440 tok/s. An [earlier broad-policy trial](/mnt/wsl/transfer/minimax-wsl/runs/r4-20261102/queue-runtime/summary.json)
recorded 15.68 tok/s; it does not establish a stable higher rate. Original effort
labels and conditioning remain part of these receipts. New high-effort/lenience
source cannot inherit their quality or performance claims.

| Finding and receipt | Demonstrated scope and limitation |
|---|---|
| [Cache-prior routing](/mnt/wsl/transfer/minimax-wsl/runs/cr-20261102/held-v2/summary.json): 11.69 → 14.61 tok/s | Cooperation retained; broader quality unresolved. |
| [Native FP4/live-helper group](/mnt/wsl/transfer/minimax-wsl/runs/r4-20261102/helper-fp4/group-qualified-summary.json): 13.88 → 14.94 tok/s | Constituents were not isolated; multiplying this gain by the routing gain is unsupported. |
| [Compressed nonresident service](/mnt/wsl/transfer/minimax-wsl/runs/cr-20261102/host-allocation-diagnostic/scale-service/summary-v3.json): 26–29% slower | Matched component result; no isolated full-model compression penalty was measured. |
| Function/tool checks and reasoning failures | Function checks and simulated tool round trips passed; ordinary and approximate paths both failed a small reasoning fixture. This does not demonstrate acceptable reasoning quality or identify the approximation as the sole cause. [Stopped record](history-2026-11-03-stopped-plan.md) preserves raw references and conditions. |

Historical fixed-bank comparisons restored placement between arms, rather than
necessarily recalibrating before each prompt. They establish performance from
calibrated starting conditions. The [current measurement protocol](README.md#generation-timing-and-expert-cache-conditioning)
uses a different primary timing/conditioning contract.

## Base comparisons

| Runtime | Common composition, pipeline off | Same approaches plus three-layer pipeline |
|---|---:|---:|
| Maya | 14.6083 tok/s | 14.2600 tok/s |
| Strata | 14.4319 tok/s | 13.5164 tok/s |
| Infernix | Not measured | Not measured |

Sources: [Maya composition](/mnt/wsl/transfer/minimax-wsl/runs/r1-20261102/summary.json),
[base comparisons](/mnt/wsl/transfer/minimax-wsl/runs/bases-20261103/summary.json).
These are generation suffixes on two coding tasks, not a general base ranking.
The common compositions included cooperation, cache-prior routing and the old
DSpark/Fast-HSD cutoff. Maya's 1.21% advantage over Strata was inconclusive; old
incumbent bookkeeping did not select a base.

Infernix's 5.0931 tok/s ordinary CPU/helper-off compressed-storage control cannot
rank the required composition. CPU/helper-off pipeline cases near 6.3 tok/s were
also compressed, so they do not isolate cooperation's effect. Its cooperative
load was stopped before READY/inference, as recorded in the central
[handoff](README.md#historical-evidence-and-stopped-handoff).

## Adaptive residency evidence

The [combined calibrated comparison](/mnt/wsl/transfer/minimax-wsl/runs/r4-20261102/helper-fp4/group-primary-reassessment.json)
measured **15.4222 tok/s adaptive versus 15.4799 fixed** with FP4/live helper,
prior 0.05 and the old verifier. It established joint execution, not a speed win
or profile-free operation. Four adaptive receipts recorded 34–54 primary exchanges
each and zero `adaptive_reused`; exchanges paid incoming copies and outgoing preservation.

Source inspection found admission choosing the hottest missing expert before
checking copy reuse, at the target-forward boundary. Useful earlier uploads may
have left staging by then. This is a candidate/lifetime hypothesis for ordered
traces, not a diagnosed cause. The [central design](README.md#adaptive-residency)
owns the selected response to that evidence.

Upstream Maya GLM at `3a47bbe8883f61ed4724d4b43f9dd46f7703a733` has replaceable
main slots with resident/spare/landing/retiring lifecycle states and capacity
partitioned per layer. Exclusive-tier mode performs RAM→GPU promotions and GPU→RAM
demotions. It supplies a lifecycle reference, while the approved MiniMax design
uses globally movable primary capacity. Sources: [pool states](https://github.com/mw00/project-maya/blob/3a47bbe8883f61ed4724d4b43f9dd46f7703a733/src/core/glm_fast_state.hpp#L512),
[replacement/transfers](https://github.com/mw00/project-maya/blob/3a47bbe8883f61ed4724d4b43f9dd46f7703a733/src/core/glm_fast_path.cu#L5183).

| Local GLM configuration | Recorded warmed generation tok/s |
|---|---:|
| No demand landing; normal paid promotions | 22.6908 |
| No demand landing; paid promotions off | 21.2840 |
| Two conditional landing frames per layer; normal promotions | 23.4209 |
| Same landing; paid promotions off | 23.6136 |

The [promotion-off-only report](../../../Infernix-glm5-local/profiles/bench/glm5-speed-campaign-20261028/w9/w9-promotion-off-only-vs-control-full.md)
records a 6.200% loss over ten requests with matching outputs. The
[combined-versus-landing report](../../../Infernix-glm5-local/profiles/bench/glm5-speed-campaign-20261028/w9/w9-combined-vs-landing-only-full.md)
records a 0.823% gain with one differing request, limiting exact attribution.
The [local account](../../../Infernix-glm5-local/docs/maintainer/glm5_3-flash-implementation-plan.md#landing-promotion-separation)
also retains the near-neutral physical unified-pool result. Disabling paid
promotions is not equivalent to removing demand service or fixing the bank.
These findings motivate retention/refill mechanisms, not a transferable MiniMax
gain or imported GLM arithmetic/output contract.

## Parked mechanisms

- **Predictive prefetch:** [recorded trial](/mnt/wsl/transfer/minimax-wsl/runs/rl-20261102/summary-confirmed.json)
  14.0324 tok/s off versus 14.0442 on (+0.084%), with opposing task changes and
  substantial candidate variation; no useful gain demonstrated. The
  [implementation](../../src/mm3/mm3_native_prefetch.inc) excludes adaptive primary
  residency. It applies the next router to current normalized input, aggregates
  predicted top-four choices across a verification group and may upload one
  missing HOST expert to staging. It changes optional-copy priority as well as
  prediction. Its off arm retains normal admission/refill; prediction is separate
  from replacement, demand transfers, live helper service and DSpark drafting.
  Nonblocking copies can still contend with useful work.
- **Dampening:** the tested formula rescaled the entire routed contribution and
  could amplify it. It did not guarantee attenuation or measure reduced hidden-state
  drift. Timing changes may come from altered later routing, output and draft
  acceptance, rather than coefficient arithmetic. Its limitations do not reject
  every possible variant. [Original proposal/evidence context](history-2026-11-03-pre-consolidation.md#snapshot-central-plan).
- **Pipeline and inactive waits:** the tested pipelines supplied no demonstrated
  advantage in the base table above. Fully-primary-resident profile readbacks were
  inactive in the reference placement; first-prefill calibration was setup work.
  No recurring bottleneck or isolated gain was established for those waits or
  profile saving. Deferred retirement does not cancel already-running writes.
- **Device-queue dispatcher:** retained implementation/evidence did not establish
  a useful advantage; the [stopped record](history-2026-11-03-stopped-plan.md)
  preserves the experiment rather than assigning it new work.

## Weight geometry and source provenance

The initial official-header inspection observed packed U8 expert weights/scales,
MXFP8 dense/draft and BF16/F32 auxiliaries, including block/global weight scales
and input-scale calibration. It covered two target headers and one draft header,
not full integrity or a complete parameter-weighted inventory. Later shared-file
verification is recorded by the central [data section](README.md#shared-weights-and-data)
and [inventory](official-checkpoint.json); these are distinct observations.

Payload geometry is `57 * 128 * 3 * 6144 * 3072 * (36/64)` bytes = 216.421875 GiB,
or 30.375 MiB per expert, before official auxiliary metadata. At a 22-decimal-GB
helper payload budget this fits about 690 experts, versus 884 in the old EXL3
profile; actual packing/scratch determine capacity. Faster arithmetic can therefore
lose through reduced coverage. This does not establish a memory ceiling.

[Base provenance](base-provenance.json) records initial checkouts. The
[cooperative](proposal-1-cooperative-experts.md), [routing/residency](proposal-2-cache-aware-routing.md)
and [verification/stage](proposal-3-speculative-pipeline.md) source maps identify
reusable implementation. Donor versions and external measurements below retain
their dates and cannot be read as current HEAD or local MiniMax qualification.

## Earlier evidence index

The complete E1–E10 bodies are preserved in the pre-consolidation archive. Their
former repair, calibration and experiment instructions are superseded by the
central plan; the observations remain evidence at the stated snapshot.

| Record | Useful contents and limits |
|---|---|
| [E1: local artifacts/hardware](history-2026-11-03-pre-consolidation.md#e1-local-artifacts-and-hardware) | Dated GPU/CPU/WSL/driver/link observations; EXL3/GGUF sizes and precision. Header presence did not prove integrity; old quants were not retained as required arms. |
| [E2: local measurements](history-2026-11-03-pre-consolidation.md#e2-results-already-recorded-on-this-computer) | Windows helper/placement/fetch gains and losses, allocation evidence, GLM MTP and Qwen stage outcomes. Different models/OS/histories cannot supply a MiniMax forecast. |
| [E3: external measurements](history-2026-11-03-pre-consolidation.md#e3-other-users-measurements-and-what-transfers) | Maya, Xolotl, Strata, Basalt and FOMOE reports with hardware/precision/protocol qualifications; no numerical local ceiling. |
| [E4: codebase assessment](history-2026-11-03-pre-consolidation.md#e4-codebase-assessment) | Local commits and remote file blobs, actual helper scheduling differences and adaptation limits. Blob identities are not repository commits. |
| [E5: approximation lessons](history-2026-11-03-pre-consolidation.md#e5-approximation-and-speculation-lessons) | Withdrawn Strata prior, FOMOE original-demand recovery, asynchronous routing, fluent-but-worse outputs and MiniMax DSpark dependencies. |
| [E6: original Fast-HSD integration](history-2026-11-03-pre-consolidation.md#e6-selected-fast-hsd-acceptance) | Pinned cutoff code and sampled/residual integration points. Old support/cutoff tests do not validate the selected lenience path. |
| [E7: implementation review](history-2026-11-03-pre-consolidation.md#e7-implementation-review-2026-11-01) | Recomputed suffix rates, unmatched API/prefill totals, 161 parser tests, lifetime/state findings and historical absent listener. |
| [E8: timing/conditioning audit](history-2026-11-03-pre-consolidation.md#e8-generation-timing-and-helper-calibration-audit-2026-11-01) | Old token-32 boundary was not cache convergence and could exclude work for later counted tokens. Profile increments were 85.9% prefill-derived; automatic service calibration did not replace helper placement. [Machine-readable audit](generation-conditioning-audit.json). |
| [E9: inherited residency](history-2026-11-03-pre-consolidation.md#e9-inherited-primary-residency-and-local-comparison-evidence-2026-11-01) | Original fixed primary versus eight adaptive slots and the four-cell GLM retention/promotion evidence. |
| [E10: donor/executed-path audit](history-2026-11-03-pre-consolidation.md#e10-donor-fidelity-and-executed-path-audit-2026-11-01) | Actual source reuse, helper progress, empty jobs, DSpark ring/prefix defect and then-missing FP4 comparison. Historical repair findings are not a new repair queue. |

<a id="e1-local-artifacts-and-hardware"></a>
<a id="e2-results-already-recorded-on-this-computer"></a>
<a id="e3-other-users-measurements-and-what-transfers"></a>
<a id="e4-codebase-assessment"></a>
<a id="e5-approximation-and-speculation-lessons"></a>
<a id="e6-selected-fast-hsd-acceptance"></a>
<a id="e7-implementation-review-2026-11-01"></a>
<a id="e8-generation-timing-and-helper-calibration-audit-2026-11-01"></a>
<a id="e9-inherited-primary-residency-and-local-comparison-evidence-2026-11-01"></a>
<a id="e10-donor-fidelity-and-executed-path-audit-2026-11-01"></a>

Legacy evidence anchors lead to this index; use the corresponding archived record
for its full sources and original wording. Routine restart uses the central plan.
