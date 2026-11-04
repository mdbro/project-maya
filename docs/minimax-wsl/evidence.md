# MiniMax evidence and provenance

The [central plan](README.md) owns requirements, status and work orders. This file
records observations at their original scope; it is not a second tracker.
The initial documentation consolidation ran no model, benchmark or build.
The subsequent Maya implementation and scoped measurements are recorded below.

**Qualification correction, 2026-11-04:** the implementation report called the
active Maya milestone delivered. That overall completion claim is withdrawn by
review; the [central work status](README.md#active-maya-work) and
[completion contract](README.md#execution-and-completion) own the remaining work.
The measured sequential-start policy improvement and passing scoped checks remain
valid at their recorded scope; they do not establish warmed policy selection.
The speed arms held lenience at 0.4 and the cache bonus at 0.05; they do not perform
the required value optimization of either, or close the other active parameter
families in the [central optimization scope](README.md#required-parameter-optimization).
No comparison below establishes a gain against the previously usable engine under
the new protocol. Raw receipts, including their original delivery filenames and
claims, are preserved rather than rewritten.

## Placement and residency audit

**Qualification correction, 2026-11-04:** the later cold helper screen is retained
in [cold-summary.json](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/helper-static-adaptive/cold-summary.json)
and cannot select warmed helper behavior. The same initialization limitation
applies to using the sequential-start priced-versus-legacy comparison below as
warmed policy optimization. Its measured rates and counters remain observations;
selection under a representative learned state is unresolved. The central
[conditioning gate](README.md#placement-and-residency-qualification-correction)
owns the remedy and order of work.

History inspection confirms `4e8792c` removed both `--profile` and `--profile-out`
from `native-server.json`, while changing the runtime to adaptive primary and
sampled lenience. The removed startup input was
`restart-20261102/calibration-maya/generation-profile.bin`. Its presence in the old
recipe does not by itself qualify that profile's workload/effort or warmed state.
Main-bank adaptation was introduced in `e4f37d1`; sequential no-profile placement
and 0.95 aging predate it. There is no measured isolated speed regression assigned
to removing the profile.

Inspection of the pre-baseline implementation established the following:

- [Demand scoring](../../src/mm3/mm3_native_demand.inc) credits original routed
  expert rows, including prefill, and lazily ages by target-forward count.
  Executed cache substitutes do not create original-demand credit.
- [Legacy primary admissions](../../src/mm3/mm3_native_adaptive.inc) use the two
  exchange descriptors and thresholds 1.8 and `incoming > 1.25 * victim + 1`.
  Primary/helper admissions require `generation_active`; demand accounting does
  not. Replacement continued through the end of the retained cold trace rather
  than ending at a programmed warmed fraction.
- The priced policy uses horizon 20, a provisional half-CPU1 saved-service price
  and 2 ms incoming/outgoing prices. Its bounded cold-trace replay and actual
  sequential-start comparison do not validate these choices for warmed operation.
- [Existing placement control](../../src/mm3/mm3_native_placement.inc) clears
  live scores, adaptive-resident provenance and ordinary staging, then ranks
  primary before helper. [Generation-profile controls](../../src/mm3/mm3_model.cu)
  save aggregate generation counts only. Neither restores a learned policy state.
- Learned lane service estimates can affect both cooperative assignment and the
  priced admission calculation. A matched learned-state comparison must account
  for those estimates as well as expert identities and demand scores.

The declared system review and prospective repairs are not new speed evidence.
Earlier raw traces, partial detailed captures, outputs and adverse results remain
unchanged; no historical missing events are reconstructed as observations.

### Anchor-first staging audit

Read-only reconstruction on 2026-11-04 matched the visualizer's 163-staged anchor
to coding request 3, forward 112, input position 1068 (`q`). Across 57 MoE layers,
its 228 executed selections comprised 26 primary residents, 38 helper residents,
163 newly staged selections and one CPU selection: 163 of 164 cache misses went
through staging. The following four rows had 51, 22, 20 and 14 staged selections.
The complete five-row forward uploaded 209 distinct expert records, each
31,850,752 bytes, for 6,656,807,168 bytes (6.20 GiB). Multiple rows can use one
upload; adding their staged-selection counts would overstate transfer volume.

The source's staging candidate scan walks rows in order and takes the first
eligible distinct HOST experts up to the chosen count. The anchor therefore gets
first choice. Reconstructed uploaded identities matched the corresponding prefix
of that candidate list and the `demand_uploads` count in all 9,405 captured layer
calls. The split estimate uses distinct-expert count with per-expert service prices;
it does not account for each candidate's number of users in the verification batch.
This establishes an assignment-order weakness, not a measured optimal upload count
or a speedup from any proposed replacement.

Sources: [raw routes/service records](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/residency-trace.jsonl),
[compiled visualizer data](/home/mdwbr/infernix-viz-ports/expert_cache_viz_minimax/data/delivery-trace.json),
and [`lane_split` / `window_post`](../../src/mm3/mm3_model.cu).
The capture is the already qualified cold/adapting legacy run; it does not describe
the new baseline required by the revised plan. The central
[implementation priority](README.md#implementation-first-then-measurement-and-tuning)
owns the response and work order.

## Current reference

The 2026-11-03 Maya implementation checkpoint used `419f1c2` plus working source captured in the
[build manifest](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/v2/source.json).
The live engine/frontend files and binary were checked against that manifest at
handoff. Binary SHA-256 is
`d7750c6cf0acf209c30285238d17ebb41db3aca7c37ae1ecbd22217dbff37016`.
The shared native expert service came from Infernix `5c0176343ecdae858794ad2b8625389e997427d1`
([file hashes](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/v2/shared-service-source.json));
no held Infernix or Strata runtime was advanced. Existing documentation edits
were preserved; this work was not committed or pushed.

### Matched integrated measurement

The [declared protocol](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/v2/comparison-protocol.json)
uses Python then prose, sequentially, 128 emitted tokens each, high effort,
temperature 1/top-p 0.95, official sampled DSpark/lenience 0.4, up to four draft
proposals plus the anchor (request `spec=5`),
confidence floor 0.3, ready-cache prior 0.05, native FP4/CPU/worker-helper cooperation
and live helper. Both arms have adaptive primary residency: the control uses the
old threshold policy; the candidate uses priced retention/refill and protected
per-layer publication. Main/helper capacities and all selected allocations match.

Each arm starts from the original profile-free round-robin/complementary roster.
A small roster-only reset file restores those identities between arms; it is
measurement setup, not a learned placement profile or production dependency.
Startup cost calibration and a neutral basil generation prime hardware. Each
scored request has fresh target/DSpark attention context, while the second retains
expert residency learned during the first. No scored-prompt placement fitting,
per-task reseeding, suffix discard or completion waits were added for timing.
Traces were disabled. All emitted reasoning/final tokens count; the API does not
report separate reasoning-token counts. Samples end at their output limit and
are speed evidence, not completed-function passes. Natural text and acceptance
can differ with adaptive routing; the effect is a complete-engine effect, not an
isolated copy-cost measurement.

| Policy order | Legacy adaptive | Selected priced adaptive | Change |
|---|---:|---:|---:|
| Legacy then priced | 10.5528 tok/s | 11.5372 tok/s | +9.33% |
| Priced then legacy | 10.8455 tok/s | 11.3334 tok/s | +4.50% |
| Both orders, pooled counts/time | 512 / 47.863285 s = 10.6971 | 512 / 44.777231 s = 11.4344 | **+6.89%** |
| Python, pooled | 10.3889 | 10.9931 | +5.82% |
| Prose, pooled | 11.0243 | 11.9126 | +8.06% |

[Summary and all receipt references](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/v2/speed-summary.json).
The control and candidate moved between repetitions; all four task comparisons
favored the candidate. This short screen does not establish universal speed or
an optimum. It cannot be compared directly to the earlier roughly-15 tok/s
coding suffix results, whose workload, effort, verifier and conditioning differ.
That limitation leaves the [starting-engine comparison](README.md#comparison-with-the-usable-starting-engine)
open. The 10.6971 tok/s control is the legacy residency policy inside the new
high-effort/lenience/profile-free setup, not a measurement of the previous usable
configuration. All positive per-task percentages in this table have that same scope.

Prefill remained about 50.0–50.1 seconds for the 1,056-token coding prompt and
8.4–8.5 seconds for the 240-token prose prompt. Candidate end-to-end API times
were 61.5–61.9 seconds and 19.2–19.3 seconds respectively. Startup and placement
resets are recorded separately; neither enters the generation ranking. The
large prompt cost is not hidden by the post-prefill result.

### Residency behavior, adverse results and replay

The first implementation kept both exchange descriptors until the next complete
forward. It reused every priced admission in its prose request but issued no fresh
primary refill. Its [matched pair](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/first-cost.json)
regressed 10.7945 → 10.1316 tok/s (6.1% loss). That result is preserved.

The selected correction publishes completed exchanges at existing joined layer
boundaries, protecting both incoming and victim identities from the current
layer's already-frozen routing view. Incomplete transfers retain their leases
without waits. It retains eight working frames/two simultaneous exchanges and
caps admissions at eight per forward, reserving one budget unit for an end-of-forward
opportunity. Reservation does not promise a free descriptor when copies are still
pending. The coldest eligible resident can come from any MoE layer; no permanent
per-layer quota is introduced. Original demand alone earns credit.

Across the four scored requests per arm:

| Counter | Legacy | Selected |
|---|---:|---:|
| Issued primary admissions | 340 | 1,016 |
| Retained staging uploads | 0 | 945 |
| Additional fresh primary fills | 340 | 71 |
| Issued optional H2D payload | 10.829 GB | 2.261 GB |
| Issued preservation D2H payload | 10.829 GB | 32.360 GB |
| Required demand staging copies | 25,600 | 23,528 |
| CPU expert jobs | 51,913 | 47,563 |
| Admission-policy host time | 25.7 ms | 78.6 ms |

These are engine counters, not isolated PCIe critical-path measurements.
Necessary outgoing preservation increased; fewer demand uploads and CPU jobs
accompanied the gain. All eight measured requests recorded zero Linux process
`read_bytes` deltas and zero expert disk counters. This does not measure Windows
pagefile/driver activity or establish universal zero disk traffic.

The [ordered trace](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/residency-trace.jsonl)
records original and executed row/layer identities, speculative attempted rows,
prefill/generation, available ownership, staging/pending exchanges and composite
CPU/helper service observations. The initial legacy-replay ranking was withdrawn:
it undercounted actual exchanges, first 60 and then 110 versus 168 observed.
The corrected tool uses the measured control as baseline and a fixed helper
roster for the counterfactual, with transfer/interference ranges and no claim of
exact scheduler replay. Source-matched
[budget 8](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/replay-frozen-helper-v2-h20-b8.json)
estimated +0.22 to +1.79 seconds (adverse); budget 16 had only a small estimated
advantage over 8. The actual sequential-start engine result determined the provisional setting;
that decision now requires warmed revalidation. V3's fixed-seven
early limit was unselected; implemented early allowance is budget minus one.

Pricing uses decayed original demand, horizon 20, an initial saving estimate of
half measured CPU1 service, and 2 ms per incoming/outgoing record. A ready upload
pays no additional incoming charge. These cost estimates are explicit heuristics,
not measured isolated service savings. Together with the corrected replay's
adverse estimate, they leave policy economics and the adequacy of the chosen tuning
open for the central optimization work; the local speed result alone does not
establish that the complete optimization scope is finished. The helper retains its existing live
exchange policy and protects leased identities; helper-only canonical records
become primary candidates after returning to HOST through that exchange.

### Functionality and bounded reasoning account

[Quality summary](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/v2/quality-summary.json)
and [raw public-server workflows](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/v2/quality.json)
preserve code, reasoning, tools, streaming and cancellation outcomes.

- Two independently generated `count_runs` functions completed and passed five
  functional checks, including empty inputs, transitions, strings, `None` and
  no input mutation.
- One streamed `read_file(path="notes.txt")` call read an actual isolated fixture;
  the response correctly reported version 7.4.2. Parse outcomes were one raw,
  zero repaired, zero failed, zero retries. Complete tool workflow took 56.9 s.
- Disconnect during reasoning was observed as cancelled, followed by successful
  service returning `READY`. The earlier actual lenience state check forced a
  first rejection, continued afterward and reused a matching 221-token prefix.
- The scheduling attempt at high effort/2,048 tokens ended before a final answer.
  It remains incomplete, not a failed completed answer. One declared completion
  check with a 6,144-token limit ended naturally after 1,613 tokens (110.3 s API,
  96.2 s generation). Its [reviewed schedules](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/v2/reasoning-checked.json)
  attained 11 and 12, passed dependency/worker/machine checks and gave valid lower
  bounds for both. Sampling/cache history differed, so shorter output is not
  attributed solely to the larger budget.
- A separate completed prose/extraction request preserved the six supplied facts
  about library closure, returns, digital access and reopening, with no added
  policies or promises. It used 177 tokens.

The [historical reasoning review](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/v2/historical-reasoning-review.json)
checked exact prompts, expected 6/4/9/19, raw reasoning/final text, effort, budgets
and parsers. Both nonblocking-demand receipts were low effort and capped at 192;
selected output contained a wrong complete JSON object followed by an unfinished
self-correction, while ordinary JSON was incomplete. Another saved low-effort
pair completed with wrong numerical fields. The checker was not inventing those
errors. Neither these shared failures nor the new pass isolates a routing,
verification or arithmetic cause. The retired bin fixture was not rerun. These
are bounded usability outcomes, not broad reasoning qualification or proof that
lenience preserves the original distribution.

### Resources, validation and serving

[Final ownership/allocation report](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/v2/final-placement.json):
438 primary + 741 helper + 6,117 HOST = all 7,296 routed experts. HOST physical
payload is 194,831,049,984 bytes (181.45 GiB), with no full RAM duplicate of GPU
residents. The eight primary working frames total 254,806,016 bytes; two primary
HOST exchange records and two helper exchange records total 127,403,008 bytes.
Target/DSpark share the resident HOST embedding. Primary FP4 arena is 1,149,184
bytes; helper pool is 23,601,217,536 bytes plus 2,706,432 bytes device scratch.
Codec, device-queue and predictive-prefetch allocations report zero, consistent
with the selected-path source guards. The final exclusive-ownership/frame audit
passed after all workflows.

The [engine startup allocation ledger](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/v2/gpu-allocation-map.json)
charges 27,481,663,488 primary weight bytes, 644,874,240 target KV/index bytes,
44,133,280 activation bytes and 2,372,317,980 drafter/tap bytes, alongside the
working pool above. Its remaining device use is accounted as unclassified
runtime/driver/display space (2.337 GB primary, 1.339 GB helper), not silently
assigned to a weight route. This ledger is marked `start`; the separate final
GPU snapshot below is current at the recorded handoff.

At the recorded handoff the 5090 used 30,447 MiB and the 3090 23,132 MiB, with
1,740/1,191 MiB reported free. Effective HOST physical headroom was 33.72 GB;
WSL available memory was 44.13 GB. These are snapshots, not capacity guarantees.
[Precision inventory/provenance](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/precision-and-provenance.json)
accounts for 426,186,370,176 text parameters: 4.1713 weight-only bpw and 4.6622
including scales. Draft and unexecuted vision are separate. All 73 pinned
checkpoint files were present at their recorded sizes; no second weight copy or
new quantization was created.

Focused CPU distribution, frontier, ownership/failure, budget, API-control and
harness/replay tests passed. A tiny actual CUDA HOST-VMM test exercised held
optional copies, independent readers and pointer rotation; it uses stand-in
byte records and is not a numerical expert-output oracle. Actual model ownership,
state and functionality checks supply separate integrated evidence.
[Handoff](/mnt/wsl/transfer/minimax-wsl/runs/maya-delivery-20261103/v2/handoff.json)
records the resident idle Maya process, source/default matching and local endpoint.
Vision/video execution, long-context qualification and held base comparisons were
not performed.

### Routing-policy provenance

The November 3 approved starting routing policy remains the additive cache prior.
In the [actual router](../../src/mm3/mm3_resident_router.cu), `Policy::Prior` adds
the configured absolute bonus to sigmoid-plus-correction-bias ranking scores for
ready primary/helper experts, then selects four from all 128. The separate
`Policy::Substitute` route considers replacements for the lower two selected
ranks among the original top eight; it is not stacked with the prior. Both use
original sigmoid strengths for the normal selected-set mixture. The
[candidate defaults](native-runtime.json) select `prior` and `routing_prior_bonus=0.05`;
`routing_ratio=0.8` does not control that selected prior path.

The approved algorithm replacement was the draft-token verifier's cutoff to
lenience change. The time-priced retention/refill change selects resident weights,
not routed expert IDs. The score-range-scaled BigMoeOnEdge mechanism in the
[donor map](proposal-2-cache-aware-routing.md#routing-donors) was a reference, not
an implemented or selected replacement for the additive prior. Neither the
historical routing-off comparison nor the new residency-policy comparison
establishes an optimal routing algorithm or bonus. Current requirements and
remaining decisions are owned by the [central plan](README.md#implementation-optimization).

## Stopped-round reference

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
- **Device-queue dispatcher:** the DS4-style mapped, sequence-tagged request
  comparator was implemented and measured in the model. Its
  [runtime comparison](/mnt/wsl/transfer/minimax-wsl/runs/r4-20261102/queue-runtime/summary.json)
  recorded 15.6764 tok/s worker versus 15.2350 queue, with 4.87% worker movement.
  The outcome was inconclusive, worker was retained and that named investigation
  ended. Older archive paragraphs calling the worker comparison open predate
  this result. It neither establishes a universal worker advantage nor requires
  another port of a differently named donor; reopening follows the central policy.

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
