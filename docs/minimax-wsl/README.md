# MiniMax M3.1 Flash on WSL2: central plan

**Execution remains paused. Revised 2026-11-03 UTC at the user's request.**
This revision authorizes documentation work, not implementation, builds, trace
collection, simulations or model runs. Resume execution only on a new user
instruction. Do not restart an old runner or continue its remaining checklist.

**Outcome of the stopped round: no demonstrated practical speed improvement.**
We started near 15 tok/s and ended near 15 tok/s. Repairs, component gains and
completed comparisons are reusable work; they do not constitute faster inference.
The central decision between the three runtime bases also remains unfinished.
No base has been selected by the user; Maya is only an existing comparison reference.

This file alone owns current requirements, decisions, priorities and completion
across the three assigned repositories. The user's latest instructions take
precedence. The [stopped plan and complete evidence record](history-2026-11-03-stopped-plan.md)
preserves the previous plan verbatim after an archival notice. Its mandatory
cases, locked gates, repair sequences, commit authorization and launch instructions
are **superseded**, not a queue to finish. Proposal documents are technical
references; their old implementation/exit checklists do not activate work.

## Objective and current reference

**Compare Maya, Infernix and Strata to choose the runtime base**, then deliver a
faster, usable local MiniMax engine on this PC for one interactive
request at a time, across coding, prose, summarization/extraction, reasoning and
tool use. Generation speed remains the principal performance objective. Keep
startup, prefill and total request latency visible separately. Do not claim an
optimum or set an invented hardware ceiling.

**User decisions, 2026-11-03:** CPU/5090/3090 cooperation is a given, not a
selection question. Every candidate and delivered engine must use it, with
**cache-aware routing and fully adaptive residency**.
Optimize their implementation together; fixed residency and routing-off are
diagnostic controls, not alternative final designs. Use raw HOST storage, without
the tested lossless codec. Comparing all three bases is required. Reassess work
by evidence, expected benefit and remaining cost, without fixed time durations.

The engineering reference is the existing **Maya cooperative configuration,
roughly 15 tok/s on two short coding prompts**:

| Part | Reference to preserve |
|---|---|
| Expert execution | CPU, RTX 5090 and RTX 3090 cooperate; each executes complete expert computations |
| 5090 arithmetic | Native FP4 |
| Primary cache | 438 calibrated main residents plus eight adaptive staging slots |
| 3090 cache | 741 records, live replenishment, worker dispatch |
| Routing | A 0.05 ranking bonus for ready GPU residents; original scores determine mixture weights |
| Token speculation | Official DSpark, grouped verification, Fast-HSD; window 8 and rho 0.2 |
| Other policies | Layer pipeline off; predictive prefetch off; lossless HOST compression off |
| Sampling | Temperature 1, top-p 0.95; do not lower these to inflate acceptance |

The held M-110 reference measured **379 scored emitted tokens / 25.944107 seconds
= 14.608327 tok/s**, pooling four 128-token requests after separate priming and a
completed-round suffix cutoff. The two coding tasks pooled 17.9921 and 12.2440
tok/s. An earlier trial of the same broad policy recorded 15.68 tok/s; it does not
establish a stable higher rate. These are historical generation-suffix
results, not mixed-workload or whole-request rates.

The stopped handoff records Maya execution source `d4fa863` plus later
documentation/configuration changes. Later guards were not remeasured on the
selected configuration. The checked-in [launch configuration](native-server.json)
and [runtime defaults](native-runtime.json) describe the calibrated reference,
**not an implemented profile-free adaptive default**. Do not silently change them
during this planning revision or assign the historical rate to a new binary.

Useful evidence to carry forward:

| Finding | Engineering consequence and limit |
|---|---|
| Cache-prior routing: 11.69 to 14.61 tok/s with cooperation retained | Keep as the fast starting policy; broader quality remains unresolved. |
| Native FP4 plus live helper replenishment: 13.88 to 14.94 tok/s | Retain the group. Its constituents were not isolated; do not multiply this gain by the routing gain. |
| Tested three-layer pipeline supplied no demonstrated benefit; Strata common configuration measured 14.43 tok/s | Park the tested pipeline. The common configuration was inconclusive against Maya; this does not settle the runtime-base choice. |
| CPU/helper-off pipeline cases were around 6.3 tok/s | Poor complete configurations. They also used compressed HOST storage, so they do not isolate cooperation's contribution. |
| Compressed nonresident-expert service was 26–29% slower in matched component checks | Keep raw HOST storage in the selected path. No isolated full-model compression penalty was measured. |
| Adaptive main-bank implementation exists; calibrated coding comparisons were inconclusive | Reuse it. Neither a rejection of adaptation nor evidence for operation without calibration. |
| Function checks and simulated tool round trips passed; ordinary and approximate configurations failed a small reasoning fixture | Diagnose the shared failure. This is not evidence of acceptable reasoning quality. |

Receipts: [held routing comparison](/mnt/wsl/transfer/minimax-wsl/runs/cr-20261102/held-v2/summary.json),
[FP4/helper group](/mnt/wsl/transfer/minimax-wsl/runs/r4-20261102/helper-fp4/group-qualified-summary.json),
[earlier 15.68 result](/mnt/wsl/transfer/minimax-wsl/runs/r4-20261102/queue-runtime/summary.json),
[base comparisons](/mnt/wsl/transfer/minimax-wsl/runs/bases-20261103/summary.json),
[compressed service](/mnt/wsl/transfer/minimax-wsl/runs/cr-20261102/host-allocation-diagnostic/scale-service/summary-v3.json).
A narrower improvement against a slower control cannot stand in for improving
the usable starting engine.

## Starting and resuming work

Start the primary agent in `/home/mdwbr/project-maya` on
`minimax-wsl/proposal-2`. A future explicit execution instruction can be:
"Resume the revised MiniMax engineering plan. Compare all three bases with the
required cooperative, cache-aware and fully adaptive design, then pursue
integrated speed gains." The existence of this example
does not authorize execution now.

### Mandatory restart order

1. **Recover the three existing implementations.** Read this plan and the
   [stopped handoff](#stopped-work-handoff-2026-11-03); inspect source, branch,
   binary/configuration and actual process/resource state. Preserve all work.
   Reuse valid ownership/state tests. Build or recheck only what the relevant
   source needs; completed historical repairs are not a fresh repair campaign.
2. **Prepare the required common design for comparison.** Reuse the existing
   cooperative kernels, routing and adaptive implementation, completing only
   missing integration and material defects needed for each base's real runtime.
   Fix one small mixed workload and honest timing protocol. Use bounded diagnosis
   for the known reasoning failure, retaining cooperation in ordinary controls.
   Do not make one base's extended cache tuning, simulator, profiler or policy
   campaign a prerequisite for measuring the other two.
3. **Measure all three bases with the required design.** Run Maya, Infernix and
   Strata with cooperation, cache-aware routing, full adaptation, DSpark/Fast-HSD
   and raw HOST storage. Use the actual serving/runtime path, comparable hardware
   budgets and the mixed session. Maya may run first as an available reference;
   prioritize Infernix next because its useful configuration was never measured.
   Complete Strata's corresponding comparison. Sequence is not preference.
   Reuse compatible state and collect task outcomes and useful cost/trace data
   during these runs. Do not reload CPU/helper-off controls or resume the old queue.
4. **Choose the base, then optimize the required system.** Make the three-way
   decision from actual complete-engine performance, usable behavior, resources
   and remaining integration. Use real-trace replay and the measured costs to
   improve adaptive admission, routing, arithmetic, dispatch or speculative-work
   efficiency in the selected implementation. Share a necessary correction across
   candidates when it materially changes the base decision; avoid extensive tuning
   of one contender before the comparison exists.
5. **Integrate and deliver the gain.** Carry the promising change through the
   actual server, then check the final combination against the starting reference
   on the same declared workload. Finish affected serving/state/resource checks.
   If no gain survives, report no speed delivery and the next justified decision;
   do not substitute completed experiments for the objective.

The three-base comparison is required; optional optimization campaigns are not.
Combine useful observations in each loaded session. Reassess after each result;
do not replay unchanged tests at a context handoff.

## Current decisions and work status

| Item | Current disposition |
|---|---|
| Model/GPU execution | Paused by user; this revision does not resume it |
| Runtime-base selection | Required three-way comparison incomplete; no selected or retained base |
| Maya | One candidate with an existing comparison reference; no default selection |
| CPU/5090/3090 cooperation | Fixed requirement for every base and control; never an on/off selection factor |
| Cache-aware routing and fully adaptive residency | Required delivered behavior; select efficient policies, not whether to include these features |
| Lossless HOST compression | Excluded from the selected path; use raw storage |
| Tensor parallelism; CPU/helper-off candidates or performance controls | Outside the work queue; ordinary diagnostic decoding keeps cooperation |
| Three-layer pipeline; additional split/composition matrix | Parked; new work needs new evidence of competitive benefit |
| Infernix base | Required relevant comparison unmeasured; old routing-off/CPU-helper-off queue retired |
| Strata base | Partial existing evidence; required fully adaptive/cache-aware/cooperative comparison remains open |
| Device-queue dispatcher, predictive prefetch, tested dampening variant | Parked; no demonstrated advantage warrants more work now |
| Mixed workload, reasoning diagnosis and current-source measurements | Shared preparation/evidence for the three-base decision |
| Profile-free, fully adaptive primary residency | Required cache work remains open; use existing implementation and real traces |
| Integrated faster usable engine | Not delivered |

<a id="base-choice-and-strategy-combinations"></a>
<a id="strategic-comparison-and-locked-gates"></a>

### Runtime-base decision remains open

The stopped agent used a Maya configuration as an incumbent for its experiment
rules. That did not select or retain Maya as the project's base. The relevant measured compositions
already included CPU/5090/3090 cooperation, cache-prior routing and DSpark/Fast-HSD:

| Runtime | Common composition, pipeline off | Same approaches plus three-layer pipeline | Actual disposition |
|---|---:|---:|---|
| Maya | 14.6083 tok/s | 14.2600 tok/s | Comparison measurements available; no base selection |
| Strata | 14.4319 tok/s | 13.5164 tok/s | Common configuration inconclusive against Maya; no demonstrated benefit from the tested pipeline |
| Infernix | Not measured | Not measured | Useful cooperative configuration unranked when stopped |

These are generation-suffix results on the same two coding tasks, not a general
base ranking. The 1.21% Maya/Strata difference was treated as inconclusive and
the agent carried Maya forward under its old incumbent rule. That bookkeeping
choice did not complete the required base decision. Infernix's 5.0931 tok/s result was
an ordinary, CPU/helper-off, compressed-storage control; it cannot rank the
required cooperative/cache-aware/adaptive configuration. Its cooperative load
was cancelled before READY or inference. See the
[Maya composition results](/mnt/wsl/transfer/minimax-wsl/runs/r1-20261102/summary.json)
and [base results](/mnt/wsl/transfer/minimax-wsl/runs/bases-20261103/summary.json).

The ports share much model mathematics and expert service, while retaining
different request/state/scheduling machinery. A repository name is not a distinct
expert-cache architecture. Assess the actual mechanism and integration retained
by each base. Maya's existing result is a comparison reference while choosing the final home;
do not eliminate Infernix from its ordinary control or Strata from a tied common
configuration. The new required comparison is three complete implementations of
the agreed design, not the old matrix: **Maya, Infernix and Strata each receive an
actual full-model comparison with cooperation, cache-aware routing, fully adaptive
residency, DSpark/Fast-HSD and raw storage**. Match weights, workload, sampling,
memory budgets and measurement conditions. Base-specific scheduling/cache policies
may differ where that is the real engineering advantage; state those differences.
Exercise each base's actual request/state/scheduler ownership rather than invoking
one runtime through three wrappers. Reuse existing integrations and valid checks.

A build or integration blocker leaves that base's comparison unfinished; it does
not silently select another base. Reassess the concrete work needed if a blocker
changes feasibility. Only a new user scope decision can drop one of the three.
Do not use the optional-opportunity thresholds below to skip this central comparison.

### Choosing worthwhile work

The old C/R/P factorial matrix and its auxiliary obligations are retired. The
three-base comparison above remains required. Keep historical
results and limitations; do not require a full-model experiment merely because
an interaction cannot be rigorously excluded. A branch can be parked on an
engineering judgment without claiming that the architecture has been disproved.

Before substantial implementation or a full-model load, record a short decision
here: the measured cost or missing behavior; the proposed mechanism; its plausible
complete-engine benefit and costs; the scope of the cheapest informative check;
and what each likely result would change. Existing counters, source
inspection, traces and rough cost estimates can justify investment or deprioritization.
They are not measured speedups or mathematical performance bounds.

Judge magnitude against the **best usable complete configuration**. Recovering a
few percent of a configuration running at 6.3 tok/s cannot make it competitive with
15 tok/s. Uncertainty is a reason to seek a cheap discriminating observation, not
an automatic order to build another runtime.

Use these planning defaults; they are engineering priorities, not empirical limits:

- For optional substantial new work beyond the required three-base integration,
  normally seek a credible opportunity
  of roughly 10% or more in complete-engine speed, or necessary operating behavior
  such as removing mandatory calibration. A cheap, integrated 2–5% gain is still
  valuable; it does not justify disproportionate new infrastructure by itself.
- Reassess an uncertain branch after a meaningful result, integration step,
  newly exposed blocker or material expansion of scope. Judge expected benefit,
  remaining cost and the cheapest useful next action. Do not assign fixed time
  durations or wall-clock deadlines to the work, or repeatedly ask for permission
  at ordinary reassessment points.
- Screen linked changes as a group when their mechanism explains why they belong
  together. Isolate contributors only after a useful result or for a real defect.
  Generic "synergy is possible" is insufficient.
- If a candidate needs an enlarged bank, new compression, a new executor or major
  port just to run a weak control, reassess the candidate before building those
  accommodations. An optional experiment's allocation failure is not automatically
  a new product requirement.
- Stop a weak branch without proving impossibility. Keep uncertainty explicit.
  Reopen only for new evidence, a relevant implementation change or a user request,
  not because the next agent sees an unchecked historical row.

Keep one implementation branch in focus. Read-only source work may be delegated,
but multiple agents must not independently launch speculative optimization campaigns.

### Preferred options when performance is practically tied

Replace automatic incumbent retention with these **predeclared preferences**.
A practical tie is a decision supported by comparable measurements and no material
task/resource disadvantage; an observed difference inside +/-2% with large control
movement is simply unresolved. Do not demand a statistical-equivalence study.
These preferences apply among implementations meeting the required cooperative,
cache-aware and fully adaptive design. Routing-off and fixed-cache controls are
not eligible final alternatives, regardless of a tie.

| Comparison | Preferred option in an adequately supported practical tie |
|---|---|
| Optional predictive prefetch or separate background refill on/off | Off if adaptation and useful service remain effective; demand-driven admission stays available |
| Stage pipeline on/off | Off |
| Runtime bases | No preference by repository name. Prefer verified serving/state integration and lower remaining integration cost among complete eligible engines with adequately tied speed and task behavior. Missing performance evidence is not a tie |
| Other policies with comparable task outcomes | Prefer lower measured peak memory, then less separate optional refill traffic; if neither distinguishes them, retain the working setting |

These preferences are part of this revision, not choices left to the implementing
agent after a tie. Changing them requires a new plan decision before the comparison.

Native device arithmetic remains chosen for complete service speed; no
cross-device numerical-equivalence requirement is introduced. DSpark/Fast-HSD
remains supported functionality. Diagnostic ordinary decoding does not remove it.

For speed selection, retain the useful existing guardrails: investigate a 5%
task-class regression; a confirmed >10% regression blocks default selection until
addressed. Review sampled output/work differences before calling a loss repeatable.
Report all per-task outcomes. Roughly 2% is a practical closeness band, not a reason
to run repeatedly until a threshold is crossed. A substantial engineering effort
must earn its cost in the integrated result.

## Adaptive residency: next cache work

<a id="r4-evidence-based-architecture-comparison-and-adaptive-primary-residency"></a>

The intended behavior is that **every eligible primary resident can be replaced
during inference**, without requiring an offline expert-demand profile or explicit
between-prompt recalibration. Temporary reader/writer protection is compatible
with adaptation; permanently fixed expert identities are not full-bank adaptation.
Keep the live 3090 helper and CPU expert execution.

An optional saved profile may improve startup but must not be required. Start
without it from a valid, declared placement that fits the existing RAM/VRAM bank;
"profile-free" does not mean moving every expert to RAM or starting with empty
VRAM. Preserve learned placement across prompts, topic changes and returns.

Build from [the existing adaptive path](../../src/mm3/mm3_native_adaptive.inc),
which already has bounded exchange storage, pointer rotation and nonblocking
publication. Avoid rebuilding a generic cache framework or porting a whole engine
to acquire one mechanism. Donor implementations inform the design; their results
do not predict MiniMax speed.

The first coherent candidate should favor retaining useful weights from transfers
already required for computation, with cost-aware admission and bounded additional
refills when worthwhile:

- **Demand landing** retains an already-required upload as a resident copy.
  **Background refill** starts an additional upload for expected future use.
  Both are forms of admission/promotion. Turning off separate refill traffic
  does not mean adaptation is off. The GLM result supports this specific
  substitution of admission mechanisms, not arbitrary combinations.
- CPU- or helper-served misses may cause no primary upload. Landing alone cannot
  adapt the main bank from demand it never receives; price additional admissions
  by expected service saved and actual incoming/outgoing cost.
- Do not copy an uploaded record again merely to change its retention role.
  Account for preserving an outgoing expert when VRAM holds its last canonical
  copy. This is real traffic; do not assume a duplicate full RAM bank.
- Keep total VRAM, staging/exchange capacity and HOST memory explicit and matched.
  Protect active readers and pending transfers; optional refills must not stall
  generation. Finite transfer bandwidth and staging pressure matter more than an
  unmeasured assertion that bookkeeping is complicated.
- Credit the original router's top-four demand before cache bias/substitution.
  Executing a cached substitute does not manufacture demand credit for it.

### Real-trace replay and optimization

Histograms alone cannot evaluate recency, topic response or transfer scheduling.
Use ordered traces containing request/round/row and layer/expert identities,
original demand and executed selections, prefill versus generation, available
residency/readiness, and enough transfer/service timing to price the decision.
Record attempted speculative rows, including later rejected work. Use existing
instrumentation; add only the fields needed. If trace capture perturbs throughput,
use it for policy replay and turn it off for the final speed comparison.

Compare a small set: the frozen reference; existing adaptive policy; simple
recency/aged-frequency admission; and retaining demand uploads with cost-aware
additional admissions. Combine equivalent policies rather than enumerate every
decay, threshold and cache size. Tune on some task sequences and assess on held-out
sequences with topic changes/returns. Hold total physical capacity constant.

Charge incoming and necessary outgoing bytes, transfer occupancy, CPU-memory
contention, unavailable slots, helper interaction and displaced useful residents.
Model overlap explicitly without summing overlapping durations as if all were
exposed. Where timing is unknown, use plausible ranges and show sensitivity;
do not expand this into a scheduler simulator project.

Rank estimated generation time; report hits, traffic, evictions and adaptation
delay as explanations. A fixed-demand replay cannot predict the new routing,
text and speculative acceptance induced by a changed cache. It shortlists a
policy; a small actual-engine comparison must decide performance.

Use two clearly labelled starting conditions only where they answer distinct needs:
the saved-profile reference for continuity, and profile-free startup followed by
continued mixed use for the desired operation. Do not reset or reapply calibration
between requests in the latter. Include early adaptation costs and topic-switch
costs in its recorded generation intervals.

Fully adaptive residency and cache-aware routing are required, including when a
particular first implementation fails to improve speed. Improve or replace that
implementation rather than declaring the feature optional. The existing fixed-cache
configuration remains a usable development reference, not an eligible final design.
Removing mandatory calibration is an operational improvement; only measured
integrated throughput establishes an additional speed gain.

### Optional next-layer prefetch

The separate `routing_prefetch` experiment predicts an upcoming layer's expert
demand before that layer's actual router input exists. It applies the next
layer's router to the current layer's available normalized input, aggregates the
predicted top-four choices across a verification group, and can upload one missing
HOST-owned expert into an existing 5090 staging slot ahead of use. It changes
optional-copy priority as well as adding prediction; the off arm keeps ordinary
cache admission/refills.

It is optional because the prediction may be wrong and normal expert service is
available. Incomplete copies are not published as ready, and generation must not
wait merely to finish the prediction. Transfers can still contend with useful
work even without a blocking wait. This is distinct from required adaptive
residency and from DSpark token drafting; disabling this predictor does not disable
cache replacement, demand transfers or the live helper.

The full recorded trial was 14.0324 tok/s off versus 14.0442 on (+0.084%), with
opposing task changes and substantial candidate variation: no demonstrated useful
speed gain. The implementation also explicitly excludes adaptive primary residency,
so it is not compatible as-is with the required final configuration. Leave it off;
do not turn adapting this predictor into a prerequisite for adaptive residency.
Evidence: [implementation](../../src/mm3/mm3_native_prefetch.inc) and
[complete comparison](/mnt/wsl/transfer/minimax-wsl/runs/rl-20261102/summary-confirmed.json).

## Short experiments with resident weights

### Mixed workload and task outcomes

Use a small fixed sequence with roughly 0.5–4K-token inputs and short responses,
usually 128–256 output tokens where adequate for a completed task:

1. The existing log-summarization coding task.
2. A bounded prose explanation with an explicit relevance/coherence rubric.
3. Summarization and structured extraction from supplied facts.
4. Short reasoning with independently checked expected answers, including the
   known transfer-account fixture and one different simple problem.
5. The existing dependency-ordering coding task.
6. A declared tool call, a labelled simulated tool result and the resulting answer.
7. Return to the earlier prose domain with new content.

This is seven workflows, normally eight requests because the tool workflow takes
two turns. Freeze exact text, order, sampling, budgets and checks before timing.
It is a smoke workload, not a comprehensive quality benchmark. Retain the old
coding tasks for continuity without letting them determine all policy choices.

Use the same generated outputs for speed and task inspection. A truncated code
prefix is not a passing function; a short tool response is functionality evidence,
not a sustained-throughput selector. Set enough budget for reasoning/structured
output before comparison; diagnose truncation separately from wrong completed
answers. Do not inflate token counts with forced filler or substitute favorable
continuations.

### Generation timing and expert-cache conditioning

<a id="qualification-of-existing-token-32-results"></a>

Measure emitted tokens and complete generation wall time at normal completed-round
boundaries, including drafting, rejected work, sampling, transfers and cache
maintenance. No new synchronization solely for measurement. Pool counts over
time, not an average of rates. Preserve individual task rates and outputs.

The new primary comparison uses **whole generation after prefill** across the
declared mixed session. The historical completed-round suffix metric can be
reported separately for continuity; it cannot be directly equated to this metric.
Do not discard first-request or topic-switch adaptation costs to improve the rate.

A loaded/hardware-primed engine and an expert cache trained on the scored tasks are
different conditions. Declare any neutral setup generation and initial placement.
Do not fit residency to each scored prompt, require fixed cache contents, or wait
for stable routing. Normal ongoing adaptation is part of the system's behavior.
A frozen generation profile is a historical control condition, not a universal
prerequisite for performance claims.

Reuse compatible model state across policy changes at safe boundaries. Each
comparison arm starts from the declared profile/history condition; requests within
an arm retain their cache history. Record source/settings, capacities and sequence
order. Include online refill costs in generation; show startup, setup placement,
prefill and end-to-end workflow latency separately.

Start with the smallest representative screen that can change the decision,
then one reversed-order confirmation for a promising or close result. If a
specific anomaly remains, allow one targeted resolving check and record why its
outcome matters. Do not rerun until control movement or gains meet a desired number.
Large unexplained variation prevents a close speed claim; it does not order
unlimited measurements. Keep the current usable setting while an uncertain
choice is parked, without calling that setting superior.

### Bounded reasoning diagnosis

Both the selected approximations and an ordinary control failed the historical
fixture. Preserve the failure; a failed control cannot certify quality preservation.

Inspect the exact prompt/template, requested reasoning effort, token budget,
raw reasoning/final answer, parser and expected result first. For the account
fixture the expected values are A=6, B=4, C=9, total=19. Distinguish the earlier
truncated JSON from the later completed but incorrect JSON.

If an actual comparison is needed, keep cooperation, raw HOST storage and device
arithmetic and use original expert selection with ordinary target decoding as
the diagnostic reference. This disables routing modification and token speculation,
**not CPU/helper execution**. It is a same-engine diagnostic, not an independently
validated reference implementation. Compare selected behavior on the same complete
task, reusing adequate saved evidence.

If ordinary decoding still fails, investigate the shared prompt/serving/model path
with the smallest distinguishing check; selectively consult reference computation
if necessary. If only the approximate path fails, isolate the relevant approximation.
Do not launch a compulsory factorial suite or broaden into a general quality study.
Independent speed work may proceed, but usable delivery cannot rest on dismissing
the repeated semantic failure as "both fail." Resolve or explain it with evidence
sufficient to judge practical task behavior; one occasional model mistake is not
a requirement to achieve universal task perfection.

### Cost observations and retained implementation checks

Use the reference session's existing counters before adding profiling. Separate
exposed critical-path time from overlapping CPU/helper/GPU service. In particular,
CPU/helper service divided by CPU job count is a composite, not a pure CPU-kernel
measurement. Hit rate and speculative acceptance are explanatory metrics, not goals.

Do not optimize inactive paths: the selected stage pipeline is off, and the cited
fully-primary-resident profile readbacks do not run in the recorded placement.
First-prefill calibration is setup work. No recurring bottleneck or isolated
speed benefit was established for those waits or for profile saving.

<a id="nonblocking-cache-publication-and-test-disposition"></a>
<a id="original-demand-promotion-and-retention"></a>
<a id="r2-cuda-failure-path-publication-and-retirement"></a>
<a id="r3-speculative-state-verification"></a>
<a id="required-helper-only-refresh"></a>

Retain the repaired production invariants: original-demand-only credit; successful
submission and completion before publication; nonblocking optional completion
queries; storage protected until readers/writers retire; bounded helper failure;
correct target/DSpark rejection, continuation, cancellation and prefix/ring reuse.
The same-process helper placement refresh already exists. Use it only when a
declared experiment needs a placement reset; do not make it a real-world calibration
chore or reimplement it.

Run focused checks for changed ownership, indexing, arithmetic or state callers.
Reuse valid unchanged checks; do not turn each optimization into a broad audit.
Optional work may remain pending; required data dependencies and safe teardown
still need correct ordering.

## Requirements shared by all three

- Cooperative CPU/5090/3090 execution, cache-aware routing and fully adaptive
  residency are required delivered behavior. Optimize them as a combined system.
  Routing-off, fixed residency and ordinary decoding may be targeted diagnostic
  controls; they do not redefine the selected design.
- Specialize for Ryzen 9950X3D, RTX 5090, RTX 3090 and WSL2. No portability/fallback
  framework or unrelated model setup. Use actual link/capacity observations;
  do not assume P2P or an NVLink-class interconnect.
- Use the pinned official MiniMax-M3.1-Flash and official DSpark companion.
  Preserve supplied NVFP4 expert codes/scales, MXFP8 dense/draft formats and
  BF16/F32 tensors through layout changes. Main text-model weights should average
  roughly 3.5 bpw or higher; about 3.4 is acceptable. Keep a parameter-weighted
  inventory, reporting scales, metadata, vision and draft separately. No new
  checkpoint/quantization campaign; vision execution remains outside this stage.
- Choose fast packed-weight arithmetic independently on CPU, 5090 and 3090.
  Different activation formats, reductions, rounding and approximations are allowed;
  neither exact nor approximate cross-device expert-output agreement is required.
  Do not use a slow compatibility path to make implementations numerically match.
- Keep all target layers and stored experts; cache substitution is not permanent
  model pruning. Keep temperature 1/top-p 0.95 for speculative comparisons.
  Fast-HSD intentionally relaxes acceptance within target sampling support;
  do not claim distribution-preserving sampling or unmodified-model quality.
- Preserve efficient loading, local serving/streaming, reasoning and useful tool
  calling. Tolerant parsing and at most one corrective tool turn are allowed for
  the initial workflow; report raw, repaired and failed outcomes and charge retry
  work. Vision/video and long-context qualification remain later work.
- Keep one full model resident at a time across bases; retain compatible weights
  for sequential tests. Account for real GPU/DRAM allocations, canonical ownership,
  staging, exchange space, scratch, KV and draft. No hidden duplicate full banks.
  Use raw HOST storage. The tested lossless codec adds reconstruction cost without
  reducing PCIe weight-transfer volume and is excluded from the selected path;
  do not reintroduce it to accommodate an unwanted control.
- No determinism requirement for text, routing or cache history. Never block
  generation merely to finish optional refills or make a benchmark repeatable.
  State/index/ownership correctness remains required despite permissive arithmetic.
- Favor no steady-state SSD expert service and report actual reads. Verify current
  memory and backing-storage headroom before large allocations; no GLM-specific
  cap or exact-output benchmark contract is inherited.

## Dedicated bases

| Repository | Assigned branch | Current role |
|---|---|---|
| `/home/mdwbr/project-maya` | `minimax-wsl/proposal-2` | Candidate with an existing comparison reference; home of the sole central authority |
| `/home/mdwbr/Infernix-minimax` | `minimax-wsl/proposal-1` | Required candidate; useful cooperative composition not measured in the stopped run |
| `/home/mdwbr/strata` | `minimax-wsl/proposal-3` | Required candidate with partial comparison evidence; tested pipeline parked |

These are independent repositories. Origins are the owner's forks; upstreams
are fetch-only with pushing disabled. Follow each checkout's entry point and this
central authority. Cross-base edits belong only to explicitly shared work.
The [base provenance record](base-provenance.json) describes their initial source
states, not new execution obligations.

The original `/home/mdwbr/Infernix-glm5-local`, other pre-existing Infernix worktrees
and the Windows Maya checkout are read-only donors, including their Git state
and running processes. Preserve copied source deltas and attribution. Do not reset,
clean or discard work. The proposal designs and evidence note remain subordinate.

## Sub-agent delegation

Carried over from the [GLM delegation guidance](/home/mdwbr/Infernix-glm5-local/AGENTS.md#sub-agent-delegation). This section owns the policy for all three MiniMax bases.

Use only **GPT-6.1 Terra** sub-agents. Create every sub-agent with a fresh context:
`model="gpt-6.1-terra"`, `fork_turns="none"`, and an explicit `reasoning_effort`.
Supply the task and relevant context yourself. Do not inherit the full conversation: that also
inherits the parent's model and effort instead of the required Terra selection. Never use Ultra.

These effort choices are starting recommendations, not measured Terra results on MiniMax:

| Effort | Decision difficulty and appropriate assignments |
|---|---|
| low | Small, fully defined, read-only work whose instructions determine nearly every decision: find call sites or extract specified log fields. Low must not edit code or documentation. |
| medium | Default for well-defined, routine work with clear boundaries and an established approach: fix a Python reporting tool, implement a specified CLI change, complete a bounded app workflow, apply exact documentation edits, summarize a supplied diff, or make straightforward visual/text corrections. Use medium or higher for edits. |
| high | Default for substantive Terra assignments involving interacting constraints and implementation judgment: scoped C++ changes across callers/tests, loader/runtime bug tracing, resource-ownership review, benchmark reconciliation, documentation updates from verified evidence, or reproducible local bug diagnosis. |
| xhigh | Rare, carefully bounded assistance with subtle numerical or concurrency questions: CUDA stream/event ordering, host/GPU lifetimes, quantization and cast boundaries, independent-oracle design review, or explaining a performance regression from traces and code. The primary agent must independently review the reasoning and evidence. |

Choose effort by the difficulty of the decisions and the consequences of missing something,
not task length or diff size. A fixed protocol may need medium; a twenty-line CUDA synchronization
change may justify xhigh. More effort cannot replace missing traces, representative measurements
or an independent oracle, and subtle Terra conclusions must not be trusted blindly.

Delegate sparingly, weighing the expected benefit against the primary agent's context-preparation,
review and repair time. Give each assignment a clear outcome, relevant context, constraints and
appropriate acceptance checks. Keep a lower effort when it already produces a complete, correct
result. If Terra repeatedly struggles despite adequate context, do the work in the primary agent
rather than automatically escalating to xhigh or max. Do not delegate tasks with lengthy waits,
such as compiling code or running models.


## Shared weights and data

| Item | Location |
|---|---|
| Official revision | `2f33a8b9ae49ce7b51c994df97166a7fd059bc2e` |
| Official target and DSpark | `/mnt/wsl/transfer/minimax-wsl/checkpoints/MiniMax-M3.1-Flash` |
| Existing generated layouts | `/mnt/wsl/transfer/minimax-wsl/artifacts/proposal-{1,2,3}` |
| Large captures and receipts | `/mnt/wsl/transfer/minimax-wsl/runs/` |

The [official inventory](official-checkpoint.json) records the verified shared
download: all 73 files, 252,446,157,672 bytes including DSpark and non-executed
vision assets. Recheck readiness before loading; do not launch upstream installers,
download another model, or make per-repository full copies as a setup shortcut.
The old Unsloth/GGUF copy and Windows artifacts are historical sources only.

Keep large assets outside source repositories. Check transfer filesystem space,
existing VHDX allocation and Windows backing-drive headroom together before new
large data. Do not delete unrelated models/evidence or repeat the old superseded
copy. Reuse existing packs where compatible.

## Execution and completion

Delivery means the selected **usable server** is demonstrably faster than its
starting configuration under comparable conditions, with no material task,
state or resource regression. Report the exact workload and absolute rates;
mixed-session and historical coding-suffix figures remain distinct.

For the next delivery, the central record must contain:

1. **The completed Maya/Infernix/Strata base decision:** actual comparable full-model
   evidence with the required cooperative/cache-aware/fully adaptive design in
   all three, actual runtime integration and a stated selection rationale. Missing
   results are unfinished work, not evidence for choosing the first implementation.
2. Starting and final source/configuration, real effective execution, a short
   matched mixed-workload comparison and all material per-task gains/losses.
3. Actual final integration: loader, target/drafter, cooperative dispatch,
   runtime controls, streaming/tools, and affected cancellation/reuse/state checks.
4. Actual memory/cache/transfer accounting and preserved useful overlap. Hit-rate
   replay, component timing, passing builds and plausible output are supporting
   evidence, not emitted-token speed delivery.
5. A defensible account of the reasoning failure and brief usable task outcomes.
   Broader quality remains unqualified; do not label deliberate approximations
   "quality preserving" from a few passes.
6. Cache-aware routing and fully adaptive residency actually enabled in the
   delivered path: no-profile operation, retained state across topics, measured
   speed/cost and actual eligible-bank replacement. If unfinished, final delivery
   remains incomplete; a faster interim build does not waive these requirements.
7. An observed handoff of the selected endpoint/source when requested, with
   current process/resource state and remaining limitations.

The required three-base decision replaces the old matrix. There is no requirement
to finish its discarded controls, interactions or component attribution. Preserve
a small decision record here for new work; link raw receipts
externally. Use "parked/unmeasured" for a deprioritized opportunity, "inconclusive"
for unresolved measured effects and "delivered gain" only for integrated evidence.
A completed investigation with no faster usable result is **no speed delivery**.

### Regular commits

Preserve all existing work. Under the user's current checkout instructions,
**create commits or push only when requested**. The stopped round's standing local
commit authorization is historical and does not authorize a commit of this revision.
A commit does not require a new benchmark or turn incomplete evidence into delivery.

## Historical evidence and stopped handoff

<a id="stopped-work-handoff-2026-11-03"></a>
<a id="phase-2-exit-record-2026-11-02"></a>
<a id="review-follow-up-2026-11-01"></a>
<a id="r1-smaller-prefix-and-suffix-helper-decision"></a>
<a id="required-27-case-matrix"></a>

The [complete stopped record](history-2026-11-03-stopped-plan.md) preserves the
old rules, source identities, measurements, adverse outcomes and their original
wording. It is historical evidence only. In particular:

- [Phase-2 repairs and replacement measurements](history-2026-11-03-stopped-plan.md#phase-2-exit-record-2026-11-02)
  need not be repeated without an affected change or concrete defect.
- [Pipeline and later base work](history-2026-11-03-stopped-plan.md#b0-runtime-shortlist-and-integration-declaration-2026-11-03)
  explains completed results, not an obligation to finish the old matrix.
- [Stopped handoff](history-2026-11-03-stopped-plan.md#stopped-work-handoff-2026-11-03)
  records that Infernix `i-c1` was cancelled before READY/inference. Its runner
  and monitor were stopped; the old PID-bound advancement script was disabled.
  Preserve the incomplete run. The prepared Maya final-run scripts were never
  executed. None is a resume instruction.
- At that handoff the owned processes/listener were absent. This is a historical
  observation, not a claim that current process/resource state was checked during
  this documentation revision.

The legacy anchors above remain for incoming links. They no longer activate
archived cases or instructions. If another repository's entry text still says
"locked comparison" or "regular commits authorized," follow this current plan
and the user's latest instructions.
