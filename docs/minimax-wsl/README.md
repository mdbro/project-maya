# MiniMax M3.1 Flash on WSL2: central plan

**Execution remains paused. Consolidated 2026-11-03 at the user's request.**
Only documentation work is authorized: no implementation, builds, trace collection,
simulations or model runs until a new user instruction. Old runners stay stopped.

This file alone owns current requirements, decisions, status and completion across
all three bases. Follow the user's latest instructions and update this record when
an accepted decision changes. Proposal documents contain source maps and technical
background; [evidence](evidence.md) contains scoped findings and receipts.

<a id="objective-and-current-reference"></a>

## Objective and scope

First deliver the requested system in **Maya**, with focused residency optimization.
Build for one interactive request at a time across coding, prose,
summarization/extraction, reasoning and tools. Generation speed is the principal
performance objective; do not claim an optimum or invent a hardware ceiling.
The [later three-base comparison](#runtime-base-decision-remains-open) remains open
and deferred, and does not block or automatically follow this milestone.

**No practical speed improvement was demonstrated** in the stopped round:
approximately 15 tok/s at its start and end. Its repairs and component gains remain reusable.
The [reference and measured limits](evidence.md#current-reference) preserve the
actual configuration, historical rates and adverse results. A gain against a slower
control cannot stand in for improving the best usable complete configuration.

<a id="starting-and-resuming-work"></a>

<a id="maya-first-implementation-gap"></a>

<a id="current-decisions-and-work-status"></a>

## Current status and restart

Source/configuration and saved-result inspection on 2026-11-03 established:

| Area | Existing state and remaining work |
|---|---|
| Cooperative execution and cache-aware routing | Implemented and exercised; native FP4, worker dispatch, original-demand credit and live helper are reusable. No new executor or routing port is needed. |
| DSpark verification | Official drafter and grouped verification exist; defaults still use the old cutoff. Selected lenience integration and its affected checks remain open. |
| HOST storage | `compressed_host()` is false under `native_cooperative`; verify codec allocations remain absent in the delivered path. |
| Primary/helper adaptation | `helper_live=1`; primary adaptation supports cross-layer swaps, pointer rotation, completed-staging reuse and asynchronous publication, but defaults set `primary_adaptive=0`. |
| Profile-free start | Launch still supplies `--profile`; the primary loader supports round-robin initialization. Combined primary/helper startup and adaptation without a profile remain to be verified. |
| Residency optimization | Current high/low decayed-demand admission uses fixed thresholds without explicitly pricing transfer/eviction against service saved. The approved design below remains implementation work. |
| Serving and measurement | High-effort Python/prose harness, replacement reasoning fixture, current-source measurements and final integrated validation remain open. |
| Delivery | Working configuration, optimized residency and demonstrated speed gain are separate claims; the integrated faster usable engine is not delivered. |

The [launch](native-server.json) and [runtime](native-runtime.json) files describe
the calibrated reference, not an implemented adaptive/lenience default. This
consolidation does not change them or assign old rates to new source.

### Mandatory restart order

Start in `/home/mdwbr/project-maya`, branch `minimax-wsl/proposal-2`. After explicit
execution resumption:

1. Recover the existing implementation using this plan, its
   [handoff summary](#historical-evidence-and-stopped-handoff) and the
   [source map](proposal-2-cache-aware-routing.md). Inspect branch, worktree,
   binary/configuration and actual process/resource state. Archive reading is
   needed only for a specific unresolved historical question.
2. Make the [selected engine](#selected-engine) the working path: enable the
   existing adaptive-primary control, remove the frozen-profile dependency,
   integrate lenience and verify actual server settings, allocations and cache changes.
3. Implement and tune the [approved residency design](#adaptive-residency), using
   its ordered replay and the [short speed screen](#default-speed-screen).
4. Complete the [reasoning diagnosis](#approved-reasoning-check-and-bounded-diagnosis)
   and [delivery evidence](#execution-and-completion), including actual serving.

Apply the focused-check policy in [synchronization](#synchronization-and-buffer-retirement).
Completed historical repairs and unchanged checks do not become a fresh campaign
at restart or a context handoff.

<a id="requirements-shared-by-all-three"></a>

## Selected engine

These requirements apply to the Maya milestone and each later base candidate.

- **CPU/5090/3090 cooperation, cache-aware routing and fully adaptive residency
  are required together.** Assign complete expert computations to the devices;
  retain the native FP4/worker/live-helper route. Routing-off, fixed residency and
  ordinary decoding are targeted diagnostic controls, not eligible final designs.
  Tensor parallelism and CPU/helper-off candidates or performance controls are
  outside this work. A first adaptive implementation's weak speed result calls
  for improving its policy, not dropping the requirement.
- Specialize for Ryzen 9950X3D, RTX 5090, RTX 3090 and WSL2. Use observed links and
  capacities; assume neither P2P nor NVLink-class bandwidth. No portability/fallback
  framework or unrelated model setup.
- Use the [pinned official target and DSpark](#shared-weights-and-data), preserving
  supplied NVFP4 expert codes/scales, MXFP8 dense/draft formats and BF16/F32 tensors
  through layout changes. Keep persistent weights packed and conversions in bounded
  workspace. Retain every target layer and stored expert; substitution
  is not permanent pruning. Main text-model weights should average roughly 3.5 bpw
  or higher; about 3.4 is acceptable. Keep a parameter-weighted inventory with
  scales, metadata, vision and draft reported separately. No new quantization campaign.
- Choose fast packed-weight arithmetic independently on each device. Activation
  formats, reductions, rounding and approximations may differ; neither exact nor
  approximate cross-device expert-output agreement is required. Do not introduce
  a slow compatibility path for numerical matching. State/index/ownership
  correctness remains required, with no determinism requirement for text, routing
  or cache history.
- Use **uncompressed HOST storage**. The tested lossless codec is excluded; its
  reconstruction does not reduce PCIe weight bytes. Run only one full model at a
  time across bases, retain compatible weights for sequential checks, and account for
  actual GPU/DRAM allocations, staging, exchange space, scratch, KV and draft.
  Allocate only the selected execution routes.
  Favor no steady-state SSD expert service and report actual reads. Check current
  memory/backing-storage headroom before large allocations; no GLM memory cap or
  exact-output benchmark contract is inherited.
- Preserve efficient loading, local serving/streaming, reasoning and useful tools.
  Tolerant parsing and at most one corrective tool turn are allowed initially;
  report raw, repaired and failed outcomes and charge retry work. Vision/video
  execution and long-context qualification remain later work.

<a id="optional-next-layer-prefetch"></a>

<a id="dampening-disposition"></a>

### Parked work

| Mechanism | Disposition |
|---|---|
| Layer pipeline and further split/composition matrices | Keep disabled. Deferred retirement of its speculative buffers and its inactive waits are outside Maya-first. |
| Next-layer predictive prefetch | Keep `routing_prefetch` off; adapting it is not a prerequisite for residency work. |
| Substitution dampening | Keep off and outside Maya-first. Its inconclusive/limited evidence does not require testing another variant. |
| Device-queue dispatcher | Parked; no demonstrated advantage justifies further work now. |
| Old C/R/P factorial matrix and auxiliary gates | Retired. Unchecked historical rows do not activate experiments. |

[Evidence and limitations](evidence.md#parked-mechanisms) explain these dispositions;
reopening follows [investment policy](#choosing-worthwhile-work).

### Expert mixing and initial routing/verification settings

Keep the existing cache-prior routing as the starting policy. Rank experts using their sigmoid
routing strengths, the model's correction bias and an initial 0.05 bonus for ready
GPU-resident experts. Select four, then compute their mixture weights from their
original sigmoid strengths, normalized over the selected set. Preserve the model's
routed scaling factor of 2.0 and shared-expert contribution. The cache bonus affects
selection, not mixture weighting; this is not dampening. Keep four distinct valid IDs.
Only successfully published ready residents qualify; freeze the relevant pass
view to protect readers, without promising repeatability across cache histories.
RAM residency is not a free GPU hit. Record original demand, executed choices,
substitutions and useful backfills separately.

Keep official DSpark drafting and grouped target verification, but replace the
target-relative cutoff with **lenience-based verification**. For a proposed token
`x`, accept with probability `min(1, p(x) / (ell * q(x)))`, where `0 < ell <= 1`.
Here `p` is the configured approximate target distribution and `q` is the actual
distribution that produced the draft. Verification does not restore original top-4
routing or establish invariance across window widths. Keep target temperature 1
and top-p 0.95. For `ell < 1`,
this intentionally changes the sampling distribution; do not describe it as
distribution-preserving or as restoring unmodified-model quality.

Hold the declared routing and speculation settings steady while implementing and
comparing the approved residency design. Their inclusion is settled; this
clarification does not authorize another routing, speculation or cooperation
comparison matrix. The 0.05 cache bonus is an initial configurable setting, not a
permanent requirement or demonstrated optimum. **Start with lenience factor
`ell=0.4`**, configurable and held constant during residency comparisons. Treat
it as an initial setting, not a demonstrated optimum. **Do not reuse or translate
the old rho 0.2 as ell.** Address any implicated routing or verification issue
through the bounded quality diagnosis.

The integration must preserve the complete lenience algorithm, not change only
one comparison in the old cutoff branch:

- Use DSpark's actual proposal probabilities, including its sampling filters and
  normalization. The existing sampled-DSpark path is the implementation starting
  point. The current cutoff path uses argmax drafts; its softmax confidence is
  not the probability of that deterministic proposal. Do not substitute a
  confidence-head score or an unrelated raw softmax value for `q`.
- Commit the longest accepted prefix. On the first rejection, discard dependent
  draft state and draw the replacement from the normalized positive residual
  `max(p - q, 0)`, matching the selected lenience method. Do not retain the old
  cutoff branch's direct target draw on rejection. After full acceptance, draw
  the normal bonus token from `p`, subject to EOS/output limits.
- Validate the acceptance and residual calculation with focused distribution
  checks, including `ell=1` reducing to standard speculative sampling relative
  to the configured target and a case with excessive draft probability. Check
  affected actual rejection/continuation, cancellation and prefix-reuse callers;
  old cutoff tests alone do not validate the new route. This is not a cross-device
  arithmetic-equivalence or output-identity requirement.
- Expose the effective verifier and lenience factor through serving controls and
  diagnostics. Charge any proposal-probability, sampling and residual-draw work
  to complete generation timing. Update launch defaults only when the new route
  exists and passes its affected checks; keep old receipts/settings labelled as
  cutoff evidence.

Algorithm reference: Fast-HSD
[`lenience_accept_prob`](https://github.com/ZhouYuxuanYX/Fast-HSD/blob/66d0b731ac9e784ffaa2d7cf9b8f7da68ca4bd17/fast_hsd/core/collaborative_verification.py#L20)
and the lenience acceptance/residual branches in
[`_speculative_sampling`](https://github.com/ZhouYuxuanYX/Fast-HSD/blob/66d0b731ac9e784ffaa2d7cf9b8f7da68ca4bd17/transformers/generation/utils.py#L5537).

<a id="adaptive-residency-next-cache-work"></a>

## Adaptive residency

<a id="r4-evidence-based-architecture-comparison-and-adaptive-primary-residency"></a>

This architecture is approved; remaining work is implementation, tuning and
verification of achieved speed.
Build from [Maya's existing adaptive path](../../src/mm3/mm3_native_adaptive.inc),
including its bounded exchange storage, pointer rotation and nonblocking
publication. Donors supply useful mechanisms, not a required series of ports.

1. **One fully replaceable primary expert cache, with capacity movable across
   layers.** Keep existing physical allocations where practical, but manage expert
   placement globally. There are no permanently fixed main-bank identities or
   permanent per-layer quotas. Keep a small shared reserve for demanded uploads
   and exchanges, initially the existing eight working slots within the current
   total memory budget. Temporary protection for active readers/writers is not
   permanent residency.
2. **Retain useful demand uploads in their existing frames.** Decide whether to
   retain an expert while its uploaded copy is still available, rather than only
   at the end of the complete target pass. Reserve that frame when retention is
   worthwhile and the exchange budget permits it. For uploaded expert B replacing
   resident A, keep B in place, preserve A in RAM if it has no other valid backing,
   then change ownership mappings after the relevant operations finish: B's frame
   becomes resident and A's former frame becomes working storage. Do not upload
   or copy B again merely to change its retention role. Saving A is a distinct,
   potentially necessary transfer. Do not force retention of an unhelpful record
   to improve a reuse counter.
3. **Exclusive canonical storage with bounded temporary copies.** Continue
   distributing the uncompressed expert bank across RAM, the 5090 and the 3090.
   Do not introduce a complete RAM duplicate of GPU residents. Bidirectional
   transfers remain part of the design: minimize unnecessary exchanges and
   overlap necessary outgoing preservation and incoming fills. Keep every expert's
   last valid copy and all active sources protected until ownership can safely move.
4. **Additional refills based on expected execution time saved.** CPU- or
   helper-served experts may never produce a primary demand upload, so retaining
   uploads alone is insufficient. Use observed original-router demand, expected
   reuse and service cost for additional admissions, charging incoming traffic,
   outgoing preservation and interference with current work. An already-uploaded
   candidate has no additional incoming-transfer cost; one requiring a fresh
   upload does. Account for that difference before selecting an admission winner.
   Preserve original-demand-only credit: executing a cache-selected substitute
   does not manufacture demand credit. Background refill here uses observed
   demand and does not reactivate the parked next-layer prediction experiment.
5. **Adaptive helper residency with coordinated ownership.** Keep the existing
   3090 live replacement mechanism. Primary and helper admissions respect each
   other's pending transfers and active readers. Optional exchanges stay bounded
   and nonblocking; unavailable cache entries continue through normal cooperative
   service according to actual weight ownership.

No offline placement profile is required. Start from a valid declared placement
that fits the existing RAM/VRAM bank; this does not mean moving every expert to
RAM or starting with empty VRAM. An optional saved profile may improve startup.

<a id="independent-request-state"></a>

For independent requests, keep model weights loaded and learned expert residency
alive without manual reseeding. Give each fresh target and DSpark KV/context,
without conversation history or prefix/KV reuse between these requests.
Expert-weight residency and attention KV state are distinct.

The [donor evidence](evidence.md#adaptive-residency-evidence) explains the upstream
GLM lifecycle and its different per-layer capacity policy.

### Real-trace replay and optimization

**Use replay to tune the chosen design, not to reopen the architecture decision.**
Start with ordered traces from the [default screen](#default-speed-screen), under
its request-state and workload scope. Histograms alone cannot reproduce
admission or transfer scheduling. Record original demand and executed choices,
request/round/row and layer identities, available copies and pending transfers,
prefill versus generation, and the service observations needed to price decisions.
Include attempted speculative rows that are later rejected. Reuse instrumentation
and add only missing fields. If capture perturbs throughput, use it for replay
and turn it off for the actual speed comparison.

Compare the existing adaptive policy with the approved retention/admission policy.
Tune a small number of controls, such as aging, admission margin and refill budget.
Do not require separate implementations of every donor cache or a broad sweep
over unrelated replacement algorithms.

Hold total memory capacity constant, including working and exchange storage.
Charge incoming transfers, necessary outgoing preservation, occupied/unavailable
frames, CPU-memory contention, helper interaction and competing useful service.
Represent overlap without summing simultaneous work as if it were all exposed.
Where timing is unknown, use plausible ranges rather than building a general
scheduler simulator.

Rank estimated execution time; use hits, transferred bytes and replacement counts
to explain the result. Fixed-trace replay cannot prove final speed or quality:
changed residency can change routing, generated text and speculative acceptance.
Confirm useful changes in the actual combined engine with the short two-task
screen. Initial conditioning and costs remain explicit; the old calibrated
coding comparisons do not establish operation without calibration.

Removing mandatory calibration is an operational improvement; only measured
integrated throughput establishes an additional speed gain.

### Synchronization and buffer retirement

<a id="nonblocking-cache-publication-and-test-disposition"></a>

<a id="original-demand-promotion-and-retention"></a>

<a id="r2-cuda-failure-path-publication-and-retirement"></a>

<a id="r3-speculative-state-verification"></a>

<a id="required-helper-only-refresh"></a>

Preserve nonblocking optional cache publication: query completion without waiting
or spinning. Check submission and completion success before publishing; incomplete
or failed records stay unavailable and keep their storage while independent work
proceeds. Do not drain optional transfers to make cache contents repeatable.

Recycle storage only after all readers and writers finish. A failed stream join
or timeout does not establish safe release or permit fallback to absent HOST
weights. Collect and report worker errors after joining started work; preserve
bounded helper failure and safe ownership when completion is uncertain.
Required-data dependencies and safe teardown still need correct ordering.

Reuse existing ownership/completion machinery. Deferring reclamation permits
useful overlap only with independent work and storage; it does not cancel writes.
Change synchronization only for a concrete active-path opportunity, without a
general retirement framework. “Optional” describes the cache optimization, not
whether required expert computation can be skipped.

Retain correct target/DSpark rejection, continuation, cancellation and prefix/ring
reuse, including affected EOS/output limits, KV/indexer/context frontiers and
exclusion of discarded writes. Run focused checks for changed ownership, indexing,
arithmetic or state callers, using bounded injected failures where appropriate.
Reuse valid unchanged checks rather than expanding each change into a broad audit.
The existing same-process helper placement refresh serves declared measurement
resets; it is neither new implementation work nor a normal-operation calibration chore.

## Short experiments with resident weights

### Effort and harness

Evaluate MiniMax at **high reasoning effort or higher**. High is the default for
quality checks and both speed requests; a supported higher level may be declared.
Hold effort constant within comparisons. The user reports poor reasoning and
other output quality below high. Do not lower effort to shorten answers or inflate
rates; historical lower-effort receipts do not qualify high-effort behavior.
This is a model request setting, separate from coding-agent delegation.

Update the existing harness and fixtures on authorized resumption: remove
`native_screen.py`'s hardcoded `reasoning_effort='none'`, replace the quality
harness's `low`/`none` defaults and old reasoning fixture, and verify that
serving/templates honor the requested effort. Record effective settings. Listing
the protocol here does not implement it.

### Default speed screen

Use exactly two independent requests: **write actual Python code** and **write
prose for a non-coding task**, with prompts, order, sampling and short output
budgets declared before candidate/control comparison. The declared order is held
within a comparison; no follow-up, topic return or additional category is appended
automatically. Completing a speed task is unnecessary; longer generation needs a
specific unresolved question that a shorter check cannot answer. Do not force
filler, substitute favorable continuations or extend a run merely to finish a long answer.

Use the [independent-request state rules](#independent-request-state) for both.

Include generated reasoning and its work in timing/accounting; separate reasoning
and final-answer tokens when available. Inspect output for obvious problems and
label a stop before the final answer. Truncated output supplies speed evidence,
not a passed functionality check. Separate functional/cache-behavior checks need
a stated purpose. Use small tasks whose complete answers fit short budgets for
functionality, reusing speed outputs when adequate; the reasoning check has its
own completion budget below.

### Generation timing and expert-cache conditioning

<a id="qualification-of-existing-token-32-results"></a>

The primary metric is **whole generation after prefill** across the declared
screen, measured at normal completed-round boundaries. Include drafting, rejected
work, sampling, transfers and cache maintenance, without adding synchronization
for timing. Pool emitted counts over elapsed time, not averages of rates, and
retain individual task rates and outputs. Historical coding/suffix results remain
separately labelled and cannot be equated to this metric. Do not discard initial
or topic-switch adaptation costs.

Distinguish a loaded/hardware-primed engine from a cache trained on scored tasks.
Declare neutral setup generation, initial placement, source/settings, capacities
and sequence order. Do not fit residency per scored prompt, require fixed cache
contents or wait for stable routing. Normal ongoing adaptation is part of operation.
Automatic CPU/GPU service-cost calibration may remain: it measures execution costs,
not workload-specific placement. Saving demand counts does not itself apply placement.

Controlled arm-to-arm placement/policy resets establish declared starting conditions;
they are measurement setup, not normal-operation requirements. Each arm's requests
retain cache history. Reuse compatible model state across safe policy changes.
Include online refill in generation; report startup, setup placement, prefill and
end-to-end workflow latency separately. Those diagnostics do not enter the
current generation-speed ranking or reject a candidate on their own.

Start with the smallest representative screen that changes a decision, then one
confirmation with candidate/control order reversed for a promising or close result.
One further targeted
check is allowed for a specific unresolved anomaly, with its decision value recorded.
Do not repeat until gains or control movement reach a desired number. Large
unexplained variation prevents a close speed claim; keep the current usable setting
while parking an uncertain choice, without calling it superior.

<a id="bounded-reasoning-diagnosis"></a>

### Approved reasoning check and bounded diagnosis

Use the approved scheduling problem below with a changed condition in the same
request, under the [effort requirement](#effort-and-harness).
It replaces the three-bin arithmetic problem as the primary reasoning check.
Keep the old responses and failures as historical evidence, with their effort
and truncation conditions; do not rerun that fixture by default or use its
lower-effort failures to qualify high-effort reasoning. This quality check is
separate from the two-task Python/prose speed screen.

The prompt is:

Two workers are available from time 0. Each task needs one worker throughout its
execution. Tasks cannot be paused or split.

| Task | Duration | Prerequisites |
|---|---:|---|
| A | 3 | None |
| B | 2 | None |
| C | 4 | A |
| D | 3 | B |
| E | 2 | C and D |

Tasks C and D also require the same machine, so they cannot overlap. All other
tasks may overlap when workers are available.

1. What is the earliest time all tasks can finish? Give a feasible schedule and
   briefly explain why an earlier finish is impossible.
2. Now change only B's duration from 2 to 4. Find the new earliest finish, give
   a feasible schedule, and explain what changes.

Use concise prose and task start/end times. No code or JSON is needed.

Reference outcomes for the checker, not included in the model prompt:

| Scenario | Optimal finish | One valid schedule |
|---|---:|---|
| B takes 2 | 11 | A 0–3; B 0–2; D 2–5; C 5–9; E 9–11 |
| B takes 4 | 12 | A 0–3; B 0–4; C 3–7; D 7–10; E 10–12 |

In the first case the shared machine cannot start before time 2; its seven units
of work and E's two units give a lower bound of 11, attained by the schedule.
In the second case its earliest start is time 3, giving an attained lower bound
of 12. The optimal machine-task order changes from D/C to C/D.

Check completion time, schedule feasibility and minimum-time justification
separately for each scenario. Accept alternative valid optimal schedules.
Validate dependencies, worker capacity and machine exclusivity rather than
matching one expected answer string. Allow sufficient total generation budget
for high-effort reasoning to finish; the requested final response remains concise.
Do not inherit the old 192-token cap or silently force reasoning to close to fit it.
Truncation is a budget/completion limitation, not by itself an incorrect reasoning answer. The actual fixture/checker implementation
remains work for authorized resumption.

Address recorded reasoning failures through bounded diagnosis. Inspect the exact
prompt, expected answer, reasoning effort, output budget, raw reasoning/final text
and parser. Distinguish incomplete output from an incorrect completed answer.
Reuse existing evidence before requesting another generation.

If a model comparison is needed, use a small complete task and an ordinary-target
diagnostic control that retains cooperation, uncompressed HOST storage and device
arithmetic. Original expert selection and ordinary target decoding may isolate
the approximate-path group for this diagnosis; neither is an alternative final
design. This is a same-engine control, not an independently validated reference.
If both paths fail, investigate the shared cause; if only the approximate path
fails, isolate the implicated approximation. Do not run a compulsory combination
matrix or append these checks to every speed screen.

A failing control does not establish preserved quality. Switching to lenience
does not itself resolve the historical failure, and those old failures do not
establish that the new verifier fails. Record actual outcomes without requiring
universal reasoning perfection or expanding into a broad quality campaign.

<a id="preferred-options-when-performance-is-practically-tied"></a>

<a id="cost-observations-and-retained-implementation-checks"></a>

### Cost observations and selection

Use existing counters before new profiling. Separate exposed critical-path time
from overlapping CPU/helper/GPU service: CPU/helper service divided by CPU job
count is a composite, not a pure CPU-kernel measurement. Hits, acceptance and
replacement counts explain time; they are not optimization goals themselves.
Inactive paths and setup-only costs are not recurring bottlenecks without evidence.

Among implementations meeting the requirements, declare a preference before
measurement for practically tied results; do not default to incumbency. Roughly
2% is a practical closeness band, not a target for repeated testing. Operational
selection does not prove equal speed; noisy unresolved effects remain inconclusive.
Investigate a 5% task-class regression; a confirmed >10% regression blocks default
selection until addressed. Review sampled output/work differences before calling
a loss repeatable and report all task outcomes.

## Choosing worthwhile work

Before substantial implementation or a full-model load, record a short decision
here: measured cost or missing behavior, proposed mechanism, plausible complete-engine
benefit/costs, cheapest informative check, and what its likely outcomes would change.
Source, counters, traces and rough estimates can guide investment; they are not
measured speedups or mathematical bounds.

- For optional substantial work beyond required integration/residency, normally
  seek a credible opportunity near 10% or more in complete-engine speed, or necessary
  operating behavior. A cheap integrated 2–5% gain is useful but does not justify
  disproportionate infrastructure. These are priorities, not empirical limits.
- Reassess after meaningful results, integration steps, blockers or scope growth,
  by expected benefit and remaining cost. No fixed durations/deadlines or repeated
  permission questions at ordinary reassessment points.
- Screen linked changes as a group when a concrete mechanism connects them;
  isolate contributors after a useful result or for a real defect. An unexcluded
  interaction or generic possible synergy does not require another model experiment.
- Reassess an enlarged bank, compression, new executor or major port needed only
  to run a weak control. Optional allocation failure is not a new product requirement.
- Park weak branches on engineering judgment without claiming impossibility.
  Reopen only for new evidence, relevant implementation changes or a user request.
  Keep one implementation branch in focus; parallel agents must not launch
  independent optimization campaigns. Stop adding mechanisms when the measured
  problem has an adequate solution.

### Runtime-base decision remains open

<a id="base-choice-and-strategy-combinations"></a>

<a id="strategic-comparison-and-locked-gates"></a>

The deferred comparison gives **Maya, Infernix and Strata each an actual full-model
comparison of the selected design**. Match weights, workload, sampling, memory
budgets and measurement conditions. Base-specific scheduling/cache policies may
differ where that is the engineering advantage; state the differences. Exercise
each runtime's own request/state/scheduler ownership, not one engine behind three
wrappers. Reuse existing integrations and valid checks.

A repository name is not a cache architecture. [Existing base evidence](evidence.md#base-comparisons)
neither selects Maya by incumbency, eliminates Infernix from its ordinary control,
nor eliminates Strata from a tied common configuration. A build/integration blocker
leaves that comparison incomplete; reassess feasibility rather than silently
selecting another base. Only a new user scope decision can drop one of the three.

## Execution and completion

Report **working functionality, optimized residency and demonstrated speed gain
separately**. A gain requires comparable integrated measurements without material
task, state or resource regression. A switch being enabled, improved cache counters,
a build, plausible output or component timing is insufficient. The central record
for delivery must contain:

| Evidence | Required account |
|---|---|
| Actual system | The [selected engine](#selected-engine) running through the real server, working launch defaults, effective settings, starting/final source/configuration, loader, target/drafter, dispatch, controls, streaming/tools and affected state checks. |
| Residency | [Approved design](#adaptive-residency) operating without mandatory placement profiles or manual reseeding: retained state across topics, actual eligible-bank replacement, investigated cost, retained changes and complete-engine effect. |
| Measurement | [Matched short protocol](#short-experiments-with-resident-weights), absolute overall/per-task rates, all material gains/losses, and actual memory/cache/transfer accounting with useful overlap preserved. |
| Quality | Defensible reasoning-failure diagnosis and brief usable-task outcomes. A few passes do not qualify broad quality or make deliberate approximations “quality preserving.” |
| Handoff | When requested, observed endpoint/source, current process/resource state and remaining limitations. |

Incomplete required behavior leaves delivery incomplete even if an interim build
is faster. Maya-first completion does not finish the deferred base comparison.
Discarded controls, interactions and component attribution are not completion gates.
Use “parked/unmeasured” for deprioritized opportunities, “inconclusive” for unresolved
measured effects and “delivered gain” only for integrated evidence. An investigation
with no faster usable result is no speed delivery. Link raw receipts externally.

## Dedicated bases and work boundaries

<a id="dedicated-bases"></a>

| Repository | Branch | Role |
|---|---|---|
| `/home/mdwbr/project-maya` | `minimax-wsl/proposal-2` | First implementation milestone; central authority |
| `/home/mdwbr/Infernix-minimax` | `minimax-wsl/proposal-1` | Deferred base comparison; retain existing integration/component source |
| `/home/mdwbr/strata` | `minimax-wsl/proposal-3` | Deferred base comparison; retain integration/evidence |

These are independent repositories. Origins are the owner's forks; upstreams are
fetch-only with pushing disabled. Inspect branch/worktree before edits and work in
the assigned base. Cross-base edits require explicitly shared scope. Preserve local
changes, copied source deltas, attribution and applicable notices; never reset,
clean or discard work to make a checkout clean. Inherited product/setup documentation
is technical reference, not an alternative plan. [Initial provenance](base-provenance.json)
is historical, not an execution obligation.

`/home/mdwbr/Infernix-glm5-local`, other pre-existing Infernix worktrees and Windows
Maya are read-only donors, including files, Git state and running processes.
No separate current plan/status tracker belongs in another checkout.

### Regular commits

Create commits or push **only when requested**. The stopped round's standing local
commit authorization is superseded. A commit neither requires a new benchmark nor
turns incomplete evidence into delivery.

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

## Historical evidence and stopped handoff

<a id="stopped-work-handoff-2026-11-03"></a>

<a id="phase-2-exit-record-2026-11-02"></a>

<a id="review-follow-up-2026-11-01"></a>

<a id="r1-smaller-prefix-and-suffix-helper-decision"></a>

<a id="required-27-case-matrix"></a>

The stopped handoff recorded Maya execution source `d4fa863` plus later
documentation/configuration changes; later guards were not remeasured on that
configuration. Infernix `i-c1` was cancelled before READY/inference; its runner and
monitor were stopped and the PID-bound advancement script disabled. Prepared Maya
final-run scripts were never executed. Preserve that incomplete run.
Owned processes/listener were absent at the handoff; current state must be observed
on execution resumption, not inferred from this historical record.

The [evidence index](evidence.md) provides scoped results and source links.
The [stopped-round archive](history-2026-11-03-stopped-plan.md) and
[pre-consolidation snapshot](history-2026-11-03-pre-consolidation.md) preserve full
original wording, adverse outcomes and superseded queues. Consult them only for a
specific historical question; neither is routine restart reading or a work order.
Legacy anchors remain for incoming links and do not reactivate archived instructions.
