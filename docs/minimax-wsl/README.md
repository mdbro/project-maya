# MiniMax M3.1 Flash on WSL2: central plan

**2026-11-07: the final configuration beats the preserved best usable engine by +24.40% at 128 tokens and +28.08% at 384, confirmed across three load pairs with no regression ([result](evidence.md#final-comparison-against-the-preserved-best-usable-configuration)). The serving defaults now carry the spinner and prior bonus 0.20. The predictor [did not qualify](#speculative-decision-validation), and per-pass work remains open, so Maya optimization is not complete.**
**Central optimization objective, user direction stated repeatedly: the CPU lane and the copy-plus-primary path finish
together on each individual pass, every activation through every individual layer.** Compare completion times from a
common origin, never durations, and report per-layer mean abs(Δ) between paths and between each pass's slowest and
second-slowest path. [Details](#per-pass-balance).

The user lifted the work/GPU pause for the Maya milestone: implementation, builds,
trace collection, simulations and model runs may proceed under this plan.
[Infernix and Strata base work is on hold](#base-holds). Old runners and retired
experiment queues remain superseded; follow the current restart order.

This file alone owns current requirements, decisions, status and completion across
all three bases. Follow the user's latest instructions and update this record when
an accepted decision changes. Proposal documents contain source maps and technical
background; [evidence](evidence.md) contains scoped findings and receipts.

<a id="objective-and-current-reference"></a>

## Objective and scope

Complete the requested system and performance work in **Maya**. The active scope
includes the selected engine, all five approved residency-design requirements,
[implementation optimization](#implementation-optimization),
[parameter optimization](#required-parameter-optimization), and final serving and
performance qualification. Enabling the features and obtaining a first policy
improvement are intermediate results. Performance investigation covers material
costs in the active cooperative execution, residency, routing and drafting paths;
it is not limited to tuning existing flags or the first primary-cache heuristic
an agent implements.

Build for one interactive request at a time across coding, prose,
summarization/extraction, reasoning and tools. Generation speed is the principal
performance objective; do not claim an optimum or invent a hardware ceiling.
The [completion contract](#execution-and-completion) governs the whole active scope.
The [three-base comparison](#runtime-base-decision-remains-open) remains unfinished
and subject to the base hold; it does not block or automatically follow this milestone.

**No practical speed improvement was demonstrated** in the stopped round:
approximately 15 tok/s at its start and end. Its repairs and component gains remain reusable.
The [stopped reference and measured limits](evidence.md#stopped-round-reference) preserve the
actual configuration, historical rates and adverse results. A gain against a slower
control cannot stand in for improving the best usable complete configuration.

<a id="starting-and-resuming-work"></a>

<a id="maya-first-implementation-gap"></a>

<a id="current-decisions-and-work-status"></a>

## Current status and restart

### Timing-model replacement

**User decision, 2026-11-06: “this timing-model approach needs to be replaced.”**
Retire the factorized per-operation timing reconstruction as the active route to
optimization. Its probe-based full-work comparator missed the measured control
by 80.2% and resolved none of the three prospective intervention decisions.
The earlier foreground refit, its separate 5.57% miss and every adverse receipt
remain preserved. Further calibration or qualification of this timing approach
is no longer the next task or a prerequisite for its replacement.

Retain the useful state replay, shared native policy entry points, original-demand
accounting, ownership/capacity checks, instrumentation, fixed-work harness and
repaired validation principles. Preserve the retired timing code and frozen
outputs as historical references; do not select runtime parameters from them or
resume their improvement as the default workstream. The selected inference
engine and cache-aware top-four routing remain in scope. Admission learning must
still credit original-router demand only.

**Replacement selection is open.** The [six collected proposals](timing-predictor-proposal-measured-transitions.md)
contain two primary-agent proposals and one each from Terra, Fable, Opus
and sol-6.1. The user supplied or requested the external proposals; do not contact
or spawn their authors. Compare all six before choosing the implementation.
This is a design comparison, not an instruction to build six predictors.
Consolidation does not validate the submissions' claims. No replacement is
selected, implemented or qualified by this decision.

The earlier [joint boundary-response regression design](timing-predictor-proposal-measured-transitions.md#primary-agent-proposal-1)
is one candidate reference. Its readiness heads, state representation and large
pilot are not the next implementation queue. The user now requires the bounded
[diagnostic study](#shared-model-diagnostic-study) before selecting the new
first-stage design's mechanisms.

<a id="generation-only-modeling"></a>

**User direction, 2026-11-06: model token generation only, with speculative
decoding disabled first.** Build, calibrate and independently validate the first
predictor on ordinary one-token decode forwards (`T=1`). Source captures,
conditioning generation, fixed-work generation and live confirmation must use
that regime. Verify actual one-row decode forwards and zero draft proposals and
speculative verification work. Preserve cooperative CPU/5090/3090 execution,
adaptive residency and cache-aware routing while changing the decoding mode.
The public request override is `strata_tune.spec=1`, with
`spec_policy="existing"`, `verify_segment=0` and `stage_overlap=0`.
The historical `native_delivery.py --arm ordinary` also disables routing;
it is not this baseline. Keep the selected prior routing and its explicit bonus
when using `native_screen.py --spec-policy ordinary`.

**Prefill is outside the modeled duration and objective.** It may run as native
setup, and its resulting placement, learned demand, pending work and context
length are explicit generation-start inputs. Do not predict prefill time, fit
generation costs to prefill observations, or treat the starting state as empty
unless the declared setup makes it so. The three initial dense decoder layers
still execute on generated tokens; they are not prompt prefill. Keep complete
generation accounting and the policy's shared-resource and later-use effects.

Dropping draft time from the old fixed-work traces does not meet this contract:
those traces retain multi-row speculative verification and rejected rows. Use
genuine ordinary-generation work; do not relabel or split old speculative batches
and call them an ordinary-request baseline. The existing recorded-assignment,
no-copy generation-only diagnostic also does not implement this new policy test.

Once the ordinary-generation model has demonstrated useful policy-decision
accuracy, extend and qualify it for speculative generation and its grouped work.
That is a later extension within token generation, not a prefill-modeling phase.
Keep the six proposals and the broader systems/modeling literature as sources of
candidate methods. Select only mechanisms needed for a coherent, testable design;
there is no requirement to take a contribution from every proposal or paper.
The simpler first regime does not waive independent validation, correct timing
boundaries, instrumentation checks or declared measurement budgets. Existing
proposal budgets are ceilings to assess, not mandatory experimental queues.

<a id="shared-model-diagnostic-study"></a>

### Shared model and diagnostic study

**User decision, 2026-11-06: start with one shared model evaluated at each MoE
layer.** The 57 MoE layers have common geometry and ordinary decode has one token
row. Do not start with separately fitted layer parameters, layer-specific offsets,
or a campaign to establish whether every layer deserves its own model. Reconsider
that choice only if the shared approach has a demonstrated failure. Layer IDs
remain observation/alignment keys, not proposed fitted parameters.

First measure which effects materially change generation timing. Use those
findings to choose the first-stage model's inputs, state and composition rules.
In particular, test whether explicit cross-layer cache contents, buffer state and
pending-transfer state can be omitted or represented by smaller summaries. Do
not assume a full stateful timing model is needed merely because the real engine
has state. Full diagnostic records provide a reference; recording a field does
not commit the predictor to using it. Retain the real engine's ownership and
reader protections throughout.

**Initial campaign, completed 2026-11-07;** see the
[result](#shared-model-diagnostic-result). The protocol as declared:

Use the same build in two sequential engine loads: the standalone series without
Nsight/CUPTI injection, then the separate Nsight-launched diagnostic process.
Use the existing Python-code/prose prompts for the scored observations, high effort,
seed `20261101`, and a 384-output-token cap for ordinary requests. Preserve EOS
and actual outputs; report actual evaluated/generated work. Keep the selected
arithmetic, CPU participants, worker helper, prior routing with bonus `0.10`, and
all settings except the stated intervention fixed. Every source and measurement
has speculation off. Prefill remains native setup and is excluded from the
timing objective. All setup time still counts against the experiment budget.

**Clarification, 2026-11-07:** conditioning uses the retained independent coding
and prose prompts, not the scored prompts. The execution protocol selects the
`count_initials` coding prompt and seed-envelope prose prompt from that history.
Use the same declared conditioning sequence in each load. Each process performs
its bootstrap/calibration and helper-first setup, then those two conditioning
requests, and saves its own learned-state origin. Learned state is process-local;
do not transfer it across loads or infer equality merely from identical prompts.
Restore the process's own origin before each comparison arm, process its scored
prompt as setup and record the actual generation-start state. Do not claim
identical physical cache warmth merely from a matching logical-state hash.
The trace pair uses the Nsight process's own origin; its results remain a separate
diagnostic pair rather than a matched timing continuation of the standalone load.

| Run group | Count | Purpose |
|---|---:|---|
| Bootstrap/calibration setup | 2 requests | One short bootstrap per engine load; maximum 32 output tokens each |
| Independent ordinary conditioning | 4 requests | One independent coding and one independent prose request per load; save a separate origin in each process |
| Longer ordinary observations | 6 requests | For each prompt: monitoring off, light records, monitoring off; 384-token cap per request |
| Matched monitoring test | 4 short fixture runs | Off/light/light/off on identical newly captured `T=1` expert work and restored starting state |
| Grouped state-effects test | 4 short fixture runs | Normal optional residency updates versus holding new primary/helper admissions, in both orders; demand servicing and helper computation remain enabled |
| GPU trace companion | 2 short fixture runs | One off and one detailed/Nsight run of the same control fixture in the separately conditioned Nsight load; diagnostic comparison only |
| Conditional follow-up reserve | 0 additional runs | The required second-load setup and bootstraps consume the former four-run reserve |

This is **22 planned executions at the existing 22-execution cap**: **16
measurement runs** (six longer requests and ten short fixtures), **four independent
conditioning requests**, and **two bootstrap requests**. The earlier total of
18 already included the standalone load's two conditioning requests; it was not
18 measurements. The 2026-11-07 [execution protocol](shared-model-diagnostic-protocol.json)
enumerates these 22 requests and fixtures, and its metadata now says 22 total and
zero additional reserve; setup does not sit outside the count cap. The driver has
no path for further runs. Helper-first placement and snapshot/restore operations
are also setup and count against the warmed time cap. Each short fixture uses
the same predeclared 64 ordinary-generation forwards, with every MoE call retained
and the required native KV/prefill setup outside scored generation. Choose the
first 64 complete generation forwards of the coding light capture; use prose
only if coding ended before 64. This selection rule precedes timing inspection.
These short windows
are mechanism diagnostics; they do not replace the complete longer requests or
qualify serving performance. Cap warmed campaign time across both loads, including
native setup and conditioning, at 30 minutes; report both loads and build time
separately. There is no additional run reserve in this reconciled allocation.

**Monitoring preparation.** Existing `timeline` mode lacks the required per-layer
work records. Existing `detailed` mode constructs/compresses text and enables
NVTX. Add a small, preallocated binary record mode for this study; do not build a
replacement predictor first. Give light mode its own predicate; the existing
`mode > 1` detailed branch would also enable the heavier instrumentation.
Record generation start/end, existing layer/join
and host-service boundaries, the four original/executed expert IDs and their
service locations, actual demand-copy work, free/reserved frame counts, and
existing pending/publication observations. Capture the starting cache map once
and mutations thereafter, rather than dumping the complete map at every layer.
Collect only already available CPU/helper timing observations. No additional
CUDA events, queries or waits in the light mode; serialize outside generation.
Overflow or missing required intervals invalidates that capture. Include enough
source-work information to initialize the fixed fixture, without fitting prefill
timing. Reuse the existing trace and policy entry points where appropriate.

The longer requests reveal the actual work and queue exposure. They are not,
by themselves, an isolated monitoring-overhead test: equal prompts and seeds
may still produce different routing/work trajectories. The matched fixed-work
off/light test measures the net effect of recording, including any changed
publication behavior; no constant overhead is subtracted. The detailed GPU
companion establishes ordering and overlap at its recorded scope. Its CUDA event
envelopes are not treated as pure arithmetic costs, nor as timing truth for an
uninstrumented run.

**Minimal effect questions and omission tests:**

| Question | First check | Consequence for model design |
|---|---|---|
| Is immediate foreground work sufficient to explain time? | Tabulate elapsed intervals by actual CPU/primary/helper assignments and demanded transfers across the light captures; no layer-specific fit | Start with these work descriptors; add information only for a material unexplained effect |
| Is the complete cache map needed by the timing predictor? | Track whether history affects time beyond the current selected experts' placement/reuse and resulting work; separately retain later hits/misses and displacement | Try the current-work summary before full-map timing features; do not assume that observing today's placements predicts a changed policy's future placements |
| Do pending copies or occupied buffers add material delay? | Use the grouped normal/held-admission fixture comparison and trace actual pending intervals, availability and work changes | No detailed queue model by default; test a busy/count/bytes summary before phase/age modeling if an effect is present |
| Can readiness be handled by execution order alone? | Audit publication/reader retirement against existing joins on the new ordinary traces; check a declared order-only replay rule using no future availability | Retain a timing-free rule only where its resulting service/transfer decisions remain adequate; localize its failures before adding readiness prediction |

Holding admissions changes future residency and therefore can change foreground
assignments. Fixed expert IDs alone do not freeze that work. Report this fixture
as a **net policy-effect screen**, and show its changed assignments, demand copies,
retained reuse and pending exposure. It does not identify an isolated copy-price
or prove that a zero net effect means every constituent effect is negligible.
If the observations cannot distinguish changed work from contention, leave that
effect unresolved under this allocation. A specifically matched foreground-work
follow-up would need a separately declared budget. Do not infer a causal per-upload price from observational
counts alone.

Use **2% of complete scored generation time** as the proposed materiality band
for this diagnostic screen, reporting milliseconds as well. A component/interval
effect must be related to its actual exposure over generation; do not add
overlapping spans or nested residuals. Treat whole paired executions/blocks as
replicates. Two-pair half-ranges are descriptive spread, not confidence intervals.
Classify an effect as material, bounded below the practical band, or unresolved;
an inconclusive comparison is not evidence that a state field can be omitted.
If the small budget cannot resolve an omission, retain that uncertainty for the
design decision rather than silently escalating to a large calibration campaign.

For prospective policy use, a simplified model cannot recover omitted state by
reading the control trace's future hits, assignments or publication times. Keep
two questions separate: which state the existing policy/ownership bookkeeping
needs, and which state the timing predictor needs beyond its current-work inputs.
Any claimed omission must state which of those interfaces it applies to.

The study's deliverable is a short evidence-backed disposition for each candidate
effect: observed size, monitoring sensitivity, the smallest justified summary,
and unresolved coverage. Then write the new shared first-stage design. The six
proposals inform the choices where useful. Broader references used here are
[causal profiling](https://github.com/plasma-umass/coz), for evaluating whole-program
effects rather than optimizing timer categories, and
[switchback experiment design](https://arxiv.org/abs/2009.00148), for respecting
effects that persist across time. Their application to this engine is a design
inference, not evidence that a particular estimator or predictor will work.

<a id="shared-model-diagnostic-result"></a>

**Diagnostic result, 2026-11-07.** All 22 declared executions ran without step
errors, using 1,304.5 s of the 1,800 s warmed cap. The
[full account](evidence.md#shared-model-diagnostic-study) holds the measurements,
limits and receipts. Coverage is `T=1`, eight staging frames, the two scored prompts
and positions 240-1439. "Rule" is the predeclared class; the reading beside it may
draw on post hoc checks, which changed no label. Each disposition concerns the
**timing predictor's inputs**. Existing policy and ownership bookkeeping keeps
whatever state it needs, and supplies the predictor's current-work inputs.

| Effect | Rule | Observed | First-stage timing predictor |
|---|---|---|---|
| Current service mix | Held-out table unresolved | Work-mix median table misses -1.43%, +1.56% between prompts; all-forward table -8.9%, +10.3%, mostly slow spells | **Represent:** per-layer counts of CPU, helper, primary and new-copy picks |
| Demand copies | Material as part of work | +0.54-0.55 ms each in every capture; 9.9-14.8% of generation in long requests | **Represent:** count per layer |
| CPU and helper overlap | Post hoc | Span matches `max(CPU, helper)` within 0.014 ms; the sum misses by over 0.5 ms; hides 15-16% of generation | **Represent:** a maximum, not a sum |
| Primary path at `T=1` | Post hoc | No overlap with the host lanes, with or without a profiler | **Represent:** sequential after the host lanes in this regime |
| Forward constants | Post hoc | Prefix about 2.6 ms (1.45 with admissions held), final layer about 3 ms, first MoE window +0.41 ms, outside forwards 2.0-2.2% | **Represent:** small per-forward and per-token constants |
| Optional admissions, net | Material | Holding them was slower, +3.14%, +5.61%; changed work alone predicts +346 to +386 ms, against +223, +375 ms | **Represent through policy-driven changes in work;** no isolated copy price claimed |
| Layer identity | Not screened | Median residual about ±0.03 ms apart from the first windows | **Omit:** shared model, no demonstrated failure |
| Foreground copies pending | No exposure | Always retired at plan boundaries | **Omit:** join ordering accounts for it |
| Available buffers | Free count unresolved, no exposure | At least 6 of 8 free in 65,550 windows; limit never bound | **Omit in this regime:** use the resulting plan. Not exercised under pressure |
| Reserved frame or pending primary exchange | Bounded, +0.25 to +0.38% | Same windows for both | **Omit:** at most part of a busy flag |
| Cache history | Unresolved | Long requests within ±0.54%; fixtures up to +3.90% with mixed signs; under 1.4x median within ±1.11% | **Start without;** not shown negligible |
| Helper lease or busy flag | Unresolved | +0.86-1.46% and +1.12-1.73%, near the band | **Candidate one-bit flag** if needed; no phase or age model |
| Context length | Unresolved | Join wait unchanged at positions 240-623 and 1056-1439; later-position excess sits in the helper lane | **Start without** in this range |
| Helper speed state | Not predeclared | `project` kernel 0.076-0.627 ms for similar work; forwards over 1.4x median carry up to 14.1%; 19-30% of generation waits on the helper | **Unresolved:** work inputs cannot represent it. Treat as an operating condition and noise source |
| Light monitoring | Material, +19.33%, +10.85% | Recorder work 0.009%; long requests -0.21% pooled; prefill pairs -6.28%, +0.29%. Recording is not established as the cause | **Unresolved:** trust light records for structure, not short-fixture absolute comparisons |

**Consequence for evaluation:** the same work in the same mode spanned 6,686-8,037 ms
over 64 forwards, so short fixtures cannot resolve a 2% decision here while helper
speed varies this much. A prospective test needs longer work, more replicates or
control of that state.

**Two leads, neither tested in this study nor selected, both needing a separately
declared budget:** the 3090's speed state, and starting primary work during the host
lanes at `T=1`. Low 3090 clocks were [already on record](evidence.md#shared-model-diagnostic-study)
but had not reached this plan. Forcing clocks has no measured result: the 2026-11-01
attempt was refused for lack of permission and the user-run test never ran. Testing it
needs administrator rights and the user's decision. Each lead could change the
composition the first-stage model must describe. The earlier
`primary_submit_query` result used grouped speculative work, so it does not cover
this regime; it remains off and no runtime default changed.

**Corrected reading, user review 2026-11-07.** Accept the campaign as completed and use
it to develop a compact shared-model hypothesis. Do not accept that these inputs are
sufficient. The median-table figure replaces the target's work too and is not accuracy
against actual duration; the 0.014 ms agreement uses observed branch times; the recorder
microbenchmark does not clear monitoring; the helper slowdown is localized, not explained.
[Details and the paired per-invocation recheck](evidence.md#corrected-reading-of-the-diagnostic-study).

**Why the model works per pass.** A pass's completion depends on when its operations start,
which run together, which depend on others and which ultimately delays progress. Keep the
joint timeline of each `(forward, layer)`: actual assignments and required transfers, CPU
and helper submission, execution and completion boundaries, copy and kernel times where a
trace supplies them, and existing joins. A submission returning is not the GPU starting. A
wait at the next join can include this layer's unfinished work. Establish consequences per
invocation, then aggregate them; compare completion times from a common origin rather than
averages of durations.

<a id="shared-model-follow-up"></a>

**Rough plan, user direction 2026-11-07, executed in this order; the active sequence below
stands with these checks placed before the timing formula is fixed:**

1. Finish the interpretation from existing per-pass records. Done above.
2. One bounded follow-up on helper variability and recording together.
3. A bounded `T=1` test of the existing submission query. Freeze the execution
   configuration before final calibration. A negative result ends that experiment.
4. Design and implement the smallest supported shared predictor from complete in-engine
   intervals, current-work descriptors and the observed composition. Keep admission-dependent
   prefix and tail costs visible. Existing policy bookkeeping generates each candidate's
   placements and work; check separately whether simple ordering reproduces optional-transfer
   availability. Small direct timing effects do not make readiness irrelevant to future work.
5. Validate prospective decisions with speculation off: freeze predictions before untouched
   workloads and interventions, judge complete elapsed work and candidate-minus-control
   differences, and keep slow periods and uncertainty across chronological blocks.
6. Extend the qualified model to speculative generation by measured execution. Prefill stays setup.
7. Use the predictor for admission repair and supported optimization, accounting for displaced
   experts, transfers and later reuse. Confirm actual serving speed and usability against the
   preserved best usable complete configuration. The ordinary-decode diagnostic rate is not
   the final benchmark.

**Declared follow-up, frozen before its runs.** [Protocol](shared-model-followup-protocol.json),
rules in `tools/mm3/native_followup_analysis.py`, limit 63 executions and 3,600 s warmed. "Bounded"
means a declared run limit and stopping rule, not an open-ended search.

| Check | Arms and runs | Decision |
|---|---|---|
| Timing variation | Two 4x4 Williams squares, 32 runs of a 241-forward fixture: recorder off without sampler, recorder off with sampler, timestamps only, full records | Each effect is material, bounded or unresolved against 2% by its 95% interval. Clock link needs clock state to explain half the between-forward helper variance and 80% of slow forwards at lower clocks |
| Operating state | Three diagnostics: GPU metrics on each card, clock-event reasons | Descriptive only |
| `T=1` submission | 12 fixture runs in ABBA, BAAB, ABBA with full records; a separate Nsight load for per-pass start and end times, query off and on; 8 complete requests only if the fixture continues | Kept only for at least 2% gain with an interval excluding zero, confirmed on complete requests with no prompt losing. Otherwise the switch stays off and the model describes existing execution |

Forcing clocks stays out of this: it changes a system setting and needs the user's decision
and administrator rights.

**Results, 2026-11-07.** [Timing variation](evidence.md#shared-model-follow-up-timing-variation):
recording is bounded under 2% at both levels, the earlier fixture differences do not replicate, and
the helper's slow periods follow the 3090's clocks, which the driver reports as idle at a median 315 MHz.
Over 241 forwards, a whole-run paired mean resolves 2% with about 5-8 pairs and 1% with 11-25. Keep
full records on where work identity matters, and compare complete totals in the same mode.
[`T=1` submission](evidence.md#shared-model-follow-up-primary-submission-at-t1): **useful**, -15.16%
pooled on complete requests. The ordinary-decoding configuration now has `primary_submit_query=1`. The
timing model describes that execution. Speculative generation keeps it off until tested.

<a id="helper-occupancy-test"></a>

**Worthwhile-work decision, helper occupancy, 2026-11-07.** Measured cost: the driver reports the
3090 as idle in P3 at a median 315 MHz during generation, and a per-pass replay at its 700 MHz level
saves a virtual 13.45% of generation. Mechanism: one minimal kernel on that GPU during generation,
off by default, changing no device setting. Costs: more 3090 power while generating, one mostly
sleeping host thread and any kernel contention. Cheapest check: the
[declared protocol](helper-occupancy-protocol.json), 12 fixture runs and 8 conditional complete
requests under the submission-query rule. A clock rise with no generation gain removes the control.
It must be settled before the final timing calibration, because it changes the helper lane.

**First design, negative, removed.** A separate continuously running kernel held the 3090 at P2,
9,501 MHz memory and about 1,950 MHz core in all samples, without administrator rights, but each pass
waited about 2.9 ms behind it and generation took 165% longer.
[Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/helper-occupancy-20261107/occupancy_detail.txt).

<a id="helper-gap-spinner-test"></a>

**Final design, one test only.** The clock response is real and the loss is the collision. A polling
kernel now fills only the gaps between passes on the helper's own stream, and the worker releases it
just before submitting, so a pass cannot queue behind it. [Protocol](helper-spinner-protocol.json):
same arms and rule, 12 fixture runs and 8 conditional requests. Whatever the result, no further
occupancy design follows without a user decision.

**Result: useful.** [-13.54% pooled on complete requests](evidence.md#helper-occupancy), on top of the
query. The 3090 holds P2 at 9,501 MHz and about 1,950 MHz throughout, one-job helper time falls from 0.585
to 0.317 ms and same-setting requests agree within 0.3%, at about 90 W more while generating.

**Frozen ordinary-decoding configuration, 2026-11-07:** `primary_submit_query=1`, `helper_occupancy=1`,
all else as selected. Final calibration and decision validation use this. Both controls stay per request,
and neither is yet a serving default: speculative generation has not been tested with them.

<a id="shared-model-decision-validation"></a>

**Decision validation result, 2026-11-07: not qualified under its predeclared rule.** No reversals, all
non-tie predictions confirmed, median error 0.96 points, but the 90th-percentile check failed at 4.36 points
against 3 after the one allowed repair. Whole-engine stalls drive that failure; clipped descriptively, the largest
error is 1.46 points, which does not change the verdict. [Evidence](evidence.md#shared-model-decision-validation-speculation-off).

**Consequence for steps 6 and 7, primary-agent decision in the user's absence.** The user asked for the complete
plan and is not available to adjust the rule. The model continues as a **screen only, labelled unqualified**:
it proposes and orders candidates, and nothing is selected without hardware confirmation on complete
generation. A user who prefers to stop at the failed rule can do so; the receipts stand either way.

<a id="speculative-configuration-gate"></a>

**Speculative configuration gate, declared 2026-11-07.** `T=1` results do not transfer automatically: grouped
windows already flush launches, and drafting shares the primary. [Protocol](speculative-gate-protocol.json): selected
spec 5, lenience 0.4, a 2 by 2 factorial of query and spinner, 32 complete 384-token requests in Williams squares with
one square per prompt. A control is selected for speculative generation only at -2% or better with a 95% interval
excluding zero and neither prompt slower. These same full-record runs supply the speculative training data.

**Result: the spinner is selected, -5.37% [-7.36, -3.38], and the query is not, -0.70% [-2.60, +1.20].**
[Evidence](evidence.md#speculative-generation-configuration-gate-and-grouped-model). Query stays off with speculation.

<a id="speculative-decision-validation"></a>

**Speculative decision validation, 2026-11-07: not qualified under its predeclared rule.** Round 1 failed on the
90th-percentile error, 8.67 points against 3, with 100% class agreement. Round 2, on untouched workloads after the one
allowed repair, had no reversals and 7 confirmed non-ties, but failed agreement at 77.8% against 80%, median error at
1.53 against 1.5, and 90th-percentile error at 4.32 against 3. The rule ends here. The model still orders arms correctly
and underpredicts the size of damaging ones. It stays a **labelled, unqualified screen**: it proposes
candidates, and hardware on complete generation decides.

<a id="integrated-screens"></a>

**Step 7 screens, declared 2026-11-07 before their runs.** [Protocol](integrated-screens-protocol.json): 152 complete
384-token requests in the resident engine with speculation and spinner on, recorder off, Williams squares per prompt
on the untouched `val-prose` and `val-code`. Judged by `tools/mm3/native_arm_screen.py`: at least -2%, 95% interval
excluding zero, neither prompt slower. Anything that changes output also needs the bounded usability checks.

| Screen | Control | Arms | Why these |
|---|---|---|---|
| Admission | Effective 0.58 ms saved per use, 8 exchanges | Repaired price 0.32 ms and 0.15 ms interference; budget 16; both | Plan requires testing the repair. An expert's marginal saving on recorded passes is 0.29-0.37 ms, and an exchange costs 1.34 ms against 1.19 charged. The unqualified screen predicts a tie or small loss for all prices tried |
| Cache-prior bonus | 0.10 | 0.15, 0.20, 0.30 | Required value comparison; 0.05 lost earlier |
| Lenience | 0.4 | 0.3, 0.5, 0.6 | Required value comparison; 0.2 lost earlier |
| Drafting | Spec 5, floor 0.3 | Spec 4, spec 6, floor 0.45 | CPU sets the pace at wide windows, and the spinner changed each row's price since the earlier two-draft test |
| DRAM backing, exclusive control | No backing | Backing with the old D2H charge kept as interference, so the schedule stays feasible; backing with the charge dropped | The reopened decision: a lower transfer cost leaves the old schedule available. Needs 13.9 GB plus the 16 GiB reserve; a refusal ends the phase as untested |

<a id="per-pass-balance"></a>

**User direction, 2026-11-07: per-pass balance is the central objective for dividing work.** Aim for the CPU lane
and the copy-plus-primary path to finish together on every individual activation through every individual layer, judged
from the joint timeline and a common origin, never from separate averages. Working form: each pass's finish time
is what is minimised, and the test for a pass is whether any single move of work would make it finish sooner. Work
moves in whole jobs, and a hit removes work outright, so an all-resident pass is unbalanced and still the fastest.

**Correction to the agent's reading.** The claim that wide passes have two nearly equal limiting paths was unqualified model
output. Grouped per-pass copy and kernel times have not been measured with the spinner on. The 0.29-0.37 ms marginal
saving is an average of per-pass consequences, and the `repaired` admission arm above uses it as one global price, so that
arm is not the valuation repair this plan requires. A flat response to a global knob is not evidence of balance: it is also
what passes imbalanced in both directions would show. How far individual passes sit from balance has not been measured.

**Where the engine stands.** [`plan_batch_staging`](../../src/mm3/mm3_batch_staging.hpp) minimises a per-pass maximum, but
on estimated durations, with a flat 1.3 on CPU work and uncontended copies, and its only move is CPU to primary.
[Primary admission](../../src/mm3/mm3_cooperative_policy.hpp) looks at no individual pass.

**Bounded per-pass study, before the final comparison; the screens above continue.**

1. From existing speculative records: which path finished last on each pass and the slack on the others, as distributions.
2. One Nsight load with speculation and spinner on, for copy and kernel start and end times per pass on wide windows.
3. For each pass, the best assignment of its actual experts against the planner's choice. The total bounds what better
   balancing could save on fixed demand.
4. If that bound reaches 2% of generation: move the planner to completion times with contention, value admission by the
   finish-time change each expert causes per pass rather than its frequency, and test both on hardware against control.
   Below that, report that passes are already close to balance. Hardware decides either way; the model stays labelled unqualified.

<a id="per-pass-redo"></a>

**User direction, 2026-11-07: go back before speculative decoding and redo the ordinary-decoding steps with per-pass
balance, for each individual layer, as the objective.** The balance objective was adopted only after the work had moved to
speculative generation, so the steps that shaped the predictor and policies at `T=1` did not use it. Carried forward because
they were already measured per pass: the submission query (copies moved from 0.93 ms into the pass to 0.03 ms, ending a median
0.09 ms from the lanes on one-copy passes), the spinner (one-job helper time 0.585 to 0.317 ms), and the recording and noise
results. Redone at `T=1` with query and spinner on, speculation off:

1. Balance per pass and per layer: which path finishes last, the slack on the others, and pass time against misses. Existing
   records first, then one Nsight load that traces a `T=1` fixture and a wide-window fixture with the spinner on.
2. The predictor judged on per-pass consequences. The earlier validation arms were all global knobs.
3. Residency and admission valued by per-pass finish-time change. First an offline bound at `T=1`: the same residents
   placed to minimise summed pass time instead of maximising hits. At 2% or more, implement and test on hardware against control.
4. Then extend to speculative generation and re-verify the bonus and lenience on the new base. Their screens remain
   measurements and select nothing until then.

**Redo status, 2026-11-07, steps of the rough plan.**
- **Step 1, per-pass interpretation: done** at `T=1` and in wide windows, with the per-layer abs(Δ) metrics and host
  bandwidth measured. The held-launch correction is [withdrawn at `T=1`](evidence.md#copy-to-kernel-handoff): what it caught
  there is a fixed copy-to-kernel handoff on every upload. In wide windows it mostly stands.
- **Steps 2 and 3: stand.**
- **Step 4, per-pass completion model at `T=1`: done** ([evidence](evidence.md#per-pass-model-redo-step-4)). The slowest path
  is right on 95.0-97.7% of passes against about 50% for a layer-aware guess, and every pair's per-layer abs(Δ) is reproduced
  within about 0.01-0.03 ms. Constants were fitted on one trace and tested on the other. It describes observed work and has
  not yet predicted an intervention.
- **Settled by bounds under 2%:** placement valuation 0.18%, splitting misses between CPU and uploads 0.08-0.51%, splitting one
  expert 0.62%, staging frames at bonus 0.20 0.4-0.6%, exchange retiming at most 1.1%, removing the copy-to-kernel handoff
  0.6-1.3% on its own.
<a id="work-list"></a>

- **Work list, in order** (the user's decisions are [below](#user-decisions-2026-11-07)). User direction: make `T=1` as
  strong a foundation as possible before speculative decoding returns.
  1. **Finish the [HAGS study](#hags-study).** The HAGS-on loads, the in-situ traces and the evaluation, then keep or revert.
  2. **Refit and tune at `T=1` under the kept scheduling mode.**
     - Refit the 5090 structure of the per-pass model, and rerun the step-4 check. **Done 2026-11-07**
       ([evidence](evidence.md#per-pass-model-refitted-under-hags-work-list-item-2)). The model now takes the copy's
       lead over the CPU. Every pass check matches the HAGS-off foundation's or does better, and the slowest path is right
       on 97.9-98.2% of passes.
     - **Optimize the continuous `T=1` parameters offline** on the refitted model. **Planner and admission settings: done
       2026-11-08.** The current values are at the model's optimum, the best is -0.02%, and the CPU price sits 0.1 above a 3%
       cliff ([evidence](evidence.md#offline-optimization-of-t1-policy-parameters-under-hags-work-list-item-2)). Still to
       do: the bonus, once the router scores are written out. The parameters searched: the planner's CPU price, the upload cap and
       staging, the bonus once the router scores are written out, and the others. Values are searched, not taken as multiples
       of 0.1. Then confirm on hardware, with usability checks for anything that changes output.
  3. **HAGS changed the timing ([per-pass structure](evidence.md#per-pass-structure-with-hags-on)), so re-evaluate under it**,
     the new imbalance first:
     - **start the CPU batch before issuing the GPU launches**, or issue those from another host thread. With HAGS the host
       starts its CPU batch 0.03 ms later at `T=1` and 0.10-0.16 ms later in wide windows. In wide code the CPU is now the
       slowest path on 82% of passes, by about 0.8 ms.
     - **the upload cap and staging frames:** the cap of four binds in wide code, while the 5090's path has room;
     - **the planner's CPU calibration:** for the same decision its estimate rose 13.5% while the CPU work got 3-23% faster;
     - recompute the HAGS-off bounds from the HAGS-on traces: staging frames, exchange retiming, expert splitting, misses
       between CPU and uploads, and the handoff;
     - the `T=1` submission query, -15% with HAGS off: it worked around Windows' submission batching, which HAGS changes;
     - the spinner against no spinner; it shares the 3090's queue with the helper's work;
     - the helper dispatch mode;
     - the speculative query, -0.70% and not selected with HAGS off, belongs with item 7;
     - the copy-stream upload, item 4, and the SM-driven upload fallback, both specific to the scheduling mode.

     The power plan and clock locks are not repeated unless the spinner's result changes.
  4. **Step 5: frozen per-pass predictions for `T=1` candidates, with hardware tests.** They run under the predictor rule;
     the [rule](#predictor-requirement), decided by the user, includes the stall check. Candidates:
     - per-pass routing, which changes output, so usability checks too;
     - helper refill timing;
     - contention-aware planner estimates;
     - if HAGS is kept, demand copies on their own copy stream, with the compute stream waiting on an event. With HAGS on,
       the probe's 5090 upload-to-kernel gap is 32.8 µs on one stream and 3.8 µs this way. The 3090 shows 52 against 20 µs,
       for its refills. The bound is to be recomputed from the HAGS-on trace ([evidence](evidence.md#copy-to-kernel-handoff)).
     - **bound first: a device-side upload decision**, the one piece of the audit's GPU-driven paging not covered more
       cheaply. The 5090's demand copies would start right after routing rather than after the host's round trip: the
       fitted host preparation is 0.067-0.104 ms per `T=1` pass, plus about 0.03 ms to the copy's start. That only matters
       where copy-plus-primary finishes last, 23-35% of traced `T=1` passes. **Ways to start the copy without the host**,
       to be measured in the probe first:
       - a graph launched from the GPU: CUDA 13.4 permits memcpy nodes in device-launched graphs for pinned,
         device-mapped host memory, so the copy may keep the copy engine. A node's addresses are fixed when it is built,
         so the choice among experts needs prebuilt nodes or a switch conditional node, which a kernel sets with
         `cudaGraphSetConditional`;
       - a mapped-memory read by the SMs: 8% slower in the naive probe kernel, which a better kernel may narrow;
       - shortening the host's round trip instead: the router writes its picks into mapped host memory, and the host
         polls, plans and issues copy-engine copies at once;
       - prefetch on a prediction before routing ends: the engine's `routing_prefetch`, now off.

       **Existing local implementations, agent search 2026-11-07:**
       - Basalt's `dm_stage_kernel`, in `/home/mdwbr/basalt/src/kernels/cuda/dec_moe.cuh:390-444`. A device kernel picks the
         uploads from the router's ids and a device residency row, then copies with SM loads from mapped memory on a
         high-priority side stream. Single GPU only.
       - Infernix's `stage_kernel`, in `/home/mdwbr/Infernix/src/ops/offloaded_sparse_moe/cuda/moe_layer.cu:295-420`:
         device-ranked misses, staging slots or landing frames, SM copies. Primary GPU only.
       - Neither uses copy-engine copies started from the GPU. A memcpy node cannot be retargeted from the device, so
         that route needs prebuilt copies per expert and slot.

       **A minimal Maya test is estimated at about 400 lines behind a flag:**
       - a one-block device port of `plan_batch_staging`;
       - a stage kernel launched right after `route_kernel`, reusing Basalt's copy loop and `NativeHostBank`'s
         existing mapping for the 5090;
       - slots reserved before routing, with the host adopting the device's plan after the existing join.

       **Risks:**
       - SM reads from Maya's own host bank under WSL2 are untested. Infernix's 24 GB/s was another machine, at Gen5 x8.
         This machine's naive probe kernel reached 53.6 GB/s from `cudaHostAlloc` memory.
       - Slot ownership.
       - Long spins under TDR.

       **Order:** first probe SM reads from the bank's actual allocation, with Basalt's loop and the CPU reading at the
       same time. Then bound the chosen mechanism from the HAGS-on traces, with the host time refitted under HAGS. Build
       only at 2% or more.
     - **`routing_prefetch`, user direction:** another agent wrote it, and it is off today. Review it, tune its settings
       from the per-pass information, and try improvements of our own, frozen and tested like the other candidates.
     - **3090 placement, from the user's question.** Helper-first placement puts the globally hottest experts, by
       original demand, on the 3090 at load. During generation the 3090 only swaps its coldest resident for the hottest
       CPU-served expert, when the hot score exceeds 1.25 × cold + 1. An expert that turns hot while resident on the
       5090 is never moved to the 3090. So "hottest on the 3090" holds at load and only roughly afterwards. Bound a
       re-homing rule on the per-pass model, counting the 6.4 GB/s refill and whether the 2026-11-07 placement bound
       already covered which GPU holds which expert.
  5. **Kernel candidates at `T=1`**, user direction 2026-11-07. Each is measured on complete individual layer passes and on
     complete generation, with the trace bound kept for interpretation:
     - **combined target QKV and indexer projections**, against the current overlapping branches. The layer's preparation is
       sequential with its expert pass, so a saving there shortens every pass.
     - **NVFP4 expert kernels, the 3090 first:** SGLang's Marlin W4A16 path, keeping Maya's separate gate, up and down scales.
       At `T=1` the helper is the slowest path on 11-28% of traced passes. The user's nuance applies: those are
       almost all passes with no CPU job, 31.4% of `T=1` passes in the records. There the GPUs hold every needed expert
       and the pass is already short, so helper-last does not mean the 3090 is slow. **User direction: that is no
       reason to skip it.** A clean speed gain counts, and faster helper kernels may matter more later. On those passes
       the 5090 finishes very early, so a faster helper turns almost fully into a shorter pass. **Target both parts of
       the 3090's time:**
       - the kernel, through NVFP4 and Marlin;
       - the startup before it, the activation transfer, worker dispatch, launch and handoff. In the layer-33
         example its kernels ran 0.217-0.301 ms after the pass began.

       Build and measure regardless of the 2% bound. In wide windows the helper is rarely last. The 5090's SM120
       kernels follow only if MiniMax's clipped SwiGLU can be kept.
  6. **Step 6: the speculative extension, the hardest step.** An earlier attempt struggled here, and the user sent the work
     back to build the `T=1` foundation first. Six proposals from five agents exist because this is hard. It starts with the
     design comparison the plan already requires, of the [six collected proposals](timing-predictor-proposal-measured-transitions.md),
     against the per-pass evidence gathered at `T=1`. The `T=1` model is not extended ad hoc.
  7. **With speculative decoding back on, re-evaluate earlier `T=1` decisions that may not transfer:**
     - the query and the spinner, both already gated once for speculation;
     - the HAGS items above;
     - staging frames and the upload cap;
     - admission prices;
     - the two kernel candidates again, since wide windows change the kernel shapes;
     - then DSpark context-projection batching and GPU-only graph replay, which exist only with drafting.
  8. **Step 7 on the new base:**
     - per-pass routing and refill timing under speculation;
     - **offline optimization of all continuous parameters**, lenience and drafting included, in place of round-value screens
       such as the lenience 0.3 retest;
     - hardware confirmation with usability checks, and a new final comparison if the configuration changes.
  9. **Close-out:** dispositions, memory and the final report.
- **Dropped by measurement:**
  - The post-copy flush, the user's build item and polling-thread candidate. The probe shows polling leaves the handoff
    unchanged, 46 against 44 µs, and the host already makes CUDA calls inside the gap.
  - An SM-driven upload on the 5090, kept as a fallback only: its 8% slower transfer cancels the saving.
- **Waiting on the user:** fast-forwarding their checkout to `minimax-wsl/objective-port`.

The configuration confirmed today stands until then.

<a id="user-decisions-2026-11-07"></a>

**User decisions, 2026-11-07 evening, in the user's numbering, with status.**
1. **Spinner kept** over a held 3090 clock: same speed, the lock about 16 W less. *Done: serving default unchanged.*
2. **HAGS test approved**, two reboots at most. The user expects to keep it after one. *In progress:* [rule and run](#hags-study).
3. **Build and measure.** *On the [work list](#work-list) unless noted:*
   - post-copy flush, small: *dropped by measurement*, since polling leaves the handoff unchanged, 46 against 44 µs;
   - per-pass routing, which changes output: work-list item 4, then 8;
   - refill timing: work-list item 4, then 8;
   - planner estimates: work-list item 4;
   - from the user's audit, combined QKV/indexer projections at `T=1`, item 5, and again with speculation, item 7; DSpark
     projection batching and graphs with speculation, item 7. Their benefit is still to be measured.

   The handoff evidence adds one item, if HAGS is kept: demand copies on a copy stream with an event.
4. **Predictor: decided by the user, 2026-11-07.** Fund a new predictor on the per-pass law, and test it in step 5 against the
   same pass marks. Stalls are detected by a fixed check, removed from both the predicted and measured times, and reported
   on their own. They do not decide pass or fail, unless a candidate causes them and they are excessive. That test and its
   size threshold are fixed in the step 5 protocol before the round. [Details](#predictor-requirement).
5. **Push the feature branch.** *Done.* `minimax-wsl/shared-model-diagnostic` is on origin; feature branch only.
6. **Bring the AGENTS.md and CLAUDE.md objective into the user's checkout.** *Prepared.* This session's harness refuses git in
   the shared checkout. Branch `minimax-wsl/objective-port` (04f47ad) holds only those two files on top of the checkout's
   4b56b15. The user runs `git -C /home/mdwbr/project-maya merge --ff-only minimax-wsl/objective-port`, or cherry-picks it.
7. **Lenience 0.3 retest at bonus 0.20: replaced by offline optimization.** Lenience, bonus and the other parameters are
   searched as continuous values, not multiples of 0.1. The search uses the per-pass model, the planner and trace replay,
   then hardware confirmation with usability checks. *Work list: at `T=1` in item 2, with speculation in item 8.*
8. **The user's NIM/SGLang audit**, `docs/minimax-wsl/nim-sglang-reference-audit.md`, on branch `minimax-wsl/proposal-2` at
   4b56b15 (not on this branch). *Taken into the work list:*
   - QKV/indexer fusion against the current overlapping branches, and NVFP4 expert kernels for the 3090 (the faster helper
     kernels), both at `T=1` before speculative decoding returns, item 5, and again after it, item 7. That is the user's
     direction of 2026-11-07, later than the audit's order.
   - DSpark context-projection batching and GPU-only graph replay, item 7, since drafting exists only with speculation.

   *Limits recorded from it, not to be adopted:*
   - **MiniMax MegaKernel:** targets SM100/SM103 and TP4, not the 5090's SM120.
   - **NIM recipes:** their settings target 128 concurrent requests; they are not interactive defaults.
   - **SGLang's captured sampling-acceptance path:** it excludes top-p, so it cannot replace Maya's top-p and lenience sampler.
   - **SGLang's KTransformers CPU/GPU wrapper:** not a replacement for the adaptive two-GPU arrangement.
   - **GPU-driven expert paging, not adopted as a whole.** Its documented path excludes speculation and multiple GPUs,
     adapting it would be substantial, and its per-layer LRU slots are a weaker policy than Maya's adaptive residency.
     Its copy mechanism's benefit, no copy-to-kernel handoff, comes more cheaply from the copy stream with an event under
     HAGS. One piece is bounded first in work-list item 4: deciding the 5090's uploads on the GPU so its copies start
     before the host's round trip.
   - **The NIM image:** it declares an older SGLang base plus NVIDIA patches, so no upstream feature is assumed present.
9. **Two candidates for the post-copy gap.**
   - The polling host thread: *dropped by measurement*, as in 3. The "held on 19-26% of passes" premise was withdrawn: the gap
     is a fixed copy-to-kernel handoff on every upload pass.
   - HAGS: *in progress*, item 2.

<a id="predictor-requirement"></a>

**Predictor requirement, evaluated for the user.** The requirement is the predeclared decision-validation rule: no reversals,
at least 80% class agreement, median error at most 1.5 points, 90th percentile at most 3 points, on frozen prospective decisions.
- At `T=1` it failed only on the 90th percentile, 4.36 points. Whole-engine stalls drove that; clipped, the largest error is
  1.46 points.
- Speculative round 2 failed narrowly on agreement, median and 90th percentile.

The new per-pass model does not meet the requirement either: it is validated on observed work, not on predicted interventions.

| Option | What it gives | Against |
|---|---|---|
| Accept it as unmet | No new cost; the model stays a labelled screen and hardware decides | The offline optimizer the user asked for would then rank by an unqualified model, and Maya cannot be called complete while the requirement stands |
| Fund a predictor on the per-pass law | Builds on what is now validated per pass. It needs two missing parts: routing's response to the candidate (replaying recorded router scores under the candidate's bonus and lenience) and stalls kept as a measured condition. Redo step 5 is its validation round anyway | A day or more of work and a validation campaign. It can still fail |
| Change the requirement | Judge decisions, not totals that include stalls. The plan already keeps operating-state slow periods visible rather than priced | Retroactive relaxation would pass a model that failed. It is legitimate only if declared before a new round |

**Recommendation, in plain terms.**
- **Build the new predictor** on the per-pass model.
- **Test it in step 5 the same way the old one was tested.** Freeze its predictions for each candidate change, then run them
  on hardware, with the same pass marks: no reversals, at least 80% agreement on which way a change goes, median error at
  most 1.5 points, 90th percentile at most 3.
- **Change one thing, written down before the test.** A stall is a whole-engine pause unrelated to the work being done, which
  no model of the work can foresee. Stalls caused the old `T=1` failure. They would be identified by a test fixed in advance,
  removed from both the predicted and the measured times, and reported separately.
- **If the new predictor still fails,** the requirement is accepted as unmet, and hardware keeps deciding everything.

**Decided by the user, 2026-11-07:** as recommended. Stalls do not factor into pass or fail unless a candidate causes them
and they are excessive. Then they count against that candidate in both the measurement and the prediction error.

<a id="hags-study"></a>

**HAGS study, declared 2026-11-07 before its runs.** [Generator and rule](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/make_protocols.py).
The HAGS-off arm is the final comparison's three new-arm loads, plus a drift check: the same twelve scored requests on the
still-resident `load6-new`. The HAGS-on arm is three fresh loads after one reboot. The decision:
- **Revert** with a second reboot if either budget is credibly slower, its 95% interval across pairs below zero, or its mean
  is below -2%.
- **Otherwise keep it on**, reported as faster, or as a tie with or without a shorter handoff in `native_handoff_probe`.
- **Drift:** if the check moves more than 3% at either budget, the engine comparison is confounded, and the probe and an
  in-situ `T=1` trace carry the result.

The removable handoff alone is worth about 1%, below what three loads per arm can resolve. Only a larger effect, through
submission in general, would show in tok/s.

**Drift check: confounded under the rule.** The twelve requests on the resident `load6-new` ran at 17.78 tok/s at 128 tokens
and 19.04 at 384, against 19.53 and 20.14 at 16:26: **-8.98% and -5.47%**, or -5.96% and -5.71% over repeats 2-3
([evaluation](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/drift-evaluation.txt)). Its restores took 64-209 s a step,
against 30-40 s at 16:26, after about two hours of further experiments on that engine. That points to the aged load rather
than the system, but one check cannot separate them. It now sits inside the spread of the other fresh loads, 17.8-19.1.

**How the probe and the in-situ trace decide, fixed before any HAGS-on data:**
1. *Mechanism:* the 5090's large-copy handoff falls by at least half, either in the probe (same-stream median at most
   22 µs) or in situ (median at most 0.029 ms after the last demand copy).
2. *Per-pass speed:* the summed generation time of the four traced fixtures is no more than 2% above the HAGS-off traces.
3. *Keep HAGS* if 2 holds and either 1 holds or the traced fixtures are at least 2% faster. *Otherwise revert.*
4. *Second reboot:* if the three fresh loads are credibly slower (95% interval below zero and mean below -2%), run three
   fresh HAGS-off loads before deciding, for a comparison in the same condition.

**Fresh-load control, user direction before the reboot:** "do a full reload of the engine and test again". One fresh HAGS-off
load with the same setup and twelve requests ([generator and rule](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/make_fresh_off.py)).
Rule, fixed before its data:
- **Within range:** its pooled rate lies within the earlier fresh loads' range widened by 3% at both budgets, 17.28-20.12 tok/s
  at 128 and 18.26-20.74 at 384. Then the system has not drifted and the drift check is put down to the aged load. The engine
  comparison stands, and this load joins the HAGS-off arm as a fourth load in a secondary comparison.
- **Outside it:** the decision above applies.

**Fresh-load control result: inside, so the engine comparison stands.** `off-fresh1` ran 18.05 tok/s at 128 tokens and
19.17 at 384, against the earlier fresh loads' 17.81-19.53 and 18.82-20.14. Its restores took 40-55 s, as at 16:26
([evaluation](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/control-evaluation.txt)). The drift check mostly reflects
load-to-load spread: `load6-new` had been the fastest fresh load, and the aged engine's 17.78 and 19.04 tok/s are within
1.5% of this fresh load.

**Revised by user direction, before any 2048-token or HAGS-on data:** "Run 2048 tokens for hardware acceleration control and also the
after reboot test to get a more accurate test", and "run once with monitoring and once without monitoring to get a no monitoring
average token speed only measurement". [Generator and rule](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/make_long.py).

Six fresh loads: three with HAGS off before the reboot, `off-fresh1` to `off-fresh3`, and three with HAGS on after it,
`hags-load1` to `hags-load3`. Each runs the same sequence:
1. the final comparison's setup and twelve scored requests;
2. six 2048-token requests, three repeats of each prompt, monitoring off: **the speed measurement**;
3. two 2048-token requests with full records and GPU telemetry, for per-pass balance and clocks only.

**Decision:** at 2048 tokens, over the pairs (`off-freshN`, `hags-loadN`), revert if the 95% interval lies below zero or the mean
is below -2%. Otherwise keep HAGS on, reported as faster if the interval lies above zero, else as a tie.

**Reported, deciding nothing:** the 128/384 results against both sets of HAGS-off loads, the probe, the in-situ trace and the
monitored per-pass balance. Same-evening fresh loads replace the aged-load drift check as the control, so the rule written
under that confound is superseded. One reboot's own effect cannot be separated from HAGS without a second.

**HAGS-off arm done, 2026-11-07 19:00-20:40.** [Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/off-evaluation.txt),
[per request](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/off-long-detail.txt).

| Fresh load | 2048 tokens, speed | Python-code | Prose | 128 | 384 |
|---|---:|---:|---:|---:|---:|
| `off-fresh1` | 20.45 | 22.19 | 17.79 | 18.05 | 19.17 |
| `off-fresh2` | 20.39 | 21.96 | 18.16 | 19.18 | 19.35 |
| `off-fresh3` | 20.91 | 22.00 | 19.56 | 19.58 | 19.37 |

Rates are tok/s, monitoring off.

- **Natural stops.** Every 2048-token request ended on the model's own stop, at 365-1,449 tokens, mean about 830. Each load
  still pools about 5,000 tokens, against 2,300 at 384.
- **Spread.** The per-load spread at 2048 is about ±1.4%, against ±4% at 128. Python-code agrees within ±0.5%; prose varies
  with its length and content.
- **Monitored pair:** 21.4-23.3 tok/s for code and 16.9-18.8 for prose, kept for diagnosis only.

**Result, 2026-11-07 22:13: keep HAGS on, a tie under the rule.** [Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/evaluation.txt).

| Comparison | HAGS on over off | 95% interval | Per prompt |
|---|---:|---:|---|
| **2048 tokens, three pairs: decides** | **+0.12%** | -8.40, +8.63 | prose +1.74%, code +2.74% |
| 384 tokens, three pairs | +5.08% | **+0.69, +9.47** | prose +4.03%, code +7.15% |
| 128 tokens, three pairs | +6.68% | -3.59, +16.95 | prose +8.27%, code +5.36% |
| 384 tokens, all six HAGS-off fresh loads against three on | +4.92% | **+2.01, +7.83** | |
| 128 tokens, the same | +7.56% | **+2.80, +12.32** | |

- **The 2048 result is a tie.** HAGS-on ran 20.58, 21.05 and 20.17 tok/s against 20.45, 20.39 and 20.91, pairs +0.62%, +3.26% and -3.54%. The pooled rate
  mixes prompts whose natural stopping lengths differ by load, so both prompts gain while the pooled mean barely moves.
- **The shorter budgets gain credibly**, about 5-8%. Their requests are capped, so composition does not move them.
- **The 3090 was busy.** In every HAGS-on monitored request it held P2, 1,965 MHz, 87-89% NVML utilization and
  184-205 W, against 80-84% with HAGS off ([telemetry](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/telemetry-compare.txt)).
  The user saw it near idle in a Windows view, which plausibly undercounts CUDA work with HAGS on. That is not verified.
- **Next.** The in-situ traces run next. HAGS changed the timing, so work-list item 3 applies.

After the reboot, in order:
1. `native_host_control.py mount-transfer`, then `hags-query`.
2. `native_handoff_probe.py` into `handoff-probe-20261107/hags-on`.
3. The three loads of `protocol-hags.json`.
4. The `T=1` fixture traces.
5. Evaluate, then keep or revert.

**Per-pass study outcome, 2026-11-07.** [Evidence](evidence.md#per-pass-balance). The CPU lane and the copy-plus-primary
path already finish within about 0.18-0.32 ms of each other at `T=1` and in wide windows. Every bound for moving work
between them is under the 2% rule: how misses are split 0.08-0.51%, placement 0.07-0.18%, exchange retiming at most
1.1% at `T=1` and 0.6% wide. The large per-pass imbalance is CPU against helper. The helper finishes 0.3-0.6 ms earlier at
`T=1` and 0.9-2.7 ms earlier in wide windows on most passes, and the 3090's Gen4 x4 link makes per-pass uploads to it
impossible, so only routing can move work there. The uniform bonus pulls work toward primary and helper residents
alike, and in every pass. Next design to test: substitution applied only where it shortens the pass, toward the path
with slack in passes the CPU finishes last. The bonus usability review comes first, because every value changes output.

**Model outline, 2026-11-07; formula and coefficients wait for the two checks above.** One shared
model for all MoE layers, working on the join-to-join cycle that partitions a forward exactly:

| Part | Outline | Source of its inputs |
|---|---|---|
| Work | Each candidate's assignments, copies and exchanges per pass | Existing policy bookkeeping replayed over recorded demand; no future availability from the control run |
| Optional-transfer availability | Declared order rule in layer joins, checked separately against recordings | Fixed before use; failures localized before anything heavier |
| Pass | Host preparation + slower lane's completion from the common origin + resume and wait for copies and primary picks | Current-work descriptors only; no layer parameters |
| Forward | Prefix + first-window extra + cycles + final-layer extra + tail, with admission-dependent prefix and tail kept visible | Exchanges and leases from the replayed work |
| Operating state | Helper stretch stays a measured condition and a noise source; not hidden in a coefficient | Follow-up telemetry decides how it is represented |
| Evaluation | Complete elapsed work and candidate-minus-control differences across chronological blocks | Slow periods stay in the data |

Judge a replacement by demonstrated accuracy in predicting consequential
candidate-minus-control performance changes, its measurement and implementation
cost, and its declared operating domain. It must retain causal use of demand and
state, account for displacement/transfers/overlap at the resolution needed for
the decisions, and avoid importing a control run's future availability or clock
trajectory into a changed schedule. A coarse model is a candidate, not a proven
winner; a joint lookup over every lane/transfer combination is not required.

Before new hardware measurements, declare a finite comparison budget, the
workloads and controls, practical-equivalence margins and the model-selection
rule. Use independent chronological conditioning and untouched evaluation work.
Keep nested timing residuals separate, preserve correlation in paired
differences, distinguish supported practical ties from inconclusive intervals,
and require evidence of useful discrimination rather than an all-tie predictor.
Predictions outside supported cost/operating coverage remain unqualified.

The active sequence is **bounded generation-only diagnostic study → shared
first-stage design using the measured material effects → implementation and
decision validation with ordinary decoding → speculative-generation extension
and qualification → repair admission valuation and
perform supported offline optimization → short integrated confirmation**. Reuse
the valid state-reproduction checks throughout. The former reconstruction's
stage-2/3 passes are not required to start this sequence. Replacement decision
validation, actual operating benefit, bounded usability and the existing
[completion contract](#execution-and-completion) remain required; this decision
does not waive unfinished admission, parameter or integrated-performance work.
The primary agent performs the modeling and evaluation without delegation.
Held bases remain held. Query-on remains unselected after its ordinary-request
test; no further repeat of that screen follows from the replacement decision.

### Bounded cooperative replay

**Historical approach and receipts, superseded by the
[replacement decision](#timing-model-replacement).** The earlier six-stage order
and next-step wording below describe that effort, not a current instruction to
keep repairing its timing model. Retained state/instrumentation requirements are
carried forward above.

**User instruction, 2026-11-04:** build a bounded event-driven replay of the
actual cooperative engine. This supersedes the earlier approximate replay scope
and the order of the reopened admission work. The primary agent performs all of
this work; **no sub-agents** are permitted for this task.

The implementation order is **state-faithful replay → independently calibrated
execution model → intervention validation → repaired admission valuation → offline
parameter search → short integrated confirmation**. These are separate gates;
agreement between a policy and a simulator sharing its prices is not validation.

- Extract hardware-independent production decisions for demand, staging,
  retention/refill, eviction and helper actions. Replay gives these decisions
  only the historical prefix and current observed or simulated state. Original
  router selections alone update demand; executed selections, including rejected
  speculative rows, determine the serviced work. Cache-aware top-four stays on.
- Schedule work and dependencies on CPU, primary/helper compute, copy queues and
  shared host/transfer resources. Recompute availability, transfers, publication
  and completion for each policy. Recorded launch times, staging opportunities
  and operation durations are observations, never counterfactual predictions.
  Exchange descriptors reserve existing staging frames; count real allocations
  and aliases, active readers, pending work and preservation copies.
- Capture complete starting policy state and missing dependencies. Reject
  incomplete/unsupported captures, or explicitly restrict them to complete
  intervals. Preserve learned state across requests; changing learning rules
  requires replaying chronological conditioning under the candidate.
- Keep physical service tables independent of policy knobs, including
  `cooperative_cpu_scale`. Start with grouped measurements and fill only
  decision-relevant coverage gaps: mixed groups/working sets, measured worker
  configurations, primary/helper submission and reduction, transfer directions,
  queueing and concurrent CPU/copy/compute contention. Retain sparse-coverage
  uncertainty; thread count alone cannot predict worker speed.
- Validate (1) decisions/state/bytes with recorded availability, (2) independent
  timing against held-out service, queues, publication, branch and request
  observations, and (3) predicted interventions on hardware. First interventions
  are backing, reduced optional admissions and changed staging amount. Use fixed
  expert workloads to isolate scheduling, then ordinary model requests for the
  integrated result. Preserve adverse outcomes and unresolved rankings.
- Value a bounded incoming/victim shortlist by the marginal completion-time
  reduction over identical recent original-demand batch patterns, less an
  uncertainty margin. Include displaced service, necessary copies, unavailable
  frames, contention, overlap and irrevocable issued work inside both alternatives.
  No second interference surcharge. Keeping the allocation with no extra refill
  is always a candidate. Offline evaluation may use future work; policy may not.
- Check fixed-schedule monotonicity when transfers shorten; when backing removes
  D2H, retain the previous feasible schedule as a diagnostic comparison. This
  adds no inference waits. Timing acceptance must resolve the proposed effect:
  a 3% gain within 5% uncertainty remains unresolved.
- Search coupled policy/capacity/scheduling choices within measured ranges,
  including incumbent and conservative/no-extra-refill controls. Rank total
  generation time for recorded emitted work on held-out complete requests/topic
  sequences, with per-task and cost-uncertainty checks. Routing bonuses and
  speculative controls support only conditional screening on covered scores and
  proposals; changed hidden states/tokens require live confirmation.

**Retained implementation:** shared production decisions, complete starting-state
checks and frame/byte accounting passed scoped reproduction. Both the older
approximate replay and the later factorized resource/timing model are historical
diagnostics. The later timing model never qualified; its retirement does not
invalidate the useful state checks. Follow the replacement sequence above.

**Replay implementation progress, 2026-11-04:** the first complete extended
capture passes observation-fed reproduction: 1,710 staging choices, 261 primary
choices, 16 helper choices, 18,240 generation original selections (80 target
rows across 57 MoE layers), 128 primary issues/127 publications, 112 retained
uploads, transfer bytes and final canonical/frame ownership. One primary
exchange remains pending at the recorded end and stays pending in the account.
[Stage-1 receipt](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/state-capture-v2/reproduction.json).
The first extended schema omitted cumulative starting counters; its first
pre-exchange prefill service supplies that verified baseline. Subsequent captures
record the counters in the starting snapshot. This is decision/state evidence,
**not timing validation**.

Production demand, grouped staging and incumbent primary/helper selection now
have hardware-independent entry points. The new offline implementation has a
separate resource scheduler, physical measurement tables, actual frame rotation,
bounded input checks and three independent evidence gates. Native measurement
hooks and a fixed-expert-work target runner are exercised below. The old replay
remains historical diagnostic code. Ten focused offline checks and four native
policy/ownership checks pass; independently calibrated timing, prospective
interventions, repaired admission, held-out search and integrated confirmation
remain open. Do not use the partial execution model to select runtime defaults.

The first independent service probe has 431 cells and 30 concurrency samples,
but its worker comparison is **unqualified**: it kept the alternate pool alive
while the normal workers could still spin on the same pinned cores. Preserve
that probe at `cooperative-replay-20261104/physical/service-probe.json`; the fitter
now rejects it. The repair measures one pool at a time and adds partial/mixed
groups. This load's actual CPU configuration is **15 participants total (14
workers plus the caller)**, observed in its native startup and probe. Earlier
wording that described 15 workers plus the caller was incorrect for this source's
automatic configuration; do not infer worker performance from that count.
Both new training and held-out captures pass state/decision reproduction; their
unchanged request execution can be reused after correcting the probe.

**User direction, 2026-11-05: foreground reconciliation comes next.** The immediate
deliverable is a reconciled timing account of the existing fixed expert workload
with weight copies disabled. First hold recorded assignments and dependency order
fixed while independently predicting execution; report isolated service, scheduled
service including contention, queue/host-submission delay, observed service and
observed completion separately. Only then enable autonomous policy scheduling.
Explain measurement-boundary, launch-structure, working-set and contention effects
at the scope supported by the observations. GPU event spans can include host
submission gaps; do not label them pure arithmetic time.

The corrected isolated-probe model predicted the held-out generation **9.4% high**.
The separately reported **CPU -34%, helper -71%, primary +139%** are isolated
**table-to-live discrepancies**. They omit scheduler contention adjustments and
are not a decomposition of that whole-request prediction error. Preserve both
accounts at their correct scope. No global helper/primary multiplier is justified.

Preserve helper job widths after the actual eight-column split and the distinct
width launches; model fixed cost plus a measured service per occupied width/job
count, with preparation and activation/result transfers separate. Use corresponding
CPU/primary chunk and kernel boundaries. Replace the provisional
`min(elapsed_a, elapsed_b)` contention fit: only aligned measured intervals can
establish overlap. The maximum-of-pair-factors approximation needs common multiway
checks. Validate interpolation by holding out measured shapes within the same
worker/kernel/launch regime, and derive uncertainty from residuals and repeat
variation, rather than fixed 5%/20% allowances or requiring every mixed shape.

After the corrected foreground model passes fresh held-out work, add transfer
scheduling and prospectively test backing, admission-budget and staging changes.
Acceptance is whether uncertainty can reverse the **difference between schedules**,
with important costs represented, not a percentage requirement on every tiny
operation. The existing copy-free grouped and C1 captures and rejected probe remain
available; do not expand this into a general GPU simulator or resume parameter
ranking before these gates.

**Foreground timing account and fresh validation, 2026-11-05:** the
[reconciled account](evidence.md#foreground-timing-reconciliation) now reports
isolated service, modeled resource service, queue/submission dependencies and
observed measurement spans separately. Original GPU endpoints that were not
captured remain unavailable. Helper width groups, real chunk counts and record
reuse are retained; interpolation holds out whole shapes. Duration-only
contention observations and universal percentage allowances are rejected.

The no-copy CUDA/NVTX companion measured primary event spans of **7,155.812 ms**
but only **215.264 ms** in recorded kernels. Helper spans were **2,671.522 ms**,
with **2,331.370 ms** in kernels. The fixture-only nonblocking stream-query test
moved most of the primary CPU-dependent delay from the compute timer into its
explicit wait timer. It does not select a runtime optimization. The original
9,995.161 ms fixed generation also spent **565.240 ms** formatting route records,
an omitted capture cost. Working-set causality remains unproved: on this same
fixed work, the older CPU table was **0.24% low**, while the new full-bank probe
was **7.26% low**. The earlier -34% figure concerned a different ordinary request.

The frozen model predicted **5,228.542 ms** for fresh held-out fixed work; hardware
measured **4,952.715 ms (+5.57% prediction error)**. Its CPU service was +13.95%
and helper event-span prediction +42.57%; these overlap and do not decompose the
request error. The conservative 1,974.761–12,670.227 ms envelope is not a
confidence interval or an accuracy pass. State/decision reproduction passed
1,995 staging choices and 19,608 original generation selections over 4,902
evaluated layer rows. Exact work, static rosters and zero weight copies were
verified. [Prospective plan and evaluation](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/foreground-heldout/evaluation.json).

**Timing remained unqualified at this checkpoint.** Further foreground cost and
submission-context generalization was then proposed; that continuation is now
superseded by the replacement decision. No global lane multiplier or broader
passing tolerance made the model qualified. No parameter defaults were selected
by this work.
The native build, 29 focused Python checks and four native CPU checks pass.

**Revised immediate order, user direction 2026-11-05:** repair acceptance logic,
then test one nonblocking `cudaStreamQuery` immediately after primary expert
submission and before CPU service in ordinary inference. The existing Windows
`k::submit` runs later and does not reproduce this placement. Use the existing
warmed Python/prose comparison with detailed capture and Nsight absent. Adopt
only if an integrated benefit is demonstrated; then calibrate the operating
regime intended for use and prospectively test backing, admission budget and
staging changes with the existing factorized model. Coarser empirical episodes
remain an alternative if decision failures justify them, not the chosen design.

Nested operation/branch/request residuals must remain separate diagnostics.
Prospective candidate-minus-control differences use uncertainty that preserves
correlation across arms; practical ties can pass only inside a predeclared
equivalence margin. An interval merely containing zero is unresolved. The
defective gate did **not** reject the latest held-out run: its evaluator recorded
the diagnostic result as unresolved directly. Gate repairs do not erase the
275.827 ms whole-generation prediction error. The detailed investigation remains
useful evidence of a potentially controllable queue-progress effect, not a
guarantee that a completion-status query flushes driver work.

**Ordinary query comparison, 2026-11-05:** completed both orders with detailed
capture, Nsight and optional monitoring absent. Pair effects were +6.48% and
+0.18%; the control moved +5.83%. The first off prose response took five extra
target rounds, while the later off and both on outputs/work matched. The pooled
+3.32% is not a demonstrated useful gain under that variation. Retain
`primary_submit_query=0`; the request control remains available for diagnostics.
No additional ordinary speed repetition is planned. The
[full receipt](evidence.md#acceptance-correction-and-ordinary-primary-query-test)
preserves all four arms, per-task results, validation and truncation limits.

**Prospective fixed-work interventions completed, 2026-11-05:** the new complete
setup snapshot permits prediction binding without detailed capture during timed
work. Two reserved requests passed state reproduction and supplied 64/85 evaluated
generation rows. Predictions for backing, admission budget 8→2 and staging 4→2
were frozen before both hardware orders. All three measured paired differences
changed sign; the control moved 6.05%, so none supports a resolved decision.
The full-work probe-based comparator predicted 12.062 s versus 6.693 s observed
mean (**+80.2%**). It is separate from the earlier generation-only foreground
refit, which does not cover these prefill/transfer interventions. Exact admission
counts and verified removed D2H do not qualify its timing. The retained fixed
schedule passes the shortened-transfer monotonicity diagnostic.
[Complete intervention account](evidence.md#prospective-fixed-work-intervention-result).

The acceptance logic now keeps nested timing residuals separate and permits
supported practical ties. Its repaired gate leaves these results unresolved.
The user subsequently required [replacement](#timing-model-replacement), rather
than another calibration round or an optional comparison with the same approach.
Submission context and useful resolution of paired differences remain design
issues for the replacement. No global lane multiplier, relaxed tolerance,
further repeat to obtain a favorable number or parameter selection follows from
these results. Admission repair, supported offline search and integrated
confirmation remain unfinished; qualification of the retired timing model is
not their continuing prerequisite.

The 2026-11-05 handoff recorded a resident engine at `http://127.0.0.1:8097/v1`, server **2161071**,
engine **2161074**, with 438 primary, 741 helper and 6,117 HOST records. The common
independent five-request learned state was restored after the synthetic tests.
Query and backing are off; helper-live and primary adaptation are on. The
capture-control file is absent, optional monitoring is disabled and Nsight/CUPTI
libraries are absent. Ordinary runtime defaults are unchanged. This is a healthy
resident handoff, not final simulator, serving or performance qualification.
[Observed handoff](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/fixed-interventions-20261105/resident-handoff.json).
It ran the preserved `maya-mm3-fixed-snapshot` executable with SHA-256
`4120a4dac2dfcacbd6d4f4de7fedb11da8766f97b64cb1177418a82b82cba8cc`;
later Python input handling and documentation changes need no weight reload.
Match source and executable before collecting further evidence. The native build,
27 focused replay checks, complete snapshot/ownership verification and the two
new source-state reproductions pass; none substitutes for timing qualification.

### Policy evaluation correction

**Latest user correction, 2026-11-04:** evaluate the approaches and choices in
this plan against the objective and evidence, not just whether code follows the
plan. An explanation of a slower result does not justify the policy that produced
it. The prior completion review was too narrow. The admission/scheduling and
DRAM-backing dispositions below are reopened; the implementation, actual speed
measurements, functional checks and monitoring-off result remain valid at their
recorded scope. The active Maya optimization is **not complete**.

The user's clarification concerns **GPU admissions**: only original-router demand
may credit residency learning. Cache-induced executed hits must not manufacture
admission demand. Cache-aware model routing remains part of the selected engine.
The current native score updates use the original IDs, so this is not a newly
identified hit-credit configuration bug. The 0.10 routing bonus was tested against
0.05 in two orders (+9.48% pooled); it is a measured provisional setting, not a
global optimum or validation of the admission policy.

The primary-agent [policy reassessment](evidence.md#policy-objective-reassessment)
identifies two gaps that the earlier successful checks did not close:

- The foreground planner uses grouped service curves and a maximum across
  overlapping lanes. Primary admission instead multiplies row-frequency scores by
  a single-column CPU/GPU service difference. It does not share that grouped
  critical-path valuation, and its configured interference surcharge is zero.
  Removing D2H correctly lowers a physical cost; the remaining benefit estimate
  still needs to represent the work and opportunity costs it claims to save.
- The retention report's `used` category includes cache-induced substitutions.
  In the selected trace, seven of the 58 admissions described as used before
  eviction had only substitution-induced executed uses. Correct storage reuse and executed hits
  do not establish positive net admission value, including the evicted victim's
  later service, transfers and occupied working frames.

The next work is a focused correction of these existing workstreams:

1. Preserve original-only demand credit and the defined generation-row clock.
   Keep original demand, executed service and substitution-induced uses separate
   in both decision inputs and evaluations. Reuse valid ownership/state checks.
2. Repair admission valuation to use grouped avoided work, the displaced resident's
   value, transfer interference and useful overlap. Use a small inspectable model
   and existing measured curves; a large predictor is not a prerequisite. Having
   bandwidth available or admission credits remaining is an opportunity, not an
   obligation to admit. Keeping the current allocation and doing no extra refill
   must remain an explicit alternative.
3. Have the primary agent use bounded same-demand modeling/diagnostics to challenge
   the decisions before tuning numerical values. With required work, state and
   other resources fixed, lowering a transfer cost leaves the old schedule feasible;
   assess the selected plan against that alternative. Validate model conclusions
   with focused real tests, including the backing interaction. This diagnostic
   does not require deterministic text/cache history or optional-transfer waits
   in ordinary inference, and modeled savings are not live speed evidence.
4. Revalidate the affected implementation and parameter/backing choices in the
   combined engine. Preserve all adverse results and actual original-engine gains.
   A gain from the complete composition cannot by itself justify every internal
   policy. Close this work on justified decisions and actual net operating benefit,
   not compliance with the old plan or explanatory wording.

Keep the currently usable resident configuration while these choices are repaired
and evaluated; no runtime setting is changed by this documentation correction.
The old instruction that no further backing check is planned is superseded by
this focused diagnosis. Held bases and unrelated parked mechanisms remain held.

**Monitoring-off benchmark completed, 2026-11-04:** the requested final-engine
screen measured **15.5226 tok/s pooled** (256 tokens / 16.492107 s): Python
**16.0539**, prose **15.0253**. Both high-effort responses reached the 128-token
cap in reasoning. [Primary-agent evaluation and raw outputs](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/monitoring-off/evaluation.json).
This is a single absolute-speed screen; it is not a paired estimate of monitoring
overhead because the reload changed calibration and learned/generated histories.
The original-engine comparisons below retain their earlier monitoring conditions.

The previous standalone server still ran its NVML hardware sampler, and capture
mode `off` incorrectly retained frontend token receipts. Those controls are now
fixed at `a105c56`; the native build and 16 focused checks pass. The native change
only gates serving progress/statistic output; model arithmetic/kernels and selected
policies are unchanged. The monitoring-off binary is SHA-256
`94cd6ffcb63407f3f5534cc016136d4b6d23967f342b80691b6285b395376b4c`.
The run disables CPU/RAM/disk/GPU hardware sampling, GPU-map reads, profiler
injection, native/frontend token-round capture, intermediate native STAT/PP,
periodic frontend progress/rate sampling, client stream tracing and process-I/O
sampling. Only generation start/final samples and normal completion measurements
remain. Before/after checks show no monitor processes, `monitoring_enabled=false`,
empty hardware samples, two generation samples per request and zero trace artifacts.
No CPU/GPU polling occurred during the scored requests. The user's request
supersedes the earlier capture requirement for this measurement.

That monitoring-off handoff left server **2093734**, engine **2093738**, idle with
weights resident at `http://127.0.0.1:8097/v1` and monitoring disabled; it is
superseded by the foreground diagnostic handoff above. Its configuration/logs
are under [monitoring-off](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/monitoring-off/).
Use `--no-monitoring` for frontend hardware/progress sampling and
`MM3_SERVE_MONITORING=0` for native boundary-only statistics; the actual run config
also disables the activity-sampling watchdog and has no capture environment.
Earlier process/resource handoffs below are dated observations of superseded processes.

**Preserved implementation and speed qualification; optimization closure reopened above.** The final resident
server passed default-serving and ownership/resource checks at **2026-11-04
16:53 UTC**. The final comparison load measured **14.7816 tok/s versus the exact
original's 13.1024 (+12.82%)**, with Python +6.80% and prose +20.38%. The earlier
declared two orders measured 16.1144 versus 12.8314 (+25.59%); both results and the
lower final-load speed are retained. A stable 25.6% gain across reloads is not claimed.
[Final-load evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/final-load-evaluation.json)
and [resident handoff](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/handoff-reviewed.json).

The selected engine has actual five-part residency implementation, original-engine
comparison, bounded complete-task usability and default/resource evidence. Those
observations remain accounted for in the work table; the reopened policy valuation
and backing decisions prevent treating them as full optimization completion.
Short capped speed screens, approximate service prices and sampling changes limit
the claim; this is no hardware optimum or broad quality guarantee. Infernix and
Strata remain on hold, and their unfinished comparison is outside this completion.

The earlier 2026-11-03 declaration was premature: its sequential-start policy gain
and feature checks did not establish the later conditioning/optimization/whole-engine
outcome. That withdrawal and all adverse receipts remain preserved; the completed
work below supplies the previously missing evidence.

### Active Maya work

Follow [timing-model replacement](#timing-model-replacement) and the
[policy evaluation correction](#policy-evaluation-correction) for current work.
The retained requirements and chronological receipts below remain evidence
and design context; superseded completion language does not close the reopened work.

### Implementation first, then measurement and tuning

**User priority correction, 2026-11-04:** first implement the best complete design
we can justify from the architecture, existing evidence and engineering judgment.
The current implementation is not a reasonable baseline for another optimization
campaign. Repair known material design weaknesses together instead of tuning around
them or requiring a separate speed experiment before each logical repair.
This priority supersedes the earlier comparison-first sequence below.

1. **Build the reasonable baseline.** Implement the
   [required baseline behavior](#required-baseline-implementation) across placement,
   persistent learning, whole-batch cooperative assignment, staging, retention,
   refill and backing. Choose coherent initial policies and numerical estimates;
   record assumptions and use the existing packed kernels and ownership machinery.
   The current first-seen staging assignment is a diagnostic control, not the
   default to preserve until a local parameter win permits its replacement.
2. **Check it as it is built.** Run focused correctness, ownership, resource and
   production-caller checks. Collect bounded independent demand or service data
   when it supplies a missing implementation input. These checks and necessary
   diagnostics may accompany implementation; ranked policy screens and parameter
   sweeps wait. Reuse already collected history and receipts at their valid scope.
3. **Establish operation, then optimize.** Once the baseline is implemented and
   its affected checks pass, demonstrate its behavior under the
   [independent conditioning protocol](#placement-and-residency-qualification-correction).
   Re-establish the usable original-engine comparison, perform the explicitly
   isolated helper/backing checks, and then tune the material remaining parameters
   and interactions using short complete-engine measurements.

The implementation gate is finite and closes on **integrated full-model behavior**:
ownership, capacity, state continuity, useful retention, bounded transfers and the
actual ordinary/grouped callers must be demonstrated. Predicted savings, a build
or passing component tests alone cannot close it. It does not require anticipating
every possible optimization. Logical justification selects what to implement first;
it does not prove optimality or a speed gain.
The new baseline is a candidate engineering starting point, not a replacement for
the preserved best usable engine in the final performance comparison. Required
measurement, quality checks and the completion contract still follow.
The defined demand clock, small batch-cost model and integrated acceptance criteria
below are refinements of this existing baseline, not additional workstreams.

### Baseline implementation progress

The following progress receipts are chronological. Intermediate next-step wording
is historical; the final outcome and active work table own current status.

The independent high-effort history was collected once: five requests and 482,676
generation original-router selections. Code/prose/extraction/reasoning reached
their 384-token caps in reasoning (including draft code/prose); the tool proposal
completed in 44 tokens. This is scoped demand history, not five complete-task
passes. The actual joined helper-first setup verified 741 helper and 437 primary
experts, exact agreement with frequency ranking, full ownership and retained
allocations. Coverage on that training history is 35.28% helper plus 12.74% primary;
those are placement calculations, not held-out hits or speed.
[Receipts](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/history-summary.json).

The implementation under validation makes these intentional initial choices:

- All native demand-derived initialization ranks the helper globally first and
  the primary next. A histogram also seeds a frequency prior for admission scores;
  it does not reconstruct chronological learned state. A helper-only refresh
  rejects incompatible primary contents. Missing history is labelled an adaptive
  bootstrap; an explicitly supplied invalid profile fails instead of silently
  selecting sequential identities.
- Native residency scores track original selections per **evaluated generation
  row**, including rejected rows. Prefill counters remain captured but do not age
  or credit generation residency. Scores are normalized to 100 times frequency.
  Aging is `0.95^(rows/5)`; block-uniform credit preserves the old recurrence at
  five rows and removes its accidental width-dependent scale. This is an initial
  unit conversion, not an optimized decay choice. Horizon 20 means 100 evaluated
  generation rows. Primary/helper admission credits accrue per generation row,
  bounded at eight/one respectively; these capacities replenish per five rows.
- Grouped staging chooses count and expert identities from the whole batch using
  measured CPU/primary/helper grouped-service curves and one upload per distinct
  expert. The bounded greedy objective includes existing GPU work and a helper
  completion estimate; ties prefer no upload, then expert ID. Its additive service,
  interpolation and overlap assumptions remain approximations, and greedy selection
  is not a global optimum. The old first-seen allocator is an explicit diagnostic
  control. The `cooperative_stage_reuse=1` control now means this complete batch
  planner, superseding the unselected reuse-sort prototype.
- Priced retention/refill shares the row-based budget and uses observed grouped
  marginal service and transfer proxies by default (`primary_h2d_ms=0` and `primary_d2h_ms=0` select measured prices). Legacy unpriced temporary promotions are suppressed while
  priced primary adaptation owns admissions. Optional publication remains
  nonblocking and all active readers retain their leases.
- Full-primary DRAM backing is implemented as a separate explicit setup control.
  Existing HOST pointers rotate between canonical ownership and primary backing;
  backed eviction does not download weights. Its allocation checks keep 16 GiB
  physical/commit headroom. The exclusive control remains available; implementing
  the mode does not select its performance or mix it into the helper comparison.

**The integrated baseline implementation gate is established, 2026-11-04.** The
[qualification receipt](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline-integrated-qualification.json)
records the actual ordinary/grouped behavior, ownership, fixed capacity, learning,
retention, bounded transfers, backing allocation/release and exact state restore.
This is an untuned engineering baseline, not plan completion or a speed gain.
Implementation checkpoint `7125f8a` supplies the native binary; the focused build,
CPU ownership, row-accounting, planner and checkpoint checks remain valid.
Checkpoints retain rosters, scores, admission
credits and grouped service state in-process; exported JSON is a review record,
not a cross-restart image.
No parameter sweep or qualified performance selection has run in this phase.

**Service-calibration capture correction:** the first timing screen measured
12.5137 tok/s pooled (Python 11.7747, prose 13.3515), but its common checkpoint
retained the one-time post-prefill service recalibration performed during the
earlier Nsight/detailed capture. The size of that instrumentation effect on
learned prices and policy is unmeasured. Preserve this as
[diagnostic timing](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/starting-engine/baseline-a-qualification.json),
not a normal-calibration winner or completed starting-engine comparison. On the
already required next loads, run the first independent conditioning request with
Nsight stopped and detailed capture off so recalibration finishes before tracing.
Then retain those prices across conditioning and comparisons. Repeat only the
affected baseline timing; unchanged integration checks remain valid. The extra
companion for the profiled-calibration timing is not run.

**Calibration input repair integrated at `e6d0cb4`:** the primary agent's matched-layer
[diagnostic](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/profiled-cost-model-diagnostic.json)
finds median helper prediction / observed ready-and-reduction proxy of 3.69 on
2,738 generation layer calls; CPU wall time / prediction was about 1.7 in the
same profiled check. These ratios do not isolate a cause or measure a speed gain.
Source inspection shows repeated timing of only four CPU records and no sustained
helper warm-up. The next build rotates existing CPU HOST records across up to 64
identities (no weight allocation or cache flush) and warms helper service for a
bounded 0.5 seconds before its short measurements. This repairs known input
representativeness issues without adding a parameter sweep. The native build passed;
the profile-free bootstrap returned the correct answer, and its unprofiled
post-prefill calibration recorded 64 CPU records and 500 ms of helper warm-up.
Five further independent requests retained the exact service estimates and fresh
KV while primary learning continued with a static ranked helper. Full ownership,
437/741/6,118 capacities and zero helper weight traffic were verified. The
[current diagnostic](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/adaptive-companion-reconciliation.json)
also reconciles 9,234 planner calls and 548 primary exchanges, including 471 retained
uploads serving 5,152 later uses. The earlier ownership/backing receipts remain
valid at their source scope; final performance selection remains open.

**First usable-engine comparison:** the exact original executable measured
12.5713 tok/s pooled under the current protocol. The corrected current baseline
measured 13.3001 tok/s, a first-screen gain of 5.80% (Python +8.91%, prose +2.71%).
[Evaluation and limits](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/starting-engine/first-current-comparison.json).
Both responses reached 128-token caps. This is not final confirmation or a quality
pass, and does not close remaining optimization. The same current adaptive arm is
reused as the helper-test control. The completed helper comparison is inconclusive:
adaptive scored 13.3001 then 13.7574 tok/s; static scored 14.0226 then 13.4545.
Static's pair effects reverse (+5.43%, then -2.20%). Across both orders it is
1.54% faster pooled, with Python 6.00% slower and prose 10.22% faster; sampled
outputs and speculative work differ. Static verified zero helper weight traffic
and replacements. The common restored logical state and identity rosters match;
physical frame assignments can differ. [Primary-agent evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/helper-comparison-evaluation.json).
No confirmed static advantage or adaptive superiority is established, and the
user's static-win stop condition is not triggered. No further helper repeat is
planned. The separately authorized full-primary backing comparison follows, with
the latest compatible static/exclusive arm reused as its declared control and
both earlier exclusive rates preserved. Final static adoption is not implied.

**Original-engine comparison protocol:** the first comparison above used the
preserved held M-110 executable under the same current high-effort Python/prose
prompts, 128-token caps, seed, sampling, memory budgets and complete-generation
timing. The original executable's SHA-256 exactly matches both the held M-110
qualification and its cooperative source receipt (`00d6dff`); this is the original
binary, not a new-policy control renamed as the reference. Its fixed-primary,
live-helper, seven-draft cutoff recipe remains a diagnostic control. Apply the
independent global helper-first rosters before conditioning, retain normal helper
learning between requests, and do not reset placement before scoring. Historical
primary-first setup is driven by an explicitly labelled synthetic roster-control
file whose resulting identities are checked against the genuine demand histogram.
The existing collector now supports this pinned historical setup and fixed-primary
conditioning; the scored runner verifies uninterrupted process/request history.
The old engine cannot restore the new learned checkpoint or expose its newer
trace fields. Record those differences and the fresh-KV adapter explicitly.
The short pair established the first comparison; confirmation follows the selected
final candidate. It does not replace the isolated helper/backing work or value tuning.

**Full-primary backing comparison completed:** backed scored 13.1431 and 13.1434
tok/s; its declared exclusive controls scored 13.4545 and 14.5047. The earlier
compatible exclusive score of 14.0226 is also retained. Across the declared pairs,
backing was 5.85% slower pooled (Python -7.53%, prose -4.04%). The exclusive rates
and sampled work vary, so this is no demonstrated practical gain, not an isolated
measurement of the cost of backing. [Evaluation and raw receipts](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/backing-comparison-evaluation.json).
All 437 primary records were backed with 13,918,797,824 additional bytes; allocation
left 18.68 GiB Windows physical headroom. Both ranked backed runs verified zero
primary/helper expert-weight D2H and zero helper replacements. The detailed
companion reconciled 10,032 planner calls and 630 primary exchanges, including
539 retained uploads serving 4,841 later uses. Disabling backing repatriated 182
canonical HOST records, freed the full bank and retained ownership/capacity.
The mode remains available and currently disabled. The measured loss does not
justify rejecting backing or closing admission/scheduling optimization; the
[focused policy diagnosis](#policy-evaluation-correction) supersedes the earlier
decision to stop investigation here.
Nsight reconciliation of the three new companions passes the selected request,
layer, round and expert-weight copy joins on both GPUs: 28,956 layer calls total.
Each report also records five incomplete CUPTI events dropped at profiling stop,
plus possible-loss warnings; this is not a claim of complete coverage of every
event. [Capture index](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/companion-capture-index.json).
Their raw reports and complete native timelines are retained. Final adaptive-helper
operation, required parameter selection and final original-engine confirmation
remain open.

**Cache-prior value result:** 0.10 gained in both conditioned orders over 0.05
(+11.27%, +7.67%), with live helper, exclusive primary and all other controls fixed.
The pooled pair totals are 15.3507 versus 14.0219 tok/s (+9.48%); Python +16.07%,
prose +3.25%. Reverse-order prose was 2.88% slower, retained alongside the gains.
Both values used high request effort; the candidate Python output began final code
directly while the controls reasoned. These are actual-work speeds, not equal-token
compute. [Primary-agent evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/bonus-comparison-evaluation.json).
Carry 0.10 as the candidate, pending complete-task usability; no final default
or quality claim is made from the capped outputs. The changed configuration has
a bounded diagnostic companion with complete native capture and reconciled
selected Nsight joins. Next compare lenience 0.4 versus 0.2 at bonus 0.10, changing
only lenience; reuse the compatible 0.4 controls. This evaluates lenience directly,
not a translation of historical cutoff rho. Material residency/cooperative-work
assessment and final original-engine confirmation remain required.

**Lenience and admission-budget decision:** at bonus 0.10, lenience 0.2 measured
14.8415 tok/s versus the latest 0.4 control's 15.3932 (-3.58%; Python -4.50%,
prose -2.70%). Retain 0.4 without another ranked repeat; preserve the earlier
compatible 15.3084 control and the capped outputs. A single prose diagnostic
captures the changed verifier configuration. [Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/lenience-comparison-evaluation.json).

The primary agent's [causal forecast sensitivity](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/demand-age-budget-sensitivity.json)
holds recorded original demand fixed and omits actual prices, transfers, active
readers and physical-frame constraints. It is not a production replay or speed
prediction. At the current budget, faster aging changed hypothetical primary hits
only 1.5–3.5%; raising the admission budget from 8 to 16 changed them 19–26%, with
roughly twice the admissions. Actual captures already show the current row-based
budget being spent and useful in-place retention. This prioritizes one cheap live
8-versus-16 budget comparison, keeping age 0.95 per five rows, horizon 20 and
physical capacity fixed. Extra preservation/refill traffic may erase its benefit;
inspect that tradeoff in complete generation. Do not infer a live hit rate or
optimized decay from the forecast. The [declared protocol](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/budget-protocol.json)
reuses the compatible bonus-0.10/lenience-0.4 controls and allows one reversed
confirmation if promising or close, not a budget sweep.

**Admission-budget outcome:** budget 16 measured 15.5718 and 15.2987 tok/s,
versus the declared budget-8 controls 15.3932 and 15.3929. Pair directions differ;
the pooled difference is +0.27% (Python -2.55%, prose +3.07%). Retain 8 under the
declared tie preference: 16 roughly doubled optional primary exchanges and
preservation traffic without a demonstrated generation-speed advantage.
[Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/budget-comparison-evaluation.json).
This does not prove 8 is faster. The first single-prose diagnostic accidentally
used the runtime default of 8; it is preserved as invalid budget-16 evidence.
The collector now explicitly pins and checks that control, and the corrected
budget-16 capture reconciles native events and selected Nsight joins. Neither
diagnostic rate is ranked; no further budget repeat is planned.

**Cooperative CPU-price refinement:** the primary agent's
[unprofiled accounting](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/unprofiled-cpu-price-bound.json)
compares actual generation CPU batch time with the recorded calibration. Even
charging every CPU group the largest measured 1–5-column marginal and every layer
the CPU base charge, observed time exceeds that prediction by 13–22% across twelve task
requests. These are bounds on that model's predictions, not hardware bounds or
isolated critical-path savings. Mixed groups, concurrency and memory/scheduling
effects remain combined. Add one explicit, checkpointed CPU price scale used by
both batch assignment and automatic residency service pricing, plus an aggregate
predicted-CPU counter for direct unprofiled reconciliation. Preserve raw measured
curves and scale 1.0 as the control. A bounded 1.0-versus-1.3 comparison will test
whether this correction improves complete generation; no online predictor or
separate executor is warranted. This is unfinished cooperative optimization,
not another prerequisite added to the already established baseline gate.
Implementation `ebe02b0` builds; the planner and learned-placement checks plus
14 focused frontend/checkpoint checks pass. Its new load completed the profile-free
bootstrap with answer 60, exposed the actual scale 1.0, and retained 437 primary,
741 helper and 6,118 HOST records with complete ownership. One unprofiled
post-prefill calibration was captured, using 64 rotating CPU records and 500 ms
helper warm-up. Independent conditioning and the new common snapshot now pass
[review](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/conditioning-verified.json):
raw prices remained unchanged, learned rows stayed continuous, and the static
helper had zero weight traffic. The direct CPU-time/prediction ratios were
1.23–1.25 in the grouped checks and 0.81 in the ordinary tool check; this shape
dependence remains explicit. The earlier qualified executable remains preserved; none of these
implementation checks selects scale 1.3 or establishes its speed.

**CPU-price comparison:** scale 1.3 measured 16.1732 and 16.0560 tok/s; scale 1.0
measured 15.7816 and 15.0539. Both paired effects favor 1.3 (+2.48%, +6.66%);
pooled rates are 16.1144 versus 15.4092 (+4.58%; Python +3.17%, prose +5.90%).
Control movement and differing sampled work limit the precision of that component
gain. Actual/predicted CPU time changed from 1.21–1.26 at scale 1.0 to 1.00–1.014
at 1.3 in these grouped speed requests. Recorded expert-weight transfer bytes
increased about 13%, so the improved balance is not a reduction in every cost.
[Primary-agent evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/price-comparison-evaluation.json).
Carry scale 1.3 in the candidate; full-task usability and the final original-engine
comparison still follow. Actual inspection verifies scale restoration independently
of the saved report, and detailed records reconcile the new prediction counter.

**Next bounded scheduling check:** compare two draft proposals (at most three
verification rows) with four (at most five), at prior 0.10, lenience 0.4, CPU scale
1.3 and primary budget 8. Reuse the compatible four-draft controls. Current
verification evaluates more target rows than it commits; fewer proposals can
reduce discarded tails but require more target rounds and reduce weight reuse.
Use complete-generation results, not acceptance alone, with one reversed
confirmation if promising or close. The [declared protocol](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/width-protocol.json)
defines the tie preference and actual-width verification; it is not a width grid.

**Scheduling outcome:** retain four drafts. Two drafts measured 15.0162 tok/s
versus the compatible 16.0560 control (-6.48%; Python -13.24%, prose +0.97%).
Both Python runs evaluated 153 target rows, but two drafts required 52 drafting
batches versus 32; shortening the window did not reduce its target work.
Outputs differ, so this is not isolated launch-cost attribution. No further
ranked repeat or width grid is warranted. [Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/width-first-evaluation.json).
The [timeline audit](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/drafting-audit.json)
verifies actual proposal/window sizes, draft counters, scalar lenience probabilities
and decisions, and captured residual-choice support. The existing confidence floor
0.3 pruned five of 32 coding batches and 16 of 43 prose batches at four drafts;
discarded work remains charged. Keep that floor and the disabled product cutoff;
these results do not justify a new confidence scheduler or more values now.

The candidate entering final qualification uses **prior 0.10, lenience 0.4, four
drafts, CPU price scale 1.3 and primary budget 8/horizon 20**, with the grouped
planner, live helper and exclusive primary storage. These values are now in the
serving runtime defaults. The final original-engine comparison, complete-task
checks and resident source/resource handoff below complete their qualification.

**Selected-candidate usability passed, 2026-11-04.** Both complete `count_runs`
responses pass five functional cases each. The high-effort scheduling response
stopped naturally at 2,146 tokens: both 11/12 schedules are feasible and the
minimum-time arguments pass primary-agent semantic review. The real `read_file`
round trip returned version 7.4.2 on its first raw call, with no repair or retry.
The facts-only library announcement completed in 92 tokens with all supplied
facts and no added policy. Disconnect was observed, native retirement ended in
`cancel`, and the follow-up returned READY. All completed workflows retained the
selected effective settings. [Reviewed outputs and capture limits](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/quality-reviewed.json).
These are bounded usable-task checks, not broad quality qualification. Their
native timelines have no dropped events; the cancelled API is intentionally
partial. GPU/expert detail is supplied by the same-source selected-configuration
companion, not claimed for these unprofiled full-task intervals.

**Final original-engine confirmation passed, 2026-11-04.** Across the declared
orders, the exact original executable measured 12.5713 and 13.1024 tok/s; the
selected engine measured 16.1732 and 16.0560. Pooled rates are **12.8314 → 16.1144
tok/s (+25.59%)**, with Python **12.3497 → 16.6072 (+34.47%)** and prose
**13.3522 → 15.6500 (+17.21%)**. Both pair effects are positive (+28.65%, +22.54%).
[Primary-agent evaluation, all outputs and latency diagnostics](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-original/final-comparison-evaluation.json).
This demonstrates a bounded usable complete-engine gain, not isolated feature
speedups or a hardware optimum. Compatible candidate arms were reused as declared
before confirmation; every speed response reached its 128-token cap. Candidate
Python entered final code directly while the original reasoned, under the same
high request effort. Separate completed tasks supply usability evidence.

Configured memory budgets, prompts/order, caps, seed, target sampling and whole
generation timing match. Actual residual allocation on the final original load
was 438 primary/741 helper/6,117 HOST, versus 437/741/6,118 in the earlier arms.
The final original ran standalone; earlier scored arms had Nsight launched with
collection off. These differences prevent a claim of identical physical allocation
or instrumentation overhead. Original control movement was 4.22%; it is preserved
alongside the larger gains. The original cannot restore the new learned checkpoint;
both used independent helper-first ranks and retained their respective ordinary
learning. The subsequent final-load check and observed default/resource handoff follow.

**Final-load variation and handoff, 2026-11-04.** The normal standalone reload
changed automatic service estimates; observed/predicted grouped CPU time became
1.17–1.22 (ordinary 0.74). The one allowed targeted check was declared for that
specific uncertainty, with no parameter change. It measured **14.7816 versus
13.1024 tok/s (+12.82%)** against the final original arm, with matching standalone
environments and actual 438/741/6,117 capacities. Python was **13.2709 versus
12.4264 (+6.80%)**; prose was **16.6803 versus 13.8561 (+20.38%)**.
[All outputs, service/work counters and limits](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/final-load-evaluation.json).

This check is not pooled into the earlier two-order result: final independent
conditioning used live helper learning, automatic prices changed, and sampled work
differed. Coding was 19.65% slower than the earlier selected-b arm while evaluated
target rows rose from 153 to 192 (+25.49%); target time per evaluated row was nearly
unchanged. This supports work-count sensitivity without isolating its cause. Prose
was faster. The current usable engine still improves both tasks over the original;
the 16.11 rate is not a promise for every reload. No additional repeat is planned.
Approximate service pricing retains meaningful shape/load error; its numerical
fit is not itself the completion or speed criterion.

The final process loaded without a placement profile. Its 32-token bootstrap
reached the cap in reasoning and is only an operation check; the earlier same-source
profile-free bootstrap completed correctly. Independent helper-first initialization
then preceded five fresh-KV requests with continuous primary/helper learning and
one unprofiled post-prefill calibration. The ordinary tool caller verified CPU scale
1.3. [Operating receipt](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/operation-reviewed.json).
A normal API request with no tuning, effort or sampling overrides returned READY
and verified the selected effective controls; its 195 cached prefix tokens are
normal serving reuse, not the fresh-KV benchmark protocol. At 16:53 UTC the server
was idle with zero queued requests, complete ownership and weights resident.
The HOST bank is 194,831,712,256 bytes (181.45 GiB); observed Windows physical
headroom is 41.30 GiB, above the 16 GiB reserve. GPU total usage is 30,447 MiB on
5090 and 23,132 MiB on 3090, including driver/other use. No primary backing or HOST
compression is active. Both final speed requests recorded zero native expert-disk
service and zero Linux process `read_bytes`; these are not a Windows driver/pagefile
trace. Resource figures are point-in-time observations.

### Current implementation and parameter dispositions

These are the prior dispositions and the evidence behind them. The
[policy evaluation correction](#policy-evaluation-correction) reopens admission,
scheduling valuation and backing selection, including affected parameter choices.
Their implementation checks and measured rates remain evidence, not sufficient
justification for closing those decisions. Other scoped findings remain reusable.

- **Cooperative service and scheduling:** keep the new whole-batch grouped-cost
  planner and CPU price scale 1.3. The live comparison improved 4.58% pooled and
  brought actual/predicted grouped CPU time to 1.00–1.014. Packed CPU tiles already
  reuse each 144-byte weight unit across up to eight columns; the selected verifier
  uses at most five. Extending that width is inactive for this workload. Keep
  15 CPU workers plus the caller, the existing fine output-row partition and the
  primary FP4/helper packed kernels. One compute participant per physical core
  avoids adding SMT competition; no measured deficit in this partition justifies
  another worker grid after the service-price repair. Unprofiled host planning is
  52–59 ms per 7.7–8.2 s request. The helper's conditional dispatch exposure is
  222–376 ms (about 3–5%), overlapping other work and not isolated removable time.
  Retain the worker dispatcher; the tested mapped queue remains parked.
- **Drafting and grouped verification:** keep four drafts, confidence floor 0.3
  and the disabled product cutoff. The two-draft test lost 6.48% and did not reduce
  coding target rows; the [drafting audit](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/drafting-audit.json)
  reconciles actual widths, proposals, acceptance, residual decisions and discarded
  work. Confidence pruning already limits weak tails. No new scheduler or wider
  grid is justified by that result. Lenience 0.4 remains selected after the direct
  0.2 comparison; lower acceptance alone was not the decision criterion.
- **Routing and device preference:** select the measured uniform prior 0.10.
  The [selected trace audit](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/routing-service-audit.json)
  checks 10,317 generation routing rows and 41,268 mixture/scaling entries at
  recorded precision. All four IDs are valid/distinct, ranks follow their recorded
  scores, and mixtures use original sigmoid weights with scale 2.0. It observes
  7,142 substitutions: 2,692 toward primary-ready experts and 4,450 toward helper.
  Median base fourth/fifth gap is 0.02097; the largest displaced gap stays below
  the 0.10 bonus. Predicted limiting lanes vary across CPU, primary/copy and helper;
  those are model outputs, not measured critical paths. Keep the measured uniform
  policy: the evidence does not justify a separate device-specific or score-range
  policy now. Inactive substitution-ratio knobs do not count as tuning this prior.
- **Primary retention, refill and capacity:** keep eight admissions per five
  generation rows, 0.95 decay per five rows, horizon 20 (100 rows), measured copy
  and scaled service prices, and zero added margin/interference surcharge.
  The budget-16 comparison was practically tied (+0.27%) with roughly doubled
  optional exchanges; retain eight. The causal score forecast predicted a much
  larger hit opportunity that failed to become a speed gain: it omits prices,
  leases and changing text and is not a production simulator. Faster decay gave
  only a small forecast change at budget eight; slower decay was worse. Combined
  reuse evidence previously supported retaining the current age/horizon/margin.
  That disposition is reopened: raw reuse and a score forecast do not establish
  net value under the actual grouped/overlapped service. In the selected prose companion, 237 demanded
  uploads stayed in their frames and served 2,235 later uses; 58 of 68 subsequently
  evicted admissions had executed uses, seven exclusively from cache-induced
  substitutions; still-resident outcomes are censored. These counts are not
  original-demand benefit or net savings. Fifty
  additional uploads and outgoing preservation are charged separately. Eight
  working frames and two exchange slots retain fixed total capacity; the checked
  production ownership/availability machinery bounds admissions. These observations
  do not establish universal residency economics or optimal numerical values.
- **Helper placement, capacity and transfers:** retain global independent-demand
  helper-first initialization, continuous live adaptation and coordinated
  ownership. The isolated static/live comparison is inconclusive, so it provides
  no basis to remove required live behavior. Keep the 741-record helper and
  437-record primary budgets with their established workspace/headroom; no enlarged
  bank is justified by the inconclusive policy result. The full-primary backing
  comparison was 5.85% slower pooled despite eliminating expert D2H. Backing is
  currently disabled, but its policy interaction is reopened rather than accepted
  as a reason to reject the design. Helper activation/result traffic remains distinct.
- **Transfer implementation and overlap:** the new planner prices complete grouped
  service and one upload per distinct record, replacing anchor-first staging.
  Retained uploads avoid a redundant incoming copy; optional refills stay bounded
  and nonblocking. Preserve the actual overlap and source accounting in the
  [companion audit](evidence.md#companion-overlap-and-capture-audit) and new captures.
  CPU/copy durations overlap and cannot be added as critical-path costs. A GPU
  demand-fetch port would still need CPU/helper assignment and ownership retirement;
  neither the old Windows port nor traced launch groups establish a present WSL
  gain. The measured planner/price repair addresses the observed assignment defect;
  current evidence does not justify a new fetch executor. This is an engineering
  investment decision, not a claim that all further gains are impossible.

**Retained implementation-gate check status:** implementation checkpoint `7125f8a` builds; nine focused
CMake/CPU checks and the affected Python checks pass. The full model has now
loaded from the independent frequency profile at
[the source-bound baseline launch](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline/launch.json).
Its initial memory-reserve wait completed and the server reached READY. Startup
inspection verifies 437 primary + 741 helper + 6,118 HOST identities, full ownership,
global frequency-ranked rosters and the initialized frequency prior. Four grouped
requests and the corrected ordinary caller completed with validated captures;
retention/refill activity is present. The first ordinary payload was rejected by
the frontend before execution because it combined `spec=1` with lenience; it is
preserved, and only that unexecuted request was corrected. Full trace reconciliation
and actual learned-state restore now pass. Across five executed requests, 885
evaluated generation rows retained the learning clock; 361 demanded uploads were
retained in their existing frames and served 9,096 later expert uses (7,227
original-demand uses). Of the 218 retained admissions evicted within the capture,
206 were used before eviction; still-resident admissions remain censored. All
18,354 captured ordinary/grouped planner calls reconcile with the implemented
decision, and ownership remains complete. The joined restore reproduced saved
rosters, scores, admission credits and service state after changing 295 primary
and 130 helper identities. These are implementation checks, not a speed improvement
or evidence of optimized parameters. [Receipts](evidence.md#baseline-integrated-behavior).

The first full-primary backing allocation was safely refused before allocation: its
13,918,797,824-byte bank plus the 16 GiB reserve exceeded the observed Windows
physical availability of 29,572,321,280 bytes. Guest availability alone does not
establish Windows headroom. The failed receipt is preserved; no reserve was lowered
and no capacity was reduced. After user-assisted host reclamation and the explicitly
requested REA/Claude process cleanup, the retry allocated all 437 backing records
with 16.35 GiB Windows physical headroom remaining. Two bounded grouped/ordinary
requests issued 89 primary exchanges with zero expert-weight D2H; five-second
resource samples stayed at or above 16.23 GiB. Disabling backing repatriated 71
canonical HOST records, released the full bank and preserved complete ownership.
The common learned-state hash then restored exactly. Native trace/counter checks
pass; Nsight matches all 4,674 layer calls but captures 4,271 of 4,272 submitted
weight uploads. Its possible-loss warnings and this one-copy gap remain explicit;
they are not reported as complete GPU-event coverage.
The combined check used live helper adaptation to exercise that implementation.
A separate declared five-request conditioning check has now established the
common static-helper state: independent global helper-first ranks, primary
adaptation on, fresh KV per request and continuous learned residency. The helper
roster stayed unchanged with zero weight traffic/replacements. Its 959 evaluated
generation rows retained 429 uploads in place, serving 12,002 later uses. All
19,038 planner calls and trace/counter joins reconcile; ownership remained complete.
The resulting in-process [common checkpoint](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/common-checkpoint/learned-checkpoint.json)
was saved at generation row 1,844. This establishes observed operation under the
declared history, not convergence, optimality or a speed gain. That earlier
in-process checkpoint was superseded on the corrected normal-calibration load;
the current helper/backing comparisons use its separately recorded common state.
The extraction response preserved the supplied facts but added a note beyond the
requested three bullets; the capped outputs are not complete-task passes.
These earlier receipts establish the baseline gate. Subsequent helper/backing and
parameter decisions are recorded above; the active work table owns what remains.

### Placement and residency qualification correction

**User correction, 2026-11-04:** treat initialization, persistent learned state,
demand accounting/aging, global primary allocation, helper placement/adaptation,
retention/refill, transfer scheduling and selective DRAM backing as **one operating
system**. Sequential placement and 0.95-per-forward aging were inherited from
upstream Maya; main-bank adaptation was added later. Checkpoint `4e8792c` removed
the saved startup profile and profile output without establishing a replacement
warmed baseline. This is a setup regression as well as a longstanding validation
gap; no corresponding execution-speed regression has yet been isolated.

The legacy trace continued replacing experts through the final generation forward.
Its primary/helper admissions were disabled during prefill; the 66 coding-prefill
forwards therefore did not condition those banks. The 128-token output limit and
bounded admission cadence left most initial primary identities in place. Prose
continued that partially adapted history. Neither that observation nor a fixed
suffix cutoff establishes convergence or a permanently fixed fraction of the bank.

**Do not use the affected cold/adapting results to select warmed policies.** Preserve
the priced-policy comparison/replays, parameter screens, historical recipe checks
and helper screen at their actual initial-state scope. The 6.89% priced-policy gain
is a real result against its sequential-start legacy control, not validated warmed
LRFU optimization. Horizons 20/40 and budgets 2/8/16 on the cold trace do not resolve
aging, admission economics or warmed behavior. Numerical defaults remain provisional.

<a id="static-placement-order"></a>

**Static placement amendment, 2026-11-04:** for every static GPU placement, rank
experts globally across all MoE layers by **original-router demand frequency**
from the declared independent representative history. Fill the **3090 first**
with the most frequently demanded experts up to its actual resident capacity;
fill the **5090 next** with the most frequently demanded experts remaining.
Use the same ranking for demand-derived initialization of adaptive banks.
Expert IDs and per-layer allotments are not popularity ranks. Executing a
cache-selected substitute does not add original-demand credit. Retain the source
history, counting/phase scope and resulting ranked rosters with the experiment;
do not fit them to scored prompts. This governs roster selection at setup.
Adaptive banks may subsequently change through ordinary online policy; the rule
does not require reseeding between requests or frozen adaptive cache contents.

After the [baseline implementation](#implementation-first-then-measurement-and-tuning),
use the sequence below for conditioning and comparisons. Its setup/capture repairs
and necessary demand collection are implementation prerequisites, not reasons to
postpone known execution-policy repairs:

1. Complete faithful common-state setup using the implemented capture/checkpoint
   controls and remaining affected checks. Legacy `PLACEMENT all` resets live
   scores and chooses primary before helper; a demand histogram is not a
   learned-state checkpoint. At an explicit joined setup boundary, save published
   primary/helper identities, normalized demand scores, policy configuration and
   learned service/scheduling state. Restore once per arm, with ordinary staging
   cleared equally and that limitation reported. Do not add joins, forced cache
   contents or resets to ordinary inference or redo completed component repairs.
2. Inventory existing independent high-effort original-router demand. Reuse only
   histories with known prompts/settings/phases and no scored-prompt fitting.
   Fill missing coverage with a declared bounded independent history, selecting
   both GPU rosters in the [required frequency order](#static-placement-order).
   A profile supplies initial evidence, not proof of warmed operation. Preserve
   normal learned state across subsequent independent fresh-KV requests.
3. Demonstrate the operating behavior on independent demand: original/executed
   coverage, miss service/weight traffic, useful retention and rereads, turnover,
   unavailable frames, admission reasons/costs and complete latency over request
   and round history. Ongoing turnover is allowed. No arbitrary token cutoff,
   requirement for frozen identities or indefinite warming substitutes for this
   evidence. Material continuing loss/churn calls for a policy repair, then a
   bounded affected check, not longer warming until the result looks favorable.
4. Re-establish the usable original-engine baseline under that declared history
   and current timing/workload protocol. Redo only the decisions invalidated by
   initialization or materially interacting policy changes. The helper-only
   comparison below must start with a conditioned primary and change only
   `helper_live`; the separately authorized primary-backing test follows it.
   Revalidate aging/admission/refill and interacting width/prior/lenience decisions
   using the smallest useful experiments. Unaffected functional checks remain valid.

In the inherited policy, original demand was credited per evaluated row (including
rejected speculative rows and prefill), while 0.95 aging and admission opportunities
advanced per target forward. Changing width or prefill chunk size therefore changed
their relationship to emitted tokens and request history. The baseline's defined
generation-row clock and separate prefill accounting replace those mixed units;
their numerical values remain provisional. Neither a replacement count nor a
hit-rate improvement alone qualifies the resulting policy. The
[evidence correction](evidence.md#placement-and-residency-audit)
preserves the implementation/history findings separately from unmeasured remedies.

**Initial prerequisite implementation, `2a5bcab`:** an in-process joined checkpoint
now saves/restores published rosters, normalized scores, primary policy and learned
lane timing; ordinary staging is cleared equally at that explicit boundary.
Helper-first profile placement is available without changing live admission rules.
The detailed capture buffer now compresses records in memory and retains explicit
overflow reporting. Focused CPU ownership/state/capture checks and the native build
pass; actual-model restore and complete new capture remain to be demonstrated.
Neither checkpoint JSON nor a generation profile is a durable cross-restart state.

**Prepared independent-history protocol:** the old saved profile's actual requests
used none/low effort, and the available recent traces include scored prompts.
Reuse any valid collection already completed or underway; otherwise collect one separate
high-effort five-request history (code, prose, extraction, reasoning, tool proposal;
at most 384 output tokens per request), then apply global helper-first demand ranks.
After the baseline implementation and its affected checks, run one independent
five-request behavior check (at most 192 tokens each), primary adaptation on and
helper static, retaining state between requests. These are bounded
collection budgets, not warming thresholds. The protocols and provenance are under
[conditioned-residency](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/history-provenance.json).
Inspect original-demand coverage, service/miss traffic, retention/turnover and
latency trajectory; if continuing admission loss is material, diagnose it before
scoring policies rather than extending warming. Detailed capture/Nsight timings
are diagnostic. A complete independent operating account and verified restore
permit the matched baseline/helper steps; adverse behavior requires a focused repair.
Do not use this collection protocol to resume tuning the old implementation before
the [implementation priority](#implementation-first-then-measurement-and-tuning) is met.

### Bounded static versus adaptive 3090 comparison

**Latest user correction, 2026-11-04:** begin from a representative, conditioned
primary cache, preserve learned residency between independent requests, and change
only the behavior being evaluated. The completed screen below began from the
arbitrary sequential primary roster. **It is cold-start evidence, not a decision
about warmed helper policy.** Frequency-ranking the helper alone did not condition
the primary. Expert IDs are identities, not popularity ranks.

The [cold receipts](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/helper-static-adaptive/cold-summary.json)
retain both orders: adaptive 11.5124 and static 11.7460 tok/s pooled (+2.03% static),
with substantial Python variation and opposite pair directions. Static verified zero
helper expert-weight uploads/downloads/replacements. Actual capacity was 437 primary,
741 helper and 6,118 HOST records under unchanged launch memory limits; the older
process had 438 primary records. Do not use these results to decide the warmed question.
The older 10.3% versus 39.7% helper demand-coverage calculation is historical placement
arithmetic, not observed cache-hit rate, measured speed or proof of an optimum.

Before the warmed comparison:

1. Complete the [reasonable baseline](#required-baseline-implementation) and its
   affected checks, including the detailed-capture repair. Retain the partial
   earlier captures. Token/round timelines and raw Nsight reports from the cold
   screen survive; some detailed expert/cache events do not.
2. Establish both GPU rosters from separate representative original-router demand
   in the [required frequency order](#static-placement-order). Keep that chosen
   static helper roster while conditioning the adaptive primary from the remaining
   experts, so the common snapshot retains the required static-helper placement.
   Use existing suitable evidence and a declared bounded conditioning/check capture
   as needed; do not fit to scored prompts.
3. Save one common **published placement and learned policy state** from that
   conditioned primary, with primary adaptation enabled. Demonstrate conditioning
   with actual independent demand/service/cache evidence; an applied profile or a
   token cutoff alone is not proof of completed warming. Restore that same state
   for each arm. Preserve normal learned residency between Python and prose, while
   giving each fresh target/DSpark KV. No per-request reseeding or waits for optional
   cache completion are introduced into inference.
4. Freeze four drafts (at most five verification rows), lenience 0.4, cache bonus
   0.05, the implemented baseline's primary policy/CPU/FP4/worker settings and
   memory budgets. Use the same new batch-aware staging and assignment policy in
   both arms; the old requirement to keep staging reuse off is superseded.
   **Only `helper_live` changes**. The static arm must show zero
   helper expert-weight traffic and replacements throughout inference; activation
   and result transfers continue. Use the short high-effort Python/prose screen,
   one pair and one reversed confirmation if static is promising or close.

Report per-task/pooled rates, variation, actual transfers and output limitations.
An inconclusive result does not establish adaptive superiority. A confirmed static
advantage requires an evidence report and no automatic wider optimization or final
static-policy adoption. The user's later explicit backing-copy experiment below is
separately authorized; it is not combined with this helper-policy comparison.

### Bounded primary DRAM backing experiment

**User storage amendment, 2026-11-04:** exclusivity was an implementation requirement,
not a runtime decision that surplus DRAM was unavailable. Temporary staging uploads
already keep their HOST source; main-bank promotion currently recycles that HOST
frame for another expert, without returning the allocation to the OS.

After the conditioned helper-only comparison, run the separately requested test of
retaining DRAM backing for **the entire 5090 resident bank**, with a fully static
3090 containing the most frequently demanded experts across all MoE layers. Both
control and candidate start from the same demand-calibrated GPU placement and learned
primary policy state, following the [static placement order](#static-placement-order).
Keep four drafts, lenience 0.4, prior 0.05 and the implemented baseline's primary
and batch-aware staging/assignment policies fixed in both arms. The control remains
exclusive; only primary backing retention changes in the candidate. Do not mix
that change into the helper-policy test.

Preserve all experts and the existing GPU capacities. Charge the additional HOST
records and bounded workspace, recheck actual Windows physical/commit and guest
headroom, and retain the configured 16 GiB reserve. Roughly 13 GiB can back the primary
bank; neither a snapshot nor that arithmetic proves an allocation will succeed. A
complete duplicate of both GPU banks is not required. In the backed candidate,
primary evictions preserve their already-valid HOST copy rather than downloading
weights; the static helper also issues no expert-weight downloads. Verify **zero
expert-weight VRAM-to-DRAM transfers** in inference, separately from normal
activation/result readback. Do not shrink the model or retune unrelated settings.

The current exclusive control remains available. This bounded backing mode is an
explicit exception to the old exclusive-primary requirement; keep ownership/readers,
ready publication, routing and state correctness, and report physical copies and
logical expert ownership separately.

### Remaining work and retained evidence

**Resumed investigation, 2026-11-04:** working implementation checkpoint `4e8792c`
preserves the previously uncommitted changes. At that investigation's start, the
resident binary matched the v2 source manifest. Its measured screen requested four draft
proposals (`spec=5`), overriding runtime `spec=8`; the old protocol's window-eight
description was wrong. Rates are unchanged. After the
[implementation](#implementation-first-then-measurement-and-tuning) and
[conditioning gates](#placement-and-residency-qualification-correction),
compare the preserved M-110 recipe (fixed calibrated primary, cutoff rho 0.2, seven proposals, live helper)
against the new baseline under the current high-effort Python/prose protocol.
Treat the existing sequential-start lenience, cache-prior and width screens as
provisional evidence and revalidate only affected choices under that gate. Existing timing
puts draft work near 3–4% and conditional helper dispatch exposure below 2% of
generation; prioritize grouped expert service/residency costs. Exact historical
binary reconciliation remains part of the starting-engine gate. No final selection
or completion is inferred from these planned comparisons.

The ordered legacy-policy trace exposes a concrete assignment weakness: experts
are chosen for staging in first-seen row order, giving anchors priority over the
other verification rows. Choosing the most-used eligible HOST experts for the same
staging count covers 24,453 rather than 20,062 generation expert-columns (+21.9%);
this is a fixed-trace opportunity, not speed evidence. The
[anchor audit](evidence.md#anchor-first-staging-audit) also found 163 of 164 anchor
misses staged in one five-row batch that uploaded 209 records (6.20 GiB).
Implement [batch-aware selection and staging amount](#required-baseline-implementation)
as part of the baseline now. A reuse-only reorder at the old staging count is a
useful component, not the complete repair. Retain the old first-seen assignment as
a diagnostic control; evaluate complete performance after the coherent baseline
exists. Existing host-planning/CPU-service/column and transfer counters support that check.

| Workstream | Established evidence to retain | Remaining obligation / status |
|---|---|---|
| Replacement performance predictor | State/decision replay and instrumentation are reusable. Retired timing approaches, the +5.57% foreground miss, the separate +80.2% full-work miss and unresolved interventions remain preserved. | Start with one shared model. The [bounded generation-only diagnostic study](#shared-model-diagnostic-study), with speculation off, precedes selection of its mechanisms. Use measured material effects and useful ideas from the [six proposals](timing-predictor-proposal-measured-transitions.md) and broader literature to write the new first-stage design, then implement and validate policy-decision accuracy. Do not start a layer-specific fitting study or resume retired timing calibration. |
| Reasonable implementation baseline | [Integrated qualification](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline-integrated-qualification.json): actual ordinary/grouped ownership, capacity, continuous learning, useful in-place retention, bounded transfers and full-primary backing allocation/release. | Working integration established at the recorded scope. Admission value and evaluation are reopened by the policy review; correct ownership and component/model execution alone do not establish a reasonable optimized policy. Preserve valid checks and their coverage limits. |
| Required configuration | Official target/DSpark, cooperative CPU/5090/3090, FP4 primary, worker/live helper, prior 0.10, lenience 0.4, four drafts, grouped-cost planner/CPU scale 1.3, raw exclusive HOST storage. | Closed: final profile-free load, ordinary/grouped operation and no-override default request passed; actual resources/ownership are recorded. |
| Complete adaptive design | [Five-part production account](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline-integrated-qualification.json), cross-layer frame movement, retained-upload later uses, row-based original-demand learning, separately priced additional refills/preservation, live helper replacement and exact in-process state restore. Latest selected companion reconciles all planner/ownership events and 237 in-place retentions. | Required behavior established; preserve bounded nonblocking publication and reader leases. Histogram initialization and in-process learned checkpoints remain distinct. |
| Residency optimization | Independent conditioned histories, actual useful/unused/censored retention, measured transfer/service prices, bounded age forecasts, budget-8/16 comparison, isolated static/live and exclusive/backed comparisons. | Reopened: repair grouped marginal-value accounting and evaluate admission decisions, displaced work and the backing interaction. Budget 8 and existing settings are provisional operating values. Static/live remains inconclusive; the backing result does not close this work. |
| Lenience and cache-aware selection parameters | Current conditioned comparisons: prior 0.10 versus 0.05 (+9.48% pooled); lenience 0.2 versus 0.4 (-3.58%). Actual routing/rank/mixing audit and complete-task checks pass. | Select prior 0.10 and lenience 0.4. No original-distribution or broad-quality preservation claim. |
| Inference implementation choices | Whole-batch assignment/staging replaces first-seen behavior; representative service calibration and CPU scale repair; actual packed grouped callers; routing/service audit and overlap/cost evidence. | Reopened for admission/scheduling valuation and their interaction with backing. Retain the actual whole-batch implementation and valid checks while repairing the identified decision-model gaps. Unrelated parked mechanisms do not reopen automatically. |
| Other active parameter families | CPU scale 1.3 gains 4.58% pooled, two drafts lose 6.48%, budget 16 ties; confidence decisions and rejected work reconcile. Current physical-core worker partition, capacity and transfer controls have explicit source/cost dispositions. | Retain the measured values as provisional controls. Revalidate choices materially affected by the admission/scheduling repair using focused modeling and real tests; no indiscriminate sweep. Local gains are not additive. |
| Usable starting engine and final speed | Final declared orders: original 12.8314 versus selected 16.1144 tok/s pooled (+25.59%); Python +34.47%, prose +17.21%, both pair directions positive. All source/outputs/latencies in the [evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-original/final-comparison-evaluation.json). | Closed with bounded scope. Also retain the final standalone-load result 14.7816 versus 13.1024 (+12.82%), Python +6.80%/prose +20.38%. Do not hide the lower reload speed or pool its different conditioning into the earlier result. |
| Functional and state checks | Final selected code, scheduling, real file-tool round trip, prose/extraction, cancellation/follow-up pass. Existing unaffected rejection/continuation/prefix and ownership tests retained. | Bounded usability closed. Natural reasoning completion at 2,146 tokens; no new broad matrix. Old failures/truncations remain recorded. |
| Final composition and handoff | Native source `ebe02b0`, binary SHA-256 `4c060020…384000b`; final server 2085689/engine 2085693 at port 8097. No-override default request passed, weights resident, idle/zero queued, ownership complete and 41.30 GiB Windows physical headroom. [Handoff](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/handoff-reviewed.json). | Closed at the 16:53 UTC observation. Keep the compatible process resident; later status must be observed anew. |
| Capture and artifact reconciliation | Every new configuration has native timelines and bounded detailed/Nsight companions; selected joins reconcile on both GPUs. Complete-task native timelines have no drops; cancelled API is deliberately partial. | Current captures reconciled at explicit coverage. Preserve five incomplete CUPTI stop events/possible-loss warnings in companions, the earlier one-copy gap and old-binary unsupported fields. Final-load and default/handoff timelines are retained with their uncaptured GPU-detail intervals explicit. |


The [companion overlap/capture audit](evidence.md#companion-overlap-and-capture-audit)
supplies diagnostic evidence for the existing cooperative and capture work. CPU
usually finishes last, but its lead is much smaller than its total batch duration;
most demand-copy time overlaps CPU work. Keep joint CPU/copy/5090/helper planning
as the implementation basis. The launch groups observed under Nsight remain an
unproven, bounded scheduling lead, not another baseline prerequisite or a reason
to delay the implementation-first sequence. Reuse the audit's coverage and
provenance corrections without treating either companion as a warmed speed test.

The [earlier sequential-start policy result](evidence.md#current-reference)
remains **512 / 44.777231 s = 11.4344 tok/s**, versus
**512 / 47.863285 s = 10.6971 tok/s** for its new legacy-adaptive control.
It was a local policy improvement with different initial-state/measurement scope,
not the original-engine comparison. The final conditioned comparison above now
provides that required evidence; historical roughly-15 tok/s suffix results remain
separate rather than being treated as a literal target.

The [launch](native-server.json) and [runtime](native-runtime.json) files contain
the selected profile-free/adaptive/lenience defaults, high effort and no forced
thinking closure. Native source is `ebe02b0`, with the unchanged binary identified
in the [source/weight account](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/source-and-weight-account.json).
Later commits update tools, defaults and records. The final [source-bound launch](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/launch.json)
uses server PID 2085689 and engine PID 2085693. At the recorded handoff they were
idle with weights resident at `http://127.0.0.1:8097/v1`; active configuration, logs
and all final receipts are under `conditioned-residency/final-selected/`. The
[observed handoff](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/handoff-reviewed.json)
is dated, not a guarantee of future process state. Earlier v2 observations remain
historical evidence.

Normal launch after checking actual process/resource state:

```bash
.venv/bin/python serve/server.py --engine strata \
  --config docs/minimax-wsl/native-server.json \
  --host 127.0.0.1 --port 8097 --idle-unload 0
```

Here `strata` names the shared frontend adapter; the JSON launches `maya-mm3`.

### Mandatory restart order

The [replacement decision](#timing-model-replacement) supersedes further repair
or qualification of the retired timing model. First observe actual process and
resource state and preserve a compatible resident process if one exists; old
handoff PIDs are dated observations. Do not repeat completed runs merely to
reconstruct a handoff. Retain valid implementation and policy-review evidence.
The held-base comparison remains held. Start in `/home/mdwbr/project-maya`,
branch `minimax-wsl/proposal-2`:

1. Recover the actual source/binary, worktree, running model and resource state;
   preserve compatible resident state and existing changes. Read this plan and
   the relevant evidence/source map. Consult archives only for a specific question.
2. Perform the bounded [shared-model diagnostic study](#shared-model-diagnostic-study)
   before selecting the new first-stage design's mechanisms. Start with one
   shared model; do not run a layer-specific parameter-selection study. Follow
   the [generation-only scope](#generation-only-modeling): ordinary one-token
   decoding first, with speculative decoding off and prefill outside the model.
   Use the measured effects to select and implement the design with the retained
   state/policy machinery, qualify its policy predictions, then qualify speculative
   generation as a later extension. Declare each measurement protocol before running it.
   Do not build all six proposals or restart the old calibration
   campaign. Reuse the established baseline and unaffected checks.
3. Repair the identified admission valuation and complete the affected
   [parameter optimization](#required-parameter-optimization), using supported
   replacement predictions and focused hardware confirmation. Preserve the
   [independent conditioning](#placement-and-residency-qualification-correction)
   and [usable starting-engine comparison](#comparison-with-the-usable-starting-engine)
   contracts. Revalidate only choices and comparisons materially affected by new
   implementation; retain unresolved helper/backing evidence and all losses.
   A new predictor or a local policy gain does not itself complete this step.
4. Reconcile every active workstream and the final integrated server against
   [completion](#execution-and-completion), reusing valid functional/state evidence
   and rerunning only affected checks. Report unfinished work explicitly.

Apply the focused-check policy in [synchronization](#synchronization-and-buffer-retirement).
Completed historical repairs and unchanged checks do not become a fresh campaign
at restart or a context handoff.

<a id="requirements-shared-by-all-three"></a>

## Selected engine

These requirements apply to the Maya milestone and to any base comparison the user
later resumes under the [base-hold policy](#base-holds).

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

Keep expert selection, weight placement and draft-token verification distinct.
The approved verifier change replaced the target-relative cutoff with lenience;
it did not replace the additive expert-selection prior. The separately approved
time-priced retention/refill design governs weight placement. Neither algorithm
change establishes an optimal routing bonus. The routing policy below is the
approved starting point; justified selection-policy improvements remain within
[implementation optimization](#implementation-optimization).

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

**Clarification, 2026-11-04:** cache-aware routing does not authorize using its
executed hits as admission demand. The original top-four IDs before the residency
bonus remain the residency-learning input. Observed executed reuse, original-demand
reuse and net service savings are separate quantities; none substitutes for another.

Keep official DSpark drafting and grouped target verification, but replace the
target-relative cutoff with **lenience-based verification**. For a proposed token
`x`, accept with probability `min(1, p(x) / (ell * q(x)))`, where `0 < ell <= 1`.
Here `p` is the configured approximate target distribution and `q` is the actual
distribution that produced the draft. Verification does not restore original top-4
routing or establish invariance across window widths. Keep target temperature 1
and top-p 0.95. For `ell < 1`,
this intentionally changes the sampling distribution; do not describe it as
distribution-preserving or as restoring unmodified-model quality.

The **0.05 cache bonus** and **lenience factor `ell=0.4`** are approved starting
values. Both require [parameter optimization](#required-parameter-optimization)
before final selection; neither is permanent or a demonstrated optimum. Hold them
constant within each declared residency-policy comparison so its result remains
interpretable. That temporary control does not waive their later tuning.
**Do not reuse or translate the old rho 0.2 as ell.** Address any implicated quality
issue through the bounded diagnosis. The selected mechanisms remain required;
parameter tuning does not reopen their inclusion or the retired combination matrix.

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

An initial implementation of this architecture is integrated and has scoped
measurements. Its [coverage and optimization remain under review](#active-maya-work);
implementation presence does not close the design or performance work. The five
requirements below remain the approved contract.
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
3. **Canonical storage with bounded copies and explicit backing policy.** The
   retained control distributes the uncompressed expert bank exclusively across
   RAM, the 5090 and the 3090. The user now authorizes the
   [bounded primary-backing experiment](#bounded-primary-dram-backing-experiment):
   keep a valid HOST copy for every primary resident within a measured DRAM budget,
   so primary eviction needs no preservation download. This is not a complete RAM
   duplicate of both GPU banks. Keep every expert's last valid copy and all active
   sources protected; account for logical ownership and physical backing separately.
   Necessary transfers in the exclusive control remain real measured costs.
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

No offline placement profile is required for the engine to operate. This does
not require discarding learned initialization or justify using sequential cold
placement as a warmed baseline. A valid saved learned state/profile may improve
startup; retain its provenance and distinguish histogram initialization from
actual adaptive policy state. Start from a valid placement within the existing
RAM/VRAM budget. Static controls and demand-derived initialization follow the
[3090-first frequency ranking](#static-placement-order); follow the conditioning
gate for warmed policy decisions.

<a id="independent-request-state"></a>

For independent requests, keep model weights loaded and learned expert residency
alive without manual reseeding. Give each fresh target and DSpark KV/context,
without conversation history or prefix/KV reuse between these requests.
Expert-weight residency and attention KV state are distinct.

The [donor evidence](evidence.md#adaptive-residency-evidence) explains the upstream
GLM lifecycle and its different per-layer capacity policy.

### Real-trace replay and optimization

**Use replay to tune the chosen design, not to reopen the architecture decision.**
After the [initial implementation](#required-baseline-implementation), use ordered
traces from the [default screen](#default-speed-screen), under
its request-state and workload scope. Histograms alone cannot reproduce
admission or transfer scheduling. Record original demand and executed choices,
request/round/row and layer identities, available copies and pending transfers,
prefill versus generation, and the service observations needed to price decisions.
Include attempted speculative rows that are later rejected. Existing traces may
inform the initial implementation without another replay or live comparison first.
Reuse instrumentation and add only missing fields. If capture perturbs throughput, use it for replay
and turn it off for the actual speed comparison.

Compare the implemented baseline with decision-relevant policy alternatives.
Tune a small number of controls, such as aging, admission margin and refill budget.
Do not require separate implementations of every donor cache or a broad sweep
over unrelated replacement algorithms.

Hold total memory capacity constant, including working and exchange storage.
Charge incoming transfers, necessary outgoing preservation, occupied/unavailable
frames, CPU-memory contention, helper interaction and competing useful service.
Represent overlap without summing simultaneous work as if it were all exposed.
Where timing is unknown, preserve uncertainty. The current
[replacement decision](#timing-model-replacement) retains logical state and
physical-resource constraints while replacing per-operation timing reconstruction.
Demonstrated prediction of policy differences selects the timing abstraction;
passing the retired timing model's validation gates is not required. Both older
timing approaches remain historical diagnostic evidence only.

Rank supported predicted performance differences on common work; use hits,
transferred bytes and replacement counts to explain the result. Fixed-trace
replay cannot prove final speed or quality:
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

### Required captures

**User instruction, 2026-11-04, applies throughout the remaining plan.** Extend the
existing engine/collector logging; do not create a separate benchmark harness or
change prompts, budgets, request order, cache history or policies for capture.

- Give experiments, runs and requests shared unique IDs linked to the central-plan
  item. Preserve control/candidate/order, commit/diff and binary hash, exact prompts,
  seed, model/tokenizer identity, requested/effective settings, actual draft count
  and total verification-window width as distinct fields.
- Record complete token/round timelines from generation start through completion,
  cancellation or failure: token IDs, positions/emission timestamps, rounds and
  target forwards, all proposals/acceptances/rejections, replacement/bonus tokens,
  discarded work, committed outputs and reasoning/final distinction. Lenience
  diagnostics retain actual proposal/target probabilities and decisions.
- Add correlated NVTX request/round/forward/layer and phase ranges. Nsight captures
  both GPUs' kernels, copies and streams using `--trace=cuda,nvtx
  --cuda-graph-trace=node`, alongside CPU expert work/helper submission/collection.
  Cover each remaining benchmark configuration and relevant test windows; name
  every uncaptured interval and unsupported historical-binary field explicitly.
- Preserve reconstructable expert/cache records: original/executed IDs, selection
  scores and mixture weights, ready residency, actual CPU/resident/staged/helper
  assignment, initial ownership/frame maps and ordered admission/retention/eviction,
  transfer/publication and pending/failed events with directions, bytes and decision
  inputs. Retain completed-round counters, before/after resources and allocations,
  service/jobs/columns, transfers and activity, separating prefill from generation.
- Preserve raw test output, expected/actual checks, errors, truncation and state
  transitions, including reasoning review and tool repairs/retries/costs. Tests
  without generation need operation timelines/results, not fabricated token records.
- Document clocks, units, counters and overlap. Do not fabricate per-token compute
  times or sum concurrent intervals into wall time. Buffer records and write outside
  timed work where practical, with no added CUDA synchronization, optional-transfer
  waits, per-token disk flushing, reseeding or longer output solely for capture.
- Keep detailed tracing off in ranked speed measurements while overhead is unknown
  or material. Use already planned diagnostics or a clearly bounded companion with
  matching source/settings and declared cache history; record mode/overhead/coverage.
  Account for capture serialization even after a buffer fills; a trace's host-plan
  interval can include instrumentation work absent from normal inference. The
  [companion audit](evidence.md#companion-overlap-and-capture-audit) records this case.
  No completed comparison is rerun merely to fill a visualization or historical gap.
- Store raw Nsight reports, JSONL and run-indexed manifests under the existing
  `/mnt/wsl/transfer/minimax-wsl/runs/` tree. Validate joins, token counts, round totals
  and trace/counter reconciliation. Missing fields, dropped events and incomplete
  coverage remain explicit, never zero-filled or overwritten. Preserve adverse and
  inconclusive results and reuse valid existing captures.

### Effort and harness

Evaluate MiniMax at **high reasoning effort or higher**. High is the default for
quality checks and both speed requests; a supported higher level may be declared.
Hold effort constant within comparisons. The user reports poor reasoning and
other output quality below high. Do not lower effort to shorten answers or inflate
rates; historical lower-effort receipts do not qualify high-effort behavior.
This is a model request setting, separate from coding-agent delegation.

The existing harness/fixtures now request high effort and use the replacement
reasoning task. Preserve those implemented changes, verify effective serving/template
settings in final measurements, and do not restore the old forced `none`/`low` settings.

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

### Comparison with the usable starting engine

Before closing performance work, identify and preserve the best usable, reproducible
starting configuration and its source/binary, settings and original measurement
scope. Preserve that reference now; implementing the
[reasonable baseline](#required-baseline-implementation) does not require completing
another historical-recipe benchmark first. After implementation and conditioning,
compare it with the new baseline and final candidate using the same current Python/prose
prompts, high-or-higher effort, output budgets, target sampling, memory budgets
and whole-generation timing. Record each arm's actual execution and initial cache
history; preserve normal within-arm adaptation. A previously saved placement profile
may describe the historical control, but no new fitting to scored prompts is allowed.
The final engine must still operate without a mandatory profile and meet the
approved adaptive requirements. Optional learned initialization is permitted;
profile-free operation does not require a cold reset before measurement.

Report this whole-engine comparison alongside policy comparisons within the new
configuration. A legacy-policy arm that already contains new verifier, placement
or runtime changes cannot by itself stand in for the earlier usable engine.
The roughly-15 tok/s historical figure is not a literal target under a different
protocol. If the original recipe cannot be reproduced or the conditions cannot be
reconciled, record the missing evidence and keep performance completion open.

Diagnose a material loss with the cheapest checks that can change the next action.
Include costs of required new behavior; do not assume effort, lenience, adaptation
or timing boundaries explain the gap without evidence. Historical policies may
serve as diagnostic controls without becoming eligible final designs. This does
not authorize a compulsory routing/speculation/cooperation matrix or removal of
required features. Apply the existing [selection rules](#cost-observations-and-selection)
to the relevant starting-engine comparison as well as local candidate comparisons.

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
Truncation is a budget/completion limitation, not by itself an incorrect reasoning
answer. The [fixture/checker](../../tools/mm3/native_reasoning_check.py) is implemented;
[current evidence](evidence.md#current-reference) preserves its completed and truncated outcomes.

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

Evaluate branch finish times and their margins as well as work duration. A high
CPU-last fraction does not make all CPU batch time exposed, and reducing that
branch can expose the copy/5090 or helper branch. Use the small batch-cost model
to assess the combined assignment; do not sum overlapping CPU and copy durations
or convert diagnostic finish margins into a production speedup ceiling.
The [companion evidence](evidence.md#companion-overlap-and-capture-audit) motivates
this accounting but cannot qualify uninstrumented performance. Any later launch
submission check must establish behavior without Nsight and charge its own cost;
the traced groups-of-11 pattern alone does not establish a production benefit
from flushing or make it a required baseline repair.

Among implementations meeting the requirements, declare a preference before
measurement for practically tied results; do not default to incumbency. Roughly
2% is a practical closeness band, not a target for repeated testing. Operational
selection does not prove equal speed; noisy unresolved effects remain inconclusive.
Investigate a 5% task-class regression; a confirmed >10% regression blocks default
selection until addressed. Review sampled output/work differences before calling
a loss repeatable and report all task outcomes.

## Implementation optimization

The remaining work includes implementation choices in the combined engine, as
well as parameter values. Reuse the established packed kernels, grouped execution
and ownership machinery. Their presence is a working foundation, not evidence
that every material execution cost or omitted donor mechanism has been resolved.
The [source maps](proposal-1-cooperative-experts.md#donor-map) preserve concrete
references; they are not a requirement to port every donor.

### Required baseline implementation

Implement these connected behaviors before the next ranked optimization campaign.
Use the best justified initial design, with explicit cost assumptions where live
measurements are still missing. Do not make proof of a speedup a prerequisite for
repairing these already identified weaknesses.

1. **Coherent initial placement and learned residency.** Apply the
   [global demand ranking and GPU order](#static-placement-order), retain learned
   residency across independent fresh-KV requests, and preserve the policy state
   needed for faithful comparison setup. Use a **defined demand clock: evaluated
   generation rows**, including rejected speculative rows. Express demand credit,
   aging horizons and admission-budget replenishment in those declared units before
   tuning their values. Capture prefill demand and service separately; prefill must
   not credit or age generation-residency learning or replenish its admission
   budgets. Preserve the clock and learning semantics across requests and setup
   checkpoints. The current five-row normalization is an initial parameterization,
   not an optimum. Keep global primary allocation and coordinated helper ownership.
2. **Whole-batch cooperative assignment and staging.** For each layer, plan over
   all rows actually being evaluated, group identical expert uses, and account for
   how many columns each expert serves. Choose both the staging candidates and
   the number of uploads by expected complete service time: CPU work avoided versus
   one weight upload plus grouped GPU work, available frames and interference with
   useful CPU/5090/3090 work. Use ready copies directly. An anchor's position at the
   front of the input is not a benefit score. A fixed upload count with candidates
   merely reordered by reuse is only a partial solution. Start with a **small,
   inspectable batch-cost model** whose inputs and assumptions explain each choice.
   Existing estimates and bounded grouped-service measurements can supply missing
   inputs. Building an elaborate predictor or a separate calibration campaign must
   not become another prerequisite. Keep distinct records uploaded,
   expert-column uses and bytes separate. Current-batch executed reuse can guide
   service assignment; only original-router demand credits future residency.
3. **Integrated retention, refill and transfer policy.** Preserve useful demanded
   copies in place, assess additional admissions against avoided service and actual
   incoming/outgoing obligations, and coordinate budgets with foreground staging,
   active readers and pending primary/helper exchanges. Choose a sensible bounded
   admission cadence and initial prices as part of this design. Do not rely on
   longer warming to hide an admission bottleneck, promote every upload, or count
   overlapping work as separate elapsed costs. Keep optional maintenance nonblocking.
4. **Explicit backing and complete integration.** Implement the already authorized
   full-primary DRAM-backing mode with its memory accounting and exclusive control,
   so the later isolated comparison is ready to run. Preserve the 16 GiB reserve,
   both GPU capacities, all experts and valid-copy ownership. Wire the chosen
   policies through actual ordinary/grouped execution and serving controls, using
   the selected packed arithmetic, routing and verifier. Controls for the old
   behavior remain diagnostic; active baseline settings use the new implementation
   after its affected checks pass.

Before moving to tuning, close the baseline gate with the bounded **full-model
integrated behavior check** through the actual ordinary and grouped production
callers. Establish valid ownership and reader protection; actual capacity and memory
accounting; learned-state continuity across fresh-KV requests and declared restores;
useful retention in existing frames; and bounded, correctly accounted transfers
with nonblocking optional maintenance. Use the existing check to establish these
properties together, reusing valid receipts rather than adding separate campaigns.
Predicted savings and focused CPU/component passes support implementation but do
not establish this gate. Any unverified required behavior keeps it open.
The later static/adaptive-helper and exclusive/backed-primary comparisons remain
isolated as specified; implementation of both modes does not preselect their
measured winner. Refine policy values and further implementation choices from
complete-engine results after this baseline exists.

| Area | Established starting point | Investigation and disposition required |
|---|---|---|
| Cooperative expert service | Packed CPU batching, native FP4 primary service and worker-based helper execution already run together. The mapped device-queue comparator was tested and parked. | Assess current preparation, grouping, submission/collection, CPU/primary/helper assignment and useful overlap. Improve the selected path where complete service costs justify it. Reuse valid kernel/contention evidence; do not describe already-integrated batching or FP4 as missing, or revive the tested queue under a different name. A materially different donor mechanism needs the investment case below. |
| DSpark and grouped target execution | Official DSpark and grouped expert-column service are integrated; the verifier has changed to sampled lenience. | Assess target work per committed token, rejected tails, expert-union growth, packed-record reuse and repeated launch/packing/handoff work. Decide whether changes to grouping, verification scheduling or current service implementation offer a worthwhile improvement, alongside width/confidence tuning. This work is distinct from the parked inter-device layer pipeline. |
| Demand transfer and preservation | Primary staging, helper refill and outgoing preservation operate in the exclusive bank. The original DeepSeek GPU-fetch mechanism and derivative Windows tests provide scoped evidence. | Assess exposed copy/preparation costs, required/optional transfer contention and overlap with useful service. Consider changes to transfer granularity, staging or device-driven fetch only where present-path evidence supports them. The slower Windows fetch ports do not establish a WSL result or rule out every adaptation; donor source alone does not justify a port. Demand fetch is distinct from the parked next-layer predictor. |
| Cache-aware expert selection | The current CUDA policy uses the approved additive ready-cache prior, with unchanged original-score mixing. A score-range-scaled donor and bounded-substitution implementation also exist as references. | Optimize the active prior and assess actual score gaps, original-demand misses and primary/helper service imbalance. A justified scaled or device-cost-aware selection policy remains a candidate within the required cache-aware design; none is already selected or proved superior. Preserve the routing/mixing and bounded quality contracts, and do not silently stack alternative policies. |
| Residency implementation | The [five-part adaptive design](#adaptive-residency) has an initial implementation and a scoped policy gain. | Complete its production coverage and cost-led optimization, including useful retention, global bank movement, ownership and coordinated helper/refill service. The dedicated design and trace sections own these requirements; exchange counts alone do not close them. |

For each material decision, record the actual caller, useful existing evidence,
remaining uncertainty and the next focused change/check or supported reason to
retain the implementation. Use the [investment policy](#choosing-worthwhile-work)
to choose the order and depth; use current counters/traces before constructing a
new executor. Other material costs exposed in the active generation path belong
to this assessment too. A failed prototype, missing measurement or generic claim
that a donor would be complicated does not establish that an important opportunity
is exhausted. Conversely, a settled unfavorable or inconclusive experiment does
not become mandatory again merely because an older proposal listed it.

The [parked mechanisms](#parked-work) and [held runtime bases](#base-holds) retain
their explicit dispositions. Retiring an old experiment queue or archiving a
proposal does not retire relevant mechanisms from consideration in Maya. Current
implementation decisions belong here and in the active work table, rather than
being silently labelled optional or future work in a subordinate document.

## Required parameter optimization

Complete the following optimization **after the
[reasonable baseline](#required-baseline-implementation) is implemented and its
independent operating behavior checked**. Use coherent provisional values during
implementation; do not let early value screens delay repair of known design flaws.
For **lenience** and the **cache-prior bonus**, compare the starting value with
meaningful alternatives in short matched actual-model screens, then select values
using complete generation speed and the bounded usability/quality evidence. The
starting value may remain selected, but leaving it unchanged without evaluation
is not completion of this work.

| Parameter family | Required investigation |
|---|---|
| DSpark lenience | Tune `spec_ell`, accounting for draft acceptance, rejection/residual work, output behavior and complete generation cost. Higher acceptance alone is not a win. Preserve the declared target sampling and actual-proposal probability contract. |
| Cache-aware expert selection | Tune the active `routing_prior_bonus` and assess material primary/helper preference effects. Preserve four valid experts, original-score mixture weights, routed scaling and shared-expert contribution. Verify which controls the selected policy actually uses; inactive substitution-only knobs do not count as cache-prior tuning. |
| DSpark work and scheduling | Assess verification/draft window width and confidence scheduling with the selected lenience rule; charge drafting, rejected work, expert-union growth, target service and committed tokens. Use a small informative set of compatible choices, not an automatic width grid or a new drafter-training requirement. |
| Primary residency | Assess aging/horizon, retention/admission margin, additional-refill budget and working/exchange capacity under the approved design. Price avoided service against incoming copies, necessary preservation, unavailable frames and interference. Resolve material uncertainty with observed ranges and focused trials. |
| Helper residency and transfers | Assess useful placement/retention, live-refill admission and scheduling, capacity allocation and coordination with primary ownership. Charge helper-link and CPU-memory contention, displaced work and useful completion; a larger hit count or a finished refill alone is insufficient. |
| Cooperative execution | After implementing batch-aware assignment and staging, tune material CPU/5090/3090 cost assumptions, upload amount, CPU worker/partition controls and transfer/compute overlap in the actual selected path. Evaluate whole-batch latency and emitted throughput, not a smaller staging count by itself. |

This is minimum coverage of the active system, not an instruction to sweep every
flag. Identify other material controls exposed by current evidence and record their
disposition in the [active work table](#active-maya-work). Outside the two explicitly
required value comparisons, source/counter evidence or a useful bound may justify
retaining a setting without a new run; explain why. A missing measurement, failed
prototype or build blocker is not evidence that an important opportunity is exhausted.

Hold unrelated settings fixed within a declared comparison. Screen coupled changes
as a group when their mechanism warrants it, then verify the retained combination
in the final engine; isolated gains are not additive. Reassess after useful results
under the [investment policy](#choosing-worthwhile-work). Preserve losses and bounded
quality failures. Keep high-or-higher effort and the specified target sampling;
neither is a speed-tuning knob. Existing [parked mechanisms](#parked-work) remain
parked unless their reopening rule is met.

## Choosing worthwhile work

Before substantial implementation or a full-model load, record a short decision
here: measured cost or missing behavior, proposed mechanism, plausible complete-engine
benefit/costs, cheapest informative check, and what its likely outcomes would change.
Source, counters, traces and rough estimates can guide investment; they are not
measured speedups or mathematical bounds.

For the [required initial baseline](#required-baseline-implementation), the known
design defects and the coherent proposed remedy are sufficient to proceed with
implementation. The cheapest informative check may be a focused correctness or
integration check. Do not require isolated performance proof for each repair or
use the optional-investment thresholds below to defer required baseline work.
Measure the combined implementation, then optimize and isolate contributors when
that answers a material question.

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
  problem has an adequate solution; record why that closes the relevant obligation.
  This does not close other unfinished workstreams.

### Runtime-base decision remains open

<a id="base-holds"></a>

<a id="base-choice-and-strategy-combinations"></a>

<a id="strategic-comparison-and-locked-gates"></a>

**User decision, 2026-11-03: Infernix and Strata are on hold as runtime bases.**
Do not advance their independent engine integration, base builds, model/GPU runs,
benchmarks or serving qualification until the user explicitly resumes the relevant
base. The three-base comparison is outside the active work and completion queue.

This hold does not exclude their ideas, code or evidence. Inspection, reuse and
adaptation for Maya remain allowed, including shared-component edits, builds and
focused checks needed by Maya, such as the Infernix MiniMax expert-service sources
that Maya already compiles. That shared work does not resume either base effort.
Existing read-only donor boundaries remain in force.

If the user resumes the comparison, the retained contract gives **Maya, Infernix
and Strata each an actual full-model comparison of the selected design**.
Match weights, workload, sampling, memory
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

**“Complete the plan” means the whole active Maya scope.** Only the user may remove
required work, waive a required outcome or redefine completion as a smaller milestone.
An agent may choose implementation details and park optional mechanisms under the
[investment policy](#choosing-worthwhile-work); it may not turn a difficult or
unfinished requirement into an optional follow-up. Held bases and explicitly parked
mechanisms remain governed by their existing dispositions.

Keep implemented functionality, scoped checks, policy improvements and final plan
completion distinct. The short-screen/reversed-order limits bound each experiment;
they are not a budget of work after which the entire plan is automatically done.
Elapsed time, a first positive comparison or passing a feature checklist is not a
completion criterion. Continue within existing authorization while required work
remains, using new checks only for new changes or unresolved decision-relevant questions.

The [active work table](#active-maya-work) must account for every required outcome:
what is implemented, which real execution/checks establish it, what performance was
measured, and what remains. Reuse valid earlier evidence; neither a missing receipt
nor an unchecked box requires indiscriminate reruns. Resolve the actual gap.
The [implementation-first sequence](#implementation-first-then-measurement-and-tuning)
changes the order of work, not these final evidence requirements. A reasonable
untuned baseline is an intermediate result, not plan completion or a proven optimum.
Final completion requires all of the following in the combined Maya engine:

| Evidence | Required account |
|---|---|
| Scope and design coverage | Every active Maya workstream is closed with evidence or an explicit user scope change. All five residency requirements map to actual production behavior. [Implementation choices](#implementation-optimization) and [parameter optimization](#required-parameter-optimization) have explicit results/dispositions, including measured lenience and cache-bonus value comparisons. Switches, object presence, aggregate exchange counts or tuning only existing flags are insufficient by themselves. |
| Actual system | The [selected engine](#selected-engine) through the real server, working launch defaults, effective settings, starting/final source/configuration, loader, target/drafter, cooperative dispatch, streaming/tools and affected state checks. |
| Residency optimization | The [approved design](#adaptive-residency) operates without mandatory placement profiles or manual reseeding. Account for useful retention, additional refills, outgoing preservation, CPU/helper interaction, real bank replacement and retained state across requests. Explain selected policies and remaining material cost uncertainty using traces, service observations and integrated measurements. The [replacement predictor](#timing-model-replacement) must demonstrate useful decision accuracy for supported held-out search, retaining state/ownership checks and actual integrated confirmation. The retired timing model's gates are superseded; neither exhaustive tuning nor a general-purpose simulator is required. |
| Performance outcome | The [starting-engine comparison](#comparison-with-the-usable-starting-engine) demonstrates a usable integrated gain, with all absolute/per-task results and material regressions reported under the [matched protocol](#short-experiments-with-resident-weights). A gain solely over a newly established lower control does not satisfy this outcome. No arbitrary speed target or unmeasured hardware ceiling is introduced. |
| Quality and resources | The bounded reasoning account and usable-task checks, plus actual memory/cache/transfer accounting, ownership and useful overlap in the final composition. A few passes do not establish broad quality or make the approximations “quality preserving.” |
| Handoff | Observed endpoint/source and process/resource state when requested, with remaining limitations and the precise scope of every claim. An earlier running-server handoff does not discharge later integration/performance obligations. |

Unfinished required work, unresolved comparison validity or an unaddressed material
regression means the plan is incomplete. If bounded investigation finds no useful
improvement, report that outcome and the unmet objective; do not relabel it successful
completion or run indefinitely to obtain a favorable number. Any proposed reduction
of the required outcome must be a visible user decision. Preserve adverse evidence.

Use “working integration” or “scoped improvement” for intermediate progress,
“parked/unmeasured” for optional deprioritized opportunities and “inconclusive” for
unresolved measured effects. Reserve “Maya plan complete” for the contract above.
Maya completion does not finish the held base comparison, whose scope and resumption
follow the [base-hold policy](#base-holds). Link raw receipts externally.

## Dedicated bases and work boundaries

<a id="dedicated-bases"></a>

| Repository | Branch | Role |
|---|---|---|
| `/home/mdwbr/project-maya` | `minimax-wsl/proposal-2` | First implementation milestone; central authority |
| `/home/mdwbr/Infernix-minimax` | `minimax-wsl/proposal-1` | [Runtime base on hold; code reuse allowed](#base-holds) |
| `/home/mdwbr/strata` | `minimax-wsl/proposal-3` | [Runtime base on hold; code reuse allowed](#base-holds) |

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

The user renewed local commit authorization on 2026-11-04: **commit as work
progresses**, including the preserved implementation. Push only when requested.
A commit neither requires a new benchmark nor turns incomplete evidence into delivery.

## Sub-agent delegation

Carried over from the [GLM delegation guidance](/home/mdwbr/Infernix-glm5-local/AGENTS.md#sub-agent-delegation). This section owns the policy for all three MiniMax bases.

**Primary-agent-only work, user instruction 2026-11-04:** never delegate offline
modeling, simulations, or offline optimization based on trace replays, simulations
or already collected data. The primary agent must do that work itself. Never
delegate benchmark-result evaluation: the primary agent must personally inspect
and evaluate comparison validity, results, limitations and resulting decisions.
These restrictions take precedence over the general delegation guidance below.

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
Owned processes/listener were absent at the handoff. Observe current state during
the mandatory restart checks rather than inferring it from this historical record.

The [evidence index](evidence.md) provides scoped results and source links.
The [stopped-round archive](history-2026-11-03-stopped-plan.md) and
[pre-consolidation snapshot](history-2026-11-03-pre-consolidation.md) preserve full
original wording, adverse outcomes and superseded queues. Consult them only for a
specific historical question; neither is routine restart reading or a work order.
Legacy anchors remain for incoming links and do not reactivate archived instructions.
