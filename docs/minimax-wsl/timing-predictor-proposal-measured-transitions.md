# Timing predictor replacement proposals

Six proposals for the open [replacement comparison](README.md#timing-model-replacement),
collected on 2026-11-06. The central MiniMax plan owns requirements, selection,
status and completion. This collection records designs; no replacement is selected,
implemented or qualified by their inclusion here.

Each section identifies its author and source. Supplied proposal text, reported
evidence, limitations and proposed budgets are retained; revisions to the primary
agent's own proposals are marked within their sections. Compiling the submissions does not
independently validate their measurements or causal interpretations. References
within earlier submissions to five candidates describe the original comparison;
the current collection has six.

The [current modeling scope](README.md#generation-only-modeling) is token
generation, with speculation off first; prefill is setup rather than a modeled
phase. Original external submissions below retain their authors' proposed scopes.
The collection supplies alternatives, not a requirement to combine all six.
The [shared-model diagnostic study](README.md#shared-model-diagnostic-study) now
precedes selection of the new first-stage design. Start with a shared model;
the earlier proposals' feature inventories and budgets are reference material,
not an instruction to implement them before measuring which effects matter.

1. [Terra: measured transitions at policy boundaries](#terra-proposal)
2. [Primary agent, proposal 1: joint boundary-response regression](#primary-agent-proposal-1)
3. [Primary agent, proposal 2: whole-configuration performance differences](#primary-agent-proposal-2)
4. [Fable: in-situ layer-window cost model](#fable-proposal)
5. [Opus: timing-free replay with a micro-randomized layer response](#opus-proposal)
6. [sol-6.1: empirical complete-layer transitions](#sol-61-proposal)

<a id="terra-proposal"></a>

## 1. Terra-generated proposal: measured transitions at policy boundaries

**Author: Terra.** Source: the original version of this file at commit
`8b74dae9db5a53a8cd4a76145b8d09ff42922fc7`. Its text is retained below with headings nested.

This is one proposal for the open replacement comparison. It has not been
selected, implemented, measured, or qualified. The [central MiniMax plan](README.md#timing-model-replacement)
owns requirements, decisions, status, and completion.

### Motivation

Keep Maya's useful state replay and native policy decisions, but replace the
factorized reconstruction of operation times with a model of complete intervals
between existing policy or work-completion checks. Learn each interval's elapsed
time and the availability outcomes observed at its next boundary **jointly** from
real executions.

The [prospective intervention test](evidence.md#prospective-fixed-work-intervention-result)
illustrates why this is worth considering. The retired full-work comparator
predicted 12.062 seconds for the control versus a measured mean of 6.693 seconds,
an 80.2% overestimate. Some admission and transfer counts were already close,
but none of the three candidate decisions was resolved. That result is separate
from the earlier foreground refit's 5.57% miss; neither establishes the cause of
the full-work error.

### Proposed model

1. **Retain state and decisions.** Start with the captured placement, physical
   frame roles, original-demand scores, credits, and pending transactions. Use
   the shared native policy entry points for demand, staging, and admission.
   Original-router selections alone update demand; executed selections,
   including rejected speculative rows, determine the work serviced. Preserve
   ownership, capacity, reader, writer, and transaction checks. The existing
   [observation-fed reproduction](../../tools/mm3/cooperative_replay/capture.py#L122)
   remains a separate state and decision check, not a timing prediction.
2. **Capture transitions.** At existing points where Maya makes a relevant
   decision or checks outstanding work, record the current batch and service
   shapes, CPU/5090/3090 assignments, staging, pending transfer phases and ages,
   and action taken. Record the host elapsed time to the next relevant boundary
   and, together with it, the completion statuses actually checked there.
   Unchecked work remains unknown; do not fabricate its completion time. Preserve
   forwarding, prefill/generation, and request boundaries, including any different
   boundary types needed to cover complete scored work. The current
   [route and service instrumentation](../../src/mm3/mm3_native_trace.inc#L245)
   is a starting point, not proof that all of these labels already exist.
3. **Fit a conditional empirical model.** Begin with weighted nearest neighbors
   using NumPy, which is present in the project environment. Match the native
   boundary and compatible transaction/slot topology, then choose similar
   examples by the work shape, pending ages, current state, and action. Sample
   a whole recorded outcome vector: interval elapsed time and the checked
   completion results together. Keep the mapping from sampled transaction roles
   to the candidate's own transaction identities explicit. Fit feature scales,
   neighbor settings, and support thresholds on training histories only and
   freeze them before evaluation. This is a conditional empirical model; the
   related idea of using similar examples to estimate an outcome distribution
   also appears in [quantile regression forests](https://www.jmlr.org/papers/v7/meinshausen06a.html),
   but that work does not establish accuracy or coverage for Maya.
4. **Advance the candidate's own state.** Apply the sampled outcomes only to
   transactions eligible to have been checked, enforce legal phase changes and
   ownership, and call the policy again at the next boundary. Use the candidate's
   newly generated state and history for the next prediction. The replay sees
   captured fixed work in order, while the policy sees only work and state
   available at its current decision. Never import the control run's future
   readiness or clock trajectory. Reject invalid transitions and mark states
   without adequate compatible examples unsupported rather than silently
   inventing a ranking.

The interval outcome already contains the observed combined effects of CPU work,
transfers, GPU execution, queueing, and overlap in its training regime. For
example, a backing candidate changes preservation transfers and memory pressure;
the next predicted transition and later residency decisions arise from that
candidate's own state. The model does not add independently priced CPU, copy,
and GPU durations into a synthetic wall time. Whether its examples generalize
to a candidate's altered state remains an empirical question.

Represent trajectory uncertainty by resampling whole chronological histories
and retaining recent history or a shared run-level noise regime when generating
transitions. Preserve pairing when estimating candidate-minus-control errors.
Common random numbers alone do not establish the true correlation: check the
resulting difference estimates against independent paired hardware observations.
Report whole-interval error and intermediate state/transition diagnostics
separately; do not add nested residuals into a larger error allowance.

### Proposed first evaluation

This is an illustrative finite budget to declare before any new measurement,
not an instruction to run the experiment or a claim that the sample is adequate:

- Use six independently conditioned chronological histories, each covering
  coding and prose work. Use four histories for fitting and two untouched
  histories for evaluation. Each policy conditions from the same declared
  neutral origin under its own actions; do not seed candidates from an
  incumbent's already-learned evaluation state. Preserve state across each
  history's scored work.
- Compare the incumbent, primary backing, reduced admission budget, and reduced
  staging, in forward and reversed order for each history: **48 fixed-work
  arms**. The fixed workload retains all captured evaluated rows, including
  rejected speculative work. Record per-task as well as combined results.
- Allow **eight additional capture-on/off fixed-work arms** to assess whether
  the proposed compact instrumentation changes timing or policy behavior. Use
  existing joins and checks, buffer records, and write outside timed work; add
  no synchronization, GPU queries, optional-transfer waits, or per-event disk
  flushing. If capture changes behavior materially, reduce it or leave the
  resulting model unqualified; do not assume a constant overhead subtraction
  repairs altered scheduling.
- Allow up to **24 policy-specific conditioning sequences**, plus up to
  **12 source request captures** to construct the six coding/prose histories.
  Reuse existing captures or conditioning only if they satisfy the same
  independent-history, source, configuration, and holdout requirements. These
  setup items and later ordinary-request confirmation are separate from the
  56 timed fixture arms above.

Before collecting data, freeze the workload identities, control settings,
conditioning, feature and coverage rules, candidate set, and a practical
equivalence margin of **3% of paired control elapsed time** for this proposed
screen. Fit and calibrate only on the four training histories, with chronological
blocks kept intact. Freeze predictions before observing the two evaluation
histories. Check actual whole-work candidate-minus-control timing with detailed
capture off; use appropriately matched instrumented companions for transition
diagnostics, with their overhead and coverage explicit.

At minimum, a useful model must correctly predict the direction and magnitude
of at least one independently resolved, consequential non-tie intervention. A
practical tie is supported only when both the measurement and prediction support
the declared margin. Broad intervals, misleading confident decisions, missing
coverage, and order-dependent sign changes remain inconclusive or adverse, as
appropriate; an all-tie predictor does not qualify. Evaluate paired differences
at the history/measurement-block level rather than treating thousands of
correlated layer intervals as independent observations. Two held-out histories
may be insufficient for reliable uncertainty estimates: in that case the
screen does not qualify the model, regardless of the number of recorded layers.

### Limits and follow-through

- Similar intervals may be too sparse, especially for changed transaction
  combinations, or may hide relevant queue/driver state. Unsupported regions
  and compounding prediction error could make this approach unhelpful.
- Fixed-work replay estimates scheduling effects conditional on captured expert
  rows. Changed residency can change cache-aware routing, generated text, and
  speculative acceptance. It cannot establish ordinary serving speed or quality.
- A useful supported prediction is still followed by the central plan's short,
  matched actual-engine confirmation. No runtime parameter is selected by this
  proposal or by the historical failed timing predictions.

<a id="primary-agent-proposal-1"></a>
<a id="primary-agent-proposal-1-review"></a>

## 2. Primary-agent proposal 1: joint boundary-response regression

**Author: the primary agent in this conversation. Revised design, 2026-11-06.**
This develops the cooperative-batch/replay proposal into a measurement contract,
fitted model and finite decision test. It replaces the original outline and its
review note, which remain in Git history. It is a design for comparison; no part
of the new predictor has been implemented or qualified.

**Current disposition:** this remains a candidate reference. The user's
[measurement-first direction](README.md#shared-model-diagnostic-study) requires
the new shared first-stage design to follow the bounded diagnostic study. Do not
start this candidate's readiness heads or large pilot as the immediate work.

The [accepted development order](README.md#generation-only-modeling) starts with
ordinary `T=1` generation and adds speculative generation only after that model
passes its decision tests. Prefill is outside both stages. This narrows the first
implementation without relaxing its measurement or validation requirements.

### Prediction contract

Retain the deterministic policy/ownership replay. Replace its operation scheduler
and service prices with a conditional distribution for the next **observable
boundary transition**:

```text
predict(start_state, scheduled_work, actions, boundary_kind)
    -> elapsed_host_ms, ordered_readiness_answers, support_status
```

`elapsed_host_ms` is the complete wall interval between two specified existing
execution boundaries. `ordered_readiness_answers` describes only checks the
engine actually makes at the destination: ready, not ready, or a helper phase.
A check skipped by production is unobserved. The model does not predict separate
CPU, primary or helper durations, kernel time, or hidden transfer completion
instants. Completion of foreground work guaranteed by an existing stream join
is a deterministic fact, not a learned probability.

The target quantity is the distribution of complete fixed-work generation time
and, principally, its **candidate-minus-control difference**. Every scored
interval, including prefix, tail and time between expert service and the next
join, belongs to that account. Here prefix and tail mean work inside each decode
forward, including the dense first layers and output head, not prompt prefill.
The initial domain uses actual ordinary-generation traces, with one token row
per decode forward and speculation disabled. The fixed-work target runner omits
sampling; live ordinary-generation confirmation includes it. No target-only
measurement is presented as complete request latency.

Prefill executes as setup and supplies an explicit generation-start snapshot.
Its time and multi-row observations are excluded from fitting and scoring.
Bind each modeled interval to its declared starting state; include cache content,
learning, pending transactions and context length. Where initialization or restore
has restrictions, declare the supported starting conditions rather than inventing
missing state or borrowing a control run's future availability.

### Boundaries and native observations

The current source makes a route-to-service window insufficient by itself:
`window_post` submits the primary work and returns after host reduction, while
the primary stream can still be working. `helper_live_progress` can also act in
`window_pre`, before the next route join. Therefore the boundary sequence follows
the actual host program at these sites:

| Boundary group | Observable work and state |
|---|---|
| Forward entry and its existing publication/progress checks | Starting logical state, current pending records and checks actually performed |
| Existing helper-progress group before layer preparation | Helper phase, protected HOST readers, and any resulting legal phase transition |
| Route/readback join and primary publication/retention group | Foreground readers retired by the join, actual primary query results, then native retention and original-demand updates in their existing order |
| Cooperative assignment/submission block | Exact planned CPU/5090/3090 groups, staging and newly issued exchanges; these are action descriptors, not measured service prices |
| Forward completion and its publication/admission groups | Tail work, actual readiness observations, helper/primary actions and outstanding leases |

These are insertion sites, not permission to merge intervening decisions.
Record the native order of checks and actions. If a check changes an action
inside a proposed interval, end the interval at that existing decision before
forming the next descriptor. In particular, do not move helper withdrawal or
primary publication to a convenient later layer join. A host-side assignment
block can be reproduced from known demand and state before its timing is priced;
its chosen actions must not depend on an unmodeled readiness result inside it.

Use a preallocated array of fixed-size records. Store monotonic host timestamps,
boundary/forward/layer IDs, work descriptors, transaction issue IDs, existing
poll answers and legal transition IDs. Copy results from the production query
or helper poll that has already happened. Add no CUDA events, queries, waits,
profiling ranges, string formatting, allocation or disk output during scored
execution. Snapshot complete ownership and learning state outside the scored
interval, and serialize after it. Buffer overflow invalidates the capture.

Use the fixed runner's own start/end timestamps for the outer accounting.
Nonoverlapping intervals must telescope to its reported target/generation time
apart from clock representation rounding. A missing interval is a capture bug.
CUDA event spans and `trace_service` subspans may remain diagnostic records;
neither enters the response labels or the predictor's feature vector.

Source anchors: [native layer path](../../src/mm3/mm3_model.cu),
[primary publication](../../src/mm3/mm3_native_adaptive.inc),
[helper phase transitions](../../src/mm3/mm3_native_helper_live.inc),
[existing trace code](../../src/mm3/mm3_native_trace.inc), and
[fixed workload format](../../src/mm3/mm3_replay_workload.hpp).

### Inputs available to the predictor

Build a bounded feature vector from the candidate's current state and the work
scheduled before the next boundary:

- Boundary type, context position and layer class. Decode row count is fixed at
  one in the first stage. Multi-row generation features belong to the later
  speculative extension; prefill is not a fitted regime.
- CPU and primary columns, record groups and actual eight-column chunks; helper
  job count and occupied-width histogram. Lane assignments describe work without
  supplying lane-time targets. Retain descriptors for submitted foreground work
  not yet guaranteed complete by a join, rather than treating host return as its
  completion.
- Ready/leased/reserved frame counts, already available staged records, and
  separate counts of retained uploads versus newly requested copies.
- Issued bytes by direction and queue, outstanding transaction types and issue
  order, phase ages and time since the last failed check. Ages advance on the
  candidate's clock. A pending record without a known starting age is unsupported.
- Backing allocation/validity, displaced service ownership and a bounded recent
  history of work/transfer descriptors. The deterministic replay retains exact
  identities, rosters and demand; the statistical features need not duplicate
  every expert ID.

Define queue/phase age consistently as elapsed time since entry to the existing
host decision/submission group that issued that work. This is an observable
coarsened age, not an assertion about the DMA start instant. Native records and
candidate replay use the same reference; progress within the submission group
is part of the joint response. Split a group if an intervening readiness check
can alter its subsequent actions.

Use no future control publications, observed service times, policy-estimated
savings or post-execution counter totals as inputs. Work descriptors must be
computable by running the shared policy on the starting state. Layer work not
yet revealed to the policy stays unavailable to it, even when the offline
harness possesses the complete fixed workload. Prefix intervals, for example,
use scheduled rows/context rather than the next layer's unrevealed expert IDs.

### Fitted form and joint readiness

Fit a regularized regression for the log of the whole interval duration:

```text
log(tau_k) = beta[kind] + beta_work . phi(x_k) + b_block + residual_k
```

`phi` has at most 40 declared terms. Start with work/queue counts, log row/context
terms and a one-forward summary of recent work. Compare two forms only: main
effects, and the same features plus four interactions: CPU chunks × staging
bytes, helper jobs × pending helper-upload bytes, primary chunks × CPU chunks,
and pending H2D × D2H bytes. Scale features from training data only. Fit with a
robust loss and ridge penalty; positive durations follow from the log link.
The feature list, transformations and penalty grid `{0.1, 1, 10}` are frozen
before fitting. This is a small response regression, not a table over all lane
combinations.

For the first `T=1` stage, omit constant row-count/width terms and duplicate
features such as expert columns and one-row group counts. Do not fit coefficients
for unobserved multi-row behavior. The general feature inventory above defines
possible extension inputs, not terms that must all be fitted in the first model.

Factor the joint response as duration followed by the ordered observed checks:

```text
P(tau, answers | x) = P(tau | x)
                      * product_j P(answer_j | x, tau, earlier_answers)
```

Use one regularized logistic head for ready/not-ready checks, with transaction
kind/phase, bytes, age, queue position and the duration residual as features;
use a small categorical head where the helper exposes several legal phases.
Conditioning readiness on the sampled duration and its residual allows a slow
interval to be associated with different exchange progress. It does not assume
that longer elapsed time always means faster transfers. Fit these heads using
out-of-fold duration predictions, not in-sample fitted residuals.

Apply deterministic constraints during fitting and rollout. A completed event
stays completed; helper phases cannot regress or skip an unissued phase; joined
foreground readers are finished. Within one ordered stream, observing a later
event complete implies its predecessors have physically completed by that check.
It does **not** retroactively change an earlier not-ready answer or publish an
entry whose production check was skipped. Process answers in native poll order,
since they are not simultaneous observations. Invalid answers have zero
probability; censored/unqueried outcomes are not negative labels. The state
machine, rather than the regression, decides whether readiness permits ownership
publication after reader and protected-layer checks.

This form is deliberately different from learning a separate lane timer and
combining those timers. Whole elapsed intervals are the time labels. The
readiness heads describe observable state transitions, not independently priced
physical services. It remains in the same broad empirical-transition family as
Terra and sol-6.1; the concrete choice here is a small parametric joint response
instead of nearest-neighbor outcome vectors or tree leaves. Shared family
membership is acceptable in this design comparison.

### Candidate rollout and retained implementation

Extract ownership, leases, allocation accounting and shared policy calls from
`Machine` into a timing-independent state object. `Machine` currently calls
`op`, `scheduler.done` and `scheduler.until` throughout its decisions; swapping
one timing lookup is insufficient. Keep the old scheduler frozen as historical
code, and add a separate boundary driver with this loop:

1. Apply only the preceding transition's observations at the native check sites.
2. Enforce leases and legal publication; update original-router demand and call
   the shared staging/admission policies in production order.
3. Construct the next interval's planned-work/action descriptor, obtain a joint
   response, advance that candidate's clock and pending ages, and repeat.
4. Reject illegal ownership/capacity transitions and propagate unsupported state
   to the complete candidate prediction rather than silently pricing it.

Start each arm at the same declared neutral logical origin, configure its own
backing/capacities, and retain policy-specific chronological conditioning during
ordinary generation. Prompt processing is performed by the native setup, and
each modeled generation interval begins from a bound generation-start snapshot;
the predictor does not roll through or price prefill. Preserve and account for
learned state and pending work across setup/request boundaries. Never clone the
incumbent's learned evaluation state into a changed learning policy. Only
original-router rows update demand. The later speculative extension must retain
all evaluated rows, including rejected work, in its own distinct evidence set.

The [capture/reproduction path](../../tools/mm3/cooperative_replay/capture.py),
[native policy bridge](../../tools/mm3/cooperative_replay/policy.py), snapshots,
fixed-work runner, ownership assertions and paired-difference classification
remain. Observation-fed reproduction must still match decisions and bytes
exactly. Free-running timing predictions are evaluated separately and need not
produce an identical stochastic publication history on every repetition.

### Drift and uncertainty

Use complete randomized measurement blocks containing every arm, restoring the
same neutral origin and running policy-specific conditioning for every arm.
Balance execution positions across repetitions. The block intercept `b_block`
is shared by all arms in a training block; do not fit a separate intercept for
each policy or scored request. It represents slow environmental variation,
while backing and other policy effects remain in the work/state features.

Estimate residual dependence from cross-fitted complete forward/request
sequences. Resample chronological sequences and whole paired measurement blocks,
including all their arms and shared control, when refitting and producing delta
uncertainty. Do not treat thousands of layer records as independent trials, or
force identical descriptors to have zero uncertainty. Within a simulated run,
draw a correlated residual sequence and feed each sampled duration residual to
its readiness head. Candidate and control share parameter uncertainty and the
fitted block component; their remaining noise is not assumed to cancel.

Calibrate the resulting delta-error intervals against leave-one-history-out
paired errors on training data. At prospective evaluation, integrate over the
training drift distribution: do not infer drift from the held-out scored time,
anchor predictions to same-run prefill, or import the control's future clock.
Report raw paired effects and order effects. If a prespecified order analysis
changes a resolved classification, report it unresolved. Nested interval,
readiness and request errors remain separate diagnostics.

Support is checked by boundary/phase category, feature ranges and distance to
training descriptors. Use standardized-feature Euclidean distance within a
compatible category. Set the distance cutoff from the training-only 95th
percentile of each complete cross-fitted sequence's maximum nearest distance,
rather than applying a 95% pointwise cutoff to thousands of intervals. A relevant
phase without usable observations, an unseen operating regime, or an out-of-support state prevents a
qualified complete rollout; apparent confidence from regularization is not
coverage. Report where and how often coverage fails, including failures caused
by the model's own generated states.

### Finite pilot and selection rule

Freeze prompt IDs, seeds, complete fixed workloads, neutral snapshots, binary,
operating settings, features, statistical rules and order schedule before new
measurement. Hold CPU participants, arithmetic, worker-helper dispatch, query-off
submission and cache-aware routing fixed, with speculation disabled in every
first-stage arm. Use `strata_tune.spec=1`, `spec_policy="existing"`,
`verify_segment=0` and `stage_overlap=0`; verify the actual decode path. There are six new
chronological two-request histories: three for fitting and three reserved for
evaluation. Each history contains a conditioning request followed by a scored
request. Training topic transitions are coding to prose, extraction to tool
planning, and reasoning to coding; evaluation transitions use different prompts
for coding to prose, prose to extraction, and tool planning to reasoning.
Exact text and generation caps are part of the frozen manifest. Source
capture supplies exogenous work, never evaluation timing/readiness labels for
fitting. Preserve complete ordinary-generation work and the native setup's
starting-state provenance. Exclude prefill durations from model fitting and
the scored objective; prefill execution still consumes experiment resources.

Use these arm definitions; all unspecified settings equal the incumbent:

| Arm | Primary backing | Primary budget | Staging cap | Purpose |
|---|---|---:|---:|---|
| C | Off | 8 | 4 | Incumbent control |
| B | On | 8 | 4 | Preservation/transfer intervention |
| A | Off | 2 | 4 | Reduced admissions |
| S | Off | 8 | 2 | Reduced staging |
| P | Off | 0 | 0 | Large-change diagnostic; its sign is not assumed |
| Q | Off | 4 | 3 | Combined setting withheld from fitting |

A **sequence execution** means restoring the origin, running the conditioning
generation under the arm, then its scored generation, with real prefill/setup
and generation-start snapshots explicitly accounted for. Both generation
workloads are charged to the budget; a sequence uses two native fixed-work
generation invocations, each with at most one native prefill/setup pass. Setup
time is charged to the experiment budget even though it is outside the modeled
duration. The frozen protocol must account for both calls if setup and generation
use separate native invocations.
Backing allocation/setup is also reported, even though warmed scored generation
is the performance objective.

| Stage | Maximum work | Use |
|---|---:|---|
| Source acquisition | 12 ordinary requests | Six complete two-request histories; collect only once |
| Compact-capture overhead | 16 sequence executions | C and B, four on/off pairs each, balanced orders, using training work |
| State companions | 6 sequence executions | One detailed companion per arm on training work; state/byte reproduction only, never timing fitting, including Q |
| Fitting | 30 sequence executions | Three training histories × C/B/A/S/P × two balanced repetitions, compact capture on |
| Frozen evaluation | 54 sequence executions | Three untouched histories × all six arms × three balanced repetitions, detailed and compact capture off |
| Integrated confirmation | At most 12 ordinary requests | Only after a useful supported candidate is selected; matched live control/candidate requests in both orders |

This proposal's ceiling is **106 sequence executions: 212 native fixed-work
generation workloads and at most 212 prefill/setup passes**, plus **24 ordinary
requests** including source acquisition and confirmation and their own prefill.
If setup and generation are separate calls, that allows at most 424 native
fixture invocations. It is a maximum budget for this candidate, not a required
queue; the selected design must justify its actual allocation. Also cap warmed
measurement time, including conditioning and prefill/setup, at 90 minutes;
record model load/build time separately.
Stop at the count or time cap without extending the campaign to obtain a pass.
Reusing a qualifying existing source consumes no new acquisition request, but
does not turn previously inspected evaluation labels into untouched evidence.
The Q state companion's timing and descriptors are excluded from fitting,
coverage calibration and tuning; it checks the deterministic state path only.

For the overhead stage, compare paired elapsed time and existing end-of-request
work/transfer/ownership summaries. The proposed material-overhead threshold is
**1% of control time**, one third of the decision margin. A resolved overhead
above that threshold, or a systematic policy-state change beyond off/off repeat
variation, stops this capture design. An inconclusive overhead estimate is
reported as inconclusive, never as zero. Do not subtract an average capture cost.
All qualification effects are subsequently tested against capture-off hardware,
so training on instrumented runs cannot qualify itself. Detailed companions are
for state checks only.

Select the regression form and penalty by leave-one-training-history-out
candidate-minus-control error over B/A/S/P. Use complete history blocks for the
selection uncertainty and choose the smaller form within one standard error of
the best. No held-out Q timing or evaluation history participates. Freeze the
chosen artifact, coverage rules, millisecond prediction intervals and classification rule
before observing the 54 evaluation executions.

For each held-out history, the practical-equivalence margin is **3% of its
measured capture-off control generation time**. Predict milliseconds first;
the frozen rule converts control time to the margin without refitting. Classes
are computed once that control scale is available; its observed time never
conditions a candidate's rollout. Use 95%
paired intervals and the existing `delta_classification`: a gain lies wholly
below `-margin`, a loss wholly above `+margin`, and a tie wholly inside the band.
An interval crossing zero is not sufficient evidence of a tie. The three
repetitions per history and nine complete evaluation blocks may still be
insufficient; the result can remain unqualified at the cap.

The predictor passes this pilot only if:

1. It correctly resolves at least one consequential non-tie among B/A/S/Q,
   with the prediction and hardware supporting the same direction. P alone
   cannot establish usefulness for the intended optimization decisions.
2. For that decision, the paired interval for prediction error is wholly within
   the same 3% margin; matching a sign with a badly wrong magnitude is insufficient.
3. Over all prespecified supported comparisons, mean absolute delta error is at
   least 20% lower than predicting zero change, and the paired block analysis's
   95% interval for `MAE_zero - MAE_model` lies above zero. Report every arm, including unsupported ones;
   at minimum B/A/S/Q must have complete supported rollouts to pass this pilot.
4. There is no confident gain/loss reversal on a hardware-resolved comparison,
   no illegal state transition and no order-sensitive result presented as resolved.

Scope uncertainty at the complete-history and measurement-block levels, keeping
all treatment differences sharing a control together. Report within-history
results separately from pooled results; three held-out histories cannot justify
unrestricted task generalization. A failure is preserved at the finite budget.
It does not start another term-by-term calibration campaign.

### Deliverables, use and limits

Implementation has four bounded pieces: the compact boundary recorder; extraction
of timing-independent state/decision logic; the two response heads plus support
checks and trajectory sampler; and a frozen-protocol evaluator. The principal
engineering risk is extracting the existing scheduler-dependent ownership logic
without changing its decisions. The principal statistical risk is omitted driver
or queue state causing trajectory and policy-difference error. State reproduction
checks the former; untouched capture-off intervention results decide the latter.

A qualified version can screen supported nearby residency choices and compare
keep/swap alternatives over the same recent original-demand patterns, charging
victim displacement and pending work through the subsequent state trajectory.
Already issued transfers remain irrevocable. Keeping the allocation is always
an alternative. Sharing response prices with an admission policy is not validation.
The initial implementation is an offline evaluator; a cheap compiled admission
approximation and its host overhead would be a separate, measured downstream
step, rather than running Python Monte Carlo in the inference loop.

Changed helper capacity, CPU participants, arithmetic or submission mode require
new operating coverage. Changed routing, tokens or speculative acceptance remain
outside fixed-work prediction. After ordinary-generation validation, grouped
speculative generation needs its own calibration, decision checks and integrated
confirmation. Prefill remains outside the model. Ordinary high-effort requests
establish the first-stage speed and usability result; the later selected serving
configuration needs its own integrated confirmation. This design provides a concrete model and an experiment
that can reject it; it does not claim those experiments will pass.

<a id="primary-agent-proposal-2"></a>

## 3. Primary-agent proposal 2: whole-configuration performance differences

**Author: the primary agent in this conversation.** Source: the second of the
two proposals presented to the user on 2026-11-06. The closing qualification
applies to both primary-agent proposals.

Build a surrogate that predicts **candidate-minus-control elapsed time**, without reconstructing the execution timeline.

- **Inputs:** policy settings, starting-state summaries and recorded-workload features. Only information available before executing the candidate is allowed.
- **Outputs:** expected performance difference and empirically checked uncertainty.
- **Model:** a regression ensemble trained on short, paired hardware trials. This follows the general approach of [model-based algorithm configuration](https://www.cs.ubc.ca/~hoos/Publ/HutEtAl11.pdf).
- **Search:** evaluate many candidate settings cheaply through the surrogate; choose subsequent hardware trials where they could resolve an important decision or test a promising candidate.
- **Role of state replay:** validate the actual trials, preserve comparable starting conditions and extract demand/state information. It would not manufacture hypothetical publication times.
- **Validation:** hold out complete workload histories and configurations, checking prospective rankings and uncertainty. Training labels are measured elapsed-time differences, never the admission policy’s claimed savings.

**Main advantage:** it directly targets the optimization decision and avoids modeling driver behavior separately.

**Main risk:** it may generalize poorly beyond the sampled configuration region. It also would not, by itself, explain or value individual admissions; that policy still needs a justified design and validation.

Both proposals retain original-demand-only learning and resource correctness. Fixed-work trials can support residency and scheduling optimization; routing or speculative changes that alter generated work still require ordinary-inference confirmation.

<a id="fable-proposal"></a>

## 4. Fable-generated proposal: in-situ layer-window cost model

**Author: Fable.** Source: `docs/minimax-wsl/timing-replacement-proposal-fable.md`
on branch `minimax-wsl/timing-proposal-fable`, commit
`e861ce69c35907e828d536ff0aac9273323ad824`. The full committed proposal is
retained below with headings nested.

**Status: proposal for the five-way comparison opened by the
[replacement decision](README.md#timing-model-replacement), 2026-11-06.** Nothing
here is implemented, measured or selected. The central plan owns selection;
this document is one of the five candidates it compares.

### Summary

Replace per-operation timing reconstruction with a **layer-window cost model**:
the unit that is priced is the cooperative layer window that the engine already
delimits (`layer_join` boundary → route readback → host service end), and the
price of a window is fitted from the engine's own boundary telemetry **in the
operating regime**, not from isolated probes. The retained state replay supplies
each candidate policy's sequence of windows and their work descriptors; the
window model turns those descriptors into time; candidate-minus-control is
formed window by window on common fixed work, which cancels drift and model
error on every window the two policies leave identical.

What stays: the strict capture contract and stage-1 decision/state reproduction,
the shared native policy entry points, the `Machine` ownership/frame/exchange
logic, original-demand accounting, the fixed-work harness and setup-snapshot
binding, and the repaired acceptance logic (`delta_classification`,
`intervention_validation`). What goes: the discrete-event `Scheduler`, the
per-operation `ServiceTable`, pair contention factors, and the stage-2
per-operation timing categories. Link transfer rates from the existing transfer
probe are the only isolated measurement retained, and only to turn issued bytes
into a window count for availability.

### Why the retired approach missed, and what this changes in principle

The factorized model summed isolated per-operation service through a resource
scheduler. Its receipts show where that breaks:

- The primary lane's CUDA event envelopes were 7,155.812 ms with only 215.264 ms
  of recorded kernels; the 6,761.962 ms of gaps are driver/launch-batching and
  host-submission behaviour that only exists in context. No isolated probe
  measures it, and the plan's recorded
  [launch-group observation](README.md#remaining-work-and-retained-evidence)
  (launches reaching the 5090 in groups of 11 under Nsight) says the same.
- Table-to-live discrepancies of CPU -34%, helper -71% and primary +139% show
  that isolated service does not transfer to the live window even before
  contention is modeled.
- The full-work comparator missed by +80.2%; the generation-only refit by
  +5.57% with +13.95% CPU and +42.57% helper spans. Each repair added a term
  (working set, submission context, width groups, pair slowdowns) and each term
  needed its own coverage campaign.

The replacement does not try to explain the gap mechanism. It measures the
window that contains it, in the regime where decisions will run, and asks only
the question the decisions need: **how does window time move when the window's
work descriptor moves?** That is a regression on in-context observations, with
coverage declared on the descriptor, instead of a reconstruction from parts.

### The model

#### Unit: the cooperative layer window

For each MoE layer of each target batch the engine already records, under the
residency trace, a `routes` record stamped at `trace_routes` and a `service`
record at `trace_service` carrying `host_service_ms` and the `Stats` deltas for
that window ([trace source](../../src/mm3/mm3_native_trace.inc),
[call sites](../../src/mm3/mm3_model.cu) at `layer_join`, `trace_routes`,
`trace_service`). The window spans host planning, demand staging submission,
helper preparation/dispatch, CPU batch, primary submission and reduction, and
ends where the production caller returns. Everything between windows (dense,
attention, router, shared experts, readback, draft, sampling, target prefix and
tail) is **non-window time** and is not modeled: it is policy-invariant on fixed
work except through contention, which the protocol below checks directly.

#### Descriptor

All fields already exist as `Stats` counters or are produced by the replay's
`serve_layer`:

| Group | Fields | Source in engine / replay |
|---|---|---|
| Batch | rows `T`, generation flag, regime (ordinary, fixed-work) | boundary events / capture |
| CPU lane | columns, chunks (≤8 columns per record), records | `coop_cpu_columns`, `coop_cpu_chunks`, `cpu_experts` / `structure.chunks` |
| Primary lane | columns, groups, chunks; demand uploads this window and staged columns | `coop_primary_*`, `staged`, `coop_staging_columns` / `selected`, `uses` |
| Helper lane | columns, jobs, launch width histogram | `coop_helper_columns`, `helper_experts` / `structure.helper_launches` |
| Optional transfers | H2D/D2H bytes issued in this window; H2D/D2H bytes **in flight** during it; helper exchange phase in flight | `adaptive_h2d_bytes`, `adaptive_d2h_bytes`, pending `adaptive`/`helper_exchange` state / `Machine.exchanges` |

No kernel time, no event span and no observed per-operation duration enters
the descriptor. Row-frequency scores and policy prices are excluded: the model
prices work, not the policy's opinion of it.

#### Form

Primary form, chosen for mechanism (the engine's own planner takes a maximum
over overlapping lanes):

```
T_window = β0(T, generation)
         + max( a1·cpu_chunks + a2·cpu_columns,
                b1·primary_chunks + b2·uploads + b3·primary_groups,
                h1·helper_jobs + h2·helper_columns + Σ_w h3,w·width_w )
         + κ1·inflight_h2d_bytes + κ2·inflight_d2h_bytes + κ3·[helper exchange in flight]
```

Fitted by the standard partition method for a maximum of affine functions:
assign each window to its currently largest lane, refit each lane by least
squares, iterate to a fixed point; a smooth-max fit is the cross-check. The
predeclared comparator is the same descriptor with no maximum (additive). The
launch-group observation suggests the no-flush WSL regime may be closer to
additive, because primary completion trails the CPU batch rather than running
beside it; the data decide, by the selection rule below, not by assumption. Both forms have
about fifteen coefficients, each with units of milliseconds per chunk, column,
upload, job or byte in flight, which is what makes them inspectable and what
the admission repair can consume.

Prefill windows are fitted as a separate regime. Fixed-work windows are fitted
separately from ordinary-request windows, because the GPU queue state between
forwards differs when drafting and sampling are absent; which fit a prediction
uses is part of its declared domain.

#### What the existing capture already shows

A single pass over the first extended capture
(`state-capture-v2/…/req-c8b7298157774728a4316cc7ea1c7589.jsonl`, detailed mode)
gives the scale of the problem and of the signal. These numbers include
detailed-capture formatting inside the windows, come from one request with
five-row generation batches only, and are **not a fit for use**; they are the
reason to expect the approach to work, not evidence that it does.

| Observation | Value |
|---|---:|
| Layer windows in the request | 1,710 (912 generation, 798 prefill) |
| Generation window time, median / mean | 3.18 ms / 3.36 ms |
| Generation window time, p10 / p90 | 2.34 ms / 4.73 ms |
| Additive fit on nine counters, R² / residual RMSE | 0.798 / 0.42 ms |
| CPU columns + chunks alone, R² | 0.767 |
| Fitted marginal: one more CPU chunk / one more column in a chunk | 0.474 ms / 0.038 ms |
| Fitted marginal: one demand upload | 0.262 ms |
| Median share of the window spent in the CPU/helper lane | 0.784 |

Two things matter for the decisions. First, a request has on the order of a
thousand generation windows, so a per-window residual of 0.4 ms aggregates to
a request-level standard error of roughly 13 ms if residuals were independent,
and the protocol measures how far they are from independent. The retired
whole-run pairs could not resolve ±300 ms. Second, the fitted chunk-versus-column
marginals are exactly the grouped-versus-single-column distinction the
[policy reassessment](evidence.md#admission-value-does-not-use-the-foreground-objective)
identified as missing from admission valuation.

### Composition with the retained state replay

#### Retained unchanged

- `capture.py`: strict capture contract, observation-fed stage-1 reproduction.
- `policy.py` + `policy_bridge.cpp`: hardware-independent production decisions.
- `machine.py`: canonical ownership, frames, exchange descriptors, publication,
  admission and helper-exchange phases, `assert_ownership`, the per-layer
  assignment logic in `serve_layer`.
- `structure.py`: chunk and helper launch geometry.
- `validation.py`: `delta_classification` and `intervention_validation`,
  including the practical-equivalence margin, correlated paired uncertainty,
  declared cost coverage and prospective binding by digest. The stage-2 input
  that `intervention_validation` requires becomes the step-C selection gate
  below, carrying the same digest binding; the per-operation timing categories
  of the old stage 2 are not reproduced.
- The fixed-work harness (`MM3_REPLAY` v2), the preserved fixed-snapshot
  executable, `fixed_capture` setup-snapshot binding, the common learned
  checkpoint and the conditioning/evaluation request separation.

#### Replaced

`Scheduler` and `ServiceTable` are replaced by a **window clock**: an object
with the same `submit/done/until/now` surface that ignores per-operation costs,
advances time only at window boundaries by the window model's prediction, and
resolves transfer completion by the availability rule below. `serve_layer`
keeps building the same descriptors it builds today; the only timing call is
one window price per layer. `calibration.py`, `profile.py`, `diagnostics.py`
and `overlap.py` remain as optional profile diagnostics and are never on the
prediction path.

#### Availability without the control's clock

A transfer of `B` bytes on link `ℓ` issued during window `k` is complete at the
first boundary `j > k` with `Σ_{i=k+1..j} T̂_i ≥ B / r_ℓ + q_ℓ`, where `T̂_i` are
the candidate's **own** predicted windows and `r_ℓ, q_ℓ` are the existing
transfer-probe link rate and latency. Publication then follows the engine's own
rule at the next `layer_join`. Helper exchanges advance phase by phase with the
host-copy rate. Nothing from a control run's timeline is imported. Each decision
is also evaluated with the availability lag shifted by ±1 window; a decision
whose classification changes under that bracket is reported unresolved.

#### Candidate-minus-control at window resolution

On common fixed work the two arms' windows align by `(forward, layer)`. Windows
with identical descriptors contribute zero predicted difference and zero model
uncertainty, by construction, because both arms would carry the same error.
Only windows whose descriptors differ contribute, and their difference is
`f(x_candidate) − f(x_control)` under one shared parameter vector. Uncertainty
of the request-level difference comes from a moving-block bootstrap over
conditioning windows (blocks of consecutive windows, to respect drift and
autocorrelation), refitting per replicate and recomputing the paired difference
jointly, plus the residual variance of the differing windows. This preserves
correlation across arms as the decision requires. Per-window residuals, lane
residuals and request totals remain separate diagnostics, never summed.

### Instrumentation

#### Tier A: window records (new, cheap)

A new capture mode `windows` in the existing control-file protocol
(`off | timeline | detailed | windows`). At `trace_routes` time the engine
stores the steady-clock stamp and the `Stats` snapshot; at `trace_service` time
it appends one fixed-size POD record (window identity, rows, generation flag,
begin/end stamps, the counter deltas above, pending transfer state, and the
`4·T` original and executed IDs) to a vector reserved at request start. No JSON
formatting, no score triplets, no ownership lists, no CUDA event, query or
synchronization during timed work; serialization happens in `capture::end()`
like the existing buffer. Forward boundaries use the existing timeline events,
which the `windows` mode also enables. Estimated size: under 200 bytes plus
`32·T` bytes per window.

The first measurement of the protocol is this tier's own overhead, because the
monitoring-off benchmark established that instrumentation is not free here.

#### Tier B: detailed capture (unchanged, state only)

Detailed capture and `MM3_REPLAY_CAPTURE` remain the source for stage-1
reproduction, policy-decision inputs and the setup snapshot. Their window times
are never used for fitting. A candidate arm gets one detailed companion of the
same fixed work for decision reproduction only.

### Qualification protocol, declared before hardware

**Budget.** Fixed-work runs cost about 7 s each plus a state restore. The
whole protocol is under one hour of GPU time; implementation dominates.

| Step | Runs | Purpose |
|---|---:|---|
| A. Tier-A overhead | 4 fixed runs, windows on/off interleaved (on, off, off, on), conditioning workload | Overhead and window-count equality. If the mean difference exceeds the margin, the record format is reduced before anything else. |
| B. Conditioning, ordinary | The existing five-request conditioning sequence with Tier A on | Ordinary-regime fit; chronological learned state restored as today |
| C. Conditioning, fixed-work pairs | Conditioning fixed workload × {control, staging 4→2} × 2 repeats, Tier A on | Fixed-work regime fit and **model-selection data** (observed per-window deltas) |
| D. Evaluation | Reserved workloads (64 and 85 rows) × {control, backing on, budget 8→2, staging 4→2, positive control staging 4→0} × 4 interleaved repeats, predictions frozen first | Decision accuracy on untouched work |

Conditioning requests precede evaluation requests chronologically and the
evaluation workloads never enter a fit. Predictions, cost data, model source and
descriptors are frozen and digested before step D, as the existing plan binding
does.

**Model-selection rule (step C, before D).** Fit both forms on the ordinary and
fixed-work conditioning windows. On the fixed-work conditioning pairs, compute
observed per-window deltas and predicted per-window deltas. Select the form
with the lower mean absolute per-window delta error; on a tie within the
bootstrap spread, select the additive form. The selected form must beat the
all-tie predictor: Spearman correlation of predicted and observed per-window
deltas of at least 0.3, and sign agreement of at least 70% on windows whose
predicted delta exceeds the residual RMSE. Failing that, no form is selected
and step D does not run.

**Practical-equivalence margin.** Proposed: 2% of the control's generation
time per workload, the lower edge of the plan's "cheap integrated 2–5% gain" band.
A difference inside the margin is a supported tie only if the paired interval
is inside it; an interval that merely contains zero is unresolved, as the
repaired gate already requires. The margin is a decision quantity, not a noise
estimate; see the decision points below.

**Drift ruler.** For every pair of runs, non-window time on the same fixed work
is compared; a pair whose ruler differs by more than the margin is counted but
excluded from the paired estimate, and the exclusion is reported. The ruler also
detects a candidate changing non-window time, which the model does not cover.

**Pass criteria for step D.**

1. The positive control (staging 4→0) is classified as a loss by both model
   and hardware with magnitude agreement under `intervention_validation`.
   This is the discrimination requirement: an all-tie predictor cannot pass it.
2. For backing, reduced admissions and staging 4→2, predicted class equals
   observed class, with declared cost coverage complete, for at least two of
   the three. A supported tie on both sides counts.
3. Any intervention with a descriptor outside the conditioning hull, or whose
   availability bracket flips its class, is reported unqualified rather than
   resolved.

**Outcomes and what they change.**

- All criteria met: the predictor is qualified at the demonstrated resolution
  for residency/transfer/cooperative parameters on fixed work. Proceed to the
  admission repair and supported offline search.
- Criterion 1 met, criterion 2 not: the predictor discriminates at the
  resolution of the positive control but the three subtle interventions are
  below the hardware's resolution under this budget. Record them as below
  resolution, keep incumbent values for them, and do not repeat for a number.
- Criterion 1 not met: the window abstraction is inadequate for this engine.
  Report, preserve the receipts, stop; do not add terms.

### Downstream use

- **Admission valuation repair.** The grouped avoided work for admitting expert
  `e` is the window model's lane difference with and without `e` in the CPU
  lane, taken inside the window's critical path; the displaced victim's value is
  the symmetric quantity over its expected uses; transfer interference is the
  `κ` terms over the transfer's in-flight windows. This replaces the
  single-column marginal times a count. The online policy can consume a frozen
  table of these marginals, but a policy and a predictor sharing prices is not
  validation: step D and the integrated confirmation remain the evidence.
- **Offline search.** Rank candidate budget, staging, horizon, margin and
  helper-refill settings by predicted paired difference against the incumbent
  on held-out complete requests, with coverage flags and the availability
  bracket carried into the ranking. Routing bonus and lenience remain outside:
  they change tokens, and only live screens qualify them.
- **Integrated confirmation.** Unchanged: the short two-task screen in both
  orders, Tier A off, against the retained usable configuration.

### Costs, risks and detection

| Item | Estimate |
|---|---|
| Native change | about 150 lines: window POD record, append at the two existing trace points under the new mode, serialization at `capture::end()` |
| Replay change | about 900 lines of Python: `window_model.py` (fit, bootstrap, coverage), `window_clock.py` (scheduler surface, availability rule), descriptor extraction from `Machine.layers` and Tier A records, protocol/evaluation script, focused tests |
| Unchanged reuse | `capture.py`, `policy.py`, `machine.py` state logic, `structure.py`, `validation.py` gates, fixed harness, snapshot binding |
| Hardware time | under one hour across steps A–D |
| Agent time before hardware | one to two days |

| Risk | How the protocol sees it |
|---|---|
| Tier A perturbs timing | Step A, paired on/off |
| Counters explain less of window time once capture overhead is gone | Fit statistics on steps B/C are reported before D; selection rule fails closed |
| Drift between repeats | Interleaved orders, window-level pairing, drift ruler |
| Fixed-work regime differs from ordinary serving | Separate fits; integrated confirmation stays live |
| Cross-window effects beyond in-flight bytes (memory pressure from backing, thermal) | Ruler band, residual autocorrelation reported |
| Replay decisions diverge from engine decisions under a candidate | Detailed companion reproduces decisions; timing never read from it |
| Runtime regime change later (e.g. adopting a stream flush on WSL) | Domain key includes binary hash and runtime flags; a refit is five ordinary requests |

### Declared operating domain and non-claims

Qualified only for: the preserved fixed-snapshot executable and its runtime
flags (CPU scale 1.3, primary query off, 14 workers plus caller, FP4 primary,
worker helper, four drafts), monitoring off, one request at a time, descriptor
values inside the conditioning hull, and the regime of the fit used. Prefill
remains unqualified until its own regime is covered. Not claimed: prediction of
changed tokens, routing or draft acceptance; any absolute speed; any hardware
ceiling; transfer of a fixed-work result to ordinary serving without the live
confirmation the plan already requires.

### Decision points for the user

1. **Margin.** 2% of control generation time is proposed; the plan's investment
   band would also support 3%. A larger margin resolves more ties and fewer
   gains.
2. **Positive control.** Staging 4→0 is proposed because it stays inside the
   harness protocol and moves many windows inside covered descriptor ranges.
   A different large, in-domain intervention is acceptable if preferred.
3. **Online consumption.** Whether the admission policy should read a frozen
   marginal table derived from the same fit, or only the offline search should.
   The proposal recommends the former after step D passes, with the integrated
   screen as the check that the shared prices did not merely agree with
   themselves.

<a id="opus-proposal"></a>

## 5. Opus-generated proposal: timing-free replay with a micro-randomized layer response

**Author: Opus.** Source: the proposal supplied by the user on 2026-11-06.
The evidence, process observations and preparation notes below are part of that
submission and describe its preparation-time context.

I propose replacing the discrete-event timing simulator with two separate parts. A timing-free replay works out what work each candidate does, and a small response model fitted from randomized real runs prices the difference. Nothing is implemented and no GPU was used. The evidence below comes from existing receipts, and the randomized step, which is the core of the idea, is untested.

### How it works

1. Timing-free replay (work). Keep Machine, the shared native policies, original-demand credit and the ownership checks. Replace service tables with constant placeholder durations, so availability follows joined-layer order rather than a predicted clock. The output is each layer's lane assignment, uploads and exchanges under the candidate.
2. Layer response (time). Predict only candidate minus control, summed over layers with identical demand:

   layer_ms = position/rows term + max(CPU branch, helper branch) + upload price × uploads + exchange prices + one-layer lag terms

   That's about 15–20 coefficients. The CPU branch is priced per group and the helper branch per column and width. Queues and contention aren't simulated, because they're already in the measured response.
3. Micro-randomized fitting (causal prices).
   - Fixed-work runs give each generation layer a random stage cap of 0–4 and each admission opportunity a coin-flip hold; backing varies between runs.
   - Per-forward intercepts remove slow drift.
   - Timing comes from a preallocated layer-timestamp array exported after the timed work, with capture and Nsight off.

### What the receipts show

Generation delta against control, in ms. The hardware column uses workload-0 prefill as a drift covariate, and ± is the half-range of two pairs, not a confidence interval.

```text
┌──────────────┬───────────────┬─────────────────────────────┬──────────────────────────────────────────┬───────────┐
│ Intervention │ Retired model │ Layer fit, no randomization │ Same fit, upload price 0.5 ms (post hoc) │ Hardware  │
├──────────────┼───────────────┼─────────────────────────────┼──────────────────────────────────────────┼───────────┤
│ Backing on   │           +12 │                         +78 │                                       −6 │  −10 ± 87 │
├──────────────┼───────────────┼─────────────────────────────┼──────────────────────────────────────────┼───────────┤
│ Budget 8→2   │           +25 │                        +120 │                                     +220 │ +258 ± 74 │
├──────────────┼───────────────┼─────────────────────────────┼──────────────────────────────────────────┼───────────┤
│ Staging 4→2  │          −399 │                        +442 │                                      −34 │  −8 ± 147 │
└──────────────┴───────────────┴─────────────────────────────┴──────────────────────────────────────────┴───────────┘
```

- Replay doesn't need milliseconds: across all 20 runs (durations ×0.1–×10, copies ×0.1–×10), exchanges issued matched hardware (237/237, 58/58) and D2H bytes matched. Uploads were within +0.18–0.63%, and optional H2D differed by 1–2 records in two arms. Predicted deltas moved by at most 10.5 ms.
- The layer form holds: using observed branch durations, layer time ≈ 1.1–1.7 ms + (0.97–1.11) × max(CPU, helper), with R² 0.81–0.94. The helper finishes last on 13–32% of layers.
- Randomization should resolve small effects: I injected known effects into real residuals with a random stage cap. They came back within 0.02 ms per group, which is about ±13–18 ms on the staging delta from one fixture. Raw arm pairs spread ±201–339 ms. This is a simulation on detailed-capture noise and assumes a uniform effect.
- Good totals don't mean good decisions: without randomization, leave-one-capture-out totals land within ±4.7% when anchored by same-capture prefill, yet the fit still gets staging wrong (+442). Anchoring can also absorb real policy effects when prefill work differs between arms.
- Unresolved conflict: an upload price near 0.5 ms fits the three runs above, which had capture off. It misses the static-versus-adaptive pair, run with capture on (−66 predicted, −459 measured), which needs about 0.3. Capture may distort that price, but this is unproven and the first experiment should settle it.

### Insight

- The planner stages only when it expects a benefit, so upload counts are confounded with how much a layer stands to gain. More observational data won't remove that; random assignment can.
- The noise is mostly low-frequency: layer residuals are small (0.42–0.76 ms, lag-1 −0.02 to +0.23), yet arm totals swing by hundreds of ms. Contrasts inside a run should cancel much of that drift.

### Retained and retired

- Retained: strict capture parsing, reproduce, Machine state logic, the native policy bridge, setup snapshots, the fixed-work harness, delta_classification (tools/mm3/cooperative_replay/validation.py:86), and detailed capture as a diagnostic.
- Retired from prediction: costs.py tables, calibration.py, scheduler contention factors and isolated probes.
- Estimated size: under 60 native lines, all behind replay_fixed_batch, plus about 350 lines of Python. The native hooks are:
  - per-layer cap at src/mm3/mm3_model.cu:2866
  - workload format at src/mm3/mm3_replay_workload.hpp:11
  - admission hold at src/mm3/mm3_native_adaptive.inc:289

### Prospective test (about 1¼ hours of hardware, including one engine load)

Declare all of this before any measurement:

- Fit: rows from chronological conditioning requests only.
- Evaluation: three reserved pairs of new requests across coding, prose, extraction, reasoning and tools.
- Arms:
  - control
  - backing on
  - budget 8→2
  - staging 4→2
  - budget 8→16 (a possible gain)
  - static with budget 0 and stage 0 (positive control; the same-work pair measured about +10%, across different sessions)
- Truth: 12 randomized blocks of all six arms, with a short fixed-work sentinel before each arm as a drift covariate. That resolves about ±95 ms if the two-pair spread holds.
- Margin: 2% of measured control, about 134 ms on the last workload. The last plan's 241 ms was 2% of the inflated prediction (/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/fixed-interventions-20261105/predict.py:48).
- Uncertainty: the same bootstrap coefficient draws applied to both arms, which preserves their correlation.
- Pass, on frozen predictions:
  - no gain/loss reversal on any resolved intervention
  - static control within ±25%
  - lower mean delta error than both an always-tie predictor and the non-randomized layer fit
- Stop early:
  - Ledger sums miss generation_target_ms by more than 0.5%, or ledger on/off pairs differ beyond noise.
  - After the first two randomized fixtures (about 5 minutes), the upload-price standard error exceeds 0.03 ms per group or the price is unstable across repeats. In that case, reject this proposal.

### Limits

- Fixed rows only: it can't predict changed tokens, routing, drafting or sampling, so lenience, draft width and routing bonus still need live confirmation. Integrated confirmation stays required.
- Measured regime only: prices hold for this CPU participant count, query off, on WSL. Changing any of those means refitting.
- Helper capacity: those decisions stay unqualified, because helper work isn't randomized.
- Long-horizon effects: layer-level randomization can't see them, so the arm-level truth above remains the arbiter.

If it qualifies, the same prices can feed the admission repair the plan already asks for. If the upload price really is near 0.5 ms, a staged group roughly breaks even against its 0.54 ms CPU cost, while residency saves the full cost. That would favor admissions over staging, but it's only a lead for now.

### Notes for your comparison

- Independence: a one-line summary of Fable's proposal appeared in the shared memory index mid-session, after this analysis was mostly done. I didn't open its file. Terra may see that line too if it shares this memory, so consider moving it until the comparison is done. I kept this proposal out of shared memory for the same reason.
- Changes made: I corrected one stale memory that still called the ordinary-query test open. Nothing in the repo changed or was committed; the untracked .claude/ directory isn't mine.
- Hardware: the recorded engine is no longer listening on port 8097, and something else is holding about 30.9 GB on the 5090. I left it alone.

To rerun the numbers, run logical.py first, then evidence.py; I removed the regenerable caches. Everything is in /home/mdwbr/.claude/projects/-home-mdwbr-project-maya/c33d2062-9d34-495f-b4e4-e010d613e5a2/analysis/, with results in evidence.txt and replay counts in logical.log.

If it would help the comparison, I can also publish this as a private shareable page.

<a id="sol-61-proposal"></a>

## 6. sol-6.1-generated proposal: empirical complete-layer transitions

**Author: sol-6.1.** Source: the proposal supplied by the user on 2026-11-06.

Propose an empirical model of complete layer transitions: preserve Maya’s state replay, and learn elapsed time and exchange progress jointly from the running engine.

The practical reason is the retired predictor’s failure (docs/minimax-wsl/evidence.md:88): independently measured operation costs did not compose into reliable policy comparisons. Measuring complete intervals would capture CPU/GPU overlap, host submission delays and transfer contention together.

1. Predict between existing execution boundaries.
   Use intervals between layer route/readback joins, exposing additional existing helper or forward-end decision boundaries where necessary. Each prediction returns:
   - elapsed wall time;
   - which pending transfer phases finish before the next existing readiness poll.

   Exchanges can remain pending across intervals. Instrumentation adds no synchronization.
2. Keep ownership and policy decisions explicit.
   Retain original-demand learning, shared native policy entry points, frame accounting, reader protection, backing validity and publication rules. The empirical model supplies timing and readiness predictions; the deterministic state machine applies legal transitions.

   Inputs include current expert groups and reuse, CPU/5090/3090 assignments, ready staging frames, incoming and preservation bytes, pending exchange phases, queue depth and elapsed age. This distinguishes retaining an existing upload from issuing another copy, and charges displaced service through subsequent replay.
3. Fit a small joint model from native execution.
   Start with a shallow regression tree whose leaves retain joint observations of elapsed time and pending-work progress. Use the existing fixed-work runner (src/mm3/mm3_model.cu:4948), with compact in-memory boundary records and formatting outside timed execution. Measure instrumentation overhead explicitly.

   Fit against complete cooperative intervals and paired candidate-minus-control request times. Keep CPU workers, arithmetic and submission mode fixed initially. Sparse or unfamiliar operating states produce an unqualified prediction.
4. Roll each candidate forward independently.
   Starting from a declared initial state, replay chronological conditioning under that candidate. Its predicted progress determines subsequent availability, placement and decisions. Never import the control’s future publication times or availability.

   Preserve correlation by resampling whole paired request blocks, rather than treating thousands of layer observations as independent evidence. Report interval, request and policy-difference errors separately.
5. Give it a bounded prospective test.
   A practical pilot cap would be 48 native sequence executions plus 12 ordinary requests, covering source capture, instrumentation checks and final confirmation. Freeze the allocation before measurement.

   Compare the incumbent with backing enabled, reduced admissions and reduced staging. Separate fitting sequences from untouched chronological evaluation sequences. Predeclare a practical-equivalence band, provisionally ±3% of control time.

   Qualification requires correctly resolving at least one consequential held-out difference and improving effect prediction over always predicting zero change. A supported tie requires uncertainty wholly inside the equivalence band; an interval crossing zero can remain inconclusive. Exhausting the budget without useful discrimination means the replacement remains unqualified.

This reuses the validated capture/reproduction path (tools/mm3/cooperative_replay/capture.py:122), native policy bridge (tools/mm3/cooperative_replay/policy.py:29), snapshots and ownership checks. Implementation would extract those reusable parts from the current timing scheduler and add the empirical transition interface.

Once qualified, it could screen nearby residency policies and compare admission alternatives over recent original-demand patterns. Ordinary model requests would still establish actual speed and usability, because fixed expert work cannot predict changed routing, text or speculative acceptance.

The main risk is that the boundary state omits a driver or queue condition that materially changes performance. Held-out rollouts must expose that risk. This is one proposal for comparison; it is not implemented or qualified.
