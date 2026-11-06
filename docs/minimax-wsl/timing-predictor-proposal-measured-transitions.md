# Timing predictor proposal: measured transitions at policy boundaries

This is one proposal for the open replacement comparison. It has not been
selected, implemented, measured, or qualified. The [central MiniMax plan](README.md#timing-model-replacement)
owns requirements, decisions, status, and completion.

## Motivation

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

## Proposed model

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

## Proposed first evaluation

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

## Limits and follow-through

- Similar intervals may be too sparse, especially for changed transaction
  combinations, or may hide relevant queue/driver state. Unsupported regions
  and compounding prediction error could make this approach unhelpful.
- Fixed-work replay estimates scheduling effects conditional on captured expert
  rows. Changed residency can change cache-aware routing, generated text, and
  speculative acceptance. It cannot establish ordinary serving speed or quality.
- A useful supported prediction is still followed by the central plan's short,
  matched actual-engine confirmation. No runtime parameter is selected by this
  proposal or by the historical failed timing predictions.
