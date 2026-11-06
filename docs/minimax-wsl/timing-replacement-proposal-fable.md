# Timing-model replacement proposal (Fable): in-situ layer-window cost model

**Status: proposal for the five-way comparison opened by the
[replacement decision](README.md#timing-model-replacement), 2026-11-06.** Nothing
here is implemented, measured or selected. The central plan owns selection;
this document is one of the five candidates it compares.

## Summary

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

## Why the retired approach missed, and what this changes in principle

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

## The model

### Unit: the cooperative layer window

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

### Descriptor

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

### Form

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

### What the existing capture already shows

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

## Composition with the retained state replay

### Retained unchanged

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

### Replaced

`Scheduler` and `ServiceTable` are replaced by a **window clock**: an object
with the same `submit/done/until/now` surface that ignores per-operation costs,
advances time only at window boundaries by the window model's prediction, and
resolves transfer completion by the availability rule below. `serve_layer`
keeps building the same descriptors it builds today; the only timing call is
one window price per layer. `calibration.py`, `profile.py`, `diagnostics.py`
and `overlap.py` remain as optional profile diagnostics and are never on the
prediction path.

### Availability without the control's clock

A transfer of `B` bytes on link `ℓ` issued during window `k` is complete at the
first boundary `j > k` with `Σ_{i=k+1..j} T̂_i ≥ B / r_ℓ + q_ℓ`, where `T̂_i` are
the candidate's **own** predicted windows and `r_ℓ, q_ℓ` are the existing
transfer-probe link rate and latency. Publication then follows the engine's own
rule at the next `layer_join`. Helper exchanges advance phase by phase with the
host-copy rate. Nothing from a control run's timeline is imported. Each decision
is also evaluated with the availability lag shifted by ±1 window; a decision
whose classification changes under that bracket is reported unresolved.

### Candidate-minus-control at window resolution

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

## Instrumentation

### Tier A: window records (new, cheap)

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

### Tier B: detailed capture (unchanged, state only)

Detailed capture and `MM3_REPLAY_CAPTURE` remain the source for stage-1
reproduction, policy-decision inputs and the setup snapshot. Their window times
are never used for fitting. A candidate arm gets one detailed companion of the
same fixed work for decision reproduction only.

## Qualification protocol, declared before hardware

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

## Downstream use

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

## Costs, risks and detection

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

## Declared operating domain and non-claims

Qualified only for: the preserved fixed-snapshot executable and its runtime
flags (CPU scale 1.3, primary query off, 14 workers plus caller, FP4 primary,
worker helper, four drafts), monitoring off, one request at a time, descriptor
values inside the conditioning hull, and the regime of the fit used. Prefill
remains unqualified until its own regime is covered. Not claimed: prediction of
changed tokens, routing or draft acceptance; any absolute speed; any hardware
ceiling; transfer of a fixed-work result to ordinary serving without the live
confirmation the plan already requires.

## Decision points for the user

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
