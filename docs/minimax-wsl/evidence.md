# MiniMax evidence and provenance

The [central plan](README.md) owns requirements, status and work orders. This file
records observations at their original scope; it is not a second tracker.
The initial documentation consolidation ran no model, benchmark or build.
The subsequent Maya implementation and scoped measurements are recorded below.

**Timing-model disposition, 2026-11-06:** the user requires replacement of the
factorized per-operation timing approach. The failed timing predictions and
unresolved interventions below remain evidence; their next-step language does
not authorize another calibration campaign. State replay, instrumentation and
valid ownership/decision checks remain reusable. The central
[replacement decision](README.md#timing-model-replacement) owns proposal selection,
implementation and qualification; no replacement or new speed gain is established
by this documentation update.

**Qualification correction, 2026-11-04:** the implementation report called the
active Maya milestone delivered. That overall completion claim is withdrawn by
review; the [central work status](README.md#active-maya-work) and
[completion contract](README.md#execution-and-completion) own the remaining work.
The measured sequential-start policy improvement and passing scoped checks remain
valid at their recorded scope; they do not establish warmed policy selection.
The speed arms held lenience at 0.4 and the cache bonus at 0.05; they do not perform
the required value optimization of either, or close the other active parameter
families in the [central optimization scope](README.md#required-parameter-optimization).
That sequential-start comparison did not establish a gain against the previously
usable engine under the new protocol. The later independently conditioned evidence
below is separate. Raw receipts, including their original delivery filenames and
claims, are preserved rather than rewritten.

## Shared-model diagnostic study

**2026-11-07, primary-agent work; no delegation.** Generation only, speculation
off, prefill as setup. All 22 declared executions ran without step errors: 16
measurements, four independent conditioning requests and two bootstraps. Warmed
time was 1,304.5 s of the 1,800 s cap (standalone 1,011.3 s, trace 293.1 s). The
loads took 316.5 s and 372.6 s and are reported separately. Both used binary
`7bf7dea00e3738b9cf354bda1177be354e20def337e100d25e37b1f83374cc99`, native source
`963cdb0`. No reserve remained, so anything these runs could not settle stays
unresolved. This is an effect screen, not a speed claim, predictor validation or
quality check. [Frozen protocol](shared-model-diagnostic-protocol.json),
[ledger](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-diagnostic-20261107/ledger.json),
[predeclared analysis](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-diagnostic-20261107/analysis.json),
[trace overlap](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-diagnostic-20261107/trace-overlap.json),
[post hoc scripts and outputs](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-diagnostic-20261107/posthoc/).

**Longer requests.** Each prompt produced the same output in all three arms, with
384 tokens and finish `length`. Light against each off arm: coding +2.22%, +0.82%;
prose -3.61%, -0.81%; pooled -0.21%. These are observations of actual work, not
the isolated monitoring test.

| Prompt | Off, first | Light | Off, second |
|---|---:|---:|---:|
| Coding, 1,056-token prompt | 43,918.8 ms, 8.74 tok/s | 44,895.8 ms, 8.55 tok/s | 44,528.9 ms, 8.62 tok/s |
| Prose, 240-token prompt | 38,343.1 ms, 10.01 tok/s | 36,959.3 ms, 10.39 tok/s | 37,259.9 ms, 10.31 tok/s |

**Fixtures,** 64 one-row generation forwards from the coding light capture:

| Comparison | First arm | Second arm | Third arm | Fourth arm | Pair differences | Control drift | Rule |
|---|---:|---:|---:|---:|---:|---:|---|
| Monitoring off/light/light/off | 6,734.7 ms | 8,036.6 ms | 7,088.5 ms | 6,395.0 ms | +19.33%, +10.85% | -5.04% | Material |
| Normal/held/held/normal, all light | 7,105.9 ms | 7,329.1 ms | 7,060.8 ms | 6,685.6 ms | +3.14%, +5.61% | -5.91% | Material |
| Trace pair: off, detailed plus Nsight (separate process) | 7,546.7 ms | 10,758.2 ms | | | +42.6% | | Diagnostic only |

**Observational states** against matched current-work windows, as a percentage of
generation, per light capture (coding, prose, then the four same-work fixture arms):

| State | Rule | Coding | Prose | Fixture arms | Forwards under 1.4x median only (post hoc) |
|---|---|---:|---:|---:|---:|
| CPU record not used recently | Unresolved | +0.16 | -0.11 | +0.91 to +2.17 | -0.69 to +1.11 |
| Primary record not used recently | Unresolved | +0.01 | +0.09 | -0.71 to +1.49 | -0.06 to +0.63 |
| Helper record not used recently | Unresolved | +0.17 | -0.26 | +0.81 to +3.18 | +0.16 to +1.09 |
| Residency changed in this layer recently | Unresolved | -0.54 | +0.03 | -2.23 to +3.90 | -0.61 to +0.90 |
| Primary exchange pending, or a frame reserved | Bounded | +0.38 | +0.27 | +0.25 to +0.38 | +0.31 to +0.44 |
| Helper lease open | Unresolved | +1.46 | +1.10 | +0.86 to +1.35 | +0.75 to +1.07 (bounded) |
| Either pending (busy flag) | Unresolved | +1.73 | +1.39 | +1.12 to +1.53 | +1.07 to +1.44 |
| At least 128 forwards into the request | Unresolved | +2.37 | +6.74 | no exposure | +3.28, +3.50 |

Standard errors are in the analysis file. Two exchanges at once occurred in only
1 to 3 windows per capture. These are associations, not causal prices.

**Later-position excess is not context length.** With no copy, median join wait
stayed 0.34 to 0.37 ms through both requests, at positions 240-623 and 1056-1439,
and one-job CPU span stayed 0.52 to 0.57 ms. The excess sits in the helper lane.

**Availability.** Across all 65,550 generation windows, free staging frames were 8
(63,864), 7 (1,636) or 6 (50) of 8. The staging limit of 4 never bound while CPU
work remained, at most 2 new copies occurred in a window, and no foreground copy
was unretired at a plan boundary.

**Foreground work and overlap.** Join-to-join cycles partition a forward exactly.
Generation shares in the long requests: host spans 54-55%, join wait 31%, submission
of the next layer's work 4-5%, prefix 3-4%, outside forwards 2.0-2.2%, policy tail
under 1%. These results are post hoc, measured from the fixed records:

- **CPU and helper overlap.** The lane span matches `max(CPU, helper)` with median
  absolute error 0.010-0.014 ms; their sum misses by 0.52-0.56 ms. Overlap hides
  about 15-16% of generation in the long requests (6,996 and 5,529 ms). CPU span is
  linear at 0.52-0.56 ms per job. The helper finishes last in 61-81% of windows.
- **The primary path does not overlap them at `T=1`.** Without a profiler, join wait
  does not fall as the host span grows from under 0.7 ms to over 2 ms: 0.32-0.37 ms
  with no copy and 0.90-0.92 ms with one. In the trace, primary expert kernels spent
  0.37 of 218.3 ms during host lanes and started before lane end in 10 of 3,647
  windows; demand copies spent 10.5 of 2,207.2 ms and started early in 14 of 3,102
  windows; `wait_publication` totalled 7.2 ms; the first expert kernel started a
  median 1.96 ms after its launch returned. This fits the earlier launch-group
  observation (seven launches per layer at `T=1`), which this study did not
  separately prove as the mechanism.
- **Demand copies are exposed.** Each adds 0.54-0.55 ms of join wait in every capture;
  the trace's median copy took 0.553 ms (57.6 GB/s). That is 9.9-14.8% of generation
  in the long requests and 17-23% in the fixtures.
- **Layers share a model.** After the shared work table, median residual per layer is
  within about ±0.03 ms except the first MoE window (+0.41 ms) and the next (+0.08 ms).
  Absolute residuals sum to 1.19 ms of a roughly 105 ms forward. This is a check on
  existing data, not fitted layer parameters.

**Held admissions.** Normal arms issued 105 exchanges and 13 helper leases (1,458 MiB
H2D, 3,189 MiB D2H); held arms issued none. Holding changed generation picks: CPU
3,416 to 3,763, primary resident 2,655 to 1,923, new copies 2,469 to 2,934, helper
5,994 to 5,909. A work table built from any single capture predicts +346 to +386 ms
from that changed work alone; observed differences were +223 and +375 ms. Direct
admission costs visible in normal arms are prefix +64 ms, policy tail +13 to +15 ms
and busy-window excess +91 to +95 ms per 64 forwards, 1.1-2.5% in total. Same-pair
contention beyond changed work was +53 and -82 ms, inside the noise. A near-zero net
would not have shown these parts negligible, and this result does not isolate a copy
price. Exchange copies ran mostly in gaps and rarely overlapped lanes (0 of 109 ms);
a demand copy overlapped by one took 1.03 ms rather than 0.55 ms, in 48 of 3,931 cases.
Helper lease copies took 12.0 ms for 30.4 MB; helper `project` kernels ran 1.9 times
longer during them (53 kernels).

**Helper speed state is the main noise source.** The same helper `project` kernel's
per-forward median ranged 0.076-0.627 ms in the trace over long runs of consecutive
forwards, at similar kernel counts (84-108 per forward), while the primary's stayed
0.016-0.023 ms. Without a profiler, one-job helper service had p10 0.34-0.49 ms and
p90 0.83-1.50 ms, tending to climb from about 0.37 to about 0.7 ms, spike near 1.5 ms
and reset. It tracks forward time within captures (r 0.63-0.95) and generation across
the four same-work arms (r 0.94, four points, descriptive). Forwards over 1.4 times
their capture's median carried 0-14.1% of forward time, and held arms have them too.
Waiting on the helper past the CPU lane is 19-30% of generation. One sample taken just
after the standalone series showed the 3090 in P3 at 360 MHz core and 5,001 MHz memory
while the 5090 was in P1 at 2,865 and 13,801 MHz. This points to a 3090 clock or
power state but does not prove it: clocks were not sampled during runs because
monitoring was off by design.

**Monitoring reading.** The rule labels monitoring material, and that label stands as
recorded, but recording is not established as the cause. Measured recorder work is
9.3 µs per 57-layer forward (0.009%), about 1,000 times below the fixture difference.
Prefill runs under the recorder too, and its off-to-light pairs were -6.28% and +0.29%,
while the same setup work spanned 15.9% across the eight arms. Same-work light arms
alone span 6,686-8,037 ms, and `monitor-light-1`'s excess is a burst in its final
forwards. Off arms have no records, so this cannot be attributed further.

**Foreground table, held out.** The predeclared check is unresolved: a shared table
missed generation by -8.9% and +10.3% between prompts, and -0.1%, +1.5% and -11.7% on
fixtures. Post hoc, a median table misses between prompts by -1.43% and +1.56%, so
slow spells caused most of that gap; on fixture arms it ranges -2.76% to +8.81%.
Hits by 64-forward block were coding 59.7, 68.6, 66.2, 55.2, 68.3, 66.5% and prose
73.2, 72.6, 75.5, 79.5, 80.2, 80.3%.

**Limits.** Fixtures use synthetic token input and uniform mixture weights. Post hoc
results were chosen after seeing the data and did not change any predeclared label.
Coverage is `T=1`, eight staging frames, these two prompts and positions 240-1439.
Learned state is process-local, so the trace pair is a separate diagnostic.

## Acceptance correction and ordinary primary-query test

**2026-11-05:** nested operation, branch and request residuals are no longer
summed into one error allowance. Stage 2 retains all five required measurement
categories, checks whole-request error once and reports the other residuals as
non-additive diagnostics. Passing that check alone cannot authorize ranking.
Stage 3 consumes uncertainty for the paired candidate-minus-control difference,
requires a prospective practical-equivalence margin and accepts supported ties.
A broad interval containing zero, a reversed decision or an unexplained magnitude
miss remains unresolved. Synthetic checks cover the 10 ms residual counted at
three levels, exact/practical ties, uncertain ties, gains, losses and reversals.

This corrects acceptance logic, not the existing held-out prediction. That run's
evaluator called the diagnostic comparison and explicitly recorded unresolved;
the defective gate did not reject it. Its 275.827 ms prediction error remains.

The ordinary-inference query comparison is preregistered in
[the intent](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/ordinary-query/intent.json)
and [protocol](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/ordinary-query/protocol.json).
It uses one `cudaStreamQuery` immediately after primary submission and before
CPU service. Success and not-ready statuses continue without a wait; another
status releases the CPU wait word, retires submitted helper work and marks
ownership uncertain before returning the error. There is no polling loop and
the later Windows-only `k::submit` is unchanged. A source build, 58 focused
Python checks and four native CPU checks pass.

**Ordinary-request result: no useful integrated gain established; retain query
off.** The same process, learned checkpoint, request settings and output budgets
were used in both orders, with normal adaptation within each arm. The clean load
had 438 primary, 741 helper and 6,117 HOST records. Nsight/CUPTI libraries,
detailed capture, intermediate monitoring and client stream tracing were absent.
Four-draft DSpark with lenience 0.4 and cache-aware top-four bonus 0.10 remained on.

| Arm | Python tok/s | Prose tok/s | Pooled tok/s |
|---|---:|---:|---:|
| Off, first | 17.941 | 15.434 | 16.593 |
| On, first | 17.760 | 17.576 | 17.668 |
| On, reversed | 17.723 | 17.462 | 17.592 |
| Off, reversed | 17.748 | 17.377 | 17.560 |

The pair effects were **+6.48% and +0.18%**. Pooling both orders gives +3.32%,
but the off control itself moved +5.83%; this does not meet the preregistered
useful-gain decision. Coding output and draft counts matched across all arms.
The first off prose response differed and needed 47 target rounds, versus 42 in
the other three; those later three matched in output and draft counts. The
ordinary test therefore does not confirm that the earlier fixture's queue
effect yields a useful serving improvement. It also does not prove equivalence
across workloads. No third speed repetition is justified by this result.

Each arm emitted 256 tokens across the two requests. Both responses were
truncated, so these are speed results, not completed-task functionality passes.
Enabled query counts exactly matched cooperative windows and all returned
not-ready; disabled counts were zero. No unexpected query error occurred, but
that does not exercise the asynchronous-error recovery path.
[Raw receipts and primary-agent evaluation](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/ordinary-query/evaluation.json).

## Prospective fixed-work intervention result

**2026-11-05: all three interventions remain unresolved; no parameter selection.**
Two fresh high-effort source requests supplied 64 and 85 evaluated generation
rows, including rejected speculative work. Both complete captures passed
observation-fed decision/state reproduction. Their observed durations were not
used for fitting. The native setup snapshot now exports the complete existing
replay state outside timed execution, allowing actual allocations, aliases,
credits, demand and pending state to bind the predictions without detailed
tracing during hardware tests.

The earlier foreground refit covers generation with weight copies disabled.
This separate **full-work probe-based comparator** uses the existing isolated
full-bank measurements and independent auxiliary/C1 captures. It is still the
factorized resource model; it is not a qualified extension of the foreground
refit's +5.57% result. Primary repeated-record chunks are represented as a
feature within their unchanged kernel/launch geometry. Unmeasured mixed helper
reuse receives an explicit bracket from measured distinct/paired jobs and
remains unqualified. Aligned transfer contention, multiway effects and remaining
submission-context uncertainty are not invented or absorbed into multipliers.

Cost data, model source, workload files, starting state and all predictions were
frozen before hardware. Each arm restored the same original-demand learning and
canonical rosters, then retained state across the two fixed workloads. The
sequence was control/backing/reduced-admissions/staging, then reversed. Detailed
capture, Nsight and optional monitoring were absent during all timed fixtures;
the explicit primary query stayed off. Synthetic token inputs and fixed expert
rows isolate this target-work comparison; drafting, sampling, changed output
trajectories and quality are outside its scope.

| Configuration | Predicted generation, ms | First observation, ms | Reversed observation, ms | Mean difference from control, ms |
|---|---:|---:|---:|---:|
| Control: budget 8, staging 4, no backing | 12,062.015 | 6,496.753 | 6,889.550 | 0 |
| Backing on | 12,073.787 | 6,567.770 | 6,558.197 | -130.168 |
| Admission budget 8 → 2 | 12,086.613 | 7,099.231 | 6,814.552 | +263.740 |
| Staging maximum 4 → 2 | 11,663.457 | 6,776.323 | 6,593.822 | -8.079 |

The baseline generation prediction is **80.2% high** against the measured mean.
The baseline itself moved **+6.05%** between orders. Paired candidate-minus-control
differences changed sign for all three interventions: backing +71/-331 ms,
reduced admissions +602/-75 ms and staging +280/-296 ms. Neither the preregistered
241.240 ms practical-equivalence margin nor the incomplete model uncertainty
supports a resolved gain, loss or tie. Two-repeat half-ranges are observed spread,
not confidence intervals. Broad model envelopes containing these observations do
not validate the predictor, and the large central miss remains visible.
A subsequent acceptance-only audit makes declared missing cost coverage an
explicit reason an intervention cannot pass. It leaves the frozen predictions,
hardware observations and unresolved outcome unchanged.

The physical effects occurred: backing allocated **13,952,352,256 bytes** and
removed **7,548,628,224 bytes** of primary preservation D2H across the two fixed
workloads. Reducing admissions changed the observed count from 237 to 58 in both
orders, exactly matching the count prediction. Staging uploads changed from
11,267 to 6,592 records; the corresponding predictions were 11,326 and 6,604.
These count checks do not establish exact autonomous decisions or timing.
The retained original schedule remained feasible when its D2H durations were
shortened; this offline diagnostic added no inference wait.

The complete-work CPU service prediction was 13,489 ms versus 15,465/16,585 ms
observed for the control. Those counters include prefill and overlap GPU work;
they do not decompose the generation-only +80.2% error. There are no timed GPU
endpoints in this uninstrumented experiment to assign the remaining miss to a
specific mechanism. Physical frame addresses can rotate during logical restore.
The result does not justify global lane multipliers, a larger passing tolerance,
or choosing a new policy from these rankings.

[Frozen predictions](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/fixed-interventions-20261105/plan.json),
[hardware receipts](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/fixed-interventions-20261105/hardware.json),
[primary-agent evaluation and repaired gate](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/fixed-interventions-20261105/evaluation.json),
[explicit cost-coverage gate audit](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/fixed-interventions-20261105/intervention-gate-coverage-explicit.json),
[counter/service audit](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/fixed-interventions-20261105/counter-and-service-audit.json).

## Foreground timing reconciliation

**2026-11-05, primary-agent work; no delegation.** This account concerns fixed
evaluated expert work with staging and optional weight exchanges disabled. It
does not predict changed generated tokens or qualify ordinary-request speed.
[Complete observation/prediction account](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/foreground-reconciliation/reconciliation.json).

The original fixed generation took 9,995.161 ms. Its sequential host-service
spans total 8,367.031 ms: the CPU/helper lane accounts for 7,223.177 ms,
planning 93.851 ms, helper preparation 66.922 ms and router-record formatting
565.240 ms. The remaining 417.841 ms inside those spans includes primary host
submission and unseparated local/capture overhead. Another 1,628.130 ms falls
outside the host-service spans, including dense/attention/readback, target
prefix/tail and intervening handling. This is an observation budget; the
overlapping CPU/GPU service sums must not be added to it.

The profiled companion separates GPU activity from event envelopes:

| Generation-layer observation | Primary | Helper |
|---|---:|---:|
| CUDA event envelopes | 7,155.812 ms | 2,671.522 ms |
| Recorded kernel activity | 215.264 ms | 2,331.370 ms |
| Gaps between first and last kernels | 6,761.962 ms | 49.655 ms |

Most primary gaps occurred after launch APIs had returned. A concrete empty
primary group delayed its next kernel about 9.7 ms, ending just after CPU batch
completion. A fixture-only `cudaStreamQuery` after primary submission then reduced
the generation compute-event sum from 6,908.459 to 409.682 ms, while the explicit
wait-event sum rose from 208.543 to 6,005.181 ms. This supports a queue-progress
dependency and a measurement-boundary explanation; it does not identify every
driver mechanism or justify scaling arithmetic costs.
[Query-control evidence](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/foreground-query/evaluation.json).

The query pair used identical evaluated work, published rosters and
effective starting scores. Raw score timestamps were rebased to the advancing
row clock. Fixed target generation fell from 10,073.786 to 8,920.105 ms (11.45%),
but CPU and capture costs also changed. This single pair is diagnostic evidence,
not a selected integrated speed gain. The separate original/profiled comparison
has a 438-versus-437 primary-record difference (one expert affected eight
generation layer calls) and profiling overhead; it is not a matched speed comparison.

Helper calibration now follows each occupied width after the native eight-column
split, retaining job counts and repeated records. GPU group envelopes, remaining
event-envelope cost, activation/result transfers, caller preparation, worker
submission and completion detection are separate. Two short 20-row fixtures
filled the required width-one and width-four count boundaries. Generation
measurements are separated from prefill context; new primary mixtures can
interpolate within measured chunk counts and per-record widths without pretending
every combination needs its own measurement. Such interpolation still requires
shape-holdout evidence and a timing check.

The full-bank probe measured 1,599 cells and rotated 6,118 actual HOST records,
with one CPU pool at a time. It did not establish a working-set improvement:
for the **same original fixed workload**, the 64-record table predicted
6,741.616 ms versus 6,757.804 ms observed CPU service (-0.24%); the full-bank
table predicted 6,267.514 ms (-7.26%), using the same current shape encoding and
median estimates for both.
Collection circumstances also differed, so no isolated causal working-set
effect is claimed. The earlier -34% CPU figure used a different ordinary request.
[Matched CPU comparison](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/foreground-reconciliation/matched-cpu-probes.json).

The prospective held-out check froze cost data, source and predictions before
hardware execution. It retained the source's complete evaluated work, including
speculative rows, and verified current static rosters and zero weight copies.
The generation interval covered 86 target rows across 20 target batches and
57 MoE layers. Its state check reproduced 1,995 whole-fixture staging decisions
and 19,608 original generation selections.

| Held-out generation measurement | Prediction | Observation |
|---|---:|---:|
| Complete interval | 5,228.542 ms | 4,952.715 ms |
| CPU service sum | 3,361.721 ms | 2,950.185 ms |
| Helper event-span sum | 1,990.631 ms | 1,396.285 ms |
| Host-service span sum | 4,357.087 ms | 4,056.199 ms |

The direct whole-interval error is +5.57%; CPU and helper spans are +13.95% and
+42.57%. Their overlap means they are not an additive error decomposition.
The broad low/high model envelope (1,974.761–12,670.227 ms) is not a calibrated
confidence interval. Falling inside it does not pass the timing gate. CPU and
primary shape interpolation, sparse helper cells and production-context
generalization remain unqualified.
[Frozen plan](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/foreground-heldout/plan.json),
[per-operation diagnostic](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/foreground-heldout/diagnostic.json),
[evaluation](/mnt/wsl/transfer/minimax-wsl/runs/cooperative-replay-20261104/foreground-heldout/evaluation.json).

NVTX/CUPTI durations and intersections use one profiler clock. Absolute mapping
to native steady-clock endpoints uses a single marker with unmeasured marker
latency and remains a point estimate. The frozen cost file's generic CPU
boundary description was overbroad: its actual CPU rows use the **isolated
full-bank probe**, not live durations from the held-out request. This text was
corrected after the test without changing its prices or prediction.

The tool exposes `reproduce`, `reconcile-profile`, `calibrate-foreground`,
`simulate --assignments recorded --generation-only`, and `diagnose`. The original
validation gate incorrectly summed nested timing residuals; the
[acceptance correction](#acceptance-correction-and-ordinary-primary-query-test)
supersedes that implementation. This held-out evaluator did not invoke that gate.
The foreground model is still unqualified; the required
copy-contention/intervention gates, admission repair and ranking remain open in
the central plan. Build and focused correctness passes do not close them.

## Policy objective reassessment

**User correction, 2026-11-04; source/data review 2026-11-05 UTC:** adherence to
the plan and a real whole-engine gain do not establish that each policy is a sound
approach to the objective. The [central correction](README.md#policy-evaluation-correction)
reopens admission/scheduling valuation and the backing decision. Earlier completion
judgments are superseded at that scope; raw measurements and valid implementation
checks remain preserved. The primary agent performed this review without model
runs or delegation. [Calculations and source scope](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/policy-reassessment-20261105.json).

The user's clarification preserves cache-aware model routing and requires admission
learning to use original-router demand exclusively. Current
[`observe_routes`](../../src/mm3/mm3_native_demand.inc) credits `d.original_h`,
separately records executed hits and substitutions, and excludes prefill from the
generation score clock. The native profile snapshot uses `original_prof`; helper
admissions read the same scores. No cache-hit credit leak was found in these paths.

The routing bonus 0.10 was numerically tested against 0.05 with a common restored
learned state. The [recorded comparison](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/bonus-comparison-evaluation.json)
gave 14.0219 versus 15.3507 tok/s pooled (+9.48%), with pair effects +11.27% and
+7.67%, Python +16.07% and prose +3.25%. These were two short high-effort tasks,
with differing sampled work and capped outputs. They select a provisional value
within that comparison; they do not establish a global optimum or validate the
admission model. The mechanism and the chosen numeric value are distinct decisions.

### Admission value does not use the foreground objective

The [foreground planner](../../src/mm3/mm3_batch_staging.hpp) prices grouped expert
uses and compares the maximum of overlapping CPU, GPU/copy and helper service.
[`adaptive_service_ms`](../../src/mm3/mm3_native_adaptive.inc) instead returns one
CPU-versus-GPU marginal at column count 1. Primary admission multiplies the
row-frequency score difference by that scalar and a horizon factor, subtracts
transfer prices and configured surcharges, and selects a minimum-score victim.
It does not evaluate grouped reuse, the relevant limiting branch or the working
frame's opportunity cost with the foreground objective. The retained interference
surcharge is zero.

A consistency calculation using the saved backing-comparison calibration makes
the unit mismatch concrete. At CPU scale 1.0, the single-column service difference
is 0.46259 ms. The same curves give 0.48409 ms for four columns served together,
whereas four single-column charges total 1.85037 ms (3.82 times as much). At five
columns, the recorded interpolation gives 0.50336 ms versus 2.31297 ms from five
single-column charges (4.60 times). These are calculations from the model's own
service curves, not measured critical-path savings or proof that every admission
is overvalued by that factor. They establish a structural mismatch that the earlier
model-agreement and reuse-count checks did not test.

### Executed reuse was treated as admission justification

The reconciliation script labels an admission `used` when `generation_uses > 0`;
that count includes cache-induced substitutions. It also records original uses,
but the headline category does not require them. In the selected companion, 58
of 68 evicted admissions had executed uses, while only 51 had executed uses that
also belonged to original-router selections. Seven of the reported used admissions
therefore had substitution-only executed use. The
records verify actual reuse and frame correctness; they do not establish net
benefit after displaced-victim service, rereads, transfers and interference.

The backed companion also records 132 retained admissions evicted without an
executed use, out of 348 evicted retained admissions. This is adverse churn
evidence to investigate, not a causal comparison with another companion's
different history or proof that every unused admission was predictably bad.
The detailed [backed outcomes](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/normal-calibration/backed-companion-reconciliation.json)
and [selected outcomes](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/scale13-companion-reconciliation.json)
remain unchanged.

Removing preservation D2H correctly lowers a physical cost. That must expand the
available choices, including keeping the previous admission rate/allocation; it
does not oblige the policy to spend the freed capacity. Explaining changed text or
work does not justify abandoning backing or declaring this optimization complete.
The precise causes of the observed net slowdown remain unresolved. Correct the
identified value/accounting gaps and validate the resulting decisions with bounded
same-demand analysis and actual execution, under the central plan's work order.

## Requested benchmark with monitoring off

The [monitoring-off evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/monitoring-off/evaluation.json)
records one high-effort Python/prose screen at unchanged selected policy values:
**15.5226 tok/s pooled**, Python **16.0539**, prose **15.0253**. Each response
reached the 128-token cap in reasoning. Total generation is 256 tokens in
16.492107 seconds, including drafting/rejected work and ordinary engine maintenance;
prefill and API latency are recorded separately.

The earlier standalone server still had its NVML hardware sampler. Source inspection
also found that frontend capture mode `off` recorded token receipts. Commit
`a105c56` adds a true frontend monitoring-off path, repairs that capture behavior,
suppresses native intermediate progress/statistics and allows the existing collector
to omit stream tracing. The native build and 16 focused checks pass; model/kernels
and policy values are unchanged. The new native binary SHA-256 is
`94cd6ffcb63407f3f5534cc016136d4b6d23967f342b80691b6285b395376b4c`.

Verification before and after the scored requests found no NVML/profiler/known
monitor processes. Hardware snapshots are empty with monitoring disabled, no
native/frontend/client traces were produced, and each request has only its start
and final generation samples. No runtime CPU/GPU polling was performed. Internal
policy bookkeeping and required CUDA dependencies remain engine operation.

The reload reused the independent helper-first history and five existing conditioning
prompts. New service calibration and sampled/learned state prevent attributing the
change from the previous 14.78 tok/s entirely to monitoring. No paired overhead
claim or extra repeat is made. Server 2093734/engine 2093738 remained resident and
idle with monitoring disabled after this run. The central plan owns current status.

## Conditioned final candidate, 2026-11-04

The exact original M-110 executable SHA-256
`0deefe0da96dd2cf8d0bd9bcc9c12d902d6577b15fd183014a35d854f39e338b`
is preserved and reconciled with its original source. The selected executable is
SHA-256 `4c060020589729e315a4a790118b28f7f1af057a83c7a13d2208597ce384000b`,
from native implementation `ebe02b0`. Later runtime defaults select prior 0.10,
lenience 0.4, four drafts and CPU scale 1.3 with live helper, grouped-cost staging,
primary budget 8/horizon 20 and exclusive raw HOST storage.

The [final whole-engine evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-original/final-comparison-evaluation.json)
was performed by the primary agent. Generation includes all drafting, rejected
work, copies and maintenance after prefill, with no initial suffix discarded.

| Engine | First arm, tok/s | Second arm, tok/s | Pooled tok/s | Python pooled | Prose pooled |
|---|---:|---:|---:|---:|---:|
| Preserved original | 12.5713 | 13.1024 | 12.8314 | 12.3497 | 13.3522 |
| Selected Maya | 16.1732 | 16.0560 | 16.1144 | 16.6072 | 15.6500 |

The selected/original effects are +25.59% pooled, +34.47% Python and +17.21% prose;
both declared pair directions favor the candidate. The compatible candidate arms
were reused after selection as declared before the final original run. Both tasks
are capped at 128 tokens; sampled text and reasoning/final fractions differ.
Actual primary allocation is 438 in the final original arm and 437 in the other
arms, under unchanged configured budgets; the helper holds 741. The last original
arm runs standalone while prior timed arms had Nsight launched but collection off.
No equivalence of that overhead or isolated effect of one extra record is claimed.
Original control movement is 4.22%. Prefill, first-text and API wall times remain
in the evaluation separately from the generation ranking. This is bounded usable
system performance, not a broad workload guarantee or hardware limit.

The final standalone reload exposed calibration variation and prompted one declared
targeted check, with no parameter change. It measured **14.7816 versus 13.1024
tok/s (+12.82%)**, Python **13.2709 versus 12.4264 (+6.80%)**, prose **16.6803 versus
13.8561 (+20.38%)**. Both engines in this check used standalone launches and
438/741/6,117 actual primary/helper/HOST capacities. The
[final-load evaluation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/final-load-evaluation.json)
preserves the lower candidate result. It is not pooled into the earlier table:
final conditioning used live helper and changed automatic service estimates.
Coding evaluated 192 target rows versus the prior 153, with nearly unchanged target
time per evaluated row; sampled text differed. Prose was faster. This is evidence
of usable improvement with workload/state sensitivity, not a stable 25.6% guarantee.
No further ranked repeat followed.

The [final operating check](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/operation-reviewed.json)
retains the capped 32-token profile-free bootstrap, independent helper-first setup,
five fresh-KV requests with continuous learning, ordinary/grouped CPU scale 1.3,
and the single unprofiled post-prefill calibration. The capped bootstrap is not
a quality pass. The [no-override API check](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/default-request.json)
returned READY and verified selected runtime controls. The
[handoff](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-selected/handoff-reviewed.json)
observed server 2085689/engine 2085693 idle at port 8097 with complete ownership,
181.45 GiB HOST bank, 41.30 GiB Windows physical headroom and weights resident.
These are dated observations. Native/default timelines validate with no drops;
these final intervals did not collect GPU/expert detail. Existing same-source
companions retain their declared scope and losses.

Independent original-router history supplies global helper-first initialization;
subsequent fresh-KV requests retain ordinary residency learning. The original has
fixed primary and old helper learning; it cannot restore the new checkpoint.
Selected comparison arms restore one verified logical common state, with transient
staging cleared and physical frame addresses allowed to rotate. Source-bound
[conditioning verification](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/conditioning-verified.json)
and [final original setup](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/final-original/conditioning-verified.json)
preserve the differences.

The [complete-task review](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/quality-reviewed.json)
records two code responses passing five functional cases each, naturally completed
high-effort scheduling with correct 11/12 minima and reviewed proofs, a real raw
file-tool round trip (version 7.4.2, no retry/repair), the facts-only library
announcement and cancellation/continued service. Scheduling used 2,146 tokens;
the announcement used 92. These checks establish bounded usability, not general
quality or preservation of the original sampling distribution. Full native timelines
have no drops; intentional API disconnection remains partial.

The same-source selected [detailed reconciliation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/scale13-companion-reconciliation.json)
checks 3,762 planner calls including prefill; generation retains 237 demanded
uploads in place, serving 2,235 later uses. Of those retained records, 58 were
subsequently evicted after use, 10 evicted unused, 158 remained resident after use
and 11 remained resident unused. The last two groups are censored. Fifty additional
uploads and all outgoing preservation are separate costs. The
[routing audit](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/routing-service-audit.json)
checks 10,317 generation routing rows and 41,268 mixture/scaling entries at recorded
precision. The selected [Nsight joins](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/cpu-price/scale13-companion-nsight-validation.json)
pass for the declared request/layer/round/weight-copy scope on both GPUs, retaining
five incomplete CUPTI events at stop and possible-loss warnings. Complete-task
intervals did not collect GPU/expert detail; that is not inferred from the companion.
The original binary's missing native fields remain explicit in its earlier CUDA
companion, and the baseline backing check's one-copy gap below remains unchanged.

The central [current dispositions](README.md#current-implementation-and-parameter-dispositions)
link and interpret the required value comparisons, adverse shorter-window/backing
results, inconclusive helper comparison, budget tie and source/cost decisions.
Those local gains are not additive and cannot replace this original-engine result.

## Baseline integrated behavior

At the source-bound [baseline launch](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline/launch.json),
four grouped requests and one ordinary tool proposal exercised the new placement,
row accounting, batch planner and retention/refill implementation with live helper
adaptation. The primary agent reconciled the records directly; this is diagnostic
implementation evidence, not a ranked performance comparison.

The [reconciliation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline/behavior-reconciliation.json)
checks 18,354 planner calls, 201,780 original selections and 885 evaluated generation
rows with no reconciliation errors. Generation issued 1,742 distinct demand uploads
(55,484,009,984 bytes), plus 757 primary exchanges. Of those exchanges, 361 reused
the existing uploaded frame; their later uses total 9,096, including 7,227 original
selections. Observed retained-admission outcomes are 206 evicted after use, 12 evicted
unused, 136 still resident after use and seven still resident unused. Additional
refills served 14,668 later uses; eight were evicted unused, while 11 remained
resident unused. Right-censored admissions cannot be classified as wasted solely
because capture ended. Primary copies issued 12,612,897,792 H2D bytes and
24,111,019,264 D2H bytes. At most two exchanges and two reserved staging frames were
pending. One final exchange remained unpublished in the inference capture and was
joined at the explicit setup boundary.

The [restore check](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline/state-restore-verified.json)
changed 295 primary and 130 helper identities, then restored exactly the saved
logical-state hash, including scores, admission credits, service state and generation
row 885 in the same engine. Ordinary staging was cleared at the declared boundary.
The exported checkpoint remains an in-process review receipt, not a portable state.

The [backing allocation attempt](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline/backing-enable/result.json)
was rejected before allocation because 13,918,797,824 backing bytes plus the 16 GiB
reserve exceeded Windows physical availability (29,572,321,280 bytes). This is an
actual admission failure, not proof that backing cannot fit under other host load.
The failed admission remains preserved. After user-assisted host reclamation and
the explicitly requested REA/Claude process cleanup, the
[successful retry](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline/backing-enable-after-reclaim/result.json)
allocated all 437 backing records (13,918,797,824 physical bytes), leaving
17,557,929,984 Windows physical bytes available. The
[bounded backed-path check](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/backing-check-reconciliation.json)
exercised grouped and ordinary callers with 66 evaluated generation rows and 89
primary exchanges, 52 reusing uploaded frames. Native expert D2H remained zero,
as did helper weight traffic. Five-second host samples observed at least
17,423,491,072 physical bytes available, above the unchanged 16 GiB reserve;
these are samples rather than a continuous peak guarantee. Release repatriated
71 canonical HOST records before freeing the backing bank, and the
[common-state restore](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/common-restore-after-backing-verified.json)
reproduced the exact saved logical hash in the same model.

The [Nsight reconciliation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/backing-check-nsight-validation.json)
matches all 4,674 layer calls and observes no expert-sized D2H copies, while normal
activation/result readback continues. CUDA runtime correlation attributes copies
by their submission within each NVTX request, preserving later completion. One
of 4,272 native issued H2D records is absent from Nsight (the ordinary request);
the report also contains possible-event-loss warnings. These remain explicit
partial GPU coverage, not zero dropped events. Native detailed events and final
counters reconcile completely. The
[integrated baseline qualification](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/baseline-integrated-qualification.json)
closes the finite implementation gate with these stated limits, without selecting
a performance winner or claiming complete Maya delivery.

The subsequent [static-helper conditioning](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/static-conditioning/result.json)
reapplied the independently collected global helper-first ranks at an explicit
setup boundary, then retained learning across the declared five fresh-KV requests.
It is separate from the combined live-helper implementation check. The helper
roster remained unchanged and all five requests verified zero helper weight
uploads/downloads, replacements and copy failures. The primary changed 311 of
437 starting identities through normal operation; this count is descriptive, not
a warmth criterion. The [trace reconciliation](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/static-conditioning-reconciliation.json)
passes for 19,038 planner calls and 218,652 original selections over 959 evaluated
generation rows. It verifies 429 retained uploads and 12,002 later uses, including
9,490 original selections. The [operating account](/mnt/wsl/transfer/minimax-wsl/runs/maya-completion-20261104/conditioned-residency/static-conditioning-operation.json)
preserves every round and within-request coverage/traffic trajectories without
discarding an initial suffix. A joined in-process common checkpoint preserves the
resulting published rosters, scores, budgets and service state at generation row
1,844. The extraction returned exactly three factual bullets; the ordinary tool
proposal used the correct shipment ID. Other capped outputs remain incomplete.
Raw Nsight output is saved; diagnostic timing is not a warmed speed ranking.

Detailed capture and Nsight affect timing. Code/prose/reasoning reached their caps
and are not completed-task passes; extraction preserved facts but added an extra
note; the ordinary request emitted the declared location lookup with the correct
shipment ID. The first ordinary payload was rejected before inference because
`spec=1` was combined with lenience; only that unexecuted payload was corrected.

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

### Companion overlap and capture audit

The 2026-11-04 helper companions at source `208d9b2` contain two requests each.
Their four decode phases support the following **trace-scoped diagnostic** ranges,
recomputed from the saved [timeline validation](/home/mdwbr/infernix-viz-ports/trace_dashboard_minimax/VALIDATION.md):

| Measure | Range across the four decode phases |
|---|---:|
| Layer calls attributed to CPU finishing last | 53.94–87.17% |
| Sum of CPU leads over the next finisher / decode wall time | 8.72–15.96% |
| Demand-copy duration / decode wall time | 36.96–49.68% |
| Demand-copy duration overlapping a CPU batch / demand-copy duration | 95.32–99.05% |

The next finisher after CPU is usually the 5090. In static prose, the helper's
lead totals 8.84% of decode versus the CPU's 8.72%. These are branch-finish
attributions with the page's stated treatment of held kernels and close finishes,
not isolated CPU-kernel costs or predicted savings. Accelerating CPU work under a
fixed assignment eventually exposes the next branch; changing residency or the
assignment can change all branches. Neither the CPU-last fraction nor its lead
is a global speedup ceiling. Copy duration overlaps CPU work and must not be added
to it as separate wall time. Both companions are `rankable: false`, use detailed
capture and Nsight software CUDA tracing, and continue different cache histories.
They establish neither warmed operation nor an adaptive-versus-static speed ranking.

**Unproven launch-submission lead:** 96.3% of adaptive and 92.2% of static layer
calls with a host publication match the observed groups-of-11 launch pattern.
In the captured build, `k::submit` only queries the stream under `_WIN32`, and the
grouped caller does not call it between expert submission and the CPU batch.
The page's upper attribution for holds is 2.15–4.55% of decode and includes real
5090 work, including the held kernels themselves. It is not an expected speed gain.
The release mechanism and whether the pattern occurs without Nsight remain
unestablished; the statement that kernels necessarily wait for the next host CUDA
call is too strong. The [timeline method](/home/mdwbr/infernix-viz-ports/trace_dashboard_minimax/README.md)
and [compiled margins](/home/mdwbr/infernix-viz-ports/trace_dashboard_minimax/data/limiters.js)
preserve the definition and exceptions.

**Capture and provenance limits:**

- Score triplets overflowed the 192 MiB detail tier; ownership lists were not the
  dominant payload. Neither companion has per-selection detail for coding
  generation. Prose retains five full generation forwards plus one partial:
  adaptive 283–287 plus 288 (18/57 layers), static 615–619 plus 620 (14/57 layers).
  Adaptive generation begins at 230 and 283, not 229 and 282. Whole-round counters
  and token records remain available, and checked generation kernel/copy windows
  reconcile. Missing per-selection data stays missing, never zero or reconstructed
  from round totals. See [coverage validation](/home/mdwbr/infernix-viz-ports/expert_cache_viz_minimax/VALIDATION.md).
- The captured build continues constructing routes text after the detail buffer
  fills: about 305 microseconds per five-row layer at the median. This inflates
  the companions' host-planning category; it is not ordinary ranked-run planning
  cost. The later capture-buffer repair does not retroactively remove this overhead.
- Both original companion manifests still record `nsight_validation: pending`.
  The page validations provide scoped reconciliation and proposed wording; the
  manifest status is a separate metadata gap, not proof of absent Nsight data or
  of complete per-selection coverage. Preserve their original records and limits.
- The older delivery trace is a separate run. Its own records say legacy policy;
  `validation-runtime.json` was rewritten for a later cost-policy arm. Its
  `comparison-protocol.json` declares `spec_window: 8`, while both actual requests
  set `spec: 5` (four drafts plus anchor). The reason for the stale protocol field
  is not established by these captures. The writer's binary hash is known, but
  its exact source is not pinned. The later v2 build manifest below must not be
  used as source provenance for that older trace.

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
