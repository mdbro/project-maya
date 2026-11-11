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
forwards, at similar kernel counts (84-108 in every fourth forward sampled), while the
primary's stayed 0.016-0.023 ms. Without a profiler, one-job helper service had p10 0.34-0.49 ms and
p90 0.83-1.50 ms, tending to climb from about 0.37 to about 0.7 ms, spike near 1.5 ms
and reset. It tracks forward time within captures (r 0.63-0.95) and generation across
the four same-work arms (r 0.94, four points, descriptive). Forwards over 1.4 times
their capture's median carried 0-14.1% of forward time, and held arms have them too.
Waiting on the helper past the CPU lane is 19-30% of generation.

Low 3090 clocks were already on record. The
[stopped-plan history](history-2026-11-03-stopped-plan.md) found its memory clock mostly
810 MHz in serial decode versus 9,501 MHz under continuous service, and whole-process
samples in `runs/proposal-2/*-clocks.csv` show it mostly in P5, P3 or P8 with memory at
405-5,001 MHz around ordinary runs, while the 5090 reaches P1 at 13,801 MHz. One sample
just after this standalone series matches: P3, 360 MHz core, 5,001 MHz memory. This
study turned monitoring off, so it did not sample clocks during its runs and does not
itself tie each slow spell to a clock state.

**Forcing clocks has no measured result in these records.** On 2026-11-01 an agent ran
`nvidia-smi -i 1 --lock-memory-clocks=9501,9501` from WSL. The driver refused ("The
current user does not have permission to change clocks", exit 4) and `sudo -n` needed a
password ([receipt](/mnt/wsl/transfer/minimax-wsl/runs/proposal-3/clock-lock-probe.json)).
A four-minute user-run test was requested with no answer recorded, and the plan closed
it as not run. Later session records only restate that; on 2026-10-19 Infernix work had
proposed locking clocks without trying it. The block was permission, not reported
support: on driver 617.42, `--lock-memory-clocks-info` gives switching type `Runtime` for
both cards from WSL and from Windows. No lock has been attempted with administrator
rights, so that does not prove it works.

The closest comparison is the 2026-10-31 Windows helper validation, recorded in the
Windows checkout's `docs/MINIMAX_HELPER.md` and never in this repository. With identical
generated tokens, median speeds were 7.63 and 7.49 tok/s for the two worker variants that
did not queue the next GPU request and 8.95 tok/s for the queued worker (original copy
path 7.90). A non-queued worker was observed with the 3090 in P5 at 615 MHz core and
810 MHz memory, and the queued worker in P2 at 1,965-1,980 and 9,501 MHz. Queueing also
changes the hand-off itself, so this does not isolate clocks. It used an older engine.

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

### Corrected reading of the diagnostic study

**2026-11-07, user review and primary-agent recheck.** The campaign stands as completed
and its receipts are unchanged. It supports a compact shared-model hypothesis. It does
not demonstrate that these inputs are sufficient. Four statements above went further
than their evidence:

- **The 1.43-1.56% between-prompt figure is not accuracy against actual duration.** That
  calculation also replaces the target's work groups with their medians. Against actual
  elapsed forward totals, the predeclared comparison missed by -8.9% and +10.3%.
- **The 0.014 ms agreement uses already observed branch times.** It supports `max(CPU,
  helper)` as a composition hypothesis, not prediction from expert counts. Composing from
  the other prompt's fitted lane values and expert counts misses cycle totals by -13.5%
  and -6.5% with medians, -11.2% and +3.7% with helper means, and -4.9% and -4.7% when given
  the target's own helper means, so medians also understate the tails of the other parts.
- **The 0.009% recorder figure does not clear monitoring.** It is an isolated microbenchmark
  of selected recording operations. It does not measure effects on engine scheduling and it
  omits state collection. The fixture discrepancy stays unresolved until matched.
- **The helper slowdown is localized, not explained.** Slow periods sit in the helper's
  GPU interval while CPU, join wait and submission stay level (below). A single clock sample
  after the runs is a lead. Whether recording contributes also remains unconfirmed.

**The flat-wait argument, rechecked by pairs.** Bucketed medians can hide changing mixtures,
so this compares the same `(forward, layer)` with identical work across same-work arms. When
a pass's lanes ran over 0.5 ms longer in one arm, the following join wait moved by a median
-0.022 to +0.011 ms, with slopes of -0.001 to -0.022 across seven arm pairs. The primary path
does not overlap the lanes in these passes at `T=1`. This is still ordering evidence from
host boundaries; the trace remains the only direct record of GPU start and end times.

**Per-invocation consequences.** `tools/mm3/native_passes.py` keeps each pass's timeline from
its own join and recomputes each pass's maximum before aggregating. These are virtual changes
to observed passes, as a percentage of generation, and are not validated causally:

| Capture | CPU -10% | CPU -50% | Helper -10% | Helper -50% | Helper at its capture's p10 | No demand copies |
|---|---:|---:|---:|---:|---:|---:|
| Long coding | 1.20 | 5.02 | 3.60 | 15.56 | 17.46 | 14.84 |
| Long prose | 0.85 | 3.37 | 4.10 | 17.49 | 13.68 | 9.89 |
| Fixture arms, range | 1.20-1.94 | 5.15-7.75 | 2.74-3.56 | 10.22-14.85 | 8.30-14.13 | 16.90-22.85 |

The mean of per-pass maxima exceeds the maximum of mean completions by 5.1-17.2%, so averages
would understate these lanes. In passes using both lanes, the CPU is last for only 4-10.5% of
generation time. Hiding copies inside their own pass's lanes is bounded by 9.9-22.8%, assuming they could start
when submission returns.

**Residuals by operating period.** Forwards are labelled by their median one-job helper GPU
time. Against the other prompt's shared table, coding residuals sum to -1.12% of generation
in fast forwards, +1.84% in middle and +9.61% in slow; prose to -1.21%, +2.70% and +2.29%. CPU
one-job time, join wait, planning and helper hand-off stay within 0.04 ms across periods.
Layer identity is still not needed for this.
[Script and output](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-followup-20261107/interpretation/).

## Shared-model decision validation, speculation off

**Result, 2026-11-07: not qualified under its predeclared rule. Decision classes held, and the one failed
check traces to whole-engine stalls.** Primary-agent work. Configuration frozen first: query on, gap
spinner on, speculation off, prefill as setup. Fixture arms replay fixed executed experts, 240-241 forwards
per run. Predictions froze with a [digest](shared-model-frozen-predictions.json) before each round's arms ran.
[Protocol](shared-model-validation-protocol.json), rule in `tools/mm3/native_shared_model.py`,
[receipts](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-20261107/).

| Check, limit | Round 1 | Round 2, repaired model, untouched workloads |
|---|---:|---:|
| Reversals, none | None | None |
| Class agreement over conclusive cells, at least 80% | 66.7%, **failed** | 85.7% |
| Non-tie cells confirmed, at least 2 | 4 | 6 |
| Median difference error, at most 1.5 points | 0.65 | 0.96 |
| 90th-percentile difference error, at most 3 points | 8.36, **failed** | 4.36, **failed** |
| Median control miss, at most 5% | 0.37% | 0.67% |
| Qualified | No | **No** |

The rule allowed one repair and ends qualification on a failed round 2. No further repair of the fixture
rule follows, and no parameter is selected from these predictions alone.

Round 2, percent of complete generation against same-block control, three blocks each:

| Cell | Predicted | Observed | 95% interval | Classes |
|---|---:|---:|---:|---|
| Extract, CPU price 1.0 | -0.24 | +2.58 | +0.42, +4.73 | Tie, worse |
| Extract, CPU price 2.0 | +3.12 | +3.79 | +1.63, +5.94 | Worse, worse |
| Extract, no staging | +8.24 | +9.18 | +7.02, +11.33 | Worse, worse |
| Extract, hold primary admissions | +8.19 | +12.55 | +10.40, +14.71 | Worse, worse |
| Extract, hold helper admissions | +0.05 | -0.33 | -2.48, +1.83 | Tie, inconclusive |
| Reasoning, CPU price 1.0 | -0.05 | +1.53 | -0.63, +3.68 | Tie, inconclusive |
| Reasoning, CPU price 2.0 | +4.41 | +5.39 | +3.23, +7.55 | Worse, worse |
| Reasoning, no staging | +9.58 | +10.69 | +8.54, +12.85 | Worse, worse |
| Reasoning, hold primary admissions | +19.82 | +20.00 | +17.84, +22.15 | Worse, worse |
| Reasoning, hold helper admissions | +0.55 | +0.12 | -2.04, +2.27 | Tie, inconclusive |

**What failed.** Round 1 missed CPU price 2.0 on code by 8.36 points. Replayed work was within about 0.5% of
recordings in generation, so the fault was timing: a copy slows a CPU job only while they overlap, and a second
copy outlasts the lanes. That repair is the shared rate pair, 0.61 for CPU and 0.67 for copies during overlap. In
round 2 the two largest errors sit on arms with whole-engine stalls. Clipping each arm's forwards at 1.4 times
its own median, descriptive only, gives a median error of 0.73 points and a largest of 1.46 across all ten cells.
Clipping is not part of the rule and does not change the verdict: complete totals are the measure, and stalls are
part of what a user waits through.

**Complete requests with real routing,** two orders, frozen fixed-demand prediction against observed. Each is
predicted a tie, and the observed means agree within noise:

| Cell | Predicted | Pair effects | Mean |
|---|---:|---:|---:|
| Code, CPU price 1.0 | +0.09 | -0.34, +2.27 | +0.96 |
| Prose, CPU price 1.0 | -0.01 | -0.73, +0.03 | -0.35 |
| Code, helper live off | +0.78 | +1.33, -0.13 | +0.60 |
| Prose, helper live off | +0.21 | -0.84, +4.75 (one stalled run) | +1.95 |

**Order rule.** The rule fixes time in layer joins: a primary D2H takes two joins, because the copy stream starts
at the next boundary's query and finishes by the one after; a prefix counts as two joins, so forward-end work
settles within it. It reads nothing from the control's future. On training captures it matches exchange counts
within 3 of 611, the split of forward-end and layer-boundary issues within 2, reuse within 2, and pending exchange
windows 489 against 447. Pass-level work was identical in 87.6-95.8% of passes and within about 1% in aggregate.

**The model,** `tools/mm3/shared_model/timing.py`, shared across all MoE layers. Fitted values in ms: CPU 0.030 +
0.545 per job; helper completion by job count from draws, means 0.299, 0.327, 0.355, 0.376; primary path 0.240 +
0.553 per copy; wait 0.274; prepare 0.067 and 0.104; resume 0.105; pending exchange +0.08 and open lease +0.85 per
window; final layer +1.35; prefix 1.55 + 1.11 per forward-end exchange. These associations hold in this domain:
`T=1`, query and spinner on, eight staging frames, these prompts. On the training captures' own recorded work it
misses forward time by -0.65 to +1.14%, apart from one run with a stall.

**Policy readings at this scope, confirmed on hardware:** exchanges are valuable, holding them costs 12-20% across
the four workloads. Removing demand staging costs 8-11% and raising the CPU price to 2.0 costs 3-9%. CPU price 1.0 and
holding helper leases are not resolved from ties. Run blocks are in the
[round 1](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-20261107/evaluation-round-1.json) and
[round 2](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-20261107/evaluation-round-2.json) evaluations.

## Speculative generation: configuration gate and grouped model

**Result, 2026-11-07: the spinner carries over and the query does not. The grouped model is not qualified under
its predeclared rule either.** Primary-agent work. Selected spec 5, lenience 0.4, four 2 by 2 Williams-square blocks per
prompt, 32 complete 384-token requests with full records. [Gate protocol](speculative-gate-protocol.json),
[validation protocol](speculative-validation-protocol.json),
[gate receipts](/mnt/wsl/transfer/minimax-wsl/runs/speculative-20261107/),
[validation receipts](/mnt/wsl/transfer/minimax-wsl/runs/speculative-validation-20261107/).

**Gate.** Rule: at least -2%, 95% interval excluding zero, neither prompt slower.

| Control | Main effect | 95% interval | Prose | Code | Decision |
|---|---:|---:|---:|---:|---|
| Gap spinner | -5.37% | -7.36, -3.38 | -7.07% | -3.67% | **Selected** |
| Submission query | -0.70% | -2.60, +1.20 | -0.01% | -1.38% | Not selected |
| Both, over neither | -6.07% | -7.86, -4.29 | | | |

| Tokens per second | Neither | Query | Spinner | Both |
|---|---:|---:|---:|---:|
| Prose | 15.78 | 15.64 | 16.82 | 17.00 |
| Code | 15.47 | 15.92 | 16.30 | 16.29 |

The `T=1` gain does not transfer because a window of two or more rows reaches the launch group by itself. Per pass
the query only helps at one row, -0.058 ms, and -0.262 ms with a copy. Forwards with one row are about 7% of prose
time and absent from code, so a one-row query could move little. The spinner again holds the 3090 at P2, 9,501 MHz
memory and a median 1,935 MHz in all 3,709 samples. Without it, 97% of samples sit at P3 with 5,001 MHz memory, median 645 MHz.
[Breakdown](/mnt/wsl/transfer/minimax-wsl/runs/speculative-20261107/spec_breakdown.txt),
[by rows and query](/mnt/wsl/transfer/minimax-wsl/runs/speculative-20261107/spec_rows_query.txt).

**Where the time goes, spinner arm, mean per pass.**

| Rows | Passes | Cycle ms | Lanes ms | CPU ms | Wait ms | CPU jobs | Helper jobs | Primary jobs | New copies |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 4,275 | 1.256 | 0.538 | 0.352 | 0.493 | 0.60 | 1.58 | 1.82 | 0.29 |
| 2 | 9,519 | 1.552 | 0.902 | 0.768 | 0.367 | 1.02 | 2.62 | 2.64 | 0.56 |
| 3 | 9,177 | 1.980 | 1.286 | 1.165 | 0.352 | 1.48 | 3.32 | 3.54 | 0.98 |
| 4 | 9,576 | 2.418 | 1.641 | 1.538 | 0.375 | 1.91 | 4.05 | 4.35 | 1.41 |
| 5 | 30,267 | 3.406 | 2.481 | 2.393 | 0.432 | 2.96 | 4.53 | 5.32 | 2.28 |

The CPU lane sets the pace once windows are wide. A 5-row lane runs 0.89, 1.33, 2.12, 2.94 and 4.24 ms with zero to
four new copies, while the wait stays near 0.43 ms. Time outside forwards is 14-19 ms per round, of which drafting
is about 9.7 ms. Acceptance ranged 55-76% and tokens per round 2.3-3.8. The primary bank is full, 980 MB free.

**Grouped model,** `tools/mm3/shared_model/grouped.py`, the same composition with fixed parts by rows. A window of two or
more rows overlaps the lanes; one row is sequential unless queried. Rows of a rejected draft stay in the recorded
window, so a candidate pays for them as the control did. Prefill stays setup.

**Validation against the same rule as the ordinary model.** Fixture arms replay fixed executed experts, three blocks
per cell. Predictions froze with a [digest](shared-model-frozen-predictions.json) before each round's arms ran.

| Check, limit | Round 1, code and prose | Round 2, repaired model, untouched extraction and reasoning |
|---|---:|---:|
| Reversals, none | None | None |
| Class agreement over conclusive cells, at least 80% | 100%, of 5 | 77.8%, of 9, **failed** |
| Non-tie cells confirmed, at least 2 | 5 | 7 |
| Median difference error, at most 1.5 points | 0.85 | 1.53, **failed** |
| 90th-percentile difference error, at most 3 points | 8.67, **failed** | 4.32, **failed** |
| Median control miss, at most 5% | 1.13% | 1.92% |
| Qualified | No | **No** |

The repair between rounds separated the CPU price per job from the copy stretch, using round 1's arms in training,
with no change of form. No clipping was applied and no further repair follows.

| Cell, percent of generation against same-block control | Round 1 predicted | Observed | Round 2 predicted | Observed, untouched |
|---|---:|---:|---:|---:|
| CPU price 1.0, code or extraction | +1.17 | -0.04 | +2.50 | +2.12 |
| CPU price 1.0, prose or reasoning | +0.72 | +1.31 | +1.47 | +2.94 |
| CPU price 2.0, code or extraction | +4.24 | +2.53 | +4.56 | +6.15 |
| CPU price 2.0, prose or reasoning | +5.23 | +5.06 | +5.12 | +6.11 |
| No staging, code or extraction | +23.69 | +32.36 | +22.25 | +25.06 |
| No staging, prose or reasoning | +10.44 | +15.10 | +31.97 | +35.95 |
| Hold primary admissions, code or extraction | +13.90 | +13.51 | +14.10 | +15.77 |
| Hold primary admissions, prose or reasoning | +11.25 | +11.28 | +36.39 | +40.71 |
| Hold helper admissions, code or extraction | +0.88 | +0.51 | +1.74 | +2.49 |
| Hold helper admissions, prose or reasoning | +1.75 | +0.64 | +2.65 | +2.70 |

Round 2's errors nearly all have one sign: the model underpredicts how much worse a damaging arm is. It gets the order
right and the size short, and its misses are largest on the arms that matter least, because they are far from any
candidate. The eight cells predicted under 3 points, the band decisions depend on, miss by at most 1.47 points, inside
one cell's half-width of about 2.4 to 3.4 points. This does not qualify it: the declared rule failed on three checks.

**Policy readings confirmed on hardware.** None of these arms beats the incumbent on any of the four workloads:
holding primary admissions costs +11.28 to +40.71%, removing staging +15.10 to +35.95%, CPU price 2.0 +2.53 to
+6.15%, CPU price 1.0 -0.04 to +2.94%, and holding helper admissions +0.51 to +2.70%.

## Per-pass balance

**User direction, 2026-11-07: each individual activation pass through each layer is where balance is judged.**
[Plan](README.md#per-pass-balance). First reading, primary-agent work, on the eight spinner-arm requests of the
speculative gate, 61,712 passes, bonus 0.10. [Receipts](/mnt/wsl/transfer/minimax-wsl/runs/integrated-screens-20261107/balance_records.txt),
[per-miss tables](/mnt/wsl/transfer/minimax-wsl/runs/integrated-screens-20261107/miss_law.txt). CPU and helper ends are
recorded per pass. Copy and kernel ends are not in these records, so the primary path's slack is still to be read from a trace.

**CPU against helper: far from balance, on almost every pass.** From each pass's own origin:

| Rows | Prompt | Passes running both lanes | CPU last | Helper last | CPU end minus helper end, p10, p50, p90 ms |
|---:|---|---:|---:|---:|---:|
| 5 | Prose | 11,008 | 85.7% | 10.4% | -0.07, +1.03, +2.41 |
| 5 | Code | 17,649 | 96.0% | 3.0% | +0.55, +2.10, +3.99 |
| 4 | Prose | 5,846 | 86.0% | 9.6% | -0.04, +0.78, +1.98 |
| 4 | Code | 2,949 | 93.3% | 4.8% | +0.17, +1.33, +2.75 |

No planner move can use that slack. `plan_batch_staging` only moves a job from the CPU to the primary, and a demand upload
to the 3090 would not fit: its link runs at PCIe Gen4 x4 and a record took 4.9 ms in the `T=1` trace, against 0.55 ms
on the 5090's Gen5 x16. Experts already resident on the helper are the only way onto it, which makes routing the mechanism
that reaches this slack.

**CPU against uploads: the planner divides misses evenly, and each miss costs the same wherever it goes.** Every pass with
`m` misses holds the same division: (1,1) for 2, (2,2) for 4, (3,3) for 6, (4,4) for 8 in thousands of passes each, so
records alone cannot compare divisions. Median cycle, calm 5-row passes:

| Misses | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ms | 1.54 | 1.58 | 1.74 | 2.17 | 2.57 | 3.05 | 3.44 | 3.93 | 4.30 | 4.80 | 5.35 | 5.87 | 6.41 | 6.96 | 7.52 |

Fits over 2 to 12 misses give about 0.44 ms per miss at every width: 0.446 at 2 rows, 0.436 at 3, 0.433 at 4, 0.451 at 5. Past
8 misses the steps are about 0.55 ms, because the upload limit is 4 and each further miss stays on the CPU. A CPU job or a
copy alone reads a 32 MB record in about 0.55 ms, 58 GB/s. Together they share at about 0.61 and 0.70 of that rate, about
76 GB/s in total. The host has four 64 GB DDR5 modules on two channels at 6000 MT/s, read through the host-control task on
2026-11-07: 96 GB/s in theory, so 76 GB/s is about 80% of it, a typical sustained read ceiling. That shared bandwidth is
the likely limit, an inference from the overlap rates rather than a bandwidth measurement. If so, 32 MB at 76 GB/s is
0.42 ms per miss, close to the fit.

Which means **passes already sit near balance between these two paths, and a pass is slow because of how many misses it has.**
Three checks agree:

| Check | Result | Scope |
|---|---|---|
| Same-pass division bound: for each pass, the number of uploads that finishes it soonest | 0.08% of cycle time at the limit of 4, 0.51% with up to 8, at most 1.18% on code. No single move helps on over 98% of passes | Unqualified model, fixed demand, later reuse not counted |
| Hardware arms that change the division | CPU price 2.0: +2.53 to +6.15%. CPU price 1.0: -0.04 to +2.94%. No staging: +15.10 to +35.95%. None helps | Fixed work, four workloads |
| More frames and uploads, fixed demand | 16 frames with 8 uploads -0.69% mean, 32 with 8 -1.32%, -0.12 to -2.29% per workload | Unqualified model; routing's response to more ready frames is not counted |

Still open: the primary path's end times come from the model here. An Nsight trace of wide windows with the spinner on will check them.

**Redone at `T=1`, user direction, 2026-11-07.** [Plan](README.md#per-pass-redo). Ten complete 384-token requests with full records,
speculation off, query and spinner on, bonus 0.10: 207,295 one-row passes outside the first and last layer.
[Tables](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-20261107/per-pass-balance/t1_balance.txt),
[placement bound](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-20261107/per-pass-balance/placement_bound.txt).

| Finding, per pass from its own origin | `T=1` |
|---|---|
| Passes running both CPU and helper | 50.7%; CPU last on 99.0%, by a median 0.37 ms, p10 0.22, p90 0.74 |
| CPU only, helper only, neither | 16.3%, 31.4%, 1.6% |
| Wait after the lanes | 0.27-0.31 ms median at every miss count, p90 at most 0.44: primary work rarely outlasts the lanes |
| Median cycle by misses, 0 to 4 | 0.840, 1.061, 1.418, 1.912, 2.236 ms: steps of 0.22, 0.36, 0.49 and 0.32 |
| Division of misses | Always the same per count: one miss mostly to the CPU, 1.045 ms, but 13,858 passes uploaded it at 1.108 ms; two as (1,1); three as (2,1); four as (2,2) |
| Layers | Mean misses per pass 0.76 at layer 40 to 1.65 at layer 30, two or more misses on 17-53% of passes. One shared curve fits every layer: median residual per layer -0.012 to +0.013 ms at p10 and p90, largest +0.054 |
| Placement, per-request oracle on original demand | Placing the same residents to minimise summed pass time instead of maximising hits saves 0.18% at 1,178 records, 0.08% at 437, 0.07% at 300 |

**Reading.** At `T=1` the division between the CPU and copy-plus-primary is near balance, apart from single misses that were
uploaded rather than run on the CPU, worth about 0.35% at most. Frequency placement is within 0.2% of per-pass-optimal
placement. The bound is far below the 2% rule, so valuing residents by per-pass finish time is not pursued at `T=1`. The
shared model's no-layer-parameters decision holds at pass level. The imbalance that remains is the helper's slack: the CPU
finishes last on almost every two-lane pass, and only routing can move work onto experts the helper already holds.

**Admission reading corrected.** A miss removed saves about 0.44 ms in a loaded pass, against the 0.58 ms admission believes.
The `repaired` arm's single lower price was still wrong in kind: a pass with few misses saves less than one with many, and
one price cannot say so. On hardware it lost, below. Fixed-demand screens predicted ties for every admission arm and hardware
moved by -4.4 to +7.8%, so they cannot see routing's response to residency. They should not be used to rank admission or routing.

**Path completion per pass with the spinner on, trace, 2026-11-07.** [Protocol](per-pass-trace-protocol.json),
[receipts](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-20261107/), per-layer table in
[path-metrics-by-layer.txt](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-20261107/path-metrics-by-layer.txt). One
Nsight-launched load, current binary, recorded work replayed for 64 generation forwards: `T=1` with the query, and
speculative wide windows. Completion times from each pass's common origin. Detailed capture and Nsight inflate durations,
so ordering and completion order are the evidence. Mean abs(Δ) in ms, all passes with both paths, then the range of per-layer means:

| Pair | `T=1` prose | `T=1` code | Wide prose | Wide code |
|---|---:|---:|---:|---:|
| CPU vs primary GPU transfer | 0.178 (0.080-0.312) | 0.222 (0.120-0.391) | 0.323 (0.235-1.568) | 0.253 (0.199-1.437) |
| Helper GPU transfer vs CPU | 0.441 (0.268-0.978) | 0.572 (0.360-0.883) | 0.883 (0.386-1.744) | 2.676 (1.442-4.947) |
| Primary vs helper GPU transfer | 0.684 (0.573-1.211) | 0.676 (0.601-0.873) | 1.282 (0.743-1.857) | 2.853 (0.749-5.076) |
| CPU vs helper resident compute | 0.450 | 0.582 | 0.917 | 2.714 |
| Primary GPU transfer vs helper resident compute | 0.694 | 0.685 | 1.336 | 2.891 |
| Helper transfer vs helper resident compute | 0.010 | 0.010 | 0.044 | 0.039 |
| Slowest vs second-slowest path | 0.231 (0.170-0.742) | 0.239 (0.182-0.500) | 0.266 (0.214-0.561) | 0.238 (0.199-0.507) |
| Slowest path | CPU 49%, helper 27%, primary 23% | CPU 54%, primary 35%, helper 11% | primary 81%, CPU 13%, helper 6% | primary 91%, CPU 8% |

**Open correction, user review 2026-11-07: dependency-unaware primary end.** In `trace-spec-prose`, layer 37, a pass marked
primary-last shows a long gap between the bulk of its 5090 work, right after the demand copies, and a small final piece of
5090 work. The window submission ends with a `wait_publication` kernel that spins until the CPU publishes its parts, then the
combine. A final piece gated that way belongs to the CPU's finish. The table above takes the primary end from the last
kernel it labels as expert work, without splitting at the wait. Until each pass is split at the wait and recomputed
(`trace-spec-*`, `trace-t1-*` and the follow-up's `trace-q-on`), treat these as unconfirmed: the primary-last shares, and
the CPU vs primary abs(Δ) values, which could read too small if a gated kernel pins the primary end just after the CPU's.
The records-based evidence below does not depend on this: the flat post-lane wait, the hardware division arms and the per-miss law.

**Withdrawn at `T=1`, kept in wide windows, 2026-11-07 evening.** The rule below also caught the
[copy-to-kernel handoff](#copy-to-kernel-handoff): the first kernel after a demand copy starts a fixed 0.057 ms after the
copy when the CPU has already finished, whatever the host is doing. Allowing 0.10 ms for it after the last copy flags
**no** `T=1` pass as held, against 18.9% and 26.3% before. In wide windows 38.5% (prose) and 77.3% (code) of passes stay flagged,
carrying 84% and 99% of the summed delay: those are gaps between kernels, as in the layer-37 pass, not the handoff
([split](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-20261107/held_without_handoff.txt)). The `T=1` rows of the table below
are withdrawn; the uncorrected `T=1` values above stand. The wide rows stand, with the tracing caveat after the table.

**Correction applied, 2026-11-07: held launches.** The layer-37 pass, forward 487, did not wait on the publication itself:
- Its two demand copies end at 1.567 ms, and four and a half experts' kernels run back to back until 1.901 ms.
- The last five launches start only at 2.293 ms: the fifth expert's two projections and scatter, the publication wait and the combine.
- That is after the CPU batch ends at 2.139 ms and the host join at 2.216 ms.

The window issues five launches per primary expert plus two, and the driver passes them to the GPU in groups. The remainder
waited for the host's next CUDA call, at the join, after its own CPU work. So the pass is CPU-last, and the 5090's own work had
0.24 ms of slack. `native_pass_balance.held_split` now treats a primary kernel as held when it starts after the CPU batch and
after more than 0.05 ms of idle time the stream did not need. Held expert kernels are replayed back to back, and the wait
and combine are left out of the primary's own work.
[Recomputed](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-20261107/path_metrics_held.txt), with the
[per-layer table](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-20261107/path-metrics-held-by-layer.txt):

| Trace | Passes with held launches | Primary end delayed, median and p90 | CPU vs primary transfer abs(Δ), before and after | Slowest path, corrected |
|---|---:|---:|---:|---|
| `T=1` prose | 18.9% | 0.072, 0.083 ms | 0.178, 0.136 ms | CPU 52%, helper 28%, primary 21% |
| `T=1` code | 26.3% | 0.073, 0.083 ms | 0.222, 0.192 ms | CPU 58%, primary 32%, helper 11% |
| Wide prose | 69.0% | 0.273, 0.693 ms | 0.323, 0.245 ms | primary 45%, CPU 43%, helper 12% |
| Wide code | 79.7% | 0.623, 2.005 ms | 0.253, **0.610** ms | **CPU 72%**, primary 26%, helper 2% |
| `T=1` prose before the spinner, `trace-q-on` | 21.2% | 0.075, 0.087 ms | 0.135 ms | helper 50%, CPU 26%, primary 24% |

Recomputed with the handoff allowed, `held_split` v2
([metrics](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-20261107/path_metrics_held_v2.txt),
[per layer](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-20261107/path-metrics-held-v2-by-layer.txt)):

| Trace | Passes held | Delay, median and p90 | CPU vs primary transfer abs(Δ) | Slowest path |
|---|---:|---:|---:|---|
| `T=1` prose, code, before the spinner | 0% | - | 0.178, 0.222, 0.181 ms, as uncorrected | as uncorrected |
| Wide prose | 38.5% | 0.459, 0.787 ms | 0.310 ms | primary 46%, CPU 43%, helper 12% |
| Wide code | 77.3% | 0.638, 2.036 ms | 0.614 ms | CPU 72%, primary 26%, helper 2% |

On hardware the speculative query left the wait after the lanes unchanged on 5-row passes, so the held tail itself may
mostly come from tracing. The primary's own early finish is what matters for balance. By uploads per pass in wide code,
corrected, the CPU minus the primary's own end is:

| Uploads | Passes | p10 | p50 | p90 | Reading |
|---:|---:|---:|---:|---:|---|
| 0 | 82 | +0.22 | +0.36 | +0.59 ms | One CPU job against resident-only work; only a hit closes it |
| 1-3 | 2,082 | about -0.2 | +0.2 to +0.3 | +0.5 to +0.6 ms | Near balance |
| 4, the cap | 1,484 | -0.11 | **+0.65** | **+2.41 ms** | The cap is half the 8 staging frames. Beyond it the CPU keeps the excess |

Prose with uploads sits near balance, the primary slightly later. At the final bonus of 0.20, full records from the last
load show the cap binding on 13.4% of code passes (22.9% of cycle time) and 1.1% of prose passes, against 39.6% and 6.5% at
0.10. That leaves 360 CPU jobs beyond the cap per code request, against 2,176 at 0.10
([counts](/mnt/wsl/transfer/minimax-wsl/runs/final-comparison-20261107/cap_binding.txt)). At about 0.2-0.3 ms each, more staging
frames could gain about 0.4-0.6% on code, under the 2% rule, so they are not tested now.

**A balanced pass, user review: `trace-t1-prose`, forward 339, layer 33.** Times in ms from the start of the CPU and helper work:
- **CPU:** two experts, 0.000-0.955.
- **5090:** one demand copy, 0.023-0.890; its expert kernels 0.965-1.037; the publication wait 1.039-1.041; the combine 1.043-1.048.
- **3090:** one expert, kernels 0.217-0.301.

The planner had predicted CPU 1.341, 5090 0.785 and 3090 0.220 ms and staged one upload. In fact the CPU and the 5090's own work
finished together, at 0.955 and 0.952 ms. The flat 1.3 on CPU work and the uncontended copy price erred in opposite directions
and cancelled. A second upload would have run the 5090 path to about 1.7 ms, since copies run back to back and slow while the CPU
reads. The kernels after the copy started 0.075 ms after it ended, just after the CPU finished. **Corrected the same evening:**
that is the [copy-to-kernel handoff](#copy-to-kernel-handoff), not a hold until the host's next call. The host had issued those
kernels about 0.9 ms earlier, the gap is the same when the CPU finished first, and a polling host does not shorten it. The
"held on 19-26% of passes" reading is withdrawn; every upload pays the handoff.

**Splitting an expert between CPU and 5090: not pursued.** With the measured 75.9 GB/s ceiling, each 32 MB miss cannot finish
sooner than 0.442 ms of shared reading. A pass whose CPU lane ends after `misses × 0.442 + 0.05` ms, and after the helper, loses
the difference to whole-expert granularity. The bound counts only passes with both a CPU job and an upload, so a split is possible:
- final configuration, bonus 0.20, speculative: 13.5% of passes, **0.62%** of cycle time;
- `T=1` with query and spinner: 9.6% of passes, **0.49%**.

Both are under the 2% rule before counting a split's own costs.
[Bound](/mnt/wsl/transfer/minimax-wsl/runs/final-comparison-20261107/split_bound.txt).

**Readings.**
- Corrected for held launches, CPU and copy-plus-primary are near balance where a pass has uploads below the cap. They are not
  near balance on passes at the upload cap, which bonus 0.20 mostly removed, or on passes with a single CPU job and no upload. The helper finishes far earlier: 0.4-0.6 ms at `T=1`, and 0.9-2.7 ms in wide windows, where on code it ends at a median
  0.43 ms against the CPU's 3.18 ms at 5 rows.
- **Layer 4 stands out, from optional exchanges.** In wide windows, exchange copies overlap layer 4's passes for 2.0-3.0 ms on
  59-91% of them. Its CPU vs primary gap is 1.50-1.52 ms against 0.23-0.29 ms on layers 11-59. At `T=1` the same
  happens at layer 5: 0.70 ms of overlap on 20% of passes and a 1.37 ms span against about 0.7 ms. These are admissions
  issued at forward end landing on the next forward's first MoE layers: a per-layer imbalance the policy controls.
- **Odd window widths in the trace.** With 1, 3 or 5 rows the primary path is last on every pass, and with 2 or 4 rows on
  32-42%. That fits the launch grouping, 5×rows+2 launches against groups of 11. Uninstrumented work does not show it: on
  hardware the query did not help 3- or 5-row passes (2.236 against 2.217 ms and 3.425 against 3.401 with a copy), and
  per-pass base cost rises smoothly with rows in the records. It is treated as a trace effect, and odd-row trace times
  are not used for decisions. One-row windows are the exception: there the records agree that copies wait.

### Copy-to-kernel handoff

**Found 2026-11-07 evening in the traces, then measured alone.** In all four traces the 5090's stream is idle when a pass's last
demand copy ends, and the first dependent kernel starts a median 0.057-0.078 ms later. The gap does not follow the CPU: it is
0.057 ms when the CPU had finished up to 0.3 ms before the copy, and 0.061-0.073 ms while the CPU is still working. The host had issued
every one of those kernels 0.88-0.99 ms before the copy ended (`T=1`, p10-p90), and on most passes made other CUDA calls inside
the gap. Demand copies share the kernels' stream.
[Gap by CPU end](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-20261107/copy_kernel_gap_all.txt),
[launch times](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-20261107/launch_vs_copy.txt).

`tools/mm3/native_handoff_probe.cu`, beside the idle final engine, HAGS off. One expert-sized record (31,850,752 bytes) per
iteration, 300 iterations, CUPTI timestamps, median gap from the end of the upload to the start of the next kernel
([result](/mnt/wsl/transfer/minimax-wsl/runs/handoff-probe-20261107/hags-off.txt)):

| Upload | 5090 | 3090 |
|---|---:|---:|
| Copy and kernels on one stream, as the engine | 44 µs | 88 µs |
| Copy stream and an event | 48 µs | 88 µs |
| Host spinning `cudaStreamQuery` | 46 µs | 89 µs |
| Host doing CPU work, no CUDA calls | **56 µs** | 89 µs |
| Kernel reading mapped pinned memory, no copy engine | **2.3 µs** | **2.7 µs** |
| Upload rate, copy engine and mapped kernel | 57.8 and 53.6 GB/s | 6.4 and 6.4 GB/s |

- **The cost belongs to large copies.** A 4 KB readback hands off to the next kernel in 2.5 µs. A busy host thread adds about
  12 µs on the 5090, matching the engine's 0.057-0.065 ms.
- **A polling host thread does not shorten it**, 46 against 44 µs, and the engine's host already calls in during the gap. The
  polling-thread flush is not built.
- **An SM-driven upload removes the handoff but uploads 8% slower on the 5090**, 0.595 against 0.551 ms: a wash there
  before contention. On the 3090 both run at the link's 6.4 GB/s, a clean gain, but its large copies are refills off a pass's path.
- **Bound.** Removing the handoff from every traced pass and recomputing each pass's slowest lane saves 0.81% (`T=1` prose),
  1.05% (`T=1` code), 1.31% and 1.28% (wide) of traced forward time, or 0.59-0.97% if 20 µs remain. That is under the 2% rule
  on its own. In wide windows CPU vs primary transfer abs(Δ) narrows from 0.323 to 0.273 ms (prose) and 0.253 to 0.194 (code)
  ([bound](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-20261107/handoff_bound.txt)).
- **Hardware-accelerated GPU scheduling** moves this dependency into the GPU's own scheduler. It also changes submission in
  general, which mattered more before: one stream query cut `T=1` generation 15%. The study, with its rule fixed before
  measuring, is in the [protocol generator](/mnt/wsl/transfer/minimax-wsl/runs/hags-20261107/make_protocols.py).

**With HAGS on, after the reboot, same probe and binary.** Medians, HAGS off → on
([result](/mnt/wsl/transfer/minimax-wsl/runs/handoff-probe-20261107/hags-on.txt)):

| Upload | 5090 | 3090 |
|---|---:|---:|
| Copy and kernels on one stream, as the engine | 44.3 → **32.8 µs** | 88.1 → 52.1 µs |
| Copy stream and an event | 48.4 → **3.8 µs** | 87.6 → 19.6 µs |
| Host spinning `cudaStreamQuery` | 45.7 → 31.4 µs | 88.8 → 47.8 µs |
| Host doing CPU work | 55.6 → **32.1 µs** | 88.8 → 47.7 µs |
| Kernel reading mapped memory | 2.3 → 3.4 µs | 2.7 → 4.1 µs |

Upload rates and the small copies' 2-4 µs handoffs are unchanged.
- **The busy-host penalty is gone.** The engine's busy-host handoff should fall from about 0.057 ms to about 0.033 ms.
- **The engine's same-stream form falls only 26% with HAGS.** A copy on its own stream with an event costs almost nothing.
  So with HAGS on, demand copies issued on a copy stream, with the compute stream waiting on an event, could remove nearly all
  of the remaining handoff. That is a contained engine candidate, only with HAGS on: with HAGS off the event form cost 48 µs.
  The 0.6-1.3% bound was for removing the whole HAGS-off handoff. Only the part HAGS leaves is still to remove, roughly half
  that, to be recomputed from the HAGS-on in-situ trace.
- **3090: the same pattern**, 52 µs on one stream and 20 µs with an event. Its large copies are helper refills, off a pass's
  path, so it ranks lower. It may still shorten the time from a refill landing to the expert's first use.
- **Superseded criterion.** The same-stream median does not halve, so the probe criterion of the superseded confound rule is not
  met. The revised rule decides on the 2048-token pairs.

**Bound from the HAGS-on in-situ traces, 2026-11-08: 0.8-0.9% at `T=1`, so the copy-stream upload is not built.**
[Script](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-hags-20261107/handoff_bound_hags.py),
[result](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-hags-20261107/handoff_bound_hags.txt).
- **In the engine, the handoff with HAGS is still 0.047-0.049 ms** when the CPU finishes after the copy, against 3.8 us for
  the probe's copy stream with an event. So the in-situ gap is larger than the probe's 32.8 us same-stream figure.
- **Method.** Each traced pass's dependent kernels move earlier by the observed handoff, less a residual, and the pass's
  slowest path is recomputed. Same script as the HAGS-off bound.
- **Saved, share of traced forward time:**

  | Residual | `T=1` prose | `T=1` code | Wide prose | Wide code |
  |---|---:|---:|---:|---:|
  | 3.8 us | 0.88% | 0.81% | 0.50% | 0.14% |
  | 10 us | 0.68% | 0.64% | 0.42% | 0.12% |

- **Per-pass balance gets slightly worse.** CPU against copy-plus-primary abs(delta) rises from 0.205 to 0.232 ms (`T=1`
  prose) and from 0.235 to 0.253 (code). Only the primary path gets shorter, and it is already early. Slowest minus
  second-slowest is unchanged.
- **Decision.** It falls under the 2% rule, and `early_copy` would remove a third of the demand copies it acts on. Not built.

### Per-pass structure with HAGS on

**The same declared trace campaign after the reboot, HAGS on, 2026-11-07 22:16-22:26.**
[Receipts](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-hags-20261107/),
[paths](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-hags-20261107/hags_paths_compare.txt),
[CPU lane](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-hags-20261107/hags_cpu_lane.txt),
[division](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-hags-20261107/hags_division.txt),
[CPU per job](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-hags-20261107/hags_cpu_per_job.txt). Medians, HAGS off → on,
ms from each pass's common origin, which is the CPU batch's start when there is one:

| | `T=1` prose | `T=1` code | Wide prose | Wide code |
|---|---|---|---|---|
| Traced generation | 5,821 → 5,345 ms (**-8.2%**) | 6,774 → 6,407 (**-5.4%**) | 10,622 → 9,113 (**-14.2%**) | 18,217 → 18,035 (-1.0%) |
| First demand copy's start | +0.031 → **-0.140** | +0.032 → -0.143 | +0.011 → -0.272 | -0.042 → -0.476 |
| One-upload copy duration | 0.835 → **0.554** | 0.823 → 0.708 | 0.846 → 0.701 | 0.810 → 0.628 |
| Copy end to first dependent kernel, in situ | 0.057-0.073 → **0.033-0.049** | same | | |
| Host submit start to CPU batch start | 0.078 → **0.107** | 0.079 → 0.108 | 0.103 → **0.204** | 0.176 → **0.336** |
| Uploads per pass, the replay's own decisions | 1.00 → **1.42** | 1.29 → 1.71 | 1.65 → 1.85 | 3.11 → 3.31 |
| CPU jobs per pass | 2.90 → 2.51 | 3.43 → 3.04 | 3.49 → 3.31 | 5.74 → 5.63 |
| Slowest path | CPU 49% → primary 49% | CPU 54% → primary 50% | primary 81% → CPU 45% | primary 91% → **CPU 82%** |
| CPU vs primary transfer abs(Δ) | 0.178 → 0.205 | 0.222 → 0.235 | 0.323 → 0.266 | 0.253 → **0.801** |
| Slowest minus second | 0.231 → 0.255 | 0.239 → 0.256 | 0.266 → 0.268 | 0.238 → **0.790** |

**What changed.**
- **GPU work starts sooner.** Launches reach the GPU as they are issued, so the 5090's copy begins before the CPU lane. Copy and
  CPU reads overlap less, so both run faster.
- **The CPU is faster for the same work.** With the same division of CPU jobs and uploads, its batch took 3-23% less time.
- **But the host starts its CPU batch later.** It spends longer issuing launches first: +0.03 ms at `T=1`, +0.10-0.16 ms in
  wide windows, which have more launches.
- **The planner adapted.** Its calibrated CPU estimate for the same decision rose 13.5% (1.341 → 1.522 ms) while the work itself
  got faster, so it evidently counts the late start. It now uploads single misses instead of running them on the CPU: the most
  common `T=1` division went from one CPU job, 1,524 passes, to one upload, 1,519.
- **In wide code the upload cap of four binds.** The CPU lane is now the slowest path on 82% of passes, by about 0.8 ms.
  Balance there got worse even though speed held.

**Consequences for the work list.** Each is HAGS-specific, and none was a bound before:
- start the CPU batch before issuing the GPU launches, or issue those from another host thread;
- recompute the staging-frame and upload-cap bound, which had been 0.4-0.6% with HAGS off;
- correct what the planner's CPU calibration measures.

### Offline optimization of `T=1` policy parameters under HAGS, work-list item 2

**Result, 2026-11-08: the current settings are at the model's optimum. The best candidate is -0.02%, within model error.**
- Method: each candidate replays the production policy over the four untouched HAGS-on workloads, from the load's origin,
  and the HAGS-on per-pass model prices every pass. Two coordinate sweeps over six parameters, then steps of 0.01 on the
  continuous ones.
- Size: 129 evaluations, an hour on 16 processes.
- [Log](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-hags-20261107/t1_optimizer.txt),
  [every evaluation](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-hags-20261107/t1-optimizer-log.jsonl).

Predicted change against control, the sum over four workloads (control 101,755 ms):

| Parameter | Control | Landscape | Best |
|---|---|---|---|
| CPU price scale | 1.3 | 0.6 +14.8%, 1.0 +6.7%, **1.2 +2.97%**, 1.4-2.1 +0.04 to +0.36%, **2.2 +2.86%** | 1.31, -0.02% |
| Upload cap | 4 | 0 +14.3%, 1 +0.88%, 2-4 equal | 2, no change |
| Refill budget | 8 | 0 +20.2%, 6 +1.42%, 12 +1.74%, 32 +4.19% | 8 |
| Horizon | 20 | 5 +2.69%, 10 +0.87%, 40-80 +0.07% | 20 |
| Margin | 0 | 0.05 +0.10%, 0.4 +0.56% | 0 |
| Decay | 0.95 | 0.90 +0.83%, 0.97 +2.20% | 0.95 |

**Reading.**
- No planner or admission setting offers a predicted gain at `T=1` with HAGS on.
- The CPU price sits 0.1 above a cliff: at 1.2 the planner stops uploading single misses and loses about 3%. Values up to
  2.1 stay within 0.4%. A value farther from the cliff would give up almost nothing, if hardware ever shows the cliff moving.
- The upload cap does not bind at `T=1`.
- The model is not yet qualified, so this is a screen, not a selection. The routing bonus is not in it. It has its own
  curve [below](#routing-bonus-curve-at-t1-under-hags-work-list-item-2), re-routed from the router scores that the detailed
  captures record.
- So the remaining `T=1` levers are structural:
  - the copy-stream upload;
  - starting the CPU batch before the GPU launches;
  - the kernels;
  - `routing_prefetch`;
  - the bonus.

#### Follow-ups from the user's review, 2026-11-08

Same replay, model and workloads, with the other settings at control
([follow-up](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-hags-20261107/t1_followup.txt),
[horizon log](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-hags-20261107/t1_horizon.txt); scripts
`t1_optimizer.py`, `t1_followup.py` and `t1_horizon.py` in the
[analysis folder](/mnt/wsl/transfer/minimax-wsl/runs/job-8b7e216b-analysis/)):

| Candidate | Predicted change |
|---|---:|
| Refill budget 7 | +0.67% |
| Refill budget 9 | +0.41% |
| Refill budget 10 | +0.83% |
| CPU price 1.5 | +0.05% |
| CPU price 1.65 | +0.08% |
| Horizon 40 | +0.07% |
| CPU price 1.65 and horizon 40 | +0.14% |
| CPU price 1.5 and horizon 40, the user's choice | +0.08% |

- The refill budget of 8 is a single best value, so it stays.
- The CPU price and horizon changes each cost under 0.1%, inside model error, and together they cost no more than their sum.

#### Admission horizon at finer resolution

User direction 2026-11-08: more points on the horizon curve. 22 values from 5 to 120, the other settings at control:

| Horizon | Change | Horizon | Change |
|---:|---:|---:|---:|
| 5 | +2.69% | 25 | +0.08% |
| 6 | +2.21% | 28 | +0.06% |
| 8 | +1.36% | 32 | +0.02% |
| 10 | +0.87% | 36 | +0.08% |
| 12 | +0.55% | 40 | +0.07% |
| 14 | +0.51% | 45 | +0.08% |
| 16 | +0.28% | 50 | +0.10% |
| 18 | +0.12% | 60 | +0.09% |
| **20** | **0** | 70 | +0.08% |
| 22 | +0.04% | 80 | +0.08% |
| | | 100 | +0.10% |
| | | 120 | +0.10% |

**Reading.**
- 20 is the minimum, and it sits at the knee. Below it, every step of 2 costs 0.04-0.85%, more as the horizon shrinks.
- From 22 to 120 the curve is flat within +0.10%. Its wiggles of 0.02-0.10% are inside model error, so there is no second
  minimum to chase.
- 40 gives up about 0.07% for distance from the knee.

### Routing bonus curve at `T=1` under HAGS, work-list item 2

**2026-11-08. The user asked for the curve's shape, not its minimum:** a higher bonus is faster and worse in quality.
- Data: four untouched workloads, recorded live in detailed capture at bonus 0.10, keep every router score
  ([captures](/mnt/wsl/transfer/minimax-wsl/runs/bonus-capture-hags-20261108/)).
- Method: the replay re-routes each layer under each bonus, against its own residents, and prices the passes with the HAGS-on
  model ([curve](/mnt/wsl/transfer/minimax-wsl/runs/bonus-capture-hags-20261108/bonus_curve.txt),
  [dashboard](https://claude.ai/artifact/1CMJ4igE6oABSW56Gdqgg5#bonus)).
- Self-check: at 0.10 the re-routed picks match the engine's on **96.6-97.6% of layers**.

| Bonus | Time vs 0.10 | Step per +0.01 | Picks moved off the router's top four | Router mass kept | Usability |
|---:|---:|---:|---:|---:|---|
| 0.00 | +21.97% | | 0.0% | 100.0% | |
| 0.10 | 0 | -1.61% | 18.1% | 98.2% | |
| 0.20 | -11.53% | -0.94% | 28.4% | 94.6% | passed |
| 0.25 | -15.08% | -0.71% | 31.8% | 92.8% | |
| 0.30 | -17.77% | -0.55% | 34.4% | 91.1% | failed |
| 0.40 | -21.60% | -0.39% | 38.2% | 88.0% | |

**Reading.**
- The speed curve is smooth and concave, with no cliff and no plateau: each further 0.01 buys less, from about 2.2% near 0
  to 0.4% at 0.40.
- Router mass falls almost linearly, about 0.35 points per 0.01. So speed bought per point of mass given up falls steadily:
  about 9.7% per point from 0 to 0.10, 3.2 from 0.10 to 0.20, 1.8 from 0.20 to 0.30, and 1.2 from 0.30 to 0.40.
- Between the passing 0.20 and the failing 0.30 lies about 6% of predicted `T=1` speed. Only usability checks between them
  can say how much of it quality allows.
- These are first-order numbers: the token sequence is held as recorded, and speculation is off.

### Hardware check of the selected `T=1` settings under HAGS

**Result, 2026-11-08 02:18: keep CPU price 1.5 and horizon 40, a tie under the rule. The frozen prediction failed: it lies
outside the measured interval** ([evaluation](/mnt/wsl/transfer/minimax-wsl/runs/t1-settings-check-20261108/evaluation.txt)).

| Selected over control | Effect | 95% interval | Frozen prediction |
|---|---:|---:|---:|
| **Pooled per block: decides** | **+0.01%** | -1.08, +1.09 | -2.12% |
| python-code | -0.65% | -2.28, +0.97 | -2.50% |
| prose | -0.74% | -1.13, -0.35 | -1.59% |

Per block: +0.67%, -0.39%, +0.20%, +1.06%, +0.36%, -1.86%. Control ran 15.83-16.23 tok/s, and selected 16.00-16.28.

**Reading.**
- The selected settings are not slower, so they stay as the `T=1` base. Prose is credibly faster, by 0.7%, and code leans
  the same way.
- **The pooled tie hides that both prompts gain.** The arms wrote different text.
  - At bonus 0.20 routing follows residency, so the cache settings change the output.
  - Selected code answers ran 1,079-1,361 tokens, against control's 695-711.
  - Code costs about 5 ms per token more than prose, which raises the selected arm's pooled cost.
  - Each arm repeated its own output across blocks: control's code answer was 695 tokens in five of six. So sampling is
    fixed, and the difference comes from routing.
- **A changed output needs a usability check** under the plan's own rule. It is on the work list.
- **The model overpredicted the gain**, by about 1-2 points per prompt. The diagnostic is below.
- **Stall check.**
  - By prompt, one request is flagged: b6-control-python-code at 65.42 ms per token, against its median of 63.23.
  - Without block 6 the mean is +0.38% [-0.29, +1.05], which gives the same decision.
  - The check as fixed pools both prompts and flags 11 of 12 code requests for being code. Future protocols use the
    by-prompt version.
- **Engine counters per emitted token**, diagnosis only:

  | Prompt | Arm | ms/token | CPU experts | Uploads | CPU run ms | CPU wait ms | VRAM hit |
  |---|---|---:|---:|---:|---:|---:|---:|
  | code | control | 63.67 | 16.5 | 34.0 | 13.9 | 23.5 | 0.778 |
  | code | selected | 63.20 | 12.2 | 36.6 | 11.0 | 21.8 | 0.786 |
  | prose | control | 58.68 | 9.3 | 28.0 | 7.9 | 20.0 | 0.836 |
  | prose | selected | 58.25 | 7.4 | 28.9 | 6.7 | 18.8 | 0.840 |

  - The selected settings take a fifth to a quarter of the experts away from the CPU, and upload more.
  - The CPU lane shortens by 1.2-2.9 ms per token, and the wait for it by 1.3-1.7 ms. Yet each token is only 0.4-0.5 ms
    faster.
  - So most of the CPU's saving is spent on another path of the same passes. Only a per-pass trace of both arms can show
    which path finishes last instead.

Design and rule:
- User direction: "run the hardware check with the selected values".
- Arms. Everything else is equal: speculation off, query and spinner on, refill budget 8, decay 0.95, margin 0, upload cap
  4, routing bonus 0.20.
  - Control: CPU price 1.3, horizon 20.
  - Selected: CPU price 1.5, horizon 40.
- Design ([protocol](/mnt/wsl/transfer/minimax-wsl/runs/t1-settings-check-20261108/protocol.json)):
  - one fresh load: bootstrap, helper-first, origin;
  - six blocks, each running both arms on python-code and prose at up to 2048 tokens;
  - arm order alternating by block;
  - every request starting from the restored origin;
  - monitoring off.
- **Rule, fixed before the run.** Per block, the effect is the selected arm's milliseconds per emitted token over the
  control's, minus one, pooled over both prompts.
  - Keep the selected settings unless the hardware shows them credibly slower: the 95% interval of the mean effect across
    the six blocks lies entirely above zero, and the mean is above +1%.
  - Otherwise revert to 1.3 and 20.
- **Stall check.** A request whose generation runs more than 2 ms per emitted token over its arm's median is flagged and
  reported. It does not decide the result unless an arm causes it.
- **Frozen prediction**, written before the launch:
  - file: [prediction.json](/mnt/wsl/transfer/minimax-wsl/runs/t1-settings-check-20261108/prediction.json), sha256
    `2566ba9c4601899ced18f9db4ccc3f7158202b192018580c8c79bb537ff153c1`;
  - **selected -2.12%**: extract -1.87%, reason -2.47%, code -2.50%, prose -1.59%;
  - method: re-route the four detailed captures at bonus 0.20 in both arms, then price them with the HAGS-on model
    (`predict_check.py` in the [analysis folder](/mnt/wsl/transfer/minimax-wsl/runs/job-8b7e216b-analysis/)).
**Why the prediction failed: replay diagnostic.** Before the run, the frozen prediction (-2.12%) and the offline search
(+0.08% at bonus 0.10, light captures) disagreed
([diagnostic](/mnt/wsl/transfer/minimax-wsl/runs/t1-settings-check-20261108/diagnose_gap.txt), `diagnose_gap.py`). It uses
the same model and the detailed captures throughout, and reproduces the frozen value exactly, -2.1188%.

| Detailed captures, selected over control | Predicted | CPU jobs | Uploads |
|---|---:|---:|---:|
| Routing held as recorded, the search's method | -1.47% | -28.0% | +23.7% |
| Re-routed at 0.10, the recording's own bonus | -1.93% | -29.9% | +23.1% |
| Re-routed at 0.20, the frozen prediction | -2.12% | -46.4% | +35.7% |
| 0.20, CPU price 1.5 alone | -2.16% | -46.5% | +35.6% |
| 0.20, horizon 40 alone | -0.04% | +0.3% | -0.2% |

- **The captures carry most of the gap: 1.55 of the 2.2 points.** With routing held, the detailed captures predict -1.47%,
  where the light captures predicted +0.08%. Re-routing at 0.10 adds 0.46 points, and the bonus 0.19.
- **The whole effect is the CPU price.** The horizon does nothing, and the two do not interact.
- **The two loads' planners flip at different prices.**
  - In the light captures' load, the CPU's job count barely moves from 1.3 to 1.7 (code: 7,980 jobs, and 7,954 at 1.5).
    Its cliff is at 1.22-1.23, where the count halves.
  - In the detailed captures' load, 28% of the CPU's jobs move between 1.3 and 1.5. On hardware the CPU took 20-26% fewer
    experts.
  - So in two of three loads, 1.3 sat inside the flip rather than above it, as the user suspected.
- **Why they differ.** Each load's planner prices work from its own startup measurements, and those vary:

  | HAGS-on load | CPU base, ms | One CPU expert | One GPU expert | Upload |
  |---|---:|---:|---:|---:|
  | HAGS study load 1 | 0.079 | 0.582 | 0.031 | 0.624 |
  | HAGS study load 2 | 0.177 | 0.530 | 0.030 | 0.614 |
  | HAGS study load 3 | 0.134 | 0.468 | 0.030 | 0.638 |
  | Per-pass trace | 0.092 | 0.539 | 0.028 | 0.614 |
  | Light captures, the offline search | 0.149 | 0.489 | 0.029 | 0.610 |
  | Detailed captures, the frozen prediction | 0.084 | 0.531 | 0.030 | 0.618 |
  | Settings check | 0.087 | 0.540 | 0.028 | 0.626 |

  - The CPU base varies by more than a factor of two across these loads. With HAGS off it ranged from 0.015 to 0.148.
  - The offline search ran on the load with the second-highest CPU base, where 1.3 was already past the flip.
  - The landscapes are conditional on one load's calibration, so a price chosen on one load can sit on another load's
    cliff.
- **The model probably also overvalues the shift.**
  - On hardware, a similar shift of the CPU's experts (20-26%, against 28%) bought about 0.7% per prompt.
  - The model gives 1.5% with routing held on the detailed captures, a different load.
  - A per-pass trace of both arms would measure the shift's real cost directly.
- **Quality.** At 0.20 the two arms keep the same router mass, 94.60% and 94.61%, with 28.4% of picks moved in both. The
  settings change the text without moving routing further from the router.

**Is 1.5 safe on every load? Calibration sweep, 2026-11-08.** The same demand (the detailed captures, routing held) was
replayed with each HAGS-on load's planner tables, at prices from 1.2 to 2.2
([sweep](/mnt/wsl/transfer/minimax-wsl/runs/t1-settings-check-20261108/calibration_sweep.txt), `calibration_sweep.py`; 91
replays). Every calibration reaches the same floor. They differ in where it starts and ends.

| Planner tables from | Above the best at 1.3 | At 1.5 | Within 0.15 of the best | At 2.0 | At 2.2 |
|---|---:|---:|---|---:|---:|
| HAGS study load 1 | 0.04% | 0.09% | 1.3-1.6 | 0.32% | 2.74% |
| HAGS study load 2 | 0.01% | 0.03% | 1.2-1.6 | 2.70% | 3.62% |
| HAGS study load 3 | **4.52%** | 0.05% | 1.45-1.8 | 0.21% | 0.21% |
| Per-pass trace | 0.01% | 0.07% | 1.3-1.7 | 0.36% | 1.31% |
| Light captures, the offline search | 0.08% | 0.02% | 1.3-1.7 | 0.21% | 2.71% |
| Detailed captures, the frozen prediction | **1.59%** | 0.04% | 1.4-1.7 | 0.30% | 0.32% |
| Settings check | **0.32%** | 0.07% | 1.35-1.7 | 0.31% | 0.31% |

- **1.5 is within 0.09% of the best under all seven calibrations. The prices all seven share are 1.45-1.6.**
- At 1.3, three of the seven sat above the floor, by 0.3, 1.6 and 4.5 points. Startup noise could cost an unlucky load
  several percent at the old price. At 1.5 the spread is 0.07 points.
- The upper cliff moves too: 2.0 already costs load 2 2.7%, and 2.2 costs three loads 2.7-3.6%.
- **This explains most of the failed prediction.**
  - Priced with the settings check's own tables, the model predicts -0.25% for 1.3 to 1.5 on this demand. That is inside
    the pooled interval [-1.08, +1.09], though just outside prose's.
  - The frozen prediction used the detailed captures' tables, where 1.3 sat 1.6 points above the floor.
- **Protocol change for later checks:** freeze the prediction after the run's load has measured its tables, and price it
  with them. The tables are in the load's origin file before the first check request.
- **Candidate:** make the planner's tables less noisy, either by repeating the startup measurement and taking medians, or by
  refining them from measured passes during generation.
  - At 1.5 the CPU price no longer depends on them. The second-miss decision near 2.0 and the planner's other choices still
    do.
  - It belongs to work-list item 3's planner calibration. It is the same per-pass balance between the CPU lane and the
    copy path.

### Per-pass traces of CPU price 1.3 and 1.5

**2026-11-08: on the traced load both prices made the same decisions, so the traces cannot show where the CPU's saving
goes.**
- **Design** ([run](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-settings-20261108/),
  [comparison](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-settings-20261108/trace_compare.txt)):
  - one Nsight-launched load, HAGS on;
  - the HAGS-on per-pass trace's two recorded `T=1` requests, replayed as fixtures at CPU price 1.3 and at 1.5;
  - identical demand in both arms, in ABBA order.
- **Decisions.** The planner chose the CPU on 2,198 against 2,200 code passes and 1,054 against 1,053 prose passes. Demand
  copies were 4,155 against 4,144 and 2,763 against 2,771. This load's planner tables put both prices past the single-miss
  flip, as on the light-capture load. The settings-check load instead moved 20-26% of its CPU experts.
- **The few passes that did move.** Only 5 code and 8 prose CPU jobs became uploads at 1.5. Those passes finished a median
  0.38 and 0.40 ms sooner, with the copy path last.
- **Traced generation** changed -0.02% (code) and -0.15% (prose).
- **The balance measures are unchanged.** Mean abs(delta) between the CPU and primary-transfer completions went from 0.235 to
  0.229 ms (code) and from 0.223 to 0.210 (prose). Slowest minus second-slowest went from 0.259 to 0.254 and from 0.264 to
  0.261. Per-layer values are in the comparison.
- **What a useful trace needs:** a load whose flip lies between the two prices. Those prices would be chosen from the
  load's own tables after startup, the same protocol change as for frozen predictions.

### Usability check of CPU price 1.5 and horizon 40

**Superseded the same day by the [revised rules](#rules-for-adoption-revised): every failure here was a token-budget cut-off,
and the settings have no quality mechanism, so CPU price 1.5 and horizon 40 are restored.** The record as first written:

**2026-11-08: fail under the frozen rule, so the serving defaults and the `T=1` base go back to 1.3 and 20.** This reverses
the speed check above, which kept 1.5 and 40 as a tie
([run](/mnt/wsl/transfer/minimax-wsl/runs/usability-settings-20261108/),
[evaluation](/mnt/wsl/transfer/minimax-wsl/runs/usability-settings-20261108/evaluation.txt),
[every final answer](/mnt/wsl/transfer/minimax-wsl/runs/usability-settings-20261108/review-dump.txt)).
- **Design.** The 2026-11-07 matrix's tasks, budgets, seeds and rule, run with speculation on as served, at bonus 0.20 and
  HAGS on. The two arms are 1.5 and 40 against 1.3 and 20. Each entry starts from the restored origin, and arm order
  alternates by seed.
- The primary agent read every final answer. `count_runs` was rescored with the fixed checker, the same way for both arms.

| Task, passes of 3 seeds | 1.3/20 | 1.5/40 |
|---|---:|---:|
| Scheduling: 11 and 12, valid schedules and bounds | 3 | 3 |
| Muffins: 3 combinations | 3 | 3 |
| Action items, complete within 1,024 tokens | 3 | **2** |
| Log summary, no added facts | 3 | 3 |
| `order_jobs`, eight executed checks | 3 | 3 |
| `count_runs` first attempt | 3 | **2** |
| `count_runs` return | **2** | 3 |
| Tool round trip, disconnect and follow-up | 3 | 3 |

- **Every failure in either arm is a token-budget truncation**, not a wrong answer:
  - 1.5/40, seed 1: the action items' high-effort reasoning used the 1,024 tokens, and the answer stops mid-summary;
  - 1.5/40, seed 3: the first `count_runs` was cut off mid-function at 512 tokens;
  - 1.3/20, seed 3: the same happened to its `count_runs` return.
- **No output shows repetition, broken text or a logic collapse.** Each arm has small slips:
  - in 1.3/20, seed 3 calls B's change of +2 "+1", and a choir note serves tea before the hall opens;
  - in 1.5/40, seed 1 calls a plainly expected exclusion "counterintuitive", and one choir note invents specific details.
- **Rule, fixed before the run:** the new defaults fail if, on any task, they pass on fewer seeds than control. They do on
  two tasks.
- **Precedent:** bonus 0.30 failed on a single empty answer.
- **Limits.**
  - One budget overrun per task decides the result, and control had one too.
  - Routing fidelity is unchanged (router mass kept 94.61% against 94.60%).
  - So three seeds cannot separate this from noise. A rerun with more seeds or larger budgets would be needed to bring 1.5
    and 40 back.

### `routing_prefetch` review, work-list item 4

**2026-11-08: the existing implementation cannot run in the current configuration, so it cannot be tuned as it stands.**
It is in [mm3_native_prefetch.inc](../../src/mm3/mm3_native_prefetch.inc), 229 lines.
- **Gate.** `prefetch_supported()` requires `!primary_adaptive`, among others: one device, one sequence, native window,
  worker dispatch, eight staging frames, no graphs, no vision. The serving defaults run `primary_adaptive: 1`, so turning
  `routing_prefetch` on returns an error.
- **Mechanism.** At layer *l* it:
  - applies layer *l+1*'s router (`next.gate`, bias, the same top-4 routing) to layer *l*'s normalized input `d.xn`;
  - reads the predicted ids and weights back to the host;
  - sums the support across rows;
  - queues **one** optional upload per layer: the HOST expert with the most predicted support that is not resident,
    helper-owned, leased or pending, into the least recently used frame of the upper half of the staging frames.

  The copy goes on the copy stream behind an event after the layer's demand uploads, never behind another optional copy.
  An unissued prediction is cancelled at the next layer's start, so it cannot drift into a later layer.
- **Bookkeeping.** It counts useful, late, unused and cancelled predictions, with their bytes. No demand score is read or
  changed.
- **Its only trial** was before adaptive residency: 14.0324 tok/s off against 14.0442 on (+0.08%), with opposing task
  changes ([parked mechanisms](#parked-mechanisms)).

**What is worth testing instead.** Use the same cross-layer prediction to start layer *l+1*'s likely misses copying during
layer *l*, through the planner's staging frames, compatible with adaptive residency.
- A correctly predicted miss then arrives as a hit, either removing a CPU job or taking a demand copy off the critical
  path. That is the per-pass balance this plan optimizes.
- At `T=1` there are 0.6-0.9 misses per MoE pass: 7-17 CPU experts and 28-37 uploads per token, in the settings check,
  over 57 MoE layers. The copy engine has room for about one 0.6-0.8 ms copy per pass of about 1 ms.
- The ceiling is large; the unknown is the prediction's accuracy on MiniMax, which has never been measured here.

**Order.**
1. Measure the accuracy first: run the prediction kernels in a predict-only diagnostic that adaptive residency allows,
   log predicted against actual top-4 per layer and pass in detailed capture, and issue no uploads.
2. Bound the gain on the per-pass model with the measured hit rates, counting the copy engine's capacity and the
   contention with demand copies.
3. Build only at 2% or more.

**Prediction accuracy, 2026-11-08 (order step 1).** The predict-only diagnostic ran four requests in detailed capture on its
own load ([predict-diag-20261108](/mnt/wsl/transfer/minimax-wsl/runs/predict-diag-20261108/), analysis
`predict_accuracy.py`).
- **Setup.** The build runs layer *l+1*'s router on layer *l*'s input whenever `MM3_PREDICT_DIAG=1`, and logs the predicted
  top-4 in layer *l+1*'s routes record. It issues no uploads. The run used bonus 0.20, CPU price 1.5 and horizon 40, with
  `routing_prefetch` off.
- **Coverage.** 85,792 decode passes carry a prediction, on layers 4-59. Layer 3 follows a dense layer and has none.
- **Match with the router.** The prediction matches 75.3% of the router's own top-4 (code 75.9%, extract 75.7%, prose
  78.5%, reason 71.0%). Its first pick is in the router's top-4 on 97.0% of passes.
- **Misses.** The prediction contains 91.4% of the misses, at 0.735 misses per pass:
  - 0.165 per pass are served by the CPU, 90.5% of them predicted;
  - 0.570 per pass are demand uploads, 91.7% of them predicted;
  - 54.2% of passes have a miss, and 94.7% of those have at least one predicted.
- **Match with executed picks.** It matches the executed picks less often, 64.6%, because the bonus moves 27.7% of picks
  off the router's top-4 onto resident experts.
- **Waste.**
  - A policy that copies the first predicted expert not already on a GPU would copy on 90.5% of passes, and 53.2% of those
    copies would be used.
  - Copying every such predicted expert would waste about 1.2 copies per pass.
  - The log does not record whether an expert that did not run was resident. It is inferred from the nearest observation
    of that expert in the same request, so the waste figures are approximate.
- **By layer.**
  - Overlap with the router's top-4 is 0.41 at layer 4, 0.59-0.73 at layers 5-23, and 0.80-0.86 at layers 36-55.
  - The share of misses predicted follows it: 0.78 at layer 4, 0.79-0.95 at layers 5-33, and 0.91-0.99 from layer 34 on.

**Bound on the per-pass model, 2026-11-08 (order step 2): -8.0% under the rule the engine can follow, so it is built.**
[Script](/mnt/wsl/transfer/minimax-wsl/runs/predict-diag-20261108/prefetch_bound.py),
[result](/mnt/wsl/transfer/minimax-wsl/runs/predict-diag-20261108/prefetch-bound.txt).
- **Candidate.** At layer *l*, copy layer *l+1*'s first predicted HOST expert into the least recently used free staging frame.
  The frame must not be one the current pass reads. At layer *l+1* the expert is ready, so routing gives it the residency bonus
  like any ready expert, and the planner serves it with no demand copy.
- **Replay.** The production policy over the diagnostic's own four requests, from that load's origin. Routes are re-derived
  from the recorded router scores under bonus 0.20, against the replay's own residents, at the `T=1` base (CPU price 1.3,
  horizon 20, refill budget 8). Each pass is priced with the HAGS-on per-pass model.
  - "Free" pricing charges the copy nothing.
  - "Charged" pricing makes the copy share host bandwidth with layer *l*'s CPU batch.
- **Result, generation time against no early copy:**

  | When a copy is allowed | Free | Charged | CPU jobs per pass | Uploads per pass | Copies per pass, used |
  |---|---:|---:|---:|---:|---:|
  | Never (today) | | | 0.165 | 0.571 | |
  | Layer *l* queued no demand copy, no exchange in flight (**the engine's rule**) | **-8.01%** | **-8.01%** | 0.107 | 0.360 | 0.567, 87.4% |
  | Layer *l* queued no demand copy | -8.21% | -8.21% | 0.105 | 0.354 | 0.582, 87.2% |
  | At most one demand copy (optimistic: it can arrive late) | -14.65% | -14.20% | 0.038 | 0.218 | 0.893, 88.7% |
  | Always, one copy | -14.75% | -14.31% | 0.037 | 0.217 | 0.896, 88.7% |
  | Always, two copies | -10.03% | -9.40% | 0.052 | 0.403 | 1.529, 70.5% |

  - Per workload, the engine's rule gives -7.3% to -8.7% (reason, extract, code, prose).
  - The free and charged pricings agree under it. That is because a pass with no demand copy almost never has a CPU job:
    at price 1.3 the planner uploads a pass's first miss.
- **Why 87% are used, against the 53% the accuracy analysis implied.** A copied expert is ready, so the bonus makes routing
  pick it whenever it is in the router's top-4 or close to it. The prediction partly fulfils itself.
- **Routing fidelity improves**, which also changes output:
  - router mass kept among executed picks rises from 94.78% to 95.78%;
  - picks moved off the router's top-4 fall from 27.9% to 23.7%.
- **Two copies per pass do worse than one.** Their second copies are used only 70% of the time, and they evict frames that
  later passes would have reused.
- **Per-pass balance, the plan's central objective.** Means over decode passes, from the model's path ends measured from the
  common origin:

  | Metric | Today | Early copy |
  |---|---:|---:|
  | Slowest minus second-slowest path | 0.278 ms | **0.223 ms** |
  | Copy-plus-primary against the helper path | 0.372 ms | 0.280 ms |
  | CPU against copy-plus-primary | 0.197 ms (14,221 passes) | 0.207 ms (9,201 passes) |
  | Helper against CPU | 0.678 ms | 0.682 ms |

  - Slowest minus second-slowest falls on all 57 MoE layers. For example, layer 28 goes from 0.318 to 0.263 ms, layer 44 from
    0.258 to 0.174, and layer 59 from 0.222 to 0.174.
  - Layer 3 has no prediction and barely moves (0.315 to 0.312).
  - CPU against primary rises slightly where CPU passes remain. There are a third fewer of them, and they are the harder ones.
- **Not counted.** The bound leaves out the prediction's own cost on the GPU (the next layer's router on the side stream, and
  two small readbacks per layer) and the reader's event wait when a copy is late.
- **The larger variants are later candidates.** Allowing the copy behind one demand upload predicts -14.6%, but the replay
  treats a late copy as ready. Test it only after the engine's rule is confirmed, and with the copy's arrival measured.

**Built, 2026-11-08: `early_copy`, a per-request option that is off by default.** It needs a single GPU, the native
cooperative path and `T=1`, and it excludes `routing_prefetch`.
- **Prediction.** It reuses the diagnostic's prediction kernels and readback.
- **Issue point.** It is issued after the layer's plan upload.
- **Conditions.** The layer queued no demand copy, and no exchange or promotion is pending.
- **Copy.** The first predicted HOST expert of layer *l+1* is copied:
  - into the least recently used frame that is not reserved, not read by this pass, and has no copy pending;
  - on the copy stream, behind an event on the compute stream.
- **Frame ownership.** The frame is owned at once, so the next layer's ready mask includes it. Its reader in `locate` waits on
  the frame's event.
- **Protection while the copy is unobserved.** Until its event is seen complete, the frame is excluded from:
  - demand-copy victims, and the planner's count of free frames;
  - promotion and exchange landings, and exchange reuse;
  - publication of the expert's adaptive exchange;
  - recycling of its HOST record by the 3090's refill.
- **Joins.** `drop_slots` joins the copy stream first.
- **Counters per request:** `early_copies`, `early_copy_used`, `early_copy_unused`, `early_copy_waits`,
  `early_copy_blocked`, `early_copy_no_frame`.
- **Diagnostics:**
  - `MM3_EARLY_COPY_VERIFY=1` reads each early-copied frame back at its first read, compares it byte for byte with its HOST
    record, and stops the engine on a difference;
  - `MM3_CALIBRATION_REPEATS=n` repeats the planner's startup calibration *n* times, logging each, for the calibration-noise
    item.
- **Next.** Build with the machine otherwise idle, verify frames on a load of their own, then run the timed test
  ([protocol](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-hags-20261108/make_protocol.py)). Its frozen prediction is priced
  with that load's own planner tables before the first timed request. A usability check follows any adoption, because the
  output changes.

**Frame verification, 2026-11-08: every early-copied frame was byte-identical to its HOST record.**
[Result](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-verify-20261108/verify.txt).
- **Setup.** One load of the build (sha256 `9a590c9a...`) with `MM3_EARLY_COPY_VERIFY=1`, at the `T=1` base. python-code and
  prose at 384 tokens were each run with and without `early_copy`.
- **Counts.**
  - Code: 11,780 copies, of which 10,470 were read and verified.
  - Prose: 13,205 copies, of which 11,763 were read and verified.
  - No reader had to wait on an unfinished copy, and no pass lacked a free frame.
  - The arms without `early_copy` made no copies.
- **The counts match the replay.** 0.55 copies per eligible pass against 0.567 predicted, and 89% used against 87%. 38% of
  passes were blocked by their own demand copy, against 35% predicted.
- **Text.** All four texts are coherent and on task. With and without the copy they differ, as routing changes.
- **A first verification load was stopped at 06:21.** Its server dropped the new counters from each request's stats, so the
  check could not have passed. That was fixed in 9f8c726 and the load rerun; the stopped load's files are kept under
  `aborted-0620`.
- **Mode 2 passed the same check** ([result](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-mode2-verify-20261108/verify.txt),
  build of 6737181).
  - Code: 16,885 copies, 15,826 of them read and verified.
  - Prose: 18,418 copies, 16,383 read and verified.
  - No reader waited, and every text is coherent. The code request ended on its own at 318 tokens, with a complete function.
  - Mode 2 makes about 53 copies per token, against mode 1's 31.

**First timed test, 2026-11-08: -10.51% [-11.32, -9.71], but against a base that never uploaded.**
[Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-hags-20261108/evaluation.txt).
- **Design.** One fresh load, six blocks, both arms on python-code and prose at up to 2048 tokens, at the `T=1` base.
- **Result, on over off.** Every block lay between -9.40% and -11.53%: code -9.21% [-10.89, -7.53], prose -8.69% [-10.03,
  -7.35]. The rule adopts it for its usability check.
- **The load was abnormal.** It ran the calibration-noise diagnostic (`MM3_CALIBRATION_REPEATS=12`), and its planner kept a
  table that had caught the 5090 slow (below):
  - an H2D copy of 4.41 ms, against the usual 0.61;
  - 0.857 ms per grouped expert, against 0.03.
- **So this load's planner never uploaded a miss.** Both arms show zero demand copies, and the off arm served 62 CPU jobs
  per token and ran at 74-76 ms per token. Every other `T=1` test today ran at about 60.
- **What it shows.** The comparison is valid within its load, but it measured the early copy against a broken base. On that
  base the copies did the uploads' work: CPU jobs fell from 62 to 24 per token, with 53 copies per token, 94% of them read and
  none waited for.
- **The frozen prediction**, -15.71%, was priced with the same broken tables, and the hardware gave two thirds of it.
- **It does not settle the effect on a normal base.** A three-arm test on a load with normal tables does: off, mode 1 and
  mode 2 ([protocol](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-mode2-20261108/make_protocol.py)).
  - A validity check, fixed before the run, stops the campaign before timing if the tables are abnormal: H2D or D2H at
    1.0 ms or more, or the grouped-expert extra at 0.2 ms or more.
  - The usability check runs only if mode 1 is confirmed there.

**Three-arm test on a normal load, 2026-11-08: mode 1 confirmed at -2.43%, mode 2 chosen at -2.65% more, -5.03% in all.**
[Protocol](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-mode2-20261108/make_protocol.py),
[evaluation](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-mode2-20261108/evaluation.txt).
- **Design.** One fresh load of the mode-2 build, with no calibration repeats. Six blocks, each with all three arms on
  python-code and prose at up to 2048 tokens, in a repeated Latin square, every request from the restored origin.
- **Validity, checked before timing:** H2D 0.612 ms, D2H 0.604, grouped-expert extra 0.029. These are normal.
- **Result, pooled over both prompts:**

  | Comparison | Measured | Code | Prose | Frozen prediction |
  |---|---|---|---|---|
  | Mode 1 over off | **-2.43%** [-4.49, -0.38] | -1.82% | -4.32% | -9.21%, outside |
  | Mode 2 over mode 1 | **-2.65%** [-3.56, -1.73] | -4.56% | -0.44% | -2.60%, inside |
  | Mode 2 over off | **-5.03%** [-6.39, -3.66] | -6.30% | -4.78% | -11.57%, outside |

  - In the last two blocks, off ran at 61.1 ms per token, mode 1 at 59.8 and mode 2 at 58.3-58.4.
  - The stall check flagged one request, in mode 1.
- **Decision under the rule:** mode 1 is confirmed and mode 2 is chosen. The usability check of mode 2 against off began at
  08:02.
- **Counters per emitted token:**

  | | CPU jobs | Demand uploads | Early copies (read) | CPU batch time | Host lane time |
  |---|---:|---:|---:|---:|---:|
  | Off | 18.6 | 22.2 | | 13.8 ms | 24.2 ms |
  | Mode 1 | 11.7 | 11.0 | 40.6 (36.5) | 10.3 ms | 20.9 ms |
  | Mode 2 | 8.9 | 5.9 | 49.2 (44.5) | 8.0 ms | 19.0 ms |

  No reader waited for a copy in either mode.
- **Why the model over-predicted.**
  - The early copies did remove the work: the CPU lane fell 3.5 ms per token with mode 1 and 5.8 ms with mode 2.
  - But the token fell only 1.3 and 2.8 ms. Other paths take over on many passes once the CPU job is gone.
  - That is the same pattern as CPU price 1.5 earlier today: the lane saved 1.2-2.9 ms per token, and the token
    0.4-0.5 ms.
  - Each remaining CPU job also cost more: 0.742 ms with the copies off, 0.879 with mode 1 and 0.896 with mode 2. That fits
    both fewer jobs sharing each batch's fixed cost and the copies' host-memory reads competing with the lane. These
    aggregates cannot tell the two apart.
- **A lesson for the per-pass model.** At `T=1` it credits CPU-lane savings about three times too strongly, and it credited
  the helper path's host time 40% too weakly (`helper_fast`). Its incremental prediction for mode 2 over mode 1 was right.
  - The next refit should fit the share of passes that the CPU lane really decides, from these runs' counters and a light
    capture of each arm.
  - Until then, predictions of CPU-removing changes should be read as upper bounds.
- **The pricing is not the cause; a hidden per-layer cost is the likely one.**
  [Empirical pricing](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-mode2-20261108/empirical-pricing.txt).
  - **Method.** Each replayed pass is priced by the measured mean cycle of real passes with the same work, from 87k
    uncaptured passes of the HAGS-on validation load. Examples: no CPU job, two helper jobs, one upload: 0.995 ms. One CPU
    job, one helper job, one upload: 1.402 ms.
  - **Result.** That table predicts even more than the model: mode 1 over off -13.2%, mode 2 over mode 1 -3.6%. The
    hardware gave -2.43% and -2.65%.
  - **So passes with early copies cost more than passes of equal work without them**, by about 4.3 ms per token, or about
    75 us per layer.
  - **The prime suspect is the prediction itself.** It runs on every `T=1` MoE layer, in both modes, which fits mode 2 over
    mode 1 being predicted right:
    - about four more launches (the next layer's router product, its top-k, two readbacks) on a host that already
      spends 0.236 ms per layer enqueuing, at roughly 10 us per launch;
    - its GPU time, joined into the primary stream before the routing readback.
  - **Test, built.** `early_copy=3` runs the prediction and copies nothing (361c068). The next diagnostic load times the
    preparation with it, against 0 and 2.
  - **If it is confirmed, the fix is a cheaper prediction:**
    - one fused launch, or a captured graph per layer;
    - only the ids read back;
    - kept off the routing join.

**Usability check of mode 2, 2026-11-08: it fails under the frozen rule, so `early_copy` stays off.** *Superseded the same
day by the [revised rules](#rules-for-adoption-revised): the failure was two cut-off choir notes, and the early copy has no
quality mechanism, so it is admissible once re-timed on the new base.*
[Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-usability-20261108/evaluation.txt),
[answers](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-usability-20261108/review-dump.txt).
- **Setup.** At `T=1` without drafting, the 2026-11-07 matrix unchanged in tasks, budgets, seeds and checks. Mode 2 against
  off, every entry from the restored origin. The primary agent read every answer.
- **Results, control against mode 2:**

  | Task | Control | Mode 2 | Note |
  |---|---|---|---|
  | `count_runs`, first attempt | 3 | 3 | |
  | `count_runs`, return | 3 | 3 on reading | see below |
  | `order_jobs`, scheduling, muffins, log summary | all | all | |
  | Action items complete | 2 | 3 | control's miss was a budget truncation |
  | Choir note complete | 3 | **1** | decides |

- **The `count_runs` return.** It failed the automatic check only because the checker's restricted builtins lack
  `object()`, which the answer uses as a sentinel. The prompt asks for a pure Python function. With ordinary builtins, the
  same function passes all five checked cases, so it is read as a pass.
- **The choir note decides.** Two of mode 2's notes hit the 1,024-token budget:
  - one is cut off in its last paragraph;
  - the other spent most of its budget reasoning and stops in the first bus direction.

  All three control notes are complete, at 847-1,009 tokens, so they are close to the limit too. Two of mode 2's notes also
  invent hall names and bus routes where control used placeholders.
- **Nothing broken.** No answer in either arm shows repetition, broken text or a logic collapse.
- **Decision.** As in the check of CPU price 1.5 and horizon 40, an answer cut off by its budget does not pass. Mode 2
  passes the choir note on fewer seeds than control, so it fails and `early_copy` stays off.
- **What would bring it back.** Three seeds at budgets this tight cannot separate the difference from noise. A larger check,
  with more seeds or larger budgets, could. The speed is not in doubt: -5.03% against off on a normal load.

**Diagnostic load, 2026-11-08 (eb3620b): the early copy's cost is contention with the CPU lane, the 5090's preparation is
launch-bound, and the calibration fix works.**
[Protocol](/mnt/wsl/transfer/minimax-wsl/runs/diag-prepare-20261108/make_protocol.py),
[analysis](/mnt/wsl/transfer/minimax-wsl/runs/diag-prepare-20261108/diag-analysis.txt).
- **Setup.** One load with `MM3_PREPARE_EVENTS=1` and light capture, every request from the restored origin, at the `T=1`
  base with `helper_fast`. python-code and prose at 1,024 tokens, each with five arms:
  - `early_copy` 0;
  - 3 (the prediction only, nothing copied), in its first form and in its lean form;
  - 2, in both forms.
- **The 5090's preparation, per one-row MoE layer, with no prediction:**
  - its span on the primary stream is 0.283-0.293 ms;
  - the host's enqueue time is 0.259-0.260 ms;
  - the host's wait at the join is 0.228-0.239 ms.

  The span exceeds the enqueue time by only 25-35 us, so the 5090 runs the preparation no faster than the host launches it.
  Its kernels are busy about 0.215 ms of it in the traces.
  - The join wait is mostly the 5090 reaching this layer late, about 0.2 ms after the host starts it, because of the previous
    layer's primary work.
  - **A graph-captured preparation could save up to about 0.07 ms per layer**, roughly 4 ms per token or 6% at `T=1`. It
    does not change output. It is the largest opportunity found today.
- **The prediction's cost per layer (mode 3 against 0):**

  | Form | Host enqueue | 5090 span | Join wait |
  |---|---|---|---|
  | First form (four calls) | +22.6 to +34.7 us | +6 to +17 us | -17 us |
  | Lean form (one graph launch) | +13.9 to +27.7 us | -4 to +7 us | -18 to -21 us |

  The join wait absorbs most of it: about 5-18 us net per layer in the first form, and half that or less in the lean form.
  That is far from the 75 us the shortfall implied, so the prediction explains only a small part of it.
- **Contention, the main cost.** CPU batch time on passes with one CPU job and one demand upload, where mode 2 copies beside
  the batch:

  | Prompt | Off | Mode 2 | Change |
  |---|---|---|---|
  | Code | 0.889 ms | 0.994 ms | **+11.8%** |
  | Prose | 0.902 ms | 1.017-1.023 ms | **+12.8 to +13.4%** |

  - On passes with two uploads, where mode 2 never copies, the change is +0.2 to +5.2%, on far fewer passes (40-155).
  - So an early copy's reads from host memory slow a concurrent CPU batch by about 0.1 ms. The per-pass model's sharing
    rule did not charge that.
- **A better rule, to bound next.** Copy behind one demand copy only on passes without a CPU job. At price 1.3, a pass with a
  CPU job always has an upload, so mode 1 never copies beside a CPU batch. That is why mode 1 had no contention, and why
  mode 2's extra copies cost more than the replay said.
- **Calibration fix confirmed.** This load's first calibration measured implausibly, was measured again once, and ended
  plausible: H2D 0.661 ms, grouped-expert extra 0.024 ms, 921 ms of warm-up. The second calibration was plausible on its
  first measurement.
- **Per-pass balance at the records level** (CPU against helper completion; light records hold no GPU times):
  - CPU against helper abs(delta) is 0.78-0.79 ms with the copies off;
  - with mode 2 it is 0.90-0.93 ms, over a quarter as many CPU passes;
  - the remaining CPU passes carry more work, and are slowed by the contention.

<a id="early-copy-inside-prep-graph"></a>

**`early_copy` inside `prep_graph`, 2026-11-08 (3fc50e6, task 41): verified bit for bit; timed test below.**
- **What changed.** Until 3fc50e6 `prep_graph` stayed off whenever `early_copy` was on. Now a `T=1` preparation's graph also
  holds the prediction: inside a capture, `early_copy_predict` records the next layer's router product and top-k into the
  graph, since captures cannot nest. The ids reach the host by a zero-copy kernel on the pinned buffer's device view, as the
  routing readback does, and the host reads them after the same join. The prediction's buffers, event and view are made
  before any capture. The graph key carries whether the prediction is in it, and `prep_graph 2` also compares the
  prediction ids.
- **Verification** ([protocol](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-graph-verify-20261108/make_protocol.py),
  [result](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-graph-verify-20261108/verify.txt)). One load at the branch's `T=1`
  base, up to 1,024 tokens, with `MM3_EARLY_COPY_VERIFY=1`, which reads every early-copied frame back at its first read and
  compares it with its HOST record:

  | Step | Graph replays (captures) | Verified bit for bit, mismatches | Predictions | Early copies, used | Frames verified |
  |---|---:|---:|---:|---:|---:|
  | Code, `early_copy 2`, `prep_graph 2` | 43,491 (399) | 43,092, 0 | 42,728 | 40,520, 36,852 | 36,852 |
  | Prose, the same | 27,189 (285) | 26,904, 0 | 26,712 | 22,406, 19,828 | 19,828 |
  | Code, `early_copy 2`, `prep_graph 1` | 20,976 (228) | | 20,608 | 19,429, 17,975 | 17,975 |
  | Prose, the same | 30,096 (285) | | 29,568 | 25,292, 22,532 | 22,532 |
  | Code, `early_copy 0`, `prep_graph 1` | 21,831 (228) | | 0 | 0 | |

  No fallback anywhere, and the texts read as coherent. **Pass.**
- **Less lead time.** In the two plain-replay steps, 63-68% of used early copies had their reader wait for the copy
  (12,224 of 17,975; 14,088 of 22,532). The mode 2 test before `prep_graph` had none. A shorter preparation and mode 2's
  place behind a demand copy leave the copy less time; a waited copy still saves part of an upload. This load also read
  every frame back synchronously, so its timing is not representative.
- **Timed test** ([protocol](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-graph-t1-20261108/make_protocol.py)): one fresh
  load, six blocks of both arms on code and prose at up to 2,048 tokens, `early_copy 2` against 0, with `prep_graph`,
  `helper_fast`, CPU price 1.5 and horizon 40 in both. Rule: adopt if the 95% interval lies entirely below zero.
  - Frozen prediction: **-5.70%** (code -7.13%, prose -5.41%), the earlier -5.03% scaled by the base's 11.7% speedup since,
    assuming the same milliseconds saved.
  - Recorded beside it after the verify load, not instead: -3.84% if a waited copy saves half of an unwaited one.
- **Result: -12.63% [-13.85, -11.42], so `early_copy 2` joins the `T=1` base**
  ([evaluation](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-graph-t1-20261108/evaluation.txt)). Every block lay between
  -10.96% and -13.69%: code -11.38% [-12.72, -10.05], prose -13.42% [-15.72, -11.12]. The off arm ran 52.7-53.5 ms a token
  (18.7-19.0 tok/s), the on arm 45.7-46.9 (21.3-21.9 tok/s). No stall was flagged, and every request echoed its arm.
  - **The frozen prediction failed on the good side:** -5.70% lies outside the interval, and so does the wait-adjusted -3.84%.
  - **Why it gained more than at the old base,** per emitted token:

    | | Off | On |
    |---|---:|---:|
    | CPU experts | 9.49 | 2.48 |
    | Demand uploads | 32.29 | 13.23 |
    | CPU lane busy | 8.69 ms | 2.56 ms |
    | Host waiting on the CPU | 16.00 ms | 9.80 ms |
    | Early copies, used, waited on | | 49.35, 44.85, 42.69 |
    | Graph replays; captures | 56.89; 0.59 | 56.88; 0.56 |

    At CPU price 1.5 the off arm uploads far more than the old base at 1.3 did (32.3 against 22.2 demand uploads a token),
    so the early copy has more to take over. 95% of used copies were waited on, yet the gain doubled: a copy is nearly done
    when its reader arrives.
  - **Measured over the whole generation,** as frozen. The user asked on 2026-11-09 that timed tests measure from token 128,
    with caches warm. This load kept only each request's first and last engine samples, so its from-128 figure needs a run
    with the new token-128 sample. An interval this far below zero will not flip.
  - With drafting, `early_copy` acts only on one-row windows; the serving defaults change after its own drafting test.

**One graph per layer for every position, and timed tests from token 128, 2026-11-09 (6fc35b1, 0c2246d): verified.**
[Protocol](/mnt/wsl/transfer/minimax-wsl/runs/shape-warm-verify-20261109/make_protocol.py),
[result](/mnt/wsl/transfer/minimax-wsl/runs/shape-warm-verify-20261109/verify.txt).
- **The launch shape.** The window preparation's indexer now launches with the load's most blocks; its kernels already
  skip the blocks past each query's own, read on the device, so `prep_graph`'s key no longer changes every 128
  positions. The kernel test (`mm3_kernels_test`) found the same kept blocks and the same score bits at 96 windows of 1-16
  rows, 1 or 4 heads, on both sides of block edges.
- **On the engine,** four requests at the `T=1` base with `early_copy`, up to 1,024 tokens: each captured exactly 57
  graphs, one per MoE layer (a 1,024-token request captured 399 before; restoring the origin clears them, so it is 57 a
  request). 46,854 replays were checked bit for bit, prediction included, with no mismatch or fallback.
- **The token-128 sample.** The engine writes one sample at the first completed round at or after token 128 even with
  monitoring off, and the harness measures from it (`measure_from` in the protocol). Every request's interval started at
  exactly 128. On the plain replays the warm rate was 1.7-1.9% above the whole generation's (code 20.94 against 20.55
  tok/s, prose 23.67 against 23.27): the warm-up the old metric mixed in.
- Adopted: the launch shape is exact and removes the captures, and both run in every test from here.

<a id="per-pass-picture-new-base"></a>

**The per-pass picture of the new base, 2026-11-09 (order step 2): the 5090's serial chain is now the critical path; the CPU
lane has nearly left it.** [Protocol](/mnt/wsl/transfer/minimax-wsl/runs/perpass-new-base-20261109/make_protocol.py),
[last lanes](/mnt/wsl/transfer/minimax-wsl/runs/perpass-new-base-20261109/perpass-compare.txt),
[cycle parts](/mnt/wsl/transfer/minimax-wsl/runs/perpass-new-base-20261109/cycle-parts.txt). The ten requests of the HAGS
refit, light full-record captures on a fresh load of 0c2246d at the new `T=1` base (`early_copy`, `prep_graph`,
`helper_fast`, CPU price 1.5, horizon 40, bonus 0.20), up to 1,024 tokens, from token 128 on; beside the same requests on
the refit's load (none of those, price 1.3, horizon 20, bonus 0.10). Every comparison is inside one pass, from its origin.
- **Layer cycles fell 35-40%,** 57-67 to 35-44 ms a token.
- **The CPU lane is almost gone from the critical path.** CPU jobs fell from 0.22-0.47 to 0.014-0.077 a pass and demand
  uploads from 0.66-0.94 to 0.12-0.33. The CPU finishes last of the two lanes on 1-8% of passes, against 22-43%. Where
  both lanes run, CPU against helper is 0.90-0.98 ms apart, on few passes.
- **73-92% of passes run only the 3090's lane;** its median path from the lanes' start fell from 0.16-0.20 to
  0.115-0.138 ms (`helper_fast`). Primary work outlasted the lanes by over 0.1 ms on 19-42% of those passes, and on nearly
  all passes that run neither lane (6-20%).
- **Where a pass's cycle goes** (mean ms a pass, the four held-out workloads):

  | Phase | Old base | New base |
  |---|---:|---:|
  | Cycle | 1.02-1.15 | 0.63-0.72 |
  | Prepare: planning and submission, from the join | 0.135 | 0.108-0.120 |
  | Host lanes, mostly the 3090 | 0.39-0.50 | 0.156-0.176 |
  | Finish | 0.025 | 0.010 |
  | Resume: submitting the next window | 0.25 | 0.017 |
  | **Join wait: the host waiting for the 5090** | 0.22-0.24 | **0.34-0.40** |

  - The host now waits over half of each cycle for the 5090: the rest of this layer's experts, then the next layer's
    preparation (about 0.2 ms, a graph since `prep_graph`). The 3090's lane ends inside that.
  - The chain holds a host round trip: after routing returns, the 5090 cannot start this layer's experts until the host has
    planned and submitted them, within the 0.11-0.12 ms of prepare. On a critical 5090 chain that idle is on every pass.
- **For the order:** the levers are now the 5090's chain, by likely size: the round trip (the 5090's own experts
  dispatched on the device, perhaps from `plan_dev`), the preparation's projection kernels, and early-copy waits. A
  planner for the CPU's share matters less with the CPU this idle. Light records hold no GPU times, so an Nsight trace of
  plain generation (`prep_graph` and `early_copy` kept on, kernels identified by name, no NVTX needed) measures the
  5090's idle per pass next.

<a id="five-ninety-chain-trace"></a>

**The 5090's chain per pass, traced on the new base, 2026-11-09 (order step 2's GPU half).**
[Protocol](/mnt/wsl/transfer/minimax-wsl/runs/perpass-trace-new-20261109/make_protocol.py),
[analysis](/mnt/wsl/transfer/minimax-wsl/runs/perpass-trace-new-20261109/chain_analysis.py),
[result](/mnt/wsl/transfer/minimax-wsl/runs/perpass-trace-new-20261109/chain-analysis.txt). An Nsight trace (CUDA only) of
plain generation on the `T=1` base with `early_copy`, `prep_graph` and `dense_split`, so the graphs stay on: python-code and
prose at up to 512 tokens. Nsight records each replayed preparation graph as one graph event, which anchors each pass on the
5090's main stream. From token 128, means per pass (the trace slows the host, so cycles run 15-20% above untraced ones):

| On the 5090's main stream | Prose | Code |
|---|---:|---:|
| Preparation graph | 176 us | 177 us |
| **Idle from the graph's end to the first expert kernel** | **234 (median 166)** | **343 (median 201)** |
| Experts: first kernel to last | 316 | 357 |
| Tail to the next preparation | 25 | 16 |
| Cycle | 751 | 893 |

- **The host round trip is as long as the preparation.** On passes with no demand upload (88% of prose, 78% of code)
  the 5090 still idles 160-192 us after each preparation: the host sees the routing, plans, submits the 3090's work and
  then the 5090's. The light records put the host's part at 108-120 us untraced, so the 5090 likely idles about 100+ us
  a pass with no trace. A device-side dispatch of the 5090's own experts (its residents, from the routing on the device,
  as `plan_dev` already computes) would remove most of it: perhaps 6-13% a token, the largest lever found.
- **A demand upload blocks the main stream.** On passes with one (12% prose, 21% code) the idle reaches 780-856 us: the
  copy runs on the main stream, so even the resident experts' kernels wait behind it. Running residents before or beside
  the copy is a second lever (the earlier 0.8-0.9% copy-stream bound priced only the handoff, not this).
- **An early copy is still in flight when the experts begin on 81-93% of passes** (19,888-20,654 early copies on the copy
  stream), which stretches the experts' span (median 179-186 us, mean 316-357).
- **Inside the expert phase the 5090 computes only about 85-90 us**
  ([breakdown](/mnt/wsl/transfer/minimax-wsl/runs/perpass-trace-new-20261109/expert-breakdown.txt), means a pass): the two
  grouped `project` kernels 54-62 us, `combine_window` 11.3, `setup` 6.6-6.8, `scatter` 5.6-5.8, `prepare` 2.9-3.0,
  `add` 1.2. `wait_publication`, the 5090 waiting for the 3090's results, averages 64-68 us but is a tail (median share
  of the phase 3-8%). Against a median span of 179-186 us, about 90 us a pass is the stream idle between the 5090's own
  kernels: the host submits them one by one more slowly than the GPU runs them, and some wait for early copies.
- **Bound for an expert-phase graph** ([gaps](/mnt/wsl/transfer/minimax-wsl/runs/perpass-trace-new-20261109/gap-bound.txt),
  `gap_bound.py`): on passes without a demand upload (78-88%), the idle between the expert phase's consecutive kernels is a
  median 77-78 us a pass (p10 36-64, p90 97-114, traced). Copy waits, where the next kernel starts within 10 us of an
  early copy ending, are about zero. A graph removes the submission gaps; untraced they are smaller, perhaps 30-50 us a
  pass, about 4-7% of a token over 57 layers.
- ~~So the next lever is the expert phase's submission~~ *(superseded the same session by the next item)*: capture the
  5090's expert kernels as a graph per layer, as `prep_graph` did for the preparation.
- **But the 3090's path binds on 95-97% of passes, so an expert-phase graph would save about 1 us a pass**
  ([paths](/mnt/wsl/transfer/minimax-wsl/runs/perpass-trace-new-20261109/paths-bound.txt), `paths_bound.py`). On passes
  that run `wait_publication` and have no demand upload, measured from the preparation's end: the 5090 reaches the wait
  a median 270-319 us later and then spins for the host's publication of the 3090's results on 95.3% (prose) and 97.1%
  (code) of passes, median 42 and 6 us, p90 167-194 us. Removing the 5090's submission gaps (median 42-43 us before the
  wait) would only lengthen the spin. Not built. A sub-agent's read of the dispatch code (graph shape, `wait_val` in the
  plan's tail, the query kept outside) is kept for when the 5090 binds.
- **The 3090's own graph pays about 18 us for its host copies** (`helper_graph_bench.cu`,
  [result](/mnt/wsl/transfer/minimax-wsl/runs/helper-zc-20261109/helper-graph-bench.txt)). `helper_fast`'s graph
  captures two host-to-device copies (input rows, job descriptors), the grouped FFN and a device-to-host copy of the
  result. On the 3090, with a 60 us stand-in for the FFN, that shape spans 98.3 / 102.4 / 109.6 us at 1 / 2 / 4 jobs;
  with zero-copy kernels on the pinned buffers' device views 79.9 / 84.0 / 92.2 us: **-18.4, -18.4, -17.4 us a launch**.
  The helper graph runs on most passes and the 3090's path binds, so about 0.9 ms a token, roughly 2%. Built next, as
  `prep_graph`'s zero-copy fix was. In the trace the helper graphs span a median 134 us (p90 194-238).

<a id="helper-zero-copy"></a>

**`helper_zero_copy`, built and tested 2026-11-09 (65c47c3; order step 3, the 3090's path; adopted for `T=1`).** Run
folder `/mnt/wsl/transfer/minimax-wsl/runs/helper-zc-t1-20261109/` (protocol, frozen prediction, scripts, results).
- **What it does.** `helper_zero_copy=1` gives `helper_fast` a second set of per-job-count graphs on the 3090. Their inputs,
  job descriptors and results move through `copy_words` kernels on the pinned buffers' device views instead of three host
  copy nodes; switching rebuilds nothing. `helper_zero_copy=2` also checks every such pass: the three buffers are read back
  through the copy engine and compared with the bytes moved. Per request, off by default; serving defaults unchanged.
- **Exact.** The helper test (`mm3_native_helper_test`, on the 3090 and on the 5090): for 1 to 4 one-column jobs, with varied
  inputs and the nonzero expert between all-zero ones, the separate calls, the copy-node graphs and the zero-copy graphs give
  the same bits. In the load's check request every zero-copy pass matched: **17,577 checked, 0 mismatches**.
- **Bound and frozen prediction, before the run** (`saving_bound.py` on the new base's traces). A pass can save only what the
  5090 spins for the 3090's results beyond the spin's floor: min(18 us, spin - 5.2 us). That is a mean 7.1 us a pass for code
  (the full 18 us on 35% of passes, nothing on 10%) and 11.1 us for prose (57%, 10%): 0.41 and 0.63 ms a token. Frozen:
  **-1.16%** (code -0.85%, prose -1.53%).
- **The mechanism, per pass** (`mechanism.py`: light captures, two blocks per arm at up to 384 tokens, from token 128,
  every time from the layer's own join):

  | per pass, median | code off | code on | prose off | prose on |
  |---|---|---|---|---|
  | 3090 span (worker's launch to the event seen ready) | 187.7 us | 159.0 us | 191.3 us | 165.0 us |
  | 3090 ready, from the join | 244.1 us | 211.9 us | 241.6 us | 213.0 us |
  | the host's launch call (`result_path.py`) | 46.4 us | 20.4 us | 40.7 us | 16.9 us |
  | device and the worker's poll after the call | 140.5 us | 138.0 us | 149.7 us | 147.8 us |

  The span falls 28.7 and 26.3 us, more than the bench's 18, and the 3090's completion comes 32.2 and 28.6 us earlier.
  **Most of the gain is the host's launch call.** On the busy engine host, a graph with host copy nodes takes about 25 us
  longer to launch; the device side moves only 2-2.5 us. The central objective's per-layer metrics do not move: CPU against
  helper mean abs(delta) 0.897 to 0.930 ms (code) and 0.996 to 0.970 ms (prose), with slowest minus second slowest the
  same, over the few passes that have both lanes (775-868 code, 298-364 prose). At this base the CPU lane works on few
  passes, so the binding pair is the 3090 against the 5090's own path. In both arms the 3090 is the only lane on 74-90% of
  passes.
- **Timed test, twelve blocks**: each request ran from the restored origin at up to 384 tokens, so both arms cover nearly
  the same tokens 128-384. Under the frozen rule it is **-1.91% [-3.69, -0.13]** from token 128 (whole generation
  -1.91% [-3.18, -0.64]); code -1.70% [-4.40, +1.00], prose -2.11% [-4.57, +0.35]. The frozen prediction is inside the
  interval. The check passed and the mechanism was shown on both prompts: **adopted** into the `T=1` base.
  - **Three off-arm requests were slow, and they inflate the mean.** b5 and b12 code ran 9-10% slow and b10 prose 14%, while
    doing the same work as their siblings: the seeded texts repeat, so the counters are identical. Their stat lines point
    to the system (`slow_requests.py`). b5 shows doubled adaptive host time, planning +23% and the 5090's prefix kernels
    +13-15%; b12 shows planning +50% and helper submission +17%; b10 shows no field outside its siblings' range. The arms'
    medians are code 48.44 to 48.26 ms a token (-0.37%) and prose 41.94 to 41.58 (-0.86%). The median block is about -0.6%.
    The effect is real, as the mechanism shows per pass, but likely nearer -0.6 to -1.2% than the mean's -1.9%. As a sub-2%
    gain it also qualifies under the revised rule: it retires the copy-node graphs, and the result path builds on it.
- **The user's question: why not a stand-in that reads weights from the 3090's own memory?**
  ([result](/mnt/wsl/transfer/minimax-wsl/runs/helper-zc-20261109/helper-graph-bench2.txt), `helper_graph_bench2.cu`).
  It would have been the more faithful stand-in, so it was run. The stand-in streamed n expert records of 31.85 MB each,
  5× the L2, at 570-720 GB/s. The saving is the same as with the spin: **-18.5 / -18.4 / -16.3 us on the device** at
  1 / 2 / 4 jobs, and -20.6 / -19.3 / -16.3 us from the launch call to the completion seen. The launch call itself is about
  11 us shorter in the bench and about 25 us shorter on the engine's busy host.
- **The 3090's path, cut at its host boundaries** (`result_path.py`, the zero-copy arm, medians), for the next items:
  - the worker starts 44-55 us after the layer's join;
  - the launch call takes 17-20 us;
  - the device work plus the worker's poll take 138-148 us;
  - the worker's handoff takes about 0;
  - the collector's wake, BF16 conversion, weighting and merge take 8.3-9.4 us, p90 about 16; the mean, 22-46 us, includes
    passes where the host runs the CPU lane before it collects;
  - publication and the layer's end take 8-9 us.
  So host work after the device is about 9 us a pass. A device-side publication would also remove the worker's event poll
  inside "device and poll", which is not separated here.
- <a id="after-zero-copy-bound"></a>**After zero-copy, which path binds each pass, and the next cuts, bounded per pass the
  same day.** Offline, from the new base's traces (`perpass-trace-new-20261109`) with the 3090's results moved 29 us earlier,
  the light captures' measured shift (scripts `after_zc_bound.py`, `spin_vs_helper.py` and `ffn_fixed.py` in the run folder):
  - **The 3090's FFN** (from the HAGS traces of 2026-11-07, recorded before `helper_fast`, so its kernels are traced one by
    one, about 2,900 windows each): 22.2 us fixed + 48.5 us a job. The fixed part is the input quant 3.25 us, gate/up 10.3
    us, the activation quant 3.3 us and down 5.4 us. A job streams a 31.85 MB expert at about 660 GB/s, 70% of the 3090's
    936.
  - **The binding path now follows the 3090's job count:**

    | 3090 jobs (graph span) | share of passes, code / prose | 5090's spin, median | 3090 still binds after zero-copy |
    |---|---|---|---|
    | about 1 (< 110 us) | 19% / 13% | 5.3 / 5.4 us, the floor | 0% / 0% |
    | about 2 | 45% / 40% | 5.7 / 10.8 us | 11% / 34% |
    | about 3 | 27% / 32% | 34 / 82 us | 50% / 78% |
    | 4 or more | 9% / 15% | 123 / 166 us | 93% / 98% |

    Overall the 5090's own path binds on about 69% of code passes and 46% of prose passes. Mean |delta| between the two
    paths is 96 us (code) and 69 us (prose): per-pass imbalance in both directions.
  - **Next cuts, mean saving a pass** (a pass ends at the later of the two paths; 57 MoE layers a token):

    | cut | code | prose |
    |---|---|---|
    | the 3090 by 10 us a job (projections near peak bandwidth) | 7.0 us, 0.40 ms a token | 14.6 us, 0.83 ms |
    | the 3090 by 5 us a job | 3.8 us, 0.22 ms | 7.7 us, 0.44 ms |
    | the 3090 by 9 us flat (the two quant kernels fused) | 2.6 us, 0.15 ms | 4.6 us, 0.26 ms |
    | the 5090 by 30 us (expert-phase graph; 50 us gives the same) | 19.1 us, 1.09 ms | 11.7 us, 0.67 ms |
    | both, 10 us a job and 30 us | 30.2 us, 1.72 ms | 30.0 us, 1.71 ms |

    The 5090 cut is limited by how far its path now leads, at most about 29 us under this approximation. A pass whose
    3090 finished well before the 5090 arrived could gain more, so for 5090 cuts this is conservative. The light records
    and a zero-copy trace will confirm the split before the graph is built.
- **GPU caches** (the user's question, same day; `cudaGetDeviceProperties`): neither card has an L3. The 5090 has 96 MB of
  L2, up to 60 MB of it pinnable as persisting through access-policy windows. The 3090 has 6 MB, 4.1 MB pinnable. Each
  card has up to 100 KB of shared memory per SM, from the same store as L1. At `T=1` the costly bytes, every layer's dense
  projections and experts (31.85 MB each), are read once per token, gigabytes a token. No L2 can hold them across tokens,
  and the 3090 streams each expert through an L2 a fifth its size. The kernels already use L1, shared memory and L2 within
  and between consecutive kernels. Pinning could pay for data reused every step that fits: possibly the draft model's
  weights at step 6, to be checked then.

<a id="expert-graph"></a>

**`expert_graph`, built and tested 2026-11-09 (afce80c; task 58): exact, but no measured gain at `T=1`, kept off.** Run
folder `/mnt/wsl/transfer/minimax-wsl/runs/expert-graph-t1-20261109/`.
- **What it does.**
  - Each generation window's expert phase on the 5090 is captured once per window size and wait variant, and replayed.
    The capture holds the plan's copy (a `copy_words` kernel on the pinned plan's device view), the grouped expert
    kernels, `wait_publication`, the combine and the residual add.
  - The wait word's expected value, the one argument that changes every pass, rides in the plan's free tail.
    `wait_publication` reads it from there (`WinArgs::wait_val_ptr`), so no layer is captured: 2 graphs a load.
  - The design came from a sub-agent's read-only study of the dispatch code.
- **Exact.** In mode 2 every replay is checked after the publication against the launches: 21,831 replays, 0
  mismatches, 0 fallbacks.
- **The mechanism.** Light captures, medians per pass from token 128:
  - The host finishes submitting the 5090's expert work **58.6 us (code) and 62.2 us (prose) earlier**.
  - The 3090's collection comes 6.6 and 14.5 us earlier, and the layer's host end 15.1 and 21.9 us earlier.
  - **The join-to-join cycle does not move: -4.4 and +0.6 us a pass.**
- **The timed test.**
  - Twelve blocks at up to 384 tokens give +1.85% [-3.24, +6.94] against a frozen -1.96%. The arms' medians are code
    +0.85% and prose -0.16%. Under the rule it **stays off**.
  - Block 2's on-arm code request ran 47% slow. Its CPU lane took 36.6 ms a token against its siblings' 4.2: system
    contention, not the graph.
  - **The run was contaminated.** Blocks 1-5 (and the mechanism captures) overlapped two research sub-agents' local
    tool work: greps over large repositories and a SQLite query. Blocks 6-12, after both finished, show a consistent
    -1.38% per block (code -0.24%, prose -0.73% averaged per prompt). The result is inconclusive rather than negative.
    The graph stays built, off by default, to be re-timed cleanly, and with drafting, where verify windows submit 5T
    launches a window.
- **Why the bound was wrong.** The bound assumed the 77 us of traced idle between the 5090's expert kernels was device
  idle that a graph removes. On WSL2 the traced gaps are inflated: Nsight flushes each launch. Untraced, the submission
  query after the launches flushes them as one batch, and the 5090 is then still busy with its preparation. So the
  graph frees host time (about 60 us a pass) without shortening the 5090's device timeline. The asymmetry with the
  3090 is the lesson: the 3090 sits idle until its launch arrives, so `helper_zero_copy`'s faster launch call reached
  its path.
- **Method rules from this run.**
  - Device-side gaps in Nsight traces on WSL2 are not evidence of device idle without an untraced check (events or the
    light records' cycle).
  - No sub-agent with local file scans (grep, SQLite, Python over large files) runs during a timed test: their CPU use
    perturbs the CPU lane, as any CPU-heavy work does.

<a id="preparation-projections"></a>

**The preparation's projections, bounded 2026-11-09 (order step 3; a sub-agent's read-only analysis of the HAGS-on t1-code
Nsight export, 3,648 one-row windows, checked against the code).** Scripts in `/home/mdwbr/.claude/jobs/8b7e216b/tmp/projbound/`.
- **The earlier "eight launches at 18.4 us" was one median pooled over all eight;** per launch they differ. Weights are
  MXFP8 (E4M3 with one E8M0 scale per 32, 8.25 bits a weight):

  | Projection (stream) | N × K | Blocks | MB | Median us | TB/s | % of 1.79 |
  |---|---|---:|---:|---:|---:|---:|
  | q (main) | 8192 × 6144 | 256 | 51.9 | 34.8 | 1.49 | 83% |
  | k (main) | 512 × 6144 | 16 | 3.24 | 16.9 | 0.19 | 11% |
  | v (main) | 512 × 6144 | 16 | 3.24 | 16.4 | 0.20 | 11% |
  | attention output (main) | 6144 × 8192 | 192 | 51.9 | 35.0 | 1.48 | 83% |
  | shared gate (main) | 3072 × 6144 | 96 | 19.5 | 19.6 | 0.99 | 55% |
  | shared up (main) | 3072 × 6144 | 96 | 19.5 | 18.6 | 1.05 | 58% |
  | shared down (main) | 6144 × 3072 | 192 | 19.5 | 14.8 | 1.31 | 73% |
  | indexer q (side, hidden) | 512 × 6144 | 16 | 3.24 | 16.0 | 0.20 | 11% |

  The main stream's seven take 156.9 us of the ~202 us replayed preparation.
- **Why the small ones are slow:** `mx_rows_kernel` walks K in 512-wide tiles with no prefetch, one memory round trip a tile
  (about 1.4 us). A launch too small to fill the bandwidth has a floor of about 17 us at K = 6144, whatever its N.
- **The split path exists but stops at 256 outputs:** `native_dense::mv` gives split-K partials only for `m.n <= kBand`
  (256; `mm3_native_dense.cu`), so only the indexer's k (128) takes the `mxc` path. The DSpark draft's k / v of this very
  shape run it in 3.98 + 1.06 us. Partials for 512 outputs need at most 96 KB (`mxc`, 8 rows) or 128 KB (the split
  kernel, 16 rows), within the 256 KB scratch a stream (checked here; the sub-agent's 393 KB figure was off, its
  conclusion holds).
- **Bounds per MoE layer:** k and v through `mxc` -21 to -23 us (about -2.7% a token); with q/k/v and shared gate/up
  fused into fewer launches, -38 to -41 us (about -5%); full bandwidth would save 62.8 us. The indexer chain must stay
  on the side stream. With early copies waited on, passes where the copy path binds may absorb part of it.
- **Built first: the cheap step,** as the per-request option `dense_split` (1: split K up to 512 outputs). Changing it
  drops captured graphs; the summation order of k, v and the indexer's q changes, within FP32 precision and on the same
  path the indexer's k and the draft's k / v already take, so no quality mechanism.
- **Exact to F32 rounding:** the kernel test now runs `mx_rows` at 512 × 6144 with and without the split, for 1, 2, 5, 8
  and 16 rows. The worst difference between them is 8.6e-8 relative to the sum of the terms' magnitudes, and both match an
  F64 reference to 6.8e-8.
- **Timed test, 2026-11-09: the frozen rule says adopt, but its pooled effect is mostly a length mix**
  ([protocol](/mnt/wsl/transfer/minimax-wsl/runs/dense-split-t1-20261109/make_protocol.py),
  [evaluation](/mnt/wsl/transfer/minimax-wsl/runs/dense-split-t1-20261109/evaluation.txt)). The check request passed
  (21,774 replays bit for bit, no mismatch or fallback).
  - Pooled over both prompts, from token 128: **-4.04% [-4.71, -3.37]**; over the whole generation -1.81% [-2.43, -1.19];
    frozen -2.83%.
  - **Per prompt: code +0.65% [-0.85, +2.14], prose -2.71% [-5.05, -0.37].** A pooled effect beyond both is the mix:
    the arms wrote different texts, since the summation order changes. The on arm's code answer was the same 382-token
    direct answer, with no reasoning, in 5 of 6 blocks, coherent and correct as far as read. The off arm reasoned for
    700-1,408 tokens. Pooled ms per token then weighs the faster prose more in the on arm.
  - Even per prompt the arms compare different texts over different stretches of generation, which the block-to-block
    interval cannot see.
  - **Lesson for the method:** for a change that alters the text, report per-prompt effects and their equal-weight mean,
    and measure the mechanism directly. Here that is each layer's preparation span on the 5090, timed with events
    (`dense-split-prep-20261109`), which decides adoption together with this test.
- **The mechanism, measured: -22.5 us per layer [-23.3, -21.7], so `dense_split=1` joins the `T=1` base**
  ([protocol](/mnt/wsl/transfer/minimax-wsl/runs/dense-split-prep-20261109/make_protocol.py),
  [result](/mnt/wsl/transfer/minimax-wsl/runs/dense-split-prep-20261109/span-analysis.txt)). One load with
  `MM3_PREPARE_EVENTS=1`, four blocks of both arms on code and prose at up to 384 tokens:
  - each one-row MoE layer's preparation span on the 5090 fell from 222.7-225.5 to 200.7-203.2 us, in all eight
    prompt-block pairs, as bounded (about 22 us); the host's enqueue time did not move (+0.5 us [-0.8, +1.8]);
  - here both prompts ran to the same 384 tokens, and the on arm was faster in both from token 128: code 49.70 to 48.14
    ms a token (-3.1%), prose 43.26 to 41.30 (-4.5%). The texts still differ, so those are supporting figures;
  - the join wait fell on prose (-7 to -44 us a layer) but not on code (+2 to +16): it also holds each layer's expert
    work, which follows the text.
  - **Decision:** it passed the frozen rule, its mechanism is measured exactly as bounded on the critical-path
    preparation, it is exact to F32 rounding, and it has no quality mechanism. The serving defaults stay as they are
    until step 6 (speculative decoding waits for it).

**`prep_graph`, first form, 2026-11-08 (793c78d): the preparation replays bit for bit as a CUDA graph but gets no shorter.**
**Superseded the same day:** this entry first concluded that the idle is on the GPU side and that a graph's boundaries cost
what queued launches do. Microbenchmarks then showed graphs do cut dependent boundaries; the first form lost that saving to its
five host copies, which as graph nodes cost about 13 us each (next entry).
[Protocol](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-verify-20261108/make_protocol.py),
[analysis](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-verify-20261108/pg-analysis.txt),
[idle attribution](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-verify-20261108/prepare-gaps.txt).
- **What it is.** A per-request option, off by default. Each one-row MoE layer's preparation (attention, the router and the
  shared expert on both streams, and the readbacks) is captured once as a CUDA graph and replayed.
  - The host builds the ready mask before each replay, as `route_layer` does.
  - A graph is keyed by every captured argument that can move. It is captured again when the indexer's block count changes,
    every 128 positions.
- **Verification passed.** `prep_graph` 2 checked every replay against the launches, bit for bit, on the layer's residual,
  normed input, shared expert and routing.
  - 19,437 replays on python-code and 34,485 on prose, with no mismatch and no fallback.
  - All 12 requests of the load, in all three modes, wrote the same text for their prompt.
- **Timing within the load** (`MM3_PREPARE_EVENTS=1`, per one-row MoE layer, means of two blocks, the order alternating):

  | Prompt, arm | Span on the 5090 | Host enqueue | Join wait | ms per token |
  |---|---:|---:|---:|---:|
  | Code, off | 278 us | 262 us | 268 us | 63.08 |
  | Code, on | 285 us | 98 us | 441 us | 63.56 |
  | Prose, off | 282 us | 250 us | 247 us | 54.14 |
  | Prose, on | 287 us | 95 us | 409 us | 55.60 |

  - The host saves about 160 us a layer, and its join wait grows by as much. Enqueue plus wait is about 0.5 ms in both arms, so
    the 5090 ends the preparation at the same moment either way.
  - The span grows by 6 us. (The reason, found next: the graph's five host copy nodes.)
  - Per token, on is +1.73% on average: +0.20% in both second blocks, +1.33% and +5.21% in the first.
  - Re-capture costs about 0.2 ms a graph, 34-64 ms a request. Each request from the restored origin starts at block counts the
    previous one dropped.
- **Per-pass balance is unchanged**, as expected for a change ahead of every path. CPU against helper abs(delta) is 0.775 ms in
  both arms (n 3,665; per-layer median 0.772 off, 0.767 on), and slowest minus second the same.
- **Decision.** The frozen-prediction method (span change times timed layers a token) gives +0.57%, slower. So the prepared timed
  test (`prep-graph-t1-20261108`) was not run, and `prep_graph` stays off. The code stays, off by default.
- **Where the idle is** (HAGS-on traces, one-row MoE windows): a median 68 us on code and 63 us on prose, in spans of 280-286
  us, over 16-17 gaps a window.
  - Most gaps are same-stream kernel boundaries of 2-4 us: between projections (median 3.5 us), rope to attention (3.1),
    projection to activation (3.9), add to norm (3.0).
  - The largest single gap is the side-stream fork after the first norm: a median 17-19 us in 58% of windows. The tracer's
    host overhead likely inflates it.
  - Gaps at a stream change are 27% of the idle, and gaps beside a small copy 8-11%.
- **What could shorten it** (as first written; superseded by the next entry): fewer kernel boundaries (fusion) or overlapped
  ones (programmatic dependent launch). The microbenchmarks found PDL does nothing on this platform and graphs do the job.

**`prep_graph` with zero-copy capture, 2026-11-08 (842faac): -6.99% [-8.84, -5.14] at `T=1`, adopted under the frozen rule
pending its usability check.**
Microbenchmarks ([boundaries](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-verify-20261108/pdl_bench.cu),
[the preparation's shape](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-verify-20261108/prep_shape_bench.cu)),
[verification](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-zc-verify-20261108/pg-analysis.txt),
[timed protocol](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-t1-20261108/make_protocol.py),
[evaluation](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-t1-20261108/evaluation.txt).
- **What a dependent boundary costs on the 5090** (WSL2, HAGS on): a chain of 20 GEMVs, each reading the previous one's
  output, with every launch queued before the timed part starts.

  | GEMV | Plain launches | PDL | PDL, weights loaded first | Graph |
  |---|---:|---:|---:|---:|
  | 4096 x 4096 int8 (16.8 MB) | 16.33 us a kernel | 16.33 | 18.48 | 13.67 |
  | 512 x 512 int8 (0.3 MB) | 5.55 us a kernel | 5.50 | 6.50 | 1.68 |

  So a graph cuts each dependent boundary by 2.7-3.9 us even when the host is ahead. Programmatic dependent launch does nothing
  here, and loading weights before its wait costs registers.
- **Why the first form failed.** The preparation's shape: 22 kernels on two streams with two forks and joins, one mask upload
  and four readbacks, as stream launches against a graph.

  | Variant | Stream launches | Graph |
  |---|---:|---:|
  | Two streams, host copies (the engine's shape) | 162.2 us | 178.9 us |
  | Two streams, copies as kernels on mapped pinned memory | 156.4 us | 111.2 us |
  | One stream, host copies | 193.7 us | 140.2 us |
  | One stream, copies as kernels | 193.5 us | 118.2 us |

  In a two-branch graph the host copy nodes cost about 13 us each, which undid the boundaries the graph saved.
- **The fix.** While a preparation is captured, its five host copies are `copy_words` kernels on device views of the same pinned
  buffers. The readbacks share one kernel. The host reads after the same stream synchronization, and the stream path is unchanged.
- **Verification passed.** 21,831 replays on code and 35,625 on prose, bit for bit, with no mismatch or fallback, and every
  capture zero-copy.
- **Within the verification load** (per one-row MoE layer, two blocks):
  - the span on the 5090 fell from 280 to 202 us (-78 us, -28%), and the host's enqueue from 255 to 29 us;
  - ms per token fell 7.08% (-5.55% to -7.99% over the four block-prompt pairs);
  - per-pass balance did not move: CPU against helper abs(delta) 0.788 ms off, 0.781 on.
- **Timed test.** One fresh load, six blocks, the `T=1` base with `helper_fast`, up to 2048 tokens. Frozen prediction: -7.71%,
  from the span change times the timed layers a token.
  - Result: **-6.99% [-8.84, -5.14]**, the prediction inside. Code -6.45% [-7.77, -5.13] (predicted -7.15%), prose -8.03%
    [-10.09, -5.96] (predicted -8.26%).
  - Every block is faster (-3.83% to -8.52%), and none is flagged by the stall check.
  - Off ran at 56.9-58.9 ms per token (17.0-17.6 tok/s), on at 53.3-54.7 (18.3-18.8 tok/s).
  - Re-capture: 0.53 graphs a token, 0.07 ms a token.
- **Text.** The computation is unchanged, but a shorter pass can see an asynchronous helper-live publication a token earlier or
  later. The residency bonus then picks among ready experts differently.
  - In the verification load, 2 of 6 code requests with graphs wrote another text, with 76 helper-live publications instead
    of 77.
  - In the timed test, 5 of 12 pairs match. The off arm itself wrote 4 distinct texts per prompt over the six blocks at this
    budget, and the texts recur across arms. So the variation is the base's own.
  - The plan's rule still asks for a usability check of anything that changes output. It runs next
    (`prep-graph-usability-20261108`, the early-copy check's matrix and rule).
- **Decision.** Under the frozen rule `prep_graph` is adopted for `T=1`, pending the usability check. Its drafting-window
  extension (f8e2ff8) is built and goes through its own verification and test.

**Usability check of `prep_graph` at `T=1`, 2026-11-08: it passes, so `prep_graph` joins the `T=1` base.**
[Protocol](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-usability-20261108/make_protocol.py),
[evaluation](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-usability-20261108/evaluation.txt),
[every final answer](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-usability-20261108/review-dump.txt).
- **Design.** The early-copy check's matrix, budgets, seeds and rule, at the `T=1` base with `helper_fast`. One fresh load of
  842faac, each entry from the restored origin, arm order alternating by seed. The primary agent read every final answer.

| Task, passes of 3 seeds | Off | `prep_graph` |
|---|---:|---:|
| Scheduling: 11 and 12, valid schedules and bounds | 3 | 3 |
| Muffins: 3 combinations | 3 | 3 |
| Action items, complete within 1,024 tokens | 2 | 2 |
| Log summary, no added facts | 3 | 3 |
| `order_jobs`, executed checks | 3 | 3 |
| `count_runs` first attempt | 3 | 3 |
| `count_runs` return | 3 | 3 |
| Tool round trip, disconnect and follow-up | 3 | 3 |
| Choir note, complete within 1,024 tokens | **2** | 3 |

- **Every failure in either arm is a token-budget truncation:** an empty action-items answer in each arm (off seed 2,
  `prep_graph` seed 1), and off's seed-2 choir note cut off mid-sentence.
- **No output shows repetition, broken text or a logic collapse.** Both arms have small slips:
  - off: one muddled slack explanation, and one claimed machine idle gap that its own schedule does not have;
  - `prep_graph`: one garbled word ("mid-rehehalf") in an otherwise coherent choir note, and one note that both waives and
    asks for robes;
  - choir notes in both arms invent specific halls and buses.

  The garbled word is judged a slip of the same kind as off's, not broken text.
- **Rule, fixed before the run:** `prep_graph` fails if, on any task, it passes on fewer seeds than off, or shows such damage
  that off does not. Neither holds.
- **Decision.** `prep_graph` joins the `T=1` base. Three seeds cannot establish quality preservation. The serving defaults
  follow its drafting test.

**`prep_graph` in drafting windows, 2026-11-08 (f8e2ff8): verified bit for bit at the serving configuration, -3.80% within the
load.** [Protocol](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-spec-verify-20261108/make_protocol.py),
[analysis](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-spec-verify-20261108/spec-analysis.txt).
- **What changed.** Every generation window up to 16 rows replays from a graph, one per window size, captured again at each
  new indexer block count. A prompt's chunks launch as before.
- **Verification passed** with drafting: 3,876 replays on code and 9,804 on prose, bit for bit, no mismatch or fallback, every
  capture zero-copy.
- **Within the load** (drafting 5, two blocks, order alternating): -3.80% on average, -2.79% to -4.74% over the four
  block-prompt pairs. Every pair wrote the same text, with the same drafting rounds, CPU experts and staged experts.
- **Re-capture costs more with drafting:** a graph per window size and block count, 969 captures in a 402-token prose request,
  about 0.33 ms a token (0.7%). Reading the block count on the device would remove it, but that bound (0.35-0.7% with drafting,
  0.13% at `T=1`) is under the 2% rule.
- **Next:** the timed test at the serving configuration (`prep-graph-spec-20261108`), frozen prediction -3.80% from this load.

**Timed test of `prep_graph` with drafting, 2026-11-08: -4.39% [-6.14, -2.64] at the serving configuration, so it qualifies for
the serving defaults after a serving usability check.**
[Protocol](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-spec-20261108/make_protocol.py),
[evaluation](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-spec-20261108/evaluation.txt).
- **Design.** One fresh load of f8e2ff8, six blocks of both arms on python-code and prose at up to 2048 tokens, the order
  alternating, at the serving configuration (drafting 5, lenience 0.4, draft confidence 0.3, bonus 0.20, CPU price 1.3,
  horizon 20, spinner on, query off, `helper_fast` off, HAGS on). Frozen prediction: -3.80%, the verification load's
  within-load effect; there is no per-pass model with drafting.
- **Result:** -4.39% [-6.14, -2.64], the prediction inside. Code -3.47% [-4.22, -2.72] (predicted -3.02%), prose -4.95%
  [-7.74, -2.15] (predicted -4.58%).
  - Every block is faster. Block 3's -7.74% is inflated by one slow off-arm prose request, which the stall check flags
    (50.97 ms per token against a median of 47.71). The other five blocks lie between -3.38% and -4.19%.
  - Off ran at about 47.5 ms per token (21.0 tok/s), on at about 45.8 (21.8 tok/s).
- **Same text.** All 12 pairs wrote identical text (one text per prompt over all 24 requests), with the same work: 22.48 CPU
  experts and 26.34 staged experts a token in both arms.
- **Cost of the graphs:** 2.05 captures and 0.30 ms a token, and 20.2 replays a token.
- **Decision.** Under the rule fixed before the run, `prep_graph` goes into the serving defaults after a usability check at
  the serving configuration (`prep-graph-serving-usability-20261108`), which the rule asks for although the text did not change
  here.

**Serving usability check of `prep_graph`, 2026-11-08: it passes, so `prep_graph` joins the serving defaults.**
[Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-serving-usability-20261108/evaluation.txt),
[every final answer](/mnt/wsl/transfer/minimax-wsl/runs/prep-graph-serving-usability-20261108/review-dump.txt).
- **Design.** The 1.5/40 check's matrix, budgets, seeds and rule, at the serving configuration with drafting, on f8e2ff8. The
  primary agent read every final answer.

| Task, passes of 3 seeds | Off | `prep_graph` |
|---|---:|---:|
| Scheduling: 11 and 12, valid schedules and bounds | 3 | 3 |
| Muffins: 3 combinations | 3 | 3 |
| Action items, complete within 1,024 tokens | 2 | **3** |
| Log summary, no added facts | 3 | 3 |
| `order_jobs`, executed checks | 3 | 3 |
| `count_runs` first attempt | 2 | 2 |
| `count_runs` return | 2 | 2 |
| Tool round trip, disconnect and follow-up | 3 | 3 |
| Choir note, complete within 1,024 tokens | 1 | **2** |

- **Every failure is a token-budget truncation**, except seed 2's `count_runs` return. The checker fails that answer on its
  first code block's `object()` sentinel (the known artifact). The answer also leaks a draft, its reasoning and a stray
  `</think>` before a correct final block, and it is character for character the same in both arms.
- **Seed 2 is identical in both arms in every task.** Seed 3 is identical through scheduling and `count_runs`, then diverges.
- **No damage that off does not show.** The reasoning leak is in both arms. Both arms have small slips: a self-correction
  mid-sentence, a misstated idle gap, and a stray bracket.
- **Decision.** `prep_graph` passes, and `prep_graph: 1` joins the serving defaults (`native-runtime.json`). Three seeds cannot
  establish quality preservation.

**Calibration noise, from the same load's repeats: CPU noise within one load matches the spread between loads, and a cold
5090 can wreck a table.** [Reports in order](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-hags-20261108/calibration-repeats.txt).
- **What ran.** The engine calibrates at load and again after the first prefill, so twelve repeats gave 25 reports.
- **CPU side.**
  - Among the 13 reports with a normal H2D, the one-column single-job CPU time spans 0.533-0.669 ms, median 0.600.
  - Ten earlier loads spanned 0.603-0.716.
  - So most of the load-to-load difference in the planner's CPU price is measurement noise, not a real difference between
    loads. A median over several measurements would steady the planner's single-miss threshold.
- **GPU side.** From the 18th report on, every calibration caught the 5090 slow:
  - H2D about 4.4 ms, or 7.2 GB/s against about 52;
  - the grouped-expert extra 0.45-0.86 ms.
  - Earlier reports were mostly normal, with scattered slow ones.
  - The long sequence is mostly CPU measurements, which leaves the 5090 idle between its short timed bursts.
- **Normal loads are safe so far, but not guaranteed.** A single calibration right after the lane warm-up was normal on 10 of
  10 loads. But a table taken cold would leave a load with no uploads, about 25% slower, and nothing guards against that
  today.
- **Fix to build:** warm the 5090 right before its measurements, take the median of several, and re-measure any H2D far above
  the link's expected 0.6 ms. The repeat diagnostic stays off in timed runs.

**Late copies, 2026-11-08: under 1% of passes, even when a copy may queue behind a demand copy.**
[Script](/mnt/wsl/transfer/minimax-wsl/runs/predict-diag-20261108/prefetch_late.py),
[result](/mnt/wsl/transfer/minimax-wsl/runs/predict-diag-20261108/prefetch-late.txt).
- **Method.** The first bound priced every early copy as on time. This pricing holds the receiving pass's kernels until the
  copy lands. The landing time is the issuing pass's copy end with one more copy, by the model's sharing rule. The receiving
  pass's origin follows from the issuing pass's cycle and both passes' preparation.
- **Result:**
  - the engine's rule is never late: -8.01%, unchanged;
  - behind one demand copy, the copy is late on 0.93% of passes, which moves the bound from -14.65% to -14.62%.
- **So `early_copy` mode 2 is built** (6737181): it also copies behind the layer's one demand copy, never with an exchange or
  promotion in flight.
- **Not modelled:** the copy sharing the copy engine with the next layer's own demand copies.
- **Its test.** A timed test against mode 1 decides
  ([protocol](/mnt/wsl/transfer/minimax-wsl/runs/early-copy-mode2-20261108/make_protocol.py)). It runs only if mode 1 is
  adopted, after its own frame verification. Then one usability check covers whichever mode wins.

### Rules for adoption, revised

**User direction, 2026-11-08: a usability check only where a change could lower quality; cut-offs are not results; small
gains with other benefits are adopted.** The rules are in the [plan](README.md#adoption-rules).
- **The user's reasons.** The engine is not designed to be repeatable: it prioritizes speed. The same experts run with
  different arithmetic on different devices, and routing depends on the VRAM cache state, which is highly variable during real
  inference. All of this is desired and part of the correct design. If a change is faster, the burden of proof is to show that
  it lowers usability, not that usability is kept. The answer is not longer or more usability tests. It is to think about the
  change: with no mechanism for loss of quality, no check is needed before it joins the defaults. And token-budget cut-offs
  must not count as quality regressions.
- **Earlier checks, read again under the new rules:**

  | Change | What decided its check | Under the new rules |
  |---|---|---|
  | CPU price 1.5, horizon 40 | one seed fewer on two tasks, every failure a cut-off; router mass kept unchanged | no quality mechanism: **restored** to the defaults |
  | `early_copy` mode 2 | two choir notes cut off by a 1,024-token budget | no quality mechanism: **admissible**, re-timed on the new base after integration with `prep_graph` |
  | `prep_graph`, `T=1` and serving | passed | no quality mechanism: the checks were not needed |
  | Bonus 0.30 (2026-11-07) | one empty answer, a cut-off | it moves picks away from the router's choice: a quality mechanism, so a fair check with the [template](usability-matrix-template.json)'s budgets whenever it is reconsidered |
  | Bonus 0.30 with lenience 0.3 | failed the same matrix | lenience accepts draft tokens the target is less sure of: a mechanism, so the same |

- **The 2% rule, revised.** `helper_fast` with drafting, -0.58% [-1.19, +0.03] with five of six blocks faster, joins the serving
  defaults: one setting for `T=1` and serving is simpler to run and reason about. The other candidates bounded under 2% are
  re-assessed for a benefit beyond speed.
- **Applied.** `native-runtime.json` now holds CPU price 1.5, horizon 40, `helper_fast` 1 and `prep_graph` 1. Usability checks
  use [usability-matrix-template.json](usability-matrix-template.json): every task at 6,144 tokens and each code turn at
  4,096, with cut-off answers re-run or left out.

### Contention between the CPU lane and demand uploads

**2026-11-08: both slow each other, and on CPU passes the two paths end together, bound by host memory bandwidth.**
[Measurement](/mnt/wsl/transfer/minimax-wsl/runs/contention-bound-20261108/upload-contention.txt) from the HAGS-on traces
(per-pass-trace-hags-20261107, CUPTI copy times, one-row generation passes with one upload).
- **The upload's side.** A 31.85 MB demand copy takes 554 us alone (57.5 GB/s, near PCIe 5.0 x16's limit) and 748 us when the
  pass has a CPU job (+35%, 42.6 GB/s), in both workloads.
- **The CPU's side** ([per job](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-hags-20261107/hags_cpu_per_job.txt)). One CPU
  job takes 0.548 ms with no upload beside it and 0.806 ms with one, in two-row windows (+47%). At `T=1` at price 1.3 every
  one-job pass has an upload, and its batch takes 0.844-0.854 ms.
- **Together** they read about 80 GB/s from host memory, about what it delivers. The copy-plus-primary path (the copy, the
  handoff and the grouped kernels, about 0.80-0.83 ms) and the CPU lane (0.85 ms) end within about 0.03-0.05 ms.
- **Each path alone is capped below the shared cap** (user correction, 2026-11-08). The upload's limit alone is PCIe, about 57.5
  GB/s. One CPU job reads at about 58 GB/s alone (31.85 MB in 0.548 ms). Only together do they reach the shared ~80 GB/s.
  - So removing a CPU job lets a concurrent upload rise only from 42.6 to 57.5 GB/s, and the pass's host throughput falls
    from about 80 to 57.5 GB/s.
  - Moving work from the CPU to uploads trades two parallel paths for one capped path. That explains why the CPU lane's
    1.2-2.9 ms saving per token at 1.5/40 bought the token only 0.4-0.5 ms, and why the per-pass model, without these caps,
    over-credited it.
  - For the refit: each path's rate is the lesser of its own cap (upload about 57.5 GB/s, CPU about 58 GB/s) and its share of
    the shared ~80 GB/s while both read.

**Host-bandwidth profile at several splits, 2026-11-08: the hardware splits about 79 GB/s nearly evenly between the CPU and the
uploads, whatever the CPU's thread count.**
[Benchmark](/mnt/wsl/transfer/minimax-wsl/runs/contention-bound-20261108/contention_bench.cu),
[result](/mnt/wsl/transfer/minimax-wsl/runs/contention-bound-20261108/contention-bench.txt),
[DRAM check](/mnt/wsl/transfer/minimax-wsl/runs/contention-bound-20261108/dram-check.txt). At the user's request: the rates
"beside each other" at various splits.
- **Method.** 31.85 MB expert-sized records with distinct data, in pinned memory rotating over 2 GB. CPU readers stream them as
  the lane does, one record at a time split over the threads. Uploads go to the 5090 on one or two streams, and to the 3090.
  Each case runs for a second, after a warm-up.
  - Two first versions were invalid and are kept as records. All-zero records let the CPU "read" 140 GB/s, and so did a
    510 MB rotation at 15 threads, partly served from the 96 MB L3.
  - The CPU readers do no decode, so they are an upper bound on the real kernel. They match the engine's traces: alone
    58-62 GB/s against the traced 58, and side by side 37-40 and 40-41 against the traced 37.5 and 42.6.

  | Case | CPU | 5090 upload | 3090 upload | Total |
  |---|---:|---:|---:|---:|
  | CPU lane alone (15 threads) | 61.8 | | | 61.8 |
  | CPU alone, 1 / 2 / 4 / 8 threads | 53.5 / 57.6 / 58.3 / 58.9 | | | |
  | One 5090 upload alone | | 57.4 | | 57.4 |
  | Two 5090 uploads at once | | 57.4 | | 57.4 |
  | One 3090 upload alone | | | 6.3 | 6.3 |
  | 5090 and 3090 uploads | | 52.1 | 6.4 | 58.4 |
  | CPU 15 and one 5090 upload | 37.1 | 40.8 | | 77.9 |
  | CPU 15 and two 5090 uploads | 37.9 | 41.2 | | 79.1 |
  | CPU 15, 5090 and 3090 uploads | 38.4 | 31.6 | 6.4 | 76.3 |
  | CPU 2 / 4 / 8 / 12 and one 5090 upload | 39.6 / 37.4 / 44.6 / 37.7 | 39.6 / 40.9 / 39.6 / 41.3 | | 79-84 |
  | CPU 1 thread and one 5090 upload | 26.5 | 47.4 | | 73.9 |

- **Each path's own cap.** The CPU reads about 58-62 GB/s, reaching it from two threads. One upload to the 5090 reaches 57.4
  GB/s, and a second at once adds nothing, so PCIe is full with one. The 3090's link runs at PCIe 4.0 ×4 (`nvidia-smi`: gen 4,
  width 4 of 16), 6.4 GB/s.
- **The shared cap is about 79 GB/s, split nearly evenly**: about 38 for the CPU and 41 for the upload, from 2 to 15 CPU
  threads. Only a single CPU thread shifts it, to 26.5 and 47.4.
- **The user's example, one upload beside two CPU jobs.** The lane runs one expert at a time across all 15 threads
  (`native_cpu_run_multi`, `lane.parallel_for`). So two jobs keep it reading twice as long at the same rate, and the upload
  gets the same ~41 GB/s while they overlap. More reading threads would not take more from it either.
- **The 3090's refills cost the 5090's uploads.** Beside the CPU, a concurrent 3090 upload lowers the 5090's from 40.8 to 31.6
  GB/s (-23%). Alone it lowers it from 57.4 to 52.1. The two DMA streams share about 58 GB/s.
- **The planner does not model any of this.** `plan_batch_staging` prices the CPU, the 5090 and the 3090 as independent paths,
  each upload at the calibrated 0.61 ms alone and each CPU job at its calibrated cost alone, times the CPU price.
  - The CPU price stands in for contention. Its value 1.5 happens to match the CPU's measured slowdown beside an upload, 58 to
    38 GB/s (×1.53). But it applies even when no upload runs beside the CPU, and it never charges the upload's own ×1.40.
  - In the `T=1` cases worked through (two and three misses), contention changes the predicted times more than the decisions.
    With two misses, one CPU job beside one upload ends at about 0.84 ms, against about 1.1 ms for two uploads or two CPU jobs.
- **For the refit:**
  - while a CPU job and an upload overlap, they run at about 38 and 41 GB/s;
  - when one finishes, the other returns to its own cap;
  - a second upload never adds bandwidth;
  - 3090 refills take about 5-9 GB/s from concurrent 5090 uploads.
- **New lever: refill timing.** Keeping 3090 refills off the 5090's demand copies would avoid the -23%. Helper refill timing is
  already on the work list (step 5).
- **Measured before building: the refill lever is about 0.2%**
  ([overlap](/mnt/wsl/transfer/minimax-wsl/runs/contention-bound-20261108/refill-overlap.txt), HAGS-on traces).
  - Refills run only 2.0-4.1% of generation: 26-112 copies a trace, about 5 ms each over the ×4 link.
  - Only 1.5-2.3% of demand-copy time overlaps one. At -23% that costs 0.13-0.21% of generation.
  - It is under 2% and would add a coordination point rather than simplify anything, so it is not built unless the user wants
    it. Demand copies themselves fill 37-47% of generation, so the uploads' bandwidth matters far more.

**The L3 can hold about one expert for the CPU lane, and while it does, the CPU no longer slows the upload.**
[Check](/mnt/wsl/transfer/minimax-wsl/runs/contention-bound-20261108/l3-check.txt) (`l3_check.cu`; the 9950X3D has 96 MB of L3
on its V-Cache CCD and 32 MB on the other, 16 MB of L2).

| CPU working set, 15 threads | CPU alone | CPU beside an upload | 5090 upload |
|---|---:|---:|---:|
| 1 expert (31.9 MB), read again and again | 1,518 GB/s | 1,510 GB/s | 57.5 GB/s |
| 2 experts (63.7 MB) | 1,145 | 950 | 41.8 |
| 3 experts (95.5 MB) | 916 | 920 | 40.7 |
| 6 experts (191 MB) | 563 | 533 | 41.3 |

- **A cached expert is read at about 25 times DRAM speed, and the upload keeps its full 57.5 GB/s.** Once the working set
  spills, the reads reach DRAM and the upload falls to about 41 GB/s again.
- **WSL2 places unpinned threads differently from run to run.** At 8 threads some cases read faster beside the upload than
  alone. This test's readers reread continuously, a reuse gap of only about 21 us (one expert) to 340 us (six), so it says
  nothing about threads moving. A first reading that "vCPUs float between CCDs" was an inference, corrected by the gap test
  below.
- **A prefetched expert survives a realistic gap while its threads stay busy**
  ([gap test](/mnt/wsl/transfer/minimax-wsl/runs/contention-bound-20261108/gap-check.txt), `gap_check.cpp`, at the user's
  question). 15 workers with a fixed 2.12 MB slice each read an expert from DRAM, wait a gap, then reread their slices. A slice
  reread in under 0.1 ms came from cache; from DRAM it takes about 0.55 ms.

  | Gap | Pinned, spinning | Free, spinning | Pinned, sleeping | Free, sleeping |
  |---|---:|---:|---:|---:|
  | 0 ms | 100% of slices hit | 99.8% | 67% | 68% |
  | 0.5 ms | 99.8% | 100% | 57% | 55% |
  | 1 ms | 100% | 100% | 56% | 59% |
  | 5 ms | 99.7% | 95.2% | 59% | 54% |
  | 20 ms | 71% | 55% | 20% | 35% |

  - A busy thread effectively does not move over the 0.3-1 ms an early copy would wait. A sleeping one loses about a third to
    45% of its slices even with no gap: a halted vCPU tends to resume on another core.
  - A cached reread of the whole 31.85 MB took 53-78 us, against about 550 us from DRAM, 7-10 times faster.
  - The lane already fits: its workers are pinned one a physical core (`os::pin_self`) and spin 200,000 pauses, an estimated
    1-3 ms, before sleeping (`mm3_cpu.cpp`, `worker`). A prefetch issued within that window keeps its threads warm. The lane
    hands out work from a shared ticket (`work`), though, so the prefetch and the job need a fixed item-to-worker mapping. One
    expert split over both CCDs puts about 16 MB in each L3, which fits even the 32 MB one.
- **How many experts fit: about one, not four** ([multi-expert gap test](/mnt/wsl/transfer/minimax-wsl/runs/contention-bound-20261108/multi-gap-check.txt),
  at the user's question). The same 15 busy, pinned workers prefetch k experts, wait 0.5-1 ms and reread.

  | Experts prefetched | Slices still cached | Reread time per worker (DRAM-like) |
  |---:|---:|---:|
  | 1 | 100% | 43 us (550) |
  | 2 | 72-73% | 240-246 us (1,100) |
  | 3 | 37-38% | 544-560 us (1,650) |
  | 4 | 23-24% | 934-937 us (2,200) |
  | 5-6 | 4-17% | |

  - The workers fall into two groups, likely the two CCDs. The 7 workers on vCPUs 16-28 keep about 34 MB: all of two experts'
    slices, about 78% of three. The 8 on vCPUs 0-14 keep about 17-20 MB: one expert fully, half of two.
  - So about 52 MB is usable, about 1.6 experts, well below the nominal 96 + 32 MB. Steering more rows to the larger group
    would only rebalance it.
  - Which group is the V-Cache CCD is not known: a pointer-chase map from inside WSL found every vCPU local, likely because
    Hyper-V ran the single reader on the writer's core.
  - For an L3 early copy, one expert a pass can be counted on. That suits `T=1`, where a CPU pass nearly always has one CPU
    job. With drafting, a pass needs 2-5, and only one could come from cache. *(Revised by the capacity test below: with the
    right order and fewer workers, about 2.5.)*
- **What limits that capacity, and how to raise it** ([capacity test](/mnt/wsl/transfer/minimax-wsl/runs/contention-bound-20261108/capacity-check.txt),
  `capacity_check.cpp`, at the user's push). All 30 vCPUs carry spinning workers; the active ones prefetch, wait 0.5 ms and
  reread in 1 MB chunks, either oldest first or newest first.

  | Setup | MB cached, oldest first | MB cached, newest first |
  |---|---:|---:|
  | All 15 workers, 96 MB prefetched | 25 | 56 |
  | 8 workers on vCPUs 0-14 alone, 96-128 MB | 43-49 | 73-78 |
  | 7 workers on vCPUs 16-28 alone, 96-128 MB | 50-76 | 78-85 |
  | 30 workers on both hyperthreads, 96 MB | 17 | 38 |
  | 15 workers, split 3:1 toward one group, 96 MB | 42-47 | 55-65 |

  - **Order matters.** Rereading the newest data first stops each miss evicting still-cached data, so the CPU job should
    compute the most recently prefetched rows first.
  - **Fewer workers hold more.** One group of 7-8 keeps about 73-85 MB, 2.3-2.7 experts. All 15 keep about 50-56, and 30
    keep less: SMT adds no cache. *(Revised by the layout test below: measured without rereading, every layout keeps
    about 86 MB. The rereads themselves evicted the difference.)*
  - **The CCD map taken while all workers spin** finds nearly every worker sharing the writer's L3 (12-17 ns a hop), and only
    one or two at a time remote (67-80 ns). So Windows appears to run most of the VM's busy vCPUs on one CCD, apparently the 96
    MB V-Cache one, and more workers spill some onto the other.
  - That would also explain the CPU's read ceiling. CPU reads cap at about 60 GB/s from two threads up, roughly one CCD's
    fabric link, while CPU and DMA together reach about 79 GB/s.
  - **Placement could be made deterministic from Windows.** The VM's vCPUs are threads of the `vmmem` process, which
    Windows schedules and whose affinity a host task can set. Pinning the lane's vCPUs to the V-Cache CCD would give it the
    96 MB L3 (the cache lever). Spreading them over both CCDs might lift CPU reads from about 60 toward 79 GB/s (the
    bandwidth lever). That is a host change for the user to approve. *(The user approved it; built, see below.)*
- **The capacity measured without disturbing it: about 86 MB, whatever the layout**
  ([layout test](/mnt/wsl/transfer/minimax-wsl/runs/contention-bound-20261108/layout-check.txt), `layout_check.cpp`, at the
  user's request to try layouts and SMT first). 30 workers spin on all vCPUs. The layout's workers prefetch their share of
  each 32 MB expert, all wait 0.5 ms, then each probes its 1 MB chunks newest first with a 256-hop random pointer chase,
  which touches 16 KB a chunk. The mean hop time against a hot (10.4 ns) and a cold (148.9 ns) reference gives the share
  still in the local L3.

  | Layout | Prefetched | Still in the local L3 after 0.5 ms |
  |---|---:|---:|
  | 15 workers, even split, 1 / 2 / 3 / 4 experts | 32 / 64 / 96 / 128 MB | 27.8 / 46.0 / 67.8 / 84.5 MB |
  | 8 workers on vCPUs 0-14, 2 / 3 / 4 experts | 64 / 96 / 128 | 49.1 / 76.5 / 86.4 |
  | 7 workers on vCPUs 16-28, 2 / 3 / 4 experts | 64 / 96 / 128 | 55.5 / 81.1 / 86.0 |
  | Each expert on its own 4 workers, 2 / 3 experts | 64 / 96 | 42.7 / 66.8 |
  | 4 + 4 + 4 + 3 dedicated workers, 4 experts | 128 | 79.6 |
  | 2 experts on the 8 workers + 2 on the other 7 | 128 | 82.5 |
  | 16 threads on 8 guest cores' pairs, 2 / 3 / 4 experts | 64 / 96 / 128 | 46.4 / 70.6 / 86.0 |
  | 2 experts on 16 threads + 2 on the other 14 | 128 | 87.1 |

  - **Every layout tops out at 80-87 MB, about 2.7 experts**, as one 96 MB L3 holding most of the data would. Layouts only
    change how close 2-3 experts come: 7 workers keep 87% and 84% of them, an even split over 15 keeps 72% and 71%.
  - **The earlier 50-56 MB for 15 workers, and the 52 MB of the multi-expert test, came from the rereads:** each miss
    evicted data still cached. The cache holds more than a full reread can show, so the CPU job should read newest first
    and touch only what it uses.
  - **Dedicated workers per expert and the guest's hyperthreads add nothing.** The guest's hyperthread pairs are not the
    hardware's (next item), so these were tests of more busy virtual processors, not of SMT.
- **What the combined GB/s depends on** (`bandwidth_groups.cu`, `bandwidth_repeat.cu`, `bandwidth_sweep.cu`,
  `bandwidth_peak.cu`, `bandwidth_nine.cu` and their `.txt` outputs in the same folder; at the user's question whether any
  of this raises the combined GB/s). CPU readers pinned to guest vCPUs stream 2 GB of pinned records; one upload streams
  other records to the 5090. Windows of 1 s, 5-8 rounds in rotating order.

  | Readers | CPU alone | CPU beside the upload | Upload | Total |
  |---|---:|---:|---:|---:|
  | 2 | 54-58 GB/s | 36-39 | 41 | 78-80 |
  | 8, on any of several vCPU sets | 57-60 | 41-45 | 40 | 81-84 |
  | 9, on any vCPU set | 74-79 (78-79 in most rounds) | 40-44 | 40 | 81-83 |
  | 10 | 64-67 | 38 | 41 | 79 |
  | 15, the lane's count | 61-62 | 36-37 | 41 | 77-79 |
  | 29 | 59 | 41 | 32 | 73 |
  | 15 rereading one cached expert | | 1,460 | 57.3 | 1,517 |

  - **The ~60 GB/s CPU ceiling is placement, not the memory.** Nine readers read about 79 GB/s alone, about the DRAM's
    practical limit, on whichever guest vCPUs they run: spread out, on nine guest cores in a row, even on five guest cores'
    hyperthread pairs. Eight readers covering 1/9 slices stay at 60, so the slice size is not the cause. The number of busy
    virtual processors decides how Windows places them, and some placements use both CCDs' fabric links. *(Revised by the
    affinity tests below: the VM spread over both CCDs still reads 58 GB/s, so the 9-reader case is not a both-CCD
    effect; its cause is unknown.)*
  - **The guest cannot steer it.** Under the root scheduler Windows runs each busy virtual processor on whatever logical
    processor it picks, so the guest's vCPU numbers and hyperthread pairs say nothing about the hardware. The engine's
    other threads also count, so no lane thread count reliably lands in the good case.
  - **Beside an upload the total stays near 80 GB/s for every set: the DRAM's own cap.** Placement only moves the CPU's
    share, 36 to 45 GB/s, while the upload keeps 39-42.
  - **The power plan does not change it.** Under High performance, which stops core parking, the probe repeated the same
    pattern: 9 readers 79.0 ± 0.5 GB/s alone in 8 of 8 rounds, 15 readers 61.1. Balanced was restored right after
    (receipts `power-plan-high.json`, `power-plan-restore.json`).
  - **Only bytes served from L3 beat the cap.** 15 threads rereading a cached expert read 1,460 GB/s while the upload keeps
    57.3.
  - **For the lane:** a CPU job with no upload beside it could read up to about 79 GB/s instead of 60 if the lane's threads
    span both CCDs, which would make it about a quarter shorter. Beside an upload, a one-job pass's 63.7 MB need at least
    0.80 ms at 80 GB/s, and the pass takes about 0.82 ms now, so placement gains little there. The per-pass refit (task 40)
    takes both CPU rates and shows which passes gain.
  - **Priced per pass (task 40, [result](/mnt/wsl/transfer/minimax-wsl/runs/refit-caps-20261108/repredict.txt)): -1.0% to
    -1.6% at `T=1`.** Every `T=1` CPU pass also uploads, so only the CPU's solo tail after the copy speeds up: -1.02% with
    the overlapped rates unchanged, -1.59% if the CPU's overlapped share also rises 10% (the copy's falls 2.5%), as with
    spread readers. With drafting the lane carries 2-5 jobs and is usually last, so its share should be larger. The
    affinity tests decide whether the lane reaches 79 GB/s at all.
- **The affinity action is built and waits for the user to register it** (`tools/mm3/windows/maya-host-control.ps1`,
  staged in `C:\Users\mdwbr\maya-host-control\`). `vm-affinity-query` reports the `vmmemWSL` affinity mask and the
  hypervisor's scheduler type: affinity steers virtual processors only under the root scheduler. `vm-affinity <set>`
  takes one of six fixed masks over the 32 logical processors, each of at least 16: `all` (the default and the restore
  value), `lp0-15`, `lp16-31`, `even` (one hyperthread of each core, both CCDs), `lp0-15-even16-31` and
  `even0-15-lp16-31`. A new VM process starts at `all` again. First tests: `lp0-15` against `lp16-31` (which CCD holds the
  96 MB L3), then `even` (15 readers on 15 cores over both CCDs).
- **Placement from the host: the affinity action works, but the default stays right** (task 45, 2026-11-09;
  [driver](/mnt/wsl/transfer/minimax-wsl/runs/contention-bound-20261108/run_affinity.py), outputs `affinity-*.txt` and
  receipts `affinity-receipts.jsonl` in the same folder). The user registered the action. The hypervisor runs the root
  scheduler (type 0x4), so the VM's virtual processors are Windows threads of `vmmemWSL`, and every mask applied and was
  confirmed by query; the VM was restored to `all` afterwards. Under each mask: the residency probe with 15 workers
  (`affinity_probe.cpp`), CPU and combined bandwidth at 8, 9 and 15 readers (`bandwidth_affinity.cu`), and the engine's
  CPU kernel (`l3_kernel_bench`, 15 threads, one column, medians):

  | Mask | MB still in L3, 4 experts prefetched | CPU alone, 8 / 9 / 15 readers | Kernel from DRAM, alone / beside an upload | From L3, alone / beside | Prefetched, beside an upload: fixed / ticket |
  |---|---:|---:|---:|---:|---:|
  | `all`, the default | 92-101 | 61 / 77 / 60 GB/s | 515 / 862 us | 139 / 149 | 252 / 433 |
  | `lp0-15` | 94 | 56 / 57 / 57 | 554 / 847 | 214 / 225 | 284 / 235 |
  | `lp16-31` | 64-70 | 59 / 69 / 61 | 558 / 857 | 201 / 227 | 555 / 434 |
  | `even`, one hyperthread a core | 90-104 | 58 / 58 / 58 | 516 / 867 | 137 / 154 | 386 / 442 |

  - **Logical processors 0-15, cores 0-7, are the V-Cache CCD:** confined there the cache holds about 94 MB; confined to
    16-31 it tops out lower, at a figure that likely overstates a 32 MB L3 (recently read chunks keep warm TLB entries
    that the cold reference lacks).
  - **No mask lifts CPU reads.** Confined to one CCD they stay at 56-61 GB/s, and spread over both (`even`, both L3s in
    use) still at 58. The 9-reader jump to about 77 GB/s appears only under `all`, and in 2 of 4 rounds under
    `lp16-31`: it is not a both-CCD effect, and its cause is still unknown. Beside an upload the total stays at 75-82 GB/s
    under every mask.
  - **The kernel from DRAM does not change;** confined to one CCD, 15 threads share 8 cores and the L3-hot kernel slows
    from about 139 to 201-214 us.
  - On the V-Cache CCD a prefetched expert survives even the lane's dynamic ticket (235 us beside an upload), where `all`
    needs the fixed mapping (252 us). A wash, bought by confining the whole VM to 8 cores. **Decision: keep `all`.**
- **The real CPU kernel from L3 against DRAM** (task 43;
  [benchmark](/mnt/wsl/transfer/minimax-wsl/runs/l3-kernel-20261108/l3-kernel-bench.txt), `l3_kernel_bench.cpp`). It
  times the engine's own `fast_ffn_cpu` (Infernix-minimax, built with the engine's flags) one expert a call at MiniMax's
  geometry: 6144 × 3072, 31.85 MB, 384 gate/up row groups of 54 KB, then 384 down row groups of 27 KB. The pool copies the
  lane's (`CpuLane::Impl`: pinned workers, 200,000 spins, one compare-and-swap ticket). A fixed mapping instead gives
  thread t the jobs t, t + 15, ..., so a prefetch and the kernel hand each row group to the same thread. Medians of 300
  calls, 15 threads, one column (`T=1`):

  | Expert's weights | Alone | Beside a 5090 upload |
  |---|---:|---:|
  | In DRAM (64 experts in turn) | 525 us | 878 us (upload 38.9 GB/s) |
  | In L3 (the same expert again) | 159 | 195 (upload 51.2) |
  | Prefetched by the pool 0.3 ms before, fixed mapping | 251 | 455 (upload 45.8) |
  | Prefetched, the lane's dynamic ticket | 298 | 503 (upload 43.0) |
  | The prefetch itself | 554 | 955 |

  - **A prefetched expert's job runs about twice as fast,** and the upload beside it gains 7 GB/s. Interpolating between
    DRAM and L3, 62-75% of the prefetched bytes are served from cache. The fixed mapping is worth about 50 us over the
    ticket at 15 threads.
  - **The prefetch costs more DRAM time than it saves** (554 us against 274 saved alone), so it pays only in windows where
    DRAM is otherwise idle. The preparation's ~0.2 ms covers about 36% of an expert at the prefetch's 57.5 GB/s.
  - **From L3 the kernel is compute-bound:** 159 us at 15 threads, 225 at 8 (from DRAM 525 and 539). Four columns (4
    tokens on one expert) take 371 us from L3, 573 from DRAM.
  - **Fewer threads lose share beside an upload:** the kernel at 8 threads takes 986 us beside an upload against 878 at 15.
    Unlike the streaming readers above, it computes between loads, so 15 threads keep more requests in flight. The lane's
    15 threads stay right.
  - Rough bound, before the per-layer idle-window analysis: at `T=1` about 10 CPU jobs a token, each about 100 us shorter
    on average if the 0.2 ms window is idle and the predictor's 53% hits, so at most about 1 ms of a 54 ms token (about
    2%), and only where the CPU path is the last.
- **The work around the dot products is small; no change pays** (task 46, the Astra review, 2026-11-09;
  [serial work](/mnt/wsl/transfer/minimax-wsl/runs/l3-kernel-20261108/serial-bench.txt), `serial_bench.cpp`;
  [chunked tickets](/mnt/wsl/transfer/minimax-wsl/runs/l3-kernel-20261108/chunk-bench.txt), `chunk_bench.cpp`). The engine's
  own code, copied, at MiniMax's geometry:

  | Step of one CPU expert job, `T=1` | Time |
  |---|---:|
  | Caller, before the workers: input F32 to BF16 (already vectorized by the compiler) | 0.32 us |
  | Caller, before the workers: input quantization, scalar division | 3.96 us |
  | Caller, after the workers: weighted accumulation | 0.12 us |
  | Workers, single-thread total: gate/up activation with scalar `expf` | 6.79 us |
  | Workers, single-thread total: hidden quantization | 2.03 us |
  | Pool: the kernel's three phases with empty jobs (384, 12, 384), 15 threads | 39.87 us |

  - The caller's steps take about 4.4 us of a 525 us job, under 1%, and the workers' share of the rest is under 1 us
    each. Vectorizing them, quantizing a token's input once across its experts, or fusing the 0.43 us hidden-quantization
    phase cannot pay.
  - The empty phases cost about 40 us because every job passes one compare-and-swap ticket. Handing out 2-8 row groups a
    ticket does not help the real kernel: one-column jobs from DRAM slow from 523 to 563-587 us and from L3 from 166 to
    171-188 us, since larger chunks leave threads idle at a phase's end. Only four-column jobs beside an upload gain (1,045
    to 952 us at chunk 4). In real jobs the ticket traffic overlaps the memory work.
  - Fewer workers lose beside an upload (986 against 878 us at 8 threads, above). A one-column kernel specialization would
    matter only from L3, compute-bound at 139-166 us, so only with an L3 prefetch.
  - With `early_copy` the CPU computes 2.5 experts a token instead of 9.5, so at `T=1` even a full CPU speedup is small;
    with drafting the CPU carries more, and wide jobs are where kernel work could still pay.
- **The idle DRAM window before each CPU pass, and what an L3 early copy is worth per pass** (task 43;
  [script](/mnt/wsl/transfer/minimax-wsl/runs/l3-kernel-20261108/idle_windows.py),
  [result](/mnt/wsl/transfer/minimax-wsl/runs/l3-kernel-20261108/idle-windows.txt)). The four untouched HAGS-on `T=1`
  workloads, 21,831 generation passes each, 23-42% of them with CPU work (1.02-1.06 jobs). For each CPU pass, the time
  from the previous pass's last host-memory read (its CPU end, or its copy end from the refitted model) to this pass's
  first one (its copy, 0.141 ms before the lane). A prefetch moves 57.5 GB/s in that time. Each pass is then priced with
  the refitted HAGS model, its CPU work scaled for the bytes served from L3 at 30% of their DRAM time; the expected cycle
  averages the predictor's hits and misses pass by pass.

  | Prediction available | Idle ms p10 / median / p90 | Share of the CPU bytes covered | Cycles, hit 0.53, retention 0.75 / 1.0 | Hit 0.80, retention 0.75 / 1.0 |
  |---|---|---:|---:|---:|
  | Only once this layer's preparation is submitted, +0.03 ms | 0.04 / 0.16 / 0.42 | 35-37% | -1.40% / -1.79% | -2.11% / -2.71% |
  | By the previous pass's last read (early_copy's prediction arrives at its join) | 0.32 / 0.42 / 0.75 | 74-80% | -2.91% / -3.59% | -4.39% / -5.42% |

  - A prediction made only from this layer's own input would exist once the host has submitted this layer's preparation,
    after merging the previous layer. One available at the previous pass's last read more than doubles the window and
    the bytes covered.
  - Counting the previous pass's idle time before its own copy adds almost nothing (-2.92% against -2.91%), so the
    previous CPU job streaming through the cache is no concern.
- **The engine already has the predictor for the second row.** *(Corrected the same night: this entry first called a
  router-based one-layer-ahead predictor unmeasured and likely well under 53%.)* `early_copy`'s prediction applies layer
  *l+1*'s router to layer *l*'s input during layer *l*'s preparation, and its ids come back with layer *l*'s routing, at
  layer *l*'s join ([accuracy](#routing_prefetch-review-work-list-item-4)). Its first predicted HOST expert is used on 53%
  of passes, and its top-4 holds 90.5% of the CPU-served misses. So the one-layer-ahead bound applies: **about -2.9% to
  -3.6% on today's base.**
- **But `early_copy` spends the same prediction better.** It uploads that expert to the 5090 a layer early, which in its
  bound cut CPU jobs from 0.165 to 0.107 a pass under the engine's rule, and to 0.038 behind one demand copy. It measured
  -5.03% against off. With it on, an L3 prefetch could serve only the CPU jobs that remain, for example the second
  predicted HOST expert, likely well under 1%. **So: integrate `early_copy` first (task 41), then bound the L3 prefetch on
  the remaining CPU jobs.**
- **Cheap predictors that need no GPU work are far weaker**
  ([script](/mnt/wsl/transfer/minimax-wsl/runs/l3-kernel-20261108/predictors.py),
  [result](/mnt/wsl/transfer/minimax-wsl/runs/l3-kernel-20261108/predictors.txt)). Over 28,351 CPU passes of the same four
  workloads, a hit means the named expert is served on the CPU by today's planner, or, second figure, is any host miss of
  the pass, which a planner aware of the prefetch could then serve on the CPU:

  | Predictor, known at the previous pass's last read | Names one on | Hit: CPU-served / any host miss |
  |---|---:|---:|
  | The previous token's CPU expert at this layer | 48% | 9.2% / 12.3% |
  | Co-activation with the previous layer's picks (six training captures) | 99.6% | 10.7% / 21.9% |
  | The same, also learning online within the request | 99.6% | 13.0% / 27.0% |
  | Previous token, else co-activation online | 99.6% | 16.2% / 26.6% |

  - At these rates they would give only about -0.9% to -1.5%; the router's prediction is the one to use.
  - **Decision: not built now.** On today's base an L3 prefetch of `early_copy`'s predicted expert is worth about -2.9% to
    -3.6% (75% to 100% retention, a planner that serves the prefetched expert on the CPU), but `early_copy` itself uses
    that prediction for more. After `early_copy`'s integration (task 41), the L3 prefetch is bounded again on the CPU jobs
    that remain; the host affinity's full retention would raise it.
- **Natural caching cannot help.** The CPU computes the cold tail, since the hottest experts live on the GPUs. It streams
  about 10 distinct experts a token at `T=1` (about 320 MB), 22 with drafting, so none recurs while still cached.
- **A way to use it: an early copy into L3 for the CPU.** Predict the next layer's CPU expert with the router-on-input
  predictor (its first predicted HOST expert was used on 53% of passes). Read it into L3 while DRAM is idle, during the next
  layer's preparation or on a pass without a CPU job. The CPU job then reads from cache, and the upload beside it runs at
  57.5 instead of 41 GB/s.
  - A one-job pass could fall from about 0.84 ms toward about 0.6 ms.
  - Limits: the prefetch costs the same DRAM bytes, so it only helps in idle windows. The preparation leaves about 0.25-0.3
    ms, about half an expert. Work must map to threads in a fixed way, and ideally the lane stays on the V-Cache CCD, which in
    WSL2 needs the host's affinity for the VM.
  - Rough bound: 1-2% at `T=1`, more with drafting where the CPU lane is usually last, though the cache holds only about one
    expert. To bound properly: the real NVFP4 kernel from L3 against DRAM, the idle DRAM window per layer, then the refitted
    per-pass model.
- **What it means for the central objective.** The records-level gap of 0.78 ms compares only the CPU and the helper, since
  light records hold no copy times. With the copy counted, the slowest and second-slowest paths of a one-job pass are about
  0.03-0.05 ms apart.
  - The 63.7 MB such a pass moves needs at least about 0.8 ms at 80 GB/s.
  - Uploading both misses instead serializes on the copy engine, about 1.1 ms. So the planner's choice holds under contention,
    and re-pricing for it can gain little.
  - The remaining levers are fewer bytes from host memory a pass and more work on the 3090. Placement already gives the 3090
    the hottest experts by real demand (next section), and a larger routing bonus failed usability.

### 3090 placement, work-list item 4

**2026-11-08: keep helper-first. The 3090 holds the hottest experts by real demand at load, and that is the better
placement.** It answers the user's question. Scripts and output are in
[rehoming-bound-20261108](/mnt/wsl/transfer/minimax-wsl/runs/rehoming-bound-20261108/). Demand comes from the four
light-capture workloads, at the new `T=1` base, priced by the HAGS-on model.
- **The unconstrained bound is not reachable.** For each pass, the GPU-resident experts were divided between the GPUs in the
  split the model finishes soonest.
  - That saved 8.3%, always by moving work to the 5090, on 76-92% of passes.
  - The reason is real: a pass that touches the 3090 pays its startup and collection. In the model that is 0.034 ms more
    preparation, 0.054 ms more finish and a 0.149 ms lane for one job, about 0.19 ms on a pass with no CPU job.
  - The 5090 runs the same five grouped kernels per window whatever its expert count: a median of 6.6 us each, 0.073 ms
    in all, in the HAGS-on trace. So its marginal expert is nearly free.
  - But the split ignores capacity, and the 5090 cannot hold what the 3090 holds.
- **The capacity-respecting test: primary-first is 3.5% slower.**
  - In this test, the 5090 starts with the hottest 438 experts by the setup's profile and the 3090 with the next 741. The
    union, every frame and the host sets are unchanged.
  - The fitted model gives +3.46%. With a 0.007 ms marginal cost per 5090 expert it gives +3.59%.
  - Passes touching the 3090 fall from 85.2% to 73.2%. But CPU jobs rise from 0.353 to 0.424 per pass, and uploads from
    0.822 to 0.906.
  - The 5090's slots are the adaptive cache. Filled with stable, globally hot experts, admission keeps them, and each
    request's own hot experts spill to the CPU and to uploads. The static 3090 is the right home for the stable ones.
- **Not built.** A rule that moves hot experts from the 5090 to the 3090 during generation only reshuffles locations
  within the same resident union.
- **The 3090's startup is the lever.** That supports item 5's work on the 3090's startup as well as its kernel.

### CPU batch before the GPU submissions, work-list item 3

**2026-11-08: bounded at 2.67%, so it is built as the per-request option `cpu_first`. The hardware test is pending.**
- **The host's order today.** After a layer's join, the service thread:
  1. plans;
  2. submits the 3090's work;
  3. submits the 5090's grouped kernels and the query;
  4. only then runs the CPU batch itself.
- **Uncaptured host timeline**, from the light captures of the HAGS-on validation load, CPU-job generation passes:

  | Step after the join | Median (p10-p90) |
  |---|---|
  | Adaptive publish | 0.002 ms |
  | Planning | 0.037 ms (0.027-0.058) |
  | 3090 submission | 0.025 ms |
  | 5090 kernels and query | 0.080 ms (0.057-0.103) |
  | **CPU batch starts** | **0.14 ms** (0.10-0.18) |
  | CPU batch runs | 0.88 ms |

- **Detailed capture inflates the host side about 3.5 times.** In the Nsight fixtures, recorded in detailed capture, the CPU
  batch starts 0.48-0.49 ms after the join: router scores are written out as JSON on that thread. Host-side conclusions
  from those traces need this correction.
- **Bound.** The CPU is last on 99% of passes running both CPU and helper, with a median lead of 0.37 ms. Starting it
  right after planning saves the measured submission time, less a 5 us dispatch, on each CPU-job pass:
  - per workload: extract 2.70%, reason 3.15%, code 2.65%, prose 2.11%;
  - **2.67% of generation time over all four.**
  - These loads ran CPU price 1.3. At 1.5 the CPU takes 20-26% fewer experts, which leaves about 2.0-2.1%.
  - The bound ignores the extra overlap with the demand copies, which share host bandwidth.
- **Implementation**, off by default:
  - The submit thread issues the 3090's work, the 5090's kernels and the query, while the service thread starts the CPU
    batch at once and joins the submit thread before collecting the helper.
  - Counters and capture fields are written after the join, and a guard joins on any early exit.
  - Passes with no CPU job, traces and fixed replays keep today's path unchanged.
  - The server accepts `cpu_first` per request.

**Hardware test, 2026-11-08: no gain, so `cpu_first` stays off. The frozen prediction of -0.84% failed.**
[Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/cpu-first-hags-20261108/evaluation.txt),
[arm check](/mnt/wsl/transfer/minimax-wsl/runs/cpu-first-hags-20261108/arm_check.txt).
- **Design.** One fresh load and six blocks, each with both arms on python-code and prose at up to 2048 tokens, at the `T=1`
  base. The prediction was frozen at 04:08 with the light-capture tables.
- **Result, on over off.**
  - Pooled: +0.66% [-0.36, +1.69]. The prediction, -0.84%, lies outside the interval.
  - Code: +0.38% [-2.85, +3.62]. Prose: +0.62% [-1.07, +2.31].
- **Did it run?**
  - The test binary was built before the stat line carried `cpu_first` and `cpu_first_passes`, so the evaluator cannot
    confirm the arms. The request did carry `cpu_first=1`, and the engine reported the control as available.
  - The work shows it ran. On the six block-prompt pairs where both arms wrote identical text, so the work was identical,
    the on arm's CPU batch was 4.7-5.3% longer every time: code 12.86 to 13.50 ms per token, prose 5.64 to 5.93. It was
    longer in 11 of the 12 pairs overall.
- **Why the bound did not hold.**
  - The batch does start earlier, but it slows by about 5%, roughly the time it gained.
  - The likely cause is the extra thread. The submit thread runs beside the CPU lane's 15 participants, one per physical
    core.
  - All three requests the stall check flagged were in the on arm, 2.3-3.5 ms per token over their medians.
- **The bound's premise is now known wrong.** It assumed the CPU work keeps its speed when it starts earlier. A form that
  adds no thread could still recover part of the 0.105 ms, for example a lane participant that issues the GPU work before
  joining the batch. That is not built.
- **Decision.** Under the rule (adopt only if the interval lies entirely below zero), `cpu_first` stays off.

**Re-predicted with the host-memory contention, 2026-11-08 (task 40): the model gives -0.89%, which is the frozen
prediction the hardware already refuted. So it is not re-timed.**
[Script](/mnt/wsl/transfer/minimax-wsl/runs/refit-caps-20261108/repredict.py),
[result](/mnt/wsl/transfer/minimax-wsl/runs/refit-caps-20261108/repredict.txt).
- **The refitted model already holds the per-path caps.** Its fitted shares put the CPU at 0.595 and the copy at 0.665 of
  their solo rates while both run: about 35 + 38 = 73 GB/s, against 36 + 39 = 75 for the engine's kernel beside an upload
  in `l3_kernel_bench` and 79 for pure streaming. Every `T=1` CPU pass in the four workloads also has a demand copy, so
  this one overlap is all a cap-based refit would describe. It needs no separate refit at `T=1`.
- **`cpu_first`, per pass:** the CPU batch starts 0.100 ms earlier while the copy keeps its start, so the copy's 0.141 ms
  solo lead shrinks to 0.041 and the two share host memory for longer. The CPU batch then runs about 7% longer; the
  hardware measured 4.7-5.3%. Net: **-0.89%** over the four workloads (-0.74% to -1.00%), against -2.27% if the overlap
  is ignored.
- So the contention was already in the frozen -0.84%, and the hardware's +0.66% [-0.36, +1.69] has another cause: most
  likely the extra submit thread beside the lane's 15 participants, which the model does not hold.
- **Decision.** Re-timing the same build would repeat that test. A form without the extra thread is bounded near -0.9%
  and adds a coordination point, so under the [revised rules](README.md#adoption-rules) it is not built.

### Query and spinner under HAGS, work-list item 3

**2026-11-08: both stay on; turning the spinner off costs 16.4%.**
[Protocol](/mnt/wsl/transfer/minimax-wsl/runs/query-spinner-hags-20261108/make_protocol.py),
[evaluation](/mnt/wsl/transfer/minimax-wsl/runs/query-spinner-hags-20261108/evaluation.txt),
[settings check](/mnt/wsl/transfer/minimax-wsl/runs/query-spinner-hags-20261108/settings_check.txt).
- **Why it was rechecked.** With HAGS off, the `T=1` submission query cut generation 15%, by working around Windows'
  submission batching, which HAGS changes. The spinner holds the 3090 out of its idle clock and shares its queue with the
  helper's work.
- **Design.** A 2×2 of `primary_submit_query` and `helper_occupancy` on one fresh load, at the usability-cleared `T=1` base.
  Six blocks, each with all four arms on python-code and prose at up to 2048 tokens. Arm order follows a repeated Latin square,
  and every request starts from the restored origin.
- **Rule, fixed before the run, for each factor.** Keep the factor on unless off is credibly faster: the 95% interval across the
  six blocks lies entirely below zero and the mean is below -1%.
- **Result, off against on, pooled over both prompts:**

  | Factor | Effect of turning it off | Code | Prose |
  |---|---|---|---|
  | Spinner | **+16.42%** [+15.91, +16.93] | +16.45% | +16.40% |
  | Query | +0.17% [-0.16, +0.49] | +0.41% | -0.04% |
  | Interaction | +1.15% [-0.80, +3.09] | | |

  Both arms with the spinner ran at 59.9-61.3 ms per token, and both without it at 69.5-71.5.
- **The spinner works through the 3090's results.** Without it, the host's lane time per token rises from 19.8 to 33.4 ms. That
  time is the CPU batch plus collecting the 3090's result (`decode_cpu_wait_ms`). The CPU work itself is unchanged, at
  8.7-8.8 ms per token, and so are the expert counts. So the difference is the 3090 answering more slowly at its idle clock.
- **Settings check.** The evaluator flags all 48 requests, because the engine's stat line does not echo `helper_occupancy`. The
  engine log does: exactly the 24 spinner-on requests logged nonzero occupancy launches. So every request ran with its arm's
  settings.
- **Stalls.** Two requests were flagged, both in spinner-off arms: 70.81 against 67.69 and 75.10 against 72.58 ms per token.
  They decide nothing.
- **Decision.** Under the rule, both stay on and the serving defaults are unchanged. The query is now neutral at `T=1` with HAGS,
  not the 15% it was without it. Item 7 rechecks both with speculation.

### The 3090's startup, work-list item 5

**2026-11-08: host overhead is about half the helper path, so `helper_fast` is built and bounded at 1.9-3.0%. The hardware
test is queued.** User direction: target the startup before the 3090's kernel as well as the kernel.
- **Uncaptured helper path**, from the light captures of the HAGS-on validation load. It covers the 61.9% of `T=1` passes
  with a helper job and no CPU job:

  | Step (median, p10-p90) | ms | Summed, % of generation |
  |---|---|---:|
  | Enqueue after planning | 0.007 | 0.39 |
  | Helper worker thread wakes | 0.033 (0.028-0.040) | 1.75 |
  | Worker's CUDA submission: input copies, grouped launch, output copy | 0.092 (0.069-0.115) | 4.90 |
  | Event record | 0.007 | 0.36 |
  | 3090 transfers and kernels, event record to ready | 0.150 (0.087-0.236) | 8.22 |
  | Collector thread wakes after publication | 0.032 (0.027-0.040) | 1.83 |
  | Merge into the CPU rows | 0.023 | 1.19 |
  | **Planning to collected result** | **0.354** (0.275-0.451) | 18.6 |

- **Why these passes matter.** On a pass with no CPU job and no upload, the helper lane decides the pass, and the 5090's
  grouped kernels take a fixed 0.073 ms. So a shorter helper path shortens the pass directly.
- **`helper_fast`**, a per-request option that is off by default:
  - the worker and the collector spin for up to 2 ms on lock-free mirrors of the queued and finished flags, then fall back
    to the existing condition-variable waits, whose predicates still decide;
  - one-column passes, all of them at `T=1`, submit through a CUDA graph per job count. The graph is captured once on the
    worker's stream and falls back to the separate calls if capture fails;
  - the engine reports `helper_graph_launches` per request.
- **Bound on the per-pass model.** The helper lane is shortened by the measured worker wake and the collector's finish by its
  measured wake. Replayed over the light-capture workloads at the `T=1` base:
  - spin waits alone: 1.87%;
  - with graph submission, assuming it saves 0.050 ms of the 0.092: 2.99%.
- **Frozen prediction: -2.99%** ([prediction](/mnt/wsl/transfer/minimax-wsl/runs/helper-fast-hags-20261108/prediction.json),
  sha256 `6a143483aaa976218014ae70b8484712f343ccbe2f1b7deb0a4e70b2adaa244e`).

**Hardware test of `helper_fast`, 2026-11-08: -5.04% [-5.86, -4.22], adopted for the `T=1` base. The frozen prediction of
-2.99% was too small.**
[Evaluation](/mnt/wsl/transfer/minimax-wsl/runs/helper-fast-hags-20261108/evaluation.txt),
[arm check](/mnt/wsl/transfer/minimax-wsl/runs/helper-fast-hags-20261108/arm_check.txt).
- **Design.** One fresh load and six blocks, each with both arms on python-code and prose at up to 2048 tokens, at the `T=1`
  base.
- **Result, on over off.** Every block lies between -4.26% and -6.35%, and none is flagged by the stall check.
  - Code: -4.08% [-4.53, -3.62], against a predicted -2.79%.
  - Prose: -6.27% [-7.99, -4.56], against a predicted -3.92%.
  - Off ran at 60.4-61.7 ms per token, on at 57.6-58.3.
- **It ran.** The server dropped `helper_fast` and `helper_graph_launches` from each request's stats, so the evaluator flags
  every request (fixed in 9f8c726). The work shows the arms clearly:
  - the host's lane time per token, which is the CPU batch plus collecting the 3090's result, is shorter in all 12
    block-prompt pairs, by 3.3-4.2 ms (code 23.4 to 20.1, prose 19.5 to 15.3);
  - that includes every pair with identical text;
  - the CPU batch itself is 0.5-3% longer in all 12. The spinning threads take cycles from the lane's cores, but that cost
    is far smaller than the saving.
- **Why the prediction fell short.** It used the two measured thread wakes and an assumed 0.050 ms from graph submission.
  The lane falls about 0.06 ms per pass averaged over all passes, more than the model's shift on helper passes alone
  implies. So the per-pass model under-reads how much of the helper path's host time decides a pass. That is worth knowing
  for the 3090's remaining startup and kernel work.
- **Decision.** Under the rule (adopt if the interval lies entirely below zero), `helper_fast` joins the `T=1` base for later
  tests. It does not change output, so no usability check is needed. The serving defaults, which draft, follow its recheck
  with speculation (item 7). There most passes are wide and only the spin waits apply.

**Recheck of `helper_fast` with drafting, 2026-11-08: -0.58% [-1.19, +0.03], so it stays out of the serving defaults.**
[Protocol](/mnt/wsl/transfer/minimax-wsl/runs/helper-fast-spec-20261108/make_protocol.py),
[evaluation](/mnt/wsl/transfer/minimax-wsl/runs/helper-fast-spec-20261108/evaluation.txt),
[run notes](/mnt/wsl/transfer/minimax-wsl/runs/helper-fast-spec-20261108/notes.txt).
- **Design.** Work-list item 7, at the serving configuration: drafting 5 with lenience 0.4 and draft confidence 0.3, bonus 0.20,
  CPU price 1.3, horizon 20, spinner on, submission query off, HAGS on. One fresh load of eb3620b and six blocks, each with both
  arms on python-code and prose at up to 2048 tokens, the order alternating. No model prediction: the per-pass model is fitted
  at `T=1` only.
- **Result, on over off.** Five blocks are faster and one slower (+0.49%), and none is flagged by the stall check.
  - Code: -0.12% [-0.95, +0.72]. Prose: -0.82% [-1.98, +0.34].
  - Off ran at 43.5-44.0 ms per token on code and 49.8-50.0 on prose; on at 43.4-43.9 and 49.1-50.5.
  - Both arms did the same work: 19.77 CPU experts and 25.68 staged experts a token, and the same text in all 12 block-prompt
    pairs. With drafting most passes are wide, so the on arm submitted only 3.4 graphs a token, on its one-column passes.
- **Decision.** Under the rule fixed before the run (add it if the interval lies entirely below zero), `helper_fast` stays a
  `T=1` setting. The interval misses by 0.03 points: a small gain is likely, but this test cannot show it.
- **Two false starts, set aside** (`aborted-0920/`, `aborted-0930/`): the copied binary first lacked its executable bit, then
  the protocol inherited the `T=1` check's requirement that speculation be off. Neither attempt's data is used.
- **Deterministic output.** Every request from the restored origin gave the same text for its prompt, in both arms and all six
  blocks (440 tokens of code, 723 of prose, with the same drafting rounds). Output identity across arms is therefore a usable
  check that a change leaves output alone.

**The 3090's kernel, 2026-11-08: it already streams at 79% of the card's bandwidth, so a faster kernel buys little at `T=1`.**
[Bound](/mnt/wsl/transfer/minimax-wsl/runs/rehoming-bound-20261108/helper-kernel-bound.txt),
[by kernel](/mnt/wsl/transfer/minimax-wsl/runs/rehoming-bound-20261108/helper-kernel-names.txt).
- **Measured.** Source: the HAGS-on Nsight traces (`t1-code`, `t1-prose`), CUPTI kernel times on device 1, per one-column
  window, by the window's helper job count from the trace's own capture. The spinner's `occupancy_kernel` is in no window.

  | Helper jobs | Windows | All four kernels | Gate/up projection | Down projection | Two quantizations |
  |---:|---:|---:|---:|---:|---:|
  | 1 | 2,167 | 0.072 ms | 0.044 | 0.022 | 0.007 |
  | 2 | 2,285 | 0.115 ms | 0.072 | 0.036 | 0.007 |
  | 3 | 1,427 | 0.174 ms | 0.111 | 0.055 | 0.007 |
  | 4 | 404 | 0.209 ms | 0.135 | 0.067 | 0.007 |

- **Each added job costs about 0.044 ms**: 28.7 us for gate/up and 14.8 us for down. That streams a 31.85 MB NVFP4 record
  (3 x 6144 x 3072 at 4.5 bits) at about 740 GB/s, 79% of 936 GB/s.
- **About 0.028 ms per window is fixed.** That is four kernel launches with their tails, plus the quantizations.
- **Bound on the per-pass model**, helper draws shortened by the per-job saving, light-capture workloads, `T=1` base:
  - a kernel at 85% of peak saves 6.8 us per job: -0.37%;
  - 1.5 times faster per job: -0.85%;
  - 2 times faster per job: -1.27%. That is not reachable, since it would exceed the card's bandwidth.
- **So Marlin W4A16 or another per-job kernel buys at most about 0.4% at `T=1`.** The fixed part is the larger target. A
  single fused kernel for quantization, gate/up and down would remove three launches and their tails, perhaps 0.015-0.020
  ms per window, worth roughly 0.5-1% by the spin-wait bound's scale.
- **User direction:** build and measure the 3090's kernel regardless of the 2% bar. These numbers set what to build: the
  fusion first, and a per-job kernel only for wide windows, where jobs carry more columns.

### Device-side upload decision, work-list item 4

**2026-11-08: bounded at about 0.7% of `T=1` generation, so it is not built.**
- **Idea.** Start a pass's demand copies on the GPU right after routing, instead of after the host's readback, join and
  planning.
- **Measured, uncaptured.**
  - From the light captures of the HAGS-on validation load, the host issues the first demand copy a median 0.008 ms after
    the join (p90 0.013).
  - From the HAGS-on Nsight trace, which detailed capture does not inflate on the GPU side:
    - the routing readback copy takes 1.4 us;
    - the host's synchronize returns a median 1.8 us after it ends (p90 12 us);
    - the host waits in that synchronize a median 0.079 ms, for the GPU's own attention and router work.
  - So routing on the GPU leads to a copy issued about 12 us later.
- **Bound.** A device-side decision saves only that, plus perhaps the copy's submission latency, about 20 us per pass. It
  matters only where the copy path finishes last: upload passes without a CPU job, 35-41% of `T=1` passes. That comes to
  about 0.7%. Summed directly over those passes, the host's 8 us alone is 0.29% of generation.
- The plan's earlier figure of 0.067-0.104 ms of host preparation was a fitted constant that includes GPU work. It was not
  the host's reaction time.
- This settles the audit's GPU-driven paging for `T=1`. The prediction-driven early copy under `routing_prefetch` is a
  different lever: it would start the copy a whole layer earlier, not 12 us earlier.

### Per-pass stalls, from the user's review of two HAGS-on passes

**Two kinds, both from the HAGS-on trace the user reviewed, and both also present in HAGS-off runs.** An earlier summary to
the user said "neither comes from HAGS". That was wrong as written, and not supported: the counts below are too small to say
whether HAGS changes the CPU-lane stalls, and HAGS improves the helper side.
[Helper collection](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-hags-20261107/helper_collect_delay.txt),
[CPU lane](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-hags-20261107/cpu_lane_stalls.txt).

- **Helper-collection stall: `trace-t1-prose`, forward 358, layer 12.** The pass has no CPU job, so its origin is the helper
  collection's start.
  - The 3090's result copy finished at -0.134 ms, but the host's collection ended at +3.055 ms and the helper worker's CUDA
    phase at +3.004.
  - The 5090, done at +0.292 ms, spun in `wait_publication` for 2.77 ms, and the 3090 sat idle for 3.27 ms.
  - The worker polls its completion event, yielding for the first millisecond and then sleeping 50 µs between queries. So
    either the event was observed late or the thread was not running.
  - Such a stall, the collection ending more than 0.5 ms after both the 3090 and the CPU are done, hits 0.09-0.37% of
    helper passes, with or without HAGS.
- **CPU-lane stall: `trace-t1-prose`, forward 355, layer 23.**
  - One CPU job, predicted at 0.821 ms, ran 5.457 ms. The 5090, done at +0.650 ms, spun in `wait_publication` for 4.80 ms.
  - A CPU batch more than 1 ms over the median of its own (rows, CPU jobs, uploads) group hits 0.02-0.09% of CPU passes in
    untraced records and 0.03-0.25% in traces, with or without HAGS. It costs 0.09-0.38% of CPU-lane time untraced.
  - The excess clusters at 3.5-4.0 ms, which fits a thread losing its processor for a scheduling quantum, plausibly a WSL2
    virtual CPU preempted by Windows, rather than anything in the work.

**Helper completion in untraced records, HAGS off → on**
([detection](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-hags-20261107/helper_detection.txt)). The time
from recording the 3090's completion event to seeing it ready includes the 3090's own work.
- Its median falls from 0.304-0.310 ms to **0.129-0.152 ms**.
- Detections over 1 ms fall from 2.37-3.63% of helper passes to **1.26-1.44%**.
- In 85-92% of those, the worker's last "not ready" query came within 0.2 ms of "ready". So the event became ready late
  while the thread was polling, on the driver or GPU side, not a host scheduling stall.
- The traced pass kept no polling times, so for forward 358, layer 12 the split is unknown: the 3090's copy ended at
  -0.134 ms and the host saw it about 3 ms later.

**For step 5's stall check.** Proposed definition, to be fixed in the protocol before the round: a pass whose CPU batch or
helper collection exceeds the median for its work by more than 2 ms. Small candidates, each under 2%, are spinning instead
of sleeping in the helper's poll and raising the WSL VM's priority through the allowlisted host operation.

### Per-pass model refitted under HAGS, work-list item 2

**At `T=1`, 2026-11-07 late.** [Captures](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-hags-20261107/):
a fresh load with HAGS on, and the six training and four untouched workload requests of the HAGS-off fit, with the same tune.
[Refit](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-hags-20261107/refit_t1.txt),
[shares from the traces](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-hags-20261107/refit_shares_trace.txt),
[model](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-hags-20261107/model-hags-final.json).

**Two model changes,** both reducing to the old rule at a lead of 0, so the HAGS-off fit and its validation are unchanged:
- `copy_lead_ms`, the copy's start against the CPU batch's: the traced median is -0.141 ms over 3,468 passes. The copy runs
  alone until the CPU begins, then shares host bandwidth. Upload passes with no CPU job keep the lead against their own
  origin; with HAGS off none existed, and with HAGS on they are the most common upload pass.
- **The CPU's uncontended per-job cost is taken from the HAGS-off fit.** With HAGS on, every `T=1` miss pass in the captures
  uploads, so no pass shows CPU work alone. That cost is a property of the CPU, which HAGS does not change.

The sharing rates are fitted on one trace and tested on the other (CPU 0.59-0.60, copy 0.66-0.67), and the model uses their
mean. The records-based fit gave 0.59 and 0.62, but records hold no copy times.

| Check | HAGS-off model on HAGS-on work | HAGS-on model |
|---|---|---|
| Records, four untouched workloads, 21,065 passes each: cycle error, mean / abs | +0.110 to +0.116 / 0.141-0.144 ms | **-0.012 to -0.005 / 0.052-0.054** |
| Records: CPU end / helper end, mean abs | 0.047-0.053 / 0.139-0.148 ms | 0.044-0.049 / **0.033-0.038** |
| Records: CPU-helper order; per-layer CPU-helper abs(Δ) error | 99.8-99.9%; 0.114-0.128 ms | 99.8-99.9%; **0.024-0.033** |
| Traces: slowest path right | 98.2%, 97.9% | 98.2%, 97.9% |
| Traces: 5090 upload end, mean / abs | +0.022 / 0.055, +0.031 / 0.068 | **-0.000 / 0.047, +0.011 / 0.057** |
| Traces: helper end, mean / abs | +0.134 / 0.141, +0.122 / 0.140 | +0.007 / 0.036, -0.013 / 0.048 |
| Per layer: CPU vs primary transfer abs(Δ) error | 0.037, 0.021 | 0.043, 0.024 |
| Per layer: helper vs CPU; primary vs helper | 0.131 / 0.118; 0.104 / 0.093 | **0.052 / 0.011; 0.023 / 0.035** |
| Per layer: slowest minus second | 0.053, 0.043 | **0.022, 0.019** |

Trace pairs are prose then code, each with the structure fitted on the other trace.

**Reading.** Refitted, the model describes HAGS-on passes about as well as the HAGS-off model described HAGS-off passes.
The cycle error is lower than the HAGS-off check's 0.066 ms mean abs. In the traces the CPU end comes out 0.012-0.052 ms
early, where the records show no such bias; that likely comes from instrumentation on the CPU lane. The old model, applied
to HAGS-on work, misses the helper by 0.13 ms and the cycle by 0.11 ms, so it is retired for HAGS-on decisions.

**The policy replay reproduces the HAGS-on engine's decisions.** The replay runs the production policy over each untouched
capture's recorded demand from the load's origin
([result](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-validation-hags-20261107/replay_reproduction.txt)).
- Identical generation passes: 92.4% (extraction), 91.2% (reasoning), 77.5% (code), 92.1% (prose).
- Totals: CPU jobs within 0.9-2.4%, uploads within 0.6-1.0%.
- Exchanges match exactly: 612 primary issued, 611 published, 75-76 helper leases.
- The planner's prices come from the load's own staging calibration, carried in the origin: under HAGS an upload is priced at
  0.610 ms. So the replay follows the HAGS-on planner.

That is the basis for the offline parameter search: replay a candidate's work, price it with the HAGS-on model, and send
proposals to step 5's frozen predictions and hardware tests.

### Per-pass model, redo step 4

**At `T=1`, 2026-11-07.** [Plan](README.md#per-pass-redo). The repaired shared model's CPU, copy and helper terms, with the
5090's own path rebuilt from the trace (`shared_model/pass_validation.fit_structure`):
- demand copies start 0.018-0.033 ms after the lanes and run at the shared rates;
- the first dependent kernel follows after the handoff, 0.057 ms with the CPU done and 0.065 ms while it works;
- the dependent kernels span 0.060, 0.072 and 0.091 ms for one to three 5090 experts;
- a pass without an upload ends with its resident kernels: 0.117-0.122 ms with no routed 5090 expert, 0.159-0.182 ms with one to three.

Constants fitted on one trace are tested on the other. Each traced pass's work comes from the replay's own capture. The replay
re-decides staging, so the source capture's counts had not matched its passes, which had cut slowest-path agreement to 47-58%.

| Per pass, from its own origin | `T=1` prose | `T=1` code |
|---|---:|---:|
| Passes judged | 3,378 | 3,525 |
| Slowest path right | **97.7%** | **95.0%** |
| Baselines: always the CPU; most common per layer in the other trace | 49.3%, 48.1% | 53.9%, 49.1% |
| Top-two margin where wrong, median | 0.033 ms | 0.029 ms |
| CPU end error, mean and mean abs | -0.025, 0.041 ms | -0.018, 0.034 ms |
| 5090 end with an upload, old fitted term → traced structure | +0.058/0.067 → -0.015/0.037 ms | +0.072/0.078 → -0.005/0.037 ms |
| 5090 end, resident only | +0.001/0.021 ms | -0.007/0.019 ms |
| Helper end | -0.021/0.062 ms | -0.027/0.065 ms |

Per-layer mean abs(Δ) between paths, observed and predicted, the mean per-layer error, and the correlation across layers:

| Pair | `T=1` prose | `T=1` code |
|---|---|---|
| CPU vs primary GPU transfer | 0.176, 0.172; 0.011 (r 0.77) | 0.217, 0.210; 0.009 (r 0.93) |
| Helper GPU transfer vs CPU | 0.437, 0.412; 0.030 (r 0.63) | 0.574, 0.561; 0.023 (r 0.86) |
| Primary vs helper GPU transfer | 0.687, 0.679; 0.032 (r 0.06) | 0.677, 0.677; 0.024 (r 0.38) |
| CPU vs helper resident compute | 0.447, 0.422; 0.030 | 0.584, 0.570; 0.023 |
| Primary transfer vs helper resident compute | 0.696, 0.689; 0.032 | 0.686, 0.686; 0.024 |
| Slowest vs second-slowest | 0.231, 0.208; 0.024 (r 0.07) | 0.239, 0.222; 0.018 (r 0.41) |

On untouched full records (`val-extract`, `val-reason`; 17,710 and 21,065 passes):
- CPU ends are within 0.026-0.028 ms mean abs, and helper ends within 0.062-0.069.
- The CPU-helper order is right on 99.1-99.3% of passes, and per-layer CPU-helper abs(Δ) is within 0.028-0.034 ms.
- The cycle error moves from -0.006 to -0.018 ms mean with the traced structure, because the cycle constants were fitted with
  the old 5090 term. It changes no pass's slowest lane.

**Reading.** At `T=1` the model gets individual passes right. It names the path that finishes last on 95-98% of passes, where
a layer-aware guess manages about 50%, and it reproduces every pair's per-layer balance within about 0.01-0.03 ms. The low
correlations, primary vs helper and slowest minus second in prose, come where per-layer variation is small against that error.
Trace times include instrumentation. The structure will be refitted after the HAGS decision, since HAGS changes the handoff.
[Results](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-20261107/pass_validation_t1c.txt),
[JSON](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-20261107/pass-validation-t1c.json),
[structure by work](/mnt/wsl/transfer/minimax-wsl/runs/per-pass-trace-20261107/primary_structure_by_work.txt). Earlier passes
`pass_validation_t1.txt` and `pass_validation_t1b.txt` are superseded: one used the source capture's work, the other the held rule.

## Step 6: drafters, acceptance and the first offline model

<a id="dflash2-m1"></a>

**DFlash2 M1, 2026-11-09 (c8c9eba, fixed in f18bfdd; tasks 56-57): RadixArk's draft runs in Maya at MXFP8 and drafts at
DSpark's level of acceptance.**
- **The port** ([study](dflash2-port-study.md)).
  - `tools/mm3/pack_dflash2.py` packs the BF16 checkpoint: MXFP8 for q/k/v/o, gate/up/down and both conv projections
    (E4M3 codes, one E8M0 scale per 32 with exponent ceil(log2(amax/448))), BF16 for the rest. 2,357 MB on the 5090.
  - `--dflash2 <pack>` loads it in DSpark's place. `propose_d2` runs the block (the anchor and 7 masks) through the five
    layers with two new conv kernels, the target's BF16 head over the mask columns in one pass, each column's top 16, the
    selector's projection, and a new walk kernel (`dflash2_select`: greedy or drawn, writing q over the 16 candidates and
    the temperature-1 confidence).
- **The first form drafted noise:** 0% acceptance at confidence 1.00.
  - A bisection with `dflash2_debug` bits (convs off, selector off) showed the head's top candidate was already wrong.
  - `rms_kernel` writes its 16-bit copy as FP16, and the head's cuBLAS GEMM and the selector projection read it as BF16.
  - Fixed in f18bfdd: the head goes through `x3::mv` with the F32 rows, and the selector's input is converted explicitly
    (`k::to_bf16`).
- **The same commit added `bf16_mv_cols`,** the multi-column BF16 pass. A warp takes 4 rows and reads each weight once for
  up to 16 columns in `bf16_mv`'s FMA order, so it matches `bf16_mv` per column bit for bit. The dense test checks 2-16
  columns.

<a id="drafter-acceptance"></a>

**Both drafters' acceptance, same binary and arms, 2026-11-09**
([DFlash2](/mnt/wsl/transfer/minimax-wsl/runs/dflash2-acceptance-20261109b/acceptance.txt),
[DSpark](/mnt/wsl/transfer/minimax-wsl/runs/dspark-acceptance-20261109b/acceptance.txt),
[side by side](/mnt/wsl/transfer/minimax-wsl/runs/dflash2-acceptance-20261109b/compare.txt)). The setup:
- f18bfdd, the `T=1` base's mechanics, temperature 1.0, top_p 0.95;
- arms exact7 (7 drafts, exact rejection sampling, no floor), exact4 (4 drafts) and serving (4 drafts, lenience 0.4,
  floor 0.3);
- four prompts, two requests each at up to 1,024 tokens, timeline captures.

| Tokens a verify step, DFlash2 / DSpark | python-code | prose | val-extract | val-reason | Equal-weight mean |
|---|---:|---:|---:|---:|---:|
| exact7 | 3.74 / 4.71 | 2.32 / 2.28 | 3.79 / 4.10 | 4.57 / 4.58 | 3.61 / 3.92 |
| exact4 | 2.84 / 3.83 | 2.29 / 2.10 | 3.33 / 3.75 | 4.02 / 3.75 | 3.12 / 3.36 |
| serving | 3.05 / 4.12 | 2.09 / 2.37 | 3.47 / 4.03 | 3.95 / 4.06 | 3.14 / 3.65 |

- **The texts differ.** Sampling draws differ by drafter, so each cell compares different answers. DSpark's code answers
  were short: 594-750 tokens over two requests, against DFlash2's 2,048.
  - Pooled over prompts, the mix decides. DSpark spent half its exact7 rounds on prose, so its pooled 3.36 falls below
    DFlash2's 3.54.
  - So report per prompt and the equal-weight mean.
- **DSpark leads on code and extraction; they tie on prose and reasoning.** P(position k accepted) at exact7:

  | Positions 1-7 | DFlash2 | DSpark |
  |---|---|---|
  | code | 79, 56, 42, 33, 26, 20, 19% | 89, 74, 63, 50, 42, 29, 24% |
  | prose | 65, 35, 17, 7, 4, 2, 1% | 58, 31, 17, 9, 6, 3, 1% |
- **Whether MXFP8 costs DFlash2 acceptance is not measured.** There is no BF16 pack, and the card's numbers are from SGLang
  on an NVFP4 target at a 16K context.
- **The confidence each drafter gives** (exact arms, evaluated drafts, tables fitted leaving one prompt out;
  [model](#spec-model-v1)).
  - DSpark's head is calibrated as it comes: log-loss 0.381 raw. A table fitted on the other prompts is worse (0.427),
    so DSpark's planner uses the raw confidence, or a table with the identity as prior.
  - DFlash2's selector probability is informative but compressed. At position 1, drafts below 0.2 were kept 40% of the
    time, 0.2-0.4 54%, 0.8-1.0 91%; DSpark's were 8%, 22% and 94%.
  - A table per position and confidence bin fixes DFlash2's calibration. Its log-loss is 0.581 for a constant per
    position, 0.501 raw and 0.440 calibrated, still above DSpark's 0.381.
- **The two requests of each cell were one text.** With the protocol's one seed, 22 of the 24 repeat pairs wrote identical
  generations (all 12 of DSpark's, 10 of DFlash2's), so each cell is one generation. The first cross-validation left one
  request out and so kept the held-out request's twin in the fit; its figures (log-loss 0.418 for DFlash2's table, -17%
  for the width rule) were optimistic. The harness now takes a seed per step (b19bd0e).
- **The serving floor misfits DFlash2.**
  - 0.3 was set on DSpark's head. On DFlash2's selector probability it offered no draft on 18% of verify steps (26% on
    prose, 9% on reasoning), against 3% for DSpark, each after paying the draft call.
  - A floor is per drafter, or gone: the width rule below replaces it.
- **The draft call.**
  - DFlash2 takes 5.6-5.7 ms; its block is always 7 columns.
  - DSpark takes 6.2 ms at 4 drafts and 7.3 at 7. Before f18bfdd, with its head re-read per column, 7 drafts took
    about 15 ms.

<a id="verify-head-multicolumn"></a>

**f18bfdd also changed the target's verify windows.**
- `x3::mv_cols` returns 16 for a BF16 head. So the target's verify head (`mm3_model.cu:4465`) now reads the 2.46 GB head
  once per window rather than once per row. The comment there, "~5 rows a pass", is stale.
- The one-column head takes 1.476 ms at 1,666 GB/s (perpass-trace-new-20261109's `T=1` trace), so an 8-row window paid
  about 11.8 ms of head before.
- Speculative step times measured before f18bfdd do not compare with later ones.

<a id="dflash2-profile"></a>

**DFlash2's drafting pass under Nsight, 2026-11-09 (M2's first measurement;
[protocol](/mnt/wsl/transfer/minimax-wsl/runs/dflash2-prof-20261109/make_protocol.py),
[result](/mnt/wsl/transfer/minimax-wsl/runs/dflash2-prof-20261109/draft-trace.txt)).** One Nsight-launched load of
f18bfdd, two exact7 requests (code, prose) at up to 256 tokens: 62 and 109 proposals of 148 kernels each.
- **M1 is device-bound, not launch-bound.** A proposal spans 5.53-5.60 ms on the 5090, of which the device is busy 5.13-5.20
  ms; the traced gaps (0.40 ms) are an upper bound on WSL2. The engine's draft call is 5.6-5.7 ms.

  | Per proposal | Device ms | Note |
  |---|---:|---|
  | The head, `bf16_mv_cols` over 7 columns | **2.67** | 920 GB/s: half the proposal |
  | 45 MXFP8 GEMMs (`mxc_kernel`) + split-K reduces | 1.72 + 0.08 | 1,058 GB/s over 1.82 GB |
  | top-16 (`topk_rows`) | 0.25 | one 1,024-thread block a row: 7 of 170 SMs |
  | Attention | 0.13-0.22 | the ring grows with the context |
  | The selector's walk | 0.13 | one block walks the 7 columns |
  | Norms, rope, convs, swiglu, the selector projection | 0.13 | |
- **The GEMMs by matrix** (median a call): q 45.6 us (1,139 GB/s), o 46.8, gate 73.8, up 68.6, down 69.7 (1,056-1,135
  GB/s), against 1,490 GB/s for the same q at one row (`mx_rows`, [preparation](#preparation-projections)). The small
  ones sit at the latency floor: k and v 7.0-7.6 us (426-465 GB/s), the conv deltas 11.9-12.7 us.
- **The target's verify heads, 8 rows, took 2.68 ms (916 GB/s)** under the same kernel.

<a id="head-kernel-second-form"></a>

**The several-column head pass, second form, 2026-11-09 (task 61): 1.49-1.59 ms at 2-8 columns, bit for bit.**
- **Why the first form ran at 920 GB/s at any column count** (2.66 ms at 2 columns as at 8, so not the columns' x): it
  broke out of its row loop at n, between the four rows' weight loads. So the compiler could not issue row r + 1's load
  before row r's check, and each warp kept one load in flight a step.
- **The second form** loads every row unconditionally (rows past n read the last row; their sums are never written), with
  12 rows a warp at up to 4 columns and 8 at 5-8, 4 warps a block, x from global memory. Every (row, column) sum keeps
  `bf16_mv`'s lane order and reduction.
- **A head-shaped sweep on the 5090** (200064 × 6144, standalone, each variant bit for bit against `bf16_mv` per column):
  - staging x in shared memory a K-tile at a time was slower, 1.76-1.91 ms (the block syncs);
  - the chosen variants, in the engine's own test: **1.49 ms at 2 columns, 1.50 at 4, 1.57 at 7, 1.59 at 8**, against
    1.49 for one column, 2.66-2.71 first, and 2.96-11.78 a pass a column.
- **Exact:** the dense test checks 2-16 columns at 1,001 rows (a tail in every block shape) on both GPUs, and the bench
  checks 2, 4, 7 and 8 columns at the head's shape.
- **So about 1.1 ms comes off every proposal and every verify window of 2 rows or more**, DSpark's included. It has no
  quality mechanism; the next drafting load confirms it in the engine.

<a id="spec-model-v1"></a>

**The first offline speculative model, round level, 2026-11-09 (task 60's first layer;
[script](/mnt/wsl/transfer/minimax-wsl/runs/spec-model-20261109/spec_model.py),
[result](/mnt/wsl/transfer/minimax-wsl/runs/spec-model-20261109/spec-model.txt)).** It reads the two acceptance runs'
timeline captures and drops each request's first eighth of rounds (about token 128): 5,142 DFlash2 rounds and 3,888 DSpark
rounds.
- **The time model.** A round is the draft call, the verify forward and the acceptance phase (each evaluated draft's
  decision, then the residual or bonus token). The three sum to the measured round within 0.06 ms (median remainder).
  - Verify by rows, DFlash2's load pooled: 40.9 ms at 1 row; 70.6, 89.7, 108.1 and 127.1 at 2-5; 178.1 at 8. **The
    first extra row costs 30 ms, each later one about 19.** DSpark's load: 40.2, 63.4, 80.6, 100.6, 125.4, 169.8.
  - By prompt at 5 rows: code 137 ms, prose 110, extraction 131, reasoning 129.
  - The acceptance phase: 2.4 ms with no draft evaluated; 3.0, 5.2, 7.3 and 10.9 with 1-4; 17.1 with 7. **That is
    about 2.2 ms a row of host sampling** ([below](#host-sampling)).
- **The cross-check.** Under exact sampling, draft i's verdict does not depend on later drafts. So exact7's rounds truncated
  at 4 drafts, priced with exact4's times, predict exact4's ms a token.
  - DFlash2: 46.5 predicted against 46.6 measured. DSpark: 47.9 against 45.5.
  - Per prompt, within 16%: the texts differ.
- **Width, offline on the exact7 rounds.** T=1 here means no drafting; every other policy pays the draft call each round.

  | ms a token | T=1 | Best fixed width (drafts) | Acceptance-aware width | Oracle |
  |---|---:|---:|---:|---:|
  | DFlash2, all prompts | 43.3 | 43.6 (2) | **36.1** | 30.6 |
  | DFlash2, code | 43.3 | 41.8 (2) | 33.5 | 29.9 |
  | DFlash2, prose | 43.3 | 43.3 (T=1) | 45.9 | 36.3 |
  | DFlash2, extraction | 43.3 | 42.5 (2) | 35.6 | 30.1 |
  | DFlash2, reasoning | 43.3 | 39.4 (3) | 32.8 | 28.5 |
  | DSpark, all prompts | 42.4 | 41.4 (2) | **35.5** | 29.8 |

  - **At today's costs no fixed width beats `T=1` overall. The acceptance-aware rule is predicted at -15% (DFlash2)
    and -18% (DSpark) as equal-weight means over the prompts** ([policies](/mnt/wsl/transfer/minimax-wsl/runs/spec-model-20261109/spec-policy.txt)).
  - The rule: draft; map each position's confidence through the calibration table, fitted leaving the prompt out; take
    the reach R_k as the product of the first k. Then verify the e maximizing (1 + sum of R_k) / (draft + verify(e + 1) +
    the acceptance phase at the expected evaluated count). Mean e* is 2.2 for both.
  - **What did not help, offline** (`spec_policy.py`): skipping drafting after a round whose e* was 0 (-14.8% against
    -15.3% for the plain rule, both on the leaky first split); a gate on the realized saving (-9 to -10%: too noisy);
    each prompt's own verify-by-rows table (noisier, no better); a per-request logit offset on the table (prose +3.8% to
    +1.1%, but code and reasoning lose more).
  - **Prose never pays.** Both drafters lose to `T=1` there even with the rule, because the draft call is paid first. So
    the rule must also decide whether to draft at all, from the request's recent acceptance: history calibrates, it does
    not predict a step.
  - The model's `T=1`, 43.3 ms, is about 3% under the `T=1` timed tests (44.4-45.1 ms from token 128).

<a id="host-sampling"></a>

**Host sampling costs about 2.3 ms a row (task 62).**
- The model's recommended sampling is temperature 1.0, top_p 0.95, top_k 0. With it, `cands_for` (`mm3_sample.hpp:258`)
  declines candidates. So every row of a forward is copied whole to the host, and its nucleus is computed over 200,064
  logits.
- That costs 2.4 ms once a token at `T=1`, and about 2.2 ms more for each evaluated draft.
- A GPU nucleus would keep sampling exact. It would take each row's top-K candidates and its full softmax normalizer; the
  host computes the nucleus over the candidates whenever their mass reaches top_p, and over the whole row as now otherwise.
  To be bounded, then built.

<a id="planner-v1"></a>

**Planner v1, the acceptance-aware verify width, timed 2026-11-09 (b19bd0e; task 59): -13.7% against `T=1`, inside the
model's frozen -16.3%** ([protocol](/mnt/wsl/transfer/minimax-wsl/runs/plan-v1-20261109/make_protocol.py),
[evaluation](/mnt/wsl/transfer/minimax-wsl/runs/plan-v1-20261109/evaluation.txt),
[rounds](/mnt/wsl/transfer/minimax-wsl/runs/plan-v1-20261109/check-plan.txt)).
- **What it is.** `maya-mm3 --spec-plan <file>` loads, for the drafter, a calibrated acceptance per draft position and
  confidence bin, and the round's part times. With `spec_plan=1` each round offers the e drafts that maximize expected
  tokens over expected round time.
  - The table (`dflash2-plan.txt`) came from `tools/mm3/spec_plan.py` on the acceptance run's captures, shifted -1.1 ms
    for the head pass's second form.
  - Verification is exact and the drafts are a prefix: no quality mechanism.
- **The test.** One load and two arms: `T=1` (no drafting) and plan (up to 7 drafts, exact rejection sampling). Four
  prompts, eight blocks at up to 384 tokens, from token 128. Each block has its own seed, shared by both arms, so the
  blocks sample texts as well as timing.

  | Prompt | Measured, 95% interval | Frozen prediction | Medians, ms a token (`T=1` / plan) |
  |---|---|---:|---|
  | code | -21.1% [-25.8, -16.4] | -24.3% (inside) | 47.95 / 36.50 |
  | prose | +0.7% [-5.0, +6.3] | +3.6% (inside) | 41.60 / 41.46 |
  | extraction | -14.2% [-22.2, -6.2] | -19.2% (inside) | 44.73 / 40.35 |
  | reasoning | -20.2% [-22.8, -17.5] | -25.4% (outside) | 45.98 / 37.16 |
  | **Equal-weight** | **-13.7% [-16.4, -11.0]** | **-16.3% (inside)** | |

  **Under the rule, planner v1 becomes DFlash2's drafting base:** later step-6 changes are measured against it. No serving
  default changes.
- **Round by round** (the check and the two mechanism blocks: 1,017 plan rounds from about token 128).
  - **Exact to the model:** every width the engine chose equals the offline rule's, recomputed from the captured
    confidences.
  - **The acceptance model is calibrated in aggregate.** By expected tokens a round, expected against realized: 1.00 /
    1.00, 1.82 / 1.81, 2.29 / 2.22, 2.74 / 2.73, 3.42 / 3.41, 5.48 / 5.71.
    - By prompt it is not: prose realized 12% fewer than expected, reasoning 11% more. One table serves every prompt.
  - **The time model holds at 1-4 rows and is optimistic beyond.**
    - Verify medians at 1-4 rows: 41.2, 68.2, 87.6, 104.7 ms, against the table's 40.9, 69.5, 88.6, 107.0.
    - At 5-8 rows: 133.9, 165.8, 185.1, 213.6, against 126.0, 143.0, 160.0, 177.0.
    - The 8-row figure came from the exact7 arm, where every window had 8 rows. Under the planner a wide window is rare
      among narrow ones, and the residency the narrow ones leave serves it worse [inferred; the window model per layer
      checks it].
  - **The acceptance phase follows the tokens a round produces, not the drafts evaluated.**
    - Each produced token costs one target distribution, the bonus row's when every draft is kept. So a one-draft round
      that keeps its draft costs two.
    - The planner's fully kept rounds made 1-3 evaluated drafts cost 4.18, 6.58 and 9.09 ms, against the table's 2.98,
      5.23 and 7.32.
  - These two explain most of the 2.6 points by which the frozen -16.3% was optimistic.
- **Next for the planner:**
  1. the GPU nucleus (task 62), which removes most of the acceptance phase;
  2. a table refitted on the planner's own regime: wide windows as the planner meets them, and the acceptance phase by
     produced tokens;
  3. a frozen prediction and a timed test.

<a id="gpu-nucleus"></a>

**The GPU nucleus, first round, 2026-11-09 (681d618; task 62): it works, but stays off under its rule, and the rule's
exactness check was wrong for `T=1`** ([protocol](/mnt/wsl/transfer/minimax-wsl/runs/gpu-nucleus-20261109/make_protocol.py),
[evaluation](/mnt/wsl/transfer/minimax-wsl/runs/gpu-nucleus-20261109/evaluation.txt),
[rounds](/mnt/wsl/transfer/minimax-wsl/runs/gpu-nucleus-20261109/rounds.txt)).
- **What it does** (`gpu_nucleus=1`). At the model's sampling (temperature 1, top_p 0.95, no top_k, min_p or penalties),
  the decode and verify forwards keep their head rows on the 5090.
  - `topk_rows` (512 candidates) and a new `row_stats` kernel (maximum, total of exp(x - max), finite) come back instead
    of 200,064 logits a row.
  - `mm3::sample::nucleus_from` walks `adaptive_nucleus`'s own counts over the candidates. A row it cannot settle is
    fetched whole and goes through `dist()` as before.
  - `tests/mm3_nucleus_test.cu`: 49 of 128 rows settle from candidates, all bit for bit with `dist()`.
- **The mechanism works.** 9,010 of 9,012 `T=1` rows and all 8,928 planner rows settled from candidates. The acceptance
  phase fell from 2.32 ms to 0.02 at `T=1`, and from 2.5-17.7 ms to 0.03-0.23 ms a round under the planner.
- **The effects.**
  - Planner, plann over plan: **-5.0% [-9.1, -0.9]**, -2.0 ms a token.
  - `T=1`, t1n over t1: -0.15% [-4.9, +4.6].
- **Why `T=1` gained so little: the forward grew 1.35 ms.**
  - `row_stats` took 0.73 ms a row: one block summed 200,064 double exponentials on one SM, and FP64 runs at 1/64 rate
    on the 5090.
  - `topk_rows` took 0.23 ms.
  - The other ~0.4 ms is work that had overlapped the host's sampling.
- **The rule's exactness check was a protocol error, not a finding.** It required the paired arms to write the same text
  (the same seed). That contradicts the plan's [rules for adoption](README.md#adoption-rules) of the day before: the engine
  is designed for speed, not repeatability, so text differs between runs and between arms by design. Every `T=1` pair
  differed, as that design implies; it says nothing about the nucleus.
  - Exactness is checked inside the engine instead (`gpu_nucleus=2`), and in tests of the data a feature computes. That
    is what "bit for bit" means throughout this record (helper_zero_copy, prep_graph and the rest): the data each change
    moves, never whole texts.
  - Under its rule `gpu_nucleus` stayed off in this round.
- **Second form, to be timed:**
  - `row_stats` runs over 64 blocks a row, with FP32 `expf` summed in double, in a fixed order: 0.73 to 0.02-0.04 ms
    a row.
  - `nucleus_from` widens its guard band to 8 times the total's error bound (2.5e-7). So wherever it settles, it settles
    where `adaptive_nucleus` does. In the test: the same 49 rows settle, bit for bit, totals within 3.5e-8.
  - `gpu_nucleus=2` checks real rows inside the engine: `dist()` on the fetched row beside each settled nucleus, its
    result sampled, matches and mismatches counted. This replaces the output check.
- **For planner v1.1, offline.** With the nucleus-era costs (verify at 5-8 rows as the planner meets them, the acceptance
  phase by produced tokens), the replay says:
  - the correctly priced rule is *more conservative* and slightly worse than v1's: -8.2% against -9.5% equal-weight;
  - v1's understated wide windows had offset the per-prompt miscalibration of acceptance;
  - with the per-request calibration (a logit offset, lambda 0.4), the correctly priced rule is best: -10.4%;
  - the replay rests on one text per prompt, so the next campaign collects exact7 rounds under new seeds first.

<a id="gpu-nucleus-second-round"></a>

**The GPU nucleus, second round, 2026-11-09 (1bdd4f6): exact on real rows, -5.7% at `T=1` and -5.8% under planner v1;
adopted** ([protocol](/mnt/wsl/transfer/minimax-wsl/runs/gpu-nucleus2-20261109/make_protocol.py),
[evaluation](/mnt/wsl/transfer/minimax-wsl/runs/gpu-nucleus2-20261109/evaluation.txt),
[rounds](/mnt/wsl/transfer/minimax-wsl/runs/gpu-nucleus2-20261109/rounds.txt)).
- **Exact.** The in-engine check (`gpu_nucleus=2`, one `T=1` and one planner request) compared every settled row's
  nucleus with `dist()` on the fetched whole row: 383 + 383 rows, 0 mismatches. 3 of 21,205 rows fell back to the whole
  row.
- **The timed test** (six seeded blocks, four prompts, from token 128):

  | Pair | Effect, 95% interval | Frozen | ms a token |
  |---|---|---:|---|
  | `T=1`: t1n over t1 | **-5.70% [-6.94, -4.45]** | -3.7% (outside: better) | 45.43 to 42.85 (-2.58) |
  | planner v1: plann over plan | **-5.76% [-9.48, -2.04]** | -5.8% (inside) | 38.81 to 36.57 (-2.24) |

  Every prompt was faster in both pairs. **Under the rule, `gpu_nucleus=1` joins the `T=1` base and planner v1's
  drafting base.**
- **The mechanism** (one timeline block):
  - At `T=1` the forward stayed 41.6 ms (41.61 to 41.66) while sampling fell from 2.38 ms to 0.02. The first round's
    extra 1.35 ms in the forward was the slow `row_stats` itself.
  - Under the planner, the acceptance phase fell from 2.6-18.1 ms a round to 0.02-0.24.
  - The draft call grew 0.45 ms (4.62 to 5.07). It now starts as soon as the forward ends and meets GPU work left from
    the forward [inferred].
- **Planner v1.1, not adopted, on the replay** (`spec_policy.py --extra`: the acceptance run's texts plus this
  campaign's 16 exact7 requests under seeds 4001-4004, LOPO tables; every policy priced at the nucleus era's costs,
  `plan-v11-20261109/dflash2-plan-v11.txt`):
  - v1's decisions: -9.9% equal-weight against `T=1`;
  - v1.1's decisions: -8.5%; with per-request calibration -9.1 to -9.5%;
  - the oracle: -24.2%.
  - **Correctly priced, the rule chooses narrower windows than are worth it:** the acceptance model predicts too low on
    code and reasoning. v1's understated wide-window costs had pushed the other way. So the remaining headroom (-10% to
    -24%) is in predicting each round's acceptance.
  - The engine supports per-request calibration (`adapt` in a plan file, `plan_delta` in the stat line) for when an
    acceptance model needs it.
- **The model's checks against this campaign:**
  - the replay put planner v1 at about 37.0 ms a token, against 36.57 measured;
  - its `T=1` reference, 41.1 ms (the planner's one-row forwards, medians), is about 1.7 ms under the `T=1` arm's
    measured mean (42.85). The `T=1` arm's one-row forward is about 0.6 ms slower than the planner's, and the means
    carry tails the medians do not. So the replay understates the planner's gain against `T=1`. That does not affect
    comparisons between rules.

<a id="acceptance-model"></a>

**The acceptance model on eight prompts, 2026-11-09 (task 64; no decision)**
([data](/mnt/wsl/transfer/minimax-wsl/runs/accept-data-20261109/make_protocol.py),
[study](/mnt/wsl/transfer/minimax-wsl/runs/accept-data-20261109/accept-model.txt)).
- **The data:**
  - DFlash2 drafting 7 every round with exact sampling and the GPU nucleus;
  - all eight prompts the harness holds (the four so far, plus val-code, val-prose, conditioning-code and
    conditioning-prose), under three seeds;
  - 8f5e38d's capture fields: each target distribution's top probability and nucleus size, and each draft's own q;
  - 2,656 rounds and 7,545 reached drafts.
- **Predictors, scored leaving one prompt out**, then each one's planner replayed at the nucleus era's costs:

  | Predictor | Log-loss | Brier | Replay against `T=1` |
  |---|---:|---:|---:|
  | constant per position | 0.636 | 0.222 | +7.4% |
  | planner v1's table (position x 10 confidence bins) | 0.519 | 0.172 | -5.3% |
  | **logistic: position + logit(confidence)** | **0.476** | **0.162** | **-6.8%** |
  | + the draft's own q (entropy, how far down its candidates it drew) | 0.476 | 0.162 | -7.0% |
  | + the anchor's target distribution (top probability, nucleus size) | 0.476 | 0.162 | -6.9% |
  | + the previous round's accepted share | 0.474 | 0.162 | -6.9% |
  | planner v1 as deployed (the table deciding with its own old times) | | | -4.9% |
  | oracle | | | -21.7% |

- **The gain is the form, not new features.** A smooth logistic in logit(confidence) shares strength across positions.
  The table's cells at deep positions are sparse and noisy.
  - The target's own uncertainty at the anchor does not predict the next drafts' verdicts. Neither does the drafter's q
    entropy.
- **On a broader mix, drafting gains less.** Even the best predictor replays at -7% against `T=1`.
  - Prose-like prompts lose with any rule: val-prose +7.5%, prose +1.6%.
  - The four prompts used so far flattered drafting (the replay on them: -10%).
- **Skipping drafting after a round whose best width was 0** recovers part of val-prose (+7.5% to +4.5%), but loses
  more on code and reasoning (-6.4% overall against -6.8%).
- **What the oracle's -22% needs** is the verdicts of an exact sampler at temperature 1. They stay noisy given anything a
  planner has before the verify.
- **So planner v2** (`spec_plan.py --accept logistic`: the logistic plus the nucleus era's costs) is predicted at -2.1%
  per token against v1, timed in `planner-v2-20261109`.

<a id="planner-v2"></a>

**Planner v2 timed, 2026-11-09 (72ed95a): no better than v1 on hardware, so v1 stays; both planners -8.7% against
`T=1` on eight prompts** ([protocol](/mnt/wsl/transfer/minimax-wsl/runs/planner-v2-20261109/make_protocol.py),
[evaluation](/mnt/wsl/transfer/minimax-wsl/runs/planner-v2-20261109/evaluation.txt)).
- **The test.** One load held both plan files (`--spec-plan` twice; a request picks one with `spec_plan=N`). Arms: `T=1`,
  planner v1 and planner v2, all with the GPU nucleus. All eight prompts, five seeded blocks at up to 384 tokens, from
  token 128.
- **Exact to the model:** all 70 widths of the check request match the v2 file.

  | | Equal-weight, 95% interval | Replay's prediction |
  |---|---|---:|
  | **v2 over v1** | **+0.95% [-2.68, +4.57]** | -2.1% (inside) |
  | v1 over `T=1` | -8.76% [-11.62, -5.90] | -4.9% |
  | v2 over `T=1` | -8.56% [-12.26, -4.86] | -6.8% |

  - By prompt, v2 over v1: prose -6.5%, val-prose -3.4%; but val-reason +6.0%, val-extract +4.3%, code +1.6%, both
    conditioning prompts +2%.
  - v2 chose narrower windows: mean width 1.40 against v1's 1.64.
  - **Under the rule (replace v1 only if the mean is below zero), planner v1 stays the drafting base.**
- **What it says about the offline model:**
  - The replay's absolute figures hold once its `T=1` reference (41.06 ms, medians of the planner's one-row forwards) is
    replaced by the measured `T=1` mean (42.85 ms). v1's replay -4.9% then becomes about -8.9%, as measured.
  - But the replay cannot rank policies about 2% apart: v2 was predicted 2.1% better and measured 1% worse.
  - A better acceptance model changed widths without a hardware gain. Narrower windows changed which experts each window
    needs, and the time model holds verify-by-rows fixed, whatever the width mix. That is the per-window, per-layer model
    task 60 still owes.
- **Against `T=1`, drafting with planner v1 and the nucleus gains 8.8% on the broad mix.**
  - Code -21%, reasoning -25%, conditioning-code -16%, extraction -11%.
  - Prose +1.9%, val-prose +6.0%, val-code -0.9%, conditioning-prose -3.3%.

<a id="window-model"></a>

**The window model per pass, 2026-11-09 (72ed95a): refitted and qualified on verify windows. On window passes the CPU
lane finishes after the 3090 on 91-100% of passes, and a better same-pass CPU-or-copy split is worth under 1%**
([protocol](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/make_protocol.py),
[model](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/window-model.txt),
[division](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/division.txt)).
- **The captures.** One load, light captures (every pass's lane boundaries and work) from the restored origin at up to 384
  tokens, all at the `T=1` base with the GPU nucleus:
  - `T=1` on four prompts;
  - planner v1 on all eight prompts, seeds 6001 and 6002;
  - fixed widths 2, 4 and 7 (`spec` 3, 5 and 8, no floor) on four prompts.

  32 captures, no step errors.
- **The refit** (`tools/mm3/shared_model/grouped.py`, the 24 seed-6001 captures):
  - a CPU job costs 0.557 ms, with no cost per column;
  - a demand copy costs 0.553 ms, share 0.878;
  - the 3090 costs 0.021 ms a job and 0.039 ms a column;
  - the 5090's base rises with rows, from 0.19 ms at 2 rows to 0.59 ms at 8.
- **Qualified pass by pass and on held-out captures:**
  - **By rows and new copies, against the observed mean cycle:**
    - up to two new copies: within -0.12 to +0.16 ms, mostly within 0.06;
    - three new copies: underpriced by up to 0.29 ms;
    - four or more: underpriced by 0.17-0.43 ms.
  - **Held out:** planner v1's seed-6002 captures, all eight prompts. Priced from each capture's own recorded work,
    forward time came within a median |miss| of 1.07% (-1.33% to +1.61%).
- **Per-pass balance by window width** (CLAUDE.md's central objective, applied to windows). The light records hold two
  lanes, the CPU and the 3090. The table counts passes with both. Completion times are measured from the pass's plan,
  the common origin.

  | Rows | Passes | Share of cycle | CPU last | Mean abs(CPU - 3090), ms | Per layer, ms | CPU jobs | 3090 jobs | New copies | Cycle, ms |
  |---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
  | 1 | 3,174 | 1.4% | 100% | 0.97 | 0.90-1.23 | 1.00 | 1.10 | 1.03 | 1.52 |
  | 2 | 19,468 | 8.5% | 100% | 0.74 | 0.62-1.05 | 1.08 | 2.56 | 1.06 | 1.51 |
  | 3 | 50,405 | 24.9% | 100% | 0.77 | 0.52-1.06 | 1.24 | 3.68 | 1.21 | 1.70 |
  | 4 | 14,150 | 8.4% | 97% | 0.94 | 0.53-1.40 | 1.48 | 4.37 | 1.55 | 2.04 |
  | 5 | 29,817 | 19.7% | 93% | 1.02 | 0.46-1.68 | 1.73 | 5.32 | 1.60 | 2.27 |
  | 6 | 4,738 | 3.9% | 96% | 1.44 | 0.68-2.36 | 2.23 | 5.48 | 2.20 | 2.82 |
  | 7 | 2,250 | 2.2% | 96% | 1.78 | 0.73-3.22 | 2.81 | 5.65 | 2.42 | 3.31 |
  | 8 | 32,609 | 31.0% | 91% | 1.63 | 0.83-3.05 | 2.78 | 6.67 | 2.29 | 3.27 |

  - **The 3090 waits on the CPU on nearly every window pass, and longer as windows widen.** A CPU job costs about
    0.56 ms; the 3090's resident work costs about 0.02 ms a job plus 0.04 ms a column.
  - **Open:** the copy-plus-primary metrics. The 5090's copies and kernels are not in light captures, so CPU against
    copy-plus-primary and the slowest against the second slowest need an Nsight trace of windows beside their light
    captures.
- **The CPU price for windows is bounded below 1%**
  ([moves](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/division-moves.txt)). For each pass with CPU jobs or
  new copies, the model priced every split of its uploadable experts between CPU jobs and demand copies, up to 4 uploads.
  Demand is fixed, and a copy's later reuse is not counted.
  - The best split saves 0.67% of planner v1's cycle (fixed widths 0.18-0.67%; `T=1` 1.79%).
  - **Nearly all of it comes from fewer uploads:** 1,371 of planner v1's 1,429 ms, and 461 of w7's 464 ms. Those uploads
    would have been reused, so the figure overstates the gain.
  - Allowing up to 8 uploads adds savings only at 8 rows (planner v1 1.11% in all, w7 1.67%). All of it is on passes with
    4 or more copies, which the model underprices, so it is not credible.
  - **A window CPU price is not tested on hardware:** it is worth under 2% and would add a parameter.
  - **Superseded the same day by the refit below.** Under the measured window model (v4) the bound is 1.5% for planner v1
    and 1.5-2.3% at 2-5 rows. At 2-8 rows it points the other way, to more uploads.
  - **"Would add a parameter" was the wrong objection** (user direction, 2026-11-09). The speculative-decoding planner is
    a different planner from the `T=1` one, so window-specific prices and policies are its own, not added cost.

<a id="window-paths"></a>

**Per-pass paths in verify windows, 2026-11-09 (72ed95a): with the 5090 in view, the CPU lane and the copy-plus-primary
path take turns finishing last; the 3090 finishes 1-2 ms early on wide windows**
([protocol](/mnt/wsl/transfer/minimax-wsl/runs/window-trace-20261109/make_protocol.py),
[paths](/mnt/wsl/transfer/minimax-wsl/runs/window-trace-20261109/window-paths.txt),
[by layer](/mnt/wsl/transfer/minimax-wsl/runs/window-trace-20261109/window-paths-by-layer.txt)).
- **The trace.** One Nsight-launched load (CUDA only, so `prep_graph` and `early_copy` stay on). Five requests, each from
  the restored origin at up to 384 tokens, seed 6001, each light-captured beside its trace, all at the `T=1` base with the
  GPU nucleus:
  - `T=1` on prose;
  - planner v1 on python-code and prose;
  - w7 on python-code and prose.

  No step errors.
- **One clock** ([`native_window_paths.py`](../../tools/mm3/native_window_paths.py)). The anchor is host to host. The host
  writes the word the 5090's `wait_publication` spins on, launches `add_kernel`, then stamps the light record's `end`.
  - **The fit:** a rolling median of that launch call's end against the stamp, over 401 passes. Residuals are -2.4 to
    +1.2 µs (p1-p99), and the median wanders by up to 25 µs over a trace.
  - **GPU times, checked against the launch's start:** `wait_publication` ends at the median -0.6 µs from it (p10-p90 -34
    to +6 µs).
  - **A first fit, on the device anchor, was wrong** (`wait_publication`'s end against `end`). It put GPU ends 30-138 µs
    late, because Nsight's `add_kernel` launch call takes 28-132 µs (p10-p90) between the write and the stamp. It was
    replaced before any reading.
- **Completion from the pass's plan, as mean abs(Δ) in ms, from token 128.**
  - *5090* is its arrival at `wait_publication`; on passes with demand uploads that is the copy-plus-primary path.
  - *3090* is the end of its graph. With zero-copy its transfer and resident compute end together, so the pairs against
    its compute path equal those against its transfer path.

  | Arm, prompt | Rows | Passes | Last: CPU / 5090 / 3090 | CPU vs 5090 with uploads | 3090 vs CPU | 5090 with uploads vs 3090 | Slowest - second | Per layer |
  |---|---:|---:|---|---:|---:|---:|---:|---:|
  | `T=1`, prose | 1 | 12,888 | 1 / 82 / 17% | 0.22 | 1.09 | 0.69 | 0.15 | 0.07-0.54 |
  | Planner v1, code | 1 | 2,607 | 2 / 79 / 19% | 0.13 | 0.99 | 0.67 | 0.19 | 0.04-0.50 |
  | | 2 | 1,105 | 26 / 56 / 17% | 0.23 | 0.88 | 0.74 | 0.28 | 0.19-0.93 |
  | | 3 | 1,342 | 36 / 54 / 11% | 0.26 | 0.90 | 0.87 | 0.28 | 0.18-0.74 |
  | | 4 | 336 | 49 / 45 / 6% | 0.29 | 1.14 | 1.12 | 0.29 | 0.11-0.80 |
  | | 5 | 280 | 47 / 48 / 5% | 0.33 | 1.24 | 1.30 | 0.33 | 0.17-1.08 |
  | | 6 | 224 | 54 / 42 / 4% | 0.37 | 1.37 | 1.45 | 0.36 | 0.10-1.53 |
  | | 8 | 280 | 72 / 26 / 1% | 0.84 | 2.50 | 1.89 | 0.82 | 0.11-3.54 |
  | Planner v1, prose | 1 | 3,360 | 1 / 81 / 18% | 0.22 | 1.13 | 0.67 | 0.15 | 0.07-0.36 |
  | | 2 | 2,880 | 19 / 54 / 27% | 0.21 | 0.82 | 0.63 | 0.29 | 0.20-0.76 |
  | | 3 | 1,061 | 27 / 54 / 19% | 0.23 | 0.81 | 0.71 | 0.27 | 0.12-0.57 |
  | | 4 | 559 | 38 / 48 / 13% | 0.28 | 0.94 | 0.92 | 0.28 | 0.12-0.85 |
  | w7, code | 8 | 3,416 | 72 / 22 / 5% | 0.65 | 1.95 | 1.55 | 0.61 | 0.25-2.17 |
  | w7, prose | 8 | 6,215 | 53 / 27 / 20% | 0.34 | 0.96 | 1.00 | 0.33 | 0.22-1.05 |

- **The light records' "CPU last on 91-100%" compared the CPU with the 3090 only.** With the 5090 in view:
  - **8-row windows:** the CPU lane is last on 53-72% of passes and the copy-plus-primary path on 22-27%. They end
    0.34-0.65 ms apart on average.
  - **Windows of 1-3 rows:** the 5090's own path is last on 54-82% of passes. The host lanes and the publication are done
    before it arrives (its wait is 2-9 µs at the median).
  - **This agrees with the division bound above:** the CPU-or-copy split is already near its best.
- **The 3090 is the lane with slack.** It finishes 0.8-1.1 ms before the CPU at 1-3 rows, and 1.0-2.5 ms before at 8
  rows. On layers 17, 23-34 and 48-53 of the code windows the gap reaches 3-4.4 ms, and slowest - second reaches 0.8-2.2 ms.
  - Using that slack takes more of the window's experts on the 3090: placement for windows, or more experts per GB there.
    A CPU price cannot do it.
- **The host's publication is serial on every pass.** It comes 0.012-0.020 ms after the host's last lane at 1 row and
  0.10 ms at 8 rows (under Nsight). That is mostly the merge of the 3090's results, about 2-3% of a forward.
- **The window model's 5090 path is worth refitting.** The grouped model prices it as a base by rows plus the copy end,
  with no cost per job, yet it binds on most narrow-window passes. These traces measure it pass by pass.
- **Nsight's cost was checked.** The same arm, prompt and seed ran untraced in window-passes-20261109
  ([comparison](/mnt/wsl/transfer/minimax-wsl/runs/window-trace-20261109/nsight-cost.txt)). The 5090's submission ends
  0.02 ms later at 1 row and 0.07-0.13 ms later at 5-8 rows; the `T=1` cycle is 0.5% longer. Text differs between runs
  by design, so the traced and untraced windows' cycles compare only as distributions.

<a id="window-model-v4"></a>

**The window model with measured paths (v4), 2026-11-09: three terms measured in the traces replace fitted or assumed
ones. It names each pass's later path (CPU or 5090) right on 85-92% of passes at 2-8 rows, where v1 managed 47-90%**
([copy rate](/mnt/wsl/transfer/minimax-wsl/runs/window-trace-20261109/copy-rate.txt),
[head start](/mnt/wsl/transfer/minimax-wsl/runs/window-trace-20261109/copy-head.txt),
[5090 terms](/mnt/wsl/transfer/minimax-wsl/runs/window-trace-20261109/primary-terms.txt),
[fit](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/window-model-v4.txt),
[per pass](/mnt/wsl/transfer/minimax-wsl/runs/window-trace-20261109/qualify-paths.txt)).
- **What the traces showed, and v4 takes in** (`tools/mm3/shared_model/grouped.py`, options of `fit`):
  - **The copy's share beside the CPU is 0.66, not 0.878.** A demand upload takes 0.552 ms alone and 0.836 ms entirely
    beside the CPU lane. Uploads within a pass run back to back. The cycle-level fit could not see the copy rate on
    passes the CPU binds. The CPU's share then follows from the fitted stretch: 0.64. That is 36 + 38 GB/s of host memory,
    as the `T=1` refit found.
  - **Uploads start before the CPU lane.** They are issued early in the 5090's submission, and the CPU lane starts after
    it. The head start is measured untraced from the light records' copy submissions, less the 0.045 ms from submission to
    DMA that the trace shows: 0.11 ms at 2 rows, rising to 0.52 ms at 8.
  - **The 5090's path is the later of its resident work and its uploads.** Without uploads it arrives 0.03-0.07 ms before
    the CPU lane's start plus 0.047 ms a job. With uploads it arrives a tail after the last one: 0.17 ms at 2 rows, 0.49 ms
    at 8. v1 added a base of 0.39-0.59 ms at 4-8 rows that had defaulted, too few bare passes to fit it.
- **Qualified on cycles, as v1 was:**
  - The cycle error by rows and copies is within -0.16 to +0.16 ms. v1's -0.17 to -0.43 ms at 4+ copies falls to
    -0.10 to -0.16.
  - The held-out planner captures miss forward time by a median 1.12%, against v1's 1.07% (eight captures do not separate
    them).
- **Qualified per pass, against the traces** (the CPU's end and the 5090's arrival from each pass's recorded work, moved
  to the plan by the pass's own submission):

  | Rows | Passes | Later path named right, v1 / v4 | 5090 error, v1 / v4 (ms) | CPU error, v4 (ms) | Per-layer error of mean abs(CPU - 5090), v4 (ms) |
  |---:|---:|---|---|---:|---:|
  | 2 | 1,368 | 90% / 90% | +0.02 / +0.06 | +0.02 | 0.046 |
  | 3 | 1,454 | 89% / 89% | +0.04 / +0.07 | +0.05 | 0.036 |
  | 4 | 678 | 72% / 92% | +0.22 / +0.10 | +0.05 | 0.061 |
  | 5 | 403 | 60% / 91% | +0.28 / +0.13 | +0.09 | 0.082 |
  | 6 | 313 | 68% / 91% | +0.29 / +0.09 | +0.10 | 0.064 |
  | 8 | 9,454 | 47% / 85% | +0.54 / +0.13 | +0.10 | 0.045 |

  - **One-row passes stay a cycle fit.** The window model's one-row form is sequential and names the later path right on
    35% of the 385 one-row passes with CPU jobs. The `T=1` per-pass model covers those.
- **v4 is the window model from here on** (`model-grouped-windows-v4.json`).
- **The CPU-or-copy bound under v4 points to more uploads**
  ([bound](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/division-v4.txt),
  [moves](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/division-moves-v4.txt)).
  - **Totals:** planner v1 1.53% of its cycle (1.75% with up to 4 more uploads), w2 2.26%, w4 1.54%, w7 0.44%.
  - **At 2-8 rows the saving is mostly from more uploads.** For planner v1 at 2-4 rows, more uploads save 1,809 ms and
    fewer uploads 307 ms.
  - **One-row passes still point to fewer uploads.** That is the same-pass view the `T=1` CPU price test overruled because
    of reuse.
  - **Reuse, not counted here, adds to the more-uploads side**, so in windows this bound understates rather than
    overstates.
  - **This is the bound for the planner rebuild in windows.** The CPU price is to be replaced by a planner that prices
    each pass by completion time from the measured GB/s (user direction, 2026-11-09). The speculative-decoding planner is
    a separate planner from the `T=1` one. With v4's measured terms, the best split per pass is that planner's same-pass
    decision, so the 1.5% above, before reuse, is what it can gain over today's prices in planner v1's windows.

<a id="proposal-comparison-windows"></a>

**Step 6's design comparison, 2026-11-09 evening: the six timing proposals against the window evidence. Window model v4
stays the pricing basis for the window planner's bound, with a support check added**
([proposals](timing-predictor-proposal-measured-transitions.md),
[refill overlap](/mnt/wsl/transfer/minimax-wsl/runs/window-trace-20261109/refill-overlap-windows.txt)).
- **Why now.** Step 6 starts with this comparison ([plan](README.md#step-6-next)). It was skipped on 2026-11-09: the window
  model was refitted from the `T=1` model and extended with traced terms (v4) without it.
- **What the window evidence asks of a model:**
  - each path's completion time from a common origin. The central objective's metrics need them, and so does a planner
    that moves work between the CPU lane and uploads;
  - host memory shared by the CPU lane and the copies: a copy takes 0.552 ms alone and 0.836 ms beside the CPU;
  - the 5090's path as the later of its resident work and its last upload plus a tail;
  - prices that hold on passes the planner did not choose. The bound's gain lies in more uploads at 2-4 rows.
- **The six by family:**

  | Family | Proposals | What is priced | Each path's completion | For windows |
  |---|---|---|---|---|
  | Whole-interval transitions | Terra (nearest neighbours), primary agent 1 (log-duration regression with readiness heads), sol-6.1 (regression tree) | an interval's elapsed time and its readiness answers, jointly | no | not a pricing basis: they cannot say which path ends last, or price moving work between paths |
  | Configuration surrogate | primary agent 2 | candidate minus control per configuration, from paired hardware trials | no | the search layer of the offline optimization after the rebuild |
  | Lane-structured window cost | Fable (a base plus the largest of affine lanes, plus bytes in flight), Opus (a base plus the larger of the CPU and helper branches, an upload price, randomized for causal prices) | a window as a base plus its latest lane | yes | v4's family. v4 replaces fitted lane slopes with rates measured in traces, and adds the host-memory coupling both forms leave out |

- **What v4 takes from the others:**
  1. **A support check** (Terra, primary agent 1, sol-6.1). The bound reports how much of its gain comes from passes with
     more new copies than were observed at their width. It prices those with v4's measured underpricing at 4 or more
     copies (0.10-0.16 ms a pass) as a pessimistic bracket.
  2. **Transfers in flight from earlier passes** (Fable's bytes in flight, primary agent 1's pending bytes). Today's window
     traces settle the size. 3090 refills overlap 0.4-2.7% of demand-copy time, which costs 0.03-0.12% of a request at the
     measured -23% copy rate. The 5090's exchange copies never overlap demand copies. So the rebuild charges refills beside
     demand copies, as the user wants them kept apart, but the term cannot move the bound.
  3. **Window-level pairing** (Fable). One model prices candidate and control on the same passes, so passes they leave
     alike contribute nothing. The faithful replay already does this.
- **Not taken now: randomized prices** (Opus). v4's terms are rates measured in traces, not slopes regressed on the
  planner's own choices. So the confounding Opus names, uploads issued where the planner expects a gain, does not bias
  them. Randomizing would matter only if the bound reaches 2% only through passes outside the support check. Then
  randomized extra uploads in one traced load, bounded first, would price them. Opus's one-layer lag terms: demand copies
  are joined within their own pass, and refills, which cross passes, are bounded above.
- **For the speculative model as a whole** (step 6's core item 1), all six proposals put drafting acceptance out of scope.
  Acceptance and the drafter come from the acceptance model and the drafter's measured forward. A step's time is the
  draft, each layer's window (v4), the time outside windows by width (planner v1's verify table, to be refreshed on
  today's base) and the acceptance phase.
- **Decision.** v4 prices the window planner rebuild's bound, with the support check. One-row passes stay with the `T=1`
  per-pass model: v4's one-row form names the later path right on only 35% of them.

<a id="window-planner-bound"></a>

**The window planner rebuild, bounded 2026-11-09 evening: a planner that stages each window pass by v4's predicted
completion gains 0.06-0.55% where demand is unbiased, and loses 1.4-2.6% on the control's own picks. Under 2%, and not
simpler, so not built**
([bound](/mnt/wsl/transfer/minimax-wsl/runs/planner-rebuild-windows-20261109/bound.py),
[on the control's picks](/mnt/wsl/transfer/minimax-wsl/runs/planner-rebuild-windows-20261109/bound-myopic.txt),
[with a credit per upload](/mnt/wsl/transfer/minimax-wsl/runs/planner-rebuild-windows-20261109/bound-credit.txt),
[on the router's picks](/mnt/wsl/transfer/minimax-wsl/runs/planner-rebuild-windows-20261109/bound-original.txt),
[residency](/mnt/wsl/transfer/minimax-wsl/runs/planner-rebuild-windows-20261109/diagnose_residency.py)).
- **The candidate.** Today's `plan_batch_staging` adds uploads greedily while max(CPU × price, 5090 base + uploads × h2d,
  3090) falls: prices added per path, with the CPU price 1.5 standing in for host-memory contention. The candidate prices
  each choice with window model v4, as the [design comparison](#proposal-comparison-windows) decided: host-memory shares,
  the uploads' head start, the 5090's path, the 3090. On each pass of 2 or more rows it uploads the k CPU experts, most
  columns first, that minimise v4's cycle, under today's cap. One-row passes keep today's decision.
- **The replay.** window-passes-20261109's 28 window captures (planner v1, w2, w4, w7) on the faithful replay from the
  load's origin. v4 prices every pass of both arms.
- **On the control's own executed picks:** planner v1 +1.44% (captures +0.86 to +2.97%), w2 +2.17%, w4 +1.81%, w7 +2.64%.
  - Each of the candidate's choices is v4's best for its pass. On the control's passes the same-pass best split agrees
    with the recorded division bound, and recorded and replayed passes match within 1-2% at every width.
  - Yet the candidate ends with 3-5% more misses (CPU jobs plus uploads) at every width. Experts served from a resident
    frame fall from 4.26 to 4.07 a pass at 8 rows and from 2.73 to 2.60 at 4 rows. Admissions are unchanged (602 issued,
    601 published), and reuse from staged frames is 0.01-0.03 a pass in windows.
  - That is the [executed-picks bias](#window-optimizer). The routing bonus steered the recorded picks toward the
    control's residents, so any candidate whose residents drift meets demand shaped for the control's cache.
  - A flat credit per upload only adds uploads: +1.3% at 0.1 ms, and +9 to +15% from 0.3 ms up.
- **On the router's original picks** (no bonus shaping, so demand does not depend on either arm's cache): planner v1
  -0.32% (captures -0.81 to -0.12%), w2 -0.55%, w4 -0.43%, w7 -0.06%. A credit per upload does not help. This regime has
  no routing bonus, and its control costs 70% more, so it bounds the planner's own effect, not the serving gain.
- **Per-pass balance, from v4's predicted ends** (widths 2-8, mean over layers of each layer's mean abs(Δ), router's
  picks): CPU against the 5090, 0.663 ms under today's planner and 0.700 under the candidate (planner v1); slowest minus
  second, 0.645 and 0.677. The candidate leaves the CPU lane last on every pass: v4 lets the 5090 arrive up to the resume
  time after the lanes for free, so its cheapest split rarely equalizes the two ends.
- **Decision under the rule** (2% or simpler): not built. It gains under 1% where it can be judged fairly, and it replaces
  one number and an additive table with about twenty measured terms, which is not simpler. The CPU price stays in windows.
- **What the bound found about the method.** The replay on the control's executed picks cannot rank a policy that changes
  residency in the bonus regime. Every such candidate loses resident hits to the bias, by more than the effects in
  question. Ranking a planner or residency policy fairly needs demand re-routed under the bonus from the router's scores,
  which detailed captures record (`events_from_detailed`, `replay_work(routing_bonus=...)`). The `T=1` planner rebuild's
  bound and step 6's offline optimization both need that first.

<a id="helper-merge"></a>

**The 3090's result merge, 2026-11-09 (5b2e439): branch-free and bit-identical. The host's serial time between its last
lane and the publication falls by about 60%**
([split](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/publication-parts.txt),
[confirmation](/mnt/wsl/transfer/minimax-wsl/runs/helper-merge-20261109/compare-merge.txt)).
- **What the publication waits on.** The untraced light records split the host's work between its last lane and the
  publication the 5090 spins on. On passes with CPU and 3090 jobs, medians:
  - **Collecting and merging the 3090's result:** 0.009 ms at 1 row, rising about 0.012 ms a row to 0.093 ms at 8.
  - **The publication write and the `add_kernel` launch:** 0.024-0.032 ms, flat.

  The 3090's result was ready before the CPU lane ended on 90-99.7% of passes. So on passes where the CPU lane ends last,
  the merge is on the pass's critical path.
- **The cause was the merge loop, not the memory.** The result buffer is ordinary pinned memory (`cudaMallocHost`).
  `NativeHelper::collect` checked every element and returned at the first non-finite one, so the loop did not vectorize:
  about 2 ns an element, 3,072 elements a column.
- **The change** (`src/mm3/mm3_helper_merge.hpp`):
  - The loop has no branch; the two checks reduce to flags the caller reads.
  - Each element is the same multiply then add as before, so the output is bit-identical. `mm3_helper_merge_test` checks
    400 random columns, with subnormal, huge and signed-zero values, and both flags.
  - 16 columns of 3,072 take 3.06 µs, against 35.75 µs.
  - The 3090's merged rows also keep their buffer, instead of a fresh zero-filled vector each pass.
  - `mm3_native_helper_test` passes on the 3090.
- **On hardware**, five window-passes requests were repeated with the new binary (same arms, prompts, seed and tune). Median
  ms from the later of the CPU lane's end and the 3090's readiness, to the merged rows:

  | Rows | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
  |---|---:|---:|---:|---:|---:|---:|---:|---:|
  | Old | 0.009 | 0.020 | 0.032 | 0.046 | 0.055 | 0.064 | 0.067 | 0.096 |
  | New | 0.005 | 0.009 | 0.015 | 0.019 | 0.025 | 0.026 | 0.029 | 0.039 |

  - The publication step after it is unchanged.
  - The new binary's load ran beside one single-process replay on the host (one core busy, a sub-agent's). That can only
    slow the new side, so the drop is conservative.
  - Text differs between runs by design, so cycles by rows move both ways and cannot resolve the gain. It is about 1% of an 8-row
    window, where the CPU lane is last on 53-72% of passes, and less at narrower widths.
- **Adopted under the rules:** no quality mechanism, it simplifies the merge, and it trims a serial cost that grows with
  window width. Moving the merge to the device (task 54) would take the rest.

<a id="verify-by-prompt"></a>

**Verify cost by prompt, 2026-11-09: a window of the same width costs prose prompts 0.84-0.96 times the all-prompt mean, and
code and reasoning 1.05-1.21 times; planner v1's single table is 2-13% optimistic at 4-7 rows**
([costs](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/verify-by-prompt.txt),
[bound](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/regime-bound.txt)).
- From window-passes-20261109's generation forwards from token 128, over all arms that ran each prompt:
  - prose, val-prose and conditioning-prose: 0.84-0.96;
  - python-code, conditioning-code and val-reason: 1.05-1.21;
  - val-code and val-extract: about 1.0.

  The factor is mostly common to all widths. The mean cost is 155.5, 180.3 and 183.9 ms at 6, 7 and 8 rows, against the
  table's 143.0, 160.0 and 177.0.
- **Bound.** For constant per-position acceptance from 0.3 to 0.97, the planner's rule was priced with the all-prompt costs
  and each prompt's own.
  - The table's width costs nothing at most acceptance levels.
  - Where a prompt's own costs would flip the best width, the table's choice costs 1-8% of that round's throughput, in
    narrow bands of acceptance. Those flips rest on by-rows ratios that mix arms with different residency.
- **Reading.** A per-request cost scale is cheap. Its expected gain over a round mix is small and below what the round-level
  replay can rank, so it waits behind the levers above. A refreshed verify table on today's base is the cheaper step.

<a id="early-copy-windows"></a>

**Early copies for verify windows, bounded 2026-11-09: worth building only at 2 rows, about 0.2-0.4% overall, so not built**
([bound](/mnt/wsl/transfer/minimax-wsl/runs/window-trace-20261109/early_copy_bound.py),
[result](/mnt/wsl/transfer/minimax-wsl/runs/window-trace-20261109/early-copy-bound.txt)).
- **Why look.** `early_copy` runs only at `T=1`. In windows of 1-3 rows the 5090's path is last on 54-82% of passes, set by
  the pass's demand upload.
- **The bound, on the traced passes** (the per-pass lane ends from [window-paths](#window-paths)):
  - **Gain:** a pass with one demand upload, after a pass with at most one (mode 2's condition), ends at the latest of the
    CPU, the 3090 and the 5090's resident path (v4) instead.
  - **Cost:** the copy shares host memory with the previous pass's CPU lane. When that lane was the previous pass's last,
    its end moves by up to the 0.305 ms stretch.
- **Net, as a share of the cycle at a hit rate of 0.5 / 0.7 / 0.9:**
  - 2 rows: +0.6 / +2.7 / +4.8%;
  - 3 rows: -2.5 / -1.7 / -1.0%;
  - 4-8 rows: -1.9 to -3.2% at any hit rate.
- **In windows the copies and the CPU lane draw on the same host memory, and are already balanced.** Moving a copy earlier
  moves the bottleneck to the CPU lane on the pass before. Two-row windows carry about 8.5% of the planner's cycle, so the
  realistic gain is 0.2-0.4%.

<a id="replay-fidelity"></a>

**The production-policy replay on today's base, 2026-11-09 (89e0bdc): faithful pass by pass, in sample and out**
([fidelity](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/replay-fidelity.txt),
[held out](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/replay-heldout.txt)).
- **Before:** the control replay (`tools/mm3/shared_model/work.py`) priced recorded demand +55% at `T=1`, +14% for planner
  v1 and +4-5% at fixed widths. 17-80% of passes had the same jobs.
- **What was missing.** A sub-agent found and added these from this session's brief; they were reviewed and rerun here.
  1. **Early copies (mode 2).** A frame is owned at once and kept from victims, free counts, exchange landings and the
     3090 lease's checks until a later join sees the copy done. The prediction is rebuilt from each capture, which records
     neither it nor the copy.
  2. **Demand copies take frames in pick order**, as the engine's `locate` does.
  3. **A millisecond clock** (`todays_rule`): cycle and prefix by rows, the drafting gap and transfer lengths, measured on
     this load. Counting layer joins could not fit `T=1` and windows at once.
  4. **Each request's resident order is rebuilt from the earlier requests.** It decides ties between equal victim scores.
  5. **`compare_work` compared picks with jobs in windows.** It now compares jobs with jobs, by tenth of the request.
- **After, passes with the same jobs and priced replay against recorded work** (median, with v4):

  | Arm | Passes, before → after | Priced, before → after |
  |---|---|---|
  | `T=1` | 17% → 99.7% | +55.3% → +0.17% |
  | Planner v1 | 57% → 97.8% (median 99.3%) | +13.8% → +0.11% |
  | w2 | 80% → 99.6% | +4.1% → +0.03% |
  | w4 | 68% → 100% | +5.1% → -0.04% |
  | w7 | 57% → 100% | +4.9% → -0.03% |

- **Out of sample:** helper-merge-20261109, another load, from its own origin, clock not refitted: 97.9-100% of passes,
  priced -0.04% to +0.56%.
- **What a candidate inherits from the control:** the early copy's predicted experts, the clock, and the tie order. So an
  optimizer on this replay is a screen, not a selection. Recording the early copy (layer, expert, slot) in the light
  recorder would remove the first.

<a id="window-optimizer"></a>

**The window optimizer on the faithful replay, 2026-11-09: the control comes out best on every axis, but the screen is
biased toward the control by inputs taken from its own run. It cannot rank policies under the routing bonus**
([screen](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/window-optimizer.txt),
[bias test](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/optimizer-bias.txt)).
- **The screen.** Planner v1's 16 captures and the 4 `T=1` captures were replayed as today's base runs, and v4 priced every
  pass. A coordinate sweep covered the CPU price (1.0-4.0), upload cap (2-4), refill budget (4-16) and horizon (20-80),
  then the CPU price in steps of 0.1.
- **The result.** Every step from the control costs drafting time:
  - CPU price 1.4 +1.6%, 1.6 +1.1%;
  - refill budget 4 +8.2% (`T=1` +15.5%), 12 +4.7%, 16 +7.6%;
  - upload cap 3 +1.1%;
  - horizon 20 +1.1%, 80 +0.8%.

  That is too sharp to trust: on 2026-11-08 the `T=1` screen found the CPU price flat from 1.3 to 2.1.
- **The bias test:** six configurations on the eight seed-6001 planner captures, each under three input modes, against
  the control in the same mode:

  | Configuration | The control's own inputs | Early copies predicted from executed picks | Router's original picks |
  |---|---:|---:|---:|
  | CPU price 1.3 | +2.33% | +1.27% | -1.29% |
  | CPU price 2.0 | +2.55% | +1.65% | +1.34% |
  | Refill budget 4 | +8.58% | +7.05% | +1.19% |
  | Refill budget 12 | +4.71% | +3.38% | -0.72% |
  | Upload cap 3 | +1.17% | +0.65% | +1.31% |

- **Reading.**
  - About a third of the penalties come from the early copies' predictions, which are rebuilt from the copies the control
    made.
  - Most of the rest come from the executed picks. With the routing bonus at 0.20 the engine substituted experts toward
    the control's residents, so a candidate with other residents meets demand shaped for the control's.
  - Serving the router's original picks flattens the landscape and moves the optimum (CPU price 1.3 -1.3%, refill budget
    12 -0.7%). But that is another regime: no bonus, and the control costs 70% more.
- **So "the control is best" is not a finding.** An unbiased screen re-routes each candidate under the bonus against its
  own residents. That needs the router's base scores per row, which detailed captures record and light captures do not,
  and an early-copy prediction that does not come from the control's copies.
- **The screen was out of order.** The plan replaces the CPU price with a planner that prices each pass by completion time
  from the measured GB/s ([planner rebuild](README.md#host-memory-and-l3), item 6 of the order up to step 6; user
  direction, 2026-11-09). It defers CPU-price, horizon and refill-budget tuning until then. This screen tuned a parameter the
  rebuild removes. What carries over is the faithful replay, which is how the rebuilt planner gets evaluated offline, and
  the bias finding about the executed picks.

<a id="expert-graph-windows"></a>

**The expert-phase graph in verify windows, 2026-11-09 (5e35257): exact, and it starts the CPU lane 0.05-0.39 ms sooner**
([mechanism](/mnt/wsl/transfer/minimax-wsl/runs/expert-graph-windows-20261109/mechanism.txt),
[bound](/mnt/wsl/transfer/minimax-wsl/runs/window-passes-20261109/early-lane-bound.txt)).
- **Why look again.** In windows the CPU lane starts once the host has submitted the 5090's work, and that submission grows
  with rows: 0.07 ms after the plan at 1 row, 0.50 at 8. The CPU lane is last on 25-72% of window passes.
  - Window model v4 priced a CPU lane started that much sooner at -2.5 to -4.1% of the cycle at 2-8 rows, and -2.6% of
    planner v1's pass time.
  - At `T=1` the graph had freed about 60 µs of host time without moving the cycle, because the 5090's device work binds
    there ([expert-graph](#expert-graph)).
- **Exact in windows:** `expert_graph=2` checked 11,970 and 11,457 replays against the launches under planner v1, with 0
  mismatches and 0 fallbacks. In timed mode the graph replayed 5.5-13.8 thousand times a request from 10-12 captures,
  with no fallback.
- **The mechanism.** One load, control and graph interleaved on all eight prompts. Medians in ms after the pass's plan,
  and the mean cycle:

  | Rows | Submission end | CPU end (passes with CPU jobs) | Mean cycle |
  |---:|---|---|---:|
  | 1 | 0.071 → 0.019 | 1.095 → 1.060 | -3.7% |
  | 2 | 0.129 → 0.037 | 1.020 → 0.952 | -4.5% |
  | 3 | 0.187 → 0.048 | 1.110 → 0.981 | -4.8% |
  | 4 | 0.245 → 0.061 | 1.225 → 1.046 | -5.1% |
  | 5 | 0.292 → 0.077 | 1.483 → 1.794 | +3.6% |
  | 6 | 0.367 → 0.088 | 2.018 → 1.829 | -6.9% |
  | 7 | 0.438 → 0.109 | 2.322 → 2.026 | -3.7% |
  | 8 | 0.496 → 0.105 | 2.973 → 2.666 | -9.0% |

  - Weighted by the control's width mix, pass time falls 5.2%, more than v4's -2.6%.
  - The 5-row bin holds the fewest passes (1,120 against 1,456).
  - Per-prompt means are not comparable in one block: text, and with it the width mix, differs between runs by design.
- **Timed, and adopted for both bases**
  ([protocol](/mnt/wsl/transfer/minimax-wsl/runs/expert-graph-timed-20261109/make_protocol.py),
  [evaluation](/mnt/wsl/transfer/minimax-wsl/runs/expert-graph-timed-20261109/evaluation.txt)). One load: `T=1` and
  planner v1, each with and without the graph, five seeded blocks on eight prompts at up to 384 tokens, from token 128.
  All 160 requests completed: 1,239,750 replays, 0 fallbacks.

  | Graph over control | Equal-weight, 95% interval | Frozen |
  |---|---|---:|
  | **Planner v1** | **-4.58% [-7.98, -1.18]** | -4.4% (inside) |
  | **`T=1`** | **-1.30% [-3.30, +0.69]** | -1.4% (inside) |
  | Planner v1 over `T=1`, both without | -9.65% [-11.56, -7.74] | |

  - Planner v1, by prompt: python-code -6.1%, prose -8.6%, val-code -1.4%, val-prose +2.0%, val-extract -7.2%,
    val-reason -2.3%, conditioning-code -9.0%, conditioning-prose -4.0%.
  - `T=1`, by prompt: from -4.6% (conditioning-code) to +3.3% (val-prose).
  - **Under the rule** (adopt where the mean effect is below zero) `expert_graph=1` joins the drafting base and the `T=1`
    base. Drafting with the graph runs about 13.8% faster than `T=1` without it.
  - **The window model held:** the mechanism block's -5.2% of pass time, taken at 85% of a round, predicted the drafting
    effect within 0.2 points.

<a id="topk-split"></a>

**`topk_rows` split over many blocks a row, 2026-11-09 (5e35257): the same candidates bit for bit, 2.5-3.3 times faster**
([test](../../tests/mm3_topk_split_test.cu)).
- **Why.** It ran one block a row, so its 1-8 rows used 1-8 of the 5090's 170 SMs:
  - the GPU nucleus's top-512, at the end of every `T=1` forward and verify window, on the critical path;
  - DFlash2's top-16 over the proposal's 7 columns.
- **The split.**
  - **Stage 1:** G blocks a row (the SMs over the rows, at most 32), each takes the top-kc with ties of an id-contiguous
    slice. Every element of the row's top-kc with ties is in its slice's: fewer than kc elements of the row exceed it,
    so fewer than kc of its slice do.
  - **Stage 2:** a block a row compacts the slices in slice order and selects as before. Each slice is sorted by key
    descending then id ascending, and the slices ascend in id, so equal keys arrive in id order and the output is the
    same bits.
  - **Overflow:** a slice with more ties than the 512 candidate slots sends its row back to the whole-row pass.
- **Exact:** 2,856 rows with 0 differences.
  - Rows: normal, Zipf-shaped, rounded to a few levels, constant, with masked runs and a NaN.
  - Sizes: vocabularies of 1 to 200,064, 1-16 rows, 16 and 512 candidates, with extra ids.
  - 485 of them overflow the candidate slots.
- **On the 5090:**

  | Case | Block a row | Split |
  |---|---:|---:|
  | 1 row, 512 candidates (`T=1` nucleus) | 0.227 ms | 0.089 ms |
  | 7 rows, 16 candidates (the drafter) | 0.236 ms | 0.072 ms |
  | 8 rows, 512 candidates (the nucleus in a window) | 0.239 ms | 0.079 ms |

  That is about 0.14 ms off every `T=1` token (0.3%), and about 0.16 ms off every proposal and every verify window.
- **Wiring:** the nucleus (`Dev::nuc_topk`) and DFlash2's proposal (`cand_scratch`) pass a scratch buffer. Other callers
  keep the block-a-row form.
- **Inside the engine** ([check](/mnt/wsl/transfer/minimax-wsl/runs/topk-split-20261109/check.py)): `gpu_nucleus=2` verifies
  every row the GPU nucleus settles against `dist()` over the whole row.
  - `T=1` prose, planner v1 python-code and w7 prose: 383 rows verified each, 0 mismatches.
  - Drafting ran normally on the split top-16: 235 of 296 and 234 of 1,035 drafts accepted.
  - These requests ran beside a 28-process replay sweep on the host (a sub-agent's, which the pause request reached only
    after its turn). The checks do not depend on timing, but the draft call's times from this load are not a timing
    comparison. Only the kernel's 0.16 ms is claimed.
- **Adopted under the rules:** no quality mechanism, and identical output.

<a id="speed-review-20261110"></a>

## Speed review from outside the plan, 2026-11-10

The user asked for a view from outside the plan of what limits speed, then reviewed it in four rounds. This section holds
its measurements; the [plan](README.md#order-20261110) owns the order that came from it.
- **Method.** Read-only analysis of existing light captures and of the source. No engine code, setting or GPU run.
  Scripts and their outputs: `/mnt/wsl/transfer/minimax-wsl/runs/speed-review-20261110/` (run from the worktree root with
  `python3 -I -B`). [Page](https://claude.ai/artifact/Uv78kF9jUBuzhofHGJ9Zjj).
- **Captures.** One-row generation from token 128: `expert-graph-t1-20261109` (arms `m1-on`, `m2-on`; prose and
  python-code) and `perpass-new-base-20261109` (code, prose, reasoning).
- **How sure each part is.** Read from captures: pass types, loads, demand spread, resident changes, exchange costs, idle
  time, prompt speeds. Read from source: the 16-row limit, the early copy's rule, the decay and the caps. Simulated on the
  router's own picks, without re-routing under the bonus: the cache and rotation tables. Inferred: that the early copy
  holds passes at 0.56 ms, since no capture records when a copy lands. Arithmetic only: every gain quoted in the plan.
- **Withdrawn during the review.**
  - "Residents are well chosen; a hindsight swap removes under 1% of loads." The loop in `reloads.py` stops at the first
    candidate with few loads, so it tested almost nothing.
  - "A 3090 exchange does not compete with the 5090's uploads." It does; the user corrected it.
  - A first figure of 1.2 ms an exchange divided by exchanges that fell before token 128. It is 1.5 ms.

<a id="review-pass-types"></a>

**What sets the length of a one-row pass** (`pass_classes.py`, expert-phase graph on).

| Pass | Prose: passes | Prose: time | Code: passes | Code: time | Median |
|---|---:|---:|---:|---:|---:|
| Every expert already on a GPU | 22-25% | 15-18% | 11% | 7% | 0.43-0.46 ms |
| Uses an early copy | 61% | 53-54% | 60-61% | 45-47% | 0.56-0.57 ms |
| Has a demand miss (upload or CPU job) | 15-17% | 28-32% | 28-29% | 46-48% | 1.17 ms |
| All passes, mean | 0.63-0.66 ms | | 0.75 ms | | |

- One upload of a 31.85 MB expert takes 0.554 ms at the measured 57.5 GB/s. With one early copy per layer on one link,
  passes cannot average less than one upload.
- Switching the expert-phase graph on took all-resident passes from 0.52-0.55 to 0.43-0.46 ms. Passes using an early copy
  stayed at 0.56-0.57 ms and the mean hardly moved. `helper_zero_copy` shows the same pattern.
- **The upload budget.** Experts loaded from host memory: 50-53 a token on prose and 67-68 on code (0.9 and 1.2 a pass;
  47-68 across all captures of 2026-11-09). That count times 0.554 ms sets a floor of 0.49-0.52 and 0.65-0.66 ms on the
  mean pass, against 0.63-0.66 and 0.75 ms measured: the engine runs within 15% (code) to 30% (prose) of it.
- PCIe 5.0 x16 carries at most 63 GB/s before protocol overhead, so the link is at its limit.
- The CPU lane has 0.02-0.06 jobs a one-row pass.

<a id="review-demand"></a>

**Demand against the cache** (`cache_sim.py`, `reloads.py`).
- About 800 to 900 generated tokens touch 5,814 (prose) to 6,628 (code) of the 7,296 experts. Consecutive tokens share
  23-40% of their picks.
- Residents serve 50-65% of the router's own picks, and the bonus moves another 15-27% onto residents. The best fixed set
  of the same size, chosen in hindsight, would serve 57-71%, and perfect foresight 78-86%.
- 35% of loads fetch an expert that was loaded within the previous 8 tokens, 59% within 32 and 80% within 128.
- Loads a token for a cache with perfect foresight on the router's own picks (a simulation):

| Slots | 741 | 1,175 (today) | 1,500 | 2,000 | 3,000 | 4,000 |
|---|---:|---:|---:|---:|---:|---:|
| Code | 69 | 48 | 37 | 26 | 13 | 6 |
| Prose | 48 | 31 | 23 | 15 | 6 | 3 |
| Reasoning | 70 | 49 | 39 | 27 | 13 | 7 |

  On the picks the engine executed, foresight at 1,175 slots would load 30, 20 and 30 a token.

<a id="review-rotation"></a>

**How fast the resident sets rotate** (`rotation_rates.py`, `rotation_policies.py`, `free_rotation.py`).
- **The score.** *Corrected 2026-11-10 by the [re-routed replay](#reroute-replay-first):* in the native path the score
  decays by 0.95 per five generated rows (`native_demand_decay`, `src/mm3/mm3_native_demand_units.hpp:8`), about 0.99 a
  token, with a credit of 1.02 a pick. `kDecay` per forward (`src/mm3/mm3_model.cu:591`) is the legacy path. The review
  read it as 0.95 a forward, so the simulations below ran with a memory about five times too short. Their results on
  the caps are confirmed by the replay. Their results on the decay and on two speeds are withdrawn.
- **The caps.** Exchanges are paid from a credit that grows by capacity / 5 for each generated row
  (`src/mm3/mm3_native_demand_units.hpp:20`). The 5090's capacity is 8 (`max_exchanges`): 1.6 exchanges a token at one
  row. The 3090's is 1 (`src/mm3/mm3_native_helper_live.inc:153`): one exchange every five tokens, one in flight at a
  time. Recorded: 1.60 and 0.20 a token.
- **Recorded turnover.** Over 930 to 1,020 tokens the 5090 keeps 56 to 107 of its 438 starting residents, and the 3090
  keeps 543 to 561 of its 741.
- **Simulated caps.** The router's own picks whose expert is not resident, a token, with the engine's score and its 3090
  rule (incoming score at least 1.8 and above 1.25 times the victim's plus 1):

| Exchanges allowed a token | Code | Prose | Reasoning |
|---|---:|---:|---:|
| Engine as recorded | 110.1 | 78.9 | 114.9 |
| Simulated at the engine's caps (the baseline below) | 112.9 | 80.8 | 118.0 |
| 3090 at 0.5 | -4.2% | -6.9% | -5.2% |
| 3090 at 1 | -7.7% | -12.4% | -9.8% |
| 3090 at 2 | -9.7% | -15.3% | -13.3% |
| 3090 at 4 | -11.1% | -17.0% | -15.6% |
| 5090 at 4, 3090 unchanged | -3.6% | -5.7% | -4.8% |
| 5090 at 8, 3090 unchanged | -5.6% | -8.3% | -5.8% |
| 5090 at 4 and 3090 at 2 | -12.3% | -17.9% | -16.0% |
| 3090 at 1 with decay 0.90 / 0.98 | -6.8% / -8.4% | -11.1% / -13.1% | -8.3% / -11.2% |

- **The decay with both cards free to exchange without a cap** (the user asked whether a faster decay could beat
  foresight). *Withdrawn as a test of the engine's decay: the row marked today's is 0.95 a token, and the engine runs at
  about 0.99 a token. The table stands only as a simulation of those values.*

| Policy, both cards free | Code | Prose | Reasoning |
|---|---:|---:|---:|
| Decay 0.50 | 95.0 | 63.1 | 93.5 |
| Decay 0.80 | 94.0 | 61.8 | 92.4 |
| Decay 0.90 | 91.4 | 60.0 | 90.5 |
| Decay 0.95 (today's) | 88.1 | 57.7 | 88.6 |
| Decay 0.98 | 90.1 | 59.6 | 89.8 |
| Decay 0.99 | 93.3 | 62.6 | 92.6 |
| Two speeds: slow 0.99 plus fast 0.7 at half weight | 85.4 | 55.5 | 86.0 |
| Perfect foresight | 59.3 | 38.1 | 61.3 |

- No policy that only chooses residents can beat foresight, whatever its decay. 0.95 is the best single value even when
  the cache is free, and two speeds beat any single decay by 3 to 4%.
- Foresight is not this engine's floor: it holds the picks fixed, and the bonus lets a resident stand in for a pick that
  is not cached.
- In the recorded runs 52 to 59% of the router's picks that miss the cache are loaded and the rest are replaced by the
  bonus. Whether uploads fall in proportion under another policy needs the re-routed replay.

<a id="review-exchange-cost"></a>

**What an exchange costs, and why it is a scheduling matter** (`exchange_cost.py`, `exchange_schedule.py`; seven
captures, 3,892 tokens; each pass is compared with quiet passes of the same type at the same layer).
- **What a 3090 exchange does** (`src/mm3/mm3_native_helper.cu:208-290`). The incoming expert is copied host-to-host
  into a pinned staging buffer while the victim comes down over the x4 link; the incoming goes up over the x4 link; the
  victim is copied host-to-host into the bank frame the incoming left. That is 191 MB of host memory traffic, where a
  5090 upload reads 32 MB. It lasts 15.7 ms (median): prepare 6.0, the x4 upload 5.6, the copy back 3.9.
- **How much longer passes run beside it:**

| Pass | 3090 has work in it | During prepare | During the x4 upload | During the copy back |
|---|---|---:|---:|---:|
| Uses an early copy | No | +5% | +5% | +22% |
| Has a demand miss | No | too few | +6% | +19% |
| Every expert on a GPU | Yes | +19% | +20% | +22% |
| Uses an early copy | Yes | +7% | +14% | +22% |
| Has a demand miss | Yes | +7% | +6% | +21% |

- **The bill.** One 3090 exchange costs 1.5 ms of extra pass time: 0.84 ms in the copy back, 0.57 ms in the x4 upload
  and 0.12 ms in prepare, whose host copy mostly falls in the token's dense prefix. At 0.20 exchanges a token that is
  0.31 ms a token, 0.8% of MoE time.
- **Beside which exchange:**

| The pass runs beside | Passes | Longer by | Extra ms a pass |
|---|---:|---:|---:|
| A 5090 promotion at a layer join | 3,737 | +18% | 0.16 |
| A 3090 exchange, prepare | 729 | +9% | 0.07 |
| A 3090 exchange, x4 upload | 5,383 | +11% | 0.07 |
| A 3090 exchange, copy back | 4,159 | +20% | 0.14 |
| Exchanges on both cards | 1,162 | +29% | 0.25 |
| of which the 3090's copy back with a 5090 promotion | 170 | +51% | 0.42 |

- **Coincidence.** 0.54% of passes run beside exchanges on both cards, against 0.12% if they were independent. Both are
  keyed to token ends and layer joins.
- **The 5090's own exchanges** run at 1.60 a token. The 58% that carry an upload are issued at the token's end and fall
  in the dense prefix. The rest are promotions at layer joins, in flight a median 2.0 ms.
- **The total:** 0.49 ms a token, 1.3% of MoE time, about 0.3 ms from the 3090's exchanges and 0.2 ms from the 5090's
  promotions.
- **Idle time a token:** the 3090 is idle 29.2 ms; passes with no 5090 upload in flight total 1.8 ms; the dense prefix
  and tail are 1.7 ms. One 3090 exchange needs about 11 ms of the x4 link and about 7 ms of host copying.

<a id="review-all-resident"></a>

**Inside a one-row pass with every expert on a GPU** (`all_resident_breakdown.py`, `gpu_split.py`; 9,752 passes,
expert-phase graph on; medians in ms after the layer's routing is known to the host).

| Experts on the 3090 | Share | Plan done | 3090's worker starts | 3090's results seen | Next preparation launched | Next routing known |
|---|---:|---:|---:|---:|---:|---:|
| 0 | 7% | 0.027 | none | none | 0.050 | 0.371 |
| 1 | 25% | 0.027 | 0.033 | 0.154 | 0.174 | 0.391 |
| 2 | 35% | 0.026 | 0.034 | 0.202 | 0.227 | 0.442 |
| 3 | 26% | 0.026 | 0.034 | 0.264 | 0.294 | 0.508 |
| 4 | 8% | 0.024 | 0.034 | 0.305 | 0.339 | 0.551 |

- GPU work in a pass is about 0.28 ms: the preparation 0.176, the 5090's own experts about 0.09, combine and add 0.012
  ([the 5090's chain trace](#five-ninety-chain-trace)).
- The 3090's results are seen 0.154 ms after the routing for one expert and about 0.05 ms later for each further one. Its
  computation is 0.022 ms plus 0.0485 ms an expert, so about 0.08 ms is the hand-off around its work.
- After the results a constant 0.21-0.22 ms runs until the next routing is known.
- A floor from these figures, if the 3090 started when the routing ended, the 5090's residents were dispatched on the
  device and the next preparation was already queued: 0.28, 0.28, 0.32, 0.37 and 0.41 ms. Weighted, 0.45 becomes 0.33.
- The 5090's residents were used a median 17 to 24 times in the second half of each capture, against 5 to 11 on the 3090.

<a id="handoff-design-bound"></a>

- **The next pass absorbs most of a shorter all-resident pass** (an Opus agent's design phase for item 5, 2026-11-10;
  `next_pass_absorption.py` and `next-pass-absorption.txt` in
  `/mnt/wsl/transfer/minimax-wsl/runs/host-free-handoff-20261110/`, checked before recording). Pairs of consecutive
  passes in one forward, the first with every expert on a GPU, medians in ms by the first pass's 3090 job count:

| Next pass | Share, prose / code | Sum of the two cycles, 0 to 4 jobs, code | Slope of the next cycle on the first, prose / code |
|---|---:|---:|---:|
| Every expert on a GPU | 31% / 21% | 0.77, 0.81, 0.87, 0.96, 1.05 | +0.38 / +0.34 |
| Uses an early copy | 60% / 66% | 1.10, 1.10, 1.10, 1.09, 1.10 | -0.31 / -0.62 |
| Has a demand miss | 10% / 14% | 1.70, 1.68, 1.70, 1.71, 1.69 | -0.15 / -0.76 |

  - On code the pair's sum is flat when the next pass waits for a copy: a shorter first pass only makes the next one
    wait longer. The early copy for the next layer is issued at the first pass's join and takes 0.554 ms. On prose the
    absorption is partial.
  - `expert_graph` agrees: it shortened these passes by 0.06 to 0.11 ms and moved the mean pass by 0.005 to 0.006 ms.
  - The design's per-pass prediction for the hand-off, the device dispatch and the queued preparation together: 0.299,
    0.279, 0.299, 0.347 and 0.396 ms for 0 to 4 experts on the 3090, against 0.371, 0.391, 0.442, 0.508 and 0.551
    today; weighted 0.455 to 0.317. Its model reproduces today's cycles within 0.004 ms.
  - With the absorption, the frozen token prediction is -2.0% [-1.2, -2.5] on prose and -0.7% [-0.35, -1.0] on code,
    about -1.4% equal-weight. The hand-off constant in it (0.025 ms) is assembled from the probe and not yet measured.

<a id="review-prompt-reading"></a>

**Prompt reading.**
- `expert-graph-timed-20261109` (160 requests): 41-45 tok/s on prompts of about 240 tokens, and 31.7 tok/s on the
  1,056-token python-code prompt, about 33 s. Its 384-token answer takes 14 to 17 s.
- The native cooperative path rejects chunks above 16 rows (`src/mm3/mm3_model.cu:1455`, `kWin = 16`), and the prompt
  loop (`src/mm3/mm3_main.cpp:1089-1127`) sends 16 rows through all 60 layers at a time: 66 trips for that prompt.
- All experts not on a GPU total 195 GB, which is 3.4 s at link speed.
- Upstream Maya reports 139 to 332 tok/s of prompt reading for this model on smaller machines, with smaller expert
  formats, so the comparison is indicative only.

<a id="review-drafting-infernix"></a>

**Drafting, and what the Infernix fork already does.** Two search agents read the fork; the width rule, the look-ahead
figures, the documented speeds and Maya's proposal code were checked against the files. Nothing was run and Qwen was not
loaded.
- **Maya.** `plan_width` (`src/mm3/mm3_main.cpp:219`) already chooses the width per step from each draft's confidence,
  and zero drafts is one of its choices; the draft call is still paid then. An extra verified row costs 19 to 30 ms
  against about 42 ms for a one-row forward. Lenience is `spec_policy=lenience` (`src/mm3/mm3_main.cpp:1346`).
- **Infernix, DFlash2 against MTP on Qwen Flash** (`docs/maintainer/dflash.md`, `docs/qwen3_8-flash-next.md` at ref
  `upstream/main`; RTX 5090, 2026-10-20). With automatic draft length DFlash2 is 8 to 11% faster than MTP, 26% on sampled
  code: Dense8 213 / 204 / 179 / 161 tok/s against 198 / 162 / 164 / 145. Verifying all seven drafts was 18 to 24%
  slower than MTP: "a verified column costs about a third of a plain round in expert-cache misses".
- **Its proposal** (`src/models/qwen4_exp/execution/dflash2.cpp`): one CUDA graph per batch size; fused kernels (the
  norm with the convolution's prepare, one product for Q, K and V, gate and up with the activation); a head top-16 that
  writes no logits; context keys and values written after acceptance for committed columns only. Proposal graph 0.68 ms,
  context 0.08 ms, host 1.2 ms, against 19.8 ms of target work a round. Its drafter is 776 MB and Maya's is 2.3 GB.
- **No overlap with verification.** Every proposal kernel depends on the anchor token. A pre-drafting experiment was
  removed as within chance. There is no early stop inside a verify.
- **Its width rule** (`src/runtime/speculative/draft_length.h`): the k drafts that maximize expected tokens over
  1 + 0.1 + 0.31 k plain rounds, acceptance by confidence decile learned online per request, one more draft probed every
  8 rounds.
- **Maya's port** (`src/mm3/mm3_dspark.cu:614-686`) computes all columns together, launches each projection separately,
  runs the target's full BF16 head over every mask column, writes the logits, takes a top-k and ends in a synchronize.
- **GLM layer look-ahead** (worktree `/mnt/wsl/transfer/infernix/wt/w10-gpu-prefetch`, never merged; results in
  `Infernix-glm5-local/docs/maintainer/glm5_3-flash-implementation-tracker.md`). A side stream applies the real norm,
  router and bias of the layer one or two ahead and takes the top 8. The first K predicted experts not on a GPU (2 by
  default) are copied; only completed copies are served and none is waited for.
  - Share of issued copies that were needed (4,110 rounds, K = 2): one layer ahead 52% before attention and 69% after
    it; two layers ahead 46% and 57%.
  - Speed: +1.9% at best, later left unqualified because a race could count late copies as ready, and -1.8 to -5.3% in
    the other configurations. The prediction's fixed cost was about 2.4% of a round.
  - For Qwen Flash-Next the draft's features predicted the next window's experts during the draft call, and that cost 1.5
    to 2.2% of wall time.

<a id="review-5090-memory"></a>

**What is on the 5090** ([ledger](native-memory-ledger-final.json), a planning estimate made before loading on
2026-11-01; resident counts from the captures of 2026-11-09; one expert is 31.85 MB).

| Item | GB | In experts |
|---|---:|---:|
| Resident experts, 438 | 13.95 | 438 |
| Dense layer weights, MXFP8, 60 layers | 11.07 | 348 |
| Drafter allowance (its weights are 2.3 to 2.4) | 3.22 | 101 |
| Output head, BF16 | 2.46 | 77 |
| Free reserve | 1.07 | 34 |
| KV cache, int8, 8,192 tokens | 0.52 | 16 |
| Staging frames, 8 | 0.25 | 8 |
| Planner window allowance | 0.21 | 7 |
| Indexer, logits, scratch | 0.16 | 5 |
| Not itemized | 1.28 | 40 |
| Card total | 34.19 | |

- The 3090 holds 741 residents, 23.6 GB of its 25.8 GB.
- "Not itemized" is 0.44 GB the card did not offer before loading, plus about 0.8 GB by which the measured 438 residents
  fall short of the ledger's planned 463.
- Around today's size the foresight curve falls by about 0.033 loads a token per slot on code.

<a id="reroute-replay-first"></a>

**The re-routed replay, first results, 2026-11-10** (an Opus agent under the orchestrating session; checked against the
source and its summary before recording). Tool: `tools/mm3/native_reroute_replay.py` with
`tools/mm3/test_native_reroute_replay.py` (commit 6b24b11). Run folder:
`/mnt/wsl/transfer/minimax-wsl/runs/reroute-replay-20261110/` (`grid-20261108-summary.txt`). Input: the four detailed
captures of 2026-11-08 (bonus 0.10, one row, before `early_copy`). Per token, from token 128, equal weight over the four.
- **Validation.**
  - The routing rule: against the recorded ready masks, all 87,324 rows re-route as the engine did, in rank order. The
    earlier replay's 96.6-97.6% came from its simulated residency, not from the rule.
  - The score: the native score matches all 5,502 scores the engine logged at its 2,751 exchange decisions within 5e-6.
  - The engine's residency simulated (the 5090 at 8 exchanges per five rows, the 3090 at 1 per five rows with its rule,
    8 staging frames through the production planner): 87,323 of 87,324 rows identical, and all 2,448 5090 and 303 3090
    exchange decisions identical and in order.
- **The bonus curve, with residency simulated** (against 0.20):

| Bonus | Picks needing a load, a token | Against 0.20 | Uploads against 0.20 | Moved off the top four | Router mass kept |
|---:|---:|---:|---:|---:|---:|
| 0.20 | 38.6 | | | 27.3% | 94.97% |
| 0.25 | 31.5 | -18% | -18% | 30.4% | 93.34% |
| 0.30 | 25.8 | -33% | -32% | 32.9% | 91.76% |
| 0.40 | 17.7 | -54% | -53% | 36.3% | 88.95% |

- **The 3090's cap, at bonus 0.20:**

| 3090 exchanges a token | Picks needing a load | Uploads | CPU jobs | Router's own picks off the GPUs | Moved off the top four |
|---|---:|---:|---:|---:|---:|
| 0.2 (today) | 38.6 | 21.7 | 16.9 | 101.4 | 27.3% |
| 0.5 | -6.8% | -2.3% | -13% | -4.4% | 26.5% |
| 1 | -14.7% | -6.7% | -25% | -9.3% | 25.7% |
| 2 | -21.4% | -9.6% | -36% | -13.8% | 24.8% |

  The router's own picks off the GPUs fall as the review's simulation said. Picks needing a load fall about 1.6 times
  faster, and fewer picks are moved. On this base the planner drops CPU jobs first, so uploads alone fall more slowly;
  today's base, where CPU jobs are rare, will say how that splits.
- **A two-speed score loses:** 1.1 to 2.3% more loads at 0.20 across six settings, and it stays worse than the native
  score with the 3090 at one and two exchanges a token.
- **A bonus-aware score does not gain:** 1.0% more loads at full weight, 0.1% at half weight.
- **The decay, screened at the engine's time constant** (`decay-20261108-summary.txt`; picks needing a load a token
  against today's value; from token 128, and in brackets from token 256):

| Decay per five rows | At today's caps | 3090 at one exchange a token, against 0.95 at that cap |
|---:|---:|---:|
| 0.975 | +0.4% (+1.2%) | +2.7% (+3.4%) |
| 0.95 (today) | 0 | 0 |
| 0.90 | +0.8% (+1.4%) | +0.2% (+0.8%) |
| 0.80 | +1.2% (+2.5%) | +1.4% (+2.8%) |
| 0.60 | +2.7% (+3.9%) | +3.8% (+5.5%) |
| 0.40 | +4.1% (+5.5%) | +5.2% (+7.5%) |

  Today's value is the best of the six at both caps. 0.80 and faster lose on all four requests. The deficits grow from
  token 256, so the scores inherited at the start of a request were not hiding a gain. At 0.975 the 5090's price binds
  (1.21 exchanges a token against 1.60), so part of its loss is the price and not the decay; the price is deferred with
  the planner. The thresholds keep their units at any decay, but what they ask in recent picks changes: the 3090's floor
  of 1.8 is about two recent picks at 0.95 and a single fresh pick at 0.90 or faster.

<a id="probe-captures-memory-20261110"></a>

**The probe for item 5, the detailed captures and the memory measurement, 2026-11-10** (an Opus agent; its write-up and
memory report were checked before recording). Run folder: `/mnt/wsl/transfer/minimax-wsl/runs/detailed-capture-20261110/`
(`result.md`, `probe/README.txt`, `memory-report.json`). Binary `maya-mm3-eg` at 5e35257. None of it is a timed test.
- **The probe: one pinned buffer shared by both cards under WSL2 works.** Every allocation tried gives a device view on
  both cards at the same address (`cudaHostAlloc` with or without Portable or WriteCombined, `cudaMallocHost`,
  `cudaHostRegister`). There is no peer access between the cards. A kernel on one card writes a sequence word and a
  24.6 KB payload; a kernel already running on the other polls the word and reads it, with no host call.

| Direction | Signal: median (p99, max) | Whole hand-off, median | Hand-off while the 5090 uploads at 50 GB/s |
|---|---:|---:|---:|
| 5090 to 3090 | 1.76 us (2.62, 3.41) | 11.2 us | 16.8 us |
| 3090 to 5090 | 1.31 us (1.86, 3.01) | 9.4 us | 17.9 us |
| GPU to a spinning host thread | 0.38 us (5090), 0.87 us (3090) | 3.8, 8.6 us | |

  - 10,000 trials each; no stale or torn read and no timeout in 348,000 trials over two versions of the probe.
  - A single block of scalar loads on the 3090 takes 37 us for the payload: read it wide, as the engine's copy kernels do.
  - The 3090's graph starts its next kernel 2.7 us after the wait kernel, the 5090's 1.0 us after.
  - The first version's one-way timings were wrong (the cards' clocks drift about 14 ppm apart); the second corrects it.
  - Not measured: the probe under engine load.
- **The detailed captures** keep every router score: 128 experts' score triplets in every routes record, all 57 MoE
  layers in every forward, and the residency, exchange and copy events; no dropped events.
  - One row (`detailed-t1-val-*`): code, prose and reasoning 512 tokens each, extraction 438 (it ended).
  - Drafting, DFlash2 at MXFP8 under planner v1 (`detailed-plan1-val-*`): 512 tokens each, with 157, 187, 135 and 116
    verify windows for code, prose, extraction and reasoning.
  - Under detailed capture `prep_graph` and `expert_graph` are off, and `early_copy` ran outside the graph.
- **Memory on the 5090 at load** (DFlash2 as the drafter):

| Item | GB |
|---|---:|
| Dense weights and head | 13.53 |
| Resident experts, 434 | 13.82 |
| Drafter | 2.48 |
| KV cache | 0.64 |
| Staging, 8 frames | 0.25 |
| Free when ready | 0.97 (989 MiB) |
| Not itemized | 2.35 |

  - The 2.35 GB not itemized: 0.44 reserved by the driver, 0.77 between CUDA's count and `nvidia-smi`'s (also there with
    a bare context), 0.52 context, 0.07 cuBLAS, 0.55 unknown.
  - Serving grows the 5090's use by 124 MiB (graphs). Its free low point was about 865 MiB, from one plain request an arm.
  - The 3090 holds 741 residents (23.60 GB) with 788 MiB free when ready.
  - The ledger's 463 against 434 residents, in experts: the count gap -24.2, in-process use -35.7, the drafter under its
    3 GiB allowance +23.5, the window allowance +6.7. With DSpark as the drafter the 5090 holds 438, the review's figure.
  - The drafter is sized before the residents, so its allowance and the "0.8 GB between planned and actual" are not
    levers.
- **What the replay does not model:** time; the cost of an exchange; early copies (these captures have none, and their
  prediction is not recorded); prefill. The token sequence is fixed. Two exchanges a token assumes leases can start
  inside a forward.

<a id="exchange-schedule-result-20261110"></a>

**Exchange scheduling, built and timed, 2026-11-10** (an Opus agent under the orchestrating session; its timed table,
pass comparison and promotion-cost files were checked before recording). Run folder:
`/mnt/wsl/transfer/minimax-wsl/runs/exchange-schedule-20261110/` (`result.md`, `design.md`, `analysis/`,
`mechanism-v2/`). Engine commits 1b3ba4c and 8f2c992: per-request `exchange_schedule` (0 today, 1 scheduled, 2 scheduled
with every copy verified), `residency_decay` and `helper_max_exchanges`, with defaults unchanged. The real-CUDA live
test is bit-exact in modes 0 to 2, and mode 2 checked 2,133 copies with no mismatch.
- **What was built.** The 3090's x4 copies go in pieces while it has no job in flight. The 5090's victim downloads run
  at the forward end or in passes with no upload. The host copies run in a window that opens at the forward end and
  closes 1.2 ms into the dense prefix; a host copy that has waited two forward ends runs anyway. The early copy's guard
  counts only exchanges whose upload may still be on the copy stream (`design.md`, "The early copy's guard").
- **A first version failed** and is not the one timed. It assumed 2 ms between forwards, from a capture without the GPU
  nucleus. Today that gap is 0.03 ms, so no 3090 lease finished.
- **Timed test:** one load, 12 blocks, arms alternated, two prompts, ms per token from token 128
  (`analysis/v2-timed-blocks.txt`). Frozen prediction: -0.60%.

| | Scheduled against today | 95% interval | Median of blocks |
|---|---:|---:|---:|
| Prose | -1.17% | -2.11 to -0.22 | -1.07% |
| Code | -3.21% | -5.96 to -0.45 | -1.25% |
| Equal weight | -2.19% | -3.91 to -0.46 | |

  - Prose is faster in ten of twelve blocks. Without block 12, whose scheduled answer stopped naturally at 337 tokens:
    -0.97% [-1.89, -0.04].
  - The code mean is mostly text. In five blocks an arm gave a low-demand answer that runs at 41 to 43 ms a token
    against 45 to 47 (579 to 584 router misses a token): the scheduled arm alone in four of them, both arms in one. In
    the seven blocks where both arms gave the usual answer the mean is -0.31%.
  - Reading: about 1% on prose, and between 0.3% and 1.25% on code. The prediction lies inside both intervals.
- **Mechanism** (`analysis/v2-pass_compare.txt`; all captures, and in brackets only the captures that did not overlap
  another agent's CPU job):
  - MoE time a token 38.74 to 37.32 ms (38.80 to 36.24). Passes with a demand miss 23.4% to 20.8% (23.5% to 18.6%). The
    cycle of each pass type and the 3090's GPU time per job did not change.
  - The dense prefix grew from 1.59 to 1.83 ms: the host copies and the 5090's refills now share it.
  - The two cards' exchanges coincide as often as chance: 43.3% of passes against 42.6%. Today: 0.64% against 0.20%.
  - Of 23 x4 pieces a token, 0.011 to 0.025 overlapped a 3090 job. 79 to 86% of the 5090's downloads ran at the forward
    end, the rest in idle passes. The host copies went in 3 pieces a token, and the two-forward bound fired on 0.38
    forward ends a token. How the host pieces split between windows and passes was not counted.
  - The review's per-exchange method does not work on the scheduled arm: a phase now lasts 35 to 90 ms, so 76% of
    passes count as beside one.
- **A 5090 promotion costs what it does through the early copy's guard** (`analysis/promo_cost.py`). Today, beside a
  promotion, 5.6% of passes use an early copy against 61.4% of quiet passes, and 58.0% have a demand miss against
  22.0%: +17.3% a pass, 0.093 ms a token, with 1.08 early copies blocked a token. Scheduled, the mix beside a promotion
  matches quiet passes: +0.013 ms a token, with 0.24 blocked a token. The review's +18% held the pass type fixed and so
  missed the cause.
- **Exchanges completed a token.** 5090: 1.594 in both arms. 3090: 0.195 today, 0.184 to 0.186 scheduled. A scheduled
  lease lasts 163 ms, 3.9 tokens, against 15.7 ms today, because the host copies wait for their window. With one lease
  in flight the scheduled 3090 can exchange at most 0.26 times a token.
- **Not verified:** the agent's estimate that the 5% fewer 3090 exchanges cost the scheduled arm at most 0.4% of a
  token.
- **An orchestration error, not a measurement error:** the campaign ended at 05:46 and the GPUs stayed locked and idle
  until 11:07, because no agent turn ran in between. Campaign scripts now release the lock themselves.

<a id="handoff-bench-20261110"></a>

**The hand-off bench and the re-bounds of item 5, 2026-11-10** (the design agent of item 5; `bench/host-free-bench.txt`,
`rebound-measured.txt` and `design.md` section 13 in `/mnt/wsl/transfer/minimax-wsl/runs/host-free-handoff-20261110/`;
the bench output was read before recording). A stand-alone bench on both cards, 1,000 iterations, 29 s. It is not the
engine.
- **Exact.** For 1 to 4 experts on the 3090 the queued graph matches today's path bit for bit: the merged F32 row, the
  3090's BF16 outputs and the BF16 input row. Zero-job tickets, flushes and expiries consume nothing twice and lose
  nothing.
- **The hand-off, from the routing's end to the 5090 seeing the 3090's rows, median us:**

| Experts on the 3090 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| Assumed in the design | 95.5 | 144 | 192.5 | 241 |
| Measured | 116.0 | 153.7 | 213.6 | 249.1 |
| The bench's stand-in for today's path | 285.1 | 338.0 | 405.2 | 445.9 |
| The stand-in while the 5090 uploads, mean | 363 | 462 | 471 | 484 |

  - The hand-off does not change while the 5090 uploads (means 118, 154, 215 and 249). The stand-in for today's path
    grows a tail there, with a 90th percentile up to 1.17 ms.
  - The measured hand-off is 8 to 21 us above the design's figure. The agent's explanation, a launch held without a
    stream query, was tested and is wrong: a re-run at 12:48 with a stream query after every arm leaves the 18.7 us
    outside the 3090's span unchanged (18.3 to 18.9 us). It stays unexplained, and it is already inside the measured
    figures the predictions use. The re-run is exact again and moves the re-bounds by under 0.01%
    (`bench/host-free-bench-flush.txt`, `compare-flush.txt`). The IF node measured 10.8 us skipped there.
  - The stand-in for today's path is about 0.065 ms slower than the engine, so the predictions keep the engine's
    baseline.
- **Conditional graph nodes work under WSL2 but are slow:** an IF node costs 9.0 us when skipped and 9.6 more when
  taken, against 4.5 us for six flag-gated kernels skipped. Step 3 of item 5 therefore uses flag-gated kernels.
- **Re-bound with the measured figures.** Cycles of a pass with every expert on a GPU, for 0 to 4 experts on the 3090:
  0.297, 0.277, 0.308, 0.368 and 0.404 ms, against 0.371, 0.391, 0.442, 0.508 and 0.551 today.
  - Steps 2 to 4: prose -1.9% [-1.2, -2.2], code -0.7% [-0.4, -1.0], about -1.3%. The first frozen prediction (-2.0%
    and -0.7%) is kept.
  - Step 5, the preparation's fusion at 15.5 to 18.5 us a layer: prose -0.51 to -0.61%, code -0.23 to -0.28%. If
    nothing were absorbed it would be -2.1 to -2.5%. `dense_split`'s timed code result, +0.65% [-0.85, +2.14], fits the
    absorbed figure and not the other.
  - Step 6, the 3090's time per expert: -0.31% on prose and -0.07% on code today. The plan's 3.6 to 4.1% assumed the
    3090 finishes last on most passes. It does so only in passes with every expert on a GPU.
- **A formula to read the gain off:** ms a token = 57 x a x (q x D + (1 - q) x e). Here a is the share of passes with
  every expert on a GPU, q the share of those followed by another such pass, D = 0.129 ms and e = 0.035 ms. Today a and
  q are 0.236 and 0.306 on prose and 0.112 and 0.205 on code. The gain passes 2% at about a >= 0.2 with q >= 0.45, at
  a >= 0.25 with q >= 0.3, or at a >= 0.3.

<a id="reroute-replay-today-20261110"></a>

**The re-routed replay on today's base, 2026-11-10** (the replay agent's final report; its summary files were checked
before recording). Run folder: `/mnt/wsl/transfer/minimax-wsl/runs/reroute-replay-20261110/` (`result.md`,
`grid-20261110-t1-summary.txt`, `grid-20261110-plan1-summary.txt`, `decay-20261110-t1-summary.txt`,
`windows-20261110-plan1.txt`). Tool at commit 6b41be0. Input: the eight detailed captures of 2026-11-10, four one-row
and four drafting, with `early_copy` and the graphs on. Per token from token 128, equal weight over four requests. No
timing is simulated: exchange costs are not charged and the token sequence is fixed.
- **Validation on the new captures.** Routing is exact on the recorded masks, and the score rule matches every logged
  score. The engine's rule simulated reproduces 99.991 to 100% of rows and every exchange decision, at one row and in
  windows. Every recorded early copy is replayed.
- **One row, by bonus:**

| Bonus | Picks needing a load | Uploads | CPU jobs | Early copies | Transfers | Moved off the top four | Router mass kept |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.20 | 15.14 | 12.87 | 2.27 | 50.27 | 63.14 | 22.9% | 95.99% |
| 0.25 | 11.28 | 9.92 | 1.35 | 49.48 | 59.40 (-5.9%) | 24.9% | 94.96% |
| 0.30 | 8.23 | 7.41 | 0.82 | 49.25 | 56.67 (-10.2%) | 26.3% | 94.06% |
| 0.40 | 4.71 | 4.39 | 0.32 | 48.90 | 53.29 (-15.6%) | 28.1% | 92.65% |

  - Of 228 picks a token at 0.20, 74.6 are on the 5090's residents, 93.2 on the 3090's, 45.1 in staged frames (44.7 of
    them early copies) and 15.1 need a load.
  - Against the base of 2026-11-08 at 0.20: picks needing a load fell from 38.6 to 15.1, uploads from 21.7 to 12.9 and
    CPU jobs from 16.9 to 2.3. The early copy took them: 50 a token, 89% of them read.
  - **The bonus now acts on the 15 picks that need a load, not on the early copies**, which stay at 49 to 50 a token
    at every bonus. Picks needing a load fall 25%, 46% and 69% at 0.25, 0.30 and 0.40. Transfers fall much less than
    on 2026-11-08, when the same steps gave -18%, -32% and -53%.
  - At 0.40 today the bonus moves 28.1% of picks. On the base where 0.20 passed its usability check, 0.20 moved 27.3%.
    Router mass kept is lower: 92.65% against 94.97%.
- **Drafting, by bonus:** picks needing a load 41.50, 33.45, 27.28 and 18.17 a token; transfers 31.24, 27.42, 24.55
  and 19.88 (-12%, -21%, -36%); moved picks 26.3%, 29.4%, 31.7% and 35.1%; mass kept 95.28%, 93.71%, 92.24% and
  89.60%. Early copies are 9 to 10 a token there.
- **The 3090's cap, at bonus 0.20:**

| 3090 exchanges a token | One row: picks needing a load | One row: transfers, best to worst | One row: picks on the 5090's and the 3090's residents | Drafting: picks needing a load | Drafting: exchanges delivered |
|---|---:|---:|---:|---:|---:|
| 0.2 (today) | 15.14 | 63.14 | 74.6 and 93.2 | 41.50 | 0.18 |
| 0.5 | -3.4% | -6.4% to -0.9% | 70.2 and 101.7 | -7.1% | 0.45 |
| 1 | -5.1% | -12.8% to -1.2% | 64.5 and 111.4 | -7.7% | 0.49 |
| 2 | -3.4% | -17.6% to -0.9% | 60.1 and 118.5 | -21.6% | 2.09 |

  - At one row the cap moves picks needing a load by 3 to 5% only. The spread in transfers is the early copies: under
    the new residents some recorded early copies would have found their pick already on a GPU, and the capture does
    not record what the engine would have copied instead. The worst case assumes it copies something else every time.
  - The cap moves 4 to 15 picks a token from the 5090's residents to the 3090's.
  - In drafting, caps of 0.5 and 1 deliver under 0.5 exchanges a token, because a lease starts only at a forward's
    end. The row for two a token assumes a lease can start inside a forward.
- **The two scores lose again:** two speeds, 16% more picks needing a load at one row and 3.4% more in drafting; the
  bonus-aware score, 7.4% and 2.0% more at full weight, 4.2% and 1.0% more at half weight.
- **The decay, one row, picks needing a load against 0.95 per five rows** (in brackets with the 3090 at one exchange a
  token): 0.975 +11.6% (+3.8%), 0.90 +12.0% (+2.2%), 0.80 +18.4% (+6.1%), 0.60 +27.4% (+12.3%), 0.40 +31.4% (+17.8%).
  From token 256 the deficits are +12 to +33%. Today's value is the best of six on this base too, by a wider margin
  than on 2026-11-08.
- **Rows of a verify window that share experts (item 6), drafting captures, bonus 0.20:**

| Window width | 2 | 3 | 4 | 5 | 6 | 7 | 8 | All |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Load picks within 0.20 of another row's pick | 4.5% | 8.3% | 11.5% | 15.9% | 17.0% | 19.3% | 26.5% | 13.2% |
| Loads removed by the greedy scheme | 2.0% | 3.6% | 5.2% | 7.7% | 8.2% | 8.9% | 13.1% | 6.0% |

  - The greedy scheme loads the most shared experts first and gives the bonus only to those. It removes 4,405 of
    73,814 loads and moves 5,534 picks; mass kept goes from 94.57% to 94.34%. At bonus 0.25, 0.30 and 0.40 it removes
    5.1%, 4.3% and 3.0%.
  - Routing again on the whole union, as the review wrote it, removes no load at any width and moves 27,019 picks.
  - The first run of this count hung for five hours on a loop in the greedy step. It is fixed and tested, and the
    count takes 3 s.
- **Passes with every pick on a resident, for item 5's formula.** Here a is their share of passes and q the share of
  them followed by another such pass (the agent gave the joint share; q is converted from it):

| | a | q |
|---|---:|---:|
| Today, bonus 0.20 | 18.0% | 27% |
| Bonus 0.25, 0.30, 0.40 | 19.2%, 19.9%, 21.1% | 28%, 28%, 29% |
| 3090 at 0.5, 1, 2 a token, bonus 0.20 | 22.7%, 27.8%, 31.7% | 32%, 36%, 40% |
| 3090 at 2 a token, bonus 0.40 | 36.4% | 43% |

  By the formula of the hand-off bench, item 5's steps 2 to 4 are worth about 1.5% of a token today, 1.8% at bonus
  0.40, 2.6% with the 3090 at one exchange a token and 3.7% at two a token with bonus 0.40 (the orchestrator's
  arithmetic at 42 ms a token).
- **Predictions frozen for the timed test of the bonus** (`bonus-timed-prediction.md` and `pricing/` in the run folder;
  the file was read before recording). Each pass of the replay is priced by a measured cycle, one row by pass type and
  number of loads (0.450 ms with every pick on a resident, 0.552 ms with a staged frame, 1.145, 1.463, 1.961 and 2.120
  ms with 1, 2, 3 and 4 or more loads; off arm of the scheduling test), drafting by window width and loads from the
  graph arm of 2026-11-09. MoE time is 90.6% of a one-row token and 87.5% of a drafting token.

| Against bonus 0.20, share of a token | 0.25 | 0.30 | 0.40 |
|---|---:|---:|---:|
| One row, four replayed requests | -4.9% (-5.6 to -4.2) | -8.9% (-9.5 to -8.0) | -13.7% (-14.4 to -12.4) |
| One row, the timed prose prompt | -3.1% | -6.0% | -9.7% |
| One row, the timed code prompt | -5.9% | -10.8% | -16.6% |
| One row, the two timed prompts | -4.5% (-5.3 to -3.7) | -8.4% (-9.1 to -7.3) | -13.2% (-14.1 to -11.8) |
| Drafting, four replayed requests | -7.1% | -12.5% | -20.7% |

  - One row: passes with a load fall from 12.65 a token to 9.82, 7.36 and 4.37. Most become passes with a staged
    frame, not passes with every pick on a resident, because early copies stay at 49 to 50 a token. The mix at 0.20 is
    18.0% resident, 59.8% staged and 22.2% with a load; at 0.40 it is 21.1%, 71.2% and 7.7%.
  - The timed prompts are not among the replayed requests. The prose prompt has the same share of passes with a load
    as the replayed prose request; the code prompt has 11% more than the replayed one, and its figures are scaled.
  - Drafting is less sure: widths and acceptance are held as recorded while the bonus changes the text. The agent
    rates it good for direction and to about a third in size. Its pricing reproduces measured MoE time within 3% on
    two recorded requests.
  - Held fixed and not verified: the price of each pass type across bonus values, and the text.
- **Drafting: what the 3090's cap is worth against exchanges delivered** (bonus 0.20, exchanges not charged). With
  leases only at a forward's end, delivery stops at 0.49 a token: 7.7% fewer picks needing a load, MoE time -3.2%. With
  leases at any layer pass, 0.52, 1.07, 2.09 and 2.52 a token give 9.3%, 17.1%, 21.6% and 23.1% fewer and MoE time
  -3.7%, -6.5%, -8.0% and -8.4%. It saturates near 2.5 a token, where the exchange rule runs out of candidates.
- **Not verified, and what would close it.** One-row transfers under a changed policy are bounded only, as above. A
  detailed capture that records the early copy's predicted top four would close the bound. Its main use is item 4,
  which stays deferred, so it is not taken yet.

<a id="prompt-reading-design-20261110"></a>

**Reading prompts one layer at a time: design and bound, 2026-11-10** (an Opus design agent; its result, anatomy,
crossover and sizing files were read before recording). Folder:
`/mnt/wsl/transfer/minimax-wsl/runs/prompt-reading-20261110/` (`design.md`, `result.md`, `anatomy.txt`, `bound.txt`,
`crossover.txt`, `copy_drafts.txt`, `handoff_sizing.txt`). No engine change and no GPU use. Input: the prompt rows of
the eight detailed captures of 2026-11-10, 237 to 300 tokens each.
- **Today's prompt read, measured.** A pass of 16 rows through one MoE layer takes 7.0 to 7.4 ms, and the CPU lane
  runs 4.0 to 4.5 ms of it: at most 4 experts are uploaded a pass (`mm3_model.cu:3050`), so 6 to 7 experts a pass go
  to the CPU. A chunk of 16 rows takes 0.41 to 0.44 s.
- **Routing during the prompt.** Of the router's own picks, 85 to 90% are not on a GPU. Of the picks executed, 40 to
  42% run on the 3090, 25 to 27% on the 5090's residents, 20% on staged frames and 12 to 13% on the CPU. **The bonus
  moves 52 to 57% of prompt picks off the router's top four**, against 23% in generation.
- **The design.** Per layer: dense work for every row in the same 16-row groups as today; routing of every row
  against the residents; the 5090's residents first; each host expert uploaded once through the 8 staging frames and
  applied to every row that picked it; the 3090's jobs built per group as today; no CPU lane. Rows are kept on the
  5090 at 110.6 KB a row. The check is bitwise equality of hidden states with the chunked path under the same mask
  setting.
- **Predicted prompt time:**

| Prompt | Today | Layer by layer |
|---|---|---|
| 16 rows | 0.35 to 0.38 s | 0.32 to 0.38 s |
| 32 rows | 0.70 to 0.76 s | 0.45 to 0.53 s |
| 150 tokens | 3.5 to 3.8 s | 1.1 to 1.3 s |
| 237 to 300 tokens | 5.3 to 7.2 s (measured) | 1.6 to 2.6 s |
| 1,056 tokens | 33 to 34 s (measured) | 4.2 s (3.6 to about 7) |
| 4,000 tokens | 125 to 260 s (estimated) | 16 s (14 to 22) |

  - The new read wins from 32 rows. Above about 700 tokens the 3090 sets the pace, at 32 to 40 us a column.
  - Without the bonus during the prompt: about 12 s at 4,000 tokens, but 2.1 s at 150. Its hidden states differ from
    today's.
  - Assumed, not measured: the dense time for 16 rows (0.2 to 0.6 ms a layer), the 5090's job time in large launches,
    the 3090 beyond 16 columns, and a 0.4 GB VRAM budget. Expert counts beyond 300 rows are extrapolated, and the four
    prompts share a prefix of about 150 tokens. That the kernels compute columns independently is read from source
    only; the first build step tests it.
- **What the prompt can give generation and drafting** (the user's question of 2026-11-10):
  - Seeding the residents from the prompt's picks: the share of generation's picks that land on a resident over
    tokens 0 to 127 falls by 1 to 10 points on three requests and rises by 0 to 1.8 on extraction. No gain is
    expected.
  - A table from token to experts: for tokens seen before, it predicts 46 to 48% of picks. At most 2 to 4% of
    drafting time.
  - Drafts copied from the context: 19.2% of tokens accepted on extraction (11.6% from the prompt, 7.5% from earlier
    output), 6.6% on code, 1.2% on prose and 1.4% on reasoning.
  - The drafter's context already covers every prompt row.

<a id="usability-block1-20261110"></a>

**Usability re-run, block 1: the routing bonus at 0.20, 0.25, 0.30 and 0.40, 2026-11-10** (an Opus agent read every
final answer; its write-up was read before recording). Folder:
`/mnt/wsl/transfer/minimax-wsl/runs/usability-rerun-20261110/block1/` (`results-block1.md`, `review-dump.txt`,
`review.json`). One load of `maya-mm3-eg` (5e35257), 11:09 to 12:33, at the serving configuration: **DSpark with
lenience 0.4**, `spec_min_p` 0.3, CPU price 1.5, horizon 40, `prep_graph` 1, `helper_fast` 1. Template tasks and
budgets, three seeds, 12 entries interleaved by seed, each from the restored origin.
- **Passes of three seeds, completed answers only:**

| Task | 0.20 | 0.25 | 0.30 | 0.40 |
|---|---:|---:|---:|---:|
| Scheduling | 3 | 2 | 3 | 3 |
| Muffins | 3 | 3 | 3 | 3 |
| Action items | 2 | 3 | 3 | 3 |
| Log summary | 3 | 3 | 3 | 3 |
| `order_jobs` | 2 | 2 | 3 | 3 |
| `count_runs`, first and return | 3 | 3 | 3 | 3 |
| Tool round trip | 3 | 3 | 3 | 3 |
| Choir note | 3 | 3 | 3 | 3 |

- **Wrong answers, read from the text.** Control: an `order_jobs` that walks prerequisites as dependents, and an
  action-items answer with an invented fourth item. 0.25: an `order_jobs` with its edges reversed, and a scheduling
  answer that claims two overlapping jobs do not overlap. 0.30 and 0.40: none. Three answers are doubtful and counted
  as passes (guessed dates at 0.25; an assumed meeting date and a false remark about the log at 0.30).
- **One cut-off:** 0.25, seed 3, scheduling, at 6,144 tokens with a correct answer in progress. Its re-run finished and
  is the wrong answer above.
- **Visible damage grows with the bonus.** Merged, doubled or garbled words, by the agent's reading: about 0 at 0.20,
  2 at 0.25, 12 at 0.30 and 25 at 0.40. Examples at 0.30: "because ofof the journey", "ordering of0f the jobs",
  "instead than 2", "boxes for 6" six times. At 0.40: "a a bus", "and and", "the the", "limit of10", and reasoning
  leaked into three answers, one with about 150 lines of deliberation and a literal `</think>` before the final code.
  All final functions pass their checks. One earlier control (bonus 0.20, 2026-11-08) leaked a draft once.
- **The agent's verdict:** 0.30 passes by outcomes; 0.25 fails the letter of the rule on one wrong answer, which is not
  a trend; 0.40's outcomes pass but its leaks are for a human to judge. Three seeds cannot establish that quality is
  preserved.
- **The orchestrator's reading: no higher bonus is adopted on this block.** The rule is that a faster change is
  adopted unless there is evidence of harm, and damage that rises with the dose is such evidence. It is not yet known
  whether the bonus does this by itself or only together with lenient acceptance, which keeps drafted tokens the
  target is less sure of. Two checks separate them: the same damage count on the timed tests' outputs (one row, and
  DFlash2 with exact verification), and a second block on DFlash2 crossing bonus 0.20, 0.30 and 0.40 with exact
  verification and lenience 0.4.
- **The `delivery exit` codes:** 2 means every automatic gate passed; 1 means a gate failed, here always a strict
  checker rejecting an `object()` sentinel or extra top-level statements. Run in the sandbox with `object` allowed,
  all 24 `count_runs` functions pass.

<a id="item4-bound-20261110"></a>

**Item 4 bounded: early copies only for predicted misses, 2026-11-10** (the replay agent; `item4-bound.md` was read
before recording). Folder: `/mnt/wsl/transfer/minimax-wsl/runs/reroute-replay-20261110/` (`item4-bound.md`,
`pricing/item4_bound.json`, `pricing/item4_tables.txt`, `capture/`). The user allowed the bound; nothing is built and
nothing was timed.
- **The capture** (engine commit e1561fc, capture-only): with detailed capture on, each one-row pass records the early
  copy's four predicted experts, the router's score for all 128, and what was copied or why nothing was. Taken 12:34
  to 12:48, four one-row and four drafting requests. All 138,544 predictions reproduce from the recorded scores,
  today's rule re-decided from them reproduces all 123,121 early copies and every exchange decision, and 99.997 to
  100% of rows route identically.
- **The rules replayed.** Today: the first predicted pick, routed without the bonus, that is not on a GPU is copied
  and counts as ready at once. Item 4: the prediction is routed with the bonus and the real mask, and the first pick
  still not on a GPU is copied. Item 4b: also the other predicted misses when the pass queued no demand copy; those
  count as ready only once landed. Passes are priced as in the bonus prediction; item 5 is added by the hand-off
  bench's formula per pass.
- **One row, per token from token 128, equal weight over four requests:**

| Rule | Bonus | Early copies | Picks needing a load | Resident / staged / load passes | Moved off the top four | Mass kept | MoE time against today at 0.20 | With item 5 |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| Today | 0.20 | 49.2 | 12.40 | 21 / 60 / 19% | 20.6% | 96.62% | 36.55 ms | -2.1% |
| | 0.25 | 49.3 | 8.82 | 22 / 65 / 14% | 22.1% | 95.78% | -5.3% | -7.4% |
| | 0.30 | 49.3 | 6.29 | 22 / 68 / 10% | 23.2% | 95.08% | -9.1% | -11.3% |
| | 0.40 | 49.3 | 3.27 | 23 / 72 / 5% | 24.5% | 94.01% | -13.8% | -16.0% |
| Item 4 | 0.20 | 23.8 | 16.57 | 45 / 30 / 25% | 25.5% | 95.54% | +2.5% | -3.5% |
| Item 4b | 0.20 | 27.8 | 13.66 | 45 / 34 / 21% | 25.5% | 95.52% | -1.7% | -7.8% |
| | 0.25 | 23.1 | 10.73 | 52 / 31 / 17% | 28.5% | 94.05% | -7.1% | -14.6% |
| | 0.30 | 19.2 | 8.85 | 58 / 27 / 14% | 30.9% | 92.63% | -10.8% | -19.5% |
| | 0.40 | 13.4 | 6.04 | 69 / 21 / 10% | 34.3% | 89.99% | -16.7% | -27.8% |

  - Item 4 halves the early copies and doubles the passes with every pick on a resident, 21% to 45%. Alone it is
    slower, because picks needing a load rise: today's copies happen to cover picks the bonus-routed prediction does
    not. Item 4b gets most of them back.
  - **Its worth is with item 5.** With 45% of passes needing no upload and 53% of those followed by another, item 5
    saves 2.2 ms a token instead of 0.75. Item 4b with item 5 is -7.8% of MoE time at 0.20, about -7.1% of a token,
    against -2.1% for item 5 on today's rule. The direction holds on all four requests.
  - **The routing give-back.** Moved picks rise from 20.6% to 25.5% at 0.20, with 95.5% of router mass kept. Bonus
    0.20 passed usability on 2026-11-08 at 27.3% moved and 94.97% kept, so item 4 at 0.20 stays inside that. At 0.25
    and above it goes beyond it.
  - In drafting item 4 matters little, since early copies run only in one-row passes: item 4b with item 5 is -2.3%.
- **The 3090's cap, one row, with the early copies re-decided** (this closes the bound of the earlier replay, which
  could not re-decide them and showed 3 to 5%). At 0.5, 1 and 2 exchanges a token, bonus 0.20: picks needing a load
  -10.7%, -19.3% and -25.5%; transfers -3.0%, -5.8% and -7.7%; MoE time -2.3%, -4.3% and -5.7% with exchanges free,
  and -4.7%, -7.1% and -8.7% with item 5. As built the extra exchanges cost about 0.45, 1.2 and 2.7 ms a token
  against 0.86, 1.57 and 2.09 ms saved. One exchange a token with item 4 and item 5: -9.6%.
- **More residents on the 5090** (bonus 0.20, today's rule; for the drafter and head of item 7). With 35, 60 and 95
  more: one row MoE time -1.0%, -1.8% and -2.8%; drafting -1.3%, -2.4% and -3.7%.
- **Held fixed and not verified:** the price of each pass type under the new pass mixes; exchange costs; the text and
  acceptance; that item 4b's extra copies never delay a demand copy; item 5's formula, measured on the bench and not
  in the engine.

<a id="bonus-timed-t1-20261110"></a>

**The routing bonus timed at one row, 2026-11-10, 12:49 to 13:46** (the scheduling agent ran it; `result.md` was read
before recording). Folder: `/mnt/wsl/transfer/minimax-wsl/runs/exchange-schedule-20261110/bonus-t1/` (`result.md`,
`evaluate.out`, `prediction.json`, `code-requests.txt`, `moved_share.txt`). Binary 827b8cf, one load, the one-row
base with `exchange_schedule` 0 in every arm. Twelve code blocks and eight prose blocks, arms alternated, 384 tokens,
from token 128, each request from the restored origin. All 80 timed requests are valid. The prediction was frozen
from the re-routed replay before the run and its hash is intact.

| Bonus against 0.20 | Equal weight, with 95% interval | Predicted (range) | Code, median | Predicted | Prose, median | Predicted |
|---|---|---|---|---|---|---|
| 0.25 | **-4.76%** [-6.12, -3.39] | -4.5 (-5.3 to -3.7) | -6.19%, -5.76% | -5.9 | -3.13%, -2.69% | -3.1 |
| 0.30 | **-8.94%** [-9.82, -8.06] | -8.4 (-9.1 to -7.3) | -13.61%, -13.75% | -10.8 | -4.51%, -4.23% | -6.0 |
| 0.40 | **-13.60%** [-14.35, -12.85] | -13.2 (-14.1 to -11.8) | -18.44%, -18.43% | -16.6 | -8.99%, -8.76% | -9.7 |

- **The equal-weight prediction held at all three values**, each within half a point. This is the first timed check
  of the re-routed replay priced by pass type, and it passes.
- Per prompt it is less exact: code is faster than its range at 0.30 and 0.40, and prose slower than its range at 0.30.
- With router misses a token as a covariate: code -6.05%, -12.36% and -16.89%; prose -3.60%, -5.72% and -9.22%. The
  counter includes the prompt's rows, so it is diluted.
- **Against the replay.** Picks needing a load a token in the timed requests: code 23.5, 16.4, 10.9 and 5.5; prose
  11.6, 8.7, 6.7 and 3.5. Relative to 0.20 that is -30%, -54% and -77% on code and -25%, -42% and -70% on prose,
  against the replay's -25%, -46% and -69%.
- **Picks moved off the top four, measured in the engine from token 128** (one captured request an arm and prompt):
  prose 18.1%, 18.3%, 19.7% and 20.1%; code 31.9%, 35.5%, 34.2% and 32.2%. On prose the rise is 2 points where the
  replay expected 5. On code it is not monotone, with one request an arm and a different answer in each.
- **The form of the code answer.** At 0.20 the twelve code answers run at 44.4 to 45.9 ms a token and ten stop
  naturally at 323 to 359 tokens. At 0.25 two forms appear, five at 43.5 to 43.8 ms and seven at 41.5 to 42.9 ms, so
  the 0.25 code figure mixes two texts. At 0.30 and 0.40 the answers are tight.
- This is speed only. Adoption waits for the usability question
  ([usability block 1](#usability-block1-20261110)).
- **The same answers read for text damage** (the usability agent, `timed/bonus-t1-review.md` in its run folder; 22
  answers an arm, about 8,000 tokens an arm, one row, no drafting). By the fixed-rule counter: 0.12 slips per 1,000
  tokens at 0.20 (one answer that deliberates after its code) and none at 0.25, 0.30 or 0.40. By reading: no merged,
  doubled or substituted words at any bonus. So the bonus without drafting does not show the damage of block 1.
  - Power: at block 1's rate for 0.40 about 2.6 slips an arm would be expected, so a zero has a chance of about 7%.
    The answers of an arm share one seed and one to four openings, so they are not independent.
  - **A change of behaviour with the bonus:** code answers finish within 384 tokens 11, 6, 1 and 0 times of 13 at
    0.20, 0.25, 0.30 and 0.40, and at 0.40 every code answer reasons first. Whether a higher bonus lengthens finished
    answers, and what that does to the time to a finished answer, is being counted on block 1.
  - Block 1 again with the reasoning text included: 0, 0.04, 0.22 and 1.17 slips per 1,000 tokens at 0.20 to 0.40;
    the damage sits mostly in the answers.
- **Length of finished answers by bonus** (`timed/lengths-review.md` in the usability run folder; tokens counted with
  the pack's own tokenizer).
  - Block 1, DSpark with lenience 0.4, natural stops only, summed over ten tasks and averaged over seeds, reasoning
    plus answer: 6,804 at 0.20; 7,226 at 0.25 (1.06 times); 8,894 at 0.30 (1.31 times, the reasoning grows 1.51
    times, in scheduling, `order_jobs` and `count_runs`); 8,207 at 0.40 (1.21 times, the answers grow 1.43 times,
    mostly leaked deliberation in `order_jobs`). Left out: the one cut-off and its re-run.
  - Per seed the totals are 6.3, 5.8 and 8.3 thousand at 0.20; 8.0, 10.0 and 8.7 at 0.30; 7.6, 8.9 and 8.1 at 0.40.
    Welch t about 2.3 and 1.7 on three seeds: suggestive, not established.
  - Multiplied by the timed gain a token, the time to a finished answer is 1.01, 1.19 and 1.04 times the control's
    at 0.25, 0.30 and 0.40.
  - **Paired over every finished answer** (29 to 30 pairs an arm, each the arm's tokens over the control's for the
    same task and seed; bootstrap over tasks, then seeds). Ratio of totals: 1.04 [0.80, 1.26] at 0.25, 1.31 [0.98,
    1.60] at 0.30, 1.21 [0.90, 1.81] at 0.40. Weighting each answer equally: 1.04, 1.06 and 1.39. Pairs above and
    below 1: 17 and 12, 17 and 11, 18 and 11. The only interval that excludes 1 is the reasoning at 0.30, 1.51 [1.13,
    1.87], carried by `order_jobs`, scheduling and the first `count_runs`. At 0.40 the three leaked answers trimmed to
    their final code take the total from 1.21 to 1.07 [0.85, 1.36].
  - Time to a finished answer with intervals: 0.99 [0.77, 1.20] at 0.25, 1.19 [0.89, 1.46] at 0.30, 1.04 [0.78,
    1.56] at 0.40, or 0.92 [0.74, 1.17] there with the leaks trimmed. **Every interval spans 1: three seeds cannot
    settle it.**
  - One row, 384 tokens, mean reasoning and answer tokens for code: 0 and 352 at 0.20, 0 and 355 at 0.25, 0 and 383
    at 0.30, 377 and 6 at 0.40. The requests ask for high reasoning effort with no budget, so reasoning is an allowed
    channel and not a leak. All requests share one seed, so "every code answer reasons first at 0.40" is close to
    one decision repeated.

<a id="options-fidelity-20261110"></a>

**Speed against routing fidelity: the options side by side, 2026-11-10** (the replay agent; `options-table.md` was
read before recording). Folder: `/mnt/wsl/transfer/minimax-wsl/runs/reroute-replay-20261110/` (`options-table.md`,
`row-fidelity.md`, `pricing/fidelity_tables.txt`, `pricing/floor_ranking.txt`). The same captures and pricing as
item 4's bound; 544 replays. Of these rows only today's rule at one row is timed (the replay gives -4.8%, -8.2% and
-12.5% of a token at 0.25, 0.30 and 0.40 on these captures; the timed test gave -4.76%, -8.94% and -13.60%). The rest
rests on the replay, and item 5 also on the hand-off bench's formula.
- **One row, share of a token against today at bonus 0.20:**

| Rule | Bonus | Alone | With item 5 | Picks moved | Router mass kept |
|---|---:|---:|---:|---:|---:|
| Today | 0.20 | 0 | -1.9% | 20.6% | 96.62% |
| Today | 0.25 | -4.8% | -6.7% | 22.1% | 95.78% |
| Today | 0.30 | -8.2% | -10.2% | 23.2% | 95.08% |
| Today | 0.40 | -12.5% | -14.5% | 24.5% | 94.01% |
| Item 4b | 0.20 | -1.6% | -7.1% | 25.5% | 95.52% |
| Item 4b | 0.25 | -6.4% | -13.2% | 28.5% | 94.05% |
| 3090 at one exchange a token, exchanges free | 0.20 | -3.9% | -6.4% | 18.9% | 97.01% |
| 3090 at one a token | 0.30 | -10.6% | -13.2% | 21.0% | 95.76% |
| 3090 at one a token, with item 4b | 0.20 | -5.2% | -11.8% | 24.0% | 95.90% |

- **At equal router mass kept, item 4b is not ahead of a higher bonus.** At about 95.5% kept: item 4b at 0.20 with
  item 5 gives -7.8% of MoE time; today's rule at a bonus of about 0.27 with item 5 gives -8.8% and moves 22.5% of
  picks against 25.5%. At about 94.0%: item 4b at 0.25, -14.6%; today's rule at 0.40, -16.0%.
- **The 3090's cap moves the frontier.** One exchange a token is faster and keeps more of the router's mass than
  today (97.0% against 96.6%), with fewer picks moved. It needs the exchanges to be free.
- **Drafting** already moves 26.7% of picks at 0.20 and keeps 95.10%; at 0.25 it keeps 93.50%, below the 94.97% at
  which 0.20 passed. Item 4b changes little there.
- **The mean hides a tail** (`row-fidelity.md`). At one row and bonus 0.40 the mean kept is 94.0%, but 8.7% of rows
  keep under 0.8 of their top-four mass and 2.8% under 0.7. Drafting at 0.20 already has 4.9% of rows under 0.8.
- **A floor on each row.** Substitutions are applied cheapest first while the row keeps at least a fraction F of its
  top-four mass. A floor of 0.80 is the best of those tried: it removes every row under 0.8 and keeps 34 to 43% of the
  bonus's gain at one row and 17 to 30% in drafting. At equal speed at one row, bonus 0.40 with the floor gives -5.6%
  where plain 0.25 gives -5.3%, with no row under 0.8 against 3.9% of rows. A cap of two moved picks a row keeps more
  of the gain but leaves about 60% of the rows under 0.8. Higher floors and a cap of one are dominated. The floor is
  not in the engine and not timed.

<a id="bonus-timed-plan1-20261110"></a>

**The routing bonus and scheduled exchanges timed under drafting, 2026-11-10, 13:46 to 14:42** (the scheduling agent
ran it; `result.md` was read before recording). Folder:
`/mnt/wsl/transfer/minimax-wsl/runs/exchange-schedule-20261110/bonus-plan1/` (`result.md`, `evaluate.out`,
`answers.txt`, `moved_share.txt`, `prediction.json`). Binary 827b8cf, planner v1: DFlash2 at MXFP8, exact
verification, up to 8 drafts. Ten code blocks and six prose blocks, 384 tokens, from token 128, each request from the
restored origin. All 80 timed requests are valid and the frozen prediction's hash is intact.

| Arm, against bonus 0.20 unscheduled | Equal weight, 95% interval | Predicted | Code, median | Prose, median |
|---|---|---|---|---|
| Bonus 0.20 with `exchange_schedule` 1 | **+8.31%** [+5.10, +11.52] | -0.5 | +11.80%, +10.48% | +5.79%, +4.84% |
| Bonus 0.25 | **-6.56%** [-9.48, -3.64] | -7.1 | -15.73%, -14.32% | +2.25%, +1.86% |
| Bonus 0.30 | **-17.15%** [-23.20, -11.10] | -12.5 | -28.28%, -28.69% | -5.84%, -6.55% |
| Bonus 0.40 | **-9.08%** [-17.88, -0.28] | -20.7 | -6.00%, -7.90% | -13.12%, -14.73% |

- **The bonus under drafting.** 0.25 lands on its prediction. 0.30 is faster than predicted and 0.40 much slower,
  and both are the form of the code answer, which the replay holds fixed:
  - at 0.30 every code answer is the same 310-token text, and it drafts well;
  - at 0.40 no code answer finishes within 384 tokens (9 of 10 do at 0.20), drafts accepted a token fall from 0.76
    to 0.55, and verify rounds a token rise from 0.24 to 0.45.
  - Prose is the cleaner read, since none of its answers finish within 384 tokens at any value: +2%, -6% and -13% at
    0.25, 0.30 and 0.40, with wide intervals.
- **Against the replay.** Picks needing a load a token: code 56.1, 42.5, 35.7 and 21.1 (-24%, -36%, -62%); prose
  35.2, 29.8, 24.2 and 17.7 (-15%, -31%, -50%); the replay had -19%, -34% and -56%. Picks moved off the top four in
  the capture block: prose 22.5%, 24.1%, 27.9% and 29.9%; code 38.6%, 43.7%, 48.4% and 35.7%.
- **Finished code answers within 384 tokens:** 9 of 10 at 0.20 (321 tokens), 9 of 10 at 0.25 (366), 9 of 10 at 0.30
  (310), none at 0.40. With scheduled exchanges at 0.20: 3 of 10 (355).
- **Scheduled exchanges are 8% slower under drafting**, where 0.5% faster was predicted. The 5090's exchanges fall
  from 1.45 to 0.53 a token and the 3090's from 0.153 to 0.079, and picks needing a load rise 8% on code and 3% on
  prose. Scheduled victim downloads wait for an idle pass or the forward's end; verify windows have few idle passes
  and there is about half a forward a token, so the 5090's two transactions stay pending and block new ones. At one
  row the 5090 is limited by its credit instead and kept 1.594 exchanges a token in both arms. `exchange_schedule`
  stays 0 under drafting.
- **The same answers read for text damage** (the usability agent, `timed/bonus-plan1-review.md` in its run folder; 18
  answers an arm, about 7,000 tokens an arm, one seed). **At 0.40 the doubled and merged words of block 1 appear
  without lenience**, at about block 1's rate: 0.29 per 1,000 tokens in answers against 0.32 there ("Even with with
  the doors closed", "at the entrance oflocated at", and in the reasoning "set of of dependents"); three of seven
  prose answers carry a slip. At 0.20, at 0.20 with scheduled exchanges, at 0.25 and at 0.30: none by the counter,
  and one awkward phrase at 0.25 by reading. At block 1's rate for 0.30 only about 0.9 slips would be expected, so
  that zero says little.
  - At 0.40 every code answer reasons first and reaches 384 tokens with no code, as at one row.
  - The share of code answers that finish within 384 tokens is not a length signal: the second control, the same
    bonus with scheduled exchanges, finishes 3 of 11 where the control finishes 10.
- **A pattern across the three tests, the orchestrator's reading, not established.** The damage follows the share of
  rows that keep little of their router mass ([per-row figures](#options-fidelity-20261110)), not the mean:

| Test | Bonus | Rows keeping under 0.8 of their router mass (replay) | Word damage seen |
|---|---:|---:|---|
| One row | 0.40 | 8.7% | none in 8,400 tokens |
| Drafting, exact | 0.25 | 8.8% | none in 7,000 tokens |
| Drafting, exact | 0.30 | 13.3% | none in 7,000 tokens (weak) |
| Drafting, exact | 0.40 | 21.1% | 0.29 per 1,000 tokens |
| DSpark with lenience 0.4 | 0.30, 0.40 | not replayed | about 12 and 25 slips by reading |

  Drafting moves more picks than one row at the same bonus, so the same bonus is a larger dose there. Block 2 tests
  0.30 and 0.40 under DFlash2 with about three times the text.

<a id="usability-block2-20261110"></a>

**Usability re-run, block 2: the bonus crossed with exact verification and lenience, on DFlash2, 2026-11-10, 14:42 to
16:23** (the usability agent read every final answer; `results-block2.md` was read before recording). Folder:
`/mnt/wsl/transfer/minimax-wsl/runs/usability-rerun-20261110/block2/` (`results-block2.md`, `review-dump.txt`,
`slips-all.txt`, `lengths-paired.txt`). One load of the drafting timing test's binary (827b8cf) and launch: DFlash2 at
MXFP8, planner v1, `exchange_schedule` 0. Six arms, three seeds, 18 entries interleaved by seed, each from the
restored origin. No cut-offs. The launch receipt matches the timing test's binary and runtime defaults.

| Task, passes of 3 | 0.20 exact | 0.30 exact | 0.40 exact | 0.20 lenient | 0.30 lenient | 0.40 lenient |
|---|---:|---:|---:|---:|---:|---:|
| Scheduling, muffins, action items, `count_runs`, choir note | 3 | 3 | 3 | 3 | 3 | 3 |
| Log summary | 3 | 3 | 3 | 3 | 3 | **2** |
| `order_jobs` | 3 | 3 | **2** | 3 | **2** | 3 |
| Tool round trip | 3 | 3 | **2** | 3 | 3 | **1** |
| Doubled and merged words per 1,000 tokens (counter) | 0 | 0.20 | 0.30 | 0 | 0.11 | 0.29 |
| Answers with a word slip, of 30 (by reading) | 0 | 5 | 6 | 0 | 2 | 6 |

- **0.40 fails under both acceptance rules.** In three entries the model reasons "Let me call the tool", then writes
  the plain text `</tool_call>` and stops, so no call is made; all 12 entries at 0.20 and 0.30 call correctly. One
  `order_jobs` answer is empty, its 466 tokens all reasoning that ends in a muddled draft. One log summary omits a
  fact. Sentences break: "The 14 is the last bus of can get you home".
- **0.30 with exact verification passes on outcomes and carries slips**: "with with the path", "because of of", "a
  topological order ofdeps' jobs", "a cupboard of aren't mine", "Boxes by 6". About one answer in six. With lenience,
  one `order_jobs` answer is wrong, a kind of bug the control of block 1 also made.
- **Lenience 0.4 at bonus 0.20 is clean:** no slip in 21,600 tokens by the counter or by reading, every task 3 of 3.
- **The three questions the block was built for.**
  - The bonus damages text under exact verification, at 0.30 at a low rate and at 0.40 with broken outputs.
  - Lenience 0.4 does not damage text at bonus 0.20.
  - Block 1's damage is not the two together. It follows the bonus: the exact and lenient rates are close at each
    value.
- **Length of finished answers, paired by task and seed against the same rule's 0.20 arm, ratio of totals.** At 0.30,
  pooled over both rules (60 pairs): 0.91 [0.73, 1.13]. At 0.40 (57 pairs): 1.08 [0.89, 1.37]; with exact
  verification alone 1.19, its reasoning 1.29. So block 1's lengthening at 0.30 does not repeat.
- **Time to a finished answer**, with the drafting timing (-17.15% and -9.08% a token): at 0.30, 0.76 times the
  control's [0.60, 0.95] pooled and 0.82 [0.60, 1.08] with exact verification alone; at 0.40, 0.98 [0.78, 1.27]
  pooled and 1.08 [0.83, 1.40] exact.
- **Leaked deliberation** does not follow the bonus: one answer at 0.20 exact, two at 0.30 exact, one at 0.30
  lenient, none at 0.40. All in `count_runs`, and every final function passes.
- **Verdicts.** 0.40: fails, with both rules. 0.30 with exact verification: a qualified pass, right outcomes and a
  faster finished answer, with word slips in about one answer in six; the agent leaves the trade to the user. 0.30
  with lenience: fails the letter on one answer. Lenience 0.4 at 0.20: passes. Three seeds cannot establish that
  quality is preserved; they do show 0.40 breaking outputs.

## Step 7 parameter screens

**Result, 2026-11-07.** [Protocol](integrated-screens-protocol.json), [receipts](/mnt/wsl/transfer/minimax-wsl/runs/integrated-screens-20261107/).
152 complete 384-token requests, resident engine, speculation and spinner on, recorder off, eight blocks per screen on
`val-prose` and `val-code`, no step errors. Percent of milliseconds per token over the same block's control. Qualifying needs at
least -2%, a 95% interval excluding zero and neither prompt slower. Qualifying is only a speed result.

| Screen | Control | Arm | Mean | 95% interval | Prose | Code | Qualifies |
|---|---|---|---:|---:|---:|---:|---|
| Admission | 0.58 ms saved per use, 8 exchanges | Price 0.32, interference 0.15 | +4.05% | -0.84, +8.94 | +7.78% | +0.32% | No |
| | | Budget 16 | -4.42% | -9.28, +0.43 | -6.23% | -2.61% | No |
| | | Both | -1.65% | -3.21, -0.09 | -1.26% | -2.04% | No |
| Cache-prior bonus | 0.10 | 0.15 | -6.86% | -10.66, -3.07 | -8.58% | -5.15% | Yes |
| | | 0.20 | -19.08% | -22.40, -15.77 | -16.90% | -21.26% | Yes |
| | | 0.30 | -21.54% | -26.92, -16.16 | -16.35% | -26.73% | Yes |
| Lenience | 0.4 | 0.3 | -5.65% | -8.60, -2.70 | -5.46% | -5.84% | Yes |
| | | 0.5 | +0.59% | -3.34, +4.52 | -1.53% | +2.72% | No |
| | | 0.6 | +2.52% | -1.83, +6.88 | -0.65% | +5.69% | No |
| Drafting | Spec 5, floor 0.3 | Spec 4 | -3.75% | -7.69, +0.18 | -1.69% | -5.82% | No |
| | | Floor 0.45 | -3.32% | -7.79, +1.15 | -1.59% | -5.04% | No |
| | | Spec 6 | +1.08% | -4.08, +6.24 | +2.24% | -0.08% | No |
| DRAM backing | Exclusive | Old D2H charge kept as interference | -1.23% | -3.04, +0.58 | -0.98% | -1.48% | No |
| | | Charge dropped | +6.74% | +4.42, +9.05 | +5.14% | +8.33% | No, slower |

**The bonus reaches the helper's slack, at a price in fidelity.** From 0.10 to 0.30 on code, substitutions rose 41,922 to 79,650 per
request, primary hits 50,988 to 66,834 and helper hits 61,549 to 85,487. These counters include prefill routing. Tokens per
second rose to 20.54 and 21.14 at 0.20, and 20.37 and 22.77 at 0.30, for prose and code. 0.30 over 0.20 is -3.04%, standard
deviation 6.53 across blocks, unresolved. Executed experts move further from the router's own top four as the bonus grows, so
these speeds only count if the output holds up. See the usability result below.

**Usability matrix, 2026-11-07: bonus 0.20 passes; bonus 0.30 and bonus 0.30 with lenience 0.3 fail.** [Protocol](usability-matrix-protocol.json),
[receipts](/mnt/wsl/transfer/minimax-wsl/runs/usability-matrix-20261107/), [review dump](/mnt/wsl/transfer/minimax-wsl/runs/usability-matrix-20261107/review-dump.txt).
Twelve entries, three seeds per arm, each from the restored origin. The primary agent read every final answer. Scheduling
schedules were checked by hand against dependencies, workers and the machine, not transcribed for `native_reasoning_check.py`.
The `count_runs` checker lacked `StopIteration` among its builtins and failed one correct function. It was fixed and all 24
outputs were rescored the same way, which changed only that one ([rescoring](/mnt/wsl/transfer/minimax-wsl/runs/usability-matrix-20261107/count-runs-rescored.json)).
The first run's partial entry, interrupted to sandbox the generated-code checker, is kept in `interrupted-weak-checker/`.

| Task, passes of 3 seeds | Control | Bonus 0.20 | Bonus 0.30 | Bonus 0.30, lenience 0.3 |
|---|---:|---:|---:|---:|
| Scheduling: 11 and 12, valid schedules and bounds | 3 | 3 | 3 | 3 finishes; one wrong explanation |
| Muffins: 3 combinations | 3 | 3 | 3 | 3 |
| Action items, complete within 1,024 tokens | 2 | 2 | 2 | **1** |
| Log summary, no added facts | 3 | 3 | 3 | 3 |
| `order_jobs`, eight executed checks | 3 | 3 | 3 | 3 |
| `count_runs` first attempt | 3 | 3 | **2** | 3 |
| `count_runs` return | 2 | 2 | 2 | 3 |
| Tool round trip, disconnect and follow-up | 3 | 3 | 3 | 3 |

Every failed or empty answer above ran out of its token budget during high-effort reasoning; control has two of them.
Bonus 0.30 fails the rule on `count_runs` because of one empty answer. The lenience 0.3 arm fails on action items and on
visible damage control does not show:
- a wrong explanation ("the machine idles 4-7 waiting for B", while C occupies it 3-7);
- a self-correction in the middle of the answer ("the machine idled 5-... no, it ran continuously");
- tasks listed on both workers;
- word substitutions in all three seeds: "boxes for 6", "boxes on 6", "because of of".

The word substitutions fit lenient acceptance of drafted tokens. Bonus 0.30 alone had two smaller slips ("instead than", a
garbled bus direction). Control had muddled steps too, but no substituted words. This screens for visible damage only: three
seeds per arm do not establish quality preservation.

**Selection.** Bonus 0.20 is selected for the final comparison: -19.08% [-22.40, -15.77] on speed, and it passes usability.
Lenience stays 0.4. Its 0.3 speed result stands, but its only usability arm failed with damage of the kind lenience produces.

**Backing is answered at this scope.** With the old schedule kept feasible, removing the D2H gains at most 3% at the interval's edge
and qualifies for nothing, for 13.9 GB of DRAM. Dropping the charge loses 6.74%, in line with the earlier 5.85% loss, so
that loss came from admission spending the freed charge, not from backing itself. All eight allocations and releases succeeded.

## Final comparison against the preserved best usable configuration

**Result, 2026-11-07: confirmed at both budgets, no regression.** [Protocol](final-comparison-protocol.json), rule in
`tools/mm3/native_final_comparison.py`, [evaluation](/mnt/wsl/transfer/minimax-wsl/runs/final-comparison-20261107/evaluation.json),
[receipts](/mnt/wsl/transfer/minimax-wsl/runs/final-comparison-20261107/). Primary-agent evaluation.

**Setup.** Six separate loads in the order old, new, new, old, old, new, each with the preserved reference setup:
- a profile-free bootstrap, helper-first placement from the declared history profile, and five independent conditioning requests including the tool case;
- then the learned-state origin, and twelve scored requests: three repeats at 128 and 384 tokens of `python-code` then `prose`, each pair from the origin;
- monitoring off, no telemetry, high effort, seed 20261101, identical sampling.

A load is the replicate, and each pair is two consecutive loads. All 124 steps ran without error.

**Arms.**
- Old: source `a105c563f99dc9f832b090d12c16e2be1d8635ca`, rebuilt from its own tree with its own server and defaults. The rebuilt
  binary's hash is `63e53f2e`, not the reference's `94cd6ffc`, whose file was not retained, so the arm is identified by source commit.
- New: engine `c945940` (binary `66c9271c`) with this branch's server, the spinner, and prior bonus 0.20.

| Budget | Mean pooled gain | 95% interval across pairs | Pairs | Prose | Code | Confirmed |
|---|---:|---:|---|---:|---:|---|
| 128 tokens | **+24.40%** | +8.59, +40.21 | +24.45, +18.01, +30.74 | +26.74% | +22.36% | Yes |
| 384 tokens | **+28.08%** | +21.87, +34.28 | +26.94, +26.35, +30.94 | +30.91% | +25.42% | Yes |

| Pooled tokens per second | 128 tokens | 384 tokens |
|---|---:|---:|
| Old, by load | 14.94 (14.81, 15.09, 14.94) | 15.10 (15.04, 14.89, 15.38) |
| New, by load | 18.56 (18.43, 17.81, 19.53) | 19.34 (19.10, 18.82, 20.14) |
| Old per task | Python 14.98, prose 14.91 | Python 16.12, prose 14.19 |
| New per task | Python 18.29, prose 18.84 | Python 20.16, prose 18.57 |

Every pair, prompt and budget is faster: the regression list is empty. The rebuilt old engine reproduces the historical
monitoring-off reference of 15.52 tok/s at 128 tokens to within about 4% (14.94 pooled across three loads). The scored outputs
are capped at 128 or 384 tokens, so they are speed evidence. Usability comes from the matrix above, where bonus 0.20 passed.

**Defaults check, after the comparison.** In the last load, a request with no `strata_tune` sent `helper_occupancy=1`,
`routing_prior_bonus=0.2` and `spec_ell=0.4`, completed, and held the 3090 at P2, 1,950-1,965 MHz and 9,501 MHz memory
for 27 of 33 samples while generating. `docs/minimax-wsl/native-runtime.json` now carries these values. It is identical to the file the new arm used.

**Per-pass balance at the two bonus values.** Diagnostic only, same load and origin, full records, about 14,800 passes each.
[Tables](/mnt/wsl/transfer/minimax-wsl/runs/final-comparison-20261107/bonus_balance.txt).

| Per pass | Bonus 0.10 | Bonus 0.20 |
|---|---:|---:|
| Misses per pass | 4.07 | 2.49 |
| Total cycle time | 39,983 ms | 29,845 ms |
| Mean abs(Δ) CPU vs helper, by layer | 1.436 ms (0.77-2.26) | 0.843 ms (0.42-1.57) |
| 5-row passes: CPU jobs, helper jobs, cycle | 2.78, 4.08, 3.31 ms | 1.80, 4.83, 2.51 ms |
| 5-row passes: CPU last, helper last, median CPU minus helper | 93.1%, 4.6%, +1.52 ms | 81.8%, 12.6%, +0.88 ms |

The bonus works by moving misses onto experts the helper already holds, and the CPU-helper gap shrinks accordingly. The CPU
is still last on 82% of wide passes, so the helper keeps slack that bonus 0.30 reached only by failing usability.

## Host settings

**Power plan, 2026-11-07: Balanced stays.** [Protocol](host-power-plan-protocol.json),
[receipts](/mnt/wsl/transfer/minimax-wsl/runs/host-power-plan-20261107/). Sixteen complete 384-token requests on the loaded final
configuration, the plan switched through the host-control task between paired requests. By user direction the plan with the
lower mean time per token is kept, and idle power does not matter.

- **Speed:** High performance was +1.20% slower in time per token, interval -4.10 to +6.51, with code -0.26% and prose +2.67%.
  Tokens per second: Balanced 20.00 code and 18.99 prose; High performance 20.13 and 18.50.
- **Frequency:** the 1 s host counters show the same effective CPU speed under both plans, 121.5% of the 4.3 GHz base, at
  about 60% utility. The WSL VM's virtual processors waited for dispatch equally in both arms. During generation, Balanced already
  boosts as far as High performance.
- **Ending state:** the phase ended on Balanced, so no system setting was left changed.

**Spinner against locked clocks, 2026-11-07: a speed tie; the lock uses about 16 W less on the 3090.** [Protocol](clock-lock-protocol.json),
[receipts](/mnt/wsl/transfer/minimax-wsl/runs/clock-lock-20261107/), [analysis](/mnt/wsl/transfer/minimax-wsl/runs/clock-lock-20261107/analysis.txt).
48 complete 384-token requests on the loaded final configuration, all 38 host operations successful, every lock reset at the end.

**What the locks did.** Locks followed NVIDIA's WSL benchmarking guidance: the maximum supported clocks, 2,100 / 9,751 MHz on
the 3090 and 3,090 / 14,001 MHz on the 5090. The driver accepted each lock but the cards ran below it. While busy, the 3090
held P2 at 1,935-1,965 / 9,501 MHz under the lock, as under the spinner, and the 5090 held P1 at about 2,850 / 13,801 MHz either
way. The 3090 reports Reliability and Board Limit as active clock-event reasons. The lock did keep the 3090 out of its idle
state without the spinner, which is the effect the spinner was built for.

| Arm | Speculative, time per token against spinner | `T=1` against spinner | 3090 busy power, median | 3090 energy per emitted token |
|---|---:|---:|---:|---:|
| Spinner, default clocks (control) | 19.51 tok/s | 14.36 tok/s | 198 W, 191 W | 21.03 J, 24.49 J |
| 3090 locked, no spinner | +0.84% [-4.16, +5.84] | **-0.26% [-0.46, -0.05]** | 182 W, 174 W | 20.13 J, 23.05 J |
| Both locked, no spinner | -0.33% [-3.71, +3.05] | — | 183 W | 19.93 J |
| 3090 locked and spinner | -0.17% [-4.90, +4.56] | — | 200 W | 21.67 J |

Paired values are speculative, then `T=1`. The 5090 drew 137-161 W in every arm.

**Reading.** No arm qualifies under the frozen rule, so the spinner stays in the defaults. That bounds its costs:
- Its launches, polls and handoffs cost at most about 0.26% at `T=1`, the one interval that excludes zero, and nothing
  resolvable with speculation.
- Its polling costs about 16-17 W on the 3090 while generating, about 4-6% of that card's energy per token.
- A held clock would match it with no polling, but it is a system-wide setting that would have to be applied at every engine start.
- Locking the 5090 changes nothing: it already runs at its compute clocks.

**Host-memory bandwidth, measured, 2026-11-07.** `tools/mm3/native_bandwidth_probe.cu`, run beside the idle final engine with no
settings changed. [Result](/mnt/wsl/transfer/minimax-wsl/runs/bandwidth-probe-20261107/result.json). Readers: 15 CPU threads summing
a 2 GB buffer, and repeated 32 MB uploads from pinned memory to each GPU. Two runs of 3 s per case, GB/s:

| Case | CPU | 5090 upload | 3090 upload | Total |
|---|---:|---:|---:|---:|
| CPU alone | 59.0-60.6 | | | 59-61 |
| 5090 alone | | 55.7-55.8 | | 56 |
| 3090 alone | | | 5.8-6.3 | 6 |
| CPU and 5090 | 39.6-40.3 | 35.6-36.2 | | **75.8-75.9** |
| CPU and 3090 | 55.7-58.9 | | 5.8-6.0 | 62-65 |
| 5090 and 3090 | | 49.6-50.1 | 6.4 | 56 |
| All three | 40.4-40.6 | 25.9-27.8 | 6.2 | 72.7-74.5 |

**Readings.**
- The inferred 76 GB/s ceiling is confirmed. When a CPU job and a 5090 upload overlap, each runs at about two-thirds of its
  solo rate, close to the 0.61 and 0.70 shares fitted per pass.
- The 3090 barely affects the CPU. With all three running it takes most of its share from the 5090, whose upload rate falls
  from about 36 to 26-28 GB/s, so a 32 MB record would take about 1.2 ms instead of 0.88 while a helper refill runs.
- GPU uploads together stay near 56 GB/s, the same as the 5090 alone.
- So a helper weight refill slows the copy path of every pass it overlaps. That agrees with the model's open-lease excess,
  and makes the timing of refills a per-pass balance question.

## Shared-model follow-up: timing variation

**2026-11-07, primary-agent work; no delegation. Generation only, speculation off, prefill as
setup.** The [frozen protocol](shared-model-followup-protocol.json) ran 32 runs of a fixed
241-forward fixture in two 4x4 Williams squares, then three operating-state diagnostics, on one
standalone load with binary `2c695907`, source `8f8333c`. This 241-forward fixture is the first
241 generation forwards of the earlier prose capture. All 32 runs executed identical aggregate
work: 7,576 demand uploads, 386 exchanges and 48 leases; full-record arms matched pass by pass in
at least 99.99% of passes. [Ledger](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-followup-20261107/ledger.json),
[evaluation](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-followup-20261107/recorder-evaluation.json),
[clock detail](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-followup-20261107/interpretation/clock_series.txt),
[GPU metrics](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-followup-20261107/interpretation/gpu_metrics.txt).

**Recording does not materially slow these runs.** The balanced design settles what the earlier
four short runs could not. Effects are per block against the recorder-off arm with the sampler on:

| Effect | Mean | 95% interval | Rule |
|---|---:|---:|---|
| Timestamps only | -0.45% | -1.12 to +0.23% | Bounded |
| Full records | +0.11% | -0.87 to +1.09% | Bounded |
| Full over timestamps | +0.57% | -0.63 to +1.77% | Bounded |
| Sampler, recorder off | -0.76% | -3.41 to +1.88% | Unresolved |

The sampler stays unresolved under the rule because of one run: `rec-b5-off-bare` took 24,939 ms,
8.9% above the same block's sampler arm, and that arm by design has no records or telemetry to
explain it. The other seven differences run -0.5 to +2.5%. Recording's earlier +19.33% and +10.85%
do not replicate: they came from short fixtures under helper variation, not from recording. This
covers this fixture and these arms; subdividing recorder components is not needed.

**The slowdown follows the 3090's clocks, and the driver reports it as idle.** During generation
the 3090 spent 99.5% of samples in P3 with memory at 5,001 MHz, and its core clock ran 270, 315, 795
MHz at p5, p50, p95 against a 2,100 MHz maximum. The 5090 held P1 at 13,801 MHz and 2,730-2,857 MHz.
Sampled during a run at 2 Hz, the 3090's clock-event reason was `0x1`, GPU idle, in 60 of 64 samples
and the undocumented `0x400` in the other 4; the 5090 reported `0x400` throughout. 3090 utilization
was 18-41% at about 88 W. GPU metrics agree: GR active about 30%, GPC clock about 350 MHz.

| Helper GPU time, median ms, by job count | Under 250 MHz | 250-350 | 350-450 | 450-550 | 550-700 | 700-900 | Memory at 810 MHz |
|---|---:|---:|---:|---:|---:|---:|---:|
| One job | 1.136 | 0.624 | 0.542 | 0.493 | 0.450 | 0.410 | 0.938 |
| Two jobs | 1.489 | 0.850 | 0.721 | 0.642 | 0.573 | 0.520 | 1.631 |
| Three jobs | 1.730 | 1.134 | 0.888 | 0.776 | 0.689 | 0.608 | 2.211 |
| Four jobs | | 1.275 | 1.129 | 0.881 | 0.773 | 0.677 | 2.924 |

Above about 900 MHz the level stops falling within P3 and is noisier. The predeclared link rule
returns **established**: clock state explains 62.7% of the variance of forwards' one-job helper time,
and 20 of 24 forwards over 1.4 times their run's median ran at lower clocks. At pass level, adding the
graphics clock lifts explained variance of log helper time from 38.5% to 65.1%. Clock samples also
predict helper time better 8 forwards ahead (r -0.725) than 8 behind (r -0.551), which is consistent
with clock changes leading, not with helper load setting the clock. Only 24 slow forwards occurred in these arms, which limits that part of the rule.

This is an association in timestamped observations. No clock was forced, so it is not a causal test.
Setting the helper's GPU part to its level at 700 MHz or above in every pass, then recomputing each
pass's maximum, gives a virtual saving of 13.45% of generation (12.6-15.0% across the 16 recording arms).
That figure is a per-invocation consequence, not a measured gain.

**What this does for measurement.** Lengthening the fixture cuts noise. Same-mode, same-work runs
over 241 forwards had a residual standard deviation of about 1.0% with recording on, 1.68% across all
arms including the one excursion, and 0.8% with helper state standardized. Standardizing removes helper
state by construction, so it can hide an intervention's own effect on the helper; complete totals stay
primary. At 1.68%, a whole-run paired mean needs 8 pairs for a half width of 2%, 25 for 1% and 88 for
0.5%; at 1.0%, about 5 pairs for 2%, 11 for 1% and 34 for 0.5%. So these fixtures can distinguish a 2%
policy effect with a modest declared budget, and short 64-forward fixtures cannot. The same work
does hold a rare excursion: treat it as data, not an outlier to drop.

## Shared-model follow-up: primary submission at `T=1`

**Result, 2026-11-07: useful under the frozen rule; keep `primary_submit_query=1` for ordinary
decoding.** Same load, binary, protocol and speculation-off scope as above. One `cudaStreamQuery`
after primary submission, the existing switch; there is no other change.

| Scope | Off | On | Effect |
|---|---:|---:|---:|
| Fixture, 241 forwards, 6 pairs in ABBA, BAAB, ABBA | 22,791-23,620 ms | 19,210-20,551 ms | **-15.45%**, 95% interval -18.04 to -12.87% |
| Prose request, 384 tokens, both orders | 36,473; 36,368 ms | 31,011; 31,060 ms | **-14.79%** |
| Coding request, 384 tokens, both orders | 42,305; 42,331 ms | 35,614; 35,872 ms | **-15.54%** |

Pooled across the requests that is -15.16% of generation time. Rates over the same 384 tokens move from
10.54 to 12.37 tok/s for prose and from 9.07 to 10.74 tok/s for coding. Every enabled window queried
and returned not-ready; no other status occurred. Fixture work stayed nearly identical: 7,572-7,585
demand uploads against 7,576, 386 exchanges and 48 leases in both. Request outputs are not identical
between all arms. The coding off pair matched and every other same-setting pair diverged, likely because
cache-aware routing follows a residency that changes at slightly different times. All ran 384 tokens,
finish `length`, so these are speed results, not functional passes.
[Fixture evaluation](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-followup-20261107/query-evaluation.json),
[with requests](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-followup-20261107/query-evaluation-with-requests.json),
[per-pass detail](/mnt/wsl/transfer/minimax-wsl/runs/shared-model-followup-20261107/interpretation/query_detail.txt).

**Actual GPU start and end times,** from a separately loaded Nsight pair with detailed capture on 64
forwards. This is ordering at profiled scope, not uninstrumented duration. Times are from lane begin.

| Observation | Query off | Query on |
|---|---:|---:|
| Demand copy time during host lanes | 6.2 of 932.7 ms, 0.7% | 1,282.4 of 1,285.3 ms, 99.8% |
| Copies starting before lane end | 7 of 1,631 windows | 1,631 of 1,631 |
| Copy start, median | 0.931 ms | 0.028 ms |
| Median copy duration | 0.552 ms | 0.827 ms |
| Primary expert kernel time during host lanes | 0.3 of 200.1 ms, 0.1% | 142.9 of 193.8 ms, 73.7% |
| Expert kernels starting before lane end | 9 of 3,630 windows | 2,947 of 3,630 |
| First expert start, median | 1.252 ms | 0.130 ms |
| `wait_publication` kernel time | 7.3 ms | 1,449.8 ms |

This settles the mechanism the earlier study only inferred. Without the query, launches sit unsubmitted
through the host lanes. With it, primary work runs during them and then waits on the host's word.

**Same pass, query on minus off.** Mean ms over same-work windows, pooled over the six blocks:

| New copies, lanes present | Windows | Cycle off | Cycle | Join wait | Lane span | CPU | Helper GPU |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0, helper only | 22,050 | 1.490 | -0.215 | -0.071 | -0.163 | | -0.136 |
| 0, CPU only | 2,617 | 1.070 | -0.024 | -0.103 | +0.013 | +0.013 | |
| 0, both | 30,593 | 1.469 | -0.211 | -0.065 | -0.157 | +0.010 | -0.135 |
| 1, CPU only | 5,937 | 1.843 | -0.255 | -0.599 | +0.340 | +0.340 | |
| 1, both | 18,056 | 1.998 | -0.419 | -0.583 | +0.155 | +0.332 | -0.074 |
| 2, CPU only | 420 | 2.711 | -0.445 | -1.128 | +0.668 | +0.668 | |

- **Hiding a copy is not free.** Join wait falls by about the full copy, 0.908 to 0.321 ms at one copy and
  1.450 to 0.322 ms at two. But a CPU job running through a copy stretches from about 0.56 to 0.89 ms, and
  in the trace the copy itself stretches to 0.83 ms. Both read host memory, so contention is the likely
  mechanism; this study did not isolate it. Net saving per copy is 0.26 to 0.42 ms, not the 0.55 ms that
  the earlier exposure bound implied. Averaging the copy's duration alone would miss this.
- **Helper time improved too,** 0.640 to 0.568 ms for one job, while the 3090's median clock rose from
  330 to 405 MHz and it left the 810 MHz memory state entirely in these samples. A faster run gives
  the helper a higher duty cycle. That fits a utilization-driven clock, without proving it.
- **The join wait stays flat once lanes outlast primary work:** about 0.276 ms with no copy for lanes
  over 0.3 ms, and 0.421 ms in primary-only windows with no lane to hide its kernels. That is the maximum
  the shared model needs.

**This does not reverse the earlier speculative result.** That test used grouped work, whose launches
already reach the GPU in groups, at 128 tokens with a moving control. Speculative generation keeps the
query off until tested by [step 6](README.md#shared-model-follow-up); this is new evidence for doing so.

## Helper occupancy

**Result, 2026-11-07: useful under the frozen rule, in its second and last design.** Generation only,
speculation off, query on, prefill as setup. No device setting changed and no administrator rights
were used. This also answers whether utilization sets the 3090's clocks: it does.

**First design, negative and removed.** One thread spinning for a fixed cycle count, on its own stream
with one more queued. It held the 3090 at P2, 9,501 MHz memory and 1,935-1,965 MHz core in all 6,412
samples, at 99% utilization and 164 W, but every pass waited behind it: one-job helper time rose from
0.58 to 3.53 ms and generation by 165.3% (interval 158.3-172.3%). Kernels on different streams did not
overlap here. [Receipts](/mnt/wsl/transfer/minimax-wsl/runs/helper-occupancy-20261107/occupancy_detail.txt).

**Final design.** One polling kernel on the helper's own stream, launched after each pass and released
by a host word just before the next submission, so a pass cannot queue behind it. It is bounded at 40
million cycles in case the host goes away.
[Protocol](helper-spinner-protocol.json), binary `66c9271c`, source `c945940`.

| Scope | Off | On | Effect |
|---|---:|---:|---:|
| Fixture, 241 forwards, 6 pairs | 19,550-21,185 ms | 16,543-17,576 ms | **-14.36%**, 95% interval -16.19 to -12.53% |
| Prose request, 384 tokens, both orders | 31,554; 31,512 ms | 26,852; 26,780 ms | **-14.96%** |
| Coding request, 384 tokens, both orders | 35,831; 35,775 ms | 31,440; 31,491 ms | **-12.12%** |

Over 384 tokens that is 14.32 tok/s for prose and 12.20 tok/s for coding, with the query already on.
Same-setting request pairs now agree within 0.3%. All outputs ran to `length`. As with the query, outputs
can differ between same-setting runs: one coding off arm and one on arm matched, the rest diverged.
[Fixture detail](/mnt/wsl/transfer/minimax-wsl/runs/helper-spinner-20261107/spinner_detail.txt),
[with requests](/mnt/wsl/transfer/minimax-wsl/runs/helper-spinner-20261107/spinner-evaluation-with-requests.json).

| 3090 during generation | Spinner off | Spinner on |
|---|---:|---:|
| State | P3, 5,001 MHz; brief P5, 810 MHz | P2, 9,501 MHz in all 2,045 samples |
| Core clock, p5/p50/p95 | 345, 390, 870 MHz | 1,935, 1,950, 1,965 MHz |
| Utilization, median | 34% | 79% |
| Power, median | 90 W | 180 W |
| One-job helper GPU time, mean and p90 | 0.585, 0.848 ms | 0.317, 0.381 ms |
| Four-job helper GPU time, mean and p90 | 1.000, 1.158 ms | 0.376, 0.393 ms |

The cost is about 90 W on the 3090 while generating; at 47 C it stayed well within limits. It returned to
P8 at 19 W after the requests. The control stays per request and is off unless asked, pending integrated
confirmation with speculation on. Forcing clocks stays untested and, for this purpose, looks unnecessary.

**A separate noise source, unresolved: whole-engine stalls.** Across 90 full-record runs, 15 episodes
held single forwards up to 1,187 ms, mostly in host-side submission of the next window and primary
submission, with clocks unchanged and no regular period. An episode can move a whole 241-forward run by
up to about 8%. They appear with the spinner on and off. They stay in the data. Their cause is not known.

**Separate finding: the staging decision is close to a coin flip.** The same fixture needed 7,576, 14,399
and about 8,890 demand copies across three loads, because load-time calibration moves `1.3 x CPU` against
`GPU + H2D` for a pass's last CPU job. That margin was -0.062, +0.106, +0.012 ms in these loads and +0.001 ms
in the original diagnostic. Decisions on a whole class of passes turn on measurement noise.

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
