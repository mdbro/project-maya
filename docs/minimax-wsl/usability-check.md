# The short usability check: how to run it and how to read it

Written 2026-11-14 after an audit of that day's usability runs (`/mnt/wsl/transfer/minimax-wsl/runs/usability-audit-20261114/`). The user's direction: "The primary objective is to have these be accurate going forward."

The check is a tripwire for gross damage from a change that could lower quality. It asks the engine for a few complete answers with the change on and off, and compares how many are wrong. It cannot see small shifts; the forced-replay meter is the sensitive measure for those.

## What a run of a given size can show

A run compares wrong-answer counts on one task between a tested arm and a control arm. The figures below are exact binomial arithmetic (`tools/mm3/usability_rule.py`), for a task whose control fails 10% or 18% of the time. order-jobs failed 11 of 112 finished answers since 2026-11-07, and 4 of 22 on 2026-11-14.

| Answers per arm | Can any outcome show harm? | A tested arm failing 50% is shown as harm (control 10%) | Same, control 18% | Most wrong answers that still exclude a 50% failure rate |
|---:|---|---:|---:|---|
| 2 | no | 0% | 0% | none (0 wrong still allows 78%) |
| 4 | only 4 wrong against 0 | 4% | 3% | none (0 wrong still allows 53%) |
| 8 | yes | 33% | 19% | 1 of 8 |
| 12 | yes | 60% | 40% | 2 of 12 |
| 16 | yes | 71% | 47% | 4 of 16 |
| 20 | yes | 84% | 60% | 5 of 20 |
| 24 | yes | 90% | 67% | 7 of 24 |
| 32 | yes | 97% | 81% | 10 of 32 |

- To show a tested arm that fails half the time, 80% of the time, a run needs about 19 answers per arm when the control fails 10%, and about 32 when it fails 18%.
- To show a doubling from 18% to 36% it needs about 84 answers per arm. The short check is not the tool for that.
- An order-jobs answer takes about 30 seconds (modeled from the 2026-11-14 runs), so 20 answers per arm is about 20 minutes for both arms.

The size of a run is the user's choice. `usability_eval.py preflight` prints, before the run, what the chosen size can and cannot show.

## The three outcomes

For each task the evaluator prints one of three outcomes, from the wrong-answer counts of labelled, finished answers:

| Outcome | When | What may be said |
|---|---|---|
| harm shown | the tested arm is wrong more often than chance explains (one-sided exact test, p ≤ 0.05) | the change lowers quality on this task |
| no harm of the stated size | harm is not shown, and the 95% bound on the tested arm's failure rate is below 50% | the change does not make the task fail half the time; smaller harm was not tested |
| can't tell | neither | nothing: the run was too small |

- "Can't tell" is not a pass and not a failure.
- The adoption rule is unchanged: the burden of proof is on harm (the user, 2026-11-08). Only "harm shown", or damage a reader finds, counts against a change. The outcome lines say what size of harm the run could not have seen.
- The 50% is `meaningful_failure_rate` in the protocol; a check may state another.
- The outcome for the automatic tasks together uses an exact test that keeps each task apart. It says "no harm of the stated size" only when every task says so.

This replaces the earlier rule, "an arm fails a task if it is wrong on more seeds than control". That rule has no margin. When both arms share an 18% failure rate it prints FAILS 23% of the time at 2 answers per arm, 31% at 4 and 37% at 8.

## Before the run

A protocol must have these, or the evaluator prints tables and no verdict:

1. **The control is an arm of the same protocol**, named in `control`. Both arms then run on one engine start. A control from another run folder (`control_from`) is another engine start: a difference between the arms cannot be told from a difference between the starts.
2. **Every seed runs every arm, and the order alternates**: control, tested, tested, control, and so on.
3. **`expect` lists each arm's settings as an answer reports them, and the two lists differ.** If they are the same, the tested setting is not in each answer's record, so no answer can be shown to have run its arm. This is the case for a start option such as the K/V cache format; record it in the answer first, or the comparison cannot be verified.

Check it with:

```
python tools/mm3/usability_eval.py preflight RUN_DIR/protocol.json
```

It exits 1 when the run could give tables but no verdict. `usability_eval.py sequence --control old --tested new --first-seed N --seeds K` prints a sequence in the alternating order.

The protocol may also state `meaningful_failure_rate` (default 0.5) and `base_rates` for its tasks; the template carries both.

## The run

Unchanged: `tools/mm3/native_usability_matrix.py --protocol ... --run-dir ... --engine-pid ...`. Each tasks file now keeps the engine's pid, and each answer keeps the sampling parameters it was sent and the label of its automatic check.

## After the run

```
python tools/mm3/usability_eval.py verdict RUN_DIR
python tools/mm3/usability_eval.py blind RUN_DIR
python tools/mm3/usability_eval.py verdict RUN_DIR --labels RUN_DIR/review-labels.json
```

`verdict` writes `usability-verdict.txt` and `usability-verdict.json`. It prints an outcome only after every gate passes:

- every answer came from one engine start;
- every answer reported its arm's expected settings, and those differ between the arms;
- every arm sent the same sampling parameters and budgets;
- every entry of the protocol finished.

When a gate fails it prints the counts and "NO VERDICT", with the reason.

For tasks files written before 2026-11-14 the engine pid is not in the file. `--one-start "evidence"` states the evidence that one engine start served every answer, and prints it with the verdict.

## The checks and their labels

Every answer gets one label, with a reason:

| Label | Meaning | Counted as a failure? |
|---|---|---|
| right | the answer meets the task | no |
| wrong | the model's answer fails the task | yes |
| read | the check cannot decide; a reader labels it | not until a reader says so |
| not scored | the harness could not score it | never |
| cut off | the token budget ended the answer | never (the user, 2026-11-08) |

A task with an answer still marked read or not scored stays open: it has no outcome until a reader labels the answer.

| Task | What the automatic check does |
|---|---|
| order-jobs | Runs the function in the sandbox on eight cases, each a requirement of the task statement. wrong: no function, code that does not parse, a failed case, or no return within the 3-second CPU limit. not scored: the checker's own rules refuse a construct, the sandbox is missing, the wall clock ran out under machine load, or the only failures are builtins the runner withholds. |
| muffins | right: the final section states 3 first and the three valid pairs are named. wrong: the final section states another count and never 3. Anything else is read. |
| action-items | Structure only. wrong: the JSON list does not have exactly the three owners. read: calendar dates the notes do not give. |
| log-summary, choir-note | No automatic check: the reader labels them. |

An answer cut off while repeating itself is flagged for the reader: a loop is damage, not a budget matter.

Small text damage is counted for every arm by one fixed rule (doubled words, merged tokens, a leaked think tag, a line of thinking aloud) and reported per 1,000 tokens with each quote. It is never a failure by itself: "Substituted words alone are not a failure" (the user, 2026-11-10).

## The reader's protocol

**How to label without anchoring**

1. Work from `review-blind.md`. Each answer has a neutral name (P, Q); the arm, the token count and the settings are not shown. Do not look up which arm wrote an answer before `review-labels.json` is saved.
2. Label each answer on its own against the criteria below, `right` or `wrong`. Do not compare the two answers of a seed first.
3. For every `wrong`, give the criterion it fails and a quote. Record small damage with a quote even when the answer is right.
4. There is no "partly correct". An answer either meets every criterion or fails one.
5. Do not read the reasoning text to decide a label. The answer is what a user receives.

**What counts as wrong**

| Task | Wrong when |
|---|---|
| action-items | The list does not have exactly the three items (Tomas rebooks the crew, Lena fills the binder gaps, Priya asks finance). Or an owner, an action or a deadline contradicts the notes. Or a date the answer worked out contradicts the notes or the answer's own stated assumption. Or the summary is missing. |
| log-summary | It is not three bullets. Or one of the four facts is missing (Pump B inspected Monday; its seal replaced Tuesday; no leakage found Wednesday; Pump A not inspected). Or a fact is added, including giving a day to "Pump A was not inspected". |
| choir-note | It leaves out the move to a new hall next month, the way there by bus, or what to bring on the first evening. Or its instructions contradict each other on where or when. Or it repeats itself, breaks off, or carries thinking aloud. Placeholders in brackets are fine. |
| muffins (when left to read) | It does not conclude that 3 combinations work, or its three are not (7,3), (4,5), (1,7). |
| order-jobs (when not scored) | Read the function against the task statement; if in doubt leave it not scored and say why. |

Two of these lines are proposals that change how earlier answers would have been rated, and need the user's confirmation:

- **Worked-out dates.** A relative deadline turned into a calendar date under a stated, consistent assumption is right. The 2026-11-14 reviews called it "partly correct".
- **A day given to Pump A.** Filing "Pump A was not inspected" under a weekday is an added fact, so wrong. The 2026-11-14 review called it a small framing point.

**What counts as worse.** An arm is worse on a judged task only through its wrong counts, by the same three outcomes as the automatic tasks. A reader's impression that one arm reads better is recorded as a note and is not an outcome.

## Reading the verdict file

The file has five parts, in this order:

1. **Can a verdict be given?** The gates. If any says no, nothing below is attributed to the tested setting.
2. **Answers by task.** Measured counts for each arm: answers, right, wrong, to read, not scored, cut off.
3. **Outcome by task.** The counts, the outcome, the p value, the 95% bound on the tested arm's failure rate, and what this size can show.
4. **Small text damage.** Slips per arm and per 1,000 tokens, with each quote.
5. **For a reader.** Every answer still open, with the reason.

## What the 2026-11-14 audit found

- order-jobs was scored right on all 22 answers of the day.
- The muffins checker marked 7 of 22 right answers wrong at run time. Before 13:47 PST that day it had never marked a real answer right. Its rewrite (3adf4455) still marked 4 of the 89 earlier real answers wrong.
- Three of the day's five comparisons used a control from another engine start.
- No outcome of a 2-seed run can show harm, and a clean 2-seed result leaves the failure rate as high as 78%.

The answers, the hand labels and the scripts are in the audit folder. The unit tests (`tools/mm3/test_usability_checks.py`) run the checks on those real answers.
