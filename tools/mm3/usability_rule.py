#!/usr/bin/env python3
"""The short usability check's decision rule (2026-11-14, from the audit in runs/usability-audit-20261114). Exact arithmetic,
standard library only.

Why: the template's rule, "an arm fails a task if it is wrong on more answers than control", has no margin. When both arms
share one failure rate of 18% it prints FAILS 23% of the time at 2 answers an arm, 31% at 4 and 37% at 8 (it rises toward 50%
with more answers), and at 2 answers an arm even 2 wrong against 0 has a one-sided probability of 0.17 under no difference.

The rule here gives one of three outcomes for wrong-answer counts of a tested arm and a control arm, both from ONE engine start:
  harm shown                   the tested arm is wrong more often than chance explains: one-sided exact p <= 0.05
  no harm of the stated size   harm is not shown AND a failure rate of the stated size is excluded: the tested arm's 95% upper
                               bound lies below MEANINGFUL (0.50, "failing half the time", unless the protocol states another)
  can't tell                   neither: this many answers cannot separate harm of that size from no harm
"can't tell" is not a pass and not a failure. Neither it nor "no harm of the stated size" blocks an adoption by itself: the burden
of proof is on harm (the user, 2026-11-08). They differ in what may be said afterwards: one names the size that was excluded,
the other says the run was too small to exclude it.

  fisher_one_sided(tw, tn, cw, cn)   P(tested wrong >= tw | the margins), one failure rate in both arms
  stratified_p(tables)               the same over several tasks, each kept in its own stratum (exact convolution)
  upper95(n, k)                      one-sided 95% Clopper-Pearson upper bound on a failure rate after k wrong of n
  power(tn, cn, pt, pc)              the chance the rule shows harm at true rates pt (tested) and pc (control)
  detectable(tn, cn, pc)             the smallest tested-arm rate shown as harm 80% of the time (None: no rate is)
  can_show_harm(tn, cn)              whether ANY outcome of this size reaches p <= 0.05
  size_for(pt, pc)                   answers an arm for 80% power at those rates
  decide(tw, tn, cw, cn)             the outcome with its figures
  what_it_can_show(tn, cn, base)     one sentence for a preflight or a table
"""
from __future__ import annotations

from math import comb

ALPHA = 0.05
POWER = 0.80
MEANINGFUL = 0.50   # the failure rate that would matter if a change caused it: "failing half the time" (a protocol may state another)
# Failure rates of the automatic tasks in the answers on disk when this was written (finished answers, hand-checked labels):
#   order-jobs 11 wrong of 112 since 2026-11-07 (4 of 22 on 2026-11-14); muffins 0 of 111; action-items (a fourth item) 4 of 101.
# They size a run before it starts; they never judge one.
BASE_RATES = {'order-jobs': 0.10, 'muffins': 0.01, 'action-items': 0.04}


def _pmf(n: int, p: float) -> list[float]:
    return [comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)]


def fisher_one_sided(tw: int, tn: int, cw: int, cn: int) -> float:
    tot = tw + cw
    den = comb(tn + cn, tot)
    return sum(comb(tn, x) * comb(cn, tot - x) for x in range(tw, min(tn, tot) + 1) if 0 <= tot - x <= cn) / den


def stratified_p(tables: list[tuple[int, int, int, int]]) -> float:
    """tables: (tested wrong, tested n, control wrong, control n) a task. P(the tested arm's total wrong >= observed) when, in
    each task, both arms share that task's failure rate: the convolution of each task's hypergeometric distribution."""
    dist = {0: 1.0}
    observed = 0
    for tw, tn, cw, cn in tables:
        tot = tw + cw
        den = comb(tn + cn, tot)
        task = {x: comb(tn, x) * comb(cn, tot - x) / den for x in range(max(0, tot - cn), min(tn, tot) + 1)}
        dist = _convolve(dist, task)
        observed += tw
    return min(1.0, sum(p for x, p in dist.items() if x >= observed))


def _convolve(a: dict[int, float], b: dict[int, float]) -> dict[int, float]:
    out: dict[int, float] = {}
    for x, p in a.items():
        for y, q in b.items():
            out[x + y] = out.get(x + y, 0.0) + p * q
    return out


def upper95(n: int, k: int) -> float:
    if n == 0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sum(_pmf(n, mid)[:k + 1]) > 0.05:
            lo = mid
        else:
            hi = mid
    return hi


def power(tn: int, cn: int, pt: float, pc: float, alpha: float = ALPHA) -> float:
    a, b = _pmf(tn, pt), _pmf(cn, pc)
    return sum(a[i] * b[j] for i in range(tn + 1) for j in range(cn + 1) if fisher_one_sided(i, tn, j, cn) <= alpha)


def detectable(tn: int, cn: int, pc: float, want: float = POWER, alpha: float = ALPHA) -> float | None:
    if tn == 0 or cn == 0:
        return None
    for i in range(1, 100):
        if power(tn, cn, i / 100, pc, alpha) >= want:
            return i / 100
    return None


def can_show_harm(tn: int, cn: int, alpha: float = ALPHA) -> bool:
    return tn > 0 and cn > 0 and fisher_one_sided(tn, tn, 0, cn) <= alpha


def size_for(pt: float, pc: float, want: float = POWER, alpha: float = ALPHA, limit: int = 200) -> int | None:
    """The fewest answers an arm (equal arms) at which the rule shows a tested rate pt against a control rate pc `want` of the
    time. The power is not monotonic in n for an exact test, so this is the first n from which it holds for 4 sizes in a row."""
    run = 0
    for n in range(2, limit + 1):
        run = run + 1 if power(n, n, pt, pc, alpha) >= want else 0
        if run == 4:
            return n - 3
    return None


def decide(tw: int, tn: int, cw: int, cn: int, meaningful: float = MEANINGFUL, alpha: float = ALPHA, base_rate: float | None = None) -> dict:
    """tw of tn tested answers wrong, cw of cn control answers wrong: finished, labelled answers of one task, both arms on one
    engine start. meaningful: the failure rate that would matter if the change caused it. base_rate (optional): the control's
    usual failure rate, to add the smallest rate this size shows as harm 80% of the time (smallest_rate_shown_80pct)."""
    if tn == 0 or cn == 0:
        return dict(outcome="can't tell", p=None, tested_upper95=1.0, meaningful=meaningful, can_show_harm=False, any_outcome_can_show_harm=False,
                    smallest_rate_shown_80pct=None, why='an arm has no labelled answer')
    p = fisher_one_sided(tw, tn, cw, cn)
    ub = upper95(tn, tw)
    outcome = 'harm shown' if p <= alpha else 'no harm of the stated size' if ub < meaningful else "can't tell"
    shows = can_show_harm(tn, cn, alpha)
    return dict(outcome=outcome, p=p, tested_upper95=ub, meaningful=meaningful, can_show_harm=shows, any_outcome_can_show_harm=shows,
                smallest_rate_shown_80pct=detectable(tn, cn, base_rate, alpha=alpha) if base_rate is not None else None)


def what_it_can_show(tn: int, cn: int, base_rate: float, meaningful: float = MEANINGFUL) -> str:
    """One sentence for a preflight or a table: what a run of this size can and cannot show for one task whose control fails
    near base_rate."""
    excl = [k for k in range(tn + 1) if upper95(tn, k) < meaningful]
    excludes = (f'up to {max(excl)} wrong of {tn} excludes a failure rate of {meaningful:.0%}' if excl else
                f'no result excludes a failure rate of {meaningful:.0%} (0 wrong of {tn} leaves it as high as {upper95(tn, 0):.0%})')
    if not can_show_harm(tn, cn):
        return (f'{tn} against {cn} answers: no outcome can show harm (even {tn} wrong against 0 has p = {fisher_one_sided(tn, tn, 0, cn):.2f}); {excludes}')
    return (f'{tn} against {cn} answers, control near {base_rate:.0%}: a tested arm failing {meaningful:.0%} of the time is shown as harm '
            f'{power(tn, cn, meaningful, base_rate):.0%} of the time; {excludes}')
