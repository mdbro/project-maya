"""Shared timing of one activation pass from its current work. One model for all MoE layers.

A pass is the join-to-join cycle that starts with a layer's work:

    cycle = host_prepare(helper used?)
          + lane(CPU jobs, helper jobs)      completion of the slower lane, from the common origin
          + resume_and_wait(new copies, primary picks)

Lanes are a table over the actual job pair, because the mean of per-pass maxima is not the
maximum of mean lane durations and helper operating state stretches one lane only. Forward
constants are kept apart and visible, including their dependence on admissions:

    forward = prefix(exchanges issued at the previous forward's end, helper lease open)
            + first-window extra + sum(cycles) + final-layer extra + tail(admission activity)

Parameters come from observed passes by plain means, so complete totals are preserved within
the training data. Nothing here reads a control run's future, and no parameter is per layer.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
import math
import statistics


def _mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def _least_squares(rows: list[list[float]], targets: list[float]) -> list[float]:
    import numpy as np
    solution, *_ = np.linalg.lstsq(np.array(rows, dtype=float), np.array(targets, dtype=float), rcond=None)
    return [float(x) for x in solution]


@dataclass
class SharedTiming:
    lane: dict[str, float] = field(default_factory=dict)   # 'c,h' -> mean ms from lane begin to host end
    lane_counts: dict[str, int] = field(default_factory=dict)
    prepare: dict[str, float] = field(default_factory=dict)  # '0'/'1' helper used -> join to lane begin
    resume: float = 0.0                                     # host end to the next window's submission end
    wait_base: float = 0.0
    wait_per_copy: float = 0.0
    wait_per_primary: float = 0.0
    final_extra: float = 0.0
    first_extra: float = 0.0
    prefix_base: float = 0.0
    prefix_per_exchange: float = 0.0
    prefix_per_lease: float = 0.0
    tail_base: float = 0.0
    tail_per_exchange: float = 0.0
    tail_per_lease: float = 0.0
    outside_per_token: float = 0.0
    notes: dict = field(default_factory=dict)

    def lane_ms(self, cpu: int, helper: int) -> float:
        key = f'{cpu},{helper}'
        if key in self.lane:
            return self.lane[key]
        # A pair unseen in training: the nearest seen pair with the same active lanes, by jobs.
        candidates = [(abs(int(k.split(',')[0]) - cpu) + abs(int(k.split(',')[1]) - helper), k) for k in self.lane
                      if (int(k.split(',')[0]) > 0) == (cpu > 0) and (int(k.split(',')[1]) > 0) == (helper > 0)]
        if not candidates:
            raise ValueError(f'no lane estimate for pair {key}')
        return self.lane[min(candidates)[1]]

    def cycle_ms(self, cpu: int, helper: int, primary: int, new: int) -> float:
        return (self.prepare['1' if helper else '0'] + self.lane_ms(cpu, helper) + self.resume +
                self.wait_base + self.wait_per_copy * new + self.wait_per_primary * primary)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(values: dict) -> 'SharedTiming':
        return SharedTiming(**values)


def fit(passes_by_capture: dict[str, list], budgets_by_capture: dict[str, list[dict]], activity_by_capture: dict[str, dict[int, dict]] | None = None) -> SharedTiming:
    """Fit from observed one-row passes. All captures weigh by their passes; means, not medians."""
    rows = [r for rows in passes_by_capture.values() for r in rows]
    model = SharedTiming()
    lane_samples: dict[str, list[float]] = defaultdict(list)
    prepare_samples: dict[str, list[float]] = defaultdict(list)
    resume_samples, wait_rows, wait_targets = [], [], []
    regular_residuals_first, regular_residuals_final = [], []
    for r in rows:
        lane_samples[f'{r.cpu},{r.helper}'].append(r.end - r.lane_begin)
        prepare_samples['1' if r.helper else '0'].append(r.lane_begin)
        if r.pre_next_ms is not None:
            resume_samples.append(r.pre_next_ms)
            wait_rows.append([1.0, r.new, r.resident + r.ready + r.new])
            wait_targets.append(r.join_wait_next_ms)
    model.lane = {k: _mean(v) for k, v in lane_samples.items()}
    model.lane_counts = {k: len(v) for k, v in lane_samples.items()}
    model.prepare = {k: _mean(v) for k, v in prepare_samples.items()}
    if '0' not in model.prepare:
        model.prepare['0'] = model.prepare['1']
    if '1' not in model.prepare:
        model.prepare['1'] = model.prepare['0']
    model.resume = _mean(resume_samples)
    model.wait_base, model.wait_per_copy, model.wait_per_primary = _least_squares(wait_rows, wait_targets)
    for r in rows:
        predicted = model.cycle_ms(r.cpu, r.helper, r.resident + r.ready + r.new, r.new)
        if r.final_layer:
            regular_residuals_final.append(r.cycle_ms - predicted)
        elif r.index == 0:
            regular_residuals_first.append(r.cycle_ms - predicted)
    model.final_extra = _mean(regular_residuals_final)
    model.first_extra = _mean(regular_residuals_first)
    prefix_rows, prefix_targets, tail_rows, tail_targets = [], [], [], []
    for name, budgets in budgets_by_capture.items():
        activity = (activity_by_capture or {}).get(name, {})
        for b in budgets:
            current = activity.get(b['ordinal'], {})
            prefix_rows.append([1.0, current.get('issued_before', 0), current.get('lease_open', 0)])
            prefix_targets.append(b['prefix_ms'])
            tail_rows.append([1.0, current.get('issued_now', 0), current.get('lease_now', 0)])
            tail_targets.append(b['tail_ms'])
    model.prefix_base, model.prefix_per_exchange, model.prefix_per_lease = _least_squares(prefix_rows, prefix_targets)
    model.tail_base, model.tail_per_exchange, model.tail_per_lease = _least_squares(tail_rows, tail_targets)
    model.notes = dict(passes=len(rows), captures=sorted(passes_by_capture))
    return model


def activity(capture) -> dict[int, dict]:
    """Per generation forward: exchanges issued at the previous end, lease open, issued at this end.

    Layer-boundary retentions and forward-end issues both count; a full capture records each.
    """
    forwards = capture.generation_forwards()
    ordinal = {f['forward']: i for i, f in enumerate(forwards)}
    result = {i: dict(issued_before=0, lease_open=0, issued_now=0, lease_now=0) for i in range(len(forwards))}
    lease = False
    leases = {}
    for r in capture.records:
        if r['type'] == 'primary_issue' and r['forward'] in ordinal and r['early_layer'] < 0:
            i = ordinal[r['forward']]
            result[i]['issued_now'] += 1
            if i + 1 in result:
                result[i + 1]['issued_before'] += 1
        elif r['type'] == 'helper_event' and r['forward'] in ordinal:
            i = ordinal[r['forward']]
            if r['phase'] == 'lease':
                result[i]['lease_now'] = 1
                leases[i] = True
                lease = True
            elif r['phase'] == 'retired':
                lease = False
                result[i]['lease_open'] = 1
        if r['type'] == 'forward' and r.get('generation') and lease and r['forward'] in ordinal:
            result[ordinal[r['forward']]]['lease_open'] = 1
    return result


def price(model: SharedTiming, passes: list[dict], *, activity_counts: dict | None = None, forwards: int | None = None) -> dict:
    """Generation time of replayed work: per-pass, then forward constants, nothing else."""
    generation = [p for p in passes if p['generation']]
    by_forward: dict[int, list[dict]] = defaultdict(list)
    for p in generation:
        by_forward[p['forward']].append(p)
    cycles = 0.0
    breakdown = Counter()
    for forward, group in by_forward.items():
        ordered = sorted(group, key=lambda p: p['layer'])
        for i, p in enumerate(ordered):
            cycle = model.cycle_ms(p['cpu'], p['helper'], p['primary'], p['new'])
            cycles += cycle
            breakdown['lane_ms'] += model.lane_ms(p['cpu'], p['helper'])
            breakdown['copies'] += p['new']
        cycles += model.first_extra + model.final_extra
    counts = activity_counts or {}
    n = len(by_forward)
    prefix = n * model.prefix_base + counts.get('issued', 0) * model.prefix_per_exchange + counts.get('lease_forwards', 0) * model.prefix_per_lease
    tail = n * model.tail_base + counts.get('issued', 0) * model.tail_per_exchange + counts.get('leases', 0) * model.tail_per_lease
    total = cycles + prefix + tail
    return dict(total_ms=total, cycles_ms=cycles, prefix_ms=prefix, tail_ms=tail, forwards=n, passes=len(generation),
                lane_ms=breakdown['lane_ms'], copies=breakdown['copies'])
