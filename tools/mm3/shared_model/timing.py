"""Shared timing of one activation pass from its current work. One model for all MoE layers.

A pass is the join-to-join cycle that starts with a layer's work. With the submission query on,
primary copies and kernels run during the host lanes and everything meets at the host's word:

    t = 0 at primary submission returning
    cpu      = cpu_base + cpu_per_job * jobs + cpu_per_copy * new copies    (a copy slows a running job)
    helper   = a draw from the measured completion times for this many helper jobs
    lanes    = max(cpu, helper)                                             per draw, then averaged
    primary  = gpu_base + gpu_per_primary * picks + copy_ms * new copies
    cycle    = prepare + lanes + resume + wait_base + max(0, primary - lanes - resume)

Helper time stays a measured distribution, not a coefficient, because its operating state moves
it and the mean of per-pass maxima is not the maximum of mean lane durations. Forward constants
stay apart and visible, including their dependence on admissions:

    forward = prefix + first-window extra + sum(cycle) + final-layer extra + tail
    prefix  = prefix_base + prefix_per_exchange * exchanges issued at the previous forward's end
    tail    = tail_base + tail_per_exchange * exchanges issued at this end + tail_per_lease * lease begun here
    cycle  += pending_excess * (exchange or lease pending in this window)

Every parameter comes from observed passes. No parameter is per layer, and nothing reads the
future of a control run. See docs/minimax-wsl/README.md#shared-model-follow-up.
"""
from __future__ import annotations
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
import json
import math
import statistics

QUANTILES = 101


def _mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def _least_squares(rows: list[list[float]], targets: list[float]) -> list[float]:
    import numpy as np
    solution, *_ = np.linalg.lstsq(np.array(rows, dtype=float), np.array(targets, dtype=float), rcond=None)
    return [float(x) for x in solution]


def _quantiles(values: list[float], n: int = QUANTILES) -> list[float]:
    ordered = sorted(values)
    return [ordered[min(len(ordered) - 1, round(i * (len(ordered) - 1) / (n - 1)))] for i in range(n)]


@dataclass
class SharedTiming:
    prepare: dict[str, float] = field(default_factory=dict)
    cpu_base: float = 0.0
    cpu_per_job: float = 0.0
    cpu_per_copy: float = 0.0
    helper: dict[str, list[float]] = field(default_factory=dict)  # jobs -> completion-time quantiles from lane begin
    resume: float = 0.0
    wait_base: float = 0.0
    gpu_base: float = 0.0
    gpu_per_primary: float = 0.0
    copy_ms: float = 0.0
    pending_excess: float = 0.0
    final_extra: float = 0.0
    first_extra: float = 0.0
    prefix_base: float = 0.0
    prefix_per_exchange: float = 0.0
    tail_base: float = 0.0
    tail_per_exchange: float = 0.0
    tail_per_lease: float = 0.0
    outside_per_token: float = 0.0
    regime: str = 'query_on'
    notes: dict = field(default_factory=dict)

    def cpu_ms(self, cpu: int, new: int) -> float:
        return self.cpu_base + self.cpu_per_job * cpu + self.cpu_per_copy * new if cpu else 0.0

    def lanes_ms(self, cpu: int, helper: int, new: int) -> float:
        """Mean of per-draw maxima from the common origin."""
        c = self.cpu_ms(cpu, new)
        if not helper:
            return c
        draws = self.helper[str(min(helper, max(int(k) for k in self.helper)))]
        return sum(max(c, h) for h in draws) / len(draws)

    def primary_ms(self, primary: int, new: int) -> float:
        return self.gpu_base + self.gpu_per_primary * primary + self.copy_ms * new

    def cycle_ms(self, cpu: int, helper: int, primary: int, new: int, pending: bool = False) -> float:
        lanes = self.lanes_ms(cpu, helper, new)
        if self.regime == 'query_on':
            unhidden = self.exposed_ms(lanes, primary, new)
        else:
            unhidden = self.primary_ms(primary, new)
        return (self.prepare['1' if helper else '0'] + lanes + self.resume + self.wait_base + unhidden +
                (self.pending_excess if pending else 0.0))

    def exposed_ms(self, lanes: float, primary: int, new: int) -> float:
        return max(0.0, self.primary_ms(primary, new) - lanes - self.resume)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @staticmethod
    def from_dict(values: dict) -> 'SharedTiming':
        return SharedTiming(**values)


def fit(passes_by_capture: dict[str, list], budgets_by_capture: dict[str, list[dict]], activity_by_capture: dict[str, dict[int, dict]] | None = None,
        *, regime: str = 'query_on') -> SharedTiming:
    """Fit from observed one-row passes in full-record captures. Means, so totals are preserved."""
    rows = [r for rows in passes_by_capture.values() for r in rows]
    model = SharedTiming(regime=regime)
    prepare: dict[str, list[float]] = defaultdict(list)
    helper: dict[str, list[float]] = defaultdict(list)
    cpu_rows, cpu_targets, resume = [], [], []
    for r in rows:
        prepare['1' if r.helper else '0'].append(r.lane_begin)
        if r.cpu_end is not None and r.cpu:
            cpu_rows.append([1.0, r.cpu, r.new])
            cpu_targets.append(r.cpu_ms)
        if r.helper and r.helper_published is not None:
            helper[str(r.helper)].append(r.helper_published - r.lane_begin)
        if r.pre_next_ms is not None:
            resume.append(r.pre_next_ms)
    model.prepare = {k: _mean(v) for k, v in prepare.items()}
    model.prepare.setdefault('0', model.prepare.get('1', 0.0))
    model.prepare.setdefault('1', model.prepare.get('0', 0.0))
    model.cpu_base, model.cpu_per_job, model.cpu_per_copy = _least_squares(cpu_rows, cpu_targets)
    model.helper = {k: _quantiles(v) for k, v in helper.items()}
    model.resume = _mean(resume)
    # The wait once lanes outlast primary work, and primary time from what remains exposed.
    waits = [r for r in rows if r.join_wait_next_ms is not None]
    settled = [r.join_wait_next_ms for r in waits if r.new == 0 and r.lane_end - r.lane_begin > 0.3 and not r.exchanges_pending and not r.helper_lease]
    if regime == 'query_on':
        model.wait_base = _mean(settled) if settled else _mean(r.join_wait_next_ms for r in waits)
        # Primary-only passes expose their kernels; those without lanes identify gpu_base and per-pick time.
        bare = [r for r in waits if r.cpu == 0 and r.helper == 0 and r.new == 0]
        primary_rows, primary_targets = [], []
        for r in bare:
            primary_rows.append([1.0, r.resident + r.ready])
            primary_targets.append(max(0.0, r.join_wait_next_ms - model.wait_base) + (r.lane_end - r.lane_begin) + model.resume)
        if len(primary_rows) >= 10:
            base, per = statistics.fmean(primary_targets), 0.0
            model.gpu_base, model.gpu_per_primary = base - 4 * per, per
        # A demand copy's own duration when nothing contends with it, from the uncontended trace.
        model.copy_ms = 0.553
    else:
        rows_w = [[1.0, r.new, r.resident + r.ready + r.new] for r in waits]
        wait_base, model.copy_ms, model.gpu_per_primary = _least_squares(rows_w, [r.join_wait_next_ms for r in waits])
        model.wait_base = wait_base
    calm = [r for r in rows if not r.final_layer and r.index > 0]
    residual_pending = [r.cycle_ms - model.cycle_ms(r.cpu, r.helper, r.resident + r.ready + r.new, r.new) for r in calm if r.exchanges_pending or r.helper_lease]
    residual_quiet = [r.cycle_ms - model.cycle_ms(r.cpu, r.helper, r.resident + r.ready + r.new, r.new) for r in calm if not (r.exchanges_pending or r.helper_lease)]
    model.pending_excess = (_mean(residual_pending) - _mean(residual_quiet)) if residual_pending and residual_quiet else 0.0
    final, first = [], []
    for r in rows:
        predicted = model.cycle_ms(r.cpu, r.helper, r.resident + r.ready + r.new, r.new, bool(r.exchanges_pending or r.helper_lease))
        if r.final_layer:
            final.append(r.cycle_ms - predicted)
        elif r.index == 0:
            first.append(r.cycle_ms - predicted)
    model.final_extra = _mean(final)
    model.first_extra = _mean(first)
    prefix_rows, prefix_targets, tail_rows, tail_targets = [], [], [], []
    for name, budgets in budgets_by_capture.items():
        forward_activity = (activity_by_capture or {}).get(name, {})
        for b in budgets:
            current = forward_activity.get(b['ordinal'], {})
            prefix_rows.append([1.0, current.get('issued_before', 0)])
            prefix_targets.append(b['prefix_ms'])
            tail_rows.append([1.0, current.get('issued_now', 0), current.get('lease_now', 0)])
            tail_targets.append(b['tail_ms'])
    model.prefix_base, model.prefix_per_exchange = _least_squares(prefix_rows, prefix_targets)
    model.tail_base, model.tail_per_exchange, model.tail_per_lease = _least_squares(tail_rows, tail_targets)
    model.notes = dict(passes=len(rows), captures=sorted(passes_by_capture), helper_samples={k: len(v) for k, v in helper.items()})
    return model


def activity(capture) -> dict[int, dict]:
    """Per generation forward: exchanges issued at the previous forward's end, at this end, lease begun."""
    forwards = capture.generation_forwards()
    ordinal = {f['forward']: i for i, f in enumerate(forwards)}
    result = {i: dict(issued_before=0, issued_now=0, lease_now=0) for i in range(len(forwards))}
    for r in capture.records:
        if r['type'] == 'primary_issue' and r['forward'] in ordinal and r['early_layer'] < 0:
            i = ordinal[r['forward']]
            result[i]['issued_now'] += 1
            if i + 1 in result:
                result[i + 1]['issued_before'] += 1
        elif r['type'] == 'helper_event' and r['forward'] in ordinal and r['phase'] == 'lease':
            result[ordinal[r['forward']]]['lease_now'] += 1
    return result


def price(model: SharedTiming, passes: list[dict], *, issued_end: int = 0, leases: int = 0) -> dict:
    """Generation time of replayed work: per pass, then forward constants, nothing else."""
    by_forward: dict[int, list[dict]] = defaultdict(list)
    for p in passes:
        if p['generation']:
            by_forward[p['forward']].append(p)
    cycles = 0.0
    lanes = 0.0
    copies = 0
    for group in by_forward.values():
        for p in group:
            cycles += model.cycle_ms(p['cpu'], p['helper'], p['primary'], p['new'], bool(p.get('pending')))
            lanes += model.lanes_ms(p['cpu'], p['helper'], p['new'])
            copies += p['new']
        cycles += model.first_extra + model.final_extra
    n = len(by_forward)
    prefix = n * model.prefix_base + issued_end * model.prefix_per_exchange
    tail = n * model.tail_base + issued_end * model.tail_per_exchange + leases * model.tail_per_lease
    return dict(total_ms=cycles + prefix + tail, cycles_ms=cycles, prefix_ms=prefix, tail_ms=tail, forwards=n,
                passes=sum(len(g) for g in by_forward.values()), lanes_ms=lanes, copies=copies, issued_end=issued_end, leases=leases)


def price_observed(model: SharedTiming, rows: list, capture) -> dict:
    """Prices a capture's own recorded work. Its error is the timing model's, free of replay error."""
    passes = [dict(forward=r.forward, layer=r.layer, generation=True, cpu=r.cpu, helper=r.helper, primary=r.resident + r.ready + r.new,
                   new=r.new, pending=bool(r.exchanges_pending or r.helper_lease)) for r in rows]
    counts = activity(capture)
    return price(model, passes, issued_end=sum(v['issued_now'] for v in counts.values()), leases=sum(v['lease_now'] for v in counts.values()))
