"""Shared timing of one activation pass from its current work. One model for all MoE layers.

A pass is the join-to-join cycle that starts with a layer's work. With the submission query on,
primary copies and kernels run during the host lanes and everything meets at the host's word:

    t = 0 at primary submission returning
    cpu work = cpu_base + cpu_per_job * jobs; copy work = copy_ms * new copies
    while both run they share host memory, each at its own rate; whichever remains then runs alone
    helper   = a draw from the measured completion times for this many helper jobs
    lanes    = max(cpu end, helper)                                         per draw, then averaged
    primary  = gpu_base + gpu_per_primary * picks + copy end
    finish   = the host waking and merging results after the slower lane
    cycle    = prepare + lanes + finish + resume + wait_base + max(0, primary - lanes - finish - resume) + level

Helper time stays a measured distribution, not a coefficient, because its operating state moves
it and the mean of per-pass maxima is not the maximum of mean lane durations. Forward constants
stay apart and visible, including their dependence on admissions:

    forward = prefix + first-window extra + sum(cycle) + final-layer extra + tail
    prefix  = prefix_base + prefix_per_exchange * exchanges issued at the previous forward's end
    tail    = tail_base + tail_per_exchange * exchanges issued at this end + tail_per_lease * lease begun here
    cycle  += exchange_excess * (exchange pending) + lease_excess * (lease open), as observed in this window

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

DRAWS = 100


def _mean(values) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def _least_squares(rows: list[list[float]], targets: list[float]) -> list[float]:
    import numpy as np
    solution, *_ = np.linalg.lstsq(np.array(rows, dtype=float), np.array(targets, dtype=float), rcond=None)
    return [float(x) for x in solution]


def _draws(values: list[float], n: int = DRAWS) -> list[float]:
    """Means of equal-count bins of the sorted samples. This keeps the mean, including its tail."""
    ordered = sorted(values)
    size = len(ordered) / n
    return [statistics.fmean(ordered[int(i * size):max(int(i * size) + 1, int((i + 1) * size))]) for i in range(n)]


def _fit_shares(model: 'SharedTiming', waits: list) -> tuple[float, float]:
    """Rates during overlap, from passes holding CPU work and copies: CPU ends and primary ends."""
    groups: dict[tuple[int, int], dict[str, list[float]]] = defaultdict(lambda: dict(cpu=[], primary=[]))
    for r in waits:
        if r.cpu and r.new:
            g = groups[r.cpu, r.new]
            g['cpu'].append(r.cpu_ms)
            # What the wait reveals: primary end from lane begin, when it outlasted the host.
            g['primary'].append(r.lane_end - r.lane_begin + r.pre_next_ms + r.join_wait_next_ms - model.wait_base)
    means = [(c, n, len(g['cpu']), _mean(g['cpu']), _mean(g['primary'])) for (c, n), g in groups.items() if len(g['cpu']) >= 10]
    if not means:
        return 1.0, 1.0
    best, best_error = (1.0, 1.0), math.inf
    trial = SharedTiming(**{**asdict(model), 'helper': {}})
    for a in range(35, 101):
        trial.cpu_share = a / 100
        for b in range(35, 101):
            trial.copy_share = b / 100
            error = 0.0
            for c, n, count, cpu_mean, primary_mean in means:
                cpu_end, copy_end = trial.overlap_ends(c, n)
                error += count * (cpu_end - cpu_mean) ** 2
                # The primary path is only visible once it ends after the host's resume.
                host = cpu_end + trial.resume
                error += count * (max(host, model.gpu_base + copy_end) - primary_mean) ** 2
            if error < best_error:
                best, best_error = (a / 100, b / 100), error
    return best


@dataclass
class SharedTiming:
    prepare: dict[str, float] = field(default_factory=dict)
    finish: dict[str, float] = field(default_factory=dict)
    level: float = 0.0
    cpu_base: float = 0.0
    cpu_per_job: float = 0.0
    cpu_share: float = 1.0
    copy_share: float = 1.0
    helper: dict[str, list[float]] = field(default_factory=dict)  # jobs -> completion-time draws from lane begin
    resume: float = 0.0
    wait_base: float = 0.0
    gpu_base: float = 0.0
    gpu_per_primary: float = 0.0
    copy_ms: float = 0.0
    exchange_excess: float = 0.0
    lease_excess: float = 0.0
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

    def overlap_ends(self, cpu: int, new: int) -> tuple[float, float]:
        """End of CPU work and of copy work, from their common start."""
        cpu_work = self.cpu_base + self.cpu_per_job * cpu if cpu else 0.0
        copy_work = self.copy_ms * new
        if self.regime != 'query_on' or not cpu_work or not copy_work:
            return cpu_work, copy_work
        cpu_alone, copy_alone = cpu_work / self.cpu_share, copy_work / self.copy_share
        if copy_alone <= cpu_alone:
            return copy_alone + (cpu_work - self.cpu_share * copy_alone), copy_alone
        return cpu_alone, cpu_alone + (copy_work - self.copy_share * cpu_alone)

    def cpu_ms(self, cpu: int, new: int) -> float:
        return self.overlap_ends(cpu, new)[0]

    def lanes_ms(self, cpu: int, helper: int, new: int) -> float:
        """Mean of per-draw maxima from the common origin."""
        c = self.cpu_ms(cpu, new)
        if not helper:
            return c
        draws = self.helper[str(min(helper, max(int(k) for k in self.helper)))]
        return sum(max(c, h) for h in draws) / len(draws)

    def primary_ms(self, cpu: int, primary: int, new: int) -> float:
        return self.gpu_base + self.gpu_per_primary * primary + self.overlap_ends(cpu, new)[1]

    def cycle_ms(self, cpu: int, helper: int, primary: int, new: int, exchange: bool = False, lease: bool = False) -> float:
        key = '1' if helper else '0'
        lanes = self.lanes_ms(cpu, helper, new) + self.finish.get(key, 0.0)
        if self.regime == 'query_on':
            unhidden = max(0.0, self.primary_ms(cpu, primary, new) - lanes - self.resume)
        else:
            unhidden = self.primary_ms(cpu, primary, new)
        return (self.prepare[key] + lanes + self.resume + self.wait_base + unhidden + self.level +
                (self.exchange_excess if exchange else 0.0) + (self.lease_excess if lease else 0.0))

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
    finish: dict[str, list[float]] = defaultdict(list)
    cpu_rows, cpu_targets, resume = [], [], []
    for r in rows:
        prepare['1' if r.helper else '0'].append(r.lane_begin)
        finish['1' if r.helper else '0'].append(r.end - r.composed_lane_end)
        if r.cpu_end is not None and r.cpu and not r.new:
            cpu_rows.append([1.0, r.cpu])
            cpu_targets.append(r.cpu_ms)
        if r.helper and r.helper_published is not None:
            helper[str(r.helper)].append(r.helper_published - r.lane_begin)
        if r.pre_next_ms is not None:
            resume.append(r.pre_next_ms)
    model.prepare = {k: _mean(v) for k, v in prepare.items()}
    model.prepare.setdefault('0', model.prepare.get('1', 0.0))
    model.prepare.setdefault('1', model.prepare.get('0', 0.0))
    model.finish = {k: _mean(v) for k, v in finish.items()}
    model.cpu_base, model.cpu_per_job = _least_squares(cpu_rows, cpu_targets)
    model.helper = {k: _draws(v) for k, v in helper.items()}
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
        model.cpu_share, model.copy_share = _fit_shares(model, waits)
    else:
        rows_w = [[1.0, r.new, r.resident + r.ready + r.new] for r in waits]
        wait_base, model.copy_ms, model.gpu_per_primary = _least_squares(rows_w, [r.join_wait_next_ms for r in waits])
        model.wait_base = wait_base
    calm = [r for r in rows if not r.final_layer and r.index > 0]
    # An association within matched work. Admissions change it only through these flags and counts.
    model.level, model.exchange_excess, model.lease_excess = _least_squares(
        [[1.0, bool(r.exchanges_pending), bool(r.helper_lease)] for r in calm],
        [r.cycle_ms - model.cycle_ms(r.cpu, r.helper, r.resident + r.ready + r.new, r.new) for r in calm])
    final, first = [], []
    for r in rows:
        predicted = model.cycle_ms(r.cpu, r.helper, r.resident + r.ready + r.new, r.new, bool(r.exchanges_pending), bool(r.helper_lease))
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
            cycles += model.cycle_ms(p['cpu'], p['helper'], p['primary'], p['new'], bool(p.get('exchange')), bool(p.get('lease')))
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
                   new=r.new, exchange=bool(r.exchanges_pending), lease=bool(r.helper_lease)) for r in rows]
    counts = activity(capture)
    return price(model, passes, issued_end=sum(v['issued_now'] for v in counts.values()), leases=sum(v['lease_now'] for v in counts.values()))
