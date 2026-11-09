"""The shared pass model for grouped windows, with speculation on. One model for all MoE layers.

Grouped windows change what a pass holds, not how it composes. Rows set the fixed parts. Lanes
work on distinct experts and their columns. A window of two or more rows reaches the launch group, so
its primary work overlaps the lanes without a query; a one-row window does not, unless queried.

    t = 0 at primary submission returning
    cpu work  = cpu_base[rows] + cpu_per_job * jobs + cpu_per_column * (columns - jobs)
    copy work = copy_ms * new copies; while both run they share host memory at the fitted rates
    helper    = helper_base[rows] + helper_per_job * jobs + helper_per_column * columns + a residual draw
    lanes     = max(cpu end, helper)                                        per draw, then averaged
    primary   = gpu_base[rows] + gpu_per_job * primary jobs + copy end
    cycle     = prepare[rows, helper used] + lanes + finish + resume[rows] + wait[rows]
                + max(0, primary - lanes - finish - resume) when overlapping, or + primary when sequential

Forward constants per rows, and time outside forwards from drafting and sampling, stay visible.
Rows of a rejected draft stay in the recorded window, so a candidate pays for them as the control did.
Nothing here is fitted per layer, and nothing reads a control run's future.
"""
from __future__ import annotations
from collections import defaultdict
from dataclasses import asdict, dataclass, field
import json
import math
import statistics

from .timing import _draws, _least_squares, _mean, activity


@dataclass
class GroupedTiming:
    prepare: dict[str, float] = field(default_factory=dict)      # 'rows,helper' -> ms
    finish: dict[str, float] = field(default_factory=dict)       # '0' or '1'
    resume: dict[str, float] = field(default_factory=dict)       # rows -> ms
    wait: dict[str, float] = field(default_factory=dict)         # rows -> ms
    cpu_base: dict[str, float] = field(default_factory=dict)     # rows -> ms
    cpu_per_job: float = 0.0
    cpu_per_column: float = 0.0
    cpu_share: float = 0.61
    copy_share: float = 0.67
    copy_ms: float = 0.553
    copy_head: dict[str, float] = field(default_factory=dict)  # rows -> ms the uploads run before the CPU lane starts
    helper_base: dict[str, float] = field(default_factory=dict)  # rows -> ms
    helper_per_job: float = 0.0
    helper_per_column: float = 0.0
    helper_residual: list[float] = field(default_factory=list)
    gpu_base: dict[str, float] = field(default_factory=dict)     # rows -> ms
    gpu_per_job: float = 0.0
    # Measured 5090 path (window-trace-20261109), for rows that overlap: the later of its resident path and the last
    # upload's end plus the kernels that read the uploads. Empty: the additive gpu_base form above.
    gpu_resident: dict[str, float] = field(default_factory=dict)  # rows -> ms from the CPU lane's start
    gpu_resident_per_job: float = 0.0
    gpu_tail: dict[str, float] = field(default_factory=dict)      # rows -> ms from the last upload's end to the arrival
    exchange_excess: float = 0.0
    lease_excess: float = 0.0
    level: float = 0.0
    final_extra: dict[str, float] = field(default_factory=dict)  # rows -> ms
    first_extra: float = 0.0
    prefix_base: dict[str, float] = field(default_factory=dict)  # rows -> ms
    prefix_per_exchange: float = 0.0
    tail_base: float = 0.0
    tail_per_exchange: float = 0.0
    tail_per_lease: float = 0.0
    outside_base: float = 0.0
    outside_per_round: float = 0.0
    outside_per_token: float = 0.0
    query: bool = False
    notes: dict = field(default_factory=dict)

    def _rows(self, table: dict[str, float], rows: int) -> float:
        key = str(rows)
        if key in table:
            return table[key]
        known = sorted(int(k) for k in table)
        return table[str(min(known, key=lambda k: abs(k - rows)))]

    def overlaps(self, rows: int) -> bool:
        return rows >= 2 or self.query

    def head(self, rows: int) -> float:
        """How long the uploads run, alone, before the CPU lane starts (they are issued early in the submission)."""
        return self._rows(self.copy_head, rows) if self.copy_head and self.overlaps(rows) else 0.0

    def overlap_ends(self, rows: int, cpu_jobs: int, cpu_columns: int, new: int) -> tuple[float, float]:
        """The CPU lane's and the uploads' ends, from the CPU lane's start. Uploads run alone for their head start, then
        share host memory with the CPU lane until one finishes; the other then runs at its own rate."""
        cpu_work = self._rows(self.cpu_base, rows) + self.cpu_per_job * cpu_jobs + self.cpu_per_column * (cpu_columns - cpu_jobs) if cpu_jobs else 0.0
        copy_work = self.copy_ms * new
        lead = self.head(rows)
        rest = max(0.0, copy_work - lead)
        if not self.overlaps(rows) or not cpu_work or not rest:
            return cpu_work, copy_work - lead
        cpu_alone, copy_alone = cpu_work / self.cpu_share, rest / self.copy_share
        if copy_alone <= cpu_alone:
            return copy_alone + (cpu_work - self.cpu_share * copy_alone), copy_alone
        return cpu_alone, cpu_alone + (rest - self.copy_share * cpu_alone)

    def lanes_ms(self, rows: int, cpu_jobs: int, cpu_columns: int, helper_jobs: int, helper_columns: int, new: int) -> float:
        cpu_end = self.overlap_ends(rows, cpu_jobs, cpu_columns, new)[0]
        if not helper_jobs:
            return cpu_end
        mean = self._rows(self.helper_base, rows) + self.helper_per_job * helper_jobs + self.helper_per_column * helper_columns
        return sum(max(cpu_end, mean + r) for r in self.helper_residual) / len(self.helper_residual)

    def primary_ms(self, rows: int, cpu_jobs: int, cpu_columns: int, primary_jobs: int, new: int) -> float:
        """The 5090's arrival at the wait, from the CPU lane's start."""
        copy_end = self.overlap_ends(rows, cpu_jobs, cpu_columns, new)[1]
        if self.gpu_tail and self.overlaps(rows):
            resident = self._rows(self.gpu_resident, rows) + self.gpu_resident_per_job * primary_jobs
            return max(resident, copy_end + self._rows(self.gpu_tail, rows)) if new else resident
        return self._rows(self.gpu_base, rows) + self.gpu_per_job * primary_jobs + copy_end

    def cycle_ms(self, rows: int, cpu_jobs: int, cpu_columns: int, helper_jobs: int, helper_columns: int, primary_jobs: int, new: int,
                 exchange: bool = False, lease: bool = False) -> float:
        key = '1' if helper_jobs else '0'
        lanes = self.lanes_ms(rows, cpu_jobs, cpu_columns, helper_jobs, helper_columns, new) + self.finish.get(key, 0.0)
        resume = self._rows(self.resume, rows)
        primary = self.primary_ms(rows, cpu_jobs, cpu_columns, primary_jobs, new)
        exposed = max(0.0, primary - lanes - resume) if self.overlaps(rows) else primary
        prepare = self.prepare.get(f'{rows},{key}', self.prepare.get(f'{rows},0', 0.1))
        return (prepare + lanes + resume + self._rows(self.wait, rows) + exposed + self.level +
                (self.exchange_excess if exchange else 0.0) + (self.lease_excess if lease else 0.0))

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @staticmethod
    def from_dict(values: dict) -> 'GroupedTiming':
        return GroupedTiming(**values)


def _columns(r) -> tuple[int, int]:
    return r.cpu, r.helper  # picks per lane are its columns


def shares_from_stretch(stretch: float, copy_ms: float, *, cpu_share: float = 0.61, copy_share: float | None = None) -> tuple[float, float]:
    """The CPU's and the copy's rates while both run, as shares of each one's rate alone.

    The CPU lane's stretch per concurrent copy is the copy's duration beside the CPU times the share the CPU loses:
    stretch = (copy_ms / copy_share) * (1 - cpu_share). Given one share, the other follows: the CPU's from the one-row
    study by default, or the copy's when a trace measured it (window-trace-20261109: a demand upload beside the CPU lane
    takes 0.836 ms against 0.552 alone, a share of 0.66)."""
    if copy_share is None:
        return cpu_share, (1 - cpu_share) * copy_ms / stretch
    cpu = 1 - stretch * copy_share / copy_ms
    if not 0 < cpu <= 1:
        raise ValueError(f'a copy share of {copy_share} with a stretch of {stretch:.3f} ms implies a CPU share of {cpu:.3f}')
    return cpu, copy_share


def copy_heads(passes_by_capture: dict[str, list], captures: dict, lag_ms: float) -> dict[str, float]:
    """Median head start of the uploads on the CPU lane by rows: the submission's end (the CPU lane's start) minus the first
    demand copy's submission end, less the lag from a copy's submission to its DMA start (window-trace-20261109: 0.045 ms)."""
    heads: dict[str, list[float]] = defaultdict(list)
    for name, passes in passes_by_capture.items():
        rows = {(p.forward, p.layer): p.rows for p in passes}
        first: dict[tuple[int, int], int] = {}
        for r in captures[name].of('demand_copy'):
            key = (r['forward'], r['layer'])
            t = r['t_ns'] + r['submit_end']
            first[key] = min(first.get(key, t), t)
        for r in captures[name].of('layer'):
            key = (r['forward'], r['layer'])
            if key in first and key in rows:
                heads[str(rows[key])].append((r['t_ns'] + r['primary_submit_end'] - first[key]) * 1e-6 - lag_ms)
    return {k: max(0.0, statistics.median(v)) for k, v in heads.items()}


def fit(passes_by_capture: dict[str, list], budgets_by_capture: dict[str, list[dict]], captures: dict, *, query: bool = False,
        copy_share: float | None = None, copy_lag_ms: float | None = None, gpu_terms: dict | None = None) -> GroupedTiming:
    """copy_share: the copy's measured share while the CPU runs; the CPU's share then follows from the fitted stretch.
    copy_lag_ms: give the uploads their measured head start on the CPU lane (copy_heads), with this submission-to-DMA lag.
    gpu_terms: the 5090 path measured in a trace (resident by rows, resident_per_job, tail by rows), in place of gpu_base."""
    rows = [r for t in passes_by_capture.values() for r in t]
    model = GroupedTiming(query=query)
    if copy_lag_ms is not None:
        model.copy_head = copy_heads(passes_by_capture, captures, copy_lag_ms)
    if gpu_terms:
        model.gpu_resident = {str(k): float(v) for k, v in gpu_terms['resident'].items()}
        model.gpu_resident_per_job = float(gpu_terms['resident_per_job'])
        model.gpu_tail = {str(k): float(v) for k, v in gpu_terms['tail'].items()}
    prepare, resume, wait, finish = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    for r in rows:
        key = '1' if r.helper_jobs else '0'
        prepare[f'{r.rows},{key}'].append(r.lane_begin)
        finish[key].append(r.end - r.composed_lane_end)
        if r.pre_next_ms is not None:
            resume[str(r.rows)].append(r.pre_next_ms)
    model.prepare = {k: _mean(v) for k, v in prepare.items()}
    model.finish = {k: _mean(v) for k, v in finish.items()}
    model.resume = {k: _mean(v) for k, v in resume.items()}
    # CPU work and its stretch per copy from all passes where every copy can overlap it. Passes with no
    # copies nearly always hold one job, so they alone cannot identify the price per job. The CPU's rate
    # during overlap comes from the one-row study; the copy's rate follows from the stretch.
    known = sorted({r.rows for r in rows})
    cpu_rows, cpu_targets = [], []
    for r in rows:
        if r.cpu_jobs and r.cpu_end is not None and r.new <= r.cpu_jobs and (model.overlaps(r.rows) or not r.new):
            overlapping = max(0.0, model.copy_ms * r.new - model.head(r.rows)) / model.copy_ms   # copies left when the CPU starts
            cpu_rows.append([float(r.rows == k) for k in known] + [r.cpu_jobs, r.cpu - r.cpu_jobs, overlapping])
            cpu_targets.append(r.cpu_ms)
    solved = _least_squares(cpu_rows, cpu_targets)
    model.cpu_base = {str(k): v for k, v in zip(known, solved)}
    model.cpu_per_job, model.cpu_per_column = solved[len(known)], solved[len(known) + 1]
    stretch = solved[len(known) + 2]
    model.cpu_share, model.copy_share = shares_from_stretch(stretch, model.copy_ms, cpu_share=model.cpu_share, copy_share=copy_share)
    # Helper completion from lane begin: fitted mean plus a residual draw, so a slow pass stays in.
    helper_rows, helper_targets, helper_items = [], [], []
    for r in rows:
        if r.helper_jobs and r.helper_published is not None:
            helper_rows.append([float(r.rows == k) for k in known] + [r.helper_jobs, r.helper])
            helper_targets.append(r.helper_published - r.lane_begin)
            helper_items.append(r)
    solved = _least_squares(helper_rows, helper_targets)
    model.helper_base = {str(k): v for k, v in zip(known, solved)}
    model.helper_per_job, model.helper_per_column = solved[len(known)], solved[len(known) + 1]
    residual = [t - sum(a * b for a, b in zip(row, solved)) for row, t in zip(helper_rows, helper_targets)]
    model.helper_residual = _draws(residual)
    # The settled wait per rows: no copy, lanes that outlast primary work, nothing pending.
    settled = defaultdict(list)
    for r in rows:
        if r.join_wait_next_ms is not None and r.new == 0 and r.lane_end - r.lane_begin > 0.3 and not r.exchanges_pending and not r.helper_lease and model.overlaps(r.rows):
            settled[str(r.rows)].append(r.join_wait_next_ms)
    model.wait = {k: _mean(v) for k, v in settled.items()}
    # Sequential one-row windows: wait with no copy, and primary work after it.
    sequential = [r for r in rows if not model.overlaps(r.rows) and r.join_wait_next_ms is not None]
    if sequential:
        no_copy = [r.join_wait_next_ms for r in sequential if r.new == 0]
        model.wait['1'] = _mean(no_copy)
        model.gpu_base['1'] = 0.0
    # Primary path: bare passes per rows show what lanes do not hide.
    for k in known:
        if str(k) in model.gpu_base:
            continue
        bare = [r for r in rows if r.rows == k and r.join_wait_next_ms is not None and r.cpu_jobs == 0 and r.helper_jobs == 0 and r.new == 0]
        base = model.wait.get(str(k), 0.3)
        if len(bare) >= 10:
            model.gpu_base[str(k)] = _mean(max(0.0, r.join_wait_next_ms - base) + model.resume[str(k)] for r in bare)
        else:
            model.gpu_base[str(k)] = 0.24 + 0.05 * (k - 1)
    calm = [r for r in rows if not r.final_layer and r.index > 0]
    predict = lambda r, flags: model.cycle_ms(r.rows, r.cpu_jobs, r.cpu, r.helper_jobs, r.helper, r.primary_jobs, r.new, *flags)
    model.level, model.exchange_excess, model.lease_excess = _least_squares(
        [[1.0, bool(r.exchanges_pending), bool(r.helper_lease)] for r in calm], [r.cycle_ms - predict(r, (False, False)) for r in calm])
    final = defaultdict(list)
    first = []
    for r in rows:
        value = r.cycle_ms - predict(r, (bool(r.exchanges_pending), bool(r.helper_lease)))
        if r.final_layer:
            final[str(r.rows)].append(value)
        elif r.index == 0:
            first.append(value)
    model.final_extra = {k: _mean(v) for k, v in final.items()}
    model.first_extra = _mean(first)
    prefix_rows, prefix_targets, tail_rows, tail_targets = [], [], [], []
    forwards_rows = {}
    for name, budgets in budgets_by_capture.items():
        counts = activity(captures[name])
        by_ordinal = {i: f['rows'] for i, f in enumerate(captures[name].generation_forwards())}
        for b in budgets:
            current = counts.get(b['ordinal'], {})
            prefix_rows.append([float(by_ordinal[b['ordinal']] == k) for k in known] + [current.get('issued_before', 0)])
            prefix_targets.append(b['prefix_ms'])
            tail_rows.append([1.0, current.get('issued_now', 0), current.get('lease_now', 0)])
            tail_targets.append(b['tail_ms'])
    solved = _least_squares(prefix_rows, prefix_targets)
    model.prefix_base = {str(k): v for k, v in zip(known, solved)}
    model.prefix_per_exchange = solved[len(known)]
    model.tail_base, model.tail_per_exchange, model.tail_per_lease = _least_squares(tail_rows, tail_targets)
    outside_rows, outside_targets = [], []
    for name, capture in captures.items():
        end = capture.of('generation_end')
        if not end or end[0]['scope'] != 'request':
            continue
        forwards = capture.generation_forwards()
        total = capture.scored_ms() - sum(f['end'] for f in forwards) * 1e-6
        outside_rows.append([len(forwards), end[0]['emitted']])
        outside_targets.append(total)
    if outside_rows:
        model.outside_per_round, model.outside_per_token = _least_squares(outside_rows, outside_targets)
    model.notes = dict(passes=len(rows), captures=sorted(passes_by_capture), rows=known, stretch_ms=stretch,
                       shares='copy share measured, CPU share derived' if copy_share is not None else 'CPU share assumed, copy share derived')
    return model


def price(model: GroupedTiming, passes: list[dict], *, issued_end: int = 0, leases: int = 0, rounds: int | None = None, tokens: int = 0) -> dict:
    by_forward: dict[int, list[dict]] = defaultdict(list)
    for p in passes:
        if p['generation']:
            by_forward[p['forward']].append(p)
    cycles = lanes = 0.0
    copies = 0
    prefix = 0.0
    for group in by_forward.values():
        rows = group[0]['rows']
        for p in group:
            cycles += model.cycle_ms(p['rows'], p['cpu_jobs'], p['cpu_columns'], p['helper_jobs'], p['helper_columns'], p['primary_jobs'], p['new'], bool(p.get('exchange')), bool(p.get('lease')))
            lanes += model.lanes_ms(p['rows'], p['cpu_jobs'], p['cpu_columns'], p['helper_jobs'], p['helper_columns'], p['new'])
            copies += p['new']
        cycles += model.first_extra + model._rows(model.final_extra, rows)
        prefix += model._rows(model.prefix_base, rows)
    n = len(by_forward)
    prefix += issued_end * model.prefix_per_exchange
    tail = n * model.tail_base + issued_end * model.tail_per_exchange + leases * model.tail_per_lease
    outside = (rounds if rounds is not None else 0) * model.outside_per_round + tokens * model.outside_per_token
    return dict(total_ms=cycles + prefix + tail + outside, forward_ms=cycles + prefix + tail, cycles_ms=cycles, prefix_ms=prefix, tail_ms=tail, outside_ms=outside,
                forwards=n, passes=sum(len(g) for g in by_forward.values()), lanes_ms=lanes, copies=copies, issued_end=issued_end, leases=leases)


def recorded_passes(rows: list) -> list[dict]:
    return [dict(forward=r.forward, layer=r.layer, generation=True, rows=r.rows, cpu_jobs=r.cpu_jobs, cpu_columns=r.cpu, helper_jobs=r.helper_jobs, helper_columns=r.helper,
                 primary_jobs=r.primary_jobs, new=r.new, exchange=bool(r.exchanges_pending), lease=bool(r.helper_lease)) for r in rows]


def price_observed(model: GroupedTiming, rows: list, capture) -> dict:
    counts = activity(capture)
    return price(model, recorded_passes(rows), issued_end=sum(v['issued_now'] for v in counts.values()), leases=sum(v['lease_now'] for v in counts.values()))
