#!/usr/bin/env python3
"""The joint timeline of each activation pass through each MoE layer, from a light capture.

One row per (forward, layer). Every time is in milliseconds from that layer's own
join, the common origin, so operations of one pass can be compared by when they
complete rather than by how long they lasted. Rows keep the pass's actual work.

What a light capture can and cannot say:
  - Host boundaries and helper-worker boundaries are direct observations.
  - A CUDA submission returning is not the GPU starting. Light records hold no GPU
    start or end times; `native_shared_trace.py` reads those from an Nsight export.
  - A wait observed at the next layer's join can include unfinished work from this
    layer, so `next_join_wait_ms` is attributed to the join-to-join cycle as a whole.

Consequence calculations (`consequence`) act on each pass and then aggregate. An
average of per-pass maxima is not a maximum of averages.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
import statistics

SERVICES = ('cpu', 'helper', 'primary_resident', 'staged_ready', 'staged_new')
MS = 1e-6


@dataclass
class Pass:
    forward: int
    ordinal: int
    layer: int
    index: int
    position: int
    final_layer: bool
    # Actual work of this pass.
    cpu: int
    helper: int
    resident: int
    ready: int
    new: int
    # Boundaries, ms after this layer's join. None when the operation did not occur.
    boundary: float
    plan: float
    helper_submit: float | None
    primary_submit: float
    cpu_end: float | None
    collect_end: float | None
    end: float
    helper_enqueue: float | None
    helper_worker_start: float | None
    helper_record_end: float | None
    helper_ready: float | None
    helper_published: float | None
    # Dependent work: the host resumes, submits the next window, then waits at its join.
    next_pre_end: float | None
    cycle_ms: float
    # Observed states kept for residual checks. They are not predictor inputs by default.
    exchanges_pending: int = 0
    helper_lease: int = 0
    reserved_slots: int = 0
    prior_new: int = 0
    prior_cpu: int = 0
    prior_helper: int = 0
    # Grouped windows: rows evaluated, and distinct experts per lane. One row makes them the picks.
    rows: int = 1
    cpu_jobs: int = 0
    helper_jobs: int = 0
    primary_jobs: int = 0

    @property
    def key(self) -> tuple[int, int, int, int, int]:
        return self.cpu, self.helper, self.resident, self.ready, self.new

    @property
    def lane_begin(self) -> float:
        """The CPU lane starts once primary submission has returned."""
        return self.primary_submit

    @property
    def cpu_ms(self) -> float:
        return self.cpu_end - self.lane_begin if self.cpu_end is not None else 0.0

    @property
    def helper_service_ms(self) -> float | None:
        """Enqueue to ready observed; includes hand-off, submission and GPU execution."""
        if self.helper_enqueue is None or self.helper_ready is None:
            return None
        return self.helper_ready - self.helper_enqueue

    @property
    def helper_gpu_ms(self) -> float | None:
        """Event recorded to ready observed; the part that stretches in slow periods."""
        if self.helper_record_end is None or self.helper_ready is None:
            return None
        return self.helper_ready - self.helper_record_end

    @property
    def lane_end(self) -> float:
        """Observed completion of the host lanes."""
        return max(x for x in (self.cpu_end, self.collect_end, self.lane_begin) if x is not None)

    @property
    def composed_lane_end(self) -> float:
        """Maximum of the two lanes' completion times from the common origin."""
        return max(x for x in (self.cpu_end, self.helper_published, self.lane_begin) if x is not None)

    @property
    def host_ms(self) -> float:
        return self.end

    @property
    def pre_next_ms(self) -> float | None:
        return None if self.next_pre_end is None else self.next_pre_end - self.end

    @property
    def join_wait_next_ms(self) -> float | None:
        return None if self.next_pre_end is None else self.cycle_ms - self.next_pre_end


def _after(row: dict, name: str, join_ns: int) -> float | None:
    value = row[name]
    if not value:
        return None
    return (row['t_ns'] + value - join_ns) * MS


def work_of(capture) -> dict[tuple[int, int], Counter]:
    """Service counts by (generation ordinal, layer) from a full capture."""
    ordinals = {f['forward']: i for i, f in enumerate(capture.generation_forwards())}
    return {(ordinals[r['forward']], r['layer']): Counter(r['service']) for r in capture.of('layer') if r['forward'] in ordinals}


def passes(capture, generation_only: bool = True, work: dict | None = None) -> list[Pass]:
    """All passes of the capture's forwards, in execution order.

    A timestamps-only capture holds no assignments; pass `work` from a full capture of
    the same fixed work. Identical work across arms is checked separately, not assumed.
    """
    if capture.meta.get('mode') == 'timestamps' and work is None:
        raise ValueError('a timestamps-only capture needs the work of a matched full capture')
    layers: dict[int, list[dict]] = defaultdict(list)
    for r in capture.of('layer'):
        layers[r['forward']].append(r)
    forwards = capture.generation_forwards() if generation_only else capture.of('forward')
    rows: list[Pass] = []
    previous: Pass | None = None
    for ordinal, f in enumerate(forwards):
        group = sorted(layers.get(f['forward'], ()), key=lambda r: r['layer'])
        if not group:
            raise ValueError(f"forward {f['forward']} has no layer records")
        for i, r in enumerate(group):
            join = r['t_ns'] + r['join_end']
            following = group[i + 1] if i + 1 < len(group) else None
            if following is not None:
                cycle_end = following['t_ns'] + following['join_end']
                next_pre = (following['t_ns'] + following['pre_end'] - join) * MS
            else:
                cycle_end = f['t_ns'] + f['sync_end']
                next_pre = None
            counts = work[ordinal, r['layer']] if work is not None else Counter(r['service'])
            jobs: dict[str, set[int]] = {'cpu': set(), 'helper': set(), 'primary': set()}
            for service, expert in zip(r['service'], r['executed']):
                jobs[service if service in ('cpu', 'helper') else 'primary'].add(expert)
            row = Pass(
                forward=f['forward'], ordinal=ordinal, layer=r['layer'], index=i, position=f['position'],
                final_layer=following is None,
                cpu=counts.get('cpu', 0), helper=counts.get('helper', 0), resident=counts.get('primary_resident', 0),
                ready=counts.get('staged_ready', 0), new=counts.get('staged_new', 0),
                boundary=(r['boundary_end'] - r['join_end']) * MS, plan=(r['plan_end'] - r['join_end']) * MS,
                helper_submit=_after(r, 'helper_submit_end', join), primary_submit=(r['primary_submit_end'] - r['join_end']) * MS,
                cpu_end=_after(r, 'cpu_end', join), collect_end=_after(r, 'collect_end', join), end=(r['end'] - r['join_end']) * MS,
                helper_enqueue=_after(r, 'helper_enqueue', join), helper_worker_start=_after(r, 'helper_worker_start', join),
                helper_record_end=_after(r, 'helper_record_end', join), helper_ready=_after(r, 'helper_ready_observed', join),
                helper_published=_after(r, 'helper_worker_published', join),
                next_pre_end=next_pre, cycle_ms=(cycle_end - join) * MS,
                exchanges_pending=r.get('exchanges_pending', 0), helper_lease=r.get('helper_lease', 0),
                reserved_slots=r.get('reserved_slots', 0), rows=r['rows'],
                cpu_jobs=len(jobs['cpu']) if r['service'] else counts.get('cpu', 0),
                helper_jobs=len(jobs['helper']) if r['service'] else counts.get('helper', 0),
                primary_jobs=len(jobs['primary']) if r['service'] else counts.get('primary_resident', 0) + counts.get('staged_ready', 0) + counts.get('staged_new', 0))
            if previous is not None and previous.ordinal == ordinal:
                row.prior_new, row.prior_cpu, row.prior_helper = previous.new, previous.cpu, previous.helper
            rows.append(row)
            previous = row
    return rows


def forward_budgets(capture) -> list[dict]:
    """Prefix, join-to-join cycles and tail of each generation forward. They partition it exactly."""
    layers: dict[int, list[dict]] = defaultdict(list)
    for r in capture.of('layer'):
        layers[r['forward']].append(r)
    result = []
    for ordinal, f in enumerate(capture.generation_forwards()):
        group = sorted(layers[f['forward']], key=lambda r: r['layer'])
        first_join = group[0]['t_ns'] + group[0]['join_end']
        result.append(dict(ordinal=ordinal, forward=f['forward'], position=f['position'], t_ns=f['t_ns'], forward_ms=f['end'] * MS,
                           prefix_ms=(first_join - f['t_ns']) * MS, tail_ms=(f['end'] - f['sync_end']) * MS,
                           cycles_ms=(f['t_ns'] + f['sync_end'] - first_join) * MS))
    return result


def operating_period(rows: list[Pass], fast: float = 0.45, slow: float = 0.9) -> dict[int, str]:
    """Label each forward by its median one-job helper GPU time; a measurement, not a cause."""
    samples: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r.helper == 1 and r.helper_gpu_ms is not None:
            samples[r.ordinal].append(r.helper_gpu_ms)
    labels = {}
    for ordinal, values in samples.items():
        level = statistics.median(values)
        labels[ordinal] = 'fast' if level < fast else 'slow' if level >= slow else 'middle'
    return labels


@dataclass
class Virtual:
    """A virtual change applied to one pass at a time. All factors act on observed durations."""
    cpu_factor: float = 1.0
    helper_factor: float = 1.0
    drop_copies: bool = False
    copy_ms: float = 0.55


def lanes_saved(row: Pass, change: Virtual) -> float:
    """Host-lane time this pass would save, recomputing the maximum from the common origin."""
    now = row.composed_lane_end
    begin = row.lane_begin
    cpu = begin + row.cpu_ms * change.cpu_factor if row.cpu_end is not None else begin
    helper = None
    if row.helper_enqueue is not None and row.helper_published is not None:
        helper = row.helper_enqueue + (row.helper_published - row.helper_enqueue) * change.helper_factor
    after = max(x for x in (cpu, helper, begin) if x is not None)
    return now - after


def consequence(rows: list[Pass], change: Virtual) -> float:
    """Total ms saved, per pass then summed. Only lanes and exposed copies are moved."""
    saved = sum(lanes_saved(r, change) for r in rows)
    if change.drop_copies:
        saved += sum(r.new for r in rows) * change.copy_ms
    return saved


def overlap_opportunity(rows: list[Pass], copy_ms: float = 0.553, expert_ms: float = 0.06) -> dict:
    """Upper bound if a pass's copies and primary expert kernels ran during its own host lanes.

    Assumes they could start the moment primary submission returned and that nothing else
    on the stream delays them. Per pass, what can be hidden is limited by the lane span.
    """
    hidden = 0.0
    hidden_copies_only = 0.0
    windows = 0
    for r in rows:
        lane = r.lane_end - r.lane_begin
        work = r.new * copy_ms
        hidden_copies_only += min(lane, work)
        if r.resident + r.ready + r.new:
            work += expert_ms
        hidden += min(lane, work)
        windows += work > 0
    return dict(upper_bound_ms=hidden, copies_only_ms=hidden_copies_only, windows_with_primary_work=windows,
                total_copies=sum(r.new for r in rows), copy_ms=copy_ms, expert_ms=expert_ms)


def mean_of_max_vs_max_of_means(rows: list[Pass]) -> dict:
    both = [r for r in rows if r.cpu_end is not None and r.helper_enqueue is not None and r.helper_published is not None]
    if not both:
        return dict(windows=0)
    cpu = [r.cpu_end for r in both]
    helper = [r.helper_published for r in both]
    joint = [max(a, b) for a, b in zip(cpu, helper)]
    return dict(windows=len(both), mean_cpu_completion=statistics.fmean(cpu), mean_helper_completion=statistics.fmean(helper),
                max_of_means=max(statistics.fmean(cpu), statistics.fmean(helper)), mean_of_max=statistics.fmean(joint),
                helper_last_share=sum(b > a for a, b in zip(cpu, helper)) / len(both))


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    return ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))]


def describe(values: list[float]) -> dict:
    if not values:
        return dict(n=0)
    return dict(n=len(values), mean=statistics.fmean(values), median=statistics.median(values),
                p10=_quantile(values, .1), p90=_quantile(values, .9))
