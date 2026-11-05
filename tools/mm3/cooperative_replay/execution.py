"""Independent discrete-event execution; no policy prices or recorded launch times.

Durations are isolated measured service. Optional measured pair slowdowns model
shared host/transfer pressure while operations overlap. Queue order and resource
dependencies are explicit. No interference surcharge is added to completion time.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from collections import defaultdict
import math


@dataclass(frozen=True)
class Estimate:
    ms: float
    low: float
    high: float
    evidence: tuple[str, ...] = ()
    qualified: bool = True

    def __post_init__(self):
        if not all(math.isfinite(v) and v >= 0 for v in (self.ms, self.low, self.high)) or not self.low <= self.ms <= self.high:
            raise ValueError('invalid physical service range')


@dataclass
class Operation:
    id: int
    kind: str
    work: Estimate
    resources: tuple[str, ...]
    dependencies: tuple[int, ...]
    submitted: float
    label: str
    bytes: int = 0
    remaining: float = 0
    start: float | None = None
    finish: float | None = None
    queue_ms: float = 0
    # Dynamic pressure is integrated over active intervals, not charged twice.
    pressure_ms: float = 0
    unknown_contention_ms: float = 0


class Scheduler:
    def __init__(self, contention=None, variant='central', max_operations=2_000_000):
        self.now = 0.0
        self.ops: list[Operation] = []
        self.active: set[int] = set()
        self.waiting: set[int] = set()
        self.last_queue: dict[str, int] = {}
        self.contention = contention or {}
        self.variant = variant
        self.max_operations = max_operations
        self.unqualified_pairs: set[tuple[str, str]] = set()
        self.unqualified_exposure_ms=defaultdict(float)

    def submit(self, kind, work, resources=(), dependencies=(), *, label='', bytes=0, queue=None):
        if len(self.ops) >= self.max_operations: raise ValueError('bounded operation limit exceeded')
        dependencies = set(dependencies)
        if queue in self.last_queue: dependencies.add(self.last_queue[queue])
        if any(type(d) is not int or d < 0 or d >= len(self.ops) for d in dependencies):
            raise ValueError('dependency must name previously issued work')
        if len(set(resources)) != len(resources): raise ValueError('duplicate resource')
        estimate = work.ms if self.variant == 'central' else getattr(work, self.variant)
        op = Operation(len(self.ops), kind, work, tuple(resources), tuple(sorted(dependencies)),
                       self.now, label, bytes=bytes, remaining=estimate)
        self.ops.append(op);self.waiting.add(op.id)
        if queue is not None: self.last_queue[queue] = op.id
        return op.id

    def done(self, op): return self.ops[op].finish is not None

    def _start(self):
        busy = {r for i in self.active for r in self.ops[i].resources}
        for i in sorted(self.waiting):
            op = self.ops[i]
            if any(not self.done(d) for d in op.dependencies) or busy.intersection(op.resources): continue
            op.start = self.now
            ready = max([op.submitted, *(self.ops[d].finish for d in op.dependencies)])
            op.queue_ms = self.now-ready
            busy.update(op.resources);self.waiting.remove(i);self.active.add(i)

    def _rate(self, i):
        slowdown = 1.0
        for j in self.active:
            if i == j: continue
            a, b = self.ops[i].kind, self.ops[j].kind
            pair = (a, b)
            entry = self.contention.get(pair)
            weight_copies={'p_weight_h2d','p_weight_d2h','h_weight_h2d','h_weight_d2h','host_weight_copy'}
            pressure_sources=weight_copies|{'cpu','primary','helper'}
            if entry is None and a in pressure_sources and b in pressure_sources and (a in weight_copies or b in weight_copies):
                self.unqualified_pairs.add(pair)
                self.active_unqualified.add(pair)
                # Point diagnostic assumes no additional slowdown. No arbitrary
                # multiplier supplies a false uncertainty bound. Exposure is
                # reported separately and blocks a supported ranking.
                entry=Estimate(1,1,1,('unmeasured-contention',),False)
            if entry is not None:
                factor = entry.ms if self.variant == 'central' else getattr(entry, self.variant)
                if factor < 1: raise ValueError('contention slowdown below one')
                # Pair factors combine conservatively by maximum. More than two
                # competing kinds is explicitly unqualified until measured.
                slowdown = max(slowdown, factor)
                if not entry.qualified:self.unqualified_pairs.add(pair);self.active_unqualified.add(pair)
        competing = [self.ops[j].kind for j in self.active if j != i and
                     ((self.ops[i].kind, self.ops[j].kind) in self.contention or
                      (self.ops[i].kind,self.ops[j].kind) in self.active_unqualified)]
        if len(competing) > 1:
            self.unqualified_pairs.add((self.ops[i].kind, 'multiple'));self.active_unqualified.add((self.ops[i].kind,'multiple'))
        return 1.0/slowdown

    def step(self):
        self._start()
        if not self.active:
            if self.waiting: raise ValueError('dependency/resource deadlock')
            return False
        self.active_unqualified=set()
        rates = {i:self._rate(i) for i in self.active}
        dt = min(self.ops[i].remaining/rates[i] for i in self.active)
        if not math.isfinite(dt) or dt < -1e-9: raise ValueError('invalid scheduler progress')
        dt = max(dt, 0)
        self.now += dt
        for pair in self.active_unqualified:self.unqualified_exposure_ms[pair]+=dt
        finished = []
        for i in self.active:
            op = self.ops[i];op.remaining = max(0.0, op.remaining-dt*rates[i])
            op.pressure_ms += dt*(1-rates[i])
            if any(a==op.kind for a,_ in self.active_unqualified):op.unknown_contention_ms+=dt
            if op.remaining <= 1e-10:
                op.finish = self.now;finished.append(i)
        for i in finished: self.active.remove(i)
        if not finished: raise ValueError('scheduler made no progress')
        return True

    def until(self, operation):
        while not self.done(operation):
            if not self.step(): raise ValueError('missing completion')
        return self.now

    def drain(self):
        while self.step(): pass
        return self.now

    def join(self, dependencies, label='join'):
        op = self.submit('boundary', Estimate(0,0,0), dependencies=dependencies, label=label)
        self.until(op)
        return op

    def records(self):
        return [dict(id=o.id, kind=o.kind, label=o.label, start_ms=o.start, finish_ms=o.finish,
                     submit_ms=o.submitted, queue_ms=o.queue_ms, pressure_ms=o.pressure_ms,
                     dependency_delay_ms=max([o.submitted,*(self.ops[d].finish or o.submitted for d in o.dependencies)])-o.submitted,
                     predicted_service_ms=o.finish-o.start if o.finish is not None else None,
                     unknown_contention_exposure_ms=o.unknown_contention_ms,
                     isolated_ms=o.work.ms, dependencies=o.dependencies, resources=o.resources,
                     bytes=o.bytes, evidence=o.work.evidence, qualified=o.work.qualified) for o in self.ops]


def fixed_schedule_witness(operations, shortened):
    """Retain an earlier feasible schedule for monotone-cost diagnostics.

    Releases/resource reservations stay fixed at the previous simulation's
    starts/finishes; only selected service upper bounds shrink. The witness is
    conservative, not an alternate online policy and not a reason to wait in
    inference. It prevents an optimizer from discarding a feasible incumbent
    merely because rescheduling after removing D2H produced a worse heuristic.
    """
    if any(o.start is None or o.finish is None for o in operations):
        raise ValueError('fixed schedule must be complete')
    finish = {}
    for o in operations:
        duration = o.finish-o.start
        new = shortened.get(o.id, duration)
        if not math.isfinite(new) or not 0 <= new <= duration+1e-9:
            raise ValueError('monotone witness requires nonincreasing durations')
        if any(finish[d] > o.start+1e-8 for d in o.dependencies):
            raise ValueError('previous schedule violated dependency')
        finish[o.id] = o.start+new
    return max(finish.values(), default=0.0)
