"""A candidate policy's placements and work, from recorded demand and the production policy.

The retained replay machine calls the same native policy functions as the engine, so it
supplies each pass's assignments, copies and exchanges. What it needs from time is only when
an optional transfer becomes visible. Here that is a declared order rule: time counts layer
joins, and each optional transfer occupies a fixed number of them. It reads nothing from the
control run's future: no recorded availability, hit or publication time.

Whether this rule reproduces the engine is an empirical question. `compare_work` measures it
pass by pass against a full capture, and callers keep the result beside any prediction.
"""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass, field, replace
import json
import pathlib

from cooperative_replay.capture import Capture
from cooperative_replay.execution import Estimate
from cooperative_replay.machine import Machine
from cooperative_replay.policy import PolicyConfig


@dataclass(frozen=True)
class OrderRule:
    """Layer joins occupied by each optional transfer; foreground work takes one join per pass."""
    primary_h2d: float = 0.5
    primary_d2h: float = 2.0  # starts at the next boundary's query, done by the one after
    helper_d2h: float = 0.0
    helper_h2d: float = 1.0
    helper_copy: float = 3.0
    prefix: float = 2.0  # a forward's prefix is longer than one pass, so forward-end work settles in it


class _NoPrices:
    """Machine asks the physical model for prices. Order replay never supplies milliseconds."""
    contention: dict = {}
    sha256 = 'order-rule'
    document = dict(cpu_working_set='rotating64')
    queries: list = []
    interpolation_checks: dict = {}

    def helper_groups(self, groups, *, working_set='resident'):
        from cooperative_replay.structure import helper_launches
        return [(launch, Estimate(0, 0, 0)) for launch in helper_launches(groups)]


class OrderReplay(Machine):
    """Machine with time in layer joins. One tick per pass; optional transfers per the rule."""

    def __init__(self, snapshot, config: PolicyConfig | None = None, rule: OrderRule | None = None, *, bridge=None, backing=None,
                 hold_primary: bool = False, hold_helper: bool = False):
        self.rule = rule or OrderRule()
        self.hold_primary, self.hold_helper = hold_primary, hold_helper
        self.issued_end = 0
        self.pending_flags = (False, False)
        super().__init__(snapshot, _NoPrices(), config or PolicyConfig(), bridge=bridge, backing=backing)

    def _ticks(self, kind: str, queue: str | None) -> float:
        rule = self.rule
        if kind == 'layer_pre':
            return 1.0
        if kind == 'target_prefix':
            return rule.prefix
        if queue == 'p_optional':
            return {'p_weight_h2d': rule.primary_h2d, 'p_weight_d2h': rule.primary_d2h}.get(kind, 0.0)
        if queue == 'h_exchange':
            return {'h_weight_d2h': rule.helper_d2h, 'h_weight_h2d': rule.helper_h2d, 'host_weight_copy': rule.helper_copy}.get(kind, 0.0)
        return 0.0

    def op(self, kind, resources=(), dependencies=(), **kwargs):
        label = kwargs.pop('label', f'f{self.forward}')
        queue = kwargs.pop('queue', None)
        ticks = self._ticks(kind, queue)
        # Optional transfers run on their own streams: an order clock must not let them hold a boundary.
        if queue in ('p_optional', 'h_exchange'):
            resources = ()
        # Foreground lanes share the host resource in Machine; ticks are carried only by layer_pre.
        return self.scheduler.submit(kind, Estimate(ticks, ticks, ticks), resources, dependencies, label=label, queue=queue,
                                     bytes=kwargs.get('bytes', 0))

    def admission(self, protected_layer=-1):
        before = self.counts['primary_issued']
        if not self.hold_primary:
            super().admission(protected_layer)
        if protected_layer < 0 and self.generation:
            self.issued_end += self.counts['primary_issued'] - before
        # The engine records these flags after its layer boundary has published and retained.
        self.pending_flags = (bool(self.exchanges), bool(self.helper_exchange))

    def serve_layer(self, route):
        super().serve_layer(route)
        self.layers[-1]['exchange'], self.layers[-1]['lease'] = self.pending_flags

    def helper_admission(self):
        if self.hold_helper:
            return
        super().helper_admission()


def events_from_light(capture, *, first_forward: int = 0) -> list[dict]:
    """The machine's event stream from a full light capture's own work records."""
    layers: dict[int, list[dict]] = {}
    for r in capture.of('layer'):
        layers.setdefault(r['forward'], []).append(r)
    events: list[dict] = [dict(type='fixed_workload_begin', diagnostic_primary_query=False)]
    started = False
    forward = first_forward
    for f in capture.of('forward'):
        forward += 1
        if f['generation'] and not started:
            events.append(dict(type='generation_begin'))
            started = True
        events.append(dict(type='boundary', kind='forward_begin', forward=forward, rows=f['rows'], generation=f['generation']))
        positions = list(range(f['position'], f['position'] + f['rows']))
        for r in sorted(layers.get(f['forward'], ()), key=lambda x: x['layer']):
            events.append(dict(type='routes', rows=f['rows'], layer=r['layer'], positions=positions, generation=f['generation'],
                               original=list(r['original']), executed=list(r['executed']), forward=forward))
        events.append(dict(type='boundary', kind='forward_end', forward=forward, rows=f['rows']))
    events.append(dict(type='generation_end'))
    return events


def events_from_fixture(path) -> list[dict]:
    """The machine's event stream from a fixed workload file; generation is flagged per forward."""
    tokens = pathlib.Path(path).read_text().split()
    version = int(tokens[1])  # MM3_REPLAY version budget stage cpu margin [query [hold hold [occupancy]]]
    at = 6 + ({1: 0, 2: 1, 3: 3, 4: 4}[version])
    events: list[dict] = [dict(type='fixed_workload_begin', diagnostic_primary_query=False)]
    started = False
    forward = 0
    while tokens[at] != 'END':
        if tokens[at] != 'F':
            raise ValueError('malformed fixed workload')
        rows, position, generation = int(tokens[at + 1]), int(tokens[at + 2]), bool(int(tokens[at + 3]))
        at += 4
        forward += 1
        if generation and not started:
            events.append(dict(type='generation_begin'))
            started = True
        events.append(dict(type='boundary', kind='forward_begin', forward=forward, rows=rows, generation=generation))
        positions = list(range(position, position + rows))
        while tokens[at] == 'L':
            layer = int(tokens[at + 1])
            at += 2
            original = [int(x) for x in tokens[at:at + 4 * rows]]
            at += 4 * rows
            executed = [int(x) for x in tokens[at:at + 4 * rows]]
            at += 4 * rows
            events.append(dict(type='routes', rows=rows, layer=layer, positions=positions, generation=generation,
                               original=original, executed=executed, forward=forward))
        events.append(dict(type='boundary', kind='forward_end', forward=forward, rows=rows))
    events.append(dict(type='generation_end'))
    return events


def renumber(events: list[dict], first_forward: int) -> list[dict]:
    out = []
    for e in events:
        if 'forward' in e:
            e = {**e, 'forward': e['forward'] + first_forward}
        out.append(e)
    return out


def replay_work(snapshot: dict, events: list[dict], config: PolicyConfig | None = None, rule: OrderRule | None = None, *, bridge=None,
                backing=None, hold_primary: bool = False, hold_helper: bool = False) -> dict:
    """Runs the production policy over recorded demand. Returns per-pass work and residency actions."""
    machine = OrderReplay(snapshot, config, rule, bridge=bridge, backing=backing, hold_primary=hold_primary, hold_helper=hold_helper)
    capture = Capture(pathlib.Path('order-replay'), 'order-replay', 'order-replay', renumber(events, snapshot['forward']), snapshot, {})
    result = machine.replay([capture], assignments='policy')
    passes = [dict(forward=x['forward'], layer=x['layer'], rows=x['rows'], generation=x['generation'],
                   cpu=len(x['cpu']), helper=len(x['helper']), primary=len(x['primary']), new=len(x['staged']),
                   exchange=x.get('exchange', False), lease=x.get('lease', False),
                   cpu_jobs=len(x['cpu']), cpu_columns=sum(x['uses'][e] for e in x['cpu']), helper_jobs=len(x['helper']), helper_columns=sum(x['uses'][e] for e in x['helper']),
                   primary_jobs=len(x['primary'])) for x in result['layers']]
    return dict(passes=passes, actions=result['actions'], counts=result['counts'], pending_primary=result['pending_primary'],
                pending_helper=result['pending_helper'], ticks=machine.scheduler.now, rule=rule or OrderRule(), issued_end=machine.issued_end)


def recorded_work(capture) -> list[dict]:
    """The same per-pass work as observed in a full capture, generation and prefill."""
    forwards = {f['forward']: f for f in capture.of('forward')}
    rows = []
    for r in capture.of('layer'):
        f = forwards.get(r['forward'])
        if f is None:
            continue
        counts = Counter(r['service'])
        rows.append(dict(forward=r['forward'], layer=r['layer'], rows=r['rows'], generation=f['generation'],
                         exchange=bool(r['exchanges_pending']), lease=bool(r['helper_lease']),
                         cpu=counts.get('cpu', 0), helper=counts.get('helper', 0),
                         primary=counts.get('primary_resident', 0) + counts.get('staged_ready', 0) + counts.get('staged_new', 0),
                         new=counts.get('staged_new', 0)))
    return rows


def compare_work(predicted: dict, capture) -> dict:
    """Pass-by-pass agreement of replayed work with a capture of the same fixed work."""
    observed = recorded_work(capture)
    rows = [p for p in predicted['passes']]
    if len(rows) != len(observed):
        raise ValueError(f'replayed {len(rows)} passes against {len(observed)} recorded')
    result: dict = {}
    for scope, keep in (('generation', True), ('prefill', False)):
        pairs = [(a, b) for a, b in zip(rows, observed) if b['generation'] == keep]
        if not pairs:
            continue
        same = sum(all(a[k] == b[k] for k in ('cpu', 'helper', 'primary', 'new')) for a, b in pairs)
        result[scope] = dict(
            passes=len(pairs), identical_passes=same, identical_share=same / len(pairs),
            predicted=dict(cpu=sum(a['cpu'] for a, _ in pairs), helper=sum(a['helper'] for a, _ in pairs),
                           primary=sum(a['primary'] for a, _ in pairs), new=sum(a['new'] for a, _ in pairs),
                           exchange_windows=sum(a['exchange'] for a, _ in pairs), lease_windows=sum(a['lease'] for a, _ in pairs)),
            observed=dict(cpu=sum(b['cpu'] for _, b in pairs), helper=sum(b['helper'] for _, b in pairs),
                          primary=sum(b['primary'] for _, b in pairs), new=sum(b['new'] for _, b in pairs),
                          exchange_windows=sum(b['exchange'] for _, b in pairs), lease_windows=sum(b['lease'] for _, b in pairs)))
    counts = predicted['counts']
    result['exchanges'] = dict(predicted=dict(primary_issued=counts.get('primary_issued', 0), primary_published=counts.get('primary_published', 0),
                                              helper_issued=counts.get('helper_issued', 0)),
                               observed=dict(primary_issued=len(capture.of('primary_issue')), primary_published=len(capture.of('primary_publish')),
                                             helper_issued=sum(r['phase'] == 'lease' for r in capture.of('helper_event'))))
    return result


def transfer_ticks(capture) -> dict:
    """Layer joins between an optional transfer's issue and its visible steps, in a full capture."""
    records = capture.records
    joins = 0
    issues: dict = {}
    helper_stages: dict = {}
    primary, lease_to_withdrawn, withdrawn_to_published, published_to_retired = [], [], [], []
    for r in records:
        if r['type'] == 'layer':
            joins += 1
        elif r['type'] == 'primary_issue':
            issues[r['incoming'], r['victim']] = joins
        elif r['type'] == 'primary_publish':
            at = issues.pop((r['incoming'], r['victim']), None)
            if at is not None:
                primary.append(joins - at)
        elif r['type'] == 'helper_event':
            key = r['incoming'], r['victim']
            if r['phase'] == 'lease':
                helper_stages[key] = dict(lease=joins)
            elif key in helper_stages:
                stages = helper_stages[key]
                if r['phase'] == 'withdrawn':
                    stages['withdrawn'] = joins
                    lease_to_withdrawn.append(joins - stages['lease'])
                elif r['phase'] == 'published' and 'withdrawn' in stages:
                    stages['published'] = joins
                    withdrawn_to_published.append(joins - stages['withdrawn'])
                elif r['phase'] == 'retired' and 'published' in stages:
                    published_to_retired.append(joins - stages['published'])
    return dict(primary=Counter(primary), helper_lease_to_withdrawn=Counter(lease_to_withdrawn),
                helper_withdrawn_to_published=Counter(withdrawn_to_published), helper_published_to_retired=Counter(published_to_retired))


def load_snapshot(origin_path) -> dict:
    return json.loads(pathlib.Path(origin_path).read_text())['replay_snapshot']
