"""A candidate policy's placements and work, from recorded demand and the production policy.

The retained replay machine calls the same native policy functions as the engine, so it
supplies each pass's assignments, copies and exchanges. What it needs from time is only when
an optional transfer becomes visible. Here that is a declared order rule: a clock that counts
layer joins, or milliseconds per pass by window width (todays_rule), and a fixed length for each
optional transfer. The machine's decisions read no recorded availability, hit or publication
time.

Two inputs the engine has and a light capture lacks are reconstructed beforehand from recorded
data, so they do read the control run's records:
  - the early copy's prediction (the next layer's router on this layer's input), from what the
    next pass and later records show of the copy (events_from_light(early='capture')). Like the
    recorded routing it is fixed by the control's trajectory, and a candidate replays it
    unchanged; it is exact only where the candidate's state equals the control's;
  - each layer's resident order at a request's start, which decides ties between equal victim
    scores, from the earlier requests of the same load (restored_primary_order).

Today's base (engine at 72ed95a) is replayed with
    snapshot = with_primary_order(origin, restored_primary_order(origin, earlier_captures))
    replay_work(snapshot, events_from_light(capture, early='capture'), config,
                todays_rule(speculative), early_copy=2)

Whether this reproduces the engine is an empirical question. `compare_work` measures it
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
    """When an optional transfer becomes visible. Units are layer joins (one per pass) unless `cycle` is given; then they
    are milliseconds and a pass of r rows lasts cycle[r] (join to join), so a transfer spans fewer passes in wide windows.

    In milliseconds, a forward is: its prefix (begin to the first layer's join, prefix[r]), one cycle per layer, the last
    layer's cycle before the forward-end boundary, then `gap` to the next forward's begin (sampling; drafting with
    speculation). Transfers: a primary exchange's copies (primary_h2d unless it reuses a frame, then primary_d2h); a helper
    lease's preparation (helper_copy, then helper_d2h), upload (helper_h2d) and recycling (helper_copy); an early copy."""
    primary_h2d: float = 0.5
    primary_d2h: float = 2.0  # starts at the next boundary's query, done by the one after
    helper_d2h: float = 0.0
    helper_h2d: float = 1.0
    helper_copy: float = 3.0
    prefix: float = 2.0  # a forward's prefix is longer than one pass, so forward-end work settles in it
    early_copy: float = 1.5  # an early copy is still in flight at the next layer's join and seen done at the one after
    gap: float = 0.0  # between a forward's end and the next one's begin (sampling; drafting with speculation)
    cycle: dict = field(default_factory=dict)       # rows -> ms; empty: the clock counts joins
    prefix_ms: dict = field(default_factory=dict)   # rows -> ms, with `cycle`

    def by_rows(self, table: dict, rows: int) -> float:
        if rows in table:
            return table[rows]
        return table[min(table, key=lambda k: (abs(k - rows), k))]


def todays_rule(speculative: bool) -> OrderRule:
    """The millisecond clock of today's base (engine at 72ed95a: early_copy 2, prep_graph, helper_fast), measured on the
    light captures of window-passes-20261109: mean join-to-join cycle and forward prefix by rows; the time from a forward's
    end to the next one's begin (0.3 ms at T=1, 6.3 ms with drafting before each verify window); and the transfers' lengths
    from the intervals between their recorded steps (helper lease to withdrawal 6.3-6.6 ms, withdrawal to publication
    5.5-6.1, publication to retirement 3.7-6.3; a reused frame's exchange published at the next join; an early copy in
    flight at the next join on 95% of used copies and done by the one after). One clock serves every policy: it is not
    refitted to a candidate's work."""
    if speculative:
        cycle = {1: 0.70, 2: 1.22, 3: 1.58, 4: 1.97, 5: 2.23, 6: 2.81, 7: 3.30, 8: 3.24}
        prefix = {1: 1.37, 2: 1.54, 3: 1.51, 4: 1.60, 5: 1.79, 6: 1.76, 7: 1.92, 8: 2.09}
        gap = 6.3
    else:
        cycle, prefix, gap = {1: 0.74}, {1: 1.70}, 0.3
    return OrderRule(primary_h2d=0.5, primary_d2h=0.35, helper_copy=4.0, helper_d2h=1.5, helper_h2d=5.2,
                     early_copy=1.0, gap=gap, cycle=cycle, prefix_ms=prefix)


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
    """Machine on the rule's clock: one cycle per pass (a join, or its milliseconds by rows); optional transfers per the
    rule. Adds the engine's early copy (early_copy mode) on one-row generation passes."""

    def __init__(self, snapshot, config: PolicyConfig | None = None, rule: OrderRule | None = None, *, bridge=None, backing=None,
                 hold_primary: bool = False, hold_helper: bool = False, routing_bonus: float | None = None, early_copy: int = 0):
        self.rule = rule or OrderRule()
        self.routing_bonus = routing_bonus  # re-route from recorded score triplets under this bonus; None keeps the recorded picks
        self.reroute_checked = self.reroute_same = 0
        # Generation-only routing fidelity under the bonus: picks moved off the router's own top four, and router mass kept.
        self.reroute_picks = self.reroute_moved = 0
        self.reroute_mass_original = self.reroute_mass_executed = 0.0
        self.hold_primary, self.hold_helper = hold_primary, hold_helper
        self.issued_end = 0
        self.pending_flags = (False, False)
        # The engine's early_copy (mm3_native_prefetch.inc): 0 off; 1 only when the layer queued no demand copy; 2 also
        # behind its one demand copy. Each one-row generation route may carry 'early', the next layer's predicted top four.
        if early_copy not in (0, 1, 2):
            raise ValueError('early_copy is 0, 1 or 2')
        self.early_copy = early_copy
        self.early = Counter()
        self.first_pass = False  # with a millisecond clock, a forward's first join ends its prefix, not a cycle
        self.poll_at_join = False
        super().__init__(snapshot, _NoPrices(), config or PolicyConfig(), bridge=bridge, backing=backing)

    def _ticks(self, kind: str, queue: str | None, rows: int = 1) -> float:
        rule = self.rule
        timed = bool(rule.cycle)
        if kind == 'layer_pre':
            if not timed:
                return 1.0
            first, self.first_pass = self.first_pass, False
            return 0.0 if first else rule.by_rows(rule.cycle, rows)
        if kind == 'target_prefix':
            return rule.by_rows(rule.prefix_ms, rows) if timed else rule.prefix
        if kind == 'target_tail':
            return rule.by_rows(rule.cycle, rows) if timed else 0.0
        if kind == 'forward_gap':
            return rule.gap
        if kind == 'early_h2d':
            return rule.early_copy
        if queue == 'p_optional':
            return {'p_weight_h2d': rule.primary_h2d, 'p_weight_d2h': rule.primary_d2h}.get(kind, 0.0)
        if queue == 'h_exchange':
            return {'h_weight_d2h': rule.helper_d2h, 'h_weight_h2d': rule.helper_h2d, 'host_weight_copy': rule.helper_copy}.get(kind, 0.0)
        return 0.0

    # An early-copied frame is owned at once, but until its copy is seen complete at a layer join it is no one else's:
    # not a demand-copy victim nor a free frame for the planner, not an exchange landing or reused source, and its expert's
    # exchange or helper lease cannot publish (slot_early_busy in the engine).
    def busy(self, frame):
        return frame.early is not None or super().busy(frame)

    def source_busy(self, frame):
        return frame.early is not None or super().source_busy(frame)

    def publish(self, protected_layer=-1, *, helper=True):
        if protected_layer >= 0:
            if self.poll_at_join:
                # The engine polls a helper lease when it submits a layer's preparation (window_pre), at the end of the
                # previous layer's host work, just before this join; Machine polls at the previous join. With a
                # millisecond clock the difference is a whole cycle, so the poll is made here.
                self.poll_at_join = False
                super().helper_progress()
            # early_copy_retire: a query at each layer join, before the join's publication and retention.
            for f in self.slots:
                if f.early is not None and self.scheduler.done(f.early):
                    f.early = None
        super().publish(protected_layer, helper=helper)

    def helper_progress(self):
        if not self.poll_at_join:
            super().helper_progress()

    def between_forwards(self, event):
        self.first_pass = True
        if self.rule.gap:
            self.scheduler.until(self.op('forward_gap', ('p_compute', 'host'), queue='p_main'))

    def issue_early_copy(self, route) -> None:
        """early_copy_issue: after the layer's plan, the next layer's first predicted HOST expert into the least recently
        used free frame, on the copy stream; the frame is owned at once, so the next layer's routing and plan see it ready."""
        x = self.layers[-1]
        li = route['layer']
        self.early['eligible'] += 1
        if len(x['staged']) > (1 if self.early_copy >= 2 else 0) or self.exchanges:
            self.early['blocked'] += 1
            return
        # The first predicted expert that is HOST only: not resident, not the 3090's, not in a lease or exchange, in no frame.
        host_only = lambda key: not (key in self.primary or key in self.helper or key not in self.host or self.leased(key) or
                                     any(f.owner == key for f in self.slots))
        pick = None
        for e in route['early']:
            if e == UNUSED:
                pick = UNUSED
                break
            if not 0 <= e < 128:
                return  # a malformed prediction issues nothing
            key = (li + 1) * 1024 + e
            if host_only(key):
                pick = key
                break
        if pick is None:
            self.early['no_candidate'] += 1
            return
        readers = x['done_op']
        free = [i for i, f in enumerate(self.slots) if f.reader != readers and not f.reserved and f.early is None]
        if not free:
            self.early['no_frame'] += 1
            return
        i = min(free, key=lambda s: (self.slots[s].used, s))
        if pick == UNUSED:
            # A copy the next pass will not read. A later record may name what its frame held; otherwise it holds an
            # identity no pass asks for (expert 1023 of the next layer).
            named = route.get('early_unread', {}).get(i)
            pick = (li + 1) * 1024 + named if named is not None and host_only((li + 1) * 1024 + named) else (li + 1) * 1024 + 1023
            self.early['unread'] += 1
            self.early['unread_named'] += pick % 1024 != 1023
        frame = self.slots[i]
        frame.owner, frame.used = pick, self.forward
        # Its own clock, not a queue: copies a layer apart do not wait for each other (each takes about one layer cycle).
        frame.early = self.op('early_h2d', (), bytes=self.record_bytes)
        x['early_copy'] = pick
        self.early['issued'] += 1
        self.counts['early_copies'] += 1

    def op(self, kind, resources=(), dependencies=(), **kwargs):
        label = kwargs.pop('label', f'f{self.forward}')
        queue = kwargs.pop('queue', None)
        ticks = self._ticks(kind, queue, kwargs.get('rows', 1))
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

    def ready_for_routing(self, layer: int, expert: int) -> bool:
        key = layer * 1024 + expert
        return key in self.helper or key in self.primary or any(f.owner == key and not f.reserved for f in self.slots)

    def serve_layer(self, route):
        self.poll_at_join = bool(self.rule.cycle)
        if self.routing_bonus is not None and route.get('scores'):
            li = route['layer']
            executed = reroute(route['scores'], route['rows'], self.routing_bonus, lambda e: self.ready_for_routing(li, e))
            self.reroute_checked += 1
            self.reroute_same += sorted(executed) == sorted(route['executed'])
            if route.get('generation'):
                scores, original = route['scores'], route['original']
                for row in range(route['rows']):
                    own, picked = set(original[4 * row:4 * row + 4]), executed[4 * row:4 * row + 4]
                    self.reroute_picks += 4
                    self.reroute_moved += sum(e not in own for e in picked)
                    self.reroute_mass_original += sum(scores[row * 384 + 3 * e] for e in own)
                    self.reroute_mass_executed += sum(scores[row * 384 + 3 * e] for e in picked)
            route = {**route, 'executed': executed}
        super().serve_layer(route)
        self.layers[-1]['exchange'], self.layers[-1]['lease'] = self.pending_flags
        if self.early_copy and self.generation and route['rows'] == 1 and route.get('early') is not None:
            self.issue_early_copy(route)

    def helper_admission(self):
        if self.hold_helper:
            return
        super().helper_admission()


UNUSED = -1  # in an early-copy prediction: an expert of the next layer that its pass does not use (which one is unrecorded)


def _distinct(ids) -> list[int]:
    return list(dict.fromkeys(ids))


def recorded_early_prediction(here: dict, after: dict, mode: int = 2) -> list[int]:
    """The early copy's prediction for the layer after `here`, as far as a light capture shows it (both one-row records).

    A light capture keeps neither the prediction (the next layer's router on this layer's input) nor the copy, but its
    effect: an early copy the next pass reads is served there from a frame ('staged_ready'), and one it does not read is
    still in flight at the next join, so its frame is neither free, reserved nor read (8 - free - reserved - ready is 1).
    Where the engine tried a copy (this pass queued at most `mode - 1` demand copies and no exchange was pending), the
    prediction is the next pass's ready experts, in pick order, and UNUSED if a frame was taken by an unread copy: the
    replay copies the first that is HOST, which is the engine's copy when the replay holds the engine's frames. Where it did
    not try, nothing is shown and the proxy is the next pass's executed picks."""
    tried = sum(s == 'staged_new' for s in here['service']) <= (1 if mode >= 2 else 0) and not here['exchanges_pending']
    if not tried:
        return _distinct(after['executed'])
    ready = _distinct(e for e, s in zip(after['executed'], after['service']) if s == 'staged_ready')
    unread = after['total_slots'] - after['free_slots'] - after['reserved_slots'] - len(ready)
    return ready + ([UNUSED] if unread >= 1 else [])


def unread_identities(records: list[dict], start: int, layer: int, unused: set, horizon: int = 60) -> dict[int, int]:
    """Which expert an unread early copy for `layer` holds, by frame: the first later record that names a frame's owner (a
    demand copy's displaced expert, or the incoming expert of an exchange that reuses the frame) shows it, if that owner is
    an expert of `layer` its pass did not use. A frame first taken by anything else stays unknown.

    Early copies are not recorded, so a frame may have been refilled by one before a record names it. Only one-row passes
    make them, nearly one a pass over eight frames, so the search stops after `horizon` one-row passes: a forward's worth,
    which reaches the next forward's first join, where exchanges retain such frames. On window-passes-20261109, 60 passes
    gave 99.7% of T=1 passes the engine's work against 98.7% for an unbounded search and 97.8% for none."""
    seen: dict[int, int | None] = {}
    passes = 0
    for r in records[start:]:
        if r['type'] == 'layer':
            passes += r['rows'] == 1
            if passes > horizon:
                break
            continue
        if r['type'] == 'demand_copy':
            slot, owner = r['slot'], r['displaced']
        elif r['type'] == 'primary_issue':
            slot, owner = r['landing'], (r['incoming'] if r['reuse'] else None)
        else:
            continue
        if slot not in seen:
            seen[slot] = owner
    return {s: k % 1024 for s, k in seen.items() if k is not None and k >= 0 and k // 1024 == layer and k % 1024 not in unused}


def events_from_light(capture, *, first_forward: int = 0, early: str | None = None, early_mode: int = 2) -> list[dict]:
    """The machine's event stream from a full light capture's own work records.

    early attaches to each one-row generation route followed by its forward's next layer that layer's early-copy prediction
    ('early'): 'capture' as recorded_early_prediction reconstructs it for a capture made with early_copy `early_mode`, or a
    proxy from the next layer's own picks ('executed', or the router's 'original' ranking). None attaches nothing. With
    'capture', a route whose prediction holds UNUSED also carries 'early_unread': frame -> the expert a later record shows
    that frame held (unread_identities)."""
    if early not in (None, 'capture', 'executed', 'original'):
        raise ValueError('early is None, capture, executed or original')
    index = {(r['forward'], r['layer']): i for i, r in enumerate(capture.records) if r['type'] == 'layer'}
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
        group = sorted(layers.get(f['forward'], ()), key=lambda x: x['layer'])
        for i, r in enumerate(group):
            event = dict(type='routes', rows=f['rows'], layer=r['layer'], positions=positions, generation=f['generation'],
                         original=list(r['original']), executed=list(r['executed']), forward=forward)
            after = group[i + 1] if i + 1 < len(group) else None
            if early and f['generation'] and f['rows'] == 1 and after is not None and after['layer'] == r['layer'] + 1:
                event['early'] = recorded_early_prediction(r, after, early_mode) if early == 'capture' else _distinct(after[early])
                if early == 'capture' and UNUSED in event['early']:
                    event['early_unread'] = unread_identities(capture.records, index[(r['forward'], r['layer'])] + 1,
                                                              after['layer'], set(after['executed']))
            events.append(event)
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


def events_from_detailed(path) -> list[dict]:
    """The machine's event stream from a detailed capture's routes records, keeping the router's score triplets
    (sigmoid, base, bonus-adjusted) per expert and row, so a replay can re-route under another bonus."""
    routes = []
    with open(path) as f:
        for line in f:
            if '"routes"' not in line:
                continue
            d = json.loads(line)
            r = d.get('record', {})
            if r.get('type') == 'routes':
                routes.append(r)
    events: list[dict] = [dict(type='fixed_workload_begin', diagnostic_primary_query=False)]
    started = False
    forward = 0
    current = None
    for r in routes:
        if r['forward'] != current:
            if current is not None:
                events.append(dict(type='boundary', kind='forward_end', forward=forward, rows=rows))
            current, rows, generation = r['forward'], r['rows'], bool(r['generation'])
            forward += 1
            if generation and not started:
                events.append(dict(type='generation_begin'))
                started = True
            events.append(dict(type='boundary', kind='forward_begin', forward=forward, rows=rows, generation=generation))
        events.append(dict(type='routes', rows=r['rows'], layer=r['layer'], positions=list(r['positions']), generation=bool(r['generation']),
                           original=list(r['original']), executed=list(r['executed']), forward=forward,
                           scores=r.get('all_score_triplets_sigmoid_base_prior')))
    if current is not None:
        events.append(dict(type='boundary', kind='forward_end', forward=forward, rows=rows))
    events.append(dict(type='generation_end'))
    return events


def reroute(scores: list[float], rows: int, bonus: float, ready) -> list[int]:
    """The engine's prior policy: per row, the top four of base score plus the bonus on ready experts, over all 128;
    higher score first, then lower expert id. A zero bonus keeps the original picks, as the engine bypasses the pass."""
    out = []
    for row in range(rows):
        base = scores[row * 384 + 1: (row + 1) * 384: 3]
        ranked = sorted(range(128), key=lambda e: (-(base[e] + (bonus if ready(e) else 0.0)), e))
        out.extend(ranked[:4])
    return out


def with_early_predictions(events: list[dict], source: str = 'executed') -> list[dict]:
    """For event streams without light records (fixtures, detailed captures): each one-row generation route followed by
    its forward's next layer carries 'early', a proxy for that layer's prediction from its own picks. 'executed' is the
    closer proxy (an early-copied expert gets the routing bonus, so the engine's copy is usually an executed pick);
    'original' is the router's ranking. A light capture supports the reconstruction instead (events_from_light)."""
    out = []
    for i, e in enumerate(events):
        if e['type'] == 'routes' and e['rows'] == 1 and e.get('generation'):
            nxt = events[i + 1] if i + 1 < len(events) else None
            if nxt is not None and nxt['type'] == 'routes' and nxt['forward'] == e['forward'] and nxt['layer'] == e['layer'] + 1:
                e = {**e, 'early': _distinct(nxt[source])}
        out.append(e)
    return out


def renumber(events: list[dict], first_forward: int) -> list[dict]:
    out = []
    for e in events:
        if 'forward' in e:
            e = {**e, 'forward': e['forward'] + first_forward}
        out.append(e)
    return out


def replay_work(snapshot: dict, events: list[dict], config: PolicyConfig | None = None, rule: OrderRule | None = None, *, bridge=None,
                backing=None, hold_primary: bool = False, hold_helper: bool = False, routing_bonus: float | None = None,
                early_copy: int = 0) -> dict:
    """Runs the production policy over recorded demand. Returns per-pass work and residency actions.

    routing_bonus re-routes every layer with recorded score triplets (events_from_detailed) under that bonus, against the
    replay's own residents; demand learning still sees only the original picks, as in the engine.
    early_copy is the engine's mode; it acts on one-row generation routes that carry an 'early' prediction
    (events_from_light's early, or with_early_predictions).

    Per pass: cpu, helper and primary count distinct experts (jobs); *_columns count their picks; new counts uploads."""
    machine = OrderReplay(snapshot, config, rule, bridge=bridge, backing=backing, hold_primary=hold_primary, hold_helper=hold_helper,
                          routing_bonus=routing_bonus, early_copy=early_copy)
    capture = Capture(pathlib.Path('order-replay'), 'order-replay', 'order-replay', renumber(events, snapshot['forward']), snapshot, {})
    result = machine.replay([capture], assignments='policy')
    passes = [dict(forward=x['forward'], layer=x['layer'], rows=x['rows'], generation=x['generation'],
                   cpu=len(x['cpu']), helper=len(x['helper']), primary=len(x['primary']), new=len(x['staged']),
                   exchange=x.get('exchange', False), lease=x.get('lease', False),
                   cpu_jobs=len(x['cpu']), cpu_columns=sum(x['uses'][e] for e in x['cpu']), helper_jobs=len(x['helper']), helper_columns=sum(x['uses'][e] for e in x['helper']),
                   primary_jobs=len(x['primary']), primary_columns=sum(x['uses'][e] for e in x['primary']),
                   early_copy=x.get('early_copy', -1)) for x in result['layers']]
    return dict(passes=passes, actions=result['actions'], counts=result['counts'], pending_primary=result['pending_primary'],
                pending_helper=result['pending_helper'], ticks=machine.scheduler.now, rule=rule or OrderRule(), issued_end=machine.issued_end,
                early=dict(machine.early),
                reroute=dict(checked=machine.reroute_checked, same_as_recorded=machine.reroute_same, picks=machine.reroute_picks,
                             moved=machine.reroute_moved, mass_original=machine.reroute_mass_original, mass_executed=machine.reroute_mass_executed))


def recorded_work(capture) -> list[dict]:
    """The same per-pass work as observed in a full capture, generation and prefill, in the replay's units: cpu, helper and
    primary count distinct experts (jobs), *_columns their picks, new the uploads (a staged expert's first pick)."""
    forwards = {f['forward']: f for f in capture.of('forward')}
    rows = []
    for r in capture.of('layer'):
        f = forwards.get(r['forward'])
        if f is None:
            continue
        lanes: dict[str, set] = {'cpu': set(), 'helper': set(), 'primary': set()}
        columns = Counter()
        for service, expert in zip(r['service'], r['executed']):
            lane = service if service in ('cpu', 'helper') else 'primary'
            lanes[lane].add(expert)
            columns[lane] += 1
        rows.append(dict(forward=r['forward'], layer=r['layer'], rows=r['rows'], generation=f['generation'],
                         exchange=bool(r['exchanges_pending']), lease=bool(r['helper_lease']),
                         cpu=len(lanes['cpu']), helper=len(lanes['helper']), primary=len(lanes['primary']),
                         cpu_columns=columns['cpu'], helper_columns=columns['helper'], primary_columns=columns['primary'],
                         new=sum(s == 'staged_new' for s in r['service'])))
    return rows


WORK_FIELDS = ('cpu', 'helper', 'primary', 'new')


def compare_work(predicted: dict, capture) -> dict:
    """Pass-by-pass agreement of replayed work with a capture of the same fixed work: a pass agrees when its CPU, 3090 and
    5090 jobs (distinct experts) and its uploads are all equal. Totals carry jobs and picks (columns) separately; with more
    than one row a pass's picks exceed its jobs. Agreement is also given by tenth of the request, to show drift."""
    observed = recorded_work(capture)
    rows = [p for p in predicted['passes']]
    if len(rows) != len(observed):
        raise ValueError(f'replayed {len(rows)} passes against {len(observed)} recorded')
    result: dict = {}
    for scope, keep in (('generation', True), ('prefill', False)):
        pairs = [(a, b) for a, b in zip(rows, observed) if b['generation'] == keep]
        if not pairs:
            continue
        agree = [all(a[k] == b[k] for k in WORK_FIELDS) for a, b in pairs]
        n = len(agree)
        tenths = [sum(agree[i * n // 10:(i + 1) * n // 10]) / max(1, (i + 1) * n // 10 - i * n // 10) for i in range(10)]
        totals = lambda side: {**{k: sum(p[side][k] for p in pairs) for k in WORK_FIELDS},
                               **{f'{k}_columns': sum(p[side].get(f'{k}_columns', 0) for p in pairs) for k in ('cpu', 'helper', 'primary')},
                               'exchange_windows': sum(bool(p[side]['exchange']) for p in pairs),
                               'lease_windows': sum(bool(p[side]['lease']) for p in pairs)}
        pairs = [dict(zip(('replay', 'engine'), p)) for p in pairs]
        result[scope] = dict(passes=n, identical_passes=sum(agree), identical_share=sum(agree) / n, identical_by_tenth=tenths,
                             predicted=totals('replay'), observed=totals('engine'))
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


def restored_primary_order(snapshot: dict, earlier: list) -> dict[int, list[int]]:
    """Each layer's resident order (the engine's res_ids) at the start of a request that restored the origin after the
    given earlier requests of the same load, as light captures in the order they ran.

    Restoring the origin restores its resident set, not its order. The engine's placement walks the current frames layer by
    layer in their res_ids order, puts each missing origin resident into the next frame whose expert the origin does not
    want, and rebuilds every layer's res_ids in that frame order (mm3_native_placement.inc). Before it, the restore's join
    publishes the exchanges still open at generation end. Within a request each publication moves the layer's last resident
    into the victim's place and appends the incoming expert (adaptive_publish). Victim choice keeps the first of equal
    scores in this order, and the restored scores tie often, so the order decides real choices. It needs only the earlier
    requests' recorded publications, nothing from the request being replayed."""
    origin = {l['layer']: list(l['primary']) for l in snapshot['layers']}
    layers = sorted(origin)
    desired = [(li, e) for li in layers for e in origin[li]]
    wanted = set(desired)
    current = {li: list(v) for li, v in origin.items()}
    for capture in earlier:
        published = [(r['incoming'], r['victim']) for r in capture.of('primary_publish')]
        ends = [m for m in capture.of('map') if m['phase'] == 1]
        if len(ends) != 1:
            raise ValueError('an earlier request lacks its generation-end map')
        published += [(x['incoming'], x['victim']) for x in ends[0]['exchanges']]
        for incoming, victim in published:
            row = current[victim // 1024]
            i = row.index(victim % 1024)
            row[i] = row[-1]
            row.pop()
            current[incoming // 1024].append(incoming % 1024)
        frames = [(li, e) for li in layers for e in current[li]]
        present = set(frames)
        free = 0
        for item in desired:
            if item in present:
                continue
            while frames[free] in wanted:
                free += 1
            frames[free] = item
            free += 1
        current = {li: [] for li in layers}
        for li, e in frames:
            current[li].append(e)
    if any(sorted(current[li]) != sorted(origin[li]) for li in layers):
        raise ValueError('restored resident set differs from the origin')
    return current


def with_primary_order(snapshot: dict, order: dict[int, list[int]]) -> dict:
    """A copy of the snapshot whose layers list their residents in the given order (frames follow their experts)."""
    layers = []
    for l in snapshot['layers']:
        frames = dict(l['primary_frames'])
        primary = order.get(l['layer'], l['primary'])
        if sorted(primary) != sorted(l['primary']):
            raise ValueError('order changes the resident set')
        layers.append({**l, 'primary': list(primary), 'primary_frames': [[e, frames[e]] for e in primary]})
    return {**snapshot, 'layers': layers}
