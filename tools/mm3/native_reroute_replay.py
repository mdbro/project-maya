#!/usr/bin/env python3
"""Re-routed residency replay from detailed captures (plan: docs/minimax-wsl/README.md, "The order of work from the
2026-11-10 review", the shared first step).

A detailed capture keeps, for every routed layer pass, the router's score triplets per row and expert (sigmoid, sigmoid
plus bias, and that plus the bonus where the expert was ready) and the ready masks the routing read. This tool

  - replays residency under a policy:
      P0  the recorded residency, pass by pass (the ready masks the engine routed against);
      P1  the engine's rule simulated: native demand score (0.95 per five generation rows, mm3_native_demand_units.hpp),
          the 5090's priced exchange at 8 per five rows (mm3_native_adaptive.inc), the 3090's lease at 1 per five rows
          with its rule (incoming score >= 1.8 and above 1.25 times the victim's plus 1, mm3_cooperative_policy.hpp,
          mm3_native_helper_live.inc), eight staging frames filled by the production batch-staging planner;
      P2  P1 with another cap on the 3090's exchanges;
      P3  P1 with a two-speed score (a slow and a fast average, normalized to the native score's units);
      P4  P1 with a bonus-aware score: an expert is also credited whenever it could have replaced a pick that was loaded
          (its base score within the bonus of that pick);
  - re-routes every generation pass under a bonus against that policy's ready set with the engine's rule
    (resident::route, Policy::Prior: per row the top four of float32 base + bonus on ready experts over all 128, higher
    first, ties to the lower id; a zero bonus keeps the router's own picks);
  - counts per generated token from token 128: picks moved off the router's top four, router mass kept, picks executed on
    each card's residents and on staged frames, picks that need a load, and the router's own picks not on a GPU.

What it cannot model: time. Transfers complete after fixed numbers of layer passes measured on the capture, and no pass
is priced. The token sequence is held as recorded (first order: a re-routed pick changes no later hidden state). Early
copies are replayed as recorded where the policy's state allows them (P0 has them in its masks); their prediction is not
recorded, so a simulated policy cannot choose a different early copy.

Run from the worktree root:
  python3 -I -B tools/mm3/native_reroute_replay.py extract --capture <req.jsonl> --out <dir>/<case>.npz
  python3 -I -B tools/mm3/native_reroute_replay.py run --extract <dir>/<case>.npz [...] --origin <origin.json> --out <json>
Before any CPU-heavy step the CLI waits while the quiet flag (--quiet-lock) exists.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import pathlib
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
QUIET = pathlib.Path('/mnt/wsl/transfer/minimax-wsl/runs/plan-completion-20261110/locks/quiet.lock')

E = 128          # experts per layer
K = 4            # picks per row
KEY = 1024       # the engine's key: layer * 1024 + expert

# Event kinds in an extract.
EV = dict(primary_issue=1, primary_publish=2, helper_issue=3, helper_withdraw=4, helper_publish=5, helper_retire=6,
          demand_copy=7, early_copy=8, forward_begin=9, forward_end=10)
SERVICE = {'cpu': 1, 'helper': 2, 'primary_staged': 3, 'primary_resident': 4}
PRED_OUTCOME = {'copied': 0, 'blocked': 1, 'no_candidate': 2, 'no_frame': 3, 'malformed': 4, 'mode3': 5}


def wait_quiet(lock: pathlib.Path | None = QUIET, poll: float = 20.0) -> None:
    """Block while another agent's timed test holds the quiet flag."""
    if lock is None:
        return
    announced = False
    while lock.exists():
        if not announced:
            print(f'quiet flag {lock} present; waiting', file=sys.stderr, flush=True)
            announced = True
        time.sleep(poll)


_QUIET = dict(lock=QUIET, next=0.0)


def set_quiet(lock: pathlib.Path | None) -> None:
    _QUIET['lock'], _QUIET['next'] = lock, 0.0


def quiet_tick(every: float = 30.0) -> None:
    """Inside long passes: about every 30 s, wait while the quiet flag exists (AGENT_RULES, last section)."""
    now = time.monotonic()
    if now >= _QUIET['next']:
        wait_quiet(_QUIET['lock'])
        _QUIET['next'] = time.monotonic() + every


def bits_to_mask(words) -> np.ndarray:
    """Ready words (4 x uint32, bit e % 32 of word e / 32) to bool[..., 128]."""
    w = np.asarray(words, dtype='<u4')
    return np.unpackbits(w.view(np.uint8).reshape(*w.shape[:-1], 16), axis=-1, bitorder='little').astype(bool)


def ids_to_mask(ids) -> np.ndarray:
    m = np.zeros(E, bool)
    if len(ids):
        m[np.asarray(ids, dtype=np.int64)] = True
    return m


# ----------------------------------------------------------------------------------------------------------------------
# Extraction


def extract_capture(path) -> dict:
    """Compact arrays from one detailed-capture JSONL (one request)."""
    passes = dict(forward=[], layer=[], rows=[], generation=[], position=[], row0=[], ready_primary=[], ready_helper=[],
                  res_primary=[], res_helper=[], staging=[], exchanges=[], helper_exchange=[], cpu_jobs=[], helper_jobs=[],
                  uploads=[], primary_order=[])
    rows = dict(base=[], sig=[], adjusted_ready=[], original=[], executed=[], service=[], pass_index=[])
    events = dict(kind=[], at=[], forward=[], layer=[], incoming=[], victim=[], reuse=[], early_layer=[], in_score=[],
                  vic_score=[], slot=[], credit=[], demand_rows=[], rows=[], position=[])
    preds = dict(at=[], forward=[], layer=[], target=[], outcome=[], expert=[], slot=[], demand_copies=[], exchanges=[],
                 ids=[], base=[], has_base=[])
    meta: dict = {}
    forwards = []      # (forward, phase, rows, position)
    rounds = []        # round_end: committed_tokens, offered, evaluated
    npass = 0
    pending_assign = None

    def event(kind, d, **kw):
        events['kind'].append(EV[kind]); events['at'].append(npass)
        events['forward'].append(d.get('forward', -1)); events['layer'].append(d.get('layer', -1))
        events['incoming'].append(kw.get('incoming', d.get('incoming', -1)))
        events['victim'].append(kw.get('victim', d.get('victim', -1)))
        events['reuse'].append(int(d.get('reuse', -1)))
        events['early_layer'].append(int(d.get('early_layer', -9)))
        events['in_score'].append(float(d.get('incoming_score', np.nan)))
        events['vic_score'].append(float(d.get('victim_score', np.nan)))
        events['slot'].append(int(kw.get('slot', d.get('landing_slot', d.get('slot', -1)))))
        events['credit'].append(float(d.get('admission_credit', np.nan)))
        events['demand_rows'].append(int(d.get('demand_rows', -1)))
        events['rows'].append(int(d.get('rows', -1)))
        events['position'].append(int(d.get('position', -1)))

    with open(path) as f:
        for line in f:
            d = json.loads(line)
            typ = d['type']
            r = d.get('record')
            if typ == 'residency' and isinstance(r, dict):
                rt = r['type']
                if rt == 'request':
                    meta['request'] = {k: v for k, v in r.items() if k not in ('batch_staging_calibration',)}
                    meta['batch_staging_calibration'] = r.get('batch_staging_calibration')
                elif rt == 'routes':
                    T = r['rows']
                    passes['forward'].append(r['forward']); passes['layer'].append(r['layer'])
                    passes['rows'].append(T); passes['generation'].append(bool(r['generation']))
                    passes['position'].append(min(r['positions']))
                    passes['ready_primary'].append(r['ready_primary']); passes['ready_helper'].append(r['ready_helper'])
                    passes['res_primary'].append(np.packbits(ids_to_mask(r['primary']), bitorder='little'))
                    passes['res_helper'].append(np.packbits(ids_to_mask(r['helper']), bitorder='little'))
                    passes['primary_order'].append(r['primary'])
                    st = np.full((8, 6), -1, np.int64)
                    for i, s in enumerate(r['staging'][:8]):
                        st[i] = s[:6]
                    passes['staging'].append(st)
                    passes['exchanges'].append([x[:2] for x in r['exchanges']])
                    passes['helper_exchange'].append(r['helper_exchange'])
                    if r['generation']:
                        passes['row0'].append(len(rows['base']))
                        t = np.asarray(r['all_score_triplets_sigmoid_base_prior'], dtype=np.float64).astype(np.float32)
                        t = t.reshape(T, E, 3)
                        for row in range(T):
                            rows['sig'].append(t[row, :, 0]); rows['base'].append(t[row, :, 1])
                            rows['adjusted_ready'].append(t[row, :, 2] != t[row, :, 1])
                            rows['original'].append(r['original'][K * row:K * row + K])
                            rows['executed'].append(r['executed'][K * row:K * row + K])
                            rows['service'].append([0] * K)
                            rows['pass_index'].append(npass)
                    else:
                        passes['row0'].append(-1)
                    for k in ('cpu_jobs', 'helper_jobs', 'uploads'):
                        passes[k].append(-1)
                    npass += 1
                elif rt == 'service' and npass:
                    passes['cpu_jobs'][-1] = r.get('cpu_jobs', -1)
                    passes['helper_jobs'][-1] = r.get('helper_jobs', -1)
                    passes['uploads'][-1] = r.get('demand_uploads', -1)
                continue
            if typ == 'expert_assignment':
                # Follows its layer's routes record.
                if npass and passes['generation'][-1] and passes['forward'][-1] == d['forward'] and passes['layer'][-1] == d['layer']:
                    row0 = passes['row0'][-1]
                    for a in d['assignments']:
                        rows['service'][row0 + a['row']][a['pick']] = SERVICE.get(a['service'], 9)
                continue
            if typ == 'primary_exchange_issue':
                event('primary_issue', d)
            elif typ == 'primary_exchange_publish':
                event('primary_publish', d)
            elif typ == 'helper_exchange_issue':
                event('helper_issue', d)
            elif typ == 'helper_exchange_withdraw':
                event('helper_withdraw', d)
            elif typ == 'helper_exchange_publish':
                event('helper_publish', d)
            elif typ == 'helper_exchange_retire':
                event('helper_retire', d)
            elif typ == 'demand_copy':
                event('demand_copy', d, incoming=d['layer'] * KEY + d['expert'], victim=d['displaced'])
            elif typ == 'early_copy':
                event('early_copy', d, incoming=d['layer'] * KEY + d['expert'], victim=d.get('displaced', -1))
            elif typ == 'early_copy_prediction':
                # (e1561fc and later) the early copy's whole prediction for the next layer and what became of it
                preds['at'].append(npass); preds['forward'].append(d.get('forward', -1)); preds['layer'].append(d.get('layer', -1))
                preds['target'].append(d['target_layer']); preds['outcome'].append(PRED_OUTCOME.get(d['outcome'], 9))
                preds['expert'].append(d.get('expert', -1)); preds['slot'].append(d.get('slot', -1))
                preds['demand_copies'].append(d.get('demand_copies', -1)); preds['exchanges'].append(d.get('exchanges_pending', -1))
                preds['ids'].append(list(d['predicted'])[:K])
                b = d.get('base')
                preds['has_base'].append(b is not None)
                preds['base'].append(np.asarray(b if b is not None else [np.nan] * E, np.float64).astype(np.float32))
            elif typ == 'target_forward_begin':
                event('forward_begin', d)
                forwards.append((d.get('phase', ''), d.get('rows', -1), d.get('position', -1)))
            elif typ == 'target_forward_end':
                event('forward_end', d)
            elif typ == 'round_end':
                rounds.append((d.get('round', -1), d.get('forward', -1), d.get('committed_tokens', -1), d.get('offered', -1),
                               d.get('evaluated', -1)))
            elif typ == 'request_config':
                meta['request_config'] = {k: v for k, v in d.items() if k not in ('schema',)}
            elif typ == 'generation_begin':
                meta['generation_begin'] = dict(forward=d['forward'], state_position=d.get('state_position'))
            elif typ == 'request_end':
                meta['request_end'] = {k: d.get(k) for k in ('finish', 'emitted', 'committed_tokens', 'drafts_offered', 'drafts_accepted')}
            elif typ == 'capture_footer':
                meta['footer'] = {k: d.get(k) for k in ('recorded_events', 'dropped_events', 'detail_dropped_events')}
    P = npass
    out = dict(
        pass_forward=np.asarray(passes['forward'], np.int32), pass_layer=np.asarray(passes['layer'], np.int16),
        pass_rows=np.asarray(passes['rows'], np.int16), pass_generation=np.asarray(passes['generation'], bool),
        pass_position=np.asarray(passes['position'], np.int32), pass_row0=np.asarray(passes['row0'], np.int32),
        ready_primary=np.asarray(passes['ready_primary'], np.uint32).reshape(P, 4),
        ready_helper=np.asarray(passes['ready_helper'], np.uint32).reshape(P, 4),
        res_primary=np.asarray(passes['res_primary'], np.uint8).reshape(P, 16),
        res_helper=np.asarray(passes['res_helper'], np.uint8).reshape(P, 16),
        staging=np.asarray(passes['staging'], np.int64).reshape(P, 8, 6),
        cpu_jobs=np.asarray(passes['cpu_jobs'], np.int16), helper_jobs=np.asarray(passes['helper_jobs'], np.int16),
        uploads=np.asarray(passes['uploads'], np.int16),
        row_base=np.asarray(rows['base'], np.float32).reshape(-1, E), row_sig=np.asarray(rows['sig'], np.float32).reshape(-1, E),
        row_adjusted_ready=np.asarray(rows['adjusted_ready'], bool).reshape(-1, E),
        row_original=np.asarray(rows['original'], np.int16).reshape(-1, K),
        row_executed=np.asarray(rows['executed'], np.int16).reshape(-1, K),
        row_service=np.asarray(rows['service'], np.int8).reshape(-1, K), row_pass=np.asarray(rows['pass_index'], np.int32),
        **{f'ev_{k}': np.asarray(v, np.float64 if k in ('in_score', 'vic_score', 'credit') else np.int64) for k, v in events.items()},
        pred_at=np.asarray(preds['at'], np.int32), pred_forward=np.asarray(preds['forward'], np.int32),
        pred_layer=np.asarray(preds['layer'], np.int16), pred_target=np.asarray(preds['target'], np.int16),
        pred_outcome=np.asarray(preds['outcome'], np.int8), pred_expert=np.asarray(preds['expert'], np.int16),
        pred_slot=np.asarray(preds['slot'], np.int16), pred_demand_copies=np.asarray(preds['demand_copies'], np.int16),
        pred_exchanges=np.asarray(preds['exchanges'], np.int16), pred_ids=np.asarray(preds['ids'], np.int16).reshape(-1, K),
        pred_base=np.asarray(preds['base'], np.float32).reshape(-1, E), pred_has_base=np.asarray(preds['has_base'], bool),
    )
    # The 5090's resident order (res_ids) where each layer is first seen and at every pass, for victim ties.
    meta['primary_order_first_generation'] = {}
    for i in range(P):
        if passes['generation'][i] and passes['layer'][i] not in meta['primary_order_first_generation']:
            meta['primary_order_first_generation'][int(passes['layer'][i])] = passes['primary_order'][i]
    meta['forwards'] = forwards
    meta['rounds'] = rounds
    meta['source'] = str(path)
    return dict(arrays=out, meta=meta)


def save_extract(x: dict, path) -> None:
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **x['arrays'])
    path.with_suffix('.json').write_text(json.dumps(x['meta']))


def load_extract(path) -> dict:
    path = pathlib.Path(path)
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    return dict(arrays=arrays, meta=json.loads(path.with_suffix('.json').read_text()))


# ----------------------------------------------------------------------------------------------------------------------
# The engine's routing rule


def reroute(base: np.ndarray, ready: np.ndarray, bonus: float) -> np.ndarray:
    """resident::route with Policy::Prior: per row the top four of base (float32 sigmoid + bias) plus the bonus on ready
    experts, over all 128, higher first and ties to the lower id. A zero bonus bypasses the pass (the router's top four).
    base: [R, 128] float32; ready: [R, 128] or [128] bool. Returns [R, 4] int16 in rank order."""
    base = np.asarray(base, np.float32)
    if bonus > 0:
        adj = base + np.where(ready, np.float32(bonus), np.float32(0))
    else:
        adj = base
    order = np.argsort(-adj, axis=-1, kind='stable')
    return order[..., :K].astype(np.int16)


def router_top4(base: np.ndarray) -> np.ndarray:
    return reroute(base, np.zeros(base.shape, bool), 0.0)


# ----------------------------------------------------------------------------------------------------------------------
# Scores (mm3_native_demand_units.hpp)


def decay_factor(rows: int, decay5: float) -> np.float32:
    """native_demand_decay: the decay over `rows` evaluated rows, `decay5` per five rows, as a float."""
    return np.float32(math.exp(math.log(decay5) * rows / 5.0))


def credit(rows: int, decay5: float) -> np.float32:
    """native_demand_credit: each original pick's credit, so a steady pick converges to 100."""
    return np.float32(-100.0 * math.expm1(math.log(decay5) * rows / 5.0) / rows) if rows > 0 else np.float32(0)


def budget_advance(value: float, last: int, row: int, capacity: float) -> float:
    """NativeAdmissionBudget::advance: capacity / 5 per evaluated row, bounded by capacity."""
    return min(capacity, max(0.0, value) + max(0, row - last) * capacity / 5.0)


NATIVE_DECAY5 = 0.95


def per_row_to_decay5(d: float) -> float:
    return d ** 5


# ----------------------------------------------------------------------------------------------------------------------
# Policies


class PolicySpec:
    """One residency policy. decay5 is the score's decay per five generation rows (the engine: 0.95). A fast component
    (fast5, fast_weight) makes a two-speed score, normalized as (slow + w * fast) / (1 + w) so a steady pick still
    converges to 100 and the admission thresholds keep their units. bonus_credit > 0 adds the bonus-aware credit, as a
    fraction of a pick's credit. helper_capacity is the 3090's NativeAdmissionBudget capacity (1: one in five rows);
    helper_anywhere lets a new lease start at any pass once the last one retired (needed above one a token, since the
    engine issues leases only at a forward's end)."""

    def __init__(self, name, recorded=False, decay5=NATIVE_DECAY5, fast5=None, fast_weight=0.0, bonus_credit=0.0,
                 helper_capacity=1.0, helper_anywhere=False, primary_capacity=8.0, early='auto', extra_residents=0):
        self.name, self.recorded = name, recorded
        # early: 'replay' the recorded early copy where allowed; 'today' re-decides today's rule from the recorded prediction
        # (e1561fc captures); 'item4' routes the prediction with the bonus and this policy's mask and copies the first
        # predicted miss; 'item4b' adds copies of the other predicted misses when the pass queued no demand copy, hidden
        # from routing until landed (served from their frame if picked); 'off' copies nothing. 'auto': 'today' where the
        # capture has the prediction, else 'replay'.
        self.early, self.extra_residents = early, extra_residents
        self.decay5, self.fast5, self.fast_weight, self.bonus_credit = decay5, fast5, fast_weight, bonus_credit
        self.helper_capacity, self.helper_anywhere, self.primary_capacity = helper_capacity, helper_anywhere, primary_capacity

    def describe(self) -> dict:
        return {k: v for k, v in vars(self).items()}


def default_policies() -> list[PolicySpec]:
    slow = per_row_to_decay5
    return [
        PolicySpec('P0', recorded=True),
        PolicySpec('P1'),
        PolicySpec('P2-0.5', helper_capacity=2.5),
        PolicySpec('P2-1', helper_capacity=5.0),
        PolicySpec('P2-2', helper_capacity=10.0, helper_anywhere=True),
        PolicySpec('P3', decay5=slow(0.99), fast5=slow(0.7), fast_weight=0.5),
        PolicySpec('P3-0.99/0.7/1', decay5=slow(0.99), fast5=slow(0.7), fast_weight=1.0),
        PolicySpec('P3-0.99/0.5/1', decay5=slow(0.99), fast5=slow(0.5), fast_weight=1.0),
        PolicySpec('P3-0.98/0.8/1', decay5=slow(0.98), fast5=slow(0.8), fast_weight=1.0),
        PolicySpec('P3-0.995/0.8/1', decay5=slow(0.995), fast5=slow(0.8), fast_weight=1.0),
        PolicySpec('P3-native/0.7/0.5', fast5=slow(0.7), fast_weight=0.5),
        PolicySpec('P4', bonus_credit=1.0),
        PolicySpec('P4-0.5', bonus_credit=0.5),
        # The scores again under the higher 3090 caps that scheduling is meant to carry.
        PolicySpec('P2-1+P3', helper_capacity=5.0, decay5=slow(0.99), fast5=slow(0.7), fast_weight=0.5),
        PolicySpec('P2-2+P3', helper_capacity=10.0, helper_anywhere=True, decay5=slow(0.99), fast5=slow(0.7), fast_weight=0.5),
        PolicySpec('P2-1+P4', helper_capacity=5.0, bonus_credit=1.0),
        PolicySpec('P2-2+P4', helper_capacity=10.0, helper_anywhere=True, bonus_credit=1.0),
        PolicySpec('P2-1+P4-0.5', helper_capacity=5.0, bonus_credit=0.5),
        PolicySpec('P2-2+P4-0.5', helper_capacity=10.0, helper_anywhere=True, bonus_credit=0.5),
    ]



def decay_policies() -> list[PolicySpec]:
    """The decay screen (orchestrator's direction, 2026-11-10): the native score's decay per five rows, at today's caps and
    with the 3090 at one exchange a token. Thresholds and the 5090's price are the engine's, unchanged."""
    out = [PolicySpec('P1')]
    for d in (0.975, 0.95, 0.90, 0.80, 0.60, 0.40):
        out.append(PolicySpec(f'D{d:g}', decay5=d))
        out.append(PolicySpec(f'D{d:g}+P2-1', decay5=d, helper_capacity=5.0))
    return out


# ----------------------------------------------------------------------------------------------------------------------
# Metrics


METRICS = ('picks', 'moved', 'mass_original', 'mass_executed', 'on_primary', 'on_helper', 'on_staged', 'need_load',
           'load_jobs', 'uploads', 'cpu_jobs', 'own_not_on_gpu', 'own_not_ready', 'primary_exchanges', 'helper_exchanges',
           'primary_reused', 'early_copies', 'on_early', 'early_not_host', 'passes', 'all_resident', 'pairs',
           'all_resident_pairs', 'helper_jobs')


class Tally:
    """Per generation forward sums of the pass metrics, and the forward's committed tokens."""

    def __init__(self, forwards: int):
        self.v = {m: np.zeros(forwards, np.float64) for m in METRICS}

    def add(self, f: int, **kw):
        for k, x in kw.items():
            self.v[k][f] += x


def pick_metrics(base, sig, original, executed, ready, res, helper, early: int = -1, served=None):
    """Per-pass sums for rows of one pass: base/sig [R,128], original/executed [R,4], masks [128] (one layer). `early`
    is the expert an early copy put in a frame for this layer (-1: none); picks on it are counted in on_early. `served`
    (default: ready) also holds experts in frames the routing did not see (item 4's hidden copies)."""
    R = len(executed)
    rows = np.arange(R)[:, None]
    own = np.zeros((R, E), bool)
    own[rows, original] = True
    moved = int((~own[rows, executed]).sum())
    on_p = res[executed]
    on_h = helper[executed] & ~on_p
    rdy = (ready if served is None else served)[executed]
    staged = rdy & ~on_p & ~on_h
    load = ~rdy
    return dict(picks=R * K, moved=moved, mass_original=float(sig[rows, original].sum()),
                mass_executed=float(sig[rows, executed].sum()), on_primary=int(on_p.sum()), on_helper=int(on_h.sum()),
                on_staged=int(staged.sum()), need_load=int(load.sum()), load_jobs=int(np.unique(executed[load]).size),
                own_not_on_gpu=int((~(res | helper)[original]).sum()), own_not_ready=int((~ready[original]).sum()),
                on_early=int(((executed == early) & staged).sum()) if early >= 0 else 0,
                passes=1, all_resident=int(bool((on_p | on_h).all())), helper_jobs=int(np.unique(executed[on_h]).size))


# ----------------------------------------------------------------------------------------------------------------------
# The engine's residency rules, simulated


class Sim:
    """P1-P4: the engine's residency policy over one request's generation passes, re-routing each pass under `bonus`.

    Timing is replaced by fixed delays in layer passes, measured on the 2026-11-08 captures (one row): a 5090 exchange
    issued at a forward's end publishes at the next forward's begin, one issued at a layer join at the next pass's join;
    a 3090 lease issued at a forward's end withdraws its victim before the next forward's second pass, publishes four
    passes later and retires three passes after that."""

    HELPER_DELAYS = (1, 4, 3)
    PRIMARY_JOIN_DELAY = 1
    PRIMARY_END_DELAY = 0
    EARLY_HOLD = 1

    def __init__(self, x: dict, origin_scores: dict, spec: PolicySpec, bonus: float, bridge=None, config: dict | None = None,
                 delays: dict | None = None):
        a, meta = x['arrays'], x['meta']
        self.a, self.meta, self.spec, self.bonus = a, meta, spec, np.float32(bonus)
        d = measure_delays(x) if delays is None else delays
        self.helper_delays = tuple(d.get('helper', self.HELPER_DELAYS))
        self.primary_join_delay = int(d.get('primary_join', self.PRIMARY_JOIN_DELAY))
        self.primary_end_delay = int(d.get('primary_end', self.PRIMARY_END_DELAY))
        # An early copy's frame is busy (slot_early) until a layer join sees the copy complete: early_hold passes.
        self.early_hold = int(d.get('early_hold', self.EARLY_HOLD))
        # Where a forward-end exchange publishes: the next forward's begin, or (one row on today's base, whose copies are
        # not done by then) its first layer join.
        self.primary_end_at_join = bool(d.get('primary_end_at_join', False))
        req = meta['request']
        self.L = req['layer_count']
        L = self.L
        self.moe = np.zeros(L, bool)
        self.res = np.zeros((L, E), bool)
        self.helper = np.zeros((L, E), bool)
        self.res_ids: dict[int, list[int]] = {}
        for l in req['layers']:
            li = l['layer']
            self.moe[li] = True
            self.res_ids[li] = list(l['primary'])
            self.res[li, l['primary']] = True
            self.helper[li, l['helper']] = True
        self.layers = [li for li in range(L) if self.moe[li]]
        self.exists = self.moe[:, None] & np.ones((1, E), bool)
        # Scores: the origin's, restored at the request's start.
        self.slow = np.zeros((L, E), np.float32)
        for k, v in origin_scores.items():
            li, e = divmod(int(k), KEY)
            if li < L and e < E:
                self.slow[li, e] = v
        self.fast = self.slow.copy() if spec.fast5 is not None else None
        if spec.extra_residents:
            self.add_residents(origin_scores, spec.extra_residents)
        policy = req['policy']
        self.benefit = (1.0 - float(np.float32(0.95)) ** policy['horizon']) * policy['service_saved_ms']
        self.h2d, self.d2h = policy['h2d_ms'], (0.0 if policy.get('primary_backing') else policy['d2h_ms'])
        self.interference, self.margin = policy.get('interference_ms', 0.0), policy.get('margin_ms', 0.0)
        self.prices = meta['batch_staging_calibration']
        self.cpu_scale = req['cooperative_cpu_scale']
        self.bridge = bridge
        self.row = 0
        self.pcredit, self.prow = 0.0, 0
        self.hcredit, self.hrow = 0.0, 0
        # Staging frames: owner (layer, expert) or None, last use (forward), reserved, early copy busy until pass.
        self.n_slots = 8
        self.owner = [None] * self.n_slots
        self.used = [-1] * self.n_slots
        self.reserved = [False] * self.n_slots
        self.early_busy = [-1] * self.n_slots
        self.hidden = [-1] * self.n_slots      # item 4b: a copy not ready for routing until after this pass index
        self.now = 0
        g = np.where(a['pass_generation'])[0]
        if len(g):
            st = a['staging'][g[0]]
            for s in range(self.n_slots):
                k = int(st[s, 0])
                if k >= 0 and not st[s, 4]:
                    self.owner[s] = divmod(k, KEY)
                    self.used[s] = int(st[s, 5])
        self.forward = -1
        self.last_early = None
        self.pass_log: list = []
        self.px: list[dict | None] = [None, None]     # 5090 exchanges in flight (two descriptors)
        # The recorded early-copy predictions (e1561fc captures) by the pass that made them.
        self.pred = {}
        if 'pred_at' in a and len(a['pred_at']):
            for i, (at, lay) in enumerate(zip(a['pred_at'], a['pred_layer'])):
                q = int(at) - 1
                if q >= 0 and int(a['pass_layer'][q]) == int(lay) and a['pred_has_base'][i]:
                    self.pred[q] = i
        self.early_mode = spec.early if spec.early != 'auto' else ('today' if self.pred else 'replay')
        self.lease: dict | None = None               # the 3090's lease
        self.counts = dict(primary_issued=0, primary_reused=0, primary_join=0, helper_issued=0, early_replayed=0, early_skipped=0,
                           early_blocked=0, early_not_host=0, early_no_frame=0, early_no_candidate=0)
        self.early_not_host_now = False
        self.issues: list[tuple] = []                # (kind, forward, incoming key, victim key)
        self.executed = np.zeros_like(a['row_executed'])

    # -- score
    def score(self) -> np.ndarray:
        if self.fast is None:
            return self.slow
        w = np.float32(self.spec.fast_weight)
        return (self.slow + w * self.fast) / (np.float32(1) + w)

    def begin_forward(self, rows: int):
        self.row += rows
        self.slow *= decay_factor(rows, self.spec.decay5)
        if self.fast is not None:
            self.fast *= decay_factor(rows, self.spec.fast5)

    def observe(self, li: int, original: np.ndarray, rows: int, extra: np.ndarray | None = None):
        c = credit(rows, self.spec.decay5)
        np.add.at(self.slow[li], original.ravel().astype(np.int64), c)
        if self.fast is not None:
            np.add.at(self.fast[li], original.ravel().astype(np.int64), credit(rows, self.spec.fast5))
        if extra is not None and extra.size:
            w = np.float32(self.spec.bonus_credit)
            np.add.at(self.slow[li], extra.astype(np.int64), np.float32(c * w))
            if self.fast is not None:
                np.add.at(self.fast[li], extra.astype(np.int64), np.float32(credit(rows, self.spec.fast5) * w))

    # -- state queries
    def pending_keys(self) -> set:
        out = set()
        for x in self.px:
            if x is not None:
                out.add(x['incoming']); out.add(x['victim'])
        return out

    def leased_keys(self) -> set:
        return set() if self.lease is None else {self.lease['incoming'], self.lease['victim']}

    def host(self) -> np.ndarray:
        """Experts with a HOST record: not resident on either card. A 3090 lease's victim is HOST once withdrawn; its
        incoming stays HOST until published."""
        return self.exists & ~self.res & ~self.helper

    def visible(self, li: int, hidden_too: bool = False) -> np.ndarray:
        m = np.zeros(E, bool)
        for s in range(self.n_slots):
            o = self.owner[s]
            if o is not None and o[0] == li and not self.reserved[s] and (hidden_too or self.hidden[s] < self.now):
                m[o[1]] = True
        return m

    def ready(self, li: int) -> np.ndarray:
        return self.res[li] | self.helper[li] | self.visible(li)

    # -- 5090 exchanges (mm3_native_adaptive.inc)
    def publish_primary(self, now: int, protected: int = -1, join: bool = False):
        """Publication at a boundary: a layer join (join=True, `protected` is its layer) or a forward's begin or end."""
        for i, x in enumerate(self.px):
            if x is None or x['due'] > now or protected in (x['incoming'][0], x['victim'][0]) or (x['join_only'] and not join):
                continue
            (al, ae), (bl, be) = x['incoming'], x['victim']
            row = self.res_ids[bl]
            j = row.index(be)
            row[j] = row[-1]
            row.pop()
            self.res[bl, be] = False
            self.res_ids[al].append(ae)
            self.res[al, ae] = True
            s = x['slot']
            self.owner[s], self.used[s], self.reserved[s] = (bl, be), self.forward - 1, False
            for t in range(self.n_slots):
                if t != s and self.owner[t] == (al, ae):
                    self.owner[t] = None
            self.px[i] = None

    def admit_primary(self, now: int, early_layer: int = -1, tally=None, f: int = -1):
        self.pcredit = budget_advance(self.pcredit, self.prow, self.row, self.spec.primary_capacity)
        self.prow = self.row
        need = 2.0 if early_layer >= 0 else 1.0
        sc = self.score()
        for tx in range(len(self.px)):
            if not self.pcredit + 1e-9 >= need:
                break
            if self.px[tx] is not None:
                continue
            pend, leased = self.pending_keys(), self.leased_keys()
            frames = [s for s in range(self.n_slots) if self.owner[s] is not None and not self.reserved[s]
                      and self.early_busy[s] < now and self.owner[s][0] != early_layer
                      and self.owner[s] not in pend and self.owner[s] not in leased]
            if early_layer >= 0:
                threshold = self.d2h + self.interference + self.margin
                if not any(float(sc[self.owner[s]]) * self.benefit > threshold for s in frames):
                    break
            vk = [(li, e) for li in self.layers for e in self.res_ids[li] if (li, e) not in pend and (li, e) not in leased]
            if not vk:
                break
            vs = np.array([sc[k] for k in vk], np.float32)
            v = int(np.argmin(vs))
            victim, cold = vk[v], vs[v]
            host = self.host()
            cand, slots = [], []
            for s in frames:
                k = self.owner[s]
                if host[k]:
                    cand.append(k); slots.append(s)
            ready_keys = set(cand)
            if early_layer < 0:
                hm = host.copy()
                for k in pend | leased:
                    hm[k] = False
                for k in ready_keys:
                    hm[k] = False
                li_, e_ = np.nonzero(hm)
                cand_host = list(zip(li_.tolist(), e_.tolist()))
            else:
                cand_host = []
            if not cand and not cand_host:
                break
            cs = np.array([sc[k] for k in cand] + ([] if not cand_host else sc[tuple(np.array(cand_host).T)].tolist()), np.float32)
            reuse = np.array([1] * len(cand) + [0] * len(cand_host), bool)
            value = (cs - cold).astype(np.float64) * self.benefit - self.d2h - np.where(reuse, 0.0, self.h2d) - self.interference - self.margin
            i = int(np.argmax(value))
            if not value[i] > 0:
                break
            incoming = (cand + cand_host)[i]
            if reuse[i]:
                slot = slots[i]
            else:
                free = [s for s in range(self.n_slots) if not self.reserved[s] and self.early_busy[s] < now]
                if not free:
                    break
                slot = min(free, key=lambda s: (self.used[s], s))
            self.reserved[slot] = True
            self.owner[slot] = None
            due = now + (self.primary_join_delay if early_layer >= 0 else self.primary_end_delay)
            self.px[tx] = dict(incoming=incoming, victim=victim, slot=slot, reuse=bool(reuse[i]), due=due,
                               join_only=early_layer < 0 and self.primary_end_at_join)
            self.pcredit = max(0.0, self.pcredit - 1.0)
            self.counts['primary_issued'] += 1
            self.counts['primary_reused'] += int(reuse[i])
            self.counts['primary_join'] += int(early_layer >= 0)
            self.issues.append(('primary', self.forward, incoming[0] * KEY + incoming[1], victim[0] * KEY + victim[1]))
            if tally is not None and f >= 0:
                tally.add(f, primary_exchanges=1, primary_reused=int(reuse[i]))

    # -- 3090 lease (mm3_native_helper_live.inc)
    def helper_progress(self, now: int):
        x = self.lease
        if x is None or x['due'] > now:
            return
        if x['phase'] == 'prepared':
            self.helper[x['victim']] = False
            x['phase'], x['due'] = 'uploaded', now + self.helper_delays[1]
        elif x['phase'] == 'uploaded':
            self.helper[x['incoming']] = True
            for s in range(self.n_slots):
                if self.owner[s] == x['incoming']:
                    self.owner[s] = None
            x['phase'], x['due'] = 'recycled', now + self.helper_delays[2]
        else:
            self.lease = None

    def admit_helper(self, now: int, tally=None, f: int = -1):
        if self.lease is not None:
            return
        self.hcredit = budget_advance(self.hcredit, self.hrow, self.row, self.spec.helper_capacity)
        self.hrow = self.row
        if not self.hcredit + 1e-9 >= 1.0:
            return
        sc = self.score()
        busy = self.pending_keys()
        for s in range(self.n_slots):
            if self.owner[s] is not None and self.early_busy[s] >= now:
                busy.add(self.owner[s])
        vm = self.helper.copy()
        im = self.host()
        for k in busy:
            vm[k] = False; im[k] = False
        if not vm.any():
            return
        vflat = np.where(vm.ravel(), sc.ravel(), np.float32(np.inf))
        v = int(np.argmin(vflat))
        cold = vflat[v]
        iflat = np.where(im.ravel() & (sc.ravel() >= np.float32(1.8)), sc.ravel(), np.float32(-np.inf))
        i = int(np.argmax(iflat))
        hot = iflat[i]
        if not np.isfinite(hot) or not hot > np.float32(np.float32(1.25) * cold + np.float32(1.0)):
            return
        self.lease = dict(incoming=divmod(i, E), victim=divmod(v, E), phase='prepared', due=now + self.helper_delays[0])
        self.hcredit = max(0.0, self.hcredit - 1.0)
        self.counts['helper_issued'] += 1
        self.issues.append(('helper', self.forward, self.lease['incoming'][0] * KEY + self.lease['incoming'][1],
                            self.lease['victim'][0] * KEY + self.lease['victim'][1]))
        if tally is not None and f >= 0:
            tally.add(f, helper_exchanges=1)

    # -- one pass's demand
    def plan(self, li: int, executed: np.ndarray, ready_slot: dict, rows: int, now: int, forward: int) -> tuple[int, int, list]:
        """The batch-staging planner (production code through the bridge), then the demand copies into frames. Returns
        (uploads, cpu jobs, the frames loaded)."""
        uses = np.bincount(executed.ravel().astype(np.int64), minlength=E)
        cpu, gpu, hlp = [], [], []
        for e in np.nonzero(uses)[0].tolist():
            if self.helper[li, e]:
                hlp.append(e)
            elif self.res[li, e] or e in ready_slot:
                gpu.append(e)
            else:
                cpu.append(e)
        busy = set(ready_slot.values())
        available = [s for s in range(self.n_slots) if not self.reserved[s] and s not in busy and self.early_busy[s] < now]
        selected = []
        if cpu:
            maximum = min(len(available), self.n_slots // 2, 4)
            if self.bridge is not None:
                selected, _ = self.bridge.stage(uses.tolist(), cpu, gpu, hlp, cpu, maximum, rows, self.prices, self.cpu_scale)
            else:
                selected = cpu[:maximum]
        first = {}
        for e in executed.ravel().tolist():
            first.setdefault(e, len(first))
        loaded = []
        for e in sorted(selected, key=first.__getitem__):
            s = min(available, key=lambda t: (self.used[t], t))
            available.remove(s)
            self.owner[s], self.used[s] = (li, e), forward
            loaded.append(s)
        return len(selected), len(cpu) - len(selected), loaded

    def add_residents(self, origin_scores: dict, n: int) -> None:
        """Task 3 (slots freed by a lower-precision drafter or head): n more 5090 residents at the request's start, the HOST
        experts with the highest origin scores (layer, expert order on ties), appended to their layers' resident order."""
        host = self.exists & ~self.res & ~self.helper
        sc = np.where(host, self.slow, -np.inf).ravel()
        order = np.argsort(-sc, kind='stable')[:n]
        for k in order.tolist():
            li, e = divmod(k, E)
            self.res[li, e] = True
            self.res_ids[li].append(e)

    def early_decide(self, li: int, i: int, busy: set, staged: int, now: int, forward: int) -> int:
        """The early copy for layer li + 1 from the recorded prediction i, under this policy's rule (today / item4 /
        item4b). Returns the copies issued; the first is ready for the next routing, item 4b's others are not."""
        a = self.a
        nl = li + 1
        if nl >= self.L or not self.moe[nl]:
            return 0
        if staged > 1 or any(x is not None for x in self.px):
            self.counts['early_blocked'] += 1
            return 0
        pend, leased = self.pending_keys(), self.leased_keys()
        owned = {o for o in self.owner if o is not None}

        def host_only(e):
            k = (nl, e)
            return not (self.res[nl, e] or self.helper[nl, e] or k in leased or k in pend or k in owned)
        if self.early_mode == 'today':
            ids = [int(e) for e in a['pred_ids'][i]]
            cands = [e for e in ids if host_only(e)][:1]
        else:
            mask = self.ready(nl)
            ids = reroute(a['pred_base'][i][None], mask, float(self.bonus))[0].tolist()
            misses = [e for e in ids if not mask[e] and host_only(e)]
            cands = misses[:1] if (self.early_mode == 'item4' or staged > 0) else misses
        if not cands:
            self.counts['early_no_candidate'] += 1
            return 0
        issued = 0
        for j, e in enumerate(cands):
            free = [s for s in range(self.n_slots) if s not in busy and not self.reserved[s] and self.early_busy[s] < now]
            if not free:
                self.counts['early_no_frame'] += int(j == 0)
                break
            s = min(free, key=lambda t: (self.used[t], t))
            self.owner[s], self.used[s], self.early_busy[s] = (nl, e), forward, now + self.early_hold - 1
            self.hidden[s] = now + 1 if j > 0 else -1
            busy = busy | {s}
            issued += 1
            if j == 0:
                self.last_early = (now, (nl, e))
        self.counts['early_replayed'] += issued
        return issued

    def early_copy(self, li: int, key, executed_slots: set, staged: int, now: int, forward: int) -> None:
        """Replays a recorded early copy (mode 2) where this policy's state allows it: the pass queued at most one demand
        copy, no 5090 exchange is pending, the expert is HOST and in no frame. Its prediction is not recorded, so where the
        recorded expert is not HOST here nothing is copied."""
        nl, ne = key
        if staged > 1 or any(x is not None for x in self.px):
            self.counts['early_skipped'] += 1
            self.counts['early_blocked'] += 1
            return
        if self.res[nl, ne] or self.helper[nl, ne] or (nl, ne) in self.leased_keys() or any(o == (nl, ne) for o in self.owner):
            # The engine would have tried its next predicted pick, which is not recorded: counted for an upper bound.
            self.counts['early_skipped'] += 1
            self.counts['early_not_host'] += 1
            self.early_not_host_now = True
            return
        free = [s for s in range(self.n_slots) if s not in executed_slots and not self.reserved[s] and self.early_busy[s] < now]
        if not free:
            self.counts['early_skipped'] += 1
            self.counts['early_no_frame'] += 1
            return
        s = min(free, key=lambda t: (self.used[t], t))
        self.owner[s], self.used[s], self.early_busy[s] = (nl, ne), forward, now + self.early_hold - 1
        self.counts['early_replayed'] += 1
        return True

    # -- the request
    def run(self, tally_for=None) -> dict:
        a = self.a
        gen = np.where(a['pass_generation'])[0]
        forwards = a['pass_forward'][gen]
        fidx = np.unique(forwards, return_inverse=True)[1]
        tally = Tally(int(fidx.max()) + 1 if len(gen) else 0)
        early = early_copies_by_pass(a)
        prev_f = None
        prev_pass = None
        self.forward = -1
        for n, p in enumerate(gen):
            quiet_tick()
            f = int(fidx[n])
            forward = int(a['pass_forward'][p])
            rows = int(a['pass_rows'][p])
            if forward != prev_f:
                if prev_f is not None:
                    self.end_forward(n, tally, int(fidx[n - 1]))
                self.forward = forward
                self.begin_forward(rows)
                self.publish_primary(n)
                self.helper_progress(n)
                prev_f = forward
            li = int(a['pass_layer'][p])
            self.now = n
            # window_pre: the lease progresses (and, above one a token, may start) before the routing.
            self.helper_progress(n)
            if self.spec.helper_anywhere:
                self.admit_helper(n, tally, f)
            r0 = int(a['pass_row0'][p])
            base, sig = a['row_base'][r0:r0 + rows], a['row_sig'][r0:r0 + rows]
            original = a['row_original'][r0:r0 + rows].astype(np.int64)
            ready = self.ready(li)
            executed = reroute(base, ready, float(self.bonus)).astype(np.int64)
            self.executed[r0:r0 + rows] = executed
            ek = self.last_early if self.last_early is not None and self.last_early[0] == n - 1 and self.last_early[1][0] == li else None
            served = ready | self.visible(li, hidden_too=True)
            m = pick_metrics(base, sig, original, executed, ready, self.res[li], self.helper[li], ek[1][1] if ek else -1,
                             served=served)
            pair_with_previous(m, prev_pass, forward, li)
            prev_pass = (forward, li, m['all_resident'])
            # The layer join: publication (this layer protected) and the retention of staged frames.
            self.publish_primary(n, protected=li, join=True)
            self.admit_primary(n, early_layer=li, tally=tally, f=f)
            extra = self.bonus_aware(base, executed, ready) if self.spec.bonus_credit > 0 else None
            self.observe(li, original, rows, extra)
            ready_slot = {}
            for s in range(self.n_slots):
                o = self.owner[s]
                if o is not None and o[0] == li and not self.reserved[s] and o[1] in executed and served[o[1]] and not self.res[li, o[1]] and not self.helper[li, o[1]]:
                    ready_slot[o[1]] = s
                    self.used[s] = forward
            up, cpu, loaded = self.plan(li, executed, ready_slot, rows, n, forward)
            m.update(uploads=up, cpu_jobs=cpu)
            self.pass_log.append(pass_record(f, rows, li, m))
            tally.add(f, **m)
            if rows == 1 and self.early_mode in ('today', 'item4', 'item4b') and p in self.pred:
                k = self.early_decide(li, self.pred[p], set(ready_slot.values()) | set(loaded), up, n, forward)
                if k:
                    tally.add(f, early_copies=k)
            elif rows == 1 and self.early_mode == 'replay' and p in early:
                # The frames this pass reads and the ones it just loaded (their copies are still foreground) are not free.
                self.early_not_host_now = False
                if self.early_copy(li, early[p], set(ready_slot.values()) | set(loaded), up, n, forward):
                    self.last_early = (n, early[p])
                    tally.add(f, early_copies=1)
                elif self.early_not_host_now:
                    tally.add(f, early_not_host=1)
        if prev_f is not None:
            self.end_forward(len(gen), tally, int(fidx[-1]))
        return dict(tally=tally, counts=self.counts, issues=self.issues, executed=self.executed,
                    passes=np.asarray(self.pass_log, np.int32).reshape(-1, len(PASS_FIELDS)))

    def end_forward(self, now: int, tally, f: int):
        self.publish_primary(now)
        self.helper_progress(now)
        self.admit_helper(now, tally, f)
        self.admit_primary(now, early_layer=-1, tally=tally, f=f)

    def bonus_aware(self, base, executed, ready) -> np.ndarray:
        """Experts that could have replaced a loaded pick: not ready, not selected, base + bonus at least that pick's base
        (ties to the lower id). Once per row."""
        out = []
        b = self.bonus
        for r in range(len(executed)):
            row = executed[r]
            loaded = [e for e in row.tolist() if not ready[e]]
            if not loaded:
                continue
            adj = base[r] + b
            ok = np.zeros(E, bool)
            for e in loaded:
                ok |= (adj > base[r, e]) | ((adj == base[r, e]) & (np.arange(E) < e))
            ok &= ~ready
            ok[row] = False
            out.extend(np.nonzero(ok)[0].tolist())
        return np.asarray(out, np.int64)


def measure_delays(x: dict) -> dict:
    """The transfer delays in layer passes, as the capture shows them (medians): a 3090 lease from issue to withdrawal,
    to publication and to retirement; a 5090 exchange from a layer join's issue to its publication, and from a forward
    end's issue to its publication (0: at the next forward's begin). Event positions are the index of the next routes
    record. Falls back to the 2026-11-08 one-row values where the capture has none."""
    a = x['arrays']
    k, at = a['ev_kind'], a['ev_at']
    inc, vic = a['ev_incoming'], a['ev_victim']
    out: dict = {}
    leases: dict = {}
    for i in np.where((k >= EV['helper_issue']) & (k <= EV['helper_retire']))[0]:
        key = (int(inc[i]), int(vic[i]))
        if k[i] == EV['helper_issue']:
            leases[key] = [int(at[i])]
        elif key in leases:
            leases[key].append(int(at[i]))
    full = np.array([v for v in leases.values() if len(v) == 4])
    if len(full):
        out['helper'] = tuple(int(max(1, round(float(np.median(np.diff(full, axis=1)[:, j]))))) for j in range(3))
    issued: dict = {}
    join, end, end_join = [], [], []
    for i in np.where((k == EV['primary_issue']) | (k == EV['primary_publish']))[0]:
        key = (int(inc[i]), int(vic[i]))
        if k[i] == EV['primary_issue']:
            issued[key] = i
        elif key in issued:
            j = issued.pop(key)
            if a['ev_early_layer'][j] >= 0:
                join.append(int(at[i] - at[j]))
            else:
                end.append(int(at[i] - at[j]))
                end_join.append(int(a['ev_layer'][i]) >= 0)
    if join:
        out['primary_join'] = int(max(1, round(float(np.median(join)))))
    if end:
        out['primary_end'] = int(max(0, round(float(np.median(end)))))
        out['primary_end_at_join'] = bool(np.mean(end_join) > 0.5)
    return out


PASS_FIELDS = ('forward_index', 'rows', 'layer', 'type', 'loads', 'uploads', 'cpu_jobs', 'helper_jobs')   # type: 0 resident, 1 staged, 2 load


def pass_record(f: int, rows: int, layer: int, m: dict) -> tuple:
    """One pass for pricing: every pick on a GPU resident (0), some pick on a staged or early-copied frame and none
    needing a load (1), or some pick needing a load (2); its distinct loads, uploads and CPU jobs."""
    kind = 2 if m['need_load'] else (1 if m['on_staged'] else 0)
    return (f, rows, layer, kind, m['load_jobs'], m.get('uploads', 0), m.get('cpu_jobs', 0), m.get('helper_jobs', 0))


def pair_with_previous(m: dict, prev, forward: int, layer: int) -> None:
    """Consecutive layer pairs of one forward, for item 5's pair share."""
    if prev is not None and prev[0] == forward and prev[1] + 1 == layer:
        m['pairs'] = 1
        m['all_resident_pairs'] = int(prev[2] and m['all_resident'])


def early_copies_by_pass(a: dict) -> dict:
    """Recorded early copies (detailed 'early_copy' events) by the generation pass that issued them: the pass whose layer
    precedes the copied expert's layer, in the same forward."""
    k = a['ev_kind']
    idx = np.where(k == EV['early_copy'])[0]
    out = {}
    for i in idx:
        at = int(a['ev_at'][i])            # the next routes record's index
        key = divmod(int(a['ev_incoming'][i]), KEY)
        p = at - 1
        if p >= 0 and int(a['pass_layer'][p]) + 1 == key[0]:
            out[p] = key
    return out


# ----------------------------------------------------------------------------------------------------------------------
# P0: the recorded residency


def run_recorded(x: dict, bonus: float, bridge=None) -> dict:
    """Re-route every generation pass against the ready masks the engine recorded (residency held as it was). Uploads
    and CPU jobs come from the production planner on the recorded frames (at the recorded bonus they are the recorded
    ones, given too)."""
    a = x['arrays']
    gen = np.where(a['pass_generation'])[0]
    forwards = a['pass_forward'][gen]
    fidx = np.unique(forwards, return_inverse=True)[1]
    tally = Tally(int(fidx.max()) + 1 if len(gen) else 0)
    rp = a['row_pass']
    ready_p = bits_to_mask(a['ready_primary'])
    ready_h = bits_to_mask(a['ready_helper'])
    res_p = np.unpackbits(a['res_primary'], axis=1, bitorder='little').astype(bool)
    res_h = np.unpackbits(a['res_helper'], axis=1, bitorder='little').astype(bool)
    ready_rows = (ready_p | ready_h)[rp]
    executed = reroute(a['row_base'], ready_rows, bonus).astype(np.int64)
    same = (executed == a['row_executed']).all(axis=1)
    prices = x['meta']['batch_staging_calibration']
    scale = x['meta']['request']['cooperative_cpu_scale']
    early_for_issue = early_copies_by_pass(a)
    early_for = {p + 1: key for p, key in early_for_issue.items()}   # the pass that can read each copy
    prev_pass = None
    pass_log: list = []
    for n, p in enumerate(gen):
        quiet_tick()
        f = int(fidx[n])
        rows = int(a['pass_rows'][p])
        r0 = int(a['pass_row0'][p])
        ex = executed[r0:r0 + rows]
        ready = ready_p[p] | ready_h[p]
        ek = early_for.get(p)
        m = pick_metrics(a['row_base'][r0:r0 + rows], a['row_sig'][r0:r0 + rows], a['row_original'][r0:r0 + rows].astype(np.int64),
                         ex, ready, res_p[p], res_h[p], ek[1] if ek is not None and ek[0] == int(a['pass_layer'][p]) else -1)
        if p in early_for_issue:
            m['early_copies'] = 1
        pair_with_previous(m, prev_pass, int(a['pass_forward'][p]), int(a['pass_layer'][p]))
        prev_pass = (int(a['pass_forward'][p]), int(a['pass_layer'][p]), m['all_resident'])
        # The planner on the recorded frames: free frames are those not reserved and not read by this pass.
        st = a['staging'][p]
        li = int(a['pass_layer'][p])
        uses = np.bincount(ex.ravel(), minlength=E)
        cpu = [e for e in np.nonzero(uses)[0].tolist() if not ready[e]]
        gpu = [e for e in np.nonzero(uses)[0].tolist() if ready_p[p][e]]
        hlp = [e for e in np.nonzero(uses)[0].tolist() if ready_h[p][e] and not ready_p[p][e]]
        busy = sum(1 for s in range(8) if st[s, 0] >= 0 and st[s, 0] // KEY == li and uses[st[s, 0] % KEY] and not st[s, 4])
        available = sum(1 for s in range(8) if not st[s, 4]) - busy
        up = 0
        if cpu:
            maximum = max(0, min(available, 4))
            if bridge is not None:
                selected, _ = bridge.stage(uses.tolist(), cpu, gpu, hlp, cpu, maximum, rows, prices, scale)
            else:
                selected = cpu[:maximum]
            up = len(selected)
        m.update(uploads=up, cpu_jobs=len(cpu) - up)
        pass_log.append(pass_record(f, rows, int(a['pass_layer'][p]), m))
        tally.add(f, **m)
    # Recorded exchanges by generation forward.
    order = {int(fw): i for i, fw in enumerate(np.unique(forwards))}
    k = a['ev_kind']
    for i in np.where((k == EV['primary_issue']) | (k == EV['helper_issue']))[0]:
        f = order.get(int(a['ev_forward'][i]))
        if f is None:
            continue
        if k[i] == EV['primary_issue']:
            tally.add(f, primary_exchanges=1, primary_reused=int(a['ev_reuse'][i] == 1))
        else:
            tally.add(f, helper_exchanges=1)
    recorded = dict(uploads=a['uploads'][gen].astype(np.float64), cpu_jobs=a['cpu_jobs'][gen].astype(np.float64))
    rec = {k: np.bincount(fidx, weights=v, minlength=len(order)) for k, v in recorded.items()}
    return dict(tally=tally, same_rows=same, recorded=rec, passes=np.asarray(pass_log, np.int32).reshape(-1, len(PASS_FIELDS)))


# ----------------------------------------------------------------------------------------------------------------------
# Tokens and summaries


def forward_tokens(x: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per generation forward (in order): the generated-token index of its first row, and the tokens it commits (the next
    forward's first position minus its own; the last forward commits its rows' worth, at most)."""
    a = x['arrays']
    gen = np.where(a['pass_generation'])[0]
    forwards, first = np.unique(a['pass_forward'][gen], return_index=True)
    pos = a['pass_position'][gen][first].astype(np.int64)
    prompt = int(x['meta']['request_config']['prompt_tokens'])
    start = pos - prompt
    commit = np.diff(np.append(pos, pos[-1] + 1)) if len(pos) else pos
    return start, commit


def summarize(tally: Tally, x: dict, first_token: int = 128) -> dict:
    start, commit = forward_tokens(x)
    keep = start >= first_token
    tokens = float(commit[keep].sum())
    v = {k: float(arr[keep].sum()) for k, arr in tally.v.items()}
    out = dict(tokens=tokens, forwards=int(keep.sum()))
    per = lambda k: v[k] / tokens if tokens else float('nan')
    for k in METRICS:
        out[k + '_per_token'] = per(k)
    out['moved_share'] = v['moved'] / v['picks'] if v['picks'] else float('nan')
    out['mass_kept'] = v['mass_executed'] / v['mass_original'] if v['mass_original'] else float('nan')
    out['load_share'] = v['need_load'] / v['picks'] if v['picks'] else float('nan')
    # Every expert copied from host memory to the 5090 for this pass's work: demand uploads and early copies.
    out['transfers_per_token'] = (v['uploads'] + v['early_copies']) / tokens if tokens else float('nan')
    # Upper bound: had the engine copied another predicted pick wherever the recorded one was already on a GPU here.
    # Item 5 (host-free hand-off): the share of passes with every pick on a GPU resident (5090 or 3090; a staged or
    # early-copied frame does not count), and of consecutive layer pairs of a forward that are both such passes.
    out['all_resident_share'] = v['all_resident'] / v['passes'] if v['passes'] else float('nan')
    out['all_resident_pair_share'] = v['all_resident_pairs'] / v['pairs'] if v['pairs'] else float('nan')
    out['all_resident_next_given'] = v['all_resident_pairs'] / max(1.0, v['all_resident']) if v['all_resident'] else float('nan')
    out['transfers_upper_per_token'] = (v['uploads'] + v['early_copies'] + v['early_not_host']) / tokens if tokens else float('nan')
    return out


def load_origin_scores(path) -> dict:
    """The origin snapshot's demand scores (key -> value), restored at each request's start."""
    d = json.loads(pathlib.Path(path).read_text())
    snap = d.get('replay_snapshot', d)
    return {int(k): float(v) for k, v, _ in snap['replay_state']['scores']}


def score_check(x: dict, origin_scores: dict, decay5: float = NATIVE_DECAY5) -> dict:
    """The native score replayed from the router's own picks, against the scores each recorded exchange issue logged
    for its incoming expert and its victim (the engine's own values at the decision)."""
    sim = Sim(x, origin_scores, PolicySpec('check', decay5=decay5), 0.0)
    a = x['arrays']
    k = a['ev_kind']
    issues = np.where((k == EV['primary_issue']) | (k == EV['helper_issue']))[0]
    by_forward: dict[int, list[int]] = {}
    for i in issues:
        by_forward.setdefault(int(a['ev_forward'][i]), []).append(int(i))
    gen = np.where(a['pass_generation'])[0]
    diffs, checked = [], 0
    worst = []

    def compare(i):
        nonlocal checked
        sc = sim.score()
        for field, key in (('in_score', 'ev_incoming'), ('vic_score', 'ev_victim')):
            li, e = divmod(int(a[key][i]), KEY)
            mine, theirs = float(sc[li, e]), float(a['ev_' + field][i])
            rel = abs(mine - theirs) / max(1e-6, abs(theirs))
            diffs.append(rel)
            if rel > 1e-4:
                worst.append((int(a['ev_forward'][i]), int(a['ev_kind'][i]), field, mine, theirs))
        checked += 1

    prev = None
    for n, p in enumerate(gen):
        forward = int(a['pass_forward'][p])
        rows = int(a['pass_rows'][p])
        if forward != prev:
            if prev is not None:
                for i in by_forward.get(prev, []):
                    if a['ev_early_layer'][i] < 0 or k[i] == EV['helper_issue']:
                        compare(i)
            sim.begin_forward(rows)
            prev = forward
        li = int(a['pass_layer'][p])
        for i in by_forward.get(forward, []):
            if k[i] == EV['primary_issue'] and a['ev_early_layer'][i] == li:
                compare(i)
        r0 = int(a['pass_row0'][p])
        sim.observe(li, a['row_original'][r0:r0 + rows].astype(np.int64), rows)
    if prev is not None:
        for i in by_forward.get(prev, []):
            if a['ev_early_layer'][i] < 0 or k[i] == EV['helper_issue']:
                compare(i)
    d = np.asarray(diffs)
    return dict(issues=checked, scores=len(d), max_rel=float(d.max()) if len(d) else None,
                within_1e4=float((d <= 1e-4).mean()) if len(d) else None, worst=worst[:8])


def recorded_issues(x: dict) -> list[tuple]:
    a = x['arrays']
    k = a['ev_kind']
    out = []
    for i in np.where((k == EV['primary_issue']) | (k == EV['helper_issue']))[0]:
        out.append(('primary' if k[i] == EV['primary_issue'] else 'helper', int(a['ev_forward'][i]), int(a['ev_incoming'][i]),
                    int(a['ev_victim'][i])))
    return out


def fidelity(x: dict, result: dict) -> dict:
    """A simulated run at the recorded bonus against the recording: rows routed identically, and the exchange decisions
    (same incoming and victim, in order) per card."""
    a = x['arrays']
    same = (result['executed'] == a['row_executed']).all(axis=1)
    rec, sim = recorded_issues(x), result['issues']
    out = dict(rows=int(len(same)), rows_identical=float(same.mean()) if len(same) else None)
    first_row_diff = int(np.argmin(same)) if not same.all() else None
    out['first_differing_forward'] = int(a['pass_forward'][a['row_pass'][first_row_diff]]) if first_row_diff is not None else None
    for kind in ('primary', 'helper'):
        r = [t for t in rec if t[0] == kind]
        s = [t for t in sim if t[0] == kind]
        prefix = 0
        while prefix < min(len(r), len(s)) and r[prefix] == s[prefix]:
            prefix += 1
        rs, ss = {t[1:] for t in r}, {t[1:] for t in s}
        out[kind] = dict(recorded=len(r), simulated=len(s), identical_prefix=prefix, common=len(rs & ss))
    return out



# ----------------------------------------------------------------------------------------------------------------------
# Item 6 of the plan: rows of a verify window that share experts


def _beats_matrix(base, members, th):
    """B[r, i, j]: in row r, member i with the bonus outranks member j without it (float32, ties to the lower id)."""
    m = np.asarray(members, np.int64)
    adj = (base[:, m] + th)[:, :, None]
    b = base[:, m][:, None, :]
    return (adj > b) | ((adj == b) & (m[:, None] < m[None, :])[None])


def share_greedy(base, ready, ex, bonus):
    """Drop union members while every row that picks one has another union member (not already its pick) that outranks
    it with the bonus; each dropped expert's rows re-route with the bonus on ready plus the remaining union. Order: the
    member used by the fewest rows first, then the lower id. It never adds a load. Returns the final picks."""
    th = np.float32(bonus)
    ex = ex.copy()
    while True:
        union = sorted(set(ex[~ready[ex]].tolist()))
        if len(union) < 2:
            return ex
        B = _beats_matrix(base, union, th)
        in_row = np.stack([np.isin(union, ex[r]) for r in range(len(ex))])      # [R, U]
        uses = in_row.sum(axis=0)
        dropped = False
        for j in sorted(range(len(union)), key=lambda j: (uses[j], union[j])):
            rows_j = np.nonzero(in_row[:, j])[0]
            if not len(rows_j):
                continue
            cand = ~in_row[rows_j]                       # other members not already picked by that row
            cand[:, j] = False
            if not (cand & B[rows_j, :, j]).any(axis=1).all():
                continue
            mask = ready.copy()
            mask[[u for u in union if u != union[j]]] = True
            trial = reroute(base[rows_j], mask, bonus)
            # Accept only if the member really leaves every row that used it. An outranking member can instead push out
            # one of the row's resident picks; then nothing is saved and the member stays.
            if np.isin(union[j], trial):
                continue
            ex[rows_j] = trial
            dropped = True
            break
        if not dropped:
            return ex


def window_sharing(x: dict, bonus: float, threshold: float | None = None) -> dict:
    """Per generation pass against the recorded ready masks (P0), by window width (rows):
      - load picks: picks not on a GPU (not ready), and their distinct experts U (the pass's loads);
      - near_other_row (the plan's count): load picks p of a row r for which another row's load pick q (q not picked by
        r) lies within `threshold` of p in row r's scores (float32 base_r[q] + threshold outranks base_r[p]);
      - literal: route again with the same bonus on ready plus U. Every member of U keeps its bonus, so no row drops
        one: it saves no load by construction and only moves resident picks onto U (counted as a check);
      - greedy: drop members of U while every row picking one can take another member of U within the bonus
        (share_greedy). Loads saved, picks moved and router mass kept after it."""
    a = x['arrays']
    th = np.float32(bonus if threshold is None else threshold)
    gen = np.where(a['pass_generation'])[0]
    ready_all = bits_to_mask(a['ready_primary']) | bits_to_mask(a['ready_helper'])
    out: dict = {}
    for p in gen:
        quiet_tick()
        R = int(a['pass_rows'][p])
        r0 = int(a['pass_row0'][p])
        base, sig = a['row_base'][r0:r0 + R], a['row_sig'][r0:r0 + R]
        original = a['row_original'][r0:r0 + R].astype(np.int64)
        ready = ready_all[p]
        ex = reroute(base, ready, bonus).astype(np.int64)
        load = ~ready[ex]
        U = sorted(set(ex[load].tolist()))
        w = out.setdefault(R, dict(passes=0, rows=0, picks=0, load_picks=0, loads=0, near_other_row=0,
                                   literal_saved=0, literal_moved=0, greedy_saved=0, greedy_moved=0,
                                   mass_original=0.0, mass_before=0.0, mass_greedy=0.0))
        rows = np.arange(R)[:, None]
        w['passes'] += 1; w['rows'] += R; w['picks'] += R * K
        w['load_picks'] += int(load.sum()); w['loads'] += len(U)
        w['mass_original'] += float(sig[rows, original].sum()); w['mass_before'] += float(sig[rows, ex].sum())
        if R < 2 or len(U) < 2:
            w['mass_greedy'] += float(sig[rows, ex].sum())
            continue
        B = _beats_matrix(base, U, th)
        in_row = np.stack([np.isin(U, ex[r]) for r in range(R)])                   # [R, U]: row r picks member
        # Row r's load pick j has another row's load i (not picked by r) within the threshold.
        w['near_other_row'] += int((in_row & ((~in_row)[:, :, None] & B).any(axis=1)).sum())
        mask = ready.copy()
        mask[U] = True
        lit = reroute(base, mask, bonus).astype(np.int64)
        w['literal_saved'] += len(U) - len(set(lit[~ready[lit]].tolist()))
        w['literal_moved'] += sum(len(set(lit[r].tolist()) - set(ex[r].tolist())) for r in range(R))
        gr = share_greedy(base, ready, ex, bonus)
        w['greedy_saved'] += len(U) - len(set(gr[~ready[gr]].tolist()))
        w['greedy_moved'] += sum(len(set(gr[r].tolist()) - set(ex[r].tolist())) for r in range(R))
        w['mass_greedy'] += float(sig[rows, gr].sum())
    return dict(bonus=float(bonus), threshold=float(th), by_width={int(k): v for k, v in sorted(out.items())})


# ----------------------------------------------------------------------------------------------------------------------
# The grid and its report

_W: dict = {}


def _worker_init(origin, quiet):
    from cooperative_replay.policy import NativePolicy
    _W['bridge'] = NativePolicy()
    _W['origin'] = load_origin_scores(origin) if origin else {}
    _W['quiet'] = pathlib.Path(quiet) if quiet else None
    set_quiet(_W['quiet'])
    _W['extracts'] = {}


def _task(args):
    path, spec, bonus = args
    wait_quiet(_W['quiet'])
    x = _W['extracts'].get(path)
    if x is None:
        x = _W['extracts'][path] = load_extract(path)
    t0 = time.monotonic()
    if spec.recorded:
        res = run_recorded(x, bonus, _W['bridge'])
        out = summarize(res['tally'], x)
        out['rows_identical_to_recorded'] = float(res['same_rows'].mean())
        start, commit = forward_tokens(x)
        keep = start >= 128
        out['recorded_uploads_per_token'] = float(res['recorded']['uploads'][keep].sum() / max(1, commit[keep].sum()))
        out['recorded_cpu_jobs_per_token'] = float(res['recorded']['cpu_jobs'][keep].sum() / max(1, commit[keep].sum()))
    else:
        res = Sim(x, _W['origin'], spec, bonus, _W['bridge']).run()
        out = summarize(res['tally'], x)
        out['counts'] = res['counts']
        out['fidelity'] = fidelity(x, res)
    out['from256'] = summarize(res['tally'], x, first_token=256)
    out['seconds'] = time.monotonic() - t0
    return pathlib.Path(path).stem, spec.name, bonus, out


def _window_task(args):
    path, bonus, threshold, quiet = args
    set_quiet(pathlib.Path(quiet) if quiet else None)
    wait_quiet(_QUIET['lock'])
    return pathlib.Path(path).stem, bonus, window_sharing(load_extract(path), bonus, threshold)


def recorded_bonus(x: dict) -> float | None:
    """The request's routing bonus, from the capture manifest beside the JSONL."""
    src = pathlib.Path(x['meta']['source'])
    m = src.parent / 'manifest.json'
    if not m.exists():
        return None
    for r in json.loads(m.read_text()).get('requests', []):
        tune = r.get('request', {}).get('strata_tune', {})
        if 'routing_prior_bonus' in tune:
            return float(tune['routing_prior_bonus'])
    return None


def run_grid(paths, origin, bonuses, policies, workers=8, quiet=QUIET):
    import multiprocessing as mp
    tasks = [(str(p), s, b) for p in paths for b in bonuses for s in policies]
    results: dict = {}
    t0 = time.monotonic()
    with mp.get_context('fork').Pool(workers, initializer=_worker_init, initargs=(str(origin), str(quiet) if quiet else None)) as pool:
        for i, (case, name, bonus, out) in enumerate(pool.imap_unordered(_task, tasks), 1):
            results.setdefault(case, {}).setdefault(name, {})[f'{bonus:.2f}'] = out
            print(f'[{time.monotonic() - t0:6.0f} s] {i}/{len(tasks)} {case} {name} {bonus:.2f} ({out["seconds"]:.1f} s)', file=sys.stderr, flush=True)
    return results


def equal_weight(results: dict, policy: str, bonus: str, key: str) -> float:
    vals = [results[c][policy][bonus][key] for c in results if policy in results[c] and bonus in results[c][policy]]
    return float(np.mean(vals)) if vals else float('nan')


REPORT_COLUMNS = [('moved_share', 'moved off top 4', '{:.1%}'), ('mass_kept', 'router mass kept', '{:.2%}'),
                  ('on_primary_per_token', '5090 res/tok', '{:.1f}'), ('on_helper_per_token', '3090 res/tok', '{:.1f}'),
                  ('on_staged_per_token', 'staged/tok', '{:.2f}'), ('need_load_per_token', 'need load/tok', '{:.2f}'),
                  ('uploads_per_token', 'uploads/tok', '{:.2f}'), ('cpu_jobs_per_token', 'CPU jobs/tok', '{:.2f}'),
                  ('own_not_on_gpu_per_token', 'own picks off GPU/tok', '{:.2f}'),
                  ('primary_exchanges_per_token', '5090 exch/tok', '{:.2f}'), ('helper_exchanges_per_token', '3090 exch/tok', '{:.2f}')]


def report(results: dict, policies: list[str], bonuses: list[str]) -> str:
    lines = []
    cases = sorted(results)
    lines.append('Equal-weight mean over requests: ' + ', '.join(cases) + '. Per generated token from token 128.')
    for b in bonuses:
        lines.append('')
        lines.append(f'bonus {b}')
        head = f"{'policy':22s}" + ''.join(f'{label:>22s}' for _, label, _ in REPORT_COLUMNS) + f"{'need load vs P1':>18s}{'own off GPU vs P1':>20s}"
        lines.append(head)
        p1_load = equal_weight(results, 'P1', b, 'need_load_per_token')
        p1_own = equal_weight(results, 'P1', b, 'own_not_on_gpu_per_token')
        for p in policies:
            if not any(p in results[c] and b in results[c][p] for c in cases):
                continue
            row = f'{p:22s}'
            for key, _, fmt in REPORT_COLUMNS:
                row += f'{fmt.format(equal_weight(results, p, b, key)):>22s}'
            nl, no = equal_weight(results, p, b, 'need_load_per_token'), equal_weight(results, p, b, 'own_not_on_gpu_per_token')
            row += f'{nl / p1_load - 1:>+18.1%}{no / p1_own - 1:>+20.1%}'
            lines.append(row)
    return '\n'.join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    e = sub.add_parser('extract', help='compact numpy extract of one detailed-capture JSONL')
    e.add_argument('--capture', required=True)
    e.add_argument('--out', required=True)
    r = sub.add_parser('run', help='the policy x bonus grid over extracts')
    r.add_argument('--extract', nargs='+', required=True)
    r.add_argument('--origin', required=True, help="origin snapshot JSON (the load's restored state, with its scores)")
    r.add_argument('--bonus', nargs='+', type=float, default=[0.0, 0.10, 0.20, 0.25, 0.30, 0.40])
    r.add_argument('--policies', nargs='*', default=None)
    r.add_argument('--set', choices=('grid', 'decay'), default='grid', help='the policy list: the grid (P0-P4) or the decay screen')
    r.add_argument('--workers', type=int, default=8)
    r.add_argument('--out', required=True)
    wnd = sub.add_parser('windows', help="item 6: picks of verify windows that another row's load could replace")
    wnd.add_argument('--extract', nargs='+', required=True)
    wnd.add_argument('--bonus', nargs='+', type=float, default=[0.20])
    wnd.add_argument('--threshold', type=float, default=None, help='default: the bonus')
    wnd.add_argument('--out', required=True)
    wnd.add_argument('--workers', type=int, default=8)
    for p in (e, r, wnd):
        p.add_argument('--quiet-lock', default=str(QUIET), help='wait while this file exists (empty string: never)')
    args = ap.parse_args(argv)
    quiet = pathlib.Path(args.quiet_lock) if args.quiet_lock else None
    set_quiet(quiet)
    wait_quiet(quiet)
    if args.cmd == 'extract':
        save_extract(extract_capture(args.capture), args.out)
        return 0
    if args.cmd == 'windows':
        import multiprocessing as mp
        tasks = [(str(pth), b, args.threshold, str(quiet) if quiet else None) for pth in args.extract for b in args.bonus]
        doc: dict = {}
        t0 = time.monotonic()
        with mp.get_context('fork').Pool(min(len(tasks), args.workers)) as pool:
            for i, (case, b, res) in enumerate(pool.imap_unordered(_window_task, tasks), 1):
                doc.setdefault(case, {})[f'{b:.2f}'] = res
                print(f'[{time.monotonic() - t0:6.0f} s] {i}/{len(tasks)} {case} {b:.2f}', file=sys.stderr, flush=True)
        doc = {c: dict(sorted(v.items())) for c, v in sorted(doc.items())}
        pathlib.Path(args.out).write_text(json.dumps(doc, indent=1))
        lines = ['width | passes | load picks | loads | near another row (plan count) | literal re-route: loads saved, picks moved | greedy: loads saved, picks moved | mass kept before -> greedy']
        for case, per in doc.items():
            for b, res in per.items():
                lines.append(f'{case} bonus {b} (threshold {res["threshold"]:.2f})')
                for width, v in res['by_width'].items():
                    lines.append(f"  {width:2d} | {v['passes']:6d} | {v['load_picks']:6d} | {v['loads']:6d} | {v['near_other_row']:6d} "
                                 f"({v['near_other_row'] / max(1, v['load_picks']):.1%}) | {v['literal_saved']:4d}, {v['literal_moved']:5d} | "
                                 f"{v['greedy_saved']:5d} ({v['greedy_saved'] / max(1, v['loads']):.1%}), {v['greedy_moved']:5d} | "
                                 f"{v['mass_before'] / max(1e-9, v['mass_original']):.2%} -> {v['mass_greedy'] / max(1e-9, v['mass_original']):.2%}")
        text = '\n'.join(lines)
        pathlib.Path(args.out).with_suffix('.txt').write_text(text + '\n')
        print(text)
        return 0
    specs = default_policies() + decay_policies() if args.set == 'grid' and args.policies else (decay_policies() if args.set == 'decay' else default_policies())
    if args.policies:
        specs = [s for s in specs if s.name in set(args.policies)]
    specs = list({s.name: s for s in specs}.values())
    t0 = time.monotonic()
    results = run_grid(args.extract, args.origin, args.bonus, specs, args.workers, quiet)
    out = pathlib.Path(args.out)
    doc = dict(tool='tools/mm3/native_reroute_replay.py', extracts=args.extract, origin=args.origin, bonuses=args.bonus,
               policies=[s.describe() for s in specs], results=results, seconds=time.monotonic() - t0)
    out.write_text(json.dumps(doc, indent=1))
    text = report(results, [s.name for s in specs], [f'{b:.2f}' for b in args.bonus])
    out.with_suffix('.txt').write_text(text + '\n')
    print(text)
    return 0


if __name__ == '__main__':
    sys.path.insert(0, str(HERE))
    raise SystemExit(main())
