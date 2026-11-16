#!/usr/bin/env python3
"""A stored reference state for benchmark engine starts: save one, force a start into it, and read what a start is in.

The user (2026-11-14): "force engine to start from the same place for each benchmark run. ... We should not need to repeat
the control every time. Speed test runs should be comparable to a common control, regardless of if they come from the same
engine start."

Two starts of one binary and one set of start options differ, before any request, in two things (read from the engine and
its logs, 2026-11-14): each start measures its own lane costs, twice - at load and inside its first request - and the 5090's
resident count follows the VRAM free at load. The lane costs decide in every pass whether a host expert runs on the CPU lane
or is copied to the 5090, and they set the 5090's exchange threshold; on 2026-11-14's starts they ranged from 0.441 to
0.593 ms for one expert on the CPU lane. The three cards' rosters follow from the counts and the placement profile.

A reference state is the engine's learned checkpoint (PLACEMENT snapshot) made durable, with the 4080's roster:

    geometry, arithmetic       the model's and the 5090's, which a start must share to take the state
    primary, helper, third     the 5090's, the 3090's and the 4080's rosters (the host bank holds the rest)
    scores, admission_credits  the demand scores the swaps read, and the swap credits
    lane, batch_staging_calibration, policy
                               the lane costs and the grouped staging table a start otherwise measures, with the flag that
                               says they are final (so the first request does not measure them again), and the residency policy

Its identity is the SHA-256 of that content in canonical JSON. NAME.json holds the content, the identity and where it came
from; NAME.mm3ref beside it is the same content as the text file the engine reads (PLACEMENT reference FILE).

    native_reference_state.py save OUT.json [--note TEXT]     the engine's state now (it takes a PLACEMENT snapshot, which
                                                              also becomes the engine's in-process checkpoint)
    native_reference_state.py apply STATE.json [--own OUT.json] [--record FILE]
                                                              force the engine that is up into STATE. --own first saves the
                                                              start's own state (its measured lane costs) as a state file
    native_reference_state.py check STATE.json                is the engine in STATE's rosters now (PLACEMENT inspect; nothing
                                                              changes); exit 1 when not
    native_reference_state.py show STATE.json [OTHER.json]    what a state holds; with OTHER, what differs

What a reference state cannot force: within a request, when a staging copy, an exchange or a swap is seen complete decides
the lane of later picks; and each lane's real speed on that start. Both are measured instead (native_stored_control.py).
Never starts or stops an engine. Standard library only."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import os
import pathlib
import sys
import urllib.error
import urllib.request
import zoneinfo

FORMAT = 'mm3-reference-state/1'
LEARNED_FIELDS = ('geometry', 'arithmetic', 'primary_adaptive', 'policy', 'lane', 'scores', 'adaptive_main')   # (native_calibrate.learned_state_signature)
POLICY_KEYS = ('legacy', 'horizon', 'max_exchanges', 'service_saved_ms', 'h2d_ms', 'd2h_ms', 'interference_ms', 'margin_ms')
COST_ROWS = ('cpu_extra_ms', 'gpu_extra_ms', 'helper_extra_ms', 'gpu_base_ms')


def now_pacific() -> str:
    return datetime.datetime.now(zoneinfo.ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M:%S %Z')


def post(base: str, endpoint: str, body: dict, timeout: float = 900) -> dict:
    request = urllib.request.Request(base.rstrip('/') + endpoint, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'{endpoint} {body.get("action")}: HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc


def canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


# ---- a state's content -----------------------------------------------------------------------------------------------------------
def content_from_reply(reply: dict, third: list | None = None) -> dict:
    """A state's content from a PLACEMENT snapshot, restore or reference reply. The rosters keep the reply's order (the
    5090's and the 3090's slot order). `third`: the 4080's roster when the reply's placement report has none (a binary
    before "third_roster")."""
    if reply.get('scope') != 'in-process only' or reply.get('staging') != 'cleared at joined boundary':
        raise ValueError('not a learned-state reply (PLACEMENT snapshot, restore or reference)')
    placement = reply.get('placement') or {}
    if not placement.get('ownership_complete'):
        raise ValueError('the reply does not show complete ownership')
    missing = [k for k in LEARNED_FIELDS + ('admission_credits',) if k not in reply]
    if missing:
        raise ValueError(f'the reply lacks {", ".join(missing)}')
    if third is None:
        if 'third_roster' not in placement:
            raise ValueError('the reply has no third_roster (an engine binary before PLACEMENT reference): give the 4080\'s roster')
        third = placement['third_roster']
    helper = sorted(placement['helper_ids'], key=lambda r: r[2])
    content = dict(format=FORMAT, geometry=list(reply['geometry']), arithmetic=reply['arithmetic'], primary_adaptive=bool(reply['primary_adaptive']),
                   policy={k: reply['policy'][k] for k in POLICY_KEYS}, lane=dict(reply['lane']),
                   batch_staging_calibration=reply.get('batch_staging_calibration'), admission_credits=list(reply['admission_credits']),
                   primary=[[r[0], r[1]] for r in placement['primary_ids']], helper=[[r[0], r[1]] for r in helper],
                   third=[[r[0], r[1]] for r in third], scores=[list(r) for r in reply['scores']],
                   adaptive_main=[list(r) for r in reply['adaptive_main']])
    if 'checkpoint_version' in reply:
        content['checkpoint_version'] = reply['checkpoint_version']
    validate(content)
    return content


def validate(content: dict) -> None:
    if content.get('format') != FORMAT:
        raise ValueError(f"not a {FORMAT} state")
    layers, experts = content['geometry'][0], content['geometry'][1]
    seen = {}
    for card in ('primary', 'helper', 'third'):
        for li, e in content[card]:
            if not (0 <= li < layers and 0 <= e < experts):
                raise ValueError(f'{card}: expert ({li}, {e}) is outside the model')
            if (li, e) in seen:
                raise ValueError(f'expert ({li}, {e}) is on {seen[(li, e)]} and on {card}')
            seen[(li, e)] = card
    lane = content['lane']
    for k in ('cpu1', 'cpu_add', 'copy', 'cpu_price_scale', 'calibrated', 'calibration_arithmetic', 'retime'):
        if k not in lane:
            raise ValueError(f'lane lacks {k}')
    if len(lane['retime']) != 12:
        raise ValueError('lane retime is not twelve numbers')
    cal = content['batch_staging_calibration']
    if cal is not None:
        for k in COST_ROWS:
            if len(cal.get(k) or []) != 17:
                raise ValueError(f'the grouped staging table lacks {k}')
        for k in ('cpu_base_ms', 'helper_base_ms', 'h2d_ms', 'd2h_ms', 'arithmetic'):
            if k not in cal:
                raise ValueError(f'the grouped staging table lacks {k}')


def ensure_reusable(content: dict) -> None:
    """A bootstrap inventory is inspectable, but cannot become a benchmark origin."""
    validate(content)
    if content.get('checkpoint_version') != 2:
        raise ValueError('reusable reference requires checkpoint version 2 (all rosters and transient reset)')
    lane, cal = content['lane'], content['batch_staging_calibration']
    if not lane['calibrated'] or not cal:
        raise ValueError('reusable reference requires final lane and grouped calibration')
    if lane['calibration_arithmetic'] != content['arithmetic'] or cal['arithmetic'] != content['arithmetic']:
        raise ValueError('reference calibration arithmetic does not match the model')
    values = [lane[k] for k in ('cpu1', 'cpu_add', 'copy', 'cpu_price_scale')]
    values += [cal[k] for k in ('cpu_base_ms', 'helper_base_ms', 'h2d_ms', 'd2h_ms')]
    values += [v for k in COST_ROWS for v in cal[k]]
    if any(not math.isfinite(float(v)) or v < 0 for v in values) or lane['cpu1'] <= 0 or lane['copy'] <= 0:
        raise ValueError('reference calibration has invalid measured costs')


def identity(content: dict) -> str:
    return hashlib.sha256(canonical(content)).hexdigest()


def learned_signature(content: dict) -> str:
    """The signature native_calibrate.learned_state_signature gives a restore reply of an engine in this state: what the
    ledger's per-step restore is checked against. Version 2 includes the 4080; legacy signatures remain readable."""
    state = {k: content[k] for k in LEARNED_FIELDS}
    state['batch_staging_calibration'] = content['batch_staging_calibration']
    state['admission_credits'] = content['admission_credits']
    state['primary_ids'] = sorted(r[:2] for r in content['primary'])
    state['helper_ids'] = sorted(r[:2] for r in content['helper'])
    if content.get('checkpoint_version', 1) == 2:
        state['checkpoint_version'] = 2
        state['third_ids'] = sorted(r[:2] for r in content['third'])
    return hashlib.sha256(canonical(state)).hexdigest()


def slot_order_hash(report: dict) -> str:
    """The physical order experts sit in on the 5090 (a layer's slots in order) and the 3090 (its slots), from a placement
    report. Not part of a state's identity: it follows a start's history, and decides only ties between equal scores."""
    rows = [[r[0], r[1]] for r in report['primary_ids']] + [['h', r[2], r[0], r[1]] for r in sorted(report['helper_ids'], key=lambda r: r[2])]
    return hashlib.sha256(canonical(rows)).hexdigest()


# ---- the engine's text file ------------------------------------------------------------------------------------------------------
def _g(x) -> str:
    """A number as the engine prints it (ostream precision 17), so a parsed value is written back unchanged."""
    if isinstance(x, bool):
        return '1' if x else '0'
    if isinstance(x, int):
        return str(x)
    return format(float(x), '.17g')


def _compact(obj) -> str:
    """JSON as the engine wrote it: no spaces, numbers at 17 digits."""
    if isinstance(obj, dict):
        return '{' + ','.join(json.dumps(str(k)) + ':' + _compact(v) for k, v in obj.items()) + '}'
    if isinstance(obj, (list, tuple)):
        return '[' + ','.join(_compact(v) for v in obj) + ']'
    if isinstance(obj, bool):
        return 'true' if obj else 'false'
    if obj is None:
        return 'null'
    if isinstance(obj, (int, float)):
        return _g(obj)
    return json.dumps(str(obj))


def engine_text(content: dict, ident: str | None = None) -> str:
    """The content as the text file the engine reads (src/mm3/mm3_reference_state.hpp)."""
    validate(content)
    ident = ident or identity(content)
    layers, experts, dim, ff, cap = content['geometry']
    lane, pol, cal = content['lane'], content['policy'], content['batch_staging_calibration']
    word = lambda s: s if s else '-'
    ids = lambda rows: ' '.join(str(li * experts + e) for li, e in rows)
    out = ['MM3_REFERENCE_STATE 1', f'ID {ident}', f'GEOMETRY {layers} {experts} {dim} {ff} {cap}', f"ARITHMETIC {word(content['arithmetic'])}",
           f"CALIBRATION_ARITHMETIC {word(lane['calibration_arithmetic'])}", f"CALIBRATED {_g(bool(lane['calibrated']))}",
           f"PRIMARY_ADAPTIVE {_g(bool(content['primary_adaptive']))}", 'CREDITS ' + ' '.join(_g(x) for x in content['admission_credits']),
           'POLICY ' + ' '.join(_g(pol[k]) for k in POLICY_KEYS),
           'LANE ' + ' '.join(_g(lane[k]) for k in ('cpu1', 'cpu_add', 'copy', 'cpu_price_scale')), 'RETIME ' + ' '.join(_g(x) for x in lane['retime'])]
    if cal is None:
        zero = ' '.join(['0'] * 17)
        out += ['COSTS 0 0 0 0 0'] + [f'COSTS_{k[:-3].upper()} {zero}' for k in COST_ROWS] + ['BATCH_ARITHMETIC -', 'BATCH_CALIBRATION 0', '']
    else:
        text = _compact(cal)
        out += ['COSTS 1 ' + ' '.join(_g(cal[k]) for k in ('cpu_base_ms', 'helper_base_ms', 'h2d_ms', 'd2h_ms'))]
        out += [f'COSTS_{k[:-3].upper()} ' + ' '.join(_g(x) for x in cal[k]) for k in COST_ROWS]
        out += [f"BATCH_ARITHMETIC {word(cal['arithmetic'])}", f'BATCH_CALIBRATION {len(text.encode())}', text]
    for name in ('primary', 'helper', 'third'):
        out.append(f'{name.upper()} {len(content[name])} {ids(content[name])}'.rstrip())
    out.append(f"SCORES {len(content['scores'])} " + ' '.join(f'{li} {e} {_g(v)}' for li, e, v in content['scores']))
    out.append(f"ADAPTIVE_MAIN {len(content['adaptive_main'])} {ids(content['adaptive_main'])}".rstrip())
    out.append('END')
    return '\n'.join(s.rstrip() for s in out) + '\n'


# ---- files -----------------------------------------------------------------------------------------------------------------------
def text_path(state_path: pathlib.Path) -> pathlib.Path:
    return state_path.with_suffix('.mm3ref')


def write_state(path: pathlib.Path, content: dict, provenance: dict, *, bootstrap: bool = False) -> dict:
    if not bootstrap:
        ensure_reusable(content)
    ident = identity(content)
    doc = dict(format=FORMAT, sha256=ident, learned_signature=learned_signature(content), provenance=provenance, content=content, purpose="bootstrap inventory" if bootstrap else "reusable benchmark")
    path = pathlib.Path(path)
    if path.exists() or text_path(path).exists():
        raise FileExistsError(f'{path} or {text_path(path)} exists: a stored state is not overwritten')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1) + '\n')
    text_path(path).write_text(engine_text(content, ident))
    return doc


def read_state(path: pathlib.Path) -> dict:
    """A stored state, checked: its identity is its content's, and the engine's text file beside it is that content's."""
    path = pathlib.Path(path)
    doc = json.loads(path.read_text())
    validate(doc.get('content') or {})
    if doc.get('sha256') != identity(doc['content']):
        raise ValueError(f'{path}: the content does not have the stated identity {doc.get("sha256")}')
    if doc.get('learned_signature') != learned_signature(doc['content']):
        raise ValueError(f'{path}: the stated learned signature is not the content\'s')
    text = text_path(path)
    if not text.exists() or text.read_text() != engine_text(doc['content'], doc['sha256']):
        raise ValueError(f'{text}: missing, or not the engine file of {path.name}')
    return doc


# ---- the engine that is up -------------------------------------------------------------------------------------------------------
def engine_facts(pid: int, base_url: str, timeout: float = 30) -> dict:
    """What identifies the engine process: its start (pid and start time), its binary (path and SHA-256), its start options,
    and what the server reports about it. The fields native_usability_matrix.engine_facts keeps, read the same way."""
    facts: dict = dict(pid=pid)
    try:
        exe = os.readlink(f'/proc/{pid}/exe')
        stat = pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        facts.update(exe=exe, exe_sha256=hashlib.sha256(pathlib.Path(f'/proc/{pid}/exe').read_bytes()).hexdigest(), started=int(stat[19]),
                     args=[x.decode(errors='replace') for x in pathlib.Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')[1:] if x])
    except (OSError, IndexError, ValueError) as exc:
        facts['process_error'] = f'{type(exc).__name__}: {exc}'
    try:
        with urllib.request.urlopen(base_url.rstrip('/') + '/metrics', timeout=timeout) as response:
            info = json.load(response).get('engine') or {}
        facts['info'] = {k: v for k, v in info.items() if v is None or isinstance(v, (str, int, float, bool))}
    except Exception as exc:   # (a record, not a step: a failed read is kept, not raised)
        facts['info_error'] = f'{type(exc).__name__}: {exc}'
    return facts


def measured_costs(content: dict) -> dict:
    """The lane costs a state carries, in the engine's terms (ms): what a start measured for itself."""
    lane, cal = content['lane'], content['batch_staging_calibration'] or {}
    return dict(cpu_one_expert_ms=lane['cpu1'], cpu_each_more_ms=lane['cpu_add'], primary_upload_and_compute_ms=lane['copy'],
                calibrated_after_prefill=bool(lane['calibrated']), h2d_ms=cal.get('h2d_ms'), d2h_ms=cal.get('d2h_ms'),
                cpu_base_ms=cal.get('cpu_base_ms'), helper_base_ms=cal.get('helper_base_ms'))


def save(base: str, out: pathlib.Path, note: str = '', timeout: float = 900, *, bootstrap: bool = False) -> dict:
    reply = post(base, '/v1/minimax/placement', {'action': 'snapshot'}, timeout)
    content = content_from_reply(reply)
    provenance = dict(created=now_pacific(), how='PLACEMENT snapshot of the engine that was up', note=note, engine=engine_facts(reply['engine_pid'], base),
                      slot_order_sha256=slot_order_hash(reply['placement']), snapshot_forward=reply.get('snapshot_forward'),
                      snapshot_generation_row=reply.get('snapshot_generation_row'))
    return write_state(out, content, provenance, bootstrap=bootstrap)


def apply(base: str, state_path: pathlib.Path, own: pathlib.Path | None = None, timeout: float = 900) -> dict:
    """Forces the engine that is up into the stored state and checks that it took it: the engine's reply names the state's
    identity, its learned signature is the state's, and the 4080 holds the state's roster. With `own`, the start's own state
    is saved first as a state file (its measured lane costs, its rosters). Returns the record of what was done."""
    doc = read_state(state_path)
    content = doc['content']
    ensure_reusable(content)
    record: dict = dict(checkpoint_version=2, applied=now_pacific(), state=str(pathlib.Path(state_path).resolve()), sha256=doc['sha256'], learned_signature=doc['learned_signature'])
    if own is not None:
        mine = save(base, own, note=f'this start\'s own state, saved before {pathlib.Path(state_path).name} was applied', timeout=timeout, bootstrap=True)
        record['own'] = dict(path=str(pathlib.Path(own).resolve()), sha256=mine['sha256'], costs=measured_costs(mine['content']),
                             counts={k: len(mine['content'][k]) for k in ('primary', 'helper', 'third')},
                             same_rosters={k: sorted(mine['content'][k]) == sorted(content[k]) for k in ('primary', 'helper', 'third')})
    reply = post(base, '/v1/minimax/placement', {'action': 'reference', 'state_path': str(text_path(pathlib.Path(state_path)).resolve())}, timeout)
    errors = []
    if reply.get('reference_state') != doc['sha256']:
        errors.append(f"the engine names the state {reply.get('reference_state')}, not {doc['sha256']}")
    got = content_from_reply(reply)
    if learned_signature(got) != doc['learned_signature']:
        errors.append('the engine\'s learned state after the command is not the stored one: ' + ', '.join(differences(content, got)))
    if sorted(got['third']) != sorted(content['third']):
        errors.append('the 4080 does not hold the stored roster')
    record.update(engine=engine_facts(reply['engine_pid'], base), costs=measured_costs(content),
                  counts={k: len(content[k]) for k in ('primary', 'helper', 'third')},
                  slot_order_sha256=slot_order_hash(reply['placement']), stored_slot_order_sha256=(doc.get('provenance') or {}).get('slot_order_sha256'),
                  moved=dict(primary=reply['placement'].get('primary_changed'), helper=reply['placement'].get('helper_changed'), third=reply.get('third_changed', 0)),
                  duration_ms=reply.get('restore_duration_ms', reply['placement'].get('duration_ms')), errors=errors)
    if errors:
        raise RuntimeError('the reference state was not taken: ' + '; '.join(errors))
    return record


def check(base: str, state_path: pathlib.Path, timeout: float = 120) -> list[str]:
    """What the engine's rosters differ from the state's in, from PLACEMENT inspect (nothing changes). An engine before
    "third_roster" cannot show the 4080's."""
    content = read_state(state_path)['content']
    report = post(base, '/v1/minimax/placement', {'action': 'inspect'}, timeout)
    out = []
    for card, key, cut in (('primary', 'primary_ids', 2), ('helper', 'helper_ids', 2), ('third', 'third_roster', 2)):
        if key not in report:
            out.append(f'{card}: the engine does not report it')
            continue
        have, want = {tuple(r[:cut]) for r in report[key]}, {tuple(r) for r in content[card]}
        if have != want:
            out.append(f'{card}: {len(want - have)} of the state\'s {len(want)} experts are not there ({len(have)} held)')
    return out


def differences(a: dict, b: dict) -> list[str]:
    """What two states' contents differ in, in a reader's terms."""
    out = []
    for k in ('geometry', 'arithmetic', 'primary_adaptive', 'policy', 'admission_credits'):
        if a[k] != b[k]:
            out.append(f'{k} ({a[k]} / {b[k]})')
    for card in ('primary', 'helper', 'third'):
        sa, sb = {tuple(r) for r in a[card]}, {tuple(r) for r in b[card]}
        if sa != sb:
            out.append(f'{card} roster ({len(sa)} / {len(sb)} experts, {len(sa - sb)} only in the first)')
    for k in ('cpu1', 'cpu_add', 'copy', 'cpu_price_scale', 'calibrated', 'calibration_arithmetic', 'retime'):
        if a['lane'][k] != b['lane'][k]:
            out.append(f"lane {k} ({a['lane'][k]} / {b['lane'][k]})")
    if a['batch_staging_calibration'] != b['batch_staging_calibration']:
        out.append('grouped staging table')
    if a['scores'] != b['scores']:
        out.append(f"demand scores ({len(a['scores'])} / {len(b['scores'])} nonzero)")
    if a['adaptive_main'] != b['adaptive_main']:
        out.append('adaptive marks')
    return out


def show(doc: dict) -> list[str]:
    c, p = doc['content'], doc.get('provenance') or {}
    costs = measured_costs(c)
    e = p.get('engine') or {}
    return [f"reference state {doc['sha256']}", f"  saved {p.get('created')}: {p.get('how')}" + (f" ({p['note']})" if p.get('note') else ''),
            f"  engine: {pathlib.Path(str(e.get('exe') or '?')).name} (sha256 {str(e.get('exe_sha256'))[:12]}), options {' '.join(e.get('args') or [])}",
            f"  experts held: the 5090 {len(c['primary'])}, the 3090 {len(c['helper'])}, the 4080 {len(c['third'])}; demand scores {len(c['scores'])} nonzero; "
            f"admission credits {c['admission_credits']}",
            f"  lane costs (as that start measured them, ms): CPU lane one expert {costs['cpu_one_expert_ms']:.3f}, each more {costs['cpu_each_more_ms']:.3f}; "
            f"5090 upload and compute {costs['primary_upload_and_compute_ms']:.3f}; grouped table h2d {costs['h2d_ms']}, d2h {costs['d2h_ms']}; "
            f"measured after a real prefill: {'yes' if costs['calibrated_after_prefill'] else 'no (the first request measures again)'}",
            f"  learned signature (what each restore is checked against): {doc['learned_signature']}"]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--base-url', default='http://127.0.0.1:8097')
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('save'); a.add_argument('out', type=pathlib.Path); a.add_argument('--note', default='')
    a = sub.add_parser('apply'); a.add_argument('state', type=pathlib.Path); a.add_argument('--own', type=pathlib.Path); a.add_argument('--record', type=pathlib.Path)
    a = sub.add_parser('check'); a.add_argument('state', type=pathlib.Path)
    a = sub.add_parser('show'); a.add_argument('state', type=pathlib.Path); a.add_argument('other', type=pathlib.Path, nargs='?')
    args = ap.parse_args(argv)
    if args.cmd == 'save':
        doc = save(args.base_url, args.out, args.note)
        print('\n'.join(show(doc)))
        print(f'  written: {args.out} and {text_path(args.out)}')
        return 0
    if args.cmd == 'apply':
        record = apply(args.base_url, args.state, args.own)
        if args.record:
            args.record.write_text(json.dumps(record, indent=1) + '\n')
        print(f"the engine (pid {record['engine'].get('pid')}) took reference state {record['sha256'][:16]}: moved {record['moved']['primary']} experts on the 5090, "
              f"{record['moved']['helper']} on the 3090, {record['moved']['third']} on the 4080")
        if record.get('own'):
            own = record['own']
            print(f"  this start's own state {own['sha256'][:16]} ({own['path']}): CPU lane one expert {own['costs']['cpu_one_expert_ms']:.3f} ms against the "
                  f"reference's {record['costs']['cpu_one_expert_ms']:.3f}; same rosters as the reference: {own['same_rosters']}")
        return 0
    if args.cmd == 'check':
        diffs = check(args.base_url, args.state)
        print('the engine holds the state\'s rosters' if not diffs else 'the engine is NOT in the state\'s rosters: ' + '; '.join(diffs))
        return 1 if diffs else 0
    doc = read_state(args.state)
    print('\n'.join(show(doc)))
    if args.other:
        other = read_state(args.other)
        print('\n'.join(show(other)))
        diffs = differences(doc['content'], other['content'])
        print('the two states differ in: ' + ('; '.join(diffs) if diffs else 'nothing'))
    return 0


if __name__ == '__main__':
    sys.exit(main())
