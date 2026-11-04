"""Capture control, source provenance and honest coverage checks for native screens."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import pathlib
import re
import subprocess
import tarfile
import time
import uuid

ROOT = pathlib.Path(__file__).resolve().parents[2]
ID = re.compile(r'^[A-Za-z0-9_.-]{1,160}$')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''): h.update(block)
    return h.hexdigest()

def args(ap):
    ap.add_argument('--capture-mode', choices=('timeline', 'detailed', 'off'), default=None,
                    help='defaults to timeline when --capture-control is supplied; otherwise explicitly uncaptured')
    ap.add_argument('--capture-control', type=pathlib.Path)
    ap.add_argument('--experiment-id')
    ap.add_argument('--run-id')
    ap.add_argument('--plan-item', default=None)
    ap.add_argument('--nsys-report', default=None, help='expected Nsight report path; verify after collection')
    ap.add_argument('--profiling-session', default=None, help='Nsight session and trace configuration')
    ap.add_argument('--model-revision', default=None)
    ap.add_argument('--tokenizer-revision', default=None)
    ap.add_argument('--capture-conditioning', default=None, help='prior requests, cache state and hardware conditioning')

def executable_info(pid, control):
    executable = {'pid': pid, 'status': 'unavailable'}
    if pid:
        try:
            p = pathlib.Path('/proc') / str(pid)
            exe = p / 'exe'
            executable = {'pid': pid, 'path': os.readlink(exe), 'sha256': sha(exe),
                          'cmdline': p.joinpath('cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace').strip()}
            env = p.joinpath('environ').read_bytes().split(b'\0')
            executable['capture_control_in_environment'] = (b'MM3_CAPTURE_CONTROL=' + os.fsencode(str(control.resolve()))) in env if control else False
        except OSError as exc: executable['error'] = str(exc)
    return executable

def bind_executable(c, pid):
    c['manifest']['executable'] = executable_info(pid, c['control'])
    note = 'Engine capture control environment not verified; capture may be missing.'
    if c['manifest']['executable'].get('capture_control_in_environment'):
        c['manifest']['limitations'] = [x for x in c['manifest']['limitations'] if x != note]
    save(c)

def init(a, output, engine_pid=None):
    mode = a.capture_mode or ('timeline' if a.capture_control else 'off')
    if mode != 'off' and not a.capture_control: raise ValueError('capture requires --capture-control')
    experiment = a.experiment_id or 'exp-' + uuid.uuid4().hex
    run = a.run_id or 'run-' + uuid.uuid4().hex
    if not ID.fullmatch(experiment) or not ID.fullmatch(run): raise ValueError('invalid experiment or run ID')
    directory = output.parent.resolve() / (output.name + '.capture')
    if ' ' in str(directory): raise ValueError('capture output path cannot contain spaces')
    directory.mkdir(exist_ok=False)
    def git(*cmd): return subprocess.check_output(['git', *cmd], cwd=ROOT)
    diff = directory / 'source.diff'
    diff.write_bytes(git('diff', '--binary', 'HEAD', '--'))
    names = git('ls-files', '--others', '--exclude-standard', '-z').split(b'\0')
    archive = directory / 'untracked.tar'
    archived, omitted = [], []
    with tarfile.open(archive, 'w') as tar:
        for name in names:
            if name:
                path = ROOT / os.fsdecode(name)
                if path.is_file() and not path.is_symlink() and path.stat().st_size <= 16 * 1024 * 1024:
                    tar.add(path, arcname=os.fsdecode(name), recursive=False); archived.append(os.fsdecode(name))
                else: omitted.append(os.fsdecode(name))
    executable = executable_info(engine_pid, a.capture_control)
    manifest = {'schema': 1, 'experiment_id': experiment, 'run_id': run, 'plan_item': a.plan_item,
                'mode': mode, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
                'source': {'commit': git('rev-parse', 'HEAD').decode().strip(), 'diff': str(diff), 'diff_sha256': sha(diff),
                           'untracked_archive': str(archive), 'untracked_sha256': sha(archive), 'untracked_archived': archived, 'untracked_omitted': omitted},
                'executable': executable, 'requests': [], 'control_operations': [], 'limitations': [],
                'model_revision': a.model_revision, 'tokenizer_revision': a.tokenizer_revision,
                'conditioning': a.capture_conditioning, 'nsys_report': a.nsys_report,
                'profiling_session': a.profiling_session,
                'nsight_validation': {'status': 'pending' if a.nsys_report else 'missing', 'note': 'A declared path does not establish report existence or GPU/trace coverage; inspect the completed report.'},
                'capture_cost': 'buffered timeline on ranked run' if mode == 'timeline' else ('detailed companion diagnostic; do not rank' if mode == 'detailed' else 'uncaptured'),
                'missing_fields': ['model revision', 'tokenizer revision', 'prior request history', 'hardware conditioning', 'Nsight report']}
    manifest['missing_fields'] = [name for name, value in (('model revision', a.model_revision), ('tokenizer revision', a.tokenizer_revision), ('prior request history and hardware conditioning', a.capture_conditioning), ('Nsight report', a.nsys_report)) if not value]
    if mode == 'off': manifest['limitations'].append('Engine timeline and NVTX unavailable; SSE is frontend reception, not engine emission. Historical binaries require external tracing and cannot supply instrumented engine events.')
    if mode != 'off' and not executable.get('capture_control_in_environment'):
        manifest['limitations'].append('Engine capture control environment not verified; capture may be missing.')
    c = {'manifest': manifest, 'directory': directory, 'control': a.capture_control.resolve() if a.capture_control else None}
    save(c)
    return c

def save(c):
    p = c['directory'] / 'manifest.json'
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(c['manifest'], indent=2) + '\n')
    os.replace(tmp, p)

def write_control(c, request_id):
    if c['control']:
        c['control'].parent.mkdir(parents=True, exist_ok=True)
        tmp = c['control'].with_name(c['control'].name + '.' + uuid.uuid4().hex + '.tmp')
        tmp.write_text(f"{c['manifest']['mode']} {c['manifest']['experiment_id']} {c['manifest']['run_id']} {request_id} {c['directory']}\n")
        os.replace(tmp, c['control'])

def clear_control(c, request_id):
    if c['control']:
        try:
            if c['control'].read_text().split()[3] == request_id: c['control'].unlink()
        except (OSError, IndexError): pass

def begin(c, case, payload):
    request_id = 'req-' + uuid.uuid4().hex
    row = {'request_id': request_id, 'case': case['id'], 'request': payload,
           'engine_jsonl': str(c['directory'] / (request_id + '.jsonl')),
           'control_written_unix_ns': time.time_ns()}
    c['manifest']['requests'].append(row)
    write_control(c, request_id)
    save(c)
    return row

def operation(c, name, callback):
    """Capture one already-declared non-generating engine operation."""
    rows = c['manifest']['control_operations']
    operation_id = f"ctl-{len(rows) + 1:04d}-{uuid.uuid4().hex}"
    row = {'operation_id': operation_id, 'request_id': operation_id, 'operation': name,
           'engine_jsonl': str(c['directory'] / (operation_id + '.jsonl')),
           'control_written_unix_ns': time.time_ns(), 'status': 'in_progress'}
    rows.append(row)
    write_control(c, operation_id)
    save(c)
    try:
        result = callback()
        row['status'] = 'completed'
        return result
    except BaseException as exc:
        row['status'] = 'failed'
        row['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        clear_control(c, operation_id)
        ids = {k: c['manifest'][k] for k in ('experiment_id', 'run_id')}
        ids['request_id'] = operation_id
        row['validation'] = validate_control(row['engine_jsonl'], ids, name) if c['manifest']['mode'] != 'off' else {'status': 'uncaptured', 'reason': 'engine instrumentation off or unavailable'}
        save(c)

def validate_envelope(path, ids):
    result = {'status': 'partial', 'errors': [], 'missing_fields': []}
    try:
        with open(path) as f: rows = [json.loads(line) for line in f]
    except (OSError, ValueError) as e:
        result['errors'].append('engine artifact unavailable or malformed: ' + str(e)); return result, []
    if not rows or rows[-1].get('type') != 'capture_footer':
        result['errors'].append('capture footer missing'); return result, []
    footer, events = rows[-1], rows[:-1]
    partial_marker = pathlib.Path(path).with_suffix('.partial')
    result['partial_marker_present'] = partial_marker.exists()
    if result['partial_marker_present']: result['errors'].append('capture partial marker remains')
    result['footer'] = footer
    if any(any(r.get(k) != v for k, v in ids.items()) for r in rows): result['errors'].append('identity mismatch')
    if footer.get('recorded_events') != len(events): result['errors'].append('footer recorded count mismatch')
    if footer.get('attempted_events') != footer.get('recorded_events', 0) + footer.get('dropped_events', -1): result['errors'].append('footer attempted count mismatch')
    if footer.get('dropped_events') != 0: result['errors'].append('dropped events')
    if 'timeline_dropped_events' in footer or 'detail_dropped_events' in footer:
        timeline_drops, detail_drops = footer.get('timeline_dropped_events'), footer.get('detail_dropped_events')
        result['coverage'] = {'timeline_dropped_events': timeline_drops, 'detail_dropped_events': detail_drops}
        if type(timeline_drops) is not int or type(detail_drops) is not int or timeline_drops < 0 or detail_drops < 0 or timeline_drops + detail_drops != footer.get('dropped_events'):
            result['errors'].append('per-class drop counts invalid')
        else:
            if timeline_drops: result['errors'].append('timeline events dropped')
            if detail_drops: result['errors'].append('detail events dropped')
    else:
        result['coverage'] = {'per_class_drops': 'unavailable in legacy footer'}
    if [r.get('event_seq') for r in events] != list(range(1, len(events)+1)): result['errors'].append('sequence gap or duplicate')
    return result, events

def validate_control(path, ids, operation=None):
    result, events = validate_envelope(path, ids)
    if not events: return result
    result['event_types'] = sorted(set(r.get('type') for r in events if isinstance(r.get('type'), str)))
    begin = [r for r in events if r.get('type') == 'control_begin']
    done = [r for r in events if r.get('type') == 'control_end']
    if len(begin) != 1 or len(done) != 1:
        result['missing_fields'].append('one control_begin and control_end')
    else:
        result['control_begin'] = begin[0]; result['control_end'] = done[0]
        expected_verb = 'PLACEMENT' if operation and operation.startswith('placement_') else None
        expected_action = ('inspect' if operation and operation.startswith('placement_inspect_') else
                           operation[len('placement_'):] if operation and operation.startswith('placement_') else None)
        if any(begin[0].get(k) != done[0].get(k) for k in ('verb', 'action')): result['errors'].append('control begin/end mismatch')
        if expected_verb and (begin[0].get('verb'), begin[0].get('action')) != (expected_verb, expected_action): result['errors'].append('unexpected control verb/action')
        if done[0].get('ok') is not True: result['missing_fields'].append('successful control completion')
    end = next((r for r in reversed(events) if r.get('type') == 'capture_end'), None)
    if not end: result['missing_fields'].append('capture_end')
    elif end.get('status') in ('cancel', 'error', 'error_or_early_exit'): result['missing_fields'].append('successful engine completion')
    if not result['errors'] and not result['missing_fields']: result['status'] = 'validated'
    return result

HELPER_COUNTERS = ('h2d_issued_ops', 'h2d_issued_bytes', 'd2h_issued_ops',
                   'd2h_issued_bytes', 'replacements_completed', 'copy_failures')

def helper_traffic(events, static):
    phases = ('before_prefill', 'generation_start', 'final')
    snapshots = {}
    errors, missing = [], []
    for phase in phases:
        rows = [r for r in events if r.get('type') == 'helper_weight_traffic' and r.get('phase') == phase]
        if len(rows) != 1:
            missing.append('one helper weight traffic snapshot at ' + phase)
            continue
        values = {key: rows[0].get(key) for key in HELPER_COUNTERS}
        pending = rows[0].get('helper_live_pending')
        if static and pending != 0: missing.append('no pending helper live copy at ' + phase)
        if any(type(value) is not int or value < 0 for value in values.values()):
            errors.append('invalid helper weight counter at ' + phase)
        else: snapshots[phase] = {**values, 'helper_live_pending': pending}
    deltas = {}
    for start, end in (('before_prefill', 'generation_start'), ('generation_start', 'final'), ('before_prefill', 'final')):
        if start in snapshots and end in snapshots:
            delta = {k: snapshots[end][k] - snapshots[start][k] for k in HELPER_COUNTERS}
            deltas[start + '_to_' + end] = delta
            if any(v < 0 for v in delta.values()): errors.append('helper weight counter decreased')
    whole = deltas.get('before_prefill_to_final')
    if static and whole and any(whole.values()): errors.append('static helper weight traffic or failure occurred during inference')
    return {'status': 'partial' if errors or missing else 'validated', 'static_required': static,
            'snapshots': snapshots, 'deltas': deltas, 'errors': errors, 'missing_fields': missing,
            'note': 'Per-process issued helper weight payload transfers; excludes activations, results and primary traffic. Final settlement is checked separately.'}

def helper_placement_comparison(before, after, static):
    """Compare existing placement inspections; they do not force copy settlement."""
    result = {'static_required': static, 'status': 'partial', 'errors': [], 'missing_fields': []}
    snapshots = {}
    for name, report in (('after_reset', before), ('final_inspection', after)):
        counters = report.get('helper_weight_traffic') if isinstance(report, dict) else None
        pending = report.get('helper_live_pending') if isinstance(report, dict) else None
        ids = report.get('helper_ids') if isinstance(report, dict) else None
        if not isinstance(counters, dict) or any(type(counters.get(k)) is not int or counters[k] < 0 for k in HELPER_COUNTERS):
            result['missing_fields'].append(name + ' helper weight counters')
        else: snapshots[name] = {k: counters[k] for k in HELPER_COUNTERS}
        if pending not in (0, 1): result['missing_fields'].append(name + ' helper live pending status')
        elif pending: result['missing_fields'].append(name + ' helper live work still pending')
        if not isinstance(ids, list): result['missing_fields'].append(name + ' helper roster')
        else:
            result[name + '_roster_sha256'] = hashlib.sha256(json.dumps(sorted(ids), separators=(',', ':')).encode()).hexdigest()
    if len(snapshots) == 2:
        result['after_reset_to_final'] = {k: snapshots['final_inspection'][k] - snapshots['after_reset'][k] for k in HELPER_COUNTERS}
        if any(v < 0 for v in result['after_reset_to_final'].values()): result['errors'].append('helper counter decreased')
        if static and any(result['after_reset_to_final'].values()): result['errors'].append('static helper weight traffic or failure occurred between inspections')
    if static and result.get('after_reset_roster_sha256') != result.get('final_inspection_roster_sha256'):
        result['errors'].append('static helper roster changed between inspections')
    if not result['errors'] and not result['missing_fields']: result['status'] = 'validated'
    result['note'] = 'Between-inspection traffic includes any control maintenance; per-request capture isolates inference. Equal roster digests across arms must also be checked.'
    return result

def validate(path, ids, usage, completed, static_helper=False):
    result, events = validate_envelope(path, ids)
    if not events: return result
    result['helper_weight_traffic'] = helper_traffic(events, static_helper)
    if static_helper and result['helper_weight_traffic']['status'] != 'validated':
        result['missing_fields'].append('validated static helper weight traffic')
    outputs = [r for r in events if r.get('type') == 'token_output']
    result['emitted_token_ids'] = [r.get('token') for r in outputs]
    if any(type(r.get('token')) is not int or r.get('output_index') != i for i, r in enumerate(outputs)): result['errors'].append('invalid output token/index')
    count = (usage or {}).get('completion_tokens')
    if type(count) is int:
        result['api_completion_tokens'] = count
        if len(outputs) != count: result['errors'].append('emitted/API completion count mismatch')
    else: result['missing_fields'].append('API completion token count')
    rounds = {}
    for r in events:
        if r.get('type') in ('draft_end', 'draft_proposal'):
            q = rounds.setdefault(r.get('round'), {'proposed': None, 'observed': 0})
            if r['type'] == 'draft_end': q['proposed'] = r.get('proposed')
            else: q['observed'] += 1
    result['draft_rounds'] = rounds
    if any(v['proposed'] != v['observed'] for v in rounds.values()): result['errors'].append('draft proposal count mismatch')
    starts = [r for r in events if r.get('type') == 'target_forward_begin']
    ends = [r for r in events if r.get('type') == 'target_forward_end']
    result['target_forward_begins'] = len(starts); result['target_forward_ends'] = len(ends)
    if len(starts) != len(ends): result['errors'].append('target forwards incomplete')
    if any(r.get('ok') == 0 for r in ends): result['missing_fields'].append('successful target forwards')
    round_starts = sum(r.get('type') == 'round_begin' for r in events)
    round_ends = sum(r.get('type') == 'round_end' for r in events)
    result['round_begins'] = round_starts; result['round_ends'] = round_ends
    if round_starts != round_ends: result['missing_fields'].append('complete round boundaries')
    end = next((r for r in reversed(events) if r.get('type') == 'capture_end'), None)
    result['engine_end'] = end
    if not end: result['missing_fields'].append('capture_end')
    elif end.get('status') in ('cancel', 'cancel_prefill', 'error', 'error_or_early_exit'): result['missing_fields'].append('successful engine completion')
    if not completed: result['missing_fields'].append('complete API stream')
    if not result['errors'] and not result['missing_fields']: result['status'] = 'validated'
    return result

def finish(c, row, item):
    ids = {k: c['manifest'][k] for k in ('experiment_id', 'run_id')}; ids['request_id'] = row['request_id']
    clear_control(c, row['request_id'])
    row['api_complete'] = item.get('complete', False)
    row['api_usage'] = item.get('usage')
    row['effective'] = (item.get('engine_stats') or {}).get('final')
    row['validation'] = validate(row['engine_jsonl'], ids, row['api_usage'], row['api_complete'], (row['request'].get('strata_tune') or {}).get('helper_live') == 0) if c['manifest']['mode'] != 'off' else {'status': 'uncaptured', 'reason': 'engine instrumentation off or unavailable'}
    frontend = c['directory'] / (row['request_id'] + '.frontend.jsonl')
    row['frontend_jsonl'] = str(frontend)
    try:
        with frontend.open() as f: received = [json.loads(line) for line in f]
        footer = received[-1]
        records = received[:-1]
        errors = []
        if footer.get('type') != 'frontend_footer' or footer.get('recorded_events') != len(records) or footer.get('dropped_events') != 0: errors.append('frontend footer or coverage invalid')
        if any(any(r.get(k) != v for k, v in ids.items()) for r in received): errors.append('frontend identity mismatch')
        if [r.get('event_seq') for r in records] != list(range(1, len(records) + 1)): errors.append('frontend sequence gap')
        token_ids = [r.get('token') for r in records if not r.get('drained')]
        emitted = row['validation'].get('emitted_token_ids')
        if emitted is not None and token_ids != emitted: errors.append('frontend/engine token IDs differ')
        count = (row['api_usage'] or {}).get('completion_tokens')
        if type(count) is int and len(token_ids) != count: errors.append('frontend/API completion count mismatch')
        row['frontend_validation'] = {'status': 'invalid' if errors else ('partial' if type(count) is not int or not row['api_complete'] else 'validated'), 'errors': errors, 'received_count': len(records), 'drained_count': len(records) - len(token_ids), 'footer': footer}
    except (OSError, ValueError, IndexError) as exc:
        row['frontend_validation'] = {'status': 'missing', 'reason': str(exc)}
    save(c)
