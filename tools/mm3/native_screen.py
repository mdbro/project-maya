#!/usr/bin/env python3
"""Run one declared arm against an already running native MiniMax server.

Writes one exclusive JSON result and a timestamped raw-SSE sidecar. This tool
never starts an engine or chooses additional arms. Record conditioning and any
other server traffic with --note; results require a quiet single-request server.
"""
from __future__ import annotations

import argparse
import base64
import ast
import datetime as dt
import json
import math
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

if __package__:
    from . import native_capture as capture
else:
    import native_capture as capture

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE = pathlib.Path(__file__).with_name('native_screen_prompts.json')


def metrics(base: str, timeout: float) -> dict:
    with urllib.request.urlopen(base + '/metrics', timeout=timeout) as response:
        return json.load(response)


def process_io(pid: int | None) -> dict | None:
    if pid is None:
        return None
    try:
        root = pathlib.Path('/proc') / str(pid)
        ticks = root.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]
        counters = {k: int(v) for k, v in (line.split(':', 1) for line in root.joinpath('io').read_text().splitlines())}
        return {'pid': pid, 'start_ticks': ticks, 'counters': counters}
    except (OSError, ValueError, IndexError) as exc:
        return {'pid': pid, 'error': str(exc)}


def history(m: dict) -> list[dict]:
    columns = (m.get('tiers') or {}).get('history') or {}
    count = len(columns.get('request_seq') or [])
    return [{k: v[i] if isinstance(v, list) and i < len(v) else None for k, v in columns.items()}
            for i in range(count)]


def seqs(m: dict) -> set[int]:
    return {int(r['request_seq']) for r in history(m) if isinstance(r.get('request_seq'), (int, float))}


def summarize_stats(before: dict, after: dict, usage: dict | None, start_emitted: int = 32) -> dict:
    """Join one engine request and measure only fully completed rounds.

    The zero boundary is required: without it a rotated history can hide an
    earlier qualifying start boundary. All observed samples are validated,
    including interior samples that do not themselves define intervals.
    """
    rows = history(after)
    new = {int(r['request_seq']) for r in rows if isinstance(r.get('request_seq'), (int, float))} - seqs(before)
    if len(new) != 1:
        return {'join': 'ambiguous', 'new_request_sequences': sorted(new), 'generation_interval': None,
                'reason': 'Expected exactly one new engine request sequence; concurrent traffic, restart, missing or rotated history prevents attribution.'}
    seq = next(iter(new))
    samples = [r for r in rows if r.get('request_seq') == seq]
    result = {'join': 'unverified', 'request_seq': seq, 'samples': samples, 'generation_interval': None,
              'start_emitted_requested': start_emitted,
              'note': 'Sequence/count matching does not establish exclusive access or warmed expert caches. Route counters begin before prefill.'}
    def fail(reason):
        result['join'] = 'unverified'
        result['reason'] = reason
        return result
    generated = (usage or {}).get('completion_tokens')
    if type(generated) is not int or generated < 0:
        return fail('Missing or invalid API completion-token count.')
    previous_count, previous_ms = -1, -1.0
    for row in samples:
        count, ms = row.get('emitted'), row.get('elapsed_ms')
        if (isinstance(count, bool) or not isinstance(count, (int, float)) or not math.isfinite(count)
                or count < 0 or int(count) != count or isinstance(ms, bool) or not isinstance(ms, (int, float))
                or not math.isfinite(ms) or ms < 0):
            return fail('A sample lacks a valid emitted count or elapsed time.')
        if count < previous_count or ms < previous_ms:
            return fail('Emitted counts or elapsed times are not monotonic.')
        previous_count, previous_ms = count, ms
        if row.get('round_complete') not in (0, 1) or row.get('generation_final') not in (0, 1):
            return fail('Completed-round or final markers are missing or invalid; legacy samples cannot qualify.')
        if row['generation_final'] == 1 and row['round_complete'] != 1:
            return fail('Final marker is not a completed-round boundary.')
    if not samples or samples[0]['emitted'] != 0 or samples[0]['elapsed_ms'] != 0 or samples[0]['round_complete'] != 1 or samples[0]['generation_final'] != 0:
        return fail('The initial zero boundary is missing; the retained history may have rotated.')
    finals = [i for i, row in enumerate(samples) if row['generation_final'] == 1]
    if finals != [len(samples) - 1]:
        return fail('Expected one final completed boundary as the last sample.')
    final = samples[-1]
    result['final'] = final
    if final['emitted'] != generated:
        return fail('Final engine emitted count differs from the API completion-token count.')
    result['join'] = 'one_new_sequence_and_matching_completion_count'
    start = next((row for row in samples if row['round_complete'] == 1 and row['emitted'] >= start_emitted), None)
    if start is None or start is final or final['emitted'] <= start['emitted']:
        result['interval_status'] = 'not_eligible'
        result['reason'] = 'Generation ended before a measurable complete interval at the requested threshold.'
        return result
    elapsed = final['elapsed_ms'] - start['elapsed_ms']
    if elapsed <= 0:
        return fail('The completed interval has nonpositive elapsed time.')
    tokens = int(final['emitted'] - start['emitted'])
    result['interval_status'] = 'measured'
    result['generation_interval'] = {'requested_start_emitted': start_emitted,
        'start_emitted': int(start['emitted']), 'end_emitted': int(final['emitted']), 'tokens': tokens,
        'start_elapsed_ms': start['elapsed_ms'], 'end_elapsed_ms': final['elapsed_ms'],
        'interval_ms': elapsed, 'tokens_per_second': 1000 * tokens / elapsed,
        'method': 'first completed round at or after requested count through final completed boundary'}
    return result


def syntax_check(text: str, name: str) -> dict:
    match = re.search(r'```(?:python|py)?\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
    if not match:
        return {'kind': 'syntax_only', 'status': 'no_complete_python_fence', 'function': name}
    try:
        tree = ast.parse(match.group(1))
    except SyntaxError as exc:
        return {'kind': 'syntax_only', 'status': 'syntax_error', 'error': str(exc), 'function': name}
    found = any(isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef)) and x.name == name for x in tree.body)
    return {'kind': 'syntax_only', 'status': 'function_present' if found else 'function_missing', 'function': name,
            'note': 'Generated code was not executed; fixture acceptance examples need a separate functional check.'}


def run_case(base: str, case: dict, payload: dict, timeout: float, raw, result: dict, engine_pid: int | None = None, start_emitted: int = 32, capture_context: dict | None = None) -> None:
    if payload.get('n', 1) != 1:
        raise ValueError('Only one completion sequence is supported')
    before = metrics(base, min(timeout, 30))
    item = {'case': case['id'], 'request': payload, 'acceptance_examples': case['checks'], 'metrics_before': before,
            'events': [], 'content': '', 'reasoning_content': '', 'complete': False,
            'process_io_before': process_io(engine_pid)}
    result['requests'].append(item)
    capture_row = capture.begin(capture_context, case, payload) if capture_context else None
    if capture_row: item['capture_request_id'] = capture_row['request_id']
    request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(payload).encode(),
                                     headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'}, method='POST')
    start = time.monotonic(); first = None; usage = None; timings = None; finish = None
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            item['http_status'] = response.status
            item['content_type'] = response.headers.get('Content-Type')
            for line in response:
                elapsed = (time.monotonic() - start) * 1000
                raw.write(json.dumps({'case': case['id'], 'elapsed_ms': elapsed,
                                      'request_id': item.get('capture_request_id'), 'reception_monotonic_ns': time.monotonic_ns(),
                                      'line_base64': base64.b64encode(line).decode('ascii')}, ensure_ascii=False) + '\n')
                if not line.startswith(b'data:'): continue
                data = line[5:].strip()
                if data == b'[DONE]': item['complete'] = True; break
                if not data: continue
                event = json.loads(data)
                if event.get('error'): raise RuntimeError('stream error: ' + json.dumps(event['error']))
                if event.get('usage') is not None: usage = event['usage']
                if event.get('timings') is not None: timings = event['timings']
                choices = event.get('choices') or []
                if len(choices) > 1 or any(c.get('index', 0) != 0 for c in choices):
                    raise ValueError('Multiple completion sequences are not supported')
                for choice in choices:
                    delta = choice.get('delta') or {}
                    for key in ('content', 'reasoning_content'):
                        text = delta.get(key)
                        if isinstance(text, str) and text:
                            if first is None: first = elapsed
                            item[key] += text
                    if choice.get('finish_reason') is not None: finish = choice['finish_reason']
                # Preserve the parsed progress/finish/usage plus their timestamps;
                # the sidecar retains every original SSE line including comments.
                item['events'].append({'elapsed_ms': elapsed, 'data': event})
        item['api_wall_ms'] = (time.monotonic() - start) * 1000
        item['api_first_text_ms'] = first
        item['usage'] = usage; item['timings'] = timings; item['finish_reason'] = finish
        if case.get('function'):
            item['syntax_check'] = syntax_check(item['content'], case['function'])
        item['completion_status'] = ('truncated' if finish == 'length' else 'final_answer' if item['content'].strip() else 'no_final_answer')
        after = metrics(base, min(timeout, 30)); item['metrics_after'] = after
        item['engine_stats'] = summarize_stats(before, after, usage, start_emitted)
        item['process_io_after'] = process_io(engine_pid)
        ib, ia = item['process_io_before'], item['process_io_after']
        if ib and ia and 'counters' in ib and 'counters' in ia and ib['start_ticks'] == ia['start_ticks']:
            item['process_io_delta'] = {k: v - ib['counters'][k] for k, v in ia['counters'].items()}
            item['process_io_note'] = 'Linux process I/O counters across the whole request; not a Windows driver/pagefile trace.'
        if not item['complete']: item['error'] = 'stream ended without [DONE]'
    except Exception as exc:
        item['api_wall_ms'] = (time.monotonic() - start) * 1000
        item['api_first_text_ms'] = first
        if isinstance(exc, urllib.error.HTTPError):
            item['http_status'] = exc.code
            item['http_error_body'] = exc.read().decode('utf-8', errors='replace')
        item['error'] = f'{type(exc).__name__}: {exc}'
        try: item['metrics_after'] = metrics(base, min(timeout, 30))
        except Exception as m: item['metrics_error'] = f'{type(m).__name__}: {m}'
    finally:
        if capture_row: capture.finish(capture_context, capture_row, item)


def fresh_context_check(item: dict) -> dict:
    timings = item.get('timings') or {}
    usage = item.get('usage') or {}
    details = usage.get('prompt_tokens_details') or {}
    values = {name: value for name, value in (
        ('timings.cache_n', timings.get('cache_n')),
        ('usage.prompt_tokens_details.cached_tokens', details.get('cached_tokens')))
        if value is not None}
    verified = bool(values) and all(type(value) is int and value == 0 for value in values.values())
    return {'verified': verified, 'observed': values,
            'note': 'A missing, nonzero, invalid or conflicting cache count cannot verify fresh context.'}


def aggregate_screen(requests: list[dict], expected: int, start_emitted: int) -> dict:
    intervals = [(x['case'], (x.get('engine_stats') or {}).get('generation_interval')) for x in requests]
    valid = len(intervals) == expected and all(interval and x.get('complete') and not x.get('error')
                                               for x, (_, interval) in zip(requests, intervals))
    tokens = sum(i['tokens'] for _, i in intervals if i)
    ms = sum(i['interval_ms'] for _, i in intervals if i)
    return {'status': 'measured' if valid else 'incomplete', 'whole_post_prefill': start_emitted == 0,
            'pooled_tokens': tokens, 'pooled_ms': ms,
            'pooled_tokens_per_second': 1000 * tokens / ms if valid and ms else None,
            'individual': {name: interval for name, interval in intervals}}


def cooperation_error(final: dict, expected: int) -> str | None:
    if final.get('native_cooperative') != expected:
        return 'generation cooperation mode did not match the declared startup control'
    if not expected and (final.get('decode_cpu_experts') != 0 or final.get('helper_experts') != 0 or
                         final.get('owner_entries', 0) <= 0):
        return 'cooperation-off dispatch evidence is missing or includes CPU/helper expert work'
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url', default='http://127.0.0.1:8097')
    ap.add_argument('--case', choices=['all', 'python-code', 'prose'], default='all')
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--start-emitted', type=int, default=0, help='historical diagnostic threshold; nonzero is not the default speed metric')
    ap.add_argument('--reasoning-effort', choices=['high', 'xhigh', 'max'], default='high')
    ap.add_argument('--case-order', choices=['forward', 'reverse'], default='forward')
    ap.add_argument('--routing-prefetch', type=int, choices=(0, 1), default=None)
    ap.add_argument('--routing-dampen', type=int, choices=(0, 1), default=None)
    ap.add_argument('--routing', choices=['off', 'substitute', 'prior'], required=True)
    ap.add_argument('--ratio', type=float, default=0.8)
    ap.add_argument('--prior-bonus', type=float, default=None, help='additive ready-expert bonus (0..1; prior default 0.05)')
    ap.add_argument('--spec-policy', choices=['ordinary', 'sampled', 'fast_hsd', 'lenience'], required=True)
    ap.add_argument('--rho', type=float, default=0.2, help='historical cutoff only')
    ap.add_argument('--ell', type=float, default=0.4)
    ap.add_argument('--width', type=int, default=4, help='number of proposed draft tokens (1..7)')
    ap.add_argument('--draft-confidence', type=float, default=None, help='optional DSpark scheduling confidence floor, 0..1')
    ap.add_argument('--verify-segment', type=int, default=0, help='0 grouped verification; 1..7 serial target segments')
    ap.add_argument('--stage-overlap', type=int, choices=(0,1), default=None, help='explicit serial (0) or overlapped (1) stage path; omitted leaves the server default')
    ap.add_argument('--primary-adaptive', type=int, choices=(0,1), default=None, help='explicit fixed or adaptive native primary cache')
    ap.add_argument('--primary-max-exchanges', type=int, default=None,
                    help='explicit primary admission budget per five evaluated generation rows')
    ap.add_argument('--cooperative-cpu-scale', type=float, default=None,
                    help='explicit multiplier for measured CPU service prices')
    ap.add_argument('--helper-live', type=int, choices=(0,1), default=None, help='explicit static or live native helper cache')
    ap.add_argument('--helper-dispatch', choices=('worker','queue'), default=None, help='explicit helper request transport')
    ap.add_argument('--primary-arithmetic', choices=('dp4a','fp4'), default=None, help='explicit native primary expert arithmetic')
    ap.add_argument('--expect-cooperative', type=int, choices=(0,1), default=None,
                    help='require this actual startup cooperation mode; does not change the server')
    ap.add_argument('--seed', type=int, default=20261101)
    ap.add_argument('--note', default='', help='conditioning and other relevant request history')
    ap.add_argument('--timeout', type=float, default=900)
    ap.add_argument('--engine-pid', type=int, default=None, help='local engine PID for whole-request Linux I/O counters')
    ap.add_argument('--output', required=True, type=pathlib.Path, help='new JSON file outside the source repository')
    capture.args(ap)
    a = ap.parse_args()
    if a.start_emitted < 0: ap.error('start-emitted must be nonnegative')
    if not 1 <= a.tokens <= 256 or not 0 <= a.ratio <= 1 or not 0 < a.rho <= 1 or not 0 < a.ell <= 1 or not 1 <= a.width <= 7 or a.timeout <= 0:
        ap.error('tokens 1..256, ratio 0..1, rho and ell (0,1], width 1..7 and positive timeout required')
    if a.draft_confidence is not None and not 0 <= a.draft_confidence <= 1:
        ap.error('draft confidence must be 0..1')
    if a.prior_bonus is not None and not 0 <= a.prior_bonus <= 1:
        ap.error('prior bonus must be 0..1')
    if not 0 <= a.verify_segment <= 7 or (a.spec_policy == 'ordinary' and a.verify_segment):
        ap.error('verify segment must be 0..7 and zero for ordinary decoding')
    if a.stage_overlap == 1 and (a.verify_segment != 4 or a.spec_policy == 'ordinary'):
        ap.error('stage overlap requires verify-segment 4 and speculative decoding')
    if a.engine_pid is not None and a.engine_pid <= 0:
        ap.error('engine PID must be positive')
    if a.primary_max_exchanges is not None and not 2 <= a.primary_max_exchanges <= 32:
        ap.error('primary admission budget must be 2..32')
    if a.cooperative_cpu_scale is not None and not 0.25 <= a.cooperative_cpu_scale <= 4:
        ap.error('cooperative CPU scale must be finite and 0.25..4')
    output = a.output.expanduser().resolve()
    if output.is_relative_to(ROOT): ap.error('output must be outside the source repository')
    if not output.parent.is_dir(): ap.error('output parent directory must already exist')
    raw_path = output.with_name(output.name + '.sse.jsonl')
    if a.expect_cooperative is not None:
        actual = metrics(a.base_url.rstrip('/'), min(a.timeout, 30)).get('engine', {}).get('native_cooperative')
        if actual != a.expect_cooperative:
            ap.error(f'expected native_cooperative={a.expect_cooperative}, engine reports {actual!r}')
    cases = json.loads(FIXTURE.read_text())['cases']
    cases = [c for c in cases if a.case == 'all' or c['id'] == a.case]
    if a.case_order == 'reverse': cases.reverse()
    tune = {'fresh_context': 1, 'routing': a.routing, 'routing_ratio': a.ratio,
            'spec': 1 if a.spec_policy == 'ordinary' else a.width + 1,
            'spec_policy': 'existing' if a.spec_policy == 'ordinary' else a.spec_policy}
    if a.routing_prefetch is not None: tune['routing_prefetch'] = a.routing_prefetch
    if a.routing_dampen is not None: tune['routing_dampen'] = a.routing_dampen
    if a.routing == 'prior' or a.prior_bonus is not None:
        tune['routing_prior_bonus'] = 0.05 if a.prior_bonus is None else a.prior_bonus
    if a.spec_policy == 'fast_hsd': tune['spec_rho'] = a.rho
    if a.spec_policy == 'lenience': tune['spec_ell'] = a.ell
    if a.draft_confidence is not None: tune['spec_min_p'] = a.draft_confidence
    tune['verify_segment'] = a.verify_segment
    if a.stage_overlap is not None: tune['stage_overlap'] = a.stage_overlap
    if a.primary_adaptive is not None: tune['primary_adaptive'] = a.primary_adaptive
    if a.primary_max_exchanges is not None: tune['primary_max_exchanges'] = a.primary_max_exchanges
    if a.cooperative_cpu_scale is not None: tune['cooperative_cpu_scale'] = a.cooperative_cpu_scale
    if a.helper_live is not None: tune['helper_live'] = a.helper_live
    if a.helper_dispatch is not None: tune['helper_dispatch'] = a.helper_dispatch
    if a.primary_arithmetic is not None: tune['primary_arithmetic'] = a.primary_arithmetic
    result = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'base_url': a.base_url,
              'fixture': str(FIXTURE), 'arm': {'routing': a.routing, 'ratio': a.ratio,
              'prior_bonus': tune.get('routing_prior_bonus'), 'routing_dampen': a.routing_dampen,
              'routing_prefetch': a.routing_prefetch,
              'spec_policy': a.spec_policy, 'width': 0 if a.spec_policy == 'ordinary' else a.width,
              'rho': a.rho if a.spec_policy == 'fast_hsd' else None, 'ell': a.ell if a.spec_policy == 'lenience' else None, 'seed': a.seed,
              'draft_confidence': a.draft_confidence, 'verify_segment': a.verify_segment,
              'stage_overlap': a.stage_overlap, 'primary_adaptive': a.primary_adaptive, 'helper_live': a.helper_live, 'helper_dispatch': a.helper_dispatch, 'primary_arithmetic': a.primary_arithmetic,
              'primary_max_exchanges': a.primary_max_exchanges,
              'cooperative_cpu_scale': a.cooperative_cpu_scale,
              'expected_native_cooperative': a.expect_cooperative},
              'conditioning_note': a.note, 'conditioning_precondition': 'Declare neutral setup generation, starting placement and prior request history; do not train placement on scored prompts.',
              'request_state': 'fresh target and draft context for each request; retain learned expert-weight residency',
              'measurement_protocol': {'default_two_task_screen': len(cases) == 2, 'reasoning_effort': a.reasoning_effort, 'max_tokens_per_request': a.tokens, 'temperature': 1, 'top_p': 0.95, 'top_k': 0, 'min_p': 0, 'start_emitted': a.start_emitted, 'case_order': a.case_order, 'cases': [c['id'] for c in cases],
                                       'interval': 'first completed round at or after requested count through final completed boundary'},
              'raw_sse_jsonl': str(raw_path), 'requests': []}
    capture_context = capture.init(a, output, a.engine_pid)
    capture_context['manifest'].update({'receipt': str(output), 'raw_sse': str(raw_path), 'conditioning': a.note, 'requested_arm': result['arm'], 'measurement_protocol': result['measurement_protocol']})
    capture.save(capture_context)
    result['capture_manifest'] = str(capture_context['directory'] / 'manifest.json')
    with output.open('x', encoding='utf-8') as saved:
        try:
            with raw_path.open('x', encoding='utf-8') as raw:
                for case in cases:
                    payload = {'model': 'MiniMax-M3.1-Flash-native-wsl', 'messages': [{'role': 'user', 'content': case['prompt']}],
                               'stream': True, 'return_progress': True, 'max_tokens': a.tokens, 'temperature': 1,
                               'top_p': 0.95, 'top_k': 0, 'min_p': 0, 'seed': a.seed,
                               'reasoning_effort': a.reasoning_effort, 'loop_guard': False, 'strata_tune': tune}
                    try:
                        run_case(a.base_url.rstrip('/'), case, payload, a.timeout, raw, result, a.engine_pid, a.start_emitted, capture_context)
                        item = result['requests'][-1]
                        item['fresh_context_check'] = fresh_context_check(item)
                        if not item['fresh_context_check']['verified']:
                            item['error'] = 'fresh context not verified by zero cached prompt tokens'
                        if a.primary_max_exchanges is not None:
                            actual_budget = item.get('engine_stats', {}).get('final', {}).get('primary_max_exchanges')
                            item['primary_budget_check'] = {'expected':a.primary_max_exchanges,'observed':actual_budget}
                            if actual_budget != a.primary_max_exchanges:
                                item['error'] = 'effective primary admission budget differs from request'
                        if a.cooperative_cpu_scale is not None:
                            actual_scale = item.get('engine_stats', {}).get('final', {}).get('cooperative_cpu_scale')
                            item['cpu_price_check'] = {'expected':a.cooperative_cpu_scale,'observed':actual_scale}
                            if actual_scale != a.cooperative_cpu_scale:
                                item['error'] = 'effective CPU price scale differs from request'
                        if a.expect_cooperative is not None:
                            item = result['requests'][-1]
                            final = item.get('engine_stats', {}).get('final', {})
                            error = cooperation_error(final, a.expect_cooperative)
                            if error: item['error'] = error
                    except Exception as exc:
                        result['requests'].append({'case': case['id'], 'request': payload, 'complete': False,
                                                   'error': f'{type(exc).__name__}: {exc}'})
                    saved.seek(0); saved.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n'); saved.truncate(); saved.flush()
                    if not result['requests'][-1].get('complete') or result['requests'][-1].get('error'): break
        finally:
            result['pooled_generation'] = aggregate_screen(result['requests'], len(cases), a.start_emitted)
            saved.seek(0); saved.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n'); saved.truncate(); saved.flush()
    print(output)
    return 0 if len(result['requests']) == len(cases) and all(x.get('complete') and not x.get('error') for x in result['requests']) else 1


if __name__ == '__main__':
    sys.exit(main())
