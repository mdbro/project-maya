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
import pathlib
import re
import sys
import time
import urllib.error
import urllib.request

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


def summarize_stats(before: dict, after: dict, usage: dict | None) -> dict:
    rows = history(after)
    new = {int(r['request_seq']) for r in rows if isinstance(r.get('request_seq'), (int, float))} - seqs(before)
    if len(new) != 1:
        return {'join': 'ambiguous', 'new_request_sequences': sorted(new), 'reason':
                'Expected exactly one new engine request sequence; concurrent traffic, restart, missing or rotated history prevents attribution.'}
    seq = next(iter(new)); samples = [r for r in rows if r.get('request_seq') == seq]
    final = samples[-1]
    generated = (usage or {}).get('completion_tokens')
    if not isinstance(final.get('emitted'), (int, float)) or generated is None or final['emitted'] != generated:
        return {'join': 'unverified', 'request_seq': seq, 'samples': samples,
                'reason': 'Final engine emitted count differs from or lacks the API completion-token count.'}
    start = next((r for r in samples if r.get('emitted') == 32), None)
    suffix = None
    if start and isinstance(start.get('elapsed_ms'), (int, float)) and isinstance(final.get('elapsed_ms'), (int, float)):
        interval = final['elapsed_ms'] - start['elapsed_ms']
        n = final['emitted'] - 32
        if n > 0 and interval > 0:
            suffix = {'start_emitted': 32, 'end_emitted': final['emitted'], 'tokens': n,
                      'start_elapsed_ms': start['elapsed_ms'], 'end_elapsed_ms': final['elapsed_ms'],
                      'interval_ms': interval, 'tokens_per_second': 1000 * n / interval}
    return {'join': 'one_new_sequence_and_matching_completion_count', 'request_seq': seq,
            'samples': samples, 'final': final, 'suffix_after_32': suffix,
            'note': 'Sequence/count matching does not establish exclusive access. Route counters begin before prefill.'}


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


def run_case(base: str, case: dict, payload: dict, timeout: float, raw, result: dict, engine_pid: int | None = None) -> None:
    before = metrics(base, min(timeout, 30))
    item = {'case': case['id'], 'request': payload, 'acceptance_examples': case['checks'], 'metrics_before': before,
            'events': [], 'content': '', 'reasoning_content': '', 'complete': False,
            'process_io_before': process_io(engine_pid)}
    result['requests'].append(item)
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
                                      'line_base64': base64.b64encode(line).decode('ascii')}, ensure_ascii=False) + '\n')
                raw.flush()
                if not line.startswith(b'data:'): continue
                data = line[5:].strip()
                if data == b'[DONE]': item['complete'] = True; break
                if not data: continue
                event = json.loads(data)
                if event.get('error'): raise RuntimeError('stream error: ' + json.dumps(event['error']))
                if event.get('usage') is not None: usage = event['usage']
                if event.get('timings') is not None: timings = event['timings']
                for choice in event.get('choices') or []:
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
        item['syntax_check'] = syntax_check(item['content'], 'summarize' if case['id'] == 'log-summary' else 'order_jobs')
        after = metrics(base, min(timeout, 30)); item['metrics_after'] = after
        item['engine_stats'] = summarize_stats(before, after, usage)
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


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url', default='http://127.0.0.1:8097')
    ap.add_argument('--case', choices=['all', 'log-summary', 'dependency-order'], default='all')
    ap.add_argument('--tokens', type=int, default=128)
    ap.add_argument('--routing', choices=['off', 'substitute', 'prior'], required=True)
    ap.add_argument('--ratio', type=float, default=0.8)
    ap.add_argument('--prior-bonus', type=float, default=None, help='additive ready-expert bonus (0..1; prior default 0.05)')
    ap.add_argument('--spec-policy', choices=['ordinary', 'sampled', 'fast_hsd'], required=True)
    ap.add_argument('--rho', type=float, default=0.2)
    ap.add_argument('--width', type=int, default=4, help='number of proposed draft tokens (1..7)')
    ap.add_argument('--draft-confidence', type=float, default=None, help='optional DSpark scheduling confidence floor, 0..1')
    ap.add_argument('--verify-segment', type=int, default=0, help='0 grouped verification; 1..7 serial target segments')
    ap.add_argument('--stage-overlap', type=int, choices=(0,1), default=None, help='explicit serial (0) or overlapped (1) stage path; omitted leaves the server default')
    ap.add_argument('--seed', type=int, default=20261101)
    ap.add_argument('--note', default='', help='conditioning and other relevant request history')
    ap.add_argument('--timeout', type=float, default=900)
    ap.add_argument('--engine-pid', type=int, default=None, help='local engine PID for whole-request Linux I/O counters')
    ap.add_argument('--output', required=True, type=pathlib.Path, help='new JSON file outside the source repository')
    a = ap.parse_args()
    if not 1 <= a.tokens <= 256 or not 0 <= a.ratio <= 1 or not 0 < a.rho <= 1 or not 1 <= a.width <= 7 or a.timeout <= 0:
        ap.error('tokens 1..256, ratio 0..1, rho (0,1], width 1..7 and positive timeout required')
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
    output = a.output.expanduser().resolve()
    if output.is_relative_to(ROOT): ap.error('output must be outside the source repository')
    if not output.parent.is_dir(): ap.error('output parent directory must already exist')
    raw_path = output.with_name(output.name + '.sse.jsonl')
    cases = json.loads(FIXTURE.read_text())['cases']
    cases = [c for c in cases if a.case == 'all' or c['id'] == a.case]
    tune = {'routing': a.routing, 'routing_ratio': a.ratio,
            'spec': 1 if a.spec_policy == 'ordinary' else a.width + 1,
            'spec_policy': 'existing' if a.spec_policy == 'ordinary' else a.spec_policy}
    if a.routing == 'prior' or a.prior_bonus is not None:
        tune['routing_prior_bonus'] = 0.05 if a.prior_bonus is None else a.prior_bonus
    if a.spec_policy == 'fast_hsd': tune['spec_rho'] = a.rho
    if a.draft_confidence is not None: tune['spec_min_p'] = a.draft_confidence
    tune['verify_segment'] = a.verify_segment
    if a.stage_overlap is not None: tune['stage_overlap'] = a.stage_overlap
    result = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(), 'base_url': a.base_url,
              'fixture': str(FIXTURE), 'arm': {'routing': a.routing, 'ratio': a.ratio,
              'prior_bonus': tune.get('routing_prior_bonus'),
              'spec_policy': a.spec_policy, 'width': 0 if a.spec_policy == 'ordinary' else a.width,
              'rho': a.rho if a.spec_policy == 'fast_hsd' else None, 'seed': a.seed,
              'draft_confidence': a.draft_confidence, 'verify_segment': a.verify_segment,
              'stage_overlap': a.stage_overlap},
              'conditioning_note': a.note, 'raw_sse_jsonl': str(raw_path), 'requests': []}
    with output.open('x', encoding='utf-8') as saved:
        try:
            with raw_path.open('x', encoding='utf-8') as raw:
                for case in cases:
                    payload = {'model': 'MiniMax-M3.1-Flash-native-wsl', 'messages': [{'role': 'user', 'content': case['prompt']}],
                               'stream': True, 'return_progress': True, 'max_tokens': a.tokens, 'temperature': 1,
                               'top_p': 0.95, 'top_k': 0, 'min_p': 0, 'seed': a.seed,
                               'reasoning_effort': 'none', 'strata_tune': tune}
                    try: run_case(a.base_url.rstrip('/'), case, payload, a.timeout, raw, result, a.engine_pid)
                    except Exception as exc:
                        result['requests'].append({'case': case['id'], 'request': payload, 'complete': False,
                                                   'error': f'{type(exc).__name__}: {exc}'})
                    saved.seek(0); saved.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n'); saved.flush()
                    if not result['requests'][-1].get('complete') or result['requests'][-1].get('error'): break
        finally:
            saved.seek(0); saved.write(json.dumps(result, indent=2, ensure_ascii=False) + '\n'); saved.flush()
    print(output)
    return 0 if len(result['requests']) == len(cases) and all(x.get('complete') and not x.get('error') for x in result['requests']) else 1


if __name__ == '__main__':
    sys.exit(main())
