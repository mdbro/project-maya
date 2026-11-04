#!/usr/bin/env python3
"""Run one predeclared two-request arm on a resident engine, preserving raw evidence.

The protocol supplies explicit settings and placement. This does not launch engines,
fit placement, choose a winner, or modify the server's runtime defaults.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import sys
import urllib.request

import native_calibrate as calibration
import native_screen as screen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=pathlib.Path, required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:8097')
    args = ap.parse_args()
    protocol_bytes = args.protocol.read_bytes()
    protocol = json.loads(protocol_bytes)
    arm = protocol['arms'][args.arm]
    cases = protocol['requests']
    if [c['id'] for c in cases] != ['python-code', 'prose']:
        ap.error('protocol must declare Python then prose')
    if args.output.resolve().is_relative_to(screen.ROOT):
        ap.error('receipts must be outside the repository')
    if args.output.exists():
        ap.error('preserve existing receipt')
    result = {'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'protocol': str(args.protocol.resolve()),
              'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
              'arm': args.arm, 'declared': arm, 'events': [], 'requests': []}
    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    # Claim the output before any state change.
    args.output.open('x').close()
    save()
    try:
        before = calibration.inspect(args.base_url, result['events'], 120)
        result['initial_inspection'] = before
        profile = pathlib.Path(arm['placement'])
        result['placement_sha256'] = hashlib.sha256(profile.read_bytes()).hexdigest()
        changed = calibration.post(args.base_url, '/v1/minimax/placement',
            {'action': 'all', 'profile_path': str(profile)}, result['events'], 900)
        calibration.same_process(before, changed)
        assert changed['ownership_complete']
        result['reset'] = changed
        result['after_reset'] = calibration.inspect(args.base_url, result['events'], 120)
        save()
        with args.output.with_suffix('.sse.jsonl').open('x') as raw:
            for case in cases:
                req = {'model': 'MiniMax-M3.1-Flash-native-wsl',
                       'messages': [{'role': 'user', 'content': case['prompt']}],
                       'stream': True, 'return_progress': True,
                       'max_tokens': protocol['max_tokens_each'],
                       'temperature': 1, 'top_p': .95, 'top_k': 0, 'min_p': 0,
                       'seed': protocol['seed'], 'reasoning_effort': 'high',
                       'loop_guard': False, 'strata_tune': arm['tune']}
                screen.run_case(args.base_url, case, req, 900, raw, result,
                                before['engine_pid'], start_emitted=0)
                q = result['requests'][-1]
                q['fresh_context_check'] = screen.fresh_context_check(q)
                f = q.get('engine_stats', {}).get('final', {})
                q['effective_checks'] = {k: {'expected': v, 'observed': f.get(k)}
                                         for k, v in arm['expect'].items()}
                save()
                assert q['complete'] and not q.get('error'), q.get('error')
                assert q['fresh_context_check']['verified'], q['fresh_context_check']
                assert screen.cooperation_error(f, 1) is None
                assert all(v['expected'] == v['observed'] for v in q['effective_checks'].values()), q['effective_checks']
                assert q['engine_stats']['interval_status'] == 'measured'
                interval = q['engine_stats']['generation_interval']
                print(args.arm, case['id'], round(interval['tokens_per_second'], 4), flush=True)
        result['pooled_generation'] = screen.aggregate_screen(result['requests'], 2, 0)
        result['final_inspection'] = calibration.inspect(args.base_url, result['events'], 120)
        calibration.same_process(before, result['final_inspection'])
        result['complete'] = True
        save()
        print(json.dumps(result['pooled_generation']), flush=True)
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise


if __name__ == '__main__':
    main()
