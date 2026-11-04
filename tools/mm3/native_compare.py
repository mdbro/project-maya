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

import native_capture as capture
import native_calibrate as calibration
import native_screen as screen


def verify_conditioned_history(prior, before, current_io, current_metrics, executable_sha):
    """Bind a continuous historical arm without pretending to restore learning."""
    if not prior.get('complete') or not prior.get('historical_fixed_primary') or not prior.get('requests'):
        raise ValueError('historical arm requires completed explicitly fixed-primary conditioning')
    saved=prior.get('final_inspection') or {}
    calibration.same_process(saved,before)
    if not saved.get('ownership_complete') or not before.get('ownership_complete'):
        raise ValueError('conditioned ownership is incomplete')
    for key in ('target_object','draft_object','host_address','host_bytes','host_records','helper_pool'):
        if key not in saved or saved[key]!=before.get(key):
            raise ValueError('conditioned model allocation differs: '+key)
    for key in ('primary_ids','helper_ids'):
        if not isinstance(saved.get(key),list) or sorted(saved[key])!=sorted(before.get(key,[])):
            raise ValueError('conditioned roster differs: '+key)
    last=prior['requests'][-1]
    old_io=last.get('process_io_after') or {}
    if (not current_io or not old_io.get('start_ticks') or
            current_io.get('start_ticks')!=old_io['start_ticks'] or current_io.get('pid')!=saved['engine_pid']):
        raise ValueError('conditioned process identity differs')
    expected_seq=(last.get('engine_stats') or {}).get('request_seq')
    sequences=screen.seqs(current_metrics)
    if expected_seq is None or not sequences or max(sequences)!=expected_seq:
        raise ValueError('request history changed after conditioning')
    required=prior.get('required_binary') or {}
    if not executable_sha or executable_sha!=required.get('observed') or executable_sha!=required.get('expected'):
        raise ValueError('conditioned executable differs from pinned history')
    return {'request_seq':expected_seq,'engine_pid':saved['engine_pid'],'process_start_ticks':old_io['start_ticks'],
            'binary_sha256':executable_sha,'scope':'Continuous historical helper learning; no reset or checkpoint restore. Historical scores are not exported; process, exact rosters and uninterrupted request history are verified.'}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', type=pathlib.Path, required=True)
    ap.add_argument('--arm', required=True)
    ap.add_argument('--output', type=pathlib.Path, required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:8097')
    ap.add_argument('--continue-from', type=pathlib.Path, help='detailed companion immediately following this completed ranked receipt; no placement controls')
    capture.args(ap)
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
    if args.continue_from and (args.capture_mode != 'detailed' or not args.capture_control):
        ap.error('--continue-from requires detailed capture and a control file')
    result = {'kind': 'diagnostic_companion' if args.continue_from else 'ranked',
              'rankable': not bool(args.continue_from), 'base_url': args.base_url, 'created_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
              'protocol': str(args.protocol.resolve()),
              'protocol_sha256': hashlib.sha256(protocol_bytes).hexdigest(),
              'arm': args.arm, 'declared': arm, 'events': [], 'requests': []}
    def save():
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    # Claim the output before any state change.
    args.output.open('x').close()
    save()
    try:
        capture_context = capture.init(args, args.output.resolve())
        capture_context['manifest'].update({'receipt': str(args.output.resolve()), 'protocol': str(args.protocol.resolve()), 'protocol_sha256': result['protocol_sha256'], 'arm': args.arm, 'declared': arm, 'request_order': [c['id'] for c in cases]})
        capture.save(capture_context)
        result['capture_manifest'] = str(capture_context['directory'] / 'manifest.json')
        if args.continue_from:
            prior_path = args.continue_from.resolve()
            prior = json.loads(prior_path.read_text())
            if (not prior.get('complete') or prior.get('rankable') is not True or prior.get('arm') != args.arm or
                    prior.get('protocol_sha256') != result['protocol_sha256'] or prior.get('base_url') != args.base_url):
                raise ValueError('companion requires a complete ranked receipt for the same arm, protocol and endpoint')
            before = prior.get('final_inspection') or {}
            pid = before.get('engine_pid')
            last_io = (prior.get('requests') or [{}])[-1].get('process_io_after') or {}
            current_io = screen.process_io(pid)
            if not isinstance(pid, int) or not current_io or 'start_ticks' not in current_io or current_io['start_ticks'] != last_io.get('start_ticks'):
                raise ValueError('companion engine PID and process start identity do not match ranked receipt')
            capture.bind_executable(capture_context, pid)
            prior_manifest = json.loads(pathlib.Path(prior['capture_manifest']).read_text())
            if (capture_context['manifest']['executable'].get('sha256') is None or
                    capture_context['manifest']['executable'].get('sha256') != prior_manifest.get('executable', {}).get('sha256') or
                    capture_context['manifest']['source']['diff_sha256'] != prior_manifest.get('source', {}).get('diff_sha256') or
                    capture_context['manifest']['source']['untracked_sha256'] != prior_manifest.get('source', {}).get('untracked_sha256') or
                    capture_context['manifest']['source']['commit'] != prior_manifest.get('source', {}).get('commit')):
                raise ValueError('companion engine executable or tracked source differs from ranked receipt')
            result['continued_from'] = str(prior_path)
            result['initial_inspection'] = before
            result['inspection_provenance'] = 'dated final inspection from prior ranked arm; no new placement control issued'
            result['placement_sha256'] = prior.get('placement_sha256')
            capture_context['manifest'].update({'kind': 'diagnostic_companion', 'rankable': False, 'continued_from': str(prior_path),
                'prior_history': 'Continues from ranked arm final inspection and all of its Python/prose generation; no reset or added placement inspection.',
                'dated_prior_inspection': before})
        else:
            capture_context['manifest'].update({'kind': 'ranked', 'rankable': True})
            before = capture.operation(capture_context, 'placement_inspect_initial', lambda: calibration.inspect(args.base_url, result['events'], 120))
            result['initial_inspection'] = before
            capture.bind_executable(capture_context, before.get('engine_pid'))
            actual_sha=capture_context['manifest']['executable'].get('sha256')
            if protocol.get('binary_sha256') and actual_sha!=protocol['binary_sha256']:
                raise ValueError('resident executable differs from protocol binary')
            capture_context['manifest']['prior_history'] = before
            capture.save(capture_context)
            if arm.get('conditioned_history'):
                if arm.get('learned_checkpoint') or arm.get('placement'):
                    raise ValueError('continuous conditioned history cannot also reset placement')
                if arm.get('tune',{}).get('primary_adaptive')!=0 or arm.get('tune',{}).get('helper_live')!=1:
                    raise ValueError('historical conditioned arm must retain fixed primary and live helper')
                prior_path=pathlib.Path(arm['conditioned_history'])
                prior_bytes=prior_path.read_bytes();prior=json.loads(prior_bytes)
                result['conditioned_history_verification']=verify_conditioned_history(
                    prior,before,screen.process_io(before['engine_pid']),screen.metrics(args.base_url,120),actual_sha)
                result['conditioned_history_sha256']=hashlib.sha256(prior_bytes).hexdigest()
                rosters={k:before[k] for k in ('primary_ids','helper_ids')}
                result['placement_sha256']=hashlib.sha256(json.dumps(rosters,sort_keys=True).encode()).hexdigest()
                result['reset_kind']='none_continuous_conditioned_history'
                changed={'applied':False,'history':str(prior_path.resolve()),'boundary':'Initial inspection after the declared final conditioning inspection; no intervening generation.'}
            elif arm.get('learned_checkpoint'):
                if arm.get('placement'):
                    raise ValueError('choose learned checkpoint or histogram placement, not both')
                checkpoint = json.loads(pathlib.Path(arm['learned_checkpoint']).read_text())
                calibration.same_process(before, checkpoint)
                expected = calibration.learned_state_signature(checkpoint)
                changed = capture.operation(capture_context, 'placement_restore', lambda: calibration.post(args.base_url, '/v1/minimax/placement',
                    {'action': 'restore'}, result['events'], 900))
                calibration.same_process(before, changed)
                if calibration.learned_state_signature(changed) != expected:
                    raise ValueError('restored learned state differs from declared checkpoint')
                if changed['placement']['target_object'] != checkpoint['placement']['target_object']:
                    raise ValueError('checkpoint belongs to a different model instance')
                result['learned_state_sha256'] = expected
                result['placement_sha256'] = expected
                result['reset_kind'] = 'published_rosters_and_learned_state'
            else:
                profile = pathlib.Path(arm['placement'])
                result['placement_sha256'] = hashlib.sha256(profile.read_bytes()).hexdigest()
                changed = capture.operation(capture_context, 'placement_all', lambda: calibration.post(args.base_url, '/v1/minimax/placement',
                    {'action': 'all', 'profile_path': str(profile)}, result['events'], 900))
                calibration.same_process(before, changed)
                assert changed['ownership_complete']
                result['reset_kind'] = changed.get('initialization','legacy_histogram_placement_scores_unverified')
            result['reset'] = changed
            result['after_reset'] = before if arm.get('conditioned_history') else capture.operation(capture_context, 'placement_inspect_after_reset', lambda: calibration.inspect(args.base_url, result['events'], 120))
            capture_context['manifest']['placement_reset'] = changed
        capture_context['manifest']['placement_sha256'] = result['placement_sha256']
        capture_context['manifest']['raw_sse'] = str(args.output.with_suffix('.sse.jsonl').resolve())
        capture.save(capture_context)
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
                                before['engine_pid'], start_emitted=0, capture_context=capture_context)
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
        if args.continue_from:
            result['final_inspection'] = None
            result['helper_weight_placement_check'] = {'status': 'not_collected', 'reason': 'No additional placement inspection in diagnostic companion; per-request traffic and pending snapshots remain available.'}
        else:
            result['final_inspection'] = capture.operation(capture_context, 'placement_inspect_final', lambda: calibration.inspect(args.base_url, result['events'], 120))
            calibration.same_process(before, result['final_inspection'])
            result['helper_weight_placement_check'] = capture.helper_placement_comparison(
                result['after_reset'], result['final_inspection'], arm.get('tune', {}).get('helper_live') == 0)
        capture_context['manifest']['helper_weight_placement_check'] = result['helper_weight_placement_check']
        static = arm.get('tune', {}).get('helper_live') == 0
        traffic = [r.get('validation', {}).get('helper_weight_traffic', {}) for r in capture_context['manifest']['requests']]
        result['helper_weight_qualification'] = {'static_required': static,
            'request_statuses': [x.get('status', 'missing') for x in traffic],
            'placement_status': result['helper_weight_placement_check']['status'],
            'status': ('validated' if len(traffic) == len(cases) and all(x.get('status') == 'validated' for x in traffic)
                       and (not static or args.continue_from or result['helper_weight_placement_check']['status'] == 'validated') else 'partial'),
            'note': 'Companion has no post-run placement inspection; compare starting roster hashes across ranked arms separately.'}
        capture_context['manifest']['helper_weight_qualification'] = result['helper_weight_qualification']
        capture.save(capture_context)
        if static and result['helper_weight_qualification']['status'] != 'validated':
            raise RuntimeError('static helper weight-traffic invariant was not validated; preserve this incomplete arm')
        result['complete'] = True
        save()
        print(json.dumps(result['pooled_generation']), flush=True)
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        save()
        raise


if __name__ == '__main__':
    main()
