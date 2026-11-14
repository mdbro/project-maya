#!/usr/bin/env python3
"""Bounded measurement campaigns for the shared generation model.

Extends the diagnostic campaign driver with what the follow-up needs: timestamped
telemetry during an arm, the timestamps-only recorder level, fixtures carrying the
submission query, and short GPU-metrics captures. The protocol file declares every
step, the run limit and the decision rules before any run. The driver has no way to
run an undeclared step, so a limit reached is a limit reached.

Every arm goes through the same capture path, including recorder-off arms, so that
pre-run activity does not differ between arms.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import pathlib
import signal
import subprocess
import sys
import time

import native_capture as capture
import native_light as light
import native_screen as screen
from native_shared_diagnostic import Campaign, ROOT, post, utc

TELEMETRY = pathlib.Path(__file__).with_name('native_telemetry.py')
LIGHT_MODES = ('light', 'timestamps')


def prefix_reuse_check(item: dict) -> dict:
    """A conversation's next turn (a step with expect_reuse: fresh_context 0 and no restore): the engine must have kept
    the earlier turn's prefix and read only the rest - every cache count the request reports present, an int, above zero."""
    observed = screen.fresh_context_check(item)['observed']
    return dict(verified=bool(observed) and all(type(v) is int and v > 0 for v in observed.values()), observed=observed)


def final_stat(item: dict) -> dict:
    """A request's final statistics: the engine's last sample as the server's history keeps it (engine_stats.final), with
    the keys only the latest STAT line carries added (metrics_after.tiers.now - the server's history keeps a fixed list
    without serve_monitoring, routing_floor, routing_prior_bonus_one, spec_plan_file, cpu_threads, warnings and the
    prompt_* keys, so a protocol's expected settings and the prompt-reading times were not in the ledger), when that line
    is this request's (the same request_seq). Where both have a key, the history's value is kept."""
    final = (item.get('engine_stats') or {}).get('final') or {}
    now = (((item.get('metrics_after') or {}).get('tiers') or {}).get('now')) or {}
    seq = final.get('request_seq')
    if final and isinstance(now, dict) and seq is not None and now.get('request_seq') == seq:
        return {**now, **final}
    return dict(final)


class FollowUp(Campaign):
    def __init__(self, a):
        super().__init__(a)
        # A campaign may run in an already resident engine: its capture control and learned-state
        # origin then belong to the campaign that loaded it. Restores still verify that signature.
        if self.protocol.get('capture_control'):
            self.control = pathlib.Path(self.protocol['capture_control'])
        adopted = self.protocol.get('adopt_origin')
        if adopted and adopted['phase'] not in self.ledger['phases']:
            earlier = json.loads(pathlib.Path(adopted['ledger']).read_text())
            self.ledger['phases'][adopted['phase']] = dict(origin_signature=earlier['phases'][adopted['phase']]['origin_signature'], adopted_from=adopted['ledger'])

    def origin_phase(self, phase: str) -> str:
        return self.protocol['phases'][phase].get('origin_phase', phase)

    def restore(self, phase: str) -> dict:
        return super().restore(self.origin_phase(phase))

    def payload(self, case: dict, max_tokens: int, tune: dict | None = None) -> dict:
        # A conditioning case may carry messages (a re-sent multi-turn conversation: user and assistant turns) instead of
        # only a prompt; a non-empty list is sent as it is (a copy), otherwise one user message from case['prompt'].
        messages = case.get('messages')
        if isinstance(messages, list) and messages:
            payload = super().payload({**case, 'prompt': case.get('prompt', '')}, max_tokens)
            payload['messages'] = copy.deepcopy(messages)
        else:
            payload = super().payload(case, max_tokens)
        if tune:
            payload['strata_tune'] = {**payload['strata_tune'], **tune}
        if case.get('tools'):
            payload['tools'] = case['tools']
        return payload

    def fixture_path(self, variant: str) -> pathlib.Path:
        return self.directory / f'fixture-{variant}.replay'

    def export_fixture(self, step: dict) -> dict:
        """Selection is by order only and was declared before any timing was inspected."""
        receipts = {}
        for variant, rule in self.protocol['fixtures'].items():
            if self.fixture_path(variant).exists():
                raise RuntimeError(f'fixture {variant} already exists')
            if step.get('variants') and variant not in step['variants']:
                continue
            artifact = rule.get('source_artifact')
            if rule.get('source_step'):
                artifact = next(s for s in self.ledger['steps'] if s['step'] == rule['source_step'])['summary']['artifact']
            source = light.read(artifact)
            if rule.get('source_sha256') and source.sha256 != rule['source_sha256']:
                raise RuntimeError(f'fixture source {variant} differs from the declared capture')
            forwards = rule['generation_forwards']
            if rule.get('up_to'):
                # Order only: all generation forwards up to the native limit on batches, no timing read.
                prefill = sum(not f['generation'] for f in source.of('forward'))
                forwards = min(forwards, 256 - prefill, len(source.generation_forwards()))
            receipts[variant] = light.export_fixture(source, self.fixture_path(variant), forwards,
                budget=rule['budget'], stage_limit=rule['stage_limit'], cpu_scale=rule['cpu_scale'], margin_ms=rule['margin_ms'],
                hold_primary=rule.get('hold_primary', False), hold_helper=rule.get('hold_helper', False),
                primary_query=rule.get('primary_query', False), helper_occupancy=rule.get('helper_occupancy', False),
                ordinary=rule.get('ordinary', True))
        self.ledger.setdefault('fixtures', {}).update(receipts)
        return dict(receipts=receipts, errors=[])

    def run_fixture(self, step: dict, output: pathlib.Path, pid: int, phase: dict) -> dict:
        variant, mode = step['variant'], step.get('capture', 'off')
        rule = self.protocol['fixtures'][variant]
        body = {'action': 'replay', 'workload_path': str(self.fixture_path(variant))}
        context = self.capture_context(step, output, mode, pid, phase)
        report = capture.operation(context, 'placement_replay', lambda: post(self.base, '/v1/minimax/placement', body, self.a.timeout))
        with output.open('x') as f:
            json.dump(dict(step=step, created_utc=utc(), report=report), f, indent=2)
            f.write('\n')
        errors = []
        forwards = self.ledger['fixtures'][variant]['generation_forwards']
        if report.get('generation_batches') != forwards:
            errors.append(f'fixture generation work is not {forwards} forwards')
        if rule.get('ordinary', True) and report.get('generation_rows') != forwards:
            errors.append(f'fixture generation work is not {forwards} one-row forwards')
        if bool(report.get('diagnostic_primary_query')) != bool(rule.get('primary_query', False)):
            errors.append('fixture submission query differs from the declared variant')
        if bool(report.get('helper_occupancy')) != bool(rule.get('helper_occupancy', False)):
            errors.append('fixture helper occupancy differs from the declared variant')
        if rule.get('helper_occupancy') and not report.get('helper_occupancy_launches'):
            errors.append('declared helper occupancy launched nothing')
        if report.get('primary_submit_queries') != (report.get('windows') if rule.get('primary_query') else 0):
            errors.append(f"fixture query count {report.get('primary_submit_queries')} does not match its windows {report.get('windows')}")
        summary = dict(variant=variant, capture=mode, generation_ms=report.get('generation_target_ms'), report=report)
        row = context['manifest']['control_operations'][-1]
        summary['artifact'] = row['engine_jsonl']
        summary['validation'] = row['validation']
        if mode in LIGHT_MODES and row['validation'].get('status') != 'validated':
            errors.append(f"capture invalid: {row['validation'].get('errors')}")
        summary['errors'] = errors
        return summary

    def run_request(self, step: dict, output: pathlib.Path, pid: int, phase: dict) -> dict:
        case = self.cases[step['case']]
        mode = step.get('capture', 'off')
        context = self.capture_context(step, output, mode, pid, phase)
        result = dict(step=step, created_utc=utc(), requests=[])
        payload = self.payload(case, step.get('max_tokens', self.protocol['max_tokens']), step.get('tune'))
        if step.get('seed') is not None:
            # (a seed per step: the protocol's one seed makes every repeat of an arm and prompt write the same text - 22 of
            # 24 repeat pairs were identical in the step-6 acceptance runs - so blocks sampled timing noise only, not text)
            payload['seed'] = int(step['seed'])
        # The measured interval starts at the first completed round at or after `measure_from` emitted tokens (the protocol's;
        # timed tests from 2026-11-09 use 128, caches warm - the engine records a sample there), 0 for older protocols.
        measure_from = int(step.get('measure_from', self.protocol.get('measure_from', 0)))
        screen.run_case(self.base, case, payload, self.a.timeout, None, result, pid, measure_from, context, trace_stream=False)
        item = result['requests'][-1]
        item['fresh_context_check'] = screen.fresh_context_check(item)
        if step.get('expect_reuse'):
            item['reuse_check'] = prefix_reuse_check(item)
        with output.open('x') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write('\n')
        final = final_stat(item)
        errors = []
        if not item.get('complete') or item.get('error'):
            errors.append(f"request incomplete: {item.get('error')}")
        if step.get('expect_reuse'):
            if not item['reuse_check']['verified']:
                errors.append(f"prefix reuse not verified: {item['reuse_check']['observed']}")
        elif not item['fresh_context_check']['verified']:
            errors.append('fresh context not verified')
        if (item.get('engine_stats') or {}).get('join') != 'one_new_sequence_and_matching_completion_count':
            errors.append('engine/API request attribution not verified')
        if self.protocol.get('speculation_off', True) and (final.get('draft_off') != 0 or final.get('spec_rounds') != 0):
            errors.append(f"speculation not off: draft_off={final.get('draft_off')} spec_rounds={final.get('spec_rounds')}")
        import hashlib
        row = context['manifest']['requests'][-1]
        summary = dict(case=step['case'], capture=mode, emitted=final.get('emitted'), generation_ms=final.get('elapsed_ms'),
                       finish=item.get('finish_reason'), api_wall_ms=item.get('api_wall_ms'), target_rounds=final.get('target_rounds'),
                       final_stat=final, artifact=row['engine_jsonl'], validation=row['validation'],
                       output_sha256=hashlib.sha256((item.get('reasoning_content', '') + '\0' + item.get('content', '')).encode()).hexdigest())
        interval = (item.get('engine_stats') or {}).get('generation_interval') or {}
        summary.update(measure_from=measure_from, interval_status=(item.get('engine_stats') or {}).get('interval_status'),
                       interval_start_emitted=interval.get('start_emitted'), interval_tokens=interval.get('tokens'),
                       interval_ms=interval.get('interval_ms'))
        if mode in LIGHT_MODES and row['validation'].get('status') != 'validated':
            errors.append(f"capture invalid: {row['validation'].get('errors')}")
        summary['errors'] = errors
        return summary

    def telemetry_start(self, step: dict):
        setting = step.get('telemetry')
        if not setting:
            return None
        rule = self.protocol['telemetry'] if setting is True else {**self.protocol['telemetry'], **setting}
        path = self.directory / f"{step['step']}.telemetry.json"
        process = subprocess.Popen([sys.executable, '-I', '-B', str(TELEMETRY), '--output', str(path), '--gpus', ','.join(map(str, rule['gpus'])),
                                    '--hz', str(rule['hz']), '--reasons-hz', str(rule.get('reasons_hz', 0))],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        time.sleep(float(rule.get('settle_s', 0.5)))
        if process.poll() is not None:
            raise RuntimeError(f'telemetry exited early: {process.stderr.read().decode(errors="replace")}')
        return process, path

    @staticmethod
    def telemetry_stop(running) -> str | None:
        if not running:
            return None
        process, path = running
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            raise RuntimeError('telemetry did not stop')
        if process.returncode or not path.exists():
            raise RuntimeError(f'telemetry failed: {process.stderr.read().decode(errors="replace")}')
        return str(path)

    def gpu_metrics_start(self, step: dict):
        setting = step.get('gpu_metrics')
        if not setting:
            return None
        report = self.directory / f"{step['step']}.gpu-metrics.nsys-rep"
        process = subprocess.Popen(['nsys', 'profile', f"--gpu-metrics-devices={setting['devices']}", f"--gpu-metrics-frequency={setting['hz']}",
                                    '--trace=none', '--sample=none', '--cpuctxsw=none', f'--output={report}', 'sleep', str(setting['seconds'])],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
        time.sleep(float(setting.get('settle_s', 3)))
        return process, report

    @staticmethod
    def gpu_metrics_stop(running) -> str | None:
        if not running:
            return None
        process, report = running
        process.wait(timeout=600)
        return str(report) if report.exists() else None

    def run_backing(self, step: dict) -> dict:
        """Allocates or releases the primary's DRAM copies. The engine refuses before allocating if its reserve would not hold."""
        began = time.monotonic()
        enable = step['action'] == 'backing-on'
        reply = post(self.base, '/v1/minimax/placement', {'action': step['action']}, self.a.timeout)
        output = self.directory / f"{step['step']}.json"
        with output.open('x') as f:
            json.dump(reply, f)
        inspection = post(self.base, '/v1/minimax/placement', {'action': 'inspect'}, self.a.timeout)
        errors = []
        if bool(inspection.get('primary_backing_enabled')) != enable:
            errors.append(f"backing state is {inspection.get('primary_backing_enabled')} after {step['action']}")
        if not inspection.get('ownership_complete'):
            errors.append('ownership is not complete after the backing change')
        return dict(path=str(output), engine_pid=reply.get('engine_pid'), errors=errors, step_seconds=time.monotonic() - began, restore=None,
                    backing_physical_bytes=inspection.get('primary_backing_physical_bytes'), backing_valid_records=inspection.get('primary_backing_valid_records'),
                    windows_physical_before=reply.get('pre_allocation_windows_physical'), windows_physical_after=reply.get('post_allocation_windows_physical'))

    def run_host(self, step: dict) -> dict:
        """One allowlisted host operation through the user-registered MayaHostControl task; the result file is the receipt."""
        import native_host_control as host
        began = time.monotonic()
        result = host.request(step['actions'])
        output = self.directory / f"{step['step']}.json"
        with output.open('x') as f:
            json.dump(result, f, indent=1)
        errors = [] if result.get('ok') else [f"host operation failed: {result.get('error') or [a.get('output') for a in result.get('actions', []) if a.get('exit')]}"]
        return dict(path=str(output), errors=errors, step_seconds=time.monotonic() - began, restore=None, host_ok=result.get('ok'))

    def run_step(self, step: dict, pid: int, phase: dict) -> dict:
        kind = step['kind']
        if kind == 'backing':
            return self.run_backing(step)
        if kind == 'host':
            return self.run_host(step)
        if kind not in ('request', 'fixture'):
            return super().run_step(step, pid, phase)
        output = self.directory / f"{step['step']}.json"
        metrics = self.gpu_metrics_start(step)
        telemetry = self.telemetry_start(step)
        began = time.monotonic()
        try:
            restore = self.restore(step['phase']) if step.get('restore', True) else None
            if kind == 'request':
                # (a request may be traced too: plain generation keeps prep_graph and early_copy, which a fixture replay or
                #  detailed capture turns off)
                if step.get('nsys_report'):
                    self.nsys('start', phase, '--sample=none', '--cpuctxsw=none',
                              f"--output={self.directory / step['nsys_report']}", '--force-overwrite=true')
                try:
                    summary = self.run_request(step, output, pid, phase)
                finally:
                    if step.get('nsys_report'):
                        self.nsys('stop', phase)
            else:
                if step.get('nsys_report'):
                    self.nsys('start', phase, '--sample=none', '--cpuctxsw=none',
                              f"--output={self.directory / step['nsys_report']}", '--force-overwrite=true')
                try:
                    summary = self.run_fixture(step, output, pid, phase)
                finally:
                    if step.get('nsys_report'):
                        self.nsys('stop', phase)
        finally:
            telemetry_path = self.telemetry_stop(telemetry)
            metrics_path = self.gpu_metrics_stop(metrics)
        summary['step_seconds'] = time.monotonic() - began
        summary['restore'] = restore
        summary['telemetry'] = telemetry_path
        summary['gpu_metrics'] = metrics_path
        return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True, type=pathlib.Path)
    ap.add_argument('--run-dir', required=True, type=pathlib.Path)
    ap.add_argument('--phase', required=True)
    ap.add_argument('--base-url', default='http://127.0.0.1:8097')
    ap.add_argument('--timeout', type=float, default=900)
    ap.add_argument('--continue-after-failure', action='store_true')
    a = ap.parse_args()
    if not a.run_dir.is_dir():
        ap.error('run directory must already exist')
    if a.run_dir.resolve().is_relative_to(ROOT):
        ap.error('run directory must be outside the source repository')
    return FollowUp(a).run()


if __name__ == '__main__':
    sys.exit(main())
