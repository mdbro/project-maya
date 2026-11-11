#!/usr/bin/env python3
"""Bounded generation-only diagnostic campaign for the shared first-stage model.

Runs the declared steps of a protocol file against an already resident native
server, in order, with speculation off and prefill treated as setup. It never
starts an engine. Every step is journaled, so a failed or capped campaign keeps
its receipts and cannot silently re-run completed measurements. The sequence is
the whole allocation: there is no reserve, so an ambiguity stays unresolved.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import time
import types
import urllib.error
import urllib.request

import native_calibrate
import native_capture as capture
import native_light as light
import native_screen as screen

ROOT = pathlib.Path(__file__).resolve().parents[2]
FIXTURE_HOLDS = {'control': (False, False), 'held': (True, True)}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def post(base: str, endpoint: str, body: dict, timeout: float) -> dict:
    request = urllib.request.Request(base + endpoint, data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'{endpoint} {body.get("action")}: HTTP {exc.code}: {exc.read().decode(errors="replace")}') from exc


def process_environment(pid: int) -> dict[str, str]:
    rows = (pathlib.Path('/proc') / str(pid) / 'environ').read_bytes().split(b'\0')
    return dict(row.decode(errors='replace').split('=', 1) for row in rows if b'=' in row)


def injected_libraries(pid: int) -> list[str]:
    maps = (pathlib.Path('/proc') / str(pid) / 'maps').read_text(errors='replace')
    return sorted({line.split()[-1] for line in maps.splitlines()
                   if any(name in line.lower() for name in ('nsys', 'nsight', 'cupti', 'toolsinjection'))})


def precheck(phase: dict, pid: int, control: pathlib.Path) -> dict:
    """Refuse to measure unless the process matches the phase's declared instrumentation."""
    environment, libraries = process_environment(pid), injected_libraries(pid)
    errors = []
    if environment.get('MM3_SERVE_MONITORING') != '0':
        errors.append('engine monitoring is not disabled')
    if environment.get('MM3_CAPTURE_CONTROL') != str(control):
        errors.append(f'engine capture control is not {control}')
    if control.exists():
        errors.append('capture control file is present before the campaign step')
    if phase['profiler'] == 'absent' and (libraries or 'CUDA_INJECTION64_PATH' in environment):
        errors.append(f'profiler libraries present in an uninstrumented phase: {libraries}')
    if phase['profiler'] == 'launched' and not libraries:
        errors.append('declared Nsight launch has no profiler library in the engine process')
    executable = pathlib.Path('/proc') / str(pid) / 'exe'
    result = dict(pid=pid, executable=os.readlink(executable), executable_sha256=sha256(executable),
                  profiler_libraries=libraries, monitoring=environment.get('MM3_SERVE_MONITORING'),
                  replay_capture=environment.get('MM3_REPLAY_CAPTURE'), errors=errors)
    if errors:
        raise RuntimeError('precheck failed: ' + '; '.join(errors))
    return result


def gpu_sample() -> str:
    # Called only between phases, never during a measured step.
    return subprocess.run(['nvidia-smi', '--query-gpu=index,name,pstate,clocks.sm,clocks.mem,temperature.gpu,power.draw,memory.used',
                           '--format=csv,noheader'], capture_output=True, text=True, check=False).stdout


class Campaign:
    def __init__(self, a):
        self.a = a
        self.protocol_path = a.protocol.resolve()
        self.protocol = json.loads(self.protocol_path.read_bytes())
        self.directory = a.run_dir.resolve()
        self.base = a.base_url.rstrip('/')
        self.ledger_path = self.directory / 'ledger.json'
        self.control = self.directory / 'capture-control'
        self.cases = {c['id']: c for c in json.loads(screen.FIXTURE.read_text())['cases']}
        self.cases.update({c['id']: {**c, 'checks': c.get('checks', [])} for c in self.protocol['conditioning']})
        self.ledger = json.loads(self.ledger_path.read_text()) if self.ledger_path.exists() else dict(
            protocol=str(self.protocol_path), protocol_sha256=sha256(self.protocol_path), created_utc=utc(),
            warmed_seconds=0.0, steps=[], phases={}, fixtures={}, source_capture=None)
        if self.ledger['protocol_sha256'] != sha256(self.protocol_path):
            raise ValueError('protocol changed after the campaign began')

    @property
    def cap(self) -> float:
        return float(self.protocol['warmed_time_cap_seconds'])

    def save(self) -> None:
        temporary = self.ledger_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(self.ledger, indent=2, allow_nan=False) + '\n')
        os.replace(temporary, self.ledger_path)

    def origin_key(self, phase: str) -> str:
        return f'origin:{phase}'

    def restore(self, phase: str) -> dict:
        began = time.monotonic()
        reply = post(self.base, '/v1/minimax/placement', {'action': 'restore'}, self.a.timeout)
        wall = time.monotonic() - began
        signature = native_calibrate.learned_state_signature(reply)
        expected = self.ledger['phases'][phase]['origin_signature']
        if signature != expected:
            raise RuntimeError(f'restored learned state differs from the declared origin: {signature} != {expected}')
        return dict(wall_seconds=wall, signature=signature, engine_pid=reply.get('engine_pid'))

    def capture_context(self, step: dict, output: pathlib.Path, mode: str, pid: int, phase: dict):
        namespace = types.SimpleNamespace(
            capture_mode=mode, capture_control=self.control, experiment_id=self.protocol['experiment_id'],
            run_id=step['step'], plan_item='shared-model-diagnostic-study', nsys_report=step.get('nsys_report'),
            profiling_session=phase.get('session') if mode == 'detailed' else None,
            model_revision=self.protocol.get('model_revision'), tokenizer_revision=self.protocol.get('tokenizer_revision'),
            capture_conditioning=f"declared origin of phase {step['phase']}; see {self.ledger_path}")
        return capture.init(namespace, output, pid)

    def payload(self, case: dict, max_tokens: int) -> dict:
        p = self.protocol
        return {'model': p['model_name'], 'messages': [{'role': 'user', 'content': case['prompt']}], 'stream': True,
                'return_progress': True, 'max_tokens': max_tokens, 'temperature': 1, 'top_p': 0.95, 'top_k': 0,
                'min_p': 0, 'seed': p['seed'], 'reasoning_effort': p['reasoning_effort'], 'loop_guard': False,
                'strata_tune': p['tune']}

    def run_placement(self, step: dict) -> dict:
        body = {'action': step['action'], 'profile_path': self.protocol['placement_profile']}
        if step.get('order'):                              # (three: which card takes the hottest tier, the next, the rest)
            body['order'] = step['order']
        reply = post(self.base, '/v1/minimax/placement', body, self.a.timeout)
        output = self.directory / f"{step['step']}.json"
        with output.open('x') as f:
            json.dump(reply, f)
        errors = [] if reply.get('ownership_complete') else ['placement ownership is not complete']
        return dict(path=str(output), engine_pid=reply.get('engine_pid'), errors=errors,
                    applied_profile_sha256=reply.get('applied_profile_sha256'))

    def run_request(self, step: dict, output: pathlib.Path, pid: int, phase: dict) -> dict:
        case = self.cases[step['case']]
        mode = step.get('capture', 'off')
        context = self.capture_context(step, output, mode, pid, phase) if mode != 'off' else None
        result = dict(step=step, created_utc=utc(), requests=[])
        payload = self.payload(case, step.get('max_tokens', self.protocol['max_tokens']))
        screen.run_case(self.base, case, payload, self.a.timeout, None, result, pid, 0, context, trace_stream=False)
        item = result['requests'][-1]
        item['fresh_context_check'] = screen.fresh_context_check(item)
        with output.open('x') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            f.write('\n')
        final = (item.get('engine_stats') or {}).get('final') or {}
        errors = []
        if not item.get('complete') or item.get('error'):
            errors.append(f"request incomplete: {item.get('error')}")
        if not item['fresh_context_check']['verified']:
            errors.append('fresh context not verified')
        if (item.get('engine_stats') or {}).get('join') != 'one_new_sequence_and_matching_completion_count':
            errors.append('engine/API request attribution not verified')
        # Speculation off means zero proposals and zero speculative rounds in every arm.
        if final.get('draft_off') != 0 or final.get('spec_rounds') != 0:
            errors.append(f"speculation not off: draft_off={final.get('draft_off')} spec_rounds={final.get('spec_rounds')}")
        summary = dict(case=step['case'], capture=mode, emitted=final.get('emitted'), generation_ms=final.get('elapsed_ms'),
                       finish=item.get('finish_reason'), api_wall_ms=item.get('api_wall_ms'),
                       target_rounds=final.get('target_rounds'), final_stat=final,
                       output_sha256=hashlib.sha256((item.get('reasoning_content', '') + '\0' + item.get('content', '')).encode()).hexdigest())
        if context:
            row = context['manifest']['requests'][-1]
            summary['artifact'] = row['engine_jsonl']
            summary['validation'] = row['validation']
            if mode == 'light' and row['validation'].get('status') != 'validated':
                errors.append(f"light capture invalid: {row['validation'].get('errors')}")
            if mode == 'light' and row['validation'].get('generation_rows_other_than_one'):
                errors.append('light capture contains generation forwards with more than one row')
        summary['errors'] = errors
        return summary

    def fixture_path(self, variant: str) -> pathlib.Path:
        return self.directory / f'fixture-{variant}.replay'

    def export_fixture(self, step: dict) -> dict:
        """The selection rule was declared before any timing was inspected."""
        rule = self.protocol['fixture']
        chosen = None
        for name in rule['source_steps']:
            entry = next((s for s in self.ledger['steps'] if s['step'] == name), None)
            if not entry:
                raise RuntimeError(f'fixture source step {name} has not run')
            source = light.read(entry['summary']['artifact'])
            if len(source.generation_forwards()) >= rule['generation_forwards']:
                chosen = (name, source)
                break
        if not chosen:
            raise RuntimeError('no declared source capture has enough complete generation forwards')
        name, source = chosen
        receipts = {}
        for variant, (hold_primary, hold_helper) in FIXTURE_HOLDS.items():
            receipts[variant] = light.export_fixture(source, self.fixture_path(variant), rule['generation_forwards'],
                budget=rule['budget'], stage_limit=rule['stage_limit'], cpu_scale=rule['cpu_scale'], margin_ms=rule['margin_ms'],
                hold_primary=hold_primary, hold_helper=hold_helper)
        self.ledger['fixtures'] = receipts
        self.ledger['source_capture'] = dict(step=name, sha256=source.sha256, path=str(source.path))
        return dict(source_step=name, source_sha256=source.sha256, receipts=receipts, errors=[])

    def run_fixture(self, step: dict, output: pathlib.Path, pid: int, phase: dict) -> dict:
        variant = step.get('variant', 'control')
        mode = step.get('capture', 'off')
        body = {'action': 'replay', 'workload_path': str(self.fixture_path(variant))}
        call = lambda: post(self.base, '/v1/minimax/placement', body, self.a.timeout)
        context = self.capture_context(step, output, mode, pid, phase) if mode != 'off' else None
        report = capture.operation(context, 'placement_replay', call) if context else call()
        with output.open('x') as f:
            json.dump(dict(step=step, created_utc=utc(), report=report), f, indent=2)
            f.write('\n')
        hold_primary, hold_helper = FIXTURE_HOLDS[variant]
        forwards = self.protocol['fixture']['generation_forwards']
        errors = []
        if report.get('generation_batches') != forwards or report.get('generation_rows') != forwards:
            errors.append(f"fixture generation work is not {forwards} one-row forwards")
        if bool(report.get('hold_primary_admissions')) != hold_primary or bool(report.get('hold_helper_admissions')) != hold_helper:
            errors.append('fixture admission hold differs from the declared variant')
        if hold_primary and report.get('primary_issued') != 0:
            errors.append('held primary admissions still issued exchanges')
        if hold_helper and report.get('helper_started') != 0:
            errors.append('held helper admissions still started a lease')
        summary = dict(variant=variant, capture=mode, generation_ms=report.get('generation_target_ms'), report=report)
        if context:
            row = context['manifest']['control_operations'][-1]
            summary['artifact'] = row['engine_jsonl']
            summary['validation'] = row['validation']
            if mode == 'light' and row['validation'].get('status') != 'validated':
                errors.append(f"light capture invalid: {row['validation'].get('errors')}")
        summary['errors'] = errors
        return summary

    def run_origin(self, phase: str) -> dict:
        reply = post(self.base, '/v1/minimax/placement', {'action': 'snapshot'}, self.a.timeout)
        signature = native_calibrate.learned_state_signature(reply)
        output = self.directory / f'origin-{phase}.json'
        with output.open('x') as f:
            json.dump(reply, f)
        self.ledger['phases'][phase]['origin_signature'] = signature
        return dict(signature=signature, path=str(output), engine_pid=reply.get('engine_pid'), errors=[])

    def run_step(self, step: dict, pid: int, phase: dict) -> dict:
        name, kind = step['step'], step['kind']
        output = self.directory / f'{name}.json'
        restore = None
        if kind in ('request', 'fixture') and step.get('restore', True):
            restore = self.restore(step['phase'])
        began = time.monotonic()
        if kind == 'request':
            summary = self.run_request(step, output, pid, phase)
        elif kind == 'fixture':
            if step.get('nsys_report'):
                self.nsys('start', phase, '--sample=none', '--cpuctxsw=none',
                          f"--output={self.directory / step['nsys_report']}", '--force-overwrite=true')
            try:
                summary = self.run_fixture(step, output, pid, phase)
            finally:
                if step.get('nsys_report'):
                    self.nsys('stop', phase)
        elif kind == 'placement':
            summary = self.run_placement(step)
        elif kind == 'origin':
            summary = self.run_origin(step['phase'])
        elif kind == 'export-fixture':
            summary = self.export_fixture(step)
        else:
            raise ValueError(f'unknown step kind {kind}')
        summary['step_seconds'] = time.monotonic() - began
        summary['restore'] = restore
        return summary

    def nsys(self, verb: str, phase: dict, *arguments: str) -> None:
        subprocess.run(['nsys', verb, f"--session={phase['session']}", *arguments], check=True, capture_output=True, timeout=600)

    def execute(self, step: dict, pid: int, phase: dict) -> bool:
        if self.ledger['warmed_seconds'] >= self.cap:
            self.ledger['stopped'] = dict(reason='warmed-time cap reached', before_step=step['step'], utc=utc())
            self.save()
            print(f"cap reached before {step['step']}: {self.ledger['warmed_seconds']:.1f}s", flush=True)
            return False
        began = time.monotonic()
        started = utc()
        try:
            summary = self.run_step(step, pid, phase)
        except Exception as exc:
            summary = dict(errors=[f'{type(exc).__name__}: {exc}'])
        wall = time.monotonic() - began
        self.ledger['warmed_seconds'] += wall
        entry = dict(step=step['step'], kind=step['kind'], phase=step['phase'], started_utc=started, wall_seconds=wall, summary=summary)
        self.ledger['steps'].append(entry)
        self.save()
        timing = summary.get('generation_ms')
        print(f"{step['step']}: {wall:.1f}s" + (f' generation {timing:.1f} ms' if timing else '') +
              f" warmed {self.ledger['warmed_seconds']:.1f}s" + (f" ERRORS {summary['errors']}" if summary.get('errors') else ''), flush=True)
        return not summary.get('errors')

    def begin_phase(self, name: str) -> tuple[dict, int]:
        phase = self.protocol['phases'][name]
        inspection = post(self.base, '/v1/minimax/placement', {'action': 'inspect'}, self.a.timeout)
        if not inspection.get('ownership_complete'):
            raise RuntimeError('engine ownership is not complete')
        pid = inspection['engine_pid']
        record = self.ledger['phases'].setdefault(name, {})
        check = precheck(phase, pid, self.control)
        if 'precheck' not in record:
            record.update(precheck=check, started_utc=utc(), gpu_start=gpu_sample())
        elif record['precheck']['pid'] != pid or record['precheck']['executable_sha256'] != check['executable_sha256']:
            raise RuntimeError('resident engine changed during the phase; in-process origin state is gone')
        self.save()
        return phase, pid

    def run(self) -> int:
        name = self.a.phase
        phase, pid = self.begin_phase(name)
        done = {s['step']: s for s in self.ledger['steps']}
        failed = [s['step'] for s in self.ledger['steps'] if s['summary'].get('errors')]
        if failed and not self.a.continue_after_failure:
            print(f'earlier step failed: {failed}; inspect before continuing', file=sys.stderr)
            return 1
        for step in self.protocol['sequence']:
            if step['phase'] != name or step['step'] in done:
                continue
            if not self.execute(step, pid, phase):
                return 1
        self.ledger['phases'][name]['finished_utc'] = utc()
        self.ledger['phases'][name]['gpu_end'] = gpu_sample()
        self.save()
        return 0


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
    return Campaign(a).run()


if __name__ == '__main__':
    sys.exit(main())
