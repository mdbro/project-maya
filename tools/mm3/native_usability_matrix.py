#!/usr/bin/env python3
"""Bounded usability comparison of frozen arms and seeds on the resident engine, against control.

Settings that change which experts run or which drafts are kept change the text, so speed alone cannot select them.
Each (arm, seed) restores the learned-state origin, runs the existing delivery workflows, then a few complete
tasks with answers that can be checked. Outputs are kept raw for the primary agent's own review. Three seeds cannot
establish broad quality: this catches visible damage and nothing more.

A protocol with "builtins": true (since 2026-11-14) runs serving's own settings: each request sends only fresh_context,
the protocol's tune and the arm's keys, so every other setting is the engine's built-in and, with no "spec", the drafting
planner picks each round's drafts (its table the start's, or the arm's spec_plan_file) instead of a fixed width. The
values the delivery workflows' command line needs are read through engine_builtins.py, not from the protocol, and the
workflows send only the same keys (native_delivery.py --tune-only). "delivery": false leaves the delivery workflows out
(the tasks only). Each task row records the settings the request reported (its final statistics with the keys only its
last STAT line carries), so a run can be checked against the arm it declared.

Since 2026-11-14 (the usability audit, runs/usability-audit-20261114): the answer checks are tools/mm3/usability_checks.py, each
with a label (right, wrong, read, not scored) and its reason; an order-jobs check separates the answer's failures from the
harness's; each row keeps the request's sampling parameters and each tasks file the engine's pid, so tools/mm3/usability_eval.py
can refuse a verdict when the arms did not share one engine start or an answer did not report its arm's settings.
"""
from __future__ import annotations
import argparse
import ast
import json
import pathlib
import re
import shutil
import subprocess
import sys

import native_calibrate
import native_delivery as delivery
import usability_checks
from native_followup import final_stat
from native_shared_diagnostic import post, utc

TOOLS = pathlib.Path(__file__).resolve().parent
ALLOWED_IMPORTS = ('heapq', 'collections', 'collections.abc', 'typing', 'bisect', 'itertools', 'functools', 'graphlib', '__future__')
FORBIDDEN_NAMES = frozenset(('eval', 'exec', 'compile', 'open', 'globals', 'locals', 'vars', 'dir', 'getattr', 'setattr', 'delattr', 'hasattr', 'breakpoint', 'input', 'memoryview',
                             'type', 'object', 'super', 'help', 'exit', 'quit', 'classmethod', 'staticmethod', 'property'))
FORBIDDEN_ATTRIBUTES = frozenset(('sys', 'os', 'modules', 'builtins', 'subprocess', 'importlib', 'types', 'gc', 'inspect', 'gi_frame', 'gi_code', 'cr_frame', 'cr_code', 'ag_frame', 'ag_code',
                                  'f_globals', 'f_locals', 'f_builtins', 'f_back', 'f_code', 'tb_frame', 'tb_next'))
# Generated code is never trusted. The sandbox is what isolates it: no network, no home directory, read-only system files,
# an empty environment and an empty temporary directory. The AST rules and short builtin list are defence in depth only.
SANDBOX = ('--ro-bind', '/usr', '/usr', '--ro-bind', '/lib', '/lib', '--ro-bind', '/lib64', '/lib64', '--ro-bind', '/bin', '/bin', '--proc', '/proc', '--dev', '/dev',
           '--unshare-all', '--die-with-parent', '--new-session', '--tmpfs', '/tmp', '--chdir', '/tmp')
# The runner prints one JSON line an event, flushed at once, so a function that never returns still leaves what happened before:
# ready (the sandbox and the runner started), top (the answer's top-level code ran, or its error), check (one executed case),
# start (a case began), done. A NameError for a builtin the runner withholds is the harness's limit, not the answer's error.
ORDER_BUILTINS = ('len', 'range', 'enumerate', 'zip', 'int', 'bool', 'list', 'sum', 'iter', 'next', 'set', 'dict', 'tuple', 'sorted', 'min', 'max', 'map', 'filter', 'any', 'all',
                  'str', 'isinstance', 'reversed', 'frozenset', 'abs', 'repr', 'format', 'hash', 'float', 'round', 'divmod', 'pow', 'ord', 'chr', 'slice', 'callable', 'id',
                  'Exception', 'ValueError', 'TypeError', 'KeyError', 'StopIteration', 'RuntimeError', 'IndexError', 'AttributeError', 'AssertionError', 'LookupError',
                  'ArithmeticError', 'ZeroDivisionError', 'OverflowError', 'RecursionError', 'NotImplementedError', 'NameError', 'UnboundLocalError', 'MemoryError')
ORDER_RUNNER = r"""import json,sys,resource
for limit,value in ((resource.RLIMIT_CPU,3),(resource.RLIMIT_AS,256<<20),(resource.RLIMIT_FSIZE,0),(resource.RLIMIT_NPROC,0),(resource.RLIMIT_NOFILE,16)):
 resource.setrlimit(limit,(value,value))
s=json.loads(sys.stdin.read())
allowed=set(json.loads(sys.argv[1]))
names=json.loads(sys.argv[2])
real=__import__
def guarded(name,globals=None,locals=None,fromlist=(),level=0):
 if name not in allowed:raise ImportError(name)
 return real(name,globals,locals,fromlist,level)
import builtins
withheld=set(dir(builtins))-set(names)-{'print','__import__'}
out=sys.stdout
def emit(o):
 out.write(json.dumps(o,default=repr)+'\n');out.flush()
def harness(e):
 return isinstance(e,NameError) and getattr(e,'name',None) in withheld
def text(e):
 return type(e).__name__+': '+str(e)[:200]
scope={'__builtins__':{**{n:getattr(builtins,n) for n in names},'__import__':guarded,'print':lambda *a,**k:None},'__name__':'generated'}
emit({'ready':True})
top=None
try:
 exec(compile(s,'<generated>','exec'),scope)
except BaseException as e:
 top={'error':text(e),'harness':harness(e)}
f=scope.get('order_jobs')
emit({'top':top,'function':callable(f)})
if callable(f):
 cases=[({},[]),({'b':['a'],'a':[]},['a','b']),({'c':['a','b']},['a','b','c']),({'b':['a','a']},['a','b']),({'b':('a',)},['a','b']),({'z':[],'y':[],'x':['z']},['y','z','x']),({'a':['a']},'ValueError'),({'a':['b'],'b':['a']},'ValueError')]
 for x,want in cases:
  old=json.dumps(x,sort_keys=True)
  emit({'start':old,'expected':want})
  try:
   got=f(x);r=dict(input=old,expected=want,actual=got if isinstance(got,list) else repr(got)[:200],**{'pass':got==want and json.dumps(x,sort_keys=True)==old})
  except ValueError as e:
   r=dict(input=old,expected=want,actual='ValueError',**{'pass':want=='ValueError'})
  except BaseException as e:
   r=dict(input=old,expected=want,error=text(e),harness=harness(e),**{'pass':False})
  emit({'check':r})
emit({'done':True})"""


def fenced_code(text: str) -> str:
    """The answer's code block (usability_checks.fenced_code): the last block that DEFINES order_jobs, any fence style."""
    return usability_checks.fenced_code(text)


def generated_code_problem(code: str) -> str | None:
    """The first construct this checker will not run, or None. Rejected code counts as not passing."""
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return f'SyntaxError: {exc}'
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            return 'class definition'
        if isinstance(node, ast.Import) and any(a.name not in ALLOWED_IMPORTS for a in node.names):
            return f'import {[a.name for a in node.names]}'
        if isinstance(node, ast.ImportFrom):
            if node.level or node.module not in ALLOWED_IMPORTS or any(a.name == '*' or a.name in FORBIDDEN_ATTRIBUTES or a.name.startswith('_') for a in node.names):
                return f'import from {node.module}'
        if isinstance(node, ast.Attribute) and (node.attr.startswith('_') or node.attr in FORBIDDEN_ATTRIBUTES):
            return f'attribute {node.attr}'
        if isinstance(node, ast.Name) and node.id != '__name__' and (node.id.startswith('__') or node.id in FORBIDDEN_NAMES):
            return f'name {node.id}'
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and '__' in node.value and node.value != '__main__':
            return 'string holding a dunder'
    return None


def sandboxed(arguments: list[str], stdin: str, timeout: float = 8) -> subprocess.CompletedProcess:
    """Runs the system Python in the sandbox with an empty environment. Refuses to run without the sandbox."""
    bwrap = shutil.which('bwrap')
    if not bwrap:
        raise FileNotFoundError('bwrap')
    return subprocess.run([bwrap, *SANDBOX, '/usr/bin/python3', '-I', *arguments], input=stdin, text=True, capture_output=True, timeout=timeout, env={})


def _order_result(status: str, label: str, fault: str | None, why: str, **more) -> dict:
    return dict(status=status, label=label, fault=fault, why=why, **more)


def check_order_jobs(text: str) -> dict:
    """Runs the generated function only inside the sandbox, after the AST rules. status keeps its old values (pass, fail,
    rejected, unchecked, execution_error); label and fault say whose failure it is:
      right       every executed check passes
      wrong       the ANSWER fails (fault model): no function, code that does not parse, an executed check fails, or the function
                  does not return within the 3 s CPU limit
      not scored  the HARNESS could not score it (fault harness): the AST rules refuse a construct, no sandbox, the runner did not
                  start, the wall clock ran out before the CPU limit (machine load), or the only failures are NameErrors for
                  builtins the runner withholds. Never counted as the model's failure: a reader, or a re-check, decides."""
    R, W, N = usability_checks.RIGHT, usability_checks.WRONG, usability_checks.NOT_SCORED
    code = fenced_code(text)
    if not code.strip():
        return _order_result('rejected', W, 'model', 'the answer has no code', reason='no code or code too long')
    if len(code) > 16000:
        return _order_result('rejected', W, 'model', 'the code is longer than 16,000 characters', reason='no code or code too long')
    problem = generated_code_problem(code)
    if problem and problem.startswith('SyntaxError'):
        return _order_result('rejected', W, 'model', 'the code does not parse: ' + problem, reason=problem)
    if problem:
        return _order_result('rejected', N, 'harness', 'the checker\'s rules refuse to run it: ' + problem, reason=problem)
    timed_out = False
    try:
        proc = sandboxed(['-c', ORDER_RUNNER, json.dumps(ALLOWED_IMPORTS), json.dumps(ORDER_BUILTINS)], json.dumps(code))
        stdout, stderr, code_ = proc.stdout, proc.stderr, proc.returncode
    except FileNotFoundError:
        return _order_result('unchecked', N, 'harness', 'no sandbox available, so the generated code was not run',
                             reason='no sandbox available, so the generated code was not run')
    except subprocess.TimeoutExpired as exc:
        timed_out, code_ = True, None
        stdout = exc.stdout.decode(errors='replace') if isinstance(exc.stdout, bytes) else (exc.stdout or '')
        stderr = exc.stderr.decode(errors='replace') if isinstance(exc.stderr, bytes) else (exc.stderr or '')
    except OSError as exc:
        return _order_result('execution_error', N, 'harness', f'the sandbox did not start: {type(exc).__name__}: {exc}', reason=f'{type(exc).__name__}: {exc}')
    events = []
    for line in stdout.splitlines():
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    if not any(e.get('ready') for e in events):
        return _order_result('execution_error', N, 'harness', 'the runner did not start inside the sandbox', stderr=stderr[-1500:])
    top = next((e for e in events if 'function' in e), None)
    checks = [e['check'] for e in events if 'check' in e]
    started = [e for e in events if 'start' in e]
    done = any(e.get('done') for e in events)
    if top is None:
        if timed_out:
            return _order_result('execution_error', N, 'harness', 'the wall clock ran out before the 3 s CPU limit while the answer\'s top-level code ran (machine load): re-check')
        return _order_result('fail', W, 'model', 'the answer\'s top-level code did not finish (the 3 s CPU limit, or it ended the interpreter)', checks=[], stderr=stderr[-500:])
    if not top['function']:
        t = top.get('top') or {}
        if t.get('harness'):
            return _order_result('execution_error', N, 'harness', 'the runner withholds a builtin the top-level code uses: ' + t.get('error', ''))
        return _order_result('fail', W, 'model', 'no function named order_jobs is defined' + (f" (top-level error: {t['error']})" if t.get('error') else ''), checks=[])
    if not done:                                   # a case began and never reported
        hung = started[len(checks)] if len(started) > len(checks) else None
        if timed_out:
            return _order_result('execution_error', N, 'harness', 'the wall clock ran out before the 3 s CPU limit (machine load): re-check', checks=checks)
        if hung is not None:
            checks.append(dict(input=hung['start'], expected=hung.get('expected'), error='did not return: stopped at the 3 s CPU limit or the interpreter ended', **{'pass': False}))
        return _order_result('fail', W, 'model', 'the function did not return on ' + (hung['start'] if hung else 'a case') + f' (exit {code_})', checks=checks)
    failing = [c for c in checks if not c['pass']]
    if not failing:
        return _order_result('pass', R, None, 'the eight executed checks pass', checks=checks)
    if all(c.get('harness') for c in failing):
        return _order_result('execution_error', N, 'harness', 'the only failures are builtins the runner withholds: ' + '; '.join(sorted({c['error'] for c in failing})), checks=checks)
    return _order_result('fail', W, 'model', 'executed checks fail: ' + '; '.join(c['input'] for c in failing if not c.get('harness')), checks=checks)


def order_jobs_label(check: dict | None, finish: str | None = 'stop') -> str:
    """right, wrong, not scored or cut off (usability_checks.order_jobs_label, for a check stored before or after 2026-11-14)."""
    return usability_checks.order_jobs_label(check, finish)['label']


# The muffins task (46 muffins in boxes of 4 and 6, at most 10 boxes): 3 combinations, (7,3), (4,5), (1,7) boxes of 4 and
# of 6. (10,1) also makes 46 muffins but needs 11 boxes. The check is usability_checks.check_muffins. Its history: four
# checkers in a row marked right answers wrong (every real answer until 2026-11-14 13:47 PST; then exclusion wordings, 97527cc
# and cb4f834; then 3adf4455, which took the LAST stated count and read "the 10-box-of-4 plus 1-box-of-6 combination" as a
# count of 6 and "3 boxes for 6" as no pair - 4 of the 89 distinct answers of 2026-11-07..10). This one never says wrong
# without a stated count other than 3; what it cannot read goes to a reader.
MUFFIN_VALID = usability_checks.MUFFIN_VALID


def check_muffins(text: str, finish: str | None = 'stop') -> dict:
    """usability_checks.check_muffins: label right | wrong | read | cut off, final_count, pairs, why. states_three, pairs_found,
    expected and correct are kept for the scripts that read them (correct is label == right: read is NOT wrong - see label)."""
    v = usability_checks.check_muffins(text, finish)
    return dict(label=v['label'], verdict=v['label'], why=v['why'], final_count=v['final_count'], states_three=v['final_count'] == 3, pairs_found=v['pairs'],
                expected=[list(p) for p in MUFFIN_VALID], correct=v['label'] == usability_checks.RIGHT)


SETTINGS_SEEN = ('spec_policy', 'spec_ell', 'spec_plan', 'spec_plan_file', 'routing_prior_bonus', 'routing_dampen', 'routing_floor',
                 'helper_live', 'third', 'prompt_path', 'warnings',
                 'dense_fp8', 'dense_fp8_kernel', 'dense_fp8_launches', 'dense_fp8_fallbacks',   # (kernel item 3's 8-bit path, 2026-11-14)
                 'head_lowp', 'head_lowp_format',   # (kernel item 10's head copy, 2026-11-14)
                 'draft_proj', 'draft_graph',   # (kernel item 6's drafter projections and graph form, 2026-11-14)
                 'cpu_kernel', 'helper_kernel', 'helper_kernel_one', 'third_kernel', 'third_kernel_one', 'helper_q_rows',
                 'third_sm89')   # (the lanes' arithmetic: the checkpoint-precision check of kbatch2, 2026-11-14)


def request_tune(protocol: dict, arm: str) -> dict:
    """The strata_tune a request of this arm sends: with "builtins", fresh_context, the protocol's tune and the arm's keys
    only (the engine's built-ins and its planner decide the rest); otherwise the full tune merged with the arm, as before."""
    if protocol.get('builtins'):
        return {'fresh_context': 1, **(protocol.get('tune') or {}), **protocol['arms'][arm]}
    return {**protocol['tune'], **protocol['arms'][arm]}


def command_values(protocol: dict, tune: dict) -> dict:
    """The values the delivery command line takes (bonus, lenience, width, draft confidence): with "builtins", the engine's
    built-ins for a drafting request (engine_builtins.request_builtins) under the request's own keys."""
    if protocol.get('builtins'):
        import engine_builtins
        return {**engine_builtins.request_builtins(True), **tune}
    return tune


def settings_seen(item: dict) -> dict:
    """What the request reported for the settings an arm may change, and the drafts offered a drafting round."""
    f = final_stat(item)
    out = {k: f.get(k) for k in SETTINGS_SEEN}
    rounds = f.get('spec_rounds') or 0
    out['drafts_per_round'] = (f.get('draft_off') or 0) / rounds if rounds else None
    return out


def run_entry(a, protocol: dict, entry: dict, origin: str | None) -> dict:
    arm, seed = entry['arm'], entry['seed']
    name = f'{arm}-s{seed}'
    summary_path = a.run_dir / f'tasks-{name}.json'
    if summary_path.exists():
        return json.loads(summary_path.read_text())
    reply = post(a.base_url, '/v1/minimax/placement', {'action': 'restore'}, a.timeout)
    signature = native_calibrate.learned_state_signature(reply)
    if origin and signature != origin:
        raise RuntimeError(f'restored learned state differs from the origin: {signature}')
    tune = request_tune(protocol, arm)
    values = command_values(protocol, tune)
    delivery_path = a.run_dir / f'delivery-{name}.json'
    if protocol.get('delivery', True):
        # A T=1 matrix (spec 1, no drafting) passes a nominal width of 1 and nominal drafting knobs: the full tune is merged last
        # in native_delivery, so its spec of 1 decides. Matrices with drafting are unchanged. With "builtins" the workflows
        # send only the request's keys (--tune-only): the command line's values are then nominal.
        command = [sys.executable, '-B', str(TOOLS / 'native_delivery.py'), '--base-url', a.base_url, '--arm', 'selected', '--routing', 'prior', '--prior-bonus', str(values['routing_prior_bonus']),
                   '--ell', str(values.get('spec_ell', 0.4)), '--width', str(min(7, max(1, int(values['spec']) - 1))), '--draft-confidence', str(values.get('spec_min_p', 0.3)), '--tokens', str(protocol['delivery_tokens']),
                   '--reasoning-tokens', str(protocol['reasoning_tokens']), '--seed', str(seed), '--engine-pid', str(a.engine_pid), '--tune-json', json.dumps(tune),
                   '--note', f'usability matrix {protocol["experiment_id"]}: origin restored, arm {arm}, seed {seed}', '--output', str(delivery_path)] + \
            (['--tune-only'] if protocol.get('builtins') else [])
        print(f'  {name}: delivery workflows start {utc()}', flush=True)
        code = subprocess.run(command, cwd=TOOLS, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=3600)
        print(f'  {name}: delivery workflows exit {code.returncode} {utc()}', flush=True)
        delivery_exit, delivery_stderr, delivery_file = code.returncode, code.stderr[-800:], str(delivery_path)
    else:
        delivery_exit, delivery_stderr, delivery_file = None, '', 'not run (the protocol\'s delivery is false)'
    result: dict = dict(arm=arm, seed=seed, created_utc=utc(), engine_pid=a.engine_pid, tune=tune, delivery=delivery_file, delivery_exit=delivery_exit, delivery_stderr=delivery_stderr, measurement_protocol=dict(start_emitted=0), requests=[], tasks=[])
    with (a.run_dir / f'tasks-{name}.sse.jsonl').open('x') as raw:
        for task in protocol['tasks']:
            req = delivery.payload([{'role': 'user', 'content': task['prompt']}], tune, task['max_tokens'], seed)
            item = delivery.invoke(a.base_url, task['id'], req, a.timeout, raw, result, a.engine_pid)
            final = (item.get('engine_stats') or {}).get('final') or {}
            row = dict(id=task['id'], complete=item.get('complete', False), finish=item.get('finish_reason'), emitted=final.get('emitted'), generation_ms=final.get('elapsed_ms'),
                       draft_acc=final.get('draft_acc'), plan_rounds=final.get('plan_rounds'),   # (drafting: accepted drafts and planned rounds)
                       request={k: v for k, v in req.items() if k not in ('messages', 'strata_tune', 'model', 'stream', 'return_progress')},   # (the sampling parameters sent)
                       settings=settings_seen(item), content=item.get('content', ''), reasoning=item.get('reasoning_content', ''))
            if task['id'] == 'order-jobs':
                row['check'] = check_order_jobs(item.get('content', ''))
            elif task['id'] == 'muffins':
                row['check'] = check_muffins(item.get('content', ''), row['finish'])
            elif task['id'] == 'action-items':
                row['check'] = usability_checks.check_action_items(item.get('content', ''), row['finish'])
            result['tasks'].append(row)
            # Each finished answer is on disk at once (the summary below marks the entry done for a resume).
            partial = {k: v for k, v in result.items() if k != 'requests'}
            (a.run_dir / f'tasks-{name}.partial.json').write_text(json.dumps(partial, indent=1, ensure_ascii=False) + '\n')
            print(f'  {name} {task["id"]}: finish {row["finish"]}, {row["emitted"]} tokens', flush=True)
    result.pop('requests')
    summary_path.write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--protocol', required=True, type=pathlib.Path)
    ap.add_argument('--run-dir', required=True, type=pathlib.Path)
    ap.add_argument('--base-url', default='http://127.0.0.1:8097')
    ap.add_argument('--engine-pid', required=True, type=int)
    ap.add_argument('--origin-signature', default=None)
    ap.add_argument('--timeout', type=float, default=1200)
    a = ap.parse_args()
    protocol = json.loads(a.protocol.read_text())
    for entry in protocol['sequence']:
        result = run_entry(a, protocol, entry, a.origin_signature)
        print(entry['arm'], entry['seed'], 'delivery exit', result['delivery_exit'], [(t['id'], t['complete'], (t.get('check') or {}).get('label')) for t in result['tasks']], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
