#!/usr/bin/env python3
"""Bounded usability comparison of frozen arms and seeds on the resident engine, against control.

Settings that change which experts run or which drafts are kept change the text, so speed alone cannot select them.
Each (arm, seed) restores the learned-state origin, runs the existing delivery workflows, then a few complete
tasks with answers that can be checked. Outputs are kept raw for the primary agent's own review. Three seeds cannot
establish broad quality: this catches visible damage and nothing more.
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
ORDER_RUNNER = r"""import json,sys,resource
for limit,value in ((resource.RLIMIT_CPU,3),(resource.RLIMIT_AS,256<<20),(resource.RLIMIT_FSIZE,0),(resource.RLIMIT_NPROC,0),(resource.RLIMIT_NOFILE,16)):
 resource.setrlimit(limit,(value,value))
s=json.loads(sys.stdin.read())
allowed=set(json.loads(sys.argv[1]))
real=__import__
def guarded(name,globals=None,locals=None,fromlist=(),level=0):
 if name not in allowed:raise ImportError(name)
 return real(name,globals,locals,fromlist,level)
names=('len','range','enumerate','zip','int','bool','list','sum','iter','next','set','dict','tuple','sorted','min','max','map','filter','any','all','str','isinstance','reversed','frozenset','Exception','ValueError','TypeError','KeyError','StopIteration','RuntimeError','abs','repr','format','hash')
import builtins
scope={'__builtins__':{**{n:getattr(builtins,n) for n in names},'__import__':guarded,'print':lambda *a,**k:None},'__name__':'generated'}
exec(compile(s,'<generated>','exec'),scope)
f=scope['order_jobs']
cases=[({},[]),({'b':['a'],'a':[]},['a','b']),({'c':['a','b']},['a','b','c']),({'b':['a','a']},['a','b']),({'b':('a',)},['a','b']),({'z':[],'y':[],'x':['z']},['y','z','x']),({'a':['a']},'ValueError'),({'a':['b'],'b':['a']},'ValueError')]
r=[]
for x,want in cases:
 old=json.dumps(x,sort_keys=True)
 try:
  got=f(x);r.append(dict(input=old,expected=want,actual=got if isinstance(got,list) else repr(got),**{'pass':got==want and json.dumps(x,sort_keys=True)==old}))
 except ValueError as e:
  r.append(dict(input=old,expected=want,actual='ValueError',**{'pass':want=='ValueError'}))
 except Exception as e:
  r.append(dict(input=old,expected=want,error=type(e).__name__+': '+str(e),**{'pass':False}))
sys.stdout.write(json.dumps(r))"""


def fenced_code(text: str) -> str:
    blocks = re.findall(r'```(?:python|py)?\s*\n(.*?)```', text, re.S | re.I)
    for block in reversed(blocks):
        if 'order_jobs' in block:
            return block
    return blocks[-1] if blocks else text.strip()


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


def check_order_jobs(text: str) -> dict:
    """Runs the generated function only inside the sandbox, after the AST rules. Without bwrap it stays unchecked."""
    code = fenced_code(text)
    if not code or len(code) > 16000:
        return dict(status='rejected', reason='no code or code too long')
    problem = generated_code_problem(code)
    if problem:
        return dict(status='rejected', reason=problem)
    try:
        proc = sandboxed(['-c', ORDER_RUNNER, json.dumps(ALLOWED_IMPORTS)], json.dumps(code))
    except FileNotFoundError:
        return dict(status='unchecked', reason='no sandbox available, so the generated code was not run')
    except (OSError, subprocess.TimeoutExpired) as exc:
        return dict(status='execution_error', reason=f'{type(exc).__name__}: {exc}')
    if proc.returncode:
        return dict(status='execution_error', stderr=proc.stderr[-1500:])
    try:
        checks = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return dict(status='execution_error', reason=f'{exc}', stdout=proc.stdout[-500:])
    return dict(status='pass' if all(c['pass'] for c in checks) else 'fail', checks=checks)


def check_muffins(text: str) -> dict:
    """Expected: 3 combinations, (7,3), (4,5), (1,7). Reads only the final answer; the agent reviews the reasoning."""
    pairs = {(int(a), int(b)) for a, b in re.findall(r'(\d+)\s*(?:boxes of 4|4-box|×\s*4|x\s*4)[^\n]{0,30}?(\d+)\s*(?:boxes of 6|6-box|×\s*6|x\s*6)', text)}
    return dict(states_three=bool(re.search(r'\b(3|three)\b', text)), pairs_found=sorted(pairs), expected=[[1, 7], [4, 5], [7, 3]])


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
    tune = {**protocol['tune'], **protocol['arms'][arm]}
    delivery_path = a.run_dir / f'delivery-{name}.json'
    # A T=1 matrix (spec 1, no drafting) passes a nominal width of 1 and nominal drafting knobs: the full tune is merged last
    # in native_delivery, so its spec of 1 decides. Matrices with drafting are unchanged.
    command = [sys.executable, '-B', str(TOOLS / 'native_delivery.py'), '--base-url', a.base_url, '--arm', 'selected', '--routing', 'prior', '--prior-bonus', str(tune['routing_prior_bonus']),
               '--ell', str(tune.get('spec_ell', 0.4)), '--width', str(max(1, tune['spec'] - 1)), '--draft-confidence', str(tune.get('spec_min_p', 0.3)), '--tokens', str(protocol['delivery_tokens']),
               '--reasoning-tokens', str(protocol['reasoning_tokens']), '--seed', str(seed), '--engine-pid', str(a.engine_pid), '--tune-json', json.dumps(tune),
               '--note', f'usability matrix {protocol["experiment_id"]}: origin restored, arm {arm}, seed {seed}', '--output', str(delivery_path)]
    print(f'  {name}: delivery workflows start {utc()}', flush=True)
    code = subprocess.run(command, cwd=TOOLS, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, timeout=3600)
    print(f'  {name}: delivery workflows exit {code.returncode} {utc()}', flush=True)
    result: dict = dict(arm=arm, seed=seed, created_utc=utc(), delivery=str(delivery_path), delivery_exit=code.returncode, delivery_stderr=code.stderr[-800:], measurement_protocol=dict(start_emitted=0), requests=[], tasks=[])
    with (a.run_dir / f'tasks-{name}.sse.jsonl').open('x') as raw:
        for task in protocol['tasks']:
            req = delivery.payload([{'role': 'user', 'content': task['prompt']}], tune, task['max_tokens'], seed)
            item = delivery.invoke(a.base_url, task['id'], req, a.timeout, raw, result, a.engine_pid)
            final = (item.get('engine_stats') or {}).get('final') or {}
            row = dict(id=task['id'], complete=item.get('complete', False), finish=item.get('finish_reason'), emitted=final.get('emitted'), generation_ms=final.get('elapsed_ms'),
                       draft_acc=final.get('draft_acc'), plan_rounds=final.get('plan_rounds'),   # (drafting: accepted drafts and planned rounds)
                       content=item.get('content', ''), reasoning=item.get('reasoning_content', ''))
            if task['id'] == 'order-jobs':
                row['check'] = check_order_jobs(item.get('content', ''))
            elif task['id'] == 'muffins':
                row['check'] = check_muffins(item.get('content', ''))
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
        print(entry['arm'], entry['seed'], 'delivery exit', result['delivery_exit'], [(t['id'], t['complete'], (t.get('check') or {}).get('status')) for t in result['tasks']], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
