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
from native_followup import final_stat
from native_shared_diagnostic import post, utc

TOOLS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import usability_checks   # (the audited checks and decision rule, 2026-11-14: runs/usability-audit-20261114/)
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
names=('len','range','enumerate','zip','int','bool','list','sum','iter','next','set','dict','tuple','sorted','min','max','map','filter','any','all','str','isinstance','reversed','frozenset','Exception','ValueError','TypeError','KeyError','StopIteration','RuntimeError','abs','repr','format','hash','float','IndexError')
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
    """The answer's code block (usability_checks.fenced_code_v2): the last block that defines order_jobs, any fence style."""
    return usability_checks.fenced_code_v2(text)


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


def order_jobs_label(check: dict) -> str:
    """'right' (every executed check passes), 'wrong' (an executed check fails), or 'not scored' (rejected by the AST rules,
    an execution error, or no sandbox): a harness outcome, never counted as the model's failure; the reader sees it."""
    return {'pass': 'right', 'fail': 'wrong'}.get(check.get('status'), 'not scored')


# The muffins task (46 muffins in boxes of 4 and 6, at most 10 boxes): 3 combinations, (7,3), (4,5), (1,7) boxes of 4 and
# of 6. (10,1) also makes 46 muffins but needs 11 boxes. The checker judges the answer's final stated count of combinations
# and whether the three valid pairs appear; it does not try to read how an answer rules (10,1) out - three rounds of
# exclusion phrases (97527cc, cb4f834, then "violates the 10-box limit" and "needs 11 boxes") each missed the next wording.
MUFFIN_VALID = ((1, 7), (4, 5), (7, 3))
MUFFIN_WORDS = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
                'ten': 10}
_MUFFIN_NUM = r'(\d{1,2}|zero|one|two|three|four|five|six|seven|eight|nine|ten)'
# a stated count: "3 combinations", "**3 different combinations**", "three valid box combinations", "the answer is 3",
# "\boxed{3}" (primary); "3 solutions", "the three valid pairs" (secondary: answers use them for side counts too)
_MUFFIN_COUNT = re.compile(r'(?<![\w.])' + _MUFFIN_NUM + r'\**\s+(?:[A-Za-z][\w-]*\s+){0,3}?'
                           r'(combinations?|ways?|options?|possibilit(?:y|ies)|choices?|solutions?|pairs?)\b', re.I)
_MUFFIN_ANSWER = re.compile(r'\banswer[\s*]*(?:is|:)[\s*]*' + _MUFFIN_NUM + r'\b|\\boxed\{\s*' + _MUFFIN_NUM + r'\s*\}', re.I)
# the final section: after the last heading (markdown or a bold line) named Answer, Final answer, Conclusion, Result, Summary
_MUFFIN_FINAL = re.compile(r'^[ \t]*(?:#{1,6}[ \t]*|\*\*)[ \t]*(?:final[ \t]+)?(?:answer|conclusion|result|summary)\b', re.I | re.M)


def muffin_sentence(text: str, start: int, end: int) -> str:
    """The sentence of the line around text[start:end] (bounded by the line and by '.', '!', '?' or ';' followed by a space)."""
    a = text.rfind('\n', 0, start) + 1
    b = text.find('\n', end)
    b = len(text) if b < 0 else b
    for m in re.finditer(r'[.!?;]\s', text[a:start]):
        a2 = a + m.end()
        if a2 <= start:
            a = a2
    m = re.search(r'[.!?;](?:\s|$)', text[end:b])
    return text[a:end + m.end() if m else b]


def muffin_final_count(text: str) -> int | None:
    """The answer's final stated count of combinations: the last count stated in the final section (after the last Answer /
    Conclusion heading; the whole text without one), counts named as combinations, ways, options or "the answer is N" taken
    before counts named as solutions or pairs, and counts in a counterfactual sentence ("without the limit there would be 4
    combinations") left out. None: no count stated."""
    heads = list(_MUFFIN_FINAL.finditer(text))
    scopes = [text[heads[-1].start():]] if heads else []
    scopes.append(text)
    for scope in scopes:
        primary, secondary = [], []
        for m in _MUFFIN_COUNT.finditer(scope):
            if re.search(r'\bwould\b', muffin_sentence(scope, m.start(), m.end()), re.I):
                continue
            n = m.group(1).lower()
            (secondary if m.group(2).lower().startswith(('solution', 'pair')) else primary).append((m.start(), n))
        for m in _MUFFIN_ANSWER.finditer(scope):
            if not re.search(r'\bwould\b', muffin_sentence(scope, m.start(), m.end()), re.I):
                primary.append((m.start(), (m.group(1) or m.group(2)).lower()))
        for found in (primary, secondary):
            if found:
                n = max(found)[1]
                return MUFFIN_WORDS[n] if n in MUFFIN_WORDS else int(n)
    return None


def muffin_valid_pairs(text: str) -> list[tuple[int, int]]:
    """Which of the three valid pairs (boxes of 4, boxes of 6) the answer names, in any of its notations: "7 boxes of 4 + 3
    boxes of 6" (either order, "box", "6-packs"), "(7, 3)", "7(4) + 3(6)", "4(7) + 6(3)", "7 x 4 + 3 x 6", "a = 7, b = 3", or a
    table row under headers naming both box sizes. Where a pair is named does not matter: the final count judges the rest."""
    t = text.replace('\\cdot', '×').replace('\\times', '×').replace('‑', '-').replace('‐', '-')
    t = re.sub(r'[$*`\\]', '', t)
    four = r'(?:(?:box(?:es)?|packs?|cartons?)\s+of\s+4|4-?\s?(?:box(?:es)?|packs?))\b'
    six = r'(?:(?:box(?:es)?|packs?|cartons?)\s+of\s+6|6-?\s?(?:box(?:es)?|packs?))\b'
    times = r'\s*(?:\(\s*{n}\s*\)|[×x·*]\s*{n})'
    found = set()
    for a, b in MUFFIN_VALID:
        pats = (rf'(?<!\d){a}\s*{four}[^\n]{{0,40}}?(?<!\d){b}\s*{six}',            # 7 boxes of 4 + 3 boxes of 6
                rf'(?<!\d){b}\s*{six}[^\n]{{0,40}}?(?<!\d){a}\s*{four}',            # 3 boxes of 6 and 7 boxes of 4
                rf'\(\s*{a}\s*,\s*{b}\s*\)',                                         # (7, 3)
                rf'(?<![\d.]){a}' + times.format(n=4) + rf'\s*\+\s*{b}' + times.format(n=6) + r'(?![\d.])',   # 7(4) + 3(6)
                r'(?<![\d.])4' + times.format(n=a) + r'\s*\+\s*6' + times.format(n=b) + r'(?![\d.])',          # 4(7) + 6(3)
                rf'\b[a-z]\s*=\s*{a}\s*(?:,|;|and)?\s*[a-z]\s*=\s*{b}(?!\d)')                           # a = 7, b = 3
        if any(re.search(q, t, re.I) for q in pats):
            found.add((a, b))
    lines = t.splitlines()
    for i, line in enumerate(lines[:-1]):   # (tables whose header names both box sizes: one pair a row)
        if not line.lstrip().startswith('|') or not re.match(r'^\s*\|?\s*:?-{3,}', lines[i + 1]):
            continue
        head = [c.strip().lower() for c in line.strip().strip('|').split('|')]
        f = [k for k, c in enumerate(head) if re.search(four, c)]
        x = [k for k, c in enumerate(head) if re.search(six, c)]
        if len(f) != 1 or len(x) != 1:
            continue
        for row in lines[i + 2:]:
            if not row.lstrip().startswith('|'):
                break
            cells = [c.strip() for c in row.strip().strip('|').split('|')]
            if len(cells) > max(f[0], x[0]) and cells[f[0]].isdigit() and cells[x[0]].isdigit():
                pair = (int(cells[f[0]]), int(cells[x[0]]))
                if pair in MUFFIN_VALID:
                    found.add(pair)
    return sorted(found)


def check_muffins(text: str, finish: str | None = 'stop') -> dict:
    """usability_checks.check_muffins_v2: verdict 'right' | 'wrong' | 'read' | 'cut off'. 'wrong' only when the final section
    states a count other than 3 and never 3; anything else it cannot read goes to the reader. Kept for the evaluators that
    read them: final_count, states_three, pairs_found, expected, correct (= verdict 'right')."""
    v = usability_checks.check_muffins_v2(text, finish)
    expected = [list(p) for p in MUFFIN_VALID]
    return dict(verdict=v['verdict'], why=v['why'], final_count=v['final_count'], states_three=v['final_count'] == 3,
                pairs_found=v['pairs'], expected=expected, correct=v['verdict'] == 'right')


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
    result: dict = dict(arm=arm, seed=seed, created_utc=utc(), tune=tune, delivery=delivery_file, delivery_exit=delivery_exit, delivery_stderr=delivery_stderr, measurement_protocol=dict(start_emitted=0), requests=[], tasks=[])
    with (a.run_dir / f'tasks-{name}.sse.jsonl').open('x') as raw:
        for task in protocol['tasks']:
            req = delivery.payload([{'role': 'user', 'content': task['prompt']}], tune, task['max_tokens'], seed)
            item = delivery.invoke(a.base_url, task['id'], req, a.timeout, raw, result, a.engine_pid)
            final = (item.get('engine_stats') or {}).get('final') or {}
            row = dict(id=task['id'], complete=item.get('complete', False), finish=item.get('finish_reason'), emitted=final.get('emitted'), generation_ms=final.get('elapsed_ms'),
                       draft_acc=final.get('draft_acc'), plan_rounds=final.get('plan_rounds'),   # (drafting: accepted drafts and planned rounds)
                       settings=settings_seen(item), content=item.get('content', ''), reasoning=item.get('reasoning_content', ''))
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
