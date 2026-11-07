#!/usr/bin/env python3
"""One short, explicitly chosen delivery arm against an already resident server.

Run selected and ordinary arms separately in the desired order on a quiet server.
Each run records four workflows (code, reasoning, streamed tool round trip, code
return), then one disconnect/follow-up lifecycle check. No server is started.
Raw SSE and metrics are kept with a new result outside the source repository.
"""
from __future__ import annotations
import argparse
import ast
import base64
import datetime as dt
import json
import pathlib
import re
import subprocess
import sys
import time
import urllib.request

import native_screen as screen
import native_reasoning_check as reasoning
import native_capture as capture

ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA = "\n".join(f"batch-{i:02d}: " + " ".join(map(str, [i % 3, i % 3, (i+1) % 3, (i+1) % 3, i % 3]))
                 for i in range(1, 31))
CODE = ("Write a pure Python function count_runs(items) returning the number of contiguous runs of equal values. "
        "The empty sequence has zero runs; a nonempty sequence starts one run, and each change starts another. "
        "Compare values with equality, don't mutate the input, and return only a Python code block. "
        "The following independent small sensor batches illustrate adjacent equal readings; each line is a separate "
        "batch and is not part of the function's input format:\n\n" + DATA +
        "\n\nExamples: count_runs([]) is 0; count_runs([1,1,2,2,1]) is 3.")
REASON = reasoning.PROMPT
TOOL_PROMPT = ("Read the file notes.txt using the declared read_file tool. Report the exact version it contains; "
               "do not guess it before the tool result arrives.")
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a named local text file",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]
TOOL_FIXTURE_CONTENT = "notes.txt\nRelease version: 7.4.2\n"


def check_code(text: str) -> dict:
    match = re.search(r'```(?:python|py)?\s*\n(.*?)```', text, re.S | re.I)
    code = match.group(1) if match else text.strip()
    if len(code) > 16000:
        return {"status": "rejected", "reason": "code exceeds 16000 characters"}
    try:
        tree = ast.parse(code)
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef) or tree.body[0].name != 'count_runs':
            raise ValueError('expected one count_runs function')
        banned = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef, ast.Attribute)
        for node in ast.walk(tree):
            if isinstance(node, banned) or (isinstance(node, ast.Name) and node.id.startswith('__')):
                raise ValueError('unsupported code construct')
        runner = r'''import json,sys
s=json.loads(sys.stdin.read()); scope={'__builtins__':{'len':len,'range':range,'enumerate':enumerate,'zip':zip,'int':int,'bool':bool,'list':list,'sum':sum,'iter':iter,'next':next,'StopIteration':StopIteration}}
exec(compile(s,'<generated>','exec'),scope)
f=scope['count_runs']; checks=[([],0),([1],1),([1,1,2,2,1],3),(['a','a','b'],2),([None,None,1,None],3)]
r=[]
for x,want in checks:
 old=x[:]
 try:
  got=f(x);r.append({'input':x,'expected':want,'actual':got,'pass':type(got) is int and got==want and x==old})
 except Exception as e:r.append({'input':x,'expected':want,'error':type(e).__name__+': '+str(e),'pass':False})
print(json.dumps(r))'''
        import resource
        def limit():
            resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
            resource.setrlimit(resource.RLIMIT_AS, (256 << 20, 256 << 20))
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
        proc = subprocess.run([sys.executable, '-I', '-c', runner], input=json.dumps(code), text=True,
                              capture_output=True, timeout=4, cwd='/tmp', preexec_fn=limit)
        if proc.returncode:
            return {'status': 'execution_error', 'stderr': proc.stderr[-2000:]}
        checks = json.loads(proc.stdout)
        return {'status': 'pass' if all(c['pass'] for c in checks) else 'fail', 'checks': checks}
    except (SyntaxError, ValueError, OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        return {'status': 'rejected_or_error', 'reason': f'{type(exc).__name__}: {exc}'}


def payload(messages, tune, tokens, seed, tools=None, reasoning_effort='high'):
    req = {'model': 'MiniMax-M3.1-Flash-native-wsl', 'messages': messages, 'stream': True,
           'return_progress': True, 'max_tokens': tokens, 'temperature': 1, 'top_p': .95,
           'top_k': 0, 'min_p': 0, 'seed': seed, 'reasoning_effort': reasoning_effort, 'loop_guard': False, 'strata_tune': tune}
    if tools: req['tools'] = tools
    return req


def invoke(base, name, req, timeout, raw, result, pid, capture_context=None):
    # The screen collector preserves all lines, timestamps, engine history, and
    # completed generation intervals. Its fixture-specific syntax check is
    # discarded here in favor of the checks specific to this workflow.
    screen.run_case(base, {'id': name, 'checks': []}, req, timeout, raw, result, pid,
                    result['measurement_protocol']['start_emitted'], capture_context=capture_context)
    item = result['requests'][-1]
    item.pop('syntax_check', None)
    return item


def streamed_calls(item):
    calls = {}
    for event in item.get('events', []):
        for choice in event['data'].get('choices') or []:
            for delta in (choice.get('delta') or {}).get('tool_calls') or []:
                i = delta.get('index', 0)
                entry = calls.setdefault(i, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
                if delta.get('id'): entry['id'] += delta['id']
                fn = delta.get('function') or {}
                for key in ('name', 'arguments'):
                    if isinstance(fn.get(key), str): entry['function'][key] += fn[key]
    return [calls[k] for k in sorted(calls)]


def check_reasoning(item):
    return reasoning.check(item, {})


def read_fixture_call(call, fixture_dir):
    """Execute the one declared read against our newly created, isolated fixture."""
    if call.get('function', {}).get('name') != 'read_file':
        raise ValueError('unexpected tool')
    arguments = json.loads(call['function']['arguments'])
    if arguments != {'path': 'notes.txt'}:
        raise ValueError('tool path is outside the declared fixture')
    path = fixture_dir / 'notes.txt'
    if path.is_symlink() or not path.is_file():
        raise ValueError('declared fixture is missing or is a symlink')
    return path.read_text(encoding='utf-8')


def cancel_then_follow(base, tune, seed, timeout, raw, result, pid, capture_context=None):
    case = 'lifecycle-disconnect'
    req = payload([{'role': 'user', 'content': 'Write a long numbered list of independent ideas about maintaining a small garden.'}],
                  tune, 256, seed)
    item = {'case': case, 'request': req, 'metrics_before': screen.metrics(base, min(timeout,30)),
            'process_io_before': screen.process_io(pid), 'disconnect_after_text': False}
    result['lifecycle'] = item
    capture_row = capture.begin(capture_context, {'id':case}, req) if capture_context else None
    if capture_row:
        item['capture_request_id'] = capture_row['request_id']
        item['capture_note'] = ('The API stream is intentionally disconnected; validate the native/footer and '
                                'frontend drained records separately from API completion.')
    request = urllib.request.Request(base + '/v1/chat/completions', data=json.dumps(req).encode(),
                                    headers={'Content-Type': 'application/json', 'Accept': 'text/event-stream'}, method='POST')
    start = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            item['http_status'] = response.status
            for line in response:
                elapsed = (time.monotonic()-start)*1000
                raw.write(json.dumps({'case':case,'elapsed_ms':elapsed,'line_base64':base64.b64encode(line).decode()})+'\n');raw.flush()
                if line.startswith(b'data:') and line[5:].strip() not in (b'', b'[DONE]'):
                    data = json.loads(line[5:].strip())
                    if data.get('error'): raise RuntimeError(str(data['error']))
                    if any((c.get('delta') or {}).get('content') or
                           (c.get('delta') or {}).get('reasoning_content') for c in data.get('choices') or []):
                        item['disconnect_after_text'] = True
                        item['disconnect_elapsed_ms'] = elapsed
                        break
        follow = payload([{'role':'user','content':'Reply with exactly the word READY.'}],tune,64,seed)
        after = invoke(base,'lifecycle-followup',follow,timeout,raw,result,pid,capture_context)
        item['followup_complete'] = after.get('complete',False)
        item['followup_content'] = after.get('content','')
        item['followup_ok'] = after.get('complete',False) and 'READY' in after.get('content','').upper()
        # A finished follow-up proves continued service; a disconnect request's
        # own history finish must still be checked separately, not inferred.
        recent = (after.get('metrics_after') or {}).get('requests') or []
        since = time.time()-(time.monotonic()-start)-1
        rows = [r for r in recent if isinstance(r.get('time'),(int,float)) and r['time']>=since]
        item['recent_request_finishes'] = [{'time':r.get('time'),'finish':r.get('finish')} for r in rows]
        item['cancel_observation'] = 'observed' if any(r.get('finish') in ('cancel','disconnect') for r in rows) else 'unverified'
    except Exception as exc:
        item['error'] = f'{type(exc).__name__}: {exc}'
    finally:
        # The completed follow-up also gives the cancelled request's normal
        # retirement path time to write its buffered artifacts. No extra wait,
        # token record or successful API completion is fabricated for it.
        if capture_row:
            capture.finish(capture_context, capture_row, item)
    item['wall_ms'] = (time.monotonic()-start)*1000


def aggregate(requests):
    eligible=[]; missing=[]
    for item in requests:
        stats=item.get('engine_stats') or {}
        interval=stats.get('generation_interval')
        if (stats.get('join') == 'one_new_sequence_and_matching_completion_count' and
                stats.get('interval_status') == 'measured' and interval):
            eligible.append((item['case'],interval))
        else: missing.append(item['case'])
    tokens=sum(s['tokens'] for _,s in eligible);ms=sum(s['interval_ms'] for _,s in eligible)
    return {'scope':'eligible main-workflow requests; first completed boundary at or after requested start through final boundary',
            'eligible':len(eligible),'missing':missing,'pooled_tokens':tokens,'pooled_ms':ms,
            'pooled_tokens_per_second':1000*tokens/ms if ms else None,
            'worst_tokens_per_second':min((s['tokens_per_second'] for _,s in eligible),default=None),
            'individual':{name:s for name,s in eligible}}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base-url',default='http://127.0.0.1:8097')
    ap.add_argument('--arm',choices=('ordinary','selected'),required=True)
    ap.add_argument('--routing-prefetch', type=int, choices=(0, 1), default=None)
    ap.add_argument('--routing-dampen', type=int, choices=(0, 1), default=None)
    ap.add_argument('--routing',choices=('off','substitute','prior'),default='substitute')
    ap.add_argument('--ratio',type=float,default=.8);ap.add_argument('--prior-bonus',type=float,default=.05)
    ap.add_argument('--spec-policy',choices=('sampled','fast_hsd','lenience'),default='lenience')
    ap.add_argument('--rho',type=float,default=.2);ap.add_argument('--ell',type=float,default=.4);ap.add_argument('--width',type=int,default=7)
    ap.add_argument('--draft-confidence',type=float,default=.3);ap.add_argument('--verify-segment',type=int,default=0)
    ap.add_argument('--stage-overlap',type=int,choices=(0,1),default=None,help='explicit serial (0) or overlapped (1) stage path')
    ap.add_argument('--tokens',type=int,default=512);ap.add_argument('--reasoning-tokens',type=int,default=6144);ap.add_argument('--seed',type=int,default=20261101)
    ap.add_argument('--start-emitted',type=int,default=32,help='first completed round at or after this count')
    ap.add_argument('--tune-json',type=json.loads,default={},help='extra strata_tune keys, merged last')
    ap.add_argument('--engine-pid',type=int);ap.add_argument('--note',default='');ap.add_argument('--timeout',type=float,default=900)
    ap.add_argument('--output',type=pathlib.Path,required=True)
    capture.args(ap)
    a=ap.parse_args()
    if a.start_emitted < 0: ap.error('start-emitted must be nonnegative')
    if not 1<=a.tokens<=1024 or not 0<=a.ratio<=1 or not 0<=a.prior_bonus<=1 or not 0<a.rho<=1 or not 0<a.ell<=1 or a.reasoning_tokens < 1 or not 1<=a.width<=7 or not 0<=a.draft_confidence<=1 or not 0<=a.verify_segment<=7 or a.timeout<=0:
        ap.error('invalid token, policy, or timeout setting')
    if a.engine_pid is not None and a.engine_pid<=0: ap.error('engine PID must be positive')
    if a.stage_overlap == 1 and (a.arm!='selected' or a.verify_segment!=4):
        ap.error('stage overlap requires the selected arm and verify-segment 4')
    output=a.output.expanduser().resolve()
    if output.is_relative_to(ROOT) or not output.parent.is_dir(): ap.error('output must be outside source repository in an existing directory')
    rawpath=output.with_name(output.name+'.sse.jsonl');base=a.base_url.rstrip('/')
    fixture_dir=output.with_name(output.name+'.tool-fixture')
    tune={'fresh_context':1,'routing':'off' if a.arm=='ordinary' else a.routing,'routing_ratio':a.ratio,
          'spec':1 if a.arm=='ordinary' else a.width+1,
          'spec_policy':'existing' if a.arm=='ordinary' else a.spec_policy,
          'verify_segment':0 if a.arm=='ordinary' else a.verify_segment}
    if a.routing_prefetch is not None: tune['routing_prefetch'] = a.routing_prefetch
    if a.routing_dampen is not None: tune['routing_dampen'] = a.routing_dampen
    if a.arm=='selected':
        tune['spec_min_p']=a.draft_confidence
        if a.spec_policy=='fast_hsd':tune['spec_rho']=a.rho
        if a.spec_policy=='lenience':tune['spec_ell']=a.ell
        if a.routing=='prior':tune['routing_prior_bonus']=a.prior_bonus
        if a.stage_overlap is not None:tune['stage_overlap']=a.stage_overlap
    tune.update(a.tune_json)
    result={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'arm':a.arm,'tune':tune,
            'conditioning_note':a.note,
            'conditioning_precondition':'Declare neutral setup generation, starting placement and prior request history; do not train placement on scored prompts.',
            'measurement_protocol':{'start_emitted':a.start_emitted,'interval':'first completed round at or after requested count through final completed boundary'},
            'base_url':base,'raw_sse_jsonl':str(rawpath),'requests':[],'workflows':[]}
    capture_context=capture.init(a,output,a.engine_pid)
    capture_context['manifest'].update({'receipt':str(output),'raw_sse':str(rawpath),
        'kind':'usability_and_lifecycle','rankable':False,'requested_arm':a.arm,
        'measurement_protocol':result['measurement_protocol']})
    capture.save(capture_context)
    result['capture_manifest']=str(capture_context['directory']/'manifest.json')
    with output.open('x',encoding='utf-8') as saved:
      try:
       fixture_dir.mkdir()
       with (fixture_dir / 'notes.txt').open('x', encoding='utf-8') as fixture:
           fixture.write(TOOL_FIXTURE_CONTENT)
       result['tool_fixture'] = str(fixture_dir / 'notes.txt')
       with rawpath.open('x',encoding='utf-8') as raw:
        def write():
            saved.seek(0);saved.write(json.dumps(result,indent=2,ensure_ascii=False)+'\n');saved.truncate();saved.flush()
        for ident,prompt in [('code-initial',CODE),('reasoning',REASON)]:
            item=invoke(base,ident,payload([{'role':'user','content':prompt}],tune,a.reasoning_tokens if ident=='reasoning' else a.tokens,a.seed,
                                         reasoning_effort='high'),a.timeout,raw,result,a.engine_pid,capture_context)
            if ident=='code-initial': item['quality']=check_code(item.get('content',''))
            else:item['quality']=check_reasoning(item)
            result['workflows'].append({'case':ident,'complete':item.get('complete',False),'quality':item['quality']});write()
            if not item.get('complete') or item.get('error'):return 1
        tool_started=time.monotonic()
        first=invoke(base,'tool-call',payload([{'role':'user','content':TOOL_PROMPT}],tune,a.tokens,a.seed,TOOLS),a.timeout,raw,result,a.engine_pid,capture_context)
        calls=streamed_calls(first); tool={'case':'tool-roundtrip','calls':calls,'initial_wall_ms':first.get('api_wall_ms'),
                                         'raw_repaired_failed':((first.get('metrics_after') or {}).get('requests') or [{}])[0].get('tool_parse_outcomes'),
                                         'retries':0}
        result['workflows'].append(tool)
        try:
            require = len(calls)==1 and calls[0]['id'] and calls[0]['function']['name']=='read_file' and json.loads(calls[0]['function']['arguments'])=={'path':'notes.txt'}
        except (ValueError,TypeError):require=False
        tool['declared_call_valid']=bool(require)
        if require and first.get('complete'):
            tool_result = read_fixture_call(calls[0], fixture_dir)
            tool['executed_fixture_read'] = str(fixture_dir / 'notes.txt')
            messages=[{'role':'user','content':TOOL_PROMPT},{'role':'assistant','content':first.get('content') or None,'tool_calls':calls},
                      {'role':'tool','tool_call_id':calls[0]['id'],'content':tool_result},
                      {'role':'user','content':'State the exact version in the file and explain briefly how you obtained it.'}]
            follow=invoke(base,'tool-followup',payload(messages,tune,a.tokens,a.seed,TOOLS),a.timeout,raw,result,a.engine_pid,capture_context)
            tool['followup_complete']=follow.get('complete',False);tool['version_present']='7.4.2' in follow.get('content','')
            tool['api_wall_sum_ms']=(first.get('api_wall_ms') or 0)+(follow.get('api_wall_ms') or 0)
        else:tool['error']='no valid declared read_file call; no tool was executed'
        tool['workflow_wall_ms']=(time.monotonic()-tool_started)*1000
        tool['complete']=bool(first.get('complete') and tool.get('followup_complete'))
        tool['quality_pass']=bool(tool['declared_call_valid'] and tool.get('version_present'))
        counts=tool['raw_repaired_failed'] or {}
        tool['parse_outcome']=('failed' if counts.get('failed') else 'repaired' if counts.get('repaired') else
                               'raw' if counts.get('raw') else 'unreported')
        write()
        last=invoke(base,'code-return',payload([{'role':'user','content':CODE}],tune,a.tokens,a.seed),a.timeout,raw,result,a.engine_pid,capture_context)
        last['quality']=check_code(last.get('content',''));result['workflows'].append({'case':'code-return','complete':last.get('complete',False),'quality':last['quality']})
        result['main_generation_interval']=aggregate(result['requests']);write()
        cancel_then_follow(base,tune,a.seed,a.timeout,raw,result,a.engine_pid,capture_context);write()
      finally:
       saved.seek(0);saved.write(json.dumps(result,indent=2,ensure_ascii=False)+'\n');saved.truncate();saved.flush()
    print(output)
    lifecycle=result.get('lifecycle',{})
    quality=(len(result['workflows'])==4 and all(w.get('complete',False) for w in result['workflows']) and
             result['workflows'][0]['quality']['status']=='pass' and
             result['workflows'][2]['quality_pass'] and result['workflows'][3]['quality']['status']=='pass')
    if not (quality and lifecycle.get('disconnect_after_text') and lifecycle.get('followup_ok') and lifecycle.get('cancel_observation')=='observed'):
        return 1
    print('Reasoning requires review with native_reasoning_check.py', file=sys.stderr)
    return 2

if __name__=='__main__':sys.exit(main())
