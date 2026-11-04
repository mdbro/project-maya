#!/usr/bin/env python3
"""Collect generation-only MiniMax demand or apply a frozen profile in place.

All outputs go to a newly created directory outside the source tree. Commands
never start or reload an engine. Collection prompts are separate from the scored
screen/delivery fixtures. Applying a profile is separate from timing inference.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import pathlib
import struct
import sys
import urllib.error
import urllib.request

import native_screen as screen
import native_capture as capture

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROFILE_BYTES = 30720
PROMPTS = [
    ('calibration-code', 'Write a Python function that returns a new list containing only the first occurrence of each integer in its input. Include a short example.'),
    ('calibration-reason', 'A workshop has 12 bolts. It uses 3, receives 5 more, then divides all bolts equally among 7 trays. Explain the count on each tray.'),
    ('calibration-tool', 'Use the declared lookup_status tool for sensor alpha. Do not guess the result.'),
]
TOOLS = [{'type':'function','function':{'name':'lookup_status','description':'Look up a sensor status',
          'parameters':{'type':'object','properties':{'sensor':{'type':'string'}},'required':['sensor']}}}]
FOLLOWUP = 'Reply briefly with the number of sides on a triangle.'


def post(base, endpoint, body, events, timeout):
    event = {'endpoint': endpoint, 'request': body}
    events.append(event)
    request = urllib.request.Request(base + endpoint, data=json.dumps(body).encode(),
                                     headers={'Content-Type':'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            event['status'] = response.status
            event['response'] = json.load(response)
    except urllib.error.HTTPError as exc:
        event['status'] = exc.code
        event['response_raw'] = exc.read().decode('utf-8', errors='replace')
        raise RuntimeError(f'{endpoint}: HTTP {exc.code}: {event["response_raw"]}') from exc
    except Exception as exc:
        event['error'] = f'{type(exc).__name__}: {exc}'
        raise
    return event['response']


def counts_from_bytes(data):
    if len(data) != PROFILE_BYTES:
        raise ValueError(f'profile must have {PROFILE_BYTES} bytes')
    return struct.unpack('<7680I', data)


def snapshot(base, events, timeout):
    reply = post(base, '/v1/minimax/profile', {'action':'snapshot'}, events, timeout)
    try:
        data = bytes.fromhex(reply['profile_hex'])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError('invalid profile snapshot') from exc
    counts_from_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    if reply.get('profile_sha256') != digest:
        raise ValueError('snapshot SHA-256 mismatch')
    return reply, data


def inspect(base, events, timeout):
    report = post(base, '/v1/minimax/placement', {'action':'inspect'}, events, timeout)
    if not report.get('ownership_complete') or not isinstance(report.get('engine_pid'), int):
        raise ValueError('inspection does not establish resident process and complete ownership')
    return report


def same_process(first, second):
    if first.get('engine_pid') != second.get('engine_pid'):
        raise ValueError('engine process changed during calibration')


def learned_state_signature(report):
    """Fingerprint logical checkpoint state, excluding timestamps and addresses."""
    if report.get('scope') != 'in-process only' or report.get('staging') != 'cleared at joined boundary':
        raise ValueError('unsupported learned-state checkpoint')
    placement = report.get('placement', {})
    if not placement.get('ownership_complete'):
        raise ValueError('checkpoint ownership is not complete')
    fields = ('geometry', 'arithmetic', 'primary_adaptive', 'policy', 'lane', 'scores', 'adaptive_main')
    if any(k not in report for k in fields):
        raise ValueError('checkpoint learned state is incomplete')
    state = {k: report[k] for k in fields}
    state['batch_staging_calibration'] = report.get('batch_staging_calibration')
    state['admission_credits'] = report.get('admission_credits')
    for name in ('primary_ids', 'helper_ids'):
        if not isinstance(placement.get(name), list):
            raise ValueError('checkpoint roster is missing')
        state[name] = sorted(row[:2] for row in placement[name])
    data = json.dumps(state, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def selection(profile, before, mode):
    """Recompute the placement policy from profile and complete engine geometry."""
    counts = counts_from_bytes(profile)
    layers = before.get('layer_devices')
    head = before.get('helper_eligible_cuda_device')
    moe = before.get('layer_moe')
    width = before.get('experts_per_layer')
    cap = before.get('resident_cap')
    if (type(width) is not int or width <= 0 or not isinstance(layers, list) or
            len(layers) * width != len(counts) or not isinstance(moe, list) or
            len(moe) != len(layers) or any(type(m) is not bool for m in moe)):
        raise ValueError('complete layer_devices/layer_moe/expert geometry missing')
    if type(head) is not int or type(cap) is not int:
        raise ValueError('head device or primary per-layer capacity missing')
    cap = width if cap < 0 else min(cap, width)
    if not 0 <= cap <= width or any(x is not None and type(x) is not int for x in layers):
        raise ValueError('invalid placement geometry')
    if cap != width or {d for d,m in zip(layers,moe) if m} != {head}:
        raise ValueError('global placement requires one primary device and no per-layer cap')
    primary_rows = before.get('primary_ids')
    helper_rows = before.get('helper_ids')
    if not isinstance(primary_rows, list) or not isinstance(helper_rows, list):
        raise ValueError('placement identities missing')
    old_primary = {(r[0],r[1]) for r in primary_rows}
    if len(old_primary) != len(primary_rows): raise ValueError('duplicate primary identities')
    if any(len(r) != 4 or not 0 <= r[0] < len(layers) or not moe[r[0]] or
           not 0 <= r[1] < width or r[2] != layers[r[0]] for r in primary_rows):
        raise ValueError('primary identities disagree with layer geometry')
    if mode not in ('all','helper-first','helper'): raise ValueError('invalid placement mode')
    ranked = [(li,e) for e in range(width) for li,d in enumerate(layers) if d == head and moe[li]]
    ranked.sort(key=lambda x: -counts[x[0]*width+x[1]])
    if len(ranked)<len(helper_rows): raise ValueError('insufficient helper candidates')
    helper_first = set(ranked[:len(helper_rows)])
    if mode in ('all','helper-first','helper'):
        desired = set()
        for device in sorted({x for x in layers if x is not None}):
            slots = sum(1 for r in primary_rows if r[2] == device)
            candidates = [(li,e) for li,d in enumerate(layers) if d == device and moe[li] for e in range(width)
                          if (li,e) not in helper_first]
            candidates.sort(key=lambda x: -counts[x[0]*width+x[1]])
            per = {}
            picked = 0
            for li,e in candidates:
                if picked == slots: break
                if per.get(li,0) >= cap: continue
                desired.add((li,e)); per[li] = per.get(li,0)+1; picked += 1
            if picked != slots: raise ValueError('primary capacity cannot be filled from reported geometry')
    if mode == 'helper' and desired != old_primary:
        raise ValueError('helper-only refresh would violate global helper-first placement; use all')
    return desired, helper_first


def verify_placement(before, after, profile, mode):
    same_process(before,after)
    if not after.get('ownership_complete'): raise ValueError('ownership is incomplete after refresh')
    for key in ('target_object','draft_object','host_address','host_bytes','host_records','helper_pool'):
        if key not in before or before[key] != after.get(key):
            raise ValueError(f'retained allocation changed or missing: {key}')
    if sorted(before.get('allocations', [])) != sorted(after.get('allocations', [])) or 'allocations' not in before:
        raise ValueError('model allocation identities changed or missing')
    rotation = False
    if 'staging_frames' in before or 'staging_frames' in after:
        def frames(report):
            staging = report.get('staging_frames')
            if (not isinstance(staging, list) or any(not isinstance(r, list) or len(r) != 5 or
                    any(type(v) is not int for v in r) or r[0] < 0 or r[1] < 0 or r[2] <= 0 or
                    r[4] not in (0, 1) for r in staging)):
                raise ValueError('invalid staging frame inventory')
            slots = [(r[0], r[1]) for r in staging]
            physical = [(r[2], r[3]) for r in report['primary_ids']] + [(r[0], r[2]) for r in staging]
            if len(set(slots)) != len(slots) or len(set(physical)) != len(physical):
                raise ValueError('duplicate primary/staging physical frame')
            return sorted(slots), sorted(physical)
        if frames(before) != frames(after):
            raise ValueError('primary/staging physical frames changed')
        if before.get('adaptive_host_scratch_bytes') != after.get('adaptive_host_scratch_bytes'):
            raise ValueError('adaptive HOST scratch changed')
        rotation = sorted(r[3] for r in before['primary_ids']) != sorted(r[3] for r in after['primary_ids'])
        checked_slots = (('helper_ids',2),)
    else:
        checked_slots = (('primary_ids',3),('helper_ids',2))
    for name, index in checked_slots:
        if sorted(r[index] for r in before[name]) != sorted(r[index] for r in after[name]):
            raise ValueError(f'{name} physical slots changed')
    desired_primary, desired_helper = selection(profile,before,mode)
    actual_primary = {(r[0],r[1]) for r in after['primary_ids']}
    actual_helper = {(r[0],r[1]) for r in after['helper_ids']}
    if len(actual_primary) != len(after['primary_ids']) or len(actual_helper) != len(after['helper_ids']):
        raise ValueError('duplicate actual placement identities')
    if actual_primary != desired_primary or actual_helper != desired_helper:
        raise ValueError('actual placement differs from deterministic profile selection')
    if actual_primary & actual_helper: raise ValueError('primary and helper placements overlap')
    layers = before['layer_devices']
    if any(r[2] != layers[r[0]] for r in after['primary_ids']):
        raise ValueError('primary identity assigned to the wrong device')
    if mode == 'helper' and {tuple(r[:2]) for r in before['helper_ids']} == actual_helper:
        raise ValueError('helper refresh did not change identities; A-to-B transition not demonstrated')
    return {'primary_count':len(actual_primary),'helper_count':len(actual_helper),
            'primary_ids_match':True,'helper_ids_match':True,'allocations_retained':True,
            'ownership_complete':True,'primary_staging_frame_rotation':rotation}


def payload(prompt, tune, tokens, seed, tool=False, reasoning=False):
    request={'model':'MiniMax-M3.1-Flash-native-wsl','messages':[{'role':'user','content':prompt}],
             'stream':True,'return_progress':True,'max_tokens':tokens,'temperature':1,'top_p':0.95,
             'top_k':0,'min_p':0,'seed':seed,'reasoning_effort':'high',
             'strata_tune':tune}
    if tool: request['tools']=TOOLS
    return request


def tool_calls_observed(item):
    names={}
    for event in item.get('events', []):
        for choice in (event.get('data') or {}).get('choices') or []:
            for call in (choice.get('delta') or {}).get('tool_calls') or []:
                index=call.get('index',0)
                names[index]=names.get(index,'')+(call.get('function') or {}).get('name','')
    return 'lookup_status' in names.values()


def run_generation(base, name, request, timeout, raw, result, capture_context=None, engine_pid=None):
    kwargs = {'capture_context':capture_context, 'engine_pid':engine_pid, 'start_emitted':0} if capture_context else {}
    screen.run_case(base, {'id':name,'checks':[]}, request, timeout, raw, result, **kwargs)
    item=result['requests'][-1]
    item.pop('syntax_check',None)
    if not item.get('complete') or item.get('error'):
        raise RuntimeError(f'{name}: generation did not complete: {item.get("error", "missing DONE")}')
    if (item.get('engine_stats') or {}).get('join') != 'one_new_sequence_and_matching_completion_count':
        raise RuntimeError(f'{name}: engine/API request attribution was not verified')
    return item


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('collect','helper','all','helper-first','historical-helper-first','inspect','snapshot','restore','backing-on','backing-off'))
    ap.add_argument('--base-url',default='http://127.0.0.1:8097')
    ap.add_argument('--output-dir',type=pathlib.Path,required=True)
    ap.add_argument('--profile',type=pathlib.Path,help='frozen generation profile, required for helper/all')
    ap.add_argument('--source-profile',type=pathlib.Path,help='declared profile used to load the resident process (required for collect)')
    ap.add_argument('--protocol',type=pathlib.Path,help='explicit independent history: requests, tune, max_tokens_each, seed; no placement changes')
    ap.add_argument('--historical-fixed-primary',action='store_true',help='explicit historical conditioning with fixed primary and live helper; requires a pinned binary')
    ap.add_argument('--require-binary-sha256',help='reject a different resident executable before any model control')
    ap.add_argument('--require-engine-pid',type=int,help='resident PID from the source-bound launch, required with a binary pin')
    ap.add_argument('--tokens',type=int,default=96)
    ap.add_argument('--seed',type=int,default=20261102)
    ap.add_argument('--timeout',type=float,default=900)
    ap.add_argument('--routing-prefetch', type=int, choices=(0, 1), default=None)
    ap.add_argument('--routing-dampen', type=int, choices=(0, 1), default=None)
    ap.add_argument('--routing',choices=('off','prior','substitute'),default='substitute')
    ap.add_argument('--ratio',type=float,default=.8)
    ap.add_argument('--verify-segment',type=int,default=0)
    ap.add_argument('--stage-overlap',type=int,choices=(0,1))
    ap.add_argument('--note',default='',help='existing conditioning, request history and selected runtime settings')
    capture.args(ap)
    a=ap.parse_args(argv)
    if not 1<=a.tokens<=256 or a.timeout<=0 or not 0<=a.ratio<=1 or not 0<=a.verify_segment<=7:
        ap.error('invalid token, timeout, routing ratio or verification segment')
    if a.stage_overlap==1 and a.verify_segment!=4: ap.error('stage overlap requires verify-segment 4')
    if (a.mode in ('helper','all','helper-first','historical-helper-first')) != bool(a.profile): ap.error('--profile is required only for placement modes')
    if a.mode != 'collect' and (a.source_profile or a.protocol): ap.error('history inputs require collect')
    if a.mode == 'collect' and not (a.source_profile or a.protocol): ap.error('collect requires --source-profile or an explicit --protocol')
    if a.historical_fixed_primary and (a.mode!='collect' or not a.protocol): ap.error('historical fixed-primary conditioning requires collect and a declared protocol')
    if (a.historical_fixed_primary or a.mode=='historical-helper-first') and not a.require_binary_sha256: ap.error('historical controls require --require-binary-sha256')
    if bool(a.require_binary_sha256)!=bool(a.require_engine_pid): ap.error('binary pin and engine PID must be supplied together')
    output=a.output_dir.expanduser().resolve()
    if output.is_relative_to(ROOT): ap.error('output directory must be outside the source repository')
    output.mkdir(parents=False,exist_ok=False)
    result={'created_utc':dt.datetime.now(dt.timezone.utc).isoformat(),'mode':a.mode,
            'base_url':a.base_url,'note':a.note,'events':[],'requests':[],
            'conditioning_statement':'This calibration is not itself proof of warmed expert caches; observe priming separately.',
            'complete':False}
    result['historical_fixed_primary']=a.historical_fixed_primary
    result_path=output/'result.json'; raw_path=output/'requests.sse.jsonl'
    def save(): result_path.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    tune={'routing':a.routing,'routing_ratio':a.ratio,'spec':8,'spec_policy':'fast_hsd',
          'spec_rho':.2,'spec_min_p':.3,'verify_segment':a.verify_segment}
    if a.routing_prefetch is not None: tune['routing_prefetch'] = a.routing_prefetch
    if a.routing_dampen is not None: tune['routing_dampen'] = a.routing_dampen
    if a.stage_overlap is not None: tune['stage_overlap']=a.stage_overlap
    cases=[{'id':i,'prompt':p,'tools':TOOLS if i=='calibration-tool' else None} for i,p in PROMPTS]
    tokens,seed=a.tokens,a.seed
    if a.protocol:
        protocol_bytes=a.protocol.read_bytes();protocol=json.loads(protocol_bytes)
        cases=protocol['requests'];tune=protocol['tune'];tokens=protocol['max_tokens_each'];seed=protocol['seed']
        if not 1<=tokens<=1024 or not cases or any(not c.get('id') or not c.get('prompt') for c in cases):
            raise ValueError('invalid independent-history protocol')
        expected_primary=0 if a.historical_fixed_primary else 1
        if (tune.get('fresh_context') != 1 or tune.get('primary_adaptive') != expected_primary or
                tune.get('helper_live') not in ((1,) if a.historical_fixed_primary else (0,1))):
            raise ValueError('history must declare fresh KV and its required primary/helper policies')
        result['protocol']={'path':str(a.protocol.resolve()),'sha256':hashlib.sha256(protocol_bytes).hexdigest(),'declared':protocol}
    capture_context=capture.init(a,result_path) if a.capture_control or a.capture_mode else None
    if capture_context: result['capture_manifest']=str(capture_context['directory']/'manifest.json')
    def control(name, fn):
        return capture.operation(capture_context,name,fn) if capture_context else fn()
    result['generation_settings']=tune
    try:
        with raw_path.open('x',encoding='utf-8') as raw:
            base=a.base_url.rstrip('/')
            if a.require_binary_sha256:
                pid=a.require_engine_pid
                observed=capture.sha(pathlib.Path('/proc')/str(pid)/'exe')
                result['required_binary']={'pid':pid,'expected':a.require_binary_sha256,'observed':observed}
                if observed!=a.require_binary_sha256: raise ValueError('resident executable differs from required binary')
            initial=control('placement_inspect_initial',lambda:inspect(base,result['events'],a.timeout));result['initial_inspection']=initial;save()
            if a.require_engine_pid and initial['engine_pid']!=a.require_engine_pid:
                raise ValueError('server controls a different engine PID')
            if capture_context: capture.bind_executable(capture_context,initial['engine_pid'])
            if a.mode=='collect':
                if a.source_profile:
                    source_path=a.source_profile.expanduser().resolve()
                    source=a.source_profile.read_bytes(); counts_from_bytes(source)
                    result['declared_source_profile']={'path':str(source_path),'bytes':len(source),
                        'sha256':hashlib.sha256(source).hexdigest(),
                        'verification':'caller-declared startup input; live startup identity not independently confirmed'}
                else:
                    result['declared_source_profile']=None
                reply=control('profile_reset',lambda:post(base,'/v1/minimax/profile',{'action':'reset'},result['events'],a.timeout))
                same_process(initial,reply)
                snap,data=control('profile_snapshot',lambda:snapshot(base,result['events'],a.timeout));same_process(initial,snap)
                prior=counts_from_bytes(data)
                if any(prior): raise ValueError('generation profile reset did not produce zero counts')
                result['phases']=[]
                for case in cases:
                    ident=case['id'];request_tune={**tune,**case.get('tune',{})}
                    if a.protocol and (request_tune.get('fresh_context')!=1 or request_tune.get('primary_adaptive')!=expected_primary or
                            (a.historical_fixed_primary and request_tune.get('helper_live')!=1)):
                        raise ValueError('history case must retain fresh KV and the declared primary/helper contract')
                    req=payload(case['prompt'],request_tune,tokens,seed)
                    if case.get('tools'): req['tools']=case['tools']
                    req['loop_guard']=False
                    item=run_generation(base,ident,req,a.timeout,raw,result,capture_context,initial['engine_pid'])
                    if a.protocol:
                        item['fresh_context_check']=screen.fresh_context_check(item)
                        if not item['fresh_context_check']['verified']: raise ValueError('fresh request context was not verified')
                        final=item.get('engine_stats',{}).get('final',{})
                        expected={**protocol.get('expect',{}),**case.get('expect',{})}
                        item['effective_checks']={k:{'expected':v,'observed':final.get(k)} for k,v in expected.items()}
                        if any(v['expected']!=v['observed'] for v in item['effective_checks'].values()): raise ValueError('history effective settings mismatch')
                    if ident == 'calibration-tool' and not a.protocol:
                        item['declared_tool_observed']=tool_calls_observed(item)
                        if not item['declared_tool_observed']:
                            raise ValueError('tool calibration emitted no lookup_status tool call')
                    snap,data=control('profile_snapshot',lambda:snapshot(base,result['events'],a.timeout));same_process(initial,snap)
                    current=counts_from_bytes(data)
                    if any(y<x for x,y in zip(prior,current)): raise ValueError('generation profile decreased or wrapped')
                    delta=sum(y-x for x,y in zip(prior,current))
                    result['phases'].append({'case':ident,'generation_demand_delta':delta,'cumulative_demand':sum(current),
                                             'completion_tokens':(item.get('usage') or {}).get('completion_tokens',None),
                                             'snapshot_sha256':hashlib.sha256(data).hexdigest()})
                    if delta == 0: raise ValueError(f'{ident}: generation demand did not increase')
                    prior=current;save()
                result['final_inspection']=control('placement_inspect_final',lambda:inspect(base,result['events'],a.timeout));same_process(initial,result['final_inspection'])
                frozen=output/'generation-profile.bin'
                with frozen.open('xb') as f: f.write(data)
                result['profile']={'path':str(frozen),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),
                                   'total_generation_demand':sum(prior),'source':snap.get('source')}
            elif a.mode in ('backing-on','backing-off'):
                changed=control('placement_'+a.mode,lambda:post(base,'/v1/minimax/placement',{'action':a.mode},result['events'],a.timeout))
                same_process(initial,changed)
                if changed.get('enabled') != (a.mode=='backing-on'): raise ValueError('backing mode mismatch')
                result['backing']=changed
                result['final_inspection']=control('placement_inspect_final',lambda:inspect(base,result['events'],a.timeout))
                same_process(initial,result['final_inspection'])
            elif a.mode in ('snapshot','restore'):
                changed=control('placement_'+a.mode,lambda:post(base,'/v1/minimax/placement',{'action':a.mode},result['events'],a.timeout))
                same_process(initial,changed)
                result['learned_state_sha256']=learned_state_signature(changed)
                (output/'learned-checkpoint.json').write_text(json.dumps(changed,indent=2)+'\n')
                result['checkpoint']=changed
            elif a.mode in ('helper','all','helper-first','historical-helper-first'):
                profile_path=a.profile.expanduser().resolve();profile=profile_path.read_bytes()
                counts_from_bytes(profile)
                if not any(profile): raise ValueError('profile is all zero')
                # Reject incomplete geometry before changing the live placement.
                verify_mode='helper-first' if a.mode=='historical-helper-first' else a.mode
                desired_primary, desired_helper = selection(profile,initial,verify_mode)
                if a.mode == 'helper' and desired_helper == {tuple(r[:2]) for r in initial['helper_ids']}:
                    raise ValueError('profile would not change helper identities; choose a distinct A-to-B profile')
                result['input_profile']={'path':str(profile_path),'bytes':len(profile),'sha256':hashlib.sha256(profile).hexdigest()}
                submitted_path=profile_path;submitted=profile;action=a.mode
                if a.mode=='historical-helper-first':
                    if initial.get('resident_cap')!=-1 or len(set(initial['layer_devices']))!=1:
                        raise ValueError('historical roster encoding requires one device and no per-layer cap')
                    values=[0]*len(counts_from_bytes(profile));width=initial['experts_per_layer']
                    for li,e in desired_primary: values[li*width+e]=2
                    for li,e in desired_helper: values[li*width+e]=1
                    submitted=struct.pack('<'+'I'*len(values),*values)
                    submitted_path=output/'historical-roster-control.bin';submitted_path.write_bytes(submitted);action='all'
                    result['roster_control']={'path':str(submitted_path),'sha256':hashlib.sha256(submitted).hexdigest(),
                        'meaning':'Synthetic tier membership for historical primary-first PLACEMENT all, not demand counts. Desired rosters derive only from the original histogram using helper-first ranking; verify against that original histogram.'}
                changed=control('placement_'+a.mode,lambda:post(base,'/v1/minimax/placement',{'action':action,'profile_path':str(submitted_path)},result['events'],a.timeout))
                result['placement_response']=changed
                same_process(initial,changed)
                if changed.get('applied_profile_sha256') != hashlib.sha256(submitted).hexdigest():
                    raise ValueError('applied profile SHA-256 disagrees with frozen input')
                after=control('placement_inspect_final',lambda:inspect(base,result['events'],a.timeout));result['final_inspection']=after
                result['verification']=verify_placement(initial,after,profile,verify_mode)
                result['refresh_duration_ms']=changed.get('duration_ms')
                if a.mode=='helper':
                    item=run_generation(base,'after-helper-refresh',payload(FOLLOWUP,tune,32,a.seed),a.timeout,raw,result)
                    if not item.get('content','').strip() or (item.get('usage') or {}).get('completion_tokens',0) <= 0:
                        raise ValueError('post-refresh generation returned no visible completion')
                    end=inspect(base,result['events'],a.timeout);same_process(initial,end)
                    result['after_generation_inspection']=end
            result['complete']=True
    except Exception as exc:
        result['error']=f'{type(exc).__name__}: {exc}'
    finally: save()
    print(result_path)
    if result.get('error'): print(result['error'],file=sys.stderr)
    return 0 if result['complete'] else 1

if __name__=='__main__': sys.exit(main())
