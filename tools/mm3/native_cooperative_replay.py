#!/usr/bin/env python3
"""Bounded native cooperative replay with independent evidence gates.

No default physical costs: policy calibration cannot masquerade as validation.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from cooperative_replay.capture import Capture, read_capture, read_history, reproduce
from cooperative_replay.costs import ServiceTable
from cooperative_replay.machine import Machine
from cooperative_replay.policy import PolicyConfig
from cooperative_replay.validation import timing_validation, intervention_validation, require_search_gates, rank_held_out, digest
from cooperative_replay.calibration import fit,add_profile_launches,add_foreground_boundaries
from cooperative_replay.diagnostics import compare
from cooperative_replay.profile import read_profile


def load(path): return json.loads(Path(path).read_text())
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')


def export_workload(capture,output,config):
    lines=[f'MM3_REPLAY 1 {config.refill_budget} {config.stage_limit} {config.cpu_scale} {config.margin_ms}']
    count=0
    for e in capture.events:
        if e['type']=='boundary' and e['kind']=='forward_begin':
            count+=1
            # The positions are supplied by the complete captured target batch.
            routes=[r for r in capture.routes if r['forward']==e['forward']]
            pos=routes[0]['positions']
            if pos!=list(range(pos[0],pos[0]+e['rows'])):raise ValueError('fixed harness requires contiguous rows')
            lines.append(f"F {e['rows']} {pos[0]} {int(e['generation'])}")
        elif e['type']=='routes':
            lines.append(' '.join(map(str,['L',e['layer'],*e['original'],*e['executed']])))
    if count>256:raise ValueError('fixed workload exceeds native bounded batch limit')
    lines.append('END')
    with Path(output).open('x') as f:f.write('\n'.join(lines)+'\n')


def run_simulation(captures,costs,config,variant='central',backing=None,assignments='policy',generation_only=False):
    physical=ServiceTable.read(costs,held_out=[c.request_id for c in captures])
    machine=Machine(captures[0].initial,physical,config,variant=variant,backing=backing)
    result=machine.replay(captures,assignments=assignments,generation_only=generation_only)
    result['policy']=asdict(config)
    result['coverage_gaps']=sorted({str(q[0])+' '+q[3] for q in physical.queries if q[2]} |
                                   {str(p) for p in machine.scheduler.unqualified_pairs})
    result['interpolation_validation']=physical.interpolation_checks
    return result


def fixed_capture(source, checkpoint):
    """Bind complete evaluated work to an actual joined setup snapshot.

    No observed service/launch timestamp, draft duration or recorded assignment
    enters this fixed-target workload. The native runner uses synthetic tokens
    and excludes drafting/sampling; the source's original/executed IDs remain.
    """
    snapshot=checkpoint['replay_snapshot']
    if not checkpoint['placement']['ownership_complete'] or snapshot.get('replay_schema')!=1 or not snapshot.get('replay_supported'):
        raise ValueError('complete supported setup snapshot required')
    state=snapshot['replay_state']
    if (state['pending_sources'] or state['helper_queue_pending'] or snapshot['helper_exchange'][0]>=0 or
        any(s[0]>=0 or s[1]>=0 or s[2] or s[3] or s[4] for s in snapshot['staging'])):
        raise ValueError('fixed-work prediction requires an idle checkpoint with cleared staging')
    if (snapshot['expert_count'],snapshot['dimension'],snapshot['layer_count'])!=(
            source.initial['expert_count'],source.initial['dimension'],source.initial['layer_count']):
        raise ValueError('workload and setup geometry differ')
    events=[dict(type='fixed_workload_begin',diagnostic_primary_query=False)]
    forward=snapshot['forward'];generation=False
    for e in source.events:
        if e['type']=='boundary' and e['kind']=='forward_begin':
            forward+=1
            if e['generation'] and not generation:
                events.append(dict(type='generation_begin'));generation=True
            events.append(dict(type='boundary',kind='forward_begin',forward=forward,rows=e['rows'],generation=e['generation']))
        elif e['type']=='routes':
            events.append({k:e[k] for k in ('type','rows','layer','positions','generation','original','executed')}|dict(forward=forward))
        elif e['type']=='boundary' and e['kind']=='forward_end':
            events.append(dict(type='boundary',kind='forward_end',forward=forward,rows=e['rows']))
    if not generation:raise ValueError('fixed workload lacks generation')
    events.append(dict(type='generation_end'))
    return Capture(source.path,source.sha256,source.request_id,events,snapshot,{})


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    for name in ('reproduce','simulate','export-workload'):
        p=sub.add_parser(name);p.add_argument('captures',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
        if name!='reproduce':p.add_argument('--policy',type=Path)
        if name=='simulate':
            p.add_argument('--costs',required=True,type=Path);p.add_argument('--variant',choices=('central','low','high'),default='central')
            p.add_argument('--assignments',choices=('recorded','policy'),default='policy')
            p.add_argument('--generation-only',action='store_true')
    p=sub.add_parser('simulate-fixed',help='predict fixed target work from a complete joined hardware setup snapshot')
    p.add_argument('capture',type=Path);p.add_argument('--checkpoint',required=True,type=Path)
    p.add_argument('--costs',required=True,type=Path);p.add_argument('--policy',type=Path)
    p.add_argument('--backing',choices=('on','off'),default='off')
    p.add_argument('--variant',choices=('central','low','high'),default='central');p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('validate-timing');p.add_argument('--prediction',required=True);p.add_argument('--observations',required=True);p.add_argument('--effect',required=True,type=float);p.add_argument('--output',required=True)
    p=sub.add_parser('validate-interventions');p.add_argument('--plan',required=True);p.add_argument('--hardware',required=True);p.add_argument('--timing-gate',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('search');p.add_argument('manifest',type=Path);p.add_argument('--output',required=True)
    p=sub.add_parser('calibrate');p.add_argument('--probe',required=True,type=Path);p.add_argument('captures',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('calibrate-foreground',help='bounded no-copy generation model with independent probes and explicit submission boundaries')
    p.add_argument('--probe',required=True,type=Path);p.add_argument('--auxiliary',nargs='+',required=True,type=Path)
    p.add_argument('--profiles',nargs='+',required=True,type=Path);p.add_argument('--submission',nargs='+',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p=sub.add_parser('diagnose');p.add_argument('capture',type=Path);p.add_argument('--prediction',required=True);p.add_argument('--costs',required=True);p.add_argument('--output',required=True);p.add_argument('--isolated-costs')
    p=sub.add_parser('reconcile-profile');p.add_argument('capture',type=Path);p.add_argument('--sqlite',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=ap.parse_args(argv)
    if a.command in ('reproduce','simulate','export-workload'):
        captures=read_history(a.captures)
        if a.command=='reproduce':
            reports=[]
            for c in captures:reports.append(reproduce(c,previous=reports[-1] if reports else None))
            save(a.output,dict(stage=1,status='pass',requests=reports,timing_validated=False))
        else:
            config=PolicyConfig(**(load(a.policy) if a.policy else {}))
            if a.command=='export-workload':
                if len(captures)!=1:raise ValueError('export one complete request at a time')
                export_workload(captures[0],a.output,config)
            else:save(a.output,run_simulation(captures,a.costs,config,a.variant,assignments=a.assignments,generation_only=a.generation_only))
    elif a.command=='simulate-fixed':
        source=read_capture(a.capture);checkpoint=load(a.checkpoint);c=fixed_capture(source,checkpoint)
        result=run_simulation([c],a.costs,PolicyConfig(**(load(a.policy) if a.policy else {})),a.variant,backing=a.backing=='on')
        result['input_provenance']=dict(source_capture_sha256=source.sha256,checkpoint_sha256=digest(checkpoint),
            engine_pid=checkpoint['engine_pid'],recorded_emitted=sum(e['type']=='token_output' for e in source.events),
            scope='synthetic fixed target work; real original/executed rows; no drafting, sampling, capture or new outputs')
        save(a.output,result)
    elif a.command=='calibrate':save(a.output,fit(load(a.probe),[read_capture(p) for p in a.captures],probe_source=str(a.probe.resolve())))
    elif a.command=='calibrate-foreground':
        doc=fit(load(a.probe),[read_capture(p) for p in a.auxiliary],probe_source=str(a.probe.resolve()),foreground_services=False)
        for path in a.profiles:doc=add_profile_launches(doc,load(path),profile_source=str(path.resolve()),generation_only=True)
        save(a.output,add_foreground_boundaries(doc,[read_capture(p) for p in a.submission]))
    elif a.command=='diagnose':
        c=read_capture(a.capture);save(a.output,compare(c,load(a.prediction),ServiceTable.read(a.costs,held_out=[c.request_id]),
            ServiceTable.read(a.isolated_costs,held_out=[c.request_id]) if a.isolated_costs else None))
    elif a.command=='reconcile-profile':save(a.output,read_profile(a.sqlite,read_capture(a.capture)))
    elif a.command=='validate-timing':save(a.output,timing_validation(load(a.prediction),load(a.observations),required_effect_fraction=a.effect))
    elif a.command=='validate-interventions':save(a.output,intervention_validation(load(a.plan),load(a.hardware),timing_gate=load(a.timing_gate)))
    else:
        manifest=load(a.manifest);physical=ServiceTable.read(manifest['costs'])
        require_search_gates([load(p) for p in manifest['gates']],physical.sha256)
        history=manifest['history'];roles=[r['role'] for r in history]
        if 'conditioning' not in roles or 'held_out' not in roles or roles!=sorted(roles,key=lambda x:0 if x=='conditioning' else 1):
            raise ValueError('search requires chronological conditioning then complete held-out requests')
        captures=read_history([r['path'] for r in history])
        origin=manifest.get('conditioning_origin',{})
        if origin.get('kind') not in ('cold','independent_profile_seed') or not origin.get('evidence'):
            raise ValueError('conditioning must start from a declared cold or independent profile seed, not settled incumbent scores')
        if origin['kind']=='cold' and any(v for _,v,_ in captures[0].initial['replay_state']['scores']):
            raise ValueError('cold conditioning origin contains learned scores')
        candidates=manifest['candidates'];names={c['name'] for c in candidates}
        if manifest['incumbent'] not in names or not any(not c['policy'].get('extra_refill',True) for c in candidates):
            raise ValueError('include incumbent and no-extra-refill controls')
        if len(candidates)>64:raise ValueError('bounded search limit exceeded')
        held={c.request_id for c,r in zip(captures,history) if r['role']=='held_out'}
        results=[]
        for candidate in candidates:
            config=PolicyConfig(**candidate['policy'])
            # Every candidate replays all conditioning, rather than reusing the
            # incumbent's learned scores at the held-out boundary.
            runs=[run_simulation(captures,manifest['costs'],config,v) for v in ('central','low','high')]
            if any(r['coverage_gaps'] for r in runs):raise ValueError('candidate leaves qualified measurement coverage')
            rows=[]
            for central,low,high in zip(*(r['requests'] for r in runs)):
                if central['request_id'] in held:rows.append({**central,'uncertainty_ms':max(abs(central['generation_ms']-low['generation_ms']),abs(high['generation_ms']-central['generation_ms']))})
            results.append(dict(name=candidate['name'],policy=candidate['policy'],requests=rows))
        save(a.output,dict(scope='conditional recorded-work ranking; live confirmation required',results=results,ranking=rank_held_out(results,manifest['incumbent'])))


if __name__=='__main__':main()
