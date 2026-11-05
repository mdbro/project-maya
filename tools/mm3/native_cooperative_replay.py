#!/usr/bin/env python3
"""Bounded native cooperative replay with independent evidence gates.

No default physical costs: policy calibration cannot masquerade as validation.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from cooperative_replay.capture import read_capture, reproduce
from cooperative_replay.costs import ServiceTable
from cooperative_replay.machine import Machine
from cooperative_replay.policy import PolicyConfig
from cooperative_replay.validation import timing_validation, intervention_validation, require_search_gates, rank_held_out


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


def run_simulation(captures,costs,config,variant='central',backing=None):
    physical=ServiceTable.read(costs,held_out=[c.request_id for c in captures])
    machine=Machine(captures[0].initial,physical,config,variant=variant,backing=backing)
    result=machine.replay(captures)
    result['policy']=asdict(config)
    result['coverage_gaps']=sorted({str(q[0])+' interpolation' for q in physical.queries if q[2]} |
                                   {str(p) for p in machine.scheduler.unqualified_pairs})
    return result


def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__);sub=ap.add_subparsers(dest='command',required=True)
    for name in ('reproduce','simulate','export-workload'):
        p=sub.add_parser(name);p.add_argument('captures',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
        if name!='reproduce':p.add_argument('--policy',type=Path)
        if name=='simulate':p.add_argument('--costs',required=True,type=Path);p.add_argument('--variant',choices=('central','low','high'),default='central')
    p=sub.add_parser('validate-timing');p.add_argument('--prediction',required=True);p.add_argument('--observations',required=True);p.add_argument('--effect',required=True,type=float);p.add_argument('--output',required=True)
    p=sub.add_parser('validate-interventions');p.add_argument('--plan',required=True);p.add_argument('--hardware',required=True);p.add_argument('--timing-gate',required=True);p.add_argument('--output',required=True)
    p=sub.add_parser('search');p.add_argument('manifest',type=Path);p.add_argument('--output',required=True)
    a=ap.parse_args(argv)
    if a.command in ('reproduce','simulate','export-workload'):
        captures=[read_capture(p) for p in a.captures]
        if a.command=='reproduce':save(a.output,dict(stage=1,status='pass',requests=[reproduce(c) for c in captures],timing_validated=False))
        else:
            config=PolicyConfig(**(load(a.policy) if a.policy else {}))
            if a.command=='export-workload':
                if len(captures)!=1:raise ValueError('export one complete request at a time')
                export_workload(captures[0],a.output,config)
            else:save(a.output,run_simulation(captures,a.costs,config,a.variant))
    elif a.command=='validate-timing':save(a.output,timing_validation(load(a.prediction),load(a.observations),required_effect_fraction=a.effect))
    elif a.command=='validate-interventions':save(a.output,intervention_validation(load(a.plan),load(a.hardware),timing_gate=load(a.timing_gate)))
    else:
        manifest=load(a.manifest);physical=ServiceTable.read(manifest['costs'])
        require_search_gates([load(p) for p in manifest['gates']],physical.sha256)
        history=manifest['history'];roles=[r['role'] for r in history]
        if 'conditioning' not in roles or 'held_out' not in roles or roles!=sorted(roles,key=lambda x:0 if x=='conditioning' else 1):
            raise ValueError('search requires chronological conditioning then complete held-out requests')
        captures=[read_capture(r['path']) for r in history]
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
