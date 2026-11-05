"""Construct physical tables from isolated probes and independent captures.

Policy calibration is intentionally never read. Original timestamps are used
only to derive calibration observations, not as launch times in a replay.
"""
from __future__ import annotations
from collections import defaultdict, Counter
import copy
import hashlib
import json
import statistics
from .overlap import fit_observation


def fit(probe, captures, *, probe_source):
    if probe.get('worker_measurement')!='one CPU pool exists at a time' or not probe.get('cpu_participants'):
        raise ValueError('service probe lacks isolated worker configurations and explicit participant count')
    doc=copy.deepcopy(probe)
    doc['source_probe']=probe_source;doc['calibration_request_ids']=[c.request_id for c in captures]
    doc.pop('interpolation_uncertainty',None)
    for row in doc['measurements']:
        row['evidence']=[probe_source]
        row.pop('relative_uncertainty',None)
        # Native launch count depends on parent width, not the device-side
        # number of active experts. Keep a measured width table for host launch
        # overhead; expert compute retains its full group composition.
        if row['kind']=='primary_submit':row['groups']=[]
    auxiliary=defaultdict(list);provenance=defaultdict(set)
    concurrent=defaultdict(list);concurrent_sources=defaultdict(set);clean_counts={}
    def add(c,kind,rows,value,groups=(),context=0):
        if value<=0:return
        key=(kind,rows,tuple(sorted(groups)),context)
        auxiliary[key].append(value);provenance[key].add(c.request_id)
    for c in captures:
        route_by={(e['forward'],e['layer']):e for e in c.routes}
        fixed=any(e['type']=='fixed_workload_begin' for e in c.events)
        clean=set();assignments={}
        for e in c.events:
            if e['type']=='service':
                key=e['forward'],e['layer'];r=route_by[key]
                if e['demand_uploads']==0 and all(x[0]<0 for x in r['exchanges']) and r['helper_exchange'][0]<0:clean.add(key)
            elif e['type']=='expert_assignment':
                lanes=defaultdict(Counter)
                for item in e['assignments']:
                    lane='primary' if item['service'].startswith('primary') else item['service']
                    lanes[lane][item['expert']]+=1
                assignments[e['forward'],e['layer']]=lanes
        clean_counts[c.request_id]=len(clean)
        def observed(kind,key,value,lane):
            if key not in clean or value<=0:return
            groups=tuple(sorted(assignments[key].get(lane,{}).values()))
            if not groups and kind not in ('primary',):return
            workers=probe['cpu_participants'] if kind=='cpu' else 'none'
            shape=kind,route_by[key]['rows'],groups,workers
            concurrent[shape].append(value);concurrent_sources[shape].add(c.request_id)
        begin={};target_ms=draft_ms=0;rounds=0;generation=None;duration=None
        for e in c.events:
            typ=e['type'];key=(e.get('forward'),e.get('layer'))
            route=route_by.get(key)
            if typ=='service':observed('cpu',key,e['coop_cpu_run_ms'],'cpu')
            elif typ=='primary_service_observation':
                observed('primary',(e['measured_forward'],e['measured_layer']),e['compute_ms'],'primary')
            elif typ=='helper_service_observation':
                for name,field in (('helper','compute_ms'),('h_activation','activation_ms'),('h_result','result_ms'),
                                   ('helper_prepare','prepare_ms'),('helper_dispatch','dispatch_ms'),('helper_reduce','reduce_ms')):
                    observed(name,key,e[field],'helper')
            if typ=='generation_begin':generation=e['steady_ns']
            elif typ=='generation_end':duration=e.get('elapsed_ms',(e['steady_ns']-generation)*1e-6)
            elif typ=='target_forward_begin':begin['target']=e
            elif typ=='target_forward_end' and 'target' in begin:
                first=begin.pop('target')
                if first.get('phase')!='prefill':target_ms+=(e['steady_ns']-first['steady_ns'])*1e-6
            elif typ=='draft_begin':begin['draft']=e
            elif typ=='draft_end' and 'draft' in begin:
                first=begin.pop('draft');ms=(e['steady_ns']-first['steady_ns'])*1e-6;draft_ms+=ms
                add(c,'draft',max(1,e['proposed']),ms)
            elif typ=='round_end':rounds+=1
            elif typ=='prepare_gpu_observation':
                # Context interpolation is bounded by actually measured points.
                add(c,'layer_pre',e['rows'],e['gpu_ms'],context=e['context'])
            elif typ=='prepare_host_observation':add(c,'host_pre',e['rows'],e['host_ms'])
            elif typ=='route_readback_observation':add(c,'route_readback',e['rows'],e['gpu_ms'])
            elif typ=='target_overhead_observation':
                add(c,'target_prefix',e['rows'],e['prefix_ms'])
                add(c,'target_tail',e['rows'],e['tail_ms'])
            elif typ=='primary_service_observation':add(c,'plan_upload',e['rows'],e['plan_upload_ms'])
            elif typ=='service' and route:
                uses=Counter(route['executed'])
                add(c,'host_plan',route['rows'],e['coop_plan_ms'])
                # CPU flag handling + final HOST merge is bounded by the helper
                # reduction samples; that reduction is modeled separately. A
                # direct zero-work HOST bookkeeping measurement is required.
        if fixed:continue
        if duration is None:raise ValueError('training capture lacks a generation end before trace flushing')
        if rounds and duration>target_ms+draft_ms:
            add(c,'sampling',1,(duration-target_ms-draft_ms)/rounds)
    for (kind,rows,groups,context),samples in auxiliary.items():
        doc['measurements'].append(dict(kind=kind,rows=rows,groups=list(groups),context=context,
            samples_ms=samples,evidence=sorted(provenance[(kind,rows,groups,context)]),
            request_ids=sorted(provenance[(kind,rows,groups,context)])))
    if any(any(e['type']=='fixed_workload_begin' for e in c.events) for c in captures):
        doc['cooperative_service_kinds']=sorted({k[0] for k in concurrent})
        doc['cooperative_clean_intervals']=clean_counts
        doc['cpu_working_set']='cooperative'
        for (kind,rows,groups,workers),samples in concurrent.items():
            key=kind,rows,groups,workers
            doc['measurements'].append(dict(kind=kind,rows=rows,groups=list(groups),workers=workers,
                working_set='cooperative',samples_ms=samples,evidence=sorted(concurrent_sources[key]),
                request_ids=sorted(concurrent_sources[key]),
                condition='concurrent foreground; no expert weight copies'))
        doc['limitations'].append('Concurrent foreground tables use only independently captured intervals with zero demand weight uploads and no pending primary/helper exchanges. Their base already includes foreground cooperation; only separate weight-copy contention is added.')
    factors=defaultdict(list);rejected=[]
    for index,row in enumerate(probe.get('contention_observations',[])):
        try:
            for fitted in fit_observation(row):factors[fitted['pair']].append(fitted['factor'])
        except ValueError as exc:rejected.append(dict(observation=index,pair=[row['a'],row['b']],reason=str(exc)))
    doc['contention']=[]
    doc['rejected_contention_observations']=rejected
    for pair,values in factors.items():
        med=statistics.median(values)
        doc['contention'].append(dict(pair=list(pair),factor=med,low=min(values),high=max(values),
            evidence=[probe_source],qualified=False,scope='aligned-overlap pair fit; repeat range; held-out and multiway validation required'))
    doc['limitations']+=['Auxiliary timing captures include their declared instrumentation overhead.',
        'Pair-rate fitting has not validated multiway contention or unmeasured directions.',
        'No physical measurement is derived from cooperative_cpu_scale or admission prices.']
    return doc


def add_profile_launches(document, profile, *, profile_source):
    """Add measured width groups and the remaining event-envelope overhead.

    These are representative foreground observations, not isolated arithmetic.
    No primary/helper multiplier is fitted. The calibration request is registered
    so none of its durations can later be used as held-out evidence.
    """
    doc=copy.deepcopy(document);cells=defaultdict(list);fixed=defaultdict(list)
    request=profile['request_id']
    if request not in doc['calibration_request_ids']:doc['calibration_request_ids'].append(request)
    for row in profile['rows']:
        if row['kind']!='helper':continue
        spans=0
        for group in row['width_groups']:
            key=group['width'],group['jobs'],tuple(group['reuse'])
            cells[key].append(group['span_ms']);spans+=group['span_ms']
        remainder=row['event_span_ms']-spans
        if remainder<-.001:raise ValueError('profiled groups exceed the measured helper event envelope')
        fixed[row['parent_rows'],len(row['width_groups'])].append(max(1e-6,remainder))
    for (width,jobs,reuse),samples in cells.items():
        doc['measurements'].append(dict(kind='helper_width_group',width=width,jobs=jobs,reuse=list(reuse),
            working_set='cooperative',samples_ms=samples,evidence=[profile_source],request_ids=[request],
            boundary='first kernel start to last kernel end for one occupied width'))
    for (rows,launches),samples in fixed.items():
        doc['measurements'].append(dict(kind='helper_fixed',rows=rows,context=launches,groups=[],
            working_set='cooperative',samples_ms=samples,evidence=[profile_source],request_ids=[request],
            boundary='helper event envelope minus observed per-width spans; includes inter-launch/driver gaps'))
    doc['helper_width_measurements']=True
    doc['service_boundaries']={'cpu':'native CPU batch wall service in full-bank foreground',
        'primary':'primary CUDA event envelope in production foreground, including driver/submission gaps',
        'helper':'per-width kernel envelopes plus separately measured inter-launch/driver gaps'}
    doc['limitations'].append('Foreground conditioning is explicit. Component observations include their measured concurrent environment; their difference from an isolated probe is not identified as a causal contention factor.')
    return doc
