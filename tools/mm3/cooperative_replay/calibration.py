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


def fit(probe, captures, *, probe_source):
    doc=copy.deepcopy(probe)
    doc['source_probe']=probe_source;doc['calibration_request_ids']=[c.request_id for c in captures]
    doc['interpolation_uncertainty']=.20
    for row in doc['measurements']:
        row['evidence']=[probe_source]
    auxiliary=defaultdict(list);provenance=defaultdict(set)
    def add(c,kind,rows,value,groups=(),context=0):
        if value<=0:return
        key=(kind,rows,tuple(sorted(groups)),context)
        auxiliary[key].append(value);provenance[key].add(c.request_id)
    for c in captures:
        route_by={(e['forward'],e['layer']):e for e in c.routes}
        begin={};target_ms=draft_ms=0;rounds=0;generation=None;duration=None
        for e in c.events:
            typ=e['type'];key=(e.get('forward'),e.get('layer'))
            route=route_by.get(key)
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
                add(c,'host_plan',route['rows'],e['coop_plan_ms'],uses.values())
                # CPU flag handling + final HOST merge is bounded by the helper
                # reduction samples; that reduction is modeled separately. A
                # direct zero-work HOST bookkeeping measurement is required.
        if duration is None:raise ValueError('training capture lacks a generation end before trace flushing')
        if rounds and duration>target_ms+draft_ms:
            add(c,'sampling',1,(duration-target_ms-draft_ms)/rounds)
    for (kind,rows,groups,context),samples in auxiliary.items():
        doc['measurements'].append(dict(kind=kind,rows=rows,groups=list(groups),context=context,
            samples_ms=samples,evidence=sorted(provenance[(kind,rows,groups,context)]),
            request_id=next(iter(provenance[(kind,rows,groups,context)])),relative_uncertainty=.10))
    # Fit directional pair rates from actual overlap, not raw elapsed ratios
    # applied over an entire future operation. Retain repeat variation and the
    # imperfect host/GPU alignment as uncertainty.
    factors=defaultdict(list)
    for row in probe.get('contention_observations',[]):
        a,b=row['a'],row['b'];ia,ib=row['isolated_a_ms'],row['isolated_b_ms']
        ca,cb=row['concurrent_a_ms'],row['concurrent_b_ms'];overlap=min(ca,cb)
        if min(ia,ib,overlap)<=0:raise ValueError('invalid contention observation')
        for pair,isolated,concurrent in (((a,b),ia,ca),((b,a),ib,cb)):
            lost=max(0,concurrent-isolated)
            factor=1/max(.25,1-lost/overlap)
            factors[pair].append(factor)
    doc['contention']=[]
    for pair,values in factors.items():
        med=statistics.median(values)
        doc['contention'].append(dict(pair=list(pair),factor=med,low=max(1,min(values)*.9),high=max(values)*1.1,
            evidence=[probe_source],qualified=False,scope='bounded measured pair fit; held-out validation required'))
    doc['limitations']+=['Auxiliary timing captures include their declared instrumentation overhead.',
        'Pair-rate fitting has not validated multiway contention or unmeasured directions.',
        'No physical measurement is derived from cooperative_cpu_scale or admission prices.']
    return doc
