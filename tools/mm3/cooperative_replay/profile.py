"""Read-only CUPTI/NVTX reconciliation for a captured fixed expert workload."""
from __future__ import annotations
from bisect import bisect_right
from collections import Counter,defaultdict
from pathlib import Path
import re
import sqlite3
from .structure import helper_launches,execution_shape
from .overlap import union,duration_ns,intersection_ns


def read_profile(path,capture):
    connection=sqlite3.connect('file:'+str(Path(path).resolve())+'?mode=ro',uri=True)
    connection.row_factory=sqlite3.Row
    strings={r['id']:r['value'] for r in connection.execute('select id,value from StringIds')}
    ranges=defaultdict(list);sync=[]
    for r in connection.execute('select start,end,text,textId,globalTid from NVTX_EVENTS'):
        text=r['text'] or strings.get(r['textId'],'')
        if 'request='+capture.request_id not in text:continue
        match=re.search(r'clock_sync_steady_ns=(\d+)',text)
        if match:sync.append(int(match[1])-r['start'])
        m=re.search(r'forward=(-?\d+) layer=(-?\d+) phase=(\S+)',text)
        if m and r['end'] is not None:
            f,l,phase=m.groups();ranges[r['globalTid'],phase].append((r['start'],r['end'],int(f),int(l)))
    if len(sync)!=1:raise ValueError('exactly one request clock alignment marker is required')
    offset=sync[0]
    for v in ranges.values():v.sort()
    phases={}
    for (_,phase),entries in ranges.items():
        for begin,end,f,l in entries:phases[(f,l),phase]=[begin+offset,end+offset]
    runtime={r['correlationId']:dict(r) for r in connection.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME')}
    lane_kernels=defaultdict(list);copies=defaultdict(list)
    def locate(api,phase):
        entries=ranges.get((api['globalTid'],phase),[])
        i=bisect_right(entries,(api['start'],float('inf'),float('inf'),float('inf')))-1
        if i>=0 and entries[i][0]<=api['start']<=entries[i][1]:return entries[i][2:]
        return None
    for r in connection.execute('select start,end,deviceId,correlationId,demangledName,gridX,gridY,gridZ,streamId from CUPTI_ACTIVITY_KIND_KERNEL'):
        api=runtime.get(r['correlationId'])
        if api is None:continue
        kind='helper' if r['deviceId']==1 else 'primary'
        key=locate(api,'helper_worker_cuda' if kind=='helper' else 'primary_expert_submit_and_join')
        if key is None:continue
        name=strings[r['demangledName']]
        if kind=='primary' and not ('minimax_ffn' in name or ('native_experts' in name and ('::setup(' in name or '::scatter(' in name))):continue
        lane_kernels[key,kind].append(dict(start_ns=r['start']+offset,end_ns=r['end']+offset,
            api_start_ns=api['start']+offset,api_end_ns=api['end']+offset,name=name,
            grid=[r['gridX'],r['gridY'],r['gridZ']],stream=r['streamId']))
    for r in connection.execute('select start,end,deviceId,correlationId,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY'):
        api=runtime.get(r['correlationId'])
        if api is None:continue
        key=locate(api,'helper_worker_cuda') if r['deviceId']==1 else None
        if key is not None:copies[key].append(dict(start_ns=r['start']+offset,end_ns=r['end']+offset,bytes=r['bytes'],direction=r['copyKind']))
    connection.close()
    routes={(r['forward'],r['layer']):r for r in capture.routes};assignments={};observations={};cpu={};services={}
    for e in capture.events:
        key=e.get('forward'),e.get('layer')
        if e['type']=='expert_assignment':
            lanes=defaultdict(Counter)
            for a in e['assignments']:
                lane='primary' if a['service'].startswith('primary') else a['service'];lanes[lane][a['expert']]+=1
            assignments[key]=lanes
        elif e['type']=='primary_service_observation':observations[(e['measured_forward'],e['measured_layer']),'primary']=e
        elif e['type']=='helper_service_observation':observations[key,'helper']=e
        elif e['type']=='service':services[key]=e
    for (_,phase),entries in ranges.items():
        if phase=='cpu_expert_batch':
            for begin,end,f,l in entries:cpu[f,l]=[begin+offset,end+offset]
    rows=[]
    for (key,kind),kernels in sorted(lane_kernels.items()):
        if key not in routes:continue
        r=routes[key];kernels.sort(key=lambda k:k['start_ns'])
        active=[[k['start_ns'],k['end_ns']] for k in kernels];begin=active[0][0];end=active[-1][1]
        busy=duration_ns(active)*1e-6;span=(end-begin)*1e-6
        late=sum(max(0,min(k['start_ns'],k['api_start_ns'])-previous['end_ns']) for previous,k in zip(kernels,kernels[1:]))*1e-6
        jobs=assignments[key].get(kind,{});groups=list(jobs.values());shape=execution_shape(kind,r['rows'],groups)
        expected=4*len(helper_launches(groups)) if kind=='helper' else 5*r['rows']
        if len(kernels)!=expected:raise ValueError(f'{kind} kernel/launch mismatch at {key}: {len(kernels)} != {expected}')
        observed=observations[key,kind]['compute_ms'];route_ns=int(r['time_s']*1e9)
        item=dict(forward=key[0],layer=key[1],generation=bool(r['generation']),parent_rows=r['rows'],kind=kind,
            groups=groups,expert_ids=list(jobs),shape=shape,kernels=kernels,
            gpu_active_ms=busy,gpu_first_to_last_ms=span,event_span_ms=observed,
            inter_kernel_gap_ms=span-busy,late_host_submission_lower_bound_ms=late,
            event_span_minus_active_ms=observed-busy,
            start_ns=begin,end_ns=end,start_from_route_ms=(begin-route_ns)*1e-6,
            completion_from_route_ms=(end-route_ns)*1e-6,
            cpu_overlap_ms=intersection_ns(active,[cpu[key]])*1e-6 if key in cpu else 0,
            cpu_interval_ns=cpu.get(key),helper_transfers=copies.get(key,[]))
        primary_host=phases.get((key,'primary_expert_submit_and_join'))
        prepare_host=phases.get((key,'helper_submit'))
        collect_host=phases.get((key,'helper_collect'))
        if primary_host:
            cpu_or_collect=cpu.get(key) or collect_host
            if cpu_or_collect:item['host_submission_ms']=(cpu_or_collect[0]-primary_host[0])*1e-6
            item['host_branch_end_ns']=primary_host[1]
        if prepare_host and key in services:
            item['route_capture_host_ms']=(prepare_host[0]-route_ns)*1e-6-services[key]['coop_plan_ms']
        if kind=='helper':
            worker=phases[key,'helper_worker_cuda'];transfer=copies[key]
            uploads=[v for v in transfer if v['direction']==1];downloads=[v for v in transfer if v['direction']==2]
            if len(uploads)!=2 or len(downloads)!=1:raise ValueError('helper activation/jobs/result transfers are incomplete')
            first=min(v['start_ns'] for v in uploads);last=max(v['end_ns'] for v in uploads)
            result_end=downloads[0]['end_ns'];observation=observations[key,'helper']
            activation_extra=max(0,observation['activation_ms']-(last-first)*1e-6)
            result_extra=max(0,observation['result_ms']-(result_end-downloads[0]['start_ns'])*1e-6)
            initial_delay=max(0,(first-worker[0])*1e-6)
            detection=max(0,(worker[1]-result_end)*1e-6)
            item.update(worker_interval_ns=worker,
                submission_delay_low_ms=max(0,initial_delay-activation_extra),submission_delay_high_ms=initial_delay,
                publication_delay_low_ms=max(0,detection-result_extra),publication_delay_high_ms=detection,
                timer_alignment_note='bounds subtract the event envelope outside the correlated transfer intervals')
        if kind=='helper':
            item['width_groups']=[]
            for launch,first in zip(helper_launches(groups),range(0,len(kernels),4)):
                subset=kernels[first:first+4]
                item['width_groups'].append(dict(**launch,kernels=subset,
                    active_ms=sum(k['end_ns']-k['start_ns'] for k in subset)*1e-6,
                    span_ms=(subset[-1]['end_ns']-subset[0]['start_ns'])*1e-6))
        rows.append(item)
    summaries={}
    for kind in ('primary','helper'):
        selected=[r for r in rows if r['kind']==kind and r['generation']]
        summaries[kind]=dict(calls=len(selected),**{name:sum(r[name] for r in selected) for name in (
            'gpu_active_ms','gpu_first_to_last_ms','event_span_ms','inter_kernel_gap_ms',
            'late_host_submission_lower_bound_ms','event_span_minus_active_ms','cpu_overlap_ms')})
    return dict(request_id=capture.request_id,sqlite=str(Path(path).resolve()),clock_offset_ns=offset,
        clock_alignment_scope='NVTX/CUPTI intervals share the Nsight clock. Absolute mapping to the native clock uses one marker and has unmeasured marker-call latency; native-relative endpoints are point estimates.',
        scope='profiled companion; CUDA/NVTX overhead included; not original-run execution endpoints',
        summary=summaries,rows=rows,cpu_rows=[dict(forward=f,layer=l,start_ns=v[0],end_ns=v[1],duration_ms=(v[1]-v[0])*1e-6) for (f,l),v in cpu.items()])
