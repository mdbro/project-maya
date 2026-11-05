"""Matched observed and independently scheduled operations.

This consumes predictions read-only. Observed durations never enter the scheduler.
Missing execution endpoints remain missing; a CUDA event span does not locate its
start or finish on the host clock and is not labelled pure arithmetic service.
"""
from __future__ import annotations
from collections import Counter,defaultdict
import statistics
from .structure import execution_shape


def _summary(rows):
    out={}
    for kind in sorted({r['kind'] for r in rows}):
        matched=[r for r in rows if r['kind']==kind and r['assignments_match']]
        if not matched:continue
        actual=sum(r['observed_service_ms'] for r in matched)
        isolated=sum(r['estimated_isolated_ms'] for r in matched)
        scheduled=sum(r['predicted_service_ms'] for r in matched)
        errors=sorted(abs(r['predicted_service_ms']/r['observed_service_ms']-1) for r in matched if r['observed_service_ms']>0)
        out[kind]=dict(calls=len(matched),observed_sum_ms=actual,isolated_sum_ms=isolated,
            scheduled_sum_ms=scheduled,isolated_to_live_relative_error=isolated/actual-1 if actual else None,
            scheduled_to_live_relative_error=scheduled/actual-1 if actual else None,
            contention_extension_sum_ms=sum(r['contention_extension_ms'] for r in matched),
            queue_sum_ms=sum(r['queue_ms'] for r in matched),
            median_absolute_relative_error=statistics.median(errors) if errors else None,
            p90_absolute_relative_error=errors[int(.9*(len(errors)-1))] if errors else None)
    return out


def compare(capture, prediction, physical):
    if prediction['physical_sha256']!=physical.sha256:raise ValueError('prediction and diagnostic physical versions differ')
    routes={(e['forward'],e['layer']):e for e in capture.routes}
    layers={(e['forward'],e['layer']):e for e in prediction['layers']}
    ops={o['id']:o for o in prediction['operations']}
    assignments={};service={};rows=[]
    for e in capture.events:
        key=(e.get('forward'),e.get('layer'))
        if e['type']=='expert_assignment':
            lanes=defaultdict(Counter)
            for a in e['assignments']:
                lane='primary' if a['service'].startswith('primary') else a['service']
                lanes[lane][a['expert']]+=1
            assignments[key]=lanes
        elif e['type']=='service':service[key]=e
    def sample(kind,e,value,field,lane=None,key=None,boundary='host wall span'):
        key=key or (e['forward'],e['layer']);route=routes[key]
        if not route['generation']:return
        predicted=layers[key];indices=predicted[field]
        if indices is None or indices==[]:return
        if not isinstance(indices,list):indices=[indices]
        selected=[ops[i] for i in indices]
        if any(o['finish_ms'] is None for o in selected):raise ValueError('matched foreground operation has no predicted finish')
        group_map=assignments[key].get(lane,{}) if lane else {}
        match=lane is None or set(predicted[lane])==set(group_map)
        isolated=sum(o['isolated_ms'] for o in selected)
        scheduled=sum(o['predicted_service_ms'] for o in selected)
        # CPU/helper/primary event timers do not carry aligned execution endpoints
        # in the original capture. A service report time is not GPU completion.
        rows.append(dict(kind=kind,forward=key[0],layer=key[1],parent_rows=route['rows'],
            groups=list(group_map.values()),shape=execution_shape(kind,route['rows'],list(group_map.values())),
            assignments_match=match,estimated_isolated_ms=isolated,
            predicted_service_ms=scheduled,contention_extension_ms=sum(o['pressure_ms'] for o in selected),
            queue_ms=sum(o['queue_ms'] for o in selected),
            dependency_delay_ms=selected[0]['dependency_delay_ms'],
            predicted_start_from_route_ms=selected[0]['start_ms']-predicted['service_ready_ms'],
            predicted_completion_from_route_ms=selected[-1]['finish_ms']-predicted['service_ready_ms'],
            host_submission_ms=ops[predicted['primary_submit_op']]['predicted_service_ms'] if lane=='primary' else None,
            unknown_contention_exposure_ms=sum(o.get('unknown_contention_exposure_ms',0) for o in selected),
            observed_service_ms=value,observed_boundary=boundary,
            observed_start_from_route_ms=None,observed_completion_from_route_ms=None,
            observation_report_ns=e['steady_ns'],
            execution_endpoint_status='not captured on an aligned clock; report timestamp is not execution completion',
            operations=indices))
    for e in capture.events:
        kind=e['type'];key=(e.get('forward'),e.get('layer'))
        if kind=='service':sample('cpu',e,e['coop_cpu_run_ms'],'cpu_op','cpu')
        elif kind=='primary_service_observation':
            key=(e['measured_forward'],e['measured_layer'])
            sample('primary',e,e['compute_ms'],'primary_op','primary',key,'CUDA event span; includes possible host enqueue gaps')
        elif kind=='helper_service_observation':
            for name,observed,field,lane,boundary in (
                ('helper','compute_ms','helper_ops','helper','CUDA event span across all occupied widths; includes possible enqueue gaps'),
                ('helper_dispatch','dispatch_ms','helper_dispatch_op','helper','enqueue to worker start'),
                ('helper_prepare','prepare_ms','helper_prepare_op','helper','caller preparation and submit'),
                ('h_activation','activation_ms','helper_activation_op','helper','CUDA event span for activation and job descriptor uploads'),
                ('h_result','result_ms','helper_result_op','helper','CUDA event span for result download'),
                ('helper_reduce','reduce_ms','helper_reduce_op','helper','caller collection and merge')):
                sample(name,e,e[observed],field,lane,boundary=boundary)
    branch=[]
    for key,e in service.items():
        if not routes[key]['generation']:continue
        p=layers[key];last=ops[p['host_reduce_op']]
        branch.append(dict(forward=key[0],layer=key[1],parent_rows=p['rows'],
            predicted_host_branch_ms=last['finish_ms']-p['service_ready_ms'],
            observed_host_branch_ms=e['host_service_ms'],
            observed_host_completion_ns=e['steady_ns'],
            observed_primary_join_at_next_route_ns=None,
            scope='route decision boundary to caller service completion; primary completion may follow'))
    begin=next(e for e in capture.events if e['type']=='generation_begin')
    end=next(e for e in capture.events if e['type']=='generation_end')
    actual=end.get('elapsed_ms',(end['steady_ns']-begin['steady_ns'])*1e-6)
    request=next(r for r in prediction['requests'] if r['request_id']==capture.request_id)
    counts=Counter(e['type'] for e in capture.events)
    return dict(stage=2,status='diagnostic',request_id=capture.request_id,
        physical_sha256=physical.sha256,assignment_mode=prediction.get('assignment_mode','policy'),
        observations_used_as_predictions=False,
        generation=dict(observed_ms=actual,predicted_ms=request['generation_ms'],
            relative_error=request['generation_ms']/actual-1,
            observed_emitted=end.get('emitted'),predicted_recorded_emitted=request['emitted'],
            evaluated_rows=sum(e['rows'] for e in capture.events if e['type']=='boundary' and e['kind']=='forward_begin' and e['generation'])),
        service=_summary(rows),service_by_parent_width={str(w):_summary([r for r in rows if r['parent_rows']==w]) for w in sorted({r['parent_rows'] for r in rows})},
        service_rows=rows,branch_rows=branch,
        actions=dict(observed_primary_issues=counts['primary_exchange_issue'],predicted_primary_issues=prediction['counts'].get('primary_issued',0),
            observed_helper_issues=counts['helper_exchange_issue'],predicted_helper_issues=prediction['counts'].get('helper_issued',0)),
        coverage_gaps=prediction.get('coverage_gaps',[]),
        limitations=['Service sums overlap across lanes; they are not an additive decomposition of request error.',
            'Recorded assignments isolate physical/dependency error; policy mode can additionally diverge in assignments.',
            'Original timers lack aligned GPU execution endpoints. A profiled companion can identify boundaries but cannot retroactively supply original endpoints.',
            'No ranking or live speed claim follows from aggregate timing agreement.'])
