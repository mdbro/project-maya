#!/usr/bin/env python3
"""Approximate ordered residency replay; never a prediction of end-to-end speed.

Consumes schema-1 request/routes/service JSONL emitted by mm3_native_trace.inc.
Both arms start from the first ownership snapshot, retain primary state across
requests and use the initial helper roster as frozen exogenous ownership. Every routed
row, including later-rejected speculative rows, contributes original demand.
Observed staging slots are opportunities, not a reconstruction of hypothetical
staging; service observations and transfer ranges are estimates, not measurements
of the alternative execution. See output assumptions before interpreting ranks.
"""
from __future__ import annotations
import argparse
import collections
import json
import math
from pathlib import Path


def read_trace(path):
    requests, events = [], []
    pending = None
    last_event = -1
    active = None
    for number, line in enumerate(Path(path).read_text().splitlines(), 1):
        try: row = json.loads(line)
        except json.JSONDecodeError as exc: raise ValueError(f'line {number}: invalid JSON') from exc
        typ = row.get('type')
        if typ == 'request':
            if pending or active is not None: raise ValueError('request before preceding footer')
            if row.get('schema') != 1 or not row.get('layers') or row.get('record_bytes', 0) <= 0:
                raise ValueError('invalid request snapshot')
            if requests and row.get('request') <= requests[-1].get('request'): raise ValueError('request sequence regressed')
            requests.append(row); active = row['request']
        elif typ == 'routes':
            if pending or active is None or row.get('request') != active:
                raise ValueError('unpaired routes or missing request snapshot')
            if type(row.get('event')) is not int or row['event'] <= last_event:
                raise ValueError('events must be strictly ordered')
            last_event = row['event']; count = row.get('rows')
            if type(count) is not int or count <= 0 or len(row.get('positions', [])) != count:
                raise ValueError('invalid route row count or positions')
            original, executed = row.get('original', []), row.get('executed', [])
            if len(original) != len(executed) or len(original) != count * 4:
                raise ValueError('expected four original and executed expert IDs per row')
            if any(type(x) is not int or x < 0 for x in original + executed):
                raise ValueError('invalid routed expert ID')
            pending = row
        elif typ == 'service':
            if not pending or any(row.get(k) != pending.get(k) for k in ('request', 'event', 'forward')):
                raise ValueError('service does not match route event')
            if not isinstance(row.get('host_service_ms'), (float, int)) or row['host_service_ms'] < 0:
                raise ValueError('invalid host service observation')
            events.append((pending, row)); pending = None
        elif typ == 'end':
            if pending or active is None or row.get('request') != active:
                raise ValueError('unmatched request footer')
            if (type(row.get('forward')) is not int or row['forward'] < requests[-1]['forward'] or
                (events and events[-1][0]['request'] == active and row['forward'] < events[-1][0]['forward'])):
                raise ValueError('invalid footer forward')
            active = None
        else: raise ValueError(f'line {number}: unknown record type')
    if pending or active is not None or not requests or not events: raise ValueError('incomplete or empty trace')
    if not any(route.get('generation') for route, _ in events): raise ValueError('trace has no generation events')
    return requests, events


def initial_state(request):
    primary, helper, host = set(), set(), set()
    for layer in request['layers']:
        li = layer['layer']
        for name, target in [('primary', primary), ('helper', helper), ('host', host)]:
            for expert in layer[name]:
                key = (li, expert)
                if key in target: raise ValueError('duplicate ownership')
                target.add(key)
    if primary & helper or primary & host or helper & host:
        raise ValueError('snapshot ownership overlaps; replay requires exclusive initial ownership')
    return primary, helper, host


def replay(requests, events, policy, *, decay=.95, horizon=20, margin=0., refill_budget=8,
           h2d_ms=2., d2h_ms=2., interference_ms=.1, service_saved_ms=1.):
    primary, helper, host = initial_state(requests[0]); capacity = len(primary); helper_count = len(helper)
    if not primary or not host: raise ValueError('snapshot must include primary and host records')
    if any(r['record_bytes'] != requests[0]['record_bytes'] for r in requests):
        raise ValueError('record geometry changed across requests')
    initial_records = primary | helper | host
    scores = {}; updated = {}; pending = []; clock_ms = 0.; current_forward = None
    totals = collections.Counter(); rows = []; per_request = collections.defaultdict(collections.Counter); current_request = None
    record_bytes = requests[0]['record_bytes']; remaining = refill_budget; last_free_slots = 8; ordinary_pending = 0; protected = set();  transfer_charge = 0.; previous_service_time = None
    def score(key, forward):
        return scores.get(key, 0.) * decay ** max(0, forward - updated.get(key, forward))
    def publish(forward, current_layer):
        nonlocal pending
        kept=[]
        for entry in pending:
            if entry['deadline_ms'] <= clock_ms and entry['incoming'][0] != current_layer and entry['victim'][0] != current_layer:
                primary.remove(entry['victim']); host.add(entry['victim'])
                host.remove(entry['incoming']); primary.add(entry['incoming'])
                totals['replacements'] += 1
            else: kept.append(entry)
        pending = kept
        assert len(primary) == capacity and len(helper) == helper_count and not primary & host
    def submit(forward, ready, early):
        nonlocal remaining, transfer_charge
        if remaining <= (1 if early else 0) or len(pending) >= 2 or len(pending) >= min(last_free_slots, 4) or len(pending) + ordinary_pending >= 4: return
        locked = {x[k] for x in pending for k in ('incoming', 'victim')}
        victims = primary - locked - protected
        if not victims: return
        victim = min(victims, key=lambda x: (score(x,forward), x)); cold = score(victim,forward)
        candidates = host - locked - protected
        if early: candidates &= ready
        if not candidates: return
        def gain(key):
            delta = score(key,forward) - cold
            return delta * (1-decay**horizon) * service_saved_ms - d2h_ms - (0 if key in ready else h2d_ms) - interference_ms - margin
        if policy == 'legacy':
            candidate=max(candidates,key=lambda x:(score(x,forward),x)); hot=score(candidate,forward)
            if hot < 1.8 or hot <= 1.25*cold+1: return
        else:
            candidate=max(candidates,key=lambda x:(gain(x),x))
            if gain(candidate) <= 0: return
        reused = candidate in ready
        # The source and victim remain protected until a forward-boundary publication.
        duration = d2h_ms + (0 if reused else h2d_ms)
        pending.append({'incoming':candidate,'victim':victim,'deadline_ms':clock_ms+duration})
        transfer_charge += duration * interference_ms
        per_request[current_request]['simulated_transfer_interference_ms'] += duration * interference_ms
        remaining -= 1; totals['issued'] += 1; totals['early_retained' if early else 'forward_end_admissions'] += 1
        totals['h2d_bytes'] += 0 if reused else record_bytes; totals['d2h_bytes'] += record_bytes
        per_request[current_request]['h2d_bytes'] += 0 if reused else record_bytes
        per_request[current_request]['d2h_bytes'] += record_bytes
        totals['reused_staging'] += int(reused)
    for route, service in events:
        fwd=route['forward']
        if type(fwd) is not int: raise ValueError('invalid forward')
        if current_forward is not None and fwd < current_forward: raise ValueError('forward counter regressed')
        if fwd != current_forward and current_forward is not None and last_generation:
            for _ in range(2): submit(current_forward, last_ready, False)
        route_time=route.get('time_s')
        if previous_service_time is not None and isinstance(route_time,(int,float)):
            gap=max(0., (route_time-previous_service_time)*1000.)
            clock_ms+=gap; totals['observed_inter_event_gap_ms']+=gap
        if fwd != current_forward:
            current_forward=fwd; remaining=refill_budget
        previous_service_time=service.get('time_s')
        current_request=route['request']
        last_generation=bool(route.get('generation')); layer=route['layer']
        publish(fwd,layer)
        original=[(layer, e) for e in route['original']]
        if any(key not in initial_records for key in original): raise ValueError('route expert missing from initial snapshot')
        for key in original:
            scores[key]=score(key,fwd)+1; updated[key]=fwd
        # Each expert job may cover multiple rows. Do not credit executed substitutions as demand.
        executed={(layer, e) for e in route['executed']}
        if not executed <= initial_records: raise ValueError('executed expert missing from initial snapshot')
        observed_helper={(layer,e) for e in route.get('helper', [])}
        observed_primary={(layer,e) for e in route.get('primary', [])}
        unique=executed; helpers=len(unique & helper); hits=len(unique & primary)
        misses=len(unique)-hits-helpers
        totals['original_rows'] += len(original); totals['original_primary_hits'] += len(set(original) & primary); totals['primary_hits'] += hits; totals['helper_hits'] += helpers
        totals['host_jobs'] += misses; totals['attempted_rows'] += route['rows']
        for field in ('cpu_jobs', 'helper_jobs', 'demand_uploads', 'cpu_helper_lane_ms', 'helper_collect_ms'):
            value = service.get(field)
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value >= 0:
                totals['observed_'+field] += value
        ready=set()
        staging = route.get('staging', [])
        if len(staging) > 8: raise ValueError('trace exceeds modeled eight staging slots')
        last_free_slots = (sum(1 for slot in staging if len(slot) >= 5 and slot[1] < 0 and
                               not slot[2] and not slot[3] and not slot[4]) if staging else 8)
        for slot in staging:
            if len(slot) >= 5 and slot[0] >= 0 and slot[1] < 0 and not slot[2] and not slot[3] and not slot[4]:
                ready.add((slot[0]//1024,slot[0]%1024))
        last_ready=ready & host
        ordinary_pending=sum(1 for slot in staging if len(slot) >= 5 and slot[1] >= 0 and not slot[4])
        lease=route.get('helper_exchange', [])
        protected={(key//1024,key%1024) for key in lease[:2] if type(key) is int and key >= 0}
        protected |= helper
        if policy == 'priced' and last_generation:
            for _ in range(2): submit(fwd,{key for key in last_ready if key[0] != layer},True)
        observed=service['host_service_ms']; lane=service.get('cpu_helper_lane_ms')
        cpu=service.get('cpu_jobs'); helper_jobs=service.get('helper_jobs')
        if any(type(x) not in (int,float) or not math.isfinite(x) or x < 0 for x in (lane,cpu,helper_jobs)):
            raise ValueError('missing or invalid observed lane/job service fields')
        floor=max(0., observed-lane)
        observed_host_jobs=len(executed - observed_primary - observed_helper)
        alternative=max(0., lane+(misses-observed_host_jobs)*service_saved_ms)
        collect=service.get('helper_collect_ms', 0.)
        if type(collect) not in (int,float) or not math.isfinite(collect) or collect < 0:
            raise ValueError('invalid observed helper collect time')
        charged=transfer_charge; transfer_charge=0.
        estimated=floor+max(alternative,collect)+charged; clock_ms+=estimated
        totals['transfer_interference_ms'] += charged
        if last_generation:
            per_request[current_request]['observed_layer_ms'] += observed
            per_request[current_request]['estimated_layer_ms'] += estimated
            per_request[current_request]['layer_delta_ms'] += estimated-observed-charged
        totals['observed_layer_ms']+=observed; totals['estimated_layer_ms']+=estimated
        phase='generation' if last_generation else 'prefill'
        totals[phase+'_estimated_layer_ms']+=estimated; totals[phase+'_observed_layer_ms']+=observed
        rows.append({'request':route['request'],'event':route['event'],'forward':fwd,'layer':layer,
                     'generation':last_generation,'primary_hits':hits,'helper_hits':helpers,'host_jobs':misses,
                     'observed_lane_ms':lane,'observed_host_jobs':observed_host_jobs,
                     'simulated_host_jobs':misses,'nonservice_floor_ms':floor,'estimated_lane_ms':alternative,
                     'transfer_interference_ms':charged,'estimated_layer_ms':estimated})
        # Forward-end decisions use the post-layer service staging, not route-entry slots.
        service_staging=service.get('staging', staging)
        last_ready={(slot[0]//1024,slot[0]%1024) for slot in service_staging if len(slot)>=5 and
                    slot[0]>=0 and slot[1]<0 and not slot[2] and not slot[3] and not slot[4]} & host
        last_free_slots=(sum(1 for slot in service_staging if len(slot)>=5 and slot[1]<0 and
                             not slot[2] and not slot[3] and not slot[4]) if service_staging else 8)
        ordinary_pending=sum(1 for slot in service_staging if len(slot)>=5 and slot[1]>=0 and not slot[4])
        lease=service.get('helper_exchange', lease)
        protected |= {(key//1024,key%1024) for key in lease[:2] if type(key) is int and key>=0}
    if last_generation:
        for _ in range(2): submit(current_forward,last_ready,False)
    totals['pending_at_end']=len(pending)
    totals['unattributed_terminal_transfer_interference_ms']=transfer_charge
    totals['generation_estimated_layer_ms']+=transfer_charge
    return {'policy':policy,'primary_capacity':capacity,'helper_capacity':helper_count,'staging_slots':8,'exchange_slots':2,
            'counters':dict(totals),'per_request':{str(k):dict(v) for k,v in per_request.items()},'events':rows}


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('trace',type=Path); ap.add_argument('--decay',type=float,default=.95)
    ap.add_argument('--horizon',type=int,choices=(10,20,40),default=20); ap.add_argument('--margin-ms',type=float,choices=(0.,1.),default=0.)
    ap.add_argument('--refill-budget',type=int,choices=(2,8,16),default=8); ap.add_argument('--service-saved-ms',type=float,default=1.)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--observed-result',type=Path,help='matching trace-screen JSON for observed control counters')
    a=ap.parse_args()
    if not 0<a.decay<1 or a.horizon<1 or a.refill_budget not in (2,8,16) or a.margin_ms<0 or a.service_saved_ms<=0:
        ap.error('invalid policy parameters')
    requests,events=read_trace(a.trace)
    observed_control=None
    if not a.observed_result: ap.error('--observed-result is required for measured baseline')
    if a.observed_result:
        receipt=json.loads(a.observed_result.read_text())
        measured=receipt.get('requests',[])
        if len(measured) != len(requests): raise ValueError('observed result request count differs from trace')
        fields=('adaptive_issued','adaptive_published','adaptive_h2d_bytes','adaptive_d2h_bytes','adaptive_reused')
        finals=[(item.get('engine_stats') or {}).get('final') or {} for item in measured]
        if any(type(final.get(field)) not in (int,float) for final in finals for field in fields):
            raise ValueError('observed result lacks control adaptive counters')
        observed_control={'source':str(a.observed_result),'counters':{field:sum(final[field] for final in finals) for field in fields},
                          'pooled_generation':receipt.get('pooled_generation'),
                          'requests':[{'request':requests[i]['request'],'case':item.get('case'),
                                       'generation_interval':(item.get('engine_stats') or {}).get('generation_interval'),
                                       'counters':{field:finals[i][field] for field in fields}} for i,item in enumerate(measured)]}
    estimates=[]
    for label,h2d,d2h,interference,saving_factor in [('low',1.,1.,.05,.5),('central',2.,2.,.15,1.),('high',4.,4.,.35,1.5)]:
        priced=replay(requests,events,'priced',decay=a.decay,horizon=a.horizon,margin=a.margin_ms,
                      refill_budget=a.refill_budget,h2d_ms=h2d,d2h_ms=d2h,interference_ms=interference,
                      service_saved_ms=a.service_saved_ms*saving_factor)
        components=[]
        for base in observed_control['requests']:
            sim=priced['per_request'].get(str(base['request']), {})
            measured=base['generation_interval'] or {}
            counters=base['counters']; record_bytes=requests[0]['record_bytes']
            observed_charge=(counters['adaptive_h2d_bytes']/record_bytes*h2d +
                             counters['adaptive_d2h_bytes']/record_bytes*d2h)*interference
            simulated_delta=sim.get('layer_delta_ms',0.)+sim.get('simulated_transfer_interference_ms',0.)
            delta=simulated_delta-observed_charge
            components.append({'request':base['request'],'case':base['case'],
                               'observed_generation_ms':measured.get('interval_ms'),
                               'observed_generation_tokens':measured.get('tokens'),
                               'simulated_affected_layer_delta_ms_including_transfer':simulated_delta,
                               'estimated_observed_transfer_interference_ms':observed_charge,
                               'estimated_generation_delta_ms':delta,
                               'estimated_generation_ms':measured['interval_ms']+delta if measured.get('interval_ms') is not None else None})
        estimates.append({'range':label,'h2d_ms':h2d,'d2h_ms':d2h,'exposed_transfer_fraction':interference,
                          'service_saved_ms':a.service_saved_ms*saving_factor,'per_request':components,
                          'estimated_generation_delta_ms':sum(c['estimated_generation_delta_ms'] for c in components),
                          'priced':priced})
    out={'observed_control':observed_control,
         'interpretation':'Measured legacy generation is baseline. Estimated affected-layer deltas are counterfactual fixed-trace scenarios, not observed engine speed or proof of a gain.',
         'trace':str(a.trace),'requests':len(requests),'route_events':len(events),
         'controls':{'decay':a.decay,'horizon':a.horizon,'margin_ms':a.margin_ms,'refill_budget_per_forward':a.refill_budget,
                     'service_saved_ms_per_unique_host_job':a.service_saved_ms},
         'assumptions':['Fixed trace: executed choices do not respond to the simulated placement; all original attempted rows are demand, executed IDs determine simulated service.',
           'Helper placement is frozen to the initial request roster throughout counterfactual simulation; observed helper changes only affect the measured baseline.',
           'Observed ready staging slots and free-slot count are hypothetical opportunities; pending trace transfers and service costs are not recomputed.',
           'Pending exchanges serialize their H2D and D2H estimates, advance through observed inter-event gaps, publish at layer boundaries excluding the active incoming or victim layer, and protect their slots until modeled deadlines.',
           'Nonservice floor is max(0, observed host wall minus composite lane); add max(adjusted lane, observed helper collect) and exposed transfer interference. Lane adjustment uses change in executed HOST-service identities including staged records, not CPU-job counters. Lane is composite, not pure CPU; this is not a bound or complete-engine time.',
           'Transfer costs are illustrative ranges, not measured hardware bandwidth; candidate gains do not establish speedup.',
           'At most budget minus one early admissions per forward, with one budget unit held for forward-end refill. Eight staging frames and two concurrent exchanges are capacity assumptions; at most four ordinary plus simulated pending transfers and at least four unreserved staging frames; initial primary capacity is preserved.',
           'Observed transfer interference is estimated from measured generation bytes and subtracted from simulated transfer interference; these are uncertain net costs, not measured isolated overhead.',
           'Per-request measured full generation timing includes draft and head work absent from layer estimates; add estimated affected-layer delta only.'], 'estimates':estimates}
    with a.output.open('x') as f: json.dump(out,f,indent=2);f.write('\n')

if __name__=='__main__': main()
