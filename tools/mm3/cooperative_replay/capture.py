"""Strict capture contract and observation-fed decision/state reproduction."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from .policy import Demand, NativePolicy


@dataclass
class Capture:
    path: Path
    sha256: str
    request_id: str
    events: list[dict]
    initial: dict
    final: dict

    @property
    def routes(self): return [e for e in self.events if e['type']=='routes']


def read_capture(path, *, max_bytes=512*1024*1024, max_events=2_000_000, continuation=False):
    path = Path(path)
    if path.stat().st_size > max_bytes: raise ValueError('bounded capture byte limit exceeded')
    raw = path.read_bytes();rows = [json.loads(line) for line in raw.splitlines()]
    if not rows or len(rows)>max_events or path.with_suffix('.partial').exists():
        raise ValueError('empty, partial, or oversized capture')
    footer = rows.pop()
    if (footer.get('type')!='capture_footer' or footer.get('dropped_events')!=0 or
        footer.get('recorded_events')!=len(rows) or footer.get('attempted_events')!=len(rows)):
        raise ValueError('capture footer missing, dropped events, or incomplete interval')
    if [r.get('event_seq') for r in rows]!=list(range(1,len(rows)+1)):
        raise ValueError('capture sequence gap')
    identity = footer.get('request_id')
    if not identity or any(r.get('request_id')!=identity for r in rows):
        raise ValueError('request identities differ')
    events = []
    for row in rows:
        if row['type']=='residency':
            row = {**row, **row['record']};row.pop('record')
        events.append(row)
    if any(r['type']=='request_error' or (r['type']=='capture_end' and r.get('status') in ('error','cancel')) for r in events):
        raise ValueError('failed/cancelled capture is not a complete request interval')
    for row in events:
        if row['type']=='request_end' and row.get('emitted')!=sum(e['type']=='token_output' for e in events):
            raise ValueError('emitted-token coverage differs from completion')
    starts = [r for r in events if r['type']=='request']
    ends = [r for r in events if r['type']=='end']
    if len(starts)!=1 or len(ends)!=1: raise ValueError('complete native request snapshot/footer required')
    start, end = starts[0], ends[0]
    if start.get('replay_schema')!=1 or not start.get('replay_supported') or 'replay_state' not in end:
        raise ValueError('capture lacks complete replay state or uses an unsupported engine path')
    if not start.get('batch_staging_calibration'): raise ValueError('initial policy prices missing')
    if start['host_free_layer_count'] or start['compressed_host'] or start['policy']['legacy']:
        raise ValueError('unsupported host-free/compressed/legacy path')
    if not start['primary_adaptive'] or any(e['type'].startswith(('promotion_','prefetch_')) for e in events):
        raise ValueError('optional promotion/prefetch path is outside this bounded cooperative replay')
    # A fresh request may retain ordinary staging, but initial in-flight work
    # requires a capture starting before its submission. Never invent residuals.
    if not continuation and (start['replay_state']['pending_sources'] or start['helper_exchange'][0]>=0 or
        start['replay_state']['helper_queue_pending'] or
        any(s[1]>=0 or s[2] or s[3] or s[4] for s in start['staging'])):
        raise ValueError('initial work is in flight; capture an earlier complete request boundary')
    layers = [l['layer'] for l in start['layers']]
    active = None;seen=[];pending=None
    for row in events:
        typ = row['type']
        if typ=='boundary' and row['kind']=='forward_begin':
            if active is not None: raise ValueError('nested target batch')
            active=row;seen=[]
        elif typ=='routes':
            if active is None or row['forward']!=active['forward'] or pending is not None:
                raise ValueError('route outside ordered target batch')
            if row['rows']!=active['rows'] or row['generation']!=active['generation']:
                raise ValueError('target batch shape/phase mismatch')
            for name in ('original','executed'):
                ids=row[name]
                if len(ids)!=4*row['rows'] or any(type(e) is not int or not 0<=e<start['expert_count'] for e in ids):
                    raise ValueError('incomplete evaluated routing rows')
                if any(len(set(ids[i:i+4]))!=4 for i in range(0,len(ids),4)):
                    raise ValueError('top-four IDs must be distinct per evaluated row')
            seen.append(row['layer']);pending=row
        elif typ=='service':
            if pending is None or row['event']!=pending['event']: raise ValueError('unpaired service observation')
            pending=None
        elif typ=='boundary' and row['kind']=='forward_end':
            if active is None or row['forward']!=active['forward'] or seen!=layers or pending:
                raise ValueError('incomplete target layers (including speculative rows)')
            active=None
    if active is not None or pending or not any(r['generation'] for r in events if r['type']=='routes'):
        raise ValueError('incomplete or empty evaluated generation')
    return Capture(path,hashlib.sha256(raw).hexdigest(),identity,events,start,end)


def read_history(paths):
    captures=[]
    for path in paths:
        c=read_capture(path,continuation=bool(captures))
        if captures:
            before=captures[-1].final;after=c.initial
            if before['forward']!=after['forward'] or before['replay_state']['row_clock']!=after['replay_state']['row_clock']:
                raise ValueError('uncaptured target work between requests')
            for kind in ('primary','helper','host'):
                if roster(before,kind)!=roster(after,kind):raise ValueError('uncaptured ownership change between requests')
            if before['staging']!=after['staging'] or before['exchanges']!=after['exchanges'] or before['helper_exchange']!=after['helper_exchange']:
                raise ValueError('uncaptured availability/control change between requests')
            for key in ('scores','primary_credit','helper_credit','pending_sources'):
                if before['replay_state'][key]!=after['replay_state'][key]:raise ValueError('uncaptured learned-state change between requests')
        captures.append(c)
    if not captures:raise ValueError('empty history')
    return captures


def roster(snapshot, kind):
    if kind=='helper': return {l['layer']*1024+e for l in snapshot['layers'] for e in l[kind]}
    return {l['layer']*1024+e:frame for l in snapshot['layers'] for e,frame in l[kind+'_frames']}


def reproduce(capture: Capture, bridge=None, previous=None):
    """Stage 1 only: observed availability is permitted; timing is not predicted."""
    bridge = bridge or NativePolicy()
    initial = capture.initial;state = initial['replay_state'];policy = initial['policy']
    demand = Demand(bridge,row=state['row_clock'],block_rows=state['block_rows'],
                    scores={k:(v,last) for k,v,last in state['scores']})
    primary, host, helper = roster(initial,'primary'), roster(initial,'host'), roster(initial,'helper')
    helper_slots={li*1024+expert:slot for li,expert,slot in initial['helper_frames']}
    slots = [list(s) for s in initial['staging']]
    frames = set(primary.values()) | {s[6] for s in slots}
    if len(frames)!=len(primary)+len(slots): raise ValueError('initial GPU allocations alias')
    if set(primary)&set(host) or set(primary)&helper or set(host)&helper:
        raise ValueError('initial canonical ownership overlap')
    prices = initial['batch_staging_calibration']
    pc,pr=state['primary_credit'],state['primary_credit_row']
    hc,hr=state['helper_credit'],state['helper_credit_row']
    pending={}
    for a,b,slot,submitted,source_host,source_gpu,landing,target_host in state['pending_sources']:
        if not submitted:raise ValueError('initial pending exchange is not submitted')
        pending[a]=dict(incoming=a,victim=b,landing_slot=slot,source_host_frame=source_host,
            victim_frame=source_gpu,landing_frame=landing,target_host_frame=target_host)
    helper_pending=None
    helper_withdrawn,helper_published=map(bool,initial['helper_exchange'][2:4])
    if initial['helper_exchange'][0]>=0:
        helper_pending=(previous or {}).get('terminal_transactions',{}).get('helper')
        if not helper_pending or [helper_pending['incoming'],helper_pending['victim']]!=initial['helper_exchange'][:2]:
            raise ValueError('initial helper transaction lacks captured preceding submission')
    choices={};counts=Counter();checks=Counter();last_route=None
    def equal(a,b,what,tol=2e-5):
        if not math.isclose(a,b,rel_tol=tol,abs_tol=2e-6): raise ValueError(f'{what}: {a} != {b}')
    def ownership():
        if set(primary)&set(host) or set(primary)&helper or set(host)&helper:
            raise ValueError('canonical ownership overlap')
        if set(primary.values())|{s[6] for s in slots} != frames or len(primary)+len(slots)!=len(frames):
            raise ValueError('lost or invented GPU frame')
    for e in capture.events:
        typ=e['type']
        if typ=='batch_staging_calibration': prices=e['calibration']
        elif typ=='boundary':
            if e['kind']=='forward_begin':
                expected=demand.row+(e['rows'] if e['generation'] else 0)
                if e['row_clock']!=expected: raise ValueError('generation-row clock mismatch')
                demand.row=expected;demand.block_rows=e['rows'] if e['generation'] else demand.block_rows
            # Completion/availability observations are the stage-1 input.
            # Frame identities may only change through publications below.
            if [s[6] for s in e['staging']]!=[s[6] for s in slots]: raise ValueError('unexplained working-frame rotation')
            slots=[list(s) for s in e['staging']]
        elif typ=='routes':
            demand.observe(e['layer'],e['original'],e['rows'],e['generation']);last_route=e
            checks['original_selections']+=len(e['original']) if e['generation'] else 0
            checks['evaluated_layer_rows']+=e['rows'] if e['generation'] else 0
            base=e['layer']*1024
            if {base+x for x in e['primary']}!={k for k in primary if k//1024==e['layer']}:
                raise ValueError('route primary ownership differs')
            if {base+x for x in e['helper']}!={k for k in helper if k//1024==e['layer']}:
                raise ValueError('route helper ownership differs')
            slots=[list(s) for s in e['staging']]
        elif typ=='batch_staging_decision':
            if 'uses' not in e or last_route is None: raise ValueError('missing planner decision inputs')
            actual=Counter(last_route['executed'])
            if e['uses'] != [actual[i] for i in range(128)]: raise ValueError('staging work differs from evaluated rows')
            selected,predicted=bridge.stage(e['uses'],e['cpu'],e['gpu'],e['helper'],e['candidates'],
                e['max_new'],e['rows'],prices,e['cpu_price_scale'])
            if selected!=e['selected']: raise ValueError('production staging decision differs')
            for x,name in zip(predicted,('predicted_cpu_ms','predicted_gpu_ms','predicted_helper_ms','critical_before_ms','critical_after_ms')):
                equal(x,e[name],name,1e-9)
            checks['staging_decisions']+=1
        elif typ=='policy_decision':
            if e['row_clock']!=demand.row: raise ValueError('policy sees wrong history prefix')
            incoming=[];victims=[]
            for name,target in (('incoming',incoming),('victims',victims)):
                for key,value,slot in e[name]:
                    observed=demand.score(key);equal(observed,value,'demand score')
                    target.append((key,observed,slot))
                    if name=='incoming' and key not in host: raise ValueError('non-HOST admission candidate')
                    if name=='victims' and key not in (helper if e['kind']=='helper' else primary):
                        raise ValueError('victim not resident')
            if e['kind']=='primary':
                pc=bridge.lib.mm3_replay_budget(pc,pr,demand.row,policy['max_exchanges']);pr=demand.row
                equal(pc,e['primary_credit'],'primary budget')
                # kDecay is float32 in the incumbent implementation.
                import ctypes
                benefit=(1-float(ctypes.c_float(.95).value)**policy['horizon'])*policy['service_saved_ms']
                equal(benefit,e['benefit_per_score'],'incumbent benefit',1e-8)
            else:
                hc=bridge.lib.mm3_replay_budget(hc,hr,demand.row,1.0);hr=demand.row
                equal(hc,e['helper_credit'],'helper budget');benefit=0
            decision=bridge.choose(e['kind'],incoming,victims,[benefit,policy['h2d_ms'],policy['d2h_ms'],policy['interference_ms'],policy['margin_ms']])
            if decision[:3]!=e['choice'][:3]: raise ValueError(f"{e['kind']} choice differs: {decision} vs {e['choice']}")
            equal(decision[3],e['choice'][3],'admission value',1e-4)
            choices[e['kind']]=decision;checks[e['kind']+'_decisions']+=1
        elif typ=='primary_exchange_issue':
            c=choices.get('primary')
            if c is None or c[:2]!=[e['incoming'],e['victim']]: raise ValueError('issue without reproduced primary choice')
            if e['incoming'] in pending: raise ValueError('duplicate pending exchange')
            if e['landing_frame'] not in frames or e['victim_frame']!=primary[e['victim']]:
                raise ValueError('exchange invented GPU capacity')
            if e['source_host_frame']!=host[e['incoming']]: raise ValueError('exchange lost source ownership')
            if e['h2d_bytes']!=(0 if e['reuse'] else initial['record_bytes']) or e['d2h_bytes']!=(0 if policy['primary_backing'] else initial['record_bytes']):
                raise ValueError('primary transfer byte mismatch')
            pending[e['incoming']]=e;pc=max(0,pc-1)
            counts.update(primary_issued=1,h2d_bytes=e['h2d_bytes'],d2h_bytes=e['d2h_bytes'],reused=e['reuse'])
        elif typ=='primary_exchange_publish':
            x=pending.pop(e['incoming']);a,b=x['incoming'],x['victim'];slot=x['landing_slot']
            if e['resident_frame']!=x['landing_frame'] or e['working_frame']!=primary[b]: raise ValueError('invalid publication alias')
            primary.pop(b);primary[a]=x['landing_frame'];host.pop(a);host[b]=x['target_host_frame']
            slots[slot][6]=x['victim_frame'];slots[slot][0]=b;slots[slot][4]=0
            counts['primary_published']+=1;ownership()
        elif typ=='helper_exchange_issue':
            if choices.get('helper',[])[:2]!=[e['incoming'],e['victim']] or helper_pending:
                raise ValueError('unreproduced helper issue')
            if helper_slots[e['victim']]!=e['helper_slot']:raise ValueError('helper exchange frame differs')
            if e['planned_h2d_bytes']!=initial['record_bytes']-256 or e['planned_d2h_bytes']!=initial['record_bytes']-256:
                raise ValueError('helper payload byte mismatch')
            helper_pending=e;helper_withdrawn=False;helper_published=False;hc=max(0,hc-1);counts['helper_issued']+=1
            counts['helper_h2d_bytes_planned']+=e['planned_h2d_bytes'];counts['helper_d2h_bytes_planned']+=e['planned_d2h_bytes']
        elif typ=='helper_exchange_withdraw':
            x=helper_pending
            if not x or [e['incoming'],e['victim']]!=[x['incoming'],x['victim']]: raise ValueError('helper withdrawal identity')
            helper.remove(x['victim']);helper_slots.pop(x['victim']);host[x['victim']]=x['outgoing_buffer'];helper_withdrawn=True;ownership()
        elif typ=='helper_exchange_publish':
            x=helper_pending;helper.add(x['incoming']);helper_slots[x['incoming']]=x['helper_slot'];host.pop(x['incoming']);helper_published=True;ownership()
        elif typ=='helper_exchange_retire':
            x=helper_pending;host[x['victim']]=x['host_frame'];helper_pending=None;helper_withdrawn=False;helper_published=False;ownership()
        elif typ=='service': slots=[list(s) for s in e['staging']]
    final=capture.final
    if primary!=roster(final,'primary') or host!=roster(final,'host') or helper!=roster(final,'helper'):
        raise ValueError('final ownership differs')
    if 'helper_frames' in final and helper_slots!={li*1024+expert:slot for li,expert,slot in final['helper_frames']}:
        raise ValueError('final helper frame mapping differs')
    fs=final['replay_state']
    expected_pending=sorted([[x['incoming'],x['victim'],x['landing_slot'],1,x['source_host_frame'],
        x['victim_frame'],x['landing_frame'],x['target_host_frame']] for x in pending.values()])
    if expected_pending!=sorted(fs['pending_sources']):raise ValueError('final primary transactions differ')
    expected_helper=([helper_pending['incoming'],helper_pending['victim']] if helper_pending else [-1,-1])+[int(helper_withdrawn),int(helper_published)]
    if expected_helper!=final['helper_exchange']:raise ValueError('final helper transaction differs')
    if demand.row!=fs['row_clock']: raise ValueError('final demand clock differs')
    pc=bridge.lib.mm3_replay_budget(pc,pr,fs['primary_credit_row'],policy['max_exchanges'])
    hc=bridge.lib.mm3_replay_budget(hc,hr,fs['helper_credit_row'],1.0)
    equal(pc,fs['primary_credit'],'final primary credit');equal(hc,fs['helper_credit'],'final helper credit')
    # Normalized comparison tolerates lazy read order, not different demand.
    for key,value,last in fs['scores']:
        equal(demand.score(key),bridge.lib.mm3_replay_demand(value,last,demand.row,0),'final normalized demand')
    counter_start=initial
    if 'adaptive_issued' not in initial:
        # The first extended capture omitted cumulative starting counters.
        # A complete prefill service before any exchange provides the same
        # boundary observation; never assume these lifetime counters are zero.
        for e in capture.events:
            if e['type'] in ('primary_exchange_issue','primary_exchange_publish'):
                raise ValueError('missing initial counters before exchanged work')
            if e['type']=='service':counter_start=e;break
        if counter_start is initial:raise ValueError('no initial counter observation')
    for field,count in (('adaptive_issued','primary_issued'),('adaptive_published','primary_published'),
                        ('adaptive_reused','reused'),('adaptive_h2d_bytes','h2d_bytes'),('adaptive_d2h_bytes','d2h_bytes')):
        if final[field]-counter_start[field]!=counts[count]: raise ValueError(f'final {field} differs')
    return dict(stage=1,status='pass',request_id=capture.request_id,source=str(capture.path),sha256=capture.sha256,
                checks=dict(checks),actions=dict(counts),pending_primary=len(pending),
                terminal_transactions=dict(primary=list(pending.values()),helper=helper_pending),
                counter_baseline='request snapshot' if counter_start is initial else 'first pre-exchange service observation',
                timing_validated=False,scope='recorded availability; shared native policy; original-demand, action, bytes and final ownership reproduction')
