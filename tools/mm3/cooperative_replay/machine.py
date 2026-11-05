"""Bounded model of the native single-primary, worker-helper cooperative path."""
from __future__ import annotations
from collections import Counter
from dataclasses import dataclass
import ctypes
from .capture import roster
from .execution import Scheduler
from .policy import Demand, NativePolicy, PolicyConfig


@dataclass
class Frame:
    address: int
    owner: int = -1
    used: int = -1
    reader: int | None = None
    reserved: bool = False


@dataclass
class Exchange:
    incoming: int
    victim: int
    slot: int
    source_host: int
    target_host: int
    source_gpu: int
    completion: int
    descriptor: int
    reused: bool


class Machine:
    def __init__(self, snapshot, physical, config=None, *, bridge=None, variant='central', backing=None):
        self.initial=snapshot;self.physical=physical;self.config=config or PolicyConfig()
        self.bridge=bridge or NativePolicy();self.scheduler=Scheduler(physical.contention,variant)
        self.primary=roster(snapshot,'primary');self.host=roster(snapshot,'host')
        self.primary_order={l['layer']:[l['layer']*1024+e for e in l['primary']] for l in snapshot['layers']}
        self.helper={li*1024+e:slot for li,e,slot in snapshot['helper_frames']}
        self.slots=[Frame(s[6],s[0],s[5]) for s in snapshot['staging']]
        self.gpu_frames=set(self.primary.values())|{s.address for s in self.slots}
        self.primary_capacity=len(self.primary);self.helper_capacity=len(self.helper)
        self.record_bytes=snapshot['record_bytes'];self.dimension=snapshot['dimension']
        self.backing=snapshot['policy']['primary_backing'] if backing is None else backing
        self.backing_frames={l['layer']*1024+e:p for l in snapshot['layers'] for e,p in l.get('primary_backing_frames',[])}
        # A backing intervention explicitly allocates a full HOST bank. It never
        # changes GPU capacity. Synthetic addresses identify the modeled extra
        # allocation; callers must independently check real HOST headroom.
        if self.backing and not self.backing_frames:
            self.backing_frames={k:-(i+1)*self.record_bytes for i,k in enumerate(sorted(self.primary))}
        if not self.backing:self.backing_frames={}
        state=snapshot['replay_state']
        self.spares=list(state['host_spares']);self.helper_buffers=list(state['helper_buffers'])
        self.demand=Demand(self.bridge,state['row_clock'],state['block_rows'],
                           {k:(v,r) for k,v,r in state['scores']},decay=self.config.decay)
        self.primary_credit=state['primary_credit'];self.primary_credit_row=state['primary_credit_row']
        self.helper_credit=state['helper_credit'];self.helper_credit_row=state['helper_credit_row']
        self.exchanges: list[Exchange]=[];self.helper_exchange=None
        self.prices=snapshot['batch_staging_calibration']
        self.forward=snapshot['forward'];self.generation=False
        self.counts=Counter();self.actions=[];self.boundaries=[];self.layers=[];self.requests=[]
        self.previous_gpu=None;self.previous_host=None
        self.assert_ownership()

    def assert_ownership(self):
        p,h,a=set(self.primary),set(self.host),set(self.helper)
        if p&h or p&a or h&a: raise ValueError('simulated canonical ownership overlaps')
        if len(p)!=self.primary_capacity: raise ValueError('primary capacity changed')
        if len(a) not in (self.helper_capacity,self.helper_capacity-1): raise ValueError('helper capacity changed')
        physical=list(self.primary.values())+[s.address for s in self.slots]
        if len(set(physical))!=len(physical) or set(physical)!=self.gpu_frames:
            raise ValueError('exchange descriptor invented/aliased a weight frame')
        if len(self.exchanges)>len(self.spares): raise ValueError('exchange descriptor capacity exceeded')
        if sum(s.reserved for s in self.slots)!=len(self.exchanges): raise ValueError('exchange must reserve an existing working frame')
        if self.backing and set(self.backing_frames)!=p: raise ValueError('backing ledger incomplete')

    def cost(self,kind,**kwargs): return self.physical.estimate(kind,**kwargs)
    def op(self,kind,resources=(),dependencies=(),**kwargs):
        label=kwargs.pop('label',f'f{self.forward}')
        queue=kwargs.pop('queue',None);byte_count=kwargs.get('bytes',0)
        return self.scheduler.submit(kind,self.cost(kind,**kwargs),resources,dependencies,
                                     label=label,queue=queue,bytes=byte_count)
    def busy(self,frame): return frame.reader is not None and not self.scheduler.done(frame.reader)
    def leased(self,key):
        return any(key in (x.incoming,x.victim) for x in self.exchanges) or (
            self.helper_exchange and key in (self.helper_exchange['incoming'],self.helper_exchange['victim']))

    def publish(self,protected_layer=-1):
        s=self.scheduler
        for x in list(self.exchanges):
            if (not s.done(x.completion) or protected_layer in (x.incoming//1024,x.victim//1024) or
                any(f.owner==x.incoming and self.busy(f) for f in self.slots)):
                continue
            if self.primary.get(x.victim)!=x.source_gpu or self.host.get(x.incoming)!=x.source_host:
                raise ValueError('pending primary source changed')
            frame=self.slots[x.slot]
            self.primary.pop(x.victim);self.primary[x.incoming]=frame.address
            old=self.primary_order[x.victim//1024];index=old.index(x.victim);old[index]=old[-1];old.pop()
            self.primary_order[x.incoming//1024].append(x.incoming)
            self.host.pop(x.incoming);self.host[x.victim]=x.target_host
            if self.backing:
                self.backing_frames.pop(x.victim);self.backing_frames[x.incoming]=x.source_host
            else:self.spares[x.descriptor]=x.source_host
            frame.address=x.source_gpu;frame.owner=x.victim;frame.used=self.forward-1;frame.reserved=False
            for i,f in enumerate(self.slots):
                if i!=x.slot and f.owner==x.incoming:f.owner=-1
            self.exchanges.remove(x);self.counts['primary_published']+=1
            self.actions.append(dict(kind='primary_publish',time_ms=s.now,incoming=x.incoming,victim=x.victim,slot=x.slot))
        self.helper_progress()
        self.assert_ownership()

    def helper_progress(self):
        x=self.helper_exchange;s=self.scheduler
        if not x or not s.done(x['completion']):return
        # Every call is an existing joined boundary; never wait on optional work.
        if x['phase']=='prepared':
            self.helper.pop(x['victim']);self.host[x['victim']]=self.helper_buffers[1]
            x['completion']=self.op('h_weight_h2d',('h_h2d',),bytes=self.record_bytes-256,queue='h_exchange')
            x['phase']='uploaded'
        elif x['phase']=='uploaded':
            if any(f.owner==x['incoming'] and self.busy(f) for f in self.slots):return
            self.helper[x['incoming']]=x['slot'];self.host.pop(x['incoming'])
            for f in self.slots:
                if f.owner==x['incoming']:f.owner=-1
            x['completion']=self.op('host_weight_copy',('helper_exchange_worker',),bytes=self.record_bytes-256,queue='h_exchange')
            x['phase']='recycled'
        else:
            if any(f.owner==x['victim'] and self.busy(f) for f in self.slots):return
            self.host[x['victim']]=x['source_host'];self.helper_exchange=None
            self.counts['helper_published']+=1
        self.actions.append(dict(kind='helper_'+x['phase'],time_ms=s.now,incoming=x['incoming'],victim=x['victim']))

    def admission(self,protected_layer=-1):
        if not self.generation:return
        s=self.scheduler;c=self.config;d=self.demand;p=self.initial['policy']
        self.primary_credit=self.bridge.lib.mm3_replay_budget(self.primary_credit,self.primary_credit_row,d.row,c.refill_budget)
        self.primary_credit_row=d.row
        # Policy price may differ; the physical model is never read here.
        saved=p['service_saved_ms'] if p.get('configured_service_saved_ms',0)>0 else max(.05,c.cpu_scale*self.prices['cpu_extra_ms'][1]-self.prices['gpu_extra_ms'][1])
        benefit=(1-float(ctypes.c_float(.95).value)**c.horizon)*saved
        h2d=p['configured_h2d_ms'] or self.prices['h2d_ms']
        d2h=0 if self.backing else (p['configured_d2h_ms'] or self.prices['d2h_ms'])
        for descriptor in range(len(self.spares)):
            if self.primary_credit+1e-9<(2 if protected_layer>=0 else 1):break
            if any(x.descriptor==descriptor for x in self.exchanges):continue
            ready=[]
            for i,f in enumerate(self.slots):
                if f.owner>=0 and f.owner//1024!=protected_layer and not f.reserved and not self.busy(f) and not self.leased(f.owner) and f.owner in self.host:
                    ready.append((f.owner,d.score(f.owner),i))
            if protected_layer>=0 and not any(v*benefit>d2h+p['interference_ms']+c.margin_ms for _,v,_ in ready):break
            victims=[(key,d.score(key),-1) for layer in sorted(self.primary_order) for key in self.primary_order[layer] if not self.leased(key)]
            incoming=list(ready);ready_keys={k for k,_,_ in ready}
            if protected_layer<0 and c.extra_refill:
                incoming += [(key,d.score(key),-1) for key in sorted(self.host) if key not in ready_keys and not self.leased(key)]
            choice=self.bridge.choose('primary',incoming,victims,[benefit,h2d,d2h,p['interference_ms'],c.margin_ms])
            a,b,slot,value=choice
            if a<0:break
            reuse=slot>=0
            if slot<0:
                available=[i for i,f in enumerate(self.slots) if not f.reserved and not self.busy(f)]
                if not available:break
                slot=min(available,key=lambda i:(self.slots[i].used,i))
            frame=self.slots[slot];frame.reserved=True;frame.owner=-1
            deps=[]
            if not reuse:
                deps=[self.op('p_weight_h2d',('p_h2d',),bytes=self.record_bytes,queue='p_optional')]
            if not self.backing:
                deps=[self.op('p_weight_d2h',('p_d2h',),deps,bytes=self.record_bytes,queue='p_optional')]
            done=s.submit('publication_ready',self._zero(),dependencies=deps,queue='p_optional')
            self.exchanges.append(Exchange(a,b,slot,self.host[a],self.backing_frames[b] if self.backing else self.spares[descriptor],self.primary[b],done,descriptor,reuse))
            self.primary_credit=max(0,self.primary_credit-1)
            self.counts.update(primary_issued=1,reused=int(reuse),optional_h2d_bytes=0 if reuse else self.record_bytes,
                               optional_d2h_bytes=0 if self.backing else self.record_bytes)
            self.actions.append(dict(kind='primary_issue',time_ms=s.now,incoming=a,victim=b,slot=slot,reuse=reuse,value_ms=value))
        self.assert_ownership()

    @staticmethod
    def _zero():
        from .execution import Estimate
        return Estimate(0,0,0)

    def helper_admission(self):
        if not self.generation or not self.config.helper_live or self.helper_exchange:return
        d=self.demand
        self.helper_credit=self.bridge.lib.mm3_replay_budget(self.helper_credit,self.helper_credit_row,d.row,1.0);self.helper_credit_row=d.row
        if self.helper_credit+1e-9<1:return
        busy={f.owner for f in self.slots if self.busy(f)}
        inc=[(k,d.score(k),-1) for k in sorted(self.host) if not self.leased(k) and k not in busy]
        vic=[(k,d.score(k),-1) for k in sorted(self.helper) if not self.leased(k) and k not in busy]
        a,b,_,_=self.bridge.choose('helper',inc,vic,[0]*5)
        if a<0:return
        copied=self.op('host_weight_copy',('helper_exchange_worker',),bytes=self.record_bytes-256,queue='h_exchange')
        downloaded=self.op('h_weight_d2h',('h_d2h',),(copied,),bytes=self.record_bytes-256,queue='h_exchange')
        self.helper_exchange=dict(incoming=a,victim=b,slot=self.helper[b],source_host=self.host[a],phase='prepared',completion=downloaded)
        self.helper_credit=max(0,self.helper_credit-1)
        self.counts['helper_issued']+=1
        self.actions.append(dict(kind='helper_issue',time_ms=self.scheduler.now,incoming=a,victim=b))

    def serve_layer(self,route):
        s=self.scheduler;li=route['layer'];rows=route['rows'];context=max(route['positions'])+1
        label=f'f{self.forward}/l{li}'
        self.helper_progress()
        deps=[] if self.previous_gpu is None else [self.previous_gpu]
        # Dense, attention, router and shared work precede the joined route
        # snapshot. Its independently measured table includes the side stream.
        host_pre=self.op('host_pre',('host',),rows=rows,label=label)
        pre=self.op('layer_pre',('p_compute',),deps,rows=rows,context=context,label=label,queue='p_main')
        route_copy=self.op('route_readback',('p_d2h',),(pre,host_pre),rows=rows,label=label,queue='p_main')
        s.until(route_copy)
        for f in self.slots:
            if not self.busy(f):f.reader=None;f.used=self.forward-1
        self.publish(li);self.admission(li)
        self.demand.observe(li,route['original'],rows,self.generation)
        uses=[0]*128
        for e in route['executed']:uses[e]+=1
        cpu=[];gpu=[];helper=[];ready={}
        for e,n in enumerate(uses):
            if not n:continue
            key=li*1024+e
            if key in self.helper:helper.append(e)
            elif key in self.primary:gpu.append(e)
            else:
                slots=[i for i,f in enumerate(self.slots) if f.owner==key and not f.reserved]
                if slots:gpu.append(e);ready[e]=slots[-1]
                elif key in self.host:cpu.append(e)
                else:raise ValueError('executed expert has no service owner')
        unavailable=set(ready.values())
        available=[i for i,f in enumerate(self.slots) if not f.reserved and not self.busy(f) and i not in unavailable]
        selected,predicted=self.bridge.stage(uses,cpu,gpu,helper,cpu,min(len(available),len(self.slots)//2,self.config.stage_limit),rows,self.prices,self.config.cpu_scale)
        plan=self.op('host_plan',('host',),rows=rows,groups=[uses[e] for e in range(128) if uses[e]],label=label)
        s.until(plan)
        for e in selected:
            i=min(available,key=lambda x:(self.slots[x].used,x));available.remove(i)
            f=self.slots[i];f.owner=li*1024+e;f.used=self.forward;ready[e]=i
            self.op('p_weight_h2d',('p_h2d',),(plan,),bytes=self.record_bytes,queue='p_main',label=label)
            self.counts['demand_h2d_bytes']+=self.record_bytes
        gpu+=selected;cpu=[e for e in cpu if e not in selected]
        plan_copy=self.op('plan_upload',('p_h2d',),(plan,),rows=rows,queue='p_main',label=label)
        cpu_dep=plan
        helper_end=None
        if helper:
            prep=self.op('helper_prepare',('host',),(plan,),rows=rows,groups=[uses[e] for e in helper],label=label)
            cpu_dep=prep
            dispatch=self.op('helper_dispatch',('helper_worker',),(prep,),rows=rows,groups=[uses[e] for e in helper],label=label)
            hin=self.op('h_activation',('h_h2d',),(dispatch,),rows=rows,groups=[uses[e] for e in helper],queue='h_compute',label=label)
            compute=self.op('helper',('h_compute','helper_worker'),(hin,),rows=rows,groups=[uses[e] for e in helper],queue='h_compute',label=label)
            helper_end=self.op('h_result',('h_d2h',),(compute,),rows=rows,groups=[uses[e] for e in helper],queue='h_compute',label=label)
        submit=self.op('primary_submit',('host',),(cpu_dep,),rows=rows,groups=[uses[e] for e in gpu],label=label)
        primary=self.op('primary',('p_compute',),(plan_copy,cpu_dep),rows=rows,groups=[uses[e] for e in gpu],queue='p_main',label=label)
        cpu_end=submit
        if cpu:
            cpu_end=self.op('cpu',('cpu','host'),(submit,),rows=rows,groups=[uses[e] for e in cpu],workers=16,working_set='rotating64',label=label)
        if helper_end is not None:
            helper_end=self.op('helper_reduce',('host',),(cpu_end,helper_end),rows=rows,groups=[uses[e] for e in helper],label=label)
        reduced=self.op('host_reduce',('host',),[cpu_end]+([helper_end] if helper_end is not None else []),rows=rows,label=label)
        joined=self.op('primary_reduce',('p_compute',),(primary,reduced),rows=rows,queue='p_main',label=label)
        for e,i in ready.items():self.slots[i].reader=joined;self.slots[i].used=self.forward
        self.previous_gpu=joined;self.previous_host=reduced
        s.until(reduced)  # Actual caller returns before primary-stream completion.
        self.layers.append(dict(forward=self.forward,layer=li,rows=rows,generation=self.generation,
            cpu=cpu,primary=gpu,helper=helper,staged=selected,policy_prediction=predicted,
            cpu_op=cpu_end,primary_op=primary,helper_op=helper_end,done_op=joined))

    def replay(self,captures):
        for capture in captures:
            beginning=self.scheduler.now;generation_begin=None;generation_end=None
            for event in capture.events:
                typ=event['type']
                if typ=='generation_begin': generation_begin=self.scheduler.now
                elif typ=='draft_begin':
                    # Draft widths are part of the fixed captured workload, not
                    # a policy oracle for changed token/acceptance trajectories.
                    pass
                elif typ=='draft_end':
                    width=event.get('proposed',event.get('count',event.get('rows',4)))
                    op=self.op('draft',('p_compute','host'),rows=max(1,width),queue='p_main')
                    self.scheduler.until(op)
                elif typ=='boundary' and event['kind']=='forward_begin':
                    self.forward=event['forward'];self.generation=bool(event['generation'])
                    if self.generation:self.demand.row+=event['rows'];self.demand.block_rows=event['rows']
                    self.publish()
                    op=self.op('target_prefix',('p_compute','host'),rows=event['rows'],context=0,queue='p_main')
                    self.scheduler.until(op)
                elif typ=='routes':self.serve_layer(event)
                elif typ=='boundary' and event['kind']=='forward_end':
                    deps=[] if self.previous_gpu is None else [self.previous_gpu]
                    tail=self.op('target_tail',('p_compute','host'),deps,rows=event['rows'],queue='p_main')
                    self.scheduler.until(tail);self.publish();self.helper_admission();self.admission()
                elif typ=='round_end':
                    op=self.op('sampling',('host',),rows=1);self.scheduler.until(op)
                elif typ=='generation_end':generation_end=self.scheduler.now
            if generation_begin is None:raise ValueError('missing generation boundary')
            finish=generation_end if generation_end is not None else self.scheduler.now
            self.requests.append(dict(request_id=capture.request_id,total_ms=self.scheduler.now-beginning,
                generation_ms=finish-generation_begin,emitted=sum(e['type']=='token_output' for e in capture.events)))
            self.assert_ownership()
        return dict(requests=self.requests,actions=self.actions,counts=dict(self.counts),layers=self.layers,
            physical_sha256=self.physical.sha256,operations=self.scheduler.records(),
            pending_primary=len(self.exchanges),pending_helper=bool(self.helper_exchange),
            capacity=dict(primary=self.primary_capacity,staging=len(self.slots),helper=self.helper_capacity,
                          primary_gpu_frames=len(self.gpu_frames),exchange_descriptors=len(self.spares),
                          additional_backing_bytes=self.primary_capacity*self.record_bytes if self.backing else 0),
            scope='conditional fixed evaluated work; no prediction of changed hidden states or generation')
