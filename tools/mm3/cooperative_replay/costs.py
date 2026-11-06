"""Versioned physical measurements, independently fitted from policy estimates."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import statistics
from pathlib import Path
from .execution import Estimate
from .structure import width_histogram,helper_launches,execution_shape


class CoverageError(ValueError): pass


class ServiceTable:
    """Measured tables with bounded interpolation and explicit coverage limits.

    No thread-count scaling, CPU admission scale, or per-operation observed
    duration is accepted here. Equal-shape samples are aggregated before use;
    validation request IDs are forbidden in calibration provenance.
    """
    def __init__(self, document, *, held_out=()):
        if document.get('schema') != 1 or document.get('units') != 'ms':
            raise ValueError('invalid physical measurement schema')
        self.document = document
        if document.get('point_estimate','median') not in ('mean','median'):raise ValueError('unknown service center')
        self.center=statistics.mean if document.get('point_estimate')=='mean' else statistics.median
        implementation=b''.join(Path(__file__).with_name(name).read_bytes() for name in ('costs.py','structure.py','execution.py','machine.py'))
        self.sha256 = hashlib.sha256(json.dumps(document,sort_keys=True).encode()+implementation).hexdigest()
        self.entries = defaultdict(list)
        self.queries = []
        self.interpolation_checks={}
        self._estimate_cache={};self._helper_cache={}
        held_out = set(held_out)
        if held_out.intersection(document.get('calibration_request_ids',[])):
            raise ValueError('held-out request was used to fit physical tables')
        for row in document.get('measurements', []):
            if not row.get('evidence') or row.get('request_id') in held_out or held_out.intersection(row.get('request_ids',[])):
                raise ValueError('missing provenance or held-out timing leakage')
            if any(k in row for k in ('cooperative_cpu_scale','admission_margin','horizon')):
                raise ValueError('policy knob in physical service measurements')
            values = row.get('samples_ms')
            if not values or any(not math.isfinite(x) or x <= 0 for x in values):
                raise ValueError('invalid measured service samples')
            signature = (row['kind'], str(row.get('workers','none')), row.get('working_set','resident'))
            self.entries[signature].append(row)
        self.contention = {}
        for row in document.get('contention', []):
            if not row.get('evidence') or row.get('request_id') in held_out:
                raise ValueError('invalid contention provenance')
            self.contention[tuple(row['pair'])] = Estimate(row['factor'], row['low'], row['high'],
                tuple(row['evidence']), row.get('qualified',False))

    @classmethod
    def read(cls, path, **kwargs): return cls(json.loads(Path(path).read_text()), **kwargs)

    @staticmethod
    def _features(row):
        groups = row.get('groups', [])
        histogram=width_histogram(groups)
        return (row.get('rows',1),len(groups),sum(groups),max(groups,default=0),
                row.get('bytes',0),row.get('context',0),*histogram,
                sum(histogram),sum(bool(n) for n in histogram),(sum(histogram)+3)//4,
                row.get('rows',1)*4-sum(histogram),sum(histogram)-len(groups))

    @staticmethod
    def _regime(kind,row):
        if kind=='primary':
            # run() always submits parent_rows*4 descriptors in batches of four.
            shape=execution_shape(kind,row.get('rows',1),row.get('groups',[]))
            # Repeated chunks reuse a record but do not change run()'s kernel
            # or launch geometry. Keep reuse as an interpolation feature rather
            # than an artificial boundary that excludes measured mixed groups.
            return (shape['launch_batches'],)
        return ()

    @staticmethod
    def _reuse_regime(reuse):
        return ('uniform',reuse[0]) if reuse and len(set(reuse))==1 else ('mixed',*reuse)

    @staticmethod
    def _in_domain(kind,target,low,high):
        dimensions=range(6)
        if kind=='primary':
            # Native primary always submits rows batches of four C=8 jobs.
            # A new mixture may have more useful columns than any one measured
            # mixture while retaining measured job counts and per-record widths.
            # That is a mixed-shape interpolation inside the same launch/kernel
            # regime, not a new parent-width or worker configuration. Holdouts
            # must test these corner mixtures too; support is not automatic.
            dimensions=(0,1,3,4,5,14,16,18)
        return all(low[d]<=target[d]<=high[d] for d in dimensions)

    def helper_group(self,width,jobs,reuse,*,working_set='resident'):
        """One actual fast_ffn_cuda_packed(width, jobs) launch group.

        Homogeneous measurements bracket the same GPU event work. They include
        any enqueue gaps inside that span; they are not pure arithmetic times.
        Parent widths do not change this kernel geometry. Their repeat spread
        is retained, rather than hidden by a global helper multiplier.
        """
        cache_key=width,jobs,tuple(reuse),working_set
        if cache_key in self._helper_cache:return self._helper_cache[cache_key]
        key=('helper','none',working_set);regime=self._reuse_regime(reuse)
        nodes=defaultdict(list)
        explicit=self.entries.get(('helper_width_group','none',working_set),[])
        if explicit:
            for row in explicit:
                if row['width']==width and self._reuse_regime(row['reuse'])==regime:nodes[row['jobs']].append(row)
        else:
            for row in self.entries.get(key,[]):
                launches=helper_launches(row.get('groups',[]))
                if len(launches)==1 and launches[0]['width']==width and self._reuse_regime(launches[0]['reuse'])==regime:
                    nodes[launches[0]['jobs']].append(row)
        if not nodes and width==8 and regime[0]=='mixed' and set(reuse)<={1,2}:
            # At <=16 parent columns, only width-eight jobs can repeat a
            # record within one launch. Bracket mixed reuse by independently
            # measured all-distinct/all-paired jobs of the same width. This is
            # an explicit diagnostic interpolation hypothesis, not qualification.
            distinct=self.helper_group(width,jobs,[1]*jobs,working_set=working_set)
            lower=jobs//2*2;upper=(jobs+1)//2*2
            if not lower:raise CoverageError('mixed reuse needs at least two jobs')
            paired_low=self.helper_group(width,lower,[2]*(lower//2),working_set=working_set)
            paired_high=self.helper_group(width,upper,[2]*(upper//2),working_set=working_set)
            fraction=(jobs-lower)/(upper-lower) if upper!=lower else 0
            paired_ms=paired_low.ms*(1-fraction)+paired_high.ms*fraction
            share=(jobs-len(reuse))/(jobs/2)
            center=distinct.ms*(1-share)+paired_ms*share
            envelope=(distinct,paired_low,paired_high)
            out=Estimate(center,min(c.low for c in envelope),max(c.high for c in envelope),
                tuple(sorted({e for c in envelope for e in c.evidence})),False)
            self.queries.append((key,(width,jobs,*reuse),True,'unvalidated_mixed_record_reuse'))
            self._helper_cache[cache_key]=out;return out
        if not nodes:raise CoverageError(f'no measured helper width/reuse regime: width={width}, reuse={regime}, {working_set}')
        positions=sorted(nodes)
        samples={n:[x for r in nodes[n] for x in r['samples_ms']] for n in positions}
        centers={n:self.center(samples[n]) for n in positions}
        checks=[]
        # Leave out the whole measured shape, including all of its repeats.
        for i,n in enumerate(positions[1:-1],1):
            left,right=positions[i-1],positions[i+1];fraction=(n-left)/(right-left)
            predicted=centers[left]*(1-fraction)+centers[right]*fraction
            checks.append(dict(jobs=n,predicted_ms=predicted,observed_ms=centers[n],
                residual_ms=centers[n]-predicted,
                inside_neighbor_repeat_envelope=min(samples[left]+samples[right])<=centers[n]<=max(samples[left]+samples[right])))
        check_key=str((key,width,regime))
        self.interpolation_checks[check_key]=dict(method='leave one whole homogeneous shape out',checks=checks,
            supported=len(checks)>=2 and all(c['inside_neighbor_repeat_envelope'] for c in checks))
        if jobs in nodes:
            chosen=[jobs];weights=[1.0];center=centers[jobs];mode='measured_width_group'
            low=min(samples[jobs]);high=max(samples[jobs]);qualified=len(samples[jobs])>=3 and all(r.get('qualified',True) for r in nodes[jobs])
        else:
            left=max((n for n in positions if n<jobs),default=None);right=min((n for n in positions if n>jobs),default=None)
            if left is None or right is None:raise CoverageError(f'helper job count outside measured width-{width} range: {jobs} not in {positions}')
            fraction=(jobs-left)/(right-left);chosen=[left,right];weights=[1-fraction,fraction]
            center=centers[left]*(1-fraction)+centers[right]*fraction
            residual=max((abs(c['residual_ms']) for c in checks),default=0)
            low=max(0,sum(w*min(samples[n]) for w,n in zip(weights,chosen))-residual)
            high=sum(w*max(samples[n]) for w,n in zip(weights,chosen))+residual
            qualified=self.interpolation_checks[check_key]['supported'] and all(r.get('qualified',True) for n in chosen for r in nodes[n])
            mode='validated_width_interpolation' if qualified else 'untested_width_interpolation'
        evidence=tuple(sorted({p for n in chosen for r in nodes[n] for p in r['evidence']}))
        out=Estimate(center,min(low,center),max(high,center),evidence,qualified)
        self.queries.append((key,(width,jobs,*reuse),not qualified,mode))
        self._helper_cache[cache_key]=out
        return out

    def helper_groups(self,groups,*,working_set='resident'):
        return [(launch,self.helper_group(launch['width'],launch['jobs'],launch['reuse'],working_set=working_set))
                for launch in helper_launches(groups)]

    def _shape_check(self,key,regime,candidates):
        identity=str((key,regime))
        if identity in self.interpolation_checks:return self.interpolation_checks[identity]
        clustered=defaultdict(list)
        for row in candidates:clustered[self._features(row)].append(row)
        shapes=sorted(clustered);checks=[]
        # A finite, deterministic shape holdout. All repeats of the held-out
        # shape are excluded, including measurements with another record order.
        stride=max(1,(len(shapes)+127)//128)
        for target in shapes[::stride]:
            other=[f for f in shapes if f!=target]
            if len(other)<3:continue
            low=[min(f[d] for f in other) for d in range(len(target))]
            high=[max(f[d] for f in other) for d in range(len(target))]
            if not self._in_domain(key[0],target,low,high):continue
            scale=[max(hi-lo,1) for lo,hi in zip(low,high)]
            near=sorted((sum(((x-y)/s)**2 for x,y,s in zip(target,f,scale)),f) for f in other)[:4]
            inverse=[1/max(d,1e-6) for d,_ in near];weights=[v/sum(inverse) for v in inverse]
            values=[[v for r in clustered[f] for v in r['samples_ms']] for _,f in near]
            predicted=sum(w*self.center(v) for w,v in zip(weights,values))
            actual=self.center([v for r in clustered[target] for v in r['samples_ms']])
            envelope=[v for data in values for v in data]
            checks.append(dict(shape=target,predicted_ms=predicted,observed_ms=actual,
                relative_residual=abs(actual-predicted)/max(predicted,1e-12),
                inside_neighbor_repeat_envelope=min(envelope)<=actual<=max(envelope)))
        residuals=sorted(c['relative_residual'] for c in checks)
        covered=sum(c['inside_neighbor_repeat_envelope'] for c in checks)/len(checks) if checks else 0
        result=dict(method='leave one whole shape out; at most 128 deterministic holdouts',checks=checks,
            empirical_p95_relative_residual=residuals[math.ceil(.95*len(residuals))-1] if residuals else None,
            neighbor_repeat_envelope_coverage=covered,
            supported=len(checks)>=4 and covered>=.9)
        self.interpolation_checks[identity]=result;return result

    def estimate(self, kind, *, rows=1, groups=(), workers='none', working_set='resident', bytes=0, context=0):
        cache_key=kind,rows,tuple(sorted(groups)),str(workers),working_set,bytes,context
        if cache_key in self._estimate_cache:return self._estimate_cache[cache_key]
        if kind=='helper':
            estimates=[cost for _,cost in self.helper_groups(groups,working_set=working_set)]
            out=Estimate(sum(c.ms for c in estimates),sum(c.low for c in estimates),sum(c.high for c in estimates),
                         tuple(sorted({e for c in estimates for e in c.evidence})),all(c.qualified for c in estimates))
            self._estimate_cache[cache_key]=out;return out
        key = kind, str(workers), working_set
        candidates = self.entries.get(key, [])
        if not candidates: raise CoverageError(f'no measured configuration: {key}')
        target = dict(kind=kind,rows=rows,groups=list(groups),bytes=bytes,context=context)
        candidates=[r for r in candidates if self._regime(kind,r)==self._regime(kind,target)]
        if not candidates:raise CoverageError(f'unmeasured kernel/launch regime: {key}, {self._regime(kind,target)}')
        features = self._features(target)
        exact = [r for r in candidates if self._features(r)==features and sorted(r.get('groups',[]))==sorted(groups)]
        interpolated = not exact
        if exact:
            selected, weights = exact, [1/len(exact)]*len(exact)
        else:
            values = [self._features(r) for r in candidates]
            low = [min(v[d] for v in values) for d in range(len(features))]
            high = [max(v[d] for v in values) for d in range(len(features))]
            # New mixtures of measured chunk widths are interpolation, not new
            # hardware configurations. The aggregate domain still bounds work.
            if not self._in_domain(kind,features,low,high):
                raise CoverageError(f'outside measured range: {key} {features}; {low}..{high}')
            scale = [max(hi-lo,1) for lo,hi in zip(low,high)]
            distances = sorted((sum(((a-b)/s)**2 for a,b,s in zip(features,v,scale)),i) for i,v in enumerate(values))[:4]
            selected = [candidates[i] for _,i in distances]
            inverse = [1/max(d,1e-6) for d,_ in distances]
            weights = [w/sum(inverse) for w in inverse]
        means = [self.center(r['samples_ms']) for r in selected]
        mean = sum(w*m for w,m in zip(weights,means))
        # Repeat variation is observed. No universal percentage is added to make
        # a prediction pass. Non-helper interpolation awaits shape residuals.
        lo = sum(w*min([*r['samples_ms'],*(v[0] for v in r.get('measurement_bounds_ms',[]))]) for w,r in zip(weights,selected))
        hi = sum(w*max([*r['samples_ms'],*(v[1] for v in r.get('measurement_bounds_ms',[]))]) for w,r in zip(weights,selected))
        check=None
        if interpolated:
            check=self._shape_check(key,self._regime(kind,target),candidates)
            residual=check['empirical_p95_relative_residual']
            if residual is not None:lo=max(0,lo-mean*residual);hi+=mean*residual
        evidence = tuple(sorted({e for r in selected for e in r['evidence']}))
        supported=(check['supported'] if interpolated else True) and all(r.get('qualified',True) for r in selected)
        out = Estimate(mean,max(0,min(lo,mean)),max(hi,mean),evidence,supported)
        mode=('validated_shape_interpolation' if supported else 'unvalidated_shape_interpolation') if interpolated else 'measured_shape'
        self.queries.append((key,features,not out.qualified,mode))
        self._estimate_cache[cache_key]=out
        return out
