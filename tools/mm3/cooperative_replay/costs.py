"""Versioned physical measurements, independently fitted from policy estimates."""
from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from .execution import Estimate


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
        self.sha256 = hashlib.sha256(json.dumps(document,sort_keys=True).encode()).hexdigest()
        self.entries = defaultdict(list)
        self.queries = []
        held_out = set(held_out)
        for row in document.get('measurements', []):
            if not row.get('evidence') or row.get('request_id') in held_out:
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
        # Composition includes count at each width, not frequency or hits.
        return (row.get('rows',1), len(groups), sum(groups), max(groups,default=0),
                row.get('bytes',0), row.get('context',0))

    def estimate(self, kind, *, rows=1, groups=(), workers='none', working_set='resident', bytes=0, context=0):
        key = kind, str(workers), working_set
        candidates = self.entries.get(key, [])
        if not candidates: raise CoverageError(f'no measured configuration: {key}')
        target = dict(rows=rows,groups=list(groups),bytes=bytes,context=context)
        features = self._features(target)
        exact = [r for r in candidates if self._features(r)==features and sorted(r.get('groups',[]))==sorted(groups)]
        interpolated = not exact
        if exact:
            selected, weights = exact, [1/len(exact)]*len(exact)
        else:
            values = [self._features(r) for r in candidates]
            low = [min(v[d] for v in values) for d in range(6)]
            high = [max(v[d] for v in values) for d in range(6)]
            if any(t<lo or t>hi for t,lo,hi in zip(features,low,high)):
                raise CoverageError(f'outside measured range: {key} {features}; {low}..{high}')
            scale = [max(hi-lo,1) for lo,hi in zip(low,high)]
            distances = sorted((sum(((a-b)/s)**2 for a,b,s in zip(features,v,scale)),i) for i,v in enumerate(values))[:4]
            selected = [candidates[i] for _,i in distances]
            inverse = [1/max(d,1e-6) for d,_ in distances]
            weights = [w/sum(inverse) for w in inverse]
        means = [sum(r['samples_ms'])/len(r['samples_ms']) for r in selected]
        mean = sum(w*m for w,m in zip(weights,means))
        # Sample ranges include repeat variation. Interpolation adds a declared
        # coverage margin; it is never silently treated as qualified mixed fit.
        margin = max((r.get('relative_uncertainty',.05) for r in selected), default=.05)
        if interpolated: margin = max(margin,self.document.get('interpolation_uncertainty',.20))
        lo = min(mean*(1-margin),sum(w*min(r['samples_ms']) for w,r in zip(weights,selected)))
        hi = max(mean*(1+margin),sum(w*max(r['samples_ms']) for w,r in zip(weights,selected)))
        evidence = tuple(sorted({e for r in selected for e in r['evidence']}))
        out = Estimate(mean,max(0,lo),hi,evidence,not interpolated and all(r.get('qualified',True) for r in selected))
        self.queries.append((key,features,interpolated))
        return out
