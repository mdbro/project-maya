"""Aligned execution intervals. Enqueue timestamps are not execution intervals."""
import math


def union(intervals):
    result=[]
    for start,end in sorted(intervals):
        if not math.isfinite(start) or not math.isfinite(end) or end<=start:raise ValueError('invalid execution interval')
        if result and start<=result[-1][1]:result[-1][1]=max(end,result[-1][1])
        else:result.append([start,end])
    return result


def duration_ns(intervals):return sum(end-start for start,end in union(intervals))


def intersection_ns(a,b):
    a,b=union(a),union(b);i=j=0;total=0
    while i<len(a) and j<len(b):
        total+=max(0,min(a[i][1],b[j][1])-max(a[i][0],b[j][0]))
        if a[i][1]<=b[j][1]:i+=1
        else:j+=1
    return total


def fit_observation(row):
    timing=row.get('execution_intervals')
    if not timing or timing.get('scope')!='execution' or not timing.get('clock_id') or not timing.get('complete'):
        raise ValueError('aligned, complete execution start/end intervals are required')
    # Both series must already be in the named common clock domain, e.g. a
    # reconciled CUPTI/NVTX timeline. Independent per-device event clocks do not
    # satisfy this contract without a measured clock alignment.
    if timing.get('a_clock_id')!=timing['clock_id'] or timing.get('b_clock_id')!=timing['clock_id']:
        raise ValueError('execution intervals are not in a common measured clock')
    a,b=timing['a_ns'],timing['b_ns'];overlap=intersection_ns(a,b)*1e-6
    if overlap<=0:raise ValueError('no measured overlap; slowdown cannot be attributed to contention')
    if row.get('isolated_boundary')!='active_service':raise ValueError('isolated and concurrent service boundaries differ or are unspecified')
    concurrent=[duration_ns(a)*1e-6,duration_ns(b)*1e-6]
    fitted=[]
    for name,other,base,actual in zip((row['a'],row['b']),(row['b'],row['a']),
            (row['isolated_a_ms'],row['isolated_b_ms']),concurrent):
        lost=actual-base
        if lost<0:raise ValueError('concurrent service is faster than this isolated control; a pure slowdown fit is unresolved')
        if lost>=overlap:raise ValueError('service loss exceeds measured overlap; launch delay or another condition is unresolved')
        fitted.append(dict(pair=(name,other),factor=1/(1-lost/overlap),overlap_ms=overlap,
                           isolated_ms=base,concurrent_active_ms=actual))
    return fitted
