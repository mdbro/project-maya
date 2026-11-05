"""Production operation geometry, not execution prices.

NativeHelper::submit splits at 8 and groups jobs in ascending column width.
The primary native path submits parent_rows*4 descriptors in batches of four,
including empty tail descriptors. CPU service has one batch and three executor
phases over the same <=8-column chunks.
"""
from collections import Counter


def chunks(groups):
    result=[]
    for record,columns in enumerate(groups):
        if type(columns) is not int or not 1<=columns<=16:raise ValueError('expert group width outside native window')
        for start in range(0,columns,8):result.append((record,min(8,columns-start)))
    return result


def width_histogram(groups):
    counts=Counter(width for _,width in chunks(groups))
    return tuple(counts[w] for w in range(1,9))


def helper_launches(groups):
    pieces=chunks(groups);result=[]
    for width in range(1,9):
        records=Counter(record for record,w in pieces if w==width)
        if records:result.append(dict(width=width,jobs=sum(records.values()),records=len(records),
                                      reuse=tuple(sorted(records.values()))))
    return result


def execution_shape(kind,rows,groups):
    histogram=width_histogram(groups);count=sum(histogram)
    shape=dict(chunk_width_histogram=histogram,expert_groups=len(groups),columns=sum(groups),chunks=count,
               repeated_record_chunks=count-len(groups))
    if kind.startswith('helper') or kind in ('h_activation','h_result'):
        launches=helper_launches(groups);shape.update(launches=launches,launch_count=len(launches))
    elif kind in ('primary','primary_submit'):
        shape.update(submitted_descriptors=rows*4,launch_batches=rows,
                     active_batches=(count+3)//4,empty_descriptors=rows*4-count)
    elif kind=='cpu':
        shape.update(executor_phases=3,gate_up_tasks=count*384,
                     hidden_quantization_tasks=sum(groups)*12,down_tasks=count*384)
    return shape
