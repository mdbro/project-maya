#!/usr/bin/env python3
"""Overlap and completion boundaries for one profiled fixture, read from an Nsight SQLite export.

Answers which GPU and host work actually run at the same time in generation
forwards. Profiler and detailed-capture overhead are included, so durations are
evidence of ordering and overlap, not timing truth for an uninstrumented run.
NVTX and CUPTI intervals share the profiler clock.
"""
from __future__ import annotations
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
import pathlib
import re
import sqlite3
import statistics
import sys

WEIGHT_COPY_BYTES = 16 << 20  # expert records are about 32 MB; logits readback is about 4 MB
RANGE = re.compile(r'forward=(-?\d+) layer=(-?\d+) phase=(\S+)')
EXPERT_KERNELS = ('native_experts', 'minimax_ffn')


def union(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for a, b in sorted(intervals):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        elif b > a:
            merged.append([a, b])
    return [tuple(x) for x in merged]


def active(intervals: list[tuple[int, int]]) -> int:
    return sum(b - a for a, b in union(intervals))


def overlap(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> int:
    total, i, j = 0, 0, 0
    x, y = union(a), union(b)
    while i < len(x) and j < len(y):
        low, high = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        total += max(0, high - low)
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return total


def kernel_role(name: str) -> str:
    if 'wait_publication' in name:
        return 'wait_publication'
    if 'combine_window' in name:
        return 'combine'
    if any(k in name for k in EXPERT_KERNELS):
        return 'expert'
    return 'other'


def read_trace(path: pathlib.Path, generation_forwards: int) -> dict:
    db = sqlite3.connect('file:' + str(path.resolve()) + '?mode=ro', uri=True)
    db.row_factory = sqlite3.Row
    strings = {r['id']: r['value'] for r in db.execute('select id,value from StringIds')}
    ranges: dict[tuple[int, str], list[tuple[int, int, int, int]]] = defaultdict(list)
    phases: dict[tuple[int, int, str], tuple[int, int, int]] = {}
    for r in db.execute('select start,end,text,textId,globalTid from NVTX_EVENTS where end is not null'):
        text = r['text'] or strings.get(r['textId'], '')
        match = RANGE.search(text)
        if match:
            forward, layer, phase = int(match[1]), int(match[2]), match[3]
            ranges[r['globalTid'], phase].append((r['start'], r['end'], forward, layer))
            phases[forward, layer, phase] = (r['start'], r['end'], r['globalTid'])
    for values in ranges.values():
        values.sort()
    forwards = sorted({f for (f, _, p) in phases if p == 'forward_chunk'})
    generation = forwards[-generation_forwards:]
    if len(generation) != generation_forwards:
        raise ValueError(f'trace has {len(generation)} of {generation_forwards} generation forwards')
    begin, end = phases[generation[0], -1, 'forward_chunk'][0], phases[generation[-1], -1, 'forward_chunk'][1]
    chosen = set(generation)
    runtime = {}
    apis = []
    for r in db.execute('select start,end,globalTid,correlationId,nameId from CUPTI_ACTIVITY_KIND_RUNTIME where end>=? and start<=?', (begin, end)):
        runtime[r['correlationId']] = dict(r)
        apis.append(r)
    def locate(api, phase):
        entries = ranges.get((api['globalTid'], phase), [])
        i = bisect_right(entries, (api['start'], float('inf'), float('inf'), float('inf'))) - 1
        if i >= 0 and entries[i][0] <= api['start'] <= entries[i][1]:
            return entries[i][2], entries[i][3]
        return None
    windows: dict[tuple[int, int], dict] = defaultdict(lambda: dict(expert=[], wait=[], combine=[], other=[], copies=[], helper=[], helper_copies=[]))
    all_kernels: list[dict] = []
    for r in db.execute('select start,end,deviceId,correlationId,demangledName,streamId from CUPTI_ACTIVITY_KIND_KERNEL where end>=? and start<=? order by start', (begin, end)):
        api = runtime.get(r['correlationId'])
        name = strings.get(r['demangledName'], '?')
        role = kernel_role(name)
        kernel = dict(start=r['start'], end=r['end'], device=r['deviceId'], stream=r['streamId'], name=name, role=role)
        all_kernels.append(kernel)
        if api is None:
            continue
        if r['deviceId'] == 1:
            key = locate(api, 'helper_worker_cuda')
            if key and key[0] in chosen:
                windows[key]['helper'].append((r['start'], r['end']))
            continue
        key = locate(api, 'primary_expert_submit_and_join')
        if key and key[0] in chosen and role in ('expert', 'wait_publication', 'combine'):
            bucket = {'expert': 'expert', 'wait_publication': 'wait', 'combine': 'combine'}[role]
            windows[key][bucket].append((r['start'], r['end']))
            continue
        key = locate(api, 'layer_prepare')
        if key and key[0] in chosen:
            windows[key]['other'].append((r['start'], r['end']))
    # Demand copies share the kernels' stream; optional exchanges use the copy stream,
    # even when one is issued from inside a layer's join boundary.
    main_streams = {k['stream'] for k in all_kernels if k['role'] == 'wait_publication'}
    if len(main_streams) != 1:
        raise ValueError(f'expected one primary stream, found {main_streams}')
    main_stream = next(iter(main_streams))
    weight_copies = []
    for r in db.execute('select start,end,deviceId,streamId,correlationId,bytes,copyKind from CUPTI_ACTIVITY_KIND_MEMCPY where end>=? and start<=? order by start', (begin, end)):
        api = runtime.get(r['correlationId'])
        if r['deviceId'] == 1:
            key = locate(api, 'helper_worker_cuda') if api else None
            if key and key[0] in chosen and r['bytes'] < WEIGHT_COPY_BYTES:
                windows[key]['helper_copies'].append((r['start'], r['end']))
            if r['bytes'] >= WEIGHT_COPY_BYTES:
                weight_copies.append(dict(start=r['start'], end=r['end'], device=1, stream=r['streamId'], bytes=r['bytes'], kind=r['copyKind'], role='helper_exchange'))
            continue
        if r['bytes'] >= WEIGHT_COPY_BYTES:
            key = locate(api, 'layer_service') if api else None
            role = 'demand' if r['streamId'] == main_stream else 'primary_exchange'
            copy = dict(start=r['start'], end=r['end'], device=0, stream=r['streamId'], bytes=r['bytes'], kind=r['copyKind'], role=role)
            weight_copies.append(copy)
            if role == 'demand' and key and key[0] in chosen:
                windows[key]['copies'].append((r['start'], r['end']))
    db.close()
    return dict(phases=phases, windows=windows, generation=generation, begin=begin, end=end, kernels=all_kernels, weight_copies=weight_copies)


def summarize(trace: dict) -> dict:
    phases, windows = trace['phases'], trace['windows']
    rows = []
    for (forward, layer), w in sorted(windows.items()):
        host = phases.get((forward, layer, 'primary_expert_submit_and_join'))
        if not host:
            continue
        lanes = [phases[forward, layer, p] for p in ('cpu_expert_batch', 'helper_collect') if (forward, layer, p) in phases]
        if not lanes:
            continue  # all four picks on the primary: there is no host lane to overlap
        lane_begin = min(x[0] for x in lanes)
        lane_end = max(x[1] for x in lanes)
        lane = [(lane_begin, lane_end)]
        expert_active, copy_active = active(w['expert']), active(w['copies'])
        first_expert = min((a for a, _ in w['expert']), default=None)
        first_copy = min((a for a, _ in w['copies']), default=None)
        wait = active(w['wait'])
        rows.append(dict(forward=forward, layer=layer, lane_ms=(lane_end - lane_begin) * 1e-6, expert_ms=expert_active * 1e-6, copy_ms=copy_active * 1e-6,
                         expert_during_lane_ms=overlap(w['expert'], lane) * 1e-6, copy_during_lane_ms=overlap(w['copies'], lane) * 1e-6,
                         wait_ms=wait * 1e-6, helper_ms=active(w['helper']) * 1e-6,
                         expert_starts_before_lane_end=first_expert is not None and first_expert < lane_end,
                         copy_starts_before_lane_end=first_copy is not None and first_copy < lane_end,
                         expert_start_after_lane_end_ms=(first_expert - lane_end) * 1e-6 if first_expert is not None else None,
                         has_expert=bool(w['expert']), has_copy=bool(w['copies'])))
    total = lambda k: sum(r[k] for r in rows)
    with_expert = [r for r in rows if r['has_expert']]
    with_copy = [r for r in rows if r['has_copy']]
    copies = trace['weight_copies']
    device0 = [(k['start'], k['end']) for k in trace['kernels'] if k['device'] == 0]
    roles: dict[str, dict] = {}
    for role in ('demand', 'primary_exchange', 'helper_exchange'):
        chosen = [c for c in copies if c['role'] == role]
        spans = [(c['start'], c['end']) for c in chosen]
        roles[role] = dict(copies=len(chosen), bytes=sum(c['bytes'] for c in chosen), active_ms=active(spans) * 1e-6,
                           during_primary_kernels_ms=overlap(spans, device0) * 1e-6,
                           median_ms=statistics.median((c['end'] - c['start']) * 1e-6 for c in chosen) if chosen else None,
                           streams=sorted({c['stream'] for c in chosen}), kinds=sorted({c['kind'] for c in chosen}))
    return dict(generation_span_ms=(trace['end'] - trace['begin']) * 1e-6, windows_with_host_lane=len(rows),
                host_lane_ms=total('lane_ms'), primary_expert_kernels_ms=total('expert_ms'),
                primary_expert_during_host_lane_ms=total('expert_during_lane_ms'),
                demand_copy_ms=total('copy_ms'), demand_copy_during_host_lane_ms=total('copy_during_lane_ms'),
                wait_publication_ms=total('wait_ms'), helper_kernels_ms=total('helper_ms'),
                windows_with_expert=len(with_expert), expert_starts_before_lane_end=sum(r['expert_starts_before_lane_end'] for r in with_expert),
                median_expert_start_after_lane_end_ms=statistics.median(r['expert_start_after_lane_end_ms'] for r in with_expert) if with_expert else None,
                windows_with_copy=len(with_copy), copy_starts_before_lane_end=sum(r['copy_starts_before_lane_end'] for r in with_copy),
                copies=roles, device0_kernel_active_ms=active(device0) * 1e-6,
                scope='Profiled companion with CUDA and NVTX tracing and detailed capture; ordering and overlap at this scope, not uninstrumented timing.')


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('sqlite', type=pathlib.Path)
    ap.add_argument('--generation-forwards', type=int, default=64)
    ap.add_argument('--output', required=True, type=pathlib.Path)
    a = ap.parse_args()
    result = summarize(read_trace(a.sqlite, a.generation_forwards))
    with a.output.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('copies', 'scope')}, indent=1))
    print(json.dumps(result['copies'], indent=1))
    return 0


if __name__ == '__main__':
    sys.exit(main())
