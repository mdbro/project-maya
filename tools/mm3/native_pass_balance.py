#!/usr/bin/env python3
"""How close each activation pass through each layer is to balance, from its own joint timeline.

Every comparison is made inside one pass, on completion times from that pass's common origin, and only then
aggregated. Means of durations are never compared across lanes.

Three levels, each with its own limits:
  records   A full light capture gives CPU and helper completion directly. It holds no GPU times, so when
            copies and primary kernels finish before the lanes their slack is invisible, and when they finish
            after, only the wait at the next join shows it.
  trace     An Nsight export adds copy and kernel start and end times per pass, at the scope of a detailed
            capture. Durations include instrumentation; ordering and overlap are the evidence.
  division  For each pass, how its actual uploadable experts could be divided between the CPU and uploads,
            priced by the grouped model. The model is unqualified and demand is fixed, and a copy's later reuse
            is not counted, so this is a same-pass bound that proposes candidates. Hardware decides.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import json
import pathlib
import statistics
import sys

import native_light as light
import native_passes as passes
from shared_model import grouped

LANE_BAND_MS = 0.05  # lanes this close count as finishing together
EXPOSED_MS = 0.10    # wait beyond a pass's settled wait that counts as primary work outlasting the lanes


def quantiles(values: list[float]) -> dict:
    if not values:
        return dict(n=0)
    ordered = sorted(values)
    at = lambda q: ordered[min(len(ordered) - 1, int(q * len(ordered)))]
    return dict(n=len(ordered), p10=at(.1), p50=at(.5), p90=at(.9), mean=statistics.fmean(ordered))


def settled_waits(rows: list) -> dict[int, float]:
    """The join wait of passes with no copy and lanes long enough to hide primary work, per rows."""
    groups: dict[int, list[float]] = defaultdict(list)
    for r in rows:
        if r.join_wait_next_ms is not None and r.new == 0 and r.lane_end - r.lane_begin > 0.3 and not r.exchanges_pending and not r.helper_lease:
            groups[r.rows].append(r.join_wait_next_ms)
    return {k: statistics.median(v) for k, v in groups.items() if v}


def classify(r, settled: dict[int, float]) -> dict:
    """Which lane finished last in one pass, from its own origin, and whether primary work outlasted the lanes."""
    begin = r.lane_begin
    cpu = r.cpu_end - begin if r.cpu_end is not None else None
    helper = r.helper_published - begin if r.helper_published is not None else None
    if cpu is None and helper is None:
        last, gap = 'neither', 0.0
    elif helper is None:
        last, gap = 'cpu_only', cpu
    elif cpu is None:
        last, gap = 'helper_only', helper
    else:
        gap = cpu - helper
        last = 'together' if abs(gap) <= LANE_BAND_MS else ('cpu' if gap > 0 else 'helper')
    base = settled.get(r.rows)
    wait = r.join_wait_next_ms
    excess = None if wait is None or base is None else wait - base
    return dict(last=last, gap=gap, excess=excess, exposed=excess is not None and excess > EXPOSED_MS, cycle=r.cycle_ms,
                rows=r.rows, new=r.new, cpu_jobs=r.cpu_jobs, helper_jobs=r.helper_jobs, cpu=cpu, helper=helper)


def records_summary(rows: list) -> dict:
    settled = settled_waits(rows)
    items = [classify(r, settled) for r in rows if not r.final_layer]
    total = sum(x['cycle'] for x in items)
    out: dict = dict(passes=len(items), cycle_ms=total, settled_wait_ms=settled, by_rows={})
    for k in sorted({x['rows'] for x in items}):
        group = [x for x in items if x['rows'] == k]
        cycle = sum(x['cycle'] for x in group)
        classes = {}
        for label in ('cpu', 'helper', 'together', 'cpu_only', 'helper_only', 'neither'):
            chosen = [x for x in group if x['last'] == label]
            if chosen:
                classes[label] = dict(passes=len(chosen), share_of_passes=len(chosen) / len(group), share_of_cycle=sum(x['cycle'] for x in chosen) / cycle,
                                      exposed_share=sum(x['exposed'] for x in chosen) / len(chosen), gap_ms=quantiles([x['gap'] for x in chosen]))
        by_copies = {}
        for c in sorted({x['new'] for x in group}):
            chosen = [x for x in group if x['new'] == c]
            by_copies[c] = dict(passes=len(chosen), share_of_cycle=sum(x['cycle'] for x in chosen) / cycle, cycle_ms=quantiles([x['cycle'] for x in chosen]),
                                cpu_ms=quantiles([x['cpu'] for x in chosen if x['cpu'] is not None]), helper_ms=quantiles([x['helper'] for x in chosen if x['helper'] is not None]),
                                cpu_jobs=statistics.fmean(x['cpu_jobs'] for x in chosen), exposed_share=sum(x['exposed'] for x in chosen) / len(chosen),
                                excess_ms=quantiles([x['excess'] for x in chosen if x['excess'] is not None]))
        out['by_rows'][k] = dict(passes=len(group), share_of_cycle=cycle / total, classes=classes, by_copies=by_copies)
    return out


def best_division(r, model: grouped.GroupedTiming, limit: int) -> dict:
    """The number of uploads that finishes this pass soonest under the model, against what it did."""
    movable = r.cpu_jobs + r.new
    columns = r.cpu / r.cpu_jobs if r.cpu_jobs else 1.0
    exchange, lease = bool(r.exchanges_pending), bool(r.helper_lease)
    fixed_primary = r.primary_jobs - r.new

    def price(uploads: int) -> float:
        jobs = movable - uploads
        return model.cycle_ms(r.rows, jobs, jobs * columns, r.helper_jobs, r.helper, fixed_primary + uploads, uploads, exchange, lease)
    actual = price(r.new)
    choices = {k: price(k) for k in range(0, min(movable, limit) + 1)}
    best = min(choices, key=lambda k: (choices[k], abs(k - r.new)))
    # A pass at the upload limit with CPU work left is a pass the limit may be holding back.
    wider = {k: price(k) for k in range(0, min(movable, 8) + 1)}
    return dict(actual=actual, best=choices[best], best_uploads=best, saved=max(0.0, actual - choices[best]),
                saved_wider=max(0.0, actual - min(wider.values())), best_uploads_wider=min(wider, key=lambda k: (wider[k], abs(k - r.new))))


def division_summary(rows: list, model: grouped.GroupedTiming, limit: int = 4) -> dict:
    out: dict = dict(by_rows={}, limit=limit)
    total = saved = wider = 0.0
    for k in sorted({r.rows for r in rows}):
        group = [r for r in rows if r.rows == k and not r.final_layer and r.cpu_jobs + r.new > 0]
        results = [(r, best_division(r, model, limit)) for r in group]
        priced = sum(x['actual'] for _, x in results)
        cycle = sum(r.cycle_ms for r in rows if r.rows == k and not r.final_layer)
        moves = defaultdict(lambda: dict(passes=0, saved=0.0))
        for r, x in results:
            key = x['best_uploads'] - r.new
            moves[key]['passes'] += 1
            moves[key]['saved'] += x['saved']
        out['by_rows'][k] = dict(passes=len(group), cycle_ms=cycle, priced_ms=priced, saved_ms=sum(x['saved'] for _, x in results), saved_wider_ms=sum(x['saved_wider'] for _, x in results),
                                 moves={int(a): dict(b) for a, b in sorted(moves.items())})
        total += cycle
        saved += out['by_rows'][k]['saved_ms']
        wider += out['by_rows'][k]['saved_wider_ms']
    out.update(cycle_ms=total, saved_ms=saved, saved_wider_ms=wider, saved_fraction=saved / total if total else None, saved_wider_fraction=wider / total if total else None)
    return out


def trace_rows(path: pathlib.Path, forwards: int) -> list[dict]:
    """Per-pass completion from the lanes' own begin, ms. Primary work ends with its last kernel, which follows its copies."""
    import native_shared_trace as trace
    t = trace.read_trace(path, forwards)
    phases, windows = t['phases'], t['windows']
    rows = []
    for (forward, layer), w in sorted(windows.items()):
        cpu = phases.get((forward, layer, 'cpu_expert_batch'))
        helper = phases.get((forward, layer, 'helper_collect'))
        submit = phases.get((forward, layer, 'primary_expert_submit_and_join'))
        lanes = [x for x in (cpu, helper) if x]
        if not lanes or not submit:
            continue
        begin = min(x[0] for x in lanes)
        ms = lambda value: (value - begin) * 1e-6
        rows.append(dict(forward=forward, layer=layer, cpu_end=ms(cpu[1]) if cpu else None,
                         helper_end=ms(max(b for _, b in w['helper'])) if w['helper'] else None,
                         copy_start=ms(min(a for a, _ in w['copies'])) if w['copies'] else None, copy_end=ms(max(b for _, b in w['copies'])) if w['copies'] else None,
                         primary_start=ms(min(a for a, _ in w['expert'])) if w['expert'] else None, primary_end=ms(max(b for _, b in w['expert'])) if w['expert'] else None,
                         combine_end=ms(max(b for _, b in w['combine'])) if w['combine'] else None, copies=len(w['copies']), host_end=ms(submit[1])))
    return rows


PATHS = ('cpu', 'primary_transfer', 'primary_compute', 'helper_transfer', 'helper_compute')
PAIRS = (('cpu', 'primary_transfer'), ('helper_transfer', 'cpu'), ('primary_transfer', 'helper_transfer'),
         ('cpu', 'helper_compute'), ('primary_transfer', 'helper_compute'), ('helper_transfer', 'helper_compute'))
LANES = ('cpu', 'primary', 'helper')  # independent resources, for slowest against second slowest


def path_ends(path: pathlib.Path, forwards: int) -> list[dict]:
    """Completion of each path per pass, ms from the pass's common origin, the first CPU or helper lane start.

    primary_transfer: demand copies and the expert kernels that start once the first copy has finished.
    primary_compute: expert kernels that start before then, which on this stream are the resident experts. A kernel is
    not tied to its expert in the trace, so the split is by time and is approximate when kernels queue behind a copy.
    helper_transfer: the helper's last activation or result copy. helper_compute: its last kernel.
    """
    import native_shared_trace as trace
    t = trace.read_trace(path, forwards)
    phases, windows = t['phases'], t['windows']
    rows = []
    for (forward, layer), w in sorted(windows.items()):
        cpu = phases.get((forward, layer, 'cpu_expert_batch'))
        helper = phases.get((forward, layer, 'helper_collect'))
        lanes = [x for x in (cpu, helper) if x]
        if not lanes:
            continue
        begin = min(x[0] for x in lanes)
        ms = lambda value: (value - begin) * 1e-6
        end = dict.fromkeys(PATHS)
        if cpu:
            end['cpu'] = ms(cpu[1])
        if w['copies']:
            first = min(b for _, b in w['copies'])
            after = [b for a, b in w['expert'] if a >= first]
            before = [b for a, b in w['expert'] if a < first]
            end['primary_transfer'] = ms(max([max(b for _, b in w['copies'])] + after))
            end['primary_compute'] = ms(max(before)) if before else None
        elif w['expert']:
            end['primary_compute'] = ms(max(b for _, b in w['expert']))
        if w['helper_copies']:
            end['helper_transfer'] = ms(max(b for _, b in w['helper_copies']))
        if w['helper']:
            end['helper_compute'] = ms(max(b for _, b in w['helper']))
        primary = [x for x in (end['primary_transfer'], end['primary_compute']) if x is not None]
        helper_end = [x for x in (end['helper_transfer'], end['helper_compute']) if x is not None]
        rows.append(dict(forward=forward, layer=layer, copies=len(w['copies']), ends=end,
                         lanes=dict(cpu=end['cpu'], primary=max(primary) if primary else None, helper=max(helper_end) if helper_end else None)))
    return rows


HELD_GAP_MS = 0.05  # idle time before a primary kernel that counts as waiting in the driver's submission batch


def held_split(w: dict, cpu_end_ns: int | None, begin_ns: int) -> dict:
    """The primary's own completion if every launch had reached the GPU at once, and what the held launches added.

    Launches reach the GPU in groups, and the rest of a window wait for the host's next CUDA call, which comes at the join,
    after the host's own CPU work. A primary kernel is held when it starts after the CPU batch ended and after an idle gap
    the stream did not need: longer than HELD_GAP_MS past both the previous primary kernel and the last copy. Held expert
    kernels are replayed back to back after the last unheld work, keeping their own durations; the publication wait and the
    combine depend on the lanes and are left out of the primary's own work.
    """
    ms = lambda v: (v - begin_ns) * 1e-6
    copies = sorted(w['copies'])
    copy_end = max((b for _, b in copies), default=None)
    work = sorted([(a, b, 'expert') for a, b in w['expert']] + [(a, b, 'wait') for a, b in w['wait']] + [(a, b, 'combine') for a, b in w['combine']])
    previous, own_end, held_ms, held = None, copy_end, 0.0, 0
    replay = None
    for a, b, role in work:
        ready = max(x for x in (previous, copy_end, begin_ns) if x is not None)
        is_held = cpu_end_ns is not None and a > cpu_end_ns and (a - ready) * 1e-6 > HELD_GAP_MS
        if is_held or replay is not None:
            if replay is None:
                replay = ready
            if role == 'expert':
                replay += (b - a)
                own_end = replay
                held += 1
                held_ms += (b - a) * 1e-6
        elif role == 'expert':
            own_end = b if own_end is None else max(own_end, b)
        previous = b
    actual = max((b for _, b, role in work if role == 'expert'), default=None)
    return dict(own_end=ms(own_end) if own_end is not None else None, actual_end=ms(actual) if actual is not None else None,
                held_kernels=held, held_by_ms=(ms(actual) - ms(own_end)) if (actual is not None and own_end is not None) else 0.0)


def path_ends_held(path: pathlib.Path, forwards: int) -> list[dict]:
    """As path_ends, with the primary's own end corrected for launches held in the submission batch."""
    import native_shared_trace as trace
    t = trace.read_trace(path, forwards)
    phases, windows = t['phases'], t['windows']
    rows = []
    for (forward, layer), w in sorted(windows.items()):
        cpu = phases.get((forward, layer, 'cpu_expert_batch'))
        helper = phases.get((forward, layer, 'helper_collect'))
        lanes = [x for x in (cpu, helper) if x]
        if not lanes:
            continue
        begin = min(x[0] for x in lanes)
        ms = lambda value: (value - begin) * 1e-6
        split = held_split(w, cpu[1] if cpu else None, begin)
        end = dict.fromkeys(PATHS)
        if cpu:
            end['cpu'] = ms(cpu[1])
        if w['copies']:
            end['primary_transfer'] = max(ms(max(b for _, b in w['copies'])), split['own_end'] or 0.0)
        elif split['own_end'] is not None:
            end['primary_compute'] = split['own_end']
        if w['helper_copies']:
            end['helper_transfer'] = ms(max(b for _, b in w['helper_copies']))
        if w['helper']:
            end['helper_compute'] = ms(max(b for _, b in w['helper']))
        primary = [x for x in (end['primary_transfer'], end['primary_compute']) if x is not None]
        helper_end = [x for x in (end['helper_transfer'], end['helper_compute']) if x is not None]
        rows.append(dict(forward=forward, layer=layer, copies=len(w['copies']), ends=end, held=split,
                         lanes=dict(cpu=end['cpu'], primary=max(primary) if primary else None, helper=max(helper_end) if helper_end else None)))
    return rows


def records_ends(rows: list) -> list[dict]:
    """The same shape from full records: CPU end and helper publication only. Records hold no GPU times."""
    out = []
    for r in rows:
        if r.final_layer:
            continue
        cpu = r.cpu_end - r.lane_begin if r.cpu_end is not None else None
        helper = r.helper_published - r.lane_begin if r.helper_published is not None else None
        out.append(dict(forward=r.forward, layer=r.layer, copies=r.new, rows=r.rows, ends=dict(cpu=cpu, helper_transfer=helper), lanes=dict(cpu=cpu, primary=None, helper=helper)))
    return out


def deltas_by_layer(rows: list[dict], pairs=PAIRS, lanes=LANES) -> dict:
    """Per layer, mean abs(Δ) of completion times for each pair present in a pass, and slowest minus second slowest lane."""
    out: dict = {}
    for layer in sorted({r['layer'] for r in rows}):
        group = [r for r in rows if r['layer'] == layer]
        entry: dict = dict(passes=len(group))
        for a, b in pairs:
            values = [abs(r['ends'][a] - r['ends'][b]) for r in group if r['ends'].get(a) is not None and r['ends'].get(b) is not None]
            entry[f'{a}|{b}'] = dict(n=len(values), mean=statistics.fmean(values) if values else None)
        gaps, slowest = [], defaultdict(int)
        for r in group:
            present = {k: v for k, v in r['lanes'].items() if k in lanes and v is not None}
            if len(present) >= 2:
                ordered = sorted(present.values(), reverse=True)
                gaps.append(ordered[0] - ordered[1])
                slowest[max(present, key=present.get)] += 1
        entry['slowest_minus_second'] = dict(n=len(gaps), mean=statistics.fmean(gaps) if gaps else None, slowest={k: v / len(gaps) for k, v in slowest.items()} if gaps else {})
        out[layer] = entry
    return out


def overall(rows: list[dict], pairs=PAIRS, lanes=LANES) -> dict:
    """The same measures over all passes together, and the spread of the per-layer means."""
    by_layer = deltas_by_layer(rows, pairs, lanes)
    result = {}
    for key in [f'{a}|{b}' for a, b in pairs] + ['slowest_minus_second']:
        if key == 'slowest_minus_second':
            values = []
            for r in rows:
                present = sorted((v for k, v in r['lanes'].items() if k in lanes and v is not None), reverse=True)
                if len(present) >= 2:
                    values.append(present[0] - present[1])
        else:
            a, b = key.split('|')
            values = [abs(r['ends'][a] - r['ends'][b]) for r in rows if r['ends'].get(a) is not None and r['ends'].get(b) is not None]
        layer_means = [v[key]['mean'] for v in by_layer.values() if v[key]['mean'] is not None and v[key]['n'] >= 5]
        result[key] = dict(n=len(values), mean=statistics.fmean(values) if values else None,
                           layer_min=min(layer_means) if layer_means else None, layer_median=statistics.median(layer_means) if layer_means else None, layer_max=max(layer_means) if layer_means else None)
    return dict(overall=result, by_layer=by_layer)


def trace_summary(rows: list[dict]) -> dict:
    """Which path finished last per pass, grouped by copies, with the slack each other path left."""
    groups: dict[int, list[dict]] = defaultdict(list)
    for r in rows:
        groups[r['copies']].append(r)
    out = {}
    for copies, group in sorted(groups.items()):
        last = defaultdict(int)
        cpu_minus_primary, helper_minus_primary = [], []
        for r in group:
            ends = {k: r[f'{k}_end'] for k in ('cpu', 'helper', 'primary') if r[f'{k}_end'] is not None}
            if ends:
                last[max(ends, key=ends.get)] += 1
            if r['cpu_end'] is not None and r['primary_end'] is not None:
                cpu_minus_primary.append(r['cpu_end'] - r['primary_end'])
            if r['helper_end'] is not None and r['primary_end'] is not None:
                helper_minus_primary.append(r['helper_end'] - r['primary_end'])
        out[copies] = dict(passes=len(group), last={k: v / len(group) for k, v in last.items()}, cpu_minus_primary_ms=quantiles(cpu_minus_primary), helper_minus_primary_ms=quantiles(helper_minus_primary),
                           cpu_end_ms=quantiles([r['cpu_end'] for r in group if r['cpu_end'] is not None]), primary_end_ms=quantiles([r['primary_end'] for r in group if r['primary_end'] is not None]),
                           copy_end_ms=quantiles([r['copy_end'] for r in group if r['copy_end'] is not None]), helper_end_ms=quantiles([r['helper_end'] for r in group if r['helper_end'] is not None]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--capture', action='append', default=[], type=pathlib.Path, help='full light capture artifact')
    ap.add_argument('--model', type=pathlib.Path, help='grouped model JSON, for the same-pass division bound')
    ap.add_argument('--trace', action='append', default=[], help='sqlite:generation_forwards')
    ap.add_argument('--output', required=True, type=pathlib.Path)
    a = ap.parse_args()
    result: dict = dict(captures={}, traces={})
    model = grouped.GroupedTiming.from_dict(json.loads(a.model.read_text())) if a.model else None
    for path in a.capture:
        rows = passes.passes(light.read(path))
        result['captures'][str(path)] = dict(records=records_summary(rows))
        if model:
            result['captures'][str(path)]['division'] = division_summary(rows, model)
    for item in a.trace:
        path, forwards = item.rsplit(':', 1)
        rows = trace_rows(pathlib.Path(path), int(forwards))
        result['traces'][path] = dict(rows=rows, summary=trace_summary(rows))
    a.output.write_text(json.dumps(result, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
