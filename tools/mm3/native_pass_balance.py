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
