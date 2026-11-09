#!/usr/bin/env python3
"""Per-pass lane completion from an Nsight SQLite export (CUDA only) beside the same request's light capture, on one clock.

CLAUDE.md's central objective, per individual pass and layer: the CPU lane against the copy-plus-primary path and the 3090,
and the slowest lane against the second slowest, with completion times from a common origin (the pass's plan: the light
record's plan_end). abs(delta) between two completions does not depend on the origin, so only the clocks must agree.

Clock. Export times plus TARGET_INFO_SESSION_START_TIME.systemClockNs are on Nsight's host timebase, and CUPTI converts
GPU times to it. The light records use the host's steady clock. The anchor is host to host: on each pass with CPU or 3090
work the host writes the word the 5090's wait_publication spins on, launches add_kernel and then stamps the light record's
`end`, so that launch call's end (runtime API record) and the stamp are microseconds apart. A rolling median of their
difference over 401 passes maps every export time to the steady clock (window-trace-20261109: within -2 to +1 us at
p1-p99 around the median, and the median wanders about 25 us over a trace). Mapped times run late by the gap between the
launch's return and the stamp, about 2 us. The launch call's start dates the publication.
The GPU conversion itself was checked against it: wait_publication ends at the launch's start to a median -0.6 us, p10 -34,
p90 +6 us. Passes with CPU or 3090 work pair with wait_publication kernels by order; the counts must match.

Per pass with CPU or 3090 work (generation forwards, from token 128), completion from the pass's plan:
  cpu       the CPU lane's end (light cpu_end), on passes with CPU jobs
  primary   the 5090's arrival at wait_publication: its own expert kernels are done, including those that read the pass's
            demand uploads. On passes with uploads this is the copy-plus-primary path (primary_upload); on passes without,
            resident compute. Counted on passes with primary jobs.
  helper    the 3090's end: the last of its graphs, kernels and copies (the polling occupancy kernel excluded) that start
            after the pass's helper enqueue and end before the host observed it ready. With zero-copy its results reach
            host memory inside the graph, so its transfer and compute paths end together.
  publication  the host's write of the word (the add_kernel launch call's start); the pass cannot finish before it
  uploads   the end of the pass's last demand upload (the 5090's main stream, paired with light demand_copy records by order)

Usage: native_window_paths.py --trace T.sqlite --capture C.light [--label NAME] [--json OUT]   (pairs may repeat)
"""
from __future__ import annotations

import argparse
import bisect
import collections
import json
import pathlib
import sqlite3
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import native_light as light  # noqa: E402
import native_passes as P  # noqa: E402

BIG = 16 << 20
SLACK_NS = 30_000
HALF_WINDOW = 200
LANES = ('cpu', 'primary', 'helper')
PAIRS = (('cpu', 'primary_upload'), ('cpu', 'primary'), ('helper', 'cpu'), ('primary_upload', 'helper'), ('primary', 'helper'))


def read_trace(path: pathlib.Path) -> dict:
    db = sqlite3.connect(str(path))
    names = dict(db.execute('select id, value from StringIds'))
    sysns = db.execute('select systemClockNs from TARGET_INFO_SESSION_START_TIME').fetchone()[0]
    wait_ids = {i for i, v in names.items() if v == 'wait_publication'}
    add_ids = {i for i, v in names.items() if v == 'add_kernel'}
    occupancy_ids = {i for i, v in names.items() if v == 'occupancy_kernel'}
    if not wait_ids:
        raise SystemExit(f'{path}: no wait_publication kernels')
    q = ','.join(map(str, wait_ids))
    placement = collections.Counter(db.execute(f'select deviceId, streamId from CUPTI_ACTIVITY_KIND_KERNEL where shortName in ({q})'))
    primary, main = placement.most_common(1)[0][0]
    api = {c: (s + sysns, e + sysns) for s, e, c in db.execute('select start, end, correlationId from CUPTI_ACTIVITY_KIND_RUNTIME')}
    on_main = list(db.execute('select start, end, shortName, correlationId from CUPTI_ACTIVITY_KIND_KERNEL '
                              'where deviceId = ? and streamId = ? order by start', (primary, main)))
    waits, launches = [], []
    for i, (s, e, n, c) in enumerate(on_main):
        if n not in wait_ids:
            continue
        waits.append((s + sysns, e + sysns))
        j = i + 1
        while j < len(on_main) and on_main[j][2] not in add_ids and on_main[j][2] not in wait_ids:
            j += 1
        launches.append(api.get(on_main[j][3]) if j < len(on_main) and on_main[j][2] in add_ids else None)
    uploads = [(s + sysns, e + sysns) for s, e in db.execute(
        'select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId = ? and streamId = ? and copyKind = 1 and bytes >= ? '
        'order by start', (primary, main, BIG))]
    devices = {d for (d,) in db.execute('select distinct deviceId from CUPTI_ACTIVITY_KIND_KERNEL')}
    helpers = sorted(devices - {primary})
    helper_work: list[tuple[int, int]] = []
    if helpers:
        h = helpers[0]
        helper_work += [(s + sysns, e + sysns) for s, e, n in db.execute(
            'select start, end, shortName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId = ?', (h,)) if n not in occupancy_ids]
        helper_work += [(s + sysns, e + sysns) for s, e in db.execute(
            'select start, end from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId = ?', (h,))]
        tables = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
        if 'CUPTI_ACTIVITY_KIND_GRAPH_TRACE' in tables:
            helper_work += [(s + sysns, e + sysns) for s, e in db.execute(
                'select start, end from CUPTI_ACTIVITY_KIND_GRAPH_TRACE where deviceId = ?', (h,))]
    helper_work.sort()
    return dict(waits=waits, launches=launches, uploads=uploads, helper=helper_work, primary_device=primary, main_stream=main,
                helper_device=helpers[0] if helpers else None)


def clock_map(launches: list, stamps: list[int]) -> tuple:
    """Correction (export minus steady clock, ns) as a function of export time, from the rolling median of the add launch's
    API end against the light end stamp; and the fit's statistics."""
    at, diff = [], []
    for launch, stamp in zip(launches, stamps):
        if launch is not None:
            at.append(launch[1])
            diff.append(launch[1] - stamp)
    if len(diff) < 50:
        raise SystemExit('too few add_kernel launches to anchor the clock')
    at_a, diff_a = np.array(at, dtype=float), np.array(diff, dtype=float)
    med = np.array([np.median(diff_a[max(0, i - HALF_WINDOW):i + HALF_WINDOW + 1]) for i in range(len(diff_a))])
    residual = (diff_a - med) / 1e3

    def correction(t):
        return np.interp(t, at_a, med)
    return correction, dict(anchored_passes=len(diff), median_us=float(np.median(diff_a) / 1e3),
                            wander_us=[round(float(med.min() / 1e3), 1), round(float(med.max() / 1e3), 1)],
                            residual_us=[round(float(x), 2) for x in np.percentile(residual, [1, 10, 50, 90, 99])])


def passes_of(trace: dict, capture: light.LightCapture, measure_from: int = 128) -> tuple[list[dict], dict]:
    layers = capture.of('layer')
    pubs = sorted((r['t_ns'] + r['end'], i) for i, r in enumerate(layers)
                  if r['end'] and (r['cpu_end'] or r['helper_submit_end']))
    if len(pubs) != len(trace['waits']):
        raise SystemExit(f"{len(pubs)} light publications against {len(trace['waits'])} wait_publication kernels")
    correction, clock = clock_map(trace['launches'], [p for p, _ in pubs])

    def host(ns: float) -> float:
        return float(ns - correction(ns))

    demand = sorted(capture.of('demand_copy'), key=lambda r: r['t_ns'])
    upload_end: dict[tuple[int, int], float] = {}
    if len(demand) == len(trace['uploads']):
        for r, (_, e) in zip(demand, trace['uploads']):
            key = (r['forward'], r['layer'])
            upload_end[key] = max(upload_end.get(key, 0.0), host(e))
        clock['uploads_paired'] = len(demand)
    else:
        clock['uploads_paired'] = 0
        clock['uploads_unpaired'] = dict(light=len(demand), trace=len(trace['uploads']))
    work = {(p.forward, p.layer): p for p in P.passes(capture)}
    begin = capture.of('generation_begin')
    first = begin[0]['position'] if begin else 0
    positions = {f['forward']: f['position'] for f in capture.of('forward')}
    helper_host = [(host(s), host(e)) for s, e in trace['helper']]
    hs = [s for s, _ in helper_host]
    out = []
    for (stamp, i), (ws, we), launch in zip(pubs, trace['waits'], trace['launches']):
        r = layers[i]
        p = work.get((r['forward'], r['layer']))
        if p is None or p.final_layer or positions.get(r['forward'], 0) < first + measure_from:
            continue
        base = r['t_ns'] + r['plan_end']
        ms = lambda t: (t - base) / 1e6
        ends: dict[str, float | None] = dict(cpu=None, primary=None, primary_upload=None, helper=None)
        if r['cpu_end'] and p.cpu_jobs:
            ends['cpu'] = ms(r['t_ns'] + r['cpu_end'])
        arrival = host(ws)
        if p.primary_jobs:
            ends['primary'] = ms(arrival)
            if (r['forward'], r['layer']) in upload_end:
                ends['primary_upload'] = ends['primary']
        if r['helper_submit_end'] and p.helper_jobs:
            lo = r['t_ns'] + (r['helper_enqueue'] or r['helper_submit_end']) - SLACK_NS
            seen = r['helper_ready_observed'] or r['collect_end']
            hi = r['t_ns'] + (seen if seen else r['end']) + SLACK_NS
            j = bisect.bisect_left(hs, lo)
            last = None
            while j < len(hs) and hs[j] <= hi:
                e = helper_host[j][1]
                if e <= hi:
                    last = e if last is None else max(last, e)
                j += 1
            if last is not None:
                ends['helper'] = ms(last)
        up = upload_end.get((r['forward'], r['layer']))
        out.append(dict(forward=r['forward'], layer=r['layer'], rows=p.rows, cpu_jobs=p.cpu_jobs, helper_jobs=p.helper_jobs,
                        primary_jobs=p.primary_jobs, new=p.new, cycle_ms=p.cycle_ms, ends=ends,
                        uploads_end=None if up is None else ms(up), wait_ms=(we - ws) / 1e6, wait_end=ms(host(we)),
                        publication=ms(host(launch[0])) if launch is not None else None))
    return out, clock


def summarize(rows: list[dict]) -> dict:
    by_rows: dict = {}
    names = [f'{a}|{b}' for a, b in PAIRS] + ['slowest_minus_second']
    for k in sorted({min(r['rows'], 8) for r in rows}):
        group = [r for r in rows if min(r['rows'], 8) == k]
        last = collections.Counter()
        values = {name: [] for name in names}
        per_layer: dict = collections.defaultdict(lambda: collections.defaultdict(list))
        after_last, spins = [], []
        for r in group:
            e = r['ends']
            present = {n: e[n] for n in LANES if e[n] is not None}
            if len(present) >= 2:
                order = sorted(present.values())
                last[max(present, key=present.get)] += 1
                values['slowest_minus_second'].append(order[-1] - order[-2])
                per_layer[r['layer']]['slowest_minus_second'].append(order[-1] - order[-2])
            for a, b in PAIRS:
                if e[a] is not None and e[b] is not None:
                    values[f'{a}|{b}'].append(abs(e[a] - e[b]))
                    per_layer[r['layer']][f'{a}|{b}'].append(abs(e[a] - e[b]))
            if present and r['publication'] is not None:
                host_lanes = [v for n, v in present.items() if n != 'primary']
                if host_lanes:
                    after_last.append(r['publication'] - max(host_lanes))
            spins.append(r['wait_ms'])
        n_last = sum(last.values())
        layer_means = {name: [statistics.fmean(v[name]) for v in per_layer.values() if v.get(name)] for name in names}
        lane_ends = {n: [r['ends'][n] for r in group if r['ends'][n] is not None] for n in LANES}
        by_rows[k] = dict(
            passes=len(group), cycle_ms=sum(r['cycle_ms'] for r in group),
            last={n: c / n_last for n, c in last.most_common()} if n_last else {},
            mean={name: (statistics.fmean(v) if v else None) for name, v in values.items()},
            passes_with={name: len(v) for name, v in values.items()},
            per_layer_range={name: ([min(v), max(v)] if v else None) for name, v in layer_means.items()},
            per_layer={int(layer): {name: statistics.fmean(v) for name, v in d.items() if v} for layer, d in sorted(per_layer.items())},
            lanes_ms={n: (statistics.median(v) if v else None) for n, v in lane_ends.items()},
            publication_after_host_lanes_ms=(statistics.median(after_last) if after_last else None),
            wait_ms=statistics.median(spins) if spins else None)
    return by_rows


def report(label: str, by_rows: dict, clock: dict) -> list[str]:
    lines = [f"{label}: clock anchored on {clock['anchored_passes']} passes, median {clock['median_us']:.1f} us, wander "
             f"{clock['wander_us'][0]}..{clock['wander_us'][1]} us, residual p1-p99 {clock['residual_us'][0]}..{clock['residual_us'][-1]} us; "
             f"uploads paired {clock.get('uploads_paired')}"]

    def f(x):
        return '   -  ' if x is None else f'{x:6.3f}'

    def span(r):
        return '-' if r is None else f'{r[0]:.2f}-{r[1]:.2f}'

    for k, v in by_rows.items():
        lanes = v['lanes_ms']
        lines.append(f"  rows {k}: {v['passes']:6d} passes; last " + ', '.join(f'{n} {100 * s:.0f}%' for n, s in v['last'].items()) +
                     f"; median end ms cpu {f(lanes['cpu'])} 5090 {f(lanes['primary'])} 3090 {f(lanes['helper'])}; publication after "
                     f"the host lanes {f(v['publication_after_host_lanes_ms'])}; 5090 wait {f(v['wait_ms'])}")
        lines.append('     mean abs(delta) ms: ' + ', '.join(
            f"{name} {f(m)} (n {v['passes_with'][name]}, layers {span(v['per_layer_range'][name])})" for name, m in v['mean'].items()))
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--trace', action='append', required=True, type=pathlib.Path)
    ap.add_argument('--capture', action='append', required=True, type=pathlib.Path)
    ap.add_argument('--label', action='append')
    ap.add_argument('--json', type=pathlib.Path)
    ap.add_argument('--measure-from', type=int, default=128)
    a = ap.parse_args()
    if len(a.trace) != len(a.capture):
        ap.error('one --capture per --trace')
    labels = a.label or [t.stem for t in a.trace]
    result = {}
    for label, t, c in zip(labels, a.trace, a.capture):
        rows, clock = passes_of(read_trace(t), light.read(c), a.measure_from)
        by_rows = summarize(rows)
        print('\n'.join(report(label, by_rows, clock)), flush=True)
        result[label] = dict(clock=clock, by_rows=by_rows)
    if a.json:
        a.json.write_text(json.dumps(result, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
