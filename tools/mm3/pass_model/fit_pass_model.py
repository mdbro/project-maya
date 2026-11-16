#!/usr/bin/env python3
"""Fit the bandwidth-aware pass model's constants on light captures, check them on held-out captures, and write them.

    python3 fit_pass_model.py --train 'GLOB' [--train 'GLOB' ...] --check NAME='GLOB' [--check ...] --out DIR
           [--calibration JSON] [--header src/mm3/mm3_pass_model_fit.hpp]

Inputs are format-2 light captures (corrected 5090 stamps; passes at token 128 and later). Every constant is a measured
rate or a fixed cost estimated from the passes where it is isolated (pass_model.py names the terms):
  copy_start     median first-copy submission after the plan's end, by path (1: one-row passes, 0: windows)
  gpu_start      median 5090 upload stamp on passes with no copy, by rows (when the 5090 reaches its experts)
  copy_gbps      record bytes over the median per-copy time on passes with copies and no CPU job, by path
  cpu_*          the CPU lane alone (passes with a CPU job and no copy): fixed, per expert (as GB/s), per extra column
  host_gbps      the ceiling the CPU lane, uploads and downloads share: least squares over windows with a CPU job and
                 copies and no exchange in flight
  exchange_*     a pending exchange's upload and download records: least squares over windows with exchanges in flight
  early_draw     the early copy's host-memory draw at one row: least squares over one-row passes with a CPU job
  reuse          an upload's later uses from its frame within 8 forwards, by path (counted)
  gpu_*          the 5090's compute (expert end - upload stamp): base by rows, per expert (as GB/s), per pick, per round
  helper_*       the 3090's publication: base by rows, per expert (as GB/s), per column, lease/exchange pending
  third_*        the 4080's publication: base by rows, per expert (as GB/s), per column
  publish        median of the 5090's wait end - the last host lane, by rows, where a host lane ended last
Least squares drop the worst 2% of residuals and solve again. Rows with too few passes take the nearest fitted row.

Writes DIR/pass-model-fit.json (the Rates), DIR/pass-model-fit.txt (the same for maya-mm3 --pass-model-fit),
DIR/pass-model-check.txt/.json (errors by lane and mode, measured against
modeled) and, with --header, the C++ constants and golden cases for tests/mm3_pass_model_test.cpp. With --calibration
(an engine start's "grouped staging calibration" JSON) the check also prices each pass the way today's planner does
(CPU lane x the CPU price scale, --cpu-scale, default 1.5) for comparison."""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import pathlib
import statistics
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass_data  # noqa: E402
import pass_model as pm  # noqa: E402

MIN_ROWS = 40


def median(v):
    return statistics.median(v) if v else None


def lstsq_trim(X, y, trim=0.02):
    X, y = np.asarray(X, float), np.asarray(y, float)
    coef = np.linalg.lstsq(X, y, rcond=None)[0]
    res = np.abs(X @ coef - y)
    keep = res <= np.quantile(res, 1 - trim)
    return np.linalg.lstsq(X[keep], y[keep], rcond=None)[0]


def by_rows_table(values: dict[int, float], default: float) -> list[float]:
    table = [0.0] * pm.ROWS
    known = sorted(values)
    for t in range(1, pm.ROWS):
        if t in values:
            table[t] = values[t]
        elif known:
            table[t] = values[min(known, key=lambda k: (abs(k - t), -k))]
        else:
            table[t] = default
    return table


def bounded(values: list[dict], maximum: int = 512) -> list[dict]:
    """Deterministic bounded calibration cost; validation always retains the full population."""
    if len(values) <= maximum:
        return values
    return [values[int(i)] for i in np.linspace(0, len(values)-1, maximum)]


def fit(rows: list[dict], gpu_gbps: float, d2h_gbps: float) -> tuple[pm.Rates, dict]:
    r = pm.Rates()
    r.gpu_gbps = gpu_gbps
    r.d2h_gbps = d2h_gbps
    mb = r.record_bytes * 1e-6
    notes = {}
    path = lambda x: 1 if x['rows'] == 1 else 0
    # copies: their start and their rate alone, by path
    for p in (0, 1):
        s = [x['first_copy_submit'] for x in rows if x['uploads'] and x['first_copy_submit'] is not None and path(x) == p]
        r.copy_start[p] = median(s) if s else 0.0
    starts, ready = collections.defaultdict(list), collections.defaultdict(list)
    for x in rows:
        if x['gpu_ok'] and x['upload'] is not None and not x['uploads']:
            (ready if x['g_ready'] else starts)[x['rows']].append(x['upload'])
    r.gpu_start = by_rows_table({t: median(v) for t, v in starts.items() if len(v) >= MIN_ROWS}, 0.03)
    r.gpu_start_ready = by_rows_table({t: median(v) for t, v in ready.items() if len(v) >= MIN_ROWS}, 0.03)
    r.gpu_start_ready = [max(a_, b_) for a_, b_ in zip(r.gpu_start_ready, r.gpu_start)]
    for p in (0, 1):
        per = [(x['upload'] - x['first_copy_submit']) / x['uploads'] for x in rows
               if x['gpu_ok'] and x['upload'] is not None and x['uploads'] and not x['cpu_experts'] and not pm.load_of(x).exchanges and path(x) == p and
               x['first_copy_submit'] is not None and x['upload'] > x['first_copy_submit'] and
               x['upload'] > pm._row(r.gpu_start_ready if x['g_ready'] else r.gpu_start, x['rows']) + 0.05]
        if len(per) >= MIN_ROWS:
            r.copy_gbps[p] = mb / median(per)
        notes[f'copy_path{p}_passes'] = len(per)
    # the CPU lane: its start (after the primary submission) and its work alone
    cs = collections.defaultdict(list)
    for x in rows:
        if x['cpu_experts'] and x['primary_submit'] is not None:
            cs[x['rows']].append(x['primary_submit'])
    r.cpu_start = by_rows_table({t: median(v) for t, v in cs.items() if len(v) >= MIN_ROWS // 2}, 0.04)
    # (alone: no copy and no exchange in flight; nearly all hold one expert, so its time is the per-expert work and the
    #  fixed part stays 0: the passes cannot separate them; extra columns by a slope through the origin)
    alone = [x for x in rows if x['cpu_experts'] and not x['uploads'] and not x['exchanges_pending'] and x['cpu'] is not None]
    if len(alone) >= MIN_ROWS:
        one = [x['cpu'] - pm._row(r.cpu_start, x['rows']) for x in alone if x['cpu_experts'] == 1]
        per_expert = median(one)
        r.cpu_fixed = 0.0
        r.cpu_gbps = mb / per_expert
        ex = [(x['cpu_columns'] - x['cpu_experts'], x['cpu'] - pm._row(r.cpu_start, x['rows']) - x['cpu_experts'] * per_expert) for x in alone]
        num = sum(a_ * b_ for a_, b_ in ex); den = sum(a_ * a_ for a_, _ in ex)
        r.cpu_column = max(0.0, num / den) if den else 0.0
    notes['cpu_alone_passes'] = len(alone)
    # host memory: its ceiling from the windows with a CPU job and copies and no exchange in flight; then the pending
    # exchanges' upload and download records from the windows with them (the download rate given, d2h_gbps)
    def host_err(sel):
        err = 0.0
        for x in sel:
            f = pm.predict(pm.load_of(x), r)
            err += (f.cpu - x['cpu']) ** 2
            if x['uploads'] and x['upload'] is not None and x['upload'] > pm._row(r.gpu_start, x['rows']) + 0.05:
                err += (f.copies - x['upload']) ** 2
        return err
    both = [x for x in rows if x['cpu_experts'] and x['uploads'] and not pm.load_of(x).exchanges and x['cpu'] is not None and x['upload'] is not None and x['gpu_ok'] and path(x) == 0]
    if len(both) >= MIN_ROWS:
        best = None
        for b_ in np.arange(50.0, 120.01, 1.0):
            r.host_gbps = float(b_)
            e = host_err(bounded(both))
            best = (e, float(b_)) if best is None or e < best[0] else best
        for b_ in np.arange(best[1] - 1.0, best[1] + 1.001, 0.1):
            r.host_gbps = float(b_)
            e = host_err(bounded(both))
            best = (e, float(b_)) if e < best[0] else best
        r.host_gbps = best[1]
    notes['shared_passes'] = len(both)
    ex = [x for x in rows if pm.load_of(x).exchanges and (x['cpu_experts'] or x['uploads']) and x['gpu_ok'] and
          (x['cpu'] is not None or not x['cpu_experts']) and x['upload'] is not None and path(x) == 0]
    if len(ex) >= MIN_ROWS:
        ex_fit = bounded(ex, 256)
        best = None
        for u_ in np.arange(0.0, 1.501, 0.05):
            for d_ in np.arange(0.0, 2.001, 0.1):
                r.exchange_up, r.exchange_down = float(u_), float(d_)
                e = host_err([x for x in ex_fit if x['cpu_experts']]) + sum(
                    (pm.predict(pm.load_of(x), r).copies - x['upload']) ** 2 for x in ex_fit
                    if not x['cpu_experts'] and x['uploads'] and x['upload'] > pm._row(r.gpu_start, x['rows']) + 0.05)
                best = (e, float(u_), float(d_)) if best is None or e < best[0] else best
        r.exchange_up, r.exchange_down = best[1], best[2]
    notes['exchange_passes'] = len(ex)
    # the early copy's draw at one row: from the one-row passes with a CPU job (the CPU lane's end)
    # (the one-row copy rate was measured with the early copy running beside it: it is the rate after host memory's share,
    #  so the rate alone follows from it and the early copy's draw - r x host / (r + early) = measured - and both are
    #  solved together, twice)
    one = [x for x in rows if x['rows'] == 1 and x['cpu_experts'] and x['cpu'] is not None]
    measured1 = r.copy_gbps[1]
    if len(one) >= MIN_ROWS:
        for _ in range(2):
            best = None
            for e_ in np.arange(0.0, 80.01, 1.0):
                r.early_draw = float(e_)
                err = sum((pm.predict(pm.load_of(x), r).cpu - x['cpu']) ** 2 for x in bounded(one))
                best = (err, float(e_)) if best is None or err < best[0] else best
            r.early_draw = best[1]
            if measured1 < r.host_gbps:
                alone = measured1 * r.early_draw / (r.host_gbps - measured1)
                r.copy_gbps[1] = alone if alone + r.early_draw > r.host_gbps else measured1
    notes['early_passes'] = len(one)
    # the 5090's compute after its start
    # (per expert: the record at the measured read rate, gpu_gbps, given; the rest fitted)
    g = []
    split_ambiguous = 0
    for x in rows:
        if not x['gpu_ok'] or x['g'] is None or x['upload'] is None:
            continue
        z = dict(x)
        split = x['overlap'] and 0 < x['late_experts'] < x['g_experts']
        if split:
            # The measured copy endpoint does not mark ready-compute completion. Fit late-group work only when
            # the conservative ready estimate ends well before it; validate every split pass separately afterwards.
            ready_work = ((x['g_experts']-x['late_experts'])*mb/r.gpu_gbps +
                          (x['g_picks']-x['late_picks'])*r.gpu_pick + pm._row(r.gpu_base,x['rows']))
            if x['upload'] < pm._row(r.gpu_start,x['rows']) + ready_work + .05:
                split_ambiguous += 1
                continue
            z['compute_experts'], z['compute_picks'] = x['late_experts'], x['late_picks']
        else:
            z['compute_experts'], z['compute_picks'] = x['g_experts'], x['g_picks']
        g.append(z)
    tr = sorted({x['rows'] for x in g})
    X = [[float(x['rows'] == t) for t in tr] + [x['compute_picks'], max(0, math.ceil(x['compute_experts'] / x['gpu_rounds_of']) - 1)] for x in g]
    y = [x['g'] - x['upload'] - x['compute_experts'] * mb / r.gpu_gbps for x in g]
    c = lstsq_trim(X, y)
    r.gpu_base = by_rows_table({t: max(0.0,float(c[i])) for i, t in enumerate(tr)}, 0.05)
    r.gpu_pick, r.gpu_round = max(0.0, c[len(tr)]), max(0.0, c[len(tr) + 1])
    notes['gpu_compute_passes'] = len(g)
    notes['gpu_compute_split_ambiguous_omitted_from_fit_only'] = split_ambiguous
    notes['gpu_observed_widths'] = tr
    notes['gpu_unobserved_widths_nearest_estimate'] = sorted(set(range(1,pm.ROWS))-set(tr))
    # the 3090 and the 4080
    for lane, ek, ck, base, extra in (('helper', 'helper_experts', 'helper_columns', 'helper_base', True),
                                       ('third', 'third_experts', 'third_columns', 'third_base', False)):
        sel = [x for x in rows if x[ek] and x[lane] is not None]
        tr = sorted({x['rows'] for x in sel})
        X = [[float(x['rows'] == t) for t in tr] + [x[ek], x[ck]] + ([1.0 if x['helper_lease'] else 0.0] if extra else [])
             for x in sel]
        y = [x[lane] for x in sel]
        c = lstsq_trim(X, y)
        setattr(r, base, by_rows_table({t: max(0.0,float(c[i])) for i, t in enumerate(tr)}, 0.1))
        if c[len(tr)] > 0:
            setattr(r, f'{lane}_gbps', mb / c[len(tr)])
        setattr(r, f'{lane}_column', max(0.0, float(c[len(tr) + 1])))
        if extra:
            r.helper_lease = max(0.0, float(c[len(tr) + 2]))
        notes[f'{lane}_passes'] = len(sel)
    # the publication after the last host lane
    pub = collections.defaultdict(list)
    for x in rows:
        host = [v for v in (x['cpu'], x['helper'], x['third']) if v is not None]
        if x['gpu_ok'] and host and x['wait'] is not None and max(host) > x['g'] + 0.05:
            pub[x['rows']].append(x['wait'] - max(host))
    r.publish = by_rows_table({t: max(0.0, median(v)) for t, v in pub.items() if len(v) >= MIN_ROWS}, 0.02)
    return r, notes


def today(x: dict, cal: dict, scale: float) -> dict:
    """Today's planner's lane prices for the pass (mm3_batch_staging.hpp), as finishes after the plan's end, given the
    measured submission ends (an advantage to it): CPU from the primary submission, 5090 from it too, 3090 from its own."""
    n = lambda cols: min(16, max(1, cols))
    out = {}
    if x['cpu_experts'] and x['primary_submit'] is not None:
        out['cpu'] = x['primary_submit'] + scale * (cal['cpu_base_ms'] + sum(cal['cpu_extra_ms'][n(u)] for u in x['cpu_uses']))
    if x['primary_submit'] is not None:
        out['gpu'] = x['primary_submit'] + cal['gpu_base_ms'][min(16, x['rows'])] + sum(cal['gpu_extra_ms'][n(u)] for u in x['gpu_uses']+x['new_uses']) + x['uploads'] * cal['h2d_ms']
    if x['helper_experts'] and x['helper_submit'] is not None:
        out['helper'] = x['helper_submit'] + cal['helper_base_ms'] + sum(cal['helper_extra_ms'][n(u)] for u in x['helper_uses'])
    return out


def errors(pairs):
    if not pairs:
        return dict(n=0)
    d = np.array([p - m for p, m in pairs])
    a = np.abs(d)
    meas = np.array([m for _, m in pairs])
    return dict(n=len(d), bias=float(d.mean()), mae=float(a.mean()), median_abs=float(np.median(a)), p90_abs=float(np.quantile(a, .9)),
                measured_mean=float(meas.mean()), relative_mae=float(a.mean() / max(1e-9, meas.mean())))


def check(rows: list[dict], r: pm.Rates, cal: dict | None, scale: float) -> dict:
    out = {}
    for mode, sel in (('drafting', [x for x in rows if x['mode'] == 'drafting']), ('one row', [x for x in rows if x['mode'] == 'one row'])):
        if not sel:
            continue
        lanes = collections.defaultdict(list)
        old = collections.defaultdict(list)
        last_ok = last_n = 0
        last_old_ok = last_old_n = 0
        for x in sel:
            f = pm.predict(pm.load_of(x), r)
            if x['cpu_experts'] and x['cpu'] is not None: lanes['cpu'].append((f.cpu, x['cpu']))
            if x['uploads'] and x['gpu_ok'] and x['upload'] is not None: lanes['copies'].append((max(f.copies, pm._row(r.gpu_start, x['rows'])), x['upload']))
            if x['gpu_ok']: lanes['5090'].append((f.gpu, x['g']))
            if x['helper_experts'] and x['helper'] is not None: lanes['3090'].append((f.helper, x['helper']))
            if x['third_experts'] and x['third'] is not None: lanes['4080'].append((f.third, x['third']))
            if x['gpu_ok'] and x['wait'] is not None: lanes['pass'].append((f.passed, x['wait']))
            meas = {k: v for k, v in (('cpu', x['cpu']), ('3090', x['helper']), ('4080', x['third']), ('5090', x['g'] if x['gpu_ok'] else None)) if v is not None}
            pred = {k: v for k, v in (('cpu', f.cpu), ('3090', f.helper), ('4080', f.third), ('5090', f.gpu)) if v is not None}
            if x['gpu_ok'] and len(meas) >= 2:
                last_n += 1
                last_ok += max(meas, key=meas.get) == max(pred, key=pred.get)
            if cal:
                t = today(x, cal, scale)
                if 'cpu' in t and x['cpu'] is not None: old['cpu'].append((t['cpu'], x['cpu']))
                if 'gpu' in t and x['gpu_ok']: old['5090'].append((t['gpu'], x['g']))
                if 'helper' in t and x['helper'] is not None: old['3090'].append((t['helper'], x['helper']))
                m3 = {k: v for k, v in (('cpu', x['cpu']), ('3090', x['helper']), ('5090', x['g'] if x['gpu_ok'] else None)) if v is not None}
                p3 = {k: t[{'cpu': 'cpu', '3090': 'helper', '5090': 'gpu'}[k]] for k in m3 if {'cpu': 'cpu', '3090': 'helper', '5090': 'gpu'}[k] in t}
                if x['gpu_ok'] and len(m3) >= 2 and set(p3) == set(m3):
                    last_old_n += 1
                    last_old_ok += max(m3, key=m3.get) == max(p3, key=p3.get)
        out[mode] = dict(passes=len(sel), lanes={k: errors(v) for k, v in lanes.items()},
                         last_lane=dict(n=last_n, agree=last_ok / max(1, last_n)),
                         today=dict(lanes={k: errors(v) for k, v in old.items()}, last_lane_of_three=dict(n=last_old_n, agree=last_old_ok / max(1, last_old_n))) if cal else None)
    return out


def load_rows(patterns: list[str]) -> list[dict]:
    rows = []
    for pat in patterns:
        for p in sorted(glob.glob(pat)):
            if '-pe1-' in p and 'pe1' not in pat:
                continue
            rows.extend(pass_data.as_dicts(pass_data.rows_of(p)))
    return rows


def header(r: pm.Rates, cases: list[tuple[pm.Load, pm.Finish]], source: str) -> str:
    arr = lambda v: '{' + ', '.join(f'{float(a):.17g}' for a in v) + '}'
    lines = ['// Generated by tools/mm3/pass_model/fit_pass_model.py - do not edit by hand. The bandwidth-aware pass model\'s',
             f'// constants fitted on {source}, and golden cases the C++ model must reproduce (tests/mm3_pass_model_test.cpp).',
             '#pragma once', '#include "mm3_pass_model.hpp"', '', 'namespace mm3::pass_model {',
             'inline Rates fitted_rates() {', '    Rates r;']
    for k, v in r.to_dict().items():
        if isinstance(v, list):
            lines.append(f'    {{ const double v[] = {arr(v)}; for (size_t i = 0; i < sizeof v / sizeof v[0]; ++i) r.{k}[i] = v[i]; }}')
        else:
            lines.append(f'    r.{k} = {float(v):.17g};')
    lines += ['    return r;', '}', 'struct GoldenCase { Load load; double cpu, copies, gpu, helper, third, passed; };',
              'inline const GoldenCase* golden_cases(size_t& n) {', '    static const GoldenCase cases[] = {']
    nan = lambda v: 'kNone' if v is None else f'{v:.17g}'
    for ld, f in cases:
        lines.append(f'        {{{{{ld.rows}, {ld.cpu_experts}, {ld.cpu_columns}, {ld.copies}, {ld.exchanges}, {ld.ready}, {ld.early}, {ld.gpu_experts}, {ld.gpu_picks}, '
                     f'{ld.helper_experts}, {ld.helper_columns}, {ld.helper_lease}, {ld.third_experts}, {ld.third_columns}, {ld.gpu_rounds_of}, '
                     f'{str(ld.overlap).lower()}, {ld.late_experts}, {ld.late_picks}, {ld.exchange_up_records}, {ld.exchange_down_records}}}, '
                     f'{nan(f.cpu)}, {nan(f.copies)}, {nan(f.gpu)}, {nan(f.helper)}, {nan(f.third)}, {nan(f.passed)}}},')
    lines += ['    };', '    n = sizeof cases / sizeof cases[0];', '    return cases;', '}', '}  // namespace mm3::pass_model', '']
    return '\n'.join(lines)


def rates_text(r: pm.Rates, source: str) -> str:
    """The constants as maya-mm3 reads them (--pass-model-fit FILE; mm3_pass_model.hpp load_rates): a name and its values a line."""
    lines = [f'# the bandwidth-aware pass model\'s constants (tools/mm3/pass_model/fit_pass_model.py), fitted on {source}']
    for k, v in r.to_dict().items():
        vals = v if isinstance(v, list) else [v]
        lines.append(k + ' ' + ' '.join(f'{float(x):.17g}' for x in vals))
    return '\n'.join(lines) + '\n'


def golden(r: pm.Rates) -> list[tuple[pm.Load, pm.Finish]]:
    loads = [pm.Load(rows=1, gpu_experts=2, gpu_picks=2, helper_experts=1, helper_columns=1),
             pm.Load(rows=1, gpu_experts=2, gpu_picks=2, ready=1, helper_experts=1, helper_columns=1, helper_lease=1),
             pm.Load(rows=1, copies=1, gpu_experts=3, gpu_picks=3, third_experts=1, third_columns=1),
             pm.Load(rows=1, cpu_experts=1, cpu_columns=1, copies=1, early=1, gpu_experts=3, gpu_picks=3, helper_experts=1, helper_columns=1),
             pm.Load(rows=1, cpu_experts=2, cpu_columns=2, early=1, gpu_experts=2, gpu_picks=2),
             pm.Load(rows=2, cpu_experts=1, cpu_columns=1, exchanges=1, gpu_experts=2, gpu_picks=3, helper_experts=2, helper_columns=2),
             pm.Load(rows=5, cpu_experts=2, cpu_columns=2, copies=1, exchanges=2, gpu_experts=5, gpu_picks=9, helper_experts=4, helper_columns=6),
             pm.Load(rows=4, cpu_experts=1, cpu_columns=2, copies=1, gpu_experts=4, gpu_picks=7, helper_experts=4, helper_columns=5, helper_lease=1, third_experts=2, third_columns=2),
             pm.Load(rows=8, cpu_experts=2, cpu_columns=3, copies=3, gpu_experts=7, gpu_picks=14, helper_experts=5, helper_columns=9, third_experts=3, third_columns=4),
             pm.Load(rows=8, cpu_experts=3, cpu_columns=3, gpu_experts=6, gpu_picks=12, helper_experts=6, helper_columns=12, third_experts=3, third_columns=5, gpu_rounds_of=32),
             pm.Load(rows=3, cpu_experts=1, cpu_columns=1, gpu_experts=0, gpu_picks=0)]
    return [(ld, pm.predict(ld, r)) for ld in loads]


def text(checks: dict, r: pm.Rates, notes: dict) -> str:
    L = ['The bandwidth-aware pass model: fitted constants (measured rates and fixed costs) and its error against measured',
         'finishes (ms after the plan\'s end; modeled minus measured). Passes at token 128 and later, corrected 5090 stamps.', '']
    L.append('Fitted rates (record bytes a ms of each lane\'s own work):')
    L.append(f'  copies alone          {r.copy_gbps[0]:.1f} GB/s in windows of 2+ rows, {r.copy_gbps[1]:.1f} GB/s at one row; start {r.copy_start[0]:.3f} / {r.copy_start[1]:.3f} ms')
    L.append(f'  CPU lane alone        {r.cpu_gbps:.1f} GB/s, fixed {r.cpu_fixed:.3f} ms, {r.cpu_column:.4f} ms an extra column')
    L.append(f'  host memory           {r.host_gbps:.1f} GB/s shared by what runs (CPU lane alone and an upload alone together would draw '
             f'{r.cpu_gbps + r.copy_gbps[0]:.1f}: each then at {100 * min(1.0, r.host_gbps / (r.cpu_gbps + r.copy_gbps[0])):.0f}%)')
    L.append(f'  exchanges in flight   {r.exchange_up:.2f} record of upload ahead of a window\'s copies and {r.exchange_down:.2f} of download at '
             f'{r.d2h_gbps:.0f} GB/s (given) each')
    L.append(f'  early copy (one row)  draws {r.early_draw:.0f} GB/s of host memory all pass long')
    L.append(f'  reuse                 an upload is read again from its frame {r.reuse[0]:.2f} times (windows) / {r.reuse[1]:.2f} (one row) within 8 forwards')
    L.append(f'  5090 experts          {r.gpu_gbps:.0f} GB/s (given: the bench\'s read rate), {1000 * r.gpu_pick:.1f} us a pick, {1000 * r.gpu_round:.1f} us an extra round of 4')
    L.append(f'  3090 experts          {r.helper_gbps:.0f} GB/s, {1000 * r.helper_column:.1f} us a column, +{1000 * r.helper_lease:.1f} us while it holds a lease')
    L.append(f'  4080 experts          {r.third_gbps:.0f} GB/s, {1000 * r.third_column:.1f} us a column')
    fmt_row = lambda name, t: ' '.join(f'{v:.3f}' for v in t[1:9])
    for name in ('gpu_start', 'gpu_start_ready', 'gpu_base', 'cpu_start', 'helper_base', 'third_base', 'publish'):
        L.append(f'  {name:<20}  rows 1-8: {fmt_row(name, getattr(r, name))} ms')
    L.append(f'  passes used: {notes}')
    L.append('')
    for name, c in checks.items():
        L.append(f'== {name}')
        for mode, m in c.items():
            L.append(f'  {mode}: {m["passes"]} passes; the last lane named right on {100 * m["last_lane"]["agree"]:.0f}% of {m["last_lane"]["n"]} passes with 2+ lanes')
            L.append(f'    {"lane":<8} {"passes":>7} {"measured mean":>14} {"bias":>8} {"mean abs":>9} {"median abs":>11} {"p90 abs":>8}   (ms)')
            for lane in ('cpu', 'copies', '5090', '3090', '4080', 'pass'):
                e = m['lanes'].get(lane)
                if not e or not e['n']:
                    continue
                L.append(f'    {lane:<8} {e["n"]:>7} {e["measured_mean"]:>14.3f} {e["bias"]:>+8.3f} {e["mae"]:>9.3f} {e["median_abs"]:>11.3f} {e["p90_abs"]:>8.3f}')
            if m.get('today'):
                L.append(f'    today\'s planner prices (CPU price {SCALE}; given the measured submissions); its last of three lanes named right on '
                         f'{100 * m["today"]["last_lane_of_three"]["agree"]:.0f}% of {m["today"]["last_lane_of_three"]["n"]}:')
                for lane in ('cpu', '5090', '3090'):
                    e = m['today']['lanes'].get(lane)
                    if not e or not e['n']:
                        continue
                    L.append(f'    {lane:<8} {e["n"]:>7} {e["measured_mean"]:>14.3f} {e["bias"]:>+8.3f} {e["mae"]:>9.3f} {e["median_abs"]:>11.3f} {e["p90_abs"]:>8.3f}')
        L.append('')
    return '\n'.join(L)


SCALE = 1.5


def main() -> int:
    global SCALE
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--train', action='append', required=True)
    ap.add_argument('--check', action='append', default=[], help='NAME=GLOB (repeatable); GLOBs separated by ,')
    ap.add_argument('--out', required=True, type=pathlib.Path)
    ap.add_argument('--calibration', type=pathlib.Path)
    ap.add_argument('--cpu-scale', type=float, default=1.5)
    ap.add_argument('--header', type=pathlib.Path)
    ap.add_argument('--gpu-gbps', type=float, default=1679.0, help="the 5090's measured record read rate (kernel item 7's bench, 2026-11-14)")
    ap.add_argument('--d2h-gbps', type=float, default=53.0, help="the exchanges' download rate (k2's standalone measurement, 2026-11-14)")
    a = ap.parse_args()
    SCALE = a.cpu_scale
    train = load_rows(a.train)
    r, notes = fit(train, a.gpu_gbps, a.d2h_gbps)
    counts = {0: [0, 0], 1: [0, 0]}   # (the uses from its frame an upload buys, by path: measured on the training captures)
    for pat in a.train:
        for p in sorted(glob.glob(pat)):
            if '-pe1-' in p and 'pe1' not in pat:
                continue
            for k, (c, u) in pass_data.reuse_counts(p).items():
                counts[k][0] += c; counts[k][1] += u
    r.reuse = [counts[k][1] / counts[k][0] if counts[k][0] else 0.0 for k in (0, 1)]
    notes['reuse_counts'] = counts
    cal = json.loads(a.calibration.read_text()) if a.calibration else None
    checks = {'fitted passes (training)': check(train, r, cal, a.cpu_scale)}
    for spec in a.check:
        name, pats = spec.split('=', 1)
        checks[name] = check(load_rows(pats.split(',')), r, cal, a.cpu_scale)
    a.out.mkdir(parents=True, exist_ok=True)
    r.save(a.out / 'pass-model-fit.json')
    (a.out / 'pass-model-fit.txt').write_text(rates_text(r, ', '.join(a.train)))
    (a.out / 'pass-model-check.json').write_text(json.dumps(dict(notes=notes, checks=checks), indent=1) + '\n')
    t = text(checks, r, notes)
    (a.out / 'pass-model-check.txt').write_text(t + '\n')
    print(t)
    if a.header:
        a.header.write_text(header(r, golden(r), ', '.join(t.replace('/mnt/wsl/transfer/minimax-wsl/runs/', 'runs/') for t in a.train)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
