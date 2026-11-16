#!/usr/bin/env python3
"""What the window's staging would decide with pass_model 1 against today's planner, on recorded passes. MODELED.

    python3 staging_whatif.py --fit pass-model-fit.json --calibration CAL.json [--cpu-scale 1.5] GLOB [GLOB ...]

For each recorded pass at token 128 and later with an expert that could be staged (the CPU lane's and the ones uploaded),
both planners start from the pass with all of those on the CPU lane and choose which to upload, at most the pass's
recorded limit (stage_max):
  today   plan_batch_staging (mm3_batch_staging.hpp): the calibration's additive lanes, the CPU lane times the CPU price,
          the 4080 left out, the greedy that lowers the largest lane
  model   pass_model::plan_staging (mm3_pass_model.hpp): the pass model's predicted finish, the 4080 included, less the
          later uses each upload buys (reuse[path] of them, each worth the per-use value: the average marginal the
          engine keeps, here the mean over the recorded windows under the model's own choices)
Checks first that today's replica picks as many uploads as the engine did (it should, on nearly every pass), then
reports how often the two choices differ, the pass model's predicted finish of each choice, the later uses the model's
fewer uploads give up, and the net over the requests' time from token 128. The model judges both, so the difference is
the model's own view of its choice, not a measurement; the timed test measures it."""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import pathlib
import statistics
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass_data  # noqa: E402
import pass_model as pm  # noqa: E402


def today_plan(row, cand, cal, scale, max_new):
    """plan_batch_staging's greedy on the recorded pass: cand = [(uses), ...] on the CPU lane; returns the count chosen."""
    n = lambda c: min(16, max(1, c))
    cpu_fixed = [u for u in row['cpu_uses']]   # (the CPU experts that are not candidates: none here, all are)
    gpu = [u for u in row['gpu_uses']]
    helper = [u for u in row['helper_uses']]

    def evaluate(sel):
        c_list = [u for i, u in enumerate(cand) if not sel[i]]
        c = sum(cal['cpu_extra_ms'][n(u)] for u in c_list) + (cal['cpu_base_ms'] if c_list else 0.0)
        c *= scale
        g = cal['gpu_base_ms'][min(16, row['rows'])] + sum(cal['gpu_extra_ms'][n(u)] for u in gpu) + \
            sum(cal['gpu_extra_ms'][n(u)] + cal['h2d_ms'] for i, u in enumerate(cand) if sel[i])
        h = (sum(cal['helper_extra_ms'][n(u)] for u in helper) + cal['helper_base_ms']) if helper else 0.0
        return max(c, g, h)
    sel = [False] * len(cand)
    cur = evaluate(sel)
    for _ in range(max_new):
        best, best_v = -1, cur
        for i in range(len(cand)):
            if sel[i]:
                continue
            sel[i] = True
            v = evaluate(sel)
            sel[i] = False
            if v < best_v - 1e-12 * max(1.0, abs(cur), abs(best_v), abs(v)):
                best, best_v = i, v
        if best < 0:
            break
        sel[best] = True
        cur = best_v
    return sel


def staged_load(base: pm.Load, cand, sel) -> pm.Load:
    t = pm.Load(**{**base.__dict__})
    for i, u in enumerate(cand):
        if sel[i]:
            t.cpu_experts -= 1; t.cpu_columns -= u; t.copies += 1; t.gpu_experts += 1; t.gpu_picks += u
            t.late_experts += 1; t.late_picks += u
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fit', required=True, type=pathlib.Path)
    ap.add_argument('--calibration', required=True, type=pathlib.Path)
    ap.add_argument('--cpu-scale', type=float, default=1.5)
    ap.add_argument('--out', type=pathlib.Path)
    ap.add_argument('--json', type=pathlib.Path)
    ap.add_argument('globs', nargs='+')
    a = ap.parse_args()
    r = pm.Rates.load(a.fit)
    cal = json.loads(a.calibration.read_text())
    passes = collections.defaultdict(list)   # mode -> [(base, candidates, max_new, today's selection, engine's uploads)]
    counts = collections.defaultdict(collections.Counter)
    span = collections.defaultdict(float)
    for pat in a.globs:
        for path in sorted(glob.glob(pat)):
            if '-pe1-' in path and 'pe1' not in pat:
                continue
            observed = pass_data.as_dicts(pass_data.rows_of(path))
            if not observed: continue
            mode = observed[0]['mode']
            span[mode] += pass_data.forward_span(path)
            for x in observed:
                counts[mode]['passes'] += 1
                uses = x['cpu_uses'] + x['new_uses']
                if not uses:
                    continue
                ld = pm.load_of(x)
                base = pm.Load(**{**ld.__dict__})
                base.cpu_experts = len(uses); base.cpu_columns = sum(uses); base.copies = 0
                base.gpu_experts = ld.gpu_experts - len(x['new_uses']); base.gpu_picks = ld.gpu_picks - sum(x['new_uses'])
                base.late_experts = max(0,ld.late_experts-len(x['new_uses'])); base.late_picks = max(0,ld.late_picks-sum(x['new_uses']))
                max_new = x['stage_max'] if x.get('stage_max') is not None else 4
                old = today_plan(x, uses, cal, a.cpu_scale, max_new)
                passes[mode].append((base, list(enumerate(uses)), max_new, old, x['uploads']))
    # the per-use value: the engine's running average of use_samples over the windows the model planned; offline, the
    # mean over the recorded windows under the model's own choices, iterated from the rates' value
    mb = r.record_bytes * 1e-6
    value = max(0.0, mb / (r.cpu_gbps * min(1.0, r.host_gbps / (r.cpu_gbps + r.copy_gbps[0]))) - (mb / r.gpu_gbps + r.gpu_pick))
    values = [value]
    for _ in range(3):
        samples = []
        for mode in passes:
            for base, cand, max_new, _, _ in passes[mode]:
                sel, _, _, _ = pm.plan_staging(base, cand, max_new, r, value)
                samples += pm.use_samples(staged_load(base, [u for _, u in cand], [i in sel for i, _ in enumerate(cand)]), r,
                                          [u for i,u in cand if i not in sel], [dict(cand)[i] for i in sel])
        value = statistics.fmean(samples) if samples else 0.0
        values.append(value)
    out = dict(use_value_ms=value, use_value_iterations=values, reuse=r.reuse, modes={})
    lines = ['Staging by the pass model against today\'s planner on recorded passes (MODELED: the pass model judges both choices)',
             f'The value of one later use by an expert on the 5090 (the planner\'s average marginal): {1000 * value:.1f} us '
             f'(iterations {", ".join(f"{1000 * v:.1f}" for v in values)}); later uses an upload buys: {r.reuse[0]:.2f} in windows, '
             f'{r.reuse[1]:.2f} at one row', '']
    for mode in passes:
        c = counts[mode]
        n = len(passes[mode])
        gain = credit_lost = 0.0
        for base, cand, max_new, old, engine in passes[mode]:
            c['replica matches'] += sum(old) == engine
            sel, _, f_new, _ = pm.plan_staging(base, cand, max_new, r, value)
            f_old = pm.predict(staged_load(base, [u for _, u in cand], old), r)
            c['same count'] += sum(old) == len(sel)
            c['more'] += len(sel) > sum(old)
            c['fewer'] += len(sel) < sum(old)
            c['uploads today'] += sum(old)
            c['uploads model'] += len(sel)
            gain += f_old.passed - f_new.passed
            credit_lost += (sum(old) - len(sel)) * r.reuse[1 if base.rows == 1 else 0] * value
        net = gain - credit_lost
        total = span[mode]
        out['modes'][mode] = dict(passes=c['passes'], candidate_passes=n, replica_matches=c['replica matches'] / max(1, n),
                                  same=c['same count'] / max(1, n), more=c['more'] / max(1, n), fewer=c['fewer'] / max(1, n),
                                  uploads_today=c['uploads today'] / max(1, n), uploads_model=c['uploads model'] / max(1, n),
                                  finish_gain_ms=gain, reuse_lost_ms=credit_lost, net_ms=net, span_ms=total,
                                  net_us_per_pass=1000 * net / max(1, c['passes']),
                                  tok_s=total / (total - net) - 1 if total > net else None)
        lines.append(f"== {mode}: {c['passes']} passes, {n} with an expert that could be staged")
        lines.append(f"  today's replica picks the engine's upload count on {100 * c['replica matches'] / max(1, n):.1f}% of them")
        lines.append(f"  same count {100 * c['same count'] / max(1, n):.1f}%, the model stages more on {100 * c['more'] / max(1, n):.1f}%, "
                     f"fewer on {100 * c['fewer'] / max(1, n):.1f}%; uploads a candidate pass: today {c['uploads today'] / max(1, n):.3f}, "
                     f"the model {c['uploads model'] / max(1, n):.3f}")
        lines.append(f"  the model's predicted pass finish, today's choice minus its own: {1000 * gain / max(1, n):.1f} us a candidate pass; "
                     f"less the later uses the fewer uploads give up ({1000 * credit_lost / max(1, n):.1f} us): net "
                     f"{1000 * net / max(1, n):.1f} us a candidate pass, {1000 * net / max(1, c['passes']):.2f} us a pass")
        if total > net:
            lines.append(f"  over the requests' time from token 128 ({total / 1000:.1f} s): tokens per second {100 * (total / (total - net) - 1):+.2f}%")
        lines.append('')
    text = '\n'.join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + '\n')
    if a.json:
        a.json.write_text(json.dumps(out, indent=1) + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
