#!/usr/bin/env python3
"""What the window's staging would decide with pass_model 1 against today's planner, on recorded passes. MODELED.

    python3 staging_whatif.py --fit pass-model-fit.json --calibration CAL.json [--cpu-scale 1.5] GLOB [GLOB ...]

For each recorded pass at token 128 and later with an expert that could be staged (the CPU lane's and the ones uploaded),
both planners start from the pass with all of those on the CPU lane and choose which to upload, at most the pass's
recorded limit (stage_max):
  today   plan_batch_staging (mm3_batch_staging.hpp): the calibration's additive lanes, the CPU lane times the CPU price,
          the 4080 left out, the greedy that lowers the largest lane
  model   pass_model::plan_staging (mm3_pass_model.hpp): the pass model's predicted finish, the 4080 included
Checks first that today's replica picks as many uploads as the engine did (it should, on nearly every pass), then
reports how often the two choices differ and the pass model's predicted finish of each choice. The model judges both, so
the difference is the model's own view of its choice, not a measurement; the timed test measures it."""
from __future__ import annotations

import argparse
import collections
import glob
import json
import math
import pathlib
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


def model_plan(base: pm.Load, cand, r, max_new):
    cur = base
    f = pm.predict(cur, r)
    sel = [False] * len(cand)
    for _ in range(max_new):
        best, best_f, best_l = -1, f, cur
        for i, u in enumerate(cand):
            if sel[i]:
                continue
            t = pm.Load(**{**cur.__dict__})
            t.cpu_experts -= 1; t.cpu_columns -= u; t.copies += 1; t.gpu_experts += 1; t.gpu_picks += u
            tf = pm.predict(t, r)
            if tf.passed < best_f.passed - 1e-12 * max(1.0, abs(f.passed), abs(best_f.passed), abs(tf.passed)):
                best, best_f, best_l = i, tf, t
        if best < 0:
            break
        sel[best] = True
        cur, f = best_l, best_f
    return sel, f


def staged_load(base: pm.Load, cand, sel) -> pm.Load:
    t = pm.Load(**{**base.__dict__})
    for i, u in enumerate(cand):
        if sel[i]:
            t.cpu_experts -= 1; t.cpu_columns -= u; t.copies += 1; t.gpu_experts += 1; t.gpu_picks += u
    return t


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--fit', required=True, type=pathlib.Path)
    ap.add_argument('--calibration', required=True, type=pathlib.Path)
    ap.add_argument('--cpu-scale', type=float, default=1.5)
    ap.add_argument('--out', type=pathlib.Path)
    ap.add_argument('globs', nargs='+')
    a = ap.parse_args()
    r = pm.Rates.load(a.fit)
    cal = json.loads(a.calibration.read_text())
    by_mode = collections.defaultdict(lambda: collections.Counter())
    gain = collections.defaultdict(float)
    span = collections.defaultdict(float)
    for pat in a.globs:
        for path in sorted(glob.glob(pat)):
            if '-pe1-' in path and 'pe1' not in pat:
                continue
            mode = 'one row' if 'one-row' in path else 'drafting'
            rows = pass_data.as_dicts(pass_data.rows_of(path))
            for x in rows:
                cand_ids = x['cpu_uses'] + x['new_uses']   # (columns; order by expert id is lost across the two lists: fine for counts)
                c = by_mode[mode]
                c['passes'] += 1
                if not cand_ids:
                    continue
                c['with candidates'] += 1
                ld = pm.load_of(x)
                base = pm.Load(**{**ld.__dict__})
                base.cpu_experts = len(cand_ids); base.cpu_columns = sum(cand_ids); base.copies = 0
                base.gpu_experts = ld.gpu_experts - len(x['new_uses']); base.gpu_picks = ld.gpu_picks - sum(x['new_uses'])
                max_new = x['stage_max'] or 4
                old = today_plan(x, cand_ids, cal, a.cpu_scale, max_new)
                c['replica matches'] += sum(old) == x['uploads']
                new, f_new = model_plan(base, cand_ids, r, max_new)
                f_old = pm.predict(staged_load(base, cand_ids, old), r)
                c['same count'] += sum(old) == sum(new)
                c['model stages more'] += sum(new) > sum(old)
                c['model stages fewer'] += sum(new) < sum(old)
                c['uploads today'] += sum(old)
                c['uploads model'] += sum(new)
                gain[mode] += f_old.passed - f_new.passed
    lines = ['Staging by the pass model against today\'s planner on recorded passes (MODELED: the pass model judges both choices)', '']
    for mode, c in by_mode.items():
        n = max(1, c['with candidates'])
        lines.append(f"== {mode}: {c['passes']} passes, {c['with candidates']} with an expert that could be staged")
        lines.append(f"  today's replica picks the engine's upload count on {100 * c['replica matches'] / n:.1f}% of them")
        lines.append(f"  same count {100 * c['same count'] / n:.1f}%, the model stages more on {100 * c['model stages more'] / n:.1f}%, fewer on {100 * c['model stages fewer'] / n:.1f}%")
        lines.append(f"  uploads a candidate pass: today {c['uploads today'] / n:.3f}, the model {c['uploads model'] / n:.3f}")
        lines.append(f"  the model's predicted pass finish, today's choice minus its own: {1000 * gain[mode] / n:.1f} us a candidate pass, "
                     f"{1000 * gain[mode] / max(1, c['passes']):.2f} us a pass")
        lines.append('')
    text = '\n'.join(lines)
    print(text)
    if a.out:
        a.out.write_text(text + '\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
