#!/usr/bin/env python3
"""Evaluation of the bounded follow-up. Rules are fixed here before the runs exist.

Comparisons use whole runs inside declared blocks as replicates, so slow periods stay
in the data. Per-pass results explain a total; they never replace it.

Effect rule, for a mean paired fraction with a 95% t interval [low, high] and band b:
    material   the interval lies wholly outside +-b
    bounded    the interval lies wholly inside +-b
    unresolved otherwise
Telemetry link rule, per generation forward, from clocks sampled during that forward:
    established  clock state explains at least half of the between-forward variance of the
                 helper's one-job GPU time, and slow forwards run at lower 3090 clocks than
                 the capture's other forwards in at least 80% of slow forwards
    contradicted slow forwards show no clock difference (under 20%) and clock state explains
                 under a tenth of the variance
    partial      otherwise
Submission query rule on complete generation:
    useful       mean improvement at least the band, with the 95% interval excluding zero,
                 and complete requests improve by at least the band pooled with no prompt losing
    negative     the interval lies wholly inside +-band, or shows a slowdown
    inconclusive otherwise; the switch stays off without evidence of a useful benefit
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import json
import math
import pathlib
import statistics
import sys

import native_light as light
import native_passes as passes
import native_telemetry as telemetry

BAND = 0.02
T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201,
       12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 18: 2.101, 20: 2.086, 24: 2.064, 30: 2.042, 40: 2.021, 60: 2.000}


def t95(df: int) -> float:
    if df < 1:
        return math.inf
    keys = sorted(T95)
    for k in keys:
        if df <= k:
            return T95[k]
    return 1.96


def interval(values: list[float]) -> dict:
    n = len(values)
    mean = statistics.fmean(values) if values else math.nan
    if n < 2:
        return dict(n=n, mean=mean, sd=None, half_width=None, low=None, high=None)
    sd = statistics.stdev(values)
    half = t95(n - 1) * sd / math.sqrt(n)
    return dict(n=n, mean=mean, sd=sd, half_width=half, low=mean - half, high=mean + half)


def classify_interval(result: dict, band: float = BAND) -> str:
    if result.get('low') is None:
        return 'unresolved'
    if result['low'] >= band or result['high'] <= -band:
        return 'material'
    if result['low'] > -band and result['high'] < band:
        return 'bounded'
    return 'unresolved'


def block_effect(rows: list[dict], treated: str, control: str, band: float = BAND) -> dict:
    """Rows hold one generation time per arm within a block; effect is treated over control minus one."""
    differences = [row[treated] / row[control] - 1 for row in rows if treated in row and control in row]
    result = interval(differences)
    result.update(differences=differences, classification=classify_interval(result, band))
    return result


def two_way_residual_sd(rows: list[dict], arms: list[str]) -> dict:
    """Residual standard deviation of log time after removing block and arm effects."""
    table = [[math.log(row[a]) for a in arms] for row in rows if all(a in row for a in arms)]
    b, a = len(table), len(arms)
    if b < 2 or a < 2:
        return dict(blocks=b, residual_sd=None)
    grand = statistics.fmean(x for row in table for x in row)
    block = [statistics.fmean(row) for row in table]
    arm = [statistics.fmean(row[j] for row in table) for j in range(a)]
    ss = sum((table[i][j] - block[i] - arm[j] + grand) ** 2 for i in range(b) for j in range(a))
    df = (b - 1) * (a - 1)
    return dict(blocks=b, arms=a, df=df, residual_sd=math.sqrt(ss / df),
                block_sd=statistics.stdev(block), arm_means={arms[j]: math.exp(arm[j]) for j in range(a)})


def pairs_needed(residual_sd: float | None, half_width: float) -> int | None:
    """Paired runs needed so a whole-run paired mean has a 95% half width of half_width."""
    if not residual_sd:
        return None
    n = 2
    while n < 100000 and t95(n - 1) * residual_sd * math.sqrt(2) / math.sqrt(n) > half_width:
        n += 1
    return n


def forward_states(capture, record: dict, gpu: int = 1) -> list[dict]:
    """For each generation forward, the samples taken while it ran. A short forward can hold none."""
    samples = sorted(record['per_gpu'][gpu], key=lambda s: s['t_ns'])
    rows = []
    at = 0
    for ordinal, f in enumerate(capture.generation_forwards()):
        begin, end = f['t_ns'], f['t_ns'] + f['end']
        while at < len(samples) and samples[at]['t_ns'] < begin:
            at += 1
        inside = []
        i = at
        while i < len(samples) and samples[i]['t_ns'] <= end:
            inside.append(samples[i])
            i += 1
        if not inside and at:
            inside = [samples[at - 1]]  # the latest sample before it started
        rows.append(dict(ordinal=ordinal, forward_ms=f['end'] * 1e-6, samples=len(inside),
                         graphics_mhz=statistics.median(s['graphics_mhz'] for s in inside) if inside else None,
                         memory_mhz=statistics.median(s['memory_mhz'] for s in inside) if inside else None,
                         pstate=statistics.median(s['pstate'] for s in inside) if inside else None,
                         power_mw=statistics.median(s['power_mw'] for s in inside) if inside else None,
                         util_gpu=statistics.median(s['util_gpu'] for s in inside) if inside else None))
    return rows


def state_label(row: dict) -> tuple:
    graphics = None if row['graphics_mhz'] is None else 100 * round(row['graphics_mhz'] / 100)
    return row['pstate'], row['memory_mhz'], graphics


def explained_variance(values: list[float], labels: list) -> float | None:
    if len(values) < 3:
        return None
    grand = statistics.fmean(values)
    total = sum((v - grand) ** 2 for v in values)
    if total == 0:
        return None
    groups: dict = defaultdict(list)
    for v, label in zip(values, labels):
        groups[label].append(v)
    within = sum((v - statistics.fmean(g)) ** 2 for g in groups.values() for v in g)
    return 1 - within / total


def clock_link(entries: list[tuple], slow_factor: float = 1.4) -> dict:
    """entries: (name, capture, telemetry record, rows). Pools forwards, within-run centred."""
    values, labels, slow_lower, slow_total, by_state = [], [], 0, 0, defaultdict(list)
    runs = {}
    for name, capture, record, rows in entries:
        states = forward_states(capture, record)
        one_job: dict[int, list[float]] = defaultdict(list)
        for r in rows:
            if r.helper == 1 and r.helper_gpu_ms is not None:
                one_job[r.ordinal].append(r.helper_gpu_ms)
        median_forward = statistics.median(s['forward_ms'] for s in states)
        slow = [s for s in states if s['forward_ms'] > slow_factor * median_forward and s['memory_mhz'] is not None]
        other = [s for s in states if s['forward_ms'] <= slow_factor * median_forward and s['memory_mhz'] is not None]
        if other:
            reference_memory = statistics.median(s['memory_mhz'] for s in other)
            reference_graphics = statistics.median(s['graphics_mhz'] for s in other)
        for s in slow:
            slow_total += 1
            if other and (s['memory_mhz'] < reference_memory or s['graphics_mhz'] < 0.9 * reference_graphics):
                slow_lower += 1
        kept = []
        for s in states:
            level = one_job.get(s['ordinal'])
            if level and s['memory_mhz'] is not None:
                level = statistics.median(level)
                values.append(math.log(level))
                labels.append(state_label(s))
                by_state[state_label(s)].append(level)
                kept.append(level)
        runs[name] = dict(forwards=len(states), slow_forwards=len(slow), helper_level_p10=passes._quantile(kept, .1) if kept else None,
                          helper_level_p90=passes._quantile(kept, .9) if kept else None,
                          states=dict(Counter(str(state_label(s)) for s in states).most_common(8)))
    explained = explained_variance(values, labels)
    slow_share = slow_lower / slow_total if slow_total else None
    if explained is not None and slow_share is not None and explained >= 0.5 and slow_share >= 0.8:
        verdict = 'established'
    elif explained is not None and slow_share is not None and explained < 0.1 and slow_share < 0.2:
        verdict = 'contradicted'
    else:
        verdict = 'partial'
    table = sorted(((str(k), len(v), statistics.median(v)) for k, v in by_state.items()), key=lambda x: -x[1])
    return dict(verdict=verdict, explained_variance=explained, slow_forwards=slow_total, slow_lower_share=slow_share,
                level_by_state=[dict(state=k, forwards=n, median_helper_gpu_ms=m) for k, n, m in table[:20]], runs=runs,
                state_key='(pstate, memory MHz, graphics rounded to 100 MHz)')


def standardized_total(rows: list, budgets: list[dict], reference: dict[int, float]) -> float:
    """Generation time with each pass's helper GPU part set to a shared level for its job count.

    This removes helper operating state by construction, so it can hide an intervention's own
    effect on the helper. It is reported beside the complete total and never replaces it.
    """
    change = 0.0
    for r in rows:
        if r.helper_gpu_ms is None or r.helper_published is None or r.helper not in reference:
            continue
        moved = r.helper_published - (r.helper_gpu_ms - reference[r.helper])
        cpu = r.cpu_end if r.cpu_end is not None else r.lane_begin
        change += r.composed_lane_end - max(cpu, moved, r.lane_begin)
    return sum(b['forward_ms'] for b in budgets) - change


def helper_reference(rows_by_run: list[list]) -> dict[int, float]:
    samples: dict[int, list[float]] = defaultdict(list)
    for rows in rows_by_run:
        for r in rows:
            if r.helper_gpu_ms is not None:
                samples[r.helper].append(r.helper_gpu_ms)
    return {k: statistics.median(v) for k, v in samples.items()}


def paired_passes(control: list, treated: list) -> dict:
    """Same (forward, layer) in two arms: what moved, by copies in the pass."""
    index = {(r.ordinal, r.layer): r for r in treated}
    groups: dict[int, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    same, total = 0, 0
    for a in control:
        b = index.get((a.ordinal, a.layer))
        if b is None or a.join_wait_next_ms is None or b.join_wait_next_ms is None:
            continue
        total += 1
        if a.key != b.key:
            continue
        same += 1
        g = groups[a.new]
        g['cycle'].append(b.cycle_ms - a.cycle_ms)
        g['wait'].append(b.join_wait_next_ms - a.join_wait_next_ms)
        g['host'].append(b.host_ms - a.host_ms)
        g['submit'].append(b.primary_submit - a.primary_submit)
        g['lane'].append((b.lane_end - b.lane_begin) - (a.lane_end - a.lane_begin))
    summary = {}
    for new, g in sorted(groups.items()):
        summary[str(new)] = {k: dict(mean=statistics.fmean(v), median=statistics.median(v), n=len(v)) for k, v in g.items()}
    return dict(same_work_windows=same, windows=total, by_new_copies=summary)


def query_decision(fixture: dict, requests: dict | None, band: float = BAND) -> dict:
    """Fixture effect is on minus off; negative is faster."""
    mean, low, high = fixture.get('mean'), fixture.get('low'), fixture.get('high')
    if low is None:
        return dict(decision='inconclusive', reason='no interval')
    if high < 0 and mean <= -band:
        if requests is None:
            return dict(decision='continue_to_requests', reason='fixture gain at or above the band with an interval excluding zero')
        pooled, per_prompt = requests['pooled'], requests['per_prompt']
        if pooled <= -band and all(v < 0 for v in per_prompt.values()):
            return dict(decision='useful', reason='complete generation gain confirmed')
        return dict(decision='inconclusive', reason='fixture gain was not confirmed on complete requests')
    if low > -band and high < band:
        return dict(decision='negative', reason='interval inside the band')
    if low >= 0:
        return dict(decision='negative', reason='slower with an interval excluding zero')
    return dict(decision='inconclusive', reason='interval straddles the band; no evidence of a useful benefit')


def load_runs(ledger: dict, steps: list[str], work_source: str | None) -> tuple[dict, dict]:
    entries = {s['step']: s for s in ledger['steps']}
    work = None
    if work_source and work_source in entries:
        work = passes.work_of(light.read(entries[work_source]['summary']['artifact']))
    tables, captures = {}, {}
    for name in steps:
        summary = entries[name]['summary']
        if summary.get('capture') in ('light', 'timestamps') and summary.get('artifact'):
            capture = light.read(summary['artifact'])
            captures[name] = capture
            tables[name] = passes.passes(capture, work=work if capture.meta.get('mode') == 'timestamps' else None)
    return tables, captures


def recorder_study(protocol: dict, ledger: dict) -> dict:
    study = protocol['recorder_study']
    entries = {s['step']: s for s in ledger['steps']}
    rows = []
    for block in study['blocks']:
        row = {arm: entries[step]['summary']['generation_ms'] for arm, step in block['arms'].items()
               if step in entries and not entries[step]['summary'].get('errors')}
        rows.append(row)
    arms = study['arm_order']
    result = dict(rows=rows, by_arm={a: passes.describe([r[a] for r in rows if a in r]) for a in arms},
                  telemetry_effect=block_effect(rows, 'off-telemetry', 'off-bare'),
                  timestamps_effect=block_effect(rows, 'timestamps', 'off-telemetry'),
                  full_effect=block_effect(rows, 'full', 'off-telemetry'),
                  full_over_timestamps=block_effect(rows, 'full', 'timestamps'),
                  variance=two_way_residual_sd(rows, arms))
    sd = result['variance'].get('residual_sd')
    result['pairs_for_half_width'] = {str(h): pairs_needed(sd, h) for h in (0.005, 0.01, 0.02)}
    first = study['blocks'][0]['arms']
    recording_steps = [step for block in study['blocks'] for arm, step in block['arms'].items() if arm in ('timestamps', 'full') and step in entries]
    full_steps = [step for block in study['blocks'] for arm, step in block['arms'].items() if arm == 'full' and step in entries]
    tables, captures = load_runs(ledger, recording_steps, full_steps[0] if full_steps else None)
    # Standardized totals across recording arms: what remains once helper state is held fixed.
    reference = helper_reference(list(tables.values()))
    standardized_rows = []
    budgets = {name: passes.forward_budgets(c) for name, c in captures.items()}
    for block in study['blocks']:
        row = {}
        for arm in ('timestamps', 'full'):
            step = block['arms'].get(arm)
            if step in tables:
                row[arm] = standardized_total(tables[step], budgets[step], reference)
        standardized_rows.append(row)
    result['standardized'] = dict(reference_helper_gpu_ms=reference, full_over_timestamps=block_effect(standardized_rows, 'full', 'timestamps'),
                                  variance=two_way_residual_sd(standardized_rows, ['timestamps', 'full']),
                                  scope='Helper GPU part set to a shared level per job count; hides any change in helper state.')
    raw_rows = [{arm: r[arm] for arm in ('timestamps', 'full') if arm in r} for r in rows]
    result['raw_recording_variance'] = two_way_residual_sd(raw_rows, ['timestamps', 'full'])
    entries_for_link = []
    for name in recording_steps:
        path = entries[name]['summary'].get('telemetry')
        if path and name in tables:
            entries_for_link.append((name, captures[name], telemetry.read(path), tables[name]))
    result['clock_link'] = clock_link(entries_for_link) if entries_for_link else dict(verdict='unavailable')
    return result


def query_study(protocol: dict, ledger: dict) -> dict:
    study = protocol['query_study']
    entries = {s['step']: s for s in ledger['steps']}
    fixture_rows = []
    per_pass = {}
    for block in study['fixture_blocks']:
        row = {arm: entries[step]['summary']['generation_ms'] for arm, step in block.items()
               if step in entries and not entries[step]['summary'].get('errors')}
        fixture_rows.append(row)
    effect = block_effect(fixture_rows, 'on', 'off')
    tables, captures = {}, {}
    steps = [s for block in study['fixture_blocks'] for s in block.values() if s in entries]
    tables, captures = load_runs(ledger, steps, None)
    for i, block in enumerate(study['fixture_blocks']):
        if block['off'] in tables and block['on'] in tables:
            per_pass[f"block{i}"] = paired_passes(tables[block['off']], tables[block['on']])
    work = {name: {k: entries[name]['summary']['report'].get(k) for k in ('demand_uploads', 'primary_issued', 'primary_published', 'helper_started',
            'primary_submit_queries', 'primary_submit_not_ready', 'windows')} for name in steps}
    requests = None
    request_rows: dict[str, list[dict]] = defaultdict(list)
    for prompt, blocks in study.get('request_blocks', {}).items():
        for block in blocks:
            if all(step in entries and not entries[step]['summary'].get('errors') for step in block.values()):
                request_rows[prompt].append({arm: entries[step]['summary']['generation_ms'] for arm, step in block.items()})
    if request_rows:
        per_prompt = {prompt: statistics.fmean(r['on'] / r['off'] - 1 for r in rows) for prompt, rows in request_rows.items()}
        requests = dict(per_prompt=per_prompt, pooled=statistics.fmean(per_prompt.values()), rows=dict(request_rows))
    return dict(fixture_rows=fixture_rows, fixture_effect=effect, per_pass=per_pass, work=work, requests=requests,
                decision=query_decision(effect, requests))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--protocol', required=True, type=pathlib.Path)
    ap.add_argument('--ledger', required=True, type=pathlib.Path)
    ap.add_argument('--study', choices=('recorder', 'query'), required=True)
    ap.add_argument('--output', type=pathlib.Path)
    a = ap.parse_args()
    protocol, ledger = json.loads(a.protocol.read_text()), json.loads(a.ledger.read_text())
    result = recorder_study(protocol, ledger) if a.study == 'recorder' else query_study(protocol, ledger)
    text = json.dumps(result, indent=2, default=str)
    if a.output:
        a.output.write_text(text + '\n')
    else:
        print(text)
    return 0


if __name__ == '__main__':
    sys.exit(main())
