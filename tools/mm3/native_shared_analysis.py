#!/usr/bin/env python3
"""Effect screen for the shared first-stage model: reads a campaign ledger, classifies effects.

The accounting unit is the interval between consecutive layer joins. Each join
drains the primary stream, so for one forward
    forward time = prefix + sum(cycles) + policy tail
exactly, with no overlapping spans to double-count. A cycle is keyed by the
expert work of the layer it begins with.

Classification rules are fixed here, before any campaign data exists:
  paired comparisons   material: every pair has the same sign and |d| >= band
                       bounded:  every |d| < band and the control's own drift < band
                       otherwise unresolved
  observational states material: every replicate same sign with |v| - 2se >= band
                       bounded:  every |v| + 2se < band, at least two replicates, enough coverage
                       otherwise unresolved
Observational results are associations within matched work, not causal prices.
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

BAND = 0.02
COLD_FORWARDS = 8      # a pick is cold when its record did not run on that device this recently
MIN_REFERENCE = 5      # matched-work reference windows required for a stratum
MIN_EXPOSED = 20       # exposed windows required before a replicate counts
MIN_COVERAGE = 0.5     # share of exposed windows that must have a matched reference

SERVICES = ('cpu', 'helper', 'primary_resident', 'staged_ready', 'staged_new')


def classify_paired(differences: list[float], control_drift: float | None = None, band: float = BAND) -> str:
    if not differences:
        return 'unresolved'
    same_sign = all(d > 0 for d in differences) or all(d < 0 for d in differences)
    if same_sign and min(abs(d) for d in differences) >= band:
        return 'material'
    if max(abs(d) for d in differences) < band and (control_drift is None or abs(control_drift) < band):
        return 'bounded'
    return 'unresolved'


def classify_observational(replicates: list[dict], band: float = BAND) -> str:
    """Each replicate has value (fraction of generation), se (same unit), exposed and coverage."""
    counted = [r for r in replicates if r['exposed'] >= MIN_EXPOSED]
    if not counted:
        return 'unresolved'
    sign = {math.copysign(1, r['value']) for r in counted}
    if len(sign) == 1 and all(abs(r['value']) - 2 * r['se'] >= band for r in counted):
        return 'material'
    if (len(counted) >= 2 and all(abs(r['value']) + 2 * r['se'] < band for r in counted) and
            all(r['coverage'] >= MIN_COVERAGE for r in counted)):
        return 'bounded'
    return 'unresolved'


def windows(capture: light.LightCapture) -> tuple[list[dict], list[dict]]:
    """Join-to-join cycles and per-forward budgets for one-row generation forwards."""
    by_forward: dict[int, list[dict]] = defaultdict(list)
    for r in capture.of('layer'):
        by_forward[r['forward']].append(r)
    cold_clock: dict[tuple[str, int, int], int] = {}
    mutated: dict[int, int] = {}
    mutations_by_forward: dict[int, set[int]] = defaultdict(set)
    for r in capture.records:
        if r['type'] == 'primary_publish' or (r['type'] == 'helper_event' and r['phase'] == 'published'):
            mutations_by_forward[r['forward']].update({r['incoming'] // 1024, r['victim'] // 1024})
    cycles, budgets = [], []
    forwards = capture.generation_forwards()
    for ordinal, f in enumerate(forwards):
        rows = sorted(by_forward.get(f['forward'], []), key=lambda r: r['layer'])
        if not rows:
            raise ValueError(f"forward {f['forward']} has no layer records")
        begin = f['t_ns']
        first_join = rows[0]['t_ns'] + rows[0]['join_end']
        budget = dict(forward=f['forward'], position=f['position'], forward_ms=f['end'] * 1e-6,
                      prefix_ms=(first_join - begin) * 1e-6, policy_tail_ms=(f['end'] - f['sync_end']) * 1e-6,
                      cycles_ms=0.0)
        for layer in mutations_by_forward.get(f['forward'] - 1, ()):
            mutated[layer] = ordinal - 1
        for i, r in enumerate(rows):
            join = r['t_ns'] + r['join_end']
            if i + 1 < len(rows):
                following = rows[i + 1]
                end = following['t_ns'] + following['join_end']
                wait = (following['join_end'] - following['pre_end']) * 1e-6
                pre = following['pre_end'] * 1e-6
            else:
                end = begin + f['sync_end']
                wait, pre = None, None
            counts = Counter(r['service'])
            key = tuple(counts.get(s, 0) for s in SERVICES)
            cold = dict(cpu=0, helper=0, primary=0)
            for service, expert in zip(r['service'], r['executed']):
                lane = {'cpu': 'cpu', 'helper': 'helper', 'primary_resident': 'primary'}.get(service)
                if lane is None:
                    continue
                identity = (lane, r['layer'], expert)
                last = cold_clock.get(identity)
                if last is None or ordinal - last > COLD_FORWARDS:
                    cold[lane] += 1
                cold_clock[identity] = ordinal
            host_end = r['end']
            plan = r['plan_end'] - r['boundary_end']
            after_plan = max(r['plan_end'], r['helper_submit_end'])
            lane_begin = r['primary_submit_end']
            cpu = r['cpu_end'] - lane_begin if r['cpu_end'] else 0
            lane_after_cpu = max(lane_begin, r['cpu_end'])
            collect = r['collect_end'] - lane_after_cpu if r['collect_end'] else 0
            last_mutation = mutated.get(r['layer'])
            cycle_ms = (end - join) * 1e-6
            budget['cycles_ms'] += cycle_ms
            cycles.append(dict(
                forward=f['forward'], ordinal=ordinal, layer=r['layer'], position=f['position'], key=key,
                final_layer=i + 1 == len(rows), cycle_ms=cycle_ms, host_ms=(host_end - r['join_end']) * 1e-6,
                join_wait_next_ms=wait, pre_next_ms=pre, boundary_ms=(r['boundary_end'] - r['join_end']) * 1e-6,
                plan_ms=plan * 1e-6, helper_submit_ms=max(0, r['helper_submit_end'] - r['plan_end']) * 1e-6,
                primary_submit_ms=max(0, r['primary_submit_end'] - after_plan) * 1e-6, cpu_ms=cpu * 1e-6, collect_ms=collect * 1e-6,
                free_slots=r['free_slots'], reserved_slots=r['reserved_slots'], foreground_slots=r['foreground_slots'],
                exchanges_pending=r['exchanges_pending'], helper_lease=r['helper_lease'], stage_max=r['stage_max'],
                stage_selected=r['stage_selected'], total_slots=r['total_slots'], cold_cpu=cold['cpu'], cold_helper=cold['helper'],
                cold_primary=cold['primary'], recent_mutation=last_mutation is not None and ordinal - last_mutation <= COLD_FORWARDS))
        budgets.append(budget)
    return cycles, budgets


# Candidate states evaluated against matched current work. level() returns None
# for the reference level, otherwise an exposed level.
FEATURES = {
    'cache_history_cpu_cold': lambda w: 'cold' if w['cold_cpu'] else None,
    'cache_history_primary_cold': lambda w: 'cold' if w['cold_primary'] else None,
    'cache_history_helper_cold': lambda w: 'cold' if w['cold_helper'] else None,
    'cache_history_recent_mutation': lambda w: 'recent' if w['recent_mutation'] else None,
    'buffers_free_count_low': lambda w: 'low' if w['free_slots'] < 3 else None,
    'buffers_any_reserved': lambda w: 'reserved' if w['reserved_slots'] else None,
    'buffers_foreground_unretired': lambda w: 'foreground' if w['foreground_slots'] else None,
    'pending_primary_exchange': lambda w: 'pending' if w['exchanges_pending'] else None,
    'pending_helper_lease': lambda w: 'lease' if w['helper_lease'] else None,
    'pending_busy_flag': lambda w: 'busy' if (w['exchanges_pending'] or w['helper_lease']) else None,
}


def stratified_excess(cycles: list[dict], level, generation_ms: float) -> dict:
    """Total excess cycle time of exposed windows over matched-work reference windows.

    Standard error clusters by forward and adds the reference means' uncertainty.
    """
    reference: dict[tuple, list[float]] = defaultdict(list)
    exposed = [w for w in cycles if level(w) is not None]
    for w in cycles:
        if level(w) is None:
            reference[w['key']].append(w['cycle_ms'])
    matched = [w for w in exposed if len(reference.get(w['key'], ())) >= MIN_REFERENCE]
    excess_by_forward: dict[int, float] = defaultdict(float)
    reference_variance = 0.0
    counts = Counter(w['key'] for w in matched)
    for key, n in counts.items():
        values = reference[key]
        reference_variance += (n ** 2) * statistics.variance(values) / len(values)
    means = {k: statistics.fmean(v) for k, v in reference.items() if len(v) >= MIN_REFERENCE}
    for w in matched:
        excess_by_forward[w['forward']] += w['cycle_ms'] - means[w['key']]
    total = sum(excess_by_forward.values())
    forwards = sorted({w['forward'] for w in cycles})
    per_forward = [excess_by_forward.get(f, 0.0) for f in forwards]
    cluster_variance = len(per_forward) * statistics.variance(per_forward) if len(per_forward) > 1 else 0.0
    se_ms = math.sqrt(cluster_variance + reference_variance)
    return dict(exposed=len(exposed), matched=len(matched), coverage=len(matched) / len(exposed) if exposed else 0.0,
                windows=len(cycles), exposure_share=len(exposed) / len(cycles) if cycles else 0.0,
                excess_ms=total, se_ms=se_ms, value=total / generation_ms if generation_ms else 0.0,
                se=se_ms / generation_ms if generation_ms else 0.0)


def lag_one_correlation(cycles: list[dict]) -> dict:
    """Residual correlation between consecutive cycles of a forward; tests additive composition."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for w in cycles:
        groups[w['key']].append(w['cycle_ms'])
    means = {k: statistics.fmean(v) for k, v in groups.items()}
    previous = None
    pairs = []
    for w in sorted(cycles, key=lambda w: (w['ordinal'], w['layer'])):
        residual = w['cycle_ms'] - means[w['key']]
        if previous and previous[0] == w['ordinal']:
            pairs.append((previous[1], residual))
        previous = (w['ordinal'], residual)
    if len(pairs) < 10 or statistics.pstdev(a for a, _ in pairs) == 0 or statistics.pstdev(b for _, b in pairs) == 0:
        return dict(pairs=len(pairs), correlation=None)
    return dict(pairs=len(pairs), correlation=statistics.correlation([a for a, _ in pairs], [b for _, b in pairs]))


def budget_summary(cycles: list[dict], budgets: list[dict], generation_ms: float) -> dict:
    total = lambda k: sum(w[k] or 0 for w in cycles)
    forward = sum(b['forward_ms'] for b in budgets)
    return dict(generation_ms=generation_ms, forwards=len(budgets), forward_ms=forward,
                outside_forwards_ms=generation_ms - forward, prefix_ms=sum(b['prefix_ms'] for b in budgets),
                policy_tail_ms=sum(b['policy_tail_ms'] for b in budgets), cycles_ms=sum(b['cycles_ms'] for b in budgets),
                host_ms=total('host_ms'), join_wait_ms=total('join_wait_next_ms'), pre_submit_ms=total('pre_next_ms'),
                boundary_ms=total('boundary_ms'), plan_ms=total('plan_ms'), helper_submit_ms=total('helper_submit_ms'),
                primary_submit_ms=total('primary_submit_ms'), cpu_ms=total('cpu_ms'), collect_ms=total('collect_ms'),
                accounting_residual_ms=forward - sum(b['prefix_ms'] + b['policy_tail_ms'] + b['cycles_ms'] for b in budgets),
                note='CPU, helper and primary spans overlap GPU work; only the top-level cycle partition is additive.')


def work_table(cycles: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = defaultdict(list)
    for w in cycles:
        groups[w['key']].append(w)
    rows = []
    for key, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        values = sorted(w['cycle_ms'] for w in items)
        waits = [w['join_wait_next_ms'] for w in items if w['join_wait_next_ms'] is not None]
        rows.append(dict(zip(SERVICES, key)) | dict(windows=len(items), mean_ms=statistics.fmean(values),
            median_ms=statistics.median(values), p90_ms=values[int(0.9 * (len(values) - 1))],
            host_mean_ms=statistics.fmean(w['host_ms'] for w in items),
            join_wait_mean_ms=statistics.fmean(waits) if waits else None))
    return rows


def predict_total(training: list[dict], training_budgets: list[dict], target: list[dict], target_budgets: list[dict]) -> dict:
    """Held-out generation total from a shared table of current work; no layer parameters."""
    groups: dict[tuple, list[float]] = defaultdict(list)
    for w in training:
        groups[w['key']].append(w['cycle_ms'])
    means = {k: statistics.fmean(v) for k, v in groups.items()}
    overall = statistics.fmean(w['cycle_ms'] for w in training)
    unseen = sum(w['key'] not in means for w in target)
    predicted = sum(means.get(w['key'], overall) for w in target)
    absolute = statistics.fmean(abs(means.get(w['key'], overall) - w['cycle_ms']) for w in target)
    overhead = statistics.fmean(b['prefix_ms'] + b['policy_tail_ms'] for b in training_budgets) * len(target_budgets)
    observed = sum(b['forward_ms'] for b in target_budgets)
    total = predicted + overhead
    return dict(predicted_ms=total, observed_ms=observed, error_fraction=(total - observed) / observed,
                cycle_mean_absolute_error_ms=absolute, unseen_work_windows=unseen, target_windows=len(target))


def exchange_exposure(capture: light.LightCapture) -> dict:
    issues = {(r['incoming'], r['victim']): r for r in capture.of('primary_issue')}
    durations = []
    for r in capture.of('primary_publish'):
        issue = issues.get((r['incoming'], r['victim']))
        if issue:
            durations.append((r['t_ns'] - issue['t_ns']) * 1e-6)
    helper: dict[tuple, dict] = defaultdict(dict)
    for r in capture.of('helper_event'):
        helper[(r['incoming'], r['victim'])][r['phase']] = r['t_ns']
    leases = [(v['retired'] - v['lease']) * 1e-6 for v in helper.values() if 'lease' in v and 'retired' in v]
    copies = capture.of('demand_copy')
    return dict(primary_issued=len(capture.of('primary_issue')), primary_published=len(capture.of('primary_publish')),
                primary_reused_landings=sum(r['reuse'] for r in capture.of('primary_issue')),
                h2d_bytes=sum(r['h2d_bytes'] for r in capture.of('primary_issue')),
                d2h_bytes=sum(r['d2h_bytes'] for r in capture.of('primary_issue')),
                issue_to_publish_ms=summary_of(durations), issue_submit_ms=summary_of([r['submit_end'] * 1e-6 for r in capture.of('primary_issue')]),
                helper_leases=len(helper), helper_lease_ms=summary_of(leases), demand_copies=len(copies),
                demand_submit_ms=summary_of([r['submit_end'] * 1e-6 for r in copies]))


def summary_of(values: list[float]) -> dict:
    if not values:
        return dict(count=0)
    ordered = sorted(values)
    return dict(count=len(ordered), total=sum(ordered), mean=statistics.fmean(ordered), median=statistics.median(ordered),
                p90=ordered[int(0.9 * (len(ordered) - 1))], maximum=ordered[-1])


def state_changes(cycles: list[dict]) -> dict:
    """Assignments and exposure an admissions comparison must show alongside its timing."""
    counts = Counter()
    for w in cycles:
        for service, n in zip(SERVICES, w['key']):
            counts[service] += n
    pending = sum(1 for w in cycles if w['exchanges_pending'])
    lease = sum(1 for w in cycles if w['helper_lease'])
    busy = sum(1 for w in cycles if w['exchanges_pending'] or w['helper_lease'])
    return dict(picks=dict(counts), windows=len(cycles), exchange_pending_windows=pending, helper_lease_windows=lease, busy_windows=busy,
                stage_binding_windows=sum(1 for w in cycles if w['stage_selected'] and w['stage_selected'] == w['stage_max'] and w['key'][0]),
                free_slots=dict(sorted(Counter(w['free_slots'] for w in cycles).items())),
                host_ms=sum(w['host_ms'] for w in cycles), cpu_ms=sum(w['cpu_ms'] for w in cycles), cycle_ms=sum(w['cycle_ms'] for w in cycles))


def fixture_effect(ledger: dict, first: str, second: str, third: str, fourth: str) -> dict:
    """An ABBA block: pairs (first, second) and (fourth, third). Effect is B minus A."""
    steps = {s['step']: s['summary'] for s in ledger['steps']}
    if any(n not in steps or steps[n].get('errors') for n in (first, second, third, fourth)):
        return dict(status='unavailable', classification='unresolved')
    time = {n: steps[n]['generation_ms'] for n in (first, second, third, fourth)}
    differences = [(time[second] - time[first]) / time[first], (time[third] - time[fourth]) / time[fourth]]
    drift = (time[fourth] - time[first]) / time[first]
    return dict(status='measured', generation_ms=time, differences=differences, mean_difference=statistics.fmean(differences),
                half_range=(max(differences) - min(differences)) / 2, control_drift=drift,
                milliseconds=[time[second] - time[first], time[third] - time[fourth]],
                classification=classify_paired(differences, drift),
                note='Two-pair half-ranges are descriptive spread, not confidence intervals.')


def analyze(ledger_path: pathlib.Path) -> dict:
    ledger = json.loads(ledger_path.read_text())
    steps = {s['step']: s for s in ledger['steps']}
    captures, tables = {}, {}
    for name, entry in steps.items():
        summary = entry['summary']
        if summary.get('capture') != 'light' or summary.get('errors'):
            continue
        capture = light.read(summary['artifact'])
        cycles, budgets = windows(capture)
        generation_ms = capture.scored_ms()
        captures[name] = dict(capture=capture, cycles=cycles, budgets=budgets, generation_ms=generation_ms)
        tables[name] = dict(budget=budget_summary(cycles, budgets, generation_ms), work=work_table(cycles),
                            state=state_changes(cycles), exposure=exchange_exposure(capture),
                            lag_one=lag_one_correlation(cycles), state_digest=light.state_digest(capture),
                            validation=light.validate(capture))
    result = dict(band=BAND, rules=__doc__, ledger=str(ledger_path), warmed_seconds=ledger['warmed_seconds'], captures=tables,
                  monitoring_matched=fixture_effect(ledger, 'monitor-off-1', 'monitor-light-1', 'monitor-light-2', 'monitor-off-2'),
                  admissions_net=fixture_effect(ledger, 'state-normal-1', 'state-held-1', 'state-held-2', 'state-normal-2'))
    long_requests = {}
    for case in ('python-code', 'prose'):
        rows = {arm: steps.get(f'long-{case}-{arm}') for arm in ('off-1', 'light', 'off-2')}
        if all(rows.values()):
            long_requests[case] = {arm: dict(generation_ms=r['summary'].get('generation_ms'), emitted=r['summary'].get('emitted'),
                                             finish=r['summary'].get('finish'), output_sha256=r['summary'].get('output_sha256'),
                                             target_rounds=r['summary'].get('target_rounds')) for arm, r in rows.items()}
    result['long_requests'] = dict(arms=long_requests,
        note='Observation of actual work, not an isolated monitoring test: outputs and routing may differ between arms.')
    # Candidate states against matched current work, one replicate per light capture.
    features = {}
    replicate_names = [n for n in captures if not n.startswith('state-held')]
    for name, level in FEATURES.items():
        replicates = {}
        for capture_name in replicate_names:
            c = captures[capture_name]
            replicates[capture_name] = stratified_excess(c['cycles'], level, c['generation_ms'])
        values = list(replicates.values())
        features[name] = dict(replicates=replicates, classification=classify_observational(values))
    # Context length: exposure is every window at least 128 positions past the capture's start.
    context = {}
    for capture_name in replicate_names:
        c = captures[capture_name]
        start = min(w['position'] for w in c['cycles'])
        context[capture_name] = stratified_excess(c['cycles'], lambda w, s=start: 'late' if w['position'] - s >= 128 else None, c['generation_ms'])
    features['context_later_positions'] = dict(replicates=context, classification=classify_observational(list(context.values())))
    two = {}
    for capture_name in replicate_names:
        c = captures[capture_name]
        one_or_more = [w for w in c['cycles'] if w['exchanges_pending']]
        two[capture_name] = stratified_excess(one_or_more, lambda w: 'two' if w['exchanges_pending'] >= 2 else None, c['generation_ms']) if one_or_more else dict(
            exposed=0, matched=0, coverage=0.0, windows=0, exposure_share=0.0, excess_ms=0.0, se_ms=0.0, value=0.0, se=0.0)
    features['pending_two_exchanges_vs_one'] = dict(replicates=two, classification=classify_observational(list(two.values())))
    result['features'] = features
    # Immediate foreground work: a shared table, tested on work it did not see.
    held_out = {}
    pairs = [('long-prose-light', 'long-python-code-light'), ('long-python-code-light', 'long-prose-light'),
             ('long-prose-light', 'monitor-light-1'), ('long-prose-light', 'state-normal-1'), ('long-prose-light', 'state-held-1')]
    for source, target in pairs:
        if source in captures and target in captures:
            s, t = captures[source], captures[target]
            held_out[f'{source} -> {target}'] = predict_total(s['cycles'], s['budgets'], t['cycles'], t['budgets'])
    result['foreground_table_held_out'] = held_out
    errors = [abs(v['error_fraction']) for v in held_out.values()]
    result['foreground_table_classification'] = ('material_unexplained' if errors and min(errors) >= BAND else
                                                 'within_band' if errors and max(errors) < BAND else 'unresolved')
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('ledger', type=pathlib.Path)
    ap.add_argument('--output', required=True, type=pathlib.Path)
    a = ap.parse_args()
    result = analyze(a.ledger)
    with a.output.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')
    return 0


if __name__ == '__main__':
    sys.exit(main())
