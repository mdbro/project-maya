#!/usr/bin/env python3
"""Fit, freeze and judge the shared generation model's policy predictions. Primary-agent tool.

Predictions come first and are frozen with a digest before any intervention runs. A candidate's
work comes from the production policy replayed over recorded demand with a declared order rule,
never from a control run's future. Timing prices that work with one shared per-pass formula.

Decision classes for a candidate-minus-control fraction of complete generation, band 2%:
    predicted   better <= -band; worse >= +band; tie otherwise
    observed    better: mean <= -band and the 95% interval excludes zero
                worse:  mean >= +band and the 95% interval excludes zero
                tie:    the interval lies inside +-band
                inconclusive otherwise, which never counts as agreement or as a tie
A round qualifies the model for this operating domain only if all of these hold:
    no reversal between a predicted and an observed better or worse
    at least 80% class agreement over conclusive cells
    at least two cells predicted non-tie and confirmed in that direction
    median absolute difference error at most 1.5 points and 90th percentile at most 3 points
    median absolute miss on control generation at most 5%
Anything else is a recorded failure. Its passes and work are compared before any repair, and only
workloads no prediction has touched can validate a repair.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import pathlib
import statistics
import sys

import native_followup_analysis as analysis
import native_light as light
import native_passes as passes
from cooperative_replay.policy import PolicyConfig
from shared_model import grouped, timing, work

BAND = 0.02


def fit_model(ledger_paths: list[pathlib.Path], steps_by_ledger: dict[str, list[str]] | None = None, *, regime: str = 'query_on',
              copy_lead_ms: float = 0.0) -> timing.SharedTiming:
    tables, budgets, activity = {}, {}, {}
    for path in ledger_paths:
        ledger = json.loads(path.read_text())
        wanted = (steps_by_ledger or {}).get(str(path))
        for entry in ledger['steps']:
            summary = entry['summary']
            if summary.get('capture') != 'light' or summary.get('errors') or (wanted is not None and entry['step'] not in wanted):
                continue
            capture = light.read(summary['artifact'])
            name = f'{path.parent.name}/{entry["step"]}'
            tables[name] = passes.passes(capture)
            budgets[name] = passes.forward_budgets(capture)
            activity[name] = timing.activity(capture)
    if not tables:
        raise ValueError('no training captures')
    return timing.fit(tables, budgets, activity, regime=regime, copy_lead_ms=copy_lead_ms)


def candidate_config(arm: dict) -> tuple[PolicyConfig, bool, bool]:
    config = PolicyConfig(cpu_scale=arm.get('cpu_scale', 1.3), stage_limit=arm.get('stage_limit', 4), refill_budget=arm.get('budget', 8), margin_ms=arm.get('margin_ms', 0.0))
    return config, bool(arm.get('hold_primary')), bool(arm.get('hold_helper'))


def predict_cell(model, snapshot: dict, fixture_path, arm: dict, rule: work.OrderRule) -> dict:
    config, hold_primary, hold_helper = candidate_config(arm)
    events = work.events_from_fixture(fixture_path)
    replayed = work.replay_work(snapshot, events, config, rule, hold_primary=hold_primary, hold_helper=hold_helper)
    pricing = grouped.price if isinstance(model, grouped.GroupedTiming) else timing.price
    priced = pricing(model, replayed['passes'], issued_end=replayed['issued_end'], leases=replayed['counts'].get('helper_issued', 0))
    picks = dict(cpu=0, helper=0, primary=0, new=0)
    for p in replayed['passes']:
        if p['generation']:
            for key in picks:
                picks[key] += p[key]
    return dict(priced=priced, picks=picks, counts=dict(replayed['counts']), pending_primary=replayed['pending_primary'], pending_helper=replayed['pending_helper'])


def classify_predicted(delta: float, band: float = BAND) -> str:
    return 'better' if delta <= -band else 'worse' if delta >= band else 'tie'


def classify_observed(cell: dict, band: float = BAND) -> str:
    if cell.get('low') is None:
        return 'inconclusive'
    if cell['mean'] <= -band and cell['high'] < 0:
        return 'better'
    if cell['mean'] >= band and cell['low'] > 0:
        return 'worse'
    if cell['low'] > -band and cell['high'] < band:
        return 'tie'
    return 'inconclusive'


def pooled_cells(blocks: dict[str, list[dict]]) -> dict:
    """Paired fractions within a block. Interval from a standard deviation pooled across cells."""
    cells = {}
    squares, degrees = 0.0, 0
    for key, rows in blocks.items():
        differences = [row['candidate'] / row['control'] - 1 for row in rows]
        mean = statistics.fmean(differences)
        cells[key] = dict(differences=differences, mean=mean, n=len(differences))
        squares += sum((d - mean) ** 2 for d in differences)
        degrees += len(differences) - 1
    pooled = math.sqrt(squares / degrees) if degrees else None
    for cell in cells.values():
        half = analysis.t95(degrees) * pooled / math.sqrt(cell['n']) if pooled is not None else None
        cell.update(sd=pooled, df=degrees, half_width=half, low=cell['mean'] - half if half is not None else None,
                    high=cell['mean'] + half if half is not None else None)
    return cells


def judge(predictions: dict, observed: dict, control_miss: list[float], band: float = BAND) -> dict:
    """predictions: key -> predicted fraction. observed: key -> cell. Applies the rule in the docstring."""
    rows = []
    for key, delta in predictions.items():
        cell = observed[key]
        rows.append(dict(key=key, predicted=delta, predicted_class=classify_predicted(delta, band), observed=cell['mean'], low=cell.get('low'), high=cell.get('high'),
                         observed_class=classify_observed(cell, band), error=delta - cell['mean']))
    conclusive = [r for r in rows if r['observed_class'] != 'inconclusive']
    agreement = sum(r['predicted_class'] == r['observed_class'] for r in conclusive) / len(conclusive) if conclusive else None
    reversal = [r['key'] for r in rows if {r['predicted_class'], r['observed_class']} == {'better', 'worse'}]
    confirmed = [r['key'] for r in rows if r['predicted_class'] != 'tie' and r['predicted_class'] == r['observed_class']]
    errors = sorted(abs(r['error']) for r in rows)
    median_error = statistics.median(errors) if errors else None
    p90_error = errors[min(len(errors) - 1, int(0.9 * (len(errors) - 1) + 0.999))] if errors else None
    median_miss = statistics.median(abs(m) for m in control_miss) if control_miss else None
    checks = dict(no_reversal=not reversal, agreement=agreement is not None and agreement >= 0.8, discrimination=len(confirmed) >= 2,
                  median_error=median_error is not None and median_error <= 0.015, p90_error=p90_error is not None and p90_error <= 0.03,
                  control_miss=median_miss is not None and median_miss <= 0.05)
    return dict(rows=rows, conclusive=len(conclusive), agreement=agreement, reversal=reversal, confirmed=confirmed, median_error=median_error, p90_error=p90_error,
                median_control_miss=median_miss, control_miss=control_miss, checks=checks, qualified=all(checks.values()))


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def command_predict(a) -> int:
    protocol = json.loads(a.protocol.read_text())
    round_ = protocol['rounds'][a.round]
    values = json.loads(a.model.read_text())
    model = grouped.GroupedTiming.from_dict(values) if 'cpu_per_column' in values else timing.SharedTiming.from_dict(values)
    snapshot = work.load_snapshot(protocol['origin_snapshot'])
    rule_values = json.loads(a.order_rule.read_text())
    rule = work.OrderRule(**rule_values)
    cells = {}
    for name in round_['workloads']:
        for arm, values in protocol['arms'].items():
            print(name, arm, flush=True)
            cells[f'{name}/{arm}'] = predict_cell(model, snapshot, a.run_dir / f'fixture-{name}-{arm}.replay', values, rule)
    deltas = {}
    for name in round_['workloads']:
        base = cells[f'{name}/control']['priced']['total_ms']
        for arm in protocol['arms']:
            if arm != 'control':
                deltas[f'{name}/{arm}'] = cells[f'{name}/{arm}']['priced']['total_ms'] / base - 1
    document = dict(round=a.round, model=json.loads(a.model.read_text()), model_sha256=hashlib.sha256(a.model.read_bytes()).hexdigest(), order_rule=rule_values,
                    protocol_sha256=hashlib.sha256(a.protocol.read_bytes()).hexdigest(), cells=cells, deltas=deltas,
                    classes={k: classify_predicted(v) for k, v in deltas.items()})
    document['digest'] = digest(dict(cells=cells, deltas=deltas))
    with a.output.open('x') as f:
        json.dump(document, f, indent=1)
        f.write('\n')
    print(json.dumps(dict(digest=document['digest'], deltas={k: round(100 * v, 2) for k, v in deltas.items()}, classes=document['classes']), indent=1))
    return 0


def command_evaluate(a) -> int:
    protocol = json.loads(a.protocol.read_text())
    ledger = json.loads(a.ledger.read_text())
    frozen = json.loads(a.predictions.read_text())
    if digest(dict(cells=frozen['cells'], deltas=frozen['deltas'])) != frozen['digest']:
        raise ValueError('frozen predictions changed')
    entries = {s['step']: s for s in ledger['steps']}
    round_ = protocol['rounds'][frozen['round']]
    blocks: dict[str, list[dict]] = {}
    miss = []
    arm_totals: dict[str, list[float]] = {}
    for name in round_['workloads']:
        for block in round_['blocks'][name]:
            control = entries[block['control']]['summary']
            for arm, step in block.items():
                if arm == 'control':
                    continue
                summary = entries[step]['summary']
                if summary.get('errors') or control.get('errors'):
                    continue
                blocks.setdefault(f'{name}/{arm}', []).append(dict(candidate=summary['generation_ms'], control=control['generation_ms']))
            arm_totals.setdefault(name, []).append(control['generation_ms'])
        predicted = frozen['cells'][f'{name}/control']['priced']['total_ms']
        miss += [predicted / value - 1 for value in arm_totals[name]]
    observed = pooled_cells(blocks)
    verdict = judge(frozen['deltas'], observed, miss)
    verdict['observed_cells'] = observed
    verdict['predictions_digest'] = frozen['digest']
    # Work actually executed against predicted, from each arm's own report.
    comparison = {}
    for name in round_['workloads']:
        for arm in protocol['arms']:
            reports = [entries[b[arm]]['summary']['report'] for b in round_['blocks'][name]]
            predicted = frozen['cells'][f'{name}/{arm}']
            comparison[f'{name}/{arm}'] = dict(
                predicted_copies=predicted['priced']['copies'], observed_uploads=[r['demand_uploads'] for r in reports],
                predicted_issued=predicted['counts'].get('primary_issued', 0), observed_issued=[r['primary_issued'] for r in reports],
                predicted_leases=predicted['counts'].get('helper_issued', 0), observed_leases=[r['helper_started'] for r in reports],
                predicted_ms=predicted['priced']['total_ms'], observed_ms=[r['generation_target_ms'] for r in reports])
    verdict['work'] = comparison
    text = json.dumps(verdict, indent=1, default=str)
    a.output.write_text(text + '\n')
    print(json.dumps({k: verdict[k] for k in ('agreement', 'reversal', 'confirmed', 'median_error', 'p90_error', 'median_control_miss', 'checks', 'qualified')}, indent=1))
    for row in verdict['rows']:
        print(f"{row['key']:30s} predicted {100*row['predicted']:+6.2f}% {row['predicted_class']:6s} observed {100*row['observed']:+6.2f}% [{100*row['low']:+6.2f},{100*row['high']:+6.2f}] {row['observed_class']:12s} error {100*row['error']:+6.2f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    p = sub.add_parser('fit')
    p.add_argument('ledgers', nargs='+', type=pathlib.Path)
    p.add_argument('--steps', type=pathlib.Path, help='JSON map of ledger path to the steps to train on')
    p.add_argument('--regime', default='query_on')
    p.add_argument('--copy-lead-ms', type=float, default=0.0, help='demand copy start minus CPU batch start, from a trace (HAGS on: negative)')
    p.add_argument('--output', required=True, type=pathlib.Path)
    p = sub.add_parser('predict')
    p.add_argument('--protocol', required=True, type=pathlib.Path)
    p.add_argument('--model', required=True, type=pathlib.Path)
    p.add_argument('--run-dir', required=True, type=pathlib.Path)
    p.add_argument('--round', required=True)
    p.add_argument('--order-rule', required=True, type=pathlib.Path, help='rule chosen on training captures before predicting')
    p.add_argument('--output', required=True, type=pathlib.Path)
    p = sub.add_parser('evaluate')
    p.add_argument('--protocol', required=True, type=pathlib.Path)
    p.add_argument('--ledger', required=True, type=pathlib.Path)
    p.add_argument('--predictions', required=True, type=pathlib.Path)
    p.add_argument('--output', required=True, type=pathlib.Path)
    a = ap.parse_args()
    if a.command == 'fit':
        steps = json.loads(a.steps.read_text()) if a.steps else None
        model = fit_model(a.ledgers, steps, regime=a.regime, copy_lead_ms=a.copy_lead_ms)
        with a.output.open('x') as f:
            f.write(model.to_json() + '\n')
        return 0
    return command_predict(a) if a.command == 'predict' else command_evaluate(a)


if __name__ == '__main__':
    sys.exit(main())
