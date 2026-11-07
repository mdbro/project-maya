#!/usr/bin/env python3
"""Main effects of two per-request controls on complete generation. Rule fixed in the protocol.

Each block holds all four arms of one prompt. A main effect is the mean of the two arms with the factor
over the mean of the two without it, minus one, on milliseconds per emitted token. Blocks are replicates.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import statistics
import sys

from native_followup_analysis import interval


def rate(summary: dict) -> float:
    return summary['generation_ms'] / summary['emitted']


def main_effects(rows: list[dict], factors: dict[str, tuple[list[str], list[str]]], band: float = 0.02) -> dict:
    result = {}
    for factor, (on, off) in factors.items():
        differences = [statistics.fmean(row[a]['rate'] for a in on) / statistics.fmean(row[a]['rate'] for a in off) - 1 for row in rows]
        stats = interval(differences)
        per_prompt: dict[str, list[float]] = {}
        for row, value in zip(rows, differences):
            per_prompt.setdefault(row['prompt'], []).append(value)
        prompts = {k: statistics.fmean(v) for k, v in per_prompt.items()}
        useful = stats['low'] is not None and stats['mean'] <= -band and stats['high'] < 0 and all(v < 0 for v in prompts.values())
        result[factor] = dict(stats, differences=differences, per_prompt=prompts, selected=useful)
    return result


def evaluate(protocol: dict, ledger: dict) -> dict:
    entries = {s['step']: s['summary'] for s in ledger['steps']}
    rows = []
    for block in protocol['gate_study']['blocks']:
        row = dict(prompt=block['prompt'], order=block['order'])
        for arm, step in block['arms'].items():
            summary = entries[step]
            final = summary.get('final_stat') or {}
            row[arm] = dict(rate=rate(summary), generation_ms=summary['generation_ms'], emitted=summary['emitted'], finish=summary.get('finish'), output=summary.get('output_sha256', '')[:10],
                            drafts_offered=final.get('draft_off'), drafts_accepted=final.get('draft_acc'), target_rounds=final.get('target_rounds'), draft_time_ms=final.get('draft_time_ms'), target_time_ms=final.get('target_time_ms'), errors=summary.get('errors'))
        rows.append(row)
    effects = main_effects(rows, dict(query=(['query', 'both'], ['base', 'spinner']), spinner=(['spinner', 'both'], ['base', 'query'])))
    both = [row['both']['rate'] / row['base']['rate'] - 1 for row in rows]
    summary = {arm: {prompt: statistics.fmean(1000 / row[arm]['rate'] for row in rows if row['prompt'] == prompt) for prompt in {r['prompt'] for r in rows}} for arm in ('base', 'query', 'spinner', 'both')}
    return dict(rows=rows, effects=effects, both_over_base=interval(both), tokens_per_second=summary)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--protocol', required=True, type=pathlib.Path)
    ap.add_argument('--ledger', required=True, type=pathlib.Path)
    ap.add_argument('--output', required=True, type=pathlib.Path)
    a = ap.parse_args()
    result = evaluate(json.loads(a.protocol.read_text()), json.loads(a.ledger.read_text()))
    a.output.write_text(json.dumps(result, indent=1) + '\n')
    for name, value in result['effects'].items():
        print(name, f"mean {100*value['mean']:+.2f}% [{100*value['low']:+.2f}, {100*value['high']:+.2f}] selected {value['selected']}", {k: round(100*v, 2) for k, v in value['per_prompt'].items()})
    print('both over base', {k: (round(100*v, 2) if isinstance(v, float) else v) for k, v in result['both_over_base'].items()})
    print('tokens per second', {arm: {k: round(v, 2) for k, v in value.items()} for arm, value in result['tokens_per_second'].items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
