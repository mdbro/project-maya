#!/usr/bin/env python3
"""Candidate arms against a control on complete generation, per block. Rule fixed in the protocol.

Each block holds every arm of one prompt in one order. The effect of an arm is its milliseconds per emitted
token over the same block's control, minus one. A candidate qualifies at -2% or better with a 95% interval
excluding zero and neither prompt slower. Qualifying is only a speed result: a value that changes
routing or acceptance is selected only after the bounded usability checks pass on complete tasks.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import statistics
import sys

from native_followup_analysis import interval


def effects(rows: list[dict], control: str, band: float = 0.02) -> dict:
    result = {}
    arms = [a for a in rows[0]['arms'] if a != control]
    for arm in arms:
        differences = [row['arms'][arm]['rate'] / row['arms'][control]['rate'] - 1 for row in rows]
        stats = interval(differences)
        prompts: dict[str, list[float]] = {}
        for row, value in zip(rows, differences):
            prompts.setdefault(row['prompt'], []).append(value)
        per_prompt = {k: statistics.fmean(v) for k, v in prompts.items()}
        qualifies = stats['low'] is not None and stats['mean'] <= -band and stats['high'] < 0 and all(v < 0 for v in per_prompt.values())
        result[arm] = dict(stats, differences=differences, per_prompt=per_prompt, qualifies=qualifies)
    return result


def evaluate(protocol: dict, ledger: dict, screen: str) -> dict:
    entries = {s['step']: s['summary'] for s in ledger['steps']}
    study = protocol['screens'][screen]
    rows = []
    for block in study['blocks']:
        arms = {}
        for arm, step in block['arms'].items():
            summary = entries[step]
            final = summary.get('final_stat') or {}
            arms[arm] = dict(rate=summary['generation_ms'] / summary['emitted'], generation_ms=summary['generation_ms'], emitted=summary['emitted'], finish=summary.get('finish'),
                             drafts_offered=final.get('draft_off'), drafts_accepted=final.get('draft_acc'), target_rounds=final.get('target_rounds'), output=summary.get('output_sha256', '')[:10], errors=summary.get('errors'))
        rows.append(dict(prompt=block['prompt'], order=block['order'], arms=arms))
    result = effects(rows, study['control'])
    rates = {arm: {p: statistics.fmean(1000 / r['arms'][arm]['rate'] for r in rows if r['prompt'] == p) for p in {r['prompt'] for r in rows}} for arm in rows[0]['arms']}
    accept = {arm: statistics.fmean(float(r['arms'][arm]['drafts_accepted'] or 0) / max(1.0, float(r['arms'][arm]['drafts_offered'] or 0)) for r in rows) for arm in rows[0]['arms']}
    return dict(rows=rows, effects=result, tokens_per_second=rates, acceptance=accept)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--protocol', required=True, type=pathlib.Path)
    ap.add_argument('--ledger', required=True, type=pathlib.Path)
    ap.add_argument('--screen', required=True)
    ap.add_argument('--output', required=True, type=pathlib.Path)
    a = ap.parse_args()
    result = evaluate(json.loads(a.protocol.read_text()), json.loads(a.ledger.read_text()), a.screen)
    a.output.write_text(json.dumps(result, indent=1) + '\n')
    for arm, value in result['effects'].items():
        print(f"{arm:10s} mean {100*value['mean']:+.2f}% [{100*value['low']:+.2f}, {100*value['high']:+.2f}] qualifies {value['qualifies']}", {k: round(100*v, 2) for k, v in value['per_prompt'].items()})
    print('tokens per second', {arm: {k: round(v, 2) for k, v in value.items()} for arm, value in result['tokens_per_second'].items()})
    print('acceptance', {k: round(v, 3) for k, v in result['acceptance'].items()})
    return 0


if __name__ == '__main__':
    sys.exit(main())
