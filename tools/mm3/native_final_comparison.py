#!/usr/bin/env python3
"""Preserved engine against the final configuration across separate loads. Rule fixed in the protocol.

A load is the replicate: calibration, staging and learned state differ between loads, so requests inside
one load are not independent evidence. Each pair is two consecutive loads, one of each engine. Rates pool
emitted tokens over elapsed generation, never averages of rates. Within a pair the effect is new over old, minus one.

Confirmed at a budget: the mean pooled gain is at least the band, its 95% interval across pairs excludes zero,
and neither prompt's mean gain is negative. Anything else is reported as measured and is not a confirmation.
Every cell where the new engine is slower stays listed as a regression.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import statistics
import sys

from native_followup_analysis import interval


def pooled(entries: list[dict]) -> dict:
    tokens = sum(e['emitted'] for e in entries)
    ms = sum(e['generation_ms'] for e in entries)
    return dict(tokens=tokens, ms=ms, tokens_per_second=1000 * tokens / ms if ms else None, requests=len(entries))


def collect(protocol: dict, ledger: dict) -> list[dict]:
    summaries = {s['step']: s['summary'] for s in ledger['steps']}
    rows = []
    for item in protocol['study']['requests']:
        summary = summaries.get(item['step'])
        if summary is None or summary.get('errors') or not summary.get('emitted'):
            raise ValueError(f"missing or failed request {item['step']}: {summary.get('errors') if summary else None}")
        final = summary.get('final_stat') or {}
        rows.append(dict(item, emitted=summary['emitted'], generation_ms=summary['generation_ms'], finish=summary.get('finish'), api_wall_ms=summary.get('api_wall_ms'),
                         target_rounds=summary.get('target_rounds'), drafts_offered=final.get('draft_off'), drafts_accepted=final.get('draft_acc'), output=(summary.get('output_sha256') or '')[:10]))
    return rows


def cell(rows: list[dict], **match) -> list[dict]:
    return [r for r in rows if all(r[k] == v for k, v in match.items())]


def evaluate(rows: list[dict], pairs: list[dict], band: float = 0.02) -> dict:
    budgets = sorted({r['budget'] for r in rows})
    prompts = sorted({r['prompt'] for r in rows})
    arms = sorted({r['arm'] for r in rows})
    loads = {}
    for phase in sorted({r['phase'] for r in rows}):
        chosen = cell(rows, phase=phase)
        loads[phase] = dict(arm=chosen[0]['arm'], budgets={b: dict(pooled(cell(chosen, budget=b)), prompts={p: pooled(cell(chosen, budget=b, prompt=p)) for p in prompts}) for b in budgets})
    result = dict(loads=loads, absolute={}, budgets={}, regressions=[])
    for arm in arms:
        result['absolute'][arm] = {b: dict(pooled(cell(rows, arm=arm, budget=b)), prompts={p: pooled(cell(rows, arm=arm, budget=b, prompt=p)) for p in prompts},
                                           by_load=[loads[ph]['budgets'][b]['tokens_per_second'] for ph in loads if loads[ph]['arm'] == arm]) for b in budgets}
    rate = lambda phase, b, p=None: (loads[phase]['budgets'][b]['prompts'][p] if p else loads[phase]['budgets'][b])['tokens_per_second']
    for b in budgets:
        effects = [rate(pair['new'], b) / rate(pair['old'], b) - 1 for pair in pairs]
        stats = interval(effects)
        per_prompt = {p: [rate(pair['new'], b, p) / rate(pair['old'], b, p) - 1 for pair in pairs] for p in prompts}
        prompt_means = {p: statistics.fmean(v) for p, v in per_prompt.items()}
        confirmed = stats['low'] is not None and stats['mean'] >= band and stats['low'] > 0 and all(v >= 0 for v in prompt_means.values())
        result['budgets'][b] = dict(stats, effects=effects, per_prompt=per_prompt, prompt_means=prompt_means, confirmed=confirmed)
        for p, values in per_prompt.items():
            for pair, value in zip(pairs, values):
                if value < 0:
                    result['regressions'].append(dict(budget=b, prompt=p, pair=pair, effect=value))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--protocol', required=True, type=pathlib.Path)
    ap.add_argument('--ledger', required=True, type=pathlib.Path)
    ap.add_argument('--output', required=True, type=pathlib.Path)
    a = ap.parse_args()
    protocol = json.loads(a.protocol.read_text())
    rows = collect(protocol, json.loads(a.ledger.read_text()))
    result = evaluate(rows, protocol['study']['pairs'])
    result['rows'] = rows
    a.output.write_text(json.dumps(result, indent=1) + '\n')
    for b, value in result['budgets'].items():
        print(f"budget {b}: mean {100*value['mean']:+.2f}% [{100*value['low']:+.2f}, {100*value['high']:+.2f}] pairs {[round(100*e, 2) for e in value['effects']]} prompts {({k: round(100*v, 2) for k, v in value['prompt_means'].items()})} confirmed {value['confirmed']}")
    for arm, value in result['absolute'].items():
        for b, x in value.items():
            print(f"{arm} budget {b}: {x['tokens_per_second']:.3f} tok/s pooled, by load {[round(v, 3) for v in x['by_load']]}, prompts {({k: round(v['tokens_per_second'], 3) for k, v in x['prompts'].items()})}")
    print('regressions', [(r['budget'], r['prompt'], round(100 * r['effect'], 2)) for r in result['regressions']])
    return 0


if __name__ == '__main__':
    sys.exit(main())
