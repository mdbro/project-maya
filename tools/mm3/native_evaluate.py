#!/usr/bin/env python3
"""The shared evaluator for timed runs on the native MiniMax engine. New runs use it instead of a per-run evaluate.py.

    native_evaluate.py RUN_DIR [--text FILE] [--json FILE] [--by-layer]
    native_evaluate.py RUN_DIR --primary ARM:BASELINE [--compare ARM:BASELINE ...]
            (only for a run whose protocol declares no primary comparison: an old run; the report says it was named after the run)
    native_evaluate.py --across A=RUN_DIR B=RUN_DIR A=RUN_DIR B=RUN_DIR --arm ARM [--anchor one-row]

Why it exists (the audit of 2026-11-12: every timed run of 2026-11-11/12 used one seed for every request, so repeated
blocks wrote the same text and "n 9" was three prompts; the per-run evaluators paired blocks across separate runs,
counted repeated texts as independent, pooled unlike arms and reported the better of two arms). What it does instead:

  Pairing        inside one run only, by block seed and prompt: a unit is (prompt, seed). Requests of one arm that share
                 a unit (blocks that reused a seed) are averaged inside it, so a run with one seed for every block has
                 one unit a prompt. Units of one prompt whose texts repeat across seeds are merged into one.
  n              the number of units: prompts x seeds after merging. Each comparison is reported per prompt (n = seeds)
                 and over all units, with the exact 95% t interval.
  Snapshots      when the two arms of a unit restored different snapshots (a design that re-warms per arm), units that
                 share a snapshot pair are not independent: the interval over snapshot pairs is reported beside, and it
                 is the one to read.
  Primary        the comparison or comparisons the protocol declares in study.primary (one, or a list). Every other
                 declared comparison is reported as secondary. No arm is ever chosen as the best of several; a declared
                 selection picks its arm on its selection blocks (by the lowest mean ms a token, or the highest mean
                 tokens per second with measure "tok/s") and is compared on the other blocks only.
  Decision       every comparison is given in ms a token (arm / base - 1) and in tokens per second (base / arm - 1, per
                 unit); study.decision "tokens per second" marks the latter as the protocol's decision figure.
  A/A            the protocol's study.aa pair (identical settings) is evaluated like any comparison: its interval should
                 hold zero, and its spread is the run's own floor.
  Settings       every request's final statistics are checked against the arm's declared settings (study.expect, or the
                 arm's tune for an older run); a request that ran something else is left out and listed.
  Token time     ms a token from token 128: the engine's own interval from the first round completed at or after 128
                 emitted tokens. Answers that stopped early (a shorter window) are listed.
  Per pass       from light captures, passes of forwards at generated-token index 128 or later (position minus the
                 first generation forward's position; in drafting a forward commits several tokens, so the forward's
                 ordinal is not the token index). Per layer: each pass's finish time from its own join (join to the next
                 join), and the mean abs(difference) between the completion times of the lanes a light capture
                 records - the CPU lane, the 3090, the 4080 and the 5090's copy-plus-primary path - from that pass's
                 common origin, and between the slowest and the second-slowest of them. The 5090's completion is the
                 layer record's gpu_expert_end (the 5090 finished its own expert compute, after this pass's uploads),
                 which light capture format 2 records; a format 1 capture has no 5090 stamps, and its 5090 pairs show
                 no passes. The host's wait at the next join is reported too (in a format 1 capture the only place
                 5090 work that outlasts the lanes shows); it is not a completion time.
  Across runs    refused unless the engine starts alternate (A, B, A, B: each label at least twice, never twice in a
                 row, each run a separate engine start) and every start has the anchor arm with the same prompts and
                 seeds. Then: each start's figures, the start-to-start spread of the anchor and of the arm between
                 starts of one configuration, and B over A with an interval over start pairs.

Standard library only (and native_light.py beside it). Everything printed is measured, except the intervals, which are
the usual t intervals over the units named."""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import native_light as light  # noqa: E402

MEASURE_FROM = 128
LANES = ('cpu', '3090', '4080', '5090')   # (5090: copy-plus-primary, gpu_expert_end; light capture format 2 only)
PAIRS = (('cpu', '3090'), ('cpu', '4080'), ('3090', '4080'), ('cpu', '5090'), ('3090', '5090'), ('4080', '5090'))
# Settings an older protocol's tune can be checked by: the final statistics report them in the request's own terms.
CHECKED_KEYS = ('spec_policy', 'spec_plan', 'routing_prior_bonus', 'routing_dampen', 'routing_prompt_bonus', 'helper_live',
                'third', 'primary_arithmetic', 'early_copy', 'helper_zero_copy', 'dense_split', 'gpu_nucleus', 'expert_graph',
                'primary_submit_query', 'helper_kernel', 'helper_kernel_one', 'third_kernel', 'third_kernel_one',
                'helper_kernel_loads', 'third_kernel_loads')


# ---------------------------------------------------------------- statistics: the exact t quantile, no table

def _betacf(a: float, b: float, x: float) -> float:
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        d = 1.0 / (d if abs(d) > tiny else tiny)
        c = 1.0 + aa / c
        c = c if abs(c) > tiny else tiny
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-15:
            break
    return h


def betainc(a: float, b: float, x: float) -> float:
    """The regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def t_cdf(t: float, df: int) -> float:
    tail = 0.5 * betainc(df / 2.0, 0.5, df / (df + t * t))
    return 1.0 - tail if t > 0 else tail


def t_quantile(q: float, df: int) -> float:
    """The q quantile of Student's t with df degrees of freedom, by bisection (df 17 gives 2.110, not a table's 2.101)."""
    if df < 1:
        return math.inf
    lo, hi = 0.0, 1e6
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if t_cdf(mid, df) < q else (lo, mid)
    return (lo + hi) / 2


def interval(values: list[float]) -> dict:
    """Mean and the exact two-sided 95% t interval over independent values."""
    n = len(values)
    if n == 0:
        return dict(n=0, mean=None, sd=None, low=None, high=None)
    mean = statistics.fmean(values)
    if n < 2:
        return dict(n=1, mean=mean, sd=None, low=None, high=None)
    sd = statistics.stdev(values)
    half = t_quantile(0.975, n - 1) * sd / math.sqrt(n)
    return dict(n=n, mean=mean, sd=sd, low=mean - half, high=mean + half)


def pct(x: float | None) -> str:
    return '-' if x is None else f'{100 * x:+.1f}%'


def ci_text(c: dict) -> str:
    if not c or c.get('mean') is None:
        return '-'
    if c.get('low') is None:
        return f"{pct(c['mean'])} (n {c['n']})"
    return f"{pct(c['mean'])} [{pct(c['low'])}, {pct(c['high'])}] (n {c['n']})"


def holds_zero(c: dict) -> bool | None:
    return None if c.get('low') is None else c['low'] <= 0.0 <= c['high']


# ---------------------------------------------------------------- one run's requests

def arm_of(q: dict) -> str | None:
    """The arm a study entry names. Older protocols name it by other fields (order, profile, third, default, kernel)."""
    if q.get('arm'):
        return str(q['arm'])
    for key in ('order', 'profile', 'kernel'):
        if q.get(key):
            return str(q[key])
    for key in ('third', 'default'):
        if key in q:
            return f'{key}{q[key]}'
    return None


def approx_equal(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


class Run:
    """A run folder: protocol.json, ledger.json and the step files. Nothing in it is changed."""

    def __init__(self, path: pathlib.Path, include_captured: bool = False):
        self.path = pathlib.Path(path).resolve()
        self.protocol = json.loads((self.path / 'protocol.json').read_text())
        ledger = self.path / 'ledger.json'
        self.ledger = json.loads(ledger.read_text()) if ledger.exists() else dict(steps=[], phases={})
        self.steps = {s['step']: s for s in self.ledger.get('steps', [])}
        self.sequence = {s['step']: s for s in self.protocol.get('sequence', [])}
        self.study = self.protocol.get('study') or {}
        self.arms = self.study.get('arms') or self.protocol.get('arms') or {}
        self.notes: list[str] = []
        self.requests = self._requests(include_captured)

    def engine_pids(self) -> set:
        return {p.get('precheck', {}).get('pid') for p in self.ledger.get('phases', {}).values() if p.get('precheck')}

    def executables(self) -> set:
        return {p['precheck']['executable_sha256'] for p in self.ledger.get('phases', {}).values()
                if (p.get('precheck') or {}).get('executable_sha256')}

    def started(self) -> str:
        times = [s.get('started_utc') for s in self.ledger.get('steps', []) if s.get('started_utc')]
        return min(times) if times else self.ledger.get('created_utc', '')

    def tune_of(self, step: str) -> dict:
        return {**(self.protocol.get('tune') or {}), **((self.sequence.get(step) or {}).get('tune') or {})}

    def seed_of(self, q: dict) -> int | None:
        for value in (q.get('seed'), (self.sequence.get(q['step']) or {}).get('seed'), self.protocol.get('seed')):
            if value is not None:
                return int(value)
        return None

    def expected(self, arm: str, step: str) -> dict:
        """The settings a request of this arm must report: the protocol's study.expect, else what its tune sets among
        the keys the final statistics report in the same terms."""
        declared = (self.study.get('expect') or {}).get(arm)
        if declared is not None:
            return dict(declared)
        tune, out = self.tune_of(step), {}
        for key in CHECKED_KEYS:
            if key in tune:
                out[key] = tune[key]
        if tune.get('spec_policy') == 'lenience' and 'spec_ell' in tune:
            out['spec_ell'] = tune['spec_ell']
        if 'spec' in tune:
            out['drafting'] = int(tune['spec']) > 1
        return out

    @staticmethod
    def check(expect: dict, final: dict) -> list[str]:
        wrong = []
        for key, want in expect.items():
            if key == 'drafting':
                have = float(final.get('spec_rounds') or 0) > 0
                if have != bool(want):
                    wrong.append(f"drafting {'on' if have else 'off'}, expected {'on' if want else 'off'}")
            elif key == 'drafts_per_round':
                rounds = float(final.get('spec_rounds') or 0)
                have = float(final.get('draft_off') or 0) / rounds if rounds else 0.0
                lo, hi = want
                if not lo <= have <= hi:
                    wrong.append(f'drafts offered a drafting round {have:.2f}, expected {lo:g} to {hi:g}')
            elif key not in final:
                wrong.append(f'{key} not reported (expected {want})')
            elif not approx_equal(final[key], want):
                wrong.append(f'{key} {final[key]} (expected {want})')
        return wrong

    def _requests(self, include_captured: bool) -> list[dict]:
        lists = [('requests', 'timed'), ('mechanism', 'mechanism'), ('per_pass', 'light'), ('timeline', 'timeline')]
        out = []
        for name, kind in lists:
            for q in self.study.get(name, []) or []:
                arm = arm_of(q)
                entry = self.steps.get(q['step'])
                summary = (entry or {}).get('summary') or {}
                final = summary.get('final_stat') or {}
                capture = summary.get('capture') or (self.sequence.get(q['step']) or {}).get('capture', 'off')
                r = dict(step=q['step'], arm=arm, prompt=q.get('prompt') or summary.get('case'), block=q.get('block'),
                         seed=self.seed_of(q), list=kind, capture=capture, ran=entry is not None, why=[], final=final,
                         sha=summary.get('output_sha256'), artifact=summary.get('artifact'), emitted=summary.get('emitted'),
                         finish=summary.get('finish'), tokens=summary.get('interval_tokens'), start=summary.get('interval_start_emitted'),
                         restore=(summary.get('restore') or {}).get('signature'), validation=summary.get('validation') or {})
                if entry is None:
                    r['why'].append('not run')
                if summary.get('errors'):
                    r['why'].append(f"step errors: {summary['errors']}")
                if summary.get('interval_status') == 'measured' and summary.get('interval_tokens'):
                    r['rate'] = float(summary['interval_ms']) / float(summary['interval_tokens'])
                else:
                    r['rate'] = None
                    if entry is not None:
                        r['why'].append(f"no interval from token {summary.get('measure_from')} ({summary.get('interval_status')})")
                if entry is not None and final:
                    r['why'] += self.check(self.expected(arm, q['step']), final)
                emitted = float(r['emitted'] or 0)
                r['loads'] = (float(final.get('decode_staged') or 0) + float(final.get('decode_cpu_experts') or 0)) / emitted if emitted else None
                r['rounds_per_token'] = float(final.get('target_rounds') or 0) / emitted if emitted else None
                r['offered'] = float(final.get('draft_off') or 0)
                r['accepted'] = float(final.get('draft_acc') or 0)
                r['ok'] = not r['why']
                r['timed'] = kind == 'timed' and (capture == 'off' or include_captured)
                out.append(r)
        return out

    def timed(self) -> list[dict]:
        return [r for r in self.requests if r['timed']]


# ---------------------------------------------------------------- units, repeated texts, comparisons

def text_report(rows: list[dict]) -> list[dict]:
    """Per arm and prompt: requests, seeds, distinct texts; groups whose every block wrote one text; texts repeated
    across seeds (a seed that did not draw a new text)."""
    groups = collections.defaultdict(list)
    for r in rows:
        if r['ran'] and r['sha']:
            groups[(r['arm'], r['prompt'])].append(r)
    out = []
    for (arm, prompt), g in sorted(groups.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        by_sha = collections.defaultdict(set)
        for r in g:
            by_sha[r['sha']].add(r['seed'])
        out.append(dict(arm=arm, prompt=prompt, requests=len(g), seeds=len({r['seed'] for r in g}), texts=len(by_sha),
                        one_text=len(g) > 1 and len(by_sha) == 1,
                        repeated_across_seeds=sum(1 for seeds in by_sha.values() if len(seeds) > 1)))
    return out


class _Merge:
    def __init__(self):
        self.parent = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def join(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def units_of(rows: list[dict], arm: str) -> dict:
    """(prompt, seed) -> that arm's valid timed requests in the unit."""
    out = collections.defaultdict(list)
    for r in rows:
        if r['arm'] == arm and r['ok'] and r['rate'] is not None:
            out[(r['prompt'], r['seed'])].append(r)
    return out


def compare(rows: list[dict], arm: str, base: str, blocks: set | None = None) -> dict:
    """arm over base, ms a token from token 128: paired by (prompt, seed) inside one run, units merged where a text
    repeats across seeds, per prompt and over all units; the interval over snapshot pairs when the arms restored
    different snapshots."""
    if blocks is not None:
        rows = [r for r in rows if r['block'] in blocks]
    ua, ub = units_of(rows, arm), units_of(rows, base)
    common = sorted(set(ua) & set(ub), key=str)
    merge = _Merge()
    for prompt in {p for p, _ in common}:
        keys = [k for k in common if k[0] == prompt]
        for side in (ua, ub):
            seen = {}
            for k in keys:
                for r in side[k]:
                    if r['sha'] in seen and seen[r['sha']] != k:
                        merge.join(k, seen[r['sha']])
                    seen.setdefault(r['sha'], k)
    groups = collections.defaultdict(list)
    for k in common:
        groups[merge.find(k)].append(k)
    units = []
    for root, keys in groups.items():
        ra = [r for k in keys for r in ua[k]]
        rb = [r for k in keys for r in ub[k]]
        ma, mb = statistics.fmean(r['rate'] for r in ra), statistics.fmean(r['rate'] for r in rb)
        sig = (tuple(sorted({str(r['restore']) for r in ra})), tuple(sorted({str(r['restore']) for r in rb})))
        units.append(dict(prompt=root[0], seeds=sorted({k[1] for k in keys}, key=str), arm_ms=ma, base_ms=mb, effect=ma / mb - 1,
                          tps_effect=mb / ma - 1,
                          requests=(len(ra), len(rb)), snapshots=sig, cross_snapshot=sig[0] != sig[1],
                          # (an answer that ended before the token budget is timed over a shorter, different window)
                          short_window=[f"{r['step']} ({int(float(r['emitted'] or 0))} tokens)" for r in ra + rb
                                        if r['finish'] and r['finish'] != 'length']))
    prompts = sorted({u['prompt'] for u in units}, key=str)
    per_prompt = {p: dict(interval=interval([u['effect'] for u in units if u['prompt'] == p]),
                          tps=interval([u['tps_effect'] for u in units if u['prompt'] == p]),
                          arm_ms=statistics.fmean(u['arm_ms'] for u in units if u['prompt'] == p),
                          base_ms=statistics.fmean(u['base_ms'] for u in units if u['prompt'] == p)) for p in prompts}
    result = dict(arm=arm, base=base, units=units, per_prompt=per_prompt, merged=len(common) - len(units),
                  overall=interval([u['effect'] for u in units]), overall_tps=interval([u['tps_effect'] for u in units]),
                  prompt_means=interval([v['interval']['mean'] for v in per_prompt.values()]),
                  blocks=sorted(blocks, key=str) if blocks is not None else None)
    if any(u['cross_snapshot'] for u in units):
        by_pair = collections.defaultdict(list)
        for u in units:
            by_pair[u['snapshots']].append(u)
        result['snapshot_pairs'] = interval([statistics.fmean(u['effect'] for u in v) for v in by_pair.values()])
        result['snapshot_pairs_tps'] = interval([statistics.fmean(u['tps_effect'] for u in v) for v in by_pair.values()])
    return result


def select(rows: list[dict], spec: dict) -> dict:
    """A declared selection: over its units in the selection blocks, the candidate with the lowest mean ms a token, or
    with measure "tok/s" the highest mean tokens per second."""
    blocks = set(spec['select_blocks'])
    tps = spec.get('measure') == 'tok/s'
    means = {}
    for arm in spec['candidates']:
        units = units_of([r for r in rows if r['block'] in blocks], arm).values()
        values = [(1000 / statistics.fmean(r['rate'] for r in g)) if tps else statistics.fmean(r['rate'] for r in g) for g in units]
        if values:
            means[arm] = statistics.fmean(values)
    chosen = (max if tps else min)(means, key=means.get) if means else None
    test = spec.get('test_blocks')
    test = set(test) if test else {r['block'] for r in rows} - blocks
    return dict(name=spec['name'], chosen=chosen, measure='tok/s' if tps else 'ms', selection_means=means,
                select_blocks=sorted(blocks, key=str), test_blocks=sorted(test, key=str))


def arm_summary(rows: list[dict], arm: str) -> dict:
    g = [r for r in rows if r['arm'] == arm and r['ok'] and r['rate'] is not None]
    if not g:
        return dict(arm=arm, n=0)
    by_prompt = collections.defaultdict(list)
    for r in g:
        by_prompt[r['prompt']].append(r['rate'])
    offered = sum(r['offered'] for r in g)
    rate = statistics.fmean(statistics.fmean(v) for v in by_prompt.values())
    return dict(arm=arm, n=len(g), ms=rate, tok_s=1000 / rate, by_prompt={p: statistics.fmean(v) for p, v in by_prompt.items()},
                loads=statistics.fmean(r['loads'] for r in g if r['loads'] is not None),
                rounds_per_token=statistics.fmean(r['rounds_per_token'] for r in g if r['rounds_per_token'] is not None),
                accepted=sum(r['accepted'] for r in g) / offered if offered else None,
                offered_per_round=(offered / sum(float(r['final'].get('spec_rounds') or 0) for r in g)) if offered else None)


# ---------------------------------------------------------------- per pass, from light captures

def pass_rows(cap, measure_from: int = MEASURE_FROM) -> list[dict]:
    """Each MoE pass of the generation forwards at generated-token index measure_from or later. Times in ms from the
    pass's own join, its common origin."""
    gf = cap.generation_forwards()
    if not gf:
        return []
    first = gf[0]['position']
    keep = {f['forward']: f for f in gf if f['position'] - first >= measure_from}
    layers = collections.defaultdict(list)
    for r in cap.of('layer'):
        if r['forward'] in keep:
            layers[r['forward']].append(r)
    third = {(t['forward'], t['layer']): t for t in cap.of('third_pass') if t['forward'] in keep}
    rows = []
    for fw, f in keep.items():
        group = sorted(layers[fw], key=lambda r: r['layer'])
        for i, r in enumerate(group):
            join = r['t_ns'] + r['join_end']
            ends = {}
            if r['cpu_end']:
                ends['cpu'] = (r['t_ns'] + r['cpu_end'] - join) / 1e6
            if r['helper_worker_published']:
                ends['3090'] = (r['t_ns'] + r['helper_worker_published'] - join) / 1e6
            t = third.get((fw, r['layer']))
            if t and t['worker_published']:
                ends['4080'] = (t['t_ns'] + t['worker_published'] - join) / 1e6
            if r.get('gpu_expert_end'):   # (format 2: the 5090 finished its own expert compute after this pass's uploads; 0 / format 1: no stamp)
                ends['5090'] = (r['t_ns'] + r['gpu_expert_end'] - join) / 1e6
            nxt = group[i + 1] if i + 1 < len(group) else None
            cycle = ((nxt['t_ns'] + nxt['join_end']) if nxt else (f['t_ns'] + f['sync_end'])) - join
            rows.append(dict(forward=fw, layer=r['layer'], rows=r['rows'], token=f['position'] - first, ends=ends,
                             cycle=cycle / 1e6, next_wait=((nxt['join_end'] - nxt['pre_end']) / 1e6) if nxt else None,
                             last_layer=nxt is None))
    return rows


def forward_rows(cap, measure_from: int = MEASURE_FROM) -> list[dict]:
    gf = cap.generation_forwards()
    if not gf:
        return []
    first = gf[0]['position']
    return [dict(rows=f['rows'], ms=f['end'] / 1e6) for f in gf if f['position'] - first >= measure_from]


def balance(rows: list[dict]) -> dict:
    """The per-pass measures over a set of passes: mean finish time (join to next join), mean abs(difference) between
    each pair of recorded lanes present in a pass, the slowest against the second-slowest recorded lane, which lane
    was slowest, and the host's mean wait at the next join."""
    # (finish time over passes that end at a next join: a forward's last MoE layer ends at its sync instead)
    cycles = [r['cycle'] for r in rows if not r['last_layer']]
    out = dict(passes=len(rows), cycle_ms=statistics.fmean(cycles) if cycles else None)
    for a, b in PAIRS:
        v = [abs(r['ends'][a] - r['ends'][b]) for r in rows if a in r['ends'] and b in r['ends']]
        out[f'{a}|{b}'] = dict(n=len(v), mean=statistics.fmean(v) if v else None)
    gaps, slowest = [], collections.Counter()
    for r in rows:
        if len(r['ends']) >= 2:
            ordered = sorted(r['ends'].items(), key=lambda kv: -kv[1])
            gaps.append(ordered[0][1] - ordered[1][1])
            slowest[ordered[0][0]] += 1
    out['slowest|second'] = dict(n=len(gaps), mean=statistics.fmean(gaps) if gaps else None,
                                 slowest={k: v / len(gaps) for k, v in slowest.items()} if gaps else {})
    waits = [r['next_wait'] for r in rows if r['next_wait'] is not None]
    out['next_join_wait_ms'] = statistics.fmean(waits) if waits else None
    return out


def per_pass_report(rows: list[dict]) -> dict:
    by_layer = collections.defaultdict(list)
    by_rows = collections.defaultdict(list)
    for r in rows:
        by_layer[r['layer']].append(r)
        by_rows[r['rows']].append(r)
    layers = {k: balance(v) for k, v in sorted(by_layer.items())}
    spread = {}
    for key in ('cycle_ms', *[f'{a}|{b}' for a, b in PAIRS], 'slowest|second'):
        values = [(v[key] if key == 'cycle_ms' else v[key]['mean']) for v in layers.values()]
        values = [x for x in values if x is not None]
        spread[key] = dict(min=min(values), median=statistics.median(values), max=max(values)) if values else None
    return dict(overall=balance(rows), by_layer=layers, layer_spread=spread,
                by_rows={k: dict(passes=len(v), cycle_ms=statistics.fmean([x['cycle'] for x in v if not x['last_layer']] or [math.nan]))
                         for k, v in sorted(by_rows.items())})


def matched_rows(ra: list[dict], rb: list[dict], key: str, minimum: int = 20) -> dict | None:
    """A per-pass measure of two arms at matched pass shapes: per rows bin, weighted alike (by both arms' passes) over
    the bins both arms have at least `minimum` passes in."""
    def value(rows, k):
        if k == 'cycle_ms':
            return [r['cycle'] for r in rows if not r['last_layer']]
        if k == 'slowest|second':
            return [sorted(r['ends'].values(), reverse=True)[0] - sorted(r['ends'].values(), reverse=True)[1] for r in rows if len(r['ends']) >= 2]
        a, b = k.split('|')
        return [abs(r['ends'][a] - r['ends'][b]) for r in rows if a in r['ends'] and b in r['ends']]
    bins = sorted({r['rows'] for r in ra} & {r['rows'] for r in rb})
    used, wa, wb, total = [], 0.0, 0.0, 0
    for w in bins:
        va, vb = value([r for r in ra if r['rows'] == w], key), value([r for r in rb if r['rows'] == w], key)
        if len(va) >= minimum and len(vb) >= minimum:
            weight = len(va) + len(vb)
            wa += weight * statistics.fmean(va)
            wb += weight * statistics.fmean(vb)
            total += weight
            used.append(w)
    if not total:
        return None
    return dict(arm=wa / total, base=wb / total, effect=(wa / wb - 1) if wb else None, rows_bins=used)


def light_data(run: Run, measure_from: int) -> dict:
    """arm -> {(prompt, seed): [pass rows]} and arm -> [forward rows], from every validated light capture the study lists."""
    passes, forwards, notes = collections.defaultdict(lambda: collections.defaultdict(list)), collections.defaultdict(list), []
    for r in run.requests:
        if r['capture'] not in ('light', 'timestamps') or not r['artifact'] or not r['ran']:
            continue
        if r['validation'].get('status') != 'validated':
            notes.append(f"{r['step']}: light capture not validated ({r['validation'].get('errors')}); left out")
            continue
        path = pathlib.Path(r['artifact'])
        if not path.exists():
            notes.append(f"{r['step']}: capture file missing ({path}); left out")
            continue
        if r['why'] and any('expected' in w for w in r['why']):
            notes.append(f"{r['step']}: ran other settings ({r['why']}); left out")
            continue
        cap = light.read(path)
        passes[r['arm']][(r['prompt'], r['seed'])] += pass_rows(cap, measure_from)
        forwards[r['arm']] += forward_rows(cap, measure_from)
    return dict(passes=passes, forwards=forwards, notes=notes)


def per_pass_compare(data: dict, arm: str, base: str) -> dict | None:
    pa, pb = data['passes'].get(arm), data['passes'].get(base)
    if not pa or not pb:
        return None
    all_a = [x for v in pa.values() for x in v]
    all_b = [x for v in pb.values() for x in v]
    out = dict(arm=arm, base=base, matched={k: matched_rows(all_a, all_b, k) for k in ('cycle_ms', *[f'{a}|{b}' for a, b in PAIRS], 'slowest|second')})
    effects = []
    for unit in sorted(set(pa) & set(pb), key=str):
        m = matched_rows(pa[unit], pb[unit], 'cycle_ms', minimum=10)
        if m and m['effect'] is not None:
            effects.append(m['effect'])
    out['cycle_by_unit'] = interval(effects)
    la, lb = per_pass_report(all_a)['by_layer'], per_pass_report(all_b)['by_layer']
    both = [k for k in la if k in lb and la[k]['cycle_ms'] is not None and lb[k]['cycle_ms'] is not None]
    out['layers_faster'] = sum(1 for k in both if la[k]['cycle_ms'] < lb[k]['cycle_ms'])
    out['layers'] = len(both)
    return out


# ---------------------------------------------------------------- one run

def declared(run: Run, cli_primary: str | None, cli_compare: list[str]) -> tuple[list[dict], list[dict], list[str]]:
    notes = []
    given = run.study.get('primary')
    given = given if isinstance(given, list) else [given] if given else []
    primary = [dict(arm=p['arm'], base=p['baseline'], label=p.get('label', ''), named='in the protocol, before the run') for p in given]
    if primary:
        if cli_primary and cli_primary not in [f"{p['arm']}:{p['base']}" for p in primary]:
            notes.append(f"REFUSED --primary {cli_primary}: the protocol declares {'; '.join(p['arm'] + ' over ' + p['base'] for p in primary)}")
    elif cli_primary:
        a, b = cli_primary.split(':')
        primary = [dict(arm=a, base=b, label='', named='after the run, on the command line (the protocol declares none)')]
    secondary = [dict(arm=c['arm'], base=c['baseline'], label=c.get('label', ''), named='in the protocol')
                 for c in run.study.get('comparisons', [])]
    for c in cli_compare:
        a, b = c.split(':')
        secondary.append(dict(arm=a, base=b, label='', named='after the run, on the command line'))
    return primary, secondary, notes


def evaluate_run(path: pathlib.Path, cli_primary: str | None = None, cli_compare: list[str] | None = None,
                 include_captured: bool = False, measure_from: int = MEASURE_FROM, per_pass: bool = True) -> dict:
    run = Run(path, include_captured)
    timed = run.timed()
    out = dict(run=str(run.path), protocol_seed=run.protocol.get('seed'), notes=list(run.notes))
    seeds = sorted({r['seed'] for r in timed if r['ran']}, key=str)
    out['seeds'] = seeds
    if len(seeds) == 1 and len({r['block'] for r in timed}) > 1:
        out['notes'].append(f'every block used one seed ({seeds[0]}): blocks of an arm and prompt are one unit, so n is the number of prompts')
    out['excluded'] = [dict(step=r['step'], arm=r['arm'], why=r['why']) for r in timed if not r['ok']]
    out['texts'] = text_report(timed)
    arms = list(dict.fromkeys(r['arm'] for r in timed))
    out['arms'] = {a: arm_summary(timed, a) for a in arms}
    primary, secondary, notes = declared(run, cli_primary, cli_compare or [])
    out['notes'] += notes
    selections = {s['name']: select(timed, s) for s in run.study.get('selection', [])}
    out['selections'] = selections

    def resolve(c):
        a, b, blocks = c['arm'], c['base'], None
        for side in ('arm', 'base'):
            name = c[side]
            if name in selections:
                s = selections[name]
                if s['chosen'] is None:
                    return None
                blocks = set(s['test_blocks'])
                if side == 'arm':
                    a = s['chosen']
                else:
                    b = s['chosen']
        r = compare(timed, a, b, blocks)
        r.update(label=c.get('label', ''), named=c['named'], declared_as=f"{c['arm']} over {c['base']}")
        return r
    out['decision'] = run.study.get('decision')
    out['primaries'] = [x for x in (resolve(p) for p in primary) if x]
    out['primary'] = out['primaries'][0] if out['primaries'] else None
    if not primary:
        out['notes'].append('no primary comparison is declared: the figures below are descriptive, and no comparison is the result')
    aa = run.study.get('aa')
    if aa:
        ta, tb = run.arms.get(aa[0]), run.arms.get(aa[1])
        if ta is not None and tb is not None and ta != tb:
            out['notes'].append(f'the A/A arms {aa[0]} and {aa[1]} do not have identical settings: {ta} against {tb}')
        out['aa'] = compare(timed, aa[0], aa[1])
        out['aa']['holds_zero'] = holds_zero(out['aa']['overall'])
    else:
        out['aa'] = None
        out['notes'].append('no A/A pair is declared (study.aa): the run has no floor of its own')
    out['secondary'] = [x for x in (resolve(c) for c in secondary) if x]
    if per_pass:
        data = light_data(run, measure_from)
        out['notes'] += data['notes']
        pp = dict(arms={}, comparisons=[], forwards={})
        for arm, units in data['passes'].items():
            rows = [x for v in units.values() for x in v]
            if rows:
                pp['arms'][arm] = per_pass_report(rows)
                pp['arms'][arm]['units'] = len(units)
        for arm, fw in data['forwards'].items():
            by = collections.defaultdict(list)
            for f in fw:
                by[f['rows']].append(f['ms'])
            pp['forwards'][arm] = {k: dict(n=len(v), mean_ms=statistics.fmean(v), median_ms=statistics.median(v)) for k, v in sorted(by.items())}
        for c in primary + secondary + ([dict(arm=aa[0], base=aa[1])] if aa else []):
            if c['arm'] in selections or c['base'] in selections:
                continue
            got = per_pass_compare(data, c['arm'], c['base'])
            if got:
                pp['comparisons'].append(got)
        out['per_pass'] = pp
    return out


# ---------------------------------------------------------------- across engine starts

def evaluate_across(starts: list[tuple[str, pathlib.Path]], arm: str, anchor: str, measure_from: int = MEASURE_FROM) -> dict:
    """B over A across engine starts. Refused unless the starts alternate and each has the anchor arm."""
    runs = [(label, Run(p)) for label, p in starts]
    labels = sorted({label for label, _ in runs})
    refusals = []
    if len(labels) != 2:
        refusals.append(f'two configurations are compared; got {labels}')
    runs.sort(key=lambda lr: lr[1].started())
    order = [label for label, _ in runs]
    if any(order.count(x) < 2 for x in labels):
        refusals.append(f'each configuration needs at least two engine starts; got {order}')
    if any(a == b for a, b in zip(order, order[1:])):
        refusals.append(f'the engine starts do not alternate (in time order {order})')
    for label in labels:   # (two starts of one configuration run one binary)
        shas = {sha for lab, r in runs if lab == label for sha in r.executables()}
        if len(shas) > 1:
            refusals.append(f'the starts labelled {label} ran different binaries ({sorted(s[:12] for s in shas)})')
    pids = [r.engine_pids() for _, r in runs]
    for i in range(len(pids)):
        for j in range(i + 1, len(pids)):
            if pids[i] & pids[j]:
                refusals.append(f'{runs[i][1].path.name} and {runs[j][1].path.name} ran on one engine start (pid {sorted(pids[i] & pids[j])})')
    units = []
    for label, r in runs:
        ua, uk = units_of(r.timed(), arm), units_of(r.timed(), anchor)
        if not uk:
            refusals.append(f'{r.path.name} has no valid requests of the anchor arm {anchor}')
        if not ua:
            refusals.append(f'{r.path.name} has no valid requests of the arm {arm}')
        units.append((set(ua), set(uk)))
    if refusals:
        return dict(refused=refusals, order=order)
    common = set.intersection(*[a & k for a, k in units])
    if not common:
        return dict(refused=['no (prompt, seed) unit is common to every start'], order=order)
    table = []
    for label, r in runs:
        ua, uk = units_of(r.timed(), arm), units_of(r.timed(), anchor)
        per = {u: dict(arm=statistics.fmean(x['rate'] for x in ua[u]), anchor=statistics.fmean(x['rate'] for x in uk[u])) for u in common}
        table.append(dict(label=label, run=str(r.path), started=r.started(), units=per,
                          arm_ms=statistics.fmean(v['arm'] for v in per.values()), anchor_ms=statistics.fmean(v['anchor'] for v in per.values())))
    spread = {}
    for label in labels:
        same = [t for t in table if t['label'] == label]
        for what in ('arm', 'anchor'):
            effects = [statistics.fmean(s2['units'][u][what] / s1['units'][u][what] - 1 for u in common)
                       for s1, s2 in zip(same, same[1:])]
            spread[f'{label} {what}'] = effects
    a_lab, b_lab = labels
    pairs = []
    for i in range(0, len(table) - 1, 2):
        s1, s2 = table[i], table[i + 1]
        sa, sb = (s1, s2) if s1['label'] == a_lab else (s2, s1)
        pairs.append(dict(arm=statistics.fmean(math.log(sb['units'][u]['arm'] / sa['units'][u]['arm']) for u in common),
                          normalized=statistics.fmean(math.log((sb['units'][u]['arm'] / sb['units'][u]['anchor']) /
                                                               (sa['units'][u]['arm'] / sa['units'][u]['anchor'])) for u in common)))
    per_unit = {}
    for u in sorted(common, key=str):
        ma = statistics.fmean(t['units'][u]['arm'] for t in table if t['label'] == a_lab)
        mb = statistics.fmean(t['units'][u]['arm'] for t in table if t['label'] == b_lab)
        per_unit[str(u)] = mb / ma - 1
    est = interval([math.exp(p['arm']) - 1 for p in pairs])
    est_n = interval([math.exp(p['normalized']) - 1 for p in pairs])
    largest_spread = max((abs(x) for v in spread.values() for x in v), default=None)
    out = dict(order=order, configurations=labels, arm=arm, anchor=anchor, starts=[{k: v for k, v in t.items() if k != 'units'} for t in table],
               units=len(common), spread=spread, b_over_a_by_unit=per_unit, b_over_a=est, b_over_a_anchor_normalized=est_n,
               within_spread=None if est['mean'] is None or largest_spread is None else abs(est['mean']) <= largest_spread,
               largest_same_configuration_spread=largest_spread)
    pp = []
    for label, r in runs:
        data = light_data(r, measure_from)
        rows = [x for v in data['passes'].get(arm, {}).values() for x in v]
        if rows:
            pp.append(dict(label=label, run=r.path.name, per_pass=per_pass_report(rows)['overall']))
    out['per_pass'] = pp
    return out


# ---------------------------------------------------------------- report

def render_compare(c: dict, lines: list[str], title: str, decision: str | None = None) -> None:
    lines.append(f"{title}: {c['arm']} over {c['base']}" + (f" ({c['label']})" if c.get('label') else '') +
                 f"; named {c['named']}" + (f"; blocks {c['blocks']}" if c.get('blocks') else ''))
    lines.append(f"  ms a token from token 128 (measured); effect = {c['arm']} / {c['base']} - 1 per unit; tokens per second = "
                 f"{c['base']} / {c['arm']} - 1 per unit (above zero: {c['arm']} faster); 95% t intervals")
    lines.append(f"  {'prompt':<14} {'units':>5} {c['arm'][:16]:>16} {c['base'][:16]:>16}  {'ms a token':<32} tokens per second")
    for p, v in c['per_prompt'].items():
        lines.append(f"  {str(p):<14} {v['interval']['n']:>5} {v['arm_ms']:>16.2f} {v['base_ms']:>16.2f}  {ci_text(v['interval']):<32} {ci_text(v['tps'])}")
    lines.append(f"  all units, ms a token: {ci_text(c['overall'])}" + (f"; {c['merged']} units merged for repeated text" if c['merged'] else ''))
    tps = f"  all units, tokens per second: {ci_text(c['overall_tps'])}"
    lines.append(tps + ('   <- the decision figure (study.decision: tokens per second)' if decision and 'second' in decision else ''))
    lines.append(f"  mean of the prompt means (n = prompts), ms a token: {ci_text(c['prompt_means'])}")
    if c.get('snapshot_pairs'):
        lines.append(f"  the arms restored different snapshots: over snapshot pairs (the interval to read) ms a token "
                     f"{ci_text(c['snapshot_pairs'])}, tokens per second {ci_text(c['snapshot_pairs_tps'])}")
    short = sorted({s for u in c['units'] for s in u['short_window']})
    if short:
        lines.append(f"  answers that stopped early (shorter window): {', '.join(short)}")


def render_run(res: dict, by_layer: bool = False) -> str:
    lines = [f"Run {res['run']}", f"  seeds of the timed requests: {res['seeds']}"]
    for n in res['notes']:
        lines.append(f'  NOTE: {n}')
    if res['excluded']:
        lines.append(f"  left out ({len(res['excluded'])} requests):")
        for e in res['excluded']:
            lines.append(f"    {e['step']} ({e['arm']}): {'; '.join(e['why'])}")
    lines.append('')
    lines.append('Arms, ms a token from token 128 (measured; mean of the prompt means)')
    for a, s in res['arms'].items():
        if not s.get('n'):
            lines.append(f'  {a}: no valid request')
            continue
        extra = (f"; drafts accepted {100 * s['accepted']:.1f}%, offered a drafting round {s['offered_per_round']:.2f}" if s.get('accepted') is not None else '')
        lines.append(f"  {a:<18} {s['ms']:6.2f} ms ({s['tok_s']:5.1f} tok/s), n {s['n']}; demand loads a token {s['loads']:.2f}; "
                     f"target rounds a token {s['rounds_per_token']:.3f}{extra}")
        lines.append('      by prompt: ' + '; '.join(f'{p} {v:.2f}' for p, v in s['by_prompt'].items()))
    lines.append('')
    lines.append('Texts (measured: SHA-256 of reasoning and answer)')
    one = [t for t in res['texts'] if t['one_text']]
    lines.append(f"  arm-and-prompt groups: {len(res['texts'])}; groups whose every request wrote one text: {len(one)}" +
                 (' (' + ', '.join(f"{t['arm']}/{t['prompt']}" for t in one) + ')' if one else ''))
    rep = [t for t in res['texts'] if t['repeated_across_seeds']]
    if rep:
        lines.append('  texts repeated across different seeds: ' + ', '.join(f"{t['arm']}/{t['prompt']} ({t['repeated_across_seeds']})" for t in rep))
    for name, s in res.get('selections', {}).items():
        tps = s.get('measure') == 'tok/s'
        lines.append('')
        lines.append(f"Selection {name} (declared): on blocks {s['select_blocks']} the {'highest mean tokens per second' if tps else 'lowest mean ms a token'} "
                     f"is {s['chosen']} ({', '.join(f'{a} {m:.2f}' for a, m in sorted(s['selection_means'].items(), key=lambda kv: -kv[1] if tps else kv[1]))}); "
                     f"compared on blocks {s['test_blocks']} only")
    lines.append('')
    if res.get('decision'):
        lines.append(f"Decision rule (the protocol's): {res['decision']}")
    for i, c in enumerate(res.get('primaries') or []):
        if i:
            lines.append('')
        render_compare(c, lines, 'PRIMARY COMPARISON' + (f' {i + 1} of {len(res["primaries"])}' if len(res['primaries']) > 1 else ''), res.get('decision'))
    if not res.get('primaries'):
        lines.append('No primary comparison declared.')
    if res.get('aa'):
        lines.append('')
        render_compare(dict(res['aa'], named='in the protocol', label='identical settings'), lines, 'A/A CHECK')
        hz = res['aa']['holds_zero']
        lines.append('  the interval ' + ('holds zero' if hz else 'does not hold zero: the run is noisier than its own intervals say' if hz is False else 'cannot be computed'))
    for c in res['secondary']:
        lines.append('')
        render_compare(c, lines, 'Secondary comparison')
    pp = res.get('per_pass')
    if pp and pp['arms']:
        lines.append('')
        lines.append(f'Per pass, from light captures, passes at generated-token index {MEASURE_FROM} and later (measured)')
        lines.append("  Lanes recorded: CPU (its jobs' end), 3090 and 4080 (each worker's publication), 5090 (copy-plus-primary: "
                     "the end of its own expert compute, gpu_expert_end, which format 2 captures carry; format 1 captures have "
                     "no 5090 stamps, and then the 5090 pairs show no passes).")
        lines.append('  ms; abs(difference) of completion times inside each pass, then the mean; layer range = min / median / max of the per-layer means')
        for arm, rep_ in pp['arms'].items():
            o = rep_['overall']
            lines.append(f"  {arm}: {o['passes']} passes from {rep_['units']} requests; finish time a pass {o['cycle_ms']:.3f} ms; "
                         f"host wait at the next join {o['next_join_wait_ms'] if o['next_join_wait_ms'] is None else round(o['next_join_wait_ms'], 3)} ms")
            for key in (*[f'{a}|{b}' for a, b in PAIRS], 'slowest|second'):
                m, sp = o[key], rep_['layer_spread'].get(key)
                if m['n']:
                    lines.append(f"      {key:<15} {m['mean']:.3f} ms over {m['n']} passes; layer range {sp['min']:.3f} / {sp['median']:.3f} / {sp['max']:.3f}" +
                                 (('; slowest: ' + ', '.join(f'{k} {100 * v:.0f}%' for k, v in sorted(m['slowest'].items()))) if key == 'slowest|second' else ''))
            lines.append('      finish time a pass by rows: ' + ', '.join(f"{w}: {v['cycle_ms']:.3f} ({v['passes']})" for w, v in rep_['by_rows'].items()))
        if by_layer:
            for arm, rep_ in pp['arms'].items():
                lines.append(f'  {arm} by layer (ms; passes, finish time a pass, then mean abs(difference) per pair; - where a pair never met):')
                pair_keys = [f'{a}|{b}' for a, b in PAIRS]
                lines.append(f"    {'layer':>5} {'passes':>7} {'finish':>7} " + ' '.join(f'{k:>9}' for k in pair_keys) + f" {'slow|2nd':>9}")
                fmt = lambda m: f"{m['mean']:9.3f}" if m['n'] else f"{'-':>9}"
                for layer, v in rep_['by_layer'].items():
                    lines.append(f"    {layer:>5} {v['passes']:>7} {(v['cycle_ms'] if v['cycle_ms'] is not None else float('nan')):>7.3f} "
                                 + ' '.join(fmt(v[k]) for k in pair_keys) + f" {fmt(v['slowest|second'])}")
        for arm, fw in pp['forwards'].items():
            lines.append(f"  {arm}: forward time by rows, ms mean (median, forwards): " + ', '.join(f"{w}: {v['mean_ms']:.1f} ({v['median_ms']:.1f}, {v['n']})" for w, v in fw.items()))
        for c in pp['comparisons']:
            parts = []
            for key, m in c['matched'].items():
                if m:
                    parts.append(f"{key} {m['arm']:.3f} against {m['base']:.3f}")
            lines.append(f"  {c['arm']} against {c['base']}, at matched rows: " + '; '.join(parts) +
                         f"; finish time a pass by unit {ci_text(c['cycle_by_unit'])}; layers where {c['arm']} is faster {c['layers_faster']} of {c['layers']}")
    return '\n'.join(lines) + '\n'


def render_across(res: dict) -> str:
    if res.get('refused'):
        return 'REFUSED: cross-run comparison\n' + '\n'.join(f'  {x}' for x in res['refused']) + '\n'
    lines = [f"Across engine starts (time order {res['order']}): {res['arm']}, anchor {res['anchor']}, {res['units']} units common to every start"]
    for s in res['starts']:
        lines.append(f"  {s['label']} {pathlib.Path(s['run']).name}: arm {s['arm_ms']:.2f} ms, anchor {s['anchor_ms']:.2f} ms (measured)")
    for k, v in res['spread'].items():
        lines.append(f"  start-to-start, {k}: " + ', '.join(pct(x) for x in v))
    lines.append(f"  {res['configurations'][1]} over {res['configurations'][0]}, over start pairs: {ci_text(res['b_over_a'])}; "
                 f"normalized by the anchor {ci_text(res['b_over_a_anchor_normalized'])}")
    lines.append(f"  largest same-configuration spread {pct(res['largest_same_configuration_spread'])}: the difference is "
                 + ('within it' if res['within_spread'] else 'outside it'))
    for p in res.get('per_pass', []):
        lines.append(f"  per pass {p['label']} {p['run']}: finish time a pass {p['per_pass']['cycle_ms']:.3f} ms")
    return '\n'.join(lines) + '\n'


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', nargs='?', type=pathlib.Path)
    ap.add_argument('--primary', help='ARM:BASELINE, only for a protocol that declares none')
    ap.add_argument('--compare', action='append', default=[], help='ARM:BASELINE, a further comparison (named after the run)')
    ap.add_argument('--include-captured', action='store_true', help='count light-captured study requests as timed (they read the prompt another way)')
    ap.add_argument('--measure-from', type=int, default=MEASURE_FROM)
    ap.add_argument('--no-per-pass', action='store_true')
    ap.add_argument('--by-layer', action='store_true', help='print the per-pass measures of every layer (the JSON always has them)')
    ap.add_argument('--across', nargs='+', metavar='LABEL=RUN_DIR')
    ap.add_argument('--arm')
    ap.add_argument('--anchor', default='one-row')
    ap.add_argument('--text', type=pathlib.Path)
    ap.add_argument('--json', type=pathlib.Path)
    a = ap.parse_args()
    if a.across:
        if not a.arm:
            ap.error('--across needs --arm')
        res = evaluate_across([(x.split('=', 1)[0], pathlib.Path(x.split('=', 1)[1])) for x in a.across], a.arm, a.anchor, a.measure_from)
        text = render_across(res)
    else:
        if not a.run:
            ap.error('a run directory, or --across')
        res = evaluate_run(a.run, a.primary, a.compare, a.include_captured, a.measure_from, not a.no_per_pass)
        text = render_run(res, a.by_layer)
    print(text, end='')
    if a.text:
        a.text.write_text(text)
    if a.json:
        a.json.write_text(json.dumps(res, indent=1, default=str) + '\n')
    return 2 if res.get('refused') else 0


if __name__ == '__main__':
    raise SystemExit(main())
