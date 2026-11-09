#!/usr/bin/env python3
"""Write a planner v1 table for maya-mm3's --spec-plan (step 6) from timeline captures of drafting requests.

  spec_plan.py --run <run dir> [--run ...] --drafter dflash2 --out plan.txt [--draft-shift MS] [--verify-shift MS]
               [--accept-run DIR ...] [--timing-run DIR ... --timing-arms ARM,ARM] [--post-by tokens|evaluated]

--run supplies both the acceptance and the times. --accept-run adds runs for the acceptance only; --timing-run, when given,
supplies the times instead of --run (only the --timing-arms arms when named): planner v1.1 fits its times on the planner's
own regime. --post-by tokens indexes the acceptance phase by the tokens a round produces (one target distribution each:
post_ms[n] for n + 1 tokens, plan-v1-20261109's finding) instead of by evaluated drafts.

A run dir holds protocol.json (study.requests with step and arm; arms with spec_policy) and each request's
<step>.json.capture/*.jsonl, as the step-6 acceptance runs do. Every request's first eighth of rounds is dropped (its
warm-up, about token 128 at 1,024 tokens).

  acceptance  per draft position and confidence bin (10 bins over [0, 1]): the kept rate of the drafts verification
              reached in the exact arms (spec_policy sampled), smoothed with 2 pseudo-counts at the position's own rate.
              Only exact arms: a lenient arm keeps more, and the table must mean what exact verification does.
  draft_ms    the median draft call over every arm's drafted rounds
  verify_ms   the median verify forward by rows, every arm pooled, where 20 or more rounds have that row count; others
              are interpolated between the nearest such ones (or extended along the last two)
  post_ms     the median acceptance phase (the verify's end to the round's end) by evaluated drafts where 20 or more
              rounds have that count, else a least-squares line over all rounds

--draft-shift and --verify-shift add a measured kernel change to a table fitted on an older binary (the verify shift
applies to windows of 2 or more rows): for example the head pass's second form, -1.1 ms on both
(docs/minimax-wsl/evidence.md#head-kernel-second-form).
"""
from __future__ import annotations
import argparse, collections, json, pathlib, statistics

BINS = 10


def rounds_of(capdir: pathlib.Path):
    f = next(capdir.glob('*.jsonl'), None)
    if f is None:   # (a capture directory without a timeline: a capture-off or light request)
        return []
    rounds, cur = [], None
    for line in f.open():
        e = json.loads(line)
        t = e['type']
        if t == 'round_begin':
            cur = dict(props={}, decisions={})
        elif cur is None:
            continue
        elif t == 'draft_begin':
            cur['db'] = e['steady_ns']
        elif t == 'draft_end':
            cur['de'] = e['steady_ns']
        elif t == 'draft_proposal':
            cur['props'][int(e['index'])] = float(e['confidence'])
        elif t == 'target_forward_begin':
            cur['vb'] = e['steady_ns']
            cur['rows'] = int(e.get('rows') or 0)
        elif t == 'target_forward_end':
            cur['ve'] = e['steady_ns']
        elif t == 'draft_decision':
            cur['decisions'][int(e['index'])] = bool(e.get('keep'))
        elif t == 'round_end':
            if 'vb' in cur and 've' in cur:
                a = 0
                while a in cur['decisions'] and cur['decisions'][a]:
                    a += 1
                cur['accepted'] = a
                cur['evaluated'] = int(e.get('evaluated') or 0)
                cur['t_draft'] = (cur['de'] - cur['db']) / 1e6 if 'db' in cur and 'de' in cur else None
                cur['t_verify'] = (cur['ve'] - cur['vb']) / 1e6
                cur['t_post'] = (e['steady_ns'] - cur['ve']) / 1e6
                rounds.append(cur)
            cur = None
    return rounds


def load(runs, only_arms=None):
    """every captured request's rounds (the study's requests and mechanism blocks), each request's first eighth dropped"""
    out = []
    for run in runs:
        protocol = json.loads((run / 'protocol.json').read_text())
        arms = protocol.get('arms', {})
        study = protocol['study']
        for q in study.get('requests', []) + study.get('mechanism', []):
            if only_arms and q['arm'] not in only_arms:
                continue
            cap = run / f"{q['step']}.json.capture"
            if not cap.exists():
                continue
            rs = rounds_of(cap)
            rs = rs[len(rs) // 8:]
            exact = arms.get(q['arm'], {}).get('spec_policy') == 'sampled'
            for r in rs:
                r['exact'] = exact
                r['arm'] = q['arm']
            out.extend(rs)
    return out


def lsq(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx if sxx else 0.0
    return my - b * mx, b


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--run', action='append', default=[], type=pathlib.Path)
    ap.add_argument('--accept-run', action='append', default=[], type=pathlib.Path)
    ap.add_argument('--timing-run', action='append', default=[], type=pathlib.Path)
    ap.add_argument('--timing-arms', default='')
    ap.add_argument('--post-by', choices=('tokens', 'evaluated'), default='evaluated')
    ap.add_argument('--drafter', required=True)
    ap.add_argument('--out', required=True, type=pathlib.Path)
    ap.add_argument('--positions', type=int, default=7)
    ap.add_argument('--rows', type=int, default=8, help='verify_ms for 1..rows rows')
    ap.add_argument('--draft-shift', type=float, default=0.0)
    ap.add_argument('--verify-shift', type=float, default=0.0)
    ap.add_argument('--verify-shift-rows', type=int, default=2, help='the shift applies from this many rows (1: every forward)')
    a = ap.parse_args()
    data = load(a.run + a.accept_run)   # (the acceptance's rounds)
    timing = load(a.timing_run, set(a.timing_arms.split(',')) - {''}) if a.timing_run else load(a.run)
    if not data or not timing:
        raise SystemExit('no rounds')
    # acceptance: per position and bin over the exact arms' evaluated drafts
    pos = collections.defaultdict(lambda: [0, 0])
    cell = collections.defaultdict(lambda: [0, 0])
    for r in data:
        if not r['exact']:
            continue
        for i, keep in r['decisions'].items():
            c = r['props'].get(i)
            if c is None:
                continue
            b = min(BINS - 1, int(c * BINS))
            pos[i][0] += keep
            pos[i][1] += 1
            cell[(i, b)][0] += keep
            cell[(i, b)][1] += 1
    if not pos:
        raise SystemExit('no exact-arm decisions (spec_policy sampled) in these runs')
    accept, counts = [], []
    for i in range(a.positions):
        k, n = pos.get(i, (0, 0))
        p0 = (k + 1) / (n + 2)
        row, cnt = [], []
        for b in range(BINS):
            kb, nb = cell.get((i, b), (0, 0))
            row.append((kb + 2 * p0) / (nb + 2))
            cnt.append(nb)
        accept.append(row)
        counts.append(cnt)
    # times (from the timing rounds)
    draft = statistics.median(r['t_draft'] for r in timing if r['t_draft']) + a.draft_shift
    by_rows = collections.defaultdict(list)
    for r in timing:
        by_rows[r['rows']].append(r['t_verify'])
    measured = {w: statistics.median(v) for w, v in by_rows.items() if w >= 1 and len(v) >= 20}
    ks = sorted(measured)
    verify = []
    for w in range(1, a.rows + 1):
        if w in measured:
            v = measured[w]
        else:
            lo = max([k for k in ks if k < w], default=None)
            hi = min([k for k in ks if k > w], default=None)
            if lo is not None and hi is not None:
                v = measured[lo] + (measured[hi] - measured[lo]) * (w - lo) / (hi - lo)
            else:
                k1, k2 = (ks[-2], ks[-1]) if hi is None else (ks[0], ks[1])
                v = measured[k1] + (measured[k2] - measured[k1]) * (w - k1) / (k2 - k1)
        verify.append(v + (a.verify_shift if w >= a.verify_shift_rows else 0.0))
    # the acceptance phase by evaluated drafts, or (--post-by tokens) by produced tokens less one: the accepted drafts
    key = (lambda r: r['accepted']) if a.post_by == 'tokens' else (lambda r: r['evaluated'])
    by_eval = collections.defaultdict(list)
    for r in timing:
        by_eval[key(r)].append(r['t_post'])
    pa, pb = lsq([key(r) for r in timing], [r['t_post'] for r in timing])
    post = [statistics.median(by_eval[n]) if len(by_eval[n]) >= 20 else pa + pb * n for n in range(a.positions + 1)]
    sources = ', '.join(str(r) for r in a.run + a.accept_run)
    tsources = ', '.join(str(r) for r in a.timing_run) + (f' (arms {a.timing_arms})' if a.timing_arms else '') \
        if a.timing_run else 'the same'
    lines = ['maya-spec-plan 1',
             f'# acceptance from {sources}: {len(data)} rounds; times from {tsources}: {len(timing)} rounds; draft shift '
             f'{a.draft_shift:+g} ms, verify shift ({a.verify_shift_rows}+ rows) {a.verify_shift:+g} ms',
             f'drafter {a.drafter}', f'bins {BINS}']
    if a.post_by == 'tokens':
        lines.append('post_by tokens')
    for i, row in enumerate(accept):
        lines.append(f'# position {i}: evaluated drafts by bin ' + ' '.join(str(c) for c in counts[i]))
        lines.append(f'accept {i} ' + ' '.join(f'{x:.6f}' for x in row))
    lines.append(f'draft_ms {draft:.4f}')
    lines.append('# verify_ms: rows 1..; measured row counts ' + ' '.join(f'{w}:{len(by_rows[w])}' for w in ks))
    lines.append('verify_ms ' + ' '.join(f'{v:.4f}' for v in verify))
    lines.append(f'# post_ms: {"produced tokens 1.." if a.post_by == "tokens" else "evaluated drafts 0.."}; rounds by count '
                 + ' '.join(f'{n}:{len(by_eval[n])}' for n in range(a.positions + 1)))
    lines.append('post_ms ' + ' '.join(f'{v:.4f}' for v in post))
    a.out.write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
