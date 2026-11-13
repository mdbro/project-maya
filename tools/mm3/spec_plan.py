#!/usr/bin/env python3
"""Write a planner v1 table for maya-mm3's --spec-plan (step 6) from timeline captures of drafting requests.

  spec_plan.py --run <run dir> [--run ...] --drafter dflash2 --out plan.txt [--draft-shift MS] [--verify-shift MS]
               [--accept-run DIR ...] [--timing-run DIR ... --timing-arms ARM,ARM] [--post-by tokens|evaluated]
               [--lenience ELL [--accept-arms ARM,ARM]]

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
              With --lenience ELL (0 < ELL <= 1) each reached draft is scored instead by the probability the lenient rule
              keeps it, min(1, p / (ELL q)), from the target's p and the drafter's q recorded with the draft, so a table for
              any lenience strength can be fitted from captures of either rule (ELL 1 is the exact rule's probability).
              The drafts come from every arm whose drafts are sampled from q (spec_policy sampled or lenience; an arm's
              rule is its tune's, else its request's final statistics), or from the --accept-arms named. The table then
              means what that rule does; reach is still that of the captured rule, so drafts the laxer rule would have
              reached and the captured one did not are missing (capture with the rule the table is for where it matters).
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
            cur.setdefault('pq', {})[int(e['index'])] = (float(e.get('p') or 0.0), float(e.get('q') or 0.0))
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
    """every captured request's rounds (the study's requests, mechanism, replay and timeline blocks), each request's first
    eighth dropped; each round carries its arm and the arm's acceptance rule"""
    out = []
    for run in runs:
        protocol = json.loads((run / 'protocol.json').read_text())
        study = protocol['study']
        arms = protocol.get('arms') or study.get('arms') or {}
        ledger = run / 'ledger.json'
        finals = {s['step']: (s.get('summary') or {}).get('final_stat') or {} for s in json.loads(ledger.read_text())['steps']} \
            if ledger.exists() else {}
        for q in study.get('requests', []) + study.get('mechanism', []) + study.get('replay', []) + study.get('timeline', []):
            if only_arms and q['arm'] not in only_arms:
                continue
            cap = run / f"{q['step']}.json.capture"
            if not cap.exists():
                continue
            rs = rounds_of(cap)
            rs = rs[len(rs) // 8:]
            policy = arms.get(q['arm'], {}).get('spec_policy') or finals.get(q['step'], {}).get('spec_policy')
            for r in rs:
                r['exact'] = arms.get(q['arm'], {}).get('spec_policy') == 'sampled'
                r['policy'] = policy
                r['arm'] = q['arm']
            out.extend(rs)
    return out


def scored_decisions(data, lenience=None, accept_arms=None):
    """(position, confidence, score) of every reached draft the table is fitted on. Without lenience: the exact arms'
    verdicts (1 kept, 0 not), today's rule. With lenience ELL: every arm that samples its drafts from q (or the named
    arms), each draft scored min(1, p / (ELL q))."""
    out = []
    for r in data:
        if lenience is None:
            if not r['exact']:
                continue
        elif accept_arms:
            if r['arm'] not in accept_arms:
                continue
        elif r.get('policy') not in ('sampled', 'lenience'):
            continue
        for i, keep in r['decisions'].items():
            c = r['props'].get(i)
            if c is None:
                continue
            if lenience is None:
                out.append((i, c, 1.0 if keep else 0.0))
                continue
            p, qq = r.get('pq', {}).get(i, (0.0, 0.0))
            if qq > 0.0:
                out.append((i, c, min(1.0, p / (lenience * qq))))
    return out


def fit_accept_logit(decisions, positions, l2=1.0):
    """planner v2's acceptance: logistic regression of a reached draft's verdict (or, with --lenience, its keep
    probability) on its position (one-hot) and logit(confidence), standardized, L2 1, by IRLS
    (accept-data-20261109/accept_model.py's M2); returned de-standardized as a per-position intercept and one slope - the
    plan file's accept_logit line"""
    import math
    import numpy as np
    lg = lambda c: math.log(min(1 - 1e-6, max(1e-6, c)) / (1 - min(1 - 1e-6, max(1e-6, c))))
    rows = [(i, c, score) for i, c, score in decisions if i < positions]
    X = np.array([[1.0 if i == j else 0.0 for j in range(positions)] + [lg(c)] for i, c, _ in rows])
    y = np.array([score for _, _, score in rows])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(Z @ w)))
        step = np.linalg.solve((Z * (p * (1 - p))[:, None]).T @ Z + l2 * np.eye(Z.shape[1]), Z.T @ (p - y) + l2 * w)
        w -= step
        if np.abs(step).max() < 1e-10:
            break
    beta = w[1:] / sd
    base = w[0] - float((w[1:] * mu / sd).sum())
    intercepts = [base + beta[j] for j in range(positions)]
    return [f'# accept_logit: logistic on {len(rows)} reached drafts (position, logit confidence); a_0..a_{positions - 1} slope',
            'accept_logit ' + ' '.join(f'{x:.6f}' for x in intercepts) + f' {beta[positions]:.6f}']


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
    ap.add_argument('--adapt', type=float, default=0.0, help='planner v1.1: each request\'s logit offset on the table, '
                    'this step a verdict (0: off)')
    ap.add_argument('--accept', choices=('table', 'logistic'), default='table',
                    help='logistic (planner v2): sigmoid(a_pos + slope * logit(confidence)), fitted by IRLS on the exact '
                         'arms\' reached drafts (accept-data-20261109\'s M2); the table is written beside it')
    ap.add_argument('--lenience', type=float, default=None, metavar='ELL',
                    help='fit the acceptance for the lenient rule of strength ELL (0 < ELL <= 1): each reached draft scored '
                         'min(1, p / (ELL q)) from its recorded p and q, from every arm that samples its drafts (default: '
                         'the exact arms\' kept rate, as before)')
    ap.add_argument('--accept-arms', default='', help='with --lenience: the arms whose drafts the acceptance is fitted on')
    ap.add_argument('--drafter', required=True)
    ap.add_argument('--out', required=True, type=pathlib.Path)
    ap.add_argument('--positions', type=int, default=7)
    ap.add_argument('--rows', type=int, default=8, help='verify_ms for 1..rows rows')
    ap.add_argument('--draft-shift', type=float, default=0.0)
    ap.add_argument('--verify-shift', type=float, default=0.0)
    ap.add_argument('--verify-shift-rows', type=int, default=2, help='the shift applies from this many rows (1: every forward)')
    a = ap.parse_args()
    if a.lenience is not None and not 0.0 < a.lenience <= 1.0:
        ap.error('--lenience: 0 < ELL <= 1')
    if a.accept_arms and a.lenience is None:
        ap.error('--accept-arms goes with --lenience')
    data = load(a.run + a.accept_run)   # (the acceptance's rounds)
    timing = load(a.timing_run, set(a.timing_arms.split(',')) - {''}) if a.timing_run else load(a.run)
    if not data or not timing:
        raise SystemExit('no rounds')
    # acceptance: per position and bin over the exact arms' evaluated drafts (or, with --lenience, the scored drafts)
    decisions = scored_decisions(data, a.lenience, set(a.accept_arms.split(',')) - {''})
    pos = collections.defaultdict(lambda: [0, 0])
    cell = collections.defaultdict(lambda: [0, 0])
    for i, c, score in decisions:
        b = min(BINS - 1, int(c * BINS))
        pos[i][0] += score
        pos[i][1] += 1
        cell[(i, b)][0] += score
        cell[(i, b)][1] += 1
    if not pos:
        raise SystemExit('no exact-arm decisions (spec_policy sampled) in these runs' if a.lenience is None else
                         'no reached drafts with a recorded p and q in the arms that sample their drafts')
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
    if a.lenience is not None:
        arms = sorted({r['arm'] for r in data if (r['arm'] in set(a.accept_arms.split(',')) if a.accept_arms
                                                    else r.get('policy') in ('sampled', 'lenience'))})
        lines.insert(2, f'# acceptance rule: lenience {a.lenience:g}, each reached draft scored min(1, p / ({a.lenience:g} q)) from '
                        f'its recorded p and q; {len(decisions)} drafts from the arms {", ".join(arms)}')
    if a.post_by == 'tokens':
        lines.append('post_by tokens')
    if a.adapt > 0:
        lines.append(f'adapt {a.adapt:g}')
    if a.accept == 'logistic':
        lines += fit_accept_logit(decisions, a.positions)
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
