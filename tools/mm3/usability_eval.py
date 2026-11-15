#!/usr/bin/env python3
"""The short usability check: what a run can show before it starts, its verdict after, and the reader's blind packet
(2026-11-14, from the audit in runs/usability-audit-20261114; how to run and read it: docs/minimax-wsl/usability-check.md).
Reads only; no engine, no GPU.

    usability_eval.py preflight PROTOCOL.json
        Before the run. Says whether the run can give a verdict (a control arm in the same protocol, so on the same engine
        start; the arms interleaved; expected settings that differ between the arms, so the tested setting is in each answer's
        record) and what its size can and cannot show for each automatic task. Exit 1 when it can give tables but no verdict.

    usability_eval.py verdict RUN_DIR [--labels FILE] [--out DIR] [--one-start NOTE]
        After the run. RUN_DIR holds protocol.json and the tasks-ARM-sSEED.json files of native_usability_matrix.py. A verdict
        is printed only after two gates pass: every answer came from one engine start, and every answer reported its arm's
        expected settings (and those differ between the arms). Then, a task at a time and for the automatic tasks together:
        harm shown / no harm of the stated size / can't tell (usability_rule.py). An answer the checks could not decide
        (read, not scored) keeps its task open until a reader's label is given with --labels. Writes usability-verdict.txt
        and usability-verdict.json in DIR (default RUN_DIR). --one-start NOTE: for tasks files written before the engine pid
        was kept (2026-11-14), the evidence that one engine start served every answer (printed with the verdict).

    usability_eval.py sequence --control NAME --tested NAME [NAME ...] --first-seed N --seeds K
        The "sequence" of a protocol: every seed runs every arm, the order flipping from one seed to the next.

    usability_eval.py blind RUN_DIR [--out DIR]
        The reader's packet: the judged tasks' answers, and any automatic answer left to read, under neutral names (P, Q, ...)
        in a fixed shuffled order, with no arm name, token count or setting. Writes review-blind.md and review-labels.json (to
        fill in). Which name is which arm is not written anywhere: `verdict --labels` works it out again.

The automatic tasks are order-jobs (eight executed checks), muffins (the stated count) and action-items (the list's structure);
log-summary, choir-note and whatever the automatic checks leave open are the reader's.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import subprocess
import sys
import zoneinfo

TOOLS = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(TOOLS))
import usability_checks as checks   # noqa: E402
import usability_rule as rule       # noqa: E402

AUTOMATIC = checks.AUTOMATIC
OPEN = (checks.READ, checks.NOT_SCORED)


def same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


def tools_version() -> str:
    try:
        head = subprocess.run(['git', '-C', str(TOOLS), 'rev-parse', '--short', 'HEAD'], capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(['git', '-C', str(TOOLS), 'status', '--porcelain', '--', 'usability_checks.py', 'usability_rule.py', 'usability_eval.py',
                                'native_usability_matrix.py'], capture_output=True, text=True, timeout=10).stdout.strip()
        return (head or 'unknown') + (' with uncommitted changes' if dirty else '')
    except (OSError, subprocess.SubprocessError):
        return 'unknown'


def now_pacific() -> str:
    return datetime.datetime.now(zoneinfo.ZoneInfo('America/Los_Angeles')).strftime('%Y-%m-%d %H:%M %Z')


# ---- the protocol -------------------------------------------------------------------------------------------------------------
def protocol_facts(p: dict) -> dict:
    """What the protocol alone says about the run: its arms, the control and where it comes from, the balance of the sequence,
    the settings that differ between each tested arm and the control, and the problems that rule a verdict out."""
    arms = list(p.get('arms') or {})
    control = p.get('control')
    cf = p.get('control_from')
    problems, notes = [], []
    if cf:
        control = cf.get('arm')
        problems.append(f"the control ({control}) comes from another run folder ({cf.get('run')}): another engine start, so a difference between "
                        'the arms cannot be told from a difference between the starts')
    elif not control or control not in arms:
        problems.append('no control arm in this protocol (name one of its arms as "control"): tables only')
        control = None
    tested = [a for a in arms if a != control]
    seq = p.get('sequence') or []
    seeds = sorted({e['seed'] for e in seq})
    per_arm = collections.Counter(e['arm'] for e in seq)
    if control and not cf:
        for s in seeds:
            missing = [a for a in arms if not any(e['arm'] == a and e['seed'] == s for e in seq)]
            if missing:
                problems.append(f'seed {s} has no entry for {", ".join(missing)}: the arms are not matched')
        first = collections.Counter(next(e['arm'] for e in seq if e['seed'] == s) for s in seeds)
        if len(seeds) >= 2 and first.get(control, 0) in (0, len(seeds)):
            notes.append(f'the control runs {"first" if first.get(control) else "after the tested arm"} on every seed: alternate the order (control, tested, tested, control, ...) '
                         'so a drift during the run does not fall on one arm')
    expect = p.get('expect') if isinstance(p.get('expect'), dict) else {}
    differ = {}
    for a in tested:
        if control and not cf:
            ea, ec = expect.get(a), expect.get(control)
            if not ea or not ec:
                problems.append(f'no expected settings ("expect") for {a if not ea else control}: an answer cannot be checked against its arm')
                continue
            d = sorted(k for k in set(ea) | set(ec) if not same(ea.get(k), ec.get(k)))
            differ[a] = d
            if not d:
                problems.append(f'the expected settings of {a} and {control} are the same: the tested setting is not in each answer\'s record '
                                '(a start option, or a key native_usability_matrix.SETTINGS_SEEN does not keep), so no answer can be shown to have run its arm')
    base = {**rule.BASE_RATES, **(p.get('base_rates') or {})}
    meaningful = float(p.get('meaningful_failure_rate') or rule.MEANINGFUL)
    tasks = [t['id'] for t in p.get('tasks') or []]
    return dict(arms=arms, control=control, control_from=cf, tested=tested, seeds=seeds, answers_per_arm=dict(per_arm), differ=differ, problems=problems, notes=notes,
                base_rates=base, meaningful=meaningful, tasks=tasks, automatic=[t for t in tasks if t in AUTOMATIC], judged=[t for t in tasks if t not in AUTOMATIC])


def preflight_lines(p: dict) -> tuple[list[str], bool]:
    f = protocol_facts(p)
    L = [f"Usability check, before the run: {p.get('experiment_id', '(no experiment_id)')}", '',
         f"  arms: {', '.join(f['arms'])}; control: {f['control'] or 'none'}; seeds: {len(f['seeds'])}; tasks: {', '.join(f['tasks'])}"]
    for a, d in f['differ'].items():
        L.append(f"  settings recorded in each answer that differ, {a} against {f['control']}: {', '.join(d) or 'none'}")
    L += ['', '  Can this run give a verdict?']
    if f['problems']:
        L += [f'    NO: {x}' for x in f['problems']]
    else:
        L.append('    yes: the control is an arm of this protocol (one engine start), every seed runs every arm, and the arms\' expected settings differ')
    L += [f'    note: {x}' for x in f['notes']]
    L += ['', '  What its size can show (modeled from the answer counts and the base rates; the rule: one-sided exact test at 5%)']
    c = f['control']
    for a in f['tested']:
        tn, cn = f['answers_per_arm'].get(a, 0), f['answers_per_arm'].get(c, 0) if c and not f['control_from'] else 0
        if not cn:
            cn = tn   # (no control in the protocol: sized as if one of the same size ran beside it)
        for t in f['automatic']:
            L.append(f"    {a}, {t}: {rule.what_it_can_show(tn, cn, f['base_rates'].get(t, 0.10), f['meaningful'])}")
    m = f['meaningful']
    L += ['', f'  To show a tested arm failing {m:.0%} of the time as harm 80% of the time, the rule needs about {rule.size_for(m, 0.10)} answers an arm on a task whose '
          f'control fails near 10%, and about {rule.size_for(m, 0.18)} near 18% (order-jobs failed 11 of 112 answers since 2026-11-07, 4 of 22 on 2026-11-14).',
          '  The judged tasks (' + (', '.join(f['judged']) or 'none') + ') are labelled by a reader from the blind packet (usability_eval.py blind).']
    return L, not f['problems']


def interleaved_sequence(control: str, tested: list[str], seeds: list[int]) -> list[dict]:
    """The sequence of a protocol: every seed runs every arm, and the order flips from one seed to the next (control, tested,
    tested, control, ...), so a drift during the run does not fall on one arm."""
    arms = [control, *tested]
    return [dict(arm=a, seed=s) for i, s in enumerate(seeds) for a in (arms if i % 2 == 0 else arms[::-1])]


# ---- the run ------------------------------------------------------------------------------------------------------------------
def read_run(run: pathlib.Path) -> dict:
    p = json.loads((run / 'protocol.json').read_text())
    f = protocol_facts(p)
    sources = [(run, p, p.get('sequence') or [])]
    cf = f['control_from']
    if cf:
        cp = json.loads((pathlib.Path(cf['run']) / 'protocol.json').read_text())
        sources.append((pathlib.Path(cf['run']), cp, [e for e in cp['sequence'] if e['arm'] == cf['arm']]))
    rows, missing, pids = [], [], set()
    for folder, proto, entries in sources:
        for e in entries:
            path = folder / f"tasks-{e['arm']}-s{e['seed']}.json"
            if not path.exists():
                missing.append(str(path))
                continue
            r = json.loads(path.read_text())
            pids.add((str(folder), r.get('engine_pid')))
            exp = (proto.get('expect') if isinstance(proto.get('expect'), dict) else {}).get(r['arm']) or {}
            for t in r['tasks']:
                seen = t.get('settings') or {}
                lab = checks.label_row(t)
                rows.append(dict(task=t['id'], arm=r['arm'], seed=r['seed'], finish=t.get('finish'), complete=t.get('complete'), emitted=t.get('emitted') or 0,
                                 label=lab['label'], why=lab['why'], labelled_by='check',
                                 settings_wrong=[f"{k} {seen.get(k)!r} (expected {v!r})" for k, v in exp.items() if not same(seen.get(k), v)],
                                 settings_checked=len(exp), request=t.get('request'), content=t.get('content') or '', reasoning=t.get('reasoning') or '',
                                 looping=checks.repetition((t.get('content') or '') + ' ' + (t.get('reasoning') or ''))['looping'] if t.get('finish') != 'stop' else False))
    return dict(protocol=p, facts=f, rows=rows, missing=missing, pids=pids)


def blind_names(p: dict, task: str, seed: int, arms: list[str]) -> dict[str, str]:
    """arm -> a neutral name (P, Q, R, ...) for one task and seed: a fixed shuffle that does not follow the arms' order."""
    order = sorted(arms, key=lambda a: hashlib.sha256(f"{p.get('experiment_id')}|{task}|{seed}|{a}".encode()).hexdigest())
    return {a: 'PQRSTUVW'[i] for i, a in enumerate(order)}


def apply_labels(run: dict, labels_file: pathlib.Path | None) -> list[str]:
    """Reader labels onto the rows the checks left open and onto the judged tasks. The file is review-labels.json filled in
    (blind names, mapped back through the fixed shuffle). A reader's label never replaces a check's right or wrong."""
    if not labels_file:
        return []
    data = json.loads(labels_file.read_text())
    arms = run['facts']['arms'] + ([run['facts']['control']] if run['facts']['control_from'] else [])
    notes = []
    for key, v in (data.get('labels') or {}).items():
        label = (v or {}).get('label')
        if label not in (checks.RIGHT, checks.WRONG):
            continue
        task, seed, name = key.split('/')
        back = {n: a for a, n in blind_names(run['protocol'], task, int(seed), arms).items()}
        for r in run['rows']:
            if r['task'] == task and r['seed'] == int(seed) and r['arm'] == back.get(name):
                if r['labelled_by'] == 'check' and r['label'] in (checks.RIGHT, checks.WRONG, checks.CUT_OFF):
                    notes.append(f"{task} seed {seed} ({r['arm']}): the reader's label is not used, the check already says {r['label']}")
                else:
                    r.update(label=label, why=(v.get('note') or 'the reader\'s label'), labelled_by=f"reader {data.get('reader', '')}".strip())
    return notes


def counts(rows: list[dict], task: str, arm: str) -> dict:
    x = [r for r in rows if r['task'] == task and r['arm'] == arm]
    c = collections.Counter(r['label'] for r in x)
    other = sum(1 for r in x if r['finish'] not in ('stop', 'length'))
    return dict(answers=len(x), right=c[checks.RIGHT], wrong=c[checks.WRONG], read=c[checks.READ], not_scored=c[checks.NOT_SCORED], cut_off=c[checks.CUT_OFF],
                looping=sum(1 for r in x if r['looping']), unfinished_other=other, labelled=c[checks.RIGHT] + c[checks.WRONG])


def request_differences(rows: list[dict]) -> tuple[str, bool]:
    """Whether every arm sent a task the same sampling parameters and budget (the seed aside)."""
    if not any(r.get('request') for r in rows):
        return 'NOT RECORDED: these tasks files were written before the request\'s parameters were kept (2026-11-14)', True
    bad = []
    for t in sorted({r['task'] for r in rows}):
        kinds = {json.dumps({k: v for k, v in (r.get('request') or {}).items() if k != 'seed'}, sort_keys=True) for r in rows if r['task'] == t}
        if len(kinds) > 1:
            bad.append(t)
    return ('yes' if not bad else 'NO: they differ on ' + ', '.join(bad)), not bad


def verdict(run_dir: pathlib.Path, labels_file: pathlib.Path | None = None, one_start_note: str | None = None) -> tuple[list[str], dict]:
    run = read_run(run_dir)
    p, f, rows = run['protocol'], run['facts'], run['rows']
    label_notes = apply_labels(run, labels_file)
    control, tested = f['control'], f['tested']
    pids = {pid for _, pid in run['pids']}
    one_start = not f['control_from'] and len(pids) == 1 and (None not in pids or bool(one_start_note))
    start_text = (f'yes (engine pid {next(iter(pids))})' if one_start and None not in pids else
                  f'yes, by this evidence: {one_start_note}' if one_start else
                  'NO: the control is from another run folder' if f['control_from'] else
                  'NOT RECORDED: these tasks files were written before the engine pid was kept (2026-11-14)' if pids == {None} else
                  f'NO: {len(pids)} engine pids in the tasks files ({sorted(str(x) for x in pids)})')
    wrong_settings = [r for r in rows if r['settings_wrong']]
    unchecked = [r for r in rows if not r['settings_checked']]
    gate_settings = not wrong_settings and not unchecked and all(f['differ'].get(a) for a in tested) and bool(tested) and bool(control)
    complete = not run['missing']
    request_text, request_ok = request_differences(rows)
    can_judge = one_start and gate_settings and request_ok and complete and not f['problems']
    L = [f"Usability check: {p.get('experiment_id', run_dir.name)}",
         f"  tested: {', '.join(tested) or 'none'}; control: {control or 'none'}; {len(f['seeds'])} seeds; checks at {tools_version()}; evaluated {now_pacific()}", '',
         '1. Can a verdict be given? (every line must say yes before an outcome is attributed to the tested setting)',
         f"   every answer from one engine start ............ {start_text}",
         f"   every answer reported its arm's settings ...... " + ('yes' if not wrong_settings and not unchecked else
                                                                    f'NO: {len(wrong_settings)} answers reported other settings, {len(unchecked)} had none to check against'),
         '   the arms\' recorded settings differ in ......... ' + ('; '.join(f"{a}: {', '.join(d) or 'NOTHING'}" for a, d in f['differ'].items()) or ('not checked (the control is from another run folder)' if f['control_from'] else 'not checked (no control in this protocol)')),
         f"   the same sampling and budget in every arm .... {request_text}",
         f"   every entry of the protocol finished ......... {'yes' if complete else 'NO: missing ' + ', '.join(pathlib.Path(m).name for m in run['missing'])}"]
    L += [f'   problem: {x}' for x in f['problems']] + [f'   note: {x}' for x in f['notes']]
    for r in wrong_settings[:12]:
        L.append(f"   {r['arm']} seed {r['seed']} {r['task']}: " + '; '.join(r['settings_wrong']))
    L += ['', '2. Answers by task (measured: counts of answers; an answer cut off by its budget is never a failure)',
          f"   {'task':<13} {'arm':<10} {'answers':>7} {'right':>6} {'wrong':>6} {'to read':>8} {'not scored':>11} {'cut off':>8}  labelled by"]
    arms_shown = ([control] if control else []) + tested
    table = {}
    for t in f['tasks']:
        for a in arms_shown:
            c = counts(rows, t, a)
            table[(t, a)] = c
            who = 'the checks' if t in AUTOMATIC else ('a reader' if c['labelled'] else 'a reader (not yet)')
            L.append(f"   {t:<13} {a:<10} {c['answers']:>7} {c['right']:>6} {c['wrong']:>6} {c['read']:>8} {c['not_scored']:>11} {c['cut_off']:>8}  {who}"
                     + (f"; {c['looping']} cut off WHILE REPEATING (damage: read it)" if c['looping'] else '')
                     + (f"; {c['unfinished_other']} ended for another reason" if c['unfinished_other'] else ''))
    L += ['', '3. Outcome by task (the rule: one-sided exact test at 5% on the wrong counts of labelled answers; three outcomes)']
    outcomes, strata = {}, collections.defaultdict(list)
    if not can_judge:
        why = ('the control is from another engine start' if f['control_from'] else 'the engine start is not recorded' if not one_start else
               'the settings gate did not pass' if not gate_settings else 'the arms\' requests differ' if not request_ok else
               'the run is not complete' if not complete else 'see the problems above')
        L.append(f"   NO VERDICT: {why}. The counts above are measured; any difference between the arms cannot be attributed to the tested setting.")
    for a in tested:
        last_size = None
        for t in f['tasks']:
            if not control:
                break
            ct, cc = table[(t, a)], table[(t, control)]
            open_t = ct['read'] + ct['not_scored'] + cc['read'] + cc['not_scored']
            base = f['base_rates'].get(t, 0.10)
            if open_t:
                outcomes[(a, t)] = dict(outcome='open', open=open_t)
                L.append(f"   {t}, {a} against {control}: OPEN - {open_t} answer{'s need' if open_t != 1 else ' needs'} a reader's label before this task has an outcome"
                         + ('' if t in AUTOMATIC else ' (usability_eval.py blind, then verdict --labels)'))
                continue
            d = rule.decide(ct['wrong'], ct['labelled'], cc['wrong'], cc['labelled'], f['meaningful'])
            outcomes[(a, t)] = d
            if t in AUTOMATIC:
                strata[a].append((t, ct['wrong'], ct['labelled'], cc['wrong'], cc['labelled']))
            shown = d['outcome'] if can_judge else f"not attributable (as counts alone: {d['outcome']})"
            L.append(f"   {t}, {a} against {control}: {a} {ct['wrong']} wrong of {ct['labelled']}, {control} {cc['wrong']} wrong of {cc['labelled']} -> {shown}"
                     + (f" (p = {d['p']:.2f})" if d['p'] is not None else ''))
            size = rule.what_it_can_show(ct['labelled'], cc['labelled'], base, f['meaningful'])
            L.append(f"      here {a}'s failure rate is below {d['tested_upper95']:.0%} (95% bound). This size: " + ('as the task above' if size == last_size else size))
            last_size = size
        if control and len(strata[a]) > 1:
            tw, tn, cw, cn = (sum(x[i] for x in strata[a]) for i in (1, 2, 3, 4))
            sp = rule.stratified_p([x[1:] for x in strata[a]])
            each = [outcomes[(a, x[0])]['outcome'] for x in strata[a]]
            together = 'harm shown' if sp <= rule.ALPHA else 'no harm of the stated size' if all(o == 'no harm of the stated size' for o in each) else "can't tell"
            outcomes[(a, 'automatic tasks together')] = dict(outcome=together, p=sp, tasks=[x[0] for x in strata[a]], tested_wrong=tw, tested_n=tn, control_wrong=cw, control_n=cn)
            shown = together if can_judge else f"not attributable (as counts alone: {together})"
            L.append(f"   the automatic tasks together ({', '.join(x[0] for x in strata[a])}), {a} against {control}: {tw} wrong of {tn} against {cw} wrong of {cn} -> {shown} "
                     f"(p = {sp:.2f}, each task kept apart; \"no harm of the stated size\" only when every task says so)")
    L += ['', '4. Small text damage (measured by one fixed rule; reported with the quote, never a failure by itself - the user, 2026-11-10)',
          f"   {'arm':<10} {'answers':>7} {'tokens':>8} {'slips in answers':>17} {'in reasoning':>13} {'per 1,000 tokens':>17}"]
    slip_rows = []
    for a in arms_shown:
        x = [r for r in rows if r['arm'] == a]
        sa = [(r, k, q) for r in x for k, q in checks.slips(r['content'], 'content')]
        sr = [(r, k, q) for r in x for k, q in checks.slips(r['reasoning'], 'reasoning')]
        tokens = sum(r['emitted'] for r in x)
        L.append(f"   {a:<10} {len(x):>7} {int(tokens):>8} {len(sa):>17} {len(sr):>13} {1000 * (len(sa) + len(sr)) / max(1, tokens):>17.2f}")
        slip_rows += [(a, r['seed'], r['task'], 'answer', k, q) for r, k, q in sa] + [(a, r['seed'], r['task'], 'reasoning', k, q) for r, k, q in sr]
    L += [f"   {a} seed {s} {t} ({where}, {k}): ...{q}..." for a, s, t, where, k, q in slip_rows]
    L += ['', '5. For a reader']
    todo = [r for r in rows if r['label'] in OPEN]
    L += [f"   {r['task']} {r['arm']} seed {r['seed']}: {r['label']} - {r['why']}" for r in todo if r['task'] in AUTOMATIC]
    judged_open = sorted({r['task'] for r in todo if r['task'] not in AUTOMATIC})
    if judged_open:
        L.append(f"   the judged tasks {', '.join(judged_open)}: label them from the blind packet (usability_eval.py blind {run_dir}), then run verdict --labels review-labels.json")
    L += [f"   {r['task']} {r['arm']} seed {r['seed']}: cut off while repeating - read it, a loop is damage" for r in rows if r['looping']]
    L += [f'   {x}' for x in label_notes]
    if not todo and not any(r['looping'] for r in rows):
        L.append('   nothing left to read')
    L += ['', 'How to read this',
          f"   harm shown: the tested arm is wrong more often than chance explains (p <= 0.05). no harm of the stated size: not shown, and a failure rate of {f['meaningful']:.0%} or more",
          '   is excluded (95% bound). can\'t tell: this many answers cannot separate harm of that size from no harm; it is not a pass and not a failure.',
          '   By the adoption rule (the burden of proof is on harm; the user, 2026-11-08) only "harm shown", or damage a reader finds, counts against a change;',
          '   the lines above say what size of harm this run could and could not have seen.']
    result = dict(experiment_id=p.get('experiment_id'), run=str(run_dir), tested=tested, control=control, tools=tools_version(), can_judge=can_judge,
                  gates=dict(one_engine_start=one_start, settings=gate_settings, same_requests=request_ok, complete=complete, problems=f['problems']), counts={f'{t}|{a}': c for (t, a), c in table.items()},
                  outcomes={f'{a}|{t}': d for (a, t), d in outcomes.items()}, slips=[dict(arm=a, seed=s, task=t, where=w, kind=k, quote=q) for a, s, t, w, k, q in slip_rows],
                  to_read=[dict(task=r['task'], arm=r['arm'], seed=r['seed'], label=r['label'], why=r['why']) for r in todo],
                  rows=[{k: v for k, v in r.items() if k not in ('content', 'reasoning')} for r in rows])
    return L, result


def blind(run_dir: pathlib.Path) -> tuple[str, dict]:
    run = read_run(run_dir)
    p, f, rows = run['protocol'], run['facts'], run['rows']
    arms = f['arms'] + ([f['control']] if f['control_from'] else [])
    M = ['# Usability check: answers to label', '',
         'Each answer has a neutral name. Which arm wrote it is not shown; do not look it up before your labels are saved.',
         'Label every answer on its own against the task\'s criteria (docs/minimax-wsl/usability-check.md), `right` or `wrong`, in `review-labels.json`,',
         'with a one-line note and a quote for anything you mark wrong and for any small damage you see. Do not compare the answers of a seed before labelling them.', '']
    template = {}
    todo_tasks = [t for t in f['tasks'] if t not in AUTOMATIC] + [t for t in f['tasks'] if t in AUTOMATIC]
    for t in todo_tasks:
        section = []
        for s in f['seeds']:
            names = blind_names(p, t, s, arms)
            for a in sorted(arms, key=lambda a: names[a]):
                r = next((x for x in rows if x['task'] == t and x['seed'] == s and x['arm'] == a), None)
                if r is None or (t in AUTOMATIC and r['label'] not in OPEN):
                    continue
                bid = f'{t}/{s}/{names[a]}'
                template[bid] = dict(label='', note='')
                section += [f'### {bid}', ''] + ([f"(the automatic check could not decide: {r['why']})", ''] if t in AUTOMATIC else []) \
                    + (['(cut off by its token budget: label only visible damage, not completeness)', ''] if r['finish'] == 'length' else []) + [r['content'] or '(empty answer)', '']
        if section:
            prompt = next((x['prompt'] for x in p.get('tasks') or [] if x['id'] == t), '')
            M += [f'## {t}', '', 'The task as the model received it:', '', '> ' + prompt.replace('\n', '\n> '), ''] + section
    return '\n'.join(M) + '\n', dict(reader='', labels=template)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    a = sub.add_parser('preflight'); a.add_argument('protocol', type=pathlib.Path)
    a = sub.add_parser('verdict'); a.add_argument('run', type=pathlib.Path); a.add_argument('--labels', type=pathlib.Path); a.add_argument('--out', type=pathlib.Path)
    a.add_argument('--one-start', help='for tasks files without an engine pid: the evidence that one engine start served every answer')
    a = sub.add_parser('blind'); a.add_argument('run', type=pathlib.Path); a.add_argument('--out', type=pathlib.Path)
    a = sub.add_parser('sequence', help='print an interleaved sequence for a protocol')
    a.add_argument('--control', required=True); a.add_argument('--tested', required=True, nargs='+'); a.add_argument('--first-seed', type=int, required=True); a.add_argument('--seeds', type=int, required=True)
    args = ap.parse_args()
    if args.command == 'sequence':
        print(json.dumps(interleaved_sequence(args.control, args.tested, [args.first_seed + i for i in range(args.seeds)])))
        return 0
    if args.command == 'preflight':
        lines, ok = preflight_lines(json.loads(args.protocol.read_text()))
        print('\n'.join(lines))
        return 0 if ok else 1
    out = args.out or args.run
    out.mkdir(parents=True, exist_ok=True)
    if args.command == 'verdict':
        lines, result = verdict(args.run, args.labels, args.one_start)
        (out / 'usability-verdict.txt').write_text('\n'.join(lines) + '\n')
        (out / 'usability-verdict.json').write_text(json.dumps(result, indent=1, ensure_ascii=False) + '\n')
        print('\n'.join(lines))
        return 0
    text, template = blind(args.run)
    (out / 'review-blind.md').write_text(text)
    if not (out / 'review-labels.json').exists():
        (out / 'review-labels.json').write_text(json.dumps(template, indent=1) + '\n')
    print(f'{len(template["labels"])} answers to label: {out / "review-blind.md"}; fill in {out / "review-labels.json"}; then: usability_eval.py verdict {args.run} --labels {out / "review-labels.json"}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
