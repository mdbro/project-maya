#!/usr/bin/env python3
"""May a run be compared with a control stored from another engine start? The gate, and the short control check behind it.

The user (2026-11-14): "We should not need to repeat the control every time. Speed test runs should be comparable to a
common control, regardless of if they come from the same engine start." A control from another start is accepted only when

  1. both starts were forced into the same reference state (native_reference_state.py: the same identity in both ledgers);
  2. the same binary served both (SHA-256), with the same start options, and the engine reports the same facts about itself;
  3. the built-ins match: the check's requests report the same settings in both starts;
  4. the control check of the new start lands inside the stored start's own repeat spread (its A/A), in time and in output.

The control check is a set of forced replays: recorded texts decoded with every token forced, in both modes, so the work is
the same on every start and nothing is sampled. The stored control holds each text at least twice (its repeats); a later run
records each once. Two measures, each a text at a time and then over the texts of one mode:

  time     ms a token from token 128. The repeats of one text in the stored start give the spread of one measurement
           (pooled over the texts, as a fraction of the text's time). The new start's time over the stored mean, averaged
           over the texts, must lie within the two-sided 95% band that spread gives for such a mean. On 2026-11-14 the
           repeats differed by 0.7% a text at one row and 1.7% in drafting (kbatch2), against 2.8% and 9% for free decoding.
  output   the divergence between two records of one text (token_records.compare: the mean KL over the top-64 partition,
           nats). For each text: the new record against each stored repeat (cross), and the stored repeats against each
           other (within). The new start is refused when cross exceeds within over the texts, by a one-sided paired t test at
           5%. A start in another state differs from the first tokens on; a start in the same state differs only where a
           repeat within one start also does.

A band narrower than `min_time_band` (a stored control may state one; 0 by default) is widened to it. Everything here is
measured from the two run folders; nothing is modeled. Reads only.

    native_stored_control.py gate STORED_CHECK_DIR NEW_CHECK_DIR [--json FILE]     exit 1 when the stored control is refused
    native_stored_control.py repeats CHECK_DIR                                      a check folder's own repeat spread"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import pathlib
import statistics
import sys

HERE = pathlib.Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import native_workload
import native_evaluate as ev   # noqa: E402  (interval, t_quantile, CHECKED_KEYS)

ENGINE_FACTS = ('kv', 'max_context', 'ctx', 'head_lowp_formats', 'head_bf16', 'expert_slots', 'ram_slots', 'version', 'maya_version', 'spec_drafter',
                'spec_plan_drafter')   # (usability_eval.ENGINE_FACTS)
SETTINGS = tuple(dict.fromkeys(ev.CHECKED_KEYS + (
    'spec_ell', 'spec_plan_file', 'routing_floor', 'routing_prior_bonus_one', 'prompt_path', 'head_lowp', 'head_lowp_format', 'cpu_kernel', 'cpu_threads',
    'cpu_l3_prefetch', 'exchange_schedule', 'exchange_quiet', 'primary_adaptive', 'primary_policy', 'primary_horizon', 'primary_max_exchanges',
    'cooperative_cpu_scale', 'routing', 'routing_ratio', 'router_kernel', 'serve_monitoring',
    'head_shape', 'cooperative_stage_reuse', 'helper_dispatch', 'helper_fast', 'prep_graph', 'routing_prefetch',
    'primary_backing_enabled', 'exchange_host_wait', 'prompt_layerwise', 'prompt_two_pass', 'spec_fixed_width',
    'spec_rho', 'cpu_first', 'stage_overlap', 'verify_segment')))
MODES = (('one', 'one row'), ('draft', 'drafting'))


def _records_module(tree: pathlib.Path | None = None):
    path = (tree or HERE) / 'token_records.py'
    spec = importlib.util.spec_from_file_location('token_records', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _same(a, b) -> bool:
    try:
        return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return str(a) == str(b)


# ---- what a run folder says about its engine start --------------------------------------------------------------------------------
def run_identity(run_dir: pathlib.Path) -> dict:
    """Require every measured request's restore to belong to the recorded process and reference."""
    run_dir = pathlib.Path(run_dir)
    ledger = json.loads((run_dir / 'ledger.json').read_text())
    protocol = json.loads((run_dir / 'protocol.json').read_text())
    sequence = {x['step']: x for x in protocol.get('sequence', [])}
    phases = ledger.get('phases') or {}
    refs = [p['reference_state'] for p in phases.values() if isinstance(p.get('reference_state'), dict)]
    pre = [p['precheck'] for p in phases.values() if p.get('precheck')]
    engines = [r.get('engine') or {} for r in refs]
    learned = {r.get('learned_signature') for r in refs}
    problems = []
    if not refs or any(not r.get('sha256') or not r.get('learned_signature') or r.get('checkpoint_version') != 2 for r in refs):
        problems.append('complete version-2 reference application is missing')
    process_keys = {(e.get('pid'), e.get('started'), e.get('exe_sha256')) for e in engines}
    if len(process_keys) != 1 or any(any(v is None for v in k) for k in process_keys):
        problems.append('reference applications do not identify one complete engine process')
    for check in pre:
        if (check.get('pid'), check.get('started'), check.get('executable_sha256')) not in process_keys:
            problems.append('a phase precheck does not match its reference application process')
    if not pre:
        problems.append('phase precheck is missing')
    restores = set()
    for step in ledger.get('steps', []):
        summary = step.get('summary') or {}
        planned = sequence.get(step.get('step'), {})
        kind = step.get('kind', planned.get('kind'))
        measured = summary.get('interval_status') == 'measured'
        if not measured and not (kind in ('request', 'fixture') and planned.get('restore', True)):
            continue
        receipt = summary.get('restore') or {}
        signature = receipt.get('signature')
        restores.add(signature)
        prefix = str(step.get('step')) + ': '
        if not signature or signature not in learned:
            problems.append(prefix + 'missing or mismatched restore signature')
        key = (receipt.get('engine_pid'), receipt.get('engine_started'), receipt.get('engine_executable_sha256'))
        if any(v is None for v in key) or key not in process_keys:
            problems.append(prefix + 'restore belongs to a missing or different process identity')
        transients = receipt.get('benchmark_transients') or {}
        if receipt.get('checkpoint_version') != 2 or receipt.get('lookup_reset') is not True or any(
                transients.get(k) != 0 for k in ('pass_service_ms', 'pass_service_samples', 'exchange_host_waits',
                                                'exchange_host_urgent', 'exchange_host_open_ns')) or any(
                transients.get(k) is not True for k in ('helper_guards_cleared', 'maintenance_windows_open')):
            problems.append(prefix + 'complete benchmark transient reset is not recorded')
    return dict(run=str(run_dir), reference_states=sorted({str(r.get('sha256')) for r in refs}),
                learned_signatures=sorted(str(x) for x in learned),
                restores_elsewhere=sorted(str(x) for x in restores - learned),
                pids=sorted({p.get('pid') for p in pre}, key=str), binaries=sorted({p.get('executable_sha256') for p in pre}, key=str),
                processes=sorted(process_keys, key=str), problems=problems,
                engine=engines[0] if engines else None,
                origins=sorted({str(p.get('origin_signature')) for p in phases.values() if p.get('origin_signature')}),
                phases_without_reference=sorted(n for n, p in phases.items() if p.get('origin_signature') and not p.get('reference_state')))


def same_start(a: dict, b: dict) -> list[str]:
    """Why two run folders are not of one engine start in one reference state (an empty list: they are)."""
    out = list(a.get('problems', [])) + list(b.get('problems', []))
    if len(a.get('processes', [])) != 1 or a.get('processes') != b.get('processes'):
        out.append('different or missing process start times or executable identities')
    if len(a['pids']) != 1 or a['pids'] != b['pids']:
        out.append(f"engine processes {a['pids']} and {b['pids']}")
    if len(a['binaries']) != 1 or a['binaries'] != b['binaries']:
        out.append('different binaries')
    if len(a['reference_states']) != 1 or a['reference_states'] != b['reference_states']:
        out.append(f"reference states {[x[:12] for x in a['reference_states'] if x]} and {[x[:12] for x in b['reference_states'] if x]}")
    for x in (a, b):
        if x['phases_without_reference']:
            out.append(f"{pathlib.Path(x['run']).name}: phases restored to an origin that is not a reference state ({', '.join(x['phases_without_reference'])})")
        if x['restores_elsewhere']:
            out.append(f"{pathlib.Path(x['run']).name}: {len(x['restores_elsewhere'])} restore signatures are not the reference state's")
    return out


# ---- a check folder ---------------------------------------------------------------------------------------------------------------
def read_check(run_dir: pathlib.Path, tr=None) -> dict:
    """Read only complete, declared forced work with token-128 timing and effective controls."""
    run_dir = pathlib.Path(run_dir)
    tr = tr or _records_module()
    protocol = json.loads((run_dir / 'protocol.json').read_text())
    ledger = json.loads((run_dir / 'ledger.json').read_text())
    steps = {s['step']: s.get('summary') or {} for s in ledger.get('steps', [])}
    seq = {s['step']: s for s in protocol.get('sequence', [])}
    rows, notes = {}, []
    rule = (protocol.get('study') or {}).get('check_rule') or {}
    required_modes, texts = rule.get('required_modes') or [], rule.get('texts') or []
    if not required_modes or not texts or any(m not in dict(MODES) for m in required_modes):
        notes.append('the check has no valid declared modes and corpus')
    declared_repeats = rule.get('repeats') or []
    if not declared_repeats:
        notes.append('the required repeat names are not declared')
    records = (protocol.get('study') or {}).get('records', [])
    if not records:
        notes.append('no forced-replay records are declared')
    for r in records:
        name = r['step']
        try:
            summary, step = steps[name], seq[name]
            if summary.get('errors'):
                raise ValueError(f"the step reported errors {summary['errors']}")
            path = pathlib.Path(r['file'])
            if not path.is_absolute():
                path = run_dir / path
            cap = tr.read(path)
            force = {**protocol.get('tune', {}), **step.get('tune', {})}['force_tokens']
            bad = tr.check_forced(cap, force)
            if bad:
                raise ValueError('forced decoding not honoured: ' + '; '.join(bad[:3]))
            if not force or cap.bad_lines or cap.ids != force or (cap.done or {}).get('tokens') != len(force):
                raise ValueError('the full declared force list was not captured')
            for index, token in enumerate(cap.tokens):
                top = token.get('top') or []
                numbers = [p[1] for p in top] + [token.get('rest', math.nan)]
                if token.get('i') != index or token.get('invalid_row') or not top or any(
                        not math.isfinite(v) or v < 0 or v > 1 for v in numbers) or abs(sum(numbers) - 1) > 1e-4 or not math.isfinite(token.get('entropy', math.nan)):
                    raise ValueError(f'probability record {index} is incomplete or nonfinite')
            ms, count, start = summary.get('interval_ms'), summary.get('interval_tokens'), summary.get('interval_start_emitted')
            if summary.get('measure_from') != 128 or summary.get('interval_status') != 'measured' or not isinstance(start, (int, float)) or not 128 <= start < 144 or \
                    not isinstance(count, (int, float)) or count <= 0 or start + count != len(force) or not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0:
                raise ValueError('a complete real token-128 timing interval is missing')
            final = summary.get('final_stat') or {}
            missing = [k for k in SETTINGS if k not in final or final[k] is None]
            if missing:
                raise ValueError('effective settings missing: ' + ', '.join(missing))
            work = native_workload.declared(protocol, step)
            if summary.get('workload_sha256') != work:
                raise ValueError('the immutable workload receipt is missing or differs from the protocol')
            begin = cap.begin or {}
            begin_keys = ('prompt_tokens', 'max_new', 'forced', 'sampling', 'vocab', 'token_topk')
            if any(k not in begin for k in begin_keys) or begin.get('spec') not in (0, 1):
                raise ValueError('input/tokenization metadata is missing from the record')
            if ('draft' if begin['spec'] else 'one') != r['mode']:
                raise ValueError('the recorded executed mode differs from the declared mode')
            key = (r['prompt'], r['mode'], r['arm'])
            if key in rows:
                raise ValueError('duplicate corpus/repeat key')
            rows[key] = dict(step=name, ms=ms / count, cap=cap, tokens=count, force=force, workload=work,
                             begin={k: begin[k] for k in begin_keys}, settings={k: final[k] for k in SETTINGS})
        except (KeyError, OSError, TypeError, ValueError, IndexError) as exc:
            notes.append(f'{name}: {exc}')
    expected_keys = {(t, m, r) for t in texts for m in required_modes for r in declared_repeats}
    if set(rows) != expected_keys:
        notes.append('the declared corpus/mode/repeat coverage is incomplete or has extra records')
    band = float(rule.get('min_time_band') or 0.0)
    if not math.isfinite(band) or band < 0:
        notes.append('invalid minimum timing band')
    return dict(run=str(run_dir), identity=run_identity(run_dir), rows=rows, notes=notes, min_time_band=band,
                required_modes=required_modes, texts=texts, repeats=sorted({k[2] for k in rows}))


def divergence(tr, x, y) -> dict:
    """Two records of one forced text: the mean KL (nats), the share of positions above 0.1 nats, whether they are the same
    at every position, and the first position where they are not."""
    cmp = tr.compare(x, y)
    if cmp['aligned']['diverged_at'] is not None or cmp['aligned']['compared'] != len(x.tokens) or len(x.tokens) != len(y.tokens):
        raise ValueError('forced comparison did not cover the complete identical token sequence')
    s, ser = cmp['summary'], cmp['series']
    if not math.isfinite(s['kl']['mean']) or any(not math.isfinite(value) for value in ser['kl']):
        raise ValueError('nonfinite per-position comparison statistic')
    first = next((i for i, (k, t) in enumerate(zip(ser['kl'], ser['top1'])) if k != 0.0 or t != 1), None)
    exact = cmp['aligned']['diverged_at'] is None and first is None
    return dict(kl=s['kl'].get('mean'), tail=s['shares'].get('kl_over_0.1'), exact=exact, first_difference=first, compared=cmp['aligned']['compared'])


def _mode_rows(check: dict, mode: str) -> dict:
    """text -> {repeat: row} for one mode."""
    out: dict = {}
    for (text, m, rep), row in check['rows'].items():
        if m == mode:
            out.setdefault(text, {})[rep] = row
    return out


def repeat_spread(check: dict, mode: str, tr=None) -> dict:
    """A check folder's own repeats in one mode: the spread of one time measurement (pooled within-text variance of the
    relative time; its degrees of freedom), and each pair's divergence."""
    tr = tr or _records_module()
    texts = _mode_rows(check, mode)
    ss, df, pairs = 0.0, 0, []
    for text, reps in texts.items():
        ms = [r['ms'] for r in reps.values() if r['ms']]
        if len(ms) >= 2:
            mean = statistics.fmean(ms)
            ss += sum((x / mean - 1) ** 2 for x in ms)
            df += len(ms) - 1
        names = sorted(reps)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pairs.append(dict(text=text, pair=(names[i], names[j]), time=(reps[names[j]]['ms'] / reps[names[i]]['ms'] - 1) if reps[names[i]]['ms'] and reps[names[j]]['ms'] else None,
                                  **divergence(tr, reps[names[i]]['cap'], reps[names[j]]['cap'])))
    return dict(mode=mode, texts=len(texts), time_sd=math.sqrt(ss / df) if df else None, time_df=df, pairs=pairs,
                divergence=statistics.fmean(p['kl'] for p in pairs) if pairs else None, exact=sum(p['exact'] for p in pairs))


def compare_mode(stored: dict, new: dict, mode: str, tr=None) -> dict:
    """The new start's check against the stored start's, one mode: time and output, as the module's text describes."""
    tr = tr or _records_module()
    a, b = _mode_rows(stored, mode), _mode_rows(new, mode)
    texts = stored['texts']
    out = dict(mode=mode, texts=texts, reasons=[], rows=[])
    if not texts or set(a) != set(texts) or set(b) != set(texts):
        out['reasons'].append('the complete declared corpus is not present in both checks')
        return out
    for t in texts:
        all_rows = list(a[t].values()) + list(b[t].values())
        for row in all_rows[1:]:
            if any(row[k] != all_rows[0][k] for k in ('force', 'workload', 'begin')):
                out['reasons'].append(f'{t}: forced sequence or immutable input/measurement contract differs')
                return out
        if any(not r['ms'] or not math.isfinite(r['ms']) for r in all_rows):
            out['reasons'].append(f'{t}: a valid timing is missing')
            return out
    spread = repeat_spread(stored, mode, tr)
    out['stored_repeats'] = spread
    reps = max(len(a[t]) for t in texts)
    if min(len(a[t]) for t in texts) < 2:
        out['reasons'].append('the stored control holds a text only once: it has no repeat spread of its own')
        return out
    time_effects, deltas = [], []
    for t in texts:
        stored_rows = [a[t][k] for k in sorted(a[t])]
        new_rows = [b[t][k] for k in sorted(b[t])]
        new_ms = statistics.fmean(r['ms'] for r in new_rows)
        stored_ms = [r['ms'] for r in stored_rows]
        effect = new_ms / statistics.fmean(stored_ms) - 1
        cross = [divergence(tr, r['cap'], n['cap']) for r in stored_rows for n in new_rows]
        within = [p for p in spread['pairs'] if p['text'] == t]
        c, w = statistics.fmean(x['kl'] for x in cross), statistics.fmean(p['kl'] for p in within)
        out['rows'].append(dict(text=t, time=effect, new_ms=new_ms, stored_ms=statistics.fmean(stored_ms) if stored_ms else None, cross=c, within=w,
                                cross_exact=sum(x['exact'] for x in cross), cross_pairs=len(cross), within_exact=sum(p['exact'] for p in within), within_pairs=len(within),
                                cross_first=[x['first_difference'] for x in cross], within_first=[p['first_difference'] for p in within]))
        if effect is not None:
            time_effects.append(effect)
        deltas.append(c - w)
    # time: the mean over the texts of (new / stored mean - 1) against the band the stored repeats give for such a mean
    n = len(time_effects)
    sd, df = spread['time_sd'], spread['time_df']
    if not n or sd is None or df < 1:
        out['reasons'].append('no time could be compared')
        band = None
    else:
        band = ev.t_quantile(0.975, df) * sd * math.sqrt((1 + 1 / reps) / n)
        band = max(band, stored['min_time_band'])
    mean_time = statistics.fmean(time_effects) if time_effects else None
    out['time'] = dict(mean=mean_time, band=band, texts=n, repeat_sd=sd, df=df, within=None if band is None or mean_time is None else abs(mean_time) <= band,
                       interval=ev.interval(time_effects))
    if out['time']['within'] is False:
        out['reasons'].append(f"time drifted: {100 * mean_time:+.2f}% over {n} texts, outside the stored repeats' band of +/-{100 * band:.2f}%")
    # output: cross minus within, paired by text; refused when cross is above within (one-sided, 5%)
    m = len(deltas)
    mean_delta = statistics.fmean(deltas)
    if m >= 2 and statistics.stdev(deltas) > 0:
        limit = ev.t_quantile(0.95, m - 1) * statistics.stdev(deltas) / math.sqrt(m)
    else:
        limit = 0.0
    out['output'] = dict(cross=statistics.fmean(r['cross'] for r in out['rows']), within=statistics.fmean(r['within'] for r in out['rows']), mean_delta=mean_delta,
                         limit=limit, texts=m, within_limit=mean_delta <= limit + 1e-12,
                         cross_exact=sum(r['cross_exact'] for r in out['rows']), cross_pairs=sum(r['cross_pairs'] for r in out['rows']),
                         within_exact=sum(r['within_exact'] for r in out['rows']), within_pairs=sum(r['within_pairs'] for r in out['rows']))
    if not out['output']['within_limit']:
        out['reasons'].append(f"output drifted: the new start's records sit {out['output']['cross']:.4f} nats from the stored ones, the stored repeats "
                              f"{out['output']['within']:.4f} from each other (difference {mean_delta:+.4f}, above the limit {limit:.4f})")
    return out


# ---- the gate ---------------------------------------------------------------------------------------------------------------------
def gate(stored_dir: pathlib.Path, new_dir: pathlib.Path, tr=None, *, required_modes=None) -> dict:
    """May a control from the stored check's engine start stand in for one on the new check's start? Lines (name, passes,
    text) as usability_eval prints them, the reasons it is refused, and the measures."""
    tr = tr or _records_module()
    stored, new = read_check(stored_dir, tr), read_check(new_dir, tr)
    si, ni = stored['identity'], new['identity']
    lines, reasons = [], []

    def line(name, ok, text, why=None):
        lines.append((name, ok, text))
        if ok is False:
            reasons.append(why or f'{name}: {text}')
    for x, who in ((stored, 'the stored control'), (new, 'this run')):
        for n in x['notes']:
            reasons.append(f'{who}\'s check: {n}')
    for ident, who in ((si, 'the stored control'), (ni, 'this run')):
        reasons += [f'{who}: {problem}' for problem in ident['problems']]
    declared_modes = set(stored['required_modes'])
    needed = set(required_modes) if required_modes is not None else declared_modes
    if not needed or not needed <= declared_modes or not needed <= set(new['required_modes']):
        reasons.append('the consuming study lacks control-check coverage for its required modes')
    if set(stored['texts']) != set(new['texts']):
        reasons.append('the declared control-check corpora differ')
    states = (si['reference_states'], ni['reference_states'])
    ok = len(states[0]) == 1 and states[0] == states[1] and states[0][0] is not None
    line('both starts in one reference state', ok, f"yes: {states[0][0][:16]}" if ok else
         f"NO: the stored control {[str(x)[:16] for x in states[0]] or 'none'}, this run {[str(x)[:16] for x in states[1]] or 'none'}",
         'the two starts were not forced into the same reference state')
    for x, who in ((si, 'the stored control'), (ni, 'this run')):
        if x['phases_without_reference'] or x['restores_elsewhere']:
            line(f'{who}: every request restored to the reference state', False,
                 'NO: ' + (', '.join(x['phases_without_reference']) or f"{len(x['restores_elsewhere'])} restore signatures are not the state's"))
    ok = len(si['binaries']) == 1 and bool(si['binaries'][0]) and si['binaries'] == ni['binaries']
    line('one binary', ok, f"yes: sha256 {si['binaries'][0][:12]}" if ok else f"NO: {[str(b)[:12] for b in si['binaries']]} and {[str(b)[:12] for b in ni['binaries']]}",
         'the two starts ran different binaries')
    se, ne = si['engine'] or {}, ni['engine'] or {}
    if not se.get('args') or not ne.get('args'):
        line('the same start options', False, 'NOT RECORDED', 'the start options are not recorded for both starts')
    else:
        common = set(se['args']) & set(ne['args'])
        only = ' | '.join(' '.join(x for x in e['args'] if x not in common) or '(nothing more)' for e in (se, ne))
        ok = se['args'] == ne['args']
        line('the same start options', ok, 'yes' if ok else f'NO, only in one: {only}', f'the start options differ: {only}')
    for e, who in ((se, 'stored'), (ne, 'new')):
        missing = [k for k in ENGINE_FACTS if (e.get('info') or {}).get(k) is None]
        if missing:
            reasons.append(f'{who}: required engine facts missing: ' + ', '.join(missing))
    facts = [k for k in ENGINE_FACTS if json.dumps((se.get('info') or {}).get(k)) != json.dumps((ne.get('info') or {}).get(k))]
    line('the engine reports the same facts', not facts, 'yes' if not facts else 'NO: ' + ', '.join(f"{k} ({(se.get('info') or {}).get(k)} / {(ne.get('info') or {}).get(k)})" for k in facts),
         'what the engine reports about itself differs: ' + ', '.join(facts))
    differ = set()
    for (text, mode, rep), row in new['rows'].items():
        for other in (r for (t, m, _), r in stored['rows'].items() if t == text and m == mode):
            for k in set(row['settings']) | set(other['settings']):
                if not _same(row['settings'].get(k), other['settings'].get(k)):
                    differ.add(f"{k} ({other['settings'].get(k)} / {row['settings'].get(k)}, {mode})")
    line('the same built-ins (the settings the check\'s requests reported)', not differ, 'yes' if not differ else 'NO: ' + '; '.join(sorted(differ)),
         'the built-ins differ: ' + '; '.join(sorted(differ)))
    line('another engine start', None, 'yes' if si['pids'] != ni['pids'] else 'no: the same engine process served both')
    modes = []
    for mode, label in MODES:
        if mode not in needed:
            continue
        try:
            c = compare_mode(stored, new, mode, tr)
        except (ValueError, TypeError, KeyError, ArithmeticError) as exc:
            c = dict(mode=mode, texts=[], rows=[], reasons=[f'invalid full replay comparison: {exc}'])
        modes.append(c)
        t, o = c.get('time'), c.get('output')
        if t and t['within'] is not None:
            line(f'control check, time, {label}', t['within'], f"{100 * t['mean']:+.2f}% over {t['texts']} texts (new start over the stored mean, ms a token from token 128); "
                 f"the stored repeats' band +/-{100 * t['band']:.2f}% (one measurement's spread {100 * t['repeat_sd']:.2f}%)",
                 f"control check, time, {label}: {100 * t['mean']:+.2f}% is outside +/-{100 * t['band']:.2f}%")
        if o:
            line(f'control check, output, {label}', o['within_limit'], f"new against stored {o['cross']:.4f} nats ({o['cross_exact']} of {o['cross_pairs']} pairs the same at every "
                 f"position); stored repeats {o['within']:.4f} ({o['within_exact']} of {o['within_pairs']}); difference {o['mean_delta']:+.4f}, limit {o['limit']:.4f}",
                 f"control check, output, {label}: the new start's records differ from the stored ones by more than the stored repeats differ from each other")
        for r in c['reasons']:
            if not any(r[:30] in x for x in reasons) and 'drifted' not in r:
                reasons.append(f'control check, {label}: {r}')
                lines.append((f'control check, {label}', False, r))
    return dict(ok=not reasons and all(p is not False for _, p, _ in lines), lines=lines, reasons=reasons, modes=modes,
                stored=dict(run=stored['run'], identity=si), new=dict(run=new['run'], identity=ni),
                reference_state=states[0][0] if states[0] else None)


def strip(result: dict) -> dict:
    """The gate's result without the records themselves (for JSON)."""
    return json.loads(json.dumps(result, default=str))


def render(result: dict) -> str:
    out = [f"Stored control: {result['stored']['run']}", f"This run's check: {result['new']['run']}", '']
    for name, ok, text in result['lines']:
        out.append(f"  {'pass' if ok else '    ' if ok is None else 'FAIL'}  {name}: {text}")
    out.append('')
    out.append('The stored control is ACCEPTED for this run.' if result['ok'] else 'The stored control is REFUSED for this run: ' + '; '.join(result['reasons']))
    for c in result['modes']:
        if not c['rows']:
            continue
        label = dict(MODES)[c['mode']]
        out += ['', f'{label}, a text at a time (measured): time = new start over the stored mean, ms a token from token 128; output = mean KL, nats',
                f"  {'text':<20} {'stored ms':>10} {'new ms':>9} {'time':>8}   {'new vs stored':>14} {'stored repeats':>15}   first position that differs (new vs stored | stored repeats)"]
        for r in c['rows']:
            fmt = lambda v: 'none' if v is None else str(v)
            out.append(f"  {r['text']:<20} {r['stored_ms']:>10.3f} {r['new_ms']:>9.3f} {100 * r['time']:>+7.2f}%   {r['cross']:>14.4f} {r['within']:>15.4f}   "
                       f"{', '.join(fmt(x) for x in r['cross_first'])} | {', '.join(fmt(x) for x in r['within_first'])}")
    return '\n'.join(out) + '\n'


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('gate'); a.add_argument('stored', type=pathlib.Path); a.add_argument('new', type=pathlib.Path); a.add_argument('--json', type=pathlib.Path)
    a = sub.add_parser('repeats'); a.add_argument('check', type=pathlib.Path)
    args = ap.parse_args(argv)
    if args.cmd == 'repeats':
        check = read_check(args.check)
        for mode, label in MODES:
            s = repeat_spread(check, mode)
            if s['texts']:
                print(f"{label}: {s['texts']} texts; one time measurement's spread {100 * (s['time_sd'] or 0):.2f}% ({s['time_df']} df); repeats' divergence "
                      f"{s['divergence']:.4f} nats, {s['exact']} of {len(s['pairs'])} pairs the same at every position")
        return 0
    result = gate(args.stored, args.new)
    print(render(result), end='')
    if args.json:
        args.json.write_text(json.dumps(strip(result), indent=1) + '\n')
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
