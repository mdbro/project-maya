"""Tests of tools/mm3/native_stored_control.py, and of the speed evaluator and the usability gate taking a control stored from
another engine start:  cd tools/mm3 && python3 -B -m unittest test_native_stored_control"""
import json
import math
import pathlib
import shutil
import tempfile
import unittest

import native_evaluate as ev
import native_stored_control as sc
import native_workload
import usability_eval
from test_native_evaluate import design, make_run
from test_usability_checks import RIGHT_MUFFINS, task_row

STATE, OTHER_STATE = 'c' * 64, 'd' * 64
LEARNED = 'e' * 64
TEXTS = ('t1', 't2', 't3', 't4', 't5', 't6')
ARGS = ['--pack', '/p', '--kv-cache', 'fp4', '--serve']
FORCE = list(range(100, 260))


def softmax(xs):
    m = max(xs)
    w = [math.exp(x - m) for x in xs]
    return [v / sum(w) for v in w]


def token_file(path: pathlib.Path, shift_from: int | None, amount: float = 0.3, mode='draft') -> None:
    """A forced text's record: 40 tokens over a 6-word vocabulary; from position shift_from on, the logits are tilted by
    `amount` (None: the base record)."""
    lines = [json.dumps(dict(type='begin', schema=1, prompt_tokens=200, max_new=len(FORCE)+8, forced=len(FORCE),
                            sampling=dict(temperature=1, top_p=0.95, seed=1), vocab=300, token_topk=4, spec=int(mode=='draft')))]
    for i, tok in enumerate(FORCE):
        logits = [0.0, 1.0, 3.0, 2.0, -1.0, 0.5]
        if shift_from is not None and i >= shift_from:
            logits[1] += amount
        probs = softmax(logits)
        order = sorted(range(len(probs)), key=lambda j: (-probs[j], j))
        top = [[j, probs[j]] for j in order[:4]]
        lines.append(json.dumps({'type': 'token', 'i': i, 'pos': 200 + i, 'id': tok, 'p': 1.0, 'logp': 0.0, 'p_raw': probs[2], 'logp_raw': math.log(probs[2]), 'top': top,
                                 'rest': sum(probs[j] for j in order[4:]), 'entropy': -sum(p * math.log(p) for p in probs), 'forced': True, 'source': 'target', 'rows': 1}))
    lines.append(json.dumps({'type': 'done', 'tokens': len(FORCE), 'finish': 'forced'}))
    path.write_text('\n'.join(lines) + '\n')


def reference_record(state=STATE, pid=100, exe='a' * 64, args=ARGS, started=5000, info=None) -> dict:
    return dict(checkpoint_version=2, sha256=state, path='/states/ref.json', learned_signature=LEARNED,
                engine=dict(pid=pid, started=started, exe_sha256=exe, args=list(args), info=info or {k: dict(kv='fp4', max_context=32768).get(k, 1) for k in sc.ENGINE_FACTS}))


def restore_receipt(pid=100, exe='a' * 64, signature=LEARNED):
    return dict(signature=signature, engine_pid=pid, engine_started=5000, engine_executable_sha256=exe,
                checkpoint_version=2, lookup_reset=True,
                benchmark_transients=dict(pass_service_ms=0, pass_service_samples=0, exchange_host_waits=0,
                                          exchange_host_urgent=False, exchange_host_open_ns=0,
                                          helper_guards_cleared=True, maintenance_windows_open=True))


def make_check(root: pathlib.Path, name: str, repeats=('ref-a', 'ref-b'), ms=lambda text, mode, rep: 40.0, shift=lambda text, mode, rep: None, pid=100, exe='a' * 64,
               state=STATE, args=ARGS, settings=None, restore=LEARNED, rule=None, modes=('one', 'draft')) -> pathlib.Path:
    """A control check folder as native_followup writes it: every text in every mode, once a repeat."""
    path = root / name
    path.mkdir(parents=True)
    sequence, records, steps = [], [], []
    for text in TEXTS:
        for mode in modes:
            for rep in repeats:
                step = f'r-{rep}.{mode}-{text}'
                file = path / f'{step}.jsonl'
                token_file(file, shift(text, mode, rep), mode=mode)
                sequence.append(dict(step=step, phase='meter', kind='request', case=text, tune={'force_tokens': FORCE, 'token_log': step}))
                records.append(dict(step=step, arm=rep, mode=mode, prompt=text, token_log=step, file=str(file)))
                final = dict({**dict.fromkeys(sc.SETTINGS, 0), 'cpu_kernel': 2, 'helper_kernel': 172, 'third': 1, 'dense_fp8': 2, 'spec_policy': 'lenience' if mode == 'draft' else 'existing'}, **(settings or {}))
                steps.append(dict(step=step, kind='request', phase='meter',
                                  summary=dict(errors=[], interval_status='measured', measure_from=128, interval_start_emitted=128, interval_tokens=len(FORCE)-128, interval_ms=(len(FORCE)-128) * ms(text, mode, rep), final_stat=final,
                                               restore=restore_receipt(pid, exe, restore))))
    study = dict(records=records, check_rule=dict(required_modes=list(modes), texts=list(TEXTS), repeats=list(repeats), **(rule or {})))
    protocol = dict(sequence=sequence, study=study, conditioning=[dict(id=t, prompt='Prompt ' + t) for t in TEXTS],
                    model_name='test', max_tokens=len(FORCE)+8, seed=1, reasoning_effort='high')
    for step in sequence:
        next(s['summary'] for s in steps if s['step'] == step['step'])['workload_sha256'] = native_workload.declared(protocol, step)
    (path / 'protocol.json').write_text(json.dumps(protocol))
    phases = dict(standalone=dict(precheck=dict(pid=pid, started=5000, executable_sha256=exe), origin_signature=LEARNED, reference_state=reference_record(state, pid, exe, args)),
                  meter=dict(precheck=dict(pid=pid, started=5000, executable_sha256=exe)))
    (path / 'ledger.json').write_text(json.dumps(dict(steps=steps, phases=phases)))
    return path


def wobble(text, mode, rep, base=40.0, size=0.004):
    """A repeat spread of about 0.4% a measurement: each text's two repeats sit either side of its own time."""
    k = TEXTS.index(text)
    return (base + k) * (1 + size * (1 if (rep == 'ref-a') == (k % 2 == 0) else -1))


class Gate(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        # the stored control: each text twice; the repeats of t1 and t2 differ from position 20 on (a repeat's own drift)
        self.stored = make_check(self.tmp, 'stored', ms=wobble, shift=lambda t, m, rep: 20 if rep == 'ref-b' and t in ('t1', 't2') else None)

    def new(self, name='new', **kw):
        kw.setdefault('pid', 200)
        kw.setdefault('ms', lambda t, m, rep: 40.0 + TEXTS.index(t))
        return make_check(self.tmp, name, repeats=('ref-a',), **kw)

    def test_a_start_in_the_same_state_is_accepted(self):
        g = sc.gate(self.stored, self.new(shift=lambda t, m, rep: 25 if t == 't3' else None))
        self.assertTrue(g['ok'], g['reasons'])
        self.assertEqual(g['reference_state'], STATE)
        one = next(c for c in g['modes'] if c['mode'] == 'one')
        self.assertEqual(one['time']['texts'], 6)
        self.assertLess(abs(one['time']['mean']), one['time']['band'])
        self.assertAlmostEqual(one['stored_repeats']['time_sd'], 0.004 * math.sqrt(2), places=4)   # (two repeats 0.8% apart: one measurement 0.57%)
        self.assertTrue(one['output']['within_limit'])
        self.assertIn('ACCEPTED', sc.render(g))
        json.dumps(sc.strip(g))                                                                    # (the result goes to JSON)

    def test_another_reference_state_binary_or_start_options_is_refused(self):
        for kw, word in ((dict(state=OTHER_STATE), 'reference state'), (dict(exe='b' * 64), 'binaries'), (dict(args=ARGS + ['--native-resident-max', '400']), 'start options'),
                         (dict(settings={'helper_kernel': 188}), 'built-ins'), (dict(restore='f' * 64), 'restore')):
            g = sc.gate(self.stored, self.new('new-' + word.replace(' ', '-'), **kw))
            self.assertFalse(g['ok'], word)
            self.assertTrue(any(word in r for r in g['reasons']), (word, g['reasons']))
            self.assertIn('REFUSED', sc.render(g))

    def test_a_start_that_is_slower_is_refused(self):
        g = sc.gate(self.stored, self.new(ms=lambda t, m, rep: (40.0 + TEXTS.index(t)) * 1.02))     # (2% slower, the repeats' band is well under 1%)
        self.assertFalse(g['ok'])
        self.assertTrue(any('time' in r and '+2.00%' in r for r in g['reasons']), g['reasons'])
        one = next(c for c in g['modes'] if c['mode'] == 'one')
        self.assertFalse(one['time']['within'])

    def test_a_stated_minimum_band_widens_the_time_check(self):
        stored = make_check(self.tmp, 'stored-wide', ms=wobble, rule=dict(min_time_band=0.03))
        self.assertTrue(sc.gate(stored, self.new('slow', ms=lambda t, m, rep: (40.0 + TEXTS.index(t)) * 1.02))['ok'])

    def test_a_start_whose_output_differs_from_the_first_token_is_refused(self):
        g = sc.gate(self.stored, self.new(shift=lambda t, m, rep: 0))                               # (every text differs from position 0 on)
        self.assertFalse(g['ok'])
        self.assertTrue(any('output' in r for r in g['reasons']), g['reasons'])
        one = next(c for c in g['modes'] if c['mode'] == 'one')
        self.assertGreater(one['output']['cross'], one['output']['within'])
        self.assertEqual(one['rows'][2]['cross_first'], [0, 0])                                     # (t3: both stored repeats, from the first position)
        self.assertEqual(one['rows'][2]['within_first'], [None])                                    # (its stored repeats never differ)

    def test_a_stored_control_without_repeats_is_refused(self):
        single = make_check(self.tmp, 'single', repeats=('ref-a',))
        g = sc.gate(single, self.new())
        self.assertFalse(g['ok'])
        self.assertTrue(any('only once' in r for r in g['reasons']), g['reasons'])

    def test_what_a_folder_says_about_its_start(self):
        a, b = sc.run_identity(self.stored), sc.run_identity(self.new())
        self.assertEqual((a['reference_states'], a['pids'], a['restores_elsewhere']), ([STATE], [100], []))
        self.assertTrue(sc.same_start(a, b))                                                        # (two engine processes)
        self.assertEqual(sc.same_start(a, a), [])


class IncompleteEvidence(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        root = pathlib.Path(self.directory.name)
        self.stored = make_check(root, 'stored', ms=wobble)
        self.new = make_check(root, 'new', repeats=('check',), pid=200, ms=lambda t, m, r: 40 + TEXTS.index(t))

    def edit(self, file, change):
        path = self.new / file
        value = json.loads(path.read_text())
        change(value)
        path.write_text(json.dumps(value))

    def refused(self, text):
        result = sc.gate(self.stored, self.new)
        self.assertFalse(result['ok'])
        self.assertIn(text, '; '.join(result['reasons']))

    def test_no_replays_cannot_qualify_metadata_alone(self):
        for path in (self.stored, self.new):
            p = json.loads((path / 'protocol.json').read_text())
            p['study']['records'] = []
            (path / 'protocol.json').write_text(json.dumps(p))
        self.refused('no forced-replay records')

    def test_each_request_needs_a_restore_receipt(self):
        self.edit('ledger.json', lambda d: d['steps'][2]['summary'].pop('restore'))
        self.refused('missing or mismatched restore')

    def test_missing_settings_on_both_sides_are_not_equal_evidence(self):
        for path in (self.stored, self.new):
            p = json.loads((path / 'ledger.json').read_text())
            for step in p['steps']:
                step['summary']['final_stat'] = {}
            (path / 'ledger.json').write_text(json.dumps(p))
        self.refused('effective settings missing')

    def test_different_forced_suffix_is_not_a_shared_prefix_check(self):
        protocol = json.loads((self.new / 'protocol.json').read_text())
        sequence = {s['step']: s for s in protocol['sequence']}
        for record in protocol['study']['records']:
            sequence[record['step']]['tune']['force_tokens'][20] = 299
            path = pathlib.Path(record['file'])
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            rows[21]['id'] = 299
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        (self.new / 'protocol.json').write_text(json.dumps(protocol))
        self.refused('forced sequence or immutable input')

    def test_missing_prompt_or_repeat_is_not_an_intersection_comparison(self):
        self.edit('protocol.json', lambda d: d['study']['records'].pop())
        self.refused('coverage is incomplete')

    def test_a_missing_timing_is_not_silently_dropped(self):
        self.edit('ledger.json', lambda d: d['steps'][0]['summary'].pop('interval_ms'))
        self.refused('real token-128 timing')

    def test_forward_128_is_not_token_128(self):
        self.edit('ledger.json', lambda d: d['steps'][0]['summary'].update(interval_start_emitted=700))
        self.refused('real token-128 timing')

    def test_prompt_text_changed_under_the_same_label_is_refused(self):
        self.edit('protocol.json', lambda d: d['conditioning'][0].update(prompt='A different input'))
        self.refused('immutable workload receipt')

    def test_a_recycled_pid_does_not_match_the_application(self):
        self.edit('ledger.json', lambda d: d['steps'][0]['summary']['restore'].update(engine_started=5001))
        self.refused('different process identity')

    def test_policy_reset_must_be_complete(self):
        self.edit('ledger.json', lambda d: d['steps'][0]['summary']['restore']['benchmark_transients'].update(pass_service_samples=45))
        self.refused('complete benchmark transient reset')

    def test_one_mode_does_not_qualify_the_other(self):
        root = pathlib.Path(self.directory.name)
        a = make_check(root, 'one-stored', ms=wobble, modes=('one',))
        b = make_check(root, 'one-new', repeats=('check',), pid=200, ms=lambda t, m, r: 40 + TEXTS.index(t), modes=('one',))
        self.assertTrue(sc.gate(a, b, required_modes={'one'})['ok'])
        self.assertFalse(sc.gate(a, b, required_modes={'draft'})['ok'])

    def test_nonfinite_probability_row_is_refused(self):
        p = json.loads((self.new / 'protocol.json').read_text())
        path = pathlib.Path(p['study']['records'][0]['file'])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[20]['top'][0][1] = float('nan')
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.refused('incomplete or nonfinite')

    def test_infinite_kl_cannot_be_dropped_from_the_mean(self):
        p = json.loads((self.new / 'protocol.json').read_text())
        path = pathlib.Path(p['study']['records'][0]['file'])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        # A valid normalized distribution with zero probability for a token
        # that has positive reference probability: that position has infinite KL.
        token = rows[20]
        token['rest'] += token['top'][-1][1]
        token['top'][-1][1] = 0.0
        path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        self.refused('nonfinite per-position comparison')


class SpeedEvaluator(unittest.TestCase):
    """study.control_from: the stored arm's units join the run's only when the gate passes."""
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.stored_check = make_check(self.tmp, 'stored-check', ms=wobble)
        self.control = self.timed('stored-ab', {'ctl': lambda p, s: 30.0, 'ctl-aa': lambda p, s: 30.1}, pid=100, extra=dict(aa=['ctl-aa', 'ctl']))

    def timed(self, name, arms, pid, extra=None, state=STATE, restore=LEARNED):
        path = make_run(self.tmp, name, design(arms, restore=restore), extra or {}, pid=pid)
        ledger = json.loads((path / 'ledger.json').read_text())
        ledger['phases']['check'].update(origin_signature=LEARNED, reference_state=reference_record(state, pid))
        ledger['phases']['check']['precheck']['started'] = 5000
        protocol = json.loads((path / 'protocol.json').read_text())
        protocol.update(model_name='test', max_tokens=384, reasoning_effort='high',
                        conditioning=[dict(id=t, prompt='Prompt ' + t) for t in ('p1', 'p2', 'p3')])
        for step in protocol['sequence']:
            summary = next(s['summary'] for s in ledger['steps'] if s['step'] == step['step'])
            summary['workload_sha256'] = native_workload.declared(protocol, step)
            summary['restore'] = restore_receipt(pid, signature=restore)
        (path / 'protocol.json').write_text(json.dumps(protocol))
        (path / 'ledger.json').write_text(json.dumps(ledger))
        return path

    def run_with(self, check, **kw):
        cf = dict(run=str(self.control), arm='ctl', stored_check=str(self.stored_check), check=str(check))
        return self.timed('new-ab', {'fast': lambda p, s: 27.0}, pid=200, extra=dict(primary=[dict(arm='fast', baseline='ctl')], control_from=cf), **kw)

    def test_an_accepted_stored_control_gives_the_comparison_its_units(self):
        check = make_check(self.tmp, 'new-check', repeats=('ref-a',), pid=200, ms=lambda t, m, rep: 40.0 + TEXTS.index(t))
        res = ev.evaluate_run(self.run_with(check), per_pass=False)
        self.assertTrue(res['stored_control']['ok'], res['stored_control']['reasons'])
        self.assertEqual(res['stored_control']['units'], 6)
        self.assertEqual(res['primary']['overall']['n'], 6)                                         # (3 prompts x 2 seeds, paired across the two starts)
        self.assertAlmostEqual(res['primary']['overall']['mean'], 27.0 / 30.0 - 1)
        self.assertNotIn('snapshot_pairs', res['primary'])                                          # (both starts restored the one reference state)
        text = ev.render_run(res)
        self.assertIn("the stored control's gate (ACCEPTED", text)
        self.assertIn('another engine start', text)

    def test_several_stored_arms_one_a_mode(self):
        check = make_check(self.tmp, 'new-check', repeats=('ref-a',), pid=200, ms=lambda t, m, rep: 40.0 + TEXTS.index(t))
        cf = dict(run=str(self.control), arms={'stored': 'ctl', 'stored-aa': 'ctl-aa'}, stored_check=str(self.stored_check), check=str(check))
        run = self.timed('two-arms', {'ctl': lambda p, s: 30.3}, pid=200,
                         extra=dict(primary=[dict(arm='ctl', baseline='stored')], comparisons=[dict(arm='ctl', baseline='stored-aa')], control_from=cf))
        res = ev.evaluate_run(run, per_pass=False)
        self.assertTrue(res['stored_control']['ok'], res['stored_control']['reasons'])
        self.assertEqual(res['stored_control']['units'], 12)
        self.assertAlmostEqual(res['primary']['overall']['mean'], 30.3 / 30.0 - 1)
        self.assertAlmostEqual(res['secondary'][0]['overall']['mean'], 30.3 / 30.1 - 1)

    def test_a_refused_stored_control_leaves_the_comparison_without_units(self):
        slow = make_check(self.tmp, 'slow-check', repeats=('ref-a',), pid=200, ms=lambda t, m, rep: (40.0 + TEXTS.index(t)) * 1.03)
        res = ev.evaluate_run(self.run_with(slow), per_pass=False)
        self.assertFalse(res['stored_control']['ok'])
        self.assertEqual(res['primary']['overall']['n'], 0)
        self.assertTrue(any(n.startswith('REFUSED: the stored control') for n in res['notes']))
        self.assertIn('REFUSED', ev.render_run(res))

    def test_a_run_on_another_start_than_its_check_is_refused(self):
        check = make_check(self.tmp, 'other-check', repeats=('ref-a',), pid=300, ms=lambda t, m, rep: 40.0 + TEXTS.index(t))
        res = ev.evaluate_run(self.run_with(check), per_pass=False)                                 # (the run: pid 200; its check: pid 300)
        self.assertFalse(res['stored_control']['ok'])
        self.assertTrue(any('this run and its control check are not of one engine start' in r for r in res['stored_control']['reasons']))

    def test_a_run_restored_to_another_origin_is_refused(self):
        check = make_check(self.tmp, 'new-check', repeats=('ref-a',), pid=200, ms=lambda t, m, rep: 40.0 + TEXTS.index(t))
        res = ev.evaluate_run(self.run_with(check, restore='snap1'), per_pass=False)                # (its requests restored some other snapshot)
        self.assertFalse(res['stored_control']['ok'])


class UsabilityGate(unittest.TestCase):
    """control_from with both control checks: the answers of two engine starts give a verdict only as a stored control."""
    TASKS = [dict(id='order-jobs', prompt='Write order_jobs.', max_tokens=6144), dict(id='muffins', prompt='How many combinations?', max_tokens=6144)]

    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.stored_check = make_check(self.tmp, 'stored-check', ms=wobble)
        self.new_check = make_check(self.tmp, 'new-check', repeats=('ref-a',), pid=200, ms=lambda t, m, rep: 40.0 + TEXTS.index(t))

    def folder(self, name, arm, pid, settings, protocol, state=STATE, record_state=True, started=5000):
        d = self.tmp / name
        d.mkdir()
        (d / 'protocol.json').write_text(json.dumps(dict(experiment_id=name, tasks=self.TASKS, sequence=[dict(arm=arm, seed=s) for s in (1, 2)], **protocol)))
        engine = dict(pid=pid, started=started, exe='/artifacts/maya-mm3', exe_sha256='a' * 64, args=ARGS, info=dict(kv='fp4', max_context=32768))
        for s in (1, 2):
            rows = [task_row('order-jobs', check=dict(status='pass'), settings=dict(spec_rounds=0, **settings)), task_row('muffins', RIGHT_MUFFINS, settings=dict(spec_rounds=0, **settings))]
            doc = dict(arm=arm, seed=s, engine_pid=pid, engine=engine, tasks=rows)
            if record_state:
                doc['reference_state'] = dict(checkpoint_version=2, sha256=state, learned_signature=LEARNED, restore_signature=LEARNED, restore=restore_receipt(pid))
            (d / f'tasks-{arm}-s{s}.json').write_text(json.dumps(doc))
        return d

    def run_folder(self, check=None, **kw):
        control = self.folder('stored-usab', 'old', 100, dict(dense_fp8=1), dict(arms=dict(old={}), control='old', expect=dict(old=dict(dense_fp8=1))))
        cf = dict(required_modes=['one', 'draft'], run=str(control), arm='old', stored_check=str(self.stored_check), check=str(check or self.new_check))
        return self.folder('new-usab', 'new', 200, dict(dense_fp8=2), dict(arms=dict(new={}), control=None, control_from=cf, expect=dict(new=dict(dense_fp8=2))), **kw)

    def test_a_stored_control_gives_a_verdict(self):
        d = self.run_folder()
        lines, r = usability_eval.verdict(d)
        self.assertTrue(r['can_judge'], r['confounds'])
        self.assertEqual(r['outcomes']['new|order-jobs']['outcome'], "can't tell")                  # (0 of 2 against 0 of 2: what two seeds can show)
        text = '\n'.join(lines)
        self.assertIn('accepted as a stored control', text)
        self.assertIn('stored control: control check, time, one row', text)
        self.assertNotIn('NO VERDICT', text)
        pre, ok = usability_eval.preflight_lines(json.loads((d / 'protocol.json').read_text()))
        self.assertTrue(ok)
        self.assertIn('if the stored control is accepted after the run', '\n'.join(pre))

    def test_answers_not_recorded_in_the_reference_state_give_no_verdict(self):
        lines, r = usability_eval.verdict(self.run_folder(record_state=False))
        self.assertFalse(r['can_judge'])
        self.assertTrue(any('not recorded as restored to the reference state' in c for c in r['confounds']), r['confounds'])
        self.assertIn('NO VERDICT', '\n'.join(lines))

    def test_answer_sequence_requires_its_actual_restore_receipt(self):
        directory = self.run_folder()
        path = directory / 'tasks-new-s1.json'
        record = json.loads(path.read_text())
        del record['reference_state']['restore']
        path.write_text(json.dumps(record))
        _, result = usability_eval.verdict(directory)
        self.assertFalse(result['can_judge'])
        self.assertTrue(any('restore' in c for c in result['confounds']))

    def test_actual_drafting_requires_drafting_coverage(self):
        check = make_check(self.tmp, 'one-only', repeats=('check',), pid=200,
                           ms=lambda t, m, r: 40 + TEXTS.index(t), modes=('one',))
        directory = self.run_folder(check=check)
        protocol_path = directory / 'protocol.json'
        protocol = json.loads(protocol_path.read_text())
        protocol['control_from']['required_modes'] = ['one']
        protocol_path.write_text(json.dumps(protocol))
        for path in directory.glob('tasks-*.json'):
            record = json.loads(path.read_text())
            for task in record['tasks']:
                task['settings']['spec_rounds'] = 10
            path.write_text(json.dumps(record))
        _, result = usability_eval.verdict(directory)
        self.assertFalse(result['can_judge'])
        self.assertTrue(any('required modes' in c for c in result['confounds']))

    def test_a_drifted_control_check_gives_no_verdict(self):
        slow = make_check(self.tmp, 'slow-check', repeats=('ref-a',), pid=200, ms=lambda t, m, rep: (40.0 + TEXTS.index(t)) * 1.03)
        lines, r = usability_eval.verdict(self.run_folder(check=slow))
        self.assertFalse(r['can_judge'])
        self.assertTrue(any('the stored control is refused' in c and 'time' in c for c in r['confounds']), r['confounds'])

    def test_answers_from_another_start_than_the_check_give_no_verdict(self):
        check = make_check(self.tmp, 'other-check', repeats=('ref-a',), pid=300, ms=lambda t, m, rep: 40.0 + TEXTS.index(t))
        _, r = usability_eval.verdict(self.run_folder(check=check))
        self.assertFalse(r['can_judge'])
        self.assertTrue(any('not tied to its control check' in c for c in r['confounds']), r['confounds'])


if __name__ == '__main__':
    unittest.main()
