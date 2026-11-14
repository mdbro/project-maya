#!/usr/bin/env python3
"""The shared evaluator's rules, checked on constructed run folders before any new run exists.
Run from tools/mm3: python3 -m unittest test_native_evaluate"""
import json
import pathlib
import tempfile
import unittest

import native_evaluate as ev
import native_light as light


def make_run(root: pathlib.Path, name: str, requests: list[dict], study_extra: dict | None = None, pid: int = 100,
             exe: str = 'a' * 64, seed: int = 20261101, started: str = '2026-11-13T10:00:00+00:00') -> pathlib.Path:
    """A run folder as native_followup writes it: protocol.json (sequence, study) and ledger.json (one summary a step).
    Each request: step, arm, prompt, block, seed (or None: the protocol's), rate (ms a token), sha, and optionally
    final (extra final statistics), restore (snapshot signature), finish, tune."""
    path = root / name
    path.mkdir()
    sequence, study, steps = [], [], []
    for i, q in enumerate(requests):
        step = dict(step=q['step'], phase='check', kind='request', case=q['prompt'], max_tokens=384, capture=q.get('capture', 'off'))
        if q.get('seed') is not None:
            step['seed'] = q['seed']
        if q.get('tune'):
            step['tune'] = q['tune']
        sequence.append(step)
        entry = dict(step=q['step'], block=q['block'], arm=q['arm'], prompt=q['prompt'])
        if q.get('seed') is not None:
            entry['seed'] = q['seed']
        study.append(entry)
        final = dict(spec_rounds=100.0 if q.get('drafting', True) else 0.0, draft_off=250.0, draft_acc=200.0, target_rounds=150.0,
                     decode_staged=3000.0, decode_cpu_experts=2000.0, emitted=384.0, spec_policy='lenience', spec_ell=0.4)
        final.update(q.get('final') or {})
        steps.append(dict(step=q['step'], kind='request', phase='check', started_utc=started[:-9] + f'{i:02d}+00:00', wall_seconds=20.0,
                          summary=dict(case=q['prompt'], capture=q.get('capture', 'off'), emitted=384.0, finish=q.get('finish', 'length'),
                                       final_stat=final, output_sha256=q['sha'], measure_from=128, interval_status='measured',
                                       interval_start_emitted=128, interval_tokens=256, interval_ms=q['rate'] * 256, errors=[],
                                       restore=dict(signature=q.get('restore', 'snap1')), validation={})))
    protocol = dict(seed=seed, tune={'fresh_context': 1}, sequence=sequence, study=dict(requests=study, **(study_extra or {})))
    (path / 'protocol.json').write_text(json.dumps(protocol))
    ledger = dict(steps=steps, phases=dict(check=dict(precheck=dict(pid=pid, executable_sha256=exe))), created_utc=started)
    (path / 'ledger.json').write_text(json.dumps(ledger))
    return path


def design(arms: dict, prompts=('p1', 'p2', 'p3'), seeds=(11, 12), sha=lambda arm, p, s: f'{arm}-{p}-{s}', **kw) -> list[dict]:
    """Every arm on every prompt in every block, one seed a block: arms maps name -> f(prompt, seed) -> ms a token."""
    out = []
    for b, s in enumerate(seeds, 1):
        for p in prompts:
            for arm, rate in arms.items():
                out.append(dict(step=f'b{b}-{arm}-{p}', arm=arm, prompt=p, block=b, seed=s, rate=rate(p, s), sha=sha(arm, p, s), **kw))
    return out


class Quantile(unittest.TestCase):
    def test_exact_t_quantiles(self):
        table = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 8: 2.306, 10: 2.228, 17: 2.110, 30: 2.042, 60: 2.000}
        for df, want in table.items():
            self.assertAlmostEqual(ev.t_quantile(0.975, df), want, places=3)

    def test_interval_of_one_value_has_no_width(self):
        self.assertIsNone(ev.interval([0.1])['low'])


class Pairing(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def test_seed_per_block_pairs_by_seed_and_prompt_and_counts_prompts_times_seeds(self):
        rates = {'x': lambda p, s: 30.0 + (s - 11), 'y': lambda p, s: 33.0 + (s - 11)}
        run = make_run(self.tmp, 'r', design(rates), dict(primary=dict(arm='x', baseline='y')))
        res = ev.evaluate_run(run, per_pass=False)
        self.assertEqual(res['primary']['overall']['n'], 6)
        self.assertEqual(res['primary']['per_prompt']['p1']['interval']['n'], 2)
        self.assertAlmostEqual(res['primary']['units'][0]['effect'], res['primary']['units'][0]['arm_ms'] / res['primary']['units'][0]['base_ms'] - 1)

    def test_one_seed_for_every_block_makes_one_unit_a_prompt(self):
        reqs = design({'x': lambda p, s: 30.0 + int(p[1]) / 10, 'y': lambda p, s: 33.0}, seeds=(None, None, None), sha=lambda a, p, s: f'{a}-{p}')
        for i, r in enumerate(reqs):   # (one seed: the blocks are repeats; give the repeats distinct times)
            r['rate'] += (i % 3) * 0.1
        run = make_run(self.tmp, 'r1', reqs)
        res = ev.evaluate_run(run, cli_primary='x:y', per_pass=False)
        self.assertEqual(res['primary']['overall']['n'], 3)
        self.assertTrue(any('one seed' in n for n in res['notes']))
        self.assertIn('after the run', res['primary']['named'])
        self.assertEqual(sum(t['one_text'] for t in res['texts']), 6)

    def test_a_text_repeated_across_seeds_merges_the_units(self):
        reqs = design({'x': lambda p, s: 30.0, 'y': lambda p, s: 33.0 + s / 10},
                      sha=lambda a, p, s: f'{a}-{p}' if (a == 'x' and p == 'p1') else f'{a}-{p}-{s}')
        run = make_run(self.tmp, 'r2', reqs, dict(primary=dict(arm='x', baseline='y')))
        res = ev.evaluate_run(run, per_pass=False)
        self.assertEqual(res['primary']['merged'], 1)
        self.assertEqual(res['primary']['overall']['n'], 5)
        self.assertEqual([t['repeated_across_seeds'] for t in res['texts'] if t['arm'] == 'x' and t['prompt'] == 'p1'], [1])

    def test_a_request_that_ran_other_settings_is_left_out(self):
        reqs = design({'x': lambda p, s: 30.0, 'y': lambda p, s: 33.0})
        reqs[0]['final'] = dict(spec_policy='sampled')
        run = make_run(self.tmp, 'r3', reqs, dict(primary=dict(arm='x', baseline='y'), expect=dict(x=dict(spec_policy='lenience'), y={})))
        res = ev.evaluate_run(run, per_pass=False)
        self.assertEqual([e['step'] for e in res['excluded']], [reqs[0]['step']])
        self.assertEqual(res['primary']['overall']['n'], 5)

    def test_an_answer_that_stopped_early_is_listed(self):
        reqs = design({'x': lambda p, s: 30.0, 'y': lambda p, s: 33.0})
        reqs[1]['finish'] = 'stop'
        run = make_run(self.tmp, 'r4', reqs, dict(primary=dict(arm='x', baseline='y')))
        res = ev.evaluate_run(run, per_pass=False)
        self.assertTrue(any(u['short_window'] for u in res['primary']['units']))

    def test_arms_from_different_snapshots_report_the_interval_over_snapshot_pairs(self):
        reqs = design({'x': lambda p, s: 30.0 + p.count('2'), 'y': lambda p, s: 33.0})
        for r in reqs:
            r['restore'] = f"{r['arm']}-block{r['block']}"
        run = make_run(self.tmp, 'r5', reqs, dict(primary=dict(arm='x', baseline='y')))
        res = ev.evaluate_run(run, per_pass=False)
        self.assertEqual(res['primary']['snapshot_pairs']['n'], 2)


class Declarations(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())
        self.rates = {'x': lambda p, s: 30.0, 'y': lambda p, s: 33.0, 'y2': lambda p, s: 33.0 + (s - 11) * 0.3}

    def test_the_protocols_primary_wins_over_a_command_line_one(self):
        run = make_run(self.tmp, 'd1', design(self.rates), dict(primary=dict(arm='x', baseline='y')))
        res = ev.evaluate_run(run, cli_primary='y:x', per_pass=False)
        self.assertEqual((res['primary']['arm'], res['primary']['base']), ('x', 'y'))
        self.assertTrue(any(n.startswith('REFUSED --primary') for n in res['notes']))

    def test_no_primary_means_no_result(self):
        res = ev.evaluate_run(make_run(self.tmp, 'd2', design(self.rates)), per_pass=False)
        self.assertIsNone(res['primary'])
        self.assertTrue(any('descriptive' in n for n in res['notes']))

    def test_aa_pair_is_evaluated_and_must_have_identical_settings(self):
        reqs = design(self.rates)
        run = make_run(self.tmp, 'd3', reqs, dict(primary=dict(arm='x', baseline='y'), aa=['y', 'y2'],
                                                 arms=dict(x={'spec': 1}, y={}, y2={'spec_ell': 0.5})))
        res = ev.evaluate_run(run, per_pass=False)
        self.assertEqual(res['aa']['overall']['n'], 6)
        self.assertTrue(any('do not have identical settings' in n for n in res['notes']))

    def test_several_primaries_are_each_reported_and_tokens_per_second_is_the_decision_figure(self):
        rates = {'served': lambda p, s: 30.0, 'served-exact': lambda p, s: 32.0, 'w1': lambda p, s: 31.0}
        sel = [dict(name='best', candidates=['w1'], select_blocks=[1], measure='tok/s')]
        run = make_run(self.tmp, 'd5', design(rates, seeds=(11, 12, 13)),
                       dict(primary=[dict(arm='served', baseline='best'), dict(arm='served', baseline='served-exact')],
                            selection=sel, decision='tokens per second'))
        res = ev.evaluate_run(run, per_pass=False)
        self.assertEqual([(c['arm'], c['base']) for c in res['primaries']], [('served', 'w1'), ('served', 'served-exact')])
        self.assertAlmostEqual(res['primaries'][1]['overall_tps']['mean'], 32.0 / 30.0 - 1)
        self.assertAlmostEqual(res['primaries'][1]['overall']['mean'], 30.0 / 32.0 - 1)
        text = ev.render_run(res)
        self.assertIn('PRIMARY COMPARISON 2 of 2', text)
        self.assertIn('the decision figure', text)

    def test_a_selection_by_tokens_per_second_takes_the_fastest(self):
        # (w1: 20, 50 and 34 ms, mean 34.7 ms but 33.1 tok/s; w2: 34 ms everywhere, mean 34.0 ms but 29.4 tok/s)
        rates = {'served': lambda p, s: 30.0, 'w1': lambda p, s: {'p1': 20.0, 'p2': 50.0, 'p3': 34.0}[p], 'w2': lambda p, s: 34.0}
        for measure, want in (('tok/s', 'w1'), ('ms', 'w2')):
            sel = [dict(name='best', candidates=['w1', 'w2'], select_blocks=[1], measure=measure)]
            run = make_run(self.tmp, f'd6-{measure[:2]}', design(rates, seeds=(11, 12)), dict(primary=dict(arm='served', baseline='best'), selection=sel))
            res = ev.evaluate_run(run, per_pass=False)
            self.assertEqual(res['selections']['best']['chosen'], want)

    def test_a_selection_is_made_on_its_blocks_and_compared_on_the_others_only(self):
        rates = {'served': lambda p, s: 30.0, 'w1': lambda p, s: 31.0 if s == 11 else 29.0, 'w2': lambda p, s: 32.0}
        reqs = design(rates, seeds=(11, 12, 13))
        sel = [dict(name='best-fixed', candidates=['w1', 'w2'], select_blocks=[1])]
        run = make_run(self.tmp, 'd4', reqs, dict(primary=dict(arm='served', baseline='best-fixed'), selection=sel))
        res = ev.evaluate_run(run, per_pass=False)
        self.assertEqual(res['selections']['best-fixed']['chosen'], 'w1')
        self.assertEqual(res['primary']['blocks'], [2, 3])
        self.assertEqual(res['primary']['overall']['n'], 6)
        self.assertAlmostEqual(res['primary']['overall']['mean'], 30.0 / 29.0 - 1)


class Across(unittest.TestCase):
    def setUp(self):
        self.tmp = pathlib.Path(tempfile.mkdtemp())

    def start(self, name, label_rate, pid, t, exe='a' * 64):
        rates = {'served': lambda p, s: label_rate, 'one-row': lambda p, s: 40.0 + pid / 1000}
        return make_run(self.tmp, name, design(rates), pid=pid, started=t, exe=exe)

    def test_refused_unless_the_starts_alternate(self):
        a1 = self.start('a1', 30, 1, '2026-11-13T10:00:00+00:00')
        a2 = self.start('a2', 30, 2, '2026-11-13T11:00:00+00:00')
        b1 = self.start('b1', 31, 3, '2026-11-13T12:00:00+00:00')
        b2 = self.start('b2', 31, 4, '2026-11-13T13:00:00+00:00')
        res = ev.evaluate_across([('A', a1), ('A', a2), ('B', b1), ('B', b2)], 'served', 'one-row')
        self.assertTrue(any('alternate' in x for x in res['refused']))

    def test_refused_when_two_runs_share_an_engine_start_or_lack_a_second_start(self):
        a1 = self.start('a1', 30, 1, '2026-11-13T10:00:00+00:00')
        b1 = self.start('b1', 31, 1, '2026-11-13T11:00:00+00:00')
        res = ev.evaluate_across([('A', a1), ('B', b1)], 'served', 'one-row')
        self.assertTrue(any('one engine start' in x for x in res['refused']))
        self.assertTrue(any('at least two' in x for x in res['refused']))

    def test_refused_when_one_configuration_ran_two_binaries(self):
        a1 = self.start('a1', 30, 1, '2026-11-13T10:00:00+00:00')
        b1 = self.start('b1', 31, 2, '2026-11-13T11:00:00+00:00')
        a2 = self.start('a2', 30, 3, '2026-11-13T12:00:00+00:00', exe='c' * 64)
        b2 = self.start('b2', 31, 4, '2026-11-13T13:00:00+00:00')
        res = ev.evaluate_across([('A', a1), ('B', b1), ('A', a2), ('B', b2)], 'served', 'one-row')
        self.assertTrue(any('different binaries' in x for x in res['refused']))

    def test_alternating_starts_give_the_spread_and_an_interval_over_start_pairs(self):
        a1 = self.start('a1', 30, 1, '2026-11-13T10:00:00+00:00')
        b1 = self.start('b1', 31, 2, '2026-11-13T11:00:00+00:00')
        a2 = self.start('a2', 30.3, 3, '2026-11-13T12:00:00+00:00')
        b2 = self.start('b2', 31.6, 4, '2026-11-13T13:00:00+00:00')
        res = ev.evaluate_across([('A', a1), ('B', b1), ('A', a2), ('B', b2)], 'served', 'one-row')
        self.assertNotIn('refused', res)
        self.assertEqual(res['order'], ['A', 'B', 'A', 'B'])
        self.assertEqual(res['b_over_a']['n'], 2)
        self.assertAlmostEqual(res['spread']['A arm'][0], 0.01)
        self.assertEqual(res['units'], 6)


class PerPass(unittest.TestCase):
    """Passes from token 128 by generated-token index, on a constructed drafting capture of 3-row forwards."""

    def capture(self, forwards=100, rows=3, layers=3, gpu_expert_end=None):
        """gpu_expert_end: None, layer records as format 1 leaves them (no 5090 stamp); else layer -> the 5090's stamp
        (ns after the record's t_ns, as format 2 writes it; 0 is no stamp)."""
        recs, t = [], 1_000_000
        for i in range(forwards):
            fw = dict(type='forward', forward=500 + i, position=240 + i * rows, rows=rows, generation=True, t_ns=t,
                      sync_end=2_900_000, end=3_000_000)
            recs.append(fw)
            for li in range(layers):
                base = t + 100_000 + li * 1_000_000
                recs.append(dict(type='layer', forward=500 + i, layer=li, rows=rows, t_ns=base, pre_end=50_000, join_end=100_000,
                                 cpu_end=600_000, helper_worker_published=500_000,
                                 **({} if gpu_expert_end is None else dict(gpu_expert_end=gpu_expert_end(li)))))
                recs.append(dict(type='third_pass', forward=500 + i, layer=li, t_ns=base, worker_published=450_000))
            t += 4_000_000
        return light.LightCapture(pathlib.Path('x.light'), '0' * 64, dict(mode='light'), recs)

    def test_passes_are_kept_from_generated_token_128_not_forward_128(self):
        rows = ev.pass_rows(self.capture())
        kept_forwards = {r['forward'] for r in rows}
        self.assertEqual(len(kept_forwards), 100 - 43)          # positions 240 + 3i: index 3i >= 128 from i = 43
        self.assertEqual(min(r['token'] for r in rows), 129)

    def test_lane_differences_are_taken_inside_each_pass_from_its_join(self):
        rows = ev.pass_rows(self.capture())
        b = ev.balance(rows)
        self.assertAlmostEqual(b['cpu|3090']['mean'], 0.1)
        self.assertAlmostEqual(b['cpu|4080']['mean'], 0.15)
        self.assertAlmostEqual(b['3090|4080']['mean'], 0.05)
        self.assertAlmostEqual(b['slowest|second']['mean'], 0.1)
        self.assertEqual(b['slowest|second']['slowest'], {'cpu': 1.0})
        self.assertAlmostEqual(b['cycle_ms'], 1.0)               # join to the next layer's join
        self.assertAlmostEqual(b['next_join_wait_ms'], 0.05)
        for key in ('cpu|5090', '3090|5090', '4080|5090'):      # (no 5090 stamps, as in format 1: no 5090 lane)
            self.assertEqual(b[key], dict(n=0, mean=None))

    def test_the_5090_lane_is_its_expert_end_taken_from_the_passs_join(self):
        # join at 100 us after t_ns; CPU lane ends 0.5 ms after it, 3090 0.4, 4080 0.35. The 5090's gpu_expert_end at
        # 800 us after t_ns is 0.7 ms after the join (layers 0 and 2: the 5090 is last); at 350 us it is 0.25 (layer 1)
        rows = ev.pass_rows(self.capture(gpu_expert_end=lambda li: 350_000 if li == 1 else 800_000))
        self.assertTrue(all(abs(r['ends']['5090'] - (0.25 if r['layer'] == 1 else 0.7)) < 1e-9 for r in rows))
        b = ev.balance(rows)
        self.assertEqual(b['cpu|5090']['n'], len(rows))
        self.assertAlmostEqual(b['cpu|5090']['mean'], (0.2 + 0.25 + 0.2) / 3)
        self.assertAlmostEqual(b['3090|5090']['mean'], (0.3 + 0.15 + 0.3) / 3)
        self.assertAlmostEqual(b['4080|5090']['mean'], (0.35 + 0.1 + 0.35) / 3)
        self.assertAlmostEqual(b['cpu|3090']['mean'], 0.1)        # (the other pairs are unchanged)
        self.assertAlmostEqual(b['slowest|second']['mean'], (0.2 + 0.1 + 0.2) / 3)
        self.assertEqual(set(b['slowest|second']['slowest']), {'5090', 'cpu'})
        self.assertAlmostEqual(b['slowest|second']['slowest']['5090'], 2 / 3)
        layers = ev.per_pass_report(rows)['by_layer']
        self.assertAlmostEqual(layers[1]['cpu|5090']['mean'], 0.25)
        self.assertAlmostEqual(layers[0]['3090|5090']['mean'], 0.3)

    def test_a_layer_record_with_gpu_expert_end_0_adds_no_5090_lane(self):
        rows = ev.pass_rows(self.capture(gpu_expert_end=lambda li: 0 if li == 1 else 800_000))
        self.assertTrue(all(('5090' in r['ends']) == (r['layer'] != 1) for r in rows))
        b = ev.balance(rows)
        self.assertEqual(b['cpu|5090']['n'], sum(r['layer'] != 1 for r in rows))
        self.assertAlmostEqual(b['cpu|5090']['mean'], 0.2)
        layers = ev.per_pass_report(rows)['by_layer']
        self.assertEqual(layers[1]['cpu|5090'], dict(n=0, mean=None))
        self.assertAlmostEqual(layers[1]['slowest|second']['mean'], 0.1)   # (CPU against the 3090, as without stamps)
        self.assertEqual(layers[1]['slowest|second']['slowest'], {'cpu': 1.0})


if __name__ == '__main__':
    unittest.main()
