#!/usr/bin/env python3
"""The shared model's composition, order rule and decision rules on constructed numbers."""
import pathlib
import tempfile
import unittest

import native_shared_model as model_tool
from shared_model import grouped, timing, work


def model(**values) -> timing.SharedTiming:
    base = dict(prepare={'0': .1, '1': .15}, cpu_base=0.0, cpu_per_job=.55, cpu_share=.6, copy_share=.7, helper={'1': [.3, .3, .9], '2': [.4, .4, .4]},
                resume=.1, wait_base=.28, gpu_base=.2, gpu_per_primary=0.0, copy_ms=.55)
    base.update(values)
    return timing.SharedTiming(**base)


class Composition(unittest.TestCase):
    def test_lanes_average_per_draw_maxima_not_a_maximum_of_averages(self):
        m = model()
        # CPU 0.55 against helper draws .3, .3, .9: per draw .55, .55, .9.
        self.assertAlmostEqual(m.lanes_ms(1, 1, 0), (.55 + .55 + .9) / 3)
        self.assertGreater(m.lanes_ms(1, 1, 0), max(.55, (.3 + .3 + .9) / 3))

    def test_a_copy_stretches_a_running_cpu_job_only_while_they_overlap(self):
        m = model()
        # Copy .55 at rate .7 ends at .7857; CPU has done .4714 of .55 and finishes alone.
        cpu_end, copy_end = m.overlap_ends(1, 1)
        self.assertAlmostEqual(copy_end, .55 / .7)
        self.assertAlmostEqual(cpu_end, .55 / .7 + (.55 - .6 * .55 / .7))

    def test_a_second_copy_outlasts_the_cpu_job_and_is_exposed(self):
        m = model()
        cpu_end, copy_end = m.overlap_ends(1, 2)
        self.assertAlmostEqual(cpu_end, .55 / .6)
        self.assertAlmostEqual(copy_end, .55 / .6 + (1.1 - .7 * .55 / .6))
        self.assertGreater(copy_end, cpu_end + .1)
        self.assertGreater(m.cycle_ms(1, 0, 2, 2), .1 + cpu_end + .1 + .28)

    def test_without_cpu_work_a_copy_runs_at_full_speed(self):
        self.assertEqual(model().overlap_ends(0, 1), (0.0, .55))

    def test_a_zero_copy_lead_is_the_common_start(self):
        for c, n in ((1, 1), (1, 2), (3, 3)):
            self.assertEqual(model(copy_lead_ms=0.0).overlap_ends(c, n), model().overlap_ends(c, n))

    def test_a_copy_that_starts_first_runs_alone_until_the_cpu_begins(self):
        # HAGS on: the copy starts .14 before the CPU, does .14 of its .55 alone, then shares for the remaining .41.
        cpu_end, copy_end = model(copy_lead_ms=-.14).overlap_ends(1, 1)
        self.assertAlmostEqual(copy_end, .41 / .7)
        self.assertAlmostEqual(cpu_end, .41 / .7 + (.55 - .6 * .41 / .7))
        self.assertLess(cpu_end, model().overlap_ends(1, 1)[0])

    def test_a_copy_done_before_the_cpu_begins_leaves_the_cpu_alone(self):
        self.assertEqual(model(copy_lead_ms=-.6).overlap_ends(1, 1), (.55, -.6 + .55))

    def test_a_copy_that_starts_later_lets_the_cpu_run_alone_first(self):
        # The CPU does .1 alone; its remaining .45 at rate .6 ends at .75 after the copy starts, before the copy's .55 at .7 would.
        cpu_end, copy_end = model(copy_lead_ms=.1).overlap_ends(1, 1)
        self.assertAlmostEqual(cpu_end, .1 + .45 / .6)
        self.assertAlmostEqual(copy_end, .1 + .45 / .6 + (.55 - .7 * .45 / .6))

    def test_a_hidden_copy_adds_only_its_contention(self):
        m = model(gpu_base=.1)
        cpu_end, copy_end = m.overlap_ends(1, 1)
        self.assertLess(.1 + copy_end, cpu_end + .1)
        self.assertAlmostEqual(m.cycle_ms(1, 0, 3, 1), .1 + cpu_end + .1 + .28)

    def test_primary_work_is_exposed_when_no_lane_hides_it(self):
        m = model()
        self.assertAlmostEqual(m.cycle_ms(0, 0, 4, 1), .1 + 0 + .1 + .28 + (.75 - 0 - .1))

    def test_sequential_regime_adds_primary_work_after_the_lanes(self):
        m = model(regime='query_off', wait_base=.28, gpu_base=0.0, gpu_per_primary=0.0, copy_ms=.55)
        self.assertAlmostEqual(m.cycle_ms(1, 0, 3, 1), .1 + .55 + .1 + .28 + .55)

    def test_finish_and_level_add_to_a_pass_and_finish_can_hide_primary_work(self):
        m = model(finish={'0': .02, '1': .05}, level=.01, gpu_base=.1)
        self.assertAlmostEqual(m.cycle_ms(1, 0, 3, 1), .1 + m.cpu_ms(1, 1) + .02 + .1 + .28 + .01)
        self.assertAlmostEqual(m.cycle_ms(0, 0, 4, 1), .1 + .02 + .1 + .28 + (.65 - .02 - .1) + .01)

    def test_final_and_first_extras_are_per_forward_constants(self):
        m = model(first_extra=.4, final_extra=1.0, prefix_base=2.0, prefix_per_exchange=.5, tail_base=.1, tail_per_exchange=.2, tail_per_lease=.3)
        passes = [dict(forward=1, layer=3, generation=True, cpu=1, helper=0, primary=3, new=0)]
        priced = timing.price(m, passes, issued_end=2, leases=1)
        self.assertAlmostEqual(priced['total_ms'], m.cycle_ms(1, 0, 3, 0) + 1.4 + 2.0 + 1.0 + .1 + .4 + .3)
        self.assertEqual(priced['forwards'], 1)


class Pending(unittest.TestCase):
    def test_exchange_and_lease_excess_are_separate(self):
        m = model(exchange_excess=.2, lease_excess=.5)
        base = m.cycle_ms(1, 0, 3, 0)
        self.assertAlmostEqual(m.cycle_ms(1, 0, 3, 0, True, False), base + .2)
        self.assertAlmostEqual(m.cycle_ms(1, 0, 3, 0, False, True), base + .5)
        self.assertAlmostEqual(m.cycle_ms(1, 0, 3, 0, True, True), base + .7)

    def test_draws_keep_the_mean_including_a_tail(self):
        values = [1.0] * 999 + [1000.0]
        draws = timing._draws(values, 100)
        self.assertEqual(len(draws), 100)
        self.assertAlmostEqual(sum(draws) / 100, sum(values) / 1000)


class FixtureEvents(unittest.TestCase):
    def test_reads_each_header_version(self):
        layers = 'L 3 1 2 3 4 1 2 3 5\n'
        with tempfile.TemporaryDirectory() as directory:
            for version, extra in ((1, ''), (2, ' 1'), (3, ' 0 1 0'), (4, ' 1 0 0 1')):
                path = pathlib.Path(directory) / f'{version}.replay'
                path.write_text(f'MM3_REPLAY {version} 8 4 1.3 0{extra}\nF 1 0 0\n{layers}F 1 1 1\n{layers}END\n')
                events = work.events_from_fixture(path)
                routes = [e for e in events if e['type'] == 'routes']
                self.assertEqual([r['generation'] for r in routes], [False, True])
                self.assertEqual(routes[1]['executed'], [1, 2, 3, 5])
                self.assertEqual(routes[1]['positions'], [1])
                self.assertEqual(sum(e['type'] == 'generation_begin' for e in events), 1)


class Decisions(unittest.TestCase):
    def test_predicted_classes(self):
        self.assertEqual(model_tool.classify_predicted(-.03), 'better')
        self.assertEqual(model_tool.classify_predicted(.03), 'worse')
        self.assertEqual(model_tool.classify_predicted(.01), 'tie')

    def test_observed_class_needs_its_interval(self):
        self.assertEqual(model_tool.classify_observed(dict(mean=-.05, low=-.07, high=-.03)), 'better')
        self.assertEqual(model_tool.classify_observed(dict(mean=.05, low=.03, high=.07)), 'worse')
        self.assertEqual(model_tool.classify_observed(dict(mean=0, low=-.01, high=.01)), 'tie')
        self.assertEqual(model_tool.classify_observed(dict(mean=-.05, low=-.11, high=.01)), 'inconclusive')
        self.assertEqual(model_tool.classify_observed(dict(mean=-.01, low=-.05, high=.03)), 'inconclusive')
        self.assertEqual(model_tool.classify_observed(dict(mean=-.05, low=None, high=None)), 'inconclusive')

    def test_pooled_interval_uses_every_cell(self):
        cells = model_tool.pooled_cells(dict(a=[dict(candidate=99, control=100), dict(candidate=101, control=100)],
                                             b=[dict(candidate=110, control=100), dict(candidate=110, control=100)]))
        self.assertEqual(cells['a']['df'], 2)
        self.assertEqual(cells['a']['sd'], cells['b']['sd'])
        self.assertGreater(cells['b']['sd'], 0)

    def cell(self, mean, half=.005):
        return dict(mean=mean, low=mean - half, high=mean + half)

    def test_a_good_round_qualifies(self):
        result = model_tool.judge(dict(a=-.06, b=.05, c=.001), dict(a=self.cell(-.055), b=self.cell(.052), c=self.cell(.0)), [.02, -.03])
        self.assertTrue(result['qualified'])
        self.assertEqual(len(result['confirmed']), 2)

    def test_a_reversal_fails(self):
        result = model_tool.judge(dict(a=-.06, b=.05), dict(a=self.cell(.06), b=self.cell(.05)), [.01])
        self.assertFalse(result['qualified'])
        self.assertFalse(result['checks']['no_reversal'])

    def test_all_tie_predictor_fails_on_discrimination(self):
        result = model_tool.judge(dict(a=.001, b=.0), dict(a=self.cell(.0), b=self.cell(.001)), [.01])
        self.assertFalse(result['checks']['discrimination'])
        self.assertFalse(result['qualified'])

    def test_inconclusive_cells_are_not_agreement(self):
        result = model_tool.judge(dict(a=-.06), dict(a=dict(mean=-.05, low=-.12, high=.02)), [.01])
        self.assertEqual(result['conclusive'], 0)
        self.assertFalse(result['checks']['agreement'])

    def test_large_errors_fail_even_with_the_right_sign(self):
        result = model_tool.judge(dict(a=-.20, b=.20), dict(a=self.cell(-.05), b=self.cell(.05)), [.01])
        self.assertFalse(result['checks']['median_error'])

    def test_a_control_miss_fails(self):
        result = model_tool.judge(dict(a=-.06, b=.05), dict(a=self.cell(-.06), b=self.cell(.05)), [.20, -.20])
        self.assertFalse(result['checks']['control_miss'])


class PassBalance(unittest.TestCase):
    def row(self, **values):
        from types import SimpleNamespace
        base = dict(rows=5, new=0, cpu=0, helper=0, cpu_jobs=0, helper_jobs=0, primary_jobs=0, exchanges_pending=0, helper_lease=0, final_layer=False, cycle_ms=3.0,
                    cpu_end=None, helper_published=None, lane_begin=0.1, lane_end=1.0, join_wait_next_ms=0.4)
        return SimpleNamespace(**{**base, **values})

    def test_a_pass_is_judged_from_its_own_origin(self):
        import native_pass_balance as balance
        # Duration-wise the helper is longer, but it started later only relative to the origin: completion decides.
        r = self.row(cpu_end=2.1, helper_published=0.8)
        out = balance.classify(r, {5: 0.4})
        self.assertEqual(out['last'], 'cpu')
        self.assertAlmostEqual(out['gap'], 1.3)

    def test_lanes_within_the_band_finish_together(self):
        import native_pass_balance as balance
        self.assertEqual(balance.classify(self.row(cpu_end=1.0, helper_published=1.02), {})['last'], 'together')

    def test_a_wait_above_the_settled_level_shows_primary_outlasting_the_lanes(self):
        import native_pass_balance as balance
        self.assertTrue(balance.classify(self.row(cpu_end=1.0, join_wait_next_ms=0.8), {5: 0.4})['exposed'])
        self.assertFalse(balance.classify(self.row(cpu_end=1.0, join_wait_next_ms=0.45), {5: 0.4})['exposed'])

    def test_division_finds_the_pass_that_could_finish_sooner(self):
        import native_pass_balance as balance
        model = grouped.GroupedTiming(prepare={}, finish={}, resume={'5': 0.1}, wait={'5': 0.3}, cpu_base={'5': 0.05}, cpu_per_job=0.55, cpu_per_column=0.0, cpu_share=0.61, copy_share=0.7, copy_ms=0.553,
                                      helper_base={'5': 0.2}, helper_residual=[0.0], gpu_base={'5': 0.4})
        over = balance.best_division(self.row(cpu=1, cpu_jobs=1, new=4, primary_jobs=4), model, 4)
        self.assertLess(over['best_uploads'], 4)
        self.assertGreater(over['saved'], 0.1)
        held = balance.best_division(self.row(cpu=6, cpu_jobs=6, new=4, primary_jobs=4), model, 4)
        self.assertEqual(held['best_uploads'], 4)
        self.assertGreater(held['saved_wider'], 0.0)
        self.assertEqual(held['saved'], 0.0)


if __name__ == '__main__':
    unittest.main()
