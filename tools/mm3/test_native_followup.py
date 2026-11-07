#!/usr/bin/env python3
"""Rules of the bounded follow-up, checked on constructed numbers before any run exists."""
import unittest

import native_followup_analysis as analysis
import native_passes as passes


def row(cpu=0, helper=0, new=0, ordinal=0, layer=3, lane_begin=0.1, cpu_end=None, enqueue=None, record_end=None, ready=None, published=None,
        collect_end=None, end=1.0, next_pre=None, cycle=2.0):
    return passes.Pass(forward=ordinal + 100, ordinal=ordinal, layer=layer, index=layer - 3, position=0, final_layer=next_pre is None,
                       cpu=cpu, helper=helper, resident=4 - cpu - helper - new, ready=0, new=new, boundary=.01, plan=.02, helper_submit=None,
                       primary_submit=lane_begin, cpu_end=cpu_end, collect_end=collect_end, end=end, helper_enqueue=enqueue, helper_worker_start=None,
                       helper_record_end=record_end, helper_ready=ready, helper_published=published, next_pre_end=next_pre, cycle_ms=cycle)


class EffectRule(unittest.TestCase):
    def test_interval_inside_band_is_bounded(self):
        rows = [dict(a=100, b=100 + d) for d in (0.1, -0.1, 0.2, -0.2, 0.1, -0.1, 0.0, 0.0)]
        self.assertEqual(analysis.block_effect(rows, 'b', 'a')['classification'], 'bounded')

    def test_interval_wholly_outside_band_is_material(self):
        rows = [dict(a=100, b=110 + d) for d in (0.1, -0.1, 0.2, -0.2)]
        self.assertEqual(analysis.block_effect(rows, 'b', 'a')['classification'], 'material')

    def test_a_broad_interval_is_not_evidence_of_no_effect(self):
        rows = [dict(a=100, b=100 + d) for d in (9, -9, 8, -8, 7, -7)]
        self.assertEqual(analysis.block_effect(rows, 'b', 'a')['classification'], 'unresolved')

    def test_single_block_never_resolves(self):
        self.assertEqual(analysis.block_effect([dict(a=100, b=100)], 'b', 'a')['classification'], 'unresolved')


class Resolution(unittest.TestCase):
    def test_block_effects_do_not_count_as_noise(self):
        rows = [dict(x=100 * f, y=110 * f) for f in (1, 2, 3, 4)]
        self.assertAlmostEqual(analysis.two_way_residual_sd(rows, ['x', 'y'])['residual_sd'], 0, places=9)

    def test_more_noise_needs_more_pairs(self):
        self.assertLess(analysis.pairs_needed(.01, .01), analysis.pairs_needed(.03, .01))
        self.assertIsNone(analysis.pairs_needed(None, .01))


class QueryRule(unittest.TestCase):
    def test_gain_goes_on_to_complete_requests(self):
        effect = dict(mean=-.05, low=-.07, high=-.03)
        self.assertEqual(analysis.query_decision(effect, None)['decision'], 'continue_to_requests')

    def test_useful_needs_complete_requests(self):
        effect = dict(mean=-.05, low=-.07, high=-.03)
        self.assertEqual(analysis.query_decision(effect, dict(pooled=-.03, per_prompt=dict(a=-.05, b=-.01)))['decision'], 'useful')
        self.assertEqual(analysis.query_decision(effect, dict(pooled=-.03, per_prompt=dict(a=-.07, b=.01)))['decision'], 'inconclusive')
        self.assertEqual(analysis.query_decision(effect, dict(pooled=-.01, per_prompt=dict(a=-.01, b=-.01)))['decision'], 'inconclusive')

    def test_no_gain_is_negative(self):
        self.assertEqual(analysis.query_decision(dict(mean=0, low=-.01, high=.01), None)['decision'], 'negative')
        self.assertEqual(analysis.query_decision(dict(mean=.05, low=.01, high=.09), None)['decision'], 'negative')

    def test_uncertain_stays_off(self):
        self.assertEqual(analysis.query_decision(dict(mean=-.03, low=-.08, high=.02), None)['decision'], 'inconclusive')


class JointTimeline(unittest.TestCase):
    """The user's hypothetical executions: equal durations, different completion."""

    def test_maximum_is_taken_per_pass_from_a_common_origin(self):
        a = row(cpu=1, helper=1, lane_begin=0, cpu_end=4, enqueue=0, published=1, collect_end=4)
        b = row(cpu=1, helper=1, lane_begin=0, cpu_end=1, enqueue=0, published=4, collect_end=4, ordinal=1)
        summary = passes.mean_of_max_vs_max_of_means([a, b])
        self.assertEqual(summary['max_of_means'], 2.5)
        self.assertEqual(summary['mean_of_max'], 4)

    def test_shortening_the_lane_that_is_not_last_saves_nothing(self):
        a = row(cpu=1, helper=1, lane_begin=0, cpu_end=1, enqueue=0, published=4, collect_end=4)
        self.assertAlmostEqual(passes.lanes_saved(a, passes.Virtual(cpu_factor=.5)), 0)
        self.assertAlmostEqual(passes.lanes_saved(a, passes.Virtual(helper_factor=.5)), 2)

    def test_shortening_one_lane_can_expose_the_other(self):
        a = row(cpu=1, helper=1, lane_begin=0, cpu_end=3, enqueue=0, published=4, collect_end=4)
        self.assertAlmostEqual(passes.lanes_saved(a, passes.Virtual(helper_factor=.5)), 1)

    def test_overlap_opportunity_is_limited_by_the_pass_lane(self):
        short = row(new=1, lane_begin=0, cpu=1, cpu_end=.2, collect_end=None)
        long = row(new=1, lane_begin=0, cpu=1, cpu_end=5, collect_end=None, ordinal=1)
        result = passes.overlap_opportunity([short, long], copy_ms=.5, expert_ms=.1)
        self.assertAlmostEqual(result['upper_bound_ms'], .2 + .6)
        self.assertAlmostEqual(result['copies_only_ms'], .2 + .5)

    def test_paired_passes_require_identical_work(self):
        a = row(cpu=1, new=1, next_pre=1.1, cycle=2.0)
        same = row(cpu=1, new=1, next_pre=1.1, cycle=1.5)
        different = row(cpu=2, new=1, next_pre=1.1, cycle=1.5)
        self.assertEqual(analysis.paired_passes([a], [same])['same_work_windows'], 1)
        self.assertAlmostEqual(analysis.paired_passes([a], [same])['by_new_copies']['1']['cycle']['mean'], -.5)
        self.assertEqual(analysis.paired_passes([a], [different])['same_work_windows'], 0)


class ClockLink(unittest.TestCase):
    def test_explained_variance(self):
        self.assertAlmostEqual(analysis.explained_variance([1, 1, 2, 2], ['a', 'a', 'b', 'b']), 1)
        self.assertAlmostEqual(analysis.explained_variance([1, 2, 1, 2], ['a', 'a', 'b', 'b']), 0)
        self.assertIsNone(analysis.explained_variance([1, 1], ['a', 'b']))


class BackingStep(unittest.TestCase):
    def step(self, action, enabled):
        import pathlib, tempfile, types
        from unittest import mock
        import native_followup as followup
        replies = {action: dict(engine_pid=7, pre_allocation_windows_physical=9), 'inspect': dict(primary_backing_enabled=enabled, ownership_complete=True, primary_backing_valid_records=437)}
        with tempfile.TemporaryDirectory() as d:
            fake = types.SimpleNamespace(base='x', a=types.SimpleNamespace(timeout=1), directory=pathlib.Path(d))
            with mock.patch.object(followup, 'post', lambda base, endpoint, body, timeout: replies[body['action']]):
                return followup.FollowUp.run_backing(fake, dict(step='s', action=action))

    def test_confirms_the_state_it_asked_for(self):
        self.assertEqual(self.step('backing-on', True)['errors'], [])
        self.assertEqual(self.step('backing-off', False)['errors'], [])

    def test_reports_a_state_that_did_not_change(self):
        self.assertTrue(self.step('backing-on', False)['errors'])
        self.assertTrue(self.step('backing-off', True)['errors'])


class FinalComparison(unittest.TestCase):
    def rows(self, new_rates):
        rows = []
        for i, gain in enumerate(new_rates):
            for arm, phase, rate in (('old', f'old{i}', 15.0), ('new', f'new{i}', 15.0 * (1 + gain))):
                for prompt, factor in (('code', 1.0), ('prose', .9)):
                    rows.append(dict(phase=phase, arm=arm, budget=128, prompt=prompt, emitted=128, generation_ms=128000 / (rate * factor)))
        return rows, [dict(old=f'old{i}', new=f'new{i}') for i in range(len(new_rates))]

    def test_rates_pool_tokens_over_time(self):
        import native_final_comparison as final
        self.assertAlmostEqual(final.pooled([dict(emitted=100, generation_ms=1000), dict(emitted=100, generation_ms=3000)])['tokens_per_second'], 50)

    def test_consistent_gain_is_confirmed(self):
        import native_final_comparison as final
        rows, pairs = self.rows([.06, .07, .065])
        result = final.evaluate(rows, pairs)
        self.assertTrue(result['budgets'][128]['confirmed'])
        self.assertEqual(result['regressions'], [])

    def test_a_losing_load_stays_listed_and_blocks_confirmation(self):
        import native_final_comparison as final
        rows, pairs = self.rows([.10, -.02, .05])
        result = final.evaluate(rows, pairs)
        self.assertFalse(result['budgets'][128]['confirmed'])
        self.assertEqual(len(result['regressions']), 2)

    def test_a_missing_request_is_an_error_not_a_gap(self):
        import native_final_comparison as final
        with self.assertRaises(ValueError):
            final.collect(dict(study=dict(requests=[dict(step='a')])), dict(steps=[]))


class ArmScreen(unittest.TestCase):
    def rows(self, prose, code):
        return [dict(prompt=p, arms=dict(control=dict(rate=100.0), candidate=dict(rate=100.0 * (1 + v))))
                for p, values in (('prose', prose), ('code', code)) for v in values]

    def test_consistent_gain_qualifies(self):
        import native_arm_screen as arm_screen
        self.assertTrue(arm_screen.effects(self.rows([-.05, -.04, -.06, -.05], [-.03, -.04, -.03, -.02]), 'control')['candidate']['qualifies'])

    def test_one_prompt_slower_does_not(self):
        import native_arm_screen as arm_screen
        self.assertFalse(arm_screen.effects(self.rows([-.10, -.11, -.09, -.10], [.01, .01, .02, .01]), 'control')['candidate']['qualifies'])

    def test_gain_below_the_band_does_not(self):
        import native_arm_screen as arm_screen
        self.assertFalse(arm_screen.effects(self.rows([-.01, -.012, -.011, -.01], [-.012, -.01, -.011, -.01]), 'control')['candidate']['qualifies'])


if __name__ == '__main__':
    unittest.main()
