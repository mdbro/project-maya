#!/usr/bin/env python3
"""Decision rules and accounting for the shared-model effect screen, on synthetic records."""
import pathlib
import unittest

import native_light as light
import native_shared_analysis as analysis


def layer(forward, number, begin, service, executed=(1, 2, 3, 4), join=100, end=300, **fields):
    row = dict(type='layer', forward=forward, layer=number, t_ns=begin, rows=1, k=4, pre_end=50, join_end=join, boundary_end=join + 10,
               plan_end=join + 20, helper_submit_end=0, primary_submit_end=join + 30, cpu_end=0, collect_end=0, end=end,
               free_slots=4, reserved_slots=0, foreground_slots=0, exchanges_pending=0, helper_lease=0, stage_max=4,
               stage_selected=0, total_slots=8, service=list(service), original=list(executed), executed=list(executed))
    row.update(fields)
    return row


def capture_of(*records):
    capture = light.LightCapture(pathlib.Path('synthetic'), 'sha', {})
    capture.records.extend(records)
    return capture


class Classification(unittest.TestCase):
    def test_material_requires_every_pair_past_band_with_one_sign(self):
        self.assertEqual(analysis.classify_paired([0.03, 0.05]), 'material')
        self.assertEqual(analysis.classify_paired([-0.03, -0.021]), 'material')
        self.assertEqual(analysis.classify_paired([0.03, 0.01]), 'unresolved')
        self.assertEqual(analysis.classify_paired([0.05, -0.05]), 'unresolved')

    def test_bounded_requires_low_control_drift(self):
        self.assertEqual(analysis.classify_paired([0.005, -0.01], 0.012), 'bounded')
        self.assertEqual(analysis.classify_paired([0.005, -0.01], 0.05), 'unresolved')
        self.assertEqual(analysis.classify_paired([0.019, 0.021], 0.0), 'unresolved')

    def test_two_noisy_pairs_cannot_declare_absence(self):
        self.assertEqual(analysis.classify_paired([0.04, -0.04], 0.0), 'unresolved')
        self.assertEqual(analysis.classify_paired([]), 'unresolved')

    def test_observational_rules(self):
        replicate = lambda value, se=0.001, exposed=100, coverage=1.0: dict(value=value, se=se, exposed=exposed, coverage=coverage)
        self.assertEqual(analysis.classify_observational([replicate(0.03), replicate(0.04)]), 'material')
        self.assertEqual(analysis.classify_observational([replicate(0.005), replicate(-0.002)]), 'bounded')
        self.assertEqual(analysis.classify_observational([replicate(0.005)]), 'unresolved')       # one replicate
        self.assertEqual(analysis.classify_observational([replicate(0.005, coverage=0.1), replicate(0.001, coverage=0.1)]), 'unresolved')
        self.assertEqual(analysis.classify_observational([replicate(0.015, se=0.01), replicate(0.001)]), 'unresolved')
        self.assertEqual(analysis.classify_observational([replicate(0.03), replicate(-0.04)]), 'unresolved')
        self.assertEqual(analysis.classify_observational([replicate(0.03, exposed=3)]), 'unresolved')  # no real coverage


class Accounting(unittest.TestCase):
    def forward(self):
        service = ['cpu', 'helper', 'primary_resident', 'staged_new']
        # Two layers. Forward 1 begins at 1000; joins at 1100 and 1500; synchronized at 1900; ends at 2000.
        return capture_of(
            layer(1, 3, 1000, service),
            layer(1, 4, 1450, service, join=50, end=200, pre_end=10),
            dict(type='forward', forward=1, t_ns=1000, position=100, rows=1, generation=True, prepare_end=10, body_end=800,
                 sync_end=900, end=1000, counters={}))

    def test_cycles_partition_the_forward_exactly(self):
        cycles, budgets = analysis.windows(self.forward())
        self.assertEqual(len(cycles), 2)
        self.assertAlmostEqual(cycles[0]['cycle_ms'], 400 * 1e-6)
        self.assertAlmostEqual(cycles[1]['cycle_ms'], 400 * 1e-6)
        budget = budgets[0]
        self.assertAlmostEqual(budget['prefix_ms'], 100 * 1e-6)
        self.assertAlmostEqual(budget['policy_tail_ms'], 100 * 1e-6)
        self.assertAlmostEqual(budget['prefix_ms'] + budget['cycles_ms'] + budget['policy_tail_ms'], budget['forward_ms'])
        self.assertEqual(cycles[0]['key'], (1, 1, 1, 0, 1))
        self.assertTrue(cycles[1]['final_layer'])
        self.assertIsNone(cycles[1]['join_wait_next_ms'])
        self.assertEqual(cycles[0]['join_wait_next_ms'], 40 * 1e-6)

    def test_prefill_forwards_are_excluded(self):
        capture = self.forward()
        capture.records.append(dict(type='forward', forward=0, t_ns=1, position=0, rows=16, generation=False,
                                    prepare_end=1, body_end=2, sync_end=3, end=4, counters={}))
        capture.records.append(layer(0, 3, 1, ['cpu'] * 4))
        cycles, budgets = analysis.windows(capture)
        self.assertEqual({w['forward'] for w in cycles}, {1})
        self.assertEqual(len(budgets), 1)


class Cold(unittest.TestCase):
    def test_cold_pick_counts_by_recency_on_that_device(self):
        records = []
        service = ['cpu', 'helper', 'primary_resident', 'primary_resident']
        for forward in range(1, 13):
            executed = (1, 2, 3, 4) if forward in (1, 2, 12) else (5, 6, 7, 8)
            records.append(layer(forward, 3, forward * 10000, service, executed=executed))
            records.append(dict(type='forward', forward=forward, t_ns=forward * 10000, position=forward, rows=1, generation=True,
                                prepare_end=1, body_end=2, sync_end=400, end=500, counters={}))
        cycles, _ = analysis.windows(capture_of(*records))
        by_forward = {w['forward']: w for w in cycles}
        self.assertEqual(by_forward[1]['cold_cpu'], 1)     # never seen
        self.assertEqual(by_forward[1]['cold_primary'], 2)
        self.assertEqual(by_forward[2]['cold_cpu'], 0)     # ran last forward
        self.assertEqual(by_forward[12]['cold_cpu'], 1)    # last ran ten forwards ago
        self.assertEqual(by_forward[3]['cold_cpu'], 1)
        self.assertEqual(by_forward[4]['cold_cpu'], 0)


class Excess(unittest.TestCase):
    def window(self, forward, key, cycle, exposed):
        return dict(forward=forward, key=key, cycle_ms=cycle, exposed=exposed)

    def test_matched_excess_ignores_different_work(self):
        windows = [self.window(f, (4, 0, 0, 0, 0), 10.0, False) for f in range(10)]
        windows += [self.window(f, (0, 0, 4, 0, 0), 2.0, False) for f in range(10)]
        # Exposed windows run 3 ms long within their own work; their key is costlier than the other.
        windows += [self.window(f, (4, 0, 0, 0, 0), 13.0, True) for f in range(5)]
        result = analysis.stratified_excess(windows, lambda w: 'x' if w['exposed'] else None, 1000.0)
        self.assertAlmostEqual(result['excess_ms'], 15.0)
        self.assertAlmostEqual(result['value'], 0.015)
        self.assertEqual(result['exposed'], 5)
        self.assertEqual(result['coverage'], 1.0)

    def test_unmatched_exposure_lowers_coverage(self):
        windows = [self.window(f, (4, 0, 0, 0, 0), 10.0, False) for f in range(10)]
        windows += [self.window(f, (0, 4, 0, 0, 0), 99.0, True) for f in range(5)]
        windows += [self.window(f, (4, 0, 0, 0, 0), 10.0, True) for f in range(5)]
        result = analysis.stratified_excess(windows, lambda w: 'x' if w['exposed'] else None, 1000.0)
        self.assertEqual(result['matched'], 5)
        self.assertEqual(result['coverage'], 0.5)
        self.assertAlmostEqual(result['excess_ms'], 0.0)


class Fixture(unittest.TestCase):
    def ledger(self, times, errors=()):
        return dict(steps=[dict(step=n, summary=dict(generation_ms=t, errors=list(errors))) for n, t in times.items()])

    def test_abba_pairs_compare_like_orders(self):
        result = analysis.fixture_effect(self.ledger(dict(a1=1000.0, b1=1100.0, b2=1090.0, a2=1010.0)), 'a1', 'b1', 'b2', 'a2')
        self.assertAlmostEqual(result['differences'][0], 0.1)
        self.assertAlmostEqual(result['differences'][1], 80 / 1010)
        self.assertAlmostEqual(result['control_drift'], 0.01)
        self.assertEqual(result['classification'], 'material')

    def test_failed_arm_is_never_classified(self):
        result = analysis.fixture_effect(self.ledger(dict(a1=1000.0, b1=1100.0, b2=1090.0, a2=1010.0), ['bad']), 'a1', 'b1', 'b2', 'a2')
        self.assertEqual(result['classification'], 'unresolved')


if __name__ == '__main__':
    unittest.main()
