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

    def test_without_cpu_work_a_leading_copy_keeps_its_lead_at_full_speed(self):
        self.assertEqual(model(copy_lead_ms=-.14).overlap_ends(0, 1), (0.0, -.14 + .55))

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

    def test_one_measured_share_gives_the_other_from_the_stretch(self):
        cpu, copy = grouped.shares_from_stretch(0.2457, 0.553)
        self.assertEqual(cpu, 0.61)
        self.assertAlmostEqual(copy, 0.878, places=3)
        cpu, copy = grouped.shares_from_stretch(0.2457, 0.553, copy_share=0.66)
        self.assertEqual(copy, 0.66)
        self.assertAlmostEqual(cpu, 1 - 0.2457 * 0.66 / 0.553, places=6)
        # the stretch is the copy's duration beside the CPU times the CPU's lost share, both ways
        self.assertAlmostEqual((0.553 / copy) * (1 - cpu), 0.2457, places=6)
        with self.assertRaises(ValueError):
            grouped.shares_from_stretch(0.5, 0.553, copy_share=1.5)

    def test_uploads_with_a_head_start_share_less_of_the_cpu_lane(self):
        base = dict(cpu_base={'1': 0.0, '8': 0.0}, cpu_per_job=0.5, cpu_share=0.7, copy_share=0.66, copy_ms=0.553)
        plain = grouped.GroupedTiming(**base)
        ahead = grouped.GroupedTiming(**base, copy_head={'1': 0.3, '8': 0.5})
        # no head start: the CPU (1.0 ms of work) finishes first at 1/0.7, the copies' rest runs alone after it
        cpu, copy = plain.overlap_ends(8, 2, 2, 2)
        self.assertAlmostEqual(cpu, 1.0 / 0.7)
        self.assertAlmostEqual(copy, 1.0 / 0.7 + (1.106 - 0.66 / 0.7))
        # 0.5 ms ahead: 0.606 ms of copy work overlaps; the copies finish first and the CPU's rest runs alone
        cpu, copy = ahead.overlap_ends(8, 2, 2, 2)
        self.assertAlmostEqual(copy, 0.606 / 0.66)
        self.assertAlmostEqual(cpu, 0.606 / 0.66 + (1.0 - 0.7 * 0.606 / 0.66))
        # a copy shorter than the head start is done before the CPU lane begins
        cpu, copy = grouped.GroupedTiming(**base, copy_head={'8': 0.6}).overlap_ends(8, 2, 2, 1)
        self.assertAlmostEqual(cpu, 1.0)
        self.assertAlmostEqual(copy, 0.553 - 0.6)
        # one-row passes do not overlap, so there is no head start
        self.assertEqual(ahead.head(1), 0.0)
        self.assertEqual(ahead.overlap_ends(1, 2, 2, 1), plain.overlap_ends(1, 2, 2, 1))

    def test_the_measured_5090_path_is_the_later_of_resident_work_and_uploads(self):
        base = dict(cpu_base={'1': 0.0, '8': 0.0}, cpu_per_job=0.5, cpu_share=0.7, copy_share=0.66, copy_ms=0.553,
                    gpu_base={'1': 0.0, '8': 0.59})
        additive = grouped.GroupedTiming(**base)
        measured = grouped.GroupedTiming(**base, gpu_resident={'8': -0.08}, gpu_resident_per_job=0.05, gpu_tail={'8': 0.49})
        copy_end = additive.overlap_ends(8, 2, 2, 2)[1]
        self.assertAlmostEqual(additive.primary_ms(8, 2, 2, 6, 2), 0.59 + copy_end)
        self.assertAlmostEqual(measured.primary_ms(8, 2, 2, 6, 2), copy_end + 0.49)          # uploads bind
        self.assertAlmostEqual(measured.primary_ms(8, 2, 2, 6, 0), -0.08 + 0.05 * 6)         # resident work alone
        self.assertAlmostEqual(measured.primary_ms(8, 0, 0, 30, 1), -0.08 + 0.05 * 30)       # many resident jobs bind
        self.assertEqual(measured.primary_ms(1, 2, 2, 6, 1), additive.primary_ms(1, 2, 2, 6, 1))  # one row: unchanged


def order_snapshot(slots: int = 4, layers=(3, 4, 5)) -> dict:
    """Three MoE layers: expert 0 and 12 resident, 1, 10 and 11 the 3090's, 2-9 HOST; uploads cheap, CPU work dear."""
    prices = {k: [0] + [v] * 16 for k, v in (('cpu_extra_ms', 1.0), ('gpu_extra_ms', .01), ('helper_extra_ms', .01), ('gpu_base_ms', .01))}
    prices.update(cpu_base_ms=0, helper_base_ms=0, h2d_ms=.01, d2h_ms=1)
    rows, helper_frames, address = [], [], 1000
    for li in layers:
        host = []
        for e in range(2, 10):
            host.append([e, address])
            address += 1
        rows.append(dict(layer=li, primary=[0, 12], primary_frames=[[0, address], [12, address + 1]], host_frames=host,
                         helper=[1, 10, 11], primary_backing_frames=[]))
        address += 2
        helper_frames += [[li, e, len(helper_frames)] for e in (1, 10, 11)]
    state = dict(row_clock=5, block_rows=1, scores=[], primary_credit=0, primary_credit_row=5, helper_credit=0, helper_credit_row=5,
                 host_spares=[900, 901], helper_buffers=[902, 903])
    return dict(forward=1, record_bytes=1000, dimension=4, cpu_participants=16,
                policy=dict(primary_backing=False, service_saved_ms=1, configured_service_saved_ms=1, configured_h2d_ms=0,
                            configured_d2h_ms=0, interference_ms=0),
                layers=rows, helper_frames=helper_frames, staging=[[-1, -1, 0, 0, 0, 0, 500 + i] for i in range(slots)],
                replay_state=state, batch_staging_calibration=prices)


def one_row_forward(forward: int, picks: dict, early: dict | None = None) -> list[dict]:
    events = [dict(type='boundary', kind='forward_begin', forward=forward, rows=1, generation=True)]
    for li, executed in picks.items():
        route = dict(type='routes', rows=1, layer=li, positions=[forward], generation=True, original=list(executed),
                     executed=list(executed), forward=forward)
        if early and li in early:
            route.update(early[li])
        events.append(route)
    return events + [dict(type='boundary', kind='forward_end', forward=forward, rows=1)]


def order_replay(events, *, early_copy=2, rule=None, slots=4, stage_limit=4):
    from cooperative_replay.capture import Capture
    snap = order_snapshot(slots)
    config = work.PolicyConfig(refill_budget=0, helper_live=False, stage_limit=stage_limit)
    m = work.OrderReplay(snap, config, rule or work.OrderRule(), early_copy=early_copy)
    stream = [dict(type='fixed_workload_begin', diagnostic_primary_query=False), dict(type='generation_begin')] + events + \
        [dict(type='generation_end')]
    m.replay([Capture(pathlib.Path('x'), 'x', 'x', stream, snap, {})], assignments='policy')
    return m


class EarlyCopy(unittest.TestCase):
    """The engine's early_copy (mm3_native_prefetch.inc) in the order replay."""

    def test_the_next_layers_first_host_prediction_is_ready_there_without_an_upload(self):
        # Layer 3 uploads one expert (2); mode 2 still copies layer 4's first predicted HOST expert: 0 is resident, 5 is HOST.
        m = order_replay(one_row_forward(2, {3: [0, 1, 10, 2], 4: [12, 1, 5, 11]}, early={3: dict(early=[0, 5, 6])}))
        l3, l4 = m.layers
        self.assertEqual(l3['staged'], [2])
        self.assertEqual(l3['early_copy'], 4 * 1024 + 5)
        self.assertEqual(l4['staged'], [])
        self.assertIn(5, l4['primary'])
        # Least recently used free frame, after the pass's own upload took frame 0.
        self.assertEqual([f.owner for f in m.slots][:2], [3 * 1024 + 2, 4 * 1024 + 5])

    def test_two_uploads_or_a_pending_exchange_block_it_and_mode_one_needs_none(self):
        two = order_replay(one_row_forward(2, {3: [0, 1, 2, 3], 4: [12, 1, 5, 11]}, early={3: dict(early=[5])}))
        self.assertEqual(sorted(two.layers[0]['staged']), [2, 3])
        self.assertNotIn('early_copy', two.layers[0])
        self.assertEqual(two.layers[1]['staged'], [5])
        one = order_replay(one_row_forward(2, {3: [0, 1, 10, 2], 4: [12, 1, 5, 11]}, early={3: dict(early=[5])}), early_copy=1)
        self.assertNotIn('early_copy', one.layers[0])
        m = work.OrderReplay(order_snapshot(), work.PolicyConfig(refill_budget=0, helper_live=False), early_copy=2)
        m.layers.append(dict(staged=[], done_op=-1))
        m.exchanges.append('an exchange in flight')
        m.issue_early_copy(dict(layer=3, early=[5]))
        self.assertEqual((m.early['blocked'], m.early['issued']), (1, 0))

    def test_uploads_take_frames_in_pick_order_not_the_planners(self):
        # The planner picks 3 before 7 (equal gains, lower id first); the engine's locate copies in pick order: 7 then 3.
        m = order_replay(one_row_forward(2, {3: [0, 1, 7, 3]}), early_copy=0)
        self.assertEqual(sorted(m.layers[0]['staged']), [3, 7])
        self.assertEqual([f.owner for f in m.slots][:2], [3 * 1024 + 7, 3 * 1024 + 3])

    def test_the_next_upload_cannot_take_an_early_frame_still_in_flight(self):
        # Two frames: layer 3 uploads 2 into frame 0 and copies layer 4's expert 9 early into frame 1. At layer 4 the copy
        # is in flight, so its upload of 6 takes frame 0 again, and the early-copied expert keeps frame 1.
        m = order_replay(one_row_forward(2, {3: [0, 1, 10, 2], 4: [12, 1, 10, 6]}, early={3: dict(early=[9])}), slots=2)
        self.assertEqual(m.layers[0]['early_copy'], 4 * 1024 + 9)
        self.assertEqual(m.layers[1]['staged'], [6])
        self.assertEqual([f.owner for f in m.slots], [4 * 1024 + 6, 4 * 1024 + 9])

    def test_an_early_frame_is_busy_until_a_later_join_sees_its_copy_done(self):
        from cooperative_replay.execution import Estimate
        m = work.OrderReplay(order_snapshot(), work.PolicyConfig(refill_budget=0, helper_live=False))
        frame = m.slots[1]
        frame.owner, frame.early = 4 * 1024 + 9, m.op('early_h2d', ())   # seen done at the second join (1.5 joins)
        join = lambda: m.scheduler.until(m.scheduler.submit('layer_pre', Estimate(1, 1, 1), ('p_compute',)))
        self.assertTrue(m.busy(frame) and m.source_busy(frame))
        join()
        m.publish(4)
        self.assertIsNotNone(frame.early)
        join()
        m.publish(5)
        self.assertIsNone(frame.early)
        self.assertFalse(m.busy(frame))

    def test_an_unread_copy_takes_a_frame_named_by_a_later_record_or_a_placeholder(self):
        named = order_replay(one_row_forward(2, {3: [0, 1, 10, 2], 4: [12, 1, 10, 11]},
                                             early={3: dict(early=[work.UNUSED], early_unread={1: 8})}))
        self.assertEqual(named.layers[0]['early_copy'], 4 * 1024 + 8)
        unnamed = order_replay(one_row_forward(2, {3: [0, 1, 10, 2], 4: [12, 1, 10, 11]}, early={3: dict(early=[work.UNUSED])}))
        self.assertEqual(unnamed.layers[0]['early_copy'], 4 * 1024 + 1023)
        self.assertEqual(unnamed.early['unread'], 1)


def light_layer(forward, layer, executed, service, *, free=8, reserved=0, exchanges=0):
    return dict(type='layer', forward=forward, layer=layer, rows=len(executed) // 4, original=list(executed), executed=list(executed),
                service=list(service), free_slots=free, reserved_slots=reserved, total_slots=8, exchanges_pending=exchanges,
                helper_lease=0)


class RecordedEarlyPrediction(unittest.TestCase):
    def test_ready_picks_lead_and_an_unread_frame_adds_unused(self):
        here = light_layer(1, 3, [0, 1, 2, 3], ['primary_resident', 'helper', 'staged_new', 'cpu'])
        after = light_layer(1, 4, [7, 5, 6, 9], ['staged_ready', 'helper', 'staged_ready', 'cpu'], free=5)
        self.assertEqual(work.recorded_early_prediction(here, after), [7, 6, work.UNUSED])  # 8 - 5 free - 2 read = 1 unread
        after = light_layer(1, 4, [7, 5, 6, 9], ['staged_ready', 'helper', 'staged_ready', 'cpu'], free=6)
        self.assertEqual(work.recorded_early_prediction(here, after), [7, 6])

    def test_where_the_engine_could_not_copy_the_proxy_is_the_next_picks(self):
        busy = light_layer(1, 3, [0, 1, 2, 3], ['primary_resident', 'helper', 'staged_new', 'staged_new'])
        after = light_layer(1, 4, [7, 5, 6, 9], ['staged_ready', 'helper', 'cpu', 'cpu'])
        self.assertEqual(work.recorded_early_prediction(busy, after), [7, 5, 6, 9])
        pending = light_layer(1, 3, [0, 1, 2, 3], ['primary_resident', 'helper', 'cpu', 'cpu'], exchanges=1)
        self.assertEqual(work.recorded_early_prediction(pending, after), [7, 5, 6, 9])

    def test_unread_identity_is_the_first_owner_a_frame_shows_for_that_layer(self):
        records = [dict(type='demand_copy', slot=2, displaced=4 * 1024 + 8),
                   dict(type='demand_copy', slot=2, displaced=4 * 1024 + 9),        # later: not the first
                   dict(type='primary_issue', landing=3, incoming=4 * 1024 + 6, reuse=True),
                   dict(type='primary_issue', landing=5, incoming=9 * 1024 + 1, reuse=False),
                   dict(type='demand_copy', slot=6, displaced=7 * 1024 + 3)]       # another layer: unknown
        self.assertEqual(work.unread_identities(records, 0, 4, unused={6}), {2: 8})
        self.assertEqual(work.unread_identities(records, 0, 4, unused=set()), {2: 8, 3: 6})


class RestoredOrder(unittest.TestCase):
    class Capture:
        def __init__(self, published, open_exchanges=()):
            self.records = [dict(type='primary_publish', incoming=a, victim=b) for a, b in published] + \
                [dict(type='map', phase=1, exchanges=[dict(incoming=a, victim=b) for a, b in open_exchanges])]

        def of(self, kind):
            return [r for r in self.records if r['type'] == kind]

    def snapshot(self):
        frames = lambda li, ids: [[e, li * 100 + e] for e in ids]
        return dict(layers=[dict(layer=3, primary=[5, 6, 7], primary_frames=frames(3, [5, 6, 7])),
                            dict(layer=4, primary=[1, 2], primary_frames=frames(4, [1, 2]))])

    def test_the_first_request_keeps_the_origin_order(self):
        self.assertEqual(work.restored_primary_order(self.snapshot(), []), {3: [5, 6, 7], 4: [1, 2]})

    def test_publications_reorder_and_the_restore_refills_frames_in_order(self):
        # (4,9) replaces victim (4,1): the last resident takes the victim's place, the incoming one is appended: 4:[2,9].
        # Open at the end, published by the restore's join: (3,8) replaces (3,6): 3:[5,7,8].
        # Frames in layer order: (3,5) (3,7) (3,8) (4,2) (4,9). The origin's missing (3,6) and (4,1) take the first frames
        # holding unwanted experts, (3,8) and (4,9), and each layer's order follows its frames: 3:[5,7,6], 4:[2,1].
        earlier = [self.Capture([(4 * 1024 + 9, 4 * 1024 + 1)], [(3 * 1024 + 8, 3 * 1024 + 6)])]
        self.assertEqual(work.restored_primary_order(self.snapshot(), earlier), {3: [5, 7, 6], 4: [2, 1]})
        # A resident that left for another layer comes back in the frame order of that layer's rows.
        across = [self.Capture([(3 * 1024 + 8, 4 * 1024 + 1)])]
        self.assertEqual(work.restored_primary_order(self.snapshot(), across), {3: [5, 6, 7], 4: [1, 2]})

    def test_with_primary_order_keeps_frames_with_their_experts(self):
        snap = work.with_primary_order(self.snapshot(), {4: [2, 1]})
        self.assertEqual(snap['layers'][1]['primary'], [2, 1])
        self.assertEqual(snap['layers'][1]['primary_frames'], [[2, 402], [1, 401]])
        with self.assertRaises(ValueError):
            work.with_primary_order(self.snapshot(), {4: [2, 3]})


class CompareUnits(unittest.TestCase):
    def test_a_window_compares_jobs_with_jobs_and_reports_picks_apart(self):
        class Capture:
            records = [dict(type='forward', forward=1, generation=True),
                       light_layer(1, 3, [5, 1, 2, 3, 5, 1, 2, 4], ['cpu', 'helper', 'staged_new', 'primary_resident',
                                                                  'cpu', 'helper', 'staged_ready', 'staged_new'])]

            def of(self, kind):
                return [r for r in self.records if r['type'] == kind]
        observed = work.recorded_work(Capture())[0]
        self.assertEqual((observed['cpu'], observed['helper'], observed['primary'], observed['new']), (1, 1, 3, 2))
        self.assertEqual((observed['cpu_columns'], observed['helper_columns'], observed['primary_columns']), (2, 2, 4))
        replayed = dict(passes=[dict(generation=True, cpu=1, helper=1, primary=3, new=2, cpu_columns=2, helper_columns=2,
                                     primary_columns=4, exchange=False, lease=False)], counts={})
        result = work.compare_work(replayed, Capture())['generation']
        self.assertEqual(result['identical_passes'], 1)
        self.assertEqual(result['predicted']['primary_columns'], result['observed']['primary_columns'])


class Clock(unittest.TestCase):
    def test_a_forward_is_prefix_cycles_and_a_gap_in_milliseconds(self):
        rule = work.todays_rule(speculative=True)
        m = work.OrderReplay(order_snapshot(), work.PolicyConfig(refill_budget=0, helper_live=False), rule)
        m.between_forwards(dict(rows=8))
        self.assertEqual(m.scheduler.now, rule.gap)
        self.assertEqual(m._ticks('target_prefix', 'p_main', 8), rule.prefix_ms[8])
        self.assertEqual(m._ticks('layer_pre', 'p_main', 8), 0.0)        # the first join ends the prefix
        self.assertEqual(m._ticks('layer_pre', 'p_main', 8), rule.cycle[8])
        self.assertEqual(m._ticks('layer_pre', 'p_main', 16), rule.cycle[8])   # the nearest known width
        joins = work.OrderReplay(order_snapshot(), work.PolicyConfig(refill_budget=0, helper_live=False))
        self.assertEqual(joins._ticks('layer_pre', 'p_main', 8), 1.0)


if __name__ == '__main__':
    unittest.main()
