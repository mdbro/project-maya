#!/usr/bin/env python3
"""pass_model.py, the Python twin of src/mm3/mm3_pass_model.hpp: the host-memory flows and the lanes' terms on
cases checked by hand (tests/mm3_pass_model_test.cpp checks the C++ against fit_pass_model.py's golden cases).
Standard library only.

    python3 -m unittest tools/mm3/pass_model/test_pass_model.py"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass_model as pm  # noqa: E402


class HostFlows(unittest.TestCase):
    def test_equal_flows_share_the_ceiling(self):
        self.assertEqual(pm.host_flows([(0.0, 30.0, 60.0, 60.0), (0.0, 30.0, 60.0, 60.0)], 80.0), [0.75, 0.75])

    def test_no_work_ends_at_begin(self):
        self.assertEqual(pm.host_flows([(0.04, 0.0, 57.0, 57.0), (0.0, 10.0, 50.0, 50.0)], 80.0), [0.04, 0.2])

    def test_below_the_ceiling_each_runs_alone(self):
        e = pm.host_flows([(0.0, 10.0, 50.0, 50.0), (0.0, 10.0, 20.0, 20.0)], 80.0)
        self.assertAlmostEqual(e[0], 0.2)
        self.assertAlmostEqual(e[1], 0.5)

    def test_the_one_left_speeds_up(self):
        e = pm.host_flows([(0.0, 30.0, 60.0, 60.0), (0.0, 60.0, 60.0, 60.0)], 60.0)
        self.assertAlmostEqual(e[0], 1.0)
        self.assertAlmostEqual(e[1], 1.5)

    def test_late_start(self):
        e = pm.host_flows([(0.5, 10.0, 50.0, 50.0), (0.0, 50.0, 50.0, 50.0)], 50.0)
        self.assertAlmostEqual(e[0], 0.9)
        self.assertAlmostEqual(e[1], 1.2)


class Lanes(unittest.TestCase):
    def rates(self):
        r = pm.Rates()
        r.gpu_start = [0.03] * pm.ROWS
        r.gpu_start_ready = [0.16] * pm.ROWS
        r.gpu_base = [0.05] * pm.ROWS
        r.publish = [0.02] * pm.ROWS
        return r

    def test_gpu_only(self):
        r = self.rates()
        f = pm.predict(pm.Load(rows=2, gpu_experts=2, gpu_picks=3), r)
        mb = r.record_bytes * 1e-6
        self.assertAlmostEqual(f.gpu, 0.03 + 0.05 + 2 * mb / r.gpu_gbps + 3 * r.gpu_pick)
        self.assertEqual(f.passed, f.gpu)
        self.assertIsNone(f.cpu)
        self.assertIsNone(f.host)

    def test_ready_start_at_one_row(self):
        r = self.rates()
        a = pm.predict(pm.Load(rows=1, gpu_experts=1, gpu_picks=1), r)
        b = pm.predict(pm.Load(rows=1, gpu_experts=1, gpu_picks=1, ready=1), r)
        self.assertAlmostEqual(b.gpu - a.gpu, 0.13)

    def test_exchanges_delay_window_copies_only(self):
        r = self.rates()
        r.exchange_up, r.exchange_down = 1.0, 0.0
        w = pm.predict(pm.Load(rows=2, copies=1, exchanges=1, gpu_experts=1, gpu_picks=1), r)
        w0 = pm.predict(pm.Load(rows=2, copies=1, gpu_experts=1, gpu_picks=1), r)
        mb = r.record_bytes * 1e-6
        self.assertAlmostEqual(w.copies - w0.copies, mb / r.copy_gbps[0])
        o = pm.predict(pm.Load(rows=1, copies=1, exchanges=1, gpu_experts=1, gpu_picks=1), r)
        o0 = pm.predict(pm.Load(rows=1, copies=1, gpu_experts=1, gpu_picks=1), r)
        self.assertAlmostEqual(o.copies, o0.copies)

    def test_early_copy_slows_the_cpu_lane_at_one_row(self):
        r = self.rates()
        r.early_draw = 40.0
        a = pm.predict(pm.Load(rows=1, cpu_experts=1, cpu_columns=1, gpu_experts=1, gpu_picks=1), r)
        b = pm.predict(pm.Load(rows=1, cpu_experts=1, cpu_columns=1, early=1, gpu_experts=1, gpu_picks=1), r)
        w = pm.predict(pm.Load(rows=2, cpu_experts=1, cpu_columns=1, early=1, gpu_experts=1, gpu_picks=1), r)
        self.assertGreater(b.cpu, a.cpu)
        self.assertAlmostEqual(w.cpu - pm._row(r.cpu_start, 2), a.cpu - pm._row(r.cpu_start, 1))

    def test_rounds(self):
        r = self.rates()
        four = pm.predict(pm.Load(rows=2, gpu_experts=5, gpu_picks=5, gpu_rounds_of=4), r)
        one = pm.predict(pm.Load(rows=2, gpu_experts=5, gpu_picks=5, gpu_rounds_of=32), r)
        self.assertAlmostEqual(four.gpu - one.gpu, r.gpu_round)

    def test_pass_is_later_of_gpu_and_host(self):
        r = self.rates()
        f = pm.predict(pm.Load(rows=2, gpu_experts=1, gpu_picks=1, helper_experts=8, helper_columns=16), r)
        self.assertAlmostEqual(f.passed, max(f.gpu, f.helper + 0.02))

    def test_split_waits_after_ready_compute(self):
        r = self.rates()
        r.record_bytes = 1e6
        r.copy_gbps = [1, 1]
        r.gpu_gbps = 10
        r.gpu_pick = 0
        load = pm.Load(rows=4, gpu_experts=3, gpu_picks=3, copies=1, late_experts=1, late_picks=1,
                       overlap=True, gpu_rounds_of=32)
        # Ready: .03 + .05 + .2 = .28 ms. Upload: 1 ms. Late: .05 + .1 = .15 ms.
        self.assertAlmostEqual(pm.predict(load, r).gpu, 1.15)
        load.overlap = False
        self.assertAlmostEqual(pm.predict(load, r).gpu, 1.35)

    def test_faster_copy_does_not_make_prior_schedule_slower(self):
        r = self.rates()
        load = pm.Load(rows=4, cpu_experts=2, cpu_columns=4, gpu_experts=6, gpu_picks=9, copies=2,
                       late_experts=2, late_picks=3, overlap=True, gpu_rounds_of=32)
        previous = pm.predict(load, r).passed
        for bandwidth in (60, 75, 90):
            r.copy_gbps[0] = bandwidth
            current = pm.predict(load, r).passed
            # A faster competing link can slow the CPU under shared-memory contention. The total finish must retain
            # the former rate as a feasible choice; this fixture isolates a non-CPU-critical pass below the ceiling.
            r.host_gbps = 200
            current = pm.predict(load, r).passed
            if bandwidth == 60:
                previous = current
            self.assertLessEqual(current, previous + 1e-12)
            previous = current

    def test_signed_copy_head_start_changes_start_not_rate(self):
        r = self.rates()
        r.record_bytes = 1e6
        r.copy_gbps = [1, 1]
        load = pm.Load(rows=2, copies=1, gpu_experts=1, gpu_picks=1)
        normal = pm.predict(load, r)
        load.copy_begin = -.2
        early = pm.predict(load, r)
        self.assertAlmostEqual(normal.copies-early.copies, .2)
        self.assertEqual(r.copy_gbps, [1,1])

    def test_explicit_queued_exchange_load_is_not_free_at_one_row(self):
        r = self.rates()
        load = pm.Load(rows=1, copies=1, gpu_experts=1, gpu_picks=1)
        before = pm.predict(load, r)
        load.exchange_up_records=.125
        self.assertGreater(pm.predict(load,r).copies,before.copies)


if __name__ == '__main__':
    unittest.main()
