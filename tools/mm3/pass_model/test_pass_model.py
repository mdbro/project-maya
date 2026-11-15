#!/usr/bin/env python3
"""pass_model.py, the Python twin of src/mm3/mm3_pass_model.hpp: the shared host-memory flows and the lanes' terms on
cases checked by hand (tests/mm3_pass_model_test.cpp checks the C++ against fit_pass_model.py's golden cases).
Standard library only.

    python3 -m unittest tools/mm3/pass_model/test_pass_model.py"""
from __future__ import annotations

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pass_model as pm  # noqa: E402


class SharedFlows(unittest.TestCase):
    def test_alone(self):
        self.assertEqual(pm.shared_ends(0.04, 0.5, 0.0, 0.0, 0.6, 0.7), (0.54, 0.0))

    def test_copy_first_then_cpu_alone(self):
        cpu, copy = pm.shared_ends(0.0, 0.5, 0.0, 0.3, 0.5, 0.5)
        self.assertAlmostEqual(copy, 0.6)
        self.assertAlmostEqual(cpu, 0.8)

    def test_head_start(self):
        cpu, copy = pm.shared_ends(0.1, 0.2, 0.0, 1.0, 0.5, 0.5)
        self.assertAlmostEqual(cpu, 0.5)
        self.assertAlmostEqual(copy, 1.2)

    def test_roles_swap(self):
        a = pm.shared_ends(0.0, 0.7, 0.2, 0.4, 0.6, 0.8)
        b = pm.shared_ends(0.2, 0.4, 0.0, 0.7, 0.8, 0.6)
        self.assertAlmostEqual(a[0], b[1])
        self.assertAlmostEqual(a[1], b[0])


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
        w = pm.predict(pm.Load(rows=2, copies=1, exchanges=1, gpu_experts=1, gpu_picks=1), r)
        w0 = pm.predict(pm.Load(rows=2, copies=1, gpu_experts=1, gpu_picks=1), r)
        mb = r.record_bytes * 1e-6
        self.assertAlmostEqual(w.copies - w0.copies, mb / r.copy_gbps[0])
        o = pm.predict(pm.Load(rows=1, copies=1, exchanges=1, gpu_experts=1, gpu_picks=1), r)
        o0 = pm.predict(pm.Load(rows=1, copies=1, gpu_experts=1, gpu_picks=1), r)
        self.assertAlmostEqual(o.copies, o0.copies)

    def test_rounds(self):
        r = self.rates()
        four = pm.predict(pm.Load(rows=2, gpu_experts=5, gpu_picks=5, gpu_rounds_of=4), r)
        one = pm.predict(pm.Load(rows=2, gpu_experts=5, gpu_picks=5, gpu_rounds_of=32), r)
        self.assertAlmostEqual(four.gpu - one.gpu, r.gpu_round)

    def test_pass_is_later_of_gpu_and_host(self):
        r = self.rates()
        f = pm.predict(pm.Load(rows=2, gpu_experts=1, gpu_picks=1, helper_experts=8, helper_columns=16), r)
        self.assertAlmostEqual(f.passed, max(f.gpu, f.helper + 0.02))


if __name__ == '__main__':
    unittest.main()
