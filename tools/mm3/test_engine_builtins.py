#!/usr/bin/env python3
"""engine_builtins.py: the built-ins it reads from the engine's sources are the decided ones (CLAUDE.md's decision
table), and a field it can no longer find raises instead of passing silently. Standard library only; no engine.

    python3 -m unittest tools/mm3/test_engine_builtins.py"""
from __future__ import annotations

import pathlib
import shutil
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import engine_builtins  # noqa: E402


class DecidedBuiltins(unittest.TestCase):
    """The values a request that sets nothing gets, in both modes."""

    def test_the_cpu_lane_takes_nine_threads(self):
        # (decided by the agent under the user's overnight authorization, 2026-11-14: 9 of the lane's threads, the caller
        #  one of them; before, -1: all of them)
        for drafts in (True, False):
            self.assertEqual(engine_builtins.request_builtins(drafts)['cpu_threads'], 9)

    def test_the_adopted_routing_built_ins(self):
        # (2026-11-14: bonus 0.30 dampened, the budget 0.5)
        for drafts in (True, False):
            b = engine_builtins.request_builtins(drafts)
            self.assertEqual((b['routing_prior_bonus'], b['routing_dampen'], b['routing_floor']), (0.3, 1, 0.5))

    def test_the_helper_hand_off_stays_off(self):
        # (kernel item 1 not adopted, 2026-11-14: exact but not faster)
        for drafts in (True, False):
            self.assertEqual(engine_builtins.request_builtins(drafts)['helper_handoff'], 0)

    def test_the_cpu_lane_runs_the_checkpoint_precision_kernel_without_the_prefetch(self):
        # (kernel item 2, 2026-11-14: cpu_kernel 1 adopted - the same bits, faster; then cpu_kernel 2, NVFP4 activations, after
        #  kernel batch 1 - the checkpoint's precision, a tie (CLAUDE.md af2b7343); the L3 prefetch not adopted)
        for drafts in (True, False):
            b = engine_builtins.request_builtins(drafts)
            self.assertEqual((b['cpu_kernel'], b['cpu_l3_prefetch']), (2, 0))

    def test_the_dense_matrices_take_8bit_activations_on_every_pass(self):
        # (kernel item 3, 2026-11-14, CLAUDE.md da32d5d: dense_fp8 1 - passes of 2 or more rows - on kernel 2; then dense_fp8 2,
        #  one-row passes too, CLAUDE.md c99582a0: every lane and mode at the checkpoint's activation precision)
        for drafts in (True, False):
            b = engine_builtins.request_builtins(drafts)
            self.assertEqual((b['dense_fp8'], b['dense_fp8_kernel']), (2, 2))

    def test_prep_early_stays_off_until_timed(self):
        for drafts in (True, False):
            self.assertEqual(engine_builtins.request_builtins(drafts)['prep_early'], 0)

    def test_kernel_batch_1s_adopted_built_ins(self):
        # (kernel batch 1, runs/kbatch1-20261114, decided by the team lead 2026-11-14: expert_fast 2 with expert_overlap 1 and
        #  fp4_kernel 4 - the same bits, faster; item 5's NVFP4-activation helper kernels on both cards in both modes - 172
        #  multi-column, 68 and 72 one-column - with helper_q_rows 1 and third_sm89 1: the checkpoint's precision, CLAUDE.md
        #  c99582a0)
        for drafts in (True, False):
            b = engine_builtins.request_builtins(drafts)
            self.assertEqual({k: b[k] for k in ('expert_fast', 'expert_overlap', 'fp4_kernel', 'helper_kernel', 'third_kernel', 'helper_kernel_one',
                                                 'third_kernel_one', 'helper_q_rows', 'third_sm89')},
                             dict(expert_fast=2, expert_overlap=1, fp4_kernel=4, helper_kernel=172, third_kernel=172, helper_kernel_one=68,
                                  third_kernel_one=72, helper_q_rows=1, third_sm89=1))

    def test_the_router_kernel_and_the_head_copy_stay_off(self):
        # (kernel batch 1: router_kernel the same bits but no gain, not adopted; head_lowp 0 - the BF16 head, as the checkpoint
        #  leaves lm_head unquantized, CLAUDE.md c99582a0)
        for drafts in (True, False):
            b = engine_builtins.request_builtins(drafts)
            self.assertEqual([b[k] for k in ('router_kernel', 'head_lowp', 'head_lowp_fmt')], [0] * 3)

    def test_pass_model_stays_off_until_timed(self):
        # (plan step 6: the bandwidth-aware pass model in place of the CPU price, off until its A/B)
        for drafts in (True, False):
            self.assertEqual(engine_builtins.request_builtins(drafts)['pass_model'], 0)


class RenamedField(unittest.TestCase):
    """A built-in whose line no longer matches its pattern raises (a renamed field cannot pass silently)."""

    def test_a_missing_cpu_threads_line_raises(self):
        with tempfile.TemporaryDirectory() as d:
            src = pathlib.Path(d)
            for f in engine_builtins.SRC.iterdir():
                if f.is_file() and f.suffix in ('.hpp', '.cpp'):
                    shutil.copy(f, src / f.name)
            sample = src / 'mm3_sample.hpp'
            sample.write_text(sample.read_text().replace('int cpu_threads = 9;', 'int lane_threads = 9;'))
            old = engine_builtins.SRC
            engine_builtins.SRC = src
            try:
                with self.assertRaises(ValueError):
                    engine_builtins.request_builtins(True)
            finally:
                engine_builtins.SRC = old


if __name__ == '__main__':
    unittest.main()
