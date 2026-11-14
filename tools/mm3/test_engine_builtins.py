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

    def test_kernel_item_2s_switches_are_off_until_timed(self):
        for drafts in (True, False):
            b = engine_builtins.request_builtins(drafts)
            self.assertEqual((b['cpu_kernel'], b['cpu_l3_prefetch']), (0, 0))


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
