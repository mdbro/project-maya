import unittest

import numpy as np

import native_window_paths as w


class ClockMapTest(unittest.TestCase):
    def test_recovers_a_drifting_offset(self):
        rng = np.random.default_rng(1)
        stamps = np.arange(2000, dtype=np.int64) * 1_000_000 + 10**12       # steady clock, one pass a millisecond
        offset = 5_000 + 3 * np.arange(2000)                                  # export minus steady: 5 us, drifting 3 ns a pass
        launches = []
        for i, s in enumerate(stamps):
            if i % 7 == 3:
                launches.append(None)                                         # passes without the add launch
                continue
            end = s + offset[i] - 2_000 + rng.integers(-500, 500)            # the launch returns about 2 us before the stamp
            launches.append((end - 40_000, end))
        correction, stats = w.clock_map(launches, list(stamps))
        points = (500, 1000, 1500)
        mapped = np.array([stamps[i] + offset[i] for i in points], dtype=float)   # export events at the stamps' moments
        back = mapped - correction(mapped)
        for got, i in zip(back, points):
            # the anchor absorbs the launch-to-stamp gap, so mapped times run that gap (2 us) late
            self.assertLess(abs(got - (stamps[i] + 2_000)), 500)
        self.assertEqual(stats['anchored_passes'], sum(x is not None for x in launches))


class SummarizeTest(unittest.TestCase):
    def test_last_lane_and_pairs(self):
        rows = [dict(forward=1, layer=3, rows=8, cycle_ms=3.0, wait_ms=0.4, publication=2.1,
                     ends=dict(cpu=2.0, primary=1.5, primary_upload=1.5, helper=0.5)),
                dict(forward=2, layer=3, rows=8, cycle_ms=3.0, wait_ms=0.0, publication=1.2,
                     ends=dict(cpu=1.0, primary=1.8, primary_upload=None, helper=None))]
        s = w.summarize(rows)[8]
        self.assertEqual(s['last'], {'cpu': 0.5, 'primary': 0.5})
        self.assertAlmostEqual(s['mean']['cpu|primary'], (0.5 + 0.8) / 2)
        self.assertAlmostEqual(s['mean']['cpu|primary_upload'], 0.5)
        self.assertAlmostEqual(s['mean']['helper|cpu'], 1.5)
        self.assertAlmostEqual(s['mean']['slowest_minus_second'], (0.5 + 0.8) / 2)
        self.assertAlmostEqual(s['publication_after_host_lanes_ms'], (0.1 + 0.2) / 2)


if __name__ == '__main__':
    unittest.main()
