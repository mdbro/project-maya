import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

import native_reroute_replay as r

r.set_quiet(None)   # unit tests never wait on the shared quiet flag


def triplets(base_rows, ready, bonus):
    """Score triplets as the engine logs them: sigmoid, base, and base plus the bonus where ready."""
    out = []
    for base in base_rows:
        for e in range(128):
            b = np.float32(base[e])
            adj = np.float32(b + np.float32(bonus)) if ready[e] else b
            out += [0.01, float(b), float(adj)]
    return out


def words(ids):
    w = [0, 0, 0, 0]
    for e in ids:
        w[e // 32] |= 1 << (e % 32)
    return w


class RoutingRule(unittest.TestCase):
    def test_bonus_moves_only_ready_experts_and_ties_go_to_lower_id(self):
        base = np.zeros((1, 128), np.float32)
        base[0, [10, 11, 12, 13]] = 1.0
        base[0, 50] = 0.95
        ready = np.zeros(128, bool)
        ready[50] = True
        self.assertEqual(sorted(r.reroute(base, ready, 0.0)[0].tolist()), [10, 11, 12, 13])
        self.assertEqual(r.reroute(base, ready, 0.10)[0].tolist(), [50, 10, 11, 12])
        # Equal adjusted scores rank by expert id.
        base[0, 9] = 1.0
        self.assertEqual(r.reroute(base, np.zeros(128, bool), 0.0)[0].tolist(), [9, 10, 11, 12])

    def test_float32_addition_decides_near_ties(self):
        base = np.zeros((1, 128), np.float32)
        base[0, :4] = np.float32(16.4035243988)
        base[0, 7] = np.float32(16.3035243988)
        ready = np.zeros(128, bool)
        ready[7] = True
        got = r.reroute(base, ready, 0.10)[0].tolist()
        expect_7_first = np.float32(base[0, 7] + np.float32(0.10)) > base[0, 0]
        self.assertEqual(got[0] == 7, bool(expect_7_first))

    def test_ready_words_map_bit_e_mod_32_of_word_e_div_32(self):
        m = r.bits_to_mask(np.array([words([0, 33, 127])], np.uint32))[0]
        self.assertEqual(np.nonzero(m)[0].tolist(), [0, 33, 127])


    def test_floor_keeps_the_cheapest_substitutions_and_the_cap_counts_them(self):
        base = np.zeros((1, 128), np.float32)
        sig = np.zeros((1, 128), np.float32)
        base[0, [10, 11, 12, 13]] = [1.00, 0.99, 0.98, 0.97]       # the router's four, none ready
        sig[0, [10, 11, 12, 13]] = [0.40, 0.30, 0.20, 0.10]
        base[0, [50, 51]] = [0.90, 0.89]                            # ready, within the bonus
        sig[0, [50, 51]] = [0.08, 0.02]
        ready = np.zeros(128, bool); ready[[50, 51]] = True
        plain = sorted(r.reroute(base, ready, 0.2)[0].tolist())
        self.assertEqual(plain, [10, 11, 50, 51])                   # 12 and 13 displaced
        self.assertEqual(sorted(r.reroute_floor(base, sig, ready, 0.2)[0].tolist()), plain)
        # The cheapest pair first: 13 (0.10) for 50 (0.08) keeps 0.98; then 12 (0.20) for 51 (0.02) keeps 0.80.
        self.assertEqual(sorted(r.reroute_floor(base, sig, ready, 0.2, floor=0.9)[0].tolist()), [10, 11, 12, 50])
        self.assertEqual(sorted(r.reroute_floor(base, sig, ready, 0.2, floor=0.79)[0].tolist()), plain)
        self.assertEqual(sorted(r.reroute_floor(base, sig, ready, 0.2, cap=1)[0].tolist()), [10, 11, 12, 50])


    def test_route_rules_rank_ratio_and_drop(self):
        base = np.zeros((1, 128), np.float32); sig = np.zeros((1, 128), np.float32)
        base[0, [10, 11, 12, 13]] = [1.00, 0.99, 0.98, 0.97]; sig[0, [10, 11, 12, 13]] = [0.4, 0.3, 0.2, 0.1]
        base[0, [50, 51]] = [0.90, 0.89]; sig[0, [50, 51]] = [0.15, 0.05]
        ready = np.zeros(128, bool); ready[[11, 50, 51]] = True
        # rank protection: bonus 0.2 only at ranks 3 and 4 keeps the router's top two
        got = r.route_rule(base, sig, ready, dict(kind='rank', bonus=(0, 0, 0.2, 0.2)))[0].tolist()
        self.assertEqual(got[:2], [10, 11]); self.assertEqual(sorted(got[2:]), [50, 51])
        # ratio: 13 (0.1) -> 50 (0.15 >= 0.8 x 0.1); 12 (0.2) -> 51 needs 0.05 >= 0.16: no; 10 (0.4) not replaced
        got = r.route_rule(base, sig, ready, dict(kind='ratio', r=0.8))[0].tolist()
        self.assertEqual(got, [10, 11, 12, 50])
        got = r.route_rule(base, sig, ready, dict(kind='ratio', r=0.8, window=4))[0].tolist()
        self.assertEqual(got, [10, 11, 12, 13])     # no candidate inside the router's top four
        # drop: the unready rank-4 pick of the router's own four is dropped
        got = r.route_rule(base, sig, ready, dict(kind='drop', ranks=(3,)))[0].tolist()
        self.assertEqual(got, [10, 11, 12, -1])


class ScoreUnits(unittest.TestCase):
    def test_native_score_decays_by_095_per_five_rows_and_converges_to_100(self):
        self.assertAlmostEqual(float(r.decay_factor(5, 0.95)), 0.95, places=6)
        c = float(r.credit(1, 0.95))
        self.assertAlmostEqual(c, 100 * (1 - 0.95 ** 0.2), places=4)
        v = np.float32(0)
        for _ in range(3000):
            v = np.float32(v * r.decay_factor(1, 0.95) + r.credit(1, 0.95))
        self.assertAlmostEqual(float(v), 100.0, delta=0.05)

    def test_admission_budget_accrues_capacity_over_five_rows_and_caps(self):
        self.assertAlmostEqual(r.budget_advance(0.0, 0, 1, 8.0), 1.6)
        self.assertAlmostEqual(r.budget_advance(0.0, 0, 100, 1.0), 1.0)
        self.assertAlmostEqual(r.budget_advance(-0.5, 0, 1, 1.0), 0.2)


def capture_lines(passes, bonus=0.10, layers=(3, 4)):
    """A minimal detailed capture: request snapshot, then per pass a routes record (one row) and its service record."""
    req = {'type': 'request', 'schema': 1, 'request': 1, 'layer_count': 5, 'cooperative_cpu_scale': 1.3,
           'policy': {'horizon': 20, 'max_exchanges': 8, 'service_saved_ms': 0.66, 'h2d_ms': 0.62, 'd2h_ms': 0.6,
                      'interference_ms': 0, 'margin_ms': 0, 'primary_backing': False},
           'layers': [{'layer': li, 'primary': [0, 1], 'helper': [2, 3], 'host': list(range(4, 128))} for li in layers],
           'batch_staging_calibration': {}}
    out = [{'type': 'request_config', 'prompt_tokens': 10},
           {'type': 'residency', 'record': req}]
    for i, (forward, layer, base, primary, helper, staging) in enumerate(passes):
        ready = np.zeros(128, bool)
        ready[primary] = True
        ready[helper] = True
        for k in staging:
            if k // 1024 == layer:
                ready[k % 1024] = True
        b = np.asarray(base, np.float32)
        orig = r.router_top4(b[None])[0].tolist()
        exe = r.reroute(b[None], ready, bonus)[0].tolist()
        rec = {'type': 'routes', 'request': 1, 'event': i + 1, 'forward': forward, 'generation': 1, 'layer': layer, 'rows': 1,
               'positions': [10 + forward], 'original': orig, 'executed': exe,
               'all_score_triplets_sigmoid_base_prior': triplets([b], ready, bonus),
               'primary': primary, 'helper': helper,
               'staging': [[k, -1, 0, 0, 0, forward] for k in staging] + [[-1, -1, 0, 0, 0, 0]] * (8 - len(staging)),
               'exchanges': [[-1, -1, -1, 0], [-1, -1, -1, 0]], 'helper_exchange': [-1, -1, 0, 0],
               'ready_primary': words(list(primary) + [k % 1024 for k in staging if k // 1024 == layer]),
               'ready_helper': words(helper)}
        out.append({'type': 'residency', 'forward': forward, 'layer': layer, 'record': rec})
        out.append({'type': 'residency', 'forward': forward, 'layer': layer,
                    'record': {'type': 'service', 'cpu_jobs': 1, 'helper_jobs': 0, 'demand_uploads': 0}})
    return out


class ExtractAndReplay(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(1)
        self.passes = []
        for forward in range(3):
            for layer in (3, 4):
                base = rng.normal(0, 0.05, 128).astype(np.float32) + 1.0
                self.passes.append((forward, layer, base, [0, 1], [2, 3], [layer * 1024 + 9]))

    def extract(self, lines):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'req.jsonl'
            p.write_text(''.join(json.dumps(x) + '\n' for x in lines))
            x = r.extract_capture(p)
            r.save_extract(x, Path(tmp) / 'x.npz')
            return r.load_extract(Path(tmp) / 'x.npz')

    def test_extract_keeps_scores_masks_and_the_recorded_routing_reproduces(self):
        x = self.extract(capture_lines(self.passes))
        a = x['arrays']
        self.assertEqual(a['row_base'].shape, (6, 128))
        rp = a['row_pass']
        ready = r.bits_to_mask(a['ready_primary'])[rp] | r.bits_to_mask(a['ready_helper'])[rp]
        self.assertTrue((ready == a['row_adjusted_ready']).all())
        self.assertTrue((r.reroute(a['row_base'], ready, 0.10) == a['row_executed']).all())
        res = r.run_recorded(x, 0.10)
        self.assertTrue(res['same_rows'].all())

    def test_recorded_metrics_count_picks_by_where_they_run(self):
        base = np.full(128, 0.5, np.float32)
        base[[0, 2, 9, 40]] = 1.0          # 5090 resident, 3090 resident, staged, HOST
        passes = [(0, 3, base, [0, 1], [2, 3], [3 * 1024 + 9])]
        x = self.extract(capture_lines(passes, bonus=0.0))
        res = r.run_recorded(x, 0.0)
        v = {k: float(arr.sum()) for k, arr in res['tally'].v.items()}
        self.assertEqual((v['on_primary'], v['on_helper'], v['on_staged'], v['need_load'], v['moved']), (1, 1, 1, 1, 0))
        res = r.run_recorded(x, 0.6)       # every ready expert now outranks the HOST pick
        v = {k: float(arr.sum()) for k, arr in res['tally'].v.items()}
        self.assertEqual((v['need_load'], v['moved']), (0, 1))
        self.assertAlmostEqual(v['mass_executed'], v['mass_original'])


    def test_extract_reads_the_early_copy_prediction_record(self):
        lines = capture_lines(self.passes[:2])
        base = [float(np.float32(1.0 + 0.001 * e)) for e in range(128)]
        top = r.router_top4(np.asarray([base], np.float32))[0].tolist()
        lines.insert(4, {'type': 'early_copy_prediction', 'forward': 0, 'layer': 3, 'target_layer': 4, 'outcome': 'no_candidate',
                         'expert': -1, 'slot': -1, 'demand_copies': 0, 'exchanges_pending': 0, 'predicted': top, 'base': base})
        a = self.extract(lines)['arrays']
        self.assertEqual(a['pred_target'].tolist(), [4])
        self.assertEqual(a['pred_outcome'].tolist(), [r.PRED_OUTCOME['no_candidate']])
        self.assertTrue(a['pred_has_base'].all())
        self.assertEqual(r.router_top4(a['pred_base'])[0].tolist(), a['pred_ids'][0].tolist())


class Rules(unittest.TestCase):
    def sim(self, spec=None, bonus=0.1):
        lines = capture_lines([(0, 3, np.ones(128, np.float32), [0, 1], [2, 3], [])])
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'req.jsonl'
            p.write_text(''.join(json.dumps(x) + '\n' for x in lines))
            x = r.extract_capture(p)
        return r.Sim(x, {}, spec or r.PolicySpec('P1'), bonus)

    def test_helper_rule_needs_score_18_and_more_than_125_times_victim_plus_1(self):
        s = self.sim()
        s.hcredit, s.hrow, s.row = 1.0, 0, 0
        s.slow[3, 2], s.slow[3, 3] = 1.0, 0.5           # 3090 residents: victim is expert 3 (0.5)
        s.slow[4, 2], s.slow[4, 3] = 9.0, 9.0           # (the other layer's residents score higher)
        s.slow[3, 50] = 1.6                             # below 1.8: no lease
        s.admit_helper(0)
        self.assertIsNone(s.lease)
        s.slow[3, 50] = 1.9                             # 1.9 > 1.25 * 0.5 + 1: lease
        s.admit_helper(0)
        self.assertEqual((s.lease['incoming'], s.lease['victim']), ((3, 50), (3, 3)))
        for now in range(1, 10):
            s.helper_progress(now)
        self.assertIsNone(s.lease)
        self.assertTrue(s.helper[3, 50] and not s.helper[3, 3])

    def test_primary_exchange_is_priced_and_a_staged_copy_needs_no_upload(self):
        s = self.sim()
        s.pcredit, s.prow, s.row = 1.0, 0, 0
        s.slow[3, 0], s.slow[3, 1] = 5.0, 6.0           # 5090 residents: victim is expert 0
        s.slow[4, 0], s.slow[4, 1] = 9.0, 9.0           # (the other layer's residents score higher)
        s.slow[3, 60] = 7.5                             # HOST: (7.5 - 5) * benefit < h2d + d2h, too little
        s.forward = 0
        s.admit_primary(0)
        self.assertEqual(s.counts['primary_issued'], 0)
        s.owner[2] = (3, 60)                            # the same expert in a staged frame: only d2h is paid
        s.admit_primary(0)
        self.assertEqual(s.counts['primary_issued'], 1)
        self.assertTrue(s.px[0]['reuse'])
        s.publish_primary(0)
        self.assertTrue(s.res[3, 60] and not s.res[3, 0])
        self.assertEqual(s.owner[2], (3, 0))            # the victim stays in the landing frame

    def test_two_speed_score_keeps_native_units_for_a_steady_pick(self):
        spec = r.PolicySpec('P3', decay5=0.99 ** 5, fast5=0.7 ** 5, fast_weight=0.5)
        s = self.sim(spec)
        s.slow[:] = 0
        s.fast[:] = 0
        for _ in range(2000):
            s.begin_forward(1)
            s.observe(3, np.array([[7, 8, 9, 10]]), 1)
        self.assertAlmostEqual(float(s.score()[3, 7]), 100.0, delta=0.1)

    def test_bonus_aware_credit_goes_to_experts_within_the_bonus_of_a_loaded_pick(self):
        s = self.sim(r.PolicySpec('P4', bonus_credit=1.0), bonus=0.1)
        base = np.zeros((1, 128), np.float32)
        base[0, [40, 41, 42, 43]] = 1.0                 # all HOST, all loaded
        base[0, 44] = 0.95                              # within 0.1 of a loaded pick
        base[0, 45] = 0.85                              # not
        ready = np.zeros(128, bool)
        executed = r.reroute(base, ready, 0.1).astype(np.int64)
        self.assertEqual(s.bonus_aware(base, executed, ready).tolist(), [44])


class WindowSharing(unittest.TestCase):
    def test_plan_count_literal_and_greedy_sharing(self):
        base = np.zeros((2, 128), np.float32)
        base[0, [40, 50, 51, 52]] = [1.0, 0.95, 0.94, 0.93]
        base[0, 41] = 0.9                                  # row 1's load, within 0.2 of every load of row 0
        base[1, [41, 60, 61, 62]] = [1.0, 0.99, 0.98, 0.97]
        original = r.router_top4(base)
        a = dict(pass_generation=np.array([True]), pass_rows=np.array([2]), pass_row0=np.array([0]), row_base=base,
                 row_sig=np.full((2, 128), 0.01, np.float32), row_original=original,
                 ready_primary=np.zeros((1, 4), np.uint32), ready_helper=np.zeros((1, 4), np.uint32))
        w = r.window_sharing(dict(arrays=a, meta={}), 0.2)['by_width'][2]
        self.assertEqual((w['load_picks'], w['loads'], w['near_other_row']), (8, 8, 4))
        self.assertEqual(w['literal_saved'], 0)            # every union member keeps its bonus: nothing is dropped
        self.assertEqual((w['greedy_saved'], w['greedy_moved']), (1, 1))   # row 0 takes 41 instead of loading 40


    def test_greedy_keeps_a_member_that_an_outranking_member_cannot_displace(self):
        # Row 0's weakest pick is a resident (5, 0.75 + 0.2 = 0.95 < 1.0): row 1's load 41 would push it out, not 40.
        # The member stays, nothing is saved, and the loop ends (the 2026-11-10 run looped here for hours).
        base = np.zeros((2, 128), np.float32)
        base[0, [40, 50, 51, 5]] = [1.0, 0.99, 0.98, 0.75]
        base[0, 41] = 0.85
        base[1, [41, 60, 61, 62]] = [1.0, 0.99, 0.98, 0.97]
        ready = np.zeros(128, bool)
        ready[5] = True
        ex = r.reroute(base, ready, 0.2).astype(np.int64)
        self.assertEqual(sorted(ex[0].tolist()), [5, 40, 50, 51])
        out = r.share_greedy(base, ready, ex, 0.2)
        self.assertTrue((np.sort(out, axis=1) == np.sort(ex, axis=1)).all())


if __name__ == '__main__':
    unittest.main()
