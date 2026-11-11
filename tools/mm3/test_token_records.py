"""Tests of tools/mm3/token_records.py (the routing lab's token records):  cd tools/mm3 && python3 -B -m unittest
test_token_records"""
import json
import math
import os
import pathlib
import random
import tempfile
import threading
import time
import unittest

import token_records as tr


def softmax(xs):
    m = max(xs)
    w = [math.exp(x - m) for x in xs]
    z = sum(w)
    return [v / z for v in w]


def record(i, tok, probs, k=4, **extra):
    """A token record from a whole distribution, as the engine writes it (top k by p desc then id asc, rest, entropy)."""
    order = sorted(range(len(probs)), key=lambda j: (-probs[j], j))
    top = [[j, probs[j]] for j in order[:k]]
    rest = sum(probs[j] for j in order[k:])
    ent = -sum(p * math.log(p) for p in probs if p > 0)
    r = {"type": "token", "i": i, "pos": 100 + i, "id": tok, "p": probs[tok], "logp": math.log(probs[tok]),
         "p_raw": probs[tok], "logp_raw": math.log(probs[tok]), "top": top, "rest": rest, "entropy": ent,
         "samp": top, "samp_n": len(probs), "samp_rest": rest, "forced": False, "source": "target", "rows": 1,
         "fwd": i, "row": 0}
    r.update(extra)
    return r


def capture(recs, done=None):
    c = tr.Capture()
    c.add({"type": "begin", "schema": 1})
    for r in recs:
        c.add(r)
    c.add(done or {"type": "done", "tokens": len(recs), "finish": "stop"})
    return c


class Reading(unittest.TestCase):
    def test_read_text_and_partial_line(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "a.jsonl"
            recs = [{"type": "begin", "schema": 1}, record(0, 2, softmax([0, 1, 3, 2, -1]))]
            p.write_text("".join(json.dumps(r) + "\n" for r in recs) + '{"type":"tok')   # (a line being written)
            cap = tr.read(p, text=lambda i: f"<{i}>")
            self.assertEqual(len(cap.tokens), 1)
            self.assertFalse(cap.finished)
            t = cap.tokens[0]
            self.assertEqual(t["text"], "<2>")
            self.assertEqual([e[2] for e in t["top"]], ["<2>", "<3>", "<1>", "<0>"])
            self.assertEqual(len(t["samp"][0]), 3)
            plain = tr.read(p)
            self.assertNotIn("text", plain.tokens[0])

    def test_follow_streams_and_ends_at_done(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "b.jsonl"
            lines = [{"type": "begin"}] + [record(i, 1, softmax([0, 2, 1])) for i in range(5)] + [{"type": "done"}]

            def writer():
                time.sleep(0.15)   # (the file appears after the reader starts)
                with open(p, "w") as f:
                    for r in lines:
                        s = json.dumps(r) + "\n"
                        f.write(s[:7])          # (half a line, flushed: the reader must wait for the rest)
                        f.flush()
                        time.sleep(0.01)
                        f.write(s[7:])
                        f.flush()
            th = threading.Thread(target=writer)
            th.start()
            got = list(tr.follow(p, poll_s=0.005, timeout_s=5, wait_for_file_s=5))
            th.join()
            self.assertEqual([r["type"] for r in got], ["begin"] + ["token"] * 5 + ["done"])

    def test_follow_times_out(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "c.jsonl"
            with self.assertRaises(TimeoutError):
                list(tr.follow(p, poll_s=0.01, wait_for_file_s=0.05))
            p.write_text('{"type":"begin"}\n')
            with self.assertRaises(TimeoutError):
                list(tr.follow(p, poll_s=0.01, timeout_s=0.1))


class Kl(unittest.TestCase):
    def test_identical_is_exactly_zero(self):
        rng = random.Random(3)
        for _ in range(200):
            probs = softmax([rng.gauss(0, 3) for _ in range(50)])
            r = record(0, 1, probs, k=8)
            self.assertEqual(tr.partition_kl(r, json.loads(json.dumps(r))), 0.0)

    def test_coarsened_kl_by_hand_and_a_lower_bound(self):
        rng = random.Random(5)
        for _ in range(300):
            n = 30
            p = softmax([rng.gauss(0, 2) for _ in range(n)])
            q = softmax([x + rng.gauss(0, 0.7) for x in [math.log(v) for v in p]])
            k = rng.choice([1, 3, 8])
            a, b = record(0, 0, p, k=k), record(0, 0, q, k=k)
            top_a = {e[0] for e in a["top"]}
            top_b = {e[0] for e in b["top"]}
            both = top_a & top_b
            pr = 1 - sum(p[j] for j in both)
            qr = 1 - sum(q[j] for j in both)
            want = sum(p[j] * math.log(p[j] / q[j]) for j in both) + pr * math.log(pr / qr)
            got = tr.partition_kl(a, b)
            self.assertAlmostEqual(got, max(want, 0.0), places=10)
            full = sum(p[j] * math.log(p[j] / q[j]) for j in range(n))
            self.assertLessEqual(got, full + 1e-12)
            self.assertGreaterEqual(got, 0.0)

    def test_asymmetric_and_sensitive(self):
        p = softmax([3, 1, 0, 0, -2, -2])
        q = softmax([1, 2, 0.5, 0, -1, -3])
        a, b = record(0, 0, p), record(0, 0, q)
        self.assertGreater(tr.partition_kl(a, b), 0.1)
        self.assertNotEqual(tr.partition_kl(a, b), tr.partition_kl(b, a))
        c = json.loads(json.dumps(a))
        c["top"][0][1] *= 1.001          # (a mutation the comparison must see)
        c["rest"] -= a["top"][0][1] * 0.001
        self.assertGreater(tr.partition_kl(a, c), 0.0)


class Compare(unittest.TestCase):
    def make(self, seed, noise, n=40, ids=None):
        rng = random.Random(seed)
        base = random.Random(99)
        recs = []
        for i in range(n):
            logits = [base.gauss(0, 2) for _ in range(20)]
            probs = softmax([x + rng.gauss(0, noise) for x in logits])
            tok = ids[i] if ids else max(range(20), key=lambda j: (logits[j], -j))
            recs.append(record(i, tok, probs, k=6))
        return capture(recs)

    def test_self_test_zero_and_mutation_caught(self):
        a = self.make(1, 0.3)
        self.assertEqual(tr.self_test(a), [])
        b = json.loads(json.dumps(a.tokens))
        b[7]["entropy"] += 1e-9          # (the smallest change the comparison must still see)
        r = tr.compare(a, capture(b))
        self.assertNotEqual(r["series"]["d_entropy"][7], 0.0)
        self.assertEqual(r["series"]["d_entropy"][6], 0.0)
        c = json.loads(json.dumps(a.tokens))
        c[3]["top"][0][1] = float("nan")  # (a record the self-test must refuse: its KL with itself is not 0)
        self.assertTrue(any("kl" in s for s in tr.self_test(capture(c))))
        self.assertNotEqual(tr.self_test(tr.Capture()), [])   # (nothing to compare is a failure)

    def test_series_summaries_and_divergence(self):
        ref = self.make(1, 0.0)
        other = self.make(2, 0.5)
        r = tr.compare(ref, other)
        self.assertEqual(r["aligned"]["compared"], 40)
        self.assertIsNone(r["aligned"]["diverged_at"])
        kl = r["series"]["kl"]
        s = r["summary"]["kl"]
        self.assertAlmostEqual(s["mean"], sum(kl) / len(kl))
        srt = sorted(kl)
        self.assertAlmostEqual(s["median"], (srt[19] + srt[20]) / 2)
        self.assertAlmostEqual(s["p95"], srt[37] + (srt[38] - srt[37]) * 0.05)
        self.assertEqual(s["max"], srt[-1])
        self.assertAlmostEqual(r["summary"]["shares"]["kl_over_0.01"], sum(v > 0.01 for v in kl) / 40)
        self.assertAlmostEqual(r["summary"]["top1_agreement"], sum(r["series"]["top1"]) / 40)
        lr = r["series"]["log_ratio"]
        self.assertAlmostEqual(lr[3], other.tokens[3]["logp_raw"] - ref.tokens[3]["logp_raw"])
        # a free run: compared up to its first different id
        ids = ref.ids[:]
        ids[11] = (ids[11] + 1) % 20
        r2 = tr.compare(ref, self.make(2, 0.5, ids=ids))
        self.assertEqual(r2["aligned"]["diverged_at"], 11)
        self.assertEqual(r2["aligned"]["compared"], 11)

    def test_numpy_quantiles_agree(self):
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy")
        rng = random.Random(8)
        for n in (1, 2, 7, 100):
            v = [rng.random() for _ in range(n)]
            s = tr.summary(v)
            self.assertAlmostEqual(s["p95"], float(np.quantile(v, 0.95)))
            self.assertAlmostEqual(s["median"], float(np.median(v)))

    def test_check_forced(self):
        cap = capture([record(i, t, softmax([0, 1, 2]), forced=True) for i, t in enumerate([2, 1, 0])],
                      {"type": "done", "finish": "forced"})
        self.assertEqual(tr.check_forced(cap, [2, 1, 0]), [])
        self.assertTrue(tr.check_forced(cap, [2, 0, 0]))
        self.assertTrue(tr.check_forced(cap, [2, 1, 0, 1]))
        cap.tokens[1]["forced"] = False
        self.assertTrue(tr.check_forced(cap, [2, 1, 0]))


class Price(unittest.TestCase):
    def test_per_layer_one_row_and_window(self):
        P = tr.PASS_PRICES["one_row_type"]
        L = tr.PASS_PRICES["one_row_loads"]
        prices = {"one_row_type": P, "one_row_loads": L,
                  "window": {"3": {"a": 0.6, "b": 0.4, "n": 100, "by_loads": {"0": 0.65, "1": 1.0}}}}
        loads = [0] * 50 + [1, 1, 2, 3, 5, 0, 0]
        staged = [0] * 49 + [1] + [0] * 7
        one = {"layers": {"loads": loads, "staged": staged}}
        recs = [record(0, 1, softmax([0, 1]), fwd=0)]
        recs.append(record(1, 1, softmax([0, 1]), fwd=1, rows=1, route=one))
        win = {"layers": {"loads": [0] * 55 + [1, 1], "pass_loads": [0] * 55 + [1, 2]}}
        recs += [record(2 + j, 1, softmax([0, 1]), fwd=2, rows=3, row=j, route=win) for j in range(2)]
        r = tr.price(capture(recs), prices)
        want_one = 49 * P["A"] + P["B"] + 2 * L["1"] + L["2"] + L["3"] + L["4"] + 2 * P["A"]
        want_win = 55 * 0.65 + 1.0 + (0.6 + 0.4 * 2)
        self.assertAlmostEqual(r["by_mode"]["one_row"]["moe_ms"], want_one)
        self.assertAlmostEqual(r["by_mode"]["window"]["moe_ms"], want_win)
        self.assertEqual(r["tokens"], 3)
        self.assertTrue(r["exact"])
        self.assertAlmostEqual(r["token_ms"], want_one / 0.906 + want_win / 0.875)

    def test_window_without_pass_loads_is_a_flagged_lower_bound(self):
        prices = {"one_row_type": tr.PASS_PRICES["one_row_type"], "one_row_loads": tr.PASS_PRICES["one_row_loads"],
                  "window": {"2": {"a": 0.6, "b": 0.4, "n": 10, "by_loads": {}}}}
        rows = [{"layers": {"loads": [0] * 56 + [1]}}, {"layers": {"loads": [0] * 55 + [2, 0]}}]
        recs = [record(1 + j, 1, softmax([0, 1]), fwd=1, rows=2, row=j, route=rows[j]) for j in range(2)]
        r = tr.price(capture(recs), prices)
        self.assertFalse(r["exact"])
        self.assertAlmostEqual(r["moe_ms"], 55 * 0.6 + (0.6 + 0.8) + (0.6 + 0.4))

    def test_totals_are_flagged_and_no_route_is_none(self):
        recs = [record(1, 1, softmax([0, 1]), fwd=1, route={"loads": 3})]
        r = tr.price(capture(recs))
        self.assertFalse(r["exact"])
        A, L1 = tr.PASS_PRICES["one_row_type"]["A"], tr.PASS_PRICES["one_row_loads"]["1"]
        self.assertAlmostEqual(r["moe_ms"], 54 * A + 3 * L1)
        self.assertIsNone(tr.price(capture([record(1, 1, softmax([0, 1]), fwd=1)])))


class PlainLists(unittest.TestCase):
    def test_lists_of_records_work_as_captures(self):
        recs = [record(i, 1, softmax([0, 1, 2]), fwd=i, route={"loads": 1}) for i in range(4)]
        cap = capture(recs)
        self.assertEqual(tr.compare(recs, recs)["summary"], tr.compare(cap, cap)["summary"])
        self.assertEqual(tr.self_test(recs), [])
        whole = [{"type": "begin"}] + [dict(r, forced=True) for r in recs] + [{"type": "done", "finish": "forced"}]
        self.assertEqual(tr.check_forced(whole, [1, 1, 1, 1]), [])
        untyped = [{k: v for k, v in r.items() if k != "type"} for r in recs]
        self.assertEqual(tr.price(untyped), tr.price(cap))
        self.assertIs(tr.estimate_ms, tr.price)


class Cli(unittest.TestCase):
    def test_self_test_exit_codes(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "x.jsonl"
            recs = [{"type": "begin"}] + [record(i, 0, softmax([2, 1, 0])) for i in range(3)] + [{"type": "done"}]
            p.write_text("".join(json.dumps(r) + "\n" for r in recs))
            self.assertEqual(tr.main(["self-test", str(p)]), 0)
            p.write_text('{"type":"begin"}\n{"type":"done"}\n')
            self.assertEqual(tr.main(["self-test", str(p)]), 1)


if __name__ == "__main__":
    unittest.main()
