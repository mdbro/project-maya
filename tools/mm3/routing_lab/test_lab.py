#!/usr/bin/env python3
"""Tests of the routing lab's backend. No engine and no network: `python3 tools/mm3/routing_lab/test_lab.py`."""
from __future__ import annotations

import json
import math
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare            # noqa: E402
import engines            # noqa: E402
import lab_server         # noqa: E402
import settings_schema as schema   # noqa: E402

DEAD = "http://127.0.0.1:9"          # the discard port: nothing listens, so a request sent by mistake fails at once


class Schema(unittest.TestCase):
    def test_defaults_are_valid_and_are_todays_rule(self):
        d = schema.validate({})
        self.assertEqual(d, schema.defaults())
        self.assertEqual(d["routing_prior_bonus"], 0.20)
        self.assertFalse(schema.lab_needed(d))
        self.assertEqual(len(d["route_layer_scale"]), 57)

    def test_out_of_range_and_unknown_choices_are_refused(self):
        for bad in ({"routing_prior_bonus": 1.5}, {"route_protect": 3}, {"route_layer_scale": [1.0] * 56},
                    {"route_weights": 7}, {"route_floor": -0.1}):
            with self.assertRaises(ValueError, msg=str(bad)):
                schema.validate(bad)
        with self.assertRaises(ValueError):
            schema.validate_sampling({"reasoning_effort": "extreme"})
        with self.assertRaises(ValueError):
            schema.validate_sampling({"temperature": 9})

    def test_unknown_keys_are_dropped_not_passed_on(self):
        self.assertNotIn("exe", schema.validate({"exe": "/bin/sh"}))
        self.assertNotIn("strata_tune", schema.validate_sampling({"strata_tune": {"x": 1}}))

    def test_any_rule_beyond_the_bonus_needs_the_lab_path(self):
        d = schema.defaults()
        self.assertFalse(schema.lab_needed(dict(d, routing_prior_bonus=0.4)))
        for key, value in (("route_floor", 0.8), ("route_protect", 1), ("route_weights", 2), ("route_k_widen", 0.1),
                           ("route_bonus_draft", 0.3), ("route_layer_scale", [0.5] + [1.0] * 56)):
            self.assertTrue(schema.lab_needed(dict(d, **{key: value})), key)

    def test_the_floor_is_described_as_a_budget(self):
        self.assertEqual(schema.describe(dict(schema.defaults(), route_floor=0.8)), "bonus 0.20, budget 0.2")
        self.assertEqual(schema.describe(dict(schema.defaults(), route_lab_prompt=0, route_prompt_bonus=0.2)),
                         "bonus 0.20, prompt at 0.2, rules skip the prompt")


class Compare(unittest.TestCase):
    def test_self_test(self):
        compare.self_test()

    def test_divergence_is_zero_for_equal_records_and_a_lower_bound_otherwise(self):
        p = {"id": 1, "top": [[1, 0.5, "a"], [2, 0.3, "b"], [3, 0.1, "c"]], "rest": 0.1}
        q = {"id": 1, "top": [[1, 0.4, "a"], [2, 0.2, "b"], [4, 0.3, "d"]], "rest": 0.1}
        self.assertEqual(compare.kl_with_rest(p, p), 0.0)
        # The full distributions over five outcomes (1, 2, 3, 4, other): the bound must not exceed their KL.
        full_p, full_q = [0.5, 0.3, 0.1, 0.0, 0.1], [0.4, 0.2, 0.0, 0.3, 0.1]
        upper = sum(a * math.log(a / max(b, 1e-12)) for a, b in zip(full_p, full_q) if a > 0)
        got = compare.kl_with_rest(p, q)
        self.assertGreater(got, 0.0)
        self.assertLessEqual(got, upper)

    def test_the_pages_divergence_equals_the_engine_agents_reader(self):
        """Two implementations of one definition: the page's and tools/mm3/token_records.py's."""
        sys.path.insert(0, str(HERE.parent))
        try:
            import token_records
        except ImportError:
            self.skipTest("tools/mm3/token_records.py is not on this branch")
        import random
        rng = random.Random(7)
        for _ in range(200):
            def record():
                ids = rng.sample(range(40), rng.randint(1, 12))
                w = [rng.random() ** 3 + 1e-6 for _ in ids]
                rest = rng.random() * 0.2
                z = sum(w) / (1.0 - rest)
                return {"id": ids[0], "top": [[i, x / z] for i, x in zip(ids, w)], "rest": rest}
            a, b = record(), record()
            self.assertAlmostEqual(compare.kl_with_rest(a, b), token_records.partition_kl(a, b), places=12)

    def test_a_row_the_engine_could_not_read_is_left_out_and_counted(self):
        good = {"id": 1, "p": 0.5, "p_raw": 0.5, "top": [[1, 0.5, "a"], [2, 0.3, "b"]], "rest": 0.2, "entropy": 1.0}
        bad = {"id": 1, "p": 0, "logp": None, "p_raw": None, "top": [], "rest": None, "entropy": None, "invalid_row": True}
        c = compare.compare([good, good], [good, bad])
        self.assertEqual((c["compared"], c["unreadable_rows"]), (1, 1))
        self.assertEqual(c["summary"]["kl_mean"], 0.0)
        st = compare.summarize([good, bad])
        self.assertEqual((st["tokens"], st["unreadable_rows"]), (2, 1))
        self.assertAlmostEqual(st["perplexity"], 2.0)

    def test_shared_prefix_finds_where_two_answers_part(self):
        a = [{"id": 1, "top": [[1, 0.6, "a"]], "rest": 0.4}, {"id": 2, "top": [[2, 0.6, "b"]], "rest": 0.4}]
        b = [a[0], {"id": 3, "top": [[3, 0.7, "c"]], "rest": 0.3}]
        r = compare.shared_prefix(a, b)
        self.assertEqual((r["shared"], r["identical"]), (1, False))
        self.assertEqual(r["first_a"]["id"], 2)
        self.assertTrue(compare.shared_prefix(a, a)["identical"])

    def test_thinking_is_the_prefix_up_to_the_closing_tag(self):
        recs = [{"text": "plan"}, {"text": " more"}, {"text": "</mm:think>"}, {"text": "Hello"}]
        self.assertEqual(compare.thinking_span(recs), 3)
        self.assertEqual(compare.thinking_span(recs[3:]), 0)
        self.assertEqual(compare.thinking_span(recs[:2], saw_reasoning=True), 2)   # cut off while still thinking
        self.assertEqual(compare.summarize([dict(r, id=k, p=0.5, top=[[k, 0.5, r["text"]]]) for k, r in enumerate(recs)])
                         ["thinking_tokens"], 3)


class TokenText(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        # A tiny byte-level vocabulary: "Ġ" is the space byte in GPT-2's alphabet; "Ã" and "¯" are the bytes of "ï".
        vocab = {"a": 0, "Ġthe": 1, "Ã": 2, "¯": 3, "</mm:think>": 4, "Ċ": 5}
        (self.dir / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (self.dir / "token_type.json").write_text(json.dumps([1, 1, 1, 1, 4, 1]))
        self.tt = engines.TokenText(self.dir)

    def test_bytes_and_special_tokens(self):
        self.assertEqual(self.tt.one(1), " the")
        self.assertEqual(self.tt.one(5), "\n")
        self.assertEqual(self.tt.one(4), "</mm:think>")
        self.assertEqual(self.tt.one(99), "<99>")

    def test_control_tokens_are_marked(self):
        self.assertTrue(self.tt.special(4))
        self.assertFalse(self.tt.special(1))
        self.assertFalse(self.tt.special(99))

    def test_a_character_split_across_tokens_appears_on_the_token_that_completes_it(self):
        text = self.tt.stream()
        self.assertEqual([text(0), text(2), text(3), text(1)], ["a", "", "ï", " the"])


class Tail(unittest.TestCase):
    def test_lines_are_read_as_they_complete_and_text_is_added(self):
        d = Path(tempfile.mkdtemp())
        (d / "vocab.json").write_text(json.dumps({"a": 0, "Ġthe": 1}), encoding="utf-8")
        path = d / "lab-x.jsonl"
        tail = engines._Tail(path, engines.TokenText(d))
        self.assertEqual(tail.read(), [])                              # the file does not exist yet
        token = {"type": "token", "i": 0, "id": 1, "p": 0.5, "top": [[1, 0.5], [0, 0.2]], "rest": 0.3,
                 "samp": [[1, 1.0]], "route": {"picks": 228, "moved": 3, "layers": {"kept": [1.0] * 57}}}
        line = json.dumps(token)
        with path.open("w") as f:
            f.write(json.dumps({"type": "begin", "schema": 1}) + "\n" + line[:20])
            f.flush()
            self.assertEqual(tail.read(), [])                          # begin is kept aside; the token is not complete
            self.assertEqual(tail.begin["schema"], 1)
            f.write(line[20:] + "\n" + json.dumps({"type": "done", "tokens": 1}) + "\n")
            f.flush()
            got = tail.read()
        self.assertEqual([r["type"] for r in got], ["token", "done"])
        self.assertEqual(got[0]["text"], " the")
        self.assertEqual(got[0]["top"], [[1, 0.5, " the"], [0, 0.2, "a"]])
        self.assertEqual(got[0]["samp"], [[1, 1.0, " the"]])
        self.assertNotIn("special", got[0])
        self.assertIn("kept", got[0]["layers"])                        # per-layer arrays lifted beside the totals
        self.assertNotIn("layers", got[0]["route"])
        self.assertEqual(tail.read(), [])


class NoNetwork(unittest.TestCase):
    """Every request an adapter would send is recorded and refused at once: a test never opens a socket. (A closed
    port hangs instead of refusing under WSL's mirrored networking, so a real attempt would stall the test.)"""

    def setUp(self):
        self.sent = []
        self._real = engines.urllib.request.urlopen

        def refuse(req, *a, **k):
            self.sent.append(getattr(req, "full_url", req))
            raise engines.urllib.error.URLError("test: no network")
        engines.urllib.request.urlopen = refuse

    def tearDown(self):
        engines.urllib.request.urlopen = self._real


class Guard(NoNetwork):
    """The lab must never talk to an engine that the lab's launcher did not start."""

    def engine(self, locks):
        return engines.MayaEngine(DEAD, {"one_row": {"spec": 1}, "drafting": {"spec": 8}}, locks=locks)

    def held_by(self, owner):
        locks = Path(tempfile.mkdtemp())
        (locks / "gpu.lock.d").mkdir()
        (locks / "gpu.lock.d" / "owner").write_text(owner + "\n")
        return locks

    def test_blocked_unless_the_labs_launcher_holds_the_mutex(self):
        locks = Path(tempfile.mkdtemp())
        e = self.engine(locks)
        self.assertIn("launch_lab.py start", e.blocked())              # nobody holds it: no engine was started
        (locks / "gpu.lock.d").mkdir()
        self.assertIsNotNone(e.blocked())                              # held, owner unknown
        (locks / "gpu.lock.d" / "owner").write_text("exchange-scheduling\n")
        self.assertIn("exchange-scheduling", e.blocked())
        (locks / "gpu.lock.d" / "owner").write_text("routing-lab\n")
        self.assertIsNone(e.blocked())

    def test_nothing_is_sent_while_a_campaign_holds_the_cards(self):
        e = self.engine(self.held_by("exchange-scheduling"))
        info = e.info()
        restored = e.restore()
        events = list(e.generate([{"role": "user", "content": "hi"}], schema.defaults(), schema.sampling_defaults(),
                                 False, None, threading.Event()))
        self.assertEqual(self.sent, [])
        self.assertFalse(info["connected"])
        self.assertIn("blocked", info)
        self.assertFalse(restored["ok"])
        self.assertEqual([ev["type"] for ev in events], ["error"])
        self.assertIn("exchange-scheduling", events[0]["message"])

    def test_the_backend_sends_nothing_reports_the_reason_and_stores_no_tokens(self):
        lab = lab_server.Lab(self.engine(self.held_by("exchange-scheduling")), Path(tempfile.mkdtemp()))
        run = lab.start("chat", [{"role": "user", "content": "hi"}], {}, {"same_start": 1}, False, wait=True)
        self.assertEqual(self.sent, [])
        self.assertEqual(run.status, "error")
        self.assertTrue(run.error.startswith("the cards are in use by 'exchange-scheduling'"), run.error)   # the reason, once
        self.assertEqual(run.notes, [])                                # and no "could not restore" beside it
        self.assertEqual(run.records, [])

    def test_nothing_is_sent_while_the_launcher_is_still_loading_or_setting_up(self):
        from datetime import datetime, timezone
        locks = self.held_by("routing-lab")
        receipt = Path(tempfile.mkdtemp()) / "launch.json"
        e = engines.MayaEngine(DEAD, {"one_row": {"spec": 1}, "drafting": {"spec": 8}}, locks=locks, receipt=receipt)
        base = {"launched_utc": datetime.now(timezone.utc).isoformat(), "server_pid": 1, "typical_load_s": 300}
        for extra, phase, word in (({}, "loading", "loading the model"), ({"ready_utc": "x"}, "setup", "being prepared")):
            receipt.write_text(json.dumps(dict(base, **extra)))
            info = e.info()
            self.assertEqual((info["connected"], info["loading"]["phase"], info["loading"]["typical_s"]), (False, phase, 300))
            self.assertIn(word, e.blocked())
            self.assertFalse(e.restore()["ok"])
            events = list(e.generate([{"role": "user", "content": "hi"}], schema.defaults(), schema.sampling_defaults(),
                                     False, None, threading.Event()))
            self.assertEqual([ev["type"] for ev in events], ["error"])
        self.assertEqual(self.sent, [])                                # not even a status query during the load
        receipt.write_text(json.dumps(dict(base, ready_utc="x", setup_done_utc="y")))
        self.assertIsNone(e.blocked())
        self.assertNotIn("loading", e.info())
        self.assertTrue(self.sent)                                     # once the setup is done the engine is asked

    def test_the_labs_own_engine_is_contacted(self):
        e = self.engine(self.held_by("routing-lab"))
        info = e.info()
        self.assertTrue(self.sent and self.sent[0].endswith("/v1/status"))    # the guard is not a blanket refusal
        self.assertNotIn("blocked", info)


class RequestBody(NoNetwork):
    def setUp(self):
        super().setUp()
        self.e = engines.MayaEngine(DEAD, {"one_row": {"spec": 1, "fresh_context": 1}, "drafting": {"spec": 8}})
        self.msgs = [{"role": "user", "content": "hi"}]

    def body(self, settings=None, sampling=None, drafting=False, force=None):
        return self.e.request_body(self.msgs, schema.validate(settings or {}), schema.validate_sampling(sampling or {}),
                                   drafting, force, "lab-x", 2, 64)

    def test_plain_bonus_on_an_engine_without_the_lab_uses_the_serving_kernel(self):
        tune = self.body({"routing_prior_bonus": 0.3})["strata_tune"]
        self.assertEqual(tune["routing_prior_bonus"], 0.3)
        self.assertNotIn("route_lab", tune)
        self.assertEqual((tune["spec"], tune["fresh_context"], tune["token_log"], tune["token_topk"]), (1, 1, "lab-x", 64))

    def test_the_lab_path_is_the_default_when_the_engine_has_it_and_can_be_switched_off_for_chat(self):
        self.e.has_lab = True
        self.assertEqual(self.body()["strata_tune"]["route_lab"], 1)
        self.assertNotIn("route_lab", self.body(sampling={"lab_path": 0})["strata_tune"])
        self.assertEqual(self.body(sampling={"lab_path": 0}, force=[1, 2])["strata_tune"]["route_lab"], 1)   # the meter

    def test_only_rules_that_differ_from_today_are_sent(self):
        tune = self.body({"route_floor": 0.8, "route_weights": 1, "route_layer_scale": [0.5] + [1.0] * 56}, drafting=True)["strata_tune"]
        self.assertEqual((tune["route_floor"], tune["route_weights"], tune["route_lab"], tune["spec"]), (0.8, 1, 1, 8))
        self.assertEqual(tune["route_layer_scale"], [0.5] + [1.0] * 56)
        for absent in ("route_protect", "route_priced", "route_k_drop", "route_k_widen", "route_bonus_draft", "route_prompt_bonus"):
            self.assertNotIn(absent, tune)
        self.assertEqual(tune["route_lab_prompt"], 1)                  # the rules apply to the prompt unless switched off

    def test_the_prompt_switch_is_a_lab_key_and_never_reaches_the_serving_kernel(self):
        self.assertNotIn("route_lab_prompt", self.body({"routing_prior_bonus": 0.3})["strata_tune"])   # no lab on this engine
        alone = self.body({"route_lab_prompt": 0})["strata_tune"]      # the switch alone does not ask for the lab path
        self.assertNotIn("route_lab_prompt", alone)
        self.assertNotIn("route_lab", alone)
        tune = self.body({"route_floor": 0.8, "route_lab_prompt": 0, "route_prompt_bonus": 0.2})["strata_tune"]
        self.assertEqual((tune["route_lab"], tune["route_lab_prompt"], tune["route_prompt_bonus"]), (1, 0, 0.2))

    def test_no_cap_is_sent_unless_the_reader_sets_one(self):
        self.assertEqual(schema.sampling_defaults()["max_tokens"], 0)
        self.assertNotIn("max_tokens", self.body())                                   # the model writes until it stops
        self.assertEqual(self.body(sampling={"max_tokens": 512})["max_tokens"], 512)
        self.assertNotIn("max_tokens", self.body(force=list(range(100))))             # nor on a forced run

    def test_forced_decoding(self):
        b = self.body(sampling={"max_tokens": 16}, force=list(range(100)))
        self.assertEqual(b["strata_tune"]["force_tokens"], list(range(100)))
        self.assertEqual(b["strata_tune"]["verify_segment"], 0)
        self.assertGreaterEqual(b["max_tokens"], 100)
        self.assertIs(b["loop_guard"], False)


class LabWithMock(unittest.TestCase):
    def setUp(self):
        self.lab = lab_server.Lab(engines.MockEngine(), Path(tempfile.mkdtemp()))
        self.msgs = [{"role": "user", "content": "Describe the pump inspection."}]

    def test_a_chat_run_is_stored_with_its_records(self):
        run = self.lab.start("chat", self.msgs, {}, {"max_tokens": 48}, False, wait=True)
        self.assertEqual(run.status, "done")
        self.assertEqual(len(run.records), 48)
        stored = self.lab.get_run(run.id)
        self.assertEqual(stored["stats"]["tokens"], 48)
        self.assertGreater(stored["stats"]["thinking_tokens"], 0)
        self.assertTrue(run.reasoning and run.text)

    def test_the_meter_orders_rules_and_its_noise_floor_is_zero(self):
        run = self.lab.start("chat", self.msgs, {}, {"max_tokens": 48}, False, wait=True)
        d = schema.defaults()
        job = self.lab.start_meter(run.id, [{"name": "b20", "settings": d}, {"name": "b40", "settings": dict(d, routing_prior_bonus=0.4)}],
                                   False, True, "")
        deadline = time.time() + 30
        while job["status"] == "running" and time.time() < deadline:
            time.sleep(0.05)
        self.assertEqual(job["status"], "done", job.get("error"))
        noise, b20, b40 = job["arms"]
        self.assertTrue(noise["noise_floor"])
        self.assertEqual(noise["summary"]["kl_mean"], 0.0)
        self.assertEqual(b20["compared"], 48)
        self.assertEqual(b20["mismatched_tokens"], 0)
        self.assertLess(b20["summary"]["kl_mean"], b40["summary"]["kl_mean"])
        self.assertTrue(job["notes"] and "restored" in job["notes"][0])        # every run started from the saved state

    def test_a_failed_routing_check_is_put_on_the_answer(self):
        class Bad(engines.MockEngine):
            def generate(self, *a, **k):
                for ev in super().generate(*a, **k):
                    if ev["type"] == "done":
                        ev["summary"]["route"] = {"kernel_mismatch_rows": 0, "identity_mismatch_rows": 3}
                    yield ev
        lab = lab_server.Lab(Bad(), Path(tempfile.mkdtemp()))
        run = lab.start("chat", self.msgs, {}, {"max_tokens": 24}, False, wait=True)
        self.assertTrue(any(n.startswith("ROUTING CHECK FAILED") and "identity_mismatch_rows 3" in n for n in run.notes))
        clean = self.lab.start("chat", self.msgs, {}, {"max_tokens": 24}, False, wait=True)
        self.assertFalse(any("FAILED" in n for n in clean.notes))

    def test_a_priced_time_is_stored_when_the_reader_offers_one_and_its_failure_is_a_note(self):
        class Reader:
            def read(self, path):
                return {"path": path}
            def price(self, capture):
                return {"ms_per_token": 31.5, "tokens_per_s": 31.7, "exact": True, "tokens": 24, "by_mode": {}}
        class Broken(Reader):
            def price(self, capture):
                raise ValueError("no pass counts")
        class Priced(engines.MockEngine):
            the_reader = Reader()
            def reader(self):
                return self.the_reader
            def generate(self, *a, **k):
                for ev in super().generate(*a, **k):
                    if ev["type"] == "done":
                        ev["summary"]["capture"] = "/nowhere/lab-x.jsonl"
                    yield ev
        lab = lab_server.Lab(Priced(), Path(tempfile.mkdtemp()))
        run = lab.start("chat", self.msgs, {}, {"max_tokens": 24}, False, wait=True)
        self.assertEqual(lab.get_run(run.id)["stats"]["price"], {"ms_per_token": 31.5, "tokens_per_s": 31.7, "exact": True, "tokens": 24})
        Priced.the_reader = Broken()
        run = lab.start("chat", self.msgs, {}, {"max_tokens": 24}, False, wait=True)
        self.assertEqual(run.status, "done")
        self.assertNotIn("price", run.stats())
        self.assertTrue(any("no pass counts" in n for n in run.notes))

    def test_presets_are_valid_settings(self):
        for name, settings in self.lab.presets()["built_in"].items():
            self.assertEqual(schema.validate(settings), settings, name)

    def test_an_answer_stored_by_an_older_version_gets_todays_summary(self):
        run = self.lab.start("chat", self.msgs, {}, {"max_tokens": 24}, False, wait=True)
        path = self.lab.data / "runs" / f"{run.id}.json"
        stored = json.loads(path.read_text())
        stored["stats"] = {"tokens": 24, "price": {"ms_per_token": 40.0}}        # as an older page saved it
        path.write_text(json.dumps(stored))
        fresh = lab_server.Lab(engines.MockEngine(), self.lab.data).get_run(run.id)
        self.assertEqual(fresh["stats"]["route"]["experts_per_layer"], 4.0)
        self.assertEqual(fresh["stats"]["finish"], "length")
        self.assertEqual(fresh["stats"]["price"], {"ms_per_token": 40.0})        # what only the old file had is kept

    def test_experts_per_layer_is_reported(self):
        d = schema.defaults()
        plain = self.lab.start("chat", self.msgs, d, {"max_tokens": 48}, False, wait=True)
        self.assertEqual(plain.stats()["route"]["experts_per_layer"], 4.0)
        fewer = self.lab.start("chat", self.msgs, dict(d, route_k_drop=0.15), {"max_tokens": 48}, False, wait=True)
        self.assertLess(fewer.stats()["route"]["experts_per_layer"], 4.0)
        per_token = [r["route"]["picks"] for r in fewer.records]
        self.assertTrue(any(p < 4 * schema.N_LAYERS for p in per_token))

    def test_what_each_rule_changed_is_summarised(self):
        d = schema.defaults()
        plain = self.lab.start("chat", self.msgs, d, {"max_tokens": 48}, False, wait=True).stats()["route"]
        self.assertGreater(plain["changed"]["bonus"], 0)                  # today's rule substitutes too
        self.assertEqual(plain["changed"]["floor"], 0)                    # a rule that is off changed nothing
        budget = self.lab.start("chat", self.msgs, dict(d, routing_prior_bonus=0.6, route_floor=0.9), {"max_tokens": 48}, False,
                                wait=True).stats()["route"]
        self.assertGreater(budget["changed"]["floor"], 0)
        self.assertGreater(budget["changed_tokens"]["floor"], 0)
        self.assertLessEqual(budget["changed_tokens"]["floor"], 1.0)
        # By hand: two tokens, the bonus at 10 and 0 layers: 5 a token on average, on half the tokens.
        s = compare.summarize([{"type": "token", "id": 1, "top": [[1, 1.0]], "entropy": 0.0, "route": {"picks": 228, "changed": {"bonus": 10}}},
                               {"type": "token", "id": 1, "top": [[1, 1.0]], "entropy": 0.0, "route": {"picks": 228, "changed": {"bonus": 0}}}])
        self.assertEqual((s["route"]["changed"]["bonus"], s["route"]["changed_tokens"]["bonus"]), (5.0, 0.5))
        self.assertNotIn("changed", compare.summarize([{"type": "token", "id": 1, "top": [[1, 1.0]], "entropy": 0.0, "route": {"picks": 228}}])["route"])

    def test_an_answer_cut_at_the_cap_says_so(self):
        cut = self.lab.start("chat", self.msgs, {}, {"max_tokens": 32}, False, wait=True)
        self.assertEqual((len(cut.records), cut.stats()["finish"]), (32, "length"))
        whole = self.lab.start("chat", self.msgs, {}, {}, False, wait=True)
        self.assertEqual(whole.stats()["finish"], "stop")
        self.assertGreater(len(whole.records), 32)

    def test_one_generation_at_a_time(self):
        first = self.lab.start("chat", self.msgs, {}, {"max_tokens": 64}, False)
        second = self.lab.start("chat", self.msgs, {}, {"max_tokens": 16}, False)
        self.assertEqual(second.status, "error")
        self.assertIn("busy", second.error)
        first.cancel.set()


if __name__ == "__main__":
    unittest.main(verbosity=1)
