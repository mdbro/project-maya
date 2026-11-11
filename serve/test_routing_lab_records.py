"""The routing lab's token-record keys (runs/routing-lab-20261110/API.md, section 1): strata_tune.token_log,
token_topk and force_tokens are refused when malformed or when the engine does not take them, reach the GEN line only
for an engine that says so, leave the line as before when absent, and /v1/status reports where the records go."""
import pathlib
import types
import unittest

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, check_record_keys

ROOT = pathlib.Path(__file__).resolve().parents[1]
ON = {"engine_kind": "mm3", "token_records_control": 1, "token_records_dir": "/tmp/records", "force_tokens_control": 1}
OFF = {"engine_kind": "mm3", "token_records_control": 0, "token_records_dir": "none", "force_tokens_control": 0}


class RecordKeys(unittest.TestCase):
    def test_refusals(self):
        good = {"token_log": "lab-3f2a9c_1.x", "token_topk": 64, "force_tokens": [5, 0, 199999]}
        check_record_keys(good, ON)
        check_record_keys({}, OFF)
        check_record_keys(None, OFF)
        for bad in ({"token_log": "a/b"}, {"token_log": ".."}, {"token_log": ""}, {"token_log": "x" * 161},
                    {"token_log": 7}, {"token_log": "a", "token_topk": 0}, {"token_log": "a", "token_topk": 257},
                    {"token_log": "a", "token_topk": True}, {"token_topk": 8}, {"force_tokens": []},
                    {"force_tokens": [1, -1]}, {"force_tokens": [1, 2.0]}, {"force_tokens": [True]},
                    {"force_tokens": "1,2"}, {"force_tokens": [0] * 65537}):
            with self.assertRaises(ValueError, msg=str(bad)[:60]):
                check_record_keys(bad, ON)
        for needs in ({"token_log": "a"}, {"force_tokens": [1]}):
            with self.assertRaises(ValueError):
                check_record_keys(needs, OFF)
            with self.assertRaises(ValueError):
                check_record_keys(needs, {"engine_kind": "mm3"})   # (an engine from before: no controls at all)

    def test_gen_line(self):
        tune = {"token_log": "lab-1", "token_topk": 32, "force_tokens": [3, 1, 4]}
        line = StrataEngine.sampling_keys({"temperature": 1.0, "strata_tune": tune}, "mm3", ON)
        self.assertIn(" token_log=lab-1 token_topk=32 force_tokens=3,1,4", line)
        greedy = StrataEngine.sampling_keys({"temperature": 0, "strata_tune": tune}, "mm3", ON)
        self.assertIn(" token_log=lab-1 token_topk=32 force_tokens=3,1,4", greedy)
        self.assertNotIn("token_log", StrataEngine.sampling_keys({"strata_tune": tune}, "mm3", OFF))
        self.assertNotIn("force_tokens", StrataEngine.sampling_keys({"strata_tune": tune}, "mm3", OFF))
        # absent: the line is the one it was
        plain = {"temperature": 1.0, "top_p": 0.95, "strata_tune": {"routing_prior_bonus": 0.2}}
        with_none = StrataEngine.sampling_keys(plain, "mm3", ON)
        self.assertEqual(with_none, StrataEngine.sampling_keys(plain, "mm3", OFF))
        self.assertNotIn("token", with_none)

    def test_check_request_and_status(self):
        tok = ByteTokenizer()
        engine = MockEngine(tok, "ok", max_context=8192)
        engine.proc = types.SimpleNamespace(pid=1, poll=lambda: None)
        svc = Service(engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        req = {"messages": [{"role": "user", "content": "hi"}], "strata_tune": {"token_log": "lab-2"}}
        engine.info = dict(OFF)
        with self.assertRaises(ValueError):
            svc.check_request(dict(req))
        self.assertIsNone(svc.v1_status()["token_records_dir"])
        engine.info = dict(ON)
        svc.check_request(dict(req))
        self.assertEqual(svc.v1_status()["token_records_dir"], "/tmp/records")


if __name__ == "__main__":
    unittest.main()
