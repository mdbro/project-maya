"""Tests of tools/mm3/native_reference_state.py:  cd tools/mm3 && python3 -B -m unittest test_native_reference_state"""
import copy
import json
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock

import native_calibrate
import native_reference_state as rs

PARSER = pathlib.Path(__file__).resolve().parents[2] / 'build-local' / 'mm3_reference_state_test'   # (the engine's own reader, when built)
ROW = [0.011, 0.02, 0.031, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17]
CAL = {"units": "host wall milliseconds, joined setup measurements", "arithmetic": "fp4", "temporary_pinned_bytes": 31850752,
       "h2d_ms": 0.59000701003242284, "d2h_ms": 0.60407700948417187, "cpu_rotating_records": 64, "primary_plausible": True,
       "cpu_base_ms": 0.31, "helper_base_ms": 0.22, "samples": [{"columns": 1, "cpu1_ms": 0.583, "cpu4_ms": 2.687}],
       "cpu_extra_ms": ROW, "gpu_extra_ms": ROW, "helper_extra_ms": ROW, "gpu_base_ms": ROW, "model": "additive grouped-service approximation; max of lanes"}


def reply(**changes) -> dict:
    """A PLACEMENT snapshot reply as the server returns it: 3 layers of 4 experts, the middle layer without experts."""
    r = {"checkpoint_version": 2, "action": "snapshot", "scope": "in-process only", "scores_normalized_at_generation_row": 5000, "snapshot_forward": 900,
         "staging": "cleared at joined boundary", "snapshot_generation_row": 5000, "admission_credits": [0.5, 1],
         "geometry": [3, 4, 6144, 3072, -1], "arithmetic": "fp4", "primary_adaptive": True,
         "policy": {"legacy": False, "horizon": 40, "max_exchanges": 8, "service_saved_ms": 0, "h2d_ms": 0, "d2h_ms": 0, "interference_ms": 0, "margin_ms": 0},
         "lane": {"cpu1": 0.4635, "cpu_add": 0.5560000000000001, "copy": 0.755, "cpu_price_scale": 1.5, "calibrated": True, "calibration_arithmetic": "fp4",
                  "retime": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]},
         "batch_staging_calibration": copy.deepcopy(CAL), "scores": [[0, 0, 12.5], [2, 1, 0.10000000149011612]], "adaptive_main": [[2, 1]],
         "placement": {"ownership_complete": True, "primary_ids": [[2, 1, 0, 140000], [0, 0, 0, 150000]], "helper_ids": [[2, 0, 1], [0, 1, 0]],
                       "third_roster": [[0, 2]], "primary_changed": 0, "helper_changed": 0, "duration_ms": 12.5},
         "engine_pid": 4321}
    r.update(changes)
    return r


def parse_text(text: str) -> dict:
    """The engine file read back the way src/mm3/mm3_reference_state.hpp reads it (the numbers as Python floats)."""
    head, _, rest = text.partition('BATCH_CALIBRATION ')
    n, _, rest = rest.partition('\n')
    body, tail = rest.encode()[:int(n)].decode(), rest.encode()[int(n):].decode()
    words = {}
    for line in head.splitlines() + tail.splitlines():
        if line.strip():
            key, _, value = line.partition(' ')
            words[key] = value.split()
    return dict(words=words, calibration=json.loads(body) if body else None)


class Content(unittest.TestCase):
    @unittest.skipUnless(PARSER.exists(), 'the engine parser is not built')
    def test_shared_policy_roundtrip_and_malformed_ages(self):
        price = dict(value=[[0.01, -0.02, 0.03], [0.04, 0.05, 0.06]], samples=[[1, 2, 3], [4, 5, 6]])
        r = reply(checkpoint_version=3, shared_host=dict(bytes=64 * 2**20, keys=[[0, 2], [2, 2]]),
                  free_time=dict(clock_rows=100, rate=[0.1, 0.2, 0.3], delay_rows=[20, 30], price_ms=[0.12, 0.23],
                                 price_samples=[2, 3], layers=[copy.deepcopy(price) for _ in range(3)],
                                 global_=price, last_move=[[0, 2, 72], [2, 2, 95]]))
        r['free_time']['global'] = r['free_time'].pop('global_')
        c = rs.content_from_reply(r)
        rs.ensure_reusable(c)
        self.assertEqual(rs.learned_signature(c), native_calibrate.learned_state_signature(r))
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / 'shared.mm3ref'
            text = rs.engine_text(c)
            p.write_text(text)
            result = subprocess.run([str(PARSER), str(p), '101', '4'], text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertIn('shared 67108864 2 10', result.stdout)
            self.assertIn('free_clock 100', result.stdout)
            self.assertIn('free_last 0 2 72 2 2 95', result.stdout)
            for before, after in [('FREE_CLOCK 100', 'FREE_CLOCK -1'), ('FREE_CLOCK 100', 'FREE_CLOCK 18446744073709551616'),
                                  ('FREE_LAST 2 2 72 10 95', 'FREE_LAST 2 2 72 10 101'),
                                  ('SHARED_HOST 67108864 2 2 10', 'SHARED_HOST 67108864 2 2 1')]:
                self.assertIn(before, text)
                p.write_text(text.replace(before, after))
                self.assertEqual(subprocess.run([str(PARSER), str(p), '101', '4'], capture_output=True).returncode, 1)
        for mutate in (lambda x: x['free_time'].pop('global'), lambda x: x['free_time']['last_move'].append([0, 2, 80]),
                       lambda x: x['shared_host']['keys'].append([0, 1]), lambda x: x['free_time'].update(clock_rows=-1)):
            bad = copy.deepcopy(c); mutate(bad)
            with self.assertRaises(ValueError): rs.validate(bad)

    def test_reusable_state_requires_final_compatible_calibration(self):
        for change in ('uncalibrated', 'no_grouped', 'wrong_arithmetic', 'legacy'):
            c = rs.content_from_reply(reply())
            if change == 'uncalibrated':
                c['lane']['calibrated'] = False
            elif change == 'no_grouped':
                c['batch_staging_calibration'] = None
            elif change == 'wrong_arithmetic':
                c['batch_staging_calibration']['arithmetic'] = 'dp4a'
            else:
                del c['checkpoint_version']
            with self.subTest(change=change), self.assertRaises(ValueError):
                rs.ensure_reusable(c)

    def test_legacy_identity_remains_readable_but_cannot_qualify_new_runs(self):
        r = reply()
        del r['checkpoint_version']
        c = rs.content_from_reply(r)
        self.assertEqual(rs.learned_signature(c), native_calibrate.learned_state_signature(r))
        with tempfile.TemporaryDirectory() as directory:
            p = pathlib.Path(directory) / 'legacy.json'
            rs.write_state(p, c, {}, bootstrap=True)
            self.assertEqual(rs.read_state(p)['content'], c)
            with self.assertRaises(ValueError):
                rs.apply('unused', p)

    def test_the_content_and_its_identity(self):
        c = rs.content_from_reply(reply())
        self.assertEqual(c['primary'], [[2, 1], [0, 0]])                       # (the reply's order: the 5090's slots)
        self.assertEqual(c['helper'], [[0, 1], [2, 0]])                        # (by the 3090's slot)
        self.assertEqual(c['third'], [[0, 2]])
        self.assertEqual(rs.identity(c), rs.identity(rs.content_from_reply(reply(engine_pid=9, snapshot_forward=1))))   # (nothing of one start's)
        for change in (dict(admission_credits=[0.5, 2]), dict(scores=[[0, 0, 12.5]]), dict(arithmetic='dp4a')):
            self.assertNotEqual(rs.identity(c), rs.identity(rs.content_from_reply(reply(**change))))
        lane = copy.deepcopy(reply()['lane']); lane['cpu1'] = 0.59
        self.assertNotEqual(rs.identity(c), rs.identity(rs.content_from_reply(reply(lane=lane))))                      # (a start's own lane costs)
        moved = reply(); moved['placement']['third_roster'] = [[0, 3]]
        self.assertNotEqual(rs.identity(c), rs.identity(rs.content_from_reply(moved)))                                 # (the 4080's roster)

    def test_the_learned_signature_is_the_ledgers(self):
        r = reply()
        c = rs.content_from_reply(r)
        self.assertEqual(rs.learned_signature(c), native_calibrate.learned_state_signature(r))
        moved = reply(); moved['placement']['third_roster'] = [[0, 3]]                                                 # version 2 covers the 4080 too
        self.assertNotEqual(rs.learned_signature(rs.content_from_reply(moved)), native_calibrate.learned_state_signature(r))

    def test_a_reply_that_is_not_a_complete_state_is_refused(self):
        bad = reply(); bad['placement']['ownership_complete'] = False
        for r in (reply(scope='durable'), bad, {k: v for k, v in reply().items() if k != 'scores'}):
            with self.assertRaises(ValueError):
                rs.content_from_reply(r)
        old = reply(); del old['placement']['third_roster']                                                            # (an engine before third_roster)
        with self.assertRaises(ValueError):
            rs.content_from_reply(old)
        self.assertEqual(rs.content_from_reply(old, third=[[0, 2]])['third'], [[0, 2]])
        shared = reply(); shared['placement']['third_roster'] = [[0, 0]]                                               # (on the 5090 and the 4080)
        with self.assertRaises(ValueError):
            rs.content_from_reply(shared)


class EngineFile(unittest.TestCase):
    def test_the_engine_file_holds_every_number_unchanged(self):
        c = rs.content_from_reply(reply())
        ident = rs.identity(c)
        got = parse_text(rs.engine_text(c))
        w = got['words']
        self.assertEqual(w['MM3_REFERENCE_STATE'], ['1'])
        self.assertEqual(w['ID'], [ident])
        self.assertEqual(w['GEOMETRY'], ['3', '4', '6144', '3072', '-1'])
        self.assertEqual((w['ARITHMETIC'], w['CALIBRATION_ARITHMETIC'], w['CALIBRATED'], w['PRIMARY_ADAPTIVE'], w['BATCH_ARITHMETIC']), (['fp4'], ['fp4'], ['1'], ['1'], ['fp4']))
        self.assertEqual([float(x) for x in w['CREDITS']], [0.5, 1.0])
        self.assertEqual([float(x) for x in w['LANE']], [0.4635, 0.5560000000000001, 0.755, 1.5])
        self.assertEqual(w['POLICY'], ['0', '40', '8', '0', '0', '0', '0', '0'])
        self.assertEqual([float(x) for x in w['COSTS']], [1.0, 0.31, 0.22, 0.59000701003242284, 0.60407700948417187])
        for key in ('COSTS_CPU_EXTRA', 'COSTS_GPU_EXTRA', 'COSTS_HELPER_EXTRA', 'COSTS_GPU_BASE'):
            self.assertEqual([float(x) for x in w[key]], ROW)
        self.assertEqual(w['PRIMARY'], ['2', '9', '0'])                        # (count, then layer * experts + expert in slot order)
        self.assertEqual(w['HELPER'], ['2', '1', '8'])
        self.assertEqual(w['THIRD'], ['1', '2'])
        self.assertEqual(w['SCORES'][:4], ['2', '0', '0', '12.5'])
        self.assertEqual(float(w['SCORES'][6]), 0.10000000149011612)           # (a float32 score, as the engine printed it)
        self.assertEqual(w['ADAPTIVE_MAIN'], ['1', '9'])
        self.assertIn('END', w)
        self.assertEqual(got['calibration'], CAL)                              # (the grouped table's report, value for value)

    @unittest.skipUnless(PARSER.exists(), 'the engine\'s parser test is not built (build-local/mm3_reference_state_test)')
    def test_the_engine_reads_the_file_the_tool_writes(self):
        c = rs.content_from_reply(reply())
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'ref.mm3ref'
            path.write_text(rs.engine_text(c))
            got = subprocess.run([str(PARSER), str(path), '101', '4'], capture_output=True, text=True)
            self.assertEqual(got.returncode, 0, got.stdout)
            lines = dict(line.split(' ', 1) if ' ' in line else (line, '') for line in got.stdout.splitlines() if not line.startswith(('row', 'ids')))
            rows = [line.split()[1:] for line in got.stdout.splitlines() if line.startswith('row')]
            ids = [line.split()[1:] for line in got.stdout.splitlines() if line.startswith('ids')]
            self.assertEqual(lines['id'], rs.identity(c))
            self.assertEqual(lines['geometry'], '3 4 6144 3072 -1')
            self.assertEqual(lines['arithmetic'], 'fp4|fp4|fp4')
            self.assertEqual(lines['flags'], '1 1')
            self.assertEqual([float(x) for x in lines['credits'].split()], [0.5, 1.0])
            self.assertEqual([float(x) for x in lines['lane'].split()], [0.4635, 0.5560000000000001, 0.755, 1.5])
            self.assertEqual([float(x) for x in lines['costs'].split()], [1.0, 0.31, 0.22, 0.59000701003242284, 0.60407700948417187])
            self.assertEqual([[float(x) for x in r] for r in rows], [ROW] * 4)
            self.assertEqual(json.loads(lines['calibration']), CAL)
            self.assertEqual(ids, [['9', '0'], ['1', '8'], ['2'], ['9']])
            self.assertEqual(lines['scores'].split()[:3], ['0', '0', '12.5'])
            self.assertEqual(float(lines['scores'].split()[5]), 0.10000000149011612)
            path.write_text(rs.engine_text(c).replace('THIRD 1 2', 'THIRD 1 9'))                                        # (an expert on two cards: refused)
            self.assertEqual(subprocess.run([str(PARSER), str(path), '101', '4'], capture_output=True, text=True).returncode, 1)

    def test_a_state_without_a_grouped_table_or_a_third_card(self):
        r = reply(batch_staging_calibration=None); r['placement']['third_roster'] = []
        got = parse_text(rs.engine_text(rs.content_from_reply(r)))
        self.assertEqual(got['words']['COSTS'], ['0', '0', '0', '0', '0'])
        self.assertEqual(got['words']['THIRD'], ['0'])
        self.assertIsNone(got['calibration'])

    def test_numbers_are_written_as_the_engine_prints_them(self):
        self.assertEqual(rs._g(0.59000701003242284), '0.59000701003242284')
        self.assertEqual(rs._g(1), '1'); self.assertEqual(rs._g(1.0), '1'); self.assertEqual(rs._g(True), '1')
        self.assertEqual(rs._compact({'a': [1, 0.5, True, None, 'x y']}), '{"a":[1,0.5,true,null,"x y"]}')


class Files(unittest.TestCase):
    def test_a_stored_state_is_read_back_checked(self):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'ref.json'
            doc = rs.write_state(path, rs.content_from_reply(reply()), dict(created='now', how='test'))
            self.assertEqual(rs.read_state(path)['sha256'], doc['sha256'])
            with self.assertRaises(FileExistsError):                                                                    # (never overwritten)
                rs.write_state(path, rs.content_from_reply(reply()), {})
            text = rs.text_path(path).read_text()
            rs.text_path(path).write_text(text.replace('CREDITS 0.5 1', 'CREDITS 0.5 2'))                               # (the engine file edited)
            with self.assertRaises(ValueError):
                rs.read_state(path)
            rs.text_path(path).write_text(text)
            edited = json.loads(path.read_text()); edited['content']['admission_credits'] = [0.5, 2]
            path.write_text(json.dumps(edited))                                                                         # (the content edited)
            with self.assertRaises(ValueError):
                rs.read_state(path)

    def test_what_two_states_differ_in(self):
        a = rs.content_from_reply(reply())
        lane = copy.deepcopy(reply()['lane']); lane['cpu1'] = 0.59
        moved = reply(lane=lane); moved['placement']['third_roster'] = [[0, 3]]
        d = rs.differences(a, rs.content_from_reply(moved))
        self.assertEqual(len(d), 2)
        self.assertTrue(any(x.startswith('third roster') for x in d) and any(x.startswith('lane cpu1') for x in d))
        self.assertEqual(rs.differences(a, a), [])


class Apply(unittest.TestCase):
    def run_apply(self, engine_reply, own=False):
        with tempfile.TemporaryDirectory() as d:
            path = pathlib.Path(d) / 'ref.json'
            doc = rs.write_state(path, rs.content_from_reply(reply()), dict(created='now', how='test'))
            calls = []

            def post(base, endpoint, body, timeout=900):
                calls.append(body)
                if body['action'] == 'snapshot':
                    lane = copy.deepcopy(reply()['lane']); lane['cpu1'] = 0.59; lane['calibrated'] = False
                    return reply(lane=lane)
                self.assertEqual(body, {'action': 'reference', 'state_path': str(rs.text_path(path).resolve())})
                return engine_reply(doc)
            with mock.patch.object(rs, 'post', post), mock.patch.object(rs, 'engine_facts', lambda pid, base, timeout=30: dict(pid=pid, exe_sha256='ab' * 32)):
                record = rs.apply('http://x', path, pathlib.Path(d) / 'own.json' if own else None)
                if own:
                    self.assertEqual(rs.read_state(pathlib.Path(d) / 'own.json')['content']['lane']['cpu1'], 0.59)
            return record, calls, doc

    def test_the_engine_took_the_state(self):
        good = lambda doc: reply(action='restore', reference_state=doc['sha256'], third_ids=[[0, 2]], third_written=0, engine_pid=77)
        record, calls, doc = self.run_apply(good, own=True)
        self.assertEqual([c['action'] for c in calls], ['snapshot', 'reference'])
        self.assertEqual((record['sha256'], record['engine']['pid'], record['errors']), (doc['sha256'], 77, []))
        self.assertEqual(record['own']['costs']['cpu_one_expert_ms'], 0.59)                                             # (this start's own measurement, kept)
        self.assertEqual(record['own']['same_rosters'], dict(primary=True, helper=True, third=True))
        self.assertEqual(record['costs']['cpu_one_expert_ms'], 0.4635)

    def test_a_reply_in_another_state_is_an_error(self):
        lane = copy.deepcopy(reply()['lane']); lane['cpu1'] = 0.59
        for bad in (lambda doc: reply(reference_state='0' * 64, third_ids=[[0, 2]]),                                    # another identity
                    lambda doc: reply(reference_state=doc['sha256'], placement={**reply()['placement'], 'third_roster': [[0, 3]]}),                               # the 4080 holds something else
                    lambda doc: reply(reference_state=doc['sha256'], third_ids=[[0, 2]], lane=lane)):                   # the start's own lane costs stayed
            with self.assertRaises(RuntimeError):
                self.run_apply(bad)


class LedgerStep(unittest.TestCase):
    """native_followup's step of kind "reference": the stored state becomes the phase's origin and the ledger keeps its identity."""
    def step(self, tmp, changes=None, sha=True, own=None):
        import types
        import native_followup as followup
        path = pathlib.Path(tmp) / 'ref.json'
        doc = rs.write_state(path, rs.content_from_reply(reply()), dict(created='now', how='test'))
        run = pathlib.Path(tmp) / 'run'; run.mkdir()
        ledger = dict(phases=dict(standalone=dict(precheck=dict(pid=77))))
        fake = types.SimpleNamespace(base='x', a=types.SimpleNamespace(timeout=1), directory=run, ledger=ledger)

        def post(base, endpoint, body, timeout=900):
            return reply(action='restore', reference_state=doc['sha256'], third_ids=[[0, 2]], third_written=0, engine_pid=77, **(changes or {}))
        step = dict(step='setup-reference', phase='standalone', kind='reference', state=str(path))
        if sha:
            step['sha256'] = doc['sha256'] if sha is True else sha
        if own:
            step['own'] = own
        with mock.patch.object(rs, 'post', post), mock.patch.object(rs, 'engine_facts', lambda pid, base, timeout=30: dict(pid=pid, started=5, exe_sha256='ab' * 32, args=['--serve'])):
            return followup.FollowUp.run_reference(fake, step), ledger, doc, run

    def test_the_phase_takes_the_state_as_its_origin(self):
        with tempfile.TemporaryDirectory() as d:
            summary, ledger, doc, run = self.step(d, own='own-state.json')
            phase = ledger['phases']['standalone']
            self.assertEqual(summary['errors'], [])
            self.assertEqual(phase['origin_signature'], doc['learned_signature'])                                       # (what each later restore is checked against)
            self.assertEqual(phase['reference_state']['sha256'], doc['sha256'])
            self.assertEqual(phase['reference_state']['engine']['exe_sha256'], 'ab' * 32)
            self.assertEqual(native_calibrate.learned_state_signature(reply(action='restore')), phase['origin_signature'])
            self.assertTrue((run / 'setup-reference.json').exists() and (run / 'own-state.json').exists())

    def test_another_identity_than_the_protocols_is_refused_before_anything_is_sent(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(RuntimeError):
                self.step(d, sha='0' * 64)

    def test_an_origin_snapshot_afterwards_moves_the_identity_aside(self):
        import types
        import native_followup as followup
        from unittest import mock as m
        with tempfile.TemporaryDirectory() as d:
            _, ledger, doc, run = self.step(d)
            fake = types.SimpleNamespace(base='x', a=types.SimpleNamespace(timeout=1), directory=run, ledger=ledger)
            with m.patch('native_shared_diagnostic.post', lambda base, endpoint, body, timeout: reply(admission_credits=[1, 1])):
                followup.FollowUp.run_origin(fake, 'standalone')
            phase = ledger['phases']['standalone']
            self.assertNotIn('reference_state', phase)
            self.assertEqual(phase['reference_state_before_origin']['sha256'], doc['sha256'])
            self.assertNotEqual(phase['origin_signature'], doc['learned_signature'])


if __name__ == '__main__':
    unittest.main()
