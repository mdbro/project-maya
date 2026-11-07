#!/usr/bin/env python3
"""Campaign sequencing, state checks, cap and reserve rules, with a fake engine."""
import json
import pathlib
import tempfile
import types
import unittest
from unittest import mock

import native_light as light
import native_shared_diagnostic as diagnostic

ROOT = pathlib.Path(__file__).resolve().parents[2]
PROTOCOL = ROOT / 'docs/minimax-wsl/shared-model-diagnostic-protocol.json'


def source_capture(forwards=70):
    capture = light.LightCapture(pathlib.Path('source.light'), 'source-sha', {})
    def layer(forward, rows):
        return dict(type='layer', forward=forward, layer=3, rows=rows, original=[0, 1, 2, 3] * rows, executed=[0, 1, 2, 3] * rows)
    capture.records += [layer(1, 16), dict(type='forward', forward=1, position=0, rows=16, generation=False)]
    for i in range(forwards):
        capture.records += [layer(i + 2, 1), dict(type='forward', forward=i + 2, position=16 + i, rows=1, generation=True)]
    return capture


class FakeEngine:
    """Answers controls and requests; a request's context writes nothing."""
    def __init__(self):
        self.calls, self.signature, self.fail_steps, self.scores = [], 'origin', set(), [[3, 1, 1.0]]

    def checkpoint(self, action):
        return {'action': action, 'scope': 'in-process only', 'staging': 'cleared at joined boundary',
                'placement': {'ownership_complete': True, 'primary_ids': [[3, 1, 0, 1]], 'helper_ids': [[3, 2, 0]]},
                'geometry': [60, 128, 1, 1, 437], 'arithmetic': 'fp4', 'primary_adaptive': True, 'policy': {}, 'lane': {},
                'scores': self.scores, 'adaptive_main': [], 'batch_staging_calibration': None, 'admission_credits': [0, 0],
                'engine_pid': 4242}

    def post(self, base, endpoint, body, timeout):
        action = body['action']
        self.calls.append(action)
        if action == 'inspect':
            return {'ownership_complete': True, 'engine_pid': 4242}
        if action in ('snapshot', 'restore'):
            return self.checkpoint(action)
        if action == 'helper-first':
            assert body.get('profile_path'), 'placement requires the declared profile'
            return {'ownership_complete': True, 'engine_pid': 4242, 'applied_profile_sha256': 'profile'}
        if action == 'replay':
            held = 'held' in body['workload_path']
            primary_held = 'held-primary' in body['workload_path'] or body['workload_path'].endswith('fixture-held.replay')
            helper_held = 'held-helper' in body['workload_path'] or body['workload_path'].endswith('fixture-held.replay')
            return {'generation_batches': 64, 'generation_rows': 64, 'generation_target_ms': 5000.0,
                    'hold_primary_admissions': primary_held, 'hold_helper_admissions': helper_held,
                    'primary_issued': 0 if primary_held else 30, 'helper_started': 0 if helper_held else 12}
        raise AssertionError(action)

    def run_case(self, base, case, payload, timeout, raw, result, pid, start, context, trace_stream=True):
        self.calls.append('request:' + case['id'])
        assert trace_stream is False, 'campaign requests must not trace the stream'
        final = dict(emitted=384, elapsed_ms=25000.0, draft_off=0, spec_rounds=0, target_rounds=384)
        result['requests'].append(dict(complete=True, finish_reason='length', usage={}, timings={'cache_n': 0}, content='x',
                                       reasoning_content='y', api_wall_ms=26000.0,
                                       engine_stats=dict(join='one_new_sequence_and_matching_completion_count', final=final)))
        if context:
            context['manifest']['requests'].append(dict(engine_jsonl='source.light', validation=dict(status='validated')))


def fake_capture_init(namespace, output, pid):
    return dict(manifest=dict(requests=[], control_operations=[], mode=namespace.capture_mode), directory=output.parent, control=namespace.capture_control)


def fake_operation(context, name, callback):
    context['manifest']['control_operations'].append(dict(engine_jsonl='fixture.light', validation=dict(status='validated')))
    return callback()


class Campaign(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = pathlib.Path(self.temp.name)
        self.protocol = json.loads(PROTOCOL.read_text())
        self.protocol_path = self.directory / 'protocol.json'
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.engine = FakeEngine()
        patches = [mock.patch.object(diagnostic, 'post', self.engine.post),
                   mock.patch.object(diagnostic.screen, 'run_case', self.engine.run_case),
                   mock.patch.object(diagnostic.capture, 'init', fake_capture_init),
                   mock.patch.object(diagnostic.capture, 'operation', fake_operation),
                   mock.patch.object(diagnostic, 'precheck', lambda phase, pid, control: dict(pid=pid, executable_sha256='exe', errors=[])),
                   mock.patch.object(diagnostic, 'gpu_sample', lambda: 'gpu'),
                   mock.patch.object(diagnostic.light, 'read', lambda path: source_capture())]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.temp.cleanup)

    def campaign(self, **kwargs):
        args = dict(protocol=self.protocol_path, run_dir=self.directory, phase='standalone', base_url='http://x', timeout=5.0,
                    continue_after_failure=False)
        args.update(kwargs)
        return diagnostic.Campaign(types.SimpleNamespace(**args))

    def test_standalone_sequence_runs_declared_steps_in_order(self):
        self.assertEqual(self.campaign().run(), 0)
        ledger = json.loads((self.directory / 'ledger.json').read_text())
        names = [s['step'] for s in ledger['steps']]
        self.assertEqual(names[:5], ['bootstrap', 'helper-first', 'conditioning-code', 'conditioning-prose', 'origin'])
        self.assertEqual(names[-1], 'state-normal-2')
        self.assertFalse(any(n.startswith('trace') for n in names))
        self.assertEqual(sum(1 for s in ledger['steps'] if s['kind'] == 'request' and s['step'] != 'bootstrap'), 8)
        self.assertEqual(sum(1 for s in ledger['steps'] if s['kind'] == 'fixture'), 8)
        self.assertEqual(ledger['source_capture']['step'], 'long-python-code-light')

    def test_conditioning_precedes_any_restore_and_every_arm_restores(self):
        self.campaign().run()
        calls = self.engine.calls
        self.assertLess(calls.index('snapshot'), calls.index('restore'))
        self.assertNotIn('restore', calls[:calls.index('snapshot')])
        # Each of the 6 long requests and 8 fixture runs follows its own restore.
        self.assertEqual(calls.count('restore'), 14)
        replay_and_requests = [i for i, c in enumerate(calls) if c == 'replay' or c.startswith('request:python') or c.startswith('request:prose')]
        self.assertTrue(all(calls[i - 1] == 'restore' for i in replay_and_requests))

    def test_held_arms_use_the_held_workload(self):
        self.campaign().run()
        fixture = (self.directory / 'fixture-held.replay').read_text().splitlines()[0]
        self.assertEqual(fixture, 'MM3_REPLAY 3 8 4 1.3 0 0 1 1')
        control = (self.directory / 'fixture-control.replay').read_text().splitlines()[0]
        self.assertEqual(control, 'MM3_REPLAY 3 8 4 1.3 0 0 0 0')

    def test_fixture_selects_coding_when_long_enough_and_prose_otherwise(self):
        self.campaign().run()
        lines = (self.directory / 'fixture-control.replay').read_text().splitlines()
        self.assertEqual(sum(1 for x in lines if x.startswith('F 1 ') and x.endswith(' 1')), 64)
        self.assertEqual(sum(1 for x in lines if x.startswith('F 16 ')), 1)

    def test_restored_state_mismatch_stops_the_campaign(self):
        campaign = self.campaign()
        for step in self.protocol['sequence']:
            if step['step'] == 'long-python-code-off-1':
                break
            campaign.begin_phase('standalone')
            self.assertTrue(campaign.execute(step, 4242, self.protocol['phases']['standalone']))
        self.engine.scores = [[3, 1, 2.0]]
        ok = campaign.execute(step, 4242, self.protocol['phases']['standalone'])
        self.assertFalse(ok)
        self.assertIn('differs from the declared origin', campaign.ledger['steps'][-1]['summary']['errors'][0])

    def test_cap_stops_before_the_next_step(self):
        self.protocol['warmed_time_cap_seconds'] = 0
        self.protocol_path.write_text(json.dumps(self.protocol))
        self.assertEqual(self.campaign().run(), 1)
        ledger = json.loads((self.directory / 'ledger.json').read_text())
        self.assertEqual(ledger['stopped']['before_step'], 'bootstrap')
        self.assertEqual(ledger['steps'], [])

    def test_failed_step_blocks_silent_continuation(self):
        campaign = self.campaign()
        with mock.patch.object(campaign, 'run_step', side_effect=RuntimeError('boom')):
            self.assertEqual(campaign.run(), 1)
        self.assertEqual(self.campaign().run(), 1)
        self.assertEqual(self.engine.calls.count('request:bootstrap'), 0)

    def test_speculative_work_fails_the_step(self):
        original = self.engine.run_case
        def speculative(*args, **kwargs):
            original(*args, **kwargs)
            args[5]['requests'][-1]['engine_stats']['final']['draft_off'] = 4
        with mock.patch.object(diagnostic.screen, 'run_case', speculative):
            self.assertEqual(self.campaign().run(), 1)
        ledger = json.loads((self.directory / 'ledger.json').read_text())
        self.assertIn('speculation not off', ledger['steps'][0]['summary']['errors'][0])

    def test_changing_protocol_after_start_is_refused(self):
        self.campaign().run()
        self.protocol['seed'] = 1
        self.protocol_path.write_text(json.dumps(self.protocol))
        with self.assertRaises(ValueError):
            self.campaign()

    def test_reserve_requires_a_named_ambiguity_and_a_limit(self):
        self.campaign().run()
        args = dict(variant=['held-primary'], capture='light', ambiguity='too short')
        with self.assertRaises(ValueError):
            self.campaign(**args).reserve()
        args['ambiguity'] = 'Separate primary from helper after a material grouped admissions result'
        args['variant'] = ['held-primary'] * 5
        with self.assertRaises(ValueError):
            self.campaign(**args).reserve()
        args['variant'] = ['held-primary', 'held-helper']
        self.assertEqual(self.campaign(**args).reserve(), 0)
        args['variant'] = ['control', 'held-primary', 'held-helper']
        with self.assertRaises(ValueError):
            self.campaign(**args).reserve()
        ledger = json.loads((self.directory / 'ledger.json').read_text())
        self.assertEqual(ledger['reserve_used'], 2)
        self.assertEqual(ledger['steps'][-1]['ambiguity'], args['ambiguity'])

    def test_protocol_declares_the_agreed_counts(self):
        steps = self.protocol['sequence']
        setup = {'bootstrap', 'trace-bootstrap', 'trace-conditioning-code', 'trace-conditioning-prose'}
        measured = [s for s in steps if s['kind'] in ('request', 'fixture') and s['step'] not in setup]
        self.assertEqual(len(measured), self.protocol['planned_executions']['total'])
        self.assertEqual(self.protocol['max_tokens'], 384)
        self.assertNotIn('PENDING', json.dumps(self.protocol['conditioning']))
        scored = {c['prompt'] for c in json.loads(diagnostic.screen.FIXTURE.read_text())['cases']}
        self.assertFalse(scored & {c['prompt'] for c in self.protocol['conditioning']}, 'conditioning must not train on scored prompts')
        self.assertEqual(self.protocol['tune']['spec'], 1)
        self.assertEqual(self.protocol['fixture']['generation_forwards'], 64)
        order = [s['step'] for s in steps if s['step'].startswith('state-')]
        self.assertEqual(order, ['state-normal-1', 'state-held-1', 'state-held-2', 'state-normal-2'])
        order = [s['capture'] for s in steps if s['step'].startswith('monitor-')]
        self.assertEqual(order, ['off', 'light', 'light', 'off'])


if __name__ == '__main__':
    unittest.main()
