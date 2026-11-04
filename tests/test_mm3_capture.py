import importlib.util
import json
import pathlib
import tempfile
import unittest
import os
import sys
import hashlib
import contextlib
import io
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('native_capture', pathlib.Path(__file__).resolve().parents[1] / 'tools/mm3/native_capture.py')
capture = importlib.util.module_from_spec(spec)
spec.loader.exec_module(capture)


class CaptureValidationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = pathlib.Path(self.tmp.name) / 'request.jsonl'
        self.ids = {'experiment_id': 'experiment', 'run_id': 'run', 'request_id': 'request'}

    def write(self, dropped=0, token=3, count=1):
        records = [dict(self.ids, type='capture_begin', event_seq=1),
                   dict(self.ids, type='token_output', event_seq=2, token=token, output_index=0),
                   dict(self.ids, type='capture_end', event_seq=3, status='done'),
                   dict(self.ids, type='capture_footer', recorded_events=3, attempted_events=3+dropped, dropped_events=dropped)]
        self.path.write_text(''.join(json.dumps(x) + '\n' for x in records))
        return capture.validate(self.path, self.ids, {'completion_tokens': count}, True)

    def test_validated_and_mismatch(self):
        self.assertEqual(self.write()['status'], 'validated')
        self.assertIn('emitted/API completion count mismatch', self.write(count=2)['errors'])
        self.assertIn('invalid output token/index', self.write(token=None)['errors'])

    def test_loss_and_missing_footer(self):
        self.assertIn('dropped events', self.write(dropped=1)['errors'])
        self.path.write_text('{}\n')
        self.assertIn('capture footer missing', capture.validate(self.path, self.ids, None, False)['errors'])

    def test_per_class_drops_are_reported(self):
        self.write()
        rows = [json.loads(x) for x in self.path.read_text().splitlines()]
        rows[-1].update(attempted_events=4, dropped_events=1, timeline_dropped_events=0, detail_dropped_events=1)
        self.path.write_text(''.join(json.dumps(x) + '\n' for x in rows))
        result = capture.validate(self.path, self.ids, {'completion_tokens': 1}, True)
        self.assertIn('detail events dropped', result['errors'])
        self.assertEqual(result['coverage']['timeline_dropped_events'], 0)

    def test_missing_artifact_is_partial(self):
        self.assertEqual(capture.validate(self.path, self.ids, None, False)['status'], 'partial')


class HelperTrafficTest(unittest.TestCase):
    def test_static_inference_traffic_and_pending(self):
        counters = {k: 7 for k in capture.HELPER_COUNTERS}
        rows = [dict(type='helper_weight_traffic', phase=phase, helper_live_pending=0, **counters)
                for phase in ('before_prefill', 'generation_start', 'final')]
        self.assertEqual(capture.helper_traffic(rows, True)['status'], 'validated')
        rows[-1]['h2d_issued_bytes'] += 1
        self.assertIn('static helper weight traffic or failure occurred during inference', capture.helper_traffic(rows, True)['errors'])
        rows[-1]['helper_live_pending'] = 1
        self.assertIn('no pending helper live copy at final', capture.helper_traffic(rows, True)['missing_fields'])

    def test_static_placement_roster_and_counter(self):
        counters = {k: 9 for k in capture.HELPER_COUNTERS}
        first = {'helper_weight_traffic': counters.copy(), 'helper_live_pending': 0, 'helper_ids': [[1, 2], [3, 4]]}
        final = {'helper_weight_traffic': counters.copy(), 'helper_live_pending': 0, 'helper_ids': [[3, 4], [1, 2]]}
        self.assertEqual(capture.helper_placement_comparison(first, final, True)['status'], 'validated')
        final['helper_ids'] = [[1, 2]]
        self.assertIn('static helper roster changed between inspections', capture.helper_placement_comparison(first, final, True)['errors'])


class ControlCaptureTest(unittest.TestCase):
    def test_ordered_control_ids_and_artifact_join(self):
        with tempfile.TemporaryDirectory() as d:
            directory = pathlib.Path(d)
            control = directory / 'control'
            c = {'directory': directory, 'control': control, 'manifest': {
                'mode': 'timeline', 'experiment_id': 'experiment', 'run_id': 'run', 'control_operations': []}}
            observed = []
            def execute():
                mode, experiment, run, request, dest = control.read_text().split()
                observed.append(request)
                ids = {'experiment_id': experiment, 'run_id': run, 'request_id': request}
                rows = [dict(ids, type='capture_begin', event_seq=1),
                        dict(ids, type='control_begin', event_seq=2, verb='PLACEMENT', action='inspect'),
                        dict(ids, type='control_end', event_seq=3, verb='PLACEMENT', action='inspect', ok=True, error=''),
                        dict(ids, type='capture_end', event_seq=4, status='ok'),
                        dict(ids, type='capture_footer', recorded_events=4, attempted_events=4, dropped_events=0)]
                (pathlib.Path(dest) / (request + '.jsonl')).write_text(''.join(json.dumps(x) + '\n' for x in rows))
                return 42
            self.assertEqual(capture.operation(c, 'placement_inspect_initial', execute), 42)
            self.assertEqual(capture.operation(c, 'placement_inspect_final', execute), 42)
            self.assertTrue(observed[0].startswith('ctl-0001-'))
            self.assertTrue(observed[1].startswith('ctl-0002-'))
            self.assertNotEqual(*observed)
            self.assertFalse(control.exists())
            self.assertEqual([r['validation']['status'] for r in c['manifest']['control_operations']], ['validated', 'validated'])

    def test_failed_operation_is_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            directory = pathlib.Path(d)
            c = {'directory': directory, 'control': directory / 'control', 'manifest': {
                'mode': 'off', 'experiment_id': 'experiment', 'run_id': 'run', 'control_operations': []}}
            def fail(): raise RuntimeError('failure')
            with self.assertRaises(RuntimeError): capture.operation(c, 'placement_all', fail)
            self.assertEqual(c['manifest']['control_operations'][0]['status'], 'failed')
            self.assertFalse(c['control'].exists())


class CompanionTest(unittest.TestCase):
    def test_no_placement_controls_and_same_process(self):
        tools_path = str(pathlib.Path(__file__).resolve().parents[1] / 'tools/mm3')
        if tools_path not in sys.path: sys.path.insert(0, tools_path)
        import native_compare as compare
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            protocol = root / 'protocol.json'
            protocol.write_text(json.dumps({'arms': {'static': {'tune': {'helper_live': 0}, 'expect': {}}},
                'requests': [{'id': 'python-code', 'prompt': 'code', 'checks': []}, {'id': 'prose', 'prompt': 'prose', 'checks': []}],
                'seed': 17, 'max_tokens_each': 8}))
            manifest = root / 'prior-manifest.json'
            source = {'commit': 'abc', 'diff_sha256': 'def', 'untracked_sha256': 'ghi'}
            manifest.write_text(json.dumps({'source': source, 'executable': {'sha256': 'exe'}}))
            prior = root / 'prior.json'
            prior.write_text(json.dumps({'complete': True, 'rankable': True, 'arm': 'static', 'base_url': 'http://localhost',
                'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(), 'final_inspection': {'engine_pid': 101},
                'capture_manifest': str(manifest), 'placement_sha256': 'placement',
                'requests': [{'process_io_after': {'start_ticks': '22'}}]}))
            c = {'manifest': {'source': source, 'executable': {'sha256': 'exe'}, 'requests': []}, 'directory': root, 'control': root / 'control'}
            def run_case(base, case, req, timeout, raw, result, pid, **kw):
                result['requests'].append({'complete': True, 'case': case['id'], 'timings': {'cache_n': 0},
                    'engine_stats': {'final': {'native_cooperative': 1}, 'interval_status': 'measured',
                    'generation_interval': {'tokens': 8, 'interval_ms': 1000, 'tokens_per_second': 8}}})
            argv = ['native_compare.py', '--protocol', str(protocol), '--arm', 'static', '--output', str(root / 'out.json'),
                    '--base-url', 'http://localhost', '--continue-from', str(prior), '--capture-mode', 'detailed', '--capture-control', str(root / 'control')]
            with patch.object(sys, 'argv', argv), patch.object(compare.capture, 'init', return_value=c), \
                 patch.object(compare.capture, 'save'), patch.object(compare.capture, 'bind_executable'), \
                 patch.object(compare.capture, 'operation', side_effect=AssertionError('unexpected placement control')), \
                 patch.object(compare.screen, 'process_io', return_value={'start_ticks': '22'}), \
                 patch.object(compare.screen, 'run_case', side_effect=run_case), contextlib.redirect_stdout(io.StringIO()):
                compare.main()
            result = json.loads((root / 'out.json').read_text())
            self.assertFalse(result['rankable'])
            self.assertEqual(result['kind'], 'diagnostic_companion')
            self.assertIsNone(result['final_inspection'])
            self.assertTrue(result['complete'])


class FrontendCaptureTest(unittest.TestCase):
    def test_buffered_receipts_and_footer(self):
        from serve.server import _TokenReceiptCapture
        with tempfile.TemporaryDirectory() as d:
            control = pathlib.Path(d) / 'control'
            control.write_text(f'off experiment run request {d}\n')
            with patch.dict(os.environ, {'MM3_CAPTURE_CONTROL': str(control)}):
                recorder = _TokenReceiptCapture()
                recorder.token(21)
                recorder.token(22, drained=True)
                self.assertFalse((pathlib.Path(d) / 'request.frontend.jsonl').exists())
                recorder.finish(False)
            lines = [json.loads(x) for x in (pathlib.Path(d) / 'request.frontend.jsonl').read_text().splitlines()]
            self.assertEqual([r['token'] for r in lines[:-1]], [21, 22])
            self.assertTrue(lines[1]['drained'])
            self.assertEqual(lines[-1]['recorded_events'], 2)
            self.assertFalse(lines[-1]['engine_done_seen'])


if __name__ == '__main__': unittest.main()
