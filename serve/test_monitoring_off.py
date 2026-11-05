"""No optional hardware reads or token capture when monitoring is disabled."""
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from serve.server import Service, _TokenReceiptCapture


class MonitoringOff(unittest.TestCase):
    def test_disabled_service_does_not_start_or_poll_hardware(self):
        service = Service.__new__(Service)
        service.monitoring_enabled = False
        with patch('serve.telemetry.Telemetry', side_effect=AssertionError('hardware sampler started')), \
             patch('serve.server.nvml_pids', side_effect=AssertionError('NVML polled')):
            service.start_telemetry()
            self.assertIsNone(service._gpu_map(None))
            self.assertEqual(service._progress(123.0), 123.0)
        self.assertFalse(hasattr(service, '_queue_watch'))

    def test_off_capture_emits_no_frontend_token_artifact(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / 'control'
            control.write_text(f'off experiment run request {root}\n')
            with patch.dict(os.environ, {'MM3_CAPTURE_CONTROL': str(control)}):
                capture = _TokenReceiptCapture()
                capture.token(42)
                capture.finish(True)
            self.assertEqual(capture.rows, [])
            self.assertIsNone(capture.context)
            self.assertEqual(list(root.iterdir()), [control])

    def test_timeline_capture_still_records_when_requested(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            control = root / 'control'
            control.write_text(f'timeline experiment run request {root}\n')
            with patch.dict(os.environ, {'MM3_CAPTURE_CONTROL': str(control)}):
                capture = _TokenReceiptCapture()
                capture.token(42)
                capture.finish(True)
            self.assertEqual(len(capture.rows), 1)
            self.assertTrue((root / 'request.frontend.jsonl').is_file())


if __name__ == '__main__':
    unittest.main()
