import importlib.util
import io
import json
import pathlib
import sys
import unittest
from unittest.mock import patch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import native_screen as screen
import native_delivery as delivery


def row(count, ms, complete=1, final=0, seq=1):
    return {'request_seq': seq, 'emitted': count, 'elapsed_ms': ms,
            'round_complete': complete, 'generation_final': final}


def metrics(rows):
    keys = set().union(*(r.keys() for r in rows)) if rows else set()
    return {'tiers': {'history': {k: [r.get(k) for r in rows] for k in keys}}}


def summarize(rows, usage=None, start=32):
    return screen.summarize_stats(metrics([]), metrics(rows),
                                  {'completion_tokens': rows[-1]['emitted']} if usage is None else usage,
                                  start)


class MeasurementTest(unittest.TestCase):
    def test_stream_trace_off_keeps_output_and_boundary_timing_only(self):
        class Response:
            status = 200
            headers = {'Content-Type': 'text/event-stream'}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                yield b'data: {"choices":[{"index":0,"delta":{"content":"ok"}}]}\n'
                yield b'data: {"choices":[{"index":0,"finish_reason":"length","delta":{}}],"usage":{"completion_tokens":2}}\n'
                yield b'data: [DONE]\n'
        result, raw = {'requests': []}, io.StringIO()
        with patch.object(screen, 'metrics', side_effect=[metrics([]), metrics([row(0, 0), row(2, 10, final=1)])]), \
             patch.object(screen.urllib.request, 'urlopen', return_value=Response()), \
             patch.object(screen, 'process_io', side_effect=AssertionError('process monitoring requested')):
            screen.run_case('http://localhost', {'id':'quiet','checks':[]}, {}, 1, raw, result,
                            start_emitted=0, trace_stream=False)
        item = result['requests'][0]
        self.assertTrue(item['complete'])
        self.assertEqual(item['content'], 'ok')
        self.assertEqual(raw.getvalue(), '')
        self.assertEqual(len(item['events']), 1)
        self.assertEqual(item['engine_stats']['generation_interval']['tokens_per_second'], 200)

    def test_cooperation_dispatch_must_match_the_declared_case(self):
        off = {'native_cooperative': 0, 'decode_cpu_experts': 0,
               'helper_experts': 0, 'owner_entries': 1824}
        self.assertIsNone(screen.cooperation_error(off, 0))
        self.assertIsNone(screen.cooperation_error({'native_cooperative': 1}, 1))
        self.assertIsNotNone(screen.cooperation_error(off, 1))
        self.assertIsNotNone(screen.cooperation_error({}, 0))
        for key in ('decode_cpu_experts', 'helper_experts', 'owner_entries'):
            missing = {k: v for k, v in off.items() if k != key}
            self.assertIsNotNone(screen.cooperation_error(missing, 0))
        for delta in ({'decode_cpu_experts': 1}, {'helper_experts': 1}, {'owner_entries': 0}):
            self.assertIsNotNone(screen.cooperation_error({**off, **delta}, 0))

    def test_ordinary_complete_boundary(self):
        rows = [row(0,0), row(31,310), row(32,320), row(33,330), row(40,400,final=1)]
        stats = summarize(rows)
        self.assertEqual(stats['interval_status'], 'measured')
        self.assertEqual(stats['generation_interval']['tokens'], 8)
        self.assertEqual(stats['generation_interval']['interval_ms'], 80)
        self.assertEqual(stats['generation_interval']['tokens_per_second'], 100)

    def test_speculative_rejection_and_interior_markers(self):
        rows = [row(0,0), row(30,100), row(32,105,complete=0), row(34,120),
                row(35,135), row(38,160), row(39,170,final=1)]
        interval = summarize(rows)['generation_interval']
        self.assertEqual((interval['start_emitted'], interval['end_emitted'], interval['tokens']), (34,39,5))
        self.assertEqual(interval['interval_ms'], 50)

    def test_early_finish_and_empty_interval(self):
        stats = summarize([row(0,0), row(10,30), row(16,50,final=1)])
        self.assertEqual(stats['interval_status'], 'not_eligible')
        self.assertIsNone(stats['generation_interval'])
        stats = summarize([row(0,0), row(34,30,final=1)])
        self.assertEqual(stats['interval_status'], 'not_eligible')

    def test_threshold_zero_and_terminal_count(self):
        interval = summarize([row(0,0), row(1,5,final=1)], start=0)['generation_interval']
        self.assertEqual((interval['tokens'],interval['interval_ms']), (1,5))

    def test_legacy_missing_final_and_history_rotation_do_not_qualify(self):
        legacy = [{'request_seq':1,'emitted':0,'elapsed_ms':0}, {'request_seq':1,'emitted':40,'elapsed_ms':100}]
        for rows in (legacy, [row(0,0),row(32,50),row(40,70)], [row(24,30),row(32,50),row(40,70,final=1)]):
            self.assertIsNone(summarize(rows)['generation_interval'])

    def test_bad_time_count_usage_and_final_rejected(self):
        cases = [([row(0,0),row(32,50),row(31,60,final=1)], None),
                 ([row(0,0),row(32,50),row(40,49,final=1)], None),
                 ([row(0,0),row(32,50),row(40,70,final=1)], {'completion_tokens':39}),
                 ([row(0,0),row(32,50,final=1),row(40,70,final=1)], None),
                 ([row(0,0),row(32,50),row(40,50,final=1)], None)]
        for rows, usage in cases:
            with self.subTest(rows=rows): self.assertIsNone(summarize(rows,usage)['generation_interval'])

    def test_failed_forward_noncomplete_marker_does_not_qualify(self):
        rows = [row(0,0), row(32,100), row(40,140,complete=0,final=0)]
        stats = summarize(rows)
        self.assertIsNone(stats['generation_interval'])
        self.assertIn('final completed boundary',stats['reason'])

    def test_multiple_engine_sequences_rejected(self):
        stats = summarize([row(0,0),row(4,10,final=1),row(0,0,seq=2),row(40,80,final=1,seq=2)])
        self.assertEqual(stats['join'],'ambiguous')
        self.assertIsNone(stats['generation_interval'])

    def test_delivery_does_not_use_legacy_suffix(self):
        old = {'case':'old','engine_stats':{'suffix_after_32':{'tokens':8,'interval_ms':80,'tokens_per_second':100}}}
        stats = summarize([row(0,0),row(32,50),row(40,90,final=1)])
        out = delivery.aggregate([old,{'case':'new','engine_stats':stats}])
        self.assertEqual(out['missing'],['old'])
        self.assertEqual(out['pooled_tokens'],8)
        self.assertEqual(out['pooled_tokens_per_second'],200)

    def test_multiple_api_sequences_rejected(self):
        with self.assertRaises(ValueError):
            screen.run_case('http://invalid', {'id':'x','checks':[]}, {'n':2}, 1, io.StringIO(), {'requests':[]})

    def test_multiple_sse_choices_rejected(self):
        class Response:
            status=200
            headers={'Content-Type':'text/event-stream'}
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def __iter__(self):
                yield b'data: {"choices": [{"index": 0}, {"index": 1}]}\n'
        result={'requests':[]}
        with patch.object(screen,'metrics',return_value=metrics([])), patch.object(screen.urllib.request,'urlopen',return_value=Response()):
            screen.run_case('http://localhost',{'id':'x','checks':[]},{},1,io.StringIO(),result)
        self.assertIn('Multiple completion sequences',result['requests'][0]['error'])

if __name__ == '__main__': unittest.main()
