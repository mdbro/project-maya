#!/usr/bin/env python3
"""Checks the Python reader against records written by the real C++ recorder."""
import json
import pathlib
import struct
import subprocess
import tempfile
import unittest

import native_light as light

ROOT = pathlib.Path(__file__).resolve().parents[2]


def build_and_run(directory: pathlib.Path) -> None:
    binary = directory / 'recorder_test'
    subprocess.run(['g++', '-std=c++17', '-O1', '-pthread', '-I/usr/local/cuda/include',
                    str(ROOT / 'tests/mm3_light_recorder_test.cpp'), '-lzstd', '-o', str(binary)], check=True)
    subprocess.run([str(binary), str(directory / 'out')], check=True, stdout=subprocess.DEVNULL)


def as_format(raw: bytes, version: int) -> bytes:
    """A format 2 capture rewritten in an older or unknown version's header; for 1 also in its layout, each layer
    record's fixed part without the GPU stamps (the bytes format 1 writers produced)."""
    old, json_bytes, _ = struct.unpack_from('<IIQ', raw, 8)
    assert old == 2
    meta = json.loads(raw[24:24 + json_bytes])
    at = 24 + json_bytes + (8 - json_bytes % 8) % 8
    records = bytearray()
    while at < len(raw):
        kind, size, seq, t_ns = light.HEAD.unpack_from(raw, at)
        body = raw[at + light.HEAD.size:at + size]
        if version == 1 and light.KIND[kind] == 'layer':
            values = light.LAYER_FIXED.unpack_from(body, 0)[:len(light.LAYER_FIELDS_V1)]
            body = light.LAYER_FIXED_V1.pack(*values) + body[light.LAYER_FIXED.size:]
        records += light.HEAD.pack(kind, light.HEAD.size + len(body), seq, t_ns) + body
        at += size
    meta.update(format_version=version, record_bytes=len(records))
    text = json.dumps(meta, separators=(',', ':')).encode()
    return (light.MAGIC + struct.pack('<IIQ', version, len(text), len(records)) + text + bytes((8 - len(text) % 8) % 8)
            + bytes(records))


class RecorderLayouts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.directory = pathlib.Path(cls.temp.name)
        build_and_run(cls.directory)
        cls.capture = light.read(cls.directory / 'out/valid.light')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_layer_fields_round_trip(self):
        rows = self.capture.of('layer')
        self.assertEqual(len(rows), 4)
        first = rows[0]
        self.assertEqual((first['forward'], first['layer'], first['rows'], first['k']), (1, 3, 1, 4))
        self.assertEqual((first['free_slots'], first['reserved_slots'], first['foreground_slots']), (3, 1, 2))
        self.assertEqual((first['exchanges_pending'], first['helper_lease'], first['stage_max'], first['stage_selected']), (1, 2, 3, 1))
        self.assertEqual(first['total_slots'], 8)
        self.assertEqual([first[k] for k in ('pre_end', 'join_end', 'boundary_end', 'plan_end', 'helper_submit_end',
                                             'primary_submit_end', 'cpu_end', 'collect_end', 'end')],
                         [10, 20, 30, 40, 50, 60, 70, 80, 90])
        self.assertEqual((first['helper_enqueue'], first['helper_wait_end'], first['helper_worker_start']), (41, 79, 0))
        self.assertEqual(first['original'], [1, 2, 3, 4])
        self.assertEqual(first['executed'], [1, 2, 3, 5])
        self.assertEqual(first['service'], ['helper', 'cpu', 'primary_resident', 'staged_new'])
        # (format 2) the GPU stamps: preparation 15, uploads 45, expert compute 65, the wait's end at the pass's begin
        # (written as 1) for layer 3 and none (0) for layer 4, reference width 6, 2 uploads
        self.assertEqual(self.capture.format_version, 2)
        self.assertEqual(light.GPU_FIELDS, ('gpu_prepare_end', 'gpu_upload_end', 'gpu_expert_end', 'gpu_wait_end',
                                            'gpu_clock_span', 'gpu_uploads'))
        self.assertEqual([first[k] for k in light.GPU_FIELDS], [15, 45, 65, 1, 6, 2])
        self.assertEqual([rows[1][k] for k in light.GPU_FIELDS], [15, 45, 65, 0, 6, 2])
        self.assertEqual((rows[1]['layer'], rows[1]['original'], rows[1]['executed']), (4, [1, 2, 3, 4], [1, 2, 3, 5]))

    def test_timestamps_only_keeps_clocks_and_drops_state(self):
        capture = light.read(self.directory / 'out/stamps.light')
        self.assertEqual(capture.meta['mode'], 'timestamps')
        self.assertEqual({r['type'] for r in capture.records}, {'generation_begin', 'generation_end', 'forward', 'layer'})
        first = capture.of('layer')[0]
        self.assertEqual([first[k] for k in ('pre_end', 'join_end', 'boundary_end', 'plan_end', 'helper_submit_end',
                                             'primary_submit_end', 'cpu_end', 'collect_end', 'end')],
                         [10, 20, 30, 40, 50, 60, 70, 80, 90])
        self.assertEqual((first['original'], first['executed'], first['service']), ([], [], []))
        self.assertEqual([first[k] for k in light.GPU_FIELDS], [15, 45, 65, 1, 6, 2])
        self.assertEqual(set(capture.of('forward')[0]['counters'].values()), {0})
        self.assertEqual(light.validate(capture, expected_layers=2)['status'], 'validated')

    def test_forward_and_events_round_trip(self):
        forward = self.capture.of('forward')[0]
        self.assertEqual((forward['forward'], forward['row_clock'], forward['position'], forward['rows'], forward['want']), (1, 11, 101, 1, 1))
        self.assertTrue(forward['generation'])
        self.assertEqual((forward['prepare_end'], forward['body_end'], forward['sync_end']), (1, 2, 3))
        self.assertEqual(list(forward['counters'].values()), list(range(1, 19)))
        self.assertEqual(list(forward['counters']), list(light.FORWARD_COUNTERS))
        copy = self.capture.of('demand_copy')[0]
        self.assertEqual((copy['layer'], copy['expert'], copy['slot'], copy['displaced'], copy['bytes']), (3, 5, 7, 3 * 1024 + 9, 1234))
        retire = self.capture.of('foreground_retire')[0]
        self.assertEqual((retire['slot'], retire['owner']), (7, 3 * 1024 + 5))
        issue = self.capture.of('primary_issue')[0]
        self.assertEqual((issue['incoming'], issue['victim'], issue['landing'], issue['early_layer'], issue['reuse'],
                          issue['h2d_bytes'], issue['d2h_bytes']), (3 * 1024 + 3, 3 * 1024 + 4, 1, -1, False, 111, 222))
        published = self.capture.of('primary_publish')[0]
        self.assertEqual((published['incoming'], published['victim'], published['landing'], published['protected_layer']),
                         (3 * 1024 + 3, 3 * 1024 + 4, 1, -1))
        self.assertEqual([e['phase'] for e in self.capture.of('helper_event')], ['lease', 'withdrawn', 'published', 'retired'])

    def test_generation_and_map_round_trip(self):
        begin, end = self.capture.of('generation_begin')[0], self.capture.of('generation_end')[0]
        self.assertEqual((begin['position'], begin['scope']), (100, 'request'))
        self.assertEqual((end['emitted'], end['target_rounds'], end['spec_rounds'], end['elapsed_ms'], end['ok']), (2, 2, 0, 12.5, True))
        start, final = self.capture.of('map')
        self.assertEqual((start['layers'], start['experts'], start['phase']), (4, 8, 0))
        self.assertEqual(len(start['location']), 32)
        self.assertEqual(start['slots'][0], dict(owner=3 * 1024 + 5, pending=-1, issued=0, foreground=1, reserved=0))
        self.assertEqual(start['slots'][1]['reserved'], 1)
        self.assertEqual(start['exchanges'], [dict(incoming=3 * 1024 + 3, victim=3 * 1024 + 4, landing=1, submitted=1)])
        self.assertEqual(final['exchanges'], [])

    def test_validation_accepts_complete_capture(self):
        report = light.validate(self.capture, expected_layers=2)
        self.assertEqual(report['status'], 'validated', report['errors'])
        self.assertEqual(report['state_reconstruction']['differing_locations'], 0)
        self.assertEqual(report['generation_forwards'], 2)
        self.assertEqual(report['generation_rows_other_than_one'], 0)
        # The GPU stamps are information: the layer 3 records' wait stamp (1, at the pass's begin) comes before their
        # expert compute's end, an order break that is counted and leaves the capture valid.
        self.assertEqual({k: report[k] for k in ('format_version', 'gpu_stamped_layers', 'gpu_unstamped_layers',
                                                 'gpu_stamp_order_breaks', 'gpu_clock_span_max')},
                         dict(format_version=2, gpu_stamped_layers=4, gpu_unstamped_layers=0, gpu_stamp_order_breaks=2,
                              gpu_clock_span_max=6))

    def test_reads_format_1_without_gpu_stamps(self):
        path = self.directory / 'format1.light'
        path.write_bytes(as_format((self.directory / 'out/valid.light').read_bytes(), 1))
        old = light.read(path)
        self.assertEqual(old.format_version, 1)
        new_rows, old_rows = self.capture.of('layer'), old.of('layer')
        self.assertEqual(len(old_rows), len(new_rows))
        for new, row in zip(new_rows, old_rows):
            self.assertEqual([row[k] for k in light.GPU_FIELDS], [0] * len(light.GPU_FIELDS))
            self.assertEqual({k: v for k, v in row.items() if k not in light.GPU_FIELDS},
                             {k: v for k, v in new.items() if k not in light.GPU_FIELDS})
        self.assertEqual([r for r in old.records if r['type'] != 'layer'],
                         [r for r in self.capture.records if r['type'] != 'layer'])
        report = light.validate(old, expected_layers=2)
        self.assertEqual(report['status'], 'validated', report['errors'])
        self.assertEqual((report['format_version'], report['gpu_stamped_layers'], report['gpu_unstamped_layers']), (1, 0, 4))

    def test_refuses_an_unknown_format(self):
        path = self.directory / 'format3.light'
        path.write_bytes(as_format((self.directory / 'out/valid.light').read_bytes(), 3))
        with self.assertRaisesRegex(ValueError, 'unsupported format version 3'):
            light.read(path)

    def test_validation_rejects_missing_layers(self):
        report = light.validate(self.capture, expected_layers=57)
        self.assertEqual(report['status'], 'invalid')

    def test_overflow_invalidates(self):
        overflow = light.read(self.directory / 'out/overflow.light')
        self.assertGreater(overflow.meta['dropped_records'], 0)
        report = light.validate(overflow, expected_layers=2)
        self.assertEqual(report['status'], 'invalid')
        self.assertTrue(any('dropped' in e for e in report['errors']))

    def test_export_fixture_uses_complete_contiguous_forwards(self):
        # The test capture starts at position 101, which is not a complete KV prefix.
        with self.assertRaises(ValueError):
            light.export_fixture(self.capture, self.directory / 'rejected.replay', generation_forwards=2)
        with self.assertRaises(ValueError):
            light.export_fixture(self.capture, self.directory / 'too-many.replay', generation_forwards=3)


class FixtureExport(unittest.TestCase):
    def forwards(self):
        capture = light.LightCapture(pathlib.Path('synthetic'), 'sha', {})
        def layer(forward, number, rows):
            return dict(type='layer', forward=forward, layer=number, rows=rows,
                        original=list(range(4)) * rows, executed=list(range(4, 8)) * rows)
        capture.records.extend(layer(1, n, 2) for n in (3, 4))
        capture.records.append(dict(type='forward', forward=1, position=0, rows=2, generation=False))
        for f, position in ((2, 2), (3, 3), (4, 4)):
            capture.records.extend(layer(f, n, 1) for n in (3, 4))
            capture.records.append(dict(type='forward', forward=f, position=position, rows=1, generation=True))
        return capture

    def test_selects_first_generation_forwards_after_setup(self):
        with tempfile.TemporaryDirectory() as t:
            path = pathlib.Path(t) / 'fixture.replay'
            receipt = light.export_fixture(self.forwards(), path, generation_forwards=2, hold_primary=True, hold_helper=True)
            lines = path.read_text().splitlines()
            self.assertEqual(lines[0], 'MM3_REPLAY 4 8 4 1.3 0 0 1 1 0')
            self.assertEqual([x for x in lines if x.startswith('F ')], ['F 2 0 0', 'F 1 2 1', 'F 1 3 1'])
            self.assertEqual(lines[-1], 'END')
            self.assertEqual((receipt['prefill_forwards'], receipt['generation_forwards'], receipt['moe_layers']), (1, 2, 2))

    def test_rejects_multi_row_generation(self):
        capture = self.forwards()
        for r in capture.records:
            if r['type'] == 'forward' and r['forward'] == 2:
                r['rows'] = 2
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):
                light.export_fixture(capture, pathlib.Path(t) / 'x', generation_forwards=2)

    def test_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as t:
            path = pathlib.Path(t) / 'fixture.replay'
            light.export_fixture(self.forwards(), path, generation_forwards=1)
            with self.assertRaises(FileExistsError):
                light.export_fixture(self.forwards(), path, generation_forwards=1)


if __name__ == '__main__':
    unittest.main()
