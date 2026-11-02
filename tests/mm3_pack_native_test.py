"""Header-only native pack checks: python -m unittest discover -s tests -p mm3_pack_native_test.py."""
import json
import pathlib
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / 'tools/mm3'))
import pack_native as native


class NativePackTest(unittest.TestCase):
    def test_headers_offsets_and_index_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            d = pathlib.Path(temp)
            header = json.dumps({'x': {'dtype': 'BF16', 'shape': [2], 'data_offsets': [0, 4]}}).encode()
            (d / 'a.safetensors').write_bytes(struct.pack('<Q', len(header)) + header + b'1234')
            index = d / 'model.safetensors.index.json'
            index.write_text(json.dumps({'weight_map': {'x': 'a.safetensors'}}))
            files, tensors = native.read_headers(d)
            self.assertEqual(files, ['a.safetensors'])
            self.assertEqual(tensors['x'], ('a.safetensors', 'BF16', (2,), 8 + len(header), 4))
            index.write_text(json.dumps({'weight_map': {'x': 'a.safetensors', 'missing': 'a.safetensors'}}))
            with self.assertRaisesRegex(ValueError, 'missing'):
                native.read_headers(d)

    def test_quant_geometry_and_orphan_scales(self):
        def t(dtype, shape): return ('f', dtype, shape, 0, 0)
        dense = {'x.weight': t('F8_E4M3', (2, 64)), 'x.weight_scale_inv': t('U8', (2, 2))}
        native.check_native(dense, {}, draft=True)
        dense['x.weight_scale_inv'] = t('U8', (2, 1))
        with self.assertRaisesRegex(ValueError, 'scale'):
            native.check_native(dense, {}, draft=True)
        dense = {'x.weight_scale_inv': t('U8', (2, 2))}
        with self.assertRaisesRegex(ValueError, 'orphan'):
            native.check_native(dense, {}, draft=True)

    def test_parameter_weighted_inventory_excludes_scales(self):
        def t(dtype, shape, size): return ('f', dtype, shape, 0, size)
        main = {'language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight_packed': t('U8', (2, 4), 8),
                'language_model.model.layers.3.block_sparse_moe.experts.0.w1.weight_scale': t('F8_E4M3', (2, 1), 2),
                'language_model.model.norm.weight': t('BF16', (2,), 4),
                'vision.x': t('BF16', (100,), 200)}
        result = native.inventory(main, {})
        self.assertEqual(result['main_text_parameters'], 18)
        self.assertEqual(result['main_text_weight_bytes'], 12)
        self.assertAlmostEqual(result['main_text_bits_per_parameter'], 96 / 18)
        self.assertEqual(result['groups']['expert_scales']['F8_E4M3']['bytes'], 2)


if __name__ == '__main__':
    unittest.main()
