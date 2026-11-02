"""Index the official MiniMax M3.1 native safetensors in place; no weight data is copied.

    python tools/mm3/pack_native.py --model /path/to/MiniMax-M3.1-Flash --out /path/to/pack

Produces Maya's mm3.txt, tensors.txt and tokenizer plus precision-inventory.json.
The pack describes stored formats; it does not implement native inference. All shard
paths are relative to model_dir. Readiness/integrity must be established separately.
"""
from __future__ import annotations

import argparse
import collections
import json
import math
import pathlib
import re
import struct

import pack

P = 'language_model.model.'
DTYPE_BYTES = {'U8': 1, 'F8_E4M3': 1, 'BF16': 2, 'F32': 4}
EXPERT = re.compile(r'^language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\.(w[123])\.(\w+)$')


def read_headers(directory):
    """Read headers only; enforce index membership, byte sizes and bounds."""
    index = json.loads((directory / 'model.safetensors.index.json').read_text())['weight_map']
    files = sorted(set(index.values()))
    tensors = {}
    for fn in files:
        path = directory / fn
        with path.open('rb') as f:
            head = f.read(8)
            if len(head) != 8:
                raise ValueError(f'{path}: truncated header length')
            n = struct.unpack('<Q', head)[0]
            if n > path.stat().st_size - 8:
                raise ValueError(f'{path}: invalid header length')
            h = json.loads(f.read(n))
        base = 8 + n
        intervals = []
        for name, v in h.items():
            if name == '__metadata__':
                continue
            if name in tensors or index.get(name) != fn:
                raise ValueError(f'{name}: duplicated or inconsistent index')
            dtype, shape = v['dtype'], tuple(v['shape'])
            if dtype not in DTYPE_BYTES or any(not isinstance(x, int) or x < 0 for x in shape):
                raise ValueError(f'{name}: unsupported dtype or shape')
            a, b = v['data_offsets']
            if a < 0 or b < a or base + b > path.stat().st_size or b - a != math.prod(shape) * DTYPE_BYTES[dtype]:
                raise ValueError(f'{name}: invalid tensor range or size')
            intervals.append((a, b, name))
            tensors[name] = (fn, dtype, shape, base + a, b - a)
        intervals.sort()
        if any(a[1] > b[0] for a, b in zip(intervals, intervals[1:])):
            raise ValueError(f'{path}: overlapping tensor ranges')
    if set(index) != set(tensors):
        raise ValueError(f'{directory}: index entries missing from shard headers')
    return files, tensors


def check_native(tensors, cfg, draft=False):
    """Validate each quantized matrix and every expected routed expert record."""
    for name, (_, dtype, shape, _, _) in tensors.items():
        m = EXPERT.match(name) if not draft else None
        if m:
            li, expert, w, part = m.groups()
            li, expert = int(li), int(expert)
            if li >= cfg['num_hidden_layers'] or not cfg['moe_layer_freq'][li] or expert >= cfg['num_local_experts']:
                raise ValueError(f'{name}: invalid expert location')
            rows, cols = (cfg['hidden_size'], cfg['intermediate_size']) if w == 'w2' else (cfg['intermediate_size'], cfg['hidden_size'])
            expected = {'weight_packed': ('U8', (rows, cols // 2)), 'weight_scale': ('F8_E4M3', (rows, cols // 16)),
                        'weight_global_scale': ('F32', ()), 'input_global_scale': ('F32', ())}
            if part not in expected or (dtype, shape) != expected[part]:
                raise ValueError(f'{name}: invalid NVFP4 component')
        elif dtype == 'F8_E4M3':
            if not name.endswith('.weight') or len(shape) != 2 or shape[1] % 32:
                raise ValueError(f'{name}: invalid MXFP8 matrix')
            scale = name[:-len('.weight')] + '.weight_scale_inv'
            if scale not in tensors or tensors[scale][1:3] != ('U8', (shape[0], shape[1] // 32)):
                raise ValueError(f'{name}: missing or invalid MXFP8 scale')
        elif dtype == 'U8':
            if not name.endswith('.weight_scale_inv'):
                raise ValueError(f'{name}: unknown U8 tensor')
            weight = name[:-len('.weight_scale_inv')] + '.weight'
            if weight not in tensors or tensors[weight][1] != 'F8_E4M3':
                raise ValueError(f'{name}: orphan MXFP8 scale')
    if not draft:
        def require(name, dtype, shape):
            if name not in tensors or tensors[name][1:3] != (dtype, shape):
                raise ValueError(f'{name}: missing or invalid required tensor')
        hidden, vocab, head = cfg['hidden_size'], cfg['vocab_size'], cfg['head_dim']
        require(P + 'embed_tokens.weight', 'BF16', (vocab, hidden))
        require(P + 'norm.weight', 'BF16', (hidden,))
        require('language_model.lm_head.weight', 'BF16', (vocab, hidden))
        for li, is_moe in enumerate(cfg['moe_layer_freq']):
            layer = f'{P}layers.{li}.'
            for suffix in ('input_layernorm.weight', 'post_attention_layernorm.weight'):
                require(layer + suffix, 'BF16', (hidden,))
            for suffix in ('q_norm', 'k_norm', 'index_q_norm', 'index_k_norm'):
                require(layer + 'self_attn.' + suffix + '.weight', 'BF16', (head,))
            shapes = {'q_proj': (cfg['num_attention_heads'] * head, hidden),
                      'k_proj': (cfg['num_key_value_heads'] * head, hidden),
                      'v_proj': (cfg['num_key_value_heads'] * head, hidden),
                      'o_proj': (hidden, cfg['num_attention_heads'] * head),
                      'index_q_proj': (cfg['sparse_attention_config']['sparse_num_index_heads'] * cfg['sparse_attention_config']['sparse_index_dim'], hidden),
                      'index_k_proj': (cfg['sparse_attention_config']['sparse_index_dim'], hidden)}
            for suffix, shape in shapes.items():
                require(layer + 'self_attn.' + suffix + '.weight', 'F8_E4M3', shape)
            if is_moe:
                moe = layer + 'block_sparse_moe.'
                require(moe + 'gate.weight', 'F32', (cfg['num_local_experts'], hidden))
                require(moe + 'e_score_correction_bias', 'F32', (cfg['num_local_experts'],))
            width = cfg['shared_intermediate_size'] if is_moe else cfg['dense_intermediate_size']
            prefix = layer + ('block_sparse_moe.shared_experts.' if is_moe else 'mlp.')
            for suffix, shape in (('gate_proj', (width, hidden)), ('up_proj', (width, hidden)), ('down_proj', (hidden, width))):
                require(prefix + suffix + '.weight', 'F8_E4M3', shape)
            if is_moe:
                for e in range(cfg['num_local_experts']):
                    for w in ('w1', 'w2', 'w3'):
                        prefix = f'{P}layers.{li}.block_sparse_moe.experts.{e}.{w}.'
                        for part in ('weight_packed', 'weight_scale', 'weight_global_scale', 'input_global_scale'):
                            if prefix + part not in tensors:
                                raise ValueError(f'{prefix + part}: missing')



def check_draft(tensors, cfg, dcfg):
    def require(name, dtype, shape):
        if name not in tensors or tensors[name][1:3] != (dtype, shape):
            raise ValueError(f'{name}: missing or invalid DSpark tensor')
    prefix = P + 'dspark.'
    d, v, r = cfg['hidden_size'], cfg['vocab_size'], dcfg['dspark_markov_rank']
    taps, ff = len(dcfg['dspark_target_layer_ids']), dcfg['dspark_ffn_hidden_size']
    for name, shape in [('fc.weight', (d, taps * d)), ('hidden_norm.weight', (d,)),
                        ('final_norm.weight', (d,)), ('markov_head.markov_w1.weight', (v, r)),
                        ('markov_head.markov_w2.weight', (v, r)), ('confidence_head.proj.weight', (1, d + r)),
                        ('confidence_head.proj.bias', (1,))]:
        require(prefix + name, 'BF16', shape)
    hd, nq, nk = cfg['head_dim'], cfg['num_attention_heads'], cfg['num_key_value_heads']
    for i in range(dcfg['num_hidden_layers']):
        layer = prefix + f'layers.{i}.decoder_layer.'
        for name, shape in [('input_layernorm.weight', (d,)), ('post_attention_layernorm.weight', (d,)),
                            ('self_attn.q_norm.weight', (hd,)), ('self_attn.k_norm.weight', (hd,))]:
            require(layer + name, 'BF16', shape)
        for name, shape in [('self_attn.q_proj', (nq * hd, d)), ('self_attn.k_proj', (nk * hd, d)),
                            ('self_attn.v_proj', (nk * hd, d)), ('self_attn.o_proj', (d, nq * hd)),
                            ('mlp.gate_proj', (ff, d)), ('mlp.up_proj', (ff, d)), ('mlp.down_proj', (d, ff))]:
            require(layer + name + '.weight', 'F8_E4M3', shape)

def inventory(main, draft):
    groups = collections.defaultdict(lambda: collections.defaultdict(lambda: {'tensors': 0, 'elements': 0, 'bytes': 0}))
    for source, tensors in [('main', main), ('draft', draft)]:
        for name, (_, dtype, shape, _, nbytes) in tensors.items():
            expert = EXPERT.match(name) if source == 'main' else None
            is_scale = name.endswith(('.weight_scale', '.weight_scale_inv', '.weight_global_scale', '.input_global_scale'))
            group = ('draft' if source == 'draft' else 'vision' if not name.startswith('language_model.') else
                     'expert' if expert else 'text_dense')
            if is_scale:
                group += '_scales'
            precision = 'NVFP4' if expert and name.endswith('.weight_packed') else dtype
            elements = math.prod(shape) * (2 if precision == 'NVFP4' else 1)
            v = groups[group][precision]
            v['tensors'] += 1; v['elements'] += elements; v['bytes'] += nbytes
    groups = {g: dict(sorted(v.items())) for g, v in sorted(groups.items())}
    text = [v for group in ('expert', 'text_dense') for v in groups[group].values()]
    return {'definition': 'All main language_model stored parameter elements, including norms and biases; excludes scale tensors, vision and draft. NVFP4 byte stores two 4-bit parameters.',
            'groups': groups, 'main_text_parameters': sum(v['elements'] for v in text),
            'main_text_weight_bytes': sum(v['bytes'] for v in text),
            'main_text_bits_per_parameter': 8 * sum(v['bytes'] for v in text) / sum(v['elements'] for v in text)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', required=True, type=pathlib.Path)
    ap.add_argument('--out', required=True, type=pathlib.Path)
    a = ap.parse_args(); model = a.model.expanduser().resolve(); out = a.out.expanduser().resolve()
    if any(c.isspace() for c in str(model)):
        raise ValueError('model path cannot contain whitespace in Maya Pack file paths')
    cfg = json.loads((model / 'config.json').read_text()); t = cfg['text_config']; q = cfg['quantization_config']
    if (q.get('quant_method'), q.get('moe_quant_algo'), q.get('group_size'), q.get('weight_block_size')) != ('mxfp8', 'NVFP4', 16, [1, 32]):
        raise ValueError('unexpected official quantization configuration')
    dcfg = json.loads((model / 'dspark/config.json').read_text())
    if dcfg['quantization_config'].get('quant_method') != 'mxfp8' or dcfg['quantization_config'].get('weight_block_size') != [1, 32]:
        raise ValueError('unexpected DSpark quantization configuration')
    files, tensors = read_headers(model); dfiles, dtensors = read_headers(model / 'dspark')
    check_native(tensors, t); check_native(dtensors, t, draft=True); check_draft(dtensors, t, dcfg)
    result = inventory(tensors, dtensors)
    out.mkdir(parents=True, exist_ok=True)
    special = pack.write_tokenizer(out / 'tokenizer', model)
    gen = json.loads((model / 'generation_config.json').read_text())
    with (out / 'mm3.txt').open('w', newline='\n') as f:
        pack.flat(f, '', t); pack.flat(f, 'sparse_attention_config.', t.get('sparse_attention_config') or {})
        f.write('format native\n')
        for k in ('bos_token_id', 'eos_token_id'):
            f.write(f'{k} {gen[k]}\n')
        for k, v in special.items(): f.write(f'special.{k} {v}\n')
        pack.flat(f, 'dspark.', {k: v for k, v in dcfg.items() if k not in ('text_config', 'vision_config')})
        pack.flat(f, 'dspark.dflash.', dcfg.get('dflash_config') or {})
        pack.flat(f, 'dspark.cfg.', dcfg.get('dspark_config') or {})
        f.write(f'model_dir {model}\n')
    with (out / 'tensors.txt').open('w', newline='\n') as f:
        allfiles = files + ['dspark/' + fn for fn in dfiles]
        for i, fn in enumerate(allfiles): f.write(f'file {i} {fn}\n')
        ids = {fn: i for i, fn in enumerate(files)}; dids = {fn: i + len(files) for i, fn in enumerate(dfiles)}
        for ts, idx, pre in ((tensors, ids, ''), (dtensors, dids, 'dspark.')):
            for name, (fn, dtype, shape, offset, nb) in sorted(ts.items()):
                dims = ''.join(f' {d}' for d in shape)
                f.write(f't {pre}{name} {dtype} {idx[fn]} {offset} {nb}{dims}\n')
    result['layout'] = {'expert': 'U8 packed 2 NVFP4 codes per byte; F8_E4M3 scale per 16 weights; F32 input_global_scale and weight_global_scale scalars per matrix',
                        'dense_and_draft': 'F8_E4M3 matrix and U8 weight_scale_inv per 32 weights; other values BF16/F32 as stored',
                        'activation_scale_treatment': 'Not chosen by metadata pack; runtime arithmetic must declare its treatment.',
                        'weight_copy': False, 'model_dir': str(model)}
    (out / 'precision-inventory.json').write_text(json.dumps(result, indent=2) + '\n')
    print(f'{len(tensors)} main and {len(dtensors)} draft tensors, {len(allfiles)} shards; text {result["main_text_bits_per_parameter"]:.6f} bpw -> {out}')


if __name__ == '__main__':
    main()
