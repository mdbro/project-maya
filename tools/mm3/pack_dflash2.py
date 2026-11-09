#!/usr/bin/env python3
"""Pack RadixArk's DFlash2 draft for MiniMax-M3.1-Flash (BF16 safetensors) as a small maya-mm3 pack, its projections in
MXFP8 (step 6, the user's decision: DFlash2 at 8 bits).

The pack: <out>/mm3.txt (the draft's geometry as `dflash2.*` keys, model_dir = <out>), <out>/tensors.txt and one data file
<out>/dflash2.bin. Tensor names keep the checkpoint's, prefixed `dflash2.`.
  MXFP8 (the layout mx_rows reads, as the target's own MXFP8 tensors are stored): each layer's q / k / v / o, gate / up /
    down and the two conv kernel projections - `<name>.weight` E4M3 codes [N][K] (F8_E4M3) and `<name>.weight_scale_inv`
    a UE8M0 byte for every 32 inputs [N][K/32] (U8). The scale exponent of a group of 32 is the smallest e with every
    |w| / 2^e at most 448 (e = ceil(log2(amax / 448)) on the F32 value, as the engine's FP8 KV path computes it); a
    code is the nearest E4M3 value of w / 2^e, ties to the even code; an all-zero group has the byte 0.
  BF16 as stored: fc, the norms, the conv base kernels, the selector's hidden projection and codebooks.
Usage: pack_dflash2.py --src <dir with model.safetensors and config.json> --out <new pack dir>"""
from __future__ import annotations
import argparse, json, pathlib, struct, sys
import numpy as np

QUANT = ('self_attn.q_proj.weight', 'self_attn.k_proj.weight', 'self_attn.v_proj.weight', 'self_attn.o_proj.weight',
         'mlp.gate_proj.weight', 'mlp.up_proj.weight', 'mlp.down_proj.weight', 'attention_conv.kernel_projection.weight',
         'mlp_conv.kernel_projection.weight')
ALIGN = 256

# the 127 non-negative finite E4M3 (FN) values, code order = value order: subnormals 2^-6 * m/8, normals 2^(e-7) (1 + m/8)
E4M3 = np.array([(c & 7) / 8 * 2.0 ** -6 if c >> 3 == 0 else (1 + (c & 7) / 8) * 2.0 ** ((c >> 3) - 7) for c in range(127)],
                dtype=np.float64)


def e4m3_codes(x: np.ndarray) -> np.ndarray:
    """E4M3 codes of x (|x| <= 448): the nearest value, ties to the even code; the sign in bit 7 (0 stays 0)."""
    a = np.abs(x.astype(np.float64))
    hi = np.clip(np.searchsorted(E4M3, a, side='left'), 0, 126)
    lo = np.clip(hi - 1, 0, 126)
    dlo, dhi = a - E4M3[lo], E4M3[hi] - a
    pick = np.where(dhi < dlo, hi, np.where(dlo < dhi, lo, np.where(hi % 2 == 0, hi, lo)))
    pick = np.where(a >= E4M3[126], 126, pick).astype(np.uint8)
    return np.where((x < 0) & (pick != 0), pick | 0x80, pick).astype(np.uint8)


def mxfp8(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """w F32 [N][K] (K a multiple of 32) -> (codes uint8 [N][K], scales uint8 [N][K/32])."""
    N, K = w.shape
    g = w.reshape(N, K // 32, 32)
    amax = np.abs(g).max(axis=2).astype(np.float32)
    r = (amax / np.float32(448.0)).astype(np.float32)
    bits = r.view(np.uint32)
    e = ((bits >> 23) & 0xFF).astype(np.int32) - 127 + ((bits & 0x7FFFFF) != 0).astype(np.int32)
    e = np.where(amax > 0, np.clip(e, -127, 127), -127)
    scaled = g / np.ldexp(np.float32(1.0), e)[:, :, None].astype(np.float32)
    codes = e4m3_codes(scaled).reshape(N, K)
    return codes, (e + 127).astype(np.uint8)


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, type=pathlib.Path)
    ap.add_argument('--out', required=True, type=pathlib.Path)
    a = ap.parse_args()
    cfg = json.loads((a.src / 'config.json').read_text())
    st = a.src / 'model.safetensors'
    with st.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    header.pop('__metadata__', None)
    base = 8 + n
    mm = np.memmap(st, dtype=np.uint8, mode='r')
    a.out.mkdir(parents=True, exist_ok=False)
    data = (a.out / 'dflash2.bin').open('wb')
    lines, off = ['file 0 dflash2.bin'], 0

    def put(name: str, dtype: str, arr: np.ndarray, shape: list[int]):
        nonlocal off
        pad = (-off) % ALIGN
        if pad:
            data.write(b'\0' * pad)
            off += pad
        b = np.ascontiguousarray(arr).tobytes()
        data.write(b)
        lines.append(f"t {name} {dtype} 0 {off} {len(b)} {' '.join(str(s) for s in shape)}")
        off += len(b)

    for name, info in sorted(header.items()):
        if info['dtype'] != 'BF16':
            print(f'unexpected dtype {info["dtype"]} for {name}', file=sys.stderr)
            return 1
        lo, hi = info['data_offsets']
        u16 = np.frombuffer(mm[base + lo:base + hi], dtype=np.uint16).reshape(info['shape'])
        if name.endswith(QUANT):
            N, K = info['shape']
            if N % 32 or K % 512:
                print(f'{name}: [{N}][{K}] does not fit mx_rows (N % 32, K % 512)', file=sys.stderr)
                return 1
            codes = np.empty((N, K), np.uint8)
            scales = np.empty((N, K // 32), np.uint8)
            for r0 in range(0, N, 1024):   # (in row chunks: bounded memory)
                c, s = mxfp8(bf16_to_f32(u16[r0:r0 + 1024]))
                codes[r0:r0 + 1024], scales[r0:r0 + 1024] = c, s
            stem = 'dflash2.' + name[:-len('.weight')]
            put(stem + '.weight', 'F8_E4M3', codes, [N, K])
            put(stem + '.weight_scale_inv', 'U8', scales, [N, K // 32])
            print(f'{name}: MXFP8 [{N}][{K}]', flush=True)
        else:
            put('dflash2.' + name, 'BF16', u16, info['shape'])
    data.close()
    dc = cfg.get('dflash_config', {})
    kv = {
        'model_dir': str(a.out),
        'dflash2.architecture': cfg['architectures'][0],
        'dflash2.num_hidden_layers': cfg['num_hidden_layers'],
        'dflash2.hidden_size': cfg['hidden_size'],
        'dflash2.intermediate_size': cfg['intermediate_size'],
        'dflash2.num_attention_heads': cfg['num_attention_heads'],
        'dflash2.num_key_value_heads': cfg['num_key_value_heads'],
        'dflash2.head_dim': cfg['head_dim'],
        'dflash2.rotary_dim': cfg.get('rotary_dim', cfg['head_dim']),
        'dflash2.rope_theta': cfg['rope_theta'],
        'dflash2.rms_norm_eps': cfg['rms_norm_eps'],
        'dflash2.use_gemma_norm': 1 if cfg.get('use_gemma_norm') else 0,
        'dflash2.swiglu_alpha': cfg.get('swiglu_alpha', 1.702),
        'dflash2.swiglu_limit': cfg.get('swiglu_limit', 7.0),
        'dflash2.sliding_window': cfg.get('sliding_window', 4096),
        'dflash2.vocab_size': cfg['vocab_size'],
        'dflash2.block_size': dc['block_size'],
        'dflash2.mask_token_id': dc['mask_token_id'],
        'dflash2.selector_rank': dc['selector_rank'],
        'dflash2.selector_top_k': dc['selector_top_k'],
        'dflash2.conv_group_size': dc['conv_group_size'],
        'dflash2.conv_kernel_size': dc['conv_kernel_size'],
        'dflash2.target_layer_ids': ','.join(str(x) for x in dc['target_layer_ids']),
        'dflash2.weights': 'mxfp8',
    }
    (a.out / 'mm3.txt').write_text(''.join(f'{k} {v}\n' for k, v in kv.items()))
    (a.out / 'tensors.txt').write_text('\n'.join(lines) + '\n')
    print(f'packed {len(lines) - 1} tensors, {off / 1e9:.2f} GB, into {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
