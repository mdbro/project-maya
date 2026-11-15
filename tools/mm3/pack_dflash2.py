#!/usr/bin/env python3
"""Pack RadixArk's DFlash2 draft for MiniMax-M3.1-Flash (BF16 safetensors) as a small maya-mm3 pack, its projections in
MXFP8 (step 6, the user's decision: DFlash2 at 8 bits) or, with --weights nvfp4, in NVFP4 (kernel item 6: "mxfp8 vs nvfp4
with native fp4 on 5090"; the drafter's weight fidelity may drop if usability holds).

The pack: <out>/mm3.txt (the draft's geometry as `dflash2.*` keys, model_dir = <out>), <out>/tensors.txt and one data file
<out>/dflash2.bin. Tensor names keep the checkpoint's, prefixed `dflash2.`.
  MXFP8 (the layout mx_rows reads, as the target's own MXFP8 tensors are stored): each layer's q / k / v / o, gate / up /
    down and the two conv kernel projections - `<name>.weight` E4M3 codes [N][K] (F8_E4M3) and `<name>.weight_scale_inv`
    a UE8M0 byte for every 32 inputs [N][K/32] (U8). The scale exponent of a group of 32 is the smallest e with every
    |w| / 2^e at most 448 (e = ceil(log2(amax / 448)) on the F32 value, as the engine's FP8 KV path computes it); a
    code is the nearest E4M3 value of w / 2^e, ties to the even code; an all-zero group has the byte 0.
  NVFP4 (--weights nvfp4, the same matrices; mm3_draft_fp4.cuh's kernels read them on the 5090's FP4 tensor cores,
    retiled at the load): `<name>.weight` E2M1 codes [N][K/2] (U8: input 2i in the low nibble of byte i, the sign in each
    nibble's bit 3), `<name>.weight_scale` an E4M3 byte for every 16 inputs [N][K/16] (F8_E4M3) and `<name>.weight_scale_2`
    the matrix's global scale (F32 [1]) - a weight is e2m1(code) * e4m3(scale) * global, NVIDIA's two-level NVFP4. The
    global scale is amax(|W|) / (6 * 448); a group's scale byte is, of the nearest E4M3 value (ties to even) of
    amax(group) / 6 / global and its two neighbours, the one that gives the group the least squared error (--nvfp4-scale
    mse, the built-in: 8-9% less RMS error than amax on these matrices) or that nearest value alone (--nvfp4-scale amax,
    as ModelOpt computes it); a code is the nearest E2M1 value of
    w / (scale * global), ties to the even code, saturating at 6. Each matrix's relative RMS error against its BF16 values
    is printed, with MXFP8's beside it.
  BF16 as stored: fc, the norms, the conv base kernels, the selector's hidden projection and codebooks.
Usage: pack_dflash2.py --src <dir with model.safetensors and config.json> --out <new pack dir>
                       [--weights mxfp8|nvfp4] [--nvfp4-scale amax|mse]"""
from __future__ import annotations
import argparse, json, os, pathlib, struct, sys
# (numpy's huge-page madvise for large arrays: on this WSL host, with most memory pinned by the engine, a fresh array's
#  first touch spun in the kernel for 20+ minutes on 2026-11-14 - all system time, no faults; without it a 1024-row chunk
#  takes seconds. Read when numpy is imported.)
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE', '0')
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


# E2M1's magnitudes by code (0..7); a code's bit 3 is the sign
E2M1 = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)


def e2m1_codes(q: np.ndarray) -> np.ndarray:
    """E2M1 codes of q: the nearest value, ties to the even code (the midpoints 0.25, 1.25, 2.5 and 5 go down, 0.75, 1.75
    and 3.5 up), saturating at 6; the sign in bit 3 (kept for a value that rounds to 0, as cvt.rn.satfinite does)."""
    a = np.abs(q)
    c = np.where(a > 0.25, 1, 0)
    c = np.where(a >= 0.75, 2, c)
    c = np.where(a > 1.25, 3, c)
    c = np.where(a >= 1.75, 4, c)
    c = np.where(a > 2.5, 5, c)
    c = np.where(a >= 3.5, 6, c)
    c = np.where(a > 5.0, 7, c)
    return (c | (np.signbit(q).astype(np.int64) << 3)).astype(np.uint8)


def e2m1_values(c: np.ndarray) -> np.ndarray:
    v = E2M1[c & 7]
    return np.where(c & 8, -v, v).astype(np.float32)


def nvfp4_global(amax: float) -> np.float32:
    """The matrix's global scale (dequantization multiplier): amax / (6 * 448), so the largest group's scale is 448."""
    return np.float32(amax) / np.float32(2688.0) if amax > 0 else np.float32(1.0)


def nvfp4(w: np.ndarray, gs: np.float32, rule: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """w F32 [N][K] (K a multiple of 16), gs its global scale -> (codes uint8 [N][K/2], scale bytes uint8 [N][K/16],
    the dequantized values F32 [N][K])."""
    N, K = w.shape
    g = w.reshape(N, K // 16, 16)
    amax = np.abs(g).max(axis=2).astype(np.float32)
    s = (amax / np.float32(6.0) / gs).astype(np.float32)
    sc = e4m3_codes(s)                      # (s <= 448: never saturates)

    def quant(scb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        denom = (E4M3[scb & 0x7F].astype(np.float32) * gs).astype(np.float32)
        q = np.where(denom[:, :, None] > 0, g / np.where(denom > 0, denom, 1)[:, :, None], 0).astype(np.float32)
        c = e2m1_codes(q)
        return c, (e2m1_values(c) * denom[:, :, None]).astype(np.float32)

    codes, deq = quant(sc)
    if rule == 'mse':   # (the scale byte and its neighbours: the group's least squared error)
        best = ((deq - g) ** 2).sum(axis=2)
        for d in (-1, 1):
            alt = np.clip(sc.astype(np.int32) + d, 0, 126).astype(np.uint8)
            c2, q2 = quant(alt)
            e2 = ((q2 - g) ** 2).sum(axis=2)
            better = e2 < best
            sc = np.where(better, alt, sc)
            codes = np.where(better[:, :, None], c2, codes)
            deq = np.where(better[:, :, None], q2, deq)
            best = np.where(better, e2, best)
    packed = (codes[:, :, 0::2] | (codes[:, :, 1::2] << 4)).astype(np.uint8).reshape(N, K // 2)
    return packed, sc.astype(np.uint8), deq.reshape(N, K)


def mxfp8_values(codes: np.ndarray, scales: np.ndarray) -> np.ndarray:
    N, K = codes.shape
    v = E4M3[codes & 0x7F].astype(np.float32)
    v = np.where(codes & 0x80, -v, v).reshape(N, K // 32, 32)
    return (v * np.ldexp(np.float32(1.0), scales.astype(np.int32) - 127)[:, :, None]).reshape(N, K).astype(np.float32)


def bf16_to_f32(u16: np.ndarray) -> np.ndarray:
    return (u16.astype(np.uint32) << 16).view(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, type=pathlib.Path)
    ap.add_argument('--out', required=True, type=pathlib.Path)
    ap.add_argument('--weights', choices=('mxfp8', 'nvfp4'), default='mxfp8')
    ap.add_argument('--nvfp4-scale', choices=('amax', 'mse'), default='mse')
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
        if name.endswith(QUANT) and a.weights == 'nvfp4':
            N, K = info['shape']
            if N % 16 or K % 64:
                print(f'{name}: [{N}][{K}] does not fit mm3_draft_fp4 (N % 16, K % 64)', file=sys.stderr)
                return 1
            amax = 0.0
            for r0 in range(0, N, 1024):
                amax = max(amax, float(np.abs(bf16_to_f32(u16[r0:r0 + 1024])).max()))
            gs = nvfp4_global(amax)
            codes = np.empty((N, K // 2), np.uint8)
            scales = np.empty((N, K // 16), np.uint8)
            err4 = err8 = ref = 0.0
            for r0 in range(0, N, 1024):
                w = bf16_to_f32(u16[r0:r0 + 1024])
                c, s, deq = nvfp4(w, gs, a.nvfp4_scale)
                codes[r0:r0 + 1024], scales[r0:r0 + 1024] = c, s
                c8, s8 = mxfp8(w)
                err4 += float(((deq.astype(np.float64) - w) ** 2).sum())
                err8 += float(((mxfp8_values(c8, s8).astype(np.float64) - w) ** 2).sum())
                ref += float((w.astype(np.float64) ** 2).sum())
            stem = 'dflash2.' + name[:-len('.weight')]
            put(stem + '.weight', 'U8', codes, [N, K // 2])
            put(stem + '.weight_scale', 'F8_E4M3', scales, [N, K // 16])
            put(stem + '.weight_scale_2', 'F32', np.array([gs], dtype=np.float32), [1])
            rel = lambda e: (e / ref) ** 0.5 if ref > 0 else 0.0
            print(f'{name}: NVFP4 [{N}][{K}] global {float(gs):.6g}; relative RMS error NVFP4 {rel(err4):.5f}, '
                  f'MXFP8 {rel(err8):.5f}', flush=True)
        elif name.endswith(QUANT):
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
        'dflash2.weights': a.weights,
    }
    if a.weights == 'nvfp4':
        kv['dflash2.nvfp4_scale'] = a.nvfp4_scale
    (a.out / 'mm3.txt').write_text(''.join(f'{k} {v}\n' for k, v in kv.items()))
    (a.out / 'tensors.txt').write_text('\n'.join(lines) + '\n')
    print(f'packed {len(lines) - 1} tensors, {off / 1e9:.2f} GB, into {a.out}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
