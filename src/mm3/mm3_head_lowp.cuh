// src/mm3/mm3_head_lowp.cuh - kernel item 10 (2026-11-14): the output head [vocab x dim] in fewer bytes, quantized at
// load from the checkpoint's BF16 head, for the head passes of the target's verify rows and DFlash2's top-k.
//
// The BF16 head reads 2.46 GB a pass and its kernels already run at 87-93% of the 5090's 1.79 TB/s (bf16_mv 1.48 ms,
// bf16_mv_cols 1.49-1.59 ms), so a pass gets faster only with fewer bytes. Two formats, weights only - the activations
// stay FP32, every product and sum FP32:
//   kDense8: signed INT8 weight, binary16 scale per row group of 32 (8.5 bits/weight).
//   kFp8:   E4M3 a weight, a UE8M0 (power-of-two) scale a 32 (the checkpoint's own MXFP8 layout: 8.25 bits, 1.27 GB);
//           the scale the smallest power of two with amax / scale <= 448, so nothing saturates;
//   kNvfp4: E2M1 a weight, an E4M3 scale a 16 and one FP32 scale for the matrix (4.5 bits, 0.69 GB), ModelOpt's
//           NVFP4 weight rule: global = amax / (6 * 448), block scale = E4M3(block amax / (6 * global)) to nearest even,
//           codes E2M1(v / (block scale * global)) to nearest even, saturating.
// Dense8 is the requested default from 2026-11-17. The head fidelity review remains separate.
#pragma once

#include <cuda_runtime.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mm3::head_lowp {

__host__ __device__ inline float __int_as_float_hd(int i) {
#if defined(__CUDA_ARCH__)
    return __int_as_float(i);
#else
    float f;
    std::memcpy(&f, &i, sizeof f);
    return f;
#endif
}

enum Fmt : int { kFp8 = 1, kNvfp4 = 2, kDense8 = 4 };

struct Head {
    int fmt = 0, n = 0, k = 0;
    uint8_t* w = nullptr;    // kFp8: E4M3 [n][k]; kNvfp4: E2M1 [n][k/2], element 2i in the low nibble of byte i
    uint8_t* sc = nullptr;   // kFp8: UE8M0 [n][k/32]; kNvfp4: E4M3 [n][k/16]
    float global = 1.0f;     // kNvfp4: the matrix's scale (a dequantized weight = E2M1 x E4M3 x global)
    float* amax = nullptr;   // (device scratch: the matrix's largest magnitude while quantizing; kNvfp4)
};

// k a multiple of 256; the bytes of w and sc.
bool shape_ok(int fmt, int n, int k);
size_t weight_bytes(int fmt, int n, int k);
size_t scale_bytes(int fmt, int n, int k);
// Allocate (cudaMalloc) and quantize from a device BF16 [n][k] head; reads its global scale back (one sync). False with
// err on failure (nothing left allocated).
bool make(int fmt, const uint16_t* bf16, int n, int k, Head& out, cudaStream_t s, const char** err);
void release(Head& h);
// 0 preserves the original launch shapes; 1 uses eight rows a warp at 2-8
// columns, reducing repeated activation loads. Changes only launch grouping.
// Set between forwards; a captured caller must invalidate its graphs on change.
void set_shape_mode(int mode);
int shape_mode();
// The public MV contract batches at most16 columns (internally in runs of8).
// Zero means the encoded copy is absent/invalid. Independent of BF16 ownership.
int mv_columns(const Head& h);
// y[c][r] = sum_j W[r][j] x[c][j] for the ncol (1..16) columns x (ldx floats apart) into y (ldy apart).
bool mv(const Head& h, const float* x, int ldx, int ncol, float* y, int ldy, cudaStream_t s);

// Bench and tests: one tuned shape of the kernels (rows a warp, warps a block), and the dequantized value of a weight.
bool mv_shape(const Head& h, const float* x, int ldx, int ncol, float* y, int ldy, int rows_per_warp, int warps,
              cudaStream_t s);
__host__ __device__ inline float e4m3_float(uint8_t b) {   // (exact; 0x7f / 0xff: NaN)
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    float v = e == 0 ? float(m) * (1.0f / 512.0f) : float((8u + m) << e) * (1.0f / 1024.0f);
    if ((b & 0x7fu) == 0x7fu) v = v - v + __int_as_float_hd(0x7fc00000);
    return (b & 0x80u) ? -v : v;
}
__host__ __device__ inline float e2m1_float(uint8_t code) {
    const uint32_t m = code & 7u;
    const float v = m < 4 ? 0.5f * float(m) : float((2u + (m & 1u)) << ((m >> 1) - 1u)) * 0.5f;
    return (code & 8u) ? -v : v;
}
// a weight's dequantized value from host copies of w and sc (tests)
inline float weight_host(const Head& h, const uint8_t* w, const uint8_t* sc, int r, int j) {
    if (h.fmt == kFp8) {
        const int e = sc[(size_t) r * (h.k / 32) + j / 32];
        return e4m3_float(w[(size_t) r * h.k + j]) * std::ldexp(1.0f, e - 127);
    }
    if (h.fmt == kDense8) {
        uint16_t word; std::memcpy(&word, sc + 2 * ((size_t)r * (h.k/32) + j/32), 2);
        int exp = (word >> 10) & 31, mant = word & 1023;
        const float scale = std::ldexp(float(exp ? 1024 + mant : mant), exp ? exp - 25 : -24);
        return float(reinterpret_cast<const int8_t*>(w)[(size_t)r*h.k+j]) * scale;
    }
    const uint8_t byte = w[(size_t) r * (h.k / 2) + j / 2];
    return e2m1_float((j & 1) ? byte >> 4 : byte & 15u) * e4m3_float(sc[(size_t) r * (h.k / 16) + j / 16]) * h.global;
}

}  // namespace mm3::head_lowp
