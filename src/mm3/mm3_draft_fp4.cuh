// src/mm3/mm3_draft_fp4.cuh - DFlash2's projections in NVFP4 on the RTX 5090's FP4 tensor cores (kernel item 6, "the
// drafter, including NVFP4 on the 5090's native FP4 against today's MXFP8"; the user: "drafter and head can decrease in
// weight fidelity if we show no decrease in usibility", "mxfp8 vs nvfp4 with native fp4 on 5090").
//
// y [rows][ldy] = beta y + Q(x) [rows][K] . W^T for an NVFP4 W [N][K]: E2M1 codes, an E4M3 scale every 16 inputs and one
// F32 global scale a matrix (tools/mm3/pack_dflash2.py --weights nvfp4; the weight is code * scale * global). Q is NVFP4
// too, computed from the F32 rows the drafter holds by quantize(): per row a global scale gx = max |x| / (6 * 448), then
// per 16 inputs an E4M3 scale RNE(max |x / gx| / 6) and E2M1 codes RNE(x / gx / scale), saturating (the target's own
// activation rule, kernel 3's "4-bit" choice, with the row's global scale added so that no block's scale saturates or
// flushes). terms 2: a second NVFP4 term encodes what the first left (x / gx - code * scale, its own E4M3 scales, the same
// gx), both multiplied with the same weight fragment - activations near 8-bit fidelity for one more MMA a tile.
// The products: mma.sync kind::mxf4nvf4.block_scale.scale_vec::4X m16n8k64 (sm_120a), the weight's and the activation's
// E4M3 scales given to the instruction, FP32 accumulation; then times the weight's global scale and the row's gx.
//
// The weights live in this kernel's tiled layout (tile_host, from the pack's row-major NVFP4 at the drafter's load): a
// tile is 16 rows x 64 inputs, 512 bytes of codes in which lane l = 4g + t of a warp reads its whole A fragment as one
// 16-byte load - words (a0, a1, a2, a3) = rows (g, g + 8, g, g + 8), inputs (8t .. 8t + 7, the same, 32 + 8t .. 32 + 8t + 7,
// the same), input 8t + j in nibble j (PTX's m16n8k64 4-bit fragment; Infernix's ldmatrix mapping in nvfp4_a4_mma.cuh
// reads the same) - so a warp's loads are 512 contiguous bytes. The scales: 64 bytes a tile, row r's 4 bytes (one a
// 16-input group, in order) at 4r; lane (g, t) passes row g + 8 (t & 1) to the instruction (thread-id selector 0).
// Tiles run along K within a 16-row block: [N / 16][K / 64].
// The activations (quantize): per term, for each 64-input chunk and each 8-row tile, 256 bytes of codes - lane (g, t)'s
// 8 bytes = (b0, b1) = row g's inputs (8t .. 8t + 7, 32 + 8t .. 32 + 8t + 7) - then, after all the chunks' codes, 32 bytes
// of scales a chunk and tile (row g's 4 bytes at 4g).
//
// A warp takes one 16-row block over a slice of the chunks (warp w of KS: chunks w, w + KS, ...); the block's KS partial
// sums are added in shared memory in warp order: a fixed order, so the results repeat bit for bit. Built for sm_120a alone
// (its own library, as the FP4 expert kernel and dense_fp8 are): supported() is false on any other device, and a drafter
// pack in NVFP4 is refused there.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace mm3::draft_fp4 {

/// An NVFP4 matrix [N][K] on the device in the tiled layout above (N % 16 == 0, K % 64 == 0).
struct Mat {
    const uint8_t* w = nullptr;    // codes, N * K / 2 bytes, 16-byte aligned
    const uint8_t* sc = nullptr;   // E4M3 scales, N * K / 16 bytes, 4-byte aligned
    float gs = 1.0f;               // the global scale (dequantization multiplier)
    int N = 0, K = 0;
};
/// One matrix of a launch: jobs of one launch share `rows` and their K's quantized activations may differ (each job
/// names its own quantize() output).
struct Job {
    Mat m;
    const uint8_t* xq = nullptr;   // quantize()'s codes and scales for K = m.K (act_bytes(K, terms)), 16-byte aligned
    const float* xgs = nullptr;    // quantize()'s row global scales (device, >= rows floats)
    float* y = nullptr;            // F32 [rows][ldy]
    int ldy = 0;
    float beta = 0.0f;
};
constexpr int kMaxJobs = 4;
constexpr int kMaxRows = 16;
constexpr int kMaxTerms = 2;

/// Device bytes of a tiled matrix's codes and scales.
inline size_t codes_bytes(int N, int K) { return (size_t) N * (size_t) K / 2; }
inline size_t scales_bytes(int N, int K) { return (size_t) N * (size_t) K / 16; }
/// The bytes quantize() writes for up to kMaxRows rows of K inputs and `terms` terms.
inline size_t act_bytes(int K, int terms) { return (size_t) terms * (size_t) (K / 64) * 2 * (256 + 32); }

// ---- the layouts, shared by the kernels, the host tiling and the host tests (byte offsets)
/// Tiled codes: the 4-byte word of fragment register `reg` (0..3: a0..a3) of lane (g, t) in tile (rb, kc).
__host__ __device__ inline size_t w_word(int KC, int rb, int kc, int g, int t, int reg) {
    return ((size_t) rb * KC + kc) * 512 + 16 * (4 * g + t) + 4 * reg;
}
/// Tiled scales: row r (0..15) of the block's tile kc, 4 bytes (group j at + j).
__host__ __device__ inline size_t w_scale(int KC, int rb, int kc, int r) { return ((size_t) rb * KC + kc) * 64 + 4 * r; }
/// Activation codes of term `term`: the 4-byte word holding inputs [8 run, 8 run + 8) of chunk kc (run 0..7) of row x.
__host__ __device__ inline size_t x_word(int KC, int term, int kc, int x, int run) {
    return (size_t) term * KC * 2 * 288 + ((size_t) kc * 2 + (x >> 3)) * 256 + 8 * (4 * (x & 7) + (run & 3)) + 4 * (run >> 2);
}
/// Activation scales of term `term`: row x's byte of 16-input group j (0..3) of chunk kc.
__host__ __device__ inline size_t x_scale(int KC, int term, int kc, int x, int j) {
    return (size_t) term * KC * 2 * 288 + (size_t) KC * 2 * 256 + ((size_t) kc * 2 + (x >> 3)) * 32 + 4 * (x & 7) + j;
}

/// Host: the pack's row-major NVFP4 (codes [N][K / 2], input 2i in the low nibble of byte i; E4M3 scales [N][K / 16])
/// into the tiled layout (wt: codes_bytes, st: scales_bytes).
void tile_host(const uint8_t* codes, const uint8_t* scales, int N, int K, uint8_t* wt, uint8_t* st);

/// Whether the current device runs these kernels (an sm_120 device, the image this library holds).
bool supported();
/// quantize(): rows (1 .. kMaxRows) of K (% 64 == 0) F32 inputs, ldx floats apart (16-byte aligned, ldx % 4 == 0), into
/// xq (act_bytes(K, terms)) and xgs (kMaxRows floats; the rows of the last 8-row tile past `rows` get zero codes and a
/// zero global scale). A non-finite input makes its row's global scale NaN (the outputs then NaN). False: not launched.
bool quantize(const float* x, int rows, int ldx, int K, int terms, uint8_t* xq, float* xgs, cudaStream_t s);
/// The products: n jobs (1 .. kMaxJobs), rows 1 .. kMaxRows, terms 1 or 2 (as quantized). ks: warps a block (0 picks by
/// the grid: 4 when the jobs give at least 2 blocks an SM, else 8, else 16; 4, 8 or 16 force it). False: not launched.
bool launch(const Job* jobs, int n, int rows, int terms, cudaStream_t s, int ks = 0);

/// Host references for the tests and the bench: the E2M1 value of a code (sign in bit 3), an E4M3 byte's value, and
/// quantize()'s arithmetic for one row (codes and scales in the row-major form: codes[K / 2], scales[K / 16] per term).
float e2m1_value(unsigned code);
float e4m3_value(uint8_t b);
float quantize_row_host(const float* x, int K, int terms, uint8_t* codes, uint8_t* scales);

}  // namespace mm3::draft_fp4
