// src/mm3/mm3_native_cpu_fast.hpp - the CPU lane's reworked native NVFP4 expert kernel (kernel item 2, 2026-11-14).
//
// The same RG16 records and the same sums as the shared service's fast_ffn_cpu (Infernix-minimax
// src/ops/minimax_nvfp4_ffn/fast_cpu.cpp), with less work a byte:
// - weights: one 512-bit load serves two quarters of a unit; the codes go through a byte table to twice their E2M1 value
//   plus 12 (0..24, unsigned), so vpdpbusd takes them as they are - no absolute value, sign mask or negated activations a
//   column - and 12 times the block's activation codes is taken off once a block (DeepSeek's MXFP4 CPU kernel's trick,
//   upstream-overlap row C5). The int32 block sums are the shared kernel's exactly;
// - the E4M3 block scales decoded 16 at a time in registers instead of 16 table loads a unit;
// - a token's activations quantized once for all of its experts (the shared kernel quantizes them once a job), with
//   vector code;
// - the gate/up pass writes its hidden vector's quantized blocks itself (two row groups a task make one 16-value block),
//   so a call has two phases and one barrier between them instead of three phases.
// Activations, either:
// - kInt8: the shared kernel's INT8 per 16 (scale amax/127, round to nearest even, clamp 127): every output bit equal
//   to fast_ffn_cpu's (tests/mm3_native_cpu_fast_test.cpp);
// - kNvfp4: NVFP4 per 16 (the user's decision of 2026-11-13: 4.5 bits, an E2M1 code a value and an E4M3 scale per 16)
//   by the 5090's rule - ModelOpt's, canonical_math.h quantize_a4_codes with input scale 1: scale = E4M3(amax / 6) to
//   nearest even, saturating; codes E2M1(v / scale) to nearest even, saturating - so the CPU lane quantizes a row to
//   the same codes the 5090's kernel does, without a copy. Twice an E2M1 value times twice another is an exact integer
//   (|12 x 12|), so a block's product is exact in int32 and only the scaling and the sum over blocks round (FP32, the
//   order of the INT8 path). A block holding a non-finite value gets a NaN scale, so the output stays non-finite (the
//   helper-recovery checks rely on it; the 5090's rule has no such case).
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

namespace mm3::cpu_fast {

enum Act : int { kInt8 = 1, kNvfp4 = 2 };

// One 16-value activation block: doubled codes (INT8: the code itself; NVFP4: twice the E2M1 value, -12..12), the scale of
// those codes (INT8: amax/127; NVFP4: the E4M3 value / 2) and 12 times the codes' sum (what the unsigned weights add).
struct QBlock {
    int8_t q[16];
    float scale;
    int32_t off;
};
static_assert(sizeof(QBlock) == 24, "QBlock layout");

// The CPU has what the kernel needs (AVX-512 F/BW/VL/VNNI).
bool supported();

// FP32 -> BF16, round to nearest even, NaN quieted (the native CPU adapter's conversion).
void to_bf16(const float* x, int n, uint16_t* out);
// BF16 values (n a multiple of 16) -> n/16 blocks.
void quant(Act act, const uint16_t* x, int n, QBlock* out);
// The same one value at a time (tests).
void quant_scalar(Act act, const uint16_t* x, int n, QBlock* out);
// One E4M3 scale word -> the 5090 rule's block scale code (tests): e4m3_rn_satfinite(amax / 6).
uint8_t nvfp4_scale_word(float amax);

// One job: an expert's record (the RG16 payload, after the native header), its three multipliers (1 / weight_global),
// up to 8 columns: each column's quantized input (h/16 blocks), its hidden blocks (i/16, written by gate_up) and its
// BF16 output (h values, written by down).
struct Job {
    const uint8_t* record = nullptr;
    float multiplier[3] = {0, 0, 0};
    const QBlock* x[8] = {};
    QBlock* hidden[8] = {};
    uint16_t* y[8] = {};
    int columns = 0;
};

// Gate/up pair p (row groups 2p and 2p + 1 of the 2i interleaved gate/up rows: hidden values 16p .. 16p + 15) of every
// column, SwiGLU-OAI (g <= 7, u in [-7, 7], alpha 1.702) to BF16, then that hidden block quantized. p < i / 16.
void gate_up_pair(Act act, const Job& job, int h, int i, int p);
// Down row group r (outputs 16r .. 16r + 15) of every column, times the down multiplier, to BF16. r < h / 16.
void down_group(const Job& job, int h, int i, int r);

// The record's bytes for gate/up pairs [p0, p1) and down row groups [r0, r1): offsets from the record and lengths
// (the L3 prefetch reads exactly the bytes a worker's fixed share computes).
struct Span { size_t offset, bytes; };
Span gate_up_span(int h, int i, int p0, int p1);
Span down_span(int h, int i, int r0, int r1);
// Reads n bytes at p (every 64-byte line), stopping early when *stop turns true between 4 KiB pieces; returns the
// bytes read.
size_t touch(const uint8_t* p, size_t n, const std::atomic<bool>* stop);

}  // namespace mm3::cpu_fast
