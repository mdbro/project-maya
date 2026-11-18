// Kernel item 7: the 5090's FP4 expert kernel, with its loads batched (fp4_kernel; off unless a request asks).
//
// The user (CLAUDE.md, Kernels): "The kernel overhaul may try other 5090 arithmetic only to make it faster, for example per
// job width. Never for fidelity or repeatability"; "Quantifying experts is not permitted" (no expert requantization).
//
// Today's kernel is the donor's (Infernix-minimax, ops/minimax_nvfp4_ffn/native_fp4.cu): per job, the inputs encoded to
// NVFP4 (prepare), then the gate/up and the down projections on block-scaled FP4 tensor cores (mma.sync kind::mxf4nvf4
// m16n8k64, E4M3 scales per 16). Each warp walks its half of a 16-row group's inputs one 64-input step at a time: one step's
// loads in flight a warp, so at one row (192 blocks of four warps for an expert's up projection on 170 SMs) its time is a
// chain of memory latencies rather than the record's bytes over the bandwidth - the pattern batching fixed on the 3090
// (project2, 2026-11-11).
//
// This kernel is the donor's, copied with one change: each warp issues the loads of P steps (codes, scale bytes,
// activations) before it multiplies any of them, then multiplies them in the same order. Every product, every sum and its
// order, the hidden vector's encoding and the outputs are the donor's, so the outputs are bit for bit the donor's (the
// bench checks it). P = 1 is the donor's loop exactly. Same FastJob contract, the same workspace and its layout.
#pragma once
#include "ops/minimax_nvfp4_ffn/fast.h"
#include <cuda_runtime_api.h>
#include <cstddef>

namespace mm3::fp4_fast {
using infernix::ops::minimax_ffn::FastJob;
// As donor::native_fp4_ffn: count jobs of 1..cols columns (0: inactive), cols 1..8, work of count *
// donor::native_fp4_workspace_bytes(cols) bytes, 256-aligned. loads: the steps a batch, 1, 2, 4 or 8; 12 selects P=4 with parallel hidden encoding. Throws on bad
// operands, as the donor does.
void ffn(const FastJob* jobs, int count, int cols, void* work, int loads, cudaStream_t stream, cudaEvent_t down_wait = nullptr);
// Maya-only row-indexed gate/up input; same per-job hidden workspace and output.
// packed_rows: native layout, packed_row_count rows of6144 values,3456 bytes each;
// row_ids[job*cols+column] in [0,packed_row_count) for each active column.
// Device metadata/buffers remain live and immutable until stream completion.
// Skips prepare; no allocation/sync. Caller enqueues encoding before this call.
void ffn_rows(const FastJob* jobs, int count, int cols, void* work, int loads,
              const std::uint8_t* packed_rows, const int* row_ids, int packed_row_count, cudaStream_t stream);
// Standalone representation gate (same encoder as fused native-expert setup).
void pack_rows_f32(const float* x, int rows, int ldx, std::uint8_t* packed, cudaStream_t stream);
inline bool valid_loads(int p) { return p == 1 || p == 2 || p == 4 || p == 8 || p == 12; }
}  // namespace mm3::fp4_fast
