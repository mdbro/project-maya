// src/mm3/mm3_dense_fp8.cuh - the dense MXFP8 matrices with 8-bit activations on the RTX 5090's tensor cores
// (kernel item 3, "the 5090's preparation"; the user's decision of 2026-11-13: "MXFP8 weights x FP8 activations on the
// 5090's tensor cores for the attention projections and layers 0-2, and the target's other MXFP8 matrices, the shared
// expert and the index projections", adopted per mode wherever faster).
//
// y [rows][ldy] = beta y + Q(x) [rows][ldx] . W^T for an MXFP8 W [N][K] (E4M3 bytes, a UE8M0 scale every 32 inputs: the
// checkpoint's bytes, read as stored - never repacked or requantized), where Q is the checkpoint's activation scheme
// ("activation_scheme dynamic", block 32): each run of 32 inputs of a row to E4M3 (nearest, ties to even) under a
// UE8M0 scale 2^e, e the smallest with max |x| / 2^e <= 448 (no value saturates), computed inside the kernel from the
// F32 rows the callers already hold - so it is a drop-in for mm3_kernels.cuh's mx_rows, with no extra launch.
// The products run as mma.sync kind::mxf8f6f4.block_scale m16n8k32 (sm_120a): the weight's and the activation's UE8M0
// scales go to the instruction as they are, FP32 accumulation. A warp takes 16 output rows over a slice of K, a block
// KS warps the slices of the same 16 rows, their sums added in shared memory in warp order: a fixed order, so the
// results repeat bit for bit (not those of mx_rows: other arithmetic, for speed only - CLAUDE.md, Kernels).
//
// Built for sm_120a alone (its own library, as the FP4 expert kernel is): supported() is false on any other device,
// and callers keep mx_rows there (the 3090 and the 4080 never run these matrices).
#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

namespace mm3::dense_fp8 {

/// One matrix of a launch. Jobs of one launch share `rows`; each has its own activations and output.
struct Job {
    const uint8_t* w = nullptr;    // E4M3 [N][K], 16-byte aligned
    const uint8_t* sc = nullptr;   // UE8M0 [N][K / 32], 4-byte aligned
    int N = 0, K = 0;
    const float* x = nullptr;      // F32 [rows][ldx], 16-byte aligned, ldx % 4 == 0
    int ldx = 0;
    float* y = nullptr;            // F32 [rows][ldy]
    int ldy = 0;
    float beta = 0.0f;
};
constexpr int kMaxJobs = 8;
constexpr int kMaxRows = 16;
// Prompt-only: fragment loads through 32 rows, prepacked fragment loads through 64.
constexpr int kMaxPromptRows = 64;

/// Whether the current device runs these kernels (an sm_120 device, the image this library holds).
bool supported();
/// Whether a matrix's shape and the job's pointers fit the kernel: N % 16 == 0, K % 128 == 0, alignment as above.
bool job_ok(const Job& j);
/// The launch: n jobs (1 .. kMaxJobs), rows 1 .. kMaxRows. ks: warps a block (the slices of K): 0 picks by the grid
/// (4 when the jobs give at least 2 blocks an SM, else 8, else 16); 4, 8 or 16 force it (the bench's sweep).
/// False: invalid operands or unsupported device, or a CUDA launch error. A launch
/// error can occur after a packing kernel was queued; the workspace remains live
/// until stream completion even on failure.
/// kernel: 0 the fragment-load kernel, 1 the staged kernel (cp.async in whole lines, pipelined), 2 by rows (1 up to 8
/// rows, 0 above: each where it measured faster), -1 the built-in (set_kernel).
// IDs 3/4 pre-pack each distinct (x, K, ldx) once per grouped launch. ID 3 uses
// fragment weight loads (ID 0), ID 4 the identical staged weight pipeline of ID 1.
// Caller owns a 16-byte aligned
// workspace of workspace_bytes(), kept live until stream/graph completion and exclusive to concurrent launches.
// It must exist before graph capture. Other IDs do not use it. Insufficient storage rejects the launch.
std::size_t workspace_bytes(const Job* jobs, int n, int rows);
bool launch(const Job* jobs, int n, int rows, cudaStream_t s, int ks = 0, int kernel = -1,
            void* workspace = nullptr, std::size_t workspace_size = 0);
/// The built-in kernel of launch() (the request option dense_fp8_kernel sets it).
void set_kernel(int k);
int kernel();

/// The quantization of one run of 32 activations as the kernel does it, for the host-side references of the tests:
/// the UE8M0 scale byte (255: a non-finite input, which the kernel passes on as NaN) and the 32 E4M3 bytes.
uint8_t quantize_block_host(const float* x, uint8_t* codes);

}  // namespace mm3::dense_fp8
