#pragma once
#include "mm3_exl3.hpp"
#include <cstddef>

// Caller owns the device buffer, one per concurrently used stream. Register
// before capture/use; release only after all work on the stream has finished.
namespace strata::mm3::native_dense {
constexpr int kFmtMXFP8 = 5, kFmtBF16 = 6;
constexpr int kBand = 256, kRows = 64;
// MV-only contexts hold just split-K partials and reject GEMM calls.
size_t scratch_bytes(int max_k, bool allow_gemm = true);
bool prepare(cudaStream_t stream, int max_k, void* buffer, size_t bytes, bool allow_gemm = true);
void release(cudaStream_t stream);
bool mv(const exl3::MvJob& job, cudaStream_t stream);
bool gemm_rows(const exl3::Mat& m, const uint16_t* x, int ldx, int rows,
               float* y, int ldy, float beta, cudaStream_t stream);
}
