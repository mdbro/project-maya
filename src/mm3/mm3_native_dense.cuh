#pragma once
#include "mm3_exl3.hpp"
#include <cstddef>

// Caller owns the device buffer, one per concurrently used stream. Register
// before capture/use; release only after all work on the stream has finished.
namespace strata::mm3::native_dense {
constexpr int kFmtMXFP8 = 5, kFmtBF16 = 6;
constexpr int kBand = 256, kRows = 64;
// Outputs at most this many take split-K partials in mv (the mxc path at up to 8 rows). 512 covers the attention's k and v and
// the indexer's q (512 x 6144): alone they reach about 11% of the 5090's bandwidth, one memory round trip per 512-wide tile
// (16-17 us); split, about 5 us, as the DSpark draft's k / v of the same shape already ran. Partials need at most
// (K / 1024) x rows x 512 floats (mxc, rows <= 8) or 4 x 16 x 512 (the split kernel, rows <= 16): under 128 KB at K <= 8192,
// within scratch_bytes' 16 x kBand x 16 floats.
constexpr int kSplitN = 512;
// MV-only contexts hold just split-K partials and reject GEMM calls.
size_t scratch_bytes(int max_k, bool allow_gemm = true);
bool prepare(cudaStream_t stream, int max_k, void* buffer, size_t bytes, bool allow_gemm = true);
void release(cudaStream_t stream);
bool mv(const exl3::MvJob& job, cudaStream_t stream);
// The largest output count mv gives split-K partials: kBand by default, kSplitN under the request option dense_split.
void set_split_n(int n);
int split_n();
/// BF16 matrices with several columns: one pass reading each weight once for all of them (on, the default) or bf16_mv's
/// grid, which re-reads the matrix once a column (off). Bit for bit the same results; MAYA_MM3_MV_COLS=0 turns it off.
void set_mv_cols(bool on);
bool mv_cols_on();
bool gemm_rows(const exl3::Mat& m, const uint16_t* x, int ldx, int rows,
               float* y, int ldy, float beta, cudaStream_t stream);
}
