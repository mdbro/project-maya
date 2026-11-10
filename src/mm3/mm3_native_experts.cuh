#pragma once
#include "mm3_exl3.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
namespace strata::mm3::native_experts {
constexpr int kFmt = 7;
// 256-byte-aligned prefix; payload is the donor RG16 record.
struct Header { float weight_global[3], input_global[3]; std::uint8_t reserved[232]; };
static_assert(sizeof(Header)==256, "native expert header size");
enum class Arithmetic { Dp4a, NativeFp4 };
// Workspace includes the shared B4/C8 descriptors, input/output and partial
// buffers plus only the selected arithmetic's workspace. NativeFp4 supports
// MiniMax 6144/3072 on an RTX 5090 (sm_120) only. The query is device-independent.
size_t scratch_bytes(int h, int ff, Arithmetic arithmetic = Arithmetic::Dp4a);
bool prepare(cudaStream_t, int h, int ff, void* device_buffer, size_t bytes,
             Arithmetic arithmetic = Arithmetic::Dp4a);
// Switch the existing arena in place. The caller serializes host submissions
// and switches for this stream; already-enqueued work is ordered by the stream.
// No allocation or synchronization is performed. False preserves the old mode.
bool set_arithmetic(cudaStream_t, Arithmetic);
bool get_arithmetic(cudaStream_t, Arithmetic&);
void release(cudaStream_t); // call only after stream has completed
bool gate_up(const exl3::MoeArgs&, const float* x, cudaStream_t);
bool down(const exl3::MoeArgs&, const float* shared, float* out, cudaStream_t);
bool window(const exl3::WinArgs&, cudaStream_t);
// Scatter one host-counted plan; combine only after all passes complete.
bool window_scatter(const exl3::WinArgs&, int pieces, cudaStream_t);
bool window_combine(const exl3::WinArgs&, cudaStream_t);
bool set(const exl3::SetArgs&, const float* x, int ldx, const int* row_idx, float* y, int ldy, cudaStream_t);

// ---- prompt_layerwise (item 1 of the order of 2026-11-10): a prompt read one layer at a time.
// One job is up to eight columns of one expert: its blob (the 256-byte header, then the RG16 record) and, per column,
// the entry row * 4 + pick. The inputs are the rows' BF16 values (what setup's __float2bfloat16 would make of the F32
// rows), the outputs are written per entry as BF16, which is the kernels' own output precision. Each column's
// arithmetic is that of the window path: the kernels quantize and reduce every column on its own.
struct PromptJob { const std::uint8_t* blob; int n; int entry[8]; };
// Scratch for launches of up to max_jobs jobs under the stream's selected arithmetic (prepare() on that stream).
size_t prompt_scratch_bytes(cudaStream_t, int max_jobs);
// jobs: device array; runs jobs [first, first + count) in launches of up to max_jobs. x16 [rows][h], part16 [entries][h].
bool prompt_run(const PromptJob* jobs, int first, int count, const std::uint16_t* x16, std::uint16_t* part16,
                void* scratch, size_t scratch_bytes, int max_jobs, cudaStream_t);
// F32 rows to BF16 (__float2bfloat16, as setup converts them).
bool prompt_rows_bf16(const float* x, int rows, int h, std::uint16_t* out, cudaStream_t);
// The window path's combine and residual add for `rows` rows, in its order: v = sh; v += w * part for each pick served
// on this GPU (served[r * k + pick]), in pick order; v += helper[r] when flag[r]; h += v. part16 is BF16.
bool prompt_combine(int rows, int k, int h_dim, const float* sh, const std::uint16_t* part16, const float* w,
                    const std::uint8_t* served, const float* helper, const int* flag, float* h, cudaStream_t);
}
