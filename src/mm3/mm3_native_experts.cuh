#pragma once
#include "mm3_exl3.hpp"
#include <cstddef>
#include <cstdint>
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
bool set(const exl3::SetArgs&, const float* x, int ldx, const int* row_idx, float* y, int ldy, cudaStream_t);
}
