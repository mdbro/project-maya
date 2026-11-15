// Kernel item 5: the helper cards' expert kernels compiled for the RTX 4080's own chip (sm_89, Ada). The production
// build targets sm_86 and sm_120, so the 4080 runs the sm_86 code; this library (CUDA architecture 89 only,
// mm3_helper_sm89.cu) holds the same kernels - the tensor-core kernel of mm3_helper_kernels.cuh and the NVFP4-activation
// kernel of mm3_helper_q.cuh - so a request can run them instead (third_sm89, off by default). Same source, same
// arithmetic: only the compiler's code for the chip differs.
#pragma once
#include "ops/minimax_nvfp4_ffn/fast.h"
#include <cuda_runtime_api.h>
#include <cstdint>

namespace mm3::helper_sm89 {
using infernix::ops::minimax_ffn::FastJob;
#if defined(__CUDACC__)
// Whether this build holds the sm_89 kernels (a device of compute capability 8.9 can run them).
bool built();
// helper_tc::ffn1 (loads 1) or ffn2<S, loads> with S warps, as NativeHelper's tensor-core launch
cudaError_t tensor(int warps, int loads, const FastJob* jobs, int count, void* workspace, cudaStream_t stream);
// helper_q's kernels (mm3_helper_q.cuh)
cudaError_t q_quant_bf16(const std::uint16_t* x, std::uint8_t* q, int columns, cudaStream_t stream);
cudaError_t q_expand_packed(const std::uint8_t* packed, std::uint8_t* q, int columns, cudaStream_t stream);
cudaError_t q_ffn(int warps, int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* q_in,
                  std::uint8_t* hidden_q, cudaStream_t stream);
// helper_q's variant 2 (kernel item 5b): its inputs as packed rows, its hidden vector in the tensor-core workspace
cudaError_t q2_ffn(int warps, int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* packed,
                   void* workspace, cudaStream_t stream);
cudaError_t q2_pack_bf16(const std::uint16_t* x, std::uint8_t* packed, int columns, cudaStream_t stream);
#else
// (the CPU helper tests: no CUDA compiler, never selected)
inline bool built() { return false; }
inline cudaError_t tensor(int, int, const FastJob*, int, void*, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t q_quant_bf16(const std::uint16_t*, std::uint8_t*, int, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t q_expand_packed(const std::uint8_t*, std::uint8_t*, int, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t q_ffn(int, int, const FastJob*, int, const std::uint16_t*, const std::uint8_t*, std::uint8_t*, cudaStream_t) {
    return cudaErrorInvalidValue;
}
inline cudaError_t q2_ffn(int, int, const FastJob*, int, const std::uint16_t*, const std::uint8_t*, void*, cudaStream_t) { return cudaErrorInvalidValue; }
inline cudaError_t q2_pack_bf16(const std::uint16_t*, std::uint8_t*, int, cudaStream_t) { return cudaErrorInvalidValue; }
#endif
}  // namespace mm3::helper_sm89
