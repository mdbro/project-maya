// The helper cards' expert kernels built for sm_89 only (mm3_helper_sm89.hpp): the same headers as the production build.
#include "mm3_helper_sm89.hpp"
#include "mm3_helper_kernels.cuh"
#include "mm3_helper_q.cuh"

namespace mm3::helper_sm89 {
namespace {
namespace tc = mm3::helper_tc;
template <int S> cudaError_t tensor_s(int loads, const FastJob* jobs, int n, void* workspace, cudaStream_t stream) {
    switch (loads) {
        case 2: return tc::ffn2<S, 2>(jobs, n, workspace, stream);
        case 4: return tc::ffn2<S, 4>(jobs, n, workspace, stream);
        case 8: return tc::ffn2<S, 8>(jobs, n, workspace, stream);
        default: return tc::ffn1<S>(jobs, n, workspace, stream);
    }
}
}  // namespace
bool built() { return true; }
cudaError_t tensor(int warps, int loads, const FastJob* jobs, int count, void* workspace, cudaStream_t stream) {
    switch (warps) {
        case 1: return tensor_s<1>(loads, jobs, count, workspace, stream);
        case 2: return tensor_s<2>(loads, jobs, count, workspace, stream);
        case 4: return tensor_s<4>(loads, jobs, count, workspace, stream);
        default: return tensor_s<8>(loads, jobs, count, workspace, stream);
    }
}
cudaError_t q_quant_bf16(const std::uint16_t* x, std::uint8_t* q, int columns, cudaStream_t stream) {
    return mm3::helper_q::quant_bf16(x, q, columns, stream);
}
cudaError_t q_expand_packed(const std::uint8_t* packed, std::uint8_t* q, int columns, cudaStream_t stream) {
    return mm3::helper_q::expand_packed(packed, q, columns, stream);
}
cudaError_t q_expand_packed_codes(const std::uint8_t* packed, std::uint8_t* q, int columns, bool fp8, cudaStream_t stream) {
    return mm3::helper_q::expand_packed_codes(packed, q, columns, fp8, stream);
}
cudaError_t q_quant_bf16_codes(const std::uint16_t* x, std::uint8_t* q, int columns, bool fp8, cudaStream_t stream) {
    return mm3::helper_q::quant_bf16_codes(x, q, columns, fp8, stream);
}
cudaError_t q_ffn(int warps, int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* q_in,
                  std::uint8_t* hidden_q, cudaStream_t stream) {
    return mm3::helper_q::ffn(warps, loads, jobs, count, x_base, q_in, hidden_q, nullptr, stream);
}
cudaError_t q2_ffn(int warps, int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* packed,
                   void* workspace, cudaStream_t stream) {
    return mm3::helper_q::ffn2(warps, loads, jobs, count, x_base, packed, workspace, stream);
}
cudaError_t q2_pack_bf16(const std::uint16_t* x, std::uint8_t* packed, int columns, cudaStream_t stream) {
    return mm3::helper_q::pack_bf16(x, packed, columns, stream);
}
cudaError_t q_expand_bf16(const std::uint8_t* packed, std::uint16_t* out, int columns, cudaStream_t stream) {
    return mm3::helper_q::expand_bf16(packed, out, columns, stream);
}
cudaError_t q3_ffn(int warps, int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint16_t* expanded, void* workspace, cudaStream_t stream) {
    return mm3::helper_q::ffn3(warps, loads, jobs, count, x_base, expanded, workspace, stream);
}
cudaError_t q_fp8_ffn(int warps, int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* q, std::uint8_t* hidden, cudaStream_t stream) {
    return mm3::helper_q::ffn_fp8(warps, loads, jobs, count, x_base, q, hidden, stream);
}
cudaError_t q_bits_ffn(int warps, int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* q, std::uint8_t* hidden, cudaStream_t stream) {
    return mm3::helper_q::ffn_int_bits(warps, loads, jobs, count, x_base, q, hidden, stream);
}
cudaError_t q_queued(int family, int warps, int loads, const FastJob* jobs, const std::uint16_t* x_base, const std::uint8_t* in,
                     void* workspace, const int* count, const int* kind, int wanted, int blocks, cudaStream_t stream) {
    return mm3::helper_q::ffn_queued<true>(family, warps, loads, jobs, x_base, in, workspace, count, kind, wanted, blocks, stream);
}
}  // namespace mm3::helper_sm89
