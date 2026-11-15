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
cudaError_t q_ffn(int warps, int loads, const FastJob* jobs, int count, const std::uint16_t* x_base, const std::uint8_t* q_in,
                  std::uint8_t* hidden_q, cudaStream_t stream) {
    return mm3::helper_q::ffn(warps, loads, jobs, count, x_base, q_in, hidden_q, nullptr, stream);
}
}  // namespace mm3::helper_sm89
