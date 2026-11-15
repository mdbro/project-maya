#pragma once
#include "mm3_head_lowp.cuh"
#include "mm3_native_dense.cuh"
#include <algorithm>

namespace mm3::head_dispatch {
// Query the original matrix only when it supplies this head pass. A dropped
// BF16 allocation must never determine a live encoded copy's batching.
template<class OriginalCapacity>
int mv_columns(const head_lowp::Head* active, OriginalCapacity original) {
    return active ? head_lowp::mv_columns(*active) : std::max(1,original());
}

// Existing full-logits arithmetic: FP8 copy through the native MXFP8 GEMM,
// NVFP4 copy through its F32-activation MV, ordinary head through its GEMM.
// The caller supplies its GEMM, allowing the same dispatch to be gated without
// loading a whole model. BF16 metadata is read only for the ordinary route.
template<class Gemm>
bool rows(const head_lowp::Head* active, const strata::mm3::exl3::Mat& original,
          const float* xf,int dim,int nr,float* y,int vocab,cudaStream_t stream,Gemm gemm) {
    if(active) {
        if(active->fmt==head_lowp::kFp8) {
            strata::mm3::exl3::Mat m;
            m.fmt=strata::mm3::native_dense::kFmtMXFP8;m.raw=active->w;m.native_scale=active->sc;
            m.n=active->n;m.k=active->k;
            return gemm(m);
        }
        return head_lowp::mv(*active,xf,dim,nr,y,vocab,stream);
    }
    return gemm(original);
}
} // namespace mm3::head_dispatch
