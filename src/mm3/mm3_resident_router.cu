// Adapted from Strata src/kernels/cuda/native_router.cu (blob 6611bce), whose
// warp selection derives from llama.cpp topk-moe.cu/common.cuh at 3cf03257.
// FOMOE src/car.c (blob 8ec84fc) informed the score-ratio, ready-tier and
// no-duplicate rules; this uses MiniMax sigmoid values rather than its scores.
// Strata route_prior.cu (blob 79a7581) informed the separately selectable
// ready-set score bonus; it is not stacked with bounded substitution.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
#include "mm3_resident_router.cuh"
#include <cmath>
#include <cstddef>

namespace mm3::resident {
namespace {
__device__ bool has(const uint32_t* bits, int id) { return (bits[id >> 5] >> (id & 31)) & 1u; }
__device__ bool is_ready(const Ready* ready, int id) {
    return ready && (has(ready->primary, id) || has(ready->helper, id));
}
__device__ bool better(float a, int ai, float b, int bi) { return a > b || (a == b && ai < bi); }

__global__ void route_kernel(const float* logits, const float* bias, Policy policy, float parameter,
                             const Ready* ready, int* ids, float* weights, int* original, Stats* stats) {
    const int t = blockIdx.x, lane = threadIdx.x;
    float prob[4], score[4];
    bool invalid = false;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int e = lane + 32 * i;
        prob[i] = 1.0f / (1.0f + expf(-logits[(size_t)t * 128 + e]));
        score[i] = prob[i] + bias[e];
        invalid |= !isfinite(logits[(size_t)t * 128 + e]) || !isfinite(score[i]);
    }
    if (__any_sync(0xffffffffu, invalid)) {
        if (lane < 4) {
            ids[(size_t)t*4+lane] = -1;
            if (original) original[(size_t)t*4+lane] = -1;
            weights[(size_t)t*4+lane] = NAN;
        }
        return;
    }
    int ranked[8]; float ranked_prob[8];
    for (int rank = 0; rank < 8; ++rank) {
        float best = -INFINITY, bp = 0.0f; int bi = 0x7fffffff;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int e = lane + 32 * i;
            if (better(score[i], e, best, bi)) { best = score[i]; bi = e; bp = prob[i]; }
        }
        #pragma unroll
        for (int step = 16; step; step >>= 1) {
            float v = __shfl_xor_sync(0xffffffffu, best, step);
            int id = __shfl_xor_sync(0xffffffffu, bi, step);
            float p = __shfl_xor_sync(0xffffffffu, bp, step);
            if (better(v, id, best, bi)) { best = v; bi = id; bp = p; }
        }
        ranked[rank] = bi; ranked_prob[rank] = bp;
        if ((bi & 31) == lane) score[bi >> 5] = -INFINITY;
    }
    int selected[4] = {ranked[0], ranked[1], ranked[2], ranked[3]};
    float selected_prob[4] = {ranked_prob[0], ranked_prob[1], ranked_prob[2], ranked_prob[3]};
    if (policy == Policy::Substitute) {
        for (int r = 2; r < 4; ++r) {
            if (is_ready(ready, selected[r]) || ranked_prob[r] <= 0.0f) continue;
            for (int c = 4; c < 8; ++c) {
                bool used = false;
                #pragma unroll
                for (int j = 0; j < 4; ++j) used |= selected[j] == ranked[c];
                if (!used && is_ready(ready, ranked[c]) && ranked_prob[c] >= parameter * ranked_prob[r]) {
                    selected[r] = ranked[c]; selected_prob[r] = ranked_prob[c]; break;
                }
            }
        }
    }
    // A separate prior selects from all 128 experts. Restore the original
    // selection scores after the first top-eight pass; original/probabilities
    // remain untouched. Zero bonus bypasses the pass to preserve Off exactly.
    if (policy == Policy::Prior && parameter > 0.0f) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int e=lane+32*i;
            score[i]=prob[i]+bias[e]+(is_ready(ready,e)?parameter:0.0f);
        }
        for (int rank=0;rank<4;++rank) {
            float best=-INFINITY,bp=0.0f;int bi=0x7fffffff;
            #pragma unroll
            for (int i=0;i<4;++i) {
                int e=lane+32*i;
                if(better(score[i],e,best,bi)){best=score[i];bi=e;bp=prob[i];}
            }
            #pragma unroll
            for(int step=16;step;step>>=1) {
                float v=__shfl_xor_sync(0xffffffffu,best,step);
                int id=__shfl_xor_sync(0xffffffffu,bi,step);
                float p=__shfl_xor_sync(0xffffffffu,bp,step);
                if(better(v,id,best,bi)){best=v;bi=id;bp=p;}
            }
            selected[rank]=bi;selected_prob[rank]=bp;
            if((bi&31)==lane)score[bi>>5]=-INFINITY;
        }
    }
    float sum = selected_prob[0] + selected_prob[1] + selected_prob[2] + selected_prob[3];
    if (lane < 4) {
        ids[(size_t)t * 4 + lane] = selected[lane];
        if (original) original[(size_t)t * 4 + lane] = ranked[lane];
        // All-zero underflow is undefined in the inherited router; keep outputs
        // finite here. Ordinary finite positive sums match its arithmetic.
        weights[(size_t)t * 4 + lane] = sum > 0.0f ? selected_prob[lane] / sum : 0.25f;
    }
    if (lane == 0 && stats) {
        unsigned long long primary = 0, helper = 0, missed = 0, replacements = 0;
        for (int r = 0; r < 4; ++r) {
            if (!is_ready(ready, ranked[r])) ++missed;
            bool was_original=false;
            for(int j=0;j<4;++j)was_original |= selected[r]==ranked[j];
            if(!was_original)++replacements;
            if (ready && has(ready->primary, selected[r])) ++primary;
            else if (ready && has(ready->helper, selected[r])) ++helper;
        }
        atomicAdd(&stats->replacements, replacements);
        atomicAdd(&stats->primary_hits, primary);
        atomicAdd(&stats->helper_hits, helper);
        atomicAdd(&stats->original_misses, missed);
    }
}
}

cudaError_t route(const float* logits, const float* bias, int T, Policy policy, float parameter,
                  const Ready* ready, int* ids, float* weights, int* original, Stats* stats, cudaStream_t stream) {
    if (!logits || !bias || !ids || !weights || T < 0 ||
        (policy != Policy::Off && policy != Policy::Substitute && policy != Policy::Prior) ||
        !std::isfinite(parameter) || parameter < 0.0f || parameter > 1.0f ||
        ((policy == Policy::Substitute || policy == Policy::Prior) && !ready)) return cudaErrorInvalidValue;
    if (T == 0) return cudaSuccess;
    route_kernel<<<T, 32, 0, stream>>>(logits, bias, policy, parameter, ready, ids, weights, original, stats);
    return cudaGetLastError();
}
}
