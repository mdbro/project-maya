#pragma once
#include "ops/common/canonical_math.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Maya's native input layout; no donor API or expert record changes.
namespace mm3::fp4_rows {
constexpr int kHidden = 6144, kRowBytes = kHidden/2 + kHidden/16;
__host__ __device__ constexpr int position(int b) { return (b & ~3) | ((b & 1) << 1) | ((b & 2) >> 1); }

// Every 16 consecutive lanes encode one complete block. All lanes in the warp
// participate, including when a launch's final block has no stored result.
__device__ __forceinline__ void pack_block(float value, std::uint8_t* row, int block, bool store) {
    namespace c = infernix::ops::canon;
    // Preserve setup's conversion before the canonical NVFP4 quantizer.
    const auto h = __bfloat16_as_ushort(__float2bfloat16(value));
    const float f = c::bf16_to_f32(h);
    float amax = c::f32_from_bits(c::f32_bits(f) & 0x7fffffffu);
    if (!(amax > 0.f)) amax = 0.f; // canonical maximum ignores NaN
#pragma unroll
    for (int m = 8; m; m >>= 1) {
        const float other = __shfl_xor_sync(0xffffffffu, amax, m, 16);
        amax = other > amax ? other : amax;
    }
    const unsigned scale = c::a4_scale_word(amax, 1.f);
    const unsigned code = c::e4m3_scaled(scale) == 0 ? 0 : c::a4_code(f, scale, 1.f);
    unsigned lo = 0, hi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned a = __shfl_sync(0xffffffffu, code, j, 16);
        const unsigned b = __shfl_sync(0xffffffffu, code, 4+j, 16);
        const unsigned d = __shfl_sync(0xffffffffu, code, 8+j, 16);
        const unsigned e = __shfl_sync(0xffffffffu, code, 12+j, 16);
        lo |= (a | (b << 4)) << (8*j); hi |= (d | (e << 4)) << (8*j);
    }
    if (store && (threadIdx.x & 15) == 0) {
        const int p = position(block);
        *reinterpret_cast<uint2*>(row+8*p) = make_uint2(lo, hi);
        row[kHidden/2+p] = scale;
    }
}
} // namespace mm3::fp4_rows
