#pragma once
// The 3090 helper's result merge: one result column into the host's output row, dst[k] += w * bf16(src[k]).
//
// NativeHelper::collect runs this for every helper job and column between the CPU lane's end and the publication the 5090
// waits on, so on passes the CPU lane ends last it is serial time on the pass (window-passes-20261109: 0.009 ms at one row,
// 0.093 ms at eight). The scalar form checked each element and returned at the first non-finite one, which kept it from
// vectorizing. This form has no branch in the loop: it reduces the two checks to flags over the column and leaves the
// reaction to the caller. The arithmetic per element is unchanged (one multiply, then one add, in the same order), so the
// output is bit-identical to the scalar form; tests/mm3_helper_merge_test.cpp checks that.
//
// One difference, on the error path only: the whole column is merged before the flags are read, and a column with both a
// non-finite input and a non-finite sum reports the input. The scalar form stopped at the first bad element and reported
// whichever came first. Either way the helper is poisoned and the pass fails.

#include <cstddef>
#include <cstdint>
#include <cstring>

namespace mm3::helper_merge {

enum : unsigned { kNonfiniteInput = 1u, kNonfiniteSum = 2u };

inline unsigned merge_column(float* __restrict dst, const uint16_t* __restrict src, float weight, size_t n) {
    uint32_t bad_input = 0, bad_sum = 0;
    for (size_t k = 0; k < n; ++k) {
        const uint32_t bits = uint32_t(src[k]) << 16;
        float value;
        std::memcpy(&value, &bits, sizeof value);
        const float sum = dst[k] + weight * value;
        uint32_t sum_bits;
        std::memcpy(&sum_bits, &sum, sizeof sum_bits);
        bad_input |= uint32_t((src[k] & 0x7f80u) == 0x7f80u);
        bad_sum |= uint32_t((sum_bits & 0x7f800000u) == 0x7f800000u);
        dst[k] = sum;
    }
    return (bad_input ? kNonfiniteInput : 0u) | (bad_sum ? kNonfiniteSum : 0u);
}

}  // namespace mm3::helper_merge
