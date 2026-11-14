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
//
// merge_pass (2026-11-14): the whole pass's merge, and the user's decision of that day - a non-finite helper column is
// recomputed on another lane and logged, not a poisoned engine. With a refused list the pass's flagged columns are left
// out of the output (their rows rebuilt without them) for the caller to recompute; without one it stops at the first
// flagged column, the reaction above. Without a flagged column both give today's output bit for bit.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

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

// One result column of a pass: its token's output row, its weight, its BF16 values.
struct PassColumn { int token = 0; float weight = 0.0f; const uint16_t* src = nullptr; };
// A column the merge flagged: its index in the pass and its flags (0: none).
struct Flagged { size_t index = 0; unsigned flags = 0; };

// out [tokens][n] zero-filled, then every column added in pass order (merge_column), as NativeHelper::collect did.
//   refused null: the first flagged column ends the merge and is returned (the caller poisons the helper; out invalid).
//   refused given: every flagged column is listed there and left out, and nothing is returned. A row that had one is
//     rebuilt from its token's other columns in the same order, and rebuilt again while one of those flags in turn: a
//     column flagged only because the row already held an earlier column's Inf or NaN merges again cleanly and stays, one
//     still flagged (its own non-finite value, or a sum that overflows without the earlier ones) is refused too. A row
//     holds at most four columns (submit's limit), so this ends. Rows without a flagged column are untouched.
inline Flagged merge_pass(float* out, int tokens, size_t n, const PassColumn* cols, size_t count, std::vector<Flagged>* refused) {
    std::fill_n(out, size_t(tokens) * n, 0.0f);
    if (refused) refused->clear();
    std::vector<uint8_t> dirty;   // (the error path only: the rows that had a flagged column)
    for (size_t i = 0; i < count; ++i) {
        const unsigned flags = merge_column(out + size_t(cols[i].token) * n, cols[i].src, cols[i].weight, n);
        if (!flags) continue;
        if (!refused) return {i, flags};
        if (dirty.empty()) dirty.assign(size_t(tokens), 0);
        dirty[size_t(cols[i].token)] = 1;
    }
    if (dirty.empty()) return {};
    // Each dirty row again, from zero, without its refused columns. The same order and prefix as above, so its first
    // flagged column flags again here: a dirty row always refuses at least one.
    std::vector<uint8_t> skip(count, 0);
    for (int t = 0; t < tokens; ++t) {
        if (!dirty[size_t(t)]) continue;
        float* row = out + size_t(t) * n;
        for (bool again = true; again;) {
            again = false;
            std::fill_n(row, n, 0.0f);
            for (size_t i = 0; i < count; ++i) {
                if (cols[i].token != t || skip[i]) continue;
                const unsigned flags = merge_column(row, cols[i].src, cols[i].weight, n);
                if (flags) { skip[i] = 1; refused->push_back({i, flags}); again = true; break; }
            }
        }
    }
    std::sort(refused->begin(), refused->end(), [](const Flagged& x, const Flagged& y) { return x.index < y.index; });
    return {};
}

}  // namespace mm3::helper_merge
