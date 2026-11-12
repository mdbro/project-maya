// The helper's input check (NativeHelper::submit): a BF16 row holds no Inf or NaN, i.e. no element whose exponent
// field is all ones. One pass over the row with no early exit, so the compiler vectorizes it (16-bit compares); the
// same answer as the scalar loop `for k: if ((row[k] & 0x7f80) == 0x7f80) refuse`, which took 1.2 us a column and
// ran once for every column that names the token (review 2026-11-11, R2: runs/device-kernels-20261111/review/).
#pragma once
#include <cstddef>
#include <cstdint>

namespace mm3::helper_finite {
inline bool row_finite(const std::uint16_t* row, std::size_t n) {
    unsigned bad = 0;
    for (std::size_t k = 0; k < n; ++k) bad |= unsigned((row[k] & 0x7f80u) == 0x7f80u);
    return bad == 0;
}
}  // namespace mm3::helper_finite
