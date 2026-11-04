#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <vector>

namespace mm3 {
// Candidates are already distinct and eligible, in first-seen order.
// Only reorder that list; routing, slot eligibility and the stage count stay with the caller.
inline void rank_stage_reuse(std::vector<int>& candidates, const int* routes, size_t count) {
    std::array<int, 128> uses{};
    for (size_t i = 0; i < count; ++i)
        if (routes[i] >= 0 && routes[i] < int(uses.size())) ++uses[size_t(routes[i])];
    std::stable_sort(candidates.begin(), candidates.end(), [&](int a, int b) { return uses[size_t(a)] > uses[size_t(b)]; });
}
}
