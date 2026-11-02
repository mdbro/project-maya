#pragma once
#include <algorithm>
// Adapted from Strata src/minimax/stage_accept.hpp, retained stage-window path.
namespace mm3 {
inline int reusable_stage_frontier(int state_size, int position, int completed_rows) {
    return std::min(state_size, position + completed_rows);
}
template<class Evaluate, class Cancelled>
bool accept_stage_a(int rows, int drafts, Evaluate&& evaluate, Cancelled&& cancelled) {
    for (int i = 0; i < std::min(rows, drafts); ++i)
        if (cancelled() || !evaluate(i)) return false;
    return !cancelled();
}
}
