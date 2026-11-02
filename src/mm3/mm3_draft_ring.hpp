#pragma once

#include <algorithm>
#include <vector>

namespace mm3 {
// Logical identity of target-context rows in the physical DSpark KV ring.
// Proposal rows occupy the same slots, but are not target context.
class DraftRing {
public:
    void reset(int ring, int window) { slots_.assign(ring, -1); window_ = window; }
    void context(int pos, int count) {
        for (int i = 0; i < count; ++i) slots_[(pos + i) % slots_.size()] = pos + i;
    }
    void proposal(int pos, int count) {
        for (int i = 0; i < count; ++i) slots_[(pos + i) % slots_.size()] = -1;
    }
    bool available(int pos) const {
        if (pos < 0 || slots_.empty()) return false;
        // Include the entire preceding window, conservatively one row more than
        // the first draft query actually reads.
        for (int i = std::max(0, pos - window_); i < pos; ++i)
            if (slots_[i % slots_.size()] != i) return false;
        return true;
    }
private:
    std::vector<int> slots_;
    int window_ = 0;
};
}
