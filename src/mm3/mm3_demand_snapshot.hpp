#pragma once
#include <cstdint>
#include <vector>

namespace mm3 {
// Cumulative original-ID device counters are independent of live score reset.
// Consume a joined snapshot exactly once per device and preserve its forward tick.
class DemandSnapshot {
public:
    template<class Credit>
    bool consume(const uint32_t* current, size_t layers, size_t experts, long long at, Credit credit) {
        const size_t n=layers*experts;
        if (!current || !experts || (!seen_.empty() && seen_.size()!=n)) return false;
        if (seen_.empty()) seen_.assign(n,0);
        for (size_t i=0;i<n;++i) {
            const uint32_t delta=current[i]-seen_[i];
            if (delta) credit(i/experts,i%experts,delta,at);
            seen_[i]=current[i];
        }
        return true;
    }
private:
    std::vector<uint32_t> seen_;
};
}

namespace mm3 {
// If a forward exits before its original-ID snapshots are joined and consumed,
// do not attribute those cumulative counts to a later forward tick.
struct DemandBoundary {
    bool& poisoned;
    bool& armed;
    bool complete = false;
    ~DemandBoundary() { if (armed && !complete) poisoned=true; }
};
}
