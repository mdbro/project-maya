#pragma once

// Hardware-independent decisions shared by the engine and bounded replay.
// These are policy prices, NOT a physical execution model. In particular the
// CPU price scale is allowed to change decisions only.
#include <cmath>
#include <vector>
#include "mm3_batch_staging.hpp"
#include "mm3_native_demand_units.hpp"

namespace mm3 {
struct ResidencyCandidate {
    long long key = -1;
    float score = 0;
    int ready_slot = -1;
};
struct ResidencyChoice {
    long long incoming = -1, victim = -1;
    int ready_slot = -1;
    double value_ms = 0;
    explicit operator bool() const { return incoming >= 0 && victim >= 0; }
};

// Preserved incumbent. Keeping the allocation wins ties. Candidate order is
// significant (ready slots first, then layer/expert order), as in production.
// This formula is deliberately named incumbent: decision reproduction does
// not validate its economics, nor may physical simulation use this formula.
inline ResidencyChoice choose_incumbent_primary(
        const std::vector<ResidencyCandidate>& incoming,
        const std::vector<ResidencyCandidate>& victims,
        double benefit_per_score, double h2d_ms, double d2h_ms,
        double interference_ms, double margin_ms) {
    ResidencyChoice out;
    const ResidencyCandidate* victim = nullptr;
    for(const auto& c:victims) if(!victim || c.score<victim->score) victim=&c;
    if(!victim)return out;
    for(const auto& c:incoming) {
        const double value=(c.score-victim->score)*benefit_per_score-d2h_ms-
            (c.ready_slot>=0?0:h2d_ms)-interference_ms-margin_ms;
        if(value>out.value_ms) out={c.key,victim->key,c.ready_slot,value};
    }
    return out;
}

inline ResidencyChoice choose_incumbent_helper(
        const std::vector<ResidencyCandidate>& incoming,
        const std::vector<ResidencyCandidate>& victims) {
    const ResidencyCandidate *hot=nullptr,*cold=nullptr;
    for(const auto& c:victims)if(!cold || c.score<cold->score)cold=&c;
    for(const auto& c:incoming)if(c.score>=1.8f && (!hot || c.score>hot->score))hot=&c;
    if(!hot || !cold || !(hot->score>1.25f*cold->score+1.0f))return {};
    return {hot->key,cold->key,-1,0};
}

struct PolicySlot {
    long long owner=-1, pending=-1, used=0;
    bool issued=false, foreground=false, reserved=false;
};
inline int choose_working_slot(const std::vector<PolicySlot>& slots) {
    int best=-1;
    for(size_t i=0;i<slots.size();++i) {
        const auto& s=slots[i];
        if(!s.reserved && !s.foreground && s.pending<0 &&
           (best<0 || s.used<slots[size_t(best)].used))best=int(i);
    }
    return best;
}
}
