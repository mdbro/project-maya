#pragma once
#include <algorithm>
#include <cmath>

namespace mm3 {
// A five-row target evaluation has the inherited 0.95 decay and total
// five-unit credit. Scores converge to 100 times selections per evaluated row.
inline float native_demand_decay(long long rows,double decay=0.95) {
    return std::exp(std::log(decay) * double(rows) / 5.0);
}
inline float native_demand_credit(int rows,double decay=0.95) {
    return rows > 0 ? float(-100.0 * std::expm1(std::log(decay) * double(rows) / 5.0) / rows) : 0.0f;
}
inline float update_native_demand(float value, long long last_row,
                                  long long row, float credit,double decay=0.95) {
    return value*native_demand_decay(row-last_row,decay)+credit;
}
// Refill opportunities accrue per evaluated generation row, with a bounded
// burst. At the nominal five-row width, capacity credits arrive per call.
// The burst is at least one whole credit, the cost of an exchange: a capacity
// below 1 (helper_max_exchanges 0.5, meant as one swap per ten rows) admitted
// none before, with no warning (swap-policy.md, section 3, bug 2; the 2026-11-13
// port onto minimax-wsl/local).
struct NativeAdmissionBudget {
    double credit=0;
    long long last_row=0;
    void advance(long long row,double capacity) {
        credit=std::min(std::max(capacity,1.0),std::max(0.0,credit)+double(std::max(0LL,row-last_row))*capacity/5.0);
        last_row=row;
    }
    bool available(bool reserve_one=false) const {return credit+1e-9 >= (reserve_one?2.0:1.0);}
    void spend() {credit=std::max(0.0,credit-1.0);}
};
}
