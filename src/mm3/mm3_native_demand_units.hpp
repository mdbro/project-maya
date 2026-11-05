#pragma once
#include <algorithm>
#include <cmath>

namespace mm3 {
// A five-row target evaluation has the inherited 0.95 decay and total
// five-unit credit. Scores converge to 100 times selections per evaluated row.
inline float native_demand_decay(long long rows) {
    return std::exp(std::log(0.95) * double(rows) / 5.0);
}
inline float native_demand_credit(int rows) {
    return rows > 0 ? float(-100.0 * std::expm1(std::log(0.95) * double(rows) / 5.0) / rows) : 0.0f;
}
inline float update_native_demand(float value, long long last_row,
                                  long long row, float credit) {
    return value*native_demand_decay(row-last_row)+credit;
}
// Refill opportunities accrue per evaluated generation row, with a bounded
// burst. At the nominal five-row width, capacity credits arrive per call.
struct NativeAdmissionBudget {
    double credit=0;
    long long last_row=0;
    void advance(long long row,double capacity) {
        credit=std::min(capacity,std::max(0.0,credit)+double(std::max(0LL,row-last_row))*capacity/5.0);
        last_row=row;
    }
    bool available(bool reserve_one=false) const {return credit+1e-9 >= (reserve_one?2.0:1.0);}
    void spend() {credit=std::max(0.0,credit-1.0);}
};
}
