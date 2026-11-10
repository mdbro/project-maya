// Item 3 of the order of 2026-11-10: the 3090's exchange capacity and the score decay as runtime settings. Host-only
// checks of the shared formulas the engine now calls with those settings (mm3_native_demand_units.hpp).
#include "mm3_native_demand_units.hpp"
#include <cassert>
#include <cmath>
#include <cstdio>

// Admissions a row at one generated row a call. per_call: how many a call may admit (the 3090 admits one lease and keeps
// one in flight, the 5090 has two transactions).
static double rate(double capacity, long long rows, int per_call) {
    mm3::NativeAdmissionBudget b;
    long long done = 0;
    for (long long row = 1; row <= rows; ++row) {
        b.advance(row, capacity);
        for (int i = 0; i < per_call && b.available(); ++i) { b.spend(); ++done; }
    }
    return double(done) / double(rows);
}

int main() {
    // The 3090's capacity: 1 is today's one exchange in five rows; 2.5 one in two; 5 one a row.
    assert(std::fabs(rate(1.0, 10000, 1) - 0.2) < 1e-3);
    assert(std::fabs(rate(2.5, 10000, 1) - 0.5) < 1e-3);
    assert(std::fabs(rate(5.0, 10000, 1) - 1.0) < 1e-3);
    // 10 (two a row) is held to one a row by one admission a forward: the limit step 5 must account for.
    assert(std::fabs(rate(10.0, 10000, 1) - 1.0) < 1e-9);
    // The 5090's formula is unchanged: capacity 8 is 1.6 a row with its two transactions.
    assert(std::fabs(rate(8.0, 10000, 2) - 1.6) < 1e-3);
    // The burst cap equals the capacity, as before.
    mm3::NativeAdmissionBudget b; b.advance(1000, 2.5); assert(std::fabs(b.credit - 2.5) < 1e-12);

    // Today's decay is the default argument: passing 0.95 explicitly is bit-identical.
    for (int rows : {1, 2, 5, 7, 16}) {
        assert(mm3::native_demand_decay(rows) == mm3::native_demand_decay(rows, 0.95));
        assert(mm3::native_demand_credit(rows) == mm3::native_demand_credit(rows, 0.95));
        assert(mm3::update_native_demand(3.25f, 10, 10 + rows, 0.5f) == mm3::update_native_demand(3.25f, 10, 10 + rows, 0.5f, 0.95));
    }
    // Any decay keeps the score's scale (one selection a row converges to 100), and forgets at its own rate.
    for (double decay : {0.5, 0.8, 0.9, 0.95, 0.98, 0.999}) {
        float score = 0;
        long long last = 0;
        for (long long row = 1; row <= 200000; ++row) {
            score = mm3::update_native_demand(score, last, row, mm3::native_demand_credit(1, decay), decay);
            last = row;
        }
        assert(std::fabs(score - 100.0f) < 0.1f);
        const float later = mm3::update_native_demand(score, last, last + 50, 0.0f, decay);   // 50 idle rows
        assert(std::fabs(later - score * float(std::pow(decay, 10.0))) < 1e-3f * score);
    }
    std::printf("mm3_exchange_settings_test OK\n");
    return 0;
}
