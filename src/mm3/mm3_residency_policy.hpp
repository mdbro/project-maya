#pragma once

namespace mm3 {
// Costs are provisional per-record estimates in milliseconds. A zero service
// estimate selects the runtime CPU-lane estimate, with a conservative fallback.
struct PrimaryResidencyPolicy {
    bool legacy = false;
    int horizon = 20, max_exchanges = 8;
    double service_saved_ms = 0;
    double h2d_ms = 2, d2h_ms = 2, interference_ms = 0, margin_ms = 0;
};
}
