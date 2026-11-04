#pragma once

namespace mm3 {
// Costs are per-record service proxies in milliseconds. Zero selects measured
// grouped service/transfer estimates. Horizon and admission capacity use nominal
// five-generation-row units, independent of actual verification width.
struct PrimaryResidencyPolicy {
    bool legacy = false;
    int horizon = 20, max_exchanges = 8;
    double service_saved_ms = 0;
    double h2d_ms = 0, d2h_ms = 0, interference_ms = 0, margin_ms = 0;
};
}
