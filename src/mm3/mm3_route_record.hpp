// src/mm3/mm3_route_record.hpp - the routing lab's per-row routing diagnostics, handed from the routing path to the
// token records (mm3_token_record.hpp; routing-lab-20261110/API.md 1.5).  The routing path fills forward() while a target
// forward runs; mm3_main.cpp reads it after the forward returns and copies row r's text verbatim into the record of the
// token that row produced, so the route object's fields belong to the routing path alone.  Single-request serving only
// (one forward at a time), as the capture's state is.
#pragma once

#include <string>
#include <vector>

namespace mm3::route_record {

struct Forward {
    unsigned long long serial = 0;  // the routing path bumps it when it starts filling a forward
    int rows = 0;                   // the forward's rows (n)
    std::vector<std::string> row;   // row r's "route" object as JSON text "{...}", "" for none: either all n rows, or
                                    // only the rows that return logits (the last `want` of the call)
};

struct Request {
    std::string settings;  // "{...}": the effective lab settings, for done.settings.lab; "" none
    std::string summary;   // "{...}": the request's sums and means, for done.route; "" none
};

inline Forward& forward() {
    static Forward f;
    return f;
}

inline Request& request() {
    static Request r;
    return r;
}

/// The route text for logit row i of a forward of n rows and `want` logit rows, when the routing path filled this
/// forward (its serial moved past `before`); "" otherwise.
inline const std::string& row_text(unsigned long long before, int n, int want, int i) {
    static const std::string none;
    const Forward& f = forward();
    if (f.serial == before || i < 0 || i >= want || want > n) return none;
    if ((int) f.row.size() == want) return f.row[(size_t) i];
    if ((int) f.row.size() == n) return f.row[(size_t) (n - want + i)];
    return none;
}

}  // namespace mm3::route_record
