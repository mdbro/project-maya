#pragma once
// A helper card's non-finite result column, recomputed on the CPU lane (the user's decision of 2026-11-14: "It should
// recompute that column on another lane and carry on, make sure it logs if it happens"). NativeHelper::collect leaves the
// column out and lists it (NativeHelper::Refused); the model recomputes it from an independent copy of the expert's record
// (mm3_native_helper_recover.inc). This header holds the parts that need no model: which record to read, the guarded add
// into the pass's output, the log line and STAT key, and the repair's outcome. tests/mm3_helper_nonfinite_test.cpp checks them.
//
// Why another copy of the record: the helper's arithmetic gives finite results for finite inputs and any record bytes
// (runs/phase3-local-20261113/nonfinite-helper.md, section 4), so a non-finite column means a fault in a transient buffer
// or in the computation, most likely marginal DRAM. A recompute that reads none of the buffers involved is independent
// of that fault.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace mm3::helper_recover {

enum class Source { pack, host_bank };

inline const char* source_name(Source s) {
    return s == Source::pack ? "the native pack's record (the model's files read again)" : "the HOST record (the host bank's)";
}
inline const char* source_key(Source s) { return s == Source::pack ? "pack" : "host_bank"; }

// The record sources to try, in order, each only if it exists. Never the card's slot: it may be what failed, and a 3090
// slot cannot be exported while a swap is in flight.
//   a 3090 column (an exclusive record; its HOST frame is empty): the native pack's own bytes - the model's safetensors,
//     still mapped, assembled again into the shared record layout as the load did (assemble). About 32 MB read and the
//     RG16 packing, up to about 0.2 s. Independent of the host bank and of both cards' VRAM.
//   a 4080 column (the third pool holds copies of HOST experts): the HOST record, the same bytes without the pack read;
//     the pack if the HOST record is missing.
inline std::vector<Source> sources(bool third_card, bool pack_available, bool host_record) {
    std::vector<Source> order;
    if (third_card && host_record) order.push_back(Source::host_bank);
    if (pack_available) order.push_back(Source::pack);
    return order;
}

// One event's STAT warning key (the request's warnings= list: no commas or spaces): the card's CUDA device, the layer,
// the expert, the token's position and the lane that recomputed it, e.g. helper_recompute:cuda1:layer12:expert45:position1405:cpu
inline std::string warning_key(int device, int layer, int expert, long long position) {
    char key[160];
    std::snprintf(key, sizeof key, "helper_recompute:cuda%d:layer%d:expert%d:position%lld:cpu", device, layer, expert, position);
    return key;
}

// The repair at the next forward boundary: the card's record read back and compared with the pack's, rewritten from the
// pack if it differs, then read back again.
enum class Repair {
    equal,       // the record equals the pack's: the fault was transient (logged, nothing else)
    rewritten,   // it differed, and after the rewrite it reads back equal
    drop,        // a 4080 copy that still differs: its ready bit cleared, the HOST serves the expert like any HOST expert
    poison,      // a 3090 record that still differs: a VRAM fault, and a 3090-only expert has no other owner without an
                 // ownership change - the card is poisoned, as before the decision
};
inline Repair repair_outcome(bool equal_before, bool equal_after_rewrite, bool third_card) {
    if (equal_before) return Repair::equal;
    if (equal_after_rewrite) return Repair::rewritten;
    return third_card ? Repair::drop : Repair::poison;
}

// bf16 bits -> float, as the merge reads a column
inline float bf16_value(uint16_t v) {
    const uint32_t bits = uint32_t(v) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof f);
    return f;
}

enum class Add { ok, nonfinite_column, nonfinite_sum };
// The CPU lane's recomputed column (its weight applied: w * expert(x)) added into the pass's output row. A column with
// a non-finite value is not added (the row is left as it was); a sum that is not finite is reported after the add (the
// row is then invalid). Either failure: the caller poisons the card, as before the decision.
inline Add add_column(float* row, const float* column, size_t n) {
    for (size_t k = 0; k < n; ++k)
        if (!std::isfinite(column[k])) return Add::nonfinite_column;
    bool finite = true;
    for (size_t k = 0; k < n; ++k) {
        row[k] += column[k];
        finite &= std::isfinite(row[k]);
    }
    return finite ? Add::ok : Add::nonfinite_sum;
}

// What one event's log line names.
struct Event {
    std::string card;        // "3090 helper" or "4080 third"
    int device = -1;         // its CUDA ordinal in this process
    std::string device_name; // the card's name (cudaGetDeviceProperties), empty if unknown
    std::string where;       // "decode" or "prompt read"
    int layer = 0, expert = 0, slot = -1, token = 0, rows = 0;
    long long position = -1; // the token's position in the sequence (-1: unknown)
    float weight = 0.0f;
    bool sum = false;        // a finite column whose weighted sum overflowed
    int bad = 0;             // non-finite elements of the card's column (h_y)
    std::string detail;      // NativeHelper::Refused::detail: where the value went bad
    Source source = Source::pack;
    std::string fallback;    // sources tried before this one and why each failed (empty: the first worked)
    double ms = 0.0;         // the recompute's wall time, record read included
    long count = 0;          // events so far in this engine, this one included
};

inline std::string event_line(const Event& e) {
    std::string s = "helper nonfinite recovered #" + std::to_string(e.count) + ": " + e.card + " CUDA" + std::to_string(e.device);
    if (!e.device_name.empty()) s += " (" + e.device_name + ")";
    char head[256];
    std::snprintf(head, sizeof head, ", %s, layer %d expert %d slot %d, row %d of %d", e.where.c_str(), e.layer, e.expert, e.slot,
                  e.token, e.rows);
    s += head;
    if (e.position >= 0) s += " (position " + std::to_string(e.position) + ")";
    char body[256];
    std::snprintf(body, sizeof body, ", weight %g: %s, %d of 6144 elements non-finite; recomputed on the CPU lane in %.1f ms from ",
                  double(e.weight), e.sum ? "a finite column whose weighted sum overflowed" : "a non-finite value in the card's result",
                  e.bad, e.ms);
    s += body;
    s += source_name(e.source);
    if (!e.fallback.empty()) s += " (" + e.fallback + ")";
    s += "; detail: " + e.detail;
    return s;
}

}  // namespace mm3::helper_recover
