#pragma once
// Preallocated binary records for the generation-timing diagnostic study.
//
// One writer, the engine thread. While recording there is no allocation, lock,
// string formatting, file I/O, CUDA query, event or wait: only clock reads and
// copies into a buffer touched before generation. finish() writes the file after
// generation has ended. A full buffer invalidates the capture rather than
// truncating it silently. Recording a field does not commit a predictor to it.
//
// Two levels share one layout. Full records also hold assignments, service
// locations, slot counts, state maps and residency events. Timestamps-only keeps
// the clock reads and fixed layer/forward records and skips everything else, so a
// matched comparison can separate the clock reads from the state work.
//
// tools/mm3/native_light.py is the reader; keep these layouts in step with it.
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

namespace mm3::capture::light {

constexpr uint32_t kFormatVersion = 1;
constexpr size_t kCapacityBytes = 64u * 1024u * 1024u;
constexpr int kMaxPicks = 16 * 8;

enum Kind : uint16_t {
    kMap = 1, kGenerationBegin = 2, kGenerationEnd = 3, kForward = 4, kLayer = 5,
    kDemandCopy = 6, kForegroundRetire = 7, kPrimaryIssue = 8, kPrimaryPublish = 9, kHelperEvent = 10,
};
// Where one selected expert actually executed in one row.
enum Service : uint8_t {
    kServiceHelper = 1, kServiceCpu = 2, kServicePrimaryResident = 3, kServiceStagedReady = 4, kServiceStagedNew = 5,
};
// Map byte: which canonical owners hold an expert's record (bit mask).
enum Location : uint8_t { kLocationHost = 1, kLocationPrimary = 2, kLocationHelper = 4 };
enum HelperPhase : int32_t { kHelperLease = 1, kHelperWithdrawn = 2, kHelperPublished = 3, kHelperRetired = 4 };
enum Scope : uint32_t { kScopeRequest = 0, kScopeFixture = 1 };

struct Head { uint16_t kind, bytes; uint32_t seq; uint64_t t_ns; };
static_assert(sizeof(Head) == 16);

struct MapFixed {
    Head h;
    int64_t forward, row_clock, helper_incoming, helper_victim;
    uint16_t layers, experts, slots, exchanges;
    uint32_t phase;  // 0 generation start, 1 generation end
    uint32_t helper_flags;  // bit 0 withdrawn, bit 1 published
};
static_assert(sizeof(MapFixed) == 64);
// Followed by location[layers*experts], SlotState[slots], ExchangeState[exchanges].
struct SlotState { int64_t owner, pending; uint8_t issued, foreground, reserved, pad[5]; };
static_assert(sizeof(SlotState) == 24);
struct ExchangeState { int64_t incoming, victim; int32_t landing, submitted; };
static_assert(sizeof(ExchangeState) == 24);

struct Generation {
    Head h;
    int64_t forward, row_clock, position, emitted, drafts_offered, drafts_accepted, target_rounds, spec_rounds;
    double elapsed_ms;
    uint32_t scope, ok;
};
static_assert(sizeof(Generation) == 96);

constexpr int kForwardCounters = 18;
struct Forward {
    Head h;
    int64_t forward, row_clock;
    int32_t position, rows, want;
    uint32_t generation;
    // Nanoseconds after h.t_ns.
    uint32_t prepare_end, body_end, sync_end, end;
    // Existing cumulative counters at this forward's end; see kForwardCounterNames in the reader.
    int64_t counters[kForwardCounters];
};
static_assert(sizeof(Forward) == 64 + 8 * kForwardCounters);

// Stamps are collected while the layer runs and copied in layer() at its end.
struct LayerDraft {
    uint64_t begin = 0, pre_end = 0, join_end = 0, boundary_end = 0, plan_end = 0, helper_submit_end = 0,
             primary_submit_end = 0, cpu_end = 0, collect_end = 0, end = 0;
    uint64_t helper_enqueue = 0, helper_worker_start = 0, helper_record_begin = 0, helper_record_end = 0,
             helper_last_not_ready = 0, helper_ready_observed = 0, helper_worker_published = 0,
             helper_wait_begin = 0, helper_wait_end = 0;
    uint8_t free_slots = 0, reserved_slots = 0, foreground_slots = 0, pending_promotions = 0,
            exchanges_pending = 0, helper_lease = 0, stage_max = 0, stage_selected = 0, total_slots = 0;
    uint8_t service[kMaxPicks] = {};
};

struct LayerFixed {
    Head h;
    int64_t forward;
    int16_t layer;
    uint8_t rows, k;
    uint8_t free_slots, reserved_slots, foreground_slots, pending_promotions;
    uint8_t exchanges_pending, helper_lease, stage_max, stage_selected, total_slots, pad[3];
    // Nanoseconds after h.t_ns (window_pre begin); 0 means the interval did not occur.
    uint32_t pre_end, join_end, boundary_end, plan_end, helper_submit_end, primary_submit_end, cpu_end, collect_end, end;
    uint32_t helper_enqueue, helper_worker_start, helper_record_begin, helper_record_end, helper_last_not_ready,
             helper_ready_observed, helper_worker_published, helper_wait_begin, helper_wait_end;
};
static_assert(sizeof(LayerFixed) == 112);
// Followed by original[rows*k], executed[rows*k], service[rows*k].

struct DemandCopy {
    Head h;
    int64_t forward, displaced;
    int32_t layer, expert, slot;
    uint32_t bytes, submit_end;  // submit_end: ns after h.t_ns
    uint32_t pad;
};
static_assert(sizeof(DemandCopy) == 56);
struct ForegroundRetire { Head h; int64_t forward, owner; int32_t slot, pad; };
static_assert(sizeof(ForegroundRetire) == 40);
struct PrimaryIssue {
    Head h;
    int64_t forward, incoming, victim;
    uint64_t h2d_bytes, d2h_bytes;
    int32_t landing, early_layer;
    uint32_t reuse, submit_end;
};
static_assert(sizeof(PrimaryIssue) == 72);
struct PrimaryPublish { Head h; int64_t forward, incoming, victim; int32_t landing, protected_layer; };
static_assert(sizeof(PrimaryPublish) == 48);
struct HelperEvent { Head h; int64_t forward, incoming, victim; int32_t phase; uint32_t submit_end; };
static_assert(sizeof(HelperEvent) == 48);

inline bool g_active = false;
inline bool g_full = true;  // false: timestamps only
inline bool active() { return g_active; }
inline bool full() { return g_active && g_full; }
inline uint64_t clock_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
// A stamp only while recording: the off path is a branch, not a clock read.
inline uint64_t stamp() { return g_active ? clock_ns() : 0; }

struct Recorder {
    std::vector<uint8_t> buffer;
    size_t used = 0;
    uint64_t attempted = 0, dropped = 0;
    uint32_t seq = 0;
    int64_t forward = -1;
    std::string experiment, run, request, directory;
    int64_t wall_unix_ns = 0;
};
inline Recorder& recorder() { static Recorder r; return r; }
inline void set_forward(int64_t forward) { if (g_active) recorder().forward = forward; }

// Returns aligned storage for `bytes`, or nullptr after counting the loss.
inline uint8_t* allocate(uint16_t kind, size_t bytes, uint64_t t_ns) {
    auto& r = recorder();
    ++r.attempted;
    const size_t stride = (bytes + 7u) & ~size_t(7);
    if (stride > std::numeric_limits<uint16_t>::max() || stride > r.buffer.size() - r.used) { ++r.dropped; return nullptr; }
    uint8_t* at = r.buffer.data() + r.used;
    r.used += stride;
    auto* head = reinterpret_cast<Head*>(at);
    head->kind = kind; head->bytes = uint16_t(stride); head->seq = ++r.seq; head->t_ns = t_ns;
    return at;
}
inline uint32_t after(uint64_t base, uint64_t t) {
    if (!t || t < base) return 0;
    const uint64_t delta = t - base;
    return delta > std::numeric_limits<uint32_t>::max() ? std::numeric_limits<uint32_t>::max() : uint32_t(delta);
}

template <class Int>
inline void layer(const LayerDraft& d, int layer_index, int rows, int k, const Int* original, const Int* executed) {
    if (!g_active) return;
    const int picks = g_full ? rows * k : 0;
    if (picks < 0 || picks > kMaxPicks) { auto& r = recorder(); ++r.attempted; ++r.dropped; return; }
    auto* at = allocate(kLayer, sizeof(LayerFixed) + size_t(picks) * 3, d.begin);
    if (!at) return;
    auto* f = reinterpret_cast<LayerFixed*>(at);
    f->forward = recorder().forward;
    f->layer = int16_t(layer_index); f->rows = uint8_t(rows); f->k = uint8_t(k);
    f->free_slots = d.free_slots; f->reserved_slots = d.reserved_slots; f->foreground_slots = d.foreground_slots;
    f->pending_promotions = d.pending_promotions; f->exchanges_pending = d.exchanges_pending;
    f->helper_lease = d.helper_lease; f->stage_max = d.stage_max; f->stage_selected = d.stage_selected;
    f->total_slots = d.total_slots; f->pad[0] = f->pad[1] = f->pad[2] = 0;
    const uint64_t b = d.begin;
    f->pre_end = after(b, d.pre_end); f->join_end = after(b, d.join_end); f->boundary_end = after(b, d.boundary_end);
    f->plan_end = after(b, d.plan_end); f->helper_submit_end = after(b, d.helper_submit_end);
    f->primary_submit_end = after(b, d.primary_submit_end); f->cpu_end = after(b, d.cpu_end);
    f->collect_end = after(b, d.collect_end); f->end = after(b, d.end);
    f->helper_enqueue = after(b, d.helper_enqueue); f->helper_worker_start = after(b, d.helper_worker_start);
    f->helper_record_begin = after(b, d.helper_record_begin); f->helper_record_end = after(b, d.helper_record_end);
    f->helper_last_not_ready = after(b, d.helper_last_not_ready); f->helper_ready_observed = after(b, d.helper_ready_observed);
    f->helper_worker_published = after(b, d.helper_worker_published); f->helper_wait_begin = after(b, d.helper_wait_begin);
    f->helper_wait_end = after(b, d.helper_wait_end);
    uint8_t* tail = at + sizeof(LayerFixed);
    for (int i = 0; i < picks; ++i) *tail++ = uint8_t(original[i]);
    for (int i = 0; i < picks; ++i) *tail++ = uint8_t(executed[i]);
    std::memcpy(tail, d.service, size_t(picks));
}

inline void forward_record(uint64_t begin, uint64_t prepare_end, uint64_t body_end, uint64_t sync_end,
                           int64_t row_clock, int position, int rows, int want, bool generation,
                           const int64_t (&counters)[kForwardCounters]) {
    if (!g_active) return;
    const uint64_t end = clock_ns();
    auto* at = allocate(kForward, sizeof(Forward), begin);
    if (!at) return;
    auto* f = reinterpret_cast<Forward*>(at);
    f->forward = recorder().forward; f->row_clock = row_clock;
    f->position = position; f->rows = rows; f->want = want; f->generation = generation ? 1 : 0;
    f->prepare_end = after(begin, prepare_end); f->body_end = after(begin, body_end);
    f->sync_end = after(begin, sync_end); f->end = after(begin, end);
    if (g_full) std::memcpy(f->counters, counters, sizeof(counters));
    else std::memset(f->counters, 0, sizeof(counters));
}

inline void demand_copy(int layer_index, int expert, int slot, int64_t displaced, size_t bytes, uint64_t begin) {
    if (!full()) return;
    const uint64_t end = clock_ns();
    auto* at = allocate(kDemandCopy, sizeof(DemandCopy), begin);
    if (!at) return;
    auto* f = reinterpret_cast<DemandCopy*>(at);
    f->forward = recorder().forward; f->displaced = displaced;
    f->layer = layer_index; f->expert = expert; f->slot = slot;
    f->bytes = uint32_t(bytes); f->submit_end = after(begin, end); f->pad = 0;
}
inline void foreground_retire(int slot, int64_t owner) {
    if (!full()) return;
    auto* at = allocate(kForegroundRetire, sizeof(ForegroundRetire), clock_ns());
    if (!at) return;
    auto* f = reinterpret_cast<ForegroundRetire*>(at);
    f->forward = recorder().forward; f->owner = owner; f->slot = slot; f->pad = 0;
}
inline void primary_issue(int64_t incoming, int64_t victim, int landing, int early_layer, bool reuse,
                          uint64_t h2d_bytes, uint64_t d2h_bytes, uint64_t begin) {
    if (!full()) return;
    const uint64_t end = clock_ns();
    auto* at = allocate(kPrimaryIssue, sizeof(PrimaryIssue), begin);
    if (!at) return;
    auto* f = reinterpret_cast<PrimaryIssue*>(at);
    f->forward = recorder().forward; f->incoming = incoming; f->victim = victim;
    f->h2d_bytes = h2d_bytes; f->d2h_bytes = d2h_bytes; f->landing = landing; f->early_layer = early_layer;
    f->reuse = reuse ? 1 : 0; f->submit_end = after(begin, end);
}
inline void primary_publish(int64_t incoming, int64_t victim, int landing, int protected_layer) {
    if (!full()) return;
    auto* at = allocate(kPrimaryPublish, sizeof(PrimaryPublish), clock_ns());
    if (!at) return;
    auto* f = reinterpret_cast<PrimaryPublish*>(at);
    f->forward = recorder().forward; f->incoming = incoming; f->victim = victim;
    f->landing = landing; f->protected_layer = protected_layer;
}
inline void helper_event(int phase, int64_t incoming, int64_t victim, uint64_t begin = 0) {
    if (!full()) return;
    const uint64_t end = clock_ns();
    const uint64_t t = begin ? begin : end;
    auto* at = allocate(kHelperEvent, sizeof(HelperEvent), t);
    if (!at) return;
    auto* f = reinterpret_cast<HelperEvent*>(at);
    f->forward = recorder().forward; f->incoming = incoming; f->victim = victim;
    f->phase = phase; f->submit_end = after(t, end);
}
inline void generation_mark(uint16_t kind, uint32_t scope, bool ok, int64_t row_clock, int64_t position, int64_t emitted,
                            int64_t offered, int64_t accepted, int64_t target_rounds, int64_t spec_rounds, double elapsed_ms) {
    if (!g_active) return;
    auto* at = allocate(kind, sizeof(Generation), clock_ns());
    if (!at) return;
    auto* f = reinterpret_cast<Generation*>(at);
    f->forward = recorder().forward; f->row_clock = row_clock; f->position = position; f->emitted = emitted;
    f->drafts_offered = offered; f->drafts_accepted = accepted; f->target_rounds = target_rounds;
    f->spec_rounds = spec_rounds; f->elapsed_ms = elapsed_ms; f->scope = scope; f->ok = ok ? 1 : 0;
}
inline void map(uint32_t phase, int64_t row_clock, int layers, int experts, const std::vector<uint8_t>& location,
                const std::vector<SlotState>& slots, const std::vector<ExchangeState>& exchanges,
                int64_t helper_incoming, int64_t helper_victim, uint32_t helper_flags) {
    if (!full()) return;
    const size_t bytes = sizeof(MapFixed) + location.size() + slots.size() * sizeof(SlotState) +
                         exchanges.size() * sizeof(ExchangeState) + 8;
    auto* at = allocate(kMap, bytes, clock_ns());
    if (!at) return;
    auto* f = reinterpret_cast<MapFixed*>(at);
    f->forward = recorder().forward; f->row_clock = row_clock;
    f->helper_incoming = helper_incoming; f->helper_victim = helper_victim;
    f->layers = uint16_t(layers); f->experts = uint16_t(experts);
    f->slots = uint16_t(slots.size()); f->exchanges = uint16_t(exchanges.size());
    f->phase = phase; f->helper_flags = helper_flags;
    uint8_t* tail = at + sizeof(MapFixed);
    std::memcpy(tail, location.data(), location.size()); tail += location.size();
    // Entries are copied bytewise: the location array leaves them unaligned.
    if (!slots.empty()) std::memcpy(tail, slots.data(), slots.size() * sizeof(SlotState));
    tail += slots.size() * sizeof(SlotState);
    if (!exchanges.empty()) std::memcpy(tail, exchanges.data(), exchanges.size() * sizeof(ExchangeState));
}

inline void reset() {
    auto& r = recorder();
    g_active = false; g_full = true; r.used = 0; r.attempted = 0; r.dropped = 0; r.seq = 0; r.forward = -1;
}
inline std::string artifact_path(const Recorder& r) { return r.directory + "/" + r.request + ".light"; }
inline std::string marker_path(const Recorder& r) { return r.directory + "/" + r.request + ".partial"; }

// Called between requests. The buffer is allocated once and every page touched,
// so recording cannot fault in fresh pages during generation.
inline bool begin(const std::string& experiment, const std::string& run, const std::string& request,
                  const std::string& directory, std::string& error, bool full_records = true) {
    reset();
    auto& r = recorder();
    r.experiment = experiment; r.run = run; r.request = request; r.directory = directory;
    if (std::ifstream(artifact_path(r)).good()) { error = "capture request artifact already exists"; return false; }
    if (std::ifstream(marker_path(r)).good()) { error = "capture request partial marker already exists"; return false; }
    std::ofstream pending(marker_path(r), std::ios::out);
    pending << "{\"schema\":1,\"status\":\"partial\",\"mode\":\"" << (full_records ? "light" : "timestamps")
            << "\",\"request_id\":\"" << request << "\"}\n";
    pending.close();
    if (!pending) { error = "capture partial marker write failed"; return false; }
    if (r.buffer.empty()) r.buffer.assign(kCapacityBytes, 0);
    r.wall_unix_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    g_full = full_records;
    g_active = true;
    return true;
}

// Serializes after the recorded interval. The file is
//   "MM3LITE1", u32 version, u32 json_bytes, u64 record_bytes, JSON, padding to 8, records.
inline bool finish(const std::string& status, std::string& error) {
    if (!g_active) return true;
    g_active = false;
    auto& r = recorder();
    const std::string meta = std::string("{\"mode\":\"") + (g_full ? "light" : "timestamps") + "\",\"format_version\":" + std::to_string(kFormatVersion) +
        ",\"experiment_id\":\"" + r.experiment + "\",\"run_id\":\"" + r.run + "\",\"request_id\":\"" + r.request +
        "\",\"status\":\"" + status + "\",\"wall_unix_ns\":" + std::to_string(r.wall_unix_ns) +
        ",\"clock\":\"std::chrono::steady_clock nanoseconds\",\"attempted_records\":" + std::to_string(r.attempted) +
        ",\"dropped_records\":" + std::to_string(r.dropped) + ",\"record_bytes\":" + std::to_string(r.used) +
        ",\"capacity_bytes\":" + std::to_string(r.buffer.size()) + "}";
    std::ofstream out(artifact_path(r), std::ios::out | std::ios::binary);
    const uint32_t version = kFormatVersion, json = uint32_t(meta.size());
    const uint64_t records = r.used;
    out.write("MM3LITE1", 8);
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&json), sizeof(json));
    out.write(reinterpret_cast<const char*>(&records), sizeof(records));
    out.write(meta.data(), std::streamsize(meta.size()));
    static constexpr char zeros[8] = {};
    out.write(zeros, std::streamsize((8 - (meta.size() % 8)) % 8));
    if (r.used) out.write(reinterpret_cast<const char*>(r.buffer.data()), std::streamsize(r.used));
    out.close();
    if (!out) { error = "light capture artifact write failed"; return false; }
    if (std::remove(marker_path(r).c_str()) != 0) { error = "capture partial marker removal failed"; return false; }
    return true;
}
}  // namespace mm3::capture::light
