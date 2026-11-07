// CPU-only: exercises the light recorder through the real capture control path.
// With an output directory argument it leaves its artifacts there, so
// tools/mm3/test_native_light.py can check the Python reader against these layouts.
#include "../src/mm3/mm3_capture.hpp"
#include <cassert>
#include <filesystem>
#include <iostream>
#include <unistd.h>

namespace c = mm3::capture;
namespace l = mm3::capture::light;

static void control(const std::filesystem::path& dir, const char* mode, const char* request) {
    const auto path = dir / "control";
    std::ofstream f(path);
    f << mode << " experiment run " << request << ' ' << dir.string() << '\n';
    f.close();
    setenv("MM3_CAPTURE_CONTROL", path.c_str(), 1);
}

static void write_one_forward(int64_t forward, int row_clock) {
    l::set_forward(forward);
    const uint64_t t0 = l::clock_ns();
    l::LayerDraft draft;
    draft.begin = t0;
    draft.pre_end = t0 + 10; draft.join_end = t0 + 20; draft.boundary_end = t0 + 30; draft.plan_end = t0 + 40;
    draft.helper_submit_end = t0 + 50; draft.primary_submit_end = t0 + 60; draft.cpu_end = t0 + 70;
    draft.collect_end = t0 + 80; draft.end = t0 + 90;
    draft.helper_enqueue = t0 + 41; draft.helper_wait_end = t0 + 79;
    draft.free_slots = 3; draft.reserved_slots = 1; draft.foreground_slots = 2; draft.pending_promotions = 0;
    draft.exchanges_pending = 1; draft.helper_lease = 2; draft.stage_max = 3; draft.stage_selected = 1; draft.total_slots = 8;
    draft.service[0] = l::kServiceHelper; draft.service[1] = l::kServiceCpu;
    draft.service[2] = l::kServicePrimaryResident; draft.service[3] = l::kServiceStagedNew;
    const int original[4] = {1, 2, 3, 4}, executed[4] = {1, 2, 3, 5};
    for (int layer = 3; layer < 5; ++layer) l::layer(draft, layer, 1, 4, original, executed);
    l::demand_copy(3, 5, 7, 3 * 1024 + 9, 1234, t0);
    l::foreground_retire(7, 3 * 1024 + 5);
    const int64_t counters[l::kForwardCounters] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18};
    l::forward_record(t0, t0 + 1, t0 + 2, t0 + 3, row_clock, 100 + int(forward), 1, 1, true, counters);
}

int main(int argc, char** argv) {
    const bool keep = argc > 1;
    const std::filesystem::path dir = keep ? std::filesystem::path(argv[1]) :
        std::filesystem::temp_directory_path() / ("mm3-light-" + std::to_string(getpid()));
    std::filesystem::create_directories(dir);
    std::string error;

    // Off and timeline controls never enable the recorder.
    control(dir, "off", "off-request");
    assert(c::begin(error) && !l::active() && !c::enabled() && !c::detailed());
    assert(l::stamp() == 0);
    assert(c::end("ok", error));
    assert(!std::filesystem::exists(dir / "off-request.light"));
    // A missing control leaves everything off.
    setenv("MM3_CAPTURE_CONTROL", (dir / "absent").c_str(), 1);
    assert(c::begin(error) && !l::active());

    // Light mode must not enable text capture, compression or NVTX.
    control(dir, "light", "valid");
    assert(c::begin(error));
    assert(l::active() && !c::enabled() && !c::detailed());
    assert(std::filesystem::exists(dir / "valid.partial"));
    std::vector<uint8_t> location(4 * 8, l::kLocationHost);
    location[3 * 8 + 1] = l::kLocationPrimary;
    location[3 * 8 + 2] = l::kLocationHelper;
    location[3 * 8 + 3] = l::kLocationHost;
    location[3 * 8 + 4] = l::kLocationPrimary;
    std::vector<l::SlotState> slots(2);
    slots[0].owner = 3 * 1024 + 5; slots[0].pending = -1; slots[0].foreground = 1;
    slots[1].owner = -1; slots[1].pending = -1; slots[1].reserved = 1;
    std::vector<l::ExchangeState> exchanges{{3 * 1024 + 3, 3 * 1024 + 4, 1, 1}};
    l::map(0, 10, 4, 8, location, slots, exchanges, -1, -1, 0);
    l::generation_mark(l::kGenerationBegin, l::kScopeRequest, true, 10, 100, 0, 0, 0, 0, 0, 0);
    write_one_forward(1, 11);
    l::primary_issue(3 * 1024 + 3, 3 * 1024 + 4, 1, -1, false, 111, 222, l::clock_ns());
    l::primary_publish(3 * 1024 + 3, 3 * 1024 + 4, 1, -1);
    l::helper_event(l::kHelperLease, 3 * 1024 + 6, 3 * 1024 + 2, l::clock_ns());
    l::helper_event(l::kHelperWithdrawn, 3 * 1024 + 6, 3 * 1024 + 2);
    l::helper_event(l::kHelperPublished, 3 * 1024 + 6, 3 * 1024 + 2);
    l::helper_event(l::kHelperRetired, 3 * 1024 + 6, 3 * 1024 + 2);
    write_one_forward(2, 12);
    l::generation_mark(l::kGenerationEnd, l::kScopeRequest, true, 12, 102, 2, 0, 0, 2, 0, 12.5);
    // Apply the same mutations to the expected final map.
    location[3 * 8 + 3] = l::kLocationPrimary;
    location[3 * 8 + 4] = l::kLocationHost;
    location[3 * 8 + 6] = l::kLocationHelper;
    location[3 * 8 + 2] = l::kLocationHost;
    l::map(1, 12, 4, 8, location, slots, {}, -1, -1, 0);
    assert(c::end("ok", error));
    assert(!l::active());
    assert(std::filesystem::exists(dir / "valid.light") && !std::filesystem::exists(dir / "valid.partial"));

    // Timestamps-only keeps clock records and drops assignments, state and events.
    control(dir, "timestamps", "stamps");
    assert(c::begin(error) && l::active() && !l::full() && !c::enabled() && !c::detailed());
    assert(l::stamp() != 0);
    l::map(0, 10, 4, 8, location, slots, exchanges, -1, -1, 0);
    l::generation_mark(l::kGenerationBegin, l::kScopeRequest, true, 10, 100, 0, 0, 0, 0, 0, 0);
    write_one_forward(1, 11);
    l::primary_issue(3 * 1024 + 3, 3 * 1024 + 4, 1, -1, false, 111, 222, l::clock_ns());
    l::primary_publish(3 * 1024 + 3, 3 * 1024 + 4, 1, -1);
    l::helper_event(l::kHelperLease, 3 * 1024 + 6, 3 * 1024 + 2, l::clock_ns());
    write_one_forward(2, 12);
    l::generation_mark(l::kGenerationEnd, l::kScopeRequest, true, 12, 102, 2, 0, 0, 2, 0, 12.5);
    l::map(1, 12, 4, 8, location, slots, {}, -1, -1, 0);
    assert(c::end("ok", error));
    assert(!l::active() && !l::full());
    assert(std::filesystem::exists(dir / "stamps.light") && !std::filesystem::exists(dir / "stamps.partial"));

    // A repeat of the same request identity is refused rather than overwritten.
    control(dir, "light", "valid");
    assert(!c::begin(error) && !l::active());
    error.clear();

    // A full buffer is counted and reported, never silently truncated.
    control(dir, "light", "overflow");
    assert(c::begin(error) && l::active());
    const int original[4] = {1, 2, 3, 4};
    l::LayerDraft draft;
    draft.begin = 1;
    const size_t record = sizeof(l::LayerFixed) + 12;
    const size_t writes = l::kCapacityBytes / ((record + 7) & ~size_t(7)) + 5;
    for (size_t i = 0; i < writes; ++i) l::layer(draft, 3, 1, 4, original, original);
    assert(l::recorder().dropped == 5 || l::recorder().dropped == 6);
    assert(c::end("ok", error));

    if (!keep) std::filesystem::remove_all(dir);
    std::cout << "light recorder control, round trip and overflow passed\n";
}
