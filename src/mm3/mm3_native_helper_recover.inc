// Included in Model::Impl after mm3_native_third.inc. A helper card's result column with a non-finite value, recomputed on
// the CPU lane and added in place of the card's: the user's decision of 2026-11-14, "It should recompute that column on
// another lane and carry on, make sure it logs if it happens". The design folds in the diagnosis agent's (nonfinite-
// helper.md section 8, nonfinite-helper-recompute.diff): its record sources, its repair and its STAT warning keys.
//   - collect leaves such a column out of the pass's output and lists it (NativeHelper::Refused, with where the value went
//     bad). helper_recover runs it on the CPU lane, on the serving thread, before the publication the 5090 waits on (the
//     lane has finished its own batch by then; the prompt reader has no CPU lane work), and adds it into the token's row.
//   - Its record (mm3_helper_recover.hpp): a 3090 column's from the native pack, assembled again as at load; a 4080
//     column's the HOST record (the pack if that is missing). Never the card's slot, which may be what failed.
//   - Afterwards, at the next forward boundary (helper_repairs_run, after helper_live_progress), the expert's record on
//     that card is read back and compared with the pack's; one that differs is rewritten in its slot and read back. A
//     3090 record that still differs poisons, as before the decision; a 4080 copy that still differs leaves the pool (its
//     ready bit cleared: the HOST serves it). The expert's HOST frame, if it has one, is compared and rewritten the same
//     way. While the expert is the live swap's, or a reader holds its frame, the check waits for the next boundary.
//   - Each event: an engine log line ("mm3: helper nonfinite recovered #n: ..."), Stats::helper_nonfinite_recovered (the
//     request's STAT field of that name), and a STAT warning keyed helper_recompute:cudaD:layerL:expertE:positionP:cpu
//     (take_helper_fixes; mm3_main.cpp adds them to the request's warnings). Each check: an engine log line.
//   - A recompute that fails or is non-finite too poisons the card and fails the pass, as a failed collect did before.
//
// The cascade a recovered column prevents (03:12 PST 2026-11-14, before the decision): collect poisoned the card, then
//   1. the window set native_cpu_failed, which is never cleared (drafting and native_state_poisoned refuse from then on);
//   2. the failed forward's execution boundary (r2::execution_boundary) set retirement_uncertain: every placement refused
//      ("native placement requires a healthy, complete native HOST bank");
//   3. the 3090's lease in flight failed at its next poll (live_advance_locked on a poisoned helper) and helper_live_progress
//      set placement_failed.
// A recovered column leaves the card unpoisoned and the pass successful, so none of them fires, and a lease in flight
// carries on. Every other failure (a CUDA error, a deadline) still poisons.
long helper_nonfinite_count = 0;
uint8_t* recover_record = nullptr;     // one record with its header, portable pinned (the pack's; also a rewrite's staging)
long long recover_record_id = -1;      // the expert (layer * 1024 + e) whose pack record it holds, -1: none
struct HelperFixEvent { std::string key, message; };
std::vector<HelperFixEvent> helper_fix_events;   // since the last take_helper_fixes (the request's STAT warnings)
struct HelperRepair { bool third_card = false; int layer = 0, expert = 0; };
std::vector<HelperRepair> helper_repairs_due;    // records to check at the next forward boundary

// the positions of the decode window's rows (forward_rows: each row's own; else the chunk's, ending at cur_pos)
std::function<long long(int)> window_position(int T) const {
    return [this, T](int t) { return rseq.empty() ? (long long) cur_pos - (T - 1) + t : (long long) rpos[(size_t) t]; };
}

// The native pack's pieces of expert e are still mapped (Pack keeps every shard mapped; the load only dropped their
// pages), and the record layout is the shared one.
bool recover_pack_available(const Layer& l, int e) const {
    if (!native || e < 0 || size_t(e) >= l.src.size() || l.blob != sizeof(ne::Header) + nf::record_stride(g.dim, g.ff)) return false;
    const Src& s = l.src[size_t(e)];
    for (int r = 0; r < 3; ++r) if (!s.p[3 * r] || !s.p[3 * r + 1]) return false;
    return true;
}
// expert e of layer l from the pack into recover_record (header and RG16 payload, as the load assembled it); kept for the
// repair at the boundary. The pages read leave RAM again (drop_src), as after the load.
bool recover_from_pack(const Layer& l, int e, std::string& why) {
    const long long id = (long long) (&l - L.data()) * 1024 + e;
    if (recover_record_id == id) return true;
    if (!recover_pack_available(l, e)) { why = "the pack's pieces are not mapped"; return false; }
    if (!recover_record) {
        void* p = nullptr;
        if (cudaHostAlloc(&p, l.blob, cudaHostAllocPortable) != cudaSuccess) {
            (void) cudaGetLastError(); why = "no portable pinned buffer for one record"; return false;
        }
        recover_record = static_cast<uint8_t*>(p);
    }
    recover_record_id = -1;
    try { assemble(l, e, recover_record); }
    catch (const std::exception& ex) { why = std::string("assembling the pack's record: ") + ex.what(); return false; }
    drop_src(l, e);
    recover_record_id = id;
    return true;
}
// A rewrite whose transfer failed may leave CUDA reading the buffer: kept allocated for good, never reused.
void recover_record_abandon() { recover_record = nullptr; recover_record_id = -1; }

// The pass's refused columns recomputed and added into result ([pass rows][dim], the card's merged output). x32: the
// pass's rows as F32 (the decode window's rx; the cards had them as BF16) or x16: as BF16 (the prompt reader's). For the
// log: where ("decode", "prompt read"), row0 the pass's first row among `rows`, position(row) the row's position.
bool helper_recover(NativeHelper& card, bool third_card, const std::vector<NativeHelper::Refused>& refused, const float* x32,
                    const uint16_t* x16, float* result, const char* where, int row0, int rows,
                    const std::function<long long(int)>& position, std::string& err) {
    namespace hr = ::mm3::helper_recover;
    std::vector<float> xrow((size_t) g.dim), column((size_t) g.dim);
    const int dev = card.device();
    std::string dev_name;
    {
        cudaDeviceProp prop{};
        if (cudaGetDeviceProperties(&prop, dev) == cudaSuccess) dev_name = prop.name; else (void) cudaGetLastError();
    }
    const char* card_name = third_card ? "4080 third" : "3090 helper";
    for (const auto& r : refused) {
        const double began = now();
        const auto fail = [&](const std::string& why) {
            // (the message keeps the collect error's first words: "nonfinite helper output (device ...")
            err = r.detail + "; recomputing it on the CPU lane failed: " + why;
            card.poison(err);
            std::fprintf(stderr, "mm3: helper nonfinite NOT recovered: %s CUDA%d, %s, layer %d expert %d row %d: %s\n", card_name, dev,
                         where, r.layer, r.expert, row0 + r.token, err.c_str());
            return false;
        };
        if (!x32 && !x16) return fail("no input rows");
        if (r.layer < 0 || r.layer >= g.n_layers || r.expert < 0 || r.expert >= g.n_exp || r.token < 0) return fail("invalid column");
        if (!lane || !native_cpu_workspace) return fail("this load has no CPU lane");
        if (x32) std::copy_n(x32 + (size_t) r.token * g.dim, (size_t) g.dim, xrow.data());
        else for (int k = 0; k < g.dim; ++k) xrow[(size_t) k] = hr::bf16_value(x16[(size_t) r.token * g.dim + (size_t) k]);
        Layer& l = L[(size_t) r.layer];
        const uint8_t* record = nullptr;
        hr::Source used = hr::Source::pack;
        std::string tried;   // (sources that failed before the one used, and why)
        const bool host_record = !compressed_host() && pooled(l, r.expert) &&
                                 l.blob == sizeof(ne::Header) + nf::record_stride(g.dim, g.ff);
        for (const hr::Source s : hr::sources(third_card, recover_pack_available(l, r.expert), host_record)) {
            std::string why;
            if (s == hr::Source::host_bank) record = l.host_blob[(size_t) r.expert];
            else if (recover_from_pack(l, r.expert, why)) record = recover_record;
            if (record) { used = s; break; }
            tried += (tried.empty() ? "" : "; ") + std::string(hr::source_key(s)) + " failed: " + why;
        }
        if (!record) return fail(tried.empty() ? std::string("no record to read (neither the pack's nor a HOST record)") :
                                 "no record could be read (" + tried + ")");
        const std::vector<NativeCpuExpert> jobs{{record, r.weight}};
        const std::vector<std::vector<std::pair<int, float>>> users{{{0, r.weight}}};
        const bool ran = native_cpu_run_multi_batch(*lane, g.dim, g.ff, jobs, users, xrow.data(), 1, column.data(), nullptr,
                                                    native_cpu_workspace, native_cpu_bytes);
        const std::string from = std::string(" (record from ") + hr::source_key(used) + ")";
        if (!ran) return fail("the CPU lane refused the job" + from);
        const auto added = hr::add_column(result + (size_t) r.token * g.dim, column.data(), (size_t) g.dim);
        if (added == hr::Add::nonfinite_column) return fail("the CPU lane's column is non-finite too" + from);
        if (added == hr::Add::nonfinite_sum) return fail("the row's sum with the CPU lane's column is not finite" + from);
        hr::Event ev;
        ev.card = card_name; ev.device = dev; ev.device_name = dev_name; ev.where = where;
        ev.layer = r.layer; ev.expert = r.expert; ev.slot = r.slot; ev.token = row0 + r.token; ev.rows = rows;
        ev.position = position(r.token);
        ev.weight = r.weight; ev.sum = r.sum; ev.bad = r.bad; ev.detail = r.detail;
        ev.source = used; ev.fallback = tried; ev.ms = 1e3 * (now() - began);
        ev.count = ++helper_nonfinite_count;
        st.helper_nonfinite_recovered = helper_nonfinite_count;
        const std::string line = hr::event_line(ev);
        std::fprintf(stderr, "mm3: %s; its record on the card is checked at the next forward boundary\n", line.c_str());
        std::fflush(stderr);
        helper_fix_events.push_back({hr::warning_key(dev, r.layer, r.expert, ev.position), line});
        bool due = false;
        for (const auto& q : helper_repairs_due) due |= q.third_card == third_card && q.layer == r.layer && q.expert == r.expert;
        if (!due) helper_repairs_due.push_back({third_card, r.layer, r.expert});
        if (capture::enabled())
            capture::event("helper_nonfinite_recovered", "\"card\":" + capture::quote(card_name) + ",\"device\":" + std::to_string(dev) +
                ",\"where\":" + capture::quote(where) + ",\"layer\":" + std::to_string(r.layer) + ",\"expert\":" + std::to_string(r.expert) +
                ",\"row\":" + std::to_string(row0 + r.token) + ",\"position\":" + std::to_string(ev.position) + ",\"bad\":" + std::to_string(r.bad) +
                ",\"sum\":" + (r.sum ? "true" : "false") + ",\"source\":" + capture::quote(hr::source_key(used)) +
                ",\"detail\":" + capture::quote(r.detail));
    }
    return true;
}

// At a forward boundary (no pass in flight, the CPU lane idle, after helper_live_progress): each expert recomputed since
// is checked against the pack's record on its card, and in its HOST frame if it has one, rewritten where it differs and
// read back (see the top of this file). False only on a CUDA error or a 3090 record that a rewrite did not repair.
bool helper_repairs_run(std::string& err) {
    if (helper_repairs_due.empty()) return true;
    namespace hr = ::mm3::helper_recover;
    const size_t stride = nf::record_stride(g.dim, g.ff);
    std::vector<HelperRepair> later;
    for (size_t at = 0; at < helper_repairs_due.size(); ++at) {
        const HelperRepair r = helper_repairs_due[at];
        Layer& l = L[(size_t) r.layer];
        const int e = r.expert;
        const long long key = (long long) r.layer * 1024 + e;
        NativeHelper* h = r.third_card ? third.get() : helper.get();
        const int dev = h ? h->device() : -1;
        const char* card_name = r.third_card ? "4080 third" : "3090 helper";
        if (helper_live_leased(key)) { later.push_back(r); continue; }   // (the live swap's: the next boundary)
        auto* frame = pooled(l, e) && !compressed_host() ? const_cast<uint8_t*>(l.host_blob[(size_t) e]) : nullptr;
        if (frame && helper_live_source_busy(D[(size_t) head_dev], key, frame)) { later.push_back(r); continue; }
        const bool on_card = h && (r.third_card ? ((l.third_ready[(size_t) e / 32] >> (e % 32)) & 1u) != 0 : helper_owns(l, e));
        std::string why;
        if (!recover_from_pack(l, e, why)) {
            // (no reference to check against: logged and dropped, not retried each forward. A wrong record gives wrong
            //  finite values, never a non-finite one - nonfinite-helper.md section 4 - the same risk as before the event)
            std::fprintf(stderr, "mm3: helper record check: %s CUDA%d layer %d expert %d: not checked, the pack's record could not "
                         "be read: %s\n", card_name, dev, r.layer, e, why.c_str());
            continue;
        }
        const auto* want = reinterpret_cast<const ne::Header*>(recover_record);
        const uint8_t* payload = recover_record + sizeof(ne::Header);
        const auto frame_same = [&] {
            const auto* have = reinterpret_cast<const ne::Header*>(frame);
            return std::memcmp(frame + sizeof(ne::Header), payload, stride) == 0 &&
                   std::memcmp(have->weight_global, want->weight_global, sizeof want->weight_global) == 0 &&
                   std::memcmp(have->input_global, want->input_global, sizeof want->input_global) == 0;
        };
        bool before = false, host_same = true;
        if (on_card && !h->verify_expert(r.layer, e, payload, stride, want->weight_global, want->input_global, before, err)) {
            err = "helper record check on CUDA" + std::to_string(dev) + ": " + err; return false;
        }
        if (frame) host_same = frame_same();
        if ((on_card && !before) || !host_same) {
            // a difference: before anything is rewritten, the reference is assembled from the pack again and compared
            // again - the copy kept since the recompute sits in the same DRAM the fault is suspected in
            recover_record_id = -1;
            if (!recover_from_pack(l, e, why)) {
                std::fprintf(stderr, "mm3: helper record check: %s CUDA%d layer %d expert %d: a difference found, not repaired: the "
                             "pack's record could not be read again: %s\n", card_name, dev, r.layer, e, why.c_str());
                continue;
            }
            if (on_card && !h->verify_expert(r.layer, e, payload, stride, want->weight_global, want->input_global, before, err)) {
                err = "helper record check on CUDA" + std::to_string(dev) + ": " + err; return false;
            }
            if (frame) host_same = frame_same();
        }
        std::string outcome;
        if (on_card) {
            bool after = false;
            if (!before) {
                bool unsafe = false;
                if (!h->rewrite_expert(r.layer, e, payload, stride, want->weight_global, want->input_global, unsafe, err)) {
                    if (unsafe) recover_record_abandon();
                    err = "helper record repair on CUDA" + std::to_string(dev) + ": " + err;
                    return false;
                }
                if (!h->verify_expert(r.layer, e, payload, stride, want->weight_global, want->input_global, after, err)) {
                    err = "helper record check on CUDA" + std::to_string(dev) + ": " + err; return false;
                }
            }
            switch (hr::repair_outcome(before, after, r.third_card)) {
                case hr::Repair::equal: outcome = "its record on the card equals the pack's"; break;
                case hr::Repair::rewritten: outcome = "its record on the card differed from the pack's: rewritten from the pack, read back equal"; break;
                case hr::Repair::drop:
                    l.third_ready[(size_t) e / 32] &= ~(1u << (e % 32));
                    outcome = "its record on the card still differs from the pack's after a rewrite: left out of the pool (the HOST serves it)";
                    break;
                case hr::Repair::poison:
                    err = std::string("helper record on CUDA") + std::to_string(dev) + " layer " + std::to_string(r.layer) + " expert " +
                          std::to_string(e) + " still differs from the pack's after a rewrite";
                    std::fprintf(stderr, "mm3: helper record check: %s\n", err.c_str());
                    h->poison(err);
                    return false;
            }
        } else outcome = "not on that card now";
        if (frame) {   // (the HOST's record: every 4080 expert's; a 3090 expert's once a swap or a placement moved it)
            if (!host_same) std::memcpy(frame, recover_record, sizeof(ne::Header) + stride);
            outcome += host_same ? "; its HOST record equals the pack's" : "; its HOST record differed from the pack's: rewritten from it";
        }
        std::fprintf(stderr, "mm3: helper record check: %s CUDA%d layer %d expert %d: %s\n", card_name, dev, r.layer, e, outcome.c_str());
    }
    helper_repairs_due.swap(later);
    return true;
}
