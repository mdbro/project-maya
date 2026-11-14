// Included in Model::Impl after mm3_native_third.inc. A helper card's result column with a non-finite value, recomputed on
// the CPU lane and added in place of the card's: the user's decision of 2026-11-14, "It should recompute that column on
// another lane and carry on, make sure it logs if it happens". NativeHelper::collect leaves such a column out of the pass's
// output and lists it (NativeHelper::Refused, with where the value went bad); helper_recover reads the expert's record from
// an independent copy (mm3_helper_recover.hpp: the native pack, else the host bank for a 4080 column, else the card's slot
// read back), runs the column on the CPU lane and adds it into the pass's output row.
//
// Each event: an engine log line ("mm3: helper nonfinite recovered #n: ..."), Stats::helper_nonfinite_recovered (the
// request's STAT field of that name) and a STAT warning carrying the last event's line. If the recompute fails or is
// non-finite too, the card is poisoned as before the decision and the pass fails.
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
std::string helper_nonfinite_last;          // the last event's log line (the STAT warning's text)
uint8_t* recover_record = nullptr;          // one record assembled from the native pack (4 KiB aligned, at the first event)
uint8_t* recover_pinned = nullptr;          // one record, portable pinned: a card's slot read back (at the first such event)

// The native pack's pieces of expert e are still mapped (Pack keeps every shard mapped; the bank fill only dropped their
// pages), and the record layout is the shared one.
bool recover_pack_available(const Layer& l, int e) const {
    if (!native || e < 0 || size_t(e) >= l.src.size() || l.blob != sizeof(ne::Header) + nf::record_stride(g.dim, g.ff)) return false;
    const Src& s = l.src[size_t(e)];
    for (int r = 0; r < 3; ++r) if (!s.p[3 * r] || !s.p[3 * r + 1]) return false;
    return true;
}

// The pass's refused columns recomputed and added into result ([pass rows][dim], the card's merged output). x32: the
// pass's rows as F32 (the decode window's rx; the cards had them as BF16) or x16: as BF16 (the prompt reader's). For the
// log: where ("decode", "prompt read"), row0 the pass's first row among `rows`, position0 the position of row 0.
bool helper_recover(NativeHelper& card, bool third_card, const std::vector<NativeHelper::Refused>& refused, const float* x32,
                    const uint16_t* x16, float* result, const char* where, int row0, int rows, long long position0, std::string& err) {
    namespace hr = ::mm3::helper_recover;
    const size_t blob = sizeof(ne::Header) + nf::record_stride(g.dim, g.ff);
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
        bool pack_read = false;
        const bool host_record = !compressed_host() && pooled(l, r.expert) && l.blob == blob;
        for (const hr::Source s : hr::sources(third_card, recover_pack_available(l, r.expert), host_record)) {
            std::string why;
            if (s == hr::Source::pack) {
                if (!recover_record) recover_record = static_cast<uint8_t*>(std::aligned_alloc(4096, (blob + 4095) / 4096 * 4096));
                if (!recover_record) why = "no scratch for one record";
                else {
                    try { assemble(l, r.expert, recover_record); record = recover_record; pack_read = true; }
                    catch (const std::exception& e) { why = e.what(); }
                }
            } else if (s == hr::Source::host_bank) {
                record = l.host_blob[(size_t) r.expert];
            } else {
                if (!recover_pinned) {
                    void* p = nullptr;
                    if (cudaHostAlloc(&p, blob, cudaHostAllocPortable) == cudaSuccess) recover_pinned = static_cast<uint8_t*>(p);
                    else { (void) cudaGetLastError(); why = "no portable pinned scratch for one record"; }
                }
                if (recover_pinned) {
                    float wg[3] = {}, ig[3] = {};
                    bool unsafe = false;
                    if (card.read_record(r.layer, r.expert, recover_pinned + sizeof(ne::Header), blob - sizeof(ne::Header), wg, ig, unsafe, why)) {
                        ne::Header h{};
                        std::copy_n(wg, 3, h.weight_global);
                        std::copy_n(ig, 3, h.input_global);
                        std::memcpy(recover_pinned, &h, sizeof h);
                        record = recover_pinned;
                    } else if (unsafe) recover_pinned = nullptr;   // (CUDA may still write it: never freed nor reused)
                }
            }
            if (record) { used = s; break; }
            tried += (tried.empty() ? "" : "; ") + std::string(hr::source_key(s)) + " failed: " + why;
        }
        if (!record) return fail("no record could be read (" + tried + ")");
        const std::vector<NativeCpuExpert> jobs{{record, r.weight}};
        const std::vector<std::vector<std::pair<int, float>>> users{{{0, r.weight}}};
        const bool ran = native_cpu_run_multi_batch(*lane, g.dim, g.ff, jobs, users, xrow.data(), 1, column.data(), nullptr,
                                                    native_cpu_workspace, native_cpu_bytes);
        if (pack_read) drop_src(l, r.expert);   // (the pages just read leave RAM again, as after the bank fill)
        const std::string from = std::string(" (record from ") + hr::source_key(used) + ")";
        if (!ran) return fail("the CPU lane refused the job" + from);
        const auto added = hr::add_column(result + (size_t) r.token * g.dim, column.data(), (size_t) g.dim);
        if (added == hr::Add::nonfinite_column) return fail("the CPU lane's column is non-finite too" + from);
        if (added == hr::Add::nonfinite_sum) return fail("the row's sum with the CPU lane's column is not finite" + from);
        hr::Event ev;
        ev.card = card_name; ev.device = dev; ev.device_name = dev_name; ev.where = where;
        ev.layer = r.layer; ev.expert = r.expert; ev.slot = r.slot; ev.token = row0 + r.token; ev.rows = rows;
        ev.position = position0 >= 0 ? position0 + row0 + r.token : -1;
        ev.weight = r.weight; ev.sum = r.sum; ev.bad = r.bad; ev.detail = r.detail;
        ev.source = used; ev.fallback = tried; ev.ms = 1e3 * (now() - began);
        ev.count = ++helper_nonfinite_count;
        st.helper_nonfinite_recovered = helper_nonfinite_count;
        helper_nonfinite_last = hr::event_line(ev);
        std::fprintf(stderr, "mm3: %s\n", helper_nonfinite_last.c_str());
        std::fflush(stderr);
        if (capture::enabled())
            capture::event("helper_nonfinite_recovered", "\"card\":" + capture::quote(card_name) + ",\"device\":" + std::to_string(dev) +
                ",\"where\":" + capture::quote(where) + ",\"layer\":" + std::to_string(r.layer) + ",\"expert\":" + std::to_string(r.expert) +
                ",\"row\":" + std::to_string(row0 + r.token) + ",\"bad\":" + std::to_string(r.bad) + ",\"sum\":" + (r.sum ? "true" : "false") +
                ",\"source\":" + capture::quote(hr::source_key(used)) + ",\"detail\":" + capture::quote(r.detail));
    }
    return true;
}
