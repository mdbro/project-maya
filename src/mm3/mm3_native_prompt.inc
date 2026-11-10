// ---- prompt_layerwise (item 1 of the order of 2026-11-10; design: runs/prompt-reading-20261110/design.md).
// A prompt read one layer at a time: for each layer, the dense work for every row in 16-row groups aligned as today's
// chunks (window_pre's body unchanged, each group on its slice of an all-rows residual), the routing of every row
// against the published residents only, then each picked expert served once for all its rows: the 5090's residents in
// place, the 3090's per 16-row group exactly as window_post builds its jobs, every other expert uploaded once through
// the staging frames on the copy stream and applied to all the rows that picked it, and the window path's combine. No
// CPU lane. Default off (prompt_layerwise=0: today's chunks); prompts of kPromptMinRows rows or fewer keep the chunks.
//
// prompt_check (a diagnostic request setting) makes the two paths comparable bit for bit: in both, the prompt's routing
// mask is the residents only, and today's chunked path stages every host expert of a pass (no CPU lane; it needs enough
// frames). MM3_PROMPT_HASH=<dir> writes, for each prompt that starts at position 0, every layer's output row hashes, the
// routes and the drafter's context rows, from either path, to <dir>/prompt-<n>.txt. MM3_PROMPT_TRACE=<file> appends one
// JSON line per layer of the new read with its phases' times.
static constexpr int kPromptMinRows = 33;
static constexpr int kPromptMaxJobs = 64;          // jobs a launch (the native kernels take up to 65,535)
static constexpr size_t kPromptReserve = 450ull << 20;   // VRAM left free while the read's buffers exist
static constexpr size_t kPromptBudget = 400ull << 20;    // the read's buffers at most
static constexpr size_t kPromptRowBytes = 110592;  // h F32, shared out F32, input BF16, four BF16 partials
int prompt_layerwise_mode = 0;
int prompt_check_mode = 0;
bool prompt_reading = false;     // window_pre / route_layer: a group of the layer-by-layer read
long long prompt_hash_counter = 0;
std::FILE* prompt_hash_file = nullptr;
struct PromptHost {                 // pinned host buffers, grown on demand and kept
    int rows = 0;                   // capacity
    float* emb = nullptr;           // [rows][dim]
    uint16_t* x16 = nullptr;        // [rows][dim] the helper's inputs
    float* helper = nullptr;        // mapped [rows][dim]: the 3090's sums (0 + its result, as window_post's cpu_rows)
    int* flag = nullptr;            // mapped [rows]
    int *top = nullptr, *orig = nullptr;   // [rows][4]
    float* w = nullptr;             // [rows][4]
    float* pw = nullptr;            // [rows][4] the combine's weights
    uint8_t* served = nullptr;      // [rows][4] the pick served on the 5090
    float* taps = nullptr;          // mapped [taps][rows][dim]
    int ntaps = 0;
    ne::PromptJob* jobs = nullptr;  // [rows / 2 + 256]
    int max_jobs = 0;
} ph;
struct PromptDev {                  // device buffers, one read's
    int rows = 0;
    float *h = nullptr, *sh = nullptr, *pw = nullptr;
    uint16_t *x16 = nullptr, *part = nullptr;
    int* pos = nullptr;
    uint8_t* served = nullptr;
    ne::PromptJob* jobs = nullptr;
    void* scratch = nullptr;
    size_t scratch_bytes = 0;
};
struct PromptLayerEvents { cudaEvent_t e[9] = {}; bool link = false, used = false; long res_jobs = 0, uploads = 0, groups = 0;
                           long helper_cols = 0, helper_passes = 0; double helper_ms = 0, host_ms = 0, route_wait_ms = 0; };
std::vector<PromptLayerEvents> prompt_ev;   // per layer, created once
std::vector<cudaEvent_t> prompt_group_ev, prompt_landed, prompt_release;
cudaEvent_t prompt_start_ev = nullptr;

bool prompt_layerwise_supported() const {
    return native && o.native_cooperative && !o.native_stage_prefix && D.size() == 1 && o.seqs == 1 && helper &&
           g.k_exp == 4 && o.fast_decode;
}
// Why a prompt of n rows takes today's chunks (empty: the layer-by-layer read). No side effects.
std::string prompt_read_refusal(int n, int want) const {
    if (prompt_layerwise_mode != 1) return "off";
    if (n < kPromptMinRows) return "short";
    if (!prompt_layerwise_supported()) return "unsupported";
    if (want < 0 || want > kWin) return "want";
    if (generation_active || !rseq.empty() || !imgs.empty() || pf.on) return "state";
    if (helper_dispatch_current != "worker" || helper_live_a >= 0 || helper_queue_pending) return "helper";
    if (!primary_adaptive || adaptive_policy.legacy) return "policy";
    if (compressed_host() || capture::detailed() || capture::light::active() || replay_trace_active || replay_fixed_batch)
        return "capture";
    if (stage_poisoned || retirement_uncertain.load() || hf_demand_poisoned || placement_failed) return "poisoned";
    for (const Layer& l : L) {
        if (l.kvh.ring > 0) return "kv_streaming";
        if (l.moe && l.hf) return "host_free_layer";
    }
    int usable = 0;
    const Dev& d = D[0];
    for (int s = 0; s < d.n_slots; ++s)
        usable += !staging_reserved(d, s) && d.slot_pend[(size_t) s] < 0 && !slot_early_busy(d, s);
    if (usable < 2) return "frames";
    return {};
}
bool prompt_read_wanted(int n, int want) const { return prompt_read_refusal(n, want).empty(); }

// ---- the diagnostics
static uint64_t prompt_row_hash(const void* p, size_t bytes) {
    const uint8_t* b = (const uint8_t*) p;
    uint64_t h = 1469598103934665603ull;
    for (size_t i = 0; i + 8 <= bytes; i += 8) {
        uint64_t w; std::memcpy(&w, b + i, 8);
        h ^= w; h *= 1099511628211ull; h ^= h >> 29;
    }
    return h;
}
bool prompt_hash_on() const { return ::mm3::env("MM3_PROMPT_HASH") != nullptr; }
void prompt_hash_begin(int pos0) {
    if (!prompt_hash_on() || pos0 != 0) return;
    if (prompt_hash_file) std::fclose(prompt_hash_file);
    const std::string path = std::string(::mm3::env("MM3_PROMPT_HASH")) + "/prompt-" + std::to_string(++prompt_hash_counter) + ".txt";
    prompt_hash_file = std::fopen(path.c_str(), "w");
}
// rows F32 [n][dim] on the device at positions pos0.. : one line each
bool prompt_hash_rows(char kind, int li, const float* rows_dev, int n, int pos0, cudaStream_t s, std::string& err) {
    if (!prompt_hash_file || n <= 0) return true;
    std::vector<float> hh((size_t) n * g.dim);
    CK(cudaMemcpyAsync(hh.data(), rows_dev, hh.size() * 4, cudaMemcpyDefault, s));
    CK(cudaStreamSynchronize(s));
    for (int r = 0; r < n; ++r)
        std::fprintf(prompt_hash_file, "%c %d %d %016llx\n", kind, li, pos0 + r,
                     (unsigned long long) prompt_row_hash(hh.data() + (size_t) r * g.dim, (size_t) g.dim * 4));
    return true;
}
void prompt_hash_routes(int li, const int* top, const int* orig, int n, const int* pos) {
    if (!prompt_hash_file) return;
    for (int r = 0; r < n; ++r) {
        std::fprintf(prompt_hash_file, "r %d %d", li, pos[r]);
        for (int i = 0; i < 4; ++i) std::fprintf(prompt_hash_file, " %d", top[r * 4 + i]);
        for (int i = 0; i < 4; ++i) std::fprintf(prompt_hash_file, " %d", orig[r * 4 + i]);
        std::fputc('\n', prompt_hash_file);
    }
}
// the drafter's context input: [taps][stride][dim] (device-readable) for n rows at pos0
bool prompt_hash_taps(const float* taps_any, int n, int pos0, size_t stride, cudaStream_t s, std::string& err) {
    if (!prompt_hash_file || !draftm) return true;
    const int nt = (int) draftm->geo().taps.size();
    for (int j = 0; j < nt; ++j)
        if (!prompt_hash_rows('t', j, taps_any + (size_t) j * stride * g.dim, n, pos0, s, err)) return false;
    return true;
}
// prompt rows' router picks against executed ones (both paths: observe_routes calls this outside generation)
void prompt_route_counts(const int* top, const int* orig, int n) {
    for (int r = 0; r < n; ++r)
        for (int i = 0; i < 4; ++i) {
            const int e = top[r * 4 + i];
            bool kept = false;
            for (int j = 0; j < 4; ++j) kept |= orig[r * 4 + j] == e;
            ++st.prompt_picks; st.prompt_moved += !kept;
        }
}

// ---- buffers
bool prompt_host_grow(int rows, std::string& err) {
    const int nt = draftm && draft_running ? (int) draftm->geo().taps.size() : 0;
    if (rows <= ph.rows && nt <= ph.ntaps) return true;
    const double t0 = now();
    auto release = [](auto*& p) { if (p) cudaFreeHost((void*) p); p = nullptr; };
    release(ph.emb); release(ph.x16); release(ph.helper); release(ph.flag); release(ph.top); release(ph.orig); release(ph.w);
    release(ph.pw); release(ph.served); release(ph.taps); release(ph.jobs);
    ph.rows = 0; ph.ntaps = 0;
    const size_t R = (size_t) rows, D_ = (size_t) g.dim;
    const unsigned mapped = cudaHostAllocMapped | cudaHostAllocPortable;
    CK(cudaHostAlloc((void**) &ph.emb, R * D_ * 4, cudaHostAllocPortable));
    CK(cudaHostAlloc((void**) &ph.x16, R * D_ * 2, cudaHostAllocPortable));
    CK(cudaHostAlloc((void**) &ph.helper, R * D_ * 4, mapped));
    CK(cudaHostAlloc((void**) &ph.flag, R * 4, mapped));
    CK(cudaHostAlloc((void**) &ph.top, R * 16, cudaHostAllocPortable));
    CK(cudaHostAlloc((void**) &ph.orig, R * 16, cudaHostAllocPortable));
    CK(cudaHostAlloc((void**) &ph.w, R * 16, cudaHostAllocPortable));
    CK(cudaHostAlloc((void**) &ph.pw, R * 16, cudaHostAllocPortable));
    CK(cudaHostAlloc((void**) &ph.served, R * 4, cudaHostAllocPortable));
    if (nt) CK(cudaHostAlloc((void**) &ph.taps, (size_t) nt * R * D_ * 4, mapped));
    ph.max_jobs = rows / 2 + 256;
    CK(cudaHostAlloc((void**) &ph.jobs, (size_t) ph.max_jobs * sizeof(ne::PromptJob), cudaHostAllocPortable));
    ph.rows = rows; ph.ntaps = nt;
    st.prompt_host_alloc_ms += 1e3 * (now() - t0);
    return true;
}
void prompt_dev_free(PromptDev& b) {
    for (void* p : {(void*) b.h, (void*) b.sh, (void*) b.pw, (void*) b.x16, (void*) b.part, (void*) b.pos, (void*) b.served,
                    (void*) b.jobs, b.scratch})
        if (p) cudaFree(p);
    b = PromptDev{};
}
// the rows a read can hold (0: too few); no allocation
int prompt_rows_fit(Dev& d, std::string& err) {
    size_t fr = 0, tot = 0;
    if (cudaMemGetInfo(&fr, &tot) != cudaSuccess) return 0;
    const size_t scratch = ne::prompt_scratch_bytes(d.s, kPromptMaxJobs);
    if (!scratch || fr < kPromptReserve + scratch + (16ull << 20)) return 0;
    const size_t budget = std::min(kPromptBudget, fr - kPromptReserve) - scratch - (8ull << 20);
    const size_t rows = budget / (kPromptRowBytes + 64) / 16 * 16;
    (void) err;
    return rows >= 64 ? (int) std::min<size_t>(rows, 60000) : 0;
}
bool prompt_dev_alloc(PromptDev& b, Dev& d, int rows, std::string& err) {
    const size_t R = (size_t) rows, D_ = (size_t) g.dim;
    auto A = [&](auto*& p, size_t bytes) {
        void* q = nullptr;
        if (cudaMalloc(&q, bytes) != cudaSuccess) { cudaGetLastError(); err = "prompt read: device buffers"; return false; }
        p = (std::remove_reference_t<decltype(p)>) q; return true;
    };
    b.rows = rows;
    b.scratch_bytes = ne::prompt_scratch_bytes(d.s, kPromptMaxJobs);
    if (!A(b.h, R * D_ * 4) || !A(b.sh, R * D_ * 4) || !A(b.x16, R * D_ * 2) || !A(b.part, R * 4 * D_ * 2) ||
        !A(b.pos, R * 4) || !A(b.pw, R * 16) || !A(b.served, R * 4) ||
        !A(b.jobs, (size_t) ph.max_jobs * sizeof(ne::PromptJob)) || !A(b.scratch, b.scratch_bytes)) {
        prompt_dev_free(b);
        return false;
    }
    return true;
}
bool prompt_events_ready(Dev& d, int frames, std::string& err) {
    if (prompt_ev.empty()) {
        prompt_ev.resize((size_t) g.n_layers);
        for (auto& x : prompt_ev) for (auto& e : x.e) CK(cudaEventCreate(&e));
        CK(cudaEventCreateWithFlags(&prompt_start_ev, cudaEventDisableTiming));
    }
    const int groups = (ph.rows + 15) / 16;
    while ((int) prompt_group_ev.size() < groups) {
        cudaEvent_t e; CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming)); prompt_group_ev.push_back(e);
    }
    while ((int) prompt_landed.size() < frames) {
        cudaEvent_t a, b2;
        CK(cudaEventCreateWithFlags(&a, cudaEventDisableTiming)); CK(cudaEventCreateWithFlags(&b2, cudaEventDisableTiming));
        prompt_landed.push_back(a); prompt_release.push_back(b2);
    }
    (void) d;
    return true;
}

// ---- one layer for the rows [0, rows) of the read's buffers (positions pos0..)
struct PromptRing {
    std::vector<int> frame;              // ring slot -> staging frame
    std::vector<uint8_t> released;       // a release event was recorded for the slot's last reader
    std::vector<long long> last_key;     // the slot's frame's last upload
};
bool prompt_upload(Layer& l, Dev& d, int li, int e, int i, PromptRing& ring, PromptLayerEvents& pe, std::string& err) {
    const int F = (int) ring.frame.size(), slot = i % F, frame = ring.frame[(size_t) slot];
    if (ring.released[(size_t) slot]) CK(cudaStreamWaitEvent(d.cs, prompt_release[(size_t) slot], 0));
    else CK(cudaStreamWaitEvent(d.cs, prompt_start_ev, 0));
    if (!pe.link) { CK(cudaEventRecord(pe.e[5], d.cs)); pe.link = true; }
    CK(cudaMemcpyAsync(staging_address(d, frame), l.host_blob[(size_t) e], l.blob, cudaMemcpyHostToDevice, d.cs));
    CK(cudaEventRecord(prompt_landed[(size_t) slot], d.cs));
    ring.last_key[(size_t) slot] = (long long) li * 1024 + e;
    ++pe.uploads; ++st.staged;
    return true;
}
bool prompt_layer(Layer& l, Dev& d, int li, int rows, int pos0, PromptDev& b, PromptRing& ring, std::string& err) {
    capture::set_layer(li);
    const int G = (rows + 15) / 16, D_ = g.dim;
    PromptLayerEvents& pe = prompt_ev[(size_t) li];
    pe.link = false; pe.used = l.moe; pe.res_jobs = pe.uploads = pe.helper_cols = pe.helper_passes = 0; pe.groups = G;
    pe.helper_ms = pe.route_wait_ms = 0;
    const double host0 = now();
    float* const h0 = d.h;
    int* const p0 = d.pos;
    if (l.moe) CK(cudaEventRecord(pe.e[0], d.s));
    for (int gi = 0; gi < G; ++gi) {
        const int r0 = gi * 16, n = std::min(16, rows - r0);
        d.h = b.h + (size_t) r0 * D_;
        d.pos = b.pos + r0;
        prompt_reading = true;
        const bool ok = window_pre(l, d, 0, n, err, true);
        prompt_reading = false;
        d.h = h0; d.pos = p0;
        if (!ok) return false;
        if (!l.moe) continue;
        CK(cudaMemcpyAsync(ph.top + (size_t) r0 * 4, d.top, (size_t) n * 16, cudaMemcpyDeviceToHost, d.s));
        CK(cudaMemcpyAsync(ph.w + (size_t) r0 * 4, d.w, (size_t) n * 16, cudaMemcpyDeviceToHost, d.s));
        CK(cudaMemcpyAsync(ph.orig + (size_t) r0 * 4, d.original_d, (size_t) n * 16, cudaMemcpyDeviceToHost, d.s));
        CK(cudaMemcpyAsync(b.sh + (size_t) r0 * D_, d.sh, (size_t) n * D_ * 4, cudaMemcpyDeviceToDevice, d.s));
        if (!ne::prompt_rows_bf16(d.xn, n, D_, b.x16 + (size_t) r0 * D_, d.s)) { err = "prompt read: BF16 rows"; return false; }
        CK(cudaMemcpyAsync(ph.x16 + (size_t) r0 * D_, b.x16 + (size_t) r0 * D_, (size_t) n * D_ * 2, cudaMemcpyDeviceToHost, d.s));
        CK(cudaEventRecord(prompt_group_ev[(size_t) gi], d.s));
    }
    if (!l.moe) {
        if (prompt_hash_file && !prompt_hash_rows('h', li, b.h, rows, pos0, d.s, err)) return false;
        pe.host_ms = 1e3 * (now() - host0);
        return true;
    }
    CK(cudaEventRecord(pe.e[1], d.s));
    // the host: each group's routing as it lands; the 3090 fed as groups arrive; the uploads begun for the first host
    // experts seen; when every row is routed, the 5090's jobs enqueued; then the 3090's remaining passes and the combine
    const int F = (int) ring.frame.size();
    std::array<std::vector<int>, 128> ent;     // expert -> entries (row * 4 + pick), rows ascending
    for (auto& v : ent) v.clear();
    std::array<uint8_t, 128> seen{};
    std::vector<int> host_order;
    std::vector<std::vector<NativeHelper::Job>> hjobs((size_t) G);
    int routed = 0, next_submit = 0, inflight_group = -1, prefetched = 0;
    uint64_t inflight = 0;
    bool phase_done = false;
    std::vector<float> result((size_t) 16 * D_);
    std::vector<int> pos_list(16);
    double helper_first = 0, helper_last = 0;
    const double route_begin = now(), deadline = route_begin + 120.0;
    auto plan_group = [&](int gi) -> bool {
        const int r0 = gi * 16, n = std::min(16, rows - r0);
        // (only now: this group's routing landed after the previous layer's combine, which read the flags)
        std::fill_n(ph.flag + r0, (size_t) n, 0);
        const int* top = ph.top + (size_t) r0 * 4;
        if (!routes_ok(top, n * 4, (size_t) li, err)) return false;
        prompt_route_counts(top, ph.orig + (size_t) r0 * 4, n);
        if (prompt_hash_file) {
            for (int r = 0; r < n; ++r) pos_list[(size_t) r] = pos0 + r0 + r;
            prompt_hash_routes(li, top, ph.orig + (size_t) r0 * 4, n, pos_list.data());
        }
        for (int i = 0; i < n * 4; ++i) {
            const int o = ph.orig[(size_t) r0 * 4 + i];
            if (o >= 0 && o < g.n_exp) ++original_prof[(size_t) li * g.n_exp + o];
        }
        for (int t = 0; t < n; ++t)
            for (int i = 0; i < 4; ++i) {
                const int ent_i = (r0 + t) * 4 + i, x = top[t * 4 + i];
                const float wt = ph.w[(size_t) ent_i] * g.scale;
                ++prof[(size_t) li * g.n_exp + x];
                ph.pw[(size_t) ent_i] = wt;
                if (helper_owns(l, x)) {
                    ph.served[(size_t) ent_i] = 0;
                    auto& jobs = hjobs[(size_t) gi];
                    auto it = std::find_if(jobs.begin(), jobs.end(), [&](const NativeHelper::Job& j) {
                        return j.expert == x && j.columns < NativeHelper::kMaxColumns;
                    });
                    if (it == jobs.end()) { NativeHelper::Job j; j.layer = li; j.expert = x; jobs.push_back(j); it = jobs.end() - 1; }
                    const int c = it->columns++; it->token[c] = t; it->weight[c] = wt;
                    ++st.coop_helper_columns; ++pe.helper_cols;
                    continue;
                }
                ph.served[(size_t) ent_i] = 1;
                ent[(size_t) x].push_back(ent_i);
                if (l.res_slot[(size_t) x] >= 0) continue;
                if (!pooled(l, x)) { err = "prompt read: host expert without a pinned record"; return false; }
                if (!seen[(size_t) x]) {
                    seen[(size_t) x] = 1;
                    host_order.push_back(x);
                    if (prefetched < F) {
                        if (!prompt_upload(l, d, li, x, prefetched, ring, pe, err)) return false;
                        ++prefetched;
                    }
                }
            }
        st.profiled += (long) n * 4;
        return true;
    };
    auto expert_phase = [&]() -> bool {
        // upload order: the first F seen as they came, then by columns, most first
        std::stable_sort(host_order.begin() + prefetched, host_order.end(), [&](int a2, int b2) {
            return ent[(size_t) a2].size() > ent[(size_t) b2].size();
        });
        int nj = 0;
        auto add_jobs = [&](int e, const uint8_t* blob) -> bool {
            const auto& v = ent[(size_t) e];
            for (size_t s0 = 0; s0 < v.size(); s0 += 8) {
                if (nj >= ph.max_jobs) { err = "prompt read: job table full"; return false; }
                ne::PromptJob& j = ph.jobs[nj++];
                j.blob = blob; j.n = (int) std::min<size_t>(8, v.size() - s0);
                for (int c = 0; c < 8; ++c) j.entry[c] = c < j.n ? v[s0 + (size_t) c] : 0;
            }
            return true;
        };
        for (int e = 0; e < g.n_exp; ++e)
            if (!ent[(size_t) e].empty() && l.res_slot[(size_t) e] >= 0 && !add_jobs(e, resident_address(l, l.res_slot[(size_t) e])))
                return false;
        const int res_jobs = nj;
        std::vector<int> first((size_t) host_order.size() + 1);
        for (size_t i = 0; i < host_order.size(); ++i) {
            first[i] = nj;
            const int frame = ring.frame[(size_t) ((int) i % F)];
            if (!add_jobs(host_order[i], staging_address(d, frame))) return false;
        }
        first[host_order.size()] = nj;
        CK(cudaMemcpyAsync(b.jobs, ph.jobs, (size_t) nj * sizeof(ne::PromptJob), cudaMemcpyHostToDevice, d.s));
        CK(cudaMemcpyAsync(b.pw, ph.pw, (size_t) rows * 16, cudaMemcpyHostToDevice, d.s));
        CK(cudaMemcpyAsync(b.served, ph.served, (size_t) rows * 4, cudaMemcpyHostToDevice, d.s));
        CK(cudaEventRecord(pe.e[2], d.s));
        if (!ne::prompt_run(b.jobs, 0, res_jobs, b.x16, b.part, b.scratch, b.scratch_bytes, kPromptMaxJobs, d.s)) {
            err = "prompt read: resident experts"; return false;
        }
        CK(cudaEventRecord(pe.e[3], d.s));
        pe.res_jobs = res_jobs;
        const int U = (int) host_order.size();
        for (int i = 0; i < U; ++i) {
            const int slot = i % F;
            CK(cudaStreamWaitEvent(d.s, prompt_landed[(size_t) slot], 0));
            if (!ne::prompt_run(b.jobs, first[(size_t) i], first[(size_t) i + 1] - first[(size_t) i], b.x16, b.part,
                                b.scratch, b.scratch_bytes, kPromptMaxJobs, d.s)) {
                err = "prompt read: staged experts"; return false;
            }
            CK(cudaEventRecord(prompt_release[(size_t) slot], d.s));
            ring.released[(size_t) slot] = 1;
            if (i + F < U && i + F >= prefetched && !prompt_upload(l, d, li, host_order[(size_t) (i + F)], i + F, ring, pe, err))
                return false;
        }
        if (pe.link) CK(cudaEventRecord(pe.e[6], d.cs));
        CK(cudaEventRecord(pe.e[4], d.s));
        st.prompt_res_jobs += res_jobs; st.prompt_jobs += nj; st.prompt_uploads += U;
        return true;
    };
    auto helper_collect = [&]() -> bool {
        const int gi = inflight_group, r0 = gi * 16, n = std::min(16, rows - r0);
        if (!helper->collect(inflight, result.data(), result.size(), err)) { native_cpu_failed = true; return false; }
        helper_last = now();
        for (const auto& j : hjobs[(size_t) gi])
            for (int c = 0; c < j.columns; ++c) ph.flag[r0 + j.token[c]] = 1;
        for (int t = 0; t < n; ++t) {
            if (!ph.flag[r0 + t]) continue;
            float* dst = ph.helper + (size_t) (r0 + t) * D_;
            const float* src = result.data() + (size_t) t * D_;
            for (int k2 = 0; k2 < D_; ++k2) { float v = 0.0f; v += src[k2]; dst[k2] = v; }
        }
        inflight = 0; inflight_group = -1;
        return true;
    };
    auto helper_submit = [&]() -> bool {
        while (next_submit < routed && hjobs[(size_t) next_submit].empty()) ++next_submit;
        if (next_submit >= routed) return true;
        const int gi = next_submit++, r0 = gi * 16, n = std::min(16, rows - r0);
        if (!helper_first) helper_first = now();
        inflight = helper->submit(n, ph.x16 + (size_t) r0 * D_, hjobs[(size_t) gi], err);
        if (!inflight) return false;
        inflight_group = gi;
        ++pe.helper_passes; st.helper_experts += (long) hjobs[(size_t) gi].size();
        return true;
    };
    for (;;) {
        bool progress = false;
        if (routed < G) {
            const cudaError_t q = cudaEventQuery(prompt_group_ev[(size_t) routed]);
            if (q == cudaSuccess) {
                if (!plan_group(routed)) return false;
                ++routed; progress = true;
                if (routed == G) {
                    pe.route_wait_ms = 1e3 * (now() - route_begin);
                    if (!expert_phase()) return false;
                    phase_done = true;
                }
            } else if (q != cudaErrorNotReady) { err = std::string("prompt read: routing event: ") + cudaGetErrorString(q); return false; }
        }
        if (inflight && helper->done(inflight)) { if (!helper_collect()) return false; progress = true; }
        if (!inflight) {
            const int before = next_submit;
            if (!helper_submit()) return false;
            progress |= next_submit != before;
        }
        if (phase_done && !inflight) {
            while (next_submit < G && hjobs[(size_t) next_submit].empty()) ++next_submit;
            if (next_submit >= G) break;
        }
        if (!progress) {
            if (now() > deadline) { err = "prompt read: layer " + std::to_string(li) + " made no progress for 120 s"; return false; }
            std::this_thread::yield();
        }
    }
    if (helper_first) pe.helper_ms = 1e3 * (helper_last - helper_first);
    st.prompt_helper_cols += pe.helper_cols; st.prompt_helper_passes += pe.helper_passes;
    CK(cudaEventRecord(pe.e[7], d.s));
    if (!ne::prompt_combine(rows, 4, D_, b.sh, b.part, b.pw, b.served, ph.helper, ph.flag, b.h, d.s)) {
        err = "prompt read: combine"; return false;
    }
    CK(cudaEventRecord(pe.e[8], d.s));
    if (prompt_hash_file && !prompt_hash_rows('h', li, b.h, rows, pos0, d.s, err)) return false;
    pe.host_ms = 1e3 * (now() - host0);
    return true;
}
// per-layer times of one read (after a synchronize): stats, and MM3_PROMPT_TRACE lines
void prompt_collect_times(int rows) {
    std::FILE* tf = nullptr;
    if (const char* tp = ::mm3::env("MM3_PROMPT_TRACE")) tf = std::fopen(tp, "a");
    auto ms = [](cudaEvent_t a, cudaEvent_t b2) { float x = 0; return cudaEventElapsedTime(&x, a, b2) == cudaSuccess ? (double) x : -1.0; };
    for (int li = 0; li < g.n_layers; ++li) {
        PromptLayerEvents& pe = prompt_ev[(size_t) li];
        if (!pe.used) continue;
        const double dense = ms(pe.e[0], pe.e[1]), res = ms(pe.e[2], pe.e[3]), experts = ms(pe.e[1], pe.e[4]);
        const double comb = ms(pe.e[7], pe.e[8]), wait_helper = ms(pe.e[4], pe.e[7]), layer = ms(pe.e[0], pe.e[8]);
        const double link = pe.link ? ms(pe.e[5], pe.e[6]) : 0, link_end = pe.link ? ms(pe.e[1], pe.e[6]) : 0;
        st.prompt_dense_ms += dense; st.prompt_dense_groups += pe.groups;
        st.prompt_res_job_ms += res; st.prompt_expert_phase_ms += experts; st.prompt_combine_ms += comb;
        st.prompt_link_ms += link; st.prompt_helper_ms += pe.helper_ms; st.prompt_helper_wait_ms += wait_helper;
        st.prompt_layer_ms += layer;
        if (tf)
            std::fprintf(tf, "{\"layer\":%d,\"rows\":%d,\"groups\":%ld,\"dense_ms\":%.4f,\"res_jobs\":%ld,\"res_ms\":%.4f,"
                         "\"uploads\":%ld,\"link_ms\":%.4f,\"link_end_ms\":%.4f,\"experts_end_ms\":%.4f,\"helper_cols\":%ld,"
                         "\"helper_passes\":%ld,\"helper_host_ms\":%.4f,\"route_wait_host_ms\":%.4f,\"wait_after_experts_ms\":%.4f,"
                         "\"combine_ms\":%.4f,\"layer_ms\":%.4f,\"host_ms\":%.4f}\n",
                         li, rows, pe.groups, dense, pe.res_jobs, res, pe.uploads, link, link_end, experts, pe.helper_cols,
                         pe.helper_passes, pe.helper_ms, pe.route_wait_ms, wait_helper, comb, layer, pe.host_ms);
        pe.used = false;
    }
    if (tf) std::fclose(tf);
}

// fell_back: nothing was done (no memory for the read's buffers); the caller reads the prompt in chunks
bool prompt_read(const int* ids, int n, int pos0, int want, float* logits, bool& fell_back, std::string& err) {
    const double t0 = now();
    fell_back = false;
    Dev& d = D[0];
    CK(cudaSetDevice(d.id));
    const int fit = prompt_rows_fit(d, err);
    if (!fit) { ++st.prompt_fallbacks; fell_back = true; return true; }
    const int parts = (n + fit - 1) / fit;
    const int S = std::min(fit, ((n + parts - 1) / parts + 15) / 16 * 16);
    if (!prompt_host_grow(S, err)) return false;
    PromptDev b;
    if (!prompt_dev_alloc(b, d, S, err)) { ++st.prompt_fallbacks; err.clear(); fell_back = true; return true; }
    struct Free { Impl* m; PromptDev& b; ~Free() { cudaDeviceSynchronize(); m->prompt_dev_free(b); } } free_guard{this, b};
    // the frames: every one not reserved, pending or holding an early copy
    PromptRing ring;
    for (int s = 0; s < d.n_slots; ++s)
        if (!staging_reserved(d, s) && d.slot_pend[(size_t) s] < 0 && !slot_early_busy(d, s)) ring.frame.push_back(s);
    if (!prompt_events_ready(d, (int) ring.frame.size(), err)) return false;
    ring.released.assign(ring.frame.size(), 0);
    ring.last_key.assign(ring.frame.size(), -1);
    // chunk_'s boundary work at a forward's start
    hf_demand_active = false;
    mm3::DemandBoundary demand_boundary{hf_demand_poisoned, hf_demand_active};
    exchange_forward_begin();
    ++fwd; capture::set_forward(fwd); capture::set_layer(-1); capture::light::set_forward(fwd);
    struct CaptureLayerReset { ~CaptureLayerReset() { capture::set_layer(-1); } } capture_layer_reset;
    if (!reset_hf_route_errors(err)) return false;
    if (!refresh_slots(d, err) || !adaptive_publish(d, err) || !helper_live_progress(d, err)) return false;
    for (int f : ring.frame) {
        if (native) prefetch_forget(d, f);
        d.slot_owner[(size_t) f] = -1;
    }
    cur_pos = pos0 + n - 1;
    prompt_hash_begin(pos0);
    CK(cudaEventRecord(prompt_start_ev, d.s));
    ++st.prompt_reads; st.prompt_rows += n;
    int rows_last = 0;
    for (int a = 0; a < n; a += S) {
        const int rows = std::min(S, n - a);
        if (!embed(ids + a, rows, ph.emb, err)) return false;
        std::vector<int> pos((size_t) rows);
        for (int r = 0; r < rows; ++r) pos[(size_t) r] = pos0 + a + r;
        CK(cudaMemcpyAsync(b.pos, pos.data(), (size_t) rows * 4, cudaMemcpyHostToDevice, d.s));
        CK(cudaMemcpyAsync(b.h, ph.emb, (size_t) rows * g.dim * 4, cudaMemcpyHostToDevice, d.s));
        CK(cudaStreamSynchronize(d.s));   // (pos is pageable)
        for (int li = 0; li < g.n_layers; ++li) {
            if (!prompt_layer(L[(size_t) li], d, li, rows, pos0 + a, b, ring, err)) return false;
            if (draftm && draft_running && tap_of[(size_t) li] >= 0)
                CK(cudaMemcpyAsync(ph.taps + (size_t) tap_of[(size_t) li] * S * g.dim, b.h, (size_t) rows * g.dim * 4,
                                   cudaMemcpyDeviceToHost, d.s));
        }
        if (draftm && draft_running) {
            CK(cudaSetDevice(D[(size_t) head_dev].id));
            for (int r0 = 0; r0 < rows; r0 += 16) {
                const int nr = std::min(16, rows - r0);
                if (prompt_hash_file && !prompt_hash_taps(ph.taps + (size_t) r0 * g.dim, nr, pos0 + a + r0, (size_t) S, d.s, err))
                    return false;
                if (!draftm->context(ph.taps + (size_t) r0 * g.dim, nr, pos0 + a + r0, err, (size_t) S)) return false;
            }
            pmark(D[(size_t) head_dev], kPDraft);
        }
        CK(cudaStreamSynchronize(d.s));
        prompt_collect_times(rows);
        rows_last = rows;
    }
    // the head: chunk_'s fast head on the last `want` rows
    if (want > 0) {
        if (want > rows_last) { err = "prompt read: head rows"; return false; }
        Dev& hd = D[(size_t) head_dev];
        const float* last = b.h + (size_t) (rows_last - want) * g.dim;
        k::rms(last, norm1, want, g.dim, g.eps, hd.xn, hd.xn16, hd.s);
        const int cols = std::max(1, x3::mv_cols(head));
        for (int c0 = 0; c0 < want; c0 += cols) {
            x3::MvJob j;
            j.m = &head;
            j.xf = hd.xn + (size_t) c0 * g.dim;
            j.y = hd.lg + (size_t) c0 * g.vocab;
            j.ncol = std::min(cols, want - c0);
            j.xf_cs = (size_t) g.dim;
            j.y_cs = (size_t) g.vocab;
            if (!x3::mv(&j, 1, hd.s)) { err = "mv head"; return false; }
        }
        if (nuc_want) {
            pmark(hd, kPHead);
            k::topk_rows(hd.lg, want, g.vocab, k::kCandCap, nullptr, 0, hd.nuc_c, hd.nuc_topk, hd.s);
            k::row_stats(hd.lg, want, g.vocab, hd.nuc_s, hd.nuc_scratch, hd.s);
            CK(cudaStreamSynchronize(hd.s));
            nuc_rows = want;
        } else {
            pmark(hd, kPHead);
            CK(cudaMemcpyAsync(hd.lg_h, hd.lg, (size_t) want * g.vocab * 4, cudaMemcpyDeviceToHost, hd.s));
            CK(cudaStreamSynchronize(hd.s));
            if (logits) std::memcpy(logits, hd.lg_h, (size_t) want * g.vocab * 4);
            nuc_rows = 0;
        }
    }
    // the frames: completed copies, owned by the last expert each received, as a demand copy leaves them
    CK(cudaStreamSynchronize(d.cs));
    CK(cudaStreamSynchronize(d.s));
    for (size_t s = 0; s < ring.frame.size(); ++s) {
        const int f = ring.frame[s];
        d.slot_owner[(size_t) f] = ring.last_key[s];
        d.slot_foreground[(size_t) f] = 0;
        d.slot_used[(size_t) f] = fwd;
    }
    // chunk_'s boundary work at a forward's end
    for (Dev& dd : D) if (!queue_hf_demand(dd, dd.hf_snapshot_h, err) || !queue_hf_route_errors(dd, err)) return false;
    for (Dev& dd : D) {
        CK(cudaSetDevice(dd.id));
        CK(cudaStreamSynchronize(dd.s));
        if (!consume_hf_demand(dd, dd.hf_snapshot_h, fwd, err) || !check_hf_route_errors(dd, err)) return false;
    }
    demand_boundary.complete = true;
    if (!trace_gpu_finished(err)) return false;
    if (const int e = x3::launch_errors()) { err = "EXL3 launch errors: " + std::to_string(e); return false; }
    for (Dev& dd : D) {
        CK(cudaSetDevice(dd.id));
        if (primary_adaptive && !adaptive_policy.legacy && (!retire_foreground_slots(dd, err) || !adaptive_publish(dd, err))) return false;
        if (!helper_live_progress(dd, err) || !helper_live_issue(dd, err)) return false;
        if (!adaptive_issue(dd, err)) return false;
        if (!dd.promo_q.empty() && !issue_promotions(dd, err)) return false;
    }
    if (!exchange_forward_end(err)) return false;
    if (prompt_hash_file) std::fflush(prompt_hash_file);
    st.prompt_wall_ms += 1e3 * (now() - t0);
    return true;
}
