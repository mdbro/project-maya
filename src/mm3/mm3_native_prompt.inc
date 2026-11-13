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
static constexpr size_t kPromptRowBytes = 111152;  // h F32, shared out F32, input BF16, four BF16 partials, router
                                                   // logits F32 [128], the second pass's top / w / original [4] each
int prompt_layerwise_mode = 0;
int prompt_check_mode = 0;       // 1: residents-only prompt mask, every host expert staged; 2: as 1 with the two-pass read's
                                 // uploaded sets (MM3_PROMPT_U) also ready, the reference for prompt_two_pass
int prompt_two_pass_mode = 0;    // 1: route every row again against the residents and the layer's uploaded set
int prompt_helper_pass_mode = 16; // the most tokens a 3090 pass of the read carries (16: one group a pass; up to the helper's
                                  // pass capacity, MM3_HELPER_PASS_TOKENS at load): the routed groups waiting go together
std::FILE* prompt_u_out = nullptr;   // (MM3_PROMPT_U, two-pass reads) each layer's uploaded set, for prompt_check 2
std::vector<std::array<uint32_t, 4>> prompt_u_bits;   // (prompt_check 2) per layer, from MM3_PROMPT_U
float* prompt_logit_h = nullptr; // (MM3_PROMPT_MASS) pinned [kWin][128]: a chunked prompt pass's router logits
bool prompt_reading = false;     // window_pre / route_layer: a group of the layer-by-layer read
int prompt_mask_layer = -1;      // route_ready_host: the layer whose mask the pinned buffer holds during the read
long long prompt_hash_counter = 0;
long long prompt_hash_next = -1;  // the position after the last hashed row: a read that does not start there starts a file
std::FILE* prompt_hash_file = nullptr;
bool prompt_mass_now = false;     // (MM3_PROMPT_MASS) this read counts router mass: set at each read's start
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
    float* logits = nullptr;        // [rows][128] (MM3_PROMPT_MASS: the router mass kept)
    resident::Ready* ready2 = nullptr;   // the second pass's mask
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
    float* logits = nullptr;        // [rows][128] the router's logits
    int *top2 = nullptr, *orig2 = nullptr;   // [rows][4] the second pass
    float* w2 = nullptr;
    resident::Ready* ready2 = nullptr;
};
struct PromptLayerEvents { cudaEvent_t e[9] = {}; bool link = false, used = false; long res_jobs = 0, uploads = 0, groups = 0;
                           long helper_cols = 0, helper_passes = 0, kept = 0; double helper_ms = 0, host_ms = 0, route_wait_ms = 0;
                           double enqueue_ms = 0, first_route_ms = -1, pass2_ms = 0;
                           double host0 = 0, first_upload_host_ms = -1; };   // (the host's clock: layer start; first upload issued)
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
    if (lab.on) return "route_lab";   // (the routing lab's rules and records run in the chunked path's window_post)
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
// (the diagnostics' switches, read once: not at each read)
static const ::mm3::EnvValue& prompt_env(int which) {
    static const ::mm3::EnvValue hash = ::mm3::env_value("MM3_PROMPT_HASH"), mass = ::mm3::env_value("MM3_PROMPT_MASS"),
                                 u = ::mm3::env_value("MM3_PROMPT_U"), trace = ::mm3::env_value("MM3_PROMPT_TRACE");
    return which == 0 ? hash : which == 1 ? mass : which == 2 ? u : trace;
}
bool prompt_hash_on() const { return prompt_env(0).set; }
// a file per read: a read from position 0, or one that does not continue the last hashed row (a conversation's next
// turn reads from inside the last one), starts the next file; the chunks of one read continue it
void prompt_hash_begin(int pos0) {
    if (!prompt_hash_on() || (prompt_hash_file && pos0 != 0 && pos0 == prompt_hash_next)) return;
    if (prompt_hash_file) std::fclose(prompt_hash_file);
    prompt_hash_next = pos0;
    const std::string path = prompt_env(0).value + "/prompt-" + std::to_string(++prompt_hash_counter) + ".txt";
    prompt_hash_file = std::fopen(path.c_str(), "w");
}
// rows F32 [n][dim] on the device at positions pos0.. : one line each
bool prompt_hash_rows(char kind, int li, const float* rows_dev, int n, int pos0, cudaStream_t s, std::string& err) {
    if (!prompt_hash_file || n <= 0) return true;
    if (kind == 'h') prompt_hash_next = (long long) pos0 + n;
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
// MM3_PROMPT_MASS=1 counts every read; MM3_PROMPT_MASS=<path> only reads that start while the path exists (a campaign
// turns it on for some requests of a load)
void prompt_mass_refresh() {
    const char* v = prompt_env(1).c_str();   // (the path, read once; whether it exists, at each read)
    std::FILE* f = v && std::strcmp(v, "1") != 0 ? std::fopen(v, "r") : nullptr;
    prompt_mass_now = v && (std::strcmp(v, "1") == 0 || f);
    if (f) std::fclose(f);
}
bool prompt_mass_on() const { return prompt_mass_now; }
// router mass kept (as the re-routed replay counts it): the sigmoid of the router's logit summed over executed picks and
// over the router's own top four
void prompt_mass(const float* logits, const int* top, const int* orig, int n) {
    for (int r = 0; r < n; ++r)
        for (int i = 0; i < 4; ++i) {
            const float* lr = logits + (size_t) r * 128;
            st.prompt_mass_executed += 1.0 / (1.0 + std::exp(-(double) lr[top[r * 4 + i]]));
            st.prompt_mass_original += 1.0 / (1.0 + std::exp(-(double) lr[orig[r * 4 + i]]));
        }
}
// (prompt_check 2) the uploaded sets a two-pass read wrote to MM3_PROMPT_U: lines "layer e e e ..."
void prompt_u_load() {
    prompt_u_bits.assign((size_t) g.n_layers, std::array<uint32_t, 4>{});
    const char* path = prompt_env(2).c_str();
    std::FILE* f = path ? std::fopen(path, "r") : nullptr;
    if (!f) return;
    char line[4096];
    while (std::fgets(line, sizeof line, f)) {
        char* q = line; long li = std::strtol(q, &q, 10);
        if (li < 0 || li >= g.n_layers) continue;
        for (;;) {
            char* e0 = q; long e = std::strtol(q, &q, 10);
            if (q == e0) break;
            if (e >= 0 && e < 128) prompt_u_bits[(size_t) li][(size_t) e / 32] |= 1u << (e % 32);
        }
    }
    std::fclose(f);
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
    release(ph.pw); release(ph.served); release(ph.taps); release(ph.jobs); release(ph.logits); release(ph.ready2);
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
    CK(cudaHostAlloc((void**) &ph.logits, R * 128 * 4, cudaHostAllocPortable));
    CK(cudaHostAlloc((void**) &ph.ready2, sizeof(resident::Ready), cudaHostAllocPortable));
    ph.rows = rows; ph.ntaps = nt;
    st.prompt_host_alloc_ms += 1e3 * (now() - t0);
    return true;
}
void prompt_dev_free(PromptDev& b) {
    for (void* p : {(void*) b.h, (void*) b.sh, (void*) b.pw, (void*) b.x16, (void*) b.part, (void*) b.pos, (void*) b.served,
                    (void*) b.jobs, b.scratch, (void*) b.logits, (void*) b.top2, (void*) b.orig2, (void*) b.w2, (void*) b.ready2})
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
        !A(b.jobs, (size_t) ph.max_jobs * sizeof(ne::PromptJob)) || !A(b.scratch, b.scratch_bytes) ||
        !A(b.logits, R * 128 * 4) || !A(b.top2, R * 16) || !A(b.orig2, R * 16) || !A(b.w2, R * 16) ||
        !A(b.ready2, sizeof(resident::Ready))) {
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
// The frames the read leaves for the first generated tokens. Today's chunked read ends with its last chunk's experts of
// the last layers in the frames, and the first tokens find some of them (step 1's timed run: over tokens 0 to 16 the
// layer-by-layer read ran a median 5 more CPU jobs and 4 fewer used early copies, 26 of 32 pairs slower). At the read's
// last two MoE layers the host experts the last group picked most (the last row's first) are uploaded last, and the
// slots holding the second-to-last layer's are kept from the last layer's uploads.
struct PromptKeep { int keep = 0; std::vector<int> pinned; std::vector<int> kept_slots; };
bool prompt_upload(Layer& l, Dev& d, int li, int e, int slot, PromptRing& ring, PromptLayerEvents& pe, std::string& err) {
    const int frame = ring.frame[(size_t) slot];
    if (ring.released[(size_t) slot]) CK(cudaStreamWaitEvent(d.cs, prompt_release[(size_t) slot], 0));
    else CK(cudaStreamWaitEvent(d.cs, prompt_start_ev, 0));
    if (!pe.link) { CK(cudaEventRecord(pe.e[5], d.cs)); pe.link = true; pe.first_upload_host_ms = 1e3 * (now() - pe.host0); }
    CK(cudaMemcpyAsync(staging_address(d, frame), l.host_blob[(size_t) e], l.blob, cudaMemcpyHostToDevice, d.cs));
    CK(cudaEventRecord(prompt_landed[(size_t) slot], d.cs));
    // (a query hands the queued copy to the GPU now: under WSL queued work can wait for a later call)
    (void) cudaStreamQuery(d.cs);
    ring.last_key[(size_t) slot] = (long long) li * 1024 + e;
    ++pe.uploads; ++st.staged;
    return true;
}
bool prompt_layer(Layer& l, Dev& d, int li, int rows, int pos0, PromptDev& b, PromptRing& ring, PromptKeep& keep,
                  std::string& err) {
    capture::set_layer(li);
    const int G = (rows + 15) / 16, D_ = g.dim;
    PromptLayerEvents& pe = prompt_ev[(size_t) li];
    pe.link = false; pe.used = l.moe; pe.res_jobs = pe.uploads = pe.helper_cols = pe.helper_passes = pe.kept = 0; pe.groups = G;
    pe.helper_ms = pe.route_wait_ms = pe.pass2_ms = 0; pe.first_route_ms = -1; pe.first_upload_host_ms = -1;
    const bool two_pass = prompt_two_pass_mode == 1 && l.moe;
    const bool mass = prompt_mass_on() && l.moe;
    const double host0 = now();
    pe.host0 = host0;
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
        if (!l.moe) { (void) cudaStreamQuery(d.s); continue; }
        CK(cudaMemcpyAsync(ph.top + (size_t) r0 * 4, d.top, (size_t) n * 16, cudaMemcpyDeviceToHost, d.s));
        CK(cudaMemcpyAsync(ph.w + (size_t) r0 * 4, d.w, (size_t) n * 16, cudaMemcpyDeviceToHost, d.s));
        CK(cudaMemcpyAsync(ph.orig + (size_t) r0 * 4, d.original_d, (size_t) n * 16, cudaMemcpyDeviceToHost, d.s));
        CK(cudaMemcpyAsync(b.sh + (size_t) r0 * D_, d.sh, (size_t) n * D_ * 4, cudaMemcpyDeviceToDevice, d.s));
        if (two_pass || mass)
            CK(cudaMemcpyAsync(b.logits + (size_t) r0 * 128, d.logit, (size_t) n * 128 * 4, cudaMemcpyDeviceToDevice, d.s));
        if (mass) CK(cudaMemcpyAsync(ph.logits + (size_t) r0 * 128, d.logit, (size_t) n * 128 * 4, cudaMemcpyDeviceToHost, d.s));
        if (!ne::prompt_rows_bf16(d.xn, n, D_, b.x16 + (size_t) r0 * D_, d.s)) { err = "prompt read: BF16 rows"; return false; }
        CK(cudaMemcpyAsync(ph.x16 + (size_t) r0 * D_, b.x16 + (size_t) r0 * D_, (size_t) n * D_ * 2, cudaMemcpyDeviceToHost, d.s));
        CK(cudaEventRecord(prompt_group_ev[(size_t) gi], d.s));
        (void) cudaStreamQuery(d.s);   // (hand the group to the GPU now)
    }
    pe.enqueue_ms = 1e3 * (now() - host0);
    if (!l.moe) {
        if (prompt_hash_file && !prompt_hash_rows('h', li, b.h, rows, pos0, d.s, err)) return false;
        pe.host_ms = 1e3 * (now() - host0);
        return true;
    }
    CK(cudaEventRecord(pe.e[1], d.s));
    // the host: each group's routing as it lands; the 3090 fed as groups arrive (two-pass: after the second pass); the
    // uploads begun for the first host experts seen; when every row is routed, the 5090's jobs enqueued; then the
    // 3090's remaining passes and the combine
    std::vector<int> avail;            // ring slots this layer may use (not kept from the layer before)
    for (int sl = 0; sl < (int) ring.frame.size(); ++sl)
        if (std::find(keep.pinned.begin(), keep.pinned.end(), sl) == keep.pinned.end()) avail.push_back(sl);
    const int F = (int) avail.size();
    std::array<std::vector<int>, 128> ent;     // expert -> entries (row * 4 + pick), rows ascending
    for (auto& v : ent) v.clear();
    std::array<uint8_t, 128> seen{};
    std::array<int, 128> slot_of; slot_of.fill(-1);
    std::vector<int> host_order;
    std::vector<std::vector<NativeHelper::Job>> hjobs((size_t) G);
    int routed = 0, planned = 0, next_submit = 0, inflight_first = -1, inflight_end = -1, prefetched = 0;
    uint64_t inflight = 0;
    // a 3090 pass: the routed groups not yet submitted, up to pass_groups of them, each job from one group (so a row's
    // columns are merged in the order a one-group pass merges them: the helper sorts jobs stably by width)
    const int pass_groups = std::max(1, std::min(prompt_helper_pass_mode, helper->pass_tokens()) / 16);
    std::vector<NativeHelper::Job> pass_jobs;
    bool phase_done = false;
    std::vector<float> result((size_t) 16 * pass_groups * D_);
    std::vector<int> pos_list(16);
    double helper_first = 0, helper_last = 0;
    const double route_begin = now(), deadline = route_begin + 120.0;
    const auto ready_primary = [&](int e) { return l.res_slot[(size_t) e] >= 0; };
    std::array<uint32_t, 4> u_bits{};          // (two-pass) the layer's uploaded set after the first pass
    // first pass, or the only one: the host experts seen, and the first F of them uploaded at once
    auto seen_group = [&](int gi) -> bool {
        const int r0 = gi * 16, n = std::min(16, rows - r0);
        const int* top = ph.top + (size_t) r0 * 4;
        if (!routes_ok(top, n * 4, (size_t) li, err)) return false;
        for (int i = 0; i < n * 4; ++i) {
            const int x = top[i];
            if (helper_owns(l, x) || ready_primary(x)) continue;
            if (!pooled(l, x)) { err = "prompt read: host expert without a pinned record"; return false; }
            u_bits[(size_t) x / 32] |= 1u << (x % 32);
            if (!seen[(size_t) x]) {
                seen[(size_t) x] = 1;
                host_order.push_back(x);
                if (prefetched < F) {
                    slot_of[(size_t) x] = avail[(size_t) prefetched];
                    if (!prompt_upload(l, d, li, x, avail[(size_t) prefetched], ring, pe, err)) return false;
                    ++prefetched;
                }
            }
        }
        return true;
    };
    // the final routing of a group: entries, the 3090's jobs, weights, counts
    auto plan_group = [&](int gi) -> bool {
        const int r0 = gi * 16, n = std::min(16, rows - r0);
        const int* top = ph.top + (size_t) r0 * 4;
        const int* orig = ph.orig + (size_t) r0 * 4;
        if (two_pass && !routes_ok(top, n * 4, (size_t) li, err)) return false;
        // (only now: this group's routing landed after the previous layer's combine, which read the flags)
        std::fill_n(ph.flag + r0, (size_t) n, 0);
        prompt_route_counts(top, orig, n);
        if (mass) prompt_mass(ph.logits + (size_t) r0 * 128, top, orig, n);
        if (prompt_hash_file) {
            for (int r = 0; r < n; ++r) pos_list[(size_t) r] = pos0 + r0 + r;
            prompt_hash_routes(li, top, orig, n, pos_list.data());
        }
        // (observe_routes' counts for these rows, against the mask this routing saw)
        const auto ready_in_mask = [&](int e) {
            return ready_primary(e) || helper_owns(l, e) || (two_pass && ((u_bits[(size_t) e / 32] >> (e % 32)) & 1u));
        };
        bool missed[128]{};
        for (int i = 0; i < n * 4; ++i) {
            const int o = orig[i], e = top[i];
            if (o >= 0 && o < g.n_exp) {
                ++original_prof[(size_t) li * g.n_exp + o];
                st.route_original_misses += !ready_in_mask(o);
            }
            bool selected = false;
            for (int j = 0; j < 4; ++j) selected |= orig[(i / 4) * 4 + j] == e;
            st.route_substitutions += !selected;
            if (ready_primary(e)) ++st.route_primary_hits;
            else if (helper_owns(l, e)) ++st.route_helper_hits;
            else if (!missed[e]) { missed[e] = true; ++st.miss_union; st.miss_union_bytes += l.blob; }
            if (adaptive_main.size() == size_t(g.n_layers * g.n_exp) && ready_primary(e) && adaptive_main[(size_t) li * g.n_exp + e])
                ++st.adaptive_hits;
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
                if (ready_primary(x) || seen[(size_t) x]) continue;
                // (two-pass: the second pass picks no host expert the first did not, by construction; kept in case)
                if (!pooled(l, x)) { err = "prompt read: host expert without a pinned record"; return false; }
                seen[(size_t) x] = 1;
                host_order.push_back(x);
            }
        st.profiled += (long) n * 4;
        ++planned;
        return true;
    };
    // the second pass: every row against the residents and the layer's uploaded set
    auto second_pass = [&]() -> bool {
        const double t0 = now();
        const std::vector<int> top1(ph.top, ph.top + (size_t) rows * 4);   // the first pass's picks, landed
        *ph.ready2 = {};
        for (int e : l.res_ids) ph.ready2->primary[e / 32] |= 1u << (e % 32);
        for (int k = 0; k < 4; ++k) ph.ready2->primary[k] |= u_bits[(size_t) k];
        std::copy(l.helper_ready.begin(), l.helper_ready.end(), ph.ready2->helper);
        CK(cudaMemcpyAsync(b.ready2, ph.ready2, sizeof(resident::Ready), cudaMemcpyHostToDevice, d.s));
        const auto policy = !o.routing ? resident::Policy::Off : o.routing_prior ? resident::Policy::Prior : resident::Policy::Substitute;
        CK(resident::route(b.logits, l.bias, rows, policy, policy == resident::Policy::Prior ? router_bonus(rows) : o.routing_ratio,
                           b.ready2, b.top2, b.w2, b.orig2, nullptr, d.s, o.routing_dampen, nullptr, router_floor(policy, false)));
        CK(cudaMemcpyAsync(ph.top, b.top2, (size_t) rows * 16, cudaMemcpyDeviceToHost, d.s));
        CK(cudaMemcpyAsync(ph.w, b.w2, (size_t) rows * 16, cudaMemcpyDeviceToHost, d.s));
        CK(cudaMemcpyAsync(ph.orig, b.orig2, (size_t) rows * 16, cudaMemcpyDeviceToHost, d.s));
        CK(cudaStreamSynchronize(d.s));
        // the device's top and weights of each group feed nothing else: window_pre left them per group, the combine reads
        // the plan's weights (pw) and partials only
        pe.pass2_ms = 1e3 * (now() - t0);
        // rows the second pass changed, and against the first pass: more picks moved off the router's top four (none by
        // construction), less router mass (possible: the selection ranks by sigmoid plus bias, the mass is the sigmoid)
        for (int r = 0; r < rows; ++r) {
            const int* a1 = top1.data() + (size_t) r * 4;
            const int* a2 = ph.top + (size_t) r * 4;
            const int* o4 = ph.orig + (size_t) r * 4;
            auto in = [](const int* v, int e) { return v[0] == e || v[1] == e || v[2] == e || v[3] == e; };
            bool changed = false; int m1 = 0, m2 = 0;
            for (int i = 0; i < 4; ++i) { changed |= !in(a1, a2[i]); m1 += !in(o4, a1[i]); m2 += !in(o4, a2[i]); }
            st.prompt_pass2_changed += changed; st.prompt_pass2_more_moved += m2 > m1;
            if (mass) {
                const float* lr = ph.logits + (size_t) r * 128;
                double s1 = 0, s2 = 0;
                for (int i = 0; i < 4; ++i) {
                    s1 += 1.0 / (1.0 + std::exp(-(double) lr[a1[i]]));
                    s2 += 1.0 / (1.0 + std::exp(-(double) lr[a2[i]]));
                }
                st.prompt_mass1_executed += s1; st.prompt_pass2_less_mass += s2 < s1 - 1e-9;
            }
        }
        st.prompt_pass2_rows += rows;
        for (int gi = 0; gi < G; ++gi) if (!plan_group(gi)) return false;
        return true;
    };
    auto expert_phase = [&]() -> bool {
        // upload order: the first F seen as they came, then by columns, most first
        std::stable_sort(host_order.begin() + prefetched, host_order.end(), [&](int a2, int b2) {
            return ent[(size_t) a2].size() > ent[(size_t) b2].size();
        });
        // the frames kept for the first generated tokens (the last group's most picked host experts)
        std::vector<int> kept;
        std::vector<uint8_t> pinned_now(ring.frame.size(), 0);
        if (keep.keep > 0 && G > 0) {
            const int r0 = (G - 1) * 16, n = std::min(16, rows - r0);
            std::array<int, 128> score{};
            for (int t = 0; t < n; ++t)
                for (int i = 0; i < 4; ++i) {
                    const int x = ph.top[(size_t) (r0 + t) * 4 + i];
                    if (ready_primary(x) || helper_owns(l, x)) continue;
                    score[(size_t) x] += 1 + (t == n - 1 ? 1000 : 0);
                }
            std::vector<int> cand;
            for (int e = 0; e < g.n_exp; ++e) if (score[(size_t) e]) cand.push_back(e);
            std::stable_sort(cand.begin(), cand.end(), [&](int a2, int b2) { return score[(size_t) a2] > score[(size_t) b2]; });
            const int cap = std::min(keep.keep, std::max(0, F / 2));
            for (int e : cand) if ((int) kept.size() < cap) kept.push_back(e);
            // kept experts already in a frame keep it; the others go to the end of the order
            std::vector<int> late;
            for (int e : kept) {
                if (slot_of[(size_t) e] >= 0) pinned_now[(size_t) slot_of[(size_t) e]] = 1;
                else late.push_back(e);
            }
            std::vector<int> order;
            for (int e : host_order) if (std::find(late.begin(), late.end(), e) == late.end()) order.push_back(e);
            for (auto it = late.rbegin(); it != late.rend(); ++it) order.push_back(*it);
            host_order.swap(order);
        }
        const int U = (int) host_order.size();
        // slot of every upload: the first F as issued; each later one reuses the slot the earliest finished expert frees
        // (not one kept), in the order the 5090's jobs are queued; after[i]: the uploads queued once expert i's jobs are
        std::vector<std::vector<int>> after((size_t) U + 1);
        {
            std::deque<int> free_q;
            for (int sl : avail) {
                bool used = false;
                for (int i = 0; i < prefetched; ++i) used |= slot_of[(size_t) host_order[(size_t) i]] == sl;
                if (!used && !pinned_now[(size_t) sl]) free_q.push_back(sl);
            }
            int next = prefetched;
            while (next < U && !free_q.empty()) { slot_of[(size_t) host_order[(size_t) next]] = free_q.front(); free_q.pop_front(); after[0].push_back(next++); }
            for (int i = 0; i < U; ++i) {
                const int sl = slot_of[(size_t) host_order[(size_t) i]];
                if (!pinned_now[(size_t) sl]) free_q.push_back(sl);
                while (next < U && !free_q.empty()) {
                    slot_of[(size_t) host_order[(size_t) next]] = free_q.front(); free_q.pop_front(); after[(size_t) i + 1].push_back(next++);
                }
            }
            if (next < U) { err = "prompt read: no free staging frame for an upload"; return false; }
        }
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
        std::vector<int> first((size_t) U + 1);
        for (int i = 0; i < U; ++i) {
            first[(size_t) i] = nj;
            if (!add_jobs(host_order[(size_t) i], staging_address(d, ring.frame[(size_t) slot_of[(size_t) host_order[(size_t) i]]])))
                return false;
        }
        first[(size_t) U] = nj;
        CK(cudaMemcpyAsync(b.jobs, ph.jobs, (size_t) nj * sizeof(ne::PromptJob), cudaMemcpyHostToDevice, d.s));
        CK(cudaMemcpyAsync(b.pw, ph.pw, (size_t) rows * 16, cudaMemcpyHostToDevice, d.s));
        CK(cudaMemcpyAsync(b.served, ph.served, (size_t) rows * 4, cudaMemcpyHostToDevice, d.s));
        CK(cudaEventRecord(pe.e[2], d.s));
        if (!ne::prompt_run(b.jobs, 0, res_jobs, b.x16, b.part, b.scratch, b.scratch_bytes, kPromptMaxJobs, d.s)) {
            err = "prompt read: resident experts"; return false;
        }
        CK(cudaEventRecord(pe.e[3], d.s));
        (void) cudaStreamQuery(d.s);
        pe.res_jobs = res_jobs;
        for (int q : after[0]) if (!prompt_upload(l, d, li, host_order[(size_t) q], slot_of[(size_t) host_order[(size_t) q]], ring, pe, err)) return false;
        for (int i = 0; i < U; ++i) {
            const int sl = slot_of[(size_t) host_order[(size_t) i]];
            CK(cudaStreamWaitEvent(d.s, prompt_landed[(size_t) sl], 0));
            if (!ne::prompt_run(b.jobs, first[(size_t) i], first[(size_t) i + 1] - first[(size_t) i], b.x16, b.part,
                                b.scratch, b.scratch_bytes, kPromptMaxJobs, d.s)) {
                err = "prompt read: staged experts"; return false;
            }
            CK(cudaEventRecord(prompt_release[(size_t) sl], d.s));
            ring.released[(size_t) sl] = 1;
            for (int q : after[(size_t) i + 1])
                if (!prompt_upload(l, d, li, host_order[(size_t) q], slot_of[(size_t) host_order[(size_t) q]], ring, pe, err)) return false;
        }
        if (pe.link) CK(cudaEventRecord(pe.e[6], d.cs));
        CK(cudaEventRecord(pe.e[4], d.s));
        (void) cudaStreamQuery(d.s);
        keep.kept_slots.clear();
        for (int e : kept) keep.kept_slots.push_back(slot_of[(size_t) e]);
        pe.kept = (long) kept.size();
        st.prompt_res_jobs += res_jobs; st.prompt_jobs += nj; st.prompt_uploads += U;
        return true;
    };
    auto helper_collect = [&]() -> bool {
        const int r0 = inflight_first * 16, n = std::min(inflight_end * 16, rows) - r0;
        if (!helper->collect(inflight, result.data(), result.size(), err)) { native_cpu_failed = true; return false; }
        helper_last = now();
        for (int gi = inflight_first; gi < inflight_end; ++gi)
            for (const auto& j : hjobs[(size_t) gi])
                for (int c = 0; c < j.columns; ++c) ph.flag[gi * 16 + j.token[c]] = 1;
        for (int t = 0; t < n; ++t) {
            if (!ph.flag[r0 + t]) continue;
            float* dst = ph.helper + (size_t) (r0 + t) * D_;
            const float* src = result.data() + (size_t) t * D_;
            for (int k2 = 0; k2 < D_; ++k2) { float v = 0.0f; v += src[k2]; dst[k2] = v; }
        }
        inflight = 0; inflight_first = inflight_end = -1;
        return true;
    };
    auto helper_submit = [&]() -> bool {
        while (next_submit < planned && hjobs[(size_t) next_submit].empty()) ++next_submit;
        if (next_submit >= planned) return true;
        const int g0 = next_submit;
        int g1 = g0;
        pass_jobs.clear();
        while (g1 < planned && g1 - g0 < pass_groups) {
            for (NativeHelper::Job j : hjobs[(size_t) g1]) {
                for (int c = 0; c < j.columns; ++c) j.token[c] += (g1 - g0) * 16;
                pass_jobs.push_back(j);
            }
            ++g1;
        }
        const int r0 = g0 * 16, n = std::min(g1 * 16, rows) - r0;
        if (!helper_first) helper_first = now();
        inflight = helper->submit(n, ph.x16 + (size_t) r0 * D_, pass_jobs, err);
        if (!inflight) return false;
        inflight_first = g0; inflight_end = g1; next_submit = g1;
        ++pe.helper_passes; st.helper_experts += (long) pass_jobs.size();
        return true;
    };
    for (;;) {
        bool progress = false;
        if (routed < G) {
            const cudaError_t q = cudaEventQuery(prompt_group_ev[(size_t) routed]);
            if (q == cudaSuccess) {
                if (pe.first_route_ms < 0) pe.first_route_ms = 1e3 * (now() - route_begin);
                if (!seen_group(routed)) return false;
                if (!two_pass && !plan_group(routed)) return false;
                ++routed; progress = true;
                if (routed == G) {
                    pe.route_wait_ms = 1e3 * (now() - route_begin);
                    if (two_pass && !second_pass()) return false;
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
    if (two_pass && prompt_u_out) {
        std::fprintf(prompt_u_out, "%d", li);
        for (int e = 0; e < 128; ++e) if ((u_bits[(size_t) e / 32] >> (e % 32)) & 1u) std::fprintf(prompt_u_out, " %d", e);
        std::fputc('\n', prompt_u_out);
    }
    CK(cudaEventRecord(pe.e[7], d.s));
    if (abl_any(d, rows)) {   // (--ablate: the MoE's write into sh, projected, then added - as window_post's ablate())
        if (!ne::prompt_combine_to(rows, 4, D_, b.sh, b.part, b.pw, b.served, ph.helper, ph.flag, b.sh, d.s)) {
            err = "prompt read: combine"; return false;
        }
        ablate(d, b.sh, rows);
        k::add(b.h, b.sh, (size_t) rows * D_, d.s);
    } else if (!ne::prompt_combine(rows, 4, D_, b.sh, b.part, b.pw, b.served, ph.helper, ph.flag, b.h, d.s)) {
        err = "prompt read: combine"; return false;
    }
    CK(cudaEventRecord(pe.e[8], d.s));
    (void) cudaStreamQuery(d.s);
    if (prompt_hash_file && !prompt_hash_rows('h', li, b.h, rows, pos0, d.s, err)) return false;
    pe.host_ms = 1e3 * (now() - host0);
    return true;
}
// per-layer times of one read (after a synchronize): stats, and MM3_PROMPT_TRACE lines
void prompt_collect_times(int rows) {
    std::FILE* tf = nullptr;
    if (const char* tp = prompt_env(3).c_str()) tf = std::fopen(tp, "a");
    auto ms = [](cudaEvent_t a, cudaEvent_t b2) { float x = 0; return cudaEventElapsedTime(&x, a, b2) == cudaSuccess ? (double) x : -1.0; };
    for (int li = 0; li < g.n_layers; ++li) {
        PromptLayerEvents& pe = prompt_ev[(size_t) li];
        if (!pe.used) continue;
        const double dense = ms(pe.e[0], pe.e[1]), res = ms(pe.e[2], pe.e[3]), experts = ms(pe.e[1], pe.e[4]);
        const double comb = ms(pe.e[7], pe.e[8]), wait_helper = ms(pe.e[4], pe.e[7]), layer = ms(pe.e[0], pe.e[8]);
        const double link = pe.link ? ms(pe.e[5], pe.e[6]) : 0, link_end = pe.link ? ms(pe.e[1], pe.e[6]) : 0;
        // (the first upload's start from the layer's start: below the dense time, the link overlapped the dense work)
        const double link_start = pe.link ? ms(pe.e[0], pe.e[5]) : -1;
        st.prompt_dense_ms += dense; st.prompt_dense_groups += pe.groups;
        st.prompt_res_job_ms += res; st.prompt_expert_phase_ms += experts; st.prompt_combine_ms += comb;
        st.prompt_link_ms += link; st.prompt_helper_ms += pe.helper_ms; st.prompt_helper_wait_ms += wait_helper;
        st.prompt_layer_ms += layer; st.prompt_pass2_ms += pe.pass2_ms; st.prompt_kept += pe.kept;
        if (tf)
            std::fprintf(tf, "{\"layer\":%d,\"rows\":%d,\"groups\":%ld,\"dense_ms\":%.4f,\"res_jobs\":%ld,\"res_ms\":%.4f,"
                         "\"uploads\":%ld,\"link_ms\":%.4f,\"link_end_ms\":%.4f,\"experts_end_ms\":%.4f,\"helper_cols\":%ld,"
                         "\"helper_passes\":%ld,\"helper_host_ms\":%.4f,\"route_wait_host_ms\":%.4f,\"wait_after_experts_ms\":%.4f,"
                         "\"combine_ms\":%.4f,\"layer_ms\":%.4f,\"host_ms\":%.4f,\"enqueue_host_ms\":%.4f,"
                         "\"first_route_host_ms\":%.4f,\"pass2_host_ms\":%.4f,\"kept\":%ld,\"link_start_ms\":%.4f,\"first_upload_host_ms\":%.4f,\"two_pass\":%d}\n",
                         li, rows, pe.groups, dense, pe.res_jobs, res, pe.uploads, link, link_end, experts, pe.helper_cols,
                         pe.helper_passes, pe.helper_ms, pe.route_wait_ms, wait_helper, comb, layer, pe.host_ms, pe.enqueue_ms,
                         pe.first_route_ms, pe.pass2_ms, pe.kept, link_start, pe.first_upload_host_ms, prompt_two_pass_mode);
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
    prompt_mass_refresh();
    if (prompt_two_pass_mode == 1) {
        if (prompt_u_out) std::fclose(prompt_u_out);
        const char* up = prompt_env(2).c_str();
        prompt_u_out = up ? std::fopen(up, "w") : nullptr;
    }
    int last_moe = -1, prev_moe = -1;
    for (int li = 0; li < g.n_layers; ++li) if (L[(size_t) li].moe) { prev_moe = last_moe; last_moe = li; }
    CK(cudaEventRecord(prompt_start_ev, d.s));
    ++st.prompt_reads; st.prompt_rows += n;
    int rows_last = 0;
    for (int a = 0; a < n; a += S) {
        const int rows = std::min(S, n - a);
        prompt_mask_layer = -1;   // (the previous part's copies are done: it synchronized)
        if (!embed(ids + a, rows, ph.emb, err)) return false;
        std::vector<int> pos((size_t) rows);
        for (int r = 0; r < rows; ++r) pos[(size_t) r] = pos0 + a + r;
        CK(cudaMemcpyAsync(b.pos, pos.data(), (size_t) rows * 4, cudaMemcpyHostToDevice, d.s));
        CK(cudaMemcpyAsync(b.h, ph.emb, (size_t) rows * g.dim * 4, cudaMemcpyHostToDevice, d.s));
        CK(cudaStreamSynchronize(d.s));   // (pos is pageable)
        const bool final_part = a + rows >= n;
        PromptKeep none, keep_prev, keep_last;
        for (int li = 0; li < g.n_layers; ++li) {
            PromptKeep* kp = &none;
            if (final_part && li == prev_moe) { keep_prev = PromptKeep{}; keep_prev.keep = (int) ring.frame.size() / 2; kp = &keep_prev; }
            else if (final_part && li == last_moe) {
                keep_last = PromptKeep{}; keep_last.pinned = keep_prev.kept_slots;
                keep_last.keep = ((int) ring.frame.size() - (int) keep_prev.kept_slots.size()) / 2; kp = &keep_last;
            } else none = PromptKeep{};
            if (!prompt_layer(L[(size_t) li], d, li, rows, pos0 + a, b, ring, *kp, err)) return false;
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
    prompt_mask_layer = -1;
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
    if (prompt_u_out) { std::fclose(prompt_u_out); prompt_u_out = nullptr; }
    st.prompt_wall_ms += 1e3 * (now() - t0);
    return true;
}
