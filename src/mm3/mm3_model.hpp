// src/mm3/mm3_model.hpp - maya-mm3: MiniMax-M3.1-Flash (EXL3) on the GPUs.
//
// The layers split over the GPUs by their memory, each GPU holding its layers' dense matrices, KV cache (BF16
// [ctx][4][128] K and V) and index-key cache (BF16 [ctx][128]), then as many of its layers' routed experts as fit
// (in VRAM by expert, a blob each); the others are read from the model's files into a GPU's staging slots when a
// token routes to them.  One forward (the prompt path) runs any number of tokens at once, a decode step included.
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "mm3_pack.hpp"
#include "mm3_residency_policy.hpp"

namespace mm3 {

struct VisionGeo;
class TpEngine;

struct HelperWeightTraffic {
    uint64_t h2d_issued_ops=0, h2d_issued_bytes=0, d2h_issued_ops=0, d2h_issued_bytes=0;
    uint64_t replacements_completed=0, copy_failures=0;
};

struct Options {
    std::string pack;
    std::vector<int> gpus;        // empty: every visible GPU
    std::vector<int> layer_split; // the first layer of each GPU's share after the first (empty: by their free memory)
    int max_ctx = 32768;
    int chunk = 1024;             // prompt tokens a forward takes at most
    bool chunk_given = false;     // (--chunk: chosen - the prefill calibration's chunk is not taken)
    // the prompt path's dense GEMMs (mm3_gguf.cuh's set_gemm): -1 the prefill calibration's for this PC (else the
    // BLAS), 0 the BLAS, 1 hipBLASLt (AMD); and the calibration's cache (prefill_tune: its file - empty: the pack's
    // prefill-tune.txt; tune: read it at all)
    int prefill_gemm = -1;
    std::string prefill_tune;
    bool tune = true;
    int staging = 48;             // expert staging slots a GPU (each one expert's blob)
    double reserve_gb = -1;       // VRAM a GPU keeps free beyond the plan (< 0: 1.0, the tensor-parallel path 0.3 -
                                  // it allocates all it needs at the load)
    int resident_cap = -1;        // routed experts kept in VRAM a layer at most (-1: as many as fit)
    std::string profile;          // an expert profile (uint32 [layers][experts] routings): the most used stay in VRAM
    bool pin_pool = true;         // the experts not in VRAM assembled once in pinned host memory (a miss: one DMA)
    bool fast_decode = true;      // one token: the decode GEMVs and the token's fused experts (else the prompt path)
    int cpu_threads = -2;         // the CPU lane's workers besides the caller (-1: no CPU lane, misses are staged;
                                  // -2: this host's, CpuLane::auto_workers - one a physical core, the first the caller's,
                                  // 2 cores left free where there are no SMT siblings); --cpu-threads N: N - 1 of them
    bool dspark = false;          // load the DSpark draft (the pack's dspark/) on the head's GPU
    std::string dflash2;          // (given) RadixArk's DFlash2 draft from this pack dir instead (tools/mm3/pack_dflash2.py;
                                  // step 6), in DSpark's place: same head, taps and calls; implies dspark
    bool vision = true;           // load the vision tower (the pack's vision.*) on gpus[vision_gpu], before the experts
    bool vision_auto = true;      // (vision not asked for by name: a pack without a tower runs text only, no error)
    bool vision_must = false;     // (--images: a tower that does not load is an error; else vision off with a NOTE)
    int vision_gpu = 0;
    bool promote = true;          // the staging slots a cache: an expert the CPU lane keeps computing copied into one
                                  // (from the pinned pool, on a copy stream of its own) while the model runs
    int promote_lag = 2;          // legacy tensor-parallel policy; native promotion publication does not delay
    int tp = 0;                   // > 0: the tensor-parallel path (mm3_tp.hpp) in groups of tp GPUs (1, 2 or 4)
    double cache_gb = 1.5;        // (tp) each GPU's expert cache (promote's slots), taken from its experts in VRAM -
                                  // at least this, more where a layer's experts outside VRAM need more (tight memory)
    int tp_chunk = 1024;          // (tp) a prompt chunk's tokens: its experts not in VRAM are copied in once a chunk
    int tp_ffn_rows = 0;          // (tp) ... and its FFN half run in slices of this many tokens (0: the whole chunk) -
                                  // the same bits; 256 keeps ~37 more experts in VRAM, a prompt reads ~13% slower
    int kv_resident = 0;          // KV streaming (--kv-streaming on: 32768): the positions of each layer's K / V kept in
                                  // VRAM (a ring), all of them in pinned host memory (0, or the context or less: off)
    // the K / V caches' formats (--kv-cache, mm3_kernels.cuh's kKv*: 0 BF16 - the model's dtype, exact - 1 INT8, 2 FP8
    // E4M3 with a power-of-two scale a 32, 3 FP4 E2M1 with an E4M3 scale a 16) and whether their rows are rotated before
    // they are rounded (--kv-rot); the index keys stay BF16 (the blocks each query keeps do not move)
    int kv_k = 1, kv_v = 1;   // (INT8, rows rotated: tools/mm3/kv_gate.py decides whether FP4 joins it)
    bool kv_rot = true;
    // --parallel: conversations at once - a KV cache (and index-key cache) each, the context's room for every one of
    // them (more than 1: no KV streaming, no DSpark draft, the layer split)
    int seqs = 1;
    bool verbose = false;
    int native_stage_prefix = 0; // opt-in dense three-layer 3090 prefix
    bool native_cooperative = true; // false: primary-only native streaming executor
    // Batch cost planner is the implementation baseline; false keeps the
    // historical first-seen allocator as an explicit diagnostic control.
    bool cooperative_stage_reuse = true;
    double cooperative_cpu_scale = 1.0; // scale measured CPU prices; preserve raw calibration
    bool primary_submit_query = false; // one nonblocking status query before CPU service; experimental A/B control
    bool cpu_first = false; // CPU batch starts right after planning, GPU submissions on a submit thread; experimental A/B control
    int early_copy = 0; // at T=1, copy the next layer's predicted HOST expert a layer early: 1 only when this layer queued no demand copy, 2 also behind one; experimental A/B control
    bool early_copy_legacy = false; // the early copy's first form: prediction as four calls, every copy fenced (A/B against the lean form)
    int prep_graph = 0; // at T=1, each MoE layer's preparation (attention, router, shared expert, readbacks) replayed as a CUDA graph: 1 on, 2 verify each replay against the launches (diagnostic); experimental A/B control
    int expert_graph = 0; // each generation window's expert phase on the 5090 (plan copy, grouped kernels, wait, combine, residual add) replayed as a CUDA graph: 1 on, 2 verify each replay against the launches (diagnostic); experimental A/B control
    int dense_split = 0; // 1: the dense MXFP8 mv splits K for outputs up to 512 (attention k / v, the indexer's q), not only up to 256; A/B control
    int native_helper = 1;        // selected 3090 expert helper; -1 for a diagnostic control
    double native_helper_gb = 21.0;
    bool routing = false;
    float routing_ratio = 0.8f;
    bool routing_dampen = false;
    bool routing_prior = false;
    float routing_prior_bonus = 0.05f;
    bool routing_prefetch = false;
    std::string native_primary_arithmetic = "dp4a";
    bool native_primary_arithmetic_given = false;
    std::string native_helper_dispatch = "worker";
};

struct Stats {
    long stage_windows=0, stage_segments=0, stage_rows=0, stage_b_pending=0, stage_b_ready_before_a_end=0;
    long stage_b_prefix_rows=0, stage_b_prefix_discarded_rows=0, stage_b_suffix_skipped_rows=0, stage_b_suffix_completed_rows=0;
    double stage_prefix_b_submit_ms=0, stage_suffix_a_ms=0, stage_b_wait_ms=0;
    long owner_window_calls=0, owner_expert_groups=0, owner_entries=0, owner_passes=0;
    long route_substitutions = 0, route_original_misses = 0;
    long route_primary_hits = 0, route_helper_hits = 0, helper_experts = 0;
    // Native cooperative host-window work. Columns are executed token/expert
    // pairs; CPU chunks are at most eight columns sharing one packed record.
    long coop_windows=0, coop_cpu_columns=0, coop_cpu_chunks=0;
    long coop_primary_columns=0, coop_primary_groups=0, coop_primary_chunks=0, coop_staging_columns=0;
    long coop_helper_columns=0;
    long primary_submit_queries=0, primary_submit_not_ready=0;
    long cpu_first_passes=0;
    long long helper_graph_launches=0;   // mirrors the helper's cumulative count after each collect
    long long helper_zero_copy_checked=0, helper_zero_copy_mismatches=0;   // (likewise: helper_zero_copy=2's checks)
    // exchange_schedule: the 3090's x4 and host pieces and their summed wall ms, its passes submitted while an x4 piece was
    // in flight, the 5090's download pieces sent in idle passes and at forward ends, the idle passes used, and mode 2's
    // copies checked (3090 copies and 5090 downloads and uploads) and mismatches
    long long exchange_x4_pieces=0, exchange_host_pieces=0, exchange_job_overlaps=0;
    double exchange_x4_ms=0, exchange_host_ms=0;
    long exchange_download_pass_pieces=0, exchange_download_flush_pieces=0, exchange_download_windows=0;
    long exchange_host_urgent_forwards=0;   // forward ends whose lease's host copy had waited too long and ran at once
    long long exchange_verified=0, exchange_mismatches=0;
    // early_copy: copies issued, used by the predicted layer, evicted unused, readers that waited on an unfinished copy,
    // passes skipped because the copy engine had other work, and passes with no free frame.
    long early_copies=0, early_copy_used=0, early_copy_unused=0, early_copy_waits=0, early_copy_blocked=0, early_copy_no_frame=0;
    long early_copy_verified=0;   // MM3_EARLY_COPY_VERIFY: frames read back and found equal to their HOST record
    // MM3_PREPARE_EVENTS: one-row generation MoE layers timed; the preparation's span on the primary stream, the host's
    // enqueue time for it, and the host's wait at the join, all summed in ms.
    long prepare_events=0;
    double prepare_gpu_ms=0, prepare_host_ms=0, prepare_join_ms=0;
    // prep_graph: preparations run through a graph, graphs captured (and their host ms, capture and first launch), captures
    // that failed (the launches instead, graphs off for the load), and under verify the replays found equal / different.
    long prep_graph_launches=0, prep_graph_captures=0, prep_graph_fallbacks=0, prep_graph_verified=0, prep_graph_mismatches=0;
    long prep_graph_zero_copy=0;   // (captures whose host copies are kernels on device views of the pinned buffers)
    // expert_graph: expert phases run through a graph, graphs captured (and their host ms), captures that failed, verify
    long expert_graph_launches=0, expert_graph_captures=0, expert_graph_fallbacks=0, expert_graph_verified=0, expert_graph_mismatches=0;
    double expert_graph_capture_ms=0;
    double prep_graph_capture_ms=0;
    long long early_copy_bytes=0;
    double coop_plan_ms=0, coop_cpu_run_ms=0, coop_cpu_predicted_ms=0;
    long helper_queue_passes = 0, helper_queue_empty = 0;
    long miss_union = 0;          // distinct executed non-ready experts per MoE layer/window
    uint64_t miss_union_bytes = 0; // record bytes incl. header, CPU + foreground DMA service demand
    double lane_cpu1_ms=0, lane_cpu_add_ms=0, lane_primary_ms=0;
    // Native first-eight-layer cost for the bounded stage/window decision.
    // GPU elapsed includes stream idle during host service; submit wall ends
    // when the prefix has been enqueued. Neither is a full pipeline timing.
    double prefix8_gpu_ms=0, prefix8_submit_ms=0;
    long prefix8_calls=0;
    double load_s = 0;
    long staged = 0;              // expert blobs copied into staging slots
    long resident = 0, experts = 0;
    double vram_gb = 0;           // the resident experts' bytes (every member's slice)
    long pooled = 0;              // experts in the pinned pool (0: no pool - the misses read from the mapped files)
    double logical_pinned_gb = 0; // uncompressed logical expert bytes held in HOST
    double pinned_gb = 0;         // the pinned pool of the experts not in VRAM
    double all_gb = 0;            // every routed expert's bytes (the Monitor: what neither VRAM nor the pool holds is on the SSD)
    int kv_ring = 0;              // KV streaming: the positions a layer's VRAM cache holds (0: all - off)
    double kv_host_gb = 0;        // ... and the host memory holding every position
    long profiled = 0;            // routings counted (save_profile writes them)
    long cpu_experts = 0;         // routed experts the CPU lane computed
    int cpu_threads = 0;          // the CPU lane's threads (0: no CPU lane)
    long promoted = 0;            // experts copied into a staging slot because they kept coming back
    long promo_queries=0, promo_not_ready=0, foreground_queries=0, foreground_not_ready=0;
    long prefetch_predictions=0, prefetch_rows=0, prefetch_admitted=0, prefetch_cancelled=0;
    long prefetch_useful=0, prefetch_unused=0, prefetch_late=0;
    uint64_t prefetch_issued_bytes=0, prefetch_useful_bytes=0, prefetch_unused_bytes=0, prefetch_late_bytes=0;
    double prefetch_policy_ms=0;
    long adaptive_issued=0,adaptive_published=0,adaptive_queries=0,adaptive_not_ready=0,adaptive_hits=0;
    long adaptive_reused=0,adaptive_cancelled_promotions=0,adaptive_no_gain=0,adaptive_no_slot=0,adaptive_source_busy=0;
    uint64_t adaptive_h2d_bytes=0,adaptive_d2h_bytes=0;
    double adaptive_host_ms=0;
    long helper_live_started=0,helper_live_withdrawals=0,helper_live_publications=0,helper_live_finished=0;
    long helper_timed_passes=0;
    double helper_host_upper_ms=0,helper_queue_ms=0,helper_submit_ms=0;
    double helper_dispatch_critical_upper_ms=0;
    double helper_detection_upper_ms=0,helper_collect_block_ms=0,helper_after_ready_ms=0;
    // prompt_layerwise: reads, their rows, prompts that fell back to chunks (no memory), and the reads' work: 5090 jobs (of
    // up to 8 columns; residents' among them), uploads, 3090 columns and passes, 16-row dense groups; GPU ms of the dense
    // phases, the residents' jobs, the expert phases (dense end to the last staged expert), the combines, the copy
    // stream's upload spans and each layer from dense start to combine end; host ms of the 3090's passes (first submit to
    // last collect), the stream's wait for them after its experts, pinned allocations and whole reads.
    long prompt_reads=0, prompt_rows=0, prompt_fallbacks=0, prompt_res_jobs=0, prompt_jobs=0, prompt_uploads=0;
    long prompt_helper_cols=0, prompt_helper_passes=0, prompt_dense_groups=0;
    double prompt_dense_ms=0, prompt_res_job_ms=0, prompt_expert_phase_ms=0, prompt_combine_ms=0, prompt_link_ms=0;
    double prompt_layer_ms=0, prompt_helper_ms=0, prompt_helper_wait_ms=0, prompt_host_alloc_ms=0, prompt_wall_ms=0;
    // prompt rows (either path, outside generation): picks, and picks the bonus moved off the router's top four
    long prompt_picks=0, prompt_moved=0;
    double t_promo = 0;           // native: host time querying optional completions; legacy TP: waits
    double t_sync = 0, t_lane = 0, t_token = 0;   // decode: waiting on the routing, the CPU lane, all (seconds)
};

/// A forward's logits rows as the sampler's candidates (Model::forward_cands): per row the kc largest logits and every
/// one equal to the kc-th ((id, logit) pairs by logit desc, then id asc) and the logits of the extra ids - all a
/// sampler with top_k <= kc - extra.size() needs of the row when its penalties touch the extra ids alone.
struct Cands {
    int kc = 1;                                            // in: the largest a row
    std::vector<int> extra;                                // in: ids every row reports too (the penalised ones)
    std::vector<std::vector<std::pair<int, float>>> top;   // out: [row]
    std::vector<std::vector<float>> ex;                    // out: [row][extra]
};

/// A sampled DSpark proposal's distribution (Model::draft_sampled): the draft's biased logits through these, as the
/// request samples the target.
struct DraftSampling {
    float temperature = 1.0f, top_p = 1.0f, min_p = 0.0f;
    int top_k = 64;
};

class Model {
public:
    Model();
    ~Model();
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;

    bool load(const Options& o, std::string& err);
    bool set_routing(bool enabled, float min_ratio, std::string& err, bool prior = false, float prior_bonus = 0.05f, bool dampen = false);
    bool routing_enabled() const;
    bool routing_dampen() const;
    bool routing_dampen_supported() const;
    bool set_routing_prefetch(bool enabled, std::string& err);
    bool routing_prefetch() const;
    bool routing_prefetch_supported() const;
    uint64_t routing_prefetch_outstanding_bytes() const;
    size_t routing_prefetch_device_bytes() const;
    size_t routing_prefetch_host_bytes() const;
    bool native_cooperative() const;
    HelperWeightTraffic helper_weight_traffic() const;
    bool helper_live_pending() const;
    bool cooperative_stage_reuse() const;
    bool set_cooperative_stage_reuse(bool enabled, std::string& err);
    double cooperative_cpu_scale() const;
    bool set_cooperative_cpu_scale(double scale, std::string& err);
    bool primary_submit_query() const;
    bool set_primary_submit_query(bool enabled, std::string& err);
    bool cpu_first() const;
    bool set_cpu_first(bool enabled, std::string& err);
    // Experimental: a minimal kernel on the helper GPU during this generation. Off by default.
    void set_helper_occupancy(bool on);
    bool helper_occupancy_control() const;
    long long helper_occupancy_launches() const;
    // Experimental: the helper's spin waits and graph submission for this generation. Off by default.
    void set_helper_fast(bool on);
    bool helper_fast() const;
    // Experimental: those graphs' copies as zero-copy kernels instead of host copy nodes. Off by default.
    void set_helper_zero_copy(int mode);   // 0 off, 1 on, 2 on and every such pass checked
    int helper_zero_copy() const;
    // Experimental: the next layer's predicted HOST expert copied a layer early, for this generation. Off by default.
    bool early_copy_control() const;
    bool set_early_copy(int mode, std::string& err);   // 0 off, 1 idle copy engine only, 2 also behind one demand copy, 3 predict only (diagnostic)
    int early_copy() const;
    void set_early_copy_legacy(bool on);   // per request: the first form of the early copy, for a within-load A/B
    bool early_copy_legacy() const;
    // Experimental: each one-row MoE layer's preparation replayed as a CUDA graph, for this generation. Off by default.
    bool expert_graph_control() const;
    bool set_expert_graph(int mode, std::string& err);   // 0 off, 1 on, 2 verify (diagnostic: every replay checked against the launches)
    int expert_graph() const;
    // (DFlash2 diagnostic) a proposal's parts switched off - Draft::set_debug's bits; 0 none
    void set_dflash2_debug(int bits);
    bool prep_graph_control() const;
    bool set_prep_graph(int mode, std::string& err);   // 0 off, 1 on, 2 verify (diagnostic: every replay checked against the launches)
    int prep_graph() const;
    bool dense_split_control() const;
    bool set_dense_split(int mode, std::string& err);   // 0: split K up to 256 outputs (before); 1: up to 512
    int dense_split() const;
    // Per request (item 3 of the order of 2026-11-10). exchange_schedule: 0 today's exchanges; 1 their copies in pieces in
    // idle time; 2 as 1 with every copy checked (diagnostic). residency_decay: the native score's decay per five rows
    // (0.95). helper_max_exchanges: the 3090's exchange credit per five generated rows (1).
    // prompt_layerwise (item 1 of the order of 2026-11-10): 0 today's 16-row chunks; 1 prompts of more than 32 rows read
    // one layer at a time (each expert served once for all rows). prompt_check (diagnostic): 1 the prompt's routing sees
    // the residents only and the chunked path stages every host expert, so both paths compare bit for bit.
    size_t device_free_bytes() const;   // the first GPU's free memory now (cudaMemGetInfo)
    bool prompt_layerwise_control() const;
    bool set_prompt_layerwise(int mode, std::string& err);
    int prompt_layerwise() const;
    bool set_prompt_check(int mode, std::string& err);
    int prompt_check() const;
    // whether a prompt of n rows (want rows' logits) would be read layer by layer now; empty reason: yes
    std::string prompt_read_refusal(int n, int want) const;
    bool exchange_schedule_control() const;
    bool set_exchange_schedule(int mode, std::string& err);
    int exchange_schedule() const;
    bool set_residency_decay(double decay, std::string& err);
    double residency_decay() const;
    bool set_helper_max_exchanges(double capacity, std::string& err);
    double helper_max_exchanges() const;
    // exchange_host_wait: forward ends a scheduled lease's host copy waits for its window before it runs at once (2)
    bool set_exchange_host_wait(int forwards, std::string& err);
    int exchange_host_wait() const;
    // The loaded native helper owns exclusive records, independent of routing.
    bool native_helper_exclusive() const;
    float routing_ratio() const;
    bool routing_prior() const;
    float routing_prior_bonus() const;
    bool primary_adaptive_supported() const;
    bool primary_adaptive_enabled() const;
    // Request-boundary control. Disabling stops admission; pending copies keep
    // their storage and continue normal nonblocking completion publication.
    bool set_primary_adaptive(bool enabled,std::string& err);
    PrimaryResidencyPolicy primary_residency_policy() const;
    double primary_residency_service_ms() const;
    double primary_residency_h2d_ms() const;
    double primary_residency_d2h_ms() const;
    long long residency_demand_rows() const;
    bool primary_backing_enabled() const;
    bool set_primary_residency_policy(const PrimaryResidencyPolicy& policy,std::string& err);
    bool helper_live_supported() const;
    bool helper_live_enabled() const;
    bool set_helper_live(bool enabled,std::string& err);
    bool helper_dispatch_control() const;
    bool helper_dispatch_available(const std::string& mode) const;
    std::string helper_dispatch() const;
    size_t helper_queue_device_bytes() const;
    size_t helper_queue_primary_bytes() const;
    size_t helper_queue_helper_bytes() const;
    size_t helper_queue_pinned_bytes() const;
    bool set_helper_dispatch(const std::string& mode,std::string& err);
    // Serialized request boundary. No extra arena or stream synchronization.
    bool primary_arithmetic_control() const;
    bool primary_arithmetic_available(const std::string& mode) const;
    std::string primary_arithmetic() const;
    std::string primary_lane_calibration_arithmetic() const;
    size_t primary_arithmetic_arena_bytes() const;
    bool set_primary_arithmetic(const std::string& mode,std::string& err);
    bool set_draft_active(bool enabled, std::string& err);
    // Once per native load, remeasure the complete-bank service after actual
    // prompt execution. Joins all readers before reusing staging scratch.
    bool calibrate_service_after_prefill(std::string& err);
    bool draft_active() const;
    /// Runs ids at positions pos0 .. pos0 + n - 1 (the cache holds every position before pos0), in chunks of at most
    /// Options::chunk; logits_out (if given) gets the logits of the last `want` positions of this call, [want][vocab]
    /// F32, row-major.
    bool forward(const int* ids, int n, int pos0, int want, float* logits_out, std::string& err);
    int native_stage_prefix() const;
    // Sticky native failure flags only; false does not establish that CUDA is
    // idle or healthy. Tensor-parallel execution returns true (unsupported).
    bool native_state_poisoned() const;
    // A ticket is an opaque, single-use handle owned by this Model. Calls are
    // serialized by the caller. Each slot writes count * vocab() floats.
    class StageTicket {
        friend class Model;
        uint64_t value=0;
    public:
        StageTicket() = default;
    };
    bool stage_begin(const int* ids, int n, int pos0, StageTicket& ticket, std::string& err);
    bool stage_prefix_submit(StageTicket ticket, int slot, std::string& err);
    bool stage_prefix_ready(StageTicket ticket, int slot, bool wait, bool& ready, std::string& err);
    // Launch joins the selected prefix; each service submits one suffix layer.
    // After all layers/head are queued, ready(false) queries the actual stream.
    // Finish requires readiness and writes one slot of logits, then updates draft context.
    bool stage_suffix_launch(StageTicket ticket, int slot, std::string& err);
    bool stage_suffix_service(StageTicket ticket, int slot, bool& submitted_all, std::string& err);
    bool stage_suffix_ready(StageTicket ticket, int slot, bool wait, bool& ready, std::string& err);
    bool stage_suffix_finish(StageTicket ticket, int slot, float* logits, size_t capacity, std::string& err);
    bool stage_suffix_execute(StageTicket ticket, int slot, float* logits, size_t capacity, std::string& err);
    // Normal finish also discards any unexecuted B suffix after draining all
    // submitted work. completed_rows counts only finished suffixes. After a
    // partial suffix, resume at the completed frontier or an earlier position.
    bool stage_finish(StageTicket ticket, int& completed_rows, std::string& err);
    bool forward_stage_window(const int* ids, int n, int pos0, bool overlap,
        const std::function<bool(int)>& after_a, int& completed_rows, float* logits, std::string& err);
    /// The same with the last `want` rows as candidates instead of logits (the tensor-parallel path picks them on the
    /// head's GPU: a few KB a row back instead of the vocabulary's 800).
    bool forward_cands(const int* ids, int n, int pos0, int want, Cands& c, std::string& err);
    /// (step 6, the GPU nucleus) set_nucleus(true): the following forwards' fast-head rows stay on the head's GPU and
    /// each row's candidates (its kCandCap largest logits, desc then id asc) and stats (maximum, total of exp(x - max) in
    /// double, finite) come back instead of the logits - nucleus_rows() of them after a forward that did (0 otherwise:
    /// the logits went to logits_out as before). fetch_logits_row copies one such row's logits back (a whole-row
    /// fallback), valid until the next forward. The single-GPU-sequence path with the fast head only.
    bool nucleus_supported() const;
    void set_nucleus(bool on);
    int nucleus_rows() const;
    bool nucleus_row(int r, std::vector<std::pair<int, float>>& top, float& maximum, double& total, bool& finite) const;
    bool fetch_logits_row(int r, float* out, std::string& err);
    /// --parallel (Options::seqs): the sequences, and the one forward / forward_cands read and write (0 at the load).
    int seqs() const;
    void bind(int s);
    /// One token of each of R conversations together (R <= 16, distinct sequences): ids[r] at position pos[r] of
    /// sequence seq[r], whose cache holds its positions before it; logits_out [R][vocab].  Each row is the same bits as
    /// that token's forward alone: the dense matrices on the token kernels' several columns, the routed experts on the
    /// verify window's (each distinct expert read once for its rows), each row's attention on the token path's calls
    /// over its own cache.
    bool forward_rows(const int* ids, const int* seq, const int* pos, int R, float* logits_out, std::string& err);
    /// --parallel on a layer split: the GPUs as a pipeline.  pipe_stages(): its GPUs (1: none - one GPU, the
    /// tensor-parallel path, no fast decode, one sequence).  pipe_tick: a group of R rows (forward_rows' rows, R 0: none)
    /// enters the first GPU - free after every tick - and every group in flight runs its GPU's layers, then moves one
    /// GPU on; a group past the last GPU: its logits [R][vocab] for pipe_take (oldest first, with its tag).  Each row
    /// the same bits as forward_rows'.  A sequence is in one group at most; forward / forward_rows only with none in
    /// flight (pipe_busy() 0); pipe_drop() forgets them (after an error).
    int pipe_stages() const;
    int pipe_busy() const;
    bool pipe_tick(const int* ids, const int* seq, const int* pos, int R, int tag, std::string& err);
    bool pipe_take(int& tag, std::vector<float>& logits);
    void pipe_drop();
    /// (the tensor-parallel path, whose prompt chunks go through its GPU groups as a pipeline: a whole prompt is one
    /// call) f(tokens of the call done) after each chunk, on the calling thread; false: no further chunk starts - the
    /// call returns true with done() of its tokens in the cache (the logits only when all are).  set_progress({}) ends
    /// it.  pipelined(): whether a prompt in one call gains from it (else calls of Options::chunk are as good).
    void set_progress(std::function<bool(int)> f);
    int done() const;
    /// (the tensor-parallel path) ids as one prompt from position 0, the next-token NLL of every position kept
    /// (ppl_nll()[p] = -log p(ids[p + 1])) - its chunks one at a time, the head on each one's rows.
    bool forward_ppl(const int* ids, int n, std::string& err);
    const std::vector<double>& ppl_nll() const;
    bool pipelined() const;
    int vocab() const;
    int max_ctx() const;
    int bos() const;
    int eos() const;
    const Stats& stats() const;
    std::vector<std::string> info_lines() const;
    /// The Monitor's map of each GPU (serve/gpu_map.py's VRAM lines): what each part holds there, measured at the load.
    std::vector<std::string> vram_lines() const;
    /// The prefill calibration (maya-mm3 --calibrate-prefill): the prompt chunk a forward takes (at most the chunk the
    /// load sized its buffers for: chunk_max) and the prompt path's GEMM backend, changed between forwards.
    int chunk() const;
    int chunk_max() const;
    void set_chunk(int c);
    void set_prefill_gemm(int backend);
    int prefill_gemm() const;
    bool lt_available() const;
    /// The calibration's key of this PC (its GPUs, their architecture, the GPU runtime and driver) and the file its
    /// results are kept in (one line a PC; the load takes the line of its own PC); write_tune keeps `settings` for it.
    std::string tune_key() const;
    std::string tune_file() const;
    bool write_tune(const std::string& settings, std::string& err) const;
    /// The routings counted so far (with the loaded profile's, if any) as a profile file.
    bool save_profile(const std::string& path, std::string& err) const;
    bool native_placement(const std::vector<uint32_t>& profile, bool reseed_primary, std::string& report_json, std::string& err);
    bool native_primary_backing(bool enable, std::string& report_json, std::string& err);
    // Joined setup controls; checkpoint is in-process and is not a restart file.
    bool native_placement_helper_first(const std::vector<uint32_t>& profile, std::string& report_json, std::string& err);
    bool native_learned_checkpoint(const std::string& action, std::string& report_json, std::string& err);
    bool native_trace_begin(std::string& err);
    // Light recorder boundaries; they do nothing unless a light capture is active.
    void light_generation_begin(long long position);
    void light_generation_end(bool ok,long long position,long long emitted,long long drafts_offered,
                              long long drafts_accepted,long long target_rounds,long long spec_rounds,double elapsed_ms);
    // Explicit offline diagnostic: fixed evaluated expert work, no generation
    // or quality claim. Input is fully validated before any execution.
    bool native_replay_workload(const std::string& path, std::string& report_json, std::string& err);
    bool native_replay_services(std::string& report_json, std::string& err);
    bool generation_profile_begin(std::string& err);
    bool generation_profile_end(std::string& err);
    void generation_profile_discard();
    bool generation_profile_snapshot(std::vector<uint32_t>& counts, std::string& err) const;
    bool generation_profile_reset(std::string& err);
    bool generation_profile_read_original(std::vector<uint32_t>& counts, std::string& err) const;
    /// The DSpark draft (Options::dspark): its block (the most tokens a proposal has; 0 without the draft), and up to
    /// n tokens proposed after `anchor` at position pos (the cache holds the positions before it) with their
    /// confidences.
    int draft_block() const;
    bool draft_context_available(int pos) const;
    bool draft(int anchor, int pos, int n, std::vector<int>& drafts, std::vector<float>& conf, std::string& err);
    /// ... each draft drawn from the draft's distribution (ds; u[i] the draw of position i, uniform in [0, 1)) instead
    /// of its argmax, q[i] that distribution ((id, probability), the most likely first) - the target keeps a draft d
    /// with min(1, p(d) / q(d)) and else draws from max(0, p - q), which leaves every token the target's own sample.
    bool draft_sampled(int anchor, int pos, int n, const DraftSampling& ds, const std::vector<float>& u,
                       std::vector<int>& drafts, std::vector<float>& conf,
                       std::vector<std::vector<std::pair<int, float>>>& q, std::string& err);
    /// The vision tower (Options::vision): its geometry (null without it), a picture's rows (rgb height x width x 3,
    /// sides multiples of 28 -> F32 [tokens][hidden]), and the pictures the next forwards' image tokens take their
    /// rows from (position start .. start + rows / hidden - 1; set_images({}) after the prompt).
    struct Image {
        int start = 0;
        std::vector<float> rows;
    };
    const VisionGeo* vision() const;
    bool encode_image(const uint8_t* rgb, int width, int height, std::vector<float>& rows, std::string& err);
    void set_images(std::vector<Image> images);

private:
    struct Impl;
    Impl* p;
    TpEngine* tpe = nullptr;   // (Options::tp > 0: every call goes to it)
};

}  // namespace mm3
