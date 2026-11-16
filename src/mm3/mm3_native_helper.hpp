// Ordered MiniMax NVFP4 helper: one retained 3090 pool and one in-flight pass.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mm3 {
class NativeHelper {
public:
    static constexpr int kHidden = 6144, kIntermediate = 3072, kMaxTokens = 16, kMaxJobs = 64, kMaxColumns = 8;
    // A pass carries up to pass_tokens() tokens and four times as many jobs and columns: kMaxTokens (16) unless the
    // load sets MM3_HELPER_PASS_TOKENS (a multiple of 16 up to kMaxPassTokens), which sizes the buffers once at
    // construction. Generation submits 16 tokens at most either way; the layer-by-layer prompt read may submit more.
    static constexpr int kMaxPassTokens = 256;
    static int configured_pass_tokens();
    int pass_tokens() const;
    // Weights already include routed_scaling_factor; each output column is
    // scattered and summed into the corresponding token's F32 result row.
    struct Job {
        int layer = 0, expert = 0, columns = 0;
        int token[kMaxColumns]{};
        float weight[kMaxColumns]{};
    };
    // Exact physical allocations owned by this helper (not source mappings).
    struct Bytes { size_t pool = 0, device_scratch = 0, pinned = 0; };
    // Lifetime counters, including startup loads; bytes are transferred expert
    // payload/stride excluding Maya's metadata header, not activation I/O.
    struct WeightTraffic {
        uint64_t h2d_issued_ops=0, h2d_issued_bytes=0, d2h_issued_ops=0, d2h_issued_bytes=0;
        uint64_t replacements_completed=0, copy_failures=0;
    };
    WeightTraffic weight_traffic() const;
    struct Identity { int layer, expert, slot; };
    // Host steady-clock timestamps in nanoseconds, meaningful only as differences
    // within one process. Returned for successful collections only. They are
    // wall-clock endpoints, not CPU time, GPU elapsed time, or exposed latency.
    struct PassTiming {
        uint64_t submit_enter_ns = 0, enqueue_ns = 0, submit_published_ns = 0;
        uint64_t worker_start_ns = 0, event_record_begin_ns = 0, event_record_end_ns = 0;
        // The event completed between last_not_ready_ns and ready_observed_ns.
        // If no query returned not-ready, last_not_ready_ns is zero and the
        // record-begin timestamp provides a conservative lower endpoint instead.
        uint64_t last_not_ready_ns = 0, ready_observed_ns = 0, worker_published_ns = 0;
        uint64_t collect_enter_ns = 0, collect_wait_begin_ns = 0, collect_wait_end_ns = 0, collect_done_ns = 0;
        // Actual dispatched route; input_format: 0 original BF16, 1 packed NVFP4,
        // 2 exact NVFP4 in BF16, 3 byte codes plus scales (selected MMA encoding).
        int input_format=0, kernel=0, execution=0; // effective mask after Q1's all-one-column bit1 strip
        bool multi=false, sm89=false, graph=false, graph_fallback=false;
        bool gpu_measured=false;
        float activation_gpu_ms=0, compute_gpu_ms=0, result_gpu_ms=0;
    };

    // capacity is the number of records reserved on device. The record must be
    // the shared nvfp4_expert_rg16_v1 layout, record_bytes(6144,3072), padded
    // to record_stride. Construction allocates all scratch and pinned buffers.
    NativeHelper(int device, int capacity, bool worker_compute = true);
    ~NativeHelper();
    NativeHelper(const NativeHelper&) = delete;
    NativeHelper& operator=(const NativeHelper&) = delete;
    // Only before seal(); blocks until transfer completes. Source is a caller-owned,
    // helper-accessible portable pinned staging buffer containing the RG16 payload
    // and stride padding only, without Maya's 256-byte metadata header. Pass that
    // header's scale arrays separately. Do not pass a CPU/primary-only VMM mapping
    // directly. The source is borrowed for this call only. Returns false on invalid
    // geometry/identity/scales or CUDA failure. Scales are original values: the FFN uses the reciprocal
    // of weight_global and dynamic INT8 activations; input_global is preserved
    // as metadata but is not used by that arithmetic.
    bool add_expert(int layer, int expert, const void* record, size_t record_bytes,
                    const float weight_global[3], const float input_global[3], std::string& error);
    bool seal(std::string& error);
    // Ready bits are host snapshots of fully copied records.
    std::array<uint32_t, 4> ready(int layer) const;
    Bytes bytes() const;
    // Same allocation formula as the constructor; no CUDA allocation.
    static Bytes planned_bytes(int capacity, bool worker_compute = true);
    // Snapshots and stable pool identity are diagnostic; a poisoned helper has
    // no ready records. Replacement retains this pool and all worker scratch.
    std::vector<Identity> records() const;
    struct QueueIdentity { int slot=-1; float multiplier[3]{}; };
    // Caller must serialize snapshots with all ownership mutations and retire
    // queued readers before live withdrawal or replacement.
    bool queue_snapshot(int layer, std::array<QueueIdentity,128>& out, std::string& error) const;
    bool worker_compute_available() const;
    bool replay_probe_timing(bool enabled, std::string& error);
    // Experimental control, off unless requested: fills the gaps between passes with one polling
    // kernel on this GPU's helper stream, so its clock management does not treat it as idle during
    // generation. It changes no device setting, and the worker releases it before each pass.
    void set_occupancy(bool on);
    bool occupancy() const;
    uint64_t occupancy_launches() const;
    // Experimental A/B control, off by default: the worker and the collector spin briefly before blocking,
    // and one-column passes submit through a CUDA graph per job count instead of separate calls.
    void set_fast(bool on);
    bool fast() const;
    // The card's own expert kernel (mm3_helper_kernels.cuh) in place of the shared dp4a kernel. Off by default.
    // multi: for a pass with a job of more than one column; one: for a pass of one-column jobs. Each is
    //   0   the shared dp4a kernel (fast_ffn_cuda_packed)
    //   1, 2, 4, 8   the tensor-core kernel (BF16 activations used exactly, every column of a job in one product,
    //                one launch for all widths) with that many warps a group of 16 output rows, tuned per card
    //   16  today's dp4a arithmetic compiled per job width (today's output; one launch group a width)
    //   32  the same with jobs of every width in one launch
    //   65, 66, 68, 72   kernel item 5: NVFP4 activations (the 5090's encoding) on int8 tensor cores with 1, 2, 4, 8 warps a
    //                    group (mm3_helper_q.cuh); a precision change, off unless asked for
    //   81, 82, 84, 88   kernel item 5b: the same outputs bit for bit on the BF16 tensor-core product (variant 2: exact
    //                    BF16 activations, no int-to-float conversions, the packed rows read by the up kernel)
    //   17, 18, 20, 24   exact doubled E2M1 on FP8 tensor cores (sm_89 required)
    //   33, 34, 36, 40   int8 products, exact bounded integer-to-float bit conversion
    //   49, 50, 52, 56   variant 2 with NVFP4 values expanded once into BF16 before the up kernel
    // multi may have 100 added to a tensor-core value or to 32: such a pass is then submitted as one CUDA graph when
    // set_fast is on. Other values are taken as 0.
    void set_kernel(int multi, int one);
    int kernel_multi() const;
    int kernel_one() const;
    // The tensor-core kernel's loads a batch (one-column and multi-column passes alike): 1 variant 1 (ffn1, the
    // default); 2, 4 or 8 variant 2 (ffn2: the same output, each warp issues the loads of that many units before it
    // uses them). Other values are taken as 1. A pass reads it once; its graphs are kept apart by it.
    void set_loads(int loads);
    int loads() const;
    // Same-arithmetic execution flags: bit1 hidden encoding schedule (Q1 only
    // for jobs with >1 columns; all-one-column passes strip it before dispatch), bit2
    // compact prefetched scale metadata, bit4 Q2 early packed activation decode.
    // Nonzero modes require Q1/Q2 in both width modes; bit4 requires Q2 in both.
    // Changed modes require healthy idle ownership. Each mode retains its own
    // lazily allocated graph cache; no request recaptures another mode's graphs.
    bool set_execution(int mode, std::string& error);
    int execution() const;
    uint64_t kernel_launches() const;
    // Experimental A/B control, off by default: those graphs move their inputs, descriptors and results with copy
    // kernels on the pinned buffers (zero-copy) instead of host copy nodes. A separate set of graphs. 0 off, 1 on,
    // 2 on and each such pass checked: the buffers read back through the copy engine against the bytes moved.
    void set_zero_copy(int mode);
    int zero_copy() const;
    uint64_t zero_copy_checked() const;
    uint64_t zero_copy_mismatches() const;
    uint64_t graph_launches() const;
    uintptr_t pool_identity() const;
    size_t record_stride() const;
    // Sealed, idle helper only. Staging must be caller-owned, CUDA-portable
    // pinned memory of exactly record_stride() bytes. Completion is joined
    // before success. On failure, staging_unsafe means CUDA may still access
    // the buffer: its owner must retain it indefinitely (or until independently
    // proven retired). A transfer failure poisons the helper; no inference may
    // resume using it. Exported scales are valid only on success.
    bool export_expert(int layer, int expert, void* staging, size_t bytes,
                       float weight[3], float input[3], bool& staging_unsafe, std::string& error);
    bool replace_expert(int old_layer, int old_expert, int new_layer, int new_expert,
                        const void* staging, size_t bytes, const float weight[3], const float input[3],
                        bool& staging_unsafe, std::string& error);
    // One optional live replacement. All API calls below are serialized by the
    // caller at forward boundaries; they never wait for a copy. A and B denote
    // incoming HOST and outgoing helper identities respectively. All three
    // borrowed payload buffers are distinct, stride-sized, and remain owned by
    // the caller until live_retire proves they are no longer accessed. Incoming
    // and outgoing are portable pinned for this device; host_slot is CPU-accessible.
    enum class LivePhase { idle, preparing, prepared, uploading, uploaded, published, recycling, complete, failed };
    // storage_unsafe means the lease has not ended, even between copy stages.
    struct LiveStatus { LivePhase phase = LivePhase::idle; bool storage_unsafe = false; std::string error; };
    // Starts CPU host_slot -> incoming and device B -> outgoing concurrently.
    // B remains ready until live_withdraw; its original divisors are returned
    // synchronously. A's original divisors are copied, never transformed.
    bool live_begin(int old_layer, int old_expert, int new_layer, int new_expert,
                    void* host_slot, void* incoming, void* outgoing, size_t bytes,
                    const float weight[3], const float input[3],
                    float old_weight[3], float old_input[3], std::string& error);
    LiveStatus live_poll();
    // Caller first makes B canonical in outgoing and joins all helper readers.
    // Withdraw removes B from readiness and asynchronously uploads incoming A.
    bool live_withdraw(std::string& error);
    // Caller performs the A HOST ownership transition at the same joined boundary.
    bool live_publish(std::string& error);
    // Caller has removed A's HOST ownership; copy outgoing B to host_slot.
    bool live_recycle(std::string& error);
    // At complete, caller first switches B's canonical pointer to host_slot.
    // safe=true proves only this live operation's CPU worker and CUDA stream
    // no longer access the buffers. The caller must separately retire every
    // other reader/writer before freeing or reassigning them. Both return value
    // and safe must be true to end the lease. Until then all three buffers
    // remain leased, even if the current copy is quiescent. This is a
    // nonblocking proof, not a drain; failed helpers remain poisoned.
    bool live_retire(bool& safe, std::string& error);
    // Terminal shutdown/maintenance cancellation prevents all further stages
    // and poisons the helper. Repeat until it returns true with safe=true. A
    // busy or uncertain worker keeps safe=false. If B was withdrawn, outgoing
    // may also be the caller's canonical B storage after the helper lease ends.
    bool live_cancel(bool& safe, std::string& error);
    // exchange_schedule (off unless requested): the live exchange's copies in pieces, each started only inside a window
    // its caller opens, so they stay off the passes. 0: each copy at once, as before; 1 scheduled; 2 scheduled and each
    // finished copy compared with its source (a mismatch fails the operation). A change applies from the next piece.
    // Windows are clock_ns() values. x4: a piece on the live copy stream starts only while no pass is submitted and not
    // yet finished, and only if it is expected to end before until_ns (0 closes it; submit() closes it). host: a CPU piece
    // starts at or after open_ns and only if it is expected to end before close_ns.
    void set_live_schedule(int mode);
    int live_schedule() const;
    void live_x4_window(uint64_t until_ns);
    void live_host_window(uint64_t open_ns, uint64_t close_ns);
    // exchange_quiet: revoke BEFORE any new foreground upload submission, then
    // publish completion events for d.s/d.us/d.cs/d.xs. The generation is closed
    // until publication; stale queries/claims cannot reopen it. One already
    // admitted piece (or a prior unscheduled record) may straddle revocation;
    // this does not wait for its DMA.
    // Event handles stay alive until this helper's worker readers retire. They
    // may be re-recorded only after revoking the previous published generation.
    uint64_t live_x4_revoke();
    bool live_x4_publish(uint64_t generation, const std::array<void*,4>& events, std::string& error);
    void live_x4_guard_failed(const std::string& error); // owner-side event create/record failure: close and poison
    // Compatibility for joined maintenance: nullptr disables the guard; a
    // non-null event publishes a single dependency through the same protocol.
    void live_x4_guard(void* uploads_done);
    void live_cpu_busy(bool busy);
    static uint64_t clock_ns();
    struct LiveScheduleStats {
        uint64_t x4_pieces = 0, host_pieces = 0, x4_ns = 0, host_ns = 0;   // pieces and their summed wall time
        uint64_t job_overlaps = 0;            // passes submitted while an x4 piece was in flight
        uint64_t verified = 0, mismatches = 0;   // (mode 2) copies compared with their source
        uint64_t guard_revocations=0, guard_publications=0, guard_admissions=0;
        uint64_t guard_failed_claims=0, guard_stale_claims=0, guard_event_errors=0;
        // Observations at revocation: admitted is the atomic token boundary;
        // inflight means CUDA accepted a copy and completion was not observed.
        // Repeated revocations can observe the same piece more than once.
        uint64_t guard_already_admitted=0, guard_already_inflight=0;
    };
    LiveScheduleStats live_schedule_stats() const;
    // x is BF16 [tokens,6144] in host memory, gathered synchronously into the
    // helper's pinned request buffers. Jobs and x are reusable after return.
    // Sequence is nonzero on success. Exactly one outstanding submission; a
    // second submit is rejected until collect has retired the first.
    uint64_t submit(int tokens, const uint16_t* x, const std::vector<Job>& jobs, std::string& error);
    // Kernel item 5 (helper_q_rows): the same with the rows as the 5090 encoded them - packed NVFP4, kPackedRow (3456)
    // bytes a row (mm3_helper_q.cuh) - for a helper whose one-column and wider kernels are both the NVFP4-activation
    // kernel (set_kernel 65, 66, 68, 72 or 81, 82, 84, 88, wider ones also with 100 added); a row with a block scale 0x7f (the 5090's mark
    // of a non-finite value) is refused as a non-finite BF16 row is.
    uint64_t submit_packed(int tokens, const uint8_t* packed, const std::vector<Job>& jobs, std::string& error);
    // Expanded rows retain the NVFP4 represented values: format 2 is BF16
    // [tokens][6144], accepted only by kernels 49/50/52/56; format 3 is
    // [tokens][6528] (6144 byte codes then 384 E4M3 scales), accepted by the
    // INT8 and FP8 kernels. FP8 byte codes are E4M3 doubled E2M1 values;
    // INT8 byte codes are signed doubled E2M1 values. Caller chooses the
    // representation for the pass's selected kernel. Gathered synchronously.
    uint64_t submit_expanded(int tokens, const void* rows, int format, const std::vector<Job>& jobs, std::string& error);
    // Source-preparation eligibility across both width modes. Requested 4
    // selects each card's own compatible representation (2 or 3). Zero means this
    // card keeps packed transport; callers record that explicit per-card choice.
    int expanded_format(int requested) const;
    bool expanded_fp8() const;
    // Kernel item 5 (third_sm89): this card's expert kernels from the sm_89 build (mm3_helper_sm89.hpp) when it is a card of
    // compute capability 8.9; elsewhere nothing changes. sm89: in effect. sm89_capable: the card and the build allow it.
    void set_sm89(bool on);
    bool sm89() const;
    bool sm89_capable() const;
    // Blocks for exact sequence; writes sum [tokens,6144] including zeros.
    // At most one caller may claim an active sequence; competing collectors are
    // rejected. On execution or nonfinite-result failure the helper is poisoned (a
    // nonfinite result column: unless refused is given, below);
    // output is invalid on any failure and an uncertain buffer is never reused.
    // Optional timing reports raw, potentially overlapping endpoints. In
    // particular, worker-ready to collect-entry can be useful CPU work, not a
    // helper stall; collect processing includes required output reduction.
    //
    // refused (the user's decision of 2026-11-14: "It should recompute that column on another lane and carry on, make sure
    // it logs if it happens"): given, a column whose result holds a non-finite value, or whose weighted sum is not finite,
    // does not poison the helper. It is left out of the output, its token's row holding the sum of that token's other
    // columns (helper_merge::merge_pass), and listed here for the caller to recompute; collect then succeeds and the
    // helper stays usable. Each listed column's detail says where the value went bad: the card's own buffers read back
    // (d_y the kernel's output, d_x its input) against the pinned ones. Without it, such a column poisons as before. Every
    // other failure (a CUDA error, the deadline) poisons either way.
    struct Refused {
        int layer = 0, expert = 0, slot = -1;
        int token = 0;          // the column's row in this pass (its Job.token)
        float weight = 0.0f;    // its Job.weight
        bool sum = false;       // its values were finite but the weighted sum was not (helper_merge::kNonfiniteSum)
        int bad = 0;            // non-finite elements in the result column as it reached host memory (h_y)
        std::string detail;     // nonfinite_detail: the card, the job, h_y and h_x, the multipliers, d_y and d_x read back
    };
    bool collect(uint64_t seq, float* output, size_t output_floats, std::string& error,
                 PassTiming* timing = nullptr, std::vector<Refused>* refused = nullptr);
    // Poisons the helper for good (no further submission), as a failed collect does; why becomes its failure. For a
    // caller whose recompute of a refused column also failed.
    void poison(const std::string& why);
    int device() const;
    // The repair after a refused column (at the next forward boundary): the record of an expert this helper holds, read
    // back synchronously and compared with expected (an RG16 payload of record_stride() bytes) and these scales. Sealed,
    // unpoisoned helper, no pass in flight, and not the slot a live replacement is copying (its other slots may be read
    // beside it). Error path and repair only.
    bool verify_expert(int layer, int expert, const void* expected, size_t bytes, const float weight[3],
                       const float input[3], bool& same, std::string& error);
    // Rewrites that record in its own slot from staging (portable pinned, as for replace_expert) with these scales; the
    // identity and the slot stay. The record is not ready while it is written. Same conditions as verify_expert; a
    // transfer failure poisons, and staging_unsafe means CUDA may still read staging.
    bool rewrite_expert(int layer, int expert, const void* staging, size_t bytes, const float weight[3],
                        const float input[3], bool& staging_unsafe, std::string& error);
    // Nonblocking: whether collect(seq) would return without waiting (the pass finished, or the helper failed).
    bool done(uint64_t seq) const;
    // ---- the helper hand-off (helper_handoff, mm3_helper_handoff.hpp; off unless a request asks). A pass released to the
    // card by the 5090 runs outside submit/collect, on the hand-off's own stream and buffers.
    // Counts every change of which record a slot holds or of its scales (add, replace, rewrite, live withdraw and
    // publish): the hand-off rebuilds the 5090's tables of record addresses when it moves.
    uint64_t records_version() const;
    // A released pass in flight (from the release until its columns are merged): like an active submit, no live
    // withdrawal, replacement or rewrite of a record starts, and no scheduled x4 piece (the x4 window closes).
    void handoff_busy(bool busy);
    // collect's merge for such a pass: its columns (BF16, [columns][6144] in merge order: the jobs as given, which are in
    // submit's order, then their columns) merged into out [tokens][6144] exactly as collect merges a worker pass, with the
    // same refused list and the same injection check (MM3_HELPER_INJECT_NONFINITE). columns is the pinned result slot;
    // the injection may write into it. Without refused, a non-finite column fails (the caller poisons, as collect does).
    bool merge_external(int tokens, const std::vector<Job>& jobs, uint16_t* columns, float* out, size_t out_floats,
                        std::string& error, std::vector<Refused>* refused = nullptr);
private:
    uint64_t submit_rows(int tokens, const void* rows, int format, const std::vector<Job>& jobs, std::string& error);
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}
