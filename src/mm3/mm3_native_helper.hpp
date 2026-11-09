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
    // x is BF16 [tokens,6144] in host memory, gathered synchronously into the
    // helper's pinned request buffers. Jobs and x are reusable after return.
    // Sequence is nonzero on success. Exactly one outstanding submission; a
    // second submit is rejected until collect has retired the first.
    uint64_t submit(int tokens, const uint16_t* x, const std::vector<Job>& jobs, std::string& error);
    // Blocks for exact sequence; writes sum [tokens,6144] including zeros.
    // At most one caller may claim an active sequence; competing collectors are
    // rejected. On execution or nonfinite-result failure the helper is poisoned;
    // output is invalid on any failure and an uncertain buffer is never reused.
    // Optional timing reports raw, potentially overlapping endpoints. In
    // particular, worker-ready to collect-entry can be useful CPU work, not a
    // helper stall; collect processing includes required output reduction.
    bool collect(uint64_t seq, float* output, size_t output_floats, std::string& error,
                 PassTiming* timing = nullptr);
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}
