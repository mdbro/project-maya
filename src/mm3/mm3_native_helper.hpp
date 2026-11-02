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

    // capacity is the number of records reserved on device. The record must be
    // the shared nvfp4_expert_rg16_v1 layout, record_bytes(6144,3072), padded
    // to record_stride. Construction allocates all scratch and pinned buffers.
    NativeHelper(int device, int capacity);
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
    // Ready bits are host snapshots of fully copied records; immutable after seal.
    std::array<uint32_t, 4> ready(int layer) const;
    Bytes bytes() const;
    // x is BF16 [tokens,6144] in host memory, gathered synchronously into the
    // helper's pinned request buffers. Jobs and x are reusable after return.
    // Sequence is nonzero on success. Exactly one outstanding submission; a
    // second submit is rejected until collect has retired the first.
    uint64_t submit(int tokens, const uint16_t* x, const std::vector<Job>& jobs, std::string& error);
    // Blocks for exact sequence; writes sum [tokens,6144] including zeros.
    // At most one caller may claim an active sequence; competing collectors are
    // rejected. On execution or nonfinite-result failure the helper is poisoned;
    // output is invalid on any failure and an uncertain buffer is never reused.
    bool collect(uint64_t seq, float* output, size_t output_floats, std::string& error);
private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};
}
