#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

// Fixed geometry and byte-exact, versioned HOST representation for native
// 6144 x 3072 RG16 experts. Callers own both encoded and raw records.
namespace mm3::scale_store {
constexpr size_t kHeaderBytes = 256;
constexpr size_t kDescriptorBytes = 256;
constexpr size_t kCodeBytes = 28311552;
constexpr size_t kScaleBytes = 3538944;
constexpr size_t kRawBytes = 31850752;
constexpr size_t kBlockBytes = 65536;
constexpr size_t kBlocks = 54;
constexpr size_t kScratchBytes = kCodeBytes + kScaleBytes;
constexpr size_t kForegroundBuffers = 64;
constexpr size_t kOptionalBuffers = 4;
constexpr size_t kPinnedBytes = (kForegroundBuffers + kOptionalBuffers) * kScaleBytes;

// Input matrices are the contiguous official E4M3 scale tensors: gate/up
// [3072,384], down [6144,192]. No code bytes are read for this sizing pass.
bool encoded_size_from_scales(const uint8_t* gate, const uint8_t* up,
                              const uint8_t* down, size_t& aligned_bytes,
                              std::string& error);
bool encode(const uint8_t* raw, size_t raw_bytes, uint8_t* encoded,
            size_t capacity, size_t& aligned_bytes, std::string& error);
bool validate(const uint8_t* encoded, size_t capacity, std::string& error);
bool decode_raw(const uint8_t* encoded, size_t capacity, uint8_t* raw,
                size_t raw_capacity, std::string& error);

// Internal codec primitives also used by the production service. A successful
// validation must precede decode; the source must remain immutable throughout.
struct View {
    const uint8_t* header = nullptr;
    const uint8_t* codes = nullptr;
    const uint8_t* compressed = nullptr;
    uint32_t offsets[kBlocks + 1]{};
};
bool view(const uint8_t* encoded, size_t capacity, View& out, std::string& error);
bool decode_block(const View& source, size_t block, uint8_t* scales, std::string& error);

enum class OptionalState { Free, Queued, Running, Submitted, Failed };
// Narrow failure-injection seam; null callbacks have no production effect.
// Callback context must outlive all service workers, including quarantine.
enum class Operation { HeaderCopy, CodeCopy, ProgressEventRecord, ProgressEventQuery,
                       ScaleCopy, Interleave, CompletionEventRecord };
struct TestHooks {
    void* context = nullptr;
    cudaError_t (*before)(void*, Operation) = nullptr;
    void (*optional_gate)(void*, int frame) = nullptr;
};

// All methods except optional worker are called by one owner thread. Streams,
// scratch, destinations, completion events and source bank are caller-owned.
// Each scratch must be >= kScratchBytes and exclusive to its stream. Foreground
// buffers are never reused until reset_foreground_after_join(), which requires
// the caller to have proved its owner-window GPU join. Optional source, target,
// event and buffer stay leased until retire_optional after observed CUDA success.
class Service {
public:
    Service(int device, void* foreground_scratch, cudaStream_t foreground_stream,
            void* optional_scratch, cudaStream_t optional_stream,
            unsigned deadline_ms, std::string& error, const TestHooks* hooks = nullptr);
    ~Service();
    Service(const Service&) = delete;
    Service& operator=(const Service&) = delete;
    bool healthy() const;
    std::string error() const;
    size_t foreground_used() const;
    bool issue_foreground(void* destination, const uint8_t* source, size_t capacity);
    bool reset_foreground_after_join();
    bool enqueue_optional(int frame, void* destination, const uint8_t* source,
                          size_t capacity, cudaEvent_t completion);
    OptionalState optional_state(int frame) const;
    // Caller must first query the corresponding event and observe cudaSuccess.
    bool retire_optional(int frame);
    // Waits only for CPU decode/submission, with the configured deadline.
    bool drain_cpu();
    // Only after drain_cpu succeeds and caller has joined BOTH GPU streams and
    // retired all optional jobs. On failure/destruction without release, resources
    // and worker state are intentionally quarantined for process lifetime.
    bool release_after_gpu_join();
private:
    struct Impl;
    Impl* p_ = nullptr;
};
}
