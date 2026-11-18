#pragma once
#include "mm3_cpu.hpp"
#include "mm3_native_cpu_fast.hpp"
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>
namespace mm3 {
struct NativeCpuExpert { const uint8_t* blob = nullptr; float w = 0; };
// Workspace is caller-owned and 256-byte aligned; entries is the maximum
// number of (expert,token) pairs, at least the number supplied to run_multi.
size_t native_cpu_workspace_bytes(int dim, int ff, int entries);
// Opt-in batch candidate; up to 64 total (expert,token) entries. Query is an
// upper bound for all partitions, including input/output and per-job scratch.
size_t native_cpu_batch_workspace_bytes(int dim, int ff, int entries);
bool native_cpu_run_multi_batch(CpuLane&, int dim, int ff, const std::vector<NativeCpuExpert>&,
                                const std::vector<std::vector<std::pair<int,float>>>& users,
                                const float* x, int tokens, float* out,
                                const std::vector<std::pair<int,int>>* order,
                                void* workspace, size_t bytes, bool accumulate=false);
// The reworked kernel (mm3_native_cpu_fast.hpp; a request's cpu_kernel 1 or 2, or its cpu_l3_prefetch): run_multi_batch's
// contract and output order, act 1 its INT8 activations (every bit run_multi_batch's), 2 NVFP4 activations. l3 (if given,
// one flag an expert): the expert's bytes were read ahead by native_cpu_l3_begin on this lane, so each active worker
// computes the fixed share it read (the caller and the rest take the other experts' work as before). The workspace
// query is at most native_cpu_batch_workspace_bytes for the same entries.
size_t native_cpu_fast_workspace_bytes(int dim, int ff, int entries);
bool native_cpu_run_multi_fast(CpuLane&, int act, int dim, int ff, const std::vector<NativeCpuExpert>&,
                               const std::vector<std::vector<std::pair<int,float>>>& users,
                               const float* x, int tokens, float* out,
                               const std::vector<std::pair<int,int>>* order,
                               void* workspace, size_t bytes, bool accumulate = false,
                               const std::vector<uint8_t>* l3 = nullptr,
                               cpu_fast::Execution execution = cpu_fast::kExecutionBaseline, long* ranged_calls = nullptr);
// The L3 prefetch: each of the lane's active workers reads, in the background, its fixed share of each expert's record
// (blobs: native header + RG16 payload, in order) - the share it computes in native_cpu_run_multi_fast with l3 set -
// until the lane's next call or stop_background. bytes (if given) counts what was read. False: nothing started (no
// active worker, or nothing to read).
bool native_cpu_l3_begin(CpuLane&, int dim, int ff, const std::vector<const uint8_t*>& blobs,
                         std::atomic<unsigned long long>* bytes = nullptr);
bool native_cpu_run(CpuLane&, int dim, int ff, const std::vector<NativeCpuExpert>&,
                    const float* x, float* out, void* workspace, size_t bytes);
bool native_cpu_run_multi(CpuLane&, int dim, int ff, const std::vector<NativeCpuExpert>&,
                          const std::vector<std::vector<std::pair<int,float>>>& users,
                          const float* x, int tokens, float* out,
                          const std::vector<std::pair<int,int>>* order,
                          void* workspace, size_t bytes, bool accumulate=false);
}
