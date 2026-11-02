#pragma once
#include "mm3_cpu.hpp"
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
bool native_cpu_run(CpuLane&, int dim, int ff, const std::vector<NativeCpuExpert>&,
                    const float* x, float* out, void* workspace, size_t bytes);
bool native_cpu_run_multi(CpuLane&, int dim, int ff, const std::vector<NativeCpuExpert>&,
                          const std::vector<std::vector<std::pair<int,float>>>& users,
                          const float* x, int tokens, float* out,
                          const std::vector<std::pair<int,int>>* order,
                          void* workspace, size_t bytes, bool accumulate=false);
}
