#pragma once
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

namespace mm3 {
// Point-in-time effective limits: Linux guest and cgroup limits, reduced by
// enclosing Windows physical/commit availability under WSL. Not a reservation.
struct NativeHostMemorySnapshot {
    std::uint64_t total_physical = 0;
    std::uint64_t available_physical = 0;
    std::uint64_t available_commit = 0;
    std::uint64_t process_rss = 0;
    std::uint64_t guest_available_physical = 0;
    std::uint64_t guest_cached = 0;
    std::uint64_t guest_sreclaimable = 0;
    bool windows_host = false;
    std::uint64_t windows_available_physical = 0;
    std::uint64_t windows_available_commit = 0;
    std::string cgroup_path; // v2 process group, if the hierarchy is present
};
NativeHostMemorySnapshot native_host_reserve_snapshot();

// HOST VMM chunks mapped contiguously for the CPU and exactly one CUDA device.
// Construction succeeds only when the complete requested bank is mapped. Retire
// all CPU/GPU readers before destruction. Helper refills require caller-owned,
// bounded shared staging; this allocation grants the helper no access.
class NativeHostBank {
public:
    // Infernix's 2 GiB default is a minimum, not a complete startup ledger.
    // Pass larger reserves to cover remaining host allocations and headroom.
    static constexpr std::uint64_t default_reserve = 2ULL << 30;
    NativeHostBank(std::size_t bytes, int primary_device,
                   std::uint64_t physical_reserve = default_reserve,
                   std::uint64_t commit_reserve = default_reserve);
    ~NativeHostBank();
    NativeHostBank(const NativeHostBank&) = delete;
    NativeHostBank& operator=(const NativeHostBank&) = delete;
    std::byte* data() const noexcept;
    std::size_t size() const noexcept;
    std::size_t capacity() const noexcept;
    std::size_t allocation_count() const noexcept;
    std::size_t max_allocation_bytes() const noexcept;
    static std::size_t allocation_bytes(std::size_t bytes, int primary_device);
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
