#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
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
    std::uint64_t guest_free = 0;          // MemFree
    std::uint64_t guest_reclaimable = 0;   // MemAvailable - MemFree, at least 0 (native_host_reclaimable)
    bool windows_host = false;
    std::uint64_t windows_available_physical = 0;
    std::uint64_t windows_available_commit = 0;
    std::string cgroup_path; // v2 process group, if the hierarchy is present
};
NativeHostMemorySnapshot native_host_reserve_snapshot();

/// The guest's reclaimable memory: MemAvailable - MemFree, clamped at 0 - the page cache and slabs Linux gives back on
/// demand, for WSL's own needs, without a new Windows commit.
inline std::uint64_t native_host_reclaimable(std::uint64_t mem_available, std::uint64_t mem_free) {
    return mem_available > mem_free ? mem_available - mem_free : 0;
}

/// Under WSL the Windows-side terms limit the guest's. Windows counts the pages WSL holds as Linux file cache as used,
/// although Linux reuses them for WSL's own needs with no new Windows commit (the first start from minimax-wsl/local was
/// refused on 2026-11-14 with about 23 GB of such cache). So the guest's reclaimable memory (s.guest_reclaimable) is
/// added to both Windows terms - the physical memory available and the commit limit less the committed - before each
/// is taken with the guest's term; the guest's terms stay as they are (the user's decision of 2026-11-14, CLAUDE.md's
/// table). The raw Windows terms are kept in the snapshot; the sums saturate, and a commit above its limit counts as 0.
inline void native_host_apply_windows(NativeHostMemorySnapshot& s, std::uint64_t physical_available,
                                      std::uint64_t commit_limit, std::uint64_t committed) {
    const auto sum = [](std::uint64_t a, std::uint64_t b) {
        return a > std::numeric_limits<std::uint64_t>::max() - b ? std::numeric_limits<std::uint64_t>::max() : a + b;
    };
    s.windows_host = true;
    s.windows_available_physical = physical_available;
    s.windows_available_commit = commit_limit > committed ? commit_limit - committed : 0;
    s.available_physical = std::min(s.available_physical, sum(physical_available, s.guest_reclaimable));
    s.available_commit = std::min(s.available_commit, sum(s.windows_available_commit, s.guest_reclaimable));
}

// HOST VMM chunks mapped contiguously for the CPU and exactly one CUDA device.
// Construction succeeds only when the complete requested bank is mapped. Retire
// all CPU/GPU readers before destruction. Helper refills require caller-owned,
// bounded shared staging; this allocation grants the helper no access.
class NativeHostBank {
public:
    // Infernix's 2 GiB default is a minimum, not a complete startup ledger.
    // Pass larger reserves to cover remaining host allocations and headroom.
    static constexpr std::uint64_t default_reserve = 2ULL << 30;
    // defer_device_access: map the chunks for the CPU only. The driver then backs the memory as the caller's threads
    // first write it, in parallel, and not page by page inside cuMemSetAccess, which fills at 0.9 GiB/s with one
    // thread at a time whatever the number of callers (measured 2026-11-11: 200 s of a 350 s engine start for the
    // 195 GB bank; runs/engine-start-20261111/). Call grant_device() once the bank is filled and before the first
    // copy to the card; the grant itself runs at about 25 GiB/s. The mapping stays selective: the CPU and one card.
    NativeHostBank(std::size_t bytes, int primary_device,
                   std::uint64_t physical_reserve = default_reserve,
                   std::uint64_t commit_reserve = default_reserve,
                   bool defer_device_access = false);
    void grant_device();                    // (no effect when the card already has access)
    bool device_granted() const noexcept;
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
