// The engine-start memory check's arithmetic (mm3_native_host.hpp): WSL's reclaimable cache counted on the Windows side
// (the user's decision of 2026-11-14, after the first start from minimax-wsl/local was refused with about 23 GB of Linux
// file cache counted as used by Windows). Host only: no CUDA, no Windows query.
#include "mm3_native_host.hpp"

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <limits>

using mm3::NativeHostMemorySnapshot;

namespace {
constexpr std::uint64_t GB = 1000ull * 1000 * 1000;

NativeHostMemorySnapshot guest(std::uint64_t available, std::uint64_t free_, std::uint64_t swap = 0) {
    NativeHostMemorySnapshot s;
    s.guest_available_physical = s.available_physical = available;
    s.guest_free = free_;
    s.guest_reclaimable = mm3::native_host_reclaimable(available, free_);
    s.available_commit = available + swap;
    return s;
}
}  // namespace

int main() {
    // 1. Reclaimable: MemAvailable - MemFree, never below 0.
    assert(mm3::native_host_reclaimable(230 * GB, 207 * GB) == 23 * GB);
    assert(mm3::native_host_reclaimable(5 * GB, 7 * GB) == 0);

    // 2. The refused start of 2026-11-14 (its log's terms, rounded): after the fill the Windows commit term was 16.4 GB
    //    against the 16 GiB reserve while WSL held about 23 GB of cache. Counted, the commit term is 39.4 GB.
    {
        auto s = guest(60 * GB, 37 * GB);   // (23 GB reclaimable in the guest)
        mm3::native_host_apply_windows(s, 40 * GB, 100 * GB, 100 * GB - 16'400'000'000ull);
        assert(s.windows_available_commit == 16'400'000'000ull);          // (the raw Windows term is kept)
        assert(s.available_commit == 39'400'000'000ull);                   // (16.4 + 23)
        assert(s.available_physical == 60 * GB);                           // (min(guest 60, Windows 40 + 23 = 63))
        assert(s.windows_host && s.windows_available_physical == 40 * GB);
        const std::uint64_t reserve_now = 8ull << 30, reserve_before = 16ull << 30;
        assert(s.available_commit >= reserve_now && s.available_commit >= reserve_before);
    }

    // 3. The guest's terms are never raised: the sum is taken with them.
    {
        auto s = guest(10 * GB, 2 * GB);
        mm3::native_host_apply_windows(s, 50 * GB, 200 * GB, 100 * GB);
        assert(s.available_physical == 10 * GB && s.available_commit == 10 * GB);
    }

    // 4. With no reclaimable memory the terms are the plain minimum, as before.
    {
        auto s = guest(50 * GB, 50 * GB, 4 * GB);
        mm3::native_host_apply_windows(s, 30 * GB, 100 * GB, 80 * GB);
        assert(s.guest_reclaimable == 0);
        assert(s.available_physical == 30 * GB && s.available_commit == 20 * GB);
    }

    // 5. A commit above its limit counts as 0 (it wrapped before); the sums saturate.
    {
        auto s = guest(50 * GB, 45 * GB);
        mm3::native_host_apply_windows(s, 1 * GB, 100 * GB, 101 * GB);
        assert(s.windows_available_commit == 0 && s.available_commit == 5 * GB);
        auto t = guest(std::numeric_limits<std::uint64_t>::max(), 0);
        t.available_commit = std::numeric_limits<std::uint64_t>::max();
        mm3::native_host_apply_windows(t, std::numeric_limits<std::uint64_t>::max() - 1, std::numeric_limits<std::uint64_t>::max(), 0);
        assert(t.available_physical == std::numeric_limits<std::uint64_t>::max());
        assert(t.available_commit == std::numeric_limits<std::uint64_t>::max());
    }
    std::puts("mm3_native_host_limits_test: OK");
    return 0;
}
