#pragma once
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace mm3 {
// Outstanding HOST traffic, shared by both direct GPU movers and the separate
// HOST re-sorter. This bounds admitted non-preemptive chunks, not measured GB/s
// or a guarantee that a chunk finishes before a foreground boundary.
class BackgroundTraffic {
public:
    static constexpr std::size_t kActors=3; // 3090, 4080, HOST re-sorter
    explicit BackgroundTraffic(std::size_t capacity=2u<<20):capacity_(capacity) {}
    bool acquire(std::size_t actor,std::size_t bytes) noexcept {
        if(actor>=kActors || !bytes || bytes>capacity_)return false;
        auto used=used_.load(std::memory_order_relaxed);
        do {if((used&kPoisoned) || used>capacity_-bytes)return false;}
        while(!used_.compare_exchange_weak(used,used+bytes,std::memory_order_acq_rel,std::memory_order_relaxed));
        auto high=high_.load(std::memory_order_relaxed);
        while(high<used+bytes && !high_.compare_exchange_weak(high,used+bytes,std::memory_order_relaxed)) {}
        current_[actor].fetch_add(bytes,std::memory_order_relaxed);
        admitted_[actor].fetch_add(bytes,std::memory_order_relaxed);
        return true;
    }
    // Only after all accesses covered by this reservation have retired.
    void release(std::size_t actor,std::size_t bytes) noexcept {
        current_[actor].fetch_sub(bytes,std::memory_order_relaxed);
        used_.fetch_sub(bytes,std::memory_order_release);
    }
    std::size_t used()const noexcept {return used_.load(std::memory_order_acquire)&~kPoisoned;}
    std::size_t high()const noexcept {return high_.load(std::memory_order_relaxed);}
    std::size_t capacity()const noexcept {return capacity_;}
    void poison()noexcept {used_.fetch_or(kPoisoned,std::memory_order_acq_rel);}
    bool healthy()const noexcept {return !(used_.load(std::memory_order_acquire)&kPoisoned);}
    std::uint64_t admitted(std::size_t actor)const noexcept {return admitted_[actor].load(std::memory_order_relaxed);}
    // Observations of an already-admitted chunk at a CPU foreground boundary;
    // this is not a measured causal delay or proof the DMA has begun.
    void foreground()noexcept {
        for(std::size_t i=0;i<kActors;++i)if(current_[i].load(std::memory_order_acquire))
            overlaps_[i].fetch_add(1,std::memory_order_relaxed);
    }
    std::uint64_t overlaps(std::size_t actor)const noexcept {return overlaps_[actor].load(std::memory_order_relaxed);}
private:
    static constexpr std::size_t kPoisoned=std::size_t(1)<<(sizeof(std::size_t)*8-1);
    const std::size_t capacity_;
    std::atomic<std::size_t> used_{0},high_{0};
    std::array<std::atomic<std::size_t>,kActors> current_{};
    std::array<std::atomic<std::uint64_t>,kActors> admitted_{};
    std::array<std::atomic<std::uint64_t>,kActors> overlaps_{};
};
} // namespace mm3
