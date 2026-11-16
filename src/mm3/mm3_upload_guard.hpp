#pragma once
// Optional-copy admission against immutable event handles whose recordings may
// be refreshed by one foreground producer. Revocation precedes new uploads;
// publication follows their event records. One copy worker queries and claims.
// No CUDA calls or host waits live here, so ordering/failure fixtures exercise
// exactly the production token protocol.
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace mm3 {
class UploadGuard {
public:
    static constexpr std::size_t kEvents=4; // primary, demand-upload, early/promotion, exchange streams
    using Events=std::array<void*,kEvents>;
    enum class Query { ready, pending, error };
    enum class Result { unrestricted, closed, pending, deferred, stale, admitted, error };
    struct Claim { Result result=Result::closed; std::uint64_t token=0; std::size_t event=kEvents; };
    struct Revocation { std::uint64_t token=0; bool admitted=false; };
    // One consumer owns its cache. Ready markers cannot become incomplete
    // without a producer revoke/re-record, so their proof lasts this generation.
    // The producer never reads/writes this object; generation changes reset it.
    struct PollCache {
        std::uint64_t generation=~std::uint64_t(0),next_query_ns=0;
        unsigned ready_mask=0;
    };

    // The three low bits hold the phase; all upper bits are a generation. A
    // successful claim is the admission boundary, not proof DMA has been issued.
    // A revoke never waits for an admitted chunk. That one chunk may already be
    // submitting/in flight and must be counted separately by the owner.
    Revocation revoke() noexcept { return advance(kClosed); }
    Revocation disable() noexcept { return advance(kDisabled); }
    bool publish(std::uint64_t token,const Events& events) noexcept {
        if((token&kPhase)!=kClosed || state_.load(std::memory_order_acquire)!=token)return false;
        // Only the foreground producer publishes. A stale token cannot replace
        // the current event set. Handles remain alive until all worker readers
        // have retired; generation checks do not authorize freeing an event.
        for(std::size_t i=0;i<kEvents;++i)events_[i].store(events[i],std::memory_order_relaxed);
        return state_.compare_exchange_strong(token,(token&~kPhase)|kOpen,
                                              std::memory_order_release,std::memory_order_relaxed);
    }
    template<class Probe>
    Claim claim(Probe&& probe) {
        return claim_impl(probe,nullptr,0,0);
    }
    template<class Probe>
    Claim claim(Probe&& probe,PollCache& cache,std::uint64_t now_ns,std::uint64_t pending_interval_ns) {
        return claim_impl(probe,&cache,now_ns,pending_interval_ns);
    }
    bool current(std::uint64_t token)const noexcept {
        const auto phase=token&kPhase;
        return token && (phase==kClaimed || phase==kUnrestrictedClaimed) && state_.load(std::memory_order_acquire)==token;
    }
    // Completion/cancellation may reopen only the exact claimed generation. A
    // later foreground revoke/publication or disable always wins.
    void release(std::uint64_t token) noexcept {
        if(!token)return;
        const auto next_phase=(token&kPhase)==kUnrestrictedClaimed?kDisabled:kOpen;
        state_.compare_exchange_strong(token,(token&~kPhase)|next_phase,
                                        std::memory_order_release,std::memory_order_relaxed);
    }
private:
    static constexpr std::uint64_t kPhase=7,kDisabled=0,kClosed=1,kOpen=2,kClaimed=3,kUnrestrictedClaimed=4;
    template<class Probe>
    Claim claim_impl(Probe& probe,PollCache* cache,std::uint64_t now_ns,std::uint64_t interval_ns) {
        const auto observed=state_.load(std::memory_order_acquire);
        if(cache && cache->generation!=(observed&~kPhase)) {
            cache->generation=observed&~kPhase;cache->ready_mask=0;cache->next_query_ns=0;
        }
        if((observed&kPhase)==kDisabled) {
            auto expected=observed;
            const auto token=(observed&~kPhase)|kUnrestrictedClaimed;
            if(!state_.compare_exchange_strong(expected,token,std::memory_order_acq_rel,std::memory_order_acquire))
                return {Result::stale,0,kEvents};
            return {Result::unrestricted,token,kEvents};
        }
        if((observed&kPhase)!=kOpen)return {Result::closed,0,kEvents};
        if(cache && now_ns<cache->next_query_ns)return {Result::deferred,0,kEvents};
        for(std::size_t i=0;i<kEvents;++i) {
            if(cache && (cache->ready_mask&(1u<<i)))continue;
            void* event=events_[i].load(std::memory_order_relaxed);
            if(!event) {if(cache)cache->ready_mask|=1u<<i;continue;}
            const auto q=probe(event);
            if(q==Query::error) {
                // Invalidate the published generation. An earlier producer
                // token cannot reopen it after an event error.
                auto expected=observed;
                state_.compare_exchange_strong(expected,next(observed,kClosed),
                                                std::memory_order_acq_rel,std::memory_order_relaxed);
                return {Result::error,0,i};
            }
            if(q==Query::pending) {
                if(cache)cache->next_query_ns=now_ns+interval_ns;
                return {Result::pending,0,i};
            }
            if(cache)cache->ready_mask|=1u<<i;
        }
        if(cache)cache->next_query_ns=0;
        auto expected=observed;
        const auto token=(observed&~kPhase)|kClaimed;
        if(!state_.compare_exchange_strong(expected,token,std::memory_order_acq_rel,std::memory_order_acquire))
            return {Result::stale,0,kEvents};
        return {Result::admitted,token,kEvents};
    }
    static std::uint64_t next(std::uint64_t state,std::uint64_t phase) noexcept {
        return ((state&~kPhase)+8)|phase; // 2^61 generations exceed the engine's lifetime
    }
    Revocation advance(std::uint64_t phase) noexcept {
        auto old=state_.load(std::memory_order_acquire);
        for(;;) {
            const auto token=next(old,phase);
            if(state_.compare_exchange_weak(old,token,std::memory_order_acq_rel,std::memory_order_acquire))
                return {token,(old&kPhase)==kClaimed || (old&kPhase)==kUnrestrictedClaimed};
        }
    }
    std::atomic<std::uint64_t> state_{kDisabled};
    std::array<std::atomic<void*>,kEvents> events_{};
};
} // namespace mm3
