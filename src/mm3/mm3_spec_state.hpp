#pragma once
#include <algorithm>
#include <cstddef>
#include <vector>

namespace mm3 {
// Shared by the serving loop and its controlled state fixture. Only verified
// target rows are committed; a replacement token is committed by a later forward.
class SpecState {
public:
    enum class DraftResult { rejected, cancelled, accepted, finished };
    explicit SpecState(std::vector<int>& tokens) : tokens_(tokens) {}
    static size_t common_prefix(const std::vector<int>& state, const std::vector<int>& prompt) {
        size_t n = 0;
        while (n < state.size() && n + 1 < prompt.size() && state[n] == prompt[n]) ++n;
        return n;
    }
    static size_t reusable_prefix(size_t from, bool draft_active, bool draft_available) {
        return draft_active && !draft_available ? 0 : from;
    }
    void commit(int token) { tokens_.push_back(token); }
    template<class Emit> DraftResult draft(bool keep, bool cancelled, int token, Emit emit) {
        if (!keep) return DraftResult::rejected;
        if (cancelled) return DraftResult::cancelled;
        const bool more = emit(token);
        commit(token);
        return more ? DraftResult::accepted : DraftResult::finished;
    }
private:
    std::vector<int>& tokens_;
};
}

namespace mm3 {
// Deliberately changes one acceptance decision only in an explicitly enabled
// diagnostic request. The caller still uses the ordinary replacement sampler.
inline bool state_test_enabled(const char* value) {
    return value && value[0]=='1' && value[1]=='\0';
}
class ForceRejectFirst {
public:
    explicit ForceRejectFirst(bool requested) : requested_(requested) {}
    bool decide(bool keep, bool cancelled, long long emitted, int frontier, long long target_rounds) {
        if (!requested_ || forced || cancelled) return keep;
        forced=1; emitted_at_rejection=emitted; frontier_at_rejection=frontier;
        rounds_at_rejection=target_rounds; natural_accept=keep ? 1 : 0;
        return false;
    }
    int forced=0, frontier_at_rejection=-1, natural_accept=-1;
    long long emitted_at_rejection=-1, rounds_at_rejection=-1;
private:
    bool requested_;
};
}
