#include "../src/mm3/mm3_spec_state.hpp"
#include "../src/mm3/mm3_draft_ring.hpp"
#include <cassert>
#include <vector>
using mm3::SpecState;
using Result = SpecState::DraftResult;

// Position-tagged test double for target KV and indexer readers. It checks the
// production frontier decisions, not CUDA storage or kernel execution.
struct Target {
    std::vector<int> kv, index;
    void write(int pos, const std::vector<int>& ids) {
        if (kv.size() < (size_t)(pos + ids.size())) { kv.resize(pos + ids.size(), -1); index.resize(pos + ids.size(), -1); }
        for (size_t i=0;i<ids.size();++i) kv[pos+i]=index[pos+i]=ids[i];
    }
    void read_prefix(const std::vector<int>& state) const {
        for (size_t i=0;i<state.size();++i) assert(kv[i]==state[i] && index[i]==state[i]);
    }
};
int main() {
    assert(!mm3::state_test_enabled(nullptr));
    assert(!mm3::state_test_enabled("0"));
    assert(!mm3::state_test_enabled("true"));
    assert(!mm3::state_test_enabled("11"));
    assert(mm3::state_test_enabled("1"));
    mm3::ForceRejectFirst disabled(false), forced(true);
    assert(disabled.decide(true,false,4,10,2) && disabled.forced==0);
    assert(forced.decide(true,true,4,10,2) && forced.forced==0);
    assert(!forced.decide(true,false,5,11,3));
    assert(forced.forced==1 && forced.natural_accept==1 && forced.emitted_at_rejection==5 &&
           forced.frontier_at_rejection==11 && forced.rounds_at_rejection==3);
    assert(forced.decide(true,false,6,12,4) && forced.forced==1);
    std::vector<int> state{1,2,3}, emitted;
    SpecState frontier(state);
    Target target;
    mm3::DraftRing ring; ring.reset(12,8);
    target.write(0,state); ring.context(0,3);
    auto emit=[&](int t) { emitted.push_back(t); return true; };
    // Target verifies the anchor and a speculative suffix; reject its first
    // draft. Only the already-emitted anchor is retained.
    target.write(3,{4,90,91}); ring.proposal(3,3); ring.context(3,3);
    frontier.commit(4);
    assert(frontier.draft(false,false,90,emit)==Result::rejected);
    assert((state==std::vector<int>{1,2,3,4}) && emitted.empty());
    target.read_prefix(state); assert(ring.available(4));
    // Continue with the replacement, overwriting the rejected target row.
    target.write(4,{5,6,7}); ring.proposal(4,3); ring.context(4,3);
    frontier.commit(5);
    assert(frontier.draft(true,false,6,emit)==Result::accepted);
    assert((emitted==std::vector<int>{6})); target.read_prefix(state);
    std::vector<int> prompt{1,2,3,4,5,6,8};
    auto from=SpecState::common_prefix(state,prompt);
    assert(from==6 && SpecState::reusable_prefix(from,true,ring.available((int)from))==6);
    // Wrap, then try an older matching branch. Full replay is mandatory.
    target.write(6,{7,8,9,10,11,12,13,14,15,16,17,18}); ring.context(6,12);
    state.insert(state.end(),{7,8,9,10,11,12,13,14,15,16,17,18});
    std::vector<int> old{1,2,3,4,99};
    from=SpecState::common_prefix(state,old); assert(from==4);
    from=SpecState::reusable_prefix(from,true,ring.available((int)from)); assert(from==0);
    state.resize(from); target.write(0,old); ring.context(0,(int)old.size()); state=old;
    target.read_prefix(state); assert(ring.available((int)state.size()));
    // EOS or output limit emits and commits the accepted row, cancellation
    // emits and commits nothing. The caller owns the reason for emit=false.
    for (int reason=0;reason<2;++reason) {
        std::vector<int> s{1}; SpecState f(s); int count=0;
        target.write(1,{2});
        assert(f.draft(true,false,2,[&](int){++count;return false;})==Result::finished);
        assert(count==1 && s.size()==2); target.kv[0]=target.index[0]=1; target.read_prefix(s);
    }
    std::vector<int> s{1}; SpecState f(s); int count=0;
    assert(f.draft(true,true,2,[&](int){++count;return true;})==Result::cancelled);
    assert(count==0 && s.size()==1);
}
