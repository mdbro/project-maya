// Production submission/publication/retirement transitions with controlled
// nonblocking CUDA-query results. No model, CUDA driver, or GPU is needed.
#include "../src/mm3/mm3_r2_lifecycle.hpp"
#include <cassert>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
using mm3::r2::Completion;
struct Promo { int slot; };
int main() {
    for (int failure : {1,2}) {
        long long owner=19; unsigned char pending=0; std::string sticky,error;
        assert(!mm3::r2::foreground_issue(owner,pending,sticky,error,[&](std::string& e) {
            e=failure==1?"copy":"event"; return false;
        }));
        assert(owner==-1 && !pending && !sticky.empty());
        assert(!mm3::r2::foreground_issue(owner,pending,sticky,error,[](std::string&){return true;}));
    }
    {
        std::vector<long long> owners{9,20};std::vector<unsigned char> pending{0,0};std::string sticky,error;
        assert(mm3::r2::foreground_issue(owners[0],pending[0],sticky,error,[](std::string&){return true;}));
        owners[0]=42;
        assert(!mm3::r2::foreground_retire(pending,owners,sticky,error,[](size_t,std::string& e){e="query";return Completion::failed;}));
        assert(owners[0]==-1 && owners[1]==20 && pending[0] && sticky=="query");
    }
    {
        // Not-ready foreground records remain unavailable to a later forward,
        // reserved against cross-stream promotions, and queried once each time.
        std::vector<long long> owners{42};std::vector<unsigned char> pending{1};std::string sticky,error;
        int queries=0; bool ready=false;
        for(long long forward=11;forward<=13;++forward) {
            assert(mm3::r2::foreground_retire(pending,owners,sticky,error,[&](size_t,std::string&){++queries;return ready?Completion::ready:Completion::not_ready;}));
            assert(queries==forward-10 && pending[0] && owners[0]==42);
            assert(!mm3::r2::foreground_usable(true,10LL,forward));
            assert(!mm3::r2::snapshot_visible(true));
            assert(!mm3::r2::promotion_slot_available(true,false,10LL,forward));
        }
        // The creating forward may consume the submitted copy on its own stream,
        // but it is not published in the independent routing snapshot.
        assert(mm3::r2::foreground_usable(true,10LL,10LL));
        assert(!mm3::r2::snapshot_visible(true));
        ready=true;
        assert(mm3::r2::foreground_retire(pending,owners,sticky,error,[&](size_t,std::string&){++queries;return Completion::ready;}));
        assert(queries==4 && !pending[0] && owners[0]==42 && mm3::r2::snapshot_visible(false));
    }
    for (int failure : {1,2}) {
        std::vector<Promo> q{{0}};
        std::vector<long long> issued{0}, pending{42}, owner{-1}, used{-1};
        std::string sticky,error; int copies=0, events=0, queries=0, n=1;
        auto submit=[&](const Promo&,std::string& e) {
            ++copies; if(failure==1) {e="copy"; return false;}
            ++events; if(failure==2) {e="event"; return false;}
            return true;
        };
        assert(!mm3::r2::issue(q,issued,sticky,error,submit));
        assert(q.empty() && owner[0]==-1 && pending[0]==42);
        assert(!mm3::r2::publish(pending,issued,owner,used,n,10LL,sticky,error,
                                  [&](size_t,std::string&) { ++queries; return Completion::ready; }));
        assert(queries==0 && owner[0]==-1 && n==1 && copies==1 && events==(failure==2));
    }
    {
        // A stale/initially complete event is not proof that this record was
        // submitted. It must not even be queried before the submission marker.
        std::vector<long long> pending{55},issued{0},owner{-1},used{-1};
        std::string sticky,error; int n=1,queries=0;
        assert(mm3::r2::publish(pending,issued,owner,used,n,1LL,sticky,error,
            [&](size_t,std::string&){++queries;return Completion::ready;}));
        assert(queries==0 && n==1 && pending[0]==55 && owner[0]==-1 && issued[0]==0);
    }
    {
        // Repeated ordinary or grouped boundaries never wait, publish, or reuse
        // the pending slot. Readiness may be observed in the issuing forward.
        std::vector<Promo> q{{0}};
        std::vector<long long> issued{0}, pending{77}, owner{-1}, used{-1};
        std::string sticky,error; int n=1, queries=0; bool ready=false;
        assert(mm3::r2::issue(q,issued,sticky,error,[](const Promo&,std::string&){return true;}));
        for(long long forward : {5LL,6LL,6LL,7LL}) {
            const int before=queries;
            assert(mm3::r2::publish(pending,issued,owner,used,n,forward,sticky,error,
                [&](size_t,std::string&){++queries;return ready?Completion::ready:Completion::not_ready;}));
            assert(queries==before+1 && pending[0]==77 && owner[0]==-1 && n==1);
            assert(!mm3::r2::promotion_slot_available(false,true,0LL,forward));
        }
        ready=true;
        assert(mm3::r2::publish(pending,issued,owner,used,n,7LL,sticky,error,
            [&](size_t,std::string&){++queries;return Completion::ready;}));
        assert(pending[0]==-1 && issued[0]==0 && owner[0]==77 && used[0]==7 && n==0);
        const int before=queries;
        assert(mm3::r2::publish(pending,issued,owner,used,n,7LL,sticky,error,
            [&](size_t,std::string&){++queries;return Completion::ready;}));
        assert(queries==before);
        // Completion can be observed at the very boundary of submission.
        pending[0]=88; owner[0]=-1; issued[0]=1; n=1;
        assert(mm3::r2::publish(pending,issued,owner,used,n,7LL,sticky,error,
            [](size_t,std::string&){return Completion::ready;}));
        assert(owner[0]==88 && n==0);
    }
    {
        std::vector<long long> pending{77}, issued{1}, owner{-1}, used{-1};
        std::string sticky,error; int n=1,queries=0;
        assert(!mm3::r2::publish(pending,issued,owner,used,n,1LL,sticky,error,
            [&](size_t,std::string& e){++queries;e="completion query";return Completion::failed;}));
        assert(owner[0]==-1 && pending[0]==77 && n==1 && sticky=="completion query");
        assert(!mm3::r2::publish(pending,issued,owner,used,n,2LL,sticky,error,
            [&](size_t,std::string&){++queries;return Completion::ready;}));
        assert(queries==1 && owner[0]==-1);
    }
    {
        std::string error;
        std::thread t([&] { mm3::r2::guarded_worker([] { throw std::runtime_error("invalid record"); },
                                                       [&](const std::string& e) { error=e; }); });
        t.join(); assert(error=="invalid record");
    }
    {
        std::vector<int> devices{0,1}; int joins=0; bool released=false;
        bool safe=mm3::r2::retire(false,devices,[&](int d){ ++joins; return d!=0; });
        if(safe) released=true;
        assert(!safe && !released && joins==2);
        assert(!mm3::r2::retire(true,devices,[](int){return true;}));
        assert(mm3::r2::retire(false,devices,[](int){return true;}));
    }
}
