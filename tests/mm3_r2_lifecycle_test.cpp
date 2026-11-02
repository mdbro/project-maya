// Exercises the same submission, publication, worker exception, and retirement
// transitions called by mm3_model.cu, with controlled operation failures.
#include "../src/mm3/mm3_r2_lifecycle.hpp"
#include <cassert>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
struct Promo { int slot; };
int main() {
    for (int failure : {1,2}) { // foreground copy or its event submission
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
        owners[0]=42; // same-stream consumers may use the submitted record
        assert(!mm3::r2::foreground_retire(pending,owners,sticky,error,[](size_t,std::string& e){e="join";return false;}));
        assert(owners[0]==-1 && owners[1]==20 && pending[0] && sticky=="join");
    }
    {
        std::vector<long long> owners{42};std::vector<unsigned char> pending{1};std::string sticky,error;
        assert(mm3::r2::foreground_retire(pending,owners,sticky,error,[](size_t,std::string&){return true;}));
        assert(owners[0]==42 && !pending[0]);
    }
    for (int failure : {1,2}) { // copy fails, then event record fails
        std::vector<Promo> q{{0}};
        std::vector<long long> issued{0}, pending{42}, owner{-1}, used{-1};
        std::string sticky,error; int copies=0, events=0, waits=0, n=1;
        auto submit=[&](const Promo&,std::string& e) {
            ++copies; if(failure==1) {e="copy"; return false;}
            ++events; if(failure==2) {e="event"; return false;}
            return true;
        };
        assert(!mm3::r2::issue(q,issued,1LL,sticky,error,submit));
        assert(q.empty() && owner[0]==-1 && pending[0]==42);
        assert(!mm3::r2::publish(pending,issued,owner,used,n,10LL,0,sticky,error,
                                  [&](size_t,std::string&) { ++waits; return true; }));
        assert(waits==0 && owner[0]==-1 && n==1 && copies==1);
        assert(events==(failure==2));
    }
    {
        std::vector<Promo> q{{0}};
        std::vector<long long> issued{0}, pending{77}, owner{-1}, used{-1};
        std::string sticky,error; int n=1;
        assert(mm3::r2::issue(q,issued,1LL,sticky,error,[](const Promo&,std::string&){return true;}));
        assert(!mm3::r2::publish(pending,issued,owner,used,n,1LL,0,sticky,error,
                                 [](size_t,std::string& e){e="completion"; return false;}));
        assert(owner[0]==-1 && pending[0]==77 && n==1 && sticky=="completion");
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
        assert(!safe && !released && joins==2); // still attempts every join
        assert(!mm3::r2::retire(true,devices,[](int){return true;}));
        assert(mm3::r2::retire(false,devices,[](int){return true;}));
    }
}
