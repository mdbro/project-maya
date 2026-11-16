#include "../src/mm3/mm3_shared_frames.hpp"
#include "../src/mm3/mm3_background_traffic.hpp"
#include <array>
#include <cassert>
#include <thread>
#include <vector>

int main() {
    std::array<unsigned char,512> bytes{};mm3::SharedFrames frames;frames.initialize(bytes.data(),bytes.size(),128);
    std::string error;assert(frames.assign(0,1042,error));assert(frames.assign(1,2077,error));
    auto source=frames.take(0,1042),landing=frames.take_free();assert(source && landing && landing.frame==2);
    assert(!frames.take(0,1042) && !frames.assign(0,999,error));
    assert(frames.index(bytes.data()+1)==-1 && frames.index(bytes.data()+512)==-1);
    assert(frames.finish(landing,3111,error));assert(frames.finish(source,-1,error));
    auto next=frames.take_free();assert(next.frame==0 && next.generation!=source.generation);
    assert(!frames.finish(source,999,error) && frames.current(next));
    assert(frames.finish(next,-1,error) && !frames.busy());
    assert((frames.roster()==std::vector<long long>{-1,2077,3111,-1}));

    // Both helpers and HOST work can be admitted together within the shared
    // byte ceiling. An older one-card-only cadence cannot satisfy this case.
    mm3::BackgroundTraffic budget;
    assert(budget.acquire(0,1u<<20));assert(budget.acquire(1,128u<<10));assert(budget.acquire(2,512u<<10));
    assert(!budget.acquire(1,1u<<20));assert(budget.used()==(1664u<<10));
    budget.release(2,512u<<10);budget.release(0,1u<<20);budget.release(1,128u<<10);
    assert(!budget.used());
    std::array<std::thread,3> workers;
    for(unsigned a=0;a<workers.size();++a)workers[a]=std::thread([&,a] {
        for(int i=0;i<1000;++i) {
            while(!budget.acquire(a,1u<<20))std::this_thread::yield();
            assert(budget.used()<=budget.capacity());budget.release(a,1u<<20);
        }
    });
    for(auto& w:workers)w.join();assert(!budget.used() && budget.high()<=budget.capacity());
    assert(budget.acquire(0,128));budget.poison();budget.release(0,128);
    assert(!budget.healthy() && !budget.used() && !budget.acquire(0,1));
}
