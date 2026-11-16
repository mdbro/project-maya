// Exact production admission protocol, with controlled event outcomes and
// deterministic producer/consumer interleavings. CPU only; no CUDA dependency.
#include "../src/mm3/mm3_upload_guard.hpp"
#include <array>
#include <atomic>
#include <cassert>
#include <cstdio>
#include <thread>

using G=mm3::UploadGuard;
struct Event {G::Query state=G::Query::ready;};
static G::Events handles(std::array<Event,4>& events) {
    return {&events[0],&events[1],&events[2],&events[3]};
}
static G::Query query(void* event){return static_cast<Event*>(event)->state;}
int main() {
    G g;std::array<Event,4> events{};
    int queries=0;
    auto unrestricted=g.claim([&](void*){++queries;return G::Query::ready;});
    assert(unrestricted.result==G::Result::unrestricted && !queries && g.current(unrestricted.token));
    g.release(unrestricted.token);
    auto generation=g.revoke();
    assert(!generation.admitted && g.claim(query).result==G::Result::closed);
    assert(g.publish(generation.token,handles(events)));
    auto claim=g.claim(query);assert(claim.result==G::Result::admitted && g.current(claim.token));
    assert(g.claim(query).result==G::Result::closed); // one optional chunk at a time
    g.release(claim.token);

    // Any one of the four stream markers blocks admission independently.
    for(std::size_t i=0;i<events.size();++i) {
        generation=g.revoke();events[i].state=G::Query::pending;
        assert(g.publish(generation.token,handles(events)));
        claim=g.claim(query);assert(claim.result==G::Result::pending && claim.event==i);
        events[i].state=G::Query::ready;claim=g.claim(query);
        assert(claim.result==G::Result::admitted);g.release(claim.token);
    }
    // An event error closes/invalidate the generation; its old producer token
    // cannot reopen it even if the fake event subsequently appears complete.
    for(std::size_t i=0;i<events.size();++i) {
        generation=g.revoke();events[i].state=G::Query::error;
        assert(g.publish(generation.token,handles(events)));
        claim=g.claim(query);assert(claim.result==G::Result::error && claim.event==i);
        events[i].state=G::Query::ready;
        assert(g.claim(query).result==G::Result::closed);
        assert(!g.publish(generation.token,handles(events)));
    }
    // Query returns success for an old event recording after the producer has
    // revoked it. It must not claim the new (unpublished) generation.
    generation=g.revoke();assert(g.publish(generation.token,handles(events)));
    G::Revocation next;
    queries=0;
    claim=g.claim([&](void* event){if(++queries==4)next=g.revoke();return query(event);});
    assert(claim.result==G::Result::stale && !next.admitted);
    assert(!g.publish(generation.token,handles(events)));
    assert(g.claim(query).result==G::Result::closed);
    assert(g.publish(next.token,handles(events)));

    // A claim which preceded revocation is distinguishable from a late claim.
    // Releasing that old chunk cannot open the newer closed generation.
    claim=g.claim(query);assert(claim.result==G::Result::admitted);
    next=g.revoke();assert(next.admitted && !g.current(claim.token));
    g.release(claim.token);assert(g.claim(query).result==G::Result::closed);
    assert(g.publish(next.token,handles(events)));
    claim=g.claim(query);assert(claim.result==G::Result::admitted);
    const auto disabled=g.disable();assert(disabled.admitted);
    g.release(claim.token);unrestricted=g.claim(query);assert(unrestricted.result==G::Result::unrestricted);
    // Enabling the guard can distinguish/cancel an earlier unrestricted claim.
    next=g.revoke();assert(next.admitted && !g.current(unrestricted.token));
    g.release(unrestricted.token);assert(g.claim(query).result==G::Result::closed);
    assert(!g.publish(disabled.token,handles(events)));

    // Real threads, bounded to one controlled interleaving. The consumer holds
    // a query of retained handles while the producer revokes/re-records them.
    generation=g.revoke();assert(g.publish(generation.token,handles(events)));
    std::atomic<bool> queried{false},resume{false};G::Claim threaded;
    std::thread worker([&]{
        int n=0;
        threaded=g.claim([&](void* event){
            const auto value=query(event); // before re-record, under synchronization below
            if(++n==4){queried.store(true,std::memory_order_release);while(!resume.load(std::memory_order_acquire))std::this_thread::yield();}
            return value;
        });
    });
    while(!queried.load(std::memory_order_acquire))std::this_thread::yield();
    next=g.revoke();events[1].state=G::Query::pending;
    assert(g.publish(next.token,handles(events)));resume.store(true,std::memory_order_release);worker.join();
    assert(threaded.result==G::Result::stale);
    assert(g.claim(query).result==G::Result::pending);
    events[1].state=G::Query::ready;claim=g.claim(query);assert(claim.result==G::Result::admitted);g.release(claim.token);
    std::puts("mm3_upload_guard_test: ok");
}
