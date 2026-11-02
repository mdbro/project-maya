#include "mm3_helper_timing.hpp"
#include <cassert>
int main() {
    using namespace mm3;
    assert(helper_interval_union_ns({{10,30},{20,40},{50,55},{0,99},{60,59}})==35);
    NativeHelper::PassTiming t{};
    t.enqueue_ns=110;t.worker_start_ns=140;t.event_record_begin_ns=150;t.event_record_end_ns=160;
    t.last_not_ready_ns=200;t.ready_observed_ns=210;t.worker_published_ns=215;
    // Helper finished before CPU service reaches collect: none of its worker
    // handoff latency can be exposed, even though the raw union is nonzero.
    t.collect_wait_begin_ns=300;t.collect_wait_end_ns=300;
    assert(helper_dispatch_upper_ns(t,100,120,290,310)==40);
    // A short residual wait caps the larger earlier worker handoff union.
    t.collect_wait_end_ns=305;
    assert(helper_dispatch_upper_ns(t,100,120,290,315)==45);
    // A long collect wait includes required GPU service. Only the host upper
    // union is removable, so the entire wait must not be priced as dispatch.
    t.collect_wait_begin_ns=170;t.collect_wait_end_ns=220;
    assert(helper_dispatch_upper_ns(t,100,120,165,230)==85);
    // No observed not-ready poll uses the conservative record-start bracket.
    t.last_not_ready_ns=0;
    assert(helper_dispatch_upper_ns(t,100,120,165,230)==85);
}
