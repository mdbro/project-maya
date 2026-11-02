#pragma once
#include <algorithm>

namespace mm3 {
// Stage verifier segments have up to four rows; all stage buffers are chunk-sized.
constexpr bool native_stage_chunk_supported(int chunk) {return chunk>=4 && chunk<=16;}
// The real A/B submission order. Ops owns the device operations, counters,
// failure poisoning and final retirement; no operation may reuse a handoff
// until its producer and consumer have retired.
template<class Ops>
bool native_stage_schedule(int n, bool overlap, Ops& op) {
    if(n<1 || n>8)return false;
    const int a=std::min(4,n), b=n-a;
    if(!op.prefix(0,0,a) || !op.await_a())return false;
    if(overlap && b) {
        op.begin_prefix_b();
        if(!op.prefix(1,a,b))return false;
        op.submitted_prefix_b(b);
        if(!op.observe_b_start())return false;
    }
    op.begin_suffix_a();
    if(!op.suffix(0,0,a))return false;
    op.end_suffix_a();
    if(overlap && b && !op.observe_b_end())return false;
    if(b && op.continue_b(a)) {
        if(!overlap) {
            if(!op.prefix(1,a,b))return false;
            op.serial_prefix_b(b);
        }
        if(!op.suffix(1,a,b))return false;
        op.completed_b(b);
    } else if(b) op.skipped_b(b,overlap);
    return true;
}
}
