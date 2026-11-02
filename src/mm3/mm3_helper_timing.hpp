#pragma once
#include "mm3_native_helper.hpp"
#include <algorithm>
#include <initializer_list>
#include <utility>
#include <vector>

namespace mm3 {
inline uint64_t helper_interval_union_ns(std::initializer_list<std::pair<uint64_t,uint64_t>> ranges) {
    std::vector<std::pair<uint64_t,uint64_t>> ordered(ranges);
    std::sort(ordered.begin(),ordered.end());
    uint64_t end=0,total=0;
    for(auto [a,b]:ordered)if(a && b>a && b>end) {total+=b-std::max(a,end);end=b;}
    return total;
}
// Conditional wall-time ceiling for removing this pass's host handoff while
// leaving its work graph, GPU service and resource contention unchanged.
// Main-thread preparation/reduction are charged in full, even when they overlap
// GPU work or are unavoidable. Worker delays can affect this caller only until
// collect finishes waiting: cap their union by that actual blocking duration.
// This does not bound indirect CPU contention or an ahead-of-execution work
// graph, and cannot alone prune that architecture opportunity.
inline uint64_t helper_dispatch_upper_ns(const NativeHelper::PassTiming& t,
        uint64_t prepare_begin,uint64_t submit_end,uint64_t collect_begin,uint64_t merge_end) {
    const uint64_t lower=t.last_not_ready_ns?t.last_not_ready_ns:t.event_record_begin_ns;
    const uint64_t wake=std::max(t.worker_published_ns,t.collect_wait_begin_ns);
    const uint64_t host=helper_interval_union_ns({{prepare_begin,submit_end},
        {collect_begin,t.collect_wait_begin_ns},{t.collect_wait_end_ns,merge_end}});
    const uint64_t worker=helper_interval_union_ns({{t.enqueue_ns,t.worker_start_ns},
        {t.worker_start_ns,t.event_record_end_ns},{lower,t.ready_observed_ns},
        {t.ready_observed_ns,t.worker_published_ns},{wake,t.collect_wait_end_ns}});
    const uint64_t wait=t.collect_wait_end_ns>=t.collect_wait_begin_ns?
        t.collect_wait_end_ns-t.collect_wait_begin_ns:0;
    return host+std::min(worker,wait);
}
}
