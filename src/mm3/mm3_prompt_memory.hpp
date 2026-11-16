#pragma once
#include <algorithm>
#include <cstddef>

namespace mm3::prompt_memory {
// Exact buffers allocated by prompt_dev_alloc. Job capacity can outlive a smaller later prompt on the host.
inline size_t device_bytes(size_t rows,size_t dim,size_t job_bytes,size_t ready_bytes,size_t scratch,size_t retained_jobs=0) {
    const size_t row_bytes=18*dim+4+16+4+128*4+16+16+16;
    return rows*row_bytes+std::max(retained_jobs,rows/2+256)*job_bytes+ready_bytes+scratch;
}
inline int rows_fit(size_t free_bytes,size_t dim,size_t job_bytes,size_t ready_bytes,size_t scratch,size_t retained_jobs,
                    size_t budget,size_t reserve) {
    constexpr size_t slack=8ull<<20;
    if(free_bytes<=reserve || std::min(budget,free_bytes-reserve)<=slack)return 0;
    const size_t available=std::min(budget,free_bytes-reserve)-slack;
    int lo=0,hi=60000/16;
    while(lo<hi) {
        const int mid=(lo+hi+1)/2;
        if(device_bytes(size_t(mid)*16,dim,job_bytes,ready_bytes,scratch,retained_jobs)<=available)lo=mid;
        else hi=mid-1;
    }
    return lo*16>=64?lo*16:0;
}
} // namespace mm3::prompt_memory
