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
// Keep the already-selected device budget, reserve and allocation slack. A candidate must
// eliminate a whole part, retain at least 64 rows of local partials for QKV scratch,
// and allocate no more than 256 MiB of mapped intermediate activations at once.
struct PartShape { int parts,rows,local_rows; };
inline PartShape fewer_parts(int tokens,int fit,size_t free_bytes,size_t dim,size_t job,size_t ready,size_t scratch,size_t retained_jobs) {
    if(tokens<=0 || fit<=0 || dim==0)return {0,0,0};
    const int parts=(tokens+fit-1)/fit;
    const auto shape=[&](int n){return ((tokens+n-1)/n+15)/16*16;};
    const PartShape base{parts,std::min(fit,shape(parts)),std::min(fit,shape(parts))};
    constexpr size_t reserve=150ull<<20,budget=700ull<<20,slack=8ull<<20,host_limit=256ull<<20;
    if(parts<=1 || !dim || free_bytes<=reserve+slack)return base;
    const size_t available=std::min(budget,free_bytes-reserve)-slack,per=8*dim;
    for(int n=std::max(1,parts-std::max(1,parts/3));n<parts;++n){
        const int rows=shape(n);const size_t needed=device_bytes(rows,dim,job,ready,scratch,retained_jobs);
        const size_t excess=needed>available?needed-available:0;
        const size_t spilled=((excess+16*per-1)/(16*per))*16;
        if(spilled>size_t(rows) || rows-int(spilled)<64 || spilled*per>host_limit)continue;
        return {n,rows,rows-int(spilled)};
    }
    return base;
}
} // namespace mm3::prompt_memory
