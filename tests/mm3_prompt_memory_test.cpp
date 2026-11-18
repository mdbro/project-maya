#include "../src/mm3/mm3_prompt_memory.hpp"
#include <cassert>
#include <cstdio>
using namespace mm3::prompt_memory;
int main() {
    constexpr size_t MiB=1ull<<20,dim=6144,job=48,ready=128,scratch=64*MiB;
    const size_t rows=2048;
    const size_t exact=rows*dim*(4+4+2+4*2)+rows*(4+16+4+512+16+16+16)+(rows/2+256)*job+ready+scratch;
    assert(device_bytes(rows,dim,job,ready,scratch)==exact);
    for(size_t free:{200*MiB,400*MiB,850*MiB}) {
        const int fit=rows_fit(free,dim,job,ready,scratch,4096,700*MiB,150*MiB);
        if(fit) {
            const size_t available=std::min(700*MiB,free-150*MiB)-8*MiB;
            assert(device_bytes(fit,dim,job,ready,scratch,4096)<=available);
            assert(device_bytes(fit+16,dim,job,ready,scratch,4096)>available);
        }
    }
    assert(rows_fit(149*MiB,dim,job,ready,scratch,0,700*MiB,150*MiB)==0);
    assert(rows_fit(800*MiB,dim,job,ready,800*MiB,0,700*MiB,150*MiB)==0);
    const size_t free_bytes=1600*MiB;
    const int fit=rows_fit(free_bytes,dim,job,ready,scratch,4096,700*MiB,150*MiB);
    assert(fewer_parts(100,0,free_bytes,dim,job,ready,scratch,4096).parts==0);
    for(int tokens:{3981,7986,32337}){
        const auto s=fewer_parts(tokens,fit,free_bytes,dim,job,ready,scratch,4096);
        const size_t tail=size_t(s.rows-s.local_rows)*8*dim;
        assert(s.parts==(tokens+s.rows-1)/s.rows && s.local_rows>=64 && s.local_rows%16==0);
        assert(device_bytes(s.rows,dim,job,ready,scratch,4096)-tail<=692*MiB && tail<=256*MiB);
        if(tokens<=fit)assert(!tail && s.parts==1);else assert(tail && s.parts<(tokens+fit-1)/fit);
    }
    std::puts("mm3_prompt_memory_test: exact allocations and capacity bounds PASS");
}
