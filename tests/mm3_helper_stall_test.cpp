// Compiles the production helper with a controlled CUDA API and short test deadline.
#include "../src/mm3/mm3_native_helper.hpp"
#include "cuda_runtime_api.h"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include <cassert>
#include <chrono>
#include <cstdio>
#include <vector>
namespace infernix::ops::minimax_ffn {
size_t record_bytes(int,int) { return 144; }
size_t record_stride(int,int) { return 144; }
size_t fast_workspace_bytes(int,int,int) { return 256; }
void fast_ffn_cuda_packed(const FastJob*,int,int,int,int,void*,cudaStream_t) {}
}
int main() {
    using H=mm3::NativeHelper;
    std::string error;
    auto* helper=new H(0,1);
    unsigned char record[144]{};
    const float scales[3]={1,1,1};
    assert(helper->add_expert(0,0,record,sizeof(record),scales,scales,error));
    assert(helper->seal(error));
    H::Job job{}; job.layer=0; job.expert=0; job.columns=1; job.weight[0]=1;
    std::vector<uint16_t> x(H::kHidden);
    std::vector<float> out(H::kHidden);
    fake_stall=true;
    auto seq=helper->submit(1,x.data(),{job},error); assert(seq);
    auto start=std::chrono::steady_clock::now();
    assert(!helper->collect(seq,out.data(),out.size(),error));
    assert(error.find("deadline")!=std::string::npos);
    assert(std::chrono::steady_clock::now()-start<std::chrono::seconds(2));
    assert(fake_queries>0);
    assert(helper->ready(0)[0]==0);
    assert(!helper->submit(1,x.data(),{job},error));
    delete helper;
    // An unretired stream must not lose any GPU-visible allocation.
    assert(fake_allocations>0 && fake_frees==0);
    // Shutdown itself must also be bounded with a queued stalled request.
    fake_stall=false;
    auto* shutdown_helper=new H(0,1);
    assert(shutdown_helper->add_expert(0,0,record,sizeof(record),scales,scales,error));
    assert(shutdown_helper->seal(error));
    fake_stall=true;
    assert(shutdown_helper->submit(1,x.data(),{job},error));
    start=std::chrono::steady_clock::now();
    delete shutdown_helper;
    assert(std::chrono::steady_clock::now()-start<std::chrono::seconds(2));
    assert(fake_frees==0);
    // A CUDA API call itself can stall: shutdown detaches only after retaining
    // the complete state the worker still references.
    fake_stall=false;
    auto* blocked_helper=new H(0,1);
    assert(blocked_helper->add_expert(0,0,record,sizeof(record),scales,scales,error));
    assert(blocked_helper->seal(error));
    fake_stall=true; fake_block_query=true;
    const int queries_before=fake_queries;
    assert(blocked_helper->submit(1,x.data(),{job},error));
    const auto wait_until=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    while (fake_queries==queries_before && std::chrono::steady_clock::now()<wait_until) std::this_thread::yield();
    assert(fake_queries>queries_before);
    start=std::chrono::steady_clock::now();
    delete blocked_helper;
    assert(std::chrono::steady_clock::now()-start<std::chrono::seconds(2));
    assert(fake_frees==0);
    fake_block_query=false; fake_stall=false;
    // A blocking CUDA stream query during cleanup must not hold the caller.
    auto* cleanup_helper=new H(0,1);
    assert(cleanup_helper->seal(error));
    fake_block_stream=true;
    std::atomic<bool> cleanup_done{false};
    std::thread cleanup([&] { delete cleanup_helper; cleanup_done=true; });
    const auto cleanup_deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(1500);
    while (!cleanup_done && std::chrono::steady_clock::now()<cleanup_deadline) std::this_thread::yield();
    const bool bounded=cleanup_done;
    fake_block_stream=false;
    cleanup.join();
    assert(bounded);
    std::puts("production helper stalled completion: OK");
}
