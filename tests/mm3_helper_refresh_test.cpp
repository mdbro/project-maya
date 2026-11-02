#include "../src/mm3/mm3_native_helper.hpp"
#include "cuda_runtime_api.h"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstring>
#include <string>
#include <vector>
namespace infernix::ops::minimax_ffn {
size_t record_bytes(int,int) { return 144; }
size_t record_stride(int,int) { return 144; }
size_t fast_workspace_bytes(int,int,int) { return 256; }
void fast_ffn_cuda_packed(const FastJob*,int,int,int,int,void*,cudaStream_t) {}
}
using H=mm3::NativeHelper;
static std::vector<unsigned char> record(unsigned char code) {
    std::vector<unsigned char> x(144,code);
    std::fill(x.begin()+128,x.end(),0x38); return x;
}
static H* setup() {
    auto* h=new H(0,2);
    const float w[3]={1,2,3}, in[3]={4,5,6};
    auto a=record(0x11), b=record(0x22); std::string err;
    assert(h->add_expert(1,3,a.data(),a.size(),w,in,err));
    assert(h->add_expert(2,4,b.data(),b.size(),w,in,err));
    assert(h->seal(err)); return h;
}
int main() {
    std::string err; bool unsafe=false; float w[3],in[3];
    {
        auto* h=setup(); const uintptr_t pool=h->pool_identity();
        auto first=record(0), replacement=record(0x33);
        fake_not_portable=true;
        assert(!h->export_expert(1,3,first.data(),first.size(),w,in,unsafe,err));
        assert(!unsafe && h->ready(1)[0]==(1u<<3));
        fake_not_portable=false;
        // A healthy event may need several immediate polls before completion.
        const int queries_before=fake_queries;
        fake_not_ready_queries=3;
        assert(h->export_expert(1,3,first.data(),first.size(),w,in,unsafe,err));
        assert(fake_not_ready_queries==0 && fake_queries-queries_before>=4);
        assert(!unsafe && first==record(0x11));
        assert(w[0]==1 && w[1]==2 && w[2]==3 && in[0]==4 && in[1]==5 && in[2]==6);
        const float nw[3]={7,8,9},ni[3]={1,3,5};
        assert(h->replace_expert(1,3,5,6,replacement.data(),replacement.size(),nw,ni,unsafe,err));
        assert(!unsafe && pool==h->pool_identity());
        auto ids=h->records(); assert(ids.size()==2);
        assert(h->ready(1)[0]==0 && h->ready(5)[0]==(1u<<6) && h->ready(2)[0]==(1u<<4));
        assert(h->export_expert(5,6,first.data(),first.size(),w,in,unsafe,err));
        assert(first==replacement && w[0]==7 && w[2]==9 && in[1]==3);
        assert(h->export_expert(2,4,first.data(),first.size(),w,in,unsafe,err));
        assert(first==record(0x22));
        // Reject maintenance while any request remains active, including one
        // whose GPU work has already finished but has not been collected.
        std::vector<uint16_t> x(H::kHidden); H::Job j{};j.layer=2;j.expert=4;j.columns=1;j.weight[0]=1;
        auto seq=h->submit(1,x.data(),{j},err); assert(seq);
        assert(!h->export_expert(2,4,first.data(),first.size(),w,in,unsafe,err));
        assert(!h->replace_expert(2,4,7,7,replacement.data(),replacement.size(),nw,ni,unsafe,err));
        assert(err.find("active")!=std::string::npos && !unsafe);
        std::vector<float> out(H::kHidden); assert(h->collect(seq,out.data(),out.size(),err));
        delete h;
    }
    // Each failed transfer is isolated in a fresh helper. Fake stream remains
    // unretired, so both the pool and borrowed staging must be retained.
    for (int fault=0;fault<4;++fault) {
        auto* h=setup(); auto* stage=new std::vector<unsigned char>(record(0x33)); const float scale[3]={1,1,1};
        const int frees_before=fake_frees;
        fake_stall=true;
        fake_fail_copy=(fault==0); fake_fail_record=(fault==1); fake_fail_query=(fault==2);
        assert(!h->replace_expert(1,3,5,6,stage->data(),stage->size(),scale,scale,unsafe,err));
        assert(unsafe && h->ready(5)[0]==0 && h->records().empty());
        delete h;
        assert(fake_frees==frees_before && stage->at(0)==0x33);
        // stage deliberately remains allocated while its CUDA reader is uncertain.
        fake_stall=false; fake_fail_copy=false; fake_fail_record=false; fake_fail_query=false;
    }
    // A CUDA query that never returns must not prevent the caller or helper
    // shutdown from returning; the worker and its borrowed source outlive both.
    {
        auto* h=setup(); auto* stage=new std::vector<unsigned char>(record(0x44));
        const float scale[3]={1,1,1};
        fake_stall=true; fake_block_query=true;
        const auto started=std::chrono::steady_clock::now();
        assert(!h->replace_expert(1,3,5,6,stage->data(),stage->size(),scale,scale,unsafe,err));
        assert(unsafe && std::chrono::steady_clock::now()-started<std::chrono::seconds(2));
        const auto shutdown_started=std::chrono::steady_clock::now();
        const int frees_before=fake_frees;
        delete h;
        assert(std::chrono::steady_clock::now()-shutdown_started<std::chrono::seconds(2));
        assert(fake_frees==frees_before && stage->at(0)==0x44);
        fake_block_query=false; fake_stall=false;
    }

}
