// Actual DSpark, target attention and indexer readers against a position-tagged
// CPU oracle after wrap, rejection/continuation and backward-branch replay.
// No model or weights.
// Build: nvcc -O2 -std=c++17 -gencode arch=compute_86,code=sm_86 \
//   -gencode arch=compute_120,code=sm_120 -Isrc/mm3 tests/mm3_draft_reader_test.cu \
//   src/mm3/mm3_kernels.cu -o /tmp/mm3_draft_reader_test
// Run: /tmp/mm3_draft_reader_test
#include "mm3_draft_ring.hpp"
#include "mm3_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace k = mm3::k;
#define CUDA(call) do { const auto e=(call); if(e!=cudaSuccess) { \
    std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e)); std::exit(1); } } while(0)
template<class T> T* device(size_t n) { T* p=nullptr; CUDA(cudaMalloc(&p,n*sizeof(T))); return p; }
int main() {
    constexpr int window=8, ring=window+64, block=3, heads=16, hd=128;
    mm3::DraftRing tracker; tracker.reset(ring,window);
    std::vector<int> tags(ring,-1);
    auto context=[&](int start,int count) {
        tracker.context(start,count);
        for(int p=start;p<start+count;++p) tags[p%ring]=p;
    };
    auto proposal=[&](int start) {
        tracker.proposal(start,block);
        for(int p=start;p<start+block;++p) tags[p%ring]=p;
    };
    auto* q=device<float>(block*heads*hd);
    auto* key=device<uint16_t>(ring*hd);
    auto* value=device<uint16_t>(ring*hd);
    auto* anchor=device<uint16_t>(hd);
    auto* mask=device<float>(hd);
    auto* h=device<float>(block*hd);
    auto* positions=device<int>(block);
    auto* lo=device<int>(block);
    auto* output=device<float>(block*heads*hd);
    auto* part=device<float>(k::attn_part_floats(block,1,16));
    CUDA(cudaMemset(q,0,block*heads*hd*sizeof(float)));
    CUDA(cudaMemset(key,0,ring*hd*sizeof(uint16_t)));
    CUDA(cudaMemset(anchor,0,hd*sizeof(uint16_t)));
    CUDA(cudaMemset(mask,0,hd*sizeof(float)));
    auto check=[&](int pos) {
        // Each retained physical row's BF16 vector has exactly one bit of
        // mass, at its logical-position index. Zero queries give uniform
        // attention, revealing the actual positions read without copying a
        // second implementation of the reader into the fixture.
        std::vector<uint16_t> v(ring*hd,0);
        for(int slot=0;slot<ring;++slot) if(tags[slot]>=0) {
            assert(tags[slot]<hd); v[slot*hd+tags[slot]]=0x3f80;
        }
        CUDA(cudaMemcpy(value,v.data(),v.size()*sizeof(uint16_t),cudaMemcpyHostToDevice));
        k::draft_block(anchor,mask,hd,block,pos,window,h,positions,lo,nullptr);
        for(int splits : {1,16}) {
            k::window_attn(q,key,value,ring,lo,pos+block-1,block,heads,1,1.0f,
                           output,nullptr,splits==1?nullptr:part,splits);
            CUDA(cudaDeviceSynchronize());
            std::vector<float> actual(block*heads*hd);
            CUDA(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
            for(int t=0;t<block;++t) {
                const int first=std::max(0,pos+t-window+1), last=pos+block-1;
                const float expected=1.0f/float(last-first+1);
                for(int head=0;head<heads;++head) for(int logical=0;logical<hd;++logical) {
                    const float want=logical>=first && logical<=last ? expected : 0.0f;
                    const float got=actual[(t*heads+head)*hd+logical];
                    if(!std::isfinite(got) || std::fabs(got-want)>1e-5f) {
                        std::fprintf(stderr,"pos=%d row=%d head=%d logical=%d splits=%d got=%g want=%g\n",
                                     pos,t,head,logical,splits,got,want); std::exit(1);
                    }
                }
            }
        }
    };
    context(0,80); // wrap the physical ring
    assert(tracker.available(80));
    proposal(80); check(80);
    // Reject the first draft: only the verified anchor becomes context.
    context(80,1); assert(tracker.available(81));
    proposal(81); check(81);
    // A matching old prefix cannot reuse the wrapped slots; replay it first.
    assert(!tracker.available(5));
    context(0,5); assert(tracker.available(5));
    proposal(5); check(5);
    // Actual target sparse-attention and indexer block-score readers use
    // position-causal rows, including after a rejected row is overwritten.
    {
        constexpr int rows=128, blk=64;
        auto* target_key=device<uint16_t>(rows*hd);
        auto* target_value=device<uint16_t>(rows*hd);
        auto* index_q=device<float>(hd);
        auto* scores=device<float>(2);
        auto* kept=device<int>(2);
        CUDA(cudaMemset(target_key,0,rows*hd*sizeof(uint16_t)));
        std::vector<int> target_tags(rows);
        for(int i=0;i<rows;++i) target_tags[i]=i;
        const int blocks[2]={0,1};
        CUDA(cudaMemcpy(kept,blocks,sizeof(blocks),cudaMemcpyHostToDevice));
        auto target_check=[&](int pos) {
            std::vector<uint16_t> v(rows*hd,0);
            for(int row=0;row<rows;++row) v[row*hd+target_tags[row]]=0x3f80;
            CUDA(cudaMemcpy(target_value,v.data(),v.size()*sizeof(uint16_t),cudaMemcpyHostToDevice));
            CUDA(cudaMemcpy(positions,&pos,sizeof(pos),cudaMemcpyHostToDevice));
            k::sparse_attn(q,target_key,target_value,kept,positions,1,heads,1,2,blk,1.0f,output,nullptr);
            std::vector<float> index_values(hd);
            for(int i=0;i<hd;++i) index_values[i]=float(i);
            CUDA(cudaMemcpy(index_q,index_values.data(),hd*sizeof(float),cudaMemcpyHostToDevice));
            k::block_scores(index_q,target_value,positions,1,1,2,blk,scores,nullptr);
            CUDA(cudaDeviceSynchronize());
            std::vector<float> actual(heads*hd), observed_scores(2);
            CUDA(cudaMemcpy(actual.data(),output,actual.size()*sizeof(float),cudaMemcpyDeviceToHost));
            CUDA(cudaMemcpy(observed_scores.data(),scores,2*sizeof(float),cudaMemcpyDeviceToHost));
            std::vector<int> selected(hd,0); float best[2]={-INFINITY,-INFINITY};
            for(int row=0;row<=pos;++row) { ++selected[target_tags[row]]; best[row/blk]=std::max(best[row/blk],float(target_tags[row])); }
            for(int head=0;head<heads;++head) for(int label=0;label<hd;++label) {
                const float want=float(selected[label])/float(pos+1);
                if(!std::isfinite(actual[head*hd+label]) || std::fabs(actual[head*hd+label]-want)>1e-5f) {
                    std::fprintf(stderr,"target pos=%d head=%d label=%d got=%g want=%g\n",pos,head,label,actual[head*hd+label],want);
                    std::exit(1);
                }
            }
            for(int b=0;b<2;++b) if(best[b]!=-INFINITY && observed_scores[b]!=best[b]) {
                std::fprintf(stderr,"index pos=%d block=%d got=%g want=%g\n",pos,b,observed_scores[b],best[b]); std::exit(1);
            }
        };
        // Verify a three-row proposal, then reject its second row and replace
        // both the rejected slot and the next speculative slot with new tags.
        target_check(82);
        target_tags[81]=100; target_tags[82]=101;
        target_check(81);
        // Follow an older prefix and ensure future cached rows do not leak in.
        target_tags[5]=105; target_tags[6]=127;
        target_check(5);
        CUDA(cudaFree(target_key)); CUDA(cudaFree(target_value)); CUDA(cudaFree(index_q));
        CUDA(cudaFree(scores)); CUDA(cudaFree(kept));
    }
    CUDA(cudaFree(q)); CUDA(cudaFree(key)); CUDA(cudaFree(value)); CUDA(cudaFree(anchor));
    CUDA(cudaFree(mask)); CUDA(cudaFree(h)); CUDA(cudaFree(positions)); CUDA(cudaFree(lo));
    CUDA(cudaFree(output)); CUDA(cudaFree(part));
    std::puts("DSpark, target and indexer readers after wrap and replay: OK");
}
