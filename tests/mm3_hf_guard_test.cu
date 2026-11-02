// Real HF guard and plan_dev indexing on invalid IDs. No model/weights.
#include "../src/mm3/mm3_resident_router.cuh"
#include "../src/mm3/mm3_kernels.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <limits>
#include <vector>
#define CUDA(call) do { auto e=(call); if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(e));return 1;} } while(0)
int main(){
    using namespace mm3::resident;
    int *ids=nullptr,*original=nullptr,*bad=nullptr;unsigned *orig_counts=nullptr,*exec_counts=nullptr;
    float *logits=nullptr,*bias=nullptr,*weights=nullptr,*plan_weights=nullptr;
    unsigned long long *table=nullptr,*plan=nullptr;Ready* ready=nullptr;
    CUDA(cudaMalloc(&ids,4*sizeof(int))); CUDA(cudaMalloc(&original,4*sizeof(int))); CUDA(cudaMalloc(&bad,sizeof(int)));
    CUDA(cudaMalloc(&orig_counts,128*sizeof(unsigned))); CUDA(cudaMalloc(&exec_counts,128*sizeof(unsigned)));
    CUDA(cudaMalloc(&logits,128*sizeof(float))); CUDA(cudaMalloc(&bias,128*sizeof(float))); CUDA(cudaMalloc(&weights,4*sizeof(float)));
    CUDA(cudaMalloc(&plan_weights,8*sizeof(float)));CUDA(cudaMalloc(&table,128*sizeof(unsigned long long)));
    CUDA(cudaMalloc(&plan,8*sizeof(unsigned long long)));CUDA(cudaMalloc(&ready,sizeof(Ready)));
    std::vector<float> x(128,0),b(128,0);x[41]=std::numeric_limits<float>::quiet_NaN();
    std::vector<unsigned long long> t(128);for(int i=0;i<128;++i)t[i]=0x1000+i;
    Ready all{};for(auto& v:all.primary)v=~uint32_t{0};
    cudaStream_t stream;CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
    CUDA(cudaMemcpyAsync(logits,x.data(),128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CUDA(cudaMemcpyAsync(bias,b.data(),128*sizeof(float),cudaMemcpyHostToDevice,stream));
    CUDA(cudaMemcpyAsync(table,t.data(),128*sizeof(unsigned long long),cudaMemcpyHostToDevice,stream));
    CUDA(cudaMemcpyAsync(ready,&all,sizeof(all),cudaMemcpyHostToDevice,stream));
    CUDA(cudaMemsetAsync(bad,0,sizeof(int),stream));CUDA(cudaMemsetAsync(orig_counts,0,128*sizeof(unsigned),stream));
    CUDA(cudaMemsetAsync(exec_counts,0,128*sizeof(unsigned),stream));
    CUDA(route(logits,bias,1,Policy::Prior,.5f,ready,ids,weights,original,nullptr,stream));
    CUDA(count_original(original,4,orig_counts,stream));
    CUDA(guard_expert_ids(ids,4,bad,stream));
    if(!mm3::k::plan_dev(ids,weights,1,4,table,1,plan,plan_weights,exec_counts,nullptr,nullptr,nullptr,nullptr,stream))return 2;
    int flag=0,host_ids[4]{},host_original[4]{};unsigned oh[128]{},eh[128]{};unsigned long long hp[8]{};
    CUDA(cudaMemcpyAsync(&flag,bad,sizeof(flag),cudaMemcpyDeviceToHost,stream));
    CUDA(cudaMemcpyAsync(host_ids,ids,sizeof(host_ids),cudaMemcpyDeviceToHost,stream));
    CUDA(cudaMemcpyAsync(host_original,original,sizeof(host_original),cudaMemcpyDeviceToHost,stream));
    CUDA(cudaMemcpyAsync(oh,orig_counts,sizeof(oh),cudaMemcpyDeviceToHost,stream));
    CUDA(cudaMemcpyAsync(eh,exec_counts,sizeof(eh),cudaMemcpyDeviceToHost,stream));
    CUDA(cudaMemcpyAsync(hp,plan,sizeof(hp),cudaMemcpyDeviceToHost,stream));
    CUDA(cudaStreamSynchronize(stream));
    assert(flag==1);for(int i=0;i<4;++i){assert(host_ids[i]==0 && host_original[i]==-1 && hp[i]==t[0]);}
    for(int i=0;i<128;++i)assert(oh[i]==0);
    assert(eh[0]==4);
    // Mixed underflow/overflow indices: valid originals remain counted, and
    // only sanitized zeros reach the real plan table/histogram.
    const int mixed[4]={-1,128,2,127};float w[4]={1,1,1,1};
    CUDA(cudaMemcpyAsync(ids,mixed,sizeof(mixed),cudaMemcpyHostToDevice,stream));
    CUDA(cudaMemcpyAsync(original,mixed,sizeof(mixed),cudaMemcpyHostToDevice,stream));
    CUDA(cudaMemcpyAsync(weights,w,sizeof(w),cudaMemcpyHostToDevice,stream));
    CUDA(cudaMemsetAsync(bad,0,sizeof(int),stream));CUDA(cudaMemsetAsync(orig_counts,0,128*sizeof(unsigned),stream));
    CUDA(cudaMemsetAsync(exec_counts,0,128*sizeof(unsigned),stream));
    CUDA(count_original(original,4,orig_counts,stream));CUDA(guard_expert_ids(ids,4,bad,stream));
    if(!mm3::k::plan_dev(ids,weights,1,4,table,1,plan,plan_weights,exec_counts,nullptr,nullptr,nullptr,nullptr,stream))return 2;
    CUDA(cudaMemcpyAsync(&flag,bad,sizeof(flag),cudaMemcpyDeviceToHost,stream));
    CUDA(cudaMemcpyAsync(oh,orig_counts,sizeof(oh),cudaMemcpyDeviceToHost,stream));
    CUDA(cudaMemcpyAsync(eh,exec_counts,sizeof(eh),cudaMemcpyDeviceToHost,stream));
    CUDA(cudaMemcpyAsync(hp,plan,sizeof(hp),cudaMemcpyDeviceToHost,stream));CUDA(cudaStreamSynchronize(stream));
    assert(flag==1 && oh[0]==0 && oh[2]==1 && oh[127]==1);
    assert(eh[0]==2 && eh[2]==1 && eh[127]==1);
    assert(hp[0]==t[0] && hp[1]==t[0] && hp[2]==t[2] && hp[3]==t[127]);
    CUDA(cudaStreamDestroy(stream));CUDA(cudaFree(ids));CUDA(cudaFree(original));CUDA(cudaFree(bad));CUDA(cudaFree(orig_counts));
    CUDA(cudaFree(exec_counts));CUDA(cudaFree(logits));CUDA(cudaFree(bias));CUDA(cudaFree(weights));CUDA(cudaFree(plan_weights));
    CUDA(cudaFree(table));CUDA(cudaFree(plan));CUDA(cudaFree(ready));std::puts("HF guard and plan: OK");
}
