// Focused production router hook test. Run only in the parent-controlled GPU
// window; cross-device transport and FFN are covered by the service fixture.
#include "mm3_native_helper_queue.cuh"
#include <cuda_runtime.h>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <vector>
namespace nf=infernix::ops::minimax_ffn;
int main(){
 int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||!count)return 77;
 constexpr int E=128,T=4,K=4,N=T*K;
 std::vector<int> top={1,2,3,4, 2,1,5,6, 3,7,1,8, 4,9,10,128};
 std::vector<float> weights(N);for(int i=0;i<N;++i)weights[i]=float(i+1)/32.f;
 std::vector<nf::QueueRecord> table(E);table[1].helper_record=0x1000;table[3].helper_record=0x2000;
 int *dt=nullptr;float *dw=nullptr;nf::QueueRecord* dr=nullptr;nf::QueuePick* dp=nullptr;
 auto ck=[](cudaError_t e){if(e!=cudaSuccess){std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e));std::abort();}};
 ck(cudaMalloc(&dt,N*sizeof(int)));ck(cudaMalloc(&dw,N*sizeof(float)));ck(cudaMalloc(&dr,E*sizeof(*dr)));ck(cudaMalloc(&dp,N*sizeof(*dp)));
 ck(cudaMemcpy(dt,top.data(),N*sizeof(int),cudaMemcpyHostToDevice));ck(cudaMemcpy(dw,weights.data(),N*sizeof(float),cudaMemcpyHostToDevice));
 ck(cudaMemcpy(dr,table.data(),E*sizeof(*dr),cudaMemcpyHostToDevice));
 mm3::helper_queue_detail::picks<<<1,32>>>(dt,dw,dr,dp,N,E,2.f);ck(cudaGetLastError());
 std::vector<nf::QueuePick> result(N);ck(cudaMemcpy(result.data(),dp,N*sizeof(*dp),cudaMemcpyDeviceToHost));
 for(int i=0;i<N;++i){int e=top[i];int want=e>=E?E:table[e].helper_record?e:-1;assert(result[i].slot==want);if(want>=0&&want<E)assert(result[i].weight==weights[i]*2.f);}
 // Withdraw expert 1 and publish a new owner at 2; next stream-ordered
 // snapshot must no longer route to the old slot.
 table[1]={};table[2].helper_record=0x3000;ck(cudaMemcpy(dr,table.data(),E*sizeof(*dr),cudaMemcpyHostToDevice));
 mm3::helper_queue_detail::picks<<<1,32>>>(dt,dw,dr,dp,N,E,2.f);ck(cudaGetLastError());
 ck(cudaMemcpy(result.data(),dp,N*sizeof(*dp),cudaMemcpyDeviceToHost));
 for(int i=0;i<N;++i){int e=top[i];int want=e>=E?E:table[e].helper_record?e:-1;assert(result[i].slot==want);}
 cudaFree(dp);cudaFree(dr);cudaFree(dw);cudaFree(dt);std::puts("queue router ownership mapping PASS");
}
