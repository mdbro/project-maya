#pragma once
#include <cuda_runtime.h>
#include "ops/minimax_nvfp4_ffn/device_queue.h"
namespace mm3 {
namespace helper_queue_detail {
namespace nf = infernix::ops::minimax_ffn;
// Must execute after route and ready-table publication on the same stream.
// Out-of-range IDs are passed through as invalid table indices so the queue
// rejects the request instead of treating a corrupted route as an empty pick.
static __global__ void picks(const int* top,const float* weights,const nf::QueueRecord* table,
                             nf::QueuePick* out,int n,int experts,float scale) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;int e=top[i];
    out[i]=(e<0||e>=experts)?nf::QueuePick{experts,0.f}:
        nf::QueuePick{table[e].helper_record?e:-1,weights[i]*scale};
}
}
}
