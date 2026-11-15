// Manual narrow gate: dropping the source BF16 allocation must not change the
// same encoded head's output or batching. No model, no new precision decision.
// Covers target/pipeline/prompt common MV capacity and actual full-logits
// dispatch (FP8->native GEMM, NVFP4->F32 MV), shapes0/1, direct and graph replay.
#include "mm3_head_dispatch.cuh"
#include <cuda_fp16.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
namespace hl=mm3::head_lowp;
namespace hd=mm3::head_dispatch;
namespace nd=strata::mm3::native_dense;
namespace x3=strata::mm3::exl3;
// This small native-only target links the same dense path as dense_fp8_bench.
namespace strata::mm3::exl3 {
bool supported(const Mat& m){
    if(m.fmt==nd::kFmtMXFP8)return m.raw&&m.native_scale&&m.k>0&&m.k%512==0&&m.n>0&&m.n%32==0;
    if(m.fmt==nd::kFmtBF16)return m.raw&&m.k>0&&m.n>0;
    return false;
}
}
constexpr int N=128,K=512,R=32;
static void ck(cudaError_t e,const char* s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
static void need(bool b,const char* s){if(!b)throw std::runtime_error(s);}
template<class T>static T* alloc(size_t n){T* p=nullptr;ck(cudaMalloc((void**)&p,n*sizeof(T)),"allocation");return p;}
static uint16_t bf(float f){uint32_t u;std::memcpy(&u,&f,4);return uint16_t((u+0x7fff+((u>>16)&1))>>16);}
__global__ void fp16_inputs(const float* x,uint16_t* out,int n){int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=__half_as_ushort(__float2half_rn(x[i]));}
struct Result{std::vector<float> values;std::vector<int> chunks;size_t kernels=0;int gemm_calls=0;};
int main(int argc,char** argv){
    try{
        ck(cudaSetDevice(argc>1?std::stoi(argv[1]):0),"device");cudaStream_t s;ck(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking),"stream");
        std::mt19937 rng(151115);std::normal_distribution<float> normal(0,0.1f);
        std::vector<uint16_t> weights(size_t(N)*K);for(auto& v:weights)v=bf(normal(rng));
        std::vector<float> inputs(size_t(R)*K);for(auto& v:inputs)v=normal(rng)*10.f;
        auto* original=alloc<uint16_t>(weights.size());auto* x=alloc<float>(inputs.size());auto* x16=alloc<uint16_t>(inputs.size());auto* y=alloc<float>(size_t(R)*N);auto* reference=alloc<float>(size_t(R)*N);
        ck(cudaMemcpyAsync(original,weights.data(),weights.size()*2,cudaMemcpyHostToDevice,s),"weights upload");
        ck(cudaMemcpyAsync(x,inputs.data(),inputs.size()*4,cudaMemcpyHostToDevice,s),"input upload");
        fp16_inputs<<<(inputs.size()+255)/256,256,0,s>>>(x,x16,int(inputs.size()));ck(cudaStreamSynchronize(s),"uploads done");
        const size_t bytes=nd::scratch_bytes(K,true);auto* scratch=alloc<uint8_t>(bytes);need(nd::prepare(s,K,scratch,bytes,true),"native dense context");
        x3::Mat source;source.fmt=nd::kFmtBF16;source.raw=original;source.n=N;source.k=K;
        hl::Head copies[2];const char* why=nullptr;
        need(hl::make(hl::kFp8,original,N,K,copies[0],s,&why),why?why:"FP8 copy");
        need(hl::make(hl::kNvfp4,original,N,K,copies[1],s,&why),why?why:"NVFP4 copy");
        // Ordinary BF16 capacity remains exactly the pre-existing max(1,value),
        // and the active-copy branch must not call the BF16 accessor at all.
        for(int capacity:{0,1,4,8,16}){int calls=0;need(hd::mv_columns(nullptr,[&]{++calls;return capacity;})==std::max(1,capacity)&&calls==1,"ordinary head capacity changed");}
        hl::Head absent;need(hl::mv_columns(absent)==0,"absent head advertised capacity");
        const auto execute=[&](const hl::Head* copy,int nr,bool full,bool graph){
            Result result;
            const auto body=[&]{
                result.chunks.clear();result.gemm_calls=0;
                ck(cudaMemsetAsync(y,0xff,size_t(nr)*N*4,s),"poison output");
                if(full){
                    result.chunks.push_back(nr);
                    need(hd::rows(copy,source,x,K,nr,y,N,s,[&](const x3::Mat& m){
                        ++result.gemm_calls;
                        if(copy)need(m.raw==copy->w&&m.native_scale==copy->sc,"full-logits did not select encoded copy");
                        else need(m.raw==source.raw&&m.fmt==source.fmt,"ordinary full-logits source changed");
                        return nd::gemm_rows(m,x16,K,nr,y,N,0.f,s);
                    }),"full-logits dispatch");
                    const int expected_gemm_calls=copy&&copy->fmt==hl::kNvfp4?0:1;
                    need(result.gemm_calls==expected_gemm_calls,"full-logits changed GEMM/MV route");
                }else{
                    int original_queries=0;
                    const int cols=hd::mv_columns(copy,[&]{++original_queries;return source.raw?16:0;});
                    need(cols==16&&original_queries==0,"active encoded head queried BF16 capacity");
                    for(int at=0;at<nr;at+=cols){const int count=std::min(cols,nr-at);need(hl::mv(*copy,x+size_t(at)*K,K,count,y+size_t(at)*N,N,s),"head MV");result.chunks.push_back(count);}
                }
            };
            body();ck(cudaStreamSynchronize(s),"uncaptured dispatch"); // initialize cuBLAS before capture
            if(graph){
                cudaGraph_t g=nullptr;cudaGraphExec_t e=nullptr;
                ck(cudaStreamBeginCapture(s,cudaStreamCaptureModeThreadLocal),"capture");body();ck(cudaStreamEndCapture(s,&g),"capture end");
                size_t nodes=0;ck(cudaGraphGetNodes(g,nullptr,&nodes),"node count");std::vector<cudaGraphNode_t> all(nodes);ck(cudaGraphGetNodes(g,all.data(),&nodes),"nodes");
                for(auto node:all){cudaGraphNodeType type;ck(cudaGraphNodeGetType(node,&type),"node type");if(type==cudaGraphNodeTypeKernel)++result.kernels;}
                ck(cudaGraphInstantiate(&e,g,0),"instantiate");ck(cudaGraphLaunch(e,s),"replay");ck(cudaStreamSynchronize(s),"replay done");cudaGraphExecDestroy(e);cudaGraphDestroy(g);
                if(!full)need(result.kernels==size_t((nr+7)/8),"MV graph lost multi-column batching");
                if(full&&copy&&copy->fmt==hl::kNvfp4)need(result.kernels==size_t((nr+7)/8),"full-logits NVFP4 graph lost multi-column batching");
            }
            result.values.resize(size_t(nr)*N);ck(cudaMemcpyAsync(result.values.data(),y,result.values.size()*4,cudaMemcpyDeviceToHost,s),"read");ck(cudaStreamSynchronize(s),"read done");
            if(full){
                // Independent pre-refactor oracle: do not call shared dispatch.
                // In particular, FP8 must keep FP16->BF16 GEMM activations;
                // substituting its F32 MV can pass retained/drop equality alone.
                if(copy&&copy->fmt==hl::kFp8){
                    x3::Mat explicit_mxfp8;explicit_mxfp8.fmt=nd::kFmtMXFP8;
                    explicit_mxfp8.raw=copy->w;explicit_mxfp8.native_scale=copy->sc;
                    explicit_mxfp8.k=copy->k;explicit_mxfp8.n=copy->n;
                    need(nd::gemm_rows(explicit_mxfp8,x16,K,nr,reference,N,0.f,s),"independent FP8 GEMM oracle");
                }else if(copy){
                    need(hl::mv(*copy,x,K,nr,reference,N,s),"independent NVFP4 F32 MV oracle");
                }else need(nd::gemm_rows(source,x16,K,nr,reference,N,0.f,s),"independent BF16 GEMM oracle");
                std::vector<float> expected(result.values.size());
                ck(cudaMemcpyAsync(expected.data(),reference,expected.size()*4,cudaMemcpyDeviceToHost,s),"oracle read");ck(cudaStreamSynchronize(s),"oracle read done");
                need(std::memcmp(expected.data(),result.values.data(),expected.size()*4)==0,"full-logits differs from independent pre-refactor route");
            }
            need(std::all_of(result.values.begin(),result.values.end(),[](float v){return std::isfinite(v);}),"nonfinite output");return result;
        };
        // Check ordinary full-logits twice before freeing its source, using the
        // exact same production selector and existing native BF16 GEMM route.
        const auto bf0=execute(nullptr,16,true,false),bf1=execute(nullptr,16,true,true);
        need(std::memcmp(bf0.values.data(),bf1.values.data(),bf0.values.size()*4)==0,"ordinary BF16 full-logits direct/graph mismatch");
        struct Case{int format,shape,nr;bool full,graph;Result retained;};std::vector<Case> cases;
        for(int f:{0,1})for(int shape:{0,1})for(int nr:{1,2,4,8,9,16,32})for(bool full:{false,true})for(bool graph:{false,true}){
            hl::set_shape_mode(shape);cases.push_back(Case{f,shape,nr,full,graph,execute(&copies[f],nr,full,graph)});
        }
        ck(cudaStreamSynchronize(s),"before source free");ck(cudaFree(original),"drop BF16");source.raw=nullptr;
        for(const auto& c:cases){
            hl::set_shape_mode(c.shape);const auto dropped=execute(&copies[c.format],c.nr,c.full,c.graph);
            need(dropped.chunks==c.retained.chunks&&dropped.kernels==c.retained.kernels&&dropped.gemm_calls==c.retained.gemm_calls,"drop changed graph/dispatch geometry");
            need(std::memcmp(dropped.values.data(),c.retained.values.data(),dropped.values.size()*4)==0,"drop changed same-copy outputs");
        }
        std::printf("head drop PASS: %zu retained/dropped same-copy cases, FP8/NVFP4, shapes0/1, widths1/2/4/8/9/16/32, direct/graph; target/pipeline/prompt batching and full-logits routes unchanged\n",cases.size());
        hl::set_shape_mode(0);nd::release(s);for(auto& h:copies)hl::release(h);cudaFree(scratch);cudaFree(reference);cudaFree(y);cudaFree(x16);cudaFree(x);cudaStreamDestroy(s);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"head drop test: %s\n",e.what());return 1;}
}
