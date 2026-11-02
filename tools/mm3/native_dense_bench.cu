// Manual bounded comparison of the existing native dense caller and its
// existing split-K MXFP8 kernel. Reads two official tensor pairs, not a model.
// Representative activations are deterministic synthetic F32; this is a kernel
// selection screen, not target-serving or stage throughput.
#include "mm3_pack.hpp"
#include "mm3_native_dense.cuh"
#include "mm3_kernels.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nd=strata::mm3::native_dense;
namespace x3=strata::mm3::exl3;
namespace k=mm3::k;
// This diagnostic links only native MXFP8, not the legacy EXL3 runtime.
namespace strata::mm3::exl3 {
bool supported(const Mat& m) {return m.fmt==nd::kFmtMXFP8 && m.raw && m.native_scale && m.k>0 && m.k%32==0 && m.n>0;}
}
namespace {
using Clock=std::chrono::steady_clock;
void ck(cudaError_t e,const char* op){if(e!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(e));}
struct Buffer {void* p=nullptr;size_t bytes=0;explicit Buffer(size_t n):bytes(n){ck(cudaMalloc(&p,n),"cudaMalloc");}
    ~Buffer(){if(p)cudaFree(p);}Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    template<class T>T* as(){return static_cast<T*>(p);}};
struct Matrix {std::string name;int n,k;const mm3::Tensor *w,*s;};
float e4m3(uint8_t c){int e=(c>>3)&15,m=c&7;if(e==15&&m==7)return std::numeric_limits<float>::quiet_NaN();
    float f=e?std::ldexp(float(8+m),e-10):std::ldexp(float(m),-9);return c&128?-f:f;}
float e8m0(uint8_t c){return c==255?std::numeric_limits<float>::quiet_NaN():std::ldexp(1.f,int(c)-127);}
struct Timing {int warm_calls=0,calls=0;double warm_wall_ms=0,wall_ms=0,event_ms=0;};
}
int main(int argc,char** argv){
    std::string path="/mnt/wsl/transfer/minimax-wsl/artifacts/proposal-2/native-pack";int device=0;
    try {
        for(int i=1;i<argc;++i){std::string a=argv[i];if(i+1>=argc)throw std::invalid_argument("option requires value");
            if(a=="--pack")path=argv[++i];else if(a=="--device")device=std::stoi(argv[++i]);else throw std::invalid_argument("unknown option: "+a);}
        const char* env=std::getenv("MAYA_MM3_MXC");if(env && std::atoi(env)==0)throw std::runtime_error("MAYA_MM3_MXC=0 disables the requested split-K route");
        ck(cudaSetDevice(device),"set device");cudaStream_t stream=nullptr;cudaEvent_t start=nullptr,stop=nullptr;
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");ck(cudaEventCreate(&start),"start event");ck(cudaEventCreate(&stop),"stop event");
        mm3::Pack pack;std::string err;if(!pack.load(path,err)||pack.str("format")!="native")throw std::runtime_error("native pack: "+err);
        const std::string base="language_model.model.layers.0.";
        Matrix matrices[]={{base+"self_attn.q_proj",8192,6144,nullptr,nullptr},{base+"mlp.down_proj",6144,12288,nullptr,nullptr}};
        for(auto& m:matrices){m.w=pack.get(m.name+".weight");m.s=pack.get(m.name+".weight_scale_inv");
            if(!m.w||!m.s||m.w->dtype!="F8_E4M3"||m.s->dtype!="U8"||m.w->shape!=std::vector<int64_t>{m.n,m.k}||
               m.s->shape!=std::vector<int64_t>{m.n,m.k/32}||m.w->bytes!=size_t(m.n)*m.k||m.s->bytes!=size_t(m.n)*m.k/32)
                throw std::runtime_error("invalid official MXFP8 tensor geometry: "+m.name);}
        constexpr int MAXK=12288, MAXN=8192, MAXR=4;
        const size_t wbytes=std::max(matrices[0].w->bytes,matrices[1].w->bytes);
        const size_t sbytes=std::max(matrices[0].s->bytes,matrices[1].s->bytes);
        const size_t ndbytes=nd::scratch_bytes(MAXK);
        // mxc_kernel stores [K/1024][rows][N] floats. Each measured
        // geometry fits this maximum; the existing native caller's own
        // internal scratch is kept separate and unchanged.
        size_t partbytes=0;for(const auto& m:matrices)partbytes=std::max(partbytes,size_t(m.k/1024)*MAXR*m.n*sizeof(float));
        Buffer w(wbytes),sc(sbytes),x(size_t(MAXR)*MAXK*4),y(size_t(MAXR)*MAXN*4),ndscratch(ndbytes),part(partbytes);
        if(!nd::prepare(stream,MAXK,ndscratch.p,ndscratch.bytes))throw std::runtime_error("native dense prepare failed");
        std::printf("{\"kind\":\"native_dense_manual_benchmark\",\"device\":%d,\"activation\":\"deterministic synthetic F32\","
                    "\"weight_allocation_bytes\":%zu,\"scale_allocation_bytes\":%zu,\"input_allocation_bytes\":%zu,"
                    "\"output_allocation_bytes\":%zu,\"native_scratch_bytes\":%zu,\"split_k_scratch_bytes\":%zu,"
                    "\"total_device_allocation_bytes\":%zu,\"max_host_input_bytes\":%zu,\"max_host_output_bytes\":%zu,"
                    "\"allocation_scope\":\"explicit buffers; excludes cuda context, cublas internals and demand-paged checkpoint\","
                    "\"warm_min_ms\":500,\"measure_min_ms\":100}\n",
                    device,wbytes,sbytes,x.bytes,y.bytes,ndbytes,partbytes,wbytes+sbytes+x.bytes+y.bytes+ndbytes+partbytes,
                    size_t(MAXR)*MAXK*sizeof(float),2*size_t(MAXR)*MAXN*sizeof(float));
        std::fflush(stdout);
        bool good=true;
        for(const auto& m:matrices){
            ck(cudaMemcpy(w.p,pack.data(*m.w),m.w->bytes,cudaMemcpyHostToDevice),"weights upload");
            ck(cudaMemcpy(sc.p,pack.data(*m.s),m.s->bytes,cudaMemcpyHostToDevice),"scales upload");
            std::vector<float> hx(size_t(MAXR)*m.k);for(size_t i=0;i<hx.size();++i)hx[i]=0.01f*float((int(i*37%101))-50);
            ck(cudaMemcpy(x.p,hx.data(),hx.size()*4,cudaMemcpyHostToDevice),"input upload");
            x3::Mat mat;mat.fmt=nd::kFmtMXFP8;mat.n=m.n;mat.k=m.k;mat.raw=w.p;mat.native_scale=sc.as<uint8_t>();
            for(int rows:{1,4}){
                std::vector<float> output[2];
                auto launch=[&](int mode){
                    if(mode==0){x3::MvJob j;j.m=&mat;j.xf=x.as<float>();j.y=y.as<float>();j.ncol=rows;j.xf_cs=m.k;j.y_cs=m.n;
                        if(!nd::mv(j,stream))throw std::runtime_error("native dense dispatch failed");}
                    else k::mx_rows(w.as<uint8_t>(),sc.as<uint8_t>(),m.n,m.k,x.as<float>(),rows,m.k,y.as<float>(),m.n,0.f,part.as<float>(),stream);
                };
                for(int mode=0;mode<2;++mode){
                    // Detect a missing launch or unwritten output rather than
                    // accepting a previous arm's finite result.
                    ck(cudaMemsetAsync(y.p,0xff,size_t(rows)*m.n*sizeof(float),stream),"poison output");
                    auto measure=[&](bool timed,Timing& t){auto begin=Clock::now();int calls=0;double ev=0;
                        do{if(timed)ck(cudaEventRecord(start,stream),"record begin");launch(mode);
                            if(timed){ck(cudaEventRecord(stop,stream),"record end");ck(cudaEventSynchronize(stop),"event sync");float v=0;ck(cudaEventElapsedTime(&v,start,stop),"event elapsed");ev+=v;}
                            else ck(cudaStreamSynchronize(stream),"warm sync");++calls;
                        }while(std::chrono::duration<double>(Clock::now()-begin).count()<(timed?.1:.5));
                        double wall=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
                        if(timed){t.calls=calls;t.wall_ms=wall;t.event_ms=ev;}else{t.warm_calls=calls;t.warm_wall_ms=wall;}};
                    Timing t;measure(false,t);measure(true,t);
                    output[mode].resize(size_t(rows)*m.n);ck(cudaMemcpy(output[mode].data(),y.p,output[mode].size()*4,cudaMemcpyDeviceToHost),"read result");
                    size_t bad=0;for(float v:output[mode])bad+=!std::isfinite(v);good &= bad==0;
                    std::printf("{\"matrix\":\"%s\",\"n\":%d,\"k\":%d,\"columns\":%d,\"arm\":\"%s\","
                      "\"warm_calls\":%d,\"warm_wall_ms\":%.6f,\"calls\":%d,\"wall_total_ms\":%.6f,"
                      "\"wall_ms_per_call\":%.9f,\"cuda_event_total_ms\":%.6f,\"cuda_event_ms_per_call\":%.9f,\"nonfinite\":%zu}\n",
                      m.name.c_str(),m.n,m.k,rows,mode?"direct_mxc_split_k":"native_mv",t.warm_calls,t.warm_wall_ms,t.calls,t.wall_ms,t.wall_ms/t.calls,t.event_ms,t.event_ms/t.calls,bad);
                    std::fflush(stdout);
                }
                double diff=0,refnorm=0,maxabs=0;for(size_t i=0;i<output[0].size();++i){double d=double(output[1][i])-output[0][i];diff+=d*d;refnorm+=double(output[0][i])*output[0][i];maxabs=std::max(maxabs,std::abs(d));}
                double rr=std::sqrt(diff/std::max(refnorm,1e-30));
                // Independent CPU dot on 16 evenly spaced real weight rows for
                // every input column; the 0.5%+0.01 gate is a numerical sanity
                // check, not an agreement requirement across devices.
                const auto* hw=pack.data(*m.w);const auto* hs=pack.data(*m.s);double max_ratio=0;size_t bad_ref=0;
                for(int c=0;c<rows;++c)for(int i=0;i<16;++i){int r=int((int64_t)i*(m.n-1)/15);double expected=0;
                    for(int j=0;j<m.k;++j)expected+=double(e4m3(hw[size_t(r)*m.k+j]))*e8m0(hs[size_t(r)*(m.k/32)+j/32])*hx[size_t(c)*m.k+j];
                    if(!std::isfinite(expected)){++bad_ref;continue;}
                    for(int mode=0;mode<2;++mode){double delta=std::abs(double(output[mode][size_t(c)*m.n+r])-expected);
                        max_ratio=std::max(max_ratio,delta/(0.01+0.005*std::abs(expected)));}}
                bool pass=std::isfinite(rr)&&rr<0.005&&bad_ref==0&&max_ratio<=1.0;good &= pass;
                std::printf("{\"matrix\":\"%s\",\"columns\":%d,\"same_device_relative_rmse\":%.9g,\"same_device_max_abs\":%.9g,"
                            "\"cpu_reference_rows_per_column\":16,\"cpu_reference_nonfinite\":%zu,\"cpu_reference_max_tolerance_ratio\":%.9g,\"sanity_pass\":%s}\n",
                            m.name.c_str(),rows,rr,maxabs,bad_ref,max_ratio,pass?"true":"false");
                std::fflush(stdout);
            }
        }
        ck(cudaStreamSynchronize(stream),"final stream");nd::release(stream);cudaEventDestroy(start);cudaEventDestroy(stop);cudaStreamDestroy(stream);
        return good?0:1;
    } catch(const std::exception& e){std::fprintf(stderr,"native dense benchmark: %s\n",e.what());return 1;}
}
