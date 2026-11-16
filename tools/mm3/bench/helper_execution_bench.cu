// Same-arithmetic helper execution gate/screen. No alternative arithmetic is
// selected: 3090 Q1=172/68 loads8; 4080 Q2=188/88 loads4 with production SM89.
// The old implementation is frozen independently in reference/ (manifest names
// the commit and original hashes). It supplies hidden bytes and final columns;
// NativeHelper's unchanged host merge supplies their expected merged result.
//
// Gate: every width1..8, jobs1/3/8 and a mixed-width pass, finite/adversarial/
// nonfinite represented inputs, direct and changed-input captured launches.
// Production: direct and actual zero-copy CUDA graphs, mode alternation, changed
// inputs/descriptors, nonfinite refusal and in-flight mode-change refusal.
// Timings: CUDA full-kernel event intervals plus submit-through-collect host
// latency on the actual serving graph/copy route. No replay_probe_timing, which
// would disable those routes. Small-job weights may fit L2; no DRAM claim.
// Run only in the parent's coordinated quiet window:
//   mm3_helper_execution_bench --card 3090|4080 [--iters 60] [--checks-only]
// Bounded follow-on for the observed 4080 one-job cache ambiguity:
//   mm3_helper_execution_bench --card 4080 --one-job-probe [--iters 64]
#include "mm3_native_helper.hpp"
#include "mm3_helper_q.cuh"
#include "mm3_helper_sm89.hpp"
#include "reference/helper_q_frozen.cuh"
#include "reference/helper_q_raw_frozen.cuh"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace q = mm3::helper_q;
namespace old = mm3::helper_q_frozen;
namespace tc = mm3::helper_tc;
using Helper = mm3::NativeHelper;
using Job = infernix::ops::minimax_ffn::FastJob;
constexpr int H=6144, I=3072, kExperts=4, kLayer=3, kJobs=8, kCols=64;
int failures=0, checks=0;
void ck(cudaError_t e,const char* what) {
    if(e!=cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e));
}
void require(bool b,const std::string& what) { if(!b) throw std::runtime_error(what); }
void verdict(bool b,const std::string& what) {
    ++checks; failures+=!b;
    std::printf("CHECK %s %s\n",b?"PASS":"FAIL",what.c_str());
}
double now_ms() { return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int find_card(const char* card) {
    int n=0; ck(cudaGetDeviceCount(&n),"device count");
    for(int d=0;d<n;++d) { cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p,d),"device properties"); if(std::strstr(p.name,card))return d; }
    throw std::runtime_error(std::string("missing card ")+card);
}
double quantile(std::vector<double> v,double p) {
    std::sort(v.begin(),v.end()); return v.empty()?0:v[std::size_t(p*double(v.size()-1)+0.5)];
}
struct Shape {
    std::vector<int> widths;
    int columns()const { int n=0;for(int w:widths)n+=w;return n; }
    int tokens()const { int n=0;for(std::size_t j=0;j<widths.size();++j)n=std::max(n,int(j/4)*8+widths[j]);return n; }
    bool multi()const { return columns()>int(widths.size()); }
    std::string name()const { std::string s;for(int w:widths){if(!s.empty())s+=",";s+=std::to_string(w);}return s; }
};
std::vector<Helper::Job> helper_jobs(const Shape& shape,int version) {
    std::vector<Helper::Job> jobs;
    for(std::size_t j=0;j<shape.widths.size();++j) {
        Helper::Job a; a.layer=kLayer;a.expert=(int(j)+version)%kExperts;a.columns=shape.widths[j];
        for(int c=0;c<a.columns;++c){a.token[c]=int(j/4)*8+c;a.weight[c]=0.125f*float(1+(j+version)%4);}
        jobs.push_back(a);
    }
    // NativeHelper orders jobs by width before both compute and merge.
    std::stable_sort(jobs.begin(),jobs.end(),[](const auto& a,const auto& b){return a.columns<b.columns;});
    return jobs;
}
std::vector<std::uint8_t> packed_rows(int tokens,int version,int pattern) {
    std::vector<std::uint8_t> out(std::size_t(tokens)*q::kPackedRow);
    std::mt19937 rng(20261115+version*193+pattern*809);
    for(int t=0;t<tokens;++t) {
        auto* row=out.data()+std::size_t(t)*q::kPackedRow;
        for(int b=0;b<H/16;++b) {
            for(int k=0;k<8;++k) row[8*b+k]=pattern==1?std::uint8_t(((2*k+b)&15)|(((2*k+b+1)&15)<<4)):std::uint8_t(rng());
            row[H/2+b]=pattern==1?std::uint8_t((b+31*t+version)%127):std::uint8_t(0x28+rng()%25);
            // Every signed zero code and zero/subnormal/max finite scale are
            // represented in pattern1. Pattern2 additionally forces NaN paths.
            if(pattern==1 && b%17==0) {for(int k=0;k<8;++k)row[8*b+k]=0x80;row[H/2+b]=0;}
            if(pattern==2 && b%127==0) row[H/2+b]=0x7f;
        }
    }
    return out;
}

// Only retained warps/loads are instantiated in this TU. The separate SM89
// wrappers and NativeHelper below additionally exercise production dispatch.
template<int S,int Exec>
cudaError_t q1_current(const Job* jobs,int n,const std::uint16_t* x,const std::uint8_t* in,std::uint8_t* ws,cudaStream_t s) {
    if constexpr(Exec==0)return q::detail::ffn_sp<S,8,false,false>(jobs,n,x,in,ws,s);
    else return q::detail::execution_sp<S,8,false,Exec,false>(jobs,n,x,in,ws,s);
}
template<int S>
cudaError_t q1_dispatch(int exec,const Job* jobs,int n,const std::uint16_t* x,const std::uint8_t* in,std::uint8_t* ws,cudaStream_t s) {
    switch(exec) {
        case 0:return q1_current<S,0>(jobs,n,x,in,ws,s);
        case 1:return q1_current<S,1>(jobs,n,x,in,ws,s);
        case 2:return q1_current<S,2>(jobs,n,x,in,ws,s);
        case 3:return q1_current<S,3>(jobs,n,x,in,ws,s);
        default:return cudaErrorInvalidValue;
    }
}
struct Direct {
    bool third;
    Helper& helper;
    cudaStream_t stream=nullptr;
    Job* jobs=nullptr;
    std::uint16_t *x=nullptr,*y=nullptr;
    std::uint8_t *packed=nullptr,*codes=nullptr,*workspace=nullptr;
    std::size_t workbytes=std::max(std::size_t(kJobs)*q::kJobHidden,tc::workspace_bytes(kJobs));
    std::array<Helper::QueueIdentity,128> ids{};
    Direct(bool t,Helper& h):third(t),helper(h) {
        std::string err;require(helper.queue_snapshot(kLayer,ids,err),err);
        ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"direct stream");
        ck(cudaMalloc(&jobs,kJobs*sizeof(Job)),"direct jobs");
        ck(cudaMalloc(&x,std::size_t(kCols)*H*2),"direct x identity");
        ck(cudaMalloc(&y,std::size_t(kCols)*H*2),"direct y");
        ck(cudaMalloc(&packed,std::size_t(kCols)*q::kPackedRow),"direct packed");
        ck(cudaMalloc(&codes,std::size_t(kCols)*q::kInStride),"direct codes");
        ck(cudaMalloc(&workspace,workbytes),"direct workspace");
    }
    ~Direct(){cudaStreamSynchronize(stream);cudaFree(jobs);cudaFree(x);cudaFree(y);cudaFree(packed);cudaFree(codes);cudaFree(workspace);cudaStreamDestroy(stream);}
    void prepare(const std::vector<Helper::Job>& host,const std::vector<std::uint8_t>& rows,bool invalid_multiplier=false) {
        std::vector<Job> desc;std::vector<std::uint8_t> gathered;int col=0;
        for(const auto& a:host) {
            Job b{};b.record=reinterpret_cast<const std::uint8_t*>(helper.pool_identity())+std::size_t(ids[a.expert].slot)*helper.record_stride();
            b.columns=a.columns;b.x=x+std::size_t(col)*H;b.y=y+std::size_t(col)*H;
            std::copy_n(ids[a.expert].multiplier,3,b.multiplier);
            if(invalid_multiplier)b.multiplier[0]=std::numeric_limits<float>::infinity();
            desc.push_back(b);
            for(int c=0;c<a.columns;++c) {
                const auto* row=rows.data()+std::size_t(a.token[c])*q::kPackedRow;
                gathered.insert(gathered.end(),row,row+q::kPackedRow);
            }
            col+=a.columns;
        }
        ck(cudaMemcpyAsync(jobs,desc.data(),desc.size()*sizeof(Job),cudaMemcpyHostToDevice,stream),"direct descriptors");
        ck(cudaMemcpyAsync(packed,gathered.data(),gathered.size(),cudaMemcpyHostToDevice,stream),"direct inputs");
        ck(q::expand_packed(packed,codes,col,stream),"direct Q1 input expansion");
        ck(cudaStreamSynchronize(stream),"direct preparation retired");
    }
    cudaError_t launch(bool frozen,int exec,const Shape& shape) {
        const int n=int(shape.widths.size());
        if(frozen) {
            if(third)return old::detail::ffn2_sp<8,4>(jobs,n,x,packed,workspace,stream);
            if(shape.multi())return old::detail::ffn_sp<8,8,false,false>(jobs,n,x,codes,workspace,stream);
            return old::detail::ffn_sp<4,8,false,false>(jobs,n,x,codes,workspace,stream);
        }
        if(third)return mm3::helper_sm89::q2_ffn_exec(exec,8,4,jobs,n,x,packed,workspace,stream);
        if(shape.multi())return q1_dispatch<8>(exec,jobs,n,x,codes,workspace,stream);
        // Mirror NativeHelper's all-one-column dispatch: strip only bit1 and
        // retain the original S4 code, not a branch in a larger parallel kernel.
        return q1_dispatch<4>(exec&~1,jobs,n,x,codes,workspace,stream);
    }
    struct Result { std::vector<std::uint8_t> hidden;std::vector<std::uint16_t> output; };
    void clear(){ck(cudaMemsetAsync(workspace,0xa5,workbytes,stream),"workspace sentinel");ck(cudaMemsetAsync(y,0x5a,std::size_t(kCols)*H*2,stream),"output sentinel");}
    Result read(const Shape& shape) {
        ck(cudaStreamSynchronize(stream),"direct retired");
        Result r;r.hidden.resize((third?tc::kJobWorkspace:q::kJobHidden)*shape.widths.size());r.output.resize(std::size_t(shape.columns())*H);
        ck(cudaMemcpy(r.hidden.data(),workspace,r.hidden.size(),cudaMemcpyDeviceToHost),"hidden bytes back");
        ck(cudaMemcpy(r.output.data(),y,r.output.size()*2,cudaMemcpyDeviceToHost),"columns back");return r;
    }
    Result oracle(const Shape& shape){clear();ck(launch(true,0,shape),"frozen oracle");return read(shape);}
};
bool equal(const Direct::Result& a,const Direct::Result& b){return a.hidden==b.hidden && a.output==b.output;}

// Raw Q1 sums call the two actual implementations, before cross-warp reduction,
// SwiGLU and BF16/hidden quantization. The hidden operand for down is frozen.
template<int S,int Exec,bool Frozen,bool Down>
__global__ void q1_raw_kernel(const Job* jobs,const std::uint16_t* xbase,const std::uint8_t* in,
                              const std::uint8_t* hidden,float* raw) {
    __shared__ float scale[q::detail::kScaleTable];
    if constexpr(Frozen)old::detail::fill_scale_table(scale,32*S);else q::detail::fill_scale_table(scale,32*S);
    const int j=blockIdx.y,group=blockIdx.x,warp=threadIdx.x>>5,lane=threadIdx.x&31,g=lane>>2,c=lane&3;
    const Job job=jobs[j];const bool live=g<job.columns;const unsigned keep=live?~0u:0u;
    constexpr int K=Down?I:H,groups=Down?H/16:2*I/16;
    const std::size_t col0=Down?0:std::size_t(job.x-xbase)/H;
    const auto* base=Down?hidden+std::size_t(j)*q::kJobHidden:in;
    constexpr auto stride=Down?q::kHidStride:q::kInStride;
    const auto* xcol=base+(col0+(live?g:0))*stride;
    const auto* s0p=base+(col0+(2*c<job.columns?2*c:0))*stride+K;
    const auto* s1p=base+(col0+(2*c+1<job.columns?2*c+1:0))*stride+K;
    const auto* unit=job.record+(Down?q::detail::kDownOffset:0)+std::size_t(group)*(K/16)*q::kUnit;
    float a=0,b=0,c0=0,d=0;
    if constexpr(Frozen)old::detail::sums<S,Down,8,false,false>(unit,xcol,s0p,s1p,keep,scale,warp,a,b,c0,d);
    else q::detail::sums<S,Down,8,false,false,Exec>(unit,xcol,s0p,s1p,keep,scale,warp,a,b,c0,d);
    const std::size_t at=((std::size_t(j)*groups+group)*S+warp)*128+lane*4;
    raw[at]=a;raw[at+1]=b;raw[at+2]=c0;raw[at+3]=d;
}
template<int Exec,bool Frozen,bool Down>
__global__ void q2_raw_kernel(const Job* jobs,const std::uint16_t* xbase,const std::uint8_t* packed,
                              std::uint8_t* workspace,float* raw) {
    if constexpr(Frozen) {
        if constexpr(Down)mm3::helper_tc_raw_frozen::detail::project2_block<8,true,4,true>(jobs,workspace,int(blockIdx.y),int(blockIdx.x),raw);
        else mm3::helper_q_raw_frozen::detail::up2_block<8,4,false,true>(jobs,xbase,packed,workspace,int(blockIdx.y),int(blockIdx.x),raw);
    } else {
        if constexpr(Down)tc::detail::project2_block<8,true,4,bool(Exec&2),true>(jobs,workspace,int(blockIdx.y),int(blockIdx.x),raw);
        else q::detail::up2_block<8,4,false,Exec,true>(jobs,xbase,packed,workspace,int(blockIdx.y),int(blockIdx.x),raw);
    }
}
template<int S,int Exec,bool Frozen>
void raw_pair(Direct& d,const Shape& shape,float* up,float* down) {
    const int n=int(shape.widths.size());
    if(d.third) {
        q2_raw_kernel<Exec,Frozen,false><<<dim3(I/16,n),512,0,d.stream>>>(d.jobs,d.x,d.packed,d.workspace,up);
        ck(cudaGetLastError(),"Q2 raw up");
        q2_raw_kernel<Exec,Frozen,true><<<dim3(H/16,n),256,0,d.stream>>>(d.jobs,d.x,d.packed,d.workspace,down);
    } else {
        q1_raw_kernel<S,Exec,Frozen,false><<<dim3(2*I/16,n),32*S,0,d.stream>>>(d.jobs,d.x,d.codes,d.workspace,up);
        ck(cudaGetLastError(),"Q1 raw up");
        q1_raw_kernel<S,Exec,Frozen,true><<<dim3(H/16,n),32*S,0,d.stream>>>(d.jobs,d.x,d.codes,d.workspace,down);
    }
    ck(cudaGetLastError(),"raw down");
}
template<int S>
void raw_dispatch(Direct& d,int exec,const Shape& shape,float* up,float* down) {
    switch(exec) {
        case -1:raw_pair<S,0,true>(d,shape,up,down);break;
        case 0:raw_pair<S,0,false>(d,shape,up,down);break;
        case 1:raw_pair<S,1,false>(d,shape,up,down);break;
        case 2:raw_pair<S,2,false>(d,shape,up,down);break;
        case 3:raw_pair<S,3,false>(d,shape,up,down);break;
        case 4:raw_pair<S,4,false>(d,shape,up,down);break;
        case 5:raw_pair<S,5,false>(d,shape,up,down);break;
        case 6:raw_pair<S,6,false>(d,shape,up,down);break;
        case 7:raw_pair<S,7,false>(d,shape,up,down);break;
        default:throw std::runtime_error("raw mode out of range");
    }
}
void raw_gate(Direct& d,int modes) {
    // Width1 selects Q1's four warps; widths3/8 select eight. Pattern1 covers
    // every finite activation scale, signed zeros and cancellation. Pattern2
    // keeps NaN propagation visible; comparisons are bits, never float ==.
    for(int width:{1,3,8})for(int pattern:{0,1,2}) {
        const Shape shape{{width,width,width}};const auto jobs=helper_jobs(shape,1);
        d.prepare(jobs,packed_rows(shape.tokens(),19,pattern));d.oracle(shape);
        const int S=shape.multi()?8:4;
        const std::size_t bytes=shape.widths.size()*(H/16)*128*(d.third?1:S)*sizeof(float);
        float *up=nullptr,*down=nullptr;ck(cudaMalloc(&up,bytes),"raw up buffer");ck(cudaMalloc(&down,bytes),"raw down buffer");
        std::vector<std::uint8_t> expected(2*bytes),got(2*bytes);
        const auto run=[&](int exec,std::vector<std::uint8_t>& target){
            ck(cudaMemsetAsync(up,0xa5,bytes,d.stream),"raw up sentinel");ck(cudaMemsetAsync(down,0x5a,bytes,d.stream),"raw down sentinel");
            if(S==4)raw_dispatch<4>(d,exec,shape,up,down);else raw_dispatch<8>(d,exec,shape,up,down);
            ck(cudaStreamSynchronize(d.stream),"raw snapshots retired");
            ck(cudaMemcpy(target.data(),up,bytes,cudaMemcpyDeviceToHost),"raw up back");
            ck(cudaMemcpy(target.data()+bytes,down,bytes,cudaMemcpyDeviceToHost),"raw down back");
        };
        run(-1,expected);
        for(int exec=0;exec<modes;++exec){run(exec,got);verdict(got==expected,"raw-accumulators exec="+std::to_string(exec)+" width="+std::to_string(width)+" pattern="+std::to_string(pattern));}
        ck(cudaFree(up),"raw up free");ck(cudaFree(down),"raw down free");
    }
    // Force exact cancellation with nonzero operands: every record row holds
    // alternating +6,-6 weights and the represented input is constant +6. Each
    // 16-element product is exactly zero before any BF16 rounding. This helper
    // belongs only to the standalone gate and is idle; restore its exact record
    // before subsequent correctness/timing work. No serving allocation or model
    // state is shared with this executable.
    const auto stride=d.helper.record_stride();
    auto* record=reinterpret_cast<std::uint8_t*>(d.helper.pool_identity())+std::size_t(d.ids[0].slot)*stride;
    std::vector<std::uint8_t> saved(stride),cancellation(stride,0);
    ck(cudaMemcpy(saved.data(),record,stride,cudaMemcpyDeviceToHost),"save cancellation record");
    for(std::size_t u=0;u+144<=stride;u+=144){for(int b=0;b<128;++b)cancellation[u+b]=(b&1)?0xff:0x77;for(int b=128;b<144;++b)cancellation[u+b]=0x38;}
    ck(cudaMemcpyAsync(record,cancellation.data(),stride,cudaMemcpyHostToDevice,d.stream),"cancellation record");ck(cudaStreamSynchronize(d.stream),"cancellation record retired");
    for(int width:{1,3,8}) {
        const Shape shape{{width}};const auto jobs=helper_jobs(shape,0);
        std::vector<std::uint8_t> rows(std::size_t(width)*q::kPackedRow,0x77);
        for(int t=0;t<width;++t)std::fill_n(rows.data()+std::size_t(t)*q::kPackedRow+H/2,H/16,0x38);
        d.prepare(jobs,rows);d.oracle(shape);const int S=shape.multi()?8:4;
        const std::size_t bytes=(H/16)*128*(d.third?1:S)*sizeof(float);
        float *up=nullptr,*down=nullptr;ck(cudaMalloc(&up,bytes),"cancellation raw up");ck(cudaMalloc(&down,bytes),"cancellation raw down");
        std::vector<std::uint8_t> expected(2*bytes),got(2*bytes);
        for(int exec=-1;exec<modes;++exec) {
            if(S==4)raw_dispatch<4>(d,exec,shape,up,down);else raw_dispatch<8>(d,exec,shape,up,down);
            ck(cudaStreamSynchronize(d.stream),"cancellation raw retired");
            auto& target=exec<0?expected:got;ck(cudaMemcpy(target.data(),up,bytes,cudaMemcpyDeviceToHost),"cancellation up back");ck(cudaMemcpy(target.data()+bytes,down,bytes,cudaMemcpyDeviceToHost),"cancellation down back");
            if(exec<0)verdict(std::all_of(expected.begin(),expected.end(),[](auto b){return b==0;}),"frozen exact nonzero-operand cancellation width="+std::to_string(width));
            else verdict(got==expected,"raw exact cancellation exec="+std::to_string(exec)+" width="+std::to_string(width));
        }
        ck(cudaFree(up),"cancellation raw up free");ck(cudaFree(down),"cancellation raw down free");
    }
    ck(cudaMemcpyAsync(record,saved.data(),stride,cudaMemcpyHostToDevice,d.stream),"restore cancellation record");ck(cudaStreamSynchronize(d.stream),"restored cancellation record retired");
}

void direct_gate(Direct& d,int modes) {
    std::vector<Shape> shapes;
    for(int width=1;width<=8;++width)for(int n:{1,3,8})shapes.push_back({std::vector<int>(std::size_t(n),width)});
    shapes.push_back({{8,1,3,7}});
    for(const auto& shape:shapes) {
        // Native sorting must be reflected in the direct launch's width list.
        Shape sorted=shape;std::sort(sorted.widths.begin(),sorted.widths.end());
        const auto jobs=helper_jobs(shape,0);const auto rows=packed_rows(shape.tokens(),0,0);
        d.prepare(jobs,rows);const auto expected=d.oracle(sorted);
        for(int exec=0;exec<modes;++exec) {
            d.clear();ck(d.launch(false,exec,sorted),"current direct");
            verdict(equal(expected,d.read(sorted)),"frozen/direct exec="+std::to_string(exec)+" widths="+shape.name());
            cudaGraph_t graph=nullptr;cudaGraphExec_t ge=nullptr;
            ck(cudaStreamBeginCapture(d.stream,cudaStreamCaptureModeThreadLocal),"capture execution");
            ck(d.launch(false,exec,sorted),"capture kernels");
            ck(cudaStreamEndCapture(d.stream,&graph),"capture end");
            ck(cudaGraphInstantiate(&ge,graph,nullptr,nullptr,0),"instantiate execution");
            bool exact=true;
            for(int change=1;change<=2;++change) {
                // Same pointers, new input and expert/multiplier descriptors.
                const auto changed_jobs=helper_jobs(shape,change);
                d.prepare(changed_jobs,packed_rows(shape.tokens(),change,change==2?1:0));
                const auto want=d.oracle(sorted);d.clear();ck(cudaGraphLaunch(ge,d.stream),"changed graph replay");
                exact &= equal(want,d.read(sorted));
            }
            verdict(exact,"frozen/graph changed-input exec="+std::to_string(exec)+" widths="+shape.name());
            ck(cudaGraphExecDestroy(ge),"destroy graph exec");ck(cudaGraphDestroy(graph),"destroy graph");
            d.prepare(jobs,rows);
        }
    }
    // Scale markers and +Inf*zero expose the changed hidden-encoding error path.
    // They are deliberately restricted to direct kernels; serving must refuse
    // a nonfinite input before publication, checked separately below.
    for(int width:{1,3,8})for(int pattern:{2,3}) {
        Shape shape{{width,width,width}};const auto jobs=helper_jobs(shape,0);
        d.prepare(jobs,packed_rows(shape.tokens(),3,pattern==2?2:1),pattern==3);
        const auto want=d.oracle(shape);
        for(int exec=0;exec<modes;++exec){d.clear();ck(d.launch(false,exec,shape),"nonfinite current");verdict(equal(want,d.read(shape)),"frozen/nonfinite exec="+std::to_string(exec)+" width="+std::to_string(width)+" pattern="+std::to_string(pattern));}
    }
}

Helper::PassTiming production_pass(Helper& helper,const Shape& shape,const std::vector<Helper::Job>& jobs,
                                   const std::vector<std::uint8_t>& rows,std::vector<float>& result) {
    std::string err;const auto seq=helper.submit_packed(shape.tokens(),rows.data(),jobs,err);require(seq!=0,"submit: "+err);
    Helper::PassTiming t;std::vector<Helper::Refused> refused;
    require(helper.collect(seq,result.data(),result.size(),err,&t,&refused),"collect: "+err);require(refused.empty(),"unexpected nonfinite production result");return t;
}
bool route_ok(const Helper::PassTiming& t,bool third,const Shape& shape,int exec,bool graph) {
    const int effective=(!third&&!shape.multi())?exec&~1:exec;
    return t.input_format==1 && t.kernel==(third?(shape.multi()?188:88):(shape.multi()?172:68)) && t.execution==effective &&
           t.multi==shape.multi() && t.sm89==third && t.graph==graph && !t.graph_fallback && !t.gpu_measured;
}
void production_gate(Direct& d,int modes) {
    Helper& helper=d.helper;std::string err;
    std::vector<Shape> shapes;
    for(int width=1;width<=8;++width)for(int n:{1,3,8})shapes.push_back({std::vector<int>(std::size_t(n),width)});
    shapes.push_back({{8,1,3,7}});
    for(bool graph:{false,true}) {
        helper.set_fast(graph);helper.set_zero_copy(graph?1:0);
        for(const auto& shape:shapes) {
            Shape sorted=shape;std::sort(sorted.widths.begin(),sorted.widths.end());
            for(int change=0;change<2;++change) {
                const auto jobs=helper_jobs(shape,change);const auto rows=packed_rows(shape.tokens(),11+change,change);
                d.prepare(jobs,rows);auto old_result=d.oracle(sorted);
                std::vector<float> expected(std::size_t(shape.tokens())*H),got(expected.size());
                require(helper.merge_external(shape.tokens(),jobs,old_result.output.data(),expected.data(),expected.size(),err),"frozen merge: "+err);
                // Reverse mode order on changed inputs and return to0. This
                // exercises retained caches after another mode has used buffers.
                for(int i=0;i<=modes;++i) {
                    const int exec=i==modes?0:(change?modes-1-i:i);
                    require(helper.set_execution(exec,err),"execution setting: "+err);
                    const auto before=helper.graph_launches();const auto t=production_pass(helper,shape,jobs,rows,got);
                    const bool exact=std::memcmp(expected.data(),got.data(),got.size()*4)==0;
                    verdict(exact && route_ok(t,d.third,shape,exec,graph) && helper.graph_launches()-before==unsigned(graph),
                        std::string("NativeHelper/")+(graph?"zero-copy-graph":"direct")+" exec="+std::to_string(exec)+" changed="+std::to_string(change)+" widths="+shape.name());
                }
            }
        }
    }
    Shape shape{{1}};const auto jobs=helper_jobs(shape,0);auto rows=packed_rows(1,0,0);std::vector<float> out(H);
    require(helper.set_execution(0,err),err);const auto seq=helper.submit_packed(1,rows.data(),jobs,err);require(seq!=0,err);
    const bool refused=!helper.set_execution(1,err);Helper::PassTiming t;
    require(helper.collect(seq,out.data(),out.size(),err,&t),err);
    verdict(refused && t.execution==0,"active submission refuses changed execution and retains snapshot");
    require(helper.set_execution(modes-1,err),err);
    rows[H/2]=0x7f;verdict(helper.submit_packed(1,rows.data(),jobs,err)==0,"nonfinite input refused before publication");
    rows=packed_rows(1,1,0);production_pass(helper,shape,jobs,rows,out);
    verdict(true,"clean request after nonfinite refusal");
    require(helper.set_execution(0,err),err);
    verdict(!helper.set_execution(d.third?8:4,err),"unsupported execution flags refused");
    require(helper.set_execution(0,err),err);
}

void timings(Direct& d,int modes,int iters) {
    Helper& helper=d.helper;helper.set_fast(true);helper.set_zero_copy(1);helper.set_occupancy(true);
    std::string err;
    const Shape shapes[]={{{1}},{{1,1,1,1}},{{4,4,4,4}},{{8,8,8,8}},{{1,3,7,8}},{{8,8,8,8,8,8,8,8}}};
    for(const auto& shape:shapes) {
        Shape sorted=shape;std::sort(sorted.widths.begin(),sorted.widths.end());
        const auto jobs=helper_jobs(shape,0);const auto rows=packed_rows(shape.tokens(),17,0);std::vector<float> output(std::size_t(shape.tokens())*H);
        std::vector<std::vector<double>> host(static_cast<std::size_t>(modes)),gpu(static_cast<std::size_t>(modes));
        // Warm every mode and graph before the rotation. CUDA timings run with
        // the helper spinner off to avoid unmodeled contention with that stream.
        for(int exec=0;exec<modes;++exec){require(helper.set_execution(exec,err),err);for(int n=0;n<3;++n){const auto t=production_pass(helper,shape,jobs,rows,output);require(route_ok(t,d.third,shape,exec,true),"timed route mismatch");}}
        for(int round=0;round<3;++round)for(int index=0;index<modes;++index) {
            const int exec=(index+round)%modes;require(helper.set_execution(exec,err),err);
            for(int n=0;n<iters/3;++n){const double start=now_ms();const auto t=production_pass(helper,shape,jobs,rows,output);host[exec].push_back(now_ms()-start);require(route_ok(t,d.third,shape,exec,true),"timed actual route mismatch");}
        }
        helper.set_occupancy(false);d.prepare(jobs,rows);
        cudaEvent_t begin=nullptr,end=nullptr;ck(cudaEventCreate(&begin),"timing event");ck(cudaEventCreate(&end),"timing event");
        for(int exec=0;exec<modes;++exec){for(int n=0;n<3;++n)ck(d.launch(false,exec,sorted),"kernel warm");}ck(cudaStreamSynchronize(d.stream),"kernel warm retired");
        for(int round=0;round<3;++round)for(int index=0;index<modes;++index) {
            const int exec=(index+round)%modes;ck(cudaEventRecord(begin,d.stream),"kernel timing begin");
            for(int n=0;n<iters/3;++n)ck(d.launch(false,exec,sorted),"kernel timing launch");
            ck(cudaEventRecord(end,d.stream),"kernel timing end");ck(cudaEventSynchronize(end),"kernel timing retired");float ms=0;ck(cudaEventElapsedTime(&ms,begin,end),"kernel elapsed");gpu[exec].push_back(ms/double(iters/3));
        }
        ck(cudaEventDestroy(begin),"event destroy");ck(cudaEventDestroy(end),"event destroy");helper.set_occupancy(true);
        for(int exec=0;exec<modes;++exec)std::printf("TIMING card=%s widths=%s execution=%d host_n=%zu host_p10_ms=%.6f host_median_ms=%.6f host_p90_ms=%.6f kernel_batch_n=3 kernel_median_ms=%.6f host_graph=1 host_zero_copy=1 kernel_graph=0 kernel_scope=prepared_up_down_batch format=1 sm89=%d\n",
            d.third?"4080":"3090",shape.name().c_str(),exec,host[exec].size(),quantile(host[exec],.1),quantile(host[exec],.5),quantile(host[exec],.9),quantile(gpu[exec],.5),int(d.third));
    }
    require(helper.set_execution(0,err),err);helper.set_occupancy(false);
}

void one_job_probe(Direct& d,int iters) {
    require(d.third,"one-job probe is bounded to the 4080 Q2 execution0/4 comparison");
    Helper& helper=d.helper;helper.set_fast(true);helper.set_zero_copy(1);helper.set_occupancy(false);
    const Shape shape{{1}};const auto rows=packed_rows(1,17,0);
    std::array<std::vector<Helper::Job>,4> plans;
    std::array<std::vector<float>,4> expected;
    std::string err;std::vector<float> output(H);
    for(int expert=0;expert<4;++expert) {
        plans[expert]=helper_jobs(shape,0);plans[expert][0].expert=expert;
        d.prepare(plans[expert],rows);auto frozen=d.oracle(shape);expected[expert].resize(H);
        require(helper.merge_external(1,plans[expert],frozen.output.data(),expected[expert].data(),H,err),"one-job frozen merge: "+err);
        for(int exec:{0,4}) {
            require(helper.set_execution(exec,err),err);const auto graphs=helper.graph_launches();
            const auto t=production_pass(helper,shape,plans[expert],rows,output);
            verdict(route_ok(t,true,shape,exec,true) && helper.graph_launches()==graphs+1 &&
                    std::memcmp(expected[expert].data(),output.data(),H*sizeof(float))==0,
                    "one-job probe frozen/production graph exec="+std::to_string(exec)+" expert="+std::to_string(expert));
        }
    }
    if(failures)return;
    cudaDeviceProp properties{};ck(cudaGetDeviceProperties(&properties,helper.device()),"probe cache properties");
    require(4*helper.record_stride()>std::size_t(properties.l2CacheSize),"four-record probe does not exceed reported L2");
    std::printf("ONE_JOB_PROBE card=4080 width=1 jobs=1 executions=0,4 record_bytes=%zu cycle_bytes=%zu l2_bytes=%d rounds=4 samples_per_arm=%d order=AB,BA,BA,AB graph_warm=1\n",
        helper.record_stride(),4*helper.record_stride(),properties.l2CacheSize,iters);
    struct Sample { int round,execution,expert;double ms; };
    for(bool rotating:{false,true}) {
        const char* kind=rotating?"rotating4":"fixed1";
        const auto expert_at=[&](int n){return rotating?n%4:0;};
        std::array<std::vector<double>,2> host,kernel;
        std::vector<Sample> samples;
        helper.set_occupancy(true);
        // Warm complete cycles in every mode, including the captured graph.
        // Each arm's timed block also ends on record3 before the next starts on
        // record0. No candidate follows a baseline on the same just-used record.
        for(int exec:{0,4}) {
            require(helper.set_execution(exec,err),err);
            for(int n=0;n<16;++n){const auto t=production_pass(helper,shape,plans[expert_at(n)],rows,output);require(route_ok(t,true,shape,exec,true),"probe warm route mismatch");}
        }
        const int per_round=iters/4;
        for(int round=0;round<4;++round)for(int i=0;i<2;++i) {
            const int first=(round==0||round==3)?0:1,index=first^i,exec=4*index;
            require(helper.set_execution(exec,err),err);
            for(int n=0;n<per_round;++n) {
                const int expert=expert_at(n);const auto graphs=helper.graph_launches();const double start=now_ms();
                const auto t=production_pass(helper,shape,plans[expert],rows,output);const double elapsed=now_ms()-start;
                require(route_ok(t,true,shape,exec,true)&&helper.graph_launches()==graphs+1,"probe timed route mismatch");
                host[index].push_back(elapsed);samples.push_back({round,exec,expert,elapsed});
            }
        }
        helper.set_occupancy(false);

        // Preload four immutable job descriptors, so rotating the GPU-only
        // batch requires no descriptor upload or synchronization between jobs.
        // This is the same prepared up/down interval as the first screen;
        // CUDA event batches can still contain host launch-submission gaps.
        d.prepare(plans[0],rows);
        std::array<Job,4> descriptors{};
        for(int expert=0;expert<4;++expert) {
            auto& job=descriptors[expert];job.record=reinterpret_cast<const std::uint8_t*>(helper.pool_identity())+std::size_t(d.ids[expert].slot)*helper.record_stride();
            job.x=d.x;job.y=d.y;job.columns=1;std::copy_n(d.ids[expert].multiplier,3,job.multiplier);
        }
        Job* device_jobs=nullptr;ck(cudaMalloc(&device_jobs,sizeof(descriptors)),"probe descriptor bank");
        ck(cudaMemcpyAsync(device_jobs,descriptors.data(),sizeof(descriptors),cudaMemcpyHostToDevice,d.stream),"probe descriptors up");
        ck(cudaStreamSynchronize(d.stream),"probe descriptors retired");
        const auto launch=[&](int exec,int n){ck(mm3::helper_sm89::q2_ffn_exec(exec,8,4,device_jobs+expert_at(n),1,d.x,d.packed,d.workspace,d.stream),"probe kernel pair");};
        for(int exec:{0,4})for(int n=0;n<16;++n)launch(exec,n);
        ck(cudaStreamSynchronize(d.stream),"probe kernel warm retired");
        cudaEvent_t begin=nullptr,end=nullptr;ck(cudaEventCreate(&begin),"probe begin event");ck(cudaEventCreate(&end),"probe end event");
        for(int round=0;round<4;++round)for(int i=0;i<2;++i) {
            const int first=(round==0||round==3)?0:1,index=first^i,exec=4*index;
            ck(cudaEventRecord(begin,d.stream),"probe event begin");
            for(int n=0;n<per_round;++n)launch(exec,n);
            ck(cudaEventRecord(end,d.stream),"probe event end");ck(cudaEventSynchronize(end),"probe interval retired");
            float elapsed=0;ck(cudaEventElapsedTime(&elapsed,begin,end),"probe elapsed");kernel[index].push_back(double(elapsed)/per_round);
        }
        ck(cudaEventDestroy(begin),"probe event destroy");ck(cudaEventDestroy(end),"probe event destroy");ck(cudaFree(device_jobs),"probe descriptor bank free");
        for(const auto& sample:samples)std::printf("PROBE_SAMPLE cache=%s round=%d execution=%d expert=%d host_ms=%.6f\n",kind,sample.round,sample.execution,sample.expert,sample.ms);
        for(int index=0;index<2;++index)std::printf("PROBE_TIMING cache=%s execution=%d host_n=%zu host_p10_ms=%.6f host_median_ms=%.6f host_p90_ms=%.6f kernel_batch_n=4 kernel_median_ms=%.6f host_graph=1 host_zero_copy=1 kernel_graph=0 kernel_scope=prepared_up_down_batch format=1 sm89=1\n",
            kind,4*index,host[index].size(),quantile(host[index],.1),quantile(host[index],.5),quantile(host[index],.9),quantile(kernel[index],.5));
        for(int round=0;round<4;++round)for(int index=0;index<2;++index)std::printf("PROBE_KERNEL_BATCH cache=%s round=%d execution=%d mean_ms=%.6f\n",kind,round,4*index,kernel[index][round]);
    }
    require(helper.set_execution(0,err),err);helper.set_occupancy(false);
}
} // namespace

int main(int argc,char** argv) {
    try {
        std::string card="3090";int iters=60;bool only=false,probe=false,iters_given=false;
        for(int a=1;a<argc;++a){const std::string arg=argv[a];if(arg=="--checks-only")only=true;else if(arg=="--one-job-probe")probe=true;else if(arg=="--card"&&a+1<argc)card=argv[++a];else if(arg=="--iters"&&a+1<argc){iters=std::stoi(argv[++a]);iters_given=true;}else throw std::runtime_error("usage: --card 3090|4080 [--iters N] [--checks-only | --one-job-probe]");}
        require(card=="3090"||card=="4080","card must be 3090 or 4080");require(!probe||(!only&&card=="4080"),"one-job probe requires 4080 and excludes checks-only");
        if(probe&&!iters_given)iters=64;
        require(probe?(iters>=16&&iters%16==0):(iters>=6&&iters%3==0),probe?"probe iters must be >=16 and divisible by 16":"iters must be >=6 and divisible by 3");
        const bool third=card=="4080";const int dev=find_card(card.c_str()),modes=third?8:4;ck(cudaSetDevice(dev),"set device");
        std::printf("helper_execution_bench card=%s device=%d frozen=7a2006db39ff66813e958b9cae30a631f68feebe modes=%d iters=%d checks_only=%d one_job_probe=%d\n",card.c_str(),dev,modes,iters,int(only),int(probe));
        Helper helper(dev,kExperts);helper.set_occupancy(false);helper.set_fast(true);helper.set_zero_copy(1);
        std::uint8_t* staging=nullptr;const auto stride=helper.record_stride();ck(cudaHostAlloc(reinterpret_cast<void**>(&staging),stride,cudaHostAllocPortable),"record staging");
        std::mt19937 rng(20261115);std::string err;
        for(int e=0;e<kExperts;++e) {
            for(std::size_t b=0;b<stride;++b)staging[b]=std::uint8_t(rng());
            for(std::size_t u=0;u+144<=stride;u+=144)for(int r=0;r<16;++r)staging[u+128+r]=std::uint8_t(0x28+rng()%24);
            const float wg[3]={2.0f+0.125f*e,2.5f,3.0f},ig[3]={1,1,1};
            require(helper.add_expert(kLayer,e,staging,stride,wg,ig,err),"load record: "+err);
        }
        ck(cudaFreeHost(staging),"release staging");require(helper.seal(err),"seal: "+err);
        helper.set_kernel(third?188:172,third?88:68);helper.set_loads(third?4:8);helper.set_sm89(third);
        require(!third||helper.sm89(),"4080 production SM89 build unavailable");
        Direct direct(third,helper);
        if(probe)one_job_probe(direct,iters);
        else {raw_gate(direct,modes);direct_gate(direct,modes);production_gate(direct,modes);if(!failures&&!only)timings(direct,modes,iters);}
        std::printf("RESULT %s checks=%d failures=%d\n",failures?"FAIL":"PASS",checks,failures);return failures?1:0;
    } catch(const std::exception& e){std::fprintf(stderr,"helper_execution_bench ERROR: %s\n",e.what());return 2;}
}
