// Bounded native adapter / HOST VMM transfer diagnostic; no model loading.
// Six fixture files contain RG16 payload followed by three FP32 multipliers.
#include "mm3_native_experts.cuh"
#include "mm3_native_host.hpp"
#include "ops/minimax_nvfp4_ffn/record.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
namespace ne=strata::mm3::native_experts;
namespace x3=strata::mm3::exl3;
namespace mf=infernix::ops::minimax_ffn;
using Clock=std::chrono::steady_clock;
constexpr int H=6144, FF=3072, RECORDS=6;
constexpr std::size_t GiB=std::size_t{1}<<30;
static void ck(cudaError_t e,const char* op){if(e!=cudaSuccess)throw std::runtime_error(std::string(op)+": "+cudaGetErrorString(e));}
struct Gpu {
    int id;std::size_t blob,workbytes;cudaStream_t stream=nullptr;cudaEvent_t start=nullptr,copied=nullptr,done=nullptr;
    std::uint8_t *staged=nullptr,*resident=nullptr;void* workspace=nullptr;float *x=nullptr,*out=nullptr;
    unsigned long long* plans=nullptr;
    Gpu(int dev,std::size_t bytes):id(dev),blob(bytes),workbytes(ne::scratch_bytes(H,FF)) {
        ck(cudaSetDevice(id),"set primary device");ck(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"create stream");
        ck(cudaEventCreate(&start),"create start event");ck(cudaEventCreate(&copied),"create copy event");ck(cudaEventCreate(&done),"create finish event");
        ck(cudaMalloc(&staged,blob),"allocate staging blob");ck(cudaMalloc(&resident,RECORDS*blob),"allocate resident blobs");
        ck(cudaMalloc(&workspace,workbytes),"allocate adapter workspace");ck(cudaMalloc(&x,H*sizeof(float)),"allocate input");ck(cudaMalloc(&out,H*sizeof(float)),"allocate output");
        ck(cudaMalloc(&plans,(RECORDS+1)*16*sizeof(unsigned long long)),"allocate persistent plans");
        if(!ne::prepare(stream,H,FF,workspace,workbytes))throw std::runtime_error("native adapter prepare failed");
        std::vector<float> input(H);for(int k=0;k<H;++k)input[k]=.01f*float((k*37)%101-50);
        ck(cudaMemcpy(x,input.data(),H*sizeof(float),cudaMemcpyHostToDevice),"upload input");
        std::array<std::array<unsigned long long,16>,RECORDS+1> host_plans{};
        for(int i=0;i<=RECORDS;++i) {
            auto* p=i==RECORDS?staged:resident+std::size_t(i)*blob;
            host_plans[i][0]=(unsigned long long)p;float one=1;std::memcpy(&host_plans[i][8],&one,sizeof(one));
        }
        ck(cudaMemcpy(plans,host_plans.data(),sizeof(host_plans),cudaMemcpyHostToDevice),"upload plans");
    }
    ~Gpu(){cudaSetDevice(id);if(stream)cudaStreamSynchronize(stream);ne::release(stream);cudaFree(plans);cudaFree(out);cudaFree(x);cudaFree(workspace);cudaFree(resident);cudaFree(staged);cudaEventDestroy(done);cudaEventDestroy(copied);cudaEventDestroy(start);cudaStreamDestroy(stream);}
    x3::MoeArgs args(int id) const {
        x3::MoeArgs a;a.plan_ptr=plans+std::size_t(id)*16;a.plan_w=(const float*)(a.plan_ptr+8);
        a.k=1;a.n_embd=H;a.n_ff=FF;a.limit=7.f;return a;
    }
    void forward(int id) {
        auto a=args(id);if(!ne::gate_up(a,x,stream)||!ne::down(a,nullptr,out,stream))throw std::runtime_error("native adapter failed");
    }
    void validate() {
        std::vector<float> host(H);ck(cudaMemcpy(host.data(),out,H*sizeof(float),cudaMemcpyDeviceToHost),"read output");
        for(float v:host)if(!std::isfinite(v))throw std::runtime_error("nonfinite adapter output");
    }
};
struct ScopedBank {
    Gpu& gpu;std::unique_ptr<mm3::NativeHostBank> bank;
    ScopedBank(Gpu& g,std::size_t bytes):gpu(g),bank(std::make_unique<mm3::NativeHostBank>(bytes,g.id,16*GiB,16*GiB)){}
    ~ScopedBank(){if(cudaSetDevice(gpu.id)==cudaSuccess)cudaStreamSynchronize(gpu.stream);}
};
struct Step {double enqueue_ms,wall_ms,copy_ms,ffn_ms;};
enum class Arm { Resident, CopyOnly, CopyAdapter };
static const char* name(Arm a){return a==Arm::Resident?"resident_adapter":a==Arm::CopyOnly?"copy_only":"copy_plus_adapter";}
static Step step(Gpu& gpu,const std::byte* source,int record,Arm arm) {
    const auto wall=Clock::now();ck(cudaEventRecord(gpu.start,gpu.stream),"start timing");
    if(arm!=Arm::Resident)ck(cudaMemcpyAsync(gpu.staged,source,gpu.blob,cudaMemcpyHostToDevice,gpu.stream),"stage host blob");
    ck(cudaEventRecord(gpu.copied,gpu.stream),"copy timing");
    if(arm!=Arm::CopyOnly)gpu.forward(arm==Arm::Resident?record:RECORDS);
    ck(cudaEventRecord(gpu.done,gpu.stream),"finish timing");
    const auto enqueued=Clock::now();ck(cudaEventSynchronize(gpu.done),"retire timed service");const auto finished=Clock::now();
    float copy=0,ffn=0;if(arm!=Arm::Resident)ck(cudaEventElapsedTime(&copy,gpu.start,gpu.copied),"copy elapsed");
    if(arm!=Arm::CopyOnly)ck(cudaEventElapsedTime(&ffn,gpu.copied,gpu.done),"adapter elapsed");
    return {std::chrono::duration<double,std::milli>(enqueued-wall).count(),std::chrono::duration<double,std::milli>(finished-wall).count(),copy,ffn};
}
static void measure(Gpu& gpu,const mm3::NativeHostBank& bank,std::size_t blob,const char* bank_label,Arm arm) {
    unsigned seq=0;auto one=[&]{int r=(seq++)%RECORDS;return step(gpu,bank.data()+std::size_t(r)*blob,r,arm);};
    auto start=Clock::now();int warm=0;do{(void)one();++warm;}while(std::chrono::duration<double>(Clock::now()-start).count()<.5);
    double warm_ms=std::chrono::duration<double,std::milli>(Clock::now()-start).count();
    start=Clock::now();int n=0;Step total{};do{auto t=one();total.enqueue_ms+=t.enqueue_ms;total.wall_ms+=t.wall_ms;total.copy_ms+=t.copy_ms;total.ffn_ms+=t.ffn_ms;++n;}while(std::chrono::duration<double>(Clock::now()-start).count()<.1);
    // Validate every fixture after the timed interval, without changing timing.
    for(int r=0;r<RECORDS;++r) {
        (void)step(gpu,bank.data()+std::size_t(r)*blob,r,arm);
        if(arm==Arm::CopyOnly){gpu.forward(RECORDS);ck(cudaStreamSynchronize(gpu.stream),"validate staged output");}
        gpu.validate();
    }
    std::printf("bank=%s arm=%s columns=1 warm_calls=%d warm_wall_ms=%.3f calls=%d enqueue_total_ms=%.3f enqueue_ms_per_call=%.6f completed_wall_total_ms=%.3f completed_wall_ms_per_call=%.6f copy_event_total_ms=%.3f copy_event_ms_per_call=%.6f adapter_event_total_ms=%.3f adapter_event_ms_per_call=%.6f output_finite=1\n",bank_label,name(arm),warm,warm_ms,n,total.enqueue_ms,total.enqueue_ms/n,total.wall_ms,total.wall_ms/n,total.copy_ms,total.copy_ms/n,total.ffn_ms,total.ffn_ms/n);
    std::fflush(stdout);
}
int main(int argc,char** argv) {
    if(argc!=3){std::fprintf(stderr,"usage: native_transfer_probe six_record_manifest primary_device\n");return 2;}
    try {
        int device=std::atoi(argv[2]);if(device<0)throw std::runtime_error("invalid CUDA device");
        std::ifstream manifest(argv[1]);if(!manifest)throw std::runtime_error("cannot open manifest");
        std::vector<std::string> files;std::string line;while(std::getline(manifest,line)){if(!line.empty()&&line.back()=='\r')line.pop_back();if(!line.empty())files.push_back(line);}
        if(files.size()!=RECORDS||std::set<std::string>(files.begin(),files.end()).size()!=RECORDS)throw std::runtime_error("expected six distinct fixture paths");
        const auto payload=mf::record_stride(H,FF),blob=sizeof(ne::Header)+payload;
        // CUDA context and every persistent buffer are initialized before either host bank.
        Gpu gpu(device,blob);
        std::printf("ledger device=%d fixture_payload=%zu record_header=%zu staging_device=%zu resident_device=%zu adapter_workspace=%zu input_output_device=%zu persistent_plan_device=%zu reserve_physical_and_commit_each=%zu\n",device,payload,sizeof(ne::Header),blob,RECORDS*blob,gpu.workbytes,2*H*sizeof(float),(RECORDS+1)*16*sizeof(unsigned long long),16*GiB);
        for(int r=0;r<RECORDS;++r)std::printf("fixture id=%d path=%s\n",r,files[r].c_str());
        for(int pass=0;pass<2;++pass) {
            const std::size_t bytes=pass?4*GiB:RECORDS*blob;const char* label=pass?"4GiB":"six_record";
            ScopedBank scope(gpu,bytes);auto& bank=*scope.bank;
            for(int r=0;r<RECORDS;++r){std::ifstream f(files[r],std::ios::binary|std::ios::ate);if(!f||f.tellg()!=std::streamoff(payload+12))throw std::runtime_error("invalid fixture: "+files[r]);
                auto* b=bank.data()+std::size_t(r)*blob;std::memset(b,0,sizeof(ne::Header));f.seekg(0);f.read((char*)b+sizeof(ne::Header),payload);float multiplier[3];f.read((char*)multiplier,sizeof(multiplier));if(!f)throw std::runtime_error("short fixture read");
                auto* h=(ne::Header*)b;for(int k=0;k<3;++k){if(!std::isfinite(multiplier[k])||multiplier[k]<=0)throw std::runtime_error("invalid fixture scale");h->weight_global[k]=1.f/multiplier[k];h->input_global[k]=1.f;}
                ck(cudaMemcpyAsync(gpu.resident+std::size_t(r)*blob,b,blob,cudaMemcpyHostToDevice,gpu.stream),"upload resident fixture");}
            ck(cudaStreamSynchronize(gpu.stream),"complete fixture uploads");
            std::printf("bank=%s requested=%zu capacity=%zu allocation_count=%zu max_allocation=%zu populated_records_bytes=%zu note=only_six_records_touched\n",label,bytes,bank.capacity(),bank.allocation_count(),bank.max_allocation_bytes(),RECORDS*blob);std::fflush(stdout);
            for(Arm arm:{Arm::Resident,Arm::CopyOnly,Arm::CopyAdapter})measure(gpu,bank,blob,label,arm);
        }
        return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"native_transfer_probe: %s\n",e.what());return 1;}
}
