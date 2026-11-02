// Manual, allocation-bearing real-CUDA helper refill test. It is not a full
// Model or performance qualification. Arguments: A.rg16 B.rg16 [primary] [helper].
// RG16 files contain one real payload followed by three reciprocal weight scales.
// Link against mm3_native (which owns NativeHelper, NativeHostBank and the packed
// FFN implementation), CUDA::cuda_driver, CUDA::cudart and Threads::Threads.
#include "../src/mm3/mm3_native_helper.hpp"
#include "../src/mm3/mm3_native_host.hpp"
#include "../src/mm3/mm3_native_experts.cuh"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include "ops/minimax_nvfp4_ffn/record.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

using H=mm3::NativeHelper;
namespace ne=strata::mm3::native_experts;
namespace ff=infernix::ops::minimax_ffn;
static void require(bool ok,const std::string& what) { if(!ok)throw std::runtime_error(what); }
static void ck(cudaError_t e,const char* what) { if(e!=cudaSuccess)throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(e)); }
static void driver(CUresult e,const char* what) {
    if(e!=CUDA_SUCCESS) {const char* name=nullptr;(void)cuGetErrorName(e,&name);throw std::runtime_error(std::string(what)+": "+(name?name:"unknown"));}
}
#define CUDA(x) ck((x),#x)
struct RealRecord {std::vector<uint8_t> payload;ne::Header header{};};
static RealRecord read_record(const char* path,size_t stride,float input_base) {
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    require(bool(in),std::string("cannot open record ")+path);
    require(in.tellg()==std::streamoff(stride+3*sizeof(float)),"invalid RG16 record or scale trailer size");
    in.seekg(0);RealRecord r;r.payload.resize(stride);float multiplier[3]{};
    in.read(reinterpret_cast<char*>(r.payload.data()),stride);
    in.read(reinterpret_cast<char*>(multiplier),sizeof(multiplier));require(bool(in),"short RG16 record");
    for(int i=0;i<3;++i) {
        require(std::isfinite(multiplier[i]) && multiplier[i]>0,"invalid real record multiplier");
        r.header.weight_global[i]=1.0f/multiplier[i];r.header.input_global[i]=input_base+i;
        require(std::isfinite(r.header.weight_global[i]) && r.header.weight_global[i]>0,"invalid original divisor");
    }
    return r;
}
static float bf16(uint16_t b) {uint32_t bits=uint32_t(b)<<16;float f;std::memcpy(&f,&bits,sizeof(f));return f;}
static void equal_output(const std::vector<float>& got,const std::vector<float>& want,const char* label) {
    require(got.size()==want.size(),"output geometry");
    for(size_t i=0;i<got.size();++i) if(!std::isfinite(got[i]) || got[i]!=want[i]) {
        std::fprintf(stderr,"%s: output[%zu] got %.9g expected %.9g\n",label,i,got[i],want[i]);
        throw std::runtime_error("helper/direct packed arithmetic mismatch");
    }
}
// Heap-owned on purpose. If any operation fails, main exits without destroying
// this object: unknown GPU/CPU work must not outlive the borrowed bank/buffers.
struct Fixture {
    int primary,device;size_t stride,blob;
    mm3::NativeHostBank* bank=nullptr;H* helper=nullptr;
    uint8_t *incoming=nullptr,*outgoing=nullptr,*slot=nullptr;
    uint8_t* d_record=nullptr;uint16_t *d_x=nullptr,*d_y=nullptr;
    ff::FastJob* d_job=nullptr;void* d_work=nullptr;cudaStream_t stream=nullptr;
    std::vector<uint16_t> x;
    uintptr_t pool=0;H::Bytes allocations{};int original_slot=-1,unchanged_slot=-1;
    Fixture(int p,int d,size_t n):primary(p),device(d),stride(n),blob(sizeof(ne::Header)+n) {}
    void init(const RealRecord& a,const RealRecord& b) {
        CUDA(cudaSetDevice(primary));
        bank=new mm3::NativeHostBank(blob,primary);
        slot=reinterpret_cast<uint8_t*>(bank->data());
        std::memcpy(slot,&a.header,sizeof(a.header));std::memcpy(slot+sizeof(ne::Header),a.payload.data(),stride);
        driver(cuInit(0),"cuInit");
        CUmemLocation loc{};loc.type=CU_MEM_LOCATION_TYPE_DEVICE;loc.id=primary;
        unsigned long long access=0;driver(cuMemGetAccess(&access,&loc,reinterpret_cast<CUdeviceptr>(slot)),"primary VMM access");
        require(access==CU_MEM_ACCESS_FLAGS_PROT_READWRITE,"HOST VMM must be primary-readable/writable");
        loc.id=device;driver(cuMemGetAccess(&access,&loc,reinterpret_cast<CUdeviceptr>(slot)),"helper VMM access");
        require(access==CU_MEM_ACCESS_FLAGS_PROT_NONE,"HOST VMM unexpectedly accessible to helper");
        loc.type=CU_MEM_LOCATION_TYPE_HOST;loc.id=0;
        driver(cuMemGetAccess(&access,&loc,reinterpret_cast<CUdeviceptr>(slot)),"CPU VMM access");
        require(access==CU_MEM_ACCESS_FLAGS_PROT_READWRITE,"HOST VMM must be CPU-readable/writable");
        CUDA(cudaHostAlloc(&incoming,blob,cudaHostAllocPortable));CUDA(cudaHostAlloc(&outgoing,blob,cudaHostAllocPortable));
        CUDA(cudaSetDevice(device));
        unsigned flags=0;CUDA(cudaHostGetFlags(&flags,incoming+sizeof(ne::Header)));
        require((flags&cudaHostAllocPortable)!=0,"incoming payload must be portable pinned");
        CUDA(cudaHostGetFlags(&flags,outgoing+sizeof(ne::Header)));
        require((flags&cudaHostAllocPortable)!=0,"outgoing payload must be portable pinned");
        CUDA(cudaMalloc(&d_record,stride));CUDA(cudaMalloc(&d_x,size_t(H::kHidden)*sizeof(uint16_t)));
        CUDA(cudaMalloc(&d_y,size_t(H::kHidden)*sizeof(uint16_t)));CUDA(cudaMalloc(&d_job,sizeof(ff::FastJob)));
        CUDA(cudaMalloc(&d_work,ff::fast_workspace_bytes(H::kHidden,H::kIntermediate,1)));
        CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        const uint16_t values[]={0xbf00,0xbe80,0x3e80,0x3f00};
        x.resize(H::kHidden);for(int i=0;i<H::kHidden;++i)x[size_t(i)]=values[(i*3)%4];
        CUDA(cudaMemcpy(d_x,x.data(),x.size()*sizeof(uint16_t),cudaMemcpyHostToDevice));
        std::memcpy(incoming+sizeof(ne::Header),b.payload.data(),stride);
        helper=new H(device,2);std::string error;
        require(helper->add_expert(3,0,incoming+sizeof(ne::Header),stride,b.header.weight_global,b.header.input_global,error),"add B: "+error);
        require(helper->add_expert(3,2,incoming+sizeof(ne::Header),stride,b.header.weight_global,b.header.input_global,error),"add unchanged C: "+error);
        require(helper->seal(error),"seal: "+error);
        pool=helper->pool_identity();allocations=helper->bytes();
        for(const auto& r:helper->records()) {if(r.expert==0)original_slot=r.slot;if(r.expert==2)unchanged_slot=r.slot;}
        require(pool && original_slot>=0 && unchanged_slot>=0 && original_slot!=unchanged_slot,"initial helper pool identities");
    }
    std::vector<float> direct(const RealRecord& r) {
        CUDA(cudaSetDevice(device));
        CUDA(cudaMemcpy(d_record,r.payload.data(),stride,cudaMemcpyHostToDevice));
        // Pageable H2D may use a default-stream staging transfer.
        CUDA(cudaStreamSynchronize(nullptr));
        ff::FastJob job{};job.record=d_record;job.x=d_x;job.y=d_y;job.columns=1;
        for(int i=0;i<3;++i)job.multiplier[i]=1.0f/r.header.weight_global[i];
        CUDA(cudaMemcpy(d_job,&job,sizeof(job),cudaMemcpyHostToDevice));CUDA(cudaStreamSynchronize(nullptr));
        ff::fast_ffn_cuda_packed(d_job,1,H::kHidden,H::kIntermediate,1,d_work,stream);
        CUDA(cudaStreamSynchronize(stream));std::vector<uint16_t> out(x.size());
        CUDA(cudaMemcpy(out.data(),d_y,out.size()*sizeof(uint16_t),cudaMemcpyDeviceToHost));
        std::vector<float> result(out.size());for(size_t i=0;i<out.size();++i)result[i]=bf16(out[i]);
        require(std::any_of(result.begin(),result.end(),[](float f){return std::isfinite(f)&&f!=0;}),"real direct FFN output is all zero or nonfinite");
        return result;
    }
    std::vector<float> compute(int expert) {
        H::Job job{};job.layer=3;job.expert=expert;job.columns=1;job.weight[0]=1;
        std::string error;auto seq=helper->submit(1,x.data(),{job},error);require(seq!=0,"helper submit: "+error);
        std::vector<float> out(x.size());require(helper->collect(seq,out.data(),out.size(),error),"helper collect: "+error);return out;
    }
    void wait(H::LivePhase desired) {
        const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(20);
        for(;;) {
            const auto s=helper->live_poll();
            if(s.phase==desired)return;
            require(s.phase!=H::LivePhase::failed && s.phase!=H::LivePhase::idle,"live helper failed: "+s.error);
            require(std::chrono::steady_clock::now()<end,"live helper progress timeout");
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    void check_pool(int current) {
        const auto size=helper->bytes();require(helper->pool_identity()==pool && size.pool==allocations.pool &&
            size.device_scratch==allocations.device_scratch && size.pinned==allocations.pinned,"helper allocations changed");
        int slot_current=-1,slot_c=-1;const auto ids=helper->records();
        require(ids.size()==2,"helper capacity changed");
        for(const auto& id:ids) {if(id.layer==3 && id.expert==current)slot_current=id.slot;if(id.layer==3 && id.expert==2)slot_c=id.slot;}
        require(slot_current==original_slot && slot_c==unchanged_slot,"helper pool frame changed");
    }
    void cycle(int old_id,int new_id,const RealRecord& old,const RealRecord& next,
               const std::vector<float>& old_ref,const std::vector<float>& new_ref,const std::vector<float>& c_ref) {
        const auto hdr=*reinterpret_cast<const ne::Header*>(slot);
        require(std::memcmp(&hdr,&next.header,sizeof(hdr))==0 &&
                std::memcmp(slot+sizeof(hdr),next.payload.data(),stride)==0,"incoming HOST record changed");
        auto* exported=reinterpret_cast<ne::Header*>(outgoing);std::memset(exported,0,sizeof(*exported));
        std::string error;
        require(helper->live_begin(3,old_id,3,new_id,slot+sizeof(hdr),incoming+sizeof(hdr),outgoing+sizeof(hdr),stride,
            hdr.weight_global,hdr.input_global,exported->weight_global,exported->input_global,error),"live begin: "+error);
        require(helper->ready(3)[0] & (1u<<old_id),"old record unavailable during export");
        equal_output(compute(old_id),old_ref,"old during export");
        wait(H::LivePhase::prepared);
        require((helper->ready(3)[0]&(1u<<old_id)) && !(helper->ready(3)[0]&(1u<<new_id)),"prepared readiness");
        require(std::memcmp(incoming+sizeof(hdr),next.payload.data(),stride)==0 &&
                std::memcmp(outgoing+sizeof(hdr),old.payload.data(),stride)==0,"exported/prepared payload mismatch");
        require(std::memcmp(exported,&old.header,sizeof(*exported))==0,"exported original scales mismatch");
        require(helper->live_withdraw(error),"live withdraw: "+error);
        require(!(helper->ready(3)[0]&(1u<<old_id)) && !(helper->ready(3)[0]&(1u<<new_id)),"withdraw readiness");
        const auto during_upload=helper->records();
        require(during_upload.size()==1 && during_upload[0].layer==3 && during_upload[0].expert==2 &&
                during_upload[0].slot==unchanged_slot,"temporary helper capacity or unchanged frame mismatch");
        equal_output(compute(2),c_ref,"unrelated helper during upload");
        wait(H::LivePhase::uploaded);
        require(!(helper->ready(3)[0]&(1u<<new_id)),"uploaded record published autonomously");
        require(std::memcmp(slot,&next.header,sizeof(hdr))==0 &&
                std::memcmp(slot+sizeof(hdr),next.payload.data(),stride)==0,"incoming HOST source changed before publication");
        require(helper->live_publish(error),"live publish: "+error);
        require(helper->ready(3)[0]&(1u<<new_id),"new record was not published");
        equal_output(compute(new_id),new_ref,"new helper/direct output");check_pool(new_id);
        // The caller has removed A's canonical HOST ownership and joined its
        // readers before reusing that frame for B's exported header/payload.
        std::memcpy(slot,outgoing,sizeof(hdr));
        require(helper->live_recycle(error),"live recycle: "+error);
        equal_output(compute(2),c_ref,"unrelated helper during CPU recycle");
        wait(H::LivePhase::complete);bool safe=false;
        require(std::memcmp(slot,&old.header,sizeof(hdr))==0 &&
                std::memcmp(slot+sizeof(hdr),old.payload.data(),stride)==0,"recycled HOST record mismatch");
        require(helper->live_retire(safe,error) && safe,"live retire: "+error);
        check_pool(new_id);
    }
    void finish() {
        delete helper;helper=nullptr;CUDA(cudaSetDevice(device));
        CUDA(cudaStreamDestroy(stream));CUDA(cudaFree(d_work));CUDA(cudaFree(d_job));CUDA(cudaFree(d_y));CUDA(cudaFree(d_x));CUDA(cudaFree(d_record));
        CUDA(cudaFreeHost(incoming));CUDA(cudaFreeHost(outgoing));delete bank;bank=nullptr;
    }
};
int main(int argc,char** argv) {
    try {
        require(argc>=3,"usage: test A.rg16 B.rg16 [primary CUDA ordinal] [helper CUDA ordinal]");
        const int primary=argc>3?std::atoi(argv[3]):0,device=argc>4?std::atoi(argv[4]):1;
        require(primary!=device,"primary and helper devices must differ");
        cudaDeviceProp p{},h{};CUDA(cudaGetDeviceProperties(&p,primary));CUDA(cudaGetDeviceProperties(&h,device));
        require(std::strstr(p.name,"5090") && std::strstr(h.name,"3090"),"pass the primary 5090 and helper 3090 ordinals");
        const size_t stride=ff::record_stride(H::kHidden,H::kIntermediate);
        const auto a=read_record(argv[1],stride,1.0f),b=read_record(argv[2],stride,4.0f);
        auto* f=new Fixture(primary,device,stride);f->init(a,b);
        const auto ar=f->direct(a),br=f->direct(b);
        require(!std::equal(ar.begin(),ar.end(),br.begin()),"real experts produce indistinguishable outputs");
        equal_output(f->compute(0),br,"initial B/helper direct");equal_output(f->compute(2),br,"initial C/helper direct");
        f->cycle(0,1,b,a,br,ar,br);f->cycle(1,0,a,b,ar,br,br);
        f->finish();delete f;
        std::printf("real RG16 live helper: two same-layer cycles OK; primary=%s helper=%s\n",p.name,h.name);
        return 0;
    } catch(const std::exception& e) {
        std::fprintf(stderr,"real RG16 live helper failed: %s\n",e.what());std::fflush(stderr);
        // A failed copy or query does not prove borrowed storage is retired.
        std::_Exit(1);
    }
}
