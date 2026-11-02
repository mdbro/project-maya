// Real production Service/codec/adapter fixture, invoked as:
// mm3_native_scale_store_cuda_test DEVICE SIX_RECORD_DIRECTORY
// The parent launches each case as a fresh process. Failed/timeout services
// deliberately retain uncertain resources until that child exits.
#include "mm3/mm3_native_scale_store.hpp"
#include "mm3/mm3_native_experts.cuh"
#include <cuda_runtime.h>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>
#include <utility>
#include <sys/types.h>
#include <sys/wait.h>
#include <signal.h>
#include <unistd.h>
namespace sc=mm3::scale_store;
namespace nx=strata::mm3::native_experts;
namespace x3=strata::mm3::exl3;
using Clock=std::chrono::steady_clock;
static void req(bool b,const std::string& s){if(!b)throw std::runtime_error(s);}
static void ck(cudaError_t e,const char* s){if(e!=cudaSuccess)throw std::runtime_error(std::string(s)+": "+cudaGetErrorString(e));}
#define CK(x) ck((x),#x)
static void wait_until(const std::function<bool()>& f,const char* what,int seconds=30) {
    auto end=Clock::now()+std::chrono::seconds(seconds);
    while(!f()){req(Clock::now()<end,std::string("timeout: ")+what);std::this_thread::sleep_for(std::chrono::milliseconds(1));}
}
struct Record {std::vector<uint8_t> raw; uint8_t* encoded=nullptr;size_t bytes=0;};
static Record load(const std::string& dir,int layer,int expert) {
    Record r;r.raw.resize(sc::kRawBytes);
    std::string path=dir+"/l"+std::to_string(layer)+"-e"+std::to_string(expert)+".rg16";
    std::ifstream in(path,std::ios::binary|std::ios::ate);
    req(bool(in)&&in.tellg()==std::streamoff(sc::kRawBytes-sc::kHeaderBytes+12),"bad record "+path);in.seekg(0);
    in.read(reinterpret_cast<char*>(r.raw.data()+sc::kHeaderBytes),sc::kRawBytes-sc::kHeaderBytes);
    float mult[3];in.read(reinterpret_cast<char*>(mult),sizeof(mult));req(bool(in),"short record");
    nx::Header header{};for(int i=0;i<3;++i){req(std::isfinite(mult[i])&&mult[i]>0,"invalid multiplier");header.weight_global[i]=1.f/mult[i];header.input_global[i]=1.f;}
    std::memcpy(r.raw.data(),&header,sizeof(header));
    CK(cudaMallocHost(reinterpret_cast<void**>(&r.encoded),sc::kRawBytes+1048576));
    std::string e;req(sc::encode(r.raw.data(),r.raw.size(),r.encoded,sc::kRawBytes+1048576,r.bytes,e),e);
    return r;
}
struct Hook {std::atomic<bool> gate{false},entered{false};sc::Operation failure=sc::Operation::HeaderCopy;bool inject=false,throw_gate=false,throw_unknown=false;};
static cudaError_t before(void* p,sc::Operation op){auto& h=*static_cast<Hook*>(p);return h.inject&&op==h.failure?cudaErrorUnknown:cudaSuccess;}
static void gate(void* p,int){auto& h=*static_cast<Hook*>(p);h.entered.store(true,std::memory_order_release);if(h.throw_unknown)throw 7;if(h.throw_gate)throw std::runtime_error("injected gate exception");while(!h.gate.load(std::memory_order_acquire))std::this_thread::sleep_for(std::chrono::milliseconds(1));}
struct Setup {
    uint8_t *fg=nullptr,*opt=nullptr,*dest=nullptr;cudaStream_t fs=nullptr,os=nullptr;
    Setup(size_t n) {CK(cudaStreamCreateWithFlags(&fs,cudaStreamNonBlocking));CK(cudaStreamCreateWithFlags(&os,cudaStreamNonBlocking));CK(cudaMalloc(&fg,sc::kScratchBytes));CK(cudaMalloc(&opt,sc::kScratchBytes));CK(cudaMalloc(&dest,n*sc::kRawBytes));}
    void cleanup(){CK(cudaStreamSynchronize(fs));CK(cudaStreamSynchronize(os));CK(cudaFree(dest));CK(cudaFree(opt));CK(cudaFree(fg));CK(cudaStreamDestroy(os));CK(cudaStreamDestroy(fs));}
};
struct Plan {unsigned long long pp[128]{};float w[128]{};unsigned long long gp[128]{};int gs[129]{},ge[128]{},ng=0;};
static void adapter_check(cudaStream_t s,uint8_t* frames,const std::vector<Record>& recs) {
    constexpr int H=6144,F=3072,T=16;
    uint8_t* direct=nullptr;void* workspace=nullptr;Plan *dp=nullptr,*hp=nullptr;
    float *dx=nullptr,*shared=nullptr,*part=nullptr,*out=nullptr;
    CK(cudaMalloc(&direct,sc::kRawBytes));size_t wb=nx::scratch_bytes(H,F,nx::Arithmetic::NativeFp4);CK(cudaMalloc(&workspace,wb));
    CK(cudaMalloc(&dp,sizeof(Plan)));CK(cudaMallocHost(&hp,sizeof(Plan)));
    CK(cudaMalloc(&dx,T*H*sizeof(float)));CK(cudaMalloc(&shared,T*H*sizeof(float)));
    CK(cudaMalloc(&part,T*8*H*sizeof(float)));CK(cudaMalloc(&out,T*H*sizeof(float)));
    std::vector<float> x(T*H),sh(T*H),a(T*H),b(T*H);
    for(int t=0;t<T;++t)for(int i=0;i<H;++i){x[t*H+i]=float((i*7+t*3)%17-8)/32;sh[t*H+i]=float((i+t)%7-3)/64;}
    CK(cudaMemcpyAsync(dx,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice,s));CK(cudaMemcpyAsync(shared,sh.data(),sh.size()*sizeof(float),cudaMemcpyHostToDevice,s));
    req(nx::prepare(s,H,F,workspace,wb,nx::Arithmetic::NativeFp4),"adapter prepare");
    x3::WinArgs args{};args.grp_ptr=dp->gp;args.grp_start=dp->gs;args.grp_ent=dp->ge;args.n_groups=&dp->ng;args.plan_ptr=dp->pp;args.plan_w=dp->w;args.x=dx;args.x_stride=H;args.part=part;args.sh_out=shared;args.out_stride=H;args.k=1;args.n_embd=H;args.n_ff=F;args.limit=7;args.out=out;
    auto run=[&](int width,uint8_t* record,std::vector<float>& result){
        *hp=Plan{};hp->ng=1;hp->gp[0]=(unsigned long long)(uintptr_t)record;hp->gs[1]=width;
        for(int i=0;i<width;++i){hp->pp[i*8]=1;hp->w[i*8]=1;hp->ge[i]=i*8;}
        CK(cudaMemcpyAsync(dp,hp,sizeof(Plan),cudaMemcpyHostToDevice,s));CK(cudaMemsetAsync(part,0xff,T*8*H*sizeof(float),s));args.T=width;
        req(nx::window(args,s),"adapter window");CK(cudaMemcpyAsync(result.data(),out,width*H*sizeof(float),cudaMemcpyDeviceToHost,s));CK(cudaStreamSynchronize(s));
    };
    for(int j=0;j<6;++j)for(int width:{1,8,16}) {
        CK(cudaMemcpyAsync(direct,recs[j].raw.data(),sc::kRawBytes,cudaMemcpyHostToDevice,s));
        run(width,direct,a);run(width,frames+size_t(j)*sc::kRawBytes,b);
        bool nonzero=false;for(int i=0;i<width*H;++i){req(std::isfinite(a[i])&&std::isfinite(b[i])&&a[i]==b[i],"adapter output mismatch");nonzero|=a[i]!=sh[i];}req(nonzero,"adapter produced no contribution");
    }
    CK(cudaStreamSynchronize(s));nx::release(s);CK(cudaFree(out));CK(cudaFree(part));CK(cudaFree(shared));CK(cudaFree(dx));CK(cudaFreeHost(hp));CK(cudaFree(dp));CK(cudaFree(workspace));CK(cudaFree(direct));
}
static sc::Operation operation(const std::string& name){
    if(name=="header")return sc::Operation::HeaderCopy;if(name=="code")return sc::Operation::CodeCopy;
    if(name=="progress")return sc::Operation::ProgressEventRecord;if(name=="query")return sc::Operation::ProgressEventQuery;
    if(name=="scale")return sc::Operation::ScaleCopy;if(name=="interleave")return sc::Operation::Interleave;
    if(name=="completion")return sc::Operation::CompletionEventRecord;throw std::runtime_error("unknown fault");
}
static void child(int device,const std::string& dir,const std::string& mode) {
    CK(cudaSetDevice(device));const bool success=mode=="success";Setup a(success?68:2);Hook h;
    sc::TestHooks hooks{&h,before,nullptr};if(success||mode=="deadline"||mode=="throw"||mode=="throw-unknown")hooks.optional_gate=gate;
    std::vector<Record> r;r.push_back(load(dir,3,0));if(success)for(auto pair: {std::pair{3,1},std::pair{30,0},std::pair{30,1},std::pair{59,0},std::pair{59,1}})r.push_back(load(dir,pair.first,pair.second));
    if(mode.rfind("fg-",0)==0||mode.rfind("opt-",0)==0){h.inject=true;h.failure=operation(mode.substr(mode.find('-')+1));}
    h.throw_gate=mode=="throw";h.throw_unknown=mode=="throw-unknown";std::string e;sc::Service svc(device,a.fg,a.fs,a.opt,a.os,mode=="deadline"?100:30000,e,&hooks);req(svc.healthy(),e);
    cudaEvent_t events[4]{};for(auto& ev:events)CK(cudaEventCreateWithFlags(&ev,cudaEventDisableTiming));
    if(success){
        for(int i=0;i<4;++i)req(svc.enqueue_optional(100+i,a.dest+size_t(64+i)*sc::kRawBytes,r[i].encoded,r[i].bytes,events[i]),"enqueue optional");
        wait_until([&]{return h.entered.load(std::memory_order_acquire);},"optional gate");
        req(svc.optional_state(100)==sc::OptionalState::Running,"first optional not running");
        for(int i=1;i<4;++i)req(svc.optional_state(100+i)==sc::OptionalState::Queued,"optional not queued");
        for(int i=0;i<64;++i)req(svc.issue_foreground(a.dest+size_t(i)*sc::kRawBytes,r[i%6].encoded,r[i%6].bytes),svc.error());
        req(svc.foreground_used()==64,"64 foreground leases absent");
        for(int i=0;i<4;++i)req(svc.optional_state(100+i)!=sc::OptionalState::Submitted,"held optional published");
        CK(cudaStreamSynchronize(a.fs)); // single proven owner-window join
        std::vector<uint8_t> back(sc::kRawBytes);
        for(int i=0;i<64;++i){CK(cudaMemcpy(back.data(),a.dest+size_t(i)*sc::kRawBytes,sc::kRawBytes,cudaMemcpyDeviceToHost));req(back==r[i%6].raw,"foreground record mismatch");}
        req(svc.reset_foreground_after_join(),"foreground reset after join");
        h.gate.store(true,std::memory_order_release);
        for(int i=0;i<4;++i){wait_until([&]{return svc.optional_state(100+i)==sc::OptionalState::Submitted||!svc.healthy();},"optional submission");req(svc.optional_state(100+i)==sc::OptionalState::Submitted,svc.error());
            wait_until([&]{auto q=cudaEventQuery(events[i]);req(q==cudaSuccess||q==cudaErrorNotReady,"event query error");return q==cudaSuccess;},"optional event");
            CK(cudaMemcpy(back.data(),a.dest+size_t(64+i)*sc::kRawBytes,sc::kRawBytes,cudaMemcpyDeviceToHost));req(back==r[i].raw,"optional record mismatch");req(svc.retire_optional(100+i),"optional retire");}
        adapter_check(a.fs,a.dest,r);
        req(svc.drain_cpu(),svc.error());CK(cudaStreamSynchronize(a.fs));CK(cudaStreamSynchronize(a.os));req(svc.release_after_gpu_join(),svc.error());
        for(auto ev:events)CK(cudaEventDestroy(ev));for(auto& rec:r)CK(cudaFreeHost(rec.encoded));a.cleanup();return;
    }
    if(mode=="deadline"||mode=="throw"||mode=="throw-unknown") {
        req(svc.enqueue_optional(100,a.dest,r[0].encoded,r[0].bytes,events[0]),"enqueue held");
        wait_until([&]{return h.entered.load(std::memory_order_acquire);},"gate entry");
        if(mode=="deadline"){req(!svc.drain_cpu(),"deadline failed to fire");req(!svc.healthy(),"deadline did not poison");req(!svc.release_after_gpu_join(),"unsafe service released");req(!svc.retire_optional(100),"unsafe job retired");}
        else {wait_until([&]{return svc.optional_state(100)==sc::OptionalState::Failed;},"thrown worker failure");req(!svc.healthy(),"throw did not poison");}
        std::fflush(nullptr);_Exit(0);
    }
    if(mode.rfind("fg-",0)==0){req(!svc.issue_foreground(a.dest,r[0].encoded,r[0].bytes),"injected foreground operation succeeded");req(!svc.healthy(),"foreground fault did not poison");req(!svc.reset_foreground_after_join(),"failed foreground lease released");}
    else {req(svc.enqueue_optional(100,a.dest,r[0].encoded,r[0].bytes,events[0]),"enqueue failure test");wait_until([&]{return svc.optional_state(100)==sc::OptionalState::Failed;},"optional failure");req(!svc.healthy(),"optional fault did not poison");req(svc.optional_state(100)!=sc::OptionalState::Submitted&&!svc.retire_optional(100),"failed optional published");}
    // Do not destroy callback context or borrowed buffers while a worker may run.
    std::fflush(nullptr);_Exit(0);
}
int main(int argc,char** argv) try {
    req(argc==3||argc==5,"usage: test DEVICE RECORD_DIR [--child MODE]");
    int device=std::atoi(argv[1]);if(argc==5){req(std::string(argv[3])=="--child","bad child invocation");child(device,argv[2],argv[4]);std::fflush(nullptr);_Exit(0);}
    std::vector<std::string> cases={"success","fg-header","fg-code","fg-progress","fg-query","fg-scale","fg-interleave","opt-header","opt-code","opt-scale","opt-interleave","opt-completion","deadline","throw","throw-unknown"};
    for(const auto& name:cases){pid_t pid=fork();req(pid>=0,"fork failed");if(pid==0){execl(argv[0],argv[0],argv[1],argv[2],"--child",name.c_str(),(char*)nullptr);_Exit(127);}
        auto end=Clock::now()+std::chrono::seconds(180);int status=0;for(;;){pid_t rc=waitpid(pid,&status,WNOHANG);req(rc>=0,"waitpid failed");if(rc==pid)break;if(Clock::now()>=end){kill(pid,SIGKILL);waitpid(pid,&status,0);throw std::runtime_error("child timed out: "+name);}std::this_thread::sleep_for(std::chrono::milliseconds(20));}
        req(WIFEXITED(status)&&WEXITSTATUS(status)==0,"child failed: "+name);std::printf("PASS %s\n",name.c_str());std::fflush(stdout);
    }return 0;
} catch(const std::exception& e){std::fprintf(stderr,"scale CUDA fixture: %s\n",e.what());return 1;}
