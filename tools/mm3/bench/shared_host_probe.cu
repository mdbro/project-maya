// Phase F allocation/access gate, NOT an engine, mover or bandwidth benchmark.
// One process owns one private HOST allocation, shared by CPU/5090/3090/4080.
// The Python ladder launches each size/backend in a fresh process so an access
// failure cannot poison a later attempt's CUDA contexts. No model data is read.
// Required: --backend vmm|pinned --mib 64..4096. Root owns build/run after E.
#include "mm3_native_host.hpp"
#include <cuda.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {
using U64=unsigned long long;
using Clock=std::chrono::steady_clock;
constexpr size_t MiB=size_t(1)<<20,kChunk=8*MiB;
constexpr U64 kReserve=8ULL<<30;
struct Failure:std::runtime_error {
    int code;std::string stage;
    Failure(int c,std::string s,std::string detail):std::runtime_error(detail),code(c),stage(std::move(s)) {}
};
std::string quote(const std::string& s) {
    std::ostringstream out;out<<'"';for(unsigned char c:s) {
        if(c=='"'||c=='\\')out<<'\\'<<char(c);
        else if(c<32)out<<"\\u"<<std::hex<<std::setw(4)<<std::setfill('0')<<unsigned(c)<<std::dec;
        else out<<char(c);
    }out<<'"';return out.str();
}
template<class T> std::string number(T n) {return std::to_string(n);}
using Fields=std::vector<std::pair<std::string,std::string>>;
void emit(const char* kind,Fields fields={}) {
    std::cout<<"{\"kind\":"<<quote(kind);for(const auto& [k,v]:fields)std::cout<<','<<quote(k)<<':'<<v;
    std::cout<<"}\n"<<std::flush;
}
void require(bool ok,const std::string& what) {if(!ok)throw Failure(1,"correctness",what);}
void cuda_check(cudaError_t e,const char* stage,int code=1) {
    if(e!=cudaSuccess)throw Failure(code,stage,cudaGetErrorString(e));
}
void driver_check(CUresult e,const char* stage,int code=3) {
    if(e==CUDA_SUCCESS)return;const char* name=nullptr;cuGetErrorName(e,&name);
    throw Failure(code,stage,name?name:"CUDA driver error");
}
#define CUDA(x) cuda_check((x),#x)
double elapsed(Clock::time_point start) {return std::chrono::duration<double>(Clock::now()-start).count();}
mm3::NativeHostMemorySnapshot host_memory(const char* phase) {
    mm3::NativeHostMemorySnapshot s;
    try {s=mm3::native_host_reserve_snapshot();}
    catch(const std::exception& e) {throw Failure(4,"host_memory_snapshot",e.what());}
    emit("host_memory",{{"phase",quote(phase)},{"effective_physical",number(s.available_physical)},
        {"effective_commit",number(s.available_commit)},{"guest_available",number(s.guest_available_physical)},
        {"guest_free",number(s.guest_free)},{"guest_reclaimable",number(s.guest_reclaimable)},
        {"windows",s.windows_host?"true":"false"},{"windows_physical",number(s.windows_available_physical)},
        {"windows_commit",number(s.windows_available_commit)},{"rss",number(s.process_rss)},
        {"cgroup",quote(s.cgroup_path)}});return s;
}
void reserve_gate(size_t bytes) {
    const auto s=host_memory("before_allocation");
    if(bytes>s.available_physical||kReserve>s.available_physical-bytes||
       bytes>s.available_commit||kReserve>s.available_commit-bytes)
        throw Failure(4,"host_reserve","allocation would consume the retained 8 GiB physical/commit reserve");
}
std::string uuid_text(const cudaUUID_t& uuid) {
    std::ostringstream out;out<<"GPU-"<<std::hex<<std::setfill('0');
    for(int i=0;i<16;++i) {if(i==4||i==6||i==8||i==10)out<<'-';out<<std::setw(2)<<unsigned(uint8_t(uuid.bytes[i]));}return out.str();
}
int find_card(const char* match) {
    int count=0,found=-1;CUDA(cudaGetDeviceCount(&count));
    for(int i=0;i<count;++i) {cudaDeviceProp p{};CUDA(cudaGetDeviceProperties(&p,i));if(std::strstr(p.name,match)) {
        if(found>=0)throw Failure(2,"device_selection",std::string("ambiguous card ")+match);found=i;}}
    if(found<0)throw Failure(2,"device_selection",std::string("missing card ")+match);return found;
}
void attribute(int device,CUdevice_attribute attr,const char* name) {
    int value=0;const auto e=cuDeviceGetAttribute(&value,attr,device);
    emit("capability",{{"device",number(device)},{"name",quote(name)},{"status",number(int(e))},
                       {"value",e==CUDA_SUCCESS?number(value):"null"}});
}
struct Region {
    std::string backend;size_t bytes=0,capacity=0;U64* cpu=nullptr;
    std::array<U64*,3> gpu{};
    CUdeviceptr address=0;CUmemGenericAllocationHandle handle=0;bool mapped=false;
    void allocate(const std::string& kind,size_t requested,const std::array<int,3>& devices) {
        backend=kind;bytes=requested;capacity=requested;CUDA(cudaSetDevice(devices[0]));
        if(kind=="vmm") {
            CUmemAllocationProp prop{};prop.type=CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location={CU_MEM_LOCATION_TYPE_HOST,0};prop.requestedHandleTypes=CU_MEM_HANDLE_TYPE_NONE;
            size_t granularity=0;driver_check(cuMemGetAllocationGranularity(&granularity,&prop,CU_MEM_ALLOC_GRANULARITY_MINIMUM),"vmm_granularity");
            if(!granularity||requested>size_t(-1)-(granularity-1))throw Failure(3,"vmm_granularity","invalid granularity");
            capacity=(requested+granularity-1)/granularity*granularity;
            if(capacity!=requested)throw Failure(3,"vmm_granularity","requested bounded size is not an allocation-granularity multiple");
            reserve_gate(capacity);
            driver_check(cuMemAddressReserve(&address,capacity,granularity,0,0),"vmm_reserve");
            driver_check(cuMemCreate(&handle,capacity,&prop,0),"vmm_create");
            driver_check(cuMemMap(address,capacity,0,handle,0),"vmm_map");mapped=true;cpu=reinterpret_cast<U64*>(address);
            // Attempt each permission separately and report each result. A
            // missing advertised helper capability does not skip its real grant.
            bool complete=true;
            for(int lane=-1;lane<3;++lane) {
                CUmemAccessDesc access{};access.location=lane<0?prop.location:CUmemLocation{CU_MEM_LOCATION_TYPE_DEVICE,devices[size_t(lane)]};
                access.flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                const auto set=cuMemSetAccess(address,capacity,&access,1);U64 flags=0;
                const auto get=cuMemGetAccess(&flags,&access.location,address);
                emit("vmm_access",{{"lane",number(lane)},{"device",lane<0?"null":number(devices[size_t(lane)])},
                    {"set_status",number(int(set))},{"get_status",number(int(get))},{"flags",number(flags)}});
                complete&=set==CUDA_SUCCESS&&get==CUDA_SUCCESS&&flags==CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            }
            for(int lane=-1;lane<3;++lane) {
                const CUmemLocation location=lane<0?prop.location:CUmemLocation{CU_MEM_LOCATION_TYPE_DEVICE,devices[size_t(lane)]};
                U64 flags=0;const auto get=cuMemGetAccess(&flags,&location,address);
                emit("vmm_final_access",{{"lane",number(lane)},{"status",number(int(get))},{"flags",number(flags)}});
                complete&=get==CUDA_SUCCESS&&flags==CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
            }
            if(!complete)throw Failure(3,"vmm_access","CPU and every GPU did not receive read/write access");
            gpu.fill(cpu);emit("allocation",{{"backend",quote(kind)},{"requested_bytes",number(bytes)},
                {"capacity_bytes",number(capacity)},{"granularity",number(granularity)},{"cpu_address",number(address)}});
        } else {
            reserve_gate(capacity);
            cuda_check(cudaHostAlloc(&cpu,capacity,cudaHostAllocPortable|cudaHostAllocMapped),"pinned_allocate",3);
            unsigned flags=0;cuda_check(cudaHostGetFlags(&flags,cpu),"pinned_flags",3);
            if((flags&(cudaHostAllocPortable|cudaHostAllocMapped))!=(cudaHostAllocPortable|cudaHostAllocMapped))
                throw Failure(3,"pinned_flags","allocation lost portable/mapped flags");
            for(size_t i=0;i<3;++i) {CUDA(cudaSetDevice(devices[i]));
                cuda_check(cudaHostGetDevicePointer(&gpu[i],cpu,0),"pinned_device_pointer",3);
                emit("pinned_alias",{{"device",number(devices[i])},{"cpu_address",number(reinterpret_cast<uintptr_t>(cpu))},
                    {"gpu_address",number(reinterpret_cast<uintptr_t>(gpu[i]))},{"flags",number(flags)}});}
            emit("allocation",{{"backend",quote(kind)},{"requested_bytes",number(bytes)},
                {"capacity_bytes",number(capacity)},{"cpu_address",number(reinterpret_cast<uintptr_t>(cpu))}});
        }
    }
    // Only before any kernel, or after every context/stream has joined.
    void release() {
        if(backend=="vmm") {
            if(mapped){driver_check(cuMemUnmap(address,capacity),"vmm_unmap",1);mapped=false;}
            if(handle){driver_check(cuMemRelease(handle),"vmm_release",1);handle=0;}
            if(address){driver_check(cuMemAddressFree(address,capacity),"vmm_address_free",1);address=0;}
        } else if(cpu)CUDA(cudaFreeHost(cpu));
        cpu=nullptr;gpu.fill(nullptr);
    }
};
__host__ __device__ U64 pattern(U64 index,U64 epoch) {
    U64 x=index+0x9e3779b97f4a7c15ULL*(epoch+1);x=(x^(x>>30))*0xbf58476d1ce4e5b9ULL;
    x=(x^(x>>27))*0x94d049bb133111ebULL;return x^(x>>31);
}
struct Check {U64 mismatch_threads,first;};
__global__ void check_words(const U64* data,U64 count,U64 global_offset,U64 epoch,Check* result) {
    for(U64 i=U64(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=U64(gridDim.x)*blockDim.x)
        if(data[i]!=pattern(global_offset+i,epoch)) {atomicAdd(&result->mismatch_threads,1ULL);atomicMin(&result->first,global_offset+i);return;}
}
__global__ void write_words(U64* data,U64 count,U64 global_offset,U64 epoch) {
    for(U64 i=U64(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=U64(gridDim.x)*blockDim.x)data[i]=pattern(global_offset+i,epoch);
}
struct Device {
    int ordinal;cudaStream_t stream=nullptr;cudaEvent_t done=nullptr;U64* scratch=nullptr;Check* check=nullptr;
    explicit Device(int d):ordinal(d) {}
    void select() {CUDA(cudaSetDevice(ordinal));}
    void memory(const char* phase) {select();size_t free=0,total=0;CUDA(cudaMemGetInfo(&free,&total));
        emit("device_memory",{{"phase",quote(phase)},{"device",number(ordinal)},{"free",number(free)},{"total",number(total)}});}
    void init() {
        select();cudaDeviceProp p{};CUDA(cudaGetDeviceProperties(&p,ordinal));unsigned flags=0;CUDA(cudaGetDeviceFlags(&flags));
        emit("device",{{"ordinal",number(ordinal)},{"name",quote(p.name)},{"uuid",quote(uuid_text(p.uuid))},
            {"pci_bus",number(p.pciBusID)},{"pci_device",number(p.pciDeviceID)},
            {"can_map_host",number(p.canMapHostMemory)},{"uva",number(p.unifiedAddressing)},{"context_flags",number(flags)}});
        attribute(ordinal,CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,"vmm");
        attribute(ordinal,CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,"host_vmm");
        attribute(ordinal,CU_DEVICE_ATTRIBUTE_HOST_NATIVE_ATOMIC_SUPPORTED,"host_native_atomics");
        memory("before_scratch");CUDA(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
        CUDA(cudaEventCreateWithFlags(&done,cudaEventDisableTiming));CUDA(cudaMalloc(&scratch,kChunk));CUDA(cudaMalloc(&check,sizeof(Check)));
        // Load both device kernels before region work. DMA endpoints are tested
        // later against the actual shared region; setup is not timed bandwidth.
        write_words<<<128,256,0,stream>>>(scratch,kChunk/8,0,0);CUDA(cudaGetLastError());join();
        verify(scratch,kChunk,0,0,"scratch_prewarm");
    }
    void join() {
        CUDA(cudaEventRecord(done,stream));const auto until=Clock::now()+std::chrono::seconds(30);
        for(;;) {const auto e=cudaEventQuery(done);if(e==cudaSuccess)return;
            if(e!=cudaErrorNotReady)cuda_check(e,"probe_stream_query");
            if(Clock::now()>=until)throw Failure(1,"probe_stream_timeout","30 seconds without completion");
            std::this_thread::sleep_for(std::chrono::milliseconds(1));}
    }
    void reset_check() {CUDA(cudaMemsetAsync(check,0,sizeof(Check),stream));CUDA(cudaMemsetAsync(&check->first,255,sizeof(U64),stream));}
    void read_check(const char* phase,size_t bytes,U64 offset,U64 epoch,Clock::time_point began) {
        join();Check got{};CUDA(cudaMemcpy(&got,check,sizeof(got),cudaMemcpyDeviceToHost));
        emit("gpu_read",{{"phase",quote(phase)},{"device",number(ordinal)},{"bytes",number(bytes)},
            {"offset_words",number(offset)},{"epoch",number(epoch)},{"mismatch_threads",number(got.mismatch_threads)},
            {"first_bad_word",got.mismatch_threads?number(got.first):"null"},{"elapsed_s",number(elapsed(began))}});
        require(!got.mismatch_threads,"GPU payload mismatch");
    }
    void verify(const U64* data,size_t bytes,U64 global_offset,U64 epoch,const char* phase) {
        select();const auto began=Clock::now();reset_check();
        for(size_t off=0;off<bytes;off+=kChunk) {const auto part=std::min(kChunk,bytes-off);
            check_words<<<128,256,0,stream>>>(data+off/8,part/8,global_offset+off/8,epoch,check);CUDA(cudaGetLastError());}
        read_check(phase,bytes,global_offset,epoch,began);
    }
    void write(U64* data,size_t bytes,U64 global_offset,U64 epoch,const char* phase,bool wait=true) {
        select();const auto began=Clock::now();
        for(size_t off=0;off<bytes;off+=kChunk) {const auto part=std::min(kChunk,bytes-off);
            write_words<<<128,256,0,stream>>>(data+off/8,part/8,global_offset+off/8,epoch);CUDA(cudaGetLastError());}
        if(wait)join();emit("gpu_write",{{"phase",quote(phase)},{"device",number(ordinal)},
            {"bytes",number(bytes)},{"offset_words",number(global_offset)},{"epoch",number(epoch)},
            {"retired",wait?"true":"false"},{"elapsed_s",number(elapsed(began))}});
    }
    void destroy() {select();join();CUDA(cudaFree(check));CUDA(cudaFree(scratch));CUDA(cudaEventDestroy(done));CUDA(cudaStreamDestroy(stream));}
};
void cpu_write(U64* data,size_t bytes,U64 offset,U64 epoch) {
    const auto began=Clock::now();for(U64 i=0;i<bytes/8;++i)data[i]=pattern(offset+i,epoch);
    emit("cpu_write",{{"bytes",number(bytes)},{"epoch",number(epoch)},{"elapsed_s",number(elapsed(began))}});
}
void cpu_verify(const U64* data,size_t bytes,U64 offset,U64 epoch) {
    const auto began=Clock::now();for(U64 i=0;i<bytes/8;++i)if(data[i]!=pattern(offset+i,epoch)) {
        emit("cpu_mismatch",{{"word",number(offset+i)},{"expected",number(pattern(offset+i,epoch))},{"actual",number(data[i])}});
        throw Failure(1,"cpu_payload","CPU payload mismatch");}
    emit("cpu_read",{{"bytes",number(bytes)},{"offset_words",number(offset)},{"epoch",number(epoch)},
                     {"exact","true"},{"elapsed_s",number(elapsed(began))}});
}
void dma_windows(Region& region,std::array<Device,3>& devices,size_t writer,U64 epoch) {
    auto& d=devices[writer];const size_t part=std::min(kChunk,region.bytes);
    const std::array<size_t,3> offsets{0,((region.bytes-part)/2/8)*8,region.bytes-part};
    for(size_t off:offsets) {
        d.select();const auto began=Clock::now();
        CUDA(cudaMemcpyAsync(d.scratch,region.cpu+off/8,part,cudaMemcpyHostToDevice,d.stream));
        d.verify(d.scratch,part,off/8,epoch,"dma_h2d_payload");
        d.write(d.scratch,part,off/8,epoch+100,"dma_pattern");
        CUDA(cudaMemcpyAsync(region.cpu+off/8,d.scratch,part,cudaMemcpyDeviceToHost,d.stream));d.join();
        cpu_verify(region.cpu+off/8,part,off/8,epoch+100);
        const size_t reader=(writer+1)%3;devices[reader].verify(region.gpu[reader]+off/8,part,off/8,epoch+100,"other_gpu_after_d2h");
        d.write(d.scratch,part,off/8,epoch,"restore_dma_window");
        CUDA(cudaMemcpyAsync(region.cpu+off/8,d.scratch,part,cudaMemcpyDeviceToHost,d.stream));d.join();
        cpu_verify(region.cpu+off/8,part,off/8,epoch);
        emit("dma_window",{{"device",number(d.ordinal)},{"offset_bytes",number(off)},{"bytes",number(part)},
            {"source_is_canonical_region","true"},{"cpu_payload_copies","0"},{"elapsed_s",number(elapsed(began))}});
    }
}
} // namespace

int main(int argc,char** argv) {
    std::thread([]{std::this_thread::sleep_for(std::chrono::seconds(180));emit("result",{{"status",quote("timeout")}});std::_Exit(1);}).detach();
    Region* region=nullptr;bool gpu_work_started=false;std::string backend;size_t bytes=0;
    try {
        int mib=0;
        for(int i=1;i<argc;++i) {const std::string a=argv[i];
            if(a=="--backend"&&i+1<argc)backend=argv[++i];else if(a=="--mib"&&i+1<argc)mib=std::stoi(argv[++i]);
            else throw Failure(2,"arguments","usage: --backend vmm|pinned --mib 64..4096");}
        if((backend!="vmm"&&backend!="pinned")||mib<64||mib>4096)throw Failure(2,"arguments","explicit backend and bounded --mib 64..4096 required");
        bytes=size_t(mib)*MiB;driver_check(cuInit(0),"driver_init",1);
        const std::array<int,3> ids{find_card("5090"),find_card("3090"),find_card("4080")};
        std::array<Device,3> devices{Device(ids[0]),Device(ids[1]),Device(ids[2])};
        for(auto& d:devices)d.init();region=new Region;
        const auto began=Clock::now();region->allocate(backend,bytes,ids);
        emit("allocated",{{"elapsed_s",number(elapsed(began))},{"single_shared_region","true"}});host_memory("allocated");
        cpu_write(region->cpu,bytes,0,0);gpu_work_started=true;
        for(size_t i=0;i<3;++i)devices[i].verify(region->gpu[i],bytes,0,0,"CPU_to_all_GPUs");
        for(size_t i=0;i<3;++i) {
            if(i)devices[i].verify(region->gpu[i],bytes,0,i,"previous_GPU_to_next_GPU");
            devices[i].write(region->gpu[i],bytes,0,i+1,"full_shared_region");
            cpu_verify(region->cpu,bytes,0,i+1);dma_windows(*region,devices,i,i+1);
        }
        for(size_t i=0;i<3;++i)devices[i].verify(region->gpu[i],bytes,0,3,"final_GPU_to_all_GPUs");
        // Both helpers can write disjoint slices of this same allocation. No
        // cross-device atomic or overlapping-writer guarantee is claimed.
        const size_t split=(bytes/2/8)*8;
        devices[1].write(region->gpu[1],split,0,201,"helpers_disjoint_submissions",false);
        devices[2].write(region->gpu[2]+split/8,bytes-split,split/8,202,"helpers_disjoint_submissions",false);
        for(size_t i:{size_t(1),size_t(2)}){devices[i].select();devices[i].join();}
        cpu_verify(region->cpu,split,0,201);cpu_verify(region->cpu+split/8,bytes-split,split/8,202);
        devices[0].verify(region->gpu[0],split,0,201,"primary_after_both_helpers");
        devices[0].verify(region->gpu[0]+split/8,bytes-split,split/8,202,"primary_after_both_helpers");
        host_memory("payload_checks_complete");for(auto& d:devices)d.memory("region_live");
        for(auto& d:devices)d.destroy();CUDA(cudaSetDevice(ids[0]));region->release();delete region;region=nullptr;
        host_memory("released");
        emit("result",{{"status",quote("pass")},{"backend",quote(backend)},{"bytes",number(bytes)},
            {"cpu_and_three_gpus","true"},{"full_range_checked","true"},{"model_speed_qualified","false"}});return 0;
    } catch(const Failure& e) {
        // A setup refusal has no region users; explicitly release its partial
        // allocation. An access/retirement failure retains all storage to exit.
        if(region&&!gpu_work_started&&(e.code==3||e.code==4))try {region->release();}catch(const std::exception& cleanup) {
            emit("cleanup_failure",{{"detail",quote(cleanup.what())}});}
        emit("result",{{"status",quote(e.code==3?"mapping_or_allocation_refused":e.code==4?"reserve_refused":"fail")},
            {"backend",quote(backend)},{"bytes",number(bytes)},{"stage",quote(e.stage)},{"detail",quote(e.what())}});
        std::_Exit(e.code);
    } catch(const std::exception& e) {emit("result",{{"status",quote("fail")},{"detail",quote(e.what())}});std::_Exit(1);}
}
