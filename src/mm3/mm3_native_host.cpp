// Selective HOST VMM and bounded WSL memory snapshot adapted from Infernix
// src/core/host_memory.cpp and host_memory_limits.h (Apache-2.0), copied from
// /home/mdwbr/Infernix-minimax; no Infernix runtime dependency.
#include "mm3_native_host.hpp"
#include <cuda.h>
#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <sys/utsname.h>
#include <sys/wait.h>
#include <poll.h>
#include <spawn.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>

namespace mm3 {
namespace {
// A "Key:   N kB" field of a /proc file, in bytes.
std::optional<std::uint64_t> proc_kib_field(const char* path, const std::string& key) {
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.rfind(key, 0) == 0) {
            std::istringstream fields(line.substr(key.size()));
            std::uint64_t kib = 0;
            if (fields >> kib) { return kib * 1024; }
        }
    }
    return std::nullopt;
}

// Resolve the process's unified cgroup against the cgroup2 mount. Memory
// ceilings can also be imposed by an ancestor, so admission checks all levels.
namespace fs = std::filesystem;
std::string unescape_mount(std::string_view text) {
    std::string out;
    for(std::size_t i=0;i<text.size();++i) {
        if(text[i]=='\\' && i+3<text.size() && text[i+1]>='0' && text[i+1]<='7' &&
           text[i+2]>='0' && text[i+2]<='7' && text[i+3]>='0' && text[i+3]<='7') {
            out.push_back(char((text[i+1]-'0')*64+(text[i+2]-'0')*8+(text[i+3]-'0')));i+=3;
        } else out.push_back(text[i]);
    }
    return out;
}
struct CgroupLocation { fs::path mount, group; };
std::optional<CgroupLocation> current_cgroup() {
    std::ifstream groups("/proc/self/cgroup");std::string line,group;
    while(std::getline(groups,line)) if(line.rfind("0::",0)==0){group=line.substr(3);break;}
    if(group.empty())return std::nullopt;
    fs::path absolute(group);if(!absolute.is_absolute())throw std::runtime_error("invalid unified cgroup path");
    for(const auto& part:absolute)if(part=="..")throw std::runtime_error("invalid unified cgroup path");
    std::ifstream mounts("/proc/self/mountinfo");
    while(std::getline(mounts,line)) {
        if(line.find(" - cgroup2 ")==std::string::npos)continue;
        std::istringstream fields(line);std::string id,parent,major,root,mount;
        if(!(fields>>id>>parent>>major>>root>>mount))throw std::runtime_error("invalid cgroup mountinfo");
        root=unescape_mount(root);mount=unescape_mount(mount);
        fs::path relative;
        if(root=="/")relative=absolute.relative_path();
        else if(group==root)relative.clear();
        else if(group.size()>root.size() && group.compare(0,root.size(),root)==0 && group[root.size()]=='/')
            relative=fs::path(group.substr(root.size()+1));
        else continue;
        fs::path mount_path=fs::path(mount).lexically_normal();
        return CgroupLocation{mount_path,(mount_path/relative).lexically_normal()};
    }
    throw std::runtime_error("cannot resolve cgroup2 mount for process");
}
std::uint64_t read_cgroup_number(const fs::path& path,const std::string& value) {
    std::uint64_t n=0;auto r=std::from_chars(value.data(),value.data()+value.size(),n);
    if(r.ec!=std::errc{} || r.ptr!=value.data()+value.size())throw std::runtime_error("invalid cgroup memory counter: "+path.string());
    return n;
}
void apply_cgroup_limit(NativeHostMemorySnapshot& out) {
    const auto where=current_cgroup();if(!where)return;out.cgroup_path=where->group.string();
    for(fs::path dir=where->group;;dir=dir.parent_path()) {
        const auto maximum=dir/"memory.max",current=dir/"memory.current";
        std::ifstream maxfile(maximum);std::string maxvalue;
        if(maxfile) {
            if(!(maxfile>>maxvalue))throw std::runtime_error("cannot read cgroup memory.max: "+maximum.string());
            if(maxvalue!="max") {
                auto limit=read_cgroup_number(maximum,maxvalue);std::ifstream currentfile(current);std::string value;
                if(!(currentfile>>value))throw std::runtime_error("cannot read cgroup memory.current: "+current.string());
                auto used=read_cgroup_number(current,value);out.available_physical=std::min(out.available_physical,limit>used?limit-used:0);
            }
        } else if(fs::exists(maximum))throw std::runtime_error("cannot open cgroup memory.max: "+maximum.string());
        if(dir==where->mount)break;
        if(dir.empty() || dir==dir.root_path())throw std::runtime_error("cgroup mount is not an ancestor");
    }
}

struct WindowsMemoryLimits { std::uint64_t physical_available, commit_limit, committed; };
WindowsMemoryLimits parse_windows_memory_limits(std::string_view text) {
    constexpr std::string_view version = "infernix-windows-memory-v1 ";
    if (!text.starts_with(version)) throw std::runtime_error("invalid Windows memory response version");
    text.remove_prefix(version.size());std::array<std::uint64_t,3> values{};
    for (auto& value : values) {
        auto begin=text.find_first_not_of(" \t\r\n");
        if(begin==std::string_view::npos)throw std::runtime_error("missing Windows memory counter");
        text.remove_prefix(begin);auto word=text.substr(0,text.find_first_of(" \t\r\n"));
        auto result=std::from_chars(word.data(),word.data()+word.size(),value);
        if(result.ec!=std::errc{} || result.ptr!=word.data()+word.size()) throw std::runtime_error("invalid Windows memory counter");
        text.remove_prefix(word.size());
    }
    if(text.find_first_not_of(" \t\r\n")!=std::string_view::npos || !values[1] || values[2]>values[1])
        throw std::runtime_error("inconsistent Windows memory counters");
    return {values[0],values[1],values[2]};
}
bool is_wsl_host() {
    utsname system{};
    if (::uname(&system) != 0) throw std::runtime_error("uname failed during host memory query");
    const std::string_view release(system.release);
    return release.find("microsoft") != std::string_view::npos ||
           release.find("Microsoft") != std::string_view::npos;
}

WindowsMemoryLimits windows_memory_limits() {
    // WSL interop executes the installed Windows provider. This runs only at reserve-sensitive
    // acquisition boundaries; it uses no model inputs, shell interpolation, user-supplied script
    // or decode-round work.
    // Query the native API directly; CIM can stall while these counters remain available.
    constexpr char script[] = R"ps($ErrorActionPreference='Stop';Add-Type -TypeDefinition 'using System;using System.Runtime.InteropServices;public static class InfernixMemory { [StructLayout(LayoutKind.Sequential)] public struct Info { public uint cb; public UIntPtr CommitTotal,CommitLimit,CommitPeak,PhysicalTotal,PhysicalAvailable,SystemCache,KernelTotal,KernelPaged,KernelNonpaged,PageSize; public uint HandleCount,ProcessCount,ThreadCount; } [DllImport("psapi.dll",SetLastError=true)] public static extern bool GetPerformanceInfo(ref Info value,uint size); }';$m=New-Object InfernixMemory+Info;$m.cb=[Runtime.InteropServices.Marshal]::SizeOf($m);if(-not [InfernixMemory]::GetPerformanceInfo([ref]$m,$m.cb)){throw 'GetPerformanceInfo failed'};$p=[decimal]$m.PageSize.ToUInt64();[Console]::WriteLine(('infernix-windows-memory-v1 {0} {1} {2}' -f ([uint64]([decimal]$m.PhysicalAvailable.ToUInt64()*$p)),([uint64]([decimal]$m.CommitLimit.ToUInt64()*$p)),([uint64]([decimal]$m.CommitTotal.ToUInt64()*$p)))))ps";
    int pipefd[2];
    if (::pipe(pipefd) != 0) throw std::runtime_error("cannot create Windows memory query pipe");
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attributes;
    bool have_actions = false, have_attributes = false;
    pid_t child = -1;
    const auto cleanup = [&] {
        if (pipefd[0] >= 0) ::close(pipefd[0]);
        if (pipefd[1] >= 0) ::close(pipefd[1]);
        if (have_actions) ::posix_spawn_file_actions_destroy(&actions);
        if (have_attributes) ::posix_spawnattr_destroy(&attributes);
        if (child > 0) {
            (void)::kill(-child, SIGKILL);
            while (::waitpid(child, nullptr, 0) < 0 && errno == EINTR) {}
        }
    };
    const auto checked = [](int error) {
        if (error) throw std::runtime_error("cannot start Windows memory query: " +
                                            std::string(std::strerror(error)));
    };
    try {
        if (::fcntl(pipefd[0], F_SETFD, FD_CLOEXEC) < 0 ||
            ::fcntl(pipefd[1], F_SETFD, FD_CLOEXEC) < 0)
            throw std::runtime_error("cannot protect Windows memory query descriptors");
        checked(::posix_spawn_file_actions_init(&actions)); have_actions = true;
        checked(::posix_spawnattr_init(&attributes)); have_attributes = true;
        checked(::posix_spawnattr_setflags(&attributes, POSIX_SPAWN_SETPGROUP));
        checked(::posix_spawnattr_setpgroup(&attributes, 0));
        checked(::posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDOUT_FILENO));
        checked(::posix_spawn_file_actions_adddup2(&actions, pipefd[1], STDERR_FILENO));
        checked(::posix_spawn_file_actions_addclose(&actions, pipefd[0]));
        checked(::posix_spawn_file_actions_addclose(&actions, pipefd[1]));
        char* argv[] = {const_cast<char*>("powershell.exe"), const_cast<char*>("-NoProfile"),
                        const_cast<char*>("-NonInteractive"), const_cast<char*>("-Command"),
                        const_cast<char*>(script), nullptr};
        checked(::posix_spawnp(&child, argv[0], &actions, &attributes, argv, ::environ));
        ::close(pipefd[1]); pipefd[1] = -1;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
        std::string output;
        for (;;) {
            const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(
                deadline - std::chrono::steady_clock::now()).count();
            if (remaining <= 0) throw std::runtime_error("Windows memory query timed out");
            pollfd descriptor{pipefd[0], POLLIN, 0};
            const int ready = ::poll(&descriptor, 1, static_cast<int>(remaining));
            if (ready < 0 && errno == EINTR) continue;
            if (ready <= 0) throw std::runtime_error("Windows memory query failed or timed out");
            char buffer[256];
            const auto count = ::read(pipefd[0], buffer, sizeof(buffer));
            if (count < 0 && errno == EINTR) continue;
            if (count < 0) throw std::runtime_error("cannot read Windows memory query");
            if (!count) break;
            output.append(buffer, static_cast<std::size_t>(count));
            if (output.size() > 1024) throw std::runtime_error("oversized Windows memory response");
        }
        int status = 0;
        for (;;) {
            const auto ended = ::waitpid(child, &status, WNOHANG);
            if (ended == child) { child = -1; break; }
            if (ended < 0 && errno != EINTR) throw std::runtime_error("cannot reap Windows memory query");
            if (std::chrono::steady_clock::now() >= deadline)
                throw std::runtime_error("Windows memory query did not retire");
            (void)::poll(nullptr, 0, 1);
        }
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            throw std::runtime_error("Windows memory provider failed: " + output);
        const auto result = parse_windows_memory_limits(output);
        cleanup();
        return result;
    } catch (...) {
        cleanup();
        throw;
    }
}
void check(CUresult result, const char* op) {
    if(result==CUDA_SUCCESS)return;const char* name=nullptr;(void)cuGetErrorName(result,&name);
    throw std::runtime_error(std::string(op)+": "+(name?name:"unknown CUDA driver error"));
}
CUmemAllocationProp properties(int ordinal) {
    static std::once_flag init;std::call_once(init,[]{check(cuInit(0),"cuInit");});
    CUdevice dev;check(cuDeviceGet(&dev,ordinal),"cuDeviceGet");int supported=0;
    check(cuDeviceGetAttribute(&supported,CU_DEVICE_ATTRIBUTE_HOST_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED,dev),"host VMM capability");
    if(!supported)throw std::runtime_error("primary device does not support host-backed VMM");
    CUmemAllocationProp p{};p.type=CU_MEM_ALLOCATION_TYPE_PINNED;p.location={CU_MEM_LOCATION_TYPE_HOST,0};return p;
}
std::size_t granularity(const CUmemAllocationProp& p) {
    std::size_t gran=0;check(cuMemGetAllocationGranularity(&gran,&p,CU_MEM_ALLOC_GRANULARITY_MINIMUM),"host VMM granularity");
    if(!gran)throw std::runtime_error("zero host VMM granularity");return gran;
}
std::size_t rounded(std::size_t bytes,std::size_t gran) {
    if(bytes>std::numeric_limits<std::size_t>::max()-(gran-1))throw std::overflow_error("host VMM size overflow");
    return (bytes+gran-1)/gran*gran;
}
void cleanup_report(CUresult r,const char* operation) noexcept {
    if(r!=CUDA_SUCCESS)std::fprintf(stderr,"NativeHostBank cleanup: %s returned %d\n",operation,int(r));
}
} // namespace

NativeHostMemorySnapshot native_host_reserve_snapshot() {
    NativeHostMemorySnapshot out;
    out.total_physical=proc_kib_field("/proc/meminfo","MemTotal:").value_or(0);
    const auto physical=proc_kib_field("/proc/meminfo","MemAvailable:");
    if(!physical)throw std::runtime_error("/proc/meminfo has no MemAvailable");
    out.guest_available_physical=*physical;
    out.guest_cached=proc_kib_field("/proc/meminfo","Cached:").value_or(0);
    out.guest_sreclaimable=proc_kib_field("/proc/meminfo","SReclaimable:").value_or(0);
    out.available_physical=*physical;
    apply_cgroup_limit(out);
    const auto swap=proc_kib_field("/proc/meminfo","SwapFree:").value_or(0);
    out.available_commit=swap>std::numeric_limits<std::uint64_t>::max()-out.available_physical
        ?std::numeric_limits<std::uint64_t>::max():out.available_physical+swap;
    out.process_rss=proc_kib_field("/proc/self/status","VmRSS:").value_or(0);
    if(is_wsl_host()) {
        const auto host=windows_memory_limits();
        out.windows_host=true;out.windows_available_physical=host.physical_available;
        out.windows_available_commit=host.commit_limit-host.committed;
        out.available_physical=std::min(out.available_physical,host.physical_available);
        out.available_commit=std::min(out.available_commit,host.commit_limit-host.committed);
    }
    return out;
}
struct NativeHostBank::Impl {
    struct Chunk { CUmemGenericAllocationHandle handle=0; std::size_t offset=0,bytes=0; bool mapped=false; };
    CUdeviceptr address=0;std::size_t size=0,capacity=0,max_allocation=0;
    std::vector<Chunk> chunks;
    CUmemAccessDesc access[2]{};   // [0] the CPU, [1] the primary card
    bool granted=true;             // false between a deferred construction and grant_device()
    ~Impl() {
        // A partially constructed bank is never exposed. Undo mappings before
        // releasing handles and the single enclosing virtual address reservation.
        for(auto it=chunks.rbegin();it!=chunks.rend();++it) {
            if(it->mapped)cleanup_report(cuMemUnmap(address+it->offset,it->bytes),"cuMemUnmap");
            if(it->handle)cleanup_report(cuMemRelease(it->handle),"cuMemRelease");
        }
        if(address)cleanup_report(cuMemAddressFree(address,capacity),"cuMemAddressFree");
    }
};
std::size_t NativeHostBank::allocation_bytes(std::size_t bytes,int primary) {
    return bytes?rounded(bytes,granularity(properties(primary))):0;
}
NativeHostBank::NativeHostBank(std::size_t bytes,int primary,std::uint64_t phys_reserve,std::uint64_t commit_reserve,bool defer_device_access)
    : impl_(std::make_unique<Impl>()) {
    impl_->size=bytes;if(!bytes)return;
    const auto p=properties(primary);const auto gran=granularity(p);impl_->capacity=rounded(bytes,gran);
    const auto snapshot=native_host_reserve_snapshot();const auto cap=std::uint64_t(impl_->capacity);
    if(cap>snapshot.available_physical || phys_reserve>snapshot.available_physical-cap ||
       cap>snapshot.available_commit || commit_reserve>snapshot.available_commit-cap)
        throw std::runtime_error("host VMM bank exceeds physical/commit headroom (capacity="+std::to_string(cap)+
            ", available physical="+std::to_string(snapshot.available_physical)+", available commit="+
            std::to_string(snapshot.available_commit)+", physical reserve="+std::to_string(phys_reserve)+
            ", commit reserve="+std::to_string(commit_reserve)+")");
    constexpr std::size_t requested_chunk=std::size_t{4}<<30;
    const auto chunk_bytes=(requested_chunk/gran)*gran;
    if(!chunk_bytes)throw std::runtime_error("host VMM granularity exceeds fixed 4 GiB chunk");
    const auto count=impl_->capacity/chunk_bytes+(impl_->capacity%chunk_bytes!=0);
    impl_->chunks.resize(count);
    check(cuMemAddressReserve(&impl_->address,impl_->capacity,gran,0,0),"reserve selective host address");
    CUdevice dev;check(cuDeviceGet(&dev,primary),"cuDeviceGet");auto& access=impl_->access;
    access[0].location=p.location;access[0].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access[1].location={CU_MEM_LOCATION_TYPE_DEVICE,dev};access[1].flags=CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    impl_->granted=!defer_device_access;
    std::size_t offset=0;
    for(std::size_t i=0;i<count;++i) {
        auto& chunk=impl_->chunks[i];chunk.offset=offset;chunk.bytes=std::min(chunk_bytes,impl_->capacity-offset);
        const auto label=std::string("host VMM chunk ")+std::to_string(i)+"/"+std::to_string(count);
        check(cuMemCreate(&chunk.handle,chunk.bytes,&p,0),(label+" create").c_str());
        check(cuMemMap(impl_->address+offset,chunk.bytes,0,chunk.handle,0),(label+" map").c_str());
        chunk.mapped=true;
        check(cuMemSetAccess(impl_->address+offset,chunk.bytes,access,defer_device_access?1:2),
              (label+(defer_device_access?" CPU access":" CPU/primary access")).c_str());
        impl_->max_allocation=std::max(impl_->max_allocation,chunk.bytes);offset+=chunk.bytes;
    }
}
void NativeHostBank::grant_device() {
    if(!impl_->size || impl_->granted)return;
    for(std::size_t i=0;i<impl_->chunks.size();++i) {
        const auto& chunk=impl_->chunks[i];
        const auto label=std::string("host VMM chunk ")+std::to_string(i)+"/"+std::to_string(impl_->chunks.size());
        check(cuMemSetAccess(impl_->address+chunk.offset,chunk.bytes,impl_->access,2),(label+" primary access after the fill").c_str());
    }
    impl_->granted=true;
}
bool NativeHostBank::device_granted() const noexcept {return impl_->granted;}
NativeHostBank::~NativeHostBank()=default;
std::byte* NativeHostBank::data() const noexcept {return reinterpret_cast<std::byte*>(impl_->address);}
std::size_t NativeHostBank::size() const noexcept {return impl_->size;}
std::size_t NativeHostBank::capacity() const noexcept {return impl_->capacity;}
std::size_t NativeHostBank::allocation_count() const noexcept {return impl_->chunks.size();}
std::size_t NativeHostBank::max_allocation_bytes() const noexcept {return impl_->max_allocation;}
} // namespace mm3

// Standalone diagnostics, no CUDA initialization or allocation:
// c++ -std=c++20 -DMM3_NATIVE_HOST_DIAGNOSTIC_MAIN -I<cuda>/include
//     mm3_native_host.cpp -L<cuda>/lib64/stubs -lcuda -o mm3-host-memory
#ifdef MM3_NATIVE_HOST_DIAGNOSTIC_MAIN
int main() {
    try {
        const auto s=mm3::native_host_reserve_snapshot();
        std::printf("guest_mem_available=%llu guest_cached=%llu guest_sreclaimable=%llu "
                    "process_rss=%llu cgroup=%s windows_host=%d windows_physical_available=%llu "
                    "windows_commit_available=%llu effective_physical=%llu effective_commit=%llu\n",
            (unsigned long long)s.guest_available_physical,(unsigned long long)s.guest_cached,
            (unsigned long long)s.guest_sreclaimable,(unsigned long long)s.process_rss,
            s.cgroup_path.c_str(),int(s.windows_host),(unsigned long long)s.windows_available_physical,
            (unsigned long long)s.windows_available_commit,(unsigned long long)s.available_physical,
            (unsigned long long)s.available_commit);
        return 0;
    } catch(const std::exception& e) {std::fprintf(stderr,"host memory snapshot: %s\n",e.what());return 1;}
}
#endif
