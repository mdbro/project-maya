// CPU-only full-call screen of same-arithmetic NVFP4 execution variants.
// Default: execution0 (old rows loop),1 (paired GU),2 (unroll2),3 (both).
// --legacy-kernels explicitly restores the old precision comparison; it is not
// part of the scheduling screen. --prefetched is an optional fixed-share timing.
// Build: mm3_cpu_lane_bench (EXCLUDE_FROM_ALL). Root coordinates every run.
// Example: --threads 9 --records 24 --calls 40 --executions 0,1,2,3 --columns 1,2,3,4,8
// Optional: --expert-counts 1,2,4 --shuffled --prefetched --quiet-lock PATH
// --balanced: one bounded anomaly diagnostic, nine threads,24 records,4 balanced
// blocks of0A/1/2/0B; whole-record rotations, then hot-record arms. No exec3.
#include "mm3_native_cpu.hpp"
#include "mm3_native_cpu_fast.hpp"
#include "mm3_native_experts.cuh"
#include <unistd.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace fast = mm3::cpu_fast;
namespace nx = strata::mm3::native_experts;
using clk = std::chrono::steady_clock;
constexpr int H = 6144, F = 3072;
static std::string quiet_lock="/mnt/wsl/transfer/minimax-wsl/runs/plan-completion-20261110/locks/quiet.lock";
static void quiet() {
    if (!quiet_lock.empty() && access(quiet_lock.c_str(), F_OK)==0)
        throw std::runtime_error("quiet lock held: refusing CPU work");
}
static std::vector<int> numbers(const std::string& arg) {
    std::vector<int> values;std::stringstream ss(arg);std::string v;
    while(std::getline(ss,v,',')){size_t at=0;int n=std::stoi(v,&at);if(at!=v.size())throw std::runtime_error("invalid list");values.push_back(n);}
    if(values.empty())throw std::runtime_error("empty list");return values;
}
static double median(std::vector<double> v) {std::sort(v.begin(),v.end());return v.empty()?0:v[v.size()/2];}
struct Variant {int act;fast::Execution exec;std::string name;};
int main(int argc,char** argv) {
    try {
        std::vector<int> threads_list{9},cols_list{1,2,3,4,8},expert_counts{1},executions{0,1,2,3};
        int records=24,calls=40;bool legacy=false,prefetched=false,shuffled=false,balanced=false,calls_given=false;
        for(int a=1;a<argc;++a){
            std::string key=argv[a];auto value=[&](){if(++a>=argc)throw std::runtime_error("missing option value");return std::string(argv[a]);};
            if(key=="--threads")threads_list=numbers(value());
            else if(key=="--records")records=std::stoi(value());
            else if(key=="--calls"){calls=std::stoi(value());calls_given=true;}
            else if(key=="--columns")cols_list=numbers(value());
            else if(key=="--expert-counts")expert_counts=numbers(value());
            else if(key=="--executions")executions=numbers(value());
            else if(key=="--quiet-lock")quiet_lock=value();
            else if(key=="--legacy-kernels")legacy=true;
            else if(key=="--prefetched")prefetched=true;
            else if(key=="--shuffled")shuffled=true;
            else if(key=="--balanced")balanced=true;
            else throw std::runtime_error("unknown option "+key);
        }
        if(records<1||calls<1)throw std::runtime_error("records/calls must be positive");
        for(int t:threads_list)if(t<1||t>65)throw std::runtime_error("threads outside1..65");
        for(int c:cols_list)if(c<1||c>8)throw std::runtime_error("columns outside1..8");
        for(int e:expert_counts)if(e<1||e>8||e>records)throw std::runtime_error("experts outside1..8 or record pool");
        for(int e:executions)if(e<0||e>3)throw std::runtime_error("execution outside0..3");
        if(balanced){
            if(records!=24||threads_list!=std::vector<int>{9}||expert_counts!=std::vector<int>{1}||
                legacy||prefetched||shuffled||(calls_given&&calls!=24))
                throw std::runtime_error("balanced diagnostic requires nine threads,24 records,one expert,24 calls and no legacy/prefetched/shuffled flags");
            calls=24;
        }
        if(!fast::supported())throw std::runtime_error("no AVX-512 VNNI");
        quiet(); // before record initialization or worker creation
        std::vector<Variant> variants;
        if(legacy)variants={{0,fast::kExecutionBaseline,"shared"},{1,fast::kExecutionBaseline,"int8"},
                           {2,fast::kExecutionBaseline,"nvfp4"},{3,fast::kExecutionBaseline,"threshold"}};
        else for(int e:executions)variants.push_back({fast::kNvfp4,fast::Execution(e),"nv4-exec"+std::to_string(e)});
        const size_t payload=size_t(3)*H*F*9/16;
        const size_t blob=(sizeof(nx::Header)+payload+4095)&~size_t(4095);
        auto* pool=(uint8_t*)std::aligned_alloc(4096,blob*records);
        if(!pool)throw std::runtime_error("record allocation");
        std::mt19937 rng(3);
        for(int r=0;r<records;++r){
            quiet();
            uint8_t* b=pool+blob*r;nx::Header header{};
            for(int k=0;k<3;++k){header.weight_global[k]=50.f;header.input_global[k]=1.f;}
            std::memcpy(b,&header,sizeof header);
            auto* w=(uint32_t*)(b+sizeof header);
            for(size_t q=0;q<payload/4;++q)w[q]=rng();
            uint8_t* rec=b+sizeof header;
            for(size_t u=0;u<size_t(3)*H*F/256;++u)
                for(int k=0;k<16;++k)rec[u*144+128+k]=uint8_t(0x18+(rng()%24));
        }
        std::vector<float> x(8*H);std::normal_distribution<float> nd(0,1);
        for(auto& v:x)v=nd(rng);
        mm3::CpuLane lane(*std::max_element(threads_list.begin(),threads_list.end())-1,H,F,1.702f,7.f,64);
        const size_t wbytes=mm3::native_cpu_batch_workspace_bytes(H,F,64);
        void* ws=std::aligned_alloc(256,(wbytes+255)&~size_t(255));
        if(!ws)throw std::runtime_error("workspace allocation");
        std::printf("CPU full-call median ms, %d measured calls; RG16 bytes/expert=%zu pool_bytes=%zu; NVFP4 division quantizer unchanged\n",calls,payload,blob*records);
        std::puts("rotate changes records each call; hot reuses the same records (cache residency depends on total working set)");
        std::printf("%7s %5s %4s %-12s %10s %10s %11s\n","threads","cols","jobs","kernel","rotate","hot","prefetched");
        for(int threads:threads_list){lane.set_active(threads);
            for(int experts:expert_counts)for(int cols:cols_list){
                const int tokens=shuffled?8:cols;
                std::vector<std::vector<std::pair<int,float>>> users(experts);
                for(int e=0;e<experts;++e)for(int c=0;c<cols;++c)
                    users[e].push_back({shuffled?(3*c+e)%8:c,0.25f});
                std::vector<mm3::NativeCpuExpert> ex(experts);
                std::vector<float> out(size_t(tokens)*H),reference(out.size());
                // Same record choices across variants. Ex/job setup belongs to
                // the caller and is not allocated again inside the timed call.
                const auto choose=[&](int base){for(int e=0;e<experts;++e)ex[e]={pool+blob*((base+e)%records),0.25f};};
                choose(records/2);
                if(!mm3::native_cpu_run_multi_fast(lane,fast::kNvfp4,H,F,ex,users,x.data(),tokens,reference.data(),nullptr,ws,wbytes))
                    throw std::runtime_error("baseline exact gate refused");
                if(balanced){
                    // Williams order: each arm occurs once in each position,
                    // and every directed adjacent-arm pair occurs exactly once.
                    static constexpr int order[4][4]={{0,1,3,2},{1,2,0,3},{2,3,1,0},{3,0,2,1}};
                    static constexpr int exec[4]={0,1,2,0};
                    static const char* names[4]={"0A","1","2","0B"};
                    const auto run=[&](int arm){
                        if(!mm3::native_cpu_run_multi_fast(lane,fast::kNvfp4,H,F,ex,users,x.data(),tokens,out.data(),nullptr,
                            ws,wbytes,false,nullptr,fast::Execution(exec[arm])))throw std::runtime_error("balanced kernel refused");
                    };
                    // Same exact full-call gate and record for every arm,
                    // outside all timed blocks. It also warms every code path.
                    for(int arm=0;arm<4;++arm){
                        choose(records/2);run(arm);
                        if(std::memcmp(reference.data(),out.data(),out.size()*sizeof(float)))
                            throw std::runtime_error("balanced full-call exact gate failed for "+std::string(names[arm]));
                    }
                    double ms[4][2][4]{};
                    for(int block=0;block<4;++block)for(int state=0;state<2;++state){
                        for(int position=0;position<4;++position){
                            quiet();const int arm=order[block][position];
                            // Warm each arm with a full pool rotation, not
                            // immediate calls on the same record across arms.
                            // Every measured arm then traverses the same whole
                            // pool beyond LLC. Hot arms use three warmups.
                            if(state==0)for(int c=0;c<records;++c){choose(c);run(arm);}
                            else {choose(records/2);for(int c=0;c<3;++c)run(arm);}
                            std::vector<double> times;times.reserve(calls);
                            for(int c=0;c<calls;++c){
                                if(state==0)choose(c);
                                const auto start=clk::now();run(arm);
                                times.push_back(std::chrono::duration<double,std::milli>(clk::now()-start).count());
                            }
                            ms[block][state][arm]=median(times);
                        }
                    }
                    // Delay output until all blocks finish: no arm gets a
                    // different printing pause during the balanced sequence.
                    for(int block=0;block<4;++block)for(int state=0;state<2;++state)
                        for(int position=0;position<4;++position){const int arm=order[block][position];
                            std::printf("BAL cols=%d block=%d state=%s position=%d arm=%s exec=%d median_ms=%.7f\n",
                                cols,block+1,state?"hot":"rotate",position+1,names[arm],exec[arm],ms[block][state][arm]);
                        }
                    for(int state=0;state<2;++state){
                        double lo=1e9,hi=-1e9,aa=0,pair=0,unroll=0;
                        for(int b=0;b<4;++b){
                            const double base=(ms[b][state][0]+ms[b][state][3])*0.5;
                            const double spread=100*(ms[b][state][3]/ms[b][state][0]-1);
                            lo=std::min(lo,spread);hi=std::max(hi,spread);aa+=spread;
                            pair+=100*(ms[b][state][1]/base-1);unroll+=100*(ms[b][state][2]/base-1);
                        }
                        std::printf("BAL_SUM cols=%d state=%s blocks=4 AA_pct_mean=%.4f AA_pct_min=%.4f AA_pct_max=%.4f pair_pct=%.4f unroll_pct=%.4f\n",
                            cols,state?"hot":"rotate",aa/4,lo,hi,pair/4,unroll/4);
                    }
                    std::fflush(stdout);continue;
                }
                for(const Variant& variant:variants){
                    quiet();
                    const auto run=[&](const std::vector<uint8_t>* fixed){
                        const bool ok=variant.act==0
                            ? mm3::native_cpu_run_multi_batch(lane,H,F,ex,users,x.data(),tokens,out.data(),nullptr,ws,wbytes)
                            : mm3::native_cpu_run_multi_fast(lane,variant.act,H,F,ex,users,x.data(),tokens,out.data(),nullptr,ws,wbytes,false,fixed,variant.exec);
                        if(!ok)throw std::runtime_error("kernel refused");
                    };
                    choose(records/2);run(nullptr);
                    if(variant.act==fast::kNvfp4 && std::memcmp(reference.data(),out.data(),out.size()*sizeof(float)))
                        throw std::runtime_error("full-call exact gate failed for "+variant.name);
                    std::vector<double> dram,l3,pre;
                    for(int c=0;c<calls+3;++c){
                        choose(c);
                        const auto t0=clk::now();run(nullptr);
                        if(c>=3)dram.push_back(std::chrono::duration<double,std::milli>(clk::now()-t0).count());
                    }
                    choose(records/2);
                    for(int c=0;c<calls+3;++c){
                        const auto t0=clk::now();run(nullptr);
                        if(c>=3)l3.push_back(std::chrono::duration<double,std::milli>(clk::now()-t0).count());
                    }
                    if(prefetched&&variant.act){
                        const std::vector<uint8_t> fixed(experts,1);
                        for(int c=0;c<calls+3;++c){
                            choose(c);std::vector<const uint8_t*> blobs;for(auto e:ex)blobs.push_back(e.blob);
                            std::atomic<unsigned long long> read{0};const auto start=clk::now();
                            if(!mm3::native_cpu_l3_begin(lane,H,F,blobs,&read))throw std::runtime_error("prefetch refused");
                            while(read.load()<payload*experts&&clk::now()-start<std::chrono::milliseconds(20))std::this_thread::yield();
                            lane.stop_background();const auto t0=clk::now();run(&fixed);
                            if(c>=3)pre.push_back(std::chrono::duration<double,std::milli>(clk::now()-t0).count());
                        }
                    }
                    std::printf("%7d %5d %4d %-12s %10.4f %10.4f %11s\n",threads,cols,experts,variant.name.c_str(),median(dram),median(l3),
                        pre.empty()?"-":std::to_string(median(pre)).c_str());
                    std::fflush(stdout);
                }
            }
        }
        std::free(ws);std::free(pool);return 0;
    }catch(const std::exception& e){std::fprintf(stderr,"CPU benchmark: %s\n",e.what());return 1;}
}
