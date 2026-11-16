#pragma once
// Host scheduling only. A serial selection/locate traversal is frozen before
// bookkeeping consumes it: locate can change CPU/cache and third-card eligibility.
#include <array>
#include <cstdint>

namespace mm3::upload_first {
constexpr int kRows = 16, kPicks = 8, kEntries = kRows * kPicks;
enum class Service : std::uint8_t { none, helper, third, cpu, primary };
struct Plan {
    std::array<Service,kEntries> service;
    std::array<unsigned long long,kEntries> pointer;
    std::array<bool,kEntries> copied;
    bool complete;
};

// Locate must perform the ordinary primary pick's actual cache lookup and
// ownership transition, and report whether it issued a fresh copy. No routing
// predicates are called again when the frozen plan is consumed.
template<class Helper,class Third,class Cpu,class Locate>
bool prepare(int rows,int picks,const int* experts,Plan& out,
             Helper helper,Third third,Cpu cpu,Locate locate) {
    out = Plan{};
    if(rows<1 || rows>kRows || picks<1 || picks>kPicks || !experts)return false;
    for(int t=0;t<rows;++t) {
        int third_columns=0;
        for(int i=0;i<picks;++i) {
            const int at=t*kPicks+i,expert=experts[t*picks+i];
            auto& service=out.service[at];
            if(helper(expert))service=Service::helper;
            else if(third(expert) && third_columns<4) {service=Service::third;++third_columns;}
            else if(cpu(expert))service=Service::cpu;
            else {
                service=Service::primary;
                if(!locate(expert,out.pointer[at],out.copied[at]))return false;
            }
        }
    }
    out.complete=true;
    return true;
}
} // namespace mm3::upload_first
