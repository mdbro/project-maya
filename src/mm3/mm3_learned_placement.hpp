#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace mm3 {
// Fill the helper allocation first; then use the primary budget for the
// remaining unique records. No expert may occupy both banks.
inline bool profiled_native_capacities(size_t experts, size_t blob, size_t primary_budget,
                                       size_t helper_allocation, size_t& primary, size_t& helper) {
    if (!experts || !blob || !helper_allocation) return false;
    helper=std::min(helper_allocation,experts);
    primary=std::min(experts-helper,primary_budget/blob);
    return true;
}
// Pure logical selection. Ties preserve the inherited helper expert/layer and
// primary layer/expert order. Device numbers here are model device indices.
inline bool helper_first_rosters(const std::vector<uint32_t>& demand, int experts,
        const std::vector<int>& devices, const std::vector<uint8_t>& moe, int head,
        const std::vector<size_t>& primary_capacity, size_t helper_capacity, int layer_cap,
        std::vector<int>& primary, std::vector<int>& helper) {
    const size_t layers=devices.size();
    if(experts<=0 || moe.size()!=layers || demand.size()!=layers*size_t(experts) ||
       head<0 || size_t(head)>=primary_capacity.size() || layer_cap<0 || layer_cap>experts)
        return false;
    primary.clear();helper.clear();
    for(size_t li=0;li<layers;++li)
        if(moe[li] && (devices[li]<0 || size_t(devices[li])>=primary_capacity.size()))return false;
    for(int e=0;e<experts;++e)for(size_t li=0;li<layers;++li)
        if(moe[li] && devices[li]==head)helper.push_back(int(li)*experts+e);
    const auto hotter=[&](int a,int b){return demand[size_t(a)]>demand[size_t(b)];};
    std::stable_sort(helper.begin(),helper.end(),hotter);
    if(helper.size()<helper_capacity)return false;
    helper.resize(helper_capacity);
    std::vector<uint8_t> picked(demand.size(),0);for(int id:helper)picked[size_t(id)]=1;
    for(size_t d=0;d<primary_capacity.size();++d) {
        std::vector<int> candidates, per_layer(layers,0);
        for(size_t li=0;li<layers;++li)if(moe[li] && devices[li]==int(d))
            for(int e=0;e<experts;++e)if(!picked[li*size_t(experts)+size_t(e)])
                candidates.push_back(int(li)*experts+e);
        std::stable_sort(candidates.begin(),candidates.end(),hotter);
        size_t selected=0;
        for(int id:candidates)if(selected<primary_capacity[d] && per_layer[size_t(id/experts)]<layer_cap) {
            primary.push_back(id);++per_layer[size_t(id/experts)];++selected;
        }
        if(selected!=primary_capacity[d])return false;
    }
    return true;
}
}
