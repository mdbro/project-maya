#include "mm3_cooperative_policy.hpp"
#include <cstdint>

extern "C" {
int mm3_replay_policy_abi() {return 1;}
float mm3_replay_demand(float value,int64_t last,int64_t row,float credit) {
    return mm3::update_native_demand(value,last,row,credit);
}
float mm3_replay_credit(int rows) {return mm3::native_demand_credit(rows);}
double mm3_replay_budget(double credit,int64_t last,int64_t row,double capacity) {
    mm3::NativeAdmissionBudget b;b.credit=credit;b.last_row=last;b.advance(row,capacity);return b.credit;
}
int mm3_replay_stage(const int* uses,const int* owners,const int* eligible,int maximum,int rows,
        const double* prices,double scale,int* chosen,double* predicted) {
    mm3::BatchStagingCosts costs;costs.valid=true;
    for(int n=0;n<17;++n) {
        costs.cpu_extra_ms[n]=prices[n];costs.gpu_extra_ms[n]=prices[17+n];
        costs.helper_extra_ms[n]=prices[34+n];costs.gpu_base_ms[n]=prices[51+n];
    }
    costs.cpu_base_ms=prices[68];costs.helper_base_ms=prices[69];costs.h2d_ms=prices[70];costs.d2h_ms=prices[71];
    std::array<int,128> count{};std::vector<int> cpu,gpu,helper,candidates;
    for(int e=0;e<128;++e) {
        count[e]=uses[e];if(!uses[e])continue;
        if(owners[e]==1)cpu.push_back(e);else if(owners[e]==2)gpu.push_back(e);else if(owners[e]==3)helper.push_back(e);
        if(eligible[e])candidates.push_back(e);
    }
    auto p=mm3::plan_batch_staging(count,cpu,gpu,helper,candidates,maximum,rows,costs,scale);
    if(!p.valid)return -1;
    for(size_t i=0;i<p.selected.size();++i)chosen[i]=p.selected[i];
    predicted[0]=p.predicted_cpu_ms;predicted[1]=p.predicted_gpu_ms;predicted[2]=p.predicted_helper_ms;
    predicted[3]=p.critical_before_ms;predicted[4]=p.critical_after_ms;
    return int(p.selected.size());
}
void mm3_replay_choose(int helper,const int64_t* ik,const float* is,const int* slots,int ni,
        const int64_t* vk,const float* vs,int nv,const double* prices,int64_t* out,double* value) {
    std::vector<mm3::ResidencyCandidate> incoming,victims;
    for(int i=0;i<ni;++i)incoming.push_back({ik[i],is[i],slots[i]});
    for(int i=0;i<nv;++i)victims.push_back({vk[i],vs[i],-1});
    auto c=helper ? mm3::choose_incumbent_helper(incoming,victims) :
        mm3::choose_incumbent_primary(incoming,victims,prices[0],prices[1],prices[2],prices[3],prices[4]);
    out[0]=c.incoming;out[1]=c.victim;out[2]=c.ready_slot;*value=c.value_ms;
}
}
