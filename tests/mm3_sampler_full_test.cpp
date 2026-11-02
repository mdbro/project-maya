// Differential check and optional CPU timing of full-vocabulary nucleus sampling.
// Build when the machine is idle, for example:
// g++ -O2 -std=c++17 -Isrc/mm3 -Iinclude tests/mm3_sampler_full_test.cpp -o /tmp/mm3_sampler_full_test
// /tmp/mm3_sampler_full_test [--bench]
#include "mm3_sample.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <vector>

using namespace mm3::sample;
constexpr int V = 200064;
using Dist = std::vector<std::pair<int,double>>;

// Frozen pre-change full-heap implementation: same float temperature scaling,
// comparator, sorted double reductions, penalties and nucleus/min-p order.
static Dist full_oracle(const std::vector<float>& row, const Req& r, const std::vector<int>& hist, bool heap) {
    std::vector<std::pair<int,float>> c(row.size());
    for(size_t i=0;i<row.size();++i)c[i]={(int)i,row[i]};
    std::unordered_map<int,int> counts;
    if (penalised(r)) {
        const size_t n=r.pen_n<0?hist.size():std::min(hist.size(),(size_t)r.pen_n);
        for(size_t i=hist.size()-n;i<hist.size();++i)++counts[hist[i]];
        for(const auto& [id,count]:counts) {
            if(id<0||(size_t)id>=c.size())continue;
            float& x=c[(size_t)id].second;
            if(r.pen_repeat>0 && r.pen_repeat!=1) x=x>0?x/r.pen_repeat:x*r.pen_repeat;
            x-=r.pen_freq*count+r.pen_present;
        }
    }
    std::vector<std::pair<float,int>> v; v.reserve(row.size());
    for(const auto& [id,x]:c)v.emplace_back(x/r.temperature,id);
    const auto cmp=[](const auto& a,const auto& b){return a.first>b.first||(a.first==b.first&&a.second<b.second);};
    if(heap)std::partial_sort(v.begin(),v.end(),v.end(),cmp);
    else std::sort(v.begin(),v.end(),cmp);
    std::vector<double> p(v.size());double sum=0;
    for(size_t i=0;i<v.size();++i)sum+=(p[i]=std::exp((double)(v[i].first-v[0].first)));
    size_t keep=v.size();double cum=0;
    for(size_t i=0;i<v.size();++i){
        cum+=p[i]/sum;
        if(r.min_p>0 && p[i]<r.min_p*p[0]){keep=std::max<size_t>(1,i);break;}
        if(r.top_p<1 && cum>=r.top_p){keep=i+1;break;}
    }
    double total=0;for(size_t i=0;i<keep;++i)total+=p[i];
    Dist out(keep);for(size_t i=0;i<keep;++i)out[i]={v[i].second,p[i]/total};return out;
}

static void require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
static std::vector<std::vector<float>> inputs(){
    std::mt19937 rng(413);std::normal_distribution<float> normal(0.f,3.f);
    std::vector<std::vector<float>> out(3,std::vector<float>(V));
    for(int i=0;i<V;++i){
        out[0][i]=normal(rng); // broad long tail
        out[1][i]=-8.f+(float)(i%17)*0.025f; // concentrated peaks + many ties
        out[2][i]=0.f; // flat distribution, nucleus spans most of vocabulary
    }
    for(int i=0;i<64;++i)out[1][i]=6.f-(float)(i%8)*0.5f;
    return out;
}
static void verify_case(const std::vector<float>& row,Req r,const std::vector<int>& hist,const char* name){
    const auto reference=full_oracle(row,r,hist,true);
    const auto actual=dist(row.data(),(int)row.size(),r,hist);
    if(actual!=reference){
        std::fprintf(stderr,"%s: full distribution differs (sizes %zu / %zu)\n",name,actual.size(),reference.size());
        throw std::runtime_error("full-vocabulary differential");
    }
    std::mt19937_64 ra(20261101), rb(20261101);
    for(int i=0;i<40;++i)require(draw(actual,ra)==draw(reference,rb),"seeded draw differs");
    for(int id : {0,1,7,63,64,10000,V-1})for(double rho : {0.01,0.2,1.0})
        require(fast_hsd_accepts(actual,id,rho)==fast_hsd_accepts(reference,id,rho),"Fast-HSD support differs");
}
static double millis(){return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();}
static void bench(const std::vector<std::vector<float>>& rows){
    const char* names[]={"long_tail","peaked_ties","flat"};
    Req r;r.temperature=1.f;r.top_p=.95f;r.top_k=0;r.min_p=0;
    for(size_t i=0;i<rows.size();++i){
        std::vector<double> old_times,sort_times,new_times;size_t support=0;
        for(int repeat=0;repeat<3;++repeat){
            // Rotate ordering; each timing includes complete distribution construction.
            auto old_call=[&]{const double t=millis();auto d=full_oracle(rows[i],r,{},true);old_times.push_back(millis()-t);support=d.size();};
            auto sort_call=[&]{const double t=millis();auto d=full_oracle(rows[i],r,{},false);sort_times.push_back(millis()-t);require(!support||d.size()==support,"sort support differs");};
            auto new_call=[&]{const double t=millis();auto d=dist(rows[i].data(),V,r,{});new_times.push_back(millis()-t);require(!support||d.size()==support,"adaptive support differs");};
            if(repeat==0){old_call();sort_call();new_call();}
            else if(repeat==1){sort_call();new_call();old_call();}
            else{new_call();old_call();sort_call();}
        }
        std::sort(old_times.begin(),old_times.end());std::sort(sort_times.begin(),sort_times.end());std::sort(new_times.begin(),new_times.end());
        std::printf("%s support=%zu old_heap_median_ms=%.3f full_sort_median_ms=%.3f adaptive_median_ms=%.3f\n",names[i],support,old_times[1],sort_times[1],new_times[1]);
    }
}
int main(int argc,char** argv){
 try{
    require(argc==1||(argc==2&&std::strcmp(argv[1],"--bench")==0),"usage: mm3_sampler_full_test [--bench]");
    auto rows=inputs();Req r;r.temperature=1.f;r.top_k=0;r.top_p=.95f;r.min_p=0;
    for(size_t i=0;i<rows.size();++i)verify_case(rows[i],r,{},i==0?"long tail":i==1?"peaked ties":"flat");
    // At and immediately above a flat-distribution half-mass cutoff, tie order
    // determines which IDs survive; the full-sort oracle preserves that order.
    r.top_p=.5f;verify_case(rows[2],r,{},"flat boundary");
    r.top_p=std::nextafter(.5f,1.f);verify_case(rows[2],r,{},"flat nextafter boundary");
    std::vector<float> boundary(V,-1000.f);boundary[0]=boundary[1]=0.f;
    r.top_p=.5f;verify_case(boundary,r,{},"two-token exact cutoff");
    std::vector<std::pair<float,int>> partial;partial.reserve(V);
    for(int i=0;i<V;++i)partial.emplace_back(boundary[(size_t)i],i);
    Dist out;require(!adaptive_nucleus(partial,.5f,out),"uncertain cutoff must fall back");
    r.top_p=std::nextafter(.5f,1.f);verify_case(boundary,r,{},"two-token nextafter cutoff");
    partial.clear();for(int i=0;i<V;++i)partial.emplace_back(rows[1][(size_t)i],i);
    require(adaptive_nucleus(partial,.95f,out),"concentrated distribution should use adaptive path");
    partial.clear();for(int i=0;i<V;++i)partial.emplace_back(0.f,i);
    require(!adaptive_nucleus(partial,.95f,out),"broad flat distribution should use full sort");
    std::mt19937 rng(27);std::normal_distribution<float> normal(0.f,4.f);
    for(int trial=0;trial<4;++trial){
        std::vector<float> random_row(V);for(float& v:random_row)v=normal(rng);
        r.top_p=trial%2?.95f:.8f;r.temperature=trial%2?1.f:.7f;
        verify_case(random_row,r,{},"random finite logits");
    }
    r.top_p=.95f;r.temperature=.7f;r.min_p=.02f;r.pen_n=6;r.pen_repeat=1.1f;r.pen_freq=.2f;r.pen_present=.3f;
    verify_case(rows[0],r,{0,1,1,60,63,64},"penalties and min-p");
    r.min_p=0;verify_case(rows[0],r,{0,1,1,60,63,64},"penalties adaptive");
    std::puts("full-vocabulary sampler: exact distributions, seeded draws and Fast-HSD support match old heap");
    if(argc==2)bench(rows);
    return 0;
 }catch(const std::exception& e){std::fprintf(stderr,"sampler test: %s\n",e.what());return 1;}
}
