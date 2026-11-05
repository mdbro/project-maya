#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <set>
#include <string>
#include <vector>

namespace mm3 {
struct ReplayLayer {int layer=0;std::vector<int> original,executed;};
struct ReplayBatch {int rows=0,position=0;bool generation=false;std::vector<ReplayLayer> layers;};
struct ReplayWorkload {
    int budget=8,stage_limit=4;double cpu_scale=1.3,margin_ms=0;
    bool diagnostic_primary_query=false;
    std::vector<ReplayBatch> batches;
};
// Simple bounded text protocol, generated from complete capture intervals.
// File parsing and validation happen before timing or mutating model state.
inline bool read_replay_workload(const std::string& path,const std::vector<int>& layers,
        int experts,int max_context,ReplayWorkload& out,std::string& err) {
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f || f.tellg()>std::streamoff(32*1024*1024)) {err="missing or oversized replay workload";return false;}
    f.seekg(0);std::string word;int version=0;
    if(!(f>>word>>version>>out.budget>>out.stage_limit>>out.cpu_scale>>out.margin_ms) ||
       word!="MM3_REPLAY" || (version!=1 && version!=2) || out.budget<0 || out.budget>32 ||
       out.stage_limit<0 || out.stage_limit>4 || !std::isfinite(out.cpu_scale) ||
       out.cpu_scale<.25 || out.cpu_scale>4 || !std::isfinite(out.margin_ms) || out.margin_ms<0 || out.margin_ms>1000) {
        err="invalid replay header";return false;
    }
    if(version==2) {
        int query=0;
        if(!(f>>query) || (query!=0 && query!=1)) {err="invalid diagnostic primary query";return false;}
        out.diagnostic_primary_query=query;
    }
    bool ended=false,generated=false;int initialized=0;
    while(f>>word) {
        if(word=="END") {ended=true;break;}
        ReplayBatch b;int generation=0;
        if(word!="F" || !(f>>b.rows>>b.position>>generation) || b.rows<1 || b.rows>16 ||
           b.position<0 || b.position+b.rows>max_context || (generation!=0 && generation!=1) || out.batches.size()>=256) {
            err="invalid or oversized replay target batch";return false;
        }
        b.generation=generation;generated|=b.generation;
        if(b.position>initialized) {err="replay omits required KV prefix work";return false;}
        initialized=std::max(initialized,b.position+b.rows);
        for(int li:layers) {
            ReplayLayer l;
            if(!(f>>word>>l.layer) || word!="L" || l.layer!=li) {err="incomplete ordered replay layers";return false;}
            for(auto* v:{&l.original,&l.executed}) {
                v->resize(size_t(b.rows)*4);
                for(int& e:*v)if(!(f>>e) || e<0 || e>=experts) {err="invalid replay expert";return false;}
                for(int i=0;i<b.rows;++i)if(std::set<int>(v->begin()+4*i,v->begin()+4*i+4).size()!=4) {
                    err="replay requires four distinct experts per evaluated row";return false;
                }
            }
            b.layers.push_back(std::move(l));
        }
        out.batches.push_back(std::move(b));
    }
    if(!ended || !generated || out.batches.empty() || (f>>word)) {err="incomplete replay workload or trailing input";return false;}
    return true;
}
}
