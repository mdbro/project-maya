// Independent oracle for the literal production next-layer prediction caller.
// Run on a GPU only after coordinating with the resident model owner.
#include "../src/mm3/mm3_kernels.cuh"
#include "../src/mm3/mm3_resident_router.cuh"
#include "../src/mm3/mm3_r2_lifecycle.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
namespace k=mm3::k;namespace resident=mm3::resident;namespace r2=mm3::r2;
constexpr int kWin=16;
#define CK(x) do {const auto c_=(x);if(c_!=cudaSuccess){err=cudaGetErrorString(c_);return false;}}while(0)
#define CUDA(x) do {const auto c_=(x);if(c_!=cudaSuccess){std::fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(c_));std::abort();}}while(0)
struct Layer {
    bool moe=true,hf=false;int dev=0;size_t blob=8;float* gate=nullptr;float* bias=nullptr;
    std::vector<int> res_slot=std::vector<int>(128,-1);
    std::vector<const uint8_t*> host_blob=std::vector<const uint8_t*>(128,nullptr);
};
struct Dev {
    struct Promo {int slot;const uint8_t* src;size_t bytes;};
    int id=0,n_slots=8,n_pending=0;size_t slot_bytes=8;cudaStream_t s=nullptr,s2=nullptr,cs=nullptr;
    std::vector<long long> slot_owner=std::vector<long long>(8,-1),slot_pend=std::vector<long long>(8,-1),slot_used=std::vector<long long>(8,-1);
    std::vector<uint8_t> slot_iss=std::vector<uint8_t>(8),slot_foreground=std::vector<uint8_t>(8),slot_prediction;
    std::vector<Promo> promo_q;std::array<cudaEvent_t,8> slot_ev{};
    float* prediction_logits=nullptr;int* prediction_ids=nullptr;float* prediction_weights=nullptr;
    uint8_t* prediction_host=nullptr;cudaEvent_t prediction_uploads=nullptr,fev=nullptr,jev=nullptr;float* xn=nullptr;uint8_t* stage=nullptr;
};
struct Model {
    struct {int n_exp=128,k_exp=4,dim=6144;} g;
    struct {int seqs=1,tp=0,chunk=16;bool fast_decode=true,promote=true,routing=true,routing_prefetch=true;} o;
    struct {long prefetch_predictions=0,prefetch_rows=0,prefetch_admitted=0,prefetch_cancelled=0,prefetch_useful=0,prefetch_unused=0,prefetch_late=0;
        uint64_t prefetch_issued_bytes=0,prefetch_useful_bytes=0,prefetch_unused_bytes=0,prefetch_late_bytes=0;double prefetch_policy_ms=0;} st;
    bool native=true,graph_on=false,primary_adaptive=false,generation_active=true,vis=false,placement_failed=false;
    std::atomic<bool> retirement_uncertain{false};
    int native_pool=1,helper=1;std::string helper_dispatch_current="worker",promotion_error;
    long long fwd=10;static constexpr int kMaxPending=4;
    std::vector<int> rseq;std::vector<Layer> L=std::vector<Layer>(2);std::vector<Dev> D=std::vector<Dev>(1);std::vector<void*> allocated;
    Model(){CUDA(cudaStreamCreate(&D[0].s));CUDA(cudaStreamCreate(&D[0].cs));}
    ~Model(){auto& d=D[0];CUDA(cudaStreamSynchronize(d.s));for(auto p:allocated)CUDA(cudaFree(p));if(d.prediction_host)CUDA(cudaFreeHost(d.prediction_host));if(d.prediction_uploads)CUDA(cudaEventDestroy(d.prediction_uploads));CUDA(cudaStreamDestroy(d.s));CUDA(cudaStreamDestroy(d.cs));}
    template<class T> bool dalloc(int,T*& p,size_t n,std::string& err){auto c=cudaMalloc(&p,n*sizeof(T));if(c!=cudaSuccess){err=cudaGetErrorString(c);return false;}allocated.push_back(p);return true;}
    bool staging_reserved(const Dev&,int)const{return false;}
    uint8_t* staging_address(Dev& d,int s){return d.stage+s*d.slot_bytes;}
    bool helper_owns(const Layer&,int)const{return false;}
    bool helper_live_leased(long long)const{return false;}
    bool adaptive_identity_pending(const Dev&,long long)const{return false;}
    int adaptive_pending(const Dev&)const{return 0;}
    bool publish_slots(Dev&,std::string&){return true;}
    cudaStream_t fork(Dev& d){return d.s2;}
    void join(Dev&){}
    static double now(){return 0;}
#include "../src/mm3/mm3_native_prefetch.inc"
};
int main(){
    Model m;auto& d=m.D[0];std::string err;assert(m.prefetch_prepare(err));
    constexpr int K=6144,N=128,T=16;
    std::vector<float> gate(size_t(N)*K),x(size_t(T)*K),bias(N);
    for(int e=0;e<N;++e){bias[e]=float(e)*0.0007f;for(int k=0;k<K;++k)gate[size_t(e)*K+k]=float(((e*13+k*7)%31)-15)*0.0004f;}
    for(int t=0;t<T;++t)for(int k=0;k<K;++k)x[size_t(t)*K+k]=float(((t*17+k*3)%41)-20)*0.04f;
    assert(m.dalloc(0,m.L[1].gate,gate.size(),err) && m.dalloc(0,m.L[1].bias,bias.size(),err) && m.dalloc(0,d.xn,x.size(),err));
    CUDA(cudaMemcpy(m.L[1].gate,gate.data(),gate.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(m.L[1].bias,bias.data(),bias.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(d.xn,x.data(),x.size()*sizeof(float),cudaMemcpyHostToDevice));
    for(int width:{1,8,16}) {
        assert(m.prefetch_predict(m.L[0],d,width,d.s,err));
        assert(m.prefetch_readback(m.L[0],d,width,err));CUDA(cudaStreamSynchronize(d.s));
        const auto* ids=reinterpret_cast<const int*>(d.prediction_host);
        const auto* weights=reinterpret_cast<const float*>(d.prediction_host+kWin*4*sizeof(int));
        for(int t=0;t<width;++t) {
            std::array<double,N> probability{},score{};std::array<int,N> order{};
            for(int e=0;e<N;++e) {
                double dot=0;for(int i=0;i<K;++i)dot+=double(gate[size_t(e)*K+i])*x[size_t(t)*K+i];
                probability[e]=1.0/(1.0+std::exp(-dot));score[e]=probability[e]+bias[e];order[e]=e;
            }
            std::stable_sort(order.begin(),order.end(),[&](int a,int b){return score[a]>score[b] || (score[a]==score[b] && a<b);});
            double total=0;for(int j=0;j<4;++j)total+=probability[order[j]];
            for(int j=0;j<4;++j){assert(ids[t*4+j]==order[j]);assert(std::abs(weights[t*4+j]-probability[order[j]]/total)<1e-5);}
        }
    }
    assert(m.st.prefetch_predictions==3 && m.st.prefetch_rows==25);
    return 0;
}
