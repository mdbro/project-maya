// Focused device test; link with src/mm3/mm3_resident_router.cu and CUDA runtime.
#include "../src/mm3/mm3_resident_router.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <cstdlib>
#include <vector>

#define CUDA(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); std::exit(1); } } while (0)
#define REQUIRE(c) do { if (!(c)) { std::fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #c); std::exit(1); } } while (0)

int main() {
    using namespace mm3::resident;
    constexpr int T = 3;
    std::vector<float> logits(T * 128, -3.0f), bias(128, -20.0f);
    // All selection scores are negative: ranking and ratio must use their
    // respective corrected and unbiased domains, without dividing by bias.
    for (int e = 0; e < 8; ++e) {
        bias[e] = -2.0f - 0.03f * e;
        for (int t = 0; t < T; ++t) logits[t * 128 + e] = 2.0f - 0.1f * e;
    }
    // Second column changes selection; third forces an exact selection tie.
    bias[1] = bias[0];
    logits[128 + 7] = 5.0f; bias[7] = -2.21f;
    logits[2 * 128 + 1] = logits[2 * 128];
    // Build an independent CPU original ranking and selected-weight oracle.
    auto oracle = [&](int t) {
        std::vector<int> rank(128); for (int e = 0; e < 128; ++e) rank[e] = e;
        std::stable_sort(rank.begin(), rank.end(), [&](int a, int b) {
            float sa = 1.0f / (1.0f + std::exp(-logits[t * 128 + a])) + bias[a];
            float sb = 1.0f / (1.0f + std::exp(-logits[t * 128 + b])) + bias[b];
            return sa > sb || (sa == sb && a < b);
        });
        return rank;
    };
    float *dl, *db, *dw; int *di, *dor; Ready* dr; Stats* ds;
    CUDA(cudaMalloc(&dl, logits.size() * sizeof(float))); CUDA(cudaMalloc(&db, bias.size() * sizeof(float)));
    CUDA(cudaMalloc(&dw, T * 4 * sizeof(float))); CUDA(cudaMalloc(&di, T * 4 * sizeof(int)));
    CUDA(cudaMalloc(&dor, T * 4 * sizeof(int))); CUDA(cudaMalloc(&dr, sizeof(Ready))); CUDA(cudaMalloc(&ds, sizeof(Stats)));
    CUDA(cudaMemcpy(dl, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(db, bias.data(), bias.size() * sizeof(float), cudaMemcpyHostToDevice));
    Ready ready{}; ready.primary[0] = (1u << 4); ready.helper[0] = (1u << 5);
    CUDA(cudaMemcpy(dr, &ready, sizeof(ready), cudaMemcpyHostToDevice));
    std::vector<int> ids(T * 4), original(T * 4); std::vector<float> w(T * 4);
    auto fetch = [&] {
        CUDA(cudaMemcpy(ids.data(), di, ids.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(original.data(), dor, original.size() * sizeof(int), cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(w.data(), dw, w.size() * sizeof(float), cudaMemcpyDeviceToHost));
    };
    auto verify = [&](bool off) {
        for (int t = 0; t < T; ++t) {
            auto rank = oracle(t); float sum = 0;
            for (int r = 0; r < 4; ++r) {
                REQUIRE(original[t * 4 + r] == rank[r]);
                int id = ids[t * 4 + r]; REQUIRE(id >= 0 && id < 128);
                for (int j = 0; j < r; ++j) REQUIRE(id != ids[t * 4 + j]);
                if (off || r < 2) REQUIRE(id == rank[r]);
                if (id != rank[r]) {
                    REQUIRE(r >= 2); REQUIRE(id == 4 || id == 5);
                    REQUIRE(std::find(rank.begin() + 4, rank.begin() + 8, id) != rank.begin() + 8);
                }
                sum += 1.0f / (1.0f + std::exp(-logits[t * 128 + id]));
            }
            for (int r = 0; r < 4; ++r) {
                float p = 1.0f / (1.0f + std::exp(-logits[t * 128 + ids[t * 4 + r]]));
                REQUIRE(std::abs(w[t * 4 + r] - p / sum) < 1e-6f);
            }
        }
    };
    CUDA(cudaMemset(ds, 0, sizeof(Stats)));
    CUDA(route(dl, db, T, Policy::Off, 0.9f, dr, di, dw, dor, ds, nullptr)); fetch(); verify(true);
    CUDA(cudaMemset(ds, 0, sizeof(Stats)));
    CUDA(route(dl, db, T, Policy::Substitute, 0.9f, dr, di, dw, dor, ds, nullptr)); fetch(); verify(false);
    REQUIRE(ids[0] == 0 && ids[1] == 1 && ids[2] == 4 && ids[3] == 5);
    Stats stats{}; CUDA(cudaMemcpy(&stats, ds, sizeof(stats), cudaMemcpyDeviceToHost));
    unsigned long long swaps = 0, pri = 0, helper = 0, miss = 0;
    for (int i = 0; i < T * 4; ++i) {
        swaps += ids[i] != original[i]; pri += ids[i] == 4; helper += ids[i] == 5;
        miss += original[i] != 4 && original[i] != 5;
    }
    REQUIRE(stats.replacements == swaps && stats.primary_hits == pri && stats.helper_hits == helper && stats.original_misses == miss);
    // No published bit: including a potentially loading expert cannot make it eligible.
    ready.primary[0] = ready.helper[0] = 0; CUDA(cudaMemcpy(dr, &ready, sizeof(ready), cudaMemcpyHostToDevice));
    CUDA(route(dl, db, T, Policy::Substitute, 0.9f, dr, di, dw, dor, nullptr, nullptr)); fetch(); verify(true);
    // A strict ratio rejects otherwise ready candidates.
    ready.primary[0] = 1u << 4; ready.helper[0] = 1u << 5;
    CUDA(cudaMemcpy(dr, &ready, sizeof(ready), cudaMemcpyHostToDevice));
    CUDA(route(dl, db, T, Policy::Substitute, 1.0f, dr, di, dw, dor, nullptr, nullptr)); fetch();
    REQUIRE(ids[2] == original[2] && ids[3] == original[3]);
    REQUIRE(route(dl, db, T, Policy::Substitute, -0.1f, dr, di, dw, dor, nullptr, nullptr) == cudaErrorInvalidValue);
    // Cache-prior fixture: experts 10 and 11 are beyond the original top
    // eight, yet ready. Unpublished expert 9 has no bonus.
    std::fill(logits.begin(), logits.end(), 0.0f);
    std::fill(bias.begin(), bias.end(), -5.0f);
    for(int t=0;t<T;++t){logits[t*128+10]=0.4f;logits[t*128+11]=0.3f;}
    for(int e=0;e<12;++e) {
        const float probability=1.f/(1.f+std::exp(-logits[e]));
        bias[e]=-0.01f*e-(probability-0.5f);
    }
    CUDA(cudaMemcpy(dl,logits.data(),logits.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(db,bias.data(),bias.size()*sizeof(float),cudaMemcpyHostToDevice));
    ready={};ready.primary[0]=1u<<10;ready.helper[0]=1u<<11;
    CUDA(cudaMemcpy(dr,&ready,sizeof(ready),cudaMemcpyHostToDevice));
    CUDA(route(dl,db,T,Policy::Off,0.f,dr,di,dw,dor,nullptr,nullptr));fetch();
    const auto off_ids=ids,off_original=original;const auto off_weights=w;
    CUDA(route(dl,db,T,Policy::Prior,0.f,dr,di,dw,dor,nullptr,nullptr));fetch();
    REQUIRE(ids==off_ids && original==off_original && w==off_weights);
    CUDA(cudaMemset(ds,0,sizeof(Stats)));
    CUDA(route(dl,db,T,Policy::Prior,0.12f,dr,di,dw,dor,ds,nullptr));fetch();
    for(int t=0;t<T;++t){
        REQUIRE(original[t*4]==0 && original[t*4+1]==1 && original[t*4+2]==2 && original[t*4+3]==3);
        REQUIRE(ids[t*4]==10 && ids[t*4+1]==11 && ids[t*4+2]==0 && ids[t*4+3]==1);
        float p[4],sum=0;for(int r=0;r<4;++r){p[r]=1.f/(1.f+std::exp(-logits[t*128+ids[t*4+r]]));sum+=p[r];}
        for(int r=0;r<4;++r){REQUIRE(std::fabs(w[t*4+r]-p[r]/sum)<1e-6f);for(int j=0;j<r;++j)REQUIRE(ids[t*4+r]!=ids[t*4+j]);}
    }
    CUDA(cudaMemcpy(&stats,ds,sizeof(stats),cudaMemcpyDeviceToHost));
    REQUIRE(stats.replacements==2*T && stats.primary_hits==T && stats.helper_hits==T && stats.original_misses==4*T);
    // All ready bits clear: no bonus at any rank, including expert 9.
    ready={};CUDA(cudaMemcpy(dr,&ready,sizeof(ready),cudaMemcpyHostToDevice));
    CUDA(route(dl,db,T,Policy::Prior,0.12f,dr,di,dw,dor,nullptr,nullptr));fetch();
    REQUIRE(ids==off_ids && original==off_original && w==off_weights);
    // A rank permutation of the original set contributes no replacement.
    ready.primary[0]=1u<<3;CUDA(cudaMemcpy(dr,&ready,sizeof(ready),cudaMemcpyHostToDevice));
    CUDA(cudaMemset(ds,0,sizeof(Stats)));
    CUDA(route(dl,db,T,Policy::Prior,0.12f,dr,di,dw,dor,ds,nullptr));fetch();
    for(int t=0;t<T;++t){REQUIRE(ids[t*4]==3);for(int r=0;r<4;++r)REQUIRE(ids[t*4+r]<4);}
    CUDA(cudaMemcpy(&stats,ds,sizeof(stats),cudaMemcpyDeviceToHost));REQUIRE(stats.replacements==0);
    REQUIRE(route(dl,db,T,Policy::Prior,0.1f,nullptr,di,dw,dor,nullptr,nullptr)==cudaErrorInvalidValue);
    REQUIRE(route(dl,db,T,Policy::Prior,std::numeric_limits<float>::quiet_NaN(),dr,di,dw,dor,nullptr,nullptr)==cudaErrorInvalidValue);
    REQUIRE(route(dl,db,T,Policy::Prior,1.01f,dr,di,dw,dor,nullptr,nullptr)==cudaErrorInvalidValue);
    // Even when all experts are ready, adding a constant prior can round two
    // scores into a tie. The original histogram must count expert 4 rather
    // than the actually executed substitute 3, across repeated rows.
    std::fill(logits.begin(),logits.end(),0.f);std::fill(bias.begin(),bias.end(),-1.f);
    bias[0]=bias[1]=bias[2]=.25f;bias[3]=-0x1p-25f;bias[4]=0.f;
    ready={};for(auto& word:ready.primary)word=~uint32_t{0};
    CUDA(cudaMemcpy(dl,logits.data(),logits.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(db,bias.data(),bias.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(dr,&ready,sizeof(ready),cudaMemcpyHostToDevice));
    unsigned* histogram=nullptr;CUDA(cudaMalloc(&histogram,128*sizeof(unsigned)));
    CUDA(cudaMemset(histogram,0,128*sizeof(unsigned)));
    CUDA(route(dl,db,T,Policy::Prior,.5f,dr,di,dw,dor,nullptr,nullptr));
    CUDA(count_original(dor,T*4,histogram,nullptr));fetch();
    std::vector<unsigned> counts(128);CUDA(cudaMemcpy(counts.data(),histogram,128*sizeof(unsigned),cudaMemcpyDeviceToHost));
    for(int t=0;t<T;++t) {
        REQUIRE(original[t*4]==0 && original[t*4+1]==1 && original[t*4+2]==2 && original[t*4+3]==4);
        REQUIRE(ids[t*4]==0 && ids[t*4+1]==1 && ids[t*4+2]==2 && ids[t*4+3]==3);
    }
    REQUIRE(counts[0]==T && counts[1]==T && counts[2]==T && counts[4]==T && counts[3]==0);
    // Production original-demand counter, exercised at C1 and wider grouped
    // widths. Separate score and generation accumulators begin with nonzero
    // history; both must receive original IDs, even when the all-ready prior
    // changes the executed ID at a floating-point rounding tie.
    constexpr int max_rows=16;
    float *wide_logits=nullptr,*wide_weights=nullptr;int *wide_ids=nullptr,*wide_original=nullptr;
    unsigned *score_counts=nullptr,*generation_counts=nullptr;
    CUDA(cudaMalloc(&wide_logits,max_rows*128*sizeof(float)));CUDA(cudaMalloc(&wide_weights,max_rows*4*sizeof(float)));
    CUDA(cudaMalloc(&wide_ids,max_rows*4*sizeof(int)));CUDA(cudaMalloc(&wide_original,max_rows*4*sizeof(int)));
    CUDA(cudaMalloc(&score_counts,128*sizeof(unsigned)));CUDA(cudaMalloc(&generation_counts,128*sizeof(unsigned)));
    std::vector<float> wide_input(max_rows*128,0.f);
    CUDA(cudaMemcpy(wide_logits,wide_input.data(),wide_input.size()*sizeof(float),cudaMemcpyHostToDevice));
    for(int width : {1,8,16}) {
        std::vector<unsigned> expected_counts(128),score(128),generation(128);
        for(int e=0;e<128;++e)expected_counts[e]=unsigned(e%7);
        CUDA(cudaMemcpy(score_counts,expected_counts.data(),128*sizeof(unsigned),cudaMemcpyHostToDevice));
        CUDA(cudaMemcpy(generation_counts,expected_counts.data(),128*sizeof(unsigned),cudaMemcpyHostToDevice));
        CUDA(route(wide_logits,db,width,Policy::Prior,.5f,dr,wide_ids,wide_weights,wide_original,nullptr,nullptr));
        CUDA(count_original(wide_original,width*4,score_counts,nullptr));
        CUDA(count_original(wide_original,width*4,generation_counts,nullptr));
        std::vector<int> got_original(width*4),got_executed(width*4);
        CUDA(cudaMemcpy(got_original.data(),wide_original,got_original.size()*sizeof(int),cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(got_executed.data(),wide_ids,got_executed.size()*sizeof(int),cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(score.data(),score_counts,128*sizeof(unsigned),cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(generation.data(),generation_counts,128*sizeof(unsigned),cudaMemcpyDeviceToHost));
        for(int t=0;t<width;++t) {
            const int original_expected[4]={0,1,2,4}, executed_expected[4]={0,1,2,3};
            for(int k=0;k<4;++k) {
                REQUIRE(got_original[t*4+k]==original_expected[k]);
                REQUIRE(got_executed[t*4+k]==executed_expected[k]);
                ++expected_counts[original_expected[k]];
            }
        }
        REQUIRE(score==expected_counts && generation==expected_counts);
    }
    // Invalid sentinels cannot index the histogram; repeated valid IDs still
    // increment once per occurrence, including the upper boundary.
    const int mixed_ids[]={-1,128,127,127,0,-7,4,4};
    CUDA(cudaMemcpy(wide_original,mixed_ids,sizeof(mixed_ids),cudaMemcpyHostToDevice));
    CUDA(cudaMemset(score_counts,0,128*sizeof(unsigned)));
    CUDA(count_original(wide_original,int(sizeof(mixed_ids)/sizeof(mixed_ids[0])),score_counts,nullptr));
    std::vector<unsigned> mixed(128),mixed_expected(128,0);
    mixed_expected[0]=1;mixed_expected[4]=2;mixed_expected[127]=2;
    CUDA(cudaMemcpy(mixed.data(),score_counts,128*sizeof(unsigned),cudaMemcpyDeviceToHost));REQUIRE(mixed==mixed_expected);
    REQUIRE(count_original(nullptr,1,score_counts,nullptr)==cudaErrorInvalidValue);
    REQUIRE(count_original(wide_original,0,score_counts,nullptr)==cudaErrorInvalidValue);
    CUDA(cudaFree(wide_logits));CUDA(cudaFree(wide_weights));CUDA(cudaFree(wide_ids));CUDA(cudaFree(wide_original));
    CUDA(cudaFree(score_counts));CUDA(cudaFree(generation_counts));
    CUDA(cudaFree(histogram));
    // Restore the preceding fixture for invalid-input checks.
    std::fill(logits.begin(),logits.end(),0.f);std::fill(bias.begin(),bias.end(),-5.f);
    for(int e=0;e<12;++e) bias[e]=-0.01f*e;
    ready={};ready.primary[0]=1u<<3;
    CUDA(cudaMemcpy(dr,&ready,sizeof(ready),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(db,bias.data(),bias.size()*sizeof(float),cudaMemcpyHostToDevice));
    // An invalid token produces sentinel IDs/NaN weights and does not mutate
    // its counters; valid neighboring tokens still route normally.
    logits[128+42]=std::numeric_limits<float>::quiet_NaN();
    CUDA(cudaMemcpy(dl,logits.data(),logits.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA(cudaMemset(ds,0,sizeof(Stats)));
    CUDA(route(dl,db,T,Policy::Prior,0.12f,dr,di,dw,dor,ds,nullptr));fetch();
    for(int r=0;r<4;++r){REQUIRE(ids[4+r]==-1 && original[4+r]==-1 && std::isnan(w[4+r]));REQUIRE(ids[r]>=0 && ids[8+r]>=0);}
    CUDA(cudaMemcpy(&stats,ds,sizeof(stats),cudaMemcpyDeviceToHost));REQUIRE(stats.original_misses==6);
    logits[128+42]=0.f;bias[47]=std::numeric_limits<float>::infinity();
    CUDA(cudaMemcpy(dl,logits.data(),logits.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(db,bias.data(),bias.size()*sizeof(float),cudaMemcpyHostToDevice));
    CUDA(route(dl,db,T,Policy::Prior,0.12f,dr,di,dw,dor,nullptr,nullptr));fetch();
    for(int i=0;i<T*4;++i)REQUIRE(ids[i]==-1 && original[i]==-1 && std::isnan(w[i]));
    CUDA(cudaFree(dl)); CUDA(cudaFree(db)); CUDA(cudaFree(dw)); CUDA(cudaFree(di)); CUDA(cudaFree(dor)); CUDA(cudaFree(dr)); CUDA(cudaFree(ds));
    std::puts("resident router: OK");
}
