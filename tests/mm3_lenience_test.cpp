#include "mm3_sample.hpp"
#include "mm3_spec_state.hpp"
#include <cassert>
#include <cmath>
#include <random>
#include <vector>
using namespace mm3::sample;
int main() {
    // The first proposal has excessive q; the second is absent from q.
    const std::vector<std::pair<int,double>> p{{0,0.2},{1,0.5},{2,0.3}};
    const std::vector<std::pair<int,float>> q{{0,0.8f},{1,0.2f}};
    assert(std::abs(lenience_accept_prob(.2,.8,1) - .25) < 1e-12);
    assert(std::abs(lenience_accept_prob(.2,.8,.4) - .625) < 1e-12);
    assert(lenience_accept_prob(.5,.2,.4) == 1);
    assert(lenience_accept_prob(0,.2,.4) == 0);
    assert(lenience_accept_prob(.2,0,.4) == 0);
    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> u(0,1);
    for (double ell : {1.0, 0.4}) {
        int counts[3] = {}, residual[3] = {};
        constexpr int N = 300000;
        for (int i=0;i<N;++i) {
            const int x = u(rng) < .8 ? 0 : 1;
            const double px = x == 0 ? .2 : .5, qx = x == 0 ? .8 : .2;
            const int y = u(rng) < lenience_accept_prob(px,qx,ell) ? x : draw_residual(p,q,rng);
            ++counts[y]; ++residual[draw_residual(p,q,rng)];
        }
        // Residual is (0, .3, .3), independent of ell. For ell=1 the
        // complete mixture equals p; for .4 it intentionally differs.
        assert(residual[0] == 0 && std::abs(double(residual[1])/N - .5) < .005);
        const double expected[] = {ell == 1.0 ? .2 : .5, ell == 1.0 ? .5 : .35, ell == 1.0 ? .3 : .15};
        for (int j=0;j<3;++j) assert(std::abs(double(counts[j])/N - expected[j]) < .005);
    }
    // The actual shared frontier commits only the accepted prefix, leaving
    // rejected suffix rows for replacement and later prefix reuse.
    std::vector<int> state{1}, out;
    mm3::SpecState f(state); f.commit(2);
    auto emit = [&](int t) { out.push_back(t); return true; };
    assert(f.draft(true,false,3,emit)==mm3::SpecState::DraftResult::accepted);
    assert(f.draft(false,false,4,emit)==mm3::SpecState::DraftResult::rejected);
    assert((state==std::vector<int>{1,2,3}) && (out==std::vector<int>{3}));
    f.commit(5);
    assert(mm3::SpecState::common_prefix(state,{1,2,3,5,6}) == 4);
    assert(f.draft(true,true,6,emit)==mm3::SpecState::DraftResult::cancelled);
    assert((state==std::vector<int>{1,2,3,5}));
}
