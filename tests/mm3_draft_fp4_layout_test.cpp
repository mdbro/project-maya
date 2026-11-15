// tests/mm3_draft_fp4_layout_test.cpp - mm3_draft_fp4.cuh's layouts, host only (no GPU): random NVFP4 weights retiled by
// tile_host and random activations quantized by quantize_row_host and laid out as quant_kernel writes them (x_word /
// x_scale), then multiplied by an emulation of mma.sync kind::mxf4nvf4 m16n8k64 that reads each operand from the lanes
// and registers PTX's fragment layout gives it (the mapping Infernix's ldmatrix fragments use, nvfp4_a4_mma.cuh),
// walking the tiles with the kernel's own addressing - against the direct FP64 product of the dequantized row-major
// operands. Also: E2M1's ties, the row global scale, and the second activation term's error.
//   mm3_draft_fp4_layout_test        exit 0: every check passes
#include "mm3_draft_fp4.cuh"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace f4 = mm3::draft_fp4;

namespace {
int fails = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("FAIL %s\n", what);
        ++fails;
    }
}
uint32_t word(const uint8_t* p) {
    uint32_t w;
    std::memcpy(&w, p, 4);
    return w;
}
double nib(uint32_t w, int j) { return f4::e2m1_value((w >> (4 * j)) & 15u); }

// one weight matrix [N][K] x rows activations through the emulated kernel and directly; returns the max relative error
double one(int N, int K, int rows, int terms, std::mt19937& rng) {
    const int KC = K / 64;
    std::vector<uint8_t> codes((size_t) N * K / 2), scales((size_t) N * K / 16);
    std::uniform_int_distribution<int> byte(0, 255), sc(0x18, 0x58);
    for (auto& c : codes) c = (uint8_t) byte(rng);
    for (auto& s : scales) s = (uint8_t) sc(rng);
    const float gs = 0.00123f;
    std::vector<uint8_t> wt(f4::codes_bytes(N, K)), st(f4::scales_bytes(N, K));
    f4::tile_host(codes.data(), scales.data(), N, K, wt.data(), st.data());
    // activations: Gaussian, every 97th input x12; quantized per row and laid out as quant_kernel writes them
    std::normal_distribution<float> nd(0.0f, 1.0f);
    std::vector<float> x((size_t) rows * K);
    for (int r = 0; r < rows; ++r)
        for (int k = 0; k < K; ++k) x[(size_t) r * K + k] = nd(rng) * (k % 97 == 0 ? 12.0f : 1.0f);
    std::vector<uint8_t> xq(f4::act_bytes(K, terms), 0);
    std::vector<float> xgs(16, 0.0f);
    std::vector<uint8_t> xc((size_t) rows * terms * K / 2), xs((size_t) rows * terms * K / 16);
    for (int r = 0; r < rows; ++r) {
        uint8_t* cr = xc.data() + (size_t) r * terms * K / 2;
        uint8_t* sr = xs.data() + (size_t) r * terms * K / 16;
        xgs[(size_t) r] = f4::quantize_row_host(x.data() + (size_t) r * K, K, terms, cr, sr);
        for (int term = 0; term < terms; ++term)
            for (int G = 0; G < K / 16; ++G) {
                const int kc = G >> 2, j = G & 3;
                std::memcpy(xq.data() + f4::x_word(KC, term, kc, r, 2 * j), cr + (size_t) term * K / 2 + 8 * G, 4);
                std::memcpy(xq.data() + f4::x_word(KC, term, kc, r, 2 * j + 1), cr + (size_t) term * K / 2 + 8 * G + 4, 4);
                xq[f4::x_scale(KC, term, kc, r, j)] = sr[(size_t) term * K / 16 + G];
            }
    }
    // the emulated kernel: for each 16-row block and chunk, every lane's registers as the kernel loads them, then the
    // MMA as PTX defines it from those registers
    std::vector<double> y((size_t) rows * N, 0.0);
    const int NB = (rows + 7) / 8;
    for (int rb = 0; rb < N / 16; ++rb)
        for (int kc = 0; kc < KC; ++kc) {
            uint32_t A[32][4], SA[32], B[2][32][2][2], SB[2][32][2];
            for (int lane = 0; lane < 32; ++lane) {
                const int g = lane >> 2, t = lane & 3;
                // the kernel: wt = w + w_word(KC, rb, 0, g, t, 0) as uint4, + 32 uint4 a chunk; st likewise, + 16 words
                const uint8_t* a = wt.data() + f4::w_word(KC, rb, 0, g, t, 0) + (size_t) kc * 512;
                for (int reg = 0; reg < 4; ++reg) A[lane][reg] = word(a + 4 * reg);
                SA[lane] = word(st.data() + f4::w_scale(KC, rb, 0, g + 8 * (t & 1)) + (size_t) kc * 64);
                for (int nb = 0; nb < NB; ++nb)
                    for (int term = 0; term < terms; ++term) {
                        const size_t o = (size_t) term * KC * 2 * 288;
                        const uint8_t* b = xq.data() + f4::x_word(KC, 0, 0, g, t) + (size_t) (2 * kc + nb) * 256 + o;
                        B[nb][lane][term][0] = word(b);
                        B[nb][lane][term][1] = word(b + 4);
                        SB[nb][lane][term] = word(xq.data() + f4::x_scale(KC, 0, 0, g, 0) + (size_t) (2 * kc + nb) * 32 + o);
                    }
            }
            for (int nb = 0; nb < NB; ++nb)
                for (int term = 0; term < terms; ++term)
                    for (int r = 0; r < 16; ++r)
                        for (int c = 0; c < 8; ++c) {
                            const int tok = nb * 8 + c;
                            if (tok >= rows) continue;
                            double acc = 0.0;
                            for (int k = 0; k < 64; ++k) {
                                // A (row r, input k): lane (r & 7, (k % 32) / 8), register a0 / a1 (k < 32) or a2 / a3,
                                // by r < 8 or not; nibble k % 8. Its scale: lane (r & 7, r < 8 ? 0 : 1), byte k / 16.
                                const int la = 4 * (r & 7) + (k % 32) / 8;
                                const int ra = (k < 32 ? 0 : 2) + (r < 8 ? 0 : 1);
                                const double av = nib(A[la][ra], k % 8);
                                const double as = f4::e4m3_value((uint8_t) (SA[4 * (r & 7) + (r < 8 ? 0 : 1)] >> (8 * (k / 16))));
                                // B (input k, column c): lane (c, (k % 32) / 8), register b0 (k < 32) or b1; its scale:
                                // lane (c, 0), byte k / 16
                                const int lb = 4 * c + (k % 32) / 8;
                                const double bv = nib(B[nb][lb][term][k < 32 ? 0 : 1], k % 8);
                                const double bs = f4::e4m3_value((uint8_t) (SB[nb][4 * c][term] >> (8 * (k / 16))));
                                acc += av * as * bv * bs;
                            }
                            y[(size_t) tok * N + rb * 16 + r] += acc;
                        }
        }
    // direct: the row-major operands dequantized
    double err = 0.0;
    for (int r = 0; r < rows; ++r) {
        std::vector<double> xd((size_t) K, 0.0);
        for (int term = 0; term < terms; ++term)
            for (int k = 0; k < K; ++k) {
                const uint8_t c = xc[(size_t) r * terms * K / 2 + (size_t) term * K / 2 + k / 2];
                xd[(size_t) k] += f4::e2m1_value((k & 1) ? c >> 4 : c & 15u) *
                                  f4::e4m3_value(xs[(size_t) r * terms * K / 16 + (size_t) term * K / 16 + k / 16]);
            }
        for (int n = 0; n < N; ++n) {
            double ref = 0.0, mag = 0.0;
            for (int k = 0; k < K; ++k) {
                const uint8_t c = codes[(size_t) n * K / 2 + k / 2];
                const double w = f4::e2m1_value((k & 1) ? c >> 4 : c & 15u) * f4::e4m3_value(scales[(size_t) n * K / 16 + k / 16]);
                ref += w * xd[(size_t) k];
                mag += std::fabs(w * xd[(size_t) k]);
            }
            const double got = y[(size_t) r * N + n] * gs * xgs[(size_t) r], want = ref * gs * xgs[(size_t) r];
            const double scale = mag * gs * xgs[(size_t) r];
            if (scale > 0) err = std::max(err, std::fabs(got - want) / scale);
        }
    }
    return err;
}
}  // namespace

int main() {
    // E2M1: nearest, ties to the even code, saturating
    {
        float v[16] = {};
        // with max |x| 6 the global scale makes the largest block's scale 448 (2688 / 6), so x / scale stays x / 1 here:
        // check codes through quantize_row_host on a block whose max is 6 (scale byte 448 = 0x7e, values x * 448 / 448)
        const float probe[16] = {6.0f, 0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f, -0.25f, -0.75f, 0.2f, 0.3f, 2.4f, 2.6f, 4.9f, 5.1f};
        std::memcpy(v, probe, sizeof v);
        uint8_t codes[8], scales[1];
        const float gx = f4::quantize_row_host(v, 16, 1, codes, scales);
        check(std::fabs(gx - 6.0f / 2688.0f) < 1e-9f, "row global scale = max / 2688");
        check(scales[0] == 0x7e, "the largest block's scale is 448");
        const float want[16] = {6.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f, 4.0f, 4.0f, -0.0f, -1.0f, 0.0f, 0.5f, 2.0f, 3.0f, 4.0f, 6.0f};
        for (int i = 0; i < 16; ++i) {
            const unsigned code = (i & 1) ? codes[i / 2] >> 4 : codes[i / 2] & 15u;
            if (f4::e2m1_value(code) != want[i]) {
                std::printf("  value %g -> code %u (%g), want %g\n", probe[i], code, f4::e2m1_value(code), want[i]);
                check(false, "E2M1 rounding");
            }
        }
    }
    // the second term: activation error against the first's
    {
        std::mt19937 rng(7);
        std::normal_distribution<float> nd(0.0f, 1.0f);
        const int K = 6144;
        std::vector<float> x((size_t) K);
        for (int k = 0; k < K; ++k) x[(size_t) k] = nd(rng) * (k % 97 == 0 ? 12.0f : 1.0f);
        double e1 = 0, e2 = 0, ref = 0;
        for (int terms = 1; terms <= 2; ++terms) {
            std::vector<uint8_t> codes((size_t) terms * K / 2), scales((size_t) terms * K / 16);
            const float gx = f4::quantize_row_host(x.data(), K, terms, codes.data(), scales.data());
            for (int k = 0; k < K; ++k) {
                double v = 0;
                for (int term = 0; term < terms; ++term) {
                    const uint8_t c = codes[(size_t) term * K / 2 + k / 2];
                    v += f4::e2m1_value((k & 1) ? c >> 4 : c & 15u) * f4::e4m3_value(scales[(size_t) term * K / 16 + k / 16]);
                }
                v *= gx;
                const double d = v - x[(size_t) k];
                (terms == 1 ? e1 : e2) += d * d;
                if (terms == 1) ref += (double) x[(size_t) k] * x[(size_t) k];
            }
        }
        e1 = std::sqrt(e1 / ref);
        e2 = std::sqrt(e2 / ref);
        std::printf("activation relative RMS error: one term %.4f, two terms %.4f\n", e1, e2);
        check(e1 > 0.05 && e1 < 0.15, "one term's error in NVFP4's range");
        check(e2 < 0.2 * e1, "the second term cuts the error five-fold or more");
    }
    // the layouts through the emulated MMA
    std::mt19937 rng(20261114);
    const struct { int N, K, rows, terms; } cases[] = {{32, 128, 8, 1}, {48, 256, 5, 1}, {32, 192, 16, 1}, {16, 64, 12, 2},
                                                       {64, 320, 8, 2}, {32, 128, 1, 1}};
    for (const auto& c : cases) {
        const double e = one(c.N, c.K, c.rows, c.terms, rng);
        std::printf("N %d K %d rows %d terms %d: max error %.3g of the products' magnitude\n", c.N, c.K, c.rows, c.terms, e);
        check(e < 1e-12, "emulated kernel = direct product");
    }
    std::printf(fails ? "%d FAILED\n" : "all passed\n", fails);
    return fails ? 1 : 0;
}
