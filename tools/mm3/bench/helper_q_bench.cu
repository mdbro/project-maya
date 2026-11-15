// Standalone bench of kernel item 5 on one helper card (the RTX 3090 or the RTX 4080): the NVFP4-activation expert kernel
// on int8 tensor cores (src/mm3/mm3_helper_q.cuh) against today's tensor-core kernel (mm3_helper_kernels.cuh, BF16
// activations), each from the sm_86 build and, on the 4080, from the sm_89 build (mm3_helper_sm89.cu); and the bytes a
// pass's inputs take across the card's link as BF16 columns and as the 5090's packed NVFP4 rows.
//
// Checks (each printed PASS or FAIL):
//   quant      the card's encoding of BF16 columns = the host's canon::quantize_a4_codes (the 5090's rule), bit for bit
//   packed     rows packed from F32 on the 5090 (or, without one, on the card) and expanded on the card = the card's own
//              encoding of the same rows rounded to BF16 on the host, bit for bit (sending the rows changes nothing)
//   products   one unit's int8 tensor-core products as the kernel forms them (decoded weights, activation fragments) =
//              the host's integer sums, for every lane, column and row (the 4-bit x 4-bit products are exact)
//   repeat     the kernel twice on the same inputs: the same bits
//   sm89       (4080) the sm_89 build's outputs = the sm_86 build's, both kernels, bit for bit
//   pack bf16  (kernel item 5b) BF16 columns packed on the card = the 5090's packed rows of the same F32 rows, bit for bit
//   variant 2  (kernel item 5b) its outputs = variant 1's bit for bit, from packed rows and from BF16 columns, at every
//              warp count (1, 2, 4, 8) on jobs of 1, 8 and 3 columns, and at the serving warps on every timed shape;
//              on the 4080 also from the sm_89 build
//   accuracy   (reported, not judged) relative RMS of each kernel against the double-precision product of its own inputs
//              (tensor-core: the BF16 activations; NVFP4: their 4-bit encoding, the hidden vector encoded again), and of
//              the NVFP4 kernel's outputs against the tensor-core kernel's: the size of the precision change
//
// Timing: today's tensor-core kernel against variant 1 as the engine runs it with the 5090's rows (expanded on the card,
// then its kernels), variant 2 from the same packed rows, and variant 1 from BF16 columns (encoded on the card).
//
//   build:  cmake --build <build dir> --target mm3_helper_q_bench -j 15        (not while a timed request runs)
//   run:    <build dir>/mm3_helper_q_bench --card 3090 [--iters 200] [--experts 16]
//           <build dir>/mm3_helper_q_bench --card 4080 ...
//           (cards found by name; nothing else may use the card; the 5090 is used for the packed check if present)
#include "mm3_helper_kernels.cuh"
#include "mm3_helper_q.cuh"
#include "mm3_helper_sm89.hpp"
#include "ops/common/canonical_math.h"
#include "ops/minimax_nvfp4_ffn/fast_math.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
namespace mf = infernix::ops::minimax_ffn;
namespace tc = mm3::helper_tc;
namespace hq = mm3::helper_q;
namespace canon = infernix::ops::canon;
using mf::FastJob;
constexpr int H = 6144, I = 3072, U = 144;
constexpr std::size_t kRecord = (std::size_t(2 * I / 16) * (H / 16) + std::size_t(H / 16) * (I / 16)) * U;
int failures = 0;
void ck(cudaError_t e, const char* what) { if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e)); }
void verdict(bool ok, const char* name, const std::string& detail) {
    std::printf("  %-9s %s  %s\n", name, ok ? "PASS" : "FAIL", detail.c_str());
    failures += !ok;
}
int find_card(const char* name, bool required) {
    int n = 0;
    ck(cudaGetDeviceCount(&n), "device count");
    for (int i = 0; i < n; ++i) { cudaDeviceProp p{}; ck(cudaGetDeviceProperties(&p, i), "properties"); if (std::strstr(p.name, name)) return i; }
    if (required) throw std::runtime_error(std::string("no ") + name);
    return -1;
}
// a record like the checkpoint's: codes with the frequencies NVFP4 gives Gaussian weights, scales E4M3 0.25 .. 1.9
void make_record(std::uint8_t* rec, std::mt19937_64& r) {
    static const double p[8] = {0.10, 0.19, 0.17, 0.14, 0.17, 0.13, 0.07, 0.03};
    std::discrete_distribution<int> mag(p, p + 8);
    for (std::size_t u = 0; u < kRecord / U; ++u) {
        std::uint8_t* unit = rec + u * U;
        for (int i = 0; i < 128; ++i) unit[i] = std::uint8_t((mag(r) | (r() & 1) << 3) | (mag(r) | (r() & 1) << 3) << 4);
        for (int row = 0; row < 16; ++row) unit[128 + row] = std::uint8_t((5 + r() % 3) << 3 | (r() & 7));
    }
}
double e4m3(unsigned x) { const int e = (x >> 3) & 15, m = x & 7; return e ? std::ldexp(double(8 + m), e - 10) : std::ldexp(double(m), -9); }
const int kT[8] = {0, 1, 2, 3, 4, 6, 8, 12};
void decode_row(const std::uint8_t* matrix, int blocks, int row, std::vector<double>& out) {
    const int group = row / 16, rr = row % 16;
    for (int b = 0; b < blocks; ++b) {
        const std::uint8_t* unit = matrix + (std::size_t(group) * blocks + b) * U;
        const double scale = 0.5 * e4m3(unit[128 + rr]);
        for (int k = 0; k < 16; ++k) {
            const unsigned byte = unit[(k / 4) * 32 + (rr % 8) * 4 + (k % 4)], code = rr < 8 ? byte & 15 : byte >> 4;
            out[std::size_t(b) * 16 + k] = (code & 8 ? -1.0 : 1.0) * kT[code & 7] * scale;
        }
    }
}
// the 5090's encoding of a BF16 vector, dequantized: code * scale (values / 2 * e4m3 of the scale word)
void a4_values(const std::uint16_t* v, int n, std::vector<double>& out) {
    out.resize(std::size_t(n));
    for (int b = 0; b < n / 16; ++b) {
        const canon::A4Codes q = canon::quantize_a4_codes(v + b * 16, 1.0f);
        for (int k = 0; k < 16; ++k) out[std::size_t(b) * 16 + k] = 0.5 * canon::e2m1_x2(q.code[k]) * e4m3(q.scale_word);
    }
}
// y = down(act(gate x, up x)) in double from given input values; a4: the hidden vector rounded to BF16 and encoded
// (the NVFP4 kernel's inputs), else rounded to BF16 only (the tensor-core kernel's)
void reference(const std::uint8_t* rec, const float* mult, const std::vector<double>& xs, int cols, bool a4, std::vector<double>& y) {
    std::vector<double> hidden(std::size_t(cols) * I), w(H), w2(H);
    std::vector<std::uint16_t> hb(std::size_t(cols) * I);
    for (int p = 0; p < I; ++p) {
        decode_row(rec, H / 16, 2 * p, w);
        decode_row(rec, H / 16, 2 * p + 1, w2);
        for (int c = 0; c < cols; ++c) {
            double g = 0, u = 0; const double* xc = xs.data() + std::size_t(c) * H;
            for (int k = 0; k < H; ++k) { g += w[k] * xc[k]; u += w2[k] * xc[k]; }
            g = std::min(g * double(mult[0]), 7.0);
            u = std::min(std::max(u * double(mult[1]), -7.0), 7.0);
            hb[std::size_t(c) * I + p] = mf::float_bf16(float(g / (1.0 + std::exp(-1.702 * g)) * (u + 1.0)));
        }
    }
    for (int c = 0; c < cols; ++c) {
        if (a4) {
            std::vector<double> v;
            a4_values(hb.data() + std::size_t(c) * I, I, v);
            std::copy(v.begin(), v.end(), hidden.begin() + std::size_t(c) * I);
        } else for (int p = 0; p < I; ++p) hidden[std::size_t(c) * I + p] = mf::bf16_float(hb[std::size_t(c) * I + p]);
    }
    y.assign(std::size_t(cols) * H, 0.0);
    std::vector<double> wd(I);
    for (int row = 0; row < H; ++row) {
        decode_row(rec + tc::kDownOffset, I / 16, row, wd);
        for (int c = 0; c < cols; ++c) {
            double s = 0; const double* hc = hidden.data() + std::size_t(c) * I;
            for (int k = 0; k < I; ++k) s += wd[k] * hc[k];
            y[std::size_t(c) * H + row] = s * double(mult[2]);
        }
    }
}
double rel_rms(const std::uint16_t* got, const std::vector<double>& want) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < want.size(); ++i) { const double d = double(mf::bf16_float(got[i])) - want[i]; num += d * d; den += want[i] * want[i]; }
    return std::sqrt(num / den);
}
double rel_rms_bf16(const std::uint16_t* a, const std::uint16_t* b, std::size_t n) {
    double num = 0, den = 0;
    for (std::size_t i = 0; i < n; ++i) { const double x = mf::bf16_float(a[i]), y = mf::bf16_float(b[i]); num += (x - y) * (x - y); den += y * y; }
    return std::sqrt(num / den);
}
// The products check: unit u of a record against q column c, each lane's four sums as the kernel forms them.
__global__ void products_kernel(const std::uint8_t* units, const std::uint8_t* q, int* out, int count) {
    const int u = blockIdx.x, lane = threadIdx.x, g = lane >> 2, c = lane & 3;
    if (u >= count) return;
    const std::uint8_t* unit = units + std::size_t(u) * U;
    const unsigned w = *reinterpret_cast<const unsigned*>(unit + c * 32 + g * 4);
    const unsigned a0 = canon::e2m1_x2_quad(w & 0x0f0f0f0fu), a1 = canon::e2m1_x2_quad((w >> 4) & 0x0f0f0f0fu);
    const unsigned b0 = *reinterpret_cast<const unsigned*>(q + std::size_t(g) * hq::kInStride + std::size_t(u % (H / 16)) * 16 + 4 * c);
    int d0, d1, d2, d3;
    hq::detail::tile_s8(d0, d1, d2, d3, a0, a1, b0);
    int* o = out + (std::size_t(u) * 32 + lane) * 4;
    o[0] = d0; o[1] = d1; o[2] = d2; o[3] = d3;
}
template <int S, int P> cudaError_t tc_sp(const FastJob* j, int n, void* ws, cudaStream_t s) { return P == 1 ? tc::ffn1<S>(j, n, ws, s) : tc::ffn2<S, P>(j, n, ws, s); }
template <int S> cudaError_t tc_s(int loads, const FastJob* j, int n, void* ws, cudaStream_t s) {
    switch (loads) { case 2: return tc_sp<S, 2>(j, n, ws, s); case 4: return tc_sp<S, 4>(j, n, ws, s); case 8: return tc_sp<S, 8>(j, n, ws, s); default: return tc_sp<S, 1>(j, n, ws, s); }
}
cudaError_t tc_launch(bool sm89, int warps, int loads, const FastJob* j, int n, void* ws, cudaStream_t s) {
    if (sm89) return mm3::helper_sm89::tensor(warps, loads, j, n, ws, s);
    switch (warps) { case 1: return tc_s<1>(loads, j, n, ws, s); case 2: return tc_s<2>(loads, j, n, ws, s); case 4: return tc_s<4>(loads, j, n, ws, s); default: return tc_s<8>(loads, j, n, ws, s); }
}
}  // namespace

int main(int argc, char** argv) {
    std::string card = "3090";
    int iters = 200, experts = 16;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() { if (i + 1 >= argc) throw std::runtime_error("missing value for " + a); return std::string(argv[++i]); };
        if (a == "--card") card = next();
        else if (a == "--iters") iters = std::stoi(next());
        else if (a == "--experts") experts = std::max(8, std::stoi(next()));
        else { std::fprintf(stderr, "usage: %s --card 3090|4080 [--iters N] [--experts N]\n", argv[0]); return 2; }
    }
    try {
        const bool is4080 = card == "4080";
        const int dev = find_card(is4080 ? "RTX 4080" : "RTX 3090", true), primary = find_card("RTX 5090", false);
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, dev), "properties");
        const bool sm89 = prop.major == 8 && prop.minor == 9 && mm3::helper_sm89::built();
        // serving's settings for the card (mm3_sample.hpp): one-column passes 4 warps (3090) / 8 (4080), wider 8; loads 8 / 4
        const int one_warps = is4080 ? 8 : 4, multi_warps = 8, loads = is4080 ? 4 : 8;
        std::printf("helper_q_bench: CUDA%d %s (sm_%d%d)%s, %d records, %d iterations a shape; warps %d one-column / %d wider, loads %d\n",
                    dev, prop.name, prop.major, prop.minor, sm89 ? ", with the sm_89 build" : "", experts, iters, one_warps, multi_warps, loads);
        ck(cudaSetDevice(dev), "set card");
        cudaStream_t s;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        std::mt19937_64 rng(20261114);
        std::uint8_t* records = nullptr;
        ck(cudaMalloc(&records, kRecord * experts), "records");
        std::vector<std::uint8_t> host_rec(kRecord);
        std::vector<std::vector<std::uint8_t>> keep(3);   // (records 0..2 kept on the host for the references)
        for (int e = 0; e < experts; ++e) {
            make_record(host_rec.data(), rng);
            if (e < 3) keep[size_t(e)] = host_rec;
            ck(cudaMemcpy(records + kRecord * e, host_rec.data(), kRecord, cudaMemcpyHostToDevice), "record up");
        }
        const float mult[3] = {1.0f / 2.0f, 1.0f / 2.5f, 1.0f / 3.0f};
        constexpr int kCols = 64;   // a pass's columns at most (16 rows x 4 picks)
        std::uint16_t *d_x = nullptr, *d_y = nullptr, *d_y2 = nullptr;
        std::uint8_t *d_q = nullptr, *d_hq = nullptr, *d_packed = nullptr, *d_q2 = nullptr;
        void *ws = nullptr, *d_jobs = nullptr;
        ck(cudaMalloc(&d_x, std::size_t(kCols) * H * 2), "x"); ck(cudaMalloc(&d_y, std::size_t(kCols) * H * 2), "y");
        ck(cudaMalloc(&d_y2, std::size_t(kCols) * H * 2), "y2");
        ck(cudaMalloc(&d_q, std::size_t(kCols) * hq::kInStride), "q"); ck(cudaMalloc(&d_q2, std::size_t(kCols) * hq::kInStride), "q2");
        ck(cudaMalloc(&d_hq, std::size_t(kCols) * hq::kJobHidden), "hidden q");
        ck(cudaMalloc(&d_packed, std::size_t(kCols) * hq::kPackedRow), "packed");
        ck(cudaMalloc(&ws, tc::workspace_bytes(kCols)), "workspace"); ck(cudaMalloc(&d_jobs, sizeof(FastJob) * kCols), "jobs");
        // inputs: Gaussian rows with a few outliers, as BF16 (the host's rounding of F32)
        std::vector<float> xf(std::size_t(kCols) * H);
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : xf) v = nd(rng) * (rng() % 97 == 0 ? 8.f : 1.f);
        std::vector<std::uint16_t> xb(xf.size());
        for (std::size_t i = 0; i < xf.size(); ++i) { std::uint32_t u; std::memcpy(&u, &xf[i], 4); xb[i] = std::uint16_t((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
        ck(cudaMemcpy(d_x, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice), "x up");

        std::vector<std::uint8_t> packed;   // (the 5090's packed rows of xf, made by the packed check)
        std::printf("\nchecks\n");
        {   // quant: the card's encoding against the host's
            ck(hq::quant_bf16(d_x, d_q, kCols, s), "quant"); ck(cudaStreamSynchronize(s), "quant");
            std::vector<std::uint8_t> got(std::size_t(kCols) * hq::kInStride);
            ck(cudaMemcpy(got.data(), d_q, got.size(), cudaMemcpyDeviceToHost), "quant back");
            long bad = 0;
            for (int c = 0; c < kCols; ++c)
                for (int b = 0; b < H / 16; ++b) {
                    const canon::A4Codes q = canon::quantize_a4_codes(xb.data() + std::size_t(c) * H + b * 16, 1.0f);
                    const std::uint8_t* col = got.data() + std::size_t(c) * hq::kInStride;
                    bad += col[H + b] != q.scale_word;
                    for (int k = 0; k < 16; ++k) bad += std::int8_t(col[b * 16 + k]) != canon::e2m1_x2(q.code[k]);
                }
            verdict(bad == 0, "quant", std::to_string(kCols * H / 16) + " blocks, " + std::to_string(bad) + " differing values");
        }
        {   // packed: F32 rows packed (on the 5090 if present), expanded on the card, against the card's own encoding
            const int pdev = primary >= 0 ? primary : dev;
            ck(cudaSetDevice(pdev), "set packer");
            float* pf = nullptr; std::uint8_t* pp = nullptr;
            ck(cudaMalloc(&pf, xf.size() * 4), "packer rows"); ck(cudaMalloc(&pp, std::size_t(kCols) * hq::kPackedRow), "packer out");
            ck(cudaMemcpy(pf, xf.data(), xf.size() * 4, cudaMemcpyHostToDevice), "rows up");
            ck(hq::pack_rows(pf, H, kCols, pp, nullptr), "pack"); ck(cudaDeviceSynchronize(), "pack");
            packed.assign(std::size_t(kCols) * hq::kPackedRow, 0);
            ck(cudaMemcpy(packed.data(), pp, packed.size(), cudaMemcpyDeviceToHost), "packed back");
            cudaFree(pf); cudaFree(pp);
            ck(cudaSetDevice(dev), "set card");
            ck(cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice), "packed up");
            ck(hq::expand_packed(d_packed, d_q2, kCols, s), "expand"); ck(cudaStreamSynchronize(s), "expand");
            std::vector<std::uint8_t> a(std::size_t(kCols) * hq::kInStride), b(a.size());
            ck(cudaMemcpy(a.data(), d_q, a.size(), cudaMemcpyDeviceToHost), "q back"); ck(cudaMemcpy(b.data(), d_q2, b.size(), cudaMemcpyDeviceToHost), "q2 back");
            verdict(a == b, "packed", std::string("rows packed on ") + (primary >= 0 ? "the 5090" : "the card (no 5090 visible)") +
                    ", " + std::to_string(hq::kPackedRow) + " bytes a row against " + std::to_string(H * 2) + " as BF16");
        }
        {   // products: 384 units of record 0 against columns 0..7
            constexpr int n = 384;
            int* d_out = nullptr;
            ck(cudaMalloc(&d_out, std::size_t(n) * 32 * 4 * 4), "products out");
            products_kernel<<<n, 32, 0, s>>>(records, d_q, d_out, n);
            ck(cudaGetLastError(), "products"); ck(cudaStreamSynchronize(s), "products");
            std::vector<int> got(std::size_t(n) * 128);
            ck(cudaMemcpy(got.data(), d_out, got.size() * 4, cudaMemcpyDeviceToHost), "products back");
            cudaFree(d_out);
            std::vector<std::uint8_t> q(std::size_t(8) * hq::kInStride);
            ck(cudaMemcpy(q.data(), d_q, q.size(), cudaMemcpyDeviceToHost), "q back");
            long bad = 0;
            for (int u = 0; u < n; ++u) {
                const std::uint8_t* unit = keep[0].data() + std::size_t(u) * U;
                for (int lane = 0; lane < 32; ++lane) {
                    const int g = lane >> 2, c = lane & 3;
                    for (int v = 0; v < 4; ++v) {
                        const int row = g + (v >= 2 ? 8 : 0), col = 2 * c + (v & 1);
                        long sum = 0;
                        for (int k = 0; k < 16; ++k) {
                            const unsigned byte = unit[(k / 4) * 32 + (row % 8) * 4 + (k % 4)], code = row < 8 ? byte & 15 : byte >> 4;
                            sum += long(canon::e2m1_x2(code)) * std::int8_t(q[std::size_t(col) * hq::kInStride + std::size_t(u % (H / 16)) * 16 + k]);
                        }
                        bad += got[(std::size_t(u) * 32 + lane) * 4 + v] != sum;
                    }
                }
            }
            verdict(bad == 0, "products", std::to_string(n * 32 * 4) + " int32 sums against the host's, " + std::to_string(bad) + " differing");
        }
        // one pass of jobs over records 0..2: job 0 one column, job 1 eight columns, job 2 three columns
        const auto make_jobs = [&](std::vector<FastJob>& jobs, const std::vector<int>& widths, int first_record) {
            jobs.clear(); int col = 0;
            for (std::size_t i = 0; i < widths.size(); ++i) {
                FastJob f{}; f.record = records + kRecord * ((first_record + int(i)) % experts);
                for (int k = 0; k < 3; ++k) f.multiplier[k] = mult[k];
                f.x = d_x + std::size_t(col) * H; f.y = d_y + std::size_t(col) * H; f.columns = widths[i];
                jobs.push_back(f); col += widths[i];
            }
            return col;
        };
        const auto run_q = [&](bool use89, int warps, const std::vector<FastJob>& jobs, int cols) {
            ck(cudaMemcpyAsync(d_jobs, jobs.data(), jobs.size() * sizeof(FastJob), cudaMemcpyHostToDevice, s), "jobs up");
            if (use89) {
                ck(mm3::helper_sm89::q_quant_bf16(d_x, d_q, cols, s), "quant 89");
                ck(mm3::helper_sm89::q_ffn(warps, loads, static_cast<FastJob*>(d_jobs), int(jobs.size()), d_x, d_q, d_hq, s), "q 89");
            } else {
                ck(hq::quant_bf16(d_x, d_q, cols, s), "quant");
                ck(hq::ffn(warps, loads, static_cast<FastJob*>(d_jobs), int(jobs.size()), d_x, d_q, d_hq, nullptr, s), "q");
            }
        };
        const auto run_tc = [&](bool use89, int warps, const std::vector<FastJob>& jobs) {
            ck(cudaMemcpyAsync(d_jobs, jobs.data(), jobs.size() * sizeof(FastJob), cudaMemcpyHostToDevice, s), "jobs up");
            ck(tc_launch(use89, warps, loads, static_cast<FastJob*>(d_jobs), int(jobs.size()), ws, s), "tc");
        };
        {   // repeat, sm89 and accuracy
            std::vector<FastJob> jobs;
            const int cols = make_jobs(jobs, {1, 8, 3}, 0);
            const std::size_t n = std::size_t(cols) * H;
            std::vector<std::uint16_t> q1(n), q2(n), t1(n);
            run_q(false, multi_warps, jobs, cols); ck(cudaStreamSynchronize(s), "q");
            ck(cudaMemcpy(q1.data(), d_y, n * 2, cudaMemcpyDeviceToHost), "y");
            run_q(false, multi_warps, jobs, cols); ck(cudaStreamSynchronize(s), "q");
            ck(cudaMemcpy(q2.data(), d_y, n * 2, cudaMemcpyDeviceToHost), "y");
            verdict(q1 == q2, "repeat", "the NVFP4 kernel twice on 3 jobs (1, 8, 3 columns)");
            run_tc(false, multi_warps, jobs); ck(cudaStreamSynchronize(s), "tc");
            ck(cudaMemcpy(t1.data(), d_y, n * 2, cudaMemcpyDeviceToHost), "y");
            if (sm89) {
                std::vector<std::uint16_t> q89(n), t89(n);
                run_q(true, multi_warps, jobs, cols); ck(cudaStreamSynchronize(s), "q89");
                ck(cudaMemcpy(q89.data(), d_y, n * 2, cudaMemcpyDeviceToHost), "y");
                run_tc(true, multi_warps, jobs); ck(cudaStreamSynchronize(s), "tc89");
                ck(cudaMemcpy(t89.data(), d_y, n * 2, cudaMemcpyDeviceToHost), "y");
                verdict(q89 == q1 && t89 == t1, "sm89", "both kernels from the sm_89 build against the sm_86 build's outputs");
            }
            // accuracy on job 0 (one column) and job 1 (eight columns)
            for (int job = 0; job < 2; ++job) {
                const int c0 = job == 0 ? 0 : 1, w = job == 0 ? 1 : 8;
                std::vector<double> xs_bf(std::size_t(w) * H), xs_a4(std::size_t(w) * H), y_bf, y_a4, v;
                for (int c = 0; c < w; ++c) {
                    const std::uint16_t* xc = xb.data() + std::size_t(c0 + c) * H;
                    for (int k = 0; k < H; ++k) xs_bf[std::size_t(c) * H + k] = mf::bf16_float(xc[k]);
                    a4_values(xc, H, v);
                    std::copy(v.begin(), v.end(), xs_a4.begin() + std::size_t(c) * H);
                }
                reference(keep[size_t(job)].data(), mult, xs_bf, w, false, y_bf);
                reference(keep[size_t(job)].data(), mult, xs_a4, w, true, y_a4);
                const std::size_t at = std::size_t(c0) * H, m = std::size_t(w) * H;
                char text[256];
                std::snprintf(text, sizeof text, "%d column%s: tensor-core vs its exact product %.2e; NVFP4 vs its exact product %.2e; "
                              "NVFP4 vs tensor-core %.3e; tensor-core vs the NVFP4 inputs' exact product %.3e", w, w > 1 ? "s" : "",
                              rel_rms(t1.data() + at, y_bf), rel_rms(q1.data() + at, y_a4), rel_rms_bf16(q1.data() + at, t1.data() + at, m),
                              rel_rms(t1.data() + at, y_a4));
                verdict(true, "accuracy", text);
            }
        }
        {   // kernel item 5b: the card's packing of BF16 columns, and variant 2 against variant 1
            std::uint8_t* d_p2 = nullptr;
            ck(cudaMalloc(&d_p2, std::size_t(kCols) * hq::kPackedRow), "packed on the card");
            ck(hq::pack_bf16(d_x, d_p2, kCols, s), "pack bf16"); ck(cudaStreamSynchronize(s), "pack bf16");
            std::vector<std::uint8_t> mine(std::size_t(kCols) * hq::kPackedRow);
            ck(cudaMemcpy(mine.data(), d_p2, mine.size(), cudaMemcpyDeviceToHost), "packed on the card back");
            verdict(mine == packed, "pack bf16", "BF16 columns packed on the card against the 5090's packed rows of the same F32 rows");
            ck(cudaMemcpy(d_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice), "packed up");
            long diffs = 0, runs = 0;
            const auto compare = [&](bool use89, int warps, const std::vector<FastJob>& jobs, int cols) {
                const std::size_t n = std::size_t(cols) * H;
                std::vector<std::uint16_t> want(n), got(n);
                ck(cudaMemcpyAsync(d_jobs, jobs.data(), jobs.size() * sizeof(FastJob), cudaMemcpyHostToDevice, s), "jobs up");
                auto* dj = static_cast<FastJob*>(d_jobs);
                ck(hq::expand_packed(d_packed, d_q, cols, s), "expand");
                ck(hq::ffn(warps, loads, dj, int(jobs.size()), d_x, d_q, d_hq, nullptr, s), "v1"); ck(cudaStreamSynchronize(s), "v1");
                ck(cudaMemcpy(want.data(), d_y, n * 2, cudaMemcpyDeviceToHost), "v1 y");
                for (int from_bf16 = 0; from_bf16 < 2; ++from_bf16) {
                    ck(cudaMemset(d_y, 0xff, n * 2), "poison y");
                    if (from_bf16) ck(use89 ? mm3::helper_sm89::q2_pack_bf16(d_x, d_p2, cols, s) : hq::pack_bf16(d_x, d_p2, cols, s), "pack");
                    const std::uint8_t* in = from_bf16 ? d_p2 : d_packed;
                    ck(use89 ? mm3::helper_sm89::q2_ffn(warps, loads, dj, int(jobs.size()), d_x, in, ws, s)
                             : hq::ffn2(warps, loads, dj, int(jobs.size()), d_x, in, ws, s), "v2");
                    ck(cudaStreamSynchronize(s), "v2");
                    ck(cudaMemcpy(got.data(), d_y, n * 2, cudaMemcpyDeviceToHost), "v2 y");
                    for (std::size_t i = 0; i < n; ++i) diffs += got[i] != want[i];
                    ++runs;
                }
            };
            std::vector<FastJob> jobs;
            int cols = make_jobs(jobs, {1, 8, 3}, 0);
            for (int w : {1, 2, 4, 8}) {
                compare(false, w, jobs, cols);
                if (sm89) compare(true, w, jobs, cols);
            }
            for (const std::vector<int>& widths : {std::vector<int>{1}, std::vector<int>{1, 1, 1, 1}, std::vector<int>{1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 8},
                                                   std::vector<int>(16, 1)}) {
                cols = make_jobs(jobs, widths, 3);
                const int w = cols == int(jobs.size()) ? one_warps : multi_warps;
                compare(false, w, jobs, cols);
                if (sm89) compare(true, w, jobs, cols);
            }
            verdict(diffs == 0, "variant 2", std::to_string(runs) + " runs (packed rows and BF16 columns, warps 1/2/4/8 and the serving warps" +
                    (sm89 ? std::string(", both builds") : std::string()) + ") against variant 1: " + std::to_string(diffs) + " differing outputs");
            cudaFree(d_p2);
        }
        // timing
        cudaEvent_t e0, e1;
        ck(cudaEventCreate(&e0), "event"); ck(cudaEventCreate(&e1), "event");
        const auto time = [&](auto&& launch) {
            for (int i = 0; i < 3; ++i) launch();
            ck(cudaEventRecord(e0, s), "e0");
            for (int i = 0; i < iters; ++i) launch();
            ck(cudaEventRecord(e1, s), "e1"); ck(cudaEventSynchronize(e1), "e1");
            float ms = 0; ck(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
            return 1000.0 * ms / iters;
        };
        std::uint8_t* h_in = nullptr;
        ck(cudaHostAlloc(&h_in, std::size_t(kCols) * H * 2, cudaHostAllocPortable), "pinned inputs");
        std::printf("\ntiming (us a pass, CUDA events, %d passes): tensor = today's kernel (BF16 rows); v1 packed = variant 1 as the engine "
                    "runs it with the 5090's rows (expanded on the card, then its kernels); v2 = variant 2 from the same packed rows; "
                    "v1 BF16 = variant 1 encoding BF16 rows on the card\n", iters);
        std::printf("%-30s %8s %10s %10s %10s %8s %10s", "pass", "columns", "tensor", "v1 packed", "v2", "v2/tensor", "v1 BF16");
        if (sm89) std::printf(" %10s %10s", "tensor 89", "v2 89");
        std::printf(" %12s %12s\n", "BF16 in, us", "packed in, us");
        struct Shape { const char* name; std::vector<int> widths; };
        const Shape shapes[] = {{"one row, 1 job", {1}}, {"one row, 2 jobs", {1, 1}}, {"one row, 3 jobs", {1, 1, 1}}, {"one row, 4 jobs", {1, 1, 1, 1}},
                                {"window, 4 jobs x 2", {2, 2, 2, 2}}, {"window, 8 jobs x 2", {2, 2, 2, 2, 2, 2, 2, 2}},
                                {"window, 6 jobs x 4", {4, 4, 4, 4, 4, 4}}, {"window, 12 jobs mixed", {1, 1, 1, 2, 2, 2, 2, 3, 3, 4, 5, 8}},
                                {"window, 16 jobs x 1", std::vector<int>(16, 1)}};
        for (const Shape& sh : shapes) {
            std::vector<FastJob> jobs;
            const int cols = make_jobs(jobs, sh.widths, 3);
            const bool one = cols == int(jobs.size());
            const int warps = one ? one_warps : multi_warps;
            ck(cudaMemcpy(d_jobs, jobs.data(), jobs.size() * sizeof(FastJob), cudaMemcpyHostToDevice), "jobs up");
            auto* dj = static_cast<FastJob*>(d_jobs);
            const double t_tc = time([&] { ck(tc_launch(false, warps, loads, dj, int(jobs.size()), ws, s), "tc"); });
            const double t_v1p = time([&] { ck(hq::expand_packed(d_packed, d_q, cols, s), "expand");
                                            ck(hq::ffn(warps, loads, dj, int(jobs.size()), d_x, d_q, d_hq, nullptr, s), "v1"); });
            const double t_v2 = time([&] { ck(hq::ffn2(warps, loads, dj, int(jobs.size()), d_x, d_packed, ws, s), "v2"); });
            const double t_q = time([&] { ck(hq::quant_bf16(d_x, d_q, cols, s), "quant"); ck(hq::ffn(warps, loads, dj, int(jobs.size()), d_x, d_q, d_hq, nullptr, s), "q"); });
            std::printf("%-30s %8d %10.1f %10.1f %10.1f %+7.1f%% %10.1f", sh.name, cols, t_tc, t_v1p, t_v2, 100.0 * (t_v2 / t_tc - 1.0), t_q);
            if (sm89) {
                const double t_tc89 = time([&] { ck(mm3::helper_sm89::tensor(warps, loads, dj, int(jobs.size()), ws, s), "tc89"); });
                const double t_v289 = time([&] { ck(mm3::helper_sm89::q2_ffn(warps, loads, dj, int(jobs.size()), d_x, d_packed, ws, s), "v2 89"); });
                std::printf(" %10.1f %10.1f", t_tc89, t_v289);
            }
            // the inputs across the link: BF16 columns against packed rows (pinned host memory to the card)
            const double t_bf = time([&] { ck(cudaMemcpyAsync(d_x, h_in, std::size_t(cols) * H * 2, cudaMemcpyHostToDevice, s), "bf16 in"); });
            const double t_pk = time([&] { ck(cudaMemcpyAsync(d_q2, h_in, std::size_t(cols) * hq::kPackedRow, cudaMemcpyHostToDevice, s), "packed in"); });
            std::printf(" %12.1f %12.1f\n", t_bf, t_pk);
            ck(cudaMemcpy(d_x, xb.data(), xb.size() * 2, cudaMemcpyHostToDevice), "x restore");
        }
        std::printf("\n%s\n", failures ? "helper_q_bench: FAIL" : "helper_q_bench: all checks PASS");
        return failures ? 1 : 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "helper_q_bench: %s\n", e.what());
        return 1;
    }
}
