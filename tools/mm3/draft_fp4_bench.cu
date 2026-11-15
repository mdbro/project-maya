// tools/mm3/draft_fp4_bench.cu - kernel item 6's standalone benchmark and error check: DFlash2's projections in the four
// arithmetics a request can pick (draft_proj): 0 MXFP8 weights x F32 activations (mx_rows, today's), 1 MXFP8 x MXFP8
// activations (mm3_dense_fp8.cuh), 2 NVFP4 x NVFP4 and 3 NVFP4 x two-term NVFP4 (mm3_draft_fp4.cuh), on the drafter's
// real weights from its two packs (pack_dflash2.py --weights mxfp8 / nvfp4).
//
// Manual, on an idle RTX 5090 (the engine stopped: it needs about 8 GB of VRAM). No model is run.
//   mm3_draft_fp4_bench [--mx DIR] [--nv DIR] [--src DIR] [--device N] [--reps N]
// Prints, all measured:
//   1. errors, layer 0's nine projections, 8 and 16 rows: (a) draft_fp4's quantize() against quantize_row_host (codes'
//      values, scale bytes and row scales must agree exactly); (b) each FP4 kernel against the FP64 product of its own
//      quantized operands (the kernel's own check: relative RMS and max error); (c) every arithmetic against the FP64
//      product of the BF16 checkpoint's weights and the F32 activations (--src: the precision each mode gives up) -
//      relative RMS error;
//   2. each matrix alone, rows 1, 2, 4, 8, 16: microseconds a call and the weight stream's GB/s (the FP4 arms with their
//      quantize() launch); the weights rotate through copies of at least 256 MB, so they come from DRAM as in the engine;
//   3. one proposal's projections through the five layers (8 rows: DFlash2's block) in the drafter's order and groups
//      (conv | q k v | o | conv | gate up | down), each mode launched one by one and replayed as a CUDA graph: ms a
//      proposal. Today's draft call spends about 1.7-1.8 ms here (drafter-test-20261114/ASSESSMENT.md, 3.2).
// Exit 0: every check passes; 1: an FP4 kernel or quantize() fails its check; 3: setup failed.
// The activations are deterministic synthetic rows (Gaussian, every 97th channel x12), not model states.
#include "mm3_dense_fp8.cuh"
#include "mm3_draft_fp4.cuh"
#include "mm3_kernels.cuh"
#include "mm3_pack.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace f4 = mm3::draft_fp4;
namespace f8 = mm3::dense_fp8;

namespace {
void ck(cudaError_t e, const char* op) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(e));
}
double e4m3d(uint8_t c) { return f4::e4m3_value(c); }

struct Proj {
    std::string name, path;   // (path: the checkpoint's tensor name after "layers.L.")
    int N = 0, K = 0;
    // MXFP8 as stored, NVFP4 row-major (host); device copies back to back (c8 / c4 of them)
    std::vector<uint8_t> hw8, hs8, hc4, hs4;
    float gs = 1.0f;
    uint8_t *w8 = nullptr, *s8 = nullptr, *w4 = nullptr, *s4 = nullptr;
    int c8 = 1, c4 = 1;
    size_t b8() const { return (size_t) N * K; }
    size_t bs8() const { return (size_t) N * (K / 32); }
    size_t b4() const { return f4::codes_bytes(N, K); }
    size_t bs4() const { return f4::scales_bytes(N, K); }
    f4::Mat mat(int c) const {
        f4::Mat m;
        m.w = w4 + (size_t) c * b4();
        m.sc = s4 + (size_t) c * bs4();
        m.gs = gs;
        m.N = N;
        m.K = K;
        return m;
    }
};

// a tensor of a safetensors file (BF16) as F32
std::vector<float> st_bf16(const std::string& file, const std::string& name, size_t want) {
    std::ifstream f(file, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + file);
    uint64_t n = 0;
    f.read((char*) &n, 8);
    std::string h(n, '\0');
    f.read(&h[0], (std::streamsize) n);
    const size_t at = h.find("\"" + name + "\"");
    if (at == std::string::npos) throw std::runtime_error(file + ": no tensor " + name);
    const size_t off = h.find("\"data_offsets\"", at), lb = h.find('[', off), cm = h.find(',', lb), rb = h.find(']', cm);
    const uint64_t a = std::stoull(h.substr(lb + 1, cm - lb - 1)), b = std::stoull(h.substr(cm + 1, rb - cm - 1));
    if (b - a != want * 2) throw std::runtime_error(name + ": unexpected size");
    std::vector<uint16_t> u(want);
    f.seekg((std::streamoff) (8 + n + a));
    f.read((char*) u.data(), (std::streamsize) (want * 2));
    std::vector<float> out(want);
    for (size_t i = 0; i < want; ++i) {
        const uint32_t v = (uint32_t) u[i] << 16;
        std::memcpy(&out[i], &v, 4);
    }
    return out;
}

// y[r][n] = sum_k W[n][k] x[r][k] in FP64 over threads (W given as a row function)
void ref_mm(int N, int K, int rows, const std::function<void(int, double*)>& wrow, const std::vector<double>& x,
            std::vector<double>& y) {
    y.assign((size_t) rows * N, 0.0);
    const int nt = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
    std::vector<std::thread> th;
    for (int t = 0; t < nt; ++t)
        th.emplace_back([&, t] {
            std::vector<double> w((size_t) K);
            for (int n = t; n < N; n += nt) {
                wrow(n, w.data());
                for (int r = 0; r < rows; ++r) {
                    double s = 0;
                    const double* xr = x.data() + (size_t) r * K;
                    for (int k = 0; k < K; ++k) s += w[(size_t) k] * xr[k];
                    y[(size_t) r * N + n] = s;
                }
            }
        });
    for (auto& t : th) t.join();
}

struct Err {
    double rms = 0, max = 0;
};
// relative RMS error of got against want over rows x N, and the largest |error| relative to want's RMS
Err rel(const std::vector<float>& got, const std::vector<double>& want) {
    double e = 0, r = 0, m = 0;
    for (size_t i = 0; i < want.size(); ++i) {
        const double d = (double) got[i] - want[i];
        e += d * d;
        r += want[i] * want[i];
        m = std::max(m, std::fabs(d));
    }
    Err out;
    out.rms = r > 0 ? std::sqrt(e / r) : 0;
    out.max = r > 0 ? m / std::sqrt(r / (double) want.size()) : 0;
    if (!std::isfinite(out.rms)) out.rms = out.max = INFINITY;
    return out;
}

struct Timer {
    cudaEvent_t a = nullptr, b = nullptr;
    Timer() { ck(cudaEventCreate(&a), "event"); ck(cudaEventCreate(&b), "event"); }
    ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
};
}  // namespace

int main(int argc, char** argv) {
    std::string mxdir = "/mnt/wsl/transfer/minimax-wsl/artifacts/dflash2-pack";
    std::string nvdir = "/mnt/wsl/transfer/minimax-wsl/artifacts/dflash2-pack-nvfp4";
    std::string src = "/mnt/wsl/transfer/minimax-wsl/artifacts/dflash2-src";
    int device = 0, reps = 200;
    int fails = 0;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("option needs a value: " + a);
            if (a == "--mx") mxdir = argv[++i];
            else if (a == "--nv") nvdir = argv[++i];
            else if (a == "--src") src = argv[++i];
            else if (a == "--device") device = std::stoi(argv[++i]);
            else if (a == "--reps") reps = std::max(10, std::stoi(argv[++i]));
            else throw std::invalid_argument("unknown option: " + a);
        }
        ck(cudaSetDevice(device), "set device");
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, device), "device properties");
        if (!f4::supported()) throw std::runtime_error(std::string("the FP4 kernels do not run on ") + prop.name);
        const bool f8ok = f8::supported();
        mm3::Pack mx, nv;
        std::string err;
        if (!mx.load(mxdir, err) || mx.str("dflash2.weights") != "mxfp8") throw std::runtime_error("MXFP8 pack: " + mxdir + " " + err);
        if (!nv.load(nvdir, err) || nv.str("dflash2.weights") != "nvfp4") throw std::runtime_error("NVFP4 pack: " + nvdir + " " + err);
        const int D = (int) mx.num("dflash2.hidden_size"), F = (int) mx.num("dflash2.intermediate_size");
        const int Q = (int) mx.num("dflash2.num_attention_heads") * (int) mx.num("dflash2.head_dim");
        const int KV = (int) mx.num("dflash2.num_key_value_heads") * (int) mx.num("dflash2.head_dim");
        const int NL = (int) mx.num("dflash2.num_hidden_layers"), G4 = 4 * (D / (int) mx.num("dflash2.conv_group_size", 16));
        const struct { const char* name; const char* path; int N, K; } geo[9] = {
            {"conv (attention)", "attention_conv.kernel_projection", G4, D}, {"q", "self_attn.q_proj", Q, D},
            {"k", "self_attn.k_proj", KV, D}, {"v", "self_attn.v_proj", KV, D}, {"o", "self_attn.o_proj", D, Q},
            {"conv (MLP)", "mlp_conv.kernel_projection", G4, D}, {"gate", "mlp.gate_proj", F, D},
            {"up", "mlp.up_proj", F, D}, {"down", "mlp.down_proj", D, F}};
        // every layer's projections, host and one device copy each (section 3); layer 0's rotated copies (section 2)
        std::vector<std::vector<Proj>> L((size_t) NL);
        size_t bytes8 = 0, bytes4 = 0;
        for (int l = 0; l < NL; ++l)
            for (const auto& gm : geo) {
                Proj p;
                p.name = gm.name;
                p.path = gm.path;
                p.N = gm.N;
                p.K = gm.K;
                const std::string t = "dflash2.layers." + std::to_string(l) + "." + gm.path;
                const mm3::Tensor *w8 = mx.get(t + ".weight"), *s8 = mx.get(t + ".weight_scale_inv");
                const mm3::Tensor *w4 = nv.get(t + ".weight"), *s4 = nv.get(t + ".weight_scale"), *g4 = nv.get(t + ".weight_scale_2");
                if (!w8 || !s8 || !w4 || !s4 || !g4 || w8->bytes != p.b8() || s8->bytes != p.bs8() || w4->bytes != p.b4() ||
                    s4->bytes != p.bs4() || g4->bytes != 4)
                    throw std::runtime_error("unexpected tensors for " + t);
                p.hw8.assign(mx.data(*w8), mx.data(*w8) + p.b8());
                p.hs8.assign(mx.data(*s8), mx.data(*s8) + p.bs8());
                p.hc4.assign(nv.data(*w4), nv.data(*w4) + p.b4());
                p.hs4.assign(nv.data(*s4), nv.data(*s4) + p.bs4());
                std::memcpy(&p.gs, nv.data(*g4), 4);
                constexpr size_t kPool = 256ull << 20;
                if (l == 0) {
                    p.c8 = (int) std::max<size_t>(2, (kPool + p.b8() - 1) / p.b8());
                    p.c4 = (int) std::max<size_t>(2, (kPool + p.b4() - 1) / p.b4());
                }
                ck(cudaMalloc(&p.w8, p.b8() * p.c8), "weights");
                ck(cudaMalloc(&p.s8, p.bs8() * p.c8), "scales");
                ck(cudaMalloc(&p.w4, p.b4() * p.c4), "weights");
                ck(cudaMalloc(&p.s4, p.bs4() * p.c4), "scales");
                std::vector<uint8_t> tw(p.b4()), ts(p.bs4());
                f4::tile_host(p.hc4.data(), p.hs4.data(), p.N, p.K, tw.data(), ts.data());
                for (int c = 0; c < p.c8; ++c) {
                    ck(cudaMemcpy(p.w8 + c * p.b8(), p.hw8.data(), p.b8(), cudaMemcpyHostToDevice), "upload");
                    ck(cudaMemcpy(p.s8 + c * p.bs8(), p.hs8.data(), p.bs8(), cudaMemcpyHostToDevice), "upload");
                }
                for (int c = 0; c < p.c4; ++c) {
                    ck(cudaMemcpy(p.w4 + c * p.b4(), tw.data(), p.b4(), cudaMemcpyHostToDevice), "upload");
                    ck(cudaMemcpy(p.s4 + c * p.bs4(), ts.data(), p.bs4(), cudaMemcpyHostToDevice), "upload");
                }
                bytes8 += p.b8() + p.bs8();
                bytes4 += p.b4() + p.bs4();
                if (l > 0) {   // (only layer 0's host bytes are needed after the upload)
                    std::vector<uint8_t>().swap(p.hw8);
                    std::vector<uint8_t>().swap(p.hs8);
                    std::vector<uint8_t>().swap(p.hc4);
                    std::vector<uint8_t>().swap(p.hs4);
                }
                L[(size_t) l].push_back(std::move(p));
            }
        constexpr int R = 16;
        const int KX = std::max(std::max(D, Q), F), NX = std::max(std::max(D, Q), F);
        std::vector<float> hx((size_t) R * KX);
        uint64_t st = 0x9e3779b97f4a7c15ull;
        auto u01 = [&]() { st = st * 6364136223846793005ull + 1442695040888963407ull; return ((st >> 11) + 0.5) * (1.0 / 9007199254740992.0); };
        for (size_t i = 0; i < hx.size(); ++i) {
            const double g = std::sqrt(-2.0 * std::log(u01())) * std::cos(6.283185307179586 * u01());
            hx[i] = float(g * ((i % KX) % 97 == 5 ? 12.0 : 1.0));
        }
        float *dx = nullptr, *dy = nullptr, *part = nullptr, *xgs = nullptr;
        uint8_t* xq = nullptr;
        ck(cudaMalloc(&dx, hx.size() * 4), "x");
        ck(cudaMalloc(&dy, (size_t) R * NX * 4 * 4), "y");   // (4 outputs' room: grouped jobs write side by side)
        ck(cudaMalloc(&part, (size_t) 16 * 16 * NX * 4), "partials");
        ck(cudaMalloc(&xq, f4::act_bytes(KX, 2)), "xq");
        ck(cudaMalloc(&xgs, 16 * 4), "xgs");
        ck(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x upload");
        ck(cudaDeviceSynchronize(), "uploads");
        cudaStream_t s = nullptr;
        ck(cudaStreamCreate(&s), "stream");
        Timer tm;
        std::printf("draft_fp4_bench: %s; MXFP8 %s (%.3f GB of projections), NVFP4 %s (%.3f GB tiled); %d calls a point; "
                    "activations synthetic (Gaussian, outlier channels x12); all figures measured\n",
                    prop.name, mxdir.c_str(), bytes8 / 1e9, nvdir.c_str(), bytes4 / 1e9, reps);
        if (!f8ok) std::printf("NOTE the 8-bit kernel (mode 1) does not run here: its rows are skipped\n");

        // a projection in mode m (one matrix, copy c) on x [rows][K]
        auto call = [&](int mode, const Proj& p, int c, const float* x, int rows, float* y) {
            if (mode == 0) {
                mm3::k::mx_rows(p.w8 + c * p.b8(), p.s8 + c * p.bs8(), p.N, p.K, x, rows, p.K, y, p.N, 0.0f, part, s);
            } else if (mode == 1) {
                f8::Job j;
                j.w = p.w8 + c * p.b8(); j.sc = p.s8 + c * p.bs8(); j.N = p.N; j.K = p.K; j.x = x; j.ldx = p.K; j.y = y; j.ldy = p.N;
                if (!f8::launch(&j, 1, rows, s)) throw std::runtime_error("8-bit launch refused: " + p.name);
            } else {
                const int terms = mode == 3 ? 2 : 1;
                if (!f4::quantize(x, rows, p.K, p.K, terms, xq, xgs, s)) throw std::runtime_error("quantize refused");
                f4::Job j;
                j.m = p.mat(c); j.xq = xq; j.xgs = xgs; j.y = y; j.ldy = p.N;
                if (!f4::launch(&j, 1, rows, terms, s)) throw std::runtime_error("FP4 launch refused: " + p.name);
            }
        };
        auto time_it = [&](auto&& body, int calls) {
            for (int i = 0; i < 10; ++i) body(i);
            ck(cudaEventRecord(tm.a, s), "record");
            for (int i = 0; i < calls; ++i) body(i);
            ck(cudaEventRecord(tm.b, s), "record");
            ck(cudaEventSynchronize(tm.b), "sync");
            float ms = 0;
            ck(cudaEventElapsedTime(&ms, tm.a, tm.b), "elapsed");
            return 1000.0 * ms / calls;
        };

        // ---- 1. errors, layer 0
        std::printf("\n1. errors, layer 0 (relative RMS / max error relative to the reference's RMS)\n");
        const std::string srcf = src + "/model.safetensors";
        const bool have_src = std::ifstream(srcf).good();
        if (!have_src) std::printf("NOTE no %s: the errors against the BF16 weights (c) are skipped\n", srcf.c_str());
        std::printf("%-17s %4s | %-21s %-21s | %-7s | %-9s %-9s %-9s %-9s\n", "matrix", "rows", "(b) w4a4 own operands",
                    "(b) w4a4x2 own", "(a) q", "(c) w8a32", "w8a8", "w4a4", "w4a4x2");
        for (const int rows : {8, 16}) {
            std::vector<double> x((size_t) rows * KX);
            for (const Proj& p : L[0]) {
                for (int r = 0; r < rows; ++r)
                    for (int k = 0; k < p.K; ++k) x[(size_t) r * p.K + k] = hx[(size_t) r * KX + k];
                std::vector<float> xr((size_t) rows * p.K);
                for (int r = 0; r < rows; ++r)
                    std::memcpy(xr.data() + (size_t) r * p.K, hx.data() + (size_t) r * KX, (size_t) p.K * 4);
                float* dxr = dx;   // (rows of K inputs: re-packed contiguously below)
                ck(cudaMemcpyAsync(dy + (size_t) 3 * R * NX, xr.data(), xr.size() * 4, cudaMemcpyHostToDevice, s), "x");
                dxr = dy + (size_t) 3 * R * NX;
                const auto w4row = [&](int n, double* w) {
                    for (int k = 0; k < p.K; ++k) {
                        const uint8_t c = p.hc4[(size_t) n * p.K / 2 + k / 2];
                        w[k] = f4::e2m1_value((k & 1) ? c >> 4 : c & 15u) * e4m3d(p.hs4[(size_t) n * p.K / 16 + k / 16]) * p.gs;
                    }
                };
                Err own[2];
                int qbad = 0;
                for (int terms = 1; terms <= 2; ++terms) {
                    // (a) quantize() against the host's, then (b) the kernel against its own operands in FP64
                    std::vector<double> xd((size_t) rows * p.K, 0.0);
                    std::vector<float> hgs((size_t) rows);
                    std::vector<uint8_t> hc((size_t) terms * p.K / 2), hs((size_t) terms * p.K / 16);
                    std::vector<uint8_t> dq(f4::act_bytes(p.K, terms));
                    std::vector<float> dg(16);
                    if (!f4::quantize(dxr, rows, p.K, p.K, terms, xq, xgs, s)) throw std::runtime_error("quantize refused");
                    ck(cudaMemcpyAsync(dq.data(), xq, dq.size(), cudaMemcpyDeviceToHost, s), "xq");
                    ck(cudaMemcpyAsync(dg.data(), xgs, 64, cudaMemcpyDeviceToHost, s), "xgs");
                    ck(cudaStreamSynchronize(s), "sync");
                    const int KC = p.K / 64;
                    for (int r = 0; r < rows; ++r) {
                        hgs[(size_t) r] = f4::quantize_row_host(xr.data() + (size_t) r * p.K, p.K, terms, hc.data(), hs.data());
                        if (dg[(size_t) r] != hgs[(size_t) r]) ++qbad;
                        for (int term = 0; term < terms; ++term)
                            for (int G = 0; G < p.K / 16; ++G) {
                                const int kc = G >> 2, j = G & 3;
                                if (dq[f4::x_scale(KC, term, kc, r, j)] != hs[(size_t) term * p.K / 16 + G]) ++qbad;
                                for (int h = 0; h < 2; ++h) {
                                    uint32_t wd, wh;
                                    std::memcpy(&wd, dq.data() + f4::x_word(KC, term, kc, r, 2 * j + h), 4);
                                    std::memcpy(&wh, hc.data() + (size_t) term * p.K / 2 + 8 * G + 4 * h, 4);
                                    for (int q = 0; q < 8; ++q)
                                        if (f4::e2m1_value((wd >> (4 * q)) & 15u) != f4::e2m1_value((wh >> (4 * q)) & 15u)) ++qbad;
                                }
                                for (int k = 0; k < 16; ++k) {
                                    const uint8_t c = hc[(size_t) term * p.K / 2 + 8 * G + k / 2];
                                    xd[(size_t) r * p.K + 16 * G + k] += f4::e2m1_value((k & 1) ? c >> 4 : c & 15u) *
                                        e4m3d(hs[(size_t) term * p.K / 16 + G]) * (double) hgs[(size_t) r];
                                }
                            }
                    }
                    std::vector<double> want;
                    ref_mm(p.N, p.K, rows, w4row, xd, want);
                    f4::Job j;
                    j.m = p.mat(0); j.xq = xq; j.xgs = xgs; j.y = dy; j.ldy = p.N;
                    if (!f4::launch(&j, 1, rows, terms, s)) throw std::runtime_error("FP4 launch refused");
                    std::vector<float> got((size_t) rows * p.N);
                    ck(cudaMemcpyAsync(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost, s), "y");
                    ck(cudaStreamSynchronize(s), "sync");
                    own[terms - 1] = rel(got, want);
                    if (!(own[terms - 1].rms < 1e-4 && own[terms - 1].max < 1e-3)) ++fails;
                }
                if (qbad) ++fails;
                // (c) every mode against the BF16 weights x F32 activations
                double cm[4] = {NAN, NAN, NAN, NAN};
                if (have_src) {
                    const std::vector<float> wb = st_bf16(srcf, "layers.0." + p.path + ".weight", (size_t) p.N * p.K);
                    std::vector<double> want;
                    ref_mm(p.N, p.K, rows, [&](int n, double* w) { for (int k = 0; k < p.K; ++k) w[k] = wb[(size_t) n * p.K + k]; }, x, want);
                    for (int mode = 0; mode < 4; ++mode) {
                        if (mode == 1 && !f8ok) continue;
                        call(mode, p, 0, dxr, rows, dy);
                        std::vector<float> got((size_t) rows * p.N);
                        ck(cudaMemcpyAsync(got.data(), dy, got.size() * 4, cudaMemcpyDeviceToHost, s), "y");
                        ck(cudaStreamSynchronize(s), "sync");
                        cm[mode] = rel(got, want).rms;
                    }
                }
                std::printf("%-17s %4d | %9.2e %9.2e   %9.2e %9.2e   | %-7s | %-9.4f %-9.4f %-9.4f %-9.4f\n", p.name.c_str(),
                            rows, own[0].rms, own[0].max, own[1].rms, own[1].max, qbad ? "DIFFERS" : "same", cm[0], cm[1],
                            cm[2], cm[3]);
                std::fflush(stdout);
            }
        }
        std::printf("pass: (a) same; (b) relative RMS < 1e-4 and max < 1e-3 (FP32 accumulation of exact products)\n");

        // ---- 2. each matrix alone
        std::printf("\n2. each matrix alone (layer 0): us a call (GB/s of its weights and scales); the FP4 arms include "
                    "quantize()\n%-17s %4s | %-17s %-17s %-17s %-17s\n", "matrix", "rows", "w8a32 (today)", "w8a8", "w4a4", "w4a4x2");
        for (const Proj& p : L[0])
            for (const int rows : {1, 2, 4, 8, 16}) {
                double us[4] = {NAN, NAN, NAN, NAN};
                for (int mode = 0; mode < 4; ++mode) {
                    if (mode == 1 && !f8ok) continue;
                    const int copies = mode <= 1 ? p.c8 : p.c4;
                    us[mode] = time_it([&](int i) { call(mode, p, i % copies, dx, rows, dy); }, reps);
                }
                const double g8 = (p.b8() + p.bs8()) / 1e3, g4 = (p.b4() + p.bs4()) / 1e3;
                std::printf("%-17s %4d | %7.1f (%6.0f)  %7.1f (%6.0f)  %7.1f (%6.0f)  %7.1f (%6.0f)\n", p.name.c_str(), rows,
                            us[0], g8 / us[0], us[1], g8 / us[1], us[2], g4 / us[2], us[3], g4 / us[3]);
                std::fflush(stdout);
            }
        ck(cudaGetLastError(), "section 2");

        // ---- 3. one proposal's projections, five layers, 8 rows, in the drafter's groups
        const int W = 8;
        float *xa = dx, *att = dx, *gt = dx;   // (inputs: the synthetic rows; their values do not change the time)
        float *y0 = dy, *y1 = dy + (size_t) R * NX, *y2 = dy + (size_t) 2 * R * NX;
        auto proposal = [&](int mode) {
            for (const auto& lp : L) {
                const Proj &ac = lp[0], &q = lp[1], &k = lp[2], &v = lp[3], &o = lp[4], &mc = lp[5], &g = lp[6], &u = lp[7], &d = lp[8];
                const auto group = [&](std::initializer_list<const Proj*> ps, const float* x, std::initializer_list<float*> ys) {
                    if (mode == 0) {
                        auto yi = ys.begin();
                        for (const Proj* p : ps) call(0, *p, 0, x, W, *yi++);
                        return;
                    }
                    if (mode == 1) {
                        f8::Job j[4];
                        int n = 0;
                        auto yi = ys.begin();
                        for (const Proj* p : ps) {
                            j[n].w = p->w8; j[n].sc = p->s8; j[n].N = p->N; j[n].K = p->K; j[n].x = x; j[n].ldx = p->K;
                            j[n].y = *yi++; j[n].ldy = p->N;
                            ++n;
                        }
                        if (!f8::launch(j, n, W, s)) throw std::runtime_error("8-bit group refused");
                        return;
                    }
                    const int terms = mode == 3 ? 2 : 1, K = (*ps.begin())->K;
                    if (!f4::quantize(x, W, K, K, terms, xq, xgs, s)) throw std::runtime_error("quantize refused");
                    f4::Job j[4];
                    int n = 0;
                    auto yi = ys.begin();
                    for (const Proj* p : ps) {
                        j[n].m = p->mat(0); j[n].xq = xq; j[n].xgs = xgs; j[n].y = *yi++; j[n].ldy = p->N;
                        ++n;
                    }
                    if (!f4::launch(j, n, W, terms, s)) throw std::runtime_error("FP4 group refused");
                };
                group({&ac}, xa, {y0});
                group({&q, &k, &v}, xa, {y0, y1, y2});
                group({&o}, att, {y0});
                group({&mc}, xa, {y0});
                group({&g, &u}, xa, {y0, y1});
                group({&d}, gt, {y0});
            }
        };
        std::printf("\n3. one proposal's projections: %d layers x (conv | q k v | o | conv | gate up | down), %d rows; ms a "
                    "proposal (GB/s of the weights read)\n%-22s %-17s %-17s\n", NL, W, "mode", "one by one", "as a graph");
        static const char* const mn[4] = {"0 w8a32 (today)", "1 w8a8", "2 w4a4", "3 w4a4x2"};
        for (int mode = 0; mode < 4; ++mode) {
            if (mode == 1 && !f8ok) continue;
            const double eager = time_it([&](int) { proposal(mode); }, std::max(10, reps / 10)) / 1000.0;
            cudaGraph_t gph = nullptr;
            cudaGraphExec_t gx = nullptr;
            ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
            proposal(mode);
            ck(cudaStreamEndCapture(s, &gph), "capture end");
            ck(cudaGraphInstantiate(&gx, gph, 0), "instantiate");
            const double graph = time_it([&](int) { ck(cudaGraphLaunch(gx, s), "graph"); }, std::max(10, reps / 10)) / 1000.0;
            cudaGraphExecDestroy(gx);
            cudaGraphDestroy(gph);
            const double gb = (mode <= 1 ? bytes8 : bytes4) / 1e6;
            std::printf("%-22s %6.3f (%5.0f)    %6.3f (%5.0f)\n", mn[mode], eager, gb / eager, graph, gb / graph);
            std::fflush(stdout);
        }
        ck(cudaDeviceSynchronize(), "end");
        std::printf("\n%s\n", fails ? "FAILED: an FP4 kernel or quantize() missed its check (section 1)" : "all checks passed");
    } catch (const std::exception& e) {
        std::fprintf(stderr, "draft_fp4_bench: %s\n", e.what());
        return 3;
    }
    return fails ? 1 : 0;
}
