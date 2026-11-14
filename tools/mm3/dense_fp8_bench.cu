// tools/mm3/dense_fp8_bench.cu - kernel item 3's standalone benchmark and error check: the dense MXFP8 matrices with
// 8-bit activations on the 5090's tensor cores (mm3_dense_fp8.cuh) against today's F32-activation kernels (mx_rows, as
// native_dense::mv calls them with dense_split 1), on one MoE layer's and one dense layer's real weights from the pack.
//
// Manual, on an idle RTX 5090 (the engine stopped: it needs about 4 GB of VRAM). No model is run.
//   mm3_dense_fp8_bench [--pack DIR] [--device N] [--layer L] [--reps N]
// Prints, all measured:
//   1. each matrix alone, rows 1-16: microseconds a call and the weight stream's GB/s, F32 kernel against 8-bit; the
//      weights rotate through copies of at least 256 MB, so they come from DRAM as in the engine (the 5090's L2 is 96 MB);
//   2. the MoE layer's projections in the engine's order (q k v | o | shared gate up | shared down, and the indexer's
//      q k on the side stream), each arm replayed as a CUDA graph as prep_graph does, by rows;
//   3. errors against an FP64 reference on the GPU: the 8-bit kernel against the exact product of the quantized
//      activations (the kernel's own check - a failure exits 1), and both kernels against the exact product of the F32
//      activations (the precision change itself), as relative RMS and max errors.
// The activations are deterministic synthetic rows (Gaussian, every 97th channel x12 as outliers), not model states.
#include "mm3_dense_fp8.cuh"
#include "mm3_kernels.cuh"
#include "mm3_native_dense.cuh"
#include "mm3_pack.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace nd = strata::mm3::native_dense;
namespace x3 = strata::mm3::exl3;
namespace f8 = mm3::dense_fp8;
// (native_dense.cu needs exl3::supported; this tool links only the native dense path, as native_dense_bench does)
namespace strata::mm3::exl3 {
bool supported(const Mat& m) { return m.fmt == nd::kFmtMXFP8 && m.raw && m.native_scale && m.k > 0 && m.k % 32 == 0 && m.n > 0; }
}

namespace {
void ck(cudaError_t e, const char* op) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(op) + ": " + cudaGetErrorString(e));
}
float e4m3(uint8_t c) {
    const int e = (c >> 3) & 15, m = c & 7;
    if (e == 15 && m == 7) return std::numeric_limits<float>::quiet_NaN();
    const float f = e ? std::ldexp(float(8 + m), e - 10) : std::ldexp(float(m), -9);
    return (c & 128) ? -f : f;
}

struct Mat {
    std::string name;
    int n = 0, k = 0;
    const mm3::Tensor *w = nullptr, *s = nullptr;
    int copies = 0;
    uint8_t *dw = nullptr, *ds = nullptr;   // copies back to back
    size_t wb() const { return (size_t) n * k; }
    size_t sb() const { return (size_t) n * (k / 32); }
    const uint8_t* W(int c) const { return dw + (size_t) c * wb(); }
    const uint8_t* S(int c) const { return ds + (size_t) c * sb(); }
};

// FP64 references: ref[r][n] = sum_k w(n,k) x(r,k), and the sum of |w x| (the scale of the kernel's rounding)
__global__ void ref_kernel(const uint8_t* w, const uint8_t* sc, int N, int K, const float* x, int rows, double* ref,
                           double* absref) {
    const int n = blockIdx.x * blockDim.y + threadIdx.y, r = blockIdx.y, lane = threadIdx.x;
    if (n >= N) return;
    double s = 0, a = 0;
    for (int k = lane; k < K; k += 32) {
        const uint8_t c = w[(size_t) n * K + k];
        const int e = (c >> 3) & 15, m = c & 7;
        double v = e ? ldexp(double(8 + m), e - 10) : ldexp(double(m), -9);
        if (c & 128) v = -v;
        v *= ldexp(1.0, int(sc[(size_t) n * (K / 32) + k / 32]) - 127);
        const double p = v * (double) x[(size_t) r * K + k];
        s += p;
        a += fabs(p);
    }
    for (int o = 16; o; o >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, o);
        a += __shfl_xor_sync(0xffffffffu, a, o);
    }
    if (lane == 0) {
        ref[(size_t) r * N + n] = s;
        absref[(size_t) r * N + n] = a;
    }
}

struct Timer {
    cudaEvent_t a = nullptr, b = nullptr;
    Timer() { ck(cudaEventCreate(&a), "event"); ck(cudaEventCreate(&b), "event"); }
    ~Timer() { cudaEventDestroy(a); cudaEventDestroy(b); }
};
}  // namespace

int main(int argc, char** argv) {
    std::string path = "/mnt/wsl/transfer/minimax-wsl/artifacts/proposal-2/native-pack";
    int device = 0, layer = 10, reps = 200;
    try {
        for (int i = 1; i < argc; ++i) {
            const std::string a = argv[i];
            if (i + 1 >= argc) throw std::invalid_argument("option needs a value: " + a);
            if (a == "--pack") path = argv[++i];
            else if (a == "--device") device = std::stoi(argv[++i]);
            else if (a == "--layer") layer = std::stoi(argv[++i]);
            else if (a == "--reps") reps = std::max(10, std::stoi(argv[++i]));
            else throw std::invalid_argument("unknown option: " + a);
        }
        ck(cudaSetDevice(device), "set device");
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, device), "device properties");
        if (!f8::supported()) throw std::runtime_error(std::string("the 8-bit kernel does not run on ") + prop.name);
        mm3::Pack pack;
        std::string err;
        if (!pack.load(path, err) || pack.str("format") != "native") throw std::runtime_error("native pack: " + err);
        const std::string L = "language_model.model.layers." + std::to_string(layer) + ".", L0 = "language_model.model.layers.0.";
        std::vector<Mat> mats = {
            {"q", 8192, 6144}, {"k", 512, 6144}, {"v", 512, 6144}, {"o", 6144, 8192}, {"index q", 512, 6144},
            {"index k", 128, 6144}, {"shared gate", 3072, 6144}, {"shared up", 3072, 6144}, {"shared down", 6144, 3072},
            {"dense gate (layer 0)", 12288, 6144}, {"dense up (layer 0)", 12288, 6144}, {"dense down (layer 0)", 6144, 12288}};
        const char* names[] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                               "self_attn.index_q_proj", "self_attn.index_k_proj",
                               "block_sparse_moe.shared_experts.gate_proj", "block_sparse_moe.shared_experts.up_proj",
                               "block_sparse_moe.shared_experts.down_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"};
        constexpr size_t kPool = 256ull << 20;
        size_t total = 0;
        for (size_t i = 0; i < mats.size(); ++i) {
            Mat& m = mats[i];
            const std::string t = (i >= 9 ? L0 : L) + names[i];
            m.w = pack.get(t + ".weight");
            m.s = pack.get(t + ".weight_scale_inv");
            if (!m.w || !m.s || m.w->dtype != "F8_E4M3" || m.w->bytes != m.wb() || m.s->bytes != m.sb())
                throw std::runtime_error("unexpected MXFP8 tensor: " + t);
            m.copies = (int) std::max<size_t>(2, (kPool + m.wb() - 1) / m.wb());
            ck(cudaMalloc(&m.dw, m.wb() * m.copies), "weights");
            ck(cudaMalloc(&m.ds, m.sb() * m.copies), "scales");
            for (int c = 0; c < m.copies; ++c) {
                ck(cudaMemcpy(m.dw + c * m.wb(), pack.data(*m.w), m.wb(), cudaMemcpyHostToDevice), "upload");
                ck(cudaMemcpy(m.ds + c * m.sb(), pack.data(*m.s), m.sb(), cudaMemcpyHostToDevice), "upload");
            }
            total += (m.wb() + m.sb()) * m.copies;
        }
        constexpr int R = 16, KX = 12288, NX = 12288;
        // activations: Gaussian (Box-Muller over a fixed LCG), every 97th channel x12
        std::vector<float> hx((size_t) R * KX);
        uint64_t st = 0x9e3779b97f4a7c15ull;
        auto u01 = [&]() { st = st * 6364136223846793005ull + 1442695040888963407ull; return ((st >> 11) + 0.5) * (1.0 / 9007199254740992.0); };
        for (size_t i = 0; i < hx.size(); ++i) {
            const double g = std::sqrt(-2.0 * std::log(u01())) * std::cos(6.283185307179586 * u01());
            hx[i] = float(g * ((i % KX) % 97 == 5 ? 12.0 : 1.0));
        }
        float *dx = nullptr, *dxq = nullptr, *dy = nullptr, *dy2 = nullptr, *part = nullptr;
        double *dref = nullptr, *dabs = nullptr, *dref2 = nullptr, *dabs2 = nullptr;
        ck(cudaMalloc(&dx, hx.size() * 4), "x");
        ck(cudaMalloc(&dxq, hx.size() * 4), "xq");
        ck(cudaMalloc(&dy, (size_t) R * NX * 4 * 8), "y");   // (8 outputs' room: the grouped jobs write side by side)
        ck(cudaMalloc(&dy2, (size_t) R * NX * 4), "y2");
        ck(cudaMalloc(&part, (size_t) 16 * 16 * 512 * 16 * 4), "partials");
        ck(cudaMalloc(&dref, (size_t) R * NX * 8), "ref");
        ck(cudaMalloc(&dabs, (size_t) R * NX * 8), "ref");
        ck(cudaMalloc(&dref2, (size_t) R * NX * 8), "ref");
        ck(cudaMalloc(&dabs2, (size_t) R * NX * 8), "ref");
        ck(cudaMemcpy(dx, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x upload");
        cudaStream_t s = nullptr;
        ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
        Timer tm;
        std::printf("dense_fp8_bench: %s, pack %s, MoE layer %d and dense layer 0; %.2f GB of weight copies; %d calls a "
                    "point; activations synthetic (Gaussian, outlier channels x12); all figures measured\n",
                    prop.name, path.c_str(), layer, total / 1e9, reps);

        // the F32 kernel as native_dense::mv calls it (dense_split 1: split-K partials for N <= 512)
        auto old_call = [&](const Mat& m, int c, const float* x, int rows, float* y) {
            mm3::k::mx_rows(m.W(c), m.S(c), m.n, m.k, x, rows, m.k, y, m.n, 0.0f, m.n <= nd::kSplitN ? part : nullptr, s);
        };
        auto new_call = [&](const Mat& m, int c, const float* x, int rows, float* y, int ks) {
            f8::Job j;
            j.w = m.W(c); j.sc = m.S(c); j.N = m.n; j.K = m.k; j.x = x; j.ldx = m.k; j.y = y; j.ldy = m.n;
            if (!f8::launch(&j, 1, rows, s, ks)) throw std::runtime_error("8-bit launch refused: " + m.name);
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

        // ---- 1. each matrix alone
        const int rowset[] = {1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16};
        std::printf("\n1. One matrix a launch, back to back (us a call; GB/s of weights and scales read)\n");
        std::printf("   %-22s %5s %10s %8s %10s %8s %7s   %s\n", "matrix (N x K)", "rows", "F32 us", "GB/s", "8-bit us", "GB/s",
                    "ratio", "8-bit us at 4 / 8 / 16 warps a block");
        for (const Mat& m : mats) {
            const double gb = (m.wb() + m.sb()) / 1e9;
            for (int rows : rowset) {
                const double to = time_it([&](int i) { old_call(m, i % m.copies, dx, rows, dy); }, reps);
                const double tn = time_it([&](int i) { new_call(m, i % m.copies, dx, rows, dy, 0); }, reps);
                std::printf("   %-22s %5d %10.2f %8.0f %10.2f %8.0f %7.2f  ", (m.name + " " + std::to_string(m.n) + "x" + std::to_string(m.k)).c_str(),
                            rows, to, gb / (to * 1e-6), tn, gb / (tn * 1e-6), to / tn);
                if (rows == 1 || rows == 8 || rows == 16)
                    for (int ks : {4, 8, 16})
                        std::printf(" %7.2f", time_it([&](int i) { new_call(m, i % m.copies, dx, rows, dy, ks); }, reps / 2));
                std::printf("\n");
            }
            ck(cudaGetLastError(), "launch");
        }

        // ---- 2. the MoE layer's projections in the engine's order, as graphs
        std::printf("\n2. The MoE layer's projections in the engine's order, each arm a CUDA graph (us a replay; the copies of the\n"
                    "   weights rotate between replays). F32: q, k, v, o, shared gate, up, down one launch each (k, v with their\n"
                    "   partials' reduce); 8-bit: [q k v] one launch, o, [gate up] one launch, down. Side stream: index q, k.\n");
        std::printf("   %5s %12s %12s %9s %14s %14s\n", "rows", "F32 main", "8-bit main", "saved", "F32 side", "8-bit side");
        const Mat &mq = mats[0], &mk = mats[1], &mv = mats[2], &mo = mats[3], &miq = mats[4], &mik = mats[5], &msg = mats[6],
                  &msu = mats[7], &msd = mats[8];
        const int sets = std::min({mq.copies, mk.copies, mo.copies, msg.copies, msd.copies, 6});
        float* yq = dy; float* yk = dy + (size_t) R * 8192; float* yv = yk + (size_t) R * 512; float* yo = yv + (size_t) R * 512;
        float* yg = yo + (size_t) R * 6144; float* yu = yg + (size_t) R * 3072; float* yd = yu + (size_t) R * 3072;
        float* yiq = yd + (size_t) R * 6144; float* yik = yiq + (size_t) R * 512;
        for (int rows : rowset) {
            double res[4];
            for (int arm = 0; arm < 4; ++arm) {   // 0 F32 main, 1 8-bit main, 2 F32 side, 3 8-bit side
                std::vector<cudaGraphExec_t> ex((size_t) sets);
                for (int c = 0; c < sets; ++c) {
                    cudaGraph_t gr = nullptr;
                    ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
                    if (arm == 0) {
                        old_call(mq, c, dx, rows, yq); old_call(mk, c, dx, rows, yk); old_call(mv, c, dx, rows, yv);
                        old_call(mo, c, dx, rows, yo); old_call(msg, c, dx, rows, yg); old_call(msu, c, dx, rows, yu);
                        old_call(msd, c, dx, rows, yd);
                    } else if (arm == 1) {
                        f8::Job j[3];
                        const Mat* g3[3] = {&mq, &mk, &mv};
                        float* y3[3] = {yq, yk, yv};
                        for (int i = 0; i < 3; ++i) { j[i].w = g3[i]->W(c); j[i].sc = g3[i]->S(c); j[i].N = g3[i]->n; j[i].K = g3[i]->k; j[i].x = dx; j[i].ldx = g3[i]->k; j[i].y = y3[i]; j[i].ldy = g3[i]->n; }
                        if (!f8::launch(j, 3, rows, s)) throw std::runtime_error("grouped launch refused");
                        new_call(mo, c, dx, rows, yo, 0);
                        const Mat* g2[2] = {&msg, &msu};
                        float* y2[2] = {yg, yu};
                        for (int i = 0; i < 2; ++i) { j[i].w = g2[i]->W(c); j[i].sc = g2[i]->S(c); j[i].N = g2[i]->n; j[i].K = g2[i]->k; j[i].x = dx; j[i].ldx = g2[i]->k; j[i].y = y2[i]; j[i].ldy = g2[i]->n; }
                        if (!f8::launch(j, 2, rows, s)) throw std::runtime_error("grouped launch refused");
                        new_call(msd, c, dx, rows, yd, 0);
                    } else if (arm == 2) {
                        old_call(miq, c % miq.copies, dx, rows, yiq); old_call(mik, c % mik.copies, dx, rows, yik);
                    } else {
                        f8::Job j[2];
                        const Mat* g2[2] = {&miq, &mik};
                        float* y2[2] = {yiq, yik};
                        for (int i = 0; i < 2; ++i) { j[i].w = g2[i]->W(c % g2[i]->copies); j[i].sc = g2[i]->S(c % g2[i]->copies); j[i].N = g2[i]->n; j[i].K = g2[i]->k; j[i].x = dx; j[i].ldx = g2[i]->k; j[i].y = y2[i]; j[i].ldy = g2[i]->n; }
                        if (!f8::launch(j, 2, rows, s)) throw std::runtime_error("grouped launch refused");
                    }
                    ck(cudaStreamEndCapture(s, &gr), "end capture");
                    ck(cudaGraphInstantiate(&ex[(size_t) c], gr, 0), "instantiate");
                    cudaGraphDestroy(gr);
                }
                res[arm] = time_it([&](int i) { ck(cudaGraphLaunch(ex[(size_t) (i % sets)], s), "graph launch"); }, reps);
                for (auto e : ex) cudaGraphExecDestroy(e);
            }
            std::printf("   %5d %12.2f %12.2f %9.2f %14.2f %14.2f\n", rows, res[0], res[1], res[0] - res[1], res[2], res[3]);
        }

        // ---- 3. errors against FP64 references (rows 16; each row is computed alone, so it holds for fewer rows)
        std::printf("\n3. Errors against FP64 references, 16 rows (relative RMS over all outputs; max relative to the output's RMS)\n");
        std::printf("   %-22s %18s %18s %18s %18s %14s\n", "matrix", "8-bit vs its own", "8-bit vs exact", "F32 vs exact",
                    "8-bit max vs exact", "rows 1 = 16?");
        bool good = true;
        for (const Mat& m : mats) {
            // the quantized activations as F32 values (exact: an E4M3 code times a power of two)
            std::vector<float> hq((size_t) R * m.k);
            for (int r = 0; r < R; ++r)
                for (int b = 0; b < m.k / 32; ++b) {
                    uint8_t codes[32];
                    const uint8_t e = f8::quantize_block_host(&hx[(size_t) r * KX + b * 32], codes);
                    for (int i = 0; i < 32; ++i) hq[(size_t) r * m.k + b * 32 + i] = e == 255 ? NAN : std::ldexp(e4m3(codes[i]), int(e) - 127);
                }
            std::vector<float> hxr((size_t) R * m.k);
            for (int r = 0; r < R; ++r) std::memcpy(&hxr[(size_t) r * m.k], &hx[(size_t) r * KX], (size_t) m.k * 4);
            float* dxr = dxq + 0;   // (rows packed at ld = K for the reference kernel)
            // (the stream is non-blocking: nothing on it may still read dxr when a synchronous copy rewrites it)
            ck(cudaStreamSynchronize(s), "sync");
            ck(cudaMemcpy(dxr, hq.data(), hq.size() * 4, cudaMemcpyHostToDevice), "xq upload");
            ref_kernel<<<dim3((m.n + 7) / 8, R), dim3(32, 8), 0, s>>>(m.W(0), m.S(0), m.n, m.k, dxr, R, dref, dabs);
            ck(cudaStreamSynchronize(s), "sync");
            ck(cudaMemcpy(dxr, hxr.data(), hxr.size() * 4, cudaMemcpyHostToDevice), "x upload");
            ref_kernel<<<dim3((m.n + 7) / 8, R), dim3(32, 8), 0, s>>>(m.W(0), m.S(0), m.n, m.k, dxr, R, dref2, dabs2);
            // the kernels on the same packed rows (ld = K)
            new_call(m, 0, dxr, R, dy, 0);
            old_call(m, 0, dxr, R, dy2);
            ck(cudaStreamSynchronize(s), "sync");
            const size_t no = (size_t) R * m.n;
            std::vector<float> yn(no), yo2(no), y1((size_t) m.n);
            std::vector<double> rq(no), aq(no), rx(no);
            ck(cudaMemcpy(yn.data(), dy, no * 4, cudaMemcpyDeviceToHost), "read");
            ck(cudaMemcpy(yo2.data(), dy2, no * 4, cudaMemcpyDeviceToHost), "read");
            ck(cudaMemcpy(rq.data(), dref, no * 8, cudaMemcpyDeviceToHost), "read");
            ck(cudaMemcpy(aq.data(), dabs, no * 8, cudaMemcpyDeviceToHost), "read");
            ck(cudaMemcpy(rx.data(), dref2, no * 8, cudaMemcpyDeviceToHost), "read");
            new_call(m, 0, dxr, 1, dy, 0);
            ck(cudaStreamSynchronize(s), "sync");
            ck(cudaMemcpy(y1.data(), dy, (size_t) m.n * 4, cudaMemcpyDeviceToHost), "read");
            double own = 0, en = 0, eo = 0, nx = 0, mx = 0;
            for (size_t i = 0; i < no; ++i) {
                own = std::max(own, std::fabs(yn[i] - rq[i]) / std::max(aq[i], 1e-30));
                en += (yn[i] - rx[i]) * (yn[i] - rx[i]);
                eo += (yo2[i] - rx[i]) * (yo2[i] - rx[i]);
                nx += rx[i] * rx[i];
            }
            const double rms = std::sqrt(nx / no);
            for (size_t i = 0; i < no; ++i) mx = std::max(mx, std::fabs(yn[i] - rx[i]) / std::max(rms, 1e-30));
            const bool same = std::memcmp(y1.data(), yn.data(), (size_t) m.n * 4) == 0;
            const bool pass = std::isfinite(own) && own < 1e-5;
            good &= pass;
            std::printf("   %-22s %18.3g %18.3g %18.3g %18.3g %14s%s\n", m.name.c_str(), own, std::sqrt(en / nx), std::sqrt(eo / nx),
                        mx, same ? "yes" : "no", pass ? "" : "   <- FAILED: the kernel is not the product of its quantized rows");
        }
        std::printf("\n   (\"8-bit vs its own\": max |y - exact(Q(x))| / sum |w Q(x)|, the kernel's FP32 accumulation alone - must be\n"
                    "    under 1e-5; \"vs exact\": against the F32 activations' exact product, the 8-bit activations' error)\n");
        for (Mat& m : mats) { cudaFree(m.dw); cudaFree(m.ds); }
        cudaFree(dx); cudaFree(dxq); cudaFree(dy); cudaFree(dy2); cudaFree(part);
        cudaFree(dref); cudaFree(dabs); cudaFree(dref2); cudaFree(dabs2);
        cudaStreamDestroy(s);
        std::printf("%s\n", good ? "PASS" : "FAIL");
        return good ? 0 : 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "dense_fp8_bench: %s\n", e.what());
        return 2;
    }
}
