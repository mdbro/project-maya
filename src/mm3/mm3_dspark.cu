// src/mm3/mm3_dspark.cu - maya-mm3's DSpark draft (mm3_dspark.hpp).
#include "mm3_dspark.hpp"
#include "mm3_draft_ring.hpp"

#include <functional>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "mm3_dense_fp8.cuh"
#include "mm3_draft_fp4.cuh"
#include "mm3_kernels.cuh"
#include "mm3_native_dense.cuh"

namespace mm3 {
namespace x3 = strata::mm3::exl3;
namespace f4 = ::mm3::draft_fp4;

#define DK(call)                                                                       \
    do {                                                                               \
        const cudaError_t e_ = (call);                                                 \
        if (e_ != cudaSuccess) {                                                       \
            err = std::string("dspark: ") + #call + ": " + cudaGetErrorString(e_);    \
            return false;                                                              \
        }                                                                              \
    } while (0)
#define DB(call)                                                                       \
    do {                                                                               \
        const cublasStatus_t b_ = (call);                                              \
        if (b_ != CUBLAS_STATUS_SUCCESS) {                                             \
            err = std::string("dspark: ") + #call + ": cublas " + std::to_string((int) b_);  \
            return false;                                                              \
        }                                                                              \
    } while (0)

namespace {
const std::string DP = "dspark.language_model.model.dspark.";
// MAYA_MM3_PROF=2: a proposal's parts timed (the host waits at each) and printed at the end
struct DraftProf {
    bool on = [] {
        const char* v = std::getenv("MAYA_MM3_PROF");
        return v != nullptr && std::atoi(v) == 2;
    }();
    double ms[4] = {};
    long n = 0;
    double last = 0;
    static double now() {
        return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    }
    void start(cudaStream_t s) {
        if (!on) return;
        cudaStreamSynchronize(s);
        last = now();
    }
    void mark(cudaStream_t s, int part) {
        if (!on) return;
        cudaStreamSynchronize(s);
        const double t = now();
        ms[part] += 1e3 * (t - last);
        last = t;
    }
    ~DraftProf() {
        if (on && n > 0)
            std::fprintf(stderr, "MM3 PROF draft: %ld proposals, layers %.2f head %.2f markov %.2f ms a proposal\n", n,
                         ms[0] / n, ms[1] / n, ms[2] / n);
    }
};
}

struct Draft::Impl {
    DraftGeo g;
    int dev = 0, ring = 0, max_rows = 0;
    DraftRing retained;
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    const x3::Mat* head = nullptr;
    const mm3::head_lowp::Head* head_lp = nullptr;   // (kernel item 10: set_head_lowp; nullptr: the BF16 head)
    // the head over n columns (hid rows ld apart) into n logit rows (V apart): the BF16 head through mv in runs of
    // mv_cols, or its low-precision copy
    bool head_pass(const float* hid, int ld, int n, float* logit, int V) {
        if (head_lp) return mm3::head_lowp::mv(*head_lp, hid, ld, n, logit, V, s);
        for (int c0 = 0; c0 < n;) {
            x3::MvJob j;
            j.m = head;
            j.xf = hid + (size_t) c0 * ld;
            j.y = logit + (size_t) c0 * V;
            j.ncol = std::min(std::max(1, x3::mv_cols(*head)), n - c0);
            j.xf_cs = (size_t) ld;
            j.y_cs = (size_t) V;
            if (!x3::mv(&j, 1, s)) return false;
            c0 += j.ncol;
        }
        return true;
    }
    const uint16_t* emb = nullptr;
    std::function<void(int, uint16_t*)> emb_row;   // (set_emb_rows: a GGUF's table, dequantized a row at a time)
    std::vector<void*> owned;
    // row id of the target's token table as BF16 into dst (D values)
    void row(int id, uint16_t* dst) const {
        if (emb_row) emb_row(id, dst);
        else std::memcpy(dst, emb + (size_t) id * g.dim, (size_t) g.dim * 2);
    }
    size_t bytes = 0;
    struct MX {   // a projection [N][K]: MXFP8 as stored (E4M3 weights, a UE8M0 scale every 32), and/or NVFP4 tiled
        const uint8_t *w = nullptr, *s = nullptr;
        int N = 0, K = 0;
        f4::Mat nv;   // (nv.w null: no NVFP4 copy)
    };
    struct L {
        MX q, k, v, o, g, u, d;
        float *in1 = nullptr, *post1 = nullptr, *qn1 = nullptr, *kn1 = nullptr;
        uint16_t *kc = nullptr, *vc = nullptr;
        // DFlash2: each branch's convolution - its kernel projection (MXFP8 [4G][D]) and base coefficients (F32 [2][2][D])
        MX ac, mc;
        float *abase = nullptr, *mbase = nullptr;
    };
    std::vector<L> L;
    uint16_t *fc = nullptr, *w1 = nullptr, *w2 = nullptr;
    // fc's GEMMs in FP16 instead of BF16 where the GPU has no BF16 tensor cores (CUDA before sm_80: cuBLAS took a
    // generic kernel there, ~1.2 ms a tap on a V100) - fc converted once at the load when every value fits FP16
    bool fc16 = false;
    float* apart = nullptr;   // window_attn's split partials (the block's queries)
    float* mkpart = nullptr;  // markov_greedy's blocks' argmax
    static constexpr int kSplits = 16;
    float *hid1 = nullptr, *fin1 = nullptr, *cw = nullptr, *cb = nullptr, *inv = nullptr;
    // scratch
    float *ctx = nullptr, *xn = nullptr, *h = nullptr, *q = nullptr, *kk = nullptr, *vv = nullptr, *att = nullptr;
    float *ao = nullptr, *gt = nullptr, *up = nullptr, *hid = nullptr, *logit = nullptr, *bias = nullptr;
    float *embf = nullptr, *conf = nullptr;
    float* mxpart = nullptr;   // mx_rows' split partials
    uint16_t *x16 = nullptr, *hid16 = nullptr, *embb = nullptr;
    int *pos = nullptr, *lo = nullptr, *prev = nullptr, *out = nullptr;
    int* host = nullptr;   // pinned: drafts then confidences
    uint16_t* anc = nullptr;   // (mapped) the anchor's embedding row, BF16
    float* maskf = nullptr;    // the mask token's embedding row, F32 (device)
    k::QRow* qh = nullptr;   // (mapped) a sampled proposal's distributions, a row a position
    // DFlash2: the selector's hidden projection [R][D] and codebooks [V][R] (BF16); scratch for the block's columns:
    // a branch's deltas [W][4G], its convolved input and output [W][D], the selector's projected hiddens [n][R]; the
    // head's top candidates per mask column and the draws (mapped)
    uint16_t *selw = nullptr, *pcb = nullptr, *scb = nullptr;
    float *dl = nullptr, *xp = nullptr, *zo = nullptr, *gsel = nullptr;
    k::CandRow* cand = nullptr;
    uint8_t* cand_scratch = nullptr;   // topk_rows' split scratch (device, owned): the top-16 over many blocks a column
    float* uu = nullptr;
    int dbg = 0;   // (DFlash2 diagnostic switches: set_debug)
    DraftProf prof;
    bool anc_busy = false;   // a proposal enqueued that may still read anc (one that failed before its own wait)
    // kernel item 6: the projections' arithmetic (set_proj: 0 MXFP8 x F32, 1 MXFP8 x MXFP8, 2 NVFP4 x NVFP4, 3 NVFP4 x
    // two-term NVFP4), its built-in, the weights held, and draft_fp4's quantized activations (one input at a time)
    int pmode = 0, pbuilt = 0;
    bool has8 = false, has4 = false;
    uint8_t* xq = nullptr;
    size_t xq_bytes = 0;   // per-Draft, stream-ordered scratch shared by projection modes 1/2/3
    float* xgs = nullptr;
    // set_graph: DFlash2's proposal region as a CUDA graph per draft count (gseen: uses before capture; gfail: a capture
    // failed - launched one by one for the rest of the load); pm: (mapped) the proposal's position, read by the graph;
    // hostm: (mapped) drafts then confidences, written by the graph form's copy kernel
    int gmode = 0;
    static constexpr int kMaxGraphs = 16;
    cudaGraphExec_t gx[kMaxGraphs] = {};
    int gseen[kMaxGraphs] = {};
    bool gfail = false;
    int* pm = nullptr;
    int* hostm = nullptr;
    // draft_graph 2's checks and draft_shadow's paired comparison (since reset_counts)
    long long g_replays = 0, g_checks = 0, g_mismatch = 0;
    int shadow = 0;
    long long sh_off[16] = {}, sh_agree[16] = {};

    void graphs_clear() {
        for (int i = 0; i < kMaxGraphs; ++i) {
            if (gx[i]) cudaGraphExecDestroy(gx[i]);
            gx[i] = nullptr;
            gseen[i] = 0;
        }
    }
    // what the captured kernels depend on besides the draft count: the projections' mode, dense_fp8's kernel (mode 1),
    // the dense path's settings the head pass reads (native_dense's multi-column form, its split) and which head copy it
    // reads (kernel item 10's head_lp) - graphs captured under another value are dropped at the next proposal
    long long gsig_at = -1;
    const void* glp_at = nullptr;
    long long gsig() const {
        namespace nd = strata::mm3::native_dense;
        return (long long) pmode | ((long long) ::mm3::dense_fp8::kernel() << 4) | ((long long) nd::mv_cols_on() << 8) |
               ((long long) ::mm3::head_lowp::shape_mode() << 9) |
               ((long long) nd::split_n() << 16);
    }

    ~Impl() {
        cudaSetDevice(dev);
        graphs_clear();
        for (void* x : owned) cudaFree(x);
        if (host) cudaFreeHost(host);
        if (qh) cudaFreeHost(qh);
        if (anc) cudaFreeHost(anc);
        if (cand) cudaFreeHost(cand);
        if (uu) cudaFreeHost(uu);
        if (pm) cudaFreeHost(pm);
        if (hostm) cudaFreeHost(hostm);
    }

    template <class T>
    bool alloc(T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaMalloc(&v, std::max<size_t>(n * sizeof(T), 16)) != cudaSuccess) {
            err = "dspark: out of device memory (" + std::to_string((n * sizeof(T)) >> 20) + " MB)";
            return false;
        }
        owned.push_back(v);
        bytes += n * sizeof(T);
        p = (T*) v;
        return true;
    }

    bool bf16(const Pack& pk, const std::string& name, uint16_t*& out, size_t want, std::string& err) {
        const Tensor* t = pk.get(name);
        if (!t || t->dtype != "BF16" || t->bytes != want * 2) {
            err = "dspark: " + name + ": not the BF16 tensor of " + std::to_string(want) + " values";
            return false;
        }
        if (!alloc(out, want, err)) return false;
        DK(cudaMemcpy(out, pk.data(*t), t->bytes, cudaMemcpyHostToDevice));
        return true;
    }

    // a float vector (BF16 stored) + add
    bool vec(const Pack& pk, const std::string& name, float add, float*& out, size_t want, std::string& err) {
        const Tensor* t = pk.get(name);
        if (!t || t->dtype != "BF16" || t->bytes != want * 2) {
            err = "dspark: " + name + ": not the BF16 vector of " + std::to_string(want);
            return false;
        }
        std::vector<float> hv(want);
        const uint16_t* b = (const uint16_t*) pk.data(*t);
        for (size_t i = 0; i < want; ++i) {
            const uint32_t u = (uint32_t) b[i] << 16;
            float f;
            std::memcpy(&f, &u, 4);
            hv[i] = f + add;
        }
        if (!alloc(out, want, err)) return false;
        DK(cudaMemcpy(out, hv.data(), want * 4, cudaMemcpyHostToDevice));
        return true;
    }

    // an MXFP8 matrix [N][K] on the device as it is stored (mx_rows reads it)
    bool mx(const Pack& pk, const std::string& pre, int N, int K, MX& m, std::string& err) {
        const Tensor* w = pk.get(pre + ".weight");
        const Tensor* sc = pk.get(pre + ".weight_scale_inv");
        if (!w || !sc || w->dtype != "F8_E4M3" || w->bytes != (uint64_t) N * K || sc->bytes != (uint64_t) N * (K / 32) ||
            N % 32 || K % 512) {
            err = "dspark: " + pre + ": not the MXFP8 matrix [" + std::to_string(N) + "][" + std::to_string(K) + "]";
            return false;
        }
        uint8_t *dw = nullptr, *ds = nullptr;
        if (!alloc(dw, w->bytes, err) || !alloc(ds, sc->bytes, err)) return false;
        DK(cudaMemcpy(dw, pk.data(*w), w->bytes, cudaMemcpyHostToDevice));
        DK(cudaMemcpy(ds, pk.data(*sc), sc->bytes, cudaMemcpyHostToDevice));
        m.w = dw;
        m.s = ds;
        m.N = N;
        m.K = K;
        return true;
    }
    // C [rows][N] F32 = X [rows][K] F32 . W^T (+ beta C), W MXFP8
    void mxg(const MX& m, const float* X, int rows, int ldx, float* C, int ldc, float beta) {
        k::mx_rows(m.w, m.s, m.N, m.K, X, rows, ldx, C, ldc, beta, mxpart, s);
    }

    // an NVFP4 matrix [N][K] from the pack's row-major NVFP4 (pack_dflash2.py --weights nvfp4: codes [N][K/2], E4M3
    // scales [N][K/16], the F32 global scale), retiled on the host for draft_fp4's kernels and put on the device
    bool mx4(const Pack& pk, const std::string& pre, int N, int K, MX& m, std::string& err) {
        const Tensor* w = pk.get(pre + ".weight");
        const Tensor* sc = pk.get(pre + ".weight_scale");
        const Tensor* g2 = pk.get(pre + ".weight_scale_2");
        if (!w || !sc || !g2 || w->dtype != "U8" || sc->dtype != "F8_E4M3" || g2->dtype != "F32" ||
            w->bytes != (uint64_t) N * K / 2 || sc->bytes != (uint64_t) N * (K / 16) || g2->bytes != 4 || N % 16 || K % 64) {
            err = "dspark: " + pre + ": not the NVFP4 matrix [" + std::to_string(N) + "][" + std::to_string(K) + "]";
            return false;
        }
        std::vector<uint8_t> tw(f4::codes_bytes(N, K)), ts(f4::scales_bytes(N, K));
        f4::tile_host(pk.data(*w), pk.data(*sc), N, K, tw.data(), ts.data());
        uint8_t *dw = nullptr, *ds = nullptr;
        if (!alloc(dw, tw.size(), err) || !alloc(ds, ts.size(), err)) return false;
        DK(cudaMemcpy(dw, tw.data(), tw.size(), cudaMemcpyHostToDevice));
        DK(cudaMemcpy(ds, ts.data(), ts.size(), cudaMemcpyHostToDevice));
        float gs = 0.0f;
        std::memcpy(&gs, pk.data(*g2), 4);
        if (!(gs > 0.0f) || !std::isfinite(gs)) {
            err = "dspark: " + pre + ": NVFP4 global scale " + std::to_string(gs);
            return false;
        }
        m.nv.w = dw;
        m.nv.sc = ds;
        m.nv.gs = gs;
        m.nv.N = N;
        m.nv.K = K;
        m.N = N;
        m.K = K;
        return true;
    }

    // one projection of a group that shares its input: C [rows][ldc] (+ beta C) = X . W^T
    struct PJ {
        const MX* m;
        float* C;
        int ldc;
        float beta;
    };
    // a group of projections of one input X [rows][ldx] (q / k / v, gate / up, or one alone) in the arithmetic set_proj
    // chose: mode 0 each through mx_rows as before (the same launches, the same bits); 1 one dense_fp8 launch a run of
    // 16 rows (mx_rows where it refuses); 2 / 3 X quantized once a run of 16 rows, then one draft_fp4 launch
    bool proj(std::initializer_list<PJ> js, const float* X, int rows, int ldx, std::string& err) {
        const int n = (int) js.size();
        if (pmode == 0) {
            for (const PJ& j : js) mxg(*j.m, X, rows, ldx, j.C, j.ldc, j.beta);
            return true;
        }
        for (int r0 = 0; r0 < rows; r0 += 16) {
            const int nr = std::min(16, rows - r0);
            const float* x = X + (size_t) r0 * ldx;
            if (pmode == 1) {
                ::mm3::dense_fp8::Job q[::mm3::dense_fp8::kMaxJobs];
                int i = 0;
                for (const PJ& j : js) {
                    q[i].w = j.m->w;
                    q[i].sc = j.m->s;
                    q[i].N = j.m->N;
                    q[i].K = j.m->K;
                    q[i].x = x;
                    q[i].ldx = ldx;
                    q[i].y = j.C + (size_t) r0 * j.ldc;
                    q[i].ldy = j.ldc;
                    q[i].beta = j.beta;
                    ++i;
                }
                if (!::mm3::dense_fp8::launch(q, n, nr, s, 0, -1, xq, xq_bytes)) {
                    err = "dspark: MXFP8 activation projection launch refused (rows " + std::to_string(nr) +
                          ", kernel " + std::to_string(::mm3::dense_fp8::kernel()) + ")";
                    return false;
                }
                continue;
            }
            const int terms = pmode == 3 ? 2 : 1;
            const int K = js.begin()->m->K;
            if (!f4::quantize(x, nr, ldx, K, terms, xq, xgs, s)) {
                err = "dspark: NVFP4 activations (rows " + std::to_string(nr) + ", K " + std::to_string(K) + ")";
                return false;
            }
            f4::Job q[f4::kMaxJobs];
            int i = 0;
            for (const PJ& j : js) {
                q[i].m = j.m->nv;
                q[i].xq = xq;
                q[i].xgs = xgs;
                q[i].y = j.C + (size_t) r0 * j.ldc;
                q[i].ldy = j.ldc;
                q[i].beta = j.beta;
                ++i;
            }
            if (!f4::launch(q, n, nr, terms, s)) {
                err = "dspark: NVFP4 projection launch (rows " + std::to_string(nr) + ")";
                return false;
            }
        }
        return true;
    }

    // (the expansion to BF16, kept for reference)
    bool mx_bf16(const Pack& pk, const std::string& pre, int N, int K, uint16_t*& out, std::string& err) {
        const Tensor* w = pk.get(pre + ".weight");
        const Tensor* sc = pk.get(pre + ".weight_scale_inv");
        if (!w || !sc || w->dtype != "F8_E4M3" || w->bytes != (uint64_t) N * K || sc->bytes != (uint64_t) N * (K / 32)) {
            err = "dspark: " + pre + ": not the MXFP8 matrix [" + std::to_string(N) + "][" + std::to_string(K) + "]";
            return false;
        }
        uint8_t *dw = nullptr, *ds = nullptr;
        DK(cudaMalloc(&dw, w->bytes));
        DK(cudaMalloc(&ds, sc->bytes));
        DK(cudaMemcpy(dw, pk.data(*w), w->bytes, cudaMemcpyHostToDevice));
        DK(cudaMemcpy(ds, pk.data(*sc), sc->bytes, cudaMemcpyHostToDevice));
        if (!alloc(out, (size_t) N * K, err)) return false;
        k::mxfp8_to_bf16(dw, ds, N, K, out, s);
        DK(cudaStreamSynchronize(s));
        cudaFree(dw);
        cudaFree(ds);
        return true;
    }

    // C [rows][N] F32 = X [rows][K] BF16 . W [N][K]^T BF16 (+ beta C) - both FP16 instead with f16
    bool gemm(const uint16_t* W, int N, int K, int ldw, const uint16_t* X, int rows, int ldx, float* C, int ldc,
              float beta, std::string& err, bool f16 = false) {
        const float one = 1.0f;
        const auto ty = f16 ? CUDA_R_16F : CUDA_R_16BF;   // (auto: the BLAS's own type name on HIP)
        DB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, N, rows, K, &one, W, ty, ldw, X, ty, ldx, &beta, C, CUDA_R_32F,
                        ldc, CUBLAS_COMPUTE_32F, f16 ? CUBLAS_GEMM_DEFAULT_TENSOR_OP : CUBLAS_GEMM_DEFAULT));
        return true;
    }

    bool load(const Pack& pk, int dev_, cudaStream_t s_, cublasHandle_t blas_, const x3::Mat* head_,
              const uint16_t* emb_, int max_rows_, std::string& err) {
        dev = dev_;
        s = s_;
        blas = blas_;
        head = head_;
        emb = emb_;
        max_rows = std::max(max_rows_, 8);
        g.dim = (int) pk.num("hidden_size");
        g.n_head = (int) pk.num("num_attention_heads");
        g.n_kv = (int) pk.num("num_key_value_heads");
        g.hd = (int) pk.num("head_dim");
        g.n_rot = (int) pk.num("rotary_dim", g.hd * pk.num("partial_rotary_factor", 1.0));
        g.vocab = (int) pk.num("vocab_size");
        g.eps = (float) pk.num("rms_norm_eps");
        g.theta = (float) pk.num("rope_theta");
        g.alpha = (float) pk.num("swiglu_alpha");
        g.limit = (float) pk.num("swiglu_limit");
        g.n_layers = (int) pk.num("dspark.num_hidden_layers");
        g.ff = (int) pk.num("dspark.dspark_ffn_hidden_size");
        g.rank = (int) pk.num("dspark.dspark_markov_rank");
        g.block = (int) pk.num("dspark.dspark_block_size");
        g.mask_id = (int) pk.num("dspark.dspark_noise_token_id");
        g.window = (int) pk.num("dspark.dflash.sliding_window", 4096);
        g.taps = pk.ints("dspark.dspark_target_layer_ids");
        if (g.n_layers <= 0 || g.ff <= 0 || g.rank <= 0 || g.block <= 0 || g.block > 16 || g.taps.empty() ||
            g.hd != 128 || g.n_head != 16 * g.n_kv) {
            err = "dspark: the pack has no DSpark draft this engine covers (tools/mm3/pack.py with dspark/ beside "
                  "the model)";
            return false;
        }
        if (pk.str("dspark.dflash.causal", "0") == "1") {
            err = "dspark: a causal draft is not covered";
            return false;
        }
        ring = g.window + 64;
        retained.reset(ring, g.window);
        cudaSetDevice(dev);
        const int D = g.dim, Q = g.n_head * g.hd, KV = g.n_kv * g.hd, F = g.ff, V = g.vocab, R = g.rank;
        const int NT = (int) g.taps.size();
        if (!bf16(pk, DP + "fc.weight", fc, (size_t) D * NT * D, err) ||
            !vec(pk, DP + "hidden_norm.weight", 1.0f, hid1, (size_t) D, err) ||
            !vec(pk, DP + "final_norm.weight", 1.0f, fin1, (size_t) D, err) ||
            !bf16(pk, DP + "markov_head.markov_w1.weight", w1, (size_t) V * R, err) ||
            !bf16(pk, DP + "markov_head.markov_w2.weight", w2, (size_t) V * R, err) ||
            !vec(pk, DP + "confidence_head.proj.weight", 0.0f, cw, (size_t) D + R, err) ||
            !vec(pk, DP + "confidence_head.proj.bias", 0.0f, cb, 1, err))
            return false;
#if !defined(STRATA_USE_HIP)
        {   // (fc in FP16 where BF16 has no tensor cores; MAYA_MM3_DSPARK_FC16=0 keeps BF16 everywhere)
            cudaDeviceProp pr{};
            const char* fv = std::getenv("MAYA_MM3_DSPARK_FC16");
            if ((fv == nullptr || std::atoi(fv) != 0) && cudaGetDeviceProperties(&pr, dev) == cudaSuccess &&
                pr.major < 8) {
                const size_t n = (size_t) D * NT * D;
                uint16_t* h = nullptr;
                if (cudaMalloc(&h, n * 2) == cudaSuccess) {
                    if (k::bf16_to_f16(fc, n, h, s)) {
                        for (void*& x : owned)
                            if (x == fc) x = h;
                        cudaFree(fc);
                        fc = h;
                        fc16 = true;
                    } else {
                        cudaFree(h);
                    }
                }
                cudaGetLastError();
            }
        }
#endif
        L.resize((size_t) g.n_layers);
        for (int i = 0; i < g.n_layers; ++i) {
            const std::string P = DP + "layers." + std::to_string(i) + ".decoder_layer.";
            auto& l = L[(size_t) i];
            if (!mx(pk, P + "self_attn.q_proj", Q, D, l.q, err) || !mx(pk, P + "self_attn.k_proj", KV, D, l.k, err) ||
                !mx(pk, P + "self_attn.v_proj", KV, D, l.v, err) || !mx(pk, P + "self_attn.o_proj", D, Q, l.o, err) ||
                !mx(pk, P + "mlp.gate_proj", F, D, l.g, err) || !mx(pk, P + "mlp.up_proj", F, D, l.u, err) ||
                !mx(pk, P + "mlp.down_proj", D, F, l.d, err) ||
                !vec(pk, P + "input_layernorm.weight", 1.0f, l.in1, (size_t) D, err) ||
                !vec(pk, P + "post_attention_layernorm.weight", 1.0f, l.post1, (size_t) D, err) ||
                !vec(pk, P + "self_attn.q_norm.weight", 1.0f, l.qn1, (size_t) g.hd, err) ||
                !vec(pk, P + "self_attn.k_norm.weight", 1.0f, l.kn1, (size_t) g.hd, err) ||
                !alloc(l.kc, (size_t) ring * KV, err) || !alloc(l.vc, (size_t) ring * KV, err))
                return false;
        }
        std::vector<float> inv_h((size_t) g.n_rot / 2);
        k::rope_inv_freq(g.theta, g.n_rot, inv_h.data());
        const int Rw = max_rows, B = g.block;
        if (!alloc(mxpart, (size_t) 16 * 16 * std::max(D, std::max(Q, F)), err)) return false;
        if (!alloc(apart, k::attn_part_floats(B, g.n_kv, kSplits), err) || !alloc(mkpart, k::markov_part_floats(), err))
            return false;
        if (!alloc(inv, inv_h.size(), err) || !alloc(ctx, (size_t) Rw * D, err) || !alloc(xn, (size_t) Rw * D, err) ||
            !alloc(h, (size_t) B * D, err) || !alloc(q, (size_t) B * Q, err) || !alloc(kk, (size_t) Rw * KV, err) ||
            !alloc(vv, (size_t) Rw * KV, err) || !alloc(att, (size_t) B * Q, err) || !alloc(ao, (size_t) B * D, err) ||
            !alloc(gt, (size_t) B * F, err) || !alloc(up, (size_t) B * F, err) || !alloc(hid, (size_t) B * D, err) ||
            !alloc(logit, (size_t) B * V, err) || !alloc(bias, (size_t) V, err) || !alloc(embf, (size_t) R, err) ||
            !alloc(conf, (size_t) B, err) || !alloc(x16, (size_t) Rw * std::max(std::max(D, Q), F), err) ||
            !alloc(hid16, (size_t) B * D, err) || !alloc(embb, (size_t) R, err) || !alloc(pos, (size_t) Rw, err) ||
            !alloc(lo, (size_t) B, err) || !alloc(prev, 1, err) || !alloc(out, (size_t) B, err))
            return false;
        DK(cudaMemcpy(inv, inv_h.data(), inv_h.size() * 4, cudaMemcpyHostToDevice));
        DK(cudaMallocHost(&host, (size_t) 2 * B * sizeof(int)));
        DK(cudaHostAlloc((void**) &qh, (size_t) B * sizeof(k::QRow), cudaHostAllocMapped | cudaHostAllocPortable));
        // (a proposal's inputs without the copy engine, which the expert cache's promotions keep busy: the anchor's
        //  row through mapped memory, the mask token's on the device, the positions made there)
        DK(cudaHostAlloc((void**) &anc, (size_t) D * 2, cudaHostAllocMapped | cudaHostAllocPortable));
        {
            std::vector<float> mf((size_t) D);
            std::vector<uint16_t> mrow((size_t) D);
            row(g.mask_id, mrow.data());
            const uint16_t* r = mrow.data();
            for (int i = 0; i < D; ++i) {
                const uint32_t u = (uint32_t) r[i] << 16;
                std::memcpy(&mf[(size_t) i], &u, 4);
            }
            if (!alloc(maskf, (size_t) D, err)) return false;
            DK(cudaMemcpy(maskf, mf.data(), (size_t) D * 4, cudaMemcpyHostToDevice));
        }
        has8 = true;
        pmode = pbuilt = 0;
        return true;
    }

    bool context(const float* taps, int T, int pos0, size_t stride, std::string& err) {
        const int NT = (int) g.taps.size();
        if (stride == 0) stride = (size_t) T;
        for (int a = 0; a < T; a += max_rows) {
            const int n = std::min(max_rows, T - a);
            for (int j = 0; j < NT; ++j)
                if (!context_tap(j, taps + ((size_t) j * stride + a) * g.dim, n, err)) return false;
            if (!context_finish(n, pos0 + a, err)) return false;
        }
        return true;
    }

    // ctx += tap_j . fc[:, j D:(j + 1) D]^T (= for j 0)
    bool context_tap(int j, const float* rows, int n, std::string& err) {
        const int D = g.dim, NT = (int) g.taps.size();
        cudaSetDevice(dev);
        if (fc16) k::to_f16(rows, n, D, D, x16, D, s);
        else k::to_bf16(rows, n, D, D, x16, D, s);
        return gemm(fc + (size_t) j * D, D, D, NT * D, x16, n, D, ctx, D, j == 0 ? 0.0f : 1.0f, err, fc16);
    }

    bool context_finish(int n, int pos0, std::string& err) {
        const int D = g.dim, KV = g.n_kv * g.hd;
        cudaSetDevice(dev);
        k::rms(ctx, hid1, n, D, g.eps, xn, nullptr, s);
        k::iota(pos, n, pos0, s);   // (on the device: an upload would wait behind the promotions' copies)
        for (auto& l : L) {
            if (!proj({{&l.k, kk, KV, 0.0f}, {&l.v, vv, KV, 0.0f}}, xn, n, D, err)) return false;
            k::qk_norm_rope(nullptr, kk, vv, nullptr, l.kn1, pos, n, 0, g.n_kv, g.hd, g.n_rot, inv, g.eps, l.kc,
                            l.vc, s, ring);
        }
        retained.context(pos0, n);
        return true;
    }

    // sa (given): each token drawn from the biased logits' distribution (sa, u[i] the draw of position i), qd its rows;
    // else greedy
    bool propose(int anchor, int P, int n, std::vector<int>& drafts, std::vector<float>& cf, const k::SampleArgs* sa,
                 const float* u, std::vector<std::vector<std::pair<int, float>>>* qd, std::string& err) {
        if (g.dflash2) return propose_d2(anchor, P, n, drafts, cf, sa, u, qd, err);
        const int D = g.dim, Q = g.n_head * g.hd, KV = g.n_kv * g.hd, F = g.ff, V = g.vocab, R = g.rank, B = g.block;
        n = std::max(1, std::min(n, B));
        if (!retained.available(P)) { err = "DSpark context is not retained at position " + std::to_string(P); return false; }
        // Every layer writes the whole block, including rows not returned.
        retained.proposal(P, B);
        cudaSetDevice(dev);
        // the block's embeddings: the anchor, then the mask token (the target's table, BF16 -> F32), and the
        // positions - made on the device (the anchor's row read where the host put it)
        // (the last proposal done with anc: it waited for its drafts at its end, unless it failed before that - no
        //  wait here otherwise, so the target's context rows queued before it go to the GPU with this proposal)
        if (anc_busy) DK(cudaStreamSynchronize(s));
        anc_busy = true;
        row(anchor, anc);
        prof.start(s);
        k::draft_block(anc, maskf, D, B, P, g.window, h, pos, lo, s);
        const float scale = 1.0f / std::sqrt((float) g.hd);
        for (auto& l : L) {
            k::rms(h, l.in1, B, D, g.eps, xn, nullptr, s);
            if (!proj({{&l.q, q, Q, 0.0f}, {&l.k, kk, KV, 0.0f}, {&l.v, vv, KV, 0.0f}}, xn, B, D, err)) return false;
            k::qk_norm_rope(q, kk, vv, l.qn1, l.kn1, pos, B, g.n_head, g.n_kv, g.hd, g.n_rot, inv, g.eps, l.kc, l.vc,
                            s, ring);
            k::window_attn(q, l.kc, l.vc, ring, lo, P + B - 1, B, g.n_head, g.n_kv, scale, att, s, apart, kSplits);
            if (!proj({{&l.o, h, D, 1.0f}}, att, B, Q, err)) return false;   // h += attn Wo
            k::rms(h, l.post1, B, D, g.eps, xn, nullptr, s);
            if (!proj({{&l.g, gt, F, 0.0f}, {&l.u, up, F, 0.0f}}, xn, B, D, err)) return false;
            k::swigluoai(gt, up, (size_t) B * F, g.alpha, g.limit, gt, nullptr, s);
            if (!proj({{&l.d, h, D, 1.0f}}, gt, B, F, err)) return false;   // h += mlp
        }
        k::rms(h, fin1, B, D, g.eps, hid, hid16, s);
        prof.mark(s, 0);
        // (the target's head on mv, F32 inputs: one pass for the block's rows - gemm_rows took ~1.8 ms on the 3090)
        if (!head_pass(hid, D, n, logit, V)) {
            err = "dspark: mv head";
            return false;
        }
        prof.mark(s, 1);
        k::set_int(prev, anchor, s);
        for (int i = 0; i < n; ++i) {
            // (greedy: the bias, the argmax and the confidence in two kernels - else the gathered row, the BLAS GEMV
            //  and markov_pick / markov_sample)
            if (!sa && k::markov_greedy(logit + (size_t) i * V, w2, w1, V, R, prev, out + i, cw, cb, hid + (size_t) i * D,
                                        D, conf + i, mkpart, nullptr, s))
                continue;
            k::gather_row(w1, R, prev, embb, embf, s);
            const float one = 1.0f, zero = 0.0f;
            DB(cublasGemmEx(blas, CUBLAS_OP_T, CUBLAS_OP_N, V, 1, R, &one, w2, CUDA_R_16BF, R, embb, CUDA_R_16BF, R,
                            &zero, bias, CUDA_R_32F, V, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
            if (sa) {
                k::SampleArgs a = *sa;
                a.u = u[i];
                k::markov_sample(logit + (size_t) i * V, bias, V, prev, out + i, cw, cb, hid + (size_t) i * D, D, embf,
                                 R, conf + i, a, qh + i, s);
            } else {
                k::markov_pick(logit + (size_t) i * V, bias, V, prev, out + i, nullptr, cw, cb, hid + (size_t) i * D,
                               D, embf, R, conf + i, s);
            }
        }
        DK(cudaMemcpyAsync(host, out, (size_t) n * 4, cudaMemcpyDeviceToHost, s));
        DK(cudaMemcpyAsync(host + B, conf, (size_t) n * 4, cudaMemcpyDeviceToHost, s));
        DK(cudaStreamSynchronize(s));
        prof.mark(s, 2);
        ++prof.n;
        anc_busy = false;
        drafts.assign(host, host + n);
        cf.resize((size_t) n);
        std::memcpy(cf.data(), host + B, (size_t) n * 4);
        if (sa && qd) {
            qd->assign((size_t) n, {});
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < qh[i].n; ++j) (*qd)[(size_t) i].push_back({qh[i].id[j], qh[i].q[j]});
        }
        return true;
    }

    // ---- DFlash2 (step 6): RadixArk's draft from its own pack (tools/mm3/pack_dflash2.py), the target's pack pk for the
    // checks. The context path (fc, the hidden norm, each layer's k / v into the ring) is DSpark's as it is.
    bool load_d2(const Pack& dp, const Pack& pk, int dev_, cudaStream_t s_, cublasHandle_t blas_, const x3::Mat* head_,
                 const uint16_t* emb_, int max_rows_, std::string& err) {
        dev = dev_;
        s = s_;
        blas = blas_;
        head = head_;
        emb = emb_;
        max_rows = std::max(max_rows_, 8);
        const std::string X = "dflash2.";
        g.dflash2 = true;
        g.dim = (int) dp.num(X + "hidden_size");
        g.n_head = (int) dp.num(X + "num_attention_heads");
        g.n_kv = (int) dp.num(X + "num_key_value_heads");
        g.hd = (int) dp.num(X + "head_dim");
        g.n_rot = (int) dp.num(X + "rotary_dim", g.hd);
        g.vocab = (int) dp.num(X + "vocab_size");
        g.eps = (float) dp.num(X + "rms_norm_eps", 1e-6);
        g.theta = (float) dp.num(X + "rope_theta", 1e7);
        g.alpha = (float) dp.num(X + "swiglu_alpha", 1.702);
        g.limit = (float) dp.num(X + "swiglu_limit", 7.0);
        g.n_layers = (int) dp.num(X + "num_hidden_layers");
        g.ff = (int) dp.num(X + "intermediate_size");
        g.rank = (int) dp.num(X + "selector_rank");
        g.sel_k = (int) dp.num(X + "selector_top_k", 16);
        g.block = (int) dp.num(X + "block_size") - 1;   // (the drafts: the block's mask columns after the anchor)
        g.mask_id = (int) dp.num(X + "mask_token_id");
        g.window = (int) dp.num(X + "sliding_window", 4096);
        g.taps = dp.ints(X + "target_layer_ids");
        const int cw = (int) dp.num(X + "conv_group_size", 16);
        g.conv_g = cw > 0 ? g.dim / cw : 0;
        if (g.dim != (int) pk.num("hidden_size") || g.vocab != (int) pk.num("vocab_size") || g.n_layers <= 0 ||
            g.ff <= 0 || g.rank <= 0 || g.rank > 4096 || g.sel_k < 1 || g.sel_k > 16 || g.block < 1 || g.block > 15 ||
            g.taps.empty() || g.hd != 128 || g.n_head != 16 * g.n_kv || cw != 16 || g.dim % cw ||
            (int) dp.num(X + "conv_kernel_size", 2) != 2 || dp.str(X + "use_gemma_norm") != "1" ||
            (dp.str(X + "weights") != "mxfp8" && dp.str(X + "weights") != "nvfp4")) {
            err = "dflash2: the pack is not a DFlash2 draft this engine covers for this target (tools/mm3/pack_dflash2.py)";
            return false;
        }
        const bool nv = dp.str(X + "weights") == "nvfp4";
        cudaSetDevice(dev_);
        if (nv && !f4::supported()) {
            err = "dflash2: an NVFP4 draft (" + dp.dir + ") needs the FP4 tensor cores of an RTX 50-series GPU (sm_120) for "
                  "its projections; use the MXFP8 pack (pack_dflash2.py --weights mxfp8)";
            return false;
        }
        ring = g.window + 64;
        retained.reset(ring, g.window);
        cudaSetDevice(dev);
        const int D = g.dim, Q = g.n_head * g.hd, KV = g.n_kv * g.hd, F = g.ff, V = g.vocab, R = g.rank;
        const int NT = (int) g.taps.size(), G4 = 4 * g.conv_g, W = g.block + 1;
        if (!bf16(dp, X + "fc.weight", fc, (size_t) D * NT * D, err) ||
            !vec(dp, X + "hidden_norm.weight", 1.0f, hid1, (size_t) D, err) ||
            !vec(dp, X + "norm.weight", 1.0f, fin1, (size_t) D, err) ||
            !bf16(dp, X + "candidate_selector.hidden_projection.weight", selw, (size_t) R * D, err) ||
            !bf16(dp, X + "candidate_selector.predecessor_codebook", pcb, (size_t) V * R, err) ||
            !bf16(dp, X + "candidate_selector.successor_codebook", scb, (size_t) V * R, err))
            return false;
        L.resize((size_t) g.n_layers);
        for (int i = 0; i < g.n_layers; ++i) {
            const std::string P = X + "layers." + std::to_string(i) + ".";
            auto& l = L[(size_t) i];
            if (!d2_projections(dp, nv, P, l, err) ||
                !vec(dp, P + "attention_conv.base_kernel", 0.0f, l.abase, (size_t) 4 * D, err) ||
                !vec(dp, P + "mlp_conv.base_kernel", 0.0f, l.mbase, (size_t) 4 * D, err) ||
                !vec(dp, P + "input_layernorm.weight", 1.0f, l.in1, (size_t) D, err) ||
                !vec(dp, P + "post_attention_layernorm.weight", 1.0f, l.post1, (size_t) D, err) ||
                !vec(dp, P + "self_attn.q_norm.weight", 1.0f, l.qn1, (size_t) g.hd, err) ||
                !vec(dp, P + "self_attn.k_norm.weight", 1.0f, l.kn1, (size_t) g.hd, err) ||
                !alloc(l.kc, (size_t) ring * KV, err) || !alloc(l.vc, (size_t) ring * KV, err))
                return false;
        }
        std::vector<float> inv_h((size_t) g.n_rot / 2);
        k::rope_inv_freq(g.theta, g.n_rot, inv_h.data());
        const int Rw = max_rows, n = g.block;
        if (!alloc(mxpart, (size_t) 16 * 16 * std::max(std::max(D, G4), std::max(Q, F)), err) ||
            !alloc(apart, k::attn_part_floats(W, g.n_kv, kSplits), err) || !alloc(inv, inv_h.size(), err) ||
            !alloc(ctx, (size_t) Rw * D, err) || !alloc(xn, (size_t) std::max(Rw, W) * D, err) ||
            !alloc(h, (size_t) W * D, err) || !alloc(q, (size_t) W * Q, err) ||
            !alloc(kk, (size_t) std::max(Rw, W) * KV, err) || !alloc(vv, (size_t) std::max(Rw, W) * KV, err) ||
            !alloc(att, (size_t) W * Q, err) || !alloc(gt, (size_t) W * F, err) || !alloc(up, (size_t) W * F, err) ||
            !alloc(hid, (size_t) W * D, err) || !alloc(hid16, (size_t) W * D, err) || !alloc(logit, (size_t) n * V, err) ||
            !alloc(conf, (size_t) n, err) || !alloc(x16, (size_t) Rw * std::max(std::max(D, Q), F), err) ||
            !alloc(pos, (size_t) std::max(Rw, W), err) || !alloc(lo, (size_t) W, err) || !alloc(out, (size_t) n, err) ||
            !alloc(dl, (size_t) W * G4, err) || !alloc(xp, (size_t) W * D, err) || !alloc(zo, (size_t) W * D, err) ||
            !alloc(gsel, (size_t) n * R, err) || !alloc(cand_scratch, k::topk_rows_scratch(n), err))
            return false;
        DK(cudaMemcpy(inv, inv_h.data(), inv_h.size() * 4, cudaMemcpyHostToDevice));
        DK(cudaMallocHost(&host, (size_t) 2 * W * sizeof(int)));
        DK(cudaHostAlloc((void**) &qh, (size_t) n * sizeof(k::QRow), cudaHostAllocMapped | cudaHostAllocPortable));
        DK(cudaHostAlloc((void**) &cand, (size_t) n * sizeof(k::CandRow), cudaHostAllocMapped | cudaHostAllocPortable));
        DK(cudaHostAlloc((void**) &uu, (size_t) n * sizeof(float), cudaHostAllocMapped | cudaHostAllocPortable));
        DK(cudaHostAlloc((void**) &anc, (size_t) D * 2, cudaHostAllocMapped | cudaHostAllocPortable));
        {
            std::vector<float> mf((size_t) D);
            std::vector<uint16_t> mrow((size_t) D);
            row(g.mask_id, mrow.data());
            for (int i = 0; i < D; ++i) {
                const uint32_t v = (uint32_t) mrow[(size_t) i] << 16;
                std::memcpy(&mf[(size_t) i], &v, 4);
            }
            if (!alloc(maskf, (size_t) D, err)) return false;
            DK(cudaMemcpy(maskf, mf.data(), (size_t) D * 4, cudaMemcpyHostToDevice));
        }
        (nv ? has4 : has8) = true;
        pmode = pbuilt = nv ? 3 : 0;   // (NVFP4's built-in: two activation terms - the same weight stream, 1% activation error)
        if (nv && !f4_scratch(err)) return false;
        DK(cudaHostAlloc((void**) &pm, 64, cudaHostAllocMapped | cudaHostAllocPortable));
        DK(cudaHostAlloc((void**) &hostm, (size_t) 2 * W * sizeof(int), cudaHostAllocMapped | cudaHostAllocPortable));
        return true;
    }

    // a layer's nine projections in one format (nv: NVFP4, else MXFP8) from pack dp
    bool d2_projections(const Pack& dp, bool nv, const std::string& P, decltype(L)::value_type& l, std::string& err) {
        const int D = g.dim, Q = g.n_head * g.hd, KV = g.n_kv * g.hd, F = g.ff, G4 = 4 * g.conv_g;
        const auto m = [&](const std::string& name, int N, int K, MX& x) {
            return nv ? mx4(dp, P + name, N, K, x, err) : mx(dp, P + name, N, K, x, err);
        };
        return m("self_attn.q_proj", Q, D, l.q) && m("self_attn.k_proj", KV, D, l.k) && m("self_attn.v_proj", KV, D, l.v) &&
               m("self_attn.o_proj", D, Q, l.o) && m("mlp.gate_proj", F, D, l.g) && m("mlp.up_proj", F, D, l.u) &&
               m("mlp.down_proj", D, F, l.d) && m("attention_conv.kernel_projection", G4, D, l.ac) &&
               m("mlp_conv.kernel_projection", G4, D, l.mc);
    }

    // Per-Draft, allocated before capture: one input of the widest K, shared in
    // stream order by NVFP4 projections and MXFP8 ID 3. A projection group uses
    // one (x,K,ldx), and its products finish before the next group repacks it.
    // This buffer is independent of the target's native_dense stream contexts.
    bool f4_scratch(std::string& err) {
        const int Kmax = std::max(std::max(g.dim, g.n_head * g.hd), g.ff);
        const size_t mx = size_t(::mm3::dense_fp8::kMaxRows) * ((size_t(Kmax) + Kmax/32 + 15) & ~size_t(15));
        if (!xq) {
            const size_t bytes_needed = std::max(mx, f4::act_bytes(Kmax, f4::kMaxTerms));
            if (!alloc(xq, bytes_needed, err)) return false;
            xq_bytes = bytes_needed;
        }
        return xgs || alloc(xgs, (size_t) f4::kMaxRows, err);
    }

    // (load_alt) the other format's projections from a second DFlash2 pack of the same geometry
    bool load_alt(const Pack& ap, std::string& err) {
        const std::string X = "dflash2.";
        if (!g.dflash2 || L.empty()) {
            err = "dflash2: an alternative pack needs a loaded DFlash2 draft";
            return false;
        }
        const std::string fmt = ap.str(X + "weights");
        const bool nv = fmt == "nvfp4";
        if ((fmt != "mxfp8" && fmt != "nvfp4") || (nv ? has4 : has8)) {
            err = "dflash2: the alternative pack " + ap.dir + " must hold the other weight format (" +
                  std::string(has4 ? "mxfp8" : "nvfp4") + ")";
            return false;
        }
        if ((int) ap.num(X + "hidden_size") != g.dim || (int) ap.num(X + "num_hidden_layers") != g.n_layers ||
            (int) ap.num(X + "intermediate_size") != g.ff || (int) ap.num(X + "num_attention_heads") != g.n_head ||
            (int) ap.num(X + "num_key_value_heads") != g.n_kv || (int) ap.num(X + "head_dim") != g.hd ||
            (int) ap.num(X + "conv_group_size", 16) != g.dim / std::max(1, g.conv_g)) {
            err = "dflash2: the alternative pack " + ap.dir + " is not the same draft's geometry";
            return false;
        }
        cudaSetDevice(dev);
        if (nv && !f4::supported()) {
            err = "dflash2: NVFP4 projections need an RTX 50-series GPU (sm_120)";
            return false;
        }
        for (int i = 0; i < g.n_layers; ++i)
            if (!d2_projections(ap, nv, X + "layers." + std::to_string(i) + ".", L[(size_t) i], err)) return false;
        (nv ? has4 : has8) = true;
        return !nv || f4_scratch(err);
    }

    // a DFlash2 proposal's region from the block to the top-k (the graph form replays it): the block [anchor, mask x B]
    // at positions P .. P + B through the layers (each branch: the norm, its convolution's input side, the projections,
    // its output side added to the residual), the final norm of the mask columns, the target's head over the n of them
    // in one pass and each column's top sel_k into cand. dyn: P read from pm (mapped) and the attention's last key from
    // pos[W - 1] - nothing that changes between calls is a kernel argument; else P as arguments (today's launches).
    bool d2_region(int P, int n, bool dyn, std::string& err) {
        const int D = g.dim, Q = g.n_head * g.hd, KV = g.n_kv * g.hd, F = g.ff, V = g.vocab;
        const int W = g.block + 1, G = g.conv_g, G4 = 4 * G;
        if (dyn) k::draft_block(anc, maskf, D, W, pm, g.window, h, pos, lo, s);
        else k::draft_block(anc, maskf, D, W, P, g.window, h, pos, lo, s);
        const int* hid_last = dyn ? pos + (W - 1) : nullptr;
        const float scale = 1.0f / std::sqrt((float) g.hd);
        for (auto& l : L) {
            k::rms(h, l.in1, W, D, g.eps, xn, nullptr, s);
            const bool conv = !(dbg & 1);
            if (conv) {   // the attention branch's convolution deltas [W][2][2][G]
                if (!proj({{&l.ac, dl, G4, 0.0f}}, xn, W, D, err)) return false;
                k::dflash2_conv_prepare(xn, dl, l.abase, W, D, G, xp, s);
            }
            const float* xa = conv ? xp : xn;
            if (!proj({{&l.q, q, Q, 0.0f}, {&l.k, kk, KV, 0.0f}, {&l.v, vv, KV, 0.0f}}, xa, W, D, err)) return false;
            k::qk_norm_rope(q, kk, vv, l.qn1, l.kn1, pos, W, g.n_head, g.n_kv, g.hd, g.n_rot, inv, g.eps, l.kc, l.vc,
                            s, ring);
            k::window_attn(q, l.kc, l.vc, ring, lo, P + W - 1, W, g.n_head, g.n_kv, scale, att, s, apart, kSplits,
                           hid_last);
            if (!proj({{&l.o, zo, D, 0.0f}}, att, W, Q, err)) return false;   // z = attn Wo, then the output side adds it
            if (conv) k::dflash2_conv_finish(zo, dl, l.abase, W, D, G, h, s);
            else k::add(h, zo, (size_t) W * D, s);
            k::rms(h, l.post1, W, D, g.eps, xn, nullptr, s);
            if (conv) {   // the MLP branch's
                if (!proj({{&l.mc, dl, G4, 0.0f}}, xn, W, D, err)) return false;
                k::dflash2_conv_prepare(xn, dl, l.mbase, W, D, G, xp, s);
            }
            const float* xm = conv ? xp : xn;
            if (!proj({{&l.g, gt, F, 0.0f}, {&l.u, up, F, 0.0f}}, xm, W, D, err)) return false;
            k::swigluoai(gt, up, (size_t) W * F, g.alpha, g.limit, gt, nullptr, s);
            if (!proj({{&l.d, zo, D, 0.0f}}, gt, W, F, err)) return false;
            if (conv) k::dflash2_conv_finish(zo, dl, l.mbase, W, D, G, h, s);
            else k::add(h, zo, (size_t) W * D, s);
        }
        k::rms(h, fin1, W, D, g.eps, hid, hid16, s);   // (rows 1 .. W - 1: the mask columns)
        prof.mark(s, 0);
        // the target's head over the n mask columns through mv, F32 inputs: the target's own head pass, which reads each
        // weight once for all the columns (a BF16 head: bf16_mv_cols). (rms's 16-bit output is FP16, not BF16: the first
        // form fed it to a BF16 GEMM and drafted noise - 2026-11-09's bisection)
        if (!head_pass(hid + D, D, n, logit, V)) { err = "dflash2: mv head"; return false; }
        k::topk_rows(logit, n, V, g.sel_k, nullptr, 0, cand, cand_scratch, s);
        return true;
    }

    // the graph form's region for n drafts, captured once (on its second use: the first runs the launches, so every
    // kernel's lazy setup is done outside a capture); a capture that fails turns the graph form off for the load
    void d2_capture(int P, int n) {
        cudaGraph_t gph = nullptr;
        if (cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal) != cudaSuccess) {
            cudaGetLastError();
            gfail = true;
            std::fprintf(stderr, "dflash2: the draft graph's capture did not begin: launched one by one\n");
            return;
        }
        std::string e2;
        const bool ok = d2_region(P, n, true, e2);
        const cudaError_t ce = cudaStreamEndCapture(s, &gph);
        cudaGraphExec_t ex = nullptr;
        if (ok && ce == cudaSuccess && gph != nullptr && cudaGraphInstantiate(&ex, gph, 0) == cudaSuccess) {
            gx[n] = ex;
        } else {
            gfail = true;
            std::fprintf(stderr, "dflash2: the draft graph (%d drafts) did not capture (%s%s): launched one by one\n", n,
                         ce != cudaSuccess ? cudaGetErrorString(ce) : "", e2.c_str());
        }
        cudaGetLastError();
        if (gph) cudaGraphDestroy(gph);
    }

    // a DFlash2 proposal: d2_region (as a graph with set_graph 1), the selector's projection and its walk (greedy, or
    // drawn with u when sa is given: q then the walk's distributions)
    bool propose_d2(int anchor, int P, int n, std::vector<int>& drafts, std::vector<float>& cf, const k::SampleArgs* sa,
                    const float* u, std::vector<std::vector<std::pair<int, float>>>* qd, std::string& err) {
        const int D = g.dim, R = g.rank;
        const int W = g.block + 1;
        n = std::max(1, std::min(n, g.block));
        if (!retained.available(P)) { err = "DFlash2 context is not retained at position " + std::to_string(P); return false; }
        retained.proposal(P, W);   // (every layer writes the whole block's K / V)
        cudaSetDevice(dev);
        if (anc_busy) DK(cudaStreamSynchronize(s));
        anc_busy = true;
        row(anchor, anc);
        if (sa)
            for (int i = 0; i < n; ++i) uu[i] = u[i];   // (the walk reads its draws through mapped memory)
        prof.start(s);
        const bool gr = gmode >= 1 && !gfail && dbg == 0 && !prof.on && n < kMaxGraphs;
        if (gr) {
            if (gsig() != gsig_at || (const void*) head_lp != glp_at) {
                graphs_clear();
                gsig_at = gsig();
                glp_at = head_lp;
            }
            *pm = P;   // (mapped: the graph's draft_block reads it; the last proposal waited for its end, as for anc)
            if (!gx[n] && gseen[n]++ >= 1) d2_capture(P, n);
            if (gx[n]) {
                DK(cudaGraphLaunch(gx[n], s));
                ++g_replays;
                if (gmode == 2) {   // (verify: the replay's candidates and hiddens against the launches', bit for bit)
                    std::vector<float> ha((size_t) n * D), hb((size_t) n * D);
                    DK(cudaMemcpyAsync(ha.data(), hid + D, ha.size() * 4, cudaMemcpyDeviceToHost, s));
                    DK(cudaStreamSynchronize(s));
                    std::vector<k::CandRow> a(cand, cand + n);
                    if (!d2_region(P, n, false, err)) return false;
                    DK(cudaMemcpyAsync(hb.data(), hid + D, hb.size() * 4, cudaMemcpyDeviceToHost, s));
                    DK(cudaStreamSynchronize(s));
                    bool same = std::memcmp(ha.data(), hb.data(), ha.size() * 4) == 0;
                    for (int i = 0; i < n && same; ++i) {
                        const int m = std::max(0, std::min(a[(size_t) i].n, k::kCandCap));
                        same = a[(size_t) i].n == cand[i].n && std::memcmp(a[(size_t) i].id, cand[i].id, (size_t) m * 4) == 0 &&
                               std::memcmp(a[(size_t) i].v, cand[i].v, (size_t) m * 4) == 0;
                    }
                    ++g_checks;
                    if (!same) ++g_mismatch;
                }
            } else if (!d2_region(P, n, false, err)) {
                return false;
            }
        } else if (!d2_region(P, n, false, err)) {
            return false;
        }
        k::to_bf16(hid + D, n, D, D, x16, D, s);   // (the mask columns' hiddens as BF16 for the selector's projection)
        if (!gemm(selw, R, D, D, x16, n, D, gsel, R, 0.0f, err)) return false;   // g = W_hidden h
        prof.mark(s, 1);
        if (dbg & 4) {   // (diagnostic: each column's top candidate, q one-hot, no selector)
            DK(cudaStreamSynchronize(s));
            prof.mark(s, 2);
            ++prof.n;
            anc_busy = false;
            drafts.resize((size_t) n);
            cf.assign((size_t) n, 1.0f);
            for (int i = 0; i < n; ++i) drafts[(size_t) i] = cand[i].n > 0 ? cand[i].id[0] : 0;
            if (sa && qd) {
                qd->assign((size_t) n, {});
                for (int i = 0; i < n; ++i) (*qd)[(size_t) i].push_back({drafts[(size_t) i], 1.0f});
            }
            return true;
        }
        k::dflash2_select(cand, gsel, pcb, scb, R, anchor, n, g.sel_k, sa ? sa->temperature : 0.0f, sa ? uu : nullptr,
                          out, conf, sa ? qh : nullptr, s);
        const int* back = host;
        if (gr) {   // (the drafts and confidences by a kernel into mapped memory: no wait behind the copy engine's queue)
            k::copy_words(out, hostm, (size_t) n * 4, s, conf, hostm + W, (size_t) n * 4);
            DK(cudaGetLastError());   // (copy_words only prints a launch error: hostm would still hold the last drafts)
            back = hostm;
        } else {
            DK(cudaMemcpyAsync(host, out, (size_t) n * 4, cudaMemcpyDeviceToHost, s));
            DK(cudaMemcpyAsync(host + W, conf, (size_t) n * 4, cudaMemcpyDeviceToHost, s));
        }
        DK(cudaStreamSynchronize(s));
        prof.mark(s, 2);
        ++prof.n;
        anc_busy = false;
        drafts.assign(back, back + n);
        cf.resize((size_t) n);
        std::memcpy(cf.data(), back + W, (size_t) n * 4);
        if (sa && qd) {
            qd->assign((size_t) n, {});
            for (int i = 0; i < n; ++i)
                for (int j = 0; j < qh[i].n; ++j) (*qd)[(size_t) i].push_back({qh[i].id[j], qh[i].q[j]});
        }
        return true;
    }

    bool set_proj(int mode, std::string& err) {
        if (mode < 0) mode = pbuilt;
        if (mode > 3) {
            err = "draft_proj must be 0 (MXFP8 x F32), 1 (MXFP8 x MXFP8), 2 (NVFP4 x NVFP4) or 3 (NVFP4 x two-term NVFP4)";
            return false;
        }
        if (mode <= 1 ? !has8 : !has4) {
            err = std::string("draft_proj ") + std::to_string(mode) + " needs the drafter's " + (mode <= 1 ? "MXFP8" : "NVFP4") +
                  " projections, which this load does not hold (" + (g.dflash2 ? "--dflash2-alt adds the other format" : "DSpark is MXFP8") + ")";
            return false;
        }
        cudaSetDevice(dev);
        if (mode == 1 && !::mm3::dense_fp8::supported()) {
            err = "draft_proj 1 needs the 8-bit tensor-core kernel of an RTX 5090 (sm_120)";
            return false;
        }
        if (mode >= 2 && !f4::supported()) {
            err = "draft_proj 2 and 3 need the FP4 tensor cores of an RTX 5090 (sm_120)";
            return false;
        }
        if (mode == 1 && !f4_scratch(err)) return false;
        if (mode != pmode) {
            pmode = mode;
            graphs_clear();
            // (the retained context's K / V were projected in the old mode: drop them, so the request's prefix is read
            //  again - draft_context_available false - and its drafter attends to keys of its own arithmetic only)
            retained.reset(ring, g.window);
        }
        return true;
    }

    // draft_shadow: the proposal just made (drafts, in mode pmode) made again in mode 0 with the same draws, compared
    // position by position: sh_off[i] counts the proposals whose drafts before i agreed, sh_agree[i] those where i did too
    bool shadow_run(int anchor, int P, int n, const std::vector<int>& drafts, const k::SampleArgs* sa, const float* u,
                    std::string& err) {
        const int keep = pmode, kg = gmode;
        pmode = 0;
        gmode = 0;
        std::vector<int> d0;
        std::vector<float> c0;
        const bool ok = propose(anchor, P, n, d0, c0, sa, u, nullptr, err);
        pmode = keep;
        gmode = kg;
        if (!ok) return false;
        for (size_t i = 0; i < drafts.size() && i < d0.size() && i < 16; ++i) {
            ++sh_off[i];
            if (d0[i] != drafts[i]) break;
            ++sh_agree[i];
        }
        return true;
    }

    int proj_modes() const {   // (the modes whose weights are held and whose kernel runs on the drafter's GPU)
        int cur = 0, m = 0;
        cudaGetDevice(&cur);
        cudaSetDevice(dev);
        if (has8) m |= 1 | (::mm3::dense_fp8::supported() ? 2 : 0);
        if (has4 && f4::supported()) m |= 4 | 8;
        cudaSetDevice(cur);
        return m;
    }
};

Draft::Draft() : p(new Impl) {}
Draft::~Draft() { delete p; }
bool Draft::load(const Pack& pk, int dev, cudaStream_t s, cublasHandle_t blas, const x3::Mat* head,
                 const uint16_t* emb, int max_rows, std::string& err) {
    return p->load(pk, dev, s, blas, head, emb, max_rows, err);
}
bool Draft::load_dflash2(const Pack& dp, const Pack& pk, int dev, cudaStream_t s, cublasHandle_t blas,
                         const x3::Mat* head, const uint16_t* emb, int max_rows, std::string& err) {
    return p->load_d2(dp, pk, dev, s, blas, head, emb, max_rows, err);
}
void Draft::set_emb_rows(std::function<void(int, uint16_t*)> f) { p->emb_row = std::move(f); }
void Draft::set_head_lowp(const mm3::head_lowp::Head* h) { p->head_lp = h; }
const DraftGeo& Draft::geo() const { return p->g; }
bool Draft::context_available(int pos) const { return p->retained.available(pos); }

bool Draft::context(const float* taps, int T, int pos0, std::string& err, size_t stride) {
    return p->context(taps, T, pos0, stride, err);
}
bool Draft::context_tap(int j, const float* rows, int T, std::string& err) { return p->context_tap(j, rows, T, err); }
bool Draft::context_finish(int T, int pos0, std::string& err) { return p->context_finish(T, pos0, err); }
bool Draft::propose(int anchor, int pos, int n, std::vector<int>& drafts, std::vector<float>& conf, float* logits,
                    std::string& err) {
    (void) logits;
    if (!p->propose(anchor, pos, n, drafts, conf, nullptr, nullptr, nullptr, err)) return false;
    return !(p->shadow && p->pmode != 0) || p->shadow_run(anchor, pos, n, drafts, nullptr, nullptr, err);
}
bool Draft::propose_sampled(int anchor, int pos, int n, const k::SampleArgs& sa, const std::vector<float>& u,
                            std::vector<int>& drafts, std::vector<float>& conf,
                            std::vector<std::vector<std::pair<int, float>>>& q, std::string& err) {
    if ((int) u.size() < std::max(1, std::min(n, p->g.block))) {
        err = "dspark: fewer draws than drafts";
        return false;
    }
    if (!p->propose(anchor, pos, n, drafts, conf, &sa, u.data(), &q, err)) return false;
    return !(p->shadow && p->pmode != 0) || p->shadow_run(anchor, pos, n, drafts, &sa, u.data(), err);
}
size_t Draft::bytes() const { return p->bytes; }
void Draft::set_debug(int bits) { p->dbg = bits; }
bool Draft::load_alt(const Pack& ap, std::string& err) { return p->load_alt(ap, err); }
bool Draft::set_proj(int mode, std::string& err) { return p->set_proj(mode, err); }
int Draft::proj() const { return p->pmode; }
int Draft::proj_builtin() const { return p->pbuilt; }
int Draft::proj_modes() const { return p->proj_modes(); }
bool Draft::set_graph(int mode, std::string& err) {
    if (mode < 0 || mode > 2) {
        err = "draft_graph must be 0, 1 or 2 (2: every replay checked against the launches)";
        return false;
    }
    if (mode == 1 && !p->g.dflash2) {
        err = "draft_graph 1 needs the DFlash2 draft (DSpark's proposal has no graph form)";
        return false;
    }
    // (graphs kept across requests while nothing they depend on changes: Impl::gsig, checked at each proposal)
    if ((mode == 0) != (p->gmode == 0)) {
        cudaSetDevice(p->dev);
        p->graphs_clear();
    }
    p->gmode = mode;
    return true;
}
int Draft::graph() const { return p->gmode; }
bool Draft::set_shadow(int on, std::string& err) {
    if (on && !p->has8) {
        err = "draft_shadow compares against mode 0, whose MXFP8 projections this load does not hold (--dflash2-alt)";
        return false;
    }
    p->shadow = on ? 1 : 0;
    return true;
}
void Draft::counts(long long& replays, long long& checks, long long& mismatches, std::vector<long long>& offered,
                   std::vector<long long>& agreed) const {
    replays = p->g_replays;
    checks = p->g_checks;
    mismatches = p->g_mismatch;
    offered.assign(p->sh_off, p->sh_off + std::max(1, std::min(p->g.block, 16)));
    agreed.assign(p->sh_agree, p->sh_agree + std::max(1, std::min(p->g.block, 16)));
}
void Draft::reset_counts() {
    p->g_replays = p->g_checks = p->g_mismatch = 0;
    std::fill(p->sh_off, p->sh_off + 16, 0LL);
    std::fill(p->sh_agree, p->sh_agree + 16, 0LL);
}

long long Draft::d2_room_delta(const Pack& dp, const Pack* alt) {
    const std::string X = "dflash2.";
    const long long D = (long long) dp.num(X + "hidden_size"), hd = (long long) dp.num(X + "head_dim");
    const long long Q = (long long) dp.num(X + "num_attention_heads") * hd, KV = (long long) dp.num(X + "num_key_value_heads") * hd;
    const long long F = (long long) dp.num(X + "intermediate_size"), NL = (long long) dp.num(X + "num_hidden_layers");
    const long long cg = (long long) dp.num(X + "conv_group_size", 16), G4 = cg > 0 ? 4 * (D / cg) : 0;
    const long long NK = NL * (Q * D + 2 * KV * D + D * Q + 2 * F * D + D * F + 2 * G4 * D);   // the projections' weights
    const long long b8 = NK + NK / 32, b4 = NK / 2 + NK / 16;   // (MXFP8: a byte + a scale a 32; NVFP4: half + one a 16)
    long long d = 0;
    if (dp.str(X + "weights") == "nvfp4") d -= b8 - b4;
    if (alt) d += alt->str(X + "weights") == "nvfp4" ? b4 : b8;
    return d;
}

size_t Draft::plan_bytes(const Pack& pk, int max_rows) {
    // (Impl::load's alloc() calls, in its order - keep the two alike)
    const size_t D = (size_t) pk.num("hidden_size"), hd = (size_t) pk.num("head_dim");
    const size_t Q = (size_t) pk.num("num_attention_heads") * hd, KV = (size_t) pk.num("num_key_value_heads") * hd;
    const size_t V = (size_t) pk.num("vocab_size"), NL = (size_t) pk.num("dspark.num_hidden_layers");
    const size_t F = (size_t) pk.num("dspark.dspark_ffn_hidden_size"), R = (size_t) pk.num("dspark.dspark_markov_rank");
    const size_t B = (size_t) pk.num("dspark.dspark_block_size"), NT = pk.ints("dspark.dspark_target_layer_ids").size();
    const size_t ring = (size_t) pk.num("dspark.dflash.sliding_window", 4096) + 64;
    const size_t n_rot = (size_t) pk.num("rotary_dim", (double) hd * pk.num("partial_rotary_factor", 1.0));
    if (NL == 0 || B == 0 || NT == 0 || D == 0) return 0;
    const size_t Rw = (size_t) std::max(max_rows, 8), M = std::max(D, std::max(Q, F));
    const auto mx = [](size_t N, size_t K) { return N * K + N * (K / 32); };   // (E4M3 + a scale every 32)
    size_t b = D * NT * D * 2 + 2 * D * 4 + 2 * V * R * 2 + (D + R) * 4 + 4;   // fc, the norms, Markov, confidence
    b += NL * (mx(Q, D) + 2 * mx(KV, D) + mx(D, Q) + 2 * mx(F, D) + mx(D, F) + 2 * D * 4 + 2 * hd * 4 +
               2 * ring * KV * 2);                                              // the layers, their rings
    b += 16 * 16 * M * 4 + k::attn_part_floats((int) B, (int) (KV / hd), Impl::kSplits) * 4 +
         k::markov_part_floats() * 4 + n_rot / 2 * 4;
    b += (2 * Rw * D + B * D + B * Q + 2 * Rw * KV + B * Q + B * D + 2 * B * F + B * D + B * V + V + R + B) * 4;
    b += (Rw * M + B * D + R) * 2 + (Rw + B + 1 + B) * 4 + D * 4;               // x16 hid16 embb, the ints, maskf
    return b;
}

}  // namespace mm3
