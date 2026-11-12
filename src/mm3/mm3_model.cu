#include "mm3_demand_snapshot.hpp"
#include "mm3_native_demand_units.hpp"
#include "mm3_profile_control.hpp"
#include "mm3_learned_placement.hpp"
#include "mm3_batch_staging.hpp"
#include "mm3_cooperative_policy.hpp"
#include "mm3_replay_workload.hpp"
#include "mm3_primary_backing_ledger.hpp"
#include "mm3_route_lab.hpp"
#include "mm3_route_record.hpp"
// src/mm3/mm3_model.cu - maya-mm3's model (mm3_model.hpp): loading, placement over the GPUs, and the forward.
//
// A layer's forward, as transformers' MiniMaxM3VLDecoderLayer:
//   x = rms(h) (1 + w);  q, k, v = x Wq, x Wk, x Wv (EXL3);  per-head RMSNorm, RoPE on 64 of 128 dims;  K, V -> cache
//   the indexer: iq = x Wiq, ik = x Wik (F32 GEMMs over the FP16 weights), normed and rotated, ik -> its cache; each
//   query's block scores (the max over a block's keys), its own block and the 15 best kept, per KV group
//   attention over the kept blocks -> x Wo;  h += it
//   x = rms(h) (1 + w');  layers 0-2: the dense swigluoai MLP;  else the router (F32 logits, sigmoid, + bias, top 4,
//   renormalised), the routed experts (EXL3 sets: the resident ones in one pass, the staged ones in passes of the
//   staging slots), the shared expert, h += 2.0 * routed + shared
#include "mm3_model.hpp"
#include "mm3_capture.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <functional>
#include <optional>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <array>
#include <deque>
#include <fstream>
#include <map>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <tuple>

#include "mm3_cands.hpp"
#include "mm3_cpu.hpp"
#include "mm3_env.hpp"
#include "mm3_dspark.hpp"
#include "mm3_exl3.hpp"
#include "mm3_geo.hpp"
#include "mm3_gguf.cuh"
#include "mm3_gguf_host.hpp"
#include "mm3_kernels.cuh"
#include "mm3_lane_plan.hpp"
#include "mm3_stage_reuse.hpp"
#include "mm3_native_dense.cuh"
#include "mm3_native_cpu.hpp"
#include "mm3_native_experts.cuh"
#include "mm3_native_helper.hpp"
#include "mm3_native_stage_schedule.hpp"
#include "mm3_native_helper_queue.cuh"
#include "mm3_helper_timing.hpp"
#include "mm3_native_host.hpp"
#include "mm3_native_scale_store.hpp"
#include <zstd.h>
#include "mm3_resident_router.cuh"
#include "mm3_r2_lifecycle.hpp"
#include "mm3_resident_loader.hpp"
#include "ops/minimax_nvfp4_ffn/record.h"
#include "ops/minimax_nvfp4_ffn/device_queue.h"
#include "mm3_os.hpp"
#include "mm3_tp.hpp"
#include "mm3_vision.hpp"

namespace mm3 {
namespace x3 = strata::mm3::exl3;
namespace nd = strata::mm3::native_dense;
namespace ne = strata::mm3::native_experts;
namespace nf = infernix::ops::minimax_ffn;

#define CK(call)                                                                       \
    do {                                                                               \
        const cudaError_t e_ = (call);                                                 \
        if (e_ != cudaSuccess) {                                                       \
            err = std::string(#call) + ": " + cudaGetErrorString(e_);                  \
            return false;                                                              \
        }                                                                              \
    } while (0)
#define CB(call)                                                                       \
    do {                                                                               \
        const cublasStatus_t b_ = (call);                                              \
        if (b_ != CUBLAS_STATUS_SUCCESS) {                                             \
            err = std::string(#call) + ": cublas status " + std::to_string((int) b_);  \
            return false;                                                              \
        }                                                                              \
    } while (0)

namespace {

constexpr size_t kAlign = 256;
constexpr int kHeadRows = 32;   // the output head's rows a GEMM (its F32 logits buffer)
constexpr int kWin = 16;        // a window's tokens at most (the token path's kernels: mv's columns, win_experts)
size_t up(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }
double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

const std::string P = "language_model.model.";

struct Src {   // an expert's nine pieces in the mapped files: (w1, w3, w2) x (trellis, suh, svh) - a GGUF's: the three
               // matrices' rows alone (pieces 0, 3, 6; the others empty)
    const uint8_t* p[9];
    size_t n[9];
    float input_global[3]{};
    float weight_global[3]{};
    int file[9]{};
    uint64_t off[9]{};
};

struct Layer {
    int dev = 0;
    bool moe = false;
    float *in1 = nullptr, *post1 = nullptr, *qn1 = nullptr, *kn1 = nullptr, *iqn1 = nullptr, *ikn1 = nullptr;
    x3::Mat q, k, v, o, mg, mu, md, sg, su, sd, niq, nik;
    float *wiq = nullptr, *wik = nullptr, *gate = nullptr, *bias = nullptr;
    // (a GGUF's Q8_0 indexer projections: kept as stored instead of wiq / wik - the token kernels read the blocks, the
    //  prompt path decodes them into Dev::wtmp for its F32 GEMM: the F32 values either way, a quarter of the bytes)
    const uint8_t *wiq8 = nullptr, *wik8 = nullptr;
    uint16_t *kc = nullptr, *vc = nullptr, *ic = nullptr;
    k::KvQ kvq;   // (--kv-cache: K's / V's quantized caches - a format other than BF16 instead of kc / vc)
    k::KvHost kvh;   // (--kv-streaming: the caches above a ring, every position in host memory too)
    // (--parallel) every sequence's caches - kc / vc / ic / kvq above are the bound one's (Model::bind)
    struct SeqKv {
        uint16_t *kc = nullptr, *vc = nullptr, *ic = nullptr;
        k::KvQ kvq;
    };
    std::vector<SeqKv> sk;
    x3::ExpertLayout lay;
    GgFmt gf;   // (a GGUF's experts: their format on the CPU lane)
    size_t blob = 0;
    std::vector<Src> src;
    std::vector<int> res_slot;     // expert -> its slot in the resident region (-1: none)
    std::vector<int> res_ids;      // slot -> expert
    uint8_t* res_base = nullptr;   // the resident region: res_ids.size() blobs, blob bytes apart
    std::vector<uint8_t*> res_ptr; // fixed reseed: addresses in retained native allocations
    std::vector<const uint8_t*> host_blob;   // expert -> its assembled blob in the pinned pool (null: none)
    std::array<uint32_t, 4> helper_ready{};
    std::array<uint32_t, 4> third_ready{};   // (third: experts with a copy in the third card's pool; the HOST still owns them)
    // host-free (every routed expert of the layer in VRAM): the decode's plan made on the device from exp_tab (expert ->
    // its blob's device address) - no wait for the routing, no round trip through the host
    bool hf = false;
    unsigned long long* exp_tab = nullptr;
};

struct Dev {
    int id = 0;
    cudaStream_t s = nullptr;
    cublasHandle_t blas = nullptr;
    cudaEvent_t hop = nullptr;   // (a forward crossing to the next GPU: its stream waits on this, not the host)
    cudaEvent_t native_prefix_begin=nullptr, native_prefix_end=nullptr;
    // a side stream (CUDA): a token's / a window's indexer and router beside its attention's and shared expert's
    // GEMVs - small latency-bound kernels in the shadow of bandwidth-bound ones (fev / jev: the fork, the join)
    cudaStream_t s2 = nullptr;
    cudaEvent_t fev = nullptr, jev = nullptr;
    std::vector<int> layers;
    size_t free0 = 0;
    // staging
    uint8_t* stg = nullptr;
    size_t slot_bytes = 0;
    int n_slots = 0;
    std::vector<long long> slot_owner;   // layer * 1024 + expert, -1
    std::vector<long long> slot_used;    // the forward counter at its last use
    // the promotions (Options::promote): a slot being filled on the copy stream (pending: its key, its event) is no
    // one's until the copy is done
    cudaStream_t cs = nullptr;
    std::vector<cudaEvent_t> slot_ev;
    std::vector<uint8_t> slot_foreground; // same-stream copies not yet observed complete
    // early_copy: an owned frame whose copy-stream copy has not been observed complete (its readers wait on slot_ev),
    // and a frame filled early and not yet read by the layer it was predicted for.
    std::vector<uint8_t> slot_early, slot_early_fresh;
    std::vector<long long> slot_pend;    // the key a pending slot will hold (-1: none)
    std::vector<uint8_t> slot_iss;       // successful background copy and event submission
    // Prediction attribution follows the frame through pending and publication.
    std::vector<uint8_t> slot_prediction;
    float* prediction_logits = nullptr;
    int* prediction_ids = nullptr;
    float* prediction_weights = nullptr;
    uint8_t* prediction_host = nullptr;
    cudaEvent_t prediction_uploads = nullptr;
    cudaEvent_t prep_ev[2] = {nullptr, nullptr};   // MM3_PREPARE_EVENTS: a T=1 layer's preparation on the primary stream
    // early_copy: per layer, the next layer's one-row prediction (router product, top-k, ids to the host) as one graph
    std::vector<cudaGraphExec_t> prediction_graphs;
    std::vector<uint8_t> prediction_graph_failed;
    // prep_graph verify (diagnostic): the residual saved before a replay, the replay's and the launches' results, and
    // both read back for the comparison (device buffers dalloc's)
    float* pv_save = nullptr;
    uint8_t *pv_a = nullptr, *pv_b = nullptr, *pv_h = nullptr;
    // prep_graph: device views of the pinned buffers its copy kernels read and write (cudaHostGetDevicePointer)
    void *ready_h_dev = nullptr, *original_h_dev = nullptr, *win_rb_dev = nullptr;
    void* prediction_host_dev = nullptr;   // (and early_copy's prediction ids, when its prediction is in the graph)
    void* win_plan_h_dev = nullptr;        // (expert_graph: the device view of the pinned plan its first kernel reads)
    int n_pending = 0;
    struct Promo {
        int slot;
        const uint8_t* src;
        size_t bytes;
    };
    std::vector<Promo> promo_q;          // the promotions a forward chose, copied at its end (its own copies done)
    // Native main-bank exchange rotates physical frames without overwriting a
    // live main resident. These are empty until the explicit policy control.
    std::vector<uint8_t*> slot_ptr;
    std::vector<uint8_t> slot_reserved;
    struct AdaptiveSwap {
        long long incoming=-1, victim=-1;
        int landing=-1;
        uint8_t *source_host=nullptr, *source_gpu=nullptr, *landing_gpu=nullptr, *target_host=nullptr;
        cudaEvent_t event=nullptr;
        bool submitted=false;
        // exchange_schedule: the upload's own event (it alone stays on the copy stream), and the victim's download owed
        // in pieces on the download stream; event is recorded only after the last piece.
        cudaEvent_t h2d_event=nullptr;
        bool scheduled=false, h2d_issued=false, h2d_done=false;
        size_t d2h_done=0, d2h_total=0;
    };
    std::array<AdaptiveSwap,2> adaptive;
    cudaStream_t xs = nullptr;   // exchange_schedule: the 5090 victims' downloads, off the copy stream (created on first use)
    uint8_t* pin = nullptr;              // a pinned blob for the copies
    unsigned long long* plan_h = nullptr;   // pinned: the token's plan (8 pointers, then 8 weights)
    unsigned long long* plan_d = nullptr;
    float* xh_h = nullptr;               // pinned: the token's normed input (the CPU lane's x), then its part + flag
    // pinned staging of a forward's inputs and the head's logits: the copies are the stream's own DMA commands, in its
    // order (a copy from or to pageable memory is not: on Windows + AMD, whose queued work starts only at a host wait,
    // such a copy may run before the kernels queued ahead of it - stale logits)
    float* emb_h = nullptr;              // [chunk][dim] the embeddings (the first layer's GPU)
    int* pos_h = nullptr;                // [chunk] the positions
    int* top_h = nullptr;                // [chunk][k] a prompt chunk's routing, read back
    float* lg_h = nullptr;               // [kHeadRows][vocab] the head's logits (the head's GPU)
    k::CandRow* nuc_c = nullptr;         // (step 6, the GPU nucleus) [kWin] a forward's head rows as candidates (mapped)
    k::RowStats* nuc_s = nullptr;        // ... and each row's maximum and total (mapped)
    void* nuc_scratch = nullptr;         // ... row_stats' partials (device)
    void* nuc_topk = nullptr;            // ... topk_rows' split scratch (device): the candidates over many blocks a row
    float* cpu_d = nullptr;              // the CPU lane's part [dim] and its flag (an int after it)
    // a window (2 .. kWin tokens: a DSpark verify): the readback (routes, weights, normed inputs), the plan for
    // win_experts (pinned, then on the device), the CPU lane's parts and flags in mapped memory with the flag the
    // device waits on, and the experts' scratch
    uint8_t* win_rb = nullptr;           // pinned: top [kWin][8] int, w [kWin][8] float, x [kWin][dim] float
    float* capture_scores_d = nullptr; // detailed capture only: kWin * 128 * 3 floats
    float* capture_scores_h = nullptr; // pinned peer of capture_scores_d
    // route_lab (mm3_native_route_lab.inc), allocated at a lab request's first use: the router's scores, a host-free
    // layer's kernel ids and weights read back and its weights written back, the masks at the routing
    float* lab_scores_d = nullptr;     // kWin * 128 * 3 floats
    float* lab_scores_h = nullptr;     // pinned peer
    uint8_t* lab_hf_h = nullptr;       // pinned: ids [kWin][4] int, weights [kWin][4] float, weights out [kWin][4] float
    route_lab::Masks lab_masks{};
    uint8_t* win_plan_h = nullptr;       // pinned: plan_ptr [kWin][8], plan_w [kWin][8], grp_ptr [kWin 8], grp_start
    uint8_t* owner_plan_h = nullptr; // immutable plans until next joined window
    uint8_t* win_plan_d = nullptr;       //   [kWin 8 + 1], grp_ent [kWin 8], n_groups
    float* win_cpu = nullptr;            // mapped: the CPU parts [kWin][dim], then the flags [kWin] int, the wait word
    unsigned win_wait = 0;
    std::vector<float> helper_result;    // the 3090's merged rows each pass (collect zero-fills them); kept, not reallocated
    std::vector<float> third_result;     // (third: the third card's merged rows, likewise)
    float *win_h = nullptr, *win_part = nullptr;
    float* attp = nullptr;               // a token's / a window's attention splits' partials (k::attn_part_floats)
    float* wtmp = nullptr;               // (Q8_0 indexer projections) the prompt path's F32 copy of one of them
    // scratch (chunk rows)
    float *h = nullptr, *xn = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *iq = nullptr, *ik = nullptr;
    float *bs = nullptr, *att = nullptr, *ao = nullptr, *g = nullptr, *u = nullptr, *logit = nullptr, *w = nullptr;
    float *y = nullptr, *sh = nullptr, *inv = nullptr, *out = nullptr, *lg = nullptr;   // (lg: the head's rows)
    uint16_t *xn16 = nullptr, *att16 = nullptr, *h16 = nullptr;
    int *pos = nullptr, *kept = nullptr, *top = nullptr, *rowidx = nullptr, *invrow = nullptr, *bounds = nullptr;
    size_t nblk = 0;
    unsigned* hf_original_d = nullptr; // original IDs on host-free native layers
    int *hf_route_error_d = nullptr, *hf_route_error_h = nullptr;
    uint32_t* hf_snapshot_h = nullptr;
    uint32_t* hf_stage_h[2] = {};
    mm3::DemandSnapshot hf_demand_seen;
    unsigned* prof_d = nullptr;   // (host-free layers) their routings counted on the device, [layer][expert]
    resident::Ready *ready_h = nullptr, *ready_d = nullptr, *hf_ready_d = nullptr;
    int *original_h = nullptr, *original_d = nullptr;
};

size_t g_up = 0;   // (the load's upload_mat / upload_f32 bytes so far: the Monitor's dense weights)

// a GGUF's matrix (pack_gguf.py: <pre>.weight, [n][k] in a ggml block type) onto the device as it is stored
x3::Mat upload_gg(const Tensor& t, const Pack& pk, const std::string& pre, std::vector<void*>& owned, std::string& err) {
    x3::Mat m;
    const int ty = gg::type_of(t.dtype);
    if (t.shape.size() != 2 || !gg::dense_type(ty)) {
        err = pre + ": a " + t.dtype + " tensor the GGUF kernels do not take as a matrix";
        return m;
    }
    m.fmt = gg::kFmtGG;
    m.gg_type = ty;
    m.n = (int) t.shape[0];
    m.k = (int) t.shape[1];
    const size_t rb = gg::row_bytes(ty, m.k);
    if (rb == 0 || t.bytes != rb * (size_t) m.n || m.k % 256 != 0) {
        err = pre + ": " + std::to_string(t.bytes) + " bytes for [" + std::to_string(m.n) + "][" + std::to_string(m.k) +
              "] " + t.dtype;
        return m;
    }
    m.ld = (int) rb;
    void* d = nullptr;
    if (cudaMalloc(&d, t.bytes) != cudaSuccess) {
        err = pre + ": out of device memory";
        return m;
    }
    owned.push_back(d);
    g_up += t.bytes;
    cudaMemcpy(d, pk.data(t), t.bytes, cudaMemcpyHostToDevice);
    m.raw = d;
    return m;
}

x3::Mat upload_mat(const Pack& pk, const std::string& pre, std::vector<void*>& owned, std::string& err) {
    x3::Mat m;
    if (pk.str("format") == "native") {
        const Tensor* w = pk.get(pre + ".weight");
        const Tensor* sc = pk.get(pre + ".weight_scale_inv");
        if (!w || w->shape.size() != 2 || (w->dtype != "BF16" && w->dtype != "F8_E4M3")) {
            err = pre + ": expected native BF16 or MXFP8 matrix";
            return m;
        }
        m.n = (int) w->shape[0];
        m.k = (int) w->shape[1];
        const bool mx = w->dtype == "F8_E4M3";
        if (w->bytes != (size_t) m.n * m.k * (mx ? 1 : 2) ||
            (mx && (!sc || sc->dtype != "U8" || sc->shape != std::vector<int64_t>{m.n, m.k / 32} ||
                    sc->bytes != (size_t) m.n * m.k / 32))) {
            err = pre + ": invalid native matrix or MXFP8 scale geometry";
            return m;
        }
        auto copy = [&](const Tensor& t) -> void* {
            void* p = nullptr;
            cudaError_t status = cudaMalloc(&p, t.bytes);
            if (status != cudaSuccess) { err = pre + ": " + cudaGetErrorString(status); return nullptr; }
            owned.push_back(p);
            g_up += t.bytes;
            status = cudaMemcpy(p, pk.data(t), t.bytes, cudaMemcpyHostToDevice);
            if (status != cudaSuccess) err = pre + ": " + cudaGetErrorString(status);
            return p;
        };
        m.fmt = mx ? nd::kFmtMXFP8 : nd::kFmtBF16;
        m.raw = copy(*w);
        if (mx && err.empty()) m.native_scale = (const uint8_t*) copy(*sc);
        if (err.empty() && !x3::supported(m)) err = pre + ": unsupported native dense geometry";
        return m;
    }
    if (const Tensor* w = pk.get(pre + ".weight"); w != nullptr && gg::type_of(w->dtype) > 0)
        return upload_gg(*w, pk, pre, owned, err);
    const Tensor* tr = pk.get(pre + ".trellis");
    const Tensor* su = pk.get(pre + ".suh");
    const Tensor* sv = pk.get(pre + ".svh");
    if (!tr || !su || !sv || !pk.get(pre + ".mul1") || tr->shape.size() != 3) {
        err = pre + ": not an EXL3 (mul1) matrix";
        return m;
    }
    void *dt = nullptr, *ds = nullptr, *dv = nullptr;
    if (cudaMalloc(&dt, tr->bytes) != cudaSuccess || cudaMalloc(&ds, su->bytes) != cudaSuccess ||
        cudaMalloc(&dv, sv->bytes) != cudaSuccess) {
        err = pre + ": out of device memory";
        return m;
    }
    owned.insert(owned.end(), {dt, ds, dv});
    g_up += tr->bytes + su->bytes + sv->bytes;
    cudaMemcpy(dt, pk.data(*tr), tr->bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, pk.data(*su), su->bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dv, pk.data(*sv), sv->bytes, cudaMemcpyHostToDevice);
    m.trellis = (const uint32_t*) dt;
    m.suh = (const uint16_t*) ds;
    m.svh = (const uint16_t*) dv;
    m.k = (int) (su->bytes / 2);
    m.n = (int) (sv->bytes / 2);
    m.k2 = (int) tr->shape[2] / 8;
    m.cb = 2;
    if (!x3::supported(m)) err = pre + ": shape / bits not covered by the EXL3 kernels";
    return m;
}

// a tensor's stored bytes on the device as they are (a GGUF's Q8_0 indexer projections: k::rows_dot_q8's blocks)
const uint8_t* upload_raw(const Pack& pk, const std::string& name, std::vector<void*>& owned, std::string& err) {
    const Tensor* t = pk.get(name);
    if (!t) {
        err = name + ": missing";
        return nullptr;
    }
    void* dp = nullptr;
    if (cudaMalloc(&dp, t->bytes) != cudaSuccess) {
        err = name + ": out of device memory";
        return nullptr;
    }
    owned.push_back(dp);
    g_up += t->bytes;
    cudaMemcpy(dp, pk.data(*t), t->bytes, cudaMemcpyHostToDevice);
    return (const uint8_t*) dp;
}

// a float vector on the device: the stored F32 / F16 / BF16 tensor (+ add to every value); a GGUF's Q8_0 (the
// indexer's projections) dequantized exactly - ggml's value, an F16 scale times an int8
float* upload_f32(const Pack& pk, const std::string& name, float add, std::vector<void*>& owned, size_t& n,
                  std::string& err) {
    const Tensor* t = pk.get(name);
    if (!t) {
        err = name + ": missing";
        return nullptr;
    }
    const bool q8 = t->dtype == "Q8_0";
    const size_t es = t->dtype == "F32" ? 4 : 2;
    n = q8 ? t->bytes / 34 * 32 : t->bytes / es;
    std::vector<float> h(n);
    const uint8_t* d = pk.data(*t);
    for (size_t i = 0; i < n; ++i) {
        float v;
        if (q8) {
            const uint8_t* b = d + i / 32 * 34;
            v = gg::host_f16((uint16_t) (b[0] | b[1] << 8)) * (float) (int8_t) b[2 + i % 32];
        } else if (t->dtype == "F32") {
            std::memcpy(&v, d + 4 * i, 4);
        } else if (t->dtype == "BF16") {
            uint16_t b;
            std::memcpy(&b, d + 2 * i, 2);
            const uint32_t u = (uint32_t) b << 16;
            std::memcpy(&v, &u, 4);
        } else if (t->dtype == "F16") {
            uint16_t b;
            std::memcpy(&b, d + 2 * i, 2);
            // FP16 -> F32
            const uint32_t s = (uint32_t) (b & 0x8000) << 16, e = (b >> 10) & 0x1f, m = b & 0x3ff;
            uint32_t u;
            if (e == 0) {
                if (m == 0) {
                    u = s;
                } else {   // subnormal
                    float f = (float) m / 1024.0f / 16384.0f;
                    std::memcpy(&u, &f, 4);
                    u |= s;
                }
            } else if (e == 31) {
                u = s | 0x7f800000u | (m << 13);
            } else {
                u = s | ((e + 112) << 23) | (m << 13);
            }
            std::memcpy(&v, &u, 4);
        } else {
            err = name + ": dtype " + t->dtype;
            return nullptr;
        }
        h[i] = v + add;
    }
    void* dp = nullptr;
    if (cudaMalloc(&dp, n * 4) != cudaSuccess) {
        err = name + ": out of device memory";
        return nullptr;
    }
    owned.push_back(dp);
    g_up += n * 4;
    cudaMemcpy(dp, h.data(), n * 4, cudaMemcpyHostToDevice);
    return (float*) dp;
}

// the prefill calibration's key of this PC: the GPUs (name, architecture, memory), the GPU runtime and driver - a
// calibration measured elsewhere (another GPU, a new driver) is not taken
std::string tune_key_of(const std::vector<int>& gpus) {
    std::string k = "mm3-prefill-1";
    for (int id : gpus) {
        cudaDeviceProp pr{};
        if (cudaGetDeviceProperties(&pr, id) != cudaSuccess) continue;
#if defined(STRATA_USE_HIP)
        std::string arch = pr.gcnArchName;
        if (arch.find(':') != std::string::npos) arch.resize(arch.find(':'));
#else
        const std::string arch = "sm_" + std::to_string(pr.major) + std::to_string(pr.minor);
#endif
        std::string name = pr.name;
        for (char& c : name)
            if (c == '\t' || c == '\n' || c == '|') c = ' ';
        k += "|" + name + " " + arch + " " + std::to_string((pr.totalGlobalMem + (1u << 29)) >> 30) + "GB";
    }
    int rt = 0, drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);
    return k + "|rt" + std::to_string(rt) + "|drv" + std::to_string(drv);
}

}  // namespace

struct Model::Impl {
    Options o;
    Pack pk;
    Geo g;
    Stats st;
    bool gguf = false;   // (the pack's format: a GGUF's - tools/mm3/pack_gguf.py - else EXL3)
    bool native = false; // official NVFP4 experts, MXFP8 dense and supplied BF16/F32
    // the RMSNorm weights' Gemma offset: the checkpoint stores w (the engine multiplies by 1 + w); llama.cpp's
    // converter writes w + 1 into a GGUF already - added once either way
    float norm_add() const { return gguf ? 0.0f : 1.0f; }
    std::vector<Layer> L;
    std::vector<Dev> D;
    std::vector<void*> owned;   // (cudaFree at exit, by device)
    std::vector<int> owned_dev;
    x3::Mat head;
    float* norm1 = nullptr;
    int head_dev = 0;
    // (step 6, the GPU nucleus) nuc_want: the next forwards keep the head's rows on the device (fast head only) and bring
    // back each row's candidates (topk_rows, kCandCap) and stats (row_stats) instead of its logits; nuc_rows: how many
    // rows the last one left (the device rows stay until the next forward, for a whole-row fallback)
    bool nuc_want = false;
    int nuc_rows = 0;
    // a token's and a verify window's attention: each query's kept blocks in this many runs on blocks of their own,
    // combined in run order (k::sparse_attn's splits - one block a KV group walking all of them left the GPU idle: a
    // query's sums depend on the splits alone, so a window's rows keep the token path's bits).  MAYA_MM3_ATTN_SPLIT=1:
    // one run (the decode as before)
    int attn_splits = 16;
    std::vector<float> inv_freq;
    std::vector<float*> inv_dev;
    long long fwd = 0;
    int cur_pos = 0;                 // the last position of the chunk running
    // --parallel: the sequence whose caches the layers' kc / vc / ic / kvq are (bind), and a forward_rows step's rows -
    // row r sequence rseq[r]'s token at position rpos[r] (empty: the forward of the bound sequence)
    int bound = 0;
    std::vector<int> rseq, rpos;
    void bind(int s) {
        if (s == bound || s < 0 || L.empty() || s >= (int) L[0].sk.size()) return;
        for (Layer& l : L) {
            const Layer::SeqKv& c = l.sk[(size_t) s];
            l.kc = c.kc;
            l.vc = c.vc;
            l.ic = c.ic;
            l.kvq = c.kvq;
        }
        bound = s;
    }
    std::vector<std::string> info;
    std::vector<std::string> vram;   // the Monitor's VRAM lines (serve/gpu_map.py), measured at the load's end
    // dalloc's bytes by what they hold, a device id's (the dense weights: upload_mat's / upload_f32's tally)
    enum { kAW, kAKV, kAAct, kAPool, kADraft, kAExpert, kAN };
    int acat = kAAct;
    std::map<int, std::array<size_t, kAN>> acct;
    std::map<int, size_t> acct_kvh;
    uint8_t* kv_host = nullptr;      // KV streaming: every layer's whole K / V cache (pinned, mapped)
    std::vector<uint32_t> prof;      // routings [layer][expert]: the loaded profile's plus this run's
    std::vector<uint32_t> original_prof; // native: unmodified router demand, separate from executed choices
    std::vector<uint32_t> generation_prof, generation_before;
    bool generation_active = false;
    // MM3_PREDICT_DIAG (mm3_native_prefetch.inc): next-layer predictions kept until their layer's routes record.
    const bool predict_diag = [] { const char* v = std::getenv("MM3_PREDICT_DIAG"); return v && std::string(v) == "1"; }();
    // MM3_EARLY_COPY_VERIFY=1 (diagnostic): at its first read, an early-copied frame is read back and compared with its HOST record.
    const bool early_copy_verify = [] { const char* v = std::getenv("MM3_EARLY_COPY_VERIFY"); return v && std::string(v) == "1"; }();
    std::vector<uint8_t> early_copy_check_buffer;
    // MM3_PREPARE_EVENTS=1 (diagnostic): per one-row generation MoE layer, two timing events bracket the preparation on the
    // primary stream (from where the stream reaches it to its routing readback), beside the host's enqueue time and its
    // wait at the join. Separates launch-bound preparation from kernel time, uncaptured.
    const bool prepare_events = [] { const char* v = std::getenv("MM3_PREPARE_EVENTS"); return v && std::string(v) == "1"; }();
    bool prep_armed = false;
    double prep_host_end = 0;
    std::vector<int> predict_diag_next, predict_diag_this;
    long long predict_diag_next_fwd = -1;
    int predict_diag_next_layer = -1, predict_diag_this_layer = -1;
    // cpu_first (experimental A/B control): one persistent thread that issues a window's GPU submissions
    // while the service thread runs the CPU batch. It is idle between posts and never outlives a window.
    struct SubmitWorker {
        std::thread thread;
        std::mutex mutex;
        std::condition_variable cv;
        std::function<bool()> task;
        bool pending=false, finished=true, stop=false, result=true;
        void ensure() { if(!thread.joinable()) thread=std::thread([this]{ loop(); }); }
        void loop() {
            std::unique_lock<std::mutex> lock(mutex);
            for(;;) {
                cv.wait(lock,[&]{return pending || stop;});
                if(stop) return;
                auto current=std::move(task);pending=false;
                lock.unlock();
                const bool ok=current();
                lock.lock();
                result=ok;finished=true;
                cv.notify_all();
            }
        }
        void post(std::function<bool()> next) {
            {std::lock_guard<std::mutex> lock(mutex);task=std::move(next);pending=true;finished=false;}
            cv.notify_all();
        }
        bool wait() {
            std::unique_lock<std::mutex> lock(mutex);
            cv.wait(lock,[&]{return finished;});
            return result;
        }
        ~SubmitWorker() {
            {std::lock_guard<std::mutex> lock(mutex);stop=true;}
            cv.notify_all();
            if(thread.joinable()) thread.join();
        }
    };
    SubmitWorker submit_worker;
    long long native_demand_rows = 0;
    int native_demand_block_rows = 0;
    std::vector<uint32_t> cpu_seen;  // (MM3_MISS_STATS: the CPU lane's computations [layer][expert] this run)
    // the promotions' scores: an expert not in VRAM, each use (on the CPU lane, or from a slot) adds 1, decayed by
    // kDecay a forward; at kPromote it is copied into the slot whose expert scores least
    bool hf_demand_poisoned = false, hf_demand_active = false;
    std::vector<float> use_score;
    std::vector<long long> use_last;
    static constexpr float kDecay = 0.95f, kPromote = 1.8f;
    static constexpr int kMaxPending = 4;
    // residency_decay (per request, default 0.95): the native score's decay per five generated rows, in its update and its
    // credit. The 5090 policy's horizon discount keeps kDecay: horizon tuning waits for the planner rebuild.
    double demand_decay = 0.95;
    uint8_t* pool = nullptr;         // the pinned pool
    std::unique_ptr<NativeHostBank> native_pool;
    std::unique_ptr<::mm3::scale_store::Service> scale_service;
    int scale_device = -1; // CUDA owner of both codec streams and every codec lease
    size_t host_stride = 0;
    uint8_t *scale_fg_scratch = nullptr, *scale_opt_scratch = nullptr;
    bool compressed_host() const { return native && !o.native_cooperative; }
    size_t host_record_bytes(const Layer& l) const { return compressed_host() ? host_stride : l.blob; }
    size_t scale_encode_scratch_bytes() const { return ::mm3::scale_store::kScaleBytes + ZSTD_compressBound(::mm3::scale_store::kBlockBytes); }
    std::unique_ptr<NativeHostBank> native_embedding;
    bool stage_poisoned = false;
    float* stage_hidden[2] = {};
    cudaEvent_t stage_ready[2] = {};
    float* serial_bridge = nullptr;
    size_t serial_bridge_bytes = 0;
    std::vector<int> native_source_fd;
    std::unique_ptr<NativeHelper> helper;
    void* native_cpu_workspace = nullptr;
    size_t native_cpu_bytes = 0;
    bool native_cpu_failed = false;
    bool native_calibrated_after_prefill = false;
    bool native_primary_fp4_capable = false;
    size_t native_primary_arena_bytes = 0;
    std::string native_primary_current = "dp4a", lane_calibration_arithmetic = "none";
    std::unique_ptr<CpuLane> lane;
    std::unique_ptr<Draft> draftm;   // the DSpark draft (on the head's GPU)
    bool draft_running = true; // per-request execution, independent of resident draft allocation
    int last_done = 0;               // the last forward's tokens in the cache
    int chunk_run = 0;               // (the prefill calibration: a forward's chunk, at most o.chunk - 0: o.chunk)
    int run_chunk() const { return chunk_run > 0 ? std::min(chunk_run, o.chunk) : o.chunk; }
    std::string tkey, tfile;         // (the prefill calibration's key of this PC and its file)
    std::unique_ptr<Vision> vis;     // the vision tower (Options::vision)
    std::vector<Model::Image> imgs;  // the pictures the image tokens of the next forwards take their rows from
    float* taps = nullptr;           // its context's input: [taps][chunk][dim] F32 (the head's GPU)
    std::vector<int> tap_of;         // layer -> its tap (-1: none)

    // ---- MAYA_MM3_PROF=1 (one GPU): each forward's GPU time by part, from events on the stream - the time between two
    // marks goes to the second's part (a host wait before it included); printed at the end by kind of forward
    enum { kPAttnIn, kPIxProj, kPIndex, kPIxSel, kPAttn, kPOGemm, kPOut, kPDense, kPRouter, kPRoute, kPSync, kPExperts,
           kPHead, kPDraft, kPN };
    struct Prof {
        bool on = false;
        bool sync = false;        // (MAYA_MM3_PROF=2: the host waits at each mark and times it - AMD's events may be
                                  //  stamped as their packet is dispatched, ahead of the kernels before them)
        std::vector<double> ht;
        std::vector<cudaEvent_t> ev;
        std::vector<int> part;
        size_t used = 0;
        double ms[3][kPN] = {};   // [token, window, prompt]
        long fwd[3] = {}, tok[3] = {};
        double wall[3] = {};
    } pf;
    void pmark(Dev& d, int part) {
        if (!pf.on) return;
        if (pf.sync) {
            cudaStreamSynchronize(d.s);
            if (pf.used == pf.ht.size()) {
                pf.ht.push_back(0.0);
                pf.part.push_back(0);
            }
            pf.ht[pf.used] = now();
            pf.part[pf.used++] = part;
            return;
        }
        if (pf.used == pf.ev.size()) {
            cudaEvent_t e = nullptr;
            if (cudaEventCreate(&e) != cudaSuccess) return;
            pf.ev.push_back(e);
            pf.part.push_back(0);
        }
        cudaEventRecord(pf.ev[pf.used], d.s);
        pf.part[pf.used++] = part;
    }
    void pcollect(int kind, int T, double wall_s) {
        if (!pf.on) return;
        if (pf.sync)
            for (size_t i = 1; i < pf.used; ++i) pf.ms[kind][pf.part[i]] += 1e3 * (pf.ht[i] - pf.ht[i - 1]);
        else if (pf.used > 1 && cudaEventSynchronize(pf.ev[pf.used - 1]) == cudaSuccess)
            for (size_t i = 1; i < pf.used; ++i) {
                float ms = 0.0f;
                if (cudaEventElapsedTime(&ms, pf.ev[i - 1], pf.ev[i]) == cudaSuccess) pf.ms[kind][pf.part[i]] += ms;
            }
        pf.used = 0;
        ++pf.fwd[kind];
        pf.tok[kind] += T;
        pf.wall[kind] += wall_s;
    }
    void preport() const {
        if (!pf.on) return;
        static const char* names[kPN] = {"attn-in", "ix-proj", "indexer", "ix-select", "attention", "o-gemm",
                                         "attn-out", "dense-ffn", "router", "router+shared", "route-sync", "experts",
                                         "head", "draft-ctx"};
        static const char* kinds[3] = {"token", "window", "prompt"};
        for (int k = 0; k < 3; ++k) {
            if (pf.fwd[k] == 0) continue;
            double tot = 0;
            for (int i = 0; i < kPN; ++i) tot += pf.ms[k][i];
            std::fprintf(stderr, "MM3 PROF %s: %ld forwards, %ld tokens, %.2f ms a forward (wall %.2f ms):", kinds[k],
                         pf.fwd[k], pf.tok[k], tot / pf.fwd[k], 1e3 * pf.wall[k] / pf.fwd[k]);
            for (int i = 0; i < kPN; ++i)
                if (pf.ms[k][i] > 0) std::fprintf(stderr, " %s %.2f", names[i], pf.ms[k][i] / pf.fwd[k]);
            std::fprintf(stderr, "\n");
        }
    }

    // A failed join leaves every allocation and host mapping potentially reachable.
    // Retain the entire object in that case, including graph and host-bank owners.
    std::atomic<bool> retirement_uncertain{false};
    std::unique_ptr<nf::DeviceQueue> helper_queue;
    nf::QueueRecord *helper_queue_table_h=nullptr,*helper_queue_table_d=nullptr;
    nf::QueuePick* helper_queue_picks_d=nullptr;
    uint64_t helper_queue_next=0,helper_queue_pending=0;
    std::string helper_dispatch_current="worker";
    std::string promotion_error;
#include "mm3_native_scale_bridge.inc"
#include "mm3_native_maintenance.inc"
    bool retire_for_destroy() {
        if (!r2::retirement_may_begin(retirement_uncertain)) return false;
        if (scale_service && !scale_service->drain_cpu()) { retirement_uncertain.store(true); return false; }
        for (auto& d : D) { std::string join_error; if (d.xs && !exchange_downloads_join(d, join_error)) { retirement_uncertain.store(true); return false; } }
        const bool retired = r2::retire(retirement_uncertain.load(), D, [](Dev& d) {
            if (cudaSetDevice(d.id) != cudaSuccess) return false;
            bool ok = true;
            if (d.s && cudaStreamSynchronize(d.s) != cudaSuccess) ok = false;
            if (d.s2 && cudaStreamSynchronize(d.s2) != cudaSuccess) ok = false;
            if (d.cs && cudaStreamSynchronize(d.cs) != cudaSuccess) ok = false;
            return ok;
        });
        if (!retired) return false;
        if (scale_service) {
            for (auto& d : D) if (d.id == scale_device) {
              if(cudaSetDevice(d.id)!=cudaSuccess)return false;
              for (size_t i=0; i<d.slot_ev.size(); ++i) {
                auto state=scale_service->optional_state(int(i));
                if (state==::mm3::scale_store::OptionalState::Submitted &&
                    (cudaEventQuery(d.slot_ev[i])!=cudaSuccess || !scale_service->retire_optional(int(i)))) return false;
                if (state==::mm3::scale_store::OptionalState::Failed || state==::mm3::scale_store::OptionalState::Queued || state==::mm3::scale_store::OptionalState::Running) return false;
              }
            }
            if (!scale_service->reset_foreground_after_join() || !scale_service->release_after_gpu_join()) return false;
        }
        return helper_queue_destroy_safe() && helper_live_cancel_retired();
    }
    ~Impl() {
        if(!D.empty())cudaSetDevice(D[size_t(head_dev)].id);
        for(auto event:replay_gpu_events)if(event)cudaEventDestroy(event);
        preport();
        helper_queue.reset();
        if(helper_queue_table_h)cudaFreeHost(helper_queue_table_h);
        if(helper_queue_table_d || helper_queue_picks_d) {cudaSetDevice(D[0].id);if(helper_queue_table_d)cudaFree(helper_queue_table_d);if(helper_queue_picks_d)cudaFree(helper_queue_picks_d);}
        helper.reset();
        third.reset();
        graphs_clear();
        for (cudaEvent_t e : pf.ev) cudaEventDestroy(e);
        if (::mm3::env("MM3_MISS_STATS") && !cpu_seen.empty()) {
            long tot = 0, distinct = 0, twice = 0;
            for (uint32_t c : cpu_seen) {
                tot += c;
                distinct += c > 0;
                twice += c > 1 ? c - 1 : 0;
            }
            std::fprintf(stderr, "CPU lane: %ld expert computations, %ld distinct experts, %ld repeats (%.0f%%)\n", tot,
                         distinct, twice, tot ? 100.0 * twice / tot : 0.0);
        }
        // Native scratch and handles are stream-owned; retire work before their
        // storage (or the weights and mapped host inputs it reads) is released.
        for (auto& d : D) {
            cudaSetDevice(d.id);
            if (d.s) cudaStreamSynchronize(d.s);
            if (d.s2) cudaStreamSynchronize(d.s2);
            if (d.cs) cudaStreamSynchronize(d.cs);
            if (d.xs) cudaStreamSynchronize(d.xs);
            if (native) { nd::release(d.s); ne::release(d.s); if (d.s2) nd::release(d.s2); }
        }
        for (size_t i = 0; i < owned.size(); ++i) {
            cudaSetDevice(owned_dev[i]);
            cudaFree(owned[i]);
        }
        if (pool && !native_pool) cudaFreeHost(pool);
        if (!D.empty()) cudaSetDevice(D[0].id);
        for(int i=0;i<2;++i) { if(stage_ready[i]) cudaEventDestroy(stage_ready[i]); if(stage_hidden[i]) cudaFreeHost(stage_hidden[i]); }
        if(serial_bridge) cudaFreeHost(serial_bridge);
        native_pool.reset();
        for (auto* p : adaptive_host_owned) if (p) cudaFreeHost(p);
        for (auto* p : helper_live_buffers) if (p) cudaFreeHost(p);
        if(batch_calibration_host)cudaFreeHost(batch_calibration_host);
        std::free(native_cpu_workspace);
        for (int fd : native_source_fd) if(fd>=0) ::close(fd);
        if (kv_host) cudaFreeHost(kv_host);
        for (auto& d : D) {
            cudaSetDevice(d.id);
            if (d.cs) cudaStreamSynchronize(d.cs);
            if (d.xs) cudaStreamSynchronize(d.xs);
            for (cudaEvent_t e : d.slot_ev) cudaEventDestroy(e);
            for (auto& x : d.adaptive) {
                if (x.event) cudaEventDestroy(x.event);
                if (x.h2d_event) cudaEventDestroy(x.h2d_event);
            }
            if (d.cs) cudaStreamDestroy(d.cs);
            if (d.xs) cudaStreamDestroy(d.xs);
            if (d.plan_h) cudaFreeHost(d.plan_h);
            if (d.emb_h) cudaFreeHost(d.emb_h);
            if (d.pos_h) cudaFreeHost(d.pos_h);
            if (d.top_h) cudaFreeHost(d.top_h);
            if (d.lg_h) cudaFreeHost(d.lg_h);
            if (d.nuc_c) cudaFreeHost(d.nuc_c);
            if (d.nuc_s) cudaFreeHost(d.nuc_s);
            if (d.nuc_scratch) cudaFree(d.nuc_scratch);
            if (d.nuc_topk) cudaFree(d.nuc_topk);
            if (d.xh_h) cudaFreeHost(d.xh_h);
            if (d.win_rb) cudaFreeHost(d.win_rb);
            if (d.capture_scores_h) cudaFreeHost(d.capture_scores_h);
            if (d.capture_scores_d) cudaFree(d.capture_scores_d);
            if (d.lab_scores_h) cudaFreeHost(d.lab_scores_h);
            if (d.lab_scores_d) cudaFree(d.lab_scores_d);
            if (d.lab_hf_h) cudaFreeHost(d.lab_hf_h);
            if (d.win_plan_h) cudaFreeHost(d.win_plan_h);
            if (d.owner_plan_h) cudaFreeHost(d.owner_plan_h);
            if (d.win_cpu) cudaFreeHost(d.win_cpu);
            if (d.pin) cudaFreeHost(d.pin);
            if (d.ready_h) cudaFreeHost(d.ready_h);
            if (d.original_h) cudaFreeHost(d.original_h);
            if (d.prediction_host) cudaFreeHost(d.prediction_host);
            if (d.prediction_uploads) cudaEventDestroy(d.prediction_uploads);
            for (auto& graph : d.prediction_graphs) if (graph) cudaGraphExecDestroy(graph);
            if (d.pv_h) cudaFreeHost(d.pv_h);
            if (d.hf_snapshot_h) cudaFreeHost(d.hf_snapshot_h);
            if (d.hf_route_error_h) cudaFreeHost(d.hf_route_error_h);
            for (auto* p : d.hf_stage_h) if (p) cudaFreeHost(p);
            if (d.blas) cublasDestroy(d.blas);
            if (d.hop) cudaEventDestroy(d.hop);
            if(d.native_prefix_begin)cudaEventDestroy(d.native_prefix_begin);
            if(d.native_prefix_end)cudaEventDestroy(d.native_prefix_end);
            if (d.fev) cudaEventDestroy(d.fev);
            if (d.jev) cudaEventDestroy(d.jev);
            if (d.s2) cudaStreamDestroy(d.s2);
            if (d.s) cudaStreamDestroy(d.s);
        }
    }

    template <class T>
    bool dalloc(int dev, T*& p, size_t n, std::string& err) {
        void* v = nullptr;
        if (cudaMalloc(&v, std::max<size_t>(n * sizeof(T), 16)) != cudaSuccess) {
            err = "CUDA" + std::to_string(dev) + ": out of device memory (" + std::to_string(n * sizeof(T) >> 20) +
                  " MB)";
            return false;
        }
        owned.push_back(v);
        owned_dev.push_back(dev);
        acct[dev][(size_t) acat] += std::max<size_t>(n * sizeof(T), 16);
        p = (T*) v;
        return true;
    }
    void own_since(size_t n0, int dev) {
        while (owned_dev.size() < owned.size()) owned_dev.push_back(dev);
        (void) n0;
    }

    // a GGUF's experts (pack_gguf.py: experts.E.w1 / w3 / w2.weight, each expert's slice of the GGUF's stacked tensor):
    // the blob gate | up | down, each [rows][blocks] as stored, 256-byte aligned; gate, up and down of any types the
    // kernels take (one each for the layer's 128 experts)
    bool expert_layout_gg(Layer& l, int li, std::string& err) {
        const std::string E = P + "layers." + std::to_string(li) + ".block_sparse_moe.experts.";
        const char* parts[3] = {"w1", "w3", "w2"};
        l.src.assign((size_t) g.n_exp, {});
        size_t off = 0;
        for (int r = 0; r < 3; ++r) {
            const int n = r < 2 ? g.ff : g.dim, k = r < 2 ? g.dim : g.ff;
            const Tensor* t0 = pk.get(E + "0." + parts[r] + ".weight");
            const int ty = t0 ? gg::type_of(t0->dtype) : -1;
            const size_t bytes = (size_t) n * gg::row_bytes(ty, k);
            if (t0 == nullptr || !gg::expert_type(ty) || bytes == 0 || t0->bytes != bytes) {
                err = E + "0." + parts[r] + ": not a [" + std::to_string(n) + "][" + std::to_string(k) +
                      "] expert matrix the GGUF kernels take (" + (t0 ? t0->dtype : std::string("missing")) + ")";
                return false;
            }
            for (int e = 0; e < g.n_exp; ++e) {
                const Tensor* t = pk.get(E + std::to_string(e) + "." + parts[r] + ".weight");
                if (t == nullptr || t->dtype != t0->dtype || t->bytes != bytes) {
                    err = E + std::to_string(e) + "." + parts[r] + ": missing, or not expert 0's type and size";
                    return false;
                }
                Src& s = l.src[(size_t) e];
                s.p[r * 3] = pk.data(*t);
                s.n[r * 3] = bytes;
                for (int q = 1; q < 3; ++q) {
                    s.p[r * 3 + q] = nullptr;
                    s.n[r * 3 + q] = 0;
                }
            }
            l.lay.trellis[r] = l.lay.suh[r] = l.lay.svh[r] = off;
            l.lay.gg_type[r] = ty;
            l.lay.k2[r] = 0;
            off += up(bytes);
        }
        l.lay.fmt = gg::kFmtGG;
        l.lay.cb = 0;
        l.blob = off;
        if (o.cpu_threads >= 0 && !gg_fmt(l.lay.gg_type[0], l.lay.gg_type[1], l.lay.gg_type[2], g.dim, g.ff, l.gf, err)) {
            // (a build without ggml-cpu, or a type it has no dot for: the experts outside VRAM are staged instead)
            info.push_back("NOTE: no CPU lane - layer " + std::to_string(li) + "'s experts: " + err);
            std::fprintf(stderr, "maya-mm3: %s\n", info.back().c_str());
            err.clear();
            o.cpu_threads = -1;
        }
        return true;
    }

    bool expert_layout(Layer& l, int li, std::string& err) {
        if (gguf) return expert_layout_gg(l, li, err);
        if (native) {
            const std::string E = P + "layers." + std::to_string(li) + ".block_sparse_moe.experts.";
            const char* roles[] = {"w1", "w3", "w2"};
            l.src.assign(g.n_exp, {});
            for (int e = 0; e < g.n_exp; ++e) for (int r = 0; r < 3; ++r) {
                const std::string pre = E + std::to_string(e) + "." + roles[r];
                const int n = r == 2 ? g.dim : g.ff, k = r == 2 ? g.ff : g.dim;
                const Tensor *w = pk.get(pre + ".weight_packed"), *s = pk.get(pre + ".weight_scale"),
                             *wg = pk.get(pre + ".weight_global_scale"), *ig = pk.get(pre + ".input_global_scale");
                if (!w || w->dtype != "U8" || w->shape != std::vector<int64_t>{n, k / 2} ||
                    w->bytes != (size_t)n*k/2 || !s || s->dtype != "F8_E4M3" ||
                    s->shape != std::vector<int64_t>{n, k / 16} || s->bytes != (size_t)n*k/16 ||
                    !wg || wg->dtype != "F32" || wg->bytes != 4 || !ig || ig->dtype != "F32" || ig->bytes != 4) {
                    err = pre + ": invalid official NVFP4 planes/scales";
                    return false;
                }
                Src& dst = l.src[e];
                for (int q = 0; q < 3; ++q) { const Tensor* t = q == 0 ? w : q == 1 ? s : wg;
                    dst.p[3*r+q] = pk.data(*t); dst.n[3*r+q] = t->bytes;
                    dst.file[3*r+q]=t->file; dst.off[3*r+q]=t->off; }
                float global;
                std::memcpy(&global, pk.data(*wg), 4);
                dst.weight_global[r]=global;
                std::memcpy(&dst.input_global[r], pk.data(*ig), 4);
                if (!(global > 0) || !std::isfinite(global) || !(dst.input_global[r] > 0) || !std::isfinite(dst.input_global[r])) {
                    err = pre + ": invalid native global scale"; return false;
                }
            }
            l.lay.fmt = ne::kFmt;
            l.blob = sizeof(ne::Header) + nf::record_stride(g.dim, g.ff);
            return true;
        }
        const std::string E = P + "layers." + std::to_string(li) + ".block_sparse_moe.experts.";
        const char* parts[3] = {"w1", "w3", "w2"};   // gate, up, down
        const char* pieces[3] = {".trellis", ".suh", ".svh"};
        l.src.assign((size_t) g.n_exp, {});
        size_t off = 0;
        for (int e = 0; e < g.n_exp; ++e) {
            size_t o2 = 0;
            for (int r = 0; r < 3; ++r) {
                for (int q = 0; q < 3; ++q) {
                    const Tensor* t = pk.get(E + std::to_string(e) + "." + parts[r] + pieces[q]);
                    if (!t) {
                        err = E + std::to_string(e) + "." + parts[r] + pieces[q] + ": missing";
                        return false;
                    }
                    l.src[(size_t) e].p[r * 3 + q] = pk.data(*t);
                    l.src[(size_t) e].n[r * 3 + q] = t->bytes;
                    if (e == 0) {
                        if (q == 0) {
                            l.lay.trellis[r] = o2;
                            l.lay.k2[r] = (int) t->shape[2] / 8;
                        } else if (q == 1) {
                            l.lay.suh[r] = o2;
                        } else {
                            l.lay.svh[r] = o2;
                        }
                    } else if (q == 0 && (int) t->shape[2] / 8 != l.lay.k2[r]) {
                        err = E + std::to_string(e) + "." + parts[r] + ": bits differ from expert 0's";
                        return false;
                    }
                    o2 += up(t->bytes);
                }
            }
            if (e == 0) off = o2;
            else if (o2 != off) {
                err = E + std::to_string(e) + ": pieces differ in size from expert 0's";
                return false;
            }
        }
        l.lay.cb = 2;
        l.blob = off;
        if (l.lay.k2[0] != l.lay.k2[1]) {
            err = E + ": gate and up at different bits";
            return false;
        }
        return true;
    }

    // expert e's blob assembled in host memory dst (l.lay's offsets)
    void assemble(const Layer& l, int e, uint8_t* dst) const {
        const Src& s = l.src[(size_t) e];
        if (native) {
            nf::NativeExpert src;
            nf::NativeMatrix* role[] = {&src.gate, &src.up, &src.down};
            for (int r = 0; r < 3; ++r) {
                role[r]->codes = s.p[3*r]; role[r]->blocks = s.p[3*r+1];
                role[r]->weight_global=s.weight_global[r];
                role[r]->input_global = s.input_global[r];
            }
            const auto scales = nf::pack_native(src, g.dim, g.ff, dst + sizeof(ne::Header), l.blob - sizeof(ne::Header));
            ne::Header header{};
            std::copy_n(scales.weight, 3, header.weight_global);
            std::copy_n(scales.input, 3, header.input_global);
            std::memcpy(dst, &header, sizeof(header));
            return;
        }
        for (int r = 0; r < 3; ++r) {
            std::memcpy(dst + l.lay.trellis[r], s.p[r * 3 + 0], s.n[r * 3 + 0]);
            if (s.n[r * 3 + 1] > 0) std::memcpy(dst + l.lay.suh[r], s.p[r * 3 + 1], s.n[r * 3 + 1]);   // (a GGUF's: none)
            if (s.n[r * 3 + 2] > 0) std::memcpy(dst + l.lay.svh[r], s.p[r * 3 + 2], s.n[r * 3 + 2]);
        }
    }

    // expert e's pieces dropped from the mapped files' pages once copied (MM3_LOAD_DROP_PAGES=0: kept - a restart then
    // starts warm and a pack that fits in RAM loads a few seconds faster without the calls)
    void drop_src(const Layer& l, int e) const {
        static const bool on = [] {
            const char* v = ::mm3::env("MM3_LOAD_DROP_PAGES");
            return v == nullptr || std::atoi(v) != 0;
        }();
        if (!on) return;
        const Src& s = l.src[(size_t) e];
        for (int i = 0; i < 9; ++i)
            if (s.n[i] > 0) {
                os::drop_mapped(s.p[i], s.n[i]);
                // MADV_DONTNEED drops process PTEs; file advice also retires
                // clean source-cache pages after the native copy is complete.
                if(native && i%3!=2 && (size_t)s.file[i]<native_source_fd.size())
                    (void)::posix_fadvise(native_source_fd[s.file[i]],(off_t)s.off[i],(off_t)s.n[i],POSIX_FADV_DONTNEED);
            }
    }

    // ---- the CPU lane or a staging copy, by their measured costs (as maya-ds4's PCIe lane): a token's experts that are
    // neither in VRAM nor in a staging slot are computed on the CPU lane, or - those in the pinned pool - copied into a
    // staging slot and computed by the GPU (the copies run on the GPU's copy engine while the CPU lane computes the
    // others, and the slot keeps the expert for the next tokens).  calibrate_lane times both on this PC at the load:
    // an expert's copy from the pool, and the CPU lane's time for one expert and for four.  MM3_LANE_SPLIT=0: every such
    // expert on the CPU lane (as before)
    double lane_cpu1 = 0, lane_cpu_add = 0, lane_copy = 0;   // ms (lane_copy 0: not measured - no split)
    LaneRetime lane_rt;   // (decode's own timing of the CPU lane: its decoded tokens' calls - mm3_lane_plan.hpp)
    BatchStagingCosts batch_staging_costs;
    const ReplayBatch* replay_fixed_batch=nullptr;
    int replay_stage_limit=-1;
    bool replay_query_primary=false;
    capture::light::LayerDraft light_draft;
    std::vector<uint8_t> light_locations;
    std::vector<capture::light::SlotState> light_slots;
    std::vector<capture::light::ExchangeState> light_exchanges;
    std::string batch_staging_calibration, batch_staging_arithmetic;
    uint8_t* batch_calibration_host=nullptr; // retained if a setup copy cannot be retired
#include "mm3_native_batch_calibration.inc"
#include "mm3_native_replay_services.inc"

    // a decoded token's CPU-lane call (n experts, ms): the split's costs timed again in decode where the start-up's do
    // not hold (#552: a busy start staged experts the CPU lane computes faster for the whole run)
    void lane_retime(int n, double ms) {
        const double c1 = lane_cpu1, add = lane_cpu_add;
        if (!lane_rt.note(n, ms, lane_cpu1, lane_cpu_add)) return;
        std::fprintf(stderr, "maya-mm3: the CPU lane timed again in decode: an expert %.2f ms (+%.2f ms each more), "
                             "planned with %.2f (+%.2f); %.2f ms to copy - 4 such experts: %d staged\n",
                     lane_cpu1, lane_cpu_add, c1, add, lane_copy, lane_split(4));
    }

    // of m such experts (each in the pool), how many to stage: the fewest that bring max(CPU lane, copies) down by more
    // than a tenth
    int lane_split(int m) const {
        if (lane_copy <= 0 || m <= 0) return 0;
        const auto cpu = [&](int c) { return c > 0 ? lane_cpu1 + (c - 1) * lane_cpu_add : 0.0; };
        int best = 0;
        double bt = cpu(m);
        for (int s = 1; s <= m; ++s) {
            const double t = std::max(cpu(m - s), s * lane_copy);
            if (t < bt * 0.9) {
                bt = t;
                best = s;
            }
        }
        return best;
    }

    void calibrate_lane() {
        if (!lane || !pool) return;
        if (const char* v = ::mm3::env("MM3_LANE_SPLIT"); v != nullptr && std::atoi(v) == 0) return;
        // eight pooled experts of one GPU's layers (different ones: the CPU's caches must not hold them yet), all of the
        // first one's format - the lane's call for four takes one layer's format for them all (with few experts outside
        // VRAM the eight can span layers whose experts differ in type: a GGUF's IQ1_S and IQ2_XXS - it read past them).
        // A round whose four experts took no longer than one was disturbed (#548: the PC busy for that moment - a
        // previous engine's pinned memory still being released - read 3.98 ms an expert instead of 0.31 and staged every
        // expert for the session): another round on eight other experts, up to kRounds, and the least disturbed one
        // (the fastest expert) is kept.  A first round that reads consistently is the only one, as before
        constexpr int kRounds = 3;
        Dev* dp = nullptr;
        std::vector<std::pair<const Layer*, int>> ex;
        const auto same_fmt = [&](const Layer& a, const Layer& b) {
            if (a.blob != b.blob) return false;
            for (int r = 0; r < 3; ++r)
                if (a.lay.gg_type[r] != b.lay.gg_type[r] || a.lay.k2[r] != b.lay.k2[r]) return false;
            return true;
        };
        for (Dev& d : D) {
            if (d.n_slots <= 0) continue;
            ex.clear();
            for (int li : d.layers) {
                const Layer& l = L[(size_t) li];
                if (!ex.empty() && !same_fmt(l, *ex[0].first)) continue;
                for (int e = 0; e < g.n_exp && ex.size() < 8 * kRounds && !l.host_blob.empty(); ++e)
                    if (l.host_blob[(size_t) e] != nullptr) ex.push_back({&l, e});
            }
            if (ex.size() >= 8) {
                dp = &d;
                break;
            }
        }
        if (dp == nullptr) return;
        if(native) {
            // The observed startup P8 calibration was an order of magnitude
            // slower than sustained real-record service. Exercise the actual
            // native staging/FFN and CPU routes before measuring the split.
            cudaSetDevice(dp->id);
            std::vector<float> x((size_t)g.dim), output((size_t)g.dim);
            for(int i=0;i<g.dim;++i)x[i]=0.01f*(float)((i*37)%101-50);
            std::memset(dp->plan_h,0,16*sizeof(unsigned long long));
            dp->plan_h[0]=(unsigned long long)staging_address(*dp,0);
            ((float*)(dp->plan_h+8))[0]=1.0f;
            if(cudaMemcpyAsync(dp->plan_d,dp->plan_h,16*sizeof(unsigned long long),cudaMemcpyHostToDevice,dp->s)!=cudaSuccess ||
               cudaMemcpyAsync(dp->xn,x.data(),(size_t)g.dim*4,cudaMemcpyHostToDevice,dp->s)!=cudaSuccess) {
                native_cpu_failed=true; return;
            }
            x3::MoeArgs a; a.plan_ptr=dp->plan_d; a.plan_w=(const float*)(dp->plan_d+8);
            a.lay=ex[0].first->lay; a.k=1; a.n_embd=g.dim; a.n_ff=g.ff; a.limit=g.limit;
            auto gpu=[&](size_t i) {
                const Layer& l=*ex[i%ex.size()].first;
                const uint8_t* src=l.host_blob[(size_t)ex[i%ex.size()].second];
                if(cudaMemcpyAsync(staging_address(*dp,0),src,l.blob,cudaMemcpyHostToDevice,dp->s)!=cudaSuccess)return false;
                if(!ne::gate_up(a,dp->xn,dp->s) || !ne::down(a,nullptr,dp->out,dp->s))return false;
                return cudaStreamSynchronize(dp->s)==cudaSuccess;
            };
            auto cpu=[&](size_t i,int n) {
                std::vector<CpuExpert> jobs;
                for(int j=0;j<n;++j) { const auto& e=ex[(i+j)%ex.size()]; jobs.push_back(cpu_expert(*e.first,e.second,1.0f/n)); }
                lane_run(jobs,x.data(),output.data());
            };
            const double warm=now();size_t turns=0;
            do { if(!gpu(turns)){native_cpu_failed=true;return;} cpu(turns,1);++turns; } while(now()-warm<0.5);
            std::vector<double> primary,one,four;
            for(size_t r=0;r<7;++r) {
                double t=now();if(!gpu(r*3)){native_cpu_failed=true;return;}primary.push_back(1e3*(now()-t));
                t=now();cpu(r*3+1,1);one.push_back(1e3*(now()-t));
                t=now();cpu(r*3+2,4);four.push_back(1e3*(now()-t));
            }
            auto median=[](std::vector<double> v){std::sort(v.begin(),v.end());return v[v.size()/2];};
            lane_cpu1=median(one);lane_cpu_add=std::max(0.0,(median(four)-lane_cpu1)/3);lane_copy=median(primary);
            st.lane_cpu1_ms=lane_cpu1;st.lane_cpu_add_ms=lane_cpu_add;st.lane_primary_ms=lane_copy;
            lane_calibration_arithmetic=native_primary_current;
            dp->slot_owner.assign((size_t)dp->n_slots,-1);
            char text[320];std::snprintf(text,sizeof text,"native sustained lane calibration: warm %.3f s / %zu rounds; CPU1 %.3f ms CPU4 %.3f ms primary upload+FFN %.3f ms; four misses stage %d",
                now()-warm,turns,lane_cpu1,median(four),lane_copy,lane_split(4));
            info.push_back(text);
            if(o.native_cooperative && o.cooperative_stage_reuse) {
                std::string error;
                // MM3_CALIBRATION_REPEATS=n (diagnostic, 1 by default): repeat the grouped calibration n times on this
                // load, each report in the startup info; the planner keeps the last. Separates repeat noise from loads.
                const char* rv=std::getenv("MM3_CALIBRATION_REPEATS");
                const int repeats=rv?std::max(1,std::min(32,std::atoi(rv))):1;
                for(int rep=0;rep<repeats && !native_cpu_failed;++rep)
                    if(!calibrate_batch_staging(error)) {
                        native_cpu_failed=true;
                        std::fprintf(stderr,"native grouped service calibration failed: %s\n",error.c_str());
                    }
            }
            return;
        }
        auto med = [](std::vector<double> v) {
            std::sort(v.begin(), v.end());
            return v[v.size() / 2];
        };
        cudaSetDevice(dp->id);
        std::vector<float> x((size_t) g.dim), out((size_t) g.dim);
        for (int i = 0; i < g.dim; ++i) x[(size_t) i] = 0.01f * (float) ((i * 37) % 101 - 50);
        const int rounds = (int) ex.size() / 8;
        int round = 0, disturbed = 0;
        double best1 = 0, best_add = 0, best_copy = 0;
        for (; round < rounds; ++round) {
            const size_t b0 = (size_t) round * 8;   // (this round's eight experts)
            std::vector<double> tc;
            for (int r = 0; r < 5; ++r) {   // (the first warms the path up)
                const Layer& l = *ex[b0 + (size_t) r].first;
                const double t0 = now();
                if (cudaMemcpyAsync(dp->stg, l.host_blob[(size_t) ex[b0 + (size_t) r].second], l.blob,
                                    cudaMemcpyHostToDevice, dp->s) != cudaSuccess ||
                    cudaStreamSynchronize(dp->s) != cudaSuccess) {
                    cudaGetLastError();
                    dp->slot_owner.assign((size_t) dp->n_slots, -1);
                    if (round == 0) return;
                    break;   // (an earlier round's numbers stand)
                }
                if (r > 0) tc.push_back(1e3 * (now() - t0));
            }
            if (tc.size() < 4) break;
            dp->slot_owner.assign((size_t) dp->n_slots, -1);   // (slot 0 was written)
            std::vector<double> t1;
            for (int r = 0; r < 4; ++r) {
                const std::vector<CpuExpert> one{
                    cpu_expert(*ex[b0 + (size_t) r].first, ex[b0 + (size_t) r].second, 1.0f)};
                const double t0 = now();
                lane_run(one, x.data(), out.data());
                t1.push_back(1e3 * (now() - t0));
            }
            std::vector<CpuExpert> four;
            for (int r = 4; r < 8; ++r)
                four.push_back(cpu_expert(*ex[b0 + (size_t) r].first, ex[b0 + (size_t) r].second, 0.25f));
            const double t0 = now();
            lane_run(four, x.data(), out.data());
            const double t4 = 1e3 * (now() - t0);
            const double c1 = med(t1);
            if (round == 0 || c1 < best1) {
                best1 = c1;
                best_add = std::max(0.0, (t4 - c1) / 3.0);
                best_copy = med(tc);
            }
            if (t4 > c1) break;   // (four took longer than one: a consistent reading)
            ++disturbed;
        }
        lane_copy = best_copy;
        lane_cpu1 = best1;
        lane_cpu_add = best_add;
        char b[260];
        std::snprintf(b, sizeof b, "CPU lane vs staging (measured): an expert %.2f ms on the CPU lane (+%.2f ms each more), "
                      "%.2f ms to copy - 4 such experts: %d staged%s", lane_cpu1, lane_cpu_add, lane_copy, lane_split(4),
                      disturbed == 0 ? "" : disturbed == 1 ? " (one disturbed reading measured again)"
                                                           : " (disturbed readings measured again)");
        info.push_back(b);
        if (disturbed > 0 && lane_split(4) >= 4)
            std::fprintf(stderr, "mm3: the CPU lane's timing at load stayed disturbed (the PC busy?): every expert outside "
                                 "VRAM is staged this session - restart when the PC is quiet, or MM3_LANE_SPLIT=0\n");
    }

    // RAM a load leaves free (the OS, the page cache's working set, the engine's own threads): a tenth of the RAM,
    // 4 to 16 GB (8 when the RAM is not known)
    static double ram_headroom() {
        const double t = (double) os::ram_total();
        return t > 0 ? std::min(16.0, std::max(4.0, t / 10.0 / (1u << 30))) * (1u << 30) : 8.0 * (1u << 30);
    }

    // a layer's dense tensors' bytes in VRAM (the EXL3 matrices as stored, the F32 / BF16 vectors and matrices as F32),
    // its routed experts left out
    size_t dense_bytes(int li) const {
        const std::string pre = P + "layers." + std::to_string(li) + ".";
        size_t b = 0;
        for (auto it = pk.t.lower_bound(pre); it != pk.t.end() && it->first.compare(0, pre.size(), pre) == 0; ++it) {
            if (it->first.find(".experts.") != std::string::npos) continue;
            const Tensor& x = it->second;
            // (a GGUF's Q8_0 indexer projections: kept as stored where q8_index takes them, else F32 on the device; its
            //  other block types as stored)
            const bool q8f = x.dtype == "Q8_0" && it->first.find(".index_") != std::string::npos && !q8_index(it->first);
            const bool float_vector = x.dtype == "BF16" || x.dtype == "F16";
            b += float_vector && (!native || x.shape.size() == 1) ? 2 * x.bytes : q8f ? x.bytes / 34 * 128 : x.bytes;
        }
        return b;
    }
    // whether an indexer projection stays in its Q8_0 blocks on the device (a GGUF's [n][dim] Q8_0 rows)
    bool q8_index(const std::string& name) const {
        const Tensor* t = pk.get(name);
        return gguf && t != nullptr && t->dtype == "Q8_0" && t->shape.size() == 2 && t->shape[1] == g.dim &&
               g.dim % 32 == 0 && t->bytes == (uint64_t) t->shape[0] * (uint64_t) (g.dim / 32) * 34u;
    }
    // the decode's indexer projections (a token's or a window's T rows of xn -> iq, ik): both Q8_0 matrices in one
    // launch, else each as it is held
    bool index_proj(const Layer& l, Dev& d, int T, cudaStream_t s) {
        const int nq = g.ix_heads * g.ix_dim;
        if (native) {
            x3::MvJob j[2];
            j[0].m = &l.niq; j[0].xf = d.xn; j[0].y = d.iq;
            j[1].m = &l.nik; j[1].xf = d.xn; j[1].y = d.ik;
            for (auto& v : j) { v.ncol = T; v.xf_cs = g.dim; v.y_cs = v.m->n; }
            return x3::mv(j, 2, s);
        }
        if (l.wiq8 != nullptr) {
            k::rows_dot_q8(l.wiq8, nq, l.wik8, g.ix_dim, g.dim, d.xn, T, d.iq, d.ik, s);
            return true;
        }
        k::rows_dot(l.wiq, nq, g.dim, d.xn, T, d.iq, s);
        k::rows_dot(l.wik, g.ix_dim, g.dim, d.xn, T, d.ik, s);
        return true;
    }
    // the side stream's branch: it waits for d.s's work so far (fork, its stream returned - d.s without one, or `use`
    // false), d.s for the branch's (join).  (A token's host-free layer keeps one stream: measured on the V100s, the
    // two cross-stream waits a branch costs outweighed what the branch hid; a window's or a layer whose routing the
    // host waits for gains.)
    cudaStream_t fork(Dev& d, bool use = true) {
        if (d.s2 == nullptr || !use) return d.s;
        cudaEventRecord(d.fev, d.s);
        cudaStreamWaitEvent(d.s2, d.fev, 0);
        return d.s2;
    }
    void join(Dev& d, bool use = true) {
        if (d.s2 == nullptr || !use) return;
        cudaEventRecord(d.jev, d.s2);
        cudaStreamWaitEvent(d.s, d.jev, 0);
    }
    // the routing the host reads (n ids) names real experts: the router's kernel leaves 0x7fffffff where a token's
    // logits are NaN (no expert compares better), and the host indexed its per-expert tables with it - an access
    // violation with nothing in the log (#672: Windows, an RTX 4070 Ti SUPER + 5070 Ti, the first request).  Such a
    // request now ends with an error that says where.
    bool routes_ok(const int* top, int n, size_t li, std::string& err) const {
        for (int i = 0; i < n; ++i)
            if (top[i] < 0 || top[i] >= g.n_exp) {
                const cudaError_t e = cudaGetLastError();
                err = "layer " + std::to_string(li) + ": the router picked expert " + std::to_string(top[i]) +
                      " of " + std::to_string(g.n_exp) + " (row " + std::to_string(i / std::max(1, g.k_exp)) +
                      ") - the layer's input holds NaN/Inf or its routing did not run (last CUDA error: " +
                      cudaGetErrorString(e) + ")";
                return false;
            }
        return true;
    }
    // a layer's expert blob (the plan's estimate before expert_layout reads it): EXL3's three trellises with their
    // scales, a GGUF's three matrices aligned
    size_t blob_estimate(int li) const {
        if (native) return sizeof(ne::Header) + nf::record_stride(g.dim, g.ff);
        const std::string E = P + "layers." + std::to_string(li) + ".block_sparse_moe.experts.0.";
        if (gguf) {
            size_t b = 0;
            for (const char* r : {"w1", "w3", "w2"})
                if (const Tensor* t = pk.get(E + r + ".weight")) b += up(t->bytes);
            return b;
        }
        const Tensor* t = pk.get(E + "w1.trellis");
        return t ? (size_t) (3.0 * (double) t->bytes * 1.02) : 0;
    }

    // plan_memory: the context (o.max_ctx), KV streaming's ring and the staging slots that fit each GPU and the RAM.
    // Per GPU: its layers' dense weights, the scratch of a chunk, the staging slots, the output head (the last GPU),
    // the DSpark draft and the vision tower (estimates), then per position its layers' K / V rows (the ring's with KV
    // streaming) and index keys - with reserve_gb and 1 GB for the experts' first slots kept free.  The host cache of
    // KV streaming against the RAM this process can still get, less ram_headroom().
    bool plan_memory(int& kv_ring, std::string& err) {
        const double GB = (double) (1u << 30);
        const size_t kvpos = k::kv_bytes(o.kv_k, o.kv_v, (size_t) g.n_kv);   // a layer's K / V of one position
        const size_t idxpos = (size_t) g.ix_dim * 2;                         // ... and its index key
        const double reserve = (o.reserve_gb < 0 ? 1.0 : o.reserve_gb) * GB + 1.0 * GB;
        size_t blob_max = 0;
        for (int li = 0; li < g.n_layers; ++li)
            if (g.moe[(size_t) li]) blob_max = std::max(blob_max, blob_estimate(li));
        auto fixed_of = [&](size_t di, int slots) {
            double b = 0;
            for (int li : D[di].layers) b += (double) dense_bytes(li);
            const double T = (double) o.chunk;
            b += T * g.dim * 4 * 9 + T * g.n_head * g.hd * 6 + T * std::max({g.ff_dense, g.ff_shared, g.ff}) * 8 +
                 T * g.k_exp * (double) g.dim * 4 + T * g.k_exp * (double) g.ff * 2;   // (the chunk's scratch)
            b += (o.native_stage_prefix && di==0 ? 0.0 : (double) slots * (double) blob_max) + 0.2 * GB; // staging and window buffers
            if (di + 1 == D.size()) {                                               // (the output head, its logits)
                const Tensor* h = pk.get(gguf || native ? "language_model.lm_head.weight" : "language_model.lm_head.trellis");
                b += (h ? (double) h->bytes : 0.0) + 32.0 * g.vocab * 4;
                if (o.dspark) b += 3.0 * GB;
            }
            // (the vision tower: no room - it stays in host memory, a picture's room lent by the staging slots)
            if (gguf) b += (double) gg::scratch_bytes();   // (the GGUF kernels' activations, FP16 bands, cuBLAS)
            if (gguf) b += 4.0 * g.ix_heads * g.ix_dim * g.dim;   // (wtmp: a Q8_0 indexer projection in F32)
            if (native) b += (D[di].s2 ? 2.0 : 1.0) * nd::scratch_bytes(std::max({g.dim, g.ff_dense, g.ff_shared, g.n_head * g.hd}), false);
            if (native) {
                if (!(o.native_stage_prefix && di==0)) b += ne::scratch_bytes(g.dim, g.ff, o.native_primary_arithmetic=="fp4" ? ne::Arithmetic::NativeFp4 : ne::Arithmetic::Dp4a);
            } else if (x3::blas_wanted()) b += 0.2 * GB;
            return b;
        };
        auto kv_of = [&](size_t di, int ctx, int ring) {   // (--parallel: a cache for each conversation)
            const double n = (double) D[di].layers.size() * (double) std::max(1, o.seqs);
            return n * ((double) (ring > 0 ? ring : ctx) * (double) kvpos + (double) ctx * (double) idxpos);
        };
        auto fits = [&](int ctx, int ring, int slots, std::string* worst) {
            bool ok = true;
            for (size_t di = 0; di < D.size(); ++di) {
                const double need = fixed_of(di, slots) + kv_of(di, ctx, ring) + reserve;
                if (need > (double) D[di].free0) {
                    ok = false;
                    if (worst)
                        *worst = "CUDA" + std::to_string(D[di].id) + " needs " + std::to_string(need / GB).substr(0, 5) +
                                 " GB, has " + std::to_string((double) D[di].free0 / GB).substr(0, 5) + " GB free";
                }
            }
            return ok;
        };
        const int want_ctx = o.max_ctx;
        int ctx = o.max_ctx, ring = kv_ring, slots = o.staging;
        std::string why;
        if (!fits(ctx, ring, slots, &why) && ring == 0 && ctx > 32768 && o.seqs <= 1) {
            ring = 32768;   // (KV streaming: the last 32768 positions a layer in VRAM, all of them in host memory)
            info.push_back("KV streaming on: the whole context's K / V do not fit beside the model (" + why + ")");
        }
        while (!fits(ctx, ring, slots, &why) && ctx > 8192) {
            ctx = std::max(8192, ctx / 2 / 4096 * 4096);
            if (ring >= ctx) ring = 0;
        }
        while (!fits(ctx, ring, slots, &why) && slots > 16) slots = std::max(16, slots / 2);
        if (!fits(ctx, ring, slots, &why)) {
            err = "not enough GPU memory for MiniMax-M3.1's dense part at 8K positions (" + why +
                  "): more GPUs, or a smaller --chunk";
            return false;
        }
        // KV streaming's host cache: every position of every layer, pinned
        if (ring > 0) {
            const double per = (double) g.n_layers * (double) kvpos;
            const double avail = (double) os::ram_avail() - ram_headroom();
            if (os::ram_avail() > 0 && (double) ctx * per > avail) {
                int c2 = (int) std::max(0.0, std::floor(avail / per / 4096.0) * 4096.0);
                if (c2 < 8192) {
                    err = "KV streaming: " + std::to_string((int) ((double) ctx * per / GB + 0.5)) + " GB of pinned RAM "
                          "for a context of " + std::to_string(ctx) + " positions, " +
                          std::to_string((int) std::max(0.0, avail / GB)) + " GB to spare: a smaller --max-context";
                    return false;
                }
                info.push_back("KV streaming: the RAM holds " + std::to_string(c2) + " positions' K / V, not " +
                               std::to_string(ctx));
                ctx = c2;
                if (ring >= ctx) ring = 0;
            }
        }
        if (ctx != want_ctx)
            info.push_back("context " + std::to_string(ctx) + " positions (asked " + std::to_string(want_ctx) +
                           "): what this PC's GPU memory and RAM hold" +
                           (o.seqs > 1 ? " with " + std::to_string(o.seqs) + " conversations' caches (--parallel)"
                                       : std::string()));
        if (slots != o.staging)
            info.push_back("staging slots " + std::to_string(slots) + " a GPU (asked " + std::to_string(o.staging) +
                           "): GPU memory");
        o.max_ctx = ctx;
        o.staging = slots;
        if (ring > 0) ring = std::max({4096, o.chunk, (ring + g.blk - 1) / g.blk * g.blk});
        kv_ring = ring >= o.max_ctx ? 0 : ring;
        return true;
    }

    bool load(const Options& opt, std::string& err) {
        const double t0 = now();
        o = opt;
        o.seqs = std::max(1, std::min(o.seqs, 8));
        if (o.seqs > 1) {   // (--parallel: one conversation's machinery left out, each with a NOTE)
            const std::string par = "--parallel " + std::to_string(o.seqs) + ": ";
            if (o.dspark) {
                o.dspark = false;
                info.push_back("NOTE: " + par + "no DSpark draft - the conversations decode together without it");
            }
            if (o.kv_resident > 0) {
                o.kv_resident = 0;
                info.push_back("NOTE: " + par + "no KV streaming - each conversation's K / V in VRAM (a shorter "
                                              "context where they do not fit)");
            }
        }
        if (!pk.load(o.pack, err) || !g.read(pk, err)) return false;
        gguf = pk.str("format") == "gguf";
        native = pk.str("format") == "native";
        if(o.native_helper_dispatch!="worker" && o.native_helper_dispatch!="queue") {err="native helper dispatch must be worker or queue";return false;}
        if(o.native_helper_dispatch=="queue" && (!native || o.seqs!=1 || o.tp || o.gpus.size()!=1 || o.native_helper<0)) {
            err="queued helper requires single-sequence single-primary native serving and a helper";return false;
        }
        if (o.native_stage_prefix && !native_stage_chunk_supported(o.chunk)) {
            err="native stage requires chunk between 4 and 16";return false;
        }
        if (native) {
            if (g.dim != 6144 || g.ff != 3072 || g.n_layers != 60 || g.n_exp != 128 || g.k_exp != 4) {
                err = "native MiniMax execution requires the selected M3.1-Flash geometry";
                return false;
            }
            info.push_back("weights: official NVFP4 codes/E4M3 block scales with reciprocal global scale; "
                           "native MXFP8 dense and BF16 head; primary expert arithmetic " + o.native_primary_arithmetic);
            // Initial native execution is the selected text path.
            o.vision = false;
            if(!o.native_cooperative) { o.cpu_threads=-1; o.native_helper=-1; }
            if(o.cpu_threads==-2) o.cpu_threads=CpuLane::auto_workers();
            if(o.chunk>kWin) { err="native cooperative execution currently requires --chunk 16 or smaller"; return false; }
            if(!o.pin_pool || !o.fast_decode || (o.native_cooperative && o.cpu_threads<0)) {
                err="native cooperative execution requires its complete host bank, grouped dispatch and CPU lane";
                return false;
            }
            if(o.gpus.size()>1 && !o.native_stage_prefix) {err="native cooperative execution uses one primary GPU and the separate expert helper";return false;}
            if(o.native_stage_prefix && (g.moe.size()!=60 || g.moe[0] || g.moe[1] || g.moe[2] ||
                std::any_of(g.moe.begin()+3,g.moe.end(),[](auto m){return !m;}))) {
                err="native stage requires three dense layers and 57 suffix MoE layers";return false;
            }
            for(const auto& file:pk.files) {
                const int fd=::open((pk.model_dir+"/"+file).c_str(),O_RDONLY|O_CLOEXEC);
                if(fd<0) { err="cannot open native source for bounded cache retirement: "+file; return false; }
                native_source_fd.push_back(fd);
            }
        }
        if (gguf) {   // (the prompt GEMMs' backend; hipBLASLt's measured kernels kept beside the pack)
            const char* ls = ::mm3::env("MM3_LT_STORE");
            gg::set_lt_store(ls != nullptr ? ls : (o.pack + "/lt-picks.txt").c_str());
            gg::set_gemm(o.prefill_gemm > 0 ? 1 : 0);
        }
        if (gguf) {
            const Tensor* emb = pk.get(P + "embed_tokens.weight");
            if (emb == nullptr || gg::host_row_bytes(emb->dtype, g.dim) * (size_t) g.vocab != emb->bytes) {
                err = "the GGUF's token embedding: " + (emb ? emb->dtype + " of " + std::to_string(emb->bytes) + " bytes"
                                                            : std::string("missing")) + ", not [vocab][dim] in a type read here";
                return false;
            }
            info.push_back("weights: GGUF " + pk.str("quant_name", "?") + " (dense " + pk.str("gguf.dense", "?") +
                           ", experts " + pk.str("gguf.experts", "?") + ")");
            if (o.dspark && !pk.has("dspark.num_hidden_layers")) {
                // (the DSpark draft is the EXL3 repo's dspark/: tools/mm3/pack_gguf.py indexes it from --companions)
                info.push_back("NOTE: DSpark drafting: this GGUF pack has no draft (tools/mm3/pack_gguf.py --companions) "
                               "- off");
                o.dspark = false;
            }
        }
        int ndev = 0;
        CK(cudaGetDeviceCount(&ndev));
        if (native && o.gpus.empty()) o.gpus = {0};
        if (o.gpus.empty())
            for (int i = 0; i < ndev; ++i) o.gpus.push_back(i);
        for (int id : o.gpus) {
            if (id < 0 || id >= ndev) {
                err = "no CUDA" + std::to_string(id);
                return false;
            }
            Dev d;
            d.id = id;
            if(native) {
                cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop,id));
                const bool expert_owner = !o.native_stage_prefix || id==o.gpus.back();
                if(expert_owner) native_primary_fp4_capable = o.seqs==1 && prop.major==12 && prop.minor==0 &&
                    std::strstr(prop.name,"RTX 5090")!=nullptr;
                if(expert_owner && o.native_primary_arithmetic_given && !native_primary_fp4_capable) {
                    err="native primary arithmetic control requires a single-sequence RTX 5090 sm_120 primary";return false;
                }
                native_primary_current=o.native_primary_arithmetic;
            }
            CK(cudaSetDevice(id));
            CK(cudaStreamCreateWithFlags(&d.s, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&d.hop, cudaEventDisableTiming));
            if(native) {
                CK(cudaEventCreate(&d.native_prefix_begin));
                CK(cudaEventCreate(&d.native_prefix_end));
            }
#if !defined(STRATA_USE_HIP)
            if (const char* sv = ::mm3::env("MM3_SIDE"); sv == nullptr || std::atoi(sv) != 0) {
                CK(cudaStreamCreateWithFlags(&d.s2, cudaStreamNonBlocking));
                CK(cudaEventCreateWithFlags(&d.fev, cudaEventDisableTiming));
                CK(cudaEventCreateWithFlags(&d.jev, cudaEventDisableTiming));
            }
#endif
            CB(cublasCreate(&d.blas));
            CB(cublasSetStream(d.blas, d.s));
            CB(cublasSetMathMode(d.blas, CUBLAS_PEDANTIC_MATH));   // F32 GEMMs in F32 (no TF32)
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            d.free0 = fr;
            D.push_back(std::move(d));
        }
        // the layers over the GPUs by what each needs against each GPU's free memory: a layer's dense weights, its routed
        // experts and its caches, the last GPU's room less the output head (and the DSpark draft) - so every GPU keeps
        // about the same share of its memory over (the dense layers hold no experts: by layer count the first GPU kept
        // GBs free while the others' experts went to the CPU lane).  MAYA_MM3_SPLIT_OLD=1: by layer count and free memory
        {
            const bool old = ::mm3::env("MM3_SPLIT_OLD") != nullptr;
            std::vector<double> w((size_t) g.n_layers, 1.0), cap(D.size());
            const double ctx = (double) o.max_ctx, ns = (double) std::max(1, o.seqs);
            const double kvr = o.kv_resident > 0 ? std::min(ctx, (double) o.kv_resident) : ctx;
            for (int li = 0; li < g.n_layers && !old; ++li)
                w[(size_t) li] = (double) dense_bytes(li) +
                                 (g.moe[(size_t) li] ? (double) g.n_exp * (double) blob_estimate(li) : 0.0) +
                                 ns * (kvr * (double) k::kv_bytes(o.kv_k, o.kv_v, (size_t) g.n_kv) + ctx * g.ix_dim * 2.0);
            double total = 0, wsum = 0;
            for (size_t i = 0; i < D.size(); ++i) {
                cap[i] = (double) D[i].free0;
                if (!old && i + 1 == D.size()) {
                    const Tensor* h = pk.get(gguf || native ? "language_model.lm_head.weight" : "language_model.lm_head.trellis");
                    cap[i] -= (h ? (double) h->bytes : 0.0) + 32.0 * g.vocab * 4 + (o.dspark ? 3.0 * (1u << 30) : 0.0);
                }
                cap[i] = std::max(cap[i], 0.05 * (double) D[i].free0);
                total += cap[i];
            }
            for (double x : w) wsum += x;
            double acc = 0, wacc = 0;
            size_t di = 0;
            L.assign((size_t) g.n_layers, {});
            const auto& sp = o.layer_split;
            if (!sp.empty() && (sp.size() + 1 != D.size() || !std::is_sorted(sp.begin(), sp.end()) || sp.front() < 1 ||
                                sp.back() >= g.n_layers || std::adjacent_find(sp.begin(), sp.end()) != sp.end())) {
                err = "--layer-split: " + std::to_string(D.size() - 1) + " increasing first layers in 1 .. " +
                      std::to_string(g.n_layers - 1) + " (one for each GPU after the first)";
                return false;
            }
            for (int li = 0; li < g.n_layers; ++li) {
                if (sp.empty()) {
                    const double mid = (wacc + 0.5 * w[(size_t) li]) / wsum * total;
                    wacc += w[(size_t) li];
                    while (di + 1 < D.size() && acc + cap[di] < mid) acc += cap[di++];
                } else {
                    while (di < sp.size() && li >= sp[di]) ++di;
                }
                L[(size_t) li].dev = (int) di;
                D[di].layers.push_back(li);
            }
        }
        inv_freq.resize((size_t) g.n_rot / 2);
        k::rope_inv_freq(g.theta, g.n_rot, inv_freq.data());
        inv_dev.resize(D.size());
        if (const char* sv = ::mm3::env("MM3_ATTN_SPLIT")) attn_splits = std::max(1, std::atoi(sv));
        attn_splits = std::min(attn_splits, g.topk);
        const int T = o.chunk;
        // KV streaming (--kv-resident): the VRAM caches a ring of the last kv_ring positions (whole blocks, 4096 and a
        // prompt chunk at least), the rest of the context read from host memory; 0: all in VRAM
        int kv_ring = 0;
        if (o.kv_resident > 0) {
            kv_ring = std::max({4096, o.chunk, (o.kv_resident + g.blk - 1) / g.blk * g.blk});
            if (kv_ring >= o.max_ctx) kv_ring = 0;
        }
        // the context this PC holds: each GPU's dense weights, scratch, staging slots and caches against its free memory,
        // the KV streaming host cache against the RAM - too little turns KV streaming on, then gives a shorter context,
        // then fewer staging slots, each with a NOTE; an error only when even 8K positions do not fit
        if (!plan_memory(kv_ring, err)) return false;
        if(o.native_stage_prefix && (kv_ring || o.staging!=8)) {err="native stage memory plan must retain eight suffix staging slots and resident KV";return false;}
        const size_t nblk = (size_t) (o.max_ctx + g.blk - 1) / g.blk;
        // the dense part, caches and scratch, GPU by GPU
        for (size_t di = 0; di < D.size(); ++di) {
            Dev& d = D[di];
            CK(cudaSetDevice(d.id));
            const size_t n0 = owned.size(), up0 = g_up;
            const int dv = d.id;
            auto A = [&](auto*& p, size_t n) { return dalloc(dv, p, n, err); };
            const int ffmax = std::max({g.ff_dense, g.ff_shared, g.ff});
            const int rows = T * g.k_exp;
            d.nblk = nblk;
            if (!A(d.h, (size_t) T * g.dim) || !A(d.xn, (size_t) T * g.dim) || !A(d.q, (size_t) T * g.n_head * g.hd) ||
                !A(d.k, (size_t) T * g.n_kv * g.hd) || !A(d.v, (size_t) T * g.n_kv * g.hd) ||
                !A(d.iq, (size_t) T * g.ix_heads * g.ix_dim) || !A(d.ik, (size_t) T * g.ix_dim) ||
                !A(d.bs, (size_t) g.ix_heads * T * nblk) || !A(d.att, (size_t) T * g.n_head * g.hd) ||
                !A(d.ao, (size_t) T * g.dim) || !A(d.g, (size_t) T * ffmax) || !A(d.u, (size_t) T * ffmax) ||
                !A(d.logit, (size_t) T * g.n_exp) || !A(d.w, (size_t) T * g.k_exp) ||
                !A(d.sh, (size_t) T * g.dim) || !A(d.out, (size_t) T * g.dim) ||
                !A(d.xn16, (size_t) T * g.dim) || !A(d.pos, (size_t) T) ||
                !A(d.kept, (size_t) g.ix_heads * T * g.topk) || !A(d.top, (size_t) T * g.k_exp) ||
                (!native && (!A(d.y, (size_t) rows * g.dim) || !A(d.att16, (size_t) T * g.n_head * g.hd) ||
                 !A(d.h16, (size_t) std::max((size_t) rows * g.ff, (size_t) T * ffmax)) ||
                 !A(d.rowidx, (size_t) rows) || !A(d.invrow, (size_t) rows) || !A(d.bounds, (size_t) g.n_exp + 1))) ||
                !A(inv_dev[di], inv_freq.size()))
                return false;
            CK(cudaMemcpy(inv_dev[di], inv_freq.data(), inv_freq.size() * 4, cudaMemcpyHostToDevice));
            // before sm_80 (V100, T4, RTX 20): the prompt path's busy matrices as cuBLAS FP16 GEMMs over their decoded
            // weights - a handle and 160 MB of scratch for this GPU's stream (none: the tensor-core tiles take them)
            if (native) {
                const int nk = std::max({g.dim, g.ff_dense, g.ff_shared, g.n_head * g.hd});
                for (cudaStream_t s : {d.s, d.s2}) {
                    if (!s) continue;
                    uint8_t* buffer = nullptr;
                    const size_t nb = nd::scratch_bytes(nk, false);
                    if (!A(buffer, nb) || !nd::prepare(s, nk, buffer, nb, false)) {
                        err = "native dense scratch/handle initialization failed on CUDA" + std::to_string(dv);
                        return false;
                    }
                }
                if (!(o.native_stage_prefix && di==0)) {
                uint8_t* buffer = nullptr;
                const auto arithmetic=o.native_primary_arithmetic=="fp4" ? ne::Arithmetic::NativeFp4 : ne::Arithmetic::Dp4a;
                const size_t nb = ne::scratch_bytes(g.dim, g.ff,arithmetic);
                if (!nb || !A(buffer, nb) || !ne::prepare(d.s, g.dim, g.ff, buffer, nb,arithmetic)) {
                    err = "native expert scratch initialization failed";
                    return false;
                }
                native_primary_arena_bytes=nb;
                info.push_back("native primary arithmetic="+o.native_primary_arithmetic+" selected arena bytes="+std::to_string(nb)+
                               " dp4a bytes="+std::to_string(ne::scratch_bytes(g.dim,g.ff,ne::Arithmetic::Dp4a))+
                               " fp4 bytes="+std::to_string(ne::scratch_bytes(g.dim,g.ff,ne::Arithmetic::NativeFp4)));
                }
            } else if (gguf) {   // (a GGUF's kernels: their activations' rows, FP16 weight bands and cuBLAS, before the weights)
                if (!gg::prepare(d.s)) {
                    err = "CUDA" + std::to_string(dv) + ": the GGUF kernels' scratch did not fit";
                    return false;
                }
                acct[dv][kAAct] += gg::scratch_bytes();
            } else if (x3::blas_wanted()) {
                if (x3::blas_enable(d.s, 160)) acct[dv][kAAct] += x3::blas_bytes(d.s);
                else info.push_back("CUDA" + std::to_string(dv) + ": no VRAM for the prompt path's FP16 GEMMs (tiles instead)");
            }
            for (int li : d.layers) {
                Layer& l = L[(size_t) li];
                const std::string N = P + "layers." + std::to_string(li) + ".";
                const std::string AT = N + "self_attn.";
                size_t n = 0;
                std::vector<void*> ow;
                l.moe = g.moe[(size_t) li] != 0;
                l.in1 = upload_f32(pk, N + "input_layernorm.weight", norm_add(), ow, n, err);
                l.post1 = upload_f32(pk, N + "post_attention_layernorm.weight", norm_add(), ow, n, err);
                l.qn1 = upload_f32(pk, AT + "q_norm.weight", norm_add(), ow, n, err);
                l.kn1 = upload_f32(pk, AT + "k_norm.weight", norm_add(), ow, n, err);
                l.iqn1 = upload_f32(pk, AT + "index_q_norm.weight", norm_add(), ow, n, err);
                l.ikn1 = upload_f32(pk, AT + "index_k_norm.weight", norm_add(), ow, n, err);
                if (native) {
                    l.niq = upload_mat(pk, AT + "index_q_proj", ow, err);
                    l.nik = upload_mat(pk, AT + "index_k_proj", ow, err);
                } else if (q8_index(AT + "index_q_proj.weight") && q8_index(AT + "index_k_proj.weight")) {
                    l.wiq8 = upload_raw(pk, AT + "index_q_proj.weight", ow, err);
                    l.wik8 = upload_raw(pk, AT + "index_k_proj.weight", ow, err);
                    if (!err.empty()) return false;
                    if (d.wtmp == nullptr && !A(d.wtmp, (size_t) std::max(g.ix_heads, 1) * g.ix_dim * g.dim)) return false;
                } else {
                    l.wiq = upload_f32(pk, AT + "index_q_proj.weight", 0.0f, ow, n, err);
                    l.wik = upload_f32(pk, AT + "index_k_proj.weight", 0.0f, ow, n, err);
                }
                if (!err.empty()) return false;
                l.q = upload_mat(pk, AT + "q_proj", ow, err);
                l.k = upload_mat(pk, AT + "k_proj", ow, err);
                l.v = upload_mat(pk, AT + "v_proj", ow, err);
                l.o = upload_mat(pk, AT + "o_proj", ow, err);
                if (!err.empty()) return false;
                if (l.moe) {
                    const std::string M = N + "block_sparse_moe.";
                    l.gate = upload_f32(pk, M + "gate.weight", 0.0f, ow, n, err);
                    l.bias = upload_f32(pk, M + "e_score_correction_bias", 0.0f, ow, n, err);
                    if (!err.empty()) return false;
                    l.sg = upload_mat(pk, M + "shared_experts.gate_proj", ow, err);
                    l.su = upload_mat(pk, M + "shared_experts.up_proj", ow, err);
                    l.sd = upload_mat(pk, M + "shared_experts.down_proj", ow, err);
                    if (!err.empty() || !expert_layout(l, li, err)) return false;
                    d.slot_bytes = std::max(d.slot_bytes, l.blob);
                    ++st.experts;
                } else {
                    l.mg = upload_mat(pk, N + "mlp.gate_proj", ow, err);
                    l.mu = upload_mat(pk, N + "mlp.up_proj", ow, err);
                    l.md = upload_mat(pk, N + "mlp.down_proj", ow, err);
                    if (!err.empty()) return false;
                }
                for (void* x : ow) {
                    owned.push_back(x);
                    owned_dev.push_back(dv);
                }
                acat = kAKV;
                const size_t kvrows = (size_t) (kv_ring > 0 ? kv_ring : o.max_ctx) * g.n_kv;
                l.sk.resize((size_t) std::max(1, o.seqs));   // (--parallel: a cache each; the first one bound)
                for (Layer::SeqKv& c : l.sk) {
                    uint8_t* kvb = nullptr;
                    if (!A(kvb, k::kv_bytes(o.kv_k, o.kv_v, kvrows))) return false;
                    k::kv_carve(kvb, kvrows, o.kv_k, o.kv_v, o.kv_rot ? 3 : 0, c.kc, c.vc, c.kvq);
                    if (!A(c.ic, (size_t) o.max_ctx * g.ix_dim)) return false;
                }
                l.kc = l.sk[0].kc;
                l.vc = l.sk[0].vc;
                l.ic = l.sk[0].ic;
                l.kvq = l.sk[0].kvq;
                if (kv_ring > 0) {
                    l.kvh.ring = kv_ring;
                    if (!A(l.kvh.tag, (size_t) kv_ring)) return false;
                    CK(cudaMemset(l.kvh.tag, 0xff, (size_t) kv_ring * 4));
                }
                acat = kAAct;
            }
            if (di + 1 == D.size()) {
                head_dev = (int) di;
                if (!A(d.lg, (size_t) kHeadRows * g.vocab)) return false;
                CK(cudaMallocHost(&d.lg_h, (size_t) kHeadRows * g.vocab * sizeof(float)));
                CK(cudaHostAlloc((void**) &d.nuc_c, (size_t) kWin * sizeof(k::CandRow), cudaHostAllocMapped | cudaHostAllocPortable));
                CK(cudaHostAlloc((void**) &d.nuc_s, (size_t) kWin * sizeof(k::RowStats), cudaHostAllocMapped | cudaHostAllocPortable));
                CK(cudaMalloc(&d.nuc_scratch, k::row_stats_scratch(kWin)));
                CK(cudaMalloc(&d.nuc_topk, k::topk_rows_scratch(kWin)));
                std::vector<void*> ow;
                size_t n = 0;
                norm1 = upload_f32(pk, P + "norm.weight", norm_add(), ow, n, err);
                if (!err.empty()) return false;
                head = upload_mat(pk, "language_model.lm_head", ow, err);
                if (!err.empty()) return false;
                for (void* x : ow) {
                    owned.push_back(x);
                    owned_dev.push_back(dv);
                }
            }
            own_since(n0, dv);
            acct[dv][kAW] += g_up - up0;
            if (d.slot_bytes > 0) {
                d.n_slots = o.staging;
                acat = kAPool;
                const bool sok = A(d.stg, (size_t) d.n_slots * d.slot_bytes);
                acat = kAAct;
                if (!sok) return false;
                d.slot_owner.assign((size_t) d.n_slots, -1);
                d.slot_used.assign((size_t) d.n_slots, -1);
                d.slot_pend.assign((size_t) d.n_slots, -1);
                d.slot_iss.assign((size_t) d.n_slots, 0);
                d.slot_foreground.assign((size_t) d.n_slots, 0);
                d.slot_early.assign((size_t) d.n_slots, 0);
                d.slot_early_fresh.assign((size_t) d.n_slots, 0);
                CK(cudaStreamCreateWithFlags(&d.cs, cudaStreamNonBlocking));
                d.slot_ev.resize((size_t) d.n_slots);
                for (auto& e : d.slot_ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
                if (!native) CK(cudaMallocHost(&d.pin, d.slot_bytes));
            }
            CK(cudaMallocHost(&d.plan_h, 16 * sizeof(unsigned long long)));
            CK(cudaMallocHost(&d.pos_h, (size_t) T * sizeof(int)));
            if (!native) CK(cudaMallocHost(&d.top_h, (size_t) T * g.k_exp * sizeof(int)));
            if (L[(size_t) 0].dev == (int) di) CK(cudaMallocHost(&d.emb_h, (size_t) T * g.dim * sizeof(float)));
            if (!A(d.plan_d, 16)) return false;
            if (native && !(o.native_stage_prefix && !o.native_cooperative && di==0)) {
                CK(cudaMallocHost(&d.ready_h, sizeof(resident::Ready)));
                CK(cudaMallocHost(&d.original_h, (size_t)T*g.k_exp*sizeof(int)));
                if (!A(d.ready_d, 1) || !A(d.original_d, (size_t)T*g.k_exp)) return false;
            }
            if (!native) {
                CK(cudaMallocHost(&d.xh_h, (size_t) (2 * g.dim + 4) * sizeof(float)));
                if (!A(d.cpu_d, (size_t) g.dim + 4)) return false;
            }
            // The C-off three-layer prefix contains no expert window at all.
            // Preserve the established cooperative allocations unchanged.
            if (!(native && !o.native_cooperative && o.native_stage_prefix && di==0)) {
                CK(cudaMallocHost(&d.win_rb, win_rb_bytes()));
                CK(cudaMallocHost(&d.win_plan_h, win_plan_bytes()));
                if (!A(d.win_plan_d, win_plan_bytes())) return false;
                if (!native || o.native_cooperative) {
                    CK(cudaHostAlloc(&d.win_cpu, ((size_t) kWin * g.dim + kWin + 4) * 4, cudaHostAllocMapped | cudaHostAllocPortable));
                    std::memset(d.win_cpu,0,((size_t)kWin*g.dim+kWin+4)*4);
                } else CK(cudaMallocHost(&d.owner_plan_h, 65 * win_plan_bytes()));
                if ((!native && !A(d.win_h, (size_t) kWin * 8 * g.ff)) || !A(d.win_part, (size_t) kWin * 8 * g.dim)) return false;
            }
            if (attn_splits > 1 && !A(d.attp, k::attn_part_floats(kWin, g.n_kv, attn_splits))) return false;
        }
        // KV streaming: every layer's whole cache in host memory (one pinned, mapped allocation, carved), written as the
        // VRAM ring is and read where the ring no longer holds a key
        if (kv_ring > 0) {
            const size_t rows = (size_t) o.max_ctx * g.n_kv;
            const size_t nb = k::kv_bytes(o.kv_k, o.kv_v, rows);
            size_t total = 0;
            for (const Layer& l : L) total += l.kvh.ring > 0 ? nb : 0;
            CK(cudaSetDevice(D[0].id));
            if (cudaHostAlloc((void**) &kv_host, total, cudaHostAllocMapped | cudaHostAllocPortable) != cudaSuccess) {
                kv_host = nullptr;
                err = "KV streaming: " + std::to_string(total >> 20) + " MB of pinned host memory not allocated";
                return false;
            }
            size_t off = 0;
            for (Layer& l : L) {
                if (l.kvh.ring <= 0) continue;
                k::kv_carve(kv_host + off, rows, o.kv_k, o.kv_v, o.kv_rot ? 3 : 0, l.kvh.kc, l.kvh.vc, l.kvh.q);
                off += nb;
                acct_kvh[D[(size_t) l.dev].id] += nb;
            }
            st.kv_ring = kv_ring;
            st.kv_host_gb = (double) total / (1u << 30);
        }
        if(native) {
            if(o.native_stage_prefix) {
                CK(cudaSetDevice(D[0].id));
                serial_bridge_bytes=(size_t)o.chunk*g.dim*sizeof(float);
                CK(cudaHostAlloc(&serial_bridge,serial_bridge_bytes,cudaHostAllocPortable));
                for(int i=0;i<2;++i) {
                    CK(cudaHostAlloc(&stage_hidden[i],(size_t)4*g.dim*sizeof(float),cudaHostAllocPortable));
                    CK(cudaEventCreateWithFlags(&stage_ready[i],cudaEventDisableTiming));
                }
            }
            const Tensor* embedding=pk.get(P+"embed_tokens.weight");
            if(!embedding || embedding->dtype!="BF16" || embedding->bytes!=(size_t)g.vocab*g.dim*2) {
                err="native token embedding is not BF16 [vocab,hidden]"; return false;
            }
            const uint8_t* source=pk.data(*embedding);
            try {
                native_embedding=std::make_unique<NativeHostBank>(embedding->bytes,D[head_dev].id,(uint64_t)ram_headroom(),(uint64_t)ram_headroom());
                std::memcpy(native_embedding->data(),source,embedding->bytes);
            } catch(const std::exception& e) { err=e.what(); return false; }
            os::drop_mapped(source,embedding->bytes);
            (void)::posix_fadvise(native_source_fd[embedding->file],(off_t)embedding->off,(off_t)embedding->bytes,POSIX_FADV_DONTNEED);
            pk.resident_embedding=(const uint8_t*)native_embedding->data();
            info.push_back("native resident BF16 embedding: "+std::to_string(native_embedding->capacity())+" host bytes, shared by target and DSpark");
        }
        // the DSpark draft on the head's GPU, before the experts take the memory
        tap_of.assign((size_t) g.n_layers, -1);
        if (o.dspark) {
            Dev& hd = D[(size_t) head_dev];
            CK(cudaSetDevice(hd.id));
            draftm = std::make_unique<Draft>();
            const Tensor* et = pk.get(P + "embed_tokens.weight");
            if (gguf)   // (the GGUF's table in its block type: a row dequantized as ggml does, then BF16 - the draft's dtype)
                draftm->set_emb_rows([this, et](int id, uint16_t* dst) {
                    std::vector<float> r((size_t) g.dim);
                    gg::host_row(et->dtype, pk.data(*et), g.dim, id, r.data());
                    for (int i = 0; i < g.dim; ++i) {
                        uint32_t u;
                        std::memcpy(&u, &r[(size_t) i], 4);
                        dst[i] = (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);   // (round to nearest even)
                    }
                });
            if (!o.dflash2.empty()) {   // (step 6) RadixArk's DFlash2 from its own pack, in DSpark's place
                Pack dp;
                if (!dp.load(o.dflash2, err) ||
                    !draftm->load_dflash2(dp, pk, hd.id, hd.s, hd.blas, &head, (const uint16_t*) pk.data(*et), o.chunk, err))
                    return false;
            } else if (!draftm->load(pk, hd.id, hd.s, hd.blas, &head, (const uint16_t*) pk.data(*et), o.chunk, err))
                return false;
            const auto& tl = draftm->geo().taps;
            if(o.native_stage_prefix && (tl.size()!=5 || tl[0]!=3 || tl[1]!=17 || tl[2]!=31 || tl[3]!=45 || tl[4]!=59)) {err="three-layer stage requires suffix-owned official draft taps";return false;}
            for (size_t j = 0; j < tl.size(); ++j) {
                if (tl[j] < 0 || tl[j] >= g.n_layers) {
                    err = "dspark: a tap at layer " + std::to_string(tl[j]);
                    return false;
                }
                tap_of[(size_t) tl[j]] = (int) j;
            }
            acat = kADraft;
            const bool tok = dalloc(hd.id, taps, tl.size() * (size_t) o.chunk * g.dim, err);
            acat = kAAct;
            if (!tok) return false;
            info.push_back(std::string(draftm->geo().dflash2 ? "DFlash2" : "DSpark") + " draft: " +
                           std::to_string(draftm->bytes() >> 20) + " MB on CUDA" + std::to_string(hd.id) + " (head " +
                           std::to_string(head.n) + "x" + std::to_string(head.k) + " fmt " + std::to_string(head.fmt) +
                           "), block " + std::to_string(draftm->geo().block) + ", taps " +
                           std::to_string(tl.size()));
        }
        // the vision tower, also before the experts take the memory (on by default: a pack without one runs text only)
        if (o.vision && o.vision_auto && !Vision::in_pack(pk)) {
            o.vision = false;
            info.push_back("vision: the pack has no vision tower - text only");
        }
        if (o.vision) {
            if (!Vision::in_pack(pk)) {
                err = "--vision: the pack has no vision tower (pack it again with tools/mm3/pack.py)";
                return false;
            }
            if (o.vision_gpu < 0 || o.vision_gpu >= (int) D.size()) {
                err = "--vision-gpu " + std::to_string(o.vision_gpu) + ": not one of the " + std::to_string(D.size()) +
                      " GPUs";
                return false;
            }
            vis = std::make_unique<Vision>();
            if (!vis->load(pk, D[(size_t) o.vision_gpu].id, err)) {
                if (o.vision_must) return false;
                info.push_back("NOTE: vision off: the tower did not load on CUDA" +
                               std::to_string(D[(size_t) o.vision_gpu].id) + " (" + err + ") - text only");
                std::fprintf(stderr, "maya-mm3: %s\n", info.back().c_str());
                err.clear();
                cudaGetLastError();
                vis.reset();
                o.vision = false;
            }
            if (vis) {
                const size_t vi = (size_t) o.vision_gpu;
                vis->set_lender([this, vi](size_t need) { return lend_slots(D[vi], need); },
                                [this, vi]() { drop_slots(D[vi]); });
                info.push_back("vision tower: on demand - its " + std::to_string(vis->host_bytes() >> 20) +
                               " MB in host memory (the pack's mapping), a picture at a time on CUDA" +
                               std::to_string(vis->device()) + " (up to " + std::to_string(vis->room_bytes() >> 20) +
                               " MB, given back after it), up to " + std::to_string(vis->geo().max_pixels) +
                               " pixels a picture");
                info.push_back("NOTE: pictures: the vision tower is not kept in VRAM - each picture copies it in, the "
                               "first one slower (its weights read from the SSD)");
            }
        }
        // the profile: routings [layer][expert] (a wrong size is ignored)
        prof.assign((size_t) g.n_layers * g.n_exp, 0);
        bool have_prof = false;
        if (!o.profile.empty()) {
            if (std::FILE* f = std::fopen(o.profile.c_str(), "rb")) {
                std::fseek(f, 0, SEEK_END);
                const long n = std::ftell(f);
                std::fseek(f, 0, SEEK_SET);
                if (n == (long) (prof.size() * 4) && std::fread(prof.data(), 4, prof.size(), f) == prof.size())
                    have_prof = true;
                std::fclose(f);
            }
            info.push_back(have_prof ? "expert profile " + o.profile
                                     : "expert profile " + o.profile + ": not read (missing or another geometry)");
            if (native && (!have_prof || std::none_of(prof.begin(), prof.end(), [](uint32_t n){ return n != 0; }))) {
                err="native expert profile is unreadable, has wrong geometry, or contains no demand";return false;
            }
            if (!have_prof) std::fill(prof.begin(), prof.end(), 0u);
        }
        if (native && !have_prof)
            info.push_back("native placement: unprofiled bootstrap; initial identities are not demand ranked");
        if (native && have_prof && (o.native_stage_prefix || D.size()!=1 ||
            (o.resident_cap>=0 && o.resident_cap<g.n_exp) || compressed_host() ||
            o.native_helper<0 || o.native_helper_gb<=0 || o.native_helper>=ndev || o.native_helper==D[0].id)) {
            err="native profiled placement requires one target, separate helper, raw HOST and no per-layer resident cap";return false;
        }
        if (compressed_host()) {
            namespace ss = ::mm3::scale_store;
            Dev& d=D[(size_t)head_dev]; CK(cudaSetDevice(d.id));
            scale_device=d.id;
            if (!dalloc(d.id,scale_fg_scratch,ss::kScratchBytes,err) ||
                !dalloc(d.id,scale_opt_scratch,ss::kScratchBytes,err)) return false;
            scale_service=std::make_unique<ss::Service>(d.id,scale_fg_scratch,d.s,scale_opt_scratch,d.cs,5000,err);
            if (!scale_service->healthy()) { if(err.empty())err=scale_service->error(); return false; }
            std::vector<const Src*> sources;
            for (const auto& l:L) if(l.moe) {
                if(l.blob!=ss::kRawBytes) {err="compressed HOST native geometry mismatch";return false;}
                for(const auto& src:l.src) sources.push_back(&src);
            }
            std::atomic<size_t> next{0}, max_bytes{0}; std::atomic<bool> bad{false};
            std::mutex mu; std::string failure; std::vector<std::thread> workers;
            auto report=[&](const std::string& x) {bad.store(true);std::lock_guard<std::mutex> lock(mu);if(failure.empty())failure=x;};
            try { for(int t=0;t<8;++t) workers.emplace_back([&] {r2::guarded_worker([&] {
                for(size_t i; !bad.load() && (i=next.fetch_add(1))<sources.size();) {
                    const Src& src=*sources[i]; size_t n=0; std::string e;
                    if(!ss::encoded_size_from_scales(src.p[1],src.p[4],src.p[7],n,e)) {report(e);break;}
                    size_t previous=max_bytes.load(); while(previous<n && !max_bytes.compare_exchange_weak(previous,n)) {}
                    for(int j : {1,4,7}) { os::drop_mapped(src.p[j],src.n[j]);
                        if(size_t(src.file[j])<native_source_fd.size()) (void)::posix_fadvise(native_source_fd[src.file[j]],(off_t)src.off[j],(off_t)src.n[j],POSIX_FADV_DONTNEED); }
                }
            },report);}); } catch(const std::exception& e) {report(e.what());} catch(...) {report("scale sizing worker start failed");}
            for(auto& t:workers)t.join();
            if(bad.load() || !max_bytes.load()) {err="scale sizing failed: "+failure;return false;}
            host_stride=max_bytes.load();
            info.push_back("native compressed HOST stride="+std::to_string(host_stride));
        }
        // the routed experts kept in VRAM while the GPU keeps reserve_gb free: by the profile, the most routed of the
        if (native) original_prof = prof;
        if(native && have_prof && !seed_native_demand_prior(prof,err))return false;
        // The profiled native loader currently supports only the single-target,
        // separate-helper geometry. Determine the helper allocation before
        // choosing either bank; do not silently apply a constrained ranking.
        int profiled_helper_capacity = 0;
        std::vector<int> profiled_helper_ids;
        if (native && have_prof) {
            CK(cudaSetDevice(o.native_helper));
            size_t fr=0,total=0; CK(cudaMemGetInfo(&fr,&total));
            const size_t budget=(size_t)(o.native_helper_gb*(1ULL<<30));
            if(fr<budget+(512ULL<<20)) {err="native helper budget leaves less than 512 MiB for scratch/reserve";return false;}
            profiled_helper_capacity=int(budget/nf::record_stride(g.dim,g.ff));
            if(profiled_helper_capacity<1) {err="native helper has no fitting expert records";return false;}
        }
        // GPU's layers first; without one round robin over its layers (expert 0 of every layer, then 1, ...)
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            size_t reserve = (size_t) ((o.reserve_gb < 0 ? 1.0 : o.reserve_gb) * (1u << 30));
            std::vector<int> moe_layers;
            for (int li : d.layers)
                if (L[(size_t) li].moe) moe_layers.push_back(li);
            // every expert of this GPU's layers in VRAM when its staging slots give way to them: then the slots are
            // only a prompt's and a picture's fallback (nothing is staged: every MoE layer host-free) - a few kept
            // (kMinSlots), the rest of their memory to the experts; and (the reserve not asked for) the GPU keeps
            // half a GB free instead of one - what the load has not taken by now is small (the BLAS's workspace, the
            // graphs) - or a picture's room more on the vision tower's GPU.  MAYA_MM3_SLOTS_FIT=0: as before
            if (d.n_slots > kMinSlots && o.resident_cap < 0) {
                const char* sv = ::mm3::env("MM3_SLOTS_FIT");
                size_t all = 0;
                for (int li : moe_layers) all += (size_t) g.n_exp * L[(size_t) li].blob;
                const size_t gain = (size_t) (d.n_slots - kMinSlots) * d.slot_bytes;
                size_t rfit = reserve;
                if (o.reserve_gb < 0) {
                    rfit = (size_t) 1 << 29;
                    if (vis && vis->device() == d.id) rfit += vis->room_bytes();
                    rfit = std::min(rfit, reserve);
                }
                if ((sv == nullptr || std::atoi(sv) != 0) && fr > rfit && all > fr - reserve &&
                    all <= fr - rfit + gain && resize_slots(d, kMinSlots, err)) {
                    CK(cudaMemGetInfo(&fr, &tot));
                    reserve = rfit;
                    info.push_back("CUDA" + std::to_string(d.id) + ": " + std::to_string(kMinSlots) + " staging slots " +
                                   "(asked " + std::to_string(o.staging) + "), " + std::to_string(reserve >> 20) +
                                   " MB kept free: every expert of its layers fits in VRAM");
                }
                if (!err.empty()) return false;
            }
            size_t budget = fr > reserve ? fr - reserve : 0;
            std::vector<int> nres(moe_layers.size(), 0);
            const int cap = o.resident_cap < 0 ? g.n_exp : std::min(o.resident_cap, g.n_exp);
            std::vector<std::vector<int>> chosen(moe_layers.size());
            if (have_prof && native) {
                // Native records have uniform size. Preserve the same primary
                // memory budget and helper allocation as the previous loader,
                // but select the helper before filling the primary complement.
                if (moe_layers.empty()) {err="native profile has no MoE layers";return false;}
                bool moe_demand=false;
                for(int li:moe_layers) for(int e=0;e<g.n_exp;++e)
                    moe_demand |= prof[size_t(li)*g.n_exp+size_t(e)]!=0;
                if(!moe_demand) {err="native expert profile has no MoE demand";return false;}
                const size_t blob=L[size_t(moe_layers[0])].blob;
                for(int li:moe_layers) if(L[size_t(li)].blob!=blob) {err="native profile requires uniform expert records";return false;}
                const size_t total=size_t(moe_layers.size())*size_t(g.n_exp);
                size_t primary_capacity=0,helper_count=0;
                if(!profiled_native_capacities(total,blob,budget,size_t(profiled_helper_capacity),primary_capacity,helper_count)) {
                    err="native profile has invalid expert or helper capacity";return false;
                }
                std::vector<int> devices, primary_ids; std::vector<uint8_t> moe;
                for(const auto& l:L) {devices.push_back(l.dev);moe.push_back(l.moe?1:0);}
                if(!helper_first_rosters(prof,g.n_exp,devices,moe,head_dev,{primary_capacity},helper_count,
                                         g.n_exp,primary_ids,profiled_helper_ids)) {
                    err="native profile cannot fill globally ranked helper and primary rosters";return false;
                }
                info.push_back("native profile placement: global original-demand ranking, helper first="+
                               std::to_string(helper_count)+", primary remainder="+std::to_string(primary_capacity));
                for(int id:primary_ids) {
                    const int li=id/g.n_exp,e=id%g.n_exp;
                    auto it=std::find(moe_layers.begin(),moe_layers.end(),li);
                    if(it==moe_layers.end()) {err="native profile selected an invalid MoE layer";return false;}
                    const size_t i=size_t(it-moe_layers.begin());++nres[i];chosen[i].push_back(e);
                }
                for(auto& v:chosen)std::sort(v.begin(),v.end());
            } else if (have_prof) {
                struct C {
                    uint32_t n;
                    int i, e;
                };
                std::vector<C> c;
                for (size_t i = 0; i < moe_layers.size(); ++i)
                    for (int e = 0; e < g.n_exp; ++e)
                        c.push_back({prof[(size_t) moe_layers[i] * g.n_exp + e], (int) i, e});
                std::stable_sort(c.begin(), c.end(), [](const C& a, const C& b) { return a.n > b.n; });
                for (const C& x : c) {
                    const Layer& l = L[(size_t) moe_layers[(size_t) x.i]];
                    if (nres[(size_t) x.i] < cap && budget >= l.blob) {
                        budget -= l.blob;
                        ++nres[(size_t) x.i];
                        chosen[(size_t) x.i].push_back(x.e);
                    }
                }
                for (auto& v : chosen) std::sort(v.begin(), v.end());
            }
            for (bool grew = !have_prof; grew;) {
                grew = false;
                for (size_t i = 0; i < moe_layers.size(); ++i) {
                    const Layer& l = L[(size_t) moe_layers[i]];
                    if (nres[i] < cap && budget >= l.blob) {
                        budget -= l.blob;
                        ++nres[i];
                        grew = true;
                    }
                }
            }
            for (size_t i = 0; i < moe_layers.size(); ++i) {
                Layer& l = L[(size_t) moe_layers[i]];
                l.res_slot.assign((size_t) g.n_exp, -1);
                l.res_ids.clear();
                acat=kAExpert;
                if (nres[i] > 0 && !dalloc(d.id, l.res_base, (size_t) nres[i] * l.blob, err)) return false;
                acat=kAAct;
                for (int sl = 0; sl < nres[i]; ++sl) {
                    const int e = have_prof ? chosen[i][(size_t) sl] : sl;
                    l.res_slot[(size_t) e] = sl;
                    l.res_ids.push_back(e);
                }
                st.resident += nres[i];
                st.vram_gb += (double) ((size_t) nres[i] * l.blob) / (1u << 30);
            }
        }
        if(o.native_stage_prefix && (D[0].n_slots!=0 || D[1].n_slots!=8)) {
            err="native stage requires no prefix expert staging and eight suffix frames";return false;
        }
        st.experts = 0;
        for (const Layer& l : L)
            if (l.moe) {
                st.experts += g.n_exp;
                st.all_gb += (double) ((size_t) g.n_exp * l.blob) / (1u << 30);
            }
        // Resident uploads retain their staging buffers until a successful join.
        {
            std::vector<r2::ResidentBatch> batches;
            for (Dev& d : D) {
                r2::ResidentBatch batch{&d,0,{}};
                for (int li : d.layers) {
                    const Layer& l=L[(size_t)li];
                    batch.max_bytes=std::max(batch.max_bytes,l.blob);
                    for(size_t slot=0;slot<l.res_ids.size();++slot) batch.jobs.push_back({&l,slot});
                }
                batches.push_back(std::move(batch));
            }
            auto make_ops=[&](r2::ResidentBatch& batch) {
                Dev* dp=static_cast<Dev*>(batch.device);
                r2::ResidentOps ops;
                auto check=[](cudaError_t status,const char* what) {
                    if(status!=cudaSuccess) throw std::runtime_error(std::string(what)+": "+cudaGetErrorString(status));
                };
                ops.set_device=[dp,check]{check(cudaSetDevice(dp->id),"resident set device");};
                ops.allocate_pinned=[](uint8_t** p,size_t n) {
                    if(cudaMallocHost(p,n)==cudaSuccess)return true;
                    (void)cudaGetLastError(); *p=nullptr; return false;
                };
                ops.create_event=[check](int)->void* {
                    cudaEvent_t e=nullptr; check(cudaEventCreateWithFlags(&e,cudaEventDisableTiming),"resident create event"); return e;
                };
                ops.record_event=[dp,check](void* e){check(cudaEventRecord(static_cast<cudaEvent_t>(e),dp->s),"resident event submission");};
                ops.wait_event=[check](void* e){check(cudaEventSynchronize(static_cast<cudaEvent_t>(e)),"resident staging wait");};
                ops.assemble=[&](const r2::ResidentJob& job,uint8_t* dst) {
                    const Layer& l=*static_cast<const Layer*>(job.layer); assemble(l,l.res_ids[job.slot],dst);
                };
                ops.copy=[dp,check](const r2::ResidentJob& job,uint8_t* src,bool async) {
                    const Layer& l=*static_cast<const Layer*>(job.layer);
                    cudaError_t status=async ? cudaMemcpyAsync(l.res_base+job.slot*l.blob,src,l.blob,cudaMemcpyHostToDevice,dp->s)
                                             : cudaMemcpy(l.res_base+job.slot*l.blob,src,l.blob,cudaMemcpyHostToDevice);
                    check(status,async?"resident copy submission":"resident copy");
                };
                ops.drop_source=[&](const r2::ResidentJob& job) {
                    const Layer& l=*static_cast<const Layer*>(job.layer); drop_src(l,l.res_ids[job.slot]);
                };
                ops.join=[dp]{return cudaStreamSynchronize(dp->s)==cudaSuccess;};
                ops.free_pinned=[](uint8_t* p){(void)cudaFreeHost(p);};
                ops.destroy_event=[](void* e){(void)cudaEventDestroy(static_cast<cudaEvent_t>(e));};
                return ops;
            };
            auto start=[](std::vector<std::thread>& threads,auto work) { threads.emplace_back(std::move(work)); };
            if(!r2::load_residents(batches,make_ops,start,retirement_uncertain,err)) return false;
        }
        // Complementary immutable helper records are loaded through one portable
        // staging record, before the CPU/primary-only host bank is allocated.
        if (native && o.native_helper >= 0 && o.native_helper_gb > 0) {
            if (std::any_of(D.begin(), D.end(), [&](const Dev& d) { return d.id == o.native_helper && !(o.native_stage_prefix && &d==&D.front()); })) {
                err = "native helper cannot also own target layers"; return false;
            }
            if (o.native_helper >= ndev) { err = "native helper device unavailable"; return false; }
            const size_t stride = nf::record_stride(g.dim, g.ff);
            CK(cudaSetDevice(o.native_helper));
            size_t fr = 0, total = 0;
            CK(cudaMemGetInfo(&fr, &total));
            size_t budget = (size_t)(o.native_helper_gb * (1ULL << 30));
            if(o.native_stage_prefix) {
                // Charge the constructor's exact workspace and conservative CUDA
                // allocation rounding, retaining at least 1 GiB after allocation.
                const size_t scratch=NativeHelper::planned_bytes(0,true).device_scratch;
                constexpr size_t reserve=(1ULL<<30), rounding=(64ULL<<20);
                if(fr<=reserve+scratch+rounding || (22ULL<<30)<=scratch) {err="native stage helper has insufficient post-prefix free memory";return false;}
                budget=std::min({budget,(size_t)(22ULL<<30)-scratch,fr-reserve-scratch-rounding});
            } else if (fr < budget + (512ULL << 20)) { err = "native helper budget leaves less than 512 MiB for scratch/reserve"; return false; }
            const int capacity = (int)(budget / stride);
            if(capacity<1) {err="native helper has no fitting expert records";return false;}
            if(have_prof && capacity!=profiled_helper_capacity) {err="native helper capacity changed after profile planning";return false;}
            try { helper = std::make_unique<NativeHelper>(o.native_helper, capacity,o.native_helper_dispatch!="queue"); }
            catch (const std::exception& e) { err = e.what(); return false; }
            if(o.native_stage_prefix) {
                size_t ready_free=0,ready_total=0; CK(cudaMemGetInfo(&ready_free,&ready_total));
                const auto charged=helper->bytes();
                if(ready_free<(1ULL<<30) || charged.pool+charged.device_scratch>(22ULL<<30)) {err="native stage helper violates charged capacity or 1 GiB free requirement";return false;}
                info.push_back("stage helper after allocation free="+std::to_string(ready_free)+" before="+std::to_string(fr)+" capacity="+std::to_string(capacity));
            }
            struct Pick { uint32_t score; int li, e; };
            std::vector<Pick> picks;
            if(have_prof) {
                for(int id:profiled_helper_ids) picks.push_back({prof[size_t(id)],id/g.n_exp,id%g.n_exp});
            } else {
                for (int e = 0; e < g.n_exp; ++e) for (int li = 0; li < g.n_layers; ++li)
                    if (L[li].moe && L[li].res_slot[e] < 0) picks.push_back({prof[(size_t)li*g.n_exp+e],li,e});
                std::stable_sort(picks.begin(), picks.end(), [](const Pick& a, const Pick& b) { return a.score > b.score; });
            }
            uint8_t* stage = nullptr;
            CK(cudaHostAlloc(&stage, sizeof(ne::Header)+stride, cudaHostAllocPortable));
            bool good = true;
            for (int i = 0; i < std::min(capacity, (int)picks.size()) && good; ++i) {
                const auto pick = picks[i]; Layer& l = L[pick.li];
                try { assemble(l, pick.e, stage); }
                catch (const std::exception& e) { err = e.what(); good = false; break; }
                const auto* h = (const ne::Header*)stage;
                good = helper->add_expert(pick.li, pick.e, stage+sizeof(ne::Header), stride, h->weight_global, h->input_global, err);
                if (good) { l.helper_ready[pick.e/32] |= 1u << (pick.e%32); drop_src(l,pick.e); }
            }
            // A failed helper copy may still have an uncertain reader of the
            // borrowed portable staging record. Keep it allocated on failure.
            if (good) cudaFreeHost(stage);
            if (!good || !helper->seal(err)) return false;
            const auto bytes = helper->bytes();
            st.resident += std::min(capacity,(int)picks.size());
            st.vram_gb += (double)bytes.pool/(1ULL<<30);
            info.push_back("native helper CUDA" + std::to_string(o.native_helper) + ": " + std::to_string(capacity) +
                           " exclusive RG16 records, " + std::to_string(bytes.pool) + " weight bytes, " +
                           std::to_string(bytes.device_scratch) + " device scratch, " + std::to_string(bytes.pinned) + " pinned bytes");
        }
        if (native && helper && o.native_third >= 0 && o.native_third_gb > 0) {
            // third: one more card's fixed pool for copies of HOST experts (empty until PLACEMENT third fills it)
            if (o.native_third >= ndev || o.native_third == o.native_helper ||
                std::any_of(D.begin(), D.end(), [&](const Dev& d) { return d.id == o.native_third; })) {
                err = "native third device must be a CUDA device other than the primary's and the helper's"; return false;
            }
            if (o.native_helper_dispatch != "worker" || o.native_stage_prefix) { err = "native third requires worker helper dispatch without a stage prefix"; return false; }
            CK(cudaSetDevice(o.native_third));
            size_t fr = 0, total = 0;
            CK(cudaMemGetInfo(&fr, &total));
            const size_t budget = (size_t)(o.native_third_gb * (1ULL << 30));
            if (fr < budget + (512ULL << 20)) { err = "native third budget leaves less than 512 MiB for scratch/reserve"; return false; }
            third_capacity = (int)(budget / nf::record_stride(g.dim, g.ff));
            if (third_capacity < 1) { err = "native third has no fitting expert records"; return false; }
            try { third = std::make_unique<NativeHelper>(o.native_third, third_capacity, true); }
            catch (const std::exception& e) { err = e.what(); return false; }
            const auto bytes = third->bytes();
            info.push_back("native third CUDA" + std::to_string(o.native_third) + ": " + std::to_string(third_capacity) +
                           " RG16 records for copies of HOST experts (empty until PLACEMENT third), " + std::to_string(bytes.pool) +
                           " weight bytes, " + std::to_string(bytes.device_scratch) + " device scratch, " + std::to_string(bytes.pinned) + " pinned bytes");
            CK(cudaSetDevice(o.native_helper));   // (as the code below found it)
        }
        if(o.native_helper_dispatch=="queue") {
            if(!helper_queue_prepare(err))return false;
            helper_dispatch_current="queue";
        }
        // the pinned pool: the experts not in VRAM assembled once (a staging copy is then one DMA, the CPU lane reads them
        // from it) - as many as the RAM this process can still get holds beside ram_headroom(), the most routed first
        // (the expert profile); the rest are read from the model's mapped files.  A pool the OS will not pin whole is
        // tried at three quarters of the size, down to an eighth
        if (o.pin_pool) {
            struct Cand {
                uint32_t n;
                int li, e;
            };
            std::vector<Cand> cand;
            size_t total = 0;
            for (size_t li = 0; li < L.size(); ++li) {
                const Layer& l = L[li];
                if (!l.moe) continue;
                for (int e = 0; e < g.n_exp; ++e)
                    if (l.res_slot[(size_t) e] < 0 && !(native && helper_owns(l,e))) {
                        cand.push_back({prof[li * g.n_exp + e], (int) li, e});
                        total += host_record_bytes(l);
                    }
            }
            std::stable_sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.n > b.n; });
            const double avail = (double) os::ram_avail();   // (the page cache counts: the mapped files can be read again)
            double budget = avail > 0 ? avail - ram_headroom() : (double) total;
            size_t want = 0, nw = 0;
            for (; nw < cand.size(); ++nw) {
                const size_t b = host_record_bytes(L[(size_t) cand[nw].li]);
                if ((double) (want + b) > budget) break;
                want += b;
            }
            CK(cudaSetDevice(D[native ? head_dev : 0].id));
            if (native) {
                if (want != total) { err = "native complete exclusive host bank does not fit the guest memory ledger"; return false; }
                try {
                    // GPU residents are already copied and immutable. Releasing
                    // their source-cache pages must precede the new physical bank.
                    // WSL's gradual balloon release can lag the Linux file advice.
                    for(size_t i=0;i<pk.map.size();++i) {
                        os::drop_mapped(pk.map[i],pk.map_bytes[i]);
                        (void)::posix_fadvise(native_source_fd[i],0,0,POSIX_FADV_DONTNEED);
                    }
                    const auto capacity=NativeHostBank::allocation_bytes(want,D[head_dev].id);
                    const uint64_t reserve=(uint64_t)ram_headroom();
                    const double until=now()+300;
                    int waits=0;
                    for(;;) {
                        const auto m=native_host_reserve_snapshot();
                        if(m.available_physical>=capacity+reserve && m.available_commit>=capacity+reserve)break;
                        if(now()>=until) { err="native host bank still lacks physical/commit headroom after source-cache retirement"; return false; }
                        if(waits++%2==0)std::fprintf(stderr,"native startup awaiting WSL reclaim: need=%llu physical=%llu commit=%llu (GPU weights retained)\n",
                            (unsigned long long)(capacity+reserve),(unsigned long long)m.available_physical,(unsigned long long)m.available_commit);
                        std::this_thread::sleep_for(std::chrono::seconds(15));
                    }
                    const auto m = native_host_reserve_snapshot();
                    std::fprintf(stderr, "native host ledger: bank=%zu reserve=%llu physical_available=%llu commit_available=%llu\n",
                        total, (unsigned long long)ram_headroom(), (unsigned long long)m.available_physical,
                        (unsigned long long)m.available_commit);
                    native_pool = std::make_unique<NativeHostBank>(want,D[head_dev].id,(uint64_t)ram_headroom(),(uint64_t)ram_headroom());
                    pool = (uint8_t*)native_pool->data();
                    std::fprintf(stderr,"native host bank allocated: capacity=%zu mappings=%zu largest_mapping=%zu\n",
                        native_pool->capacity(),native_pool->allocation_count(),native_pool->max_allocation_bytes());
                } catch (const std::exception& e) { err = e.what(); return false; }
            }
            while (!native && want > 0 && cudaHostAlloc((void**) &pool, want, cudaHostAllocPortable) != cudaSuccess) {
                pool = nullptr;
                cudaGetLastError();
                const size_t next = want / 4 * 3;
                if (next < total / 8) {
                    want = 0;
                    break;
                }
                while (nw > 0 && want > next) want -= L[(size_t) cand[--nw].li].blob;
            }
            if (want == 0 || pool == nullptr) {
                pool = nullptr;
                if (total > 0)
                    info.push_back("pinned pool: none - " + std::to_string(total >> 30) + " GB of experts outside VRAM, " +
                                   std::to_string((long long) std::max(0.0, budget / (1u << 30))) +
                                   " GB of RAM to spare: the CPU lane reads them from the model's files");
            } else {
                std::vector<std::pair<int, int>> todo;
                size_t off = 0;
                for (Layer& l : L)
                    if (l.moe) l.host_blob.assign((size_t) g.n_exp, nullptr);
                for (size_t i = 0; i < nw; ++i) {
                    Layer& l = L[(size_t) cand[i].li];
                    l.host_blob[(size_t) cand[i].e] = pool + off;
                    off += host_record_bytes(l);
                    todo.push_back({cand[i].li, cand[i].e});
                }
                std::atomic<size_t> next{0};
                std::atomic<bool> failed{false};
                std::vector<std::thread> th;
                std::mutex error_mu;
                std::string load_error;
                auto report = [&](const std::string& message) {
                    failed.store(true);
                    std::lock_guard<std::mutex> lock(error_mu);
                    if (load_error.empty()) load_error = message;
                };
                const unsigned nt = std::max(1u, std::min(compressed_host() ? 8u : 32u, std::thread::hardware_concurrency()));
                try {
                    for (unsigned t = 0; t < nt; ++t) th.emplace_back([&] {
                        r2::guarded_worker([&] {
                            std::vector<uint8_t> raw(compressed_host() ? ::mm3::scale_store::kRawBytes : 0);
                            for (size_t i; !failed.load() && (i = next.fetch_add(1)) < todo.size();) {
                                const Layer& l = L[(size_t) todo[i].first];
                                auto* dst=(uint8_t*) l.host_blob[(size_t) todo[i].second];
                                if(compressed_host()) {
                                    assemble(l,todo[i].second,raw.data()); size_t n=0; std::string e;
                                    if(!::mm3::scale_store::encode(raw.data(),raw.size(),dst,host_stride,n,e)) {report(e);break;}
                                } else assemble(l, todo[i].second, dst);
                                drop_src(l, todo[i].second);
                            }
                        }, report);
                    });
                } catch (const std::exception& e) { report(std::string("host bank worker start: ") + e.what()); }
                  catch (...) { report("unknown host bank worker start error"); }
                for (auto& t : th) t.join();
                if (failed.load()) { err = "host bank assembly failed: " + load_error; return false; }
                st.pinned_gb = (double) want / (1u << 30);
                st.logical_pinned_gb = (double)(compressed_host() ? todo.size()*::mm3::scale_store::kRawBytes : want)/(1u<<30);
                st.pooled = (long) todo.size();
                if (nw < cand.size())
                    info.push_back("pinned pool: " + std::to_string(nw) + " of the " + std::to_string(cand.size()) +
                                   " experts outside VRAM (" + std::to_string(want >> 30) + " GB: the RAM to spare), the "
                                   "most routed; the rest from the model's files");
            }
        }
        if(native && native_pool && native_pool->size()) {
            Dev& d=D[(size_t)head_dev]; CK(cudaSetDevice(d.id));
            const size_t unit=native_pool->max_allocation_bytes();
            size_t checked=0;
            for(size_t edge=0;edge<native_pool->size();edge+=unit) {
                const size_t offset=edge?edge-64:0;
                const size_t bytes=std::min<size_t>(128,native_pool->size()-offset);
                CK(cudaMemcpyAsync(d.plan_d,pool+offset,bytes,cudaMemcpyHostToDevice,d.s));
                CK(cudaMemcpyAsync(d.plan_h,d.plan_d,bytes,cudaMemcpyDeviceToHost,d.s));
                CK(cudaStreamSynchronize(d.s));
                if(std::memcmp(d.plan_h,pool+offset,bytes)!=0) { err="native host-bank primary access mismatch at "+std::to_string(offset); return false; }
                ++checked;
            }
            info.push_back("native host bank: CPU and primary copies checked at "+std::to_string(checked)+" mapping starts/boundaries");
        }
        if(native) {
            // Per-record retirement can leave filesystem readahead pages behind.
            // All target/draft weights and embeddings now have their selected
            // resident owners, so retire the whole clean source cache before
            // readiness and preserve physical Windows headroom after filling.
            for(size_t i=0;i<pk.map.size();++i) {
                os::drop_mapped(pk.map[i],pk.map_bytes[i]);
                (void)::posix_fadvise(native_source_fd[i],0,0,POSIX_FADV_DONTNEED);
            }
            const uint64_t reserve=(uint64_t)ram_headroom();
            const double until=now()+300;
            int waits=0;
            for(;;) {
                const auto m=native_host_reserve_snapshot();
                if(m.available_physical>=reserve && m.available_commit>=reserve) {
                    info.push_back("native ready host headroom: physical="+std::to_string(m.available_physical)+
                                   " commit="+std::to_string(m.available_commit)+" reserve="+std::to_string(reserve));
                    break;
                }
                if(now()>=until) {err="native loaded state lacks physical/commit reserve after source-cache retirement";return false;}
                if(waits++%2==0)std::fprintf(stderr,"native filled bank awaiting WSL source-cache reclaim: reserve=%llu physical=%llu commit=%llu\n",
                    (unsigned long long)reserve,(unsigned long long)m.available_physical,(unsigned long long)m.available_commit);
                std::this_thread::sleep_for(std::chrono::seconds(15));
            }
        }
        // host-free layers: a MoE layer with every routed expert in VRAM makes its decode plan on the device (the routing
        // read, the experts' addresses looked up and a window's experts grouped there) - a token or a verify window is
        // enqueued whole, with no wait for the host between its layers.  MAYA_MM3_HOST_FREE=0: every layer through the
        // host as before (the same plan values, so the same tokens either way)
        {
            const char* hv = ::mm3::env("MM3_HOST_FREE");
            const bool on = hv == nullptr || std::atoi(hv) != 0;
            int nhf = 0, nmoe = 0;
            for (Layer& l : L) {
                if (!l.moe) continue;
                ++nmoe;
                if (!on || (int) l.res_ids.size() != g.n_exp) continue;
                Dev& d = D[(size_t) l.dev];
                CK(cudaSetDevice(d.id));
                std::vector<unsigned long long> tab((size_t) g.n_exp);
                for (int e = 0; e < g.n_exp; ++e)
                    tab[(size_t) e] = (unsigned long long) resident_address(l, l.res_slot[(size_t) e]);
                if (!dalloc(d.id, l.exp_tab, tab.size(), err)) return false;
                CK(cudaMemcpy(l.exp_tab, tab.data(), tab.size() * sizeof(unsigned long long), cudaMemcpyHostToDevice));
                if (d.prof_d == nullptr) {
                    if (!dalloc(d.id, d.prof_d, (size_t) g.n_layers * g.n_exp, err)) return false;
                    CK(cudaMemset(d.prof_d, 0, (size_t) g.n_layers * g.n_exp * sizeof(unsigned)));
                }
                if (native && d.hf_original_d == nullptr) {
                    if (!dalloc(d.id, d.hf_original_d, (size_t)g.n_layers*g.n_exp, err)) return false;
                    CK(cudaMemset(d.hf_original_d, 0, (size_t)g.n_layers*g.n_exp*sizeof(unsigned)));
                    CK(cudaHostAlloc(&d.hf_snapshot_h,(size_t)g.n_layers*g.n_exp*sizeof(uint32_t),cudaHostAllocPortable));
                    if(o.native_stage_prefix) for(auto& h : d.hf_stage_h) CK(cudaHostAlloc(&h,(size_t)g.n_layers*g.n_exp*sizeof(uint32_t),cudaHostAllocPortable));
                    if (!dalloc(d.id,d.hf_ready_d,1,err) || !dalloc(d.id,d.hf_route_error_d,1,err)) return false;
                    CK(cudaHostAlloc(&d.hf_route_error_h,sizeof(int),cudaHostAllocPortable));
                    CK(cudaMemset(d.hf_route_error_d,0,sizeof(int)));
                    resident::Ready all{};
                    std::fill(std::begin(all.primary),std::end(all.primary),0xffffffffu);
                    CK(cudaMemcpy(d.hf_ready_d,&all,sizeof(all),cudaMemcpyHostToDevice));
                }
                l.hf = true;
                ++nhf;
            }
            if (nhf > 0)
                info.push_back("decode: " + std::to_string(nhf) + " of " + std::to_string(nmoe) +
                               " MoE layers host-free (every expert in VRAM: the routing stays on the GPU)");
            // a token's and a window's whole forward as a graph (graph_ready): one GPU, every MoE layer host-free
            const char* gv = ::mm3::env("MM3_GRAPH");
            // (several GPUs: peer access between them all - the forward's copies across them plain device copies)
            if (D.size() > 1) {
                bool all = true;
                for (Dev& a : D)
                    for (Dev& b : D) {
                        if (&a == &b) continue;
                        int can = 0;
                        if (cudaDeviceCanAccessPeer(&can, a.id, b.id) != cudaSuccess || !can) {
                            all = false;
                            continue;
                        }
                        cudaSetDevice(a.id);
                        const cudaError_t pe = cudaDeviceEnablePeerAccess(b.id, 0);
                        if (pe != cudaSuccess && pe != cudaErrorPeerAccessAlreadyEnabled) all = false;
                        cudaGetLastError();
                    }
                p2p_all = all;
            }
            graph_on = (!native || o.native_cooperative) && (gv == nullptr || std::atoi(gv) != 0) && D.size() == 1 && nhf > 0 && nhf == nmoe &&
                       !o.verbose &&
                       ::mm3::env("MM3_DUMP_HS") == nullptr;
            if (graph_on) info.push_back("decode: a token's and a verify window's forward replayed as a graph");
        }
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            if (!native && !x3::prepare(d.s)) {
                err = "exl3 prepare failed on CUDA" + std::to_string(d.id);
                return false;
            }
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            int nr = 0;
            for (int li : d.layers) nr += (int) L[(size_t) li].res_ids.size();
            info.push_back("CUDA" + std::to_string(d.id) + ": layers " + std::to_string(d.layers.front()) + "-" +
                           std::to_string(d.layers.back()) + ", " + std::to_string(nr) + " experts resident, " +
                           std::to_string(d.n_slots) + " staging slots, " + std::to_string(fr >> 20) + " MB free");
        }
        if (o.cpu_threads >= 0) {
            lane = std::make_unique<CpuLane>(o.cpu_threads, g.dim, g.ff, g.alpha, g.limit);
            if (native) {
                native_cpu_bytes = native_cpu_batch_workspace_bytes(g.dim,g.ff,lane->max_entries());
                native_cpu_workspace = std::aligned_alloc(256,up(native_cpu_bytes));
                if (!native_cpu_workspace) { err = "native CPU scratch allocation failed"; return false; }
            }
            info.push_back("CPU lane: " + std::to_string(lane->threads()) + " threads");
            st.cpu_threads = lane->threads();
            calibrate_lane();
            if(native_cpu_failed) {err="native lane calibration failed";return false;}
        }
        {
            std::string kv = std::string("KV cache: K ") + k::kv_name(o.kv_k) + ", V " + k::kv_name(o.kv_v) +
                             (o.kv_rot && (o.kv_k || o.kv_v) ? " (rows rotated)" : "") + ", " +
                             std::to_string(k::kv_bytes(o.kv_k, o.kv_v, (size_t) g.n_kv) * g.n_layers / 1024) +
                             " KB a position";
            if (o.seqs > 1)
                kv += ", " + std::to_string(o.seqs) + " conversations at once (--parallel): a cache of " +
                      std::to_string(o.max_ctx) + " positions each";
            if (st.kv_ring > 0)
                kv += ", streamed: " + std::to_string(st.kv_ring) + " positions a layer in VRAM, all " +
                      std::to_string(o.max_ctx) + " in host memory (" + std::to_string((int) (st.kv_host_gb + 0.5)) +
                      " GB pinned)";
            info.push_back(kv);
        }
        // the Monitor's map of each GPU: the dense weights, the KV caches, the scratch, the DSpark draft, the routed
        // experts' slots (resident and staging), the vision tower
        for (const Dev& d : D) {
            CK(cudaSetDevice(d.id));
            size_t fr = 0, tot = 0;
            CK(cudaMemGetInfo(&fr, &tot));
            if(fr>tot) {err="invalid CUDA free/total memory report";return false;}
            if(o.native_stage_prefix && helper && d.id==o.native_helper && fr<(1ULL<<30)) {err="native stage helper has less than 1 GiB free at ready";return false;}
            std::array<size_t, kAN> a{};
            if (auto it = acct.find(d.id); it != acct.end()) a = it->second;
            const bool head = &d - D.data() == head_dev;
            const size_t drafter = a[kADraft] + (head && draftm ? draftm->bytes() : 0);
            const size_t vision = vis && vis->device() == d.id ? vis->bytes() : 0;
            const auto helper_bytes = native && helper && d.id==o.native_helper ? helper->bytes() : NativeHelper::Bytes{};
            long long slots = d.n_slots;
            for (int li : d.layers) slots += (long long) L[(size_t) li].res_ids.size();
            char pci[32] = "-";
            if (cudaDeviceGetPCIBusId(pci, sizeof pci, d.id) != cudaSuccess) std::snprintf(pci, sizeof pci, "-");
            char b[512];
            std::snprintf(b, sizeof b,
                          "VRAM dev=%d pci=%s total=%zu free=%zu weights=%zu kv=%zu kv_ram=%zu act=%zu drafter=%zu "
                          "window=0 pool=%zu helper=%zu tail=0 borrow=0 vision=0 vision_own=%zu prompt=0 tp=0 tp_n=1 slots=%lld "
                          "ctx=%d layers=%d-%d",
                          d.id, pci, tot, fr, a[kAW]+a[kAExpert], a[kAKV], acct_kvh.count(d.id) ? acct_kvh.at(d.id) : 0,
                          a[kAAct]+helper_bytes.device_scratch, drafter, a[kAPool], helper_bytes.pool, vision, slots, o.max_ctx, d.layers.front(), d.layers.back());
            vram.push_back(b);
        }
        // An exclusive helper can be on a device with no target layers. Its
        // allocations are not present in the target device accounting above.
        if(native && helper && std::none_of(D.begin(),D.end(),[&](const Dev& d){return d.id==o.native_helper;})) {
            CK(cudaSetDevice(o.native_helper));
            size_t fr=0,tot=0; CK(cudaMemGetInfo(&fr,&tot));
            if(fr>tot) {err="invalid helper CUDA free/total memory report";return false;}
            char pci[32]="-";
            if(cudaDeviceGetPCIBusId(pci,sizeof pci,o.native_helper)!=cudaSuccess)std::snprintf(pci,sizeof pci,"-");
            const auto bytes=helper->bytes();
            char b[512];
            std::snprintf(b,sizeof b,
                "VRAM dev=%d pci=%s total=%zu free=%zu weights=0 kv=0 kv_ram=0 act=%zu drafter=0 "
                "window=0 pool=0 helper=%zu tail=0 borrow=0 vision=0 vision_own=0 prompt=0 tp=0 tp_n=1 slots=0 ctx=%d layers=none",
                o.native_helper,pci,tot,fr,bytes.device_scratch,bytes.pool,o.max_ctx);
            vram.push_back(b);
        }
        // the load's work on the legacy stream (cudaMemset of the KV rings' tags, the copies) done before the first
        // forward's on the non-blocking streams: nothing else orders the two (Windows + AMD submit queued work only when
        // the host waits - a pending memset there read as done)
        for (const Dev& d : D) {
            CK(cudaSetDevice(d.id));
            CK(cudaDeviceSynchronize());
        }
        if(o.native_stage_prefix && helper) {
            CK(cudaSetDevice(o.native_helper)); size_t ready_free=0,ready_total=0; CK(cudaMemGetInfo(&ready_free,&ready_total));
            if(ready_free<(1ULL<<30)) {err="native stage helper has less than 1 GiB free at ready";return false;}
            info.push_back("stage helper ready free="+std::to_string(ready_free));
        }
        if (const char* pv = ::mm3::env("MM3_PROF")) {
            pf.on = std::atoi(pv) != 0 && D.size() == 1;
            pf.sync = std::atoi(pv) == 2;
        }
        for (size_t di = 0; di < D.size(); ++di)   // (the pipeline's stages: the GPUs holding layers)
            if (!D[di].layers.empty()) stage_dev.push_back((int) di);
        st.load_s = now() - t0;
        return true;
    }

#include "mm3_native_placement.inc"
#include "mm3_native_adaptive.inc"
#include "mm3_native_primary_backing.inc"
#include "mm3_native_helper_live.inc"
#include "mm3_native_third.inc"
#include "mm3_native_exchange_schedule.inc"
#include "mm3_native_helper_queue.inc"
#include "mm3_native_trace.inc"

    // expert e of layer l into staging slot s of d (in stream order: the slot's earlier readers are before it)
    bool stage(const Layer& l, Dev& d, int e, int s, std::string& err) {
        if (staging_reserved(d,s)) {err="staging slot reserved by native main-bank exchange";return false;}
        const uint8_t* src = l.host_blob.empty() ? nullptr : l.host_blob[(size_t) e];
        if (src == nullptr) {
            if (native) { err = "native expert has no exclusive host record for staging"; return false; }
            CK(cudaStreamSynchronize(d.s));   // (the pinned blob is free)
            assemble(l, e, d.pin);
            src = d.pin;
        }
        const long long displaced=d.slot_owner[(size_t)s];
        const uint64_t light_submit=capture::light::stamp();
        if (!r2::foreground_issue(d.slot_owner[(size_t)s], d.slot_foreground[(size_t)s], promotion_error, err,
            [&](std::string& error) {
                cudaError_t status=cudaSuccess;
                if (compressed_host()) {
                    if(d.id!=scale_device) {error="compressed foreground submitted on non-owner device";placement_failed=true;return false;}
                    if(!scale_service->issue_foreground(staging_address(d,s),src,host_stride)) {error=scale_service->error();placement_failed=true;retirement_uncertain.store(true);return false;}
                } else status=cudaMemcpyAsync(staging_address(d,s),src,l.blob,cudaMemcpyHostToDevice,d.s);
                if (status==cudaSuccess) status=cudaEventRecord(d.slot_ev[(size_t)s],d.s);
                if (status==cudaSuccess) return true;
                error=std::string("foreground staging submission failed: ")+cudaGetErrorString(status); if(compressed_host()) {placement_failed=true;retirement_uncertain.store(true);} return false;
            })) {if(capture::detailed())capture::detail_event("demand_copy_error","\"slot\":"+std::to_string(s)+",\"error\":"+capture::quote(err));return false;}
        ++st.staged;
        capture::light::demand_copy(int(&l-L.data()),e,s,displaced,l.blob,light_submit);
        if(capture::detailed()) {std::ostringstream out;out<<"\"expert\":"<<e<<",\"slot\":"<<s
            <<",\"displaced\":"<<displaced<<",\"source_host_frame\":"<<uintptr_t(src)<<",\"destination_frame\":"<<uintptr_t(staging_address(d,s))
            <<",\"direction\":\"h2d\",\"bytes\":"<<l.blob<<",\"status\":\"submitted\"";capture::detail_event("demand_copy",out.str());}
        return true;
    }

    bool foreground_usable(const Dev& d, size_t slot) const {
        return r2::foreground_usable(d.slot_foreground[slot]!=0,d.slot_used[slot],fwd);
    }

    // the device address of expert e of layer l for this call, staging it if it is not in VRAM (busy: the slots this
    // call already took)
#include "mm3_native_demand.inc"
#include "mm3_native_prefetch.inc"
#include "mm3_native_prep_graph.inc"
#include "mm3_native_expert_graph.inc"

    size_t win_rb_bytes() const { return (size_t) kWin * 8 * 8 + (size_t) kWin * g.dim * 4; }
    size_t win_plan_bytes() const { return (size_t) kWin * 8 * 8 * 2 + (size_t) kWin * 8 * 4 * 3 + 64; }

    static uint64_t helper_clock_ns() {
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
    }
    void note_helper_pass(const NativeHelper::PassTiming& t,uint64_t prepare_begin,uint64_t submit_end,
                          uint64_t collect_begin,uint64_t merge_end) {
        // Conservative UNION of measured host/handoff intervals. It includes
        // required preparation/reduction and may overlap useful GPU work. It is
        // not an isolated exposed stall or an observed speedup from dispatch.
        const auto ms=[](uint64_t a,uint64_t b){return b>=a?double(b-a)*1e-6:0.0;};
        const uint64_t lower=t.last_not_ready_ns?t.last_not_ready_ns:t.event_record_begin_ns;
        const uint64_t wake=std::max(t.worker_published_ns,t.collect_wait_begin_ns);
        std::vector<std::pair<uint64_t,uint64_t>> intervals{
            {prepare_begin,submit_end},{t.enqueue_ns,t.worker_start_ns},
            {t.worker_start_ns,t.event_record_end_ns},{lower,t.ready_observed_ns},
            {t.ready_observed_ns,t.worker_published_ns},{collect_begin,t.collect_wait_begin_ns},
            {wake,t.collect_wait_end_ns},{t.collect_wait_end_ns,merge_end}};
        std::sort(intervals.begin(),intervals.end());
        uint64_t end=0,total=0;
        for(auto [a,b]:intervals)if(a && b>a && b>end) {total+=b-std::max(a,end);end=b;}
        ++st.helper_timed_passes;st.helper_host_upper_ms+=double(total)*1e-6;
        st.helper_dispatch_critical_upper_ms+=double(helper_dispatch_upper_ns(t,prepare_begin,submit_end,collect_begin,merge_end))*1e-6;
        st.helper_queue_ms+=ms(t.enqueue_ns,t.worker_start_ns);
        st.helper_submit_ms+=ms(t.worker_start_ns,t.event_record_end_ns);
        st.helper_detection_upper_ms+=ms(lower,t.ready_observed_ns);
        st.helper_collect_block_ms+=ms(t.collect_wait_begin_ns,t.collect_wait_end_ns);
        st.helper_after_ready_ms+=ms(wake,t.collect_wait_end_ns);
        if(replay_trace_active && t.gpu_measured) {
            std::ostringstream out;out.precision(17);
            out<<"\"activation_ms\":"<<t.activation_gpu_ms<<",\"compute_ms\":"<<t.compute_gpu_ms
               <<",\"result_ms\":"<<t.result_gpu_ms<<",\"prepare_ms\":"<<ms(prepare_begin,submit_end)
               <<",\"dispatch_ms\":"<<ms(t.enqueue_ns,t.worker_start_ns)
               <<",\"reduce_ms\":"<<ms(t.collect_wait_end_ns,merge_end);
            capture::detail_event("helper_service_observation",out.str());
        }
    }

    // a window of T tokens' layer on the token path's kernels: the dense matrices on mv (T columns, each weight
    // decoded once for them all), the routed experts on win_experts (each distinct expert read once for every token that
    // picked it), the ones neither in VRAM nor staged on the CPU lane meanwhile (its parts in mapped memory, the device
    // waits for them before the combine)
    bool layer_window(Layer& l, Dev& d, int di, int T, std::string& err) {
        capture::set_layer(&l-L.data());
        capture::Range range("layer_window");
        const bool light_on=capture::light::active() && l.moe;
        if(light_on) {light_draft=capture::light::LayerDraft{};light_draft.begin=capture::light::clock_ns();}
        const double replay_prepare_begin=replay_trace_active?now():0;
        if(replay_trace_active && l.moe)CK(cudaEventRecord(replay_gpu_events[4],d.s));
        const bool prep_timed=prepare_events && native && l.moe && !l.hf && T==1 && generation_active;
        double prep_host_begin=0;
        if(prep_timed) {
            if(!d.prep_ev[0]) {CK(cudaEventCreate(&d.prep_ev[0]));CK(cudaEventCreate(&d.prep_ev[1]));}
            prep_host_begin=now();
            CK(cudaEventRecord(d.prep_ev[0],d.s));
        }
        if(!window_pre(l,d,di,T,err,true))return false;
        if(prep_timed) {
            CK(cudaEventRecord(d.prep_ev[1],d.s));
            prep_host_end=now();prep_armed=true;st.prepare_host_ms+=1000*(prep_host_end-prep_host_begin);
        }
        if(light_on)light_draft.pre_end=capture::light::clock_ns();
        if(replay_trace_active && l.moe) {
            CK(cudaEventRecord(replay_gpu_events[6],d.s));
            capture::detail_event("prepare_host_observation","\"rows\":"+std::to_string(T)+
                ",\"host_ms\":"+std::to_string(1000*(now()-replay_prepare_begin)));
        }
        if(window_post(l,d,T,err,true))return true;
        return helper_queue_failed_window(native && l.moe && !l.hf && helper_dispatch_current=="queue");
    }
    // (the pipeline: a layer's first half - through its routing's copies to the host - enqueued before the host turns
    //  to another GPU, the second half - the wait for the routing, the plan, the experts - when it comes back)
    bool window_pre(Layer& l, Dev& d, int di, int T, std::string& err, bool regular=false) {
        capture::set_layer(&l-L.data());
        capture::Range range("layer_prepare");
        if(prefetch_active(regular)) {
            if(!prefetch_begin_layer(d,err))return false;
        }
        const auto checked_fork = [&](cudaStream_t& stream) {
            return native_window_fork(d,stream,err);
        };
        const auto checked_join = [&]() {
            return native_window_join(d,err);
        };
        if(helper_live_a>=0) {
            if(!retire_foreground_slots(d,err) || !helper_live_progress(d,err))return false;
        }
        // (the launches: run directly, or captured and replayed by prep_graph - mm3_native_prep_graph.inc; the body's
        //  lines keep their indentation)
        const auto body = [&]() -> bool {
        const auto job = [&](const x3::Mat& m, const float* x, int ldx, float* y, int ldy) {
            x3::MvJob j;
            j.m = &m;
            j.xf = x;
            j.y = y;
            j.ncol = T;
            j.xf_cs = (size_t) ldx;
            j.y_cs = (size_t) ldy;
            return j;
        };
        const int Q = g.n_head * g.hd, KV = g.n_kv * g.hd;
        // (--parallel's rows: each row's cache work - its index key and K / V written, its blocks chosen, its attention
        //  - the token path's own calls on its sequence's caches; the projections and the rest for all rows at once)
        const bool rows = !rseq.empty();
        const int IQ = g.ix_heads * g.ix_dim;
        k::rms(d.h, l.in1, T, g.dim, g.eps, d.xn, nullptr, d.s);
        {   // the indexer's branch (the side stream) beside q / k / v
            cudaStream_t xs=nullptr;
            if(!checked_fork(xs))return false;
            if(!index_proj(l,d,T,xs)) { if(!checked_join())return false; err="native index projection failed"; return false; }
            if (!rows) {
                k::index_norm_rope(d.iq, d.ik, l.iqn1, l.ikn1, d.pos, T, g.ix_heads, g.ix_dim, g.n_rot,
                                   inv_dev[(size_t) di], g.eps, l.ic, xs);
                // (the load's most blocks, not this position's: the kernels skip the blocks past each query's own, read
                //  from d.pos on the device, so the launches - and prep_graph's graphs - stay the same from position to
                //  position, with the same bits)
                const int nblk = (int) d.nblk;
                k::block_scores(d.iq, l.ic, d.pos, T, g.ix_heads, nblk, g.blk, d.bs, xs);
                k::select_blocks(d.bs, d.pos, T, g.ix_heads, nblk, g.blk, g.topk, d.kept, xs);
            } else {
                for (int r = 0; r < T; ++r) {
                    const Layer::SeqKv& c = l.sk[(size_t) rseq[(size_t) r]];
                    const int nb = rpos[(size_t) r] / g.blk + 1;
                    float* iq = d.iq + (size_t) r * IQ;
                    float* bs = d.bs + (size_t) r * g.ix_heads * d.nblk;
                    k::index_norm_rope(iq, d.ik + (size_t) r * g.ix_dim, l.iqn1, l.ikn1, d.pos + r, 1, g.ix_heads,
                                       g.ix_dim, g.n_rot, inv_dev[(size_t) di], g.eps, c.ic, xs);
                    k::block_scores(iq, c.ic, d.pos + r, 1, g.ix_heads, nb, g.blk, bs, xs);
                    k::select_blocks(bs, d.pos + r, 1, g.ix_heads, nb, g.blk, g.topk,
                                     d.kept + (size_t) r * g.ix_heads * g.topk, xs);
                }
            }
        }
        const x3::MvJob qkv[3] = {job(l.q, d.xn, g.dim, d.q, Q), job(l.k, d.xn, g.dim, d.k, KV),
                                  job(l.v, d.xn, g.dim, d.v, KV)};
        if (!x3::mv(qkv, 3, d.s)) {
            if(!checked_join())return false;
            err = "mv q/k/v (window)";
            return false;
        }
        if (!rows)
            k::qk_norm_rope(d.q, d.k, d.v, l.qn1, l.kn1, d.pos, T, g.n_head, g.n_kv, g.hd, g.n_rot, inv_dev[(size_t) di],
                            g.eps, l.kc, l.vc, d.s, 0, l.kvq, l.kvh);
        else
            for (int r = 0; r < T; ++r) {
                const Layer::SeqKv& c = l.sk[(size_t) rseq[(size_t) r]];
                k::qk_norm_rope(d.q + (size_t) r * Q, d.k + (size_t) r * KV, d.v + (size_t) r * KV, l.qn1, l.kn1,
                                d.pos + r, 1, g.n_head, g.n_kv, g.hd, g.n_rot, inv_dev[(size_t) di], g.eps, c.kc, c.vc,
                                d.s, 0, c.kvq);
            }
        pmark(d, kPAttnIn);
        if(!checked_join())return false;
        pmark(d, kPIndex);
        if (!rows)
            k::sparse_attn(d.q, l.kc, l.vc, d.kept, d.pos, T, g.n_head, g.n_kv, g.topk, g.blk,
                           1.0f / std::sqrt((float) g.hd), d.att, d.s, l.kvq, l.kvh, attn_splits, d.attp);
        else
            for (int r = 0; r < T; ++r) {
                const Layer::SeqKv& c = l.sk[(size_t) rseq[(size_t) r]];
                k::sparse_attn(d.q + (size_t) r * Q, c.kc, c.vc, d.kept + (size_t) r * g.ix_heads * g.topk, d.pos + r, 1,
                               g.n_head, g.n_kv, g.topk, g.blk, 1.0f / std::sqrt((float) g.hd), d.att + (size_t) r * Q,
                               d.s, c.kvq, k::KvHost{}, attn_splits, d.attp);
            }
        pmark(d, kPAttn);
        const x3::MvJob oj = job(l.o, d.att, Q, d.ao, g.dim);
        if (!x3::mv(&oj, 1, d.s)) {
            err = "mv o (window)";
            return false;
        }
        k::add(d.h, d.ao, (size_t) T * g.dim, d.s);
        k::rms(d.h, l.post1, T, g.dim, g.eps, d.xn, nullptr, d.s);
        pmark(d, kPOut);
        if (!l.moe) {
            const x3::MvJob gu[2] = {job(l.mg, d.xn, g.dim, d.g, g.ff_dense), job(l.mu, d.xn, g.dim, d.u, g.ff_dense)};
            if (!x3::mv(gu, 2, d.s)) {
                err = "mv dense gate/up (window)";
                return false;
            }
            k::swigluoai(d.g, d.u, (size_t) T * g.ff_dense, g.alpha, g.limit, d.g, nullptr, d.s);
            const x3::MvJob dj = job(l.md, d.g, g.ff_dense, d.ao, g.dim);
            if (!x3::mv(&dj, 1, d.s)) {
                err = "mv dense down (window)";
                return false;
            }
            k::add(d.h, d.ao, (size_t) T * g.dim, d.s);
            pmark(d, kPDense);
            return true;
        }
        {   // the router's branch (the side stream) beside the shared expert
            cudaStream_t xs=nullptr;
            if(!checked_fork(xs))return false;
            k::rows_dot(l.gate, g.n_exp, g.dim, d.xn, T, d.logit, xs);
            if (!route_layer(l,d,T,xs,err,true)) { checked_join(); return false; }
            if(predict_diag_on() && !l.hf && !predict_diag_ready(d,err)) {checked_join();return false;}
            if(early_copy_on() && T==1 && !l.hf && !early_copy_ready(d,err)) {checked_join();return false;}
            if(early_copy_on() && T==1 && !l.hf && !prefetch_active(regular) && !predict_diag_on() && !o.early_copy_legacy) {
                if(!early_copy_predict(l,d,xs,err)) {checked_join();return false;}
            } else if((prefetch_active(regular) || predict_diag_on() || (early_copy_on() && T==1)) && !l.hf && !prefetch_predict(l,d,T,xs,err)) {checked_join();return false;}
            if(native && !l.hf && helper_dispatch_current=="queue" &&
               !helper_queue_begin(l,d,T,xs,err)) {checked_join();return false;}
        }
        const x3::MvJob sgu[2] = {job(l.sg, d.xn, g.dim, d.g, g.ff_shared), job(l.su, d.xn, g.dim, d.u, g.ff_shared)};
        if (!x3::mv(sgu, 2, d.s)) {
            if(!checked_join())return false;
            err = "mv shared gate/up (window)";
            return false;
        }
        k::swigluoai(d.g, d.u, (size_t) T * g.ff_shared, g.alpha, g.limit, d.g, nullptr, d.s);
        const x3::MvJob sdj = job(l.sd, d.g, g.ff_shared, d.sh, g.dim);
        if (!x3::mv(&sdj, 1, d.s)) {
            if(!checked_join())return false;
            err = "mv shared down (window)";
            return false;
        }
        if(!checked_join())return false;
        pmark(d, kPRoute);
        if (!l.hf && !prompt_reading && !generation_active && prompt_mass_on()) {   // (MM3_PROMPT_MASS: the router's logits)
            if (!prompt_logit_h) CK(cudaHostAlloc((void**) &prompt_logit_h, (size_t) kWin * 128 * 4, cudaHostAllocPortable));
            CK(cudaMemcpyAsync(prompt_logit_h, d.logit, (size_t) T * 128 * 4, cudaMemcpyDeviceToHost, d.s));
        }
        if (!l.hf && !prompt_reading) {   // (the routing - and the CPU lane's rows - to the host; prompt_layerwise copies its own)
            if(replay_trace_active)CK(cudaEventRecord(replay_gpu_events[5],d.s));
            if (prep_zero_copy) {   // (prep_graph's capture: one kernel writes the three into the pinned buffer)
                uint8_t* rb = (uint8_t*) d.win_rb_dev;
                k::copy_words(d.top, rb, (size_t) T * g.k_exp * 4, d.s, d.w, rb + (size_t) kWin * 8 * 4, (size_t) T * g.k_exp * 4,
                              d.xn, rb + (size_t) kWin * 8 * 8, (lane || helper) ? (size_t) T * g.dim * 4 : 0);
            } else {
            CK(cudaMemcpyAsync(d.win_rb, d.top, (size_t) T * g.k_exp * 4, cudaMemcpyDeviceToHost, d.s));
            CK(cudaMemcpyAsync(d.win_rb + (size_t) kWin * 8 * 4, d.w, (size_t) T * g.k_exp * 4, cudaMemcpyDeviceToHost,
                               d.s));
            if (lane || helper)
                CK(cudaMemcpyAsync(d.win_rb + (size_t) kWin * 8 * 8, d.xn, (size_t) T * g.dim * 4,
                                   cudaMemcpyDeviceToHost, d.s));
            }
            if((prefetch_active(regular) || predict_diag_on() || (early_copy_on() && T==1 && o.early_copy_legacy)) && !prefetch_readback(l,d,T,err))return false;
        }
        return true;
        };
        if (prep_graph_eligible(l, T, regular)) return prep_graph_run(l, d, di, T, body, err);
        return body();
    }
    #include "mm3_native_owner_window.inc"
    #include "mm3_native_route_lab.inc"
    bool window_post(Layer& l, Dev& d, int T, std::string& err, bool regular=false) {
        capture::set_layer(&l-L.data());
        capture::Range range("layer_service");
        if (!l.moe) return true;
        // the device plan's layout (win_plan_d): plan_ptr [kWin][8], plan_w [kWin][8], grp_ptr [kWin 8], grp_start
        // [kWin 8 + 1], grp_ent [kWin 8], n_groups
        const size_t off_w = (size_t) kWin * 8 * 8, off_gp = off_w + (size_t) kWin * 8 * 4,
                     off_gs = off_gp + (size_t) kWin * 8 * 8, off_ge = off_gs + ((size_t) kWin * 8 + 1) * 4,
                     off_ng = off_ge + (size_t) kWin * 8 * 4;
        const auto win_args = [&]() {
            x3::WinArgs a;
            a.grp_ptr = (const unsigned long long*) (d.win_plan_d + off_gp);
            a.grp_start = (const int*) (d.win_plan_d + off_gs);
            a.grp_ent = (const int*) (d.win_plan_d + off_ge);
            a.n_groups = (const int*) (d.win_plan_d + off_ng);
            a.plan_ptr = (const unsigned long long*) d.win_plan_d;
            a.plan_w = (const float*) (d.win_plan_d + off_w);
            a.x = d.xn;
            a.x_stride = (size_t) g.dim;
            a.h = d.win_h;
            a.part = d.win_part;
            a.sh_out = d.sh;
            a.out = d.out;
            a.out_stride = (size_t) g.dim;
            a.lay = l.lay;
            a.T = T;
            a.k = g.k_exp;
            a.n_embd = g.dim;
            a.n_ff = g.ff;
            a.limit = g.limit;
            return a;
        };
        if(native && !o.native_cooperative) { ++st.owner_window_calls; st.owner_entries+=(long)T*g.k_exp; }
        if (l.hf) {   // host-free: the plan and its grouping made on the device (the host's values), nothing waited for
            const size_t lh = (size_t) (&l - L.data());
            if (!lab_hf(l, d, T, err)) return false;   // (route_lab: its routes over the kernel's where they differ)
            if (!k::plan_dev(d.top, d.w, T, g.k_exp, l.exp_tab, g.scale, (unsigned long long*) d.win_plan_d,
                             (float*) (d.win_plan_d + off_w), d.prof_d + lh * g.n_exp,
                             (unsigned long long*) (d.win_plan_d + off_gp), (int*) (d.win_plan_d + off_gs),
                             (int*) (d.win_plan_d + off_ge), (int*) (d.win_plan_d + off_ng), d.s)) {
                err = "plan_dev (window)";
                return false;
            }
            st.profiled += (long) T * g.k_exp;
            if (!x3::win_experts(win_args(), d.s)) {
                err = "win_experts";
                return false;
            }
            k::add(d.h, d.out, (size_t) T * g.dim, d.s);
            pmark(d, kPExperts);
            return true;
        }
        int* rtop = (int*) d.win_rb;
        float* rw = (float*) (d.win_rb + (size_t) kWin * 8 * 4);
        float* rx = (float*) (d.win_rb + (size_t) kWin * 8 * 8);
        const double ts = now();
        if(prefetch_active(regular)) {
            if(!prefetch_join_readers(d,err))return false;
        } else if(native && !o.native_cooperative) {
            if(!owner_window_join(d,err))return false;
        } else CK(cudaStreamSynchronize(d.s));
        const bool light_on=capture::light::active();
        const bool light_full=capture::light::full();
        if(light_on)light_draft.join_end=capture::light::clock_ns();
        st.t_sync += now() - ts;
        if(prep_armed) {
            prep_armed=false;
            float prep_ms=0;CK(cudaEventElapsedTime(&prep_ms,d.prep_ev[0],d.prep_ev[1]));
            st.prepare_gpu_ms+=prep_ms;++st.prepare_events;st.prepare_join_ms+=1000*(now()-prep_host_end);
        }
        if(replay_trace_active) {
            if(!trace_gpu_finished(err))return false;
            float ms=0;CK(cudaEventElapsedTime(&ms,replay_gpu_events[4],replay_gpu_events[5]));
            capture::detail_event("prepare_gpu_observation","\"rows\":"+std::to_string(T)+",\"context\":"+std::to_string(cur_pos+1)+",\"gpu_ms\":"+std::to_string(ms));
            CK(cudaEventElapsedTime(&ms,replay_gpu_events[5],replay_gpu_events[6]));
            capture::detail_event("route_readback_observation","\"rows\":"+std::to_string(T)+",\"gpu_ms\":"+std::to_string(ms));
        }
        const size_t li = (size_t) (&l - L.data());
        predict_diag_rotate(d,int(li),T);
        if(native && !d.slot_early.empty() && !early_copy_retire(d,err))return false;
        if(replay_fixed_batch) {
            const ReplayLayer* fixed=nullptr;
            for(const auto& item:replay_fixed_batch->layers)if(item.layer==int(li)) {fixed=&item;break;}
            if(!fixed || replay_fixed_batch->rows!=T) {err="fixed workload layer mismatch";return false;}
            std::copy(fixed->original.begin(),fixed->original.end(),d.original_h);
            std::copy(fixed->executed.begin(),fixed->executed.end(),rtop);
            std::fill(rw,rw+T*g.k_exp,1.0f/g.k_exp);
        }
        if(native && o.native_cooperative && primary_adaptive && !adaptive_policy.legacy) {
            trace_replay_boundary("layer_join",int(li),T);
            // The join already completed the preceding layer's foreground
            // copies/readers. The current layer's routing snapshot is frozen.
            if(!retire_foreground_slots(d,err) || !adaptive_publish(d,err,int(li)) ||
               !adaptive_retain_joined(d,int(li),err))return false;
        }
        if(light_on)light_draft.boundary_end=capture::light::clock_ns();
        lab_window(l, d, T, rtop, rw);   // (route_lab: the request's rules over the routes; nothing when it is off)
        lab_observe(l, d, T, rtop);      // (MM3_ROUTE_LAB_DUMP only: the pass's readiness, the lab on or off)
        if (!routes_ok(rtop, T * g.k_exp, li, err)) return false;
        if (prompt_hash_file && !generation_active) prompt_hash_routes((int) li, rtop, d.original_h, T, d.pos_h);
        if (native && !generation_active && g.k_exp == 4) prompt_route_counts(rtop, d.original_h, T);   // (prompt picks the bonus moved)
        if (native && !generation_active && g.k_exp == 4 && prompt_logit_h && prompt_mass_on()) prompt_mass(prompt_logit_h, rtop, d.original_h, T);
        begin_window_demand(l,d,rtop,T);
        trace_routes(l,d,rtop,T);
        if(native && !o.native_cooperative) return owner_window(l,d,T,rtop,rw,win_args(),err);
        // Host planning starts after route readback/join and ends before helper
        // input preparation. It includes assignment, grouping, demand policy,
        // locate/copy submission and plan upload submission; it can overlap GPU work.
        const double coop_plan_begin = native && o.native_cooperative ? now() : 0;
        const long early_staged_before = st.staged;   // early_copy: this layer's demand copies are counted from here
        const long early_copies_before = st.early_copies;   // (exchange_schedule: did this pass issue an upload)
        double coop_cpu_predicted_ms = 0;
        long coop_staging_columns = 0, coop_helper_columns = 0;
        // the plan: each token's experts (its 8 slots, k used), their pointers (0: the CPU lane's), the groups
        unsigned long long* pp = (unsigned long long*) d.win_plan_h;
        float* pw = (float*) (pp + (size_t) kWin * 8);
        unsigned long long* gp = (unsigned long long*) (pw + (size_t) kWin * 8);
        int* gs = (int*) (gp + (size_t) kWin * 8);
        int* ge = gs + kWin * 8 + 1;
        int* ng = ge + kWin * 8;
        std::vector<char> busy((size_t) d.n_slots, 0);
        if(native)for(int j=0;j<d.n_slots;++j) {
            const long long key=d.slot_owner[j];
            if(!foreground_usable(d,(size_t)j) || key<0 || key/1024!=(long long)li)continue;
            const int e=(int)(key%1024);
            busy[j]=std::find(rtop,rtop+T*g.k_exp,e)!=rtop+T*g.k_exp || lab_extra_uses(e);
        }
        if(light_full) {
            int free_slots=0,reserved_slots=0,foreground_slots=0;
            for(int j=0;j<d.n_slots;++j) {
                free_slots+=!staging_reserved(d,j) && d.slot_pend[j]<0 && foreground_usable(d,(size_t)j) && !busy[j] && !slot_early_busy(d,j);
                reserved_slots+=staging_reserved(d,j);foreground_slots+=d.slot_foreground[j]!=0;
            }
            light_draft.free_slots=uint8_t(free_slots);light_draft.reserved_slots=uint8_t(reserved_slots);
            light_draft.foreground_slots=uint8_t(foreground_slots);light_draft.pending_promotions=uint8_t(d.n_pending);
            light_draft.exchanges_pending=uint8_t(adaptive_pending(d));light_draft.total_slots=uint8_t(d.n_slots);
            light_draft.helper_lease=helper_live_a<0?0:helper_live_published?3:helper_live_withdrawn?2:1;
        }
        std::vector<CpuExpert> cpu;
        std::vector<std::vector<std::pair<int, float>>> users;
        std::vector<std::pair<int, int>> order;   // the CPU entries (expert, its user) token by token, slot by slot
        std::vector<int> cpu_of((size_t) g.n_exp, -1);
        std::vector<std::vector<int>> by_ptr_ent;
        std::vector<unsigned long long> by_ptr;
        std::vector<NativeHelper::Job> helper_jobs;
        std::vector<NativeHelper::Job> third_jobs;            // (third: this pass's picks served by the third card)
        std::array<int,kWin> third_token_columns{};           // (a helper takes four picks a token a submission)
        const bool queued_helper=helper_queue_pending!=0;
        std::array<int,128> queue_columns{};
        std::array<bool,kWin> queue_tokens{};
        std::vector<char> stage_it((size_t) g.n_exp, 0);   // (lane_split's: distinct experts staged instead)
        std::ostringstream capture_assign;
        bool capture_first=true;
        if(capture::detailed()) capture_assign << "\"assignments\":[";
        if (lane && (lane_copy > 0 || (prompt_check_mode && !generation_active && native))) {
            std::vector<int> cand;
            for (int t = 0; t < T; ++t)
                for (int i = 0; i < (lab_pass ? lab_pass_k : g.k_exp); ++i) {
                    if (lab_skipped(t, i)) continue;   // (route_lab: a skipped pick is never staged)
                    const int x = lab_pick(rtop, t, i);
                    if (on_cpu(l, d, x) && pooled(l, x) && !third_holds(l, x) && std::find(cand.begin(), cand.end(), x) == cand.end())
                        cand.push_back(x);
                }
            int available=d.n_slots;
            if(native) { available=0; for(int j=0;j<d.n_slots;++j) available+=!staging_reserved(d,j) && d.slot_pend[j]<0 && foreground_usable(d,(size_t)j) && !busy[j] && !slot_early_busy(d,j); }
            if (prompt_check_mode && !generation_active && native) {
                // prompt_check: every host expert of the pass staged, none on the CPU lane (the layer-by-layer read's
                // service), so the two prompt paths compare bit for bit; it needs a frame for each
                if ((int) cand.size() > available) {
                    err = "prompt_check: " + std::to_string(cand.size()) + " host experts in a pass, " +
                          std::to_string(available) + " free staging frames (load with more --staging)";
                    return false;
                }
                for (int x : cand) stage_it[(size_t) x] = 1;
            } else if (native && o.native_cooperative && o.cooperative_stage_reuse) {
                std::array<int,128> uses{};
                if(!lab_pass)for(int i=0;i<T*g.k_exp;++i)++uses[size_t(rtop[i])];
                else for(int t=0;t<T;++t)for(int i=0;i<lab_pass_k;++i)if(!lab_skipped(t,i))++uses[size_t(lab_pick(rtop,t,i))];
                if(third_on)for(int e=0;e<g.n_exp;++e)if(uses[size_t(e)] && third_serves(l,d,e))uses[size_t(e)]=0;   // (third: outside the plan)
                std::vector<int> cp,gp,hp;
                for(int e=0;e<g.n_exp;++e)if(uses[size_t(e)]) {
                    if(helper_owns(l,e))hp.push_back(e);
                    else if(on_cpu(l,d,e))cp.push_back(e);
                    else gp.push_back(e);
                }
                const int maximum=std::min({d.n_slots/2,available,replay_stage_limit<0?d.n_slots:replay_stage_limit});
                const auto plan=plan_batch_staging(uses,cp,gp,hp,cand,maximum,T,batch_staging_costs,o.cooperative_cpu_scale);
                if(!plan.valid || batch_staging_arithmetic!=native_primary_current) {
                    err=std::string("batch staging requires valid matching grouped service costs: ")+plan.status;return false;
                }
                for(int e:plan.selected)stage_it[size_t(e)]=1;
                coop_cpu_predicted_ms=plan.predicted_cpu_ms;
                if(light_full) {light_draft.stage_max=uint8_t(maximum);light_draft.stage_selected=uint8_t(plan.selected.size());}
                if(capture::detailed()) {
                    std::ostringstream out;out.precision(17);
                    out<<"\"policy\":\"grouped_cost_greedy\",\"rows\":"<<T<<",\"cpu_price_scale\":"<<o.cooperative_cpu_scale<<",\"eligible\":"<<cand.size()
                       <<",\"available_slots\":"<<available<<",\"max_new\":"<<maximum
                       <<",\"predicted_cpu_ms\":"<<plan.predicted_cpu_ms<<",\"predicted_gpu_ms\":"<<plan.predicted_gpu_ms
                       <<",\"predicted_helper_ms\":"<<plan.predicted_helper_ms<<",\"critical_before_ms\":"<<plan.critical_before_ms
                       <<",\"critical_after_ms\":"<<plan.critical_after_ms<<",\"selected\":[";
                    for(size_t i=0;i<plan.selected.size();++i){if(i)out<<',';out<<plan.selected[i];}
                    out<<"],\"distinct_uploads\":"<<plan.selected.size()<<",\"upload_bytes\":"<<plan.selected.size()*l.blob;
                    if(replay_trace_active) {
                        const auto ids=[&](const char* name,const std::vector<int>& values) {
                            out<<",\""<<name<<"\":[";for(size_t i=0;i<values.size();++i){if(i)out<<',';out<<values[i];}out<<']';
                        };
                        ids("cpu",cp);ids("gpu",gp);ids("helper",hp);ids("candidates",cand);
                        out<<",\"uses\":[";for(size_t i=0;i<uses.size();++i){if(i)out<<',';out<<uses[i];}out<<']';
                    }
                    capture::detail_event("batch_staging_decision",out.str());
                }
            } else {
                const int s = std::min({(int)cand.size(),lane_split((int)cand.size()),d.n_slots/2,available});
                for (int i = 0; i < s; ++i) stage_it[(size_t) cand[(size_t) i]] = 1;
            }
        }
        std::array<int,kWin> lab_helper_columns{};   // (route_lab: the 3090 takes four picks a token a submission)
        int lab_fifth = 0;                           // (route_lab: widened rows' fifth picks dispatched)
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < 8; ++i) {
                const int e = t * 8 + i;
                pp[e] = 0;
                pw[e] = 0.0f;
                if (i >= (lab_pass ? lab_pass_k : g.k_exp) || lab_skipped(t, i)) continue;
                lab_fifth += i >= g.k_exp;
                const int x = lab_pick(rtop, t, i);
                const float mixture = i < g.k_exp ? rw[t * g.k_exp + i] : lab_extra_w[size_t(t)];
                const float wt = mixture * g.scale;
                pw[e] = wt;
                if(capture::detailed()) {
                    if(!capture_first) capture_assign << ','; capture_first=false;
                    const char* service=helper_owns(l,x)?"helper":l.res_slot[x]>=0?"primary_resident":
                        third_serves(l,d,x) && third_token_columns[size_t(t)]<4?"third":
                        on_cpu(l,d,x) && !stage_it[size_t(x)]?"cpu":"primary_staged";
                    capture_assign << "{\"row\":"<<t<<",\"pick\":"<<i<<",\"expert\":"<<x
                        <<",\"original\":"<<(i < g.k_exp ? d.original_h[t*g.k_exp+i] : -1)<<",\"mixture\":"<<mixture
                        <<",\"scaled_weight\":"<<wt<<",\"service\":\""<<service<<"\"}";
                }
                ++prof[li * g.n_exp + x];
                if (native && helper_owns(l,x)) {
                    ++coop_helper_columns;
                    if(light_full && i < g.k_exp)light_draft.service[t*g.k_exp+i]=capture::light::kServiceHelper;
                    if(queued_helper) {++queue_columns[x];queue_tokens[t]=true;continue;}
                    if(lab_pass && ++lab_helper_columns[size_t(t)] > 4) {   // (a widened token's fifth 3090 pick)
                        NativeHelper::Job j; j.layer=(int)li; j.expert=x; j.columns=1; j.token[0]=t; j.weight[0]=wt;
                        lab_helper_second.push_back(j);
                        continue;
                    }
                    auto it=std::find_if(helper_jobs.begin(),helper_jobs.end(),[&](const NativeHelper::Job& j) {
                        return j.expert==x && j.columns<NativeHelper::kMaxColumns;
                    });
                    if(it==helper_jobs.end()) { NativeHelper::Job j; j.layer=(int)li; j.expert=x; helper_jobs.push_back(j); it=helper_jobs.end()-1; }
                    const int c=it->columns++; it->token[c]=t; it->weight[c]=wt;
                    continue;
                }
                if (native && third_serves(l, d, x) && third_token_columns[size_t(t)] < 4) {   // third: the copy on the third card
                    ++third_token_columns[size_t(t)];
                    ++st.third_columns;
                    if(light_full && i < g.k_exp)light_draft.service[t*g.k_exp+i]=capture::light::kServiceThird;
                    auto it=std::find_if(third_jobs.begin(),third_jobs.end(),[&](const NativeHelper::Job& j) {
                        return j.expert==x && j.columns<NativeHelper::kMaxColumns;
                    });
                    if(it==third_jobs.end()) { NativeHelper::Job j; j.layer=(int)li; j.expert=x; third_jobs.push_back(j); it=third_jobs.end()-1; }
                    const int c=it->columns++; it->token[c]=t; it->weight[c]=wt;
                    continue;
                }
                if (lane && on_cpu(l, d, x) && !stage_it[(size_t) x]) {
                    if(light_full && i < g.k_exp)light_draft.service[t*g.k_exp+i]=capture::light::kServiceCpu;
                    if (cpu_seen.empty()) cpu_seen.assign((size_t) g.n_layers * g.n_exp, 0);
                    if (cpu_of[(size_t) x] < 0) ++cpu_seen[li * g.n_exp + x];
                    if (cpu_of[(size_t) x] < 0 && !native) consider_promotion(l, d, x);
                    if (cpu_of[(size_t) x] < 0) {
                        cpu_of[(size_t) x] = (int) cpu.size();
                        cpu.push_back(cpu_expert(l, x, 1.0f));
                        users.emplace_back();
                    }
                    users[(size_t) cpu_of[(size_t) x]].push_back({t, wt});
                    order.push_back({cpu_of[(size_t) x], (int) users[(size_t) cpu_of[(size_t) x]].size() - 1});
                    continue;
                }
                const long light_staged_before=st.staged;
                if (!locate(l, d, x, busy, pp[e], err)) return false;
                if(light_full && i < g.k_exp)light_draft.service[t*g.k_exp+i]=l.res_slot[(size_t)x]>=0?capture::light::kServicePrimaryResident:
                    st.staged>light_staged_before?capture::light::kServiceStagedNew:capture::light::kServiceStagedReady;
                if(native && l.res_slot[(size_t)x]<0) ++coop_staging_columns;
                size_t gi = 0;
                while (gi < by_ptr.size() && by_ptr[gi] != pp[e]) ++gi;
                if (gi == by_ptr.size()) {
                    by_ptr.push_back(pp[e]);
                    by_ptr_ent.emplace_back();
                }
                by_ptr_ent[gi].push_back(e);
            }
        st.profiled += (long) T * g.k_exp;
        lab_after_plan(l, lab_fifth);   // (route_lab only)
        if(capture::detailed()) { capture_assign << ']'; capture::detail_event("expert_assignment",capture_assign.str()); }
        if(queued_helper) {
            ++st.helper_queue_passes;int count=0;
            for(int n:queue_columns)count+=(n+NativeHelper::kMaxColumns-1)/NativeHelper::kMaxColumns;
            st.helper_experts+=count;if(!count)++st.helper_queue_empty;
        }
        if(prefetch_active(regular))prefetch_admit(l,d,T,busy);
        finish_window_demand(l,d,rtop,T,cpu_of);
        int ne = 0;
        for (size_t gi = 0; gi < by_ptr.size(); ++gi) {
            gp[gi] = by_ptr[gi];
            gs[gi] = ne;
            for (int e : by_ptr_ent[gi]) ge[ne++] = e;
        }
        gs[by_ptr.size()] = ne;
        *ng = (int) by_ptr.size();
        if(replay_trace_active)CK(cudaEventRecord(replay_gpu_events[11],d.s));
        // expert_graph: the plan's copy is the graph's first kernel (a replay reads the pinned plan when it runs)
        const bool expert_graph_on = expert_graph_eligible(l, T, regular);
        if (!expert_graph_on) CK(cudaMemcpyAsync(d.win_plan_d, d.win_plan_h, win_plan_bytes(), cudaMemcpyHostToDevice, d.s));
        if(replay_trace_active)CK(cudaEventRecord(replay_gpu_events[12],d.s));
        if(prefetch_active(regular) && !prefetch_issue_one(d,err))return false;
        if(!early_copy_issue(l,d,T,busy,early_staged_before,err))return false;
        if(native && o.native_cooperative) {
            ++st.coop_windows;
            st.coop_plan_ms += 1000*(now()-coop_plan_begin);
            st.coop_cpu_predicted_ms += coop_cpu_predicted_ms;
            st.coop_cpu_columns += (long)order.size();
            for(const auto& u:users) st.coop_cpu_chunks += (long)((u.size()+7)/8);
            st.coop_primary_columns += ne;
            st.coop_primary_groups += (long)by_ptr.size();
            for(const auto& entries:by_ptr_ent) st.coop_primary_chunks += (long)((entries.size()+7)/8);
            st.coop_staging_columns += coop_staging_columns;
            st.coop_helper_columns += coop_helper_columns;
        }
        if(light_on)light_draft.plan_end=capture::light::clock_ns();
        if(helper_jobs.empty() && !queued_helper)exchange_x4_window(kX4NoJobNs);   // (exchange_schedule: no 3090 job this pass)
        uint64_t helper_seq=0;
        uint64_t helper_prepare_begin=0,helper_submit_end=0;
        // third: its submission beside the helper's (the rows as BF16 again, for its own pinned buffers), and the
        // join of its work when the pass fails after it was submitted
        uint64_t third_seq=0;
        std::vector<uint16_t> third_input;
        const auto third_submit=[&](std::string& error) {
            capture::Range third_range("third_submit");
            third_input.resize((size_t)T*g.dim);
            for(size_t i=0;i<third_input.size();++i) { uint32_t u; std::memcpy(&u,rx+i,4); third_input[i]=(uint16_t)((u+0x7fffu+((u>>16)&1u))>>16); }
            third_seq=third->submit(T,third_input.data(),third_jobs,error);
            return third_seq!=0;
        };
        const auto third_retire=[&]() {
            if(!third_seq)return;
            std::vector<float> retired((size_t)T*g.dim); std::string join_error;
            (void)third->collect(third_seq,retired.data(),retired.size(),join_error);
            third_seq=0;
        };
        float* cpu_rows = d.win_cpu;
        int* cpu_flags = (int*) (d.win_cpu + (size_t) kWin * g.dim);
        unsigned* wait_word = (unsigned*) (cpu_flags + kWin);
        x3::WinArgs a = win_args();
        if (lab_pass) a.k = lab_pass_k;   // (route_lab: 5 when a row is widened; empty slots add nothing)
        bool with_cpu = false;
        bool expert_added = false;   // (expert_graph: the residual add ran inside the replay)
        // cpu_first (experimental A/B control, off by default). The CPU lane is last on nearly every pass that
        // has a CPU job, and nothing in its batch waits for the GPU submissions, about 0.1 ms per pass. The
        // submit thread issues them (the helper's work, the primary's kernels, the one optional query) while
        // this thread starts the CPU batch right after planning, then joins it before collecting the helper.
        // Counters and capture fields are written here after the join, so the two threads share nothing.
        const bool cpu_first_pass = o.cpu_first && native && o.native_cooperative && !cpu.empty() &&
                                    !queued_helper && !replay_trace_active && !replay_fixed_batch;
        bool cpu_ran = false;
        double cpu_lane_begin = 0;
        if(cpu_first_pass) {
            with_cpu = true;
            a.cpu_part = cpu_rows;
            a.cpu_stride = (size_t) g.dim;
            a.cpu_flag = cpu_flags;
            a.wait_ptr = wait_word;
            a.wait_val = ++d.win_wait;
            const bool query_primary = o.primary_submit_query;
            const int device = d.id;
            cudaStream_t stream = d.s;
            std::string submit_error;
            int failure = 0;   // 1: helper submission, 2: primary kernels, 3: query
            bool queried = false, not_ready = false;
            uint64_t primary_submit_end_ns = 0;
            submit_worker.ensure();
            submit_worker.post([&, a]() -> bool {
                cudaSetDevice(device);
                if(!helper_jobs.empty()) {
                    capture::Range helper_range("helper_submit");
                    helper_prepare_begin=helper_clock_ns();
                    std::vector<uint16_t> input((size_t)T*g.dim);
                    for(size_t i=0;i<input.size();++i) { uint32_t u; std::memcpy(&u,rx+i,4); input[i]=(uint16_t)((u+0x7fffu+((u>>16)&1u))>>16); }
                    helper_seq=helper->submit(T,input.data(),helper_jobs,submit_error);
                    if(!helper_seq) {failure=1;return false;}
                    helper_submit_end=helper_clock_ns();
                }
                if(!third_jobs.empty() && !third_submit(submit_error)) {failure=1;return false;}
                capture::Range primary_range("primary_expert_submit_and_join");
                x3::WinArgs args=a;
                if(!x3::win_experts(args, stream)) {submit_error="win_experts";failure=2;return false;}
                if(query_primary) {
                    queried=true;
                    const auto status=cudaStreamQuery(stream);
                    not_ready=status==cudaErrorNotReady;
                    if(status!=cudaSuccess && status!=cudaErrorNotReady) {
                        submit_error=std::string("primary submission query: ")+cudaGetErrorString(status);failure=3;return false;
                    }
                }
                if(light_on)primary_submit_end_ns=capture::light::clock_ns();
                return true;
            });
            struct JoinGuard { SubmitWorker& worker; bool joined=false; ~JoinGuard(){ if(!joined) worker.wait(); } } join_guard{submit_worker};
            {
                capture::Range cpu_range("cpu_expert_batch");
                cpu_lane_begin = now();
                lane_run_multi(cpu, users, rx, T, cpu_rows, &order);
                st.coop_cpu_run_ms += 1000*(now()-cpu_lane_begin);
                if(light_on)light_draft.cpu_end=capture::light::clock_ns();
            }
            cpu_ran = true;
            const bool submitted = submit_worker.wait();
            join_guard.joined = true;
            ++st.cpu_first_passes;
            if(helper_seq) st.helper_experts+=(long)helper_jobs.size();
            if(queried) {++st.primary_submit_queries; if(not_ready)++st.primary_submit_not_ready;}
            if(light_on) {light_draft.helper_submit_end=helper_submit_end;light_draft.primary_submit_end=primary_submit_end_ns;}
            if(!submitted) {
                if(failure!=1) *(volatile unsigned*) wait_word = d.win_wait;   // (never leave the device waiting)
                if(helper_seq) { std::vector<float> retired((size_t)T*g.dim); std::string join_error;
                    (void)helper->collect(helper_seq,retired.data(),retired.size(),join_error); }
                third_retire();
                if(failure==3) retirement_uncertain.store(true);
                err = submit_error;
                return false;
            }
        }
        if(!cpu_first_pass && !helper_jobs.empty()) {
            capture::Range helper_range("helper_submit");
            helper_prepare_begin=helper_clock_ns();
            std::vector<uint16_t> input((size_t)T*g.dim);
            for(size_t i=0;i<input.size();++i) { uint32_t u; std::memcpy(&u,rx+i,4); input[i]=(uint16_t)((u+0x7fffu+((u>>16)&1u))>>16); }
            helper_seq=helper->submit(T,input.data(),helper_jobs,err);
            if(!helper_seq)return false;
            st.helper_experts+=(long)helper_jobs.size();
        }
        if(!cpu_first_pass && !third_jobs.empty() && !third_submit(err)) {
            if(helper_seq) { std::vector<float> retired((size_t)T*g.dim); std::string join_error;
                (void)helper->collect(helper_seq,retired.data(),retired.size(),join_error); }
            return false;
        }
        if(!cpu_first_pass) {
            if(helper_seq)helper_submit_end=helper_clock_ns();
            if(light_on)light_draft.helper_submit_end=helper_submit_end;
            with_cpu = !cpu.empty() || helper_seq!=0 || queued_helper || third_seq!=0;
            if (with_cpu) {
                a.cpu_part = cpu_rows;
                a.cpu_stride = (size_t) g.dim;
                a.cpu_flag = cpu_flags;
                a.wait_ptr = wait_word;
                a.wait_val = ++d.win_wait;
            }
        }
        std::optional<capture::Range> primary_range;
        if(!cpu_first_pass) primary_range.emplace("primary_expert_submit_and_join");
        const uint64_t primary_submit_begin=(!cpu_first_pass && replay_trace_active)?helper_clock_ns():0;
        if(!cpu_first_pass) {
        if(replay_trace_active) {
            CK(cudaEventRecord(replay_gpu_events[0],d.s));
            a.diagnostic_compute_done=replay_gpu_events[1];a.diagnostic_wait_done=replay_gpu_events[2];
        }
        if (!(expert_graph_on ? expert_graph_run(d, T, a, with_cpu, expert_added, err) : x3::win_experts(a, d.s))) {
            if (with_cpu) *(volatile unsigned*) wait_word = d.win_wait;   // (never leave the device waiting)
            if(helper_seq) { std::vector<float> retired((size_t)T*g.dim); std::string join_error;
                (void)helper->collect(helper_seq,retired.data(),retired.size(),join_error); }
            third_retire();
            if (err.empty()) err = "win_experts";
            return false;
        }
        if(replay_trace_active) {
            CK(cudaEventRecord(replay_gpu_events[3],d.s));replay_gpu_pending=true;
            replay_gpu_forward=fwd;replay_gpu_layer=int(li);replay_gpu_rows=T;
        }
        // Exactly one nonblocking query, immediately after primary submission
        // and before CPU service. A submission-progress benefit is empirical,
        // not part of cudaStreamQuery's completion-status API contract.
        const bool query_primary = replay_fixed_batch ? replay_query_primary : o.primary_submit_query;
        if(query_primary) {
            ++st.primary_submit_queries;
            const auto status=cudaStreamQuery(d.s);
            if(status==cudaErrorNotReady)++st.primary_submit_not_ready;
            if(status!=cudaSuccess && status!=cudaErrorNotReady) {
                if(with_cpu)*(volatile unsigned*)wait_word=d.win_wait;
                if(helper_seq) {
                    std::vector<float> retired((size_t)T*g.dim);std::string join_error;
                    (void)helper->collect(helper_seq,retired.data(),retired.size(),join_error);
                }
                third_retire();
                err=std::string("primary submission query: ")+cudaGetErrorString(status);
                retirement_uncertain.store(true);return false;
            }
        }
        if(light_on)light_draft.primary_submit_end=capture::light::clock_ns();
        if(replay_trace_active) {
            const auto end=helper_clock_ns();
            capture::detail_event("primary_submission_observation","\"start_ns\":"+std::to_string(primary_submit_begin)+
                ",\"end_ns\":"+std::to_string(end)+",\"diagnostic_query\":"+(query_primary?"true":"false"));
        }
        }   // !cpu_first_pass
        if (with_cpu) {   // the CPU lane's parts while the device runs the others; then the word it waits on
            const double tl = cpu_ran ? cpu_lane_begin : now();
            if(!cpu.empty()) {
                if(!cpu_ran) {   // cpu_first already ran the batch, right after planning
                    capture::Range cpu_range("cpu_expert_batch");
                    const uint64_t cpu_begin=replay_trace_active?helper_clock_ns():0;
                    lane_run_multi(cpu, users, rx, T, cpu_rows, &order);
                    if(native && o.native_cooperative) st.coop_cpu_run_ms += 1000*(now()-tl);
                    if(replay_trace_active)capture::detail_event("cpu_service_observation","\"start_ns\":"+std::to_string(cpu_begin)+
                        ",\"end_ns\":"+std::to_string(helper_clock_ns()));
                    if(light_on)light_draft.cpu_end=capture::light::clock_ns();
                }
            } else std::fill_n(cpu_rows,(size_t)T*g.dim,0.0f);
            for (int t = 0; t < T; ++t) cpu_flags[t] = 0;
            for (const auto& u : users)
                for (const auto& tu : u) cpu_flags[tu.first] = 1;
            NativeHelper::PassTiming helper_timing{};
            uint64_t helper_collect_begin=0;bool helper_ok=false;
            if(helper_seq) {
                capture::Range collect_range("helper_collect");
                helper_collect_begin=helper_clock_ns();
                std::vector<float>& result=d.helper_result;   // collect zero-fills these rows before merging
                result.resize((size_t)T*g.dim);
                bool ok=helper->collect(helper_seq,result.data(),result.size(),err,&helper_timing);
                // (route_lab: a widened token's fifth 3090 pick, a second submission before the 3090 is called idle)
                if(ok && !lab_helper_second.empty()) ok=lab_helper_second_pass(T,rx,cpu_rows,cpu_flags,err);
                helper_ok=ok;
                if(ok)exchange_x4_window(kX4AfterCollectNs);   // (exchange_schedule: the 3090 is idle until the next join)
                st.helper_graph_launches=(long long)helper->graph_launches();
                st.helper_zero_copy_checked=(long long)helper->zero_copy_checked();
                st.helper_zero_copy_mismatches=(long long)helper->zero_copy_mismatches();
                if(ok) {
                    for(size_t i=0;i<result.size();++i)cpu_rows[i]+=result[i];
                    for(const auto& j:helper_jobs)for(int c=0;c<j.columns;++c)cpu_flags[j.token[c]]=1;
                } else native_cpu_failed=true;
            }
            if(queued_helper) {
                std::vector<float> result((size_t)T*g.dim);
                if(helper_queue_finish(T,result.data(),err)) {
                    for(size_t i=0;i<result.size();++i)cpu_rows[i]+=result[i];
                    for(int t=0;t<T;++t)if(queue_tokens[t])cpu_flags[t]=1;
                } else native_cpu_failed=true;
            }
            if(third_seq) {   // third: its rows, added on the host like the helper's
                capture::Range collect_range("third_collect");
                NativeHelper::PassTiming third_timing{};
                std::vector<float>& result=d.third_result;   // collect zero-fills these rows before merging
                result.resize((size_t)T*g.dim);
                if(third->collect(third_seq,result.data(),result.size(),err,&third_timing)) {
                    for(size_t i=0;i<result.size();++i)cpu_rows[i]+=result[i];
                    int columns=0;
                    for(const auto& j:third_jobs)for(int c=0;c<j.columns;++c) {cpu_flags[j.token[c]]=1;++columns;}
                    ++st.third_passes;st.third_jobs+=(long)third_jobs.size();
                    if(light_on)capture::light::third_pass(light_draft.begin,int(li),int(third_jobs.size()),columns,third_timing.enqueue_ns,
                        third_timing.worker_start_ns,third_timing.worker_published_ns,third_timing.collect_wait_begin_ns,
                        third_timing.collect_wait_end_ns,helper_clock_ns());
                } else native_cpu_failed=true;
                third_seq=0;
            }
            if(helper_ok) {
                const uint64_t helper_merge_end=helper_clock_ns();
                note_helper_pass(helper_timing,helper_prepare_begin,helper_submit_end,helper_collect_begin,helper_merge_end);
                if(light_on) {
                    light_draft.collect_end=helper_merge_end;light_draft.helper_enqueue=helper_timing.enqueue_ns;
                    light_draft.helper_worker_start=helper_timing.worker_start_ns;
                    light_draft.helper_record_begin=helper_timing.event_record_begin_ns;
                    light_draft.helper_record_end=helper_timing.event_record_end_ns;
                    light_draft.helper_last_not_ready=helper_timing.last_not_ready_ns;
                    light_draft.helper_ready_observed=helper_timing.ready_observed_ns;
                    light_draft.helper_worker_published=helper_timing.worker_published_ns;
                    light_draft.helper_wait_begin=helper_timing.collect_wait_begin_ns;
                    light_draft.helper_wait_end=helper_timing.collect_wait_end_ns;
                }
            }
            st.t_lane += now() - tl;
            st.cpu_experts += (long) cpu.size();
            std::atomic_thread_fence(std::memory_order_seq_cst);
            *(volatile unsigned*) wait_word = d.win_wait;
        }
        if (!expert_added) k::add(d.h, d.out, (size_t) T * g.dim, d.s);
        else if (o.expert_graph == 2 && !expert_graph_verify(l, d, T, a, err)) return false;
        pmark(d, kPExperts);
        if(native_cpu_failed) { if(err.empty())err="native CPU/helper service failed"; return false; }
        if(light_on) {
            light_draft.end=capture::light::clock_ns();
            capture::light::layer(light_draft,int(li),T,g.k_exp,d.original_h,rtop);
        }
        // exchange_schedule: the 5090's owed download pieces, if this pass leaves its link idle until the next join
        if(!exchange_pass_end(d,st.staged!=early_staged_before || st.early_copies!=early_copies_before,!cpu.empty(),err))return false;
        if(!trace_service(d,err))return false;
        return true;
    }

    // a non-resident expert's use: its decayed score, now

    // Pending slots are made visible only when a boundary observes completion;
    // unfinished copies retain their slots without delaying useful work.
    // a picture's room (Vision::set_lender) when its GPU has none free: the staging slots, between forwards - their
    // experts dropped, staged or promoted again as the forwards after it ask (drop_slots)
    // (the load, before the experts are placed) d's staging slots reallocated as n of them - their old region freed
    static constexpr int kMinSlots = 4;
    bool resize_slots(Dev& d, int n, std::string& err) {
        if (!d.slot_ptr.empty()) {err="cannot resize rotated native staging frames";return false;}
        if (d.stg == nullptr || n <= 0 || n >= d.n_slots) return false;
        CK(cudaSetDevice(d.id));
        CK(cudaDeviceSynchronize());
        for (size_t i = 0; i < owned.size(); ++i)
            if (owned[i] == (void*) d.stg) {
                owned.erase(owned.begin() + (long) i);
                owned_dev.erase(owned_dev.begin() + (long) i);
                break;
            }
        CK(cudaFree(d.stg));
        acct[d.id][kAPool] -= std::min(acct[d.id][kAPool], std::max<size_t>((size_t) d.n_slots * d.slot_bytes, 16));
        d.stg = nullptr;
        acat = kAPool;
        const bool ok = dalloc(d.id, d.stg, (size_t) n * d.slot_bytes, err);
        acat = kAAct;
        if (!ok) return false;
        for (size_t s = (size_t) n; s < d.slot_ev.size(); ++s) cudaEventDestroy(d.slot_ev[s]);
        d.slot_ev.resize((size_t) n);
        d.n_slots = n;
        d.slot_owner.assign((size_t) n, -1);
        d.slot_used.assign((size_t) n, -1);
        d.slot_pend.assign((size_t) n, -1);
        d.slot_iss.assign((size_t) n, 0);
        d.slot_foreground.assign((size_t) n, 0);
        d.slot_early.assign((size_t) n, 0);   // cudaDeviceSynchronize above retired any early copy
        d.slot_early_fresh.assign((size_t) n, 0);
        d.n_pending = 0;
        return true;
    }

    void* lend_slots(Dev& d, size_t need) {
        if (!d.slot_ptr.empty()) return nullptr;
        if (d.stg == nullptr || (size_t) d.n_slots * d.slot_bytes < need) return nullptr;
        std::string err;
        if(!scale_drain(err))return nullptr;
        if(d.xs && !exchange_downloads_join(d,err)) {placement_failed=true;retirement_uncertain.store(true);return nullptr;}
        if(cudaSetDevice(d.id)!=cudaSuccess || cudaStreamSynchronize(d.s) != cudaSuccess || (d.cs && cudaStreamSynchronize(d.cs) != cudaSuccess)) {placement_failed=true;retirement_uncertain.store(true);return nullptr;}
        if(!scale_joined(d,err))return nullptr;
        return d.stg;
    }
    void drop_slots(Dev& d) {
        if (adaptive_pending(d)) {placement_failed=true;retirement_uncertain.store(true);return;}
        // An unobserved early copy still writes its frame on the copy stream: join it before the frame is anyone's.
        if (std::find(d.slot_early.begin(),d.slot_early.end(),uint8_t(1))!=d.slot_early.end() &&
            (!d.cs || cudaSetDevice(d.id)!=cudaSuccess || cudaStreamSynchronize(d.cs)!=cudaSuccess)) {
            placement_failed=true;retirement_uncertain.store(true);return;
        }
        prefetch_discard_joined(d);
        for (int s = 0; s < d.n_slots; ++s) {
            if (!d.slot_early_fresh.empty() && d.slot_early_fresh[(size_t) s]) ++st.early_copy_unused;
            d.slot_owner[(size_t) s] = -1;
            d.slot_used[(size_t) s] = -1;
            d.slot_pend[(size_t) s] = -1;
            d.slot_iss[(size_t) s] = 0;
            d.slot_foreground[(size_t) s] = 0;
            if (!d.slot_early.empty()) {d.slot_early[(size_t) s] = 0; d.slot_early_fresh[(size_t) s] = 0;}
        }
        d.n_pending = 0;
        d.promo_q.clear();
    }

    bool retire_foreground_slots(Dev& d, std::string& err) {
        return r2::foreground_retire(d.slot_foreground,d.slot_owner,promotion_error,err,
            [&](size_t slot,std::string& error) {
                ++st.foreground_queries;
                const cudaError_t status=cudaEventQuery(d.slot_ev[slot]);
                if(status==cudaSuccess) {
                    capture::light::foreground_retire(int(slot),d.slot_owner[slot]);
                    if(capture::detailed())capture::detail_event("demand_copy_retire","\"slot\":"+std::to_string(slot)+",\"owner\":"+std::to_string(d.slot_owner[slot])+",\"status\":\"completed\"");
                    return r2::Completion::ready;
                }
                if(status==cudaErrorNotReady) {++st.foreground_not_ready;return r2::Completion::not_ready;}
                error=std::string("foreground staging completion failed: ")+cudaGetErrorString(status);
                if(capture::detailed())capture::detail_event("demand_copy_error","\"slot\":"+std::to_string(slot)+",\"error\":"+capture::quote(error));
                return r2::Completion::failed;
            });
    }
#include "mm3_native_slot_publish.inc"
    bool refresh_slots(Dev& d, std::string& err) {
        if(!retire_foreground_slots(d,err))return false;
        if (!issue_promotions(d, err)) return false;
        return publish_slots(d,err);
    }

    // expert e of layer l ran on the CPU lane: when it keeps coming back, copied into a staging slot meanwhile (the
    // slot whose expert scores least, not used in this forward nor pending)
    void lane_run(const std::vector<CpuExpert>& ex, const float* x, float* out) {
        if (native) {
            std::vector<NativeCpuExpert> jobs;
            for (const auto& e : ex) jobs.push_back({e.native_blob, e.w});
            std::vector<std::vector<std::pair<int,float>>> users(jobs.size());
            for(size_t i=0;i<jobs.size();++i) users[i].push_back({0,jobs[i].w});
            if (!native_cpu_run_multi_batch(*lane, g.dim, g.ff, jobs, users, x, 1, out, nullptr, native_cpu_workspace, native_cpu_bytes)) {
                native_cpu_failed = true;
                std::fill_n(out, g.dim, 0.0f);
            }
        } else if (gguf) gg_run(*lane, g.dim, g.ff, g.limit, ex, x, out);
        else lane->run(ex, x, out);
    }
    void lane_run_multi(const std::vector<CpuExpert>& ex, const std::vector<std::vector<std::pair<int, float>>>& users,
                        const float* x, int T, float* out, const std::vector<std::pair<int, int>>* order) {
        if (native) {
            std::vector<NativeCpuExpert> jobs;
            for (const auto& e : ex) jobs.push_back({e.native_blob, e.w});
            if (!native_cpu_run_multi_batch(*lane, g.dim, g.ff, jobs, users, x, T, out, order, native_cpu_workspace, native_cpu_bytes)) {
                native_cpu_failed = true;
                std::fill_n(out, (size_t)T*g.dim, 0.0f);
            }
        } else if (gguf) gg_run_multi(*lane, g.dim, g.ff, g.limit, ex, users, x, T, out, order);
        else lane->run_multi(ex, users, x, T, out, order);
    }

    // whether expert e of layer l has its blob in the pinned pool (a staging copy is then one DMA)
    bool pooled(const Layer& l, int e) const { return !l.host_blob.empty() && l.host_blob[(size_t) e] != nullptr; }

    bool helper_owns(const Layer& l, int e) const {
        return helper && e >= 0 && e < g.n_exp && ((l.helper_ready[e/32] >> (e%32)) & 1u);
    }

    bool reset_hf_route_errors(std::string& err) {
        if (!native) return true;
        for (Dev& d:D) if (d.hf_route_error_d) {
            CK(cudaSetDevice(d.id));
            CK(cudaMemsetAsync(d.hf_route_error_d,0,sizeof(int),d.s));
        }
        return true;
    }
    bool queue_hf_route_errors(Dev& d,std::string& err) {
        if (!native || !d.hf_route_error_d) return true;
        CK(cudaSetDevice(d.id));
        CK(cudaMemcpyAsync(d.hf_route_error_h,d.hf_route_error_d,sizeof(int),cudaMemcpyDeviceToHost,d.s));
        return true;
    }
    bool check_hf_route_errors(const Dev& d,std::string& err) {
        if (native && d.hf_route_error_d && *d.hf_route_error_h) {
            err="invalid native host-free router output on CUDA"+std::to_string(d.id);
            return false;
        }
        return true;
    }

    // A pass sees only published residents. The stream consumes the pinned
    // snapshot before this GPU reaches the next host routing boundary.
    bool route_layer(Layer& l, Dev& d, int T, cudaStream_t s, std::string& err, bool window) {
        if (!native) { k::route(d.logit,l.bias,T,g.n_exp,g.k_exp,d.top,d.w,s); return true; }
        // The scope guard is armed only when HF counting can actually be submitted.
        if (window && l.hf) hf_demand_active=true;
        if (window && l.hf) {
            if(capture::detailed())capture::detail_event("capture_coverage_gap","\"component\":\"host_free_router\",\"rows\":"+std::to_string(T)+",\"reason\":\"no_existing_joined_row_readback\"");
            // Immutable for every fully resident layer, including during overlap
            // with a later layer's mutable ready snapshot.
            const auto policy = !o.routing ? resident::Policy::Off :
                                o.routing_prior ? resident::Policy::Prior : resident::Policy::Substitute;
            const bool lab_on=lab_active(l,T);   // (route_lab: the scores to the host, every expert ready)
            CK(resident::route(d.logit,l.bias,T,policy,policy == resident::Policy::Prior ? router_bonus() : o.routing_ratio,
                               d.hf_ready_d,d.top,d.w,d.original_d,nullptr,s,o.routing_dampen,lab_on ? d.lab_scores_d : nullptr));
            if(lab_on) {
                lab_snapshot_all_ready(d);
                CK(cudaMemcpyAsync(d.lab_scores_h,d.lab_scores_d,size_t(T)*128*3*sizeof(float),cudaMemcpyDeviceToHost,s));
            }
            CK(resident::count_original(d.original_d,T*g.k_exp,
                d.hf_original_d+(size_t)(&l-L.data())*g.n_exp,s));
            CK(resident::guard_expert_ids(d.top,T*g.k_exp,d.hf_route_error_d,s));
            return true;
        }
        route_ready_host(l,d);
        if(prep_zero_copy) k::copy_words(d.ready_h_dev,d.ready_d,sizeof(resident::Ready),s);   // (prep_graph's capture)
        else CK(cudaMemcpyAsync(d.ready_d,d.ready_h,sizeof(resident::Ready),cudaMemcpyHostToDevice,s));
        const auto policy = !o.routing ? resident::Policy::Off :
                            o.routing_prior ? resident::Policy::Prior : resident::Policy::Substitute;
        const bool lab_on=window && lab_active(l,T);   // (route_lab: the masks and the scores to the host)
        if(lab_on) lab_snapshot(l,d);
        float* scores_d=capture::detailed() && d.capture_scores_d ? d.capture_scores_d : lab_on ? d.lab_scores_d : nullptr;
        CK(resident::route(d.logit,l.bias,T,policy,policy == resident::Policy::Prior ? router_bonus() : o.routing_ratio,
                          d.ready_d,d.top,d.w,d.original_d,nullptr,s,o.routing_dampen,scores_d));
        if(capture::detailed() && d.capture_scores_d && d.capture_scores_h)
            CK(cudaMemcpyAsync(d.capture_scores_h,d.capture_scores_d,size_t(T)*128*3*sizeof(float),cudaMemcpyDeviceToHost,s));
        if(lab_on)
            CK(cudaMemcpyAsync(d.lab_scores_h,scores_d,size_t(T)*128*3*sizeof(float),cudaMemcpyDeviceToHost,s));
        if(prep_zero_copy) k::copy_words(d.original_d,d.original_h_dev,(size_t)T*g.k_exp*4,s);
        else if(!prompt_reading) CK(cudaMemcpyAsync(d.original_h,d.original_d,(size_t)T*g.k_exp*4,cudaMemcpyDeviceToHost,s));
        return true;
    }

    bool on_cpu(const Layer& l, const Dev& d, int e) const {
        if (native && helper_owns(l,e)) return false;
        if (l.res_slot[(size_t) e] >= 0) return false;
        const long long key = (long long) (&l - L.data()) * 1024 + e;
        for (int s = 0; s < d.n_slots; ++s)
            if (foreground_usable(d,(size_t)s) && d.slot_owner[(size_t) s] == key) return false;
        return true;
    }

    // expert e of layer l in host memory (the pinned pool's blob, or the mapped files' pieces) for the CPU lane
    CpuExpert cpu_expert(const Layer& l, int e, float w) const {
        CpuExpert c;
        c.w = w;
        const uint8_t* blob = l.host_blob.empty() ? nullptr : l.host_blob[(size_t) e];
        if (native) { c.native_blob = blob; return c; }
        if (gguf) {   // (the three matrices' rows: the pool's blob, or the GGUF's mapping)
            c.gf = &l.gf;
            for (int r = 0; r < 3; ++r) c.gg[r] = blob ? blob + l.lay.trellis[r] : l.src[(size_t) e].p[r * 3];
            return c;
        }
        for (int r = 0; r < 3; ++r) {
            strata::kernels::cpu::Exl3Part& m = c.p[r];
            if (blob) {
                m.trellis = (const uint32_t*) (blob + l.lay.trellis[r]);
                m.suh = (const uint16_t*) (blob + l.lay.suh[r]);
                m.svh = (const uint16_t*) (blob + l.lay.svh[r]);
            } else {
                const Src& sp = l.src[(size_t) e];
                m.trellis = (const uint32_t*) sp.p[r * 3 + 0];
                m.suh = (const uint16_t*) sp.p[r * 3 + 1];
                m.svh = (const uint16_t*) sp.p[r * 3 + 2];
            }
            m.k = r < 2 ? g.dim : g.ff;
            m.n = r < 2 ? g.ff : g.dim;
            m.k2 = l.lay.k2[r];
        }
        return c;
    }

    // one token's layer on the decode kernels
    bool layer_token(Layer& l, Dev& d, int di, std::string& err) {
        const auto job = [](const x3::Mat& m, const float* x, float* y) {
            x3::MvJob j;
            j.m = &m;
            j.xf = x;
            j.y = y;
            return j;
        };
        k::rms(d.h, l.in1, 1, g.dim, g.eps, d.xn, nullptr, d.s);
        {   // the indexer's branch (the side stream) beside q / k / v
            const cudaStream_t xs = fork(d, !l.hf);
            if(!index_proj(l,d,1,xs)) { join(d,!l.hf); err="native index projection failed"; return false; }
            k::index_norm_rope(d.iq, d.ik, l.iqn1, l.ikn1, d.pos, 1, g.ix_heads, g.ix_dim, g.n_rot,
                               inv_dev[(size_t) di], g.eps, l.ic, xs);
            const int nblk = cur_pos / g.blk + 1;
            k::block_scores(d.iq, l.ic, d.pos, 1, g.ix_heads, nblk, g.blk, d.bs, xs);
            k::select_blocks(d.bs, d.pos, 1, g.ix_heads, nblk, g.blk, g.topk, d.kept, xs);
        }
        const x3::MvJob qkv[3] = {job(l.q, d.xn, d.q), job(l.k, d.xn, d.k), job(l.v, d.xn, d.v)};
        if (!x3::mv(qkv, 3, d.s)) {
            join(d, !l.hf);
            err = "mv q/k/v";
            return false;
        }
        k::qk_norm_rope(d.q, d.k, d.v, l.qn1, l.kn1, d.pos, 1, g.n_head, g.n_kv, g.hd, g.n_rot, inv_dev[(size_t) di],
                        g.eps, l.kc, l.vc, d.s, 0, l.kvq, l.kvh);
        pmark(d, kPAttnIn);
        join(d, !l.hf);
        pmark(d, kPIndex);
        k::sparse_attn(d.q, l.kc, l.vc, d.kept, d.pos, 1, g.n_head, g.n_kv, g.topk, g.blk, 1.0f / std::sqrt((float) g.hd),
                       d.att, d.s, l.kvq, l.kvh, attn_splits, d.attp);
        pmark(d, kPAttn);
        const x3::MvJob oj = job(l.o, d.att, d.ao);
        if (!x3::mv(&oj, 1, d.s)) {
            err = "mv o";
            return false;
        }
        k::add(d.h, d.ao, (size_t) g.dim, d.s);
        k::rms(d.h, l.post1, 1, g.dim, g.eps, d.xn, nullptr, d.s);
        pmark(d, kPOut);
        if (!l.moe) {
            const x3::MvJob gu[2] = {job(l.mg, d.xn, d.g), job(l.mu, d.xn, d.u)};
            if (!x3::mv(gu, 2, d.s)) {
                err = "mv dense gate/up";
                return false;
            }
            k::swigluoai(d.g, d.u, (size_t) g.ff_dense, g.alpha, g.limit, d.g, nullptr, d.s);
            const x3::MvJob dj = job(l.md, d.g, d.ao);
            if (!x3::mv(&dj, 1, d.s)) {
                err = "mv dense down";
                return false;
            }
            k::add(d.h, d.ao, (size_t) g.dim, d.s);
            pmark(d, kPDense);
            return true;
        }
        {   // the router's branch (the side stream) beside the shared expert
            const cudaStream_t xs = fork(d, !l.hf);
            k::rows_dot(l.gate, g.n_exp, g.dim, d.xn, 1, d.logit, xs);
            k::route(d.logit, l.bias, 1, g.n_exp, g.k_exp, d.top, d.w, xs);
        }
        // the shared expert into sh (while the router's answer comes back)
        const x3::MvJob sgu[2] = {job(l.sg, d.xn, d.g), job(l.su, d.xn, d.u)};
        if (!x3::mv(sgu, 2, d.s)) {
            join(d, !l.hf);
            err = "mv shared gate/up";
            return false;
        }
        k::swigluoai(d.g, d.u, (size_t) g.ff_shared, g.alpha, g.limit, d.g, nullptr, d.s);
        const x3::MvJob sdj = job(l.sd, d.g, d.sh);
        if (!x3::mv(&sdj, 1, d.s)) {
            join(d, !l.hf);
            err = "mv shared down";
            return false;
        }
        join(d, !l.hf);
        pmark(d, kPRoute);
        if (l.hf) {   // host-free: the plan made on the device (the host's values), nothing waited for
            const size_t lh = (size_t) (&l - L.data());
            if (!k::plan_dev(d.top, d.w, 1, g.k_exp, l.exp_tab, g.scale, d.plan_d, (float*) (d.plan_d + 8),
                             d.prof_d + lh * g.n_exp, nullptr, nullptr, nullptr, nullptr, d.s)) {
                err = "plan_dev";
                return false;
            }
            st.profiled += g.k_exp;
            x3::MoeArgs a;
            a.plan_ptr = d.plan_d;
            a.plan_w = (const float*) (d.plan_d + 8);
            a.lay = l.lay;
            a.k = g.k_exp;
            a.n_embd = g.dim;
            a.n_ff = g.ff;
            a.limit = g.limit;
            x3::moe_gate_up(a, d.xn, nullptr, d.s);
            x3::moe_down(a, nullptr, d.sh, d.out, d.s);
            k::add(d.h, d.out, (size_t) g.dim, d.s);
            pmark(d, kPExperts);
            return true;
        }
        int top[8];
        float w[8];
        CK(cudaMemcpyAsync(d.plan_h, d.top, (size_t) g.k_exp * 4, cudaMemcpyDeviceToHost, d.s));
        CK(cudaMemcpyAsync(d.plan_h + 4, d.w, (size_t) g.k_exp * 4, cudaMemcpyDeviceToHost, d.s));
        if (lane) CK(cudaMemcpyAsync(d.xh_h, d.xn, (size_t) g.dim * 4, cudaMemcpyDeviceToHost, d.s));
        const double ts = now();
        CK(cudaStreamSynchronize(d.s));
        st.t_sync += now() - ts;
        std::memcpy(top, d.plan_h, (size_t) g.k_exp * 4);
        std::memcpy(w, d.plan_h + 4, (size_t) g.k_exp * 4);
        const size_t li = (size_t) (&l - L.data());
        if (!routes_ok(top, g.k_exp, li, err)) return false;
        for (int i = 0; i < g.k_exp; ++i) ++prof[li * g.n_exp + top[i]];
        st.profiled += g.k_exp;
        std::vector<char> busy((size_t) d.n_slots, 0);
        unsigned long long* ph = d.plan_h;
        float* pw = (float*) (d.plan_h + 8);
        std::vector<CpuExpert> cpu;
        // (of the experts the CPU lane would take, lane_split's - ones in the pinned pool - to staging slots instead)
        int to_stage = 0;
        if (lane && lane_copy > 0) {
            int cand = 0;
            for (int i = 0; i < g.k_exp; ++i)
                cand += on_cpu(l, d, top[i]) && pooled(l, top[i]);
            to_stage = lane_split(cand);
        }
        for (int i = 0; i < g.k_exp; ++i) {
            pw[i] = w[i] * g.scale;
            if (lane && on_cpu(l, d, top[i]) && to_stage > 0 && pooled(l, top[i])) {
                --to_stage;   // (staged: locate below)
            } else if (lane && on_cpu(l, d, top[i])) {   // (not in VRAM nor in a staging slot: the CPU lane's)
                if (cpu_seen.empty()) cpu_seen.assign((size_t) g.n_layers * g.n_exp, 0);
                ++cpu_seen[li * g.n_exp + top[i]];
                consider_promotion(l, d, top[i]);
                ph[i] = 0;
                cpu.push_back(cpu_expert(l, top[i], pw[i]));
                continue;
            }
            if (!locate(l, d, top[i], busy, ph[i], err)) return false;
        }
        CK(cudaMemcpyAsync(d.plan_d, ph, 16 * sizeof(unsigned long long), cudaMemcpyHostToDevice, d.s));
        x3::MoeArgs a;
        a.plan_ptr = d.plan_d;
        a.plan_w = (const float*) (d.plan_d + 8);
        a.lay = l.lay;
        a.k = g.k_exp;
        a.n_embd = g.dim;
        a.n_ff = g.ff;
        a.limit = g.limit;
        x3::moe_gate_up(a, d.xn, nullptr, d.s);
        if (lane) {   // the CPU lane's part while the GPU runs the others' gate / up
            float* part = d.xh_h + g.dim;
            int flag = 0;
            if (!cpu.empty()) {
                const double tl = now();
                lane_run(cpu, d.xh_h, part);
                const double dt = now() - tl;
                st.t_lane += dt;
                flag = 1;
                st.cpu_experts += (long) cpu.size();
                if (lane_copy > 0) lane_retime((int) cpu.size(), 1e3 * dt);
            }
            std::memcpy(part + g.dim, &flag, 4);
            CK(cudaMemcpyAsync(d.cpu_d, part, (size_t) (g.dim + 1) * 4, cudaMemcpyHostToDevice, d.s));
            a.cpu_part = d.cpu_d;
            a.cpu_flag = (const int*) (d.cpu_d + g.dim);
        }
        x3::moe_down(a, nullptr, d.sh, d.out, d.s);
        k::add(d.h, d.out, (size_t) g.dim, d.s);
        pmark(d, kPExperts);
        // (the plan's pinned copy is rewritten by the next MoE layer of this GPU after its routing sync, which
        //  waits on this one's kernels)
        return true;
    }

    // the routed experts of layer l for this chunk: top [T][K] on the host -> y rows (by the sorted rows' order) and
    // invrow; a pass over the resident experts (by resident slot), then one a batch of the staged experts (by staging
    // slot: an expert already in a slot keeps it, the others take the least recently used slots)
    bool experts(Layer& l, Dev& d, int T, const std::vector<int>& top, std::string& err) {
        if(compressed_host()) {err="compressed HOST requires the joined native owner window (chunk <= 16)";return false;}
        if (native && !d.slot_ptr.empty()) {err="rotated native staging requires the checked window path";return false;}
        const int K = g.k_exp;
        std::vector<std::vector<int>> rows_of((size_t) g.n_exp);   // expert -> its (t * K + slot) entries
        for (int t = 0; t < T; ++t)
            for (int i = 0; i < K; ++i) rows_of[(size_t) top[(size_t) t * K + i]].push_back(t * K + i);
        std::vector<int> staged;
        for (int e = 0; e < g.n_exp; ++e)
            if (!rows_of[(size_t) e].empty() && l.res_slot[(size_t) e] < 0) staged.push_back(e);
        struct Pass {
            const uint8_t* wbase;
            size_t stride;
            std::vector<int> slot_expert;   // the pass's experts by position (-1: none)
            int row0 = 0, rows = 0, max_rows = 0;
            std::vector<int> bounds;
            bool staged = false;
        };
        std::vector<Pass> passes;
        if (native && !l.res_ptr.empty()) {
            for (size_t s=0;s<l.res_ids.size();++s)
                if(!rows_of[l.res_ids[s]].empty()) passes.push_back({resident_address(l,int(s)),l.blob,{l.res_ids[s]}});
        } else if (!l.res_ids.empty()) passes.push_back({l.res_base, l.blob, l.res_ids});
        const long long lid = (long long) (&l - L.data());
        for (size_t si = 0; si < staged.size();) {
            std::vector<int> se((size_t) d.n_slots, -1), fresh;
            for (int n = 0; si < staged.size() && n < d.n_slots; ++si, ++n) {
                const int e = staged[si];
                int at = -1;
                for (int s = 0; s < d.n_slots; ++s)
                    if (foreground_usable(d,(size_t)s) && d.slot_owner[(size_t) s] == lid * 1024 + e) at = s;
                if (at >= 0 && se[(size_t) at] < 0) se[(size_t) at] = e;
                else fresh.push_back(e);
            }
            for (size_t fi = 0; fi < fresh.size(); ++fi) {
                int best = -1;
                for (int s = 0; s < d.n_slots; ++s)
                    if (se[(size_t) s] < 0 && d.slot_pend[(size_t) s] < 0 && foreground_usable(d,(size_t)s) &&
                        (best < 0 || d.slot_used[(size_t) s] < d.slot_used[(size_t) best]))
                        best = s;
                if (best < 0) {   // (every free slot pending: the rest first in the next batch)
                    if (std::none_of(se.begin(),se.end(),[](int e){return e>=0;})) {
                        // This legacy prefill path has no CPU alternative. Do
                        // not spin while optional background slots are pending.
                        err="no staging slot available for legacy prefill";
                        return false;
                    }
                    staged.insert(staged.begin() + (long) si, fresh.begin() + (long) fi, fresh.end());
                    break;
                }
                se[(size_t) best] = fresh[fi];
            }
            Pass p{d.stg, d.slot_bytes, std::move(se)};
            p.staged = true;
            passes.push_back(std::move(p));
        }
        std::vector<int> rowidx, invrow((size_t) T * K, -1);
        rowidx.reserve((size_t) T * K);
        for (Pass& p : passes) {
            p.row0 = (int) rowidx.size();
            for (int e : p.slot_expert) {
                p.bounds.push_back((int) rowidx.size());
                if (e < 0) continue;
                for (int ent : rows_of[(size_t) e]) {
                    invrow[(size_t) ent] = (int) rowidx.size();
                    rowidx.push_back(ent / K);
                }
                p.max_rows = std::max(p.max_rows, (int) rows_of[(size_t) e].size());
            }
            p.bounds.push_back((int) rowidx.size());
            p.rows = (int) rowidx.size() - p.row0;
        }
        if ((int) rowidx.size() != T * K) {
            err = "internal: routed rows " + std::to_string(rowidx.size());
            return false;
        }
        CK(cudaMemcpyAsync(d.rowidx, rowidx.data(), rowidx.size() * 4, cudaMemcpyHostToDevice, d.s));
        CK(cudaMemcpyAsync(d.invrow, invrow.data(), invrow.size() * 4, cudaMemcpyHostToDevice, d.s));
        for (const Pass& p : passes) {
            if (p.rows == 0) continue;
            if (p.staged) {
                for (int s = 0; s < d.n_slots; ++s) {
                    const int e = p.slot_expert[(size_t) s];
                    if (e < 0) continue;
                    if (d.slot_owner[(size_t) s] != lid * 1024 + e) {
                        if (!stage(l, d, e, s, err)) return false;
                        d.slot_owner[(size_t) s] = lid * 1024 + e;
                    }
                    d.slot_used[(size_t) s] = fwd;
                }
            }
            CK(cudaMemcpyAsync(d.bounds, p.bounds.data(), p.bounds.size() * 4, cudaMemcpyHostToDevice, d.s));
            x3::SetArgs sa;
            sa.wbase = p.wbase;
            sa.stride = p.stride;
            sa.lay = l.lay;
            sa.n_exp = (int) p.slot_expert.size();
            sa.bounds = d.bounds;
            sa.row0 = p.row0;
            sa.max_rows = p.max_rows;
            sa.n_embd = g.dim;
            sa.n_ff = g.ff;
            sa.limit = g.limit;
            sa.hbounds = p.bounds.data();   // (before sm_80: the busy experts through the FP16 GEMM)
            if (gguf) {   // (the rows also as FP16 - rms' xn16 - and the gate's sums of a layer whose gate and up differ
                          //  in type in this pass's own y rows, which its down writes only after them)
                sa.x16 = d.xn16;
                sa.graw = d.y + (size_t) p.row0 * g.dim;
            }
            if (native) {
                if (!ne::set(sa, d.xn, g.dim, d.rowidx + p.row0, d.y + (size_t)p.row0*g.dim, g.dim, d.s)) {
                    err = "native prompt expert execution failed";
                    return false;
                }
            } else {
                x3::set_gate_up(sa, d.xn, g.dim, d.rowidx + p.row0, d.h16, g.ff, d.s);
                x3::set_down(sa, d.h16, g.ff, d.y + (size_t) p.row0 * g.dim, g.dim, d.s);
            }
            // (the next pass rewrites the bounds and h16: let this one finish with them)
            CK(cudaStreamSynchronize(d.s));
        }
        return true;
    }

    bool layer(int li, int T, int pmax, std::string& err) {
        Layer& l = L[(size_t) li];
        Dev& d = D[(size_t) l.dev];
        const int di = l.dev;
        if (T == 1 && o.fast_decode && rseq.empty() && !native) return layer_token(l, d, di, err);   // native C1 shares grouped dispatch
        if (T <= kWin && o.fast_decode) return layer_window(l, d, di, T, err);
        // h += x W: the BLAS's beta 1; with hipBLASLt (beta 0 products only) the product into ao, then the add - the
        // same sum of the product and h
        const auto gemm_acc = [&](const x3::Mat& m, const uint16_t* x16, int ldx) {
            if (gguf && gg::gemm_backend() == 1) {
                x3::gemm_rows(m, x16, ldx, T, d.ao, g.dim, 0.0f, d.s);
                k::add(d.h, d.ao, (size_t) T * g.dim, d.s);
            } else {
                x3::gemm_rows(m, x16, ldx, T, d.h, g.dim, 1.0f, d.s);
            }
        };
        // attention
        k::rms(d.h, l.in1, T, g.dim, g.eps, d.xn, d.xn16, d.s);
        x3::gemm_rows(l.q, d.xn16, g.dim, T, d.q, g.n_head * g.hd, 0.0f, d.s);
        x3::gemm_rows(l.k, d.xn16, g.dim, T, d.k, g.n_kv * g.hd, 0.0f, d.s);
        x3::gemm_rows(l.v, d.xn16, g.dim, T, d.v, g.n_kv * g.hd, 0.0f, d.s);
        k::qk_norm_rope(d.q, d.k, d.v, l.qn1, l.kn1, d.pos, T, g.n_head, g.n_kv, g.hd, g.n_rot, inv_dev[(size_t) di],
                        g.eps, l.kc, l.vc, d.s, 0, l.kvq, l.kvh);
        pmark(d, kPAttnIn);
        const float one = 1.0f, zero = 0.0f;
        const int nq = g.ix_heads * g.ix_dim;
        // (Q8_0 projections: each decoded into wtmp first - the F32 values of an F32 upload, the same GEMM)
        if (native) {
            x3::gemm_rows(l.niq, d.xn16, g.dim, T, d.iq, nq, 0.0f, d.s);
            x3::gemm_rows(l.nik, d.xn16, g.dim, T, d.ik, g.ix_dim, 0.0f, d.s);
        } else {
        if (l.wiq8 != nullptr) k::q8_0_f32(l.wiq8, (size_t) nq * g.dim, d.wtmp, d.s);
        CB(cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, nq, T, g.dim, &one, l.wiq8 ? d.wtmp : l.wiq, g.dim, d.xn, g.dim,
                       &zero, d.iq, nq));
        if (l.wik8 != nullptr) k::q8_0_f32(l.wik8, (size_t) g.ix_dim * g.dim, d.wtmp, d.s);
        CB(cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, g.ix_dim, T, g.dim, &one, l.wik8 ? d.wtmp : l.wik, g.dim, d.xn, g.dim, &zero,
                       d.ik, g.ix_dim));
        }
        k::index_norm_rope(d.iq, d.ik, l.iqn1, l.ikn1, d.pos, T, g.ix_heads, g.ix_dim, g.n_rot, inv_dev[(size_t) di],
                           g.eps, l.ic, d.s);
        pmark(d, kPIxProj);
        const int nblk = pmax / g.blk + 1;
        k::block_scores(d.iq, l.ic, d.pos, T, g.ix_heads, nblk, g.blk, d.bs, d.s);
        pmark(d, kPIndex);
        k::select_blocks(d.bs, d.pos, T, g.ix_heads, nblk, g.blk, g.topk, d.kept, d.s);
        pmark(d, kPIxSel);
        if (const char* dp = ::mm3::env("MM3_DUMP_KEPT")) {   // (debugging: the kept blocks, int32 records)
            std::vector<int> kh((size_t) g.ix_heads * T * g.topk);
            CK(cudaMemcpyAsync(kh.data(), d.kept, kh.size() * 4, cudaMemcpyDeviceToHost, d.s));
            CK(cudaStreamSynchronize(d.s));
            if (std::FILE* f = std::fopen(dp, li == 0 && cur_pos - T + 1 == 0 ? "wb" : "ab")) {
                const int hdr[5] = {li, cur_pos - T + 1, T, g.ix_heads, g.topk};
                std::fwrite(hdr, 4, 5, f);
                std::fwrite(kh.data(), 4, kh.size(), f);
                std::fclose(f);
            }
        }
        k::sparse_attn(d.q, l.kc, l.vc, d.kept, d.pos, T, g.n_head, g.n_kv, g.topk, g.blk, 1.0f / std::sqrt((float) g.hd),
                       d.att, d.s, l.kvq, l.kvh);
        pmark(d, kPAttn);
        k::to_f16(d.att, T, g.n_head * g.hd, g.n_head * g.hd, d.att16, g.n_head * g.hd, d.s);
        gemm_acc(l.o, d.att16, g.n_head * g.hd);   // h += attn Wo
        pmark(d, kPOGemm);
        // MLP
        k::rms(d.h, l.post1, T, g.dim, g.eps, d.xn, d.xn16, d.s);
        pmark(d, kPOut);
        if (!l.moe) {
            x3::gemm_rows(l.mg, d.xn16, g.dim, T, d.g, g.ff_dense, 0.0f, d.s);
            x3::gemm_rows(l.mu, d.xn16, g.dim, T, d.u, g.ff_dense, 0.0f, d.s);
            k::swigluoai(d.g, d.u, (size_t) T * g.ff_dense, g.alpha, g.limit, nullptr, d.h16, d.s);
            gemm_acc(l.md, d.h16, g.ff_dense);   // h += mlp
            pmark(d, kPDense);
            return true;
        }
        CB(cublasSgemm(d.blas, CUBLAS_OP_T, CUBLAS_OP_N, g.n_exp, T, g.dim, &one, l.gate, g.dim, d.xn, g.dim, &zero,
                       d.logit, g.n_exp));
        if (!route_layer(l,d,T,d.s,err,false)) return false;
        pmark(d, kPRouter);
        // the shared expert into sh
        x3::gemm_rows(l.sg, d.xn16, g.dim, T, d.g, g.ff_shared, 0.0f, d.s);
        x3::gemm_rows(l.su, d.xn16, g.dim, T, d.u, g.ff_shared, 0.0f, d.s);
        k::swigluoai(d.g, d.u, (size_t) T * g.ff_shared, g.alpha, g.limit, nullptr, d.h16, d.s);
        x3::gemm_rows(l.sd, d.h16, g.ff_shared, T, d.sh, g.dim, 0.0f, d.s);
        pmark(d, kPRoute);
        CK(cudaMemcpyAsync(d.top_h, d.top, (size_t) T * g.k_exp * 4, cudaMemcpyDeviceToHost, d.s));
        CK(cudaStreamSynchronize(d.s));
        const std::vector<int> top(d.top_h, d.top_h + (size_t) T * g.k_exp);
        if (!routes_ok(top.data(), (int) top.size(), (size_t) li, err)) return false;
        observe_routes(l,d,top.data(),T);
        for (int e : top) ++prof[(size_t) li * g.n_exp + e];
        st.profiled += (long) top.size();
        pmark(d, kPSync);
        if (!experts(l, d, T, top, err)) return false;
        k::moe_combine(d.h, d.y, d.invrow, d.w, d.sh, T, g.k_exp, g.dim, g.scale, d.out, d.s);
        std::swap(d.h, d.out);
        pmark(d, kPExperts);
        return true;
    }

    // ---- graphs: a token's or a verify window's whole forward (its layers and the head's GEMV) captured once as a
    // CUDA / HIP graph and replayed - the same kernels with the same arguments, so the same bits, without a launch a
    // kernel from the host (Windows + AMD: ~2000 launches a token, each a host-side cost that left the GPU waiting).
    // Only where nothing in the forward needs the host: one GPU, every MoE layer host-free, the token kernels.  A
    // graph is keyed by its tokens, its head rows, the indexer's block count (the one argument that moves with the
    // position: every 128 tokens - a new capture then) and the residual stream's buffer (a prompt's MoE layers swap h
    // and out: a graph captured before a prompt would read the other one after it).  MAYA_MM3_GRAPH=0: no graphs
    // (--parallel: the key also holds the bound sequence - its caches are the graph's - and a forward_rows step's
    //  rows' sequences and block counts; graphs are dropped least recently used first, kGraphKeep kept)
    struct GraphEntry {
        cudaGraphExec_t exec = nullptr;
        long profiled = 0;   // (st.profiled's count of one replay: the capture's)
        long long used = 0;
    };
    std::map<std::vector<long long>, GraphEntry> graphs;
    long long graph_clock = 0;
    static constexpr size_t kGraphKeep = 12;
    bool graph_on = false;   // (load: the conditions above, and the setting)
    bool graph_ready(int T, int want) const {
        return graph_on && T <= kWin && o.fast_decode && (want == 0 || want <= kWin) && !pf.on;
    }
    void graphs_clear() {
        for (auto& kv : graphs)
            if (kv.second.exec) cudaGraphExecDestroy(kv.second.exec);
        graphs.clear();
        prep_graphs_clear();
    }
    template <class F>
    bool run_graph(int T, int want, F& body, std::string& err) {
        Dev& d = D[(size_t) 0];
        const bool rows = !rseq.empty();
        const int nblk = rows ? -1 : cur_pos / g.blk + 1;
        std::vector<long long> key{T, want, nblk, (long long) (uintptr_t) d.h, rows ? -1 : bound};
        for (size_t r = 0; r < rseq.size(); ++r) {
            key.push_back(rseq[r]);
            key.push_back(rpos[r] / g.blk + 1);
        }
        auto it = graphs.find(key);
        if (it == graphs.end()) {
            if (o.seqs <= 1) {   // (the graphs of other block counts are behind the position now: dropped)
                for (auto g2 = graphs.begin(); g2 != graphs.end();) {
                    if (g2->first[2] != nblk) {
                        cudaGraphExecDestroy(g2->second.exec);
                        g2 = graphs.erase(g2);
                    } else {
                        ++g2;
                    }
                }
            } else {   // (several conversations' positions: the least recently used beyond kGraphKeep)
                while (graphs.size() >= kGraphKeep) {
                    auto lru = graphs.begin();
                    for (auto g2 = graphs.begin(); g2 != graphs.end(); ++g2)
                        if (g2->second.used < lru->second.used) lru = g2;
                    cudaGraphExecDestroy(lru->second.exec);
                    graphs.erase(lru);
                }
            }
            const long p0 = st.profiled;
            cudaGraph_t gr = nullptr;
            cudaGraphExec_t ex = nullptr;
            bool ok = cudaStreamBeginCapture(d.s, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
            const bool ran = ok && body(true);
            const cudaError_t ee = cudaStreamEndCapture(d.s, &gr);   // (always: the stream leaves capture)
            ok = ok && ran && ee == cudaSuccess && gr != nullptr && cudaGraphInstantiate(&ex, gr, 0) == cudaSuccess;
            if (gr) cudaGraphDestroy(gr);
            const long added = st.profiled - p0;
            st.profiled = p0;
            if (!ok) {   // (nothing of the capture ran: the forward the usual way, and no graphs from now on)
                if (ex) cudaGraphExecDestroy(ex);
                cudaGetLastError();
                graph_on = false;
                info.push_back("NOTE: graphs off: a forward did not capture (" + (err.empty() ? std::string("CUDA") : err) +
                               ") - the kernels launched one by one");
                std::fprintf(stderr, "maya-mm3: %s\n", info.back().c_str());
                err.clear();
                return body(false);
            }
            it = graphs.emplace(key, GraphEntry{ex, added}).first;
        }
        it->second.used = ++graph_clock;
        CK(cudaGraphLaunch(it->second.exec, d.s));
        st.profiled += it->second.profiled;
        return true;
    }

    // a forward's residual rows from GPU a's stream to b's (dst on b): b's stream waits on a's work there through an
    // event, the host goes on enqueueing (CUDA; HIP - whose queued work Windows submits only at a host wait - and
    // MAYA_MM3_HOP_EVENTS=0: the host waits for a's stream first, as before)
    bool hop_events = [] {
#if defined(STRATA_USE_HIP)
        return false;
#else
        const char* v = ::mm3::env("MM3_HOP_EVENTS");
        return v == nullptr || std::atoi(v) != 0;
#endif
    }();
#include "mm3_native_stage_hop.inc"

    bool hop(Dev& a, Dev& b, size_t bytes, float* dst, std::string& err) {
        if(o.native_stage_prefix)return stage_serial_hop(a,b,bytes,dst,err);
        CK(cudaSetDevice(a.id));
        if (hop_events) {
            CK(cudaEventRecord(a.hop, a.s));
            CK(cudaSetDevice(b.id));
            CK(cudaStreamWaitEvent(b.s, a.hop, 0));
        } else {
            CK(cudaStreamSynchronize(a.s));
            CK(cudaSetDevice(b.id));
        }
        return copy_peer(dst, b, a.h, a, bytes, b.s, err);
    }
    // peer access between every two of the GPUs where they have it (load): the forward's copies across then a plain
    // device copy - one a graph can hold - else the runtime's peer copy (through the host where the GPUs cannot)
    bool p2p_all = false;
    bool copy_peer(void* dst, const Dev& db, const void* src, const Dev& sb, size_t bytes, cudaStream_t s,
                   std::string& err) {
        if (p2p_all) CK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s));
        else CK(cudaMemcpyPeerAsync(dst, db.id, src, sb.id, bytes, s));
        return true;
    }

    bool chunk(const int* ids, int T, int pos0, int want, float* logits, std::string& err) {
    if(placement_failed) {err="native placement previously failed; engine cannot resume";return false;}
        return r2::execution_boundary(native,retirement_uncertain,[&] {
            const double tc = now();
            const bool ok=chunk_(ids,T,pos0,want,logits,err);
            if(!ok && o.native_stage_prefix)stage_poisoned=true;
            if(T==1)st.t_token+=now()-tc;
            if(pf.on) {
                if(ok)pcollect(T==1?0:T<=kWin && o.fast_decode?1:2,T,now()-tc);
                pf.used=0;
            }
            return ok;
        });
    }

    // the embeddings of T tokens (rows from the file: BF16, or a GGUF's block type dequantized as ggml does) -> F32 h
    bool embed(const int* ids, int T, float* h, std::string& err) const {
        const Tensor* emb = pk.get(P + "embed_tokens.weight");
        const uint16_t* e16 = (const uint16_t*) pk.data(*emb);
        for (int t = 0; t < T; ++t) {
            if (ids[t] < 0 || ids[t] >= g.vocab) {
                err = "token id " + std::to_string(ids[t]) + " outside the vocabulary";
                return false;
            }
            if (emb->dtype != "BF16") {
                if (!gg::host_row(emb->dtype, pk.data(*emb), g.dim, ids[t], &h[(size_t) t * g.dim])) {
                    err = "the token embedding's type " + emb->dtype + " is not read here";
                    return false;
                }
                continue;
            }
            const uint16_t* r = e16 + (size_t) ids[t] * g.dim;
            for (int i = 0; i < g.dim; ++i) {
                const uint32_t u = (uint32_t) r[i] << 16;
                std::memcpy(&h[(size_t) t * g.dim + i], &u, 4);
            }
        }
        return true;
    }

    // ---- --parallel on a layer split: the GPUs as a pipeline (Model::pipe_tick).  A group - one token of each of its
    // conversations, as forward_rows' rows - enters the first GPU's layers while the groups before it run the later
    // GPUs'.  A tick runs each GPU's group through that GPU's layers, one layer of each GPU in turn (the oldest group
    // first): a GPU's wait for its routing on the host leaves the others running what is queued; then the last GPU's
    // group's logits are read and every other group moves one GPU on.  Each row the same bits as forward_rows' (the
    // window kernels, its attention on the token path's calls over its own cache).
    struct PGroup {
        int tag = -1;   // (-1: no group)
        std::vector<int> ids, seq, pos;
    };
    std::vector<int> stage_dev;          // the GPUs holding layers, in layer order
    std::vector<PGroup> pst;             // the group at each stage
    std::deque<std::pair<int, std::vector<float>>> pout;   // finished groups: tag, logits [rows][vocab]
    int pipe_busy() const {
        int n = stage_state.active ? 1 : 0;
        for (const PGroup& q : pst) n += q.tag >= 0;
        return n;
    }
    void pipe_drop() {
        for (PGroup& q : pst) q.tag = -1;
        pout.clear();
        for (Dev& d : D) {
            cudaSetDevice(d.id);
            cudaStreamSynchronize(d.s);
        }
        cudaGetLastError();
    }
    bool pipe_tick(const int* ids, const int* sq, const int* ps, int R, int tag, std::string& err) {
        if(native && generation_active) {err="native pipeline generation demand accounting is unsupported";return false;}
        if(hf_demand_poisoned) {err="native HF demand boundary previously failed";return false;}
        hf_demand_active = false;
        mm3::DemandBoundary demand_boundary{hf_demand_poisoned,hf_demand_active};
    if(placement_failed) {err="native placement previously failed; engine cannot resume";return false;}
        const int S = (int) stage_dev.size();
        if (pst.size() != (size_t) S) pst.assign((size_t) S, PGroup{});
        if (R > 0) {
            if (pst[0].tag >= 0 || R > kWin || tag < 0) {
                err = "pipe_tick: the first GPU is not free, or a group of " + std::to_string(R) + " rows";
                return false;
            }
            for (int r = 0; r < R; ++r) {   // (a sequence in one row of one group at most)
                bool clash = sq[r] < 0 || sq[r] >= (int) L[0].sk.size() || std::find(sq, sq + r, sq[r]) != sq + r ||
                             ps[r] + 1 > o.max_ctx;
                for (const PGroup& q : pst)
                    if (q.tag >= 0 && std::find(q.seq.begin(), q.seq.end(), sq[r]) != q.seq.end()) clash = true;
                if (clash) {
                    err = "pipe_tick: sequence " + std::to_string(sq[r]) + " at " + std::to_string(ps[r]) +
                          ": busy, out of range or past the context";
                    return false;
                }
            }
            PGroup& q = pst[0];
            q.tag = tag;
            q.ids.assign(ids, ids + R);
            q.seq.assign(sq, sq + R);
            q.pos.assign(ps, ps + R);
            Dev& d0 = D[(size_t) stage_dev[0]];
            if (!embed(ids, R, d0.emb_h, err)) {
                q.tag = -1;
                return false;
            }
            CK(cudaSetDevice(d0.id));
            CK(cudaMemcpyAsync(d0.h, d0.emb_h, (size_t) R * g.dim * 4, cudaMemcpyHostToDevice, d0.s));
        }
        ++fwd; capture::set_forward(fwd); capture::set_layer(-1);
        capture::Range forward_range("forward_pipeline");
        struct CaptureLayerReset { ~CaptureLayerReset(){capture::set_layer(-1);} } capture_layer_reset;
        if (!reset_hf_route_errors(err)) return false;
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            if (!refresh_slots(d, err)) return false;
            if (!adaptive_publish(d,err)) return false;
        }
        for (int s = 0; s < S; ++s) {   // (each GPU's positions: its group's)
            const PGroup& q = pst[(size_t) s];
            if (q.tag < 0) continue;
            Dev& d = D[(size_t) stage_dev[(size_t) s]];
            for (size_t r = 0; r < q.pos.size(); ++r) d.pos_h[r] = q.pos[r];
            CK(cudaSetDevice(d.id));
            CK(cudaMemcpyAsync(d.pos, d.pos_h, q.pos.size() * 4, cudaMemcpyHostToDevice, d.s));
        }
        // (each GPU's turn: the second half of its last layer - the wait for its routing, its experts - then the first
        //  half of its next one, so the GPU has that queued while the host turns to the others)
        std::vector<size_t> at((size_t) S, 0);
        std::vector<int> pend((size_t) S, -1);
        bool ok = true;
        for (bool more = true; more && ok;) {
            more = false;
            for (int s = S - 1; s >= 0 && ok; --s) {
                const PGroup& q = pst[(size_t) s];
                Dev& d = D[(size_t) stage_dev[(size_t) s]];
                if (q.tag < 0) continue;
                rseq = q.seq;
                rpos = q.pos;
                cur_pos = *std::max_element(rpos.begin(), rpos.end());
                const int R = (int) q.seq.size();
                ok = cudaSetDevice(d.id) == cudaSuccess;
                if (ok && pend[(size_t) s] >= 0) {
                    ok = window_post(L[(size_t) pend[(size_t) s]], d, R, err);
                    pend[(size_t) s] = -1;
                    more = true;
                }
                if (ok && at[(size_t) s] < d.layers.size()) {
                    Layer& l = L[(size_t) d.layers[at[(size_t) s]++]];
                    ok = window_pre(l, d, l.dev, R, err);
                    if (l.moe) pend[(size_t) s] = (int) (&l - L.data());
                    more = true;
                }
                k::submit(d.s);
            }
        }
        rseq.clear();
        rpos.clear();
        if (!ok) {
            if (err.empty()) err = "pipe_tick: a layer failed";
            return false;
        }
        const PGroup& last = pst[(size_t) S - 1];
        const int RL = last.tag >= 0 ? (int) last.seq.size() : 0;
        Dev& hd = D[(size_t) head_dev];
        if (RL > 0) {   // (the last GPU's group: the head's GEMV, as a token's)
            CK(cudaSetDevice(hd.id));
            k::rms(hd.h, norm1, RL, g.dim, g.eps, hd.xn, hd.xn16, hd.s);
            const int cols = std::max(1, x3::mv_cols(head));
            for (int c0 = 0; c0 < RL; c0 += cols) {
                x3::MvJob j;
                j.m = &head;
                j.xf = hd.xn + (size_t) c0 * g.dim;
                j.y = hd.lg + (size_t) c0 * g.vocab;
                j.ncol = std::min(cols, RL - c0);
                j.xf_cs = (size_t) g.dim;
                j.y_cs = (size_t) g.vocab;
                if (!x3::mv(&j, 1, hd.s)) {
                    err = "mv head";
                    return false;
                }
            }
            CK(cudaMemcpyAsync(hd.lg_h, hd.lg, (size_t) RL * g.vocab * 4, cudaMemcpyDeviceToHost, hd.s));
        }
        for (Dev& d : D) {
            if(!queue_hf_demand(d,d.hf_snapshot_h,err) || !queue_hf_route_errors(d,err))return false;
        }
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            CK(cudaStreamSynchronize(d.s));
            if(!consume_hf_demand(d,d.hf_snapshot_h,fwd,err))return false;
            if(!check_hf_route_errors(d,err))return false;
        }
        demand_boundary.complete=true;
        if (const int e = x3::launch_errors()) {
            err = "EXL3 launch errors: " + std::to_string(e);
            return false;
        }
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            if(primary_adaptive && !adaptive_policy.legacy &&
               (!retire_foreground_slots(d,err) || !adaptive_publish(d,err)))return false;
            if (!adaptive_issue(d,err)) return false;
            if (!d.promo_q.empty() && !issue_promotions(d,err)) return false;
        }
        if (RL > 0) {
            pout.emplace_back(last.tag, std::vector<float>(hd.lg_h, hd.lg_h + (size_t) RL * g.vocab));
            pst[(size_t) S - 1].tag = -1;
        }
        for (int s = S - 2; s >= 0; --s) {   // (the others one GPU on, the later ones first: each one's room is free)
            PGroup& q = pst[(size_t) s];
            if (q.tag < 0) continue;
            Dev& a = D[(size_t) stage_dev[(size_t) s]];
            Dev& b = D[(size_t) stage_dev[(size_t) s + 1]];
            CK(cudaSetDevice(b.id));
            if (!copy_peer(b.h, b, a.h, a, q.seq.size() * (size_t) g.dim * 4, b.s, err)) return false;
            CK(cudaStreamSynchronize(b.s));
            std::swap(pst[(size_t) s + 1], q);
            q.tag = -1;
        }
        return true;
    }

#include "mm3_native_stage.inc"
#include "mm3_native_prompt.inc"

    // Binary state for the light recorder. Runs only outside the scored generation
    // interval: before its first forward's clock start and after its last one ends.
    void light_map(uint32_t phase) {
        if(!capture::light::full())return;
        light_locations.assign(size_t(g.n_layers)*g.n_exp,0);
        for(int li=0;li<g.n_layers;++li) {
            const auto& l=L[size_t(li)];
            if(!l.moe)continue;
            for(int e=0;e<g.n_exp;++e) {
                uint8_t mask=0;
                if(!l.host_blob.empty() && l.host_blob[size_t(e)])mask|=capture::light::kLocationHost;
                if(l.res_slot[size_t(e)]>=0)mask|=capture::light::kLocationPrimary;
                if(helper_owns(l,e))mask|=capture::light::kLocationHelper;
                light_locations[size_t(li)*g.n_exp+e]=mask;
            }
        }
        light_slots.clear();light_exchanges.clear();
        for(const auto& d:D) {
            for(int s=0;s<d.n_slots;++s) {
                capture::light::SlotState slot{};
                slot.owner=d.slot_owner[size_t(s)];slot.pending=d.slot_pend[size_t(s)];
                slot.issued=uint8_t(d.slot_iss[size_t(s)]!=0);slot.foreground=uint8_t(d.slot_foreground[size_t(s)]!=0);
                slot.reserved=uint8_t(staging_reserved(d,s));
                light_slots.push_back(slot);
            }
            for(const auto& x:d.adaptive)if(x.incoming>=0)
                light_exchanges.push_back({x.incoming,x.victim,int32_t(x.landing),int32_t(x.submitted)});
        }
        capture::light::map(phase,native_demand_rows,g.n_layers,g.n_exp,light_locations,light_slots,light_exchanges,
            helper_live_a,helper_live_b,(helper_live_withdrawn?1u:0u)|(helper_live_published?2u:0u));
    }

    bool chunk_(const int* ids, int T, int pos0, int want, float* logits, std::string& err) {
        const uint64_t light_begin=capture::light::stamp();
        const int replay_head_rows=want;
        if(stage_poisoned || retirement_uncertain.load()) {err="native stage ownership is poisoned";return false;}
        if(hf_demand_poisoned) {err="native HF demand boundary previously failed";return false;}
        hf_demand_active = false;
        mm3::DemandBoundary demand_boundary{hf_demand_poisoned,hf_demand_active};
        const double prefix_host_begin=now();
        double prefix_submit_ms=0;
        if(native && !o.native_stage_prefix) {
            CK(cudaSetDevice(D[0].id));
            CK(cudaEventRecord(D[0].native_prefix_begin,D[0].s));
        }
        if(native && generation_active) { native_demand_rows += T; native_demand_block_rows = T; }
        if(native && !generation_active) { prompt_hash_begin(pos0); prompt_mass_refresh(); }   // (MM3_PROMPT_HASH, MM3_PROMPT_MASS)
        if(native && !generation_active && pos0 == 0 && prompt_check_mode == 2) prompt_u_load();   // (the two-pass reference)
        exchange_forward_begin();   // (exchange_schedule: the 3090's windows close early in the dense prefix)
        ++fwd; capture::set_forward(fwd); capture::set_layer(-1);
        capture::light::set_forward(fwd);
        trace_replay_boundary("forward_begin",-1,T);
        capture::Range forward_range("forward_chunk");
        struct CaptureLayerReset { ~CaptureLayerReset(){capture::set_layer(-1);} } capture_layer_reset;
        if (!reset_hf_route_errors(err)) return false;
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            if (!refresh_slots(d, err)) return false;
            if (!adaptive_publish(d,err)) return false;
            if (!helper_live_progress(d,err)) return false;
        }
        const bool rows = !rseq.empty();   // (forward_rows: row t is sequence rseq[t]'s token at rpos[t])
        cur_pos = rows ? *std::max_element(rpos.begin(), rpos.end()) : pos0 + T - 1;
        if ((rows ? cur_pos + 1 : pos0 + T) > o.max_ctx) {
            err = "the context is full (" + std::to_string(o.max_ctx) + ")";
            return false;
        }
        float* h = D[(size_t) L[0].dev].emb_h;   // (pinned: the previous forward's copy of it is done - it waited)
        if (!embed(ids, T, h, err)) return false;
        for (const Model::Image& im : imgs) {   // (a picture's image tokens: its rows)
            if (rows) break;   // (the pictures are the bound sequence's prompt's)
            const int n = (int) (im.rows.size() / (size_t) g.dim);
            for (int q = std::max(im.start, pos0); q < std::min(im.start + n, pos0 + T); ++q)
                std::memcpy(&h[(size_t) (q - pos0) * g.dim], im.rows.data() + (size_t) (q - im.start) * g.dim,
                            (size_t) g.dim * 4);
        }
        for (Dev& d : D) {
            for (int t = 0; t < T; ++t) d.pos_h[t] = rows ? rpos[(size_t) t] : pos0 + t;
            CK(cudaSetDevice(d.id));
            CK(cudaMemcpyAsync(d.pos, d.pos_h, (size_t) T * 4, cudaMemcpyHostToDevice, d.s));
        }
        int cur = L[0].dev;
        CK(cudaSetDevice(D[(size_t) cur].id));
        if(replay_trace_active)CK(cudaEventRecord(replay_gpu_events[7],D[size_t(cur)].s));
        CK(cudaMemcpyAsync(D[(size_t) cur].h, h, (size_t) T * g.dim * 4, cudaMemcpyHostToDevice, D[(size_t) cur].s));
        pmark(D[(size_t) cur], kPAttnIn);   // (MAYA_MM3_PROF: the forward's start)
        // a token's or a verify window's rows through the head's GEMV (the same bits for each row), else the prompt's
        // head below
        const bool fast_head = want > 0 && want <= kWin && o.fast_decode;
        // the layers (and a fast head's GEMV): enqueued as they come, or (cap) captured into a graph
        auto body = [&](bool cap) -> bool {
            for (int li = 0; li < g.n_layers; ++li) {
                const int dv = L[(size_t) li].dev;
                if (dv != cur) {
                    if (!hop(D[(size_t) cur], D[(size_t) dv], (size_t) T * g.dim * 4, D[(size_t) dv].h, err)) return false;
                    cur = dv;
                }
                CK(cudaSetDevice(D[(size_t) dv].id));
                if(replay_trace_active && li==3)CK(cudaEventRecord(replay_gpu_events[8],D[size_t(dv)].s));
                if (!layer(li, T, pos0 + T - 1, err)) return false;
                if (prompt_hash_file && !generation_active && !cap &&
                    !prompt_hash_rows('h', li, D[(size_t) dv].h, T, pos0, D[(size_t) dv].s, err)) return false;
                // (a token's or a window's layer: Windows + AMD start its queued work now - a host-free layer has no
                //  wait of its own that would, so the GPU runs this layer while the host enqueues the next)
                if (!cap && T <= kWin && o.fast_decode) k::submit(D[(size_t) dv].s);
                if (draftm && draft_running && tap_of[(size_t) li] >= 0) {   // (the draft's context: this layer's output rows)
                    Dev& a = D[(size_t) dv];
                    Dev& hd = D[(size_t) head_dev];
                    float* dst = taps + (size_t) tap_of[(size_t) li] * T * g.dim;
                    if (dv == head_dev)
                        CK(cudaMemcpyAsync(dst, a.h, (size_t) T * g.dim * 4, cudaMemcpyDeviceToDevice, a.s));
                    else if (!copy_peer(dst, hd, a.h, a, (size_t) T * g.dim * 4, a.s, err))
                        return false;
                }
                if (const char* dp = ::mm3::env("MM3_DUMP_HS")) {   // (debugging: every layer's residual stream, F32 rows)
                    std::vector<float> hh((size_t) T * g.dim);
                    CK(cudaMemcpyAsync(hh.data(), D[(size_t) dv].h, hh.size() * 4, cudaMemcpyDeviceToHost,
                                       D[(size_t) dv].s));
                    CK(cudaStreamSynchronize(D[(size_t) dv].s));
                    if (std::FILE* f = std::fopen(dp, li == 0 && pos0 == 0 ? "wb" : "ab")) {
                        std::fwrite(hh.data(), 4, hh.size(), f);
                        std::fclose(f);
                    }
                }
                if (o.verbose) {
                    CK(cudaStreamSynchronize(D[(size_t) dv].s));
                    const int e = x3::launch_errors();
                    if (e) {
                        err = "EXL3 launch errors: " + std::to_string(e);
                        return false;
                    }
                }
                if(native && !o.native_stage_prefix && li==7) {
                    CK(cudaEventRecord(D[0].native_prefix_end,D[0].s));
                    prefix_submit_ms=1e3*(now()-prefix_host_begin);
                }
            }
            if(replay_trace_active)CK(cudaEventRecord(replay_gpu_events[9],D[size_t(cur)].s));
            if (want <= 0) return true;
            Dev& a = D[(size_t) cur];
            Dev& hd = D[(size_t) head_dev];
            if (cur != head_dev &&   // (the last layer on another GPU: its rows across; on the head's GPU stream order holds)
                !hop(a, hd, (size_t) T * g.dim * 4, hd.h, err))
                return false;
            CK(cudaSetDevice(hd.id));
            const float* last = hd.h + (size_t) (T - want) * g.dim;
            k::rms(last, norm1, want, g.dim, g.eps, hd.xn, hd.xn16, hd.s);
            if (fast_head) {
                const int cols = std::max(1, x3::mv_cols(head));   // (the split partials' scratch: ~5 rows a pass)
                for (int c0 = 0; c0 < want; c0 += cols) {
                    x3::MvJob j;
                    j.m = &head;
                    j.xf = hd.xn + (size_t) c0 * g.dim;
                    j.y = hd.lg + (size_t) c0 * g.vocab;
                    j.ncol = std::min(cols, want - c0);
                    j.xf_cs = (size_t) g.dim;
                    j.y_cs = (size_t) g.vocab;
                    if (!x3::mv(&j, 1, hd.s)) {
                        err = "mv head";
                        return false;
                    }
                }
            }
            return true;
        };
        if (native && graph_ready(T,want) &&
            std::any_of(D.begin(),D.end(),[](const Dev& d){return d.hf_original_d!=nullptr;})) hf_demand_active=true;
        const uint64_t light_prepare_end=capture::light::stamp();
        if (!(graph_ready(T, want) ? run_graph(T, want, body, err) : body(false))) return false;
        const uint64_t light_body_end=capture::light::stamp();
        const int light_want=want;
        if (want > 0) {
            Dev& hd = D[(size_t) head_dev];
            CK(cudaSetDevice(hd.id));
            if (fast_head && nuc_want) {   // (the GPU nucleus: candidates and stats back, the rows stay on the device)
                pmark(hd, kPHead);
                k::topk_rows(hd.lg, want, g.vocab, k::kCandCap, nullptr, 0, hd.nuc_c, hd.nuc_topk, hd.s);
                k::row_stats(hd.lg, want, g.vocab, hd.nuc_s, hd.nuc_scratch, hd.s);
                CK(cudaStreamSynchronize(hd.s));
                nuc_rows = want;
                want = 0;
            } else if (fast_head) {
                pmark(hd, kPHead);
                CK(cudaMemcpyAsync(hd.lg_h, hd.lg, (size_t) want * g.vocab * 4, cudaMemcpyDeviceToHost, hd.s));
                CK(cudaStreamSynchronize(hd.s));
                std::memcpy(logits, hd.lg_h, (size_t) want * g.vocab * 4);
                nuc_rows = 0;
                want = 0;
            }
            // the head's output kHeadRows rows at a time
            for (int r0 = 0; r0 < want; r0 += kHeadRows) {
                const int nr = std::min<int>(kHeadRows, want - r0);
                x3::gemm_rows(head, hd.xn16 + (size_t) r0 * g.dim, g.dim, nr, hd.lg, g.vocab, 0.0f, hd.s);
                CK(cudaMemcpyAsync(hd.lg_h, hd.lg, (size_t) nr * g.vocab * 4, cudaMemcpyDeviceToHost, hd.s));
                CK(cudaStreamSynchronize(hd.s));
                std::memcpy(logits + (size_t) r0 * g.vocab, hd.lg_h, (size_t) nr * g.vocab * 4);
            }
            if (want > 0) pmark(hd, kPHead);
        }
        for (Dev& d : D) {
            if(!queue_hf_demand(d,d.hf_snapshot_h,err) || !queue_hf_route_errors(d,err))return false;
        }
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            if(replay_trace_active)CK(cudaEventRecord(replay_gpu_events[10],d.s));
            CK(cudaStreamSynchronize(d.s));
            if(!consume_hf_demand(d,d.hf_snapshot_h,fwd,err))return false;
            if(!check_hf_route_errors(d,err))return false;
        }
        demand_boundary.complete=true;
        const uint64_t light_sync_end=capture::light::stamp();
        if(native && !o.native_stage_prefix) {
            if(!trace_gpu_finished(err))return false;
            if(replay_trace_active) {
                float prefix=0,tail=0;
                CK(cudaEventElapsedTime(&prefix,replay_gpu_events[7],replay_gpu_events[8]));
                CK(cudaEventElapsedTime(&tail,replay_gpu_events[9],replay_gpu_events[10]));
                capture::detail_event("target_overhead_observation","\"rows\":"+std::to_string(T)+
                    ",\"head_rows\":"+std::to_string(replay_head_rows)+",\"prefix_ms\":"+std::to_string(prefix)+",\"tail_ms\":"+std::to_string(tail));
            }
            float elapsed=0;
            CK(cudaEventElapsedTime(&elapsed,D[0].native_prefix_begin,D[0].native_prefix_end));
            st.prefix8_gpu_ms+=elapsed;st.prefix8_submit_ms+=prefix_submit_ms;++st.prefix8_calls;
        }
        const int e = x3::launch_errors();
        if (e) {
            err = "EXL3 launch errors: " + std::to_string(e);
            return false;
        }
        for (Dev& d : D) {
            CK(cudaSetDevice(d.id));
            if(primary_adaptive && !adaptive_policy.legacy &&
               (!retire_foreground_slots(d,err) || !adaptive_publish(d,err)))return false;
            if (!helper_live_progress(d,err) || !helper_live_issue(d,err)) return false;
            if (!adaptive_issue(d,err)) return false;
            if (!d.promo_q.empty() && !issue_promotions(d,err)) return false;
        }
        if (!exchange_forward_end(err)) return false;   // (exchange_schedule: owed downloads out, the 3090's windows open)
        trace_replay_boundary("forward_end",-1,T);
        if (draftm && draft_running && !rows) {
            CK(cudaSetDevice(D[(size_t) head_dev].id));
            if (prompt_hash_file && !generation_active &&
                !prompt_hash_taps(taps, T, pos0, (size_t) T, D[(size_t) head_dev].s, err)) return false;
            if (!draftm->context(taps, T, pos0, err)) return false;
            pmark(D[(size_t) head_dev], kPDraft);
        }
        if(capture::light::active()) {
            int64_t counters[capture::light::kForwardCounters]={};
            if(capture::light::full()) {
                const int64_t current[capture::light::kForwardCounters]={
                    st.staged,st.foreground_queries,st.foreground_not_ready,st.adaptive_queries,st.adaptive_not_ready,
                    st.adaptive_source_busy,st.adaptive_issued,st.adaptive_published,st.adaptive_reused,st.adaptive_no_gain,
                    st.adaptive_no_slot,int64_t(st.adaptive_h2d_bytes),int64_t(st.adaptive_d2h_bytes),
                    int64_t(helper_live_started),int64_t(helper_live_withdrawals),int64_t(helper_live_publications),
                    int64_t(helper_live_finished),st.primary_submit_queries};
                std::memcpy(counters,current,sizeof(counters));
            }
            capture::light::forward_record(light_begin,light_prepare_end,light_body_end,light_sync_end,
                native_demand_rows,pos0,T,light_want,generation_active,counters);
        }
        return true;
    }
};

Model::Model() : p(new Impl) {}
Model::~Model() {
    delete tpe;
    r2::destroy_after_retire(p, [&] { return p->retire_for_destroy(); },
        [](Impl* model) { delete model; },
        [] { std::fprintf(stderr, "MM3: retaining model after failed CUDA retirement\n"); });
}
bool Model::load(const Options& o_in, std::string& err) {
    if(p->stage_state.active) {err="native stage lease is active";return false;}
    Options o = o_in;
    bool native_pack = false;
    { std::ifstream f(o.pack + "/mm3.txt"); for(std::string line;std::getline(f,line);)
        if(line=="format native") native_pack=true; }
    if(o.native_stage_prefix && !native_stage_chunk_supported(o.chunk)) {
        err="native stage requires chunk between 4 and 16";return false;
    }
    const bool c0_stage = o.native_stage_prefix==3 && !o.native_cooperative;
    if(o.native_stage_prefix && (o.native_stage_prefix!=3 || !native_pack || o.tp || o.seqs!=1 || !o.dspark || o.gpus!=std::vector<int>{1,0} || o.layer_split!=std::vector<int>{3} ||
       (o.native_cooperative ? (o.native_helper!=1 || o.native_helper_gb<=0) : (o.native_helper!=-1 || o.native_helper_gb!=0 || o.native_primary_arithmetic!="fp4")) || o.native_helper_dispatch!="worker" || o.routing_prefetch || o.staging!=8 || o.kv_resident)) {
        err="native stage requires prefix 3, GPUs 1,0, split 3, DSpark, worker dispatch, eight staging slots and no KV streaming/prefetch; cooperative mode requires helper 1, cooperation-off requires FP4 and helper -1 with zero budget";return false;
    }
    if(!o.native_cooperative && (!native_pack || o.tp>0 || o.seqs!=1 || (o.gpus.size()>1 && !c0_stage) || o.routing_prefetch || o.native_helper_dispatch!="worker" || o.chunk>16)) { err="native cooperation-off requires single-primary or the explicit three-layer stage, single-sequence native worker mode, chunk <= 16, and prefetch off"; return false; }
    if(o.routing_prefetch && (!native_pack || o.tp>0 || o.seqs!=1 || o.gpus.size()>1 || o.native_helper_dispatch!="worker")) {
        err="routing prefetch requires single-sequence native single-primary worker serving";return false;
    }
    if(o.native_helper_dispatch=="queue" && (!native_pack || o.tp>0 || o.seqs!=1 || o.gpus.size()>1)) {
        err="queued helper requires single-sequence native single-primary serving";return false;
    }
    if(o.native_primary_arithmetic!="dp4a" && o.native_primary_arithmetic!="fp4") {
        err="native primary arithmetic must be dp4a or fp4";return false;
    }
    if(o.native_primary_arithmetic_given || o.native_primary_arithmetic=="fp4") {
        if(!native_pack || o.tp>0 || o.seqs!=1 || (o.gpus.size()>1 && !o.native_stage_prefix)) {
            err="native primary arithmetic control requires single-sequence, single-primary native MiniMax without TP";return false;
        }
        o.native_primary_arithmetic_given=true;
    }
    if(native_pack) {
        if(o.tp>0) { err="native MiniMax does not use the EXL3 tensor-parallel route"; return false; }
        if(o.gpus.empty()) o.gpus={0};
        if(!o.pin_pool) { err="native MiniMax requires its exclusive resident host bank"; return false; }
    }
    if (o.gpus.empty()) {   // (no --gpus: the GPUs this engine runs on - compute capability 7.0 or newer; a display card
                            //  beside them, e.g. a Quadro P400, is left out with a NOTE)
        int n = 0;
        if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) {
            err = "no CUDA device";
            return false;
        }
        std::vector<std::string> skipped;
        for (int i = 0; i < n; ++i) {
            cudaDeviceProp pr{};
            if (cudaGetDeviceProperties(&pr, i) != cudaSuccess) continue;
            if (pr.major * 10 + pr.minor >= 70) o.gpus.push_back(i);
            else skipped.push_back("CUDA" + std::to_string(i) + " (" + pr.name + ", sm_" + std::to_string(pr.major) +
                                   std::to_string(pr.minor) + ")");
        }
        if (o.gpus.empty()) {
            err = "no GPU of compute capability 7.0 or newer (MiniMax-M3.1's engine needs a V100 or newer)";
            return false;
        }
        for (const auto& s : skipped)
            std::fprintf(stderr, "maya-mm3: NOTE: %s left out - compute capability below 7.0 (--gpus to choose)\n",
                         s.c_str());
    }
    // the prefill calibration of this PC (maya-mm3 --calibrate-prefill, the setup's calibration step): its chunk and
    // GEMM backend, where --chunk / --prefill-gemm do not choose; none: the engine's defaults, with a NOTE
    p->tkey = tune_key_of(o.gpus);
    p->tfile = o.prefill_tune.empty() ? o.pack + "/prefill-tune.txt" : o.prefill_tune;
    std::string tuned;
    if (o.tune && o.tp <= 0) {
        std::ifstream f(p->tfile);
        for (std::string line; std::getline(f, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t tab = line.find('\t');
            if (tab == std::string::npos || line.compare(0, tab, p->tkey) != 0) continue;
            std::istringstream is(line.substr(tab + 1));
            for (std::string kv; is >> kv;) {
                const size_t eq = kv.find('=');
                if (eq == std::string::npos) continue;
                const std::string k = kv.substr(0, eq);
                const int v = std::atoi(kv.c_str() + eq + 1);
                if (k == "chunk" && !o.chunk_given && v >= 64 && v <= 16384) o.chunk = v;
                else if (k == "gemm" && o.prefill_gemm < 0) o.prefill_gemm = v;
            }
            tuned = line.substr(tab + 1);
        }
    }
    if (o.tp > 0 && o.seqs > 1) {   // (--parallel: the layer split's batched steps - the tensor-parallel path runs one)
        Options o2 = o;
        o2.tp = 0;
        const bool ok = load(o2, err);
        p->info.insert(p->info.begin(), "NOTE: --tp " + std::to_string(o.tp) + " with --parallel " +
                                            std::to_string(o.seqs) + ": the layer split instead (the tensor-parallel "
                                            "path runs one conversation at a time)");
        return ok;
    }
    if (o.tp > 0) {   // (a GGUF pack: the layer split - the tensor-parallel path splits EXL3 matrices only)
        bool gg_pack = false;
        std::ifstream f(o.pack + "/mm3.txt");
        for (std::string line; std::getline(f, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            gg_pack = gg_pack || line == "format gguf";
        }
        if (gg_pack) {
            Options o2 = o;
            o2.tp = 0;
            const bool ok = p->load(o2, err);
            p->info.insert(p->info.begin(), "NOTE: --tp " + std::to_string(o.tp) + ": not with a GGUF pack - the layer "
                                            "split instead");
            return ok;
        }
    }
#if defined(_WIN32)
    // Windows: the tensor-parallel path's host waits on the GPUs' counters depend on its launches being submitted as
    // they are enqueued, which the WDDM driver batches - the layer split there (MAYA_MM3_TP_WINDOWS=1: try it anyway)
    if (o.tp > 0 && !(std::getenv("MAYA_MM3_TP_WINDOWS") && std::atoi(std::getenv("MAYA_MM3_TP_WINDOWS")) != 0)) {
        Options o2 = o;
        o2.tp = 0;
        const bool ok = p->load(o2, err);
        p->info.insert(p->info.begin(), "--tp " + std::to_string(o.tp) + ": not on Windows - the layer split instead");
        return ok;
    }
#endif
    if (o.tp > 0) {
        tpe = new TpEngine;
        if (tpe->load(o, err)) return true;
        // (the tensor-parallel path does not fit this PC - its memory, its GPUs: the layer split instead, which sizes
        //  itself to the memory, with a NOTE; its own error when that does not load either)
        const std::string why = err;
        delete tpe;   // (frees its GPU memory and pinned pool first)
        tpe = nullptr;
        Options o2 = o;
        o2.tp = 0;
        err.clear();
        std::fprintf(stderr, "maya-mm3: --tp %d did not load (%s): the layer split instead\n", o.tp, why.c_str());
        if (!p->load(o2, err)) {
            err = "--tp " + std::to_string(o.tp) + ": " + why + "; the layer split: " + err;
            return false;
        }
        p->info.insert(p->info.begin(), "NOTE: --tp " + std::to_string(o.tp) + " did not fit (" + why +
                                            "): the layer split instead");
        return true;
    }
    if (!p->load(o, err)) return false;
    if(o.routing_prefetch) {
        p->o.routing_prefetch=false;
        if(!set_routing_prefetch(true,err))return false;
    }
    if (!tuned.empty())
        p->info.push_back("prefill: this PC's calibration (" + p->tfile + "): chunk " + std::to_string(p->o.chunk) +
                          (p->gguf ? std::string(", ") + (gg::gemm_backend() == 1 ? "hipBLASLt" : "BLAS") + " GEMMs" : ""));
    else if (o.tune && o.tp <= 0)
        p->info.push_back("NOTE: prefill not calibrated on this PC (./maya.sh --calibrate measures it): chunk " +
                          std::to_string(p->o.chunk));
    return true;
}
int Model::chunk() const { return tpe ? 0 : p->run_chunk(); }
bool Model::set_routing(bool enabled,float ratio,std::string& err,bool prior,float prior_bonus,bool dampen) {
    if(!tpe && p->native && p->retirement_uncertain.load()) {err="native ownership is poisoned";return false;}
    if(!tpe && p->placement_failed) {err="native placement previously failed; engine cannot resume";return false;}
    if(!std::isfinite(ratio)||ratio<0||ratio>1) { err="routing ratio must be finite in [0,1]"; return false; }
    if(!std::isfinite(prior_bonus)||prior_bonus<0||prior_bonus>1) { err="routing prior bonus must be finite in [0,1]"; return false; }
    if(tpe || !p->native) {
        if(enabled || dampen) { err="residency-aware routing requires the native MiniMax path"; return false; }
        return true;
    }
    if(pipe_busy()) { err="cannot change routing while pipeline work is pending"; return false; }
    return r2::execution_boundary(p->native,p->retirement_uncertain,[&] {
        for(auto& d:p->D) {
            CK(cudaSetDevice(d.id));
            if(p->o.routing_prefetch) {
                if(!p->prefetch_control_join(d.s,err) || (d.s2 && !p->prefetch_control_join(d.s2,err)))return false;
            } else {
                auto rc=cudaStreamSynchronize(d.s);
                if(rc==cudaSuccess && d.s2)rc=cudaStreamSynchronize(d.s2);
                if(rc!=cudaSuccess) {
                    if(p->native) {p->retirement_uncertain.store(true);if(p->o.native_stage_prefix)p->stage_poisoned=true;}
                    err=std::string("routing control retirement: ")+cudaGetErrorString(rc);return false;
                }
            }
        }
        if(!enabled && p->o.routing_prefetch)for(auto& d:p->D)
            while(p->prefetch_cancel_one(d,true)) {}
        p->graphs_clear();
        p->o.routing=enabled; p->o.routing_ratio=ratio; p->o.routing_prior=prior; p->o.routing_prior_bonus=prior_bonus; p->o.routing_dampen=dampen;
        return true;
    });
}
bool Model::routing_dampen_supported() const { return !tpe && p->native; }
bool Model::routing_prefetch_supported() const {return !tpe && p->prefetch_supported();}
bool Model::routing_prefetch() const {return !tpe && p->o.routing_prefetch;}
bool Model::set_routing_prefetch(bool enabled,std::string& err) {
    if(tpe) {if(enabled)err="routing prefetch is unavailable with tensor parallel";return !enabled;}
    if(pipe_busy() || p->generation_active) {err="routing prefetch requires an idle request boundary";return false;}
    if(p->placement_failed || p->retirement_uncertain.load() || !p->promotion_error.empty()) {
        err="native cache ownership is poisoned";return false;
    }
    if(enabled && !p->o.native_cooperative) {err="routing prefetch is unsupported with cooperation off";return false;}
    if(enabled && !p->prefetch_prepare(err))return false;
    if(p->o.routing_prefetch!=enabled) {
        for(auto& d:p->D) {
            CK(cudaSetDevice(d.id));
            if(!p->prefetch_control_join(d.s,err) || (d.s2 && !p->prefetch_control_join(d.s2,err)))return false;
        }
        if(!enabled)for(auto& d:p->D)while(p->prefetch_cancel_one(d,true)) {}
        p->graphs_clear();p->o.routing_prefetch=enabled;
    }
    return true;
}
uint64_t Model::routing_prefetch_outstanding_bytes() const {
    if(tpe)return 0;
    uint64_t bytes=0;
    for(const auto& d:p->D)for(size_t s=0;s<d.slot_prediction.size();++s)
        if((d.slot_prediction[s]==1 && d.slot_iss[s]) || d.slot_prediction[s]==2 || d.slot_prediction[s]==4)bytes+=d.slot_bytes;
    return bytes;
}
size_t Model::routing_prefetch_device_bytes() const {
    if(tpe || p->D.empty())return 0;
    const auto& d=p->D[0];
    return (d.prediction_logits?size_t(kWin)*p->g.n_exp*sizeof(float):0) +
           (d.prediction_ids?size_t(kWin)*p->g.k_exp*sizeof(int):0) +
           (d.prediction_weights?size_t(kWin)*p->g.k_exp*sizeof(float):0);
}
bool Model::native_cooperative() const { return !tpe && p->native && p->o.native_cooperative; }
HelperWeightTraffic Model::helper_weight_traffic() const {
    if(tpe || !p->helper)return {};
    const auto t=p->helper->weight_traffic();
    return {t.h2d_issued_ops,t.h2d_issued_bytes,t.d2h_issued_ops,t.d2h_issued_bytes,t.replacements_completed,t.copy_failures};
}
bool Model::helper_live_pending() const { return !tpe && p->helper_live_a>=0; }
bool Model::cooperative_stage_reuse() const { return native_cooperative() && p->o.cooperative_stage_reuse; }
bool Model::set_cooperative_stage_reuse(bool enabled, std::string& err) {
    if (!native_cooperative()) { err="stage reuse requires native cooperative execution"; return false; }
    if(enabled && (!p->batch_staging_costs.valid || p->batch_staging_arithmetic!=p->native_primary_current ||
                   p->helper_dispatch_current!="worker")) {
        err="batch staging requires matching grouped calibration and worker helper; reload the supported baseline";return false;
    }
    p->o.cooperative_stage_reuse=enabled;
    return true;
}
bool Model::native_helper_exclusive() const { return !tpe && p->native && bool(p->helper); }
double Model::cooperative_cpu_scale() const {return native_cooperative()?p->o.cooperative_cpu_scale:1.0;}
void Model::set_helper_occupancy(bool on) {if(!tpe && p->helper)p->helper->set_occupancy(on);if(!tpe && p->third)p->third->set_occupancy(on && p->third_on);}
void Model::set_helper_fast(bool on) {if(!tpe && p->helper)p->helper->set_fast(on);if(!tpe && p->third)p->third->set_fast(on);}
bool Model::helper_fast() const {return !tpe && p->helper && p->helper->fast();}
void Model::set_helper_kernel(int multi, int one) {if(!tpe && p->helper)p->helper->set_kernel(multi,one);}
void Model::set_third_kernel(int multi, int one) {if(!tpe && p->third)p->third->set_kernel(multi,one);}
int Model::helper_kernel(bool one) const {return !tpe && p->helper ? (one ? p->helper->kernel_one() : p->helper->kernel_multi()) : 0;}
int Model::third_kernel(bool one) const {return !tpe && p->third ? (one ? p->third->kernel_one() : p->third->kernel_multi()) : 0;}
void Model::set_helper_kernel_loads(int loads) {if(!tpe && p->helper)p->helper->set_loads(loads);}
void Model::set_third_kernel_loads(int loads) {if(!tpe && p->third)p->third->set_loads(loads);}
int Model::helper_kernel_loads() const {return !tpe && p->helper ? p->helper->loads() : 1;}
int Model::third_kernel_loads() const {return !tpe && p->third ? p->third->loads() : 1;}
long long Model::helper_kernel_launches() const {return !tpe && p->helper ? (long long)p->helper->kernel_launches() : 0;}
long long Model::third_kernel_launches() const {return !tpe && p->third ? (long long)p->third->kernel_launches() : 0;}
void Model::set_helper_zero_copy(int mode) {if(!tpe && p->helper)p->helper->set_zero_copy(mode);if(!tpe && p->third)p->third->set_zero_copy(mode);}
// third: the third card's fixed pool (mm3_native_third.inc)
bool Model::third_control() const {return !tpe && p->native && bool(p->third);}
bool Model::third() const {return !tpe && p->third_on;}
void Model::set_third(bool on) {if(!tpe)p->third_on=on && p->third && p->third_filled;}
void Model::set_routing_prompt_bonus(float bonus) {if(!tpe)p->prompt_bonus=bonus>=0.0f?bonus:-1.0f;}
float Model::routing_prompt_bonus() const {return tpe?-1.0f:p->prompt_bonus;}
bool Model::native_third_fill(const std::vector<uint32_t>& profile, std::string& report, std::string& err) {
    if(tpe || pipe_busy()>0 || p->generation_active || p->stage_state.active || native_state_poisoned()) {
        err="filling the third pool requires a healthy idle native engine";return false;
    }
    return p->third_fill(profile,report,err);
}
bool Model::native_placement_three(const std::vector<uint32_t>& profile, const std::string& order, std::string& report, std::string& err) {
    if(tpe || pipe_busy()>0 || p->generation_active || p->stage_state.active || native_state_poisoned()) {
        err="three-card placement requires a healthy idle native engine";return false;
    }
    return p->placement_three(profile,order,report,err);
}
int Model::helper_zero_copy() const {return !tpe && p->helper ? p->helper->zero_copy() : 0;}
bool Model::helper_occupancy_control() const {return native_cooperative() && !tpe && bool(p->helper);}
long long Model::helper_occupancy_launches() const {return !tpe && p->helper ? (long long)p->helper->occupancy_launches() : 0;}
bool Model::early_copy_control() const {return native_cooperative() && !tpe && p->D.size()==1 && p->D[0].cs;}
int Model::early_copy() const {return early_copy_control() ? p->o.early_copy : 0;}
void Model::set_early_copy_legacy(bool on) {if(!tpe)p->o.early_copy_legacy=on;}
bool Model::early_copy_legacy() const {return !tpe && p->o.early_copy_legacy;}
bool Model::set_early_copy(int mode,std::string& err) {
    if(mode<0 || mode>3) {err="early copy mode must be 0, 1, 2 or 3 (3: predict only)";return false;}
    if(mode && (!early_copy_control() || pipe_busy())) {err="early copy requires idle single-GPU native cooperative execution";return false;}
    if(mode && p->o.routing_prefetch) {err="early copy and routing prefetch are exclusive";return false;}
    p->o.early_copy=mode;return true;
}
bool Model::prep_graph_control() const {return native_cooperative() && !tpe && p->D.size()==1;}
int Model::prep_graph() const {return prep_graph_control() ? p->o.prep_graph : 0;}
bool Model::route_lab_control() const {return !tpe && p->native && p->o.native_cooperative && p->D.size()==1 && !p->graph_on;}
bool Model::set_route_lab(const route_lab::Settings& s,std::string& err) {
    if(tpe || !p) {if(s.on)err="route lab requires the native MiniMax path";return !s.on;}
    route_lab::Settings next=s;
    next.bonus=p->lab_kernel_bonus();
    if(!next.on) next=route_lab::Settings{};
    if(std::all_of(next.layer_scale.begin(),next.layer_scale.end(),[](float x){return x==1.0f;})) next.layer_scale.clear();
    if(next.on) {
        if(!route_lab_control()) {err="route lab requires single-GPU native cooperative execution without the forward graph";return false;}
        if(p->o.routing && !p->o.routing_prior) {err="route lab requires routing=prior (or off), not substitute";return false;}
        if(p->helper_dispatch_current=="queue") {err="route lab requires helper_dispatch=worker";return false;}
        if(p->o.routing_prefetch || p->predict_diag_on()) {err="route lab requires routing_prefetch off and no prediction diagnostic";return false;}
        if(next.prompt && p->prompt_layerwise_mode) {err="route_lab_prompt requires prompt_layerwise 0 (the layer-by-layer reader plans on its own)";return false;}
        if(next.protect<0 || next.protect>2 || next.weights<0 || next.weights>2 || next.k_drop_ranks<1 || next.k_drop_ranks>2) {
            err="route lab setting out of range";return false;
        }
        if(pipe_busy()) {err="cannot change the route lab while pipeline work is pending";return false;}
        for(auto& d:p->D) if(!p->lab_buffers(d,err)) return false;
        if(!next.layer_scale.empty() && next.layer_scale.size()!=p->L.size()) {err="route_layer_scale: one value a model layer";return false;}
    }
    p->lab_map_layers();
    p->lab=next;
    return true;
}
route_lab::Settings Model::route_lab_settings() const {return tpe ? route_lab::Settings{} : p->lab_settings_now();}
std::vector<int> Model::moe_layers() const {
    std::vector<int> out;
    if(tpe) return out;
    for(size_t i=0;i<p->L.size();++i) if(p->L[i].moe) out.push_back(int(i));
    return out;
}
int Model::layer_count() const {return tpe ? 0 : int(p->L.size());}
void Model::route_lab_request_begin() {
    if(!tpe){p->lab_req=route_lab::RequestStats{};p->lab_req_prompt=route_lab::RequestStats{};p->lab_rows.clear();p->lab_rows_fwd=-1;}
}
std::string Model::route_lab_summary_json() const {return tpe || !p->lab.on ? "null" : p->lab_summary_json();}
std::string Model::route_lab_settings_json() const {return route_lab::settings_json(route_lab_settings());}
bool Model::set_prep_graph(int mode,std::string& err) {
    if(mode<0 || mode>2) {err="prep graph mode must be 0, 1 or 2 (2: verify)";return false;}
    if(mode && (!prep_graph_control() || pipe_busy())) {err="prep graph requires idle single-GPU native cooperative execution";return false;}
    p->o.prep_graph=mode;return true;
}
bool Model::expert_graph_control() const {return native_cooperative() && !tpe && p->D.size()==1;}
int Model::expert_graph() const {return expert_graph_control() ? p->o.expert_graph : 0;}
bool Model::set_expert_graph(int mode,std::string& err) {
    if(mode<0 || mode>2) {err="expert graph mode must be 0, 1 or 2 (2: verify)";return false;}
    if(mode && (!expert_graph_control() || pipe_busy())) {err="expert graph requires idle single-GPU native cooperative execution";return false;}
    p->o.expert_graph=mode;return true;
}
void Model::set_dflash2_debug(int bits) {if(!tpe && p->draftm)p->draftm->set_debug(bits);}
bool Model::dense_split_control() const {return !tpe && p->native;}
int Model::dense_split() const {return dense_split_control() ? p->o.dense_split : 0;}
bool Model::set_dense_split(int mode,std::string& err) {
    if(mode<0 || mode>1) {err="dense split must be 0 or 1";return false;}
    if(mode && !dense_split_control()) {err="dense split requires the native engine";return false;}
    if(mode==p->o.dense_split)return true;
    if(pipe_busy()) {err="cannot change the dense split while pipeline work is pending";return false;}
    // (it changes the k / v / indexer-q launches: graphs captured under the other setting are dropped)
    if(p->graph_on)p->graphs_clear();
    p->prep_graphs_clear();
    nd::set_split_n(mode ? nd::kSplitN : nd::kBand);
    p->o.dense_split=mode;return true;
}
size_t Model::device_free_bytes() const {
    if (tpe || p->D.empty()) return 0;
    int cur = 0; cudaGetDevice(&cur);
    size_t fr = 0, tot = 0;
    if (cudaSetDevice(p->D[0].id) != cudaSuccess || cudaMemGetInfo(&fr, &tot) != cudaSuccess) fr = 0;
    cudaSetDevice(cur);
    return fr;
}
bool Model::prompt_layerwise_control() const {return !tpe && p->prompt_layerwise_supported();}
int Model::prompt_layerwise() const {return tpe ? 0 : p->prompt_layerwise_mode;}
bool Model::set_prompt_layerwise(int mode,std::string& err) {
    if(mode<0 || mode>1) {err="prompt_layerwise must be 0 or 1";return false;}
    if(mode && !prompt_layerwise_control()) {err="prompt_layerwise requires single-GPU native cooperative serving with the 3090 helper";return false;}
    if(!tpe)p->prompt_layerwise_mode=mode;
    return true;
}
int Model::prompt_check() const {return tpe ? 0 : p->prompt_check_mode;}
int Model::prompt_two_pass() const {return tpe ? 0 : p->prompt_two_pass_mode;}
int Model::prompt_helper_pass() const {return tpe ? 16 : p->prompt_helper_pass_mode;}
bool Model::set_prompt_helper_pass(int tokens,std::string& err) {
    const int cap = tpe || !p->helper ? 16 : p->helper->pass_tokens();
    if(tokens<16 || tokens>cap || tokens%16) {err="prompt_helper_pass must be a multiple of 16 from 16 to "+std::to_string(cap)+" (MM3_HELPER_PASS_TOKENS at load)";return false;}
    if(!tpe)p->prompt_helper_pass_mode=tokens;
    return true;
}
bool Model::set_prompt_two_pass(int mode,std::string& err) {
    if(mode<0 || mode>1) {err="prompt_two_pass must be 0 or 1";return false;}
    if(mode && !prompt_layerwise_control()) {err="prompt_two_pass requires the layer-by-layer prompt read";return false;}
    if(!tpe)p->prompt_two_pass_mode=mode;
    return true;
}
bool Model::set_prompt_check(int mode,std::string& err) {
    if(mode<0 || mode>2) {err="prompt_check must be 0, 1 or 2";return false;}
    if(mode && (tpe || !p->native)) {err="prompt_check requires the native engine";return false;}
    if(!tpe)p->prompt_check_mode=mode;
    return true;
}
std::string Model::prompt_read_refusal(int n,int want) const {return tpe ? "unsupported" : p->prompt_read_refusal(n,want);}
bool Model::exchange_schedule_control() const {return !tpe && p->exchange_schedule_supported();}
int Model::exchange_schedule() const {return tpe ? 0 : p->exchange_schedule;}
bool Model::set_exchange_schedule(int mode,std::string& err) {
    if(tpe) {if(mode) {err="exchange schedule requires the native engine";return false;} return true;}
    if(mode!=p->exchange_schedule && pipe_busy()) {err="cannot change the exchange schedule while pipeline work is pending";return false;}
    return p->exchange_schedule_set(mode,err);
}
bool Model::set_residency_decay(double decay,std::string& err) {
    if(!(decay>=0.5 && decay<=0.999)) {err="residency decay must be in [0.5, 0.999]";return false;}
    if(tpe) {if(decay!=0.95) {err="residency decay requires the native engine";return false;} return true;}
    p->demand_decay=decay;return true;
}
double Model::residency_decay() const {return tpe ? 0.95 : p->demand_decay;}
bool Model::set_helper_max_exchanges(double capacity,std::string& err) {
    if(!(capacity>=0.1 && capacity<=20)) {err="helper exchange capacity must be in [0.1, 20] credits per five rows";return false;}
    if(tpe) {if(capacity!=1.0) {err="helper exchange capacity requires the native engine";return false;} return true;}
    p->helper_exchange_capacity=capacity;return true;
}
double Model::helper_max_exchanges() const {return tpe ? 1.0 : p->helper_exchange_capacity;}
bool Model::set_exchange_host_wait(int forwards,std::string& err) {
    if(forwards<0 || forwards>16) {err="exchange host wait must be 0 to 16 forward ends";return false;}
    if(!tpe)p->exchange_host_wait=forwards;
    return true;
}
int Model::exchange_host_wait() const {return tpe ? 2 : p->exchange_host_wait;}
bool Model::cpu_first() const {return native_cooperative() && p->o.cpu_first;}
bool Model::set_cpu_first(bool enabled,std::string& err) {
    if(!native_cooperative() || pipe_busy()) {
        err="cpu first requires idle native cooperative execution";return false;
    }
    p->o.cpu_first=enabled;return true;
}
bool Model::primary_submit_query() const {return native_cooperative() && p->o.primary_submit_query;}
bool Model::set_primary_submit_query(bool enabled,std::string& err) {
    if(!native_cooperative() || pipe_busy()) {
        err="primary submission query requires idle native cooperative execution";return false;
    }
    p->o.primary_submit_query=enabled;return true;
}
bool Model::set_cooperative_cpu_scale(double scale,std::string& err) {
    if(!native_cooperative() || pipe_busy() || !std::isfinite(scale) || scale<0.25 || scale>4.0) {
        err="CPU price scale requires idle native cooperation and a finite value in [0.25,4]";return false;
    }
    p->o.cooperative_cpu_scale=scale;return true;
}
size_t Model::routing_prefetch_host_bytes() const {
    return tpe || p->D.empty() || !p->D[0].prediction_host ? 0 : size_t(kWin)*p->g.k_exp*8;
}
bool Model::routing_dampen() const { return !tpe && p->native && p->o.routing_dampen; }
bool Model::routing_enabled() const { return !tpe && p->native && p->o.routing; }
float Model::routing_ratio() const { return p->o.routing_ratio; }
bool Model::routing_prior() const { return !tpe && p->native && p->o.routing_prior; }
float Model::routing_prior_bonus() const { return p->o.routing_prior_bonus; }
bool Model::primary_adaptive_supported() const {return !tpe && p->o.native_cooperative && p->adaptive_supported();}
bool Model::primary_adaptive_enabled() const {return !tpe && p->primary_adaptive;}
PrimaryResidencyPolicy Model::primary_residency_policy() const {return p->adaptive_policy;}
double Model::primary_residency_service_ms() const {return p->adaptive_service_ms();}
double Model::primary_residency_h2d_ms() const {return p->adaptive_h2d_ms();}
double Model::primary_residency_d2h_ms() const {return p->adaptive_d2h_ms();}
long long Model::residency_demand_rows() const {return p->native_demand_rows;}
bool Model::primary_backing_enabled() const {return bool(p->primary_backing_bank);}
bool Model::set_primary_residency_policy(const PrimaryResidencyPolicy& policy,std::string& err) {
    if(tpe || pipe_busy()) {err="primary residency policy requires idle native serving";return false;}
    const auto valid=[](double x){return std::isfinite(x) && x>=0 && x<=1000;};
    if(policy.horizon<1 || policy.horizon>1000 || policy.max_exchanges<2 || policy.max_exchanges>32 || !valid(policy.service_saved_ms) ||
       !valid(policy.h2d_ms) || !valid(policy.d2h_ms) || !valid(policy.interference_ms) || !valid(policy.margin_ms)) {
        err="invalid primary residency policy";return false;
    }
    p->adaptive_policy=policy;return true;
}
bool Model::set_primary_adaptive(bool enabled,std::string& err) {
    if(tpe || pipe_busy()) {err="adaptive primary requires idle native serving";return false;}
    if(p->placement_failed || p->retirement_uncertain.load()) {err="native cache ownership is poisoned";return false;}
    if(enabled && !p->o.native_cooperative) {err="adaptive primary is unsupported with cooperation off";return false;}
    if(enabled && p->o.routing_prefetch) {err="adaptive primary is incompatible with routing prefetch";return false;}
    if(!p->adaptive_supported()) {err="adaptive primary requires healthy single-target native window serving";return false;}
    // Disable does not allocate; prepared exchange frames remain live and any
    // in-flight transaction continues to retire through the existing path.
    if(enabled && !p->adaptive_prepare(err))return false;
    p->primary_adaptive=enabled;
    return true;
}
bool Model::primary_arithmetic_control() const {
    return !tpe && p->native && p->native_primary_fp4_capable && (p->D.size()==1 || p->o.native_stage_prefix) && p->o.seqs==1;
}
bool Model::primary_arithmetic_available(const std::string& mode) const {
    if(!primary_arithmetic_control() || (mode!="dp4a"&&mode!="fp4"))return false;
    const auto a=mode=="fp4"?ne::Arithmetic::NativeFp4:ne::Arithmetic::Dp4a;
    const size_t need=ne::scratch_bytes(p->g.dim,p->g.ff,a);
    return need && p->native_primary_arena_bytes>=need;
}
std::string Model::primary_arithmetic() const {return !tpe&&p->native?p->native_primary_current:"unavailable";}
std::string Model::primary_lane_calibration_arithmetic() const {return !tpe&&p->native?p->lane_calibration_arithmetic:"none";}
size_t Model::primary_arithmetic_arena_bytes() const {return !tpe&&p->native?p->native_primary_arena_bytes:0;}
bool Model::set_primary_arithmetic(const std::string& mode,std::string& err) {
    if(!tpe && p->placement_failed) {err="native placement previously failed; engine cannot resume";return false;}
    if(!primary_arithmetic_control()) {err="primary arithmetic requires single-sequence native RTX 5090";return false;}
    if(mode!="dp4a"&&mode!="fp4") {err="primary arithmetic must be dp4a or fp4";return false;}
    if(!primary_arithmetic_available(mode)) {err="selected primary arena is too small for "+mode+"; restart with that arithmetic";return false;}
    if(pipe_busy()) {err="cannot change primary arithmetic while pipeline work is pending";return false;}
    if(mode==p->native_primary_current)return true;
    if(p->o.cooperative_stage_reuse && p->batch_staging_arithmetic!=mode) {
        err="arithmetic switch has no matching batch staging calibration; disable batch staging for a legacy diagnostic or reload";return false;
    }
    auto& d=p->D[(size_t)p->head_dev]; CK(cudaSetDevice(d.id));
    const auto a=mode=="fp4"?ne::Arithmetic::NativeFp4:ne::Arithmetic::Dp4a;
    if(!ne::set_arithmetic(d.s,a)) {err="primary adapter rejected arithmetic change";return false;}
    if(p->graph_on)p->graphs_clear();
    p->prep_graphs_clear();
    p->native_primary_current=mode;
    // Keep the existing complete-bank lane estimate: no hidden remeasurement
    // or dispatch-policy change inside a request-boundary A/B switch.
    return true;
}
bool Model::helper_live_supported() const {return !tpe && p->helper_live_mode_supported() && bool(p->helper);}
bool Model::helper_live_enabled() const {return !tpe && p->helper_live_enabled;}
bool Model::helper_dispatch_control() const {
    if(tpe||!p->native||!p->helper||(p->D.size()!=1 && !p->o.native_stage_prefix)||p->o.seqs!=1)return false;
    cudaDeviceProp a{},b{};
    return cudaGetDeviceProperties(&a,p->D[(size_t)p->head_dev].id)==cudaSuccess && cudaGetDeviceProperties(&b,p->o.native_helper)==cudaSuccess &&
           a.major==12&&a.minor==0&&b.major==8&&b.minor==6&&std::strstr(a.name,"RTX 5090")&&std::strstr(b.name,"RTX 3090");
}
bool Model::helper_dispatch_available(const std::string& mode) const {
    return helper_dispatch_control() && ((mode=="queue" && !p->o.native_stage_prefix) || (mode=="worker"&&p->helper->worker_compute_available()));
}
std::string Model::helper_dispatch() const {return !tpe&&p->native&&p->helper?p->helper_dispatch_current:"unavailable";}
size_t Model::helper_queue_device_bytes() const {
    return helper_queue_primary_bytes()+helper_queue_helper_bytes();
}
size_t Model::helper_queue_primary_bytes() const {return tpe||!p->helper_queue?0:size_t(p->g.n_exp)*sizeof(nf::QueueRecord)+size_t(kWin)*p->g.k_exp*sizeof(nf::QueuePick);}
size_t Model::helper_queue_helper_bytes() const {return tpe||!p->helper_queue?0:p->helper_queue->bytes().helper_device;}
size_t Model::helper_queue_pinned_bytes() const {
    return tpe||!p->helper_queue?0:p->helper_queue->bytes().pinned+size_t(p->g.n_exp)*sizeof(nf::QueueRecord);
}
bool Model::set_helper_dispatch(const std::string& mode,std::string& err) {
    if(!tpe && p->o.cooperative_stage_reuse && mode!="worker") {
        err="batch staging calibration requires worker helper dispatch; disable it for a legacy diagnostic";return false;
    }
    if(!tpe && p->o.routing_prefetch && mode!="worker") {err="routing prefetch requires worker helper dispatch";return false;}
    if(!helper_dispatch_available(mode)){err="requested helper dispatch is unavailable in this startup configuration";return false;}
    if(pipe_busy()||p->helper_queue_pending||p->placement_failed||p->retirement_uncertain.load()) {err="helper dispatch requires a healthy idle request boundary";return false;}
    // Worker-only serving needs no queue allocation. A comparison first uses
    // the checked queue route, then retains those resources across both arms.
    if(mode=="queue" && !p->helper_queue_prepare(err))return false;
    if(mode!=p->helper_dispatch_current){if(p->graph_on)p->graphs_clear();p->prep_graphs_clear();p->helper_dispatch_current=mode;}
    return true;
}
bool Model::set_helper_live(bool enabled,std::string& err) {
    if(tpe || pipe_busy()) {err="live helper requires idle native serving";return false;}
    if(p->placement_failed || p->retirement_uncertain.load()) {err="native cache ownership is poisoned";return false;}
    if(!p->helper_live_mode_supported() || !p->helper) {err="live helper requires a healthy native suffix owner";return false;}
    // Retain previously prepared buffers and leases for normal retirement.
    if(enabled && !p->helper_live_prepare(err))return false;
    p->helper_live_enabled=enabled;
    return true;
}
int Model::native_stage_prefix() const {return tpe ? 0 : p->o.native_stage_prefix;}
bool Model::native_state_poisoned() const {
    return tpe || p->placement_failed || p->retirement_uncertain.load() || p->stage_poisoned ||
        p->hf_demand_poisoned || p->native_cpu_failed;
}
bool Model::stage_begin(const int* ids,int n,int pos,StageTicket& ticket,std::string& err) {
    if(tpe || p->placement_failed || p->retirement_uncertain.load() || pipe_busy()) {err="native stage requires a healthy idle model";return false;}
    uint64_t value=0;
    if(!p->stage_begin(ids,n,pos,value,err))return false;
    ticket.value=value;return true;
}
bool Model::stage_prefix_submit(StageTicket ticket,int slot,std::string& err) {
    if(tpe) {err="native stage unavailable";return false;}
    return p->stage_prefix_submit(ticket.value,slot,err);
}
bool Model::stage_prefix_ready(StageTicket ticket,int slot,bool wait,bool& ready,std::string& err) {
    ready=false;if(tpe) {err="native stage unavailable";return false;}
    return p->stage_prefix_ready(ticket.value,slot,wait,ready,err);
}
bool Model::stage_suffix_launch(StageTicket ticket,int slot,std::string& err) {
    if(tpe) {err="native stage unavailable";return false;}
    return p->stage_suffix_launch(ticket.value,slot,err);
}
bool Model::stage_suffix_service(StageTicket ticket,int slot,bool& submitted_all,std::string& err) {
    submitted_all=false;if(tpe) {err="native stage unavailable";return false;}
    return p->stage_suffix_service(ticket.value,slot,submitted_all,err);
}
bool Model::stage_suffix_ready(StageTicket ticket,int slot,bool wait,bool& ready,std::string& err) {
    ready=false;if(tpe) {err="native stage unavailable";return false;}
    return p->stage_suffix_ready(ticket.value,slot,wait,ready,err);
}
bool Model::stage_suffix_finish(StageTicket ticket,int slot,float* logits,size_t capacity,std::string& err) {
    if(tpe) {err="native stage unavailable";return false;}
    return p->stage_suffix_finish(ticket.value,slot,logits,capacity,err);
}
bool Model::stage_suffix_execute(StageTicket ticket,int slot,float* logits,size_t capacity,std::string& err) {
    if(tpe) {err="native stage unavailable";return false;}
    return p->stage_suffix_execute(ticket.value,slot,logits,capacity,err);
}
bool Model::stage_finish(StageTicket ticket,int& completed,std::string& err) {
    completed=0;if(tpe) {err="native stage unavailable";return false;}
    return p->stage_finish(ticket.value,completed,err);
}
bool Model::forward_stage_window(const int* ids,int n,int pos0,bool overlap,const std::function<bool(int)>& after_a,int& completed_rows,float* logits,std::string& err) {
    completed_rows=0;
    if(tpe || p->placement_failed || p->retirement_uncertain.load() || pipe_busy()) {err="native stage unavailable, busy or poisoned";return false;}
    return p->stage_window(ids,n,pos0,overlap,after_a,completed_rows,logits,err);
}
bool Model::draft_active() const { return tpe ? draft_block()>0 : p->draftm && p->draft_running; }
bool Model::calibrate_service_after_prefill(std::string& err) {
    if(!tpe && p->placement_failed) {err="native placement previously failed; engine cannot resume";return false;}
    if(tpe || !p->native || p->native_calibrated_after_prefill)return true;
    if(pipe_busy()) {err="cannot calibrate native service while pipeline work is pending";return false;}
    for(auto& d:p->D) {
        if(!p->scale_maintenance_join(d,err))return false;
        std::fill(d.slot_foreground.begin(),d.slot_foreground.end(),0);
        if(!p->adaptive_publish(d,err))return false;
        p->drop_slots(d);
    }
    if(!p->helper_live_settle(err))return false;
    p->graphs_clear();
    const size_t notes=p->info.size();
    p->calibrate_lane();
    if(p->native_cpu_failed){err="native post-prefill service calibration failed";return false;}
    p->native_calibrated_after_prefill=true;
    for(size_t i=notes;i<p->info.size();++i)
        std::fprintf(stderr,"native after real prefill: %s\n",p->info[i].c_str());
    return true;
}
bool Model::set_draft_active(bool enabled,std::string& err) {
    if(!tpe && p->native && p->retirement_uncertain.load()) {err="native ownership is poisoned";return false;}
    if(!tpe && p->placement_failed) {err="native placement previously failed; engine cannot resume";return false;}
    if(tpe) { if(!enabled) { err="request draft execution switching is unavailable for tensor parallel"; return false; } return true; }
    if(enabled && !p->draftm) { err="DSpark weights are not loaded"; return false; }
    if(pipe_busy()) { err="cannot change draft execution while pipeline work is pending"; return false; }
    return r2::execution_boundary(p->native,p->retirement_uncertain,[&] {
        for(auto& d:p->D) {
            auto rc=cudaSetDevice(d.id);
            if(rc==cudaSuccess)rc=cudaStreamSynchronize(d.s);
            if(rc==cudaSuccess && d.s2)rc=cudaStreamSynchronize(d.s2);
            if(rc!=cudaSuccess) {
                if(p->native) {p->retirement_uncertain.store(true);if(p->o.native_stage_prefix)p->stage_poisoned=true;}
                err=std::string("draft control retirement: ")+cudaGetErrorString(rc);return false;
            }
        }
        p->graphs_clear(); p->draft_running=enabled;
        return true;
    });
}
int Model::chunk_max() const { return tpe ? 0 : p->o.chunk; }
void Model::set_chunk(int c) { p->chunk_run = c; }
void Model::set_prefill_gemm(int b) { gg::set_gemm(b); }
int Model::prefill_gemm() const { return gg::gemm_backend(); }
bool Model::lt_available() const { return !tpe && p->gguf && gg::gemm_lt_available(); }
std::string Model::tune_key() const { return p->tkey; }
std::string Model::tune_file() const { return p->tfile; }
bool Model::write_tune(const std::string& settings, std::string& err) const {
    std::vector<std::string> keep;
    {
        std::ifstream f(p->tfile);
        for (std::string line; std::getline(f, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const size_t tab = line.find('\t');
            if (!line.empty() && (tab == std::string::npos || line.compare(0, tab, p->tkey) != 0)) keep.push_back(line);
        }
    }
    keep.push_back(p->tkey + "\t" + settings);
    const std::string tmp = p->tfile + ".tmp";
    {
        std::ofstream f(tmp, std::ios::binary | std::ios::trunc);
        for (const auto& l : keep) f << l << "\n";
        if (!f) {
            err = tmp + ": cannot write";
            return false;
        }
    }
    if (!strata::ds4::os::replace_file(tmp, p->tfile)) {
        err = p->tfile + ": cannot replace";
        return false;
    }
    return true;
}
int Model::vocab() const { return tpe ? tpe->vocab() : p->g.vocab; }
int Model::max_ctx() const { return tpe ? tpe->max_ctx() : p->o.max_ctx; }
int Model::bos() const { return tpe ? tpe->bos() : p->g.bos; }
int Model::eos() const { return tpe ? tpe->eos() : p->g.eos; }
const Stats& Model::stats() const {
    if(tpe)return tpe->stats();
    p->st.helper_live_started=p->helper_live_started;
    p->st.helper_live_withdrawals=p->helper_live_withdrawals;
    p->st.helper_live_publications=p->helper_live_publications;
    p->st.helper_live_finished=p->helper_live_finished;
    p->st.exchange_verified=p->exchange_download_verified+p->exchange_upload_verified;
    if(p->helper) {
        const auto s=p->helper->live_schedule_stats();
        p->st.exchange_x4_pieces=(long long)s.x4_pieces;p->st.exchange_host_pieces=(long long)s.host_pieces;
        p->st.exchange_x4_ms=1e-6*double(s.x4_ns);p->st.exchange_host_ms=1e-6*double(s.host_ns);
        p->st.exchange_job_overlaps=(long long)s.job_overlaps;
        p->st.exchange_verified+=(long long)s.verified;p->st.exchange_mismatches=(long long)s.mismatches;
    }
    p->st.exchange_download_pass_pieces=p->exchange_download_pass_pieces;
    p->st.exchange_download_flush_pieces=p->exchange_download_flush_pieces;
    p->st.exchange_download_windows=p->exchange_download_windows;
    p->st.exchange_host_urgent_forwards=p->exchange_host_urgent_forwards;
    return p->st;
}
std::vector<std::string> Model::info_lines() const { return tpe ? tpe->info_lines() : p->info; }
std::vector<std::string> Model::vram_lines() const { return tpe ? tpe->vram_lines() : p->vram; }

bool Model::draft_context_available(int pos) const {
    if (tpe) return tpe->draft_context_available(pos);
    return !p->draftm || p->draftm->context_available(pos);
}

int Model::draft_block() const {
    if (tpe) return tpe->draft_block();
    return p->draftm ? p->draftm->geo().block : 0;
}

const VisionGeo* Model::vision() const {
    if (tpe) return tpe->vision();
    return p->vis ? &p->vis->geo() : nullptr;
}

bool Model::encode_image(const uint8_t* rgb, int width, int height, std::vector<float>& rows, std::string& err) {
    if(!tpe && p->stage_state.active) {err="native stage lease is active";return false;}
    if (tpe) return tpe->encode_image(rgb, width, height, rows, err);
    if (!p->vis) {
        err = "no vision tower loaded (--vision on)";
        return false;
    }
    return p->vis->encode(rgb, width, height, rows, err);
}

void Model::set_images(std::vector<Image> images) {
    if (tpe) tpe->set_images(std::move(images));
    else if(!p->stage_state.active) p->imgs = std::move(images);
}

bool Model::draft(int anchor, int pos, int n, std::vector<int>& drafts, std::vector<float>& conf, std::string& err) {
    if(!tpe && p->stage_state.active) {err="native stage lease is active";return false;}
    if(!tpe && p->placement_failed) {err="native placement previously failed; engine cannot resume";return false;}
    if (tpe) return tpe->draft(anchor, pos, n, drafts, conf, err);
    if (!p->draftm) {
        err = "no DSpark draft loaded (--dspark)";
        return false;
    }
    if(p->native && (p->retirement_uncertain.load() || p->stage_poisoned || p->hf_demand_poisoned || p->native_cpu_failed)) {err="native ownership is poisoned";return false;}
    if(p->native && (anchor<0 || anchor>=p->g.vocab || pos<0 || pos>=p->o.max_ctx || n<1 || !p->draftm->context_available(pos))) {
        err="invalid native draft input or unavailable context";return false;
    }
    return r2::execution_boundary(p->native,p->retirement_uncertain,[&] {
        if(p->native) {CK(cudaSetDevice(p->D[(size_t)p->head_dev].id));}
        else (void)cudaSetDevice(p->D[(size_t)p->head_dev].id);
        return p->draftm->propose(anchor,pos,n,drafts,conf,nullptr,err);
    });
}

bool Model::draft_sampled(int anchor, int pos, int n, const DraftSampling& ds, const std::vector<float>& u,
                          std::vector<int>& drafts, std::vector<float>& conf,
                          std::vector<std::vector<std::pair<int, float>>>& q, std::string& err) {
    if(!tpe && p->stage_state.active) {err="native stage lease is active";return false;}
    if(!tpe && p->placement_failed) {err="native placement previously failed; engine cannot resume";return false;}
    if (tpe) return tpe->draft_sampled(anchor, pos, n, ds, u, drafts, conf, q, err);
    if (!p->draftm) {
        err = "no DSpark draft loaded (--dspark)";
        return false;
    }
    k::SampleArgs sa;
    sa.temperature = ds.temperature;
    sa.top_p = ds.top_p;
    sa.min_p = ds.min_p;
    sa.top_k = ds.top_k;
    if(p->native && (p->retirement_uncertain.load() || p->stage_poisoned || p->hf_demand_poisoned || p->native_cpu_failed)) {err="native ownership is poisoned";return false;}
    if(p->native && (anchor<0 || anchor>=p->g.vocab || pos<0 || pos>=p->o.max_ctx || n<1 ||
       !p->draftm->context_available(pos) || u.size()<(size_t)std::min(n,p->draftm->geo().block))) {
        err="invalid native sampled draft input or unavailable context";return false;
    }
    return r2::execution_boundary(p->native,p->retirement_uncertain,[&] {
        if(p->native) {CK(cudaSetDevice(p->D[(size_t)p->head_dev].id));}
        else (void)cudaSetDevice(p->D[(size_t)p->head_dev].id);
        return p->draftm->propose_sampled(anchor,pos,n,sa,u,drafts,conf,q,err);
    });
}

// Read original target demand only, never the seeded/executed host-free counter.
bool Model::generation_profile_read_original(std::vector<uint32_t>& counts, std::string& err) const {
    if(!tpe && p->stage_state.active) {err="native stage lease is active";return false;}
    if(!tpe && p->placement_failed) {err="native placement previously failed; engine cannot resume";return false;}
    if (tpe || !p->native) { err = "generation profile requires native single-sequence execution"; return false; }
    counts = p->original_prof;
    std::vector<uint32_t> dc(counts.size());
    for (const auto& d : p->D) {
        const unsigned* source = d.hf_original_d;
        if (!source) continue;
        if (cudaSetDevice(d.id) != cudaSuccess || cudaMemcpy(dc.data(), source, dc.size()*sizeof(uint32_t), cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaGetLastError(); err = "generation profile: original demand read failed"; return false;
        }
        for (size_t i=0;i<counts.size();++i) counts[i] += dc[i];
    }
    return true;
}
bool Model::generation_profile_begin(std::string& err) {
    if(!tpe && p->stage_state.active) {err="native stage lease is active";return false;}
    if (tpe || !p->native) return true;
    p->generation_active = false;
    if (!generation_profile_read_original(p->generation_before, err)) return false;
    p->generation_active = true;
    return true;
}
bool Model::generation_profile_end(std::string& err) {
    if(!tpe && p->stage_state.active) {err="native stage lease is active";return false;}
    if (tpe || !p->native) return true;
    if (!p->generation_active) { err = "generation profile: no active snapshot"; return false; }
    p->prefetch_end_generation();
    p->generation_active = false;
    p->exchange_windows_open();
    std::vector<uint32_t> after;
    if (!generation_profile_read_original(after, err)) return false;
    if (!generation_profile_accumulate(p->generation_before, after, p->generation_prof)) {
        err = "generation profile: geometry mismatch or count overflow"; return false;
    }
    return p->trace_end(err);
}
void Model::generation_profile_discard() {
    if (!tpe && p->native && !p->stage_state.active) {
        p->prefetch_end_generation();
        p->generation_active = false;
        p->exchange_windows_open();
        std::string trace_error;
        p->trace_end(trace_error);
    }
}
bool Model::generation_profile_snapshot(std::vector<uint32_t>& counts, std::string& err) const {
    if (tpe || !p->native || p->generation_active) { err = "generation profile: unsupported or request active"; return false; }
    counts = p->generation_prof;
    if (counts.empty()) counts.assign(p->original_prof.size(), 0);
    return true;
}
bool Model::generation_profile_reset(std::string& err) {
    if (tpe || !p->native || p->generation_active) { err = "generation profile: unsupported or request active"; return false; }
    p->generation_prof.assign(p->original_prof.size(), 0);
    return true;
}

bool Model::save_profile(const std::string& path, std::string& err) const {
    if(!tpe && p->stage_state.active) {err="native stage lease is active";return false;}
    if(!tpe && p->native && native_state_poisoned()) {err="native ownership is poisoned";return false;}
    if (tpe) return tpe->save_profile(path, err);
    // (the host-free layers' routings were counted on their GPUs: added in)
    std::vector<uint32_t> all = p->native ? p->original_prof : p->prof, dc(all.size());
    for (const auto& d : p->D) {
        const unsigned* counts=p->native ? d.hf_original_d : d.prof_d;
        if (counts == nullptr) continue;
        cudaSetDevice(d.id);
        if (cudaMemcpy(dc.data(), counts, dc.size() * 4, cudaMemcpyDeviceToHost) != cudaSuccess) {
            cudaGetLastError();
            err = "the expert profile's device counts not read";
            return false;
        }
        for (size_t i = 0; i < all.size(); ++i) all[i] += dc[i];
    }
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (!f || std::fwrite(all.data(), 4, all.size(), f) != all.size()) {
        if (f) std::fclose(f);
        err = path + ": cannot write";
        return false;
    }
    std::fclose(f);
    return true;
}

bool Model::native_placement(const std::vector<uint32_t>& profile, bool reseed_primary,
        std::string& report_json, std::string& err) {
    if(tpe || pipe_busy()>0) {err="placement requires a joined native non-TP engine";return false;}
    return p->placement(profile,reseed_primary,report_json,err);
}

bool Model::native_primary_backing(bool enable,std::string& report_json,std::string& err) {
    if(tpe || pipe_busy()>0 || p->generation_active || p->stage_state.active || native_state_poisoned()) {
        err="primary backing requires a healthy idle native engine";return false;
    }
    return p->configure_primary_backing(enable,report_json,err);
}

bool Model::native_placement_helper_first(const std::vector<uint32_t>& profile,
        std::string& report_json, std::string& err) {
    if(tpe || pipe_busy()>0 || p->generation_active || p->stage_state.active || native_state_poisoned()) {
        err="helper-first placement requires a healthy idle native engine";return false;
    }
    return p->placement_helper_first(profile,report_json,err);
}

bool Model::native_learned_checkpoint(const std::string& action, std::string& report_json, std::string& err) {
    if(tpe || pipe_busy()>0 || p->generation_active || p->stage_state.active || native_state_poisoned()) {
        err="learned checkpoint requires a healthy idle native engine";return false;
    }
    if(!p->learned_state(action,report_json,err))return false;
    // Setup-only complete ownership/policy state. Fixed-work predictions can
    // bind to the actual engine without enabling expensive per-layer capture.
    report_json.pop_back();
    report_json+=",\"replay_snapshot\":"+p->replay_snapshot_record()+"}";
    return true;
}

bool Model::native_trace_begin(std::string& err) {
    return tpe || p->trace_begin(err);
}

void Model::light_generation_begin(long long position) {
    if(tpe || !capture::light::active())return;
    p->light_map(0);
    capture::light::generation_mark(capture::light::kGenerationBegin,capture::light::kScopeRequest,true,
        p->native_demand_rows,position,0,0,0,0,0,0);
}
void Model::light_generation_end(bool ok,long long position,long long emitted,long long drafts_offered,
        long long drafts_accepted,long long target_rounds,long long spec_rounds,double elapsed_ms) {
    if(tpe || !capture::light::active())return;
    capture::light::generation_mark(capture::light::kGenerationEnd,capture::light::kScopeRequest,ok,
        p->native_demand_rows,position,emitted,drafts_offered,drafts_accepted,target_rounds,spec_rounds,elapsed_ms);
    p->light_map(1);
}

bool Model::native_replay_workload(const std::string& path,std::string& report,std::string& err) {
    if(tpe || pipe_busy()>0 || p->generation_active || p->stage_state.active || native_state_poisoned() ||
       !p->adaptive_supported() || !p->batch_staging_costs.valid || p->helper_dispatch_current!="worker") {
        err="fixed workload requires healthy idle calibrated cooperative engine";return false;
    }
    std::vector<int> layers;
    for(size_t i=0;i<p->L.size();++i)if(p->L[i].moe) {
        if(p->L[i].hf) {err="fixed workload does not support host-free layers";return false;}
        layers.push_back(int(i));
    }
    ReplayWorkload workload;
    if(!read_replay_workload(path,layers,p->g.n_exp,p->o.max_ctx,workload,err))return false;
    const auto policy=p->adaptive_policy;const double scale=p->o.cooperative_cpu_scale;
    struct Restore {
        Impl* p;PrimaryResidencyPolicy policy;double scale;
        ~Restore(){if(p->helper)p->helper->set_occupancy(false);
            p->replay_fixed_batch=nullptr;p->replay_stage_limit=-1;p->replay_query_primary=false;p->adaptive_policy=policy;
            p->primary_admissions_held=false;p->helper_admissions_held=false;
            p->o.cooperative_cpu_scale=scale;p->generation_active=false;}
    } restore{p,policy,scale};
    p->adaptive_policy.max_exchanges=workload.budget;p->adaptive_policy.margin_ms=workload.margin_ms;
    p->o.cooperative_cpu_scale=workload.cpu_scale;p->replay_stage_limit=workload.stage_limit;
    p->replay_query_primary=workload.diagnostic_primary_query;
    p->primary_admissions_held=workload.hold_primary_admissions;p->helper_admissions_held=workload.hold_helper_admissions;
    const uint64_t occupancy_before=p->helper?p->helper->occupancy_launches():0;
    if(workload.helper_occupancy && p->helper)p->helper->set_occupancy(true);
    if(!p->trace_begin(err))return false;
    std::vector<float> logits(size_t(16)*p->g.vocab);
    std::array<int,16> tokens;tokens.fill(1);
    const auto before=p->st;
    const uint64_t helper_started_before=p->helper_live_started;
    double total_ms=0,generation_ms=0;size_t evaluated_rows=0,generation_batches=0;
    capture::event("fixed_workload_begin","\"batches\":"+std::to_string(workload.batches.size())+
        ",\"diagnostic_primary_query\":"+(workload.diagnostic_primary_query?"true":"false")+
        ",\"scope\":\"fixed original/executed expert rows, synthetic token inputs, target execution only\"");
    bool generation_started=false;
    for(const auto& b:workload.batches) {
        p->generation_active=b.generation;p->replay_fixed_batch=&b;
        if(b.generation && !generation_started) {
            capture::event("generation_begin");generation_started=true;
            p->light_map(0);
            capture::light::generation_mark(capture::light::kGenerationBegin,capture::light::kScopeFixture,true,
                p->native_demand_rows,b.position,0,0,0,0,0,0);
        }
        capture::event("target_forward_begin","\"phase\":"+capture::quote(b.generation?"generation":"prefill")+
            ",\"position\":"+std::to_string(b.position)+",\"rows\":"+std::to_string(b.rows));
        const double start=now();
        const bool ok=p->chunk_(tokens.data(),b.rows,b.position,b.rows,logits.data(),err);
        const double ms=1000*(now()-start);total_ms+=ms;if(b.generation){generation_ms+=ms;evaluated_rows+=b.rows;++generation_batches;}
        capture::event("target_forward_end","\"ok\":"+std::to_string(int(ok))+
            ",\"position\":"+std::to_string(b.position)+",\"rows\":"+std::to_string(b.rows));
        if(!ok) {p->trace_end(err);return false;}
    }
    capture::event("generation_end");
    capture::light::generation_mark(capture::light::kGenerationEnd,capture::light::kScopeFixture,true,
        p->native_demand_rows,0,0,0,0,(long long)generation_batches,0,generation_ms);
    p->light_map(1);
    if(!p->trace_end(err))return false;
    std::ostringstream out;out.precision(17);
    out<<"{\"scope\":\"fixed evaluated target work; synthetic input; drafting excluded; no generation or quality claim\""
       <<",\"batches\":"<<workload.batches.size()<<",\"generation_batches\":"<<generation_batches<<",\"generation_rows\":"<<evaluated_rows
       <<",\"target_ms\":"<<total_ms<<",\"generation_target_ms\":"<<generation_ms
       <<",\"diagnostic_primary_query\":"<<(workload.diagnostic_primary_query?"true":"false")
       <<",\"helper_occupancy\":"<<(workload.helper_occupancy?"true":"false")
       <<",\"helper_occupancy_launches\":"<<(p->helper?p->helper->occupancy_launches()-occupancy_before:0)
       <<",\"primary_submit_queries\":"<<p->st.primary_submit_queries-before.primary_submit_queries
       <<",\"primary_submit_not_ready\":"<<p->st.primary_submit_not_ready-before.primary_submit_not_ready
       <<",\"windows\":"<<p->st.coop_windows-before.coop_windows
       <<",\"hold_primary_admissions\":"<<(workload.hold_primary_admissions?"true":"false")
       <<",\"hold_helper_admissions\":"<<(workload.hold_helper_admissions?"true":"false")
       <<",\"helper_started\":"<<p->helper_live_started-helper_started_before
       <<",\"helper_finished_total\":"<<p->helper_live_finished
       <<",\"primary_published\":"<<p->st.adaptive_published-before.adaptive_published
       <<",\"primary_issued\":"<<p->st.adaptive_issued-before.adaptive_issued
       <<",\"h2d_bytes\":"<<p->st.adaptive_h2d_bytes-before.adaptive_h2d_bytes
       <<",\"d2h_bytes\":"<<p->st.adaptive_d2h_bytes-before.adaptive_d2h_bytes
       <<",\"demand_uploads\":"<<p->st.staged-before.staged
       <<",\"cpu_ms\":"<<p->st.coop_cpu_run_ms-before.coop_cpu_run_ms
       <<",\"cpu_scale\":"<<workload.cpu_scale<<",\"budget\":"<<workload.budget
       <<",\"stage_limit\":"<<workload.stage_limit<<",\"backing\":"<<(p->primary_backing_bank?"true":"false")<<'}';
    report=out.str();return true;
}

bool Model::native_replay_services(std::string& report,std::string& err) {
    if(tpe || pipe_busy()>0 || p->generation_active || p->stage_state.active || native_state_poisoned()) {
        err="replay services require healthy idle native model";return false;
    }
    if(!p->helper || !p->helper->replay_probe_timing(true,err))return false;
    struct Restore {NativeHelper* helper;~Restore(){std::string ignored;helper->replay_probe_timing(false,ignored);}} restore{p->helper.get()};
    return p->measure_replay_services(report,err);
}

bool Model::forward(const int* ids, int n, int pos0, int want, float* logits_out, std::string& err) {
    if(!tpe && p->stage_state.active) {err="native stage lease is active";return false;}
    if (tpe) return tpe->forward(ids, n, pos0, want, logits_out, err);
    if(p->placement_failed) {err="native placement previously failed; engine cannot resume";return false;}
    if(p->native && p->retirement_uncertain.load()) {err="native ownership is poisoned";return false;}
    if(p->native && (n<0 || pos0<0 || pos0>p->o.max_ctx || n>p->o.max_ctx-pos0 || want<0 || (n && !ids))) {
        err="invalid native forward input";return false;
    }
    if(p->native && n && std::any_of(ids,ids+n,[&](int id){return id<0 || id>=p->g.vocab;})) {
        err="native forward token outside vocabulary";return false;
    }
    p->last_done = 0;
    p->nuc_rows = 0;   // (set again by a fast-head chunk that kept its rows on the device)
    want = std::min(want, n);
    // prompt_layerwise: the whole call read one layer at a time (no memory for it: today's chunks below)
    if (p->native && p->prompt_read_wanted(n, logits_out ? want : 0)) {
        bool fell_back = false;
        const bool ok = r2::execution_boundary(p->native, p->retirement_uncertain, [&] {
            return p->prompt_read(ids, n, pos0, logits_out ? want : 0, logits_out, fell_back, err);
        });
        if (!ok) return false;
        if (!fell_back) {
            p->last_done = n;
            return true;
        }
    }
    const int C = p->run_chunk();
    for (int a = 0; a < n; a += C) {
        const int T = std::min(C, n - a);
        // the logits this chunk owns: the call's last `want` positions that fall in it
        const int first = n - want;
        const int w0 = std::max(first, a), w1 = a + T;
        const int w = logits_out && w1 > w0 ? w1 - w0 : 0;
        if (w > 0 && w0 != a + T - w) {
            err = "internal: logits rows";
            return false;
        }
        if (!p->chunk(ids + a, T, pos0 + a, w, w > 0 ? logits_out + (size_t) (w0 - first) * p->g.vocab : nullptr,
                      err))
            return false;
    }
    p->last_done = n;
    return true;
}

void Model::set_progress(std::function<bool(int)> f) {
    if (tpe) tpe->set_progress(std::move(f));
}
int Model::done() const { return tpe ? tpe->done() : p->last_done; }
bool Model::pipelined() const { return tpe != nullptr; }
bool Model::forward_ppl(const int* ids, int n, std::string& err) {
    if (!tpe) {
        err = "forward_ppl: the tensor-parallel path's (--tp); the layer-split path's --ppl takes its logits";
        return false;
    }
    tpe->set_ppl(true);
    const bool ok = tpe->forward(ids, n, 0, 0, nullptr, err);
    tpe->set_ppl(false);
    return ok;
}
const std::vector<double>& Model::ppl_nll() const {
    static const std::vector<double> none;
    return tpe ? tpe->ppl_nll() : none;
}

int Model::seqs() const { return tpe || p->L.empty() ? 1 : (int) std::max<size_t>(1, p->L[0].sk.size()); }
void Model::bind(int s) {
    if (!tpe && !p->stage_state.active) p->bind(s);
}

bool Model::forward_rows(const int* ids, const int* seq, const int* pos, int R, float* logits_out, std::string& err) {
    if(p->placement_failed) {err="native placement previously failed; engine cannot resume";return false;}
    if (R <= 0) return true;
    if (pipe_busy() > 0) {
        err = "forward_rows: groups in the pipeline (pipe_take them first)";
        return false;
    }
    for (int r = 0; r < R; ++r)
        if (seq[r] < 0 || seq[r] >= seqs() || std::find(seq, seq + r, seq[r]) != seq + r) {
            err = "forward_rows: sequence " + std::to_string(seq[r]) + " (" + std::to_string(seqs()) +
                  " of them, each in one row at most)";
            return false;
        }
    // (one at a time where the batched step does not apply: one row, the prompt path's tokens - --no-fast-decode)
    if (tpe || R == 1 || R > kWin || !p->o.fast_decode) {
        for (int r = 0; r < R; ++r) {
            bind(seq[r]);
            if (!forward(ids + r, 1, pos[r], 1, logits_out + (size_t) r * vocab(), err)) return false;
        }
        return true;
    }
    p->rseq.assign(seq, seq + R);
    p->rpos.assign(pos, pos + R);
    p->last_done = 0;
    const bool ok = p->chunk(ids, R, 0, R, logits_out, err);
    p->rseq.clear();
    p->rpos.clear();
    if (ok) p->last_done = R;
    return ok;
}

int Model::pipe_stages() const {
    if (tpe || seqs() < 2 || !p->o.fast_decode || p->stage_dev.size() < 2 || p->stage_dev.back() != p->head_dev)
        return 1;
    return (int) p->stage_dev.size();
}
int Model::pipe_busy() const { return tpe ? 0 : p->pipe_busy(); }
bool Model::pipe_tick(const int* ids, const int* seq, const int* pos, int R, int tag, std::string& err) {
    if (pipe_stages() < 2) {
        err = "pipe_tick: no pipeline here (one GPU, or one sequence)";
        return false;
    }
    return p->pipe_tick(ids, seq, pos, R, tag, err);
}
bool Model::pipe_take(int& tag, std::vector<float>& logits) {
    if (tpe || p->pout.empty()) return false;
    tag = p->pout.front().first;
    logits = std::move(p->pout.front().second);
    p->pout.pop_front();
    return true;
}
void Model::pipe_drop() {
    if (!tpe && !p->stage_state.active) p->pipe_drop();
}

// (the layer-split path: the logits on the host, the candidates from them)
bool Model::nucleus_supported() const { return !tpe && p->o.fast_decode && !p->D.empty(); }
void Model::set_nucleus(bool on) {
    if (!tpe) p->nuc_want = on && nucleus_supported();
}
int Model::nucleus_rows() const { return tpe ? 0 : p->nuc_rows; }
bool Model::nucleus_row(int r, std::vector<std::pair<int, float>>& top, float& maximum, double& total, bool& finite) const {
    if (tpe || r < 0 || r >= p->nuc_rows) return false;
    const auto& hd = p->D[(size_t) p->head_dev];
    const k::CandRow& c = hd.nuc_c[r];
    const k::RowStats& s = hd.nuc_s[r];
    if (c.n < 1 || c.n > k::kCandCap) return false;   // (-1: ties past the candidates' room)
    top.resize((size_t) c.n);
    for (int i = 0; i < c.n; ++i) top[(size_t) i] = {c.id[i], c.v[i]};
    maximum = s.max;
    total = s.total;
    finite = s.finite != 0;
    return true;
}
bool Model::fetch_logits_row(int r, float* out, std::string& err) {
    if (tpe || r < 0 || r >= p->nuc_rows) {
        err = "fetch_logits_row: no such row on the device";
        return false;
    }
    auto& hd = p->D[(size_t) p->head_dev];
    if (cudaSetDevice(hd.id) != cudaSuccess ||
        cudaMemcpy(out, hd.lg + (size_t) r * p->g.vocab, (size_t) p->g.vocab * sizeof(float), cudaMemcpyDeviceToHost) != cudaSuccess) {
        cudaGetLastError();
        err = "fetch_logits_row: copy failed";
        return false;
    }
    return true;
}

bool Model::forward_cands(const int* ids, int n, int pos0, int want, Cands& c, std::string& err) {
    if (tpe) return tpe->forward_cands(ids, n, pos0, want, c, err);
    want = std::max(0, std::min(want, n));
    std::vector<float> lg((size_t) want * vocab());
    if (!forward(ids, n, pos0, want, lg.data(), err)) return false;
    c.top.assign((size_t) want, {});
    c.ex.assign((size_t) want, {});
    for (int r = 0; r < want; ++r)
        host_cands(lg.data() + (size_t) r * vocab(), vocab(), c.kc, c.extra, c.top[(size_t) r], c.ex[(size_t) r]);
    return true;
}

}  // namespace mm3
