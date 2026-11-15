// The helper hand-off: the 5090's request kernel and each helper card's step graph (mm3_helper_handoff.hpp).
#include "mm3_helper_handoff.hpp"
#include "mm3_helper_kernels.cuh"
#include "mm3_helper_q.cuh"
#include "mm3_helper_sm89.hpp"
#include "ops/minimax_nvfp4_ffn/fast.h"
#include "ops/minimax_nvfp4_ffn/fast_math.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <thread>

namespace mm3::handoff {
namespace {
using infernix::ops::minimax_ffn::FastJob;
using infernix::ops::minimax_ffn::float_bf16;
namespace tc = mm3::helper_tc;
namespace hq = mm3::helper_q;
constexpr int kChunks = kHidden * 2 / 16;   // 16-byte pieces of one BF16 row (768)
constexpr unsigned long long kTakeBoundNs = 1000000000ull;   // a take waits at most 1 s for its release
constexpr unsigned long long kSpinBoundNs = 20000000ull;     // the clock holder between steps: 20 ms, as the spinner

__device__ __forceinline__ unsigned long long now_ns() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}
template <class T> __device__ __forceinline__ T vload(const T* p) { return *reinterpret_cast<const volatile T*>(p); }
template <class T> __device__ __forceinline__ void vstore(T* p, T v) { *reinterpret_cast<volatile T*>(p) = v; }

// ---------------------------------------------------------------------------------------------------------------- 5090
// Two kernels on the router branch. plan_kernel (one block): thread 0 plans both cards' jobs from the picks (plan_jobs:
// the host's order) and the jobs are written to each card's slot. rows_kernel rounds to BF16 and quantizes each used
// 16-value block once, then writes each card's selected representation directly into its owned slot. It also checks
// non-finite values; its last block writes each card's header - a card using a non-finite row is
// refused, as NativeHelper::submit refuses it - and then each card's word.
__device__ int g_row_bad[kMaxRows];        // (rows_kernel; cleared by its last block)
__device__ int g_plan[2][4];               // per card: jobs, columns, kind, status (plan_kernel)
__device__ unsigned g_used[2];             // per card: the rows its jobs use (plan_kernel)
__device__ unsigned g_rows_done;           // rows_kernel's blocks finished (its last block resets it)

__global__ void __launch_bounds__(kThreads) plan_kernel(RequestArgs a) {
    __shared__ alignas(16) unsigned char tab_raw[2 * kExperts * sizeof(Entry)];
    __shared__ alignas(16) unsigned char raw[3 * kMaxJobs * sizeof(Job)];
    __shared__ int count[2];
    Entry (*tab)[kExperts] = reinterpret_cast<Entry (*)[kExperts]>(tab_raw);
    Job* scratch = reinterpret_cast<Job*>(raw);
    Job* jobs[2] = {scratch + kMaxJobs, scratch + 2 * kMaxJobs};
    const int tid = threadIdx.x;
    for (int c = 0; c < 2; ++c)
        if (tid < kExperts) {   // (the host rewrites the tables between passes: read them past every cache)
            Entry e{};
            if (a.table[c]) {
                const unsigned long long* src = reinterpret_cast<const unsigned long long*>(a.table[c] + tid);
                unsigned long long w[3] = {__ldcv(src), __ldcv(src + 1), __ldcv(src + 2)};
                memcpy(&e, w, sizeof e);
            }
            tab[c][tid] = e;
        }
    __syncthreads();
    if (tid == 0) {
        Snapshot s;
        for (int q = 0; q < 4; ++q) { s.helper[q] = vload(&a.snapshot->helper[q]); s.third[q] = vload(&a.snapshot->third[q]); }
        for (int c = 0; c < 2; ++c) {
            int n = 0, cols = 0, multi = 0, status = kServed;
            unsigned used = 0;
            if (a.request[c]) {
                const auto take = [&](int, int, int e) {
                    if (e < 0 || e >= kExperts) return false;
                    return c == 0 ? bit(s.helper, e) : (!bit(s.helper, e) && bit(s.third, e));
                };
                n = plan_jobs(a.top, a.w, a.T, a.k, a.scale, c == 1, take, scratch, jobs[c]);
                if (n < 0) { status = kBadRequest; n = 0; }
                for (int j = 0; j < n; ++j) {
                    Job& job = jobs[c][j];
                    const Entry& en = tab[c][job.expert];
                    if (!en.ready) status = kBadRecord;
                    job.record = en.record;
                    for (int q = 0; q < 3; ++q) job.multiplier[q] = en.multiplier[q];
                    cols += job.columns;
                    multi |= job.columns > 1;
                    for (int q = 0; q < job.columns; ++q) used |= 1u << job.token[q];
                }
                if (status != kServed) { n = 0; cols = 0; used = 0; }
            }
            count[c] = n;
            g_plan[c][0] = n; g_plan[c][1] = cols; g_plan[c][2] = multi; g_plan[c][3] = status;
            g_used[c] = used;
        }
    }
    __syncthreads();
    for (int c = 0; c < 2; ++c)
        if (a.request[c])
            for (int j = tid; j < count[c]; j += blockDim.x) a.request[c]->jobs[j] = jobs[c][j];
    __threadfence_system();
}
__global__ void __launch_bounds__(kThreads) rows_kernel(RequestArgs a) {
    const int per_row = kHidden / 16;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned used[2] = {vload(&g_used[0]), vload(&g_used[1])};
    if (p < a.T * per_row) {
        const int t = p / per_row, b = p % per_row;
        if (((used[0] | used[1]) >> t) & 1u) {
            uint16_t v[16];
            bool finite = true, quantized = false;
#pragma unroll
            for (int k = 0; k < 16; ++k) {
                v[k] = hq::detail::canon::f32_to_bf16_rn(a.xn[size_t(t) * kHidden + b * 16 + k]);
                finite = finite && (v[k] & 0x7f80u) != 0x7f80u;
            }
            hq::detail::canon::A4Codes q{};
            if (!finite) atomicExch(&g_row_bad[t], 1);
            for (int c = 0; c < 2; ++c) if (a.request[c] && ((used[c] >> t) & 1u)) {
                auto* row = reinterpret_cast<uint8_t*>(a.request[c]->x) + size_t(t) * row_bytes(a.format[c]);
                if (a.format[c] == 0) {
                    auto* out = reinterpret_cast<uint16_t*>(row) + b * 16;
#pragma unroll
                    for (int k = 0; k < 16; ++k) out[k] = v[k];
                } else {
                    if (!quantized) { q = hq::detail::checked_codes(v); quantized = true; }
                    hq::detail::store_codes(q, b, a.format[c] == 1 ? row : nullptr,
                        a.format[c] == 2 ? reinterpret_cast<uint16_t*>(row) : nullptr,
                        a.format[c] == 3 && !a.fp8[c] ? row : nullptr, a.format[c] == 3 && a.fp8[c] ? row : nullptr);
                }
            }
        }
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x != 0 || atomicAdd(&g_rows_done, 1u) != gridDim.x - 1) return;
    // the last block: every row is written and checked
    g_rows_done = 0;
    unsigned bad = 0;
    for (int t = 0; t < a.T; ++t) { bad |= unsigned(vload(&g_row_bad[t])) << t; g_row_bad[t] = 0; }
    for (int c = 0; c < 2; ++c) {
        Request* r = a.request[c];
        if (!r) continue;
        int status = vload(&g_plan[c][3]);
        if (status == kServed && (used[c] & bad)) status = kNonfiniteRow;
        const bool ok = status == kServed;
        r->count = ok ? vload(&g_plan[c][0]) : 0;
        r->columns = ok ? vload(&g_plan[c][1]) : 0;
        r->rows = a.T;
        r->kind = vload(&g_plan[c][2]);
        r->status = status;
        r->layer = a.layer;
        r->format = a.format[c]; r->fp8 = int(a.fp8[c]);
    }
    __threadfence_system();
    for (int c = 0; c < 2; ++c)
        if (a.request[c]) vstore(&a.request[c]->word, vload(&a.snapshot->word[c]));
}

// ------------------------------------------------------------------------------------------------------- helper card
struct alignas(16) StateBody {
    unsigned long long step, word, go, t_seen;
    int count, columns, rows, kind, status, dead;
    int kernel, format, sm89;
    unsigned arrived, done_blocks, used;
    int col_token[kMaxJobs];   // each column's row (merge order)
    int col_fan[kMaxJobs];     // the column belongs to a job of more than one column: copied into d_x
};
}  // namespace

struct Lane::State : StateBody {};

namespace {
__global__ void begin_kernel(StateBody* st) {
    st->step += 1;
    st->dead = 0;
}

// Block 0's thread 0 waits for the pass's word (or the step's abort, or the bound); block 0 checks and describes the jobs;
// the other blocks wait for it, then every block copies the pass's rows to the card and, for jobs of more than one column,
// each column's row into place (a one-column job reads its row where it landed). kTakeBlocks blocks, all resident.
struct TakeArgs {
    const Request* req;
    const unsigned long long* control;
    StateBody* st;
    uint16_t *rows, *x, *y;
    FastJob* jobs;
    unsigned long long pool, pool_end, stride;
    int m;
    Lane::Kernels kernels;
};
__global__ void __launch_bounds__(kThreads) take_kernel(TakeArgs a) {
    StateBody* st = a.st;
    const int tid = threadIdx.x;
    const bool nvfp4 = kernel_family(a.kernels.one) >= 0;
    const size_t row_size = row_bytes(a.kernels.format);
    const int chunks = int(row_size / 16);
    const unsigned long long step = vload(&st->step);
    const unsigned long long expected = (step << 8) | unsigned(a.m & 255);
    if (blockIdx.x == 0) {
        __shared__ int decision, count, rows, flaw;
        __shared__ unsigned used_rows;
        __shared__ int cols[kMaxJobs], off[kMaxJobs + 1];
        if (tid == 0) {
            const unsigned long long t0 = now_ns();
            int d = kServed;
            for (;;) {
                if (vload(&a.req->word) == expected) break;
                if (vload(&st->dead)) { d = kExpired; break; }
                if (vload(&a.control[0]) >= step) { d = kAborted; break; }
                if (now_ns() - t0 > kTakeBoundNs) { d = kExpired; vstore(&st->dead, 1); break; }
                __nanosleep(100);
            }
            __threadfence_system();
            st->t_seen = now_ns();
            decision = d;
            count = 0; rows = 1; flaw = 0; used_rows = 0;
            if (d == kServed) {
                const int s = vload(&a.req->status);
                if (s != kServed) decision = s;
                count = vload(&a.req->count);
                rows = vload(&a.req->rows);
                if (count < 0 || count > kMaxJobs || rows < 1 || rows > kMaxRows) { decision = kBadRequest; count = 0; }
                if (vload(&a.req->format) != a.kernels.format ||
                    (a.kernels.format == 3 && vload(&a.req->fp8) != int(kernel_family(a.kernels.one) == 2))) decision = kBadRequest;
                if (decision != kServed) count = 0;
            }
        }
        __syncthreads();
        // each job checked: its width, its rows, its record inside the card's pool
        if (tid < count) {
            const Job* j = &a.req->jobs[tid];
            const int c = vload(&j->columns);
            const unsigned long long rec = vload(&j->record);
            bool ok = c >= 1 && c <= kMaxColumns && rec >= a.pool && rec + a.stride <= a.pool_end && (rec - a.pool) % a.stride == 0;
            for (int q = 0; ok && q < c; ++q) { const int t = vload(&j->token[q]); ok = t >= 0 && t < rows; }
            if (!ok) atomicExch(&flaw, 1);
            cols[tid] = ok ? c : 0;
        }
        __syncthreads();
        if (tid == 0) {
            off[0] = 0;
            for (int j = 0; j < count; ++j) off[j + 1] = off[j] + cols[j];
            if (flaw || off[count] > kMaxJobs) { decision = kBadRecord; count = 0; off[0] = 0; }
        }
        __syncthreads();
        if (tid < count) {
            const Job* j = &a.req->jobs[tid];
            FastJob f;
            f.record = reinterpret_cast<const uint8_t*>(vload(&j->record));
            for (int q = 0; q < 3; ++q) f.multiplier[q] = vload(&j->multiplier[q]);
            f.columns = cols[tid];
            const int first = vload(&j->token[0]);
            f.x = !nvfp4 && f.columns == 1 ? a.rows + size_t(first) * kHidden : a.x + size_t(off[tid]) * kHidden;
            f.y = a.y + size_t(off[tid]) * kHidden;
            a.jobs[tid] = f;
            unsigned rows_used = 0;
            for (int q = 0; q < f.columns; ++q) {
                const int t = vload(&j->token[q]);
                st->col_token[off[tid] + q] = t;
                st->col_fan[off[tid] + q] = nvfp4 || f.columns > 1;
                rows_used |= 1u << t;
            }
            atomicOr(&used_rows, rows_used);
        }
        __syncthreads();
        if (tid == 0) {
            int multi = 0;
            for (int j = 0; j < count; ++j) multi |= cols[j] > 1;
            st->count = count;
            st->columns = off[count];
            st->rows = rows;
            st->used = used_rows;
            st->kind = multi;
            st->kernel = multi ? a.kernels.multi : a.kernels.one;
            st->format = a.kernels.format;
            st->sm89 = int(a.kernels.sm89);
            st->status = decision;
            st->word = expected;
            st->arrived = 0;
            __threadfence();
            vstore(&st->go, expected);
        }
        __syncthreads();
    } else {
        if (tid == 0) while (vload(&st->go) != expected) __nanosleep(64);
        __syncthreads();
        __threadfence();
    }
    const int count = vload(&st->count);
    if (vload(&st->status) != kServed || count == 0) return;
    const int rows = vload(&st->rows), columns = vload(&st->columns), multi = vload(&st->kind);
    const unsigned used = vload(&st->used);
    // the rows the jobs use, to the card: 16-byte pieces, read uncached (they arrived from another device)
    const int threads = gridDim.x * blockDim.x, me = blockIdx.x * blockDim.x + tid;
    for (int p = me; p < rows * chunks; p += threads) {
        const int t = p / chunks, q = p % chunks;
        if (!((used >> t) & 1u)) continue;
        reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(a.rows) + size_t(t) * row_size)[q] =
            __ldcv(reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(a.req->x) + size_t(t) * row_size) + q);
    }
    if (!multi && !nvfp4) return;
    // every block's rows landed before any block fans them out
    __threadfence();
    __syncthreads();
    if (tid == 0) {
        atomicAdd(&st->arrived, 1u);
        while (vload(&st->arrived) < gridDim.x) __nanosleep(32);
    }
    __syncthreads();
    __threadfence();
    for (int p = me; p < columns * chunks; p += threads) {
        const int col = p / chunks, q = p % chunks;
        if (!vload(&st->col_fan[col])) continue;
        const int t = vload(&st->col_token[col]);
        reinterpret_cast<uint4*>(reinterpret_cast<uint8_t*>(a.x) + size_t(col) * row_size)[q] =
            reinterpret_cast<const uint4*>(reinterpret_cast<const uint8_t*>(a.rows) + size_t(t) * row_size)[q];
    }
}

// Prepare the same represented input consumed by the ordinary family's FFN.
// The graph has a bounded 96-block conversion node, with unused columns skipped.
// Only the compute nodes use persistent traversal; no 64-job padded FFN grid.
__global__ void prepare_q_kernel(const uint8_t* x, uint8_t* out, const StateBody* st, int format, int family, int wanted) {
    if (vload(&st->kind) != wanted || vload(&st->status) != kServed) return;
    const int t = int(blockIdx.x * blockDim.x + threadIdx.x), blocks = kHidden / 16;
    if (t >= vload(&st->columns) * blocks) return;
    const int col = t / blocks, b = t % blocks;
    const uint8_t* row = x + size_t(col) * row_bytes(format);
    hq::detail::canon::A4Codes q{};
    if (format == 0) q = hq::detail::checked_codes(reinterpret_cast<const uint16_t*>(row) + b * 16);
    else {
        q.scale_word = row[kHidden / 2 + b];
#pragma unroll
        for (int k = 0; k < 16; ++k) q.code[k] = (row[b * 8 + k / 2] >> (4 * (k & 1))) & 15;
    }
    const size_t stride = family == 3 ? hq::kPackedRow : family == 4 ? size_t(kHidden) * 2 : hq::kInStride;
    uint8_t* dst = out + size_t(col) * stride;
    hq::detail::store_codes(q, b, family == 3 ? dst : nullptr, family == 4 ? reinterpret_cast<uint16_t*>(dst) : nullptr,
        family < 2 ? dst : nullptr, family == 2 ? dst : nullptr);
}
bool needs_preparation(int family, int format) {
    return family == 3 ? format != 1 : family == 4 ? format != 2 : format != 3;
}

// The card's tensor-core expert kernel (mm3_helper_kernels.cuh, project2) as a fixed grid that walks the pass's work:
// item = job * groups + group, each computed by one block exactly as project2's block (blockIdx.x = group, blockIdx.y =
// job) computes it - the same loads, products, sums and their order - so every column's bits are project2's at the same
// warps S (P only batches the loads; P = 1 is project1's arithmetic). Runs only for its kind of pass (0: every job one
// column, 1: some job wider), as today's helper picks kernel_one or kernel_multi.
template <int S, bool Down, int P>
__global__ void __launch_bounds__(32 * S) ffn_kernel(const FastJob* jobs, const StateBody* st, int kind, uint8_t* workspace) {
    using namespace tc::detail;
    if (vload(&st->status) != kServed || vload(&st->kind) != kind) return;
    const int count = vload(&st->count);
    if (count == 0) return;   // (an empty pass: the card has no job in this layer)
    constexpr int K = Down ? tc::kI : tc::kH, blocks = K / 16, per = blocks / S;
    constexpr int groups = Down ? tc::kH / 16 : 2 * tc::kI / 16;
    static_assert(blocks % S == 0 && per % P == 0, "a warp's share of units must split into batches of P");
    __shared__ float scale[kScaleTable];
    fill_scale_table(scale, 32 * S);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, c = lane & 3;
    const int at = (c >> 1) * 32 + g * 4, shift = 16 * (c & 1);
    for (int item = blockIdx.x; item < count * groups; item += gridDim.x) {
        const int j = item / groups, group = item % groups;
        const FastJob job = jobs[j];
        if (job.columns == 0) continue;   // (uniform over the block)
        std::uint16_t* hidden = reinterpret_cast<std::uint16_t*>(workspace + std::size_t(j) * tc::kJobWorkspace);
        const std::uint16_t* x = Down ? hidden : job.x;
        const bool live = g < job.columns;
        const unsigned keep = live ? ~0u : 0u;
        const unsigned* column = reinterpret_cast<const unsigned*>(x + std::size_t(live ? g : 0) * K) + c + std::size_t(per) * warp * 8;
        const std::uint8_t* unit = job.record + (Down ? tc::kDownOffset : 0) + (std::size_t(group) * blocks + std::size_t(per) * warp) * tc::kUnit;
        float s0 = 0.f, s1 = 0.f, s2 = 0.f, s3 = 0.f;
#pragma unroll 1
        for (int b = 0; b < per; b += P, unit += P * tc::kUnit, column += P * 8) {
            unsigned lo[P], hi[P], sa[P], sb[P], b0[P], b1[P];
#pragma unroll
            for (int p = 0; p < P; ++p) {
                lo[p] = __ldg(reinterpret_cast<const unsigned*>(unit + p * tc::kUnit + at));
                hi[p] = __ldg(reinterpret_cast<const unsigned*>(unit + p * tc::kUnit + at + 64));
                sa[p] = __ldg(unit + p * tc::kUnit + 128 + g); sb[p] = __ldg(unit + p * tc::kUnit + 136 + g);
                b0[p] = __ldg(column + p * 8); b1[p] = __ldg(column + p * 8 + 4);   // (down: the up kernel's, a launch before)
            }
#pragma unroll
            for (int p = 0; p < P; ++p) {
                const unsigned word = ((lo[p] >> shift) & 0xffffu) | ((hi[p] >> shift) << 16);
                unsigned a0, a1, a2, a3;
                decode4(word & 0x0f0f0f0fu, a0, a2);
                decode4((word >> 4) & 0x0f0f0f0fu, a1, a3);
                const float w0 = scale[sa[p]], w1 = scale[sb[p]];
                float d0, d1, d2, d3;
                tile(d0, d1, d2, d3, a0, a1, a2, a3, b0[p] & keep, b1[p] & keep);
                s0 = fmaf(d0, w0, s0); s1 = fmaf(d1, w0, s1);
                s2 = fmaf(d2, w1, s2); s3 = fmaf(d3, w1, s3);
            }
        }
        if constexpr (S > 1) {
            __shared__ float part[S][4][32];
            part[warp][0][lane] = s0; part[warp][1][lane] = s1; part[warp][2][lane] = s2; part[warp][3][lane] = s3;
            __syncthreads();
            if (warp == 0) {
#pragma unroll
                for (int w = 1; w < S; ++w) { s0 += part[w][0][lane]; s1 += part[w][1][lane]; s2 += part[w][2][lane]; s3 += part[w][3][lane]; }
            }
        }
        if (warp == 0) {
            if constexpr (Down) {
#pragma unroll
                for (int v = 0; v < 2; ++v) {
                    const int col = 2 * c + v;
                    if (col >= job.columns) continue;
                    std::uint16_t* y = job.y + std::size_t(col) * tc::kH + group * 16 + g;
                    y[0] = float_bf16((v ? s1 : s0) * job.multiplier[2]);
                    y[8] = float_bf16((v ? s3 : s2) * job.multiplier[2]);
                }
            } else {
                const float u0 = __shfl_xor_sync(0xffffffffu, s0, 4), u1 = __shfl_xor_sync(0xffffffffu, s1, 4);
                const float u2 = __shfl_xor_sync(0xffffffffu, s2, 4), u3 = __shfl_xor_sync(0xffffffffu, s3, 4);
                if (!(g & 1)) {
#pragma unroll
                    for (int v = 0; v < 2; ++v) {
                        const int col = 2 * c + v;
                        if (col >= job.columns) continue;
#pragma unroll
                        for (int half = 0; half < 2; ++half) {
                            const float gate = fminf((half ? (v ? s3 : s2) : (v ? s1 : s0)) * job.multiplier[0], 7.f);
                            const float up = fminf(fmaxf((half ? (v ? u3 : u2) : (v ? u1 : u0)) * job.multiplier[1], -7.f), 7.f);
                            hidden[std::size_t(col) * tc::kI + group * 8 + g / 2 + 4 * half] =
                                float_bf16(gate / (1.f + expf(-1.702f * gate)) * (up + 1.f));
                        }
                    }
                }
            }
        }
        if constexpr (S > 1) __syncthreads();   // (part is rewritten by the next item)
    }
}

// The pass's columns to its result slot, then (the last block) the header and the word.
struct ResultArgs { Result* res; StateBody* st; const uint16_t* y; };
__global__ void __launch_bounds__(kThreads) result_kernel(ResultArgs a) {
    StateBody* st = a.st;
    const int status = vload(&st->status), columns = status == kServed ? vload(&st->columns) : 0;
    const int threads = gridDim.x * blockDim.x, me = blockIdx.x * blockDim.x + threadIdx.x;
    for (int p = me; p < columns * kChunks; p += threads) {
        const int col = p / kChunks, q = p % kChunks;
        reinterpret_cast<uint4*>(a.res->y[col])[q] = reinterpret_cast<const uint4*>(a.y + size_t(col) * kHidden)[q];
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0 && atomicAdd(&st->done_blocks, 1u) == gridDim.x - 1) {
        st->done_blocks = 0;
        a.res->status = status;
        a.res->count = status == kServed ? vload(&st->count) : 0;
        a.res->columns = columns;
        a.res->kernel = vload(&st->kernel); a.res->format = vload(&st->format);
        a.res->sm89 = vload(&st->sm89); a.res->multi = vload(&st->kind);
        a.res->t_seen = vload(&st->t_seen);
        a.res->t_done = now_ns();
        __threadfence_system();
        vstore(&a.res->word, vload(&st->word));
    }
}

// Between steps: one thread polls the occupancy word until the next arm changes it, or the bound passes.
__global__ void spin_kernel(const unsigned long long* word, unsigned long long expected) {
    const unsigned long long t0 = now_ns();
    while (vload(word) == expected && now_ns() - t0 < kSpinBoundNs) __nanosleep(200);
}

using FfnFn = void (*)(const FastJob*, const StateBody*, int, uint8_t*);
template <int S, int P> FfnFn ffn_of(bool down) { return down ? ffn_kernel<S, true, P> : ffn_kernel<S, false, P>; }
template <int S> FfnFn ffn_s(bool down, int loads) {
    switch (loads) {
        case 2: return ffn_of<S, 2>(down);
        case 4: return ffn_of<S, 4>(down);
        case 8: return ffn_of<S, 8>(down);
        default: return ffn_of<S, 1>(down);
    }
}
FfnFn ffn_for(int warps, int loads, bool down) {
    switch (warps) {
        case 1: return ffn_s<1>(down, loads);
        case 2: return ffn_s<2>(down, loads);
        case 4: return ffn_s<4>(down, loads);
        default: return ffn_s<8>(down, loads);
    }
}
bool valid_loads(int p) { return p == 1 || p == 2 || p == 4 || p == 8; }
}  // namespace

bool launch_request(const RequestArgs& a, cudaStream_t stream) {
    if (!a.top || !a.w || !a.xn || !a.snapshot || a.T < 1 || a.T > kMaxRows || a.k < 1 || a.k > 8) return false;
    for (int c = 0; c < 2; ++c) if (a.request[c] && (a.format[c] < 0 || a.format[c] > 3)) return false;
    plan_kernel<<<1, kThreads, 0, stream>>>(a);
    rows_kernel<<<(a.T * (kHidden / 16) + kThreads - 1) / kThreads, kThreads, 0, stream>>>(a);
    return cudaGetLastError() == cudaSuccess;
}

namespace {
void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) throw std::runtime_error(std::string("helper hand-off: ") + where + ": " + cudaGetErrorString(e));
}
struct DeviceGuard {
    int previous = -1;
    explicit DeviceGuard(int id) { ck(cudaGetDevice(&previous), "get device"); ck(cudaSetDevice(id), "set device"); }
    ~DeviceGuard() { if (previous >= 0) (void)cudaSetDevice(previous); }
};
}  // namespace

Lane::Lane(int device, uintptr_t pool, size_t stride, int capacity) : device_(device), capacity_(capacity), pool_(pool), stride_(stride) {
    if (device < 0 || !pool || !stride || capacity < 1) throw std::invalid_argument("helper hand-off: invalid card or pool");
    DeviceGuard guard(device);
    try {
        cudaDeviceProp prop{};
        ck(cudaGetDeviceProperties(&prop, device), "properties");
        sms_ = prop.multiProcessorCount;
        sm89_device_ = prop.major == 8 && prop.minor == 9 && mm3::helper_sm89::built();
        ck(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking), "stream");
        const unsigned flags = cudaHostAllocPortable | cudaHostAllocMapped;
        ck(cudaHostAlloc(reinterpret_cast<void**>(&requests_), sizeof(Request) * kSlots, flags), "request slots");
        ck(cudaHostAlloc(reinterpret_cast<void**>(&results_), sizeof(Result) * kSlots, flags), "result slots");
        ck(cudaHostAlloc(reinterpret_cast<void**>(&tables_), sizeof(Entry) * kLayers * kExperts, flags), "tables");
        ck(cudaHostAlloc(reinterpret_cast<void**>(&control_), 64, flags), "control words");
        std::memset(static_cast<void*>(requests_), 0, sizeof(Request) * kSlots);
        std::memset(static_cast<void*>(results_), 0, sizeof(Result) * kSlots);
        std::memset(static_cast<void*>(tables_), 0, sizeof(Entry) * kLayers * kExperts);
        std::memset(control_, 0, 64);
        ck(cudaMalloc(reinterpret_cast<void**>(&state_), sizeof(State)), "state");
        ck(cudaMemset(state_, 0, sizeof(State)), "state");
        const size_t io = size_t(kMaxJobs) * kHidden * sizeof(uint16_t);
        ck(cudaMalloc(reinterpret_cast<void**>(&d_rows_), size_t(kMaxRows) * kHidden * sizeof(uint16_t)), "rows");
        ck(cudaMalloc(reinterpret_cast<void**>(&d_x_), io), "inputs");
        ck(cudaMalloc(reinterpret_cast<void**>(&d_y_), io), "outputs");
        ck(cudaMalloc(&d_jobs_, sizeof(FastJob) * kMaxJobs), "jobs");
        workspace_bytes_ = tc::workspace_bytes(kMaxJobs);
        ck(cudaMalloc(reinterpret_cast<void**>(&workspace_), workspace_bytes_), "workspace");
        ck(cudaDeviceSynchronize(), "allocation");
    } catch (...) {
        release();
        throw;
    }
    // (portable mapped memory has one address on every card: the graph and the 5090 use these pointers as they are)
    void* view = nullptr;
    if (cudaHostGetDevicePointer(&view, requests_, 0) != cudaSuccess || view != static_cast<void*>(requests_) ||
        cudaHostGetDevicePointer(&view, results_, 0) != cudaSuccess || view != static_cast<void*>(results_) ||
        cudaHostGetDevicePointer(&view, control_, 0) != cudaSuccess || view != static_cast<void*>(control_)) {
        (void)cudaGetLastError();
        release();
        throw std::runtime_error("helper hand-off: pinned slots without one address on the card");
    }
}

Lane::~Lane() { release(); }

void Lane::release() noexcept {
    try {
        DeviceGuard guard(device_);
        if (control_) __atomic_add_fetch(&control_[1], 1ull, __ATOMIC_SEQ_CST);   // (the clock holder exits)
        if (control_) __atomic_store_n(&control_[0], ~0ull, __ATOMIC_SEQ_CST);    // (an armed step's takes exit)
        if (stream_ && cudaStreamSynchronize(stream_) != cudaSuccess) return;     // (an unknown writer: keep its buffers)
        for (auto& kv : graphs_) if (kv.second) cudaGraphExecDestroy(kv.second);
        graphs_.clear();
        if (stream_) cudaStreamDestroy(stream_);
        if (workspace_) cudaFree(workspace_);
        if (prepared_) cudaFree(prepared_);
        if (d_jobs_) cudaFree(d_jobs_);
        if (d_y_) cudaFree(d_y_);
        if (d_x_) cudaFree(d_x_);
        if (d_rows_) cudaFree(d_rows_);
        if (state_) cudaFree(state_);
        if (control_) cudaFreeHost(control_);
        if (tables_) cudaFreeHost(tables_);
        if (results_) cudaFreeHost(results_);
        if (requests_) cudaFreeHost(requests_);
    } catch (...) {
    }
    stream_ = nullptr; workspace_ = prepared_ = nullptr; d_jobs_ = nullptr; d_y_ = d_x_ = d_rows_ = nullptr; state_ = nullptr;
    control_ = nullptr; tables_ = nullptr; results_ = nullptr; requests_ = nullptr;
}

cudaGraphExec_t Lane::build(const std::vector<int>& layers, bool windows, Kernels k, std::string& err) {
    const auto t0 = std::chrono::steady_clock::now();
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    const auto grid_of = [&](FfnFn f, int warps) {
        int per = 0;
        if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per, f, 32 * warps, 0) != cudaSuccess || per < 1) per = 1;
        (void)cudaGetLastError();
        return sms_ * per;
    };
    // (one row: every pass one-column, so only the one-column kernels; windows: both kinds, each gated)
    const int kinds = windows ? 2 : 1;
    const bool nvfp4 = kernel_family(k.one) >= 0;
    const int warps[2] = {kernel_warps(k.one), kernel_warps(k.multi)};
    FfnFn up[2] = {nullptr, nullptr}, down[2] = {nullptr, nullptr};
    int grid[2] = {k.grid_blocks_per_sm * sms_, k.grid_blocks_per_sm * sms_};
    if (!nvfp4) for (int kind = 0; kind < kinds; ++kind) {
        up[kind] = ffn_for(warps[kind], k.loads, false);
        down[kind] = ffn_for(warps[kind], k.loads, true);
        grid[kind] = grid_of(up[kind], warps[kind]);
    }
    bool ok = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) == cudaSuccess;
    if (ok) {
        begin_kernel<<<1, 1, 0, stream_>>>(state_);
        for (size_t m = 0; m < layers.size() && ok; ++m) {
            TakeArgs ta{request(int(m)), control_, state_, d_rows_, d_x_, d_y_, static_cast<FastJob*>(d_jobs_),
                        pool_, pool_ + stride_ * size_t(capacity_), stride_, int(m), k};
            take_kernel<<<kTakeBlocks, kThreads, 0, stream_>>>(ta);
            for (int kind = 0; kind < kinds; ++kind) {
                const int w = warps[kind];
                if (nvfp4) {
                    const int family = kernel_family(kind ? k.multi : k.one);
                    const auto* in = reinterpret_cast<const uint8_t*>(d_x_);
                    if (needs_preparation(family, k.format)) {
                        prepare_q_kernel<<<(kMaxJobs * (kHidden / 16) + kThreads - 1) / kThreads, kThreads, 0, stream_>>>(
                            in, prepared_, state_, k.format, family, kind);
                        ok = ok && cudaGetLastError() == cudaSuccess;
                        in = prepared_;
                    }
                    const cudaError_t e = k.sm89
                        ? mm3::helper_sm89::q_queued(family, w, k.loads, static_cast<FastJob*>(d_jobs_), d_x_, in, workspace_,
                            &state_->count, &state_->kind, kind, grid[kind], stream_)
                        : hq::ffn_queued(family, w, k.loads, static_cast<FastJob*>(d_jobs_), d_x_, in, workspace_,
                            &state_->count, &state_->kind, kind, grid[kind], stream_);
                    ok = ok && e == cudaSuccess;
                } else {
                    void* up_args[] = {&d_jobs_, &state_, const_cast<int*>(&kind), &workspace_};
                    ok = ok && cudaLaunchKernel(reinterpret_cast<const void*>(up[kind]), dim3(grid[kind]), dim3(32 * w), up_args, 0, stream_) == cudaSuccess;
                    ok = ok && cudaLaunchKernel(reinterpret_cast<const void*>(down[kind]), dim3(grid[kind]), dim3(32 * w), up_args, 0, stream_) == cudaSuccess;
                }
            }
            ResultArgs ra{result(int(m)), state_, d_y_};
            result_kernel<<<kResultBlocks, kThreads, 0, stream_>>>(ra);
        }
        ok = ok && cudaGetLastError() == cudaSuccess;
        const bool ended = cudaStreamEndCapture(stream_, &graph) == cudaSuccess;
        ok = ok && ended && graph && cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess;
    }
    if (graph) cudaGraphDestroy(graph);
    if (!ok) {
        err = std::string("helper hand-off: the step graph did not build (") + cudaGetErrorString(cudaGetLastError()) + ")";
        if (exec) cudaGraphExecDestroy(exec);
        return nullptr;
    }
    ++captures_;
    capture_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return exec;
}

bool Lane::arm(const std::vector<int>& layers, bool windows, Kernels k, std::string& err) {
    if (broken_) { err = "helper hand-off: this card's step graph failed to build earlier in the load"; return false; }
    if (armed_) { err = "helper hand-off: the previous step was not ended"; return false; }
    const int one = kernel_family(k.one), multi = kernel_family(k.multi);
    const bool nvfp4 = one >= 0;
    if ((k.grid_blocks_per_sm != 2 && k.grid_blocks_per_sm != 4 && k.grid_blocks_per_sm != 8) ||
        (!nvfp4 && k.grid_blocks_per_sm != 2)) {
        err = "helper hand-off: grid_blocks_per_sm must be 2, 4 or 8; 4/8 require an NVFP4 kernel";
        return false;
    }
    const bool format_ok = k.format == 0 || (k.format == 1 && nvfp4) || (k.format == 2 && one == 4 && multi == 4) ||
        (k.format == 3 && one >= 0 && one <= 2 && multi >= 0 && multi <= 2 && (one == 2) == (multi == 2));
    if (layers.empty() || layers.size() > 255 || !kernel_warps(k.one) || !kernel_warps(k.multi) || !valid_loads(k.loads) ||
        nvfp4 != (multi >= 0) || !format_ok || (k.sm89 && (!nvfp4 || !sm89_device_)) || ((one == 2 || multi == 2) && !k.sm89)) {
        err = "helper hand-off: incompatible kernel IDs, input format, architecture, or layer list";
        return false;
    }
    try {
        DeviceGuard guard(device_);
        if (nvfp4 && (needs_preparation(one, k.format) || (windows && needs_preparation(multi, k.format))) && !prepared_)
            ck(cudaMalloc(reinterpret_cast<void**>(&prepared_), size_t(kMaxJobs) * kHidden * 2), "NVFP4 preparation");
        const auto key = std::make_tuple(layers, windows, k.one, windows ? k.multi : 0, k.loads, k.format, k.sm89, k.grid_blocks_per_sm);
        auto it = graphs_.find(key);
        if (it == graphs_.end()) {
            cudaGraphExec_t g = build(layers, windows, k, err);
            if (!g) { broken_ = true; return false; }
            it = graphs_.emplace(key, g).first;
        }
        // the clock holder from the last step's end exits, then the step's graph queues behind it
        __atomic_add_fetch(&control_[1], 1ull, __ATOMIC_SEQ_CST);
        occupancy_armed_ = false;
        // (the graph's first kernel advances the card's step counter: the host's follows only a launch that was taken)
        if (cudaGraphLaunch(it->second, stream_) != cudaSuccess) {
            err = std::string("helper hand-off: the step did not launch (") + cudaGetErrorString(cudaGetLastError()) + ")";
            broken_ = true;
            return false;
        }
        ++step_;
        (void)cudaStreamQuery(stream_);   // (WSL groups submissions: start it now)
        armed_ = true;
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}

bool Lane::wait(int m, double deadline_s, int& status, uint64_t& seen_ns, uint64_t& wait_ns, std::string& err) {
    const Result* r = result(m);
    const unsigned long long want = word(m);
    const auto t0 = std::chrono::steady_clock::now();
    const auto ns = [](std::chrono::steady_clock::time_point t) {
        return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count());
    };
    for (unsigned spins = 0;; ++spins) {
        if (__atomic_load_n(&r->word, __ATOMIC_ACQUIRE) == want) break;
        if ((spins & 1023) == 1023) {
            const auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration<double>(now - t0).count() > deadline_s) {
                err = "helper hand-off: card " + std::to_string(device_) + " did not publish layer index " + std::to_string(m) +
                      " within " + std::to_string(deadline_s) + " s";
                return false;
            }
        }
    }
    const auto seen = std::chrono::steady_clock::now();
    std::atomic_thread_fence(std::memory_order_acquire);
    status = __atomic_load_n(&r->status, __ATOMIC_ACQUIRE);
    seen_ns = ns(seen);
    wait_ns = ns(seen) - ns(t0);
    return true;
}

void Lane::end_step(bool occupancy) {
    if (!armed_) return;
    armed_ = false;
    __atomic_store_n(&control_[0], step_, __ATOMIC_SEQ_CST);
    if (!occupancy || occupancy_armed_) return;
    try {
        DeviceGuard guard(device_);
        const unsigned long long expected = __atomic_load_n(&control_[1], __ATOMIC_SEQ_CST);
        spin_kernel<<<1, 1, 0, stream_>>>(control_ + 1, expected);
        if (cudaGetLastError() == cudaSuccess) { (void)cudaStreamQuery(stream_); occupancy_armed_ = true; }
        (void)cudaGetLastError();
    } catch (...) {
    }
}

bool Lane::read_back(int columns, std::vector<uint16_t>& out, std::string& err) {
    try {
        DeviceGuard guard(device_);
        out.resize(size_t(std::max(columns, 0)) * kHidden);
        if (!out.empty()) ck(cudaMemcpy(out.data(), d_y_, out.size() * sizeof(uint16_t), cudaMemcpyDeviceToHost), "read back");
        return true;
    } catch (const std::exception& e) {
        err = e.what();
        return false;
    }
}
}  // namespace mm3::handoff
