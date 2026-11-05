#pragma once
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
using cudaError_t = int;
using cudaStream_t = void*;
using cudaEvent_t = void*;
constexpr int cudaSuccess=0, cudaErrorNotReady=1;
constexpr unsigned cudaStreamNonBlocking=1, cudaEventDisableTiming=2, cudaHostAllocPortable=1;
enum cudaMemcpyKind { cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost };
inline std::atomic<bool> fake_stall{false}, fake_block_query{false}, fake_block_stream{false};
inline std::atomic<int> fake_allocations{0}, fake_frees{0}, fake_queries{0}, fake_not_ready_queries{0};
inline std::atomic<bool> fake_fail_copy{false}, fake_fail_record{false}, fake_fail_query{false}, fake_not_portable{false};
inline const char* cudaGetErrorString(cudaError_t) { return "injected"; }
inline cudaError_t cudaGetDevice(int* d) { *d=0; return 0; }
inline cudaError_t cudaSetDevice(int) { return 0; }
inline cudaError_t cudaStreamCreateWithFlags(cudaStream_t* s, unsigned) { *s=reinterpret_cast<void*>(1); return 0; }
inline cudaError_t cudaEventCreateWithFlags(cudaEvent_t* e, unsigned) { *e=reinterpret_cast<void*>(1); return 0; }
inline cudaError_t cudaStreamDestroy(cudaStream_t) { return 0; }
inline cudaError_t cudaEventDestroy(cudaEvent_t) { return 0; }
inline cudaError_t cudaStreamQuery(cudaStream_t) { while (fake_block_stream) std::this_thread::yield(); return fake_stall ? cudaErrorNotReady : cudaSuccess; }
inline cudaError_t cudaEventQuery(cudaEvent_t) {
    ++fake_queries;
    while (fake_block_query) std::this_thread::yield();
    if (fake_fail_query) return 9;
    int remaining=fake_not_ready_queries.load();
    while (remaining>0 && !fake_not_ready_queries.compare_exchange_weak(remaining,remaining-1)) {}
    return fake_stall || remaining>0 ? cudaErrorNotReady : cudaSuccess;
}
inline cudaError_t cudaHostGetFlags(unsigned* flags, void*) { *flags=fake_not_portable ? 0 : cudaHostAllocPortable; return 0; }
template<class T> cudaError_t cudaMalloc(T** p, size_t n) { *p=static_cast<T*>(std::calloc(1,n)); if (!*p) return 2; ++fake_allocations; return 0; }
template<class T> cudaError_t cudaMallocHost(T** p, size_t n) { return cudaMalloc(p,n); }
inline cudaError_t cudaFree(void* p) { if(p) { ++fake_frees; std::free(p); } return 0; }
inline cudaError_t cudaFreeHost(void* p) { return cudaFree(p); }
inline cudaError_t cudaMemcpy(void* d, const void* s, size_t n, cudaMemcpyKind) { std::memcpy(d,s,n); return 0; }
inline cudaError_t cudaMemcpyAsync(void* d, const void* s, size_t n, cudaMemcpyKind k, cudaStream_t) { return fake_fail_copy ? 9 : cudaMemcpy(d,s,n,k); }
inline cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t) { return fake_fail_record ? 9 : 0; }
inline cudaError_t cudaEventElapsedTime(float* ms,cudaEvent_t,cudaEvent_t) { *ms=.01f;return 0; }
