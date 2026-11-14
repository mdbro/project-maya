#pragma once
#include <atomic>
#include <cstdlib>
#include <cstring>
#include <thread>
using cudaError_t = int;
using cudaStream_t = void*;
using cudaEvent_t = void*;
constexpr int cudaSuccess=0, cudaErrorNotReady=1;
constexpr int cudaErrorMemoryAllocation=2, cudaErrorInvalidValue=3;   // (the helper's kernel paths: never taken in these tests)
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
// A join waits for the stream: a stalled one never completes, so a join of it holds its caller (as on a device).
inline cudaError_t cudaStreamSynchronize(cudaStream_t) { while (fake_block_stream || fake_stall) std::this_thread::yield(); return cudaSuccess; }
inline cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned) { return 0; }
inline cudaError_t cudaGetLastError() { return 0; }
inline cudaError_t cudaPeekAtLastError() { return 0; }
// Pinned memory is ordinary host memory here, so its device view is the same address (helper_zero_copy, 65c47c3).
inline cudaError_t cudaHostGetDevicePointer(void** device, void* host, unsigned) { *device=host; return 0; }

// Device code compiled as host code: the production helper defines a __global__ spinner (1c416af) and includes
// mm3_kernels.cuh (65c47c3), whose constexpr helpers are __host__ __device__.
#ifndef __host__
#define __host__
#endif
#ifndef __device__
#define __device__
#endif
#ifndef __global__
#define __global__
#endif
inline long long clock64() { return 0; }
// A kernel launch (`kernel<<<grid, block, shared, stream>>>(args)`) is rewritten by tests/mm3_helper_fake_source.cmake
// into fake_cuda_launch(kernel, grid, block, shared, stream)(args). Nothing runs: these fakes model no device work, and
// the only kernel, the occupancy spinner, only polls a host word. The launch is counted.
inline std::atomic<int> fake_kernel_launches{0};
template<class Kernel, class... Config> auto fake_cuda_launch(Kernel, Config...) {
    return [](auto&&...) { ++fake_kernel_launches; };
}

// CUDA graphs (helper_fast, 6fb0d30; the prediction graph). These fakes cannot capture: a capture fails to begin, which
// production treats as a graph that does not build (its passes then take the separate calls). Launching one is an error.
struct fake_graph;
using cudaGraph_t = fake_graph*;
using cudaGraphExec_t = fake_graph*;
enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal, cudaStreamCaptureModeThreadLocal, cudaStreamCaptureModeRelaxed };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive, cudaStreamCaptureStatusInvalidated };
constexpr int cudaErrorStreamCaptureUnsupported=900;
inline std::atomic<int> fake_capture_attempts{0};
inline cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode) { ++fake_capture_attempts; return cudaErrorStreamCaptureUnsupported; }
inline cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t* graph) { *graph=nullptr; return cudaErrorStreamCaptureUnsupported; }
inline cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus* status) { *status=cudaStreamCaptureStatusNone; return 0; }
inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec, cudaGraph_t, unsigned long long) { *exec=nullptr; return cudaErrorStreamCaptureUnsupported; }
inline cudaError_t cudaGraphDestroy(cudaGraph_t) { return 0; }
inline cudaError_t cudaGraphExecDestroy(cudaGraphExec_t) { return 0; }
inline cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t) { return cudaErrorStreamCaptureUnsupported; }
