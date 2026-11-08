// Copy-engine to compute handoff latency on one GPU, timed by the profiler (CUPTI), not by this program. No engine state touched.
//
// Each iteration uploads one expert-sized record (31,850,752 bytes, as the engine's demand copies) and runs a small kernel that
// reads it, then a 4 KB readback and a second small kernel, all on one stream unless the mode says otherwise:
//
//   same    copy and kernels on one stream, as the engine's demand copies are
//   event   copy on its own stream; the kernel's stream waits on an event recorded after it
//   poll    as same, with the host spinning cudaStreamQuery until the iteration is done (a polling host thread)
//   busy    as same, with the host doing plain CPU work, no CUDA calls, for 2 ms before it synchronizes (the CPU lane)
//   mapped  no copy engine: a kernel reads the record from mapped pinned host memory into device memory
//
// Run under nsys with --capture-range=cudaProfilerApi; native_handoff_probe.py reads the gaps from the export.
//
//   nvcc -O2 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_120,code=sm_120 -o native_handoff_probe native_handoff_probe.cu
//   native_handoff_probe <device> <mode> <iterations>
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } } while (0)

static const size_t RECORD = 31850752;

__global__ void touch(float *p, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = p[i] * 1.0001f + 1.0f;
}

__global__ void mapped_copy(const int4 *__restrict__ src, int4 *__restrict__ dst, size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) dst[i] = src[i];
}

static void spin_cpu(double ms) {
    auto until = std::chrono::steady_clock::now() + std::chrono::duration<double, std::milli>(ms);
    volatile double x = 1.0;
    while (std::chrono::steady_clock::now() < until) x = x * 1.0000001 + 1e-9;
}

int main(int argc, char **argv) {
    if (argc != 4) { fprintf(stderr, "usage: %s <device> <same|event|poll|busy|mapped> <iterations>\n", argv[0]); return 2; }
    const int device = atoi(argv[1]);
    const std::string mode = argv[2];
    const int iterations = atoi(argv[3]);
    if (mode != "same" && mode != "event" && mode != "poll" && mode != "busy" && mode != "mapped") { fprintf(stderr, "unknown mode\n"); return 2; }
    CK(cudaSetDevice(device));
    void *host, *host_device, *dev, *small;
    CK(cudaHostAlloc(&host, RECORD, cudaHostAllocMapped));
    memset(host, 1, RECORD);
    CK(cudaHostGetDevicePointer(&host_device, host, 0));
    CK(cudaMalloc(&dev, RECORD));
    CK(cudaHostAlloc(&small, 4096, cudaHostAllocDefault));
    cudaStream_t stream, copies;
    CK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CK(cudaStreamCreateWithFlags(&copies, cudaStreamNonBlocking));
    cudaEvent_t uploaded;
    CK(cudaEventCreateWithFlags(&uploaded, cudaEventDisableTiming));
    int sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
    auto iteration = [&]() {
        if (mode == "mapped") {
            mapped_copy<<<sms * 4, 512, 0, stream>>>((const int4 *)host_device, (int4 *)dev, RECORD / 16);
        } else if (mode == "event") {
            CK(cudaMemcpyAsync(dev, host, RECORD, cudaMemcpyHostToDevice, copies));
            CK(cudaEventRecord(uploaded, copies));
            CK(cudaStreamWaitEvent(stream, uploaded, 0));
        } else {
            CK(cudaMemcpyAsync(dev, host, RECORD, cudaMemcpyHostToDevice, stream));
        }
        touch<<<4, 256, 0, stream>>>((float *)dev, 1024);
        CK(cudaMemcpyAsync(small, dev, 4096, cudaMemcpyDeviceToHost, stream));
        touch<<<4, 256, 0, stream>>>((float *)dev, 1024);
        CK(cudaGetLastError());
        if (mode == "poll") {
            cudaError_t e;
            while ((e = cudaStreamQuery(stream)) == cudaErrorNotReady) {}
            CK(e);
        } else {
            if (mode == "busy") spin_cpu(2.0);
            CK(cudaStreamSynchronize(stream));
        }
    };
    auto warm = std::chrono::steady_clock::now() + std::chrono::seconds(2);  // leave idle clocks before measuring
    while (std::chrono::steady_clock::now() < warm) iteration();
    CK(cudaProfilerStart());
    for (int i = 0; i < iterations; i++) iteration();
    CK(cudaProfilerStop());
    printf("device %d mode %s iterations %d sms %d\n", device, mode.c_str(), iterations, sms);
    return 0;
}
