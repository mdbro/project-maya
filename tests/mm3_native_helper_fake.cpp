// Compile the real helper implementation against the fixture CUDA API. The file is the production
// src/mm3/mm3_native_helper.cu with only its kernel launch syntax rewritten for a C++ compiler
// (tests/mm3_helper_fake_source.cmake, run by the build).
#include "mm3_native_helper_rewritten.inc"
#include <cstring>
// mm3_kernels.cuh's word-copy kernel (helper_zero_copy, 65c47c3), as the fake API's copies are: done at once, device and
// pinned memory being host memory here.
namespace mm3::k {
void copy_words(const void* src, void* dst, size_t bytes, cudaStream_t, const void* src2, void* dst2, size_t bytes2,
                const void* src3, void* dst3, size_t bytes3) {
    if (bytes) std::memcpy(dst, src, bytes);
    if (bytes2) std::memcpy(dst2, src2, bytes2);
    if (bytes3) std::memcpy(dst3, src3, bytes3);
}
}
