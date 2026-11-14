"""tools/fork_cuda_only.py - our MiniMax native files (the minimax-wsl fork) that only this fork's Linux (WSL2) CUDA
build compiles: never an AMD (HIP) build and never a Windows build.  Upstream's portability checks leave them out
(tools/check_hip_compat.py, tools/test_port_portability.py), so the rest of the tree, new upstream code included, is
still checked.  The user's decision of 2026-11-13 ("Exclude our native files", CLAUDE.md, Upstream rows).

A file goes here only if it is ours (not in upstream's tree) and a check flags it; tools/test_hip_compat.py checks
that each listed file exists.
"""
NATIVE = ("the native serving path: CUDA driver API, pinned host frames and the helper cards' queues, linked with "
          "Infernix-minimax's service (mm3_native); built only by this fork's Linux CUDA build")
NATIVE_TEST = "a test of the native serving path (" + NATIVE + ")"
FILES = {
    "src/mm3/mm3_helper_kernels.cuh": NATIVE,
    "src/mm3/mm3_native_experts.cu": NATIVE,
    "src/mm3/mm3_native_helper.cu": NATIVE,
    "src/mm3/mm3_resident_router.cu": NATIVE,
    "tests/mm3_native_adaptive_test.cpp": NATIVE_TEST,
    "tests/mm3_native_compressed_placement_test.cpp": NATIVE_TEST,
    "tests/mm3_native_helper_live_cuda_test.cu": NATIVE_TEST,
    "tests/mm3_native_placement_test.cpp": NATIVE_TEST,
    "tests/mm3_native_prefetch_test.cpp": NATIVE_TEST,
    "tests/mm3_native_prompt_test.cu": NATIVE_TEST,
    "tests/mm3_r2_fake_cuda/cuda_runtime_api.h": "a stand-in CUDA header for the native path's CPU tests",
    "tests/mm3_resident_router_test.cu": NATIVE_TEST,
}
