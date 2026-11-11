#!/bin/sh
# Builds the helper cards' kernel benchmark for the RTX 3090 (sm_86) and the RTX 4080 (sm_89) in one binary.
# usage: build.sh OUTDIR        (the donor kernels: MINIMAX_SERVICE_ROOT, default /home/mdwbr/Infernix-minimax)
# Not while a timed request runs (the plan's quiet rule): it is a build.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../../.." && pwd)
DONOR=${MINIMAX_SERVICE_ROOT:-/home/mdwbr/Infernix-minimax}
OUT=${1:?usage: build.sh OUTDIR}
mkdir -p "$OUT"
NVCC=${NVCC:-/usr/local/cuda/bin/nvcc}
"$NVCC" -O3 -std=c++20 -I "$DONOR/src" -I "$ROOT/src/mm3" \
  -gencode arch=compute_86,code=sm_86 -gencode arch=compute_89,code=sm_89 \
  "$HERE/helper_tc_bench.cu" "$DONOR/src/ops/minimax_nvfp4_ffn/fast_cuda.cu" -o "$OUT/helper_tc_bench"
git -C "$ROOT" rev-parse --short HEAD > "$OUT/helper_tc_bench.source" 2>/dev/null || true
git -C "$ROOT" status --short -- src/mm3/mm3_helper_kernels.cuh tools/mm3/bench >> "$OUT/helper_tc_bench.source" 2>/dev/null || true
echo "built $OUT/helper_tc_bench"
