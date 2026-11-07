# NVIDIA NIM and SGLang references for Maya

Source audit, 2026-11-07. The [central plan](README.md) owns requirements, selected
implementations and completion. This is a technical reference, not a new work queue.
The findings below are from source, recipes and public image metadata; they establish
candidate mechanisms and compatibility limits, not measured gains in Maya.

## Sources and reproducibility

- NIM: `nvcr.io/nim/nvidia/sglang-model-free-nim-experimental:2.1.7-variant`.
  Its AMD64 manifest is
  `sha256:43aaaaa0c99a0fe811d9cedb91abd833b6280302fdd8c72577fb9a565d41763d`.
  Public image labels identify SGLang base
  `44bc478863052efdc455b0282b49830f68fdefb3`, patch profile `sglang_44bc47`,
  CUDA 13.0.3 and FlashInfer 0.7.0.post1. These are declared build identities;
  the NVIDIA patch contents were not extracted from the image.
- Five recipe files from
  [`nim/nvidia/minimax-m3.1-flash-recipe:2.1.7-variant`](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/resources/minimax-m3.1-flash-recipe).
  All five downloads matched the SHA-256 values returned by NGC.
- Public SGLang source pinned at
  [`9d547e19ff519f0d1b964b0b1c014d98dc65caf4`](https://github.com/sgl-project/sglang/tree/9d547e19ff519f0d1b964b0b1c014d98dc65caf4).
  This is newer than the NIM image's declared base; do not assume every inspected
  upstream feature is present in that image.
- MiniMax MSA pinned at
  [`1e7bf76f26a5f53cfc2d4d297e1ec2e40955e431`](https://github.com/MiniMax-AI/MSA/tree/1e7bf76f26a5f53cfc2d4d297e1ec2e40955e431),
  the pin in the inspected SGLang CUDA 13.4 Dockerfile.
- Public FlashInfer source pinned at
  [`4e30ecdb0a3c1b20e129861f7c047f0ddf97b609`](https://github.com/flashinfer-ai/flashinfer/tree/4e30ecdb0a3c1b20e129861f7c047f0ddf97b609).
  Its newer SM120 APIs are a separate source reference, not a claim about the NIM image.
- Maya implementation comparison: commit `02cc4c6975e07a1ea185f86dffbcc19bf983ead9`
  in `minimax-wsl/shared-model-diagnostic`. The assigned plan checkout was `3282d65`.
  In particular, the comparison uses the later spinner and grouped-execution evidence,
  rather than treating the initial diagnostic's questions as still unanswered.

[Downloaded recipes, image metadata, source snapshots and manifest](/mnt/wsl/transfer/minimax-wsl/research/nim-sglang-20261107/).

## What the NIM recipes actually select

| Recipe | Distinct settings |
|---|---|
| B200 TP4 | TP4/EP4, DSpark, `FMHA_SM100_ALLOW_JIT=1`, MegaMoE token capacity 32,768 |
| B200 TP8 | TP8/EP8, attention DP2, DP language-model head, otherwise the TP4 settings |
| GB200 TP4 | TP4, DSpark, `flashinfer_trtllm_routed`, breakable prefill graphs |
| H200 TP8 and H20-3e TP8 | Identical files: TP8, breakable prefill graphs, context 140,000; no DSpark flag |

All five set maximum running requests and decode-graph batch size to 128. The three
Blackwell recipes set prefill chunks to 32,768 and static memory fraction to 0.8;
the Hopper recipes use 0.85. These are throughput configurations for different
hardware and concurrency. Copying their batch, parallelism or memory settings into
Maya would not reproduce their execution.

Although their filenames contain `fp8`, the YAML files do not explicitly select a
quantization method. The model and backend resolve it. Current upstream's
`MiniMaxM3Nvfp4Config` explicitly distinguishes MXFP8 projections and packed NVFP4
routed experts. A recipe filename is not an arithmetic specification.

The recipes also illustrate a version boundary: the upstream
[Hopper DSpark change](https://github.com/sgl-project/sglang/pull/46868) landed after
the recipe release. Its author's H200 comparison found block verification faster
at high concurrency but slower at concurrency one. That result does not predict
Maya speed; it does show why throughput-oriented verification choices are not
automatic interactive-latency defaults.

## References with a practical Maya counterpart

### 1. DSpark context projection batching and small CUDA graphs

SGLang's
[`DSparkMiniMaxDraftModel._stacked_ctx_kv_params`](https://github.com/sgl-project/sglang/blob/9d547e19ff519f0d1b964b0b1c014d98dc65caf4/python/sglang/srt/models/dspark_minimax.py)
collects the draft layers' K/V projection weights and scales into one projection,
then splits its output for each layer's normalization, rotary transform and cache
store. Its DSpark proposer can run the draft through a CUDA graph with stable
input/output buffers and staged sampling parameters.

Maya's `mm3_dspark.cu::context_finish` applies K and V projections separately in
a loop over draft layers. The inspected draft implementation contains no CUDA
graph capture/replay. This is a concrete source-level difference in a GPU-only
part of the engine, without requiring the CPU expert policy to move onto the GPU.
The newer speculative evidence records drafting at about 9.7 ms per round; that is
an observed component to investigate, not an estimate of removable launch time.

**Candidate:** batch the context projections and capture eligible dense draft work
with fixed buffers. Preserve the actual sampled-proposal probabilities, random
draws, accepted-prefix/KV updates, ring reuse and cancellation lifetimes. A graph
must consume each call's positions and uniforms rather than replay captured values.

One tempting upstream shortcut is not directly compatible: the
[`DsparkSamplingAcceptGraph`](https://github.com/sgl-project/sglang/blob/9d547e19ff519f0d1b964b0b1c014d98dc65caf4/python/sglang/srt/speculative/dspark_components/dspark_sampling_accept_graph.py)
path excludes top-k/top-p and logit adjustments, whereas the current Maya sampling
contract uses top-p 0.95 and lenient acceptance. Its graph structure is a reference;
its acceptance path cannot simply replace Maya's sampler.

### 2. Combine target QKV and indexer projections

In
[`MiniMaxM3Attention.maybe_build_fused_qkv_index`](https://github.com/sgl-project/sglang/blob/9d547e19ff519f0d1b964b0b1c014d98dc65caf4/python/sglang/srt/models/minimax_m3.py),
SGLang concatenates the main and indexer projection weights/scales, executes one
linear operation, and releases the unused original allocations. It also has a
combined Q/K/index normalization-and-RoPE path for the MiniMax head geometry.

Maya already batches Q/K/V jobs and already fuses normalization with RoPE. Its
indexer projections and subsequent index work run on a side stream beside QKV.
The new comparison is therefore specifically **joint main-plus-index projection
and preparation versus the current two branches**, not generic fusion versus an
entirely unfused engine. Fewer submissions/common-input passes could help, but
combining the branches could lose useful overlap. Compare the complete individual
layer invocation, including its join, and avoid retaining duplicate dense weights.

### 3. Graph-compatible expert paging

SGLang's
[`DeviceExecutor`](https://github.com/sgl-project/sglang/blob/9d547e19ff519f0d1b964b0b1c014d98dc65caf4/python/sglang/srt/layers/moe/paged_experts/executor.py)
uses a device LRU decision kernel to build a miss-copy plan, reads the selected
experts directly from a mapped pinned host store into GPU slots, remaps expert
IDs, then runs the usual MoE kernel. Its
[`paging kernels`](https://github.com/sgl-project/sglang/blob/9d547e19ff519f0d1b964b0b1c014d98dc65caf4/python/sglang/kernels/ops/moe/paged_experts.py)
consume stable device pointers and a device-side copy count, allowing dynamic
misses within a captured graph.

This is a useful implementation reference for removing host intervention from
specific paging operations. It does not establish a better residency policy:
the design keeps a fixed number of slots per layer, preserves all expert weights
in the host store, and computes all expert work on one GPU. Its documented path
rejects speculative decoding and multiple GPUs. Its explicit format registry also
does not accept the `MiniMaxM3Nvfp4MoE` subclass inspected here. General NVFP4 support
does not establish support for this MiniMax path.

Maya's complete-forward graph gate requires all MoE layers to be host-free. The
current CPU/helper/adaptive path does not meet that condition. Setting `MM3_GRAPH=1`
alone will not capture it. Smaller GPU-only regions or selected device-side plans
are a more contained adaptation. Preserve original-demand learning, global frame
ownership and CPU/helper servicing; kernel-driven host reads also compete for host
bandwidth, so fewer host calls need not make a complete pass faster.

### 4. NVFP4 kernel and scale handling

SGLang's
[`MiniMaxM3Nvfp4MoE`](https://github.com/sgl-project/sglang/blob/9d547e19ff519f0d1b964b0b1c014d98dc65caf4/python/sglang/srt/layers/quantization/minimax_m3_nvfp4.py)
has native FP4 Blackwell routes and a Marlin W4A16 expert route for SM80 and above.
The latter keeps weights packed and dequantizes into the tensor-core computation.
Its overall model configuration still requires SM90 because of its dense FP8
path; the expert-kernel capability is a separate, narrower statement.

That makes Marlin a possible **3090 expert-kernel comparison**, not a claim that
the entire SGLang MiniMax model can run on the 3090. Maya's helper currently calls
the imported packed DP4A implementation. The source also exposes integration traps:
encode scales must be inverted, activation clamp/beta must be placed relative to
dequantization correctly, and the Marlin preparation warns that unequal gate/up
global scales are reduced to the gate scale. Maya preserves separate gate/up/down
scales. A reuse must verify those scale relationships and preserve their effect,
rather than silently taking only one scale if they differ.

Newer FlashInfer has explicit SM120/121 NVFP4 entries in
[`fused_moe/api.py`](https://github.com/flashinfer-ai/flashinfer/blob/4e30ecdb0a3c1b20e129861f7c047f0ddf97b609/flashinfer/fused_moe/api.py).
`SM12xNvfp4Config` and its BF16-activation peer can share a prepared packed-weight
view, which is a useful memory-efficiency pattern. However, those entries support
default SwiGLU scalars or ReLU2, not MiniMax's clipped activation contract as-is.
The separate `B12xNvfp4Config` preparation starts from BF16 weights and requantizes
them. Neither is a drop-in replacement preserving the selected checkpoint and
arithmetic. Their native SM120 implementation is a donor to inspect if a specific
primary-kernel bottleneck justifies adaptation.

The latest per-pass evidence usually has the CPU finishing after the helper on
wide speculative windows. A faster helper kernel must therefore be judged by the
passes where it changes completion or a justified later scheduling change. A
helper-only kernel speedup cannot stand in for a generation gain.

## References with a larger adaptation boundary

- **MiniMax MSA/MegaKernels:** the inspected
  [MSA source](https://github.com/MiniMax-AI/MSA/blob/1e7bf76f26a5f53cfc2d4d297e1ec2e40955e431/README.md)
  documents SM100/SM103 kernels, with high-TPS MegaMoE and draft paths requiring
  TP4 and specific query widths. It also identifies a plugin dependency that is
  not publicly accessible. This does not provide an RTX 5090/SM120 implementation.
  Layouts, prepare/run lifetimes, selected-page dequantization and graph metadata
  updates are useful ideas; the advertised datacenter speed is not a Maya forecast.
- **KV compression:** MSA's Q8KV4 means FP8 queries with NVFP4 KV storage. It is
  separate from NVFP4 expert weights and host expert residency. Maya already has
  configurable quantized KV paths; a different format needs an actual memory or
  attention-cost reason, plus compatible indexer/attention semantics. Do not
  replace these paths solely because a newer recipe names NVFP4.
- **CPU/GPU cooperation:** public SGLang does contain a
  [KTransformers expert wrapper](https://github.com/sgl-project/sglang/blob/9d547e19ff519f0d1b964b0b1c014d98dc65caf4/python/sglang/srt/layers/moe/kt_ep_wrapper.py).
  It submits CPU expert work, runs GPU experts, then joins and combines. Thus a
  broad claim that SGLang lacks CPU expert execution would be incorrect. The
  inspected wrapper uses a configured expert-ID split and separately supplied CPU
  weights. It is not evidence for a NIM-validated MiniMax deployment with Maya's
  two different GPUs and adaptive residency. The basic overlap pattern is already
  present in Maya.
- **Distributed features:** TP all-reduce fusion, expert parallelism, DP attention,
  HiCache and disaggregation address other deployment conditions. HiCache is a KV
  cache tier, not the adaptive expert cache. Those settings are not substitutes
  for Maya's per-pass CPU/copy/helper balance.

## Research disposition

The most contained follow-up suggested by this audit is **DSpark context-K/V
batching and GPU-only graph replay**, followed by **joint target QKV/indexer
projection and preparation** if its current exposed cost warrants it. These use
existing weights and keep the cooperative policy boundaries. Kernel replacements
and device-side paging have larger arithmetic or ownership adaptation costs.
The two contained candidates address submission and dense GPU work; neither
directly removes the observed CPU expert work or demanded weight traffic.

Any later experiment must use the selected clock/submission configuration and
actual per-invocation work. Keep CPU, helper, copies, primary kernels and joins
aligned to the same forward/layer before aggregating; measure complete generation
as well. Do not infer savings from sums of overlapping timers, independent lane
averages, or faster kernels that finish before the existing critical dependency.
The central plan still decides whether and when a candidate is implemented.
