"""serve/env_switches.py - every environment switch maya-mm3, this server and their launch path read (checked by
hand on 2026-11-12), the same list as the engine's src/mm3/mm3_env_switches.hpp (serve/test_env_switches.py keeps
them in step).

    report(environ)      [(name, value), ...] the MM3_, MAYA_ and STRATA_ variables set, and [name, ...] the unknown
    known(name)          a listed name, or one the launch path makes from one (both names of a setting)
    freeze(environ)      read the switches once, at the server's start; get() then answers from that reading
    get(name, default)   a switch's value: the reading freeze() took, or the environment before it (the tests);
                         MAYA_X and STRATA_X are one setting, either name asked for answers for both (the asked one
                         first): upstream's code reads MAYA_X since the sync of 2026-11-13 (synced through c5842c5)

A listed name is known; so are the names the launch path makes from one: MAYA_<name> of an engine switch read through
mm3::env (MAYA_ALIAS), MAYA_X of a listed STRATA_X and STRATA_X of a known MAYA_X (maya_names.alias_env and
export_env give a setting both names). The engine inherits this server's environment, so both check the whole list,
which also holds the other engines' switches this server names or sets (OTHER_ENGINES).
"""
from __future__ import annotations

import os

# maya-mm3 (src/mm3, src/ds4 and src/kernels/cpu as linked into it, and the libraries it links)
ENGINE = frozenset((
    "CUBLAS_WORKSPACE_CONFIG",
    "CUDA_DEVICE_ORDER",
    "CUDA_VISIBLE_DEVICES",
    "DS4_IQ1M_PDEP",
    "DS4_IQ1_PDEP",
    "DS4_IQ512",
    "DS4_IQ_AHEAD",
    "DS4_IQ_AVX2",
    "DS4_IQ_GATHER",
    "DS4_IQ_OWN",
    "DS4_IQ_PERSIST",
    "DS4_IQ_RP",
    "DS4_IQ_VERBOSE",
    "DS4_LT",
    "DS4_LT_VERBOSE",
    "DS4_MX_PF",
    "DS4_MX_TIER",
    "DS4_PF_ALL",
    "EXL3_WMMA",
    "EXL3_WMMA_ROWS",
    "GGML_CPU_DISABLE_FUSION",
    "GGML_META_DEBUG",
    "GGML_NO_BACKTRACE",
    "GGML_NO_IQ_PANEL",
    "GGML_SCHED_DEBUG",
    "GGML_SCHED_DEBUG_REALLOC",
    "MAYA_ABLATE_CAPTURE",   # src/core/ablate_capture.hpp, --ablate's read-out (upstream c5842c5)
    "MAYA_DS4_EXPERT_MMQ_MIN",   # src/ds4/ds4_iq.cu (upstream c5842c5)
    "MAYA_EXL3_AVX2_SHUF",   # src/kernels/cpu/exl3_cpu.cpp (upstream c5842c5)
    "MAYA_GLM_LANE_ROW4",   # src/kernels/cpu/glm_lane_row4.cpp (glmenv::env_flag), linked through native_expert.cpp (upstream c5842c5)
    "MAYA_IQ2XXS_1T",   # src/kernels/cpu/iq_avx2.cpp (upstream c5842c5)
    "MAYA_IQ2_VAR",   # src/kernels/cpu/iq_avx2.cpp (upstream c5842c5)
    "MAYA_IQ_PF_ROWS",   # src/kernels/cpu/iq_avx2.cpp (upstream c5842c5)
    "MAYA_MM3_ATTN_WIDE",
    "MAYA_MM3_DSPARK_FC16",
    "MAYA_MM3_MSA_TOK",
    "MAYA_MM3_MSA_WIDE",
    "MAYA_MM3_MV_COLS",
    "MAYA_MM3_MXC",
    "MAYA_MM3_PROF",
    "MAYA_MM3_TIMER_MS",
    "MAYA_MM3_TP_WINDOWS",
    "MAYA_MM3_WIN_SUBMIT",
    "MM3_ATTN_SPLIT",
    "MM3_CALIBRATION_REPEATS",
    "MM3_CAPTURE_CONTROL",
    "MM3_CONF",
    "MM3_CUBLAS_DETERMINISTIC",
    "MM3_DUMP_HS",
    "MM3_DUMP_KEPT",
    "MM3_EARLY_COPY_VERIFY",
    "MM3_EARLY_SAMPLE",
    "MM3_EXL3_BLAS_ROWS",
    "MM3_EXL3_WM70",
    "MM3_FULL_LOGITS",
    "MM3_GRAPH",
    "MM3_GV",
    "MM3_HELPER_PASS_TOKENS",
    "MM3_HOP_EVENTS",
    "MM3_HOST_BANK_EAGER",
    "MM3_HOST_FREE",
    "MM3_IQ_Q8K",
    "MM3_LANE_FREE",
    "MM3_LANE_OLD",
    "MM3_LANE_PROF",
    "MM3_LANE_SPLIT",
    "MM3_LOAD_DROP_PAGES",
    "MM3_LOOKUP_TRUST",   # src/mm3/mm3_main.cpp, the prompt lookup's trusted rounds (upstream c5842c5)
    "MM3_LT_STORE",
    "MM3_MISS_STATS",
    "MM3_MSA_OLD",
    "MM3_NSYS",
    "MM3_PIPE",
    "MM3_PREDICT_DIAG",
    "MM3_PREPARE_EVENTS",
    "MM3_PROF",
    "MM3_PROMOTE",
    "MM3_PROMPT_CPU",
    "MM3_PROMPT_HASH",
    "MM3_PROMPT_MASS",
    "MM3_PROMPT_TRACE",
    "MM3_PROMPT_U",
    "MM3_REPLAY_CAPTURE",
    "MM3_RESIDENCY_TRACE",
    "MM3_RESIDENCY_TRACE_ENABLE",
    "MM3_ROUTE_LAB_DUMP",
    "MM3_ROUTE_LAB_DUMP_ROWS",
    "MM3_SAMPLED_DRAFTS",
    "MM3_SERVE_MONITORING",
    "MM3_SIDE",
    "MM3_SLOTS_FIT",
    "MM3_SOURCE_COMMIT",
    "MM3_SPLIT_OLD",
    "MM3_STATE_TEST",
    "MM3_STRESS",
    "MM3_STRESS_T",
    "MM3_TIMING",
    "MM3_TOKEN_RECORDS_DIR",
    "MM3_TOPK",
    "MM3_TP_EXTRA",
    "MM3_TP_HOST_ROUTER",
    "MM3_TP_OLD",
    "MM3_TP_PIPE",
    "MM3_TP_PROF",
    "MM3_TP_TRACE",
    "MM3_VISION_LEND",
    "MM3_WARM_SAMPLE",
    "MM3_WIN_SUBMIT",
    "NVTX_INJECTION64_PATH",
    "STRATA_COMMIT_GROWTH",   # (include/strata/platform/commit.hpp: Windows only)
    "STRATA_CPU_YMM",
    "STRATA_DS4_EXPERT_MMQ",
    "STRATA_DS4_IQ_PERSIST",
    "STRATA_DS4_NO_LIGHT",
    "STRATA_DS4_NO_TC",
    "STRATA_DS4_PREFILL_FUSED",
    "STRATA_FORCE_AVX2",
    "STRATA_GLM_CPU_1T",
    "STRATA_GLM_CPU_MT",
    "STRATA_IQ_GATHER",
    "STRATA_IQ_PREFETCH",
    "STRATA_NO_IQ256",
    "STRATA_NO_IQ4NL",
    "STRATA_NO_IQ512",
    "STRATA_POOL_SPIN_US",
))
# ... of which mm3::env reads MAYA_<name> first
MAYA_ALIAS = frozenset((
    "MM3_ATTN_SPLIT",
    "MM3_CONF",
    "MM3_DUMP_HS",
    "MM3_DUMP_KEPT",
    "MM3_EXL3_BLAS_ROWS",
    "MM3_EXL3_WM70",
    "MM3_GRAPH",
    "MM3_GV",
    "MM3_HOP_EVENTS",
    "MM3_HOST_FREE",
    "MM3_IQ_Q8K",
    "MM3_LANE_FREE",
    "MM3_LANE_OLD",
    "MM3_LANE_PROF",
    "MM3_LANE_SPLIT",
    "MM3_LOAD_DROP_PAGES",
    "MM3_LOOKUP_TRUST",
    "MM3_LT_STORE",
    "MM3_MISS_STATS",
    "MM3_NSYS",
    "MM3_PIPE",
    "MM3_PROMOTE",
    "MM3_PROMPT_HASH",
    "MM3_PROMPT_MASS",
    "MM3_PROMPT_TRACE",
    "MM3_PROMPT_U",
    "MM3_REPLAY_CAPTURE",
    "MM3_RESIDENCY_TRACE",
    "MM3_RESIDENCY_TRACE_ENABLE",
    "MM3_SIDE",
    "MM3_SLOTS_FIT",
    "MM3_SPLIT_OLD",
    "MM3_STRESS",
    "MM3_STRESS_T",
    "MM3_TIMING",
    "MM3_TOPK",
    "MM3_TP_EXTRA",
    "MM3_TP_PROF",
    "MM3_TP_TRACE",
    "MM3_VISION_LEND",
))
# the server (serve/) and the config keys it reads
SERVER = frozenset((
    "MAYA_EXPERT_RAM_GB",
    "MAYA_GLM_KV_RESIDENT",
    "MAYA_LOOP_GUARD",
    "MAYA_LOOP_GUARD_COLLAPSE_MAX",
    "MAYA_MOCK_TOK_S",
    "MAYA_MODEL_MANAGEMENT",
    "MAYA_PACK_TEMPLATE",
    "MAYA_PROMPT_MEMO",   # prompt preparation by increment (upstream c5842c5)
    "MAYA_PULL_COMMAND",   # upstream c5842c5
    "MAYA_QUEUE_PING_S",
    "MAYA_RESTART_ON_UPDATE",
    "MAYA_SAVINGS_PATH",   # serve/savings.py (upstream c5842c5)
    "MAYA_TURN_TRACE",   # upstream c5842c5
    "MAYA_UPDATE_CHECK",
    "STRATA_API_KEY",
    "STRATA_DEBUG",
    "STRATA_DRAIN_S",
    "STRATA_DS4_KEEP_THINKING",
    "STRATA_DS4_RESTORE_POINTS",
    "STRATA_DS4_RESTORE_TAIL",
    "STRATA_ENGINE_QUIT_S",
    "STRATA_ENGINE_STALL_S",
    "STRATA_ENGINE_WRAP_S",
    "STRATA_FAIR_SLICE_S",
    "STRATA_GLM_RESPONSES_ASSEMBLY_BYTES",
    "STRATA_GLM_RESPONSES_CACHE_BYTES",
    "STRATA_GLM_RESPONSES_CACHE_ENTRIES",
    "STRATA_GLM_RESPONSES_CACHE_TTL_S",
    "STRATA_GLM_RESPONSES_ENTRY_BYTES",
    "STRATA_GLM_RESPONSES_EVENT_BYTES",
    "STRATA_GLM_RESPONSES_EVENT_COUNT",
    "STRATA_GLM_RESPONSES_JOB_ENTRIES",
    "STRATA_GLM_RESPONSES_JOB_INPUT_BYTES",
    "STRATA_GLM_RESPONSES_JOB_OWNER_ENTRIES",
    "STRATA_GLM_RESPONSES_JOB_TIMEOUT_S",
    "STRATA_GLM_RESPONSES_JOB_TTL_S",
    "STRATA_GLM_RESPONSES_JOB_WORKERS",
    "STRATA_GLM_RESPONSES_OWNER_BYTES",
    "STRATA_GLM_RESPONSES_OWNER_ENTRIES",
    "STRATA_GLM_RESPONSES_STREAM_OWNER_READERS",
    "STRATA_GLM_RESPONSES_STREAM_READERS",
    "STRATA_GLM_RESPONSES_STREAM_TIMEOUT_S",
    "STRATA_GLM_THINK_RESERVE",
    "STRATA_HTTP_BACKLOG",
    "STRATA_LOOP_GUARD",
    "STRATA_MAX_BODY_MIB",
    "STRATA_MOCK_RELOAD_S",
    "STRATA_MODEL_MANAGEMENT",
    "STRATA_REQUEST_QUIET_S",
    "STRATA_REQUEST_TIMEOUT_S",
    "STRATA_SERVER_LOG",
    "STRATA_VIDEO_MAX_FRAMES",
))
# the launch path (maya.py: MAYA_SH when it runs through maya.sh; MAYA_MM3_MIN_CC, its setup's test knob)
LAUNCH = frozenset(("MAYA_SH", "MAYA_MM3_MIN_CC"))
# other engines' switches the server names or sets (GLM: serve/server.py's cpu_threads, miss_skip_w and miss_skip_from
# keys, its kept conversations, the vision encoder's lent memory; DeepSeek: its kept conversation; serve/fake_multi_engine.py's
# GLM prompt pieces; since the upstream sync of 2026-11-13, the GLM-5.3 and DeepSeek settings serve/models.py,
# glm_dsa_tune.py and glm_dsa_product.py name or write; since the sync through 6651b5b (v1.1.63), the two-GPU settings
# serve/glm_link.py and serve/mixed_gpus.py name or set for GLM and DeepSeek on an AMD GPU beside an NVIDIA card), so one
# exported for another engine does not refuse a MiniMax start
OTHER_ENGINES = frozenset((
    "MAYA_DS4_DSPARK_CUM",
    "MAYA_DS4_RAM_SKIP_W",
    "MAYA_DS4_RESIDENT_TOPK",
    "MAYA_GLM_CPU_CAL",
    "MAYA_GLM_CPU_SPLIT",
    "MAYA_GLM_DFLASH_MIN",
    "MAYA_GLM_EXPERT_HELPER",
    "MAYA_GLM_HELPER_EXE",
    "MAYA_GLM_IDX_FP16",
    "MAYA_GLM_KV_INT8",
    "MAYA_GLM_MTP_ADAPT",
    "MAYA_GLM_MTP_DECODE",
    "MAYA_GLM_MTP_FIXED",
    "MAYA_GLM_MTP_KEEP",
    "MAYA_GLM_NGRAM_DRAFT",
    "MAYA_GLM_PREFETCH_AT",
    "MAYA_GLM_PREFETCH_BLOCKS",
    "MAYA_GLM_PREFETCH_N",
    "MAYA_GLM_PREFETCH_RANK",
    "MAYA_GLM_PREFILL_MB",
    "MAYA_GLM_PREFILL_SUB",
    "MAYA_GLM_PROMOTE_MIN",
    "MAYA_GLM_RESERVE_MB",
    "MAYA_GLM_RESIDENT_TOPK",
    "MAYA_GLM_ROUTE_AFFINITY",
    "MAYA_GLM_SPLIT",
    "MAYA_GLM_TRUNK_KEEP",
    "MAYA_GLM_VRAM_EVICT",
    "MAYA_HIPBLASLT_TUNING",
    "MAYA_MIXED_FRONT",
    "STRATA_DS4_SLOT_KEEP",
    "STRATA_GLM_CPU_LANE_MAX",
    "STRATA_GLM_MISS_SKIP_FROM",
    "STRATA_GLM_MISS_SKIP_W",
    "STRATA_GLM_PREFILL_INTERLEAVE",
    "STRATA_GLM_SLOT_KEEP",
    "STRATA_GLM_VISION_LEND_MB",
))
SWITCHES = ENGINE | SERVER | LAUNCH | OTHER_ENGINES
PREFIXES = ("MM3_", "MAYA_", "STRATA_")


def known(name: str) -> bool:
    """A listed switch, or a name the launch path makes from one."""
    if name in SWITCHES:
        return True
    if name.startswith("MAYA_"):
        rest = name[5:]
        return rest in MAYA_ALIAS or "STRATA_" + rest in SWITCHES
    if name.startswith("STRATA_"):
        return known("MAYA_" + name[7:])
    return False


def report(environ=None) -> tuple[list[tuple[str, str]], list[str]]:
    """([(name, value), ...] every MM3_ / MAYA_ / STRATA_ variable set, sorted; [name, ...] those not known)."""
    env = os.environ if environ is None else environ
    found = sorted((k, str(v)) for k, v in env.items() if k.startswith(PREFIXES))
    return found, [k for k, _ in found if not known(k)]


_frozen: dict | None = None


def freeze(environ=None) -> None:
    """The switches as they are now, read once: get() answers from this from now on."""
    global _frozen
    env = os.environ if environ is None else environ
    _frozen = {k: str(v) for k, v in env.items() if k.startswith(PREFIXES) or k in SWITCHES}


def frozen() -> bool:
    return _frozen is not None


def other_name(name: str) -> str | None:
    """The setting's other name: STRATA_X for MAYA_X and MAYA_X for STRATA_X (None for an MM3_ or other name)."""
    if name.startswith("MAYA_"):
        return "STRATA_" + name[5:]
    if name.startswith("STRATA_"):
        return "MAYA_" + name[7:]
    return None


def get(name: str, default=None):
    """A switch's value as freeze() read it at the server's start; before freeze() (the tests, a library use), the
    environment as it is now.  By either name of the setting, the one asked for first (other_name)."""
    src = _frozen if _frozen is not None else os.environ
    v = src.get(name)
    if v is None and other_name(name) is not None:
        v = src.get(other_name(name))
    return default if v is None else v
