// src/mm3/mm3_env_switches.hpp - every environment switch maya-mm3, its server and their launch path read (the
// switches checked by hand on 2026-11-12: 148, with the launch path's names). maya-mm3 prints every MM3_, MAYA_ and
// STRATA_ variable set at its start and refuses to start on one that is not listed here (mm3_main.cpp). The server
// checks the same list for a MiniMax engine (serve/env_switches.py, kept in step: serve/test_env_switches.py).
//
// A listed name is known; so are the names the launch path makes from one: MAYA_<name> of a switch read through
// mm3::env (maya_alias), MAYA_X of a listed STRATA_X and STRATA_X of a known MAYA_X (the server's alias_env and
// export_env give a setting both names). The server's and the launch path's switches are listed too: the engine
// inherits the server's environment; so are the other engines' switches the server names or sets.
#pragma once

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <stdlib.h>
#else
extern char** environ;
#endif

namespace mm3::env_switches {

struct Switch {
    const char* name;
    bool maya_alias;   // (read through mm3::env: MAYA_<name> is read first)
};

inline constexpr Switch kSwitches[] = {
    // maya-mm3 (src/mm3, src/ds4 and src/kernels/cpu as linked into it, and the libraries it links)
    {"CUBLAS_WORKSPACE_CONFIG", false},
    {"CUDA_DEVICE_ORDER", false},
    {"CUDA_VISIBLE_DEVICES", false},
    {"DS4_IQ1M_PDEP", false},
    {"DS4_IQ1_PDEP", false},
    {"DS4_IQ512", false},
    {"DS4_IQ_AHEAD", false},
    {"DS4_IQ_AVX2", false},
    {"DS4_IQ_GATHER", false},
    {"DS4_IQ_OWN", false},
    {"DS4_IQ_PERSIST", false},
    {"DS4_IQ_RP", false},
    {"DS4_IQ_VERBOSE", false},
    {"DS4_LT", false},
    {"DS4_LT_VERBOSE", false},
    {"DS4_MX_PF", false},
    {"DS4_MX_TIER", false},
    {"DS4_PF_ALL", false},
    {"EXL3_WMMA", false},
    {"EXL3_WMMA_ROWS", false},
    {"GGML_CPU_DISABLE_FUSION", false},
    {"GGML_META_DEBUG", false},
    {"GGML_NO_BACKTRACE", false},
    {"GGML_NO_IQ_PANEL", false},
    {"GGML_SCHED_DEBUG", false},
    {"GGML_SCHED_DEBUG_REALLOC", false},
    {"MAYA_DS4_EXPERT_MMQ_MIN", false},   // src/ds4/ds4_iq.cu (upstream c5842c5)
    {"MAYA_EXL3_AVX2_SHUF", false},   // src/kernels/cpu/exl3_cpu.cpp (upstream c5842c5)
    {"MAYA_IQ2XXS_1T", false},   // src/kernels/cpu/iq_avx2.cpp (upstream c5842c5)
    {"MAYA_IQ2_VAR", false},   // src/kernels/cpu/iq_avx2.cpp (upstream c5842c5)
    {"MAYA_IQ_PF_ROWS", false},   // src/kernels/cpu/iq_avx2.cpp (upstream c5842c5)
    {"MAYA_MM3_ATTN_WIDE", false},
    {"MAYA_MM3_DSPARK_FC16", false},
    {"MAYA_MM3_MSA_TOK", false},
    {"MAYA_MM3_MSA_WIDE", false},
    {"MAYA_MM3_MV_COLS", false},
    {"MAYA_MM3_MXC", false},
    {"MAYA_MM3_PROF", false},
    {"MAYA_MM3_TIMER_MS", false},
    {"MAYA_MM3_TP_WINDOWS", false},
    {"MAYA_MM3_WIN_SUBMIT", false},
    {"MM3_ATTN_SPLIT", true},
    {"MM3_CALIBRATION_REPEATS", false},
    {"MM3_CAPTURE_CONTROL", false},
    {"MM3_CONF", true},
    {"MM3_CUBLAS_DETERMINISTIC", false},
    {"MM3_DUMP_HS", true},
    {"MM3_DUMP_KEPT", true},
    {"MM3_EARLY_COPY_VERIFY", false},
    {"MM3_EARLY_SAMPLE", false},
    {"MM3_EXL3_BLAS_ROWS", true},
    {"MM3_EXL3_WM70", true},
    {"MM3_FULL_LOGITS", false},
    {"MM3_GRAPH", true},
    {"MM3_GV", true},
    {"MM3_HELPER_PASS_TOKENS", false},
    {"MM3_HOP_EVENTS", true},
    {"MM3_HOST_BANK_EAGER", false},
    {"MM3_HOST_FREE", true},
    {"MM3_IQ_Q8K", true},
    {"MM3_LANE_FREE", true},
    {"MM3_LANE_OLD", true},
    {"MM3_LANE_PROF", true},
    {"MM3_LANE_SPLIT", true},
    {"MM3_LOAD_DROP_PAGES", true},
    {"MM3_LT_STORE", true},
    {"MM3_MISS_STATS", true},
    {"MM3_MSA_OLD", false},
    {"MM3_NSYS", true},
    {"MM3_PIPE", true},
    {"MM3_PREDICT_DIAG", false},
    {"MM3_PREPARE_EVENTS", false},
    {"MM3_PROF", false},
    {"MM3_PROMOTE", true},
    {"MM3_PROMPT_CPU", false},
    {"MM3_PROMPT_HASH", true},
    {"MM3_PROMPT_MASS", true},
    {"MM3_PROMPT_TRACE", true},
    {"MM3_PROMPT_U", true},
    {"MM3_REPLAY_CAPTURE", true},
    {"MM3_RESIDENCY_TRACE", true},
    {"MM3_RESIDENCY_TRACE_ENABLE", true},
    {"MM3_ROUTE_LAB_DUMP", false},
    {"MM3_ROUTE_LAB_DUMP_ROWS", false},
    {"MM3_SAMPLED_DRAFTS", false},
    {"MM3_SERVE_MONITORING", false},
    {"MM3_SIDE", true},
    {"MM3_SLOTS_FIT", true},
    {"MM3_SOURCE_COMMIT", false},
    {"MM3_SPLIT_OLD", true},
    {"MM3_STATE_TEST", false},
    {"MM3_STRESS", true},
    {"MM3_STRESS_T", true},
    {"MM3_TIMING", true},
    {"MM3_TOKEN_RECORDS_DIR", false},
    {"MM3_TOPK", true},
    {"MM3_TP_EXTRA", true},
    {"MM3_TP_HOST_ROUTER", false},
    {"MM3_TP_OLD", false},
    {"MM3_TP_PIPE", false},
    {"MM3_TP_PROF", true},
    {"MM3_TP_TRACE", true},
    {"MM3_VISION_LEND", true},
    {"MM3_WARM_SAMPLE", false},
    {"MM3_WIN_SUBMIT", false},
    {"NVTX_INJECTION64_PATH", false},
    {"STRATA_COMMIT_GROWTH", false},   // (include/strata/platform/commit.hpp: Windows only)
    {"STRATA_CPU_YMM", false},
    {"STRATA_DS4_EXPERT_MMQ", false},
    {"STRATA_DS4_IQ_PERSIST", false},
    {"STRATA_DS4_NO_LIGHT", false},
    {"STRATA_DS4_NO_TC", false},
    {"STRATA_DS4_PREFILL_FUSED", false},
    {"STRATA_FORCE_AVX2", false},
    {"STRATA_GLM_CPU_1T", false},
    {"STRATA_GLM_CPU_MT", false},
    {"STRATA_IQ_GATHER", false},
    {"STRATA_IQ_PREFETCH", false},
    {"STRATA_NO_IQ256", false},
    {"STRATA_NO_IQ4NL", false},
    {"STRATA_NO_IQ512", false},
    {"STRATA_POOL_SPIN_US", false},
    // the server (serve/) and the config keys it reads
    {"MAYA_EXPERT_RAM_GB", false},
    {"MAYA_GLM_KV_RESIDENT", false},
    {"MAYA_LOOP_GUARD", false},
    {"MAYA_LOOP_GUARD_COLLAPSE_MAX", false},
    {"MAYA_MOCK_TOK_S", false},
    {"MAYA_MODEL_MANAGEMENT", false},
    {"MAYA_PACK_TEMPLATE", false},
    {"MAYA_QUEUE_PING_S", false},
    {"MAYA_RESTART_ON_UPDATE", false},
    {"MAYA_UPDATE_CHECK", false},
    {"STRATA_API_KEY", false},
    {"STRATA_DEBUG", false},
    {"STRATA_DRAIN_S", false},
    {"STRATA_DS4_KEEP_THINKING", false},
    {"STRATA_DS4_RESTORE_POINTS", false},
    {"STRATA_DS4_RESTORE_TAIL", false},
    {"STRATA_ENGINE_QUIT_S", false},
    {"STRATA_ENGINE_STALL_S", false},
    {"STRATA_ENGINE_WRAP_S", false},
    {"STRATA_FAIR_SLICE_S", false},
    {"STRATA_GLM_RESPONSES_ASSEMBLY_BYTES", false},
    {"STRATA_GLM_RESPONSES_CACHE_BYTES", false},
    {"STRATA_GLM_RESPONSES_CACHE_ENTRIES", false},
    {"STRATA_GLM_RESPONSES_CACHE_TTL_S", false},
    {"STRATA_GLM_RESPONSES_ENTRY_BYTES", false},
    {"STRATA_GLM_RESPONSES_EVENT_BYTES", false},
    {"STRATA_GLM_RESPONSES_EVENT_COUNT", false},
    {"STRATA_GLM_RESPONSES_JOB_ENTRIES", false},
    {"STRATA_GLM_RESPONSES_JOB_INPUT_BYTES", false},
    {"STRATA_GLM_RESPONSES_JOB_OWNER_ENTRIES", false},
    {"STRATA_GLM_RESPONSES_JOB_TIMEOUT_S", false},
    {"STRATA_GLM_RESPONSES_JOB_TTL_S", false},
    {"STRATA_GLM_RESPONSES_JOB_WORKERS", false},
    {"STRATA_GLM_RESPONSES_OWNER_BYTES", false},
    {"STRATA_GLM_RESPONSES_OWNER_ENTRIES", false},
    {"STRATA_GLM_RESPONSES_STREAM_OWNER_READERS", false},
    {"STRATA_GLM_RESPONSES_STREAM_READERS", false},
    {"STRATA_GLM_RESPONSES_STREAM_TIMEOUT_S", false},
    {"STRATA_GLM_THINK_RESERVE", false},
    {"STRATA_HTTP_BACKLOG", false},
    {"STRATA_LOOP_GUARD", false},
    {"STRATA_MAX_BODY_MIB", false},
    {"STRATA_MOCK_RELOAD_S", false},
    {"STRATA_MODEL_MANAGEMENT", false},
    {"STRATA_REQUEST_QUIET_S", false},
    {"STRATA_REQUEST_TIMEOUT_S", false},
    {"STRATA_SERVER_LOG", false},
    {"STRATA_VIDEO_MAX_FRAMES", false},
    // the launch path (maya.py: MAYA_SH when it runs through maya.sh; MAYA_MM3_MIN_CC, its setup's test knob)
    {"MAYA_SH", false},
    {"MAYA_MM3_MIN_CC", false},
    // other engines' switches the server names or sets (GLM: serve/server.py's cpu_threads, miss_skip_w and
    // miss_skip_from keys, its kept conversations, the vision encoder's lent memory; DeepSeek: its kept conversation;
    // serve/fake_multi_engine.py's GLM prompt pieces), so one exported for another engine does not refuse a MiniMax start
    {"STRATA_GLM_CPU_LANE_MAX", false},
    {"STRATA_GLM_MISS_SKIP_FROM", false},
    {"STRATA_GLM_MISS_SKIP_W", false},
    {"STRATA_GLM_PREFILL_INTERLEAVE", false},
    {"STRATA_GLM_SLOT_KEEP", false},
    {"STRATA_GLM_VISION_LEND_MB", false},
    {"STRATA_DS4_SLOT_KEEP", false},
};

inline const Switch* find(const std::string& name) {
    for (const Switch& s : kSwitches)
        if (name == s.name) return &s;
    return nullptr;
}
inline bool checked(const std::string& name) {   // (the prefixes the start check covers)
    return name.rfind("MM3_", 0) == 0 || name.rfind("MAYA_", 0) == 0 || name.rfind("STRATA_", 0) == 0;
}
inline bool known(const std::string& name) {
    if (find(name)) return true;
    if (name.rfind("MAYA_", 0) == 0) {
        const std::string x = name.substr(5);
        const Switch* s = find(x);
        return (s && s->maya_alias) || find("STRATA_" + x) != nullptr;
    }
    if (name.rfind("STRATA_", 0) == 0) return known("MAYA_" + name.substr(7));
    return false;
}

struct Scan {
    std::vector<std::string> set;       // NAME=VALUE, every checked variable that is set, in the environment's order
    std::vector<std::string> unknown;   // the names among them that are not known
};
// the process's environment as it is now (called once, at the start of main)
inline Scan scan() {
    Scan out;
#if defined(_WIN32)
    char** env = _environ;
#else
    char** env = environ;
#endif
    for (char** e = env; e && *e; ++e) {
        const std::string entry(*e);
        const size_t eq = entry.find('=');
        const std::string name = entry.substr(0, eq);
        if (!checked(name)) continue;
        out.set.push_back(entry);
        if (!known(name)) out.unknown.push_back(name);
    }
    return out;
}

}  // namespace mm3::env_switches
