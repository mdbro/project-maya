"""Only decisions and historical demand. This module cannot import execution costs."""
from __future__ import annotations
import ctypes as C
import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]


def build_bridge(directory: Path | None = None) -> Path:
    source = Path(__file__).with_name('policy_bridge.cpp')
    headers = [ROOT/'src/mm3'/name for name in (
        'mm3_cooperative_policy.hpp', 'mm3_batch_staging.hpp', 'mm3_native_demand_units.hpp')]
    digest = hashlib.sha256(b''.join(p.read_bytes() for p in [source, *headers])).hexdigest()[:16]
    directory = directory or ROOT/'build-minimax/replay'
    directory.mkdir(parents=True, exist_ok=True)
    output = directory/f'policy-{digest}.so'
    if not output.exists():
        temporary = output.with_suffix('.tmp.so')
        subprocess.run(['c++', '-std=c++17', '-O2', '-shared', '-fPIC', '-ffp-contract=off',
                        '-I', str(ROOT/'src/mm3'), str(source), '-o', str(temporary)], check=True)
        temporary.replace(output)
    return output


class NativePolicy:
    """Calls the same C++ functions as Model::Impl, without CUDA or model weights."""
    def __init__(self, library: Path | None = None):
        self.path = library or build_bridge()
        self.lib = C.CDLL(str(self.path))
        if self.lib.mm3_replay_policy_abi() != 1:
            raise ValueError('unsupported policy bridge ABI')
        self.lib.mm3_replay_demand.argtypes = [C.c_float, C.c_int64, C.c_int64, C.c_float]
        self.lib.mm3_replay_demand.restype = C.c_float
        self.lib.mm3_replay_credit.argtypes = [C.c_int]
        self.lib.mm3_replay_credit.restype = C.c_float
        self.lib.mm3_replay_demand_decay.argtypes=[C.c_float,C.c_int64,C.c_int64,C.c_float,C.c_double]
        self.lib.mm3_replay_demand_decay.restype=C.c_float
        self.lib.mm3_replay_credit_decay.argtypes=[C.c_int,C.c_double]
        self.lib.mm3_replay_credit_decay.restype=C.c_float
        self.lib.mm3_replay_budget.argtypes = [C.c_double, C.c_int64, C.c_int64, C.c_double]
        self.lib.mm3_replay_budget.restype = C.c_double
        ip, dp, lp, fp = C.POINTER(C.c_int), C.POINTER(C.c_double), C.POINTER(C.c_int64), C.POINTER(C.c_float)
        self.lib.mm3_replay_stage.argtypes = [ip, ip, ip, C.c_int, C.c_int, dp, C.c_double, ip, dp]
        self.lib.mm3_replay_choose.argtypes = [C.c_int, lp, fp, ip, C.c_int, lp, fp, C.c_int, dp, lp, dp]
        self.lib.mm3_replay_choose.restype = None

    def stage(self, uses, cpu, gpu, helper, candidates, maximum, rows, prices, scale=1.0):
        owners, eligible = [0]*128, [0]*128
        for kind, ids in enumerate((cpu, gpu, helper), 1):
            for e in ids:
                if owners[e]: raise ValueError('overlapping service ownership')
                owners[e] = kind
        for e in candidates: eligible[e] = 1
        packed = sum((list(prices[k]) for k in ('cpu_extra_ms', 'gpu_extra_ms', 'helper_extra_ms', 'gpu_base_ms')), [])
        packed += [prices[k] for k in ('cpu_base_ms', 'helper_base_ms', 'h2d_ms', 'd2h_ms')]
        if len(uses) != 128 or len(packed) != 72: raise ValueError('invalid policy prices/uses')
        selected, predicted = (C.c_int*128)(), (C.c_double*5)()
        n = self.lib.mm3_replay_stage((C.c_int*128)(*uses), (C.c_int*128)(*owners),
            (C.c_int*128)(*eligible), maximum, rows, (C.c_double*72)(*packed), scale, selected, predicted)
        if n < 0: raise ValueError('production staging planner rejected input')
        return list(selected)[:n], list(predicted)

    def choose(self, kind, incoming, victims, prices):
        ni, nv = len(incoming), len(victims)
        out, value = (C.c_int64*3)(), C.c_double()
        self.lib.mm3_replay_choose(kind == 'helper',
            (C.c_int64*ni)(*(c[0] for c in incoming)), (C.c_float*ni)(*(c[1] for c in incoming)),
            (C.c_int*ni)(*(c[2] for c in incoming)), ni,
            (C.c_int64*nv)(*(c[0] for c in victims)), (C.c_float*nv)(*(c[1] for c in victims)), nv,
            (C.c_double*5)(*prices), out, C.byref(value))
        return [*out, value.value]


@dataclass
class Demand:
    bridge: NativePolicy
    row: int = 0
    block_rows: int = 1
    # Store lazy float32 scores exactly as production; never credit executed IDs.
    scores: dict[int, tuple[float, int]] = field(default_factory=dict)
    patterns: dict[int, list[tuple[int, tuple[int, ...]]]] = field(default_factory=dict)
    decay: float = .95
    pattern_limit: int = 16

    def score(self, key, credit=0.0):
        value, last = self.scores.get(key, (0.0, 0))
        value = self.bridge.lib.mm3_replay_demand_decay(value,last,self.row,credit,self.decay)
        self.scores[key] = value, self.row
        return value

    def observe(self, layer, original, rows, generation):
        if not generation: return
        if len(original) != rows*4: raise ValueError('incomplete evaluated original rows')
        credit = self.bridge.lib.mm3_replay_credit_decay(rows,self.decay)
        for e in original: self.score(layer*1024+e, credit)
        history = self.patterns.setdefault(layer, [])
        history.append((rows, tuple(original)))
        del history[:-self.pattern_limit]


@dataclass(frozen=True)
class PolicyConfig:
    cpu_scale: float = 1.3
    horizon: int = 20
    margin_ms: float = 0.0
    refill_budget: int = 8
    stage_limit: int = 4
    helper_live: bool = True
    extra_refill: bool = True
    decay: float = .95
    admission: str = 'incumbent'

    def __post_init__(self):
        if self.admission!='incumbent':raise ValueError('admission implementation has not passed the required validation gates')
        if (not math.isfinite(self.cpu_scale) or not .25 <= self.cpu_scale <= 4 or not 1 <= self.horizon <= 1000 or
            not math.isfinite(self.margin_ms) or self.margin_ms < 0 or self.refill_budget < 0 or
            self.refill_budget > 32 or not 0 <= self.stage_limit <= 4 or not 0 < self.decay < 1):
            raise ValueError('invalid policy configuration')
