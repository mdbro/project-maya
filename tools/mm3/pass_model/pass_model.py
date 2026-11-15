"""The bandwidth-aware pass model, the Python twin of src/mm3/mm3_pass_model.hpp (same terms, same arithmetic order).

A layer pass of the cooperative path, from the plan's end (t = 0, ms). Each lane's finish comes from its jobs' bytes
and measured rates, plus fixed start and chain costs; the pass finishes when the 5090 is done and the host has
published the other lanes' results.

  host memory  three flows read or write it: the CPU lane, the 5090's uploads and the exchanges' downloads. Each has
            its own rate alone and draws host memory at a rate; while the flows running together would draw more than
            host_gbps, each is slowed in proportion (the measured ceiling shared by what is running).
  uploads   one record each, chained on the 5090's upload link from copy_start[path]: in windows (path 0) behind the
            unscheduled pending exchanges' uploads (exchange_up records each), alone at copy_gbps[0]; at one row (path 1)
            at copy_gbps[1], the link shared with the early copy
  early     at one row with the early copy on, its copies draw early_draw GB/s of host memory all pass long
  downloads the unscheduled pending exchanges' victims (exchange_down records each) at d2h_gbps from copy_start[path]
  cpu       the host's CPU lane from cpu_start[rows]: cpu_fixed + experts x record / cpu_gbps + extra columns x
            cpu_column of work alone
  5090      starts at the later of gpu_start[rows] (gpu_start_ready[rows] when it reads an expert staged by an earlier
            copy: at one row the early copy may still be landing) and the end of the copies it needs, then
            gpu_base[rows] + experts x record / gpu_gbps + picks x gpu_pick + extra rounds x gpu_round
            (rounds of 4 experts; gpu_round 0 when a window is one round)
  3090      helper_base[rows] + experts x record / helper_gbps + columns x helper_column (+ helper_lease while a
            lease or an exchange is pending)
  4080      third_base[rows] + experts x record / third_gbps + columns x third_column
  pass      the later of the 5090 and (the last host lane + publish[rows])

Rates are GB/s of record bytes (10^9 bytes a second); times are ms. Rows index 1..16 (index 0 unused)."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import math

ROWS = 17


def _row_table(value: float = 0.0) -> list[float]:
    return [value] * ROWS


@dataclass
class Rates:
    record_bytes: float = 31_850_752.0
    copy_gbps: list[float] = field(default_factory=lambda: [54.0, 25.0])   # (path 0 demand, 1 one-row early copy)
    copy_start: list[float] = field(default_factory=lambda: [0.0, 0.0])
    cpu_gbps: float = 56.5
    cpu_fixed: float = 0.0
    cpu_column: float = 0.02
    cpu_start: list[float] = field(default_factory=lambda: _row_table(0.04))
    host_gbps: float = 80.0
    d2h_gbps: float = 53.0
    exchange_up: float = 0.25
    exchange_down: float = 1.0
    early_draw: float = 0.0
    reuse: list[float] = field(default_factory=lambda: [0.19, 0.38])   # (uses from its frame within 8 forwards an upload buys, by path)
    gpu_gbps: float = 1679.0
    gpu_pick: float = 0.009
    gpu_round: float = 0.033
    gpu_start: list[float] = field(default_factory=lambda: _row_table(0.03))
    gpu_start_ready: list[float] = field(default_factory=lambda: _row_table(0.03))
    gpu_base: list[float] = field(default_factory=lambda: _row_table(0.05))
    helper_gbps: float = 900.0
    helper_column: float = 0.005
    helper_lease: float = 0.0
    helper_base: list[float] = field(default_factory=lambda: _row_table(0.1))
    third_gbps: float = 700.0
    third_column: float = 0.005
    third_base: list[float] = field(default_factory=lambda: _row_table(0.1))
    publish: list[float] = field(default_factory=lambda: _row_table(0.02))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(values: dict) -> 'Rates':
        return Rates(**values)

    def save(self, path) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=1)

    @staticmethod
    def load(path) -> 'Rates':
        with open(path) as f:
            return Rates.from_dict(json.load(f))


@dataclass
class Load:
    rows: int
    cpu_experts: int = 0
    cpu_columns: int = 0
    copies: int = 0            # records the 5090 waits for in this pass (uploaded now)
    exchanges: int = 0         # the 5090's pending exchanges whose copies run during the pass (exchange_schedule 0)
    ready: int = 0             # experts the 5090 reads from staging frames filled by an earlier copy
    early: int = 0             # 1: the early copy runs this pass (one row, early_copy on)
    gpu_experts: int = 0       # distinct experts the 5090 computes, including the uploaded ones
    gpu_picks: int = 0
    helper_experts: int = 0
    helper_columns: int = 0
    helper_lease: int = 0      # 1 while the 3090 holds a lease
    third_experts: int = 0
    third_columns: int = 0
    gpu_rounds_of: int = 4     # experts a round (4: expert_fast 0; 32: expert_fast 2, every window one round)


def _row(table: list[float], rows: int) -> float:
    return table[min(max(rows, 1), ROWS - 1)]


def host_flows(flows: list[tuple[float, float, float, float]], host_gbps: float) -> list[float]:
    """The ends (ms) of flows (begin ms, work MB, rate alone GB/s, host memory drawn alone GB/s) sharing host memory: while
    the running flows would draw more than host_gbps, each runs at its rate times host_gbps / their total draw. A flow
    with no work ends at its begin."""
    n = len(flows)
    ends = [0.0] * n
    left = [0.0] * n
    done = [False] * n
    for i, (begin, work, _, _) in enumerate(flows):
        left[i] = work
        if work <= 0:
            ends[i], done[i] = begin, True
    t = min((f[0] for i, f in enumerate(flows) if not done[i]), default=0.0)
    while not all(done):
        active = [i for i in range(n) if not done[i] and flows[i][0] <= t]
        later = [flows[i][0] for i in range(n) if not done[i] and flows[i][0] > t]
        if not active:
            t = min(later)
            continue
        draw = 0.0
        for i in active:
            draw += flows[i][3]
        scale = min(1.0, host_gbps / draw)
        step = math.inf
        for i in active:
            step = min(step, left[i] / (flows[i][2] * scale))
        if later:
            step = min(step, min(later) - t)
        for i in active:
            rate = flows[i][2] * scale
            if left[i] / rate <= step:
                left[i], done[i], ends[i] = 0.0, True, t + step
            else:
                left[i] -= rate * step
        t += step
    return ends


@dataclass
class Finish:
    cpu: float | None
    copies: float | None
    gpu: float
    helper: float | None
    third: float | None
    host: float | None        # the last host lane (cpu, 3090, 4080)
    passed: float             # the pass's finish


def predict(load: Load, r: Rates) -> Finish:
    path = 1 if load.rows == 1 else 0
    mb = r.record_bytes * 1e-6   # (record MB; MB / GB/s = ms)
    cpu_ms = (r.cpu_fixed + load.cpu_experts * mb / r.cpu_gbps + max(0, load.cpu_columns - load.cpu_experts) * r.cpu_column) if load.cpu_experts else 0.0
    up = load.copies * mb + (load.exchanges * r.exchange_up * mb if path == 0 else 0.0)
    down = load.exchanges * r.exchange_down * mb
    early = r.early_draw * 1000.0 if (load.early and path == 1) else 0.0   # (a flow that outlasts the pass)
    cpu_end, copy_end, _, _ = host_flows([(_row(r.cpu_start, load.rows), cpu_ms * r.cpu_gbps, r.cpu_gbps, r.cpu_gbps),
                                          (r.copy_start[path], up, r.copy_gbps[path], r.copy_gbps[path]),
                                          (r.copy_start[path], down, r.d2h_gbps, r.d2h_gbps),
                                          (r.copy_start[path], early, r.early_draw, r.early_draw)], r.host_gbps)
    chain = up > 0
    ready_start = _row(r.gpu_start_ready if load.ready else r.gpu_start, load.rows)
    start = max(ready_start, copy_end) if load.copies else ready_start
    rounds = max(0, math.ceil(load.gpu_experts / max(1, load.gpu_rounds_of)) - 1)
    gpu = start + _row(r.gpu_base, load.rows) + load.gpu_experts * mb / r.gpu_gbps + load.gpu_picks * r.gpu_pick + rounds * r.gpu_round
    helper = (_row(r.helper_base, load.rows) + load.helper_experts * mb / r.helper_gbps + load.helper_columns * r.helper_column +
              (r.helper_lease if load.helper_lease else 0.0)) if load.helper_experts else None
    third = (_row(r.third_base, load.rows) + load.third_experts * mb / r.third_gbps + load.third_columns * r.third_column) if load.third_experts else None
    lanes = [x for x in (cpu_end if load.cpu_experts else None, helper, third) if x is not None]
    host = max(lanes) if lanes else None
    passed = max(gpu, host + _row(r.publish, load.rows)) if host is not None else gpu
    return Finish(cpu=cpu_end if load.cpu_experts else None, copies=copy_end if chain else None, gpu=gpu, helper=helper,
                  third=third, host=host, passed=passed)


def plan_staging(base: Load, candidates: list[tuple[int, int]], max_new: int, r: Rates, use_value: float = 0.0):
    """mm3_pass_model.hpp plan_staging, step for step: (selected expert ids, the finish before, after, the credit)."""
    each = r.reuse[1 if base.rows == 1 else 0] * use_value
    cur = base
    before = after = predict(cur, r)
    objective = before.passed
    taken = [False] * len(candidates)
    selected = []
    for _ in range(max_new):
        best, best_f, best_obj, best_l = -1, after, objective, cur
        for i, (e, u) in enumerate(candidates):
            if taken[i]:
                continue
            t = Load(**{**cur.__dict__})
            t.cpu_experts -= 1; t.cpu_columns -= u; t.copies += 1; t.gpu_experts += 1; t.gpu_picks += u
            f = predict(t, r)
            value = f.passed - (len(selected) + 1) * each
            tol = 1e-12 * max(1.0, abs(objective), abs(best_obj), abs(value))
            if value < best_obj - tol:
                best, best_f, best_obj, best_l = i, f, value, t
        if best < 0:
            break
        taken[best] = True
        selected.append(candidates[best][0])
        cur, after, objective = best_l, best_f, best_obj
    return selected, before, after, len(selected) * each


def use_samples(after: Load, r: Rates) -> list[float]:
    """mm3_native_pass_model.inc's samples of the per-use value on a planned pass: the finish's gain if one remaining CPU
    expert (one column), or one copy, were on the 5090 already (0 where neither binds)."""
    base = predict(after, r).passed
    out = []
    if after.cpu_experts:
        x = Load(**{**after.__dict__})
        x.cpu_experts -= 1; x.cpu_columns -= 1; x.gpu_experts += 1; x.gpu_picks += 1
        out.append(max(0.0, base - predict(x, r).passed))
    if after.copies:
        x = Load(**{**after.__dict__})
        x.copies -= 1
        out.append(max(0.0, base - predict(x, r).passed))
    return out


def load_of(row: dict, gpu_rounds_of: int = 4) -> Load:
    """A pass row (pass_data.PassRow as a dict) as the planner's view of the pass. Its pending exchanges run during the
    pass in a drafting request (exchange_schedule 0, the built-in when drafting); a one-row request schedules them in
    idle time (mode 'one row')."""
    return Load(rows=row['rows'], cpu_experts=row['cpu_experts'], cpu_columns=row['cpu_columns'], copies=row['uploads'],
                exchanges=0 if row.get('mode') == 'one row' else row['exchanges_pending'], ready=row['g_ready'],
                early=1 if row['rows'] == 1 else 0,
                gpu_experts=row['g_experts'], gpu_picks=row['g_picks'], helper_experts=row['helper_experts'],
                helper_columns=row['helper_columns'], helper_lease=1 if row['helper_lease'] else 0,
                third_experts=row['third_experts'], third_columns=row['third_columns'], gpu_rounds_of=gpu_rounds_of)
