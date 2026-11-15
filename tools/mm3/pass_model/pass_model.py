"""The bandwidth-aware pass model, the Python twin of src/mm3/mm3_pass_model.hpp (same terms, same arithmetic order).

A layer pass of the cooperative path, from the plan's end (t = 0, ms). Each lane's finish comes from its jobs' bytes
and measured rates, plus fixed start and chain costs; the pass finishes when the 5090 is done and the host has
published the other lanes' results.

  copies    the 5090's uploads, one record each, in a chain on the copy engine from copy_start; alone at
            copy_gbps[path] (path 0: demand copies in windows of 2+ rows, behind the pending exchanges' copies, one
            record each; path 1: one-row passes, the early copy)
  cpu       the host's CPU lane from cpu_start[rows]: cpu_fixed + experts x record / cpu_gbps + extra columns x
            cpu_column (ms of work alone). Copies and the CPU lane read host memory: while both run, each progresses at
            its share of its own rate (copy_share, cpu_share); the one left runs alone again.
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
    cpu_share: float = 0.6
    copy_share: float = 0.6
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
    exchanges: int = 0         # the 5090's pending exchanges (their copies run ahead of a window's demand copies)
    ready: int = 0             # experts the 5090 reads from staging frames filled by an earlier copy
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


def shared_ends(cpu_begin: float, cpu_work: float, copy_begin: float, copy_work: float, cpu_share: float,
                copy_share: float) -> tuple[float, float]:
    """The CPU lane's and the copy chain's ends (ms) when each needs `work` ms alone and both progress at their shares
    while the other runs. A flow with no work ends at its begin."""
    if cpu_work <= 0 or copy_work <= 0:
        return (cpu_begin + max(cpu_work, 0.0), copy_begin + max(copy_work, 0.0))
    # (the earlier one runs alone until the later one starts)
    if copy_begin <= cpu_begin:
        first_work, first_begin, first_share, second_work, second_begin, second_share = copy_work, copy_begin, copy_share, cpu_work, cpu_begin, cpu_share
    else:
        first_work, first_begin, first_share, second_work, second_begin, second_share = cpu_work, cpu_begin, cpu_share, copy_work, copy_begin, copy_share
    alone = second_begin - first_begin
    if first_work <= alone:
        first_end, second_end = first_begin + first_work, second_begin + second_work
    else:
        first_left = first_work - alone
        t_first, t_second = first_left / first_share, second_work / second_share
        if t_first <= t_second:
            first_end = second_begin + t_first
            second_end = first_end + (second_work - second_share * t_first)
        else:
            second_end = second_begin + t_second
            first_end = second_end + (first_left - first_share * t_second)
    if copy_begin <= cpu_begin:
        return second_end, first_end
    return first_end, second_end


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
    cpu_work = (r.cpu_fixed + load.cpu_experts * mb / r.cpu_gbps + max(0, load.cpu_columns - load.cpu_experts) * r.cpu_column) if load.cpu_experts else 0.0
    chain = load.copies + (load.exchanges if path == 0 else 0)
    copy_work = chain * mb / r.copy_gbps[path]
    cpu_end, copy_end = shared_ends(_row(r.cpu_start, load.rows), cpu_work, r.copy_start[path], copy_work, r.cpu_share, r.copy_share)
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


def load_of(row: dict, gpu_rounds_of: int = 4) -> Load:
    """A pass row (pass_data.PassRow as a dict) as the planner's view of the pass."""
    return Load(rows=row['rows'], cpu_experts=row['cpu_experts'], cpu_columns=row['cpu_columns'], copies=row['uploads'],
                exchanges=row['exchanges_pending'], ready=row['g_ready'],
                gpu_experts=row['g_experts'], gpu_picks=row['g_picks'], helper_experts=row['helper_experts'],
                helper_columns=row['helper_columns'], helper_lease=1 if row['helper_lease'] else 0,
                third_experts=row['third_experts'], third_columns=row['third_columns'], gpu_rounds_of=gpu_rounds_of)
