#!/usr/bin/env python3
"""Timestamped GPU operating-state samples during one measured arm.

Runs as its own process, started before the arm and stopped with SIGTERM after it.
Samples go to memory and reach disk only after the signal. Timestamps are
CLOCK_MONOTONIC, the clock of std::chrono::steady_clock, so samples align directly
with the engine's light records.

Per-sample NVML cost on this machine was measured at about 1 ms per GPU when idle.
Clock-event reasons cost 35 ms on the 3090 and PCIe throughput 21 ms on both, so
those stay off unless an arm explicitly asks for them at a low rate. The sampler
never changes a GPU setting; only read-only NVML queries are made.
"""
from __future__ import annotations
import argparse
import ctypes as C
import json
import os
import pathlib
import signal
import time

NVML = '/usr/lib/wsl/lib/libnvidia-ml.so.1'
CLOCK_GRAPHICS, CLOCK_MEMORY = 0, 2
FIELDS = ('t_ns', 'read_ns', 'gpu', 'pstate', 'graphics_mhz', 'memory_mhz', 'power_mw', 'util_gpu', 'util_memory', 'temperature_c')


class Utilization(C.Structure):
    _fields_ = [('gpu', C.c_uint), ('memory', C.c_uint)]


class Nvml:
    def __init__(self, path: str = NVML):
        self.lib = C.CDLL(path)
        status = self.lib.nvmlInit_v2()
        if status:
            raise RuntimeError(f'nvmlInit failed: {status}')

    def handle(self, index: int):
        handle = C.c_void_p()
        status = self.lib.nvmlDeviceGetHandleByIndex_v2(index, C.byref(handle))
        if status:
            raise RuntimeError(f'nvml handle {index}: {status}')
        return handle

    def sample(self, handle, gpu: int, out: list) -> None:
        begin = time.monotonic_ns()
        value, util = C.c_uint(), Utilization()
        lib = self.lib
        lib.nvmlDeviceGetPerformanceState(handle, C.byref(value)); pstate = value.value
        lib.nvmlDeviceGetClockInfo(handle, CLOCK_GRAPHICS, C.byref(value)); graphics = value.value
        lib.nvmlDeviceGetClockInfo(handle, CLOCK_MEMORY, C.byref(value)); memory = value.value
        lib.nvmlDeviceGetPowerUsage(handle, C.byref(value)); power = value.value
        lib.nvmlDeviceGetUtilizationRates(handle, C.byref(util))
        lib.nvmlDeviceGetTemperature(handle, 0, C.byref(value)); temperature = value.value
        out.append((begin, time.monotonic_ns() - begin, gpu, pstate, graphics, memory, power, util.gpu, util.memory, temperature))

    def reasons(self, handle) -> int:
        value = C.c_ulonglong()
        self.lib.nvmlDeviceGetCurrentClocksEventReasons(handle, C.byref(value))
        return int(value.value)


def run(output: pathlib.Path, gpus: list[int], hz: float, reasons_hz: float) -> int:
    stop = {'now': False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    nvml = Nvml()
    handles = [(g, nvml.handle(g)) for g in gpus]
    samples: list[tuple] = []
    reason_samples: list[tuple] = []
    period = 1e9 / hz
    next_at = time.monotonic_ns()
    next_reason = next_at
    began = next_at
    while not stop['now']:
        for gpu, handle in handles:
            nvml.sample(handle, gpu, samples)
        now = time.monotonic_ns()
        if reasons_hz > 0 and now >= next_reason:
            for gpu, handle in handles:
                a = time.monotonic_ns()
                value = nvml.reasons(handle)
                reason_samples.append((a, time.monotonic_ns() - a, gpu, value))
            next_reason = time.monotonic_ns() + int(1e9 / reasons_hz)
        next_at += int(period)
        delay = next_at - time.monotonic_ns()
        if delay > 0:
            time.sleep(delay * 1e-9)
        else:
            next_at = time.monotonic_ns()
    document = dict(schema=1, clock='CLOCK_MONOTONIC nanoseconds, same clock as the light recorder', hz=hz, reasons_hz=reasons_hz,
                    gpus=gpus, fields=list(FIELDS), reason_fields=['t_ns', 'read_ns', 'gpu', 'reasons'], began_ns=began,
                    ended_ns=time.monotonic_ns(), samples=samples, reasons=reason_samples)
    temporary = output.with_name(output.name + '.tmp')
    temporary.write_text(json.dumps(document))
    os.replace(temporary, output)
    return 0


def read(path) -> dict:
    """Returns the document with samples per GPU as lists of field dictionaries."""
    document = json.loads(pathlib.Path(path).read_text())
    per_gpu: dict[int, list[dict]] = {g: [] for g in document['gpus']}
    for row in document['samples']:
        item = dict(zip(document['fields'], row))
        per_gpu[item['gpu']].append(item)
    document['per_gpu'] = per_gpu
    return document


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', required=True, type=pathlib.Path)
    ap.add_argument('--gpus', default='0,1')
    ap.add_argument('--hz', type=float, default=20.0)
    ap.add_argument('--reasons-hz', type=float, default=0.0)
    a = ap.parse_args()
    return run(a.output, [int(x) for x in a.gpus.split(',')], a.hz, a.reasons_hz)


if __name__ == '__main__':
    raise SystemExit(main())
