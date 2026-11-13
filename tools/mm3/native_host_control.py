#!/usr/bin/env python3
"""Allowlisted elevated host operations from WSL, through the MayaHostControl task.

The user registers the task once from an elevated prompt (tools/mm3/windows/register-maya-host-control.ps1). This tool
writes a request, starts the task without elevation and waits for its result. The task itself fixes every program and
argument and checks every number. Clock locks, power limits, the power plan and GPU scheduling are system settings:
change them only for a declared experiment, restore them afterwards, and keep the result files as receipts.

    native_host_control.py query
    native_host_control.py lock-graphics 3090 1950 1950        # GPU alias, min, max in MHz
    native_host_control.py lock-memory 3090 9501 9501
    native_host_control.py reset 3090                            # graphics and memory clocks
    native_host_control.py power-limit 3090 300 | restore-power-limit 3090
    native_host_control.py restore-power-limit all               # 5090 425 W, 3090 390 W, 4080 320 W (the user's caps)
    native_host_control.py mount-transfer
    native_host_control.py power-plan-query | power-plan high-performance | power-plan balanced
    native_host_control.py hags-query | hags on | hags off       # takes effect after a reboot
    native_host_control.py trace-start GPU | trace-start CPU | trace-stop | trace-cancel
    native_host_control.py monitor-start | monitor-stop          # 1 s host counters to CSV
    native_host_control.py host-config | gpu-details             # read only
    native_host_control.py vm-priority-query | vm-priority normal | vm-priority above-normal | vm-priority high
    native_host_control.py vm-affinity-query | vm-affinity all   # where the VM's vCPUs may run; `all` restores
    native_host_control.py vm-affinity lp0-15 | lp16-31 | even | lp0-15-even16-31 | even0-15-lp16-31
"""
from __future__ import annotations
import argparse
import json
import os
import pathlib
import subprocess
import sys
import time
import uuid

REQUESTS = pathlib.Path('/mnt/c/Users/mdwbr/maya-host-control')
TASK = 'MayaHostControl'
# (the 4080 since 2026-11-12: restore-power-limit all caps it at 320 W; the task's script must know it too -
#  tools/mm3/windows/maya-host-control.ps1, registered again with register-maya-host-control.ps1)
GPUS = {'5090': 'GPU-ea52aedc-8ee4-8e13-7ee7-c1bb774e12e9', '3090': 'GPU-ea9ac138-d20c-c5f7-38b0-6d18cccabbf1',
        '4080': 'GPU-6d5a59c8-62b5-855e-49fd-b18a1ddeb821'}


def request(actions: list[dict], timeout: float = 120.0) -> dict:
    ident = f'{time.strftime("%Y%m%dT%H%M%S")}-{uuid.uuid4().hex[:8]}'
    path = REQUESTS / 'request.json'
    if path.exists():
        raise RuntimeError(f'an earlier request is still pending: {path}')
    temporary = REQUESTS / f'.request-{ident}.tmp'
    temporary.write_text(json.dumps(dict(id=ident, actions=actions)))
    os.replace(temporary, path)
    started = subprocess.run(['schtasks.exe', '/Run', '/TN', TASK], capture_output=True, text=True)
    if started.returncode:
        path.unlink(missing_ok=True)
        raise RuntimeError(f'could not start {TASK}: {started.stdout.strip()} {started.stderr.strip()}')
    result = REQUESTS / f'result-{ident}.json'
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if result.exists():
            try:
                return json.loads(result.read_text(encoding='utf-8-sig'))
            except json.JSONDecodeError:
                pass  # still being written
        time.sleep(0.5)
    raise TimeoutError(f'no result from {TASK} after {timeout:.0f} s; see {REQUESTS / "log.jsonl"}')


def actions_for(a) -> list[dict]:
    if a.command in ('lock-graphics', 'lock-memory'):
        return [dict(op=a.command, gpu=GPUS[a.gpu], min=a.min, max=a.max)]
    if a.command == 'reset':
        gpus = GPUS.values() if a.gpu == 'all' else [GPUS[a.gpu]]
        return [dict(op=op, gpu=g) for g in gpus for op in ('reset-graphics', 'reset-memory')]
    if a.command == 'power-limit':
        return [dict(op='power-limit', gpu=GPUS[a.gpu], watts=a.watts)]
    if a.command == 'restore-power-limit':
        gpus = GPUS.values() if a.gpu == 'all' else [GPUS[a.gpu]]
        return [dict(op='restore-power-limit', gpu=g) for g in gpus]
    if a.command == 'power-plan':
        return [dict(op='power-plan', scheme=a.scheme)]
    if a.command == 'hags':
        return [dict(op='hags', mode=a.mode)]
    if a.command == 'trace-start':
        return [dict(op='trace-start', profile=a.profile)]
    if a.command == 'vm-priority':
        return [dict(op='vm-priority', level=a.level)]
    if a.command == 'vm-affinity':
        return [dict(op='vm-affinity', set=a.set)]
    return [dict(op=a.command)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='command', required=True)
    for name in ('query', 'mount-transfer', 'power-plan-query', 'hags-query', 'trace-stop', 'trace-cancel', 'monitor-start', 'monitor-stop',
                 'host-config', 'gpu-details', 'vm-priority-query', 'vm-affinity-query'):
        sub.add_parser(name)
    p = sub.add_parser('trace-start')
    p.add_argument('profile', choices=('GPU', 'CPU'))
    p = sub.add_parser('vm-priority')
    p.add_argument('level', choices=('normal', 'above-normal', 'high'))
    p = sub.add_parser('vm-affinity')
    p.add_argument('set', choices=('all', 'lp0-15', 'lp16-31', 'even', 'lp0-15-even16-31', 'even0-15-lp16-31'))
    for name in ('lock-graphics', 'lock-memory'):
        p = sub.add_parser(name)
        p.add_argument('gpu', choices=sorted(GPUS))
        p.add_argument('min', type=int)
        p.add_argument('max', type=int)
    for name in ('reset', 'restore-power-limit'):
        p = sub.add_parser(name)
        p.add_argument('gpu', choices=sorted(GPUS) + ['all'])
    p = sub.add_parser('power-limit')
    p.add_argument('gpu', choices=sorted(GPUS))
    p.add_argument('watts', type=int)
    p = sub.add_parser('power-plan')
    p.add_argument('scheme', choices=('balanced', 'high-performance'))
    p = sub.add_parser('hags')
    p.add_argument('mode', choices=('on', 'off'))
    a = ap.parse_args()
    result = request(actions_for(a))
    print(json.dumps(result, indent=1))
    return 0 if result.get('ok') else 1


if __name__ == '__main__':
    sys.exit(main())
