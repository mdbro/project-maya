#!/usr/bin/env python3
"""Bounded phase-F shared HOST probe driver. Does not build or change serving.

Root supplies an idle GPU window after phase E and an explicitly built binary.
Every attempt is a fresh process with one private region. Raw failures, power
caps, memory inventory, commands, binary hash and all JSON receipts are retained.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def sizes(maximum: int) -> list[int]:
    return sorted({n for n in (64, 256, 1024, 4096) if n <= maximum} | {maximum})


def inventory(path: Path, env: dict[str, str]) -> dict:
    smi = shutil.which("nvidia-smi") or "/usr/lib/wsl/lib/nvidia-smi"
    command = [smi, "--query-gpu=uuid,name,pci.bus_id,power.limit,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=15, check=False)
    path.write_text(result.stdout + ("\nSTDERR\n" + result.stderr if result.stderr else ""))
    if result.returncode:
        raise RuntimeError(f"read-only GPU inventory failed ({result.returncode}); see {path}")
    devices = []
    for row in csv.reader(result.stdout.splitlines()):
        if len(row) != 7:
            raise RuntimeError(f"invalid GPU inventory row: {row!r}")
        uuid, name, bus, power, total, used, free = [field.strip() for field in row]
        device = {"uuid": uuid, "name": name, "pci_bus": bus, "power_w": float(power),
                  "memory_total_mib": float(total), "memory_used_mib": float(used), "memory_free_mib": float(free)}
        cap = next((cap for card, cap in (("5090", 425), ("3090", 390), ("4080", 320)) if card in name), None)
        device["standing_cap_w"] = cap
        if cap is not None and device["power_w"] > cap + 0.01:
            raise RuntimeError(f"{name} reports {power} W above the standing {cap} W cap; no setting was changed")
        devices.append(device)
    for card in ("5090", "3090", "4080"):
        if sum(card in device["name"] for device in devices) != 1:
            raise RuntimeError(f"expected exactly one {card} in GPU inventory")
    return {"time": stamp(), "command": command, "devices": devices}


def attempt(binary: Path, out: Path, backend: str, mib: int, env: dict[str, str], timeout: float) -> dict:
    command = [str(binary), "--backend", backend, "--mib", str(mib)]
    log = out / f"{backend}-{mib}MiB.jsonl"
    started = time.monotonic()
    record = {"backend": backend, "mib": mib, "command": command, "started": stamp(), "log": str(log)}
    # A process group includes the bounded Windows-memory provider subprocess.
    # On timeout kill the whole group, then keep its partial receipt unchanged.
    with log.open("wb") as output:
        process = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        record["pid"] = process.pid
        try:
            code = process.wait(timeout=timeout)
            record["timed_out"] = False
            record["termination_confirmed"] = True
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            try:
                code = process.wait(timeout=5)
                record["termination_confirmed"] = True
            except subprocess.TimeoutExpired:
                code = None
                record["termination_confirmed"] = False
            record["timed_out"] = True
    record.update(returncode=code, elapsed_s=time.monotonic() - started, finished=stamp())
    receipts, non_json = [], []
    for line in log.read_text(errors="replace").splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                receipts.append(value)
            else:
                non_json.append(line)
        except json.JSONDecodeError:
            non_json.append(line)
    finals = [row for row in receipts if row.get("kind") == "result"]
    record["result"] = finals[-1] if finals else None
    record["receipt_count"] = len(receipts)
    record["non_json_lines"] = non_json
    expected = record["result"] or {}
    record["passed"] = (code == 0 and not record["timed_out"] and expected.get("status") == "pass"
                         and expected.get("backend") == backend and expected.get("bytes") == mib * 1024 * 1024
                         and expected.get("cpu_and_three_gpus") is True and expected.get("full_range_checked") is True)
    write_json(out / f"{backend}-{mib}MiB-result.json", record)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-mib", required=True, type=int, help="Explicit region ceiling, 64..4096 MiB; one allocation at a time")
    parser.add_argument("--backend", choices=("auto", "vmm", "pinned"), default="auto")
    parser.add_argument("--cuda-visible-devices", help="Optional UUID list from the coordinator's validated serving inventory")
    parser.add_argument("--attempt-timeout", type=int, default=180)
    parser.add_argument("--total-timeout", type=int, default=600)
    args = parser.parse_args()
    if not 64 <= args.max_mib <= 4096:
        parser.error("--max-mib must be in 64..4096")
    if not 30 <= args.attempt_timeout <= 180 or not 30 <= args.total_timeout <= 900:
        parser.error("attempt timeout must be 30..180 s; total timeout 30..900 s")
    binary = args.binary.resolve()
    if not binary.is_file() or not os.access(binary, os.X_OK):
        parser.error("--binary must name an explicitly built executable")
    if args.out.exists() and any(args.out.iterdir()):
        parser.error("--out must be new or empty; prior probe evidence is never overwritten")
    args.out.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    manifest = {
        "schema": "mm3-shared-host-probe-v1", "started": stamp(), "binary": str(binary), "binary_sha256": sha256(binary),
        "max_mib": args.max_mib, "ladder_mib": sizes(args.max_mib), "requested_backend": args.backend,
        "attempt_timeout_s": args.attempt_timeout, "total_timeout_s": args.total_timeout,
        "environment": {k: v for k, v in sorted(env.items()) if k.startswith(("CUDA_", "MM3_", "MAYA_", "STRATA_"))},
        "engine_modified": False, "private_payload_only": True, "attempts": [], "status": "in_progress",
        "limitations": "Passing sizes qualify mapping and bytes only. They neither find an untested limit nor qualify mover concurrency, bandwidth or engine speed.",
    }
    source_root = Path(__file__).resolve().parents[2]
    manifest["source_files_at_launch"] = {
        str(path.relative_to(source_root)): sha256(path)
        for path in (Path(__file__).resolve(), source_root / "tools/mm3/bench/shared_host_probe.cu",
                     source_root / "src/mm3/mm3_native_host.cpp", source_root / "src/mm3/mm3_native_host.hpp")
        if path.is_file()
    }
    manifest["source_binary_binding"] = "Coordinator build manifest required; current source hashes alone do not prove binary provenance."
    write_json(args.out / "manifest.json", manifest)
    started = time.monotonic()
    try:
        manifest["inventory_before"] = inventory(args.out / "inventory-before.csv", env)
        backends = [args.backend] if args.backend != "auto" else ["vmm", "pinned"]
        fatal = False
        for backend in backends:
            complete = True
            for mib in sizes(args.max_mib):
                remaining = args.total_timeout - (time.monotonic() - started)
                if remaining < 30:
                    manifest["status"] = "time_budget_exhausted";fatal = True;complete = False;break
                print(f"Probe {backend} {mib} MiB", flush=True)
                row = attempt(binary, args.out, backend, mib, env, min(args.attempt_timeout, remaining))
                manifest["attempts"].append(row)
                write_json(args.out / "manifest.json", manifest)
                if row["passed"]:
                    continue
                complete = False
                # Only a setup refusal permits the explicit alternative. Any
                # payload/CUDA execution failure stops, keeping adverse evidence.
                if row["returncode"] != 3 or (row["result"] or {}).get("status") != "mapping_or_allocation_refused":
                    manifest["status"] = "failed";fatal = True
                break
            if fatal or complete:
                break
        passes = [row for row in manifest["attempts"] if row["passed"]]
        manifest["largest_pass_mib_by_backend"] = {
            backend: max([row["mib"] for row in passes if row["backend"] == backend], default=0)
            for backend in ("vmm", "pinned")}
        manifest["requested_ceiling_passed"] = any(row["mib"] == args.max_mib for row in passes)
        if not fatal:
            manifest["status"] = "bounded_pass" if passes else "no_supported_region"
        manifest["inventory_after"] = inventory(args.out / "inventory-after.csv", env)
        before = {row["uuid"]: row["power_w"] for row in manifest["inventory_before"]["devices"]}
        after = {row["uuid"]: row["power_w"] for row in manifest["inventory_after"]["devices"]}
        manifest["power_caps_unchanged"] = before == after
        if before != after:
            manifest["status"] = "external_power_state_changed"
        manifest["finished"] = stamp()
        write_json(args.out / "manifest.json", manifest)
        print(json.dumps({k: manifest[k] for k in ("status", "largest_pass_mib_by_backend", "power_caps_unchanged")}, sort_keys=True))
        return 0 if manifest["status"] == "bounded_pass" else 1
    except Exception as exc:
        manifest.update(status="driver_failed", error=str(exc), finished=stamp())
        write_json(args.out / "manifest.json", manifest)
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
