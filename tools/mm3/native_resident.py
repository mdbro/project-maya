#!/usr/bin/env python3
"""Keep one engine up with its weights resident, and run tests against it without starting another.

The project's rule (the user, 2026-11-11): "minimize loading time by keeping weights resident when possible and
reloading only what we need to reload". On 2026-11-11 run scripts started and stopped an engine for every run, about
fourteen engine starts in a day. This tool is the other way of working:

    native_resident.py status                      is an engine up, from which files, for how long
    native_resident.py start [--config FILE]       start serving (default: docs/minimax-wsl/native-server.json) and leave it up
    native_resident.py adopt PID CONFIG            record an engine another script left up (same mutex name)
    native_resident.py needs-restart CONFIG        exit 0 if the engine that is up is CONFIG's (same binary, same start
                                                   options, same placement), 1 if a run on CONFIG needs an engine start
    native_resident.py run RUN_DIR [PHASE ...]     run RUN_DIR/protocol.json's phases (default: standalone check) against
                                                   the engine that is up, under the quiet flag; the engine stays up
    native_resident.py capture-control             the capture-control file the engine that is up watches: a protocol
                                                   run against it must name this file as its capture_control
    native_resident.py stop                        stop it and release the cards

An engine start is the launch until the engine is ready; a run is everything for one configuration. A run needs an
engine start only when `needs-restart` says so: another binary, or another start option (the drafter, the 5090's
arithmetic, the cards). Request settings, placements and snapshots never need one.

The engine holds the GPU mutex as `serving-resident` for as long as it is up, so nothing else takes the cards from
under it; `run` works inside that hold. Port 8097. Standard library only."""
from __future__ import annotations

import datetime
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

W = Path(__file__).resolve().parents[2]
PY = "/home/mdwbr/project-maya/.venv/bin/python"
LOCKS = Path("/mnt/wsl/transfer/minimax-wsl/runs/plan-completion-20261110/locks")
MUTEX, QUIET, NAME = LOCKS / "gpu.lock.d", LOCKS / "quiet.lock", "serving-resident"
HOME = Path("/mnt/wsl/transfer/minimax-wsl/runs/serving-resident")
STATE = HOME / "state.json"
PORT = 8097
CAPS = {"5090": 425.0, "3090": 390.0}        # never above these, by any route


def say(msg: str) -> None:
    print(f"{datetime.datetime.now():%H:%M:%S} {msg}", flush=True)


def alive(pid: int) -> bool:
    try:
        return Path(f"/proc/{pid}/stat").read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError):
        return False


def metrics() -> dict | None:
    try:
        return json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/metrics", timeout=5))
    except (OSError, ValueError):
        return None


def identity(config: dict) -> dict:
    """What decides whether two configurations are the same engine: the binary's bytes, the start options, the
    placement at start and the cards. Logs, working folders and request defaults are left out."""
    exe = Path(config["exe"])
    return {"exe_sha256": hashlib.sha256(exe.read_bytes()).hexdigest() if exe.exists() else None,
            "args": list(config.get("args", [])), "placement_order": config.get("placement_order"),
            "third_profile": config.get("third_profile"), "cards": (config.get("env") or {}).get("CUDA_VISIBLE_DEVICES")}


def started_at(pid: int) -> float:
    """When the process started, as a Unix time: its start in clock ticks after boot (field 22 of /proc/PID/stat)."""
    ticks = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19])
    boot = next(int(line.split()[1]) for line in Path("/proc/stat").read_text().splitlines() if line.startswith("btime"))
    return boot + ticks / os.sysconf("SC_CLK_TCK")


def owner() -> str | None:
    f = MUTEX / "owner"
    return f.read_text().strip() if f.exists() else None


def state() -> dict | None:
    """The recorded engine if it is really up: its process alive, its port answering, the mutex ours."""
    if not STATE.exists():
        return None
    s = json.loads(STATE.read_text())
    return s if alive(int(s["pid"])) and owner() == NAME and metrics() is not None else None


def power_ok() -> None:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,power.limit", "--format=csv,noheader,nounits"], text=True)
    for line in out.splitlines():
        name, limit = [x.strip() for x in line.split(",")]
        for card, cap in CAPS.items():
            if card in name and float(limit) > cap + 0.5:
                raise SystemExit(f"REFUSED: the {card} power limit is {limit} W, above {cap} W")


def cmd_status() -> int:
    s = state()
    if s is None:
        held = owner()
        say("no resident engine is up" + (f"; the GPU mutex is held by {held}" if held else "; the cards are free"))
        return 1
    up = time.time() - s["started"]
    say(f"resident engine up for {up / 60:.0f} min: pid {s['pid']}, port {PORT}, started from {s['config']} "
        f"(binary sha256 {str(s['identity']['exe_sha256'])[:12]}, engine start {s.get('ready_s', '?')} s)")
    return 0


def cmd_start(config_path: Path) -> int:
    if state() is not None:
        say("a resident engine is already up (status shows it); use needs-restart to see whether it is the one wanted")
        return 1
    try:
        MUTEX.mkdir()
        (MUTEX / "owner").write_text(NAME + "\n")
    except FileExistsError:
        raise SystemExit(f"REFUSED: the GPU mutex is held by {owner()}")
    server = None
    try:
        power_ok()
        used = [int(x) for x in subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()]
        if max(used) > 2000:
            raise SystemExit(f"REFUSED: a card already holds memory ({used} MiB)")
        HOME.mkdir(parents=True, exist_ok=True)
        config = json.loads(config_path.read_text())
        ident = identity(config)
        derived = dict(config, log=str(HOME / "engine.log"), monitoring=False)
        derived["env"] = dict(config.get("env") or {}, MM3_SERVE_MONITORING="0", STRATA_ENGINE_STALL_S="0", MM3_CAPTURE_CONTROL=str(HOME / "capture-control"))
        (HOME / "server.json").write_text(json.dumps(derived, indent=2) + "\n")
        env = os.environ.copy()
        for key in list(env):
            if key.startswith(("NSYS", "CUDA_INJECTION", "NVTX_INJECTION", "MM3_CAPTURE", "MM3_REPLAY", "MM3_STATE", "CUDA_VISIBLE")) or key == "LD_PRELOAD":
                env.pop(key)
        env.update(STRATA_ENGINE_QUIT_S="300")
        began = time.monotonic()
        slog = (HOME / "server.log").open("ab")
        server = subprocess.Popen([PY, "serve/server.py", "--engine", "strata", "--config", str(HOME / "server.json"), "--no-monitoring",
                                   "--host", "127.0.0.1", "--port", str(PORT), "--idle-unload", "0"], cwd=W, env=env, stdout=slog, stderr=subprocess.STDOUT, start_new_session=True)
        say(f"engine start (pid {server.pid}) from {config_path}")
        while True:
            m = metrics()
            if m and m.get("engine", {}).get("cooperative_cpu_scale_control") == 1:
                break
            if server.poll() is not None:
                raise SystemExit(f"the server exited with {server.returncode}; see {HOME / 'server.log'}")
            if time.monotonic() - began > 1500:
                raise SystemExit("not ready after 25 minutes")
            time.sleep(3)
        ready = time.monotonic() - began
        STATE.write_text(json.dumps({"pid": server.pid, "config": str(config_path), "identity": ident, "started": started_at(server.pid), "ready_s": round(ready),
                                     "capture_control": derived["env"]["MM3_CAPTURE_CONTROL"]}, indent=1) + "\n")
        say(f"ready in {ready:.0f} s; left up with its weights resident (stop: native_resident.py stop)")
        return 0
    except BaseException:
        if server is not None and server.poll() is None:
            os.kill(server.pid, signal.SIGTERM)
        if owner() == NAME:
            shutil.rmtree(MUTEX)
        raise


def cmd_adopt(pid: int, config_path: Path) -> int:
    if not alive(pid) or owner() != NAME or metrics() is None:
        raise SystemExit(f"REFUSED: pid {pid} is not an engine up on port {PORT} under the mutex name {NAME}")
    HOME.mkdir(parents=True, exist_ok=True)
    config = json.loads(config_path.read_text())
    STATE.write_text(json.dumps({"pid": pid, "config": str(config_path), "identity": identity(config), "started": started_at(pid), "ready_s": None, "adopted": True,
                                 "capture_control": (config.get("env") or {}).get("MM3_CAPTURE_CONTROL")}, indent=1) + "\n")
    say(f"adopted the engine pid {pid}, started from {config_path}")
    return 0


def cmd_needs_restart(config_path: Path) -> int:
    s = state()
    if s is None:
        say("needs an engine start: no resident engine is up")
        return 1
    want, have = identity(json.loads(config_path.read_text())), s["identity"]
    diffs = [k for k in want if want[k] != have.get(k)]
    if not diffs:
        say("no engine start needed: the engine that is up has this binary, these start options and this placement")
        return 0
    for k in diffs:
        if k == "args":
            a, b = have.get("args") or [], want["args"]
            say(f"  start options differ: up {' '.join(x for x in a if x not in b) or '(nothing extra)'} | wanted {' '.join(x for x in b if x not in a) or '(nothing extra)'}")
        else:
            say(f"  {k} differs: up {str(have.get(k))[:60]} | wanted {str(want[k])[:60]}")
    say("needs an engine start")
    return 1


def cmd_run(run_dir: Path, phases: list[str]) -> int:
    s = state()
    if s is None:
        raise SystemExit("REFUSED: no resident engine is up (native_resident.py start)")
    if QUIET.exists():
        raise SystemExit(f"REFUSED: a quiet flag is up ({QUIET.read_text().strip()})")
    protocol = run_dir / "protocol.json"
    if not protocol.exists():
        raise SystemExit(f"no {protocol}")
    named = json.loads(protocol.read_text()).get("capture_control")
    if s.get("capture_control") and named != s["capture_control"]:
        raise SystemExit(f"REFUSED: the protocol names the capture-control file {named}; the engine that is up watches {s['capture_control']} "
                         "(native_resident.py capture-control). Write the protocol with that path.")
    QUIET.write_text(NAME + "\n")
    say(f"run {run_dir.name} against the resident engine (pid {s['pid']}, up {(time.time() - s['started']) / 60:.0f} min, no engine start); quiet flag up")
    try:
        with (run_dir / "campaign.log").open("a") as log:
            for phase in phases:
                done = subprocess.run([PY, "-B", str(W / "tools/mm3/native_followup.py"), "--protocol", str(protocol), "--run-dir", str(run_dir), "--phase", phase],
                                      stdout=log, stderr=subprocess.STDOUT)
                if done.returncode:
                    raise SystemExit(f"phase {phase} failed (see {run_dir / 'campaign.log'}); the engine stays up")
                say(f"phase {phase} done")
    finally:
        if QUIET.exists() and QUIET.read_text().strip() == NAME:
            QUIET.unlink()
        say("quiet flag down; the engine stays up")
    return 0


def cmd_stop() -> int:
    if not STATE.exists():
        say("nothing recorded to stop")
        return 1
    pid = int(json.loads(STATE.read_text())["pid"])
    if alive(pid):
        os.kill(pid, signal.SIGTERM)
        say(f"asked the engine (pid {pid}) to stop")
        deadline = time.monotonic() + 600
        while alive(pid) and time.monotonic() < deadline:
            time.sleep(2)
    while True:
        used = [int(x) for x in subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True).split()]
        if max(used) < 2000:
            break
        time.sleep(3)
    if owner() == NAME:
        shutil.rmtree(MUTEX)
    STATE.unlink()
    say(f"stopped; the cards are free ({used} MiB)")
    return 0


def main() -> int:
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if a[0] == "status":
        return cmd_status()
    if a[0] == "start":
        return cmd_start(Path(a[a.index("--config") + 1]) if "--config" in a else W / "docs/minimax-wsl/native-server.json")
    if a[0] == "adopt" and len(a) == 3:
        return cmd_adopt(int(a[1]), Path(a[2]))
    if a[0] == "needs-restart" and len(a) == 2:
        return cmd_needs_restart(Path(a[1]))
    if a[0] == "run" and len(a) >= 2:
        return cmd_run(Path(a[1]).resolve(), a[2:] or ["standalone", "check"])
    if a[0] == "stop":
        return cmd_stop()
    if a[0] == "capture-control":
        s = state()
        if s is None:
            raise SystemExit("no resident engine is up")
        print(s.get("capture_control") or "")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
