#!/usr/bin/env python3
"""Start and stop the routing lab: the engine with token records switched on, then the lab's page.

    python3 tools/mm3/routing_lab/launch_lab.py start      # the page is up at once; the model load takes 5 to 7 minutes
    python3 tools/mm3/routing_lab/launch_lab.py status
    python3 tools/mm3/routing_lab/launch_lab.py restart-page   # after a change to the page's code; the engine stays up
    python3 tools/mm3/routing_lab/launch_lab.py stop       # always run this when done: it frees the cards

`start` takes the GPU mutex the test campaigns use (it refuses, and says who holds it, if a campaign has the
cards), loads one engine with the drafter so that both modes are a per-prompt choice, runs the same setup as the
timed tests (bootstrap request, helper-first placement), saves that cache state as the one the page can restore,
and starts the page on http://127.0.0.1:8796/ . While the lab is up no timed test can run: `stop` releases the cards.

The lab never touches power limits, clocks or processor affinity.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[2]
PYTHON = "/home/mdwbr/project-maya/.venv/bin/python"
RUNS = Path("/mnt/wsl/transfer/minimax-wsl/runs")
RUN = RUNS / "routing-lab-20261110"
LOCK = RUNS / "plan-completion-20261110/locks/gpu.lock.d"
QUIET = RUNS / "plan-completion-20261110/locks/quiet.lock"
OWNER = "routing-lab"
PACK = "/mnt/wsl/transfer/minimax-wsl/artifacts/proposal-2/native-pack"
DFLASH2 = "/mnt/wsl/transfer/minimax-wsl/artifacts/dflash2-pack"
SPEC_PLAN = RUNS / "exchange-schedule-20261110/bonus-plan1/dflash2-plan-v1.txt"
PLACEMENT_PROFILE = RUNS / "maya-completion-20261104/conditioned-residency/history/generation-profile.bin"
ENGINE_PORT, LAB_PORT = 8097, 8796
BASE = f"http://127.0.0.1:{ENGINE_PORT}"
SESSION = RUN / "engine"
RECEIPT = SESSION / "launch.json"


def say(msg: str) -> None:
    print(f"{dt.datetime.now():%H:%M:%S} {msg}", flush=True)


def alive(pid) -> bool:
    try:
        return Path(f"/proc/{int(pid)}/stat").read_text().split()[2] != "Z"
    except (FileNotFoundError, ProcessLookupError, TypeError, ValueError):
        return False


POWER_CAPS = {"5090": 425.0, "3090": 390.0}     # the user's limits; the factory defaults (575 W, 420 W) are higher
# The engine sees these two cards only, the 5090 first. CUDA numbers devices fastest first, so with any other card
# in the machine "device 1" stops being the 3090 (an RTX 4080 took that place on 2026-11-10 and the engine refused
# to load). The UUIDs are the ones tools/mm3/native_host_control.py uses.
CARD_UUIDS = "GPU-ea52aedc-8ee4-8e13-7ee7-c1bb774e12e9,GPU-ea9ac138-d20c-c5f7-38b0-6d18cccabbf1"
# The RTX 4080 (16 GiB, PCIe 4.0 x2), used only when start is given --third-gb: the engine then sees it as device 2
# and keeps a fixed pool of copies of HOST experts on it (the per-request setting third=1 uses the pool).
THIRD_UUID = "GPU-6d5a59c8-62b5-855e-49fd-b18a1ddeb821"


def cards() -> list[dict]:
    """The project's two cards (RTX 5090 and RTX 3090). Any other card in the machine is not the lab's to use."""
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.used,power.limit",
                                   "--format=csv,noheader,nounits"], text=True)
    found = []
    for line in out.strip().splitlines():
        name, used, limit = [x.strip() for x in line.split(",")]
        key = next((k for k in POWER_CAPS if k in name), None)
        if key:
            found.append({"card": key, "used": int(float(used)), "limit": float(limit)})
    return found


def gpu_used() -> list[int]:
    return [c["used"] for c in cards()] or [0]


def over_power_caps() -> list[str]:
    return [f"the {c['card']}'s power limit is {c['limit']:g} W, above the user's {POWER_CAPS[c['card']]:g} W"
            for c in cards() if c["limit"] > POWER_CAPS[c["card"]] + 0.5]


def post(path: str, body: dict, timeout: float = 900) -> dict:
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def receipt() -> dict:
    return json.loads(RECEIPT.read_text()) if RECEIPT.exists() else {}


def lock_owner() -> str | None:
    try:
        return (LOCK / "owner").read_text().strip()
    except FileNotFoundError:
        return "?" if LOCK.exists() else None


def release_lock() -> None:
    if lock_owner() == OWNER:                 # only our own
        shutil.rmtree(LOCK)
        say("GPU mutex released")


def engine_pids(exe: Path) -> list[int]:
    out = []
    for entry in Path("/proc").iterdir():
        if entry.name.isdigit():
            try:
                if entry.joinpath("cmdline").read_bytes().split(b"\0")[0] == str(exe).encode():
                    out.append(int(entry.name))
            except OSError:
                pass
    return out


def source_commit(exe: Path) -> str:
    """The commit the binary was built from, as the builder wrote it beside the binary ("source commit <id> ...")."""
    try:
        words = Path(str(exe) + ".source").read_text().split()
        return words[2] if words[:2] == ["source", "commit"] else "unknown"
    except (OSError, IndexError):
        return "unknown"


def typical_load_s() -> int:
    """How long a load took in earlier sessions (their receipts are kept), for the page's progress line."""
    seen = []
    for path in SESSION.glob("launch-*.json"):
        try:
            v = json.loads(path.read_text()).get("load_seconds")
        except (OSError, ValueError):
            continue
        if isinstance(v, (int, float)) and v > 60:
            seen.append(float(v))
    seen.sort()
    return int(seen[len(seen) // 2]) if seen else 330


def start_page(records: Path, stamp: str) -> tuple[subprocess.Popen, Path]:
    lab_log = SESSION / f"lab-server-{stamp}.log"
    with lab_log.open("xb") as log:
        lab = subprocess.Popen([sys.executable, str(HERE / "lab_server.py"), "--port", str(LAB_PORT),
                                "--maya", BASE, "--records", str(records), "--receipt", str(RECEIPT)],
                               stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    time.sleep(1.5)
    if lab.poll() is not None:
        raise SystemExit(f"the lab's page server exited at once; see {lab_log}")
    return lab, lab_log


def restart_page() -> None:
    """Restart the page's backend only (after a change to the page's code). The engine stays loaded."""
    r = receipt()
    if not alive(r.get("server_pid")):
        raise SystemExit("the lab's engine is not running; use start")
    if alive(r.get("lab_pid")):
        os.kill(int(r["lab_pid"]), signal.SIGTERM)
        for _ in range(50):
            if not alive(r["lab_pid"]):
                break
            time.sleep(0.1)
    lab, lab_log = start_page(Path(r["token_records"]), f"{dt.datetime.now():%Y%m%d-%H%M%S}")
    r.update(lab_pid=lab.pid, lab_log=str(lab_log))
    RECEIPT.write_text(json.dumps(r, indent=2) + "\n")
    say(f"page restarted: http://127.0.0.1:{LAB_PORT}/")


def stop(quiet: bool = False) -> None:
    r = receipt()
    for name in ("lab_pid", "server_pid"):
        pid = r.get(name)
        if alive(pid):
            os.kill(int(pid), signal.SIGTERM)
            say(f"stopping {name.split('_')[0]} (pid {pid})")
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline and (alive(r.get("server_pid")) or alive(r.get("engine_pid"))):
        time.sleep(2)
    if alive(r.get("server_pid")) or alive(r.get("engine_pid")):
        raise SystemExit("the engine is still alive after 10 minutes; the GPU mutex is kept. Look at "
                         f"{SESSION / 'server.log'} and stop it by hand, then run stop again.")
    if r:
        while time.monotonic() < deadline and max(gpu_used()) >= 2000 and lock_owner() == OWNER:
            time.sleep(3)
        r["stopped_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        RECEIPT.write_text(json.dumps(r, indent=2) + "\n")
        RECEIPT.rename(SESSION / f"launch-{dt.datetime.now():%Y%m%d-%H%M%S}.json")
    release_lock()
    if not quiet:
        say(f"stopped; GPU memory in use: {gpu_used()} MiB")


def start(a) -> None:
    r = receipt()
    if alive(r.get("server_pid")):
        raise SystemExit(f"the lab's engine is already running (pid {r['server_pid']}); the page is "
                         f"http://127.0.0.1:{LAB_PORT}/ . Run stop first to restart it.")
    exe = Path(a.exe)
    if not exe.is_file():
        raise SystemExit(f"no engine binary at {exe}. The lab needs the build with token records and the routing "
                         "lab's rules; pass --exe if it is elsewhere.")
    for need in (Path(PACK), Path(DFLASH2), SPEC_PLAN, PLACEMENT_PROFILE, Path(PYTHON)):
        if not need.exists():
            raise SystemExit(f"missing: {need}")
    if len(cards()) != 2:
        raise SystemExit(f"expected one RTX 5090 and one RTX 3090, found {[c['card'] for c in cards()]}")
    over = over_power_caps()
    if over:                                  # the limits fall back to the factory values when the driver restarts
        raise SystemExit("not loading: " + "; ".join(over) + ". Restore them first: "
                         "python3 tools/mm3/native_host_control.py restore-power-limit all")
    SESSION.mkdir(parents=True, exist_ok=True)
    records = RUN / "token-records"
    records.mkdir(parents=True, exist_ok=True)

    while True:                                   # the GPU mutex: never taken from someone else
        try:
            LOCK.mkdir()
            (LOCK / "owner").write_text(OWNER + "\n")
            break
        except FileExistsError:
            if not a.wait:
                raise SystemExit(f"the cards are in use: the GPU mutex is held by '{lock_owner()}'. Try again when "
                                 "that campaign has finished, or pass --wait to queue behind it.")
            say(f"waiting for the GPU mutex (held by {lock_owner()})")
            time.sleep(60)
    say("GPU mutex taken")
    process = lab = None
    try:
        used = gpu_used()
        if max(used) > 2000:
            raise SystemExit(f"a card already holds memory ({used} MiB) although nobody holds the mutex; not loading")
        stamp = f"{dt.datetime.now():%Y%m%d-%H%M%S}"
        third_gb = float(getattr(a, "third_gb", 0) or 0)
        uuids = CARD_UUIDS + ("," + THIRD_UUID if third_gb > 0 else "")
        defaults = SESSION / "native-runtime.json"
        shutil.copyfile(WORKTREE / "docs/minimax-wsl/native-runtime.json", defaults)
        config = {
            "exe": str(exe), "cwd": str(WORKTREE),
            "args": ["--pack", PACK, "--gpus", "0", "--max-context", "8192", "--chunk", "16", "--staging", "8",
                     "--native-helper", "1", "--native-helper-gb", "22", "--vision", "off", "--dflash2", DFLASH2,
                     "--spec-plan", str(SPEC_PLAN), "--no-prefill-tune", "--serve", "--native-primary-arithmetic", "fp4",
                     "--native-helper-dispatch", "worker", "--native-cooperative", "1"] +
                    (["--native-third", "2", "--native-third-gb", f"{third_gb:g}"] if third_gb > 0 else []),
            "tokenizer": PACK + "/tokenizer", "model_name": "MiniMax-M3.1-Flash-native-wsl",
            "log": str(SESSION / f"engine-{stamp}.log"),
            "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 0, "min_p": 0.0},
            "runtime_defaults_file": str(defaults),
            "idle_unload_s": 0, "host": "127.0.0.1", "reasoning_effort": "high", "thinking_budget": 0,
            "loop_guard": False, "monitoring": False,
            # MM3_ROUTE_LAB_DUMP: one line a layer pass saying how many picks were ready at routing, lab path or
            # serving kernel alike. It is what compares the two paths' readiness (about 8.5 KB a one-row token).
            "env": {"MM3_SERVE_MONITORING": "0", "STRATA_ENGINE_STALL_S": "0", "MM3_TOKEN_RECORDS_DIR": str(records),
                    "MM3_ROUTE_LAB_DUMP": str(records / f"route-lab-dump-{stamp}.jsonl"),
                    "MM3_SOURCE_COMMIT": source_commit(exe),
                    # (so a timed campaign can run on this load: tools/mm3/native_followup.py checks this path)
                    "MM3_CAPTURE_CONTROL": str(SESSION / "capture-control"),
                    "CUDA_VISIBLE_DEVICES": uuids},
        }
        config_path = SESSION / "server.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        server = [PYTHON, "serve/server.py", "--engine", "strata", "--config", str(config_path), "--no-monitoring",
                  "--host", "127.0.0.1", "--port", str(ENGINE_PORT), "--idle-unload", "0"]
        env = os.environ.copy()
        for key in list(env):
            if key.startswith(("NSYS", "CUDA_INJECTION", "NVTX_INJECTION", "MM3_CAPTURE", "MM3_REPLAY", "MM3_STATE")) \
                    or key == "LD_PRELOAD":
                env.pop(key)
        env.update(STRATA_ENGINE_QUIT_S="300", MM3_SERVE_MONITORING="0", STRATA_ENGINE_STALL_S="0",
                   CUDA_VISIBLE_DEVICES=uuids)
        began = time.monotonic()
        log_path = SESSION / f"server-{stamp}.log"
        with log_path.open("xb") as log:
            process = subprocess.Popen(server, cwd=WORKTREE, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
        r = dict(launched_utc=dt.datetime.now(dt.timezone.utc).isoformat(), server_pid=process.pid, command=server,
                 exe=str(exe), binary_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), token_records=str(records),
                 server_log=str(log_path),
                 server_tree_commit=subprocess.run(["git", "-C", str(WORKTREE), "rev-parse", "--short", "HEAD"],
                                                   capture_output=True, text=True).stdout.strip())
        r["typical_load_s"] = typical_load_s()
        RECEIPT.write_text(json.dumps(r, indent=2) + "\n")
        # The page comes up now, not after the load: it shows the load's progress, sends nothing to the engine
        # until the setup below is done (the receipt tells it), and then connects by itself.
        if not a.no_page:
            lab, lab_log = start_page(records, stamp)
            r.update(lab_pid=lab.pid, lab_log=str(lab_log))
            RECEIPT.write_text(json.dumps(r, indent=2) + "\n")
            say(f"the page is up: http://127.0.0.1:{LAB_PORT}/  (it shows the load and connects by itself)")
        say(f"engine loading (pid {process.pid}); about {r['typical_load_s'] // 60} min {r['typical_load_s'] % 60} s. Log: {log_path}")
        deadline, last = time.monotonic() + 1800, time.monotonic()
        while True:
            try:
                metrics = json.load(urllib.request.urlopen(BASE + "/metrics", timeout=5))
                if metrics.get("engine", {}).get("cooperative_cpu_scale_control") == 1:
                    break
            except (OSError, ValueError):
                pass
            if process.poll() is not None:
                raise SystemExit(f"the server exited with {process.returncode} while loading; see {log_path}")
            if time.monotonic() > deadline:
                raise SystemExit("the engine was not ready after 30 minutes")
            if time.monotonic() - last > 60:
                say(f"still loading ({int(time.monotonic() - began)} s; cards hold {gpu_used()} MiB)")
                last = time.monotonic()
            time.sleep(5)
        pids = engine_pids(exe)
        r.update(ready_utc=dt.datetime.now(dt.timezone.utc).isoformat(), load_seconds=round(time.monotonic() - began, 1),
                 engine_pid=pids[0] if len(pids) == 1 else None)
        RECEIPT.write_text(json.dumps(r, indent=2) + "\n")
        held = {c["card"]: c["used"] for c in cards()}
        if held.get("3090", 0) < 8000 or held.get("5090", 0) < 8000:
            raise SystemExit(f"the engine is ready but the project's cards hold {held} MiB: it did not load on the "
                             "5090 and the 3090 as expected")
        engine = metrics.get("engine", {})
        controls = {k: engine.get(k) for k in ("route_lab_control", "token_records_control", "force_tokens_control")}
        say(f"engine ready in {r['load_seconds']} s; lab controls: {controls}")
        missing = [k for k, v in controls.items() if v != 1]
        if missing:
            say("WARNING: this binary does not report " + ", ".join(missing) +
                "; the page will work only as far as the engine allows (see the page's status badge)")

        # The timed tests' setup: one short request (it triggers the one-time service calibration), then the
        # helper-first placement from the retained profile, then a snapshot the page restores from.
        profiles = json.loads((HERE / "profiles.json").read_text())
        say("setup: bootstrap request")
        post("/v1/chat/completions", {
            "model": profiles["model"], "messages": [{"role": "user", "content": "How many minutes are in one hour? Give the number."}],
            "stream": False, "max_tokens": 32, "temperature": 1, "top_p": 0.95, "top_k": 0, "min_p": 0, "seed": 20261101,
            "reasoning_effort": "high", "loop_guard": False,
            "strata_tune": dict(profiles["one_row"], routing_prior_bonus=0.2)})
        order = getattr(a, "placement_order", None)
        profile = Path(getattr(a, "placement_profile", None) or PLACEMENT_PROFILE)   # (the demand profile the cards are placed from)
        if not profile.is_file() or profile.stat().st_size != 30720:
            raise SystemExit(f"the placement profile {profile} is not a 30720-byte demand profile")
        r.update(placement_profile=str(profile))
        if third_gb > 0 and order:
            # all three cards from the profile in one command: which takes the hottest experts, the next, the rest
            say(f"setup: three-card placement, order {order}")
            filled = post("/v1/minimax/placement", {"action": "three", "profile_path": str(profile), "order": order})
            if not filled.get("ownership_complete") or not filled.get("third_capacity"):
                raise SystemExit("the three-card placement did not complete; see the server log")
            r.update(third_gb=third_gb, third_capacity=filled["third_capacity"], placement_order=order)
            (SESSION / f"third-{stamp}.json").write_text(json.dumps(filled))
            say(f"placed: the third card holds {filled['third_capacity']} experts, {filled.get('third_written')} written, "
                f"{filled.get('third_mismatched')} read back different")
        else:
            say("setup: helper-first placement")
            placed = post("/v1/minimax/placement", {"action": "helper-first", "profile_path": str(profile)})
            if not placed.get("ownership_complete"):
                raise SystemExit("placement ownership is not complete; see the server log")
        if third_gb > 0 and not order:
            say("setup: filling the third card's pool")
            filled = post("/v1/minimax/placement", {"action": "third", "profile_path": str(profile)})
            if not filled.get("third_capacity"):
                raise SystemExit("the third card's pool was not filled; see the server log")
            r.update(third_gb=third_gb, third_capacity=filled["third_capacity"], third_fill_ms=filled.get("third_fill_ms"))
            (SESSION / f"third-{stamp}.json").write_text(json.dumps(filled))
            say(f"the third card holds {filled['third_capacity']} experts (filled in {filled.get('third_fill_ms', 0) / 1000:.1f} s)")
        say("setup: snapshot of the starting cache state")
        snap = post("/v1/minimax/placement", {"action": "snapshot"})
        (SESSION / f"origin-{stamp}.json").write_text(json.dumps(snap))

        if lab is not None and lab.poll() is not None:
            raise SystemExit(f"the lab's page server stopped during the load; see {lab_log}")
        r.update(setup_done_utc=dt.datetime.now(dt.timezone.utc).isoformat())
        RECEIPT.write_text(json.dumps(r, indent=2) + "\n")
        say("READY: the engine is up" + ("" if a.no_page else f" and http://127.0.0.1:{LAB_PORT}/ is connected to it") +
            "   (when done: launch_lab.py stop)")
    except BaseException:
        say("start failed; stopping what was started and releasing the cards")
        for p in (lab, process):
            if p is not None and p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    pass
        try:
            stop(quiet=True)
        finally:
            release_lock()
        raise


def status() -> None:
    r = receipt()
    print(f"GPU mutex: {lock_owner() or 'free'}; quiet flag: {'up' if QUIET.exists() else 'down'}; "
          f"cards hold {gpu_used()} MiB")
    for line in over_power_caps():
        print("WARNING: " + line + " (native_host_control.py restore-power-limit all)")
    if not r:
        print("the lab is not running")
        return
    print(f"engine server pid {r.get('server_pid')}: {'alive' if alive(r.get('server_pid')) else 'gone'}; "
          f"engine pid {r.get('engine_pid')}: {'alive' if alive(r.get('engine_pid')) else 'gone'}; "
          f"page pid {r.get('lab_pid')}: {'alive' if alive(r.get('lab_pid')) else 'gone'}")
    print(f"binary {r.get('exe')} sha256 {str(r.get('binary_sha256'))[:16]}; loaded in {r.get('load_seconds')} s")
    try:
        with urllib.request.urlopen(BASE + "/v1/status", timeout=3) as resp:
            s = json.loads(resp.read())
        print(f"engine answers: model {s.get('model')}, up {s.get('uptime_s')} s; page http://127.0.0.1:{LAB_PORT}/")
    except (OSError, ValueError) as exc:
        print(f"engine does not answer on {BASE}: {exc}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("start")
    s.add_argument("--exe", default=str(RUN / "bin/maya-mm3-xs"), help="the engine binary built with the lab's features")
    s.add_argument("--wait", action="store_true", help="queue behind a campaign that holds the cards instead of refusing")
    s.add_argument("--placement-profile", default=None,
                   help="the demand profile the cards are placed from (default: the retained generation profile)")
    s.add_argument("--placement-order", default=None,
                   help="with --third-gb: place all three cards from the profile, e.g. hpt (h the 3090, p the 5090, "
                        "t the 4080: which takes the hottest experts, which the next, which the rest)")
    s.add_argument("--third-gb", type=float, default=0.0,
                   help="also use the RTX 4080: GiB of experts for its fixed pool (14.1 fits); needs a binary with --native-third")
    s.add_argument("--no-page", action="store_true",
                   help="the engine only, without the page: for scripts that drive the engine themselves (the fidelity sweep)")
    sub.add_parser("stop")
    sub.add_parser("status")
    sub.add_parser("restart-page", help="restart the page's backend only; the engine stays loaded")
    a = ap.parse_args()
    {"start": lambda: start(a), "stop": stop, "status": status, "restart-page": restart_page}[a.cmd]()


if __name__ == "__main__":
    main()
