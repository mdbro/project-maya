#!/usr/bin/env python3
"""Recorded router inputs for the per-row budget's check (tests/mm3_route_lab_test.cpp, tests/mm3_resident_router_test.cu):
generation rows of the detailed captures the re-routed replay extracted (tools/mm3/native_reroute_replay.py extract), as
the router kernel captured them - each row's 128 sigmoids and sigmoid-plus-bias values - with the ready masks its routing
read.

    python3 -I tests/mm3_route_floor_fixture.py --out tests/data/mm3_router_recorded.bin              # the checked-in sample
    python3 -I tests/mm3_route_floor_fixture.py --all --out <scratch>/all.bin                         # every row

The sample: from each extract, rows at even spacing, 24 the capture's bonus moved a pick in and 8 it did not. The file:
"MM3RRI1\\0", uint32 rows, then per row float32 sigmoid[128], float32 base[128], uint32 ready_primary[4],
uint32 ready_helper[4] (little-endian). Needs numpy; reads nothing but the extracts named.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

DEFAULT = sorted(Path("/mnt/wsl/transfer/minimax-wsl/runs/reroute-replay-20261110/extract-20261110").glob("detailed-*-val-*.npz"))


def rows_of(path: Path, take_all: bool, moved_n: int, still_n: int):
    import numpy as np
    z = np.load(path)
    sig, base, orig, exe, rp = z["row_sig"], z["row_base"], z["row_original"], z["row_executed"], z["row_pass"]
    primary, helper = z["ready_primary"], z["ready_helper"]
    n = len(sig)
    if take_all:
        chosen = range(n)
    else:
        moved = [i for i in range(n) if set(orig[i].tolist()) != set(exe[i].tolist())]
        still = [i for i in range(n) if set(orig[i].tolist()) == set(exe[i].tolist())]
        pick = lambda xs, k: [xs[int(j * len(xs) / k)] for j in range(min(k, len(xs)))]
        chosen = sorted(pick(moved, moved_n) + pick(still, still_n))
    for i in chosen:
        p = int(rp[i])
        yield (sig[i].astype("<f4").tobytes(), base[i].astype("<f4").tobytes(),
               primary[p].astype("<u4").tobytes(), helper[p].astype("<u4").tobytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--all", action="store_true", help="every generation row, not the sample")
    ap.add_argument("--extract", nargs="*", default=[str(p) for p in DEFAULT], help="the replay's .npz extracts")
    a = ap.parse_args()
    if not a.extract:
        print("no extracts found", file=sys.stderr)
        return 1
    body, count = bytearray(), 0
    for path in a.extract:
        for parts in rows_of(Path(path), a.all, 24, 8):
            for b in parts:
                body += b
            count += 1
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(b"MM3RRI1\0" + struct.pack("<I", count) + bytes(body))
    print(f"{out}: {count} rows from {len(a.extract)} extracts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
