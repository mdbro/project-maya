#!/usr/bin/env python3
"""Run native_handoff_probe under nsys on each GPU and mode, and report the gaps between its operations from CUPTI timestamps.

  upload -> kernel     start of the first kernel minus the end of the record's upload (copy engine, or the mapped-copy kernel)
  kernel -> readback   start of the 4 KB readback minus the end of that kernel
  readback -> kernel   start of the second kernel minus the end of the readback
  upload GB/s          the record's bytes over the upload's duration

  native_handoff_probe.py <probe binary> <output directory> [--iterations N] [--devices 0,1] [--modes same,event,poll,busy,mapped]
"""
import argparse, json, pathlib, sqlite3, statistics, subprocess

RECORD = 31850752


def quantiles(values):
    v = sorted(values)
    if not v:
        return None
    return dict(n=len(v), p10=v[len(v) // 10], p50=statistics.median(v), p90=v[9 * len(v) // 10], mean=statistics.fmean(v))


def gaps(path: pathlib.Path, device: int) -> dict:
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    names = dict(db.execute('select id, value from StringIds'))
    ops = [(s, e, 'kernel', names.get(n, '?')) for s, e, n in db.execute('select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL where deviceId=?', (device,))]
    ops += [(s, e, 'copy', b) for s, e, b in db.execute('select start, end, bytes from CUPTI_ACTIVITY_KIND_MEMCPY where deviceId=?', (device,))]
    db.close()
    ops.sort()
    out = {k: [] for k in ('upload_to_kernel', 'kernel_to_readback', 'readback_to_kernel', 'upload_gbps', 'upload_ms')}
    for i in range(len(ops) - 3):
        up, k1, rb, k2 = ops[i:i + 4]
        is_upload = (up[2] == 'copy' and up[3] == RECORD) or (up[2] == 'kernel' and 'mapped_copy' in up[3])
        if not (is_upload and k1[2] == 'kernel' and 'touch' in k1[3] and rb[2] == 'copy' and rb[3] == 4096 and k2[2] == 'kernel' and 'touch' in k2[3]):
            continue
        out['upload_to_kernel'].append((k1[0] - up[1]) * 1e-3)
        out['kernel_to_readback'].append((rb[0] - k1[1]) * 1e-3)
        out['readback_to_kernel'].append((k2[0] - rb[1]) * 1e-3)
        out['upload_ms'].append((up[1] - up[0]) * 1e-6)
        out['upload_gbps'].append(RECORD / (up[1] - up[0]))
    return {k: quantiles(v) for k, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument('probe')
    ap.add_argument('out')
    ap.add_argument('--iterations', type=int, default=300)
    ap.add_argument('--devices', default='0,1')
    ap.add_argument('--modes', default='same,event,poll,busy,mapped')
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    results = {}
    for device in (int(d) for d in a.devices.split(',')):
        for mode in a.modes.split(','):
            stem = out / f'probe-gpu{device}-{mode}'
            subprocess.run(['nsys', 'profile', '--capture-range=cudaProfilerApi', '--capture-range-end=stop', '--trace=cuda', '--sample=none',
                            '--cpuctxsw=none', '--export=sqlite', '--force-overwrite=true', '-o', str(stem), a.probe, str(device), mode, str(a.iterations)],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            r = gaps(stem.with_suffix('.sqlite'), device)
            results[f'gpu{device}/{mode}'] = r
            f = lambda k: (f"{r[k]['p10']:.1f} / {r[k]['p50']:.1f} / {r[k]['p90']:.1f}" if r[k] else '-')
            print(f"gpu{device} {mode:6s} n {r['upload_to_kernel']['n'] if r['upload_to_kernel'] else 0:4d}  upload->kernel {f('upload_to_kernel')} us   "
                  f"kernel->readback {f('kernel_to_readback')} us   readback->kernel {f('readback_to_kernel')} us   "
                  f"upload {r['upload_gbps']['p50'] if r['upload_gbps'] else 0:.1f} GB/s ({r['upload_ms']['p50'] if r['upload_ms'] else 0:.3f} ms)", flush=True)
    (out / 'result.json').write_text(json.dumps(results, indent=1))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
