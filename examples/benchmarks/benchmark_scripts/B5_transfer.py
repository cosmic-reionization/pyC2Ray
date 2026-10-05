"""B5. Host<->device transfer overhead of the ASORA raytracing call on one GPU.

Each asora.do_all_sources call zeroes the rate grid on the device, copies the
ionized fraction to the device and copies the rate and column-density grids
back (1 grid in, 2 grids out, N^3 float64 each), around the raytracing kernel.
ASORA has no internal timers, so for each mesh size this measures:
  - t_h2d: asora.density_to_device, a single host->device copy of one grid
  - t_overhead: do_all_sources with 0 sources (memset + 1 grid in + 2 out, no
    raytracing), i.e. the fixed per-call transfer cost
  - t_call: the full do_all_sources call (default 100 and 1000 sources, R=15
    and 30 cells)
and the transfer fraction t_overhead / t_call. The transfers do not depend on R
or the number of sources; the call time, and so the fraction, does. Note: arrays are numpy
(pageable) host memory, as in pyC2Ray.

Usage (on a GPU node):
  python benchmark_scripts/B5_transfer.py [--mesh-sizes 64 128 256 512] ...
"""

import argparse
import time

import numpy as np

import common

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mesh-sizes", type=int, nargs="+", default=[64, 128, 256, 512])
p.add_argument("--num-sources", type=int, nargs="+", default=[100, 1000, 10000])
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--radii", nargs="+", default=["15", "30"], help="raytracing radii in cells, or 'box'")
common.add_common_args(p, repeats=10)
args = p.parse_args()


def timed(fn, *a):
    for _ in range(args.warmup):
        fn(*a)
    ts = []
    for _ in range(args.repeats):
        t0 = time.perf_counter()
        fn(*a)
        ts.append(time.perf_counter() - t0)
        if sum(ts) > args.max_seconds:
            break
    return ts


records = []
for radius in args.radii:
    for nsrc in args.num_sources:
        for N in args.mesh_sizes:
            R = common.box_radius(N) if radius == "box" else float(radius)
            grid_bytes = N**3 * 8
            with common.asora_problem(N, args.batch_size, nsrc, R) as (do_args, info):
                ndens = do_args[4]
                t_h2d = timed(common.asora.density_to_device, ndens, N)
                no_src = do_args[:7] + (0,) + do_args[8:]
                t_over = timed(common.asora.do_all_sources, *no_src)
                t_call, valid = common.time_do_all_sources(do_args, args.warmup, args.repeats, args.max_seconds)
            rec = {"mesh_size": N, "radius": radius, "R_max": R, "num_sources": nsrc,
                   "batch_size": args.batch_size, **info, "grid_bytes": grid_bytes,
                   "t_h2d_s": t_h2d, "t_overhead_s": t_over, "times_s": t_call, "valid": valid,
                   "t_h2d_median_s": float(np.median(t_h2d)), "t_overhead_median_s": float(np.median(t_over)),
                   "time_median_s": float(np.median(t_call))}
            rec["h2d_GBps"] = grid_bytes / rec["t_h2d_median_s"] / 1e9
            rec["overhead_GBps"] = 3 * grid_bytes / rec["t_overhead_median_s"] / 1e9  # 1 in + 2 out
            rec["transfer_fraction"] = rec["t_overhead_median_s"] / rec["time_median_s"]
            print(f"R={radius} src={nsrc} N={N}: h2d {rec['t_h2d_median_s'] * 1e3:.2f} ms ({rec['h2d_GBps']:.1f} GB/s), "
                  f"overhead {rec['t_overhead_median_s'] * 1e3:.2f} ms, call {rec['time_median_s'] * 1e3:.2f} ms, "
                  f"transfer fraction {rec['transfer_fraction']:.2f}, valid={valid}", flush=True)
            records.append(rec)

common.save_results("B5_transfer", vars(args), records)
