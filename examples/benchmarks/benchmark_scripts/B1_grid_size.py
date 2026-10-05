"""B1. Grid-size scaling of the ASORA raytracing kernel on one GPU.

Fixed number of sources, mesh size N swept. Raytracing radii:
  - fixed R (default 15 and 30 cells; 15 as in tests/test_asora.py): work per
    source does not grow with N, only the N^3 grid transfers/memset do;
  - 'box': R covers the whole periodic box (cost relevant for production runs).
Mesh sizes whose ASORA allocation does not fit in device memory are skipped.

Usage (from anywhere, on a GPU node):
  python benchmark_scripts/B1_grid_size.py [--mesh-sizes 64 128 256 512 1024] ...
"""

import argparse

import numpy as np

import common

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mesh-sizes", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
p.add_argument("--num-sources", type=int, default=100)
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--radii", nargs="+", default=["15", "30", "box"], help="raytracing radius in cells, or 'box'")
p.add_argument("--warmup", type=int, default=1)
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--max-seconds", type=float, default=120.0, help="time budget per point for repeats")
args = p.parse_args()

_, total = common.gpu_mem_info()
records = []
for radius in args.radii:
    for N in args.mesh_sizes:
        R = common.box_radius(N) if radius == "box" else float(radius)
        rec = {"mesh_size": N, "radius": radius, "R_max": R, "num_sources": args.num_sources,
               "batch_size": args.batch_size,
               "device_bytes_expected": common.asora_bytes(N, args.batch_size)}
        if total is not None and rec["device_bytes_expected"] > 0.95 * total:
            rec["skipped"] = f"needs {rec['device_bytes_expected'] / 1e9:.1f} GB > 95% of {total / 1e9:.1f} GB"
            print(f"N={N} R={radius}: skipped ({rec['skipped']})", flush=True)
            records.append(rec)
            continue
        with common.asora_problem(N, args.batch_size, args.num_sources, R) as (do_args, info):
            times, valid = common.time_do_all_sources(do_args, args.warmup, args.repeats, args.max_seconds)
        rec.update(info)
        rec.update({"times_s": times, "time_median_s": float(np.median(times)), "valid": valid,
                    "time_per_source_s": float(np.median(times)) / args.num_sources})
        print(f"N={N} R={radius}: median {rec['time_median_s']:.4f} s over {len(times)} calls, "
              f"valid={valid}, device {info['device_bytes_used'] / 1e9 if info['device_bytes_used'] else float('nan'):.2f} GB",
              flush=True)
        records.append(rec)

common.save_results("B1_grid_size", vars(args), records)
