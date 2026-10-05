"""B3. Raytracing-radius scaling of the ASORA raytracing kernel on one GPU.

Fixed mesh size (default N=256) and batch size, for 100 and 1000 sources, the
raytracing radius R swept. For each source count all radii use the same problem
(same sources and density), so each result is compared with the R='box' result to quantify the
accuracy/cost trade-off:
  - rate_fraction: total photoionization rate captured, sum(phi_R) / sum(phi_box)
  - cells_within_1pc: fraction of cells whose rate is within 1% of phi_box
  - median_rel_deficit: median over cells of (phi_box - phi_R) / phi_box

Usage (on a GPU node):
  python benchmark_scripts/B3_radius.py [--radii 5 10 15 30 60 120 box] ...
"""

import argparse

import numpy as np

import common

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mesh-size", type=int, default=256)
p.add_argument("--num-sources", type=int, nargs="+", default=[100, 1000, 10000])
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--radii", nargs="+", default=["5", "10", "15", "30", "60", "120", "box"],
               help="raytracing radii in cells, or 'box' (always run, as the reference)")
common.add_common_args(p)
args = p.parse_args()

N = args.mesh_size
R_box = common.box_radius(N)
# Radii at or beyond the whole-box radius are the same as 'box'
too_large = [r for r in args.radii if r != "box" and float(r) >= R_box]
if too_large:
    print(f"N={N}: skipping R={', '.join(too_large)} (>= box radius {R_box:.1f})", flush=True)
radii = [r for r in args.radii if r != "box" and r not in too_large] + ["box"]
R_of = {r: R_box if r == "box" else float(r) for r in radii}

records = []
for nsrc in args.num_sources:
    phi, recs = {}, []
    with common.asora_problem(N, args.batch_size, nsrc, R_of["box"]) as (do_args, info):
        for r in ["box"] + radii[:-1]:  # reference first
            call = (R_of[r],) + do_args[1:]
            times, valid = common.time_do_all_sources(call, args.warmup, args.repeats, args.max_seconds)
            phi[r] = call[6].copy()
            t = float(np.median(times))
            free_now, _ = common.gpu_mem_info()
            base = info["device_bytes_free_before"]
            recs.append({"mesh_size": N, "radius": r, "R_max": R_of[r], "num_sources": nsrc,
                         "batch_size": args.batch_size, **info, "times_s": times, "time_median_s": t,
                         "valid": valid, "time_per_source_s": t / nsrc, "sources_per_s": nsrc / t,
                         # memory in use after this radius' calls, and the docs estimate
                         "device_bytes_used_after_calls": (base - free_now) if base is not None else None,
                         "device_bytes_docs_formula": float(common.docs_memory_bytes(N, args.batch_size, R_of[r], nsrc))})
            print(f"src={nsrc} R={r}: median {t:.4f} s over {len(times)} calls, valid={valid}", flush=True)

    ref = phi["box"]
    lit = ref > 0
    for rec in recs:
        p_r = phi[rec["radius"]]
        rel = (ref[lit] - p_r[lit]) / ref[lit]
        rec["rate_fraction"] = float(p_r.sum() / ref.sum())
        rec["cells_within_1pc"] = float(np.mean(np.abs(rel) <= 0.01))
        rec["median_rel_deficit"] = float(np.median(rel))
        print(f"src={nsrc} R={rec['radius']}: rate fraction {rec['rate_fraction']:.4f}, "
              f"cells within 1% {rec['cells_within_1pc']:.4f}", flush=True)
    records += sorted(recs, key=lambda rec: rec["R_max"])
records += [{"mesh_size": N, "radius": r, "R_max": float(r), "batch_size": args.batch_size,
             "skipped": f"R >= box radius {R_box:.1f}"} for r in too_large]

common.save_results("B3_radius", vars(args), records)
