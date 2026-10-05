"""Application profiled by the E (roofline) job: one warm-up and one timed
asora.do_all_sources call on the synthetic problem of the B tests. Run under
rocprof-compute (AMD) or Nsight Compute (NVIDIA), see jobs/<system>_E.sh.

Usage:
  python benchmark_scripts/E_roofline_app.py --mesh-size 256 --num-sources 1000 --radius 15 --batch-size 8
"""

import argparse
import time

import numpy as np

import common

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mesh-size", type=int, default=256)
p.add_argument("--num-sources", type=int, default=1000)
p.add_argument("--radius", default="15")
p.add_argument("--batch-size", type=int, default=8)
args = p.parse_args()

R = common.box_radius(args.mesh_size) if args.radius == "box" else float(args.radius)
with common.asora_problem(args.mesh_size, args.batch_size, args.num_sources, R) as (call, info):
    common.asora.do_all_sources(*call)  # warm-up
    t0 = time.perf_counter()
    common.asora.do_all_sources(*call)
    t = time.perf_counter() - t0
    phi = call[6]
    print(f"N={args.mesh_size} src={args.num_sources} R={args.radius} batch={args.batch_size}: call {t:.4f} s, "
          f"valid={bool(np.all(np.isfinite(phi)) and np.any(phi > 0))}", flush=True)
