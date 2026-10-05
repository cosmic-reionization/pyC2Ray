"""B2. Source-count scaling of the ASORA raytracing kernel on one GPU.

Fixed mesh size (production-relevant, default N=256) and batch size, number of
sources swept from 1 to 1e6 (as in the docs tutorial on memory/time,
https://pyc2ray.readthedocs.io/en/latest/tutorials/memory.html), for fixed
raytracing radii (default 15 and 30 cells). Records time per call, throughput (sources/s) and device memory.

Usage (on a GPU node):
  python benchmark_scripts/B2_num_sources.py [--mesh-size 64 --radii 15] ...
"""

import argparse

import common

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mesh-size", type=int, default=256)
p.add_argument("--num-sources", type=int, nargs="+", default=[1, 10, 100, 1000, 10000, 100000, 1000000])
p.add_argument("--batch-size", type=int, default=8)
p.add_argument("--radii", nargs="+", default=["15", "30"], help="raytracing radius in cells, or 'box'")
common.add_common_args(p)
args = p.parse_args()

records = [
    common.measure_point(args.mesh_size, args.batch_size, n, radius, args.warmup, args.repeats, args.max_seconds)
    for radius in args.radii
    for n in args.num_sources
]
common.save_results("B2_num_sources", vars(args), records)
