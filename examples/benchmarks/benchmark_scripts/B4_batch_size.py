"""B4. source_batch_size sweep of the ASORA raytracing kernel on one GPU.

ASORA raytraces `batch_size` sources in parallel (one GPU block per source) and
allocates one column-density grid per batch slot, so device memory is
(3 + batch_size) * N^3 * 8 bytes. For 100 and a large number of sources (default 1e4),
mesh size N=256 and R=30 cells, sweep the batch size to find the
throughput/memory sweet spot. Batch sizes that do not fit are skipped.

For N=512 batch sizes up to 32 fit on a 64 GB MI250X GCD, e.g.:
  python benchmark_scripts/B4_batch_size.py --mesh-size 512 --num-sources 100

Usage (on a GPU node):
  python benchmark_scripts/B4_batch_size.py [--batch-sizes 1 2 4 8 16 32 64 128 256] ...
"""

import argparse

import common

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mesh-size", type=int, default=256)
p.add_argument("--num-sources", type=int, nargs="+", default=[100, 10000])
p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
p.add_argument("--radius", default="30", help="raytracing radius in cells, or 'box'")
common.add_common_args(p)
args = p.parse_args()

records = [
    common.measure_point(args.mesh_size, b, n, args.radius, args.warmup, args.repeats, args.max_seconds)
    for n in args.num_sources
    for b in args.batch_sizes
]
common.save_results("B4_batch_size", vars(args), records)
