"""C1. Strong scaling of the ASORA raytracing step over several GPUs.

Fixed total problem (default N=256 with 1e4 sources at R=30 and 1e5 sources at
R=15, batch 8), run on P MPI ranks with one GPU each: sources are split over
the ranks and the rates are summed with MPI Reduce + Bcast, as in pyC2Ray's MPI
mode (see mpi_scaling.py). One run per P; the job script sweeps P.

Usage (one rank per GPU, each rank seeing only its GPU):
  srun -n P --gpus-per-task=1 python benchmark_scripts/C1_strong_scaling.py
"""

import argparse

import common
import mpi_scaling

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mesh-size", type=int, default=256)
p.add_argument("--cases", nargs="+", default=["30:10000", "15:100000"],
               help="R:num_sources pairs (total sources, split over ranks)")
p.add_argument("--batch-size", type=int, default=8)
common.add_common_args(p)
args = p.parse_args()

records = []
for case in args.cases:
    radius, nsrc = case.split(":")
    records.append(mpi_scaling.measure(args.mesh_size, args.batch_size, int(nsrc), radius,
                                       args.warmup, args.repeats, args.max_seconds))
mpi_scaling.save("C1_strong_scaling", vars(args), records)
