"""C2. Weak scaling of the ASORA raytracing step over several GPUs.

Fixed work per GPU (default N=256 with 1e4 sources per rank at R=30, 1e5 and
1e4 at R=15, batch 8: the per-GPU work of C1 at P=1), so the total number of
sources grows as P x sources per rank. Sources are split over the ranks
and the rates are summed with MPI Reduce + Bcast, as in pyC2Ray's MPI mode (see
mpi_scaling.py). One run per P; the job script sweeps P.

Usage (one rank per GPU, each rank seeing only its GPU):
  srun -n P --gpus-per-task=1 python benchmark_scripts/C2_weak_scaling.py
"""

import argparse

import common
import mpi_scaling

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mesh-size", type=int, default=256)
p.add_argument("--cases", nargs="+", default=["30:10000", "15:100000", "15:10000"],
               help="R:sources_per_rank pairs")
p.add_argument("--batch-size", type=int, default=8)
common.add_common_args(p)
args = p.parse_args()

records = []
for case in args.cases:
    radius, per_rank = case.split(":")
    records.append(mpi_scaling.measure(args.mesh_size, args.batch_size, int(per_rank) * mpi_scaling.nprocs, radius,
                                       args.warmup, args.repeats, args.max_seconds, sources_per_rank=int(per_rank)))
mpi_scaling.save("C2_weak_scaling", vars(args), records)
