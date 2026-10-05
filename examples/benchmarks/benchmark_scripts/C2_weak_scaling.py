"""C2. Weak scaling of the ASORA raytracing step over several GPUs.

Fixed work per GPU (default N=256, 1e4 sources per rank at R=30, batch 8), so
the total number of sources grows as P x 1e4. Sources are split over the ranks
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
p.add_argument("--sources-per-rank", type=int, default=10000)
p.add_argument("--radius", default="30")
p.add_argument("--batch-size", type=int, default=8)
common.add_common_args(p)
args = p.parse_args()

nsrc = args.sources_per_rank * mpi_scaling.nprocs
rec = mpi_scaling.measure(args.mesh_size, args.batch_size, nsrc, args.radius, args.warmup, args.repeats,
                          args.max_seconds, sources_per_rank=args.sources_per_rank)
mpi_scaling.save("C2_weak_scaling", vars(args), [rec])
