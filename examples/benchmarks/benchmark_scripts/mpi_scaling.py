"""Shared core of the multi-GPU benchmarks C1 (strong) and C2 (weak scaling).

Reproduces the raytracing step of pyC2Ray's MPI mode (pyc2ray/evolve.py): each
rank holds the whole grid on its own GPU and raytraces its slice of the source
list; the photoionization rates are then summed on rank 0 (MPI Reduce) and
broadcast back (MPI Bcast), as host arrays. Per call this times:
  - t_rt: asora.do_all_sources on each rank (slowest rank = critical path)
  - t_comm: Reduce + Bcast of the N^3 rate grid
  - t_step: the whole step between two barriers

Run with one MPI rank per GPU, each rank seeing only its own GPU, e.g. with
Slurm: srun -n 8 --gpus-per-task=1 python C1_strong_scaling.py. (Do not set
CUDA_VISIBLE_DEVICES on AMD: HIP applies it after ROCR_VISIBLE_DEVICES.)
"""

import time

import numpy as np
from mpi4py import MPI

import common

comm = MPI.COMM_WORLD
rank, nprocs = comm.Get_rank(), comm.Get_size()


def log(msg):
    if rank == 0:
        print(msg, flush=True)


def split(num_sources):
    """Source range of this rank, as in pyc2ray/evolve.py."""
    per = num_sources // nprocs
    i0 = rank * per
    i1 = (rank + 1) * per if rank != nprocs - 1 else num_sources
    return i0, i1


def measure(mesh_size, batch_size, num_sources, radius, warmup=1, repeats=5, max_seconds=120.0, **extra):
    """One configuration on all ranks; returns the record on rank 0 (None elsewhere)."""
    R = common.box_radius(mesh_size) if radius == "box" else float(radius)
    i0, i1 = split(num_sources)
    rec = {"mesh_size": mesh_size, "radius": str(radius), "R_max": R, "num_sources": num_sources,
           "batch_size": batch_size, "nprocs": nprocs, "sources_per_rank_max": int(num_sources - (nprocs - 1) * (num_sources // nprocs)),
           "device_bytes_expected": common.asora_bytes(mesh_size, batch_size), **extra}
    _, total = common.gpu_mem_info()
    if total is not None and rec["device_bytes_expected"] > 0.95 * total:
        rec["skipped"] = f"needs {rec['device_bytes_expected'] / 1e9:.1f} GB > 95% of {total / 1e9:.1f} GB"
        log(f"P={nprocs} N={mesh_size} src={num_sources} R={radius}: skipped ({rec['skipped']})")
        return rec if rank == 0 else None

    with common.asora_problem(mesh_size, batch_size, num_sources, R, src_slice=(i0, i1)) as (args, info):
        phi = args[6]

        def step():
            comm.Barrier()
            t0 = time.perf_counter()
            phi[:] = 0.0
            common.asora.do_all_sources(*args)
            t1 = time.perf_counter()
            if rank == 0:
                comm.Reduce(MPI.IN_PLACE, [phi, MPI.DOUBLE], op=MPI.SUM, root=0)
            else:
                comm.Reduce([phi, MPI.DOUBLE], None, op=MPI.SUM, root=0)
            comm.Bcast([phi, MPI.DOUBLE], root=0)
            t2 = time.perf_counter()
            comm.Barrier()
            t3 = time.perf_counter()
            return t1 - t0, t2 - t1, t3 - t0

        for _ in range(warmup):
            step()
        t_rt, t_comm, t_step = [], [], []
        for _ in range(repeats):
            a, b, c = step()
            t_rt.append(a), t_comm.append(b), t_step.append(c)
            # same stopping decision on all ranks
            if comm.allreduce(sum(t_step), op=MPI.MAX) > max_seconds:
                break
        # Validity per rank, on its own result before Reduce/Bcast (ASORA returns
        # silently on GPU errors, and the broadcast would hide a failed rank)
        phi[:] = 0.0
        common.asora.do_all_sources(*args)
        valid = bool(np.all(np.isfinite(phi)) and (i1 == i0 or np.any(phi > 0)))

    # gather per-rank results on rank 0
    all_rt = comm.gather(t_rt, root=0)
    all_comm = comm.gather(t_comm, root=0)
    all_step = comm.gather(t_step, root=0)
    all_valid = comm.gather(valid, root=0)
    all_used = comm.gather(info["device_bytes_used"], root=0)
    if rank != 0:
        return None

    rt = np.array(all_rt)  # (nprocs, calls)
    rt_max = rt.max(axis=0)  # critical path per call
    rec.update({
        "device_bytes_used": all_used,
        "t_rt_per_rank_s": all_rt, "t_comm_per_rank_s": all_comm, "t_step_per_rank_s": all_step,
        "t_rt_max_median_s": float(np.median(rt_max)),
        "t_rt_mean_median_s": float(np.median(rt.mean(axis=0))),
        "imbalance": float(np.median(rt_max / rt.mean(axis=0))),
        "t_comm_median_s": float(np.median(np.array(all_comm).max(axis=0))),
        "time_median_s": float(np.median(np.array(all_step).max(axis=0))),
        "valid": all(all_valid),
    })
    rec["sources_per_s"] = num_sources / rec["time_median_s"]
    log(f"P={nprocs} N={mesh_size} src={num_sources} R={radius}: step {rec['time_median_s']:.4f} s "
        f"(raytracing {rec['t_rt_max_median_s']:.4f} s, comm {rec['t_comm_median_s']:.4f} s, "
        f"imbalance {rec['imbalance']:.3f}) over {len(rt_max)} calls, valid={rec['valid']}")
    return rec


def save(test, params, records):
    if rank == 0:
        meta_extra = {"nprocs": nprocs, "mpi": MPI.Get_library_version().split("\n")[0]}
        params = {**params, **meta_extra}
        common.save_results(test, params, [r for r in records if r is not None])
