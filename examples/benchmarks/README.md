# pyC2Ray GPU benchmarks

Benchmarks of pyC2Ray's GPU raytracing (ASORA) on NAISS systems, tracked in
issue [#43](https://github.com/cosmic-reionization/pyC2Ray/issues/43) with one
sub-issue per system: [#44](https://github.com/cosmic-reionization/pyC2Ray/issues/44)
Dardel-GPU (AMD MI250X, HIP) and [#45](https://github.com/cosmic-reionization/pyC2Ray/issues/45)
Arrhenius (NVIDIA GH200, CUDA and HIP).

**For details on the Dardel-GPU benchmarks see
[#44](https://github.com/cosmic-reionization/pyC2Ray/issues/44)**: build and
branch, checklist, progress log with the findings and open problems. Test
definitions and comparisons between systems are in
[#43](https://github.com/cosmic-reionization/pyC2Ray/issues/43).

## Results

| Notebook | Contents |
|---|---|
| [`pyc2ray_gpu_validation_dardel.ipynb`](pyc2ray_gpu_validation_dardel.ipynb) | **A** on Dardel-GPU: correctness gate, I-front, shadow and multi-source tests with GPU raytracing |
| [`pyc2ray_benchmarks_dardel.ipynb`](pyc2ray_benchmarks_dardel.ipynb) | **B–E** on Dardel-GPU: plots and tables |
| [`pyc2ray_gpu_validation_arrhenius.ipynb`](pyc2ray_gpu_validation_arrhenius.ipynb) | **A** on Arrhenius: correctness gate, native CUDA build on a GH200 |
| [`pyc2ray_benchmarks_arrhenius.ipynb`](pyc2ray_benchmarks_arrhenius.ipynb) | **B, C** on Arrhenius (native CUDA): plots and tables; D and E to do |

The notebooks are stored with their outputs, so the figures show on GitHub.
The benchmark notebooks only read `benchmark_results/`; for a test without
results they print which script to run.

## Tests

- **A. Correctness gate** — `pyc2ray_gpu_validation_<system>.ipynb`; in addition every
  benchmark point checks that the ASORA output is finite and nonzero.
- **B. Single-GPU microbenchmarks** (one `asora.do_all_sources` call)
  - B1 grid size N, B2 number of sources, B3 raytracing radius R,
    B4 `source_batch_size`, B5 host↔device transfer overhead
- **C. Multi-GPU / multi-node scaling** (pyC2Ray's MPI mode: sources split over
  ranks, rates summed with MPI Reduce + Bcast)
  - C1 strong scaling, C2 weak scaling
- **D. End-to-end timestep** — GPU raytracing vs. CPU chemistry (not yet implemented)
- **E. Roofline characterization** — achieved vs. peak memory bandwidth (not yet implemented)

All tests use synthetic problems (uniform density, randomly placed sources), so
no input data is needed.

## Layout

```
pyc2ray_gpu_validation_<system>.ipynb  A, one per system
pyc2ray_benchmarks_<system>.ipynb      B-E, one per system
benchmark_scripts/                     one script per test (B1_grid_size.py, ...), common.py, mpi_scaling.py
benchmark_scripts/jobs/                Slurm job scripts, <system>_<test>.sh
benchmark_results/<system>/<test>/<test>_<slurm job id>.json
validation/                            parameter and source files of the validation notebook, its job scripts
envs/<system>_<build>.sh               environment sourced by the job scripts
logs/                                  Slurm logs (not tracked)
```

Each result file holds the parameters, the measured records and the metadata of
the run: system, GPU, build (HIP/CUDA), runtime version, Slurm job, and the
pyC2Ray source, branch and commit it was built from.

## Running

All commands are run from this folder. On Dardel:

```bash
export PYC2RAY_VENV=/path/to/venv     # virtualenv with pyC2Ray built for HIP, see envs/dardel_hip.sh
sbatch benchmark_scripts/jobs/dardel_smoke_B.sh   # quick functional check (results in smoke_results/)
sbatch benchmark_scripts/jobs/dardel_B1.sh        # one test; extra arguments go to the script
sbatch benchmark_scripts/jobs/dardel_C.sh         # C1 and C2 on 1, 2, 4, 8 GCDs of one node
sbatch -N 4 benchmark_scripts/jobs/dardel_C_multinode.sh
sbatch validation/jobs/dardel_validation_notebook.sh
```

On Arrhenius (the GPU modules only work on the aarch64 GPU nodes, so build and run through Slurm;
see `envs/arrhenius_cuda.sh` for the build recipe):

```bash
export PYC2RAY_VENV=/path/to/venv     # virtualenv with pyC2Ray built for CUDA (sm_90) and mpi4py built against MPICH
sbatch benchmark_scripts/jobs/arrhenius_smoke_B.sh   # and arrhenius_smoke_C.sh
sbatch benchmark_scripts/jobs/arrhenius_B1.sh        # B1-B5, one GH200 GPU each
sbatch benchmark_scripts/jobs/arrhenius_C.sh         # C1 and C2 on 1, 2, 4 GPUs of one node
sbatch -N 2 benchmark_scripts/jobs/arrhenius_C_multinode.sh   # P = 8; -N 4 for P = 16
sbatch benchmark_scripts/jobs/arrhenius_notebook.sh  # execute pyc2ray_benchmarks_arrhenius.ipynb in place
sbatch validation/jobs/arrhenius_validation_notebook.sh
```

Then re-execute the results notebook of the system (no GPU needed; it only reads `benchmark_results/`,
so it must be executed again after new results arrive), e.g. `pyc2ray_benchmarks_dardel.ipynb`. The Dardel
job scripts charge the NAISS allocation `naiss2026-4-1146` and the Arrhenius ones `naiss2026-4-1146-gpu`;
use `sbatch -A` to charge another one. Note that Dardel's `gpu` partition allocates whole nodes,
so a single-GCD job is billed for all 8 GCDs (on Arrhenius a single GPU is billed on its own).

A new system needs `envs/<system>_<build>.sh`, job scripts
`benchmark_scripts/jobs/<system>_*.sh` that export `BENCH_SYSTEM=<system>`, and a
copy of the results notebook with `SYSTEM` set in its first code cell, and a copy of
the validation notebook, `pyc2ray_gpu_validation_<system>.ipynb`.
