# Environment for running DISCO-DJ on Arrhenius (NSC) GPU nodes:
# 4x NVIDIA GH200 per node (aarch64 Grace CPU, 72 cores + 1 Hopper GPU per superchip, sm_90).
# Sourced by the job scripts, which are submitted from examples/benchmarks/nbody/disco_dj, e.g.
#   sbatch benchmark_scripts/jobs/arrhenius_smoke.sh
# The login node is x86_64: the conda env (aarch64, JAX with CUDA) only runs on the GPU nodes.
# Build it once with benchmark_scripts/jobs/arrhenius_setup.sh.

# Paths (override before submitting if needed)
: "${DJ_MINIFORGE:=/nobackup/proj/disk/cosmic_reion_2026/personal/sgiri/home_offload/miniforge3-aarch64}"
: "${DJ_CONDA_ENV:=discodj}"
: "${DISCODJ_SRC:=/nobackup/proj/disk/cosmic_reion_2026/personal/sgiri/myPackages/DISCO-DJ}"
# Large outputs (snapshots, grids) go here, not into the repository
: "${DJ_RUN_ROOT:=/nobackup/proj/disk/cosmic_reion_2026/personal/sgiri/projects/matched_ic_nbody/bench_runs_discodj}"
export DJ_MINIFORGE DJ_CONDA_ENV DISCODJ_SRC DJ_RUN_ROOT

# Drop the x86 miniforge from PATH and any conda state inherited from the login node
export PATH=$(echo "$PATH" | tr : '\n' | grep -v miniforge | paste -sd:)
unset CONDA_EXE CONDA_PREFIX CONDA_SHLVL CONDA_DEFAULT_ENV CONDA_PYTHON_EXE MAMBA_EXE MAMBA_ROOT_PREFIX PYTHONPATH
source "$DJ_MINIFORGE/etc/profile.d/conda.sh"
conda activate "$DJ_CONDA_ENV" 2>/dev/null || echo "conda env $DJ_CONDA_ENV not found (run arrhenius_setup.sh)"

# JAX/XLA memory: default preallocation is 75 % of HBM. The benchmarks set the fraction
# explicitly and record it; peak memory is read from jax's allocator stats.
: "${XLA_PYTHON_CLIENT_PREALLOCATE:=false}"
export XLA_PYTHON_CLIENT_PREALLOCATE
export BENCH_SYSTEM=arrhenius
export BENCH_BUILD=jax-cuda12

# Arrhenius allocates (and bills) whole GPU nodes: the batch step sees all 4 GH200s even with
# --gpus-per-node=1. Pin each benchmark process to one GPU (job scripts set it per process);
# default GPU 0 so JAX does not open contexts on all four.
: "${CUDA_VISIBLE_DEVICES:=0}"
export CUDA_VISIBLE_DEVICES
export TF_CPP_MIN_LOG_LEVEL=2   # silences the per-kernel "No SoL config for GH200" warnings
