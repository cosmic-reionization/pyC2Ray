# Environment for running pyC2Ray with native CUDA on Arrhenius (NSC):
# 4x NVIDIA GH200 per node (aarch64 Grace CPU, sm_90), one GPU per MPI rank.
# Sourced by the job scripts, which are submitted from examples/benchmarks, e.g.
#   export PYC2RAY_VENV=/path/to/venv      # virtualenv with pyC2Ray built for CUDA
#   sbatch benchmark_scripts/jobs/arrhenius_B1.sh
# The login node is x86_64 and cannot load/run the GPU modules: build and run on a GPU node.
# Drop the x86 miniforge from PATH (it cannot run on the aarch64 GPU nodes)
export PATH=$(echo "$PATH" | tr : '\n' | grep -v miniforge | paste -sd:)
module load GPU/buildenv-gcccuda/2026.03-cu13.0 >/dev/null 2>&1
module load GPU/Meson/1.8.2-eb >/dev/null 2>&1
module load GPU/Python/3.13.5-bundle-SciPy-2025.07-mpi4py-4.1.0-gcc-2025b-eb >/dev/null 2>&1
export CC=gcc CXX=g++ FC=gfortran
# The Python module's mpi4py is built against Open MPI, which hangs under srun with the
# MPICH of the buildenv: drop it from PYTHONPATH (the venv has mpi4py built against MPICH)
export PYTHONPATH=$(echo "$PYTHONPATH" | tr : '\n' | grep -vi mpi4py | paste -sd:)
# MPICH needs PMI2 for srun to wire the ranks (otherwise every rank is a singleton, size 1)
export SLURM_MPI_TYPE=pmi2
: "${PYC2RAY_VENV:?set PYC2RAY_VENV to the virtualenv with pyC2Ray built for CUDA}"
source "$PYC2RAY_VENV/bin/activate"
export BENCH_SYSTEM=arrhenius

# Building pyC2Ray for CUDA, on a GPU node (from a checkout of the `benchmark` branch):
#   salloc -A naiss2026-4-1146-gpu -p gpu --gpus=1 -t 01:00:00
#   (load the modules above, then)
#   python -m venv --system-site-packages $PYC2RAY_VENV && source $PYC2RAY_VENV/bin/activate
#   pip install ninja meson-python charset-normalizer astropy h5py tools21cm pyyaml
#   pip install --no-cache-dir --no-binary mpi4py --no-build-isolation --no-deps mpi4py==4.0.3   # against MPICH
#   pip install --no-build-isolation --no-deps . -Csetup-args=-Dgpu-architecture=90
