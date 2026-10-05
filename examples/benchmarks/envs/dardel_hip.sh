# Environment for running pyC2Ray with the HIP backend on Dardel-GPU (PDC):
# AMD MI250X, one GCD = one GPU, 8 per node. Sourced by the job scripts, which
# are submitted from examples/benchmarks, e.g.
#   export PYC2RAY_VENV=/path/to/venv      # virtualenv with pyC2Ray built for HIP
#   sbatch benchmark_scripts/jobs/dardel_B1.sh
ml PrgEnv-gnu rocm/7.0.2 >/dev/null 2>&1
export CC=gcc CXX=g++ FC=gfortran
: "${PYC2RAY_VENV:?set PYC2RAY_VENV to the virtualenv with pyC2Ray built for HIP}"
source "$PYC2RAY_VENV/bin/activate"
export BENCH_SYSTEM=dardel

# Building pyC2Ray for HIP (from a checkout of the HIP-enabled branch):
#   python -m venv $PYC2RAY_VENV && source envs/dardel_hip.sh
#   MPICC="cc -shared" pip install mpi4py==4.0.3 --no-binary mpi4py   # against cray-mpich
#   pip install . -Csetup-args=--native-file=$PWD/hipnative.ini
