#!/bin/bash
#SBATCH -J bench-C-multi
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 2
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4

# C1 (strong) and C2 (weak scaling) on all GPUs of the allocated nodes
# (default 2 nodes = 8 GPUs; e.g. sbatch -N 4 for 16), one MPI rank per GPU.
# Same GPU and CPU binding per node as arrhenius_C.sh.
source envs/arrhenius_cuda.sh || { echo "submit from examples/benchmarks"; exit 1; }
MAP=0,72,144,216
np=$((SLURM_JOB_NUM_NODES * 4))
# default: N=256 with each script's default cases, and N=100 with 10^4 sources at
# R=15; with arguments, C1 and C2 once with those arguments
if [ $# -gt 0 ]; then runs=("$*"); else runs=("" "--mesh-size 100 --cases 15:10000"); fi
for run in "${runs[@]}"; do
for test in C1_strong_scaling C2_weak_scaling; do
    echo "=== $test $run P=$np on $SLURM_JOB_NUM_NODES nodes ($(date +%T))"
    srun -n $np --ntasks-per-node=4 --gpus-per-task=1 --cpu-bind=map_cpu:$MAP \
        python benchmark_scripts/$test.py $run || echo "=== FAILED: $test P=$np"
done
done
