#!/bin/bash
#SBATCH -J bench-C
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4

# C1 (strong) and C2 (weak scaling) on 1, 2, 4 GPUs of one Arrhenius GPU node,
# one MPI rank per GPU. --gpus-per-task=1 gives each rank its own GPU.
# CPU binding: each rank on the first core of the Grace CPU of its superchip
# (4 NUMA domains of 72 cores per node; GPU k <-> cores 72k..72k+71).
source envs/arrhenius_cuda.sh || { echo "submit from examples/benchmarks"; exit 1; }
MAP=0,72,144,216
# default: N=256 with each script's default cases, and N=100 with 10^4 sources at
# R=15; with arguments, C1 and C2 once with those arguments
if [ $# -gt 0 ]; then runs=("$*"); else runs=("" "--mesh-size 100 --cases 15:10000"); fi
for run in "${runs[@]}"; do
for test in C1_strong_scaling C2_weak_scaling; do
    for np in 1 2 4; do
        echo "=== $test $run P=$np ($(date +%T))"
        srun -n $np --gpus-per-task=1 --cpu-bind=map_cpu:$(echo $MAP | cut -d, -f1-$np) \
            python benchmark_scripts/$test.py $run || echo "=== FAILED: $test P=$np"
    done
done
done
