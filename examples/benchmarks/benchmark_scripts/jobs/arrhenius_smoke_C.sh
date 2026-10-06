#!/bin/bash
#SBATCH -J bench-smoke-C
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=4

# Quick functional check of C1/C2 with tiny settings (results in smoke_results/).
source envs/arrhenius_cuda.sh || { echo "submit from examples/benchmarks"; exit 1; }
export BENCH_SYSTEM=arrhenius-smoke BENCH_RESULTS_DIR=$PWD/smoke_results
MAP=0,72,144,216
set -e
for np in 1 2 4; do
    bind=--cpu-bind=map_cpu:$(echo $MAP | cut -d, -f1-$np)
    srun -n $np --gpus-per-task=1 $bind python benchmark_scripts/C1_strong_scaling.py --mesh-size 64 --cases 15:200 30:100 --repeats 2
    srun -n $np --gpus-per-task=1 $bind python benchmark_scripts/C2_weak_scaling.py --mesh-size 64 --cases 15:50 --repeats 2
done
srun -n 2 --gpus-per-task=1 --cpu-bind=map_cpu:0,72 bash -c 'echo "rank $SLURM_PROCID CUDA=$CUDA_VISIBLE_DEVICES cpu $(taskset -pc $$ | cut -d: -f2)"'
