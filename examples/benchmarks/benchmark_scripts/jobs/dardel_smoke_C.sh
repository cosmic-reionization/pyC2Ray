#!/bin/bash
#SBATCH -J bench-smoke-C
#SBATCH -A naiss2026-4-1146
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

# Quick functional check of C1/C2 with tiny settings (results in smoke_results/).
source envs/dardel_hip.sh || { echo "submit from examples/benchmarks"; exit 1; }
export BENCH_SYSTEM=dardel-smoke BENCH_RESULTS_DIR=$PWD/smoke_results
MAP=49,57,17,25,1,9,33,41
set -e
for np in 1 2 8; do
    bind=--cpu-bind=map_cpu:$(echo $MAP | cut -d, -f1-$np)
    srun -n $np --gpus-per-task=1 $bind python benchmark_scripts/C1_strong_scaling.py --mesh-size 64 --cases 15:200 30:100 --repeats 2
    srun -n $np --gpus-per-task=1 $bind python benchmark_scripts/C2_weak_scaling.py --mesh-size 64 --sources-per-rank 50 --repeats 2
done
srun -n 2 --gpus-per-task=1 --cpu-bind=map_cpu:49,57 bash -c 'echo "rank $SLURM_PROCID ROCR=$ROCR_VISIBLE_DEVICES cpu $(taskset -pc $$ | cut -d: -f2)"'
