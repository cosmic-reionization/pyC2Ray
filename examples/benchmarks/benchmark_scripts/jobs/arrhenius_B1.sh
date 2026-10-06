#!/bin/bash
#SBATCH -J bench-B1
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1

# B1: grid-size scaling on one GH200 GPU.
source envs/arrhenius_cuda.sh || { echo "submit from examples/benchmarks"; exit 1; }
if [ $# -gt 0 ]; then
    python benchmark_scripts/B1_grid_size.py "$@"
else
    python benchmark_scripts/B1_grid_size.py                                       # 100 sources, R=15, 30, box
    python benchmark_scripts/B1_grid_size.py --num-sources 10000 --radii 15 30     # box radius too slow for 10^4
fi
