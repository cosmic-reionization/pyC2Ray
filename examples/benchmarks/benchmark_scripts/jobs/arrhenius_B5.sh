#!/bin/bash
#SBATCH -J bench-B5
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1

# B5: host<->device transfer overhead on one GH200 GPU.
source envs/arrhenius_cuda.sh || { echo "submit from examples/benchmarks"; exit 1; }
python benchmark_scripts/B5_transfer.py "$@"
