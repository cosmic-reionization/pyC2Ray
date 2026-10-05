#!/bin/bash
#SBATCH -J bench-B2
#SBATCH -A naiss2026-4-1146
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1

# B2: source-count scaling on one MI250X GCD.
source envs/dardel_hip.sh || { echo "submit from examples/benchmarks"; exit 1; }
if [ $# -gt 0 ]; then
    python benchmark_scripts/B2_num_sources.py "$@"
else
    python benchmark_scripts/B2_num_sources.py                               # N=256, R=15 and 30
    python benchmark_scripts/B2_num_sources.py --mesh-size 100 --radii 15    # N=100 (non-power-of-2), R=15
fi
