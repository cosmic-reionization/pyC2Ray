#!/bin/bash
#SBATCH -J bench-B4
#SBATCH -A naiss2026-4-1146
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1

# B4: source_batch_size sweep on one MI250X GCD.
source envs/dardel_hip.sh || { echo "submit from examples/benchmarks"; exit 1; }
if [ $# -gt 0 ]; then
    python benchmark_scripts/B4_batch_size.py "$@"
else
    python benchmark_scripts/B4_batch_size.py                                 # N=256, 100 and 10^4 sources
    python benchmark_scripts/B4_batch_size.py --mesh-size 512 --num-sources 100  # batch <= 32 fits on a GCD
    python benchmark_scripts/B4_batch_size.py --mesh-size 100 --num-sources 100  # non-power-of-2 grid
fi
