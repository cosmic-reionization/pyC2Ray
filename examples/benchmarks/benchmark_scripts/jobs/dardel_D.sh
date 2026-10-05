#!/bin/bash
#SBATCH -J bench-D
#SBATCH -A naiss2026-4-1146
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1

# D: end-to-end evolve3D timestep (GPU raytracing + CPU chemistry) on one
# MI250X GCD and one CPU core (the Fortran chemistry is not threaded).
source envs/dardel_hip.sh || { echo "submit from examples/benchmarks"; exit 1; }
python benchmark_scripts/D_timestep.py "$@" 2>&1 | grep -v "allocated\|Deallocating\|selected ID\|GPU Device"
