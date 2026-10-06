#!/bin/bash
#SBATCH -J bench-B3
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 02:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1

# B3: raytracing-radius scaling on one GH200 GPU. With 10^4 sources the box
# radius alone takes ~18 min per call.
source envs/arrhenius_cuda.sh || { echo "submit from examples/benchmarks"; exit 1; }
if [ $# -gt 0 ]; then
    python benchmark_scripts/B3_radius.py "$@"
else
    python benchmark_scripts/B3_radius.py                                     # N=256, 100/1000/10^4 sources
    python benchmark_scripts/B3_radius.py --mesh-size 100 --num-sources 100   # non-power-of-2 grid
fi
