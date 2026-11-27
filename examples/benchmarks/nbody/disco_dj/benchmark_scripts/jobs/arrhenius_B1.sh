#!/bin/bash
#SBATCH -J dj-B1
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out

# B1: time and peak GPU memory vs. N_particles on one GH200 (96 GB HBM), PM grid 1x and 2x the
# particle grid. z = 150 -> 5, 50 BullFrog steps (D time), final state only. One process per point
# (peak memory per run); OOM is recorded as a data point. The node is billed whole, so the two
# series run side by side, each on its own GPU (CUDA_VISIBLE_DEVICES) with 72 Grace cores.
# Submit from examples/benchmarks/nbody/disco_dj.
source envs/arrhenius_disco_dj.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
nvidia-smi -L
B="python benchmark_scripts/djbench.py --test B1 --n-steps 50 --repeat 2"
(
  export CUDA_VISIBLE_DEVICES=0
  for N in 128 256 384 512 640 768 896 1024 1152 1280; do
    taskset -c 0-71 $B --label N${N}_pm1 --res $N --res-pm $N
  done
) &
(
  export CUDA_VISIBLE_DEVICES=1
  for N in 128 256 384 512 640 768; do
    taskset -c 72-143 $B --label N${N}_pm2 --res $N --res-pm $((2 * N))
  done
) &
wait
