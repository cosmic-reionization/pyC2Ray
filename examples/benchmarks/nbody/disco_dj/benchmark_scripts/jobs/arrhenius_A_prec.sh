#!/bin/bash
#SBATCH -J dj-A-prec
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out

# A: single vs. double precision, time and peak memory vs. N (50 BullFrog steps, PM 1x).
# Repeat of the djbench part of job 5809172, whose records were overwritten by a concurrent writer
# (fixed: result files are now appended under a lock). One N per GPU, single then double.
# Submit from examples/benchmarks/nbody/disco_dj.
source envs/arrhenius_disco_dj.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
g=0
for Ns in "128 640" "256" "384" "512"; do
  (
    export CUDA_VISIBLE_DEVICES=$g
    for N in $Ns; do for p in single double; do
      taskset -c $((g * 72))-$((g * 72 + 71)) python benchmark_scripts/djbench.py --test A_prec \
        --label N${N}_$p --res $N --precision $p --n-steps 50 --repeat 2
    done; done
  ) &
  g=$((g + 1))
done
wait
