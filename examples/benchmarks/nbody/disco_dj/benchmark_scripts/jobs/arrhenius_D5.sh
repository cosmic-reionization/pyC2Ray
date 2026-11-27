#!/bin/bash
#SBATCH -J dj-D5
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 00:45:00
#SBATCH -o logs/%x-%j.out

# D: 768^3 PM 1x with --jit-segments. With the JAX pool at 95 % of HBM the compiled segment's constant
# (Lagrangian grid, 5.4 GB) cannot be allocated (job 5810705); at 85 % the run itself does not fit
# (job 5810858). Try 90 % and 92 % (2LPT and 1LPT ICs). Timing only, no outputs written.
# Submit from examples/benchmarks/nbody/disco_dj.
source envs/arrhenius_disco_dj.sh
B="python benchmark_scripts/djbench.py --test D --res 768 --grid 256 --outputs loop --n-steps 110 --pk-z 10 5 --repeat 0 --jit-segments --free-ics"
run() { local g=$1 f=$2 l=$3; shift 3
  XLA_PYTHON_CLIENT_MEM_FRACTION=$f CUDA_VISIBLE_DEVICES=$g taskset -c $((g * 72))-$((g * 72 + 71)) $B --label $l "$@"; }
run 0 0.90 N768_pm1_free_jit_mf090 &
run 1 0.92 N768_pm1_free_jit_mf092 &
run 2 0.90 N768_pm1_lpt1_free_jit_mf090 --lpt-order 1 &
run 3 0.92 N768_pm1_lpt1_free_jit_mf092 --lpt-order 1 &
wait
