#!/bin/bash
#SBATCH -J dj-C
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out

# C: throughput per node. k = 1, 2, 4 independent runs at the same time, one per GPU (+72 Grace cores),
# for a small (256^3, PM 2x) and a large (512^3, PM 1x; 768^3, PM 1x) case, 50 steps, 3 repeats.
# Interference = run time with k concurrent runs / run time alone. Labels: <case>_k<k>_gpu<i>.
# Wall time per process (incl. Python start, setup, compile) is in the log (WALL lines).
# Submit from examples/benchmarks/nbody/disco_dj.
source envs/arrhenius_disco_dj.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
one() {  # gpu case k args...
  local g=$1 case=$2 k=$3; shift 3
  local t0=$(date +%s)
  CUDA_VISIBLE_DEVICES=$g taskset -c $((g * 72))-$((g * 72 + 71)) python benchmark_scripts/djbench.py --test C \
    --label ${case}_k${k}_gpu$g --n-steps 50 --repeat 3 "$@"
  echo "WALL $case k=$k gpu=$g $(( $(date +%s) - t0 )) s"
}
for spec in "N256_pm2 --res 256 --res-pm 512" "N512_pm1 --res 512" "N768_pm1 --res 768"; do
  set -- $spec; case=$1; shift
  for k in 1 2 4; do
    for g in $(seq 0 $((k - 1))); do one $g $case $k "$@" & done
    wait
  done
done
