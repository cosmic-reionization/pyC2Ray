#!/bin/bash
#SBATCH -J dj-B4
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out

# B4: push past 768^3 on one GH200 (96 GB HBM). B1 found that 896^3 runs out of memory while building
# the 2LPT ICs. Variants (one GPU each, one process per point):
#  GPU0  1LPT ICs instead of 2LPT                               (memory of with_lpt)
#  GPU1  chunk_size for the PM mass assignment/interpolation     (memory of the time stepping)
#  GPU2  XLA allocator: mem fraction 1.0, and the "platform" allocator (no BFC pool)
#  GPU3  1LPT + chunk_size, larger N; and PM 2x with chunking
# Submit from examples/benchmarks/nbody/disco_dj.
source envs/arrhenius_disco_dj.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
B="python benchmark_scripts/djbench.py --test B4 --n-steps 50 --repeat 1"
(
  export CUDA_VISIBLE_DEVICES=0
  for N in 768 896 1024; do taskset -c 0-71 $B --label N${N}_lpt1 --res $N --lpt-order 1; done
) &
(
  export CUDA_VISIBLE_DEVICES=1
  for c in 16777216 67108864; do
    taskset -c 72-143 $B --label N768_chunk$c --res 768 --chunk-size $c
    taskset -c 72-143 $B --label N896_chunk$c --res 896 --chunk-size $c
  done
) &
(
  export CUDA_VISIBLE_DEVICES=2
  XLA_PYTHON_CLIENT_MEM_FRACTION=1.0 taskset -c 144-215 $B --label N896_memfrac1 --res 896
  XLA_PYTHON_CLIENT_ALLOCATOR=platform taskset -c 144-215 $B --label N768_platform --res 768
  XLA_PYTHON_CLIENT_ALLOCATOR=platform taskset -c 144-215 $B --label N896_platform --res 896
) &
(
  export CUDA_VISIBLE_DEVICES=3
  for N in 1024 1152; do taskset -c 216-287 $B --label N${N}_lpt1_chunk --res $N --lpt-order 1 --chunk-size 16777216; done
  taskset -c 216-287 $B --label N640_pm2_chunk --res 640 --res-pm 1280 --chunk-size 16777216
  taskset -c 216-287 $B --label N768_pm2_lpt1_chunk --res 768 --res-pm 1536 --lpt-order 1 --chunk-size 16777216
) &
wait
