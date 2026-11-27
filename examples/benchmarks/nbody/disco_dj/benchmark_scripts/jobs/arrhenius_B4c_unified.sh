#!/bin/bash
#SBATCH -J dj-B4c
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out

# B4c: can the GH200's coherent NVLink-C2C host memory (120 GB LPDDR5X per GPU) host runs larger than
# the 96 GB HBM? XLA's BFC allocator on CUDA managed (unified) memory: TF_FORCE_UNIFIED_MEMORY=1 with
# XLA_PYTHON_CLIENT_MEM_FRACTION > 1 (fraction of HBM). 768^3 measures the slowdown against B1/B4 in
# HBM only; 896^3, 1024^3 (1LPT) and 640^3 with PM 2x ran out of memory in B4.
# Two GPUs only, so the concurrent host-memory use stays below the node's 480 GB.
# Submit from examples/benchmarks/nbody/disco_dj.
source envs/arrhenius_disco_dj.sh
export TF_FORCE_UNIFIED_MEMORY=1 XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=2.0
B="python benchmark_scripts/djbench.py --test B4 --n-steps 50 --repeat 1"
(
  export CUDA_VISIBLE_DEVICES=0
  timeout 40m taskset -c 0-71 $B --label N768_unified --res 768
  timeout 60m taskset -c 0-71 $B --label N896_unified --res 896
) &
(
  export CUDA_VISIBLE_DEVICES=1
  timeout 40m taskset -c 72-143 $B --label N640_pm2_unified --res 640 --res-pm 1280
  timeout 60m taskset -c 72-143 $B --label N1024_lpt1_unified --res 1024 --lpt-order 1
) &
wait
