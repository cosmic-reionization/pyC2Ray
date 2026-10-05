#!/bin/bash
#SBATCH -J pyc2ray-cpu-mult
#SBATCH -A naiss2026-4-1146
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1
#SBATCH -c 16

# Multi-source test 3a with CPU (Fortran) raytracing, as cell-by-cell reference
# for the GPU run in pyc2ray_gpu_validation.ipynb (writes test_results_mult_equal_cpu/).
# (gpu partition only because the allocation is a GPU one; the GPU is unused)
source envs/dardel_hip.sh || { echo "submit from examples/benchmarks"; exit 1; }
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
time python validation/run_multi_sources.py validation/parameters_mult_equal_cpu.yml validation/src_mult_equal.txt
