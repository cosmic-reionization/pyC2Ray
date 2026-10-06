#!/bin/bash
#SBATCH -J bench-notebook
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 00:10:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1

# Execute pyc2ray_benchmarks_arrhenius.ipynb in place (plots and tables from
# benchmark_results/arrhenius/). Needs no GPU; the GPU partition is used because the
# aarch64 venv cannot run on the x86 login node.
source envs/arrhenius_cuda.sh || { echo "submit from examples/benchmarks"; exit 1; }
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
    pyc2ray_benchmarks_arrhenius.ipynb
