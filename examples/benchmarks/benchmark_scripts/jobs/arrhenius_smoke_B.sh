#!/bin/bash
#SBATCH -J bench-smoke-B
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 00:15:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1

# Quick functional check of the B scripts with tiny settings (results go to
# smoke_results/, not benchmark_results/).
source envs/arrhenius_cuda.sh || { echo "submit from examples/benchmarks"; exit 1; }
export BENCH_SYSTEM=arrhenius-smoke BENCH_RESULTS_DIR=$PWD/smoke_results
set -e
python benchmark_scripts/B1_grid_size.py --mesh-sizes 32 64 --repeats 2
python benchmark_scripts/B2_num_sources.py --mesh-size 64 --num-sources 10 100 --repeats 2
python benchmark_scripts/B3_radius.py --mesh-size 64 --num-sources 20 50 --radii 5 15 box --repeats 2
python benchmark_scripts/B4_batch_size.py --mesh-size 64 --num-sources 20 100 --batch-sizes 1 8 --repeats 2
python benchmark_scripts/B5_transfer.py --mesh-sizes 32 64 --num-sources 20 50 --repeats 2
