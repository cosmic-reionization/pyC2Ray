#!/bin/bash
#SBATCH -J pyc2ray-validation
#SBATCH -A naiss2026-4-1146
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1

# Execute pyc2ray_gpu_validation.ipynb (in place, with all plots) with GPU
# raytracing on one MI250X GCD. Section 3a's CPU comparison needs
# validation/jobs/dardel_multi_sources_cpu.sh to have run first.
source envs/dardel_hip.sh || { echo "submit from examples/benchmarks"; exit 1; }
python -c "import sys; sys.path.insert(0, 'benchmark_scripts'); import common; print(common.metadata()['pyc2ray'])"
time jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.timeout=-1 --ExecutePreprocessor.kernel_name=python3 \
    pyc2ray_gpu_validation.ipynb
