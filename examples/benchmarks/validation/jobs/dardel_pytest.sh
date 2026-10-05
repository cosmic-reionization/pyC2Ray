#!/bin/bash
#SBATCH -J pyc2ray-pytest
#SBATCH -A naiss2026-4-1146
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 00:20:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=1

# pyC2Ray's test suite (tests/) against the installed build on one MI250X GCD.
# PYC2RAY_SRC: checkout whose tests/ to run (default: this repository).
source envs/dardel_hip.sh || { echo "submit from examples/benchmarks"; exit 1; }
cd "${PYC2RAY_SRC:-../..}"
echo "commit: $(git rev-parse --short HEAD) $(git status --porcelain | wc -l) uncommitted changes"
rocminfo | grep -m1 -oE "gfx[0-9a-f]+"
# 'pytest' (not 'python -m pytest') so the installed package is imported, not ./pyc2ray
pytest tests -v -rs --benchmark-disable
