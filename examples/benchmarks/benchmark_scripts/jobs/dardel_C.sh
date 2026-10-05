#!/bin/bash
#SBATCH -J bench-C
#SBATCH -A naiss2026-4-1146
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=8

# C1 (strong) and C2 (weak scaling) on 1, 2, 4, 8 GCDs of one Dardel GPU node,
# one MPI rank per GCD. --gpus-per-task=1 gives each rank its own GCD (do not set
# CUDA_VISIBLE_DEVICES on AMD: HIP applies it after ROCR_VISIBLE_DEVICES).
# CPU binding: each rank on a core of the CCD closest to its GCD (MI250X/EPYC
# Trento topology, as on LUMI-G); GCD k -> core MAP[k].
source envs/dardel_hip.sh || { echo "submit from examples/benchmarks"; exit 1; }
MAP=49,57,17,25,1,9,33,41
for test in C1_strong_scaling C2_weak_scaling; do
    for np in 1 2 4 8; do
        echo "=== $test P=$np ($(date +%T))"
        srun -n $np --gpus-per-task=1 --cpu-bind=map_cpu:$(echo $MAP | cut -d, -f1-$np) \
            python benchmark_scripts/$test.py "$@" || echo "=== FAILED: $test P=$np"
    done
done
