#!/bin/bash
#SBATCH -J dj-unified
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 02:30:00
#SBATCH -o logs/%x-%j.out

# Follow-ups of B4c (job 5834328): XLA's pool on CUDA managed memory (TF_FORCE_UNIFIED_MEMORY=1,
# XLA_PYTHON_CLIENT_MEM_FRACTION > 1 = fraction of the 96 GB HBM), so the GH200's LPDDR5X (NVLink-C2C)
# backs allocations beyond HBM. Three GPUs only, so the concurrent host-memory use stays below 480 GB.
#  GPU0  B4: 1024^3 (1LPT and 2LPT) and 768^3 PM 2x, pool 2.2 x HBM (~211 GB)
#  GPU1  D: 768^3 PM 1x with --jit-segments (failed in HBM alone: jobs 5810705, 5810858, 5811289),
#        99 grids 256^3 (npy), snapshot z = 5, read-back check (A4)
#  GPU2  A3: 512^3 PM 2x + TSC + interlacing against pkdgrav3 (OOM in HBM alone, job 5810186),
#        pkdgrav3 ICs/snapshots of job 5809171
# Submit from examples/benchmarks/nbody/disco_dj.
source envs/arrhenius_disco_dj.sh
export TF_FORCE_UNIFIED_MEMORY=1 XLA_PYTHON_CLIENT_PREALLOCATE=false
B="python benchmark_scripts/djbench.py"
W=$DJ_RUN_ROOT/B3_D/$SLURM_JOB_ID
(
  export CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=2.2
  for a in "N1024_lpt1_unified --res 1024 --lpt-order 1" "N768_pm2_unified --res 768 --res-pm 1536" \
           "N1024_unified --res 1024"; do
    timeout 40m taskset -c 0-71 $B --test B4 --n-steps 50 --repeat 1 --label $a
  done
) &
(
  export CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=1.5
  l=N768_pm1_free_jit_unified
  timeout 80m taskset -c 72-143 $B --test D --label $l --res 768 --grid 256 --outputs loop --n-steps 110 \
    --pk-z 10 8 6 5 --repeat 0 --jit-segments --free-ics --write-dir $W/$l --grid-format npy --snap-z 5 \
    --save-delta $W/$l/final_delta.npy
) &
(
  export CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_MEM_FRACTION=1.5
  R=$DJ_RUN_ROOT/A_B2/5809171
  timeout 80m taskset -c 144-215 python benchmark_scripts/compare_pkdgrav3.py --test B2 --label "unified:" \
    --pkd-ic $(ls $R/ic_za.0000* | head -1) --pkd-snap 10:$R/ref.00060 --pkd-snap 8:$R/ref.00070 \
    --pkd-snap 6:$R/ref.00080 --pkd-snap 5:$R/ref.00090 --config bullfrog:50:res_pm=1024,worder=3,antialias=1
) &
wait
df -h $DJ_RUN_ROOT | tail -1
