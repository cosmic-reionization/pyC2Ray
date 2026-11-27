#!/bin/bash
#SBATCH -J dj-D4
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out

# D with --jit-segments for the two largest cases, which failed in job 5810705 with "Failed to allocate
# ... bytes for new constant": the compiled segment embeds DISCO-DJ's Lagrangian grid q (built with
# NumPy) as an XLA constant (5.4 GB at 768^3), allocated outside JAX's memory pool. With the pool
# limited to 85 % of HBM (instead of 95 %) there is room for it.
#  GPU0  512^3 PM 2x TSC + interlacing, grids 256^3, snapshots z = 10, 8, 6, 5
#  GPU1  768^3 PM 1x, --free-ics, grids 256^3, snapshot z = 5
#  GPU2  768^3 PM 1x, 1LPT ICs, --free-ics, grids 256^3 (npy only)
#  GPU3  512^3 PM 2x CIC (no TSC/interlacing), grids 256^3 (npy only)
# Submit from examples/benchmarks/nbody/disco_dj.
source envs/arrhenius_disco_dj.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85
python -c "import tools21cm" 2>/dev/null || pip install -q /nobackup/proj/disk/cosmic_reion_2026/personal/sgiri/myPackages/tools21cm
W=$DJ_RUN_ROOT/B3_D/$SLURM_JOB_ID
B="python benchmark_scripts/djbench.py"
LOOP="--outputs loop --n-steps 110 --pk-z 10 8 6 5 --repeat 0 --jit-segments"
run() {  # gpu test label args...
  local g=$1 t=$2 l=$3; shift 3
  CUDA_VISIBLE_DEVICES=$g taskset -c $((g * 72))-$((g * 72 + 71)) $B --test $t --label $l \
    --save-delta $W/$l/final_delta.npy "$@"
}
run 0 D N512_pm2_tsc_aa_jit --res 512 --res-pm 1024 --worder 3 --antialias 1 --grid 256 $LOOP \
  --write-dir $W/N512_pm2_tsc_aa_jit --snap-z 10 8 6 5 &
run 1 D N768_pm1_free_jit --res 768 --grid 256 $LOOP --free-ics --write-dir $W/N768_pm1_free_jit --snap-z 5 &
run 2 D N768_pm1_lpt1_free_jit --res 768 --lpt-order 1 --grid 256 $LOOP --free-ics \
  --write-dir $W/N768_pm1_lpt1_free_jit --grid-format npy &
run 3 D N512_pm2_jit --res 512 --res-pm 1024 --grid 256 $LOOP --write-dir $W/N512_pm2_jit --grid-format npy &
wait
for d in N512_pm2_tsc_aa_jit N768_pm1_free_jit; do
  [ -f $W/$d/redshift_density.txt ] && python benchmark_scripts/check_outputs.py $W/$d $W/$d/final_delta.npy
done
du -sh $W/*
