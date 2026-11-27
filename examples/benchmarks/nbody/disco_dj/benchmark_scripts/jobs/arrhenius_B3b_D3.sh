#!/bin/bash
#SBATCH -J dj-B3b-D3
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out

# B3/D with --jit-segments: one output interval compiled once (scale factors traced). In job 5809173
# every run_nbody call was traced and compiled again (~0.7 s per output at 256^3, ~2 s at 512^3).
#  GPU0  B3: 256^3 PM 1x (compare P(k) with the uncompiled loop of 5809173, same seed); 2 steps/output
#  GPU1  B3: 512^3 PM 1x, grids 256^3
#  GPU2  D:  512^3 PM 2x TSC + interlacing, grids 256^3, snapshots z = 10, 8, 6, 5
#  GPU3  D:  768^3 PM 1x with --free-ics, grids 256^3, snapshot z = 5; then 640^3 PM 1x
# Submit from examples/benchmarks/nbody/disco_dj.
source envs/arrhenius_disco_dj.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
W=$DJ_RUN_ROOT/B3_D/$SLURM_JOB_ID
B="python benchmark_scripts/djbench.py"
LOOP="--outputs loop --n-steps 110 --pk-z 10 8 6 5 --repeat 0 --jit-segments"
run() {  # gpu cores test label args...
  local g=$1 c=$2 t=$3 l=$4; shift 4
  CUDA_VISIBLE_DEVICES=$g taskset -c $c $B --test $t --label $l --save-delta $W/$l/final_delta.npy "$@"
}
(
  run 0 0-71 B3 N256_pm1_grid256_jit --res 256 --grid 256 $LOOP --write-dir $W/N256_pm1_grid256_jit --grid-format npy
  run 0 0-71 B3 N256_pm1_spo2_jit --res 256 --grid 256 $LOOP --n-steps 210 --steps-per-output 2
) &
(
  run 1 72-143 B3 N512_pm1_grid256_jit --res 512 --grid 256 $LOOP --write-dir $W/N512_pm1_grid256_jit --grid-format npy
) &
(
  run 2 144-215 D N512_pm2_tsc_aa_jit --res 512 --res-pm 1024 --worder 3 --antialias 1 --grid 256 $LOOP \
    --write-dir $W/N512_pm2_tsc_aa_jit --snap-z 10 8 6 5
) &
(
  run 3 216-287 D N768_pm1_free_jit --res 768 --grid 256 $LOOP --free-ics --write-dir $W/N768_pm1_free_jit --snap-z 5
  run 3 216-287 D N640_pm1_jit --res 640 --grid 256 $LOOP --write-dir $W/N640_pm1_jit --grid-format npy --snap-z 5
) &
wait
for d in N512_pm2_tsc_aa_jit N768_pm1_free_jit; do  # both formats only here
  [ -f $W/$d/redshift_density.txt ] && python benchmark_scripts/check_outputs.py $W/$d $W/$d/final_delta.npy
done
du -sh $W/*
