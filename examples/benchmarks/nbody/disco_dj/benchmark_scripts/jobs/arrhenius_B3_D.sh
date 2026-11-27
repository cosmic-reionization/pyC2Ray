#!/bin/bash
#SBATCH -J dj-B3-D
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out

# B3: cost of the 99 outputs (z = 30 -> 5 every 10.86 Myr): one run_nbody call per output interval,
#     density grid on the GPU, copy to the host, write as pyC2Ray input (.npy and pkdgrav3 raw).
# D:  production-style runs with all 99 outputs, HDF5 particle snapshots at z = 10, 8, 6, 5.
# A:  the written grids are read back the way pyC2Ray reads them (check_outputs.py).
# Each series compares with the same configuration run straight to z = 5 (final mode) via P(k).
# 110 steps in total: 12 from z = 150 to 30, then one per output interval.
# Submit from examples/benchmarks/nbody/disco_dj.
source envs/arrhenius_disco_dj.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
pip install -q --no-deps /nobackup/proj/disk/cosmic_reion_2026/personal/sgiri/myPackages/tools21cm 2>&1 | tail -1
python -c "import tools21cm" 2>/dev/null || pip install -q /nobackup/proj/disk/cosmic_reion_2026/personal/sgiri/myPackages/tools21cm
W=$DJ_RUN_ROOT/B3_D/$SLURM_JOB_ID
B="python benchmark_scripts/djbench.py"
LOOP="--outputs loop --n-steps 110 --pk-z 10 8 6 5 --repeat 0"
run() {  # gpu cores test label args...
  local g=$1 c=$2 t=$3 l=$4; shift 4
  CUDA_VISIBLE_DEVICES=$g taskset -c $c $B --test $t --label $l --save-delta $W/$l/final_delta.npy "$@"
}
(
  run 0 0-71 B3 N256_pm1_grid256 --res 256 --grid 256 $LOOP --write-dir $W/N256_pm1_grid256
  run 0 0-71 B3 N256_pm1_nowrite --res 256 --grid 256 $LOOP
  run 0 0-71 B3 N256_pm1_final --res 256 --grid 256 --n-steps 50 --repeat 1
  run 0 0-71 B3 N512_pm1_grid512 --res 512 --grid 512 $LOOP --grid-format npy --write-dir $W/N512_pm1_grid512 --n-outputs 20
) &
(
  run 1 72-143 B3 N512_pm1_grid256 --res 512 --grid 256 $LOOP --write-dir $W/N512_pm1_grid256
  run 1 72-143 B3 N512_pm1_final --res 512 --grid 256 --n-steps 50 --repeat 1
)  &
(
  P2="--res 512 --res-pm 1024 --worder 3 --antialias 1 --grid 256"
  run 2 144-215 D N512_pm2_tsc_aa $P2 $LOOP --write-dir $W/N512_pm2_tsc_aa --snap-z 10 8 6 5
  run 2 144-215 D N512_pm2_tsc_aa_final $P2 --n-steps 50 --repeat 1
) &
(
  run 3 216-287 D N768_pm1 --res 768 --grid 256 $LOOP --write-dir $W/N768_pm1 --snap-z 5
  run 3 216-287 D N768_pm1_final --res 768 --grid 256 --n-steps 50 --repeat 1
) &
wait
for d in N256_pm1_grid256 N512_pm1_grid256 N512_pm1_grid512 N512_pm2_tsc_aa N768_pm1; do
  [ -f $W/$d/redshift_density.txt ] && python benchmark_scripts/check_outputs.py $W/$d $W/$d/final_delta.npy
done
du -sh $W/*
