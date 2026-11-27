#!/bin/bash
#SBATCH -J dj-B4b-D2
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out

# Follow-up to B4 and D (jobs 5809174, 5809173):
#  GPU0  D: 768^3 PM 1x, 99 outputs, with --free-ics (the 768^3 loop ran out of memory with the ICs and
#        2LPT fields still allocated)
#  GPU1  D: 640^3 PM 1x, 99 outputs (without --free-ics)
#  GPU2  B4: largest N between 768^3 and 896^3, 2LPT ICs (final state, 50 steps)
#  GPU3  B4: the same with 1LPT ICs; then D: 768^3 with 1LPT ICs and --free-ics
# Submit from examples/benchmarks/nbody/disco_dj.
source envs/arrhenius_disco_dj.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
python -c "import tools21cm" 2>/dev/null || pip install -q /nobackup/proj/disk/cosmic_reion_2026/personal/sgiri/myPackages/tools21cm
W=$DJ_RUN_ROOT/B3_D/$SLURM_JOB_ID
B="python benchmark_scripts/djbench.py"
LOOP="--outputs loop --n-steps 110 --pk-z 10 8 6 5 --repeat 0"
(
  export CUDA_VISIBLE_DEVICES=0
  taskset -c 0-71 $B --test D --label N768_pm1_free --res 768 --grid 256 $LOOP --free-ics \
    --write-dir $W/N768_pm1_free --snap-z 5 --save-delta $W/N768_pm1_free/final_delta.npy
) &
(
  export CUDA_VISIBLE_DEVICES=1
  taskset -c 72-143 $B --test D --label N640_pm1 --res 640 --grid 256 $LOOP \
    --write-dir $W/N640_pm1 --snap-z 10 5 --save-delta $W/N640_pm1/final_delta.npy
  taskset -c 72-143 $B --test D --label N640_pm1_final --res 640 --grid 256 --n-steps 50 --repeat 1
) &
(
  export CUDA_VISIBLE_DEVICES=2
  for N in 800 832 864; do taskset -c 144-215 $B --test B4 --label N${N}_lpt2 --res $N --n-steps 50 --repeat 1; done
) &
(
  export CUDA_VISIBLE_DEVICES=3
  for N in 800 832 864; do taskset -c 216-287 $B --test B4 --label N${N}_lpt1 --res $N --lpt-order 1 --n-steps 50 --repeat 1; done
  taskset -c 216-287 $B --test D --label N768_pm1_lpt1_free --res 768 --lpt-order 1 --grid 256 $LOOP --free-ics \
    --save-delta $W/N768_pm1_lpt1_free/final_delta.npy
) &
wait
for d in N768_pm1_free N640_pm1; do
  [ -f $W/$d/redshift_density.txt ] && python benchmark_scripts/check_outputs.py $W/$d $W/$d/final_delta.npy
done
du -sh $W/*
