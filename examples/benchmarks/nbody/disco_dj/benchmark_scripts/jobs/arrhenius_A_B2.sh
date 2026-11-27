#!/bin/bash
#SBATCH -J dj-A-B2
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out

# A (against pkdgrav3) + B2 (accuracy vs. cost), 256^3 in L = 100 Mpc/h, from identical ICs:
#  1. pkdgrav3 Zel'dovich ICs at z = 150 (iLPT=1, bWriteIC) -> linear field for DISCO-DJ
#  2. pkdgrav3 reference run (2LPT, z = 150 -> 5, snapshots at z = 10, 8, 6, 5)
#  3. DISCO-DJ configurations from the same linear field, compared by P(k) ratio and r(k)
# Submit from examples/benchmarks/nbody/disco_dj. N=256 by default (sbatch --export=ALL,NPART=512 ...);
# PKD_REUSE=<job id> skips step 1-2 and uses that job's pkdgrav3 outputs.
: "${NPART:=256}"
: "${PKDGRAV3_EXE:=/nobackup/proj/disk/cosmic_reion_2026/personal/sgiri/projects/matched_ic_nbody/pkdgrav3-develop/build-gh200/pkdgrav3}"
HERE=$PWD
source envs/arrhenius_disco_dj.sh
RUN=$DJ_RUN_ROOT/A_B2/$SLURM_JOB_ID; mkdir -p $RUN
cp benchmark_scripts/euclid_z0_transfer_combined.dat $RUN/
COMMON="dBoxSize=100
nGrid=$NPART
iSeed=314159265
dRedFrom=150
achTfFile=\"euclid_z0_transfer_combined.dat\"
h=0.67
dOmega0=0.32
dLambda=0.68
dSigma8=0.83
dSpectral=0.96
bComove=True
bPeriodic=True
bEwald=True
bOverwrite=True
bMemUnordered=False"
printf '%s\nachOutName="ic_za"\niLPT=1\nnSteps=0\ndRedTo=150\nbWriteIC=True\n' "$COMMON" > $RUN/ic_za.par
printf '%s\nachOutName="ref"\niLPT=2\nnSteps=[60,10,10,10]\ndRedTo=[10,8,6,5]\niOutInterval=10\n' "$COMMON" > $RUN/ref.par

# pkdgrav3 in its own (aarch64) conda env, GPU 0, singleton MPI, all cores.
# PKD_REUSE=<job id>: reuse the pkdgrav3 ICs and snapshots of that job instead of running pkdgrav3.
if [ -n "$PKD_REUSE" ]; then
  RUN=$DJ_RUN_ROOT/A_B2/$PKD_REUSE; echo "reusing pkdgrav3 run $RUN"
else
(
  export PATH=$(echo "$PATH" | tr : '\n' | grep -v "envs/$DJ_CONDA_ENV" | paste -sd:)
  conda activate matched_ic
  export CUDA_VISIBLE_DEVICES=0
  cd $RUN
  git -C $(dirname $(dirname $PKDGRAV3_EXE)) log -1 --format='pkdgrav3 %h %s'
  /usr/bin/time -f "pkdgrav3 ic: %e s, %M kB" timeout 600 $PKDGRAV3_EXE ic_za.par > ic_za.log 2>&1 || { tail ic_za.log; exit 1; }
  /usr/bin/time -f "pkdgrav3 ref: %e s, %M kB" timeout 1800 $PKDGRAV3_EXE ref.par > ref.log 2>&1 || { tail ref.log; exit 1; }
  ls $RUN
) || exit 1
fi
conda activate $DJ_CONDA_ENV
cd $HERE
IC=$(ls $RUN/ic_za.0000* | head -1)
SNAPS="--pkd-snap 10:$RUN/ref.00060 --pkd-snap 8:$RUN/ref.00070 --pkd-snap 6:$RUN/ref.00080 --pkd-snap 5:$RUN/ref.00090"
CFG=""
for n in 5 10 20 50 100; do CFG="$CFG --config bullfrog:$n"; done
for n in 10 20 50 100; do CFG="$CFG --config fastpm:$n"; done
for n in 20 50 100 200; do CFG="$CFG --config symplectic:$n"; done
for n in 20 50 100; do CFG="$CFG --config bullfrog:$n:res_pm=$((2 * NPART))"; done
CFG="$CFG --config bullfrog:50:worder=3 --config bullfrog:50:antialias=1 --config bullfrog:50:n_resample=2"
CFG="$CFG --config bullfrog:50:res_pm=$((2 * NPART)),worder=3,antialias=1"
export CUDA_VISIBLE_DEVICES=0
python benchmark_scripts/compare_pkdgrav3.py --test B2 --label "" --pkd-ic $IC $SNAPS $CFG
