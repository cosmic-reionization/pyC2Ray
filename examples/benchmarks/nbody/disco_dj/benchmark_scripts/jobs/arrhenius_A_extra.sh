#!/bin/bash
#SBATCH -J dj-A-extra
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out

# Four independent series, one GPU (+72 Grace cores) each. Submit from examples/benchmarks/nbody/disco_dj.
#  GPU0  A: single vs. double precision against pkdgrav3 (256^3, reuses the pkdgrav3 run of job $PKD_REF)
#  GPU1  A: single vs. double precision, time and memory vs. N (B1 setup)
#  GPU2  A: the upstream example notebooks (1D/2D/3D) executed on the GPU
#  GPU3  B6: gradients (forward, reverse adjoint, reverse backprop, forward-mode JVP)
: "${PKD_REF:=5806041}"
source envs/arrhenius_disco_dj.sh
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
R=$DJ_RUN_ROOT/A_B2/$PKD_REF
SNAPS="--pkd-snap 10:$R/ref.00060 --pkd-snap 8:$R/ref.00070 --pkd-snap 6:$R/ref.00080 --pkd-snap 5:$R/ref.00090"
CFG="--config bullfrog:10 --config bullfrog:50:res_pm=512,worder=3,antialias=1 --config symplectic:100"
(
  export CUDA_VISIBLE_DEVICES=0
  for p in single double; do
    taskset -c 0-71 python benchmark_scripts/compare_pkdgrav3.py --test A_prec --label "$p:" --precision $p \
      --pkd-ic $R/ic_za.00000 $SNAPS $CFG
  done
) &
(
  export CUDA_VISIBLE_DEVICES=1
  for N in 128 256 384 512 640; do for p in single double; do
    taskset -c 72-143 python benchmark_scripts/djbench.py --test A_prec --label N${N}_$p --res $N --precision $p \
      --n-steps 50 --repeat 2
  done; done
) &
(
  export CUDA_VISIBLE_DEVICES=2
  NB=$DJ_RUN_ROOT/A_notebooks/$SLURM_JOB_ID; mkdir -p $NB/kernels/djpy
  # own kernel spec for this env (a user-level "python3" spec may point to another Python)
  printf '{"argv": ["%s", "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "djpy", "language": "python"}' \
    "$(command -v python)" > $NB/kernels/djpy/kernel.json
  cp $DISCODJ_SRC/disco_dj_*example.ipynb $NB/ && cd $NB
  for nb in disco_dj_1d_example disco_dj_1d_derivative_example disco_dj_2d_example disco_dj_3d_example; do
    t0=$(date +%s)
    JUPYTER_PATH=$NB taskset -c 144-215 jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.timeout=1800 --ExecutePreprocessor.kernel_name=djpy $nb.ipynb > $nb.log 2>&1
    rc=$?; echo "NOTEBOOK $nb rc=$rc time=$(( $(date +%s) - t0 ))s"
  done
) > logs/dj-A-notebooks-$SLURM_JOB_ID.txt 2>&1 &
(
  export CUDA_VISIBLE_DEVICES=3
  for N in 64 128 256; do for m in forward grad_adjoint grad_backprop jvp; do
    taskset -c 216-287 python benchmark_scripts/djgrad.py --res $N --n-steps 10 --mode $m
  done; done
  for s in 5 20 50; do for m in grad_adjoint grad_backprop; do
    taskset -c 216-287 python benchmark_scripts/djgrad.py --res 128 --n-steps $s --mode $m
  done; done
) &
wait
python - <<'PY'
# record the notebook results as a result file
import json, os, re, sys
sys.path.insert(0, "benchmark_scripts"); import djbench
lines = open(f"logs/dj-A-notebooks-{os.environ['SLURM_JOB_ID']}.txt").read()
recs = [{"notebook": m[0], "rc": int(m[1]), "time_s": int(m[2]), "status": "ok" if m[1] == "0" else "error"}
        for m in re.findall(r"NOTEBOOK (\S+) rc=(\d+) time=(\d+)s", lines)]
out = djbench.BENCH_DIR / "benchmark_results/arrhenius/A_notebooks" / f"A_notebooks_{os.environ['SLURM_JOB_ID']}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"metadata": djbench.metadata(), "records": recs}, indent=1)); print(recs)
PY
