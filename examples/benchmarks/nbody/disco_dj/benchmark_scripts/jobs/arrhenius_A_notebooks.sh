#!/bin/bash
#SBATCH -J dj-A-nb
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1 --exclusive --mem=0
#SBATCH --gpus-per-node=4
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out

# A: execute the upstream example notebooks (1D, 1D derivative, 2D, 3D) of DISCO-DJ on one GH200.
# Executed copies and logs go to $DJ_RUN_ROOT/A_notebooks/<job>; pass/fail and time to
# benchmark_results/arrhenius/A_notebooks/. Submit from examples/benchmarks/nbody/disco_dj.
source envs/arrhenius_disco_dj.sh
python -c "import seaborn" 2>/dev/null || pip install -q seaborn
export CUDA_VISIBLE_DEVICES=0
HERE=$PWD
NB=$DJ_RUN_ROOT/A_notebooks/$SLURM_JOB_ID; mkdir -p $NB/kernels/djpy
# own kernel spec for this env (a user-level "python3" spec may point to another Python)
printf '{"argv": ["%s", "-m", "ipykernel_launcher", "-f", "{connection_file}"], "display_name": "djpy", "language": "python"}' \
  "$(command -v python)" > $NB/kernels/djpy/kernel.json
cp $DISCODJ_SRC/disco_dj_*example.ipynb $NB/ && cd $NB
for nb in disco_dj_1d_example disco_dj_1d_derivative_example disco_dj_2d_example disco_dj_3d_example; do
  t0=$(date +%s)
  JUPYTER_PATH=$NB jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=2400 \
    --ExecutePreprocessor.kernel_name=djpy $nb.ipynb > $nb.log 2>&1
  rc=$?; echo "NOTEBOOK $nb rc=$rc time=$(( $(date +%s) - t0 ))s"
  [ $rc = 0 ] || grep -E "Error|error" $nb.log | tail -3
done | tee $NB/summary.txt
cd $HERE
python - <<PY
import json, os, re, sys
sys.path.insert(0, "benchmark_scripts"); import djbench
recs = [{"notebook": m[0], "rc": int(m[1]), "time_s": int(m[2]), "status": "ok" if m[1] == "0" else "error"}
        for m in re.findall(r"NOTEBOOK (\S+) rc=(\d+) time=(\d+)s", open("$NB/summary.txt").read())]
out = djbench.BENCH_DIR / "benchmark_results/arrhenius/A_notebooks" / f"A_notebooks_{os.environ['SLURM_JOB_ID']}.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps({"metadata": djbench.metadata(), "records": recs}, indent=1)); print(recs)
PY
