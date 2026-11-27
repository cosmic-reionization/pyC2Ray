#!/bin/bash
#SBATCH -J dj-setup
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1 -c 72
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out

# Build the aarch64 conda env with JAX (CUDA 12 wheels) and DISCO-DJ on a GH200 node,
# then check that JAX sees the GPU. Submit from examples/benchmarks/nbody/disco_dj.
set -e
source envs/arrhenius_disco_dj.sh
set -x
uname -m; nvidia-smi -L
if ! conda env list | grep -q "^$DJ_CONDA_ENV "; then
  conda create -y -n "$DJ_CONDA_ENV" -c conda-forge python=3.12 pip cmake make gxx=13 \
    gsl pybind11 numpy scipy h5py matplotlib ipykernel nbconvert
fi
conda activate "$DJ_CONDA_ENV"
# Versions DISCO-DJ was tested with upstream (README): jax/jaxlib 0.7.2, diffrax 0.7.0
pip install --no-cache-dir "jax[cuda12]==0.7.2" "diffrax==0.7.0" einops tqdm scikit-build-core
# the upstream example notebooks also import seaborn
pip install --no-cache-dir seaborn
# DISCO-DJ at the recorded commit (C++ part: N-GenIC noise + Tree-PM short range; needs GSL)
git -C "$DISCODJ_SRC" log -1 --format='DISCO-DJ %h %cs %s'
CMAKE_PREFIX_PATH=$CONDA_PREFIX pip install --no-cache-dir --no-deps "$DISCODJ_SRC"
pip freeze > benchmark_results/arrhenius/pip_freeze_setup_${SLURM_JOB_ID}.txt
python - <<'PY'
import jax, jax.numpy as jnp, discodj, discodj_native
print("jax", jax.__version__, "devices", jax.devices())
x = jnp.ones((1024, 1024)); print("matmul ok", float((x @ x).sum()))
from discodj import DiscoDJ
dj = DiscoDJ(dim=3, res=64, boxsize=100.0).with_timetables().with_linear_ps().with_ics().with_lpt(n_order=2)
X, P, a = dj.run_nbody(a_ini=0.02, a_end=1.0, n_steps=10, res_pm=128, stepper="bullfrog")
print("quickstart ok", X.shape, X.device, float(a))
PY
