#!/bin/bash
#SBATCH -J dj-smoke
#SBATCH -A naiss2026-4-1146-gpu
#SBATCH -p gpu
#SBATCH -N 1
#SBATCH --gpus-per-node=1 -c 72
#SBATCH -t 00:30:00
#SBATCH -o logs/%x-%j.out

# Functional check on one GH200 before the real tests: every output mode at 128^3, the
# demo, and a GPU vs. CPU run of the same small case (start of A). Submit from examples/benchmarks/nbody/disco_dj.
source envs/arrhenius_disco_dj.sh
nvidia-smi -L
python -c "import jax; print(jax.__version__, jax.devices())"
B="python benchmark_scripts/djbench.py --test smoke"
$B --label final128 --res 128 --n-steps 20 --repeat 2
$B --label loop128 --res 128 --n-steps 20 --outputs loop --n-outputs 5 --repeat 1
$B --label collect128 --res 128 --n-steps 20 --outputs collect --n-outputs 5 --repeat 1
$B --label pm2x128 --res 128 --res-pm 256 --n-steps 20 --repeat 1
$B --label tree64 --res 64 --method treepm --n-steps 10 --repeat 0
# GPU vs CPU, same case and seed (saves the final grids for the validation notebook)
$B --label gpu64 --res 64 --n-steps 20 --save-delta $DJ_RUN_ROOT/smoke/$SLURM_JOB_ID/gpu64.npy
JAX_PLATFORMS=cpu $B --label cpu64 --res 64 --n-steps 20 --save-delta $DJ_RUN_ROOT/smoke/$SLURM_JOB_ID/cpu64.npy
python "$DISCODJ_SRC/demo.py" --res 64 --nsteps 3 > /dev/null && echo "demo.py ok"
