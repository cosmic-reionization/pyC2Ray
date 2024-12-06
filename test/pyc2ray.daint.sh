#!/bin/sh
#SBATCH --job-name=pyc2ray
#SBATCH --account=c31
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH -e logs/pyc2ray.%j.err
#SBATCH -o logs/pyc2ray.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mbianc@ethz.ch
#SBATCH --mem 60G
#SBATCH -C gpu

module load daint-gpu
module load gcc/9.3.0
module load nvidia
module load cray-python/3.9.4.1

cd /users/mibianco/codes/pyC2Ray/test/fstar_simulation

export HDF5_USE_FILE_LOCKING='FALSE'
export PYTHONPATH="/users/mibianco/codes/pyC2Ray:$PYTHONPATH"

source /store/ska/sk015/pyc2ray-env/bin/activate

#export CUDA_DEVICE_ORDER=$PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID

python3 -V
#srun python3 run_test.py parameters.yml 
srun python3 run_bursty.py parameters.yml 
deactivate
