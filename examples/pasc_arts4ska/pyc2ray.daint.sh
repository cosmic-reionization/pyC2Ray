#!/bin/sh
#SBATCH --job-name=pyc2ray
#SBATCH --account=sk014
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=12:00:00
#SBATCH -e logs/pyc2ray.%j.err
#SBATCH -o logs/pyc2ray.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mbianc@ethz.ch
##SBATCH --mem 60G

module load cray/23.12
module load cray-python/3.11.5
module load cce/17.0.0
module load cray-mpich/8.1.28
#module load nvidia/24.3	# this is needed only to compile asora

export HDF5_USE_FILE_LOCKING='FALSE'
export PYTHONPATH="$HOME/codes/pyC2Ray:$PYTHONPATH"
source $HOME/myvenv/pyc2ray-env/bin/activate

#export CUDA_DEVICE_ORDER=$PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=$SLURM_LOCALID
export MPICH_GPU_SUPPORT_ENABLED=0	# for CUDA aware MPI
#export MPICH_GTL_NVIDIA_DISABLE=0

srun python run_test.py parameters.yml 

deactivate
