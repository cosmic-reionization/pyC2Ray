#!/bin/sh
#SBATCH --job-name=pyc2ray
#SBATCH --account=c45
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --constraint=gpu
#SBATCH --gres=gpu:4
#SBATCH --time=08:00:00
#SBATCH -e logs/pyc2ray.%j.err
#SBATCH -o logs/pyc2ray.%j.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=mbianc@ethz.ch
##SBATCH --mem 60G

module load cray/23.12
module load cray-python/3.11.5
module load cudatoolkit
module load craype-accel-nvidia90
module load cray-mpich/8.1.28

cd $HOME/codes/pyC2Ray/test/fstar_simulation

export LD_LIBRARY_PATH=/opt/cray/pe/mpich/8.1.28/ofi/gnu/12.3/lib-abi-mpich:$LD_LIBRARY_PATH
export HDF5_USE_FILE_LOCKING='FALSE'
export MPICH_GPU_SUPPORT_ENABLED=0  # for CUDA aware MPI

source $HOME/myvenv/pyc2ray-env/bin/activate

srun python run_test.py parameters.yml 

deactivate
