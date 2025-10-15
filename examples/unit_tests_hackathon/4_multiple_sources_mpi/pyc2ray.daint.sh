#!/bin/sh
#SBATCH --job-name=test_pyc2ray
#SBATCH --account=sk015
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH -e slurm.err
#SBATCH -o slurm.out
#SBATCH --mail-type=ALL
#SBATCH --mail-user=michele.bianco@epfl.ch
#SBATCH --mem 60G
#SBATCH -C gpu

module purge
module load daint-gpu
module load gcc/9.3.0
module load nvidia
module load cray-python/3.9.4.1

export HDF5_USE_FILE_LOCKING='FALSE'
export PYTHONPATH="/users/mibianco/codes/pyC2Ray:$PYTHONPATH"

source /store/ska/sk015/pyc2ray-env/bin/activate

python3 -V
srun python3 run_test.py --gpu --mpi -numsrc 10
#python3 run_test.py --gpu -numsrc 10
deactivate 
