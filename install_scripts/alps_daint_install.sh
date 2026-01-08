#!/bin/bash

# load the necessary modules
module purge
module load cray/23.12
module load cray-python/3.11.5
module load cray-mpich/8.1.28	# required for mpi4py
module load nvidia/24.3		# required for ASORA

# activate python environment
source ~/myvenv/pyc2ray-env/bin/activate
python -m pip install -r ../requirements.txt

# required for mpi4py
MPICC=cc
python3 -m pip install mpi4py

# get pyC2Ray directory path
cd ../
PYC2RAY_PATH=$(pwd)

# get python and numpy include paths
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path(name='include'))")
NUMPY_INCLUDE=$(python3 -c "import numpy as np; print(np.get_include())")

# compile Fortran extension module
cd $PYC2RAY_PATH/src/c2ray/
make

mkdir $PYC2RAY_PATH/pyc2ray/lib
cp libc2ray.*.so $PYC2RAY_PATH/pyc2ray/lib

# compile CUDA extension module
cd $PYC2RAY_PATH/src/asora/

# copy Makefile
cp Makefile_copy Makefile

# sostitute include path in Makefile
sed -i 's,/insert_here_path_to_python_include,'"$PYTHON_INCLUDE"',' Makefile
sed -i 's,/insert_here_path_to_numpy_include,'"$NUMPY_INCLUDE"',' Makefile

make
cp libasora.so $PYC2RAY_PATH/pyc2ray/lib

# add pyc2ray path to python paths
export PYTHONPATH="$PYC2RAY_PATH:$PYTHONPATH"

# go to home to test installation and export
cd
python3 -c "import pyc2ray as pc2r"
echo "Installation of pyc2ray successful"
