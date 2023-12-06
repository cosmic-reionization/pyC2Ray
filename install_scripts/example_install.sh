#!/bin/bash

# load the necessary modules
module purge
module load daint-gpu
module load gcc/9.3.0
module load nvidia

# activate python environment
source ../pyc2ray-env/bin/activate

# get python and numpy include paths
PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path(name='include'))")
NUMPY_INCLUDE=$(python -c "import numpy as np; print(np.get_include())")

# compile Fortran extension module
cd src/c2ray/
make

mkdir ../../pyc2ray/lib
cp libc2ray.*.so ../../pyc2ray/lib

# compile CUDA extension module
cd ../asora/

# copy Makefile
cp Makefile_copy Makefile

# sostitute include path in Makefile
sed -i 's,/insert_here_path_to_python_include,'"$PYTHON_INCLUDE"',' Makefile
sed -i 's,/insert_here_path_to_numpy_include,'"$NUMPY_INCLUDE"',' Makefile

make
cp libasora.so ../../pyc2ray/lib

# add pyc2ray path to python paths
cd ../..
PYC2RAY_PATH=$(pwd)
export PYTHONPATH="$PYC2RAY_PATH:$PYTHONPATH"

# test installation
cd
python -c "import pyc2ray as pc2r"
echo "Installation of pyc2ray successful"
