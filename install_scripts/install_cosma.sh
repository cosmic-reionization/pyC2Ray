#!/bin/bash

# 1. Load the necessary COSMA modules
module purge
module load gnu_comp/13.1.0 
module load nvhpc/25.3         
module load python/3.9.19      
module load openmpi/4.1.4

source /cosma8/data/dp004/dc-atta2/LW_in_pyc2ray/pyc2ray_implementation/pyc2ray-env/bin/activate
pip install numpy h5py  # NEW: Force install inside script to be sure

# 3. Get pyC2Ray directory path
cd ../
PYC2RAY_PATH=$(pwd)

# 4. Get python and numpy include paths
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path(name='include'))")
NUMPY_INCLUDE=$(python3 -m numpy.lib.utils -c "import numpy; print(numpy.get_include())") # NEW: Safer check

# 5. Compile Fortran extension module
cd $PYC2RAY_PATH/src/c2ray/
make clean
module unload nvhpc
# NEW: Calling f2py through python3 -m ensures it's found
CC=gcc FC=gfortran F2PY="python3 -m numpy.f2py" make 
module load nvhpc

mkdir -p $PYC2RAY_PATH/pyc2ray/lib
cp libc2ray.*.so $PYC2RAY_PATH/pyc2ray/lib

# 6. Compile CUDA extension module
cd $PYC2RAY_PATH/src/asora/
make clean
cp Makefile_copy Makefile

# NEW: Fix C++ standard for source_location support and include paths
sed -i 's/-std=c++14/-std=c++20/g' Makefile
sed -i 's,/insert_here_path_to_python_include,'"$PYTHON_INCLUDE"',' Makefile
sed -i 's,/insert_here_path_to_numpy_include,'"$NUMPY_INCLUDE"',' Makefile
sed -i 's/sm_[0-9][0-9]/sm_80/g' Makefile 

make
cp libasora.so $PYC2RAY_PATH/pyc2ray/lib

# 7. Add pyc2ray path to python paths
export PYTHONPATH="$PYC2RAY_PATH:$PYTHONPATH"

# 8. Test installation
cd $PYC2RAY_PATH
python3 -c "import pyc2ray; print('Import test successful')"