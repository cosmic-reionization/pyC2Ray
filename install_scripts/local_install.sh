#!/bin/bash
## general installation sript for local machine with GPU driver
cd ../
PYC2RAY_PATH=$(pwd)

# get python and numpy include paths
PYTHON_INCLUDE=$(python -c "import sysconfig; print(sysconfig.get_path(name='include'))")
NUMPY_INCLUDE=$(python -c "import numpy as np; print(np.get_include())")

# compile Fortran extension module
cd $PYC2RAY_PATH/src/c2ray/
make

mkdir $PYC2RAY_PATH/pyc2ray/lib
cp libc2ray.*.so $PYC2RAY_PATH/pyc2ray/lib

# compile CUDA extension module
cd $PYC2RAY_PATH/src/asora/

# copy Makefile
cp Makefilei_copy Makefile

# sostitute include path in Makefile
sed -i 's,/insert_here_path_to_python_include,'"$PYTHON_INCLUDE"',' Makefile
sed -i 's,/insert_here_path_to_numpy_include,'"$NUMPY_INCLUDE"',' Makefile

make
cp libasora.so $PYC2RAY_PATH/pyc2ray/lib

# add pyc2ray path to python paths
export PYTHONPATH="$PYC2RAY_PATH:$PYTHONPATH"

# go to home to test installation and export
cd
python -c "import pyc2ray as pc2r"
echo "...done."
