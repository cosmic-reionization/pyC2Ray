## Build Instructions for pyc2ray on Piz-Daint (September 2023)
Since the automatic build system isn't fully working yet, the extension modules must be compiled and placed in correct directories manually. After cloning the repository, create the `/lib` directory inside `/pyc2ray/` (from the root of the repository).

**Requirements**:
- C Compiler
- `gfortran` Fortran Compiler
- `nvcc` CUDA compiler
- `f2py` $\geq$ 1.24.4, provided by `numpy`

Additionally, once built, `pyc2ray` requires the `astropy` and `tools21cm` python packages to work.

### 1. Build Fortran extension module (C2Ray)

On daint, load the modules `daint-gpu` and `PrgEnv-cray` (not sure if the latter is actually needed). The tool to build the modules is `f2py`, provided by the `numpy` package. The build requires version 1.24.4 or higher, to check run `f2py` without any options. If the version is too old or the command doesn't exist, install the latest numpy version in your current virtual environment. To build the extension module, run
```
cd src/c2ray/
f2py --f2cmap f2c.f2py_f2cmap -DUSE_SUBBOX   --f90flags="-cpp" -c photorates.f90 raytracing.f90 chemistry.f90 -m libc2ray
```
and move the resulting shared library file `libc2ray.*.so` to the previously created `/pyc2ray/lib/` directory.

### 2. Build CUDA extension module (Asora)

On daint, load the modules `daint-gpu` and `nvidia` (to access the nvcc compiler). Go to `/src/asora/`. Here, you need to edit the Makefile and add the correct include paths at lines 3 and 4. To find the correct python include path (line 3), run
```
python -c "import sysconfig; print(sysconfig.get_path(name='include'))"
```
and to find the correct numpy include path (line 4), run
```
python -c "import numpy as np; print(np.get_include())"
```
Then, build the extension module by running `make`, and again move the file `libasora.so` to `/pyc2ray/lib/`.

To summarize, once the include paths in the makefile have been set, the full build process looks like this:
```
mkdir pyc2ray/lib
module purge
module load daint-gpu PrgEnv-cray
cd src/c2ray
f2py --f2cmap f2c.f2py_f2cmap -DUSE_SUBBOX   --f90flags="-cpp" -c photorates.f90 raytracing.f90 chemistry.f90 -m libc2ray
mv libc2ray.*.so ../../pyc2ray/lib
module purge
module load daint-gpu nvidia
cd ../asora
make
mv libasora.so ../../pyc2ray/lib
cd ../../
```

### 3. Test the Install
If the build was successful, go to `/examples/single_source/` and run
```
mkdir results
python run_example.py --gpu
```
This performs a RT simulation with a single source in a uniform volume. If an output is shown and no errors are produced, the install has likely succeeded.