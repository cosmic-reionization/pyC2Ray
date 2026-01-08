<div align="left">
    <a name="logo"><img src="docs/fig/banner.jpg" width="800"></img></a>
</div>

# pyC2Ray: A flexible and GPU-accelerated radiative transfer framework
`pyc2ray` is the updated version of [C2Ray](https://github.com/garrelt/C2-Ray3Dm/tree/factorization) [(G. Mellema, I.T. Illiev, A. Alvarez & P.R. Shapiro, 2006)](https://ui.adsabs.harvard.edu/abs/2006NewA...11..374M/abstract), an astrophysical radiative transfer code widely used to simulate the Epoch of Reionization (EoR). `pyc2ray` features a new raytracing method developed for GPUs, named <b>A</b>ccelerated <b>S</b>hort-characteristics <b>O</b>cthaedral <b>RA</b>ytracing (<b>ASORA</b>). `pyc2ray` has a modern python interface that allows easy and customizable use of the code without compromising computational efficiency. A full description of the update and new raytracing method can be found at [Hirling, Bianco, Giri, Iliev, Mellema & Kneib (2024)](https://arxiv.org/abs/2311.01492).

The core features of `C2Ray`, written in Fortran90, are wrapped using `f2py` as a Python extension module, while the new raytracing library, _ASORA_, is implemented in C++ using CUDA. Both are native Python C extensions and can be directly accessed from any Python script.

Visit the [ReadTheDocs](https://pyc2ray.readthedocs.io) of `pyc2ray` for the complete documentation, tutorials, installation instructions, and more.

## Installation

**Requirements**:
- C Compiler
- `gfortran` Fortran Compiler
- `nvcc` CUDA compiler

In your environment simply run

```bash
pip install .
```

If the setuptools method doesn't work, you can alway compile the libraries manually.

Please see our [documentation](https://pyc2ray.readthedocs.io/en/latest/installation.html) for step-by-step instructions on how to install `pyc2ray`.

A few example scripts summarizing the installation steps can be found in the repository [`/install_script/`](https://github.com/cosmic-reionization/pyC2Ray/tree/main/install_scripts).

<!--
Additionally, once built, `pyc2ray` requires the `astropy` and `tools21cm` python packages to work. A few example scripts, that summarize the installation steps shown here below, are given in the repository `/install_script/`.

### 1. Build Fortran extension module (C2Ray)

The tool to build the module is `f2py`, provided by the `numpy` package. The build requires version 1.24.4 or higher, to check run `f2py` without any options. If the version is too old or the command doesn't exist, install the latest numpy version in your current virtual environment. To build the extension module, run
```
cd src/c2ray/
make
cp libc2ray.*.so ../../pyc2ray/lib
```
The last command line moves the resulting shared library file `libc2ray.*.so` to the previously created `/pyc2ray/lib/` directory.
### 2. Build CUDA extension module (Asora)
```
cd ../asora/
```
Edit the Makefile and add the correct include paths of your `python` (line 3, `PYTHONINC`) and `numpy` library (line 4, `NUMPYINC`). To find the correct python include path (line 3), you can run from your terminal
```
python -c "import sysconfig; print(sysconfig.get_path(name='include'))"
```
and to find the correct numpy include path (line 4), run
```
python -c "import numpy as np; print(np.get_include())"
```
Then, build the extension module by running `make`, and again move the file `libasora.so` to `/pyc2ray/lib/`.
```
make
cp libasora.so ../../pyc2ray/lib
```
Finally, you can add `pyc2ray` path to your `PYTHONPATH`.
```
cd ../..
PYC2RAY_PATH=$(pwd)
export PYTHONPATH="$PYC2RAY_PATH:$PYTHONPATH"
```
### 3. Test the Install
You can quickly double-check with the command line:
```
python -c "import pyc2ray as pc2r"
```
If the build was successful it should not give any error message. Moreover, you can use of the test script in `/test/unit_tests_hackathon/1_single_source` and run
```
mkdir results
python run_example.py --gpu
```
This performs a RT simulation with a single source in a uniform volume, and checks for errors.

## Reproduce tests from the paper
The four tests performed in the paper are located in `paper_tests`, along with the script used to perform the raytracing benchmark. Each directory contains a Jupyter notebook with basic instructions to reproduce the plots shown in the paper. Note that for some of these tests, a reference output from the original C2Ray code is used for comparison. In these cases, you have the choice to either run C2Ray yourself by making the appropriate adjustments in the source code, or download the binary output directly, which is currently hosted [here](https://drive.proton.me/urls/0W5XJ6WXXC#QWHTxmY9qQ99).

### Note on raytracing benchmark
The raytracing benchmark (Figure 8 in the paper) might be an especially useful test to reproduce on your system. 
<div align="left">
   <a name="scaling"><img src="docs/fig/scaling.jpg" width="800" height="auto"></img></a>
</div>

The relevant script is located at `paper_tests/raytracing_benchmark/run_test.py`. This script is quite general, and allows you to measure the runtime of the GPU raytracing function for a varying number of sources, batch sizes and raytracing radii. The steps to reproduce exactly the test shown in the paper are outlined in the Jupyter Notebooks in `paper_tests/raytracing_benchmark/`. 

## Usage
A `pyc2ray` simulation is set up by creating an instance of a subclass of `C2Ray`. A few examples are provided, but in principle the idea is to create a new subclass and tailor it for the specific requirements of the simulation you wish to perform. The core functions (e.g. time evolution, raytracing, chemistry) are defined in the `C2Ray` base class, while auxilary methods specific to your use case are free to be overloaded as you wish.
-->

## TODO list
Here we list a series of numerical and astrophysical implementations we would like to include in future version of `pyc2ray`.
- Helium ionization, HeII and HeIII
- Sources radiative feedback
- Sources X-ray heating
- GPU implementation of the chemistry solver
- multi-frequency UV radiation

## CONTRIBUTING

If you find any bugs or unexpected behavior in the code, please feel free to open a [Github issue](https://github.com/cosmic-reionization/pyC2Ray/issues).
The issue page is also good if you seek help or have suggestions for us.

### Submitting changes to the code

Please follow these instructions to ensure a smooth integration, at least until a CI system is put into place:

0. **Only the first time**, install `pre-commit` in your enviornment and the pre-commit hooks with `pre-commit install`.
1. Create a new branch off the main trunk and make your modifications there.
2. Commit your changes and fix any issue highlighted by the pre-commit hooks; code format is automatically fixed.
3. Push your branch to the remote repository.
4. Open a Pull Request on GitHub to the main branch.
5. It is strongly suggested to squash all the commits into one.
6. In the PR's description on GitHub, specify blocking dependencies with the message "Depends on #..." and close issues with "Closes #...".
7. Ask the code to be reviewed before merging.

## AKNOWLEDGMENT

This project was initially developed by [Patrick Hirling](https://github.com/phirling) as part of the astrophysics practical workshop supervised by Michele Bianco during his master's degree at EPFL. You can find the original version of the code on his GitHub page: [asora](https://github.com/phirling/pyc2ray).
