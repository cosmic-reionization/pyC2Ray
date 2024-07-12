Installation
============

Installation
============
Since the automatic build system isn't fully working yet, the extension modules must be compiled and placed in correct directories manually.

**Requirements**:
- C Compiler
- ``gfortran`` Fortran Compiler
- ``nvcc`` CUDA compiler
- ``f2py`` $\geq$ 1.24.4, provided by ``numpy``

Additionally, once built, ``pyc2ray`` requires the ``astropy`` and ``tools21cm`` python packages to work. A few example scripts, that summarize the installation steps shown here below, are given in the repository ``/install_script/``.


1. Build Fortran extension module (C2Ray)
========================================

The tool to build the module is ``f2py``, provided by the ``numpy`` package. The build requires version 1.24.4 or higher, to check run ``f2py`` without any options. If the version is too old or the command doesn't exist, install the latest numpy version in your current virtual environment. To build the extension module, run

.. code-block:: bash
        cd src/c2ray/
        make
        cp libc2ray.*.so ../../pyc2ray/lib

The last command line moves the resulting shared library file ``libc2ray.*.so`` to the previously created ``/pyc2ray/lib/`` directory.

2. Build CUDA extension module (Asora)
=====================================
.. code-block:: bash
        cd ../asora/

Edit the Makefile and add the correct include paths of your ``python``  (line 3, ``PYTHONINC``) and ``numpy`` library (line 4, ``NUMPYINC``). To find the correct python include path (line 3), you can run from your terminal

.. code-block:: bash
        python -c "import sysconfig; print(sysconfig.get_path(name='include'))"

and to find the correct numpy include path (line 4), run

.. code-block:: bash
        python -c "import numpy as np; print(np.get_include())"

Then, build the extension module by running ``make``, and again move the file ``libasora.so`` to ``/pyc2ray/lib/``.

.. code-block:: bash
        make
        cp libasora.so ../../pyc2ray/lib

Finally, you can add ``pyc2ray`` path to your ``PYTHONPATH``.

.. code-block:: bash
        cd ../..
        PYC2RAY_PATH=$(pwd)
        export PYTHONPATH="$PYC2RAY_PATH:$PYTHONPATH"

3. Test the Install
===================
You can quickly double-check with the command line:

.. code-block:: bash
        python -c "import pyc2ray as pc2r"

If the build was successful it should not give any error message. Moreover, you can use of the test script in ``/test/unit_tests_hackathon/1_single_source`` and run

.. code-block:: bash
        mkdir results
        python run_example.py --gpu

This performs a RT simulation with a single source in a uniform volume, and checks for errors.


Future Installation
===================
We are currently working to make the installation easier. In the fugure to install ``pyc2ray`` you will simply run:

.. code-block:: bash
        pip install pyc2ray
