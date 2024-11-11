Installation
============

Since the automatic build system (i.e.: ``pip install pyc2ray``) isn't fully working yet (work in progress), the extension modules must be compiled and placed in correct directories manually.

However, a few example bash-shell scripts, that summarize the installation steps shown here below, can be found in the repository ``install_script/`` (link_).

You can modify and run your installation script from the same directory with the command line:

.. code-block:: bash

        cd install_script/
        source example_install.sh

.. _link: https://github.com/cosmic-reionization/pyC2Ray/tree/main/install_scripts

**Requirements**:

- C Compiler
- ``gfortran`` Fortran Compiler
- ``nvcc`` CUDA compiler
- ``f2py>=1.24.4``, provided by ``numpy``


1. GitHub Clone & Requirements
""""""""""""""""""""""""""""""""""""""""
Star by cloning the repository with the following command line:

.. code-block:: bash

        git clone https://github.com/cosmic-reionization/pyC2Ray.git

We highly recommend the use of a virtual environement, as ``pyc2ray`` requires some specific packages to work, such as ``astropy`` and ``tools21cm``. 

To install all the required python packages in an environement, use the the following command lines:

.. code-block:: bash
        
        python3 -m venv pyc2ray-env
        source ./pyc2ray-env/bin/activate
        cd pyC2Ray/
        python3 -m pip install  requirements.txt

This way will helps you to keep the required dependencies for different projects separated.

2. Build Fortran extension module
""""""""""""""""""""""""""""""""""""""""
The chemisty solver of ``pyc2ray`` is still in its original version written in Fortran90. Therefore, the build requires version ``f2py>=1.24.4``, provided by the ``numpy`` package. If the version of ``f2py`` is too old or the command doesn't exist, install the latest ``numpy`` version in your current virtual environment. 

To build the ``C2Ray`` Fortran extension module, run:

.. code-block:: bash

        mkdir pyc2ray/lib
        cd src/c2ray/
        make
        cp libc2ray.*.so ../../pyc2ray/lib

The last command line moves the resulting shared library file ``libc2ray.*.so`` to the previously created ``pyC2Ray/pyc2ray/lib/`` directory.

3. Build CUDA extension module
"""""""""""""""""""""""""""""""""""""
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

4. Test the Install
"""""""""""""""""""
You can quickly double-check with the command line:

.. code-block:: bash
        
        python -c "import pyc2ray as pc2r"

If the build was successful it should not give any error message. Moreover, you can use of the test script in ``pyC2Ray/test/unit_tests_hackathon/1_single_source`` and run

.. code-block:: bash
        
        mkdir results
        python run_example.py --gpu

This performs a RT simulation with a single source in a uniform volume, and checks for errors.


Future Installation
"""""""""""""""""""
We are currently working to make the installation easier. In the fugure to install ``pyc2ray`` you will simply run:

.. code-block:: bash

        pip install pyc2ray
