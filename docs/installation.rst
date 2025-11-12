Installation
============

This section explains the different steps required to install ``pyc2ray`` on your local machine or on a high-performance computing (HPC) system.

If you are a **user** and only want to run reionization simulations with the latest version of the code, follow instructions `Step 1 <automatic-installation_>`_ and `Step 5 <test-installation_>`_ below to install and test your setup.

If you are a **developer** and need to substantially modify the components of ``pyc2ray``, you will need to manually compile the *C++/CUDA* and *Fortran* modules and place them in the correct directories. To do so, follow instructions from `Step 2 <python-environment-and-requirements_>`_ to `Step 5 <test-installation_>`_ below.

Additionally, you can find example Bash scripts summarizing these installation steps in the ``install_scripts/`` directory (`link`_).  You can modify and run the installation script from that directory using:

.. code-block:: bash
        
        cd install_scripts/
        source example_install.sh


.. _link: https://github.com/cosmic-reionization/pyC2Ray/tree/main/install_scripts

*Remark*: If no GPUs are detected, the CPU version of the raytracing module will be compiled and installed instead. This version is not MPI-optimized (work in progress) and is intended primarily for small test cases and tutorials for students to run on a local machine. We strongly recommend **not using the CPU-only version for large cosmological simulations that include millions of ionizing sources**, but rather limiting it to runs with at most a few hundred sources.

**Requirements**
----------------

Basic requirements for installation are:

- C compiler  
- Python ``>=3.9``  
- ``gfortran`` (Fortran compiler)  
- ``nvcc`` (CUDA compiler)  
- ``f2py >= 1.24.4`` (provided by ``numpy``)

.. _automatic-installation:

1. Automatic Installation
--------------------------------

``pyC2Ray`` provides an automatic build system that allows for easy installation. It is good practice to install the code in a dedicated Python or Conda environment (see instruction `Step 2 <python-environment-and-requirements_>`_, no need to install the requirements).

.. code-block:: bash

        git clone https://github.com/cosmic-reionization/pyC2Ray.git
        cd pyC2Ray/
        pip install .

After this, no additional steps are required. A compiled version of the code will be available locally and added to your paths.

To uninstall the code, you can simply: 

.. code-block:: bash
        
        pip uninstall pyc2ray


.. _python-environment-and-requirements:

2. Python Environment and Requirements
--------------------------------

Start by cloning the repository:

.. code-block:: bash

        git clone https://github.com/cosmic-reionization/pyC2Ray.git

We **strongly recommend** using a virtual environment, as ``pyc2ray`` requires several specific packages, such as ``astropy`` and ``tools21cm``.

To create a virtual environment and install all required Python packages, run:

.. code-block:: bash

        python3 -m venv pyc2ray-env
        source ./pyc2ray-env/bin/activate
        cd pyC2Ray/
        python3 -m pip install -r requirements.txt

This approach helps you keep dependencies for different projects separate.


3. Build the Fortran Extension Module
--------------------------------

The chemistry solver in ``pyc2ray`` remains in its original Fortran90 implementation.  
Therefore, the build requires ``f2py >= 1.24.4`` (provided by ``numpy``).  
If ``f2py`` is missing or outdated, install the latest ``numpy`` version in your active virtual environment.

To build the ``C2Ray`` Fortran extension module, run:

.. code-block:: bash

        mkdir pyc2ray/lib
        cd src/c2ray/
        make
        cp libc2ray.*.so ../../pyc2ray/lib

The last command moves the resulting shared library file (``libc2ray.*.so``) to the ``pyC2Ray/pyc2ray/lib/`` directory.


4. Build the CUDA Extension Module
--------------------------------

.. code-block:: bash

        cd ../asora/

Edit the Makefile to include the correct paths for your Python and NumPy headers.  
Specifically, update the following lines:

- **Line 3 (``PYTHONINC``)** — path to the Python include directory  
- **Line 4 (``NUMPYINC``)** — path to the NumPy include directory

You can find these paths by running:

.. code-block:: bash

        python -c "import sysconfig; print(sysconfig.get_path('include'))"
        python -c "import numpy as np; print(np.get_include())"

Then, build the CUDA extension module and move the resulting library file:

.. code-block:: bash

        make
        cp libasora.so ../../pyc2ray/lib

Finally, add the ``pyc2ray`` path to your ``PYTHONPATH`` environment variable:

.. code-block:: bash

        cd ../..
        PYC2RAY_PATH=$(pwd)
        export PYTHONPATH="$PYC2RAY_PATH:$PYTHONPATH"

.. _test-installation:

5. Test the Installation
--------------------------------

You can quickly verify your installation with:

.. code-block:: bash

        python -c "import pyc2ray as pc2r"

If the build was successful, no error messages should appear.

Additionally, you can run a test simulation using one of the provided test scripts:

.. code-block:: bash

        mkdir results
        cd pyC2Ray/test/unit_tests_hackathon/1_single_source
        python run_example.py --gpu

This test performs a radiative transfer simulation with a single source in a uniform volume and checks for errors.
