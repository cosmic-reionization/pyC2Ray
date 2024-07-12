.. image:: fig/logo.png
.. _pyc2ray:

pyC2Ray: A flexible and GPU-accelerated radiative transfer framework for EoR
======================================
The **pyC2Ray** code is an updated version of the massively parallel raytracing and chemistry code, *C2Ray*. which has been extensively employed in reionization simulations and often requires millions of CPU-core hours simulated on several thousand computing nodes run on high-performance computers (HPC). 

The most time-consuming part of the code is calculating the hydrogen column density along the path of the ionizing photons. With the **pyC2Ray** update, we presented the *Accelerated Short-characteristics Octahedral RAytracing (ASORA)* method, a raytracing algorithm specifically designed to run on GPUs. 

The algorithm is written in *C++/CUDA* and wrapped in a *Python* interface that allows for easy and customized use of the code without compromising computational efficiency


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. toctree::
   :hidden:
   :maxdepth: 2

   installation
   tutorials
   modules
   contributors

