.. image:: fig/logo.png
.. _pyc2ray:

pyC2Ray: A flexible and GPU-accelerated radiative transfer framework for EoR
======================================
The **pyC2Ray** code is an updated version of the massively parallel raytracing and chemistry code, *C2Ray*. which has been extensively employed in reionization simulations and often requires millions of CPU-core hours simulated on several thousand computing nodes run on high-performance computers (HPC). 

The most time-consuming part of the code is calculating the hydrogen column density along the path of the ionizing photons. With the **pyC2Ray** update, we presented the *Accelerated Short-characteristics Octahedral RAytracing (ASORA)* method, a raytracing algorithm specifically designed to run on GPUs. 

The algorithm is written in *C++/CUDA* and wrapped in a *Python* interface that allows for easy and customized use of the code without compromising computational efficiency


Reporting issues and contributing
=================================
If you find any bugs or unexpected behavior in the code, please feel free to open a Github issue_. The issue page is also good if you seek help or have suggestions for us.

.. _issue: https://github.com/cosmic-reionization/pyC2Ray/issues


Aknowledgment
=============
The update of the original code was initially developed by Patrick Hirling (see his github_) as part of an astrophysics practical workshop for master students supervised by Michele Bianco, when he was part of EPFL. 

If you use this code in one of your scientific publications, please acknowledge it by citing the associated paper_.

.. _paper: https://arxiv.org/abs/2311.01492
.. _github: https://github.com/phirling


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

