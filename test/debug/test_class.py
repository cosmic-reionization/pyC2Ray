import numpy as np
import pyc2ray as pc2r

paramfile = 'parameters_fstar.yml'
N = 128

sim = pc2r.C2Ray_fstar(paramfile=paramfile, Nmesh=N, use_gpu=True, use_mpi=False)

