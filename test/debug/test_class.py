import numpy as np
import pyc2ray as pc2r

paramfile = 'parameters_fstar.yml'
sim = pc2r.C2Ray_fstar(paramfile=paramfile, Nmesh=128, use_gpu=True, use_mpi=False)


