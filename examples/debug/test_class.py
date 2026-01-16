import numpy as np
import pyc2ray as pc2r
import astropy.units as u

#paramfile = '/users/mibianco/codes/pyC2Ray/test/fstar_simulation/parameters.yml'
paramfile = './parameters_fstar.yml'
sim = pc2r.C2Ray_fstar(paramfile=paramfile, Nmesh=256, use_gpu=False, use_mpi=False)

idx_zred, zred_array = np.loadtxt(sim.inputs_basename+'redshift_checkpoints.txt', dtype=float, unpack=True)

k = 0
zi = zred_array[k]       # Start redshift
zf = zred_array[k+1]     # End redshift
dt = sim.set_timestep(zi, zf, 2)
print(dt, (dt*u.s).to('Myr'))

#srcpos, normflux = sim.ionizing_flux(file='CDM_200Mpc_2048.%05d.fof.txt' %idx_zred[k], z=zi, save_Mstar=sim.results_basename+'/sources')
srcpos, normflux = sim.ionizing_flux(file='test_CDM.txt', z=zi, save_Mstar=sim.results_basename+'/sources')