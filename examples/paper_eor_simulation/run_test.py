import sys
sys.path.append("../../../")
import numpy as np
import time
import pyc2ray as pc2r

# ======================================================================
# Example for pyc2ray: Cosmological simulation from N-body
# ======================================================================

# Global parameters
num_steps_between_slices = 2        # Number of timesteps between redshift slices
paramfile = sys.argv[1]             # Name of the parameter file
N = 250                             # Mesh size
use_asora = True                    # Determines which raytracing algorithm to use

# Create C2Ray object
sim = pc2r.C2Ray_244Test(paramfile=paramfile, Nmesh=N, use_gpu=use_asora, use_mpi=False)

# Get redshift list (test case)
zred_array = np.loadtxt(sim.inputs_basename+'redshifts_checkpoints.txt', dtype=float)
#zred_array = np.loadtxt('/users/sgiri/work_pyc2ray/redshifts_checkpoints_small.txt', dtype=float)

i_start = 0

# Measure time
tinit = time.time()
prev_i_zdens, prev_i_zsourc = -1, -1

# Loop over redshifts
for k in range(i_start, len(zred_array)-1):

    zi = zred_array[k]       # Start redshift
    zf = zred_array[k+1]     # End redshift

    pc2r.printlog(f"\n=================================", sim.logfile)
    pc2r.printlog(f"Doing redshift {zi:.3f} to {zf:.3f}", sim.logfile)
    pc2r.printlog(f"=================================\n", sim.logfile)

    # Compute timestep of current redshift slice
    dt = sim.set_timestep(zi, zf, num_steps_between_slices)

    # Write output
    sim.write_output(zi)

    # Read input files
    sim.read_density(z=zi)

    # Read source files
    srcpos, normflux = sim.read_sources(file='%ssources_hdf5/%.3f-coarsest_wsubgrid_sources.hdf5' %(sim.inputs_basename, zi), mass='hm', ts=num_steps_between_slices*dt)
    print(srcpos.shape)
    # Set redshift to current slice redshift
    sim.zred = zi

    # Loop over timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        pc2r.printlog(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n", sim.logfile)

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, normflux, srcpos)

    # Evolve cosmology over final half time step to reach the correct time for next slice (see note in c2ray_base.py)
    #sim.cosmo_evolve(0)
    sim.cosmo_evolve_to_now()

# Write final output
sim.write_output(zf)
