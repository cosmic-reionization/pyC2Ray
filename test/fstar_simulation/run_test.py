import sys
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
sim = pc2r.C2Ray_fstar(paramfile=paramfile, Nmesh=N, use_gpu=use_asora, use_mpi=False)

# Get redshift list (test case)
zred_array = np.loadtxt(sim.inputs_basename+'redshifts_checkpoints.txt', dtype=float)

# check for resume simulation
if(sim.resume):
    i_start = np.argmin(np.abs(zred_array - sim.zred))
else:
    i_start = 0

# Measure time
tinit = time.time()

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
    # srcpos, normflux = sim.read_sources(file=f'{sim.sources_basename:}{zi:.3f}-coarsest_wsubgrid_sources.hdf5', mass='hm', ts=num_steps_between_slices*dt)
    srcpos, normflux = sim.ionizing_flux(file=f'{zi:.3f}halo.hdf5', ts=num_steps_between_slices*dt, z=zi, save_Mstar=f'{sim.results_basename:}/sources')
    
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
