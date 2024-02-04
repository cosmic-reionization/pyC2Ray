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

# Create C2Ray object
sim = pc2r.C2Ray_fstar(paramfile=paramfile, Nmesh=256, use_gpu=True, use_mpi=False)

# Get redshift list (test case)
idx_zred, zred_array = np.loadtxt(sim.inputs_basename+'redshift_checkpoints.txt', dtype=float, unpack=True)

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

    # Read input files
    sim.read_density(fbase='CDM_200Mpc_2048.%05d.den.256.0', z=zi)

    # Read source files
    srcpos, normflux = sim.ionizing_flux(file='CDM_200Mpc_2048.%05d.fof.txt' %idx_zred[k], ts=num_steps_between_slices*dt, z=zi, save_Mstar=f'{sim.results_basename:}/sources')
    
    # Write output
    sim.write_output(zi)

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
