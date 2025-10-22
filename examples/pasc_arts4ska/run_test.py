import sys
import numpy as np, os
import time
import pyc2ray as pc2r

from c2ray_pasc import C2Ray_PASC

# ======================================================================
# Example for pyc2ray: Cosmological simulation from N-body
# ======================================================================

# Global parameters
num_steps_between_slices = 2        # Number of timesteps between redshift slices
paramfile = sys.argv[1]             # Name of the parameter file

# Create C2Ray object
sim = C2Ray_PASC(paramfile=paramfile)

# copy parameter file into the output directory
if(sim.rank == 0): os.system('cp %s %s' %(paramfile, sim.results_basename))

# Get redshift list (test case)
idx_zred, zred_array = np.loadtxt(sim.inputs_basename+'redshift_checkpoints.txt', dtype=float, unpack=True)
idx_zred, zred_array = idx_zred[:94], zred_array[:94]

# check for resume simulation
if(sim.resume):
    i_start = np.argmin(np.abs(zred_array - sim.zred))
    sim.resume = i_start+1
else:
    i_start = 0

# Measure time
timer = pc2r.Timer()
timer.start()
    
# Loop over redshifts
for k in range(i_start, len(zred_array)-1):

    zi = zred_array[k]       # Start redshift
    zf = zred_array[k+1]     # End redshift

    sim.printlog("\n=================================\nDoing redshift %.3f to %.3f\n=================================\n" %(zi, zf), sim.logfile)

    # Compute timestep of current redshift slice
    dt = sim.set_timestep(zi, zf, num_steps_between_slices)

    # Read input files
    sim.read_density(fbase='CDM_200Mpc_2048.%05d.den.256.0' %idx_zred[k], z=zi)

    # Read source files
    srcpos, normflux = sim.ionizing_flux(file='CDM_200Mpc_2048.%05d.fof.txt' %idx_zred[k], z=zi, dt=dt) #, save_Mstar=sim.results_basename+'/sources')

    # Set redshift to current slice redshift
    sim.zred = zi

    # Loop over timesteps
    for t in range(num_steps_between_slices):
        t_age = sim.cosmology.age(zi).cgs.value + t*dt
        z = sim.time2zred(t_age)
        tnow = timer.lap('z = %.3f' %z)
        sim.printlog("\n --- Timestep %d: z = %.3f, Wall clock time: %s --- \n" %(t+1, sim.zred, tnow), sim.logfile)

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(path_out=sim.results_basename, redshift=z, dt=dt, src_flux=normflux, src_pos=srcpos)

        # save outputs
        if(sim.rank == 0):    
            # Write output
            sim.write_output(z=z, ext='.npy')

    # Evolve cosmology over final half time step to reach the correct time for next slice (see note in c2ray_base.py)
    #sim.cosmo_evolve(0)
    sim.cosmo_evolve_to_now()

# stop the timer and print the summary
timer.stop()
sim.printlog(timer.summary, sim.logfile)

# Write final output
sim.write_output(zf, ext='.npy')
