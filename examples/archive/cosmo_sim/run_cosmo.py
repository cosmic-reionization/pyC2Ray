<<<<<<< HEAD
import sys, numpy as np, time
=======
import sys
import numpy as np, time
>>>>>>> main
import pyc2ray as pc2r

from astropy import units as u

# ======================================================================
# Example 2 for pyc2ray: Cosmological simulation from N-body
# ======================================================================

# TODO: find a way to replace this line
import tools21cm as t2c
t2c.set_sim_constants(boxsize_cMpc=244)

# Global parameters
num_steps_between_slices = 10       # Number of timesteps between redshift slices
paramfile = sys.argv[1]             # Name of the parameter file
N = 250                             # Mesh size
use_octa = True                     # Determines which raytracing algorithm to use

# Raytracing Parameters
max_subbox = 100                   # Maximum subbox when using C2Ray raytracing
r_RT = 40                           # When using C2Ray raytracing, sets the subbox size. When using OCTA, sets the octahedron size

# Create C2Ray object
sim = pc2r.C2Ray_CubeP3M(paramfile=paramfile, Nmesh=N, use_gpu=use_octa)

# Get redshift list (test case)
zred_array = np.loadtxt(sim.inputs_basename+'redshifts.txt', dtype=float)

# check for resume simulation
if(sim.resume):
    i_start = np.argmin(np.abs(zred_array - sim.zred_0))
else:
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
    i_zsourc = np.argmin(np.abs(sim.zred_sources[sim.zred_sources > zf] - zi))
    if(i_zsourc == prev_i_zsourc):
        pass
    else:
        srcpos, normflux = sim.read_sources(file='%ssources_hdf5/%.3f-coarsest_wsubgrid_sources.hdf5' %(sim.inputs_basename, sim.zred_sources[i_zsourc]), mass='hm')
        prev_i_zsourc = i_zsourc

    if(np.count_nonzero(normflux) == 0):
        pc2r.printlog(f"\n --- No sources at redshift: z = {sim.zred : .3f}. Skipping redshift step --- \n", sim.logfile)
        continue

    # Set redshift to current slice redshift
    sim.zred = zi

    # Loop over timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        pc2r.printlog(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n", sim.logfile)

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, normflux, srcpos, r_RT, max_subbox)

# Write final output
sim.write_output(zf)
