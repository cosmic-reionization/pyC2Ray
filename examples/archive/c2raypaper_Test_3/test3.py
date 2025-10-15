import sys
sys.path.append("../../")
import pyc2ray as pc2r
import time
import numpy as np
import argparse
import astropy.units as u

parser = argparse.ArgumentParser()
parser.add_argument("-resolution",type=str,default=None)
parser.add_argument("--cpu",action='store_true')
args = parser.parse_args()

if args.resolution is None:
    print("Set time resolution")
    exit

# ======================================================================
# t_evol = 1 Myr
# COARSE TIME: dt = t_evol / 10 = 0.1 Myr
# FINE TIME: dt = t_evol / 100 = 0.01 Myr
# ======================================================================

# Global parameters
if args.resolution == "fine":
    numzred = 101
    delta_time = 0.01
elif args.resolution == "coarse":
    numzred = 11
    delta_time = 1.5 #0.1
else:
    raise ValueError("Unknown resolution")

num_steps_between_slices = 1        # Number of timesteps between redshift slices
paramfile = "parameters.yml"        # Name of the parameter file
N = 128                             # Mesh size
Lbox = 1.4e22*u.cm #6e21*u.cm
dr_pc = Lbox.to('pc').value/N
print("Cell size (pc):",dr_pc)

if args.cpu:
    use_octa = False
else:
    use_octa = True                    # Determines which raytracing algorithm to use

# Raytracing Parameters
max_subbox = 1000                   # Maximum subbox when using C2Ray raytracing
r_RT = 128                            # When using C2Ray raytracing, sets the subbox size. When using OCTA, sets the octahedron size

# Create C2Ray object
sim = pc2r.C2Ray_Test(paramfile, N, use_octa)

# Create density field
halo_pos = np.array([63,63,63]) # <- source at 128 in fortran 1-indexing
halo_r0 = 5000 #91.5
halo_n0 = 0.015 #3.2
ndens = np.empty((N,N,N))
for i in range(0,N):
    for j in range(0,N):
        for k in range(0,N):
            r = np.sqrt((i-halo_pos[0])**2 + (j-halo_pos[1])**2 + (k-halo_pos[2])**2)
            if r == 0:
                print("origin")
                r = 1
            r *= dr_pc
            ndens[i,j,k] = halo_n0 * (halo_r0 / r)**1

sim.ndens = ndens #halo_n0 * np.ones((N,N,N))

# import matplotlib.pyplot as plt
# plt.imshow(ndens[:,:,127],norm='log')
# plt.colorbar()
# plt.show()


# Read sources
srcpos, srcstrength = sim.read_sources("source.txt",1)

# Measure time
tinit = time.time()

dt = delta_time*u.Myr.to('s')
print(dt)

# Loop over redshifts
for k in range(numzred-1):

    pc2r.printlog(f"\n=================================",sim.logfile)
    pc2r.printlog(f"Doing Slice {k:n}",sim.logfile)
    pc2r.printlog(f"=================================\n",sim.logfile)

    # Write output
    sim.write_output(k)

    # Loop over timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        pc2r.printlog(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n",sim.logfile)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, srcstrength, srcpos, r_RT, max_subbox)

# Write final output
sim.write_output(numzred-1)
pc2r.printlog(f"Done. Final time: {time.time() - tinit : .3f} seconds",sim.logfile)