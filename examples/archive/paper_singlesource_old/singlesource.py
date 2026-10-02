import sys

sys.path.append("../../")
import argparse
import time

import numpy as np

import pyc2ray as pc2r

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

# Global parameters
num_steps_between_slices = 10
numzred = 2
paramfile = "parameters.yml"
N = 128
use_octa = args.gpu

# Create C2Ray object
sim = pc2r.C2Ray_Test(paramfile, N, use_octa)

# Generate redshift list (test case)
zred_array = sim.generate_redshift_array(numzred, 1e7)

# Read sources
numsrc = 1
srcpos, srcflux = sim.read_sources("src.txt", numsrc)

# Raytracing Parameters
max_subbox = 1000
subboxsize = 64
r_RT = 150

# Measure time
tinit = time.time()

# Setup density
avgdens = 1e-3
xc = np.linspace(0, 14, N)
X, Y, Z = np.meshgrid(xc, xc, xc)
shadow_pos = np.array([76, 76, 63])
shadow_radius = 8
shadow_fact = 6
ndens = avgdens * np.ones((N, N, N))

# Statistics
mean_xfrac = np.empty(num_steps_between_slices)
mean_ionrate = np.empty(num_steps_between_slices)

for k in range(len(zred_array) - 1):
    # Compute timestep of current redshift slice
    zi = zred_array[k]
    zf = zred_array[k + 1]
    dt = sim.set_timestep(zi, zf, num_steps_between_slices)

    # Write output
    sim.write_output(zi)

    # Set density field (could be an actual cosmological field here)
    # TODO: this has to set the comoving density which is then scaled to the
    # correct redshift. In the timesteps, the density is then "diluted" gradually
    # sim.set_constant_average_density(avgdens,0)
    sim.ndens = ndens

    pc2r.printlog(f"\n=================================", sim.logfile)
    pc2r.printlog(f"Doing redshift {zi:.3f} to {zf:.3f}", sim.logfile)
    pc2r.printlog(f"=================================\n", sim.logfile)
    # Do num_steps_between_slices timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        pc2r.printlog(
            f"\n --- Timestep {t + 1:n}. Redshift: z = {sim.zred: .3f} Wall clock time: {tnow - tinit: .3f} seconds --- \n",
            sim.logfile,
        )

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        # If cosmology is disabled in parameter, this step does nothing (checked internally by the class)
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, srcflux, srcpos, r_RT, max_subbox)

if args.plot:
    import matplotlib.pyplot as plt

    plt.imshow(sim.xh[:, :, 63], norm="log", cmap="jet")
    plt.colorbar()
    plt.show()
    plt.imshow(sim.phi_ion[:, :, 63], norm="log", cmap="inferno")
    plt.show()

# Write final output
sim.write_output(zf)
pc2r.printlog(f"Done. Final time: {time.time() - tinit: .3f} seconds", sim.logfile)
