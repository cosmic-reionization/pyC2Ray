import sys
sys.path.append("../../../")
import pyc2ray as pc2r
import tools21cm as t2c
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",action='store_true')
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
zred_array = sim.generate_redshift_array(numzred,1e7)

# Read sources
numsrc = 1
srcpos, srcflux = sim.read_sources("src.txt",numsrc)

# Measure time
tinit = time.time()

# Setup density
avgdens = 1e-3
ndens = avgdens*np.ones((N,N,N))
sim.ndens = ndens

# Load reference result
xfrac_ref_c2ray = t2c.XfracFile('c2ray_xfrac_reference.refbin').xi

for k in range(len(zred_array)-1):

    # Compute timestep of current redshift slice
    zi = zred_array[k]
    zf = zred_array[k+1]
    dt = sim.set_timestep(zi,zf,num_steps_between_slices)
    
    pc2r.printlog(f"\n=================================",sim.logfile)
    pc2r.printlog(f"Doing redshift {zi:.3f} to {zf:.3f}",sim.logfile)
    pc2r.printlog(f"=================================\n",sim.logfile)

    # Do num_steps_between_slices timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        pc2r.printlog(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n",sim.logfile)

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        # If cosmology is disabled in parameter, this step does nothing (checked internally by the class)
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        sim.evolve3D(dt, srcflux, srcpos)

pc2r.printlog(f"Done. Final time: {time.time() - tinit : .3f} seconds",sim.logfile)

# Compare with reference
xfrac_pyc2ray = sim.xh

# Print basic info
print("\n \n TEST RESULTS \n")
print(f"Mean ionized fraction (C2Ray):   {xfrac_ref_c2ray.mean() : .12e}")
print(f"Mean ionized fraction (pyC2Ray): {xfrac_pyc2ray.mean() : .12e}")

# Per-cell info
abserr = xfrac_pyc2ray - xfrac_ref_c2ray
relerr = abserr / xfrac_ref_c2ray

nfail = 0
aemean = abserr.mean()
aestd =  abserr.std()
aemax =  abserr.max()
aemin =  abserr.min()

remean = relerr.mean()
restd =  relerr.std()
remax =  relerr.max()
remin =  relerr.min()

print("Absolute per-cell error:")
print(f"Mean:               {aemean : .7e}",end="       ")
if np.abs( aemean ) <= 1e-8: print("PASSED")
else: print("FAILED"); nfail += 1
print(f"Standard Deviation: {aestd : .7e}",end="       ")
if np.abs( aestd ) <= 3e-7: print("PASSED")
else: print("FAILED"); nfail += 1
print(f"Maximum:            {aemax : .7e}",end="       ")
if np.abs( aemax ) <= 5e-6: print("PASSED")
else: print("FAILED"); nfail += 1
print(f"Minimum:            {aemin : .7e}",end="       ")
if np.abs( aemin ) <= 5e-6: print("PASSED")
else: print("FAILED"); nfail += 1
print("")
print("Relative per-cell error:")
print(f"Mean:               {remean : .7e}",end="       ")
if np.abs( remean ) <= 1e-7: print("PASSED")
else: print("FAILED"); nfail += 1
print(f"Standard Deviation: {restd: .7e}",end="       ")
if np.abs( restd ) <= 3e-6: print("PASSED")
else: print("FAILED"); nfail += 1
print(f"Maximum:            {remax: .7e}",end="       ")
if np.abs( remax ) <= 2e-5: print("PASSED")
else: print("FAILED"); nfail += 1
print(f"Minimum:            {remin: .7e}",end="       ")
if np.abs( remin ) <= 2e-5: print("PASSED")
else: print("FAILED"); nfail += 1
print("")
if nfail == 0: print("ALL TESTS PASSED")
else: print(f"{nfail : n} TEST(S) FAILED")
print("")