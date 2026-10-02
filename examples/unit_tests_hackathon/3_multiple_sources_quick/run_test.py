import sys
sys.path.append("../../../")
import tools21cm as t2c
import pyc2ray as pc2r
import time
import pickle as pkl
import astropy.constants as ac
import astropy.units as u
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--gpu",action='store_true')
parser.add_argument("-numsrc",default=10,type=int,help="Number of sources to read from the test file")
args = parser.parse_args()

# Global parameters
numzred = 2                        # Number of redshift slices
num_steps_between_slices = 1        # Number of timesteps between redshift slices
paramfile = "parameters.yml"
N = 250                             # Mesh size
t_evol = 5e4
use_octa = args.gpu
nsrc = int(args.numsrc)
fgamma = 0.02

sim = pc2r.C2Ray_Test(paramfile, N, use_octa)

# Generate redshift list (test case)
zred_array = sim.generate_redshift_array(numzred,t_evol/numzred)

# Read sources and convert to flux
with open("cosmo_sources_sorted.pkl","rb") as f:
    sources_list = pkl.load(f)
t_s = 3*u.Myr.to('s')
fact = fgamma*sim.cosmology.Ob0/(sim.cosmology.Om0*t_s*ac.m_p.to('Msun').value)
srcpos = sources_list[:nsrc,:3].T
normflux = fact*sources_list[:nsrc,3]/1e48

# Set up density
df = t2c.DensityFile("dens_9.938.dat")
z = 9.938
scaling = (1+z)**3
m_H_cgs = 1.673533927065e-24 # Isotopic mass of hydrogen in grams
ndens = dens = scaling * df.cgs_density / m_H_cgs
sim.ndens = ndens

# Measure time
tinit = time.time()
    
# Loop over redshifts
for k in range(len(zred_array)-1):
    zi = zred_array[k]       # Start redshift
    zf = zred_array[k+1]     # End redshift

    # Compute timestep of current redshift slice
    dt = sim.set_timestep(zi,zf,num_steps_between_slices)
    
    # Set redshift to current slice redshift
    sim.zred = zi

    # Loop over timesteps
    for t in range(num_steps_between_slices):
        tnow = time.time()
        pc2r.printlog(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n",sim.logfile)
        pc2r.printlog(f"Mean density is: {sim.ndens.mean():.3e}, mean ionization fraction: {sim.xh.mean():.3e}",sim.logfile)
        sim.evolve3D(dt, normflux, srcpos)

# Write final output
with open(sim.results_basename + "xfrac_test_final.pkl","wb") as f:
    pkl.dump(sim.xh,f)
pc2r.printlog(f"Done. Final time: {time.time() - tinit : .3f} seconds",sim.logfile)
