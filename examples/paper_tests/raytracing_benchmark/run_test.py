import sys
import pyc2ray as pc2r
from pyc2ray.utils.sourceutils import format_sources
import numpy as np
import time
import argparse
import pickle as pkl

MYR = 3.15576E+13
m_p = 1.672661e-24
msun2g = 1.98892e33

parser = argparse.ArgumentParser()
parser.add_argument("-numsrc",default=None,type=int,help="Number of sources to read from the test file")
parser.add_argument("-R",default=10,type=int)
parser.add_argument("-numreps",default=10,type=int)
parser.add_argument("-o",default="benchmark_result.pkl",type=str)
args = parser.parse_args()

# fix seed for reproducibility
np.random.seed(918)

# Global parameters
paramfile = "parameters.yml"
fgamma = 0.02
outfn = str(args.o)
r_RT = args.R

sim = pc2r.C2Ray_Test(paramfile)

# Set up density and neutral fraction
sim.ndens = 1e-3 * np.ones((sim.N, sim.N, sim.N))
z = 9.938
scaling = (1+z)**3
m_H_cgs = 1.673533927065e-24 # Isotopic mass of hydrogen in grams
# dens = scaling * df.cgs_density / m_H_cgs
xh_av = np.copy(sim.xh)

print(f"Rmax = {r_RT:n} cells \n\n")

if args.numsrc is not None:
    nsrc_range = np.array([int(args.numsrc)])
else:
    nsrc_range = np.array([1, 10, 100, 1000, 10000, 100000, 1000000])

timings = np.empty(len(nsrc_range))

for k, nsrc in enumerate(nsrc_range):
    print(f"Doing benchmark for {nsrc:n} sources...")

    # Read sources and convert to flux
    t_s = 3*MYR
    fact = fgamma*msun2g*sim.cosmology.Ob0/(sim.cosmology.Om0*t_s*m_p)
    srcpos = np.random.randint(low=0, high=sim.N, size=nsrc*3).reshape(3, nsrc)
    normflux = fact*np.random.uniform(low=1e9, high=5e11, size=nsrc)/1e48

    if sim.gpu:
        srcpos_flat, normflux_flat = format_sources(srcpos, normflux)
        # Copy positions & fluxes of sources to the GPU in advance
        pc2r.evolve.libasora.source_data_to_device(srcpos_flat,normflux_flat,nsrc)
        coldensh_out_flat = np.ravel(np.zeros((sim.N, sim.N, sim.N),dtype='float64'))
        phi_ion_flat = np.ravel(np.zeros((sim.N, sim.N, sim.N),dtype='float64'))
        ndens_flat = np.ravel(sim.ndens).astype('float64',copy=True)
        pc2r.evolve.libasora.density_to_device(ndens_flat, sim.N)
        xh_av_flat = np.ravel(xh_av).astype('float64',copy=True)
    else:
        phi_ion = np.zeros((sim.N,sim.N,sim.N),order='F')
        coldensh_out = np.zeros((sim.N,sim.N,sim.N),order='F')

    t_ave = 0
    nreps = int(args.numreps)

    # For this test, we directly call the raytracing function of the extension module, rather than going
    # through an interface function like evolve3D, since we want to avoid any overheads and measure the
    # timing of ONLY the raytracing.
    for i in range(nreps):
        t1 = time.time()
        if sim.gpu:
            pc2r.evolve.libasora.do_all_sources(r_RT, coldensh_out_flat, sim.sig, sim.dr, ndens_flat, xh_av_flat, phi_ion_flat, nsrc, sim.N, sim.minlogtau, sim.dlogtau, sim.NumTau)
            pass
        else:
            pc2r.evolve.libc2ray.raytracing.do_all_sources(normflux, srcpos, sim.max_subbox, r_RT, coldensh_out, sim.sig, sim.dr, sim.ndens, xh_av, phi_ion, phi_ion, sim.loss_fraction, sim.photo_thin_table, sim.photo_thick_table, np.zeros(sim.NumTau+1),np.zeros(sim.NumTau+1), sim.minlogtau, sim.dlogtau, r_RT)
        t2 = time.time()
        t_ave += t2-t1
    t_ave /= nreps
    print(f"Raytracing took {t_ave:.5f} seconds (averaged over {nreps:n} runs).")
    timings[k] = t_ave

if sim.gpu:
    asora = "yes"
else:
    asora = "no"

src_batch_size = sim._ld["Raytracing"]["source_batch_size"]
result = {
    "Rmax" : r_RT,
    "nreps" : nreps,
    "ASORA" : asora,
    "numsrc" : nsrc_range,
    "batch_size" : src_batch_size,
    "timings" : timings
}

print("Saving Result in " + outfn + "...")
print(result)
with open(outfn,"wb") as f:
    pkl.dump(result,f)
