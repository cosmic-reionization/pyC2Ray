import numpy as np, time, argparse
import pickle as pkl
from tqdm import tqdm

import pyc2ray as pc2r
from pyc2ray.utils.sourceutils import format_sources
from pyc2ray.asora_core import device_init, device_close

# For reproducibility set random seed
np.random.seed(918)

MYR = 3.15576E+13
m_p = 1.672661e-24
msun2g = 1.98892e33

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action='store_true')
parser.add_argument("-numsrc", default=None, type=int, help="Number of sources to read from the test file")
parser.add_argument("-R", default=10, type=int)
parser.add_argument("-batchsize", default=10, type=int)
parser.add_argument("-numreps", default=10, type=int)
parser.add_argument("-o", default="benchmark_result.pkl", type=str)
args = parser.parse_args()

# Global parameters
use_gpu = args.gpu
fgamma = 0.02
t_s = 3*MYR
sim = pc2r.C2Ray_Test("parameters.yml")
boxsize = sim.boxsize
N = sim.N
#sigma_HI_at_ion_freq = 6.30e-18
#dr = boxsize/N
#minlogtau, dlogtau, NumTau = -20., 0.0012, 20000

outfn = str(args.o)

# Set up density and neutral fraction fields
ndens = 1e-3 * np.ones((N,N,N))
xh_av = 1e-4 * np.ones((N,N,N))

# Set up maximum mean-free path in cells
r_RT = args.R
print(f"Rmax = {r_RT:n} cells \n\n")

if((args.batchsize == None) and (args.numsrc != None)):
    # case 1: benchmark sources batch size (fix number of sources)
    src_batch_size = np.array([16, 32, 64, 128])
    nsrc_range = np.array([int(args.numsrc)])
elif((args.batchsize != None) and (args.numsrc == None)):
    # case 2: benchmark number of sources (fix batch size)
    nsrc_range = np.array([1, 10, 100, 1000, 10000, 100000, 1000000])
    #src_batch_size = np.array([int(args.batchsize)])
    src_batch_size = int(args.batchsize)
    # allocate memory on GPU (need just once)
    device_init(N, src_batch_size)

    # timeing array
    timings = np.empty(len(nsrc_range))
elif((args.batchsize == None) and (args.numsrc == None)):
    # error
    raise ValueError('Either -batchsize or -numreps must be fixed (int).')


timings = np.empty(len(nsrc_range))

# Create random sources
sources_list_full = np.random.randint(low=0, high=N, size=(nsrc_range.max(), 3))
normflux_full = fgamma*np.random.uniform(low=1e9, high=5e11, size=nsrc_range.max())/1e48

for k, nsrc in enumerate(nsrc_range):
    print(f"Doing benchmark for {nsrc:n} sources...")

    # Read sources and convert to flux
    srcpos = sources_list_full[:nsrc]
    normflux = normflux_full[:nsrc]
    srcpos_flat, normflux_flat = format_sources(srcpos, normflux)

    # Copy positions & fluxes of sources to the GPU in advance
    pc2r.evolve.libasora.source_data_to_device(srcpos_flat, normflux_flat, nsrc)
    coldensh_out_flat = np.ravel(np.zeros((N,N,N), dtype='float64'))
    phi_ion_flat = np.ravel(np.zeros((N,N,N), dtype='float64'))

    #print("ok")

    ndens_flat = np.ravel(ndens).astype('float64', copy=True)

    pc2r.evolve.libasora.density_to_device(ndens_flat, N)
    xh_av_flat = np.ravel(xh_av).astype('float64', copy=True)

    #print("ok")

    t_ave = 0
    nreps = int(args.numreps)

    # For this test, we directly call the raytracing function of the extension module, rather than going
    # through an interface function like evolve3D, since we want to avoid any overheads and measure the
    # timing of ONLY the raytracing.
    for i in tqdm(range(nreps)):
        t1 = time.time()
        pc2r.evolve.libasora.do_all_sources(r_RT, coldensh_out_flat, sim.sig, sim.dr, ndens_flat, xh_av_flat, phi_ion_flat, nsrc, N, sim.minlogtau, sim.dlogtau, sim.NumTau)
        t2 = time.time()
        t_ave += t2-t1
    t_ave /= nreps
    print(f"Raytracing took {t_ave:.5f} seconds (averaged over {nreps:n} runs).")
    timings[k] = t_ave

if use_gpu:
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
