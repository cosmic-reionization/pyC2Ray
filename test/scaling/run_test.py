import numpy as np, time, argparse
import pickle as pkl
from tqdm import tqdm

import pyc2ray as pc2r
from pyc2ray.utils.sourceutils import format_sources
from pyc2ray.asora_core import device_init, device_close

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
N = 200                # Mesh size
use_gpu = args.gpu
fgamma = 0.02
t_s = 3*MYR

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


elif((args.batchsize != None) and (args.numsrc != None)):
    # case 2: benchmark number of sources (fix batch size)
    nsrc_range = np.array([1, 10, 100, 1000, 10000, 100000, 1000000])
    src_batch_size = np.array([int(args.batchsize)])

    # allocate memory on GPU (need just once)
    device_init(N, src_batch_size)

    # timeing array
    timings = np.empty(len(nsrc_range))

    # Read sources and convert to flux
    with open("./cosmo_sources_sorted.refbin","rb") as f:
        sources_list = pkl.load(f)

    for k, nsrc in enumerate(nsrc_range):

        fact = fgamma*msun2g*0.0044/(sim.cosmology.Om0*t_s*m_p)
        srcpos = sources_list[:nsrc,:3].T
        normflux = fact*sources_list[:nsrc,3]/1e48
elif((args.batchsize == None) and (args.numsrc == None)):
    # error
    raise ValueError('Either -batchsize or -numreps must be fixed (int).')




timings = np.empty(len(nsrc_range))

for k, nsrc in enumerate(nsrc_range):
    print(f"Doing benchmark for {nsrc:n} sources...")

    # Read sources and convert to flux
    with open("./cosmo_sources_sorted.refbin","rb") as f:
        sources_list = pkl.load(f)
    print("ok")

    t_s = 3*MYR
    fact = fgamma*msun2g*sim.cosmology.Ob0/(sim.cosmology.Om0*t_s*m_p)
    srcpos = sources_list[:nsrc,:3].T
    normflux = fact*sources_list[:nsrc,3]/1e48
    print("ok")

    srcpos_flat, normflux_flat = format_sources(srcpos, normflux)

    # Copy positions & fluxes of sources to the GPU in advance
    pc2r.evolve.libasora.source_data_to_device(srcpos_flat, normflux_flat, nsrc)
    coldensh_out_flat = np.ravel(np.zeros((N,N,N), dtype='float64'))
    phi_ion_flat = np.ravel(np.zeros((N,N,N), dtype='float64'))

    print("ok")

    ndens_flat = np.ravel(ndens).astype('float64', copy=True)

    pc2r.evolve.libasora.density_to_device(ndens_flat, N)
    xh_av_flat = np.ravel(xh_av).astype('float64', copy=True)

    print("ok")

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
