import pyc2ray as pc2r
import astropy.units as u
import astropy.constants as cst
import numpy as np

from pyc2ray.lib import libasora as asora
#from pyc2ray.load_extensions import load_asora
#from pyc2ray.asora_core import device_init, device_close, photo_table_to_device
from pyc2ray.utils.sourceutils import format_sources

#asora = load_asora()

# HI cross section at its ionzing frequency (weighted by freq_factor)
sigma_HI_at_ion_freq = np.float64(6.30e-18)

# fix random seed
np.random.seed(918)

# some parameters needed to be setup but do not affect the benchmark
boxsize, N = 400.*u.Mpc, 200
freq_min, freq_max = (13.598*u.eV/cst.h).to('Hz').value, (54.416*u.eV/cst.h).to('Hz').value
source_batch_size = 8
r_RT = 15.0
dr = (boxsize/N).cgs.value

minlogtau, maxlogtau, NumTau = -20.0, 4.0, 20000
tau, dlogtau = pc2r.make_tau_table(minlogtau, maxlogtau, NumTau)

radsource = pc2r.radiation.BlackBodySource(1e5, False, freq_min, sigma_HI_at_ion_freq)
photo_thin_table, photo_thick_table = radsource.make_photo_table(tau, freq_min, freq_max, 1e48)

for source_batch_size in [8, 16, 32]:
   # allocate GPU memory for the grid and sources
   asora.device_init(N, source_batch_size)

   # allocate tables to GPU device
   asora.photo_table_to_device(photo_thin_table, photo_thick_table, NumTau)

   # define density fields (in cgs units) and column density
   coldensh_out = np.ravel(np.zeros((N, N, N), dtype='float64'))
   phi_ion = np.ravel(np.zeros((N, N, N), dtype='float64'))
   ndens = np.ravel(1e-3 * np.ones((N, N, N))).astype('float64',copy=True)
   xHII = np.ravel(1e-4 * np.ones((N, N, N))).astype('float64',copy=True)

   # copy density field to GPU device
   asora.density_to_device(ndens, N)

   # define the sources position and flux
   f_gamma, t_s = 0.02, (3.*u.Myr).cgs.value

   for nsrc in [100, 1000, 10000]:
      # define some random sources
      srcpos = np.random.randint(low=0, high=N, size=(3, nsrc))
      normflux = f_gamma*np.random.uniform(low=1e9, high=5e11, size=nsrc)/1e48
      
      # format the sources conform to the C++ module
      srcpos_flat = np.ravel(srcpos.astype('int32', copy=True))
      normflux_flat = normflux.astype('float64', copy=True)
      
      print('doing:', nsrc, source_batch_size)
      # copy source list to GPU device
      asora.source_data_to_device(srcpos_flat, normflux_flat, nsrc)

      # do the raytracing for all the sources
      asora.do_all_sources(r_RT, coldensh_out, sigma_HI_at_ion_freq, dr, ndens, xHII, phi_ion, nsrc, N, minlogtau, dlogtau, NumTau)

# delocate the GPU device and free memory
asora.device_close()

