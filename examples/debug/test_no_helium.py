import pyc2ray as pc2r, os
import numpy as np, matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as cst

from pyc2ray.lib import libasora as asora

# HI cross section at its ionzing frequency (weighted by freq_factor)
sigma_HI_at_ion_freq = np.float64(6.30e-18)

# min and max frequency of the integral
freq_min, freq_max = (13.598*u.eV/cst.h).to('Hz').value, (54.416*u.eV/cst.h).to('Hz').value

# calculate tau value
minlogtau, maxlogtau, NumTau = -20.0, 4.0, 20000
tau, dlogtau = pc2r.make_tau_table(minlogtau, maxlogtau, NumTau)

# calculate the table
radsource = pc2r.radiation.blackbody.BlackBodySource(temp=1e5, grey=False, freq0=freq_min, pl_index=2.8)
photo_thin_table, photo_thick_table = radsource.make_photo_table(tau, freq_min, freq_max, 1e48)

minlogtau, maxlogtau = np.log10(0.1), np.log10(0.9)
tau = 10**np.linspace(minlogtau, maxlogtau, NumTau)
#tau = 10**np.linspace(minlogtau, np.log10(0.5), NumTau)
dlogtau = np.diff(np.log10(tau[1:]))[0]
tau = np.append(0.0, tau)

# number of frequency bin
photo_thick_table = np.linspace(1.0e44, 1.2e45, NumTau).astype(np.float64) #photo_thick_table[:,:3]
photo_thick_table = np.array(np.arange(NumTau)*1.0e43+1.0e44)
photo_thick_table = np.vstack((np.zeros((1, 3)), photo_thick_table))
photo_thin_table = photo_thick_table #photo_thin_table[:,:3]

boxsize, N = 1.0*u.Mpc, 3
dr = (boxsize/N).cgs.value

# number of sources done in parallel on the GPU
source_batch_size = 8

# max distance (in pixel size) that the raytracing is computed
r_RT = 3.0

# allocate GPU memory for the grid and sources batch size
asora.device_init(N, source_batch_size)

# allocate tables to GPU device
asora.photo_table_to_device(photo_thin_table, photo_thick_table, NumTau)

coldensh_out_HI = np.ravel(np.zeros((N, N, N), dtype='float64'))
phi_ion_HI = np.ravel(np.zeros((N, N, N), dtype='float64')) # in ^-1
ndens = np.ravel(1e-7 * np.ones((N, N, N))).astype('float64',copy=True) # g/cm^3
xHI = np.ravel(1e-3 * np.ones((N, N, N))).astype('float64',copy=True)

# allocate density to GPU device
asora.density_to_device(ndens, N)

# efficiency factor (converting mass to photons)
f_gamma, t_s = 1., (3.*u.Myr).cgs.value

# fix random seed
np.random.seed(918)

# define some random sources
nsrc = 1
#srcpos = np.random.randint(low=0, high=N, size=(3, nsrc))
#normflux = f_gamma*np.random.uniform(low=1e10, high=1e14, size=nsrc)/1e48
srcpos = np.array([[N//2, N//2, N//2]])
normflux = np.array([[1e56]])/1e48

# format the sources conform to the C++ module
srcpos_flat = np.ravel(srcpos.astype('int32', copy=True))
normflux_flat = np.ravel(normflux).astype('float64', copy=True)

# copy source list to GPU device
asora.source_data_to_device(srcpos_flat, normflux_flat, nsrc)

# raytracing algorithm with Helium
asora.do_all_sources(r_RT, coldensh_out_HI, sigma_HI_at_ion_freq, dr, ndens, xHI, phi_ion_HI, nsrc, N, minlogtau, dlogtau, NumTau)

# plots
fig, axs = plt.subplots(figsize=(10, 5), nrows=1, ncols=1, constrained_layout=True)

chi = coldensh_out_HI.reshape(N,N,N)
im = axs.imshow(chi[...,N//2], norm='log', cmap='viridis')
plt.colorbar(im, ax=axs, label=r'$\Gamma_\mathrm{HI}$ [$s^{-1}$]', pad=0.02, fraction=0.048)
plt.savefig('HI_no_he.png', bbox_inches='tight'), plt.clf()
np.save('chi_noHe.npy', chi)

# free memory
asora.device_close()