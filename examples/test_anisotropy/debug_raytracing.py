import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as cst

import pyc2ray as pc2r
import sys
sys.path.append('../../src/asora_anisotr/')
import libasora_anisotr as asora

# HI cross section at its ionzing frequency (weighted by freq_factor)
sigma_HI_at_ion_freq = np.float64(6.30e-18)

# min and max frequency of the integral
freq_min, freq_max = (13.598*u.eV/cst.h).to('Hz').value, (54.416*u.eV/cst.h).to('Hz').value

# calculate the table
minlogtau, maxlogtau, NumTau = -20.0, 4.0, 20000
tau, dlogtau = pc2r.make_tau_table(minlogtau, maxlogtau, NumTau)

radsource = pc2r.radiation.BlackBodySource(1e5, False, freq_min, sigma_HI_at_ion_freq)
photo_thin_table, photo_thick_table = radsource.make_photo_table(tau, freq_min, freq_max, 1e48)

boxsize, N = 50.*u.pc, 50
dr = (boxsize/N).cgs.value

# number of sources done in parallel on the GPU
source_batch_size = 8

# max distance (in pixel size) that the raytracing is computed
r_RT = 15.0

# allocate GPU memory for the grid and sources batch size
asora.device_init(N, source_batch_size, 0, 1)

# allocate tables to GPU device
asora.photo_table_to_device(photo_thin_table, photo_thick_table, NumTau)

coldensh_out = np.ravel(np.zeros((N, N, N), dtype='float64'))
phi_ion = np.ravel(np.zeros((N, N, N), dtype='float64')) # in ^-1
ndens = np.ravel(1e-3 * np.ones((N, N, N))).astype('float64',copy=True) # g/cm^3
xHII = np.ravel(1e-4 * np.ones((N, N, N))).astype('float64',copy=True)

# copy density field to GPU device
asora.density_to_device(ndens, N)

# efficiency factor (converting mass to photons)
f_gamma, t_s = 100., (3.*u.Myr).cgs.value

# fix random seed
np.random.seed(918)

# define some random sources
srcpos = np.array([N//2, N//2, N//2])
srcpos = srcpos[...,None]
normflux = np.array([f_gamma*1e18/1e48]) #f_gamma*np.random.uniform(low=1e10, high=1e14, size=nsrc)/1e48

rad_dir = np.array([1., 1., 1.])
rad_dir = rad_dir[...,None]/np.linalg.norm(rad_dir, axis=0)
cos_angl = np.array([np.cos(np.deg2rad(45))])

nsrc = len(normflux)

# format the sources conform to the C++ module
srcpos_flat = np.ravel(srcpos.astype('int32', copy=True))
normflux_flat = normflux.astype('float64', copy=True)
rad_dir_flat = np.ravel(rad_dir.astype('float64', copy=True))
cos_angle_flat = cos_angl.astype('float64', copy=True)

# copy source list to GPU device
asora.source_data_to_device(srcpos_flat, normflux_flat, rad_dir_flat, cos_angle_flat, nsrc)

asora.do_all_sources(r_RT, coldensh_out, sigma_HI_at_ion_freq, dr, ndens, xHII, phi_ion, nsrc, N, minlogtau, dlogtau, NumTau)

fig, axs = plt.subplots(figsize=(18, 5), nrows=1, ncols=3, constrained_layout=True)

phi = phi_ion.reshape(N,N,N)

im = axs[0].imshow(phi[N//2], norm='log', cmap='Oranges')
plt.colorbar(im, ax=axs[0], label=r'$\Gamma_\mathrm{HI}$ [$s^{-1}$]', pad=0.02, fraction=0.048)
im = axs[1].imshow(phi[:,N//2,:], norm='log', cmap='Oranges')
plt.colorbar(im, ax=axs[1], label=r'$\Gamma_\mathrm{HI}$ [$s^{-1}$]', pad=0.02, fraction=0.048)
im = axs[2].imshow(phi[...,N//2], norm='log', cmap='Oranges')
plt.colorbar(im, ax=axs[2], label=r'$\Gamma_\mathrm{HI}$ [$s^{-1}$]', pad=0.02, fraction=0.048)
plt.show(), plt.clf()

print(phi.max(), np.where((phi == phi.max())))
asora.device_close()
