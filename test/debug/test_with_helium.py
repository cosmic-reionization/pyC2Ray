import pyc2ray as pc2r, os
import numpy as np, matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as cst

from pyc2ray.lib import libasora_He as asora_he

# min and max frequency of the integral
freq_min, freq_max = (13.598*u.eV/cst.h).to('Hz').value, (54.416*u.eV/cst.h).to('Hz').value

# calculate tau value
minlogtau, maxlogtau, NumTau = -20.0, 4.0, 20000
tau, dlogtau = pc2r.make_tau_table(minlogtau, maxlogtau, NumTau)

# calculate the table
radsource = pc2r.radiation.blackbody.BlackBodySource_Multifreq(1e5, False)
photo_thin_table, photo_thick_table = radsource.make_photo_table(tau, freq_min, freq_max, 1e48)
heat_thin_table, heat_thick_table = radsource.make_heat_table(tau, freq_min, freq_max, 1e48)

#minlogtau, maxlogtau = np.log10(0.1), np.log10(0.9)
#tau = 10**np.linspace(minlogtau, maxlogtau, NumTau)
#tau = 10**np.linspace(minlogtau, np.log10(0.5), NumTau)
#dlogtau = np.diff(np.log10(tau[1:]))[0]
#tau = np.append(0.0, tau)

# read cross section
freq, sigma_HI, sigma_HeI, sigma_HeII = np.loadtxt('/home/mbianco/codes/pyC2Ray/pyc2ray/tables/multifreq/Verner1996_crossect.txt', unpack=True)
sigma_HI = sigma_HI.ravel()
sigma_HeI = sigma_HeI.ravel()
sigma_HeII = sigma_HeII.ravel()

# number of frequency bin
#photo_thick_table = np.linspace(1.0e44, 1.2e45, (NumTau+1)*3).reshape(((NumTau+1), 3)).astype(np.float64)
#photo_thick_table = np.array(np.arange((NumTau+1)*3)*1.0e43+1.0e44).reshape((NumTau+1), 3)
#photo_thin_table = photo_thick_table
#numb1, numb2, numb3 = 1, 1, 1
numb1, numb2, numb3 = 1, 26, 20
NumFreq = numb1+numb2+numb3

assert photo_thin_table.shape[0] == NumFreq

boxsize, N = 1.0*u.Mpc, 3
dr = (boxsize/N).cgs.value

# number of sources done in parallel on the GPU
source_batch_size = 1

# max distance (in pixel size) that the raytracing is computed
r_RT = 3.0

# allocate GPU memory for the grid and sources batch size
asora_he.device_init(N, source_batch_size, NumFreq)

# allocate tables to GPU device
#asora_he.photo_table_to_device(photo_thin_table.ravel(), photo_thick_table.ravel(), NumTau, NumFreq)
asora_he.tables_to_device(photo_thin_table.ravel(), photo_thick_table.ravel(), heat_thin_table.ravel(), heat_thick_table.ravel(), NumTau, NumFreq)

# define fields
coldensh_out_HI = np.ravel(np.zeros((N, N, N), dtype='float64'))
coldensh_out_HeI = np.ravel(np.zeros((N, N, N), dtype='float64'))
coldensh_out_HeII = np.ravel(np.zeros((N, N, N), dtype='float64'))
phi_ion_HI = np.ravel(np.zeros((N, N, N), dtype='float64')) 
phi_ion_HeI = np.ravel(np.zeros((N, N, N), dtype='float64'))
phi_ion_HeII = np.ravel(np.zeros((N, N, N), dtype='float64'))
phi_heat_HI = np.ravel(np.zeros((N, N, N), dtype='float64')) 
phi_heat_HeI = np.ravel(np.zeros((N, N, N), dtype='float64'))
phi_heat_HeII = np.ravel(np.zeros((N, N, N), dtype='float64'))
ndens = np.ravel(1e-7 * np.ones((N, N, N))).astype('float64',copy=True) # g/cm^3
xHI = np.ravel(1e-3 * np.ones((N, N, N))).astype('float64',copy=True)
xHeI = np.ravel(1e-3 * np.ones((N, N, N))).astype('float64',copy=True)
xHeII = np.ravel(1e-3 * np.ones((N, N, N))).astype('float64',copy=True)

# copy density field to GPU device
asora_he.density_to_device(ndens, N)

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
asora_he.source_data_to_device(srcpos_flat, normflux_flat, nsrc)

# raytracing algorithm with Helium
asora_he.do_all_sources(r_RT, coldensh_out_HI, coldensh_out_HeI, coldensh_out_HeII, sigma_HI, sigma_HeI, sigma_HeII, numb1, numb2, numb3, dr, ndens, xHI, xHeI, xHeII, phi_ion_HI, phi_ion_HeI, phi_ion_HeII, phi_heat_HI, phi_heat_HeI, phi_heat_HeII, nsrc, N, minlogtau, dlogtau, NumTau)

# plots
fig, axs = plt.subplots(figsize=(12, 8), nrows=1, ncols=3, constrained_layout=True)

chi = coldensh_out_HI.reshape(N,N,N)
im = axs[0].imshow(chi[...,N//2], norm='log', cmap='viridis')
plt.colorbar(im, ax=axs[0], label=r'$\Gamma_\mathrm{HI}$ [$s^{-1}$]', pad=0.02, fraction=0.048)

chei = coldensh_out_HeI.reshape(N,N,N)
im = axs[1].imshow(chei[...,N//2], norm='log', cmap='viridis')
plt.colorbar(im, ax=axs[1], label=r'$N_\mathrm{HI}$ [$s^{-1}$]', pad=0.02, fraction=0.048)

cheii = coldensh_out_HeII.reshape(N,N,N)
im = axs[2].imshow(cheii[...,N//2], norm='log', cmap='viridis')
plt.colorbar(im, ax=axs[2], label=r'$N_\mathrm{HI}$ [$s^{-1}$]', pad=0.02, fraction=0.048)
plt.savefig('HI_with_he.png', bbox_inches='tight'), plt.clf()
np.save('chi_withHe.npy', chi)

# free memory
asora_he.device_close()