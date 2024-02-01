import numpy as np

from pyc2ray.lib import libasora_He, libasora
#from pyc2ray.asora_core import cuda_is_init, device_init, device_close, photo_table_to_device
from pyc2ray.utils.sourceutils import format_sources
from pyc2ray.radiation import make_tau_table, BlackBodySource

from astropy import units as u

ev2fr = 0.241838e15 * u.Hz/u.eV

eth0, ethe1 = 13.598*u.eV, 24.587*u.eV
ion_freq_HI = (ev2fr * eth0).cgs.value
ion_freq_HeII = (ev2fr * ethe1).cgs.value
cs_pl_idx_h = 2.8

freq_min = ion_freq_HI
freq_max = 10*ion_freq_HeII
NumTau = 2000

tau, dlogtau = make_tau_table(minlogtau=-20, maxlogtau=4, NumTau=NumTau)
radsource = BlackBodySource(temp=1e4, grey=False, freq0=ion_freq_HI, pl_index=cs_pl_idx_h, S_star_ref=1e48)

photo_thin_table, photo_thick_table = radsource.make_photo_table(tau=tau, freq_min=freq_min, freq_max=freq_max)


# from params file
R_max_LLS = 15.0
sig = 6.30e-18

# from evolve3D or init class
N = 256
NumCells = N*N*N
dr = 1.62022035/N

# define data
src_pos = np.array([[50, 20, 20], [10, 10, 10], [20, 20, 20], [70, 70, 60]]).astype('int32').T
src_flux = np.array([1.e50, 1.e51, 1.e52, 1.e54]) / 1e48
NumSrc = src_pos.shape[1]

print(src_flux)
srcpos_flat, normflux_flat = format_sources(src_pos, src_flux)

ndens = np.ones((N, N, N)) * 1.981e-07
xHI = np.ones((N, N, N)) * 1.2e-3
xHeI = np.ones((N, N, N)) * 1.2e-3
xHeII = np.ones((N, N, N)) * 1.2e-3

ndens_flat = np.ravel(ndens).astype('float64', copy=True)
xHI_av_flat = np.ravel(xHI).astype('float64', copy=True)
xHeI_av_flat = np.ravel(xHeI).astype('float64', copy=True)
xHeII_av_flat = np.ravel(xHeII).astype('float64', copy=True)
phi_ion_HI_flat = np.ravel(np.zeros((N,N,N), dtype='float64'))
phi_ion_HeI_flat = np.ravel(np.zeros((N,N,N), dtype='float64'))
phi_ion_HeII_flat = np.ravel(np.zeros((N,N,N), dtype='float64'))
coldensh_out_flat = np.ravel(np.zeros((N,N,N), dtype='float64'))

if(True):
    print('with He')
    # load the GPU
    libasora_He.device_init(N, 96)
    libasora_He.photo_table_to_device(photo_thin_table, photo_thick_table, NumTau)
    libasora_He.density_to_device(ndens_flat, N)
    libasora_He.source_data_to_device(srcpos_flat, normflux_flat, NumSrc)

    # do raytracing
    libasora_He.do_all_sources(R_max_LLS, coldensh_out_flat, sig, dr, ndens_flat, xHI_av_flat, xHeI_av_flat, xHeII_av_flat, phi_ion_HI_flat, phi_ion_HeI_flat, phi_ion_HeII_flat, NumSrc, N, -20., dlogtau, NumTau)

else:
    print('without He')

    # load the GPU
    libasora.device_init(N, 96)
    libasora.photo_table_to_device(photo_thin_table, photo_thick_table, NumTau)
    libasora.density_to_device(ndens_flat, N)
    libasora.source_data_to_device(srcpos_flat, normflux_flat, NumSrc)

    # do raytracing
    libasora.do_all_sources(R_max_LLS, coldensh_out_flat, sig, dr, ndens_flat, xHI_av_flat, phi_ion_HI_flat, NumSrc, N, -20., dlogtau, NumTau)

phi_ion_HI = phi_ion_HI_flat.reshape(N, N, N)
print(phi_ion_HI.max())
print(np.where((phi_ion_HI.max() == phi_ion_HI)))