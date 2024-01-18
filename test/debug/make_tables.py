import pyc2ray as p2c
import matplotlib.pyplot as plt

from astropy import units as u

ev2fr = 0.241838e15 * u.Hz/u.eV

eth0, ethe1 = 13.598*u.eV, 24.587*u.eV
ion_freq_HI = (ev2fr * eth0).cgs.value
ion_freq_HeII = (ev2fr * ethe1).cgs.value
cs_pl_idx_h = 2.8

freq_min = ion_freq_HI
freq_max = 10*ion_freq_HeII

tau, dlogtau = p2c.radiation.make_tau_table(minlogtau=-20, maxlogtau=4, NumTau=2000)
radsource = p2c.radiation.spectra.BlackBodySource(temp=1e4, grey=False, freq0=ion_freq_HI, pl_index=cs_pl_idx_h, S_star_ref=1e48)
#radsource = p2c.radiation.spectra.PowerLawSource(EddLum=1.38e44, Edd_Efficiency=1., index=1., grey=False, freq0=ion_freq_HI, pl_index=cs_pl_idx_h)

photo_thin_table, photo_thick_table = radsource.make_photo_table(tau=tau, freq_min=freq_min, freq_max=freq_max)

plt.loglog(tau, photo_thick_table, label='thick')
plt.loglog(tau, photo_thin_table, label='thin')
plt.xlabel(r'$\tau_{HI}$'), plt.ylabel(r'$\Gamma(\tau)$')
plt.legend()
plt.show(), plt.clf()
