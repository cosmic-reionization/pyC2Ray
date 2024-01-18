import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u
import astropy.constants as cst

ev2fr = 0.241838e15 * u.Hz/u.eV
eth0, ethe0, ethe1 = 13.598*u.eV, 24.587*u.eV, 54.416*u.eV

ion_freq_HI = (ev2fr * eth0).to('Hz').value
ion_freq_HeI = (ev2fr * ethe0).to('Hz').value
ion_freq_HeII = (ev2fr * ethe1).to('Hz').value


def cross_section_Verner1996(E, E0, sigma0, ya, P, yw, y0, y1):
    # Cross-section fit by Verner+ 1996 (Eq. 1 and Table 1)
    assert E.unit == u.eV 
    assert E0.unit == u.eV
    assert sigma0.unit == u.cm**2

    x = (E / E0).value - y0
    y = np.sqrt(x**2 + y1**2)
    return sigma0.value * ((x - 1)**2 + yw**2) * y**(0.5*P - 5.5) * (1 + np.sqrt(y/ya))**(-P)

# define frequency
NumBndin1, NumBndin2, NumBndin3 = 10, 30, 50

freq_max = 100*ion_freq_HeII
freq_HI = np.linspace(ion_freq_HI, ion_freq_HeI, NumBndin1) * u.Hz
freq_HeI = np.linspace(ion_freq_HeI, ion_freq_HeII, NumBndin2) * u.Hz
freq_HeII = np.linspace(ion_freq_HeII, ion_freq_HeII*100, NumBndin3) * u.Hz
frequency = np.hstack((np.hstack((freq_HI, freq_HeI)), freq_HeII))

# get cross-sections for HI
energy = (frequency*cst.h).to('eV')
cross_section_HI = cross_section_Verner1996(E=energy, E0=4.298e-1*u.eV, sigma0=5.475e-14*u.cm**2, ya=32.88, P=2.963, yw=0, y0=0, y1=0)

# get cross-sections for HeI
energy = (frequency[freq_HI.size:]*cst.h).to('eV')
cross_section_HeI = cross_section_Verner1996(E=energy, E0=1.361e1*u.eV, sigma0=9.492e-16*u.cm**2, ya=1.469, P=3.188, yw=2.039, y0=4.434e-1, y1=2.136)

# get cross-sections for HeII
energy = (frequency[freq_HI.size+freq_HeI.size:]*cst.h).to('eV')
cross_section_HeII = cross_section_Verner1996(E=energy, E0=1.720*u.eV, sigma0=1.369e-14*u.cm**2, ya=3.288e1, P=2.963, yw=0., y0=0., y1=0.)

# get total cross-sections for all species
energy = (frequency*cst.h).to('eV')
cross_section_tot = cross_section_HI.copy()
cross_section_tot[freq_HI.size:] += cross_section_HeI
cross_section_tot[freq_HI.size+freq_HeI.size:] += cross_section_HeII

# plot cross-sections
fig = plt.figure(figsize=(12, 8))
plt.loglog(frequency, cross_section_HI, label=r'$\sigma_{\rm HI}$', marker='o', markersize=2, color='blue')
plt.loglog(frequency[freq_HI.size:], cross_section_HeI, label=r'$\sigma_{\rm HeI}$', marker='o', markersize=2, color='red')
plt.loglog(frequency[freq_HI.size+freq_HeI.size:], cross_section_HeII, label=r'$\sigma_{\rm HeII}$', marker='o', markersize=2, color='lime')
plt.loglog(frequency, cross_section_tot, label=r'$\sigma_{\rm tot}$', ls='-', color='black')

plt.vlines(x=np.array([ion_freq_HI, ion_freq_HeI, ion_freq_HeII]), ymin=cross_section_HI.min(), ymax=cross_section_HI.max(), colors=['blue', 'red', 'lime'], ls='--')
plt.xlabel(r'$\nu$ [%s]' %frequency.unit), plt.ylabel(r'$\sigma_\mathcal{X}$ [cm$^2$]')
plt.legend(), plt.title('Cross-Section (Verner+ 1996, Eq. 1, Table 1)')
plt.savefig('crosssect_Verner1996.png', bbox_inches='tight')
plt.show(), plt.clf()