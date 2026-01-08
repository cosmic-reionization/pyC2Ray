import pyc2ray as pc2r
import numpy as np, matplotlib.pyplot as plt
import astropy.units as u
import astropy.constants as cst

# min and max frequency of the integral
freq_min, freq_max = (13.598*u.eV/cst.h).to('Hz').value, (54.416*u.eV/cst.h).to('Hz').value

# calculate the table
minlogtau, maxlogtau, NumTau = -20.0, 4.0, 20000
tau, dlogtau = pc2r.make_tau_table(minlogtau, maxlogtau, NumTau)

radsource = pc2r.radiation.blackbody.BlackBodySource_Multifreq(temp=1e5, grey=False)

photo_thin_table, photo_thick_table = radsource.make_photo_table(tau, freq_min, freq_max, 1e48)

plt.loglog(tau, photo_thick_table), plt.show(), plt.clf()
