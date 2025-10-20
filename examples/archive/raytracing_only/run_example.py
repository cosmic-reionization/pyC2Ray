import sys

sys.path.append("../../")
import matplotlib.pyplot as plt
import numpy as np

import pyc2ray as pc2r

kpc = 3.086e21


# ======================================================================
# Example to showcase the usage of the raytracing module of pyc2ray in
# standalone mode
# ======================================================================

# Main parameters
N = 300  # Mesh size
boxsize = 14 * kpc  # Box size in cgs
dr = boxsize / N  # Cell size in cgs
sig = 6.30e-18  # HI cross section at its ionzing frequency in cgs

pc2r.device_init(N)

# Raytracing Parameters
max_subbox = 1000
subboxsize = 100

# Read example sources
srcpos, normflux = pc2r.read_sources("sources.txt", 40, "pyc2ray_octa")

# Set up optical depth table
minlogtau = -20
maxlogtau = 4
NumTau = 2000
dlogtau = (maxlogtau - minlogtau) / NumTau
tau = np.empty(NumTau + 1)
tau[0] = 0.0
tau[1:] = 10 ** (minlogtau + np.arange(NumTau) * dlogtau)

# Set up radiation table
Teff = 5e4  # Temperature of black body
grey = False  # Use power-law opacity
freq0 = 3288513124000000.0  # Ionization threshhold frequency of Hydrogen in s^-1
pl_index = 2.8
radsource = pc2r.radiation.BlackBodySource(Teff, grey, freq0, pl_index)
photo_thin_table = radsource.make_photo_table(tau, freq0, 10 * freq0, 1e48)

pc2r.photo_table_to_device(photo_thin_table)
# Example grids
gen = np.random.default_rng(100)
ndens = 1e-3 * np.ones((N, N, N), order="F")  # Number density of Hydrogen in cgs
xh_av = 1.2e-3 * np.ones(
    (N, N, N), order="F"
)  # Current time-averaged ionized fraction of hydrogen

# Raytrace
# phi_ion, nsubbox, photonloss = pc2r.raytracing.do_all_sources(dr, normflux, srcpos, max_subbox, subboxsize, ndens, xh_av, sig, photo_thin_table, minlogtau, dlogtau)
phi_ion = pc2r.raytracing.do_all_sources_octa(
    dr, normflux, srcpos, subboxsize, ndens, xh_av, sig, minlogtau, dlogtau, NumTau
)

# Plot result
plt.imshow(phi_ion[:, :, 63], norm="log", cmap="inferno")
plt.show()
