import sys

sys.path.append("../../")
import numpy as np

import pyc2ray as pc2r

# =========================================
# There seems to be a bug with the tables,
# pyc2ray and OCTA dont have quite the same
# values in some cells, probably to do with
# difference between C++ and Fortran in rounding
# =========================================

kpc = 3.086e21

# Main parameters
N = 100  # Mesh size
boxsize = 14 * kpc  # Box size in cgs
dr = boxsize / N  # Cell size in cgs
sig = 6.30e-18  # HI cross section at its ionzing frequency in cgs

# Example grids
ndens = 1e-3 * np.ones((N, N, N), order="F")  # Number density of Hydrogen in cgs
xh_av = 1.2e-3 * np.ones(
    (N, N, N), order="F"
)  # Current time-averaged ionized fraction of hydrogen


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

# Init GPU
pc2r.device_init(N)
pc2r.photo_table_to_device(photo_thin_table)

r = 10

# Raytrace
srcpos, normflux = pc2r.read_sources("sources.txt", 1, "pyc2ray_octa")
phi_ion = pc2r.raytracing.do_all_sources_octa(
    dr, normflux, srcpos, r, ndens, xh_av, sig, minlogtau, dlogtau, NumTau
)
print("OCTA: ", phi_ion.max())
srcpos, normflux = pc2r.read_sources("sources.txt", 1, "pyc2ray")
phi_ion, nsb, photonloss = pc2r.raytracing.do_all_sources(
    dr,
    normflux,
    srcpos,
    r + 1,
    r,
    ndens,
    xh_av,
    sig,
    photo_thin_table,
    minlogtau,
    dlogtau,
)
print("C2RAY: ", phi_ion.max())

pc2r.device_close()
