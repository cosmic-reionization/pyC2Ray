import sys

sys.path.append("../../../")
import pickle as pkl
import time

import matplotlib.pyplot as plt
import numpy as np

import pyc2ray as pc2r

kpc = 3.086e21

# Main parameters
N = 400  # Mesh size
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

# numsrc = [1, 10, 50, 100, 500, 1000]
# numsrc = [300,750]
# numsrc = [1500]
numsrc = [1, 10, 50, 100, 300, 500, 750, 1000, 1500]
timings = []
mean_phi = []
max_phi = []

# do empty rt
srcpos, normflux = pc2r.read_sources("sources_more.txt", 1, "pyc2ray_octa")
phi_ion = pc2r.raytracing.do_all_sources_octa(
    dr, normflux, srcpos, 20, ndens, xh_av, sig, minlogtau, dlogtau, NumTau
)

# Raytrace
r = 50
for ns in numsrc:
    # Read example sources
    srcpos, normflux = pc2r.read_sources("sources_more.txt", ns, "pyc2ray_octa")
    print(f"Doing r = {r:.1f}")
    t1 = time.perf_counter()
    phi_ion = pc2r.raytracing.do_all_sources_octa(
        dr, normflux, srcpos, r, ndens, xh_av, sig, minlogtau, dlogtau, NumTau
    )
    t2 = time.perf_counter()
    timings.append(t2 - t1)
    mean_phi.append(phi_ion.mean())
    max_phi.append(phi_ion.max())
    print(phi_ion[8, 284, 79])
timings = np.array(timings)
mean_phi = np.array(mean_phi)
max_phi = np.array(max_phi)

pc2r.device_close()

# with open("ionrate_octa.pkl","wb") as f:
#    pkl.dump(phi_ion,f)

output = np.stack((numsrc, timings, mean_phi, max_phi), axis=1)
print(output)
np.savetxt("result_octa.out", output, ("%.2f %.6f %.6e %.6e"))
# np.savetxt("result_octa_compl.out",output,("%.2f %.6f %.6e %.6e"))
