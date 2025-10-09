import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

ev2fr = 0.241838e15 * u.Hz / u.eV
eth0, ethe0, ethe1 = 13.598 * u.eV, 24.587 * u.eV, 54.416 * u.eV

ion_freq_HI = (ev2fr * eth0).to("Hz").value
ion_freq_HeI = (ev2fr * ethe0).to("Hz").value
ion_freq_HeII = (ev2fr * ethe1).to("Hz").value

NumBndin1, NumBndin2, NumBndin3 = 1, 26, 20
NumFreqBnd = NumBndin1 + NumBndin2 + NumBndin3

cross_section_HI_powerlaw_index = np.zeros(NumFreqBnd)
cross_section_HeI_powerlaw_index = np.zeros(NumFreqBnd)
cross_section_HeII_powerlaw_index = np.zeros(NumFreqBnd)

# frequency
freq_min = np.zeros(NumFreqBnd)
freq_min[0] = ion_freq_HI
# fmt: off
freq_min[NumBndin1:NumBndin2] = ion_freq_HeI * np.array(
    [
        1.02, 1.05, 1.07, 1.10, 1.15,
        1.20, 1.25, 1.30, 1.35, 1.40,
        1.45, 1.50, 1.55, 1.60, 1.65,
        1.70, 1.75, 1.80, 1.85, 1.90,
        1.95, 2.00, 2.05, 2.10, 2.15,
    ]
)
# fmt: on
freq_min[NumBndin1 + NumBndin2 - 1] = ion_freq_HeII
# fmt: off
freq_min[NumBndin1 + NumBndin2 :] = ion_freq_HeII * np.array(
    [
        1.05, 1.10, 1.20, 1.40, 1.70,
        2.00, 2.50, 3.00, 4.00, 5.00,
        7.00, 10.00, 15.00, 20.00, 30.00,
        40.00, 50.00, 70.00, 90.00, 100.00,
    ]
)
# fmt: on

# HI
cross_section_HI_powerlaw_index[0] = 2.761
# fmt: off
cross_section_HI_powerlaw_index[NumBndin1 : NumBndin2 + 1] = np.array(
    [
        2.8277, 2.8330, 2.8382, 2.8432, 2.8509,
        2.8601, 2.8688, 2.8771, 2.8850, 2.8925,
        2.8997, 2.9066, 2.9132, 2.9196, 2.9257,
        2.9316, 2.9373, 2.9428, 2.9481, 2.9532,
        2.9582, 2.9630, 2.9677, 2.9722, 2.9766,
        2.9813,
    ]
)
cross_section_HI_powerlaw_index[NumBndin1 + NumBndin2 :] = np.array(
    [
        2.9884, 2.9970, 3.0088, 3.0298, 3.0589,
        3.0872, 3.1166, 3.1455, 3.1773, 3.2089,
        3.2410, 3.2765, 3.3107, 3.3376, 3.3613,
        3.3816, 3.3948, 3.4078, 3.4197, 3.4379,
    ]
)
# fmt: on

# HeI
# fmt: off
cross_section_HeI_powerlaw_index[NumBndin1 : NumBndin2 + 1] = np.array(
    [
        1.5509, 1.5785, 1.6047, 1.6290, 1.6649,
        1.7051, 1.7405, 1.7719, 1.8000, 1.8253,
        1.8486, 1.8701, 1.8904, 1.9098, 1.9287,
        1.9472, 1.9654, 1.9835, 2.0016, 2.0196,
        2.0376, 2.0557, 2.0738, 2.0919, 2.1099,
        2.1302,
    ]
)
cross_section_HeI_powerlaw_index[NumBndin1 + NumBndin2 :] = np.array(
    [
        2.1612, 2.2001, 2.2564, 2.3601, 2.5054,
        2.6397, 2.7642, 2.8714, 2.9700, 3.0528,
        3.1229, 3.1892, 3.2451, 3.2853, 3.3187,
        3.3464, 3.3640, 3.3811, 3.3967, 3.4203,
    ]
)
# fmt: on

# HeII
# fmt: off
cross_section_HeII_powerlaw_index[NumBndin1 + NumBndin2 :] = np.array(
    [
        2.6930, 2.7049, 2.7213, 2.7503, 2.7906,
        2.8300, 2.8711, 2.9121, 2.9577, 3.0041,
        3.0522, 3.1069, 3.1612, 3.2051, 3.2448,
        3.2796, 3.3027, 3.3258, 3.3472, 3.3805,
    ]
)
# fmt: on

# Cross section Hydrogen
# TODO: the values taken from Verner seem to be wrong!?!?!
sigma_HI_at_ion_freq = 6.3e-18 * u.cm**2
cross_section_HI = (
    sigma_HI_at_ion_freq * (freq_min / ion_freq_HI) ** -cross_section_HI_powerlaw_index
)

# Cross section Helium I
sigma_HeI_at_ion_freq = 7.5e-18 * u.cm**2
cross_section_HeI = (
    sigma_HeI_at_ion_freq
    * (freq_min / ion_freq_HeI) ** -cross_section_HeI_powerlaw_index
)
cross_section_HeI[freq_min < ion_freq_HeI] = 0.0

# Cross section Helium II
sigma_HeII_at_ion_freq = 1.6e-18 * u.cm**2
cross_section_HeII = (
    sigma_HeII_at_ion_freq
    * (freq_min / ion_freq_HeII) ** -cross_section_HeII_powerlaw_index
)
cross_section_HeII[freq_min < ion_freq_HeII] = 0.0

# save in tables
np.savetxt(
    "Verner1996_spectidx.txt",
    np.array(
        [
            freq_min,
            cross_section_HI_powerlaw_index,
            cross_section_HeI_powerlaw_index,
            cross_section_HeII_powerlaw_index,
        ]
    ).T,
    fmt="%.10e\t%.5f\t\t%.5f\t\t%.5f",
    header="freq [Hz]\t\tpower index HI\tpower index HeI\tpower index HeII",
)
np.savetxt(
    "Verner1996_crossect.txt",
    np.array([freq_min, cross_section_HI, cross_section_HeI, cross_section_HeII]).T,
    fmt="%.10e\t%e\t\t%e\t\t%e",
    header="freq [Hz]\t\tsigma_HI\tsigma_HeI\tsigma_HeII",
)

fig, axs = plt.subplots(figsize=(18, 7), ncols=2, nrows=1)
axs[0].set_title("Spectral Index Cross-Section (Verner+ 1996)")
axs[0].vlines(
    x=np.array([ion_freq_HI, ion_freq_HeI, ion_freq_HeII]),
    ymin=1.5,
    ymax=3.5,
    colors=["blue", "red", "lime"],
    ls="--",
)
axs[0].semilogx(
    freq_min, cross_section_HI_powerlaw_index, label="HI", marker="x", color="blue"
)
axs[0].semilogx(
    freq_min[NumBndin1:],
    cross_section_HeI_powerlaw_index[NumBndin1:],
    label="HeI",
    marker="x",
    color="red",
)
axs[0].semilogx(
    freq_min[NumBndin1 + NumBndin2 :],
    cross_section_HeII_powerlaw_index[NumBndin1 + NumBndin2 :],
    label="HeII",
    marker="x",
    color="lime",
)
axs[0].set_xlabel(r"$\nu$ [Hz]"), axs[0].set_ylabel("power law index")
axs[0].set_ylim(1.5, 3.5), axs[0].legend(loc=4)

axs[1].set_title("Cross-Section")
# plt.vlines(x=np.array([ion_freq_HI, ion_freq_HeI, ion_freq_HeII]), ymin=1.5, ymax=3.5, colors=['blue', 'red', 'lime'], ls='--')
axs[1].loglog(freq_min, cross_section_HI, label="HI", marker="x", color="blue")
axs[1].loglog(freq_min, cross_section_HeI, label="HeI", marker="x", color="red")
axs[1].loglog(freq_min, cross_section_HeII, label="HeII", marker="x", color="lime")
axs[1].set_xlabel(r"$\nu$ [Hz]"), axs[1].set_ylabel(r"$\sigma_\nu$ $[cm^2]$")

plt.savefig("plot_crossect.png", bbox_inches="tight")
plt.show(), plt.clf()
