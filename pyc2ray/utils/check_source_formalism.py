import matplotlib.pyplot as plt
import numpy as np
from astropy import constants as const
from astropy import units as u
from astropy.cosmology import Planck18 as cosmo


def classical_c2ray_model(g_gamma, Mh=None, z=None):
    if Mh is None:
        Mh = 10 ** np.linspace(9, 13, 20) * u.Msun
    if not isinstance(Mh, u.quantity.Quantity):
        Mh = Mh * u.Msun
    if z is None:
        z = np.linspace(5, 20, 30)
    mp = const.m_p
    if isinstance(z, float):
        z = np.array([z])
    Ndot = []
    tevol = []
    for zi in z:
        ts = 10 * u.Myr
        Ni = (g_gamma / mp) * (cosmo.Ob0 / cosmo.Om0) * Mh / ts
        Ndot.append(Ni.value)
        tevol.append(ts.value)
    out = {
        "tevol": np.array(tevol) * ts.unit,
        "Ndot": (np.array(Ndot) * Ni.unit).to("1/s"),
        "z": np.array(z),
        "Mh": Mh,
    }
    return out


def stellar_to_halo_fraction(
    Mhalo, f0=0.3, Mt=1e8, Mp=3e11, g1=0.49, g2=-0.61, g3=3, g4=-3, **kwargs
):
    """
    A parameterised stellar to halo relation (2011.12308, 2201.02210, 2302.06626).
    """
    # Cosmology
    # h = kwargs.get("h", kwargs.get("hlittle", 0.7))
    Ob = kwargs.get("Ob", kwargs.get("Omega_b", kwargs.get("OmegaB", 0.044)))
    Om = kwargs.get("Om", kwargs.get("Omega_m", kwargs.get("Omega0", 0.270)))

    # Double power law, motivated by UVLFs
    def dpl(M):
        return (2 * Ob / Om * f0) / ((M / Mp) ** g1 + (M / Mp) ** g2)

    # Suppression at the small-mass end
    def S_M(M):
        return (1 + (Mt / M) ** g3) ** g4

    fstar = dpl(Mhalo) * S_M(Mhalo)
    return fstar


def escape_fraction(Mhalo, f0_esc, Mp_esc, al_esc):
    return f0_esc * (Mhalo / Mp_esc) ** al_esc


def exp_model(
    Nion,
    Mh=None,
    z=None,
    al_h=0.79,
    f0=0.1,
    Mt=1e8,
    Mp=1e10,
    g1=-0.3,
    g2=-0.3,
    g3=0,
    g4=0,
    f0_esc=1,
    Mp_esc=1e10,
    al_esc=0,
):
    if Mh is None:
        Mh = 10 ** np.linspace(9, 13, 20) * u.Msun
    if not isinstance(Mh, u.quantity.Quantity):
        Mh = Mh * u.Msun
    if z is None:
        z = np.linspace(5, 20, 30)
    mp = const.m_p
    if isinstance(z, float):
        z = np.array([z])
    fstar = stellar_to_halo_fraction(
        Mh.value, f0=f0, Mt=Mt, Mp=Mp, g1=g1, g2=g2, g3=g3, g4=g4
    )
    fesc = escape_fraction(Mh.value, f0_esc, Mp_esc, al_esc)
    Ndot = []
    tevol = []
    for zi in z:
        ts = (1 / cosmo.H(zi) / (1 + zi) / al_h).to("Myr")
        Ni = (Nion / mp) * (cosmo.Ob0 / cosmo.Om0) * fstar * fesc * Mh / ts
        Ndot.append(Ni.value)
        tevol.append(ts.value)
    out = {
        "tevol": np.array(tevol) * ts.unit,
        "Ndot": (np.array(Ndot) * Ni.unit).to("1/s"),
        "z": np.array(z),
        "Mh": Mh,
    }
    return out


out_old = classical_c2ray_model(1.7)
out_exp = exp_model(
    2000, g1=0, g2=0
)  # made redshift independent to only see the impact of exp mass growth

fig, axs = plt.subplots(1, 2, figsize=(13, 5))
Mh_plot = [1e10, 1e11, 1e12]
lstyles = ["--", "-.", ":"]
for ii, Mi in enumerate(Mh_plot):
    axs[0].semilogy(
        out_old["z"],
        out_old["Ndot"][:, np.abs(out_old["Mh"].value - Mi).argmin()],
        c=f"C{ii}",
        label=f"$M_\mathrm{{h}}=10^{{{np.log10(Mi):.0f}}} M_\odot$",
    )
    axs[0].semilogy(
        out_exp["z"],
        out_exp["Ndot"][:, np.abs(out_exp["Mh"].value - Mi).argmin()],
        c=f"C{ii}",
        ls=lstyles[ii],
        label=None,
    )
axs[1].plot(out_old["z"], out_old["tevol"], c="C0", label="Classical model")
axs[1].plot(
    out_exp["z"], out_exp["tevol"], c="C0", ls=lstyles[0], label="Exponent growth"
)
for ax in axs:
    ax.legend()
    ax.set_xlabel("$z$", fontsize=15)
axs[0].set_ylabel("$\dot{N}_\gamma$ [1/s]", fontsize=15)
axs[1].set_ylabel("$t_\mathrm{lifetime}$ [Myr]", fontsize=15)
plt.tight_layout()
plt.show()
