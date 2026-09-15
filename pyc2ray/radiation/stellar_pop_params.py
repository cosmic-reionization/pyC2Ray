"""Stellar population properties used by the LW-enabled C2Ray model.

The module keeps three quantities conceptually separate:

* ``QH_M_real``: intrinsic H-ionizing photon rate per stellar mass;
* ``phot_per_stellar_atom``: lifetime-integrated escaped ionizing photons
  per stellar baryon;
* ``phot_per_halo_baryon``: the corresponding quantity per halo baryon,
  including the star-formation efficiency.

Minihalo mode ``MHflag == 2`` represents explicit Pop III stellar mass and
must use the per-stellar-baryon quantity. Atomic-cooling halo source masses
represent halo mass and therefore use the per-halo-baryon quantity.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy.integrate import quad


# CODATA/IAU constants in CGS. Keeping this module independent of Astropy makes
# its source-property and sampling tests usable outside a full pyC2Ray install.
H_PLANCK = 6.62607015e-27
K_BOLTZMANN = 1.380649e-16
M_PROTON = 1.67262192369e-24
M_SUN = 1.988409870698051e33
MYR = 3.15576e13
EV_TO_HZ = 2.417989242e14

NU_LW_LO = 11.2 * EV_TO_HZ
NU_LW_HI = 13.6 * EV_TO_HZ
NU_HI = 13.598 * EV_TO_HZ


@dataclass(frozen=True)
class PopIIIProperties:
    """Mass-dependent properties for one or more zero-metallicity stars."""

    mass_msun: np.ndarray
    T_eff: np.ndarray
    QH_star: np.ndarray
    QH_per_msun: np.ndarray
    t_star_Myr: np.ndarray
    emis_LW_per_msun: np.ndarray

    @property
    def lifetime_s(self) -> np.ndarray:
        return self.t_star_Myr * MYR

    @property
    def Lnu_LW_star(self) -> np.ndarray:
        return self.emis_LW_per_msun * self.mass_msun

    @property
    def ionizing_photons_total(self) -> np.ndarray:
        return self.QH_star * self.lifetime_s

    @property
    def LW_energy_per_hz_total(self) -> np.ndarray:
        return self.Lnu_LW_star * self.lifetime_s


class StellarPopulation:
    """Spectral and ionizing properties of a blackbody stellar population."""

    # Schaerer (2002), zero-metallicity stellar models.
    # Columns: M/Msun, log10(Teff/K), log10(L/Lsun), log10(QH/s), tMS/Myr.
    _SCHAERER2002_POPIII = np.array(
        [
            [5, 4.440, 2.870, np.log10(1.097e45), 61.90],
            [9, 4.622, 3.709, np.log10(1.794e47), 20.22],
            [15, 4.759, 4.324, np.log10(1.398e48), 10.40],
            [25, 4.850, 4.890, np.log10(5.446e48), 6.459],
            [40, 4.900, 5.420, np.log10(1.873e49), 3.864],
            [60, 4.943, 5.715, np.log10(3.481e49), 3.464],
            [80, 4.970, 5.947, np.log10(5.938e49), 3.012],
            [120, 4.981, 6.243, np.log10(1.069e50), 2.521],
            [200, 4.999, 6.574, np.log10(2.292e50), 2.204],
            [300, 5.007, 6.819, np.log10(4.029e50), 2.047],
            [400, 5.028, 6.984, np.log10(5.573e50), 1.974],
            [500, 5.029, 7.106, np.log10(7.380e50), 1.899],
        ],
        dtype=float,
    )

    def __init__(
        self,
        T_eff: float,
        QH_M_real: float,
        t_star_Myr: float,
        fstar: float,
        f_esc_ion: float,
        f_esc_LW: float = 1.0,
    ):
        if T_eff <= 0 or QH_M_real <= 0 or t_star_Myr <= 0:
            raise ValueError("T_eff, QH_M_real and t_star_Myr must be positive.")
        if not 0.0 <= fstar <= 1.0:
            raise ValueError("fstar must lie in [0, 1].")
        if not 0.0 <= f_esc_ion <= 1.0:
            raise ValueError("f_esc_ion must lie in [0, 1].")
        if not 0.0 <= f_esc_LW <= 1.0:
            raise ValueError("f_esc_LW must lie in [0, 1].")

        self.T_eff = float(T_eff)
        self.QH_M_real = float(QH_M_real)
        self.t_star_Myr = float(t_star_Myr)
        self.t_star_s = self.t_star_Myr * MYR
        self.fstar = float(fstar)
        self.f_esc_ion = float(f_esc_ion)
        self.f_esc_LW = float(f_esc_LW)

        self.emis_LW = self._compute_emis_LW_for_temperature(
            self.T_eff, self.QH_M_real
        )
        self.Ni = self.QH_M_real * (M_PROTON / M_SUN) * self.t_star_s

        # Use these explicit names at call sites; ``phot_per_atom`` is retained
        # as a backwards-compatible alias for stellar-mass source branches.
        self.phot_per_stellar_atom = self.Ni * self.f_esc_ion
        self.phot_per_halo_baryon = (
            self.fstar * self.Ni * self.f_esc_ion
        )
        self.phot_per_atom = self.phot_per_stellar_atom

    @staticmethod
    def _planck_shape(nu: float, temperature: float) -> float:
        """Unnormalised energy Planck function B_nu (prefactor cancels)."""
        x = H_PLANCK * nu / (K_BOLTZMANN * temperature)
        if x > 700.0:
            return 0.0
        return nu**3 / np.expm1(x)

    @classmethod
    @lru_cache(maxsize=128)
    def _compute_emis_LW_for_temperature(
        cls, temperature: float, qh_per_msun: float
    ) -> float:
        """Band-averaged LW luminosity in erg/s/Hz/Msun for a blackbody."""

        def b_nu(nu):
            return cls._planck_shape(nu, temperature)

        mean_b_lw = (
            quad(b_nu, NU_LW_LO, NU_LW_HI, limit=200)[0]
            / (NU_LW_HI - NU_LW_LO)
        )
        ionizing_integral = quad(
            lambda nu: b_nu(nu) / nu, NU_HI, 1.0e18, limit=500
        )[0]
        if ionizing_integral <= 0:
            raise ValueError("The ionizing blackbody integral is zero.")
        return qh_per_msun * H_PLANCK * mean_b_lw / ionizing_integral

    @classmethod
    def popiii_properties(cls, masses_msun) -> PopIIIProperties:
        """Interpolate Schaerer properties for Pop III masses in [5, 500] Msun.

        Interpolation is logarithmic in mass for temperature, QH and LW
        emissivity and linear in log(mass) for the main-sequence lifetime.
        Extrapolation is deliberately forbidden.
        """
        masses = np.atleast_1d(np.asarray(masses_msun, dtype=float))
        table = cls._SCHAERER2002_POPIII
        mtab = table[:, 0]

        if np.any(~np.isfinite(masses)) or np.any(masses <= 0):
            raise ValueError("All Pop III stellar masses must be finite and positive.")
        if np.any(masses < mtab.min()) or np.any(masses > mtab.max()):
            raise ValueError(
                f"Pop III masses must lie in [{mtab.min():g}, {mtab.max():g}] "
                "Msun unless the stellar table is extended."
            )

        logm = np.log10(masses)
        logmtab = np.log10(mtab)
        logT = np.interp(logm, logmtab, table[:, 1])
        logQH = np.interp(logm, logmtab, table[:, 3])
        lifetime = np.interp(logm, logmtab, table[:, 4])

        # Build the BB-derived LW table only once, then interpolate it.
        qh_per_msun_tab = 10.0 ** table[:, 3] / mtab
        lw_tab = np.array(
            [
                cls._compute_emis_LW_for_temperature(10.0**log_t, qh_m)
                for log_t, qh_m in zip(table[:, 1], qh_per_msun_tab)
            ]
        )
        log_lw = np.interp(logm, logmtab, np.log10(lw_tab))

        qh_star = 10.0**logQH
        return PopIIIProperties(
            mass_msun=masses,
            T_eff=10.0**logT,
            QH_star=qh_star,
            QH_per_msun=qh_star / masses,
            t_star_Myr=lifetime,
            emis_LW_per_msun=10.0**log_lw,
        )

    @classmethod
    def from_popiii_mass(
        cls,
        mass_msun: float,
        fstar: float,
        f_esc_ion: float,
        f_esc_LW: float = 1.0,
    ) -> "StellarPopulation":
        props = cls.popiii_properties([mass_msun])
        return cls(
            T_eff=props.T_eff[0],
            QH_M_real=props.QH_per_msun[0],
            t_star_Myr=props.t_star_Myr[0],
            fstar=fstar,
            f_esc_ion=f_esc_ion,
            f_esc_LW=f_esc_LW,
        )

    @classmethod
    def QH_per_Msun_schaerer(cls, mass_msun: float):
        """Compatibility helper returning Teff, QH/Msun and lifetime."""
        props = cls.popiii_properties([mass_msun])
        return props.T_eff[0], props.QH_per_msun[0], props.t_star_Myr[0]

    def summary(self):
        print(f"  T_eff                  = {self.T_eff:.3e} K")
        print(f"  QH_M_real              = {self.QH_M_real:.3e} photon/s/Msun")
        print(f"  t_star                 = {self.t_star_Myr:.3f} Myr")
        print(f"  emis_LW                = {self.emis_LW:.3e} erg/s/Hz/Msun")
        print(f"  Ni                      = {self.Ni:.3e}")
        print(f"  photons/stellar atom   = {self.phot_per_stellar_atom:.3e}")
        print(f"  photons/halo baryon    = {self.phot_per_halo_baryon:.3e}")
        print(f"  fstar                   = {self.fstar:.4g}")
        print(f"  f_esc_ion               = {self.f_esc_ion:.4g}")
        print(f"  f_esc_LW                = {self.f_esc_LW:.4g}")


class PopIIIIMFSampler:
    """Inverse-CDF sampler for a Chabrier-like top-heavy Pop III IMF.

    The implemented probability density is

        dN/dM proportional to M**(-alpha)
        * exp[-(M_char/M)**beta]

    over the closed interval [M_min, M_max].
    """

    def __init__(
        self,
        M_min: float = 5.0,
        M_max: float = 300.0,
        M_char: float = 20.0,
        alpha: float = 2.35,
        beta: float = 1.6,
        grid_size: int = 8192,
    ):
        if not 5.0 <= M_min < M_max <= 500.0:
            raise ValueError(
                "The current Schaerer table requires 5 <= M_min < M_max <= 500."
            )
        if M_char <= 0 or alpha <= 0 or beta <= 0:
            raise ValueError("M_char, alpha and beta must be positive.")
        if grid_size < 256:
            raise ValueError("grid_size must be at least 256.")

        self.M_min = float(M_min)
        self.M_max = float(M_max)
        self.M_char = float(M_char)
        self.alpha = float(alpha)
        self.beta = float(beta)

        self._mass_grid = np.geomspace(self.M_min, self.M_max, grid_size)
        pdf = self._mass_grid ** (-self.alpha) * np.exp(
            -((self.M_char / self._mass_grid) ** self.beta)
        )
        dm = np.diff(self._mass_grid)
        cdf = np.empty_like(self._mass_grid)
        cdf[0] = 0.0
        cdf[1:] = np.cumsum(0.5 * (pdf[:-1] + pdf[1:]) * dm)
        if not np.isfinite(cdf[-1]) or cdf[-1] <= 0:
            raise ValueError("Could not normalize the requested Pop III IMF.")
        self._cdf = cdf / cdf[-1]

    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        if size < 0:
            raise ValueError("size must be non-negative.")
        if size == 0:
            return np.empty(0, dtype=float)
        return np.interp(rng.random(size), self._cdf, self._mass_grid)