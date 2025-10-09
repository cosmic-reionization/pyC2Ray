from functools import partial

import astropy.constants as cst
import astropy.units as u
import numpy as np
from scipy.integrate import quad, quad_vec

import pyc2ray as pc2r

# For detailed comparisons with C2Ray, we use the same exact value for the constants. This can be changed to the astropy values once consistency between the two codes has been established

# Hydrogen and Helium ionization energy (in eV)
eth0, ethe0, ethe1 = 13.598 * u.eV, 24.587 * u.eV, 54.416 * u.eV
ion_freq_HI = (eth0 / cst.h).to("Hz").value
ion_freq_HeI = (ethe0 / cst.h).to("Hz").value
ion_freq_HeII = (ethe1 / cst.h).to("Hz").value

__all__ = ["Source", "BlackBodySource", "PowerLawSource"]


class Source:
    """Radiation basic class for radiation sources for Hydrogen only"""

    def __init__(self, grey=None, freq0=None, pl_index=None, S_star_ref=None):
        self.grey = grey
        self.freq0 = freq0
        self.pl_index = pl_index
        self.R_star = 1.0
        self.S_star_ref = S_star_ref

    def SED(self, freq):
        """Initialize the Spectral Energy Density (SED)"""
        return 0

    def integrate_SED(self, f1, f2):
        res = quad(self.SED, f1, f2)
        return res[0]

    def normalize_SED(self, f1, f2):
        S_unscaled = self.integrate_SED(f1, f2)
        S_scaling = self.S_star_ref / S_unscaled
        self.R_star = np.sqrt(S_scaling) * self.R_star

    def cross_section_freq_dependence(self, freq):
        if self.grey:
            return 1.0
        else:
            return (freq / self.freq0) ** (-self.pl_index)

    # C2Ray distinguishes between optically thin and thick cells, and calculates the rates differently for those two cases. See radiation_tables.F90, lines 345 -
    def _photo_thick_integrand_vec(self, freq, tau):
        itg = self.SED(freq) * np.exp(
            -tau * self.cross_section_freq_dependence(freq)
        )  # same as L611 in radiation_tables.f90 (C2Ray with helium)
        # To avoid overflow in the exponential, check
        return np.where(
            tau * self.cross_section_freq_dependence(freq) < 700.0, itg, 0.0
        )

    def _photo_thin_integrand_vec(self, freq, tau):
        itg = (
            self.SED(freq)
            * self.cross_section_freq_dependence(freq)
            * np.exp(-tau * self.cross_section_freq_dependence(freq))
        )
        return np.where(
            tau * self.cross_section_freq_dependence(freq) < 700.0, itg, 0.0
        )

    def _heat_thick_integrand_vec(self, freq, tau):
        photo_thick = self._photo_thick_integrand_vec(
            freq, tau
        )  # same as L675 in radiation_tables.f90 (C2Ray with helium)
        return cst.h.cgs.value * (freq - ion_freq_HI) * photo_thick

    def _heat_thin_integrand_vec(self, freq, tau):
        photo_thin = self._photo_thin_integrand_vec(freq, tau)
        return cst.h.cgs.value * (freq - ion_freq_HI) * photo_thin

    def make_photo_table(self, tau, freq_min, freq_max):
        self.normalize_SED(freq_min, freq_max)
        integrand_thin = partial(self._photo_thin_integrand_vec, tau=tau)
        integrand_thick = partial(self._photo_thick_integrand_vec, tau=tau)
        table_thin = quad_vec(integrand_thin, freq_min, freq_max, epsrel=1e-12)[0]
        table_thick = quad_vec(integrand_thick, freq_min, freq_max, epsrel=1e-12)[0]
        return table_thin, table_thick

    def make_heat_table(
        self, tau, freq_min, freq_max
    ):  # soubroutine at L825 in radiation_tables.f90 (C2Ray with helium)
        self.normalize_SED(freq_min, freq_max)
        integrand_thin = partial(self._heat_thin_integrand_vec, tau=tau)
        integrand_thick = partial(self._heat_thick_integrand_vec, tau=tau)
        table_thin = quad_vec(integrand_thin, freq_min, freq_max, epsrel=1e-12)[0]
        table_thick = quad_vec(integrand_thick, freq_min, freq_max, epsrel=1e-12)[0]
        return table_thin, table_thick


class Source_Multifreq:
    """Radiation basic class for radiation sources for multi-frequency case"""

    def __init__(self, grey=None, freq0=None, pl_index=None, S_star_ref=None):
        self.grey = grey
        self.freq0 = freq0
        self.pl_index = pl_index
        self.R_star = 1.0
        self.S_star_ref = S_star_ref
        self.freqs_tab, self.crossect_HI, self.crossect_HeI, self.crossect_HeII = (
            np.loadtxt(pc2r.__path__[0] + "/tables/Verner1996_spectidx.txt")
        )

        self.NumBndin1 = np.count_nonzero(self.crossect_HI) - np.count_nonzero(
            self.crossect_HeI
        )
        self.NumBndin2 = np.count_nonzero(self.crossect_HeI) - np.count_nonzero(
            self.crossect_HeII
        )
        self.NumBndin3 = np.count_nonzero(self.crossect_HeII)

    def SED(self, freq):
        """Initialize the Spectral Energy Density (SED)"""
        return 0

    def integrate_SED(self, f1, f2):
        res = quad(self.SED, f1, f2)
        return res[0]

    def normalize_SED(self, f1, f2):
        S_unscaled = self.integrate_SED(f1, f2)
        S_scaling = self.S_star_ref / S_unscaled
        self.R_star = np.sqrt(S_scaling) * self.R_star

    def cross_section_freq_dependence(self, freq, pl_index):
        if self.grey:
            return 1.0
        else:
            return (freq / self.freq0) ** (-pl_index)

    # C2Ray distinguishes between optically thin and thick cells, and calculates the rates differently for those two cases. See radiation_tables.F90, lines 345 -
    def _photo_thick_integrand_vec(self, freq, tau, pl_index):
        itg = self.SED(freq) * np.exp(
            -tau * self.cross_section_freq_dependence(freq, pl_index)
        )  # same as L611 in radiation_tables.f90 (C2Ray with helium)

        # To avoid overflow in the exponential, check
        return np.where(
            tau * self.cross_section_freq_dependence(freq, pl_index) < 700.0, itg, 0.0
        )

    def _photo_thin_integrand_vec(self, freq, tau, pl_index):
        itg = (
            self.SED(freq)
            * self.cross_section_freq_dependence(freq, pl_index)
            * np.exp(-tau * self.cross_section_freq_dependence(freq, pl_index))
        )
        return np.where(
            tau * self.cross_section_freq_dependence(freq, pl_index) < 700.0, itg, 0.0
        )

    def _heat_thick_integrand_vec(self, freq, tau, pl_index, ion_freq):
        photo_thick = self._photo_thick_integrand_vec(
            freq, tau, pl_index
        )  # same as L675 in radiation_tables.f90 (C2Ray with helium)
        return cst.h.cgs.value * (freq - ion_freq) * photo_thick

    def _heat_thin_integrand_vec(self, freq, tau, pl_index, ion_freq):
        photo_thin = self._photo_thin_integrand_vec(freq, tau, pl_index)
        return cst.h.cgs.value * (freq - ion_freq) * photo_thin

    def make_photo_table(self, tau, freq_min, freq_max):
        table_thin_HI, table_thick_HI = np.zeros(tau.size), np.zeros(tau.size)
        table_thin_HeI, table_thick_HeI = np.zeros(tau.size), np.zeros(tau.size)
        table_thin_HeII, table_thick_HeII = np.zeros(tau.size), np.zeros(tau.size)

        for i in range(self.freqs_tab.size - 1):
            f_min, f_max = self.freqs_tab[i], self.freqs_tab[i + 1]
            self.normalize_SED(f_min, f_max)

            if (freq_min >= f_min) and (freq_max <= f_max):
                # if frequency range is within the table frequency range
                if self.crossect_HI[i] != 0:
                    integrand_thin = partial(
                        self._photo_thin_integrand_vec,
                        tau=tau,
                        pl_index=self.crossect_HI[i],
                    )
                    integrand_thick = partial(
                        self._photo_thick_integrand_vec,
                        tau=tau,
                        pl_index=self.crossect_HI[i],
                    )

                    table_thin_HI += quad_vec(
                        integrand_thin, f_min, f_max, epsrel=1e-12
                    )[0]
                    table_thick_HI += quad_vec(
                        integrand_thick, f_min, f_max, epsrel=1e-12
                    )[0]

                if self.crossect_HeI[i] != 0:
                    integrand_thin = partial(
                        self._photo_thin_integrand_vec,
                        tau=tau,
                        pl_index=self.crossect_HeI[i],
                    )
                    integrand_thick = partial(
                        self._photo_thick_integrand_vec,
                        tau=tau,
                        pl_index=self.crossect_HeI[i],
                    )

                    table_thin_HeI += quad_vec(
                        integrand_thin, f_min, f_max, epsrel=1e-12
                    )[0]
                    table_thick_HeI += quad_vec(
                        integrand_thick, f_min, f_max, epsrel=1e-12
                    )[0]

                if self.crossect_HeII[i] != 0:
                    integrand_thin = partial(
                        self._photo_thin_integrand_vec,
                        tau=tau,
                        pl_index=self.crossect_HeII[i],
                    )
                    integrand_thick = partial(
                        self._photo_thick_integrand_vec,
                        tau=tau,
                        pl_index=self.crossect_HeII[i],
                    )

                    table_thin_HeII += quad_vec(
                        integrand_thin, f_min, f_max, epsrel=1e-12
                    )[0]
                    table_thick_HeII += quad_vec(
                        integrand_thick, f_min, f_max, epsrel=1e-12
                    )[0]
            else:
                # if frequency min and max is outside the table frequency range
                pass

        return (
            table_thin_HI,
            table_thick_HI,
            table_thin_HeI,
            table_thick_HeI,
            table_thin_HeII,
            table_thick_HeII,
        )

    def make_heat_table(
        self, tau, freq_min, freq_max
    ):  # soubroutine at L825 in radiation_tables.f90 (C2Ray with helium)
        self.normalize_SED(freq_min, freq_max)
        integrand_thin = partial(self._heat_thin_integrand_vec, tau=tau)
        integrand_thick = partial(self._heat_thick_integrand_vec, tau=tau)
        table_thin = quad_vec(integrand_thin, freq_min, freq_max, epsrel=1e-12)[0]
        table_thick = quad_vec(integrand_thick, freq_min, freq_max, epsrel=1e-12)[0]
        return table_thin, table_thick


# --------------------------------------------------------------------------------------------------------------
# ------------------------------ Define new Class for Radiative Source here below ------------------------------
# --------------------------------------------------------------------------------------------------------------


class BlackBodySource(Source):
    """A point source emitting a Black-body spectrum"""

    def __init__(self, temp, grey, freq0, pl_index, S_star_ref):
        super().__init__(
            grey=grey, freq0=freq0, pl_index=pl_index, S_star_ref=S_star_ref
        )
        self.temp = temp

    def SED(self, freq):
        if ((freq * u.Hz) * cst.h / cst.k_B / (self.temp * u.K)).cgs.value < 700.0:
            # ampl = 4*np.pi* (self.R_star * u.Rsun)**2 *2.0*np.pi/(cst.c.cgs.value**2)*freq**2      # TODO: for some reason was missing freq**3 ???
            ampl = (
                8.0
                * np.pi**2
                * (self.R_star * u.Rsun) ** 2
                * cst.h
                / cst.c**2
                * (freq * u.Hz) ** 5
            )
            # FIXME: is this correct?
            sed = ampl / (
                np.exp((freq * u.Hz) * cst.h / cst.k_B / (self.temp * u.K)) - 1.0
            )
        else:
            sed = 0.0
        return sed


class PowerLawSource(Source):
    """A point source emitting a Power Law spectrum"""

    def __init__(self, EddLum, Edd_Efficiency, index, grey, freq0, pl_index):
        self.EddLum = EddLum
        self.Edd_Efficiency = Edd_Efficiency
        self.index = index
        super().__init__(
            grey=grey,
            freq0=freq0,
            pl_index=pl_index,
            S_star_ref=self.EddLum * self.Edd_Efficiency,
        )

    def SED(self, freq):
        sed = cst.h.cgs.value * freq ** (1 - self.index)
        return sed


class YggdrasilModel(Source):
    """Use Yggdrasil model for SED"""

    def __init__(self, tabname, grey, freq0, pl_index, S_star_ref):
        super().__init__(
            grey=grey, freq0=freq0, pl_index=pl_index, S_star_ref=S_star_ref
        )

        self.lamb, self.sed_tab = np.loadtxt(tabname)

    def SED(self, freq):
        sed = cst.h.cgs.value * freq ** (1 - self.index)
        return sed
