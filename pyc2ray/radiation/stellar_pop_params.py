import numpy as np
from scipy.integrate import quad
import astropy.constants as cst

class StellarPopulation:
    """
    Encapsulates the spectral and ionising properties of a stellar population
    for a given effective blackbody temperature. Computes emis_LW and Ni
    automatically from T_eff + QH_M_real + t_star.

    Parameters
    ----------
    T_eff : float
        Effective blackbody temperature [K]. Used both for pyc2ray update_tables()
        (ionisation cross-section tables) and for deriving the LW emissivity.
    QH_M_real : float
        Ionising photon rate per solar mass [photons s^-1 M_sun^-1].
        Must be supplied from a stellar evolution model/table (e.g. Schaerer 2002).
        Cannot be derived from T_eff alone — depends on stellar radius/mass.
    t_star_Myr : float
        Effective stellar lifetime [Myr] over which the ionising budget is emitted.
        Used to compute Ni. Typical values: 0.3–3 Myr for massive stars.
    fstar : float
        Star formation efficiency (fraction of halo baryons → stars).
        Astrophysical input — NOT derivable from T_eff.
    f_esc : float
        Ionising photon escape fraction.
        Astrophysical input — NOT derivable from T_eff.
    """

    # Physical constants (CGS)
    _h     = cst.h.cgs.value        # erg s
    _k_B   = cst.k_B.cgs.value      # erg/K
    _m_p   = cst.m_p.cgs.value      # g
    _M_sun = cst.M_sun.cgs.value     # g
    _Myr   = 3.15576e13             # s

    # Lyman-Werner band boundaries
    _eV2Hz    = 2.417989e14         # Hz per eV
    _nu_LW_lo = 11.2 * _eV2Hz      # lower edge of LW band [Hz]
    _nu_LW_hi = 13.6 * _eV2Hz      # upper edge = Lyman limit [Hz]
    _nu_HI    = 13.598 * _eV2Hz    # HI ionisation threshold [Hz]

    def __init__(self, T_eff, QH_M_real, t_star_Myr, fstar, f_esc):
        self.T_eff      = T_eff
        self.QH_M_real  = QH_M_real
        self.t_star_s   = t_star_Myr * self._Myr
        self.fstar      = fstar
        self.f_esc      = f_esc

        # Derived quantities (computed once at construction)
        self.emis_LW     = self._compute_emis_LW()
        self.Ni          = self._compute_Ni()
        self.phot_per_atom = self.Ni * f_esc

    # ------------------------------------------------------------------
    def _B_nu(self, nu):
        """Planck function (unnormalised — prefactor cancels in all ratios)."""
        x = self._h * nu / (self._k_B * self.T_eff)
        if x > 700.0:
            return 0.0
        return nu**3 / (np.exp(x) - 1.0)

    def _compute_emis_LW(self):
        """
        LW emissivity [erg/s/Hz/M_sun] from BB integrals.

        emis_LW = QH_M_real × h × <B_ν>_LW / ∫_{ν_ion}^{∞} B_ν/ν dν

        The stellar radius R_* cancels in the ratio — only T_eff matters for
        the spectral shape. QH_M_real provides the absolute normalisation.
        """
        dnu_LW = self._nu_LW_hi - self._nu_LW_lo

        I_LW, _ = quad(self._B_nu, self._nu_LW_lo, self._nu_LW_hi, limit=200)
        B_nu_LW_mean = I_LW / dnu_LW

        # Upper limit: 1e18 Hz (~4 keV) — well beyond stellar emission
        I_ion, _ = quad(lambda nu: self._B_nu(nu) / nu,
                        self._nu_HI, 1e18, limit=500)

        ratio = self._h * B_nu_LW_mean / I_ion  # erg per photon
        return self.QH_M_real * ratio

    def _compute_Ni(self):
        """
        Ionising photons per hydrogen atom [dimensionless].

        Ni = QH_M_real × (m_p / M_sun) × t_star
        """
        return self.QH_M_real * (self._m_p / self._M_sun) * self.t_star_s

    # ------------------------------------------------------------------
    def summary(self):
        print(f"  T_eff        = {self.T_eff:.3e} K")
        print(f"  QH_M_real    = {self.QH_M_real:.3e} photons/s/M_sun")
        print(f"  t_star       = {self.t_star_s / self._Myr:.2f} Myr")
        print(f"  emis_LW      = {self.emis_LW:.3e} erg/s/Hz/M_sun  [computed]")
        print(f"  Ni           = {self.Ni:.1f}  [computed]")
        print(f"  phot_per_atom= {self.phot_per_atom:.2f}  [Ni × f_esc, computed]")
        print(f"  fstar        = {self.fstar}  [user input]")
        print(f"  f_esc        = {self.f_esc}  [user input]")


    # Schaerer (2002) Table 3 & 4 (for lifetimes), Z=0 (metal-free Pop III), selected masses
    # Columns: [M_star/M_sun, log10(T_eff/K), log10(L/L_sun), log10(Q_H/s^-1), t_MS/Myr]
    _SCHAERER2002_POPIII = np.array([
        [   5, 4.440, 2.870, np.log10(1.097e45), 61.90],
        [   9, 4.622, 3.709, np.log10(1.794e47), 20.22],
        [  15, 4.759, 4.324, np.log10(1.398e48), 10.40],
        [  25, 4.850, 4.890, np.log10(5.446e48),  6.459],
        [  40, 4.900, 5.420, np.log10(1.873e49),  3.864],
        [  60, 4.943, 5.715, np.log10(3.481e49),  3.464],
        [  80, 4.970, 5.947, np.log10(5.938e49),  3.012],
        [ 120, 4.981, 6.243, np.log10(1.069e50),  2.521],
        [ 200, 4.999, 6.574, np.log10(2.292e50),  2.204],
        [ 300, 5.007, 6.819, np.log10(4.029e50),  2.047],
        [ 400, 5.028, 6.984, np.log10(5.573e50),  1.974],
        [ 500, 5.029, 7.106, np.log10(7.380e50),  1.899],
        [1000, 5.026, 7.444, np.log10(1.607e51),  np.nan],
    ])

    def QH_per_Msun_schaerer(M_PIIIstar_msun):
        """
        Ionising photon rate per solar mass [photons/s/M_sun] for a Pop III star
        of mass M_PIIIstar_msun, interpolated from Schaerer (2002), Table 3.
        Also returns T_eff and main-sequence lifetime.
        """
        masses   = _SCHAERER2002_POPIII[:, 0]
        logT     = _SCHAERER2002_POPIII[:, 1]
        logQH    = _SCHAERER2002_POPIII[:, 3]
        t_MS     = _SCHAERER2002_POPIII[:, 4]

        logM = np.log10(M_PIIIstar_msun)
        T_eff    = 10**np.interp(logM, np.log10(masses), logT)
        QH_star  = 10**np.interp(logM, np.log10(masses), logQH)  # photons/s for ONE star
        t_MS_val = np.interp(logM, np.log10(masses), t_MS)        # Myr

        QH_M_real = QH_star / M_PIIIstar_msun  # photons/s/M_sun
        return T_eff, QH_M_real, t_MS_val