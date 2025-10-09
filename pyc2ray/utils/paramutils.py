import re

import numpy as np
import yaml
from astropy import constants as ac
from astropy import units as u

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader


class Params:
    def __init__(self, paramfile) -> None:
        # Read in YAML parameter file
        self.paramfile = paramfile
        loader = SafeLoader
        # Configure to read scientific notation as floats
        loader.add_implicit_resolver(
            "tag:yaml.org,2002:float",
            re.compile(
                """^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$""",
                re.X,
            ),
            list("-+0123456789."),
        )
        with open(self.paramfile, "r") as f:
            self._ld = yaml.load(f, loader)

        # Now, additional quantities are computed and everything is stored as attributes

        # ============================================================
        # Abundances
        # ============================================================
        abu_he = self._ld["Abundances"]["abu_he"]
        abu_h = 1.0 - abu_he
        mu = (1.0 - abu_he) + 4.0 * abu_he
        self._ld["Abundances"]["abu_h"] = abu_h
        self._ld["Abundances"]["mu"] = mu

        for at in self._ld["Abundances"]:
            val = self._ld["Abundances"][at]
            self.setparam(at, val)

        # ============================================================
        # Atomic
        # ============================================================
        gamma = 5.0 / 3.0
        gamma1 = 1.0 - gamma
        d = {"gamma": gamma, "gamma1": gamma1}
        self._ld["Atomic"] = d

        for at in self._ld["Atomic"]:
            val = self._ld["Atomic"][at]
            self.setparam(at, val)

        # ============================================================
        # General
        # ============================================================
        self._ld["General"]["lifetime"] *= u.year

        for at in self._ld["General"]:
            val = self._ld["General"][at]
            self.setparam(at, val)

        # ============================================================
        # Log
        # ============================================================
        for at in self._ld["Log"]:
            val = self._ld["Log"][at]
            self.setparam(at, val)

        # ============================================================
        # Input
        # ============================================================
        for at in self._ld["Input"]:
            val = self._ld["Input"][at]
            self.setparam(at, val)

        # ============================================================
        # Grid
        # ============================================================
        for at in self._ld["Grid"]:
            val = self._ld["Grid"][at]
            self.setparam(at, val)

        # ============================================================
        # Output
        # ============================================================
        for at in self._ld["Output"]:
            val = self._ld["Output"][at]
            self.setparam(at, val)

        # ============================================================
        # Material
        # ============================================================
        for at in self._ld["Material"]:
            val = self._ld["Material"][at]
            self.setparam(at, val)

        # ============================================================
        # Time Integration
        # ============================================================

        for at in self._ld["Time"]:
            val = self._ld["Time"][at]
            self.setparam(at, val)

        # ============================================================
        # Various CGS constants
        # ============================================================
        self.m_p = ac.m_p.cgs
        self.c = ac.c.cgs
        self.hplanck = ac.h.cgs
        self.sigma_SB = ac.sigma_sb.cgs
        self.k_B = ac.k_B.cgs
        self.G_grav = ac.G.cgs

        ### Here lets read in the parameters manually, since many have units ###

        # Planck constant factor (usually 1)
        self.hscl_fact = self._ld["CGS"]["hscl_fact"]
        # Hydrogen recombination parameter (power law index)
        self.albpow = self._ld["CGS"][
            "albpow"
        ]  # TODO: does this and the following have have a unit ?
        # Hydrogen recombination parameter (value at 10^4 K)
        self.bh00 = self._ld["CGS"]["bh00"]
        # Helium0 recombination parameter (power law index)
        self.alcpow = self._ld["CGS"]["alcpow"]
        # Helium0 recombination parameter (value at 10^4 K)
        self.bhe00 = self._ld["CGS"]["bhe00"]
        # Helium1 recombination parameter (value at 10^4 K)
        self.bhe10 = self._ld["CGS"]["bhe10"]

        # Hydrogen ionization energy (in eV)
        self.eth0 = self._ld["CGS"]["eth0"] * u.eV
        # Hydrogen collisional ionization parameter 1
        self.xih0 = self._ld["CGS"]["xih0"]  # TODO: unit ??
        # Hydrogen collisional ionization parameter 2
        self.fh0 = self._ld["CGS"]["fh0"]
        # critical electron density at which colisions become important for y-fraction ionization (Osterbrock)
        self.n_el_crit = self._ld["CGS"]["n_el_crit"]  # TODO: unit ?
        # Helium ionization energy (in eV)
        self.ethe = self._ld["CGS"]["ethe"] * u.eV
        # Units??
        self.xihe = self._ld["CGS"]["xihe"]
        self.fhe = self._ld["CGS"]["fhe"]
        # Factors for other constants
        self.colh0_fact = self._ld["CGS"]["colh0_fact"]
        self.colhe_fact = self._ld["CGS"]["colhe_fact"]

        # These Parameters are defined using the read in ones.
        # Conversion is done using Astropy rather than hardcoded values

        # Scaled Planck constant
        self.hscl = self.hscl_fact * self.hplanck
        # Hydrogen ionization energy (in erg)
        self.hionen = self.eth0.to("erg")
        # Hydrogen ionization energy expressed in K
        self.temph0 = (self.eth0 / self.k_B).to("K")
        # Hydrogen collisional ionization parameter
        self.colh0 = self.colh0_fact * self.fh0 * self.xih0 / (self.eth0 * self.eth0)
        # Helium ionization energy (in erg)
        self.heionen = [self.ethe[0].to("erg"), self.ethe[1].to("erg")]
        # Helium ionization energy expressed in K
        tmp = self.ethe / self.k_B
        self.temphe = [tmp[0].to("K"), tmp[1].to("K")]
        # Helium collisional ionization parameter
        self.colhe = [
            self.colhe_fact
            * self.fhe[0]
            * self.xihe[0]
            / (self.ethe[0] * self.ethe[0]),
            self.colhe_fact
            * self.fhe[1]
            * self.xihe[1]
            / (self.ethe[1] * self.ethe[1]),
        ]

        # TODO: INITIALIZE THESE PARAMS (see cgsconstants.f90):
        #      # Hydrogen 0 A recombination parameter
        #      arech0
        #      # Hydrogen 0 B recombination parameter
        #      brech0
        #      # Helium   0 A recombination parameter
        #      areche0
        #      # Helium   0 B recombination parameter
        #      breche0
        #      # Helium   0 1 recombination parameter
        #      oreche0
        #
        #      # Helium   + A recombination parameter
        #      areche1
        #      # Helium   + B recombination parameter
        #      breche1
        #      # Helium   + 2 recombination parameter
        #      treche1
        #
        #
        #      # H0 collisional ionization parameter at T=temp0
        #      colli_HI
        #      # He0 collisional ionization parameter at T=temp0
        #      colli_HeI
        #      # He1 collisional ionization parameter at T=temp0
        #      colli_HeII
        #      # Fraction fo He++ -> He+ recombination photons that goes into 2photon decay
        #      v

        # ============================================================
        # Photoionization
        # ============================================================

        # These are all cross sections, so we can read them in and assign an unit in a loop
        # TODO: what unit ?? Probably cm2
        unit_cm2 = u.cm * u.cm
        for at in self._ld["Photo"]:
            val = self._ld["Photo"][at] * unit_cm2
            self.setparam(at, val)
        # HI cross section at its ionzing frequency
        self.sigma_HI_at_ion_freq *= self.freq_factor
        # sigma_HI_at_ion_freq: 6.346e-18
        # HI ionization energy in frequency
        self.ion_freq_HI = (self.eth0 / self.hplanck).to("Hz")
        # HeI ionization energy in frequency
        self.ion_freq_HeI = (self.ethe[0] / self.hplanck).to("Hz")
        # HeII ionization energy in frequency
        self.ion_freq_HeII = (self.ethe[1] / self.hplanck).to("Hz")

        # ============================================================
        # Cosmology
        # ============================================================

        # TODO: add options to the param file to customize cosmology
        from astropy.cosmology import Planck18

        self.cosmology = Planck18

        self.cosmo_id = self.cosmology.name
        self.h = self.cosmology.h
        self.Omega0 = self.cosmology.Om0
        self.Omega_B = self.cosmology.Ob0
        self.Omega_L = 1.0 - self.Omega0
        self.cmbtemp = self.cosmology.Tcmb0
        self.sigma8 = 0.8  # TODO: what is this
        self.n_s = 0.96  # TODO: what is this, is it necessary with astropy?  "slope of density power spectrum"
        self.H0 = self.cosmology.H0.cgs
        self.rho_crit_0 = self.cosmology.critical_density0.cgs

        # ============================================================
        # SED
        # ============================================================

        for at in self._ld["SED"]:
            val = self._ld["SED"][at]
            self.setparam(at, val)

        # Fix units
        self.bb_Teff *= u.K
        self.mass_nominal *= u.Msun

        # Derived quantities
        self.bb_MinFreq = self.ion_freq_HI * self.bb_MinFreq_fact
        self.bb_MaxFreq = self.ion_freq_HeII * self.bb_MaxFreq_fact
        self.pl_MinFreq = self.ion_freq_HI * self.pl_MinFreq_fact
        self.pl_MaxFreq = self.ion_freq_HeII * self.pl_MaxFreq_fact

        # Eddington Luminosity at nominal BH mass
        eddlumfact = 4 * np.pi * self.G_grav * self.m_p * self.c / ac.sigma_T
        self.EddLum = (self.mass_nominal * eddlumfact).to(u.erg / u.s)

    def setparam(self, at, val):
        if hasattr(self, at):
            raise ValueError((f"Trying to set existing attribute: {at}"))
        else:
            setattr(self, at, val)
