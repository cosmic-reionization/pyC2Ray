from functools import partial

import astropy.units as u
import numpy as np
from scipy.integrate import quad_vec
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_dd
from sklearn.neighbors import KNeighborsRegressor

import pyc2ray as pc2r

# Conversion Factors.
# When doing direct comparisons with C2Ray, the difference between astropy.constants and the C2Ray values may be visible, thus we use the same exact value for the constants. This can be changed to the astropy values once consistency between the two codes has been established
# pc = 3.086e18           #(1*u.pc).to('cm').value            # C2Ray value: 3.086e18
# YEAR = 3.15576E+07      #(1*u.yr).to('s').value           # C2Ray value: 3.15576E+07
# ev2fr = 0.241838e15                     # eV to Frequency (Hz)
# ev2k = 1.0/8.617e-05                    # eV to Kelvin
# kpc = 1e3*pc                            # kiloparsec in cm
# Mpc = 1e6*pc                            # megaparsec in cm
# msun2g = 1.98892e33 #(1*u.Msun).to('g').value       # solar mass to grams
# m_p = 1.672661e-24


class StellarToHaloRelation:
    """Modelling the mass relation between dark matter halo and the residing stars/galaxies."""

    def __init__(self, model, pars, cosmo=None):
        self.cosmo = cosmo
        self.model = model
        self.Nion = pars["Nion"]
        self.f0 = pars["f0"]
        self.Mt = pars["Mt"]
        self.Mp = pars["Mp"]
        self.g1 = pars["g1"]
        self.g2 = pars["g2"]
        self.g3 = pars["g3"]
        self.g4 = pars["g4"]
        self.alph_h = pars["alpha_h"]

        if self.model == "fgamma":
            # TODO: there is something wrong with this, The photoionization gets super high like 1e-5?????? to check
            self.get = lambda Mhalo: self.cosmo.Ob0 / self.cosmo.Om0 * Mhalo * self.f0
        elif self.model == "dpl":
            self.get = self.deterministic
        elif self.model == "lognorm":
            self.get = self.stochastic_lognormal
        elif self.model == "Muv":
            self.get = self.fstar_from_Muv
        elif "spice" in self.model:
            self.get = self.deterministic
            self.spice_model = SPICE_scatterSFR(self.model)
        else:
            ValueError(
                " Selected stellar-to-halo relation model that does not exist : %s"
                % self.model
            )

    def source_liftime(self, z):
        ts = 1.0 / (self.alph_h * (1 + z) * self.cosmo.H(z=z).cgs.value)
        return ts

    def deterministic(self, Mhalo):
        fstar_mean = self.stellar_to_halo_fraction(Mhalo)
        return fstar_mean

    def stochastic_Gaussian(self, Mhalo, sigma):
        fstar_mean = self.stellar_to_halo_fraction(Mhalo)

        if isinstance(sigma, float):
            # FIXME: shouldn't the following line be = sigma * np.ones_like(Mhalo)??
            fstar_std = lambda M: sigma * np.ones_like(Mhalo)  # noqa: E731
        else:
            fstar_std = sigma

        fstar = np.clip(
            fstar_mean * (1 + np.random.normal(0, fstar_std)), a_min=0, a_max=1
        )

        return fstar

    def stochastic_lognormal(self, Mhalo, sigma=None):
        fstar_mean = self.stellar_to_halo_fraction(Mhalo)

        if isinstance(sigma, (np.ndarray, list)):
            log_fstar_std = sigma
        elif isinstance(sigma, float):
            log_fstar_std = sigma * np.ones_like(Mhalo)
        elif sigma is None:
            log_fstar_std = np.power(Mhalo / self.Mp, -1.0 / 3)

        log_fstar = np.log(fstar_mean) + np.random.normal(0, log_fstar_std)
        fstar = np.clip(a=np.exp(log_fstar), a_min=0, a_max=1)
        return fstar

    def fstar_from_Muv(self, Mhalo, z, a_s=-0.33334, b_s=4.5):
        # source life-time (for accreation mass) in cgs units
        ts = self.source_liftime(z=z)

        # mean absolute magnitude
        mean_fstar = self.stellar_to_halo_fraction(Mhalo=Mhalo)
        mean_Muv = self.UV_magnitude(fstar=mean_fstar, mdot=Mhalo / ts)

        # following Gelli+ (2024), Muv scatter is proportional to halo circular velocity: ~M^(-1/3)
        std_Muv = a_s * np.log10(Mhalo) + b_s

        # absolute magnitude with scatter
        Muv = np.random.normal(loc=mean_Muv, scale=std_Muv)

        # calibrated for 1500 Å dust-corrected rest-frame UV luminosity
        M0, k_val = 51.6, 3.64413e-36  # in [Msun/s * Hz / (s erg)]
        fstar = (
            self.cosmo.Om0
            / self.cosmo.Ob0
            * k_val
            / (Mhalo / ts)
            * np.power(10.0, (M0 - Muv) / 2.5)
        )
        return np.clip(fstar, 0.0, 1.0)

    def stellar_to_halo_fraction(self, Mhalo):
        """
        A parameterised stellar to halo relation (2011.12308, 2201.02210, 2302.06626).
        """
        # Double power law, motivated by UVLFs
        dpl = (
            2
            * self.cosmo.Ob0
            / self.cosmo.Om0
            * self.f0
            / ((Mhalo / self.Mp) ** self.g1 + (Mhalo / self.Mp) ** self.g2)
        )

        # Suppression at the small-mass end
        S_M = (1 + (self.Mt / Mhalo) ** self.g3) ** self.g4

        fstar = dpl * S_M

        return fstar

    def UV_magnitude(self, fstar, mdot):
        # corresponding to AB magnitude system (Oke 1974)
        M0 = 51.6

        # calibrated for 1500 Å dust-corrected rest-frame UV luminosity
        # k_val = 1.15e-28 # in [Msun/yr * Hz / (s erg)]
        k_val = 3.64413e-36  # in [Msun/s * Hz / (s erg)]

        M_UV = M0 - 2.5 * (
            np.log10(fstar)
            + np.log10(self.cosmo.Ob0 / self.cosmo.Om0)
            + np.log10(mdot / k_val)
        )
        return M_UV

    def sfr_SPICE(self, Mhalo, z):
        # source life-time (for accreation mass) in yr units
        ts = (self.source_liftime(z=z) * u.s).to("yr").value

        # mean fstar
        mean_fstar = self.stellar_to_halo_fraction(Mhalo=Mhalo)

        # mean star formation rate in Msun/yr units
        mean_sfr = mean_fstar * Mhalo / ts

        # get scatter from SPICE tables
        scatter_sfr = self.spice_model.get_scatter(Mhalo=np.log10(Mhalo), z=z)

        # get sfr with scatter in Msun/s units
        sfr_spice = (
            (np.random.normal(mean_sfr, scatter_sfr) * u.Msun / u.yr).to("Msun/s").value
        )

        return sfr_spice


class EscapeFraction:
    """Modelling the escape of photons from the stars/galaxies inside dark matter haloes."""

    def __init__(self, model, pars):
        self.model = model
        self.f0_esc = pars["f0_esc"]
        self.Mp_esc = pars["Mp_esc"]
        self.al_esc = pars["al_esc"]

        if self.model == "constant":
            self.get = lambda Mhalo: self.f0_esc
        elif self.model == "power" or self.model == "power_obs":
            self.get = self.deterministic
        elif self.model == "Gelli2024":
            self.get = self.fesc_Muv
        elif self.model == "thesan":
            # path to the tables
            path_tab = pc2r.__path__[0] + "/tables/fesc_thesan/"

            # get fesc tables and bins for interpolation
            tabs = np.loadtxt(path_tab + "fesc_thesan_tables.txt")
            self.redshift_tab = np.loadtxt(path_tab + "redshifts.txt")
            mass_tab = np.loadtxt(path_tab + "mass_bin.txt")
            self.mass_mid = 0.5 * (mass_tab[1:] + mass_tab[:-1])

            # use a 2D interpolation based on the redshift and mass bins
            self.interp_func = RegularGridInterpolator(
                (self.redshift_tab, self.mass_mid), tabs
            )

            self.get = self.fesc_Thesan
        else:
            ValueError(
                " Selected escaping fraction model that does not exist : %s"
                % self.model
            )

    def deterministic(self, Mhalo):
        fesc_mean = self.f0_esc * (Mhalo / self.Mp_esc) ** self.al_esc
        return np.clip(fesc_mean, 0, 1)

    def deterministic_redshift(self, z):
        fesc_mean = self.f0_esc * (1 + z) ** self.al_esc
        return np.clip(fesc_mean, 0, 1)

    def fesc_Muv(self, delta_Muv):
        # Similar to Gelli+ (2024) model
        fesc = np.exp(delta_Muv - 5)  # self.f0_esc * (delta_Muv**self.al_esc + 1.)
        # fesc[delta_Muv < 0] = self.f0_esc
        return np.clip(fesc, 0, 1)

    def fesc_Thesan(self, Mhalo, z):
        if z > self.redshift_tab.max():
            fesc = self.interp_func(
                np.array([np.full_like(Mhalo, self.redshift_tab.max()), Mhalo]).T
            )
        elif z < self.redshift_tab.min():
            fesc = self.interp_func(
                np.array([np.full_like(Mhalo, self.redshift_tab.min()), Mhalo]).T
            )
        else:
            fesc = self.interp_func(np.array([np.full_like(Mhalo, z), Mhalo]).T)

        return np.clip(fesc, 0, 1)


class BurstySFR:
    """Modelling bursty star formation"""

    def __init__(self, model, pars, alpha_h, cosmo):
        self.model = model
        self.beta1 = pars["beta1"]
        self.beta2 = pars["beta2"]
        self.tB0 = pars["tB0"]
        self.tQ_frac = pars["tQ_frac"]
        self.z0 = pars["z0"]
        self.t_rnd = pars["t_rnd"]
        self.alpha_h = alpha_h
        self.cosmo = cosmo

        self.t0 = cosmo.age(self.z0).to("Myr").value

        if self.model == "instant":
            self.get_bursty = self.instant_burst_or_quiescent_galaxies
        elif self.model == "integrate":
            ValueError(" Sorry, model not yet implemented : %s" % self.model)
        elif self.model == "no":
            ValueError(
                " You have selected %s model. You should not call this class or change the variable in the parameter file."
                % self.model
            )
        else:
            ValueError(
                " Selected burstiness model that does not exist : %s" % self.model
            )

    def time_burstiness(self, mass, z):
        if self.t_rnd:
            # FIXME: M0 used in lhs and rhs, this is a bug
            M0 = 10 ** np.random.normal(np.log10(M0), self.t_rnd)  # noqa: F821
        else:
            M0 = mass / np.exp(-self.alpha_h * (z - self.z0))

        t = self.cosmo.age(z).to("Myr").value

        # burstiness time [Myr]
        tB = (
            self.tB0
            * (M0 / 1e10) ** self.beta1
            * ((t - self.t0) * self.cosmo.H(z).to("1/Myr").value) ** self.beta2
        )

        return tB

    @np.vectorize
    def _burstiness_timescale(t_age, tB, tQ):
        """of internal use for the integrated_burst_or_quiescent_galaxies method"""
        i_time = np.floor(t_age / (tB + tQ))

        if t_age <= i_time * (tB + tQ) + tB:
            return 1
        else:
            return 0

    def integrated_burst_or_quiescent_galaxies(self, mass, z, zi, zf, cosmo):
        """This case integrate the burst or quench time withing the time-step. It return a factor between 0 and 1 for quenched (value 0) or bursting (value 1). In bewteen values indicate that the sources are quencing for a period of time withing the time-step."""
        # TODO: It is computationally expensive, for some reason, due to the quad_vec method.... to investiage

        # get burstiness and quencing time
        tB = self.time_burstiness(mass, z)
        tQ = self.tQ_frac * tB

        # get time interval limits
        ti = cosmo.age(zi).to("Myr").value - self.t0
        tf = cosmo.age(zf).to("Myr").value - self.t0

        # get time fraction that the galaxies are on
        integr = partial(self._burstiness_timescale, tB=tB, tQ=tQ)
        timefrac_on = quad_vec(integr, ti, tf)[0] / (tf - ti)

        return timefrac_on

    def instant_burst_or_quiescent_galaxies(self, mass, z):
        """This case is for instanteneous bursting or quenching. Do not account for the time integration. Mask the halo True (bursting) or False (quiescent)."""
        # get burstiness and quencing time
        tB = self.time_burstiness(mass, z)
        tQ = self.tQ_frac * tB
        # tB *= (1-self.tQ_frac)

        # get time at the corresponding redshift
        t_age = self.cosmo.age(z).to("Myr").value - self.t0
        assert t_age.all() > 0.0, (
            "Selected parameter t0 is wrong. The value of z0 is lower then the redshift of the first source file (increase the value z0)."
        )

        # find the index of the burst/quench cycle in which the time-step, t, is inside
        i_time = np.floor(t_age / (tB + tQ))

        # if True then the galaxy is bursting otherwise is quenching
        burst_or_quench = t_age <= i_time * (tB + tQ) + tB

        # print(' A total of %.2f %% of galaxies have bursty star-formation.' %(100*np.count_nonzero(burst_mask)/burst_mask.size))
        return burst_or_quench


class SPICE_scatterSFR:
    def __init__(self, model):
        """
        Initialize the KNN interpolator.

        Parameters:
        - model: string of the model for the scatter in SFR
        """
        self.model = model
        path_model = pc2r.__path__[0] + "/tables/SPICE_scatter_SFR/"
        self.redshift_fit, self.mass_fit = np.loadtxt(
            path_model + "mvir_z_bins.txt", unpack=True
        )
        if "bu" in self.model:
            self.tab = np.loadtxt(path_model + "sigma_SFR_bursty.txt", unpack=True)
        elif "hn" in self.model:
            self.tab = np.loadtxt(path_model + "sigma_SFR_hyper.txt", unpack=True)
        elif "sm" in self.model:
            self.tab = np.loadtxt(path_model + "sigma_SFR_smooth.txt", unpack=True)
        else:
            ValueError(
                " Selected SPICE star formation rate model that does not exist : %s"
                % self.model
            )

        # Create the feature matrix (m, z) and the corresponding target values
        M, Z = np.meshgrid(self.mass_fit, self.redshift_fit, indexing="ij")
        self.X_train = np.column_stack([M.ravel(), Z.ravel()])
        self.y_train = self.tab.ravel()

        # Train KNN regressor
        self.interp = KNeighborsRegressor(n_neighbors=2, weights="distance")
        self.interp.fit(self.X_train, self.y_train)

    def get_scatter(self, Mhalo, z):
        """
        Interpolates values given a mass and redshift.

        Parameters:
        - m: A single value for virial mass or 1D array
        - z: A single redshift value or 1D array
        """
        # For larger mass we assume the same scatter as the tables limit, i.e M > 10^11.325
        Mhalo = np.clip(a=Mhalo, a_min=self.mass_fit.min(), a_max=self.mass_fit.max())

        # REMARKS: strangely the K-neighbours regressor works just fine for redshift beyond the tables limits

        # allowing to pass and array an a value
        if (np.ndim(Mhalo) == 0) and (np.ndim(z) == 0):
            query_points = np.array([[Mhalo, z]])
        elif (np.ndim(Mhalo) == 1) and (np.ndim(z) == 0):
            query_points = np.vstack((Mhalo, [z] * len(Mhalo))).T
        elif (np.ndim(Mhalo) == 0) and (np.ndim(z) == 1):
            query_points = np.vstack(([Mhalo] * len(z), z)).T

        return self.interp.predict(query_points)


class Halo2Grid:
    def __init__(self, box_len, n_grid, method="nearest"):
        self.box_len = box_len
        self.n_grid = n_grid

        self.mpc_to_cm = 3.085677581491367e24  # in cm
        self.Msun_to_g = 1.988409870698051e33  # in gram
        self.pos_grid = None

    def set_halo_pos(self, pos, unit=None):
        if unit.lower() == "cm":
            self.pos_cm_to_grid(pos)
        elif unit.lower() == "mpc":
            self.pos_mpc_to_grid(pos)
        else:
            self.pos_grid = pos

    def set_halo_mass(self, mass, unit=None):
        if unit.lower() == "kg":
            self.mass_Msun = mass * 1000 / self.Msun_to_g
        elif unit.lower() in ["gram", "g"]:
            self.mass_Msun = mass / self.Msun_to_g
        elif unit.lower() == "msun":
            self.mass_Msun = mass
        else:
            print("Unknown mass units")

    def pos_cm_to_grid(self, pos_cm):
        pos_mpc = pos_cm / self.mpc_to_cm
        pos_grid = pos_mpc * self.n_grid / self.box_len
        self.pos_grid = pos_grid
        print("Halo positions converted from cm to grid units")
        return pos_grid

    def pos_mpc_to_grid(self, pos_mpc):
        pos_grid = pos_mpc * self.n_grid / self.box_len
        self.pos_grid = pos_grid
        print("Halo positions converted from Mpc to grid units")
        return pos_grid

    def construct_tree(self, **kwargs):
        pos = kwargs.get("pos", self.pos_grid)
        if pos is None:
            print('Provide the halo positions via parameter "pos".')
            return None

        print("Creating a tree...")
        kdtree = cKDTree(pos)
        self.kdtree = kdtree
        print("...done")

    def value_on_grid(self, positions, values, **kwargs):
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_dd.html
        statistic = kwargs.get("statistic", "sum")
        bins = kwargs.get("bins", self.n_grid)
        binned_mass, bin_edges, bin_num = binned_statistic_dd(
            positions, values, statistic=statistic, bins=bins
        )
        return binned_mass, bin_edges, bin_num

    def halo_mass_on_grid(self, **kwargs):
        pos = kwargs.get("pos", self.pos_grid)
        if pos is None:
            print('Provide the halo positions via parameter "pos".')
            return None
        mass = kwargs.get("mass", self.pos_grid)
        if mass is None:
            print('Provide the halo masses via parameter "mass".')
            return None
        binned_mass = kwargs.get("binned_mass")
        if binned_mass is None:
            binned_mass, bin_edges, bin_num = self.value_on_grid(
                pos, mass, statistic="sum", bins=self.n_grid
            )
        binned_pos_list = np.argwhere(binned_mass > 0)
        binned_mass_list = binned_mass[binned_mass > 0]
        return binned_pos_list, binned_mass_list

    def halo_value_on_grid(self, value, **kwargs):
        pos = kwargs.get("pos", self.pos_grid)
        if pos is None:
            print('Provide the halo positions via parameter "pos".')
            return None
        binned_value = kwargs.get("binned_value")
        if binned_value is None:
            binned_value, bin_edges, bin_num = self.value_on_grid(
                pos, value, statistic="sum", bins=self.n_grid
            )
        binned_pos_list = np.argwhere(binned_value > 0)
        binned_value_list = binned_value[binned_value > 0]
        return binned_pos_list, binned_value_list
