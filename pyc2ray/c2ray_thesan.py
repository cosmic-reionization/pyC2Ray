import h5py
import numpy as np
import tools21cm as t2c

import pyc2ray as pc2r

from .c2ray_base import YEAR, C2Ray, m_p
from .utils import bin_sources
from .utils.other_utils import (
    find_bins,
    get_extension_in_folder,
    get_redshifts_from_output,
)

# from .source_model import StellarToHaloRelation, BurstySFR, EscapeFraction, Halo2Grid

__all__ = ["C2Ray_Thesan"]

# m_p = 1.672661e-24

# ======================================================================
# This file contains the C2Ray_Thesan subclass of C2Ray, which is a
# version used for simulations that read in N-Body data from CubeP3M
# ======================================================================


# for curve fit
def func(x, a, b):
    return a * x + b


class C2Ray_Thesan(C2Ray):
    def __init__(self, paramfile):
        """Basis class for a C2Ray Simulation

        Parameters
        ----------
        paramfile : str
            Name of a YAML file containing parameters for the C2Ray simulation

        """
        super().__init__(paramfile)
        self.printlog('Running: "C2Ray for %d Mpc/h volume"' % self.boxsize)

        # path to tables
        path_data = pc2r.__path__[0] + "/tables/dotN_thesan/"

        # init the tables
        self.pdf_data = np.load("%spdf_dotN_thesan.npy" % path_data)
        self.popt_array = np.loadtxt("%spopt_extMhalo.txt" % path_data)
        self.mass_bins = np.loadtxt("%smass_bins.txt" % path_data)
        self.dotN_bins = np.loadtxt("%sdotN_bins.txt" % path_data)
        self.redshifts_thesan = np.loadtxt("%sredshifts.txt" % path_data)

    # =====================================================================================================
    # USER DEFINED METHODS
    # =====================================================================================================

    def ionizing_flux(self, file, z, dt, rad_feedback=False, save_Mstar=False):
        """Read sources from a C2Ray-formatted file
        Parameters
        ----------
        file : str
            Filename to read.
        ts : float
            time-step in Myrs.
        kind: str
            The kind of source model to use.

        Returns<
        -------
        srcpos : array
            Grid positions of the sources formatted in a suitable way for the chosen raytracing algorithm
        normflux : array
            Normalization of the flux of each source (relative to S_star)
        """
        S_star_ref = 1e48

        # read halo list
        srcpos_mpc, srcmass_msun = self.read_haloes(
            self.sources_basename + file, self.boxsize
        )

        # select table based on the closest redshift
        i_tab = np.argmin(np.abs(self.redshifts_thesan - z))

        # load tables
        popt = self.popt_array[i_tab, :2]
        std_opt = self.popt_array[i_tab, 2]

        if std_opt == 0:
            # mid point of the bins
            mass_mid = 0.5 * (self.mass_bins[i_tab, 1:] + self.mass_bins[i_tab, :-1])
            dotN_mid = 0.5 * (self.dotN_bins[i_tab, 1:] + self.dotN_bins[i_tab, :-1])

            # source model for pyc2ray
            dotN_pyc2ray = np.zeros_like(srcmass_msun)
            mask_ext = np.log10(srcmass_msun) <= self.mass_bins[i_tab].max()

            # index of the bin for each halo
            idx_mass = (
                np.digitize(x=np.log10(srcmass_msun), bins=self.mass_bins[i_tab]) - 1
            )

            # loop trough the
            for i_um in np.unique(idx_mass):
                mask_fit = idx_mass == i_um
                if i_um != mass_mid.size:
                    if self.pdf_data[i_tab, i_um].sum() != 0.0:
                        prob = np.nan_to_num(
                            self.pdf_data[i_tab, i_um]
                            / self.pdf_data[i_tab, i_um].sum()
                        )
                        dotN_pyc2ray[mask_fit] = 10 ** (
                            np.random.choice(a=dotN_mid, size=mask_fit.sum(), p=prob)
                            + np.random.normal(loc=0, scale=0.1, size=mask_fit.sum())
                        )
                    else:
                        # some bins in Thesan are empty, especially at high-z, due to the mass bin size I choose. Therefore, in that case use the linear relation with a small scatter.
                        dotN_pyc2ray[mask_fit] = 10 ** (
                            func(np.log10(srcmass_msun[mask_fit]), *popt)
                            + np.random.normal(loc=0.0, scale=0.1, size=mask_fit.sum())
                        )

            dotN_pyc2ray[~mask_ext] = 10 ** (
                func(np.log10(srcmass_msun[~mask_ext]), *popt)
                + np.random.normal(loc=0.0, scale=0.1, size=(1 - mask_ext).sum())
            )
        else:
            dotN_pyc2ray = 10 ** (
                func(np.log10(srcmass_msun), *popt)
                + np.random.normal(loc=0, scale=std_opt, size=srcmass_msun.size)
            )

        # TODO: this is for testing
        # self.dotN = dotN_pyc2ray

        # sum together masses into a mesh grid and get a list of the source positon and mass
        srcpos, dotN = bin_sources(
            srcpos_mpc=srcpos_mpc,
            mstar_msun=dotN_pyc2ray,
            boxsize=self.boxsize / self.cosmology.h,
            meshsize=self.N + 1,
        )

        # normalize flux
        normflux = dotN / S_star_ref

        # calculate total number of ionizing photons
        self.tot_phots = np.sum(normflux * dt * S_star_ref)

        self.printlog(
            "\n---- Reading source file with total of %d ionizing source:\n%s"
            % (normflux.size, file)
        )
        self.printlog(" Total Flux : %e [1/s]" % np.sum(normflux * S_star_ref))
        self.printlog(" Total number of ionizaing photons : %e" % self.tot_phots)
        self.printlog(" Source lifetime : %f Myr" % (dt / (1e6 * YEAR)))
        self.printlog(
            " min, max halo (grid) mass : %.3e  %.3e [Msun] and min, mean, max number of ionising sources : %.3e  %.3e  %.3e [1/s]"
            % (
                srcmass_msun.min(),
                srcmass_msun.max(),
                normflux.min() * S_star_ref,
                normflux.mean() * S_star_ref,
                normflux.max() * S_star_ref,
            )
        )

        return srcpos, normflux

    def read_haloes(self, halo_file, box_len):
        """Read haloes from a file.

        Parameters
        ----------
        halo_file : str
            Filename to read

        Returns
        -------
        srcpos_mpc : array
            Positions of the haloes in Mpc.
        srcmass_msun : array
            Masses of the haloes in Msun.
        """

        if halo_file.endswith(".hdf5"):
            # Read haloes from a CUBEP3M file format converted in hdf5.
            f = h5py.File(halo_file)
            h = f.attrs["h"]
            srcmass_msun = f["mass"][:] / h  # Msun
            srcpos_mpc = f["pos"][:] / h  # Mpc
            f.close()
        elif halo_file.endswith(".dat"):
            # Read haloes from a CUBEP3M file format.
            hl = t2c.HaloCubeP3MFull(filename=halo_file, box_len=box_len)
            h = self.h
            srcmass_msun = hl.get(var="m") / h  # Msun
            srcpos_mpc = hl.get(var="pos") / h  # Mpc
        elif halo_file.endswith(".txt"):
            # Read haloes from a PKDGrav converted in txt.
            hl = np.loadtxt(halo_file)
            srcmass_msun = hl[:, 0] / self.cosmology.h  # Msun
            srcpos_mpc = hl[:, 1:] + self.boxsize / 2  # Mpc/h

            # apply periodic boundary condition shift
            srcpos_mpc[srcpos_mpc > self.boxsize] = (
                self.boxsize - srcpos_mpc[srcpos_mpc > self.boxsize]
            )
            srcpos_mpc[srcpos_mpc < 0.0] = self.boxsize + srcpos_mpc[srcpos_mpc < 0.0]
            srcpos_mpc /= self.cosmology.h  # Mpc
        return srcpos_mpc, srcmass_msun

    def read_density(self, fbase, z=None):
        """Read coarser density field from C2Ray-formatted file

        This method is meant for reading density field run with either N-body or hydro-dynamical simulations. The field is then smoothed on a coarse mesh grid.

        Parameters
        ----------
        fbase : string
            the file name (cwithout the path) of the file to open

        """
        file = self.density_basename + fbase
        rdr = t2c.Pkdgrav3data(self.boxsize, self.N, Omega_m=self.cosmology.Om0)
        self.ndens = (
            self.cosmology.critical_density0.cgs.value
            * self.cosmology.Ob0
            * (1.0 + rdr.load_density_field(file))
            / (self.mean_molecular * m_p)
            * (1 + z) ** 3
        )
        self.printlog("\n---- Reading density file:\n  %s" % file)
        self.printlog(
            " min, mean and max density : %.3e  %.3e  %.3e [1/cm3]"
            % (self.ndens.min(), self.ndens.mean(), self.ndens.max())
        )

    # =====================================================================================================
    # Below are the overridden initialization routines specific to the f_star case
    # =====================================================================================================

    def _redshift_init(self):
        """Initialize time and redshift counter"""
        self.zred_density = np.loadtxt(self.density_basename + "redshift_density.txt")
        self.zred_sources = np.loadtxt(self.sources_basename + "redshift_sources.txt")
        if self.resume:
            # get the resuming redshift
            self.zred = np.min(get_redshifts_from_output(self.results_basename))
            _, self.prev_zdens = find_bins(self.zred, self.zred_density)
            _, self.prev_zsourc = find_bins(self.zred, self.zred_sources)
        else:
            self.prev_zdens = -1
            self.prev_zsourc = -1
            self.zred = self.zred_0

        self.time = self.zred2time(self.zred)

    def _material_init(self):
        """Initialize material properties of the grid"""
        if self.resume:
            # get fields at the resuming redshift
            self.ndens = self.read_density(
                fbase="CDM_200Mpc_2048.%05d.den.256.0" % self.resume, z=self.prev_zdens
            )

            # get extension of the output file
            ext = get_extension_in_folder(path=self.results_basename)
            if ext == ".dat":
                fname = "%sxfrac_z%.3f.dat" % (self.results_basename, self.zred)
                self.xh = t2c.read_cbin(filename=fname, bits=64, order="F")
                self.phi_ion = t2c.read_cbin(
                    filename="%sIonRates_z%.3f.dat"
                    % (self.results_basename, self.zred),
                    bits=32,
                    order="F",
                )
            elif ext == ".npy":
                fname = "%sxfrac_z%.3f.npy" % (self.results_basename, self.zred)
                self.xh = np.load(fname)
                self.phi_ion = np.load(
                    "%sIonRates_z%.3f.npy" % (self.results_basename, self.zred)
                )
            else:
                raise FileNotFoundError(
                    " Resume file not found: %sxfrac_%.3f.npy"
                    % (self.results_basename, self.zred)
                )

            self.printlog("\n---- Reading ionized fraction field:\n %s" % fname)
            self.printlog(
                " min, mean and max density : %.5e  %.5e  %.5e"
                % (self.xh.min(), self.xh.mean(), self.xh.max())
            )

            # TODO: implement heating
            temp0 = self._ld["Material"]["temp0"]
            self.temp = temp0 * np.ones(self.shape, order="F")
        else:
            xh0 = self._ld["Material"]["xh0"]
            temp0 = self._ld["Material"]["temp0"]
            avg_dens = self._ld["Material"]["avg_dens"]

            self.ndens = avg_dens * np.empty(self.shape, order="F")
            self.xh = xh0 * np.ones(self.shape, order="F")
            self.temp = temp0 * np.ones(self.shape, order="F")
            self.phi_ion = np.zeros(self.shape, order="F")

    def _sources_init(self):
        """Initialize settings to read source files"""
        self.printlog(" --- You are using the Thesan source model so:")
        self.printlog(" NO stallar-to-halo relaction model.")
        self.printlog(" NO stellar accretion model.")
        self.printlog(" NO bustiness model for the star formation history.")
        self.printlog(" NO escaping fraction model.")
        self.printlog(" Instead reading fit tables created from Thesan simulations.")
