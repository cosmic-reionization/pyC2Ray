import h5py
import numpy as np
import tools21cm as t2c
from astropy.cosmology import FlatLambdaCDM

from .c2ray_base import YEAR, C2Ray, Mpc, msun2g
from .utils import get_source_redshifts
from .utils.other_utils import find_bins, get_redshifts_from_output

__all__ = ["C2Ray_244Test"]

m_p = 1.672661e-24

# ======================================================================
# This file contains the C2Ray_CubeP3M subclass of C2Ray, which is a
# version used for simulations that read in N-Body data from CubeP3M
# ======================================================================


class C2Ray_244Test(C2Ray):
    def __init__(self, paramfile, Nmesh, use_gpu, use_mpi):
        """Basis class for a C2Ray Simulation

        Parameters
        ----------
        paramfile : str
            Name of a YAML file containing parameters for the C2Ray simulation
        Nmesh : int
            Mesh size (number of cells in each dimension)
        use_gpu : bool
            Whether to use the GPU-accelerated ASORA library for raytracing

        """
        super().__init__(paramfile, Nmesh, use_gpu, use_mpi)
        self.printlog('Running: "C2Ray for 244 Mpc/h test"')

    # =====================================================================================================
    # TIME-EVOLUTION METHODS
    # =====================================================================================================
    def set_timestep(self, z1, z2, num_timesteps):
        """Compute timestep to use between redshift slices

        Parameters
        ----------
        z1 : float
            Initial redshift
        z2 : float
            Next redshift
        num_timesteps : int
            Number of timesteps between the two slices

        Returns
        -------
        dt : float
            Timestep to use in seconds
        """
        t2 = self.zred2time(z2)
        t1 = self.zred2time(z1)
        dt = (t2 - t1) / num_timesteps
        return dt

    def cosmo_evolve(self, dt):
        """Evolve cosmology over a timestep

        Note that if cosmological is set to false in the parameter file, this
        method does nothing!

        Following the C2Ray convention, we set the redshift according to the
        half point of the timestep.
        """
        # Time step
        t_now = self.time
        t_half = t_now + 0.5 * dt
        t_after = t_now + dt
        self.printlog(" This is time : %f\t %f" % (t_now / YEAR, t_after / YEAR))

        # Increment redshift by half a time step
        z_half = self.time2zred(t_half)

        # Scale quantities if cosmological run
        if self.cosmological:
            # Scale density according to expansion
            dilution_factor = (1 + z_half) / (1 + self.zred)
            # dilution_factor = ( (1+z_half) / (1+self.zred) )**3
            self.ndens *= dilution_factor**3

            # Set cell size to current proper size
            # self.dr = self.dr_c * self.cosmology.scale_factor(z_half)
            self.dr /= dilution_factor
            self.printlog(f"zfactor = {1.0 / dilution_factor: .10f}")
        # Set new time and redshift (after timestep)
        self.zred = z_half
        self.time = t_after

    def cosmo_evolve_to_now(self):
        """Evolve cosmology over a timestep"""
        # Time step
        t_now = self.time

        # Increment redshift by half a time step
        z_now = self.time2zred(t_now)

        # Scale quantities if cosmological run
        if self.cosmological:
            # Scale density according to expansion
            dilution_factor = (1 + z_now) / (1 + self.zred)
            # dilution_factor = ( (1+z_half) / (1+self.zred) )**3
            self.ndens *= dilution_factor**3

            # Set cell size to current proper size
            # self.dr = self.dr_c * self.cosmology.scale_factor(z_half)
            self.dr /= dilution_factor
            self.printlog(f"zfactor = {1.0 / dilution_factor: .10f}")
        # Set new time and redshift (after timestep)
        self.zred = z_now

    # =====================================================================================================
    # UTILITY METHODS
    # =====================================================================================================
    def time2zred(self, t):
        """Calculate the redshift corresponding to an age t in seconds"""
        # TODO: it should be then z_at_value(self.cosmology.age, t*u.s).value
        # in C2Ray is defined: time2zred = -1+(1.+zred_t0)*(t0/(t0+time))**(2./3.)
        # return -1+(1.+self.zred_0)*(self.age_0/(self.age_0+t))**(2./3.)
        return -1 + (1.0 + self.zred_0) * (self.age_0 / (t)) ** (2.0 / 3.0)

    def zred2time(self, z, unit="s"):
        """Calculate the age corresponding to a redshift z

        Parameters
        ----------
        z : float
            Redshift at which to get age
        unit : str (optional)
            Unit to get age in astropy naming. Default: seconds
        """
        # TODO : it should be then self.cosmology.age(z).to(unit).value
        # In C2Ray is defined: zred2time = t0*( ((1.0+zred_t0)/(1.0+zred1))**1.5 - 1.0 )
        # return self.age_0*(((1.0+self.zred_0)/(1.0+z))**1.5 - 1.0)
        # C2Ray version, time is 0 at sim begin
        return self.age_0 * (
            ((1.0 + self.zred_0) / (1.0 + z)) ** 1.5
        )  # <- Here, we want time to be actual age (age0 + t)

    # =====================================================================================================
    # INITIALIZATION METHODS (PRIVATE)
    # =====================================================================================================

    def _cosmology_init(self):
        """Set up cosmology from parameters (H0, Omega,..)"""
        h = self._ld["Cosmology"]["h"]
        Om0 = self._ld["Cosmology"]["Omega0"]
        Ob0 = self._ld["Cosmology"]["Omega_B"]
        Tcmb0 = self._ld["Cosmology"]["cmbtemp"]
        H0 = 100 * h
        self.cosmology = FlatLambdaCDM(H0, Om0, Tcmb0, Ob0=Ob0)

        self.cosmological = self._ld["Cosmology"]["cosmological"]
        self.zred_0 = self._ld["Cosmology"]["zred_0"]

        # H0 *= 1e5/Mpc

        # self.age_0 = 2.*(1.+self.zred_0)**(-1.5)/(3.*H0*np.sqrt(Om0))
        # self.age_0 = self.zred2time(self.zred_0)
        self.age_0 = (
            2.0 * (1.0 + self.zred_0) ** (-1.5) / (3.0 * H0 * 1e5 / Mpc * np.sqrt(Om0))
        )

        # TODO: here I force z0 to equal restart slice to test
        zred_0 = self.zred_0  # 20.134

        # Scale quantities to the initial redshift
        if self.cosmological:
            self.printlog(
                "Cosmology is on, scaling comoving quantities to the initial redshift, "
                f"which is z0 = {zred_0:.3f}..."
            )
            self.printlog("Cosmological parameters used:")
            self.printlog(f"h   = {h:.4f}, Tcmb0 = {Tcmb0:.3e}")
            self.printlog(f"Om0 = {Om0:.4f}, Ob0   = {Ob0:.4f}")
            # TODO: it should be:
            # self.dr = self.cosmology.scale_factor(self.zred_0) * self.dr_c
            self.dr = self.dr_c / (1 + zred_0)
        else:
            self.printlog("Cosmology is off.")

    # =====================================================================================================
    # USER DEFINED METHODS
    # =====================================================================================================

    def read_sources(self, file, mass, ts):  # >:( trgeoip
        """Read sources from a C2Ray-formatted file

        Parameters
        ----------
        file : str
            Filename to read
        n : int
            Number of sources to read from the file

        Returns
        -------
        srcpos : array
            Grid positions of the sources formatted in a suitable way for the chosen
            raytracing algorithm
        normflux : array
            Normalization of the flux of each source (relative to S_star)
        numsrc : int
            Number of sources read from the file
        """
        S_star_ref = 1e48

        # TODO: automatic selection of low mass or high mass.
        # For the moment only high mass:
        # mass2phot = (
        #     msun2g
        #     * self.fgamma_hm
        #     * self.cosmology.Ob0
        #     / (self.mean_molecular * c.m_p.cgs.value * self.ts * self.cosmology.Om0)
        # )
        # TODO: for some reason the difference with the orginal Fortran run is of
        # the molecular weight: self.printlog('%f' %self.mean_molecular )
        mass2phot = (
            msun2g
            * self.fgamma_hm
            * self.cosmology.Ob0
            / (m_p * ts * self.cosmology.Om0)
        )

        if file.endswith(".hdf5"):
            f = h5py.File(file, "r")
            srcpos = f["sources_positions"][:].T
            assert srcpos.shape[0] == 3
            normflux = f["sources_mass"][:] * mass2phot / S_star_ref
            f.close()
        else:
            # use original C2Ray source file
            src = t2c.SourceFile(filename=file, mass=mass)
            srcpos = src.sources_list[:, :3].T
            normflux = src.sources_list[:, -1] * mass2phot / S_star_ref

        self.printlog(
            "\n---- Reading source file with total of %d ionizing source:\n%s"
            % (normflux.size, file)
        )
        self.printlog(" Total Flux : %e" % np.sum(normflux * S_star_ref))
        self.printlog(" Source lifetime : %f Myr" % (ts / (1e6 * YEAR)))
        self.printlog(
            " min, max source mass : %.3e  %.3e [Msun] and min, mean, max number "
            "of ionising sources : %.3e  %.3e  %.3e [1/s]"
            % (
                normflux.min() / mass2phot * S_star_ref,
                normflux.max() / mass2phot * S_star_ref,
                normflux.min() * S_star_ref,
                normflux.mean() * S_star_ref,
                normflux.max() * S_star_ref,
            )
        )
        return srcpos, normflux

    def read_density(self, z):
        """Read coarser density field from C2Ray-formatted file

        This method is meant for reading density field run with either
        N-body or hydro-dynamical simulations. The field is then smoothed
        on a coarse mesh grid.

        Parameters
        ----------
        n : int
            Number of sources to read from the file

        Returns
        -------
        srcpos : array
            Grid positions of the sources formatted in a suitable way for
            the chosen raytracing algorithm
        normflux : array
            density mesh-grid in csg units
        """
        if self.cosmological:
            redshift = z
        else:
            redshift = self.zred_0

        # redshift bin for the current redshift based on the density redshift
        # low_z, high_z = find_bins(redshift, self.zred_density)
        high_z = self.zred_density[
            np.argmin(
                np.abs(self.zred_density[self.zred_density >= redshift] - redshift)
            )
        ]

        if high_z != self.prev_zdens:
            file = "%scoarser_densities/%.3fn_all.dat" % (self.inputs_basename, high_z)
            self.printlog("\n---- Reading density file:\n " + file)
            self.ndens = (
                t2c.DensityFile(filename=file).cgs_density
                / (self.mean_molecular * m_p)
                * (1 + redshift) ** 3
            )
            self.printlog(
                " min, mean and max density : %.3e  %.3e  %.3e [1/cm3]"
                % (self.ndens.min(), self.ndens.mean(), self.ndens.max())
            )
            self.prev_zdens = high_z
        else:
            # no need to re-read the same file again
            # TODO: in the future use this values for a 3D interpolation
            # for the density (can be extended to sources too)
            pass

    def write_output(self, z):
        """Write ionization fraction & ionization rates as C2Ray binary files

        Parameters
        ----------
        z : float
            Redshift (used to name the file)
        """
        suffix = f"_{z:.3f}.dat"
        t2c.save_cbin(
            filename=self.results_basename + "xfrac" + suffix,
            data=self.xh,
            bits=64,
            order="F",
        )
        t2c.save_cbin(
            filename=self.results_basename + "IonRates" + suffix,
            data=self.phi_ion,
            bits=32,
            order="F",
        )

        self.printlog("\n--- Reionization History ----")
        self.printlog(
            " min, mean, max xHII : %.5e  %.5e  %.5e"
            % (self.xh.min(), self.xh.mean(), self.xh.max())
        )
        self.printlog(
            " min, mean, max Irate : %.5e  %.5e  %.5e [1/s]"
            % (self.phi_ion.min(), self.phi_ion.mean(), self.phi_ion.max())
        )
        self.printlog(
            " min, mean, max density : %.5e  %.5e  %.5e [1/cm3]"
            % (self.ndens.min(), self.ndens.mean(), self.ndens.max())
        )

    # =============================================================================
    # Below are the overridden initialization routines specific to the CubeP3M case
    # =============================================================================

    def _redshift_init(self):
        """Initialize time and redshift counter"""
        self.zred_density = t2c.get_dens_redshifts(
            self.inputs_basename + "coarser_densities/"
        )[::-1]
        # self.zred_sources = get_source_redshifts(self.inputs_basename + "sources/")[
        #     ::-1
        # ]
        # TODO: waiting for next tools21cm release
        self.zred_sources = get_source_redshifts(self.inputs_basename + "sources/")[
            ::-1
        ]
        if self.resume:
            # get the resuming redshift
            self.zred = np.min(get_redshifts_from_output(self.results_basename))
            # self.age_0 = self.zred2time(self.zred_0)
            _, self.prev_zdens = find_bins(self.zred, self.zred_density)
            _, self.prev_zsourc = find_bins(self.zred, self.zred_sources)
        else:
            self.prev_zdens = -1
            self.prev_zsourc = -1
            self.zred = self.zred_0

        self.time = self.zred2time(self.zred)
        # self.time = self.age_0

    def _material_init(self):
        """Initialize material properties of the grid"""
        if self.resume:
            # get fields at the resuming redshift
            self.ndens = (
                t2c.DensityFile(
                    filename="%scoarser_densities/%.3fn_all.dat"
                    % (self.inputs_basename, self.prev_zdens)
                ).cgs_density
                / (self.mean_molecular * m_p)
                * (1 + self.zred) ** 3
            )
            # self.ndens = self.read_density(z=self.zred)
            self.xh = t2c.read_cbin(
                filename="%sxfrac_%.3f.dat" % (self.results_basename, self.zred),
                bits=64,
                order="F",
            )
            # TODO: implement heating
            temp0 = self._ld["Material"]["temp0"]
            self.temp = temp0 * np.ones(self.shape, order="F")
            self.phi_ion = t2c.read_cbin(
                filename="%sIonRates_%.3f.dat" % (self.results_basename, self.zred),
                bits=32,
                order="F",
            )
        else:
            xh0 = self._ld["Material"]["xh0"]
            temp0 = self._ld["Material"]["temp0"]
            avg_dens = self._ld["Material"]["avg_dens"]

            self.ndens = avg_dens * np.empty(self.shape, order="F")
            self.xh = xh0 * np.ones(self.shape, order="F")
            self.temp = temp0 * np.ones(self.shape, order="F")
            self.phi_ion = np.zeros(self.shape, order="F")

    def _output_init(self):
        """Set up output & log file"""
        self.results_basename = self._ld["Output"]["results_basename"]
        self.inputs_basename = self._ld["Output"]["inputs_basename"]

        self.logfile = self.results_basename + self._ld["Output"]["logfile"]
        title = r"""
                  _________   ____
     ____  __  __/ ____/__ \ / __ \____ ___  __
    / __ \/ / / / /    __/ // /_/ / __ `/ / / /
   / /_/ / /_/ / /___ / __// _, _/ /_/ / /_/ /
  / .___/\__, /\____//____/_/ |_|\__,_/\__, /
 /_/    /____/                        /____/
"""
        if self._ld["Grid"]["resume"]:
            with open(self.logfile, "r") as f:
                log = f.readlines()
            with open(self.logfile, "w") as f:
                log.append("\n\nResuming" + title[8:] + "\n\n")
                f.write("".join(log))
        else:
            with open(self.logfile, "w") as f:
                # Clear file and write header line
                f.write(title + "\nLog file for pyC2Ray.\n\n")

    def _sources_init(self):
        """Initialize settings to read source files"""
        self.fgamma_hm = self._ld["Sources"]["fgamma_hm"]
        self.fgamma_lm = self._ld["Sources"]["fgamma_lm"]
        self.ts = self._ld["Sources"]["ts"] * YEAR * 1e6
        self.printlog(
            f"Using UV model with fgamma_lm = {self.fgamma_lm:.1f} "
            f"and fgamma_hm = {self.fgamma_hm:.1f}"
        )

    def _grid_init(self):
        """Set up grid properties"""
        # Comoving quantities
        self.boxsize_c = self._ld["Grid"]["boxsize"] * Mpc / self._ld["Cosmology"]["h"]
        self.dr_c = self.boxsize_c / self.N

        self.printlog(f"Welcome! Mesh size is N = {self.N:n}.")
        self.printlog(f"Simulation Box size (comoving Mpc): {self.boxsize_c / Mpc:.3e}")

        # Initialize cell size to comoving size (if cosmological run,
        # it will be scaled in cosmology_init)
        self.dr = self.dr_c

        # Set R_max (LLS 3) in cell units
        self.R_max_LLS = (
            self._ld["Photo"]["R_max_cMpc"]
            * self.N
            * self._ld["Cosmology"]["h"]
            / self._ld["Grid"]["boxsize"]
        )
        self.printlog(
            "Maximum comoving distance for photons from source (type 3 LLS): "
            f"{self._ld['Photo']['R_max_cMpc']: .3e} comoving Mpc"
        )
        self.printlog(f"This corresponds to {self.R_max_LLS: .3f} grid cells.")

        self.resume = self._ld["Grid"]["resume"]
