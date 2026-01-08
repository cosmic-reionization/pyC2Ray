import atexit
import os
import pickle as pkl
import re
import subprocess

import numpy as np
import tools21cm as t2c
import yaml
from astropy import constants as cst
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM, z_at_value
from mpi4py import MPI

try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader

from .asora_core import device_close, device_init, photo_table_to_device
from .evolve import evolve3D
from .radiation import BlackBodySource, YggdrasilModel, make_tau_table
from .raytracing import do_raytracing
from .sinks_model import SinksPhysics
from .utils.logutils import printlog

# ======================================================================
# This file defines the abstract C2Ray object class, which is the basis
# for a c2ray simulation. It deals with parameters, I/O, cosmology,
# and other things such as memory allocation when using the GPU.
# Any concrete simulation uses subclasses of C2Ray, with methods specific
# to certain input files (e.g. CubeP3M)
#
# Since all simulation classes inherit from this class, great care should
# be taken in editing it!
#
# -- Notes on cosmology: --
# * In C2Ray, the scale factor is 1 at z = 0. The box size is given
# in comoving units, i.e. it is the proper size at z = 0. At the
# start (in cosmo_ini), the cell size & volume are scaled down to
# the first redshift slice of the program.
#
# * There are 2 types of redshift evolution: (1) when the program
# reaches a new "slice" (where a density file would be read etc)
# and (2) at each timestep BETWEEN slices. Basically, at (1), the
# density is set, and during (2), this density is diluted due to
# the expansion.
#
# * During this dilution (at each timestep between slices), C2Ray
# has the convention that the redshift is incremented not by the
# value that corresponds to a full timestep in cosmic time, but by
# HALF a timestep.
#    ||          |           |           |           |               ||
#    ||    z1    |     z2    |     z3    |     z4    |       ...     ||
#    ||          |           |           |           |               ||
#    t0          t1          t2          t3          t4
#
#   ("||" = slice,    "|" = timestep,   "1,2,3,4,.." indexes the timestep)
#
# In terms of attributes, C2Ray.time always contains the time at the
# end of the current timestep, while C2Ray.zred contains the redshift
# at half the current timestep. This is relevant to understand the
# cosmo_evolve routine below (which is based on what is done in the
# original C2Ray)
#
# This induces a potential bug: when a slice is reached and the
# density is set, the density corresponds to zslice while
# C2Ray.zred is at the redshift "half a timestep before".
# The best solution I've found here is to just save the comoving cell
# size dr_c and always set the current cell size to dr = a(z)*dr_c,
# rather than "diluting" dr iteratively like the density.
#
# TODO ideas:
# * Add "default" values for YAML parameter file so that if the user
# omits a value in the file, a default value is used instead rather
# than throwing an error
# ======================================================================

# Conversion Factors.
# When doing direct comparisons with C2Ray, the difference between astropy.constants
# and the C2Ray values may be visible, thus we use the same exact value
# for the constants. This can be changed to the
# astropy values once consistency between the two codes has been established
pc = (1 * u.pc).cgs.value  # C2Ray value: 3.086e18
YEAR = (1 * u.yr).cgs.value  # C2Ray value: 3.15576E+07
ev2fr = 1.0 / (cst.h * u.Hz).to("eV").value  # eV to Frequency (Hz)
ev2k = 1.0 / (cst.k_B * u.K).to("eV").value  # eV to Kelvin
kpc = (1 * u.kpc).cgs.value  # kiloparsec in cm
Mpc = (1 * u.Mpc).cgs.value  # megaparsec in cm
msun2g = (1 * u.Msun).cgs.value  # solar mass to grams
m_p = cst.m_p.cgs.value  # proton mass to grams


class C2Ray:
    def __init__(self, paramfile):
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
        # Read YAML parameter file and set main properties
        self._read_paramfile(paramfile)
        self._param_init()

        # MPI setup
        if self.mpi:
            self.mpi = MPI
            self.comm = self.mpi.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.nprocs = self.comm.Get_size()
        else:
            self.mpi = False
            self.rank = 0
            self.nprocs = 1

        self.shape = (self.N, self.N, self.N)

        # Set Raytracing mode
        if self.gpu:
            # node_name = os.getenv('SLURMD_NODENAME', default='Unknown Node.')
            # task_id = int(os.getenv('SLURM_PROCID', default='Unknown Task ID.'))
            # local_task_id = int(os.getenv('SLURM_LOCALID', default='Unknown Local Task ID.'))
            # gpu_ids = os.getenv("CUDA_VISIBLE_DEVICES", default="No GPU Assigned.")
            # tot_gpus = os.getenv('SLURM_JOB_GPUS', default="No GPU on node.")
            # Number of GPUs
            try:
                nr_gpus = int(
                    os.getenv("SLURM_GPUS_ON_NODE", default="No GPU on node.")
                )
            except Exception:
                nr_gpus = int(
                    subprocess.check_output("nvidia-smi  -L | wc -l", shell=True)
                )

            # Allocate GPU memory
            src_batch_size = self._ld["Raytracing"]["source_batch_size"]
            device_init(self.N, src_batch_size, self.rank, nr_gpus)

            # Register deallocation function (automatically calls this on program termination)
            atexit.register(self._gpu_close)
        else:
            # is going to run the raytracing algorithm on CPU
            pass

        # Initialize Simulation
        self._output_init()
        self._grid_init()
        self._cosmology_init()
        self._redshift_init()
        self._material_init()
        self._sources_init()
        self._radiation_init()
        self._sinks_init()
        if self.gpu:
            # self.printlog('\tNode name: %s\n\ttask_id: %d\n\tlocal task id: %d\n\tgpu_ids: %s\n\ttot gpu job: %s\n\ttot gpu on node: %d' %(node_name, task_id, local_task_id, gpu_ids, tot_gpus, nr_gpus))
            # Print maximum shell size for info, based on LLS (qmax is s.t. Rmax fits inside of it)
            q_max = np.ceil(np.sqrt(3) * min(self.R_max_LLS, np.sqrt(3) * self.N / 2))
            self.printlog("Using ASORA Raytracing ( q_max = %d )" % q_max)
        else:
            # Print info about subbox algorithm
            self.printlog(
                "Using CPU Raytracing (subboxsize = %d, max_subbox = %d)"
                % (self.subboxsize, self.max_subbox)
            )
        if self.mpi:
            self.printlog("Using %d MPI Ranks" % self.nprocs)
        else:
            self.printlog("Running in non-MPI (single-GPU/CPU) mode")
        self.printlog("Starting simulation... \n\n")

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
            # self.printlog(f"zfactor = {1.0 / dilution_factor: .10f}")
        # Set new time and redshift (after timestep)
        self.zred = z_now

    def evolve3D(self, dt, src_flux, src_pos):
        """Evolve the grid over one timestep

        Raytrace all sources, compute cumulative photoionization rate of each cell and
        do chemistry. This is done until convergence in the ionized fraction is reached.

        Parameters
        ----------
        dt : float
            Timestep in seconds (typically generated using set_timestep method)
        src_flux : 1D-array of shape (numsrc)
            Array containing the total ionizing flux of each source, normalized by S_star (1e48 by default)
        src_pos : 2D-array of shape (3,numsrc)
            Array containing the 3D grid position of each source, in Fortran indexing (from 1)
        """
        if src_pos.shape[0] != 3 and src_pos.shape[1] == 3:
            src_pos = src_pos.T
        elif src_pos.shape[0] == 3:
            pass
        else:
            ValueError(
                "ASORA requires the shape of the src_pos array to be (3, N_src). Here, it does not appear that you are providing an array with this shape."
            )

        NumSrc = src_flux.shape[0]
        # If the number of sources exceed the number of MPI processors
        # then call the evolve designed for the MPI source splitting.
        # Otherwise all ranks are calling (independently) the evolve
        # with no source splitting until the condition above is meet.
        use_mpi = NumSrc >= self.nprocs and self.mpi
        self.xh, self.phi_ion = evolve3D(
            dt=dt,
            dr=self.dr,
            src_flux=src_flux,
            src_pos=src_pos,
            use_gpu=self.gpu,
            max_subbox=self.max_subbox,
            subboxsize=self.subboxsize,
            loss_fraction=self.loss_fraction,
            use_mpi=use_mpi,
            comm=self.comm if use_mpi else None,
            rank=self.rank if use_mpi else 0,
            nprocs=self.nprocs if use_mpi else 1,
            temp=self.temp,
            ndens=self.ndens,
            xh=self.xh,
            clump=self.clumping_factor,
            photo_thin_table=self.photo_thin_table,
            photo_thick_table=self.photo_thick_table,
            minlogtau=self.minlogtau,
            dlogtau=self.dlogtau,
            R_max_LLS=self.R_max_LLS,
            convergence_fraction=self.convergence_fraction,
            sig=self.sig,
            bh00=self.bh00,
            albpow=self.albpow,
            colh0=self.colh0,
            temph0=self.temph0,
            abu_c=self.abu_c,
            logfile=self.logfile,
            quiet=False,
        )

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
        # self.printlog(' This is time : %f\t %f' %(t_now/YEAR, t_after/YEAR))

        # Increment redshift by half a time step
        z_half = self.time2zred(t_half)

        # Scale quantities if cosmological run
        if self.cosmological:
            # Scale density according to expansion
            dilution_factor = ((1 + z_half) / (1 + self.zred)) ** 3
            self.ndens *= dilution_factor

            # Set cell size to current proper size
            self.dr = self.dr_c * self.cosmology.scale_factor(z_half)

        # Set new clumping factor if is not redshift constant
        if self.sinks.clumping_model != "constant":
            if self.sinks.clumping_model == "redshift":
                self.clumping_factor = self.sinks.calculate_clumping(z=self.zred)
            else:
                self.clumping_factor = self.sinks.calculate_clumping(
                    z=self.zred, ndens=self.ndens
                )

            self.printlog(
                " min, mean and max clumping factor at z = %.3f: %.2f  %.2f  %.2f"
                % (
                    self.zred,
                    self.clumping_factor.min(),
                    self.clumping_factor.mean(),
                    self.clumping_factor.max(),
                )
            )

        # Set new time and redshift (after timestep)
        self.zred = z_half
        self.time = t_after

        # Set new mean-free-path if it is redshift dependent
        if self.sinks.mfp_model == "Worseck2014":
            self.R_max_LLS = self.sinks.mfp_Worseck2014(z=self.zred)  # in cMpc
            self.R_max_LLS *= self.N / self.boxsize  # in number of grids
            self.printlog(
                "Mean-free-path for photons at z = %.3f (Worseck+ 2014): %.3e cMpc"
                % (self.zred, self.R_max_LLS * self.boxsize / self.N)
            )
            self.printlog("This corresponds to %.3f grid cells." % self.R_max_LLS)

    def printlog(self, s, quiet=False):
        """Print to log file and standard output

        Parameters
        ----------
        s : str
            String to print
        quiet : bool
            Whether to print only to log file or also to standard output (default)
        """
        if self.rank == 0:
            # the rank = 0 assures that we do not log many time the same text when
            if self.logfile is None:
                raise RuntimeError("Please set the log file in output_ini")
            else:
                printlog(s, self.logfile, quiet)

    def write_output(self, z, ext=".dat"):
        """Write ionization fraction & ionization rates as C2Ray binary files

        Parameters
        ----------
        z : float
            Redshift (used to name the file)
        ext : string
            extension of the output file. If '.dat' save a binary file (with tools21cm), otherwise '.npy'.
        """
        if self.rank == 0:
            suffix = f"_z{z:.3f}" + ext
            if suffix.endswith(".dat"):
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
                # t2c.save_cbin(filename=self.results_basename + "coldens" + suffix, data=self.coldens, bits=64, order='F')
            elif suffix.endswith(".npy"):
                np.save(file=self.results_basename + "xfrac" + suffix, arr=self.xh)
                np.save(
                    file=self.results_basename + "IonRates" + suffix, arr=self.phi_ion
                )
                # TODO: replace this by the differential brightness?
                # np.save(file=self.results_basename + "coldens" + suffix, arr=self.coldens)
            elif suffix.endswith(".pkl"):
                with open(self.results_basename + "xfrac" + suffix, "wb") as f:
                    pkl.dump(self.xh, f)
                with open(self.results_basename + "IonRates" + suffix, "wb") as f:
                    pkl.dump(self.phi_ion, f)

            # print min, max and average quantities
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

            # write summary output file
            summary_exist = os.path.exists(self.results_basename + "PhotonCounts2.txt")

            with open(self.results_basename + "PhotonCounts2.txt", "a") as f:
                if not (summary_exist):
                    header = "# z\ttot HI atoms\ttot phots\t mean ndens [1/cm3]\t mean Irate [1/s]\tR_mfp [cMpc]\tmean ionization fraction (by volume and mass)\n"
                    f.write(header)

                # mass-average neutral faction
                massavrg_ion_frac = np.sum(self.xh * self.ndens) / np.sum(self.ndens)

                # calculate total number of neutral hydrogen atoms
                tot_nHI = np.sum(self.ndens * (1 - self.xh) * self.dr**3)

                text = "%.3f\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n" % (
                    z,
                    tot_nHI,
                    self.tot_phots,
                    np.mean(self.ndens),
                    np.mean(self.phi_ion),
                    self.R_max_LLS / self.N * self.boxsize,
                    np.mean(self.xh),
                    massavrg_ion_frac,
                )
                f.write(text)
        else:
            # this is for the other ranks
            pass

    # =====================================================================================================
    # UTILITY METHODS
    # =====================================================================================================
    def time2zred(self, t):
        """Calculate the redshift corresponding to an age t in seconds"""
        try:
            return z_at_value(self.cosmology.age, t * u.s).value
        except Exception:
            return z_at_value(self.cosmology.age, t * u.s)

    def zred2time(self, z, unit="s"):
        """Calculate the age corresponding to a redshift z

        Parameters
        ----------
        z : float
            Redshift at which to get age
        unit : str (optional)
            Unit to get age in astropy naming. Default: seconds
        """
        return self.cosmology.age(z).to(unit).value

    def do_raytracing(self, src_flux, src_pos):
        """Standalone raytracing method

        Function to only calculate Gamma (photoionization rates) based on current
        ionized fractions, without touching the chemistry. Useful for debugging,
        benchmarking and specialized use purposes.

        Parameters
        ----------
        src_flux : 1D-array of shape (numsrc)
            Array containing the total ionizing flux of each source, normalized by S_star (1e48 by default)
        src_pos : 2D-array of shape (3,numsrc)
            Array containing the 3D grid position of each source, in Fortran indexing (from 1)
        """
        gamma = do_raytracing(
            self.dr,
            src_flux,
            src_pos,
            self.gpu,
            self.max_subbox,
            self.subboxsize,
            self.loss_fraction,
            self.ndens,
            self.xh,
            self.photo_thin_table,
            self.photo_thick_table,
            self.minlogtau,
            self.dlogtau,
            self.R_max_LLS,
            self.sig,
            self.logfile,
        )
        self.phi_ion = gamma
        return gamma

    # =====================================================================================================
    # INITIALIZATION METHODS (PRIVATE)
    # =====================================================================================================

    def _param_init(self):
        """Set up general constants and parameters

        Computes additional required quantities from the read-in parameters
        and stores them as attributes
        """
        self.N = self._ld["Grid"]["meshsize"]
        self.gpu = self._ld["Grid"]["gpu"]
        self.mpi = self._ld["Grid"]["mpi"]
        self.eth0 = self._ld["CGS"]["eth0"]
        self.ethe0 = self._ld["CGS"]["ethe0"]
        self.ethe1 = self._ld["CGS"]["ethe1"]
        self.bh00 = self._ld["CGS"]["bh00"]
        self.fh0 = self._ld["CGS"]["fh0"]
        self.xih0 = self._ld["CGS"]["xih0"]
        self.albpow = self._ld["CGS"]["albpow"]
        self.abu_h = self._ld["Abundances"]["abu_h"]
        self.abu_he = self._ld["Abundances"]["abu_he"]
        self.mean_molecular = self.abu_h + 4.0 * self.abu_he
        self.abu_c = self._ld["Abundances"]["abu_c"]
        self.colh0 = self._ld["CGS"]["colh0_fact"] * self.fh0 * self.xih0 / self.eth0**2
        self.temph0 = self.eth0 * ev2k
        self.sig = self._ld["Photo"]["sigma_HI_at_ion_freq"]
        self.loss_fraction = self._ld["Raytracing"]["loss_fraction"]
        self.convergence_fraction = self._ld["Raytracing"]["convergence_fraction"]
        self.max_subbox = self._ld["Raytracing"]["max_subbox"]
        self.subboxsize = self._ld["Raytracing"]["subboxsize"]

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
        self.age_0 = self.zred2time(self.zred_0)

        # Scale quantities to the initial redshift
        if self.cosmological:
            self.printlog(
                f"Cosmology is on, scaling comoving quantities to the initial redshift, which is z0 = {self.zred_0:.3f}..."
            )
            self.printlog("Cosmological parameters used:")
            self.printlog(f"h   = {h:.4f}, Tcmb0 = {Tcmb0:.3e}")
            self.printlog(f"Om0 = {Om0:.4f}, Ob0   = {Ob0:.4f}")
            self.dr = self.cosmology.scale_factor(self.zred_0) * self.dr_c
        else:
            self.printlog("Cosmology is off.")

    def _radiation_init(self):
        """Set up radiation tables for ionization/heating rates"""
        # Create optical depth table (log-spaced)
        self.minlogtau = self._ld["Photo"]["minlogtau"]
        self.maxlogtau = self._ld["Photo"]["maxlogtau"]
        self.NumTau = self._ld["Photo"]["NumTau"]
        self.SourceType = self._ld["Photo"]["SourceType"]
        self.grey = self._ld["Photo"]["grey"]
        self.compute_heating_rates = self._ld["Photo"]["compute_heating_rates"]

        if self.grey:
            self.printlog("Warning: Using grey opacity")
        else:
            self.printlog(
                f"Using power-law opacity with {self.NumTau:n} table points between tau=10^({self.minlogtau:n}) and tau=10^({self.maxlogtau:n})"
            )

        # The actual table has NumTau + 1 points: the 0-th position is tau=0 and the remaining NumTau points are log-spaced from minlogtau to maxlogtau (same as in C2Ray)
        self.tau, self.dlogtau = make_tau_table(
            self.minlogtau, self.maxlogtau, self.NumTau
        )

        ion_freq_HI = ev2fr * self.eth0
        ion_freq_HeII = ev2fr * self.ethe1

        # Black-Body source type
        if self.SourceType == "blackbody":
            freq_min = ion_freq_HI
            freq_max = 10 * ion_freq_HeII

            # Initialize spectrum parameters
            self.bb_Teff = self._ld["BlackBodySource"]["Teff"]
            self.cs_pl_idx_h = self._ld["BlackBodySource"]["cross_section_pl_index"]
            radsource = BlackBodySource(
                self.bb_Teff, self.grey, ion_freq_HI, self.cs_pl_idx_h
            )

            # Print info
            self.printlog(
                f"Using Black-Body sources with effective temperature T = {radsource.temp:.1e} K and Radius {(radsource.R_star / cst.R_sun.to('cm')).value: .3e} rsun"
            )
            self.printlog(
                f"Spectrum Frequency Range: {freq_min:.3e} to {freq_max:.3e} Hz"
            )
            self.printlog(
                f"This is Energy:           {freq_min / ev2fr:.3e} to {freq_max / ev2fr:.3e} eV"
            )
        elif self.SourceType == "powerlaw":
            # TODO: power law spectra is already implemented in radiation folder
            pass
        elif self.SourceType == "Zackrisson2011":
            freq_min = ion_freq_HI
            freq_max = 10 * ion_freq_HI  # maximum frequency in Zackrisson tables

            self.cs_pl_idx_h = self._ld["BlackBodySource"]["cross_section_pl_index"]
            fname = self._ld["Photo"]["sed_table"]
            radsource = YggdrasilModel(
                tabname=fname,
                grey=self.grey,
                freq0=ion_freq_HI,
                pl_index=self.cs_pl_idx_h,
                S_star_ref=1e48,
            )

            # Print info
            self.printlog(
                "Using Yggdrasil Models for SED, Zackrisson et al (2011), "
                "for PopIII or PopII sources"
            )
            self.printlog(
                f"Spectrum Frequency Range: {freq_min:.3e} to {freq_max:.3e} Hz"
            )
            self.printlog(
                f"This is Energy:           {freq_min / ev2fr:.3e} "
                f"to {freq_max / ev2fr:.3e} eV"
            )
        else:
            raise NameError("Unknown source type : ", self.SourceType)

        # Integrate table
        self.printlog("Integrating photoionization rates tables...")
        self.photo_thin_table, self.photo_thick_table = radsource.make_photo_table(
            self.tau, freq_min, freq_max, 1e48
        )

        # WIP: Heating rates
        # 30.11.23 P.Hirling: The heating tables can be calculated, and used with the standalone CPU raytracing method to calculate photo-heating rates for the whole grid. However, at this time, the chemistry solver doesn't use these rates.
        # TODO:
        # 1. Add heating rate computation to ASORA (GPU raytracing)
        # 2. Add heating (thermal) to chemistry module
        if self.compute_heating_rates:
            self.printlog("Integrating photoheating rates tables...")
            self.heat_thin_table, self.heat_thick_table = radsource.make_heat_table(
                self.tau, freq_min, freq_max, 1e48
            )  # nb integration bounds are given in log10(freq/freq_HI)
        else:
            self.printlog("INFO: No heating rates")
            self.heat_thin_table = np.zeros(self.NumTau + 1)
            self.heat_thick_table = np.zeros(self.NumTau + 1)

        # Copy radiation table to GPU
        if self.gpu:
            photo_table_to_device(self.photo_thin_table, self.photo_thick_table)
            self.printlog("Successfully copied radiation tables to GPU memory.")

    def _grid_init(self):
        """Set up grid properties"""
        # Comoving quantities
        self.boxsize = self._ld["Grid"]["boxsize"]
        self.boxsize_c = self.boxsize * Mpc
        self.dr_c = self.boxsize_c / self.N

        self.printlog(f"Welcome! Mesh size is N = {self.N:n}.")
        self.printlog(f"Simulation Box size (comoving Mpc): {self.boxsize:.3e}")

        # Initialize cell size to comoving size (if cosmological run, it will be scaled in cosmology_init)
        self.dr = self.dr_c

        # flag to set the resume
        # TODO: need to give the index of start for the redshift loop in the main
        self.resume = self._ld["Grid"]["resume"]

    def _output_init(self):
        """Set up output & log file"""
        self.results_basename = self._ld["Output"]["results_basename"]
        if not os.path.exists(self.results_basename) and self.rank == 0:
            os.mkdir(self.results_basename)
        self.inputs_basename = self._ld["Output"]["inputs_basename"]
        self.sources_basename = self._ld["Output"]["sources_basename"]
        self.density_basename = self._ld["Output"]["density_basename"]

        self.logfile = self.results_basename + self._ld["Output"]["logfile"]
        title = r"""
                 _________   ____            
    ____  __  __/ ____/__ \ / __ \____ ___  __
   / __ \/ / / / /    __/ // /_/ / __ `/ / / /
  / /_/ / /_/ / /___ / __// _, _/ /_/ / /_/ /
 / .___/\__, /\____//____/_/ |_|\__,_/\__, /
/_/    /____/                        /____/
"""

        if self.rank == 0:
            if self._ld["Grid"]["resume"]:
                title = "\n\nResuming" + title[8:] + "\n\n"
                print(title)
                with open(self.logfile, "r") as f:
                    log = f.readlines()
                with open(self.logfile, "w") as f:
                    log.append(title)
                    f.write("".join(log))
            else:
                print(title)
                with open(self.logfile, "w") as f:
                    # Clear file and write header line
                    f.write(title + "\nLog file for pyC2Ray.\n\n")

        # all processor wait for rank=0 to be done. This is to avoid that some ranks go ahead.
        self.comm.Barrier()

    def _sinks_init(self):
        """Initialize sinks physics class for the mean-free path and clumping factor"""

        # init sink physics class for MFP and clumping
        self.sinks = SinksPhysics(params=self._ld, N=self.N)

        # for clumping factor
        if self.sinks.clumping_model == "constant":
            self.clumping_factor = self.sinks.calculate_clumping
        elif self.sinks.clumping_model == "redshift":
            self.clumping_factor = self.sinks.calculate_clumping(
                z=self._ld["Cosmology"]["zred_0"]
            )
        else:
            self.clumping_factor = self.sinks.calculate_clumping(
                z=self._ld["Cosmology"]["zred_0"], ndens=self.ndens
            )

        self.printlog(
            "\n---- Calculated Clumping Factor (%s model):" % self.sinks.clumping_model
        )
        self.printlog(
            " min, mean and max clumping : %.3e  %.3e  %.3e"
            % (
                self.clumping_factor.min(),
                self.clumping_factor.mean(),
                self.clumping_factor.max(),
            )
        )

        # for mean-free-path
        if self.sinks.mfp_model == "constant":
            # Set R_max (LLS 3) in cell units
            self.R_max_LLS = self.sinks.R_mfp_cell_unit
            self.printlog(
                "\n---- Calculated Mean-Free Path (%s model):" % self.sinks.mfp_model
            )
            self.printlog(
                "Maximum comoving distance for photons from source mfp = %.2f cMpc (%s model).\n This corresponds to %.3f grid cells.\n"
                % (
                    self.R_max_LLS * self.boxsize / self.N,
                    self.sinks.mfp_model,
                    self.R_max_LLS,
                )
            )
        elif self.sinks.mfp_model == "Worseck2014":
            # set mean-free-path to the initial redshift
            self.R_max_LLS = self.sinks.mfp_Worseck2014(
                z=self._ld["Cosmology"]["zred_0"]
            )  # in cMpc
            self.R_max_LLS *= self.N / self.boxsize
            self.printlog(
                "\n---- Calculated Mean-Free Path (%s model):" % self.sinks.mfp_model
            )
            self.printlog(
                "Maximum comoving distance for photons from source mfp = %.2f cMpc (%s model) : A = %.2f Mpc, eta = %.2f.\n This corresponds to %.3f grid cells.\n"
                % (
                    self.R_max_LLS * self.boxsize / self.N,
                    self.sinks.mfp_model,
                    self.sinks.A_mfp,
                    self.sinks.etha_mfp,
                    self.R_max_LLS,
                )
            )

    # The following initialization methods are simulation kind-dependent and need to be overridden in the subclasses
    def _redshift_init(self):
        """Initialize time and redshift counter"""
        self.time = self.age_0
        self.zred = self.zred_0
        pass

    def _material_init(self):
        """Initialize material properties of the grid"""
        xh0 = self._ld["Material"]["xh0"]
        temp0 = self._ld["Material"]["temp0"]

        self.ndens = np.empty(self.shape, order="F")
        self.xh = xh0 * np.ones(self.shape, order="F")
        self.temp = temp0 * np.ones(self.shape, order="F")
        self.phi_ion = np.zeros(self.shape, order="F")
        self.avg_dens = self._ld["Material"]["avg_dens"]
        # TODO: add option of resuming simulation

    def _sources_init(self):
        """Initialize settings to read source files"""
        pass

    # =====================================================================================================
    # OTHER PRIVATE METHODS
    # =====================================================================================================

    def _read_paramfile(self, paramfile):
        """Read in YAML parameter file"""
        loader = SafeLoader
        # Configure to read scientific notation as floats rather than strings
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
        with open(paramfile, "r") as f:
            self._ld = yaml.load(f, loader)

    def _gpu_close(self):
        """Deallocate GPU memory"""
        device_close()
