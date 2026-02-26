import h5py
import numpy as np
import tools21cm as t2c
from scipy.io import FortranFile
from astropy import constants as c
from astropy import units as u
import os 
import struct

from .c2ray_base import C2Ray, msun2g
from .utils.other_utils import find_bins, get_redshifts_from_output

__all__ = ["C2Ray_CubeP3M_LW"]

# ======================================================================
# This file contains the C2Ray_CubeP3M subclass of C2Ray, which is a
# version used for simulations that read in N-Body data from CubeP3M
# ======================================================================


class C2Ray_CubeP3M_LW(C2Ray):
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
        super().__init__(paramfile)
        # Subgrid Data
        self.LGnMH_Mpc3 = None
        self.zred_array_interp = None
        self.LGdelta1min = 0
        self.LGdelta1max = 0
        self.dLGdelta1 = 0
        self.M_PIIIstar_msun = 300.0
        self.MHflag = 2  # As per Fortran parameter

        self.M_PIIIstar_msun = 300.0
        self.S_star_nominal = 1e48 # Adjust to match your Fortran c2ray_parameters
        # Internal state for suppression
        self.densNDcrit = 1.0
        self.densNDcrit_prev = 1.0

        self.StillNeutral = 0.1  # Threshold for considering a cell neutral
        self.M_grid = None  # Will be calculated in _grid_init
        self.phot_per_atom = np.array([10/6, 150/6, 833])  # Photons per atom for different populations
        self.fstar = np.array([0.008, 0.015, 0.015])  # Star formation efficiency
        self.n_box = 8000

        # --- LW Green Function State ---
        self.greenK = None
        self.HcOm = 0.0
        self.rLW_zobs = 0.0

        # LW emissivities (erg s^-1 Hz^-1 Msun^-1)
        self.emis00 = 1.67e21   
        self.emis01 = 3e21      
        self.emissub = 3e21     
        
        # Real ionizing photon rates
        self.QH_M_real00 = 6.309573445e46 
        self.QH_M_real01 = 1.2e48
        self.QH_M_real_sub = 1.2e48
        
        self.Ni  = np.array([ 6000/6, 50000/6, 55000])
        self.fstar = np.array([0.008, 0.015, 0.015])
        self.M_solar = 1.98892e33 
        # Read the fit data immediately
        if self.MHflag == 2:
            self.read_LGnMH_Mpc3()

        self.printlog('Running: "C2Ray for %d Mpc/h volume"' % self.boxsize)

        super().__init__(paramfile)

    def read_sources(self, file, source_lifetime, mass='hm'): # >:( trgeoip
        """Read sources from a Ramses-formatted file

        The way sources are dealt with is still open and will change significantly
        in the final version. For now, this method is provided:

        It reads source positions and strengths (total ionizing flux in
        photons/second) from a file that is formatted for the original C2Ray,
        and computes the source strength as normalization factors relative
        to a reference strength (1e48 by default). These normalization factors
        are then used during raytracing to compute the photoionization rate.
        (same procedure as in C2Ray)

        Moreover, the method formats the source positions correctly depending
        on whether OCTA is used or not. This is because, while the default CPU
        raytracing takes a 3D-array of any type as argument, OCTA assumes that the
        source position array is flattened and has a C single int type (int32),
        and that the normalization (strength) array has C double float type (float64).

        Parameters
        ----------
        file : str
            Filename to read
        n : int
            Number of sources to read from the file
        
        Returns
        -------
        srcpos : array
            Grid positions of the sources formatted in a suitable way for the chosen raytracing algorithm
        normflux : array
            Normalization of the flux of each source (relative to S_star)
        numsrc : int
            Number of sources read from the file
        """
        S_star_ref = 1e48
        m_p_cgs = c.m_p.cgs.value

        self.M_box = self.rho_crit_0 * self.cosmology.Om0 * (self.boxsize * self.Mpc / self.h)**3
        self.M_grid = self.M_box / (self.n_box**3)
        grid2msun = self.M_grid / msun2g

        num_total_sources = 0
        self.NumHMACHSrc = 0
        self.NumSupprbleSrc = 0
        self.NumSupprsdSrc = 0

        if file.endswith('.hdf5'):
            f = h5py.File(file, 'r')
            srcpos = f['sources_positions'][:].T
            assert srcpos.shape[0] == 3
            normflux = f['sources_mass'][:] * mass2phot / S_star_ref
            f.close()
            num_total_sources = normflux.size
            self.NumSupprsdSrc = num_total_sources

        else:
            src = np.loadtxt(file, skiprows=1)

            if len(src.shape) == 1:
                src = src.reshape(1, -1)

            src[:, :3] -= 1  # Fortran to Python indexing
            num_total_sources = src.shape[0]

            # Raw grid-unit masses (matching Fortran's SrcMass00, SrcMass01)
            srcMass00_grid = src[:, 3]  # HMACH mass in grid units
            srcMass01_grid = src[:, 4]  # LMACH mass in grid units

            # Solar mass versions (for LW calculations downstream)
            self.sM00_msun = srcMass00_grid * grid2msun
            self.sM01_msun = srcMass01_grid * grid2msun

            # Count source types
            self.NumHMACHSrc   = np.sum(srcMass00_grid > 0)
            self.NumSupprbleSrc = np.sum(srcMass01_grid > 0)

            if self.source_model == 4:
                # Iliev et al. model in c2ray
                # -------------------------------------------------------------------
                # Matches Fortran exactly:
                #   SrcMass(:,0) = SrcMass(:,1)*phot_per_atom(1) 
                #                + SrcMass(:,2)*phot_per_atom(2)
                #   NormFlux = SrcMass(ns,0) * M_grid * Omega_B 
                #              / (Omega0 * m_p) / S_star_nominal / lifetime2
                #
                # Suppression: LMACHs in ionized cells (xh > StillNeutral) 
                #              have SrcMass01 set to 0.
                #              HMACHs are never suppressed.
                # -------------------------------------------------------------------
                srcpos = src[:, :3].T
                # Apply suppression to LMACH masses
                srcMass01_suppressed = srcMass01_grid.copy()
                for i in range(num_total_sources):
                    if srcMass01_grid[i] > 0:
                        ix = int(src[i, 0])
                        iy = int(src[i, 1])
                        iz = int(src[i, 2])
                        if self.xh[ix, iy, iz] > self.StillNeutral:
                            srcMass01_suppressed[i] = 0.0
                            self.NumSupprsdSrc += 1
                            self.sM01_msun[i] = 0.0  # Keep consistent for LW use

                # Combined mass weighted by photons per atom (still in grid units)
                srcMass_combined = (srcMass00_grid * self.phot_per_atom[0] +
                                    srcMass01_suppressed * self.phot_per_atom[1])
                # NormFlux matching Fortran: M_grid * Ob0 / (Om0 * m_p) / S_star / lifetime 
                normflux = (srcMass_combined * self.M_grid * self.cosmology.Ob0 / (self.cosmology.Om0 * m_p_cgs) / S_star_ref / source_lifetime)
                # normflux = (srcMass_combined * self.M_grid * self.cosmology.Ob0 / (self.cosmology.Om0 * m_p_cgs) / S_star_ref / 60537950633080.320)
                print(normflux* S_star_ref)
                print(srcMass_combined, self.M_grid, self.cosmology.Ob0, m_p_cgs, S_star_ref, source_lifetime)

            elif self.source_model is None or self.source_model == 0:
                srcpos = src[:, :3].T
                mass2phot_hm = msun2g * self.fgamma_hm * self.cosmology.Ob0 / (self.mean_molecular * m_p_cgs * self.ts * self.cosmology.Om0)
                mass2phot_lm = msun2g * self.fgamma_lm * self.cosmology.Ob0 / (self.mean_molecular * m_p_cgs * self.ts * self.cosmology.Om0)
                normflux = (self.sM00_msun * mass2phot_hm / S_star_ref) + (self.sM01_msun * mass2phot_lm / S_star_ref)
                self.NumSupprsdSrc = 0

            elif self.source_model == 1:  # Full suppression
                srcpos = src[:, :3].T
                mass2phot_hm = msun2g * self.fgamma_hm * self.cosmology.Ob0 / (self.mean_molecular * m_p_cgs * self.ts * self.cosmology.Om0)
                mass2phot_lm = msun2g * self.fgamma_lm * self.cosmology.Ob0 / (self.mean_molecular * m_p_cgs * self.ts * self.cosmology.Om0)
                for i in range(len(src)):
                    if self.sM01_msun[i] > 0 and self.xh[int(src[i][0]), int(src[i][1]), int(src[i][2])] > self.StillNeutral:
                        self.sM01_msun[i] = 0.0
                        self.NumSupprsdSrc += 1
                normflux = (self.sM00_msun * mass2phot_hm / S_star_ref) + (self.sM01_msun * mass2phot_lm / S_star_ref)

            elif self.source_model == 2:  # Partial suppression
                srcpos = src[:, :3].T
                mass2phot_hm = msun2g * self.fgamma_hm * self.cosmology.Ob0 / (self.mean_molecular * m_p_cgs * self.ts * self.cosmology.Om0)
                mass2phot_lm = msun2g * self.fgamma_lm * self.cosmology.Ob0 / (self.mean_molecular * m_p_cgs * self.ts * self.cosmology.Om0)
                src_modified = src.copy()
                for i in range(len(src)):
                    if self.sM01_msun[i] > 0:
                        if self.xh[int(src[i][0]), int(src[i][1]), int(src[i][2])] > self.StillNeutral:
                            src_modified[i, 4] = self.sM01_msun[i] * mass2phot_hm
                            self.sM01_msun[i] = 0.0
                            self.NumSupprsdSrc += 1
                        else:
                            src_modified[i, 4] = self.sM01_msun[i] * mass2phot_lm
                normflux = (self.sM00_msun * mass2phot_hm / S_star_ref) + (src_modified[:, 4] / S_star_ref)

            elif self.source_model == 3:  # Mass-dependent suppression
                srcpos = src[:, :3].T
                mass2phot_hm = msun2g * self.fgamma_hm * self.cosmology.Ob0 / (self.mean_molecular * m_p_cgs * self.ts * self.cosmology.Om0)
                src_modified = src.copy()
                for i in range(len(src)):
                    if self.sM01_msun[i] > 0:
                        if self.xh[int(src[i][0]), int(src[i][1]), int(src[i][2])] > self.StillNeutral:
                            src_modified[i, 5] = src[i, 5] * mass2phot_hm
                            self.NumSupprsdSrc += 1
                        else:
                            src_modified[i, 5] = src[i, 4] * mass2phot_hm
                normflux = (self.sM00_msun * mass2phot_hm / S_star_ref) + (src_modified[:, 5] / S_star_ref)

        # Print statistics
        self.printlog('\n---- Reading source file with total of %d ionizing source:\n%s' % (normflux.size, file))
        self.printlog(' Src model: ',self.source_model)
        self.printlog(' Total number of source locations, no suppression: %d' % num_total_sources)
        self.printlog(' Number of suppressable sources: %d' % self.NumSupprbleSrc)
        self.printlog(' Number of suppressed sources: %d' % self.NumSupprsdSrc)
        self.printlog(' Number of massive sources: %d' % self.NumHMACHSrc)
        self.printlog(' Number of sources, after suppression: %d' % (self.NumHMACHSrc + self.NumSupprbleSrc - self.NumSupprsdSrc))
        self.printlog(' Total flux: %.3e [s^-1]' % (np.sum(normflux) * S_star_ref))
        self.printlog(' min, max normflux: %.3e  %.3e' % (normflux.min(), normflux.max()))
        return srcpos, normflux

    def read_density(self, z):
        """Read coarser density field from C2Ray-formatted file and track history.

        Parameters
        ----------
        z : float
            Current redshift to read density for.
        """
        if self.cosmological:
            redshift = z
        else:
            redshift = self.zred_0

        # Find the closest redshift density file available
        high_z = self.zred_density[
            np.argmin(
                np.abs(self.zred_density[self.zred_density >= redshift] - redshift)
            )
        ]

        if high_z != self.prev_zdens:
            # 1. Store the current density as the previous one before updating
            # If this is the very first read, self.ndens might not exist yet
            if hasattr(self, 'ndens') and self.ndens is not None:
                self.prev_ndens = self.ndens.copy()
            else:
                self.prev_ndens = None

            file = "%scoarser_densities/%.3fn_all.dat" % (self.inputs_basename, high_z)
            self.printlog("\n---- Reading density file:\n " + file)

            self.dens_ND = t2c.DensityFile(filename=file).cgs_density
            self.dens_ND = self.dens_ND/np.mean(self.dens_ND)

            file = "%scoarser_densities/%.3fntotcoarsened_all.dat" % (self.inputs_basename, high_z)
            self.printlog("\n---- Reading density file (without halos):\n " + file)   

            self.ndens = (
                t2c.DensityFile(filename=file).cgs_density
                / (self.mean_molecular * c.m_p.cgs.value)
                * (1 + redshift) ** 3
            )          

            # self.dens_ND  = self.ndens / np.mean(self.ndens)

            # 3. Logic for the first redshift of the sim: prev_ndens = ndens
            if self.prev_ndens is None:
                self.printlog(" First density read: setting prev_ndens = ndens")
                self.prev_ndens = self.ndens.copy()

            self.printlog(
                " min, mean and max density : %.3e  %.3e  %.3e [1/cm3]"
                % (self.ndens.min(), self.ndens.mean(), self.ndens.max())
            )
            self.prev_zdens = high_z
        else:
            # No new file read; ndens and prev_ndens remain as they were
            pass

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
                np.save(file=self.results_basename + "jLW3d_" + suffix, arr=self.jLW)

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

                text = "%.3f\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n" % (
                    z,
                    tot_nHI,
                    # self.tot_phots,
                    np.mean(self.ndens),
                    np.mean(self.phi_ion),
                    # self.R_max_LLS / self.N * self.boxsize,
                    np.mean(self.xh),
                    massavrg_ion_frac,
                )
                f.write(text)
        else:
            # this is for the other ranks
            pass

    # =====================================================================================================
    # Below are the overridden initialization routines specific to the CubeP3M case
    # =====================================================================================================

    def _redshift_init(self):
        super()._redshift_init()
        """Initialize time and redshift counter"""
        # self.zred_density = t2c.get_dens_redshifts(
        #     self.inputs_basename + "coarser_densities/"
        # )[::-1]
        # # self.zred_sources = get_source_redshifts(self.inputs_basename+'sources/')[::-1]
        # # TODO: waiting for next tools21cm release
        # self.zred_sources = t2c.get_source_redshifts(self.inputs_basename + "sources/")[
        #     ::-1
        # ]

        #Edited by DA
        self.density_basename = self._ld["Output"]["density_basename"]
        self.zred_density = np.loadtxt(self.inputs_basename + "redshifts_fine.dat", skiprows=1)
        self.zred_sources = np.loadtxt(self.inputs_basename + "redshifts_checkpoints.txt", skiprows=1)

        if self.resume:
            # get the resuming redshift
            self.zred = np.min(get_redshifts_from_output(self.results_basename))
            _, self.prev_zdens = find_bins(self.zred, self.zred_density)
            _, self.prev_zsourc = find_bins(self.zred, self.zred_sources)
        else:
            self.prev_zdens = -1
            self.prev_zsourc = -1

        self.time = self.zred2time(self.zred)

    def _material_init(self):
        """Initialize material properties of the grid"""
        if self.resume:
            # get fields at the resuming redshift
            self.ndens = (
                t2c.DensityFile(
                    filename="%scoarser_densities/%.3fn_all.dat"
                    % (self.inputs_basename, self.prev_zdens)
                ).cgs_density
                / (self.mean_molecular * c.m_p.cgs.value)
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
        self.source_model = self._ld["Sources"]["source_model"]
        self.ts = (self._ld["Sources"]["ts"] * u.Myr).cgs.value

    def _grid_init(self):
        """Set up grid properties"""
        super()._grid_init()
                
        # Calculate grid cell mass
        # m_p_cgs = c.m_p.cgs.value
        # cell_vol_cMpc3 = (self.boxsize / self.N)**3
        



    def skip_record_marker(f):
        np.fromfile(f, dtype=np.int32, count=1)

    def read_LGnMH_Mpc3(self, filename=None):
        if filename is None:
            filename = os.path.join(self.inputs_basename, "zred_halodelta1_nMHMpc3_Planck")

        if not os.path.exists(filename):
            print(f"CRITICAL: Minihalo fit file missing at {filename}")
            return

        with open(filename, 'rb') as f:
            # Read header: 2 x int32 (integer(kind=li))
            header = np.fromfile(f, dtype=np.int32, count=2)
            self.Nzdata, self.Ndelta1data = int(header[0]), int(header[1])
            print("Nzdata, Ndelta1data:", self.Nzdata, self.Ndelta1data)
            
            # Read bounds: 2 x float64 (real(kind=dp))
            bounds = np.fromfile(f, dtype=np.float64, count=2)
            self.LGdelta1min, self.LGdelta1max = bounds[0], bounds[1]
            print("LGdelta1min, LGdelta1max:", self.LGdelta1min, self.LGdelta1max)
            
            # Read redshift array: Nzdata x float32 (real(kind=si))
            self.zred_array_interp = np.fromfile(f, dtype=np.float32, count=self.Nzdata)
            print("zred_array_interp range:", self.zred_array_interp[0], "to", self.zred_array_interp[-1])

            # Read 2D Table: Nzdata * Ndelta1data x float64 (real(kind=dp))
            data_flat = np.fromfile(f, dtype=np.float64, count=self.Nzdata * self.Ndelta1data)
            print(f"Read {len(data_flat)} values, expected {self.Nzdata * self.Ndelta1data}")
            
            # Fortran column-major order
            self.LGnMH_Mpc3 = data_flat.reshape((self.Nzdata, self.Ndelta1data), order='F')

        self.dLGdelta1 = (self.LGdelta1max - self.LGdelta1min) / (self.Ndelta1data - 1)
        print(f"Minihalo table loaded. Z: {self.zred_array_interp[0]:.2f} to {self.zred_array_interp[-1]:.2f}")
        print(f"LGnMH_Mpc3 shape: {self.LGnMH_Mpc3.shape}")

    def get_denscrit(self, zred, dens_ND, filename=None):
        if filename is None:
            filename = os.path.join(self.inputs_basename, "z_numMH_6.3Mpc_full")

        with open(filename, 'r') as f:
            size_smallbox = float(f.readline().strip())  # in h^-1 cMpc
            Ntable = int(f.readline().strip())
            data = np.loadtxt(f)
        ztable = data[:, 0]
        numMHtable = data[:, 1]

        # Volume in (Mpc/h)^3 — matching Fortran's vol_cMpc3
        vol_cMpc3 = (self.boxsize / self.N)**3  # boxsize is in Mpc/h

        # vol_ratio: small box volume / total simulation volume
        vol_ratio = (size_smallbox / self.h)**3 / (vol_cMpc3 * self.N**3)

        # NMH_smallbox: matches Fortran's int(...) + 1 converted to 0-based
        dz_table = (ztable[Ntable-1] - ztable[0]) / float(Ntable - 1)
        NMH_smallbox = numMHtable[int((zred - ztable[0]) / dz_table)]  # 0-based = Fortran's +1

        # Find idx_zred in zred_array_interp (NOT ztable)
        # Matches Fortran: find itable where zred_array_interp(itable-1) >= zred_round
        #                                 and zred_array_interp(itable)   < zred_round
        idx_zred = 0  # 0-based default
        zred_round = np.float32(round(zred * 1000) / 1000.0)  # match float32 precision of table

        if self.Nzdata > 1:
            for itable in range(1, self.Nzdata):
                z_prev = self.zred_array_interp[itable - 1]  # already float32
                z_curr = self.zred_array_interp[itable]       # already float32
                if z_prev >= zred_round and z_curr < zred_round:
                    idx_zred = itable - 1
                    break

        idx_zred = np.clip(idx_zred, 0, self.Nzdata - 1)

        # Bisection
        lg_low, lg_high = -1.0, 1.0
        n_max_iter = 14  # matches Fortran's Nmaxiter

        log_flat_dens = np.log10(dens_ND).flatten()

        lg_mid = 0.0
        for i in range(1, n_max_iter + 1):
            lg_mid = (lg_low + lg_high) * 0.5

            mask = log_flat_dens >= lg_mid

            if np.any(mask):
                idx_delta = ((log_flat_dens[mask] - self.LGdelta1min) / self.dLGdelta1).astype(int)
                idx_delta = np.clip(idx_delta, 0, self.Ndelta1data - 1)
                nmh_simbox = np.sum(10**self.LGnMH_Mpc3[idx_zred, idx_delta])
            else:
                nmh_simbox = 0.0

            # Scale to simulation box — vol in (Mpc/h)^3, h-correction for table
            nmh_simbox *= vol_cMpc3 * (self.h / 0.7)**3

            frac = (nmh_simbox * vol_ratio - NMH_smallbox) / NMH_smallbox

            if abs(frac) <= 0.01 or i == n_max_iter:
                break

            if nmh_simbox * vol_ratio < NMH_smallbox:
                lg_high = lg_mid
            else:
                lg_low = lg_mid

        dens_crit = 10**lg_mid
        return dens_crit


    def subsrcM_msun(self, zred, dens_nd, dens_nd_crit):
        """Port of function subsrcM_msun."""
        if dens_nd < dens_nd_crit:
            return 0.0
        
        idx_zred = np.searchsorted(self.zred_array_interp, zred) - 1
        idx_zred = np.clip(idx_zred, 0, self.Nzdata - 1)
        lg_delta = np.log10(max(dens_nd, 1e-5))
        
        idx_delta = int((lg_delta - self.LGdelta1min) / self.dLGdelta1)
        idx_delta = np.clip(idx_delta, 0, self.Ndelta1data - 1)

        # n_mh per Mpc^3 * cell_volume * h-scaling
        n_mh = 10**self.LGnMH_Mpc3[idx_zred, idx_delta] * (self.boxsize/self.N)**3 * (self.h/0.7)**3
        
        return n_mh * self.M_PIIIstar_msun

    def get_jLWcrit(self, zred):
        """Port of function jLWcrit."""
        return 0.1 * 1e-21 # erg/s/cm^2/Hz/sr
    
    def get_subsrcM_msun_all(self, zred, dens_nd_grid, dens_nd_crit):
        """
        Vectorized version of subsrcM_msun for all grid cells.
        Returns mass in solar masses for each cell.
        """
        # Initialize output
        mass_grid = np.zeros_like(dens_nd_grid)
        
        # Only calculate for cells above critical density
        mask = dens_nd_grid >= dens_nd_crit
        
        if not np.any(mask):
            return mass_grid
        
        # Find redshift index
        
        # NEW idx_zred
        idx_zred = 0
        zred_round = np.float32(round(zred * 1000) / 1000.0)
        if self.Nzdata > 1:
            for itable in range(1, self.Nzdata):
                z_prev = self.zred_array_interp[itable - 1]
                z_curr = self.zred_array_interp[itable]
                
                if z_prev >= zred_round and z_curr < zred_round:
                    idx_zred = itable - 1
                    break

        else:
            idx_zred = 0

        if (idx_zred < 0):
            idx_zred = 0
        if (idx_zred >= self.Nzdata):
            idx_zred = self.Nzdata-1
        # end new inx_zred

        # Calculate log density for masked cells
        lg_delta = np.log10(np.maximum(dens_nd_grid[mask], 1e-14))
        
        # Find delta indices
        idx_delta = ((lg_delta - self.LGdelta1min) / self.dLGdelta1).astype(int)
        idx_delta = np.clip(idx_delta, 0, self.Ndelta1data - 1)
        
        # Calculate number density of minihalos per Mpc^3
        n_mh = 10**self.LGnMH_Mpc3[idx_zred, idx_delta]
        # Convert to mass per cell
        vol_cMpc3 = (self.boxsize/self.h/self.N)**3 
        h_scaling = (self.h / 0.7)**3
        mass_grid[mask] = n_mh * vol_cMpc3 * h_scaling * self.M_PIIIstar_msun
        return mass_grid

    def update_agrid_properties(self, nz, AGlifetime, jLWgrid):
        """
        Complete port of AGrid_properties handling MHflag 1 and 2.
        """
        zred_now = self.zred_array[nz]
        
        self.M_box = self.rho_crit_0* self.cosmology.Om0 *(self.boxsize*self.Mpc / self.h)**3 
        self.M_grid = self.M_box/(self.n_box**3) 
        self.M_particle = 8.0*self.M_grid 

        # 1. Calculate Critical Density Threshold (only needed for MHflag 2)
        if self.MHflag == 2:
            self.densNDcrit = self.get_denscrit(zred_now, self.dens_ND)
            if nz > 0:
                self.densNDcrit_prev = self.get_denscrit(self.zred_array[nz-1], self.dens_ND_prev)
        
        # 2. Calculate Total Potential Mass per cell
        mass_now = self.get_subsrcM_msun_all(zred_now, self.dens_ND, self.densNDcrit)
        self.printlog(f"zred: {zred_now:.4f}, densNDcrit: {self.densNDcrit:.7f}")
        # 3. Calculate Differential Mass (The "Fresh" Minihalos/Sources)
        # This prevents re-igniting sources from the previous step
        if nz > 0:
            mass_prev = self.get_subsrcM_msun_all(self.zred_array[nz-1], 
                                                 self.dens_ND_prev, 
                                                 self.densNDcrit_prev)
            diff_subsrcMsun = mass_now - mass_prev
        else:
            diff_subsrcMsun = mass_now
       

        # # Ensure no negative growth due to numerical fluctuations
        # diff_subsrcMsun = np.maximum(diff_subsrcMsun, 0.0)

        # 4. Filter for Active Grids (Neutral cells with fresh mass)
        jLWc_now = self.get_jLWcrit(zred_now)
        jLWc_min = 0.1 * jLWc_now

        # Mask: StillNeutral check and LW suppression check
        active_mask = (self.xh < self.StillNeutral) & \
                      (jLWgrid < jLWc_now) & \
                      (diff_subsrcMsun > 0)
        print("self.StillNeutral = ", self.StillNeutral)
        print("self.xh > self.StillNeutral ", np.sum(self.xh > self.StillNeutral))
        print("jLWgrid > jLWc_now ", np.sum(jLWgrid > jLWc_now))
        print("diff_subsrcMsun < 0 ", np.sum(diff_subsrcMsun <= 0))
        print("diff_subsrcMsun > 0 ", np.sum(diff_subsrcMsun > 0))
        print("****************************************************************************************")
        print("Number of active grids: ", np.sum(active_mask))
        print("np.sum(diff_subsrcMsun) = ", np.sum(diff_subsrcMsun))
        
        if not np.any(active_mask):
            return None, None, 0

        # 5. Apply Lyman-Werner Suppression
        # Get active grid indices and count
        active_indices = np.argwhere(active_mask)
        NumAGrid = len(active_indices)

        self.ssM_msun = np.zeros(NumAGrid)

        for idx, (i, j, k) in enumerate(active_indices):
            if jLWgrid[i, j, k] <= jLWc_min:
                # No suppression branch (jLW < jLWc_min)
                self.ssM_msun[idx] = diff_subsrcMsun[i, j, k]
            else:
                # Partial suppression for jLW > jLWc_min
                self.ssM_msun[idx] = diff_subsrcMsun[i, j, k] * (
                    (jLWc_now - jLWgrid[i, j, k]) /
                    (jLWc_now - jLWc_min)
                )
        
        tot_subsrcM_msun = np.sum(self.ssM_msun)
        print("tot_subsrcM_msun = ",tot_subsrcM_msun)
        self.printlog(f"Total active stellar mass in subgrids, in solar mass: {np.sum(tot_subsrcM_msun)}", self.logfile)
        print("Total mini halos generated = ", tot_subsrcM_msun/self.M_PIIIstar_msun)

        # 6. Convert to Grid Units and Normalized Flux
        # Constants from Fortran module: Omega_B, Omega0, m_p, M_SOLAR, S_star_nominal
        m_p_cgs = c.m_p.cgs.value
        m_solar_cgs = 1.98892e33 # astroconstants.M_SOLAR
        
        if self.MHflag == 1:
            # Case 1: Proportional to Baryon fraction
            # subsrcMass = self.ssM_msun * (M_SOLAR/M_grid) * phot_per_atom[2]
            subsrcMass = self.ssM_msun * (m_solar_cgs / self.M_grid) * self.phot_per_atom[2]
            subNormFlux = (subsrcMass * self.M_grid * self.cosmology.Ob0 / 
                          (self.cosmology.Om0 * m_p_cgs)) / (self.S_star_nominal * AGlifetime)
            
        elif self.MHflag == 2:
            # Case 2: Discrete Pop III stars
            # subsrcMass = self.ssM_msun * (M_SOLAR/M_grid) * phot_per_atom[2] / fstar[2]
            subsrcMass = self.ssM_msun * (m_solar_cgs / self.M_grid) * self.phot_per_atom[2] / self.fstar[2]
            subNormFlux = (subsrcMass * self.M_grid / m_p_cgs) / (self.S_star_nominal * AGlifetime)
            print('Subgrid Total flux= ',sum(subNormFlux))

        self.printlog('Subgrid Total flux= ',sum(subNormFlux))
        # 7. Save subgrid source list to file (matching Fortran format)
        z_str = f"{zred_now:6.3f}"
        sourcelistfile_sub = f"{self.results_basename}/{z_str}-coarsened_SUBsources.dat"
        with open(sourcelistfile_sub, 'w') as f:
            # Write number of active grids
            f.write(f"{NumAGrid}\n")
            # Write MHflag to distinguish physical meaning
            # When MHflag=2, self.ssM_msun is STELLAR BARYON MASS, not BARYON+DM HALO MASS
            f.write(f"{self.MHflag}\n")
            # Write source data: i, j, k, subsrcMass, self.ssM_msun
            # Format matches Fortran: 3I5,2e13.4
            for idx in range(NumAGrid):
                i, j, k = active_indices[idx]
                f.write(f"{i:5d}{j:5d}{k:5d}{subsrcMass[idx]:13.4e}{self.ssM_msun[idx]:13.4e}\n")
        print(f"Saved subgrid sources to: {sourcelistfile_sub}")
        # 8. Randomize for raytracing order (Equivalent to Fortran's call permi)
        p = np.random.permutation(NumAGrid)
        
        return active_indices[p], subNormFlux[p], NumAGrid
    
    def _init_jLW_constants(self):
        """Port of get_HcOm: Simple constant calculation in unit of Mpc^-1."""
        speed_of_light_cms = c.c.cgs.value
        # h*100*1e5 converts km/s/Mpc to cm/s/Mpc
        self.HcOm = (self.h * 100.0 * 1e5 / speed_of_light_cms) * np.sqrt(self.cosmology.Om0)
             
    def get_rLW(self, zobs):
        """Port of get_rLW: Lyman-Werner horizon in comoving Mpc."""
        self.rLW_zobs = 2/self.HcOm * (zobs+1)**(-0.5) * (1 - ((15/16)/(8/9))**(-0.5))
        
        return self.rLW_zobs

    def read_greenK(self, zsbegin, zsend, zobs):
        zb_str = f"{zsbegin:6.3f}".strip()
        ze_str = f"{zsend:6.3f}".strip()
        zo_str = f"{zobs:6.3f}".strip()
        fname = os.path.join(self.inputs_basename, "GKresults", f"gK_{zb_str}-{ze_str}-{zo_str}_dat")
        print("Reading ", fname)
        with open(fname, "rb") as f:

            # --- read mesh sizes (3 x int32) ---
            m1, m2, m3 = struct.unpack("<iii", f.read(12))
            print("m1, m2, m3 =", m1, m2, m3)

            if (m1 != self.N) & (m2 != self.N) & (m3 != self.N):  
                raise ValueError("mesh number not matched for greenK!! Aborting!")

            # --- read complex(dp) array ---
            nkx = m1 // 2 + 1
            nvals = nkx * m2 * m3

            greenK = np.fromfile(f, dtype=np.complex128, count=nvals)

        # reshape exactly like Fortran
        greenK = greenK.reshape((nkx, m2, m3), order="F")

        return greenK


    def get_srclumK(self, nz, srcpos, normflux, srcpos_mh, normflux_mh):
        """
        Port of get_srclumK: Create and FFT the source luminosity distribution.
        Returns the Fourier transform matching Fortran's layout: (N//2+1, N, N)
        """
        # Calculate C2Ray lifetime for this redshift interval
        C2ray_lifetime = abs(self.zred2time(self.zred_array[nz]) - 
                            self.zred2time(self.zred_array[nz+1]))
        
        QH_M_C2ray00  = self.Ni[0] * (self.M_solar/c.m_p.cgs.value)  /C2ray_lifetime
        QH_M_C2ray01  = self.Ni[1] * (self.M_solar/c.m_p.cgs.value)  /C2ray_lifetime
        
        # Correction coefficients
        CC00 = QH_M_C2ray00 / self.QH_M_real00
        CC01 = QH_M_C2ray01 / self.QH_M_real01
        
        # Coefficients for source luminosity
        coeff00 = self.emis00 * CC00 * self.fstar[0] * self.cosmology.Ob0 / self.cosmology.Om0
        coeff01 = self.emis01 * CC01 * self.fstar[1] * self.cosmology.Ob0 / self.cosmology.Om0

        # Initialize source luminosity grid (Fortran order is important!)
        srclum = np.zeros(self.shape, order='F')
        
        # Add HMACHs
        if srcpos is not None and normflux is not None:
            for i in range(normflux.size):
                ix, iy, iz = srcpos[:, i].astype(int)
                if 0 <= ix < self.N and 0 <= iy < self.N and 0 <= iz < self.N:
                    srclum[ix, iy, iz] += self.sM00_msun[i] * coeff00

        # Add HMACHs
        if srcpos is not None and normflux is not None:
            for i in range(normflux.size):
                ix, iy, iz = srcpos[:, i].astype(int)
                if 0 <= ix < self.N and 0 <= iy < self.N and 0 <= iz < self.N:
                    srclum[ix, iy, iz] += self.sM00_msun[i] * coeff00

        if srcpos is not None and normflux is not None:
            for i in range(normflux.size):
                ix, iy, iz = srcpos[:, i].astype(int)
                if 0 <= ix < self.N and 0 <= iy < self.N and 0 <= iz < self.N:
                    srclum[ix, iy, iz] += self.sM01_msun[i] * coeff01 # For suppressed sources self.sM01_msun = 0

        # Add minihalo/subgrid sources
        if srcpos_mh is not None and normflux_mh is not None:
            if self.MHflag == 1:
                QH_M_C2ray_sub = self.Ni[2] * (self.M_solar / c.m_p.cgs.value) / C2ray_lifetime
                CC_sub = QH_M_C2ray_sub / self.QH_M_real_sub
                coeff_sub = self.emissub * CC_sub * self.fstar[2] * self.cosmology.Ob0 / self.cosmology.Om0
                print('Sanity check: fesc_sub = ', self.phot_per_atom[2] /(self.Ni[2] *self.fstar[2]) )
            elif self.MHflag == 2:
                QH_M_C2ray_sub = self.Ni[2] * (self.M_solar / c.m_p.cgs.value) / C2ray_lifetime
                CC_sub = QH_M_C2ray_sub / self.QH_M_real_sub
                coeff_sub = self.emissub * CC_sub
                print('Sanity check: fesc_sub = ', self.phot_per_atom[2] /(self.Ni[2] *self.fstar[2]) )
            
            for i in range(normflux_mh.size):
                ix, iy, iz = srcpos_mh[:, i].astype(int)
                if 0 <= ix < self.N and 0 <= iy < self.N and 0 <= iz < self.N:
                    srclum[ix, iy, iz] = srclum[ix, iy, iz] + self.ssM_msun[i] * coeff_sub

        # CRITICAL FIX: Match Fortran FFT convention
        # Fortran reduces first dimension, Python rfftn reduces last dimension
        # Solution: transpose before and after FFT
        
        # Transpose: (N, N, N) -> (N, N, N) but with axes swapped
        srclum_T = np.transpose(srclum, (2, 1, 0))  # Now (Nz, Ny, Nx)
        
        # FFT: reduces last axis, giving (Nz, Ny, Nx//2+1)
        srclumK_T = np.fft.rfftn(srclum_T)
        
        # Transpose back to Fortran convention: (Nx//2+1, Ny, Nz)
        srclumK = np.transpose(srclumK_T, (2, 1, 0))
        # Normalize (matching Fortran normalization)
        srclumK = srclumK / float(self.N**3)
        
        return srclumK

    def save_srclumK(self, srclumK, z):
        """Save source luminosity Fourier transform for later use."""
        zstr = f"{z:6.3f}".strip()
        fname = os.path.join(self.results_basename, f"{zstr}-srcK.npy")
        np.save(fname, srclumK)
    
    def load_srclumK(self, z):
        """Load previously saved source luminosity Fourier transform."""
        zstr = f"{z:6.3f}".strip()
        fname = os.path.join(self.results_basename, f"{zstr}-srcK.npy")
        return np.load(fname)

    def compute_jLW_from_history(self, nz0, nz):
        """
        Complete port of get_jLW: Accumulate LW contributions from all past redshift slices.
        """
        # Observer redshift is the END of the current slice
        zobs = self.zred_array[nz+1]
        
        # Calculate LW horizon
        self.get_rLW(zobs)
        
        # Find starting redshift for LW calculation
        zstart = ((1.0 + zobs)**(-0.5) - self.rLW_zobs * 0.5 * self.HcOm)**(-2.0) - 1.0
        # Find which redshift index to start from
        nz_LWbegin = 0
        if zstart > self.zred_array[0]:
            nz_LWbegin = 0
        else:
            for nzz in range(nz + 1):
                if (self.zred_array[nzz] >= zstart and 
                    zstart > self.zred_array[nzz + 1]):
                    nz_LWbegin = nzz
                    break
        
        n_pastslice = nz + 1 - nz_LWbegin
        self.printlog(f"LW calculation: {n_pastslice} past slices from z={self.zred_array[nz_LWbegin]:.3f}", 
                    self.logfile)

        # Initialize jLW accumulator
        jLW_total = np.zeros(self.shape, order='F')
        
        # Loop over all past slices that contribute
        for nzz in range(nz_LWbegin, nz + 1):
            zsbegin = self.zred_array[nzz]
            zsend = self.zred_array[nzz + 1]
            
            self.printlog(f"  Adding LW contribution from z={zsbegin:.3f} to z={zsend:.3f}", 
                        self.logfile)
            
            # Read Green's function for this slice observed at zobs
            # greenK is already in correct shape (N//2+1, N, N) from read_greenK
            greenK = self.read_greenK(zsbegin, zsend, zobs)
            srclumK = self.load_srclumK(zsbegin)

            # CRITICAL FIX: Both arrays now have shape (N//2+1, N, N)
            # Direct multiplication (convolution theorem)
            gK_sK = greenK * srclumK

            # Inverse FFT back to real space
            # Need to match Fortran's C2R FFT which expects first dimension reduced
            # Transpose to Python convention, IFFT, transpose back
            gK_sK_T = np.transpose(gK_sK, (2, 1, 0))
        
            # Inverse FFT WITHOUT normalization (norm='forward' means no normalization on inverse)
            # FFTW does NOT normalize, so we need to match that
            jLW_contrib_T = np.fft.irfftn(gK_sK_T, s=(self.N, self.N, self.N), norm='forward')
            
            # Transpose back to Fortran order
            jLW_contrib = np.transpose(jLW_contrib_T, (2, 1, 0))
            
            # Accumulate
            jLW_total += jLW_contrib
        
        # Ensure non-negative and Fortran-ordered
        jLW_total = np.maximum(jLW_total, 0.0)
        jLW_total = np.asfortranarray(jLW_total)
        
        jLW_mean = np.mean(jLW_total)
        self.printlog(f"Total LW background: mean={jLW_mean:.3e} erg/s/cm^2/Hz/sr", self.logfile)
        
        return jLW_total

    def compute_jLW_grid(self, source_grid):
        """
        Uses the Green function to calculate the LW background across the grid.
        source_grid: 3D array of source intensities.
        """
        if self.greenK is None:
            raise ValueError("GreenK not loaded. Call read_greenK first.")

        # 1. FFT the sources into k-space
        # Note: We use rfftn because the physical density/source field is real.
        # However, the Fortran GreenK is (mesh/2+1, m, m). 
        # We may need to transpose source_grid to match the GreenK orientation.
        source_k = np.fft.rfftn(source_grid)

        # 2. Multiply by Green function in k-space (Convolution)
        # Note: Ensure shapes match exactly. NumPy rfftn output is (m, m, m/2+1)
        # while Fortran file is (m/2+1, m, m). 
        jLW_k = source_k * self.greenK.T 

        # 3. Inverse FFT back to real space
        jLW_grid = np.fft.irfftn(jLW_k, s=source_grid.shape)
        
        return np.maximum(jLW_grid, 0.0)
    
    def _init_jLW_constants(self):
        """Port of get_HcOm: Simple constant calculation in unit of Mpc^-1."""
        speed_of_light_cms = c.c.cgs.value
        # h*100*1e5 converts km/s/Mpc to cm/s/Mpc
        self.HcOm = (self.h * 100.0 * 1e5 / speed_of_light_cms) * np.sqrt(self.cosmology.Om0)

    def get_rLW(self, zobs):
        """Port of get_rLW: Lyman-Werner horizon in comoving Mpc."""
        factor = 1.0 - (135.0 / 128.0)**(-0.5)
        self.rLW_zobs = (2.0 / self.HcOm) * (zobs + 1.0)**(-0.5) * factor
        return self.rLW_zobs

    def compute_jLW_grid(self, source_grid):
        """
        Uses the Green function to calculate the LW background across the grid.
        source_grid: 3D array of source intensities.
        """
        if self.greenK is None:
            raise ValueError("GreenK not loaded. Call read_greenK first.")

        # 1. FFT the sources into k-space
        # Note: We use rfftn because the physical density/source field is real.
        # However, the Fortran GreenK is (mesh/2+1, m, m). 
        # We may need to transpose source_grid to match the GreenK orientation.
        source_k = np.fft.rfftn(source_grid)

        # 2. Multiply by Green function in k-space (Convolution)
        # Note: Ensure shapes match exactly. NumPy rfftn output is (m, m, m/2+1)
        # while Fortran file is (m/2+1, m, m). 
        jLW_k = source_k * self.greenK.T 

        # 3. Inverse FFT back to real space
        jLW_grid = np.fft.irfftn(jLW_k, s=source_grid.shape)
        
        return np.maximum(jLW_grid, 0.0)