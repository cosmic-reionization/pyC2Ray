import h5py
import numpy as np
import tools21cm as t2c
from astropy import constants as c
from astropy import units as u
import os 

from .c2ray_base import C2Ray, msun2g
from .utils.other_utils import find_bins, get_redshifts_from_output

__all__ = ["C2Ray_CubeP3M"]

# ======================================================================
# This file contains the C2Ray_CubeP3M subclass of C2Ray, which is a
# version used for simulations that read in N-Body data from CubeP3M
# ======================================================================


class C2Ray_CubeP3M(C2Ray):
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
        self.printlog('Running: "C2Ray for %d Mpc/h volume"' % self.boxsize)

    def read_sources(self, file, mass='hm'): # >:( trgeoip
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
        
        # TODO: automatic selection of low mass or high mass. For the moment only high mass
        mass2phot_hm = msun2g * self.fgamma_hm * self.cosmology.Ob0 / (self.mean_molecular * c.m_p.cgs.value * self.ts * self.cosmology.Om0)    
        # For the low mass
        mass2phot_lm = msun2g * self.fgamma_lm * self.cosmology.Ob0 / (self.mean_molecular * c.m_p.cgs.value * self.ts * self.cosmology.Om0)    

        if file.endswith('.hdf5'):
            f = h5py.File(file, 'r')
            srcpos = f['sources_positions'][:].T
            assert srcpos.shape[0] == 3
            normflux = f['sources_mass'][:] * mass2phot / S_star_ref
            f.close()
        else:
            # use density fields generated from yt
            src = np.loadtxt(file, skiprows=1)

            if self.source_model is None or self.source_model == 0:#---------- No Supression Model -----------------
                if len(src.shape) == 1:
                    srcpos = src[:3].T
                    srcpos = srcpos.reshape((3, 1))
                    normflux = np.array([(src[3] * mass2phot_hm / S_star_ref) + (src[4] * mass2phot_lm / S_star_ref)])
                else:    
                    srcpos = src[:, :3].T
                    normflux = (src[:, 3] * mass2phot_hm / S_star_ref) + (src[:, 4] * mass2phot_hm / S_star_ref)
            
            if self.source_model == 1:#---------- Full Supression Model -----------------
                if len(src.shape) == 1:
                    srcpos = src[:3].T
                    srcpos = srcpos.reshape((3, 1))
                    # Fully suppress any LMACH source in a region with ion_frac > 0.9
                    # The -1 is added to the src positions as they are saved in fortran indexing
                    if self.xh[int(src[0]-1),int(src[1]-1),int(src[2]-1)] >0.9:
                        src[4]=0
                    normflux = np.array([(src[3] * mass2phot_hm / S_star_ref) + (src[4] * mass2phot_lm / S_star_ref)])
                else:    
                    srcpos = src[:, :3].T
                    # Looping through all the sources
                    for i in range(len(src)):
                        if self.xh[int(src[i][0]-1), int(src[i][1]-1), int(src[i][2]-1)] > 0.9:
                            # Fully supressing sources in ionized regions (>0.9)
                            src[i][4] = 0
                    normflux = (src[:, 3] * mass2phot_hm / S_star_ref) + (src[:, 4] * mass2phot_hm / S_star_ref)

            if self.source_model == 2:#---------- Partially Supression Model -----------------
                if len(src.shape) == 1:
                    srcpos = src[:3].T
                    srcpos = srcpos.reshape((3, 1))
                    # If LMACH is in ionized region then efficiency is the same as HMACH
                    # The -1 is added to the src positions as they are saved in fortran indexing
                    if self.xh[int(src[0]-1),int(src[1]-1),int(src[2]-1)] >0.9:
                        src[4] = src[4] * mass2phot_hm
                    # If LMACH is not in ionized region then we just multiply by LMACH efficiency
                    else:
                        src[4] = src[4] * mass2phot_lm
                    normflux = np.array([(src[3] * mass2phot_hm / S_star_ref) + (src[4] / S_star_ref)])
                else:    
                    srcpos = src[:, :3].T
                    # Looping through all the sources
                    for i in range(len(src)):
                        if self.xh[int(src[i][0]-1), int(src[i][1]-1), int(src[i][2]-1)] > 0.9:
                            # Fully supressing sources in ionized regions (>0.9)
                            src[i][4] = src[i][4] * mass2phot_hm
                        else:
                            src[i][4] = src[i][4] * mass2phot_lm
                    normflux = (src[:, 3] * mass2phot_hm / S_star_ref) + (src[:, 4] / S_star_ref)
            
            if self.source_model == 3: # ---------- Mass-dependent suppression of LMACHs ----------
                if len(src.shape) == 1:
                    srcpos = src[:3].T
                    srcpos = srcpos.reshape((3, 1))
                    # If LMACH is in ionized region then efficiency gradually supressed depending on mass
                    # The -1 is added to the src positions as they are saved in fortran indexing
                    if self.xh[int(src[0]-1),int(src[1]-1),int(src[2]-1)] >0.9:
                        src[5] = src[5] * mass2phot_hm
                    # If LMACH is not in ionized region then we just multiply original mass by HMACH efficiency
                    else:
                        src[5] = src[4] * mass2phot_hm
                    normflux = np.array([(src[3] * mass2phot_hm / S_star_ref) + (src[5] / S_star_ref)])
                else:    
                    srcpos = src[:, :3].T
                    # Looping through all the sources
                    for i in range(len(src)):
                        if self.xh[int(src[i][0]-1), int(src[i][1]-1), int(src[i][2]-1)] > 0.9:
                            # Fully supressing sources in ionized regions (>0.9)
                            src[i][5] = src[i][5] * mass2phot_hm
                        else:
                            src[i][5] = src[i][4] * mass2phot_hm
                    normflux = (src[:, 3] * mass2phot_hm / S_star_ref) + (src[:, 5] / S_star_ref)

        self.printlog('\n---- Reading source file with total of %d ionizing source:\n%s' %(normflux.size, file))
        self.printlog(' min, max source mass : %.3e  %.3e [Msun] and min, mean, max number of ionising sources : %.3e  %.3e  %.3e [1/s]' %(normflux.min()/mass2phot_hm*S_star_ref, normflux.max()/mass2phot_hm*S_star_ref, normflux.min()*S_star_ref, normflux.mean()*S_star_ref, normflux.max()*S_star_ref))
        return srcpos, normflux

    def read_density(self, z):
        """Read coarser density field from C2Ray-formatted file

        This method is meant for reading density field run with either N-body or hydro-dynamical simulations. The field is then smoothed on a coarse mesh grid.

        Parameters
        ----------
        n : int
            Number of sources to read from the file

        Returns
        -------
        srcpos : array
            Grid positions of the sources formatted in a suitable way for the chosen raytracing algorithm
        normflux : array
            density mesh-grid in csg units
        """
        if self.cosmological:
            redshift = z
        else:
            redshift = self.zred_0

        # TODO: redshift bin for the current redshift based on the density redshift for interpolation (discussed with Garrelt and he's okish)
        # TODO: low_z, high_z = find_bins(redshift, self.zred_density)
        # get the strictly larger and closest redshift density file
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
                / (self.mean_molecular * c.m_p.cgs.value)
                * (1 + redshift) ** 3
            )
            self.printlog(
                " min, mean and max density : %.3e  %.3e  %.3e [1/cm3]"
                % (self.ndens.min(), self.ndens.mean(), self.ndens.max())
            )
            self.prev_zdens = high_z
        else:
            # no need to re-read the same file again
            # TODO: in the future use this values for a 3D interpolation for the density (can be extended to sources too)
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
                # np.save(file=self.results_basename + "coldens" + suffix, arr=self.coldens)

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
        self.zred_density = np.loadtxt(self.inputs_basename + "redshifts_checkpoints.txt")
        self.zred_sources = np.loadtxt(self.inputs_basename + "redshifts_checkpoints.txt")

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

        # TODO: introduce an error due to the fact that we do not use 1/h
        # t2c.set_sim_constants(boxsize_cMpc=self._ld['Grid']['boxsize'])
        self.resume = self._ld["Grid"]["resume"]