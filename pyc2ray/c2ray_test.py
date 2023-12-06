from .c2ray_base import C2Ray, YEAR, Mpc
from .utils.sourceutils import read_test_sources
import numpy as np
import pickle as pkl

__all__ = ['C2Ray_Test']

# ======================================================================
# This file contains the C2Ray_Test subclass of C2Ray, which is a
# version used for test simulations, i.e. which don't read N-body input
# and use simple source files
# ======================================================================

class C2Ray_Test(C2Ray):
    def __init__(self, paramfile, Nmesh, use_gpu, use_mpi=None):
        """A C2Ray Test-case simulation

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
        if(self.rank == 0): self.printlog('Running: "C2Ray Test"')

    def read_sources(self,file,numsrc,S_star_ref = 1e48):
        """ Read in a source file formatted for Test-C2Ray

        Read in a file that gives source positions and total ionizing flux
        in s^-1, formatted like for the original C2Ray code, e.g.

        1.0
        40 40 40 1e54 1.0

        for one source at (40,40,40) (Fortran indexing) and of total flux
        1e54 photons/s. The last row is conventional.

        Parameters
        ----------
        file : string
            Name of the file to read
        numsrc : int
            Numer of sources to read from the file
        S_star_ref : float, optional
            Flux of the reference source. Default: 1e48
            There is no real reason to change this, but if it is changed, the value in src/c2ray/photorates.f90
            has to be changed accordingly and the library recompiled.
            
        Returns
        -------
        src_pos : 2D-array of shape (3,numsrc)
            Source positions
        src_flux : 1D-array of shape (numsrc)
            Normalization of the strength of each source, i.e. total ionizing flux / reference flux
        """
        return read_test_sources(file,numsrc,S_star_ref)
        

    def density_init(self,z):
        """Set density at redshift z

        Sets the density to a constant value, specified in the parameter file,
        that is scaled to redshift if the run is cosmological.

        Parameters
        ----------
        z : float
            Redshift slice
        
        """
        self.set_constant_average_density(self.avg_dens,z)

    def write_output(self,z):
        """Write ionization fraction & ionization rates as pickle files

        Parameters
        ----------
        z : float
            Redshift (used to name the file)
        """
        suffix = f"_{z:.3f}.pkl"
        with open(self.results_basename + "xfrac" + suffix,"wb") as f:
            pkl.dump(self.xh,f)
        with open(self.results_basename + "IonRates" + suffix,"wb") as f:
            pkl.dump(self.phi_ion,f)

    def write_output_numbered(self,n):
        """Write ionization fraction & ionization rates as pickle files with number rather than redshift

        Parameters
        ----------
        n : int
            Number of the file
        """
        suffix = f"_{n:n}.pkl"
        with open(self.results_basename + "xfrac" + suffix,"wb") as f:
            pkl.dump(self.xh,f)
        with open(self.results_basename + "IonRates" + suffix,"wb") as f:
            pkl.dump(self.phi_ion,f)

    def set_constant_average_density(self,ndens,z):
        """Helper function to set the density grid to a constant value

        Parameters
        ----------
        ndens : float
            Value of the hydrogen density in cm^-3. When in a cosmological
            run, this is the comoving density, or the proper density at
            z = 0.
        z : float
            Redshift to scale the density. When cosmological is false,
            this parameter has no effect and the initial redshift specified
            in the parameter file is used at each call.
        """
        # This is the same as in C2Ray
        if self.cosmological:
            redshift = z
        else:
            redshift = self.zred_0
        self.ndens = ndens * np.ones(self.shape,order='F') * (1+redshift)**3

    def generate_redshift_array(self,num_zred,delta_t):
        """Helper function to generate a list of equally-time-spaced redshifts

        Generate num_zred redshifts that correspond to cosmic ages
        separated by a constant time interval delta_t. Useful for the
        test case of C2Ray. The initial redshift is set in the parameter file.

        Parameters
        ----------
        num_zred : int
            Number of redshifts to generate
        delta_t : float
            Spacing between redshifts in years

        Returns
        -------
        zred_array : 1D-array
            List of redshifts (including initial one)
        """
        step = delta_t * YEAR
        zred_array = np.empty(num_zred)
        for i in range(num_zred):
            zred_array[i] = self.time2zred(self.age_0 + i*step)
        return zred_array

    # =====================================================================================================
    # Below are the overridden initialization routines specific to the test case
    # =====================================================================================================

    def _redshift_init(self):
        """Initialize time and redshift counter
        """
        self.time = self.age_0
        self.zred = self.zred_0

    def _material_init(self):
        """Initialize material properties of the grid
        """
        xh0 = self._ld['Material']['xh0']
        temp0 = self._ld['Material']['temp0']

        self.ndens = np.empty(self.shape,order='F')
        self.xh = xh0 * np.ones(self.shape,order='F')
        self.temp = temp0 * np.ones(self.shape,order='F')
        self.phi_ion = np.zeros(self.shape,order='F')
        self.avg_dens = self._ld['Material']['avg_dens']

    def _output_init(self):
        """ Set up output & log file
        """
        self.results_basename = self._ld['Output']['results_basename']
        self.logfile = self.results_basename + self._ld['Output']['logfile']
        title = '                 _________   ____            \n    ____  __  __/ ____/__ \ / __ \____ ___  __\n   / __ \/ / / / /    __/ // /_/ / __ `/ / / /\n  / /_/ / /_/ / /___ / __// _, _/ /_/ / /_/ / \n / .___/\__, /\____//____/_/ |_|\__,_/\__, /  \n/_/    /____/                        /____/   \n'
        if(self.rank == 0):
            with open(self.logfile,"w") as f:
                f.write("\nLog file for pyC2Ray \n\n")
            self.printlog(title)