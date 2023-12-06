import yaml
import atexit
import re
import numpy as np
from astropy import units as u
from astropy import constants as c
from astropy.cosmology import FlatLambdaCDM, z_at_value
try:
    from yaml import CSafeLoader as SafeLoader
except ImportError:
    from yaml import SafeLoader
from .utils.logutils import printlog
from .evolve import evolve3D, evolve3D_MPI
from .raytracing import do_raytracing
from .asora_core import device_init, device_close, photo_table_to_device
from .radiation import BlackBodySource, make_tau_table

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
# start (in cosmo_ini), the cell size & volume are scaled down to
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
# When doing direct comparisons with C2Ray, the difference between astropy.constants and the C2Ray values
# may be visible, thus we use the same exact value for the constants. This can be changed to the
# astropy values once consistency between the two codes has been established
pc = 3.086e18           #(1*u.pc).to('cm').value            # C2Ray value: 3.086e18
YEAR = 3.15576E+07      #(1*u.yr).to('s').value           # C2Ray value: 3.15576E+07
ev2fr = 0.241838e15                     # eV to Frequency (Hz)
ev2k = 1.0/8.617e-05                    # eV to Kelvin
kpc = 1e3*pc                            # kiloparsec in cm
Mpc = 1e6*pc                            # megaparsec in cm
msun2g = 1.98892e33 #(1*u.Msun).to('g').value       # solar mass to grams


class C2Ray:
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
        # MPI setup
        if use_mpi:
            self.mpi = use_mpi
            self.comm = use_mpi.COMM_WORLD
            self.rank = self.comm.Get_rank()
            self.nprocs = self.comm.Get_size()
        else:
            self.mpi = False
            self.rank = 0

        # Read YAML parameter file and set main properties
        self._read_paramfile(paramfile)
        self.N = Nmesh
        self.shape = (Nmesh,Nmesh,Nmesh)

        # Set Raytracing mode
        if use_gpu:
            self.gpu = True
            # Allocate GPU memory
            src_batch_size = self._ld["Raytracing"]["source_batch_size"]
            device_init(Nmesh,src_batch_size)
            # Register deallocation function (automatically calls this on program termination)
            atexit.register(self._gpu_close)
        else:
            self.gpu = False

        # Initialize Simulation
        self._param_init()
        self._output_init()
        self._grid_init()
        self._cosmology_init()
        self._redshift_init()
        self._material_init()
        self._sources_init()
        self._radiation_init()
        if(self.rank == 0):
            if self.gpu:
                # Print maximum shell size for info, based on LLS (qmax is s.t. Rmax fits inside of it)
                q_max = np.ceil(1.73205080757*min(self.R_max_LLS,1.73205080757*self.N/2))
                self.printlog(f"Using ASORA Raytracing ( q_max = {q_max : n} )")
            else:
                # Print info about subbox algorithm
                self.printlog(f"Using CPU Raytracing (subboxsize = {self.subboxsize : n}, max_subbox = {self.max_subbox : n})")
            if (self.mpi): self.printlog(f"Using {self.nprocs:n} MPI Ranks")
            else: self.printlog(f"Running in non-MPI (single-GPU/CPU) mode")
            self.printlog("Starting simulation... \n\n")

    # =====================================================================================================
    # TIME-EVOLUTION METHODS
    # =====================================================================================================
    def set_timestep(self,z1,z2,num_timesteps):
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
        t1 = self.cosmology.lookback_time(z1).to('s').value
        t2 = self.cosmology.lookback_time(z2).to('s').value
        # we do t1-t2 since ti are lookback times (not ages)
        dt = (t1-t2)/num_timesteps
        return dt
    
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
        if self.mpi:
            NumSrc = src_flux.shape[0]
            # TODO: this is a bit ugly but it works: 
            # if the number of sources exceed the number of MPI processors then call the evolve designed for the MPI source splitting.
            # otherwise: all ranks are calling (independently) the evolve with no source splitting until the condition above is meet.
            if(NumSrc >= self.nprocs):
                self.xh, self.phi_ion = evolve3D_MPI(
                    dt, self.dr,
                    src_flux, src_pos,
                    self.gpu, self.max_subbox,self.subboxsize,self.loss_fraction,
                    self.mpi,self.comm, self.rank, self.nprocs,
                    self.temp, self.ndens, self.xh,
                    self.photo_thin_table, self.photo_thick_table,
                    self.minlogtau, self.dlogtau,
                    self.R_max_LLS, self.convergence_fraction,
                    self.sig, self.bh00, self.albpow, self.colh0, self.temph0, self.abu_c,
                    self.logfile
                )
            else:
                self.xh, self.phi_ion = evolve3D(
                    dt, self.dr,
                    src_flux, src_pos,
                    self.gpu, self.max_subbox,self.subboxsize,self.loss_fraction,
                    self.temp, self.ndens, self.xh,
                    self.photo_thin_table, self.photo_thick_table,
                    self.minlogtau, self.dlogtau,
                    self.R_max_LLS, self.convergence_fraction,
                    self.sig, self.bh00, self.albpow, self.colh0, self.temph0, self.abu_c,
                    self.logfile
                    )
        else:
            self.xh, self.phi_ion = evolve3D(
                dt, self.dr,
                src_flux, src_pos,
                self.gpu, self.max_subbox,self.subboxsize,self.loss_fraction,
                self.temp, self.ndens, self.xh,
                self.photo_thin_table, self.photo_thick_table,
                self.minlogtau, self.dlogtau,
                self.R_max_LLS, self.convergence_fraction,
                self.sig, self.bh00, self.albpow, self.colh0, self.temph0, self.abu_c,
                self.logfile
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
        t_half = t_now + 0.5*dt
        t_after = t_now + dt

        # Increment redshift by half a time step
        z_half = self.time2zred(t_half)

        # Scale quantities if cosmological run
        if self.cosmological:
            # Scale density according to expansion
            dilution_factor = ( (1+z_half) / (1+self.zred) )**3
            self.ndens *= dilution_factor

            # Set cell size to current proper size
            self.dr = self.dr_c * self.cosmology.scale_factor(z_half)

        # Set new time and redshift (after timestep)
        self.zred = z_half
        self.time = t_after


    def printlog(self,s,quiet=False):
        """Print to log file and standard output

        Parameters
        ----------
        s : str
            String to print
        quiet : bool
            Whether to print only to log file or also to standard output (default)
        """
        if self.logfile is None:
            raise RuntimeError("Please set the log file in output_ini")
        else:
            printlog(s,self.logfile,quiet)


    def write_output(self,z):
        pass


    # =====================================================================================================
    # UTILITY METHODS
    # =====================================================================================================
    def time2zred(self,t):
        """Calculate the redshift corresponding to an age t in seconds
        """
        return z_at_value(self.cosmology.age, t*u.s).value

    def zred2time(self,z,unit='s'):
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
            self.dr,src_flux,src_pos,
            self.gpu,self.max_subbox,self.subboxsize,
            self.loss_fraction,self.ndens,self.xh,
            self.photo_thin_table,self.photo_thick_table,
            self.minlogtau,self.dlogtau,
            self.R_max_LLS,self.sig,self.logfile
        )
        self.phi_ion = gamma
        return gamma
    
    # =====================================================================================================
    # INITIALIZATION METHODS (PRIVATE)
    # =====================================================================================================

    def _param_init(self):
        """ Set up general constants and parameters

        Computes additional required quantities from the read-in parameters
        and stores them as attributes
        """
        self.eth0 = self._ld['CGS']['eth0']
        self.ethe0 = self._ld['CGS']['ethe0']
        self.ethe1 = self._ld['CGS']['ethe1']
        self.bh00 = self._ld['CGS']['bh00']
        self.fh0 = self._ld['CGS']['fh0']
        self.xih0 = self._ld['CGS']['xih0']
        self.albpow = self._ld['CGS']['albpow']
        self.abu_h = self._ld['Abundances']['abu_h']
        self.abu_he = self._ld['Abundances']['abu_he']
        self.mean_molecular = self.abu_h + 4.0*self.abu_he
        self.abu_c = self._ld['Abundances']['abu_c']
        self.colh0 = self._ld['CGS']['colh0_fact']*self.fh0*self.xih0/self.eth0**2
        self.temph0=self.eth0*ev2k
        self.sig = self._ld['Photo']['sigma_HI_at_ion_freq']
        self.loss_fraction = self._ld['Raytracing']['loss_fraction']
        self.convergence_fraction = self._ld['Raytracing']['convergence_fraction']
        self.max_subbox = self._ld['Raytracing']['max_subbox']
        self.subboxsize = self._ld['Raytracing']['subboxsize']

    def _cosmology_init(self):
        """ Set up cosmology from parameters (H0, Omega,..)
        """
        h = self._ld['Cosmology']['h']
        Om0 = self._ld['Cosmology']['Omega0']
        Ob0 = self._ld['Cosmology']['Omega_B']
        Tcmb0 = self._ld['Cosmology']['cmbtemp']
        H0 = 100*h
        self.cosmology = FlatLambdaCDM(H0, Om0, Tcmb0, Ob0=Ob0)

        self.cosmological = self._ld['Cosmology']['cosmological']
        self.zred_0 = self._ld['Cosmology']['zred_0']
        self.age_0 = self.zred2time(self.zred_0)

        # Scale quantities to the initial redshift
        if self.cosmological:
            if(self.rank == 0): self.printlog(f"Cosmology is on, scaling comoving quantities to the initial redshift, which is z0 = {self.zred_0:.3f}...")
            self.dr = self.cosmology.scale_factor(self.zred_0) * self.dr_c
        else:
            if(self.rank == 0): self.printlog("Cosmology is off.")

    def _radiation_init(self):
        """Set up radiation tables for ionization/heating rates
        """
        # Create optical depth table (log-spaced)
        self.minlogtau = self._ld['Photo']['minlogtau']
        self.maxlogtau = self._ld['Photo']['maxlogtau']
        self.NumTau = self._ld['Photo']['NumTau']
        self.SourceType = self._ld['Photo']['SourceType']
        self.grey = self._ld['Photo']['grey']
        self.compute_heating_rates = self._ld['Photo']['compute_heating_rates']

        if(self.rank == 0):
            if self.grey:
                self.printlog(f"Warning: Using grey opacity")
            else:
                self.printlog(f"Using power-law opacity with {self.NumTau:n} table points between tau=10^({self.minlogtau:n}) and tau=10^({self.maxlogtau:n})")
        
        # The actual table has NumTau + 1 points: the 0-th position is tau=0 and the
        # remaining NumTau points are log-spaced from minlogtau to maxlogtau (same as in C2Ray)
        self.tau, self.dlogtau = make_tau_table(self.minlogtau,self.maxlogtau,self.NumTau)

        ion_freq_HI = ev2fr * self.eth0
        ion_freq_HeII = ev2fr * self.ethe1

        # Black-Body source type
        if self.SourceType == 'blackbody':
            freq_min = ion_freq_HI
            freq_max = 10*ion_freq_HeII

            # Initialize spectrum parameters
            self.bb_Teff = self._ld['BlackBodySource']['Teff']
            self.cs_pl_idx_h = self._ld['BlackBodySource']['cross_section_pl_index']
            radsource = BlackBodySource(self.bb_Teff, self.grey, ion_freq_HI, self.cs_pl_idx_h)

            # Print info
            if(self.rank == 0):
                self.printlog(f"Using Black-Body sources with effective temperature T = {radsource.temp :.1e} K and Radius {(radsource.R_star/c.R_sun.to('cm')).value : .3e} rsun")
                self.printlog(f"Spectrum Frequency Range: {freq_min:.3e} to {freq_max:.3e} Hz")
                self.printlog(f"This is Energy:           {freq_min/ev2fr:.3e} to {freq_max/ev2fr:.3e} eV")

            # Integrate table
            self.printlog("Integrating photoionization rates tables...")
            self.photo_thin_table, self.photo_thick_table = radsource.make_photo_table(self.tau,freq_min,freq_max,1e48)
            
            # WIP: Heating rates
            # 30.11.23 P.Hirling: The heating tables can be calculated, and used
            # with the standalone CPU raytracing method to calculate photo-heating rates
            # for the whole grid. However, at this time, the chemistry solver doesn't
            # use these rates.
            # TODO:
            # 1. Add heating rate computation to ASORA (GPU raytracing)
            # 2. Add heating (thermal) to chemistry module
            if self.compute_heating_rates:
                self.printlog("Integrating photoheating rates tables...")
                self.heat_thin_table, self.heat_thick_table = radsource.make_heat_table(self.tau,freq_min,freq_max,1e48) # nb integration bounds are given in log10(freq/freq_HI)
            else:
                self.printlog("INFO: No heating rates")
                self.heat_thin_table = np.zeros(self.NumTau+1)
                self.heat_thick_table = np.zeros(self.NumTau+1)

        # Here you could add another source type, e.g. a monochromatic, power-law,...
        # Define a class in the radiation submodule and initialize it here
        else:
            raise NameError("Unknown source type : ",self.SourceType)
        
        # Copy radiation table to GPU
        if self.gpu:
            photo_table_to_device(self.photo_thin_table,self.photo_thick_table)
            if(self.rank == 0): self.printlog("Successfully copied radiation tables to GPU memory.")

    def _grid_init(self):
        """ Set up grid properties
        """
        # Comoving quantities
        self.boxsize_c = self._ld['Grid']['boxsize'] * Mpc
        self.dr_c = self.boxsize_c / self.N

        if(self.rank == 0):
            self.printlog(f"Welcome! Mesh size is N = {self.N:n}.")
            self.printlog(f"Simulation Box size (comoving Mpc): {self.boxsize_c/Mpc:.3e}")

        # Initialize cell size to comoving size (if cosmological run, it will be scaled in cosmology_init)
        self.dr = self.dr_c

        # Set R_max (LLS 3) in cell units
        self.R_max_LLS = self._ld['Photo']['R_max_cMpc'] * self.N / self._ld['Grid']['boxsize']
        self.printlog(f"Maximum comoving distance for photons from source (type 3 LLS): {self._ld['Photo']['R_max_cMpc'] : .3e} comoving Mpc")
        self.printlog(f"This corresponds to                                             {self.R_max_LLS : .3f} grid cells.")

    # The following initialization methods are simulation kind-dependent and need to be
    # overridden in the subclasses
    def _output_init(self):
        """ Set up output & log file
        """
        pass

    def _redshift_init(self):
        """Initialize time and redshift counter
        """
        pass

    def _material_init(self):
        """Initialize material properties of the grid
        """
        pass

    def _sources_init(self):
        """Initialize settings to read source files
        """
        pass

    # =====================================================================================================
    # OTHER PRIVATE METHODS
    # =====================================================================================================

    def _read_paramfile(self,paramfile):
        """ Read in YAML parameter file
        """
        loader = SafeLoader
        # Configure to read scientific notation as floats rather than strings
        loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
            [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.')
        )
        with open(paramfile,'r') as f:
            self._ld = yaml.load(f,loader)

    def _gpu_close(self):
        """ Deallocate GPU memory
        """
        device_close()