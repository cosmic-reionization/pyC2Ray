import yaml
import atexit
import re
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import tools21cm as t2c
import h5py, os
try: from yaml import CSafeLoader as SafeLoader
except ImportError: from yaml import SafeLoader

from .utils.logutils import printlog
from .evolve import evolve3D
from .asora_core import device_init, device_close, photo_table_to_device
from .radiation import BlackBodySource, make_tau_table
from .utils.other_utils import get_redshifts_from_output, find_bins

from .utils import get_source_redshifts
from .c2ray_base import C2Ray, YEAR, Mpc, msun2g, ev2fr, ev2k

from .source_model import *

__all__ = ['C2Ray_244_fstar']

# m_p = 1.672661e-24

# ======================================================================
# This file contains the C2Ray_CubeP3M subclass of C2Ray, which is a
# version used for simulations that read in N-Body data from CubeP3M
# ======================================================================

class C2Ray_244_fstar(C2Ray):
    def __init__(self,paramfile,Nmesh,use_gpu,use_mpi):
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
        dt = (t2-t1)/num_timesteps
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
        t_half = t_now + 0.5*dt
        t_after = t_now + dt
        self.printlog(' This is time : %f\t %f' %(t_now/YEAR, t_after/YEAR))

        # Increment redshift by half a time step
        z_half = self.time2zred(t_half)

        # Scale quantities if cosmological run
        if self.cosmological:
            # Scale density according to expansion
            dilution_factor = (1+z_half) / (1+self.zred)
            #dilution_factor = ( (1+z_half) / (1+self.zred) )**3
            self.ndens *= dilution_factor**3

            # Set cell size to current proper size
            # self.dr = self.dr_c * self.cosmology.scale_factor(z_half)
            self.dr /= dilution_factor
            self.printlog(f"zfactor = {1./dilution_factor : .10f}")
        # Set new time and redshift (after timestep)
        self.zred = z_half
        self.time = t_after

    def cosmo_evolve_to_now(self):
        """Evolve cosmology over a timestep
        """
        # Time step
        t_now = self.time

        # Increment redshift by half a time step
        z_now = self.time2zred(t_now)

        # Scale quantities if cosmological run
        if self.cosmological:
            # Scale density according to expansion
            dilution_factor = (1+z_now) / (1+self.zred)
            #dilution_factor = ( (1+z_half) / (1+self.zred) )**3
            self.ndens *= dilution_factor**3

            # Set cell size to current proper size
            # self.dr = self.dr_c * self.cosmology.scale_factor(z_half)
            self.dr /= dilution_factor
            self.printlog(f"zfactor = {1./dilution_factor : .10f}")
        # Set new time and redshift (after timestep)
        self.zred = z_now

    # =====================================================================================================
    # UTILITY METHODS
    # =====================================================================================================
    def time2zred(self,t):
        """Calculate the redshift corresponding to an age t in seconds
        """
        # TODO: it should be then z_at_value(self.cosmology.age, t*u.s).value
        # in C2Ray is defined: time2zred = -1+(1.+zred_t0)*(t0/(t0+time))**(2./3.)        
        #return -1+(1.+self.zred_0)*(self.age_0/(self.age_0+t))**(2./3.)
        return -1+(1.+self.zred_0)*(self.age_0/(t))**(2./3.)

    def zred2time(self, z, unit='s'):
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
        #return self.age_0*(((1.0+self.zred_0)/(1.0+z))**1.5 - 1.0) # C2Ray version, time is 0 at sim begin
        return self.age_0*(((1.0+self.zred_0)/(1.0+z))**1.5) # <- Here, we want time to be actual age (age0 + t)
        

    # =====================================================================================================
    # INITIALIZATION METHODS (PRIVATE)
    # =====================================================================================================

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

        #H0 *= 1e5/Mpc

        #self.age_0 = 2.*(1.+self.zred_0)**(-1.5)/(3.*H0*np.sqrt(Om0))
        #self.age_0 = self.zred2time(self.zred_0)
        self.age_0 = 2.*(1.+self.zred_0)**(-1.5)/(3.*H0 * 1e5/Mpc *np.sqrt(Om0))
        
        # TODO: here I force z0 to equal restart slice to test
        zred_0 = self.zred_0 #20.134

        # Scale quantities to the initial redshift
        if self.cosmological:
            self.printlog(f"Cosmology is on, scaling comoving quantities to the initial redshift, which is z0 = {zred_0:.3f}...")
            self.printlog(f"Cosmological parameters used:")
            self.printlog(f"h   = {h:.4f}, Tcmb0 = {Tcmb0:.3e}")
            self.printlog(f"Om0 = {Om0:.4f}, Ob0   = {Ob0:.4f}")
            #TODO: it should be: self.dr = self.cosmology.scale_factor(self.zred_0) * self.dr_c
            self.dr = self.dr_c / (1 + zred_0)
        else:
            self.printlog("Cosmology is off.")


    # =====================================================================================================
    # USER DEFINED METHODS
    # =====================================================================================================

    def fstar_model(self, mhalo, kind='mass_independent'):
        if kind.lower() in ['fgamma', 'f_gamma', 'mass_independent']: 
            fstar = (self.cosmology.Ob0/self.cosmology.Om0)
        return fstar
      
    def read_sources(self, file, mass, ts, kind='fgamma'): # >:( trgeoip
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
            Grid positions of the sources formatted in a suitable way for the chosen raytracing algorithm
        normflux : array
            Normalization of the flux of each source (relative to S_star)
        numsrc : int
            Number of sources read from the file
        """
        S_star_ref = 1e48
        
        # TODO: automatic selection of low mass or high mass. For the moment only high mass
        #mass2phot = msun2g * self.fgamma_hm * self.cosmology.Ob0 / (self.mean_molecular * c.m_p.cgs.value * self.ts * self.cosmology.Om0)    
        # TODO: for some reason the difference with the orginal Fortran run is of the molecular weight
        #self.printlog('%f' %self.mean_molecular )
        
        if file.endswith('.hdf5'):
            f = h5py.File(file, 'r')
            srcpos = f['sources_positions'][:].T
            assert srcpos.shape[0] == 3
            mhalo = f['sources_mass'][:] 
            f.close()
        else:
            # use original C2Ray source file
            src = t2c.SourceFile(filename=file, mass=mass)
            srcpos = src.sources_list[:, :3].T
            mhalo = src.sources_list[:, -1] 

        fstar = self.fstar_model(mhalo, kind=kind)
        mstar = mhalo * fstar

        f_gamma = self.fgamma_hm 
        mass2phot = msun2g / (m_p * ts)  * f_gamma
        normflux = mstar * mass2phot / S_star_ref

        self.printlog('\n---- Reading source file with total of %d ionizing source:\n%s' %(normflux.size, file))
        self.printlog(' Total Flux : %e' %np.sum(normflux*S_star_ref))
        self.printlog(' Source lifetime : %f Myr' %(ts/(1e6*YEAR)))
        self.printlog(' min, max source mass : %.3e  %.3e [Msun] and min, mean, max number of ionising sources : %.3e  %.3e  %.3e [1/s]' %(normflux.min()/mass2phot*S_star_ref, normflux.max()/mass2phot*S_star_ref, normflux.min()*S_star_ref, normflux.mean()*S_star_ref, normflux.max()*S_star_ref))
        return srcpos, normflux
    
    def ionizing_flux(self, file, ts, z, box_len, n_grid, kind='fgamma', save_Mstar=False): # >:( trgeoip
        """Read sources from a C2Ray-formatted file

        Parameters
        ----------
        file : str
            Filename to read.
        ts : float
            time-step in Myrs.
        box_len : float
            Simulation box length in Mpc/h.
        n_grid : int
            Number of cells/grids along each simulation volume.
        kind: str
            The kind of source model to use.
        
        Returns
        -------
        srcpos : array
            Grid positions of the sources formatted in a suitable way for the chosen raytracing algorithm
        normflux : array
            Normalization of the flux of each source (relative to S_star)
        """
        srcpos_mpc, srcmass_msun = self.read_haloes(file, box_len)
        fstar = self.fstar_model(srcmass_msun, kind=kind)
        mstar_msun = fstar*srcmass_msun

        h = 0.7 
        hg = Halo2Grid(box_len=box_len, n_grid=n_grid)
        hg.set_halo_pos(srcpos_mpc, unit='mpc')
        hg.set_halo_mass(mstar_msun, unit='Msun')

        binned_mstar, bin_edges, bin_num = hg.value_on_grid(hg.pos_grid, mstar_msun)
        srcpos, srcmstar = hg.halo_value_on_grid(mstar_msun, binned_value=binned_mstar)

        if save_Mstar:
            folder_path = save_Mstar
            fname_hdf5 = folder_path+f'/{z:.3f}-Mstar_sources.hdf5'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
                print(f"Folder '{folder_path}' created successfully.")
            else:
                pass
                # print(f"Folder '{folder_path}' already exists.")

            # Create HDF5 file from the data
            with h5py.File(fname_hdf5,"w") as f:
                # Store Data
                dset_pos = f.create_dataset("sources_positions", data=srcpos)
                dset_mass = f.create_dataset("sources_mass", data=srcmstar)

                # Store Metadata
                f.attrs['z'] = z
                f.attrs['h'] = 0.7
                f.attrs['numhalo'] = srcmstar.shape[0]
                f.attrs['units'] = 'cMpc   Msun'

        S_star_ref = 1e48
        f_gamma = self.fgamma_hm 
        mass2phot = msun2g / (m_p * ts)  * f_gamma
        normflux = srcmstar * mass2phot / S_star_ref

        self.printlog('\n---- Reading source file with total of %d ionizing source:\n%s' %(normflux.size, file))
        self.printlog(' Total Flux : %e' %np.sum(normflux*S_star_ref))
        self.printlog(' Source lifetime : %f Myr' %(ts/(1e6*YEAR)))
        self.printlog(' min, max source mass : %.3e  %.3e [Msun] and min, mean, max number of ionising sources : %.3e  %.3e  %.3e [1/s]' %(normflux.min()/mass2phot*S_star_ref, normflux.max()/mass2phot*S_star_ref, normflux.min()*S_star_ref, normflux.mean()*S_star_ref, normflux.max()*S_star_ref))
        return srcpos, normflux

    def read_haloes(self, halo_file, box_len): # >:( trgeoip
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

        print(f'Reading {halo_file}')
        try:
            f = h5py.File(halo_file)
            h = f.attrs['h']
            srcmass_msun = f['mass'][:]/h #Msun
            srcpos_mpc = f['pos'][:]/h   #Mpc
            f.close()
        except:
            # Read haloes from a CUBEP3M file format.
            hl = t2c.HaloCubeP3MFull(filename=halo_file, box_len=box_len)
            h  = self.h
            srcmass_msun = hl.get(var='m')/h   #Msun
            srcpos_mpc  = hl.get(var='pos')/h #Mpc
            print(f'Mass and positions are converted to Msun and Mpc respectively assuming h={h}.')
        print(f'...done')
        return srcpos_mpc, srcmass_msun
    

    
    def read_density(self, z):
        """ Read coarser density field from C2Ray-formatted file

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

        # redshift bin for the current redshift based on the density redshift
        #low_z, high_z = find_bins(redshift, self.zred_density)
        high_z = self.zred_density[np.argmin(np.abs(self.zred_density[self.zred_density >= redshift] - redshift))]

        if(high_z != self.prev_zdens):
            file = '%scoarser_densities/%.3fn_all.dat' %(self.inputs_basename, high_z)
            self.printlog(f'\n---- Reading density file:\n '+file)
            self.ndens = t2c.DensityFile(filename=file).cgs_density / (self.mean_molecular * m_p) * (1+redshift)**3
            self.printlog(' min, mean and max density : %.3e  %.3e  %.3e [1/cm3]' %(self.ndens.min(), self.ndens.mean(), self.ndens.max()))
            self.prev_zdens = high_z
        else:
            # no need to re-read the same file again
            # TODO: in the future use this values for a 3D interpolation for the density (can be extended to sources too)
            pass

    def write_output(self,z):
        """Write ionization fraction & ionization rates as C2Ray binary files

        Parameters
        ----------
        z : float
            Redshift (used to name the file)
        """
        suffix = f"_{z:.3f}.dat"
        t2c.save_cbin(filename=self.results_basename + "xfrac" + suffix, data=self.xh, bits=64, order='F')
        t2c.save_cbin(filename=self.results_basename + "IonRates" + suffix, data=self.phi_ion, bits=32, order='F')

        self.printlog('\n--- Reionization History ----')
        self.printlog(' min, mean, max xHII : %.5e  %.5e  %.5e' %(self.xh.min(), self.xh.mean(), self.xh.max()))
        self.printlog(' min, mean, max Irate : %.5e  %.5e  %.5e [1/s]' %(self.phi_ion.min(), self.phi_ion.mean(), self.phi_ion.max()))
        self.printlog(' min, mean, max density : %.5e  %.5e  %.5e [1/cm3]' %(self.ndens.min(), self.ndens.mean(), self.ndens.max()))

    
    # =====================================================================================================
    # Below are the overridden initialization routines specific to the CubeP3M case
    # =====================================================================================================

    def _redshift_init(self):
        """Initialize time and redshift counter
        """
        self.zred_density = t2c.get_dens_redshifts(self.inputs_basename+'coarser_densities/')[::-1]
        #self.zred_sources = get_source_redshifts(self.inputs_basename+'sources/')[::-1]
        # TODO: waiting for next tools21cm release
        self.zred_sources = get_source_redshifts(self.inputs_basename+'sources/')[::-1]
        if(self.resume):
            # get the resuming redshift
            self.zred = np.min(get_redshifts_from_output(self.results_basename)) 
            #self.age_0 = self.zred2time(self.zred_0)
            _, self.prev_zdens = find_bins(self.zred, self.zred_density)
            _, self.prev_zsourc = find_bins(self.zred, self.zred_sources)
        else:
            self.prev_zdens = -1
            self.prev_zsourc = -1
            self.zred = self.zred_0

        self.time = self.zred2time(self.zred)
        #self.time = self.age_0

    def _material_init(self):
        """Initialize material properties of the grid
        """
        if(self.resume):
            # get fields at the resuming redshift
            self.ndens = t2c.DensityFile(filename='%scoarser_densities/%.3fn_all.dat' %(self.inputs_basename, self.prev_zdens)).cgs_density / (self.mean_molecular * m_p) * (1+self.zred)**3
            #self.ndens = self.read_density(z=self.zred)
            self.xh = t2c.read_cbin(filename='%sxfrac_%.3f.dat' %(self.results_basename, self.zred), bits=64, order='F')
            # TODO: implement heating
            temp0 = self._ld['Material']['temp0']
            self.temp = temp0 * np.ones(self.shape, order='F')
            self.phi_ion = t2c.read_cbin(filename='%sIonRates_%.3f.dat' %(self.results_basename, self.zred), bits=32, order='F')
        else:
            xh0 = self._ld['Material']['xh0']
            temp0 = self._ld['Material']['temp0']
            avg_dens = self._ld['Material']['avg_dens']

            self.ndens = avg_dens * np.empty(self.shape, order='F')
            self.xh = xh0 * np.ones(self.shape, order='F')
            self.temp = temp0 * np.ones(self.shape, order='F')
            self.phi_ion = np.zeros(self.shape, order='F')
    
    def _output_init(self):
        """ Set up output & log file
        """
        self.results_basename = self._ld['Output']['results_basename']
        self.inputs_basename = self._ld['Output']['inputs_basename']

        self.logfile = self.results_basename + self._ld['Output']['logfile']
        title = '                 _________   ____            \n    ____  __  __/ ____/__ \ / __ \____ ___  __\n   / __ \/ / / / /    __/ // /_/ / __ `/ / / /\n  / /_/ / /_/ / /___ / __// _, _/ /_/ / /_/ / \n / .___/\__, /\____//____/_/ |_|\__,_/\__, /  \n/_/    /____/                        /____/   \n'
        if(self._ld['Grid']['resume']):
            with open(self.logfile,"r") as f: 
                log = f.readlines()
            with open(self.logfile,"w") as f: 
                log.append("\n\nResuming"+title[8:]+"\n\n")
                f.write(''.join(log))
        else:
            with open(self.logfile,"w") as f: 
                # Clear file and write header line
                f.write(title+"\nLog file for pyC2Ray.\n\n") 

    def _sources_init(self):
        """Initialize settings to read source files
        """
        self.fgamma_hm = self._ld['Sources']['fgamma_hm']
        self.fgamma_lm = self._ld['Sources']['fgamma_lm']
        self.ts = self._ld['Sources']['ts'] * YEAR * 1e6
        self.printlog(f"Using UV model with fgamma_lm = {self.fgamma_lm:.1f} and fgamma_hm = {self.fgamma_hm:.1f}")

    def _grid_init(self):
        """ Set up grid properties
        """
        # Comoving quantities
        self.boxsize_c = self._ld['Grid']['boxsize'] * Mpc / self._ld['Cosmology']['h']
        self.dr_c = self.boxsize_c / self.N

        self.printlog(f"Welcome! Mesh size is N = {self.N:n}.")
        self.printlog(f"Simulation Box size (comoving Mpc): {self.boxsize_c/Mpc:.3e}")

        # Initialize cell size to comoving size (if cosmological run, it will be scaled in cosmology_init)
        self.dr = self.dr_c

        # Set R_max (LLS 3) in cell units
        self.R_max_LLS = self._ld['Photo']['R_max_cMpc'] * self.N * self._ld['Cosmology']['h']/ self._ld['Grid']['boxsize']
        self.printlog(f"Maximum comoving distance for photons from source (type 3 LLS): {self._ld['Photo']['R_max_cMpc'] : .3e} comoving Mpc")
        self.printlog(f"This corresponds to {self.R_max_LLS : .3f} grid cells.")

        self.resume = self._ld['Grid']['resume']