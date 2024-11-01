import numpy as np, pandas as pd
from astropy.cosmology import FlatLambdaCDM
from glob import glob
import astropy.constants as cst
import tools21cm as t2c
import h5py, os
try: from yaml import CSafeLoader as SafeLoader
except ImportError: from yaml import SafeLoader

from .utils.logutils import printlog
from .evolve import evolve3D
from .asora_core import device_init, device_close, photo_table_to_device
from .radiation import BlackBodySource, make_tau_table
from .utils.other_utils import get_extension_in_folder, get_redshifts_from_output, find_bins

from .utils import get_source_redshifts
from .c2ray_base import C2Ray, YEAR, Mpc, msun2g, ev2fr, ev2k

from .source_model import StellarToHaloRelation, BurstySFR, EscapeFraction, Halo2Grid, m_p
from scipy.stats import binned_statistic_dd

__all__ = ['C2Ray_fstar']

# m_p = 1.672661e-24

# ======================================================================
# This file contains the C2Ray_CubeP3M subclass of C2Ray, which is a
# version used for simulations that read in N-Body data from CubeP3M
# ======================================================================

class C2Ray_fstar(C2Ray):
    def __init__(self,paramfile):
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
        if(self.rank == 0):
            self.printlog('Running: "C2Ray for %d Mpc/h volume"' %self.boxsize)

    # =====================================================================================================
    # USER DEFINED METHODS
    # =====================================================================================================

    def ionizing_flux(self, file, z, dt=None, save_Mstar=False): # >:( trgeoip
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
        # read halo list       
        srcpos_mpc, srcmass_msun = self.read_haloes(self.sources_basename+file, self.boxsize)

        # source life-time in cgs
        if(self.acc_kind == 'EXP'):
            #ts = 1. / (self.alph_h * (1+z) * self.cosmology.H(z=z).cgs.value)
            ts = self.fstar_model.source_liftime(z=z)
        elif(self.acc_kind == 'constant'):
            ts = dt

        # get stellar-to-halo ratio
        fstar = self.fstar_model.get(Mhalo=srcmass_msun)

        # get escaping fraction
        if(self.fesc_kind == 'constant'):
            fesc = self.fesc_model.f0_esc
        elif(self.fesc_kind == 'power'):
            fesc = self.fesc_model.get(Mhalo=srcmass_msun)
        elif(self.fesc_kind == 'Muv'):
            Muv = self.fstar_model.UV_magnitude(fstar=fstar, mdot=srcmass_msun/ts)
            fesc = self.fesc_model.get(Muv=Muv, Mhalo=srcmass_msun)
            
        # get for star formation history 
        if(self.bursty_kind == 'instant' or self.bursty_kind == 'integrate'):
            burst_mask = self.bursty_model.get_bursty(mass=srcmass_msun, z=z)

            nr_switchon = np.count_nonzero(burst_mask)
            self.perc_switchon = 100*nr_switchon/burst_mask.size

            self.printlog(' A total of %.2f %% of galaxies (%d out of %d) have bursty star-formation.' %(self.perc_switchon, nr_switchon, burst_mask.size))

            # mask the sources that are switched off
            srcpos_mpc, srcmass_msun = srcpos_mpc[burst_mask], srcmass_msun[burst_mask]
            if(self.fesc_kind == 'constant'):
                fstar = fstar[burst_mask]
            else:
                fstar, fesc = fstar[burst_mask], fesc[burst_mask]

        # get stellar mass
        mstar_msun = fesc*fstar*srcmass_msun
        
        # sum together masses into a mesh grid
        if(nr_switchon > 0):
            mesh_bin = np.linspace(0, self.boxsize/self.cosmology.h, self.N+1)
            binned_mass, bin_edges, bin_num = binned_statistic_dd(srcpos_mpc, mstar_msun, statistic='sum', bins=[mesh_bin, mesh_bin, mesh_bin])        
            
            # get a list of the source positon and mass
            srcpos = np.argwhere(binned_mass>0) 
            srcmstar = binned_mass[binned_mass>0] 

            """
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
                    f.attrs['h'] = self.cosmology.h
                    f.attrs['numhalo'] = srcmstar.shape[0]
                    f.attrs['units'] = 'cMpc   Msun'
            """
            S_star_ref = 1e48

            # normalize flux
            normflux = msun2g * self.fstar_pars['Nion'] * srcmstar / (m_p * ts * S_star_ref)

            # calculate total number of ionizing photons
            self.tot_phots = np.sum(normflux * dt * S_star_ref)

            if(self.rank == 0):
                self.printlog('\n---- Reading source file with total of %d ionizing source:\n%s' %(normflux.size, file))
                self.printlog(' Total Flux : %e [1/s]' %np.sum(normflux*S_star_ref))
                self.printlog(' Total number of ionizaing photons : %e' %self.tot_phots)
                self.printlog(' Source lifetime : %f Myr' %(ts/(1e6*YEAR)))
                self.printlog(' min, max stellar (grid) mass : %.3e  %.3e [Msun] and min, mean, max number of ionising sources : %.3e  %.3e  %.3e [1/s]' %(srcmstar.min(), srcmstar.max(), normflux.min()*S_star_ref, normflux.mean()*S_star_ref, normflux.max()*S_star_ref))
            
            return srcpos, normflux
        
        else:
            if(self.rank == 0):
                self.printlog('\n---- Reading source file with total of %d ionizing source:\n%s' %(srcmass_msun.size, file))
                self.printlog(' No sources switch on. Skip computing the raytracing.')
            
            return 0, 0
    
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

        if(halo_file.endswith('.hdf5')):
            # Read haloes from a CUBEP3M file format converted in hdf5.
            f = h5py.File(halo_file)
            h = f.attrs['h']
            srcmass_msun = f['mass'][:]/h #Msun
            srcpos_mpc = f['pos'][:]/h   #Mpc
            f.close()
        elif(halo_file.endswith('.dat')):
            # Read haloes from a CUBEP3M file format.
            hl = t2c.HaloCubeP3MFull(filename=halo_file, box_len=box_len)
            h  = self.h
            srcmass_msun = hl.get(var='m')/h   #Msun
            srcpos_mpc  = hl.get(var='pos')/h #Mpc
        elif(halo_file.endswith('.txt')):
            # Read haloes from a PKDGrav converted in txt.
            hl = np.loadtxt(halo_file)
            srcmass_msun = hl[:,0]/self.cosmology.h # Msun
            srcpos_mpc = (hl[:,1:]+self.boxsize/2) # Mpc/h

            # apply periodic boundary condition shift
            srcpos_mpc[srcpos_mpc > self.boxsize] = self.boxsize - srcpos_mpc[srcpos_mpc > self.boxsize]
            srcpos_mpc[srcpos_mpc < 0.] = self.boxsize + srcpos_mpc[srcpos_mpc < 0.]
            srcpos_mpc /= self.cosmology.h # Mpc
        return srcpos_mpc, srcmass_msun
        
    def read_density(self, fbase, z=None):
        """ Read coarser density field from C2Ray-formatted file

        This method is meant for reading density field run with either N-body or hydro-dynamical simulations. The field is then smoothed on a coarse mesh grid.

        Parameters
        ----------
        fbase : string
            the file name (cwithout the path) of the file to open
        
        """
        file = self.density_basename+fbase
        rdr = t2c.Pkdgrav3data(self.boxsize, self.N, Omega_m=self.cosmology.Om0)
        self.ndens = self.cosmology.critical_density0.cgs.value * self.cosmology.Ob0 * (1.+rdr.load_density_field(file)) / (self.mean_molecular * m_p) * (1+z)**3
        if(self.rank == 0):
            self.printlog('\n---- Reading density file:\n  %s' %file)
            self.printlog(' min, mean and max density : %.3e  %.3e  %.3e [1/cm3]' %(self.ndens.min(), self.ndens.mean(), self.ndens.max()))

    # =====================================================================================================
    # Below are the overridden initialization routines specific to the f_star case
    # =====================================================================================================

    def _redshift_init(self):
        """Initialize time and redshift counter
        """
        self.zred_density = np.loadtxt(self.density_basename+'redshift_density.txt')
        self.zred_sources = np.loadtxt(self.sources_basename+'redshift_sources.txt')
        if(self.resume):
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
        """Initialize material properties of the grid
        """
        if(self.resume):
            # get fields at the resuming redshift
            self.ndens = self.read_density(fbase='CDM_200Mpc_2048.%05d.den.256.0' %self.resume, z=self.prev_zdens)
            
            # get extension of the output file
            ext = get_extension_in_folder(path=self.results_basename)
            if(ext == '.dat'):
                fname = '%sxfrac_%.3f.dat' %(self.results_basename, self.zred)
                self.xh = t2c.read_cbin(filename=fname, bits=64, order='F')
                self.phi_ion = t2c.read_cbin(filename='%sIonRates_%.3f.dat' %(self.results_basename, self.zred), bits=32, order='F')
            elif(ext == '.npy'):
                fname = '%sxfrac_%.3f.npy' %(self.results_basename, self.zred)
                self.xh = np.load(fname)
                self.phi_ion = np.load('%sIonRates_%.3f.npy' %(self.results_basename, self.zred))
            else:
                raise FileNotFoundError(' Resume file not found: %sxfrac_%.3f.npy' %(self.results_basename, self.zred))
            
            if(self.rank == 0):
                self.printlog('\n---- Reading ionized fraction field:\n %s' %fname)
                self.printlog(' min, mean and max density : %.5e  %.5e  %.5e' %(self.xh.min(), self.xh.mean(), self.xh.max()))

            # TODO: implement heating
            temp0 = self._ld['Material']['temp0']
            self.temp = temp0 * np.ones(self.shape, order='F')
        else:
            xh0 = self._ld['Material']['xh0']
            temp0 = self._ld['Material']['temp0']
            avg_dens = self._ld['Material']['avg_dens']

            self.ndens = avg_dens * np.empty(self.shape, order='F')
            self.xh = xh0 * np.ones(self.shape, order='F')
            self.temp = temp0 * np.ones(self.shape, order='F')
            self.phi_ion = np.zeros(self.shape, order='F')

    def _sources_init(self):
        """Initialize settings to read source files
        """
        # --- Stellar-to-Halo Source model ---
        self.fstar_kind = self._ld['Sources']['fstar_kind']

        # dictionary with all the f_star parameters
        self.fstar_pars = {'Nion': self._ld['Sources']['Nion'], 'f0': self._ld['Sources']['f0'], 'Mt': self._ld['Sources']['Mt'], 'Mp': self._ld['Sources']['Mp'], 'g1': self._ld['Sources']['g1'], 'g2': self._ld['Sources']['g2'], 'g3': self._ld['Sources']['g3'], 'g4': self._ld['Sources']['g4'], 'alpha_h': self._ld['Sources']['alpha_h']}

        # print message that inform of the f_star model employed
        if(self.fstar_kind == 'fgamma' and self.rank == 0):
            self.printlog(f"Using constant stellar-to-alo relation model with fgamma_hm = {self.fstar_pars['f0']:.1f}, Nion = {self.fstar_pars['Nion']:.1f}")
        elif(self.rank == 0 and (self.fstar_kind == 'dpl' or self.fstar_kind == 'lognorm')):
            self.printlog(f"Using {self.fstar_kind} to model the stellar-to-halo relation with parameters: {self.fstar_pars}.")

        # define the f_star model class (to call self.fstar_model.get_fstar(Mhalo) when reading the sources)
        self.fstar_model = StellarToHaloRelation(model=self.fstar_kind, pars=self.fstar_pars, cosmo=self.cosmology)

        # --- Halo Accretion Model --- 
        # TODO: Create class etc...
        self.acc_kind = self._ld['Sources']['accretion_model']
        self.alph_h = self.fstar_pars['alpha_h']

        # --- Burstiness Model for Star Formation ---
        self.bursty_kind = self._ld['Sources']['bursty_sfr']

        # dictionary with all the burstiness parameters
        if(self.bursty_kind == 'instant' or self.bursty_kind == 'integrate'):
            self.bursty_pars = {'beta1': self._ld['Sources']['beta1'], 'beta2': self._ld['Sources']['beta2'], 'tB0': self._ld['Sources']['tB0'], 'tQ_frac': self._ld['Sources']['tQ_frac'], 'z0': self._ld['Sources']['z0'], 't_rnd': self._ld['Sources']['t_rnd']}

            if(self.rank == 0):
                self.printlog(f"Using {self.bursty_kind} bustiness to model the star formation history with parameters: {self.bursty_pars}.")
            
        # define the burstiness SF model class
        self.bursty_model = BurstySFR(model=self.bursty_kind, pars=self.bursty_pars, alpha_h=self.alph_h, cosmo=self.cosmology)

        # --- Escaping fraction Model ---
        self.fesc_kind = self._ld['Sources']['fesc_model']
        self.fesc_pars = {'f0_esc': self._ld['Sources']['f0_esc'], 'Mp_esc': self._ld['Sources']['Mp_esc'], 'al_esc': self._ld['Sources']['al_esc']}
        if(self.rank == 0):
            if(self.fesc_kind == 'constant'):
                self.printlog(f"Using constant escaping fraction model model with f0_esc = %.1f" %(self.fesc_pars['f0_esc']))
            elif(self.fesc_kind == 'power'):
                self.printlog(f"Using mass-dependent power law model for the escaping fraction with parameters: {self.fesc_pars}")
            elif(self.fesc_kind == 'Gelli2024'):
                self.printlog(f"Using UV magnitude-dependent power law model for the escaping fraction with parameters: {self.fesc_pars}")

        self.fesc_model = EscapeFraction(model=self.fesc_kind, pars=self.fesc_pars)
