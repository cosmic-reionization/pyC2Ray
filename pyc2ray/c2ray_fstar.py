import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as cst
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

__all__ = ['C2Ray_fstar']

# m_p = 1.672661e-24

# ======================================================================
# This file contains the C2Ray_CubeP3M subclass of C2Ray, which is a
# version used for simulations that read in N-Body data from CubeP3M
# ======================================================================

class C2Ray_fstar(C2Ray):
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
        self.printlog('Running: "C2Ray for %d Mpc/h volume"' %self.boxsize)

    # =====================================================================================================
    # USER DEFINED METHODS
    # =====================================================================================================

    def fstar_model(self, mhalo):
        kind = self.fstar_kind
        if kind.lower() in ['fgamma', 'f_gamma', 'mass_independent']: 
            fstar = (self.cosmology.Ob0/self.cosmology.Om0)
        elif kind.lower() in ['dpl', 'mass_dependent']:
            model = StellarToHaloRelation(
                        f0=self.fstar_dpl['f0'], 
                        Mt=self.fstar_dpl['Mt'], 
                        Mp=self.fstar_dpl['Mp'],
                        g1=self.fstar_dpl['g1'], 
                        g2=self.fstar_dpl['g2'], 
                        g3=self.fstar_dpl['g3'], 
                        g4=self.fstar_dpl['g4'])
            star = model.deterministic(mhalo)
            fstar = star['fstar']
        else:
            print(f'{kind} fstar model is not implemented.')
        return fstar

    def ionizing_flux(self, file, ts, z, save_Mstar=False): # >:( trgeoip
        """Read sources from a C2Ray-formatted file

        Parameters
        ----------
        file : str
            Filename to read.
        ts : float
            time-step in Myrs.
        kind: str
            The kind of source model to use.
        
        Returns
        -------
        srcpos : array
            Grid positions of the sources formatted in a suitable way for the chosen raytracing algorithm
        normflux : array
            Normalization of the flux of each source (relative to S_star)
        """
        box_len, n_grid = self.boxsize, self.N
        
        srcpos_mpc, srcmass_msun = self.read_haloes(self.sources_basename+file, box_len)

        h = self.cosmology.h
        hg = Halo2Grid(box_len=box_len/h, n_grid=n_grid)
        hg.set_halo_pos(srcpos_mpc, unit='mpc')
        hg.set_halo_mass(srcmass_msun, unit='Msun')

        #TODO: NEED TO IMPLEMENT THE HALO MODEL BEFORE GRIDDING
        S_star_ref = 1e48
        if(self.acc_model == 'constant'):
            # locate LMACHs and HMACHs
            low_mask = srcmass_msun <= 1e8
            high_mask = srcmass_msun > 1e8

            # consider low mass and high mass halos efficiency            
            mhalo2phot = srcmass_msun * high_mask * self.fgamma_hm 
            mhalo2phot += srcmass_msun * low_mask * self.fgamma_lm 

            # convert from mass to number of photons
            mhalo2phot *= msun2g * self.cosmology.Ob0 / (self.cosmology.Om0 * m_p * ts * S_star_ref) 
        elif(self.acc_model == 'Schneider21'):
            # define star efficiency
            fstar = self.fstar_model(srcmass_msun) # Ob0/Om0 is already in here
            
            # define lifetime
            ts = 1. / (self.alph_h * (1+z) * self.cosmology.H(z=z).cgs.value)

            # convert from mass to number of photons
            mhalo2phot = self.fgamma_hm * fstar * srcmass_msun / (m_p * ts * S_star_ref) 

        binned_mstar, bin_edges, bin_num = hg.value_on_grid(hg.pos_grid, mhalo2phot)
        srcpos, normflux = hg.halo_value_on_grid(mhalo2phot, binned_value=binned_mstar)

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
                dset_mass = f.create_dataset("sources_mass", data=normflux)

                # Store Metadata
                f.attrs['z'] = z
                f.attrs['h'] = self.cosmology.h
                f.attrs['numhalo'] = normflux.shape[0]
                f.attrs['units'] = 'cMpc   Msun'


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
        
    def read_density(self, fbase='%.3fn_all.dat', z=None):
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

        idx_high_z = np.argmin(np.abs(self.zred_density[self.zred_density >= redshift] - redshift))
        high_z = self.zred_density[idx_high_z]

        # condition if need to read new file or use the one from the previous time-step
        if(high_z != self.prev_zdens):
            if(fbase.endswith('.dat')):
                # get file name
                file = self.density_basename+fbase %high_z

                # use tools21cm to read density file and get baryonic number density
                self.ndens = t2c.DensityFile(filename=file).cgs_density / (self.mean_molecular * m_p) * (1+redshift)**3
            elif(fbase.endswith('.0')):
                # get file name
                file = self.density_basename+fbase %(idx_high_z+1)
                
                rdr = t2c.Pkdgrav3data(self.boxsize, self.N, Omega_m=self.cosmology.Om0)
                self.ndens = self.cosmology.critical_density0.cgs.value * self.cosmology.Ob0 * (1.+rdr.load_density_field(file)) / (self.mean_molecular * m_p) * (1+redshift)**3
            
            self.printlog('\n---- Reading density file:\n  %s' %file)
            self.printlog(' min, mean and max density : %.3e  %.3e  %.3e [1/cm3]' %(self.ndens.min(), self.ndens.mean(), self.ndens.max()))
            self.prev_zdens = high_z
        else:
            # no need to re-read the same file again
            pass

    def write_output(self, z):
        """Write ionization fraction & ionization rates as C2Ray binary files

        Parameters
        ----------
        z : float
            Redshift (used to name the file)
        """
        suffix = f"_{z:.3f}.dat"
        t2c.save_cbin(filename=self.results_basename + "xfrac" + suffix, data=self.xh, bits=64, order='F')
        t2c.save_cbin(filename=self.results_basename + "IonRates" + suffix, data=self.phi_ion, bits=32, order='F')

        # print min, max and average quantities
        self.printlog('\n--- Reionization History ----')
        self.printlog(' min, mean, max xHII : %.5e  %.5e  %.5e' %(self.xh.min(), self.xh.mean(), self.xh.max()))
        self.printlog(' min, mean, max Irate : %.5e  %.5e  %.5e [1/s]' %(self.phi_ion.min(), self.phi_ion.mean(), self.phi_ion.max()))
        self.printlog(' min, mean, max density : %.5e  %.5e  %.5e [1/cm3]' %(self.ndens.min(), self.ndens.mean(), self.ndens.max()))

        # write summary output file
        summary_exist = os.path.exists(self.results_basename+'PhotonCounts2.txt')

        with open(self.results_basename+'PhotonCounts2.txt', 'a') as f:
            if not (summary_exist):
                header = '# z\ttot N_ions\ttot Irate [1/s]\tR_mfp [cMpc]\tmean ionization fraction (by volume and mass)\n'
                f.write(header)                

            tot_ions = np.sum(self.ndens*self.xh) * (self.boxsize*Mpc)**3
            massavrg_ion_frac = np.sum(self.xh*self.ndens)/np.sum(self.ndens)

            text = '%.3f\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n' %(z, tot_ions, np.sum(self.phi_ion), self.R_max_LLS/self.N*self.boxsize, np.mean(self.xh), massavrg_ion_frac)
            f.write(text)

    # =====================================================================================================
    # Below are the overridden initialization routines specific to the CubeP3M case
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
            #TODO: generalise the resuming of the simulation
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

    def _sources_init(self):
        """Initialize settings to read source files
        """
        self.fgamma_hm = self._ld['Sources']['fgamma_hm']
        self.fgamma_lm = self._ld['Sources']['fgamma_lm']
        self.ts = self._ld['Sources']['ts'] * YEAR * 1e6
        self.printlog(f"Using UV model with fgamma_lm = {self.fgamma_lm:.1f} and fgamma_hm = {self.fgamma_hm:.1f}")
        self.fstar_kind = self._ld['Sources']['fstar_kind']
        self.fstar_dpl = {
                        'f0': self._ld['Sources']['f0'],
                        'Mt': self._ld['Sources']['Mt'], 
                        'Mp': self._ld['Sources']['Mp'], 
                        'g1': self._ld['Sources']['g1'], 
                        'g2': self._ld['Sources']['g2'], 
                        'g3': self._ld['Sources']['g3'], 
                        'g4': self._ld['Sources']['g4'], 
                        }
        self.printlog(f"Using {self.fstar_kind} to model the stellar-to-halo relation, and the parameter dictionary = {self.fstar_dpl}.")

        self.acc_model = self._ld['Sources']['accreation_model']
        self.alph_h = self._ld['Sources']['alpha_h']