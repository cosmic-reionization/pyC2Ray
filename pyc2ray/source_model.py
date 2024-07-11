import numpy as np 
from scipy.spatial import cKDTree
from scipy.stats import binned_statistic_dd
import h5py
import tools21cm as t2c

from .c2ray_base import YEAR, Mpc, msun2g, ev2fr, ev2k

# Conversion Factors.
# When doing direct comparisons with C2Ray, the difference between astropy.constants and the C2Ray values
# may be visible, thus we use the same exact value for the constants. This can be changed to the
# astropy values once consistency between the two codes has been established
# pc = 3.086e18           #(1*u.pc).to('cm').value            # C2Ray value: 3.086e18
# YEAR = 3.15576E+07      #(1*u.yr).to('s').value           # C2Ray value: 3.15576E+07
# ev2fr = 0.241838e15                     # eV to Frequency (Hz)
# ev2k = 1.0/8.617e-05                    # eV to Kelvin
# kpc = 1e3*pc                            # kiloparsec in cm
# Mpc = 1e6*pc                            # megaparsec in cm
# msun2g = 1.98892e33 #(1*u.Msun).to('g').value       # solar mass to grams
m_p = 1.672661e-24

def stellar_to_halo_fraction(Mhalo, f0=0.3, Mt=1e8, Mp=3e11, g1=0.49, g2=-0.61, g3=3, g4=-3, Om=0.27, Ob=0.044):
	'''
	A parameterised stellar to halo relation (2011.12308, 2201.02210, 2302.06626).
	'''
	# Double power law, motivated by UVLFs
	dpl = 2*Ob/Om*f0/((Mhalo/Mp)**g1+(Mhalo/Mp)**g2)

	# Suppression at the small-mass end
	S_M = (1 + (Mt/Mhalo)**g3)**g4

	fstar = dpl*S_M

	return fstar

class StellarToHaloRelation:
	"""Modelling the mass relation between dark matter halo and the residing stars/galaxies."""
	def __init__(self, f0=0.3, Mt=1e8, Mp=3e11, g1=0.49, g2=-0.61, g3=3, g4=-3, cosmo=None):

		self.h, self.Ob, self.Om = cosmo.h, cosmo.Ob0, cosmo.Om0
		self.f0 = f0
		self.Mt = Mt
		self.Mp = Mp
		self.g1 = g1
		self.g2 = g2
		self.g3 = g3
		self.g4 = g4

	def deterministic(self, Mhalo):
		fstar_mean = stellar_to_halo_fraction(Mhalo, f0=self.f0, Mt=self.Mt, Mp=self.Mp, g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4, Ob=self.Ob, Om=self.Om)
		Mstar = Mhalo*fstar_mean 
		return {'fstar': fstar_mean, 'Mstar': Mstar}

	def stochastic_Gaussian(self, Mhalo, sigma):
		fstar_mean = stellar_to_halo_fraction(Mhalo, f0=self.f0, Mt=self.Mt, Mp=self.Mp, g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4, Ob=self.Ob, Om=self.Om)
		
		if isinstance(sigma, float): 
			fstar_std = lambda M: sigma*np.ones_like(Mhalo) 
		else:
			fstar_std = sigma

		fstar = np.clip(fstar_mean*(1+np.random.normal(0, fstar_std)), a_min=0, a_max=1)
		Mstar = Mhalo*fstar

		return {'fstar': fstar, 'Mstar': Mstar}

	def stochastic_lognormal(self, Mhalo, sigma, sigma_percent=False, **kwargs):
		fstar_mean = stellar_to_halo_fraction(Mhalo, f0=self.f0, Mt=self.Mt, Mp=self.Mp, g1=self.g1, g2=self.g2, g3=self.g3, g4=self.g4, Ob=self.Ob, Om=self.Om)

		if isinstance(sigma,float): 
			log_fstar_std = sigma*np.ones_like(Mhalo) 
		else:
			log_fstar_std = sigma

		log_fstar = np.log(fstar_mean)+np.random.normal(0, log_fstar_std)
		fstar = np.clip(a=np.exp(log_fstar), a_min=0, a_max=1)
		Mstar = Mhalo*fstar
		return {'fstar': fstar, 'Mstar': Mstar}

class EscapeFraction:
	""" Modelling the escape of photons from the stars/galaxies inside dark matter haloes."""
	def __init__(self, f0_esc=0.1, Mp_esc=1e10, al_esc=0):

		self.f0_esc = f0_esc
		self.Mp_esc = Mp_esc
		self.al_esc = al_esc

	def deterministic(self, Mhalo):
		fesc_mean = self.f0_esc*(Mhalo/self.Mp_esc)**self.al_esc
		return {'fesc': fesc_mean}


class BurstySFR:
	""" Modelling bursty star formation"""
	def __init__(self, beta1, beta2, tB0, tQ_frac, z0, alpha_h, cosmo, stochastic=False):
		self.beta1 = beta1
		self.beta2 = beta2
		self.tB0 = tB0
		self.tQ_frac = tQ_frac
		self.z0 = z0
		self.alpha_h = alpha_h
		self.cosmo = cosmo
		self.stochastic = stochastic

		self.t0 = cosmo.age(z0).to('Myr').value

	def time_burstiness(self, mass, z):
		if(self.stochastic):
			M0 = 10**np.random.normal(np.log10(M0), self.stochastic)
		else:
			M0 = mass/np.exp(-self.alpha_h*(z-self.z0))
		
		t = self.cosmo.age(z).to('Myr').value

		# burstiness time [Myr]
		tB = self.tB0*(M0/1e10)**self.beta1 * ((t - self.t0)*self.cosmo.H(z).to('1/Myr').value)**self.beta2

		return tB

	@np.vectorize
	def _burstiness_timescale(t_age, tB, tQ):
		""" of internal use for the integrated_burst_or_quiescent_galaxies method """
		i_time = np.floor(t_age/(tB+tQ))

		if(t_age <= i_time*(tB+tQ)+tB):
			return 1
		else:
			return 0
		
	def integrated_burst_or_quiescent_galaxies(self, mass, z, zi, zf):
		""" This case integrate the burst or quench time withing the time-step. It return a factor between 0 and 1 for quenched (value 0) or bursting (value 1). In bewteen values indicate that the sources are quencing for a period of time withing the time-step. """
		# TODO: It is computationally expensive, for some reason, due to the quad_vec method.... to investiage
		
		# get burstiness and quencing time
		tB = self.time_burstiness(mass, z)
		tQ = self.tQ_frac * tB
		
		# get time interval limits
		ti = cosmo.age(zi).to('Myr').value - self.t0
		tf = cosmo.age(zf).to('Myr').value - self.t0
		
		# get time fraction that the galaxies are on
		integr = lambda t : self.burstiness_timescale(t_age=t, tB=tB, tQ=tQ)
		timefrac_on = quad_vec(integr, ti, tf)[0]/(tf-ti)

		return timefrac_on

	def instant_burst_or_quiescent_galaxies(self, mass, z):
		""" This case is for instanteneous bursting or quenching. Do not account for the time integration. Mask the halo True (bursting) or False (quiescent). """
		# get burstiness and quencing time
		tB = self.time_burstiness(mass, z)
		tQ = self.tQ_frac * tB
		
		# get time at the corresponding redshift
		t_age = self.cosmo.age(z).to('Myr').value - self.t0
		assert t_age.all() > 0. , "Selected parameter t0 is wrong. The value of z0 is lower then the redshift of the first source file (increase the value z0)."
		
		# find the index of the burst/quench cycle in which the time-step, t, is inside 
		i_time = np.floor(t_age/(tB+tQ))
		
		# if True then the galaxy is bursting otherwise is quenching
		burst_or_quench = (t_age <= i_time*(tB+tQ)+tB)
		
		#print(' A total of %.2f %% of galaxies have bursty star-formation.' %(100*np.count_nonzero(burst_mask)/burst_mask.size))
		return burst_or_quench




class Halo2Grid:
	def __init__(self, box_len, n_grid, method='nearest'):
		self.box_len = box_len
		self.n_grid  = n_grid

		self.mpc_to_cm = 3.085677581491367e+24 # in cm
		self.Msun_to_g = 1.988409870698051e+33 # in gram
		self.pos_grid = None

	def set_halo_pos(self, pos, unit=None):
		if unit.lower()=='cm':
			self.pos_cm_to_grid(pos) 
		elif unit.lower()=='mpc':
			self.pos_mpc_to_grid(pos)
		else:
			self.pos_grid = pos 

	def set_halo_mass(self, mass, unit=None):
		if unit.lower()=='kg':
			self.mass_Msun = mass*1000/self.Msun_to_g
		elif unit.lower() in ['gram','g']:
			self.mass_Msun = mass/self.Msun_to_g
		elif unit.lower()=='msun':
			self.mass_Msun = mass
		else:
			print('Unknown mass units')

	def pos_cm_to_grid(self, pos_cm):
		pos_mpc  = pos_cm/self.mpc_to_cm
		pos_grid = pos_mpc*self.n_grid/self.box_len
		self.pos_grid = pos_grid
		print('Halo positions converted from cm to grid units')
		return pos_grid

	def pos_mpc_to_grid(self, pos_mpc):
		pos_grid = pos_mpc*self.n_grid/self.box_len
		self.pos_grid = pos_grid
		print('Halo positions converted from Mpc to grid units')
		return pos_grid

	def construct_tree(self, **kwargs):
		pos = kwargs.get('pos', self.pos_grid)
		if pos is None:
			print('Provide the halo positions via parameter "pos".')
			return None

		print('Creating a tree...')
		kdtree = cKDTree(pos)
		self.kdtree = kdtree
		print('...done')

	def value_on_grid(self, positions, values, **kwargs):
		# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic_dd.html
		statistic = kwargs.get('statistic', 'sum')
		bins = kwargs.get('bins', self.n_grid)
		binned_mass, bin_edges, bin_num = binned_statistic_dd(positions, values, statistic=statistic, bins=bins)
		return binned_mass, bin_edges, bin_num
	
	def halo_mass_on_grid(self, **kwargs):
		pos = kwargs.get('pos', self.pos_grid)
		if pos is None:
			print('Provide the halo positions via parameter "pos".')
			return None
		mass = kwargs.get('mass', self.pos_grid)
		if mass is None:
			print('Provide the halo masses via parameter "mass".')
			return None
		binned_mass = kwargs.get('binned_mass')
		if binned_mass is None: 
			binned_mass, bin_edges, bin_num = self.value_on_grid(pos, mass, statistic='sum', bins=self.n_grid)
		binned_pos_list  = np.argwhere(binned_mass>0) 
		binned_mass_list = binned_mass[binned_mass>0]
		return binned_pos_list, binned_mass_list
	
	def halo_value_on_grid(self, value, **kwargs):
		pos = kwargs.get('pos', self.pos_grid)
		if pos is None:
			print('Provide the halo positions via parameter "pos".')
			return None
		binned_value = kwargs.get('binned_value')
		if binned_value is None: 
			binned_value, bin_edges, bin_num = self.value_on_grid(pos, value, statistic='sum', bins=self.n_grid)
		binned_pos_list  = np.argwhere(binned_value>0) 
		binned_value_list = binned_value[binned_value>0]
		return binned_pos_list, binned_value_list


class SourceModel(Halo2Grid):
	"""Combines StellarToHaloRelation and Halo2Grid to model the source properties."""

	def __init__(self, Nion=1, f0=0.3, Mt=1e8, Mp=3e11,
					g1=0.49, g2=-0.61, g3=3, g4=-3,
					f0_esc=1, Mp_esc=1e10, al_esc=0,
					box_len=None, n_grid=None, method='nearest', **kwargs):
		"""
		Initialize the SourceModel.

		Parameters:
			f0 (float): Normalisation parameter for f_star.
			Mt (float): Truncation mass for f_star.
			Mp (float): Peak mass postion in f_star.
			g1 (float): Slope at low mass end.
			g2 (float): Slope at high mass end.
			g3 (float): Power-law index for the masses below Mt.
			g4 (float): Power-law index for the masses below Mt.
			f0_esc (float): Normalisation parameter for f_esc.
			Mp_esc (float): Normalisation for the masses in f_esc relation.
			al_esc (float): Power-law index for f_esc relation.
			box_len (float): Size of the simulation box.
			n_grid (int): Number of grid cells.
			method (str): Interpolation method for Halo2Grid.
			kwargs (dict): Additional parameters for parent classes.
		"""
		Halo2Grid.__init__(self, box_len=box_len, n_grid=n_grid, method=method)

		self.f_star = StellarToHaloRelation(f0=f0, Mt=Mt, Mp=Mp, g1=g1, g2=g2, g3=g3, g4=g4, **kwargs)
		self.f_esc  = EscapeFraction(f0_esc=f0_esc, Mp_esc=Mp_esc, al_esc=al_esc)

	def ionizing_flux(self, ts=10, mstar_model='deterministic', **kwargs):
		pos = kwargs.get('pos', self.pos_grid)
		if pos is None:
			print('Provide the halo positions via parameter "pos".')
			return None
		mass = kwargs.get('mass', self.pos_grid)
		if mass is None:
			print('Provide the halo masses via parameter "mass".')
			return None

		S_star_ref = 1e48
		
		# TODO: automatic selection of low mass or high mass. For the moment only high mass
		#mass2phot = msun2g * self.fgamma_hm * self.cosmology.Ob0 / (self.mean_molecular * c.m_p.cgs.value * self.ts * self.cosmology.Om0)    
		# TODO: for some reason the difference with the orginal Fortran run is of the molecular weight
		#self.printlog('%f' %self.mean_molecular )
		fgamma_hm = 1  # Set to 1 as we can absorb this into f0 in stellar-to-halo relation.

		# Mstar modelling
		if mstar_model.lower()=='deterministic':
			fstar  = self.f_star.deterministic(mass)
			mass_star = fstar['Mstar']

		# UV Escape fraction modelling
		fesc = self.f_esc.deterministic(mass)

		mass2phot = msun2g * fgamma_hm *  1/ (m_p * ts)    
		normflux = fesc['fesc']*mass_star * mass2phot / S_star_ref
		print(normflux)

		binned_flux, bin_edges, bin_num = self.value_on_grid(pos, normflux)
		binned_pos_list, binned_flux_list = self.halo_value_on_grid(normflux, binned_value=binned_flux)

		print(' Total Flux : %e' %np.sum(normflux*S_star_ref))
		print(' Source lifetime : %f Myr' %(ts/(1e6*YEAR)))
		print(' min, max source mass : %.3e  %.3e [Msun] and min, mean, max number of ionising sources : %.3e  %.3e  %.3e [1/s]' %(normflux.min()/mass2phot*S_star_ref, normflux.max()/mass2phot*S_star_ref, normflux.min()*S_star_ref, normflux.mean()*S_star_ref, normflux.max()*S_star_ref))
		self.binned_flux = binned_flux
		self.binned_pos_list  = binned_pos_list
		self.binned_flux_list = binned_flux_list
		return binned_pos_list, binned_flux_list


if __name__ == "__main__":
	import matplotlib.pyplot as plt 

	model1 = StellarToHaloRelation(f0=0.3, Mt=1e8, Mp=3e11,
					g1=0.49, g2=-0.61, g3=5, g4=-5)
	model2 = StellarToHaloRelation(f0=0.3, Mt=1e9, Mp=3e11,
					g1=0.49, g2=-0.61, g3=5, g4=-5)

	Ms = 10**np.linspace(7,14,250)
	star1 = model1.deterministic(Ms)
	star2 = model2.deterministic(Ms)

	star1_Gaussian1 = model1.stochastic_Gaussian(Ms, 0.50, sigma_percent=True)
	star1_Gaussian2 = model1.stochastic_Gaussian(Ms, 0.05, sigma_percent=True)
	star2_Gaussian1 = model2.stochastic_Gaussian(Ms, 0.50, sigma_percent=True)
	star2_Gaussian2 = model2.stochastic_Gaussian(Ms, 0.05, sigma_percent=True)

	fig, axs = plt.subplots(1,2,figsize=(13,5))
	axs[0].scatter(Ms, star1_Gaussian1['fstar'], c='C0', marker='o', edgecolor='k')
	axs[0].scatter(Ms, star1_Gaussian2['fstar'], c='C0', marker='^', edgecolor='k')
	axs[0].scatter(Ms, star2_Gaussian1['fstar'], c='C1', marker='o', edgecolor='k')
	axs[0].scatter(Ms, star2_Gaussian2['fstar'], c='C1', marker='^', edgecolor='k')
	axs[0].loglog(Ms, star1['fstar'], c='C0', lw=3, ls='-')
	axs[0].loglog(Ms, star2['fstar'], c='C1', lw=3, ls='--')
	axs[0].set_xlabel(r'$M_\mathrm{halo}$', fontsize=16)
	axs[0].set_ylabel(r'$f_\mathrm{\star}$', fontsize=16)
	axs[0].axis([3e7,8e13,1e-4,0.5])
	axs[1].scatter(Ms, star1_Gaussian1['Mstar'], c='C0', marker='o', edgecolor='k')
	axs[1].scatter(Ms, star1_Gaussian2['Mstar'], c='C0', marker='^', edgecolor='k')
	axs[1].scatter(Ms, star2_Gaussian1['Mstar'], c='C1', marker='o', edgecolor='k')
	axs[1].scatter(Ms, star2_Gaussian2['Mstar'], c='C1', marker='^', edgecolor='k')
	axs[1].loglog(Ms, star1['Mstar'], c='C0', lw=3, ls='-')
	axs[1].loglog(Ms, star2['Mstar'], c='C1', lw=3, ls='--')
	axs[1].set_xlabel(r'$M_\mathrm{halo}$', fontsize=16)
	axs[1].set_ylabel(r'$M_\mathrm{\star}$', fontsize=16)
	axs[1].axis([3e7,8e13,1e1,5e12])
	plt.tight_layout()
	plt.show()
