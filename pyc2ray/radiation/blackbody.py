import numpy as np
from scipy.integrate import quad,quad_vec
from astropy.constants import h as hplanck_ac
from astropy.constants import Ryd as Ryd_ac
from astropy.constants import c as c_ac

# For detailed comparisons with C2Ray, we use the same exact value for the constants
# This can be changed to the astropy values once consistency between the two codes has been established

h_over_k = 6.6260755e-27 / 1.381e-16
pi =3.141592654
c = 2.997925e+10
two_pi_over_c_square = 2.0*pi/(c*c)
hplanck = hplanck_ac.cgs.value
ion_freq_HI = (Ryd_ac * c_ac).cgs.value
sigma_0 = 6.3e-18

__all__ = ['BlackBodySource']

class BlackBodySource:
    """A point source emitting a Black-body spectrum
    """
    def __init__(self, temp, grey, freq0, pl_index) -> None:
        self.temp = temp
        self.grey = grey
        self.freq0 = freq0
        self.pl_index = pl_index
        self.R_star = 1.0

    def SED(self,freq):
        if (freq*h_over_k/self.temp < 700.0):
            sed = 4*np.pi*self.R_star**2*two_pi_over_c_square*freq**2/(np.exp(freq*h_over_k/self.temp)-1.0)
        else:
            sed = 0.0
        return sed
    
    def integrate_SED(self,f1,f2):
        res = quad(self.SED,f1,f2)
        return res[0]
    
    def normalize_SED(self,f1,f2,S_star_ref):
        S_unscaled = self.integrate_SED(f1,f2)
        S_scaling = S_star_ref / S_unscaled
        self.R_star = np.sqrt(S_scaling) * self.R_star

    def cross_section_freq_dependence(self,freq):
        if self.grey:
            return 1.0
        else:
            return (freq/self.freq0)**(-self.pl_index)
    
    # C2Ray distinguishes between optically thin and thick cells, and calculates the rates differently
    # for those two cases. See radiation_tables.F90, lines 345 -
    def _photo_thick_integrand_vec(self,freq,tau):
        itg = self.SED(freq) * np.exp(-tau*self.cross_section_freq_dependence(freq))
        # To avoid overflow in the exponential, check
        return np.where(tau*self.cross_section_freq_dependence(freq) < 700.0,itg,0.0)
    
    def _photo_thin_integrand_vec(self,freq,tau):
        itg = self.SED(freq) * self.cross_section_freq_dependence(freq) * np.exp(-tau*self.cross_section_freq_dependence(freq))
        return np.where(tau*self.cross_section_freq_dependence(freq) < 700.0,itg,0.0)
    
    def _heat_thick_integrand_vec(self,freq,tau):
        photo_thick = self._photo_thick_integrand_vec(freq,tau)
        return hplanck * (freq - ion_freq_HI) * photo_thick
    
    def _heat_thin_integrand_vec(self,freq,tau):
        photo_thin = self._photo_thin_integrand_vec(freq,tau)
        return hplanck * (freq - ion_freq_HI) * photo_thin
    
    def make_photo_table(self,tau,freq_min,freq_max,S_star_ref):
        self.normalize_SED(freq_min,freq_max,S_star_ref)
        integrand_thin = lambda f : self._photo_thin_integrand_vec(f,tau)
        integrand_thick = lambda f : self._photo_thick_integrand_vec(f,tau)
        table_thin = quad_vec(integrand_thin,freq_min,freq_max,epsrel=1e-12)[0]
        table_thick = quad_vec(integrand_thick,freq_min,freq_max,epsrel=1e-12)[0]
        return table_thin, table_thick
    
    def make_heat_table(self,tau,freq_min,freq_max,S_star_ref):
        self.normalize_SED(freq_min,freq_max,S_star_ref)
        integrand_thin = lambda f : self._heat_thin_integrand_vec(f,tau)
        integrand_thick = lambda f : self._heat_thick_integrand_vec(f,tau)
        table_thin = quad_vec(integrand_thin,freq_min,freq_max,epsrel=1e-12)[0]
        table_thick = quad_vec(integrand_thick,freq_min,freq_max,epsrel=1e-12)[0]
        return table_thin, table_thick
        