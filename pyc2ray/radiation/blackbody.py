import numpy as np, scipy
from scipy.integrate import quad,quad_vec
import astropy.constants as cst
import astropy.units as u

# For detailed comparisons with C2Ray, we use the same exact value for the constants
# This can be changed to the astropy values once consistency between the two codes has been established
h_over_k = (cst.h/cst.k_B).cgs.value
pi = np.pi
c = cst.c.cgs.value
two_pi_over_c_square = 2.0*pi/(c*c)
hplanck = cst.h.cgs.value
ion_freq_HI = (cst.Ryd * cst.c).cgs.value
sigma_0 = 6.3e-18

__all__ = ['BlackBodySource', 'YggdrasilModel']

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
    
    # C2Ray distinguishes between optically thin and thick cells, and calculates the rates differently for those two cases. See radiation_tables.F90, lines 345 -
    def _photo_thick_integrand_vec(self, freq, tau):
        itg = self.SED(freq) * np.exp(-tau*self.cross_section_freq_dependence(freq))
        # To avoid overflow in the exponential, check
        return np.where(tau*self.cross_section_freq_dependence(freq) < 700.0,itg,0.0)
    
    def _photo_thin_integrand_vec(self, freq, tau):
        itg = self.SED(freq) * self.cross_section_freq_dependence(freq) * np.exp(-tau*self.cross_section_freq_dependence(freq))
        return np.where(tau*self.cross_section_freq_dependence(freq) < 700.0,itg,0.0)
    
    def _heat_thick_integrand_vec(self, freq, tau):
        photo_thick = self._photo_thick_integrand_vec(freq,tau)
        return hplanck * (freq - ion_freq_HI) * photo_thick
    
    def _heat_thin_integrand_vec(self, freq, tau):
        photo_thin = self._photo_thin_integrand_vec(freq,tau)
        return hplanck * (freq - ion_freq_HI) * photo_thin
    
    def make_photo_table(self, tau, freq_min, freq_max, S_star_ref):
        self.normalize_SED(freq_min,freq_max,S_star_ref)
        integrand_thin = lambda f : self._photo_thin_integrand_vec(f,tau)
        integrand_thick = lambda f : self._photo_thick_integrand_vec(f,tau)
        table_thin = quad_vec(integrand_thin,freq_min,freq_max,epsrel=1e-12)[0]
        table_thick = quad_vec(integrand_thick,freq_min,freq_max,epsrel=1e-12)[0]
        return table_thin, table_thick
    
    def make_heat_table(self, tau, freq_min, freq_max, S_star_ref):
        self.normalize_SED(freq_min,freq_max,S_star_ref)
        integrand_thin = lambda f : self._heat_thin_integrand_vec(f,tau)
        integrand_thick = lambda f : self._heat_thick_integrand_vec(f,tau)
        table_thin = quad_vec(integrand_thin,freq_min,freq_max,epsrel=1e-12)[0]
        table_thick = quad_vec(integrand_thick,freq_min,freq_max,epsrel=1e-12)[0]
        return table_thin, table_thick
        

class YggdrasilModel:
    """ Use Yggdrasil model for SED """
    def __init__(self, tabname, grey, freq0, pl_index, S_star_ref) -> None:
        self.grey = grey
        self.freq0 = freq0
        self.tabname = tabname
        self.pl_index = pl_index
    """
    # This was used for debugging. Can be usefull in the future(?)
    def SED(self, f1, f2):
        freqs = np.linspace(f2, f1, 10) * u.Hz
        lamb = (cst.c  / freqs).to('AA')
        R_star, temp = 1*u.Rsun, 5e4*u.K

        #ampl = 8*np.pi**2 * R_star**2 *cst.h * cst.c**2 / lamb**5
        #sed = ampl / (np.exp((cst.h*cst.c/lamb/cst.k_B/temp).cgs)-1.0)
        ampl = 8*np.pi**2 * R_star**2 *cst.h * freqs**5/cst.c**3
        sed = ampl / (np.exp(freqs*cst.h/cst.k_B/temp)-1.0)
        
        return sed.to('erg / s / AA').value, freqs.value, lamb.value
        
    """
    def SED(self, f1, f2):
        lamb, flux = np.loadtxt(self.tabname, unpack=True) # wavelenght in (Angstrom), Flux in (erg/s/AA)
        freqs = (cst.c/(lamb*u.AA)).to('Hz').value

        if(freqs.min() == freqs[-1]) and (freqs.max() == freqs[0]):
            # this is for the discrete integral scipy.integrate.simpson, which require increasing value for the x-axis
            lamb = lamb[::-1] 
            freqs = freqs[::-1]
            flux = flux[::-1]
        
        int_range = (freqs >= f1) * (freqs <= f2)
        sed = flux[int_range]
        return sed, freqs[int_range], lamb[int_range]
    
    
    def integrate_SED(self, sed, freq):
        assert (freq.min() == freq[0]) 
        assert (freq.max() == freq[-1])
        
        res = scipy.integrate.simpson(x=freq, y=sed, even='simpson')
        return res
    
    def normalize_SED(self, sed, freq, S_star_ref):
        S_unscaled = self.integrate_SED(sed, freq)
        # MB: in C2Ray this was: self.R_star = np.sqrt(S_scaling) * self.R_star. Here we define the SED with the proper units so we do not need to squareroot (as we do not multiply to R_star) and instead multiply directly to the SED.
        S_scaling = S_star_ref / S_unscaled
        return sed * S_scaling 

    def cross_section_freq_dependence(self, freq):
        if self.grey:
            return 1.0
        else:
            return (freq/self.freq0)**(-self.pl_index)
    
    # C2Ray distinguishes between optically thin and thick cells, and calculates the rates differently for those two cases. See radiation_tables.F90, lines 345 -
    def _photo_thick_integrand_vec(self, sed, freq, tau):
        itg = sed * np.exp(-tau*self.cross_section_freq_dependence(freq))
        # To avoid overflow in the exponential, check
        return np.where(tau*self.cross_section_freq_dependence(freq) < 700.0, itg, 0.0)
    
    def _photo_thin_integrand_vec(self, sed, freq, tau):
        itg = sed * self.cross_section_freq_dependence(freq) * np.exp(-tau*self.cross_section_freq_dependence(freq))
        return np.where(tau*self.cross_section_freq_dependence(freq) < 700.0, itg, 0.0)
    
    def _heat_thick_integrand_vec(self, sed, freq, tau):
        photo_thick = self._photo_thick_integrand_vec(sed, sed, freq, tau)
        return hplanck * (freq - ion_freq_HI) * photo_thick
    
    def _heat_thin_integrand_vec(self, sed, freq, tau):
        photo_thin = self._photo_thin_integrand_vec(sed, sed, freq, tau)
        return hplanck * (freq - ion_freq_HI) * photo_thin
    
    def make_photo_table(self, tau, freq_min, freq_max, S_star_ref):
        sed, freqs, lamb = self.SED(f1=freq_min, f2=freq_max)
        norm_sed = self.normalize_SED(sed, freqs, S_star_ref)

        table_thin = np.array([scipy.integrate.simpson(y=self._photo_thin_integrand_vec(sed=norm_sed, freq=freqs, tau=t), x=freqs, even='simpson') for t in tau])
        table_thick = np.array([scipy.integrate.simpson(y=self._photo_thick_integrand_vec(sed=norm_sed, freq=freqs, tau=t), x=freqs, even='simpson') for t in tau])
        
        return table_thin, table_thick
    
    def make_heat_table(self,tau, freq_min, freq_max, S_star_ref):

        sed, freqs, lamb = self.SED(freq_min, freq_max)
        norm_sed = self.normalize_SED(sed, lamb, S_star_ref)
        
        table_thin = np.array([scipy.integrate.simpson(y=self._heat_thin_integrand_vec(sed=norm_sed, freq=freqs, tau=t), x=freqs, even='simpson') for t in tau])
        table_thick = np.array([scipy.integrate.simpson(y=self._heat_thick_integrand_vec(sed=norm_sed, freq=freqs, tau=t), x=freqs, even='simpson') for t in tau])
        
        return table_thin, table_thick