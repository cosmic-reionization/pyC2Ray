import numpy as np, pandas as pd
import pyc2ray as pc2r
from .utils.other_utils import find_bins

class SinksPhysics:
    def __init__(self, A_mfp=None, etha_mfp=None, z1_mfp=None, eta1_mfp=None, clump_model=None, res=None):
        # MFP parameters
        self.A_mfp = A_mfp
        self.etha_mfp = etha_mfp
        self.z1_mfp = z1_mfp
        self.eta1_mfp = eta1_mfp

        if(clump_model != 'constant'):
            # Clumping factor parameters
            self.model_res = np.loadtxt(pc2r.__path__[0]+'/tables/resolutions.txt')
            
            # use parameters from tables with similare spatial resolution
            tab_res = self.model_res[np.argmin(np.abs(self.model_res - res))]

            # get parameter files
            self.clumping_params = np.loadtxt(pc2r.__path__[0]+'/tables/par_%s_%.3fMpc.txt' %(clump_model, tab_res))
            if(clump_model == 'redshift'):
                self.c2, self.c1, self.C0 = self.clumping_params[:4]
                self.calculate_clumping = self.biashomogeneous_clumping
            elif(clump_model == 'density'):
                self.calculate_clumping = self.inhomogeneous_clumping
            elif(clump_model == 'stochastic'):
                self.calculate_clumping = self.stochastic_clumping

    def mfp_Worseck2014(self, z):
        R_mfp = self.A_mfp*((1+z)/5.)**self.etha_mfp
        R_mfp = R_mfp*(1+((1+z)/(1+self.z1_mfp))**self.eta1_mfp)
        return R_mfp
    
    def biashomogeneous_clumping(self, z):
        clump_fact = self.C0 * np.exp(self.c1*z + self.c2*z^2) + 1.
        return clump_fact
    
    def inhomogeneous_clumping(self, z, ndens):
        redshift = self.clump_params[:,0]

        # find nearest redshift bin
        zlow, zhigh = find_bins(z, redshift)
        i_low, i_high = np.digitize(zlow, redshift), np.digitize(zhigh, redshift)

        # calculate weight to 
        w_l, w_h = 1-(z-zlow)/(zhigh - zlow), 1-(zhigh-z)/(zhigh - zlow)

        # get parameters weighted   
        a, b, c = self.clump_params[i_low,1:4]*w_l + self.clump_params[i_high,1:4]*w_h        
        
        x =  np.log(1 + ndens / ndens.mean())
        clump_fact = 10**(a*x**2 + b*x**2 + c)
        return clump_fact
    
    def stochastic_clumping(self, z, ndens):
        # TODO: implement
        MaxBin = 5
        lognormParamsFile = pd.read_csv('par_stochastic_2.024Mpc.csv', index_col=0, converters={'bin%d' %i: lambda string: np.array(string[1:-1].split(', '), dtype=float) for i in range(MaxBin)})

        return 0