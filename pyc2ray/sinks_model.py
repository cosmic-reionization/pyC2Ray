import numpy as np, pandas as pd
import pyc2ray as pc2r
from .utils.other_utils import find_bins

class SinksPhysics:
    def __init__(self, params=None, N=None):
        self.clump_model = params['Sinks']['clumping_model']
        self.mfp_model = params['Sinks']['mfp_model']
        self.N = N

        res = params['Grid']['boxsize'] / self.N

        # MFP parameters
        if(self.mfp_model == 'constant'):
            # Set R_max (LLS 3) in cell units
            self.R_mfp_cell_unit = params['Sinks']['R_max_cMpc'] / res
        elif(self.mfp_model == 'Worseck2014'):
            self.A_mfp = params['Sinks']['A_mfp']
            self.etha_mfp = params['Sinks']['eta_mfp']
            self.z1_mfp = params['Sinks']['z1_mfp']
            self.eta1_mfp = params['Sinks']['eta1_mfp']
        else:
            ValueError(' MFP model not implemented : %s' %self.mfp_model)

        # Clumping factor parameters
        if(self.clump_model == 'constant'):
            self.calculate_clumping = np.ones((N, N, N), dtype=np.float64) * params['Sinks']['clumping']
        else:
            self.model_res = np.loadtxt(pc2r.__path__[0]+'/tables/resolutions.txt')*0.7     # the tables where calculated for cMpc/h with h=0.7
            
            # use parameters from tables with similare spatial resolution
            tab_res = self.model_res[np.argmin(np.abs(self.model_res - res))]

            # get parameter files
            self.clumping_params = np.loadtxt(pc2r.__path__[0]+'/tables/par_%s_%.3fMpc.txt' %(self.clump_model, tab_res))
            if(self.clump_model == 'redshift'):
                self.c2, self.c1, self.C0 = self.clumping_params[:3]
                self.calculate_clumping = self.biashomogeneous_clumping
            elif(self.clump_model == 'density'):
                self.calculate_clumping = self.inhomogeneous_clumping
            elif(self.clump_model == 'stochastic'):
                self.calculate_clumping = self.stochastic_clumping
            else:
                ValueError(' Cluming factor model not implemented : %s' %self.clump_model)

    def mfp_Worseck2014(self, z):
        R_mfp = self.A_mfp*((1+z)/5.)**self.etha_mfp
        R_mfp = R_mfp*(1+((1+z)/(1+self.z1_mfp))**self.eta1_mfp)
        return R_mfp
    
    def biashomogeneous_clumping(self, z):
        clump_fact = self.C0 * np.exp(self.c1*z + self.c2*z**2) + 1.
        return clump_fact * np.ones((self.N, self.N, self.N))
    
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
    


class SinksPhysics_old:
    def __init__(self, A_mfp=None, etha_mfp=None, z1_mfp=None, eta1_mfp=None, clump_model=None, res=None):
        # MFP parameters
        self.A_mfp = A_mfp
        self.etha_mfp = etha_mfp
        self.z1_mfp = z1_mfp
        self.eta1_mfp = eta1_mfp

        if(clump_model != 'constant' and clump_model != None):
            # Clumping factor parameters
            self.model_res = np.loadtxt(pc2r.__path__[0]+'/tables/resolutions.txt')
            
            # use parameters from tables with similare spatial resolution
            tab_res = self.model_res[np.argmin(np.abs(self.model_res - res))]

            # get parameter files
            self.clumping_params = np.loadtxt(pc2r.__path__[0]+'/tables/par_%s_%.3fMpc.txt' %(clump_model, tab_res))
            if(clump_model == 'redshift'):
                self.c2, self.c1, self.C0 = self.clumping_params[:3]
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
        clump_fact = self.C0 * np.exp(self.c1*z + self.c2*z**2) + 1.
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