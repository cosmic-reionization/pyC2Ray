import numpy as np

class SinksPhysics:
    def __init__(self, A_mfp=None, etha_mfp=None, z1_mfp=None, eta1_mfp=None):
        self.A_mfp = A_mfp
        self.etha_mfp = etha_mfp
        self.z1_mfp = z1_mfp
        self.eta1_mfp = eta1_mfp

    def mfp_Worseck2014(self, z):
        R_mfp = self.A_mfp*((1+z)/5.)**self.etha_mfp
        R_mfp = R_mfp*(1+((1+z)/(1+self.z1_mfp))**self.eta1_mfp)
        # if(R_mfp < 10.):
        #     # minimum value for R_mfp
        #     R_mfp = 10.
        return R_mfp
    
    def biashomogeneous_clumping(self, z):
        # TODO: implement
        return 0
    
    def inhomogeneous_clumping(self, z):
        # TODO: implement
        return 0