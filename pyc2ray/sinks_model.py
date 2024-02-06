import numpy as np

class SinksPhysics:
    def __init__(self, A_mfp=None, etha_mfp=None):
        self.A_mfp = A_mfp
        self.etha_mfp = etha_mfp

    def mfp_Choudhury09(self, z):
        R_mfp = self.A_mfp*((1+z)/5.)**self.etha_mfp
        if(R_mfp < 10.):
            # minimum value for R_mfp
            R_mfp = 10.
        return R_mfp
    
    def biashomogeneous_clumping(self, z):
        # TODO: implement
        return 0
    
    def inhomogeneous_clumping(self, z):
        # TODO: implement
        return 0