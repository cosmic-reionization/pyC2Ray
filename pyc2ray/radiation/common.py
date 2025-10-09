import numpy as np

# h_over_k = (ac.h/(ac.k_B)).cgs.value

# two_pi_over_c_square = 2*np.pi/ac.c.cgs.value**2
c = 2.997925e10

__all__ = ["make_tau_table"]


def make_tau_table(minlogtau, maxlogtau, NumTau):
    """Utility function to create optical depth array for C2Ray

    Parameters
    ----------
    minlogtau : float
        Base 10 log of the minimum value of the table in τ (excluding τ = 0)
    minlogtau : float
        Base 10 log of the maximum value of the table in τ
    NumTau : int
        Number of points in the table, excluding τ = 0

    Returns
    -------
    tau : 1D-array of shape (NumTau + 1)
        Array of optical depths log-distributed between minlogtau and maxlogtau. The 0-th
        entry is τ = 0 and so the array has shape NumTau+1 (same convention as c2ray)
    dlogtau : float
        Table step size in log10
    """
    dlogtau = (maxlogtau - minlogtau) / NumTau
    tau = np.empty(NumTau + 1)
    tau[0] = 0.0
    tau[1:] = 10 ** (minlogtau + np.arange(NumTau) * dlogtau)
    return tau, dlogtau
