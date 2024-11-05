import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.interpolate import interp1d

# Load MATLAB data
data = loadmat('cooling_rates_our_tables.mat')
H1A = 10 ** data['H1A']
H1B = 10 ** data['H1B']
He1 = 10 ** data['He1']
He2 = 10 ** data['He2']
He0 = 10 ** data['He0']
H0 = 10 ** data['H0']

logT = np.arange(1, 9.01, 0.01)
T = 10 ** logT

# Constants
T_H = np.array([157807, 285335, 631515])
k_b = 1.3806504e-16  # in erg/K

# Calculate lambda values
lambda_vals = 2 * (T_H[:, None] / T)

# Recombination cooling rates from Hui & Gnedin 1997
R_A = np.zeros((3, T.size))
R_B = np.zeros((3, T.size))
R_A[0, :] = 1.778e-29 * T * lambda_vals[0, :]**1.965 / (1 + (lambda_vals[0, :] / 0.541)**0.502)**2.697
R_B[0, :] = 3.435e-30 * T * lambda_vals[0, :]**1.970 / (1 + (lambda_vals[0, :] / 2.250)**0.376)**3.720

R_A[1, :] = 3.0e-14 * lambda_vals[1, :]**0.654 * k_b * T
R_B[1, :] = 1.26e-14 * lambda_vals[1, :]**0.750 * k_b * T

D_2a = 1.9e-3 * T**(-3/2) * np.exp(-0.75 * lambda_vals[2, :] / 2) * (1 + 0.3 * np.exp(-0.15 * lambda_vals[2, :] / 2)) * 0.75 * k_b * T_H[2]

R_A[2, :] = 2 * 1.778e-29 * T * lambda_vals[2, :]**1.965 / (1 + (lambda_vals[2, :] / 0.541)**0.502)**2.697
R_B[2, :] = 2 * 3.435e-30 * T * lambda_vals[2, :]**1.970 / (1 + (lambda_vals[2, :] / 2.250)**0.376)**3.720

# Collisional ionization cooling from Hui & Gnedin
CI = np.zeros((3, T.size))
CR = np.zeros((3, T.size))
CI[0, :] = 21.11 * T**(-3/2) * np.exp(-lambda_vals[0, :] / 2) * lambda_vals[0, :]**-1.089 / (1 + (lambda_vals[0, :] / 0.354)**0.874)**1.101
CR[0, :] = k_b * T_H[0] * CI[0, :]
CI[1, :] = 32.38 * T**(-3/2) * np.exp(-lambda_vals[1, :] / 2) * lambda_vals[1, :]**-1.146 / (1 + (lambda_vals[1, :] / 0.416)**0.987)**1.056
CR[1, :] = k_b * T_H[1] * CI[1, :]
CI[2, :] = 19.95 * T**(-3/2) * np.exp(-lambda_vals[2, :] / 2) * lambda_vals[2, :]**-1.089 / (1 + (lambda_vals[2, :] / 0.553)**0.735)**1.275
CR[2, :] = k_b * T_H[2] * CI[2, :]

# Collisional ionization rates
colh0 = 5.835410275968903E-11
colhe0 = 2.709585263842644E-11
colhe1 = 5.707336249831606E-12
sqrtt0 = np.sqrt(T)
acolh0 = colh0 * sqrtt0 * np.exp(-T_H[0] / T)
acolhe0 = colhe0 * sqrtt0 * np.exp(-T_H[1] / T)
acolhe1 = colhe1 * sqrtt0 * np.exp(-T_H[2] / T)

# Collisional excitation cooling from Hui & Gnedin
EC = np.zeros((3, T.size))
EC[0, :] = 7.5e-19 * np.exp(-0.75 * lambda_vals[0, :] / 2) / (1 + np.sqrt(T / 1e5))
EC[1, :] = 9.1e-27 * T**(-0.1687) * np.exp(-13179 / T)
EC[2, :] = 5.54e-17 * np.exp(-0.75 * lambda_vals[2, :] / 2) / (1 + np.sqrt(T / 1e5)) / T**0.397

# Interpolations (example only, real data from recomrates_He1_Hummer.mat should be loaded)
# Assuming TH, beta_H_B, etc. are arrays loaded from other data files
# hummer_H_B = interp1d(TH, beta_H_B, kind='linear', fill_value='extrapolate')(T)
# Similarly interpolate for all required variables

# Plotting
plt.figure()
plt.title('H1')
plt.loglog(H1A[:, 0], H1A[:, 1], 'r--', label='A, table')
plt.loglog(H1B[:, 0], H1B[:, 1], 'r-', label='B, table')
# Additional plots for Hui & Gnedin, Hummer, etc., as in the original MATLAB script
plt.legend()

plt.figure()
plt.title('H0')
plt.loglog(H0[:, 0], H0[:, 1], 'r', label='table')
plt.loglog(T, CR[0, :] + EC[0, :], 'b', label='CollExc+CollIon, Hui&Gn')
plt.loglog(T, CR[0, :], 'b-.', label='CollIon, Hui&Gn')
plt.loglog(T, EC[0, :], 'b:', label='CollExc, Hui&Gn')
plt.legend()

# Additional figures and plots following the same structure

plt.show()
