import astropy.constants as cst
import astropy.units as u
import numpy as np

from ..load_extensions import load_c2ray

# TODO: Add chemistry wrapper function


def global_pass(
    dt, ndens, temp, xh, xh_av, xh_intermed, phi_ion, bh00, albpow, colh0, abu_c
):
    """Do chemistry on the whole grid

    Parameters
    ----------
    dt : float
        Timestep in seconds
    ndens : 3D-array
        The hydrogen number density of each cell in cm^-3
    temp : 3D-array
        The initial temperature of each cell in K
    xh : 3D-array
        Initial ionized fraction of each cell
    xh_av : 3D-array
        Current time-averaged ionization fraction over the timestep ()
    bh00 : float
        Hydrogen recombination parameter at 10^4 K in the case B OTS approximation
    albpow : float
        Power-law index for the H recombination parameter
    colh0 : float
        Hydrogen collisional ionization parameter
    temph0 : float
        Hydrogen ionization energy expressed in K
    abu_c : float
        Carbon abundance
    loss_fraction : float (default: 1e-2)
        Fraction of remaining photons below we stop ray-tracing (subbox technique)
    logfile : str
        Name of the file to append logs to. Default: pyC2Ray.log
    quiet : bool
        Don't write logs to stdout. Default is false
    """
    pass


def hydrogenODE(
    dt,
    ndens,
    temp,
    xh,
    phi_ion,
    clump,
    bh00=2.59e-13,
    albpow=-0.7,
    colh0=1.3e-8,
    abu_c=7.1e-7,
):
    """Do chemistry on the whole grid just for hydrogen.
        This script is in principle for testing or for use of the chemistry alone.

    Parameters
    ----------
    dt : float
        Timestep in seconds
    ndens : 3D-array
        The hydrogen number density of each cell in cm^-3
    temp : 3D-array
        The initial temperature of each cell in K
    xh : 3D-array
        Initial ionized fraction of each cell
    xh_av : 3D-array
        Current time-averaged ionization fraction over the timestep ()
    bh00 : float
        Hydrogen recombination parameter at 10^4 K in the case B OTS approximation
    albpow : float
        Power-law index for the H recombination parameter
    colh0 : float
        Hydrogen collisional ionization parameter
    temph0 : float
        Hydrogen ionization energy expressed in K
    abu_c : float
        Carbon abundance
    loss_fraction : float (default: 1e-2)
        Fraction of remaining photons below we stop ray-tracing (subbox technique)
    logfile : str
        Name of the file to append logs to. Default: pyC2Ray.log
    quiet : bool
        Don't write logs to stdout. Default is false
    """
    if not isinstance(type(xh), np.ndarray):
        xh = np.asfortranarray(xh)
    if not isinstance(type(ndens), np.ndarray):
        ndens = np.asfortranarray(ndens)
    if not isinstance(type(temp), np.ndarray):
        temp = np.asfortranarray(temp)
    if not isinstance(type(phi_ion), np.ndarray):
        phi_ion = np.asfortranarray(phi_ion)

    xh_intermed = xh
    libc2ray = load_c2ray()

    temph0 = (13.598 * u.eV / cst.k_B).cgs.value

    # dt, ndens, temp, xh, xh_av, xh_intermed, phi_ion, bh00, albpow, colh0, temph0, abu_c
    conv_flag = libc2ray.chemistry.global_pass(
        dt,
        ndens,
        temp,
        xh,
        xh,
        xh_intermed,
        phi_ion,
        clump,
        bh00,
        albpow,
        colh0,
        temph0,
        abu_c,
    )

    convergence = conv_flag / np.size(xh_intermed)
    assert convergence < 0.01

    return xh_intermed
