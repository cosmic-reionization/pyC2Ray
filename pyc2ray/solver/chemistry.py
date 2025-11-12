import numpy as np

# Define constants
epsilon = 1e-14
minimum_fractional_change = 1.0e-3
minimum_fraction_of_atoms = 1.0e-8


def global_pass(
    dt,
    ndens,
    temp,
    xh,
    xh_av,
    xh_intermed,
    phi_ion,
    clump,
    bh00,
    albpow,
    colh0,
    temph0,
    abu_c,
):
    conv_flag = 0

    new_xh = np.zeros_like(xh)
    new_xh_av = np.zeros_like(xh)

    for k in range(xh.shape[0]):
        for j in range(xh.shape[1]):
            for i in range(xh.shape[2]):
                # Initialize local quantities
                temperature_start = temp[i, j, k]
                ndens_p = ndens[i, j, k]
                phi_ion_p = phi_ion[i, j, k]
                clump_p = clump[i, j, k]

                # Initialize local ion fractions
                xh_p = xh[i, j, k]
                xh_av_p = xh_av[i, j, k]
                xh_intermed_p = xh_intermed[i, j, k]

                # call do chemistry
                new_xh[i, j, k], new_xh_av[i, j, k], _ = do_chemistry(
                    dt,
                    ndens_p,
                    temperature_start,
                    xh_p,
                    xh_av_p,
                    xh_intermed_p,
                    phi_ion_p,
                    clump_p,
                    bh00,
                    albpow,
                    colh0,
                    temph0,
                    abu_c,
                )

                # Check for convergence (global flag). In original, convergence is tested using neutral fraction, but testing with ionized fraction should be equivalent. TODO: add temperature convergence criterion when non-isothermal mode is added later on.
                xh_av_p_old = new_xh_av[i, j, k]

                cond1 = np.abs(xh_av_p - xh_av_p_old) > minimum_fractional_change
                cond2 = (
                    np.abs((xh_av_p - xh_av_p_old) / (1.0 - xh_av_p))
                    > minimum_fractional_change
                )
                cond3 = (1.0 - xh_av_p) > minimum_fraction_of_atoms
                if cond1 * cond2 * cond3:
                    conv_flag += 1

    return new_xh, new_xh_av, conv_flag


def do_chemistry(
    dt,
    ndens_p,
    temperature_start,
    xh_p,
    xh_av_p,
    xh_intermed_p,
    phi_ion_p,
    clump_p,
    bh00,
    albpow,
    colh0,
    temph0,
    abu_c,
):
    # Initialize local quantities
    temperature_end = temperature_start

    # Calculate the new and mean ionization states
    # xh_intermed = np.copy(xh)  # Placeholder, actual intermediate state calculation may vary

    convergence, niter = False, 0
    while not convergence:
        # Save temperature solution from last iteration
        temperature_previous_iteration = temperature_end

        # At each iteration, the intial condition x(0) is reset. Change happens in the time-average and thus the electron density
        xh_av_p_old = xh_av_p

        # Calculate (mean) electron density
        de = ndens_p * (xh_av_p + abu_c)

        # Calculate the new and mean ionization states
        new_xh_p, xh_av_p = doric(
            xh_p,
            dt,
            temperature_end,
            de,
            phi_ion_p,
            bh00,
            albpow,
            colh0,
            temph0,
            clump_p,
        )

        # Check for convergence
        cond1 = (
            np.abs(xh_av_p - xh_av_p_old) / (1 - xh_av_p) < minimum_fractional_change
        )
        cond2 = 1 - xh_av_p < minimum_fraction_of_atoms
        cond3 = (
            np.abs(temperature_end - temperature_previous_iteration) / temperature_end
            < minimum_fractional_change
        )

        if (cond1 or cond2) and cond3:
            convergence = True

        # Warn about non-convergence and terminate iteration
        if niter > 400:
            print("Warning!!! non-convergence. Therefore, terminating iteration.")
            convergence = True
        else:
            niter += 1

    return new_xh_p, xh_av_p, xh_intermed_p


def doric(xh_old, dt, temp_p, rhe, phi_p, bh00, albpow, colh0, temph0, clumping):
    # Calculate the hydrogen recombination rate at the local temperature
    brech0 = clumping * bh00 * (temp_p / 1e4) ** albpow

    # Calculate the hydrogen collisional ionization rate at the local temperature
    acolh0 = colh0 * np.sqrt(temp_p) * np.exp(-temph0 / temp_p)

    # Find the true photo-ionization rate
    aphoth0 = phi_p

    # Determine ionization states
    aih0 = aphoth0 + rhe * acolh0
    delth = aih0 + rhe * brech0
    eqxh = aih0 / delth
    deltht = delth * dt
    ee = np.exp(-deltht)
    xh = (xh_old - eqxh) * ee + eqxh

    # Handle precision fluctuations
    xh = np.maximum(xh, epsilon)

    # Determine average ionization fraction over the time step
    avg_factor = np.where(deltht < 1.0e-8, 1.0, (1.0 - ee) / deltht)
    xh_av = eqxh + (xh_old - eqxh) * avg_factor
    xh_av = np.maximum(xh_av, epsilon)

    return xh, xh_av


# TODO: here you can plug at the place of the doric in the do_chemistry (making the right changes)
def friedrich(
    n_gas,
    xHII_old,
    xHeII_old,
    xHeIII_old,
    dt,
    dr,
    temp_p,
    n_e,
    phi_HI,
    phi_HeI,
    phi_HeII,
    heat_HI,
    heat_HeI,
    heat_HeII,
    X,
    Y,
):
    """
    Chemistry equation solver for H and He.

    Inputs:
        - n_gas (float):        gas number density
        - xHI_old (float):      hydrogen ionized fraction of the cell
        - xHeI_old (float):     helium first ionized fraction of the cell
        - xHeII_old (float):    helium second ionized fraction of the cell
        - dt (float):           time step in cgs units
        - dr (float):           cell size in cgs units
        - n_e (float):          electron number density of the cell
        - phi_HI (float):       photoionization rate for hydrogen of the cell
        - phi_HeI (float):      photoionization rate for first ionized helium of the cell
        - phi_HeII (float):     photoionization rate for second ionized helium of the cell
        - X (float):            abbundance of hydrogen
        - Y (float):            abbunace of helium
    Return:
        - ...
    """

    # Recombination rate of HI (Eq. 2.12 and 2.13)
    alphA_HII = (
        1.269e-13
        * np.power(315608 / temp_p, 1.503)
        / np.power(np.power(1 + 604613 / temp_p, 0.47), 1.923)
    )
    alphB_HII = (
        2.753e-14
        * np.power(315608 / temp_p, 1.5)
        / np.power(np.power(1 + 115185 / temp_p, 0.407), 2.242)
    )
    alph1_HII = alphA_HII - alphB_HII

    # Recombination rate of HeII (Eq. 2.14-17)
    if temp_p < 9e3:
        alphA_HeII = (
            1.269e-13
            * np.power(570662 / temp_p, 1.503)
            / np.power(np.power(1 + 1093222 / temp_p, 0.47), 1.923)
        )
        # alphB_HeII = (
        #     2.753e-14
        #     * np.power(570662 / temp_p, 1.5)
        #     / np.power(np.power(1 + 208271 / temp_p, 0.407), 2.242)
        # )
    else:
        alphA_HeII = 3e-14 * np.power(570662 / temp_p, 0.654) + 1.9e-3 * np.power(
            temp_p, -1.5
        ) * np.exp(-4.7e5 / temp_p) * (1 + 0.3 * np.exp(-9.4e4 / temp_p))
        # alphB_HeII = 1.26e-14 * np.power(570662 / temp_p, 0.75) + 1.9e-3 * np.power(
        #     temp_p, -1.5
        # ) * np.exp(-4.7e5 / temp_p) * (1 + 0.3 * np.exp(-9.4e4 / temp_p))

    # Recombination rate of HeIII (Eq. 2.18-20)
    alphA_HeIII = (
        2.538e-13
        * np.power(1262990 / temp_p, 1.503)
        / np.power(1 + np.power(2419521 / temp_p, 1.923), 1.923)
    )
    alphB_HeIII = (
        5.506e-14
        * np.power(1262990 / temp_p, 1.5)
        / np.power(1 + np.power(460945 / temp_p, 0.407), 2.242)
    )
    alph1_HeIII = (
        alphA_HeIII - alphB_HeIII
    )  # TODO: double check that this is the correct definition
    alph2_HeIII = 8.54e-11 * np.power(temp_p, -0.6)

    # two photons emission from recombination of HeIII
    nu = 0.285 * np.power(temp_p / 1e4, 0.119)

    # column density of half a cell
    NHI = n_gas * (1.0 - xHII_old) / (dr / 2)
    # FIXME: xHeI_old is undefined, this is a bug
    NHeI = n_gas * xHeI_old / (dr / 2)  # noqa: F821
    NHeII = n_gas * xHII_old / (dr / 2)

    # opt depth of HI at HeI ion threshold
    sigma_H_heth = 1.238e-18  # HI cross-section at HeI ionization threshold
    tau_H_heth = NHI * sigma_H_heth

    # opt depth of HeI at HeI ion threshold
    sigma_HeI_at_ion_freq = 7.430e-18  # HeI cross section at its ionzing frequency
    tau_He_heth = NHeI * sigma_HeI_at_ion_freq

    # opt depth of H at he+Lya (40.817eV)
    sigma_H_heLya = 9.907e-22  # HI cross-section at HeII Lya
    tau_H_heLya = NHI * sigma_H_heLya

    # opt depth of He at he+Lya (40.817eV)
    sigma_He_heLya = 1.301e-20
    tau_He_heLya = NHeI * sigma_He_heLya

    # opt depth of H at HeII ion threshold
    sigma_H_he2 = 1.230695924714239e-19  # HI cross-section at HeII ionization threshold
    tau_H_he2th = NHI * sigma_H_he2

    # opt depth of HeI at HeII ion threshold
    sigma_He_he2 = (
        1.690780687052975e-18  # HeI cross-section at HeII ionization threshold
    )
    tau_He_he2th = NHeI * sigma_He_he2

    # opt depth of HeII at HeII ion threshold
    sigma_HeII_at_ion_freq = 1.589e-18  # HeII cross section at its ionzing frequency
    tau_He2_he2th = NHeII * sigma_HeII_at_ion_freq

    # Ratios of these optical depths needed in doric
    y = tau_H_heth / (tau_H_heth + tau_He_heth)
    z = tau_H_heLya / (tau_H_heLya + tau_He_heLya)
    y2a = tau_He2_he2th / (tau_He2_he2th + tau_He_he2th + tau_H_he2th)
    y2b = tau_He_he2th / (tau_He2_he2th + tau_He_he2th + tau_H_he2th)

    # Fraction of photons from recombination of HeII that ionize HeI (pag 32 of Kai Yan Lee's thesis)
    p = 0.96

    # fraction of photons from 2-photon decay, energetic enough to ionize hydrogen
    ll = 1.425

    # fraction of photons from 2-photon decay, energetic enough to ionize neutral helium
    m = 0.737

    # "escape” fraction of Ly α photons, it depends on the neutral fraction
    f_lya = 1

    # Collisional ionization process (Eq. 2.21-23)
    cHI = 5.835e-11 * np.sqrt(temp_p) * np.exp(-157804 / temp_p)
    cHeI = 2.71e-11 * np.sqrt(temp_p) * np.exp(-285331 / temp_p)
    cHeII = 5.707e-12 * np.sqrt(temp_p) * np.exp(-631495 / temp_p)

    # Photo-ionization rates (Eq. 2.27-29)
    uHI = phi_HI + cHI * n_e
    uHeI = phi_HeI + cHeI * n_e
    uHeII = phi_HeII + cHeII * n_e

    # Recombination rate (Eq. 2.30-35)
    rHII2HI = -alphB_HII
    rHeII2HI = p * alphA_HeII + y * alph1_HeIII
    rHeII2HeI = (1 - y) * alph1_HII - alphA_HeII
    rHeIII2HI = (
        (1 - y2a - y2b) * alph1_HeIII
        + alph2_HeIII
        + (nu * (ll - m + m * y) + (1 - nu) * f_lya * z) * alphB_HeIII
    )
    rHeIII2HeI = (
        y2b * alph1_HeIII
        + (nu * m * (1 - y) + (1 - nu) * f_lya * (1 - z)) * alphB_HeIII
        + alphA_HeIII
        - y2a * alph1_HeIII
    )
    rHeIII2HeII = y2a * alph1_HeIII - alphA_HeIII

    # get matrix
    A11 = -uHI + rHII2HI
    A12 = 0.0
    A13 = 0.0
    A21 = Y / X * rHeII2HI * n_e
    A22 = -uHeI - uHeII + rHeII2HeI * n_e
    A23 = uHeII
    A31 = Y / X * rHeIII2HI * n_e
    A32 = -uHeI + rHeIII2HeI * n_e
    A33 = rHeIII2HeII * n_e

    # g = [uHI, uHeI, 0]

    S = np.sqrt(A33**2 - 2 * A33 * A22 + A22**2 + 4 * A32 * 23)
    K = 1 / (A23 * A32 - A33 * A22)
    R = 2 * A23 * (A33 * uHI * K - xHeII_old)
    T = -A32 * uHeI * K - xHeIII_old

    lamb1 = A11
    lamb2 = 0.5 * (A33 + A22 - S)
    lamb3 = 0.5 * (A33 + A22 + S)

    p1 = -(uHI + (A33 * A12 - A32 * A13) * uHeI * K) / A11
    # p2 = A33 * uHeI * K
    # p3 = -A32 * uHeI * K

    B11 = 1.0
    B12 = (-2 * A32 * A13 + A12 * (A33 - A22 + S)) / (2 * A32 * (A11 - lamb2))
    B13 = (-2 * A32 * A13 + A12 * (A33 - A22 - S)) / (2 * A32 * (A11 - lamb3))
    B21 = 0.0
    B22 = (-A33 + A22 - S) / (2 * A32)
    B23 = (-A33 + A22 + S) / (2 * A32)
    B31 = 0.0
    B32 = 1.0
    B33 = 1.0

    c1 = (
        (2 * p1 * S - (R + (A33 - A22) * T) * (A21 - A31)) / 2 * S
        + xHII_old
        + T / 2 * (A21 + A31)
    )
    c2 = (R + (A33 - A22 - S) * T) / (2 * S)
    c3 = -(R + (A33 - A22 + S) * T) / (2 * S)

    xHII_av = (
        B11 * c1 / (lamb1 * dt) * (np.exp(lamb1 * dt) - 1.0)
        + B12 * c2 / (lamb2 * dt)(np.exp(lamb2 * dt) - 1.0)
        + B13 * c3 / (lamb3 * dt) * (np.exp(lamb3 * dt) - 1.0)
    )
    # xHI_av = 1.0 - xHII_av
    xHeII_av = (
        B21 * c1 / (lamb1 * dt) * (np.exp(lamb1 * dt) - 1.0)
        + B22 * c2 / (lamb2 * dt) * (np.exp(lamb2 * dt) - 1.0)
        + B23 * c3 / (lamb3 * dt) * (np.exp(lamb3 * dt) - 1.0)
    )
    xHeIII_av = (
        B31 * c1 / (lamb1 * dt) * (np.exp(lamb1 * dt) - 1.0)
        + B32 * c2 / (lamb2 * dt) * (np.exp(lamb2 * dt) - 1.0)
        + B33 * c3 / (lamb3 * dt) * (np.exp(lamb3 * dt) - 1.0)
    )
    # xHeI_av = 1 - xHeII_av - xHeIII_av

    # TODO: here after there should be the heating part
    # (from eq 2.69 in Kay Lee thesis, pag 37)

    return xHII_av, xHeII_av, xHeIII_av
