import numpy as np

# Constants
epsilon = 1e-14
minimum_fractional_change = 1.0e-3
minimum_fraction_of_atoms = 1.0e-8

def global_pass(dt, ndens, temp, xfrac, xfrac_av, xfrac_intermed, irate, clumping, bh00, albpow, colh0, temph0, abu_c):
    conv_flag = 0
    xh_new = np.zeros_like(xfrac)

    m1, m2, m3 = xfrac.shape   
    for k in range(m3):
        for j in range(m2):
            for i in range(m1):
                pos = (i, j, k)
                clump = clumping[i,j,k]
                phi_ion = irate[i,j,k]
                xh = xfrac[i,j,k]
                xh_av = xfrac_av[i,j,k]
                xh_intermed = xfrac_intermed[i,j,k]
                a = evolve0D_global(dt, pos, ndens, temp, xh, xh_av, xh_intermed, phi_ion, clump, bh00, albpow, colh0, temph0, abu_c, conv_flag, m1, m2, m3)
                print(a)
                xh_new[i,j,k] = a
    return xh_new

def evolve0D_global(dt, pos, ndens, temp, xh, xh_av, xh_intermed, phi_ion, clump, bh00, albpow, colh0, temph0, abu_c, conv_flag, m1, m2, m3):
    i, j, k = pos
    temperature_start = temp #[i, j, k]
    ndens_p = ndens #[i, j, k]
    phi_ion_p = phi_ion #[i, j, k]
    clump_p = clump #[i, j, k]
    
    xh_p = xh #[i, j, k]
    xh_av_p = xh_av #[i, j, k]
    xh_intermed_p = xh_intermed #[i, j, k]
    yh_av_p = 1.0 - xh_av_p
    
    xh_av = do_chemistry(dt, ndens_p, temperature_start, xh_p, xh_av_p, xh_intermed_p, phi_ion_p, clump_p, bh00, albpow, colh0, temph0, abu_c)
    
    xh_av_p_old = xh_av #[i, j, k]
    if ((abs(xh_av_p - xh_av_p_old) > minimum_fractional_change and abs((xh_av_p - xh_av_p_old) / yh_av_p) > minimum_fractional_change and 
         yh_av_p > minimum_fraction_of_atoms)):
        conv_flag += 1

    #xh_intermed[i, j, k] = xh_intermed_p
    xh_intermed = xh_intermed_p
    #xh_av[i, j, k] = xh_av_p
    return xh_intermed

def do_chemistry(dt, ndens_p, temperature_start, xh_p, xh_av_p, xh_intermed_p, phi_ion_p, clump_p, bh00, albpow, colh0, temph0, abu_c):
    temperature_end = temperature_start
    nit = 0
    
    while True:
        nit += 1
        temperature_previous_iteration = temperature_end
        xh_av_p_old = xh_av_p

        de = ndens_p * (xh_av_p + abu_c)
        xh_av_p = doric(xh_p, dt, temperature_end, de, phi_ion_p, bh00, albpow, colh0, temph0, clump_p, xh_intermed_p, xh_av_p)
        
        if ((abs((xh_av_p - xh_av_p_old) / (1.0 - xh_av_p)) < minimum_fractional_change or 
             (1.0 - xh_av_p < minimum_fraction_of_atoms)) and 
            (abs((temperature_end - temperature_previous_iteration) / temperature_end) < minimum_fractional_change)):
            break

        if nit > 400:
            conv_flag += 1
            break

def doric(xh_old, dt, temp_p, rhe, phi_p, bh00, albpow, colh0, temph0, clumping, xh, xh_av):
    brech0 = clumping * bh00 * (temp_p / 1e4) ** albpow
    sqrtt0 = np.sqrt(temp_p)
    acolh0 = colh0 * sqrtt0 * np.exp(-temph0 / temp_p)
    aphoth0 = phi_p
    
    aih0 = aphoth0 + rhe * acolh0
    delth = aih0 + rhe * brech0
    eqxh = aih0 / delth
    deltht = delth * dt
    ee = np.exp(-deltht)
    
    xh = (xh_old - eqxh) * ee + eqxh
    if xh < epsilon:
        xh = epsilon
    
    if deltht < 1.0e-8:
        avg_factor = 1.0
    else:
        avg_factor = (1.0 - ee) / deltht
    
    xh_av = eqxh + (xh_old - eqxh) * avg_factor
    if xh_av < epsilon:
        xh_av = epsilon


def friedrich(NH, NHe, n_e, phi_HI, phi_HeI, phi_HeII, temp):
    # Recombination rate of HI (Eq. 2.12 and 2.13)
    alphA_HII = 1.269e-13 * np.power(315608/temp, 1.503) / np.power(np.power(1 + 604613/temp, 0.47), 1.923)
    alphB_HII = 2.753e-14 * np.power(315608/temp, 1.5) / np.power(np.power(1 + 115185/temp, 0.407), 2.242)
    alph1_HII = alphA_HII - alphB_HII

    # Recombination rate of HeII (Eq. 2.14-17)
    if(temp < 9e3):
        alphA_HeII = 1.269e-13 * np.power(570662/temp, 1.503) / np.power(np.power(1 + 1093222/temp, 0.47), 1.923)
        alphB_HeII = 2.753e-14 * np.power(570662/temp, 1.5) / np.power(np.power(1 + 208271/temp, 0.407), 2.242)
    else:    
        alphA_HeII = 3e-14 * np.power(570662/temp, 0.654) + 1.9e-3 * np.power(temp, -1.5) * np.exp(-4.7e5/temp) * (1 + 0.3*np.exp(-9.4e4/temp))
        alphB_HeII = 1.26e-14 * np.power(570662/temp, 0.75) + 1.9e-3 * np.power(temp, -1.5) * np.exp(-4.7e5/temp) * (1 + 0.3*np.exp(-9.4e4/temp))

    # Recombination rate of HeIII (Eq. 2.18-20)
    alphA_HeIII = 2.538e-13 * np.power(1262990/temp, 1.503) / np.power(1 + np.power(2419521/temp, 1.923), 1.923)
    alphB_HeIII = 5.506e-14 * np.power(1262990/temp, 1.5) / np.power(1 + np.power(460945/temp, 0.407), 2.242)
    alph1_HeIII = alphA_HeIII - alphB_HeIII # correct???????????????
    alph2_HeIII = 8.54e-11 * np.power(temp, -0.6)

    # two photons emission from recombination of HeIII
    nu = 0.285 * np.power(temp/1e4, 0.119)

    # opt depth of HI at HeI ion threshold
    sigma_H_heth = 1.238e-18    # HI cross-section at HeI ionization threshold
    tau_H_heth  = NH*sigma_H_heth
    
    # opt depth of HeI at HeI ion threshold
    sigma_HeI_at_ion_freq = 7.430e-18   # HeI cross section at its ionzing frequency
    tau_He_heth = NHe(0)*sigma_HeI_at_ion_freq 
    
    # opt depth of H  at he+Lya (40.817eV)
    sigma_H_heLya = 9.907e-22   # HI cross-section at HeII Lya
    tau_H_heLya = NH*sigma_H_heLya
    
    # opt depth of He at he+Lya (40.817eV)
    sigma_He_heLya = 1.301e-20
    tau_He_heLya= NHe(0)*sigma_He_heLya
    
    # opt depth of H at HeII ion threshold
    sigma_H_he2 = 1.230695924714239e-19  # HI cross-section at HeII ionization threshold
    tau_H_he2th = NH*sigma_H_he2
    
    # opt depth of HeI at HeII ion threshold
    sigma_He_he2 = 1.690780687052975e-18    # HeI cross-section at HeII ionization threshold
    tau_He_he2th = NHe(0)*sigma_He_he2
    
    # opt depth of HeII at HeII ion threshold
    sigma_HeII_at_ion_freq = 1.589e-18  # HeII cross section at its ionzing frequency
    tau_He2_he2th = NHe(1)*sigma_HeII_at_ion_freq
    
    # Ratios of these optical depths needed in doric
    y = tau_H_heth /(tau_H_heth +tau_He_heth)
    z = tau_H_heLya/(tau_H_heLya+tau_He_heLya)
    y2a =  tau_He2_he2th /(tau_He2_he2th +tau_He_he2th+tau_H_he2th)
    y2b =  tau_He_he2th /(tau_He2_he2th +tau_He_he2th+tau_H_he2th)

    # Fraction of photons from recombination of HeII that ionize HeI (pag 32 of Kai Yan Lee's thesis)
    p = 0.96

    l, m = 1.425, 0.737


    # Collisional ionization process (Eq. 2.21-23)
    cHI = 5.835e-11 * np.sqrt(temp) * np.exp(-157804/temp)
    cHeI = 2.71e-11 * np.sqrt(temp) * np.exp(-285331/temp)
    cHeII = 5.707e-12 * np.sqrt(temp) * np.exp(-631495/temp)

    # Photo-ionization rates (Eq. 2.27-29)
    uHI = phi_HI + cHI * n_e
    uHeI = phi_HeI + cHeI * n_e
    uHeII = phi_HeII + cHeII * n_e
    



    # Recombination rate (Eq. 2.30-35)
    r_HII2HI = -alphB_HII
    r_HeI2HI = p*alphA_HeII + y*alp
    pass
    