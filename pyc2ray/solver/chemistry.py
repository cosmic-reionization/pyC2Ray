import numpy as np

# Constants
epsilon = 1e-14
minimum_fractional_change = 1.0e-3
minimum_fraction_of_atoms = 1.0e-8

def global_pass(dt, ndens, temp, xh, xh_av, xh_intermed, phi_ion, clump, bh00, albpow, colh0, temph0, abu_c, m1, m2, m3):
    conv_flag = 0
    
    for k in range(m3):
        for j in range(m2):
            for i in range(m1):
                pos = (i, j, k)
                conv_flag = evolve0D_global(dt, pos, ndens, temp, xh, xh_av, xh_intermed, phi_ion, clump, bh00, albpow, colh0, temph0, abu_c, conv_flag, m1, m2, m3)
    
    return conv_flag

def evolve0D_global(dt, pos, ndens, temp, xh, xh_av, xh_intermed, phi_ion, clump, bh00, albpow, colh0, temph0, abu_c, conv_flag, m1, m2, m3):
    i, j, k = pos
    temperature_start = temp[i, j, k]
    ndens_p = ndens[i, j, k]
    phi_ion_p = phi_ion[i, j, k]
    clump_p = clump[i, j, k]
    
    xh_p = xh[i, j, k]
    xh_av_p = xh_av[i, j, k]
    xh_intermed_p = xh_intermed[i, j, k]
    yh_av_p = 1.0 - xh_av_p
    
    conv_flag = do_chemistry(dt, ndens_p, temperature_start, xh_p, xh_av_p, xh_intermed_p, phi_ion_p, clump_p, bh00, albpow, colh0, temph0, abu_c)
    
    xh_av_p_old = xh_av[i, j, k]
    if ((abs(xh_av_p - xh_av_p_old) > minimum_fractional_change and 
         abs((xh_av_p - xh_av_p_old) / yh_av_p) > minimum_fractional_change and 
         yh_av_p > minimum_fraction_of_atoms)):
        conv_flag += 1

    xh_intermed[i, j, k] = xh_intermed_p
    xh_av[i, j, k] = xh_av_p
    return conv_flag

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
    return conv_flag

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
    return xh_av
