import sys, os, time
import numpy as np
import pyc2ray as pc2r
import astropy.units as u
# ======================================================================
# Example for pyc2ray: Cosmological simulation from N-body
# ======================================================================

# Global parameters
num_steps_between_slices = 2        # Number of timesteps between redshift slices
paramfile = sys.argv[1]             # Name of the parameter file

# Create C2Ray object
sim = pc2r.C2Ray_fstar(paramfile=paramfile)

# copy parameter file into the output directory
os.system('cp %s %s' %(paramfile, sim.results_basename))

# Get redshift list (test case)
idx_zred, zred_array = np.loadtxt(sim.inputs_basename+'redshift_checkpoints.txt', dtype=float, unpack=True)

# check for resume simulation
if(sim.resume):
    i_start = np.argmin(np.abs(zred_array - sim.zred))
    sim.resume = i_start+1
else:
    i_start = 0

# Measure time
tinit = time.time()

# Loop over redshifts
for k in range(i_start, len(zred_array)-1):

    zi = zred_array[k]       # Start redshift
    zf = zred_array[k+1]     # End redshift

    pc2r.printlog(f"\n=================================", sim.logfile)
    pc2r.printlog(f"Doing redshift {zi:.3f} to {zf:.3f}", sim.logfile)
    pc2r.printlog(f"=================================\n", sim.logfile)

    # Compute timestep of current redshift slice
    dt = sim.set_timestep(zi, zf, num_steps_between_slices)

    # Read input files
    sim.read_density(fbase='CDM_200Mpc_2048.%05d.den.256.0' %idx_zred[k], z=zi)


    #TODO: move this after time-loop and change for zf
    if(sim.rank == 0 and k != i_start):    
        # Write output
        sim.write_output(z=zi, ext='.npy')

    # Set redshift to current slice redshift
    sim.zred = zi

    # Loop over timesteps
    for t in range(num_steps_between_slices):
        # Read source files (for instanteneous busrty star-formation model we require small time-step)
        t_age = sim.cosmology.age(zi).cgs.value + t*dt
        z = sim.time2zred(t_age)
        srcpos, normflux = sim.ionizing_flux(file='CDM_200Mpc_2048.%05d.fof.txt' %idx_zred[k], z=z, dt=dt) #, save_Mstar=sim.results_basename+'/sources')


        tnow = time.time()
        pc2r.printlog(f"\n --- Timestep {t+1:n}. Redshift: z = {sim.zred : .3f} Wall clock time: {tnow - tinit : .3f} seconds --- \n", sim.logfile)

        # Evolve Cosmology: increment redshift and scale physical quantities (density, proper cell size, etc.)
        sim.cosmo_evolve(dt)

        # Evolve the simulation: raytrace -> photoionization rates -> chemistry -> until convergence
        if(np.sum(srcpos) == 0 and np.sum(normflux) == 0):
            pass
        else:
            sim.evolve3D(dt, normflux, srcpos)

        if(sim.rank == 0):
            # rank 0 write an additional summary output file
            summary_exist = os.path.exists(sim.results_basename+'PhotonCounts.txt')
            with open(sim.results_basename+'PhotonCounts.txt', 'a') as f:
                if not (summary_exist):
                    header = '#z\tt [Myr]\ttot HI atoms\ttot phots\tbursty SFR [%]\tmean ndens [1/cm3]\tmean Irate [1/s]\tR_mfp [cMpc]\tmean ionization fraction (by volume and mass)\n'
                    f.write(header)
                    
                tot_nHI = np.sum(sim.ndens * (1-sim.xh) * sim.dr**3)
                massavrg_ion_frac = np.sum(sim.xh*sim.ndens)/np.sum(sim.ndens)
                t_age = (sim.cosmology.age(zi) + t*dt*u.s).to('Myr').value
                text = '%.3f\t%.2f\t%.3e\t%.3e\t%.2f\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n' %(z, t_age, tot_nHI, sim.tot_phots, sim.perc_switchon, np.mean(sim.ndens), np.mean(sim.phi_ion), sim.R_max_LLS/sim.N*sim.boxsize, np.mean(sim.xh), massavrg_ion_frac)
                f.write(text)

    # Evolve cosmology over final half time step to reach the correct time for next slice (see note in c2ray_base.py)
    #sim.cosmo_evolve(0)
    sim.cosmo_evolve_to_now()

# Write final output
sim.write_output(zf)
