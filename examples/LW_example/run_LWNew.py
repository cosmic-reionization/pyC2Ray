import os
import sys

import astropy.units as u
import numpy as np

import pyc2ray as pc2r

# ======================================================================
# Example for pyc2ray: Cosmological simulation with Lyman-Werner feedback
# ======================================================================

# Global parameters
paramfile = sys.argv[1]  # Name of the parameter file
num_outputs_per_slice = 1  # Number of outputs per redshift slice (adjust as needed)

nz0 = 0 # starting redshift index

# Create C2Ray object
sim = pc2r.C2Ray_CubeP3M_LW(paramfile=paramfile)

# Copy parameter file into the output directory
if sim.rank == 0:
    os.system("cp %s %s" % (paramfile, sim.results_basename))

# Load both redshift arrays
zred_fine = np.loadtxt(sim.inputs_basename + 'redshifts_fine.dat', 
                       dtype=float, 
                       skiprows=1)
zred_coarse = np.loadtxt(sim.inputs_basename + 'redshifts_checkpoints.txt', 
                         dtype=float, 
                         skiprows=1)

# Store as instance variables for use in class methods
sim.zred_array = zred_fine
sim.zred_array_coarse = zred_coarse

# Initialize LW constants (IMPORTANT: Must be done before LW calculations)
if hasattr(sim, '_init_jLW_constants'):
    sim._init_jLW_constants()

# Initialization for coarse tracking
idx_coarse = 0
# Find starting coarse redshift that is >= first fine redshift
while idx_coarse < len(zred_coarse) and zred_coarse[idx_coarse] > zred_fine[0]:
    idx_coarse += 1

# Validate starting conditions (matching Fortran checks)
if zred_coarse[idx_coarse] < zred_fine[0]:
    sim.printlog("ERROR: Starting zred_coarse should be equal to or larger than zred_fine", sim.logfile)
    sys.exit(1)

# Start the timer to measure wall clock time
timer = pc2r.Timer()
timer.start()

# Initialize LW intensity to zero (matching Fortran: jLW = 0d0)
sim.jLW = np.zeros(sim.shape, order='F')

# Initialize previous density tracking
sim.prev_ndens = None
sim.dens_ND = None
sim.dens_ND_prev = None

# Initialize simulation time
sim_time = 0.0

# Main Loop over FINE redshifts (equivalent to nz in Fortran)
for k in range(len(zred_fine) - 1):
    zi = zred_fine[k]
    zf = zred_fine[k + 1]
    
    sim.printlog(f"-------------------------------------", sim.logfile)
    sim.printlog(f"Doing redshift: {zi:.3f} to {zf:.3f}", sim.logfile)
    sim.printlog(f"-------------------------------------", sim.logfile)
    
    # Set simulation redshift
    sim.zred = zi
    
    # Calculate timestep parameters for this redshift interval
    dt = sim.set_timestep(zi, zf, 1)
    
    # Calculate end_time (time at zf) and start time (time at zi)
    start_time = sim.zred2time(zi)
    end_time = sim.zred2time(zf)
    
    # Update sim_time if this is the first iteration
    if k == 0:
        sim_time = start_time
    
    # Calculate output time interval
    output_time = (end_time - start_time) / num_outputs_per_slice
    
    # AGlifetime is the total time for this redshift interval (for minihalo calculation)
    AGlifetime = end_time - sim_time
    sim.printlog(f"Time: {sim_time:.3e} s to {end_time:.3e} s", sim.logfile)
    sim.printlog(f"Timestep dt = {dt:.3e} s", sim.logfile)
    sim.printlog(f"Output interval = {output_time:.3e} s", sim.logfile)
    
    # ===================================================================
    # Logic for Coarse Redshift Update (Massive halos)
    # ===================================================================
    current_z_coarse = zred_coarse[idx_coarse]
    
    if abs(current_z_coarse - zi) < 1e-4:
        # COARSE UPDATE: Read new density and massive halo sources
        sim.printlog(f"COARSE UPDATE at z={zi:.3f}", sim.logfile)
        
        # Store previous density (dens_ND_prev) before reading new one
        if sim.dens_ND is not None:
            sim.dens_ND_prev = sim.dens_ND.copy()
        else:
            # Very first redshift
            sim.dens_ND_prev = np.zeros(sim.shape, order='F')
        
        # Read current density
        sim.read_density(z=zi)
        # Read massive halo sources at coarse redshift
        srcpos_ach, normflux_ach = sim.read_sources(
            file=f'{sim.inputs_basename}src/{zi:.3f}-coarsened_sources.dat', 
            source_lifetime = AGlifetime, mass='hm'
        )
        # Move to next coarse index for next iteration
        if idx_coarse < len(zred_coarse) - 1:
            idx_coarse += 1
            
    elif current_z_coarse < zi:
        # FINE STEP: Reuse massive halos from previous coarse snapshot
        sim.printlog(f"FINE STEP at z={zi:.3f} (using coarse data from z={zred_coarse[idx_coarse-1]:.3f})", sim.logfile)
        
        # Store previous density
        if sim.dens_ND is not None:
            sim.dens_ND_prev = sim.dens_ND.copy()
        
        # Read density at current fine redshift (density needs to be updated at each fine step)
        sim.read_density(z=zi)
        
        # Massive sources remain the same as from last coarse step (srcpos_ach, normflux_ach)
        srcpos_ach, normflux_ach = sim.read_sources(source_lifetime=AGlifetime,
            file=f'{sim.inputs_basename}src/{current_z_coarse:.3f}-coarsened_sources.dat', 
            mass='hm'
        )
        
    # Set clumping if needed
    if hasattr(sim, 'set_clumping'):
        sim.set_clumping(zi)
    
    # ===================================================================
    # Calculate minihalo properties using LW feedback from PREVIOUS iteration
    # This happens at EVERY fine redshift
    # IMPORTANT: Uses sim.jLW computed in the PREVIOUS iteration!
    # ===================================================================
    if sim.MHflag == 1 or sim.MHflag == 2:
        # Get active subgrid sources with LW suppression
        # Equivalent to Fortran: call AGrid_properties(nz, end_time-sim_time, jLW)

  
        subgrid_pos, subgrid_flux, num_subgrid = sim.update_agrid_properties(
            k, AGlifetime, sim.jLW  # Uses jLW from previous iteration!
        )

        if num_subgrid is not None and num_subgrid > 0:
            sim.printlog(f"Number of active subgrid sources: {num_subgrid}", sim.logfile)
            sim.printlog(f"Subgrid Source lifetime: {AGlifetime/3.1536e13}", sim.logfile)
            sim.printlog(f"Subgrid Total flux: {np.sum(subgrid_flux)}", sim.logfile)
            srcpos_mh = subgrid_pos.T  # Convert (N, 3) to (3, N)
            normflux_mh = subgrid_flux
        else:
            srcpos_mh = np.array([]).reshape(3, 0)
            normflux_mh = np.array([])
    else:
        # Read minihalos from file if not using dynamic calculation
        srcpos_mh, normflux_mh = sim.read_sources(
            file=f'{sim.inputs_basename}src/{zi:.3f}-minihalos.dat', 
            mass='mh'
        )
    
    # ===================================================================
    # CRITICAL: Compute and save source distribution at CURRENT redshift
    # This must happen BEFORE merging sources!
    # Pass SEPARATE massive and minihalo arrays
    # ===================================================================
    # get_srclumK needs separate massive and minihalo source arrays
    srclumK = sim.get_srclumK(
        k, 
        srcpos_ach, normflux_ach,  # Atomic Cooling Halos
        srcpos_mh, normflux_mh             # Minihalo sources
    )
    sim.save_srclumK(srclumK, zi)  # Save at current redshift zi
    sim.printlog(f"Saved source distribution at z={zi:.3f}", sim.logfile)
    
    # ===================================================================
    # Compute jLW at NEXT redshift (zf) for use in NEXT iteration
    # This uses sources from all past slices including current
    # Equivalent to Fortran: call get_jLW(nz0, nz)
    # ===================================================================
    
    # Compute jLW at zf using sources from all past slices
    sim.jLW = sim.compute_jLW_from_history(nz0, k)
    sim.printlog(f"Computed jLW at z={zf:.3f} for next iteration", sim.logfile)
    sim.printlog(f"  mean={np.mean(sim.jLW):.3e}, max={np.max(sim.jLW):.3e}", sim.logfile)
    
    # ===================================================================
    # Merge massive and minihalo sources for evolve3D
    # This happens AFTER get_srclumK
    # ===================================================================
    if srcpos_ach is not None and srcpos_ach.size > 0:
        if srcpos_mh.size > 0:
            srcpos = np.concatenate([srcpos_ach, srcpos_mh], axis=1)
            normflux = np.concatenate([normflux_ach, normflux_mh])
        else:
            srcpos = srcpos_ach
            normflux = normflux_ach
    else:
        srcpos = srcpos_mh
        normflux = normflux_mh
    
    num_sources = normflux.size if normflux is not None else 0
    sim.printlog(f"Total number of sources (after merge): {num_sources}", sim.logfile)
    
    # ===================================================================
    # Inner time loop - evolve from zi to zf in timesteps
    # Equivalent to Fortran: do ... enddo loop
    # ===================================================================
    
    # Set next output time
    next_output_time = sim_time + output_time
    
    # Loop until end_time is reached
    while sim_time < end_time:
        # Calculate actual timestep (can't exceed output time or end time)
        actual_dt = min(next_output_time - sim_time, dt)

        # Get current redshift for this timestep
        z_now = sim.time2zred(sim_time)
        print("actual_dt: ", actual_dt)
        sim.printlog(f"Time: {sim_time:.3e} s, dt: {actual_dt:.3e} s, z: {z_now:.3f}", sim.logfile)
        # Cosmological evolution (equivalent to redshift_evol and cosmo_evol)
        # Evolve to mid-point of timestep
        sim.cosmo_evolve(actual_dt)
        
        # Update clumping if needed (but not for position-dependent clumping)
        if hasattr(sim, 'type_of_clumping') and sim.type_of_clumping != 5:
            if hasattr(sim, 'set_clumping'):
                sim.set_clumping(z_now)
        
        # ===================================================================
        # Evolve radiation field - this is where ionization is calculated
        # Equivalent to Fortran: call evolve3D(actual_dt, iter_restart)
        # ===================================================================
        if num_sources > 0:
            sim.evolve3D(actual_dt, normflux, srcpos)
        # Update time
        sim_time += actual_dt
        
        # ===================================================================
        # Write output when next_output_time is reached
        # ===================================================================
        if abs(sim_time - next_output_time) <= 1e-6 * sim_time:
            z_output = sim.time2zred(sim_time)
            sim.printlog(f"Writing output at z={z_output:.3f}, t={sim_time:.3e} s", sim.logfile)
            
            if sim.rank == 0:
                summary_file = sim.results_basename + "PhotonCounts.txt"
                summary_exist = os.path.exists(summary_file)
                
                with open(summary_file, "a") as f:
                    if not summary_exist:
                        header = "#z\tt [Myr]\ttot HI atoms\tmean ndens [1/cm3]\tmean Irate [1/s]\tmean xfrac (vol)\tmean xfrac (mass)\n"
                        f.write(header)
                    
                    tot_nHI = np.sum(sim.ndens * (1 - sim.xh) * sim.dr**3)
                    massavrg_ion_frac = np.sum(sim.xh * sim.ndens) / np.sum(sim.ndens)
                    t_age_myr = (sim.cosmology.age(z_output)).to("Myr").value
                    
                    text = (
                        "%.3f\t%.2f\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n"
                        % (
                            z_output,
                            t_age_myr,
                            tot_nHI,
                            np.mean(sim.ndens),
                            np.mean(sim.phi_ion),
                            np.mean(sim.xh),
                            massavrg_ion_frac,
                        )
                    )
                    f.write(text)
            
            # Write full output files
            if hasattr(sim, 'write_output'):
                sim.write_output(z_output, ext=".npy")
            
            # Update next output time
            next_output_time += output_time
        
        # Check if we've reached the end time for this redshift interval
        if abs(sim_time - end_time) <= 1e-6 * end_time:
            break
    

    sim.cosmo_evolve_to_now()

    sim.printlog(f"Completed redshift interval {zi:.3f} to {zf:.3f}", sim.logfile)
# Stop timer and print summary
timer.stop()
sim.printlog(timer.summary, sim.logfile)

# Write final output
z_final = zred_fine[-1]
if hasattr(sim, 'write_output'):
    sim.write_output(z_final, ext=".npy")

sim.printlog("Simulation completed successfully", sim.logfile)