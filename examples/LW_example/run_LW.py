import os
import sys
import astropy.units as u
import numpy as np
from astropy import constants as c
import pyc2ray as pc2r

# ======================================================================
# Example for pyc2ray: Cosmological simulation with Lyman-Werner feedback
# ======================================================================
# Global parameters
paramfile = sys.argv[1]  # Name of the parameter file
num_outputs_per_slice = 1  # Number of outputs per redshift slice
nz0 = 0  # starting redshift index

# ======================================================================
# Create C2Ray object
# ======================================================================
sim = pc2r.C2Ray_CubeP3M_LW( paramfile=paramfile )
# Copy parameter file into the output directory
if sim.rank == 0:
    os.system( "cp %s %s" % ( paramfile, sim.results_basename, ) )

# ======================================================================
# Load redshift arrays
# ======================================================================
zred_fine = np.loadtxt( sim.inputs_basename + "redshifts_fine.dat", dtype=float, skiprows=1, )
zred_coarse = np.loadtxt( sim.inputs_basename + "redshifts_checkpoints.txt", dtype=float, skiprows=1, )
# Store as instance variables for use in class methods
sim.zred_array = zred_fine
sim.zred_array_coarse = zred_coarse


# ======================================================================
# Initialize LW constants
# ======================================================================
if hasattr(sim, "_init_jLW_constants"):
    sim._init_jLW_constants()


# ======================================================================
# Coarse ACH/LMACH source tracking
# ======================================================================
#
# ACH/LMACH catalogues are only updated at coarse checkpoints.
#
# Between coarse checkpoints:
#
#   * HMACH positions remain frozen
#   * LMACH positions remain frozen
#   * HMACH luminosities remain frozen
#   * LMACH suppression state remains frozen
#   * normflux_ach remains frozen
#
# Only the minihalo population is updated at every fine timestep.
#
# This is intentional:
#
#   ACH lifetime  -> coarse source interval
#   MH lifetime   -> fine interval
#
# The existing /6 photon-budget convention is retained in
# C2Ray_CubeP3M_LW:
#
#     phot_per_atom = [10/6, 150/6, 833]
#
# and similarly:
#
#     Ni = [6000/6, 50000/6, 55000]
#
# ======================================================================
active_coarse_idx = None
srcpos_ach = None
normflux_ach = None
# The causal-source selection below assumes that coarse redshifts
# decrease with cosmic time, e.g.
#
#     30.000, 27.900, 25.854, ...
#
if ( len(zred_coarse) > 1 and not np.all( np.diff(zred_coarse) < 0.0 ) ):
    raise ValueError( "redshifts_checkpoints.txt must be strictly " "decreasing in redshift for the causal " "coarse-source selection used here." )

# ======================================================================
# Timer
# ======================================================================
timer = pc2r.Timer()
timer.start()

# ======================================================================
# Initialize LW intensity
# ======================================================================
sim.jLW = np.zeros( sim.shape, order="F", )
# ======================================================================

# Initialize previous density tracking
# ======================================================================
sim.prev_ndens = None
sim.dens_ND = None
sim.dens_ND_prev = None
# ======================================================================
# Initialize simulation time
# ======================================================================
sim_time = 0.0


# ======================================================================
# Main loop over FINE redshifts
# ======================================================================
for k in range( len(zred_fine) - 1 ):
    zi = zred_fine[k]
    zf = zred_fine[k + 1]
    sim.printlog( "-------------------------------------", sim.logfile, )
    sim.printlog( f"Doing redshift: " f"{zi:.3f} to {zf:.3f}", sim.logfile, )
    sim.printlog( "-------------------------------------", sim.logfile, )

    # Set current simulation redshift
    sim.zred = zi

    # Calculate timestep
    dt = sim.set_timestep( zi, zf, 1, )
    # Cosmic times
    start_time = sim.zred2time( zi )
    end_time = sim.zred2time( zf )
    # First iteration
    if k == 0:
        sim_time = start_time
    # ==================================================================
    # Output interval
    # ==================================================================
    output_time = ( end_time - start_time ) / num_outputs_per_slice
    # ==================================================================
    # Fine-step source lifetime
    # ==================================================================
    #
    # For MHs this is their actual source lifetime.
    #
    # For ACHs we retain the existing /6 convention:
    #
    #     phot_per_atom_HMACH = 10/6
    #     phot_per_atom_LMACH = 150/6
    #
    # The ACH population itself is frozen over the full coarse interval.
    #
    # ==================================================================
    AGlifetime = ( end_time - sim_time )
    sim.printlog( f"Time: " f"{sim_time:.3e} s " f"to {end_time:.3e} s", sim.logfile, )
    sim.printlog( f"Timestep dt = " f"{dt:.3e} s", sim.logfile, )
    sim.printlog( f"Output interval = " f"{output_time:.3e} s", sim.logfile, )
    # ==================================================================
    # Density is still updated every FINE step
    # ==================================================================
    #
    # Store the previous normalized density field before reading
    # the current one.
    #
    # This is required by update_agrid_properties(), which compares
    # current and previous MH abundances.
    #
    # ==================================================================
    if sim.dens_ND is not None:
        sim.dens_ND_prev = ( sim.dens_ND.copy() )
    else:
        sim.dens_ND_prev = np.zeros( sim.shape, order="F", )
    # Read density corresponding to the current FINE redshift
    sim.read_density( z=zi )
    # ==================================================================
    # Find the CAUSAL coarse ACH snapshot
    # ==================================================================
    valid = np.where( zred_coarse >= zi - 1e-4 )[0]
    if len(valid) == 0:
        raise RuntimeError( "No causal coarse source " f"snapshot for z={zi:.3f}" )
    # Since zred_coarse is decreasing,
    # the final valid index is the closest
    # causal source snapshot.
    coarse_idx = int( valid[-1] )
    z_coarse_used = float( zred_coarse[ coarse_idx ] )

    
    # ==================================================================
    # Safety check: prevent look-ahead
    # ==================================================================
    if ( z_coarse_used < zi - 1e-4 ):
        raise RuntimeError( "NON-CAUSAL ACH SOURCE SNAPSHOT: " f"fine z={zi:.3f}, " f"selected coarse " f"z={z_coarse_used:.3f}" )
    # ==================================================================
    # ONLY update ACH/LMACH population when coarse snapshot changes
    # ==================================================================
    if ( active_coarse_idx != coarse_idx ):
        sim.printlog( f"COARSE ACH UPDATE: " f"fine z={zi:.3f}, " f"source catalogue " f"z={z_coarse_used:.3f}", sim.logfile, )
        # ==============================================================
        # Read HMACH/LMACH catalogue ONCE
        # ==============================================================
        #
        # read_sources() does several things:
        #
        #   1. reads the coarse ACH catalogue
        #
        #   2. creates self.sM00_msun
        #
        #   3. creates self.sM01_msun
        #
        #   4. evaluates LMACH suppression using the CURRENT xHII
        #
        #   5. modifies self.sM01_msun for suppressed LMACHs
        #
        #   6. calculates normflux_ach
        #
        #
        # We only want those operations to occur ONCE per coarse
        # interval.
        #
        # The resulting:
        #
        #     srcpos_ach
        #     normflux_ach
        #     self.sM00_msun
        #     self.sM01_msun
        #
        # then remain frozen until the next coarse catalogue.
        #
        # ==============================================================
        srcpos_ach, normflux_ach = ( sim.read_sources( source_lifetime=AGlifetime, file=( f"{sim.inputs_basename}" f"src/" f"{z_coarse_used:.3f}" f"-coarsened_sources.dat" ), mass="hm", ) )
        # Remember which coarse catalogue is active
        active_coarse_idx = ( coarse_idx )
    else:
        # ==============================================================
        # Fine timestep:
        #
        # DO NOT reread ACH/LMACH sources.
        # ==============================================================
        sim.printlog( f"FINE MH STEP: " f"z={zi:.3f}; " f"ACH/LMACH population " f"frozen from " f"z={z_coarse_used:.3f}", sim.logfile, )
        # --------------------------------------------------------------
        # IMPORTANT
        # --------------------------------------------------------------
        #
        # Do NOT call:
        #
        #     sim.read_sources(...)
        #
        # here.
        #
        # Therefore:
        #
        #     srcpos_ach
        #     normflux_ach
        #     self.sM00_msun
        #     self.sM01_msun
        #
        # all remain exactly as they were at the coarse update.
        #
        # In particular, LMACHs are NOT re-suppressed by the radiation
        # generated by themselves during the following fine timesteps.
        #
        # --------------------------------------------------------------
    # ==================================================================
    # Set clumping
    # ==================================================================
    if hasattr( sim, "set_clumping", ):
        sim.set_clumping( zi )
    # ==================================================================
    # Calculate MINHALO properties
    # ==================================================================
    #
    # Unlike ACHs, MHs ARE updated at every fine timestep.
    #
    # The LW field used here is the field computed at the END of the
    # previous fine timestep.
    #
    # ==================================================================
    if ( sim.MHflag == 1 or sim.MHflag == 2 ):
        subgrid_pos, subgrid_flux, num_subgrid = ( sim.update_agrid_properties( k, AGlifetime, sim.jLW, ) )
        if ( num_subgrid is not None and num_subgrid > 0 ):
            sim.printlog( "Number of active " f"subgrid sources: " f"{num_subgrid}", sim.logfile, )
            sim.printlog( "Subgrid Source " f"lifetime: " f"{AGlifetime / 3.1536e13}", sim.logfile, )
            sim.printlog( "Subgrid Total flux: " f"{np.sum(subgrid_flux)}", sim.logfile, )
            # update_agrid_properties returns
            #
            #     (Nsource, 3)
            #
            # but evolve3D expects:
            #
            #     (3, Nsource)
            #
            srcpos_mh = ( subgrid_pos.T )
            normflux_mh = ( subgrid_flux )
        else:
            srcpos_mh = np.array( [] ).reshape( 3, 0, )
            normflux_mh = np.array( [] )
    else:
        # ==============================================================
        # File-based minihalos
        # ==============================================================
        srcpos_mh, normflux_mh = ( sim.read_sources( file=( f"{sim.inputs_basename}" f"src/" f"{zi:.3f}" f"-minihalos.dat" ), mass="mh", ) )
    # ==================================================================
    # Construct LW source distribution
    # ==================================================================
    #
    # ACH population:
    #
    #     FROZEN over coarse interval
    #
    # MH population:
    #
    #     NEW at every fine interval
    #
    #
    # get_srclumK() still runs at every fine timestep because the frozen
    # ACH population contributes to each fine LW light-cone slice.
    #
    # ==================================================================
    srclumK = sim.get_srclumK( k, srcpos_ach, normflux_ach, srcpos_mh, normflux_mh, )
    if sim.rank == 0:
        sim.save_srclumK(srclumK, zi)

    sim.comm.Barrier()
    
    sim.printlog( f"Saved source distribution " f"at z={zi:.3f}", sim.logfile, )
    # ==================================================================
    # Compute J_LW at NEXT fine redshift
    # ==================================================================
    #
    # Sources from all past slices within the LW horizon are included.
    #
    # ==================================================================
    sim.jLW = ( sim.compute_jLW_from_history( nz0, k, ) )
    sim.printlog( f"Computed jLW at " f"z={zf:.3f} " f"for next iteration", sim.logfile, )
    sim.printlog( f"  mean=" f"{np.mean(sim.jLW):.3e}, " f"max=" f"{np.max(sim.jLW):.3e}", sim.logfile, )
    # ==================================================================
    # Merge ACH and MH populations for ionizing RT
    # ==================================================================
    if ( srcpos_ach is not None and srcpos_ach.size > 0 ):
        if ( srcpos_mh.size > 0 ):
            srcpos = np.concatenate( [ srcpos_ach, srcpos_mh, ], axis=1, )
            normflux = np.concatenate( [ normflux_ach, normflux_mh, ] )
        else:
            srcpos = ( srcpos_ach )
            normflux = ( normflux_ach )
    else:
        srcpos = ( srcpos_mh )
        normflux = ( normflux_mh )
    num_sources = ( normflux.size if normflux is not None else 0 )
    sim.printlog( "Total number of sources " f"(after merge): " f"{num_sources}", sim.logfile, )
    # ==================================================================
    # Inner time loop
    # ==================================================================
    next_output_time = ( sim_time + output_time )
    while ( sim_time < end_time ):
        # ==============================================================
        # Actual RT timestep
        # ==============================================================
        actual_dt = min( next_output_time - sim_time, dt, )
        z_now = sim.time2zred( sim_time )
        print( "actual_dt: ", actual_dt, )
        sim.printlog( f"Time: " f"{sim_time:.3e} s, " f"dt: " f"{actual_dt:.3e} s, " f"z: " f"{z_now:.3f}", sim.logfile, )
        # ==============================================================
        # Cosmological evolution
        # ==============================================================
        sim.cosmo_evolve( actual_dt )
        # ==============================================================
        # Update clumping
        # ==============================================================
        if ( hasattr( sim, "type_of_clumping", ) and sim.type_of_clumping != 5 ):
            if hasattr( sim, "set_clumping", ):
                sim.set_clumping( z_now )
        # ==============================================================
        # Evolve ionizing radiation field
        # ==============================================================
        if ( num_sources > 0 ):
            sim.evolve3D( actual_dt, normflux, srcpos, )
        # Advance simulation time
        sim_time += ( actual_dt )
        # ==============================================================
        # Write output
        # ==============================================================
        if ( abs( sim_time - next_output_time ) <= 1e-6 * sim_time ):
            z_output = ( sim.time2zred( sim_time ) )
            sim.printlog( f"Writing output at " f"z={z_output:.3f}, " f"t={sim_time:.3e} s", sim.logfile, )
            # ==========================================================
            # Summary output
            # ==========================================================
            if ( sim.rank == 0 ):
                summary_file = ( sim.results_basename + "PhotonCounts.txt" )
                summary_exist = ( os.path.exists( summary_file ) )
                with open( summary_file, "a", ) as f:
                    if ( not summary_exist ):
                        header = ( "#z\t" "t [Myr]\t" "tot HI atoms\t" "mean ndens [1/cm3]\t" "mean Irate [1/s]\t" "mean xfrac (vol)\t" "mean xfrac (mass)\n" )
                        f.write( header )
                    tot_nHI = np.sum( sim.ndens * ( 1 - sim.xh ) * sim.dr**3 )
                    massavrg_ion_frac = ( np.sum( sim.xh * sim.ndens ) / np.sum( sim.ndens ) )
                    t_age_myr = ( sim.cosmology .age( z_output ) .to( "Myr" ) .value )
                    text = ( "%.3f\t" "%.2f\t" "%.3e\t" "%.3e\t" "%.3e\t" "%.3e\t" "%.3e\n" % ( z_output, t_age_myr, tot_nHI, np.mean( sim.ndens ), np.mean( sim.phi_ion ), np.mean( sim.xh ), massavrg_ion_frac, ) )
                    f.write( text )
            # ==========================================================
            # Full 3-D output
            # ==========================================================
            if hasattr( sim, "write_output", ):
                sim.write_output( z_output, ext=".npy", )
            next_output_time += ( output_time )
        # ==============================================================
        # End-of-redshift interval
        # ==============================================================
        if ( abs( sim_time - end_time ) <= 1e-6 * end_time ):
            break
    # ==================================================================
    # Move cosmological quantities to end of interval
    # ==================================================================
    sim.cosmo_evolve_to_now()
    sim.printlog( "Completed redshift interval " f"{zi:.3f} to {zf:.3f}", sim.logfile, )
# ======================================================================
# Finish simulation
# ======================================================================
timer.stop()
sim.printlog( timer.summary, sim.logfile, )
# ======================================================================
# Final output
# ======================================================================
z_final = ( zred_fine[-1] )
if hasattr( sim, "write_output", ):
    sim.write_output( z_final, ext=".npy", )
sim.printlog( "Simulation completed successfully", sim.logfile, )