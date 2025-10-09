import time

import numpy as np

from .asora_core import cuda_is_init
from .load_extensions import load_asora, load_c2ray
from .utils import printlog
from .utils.sourceutils import format_sources

# Load extension modules
libc2ray = load_c2ray()
libasora = load_asora()

__all__ = ["do_raytracing"]

# =========================================================================
# This file contains the standalone raytracing subroutine, which may be
# useful for benchmarking or possibly other applications. When doing a full
# reionization simulation, please prefer the evolve3D subroutine, which
# includes all steps (raytracing, ODE solving, convergence checks)
# internally. This subroutine is only meant for specific use-cases.
#
# Raytracing can use either the sequential (subbox, cubic) technique which
# runs in Fortran on the CPU or the accelerated technique, which runs using
# the ASORA library on the GPU.
#
# When using the latter, some notes apply:
# For performance reasons, the program minimizes the frequency at which
# data is moved between the CPU and the GPU (this is a big bottleneck).
# In particular, the radiation tables, which in principle shouldn't change
# over the run of a simulation, need to be copied separately to the GPU
# using the photo_table_to_device() method of the module.
# =========================================================================


def do_raytracing(
    dr,
    src_flux,
    src_pos,
    use_gpu,
    max_subbox,
    subboxsize,
    loss_fraction,
    ndens,
    xh_av,
    photo_thin_table,
    photo_thick_table,
    heat_thin_table,
    heat_thick_table,
    minlogtau,
    dlogtau,
    R_max_LLS,
    sig,
    logfile="pyC2Ray.log",
    quiet=False,
    stats=False,
):
    # Allow a call with GPU only if 1. the asora library is present and 2. the GPU memory has been allocated using device_init()
    if use_gpu and not cuda_is_init():
        raise RuntimeError(
            "GPU not initialized. Please initialize it by calling device_init(N)"
        )

    # Set some constant sizes
    NumSrc = src_flux.shape[0]  # Number of sources
    N = ndens.shape[0]  # Mesh size
    NumTau = photo_thin_table.shape[0]

    # When using GPU raytracing, data has to be reshaped & reformatted and copied to the device
    if use_gpu:
        # Format input data for the CUDA extension module (flat arrays, C-types,etc)
        xh_av_flat = np.ravel(xh_av).astype("float64", copy=True)
        ndens_flat = np.ravel(ndens).astype("float64", copy=True)
        srcpos_flat, normflux_flat = format_sources(src_pos, src_flux)

        # Copy positions & fluxes of sources to the GPU in advance
        libasora.source_data_to_device(srcpos_flat, normflux_flat, NumSrc)

        # Initialize Flat Column density & ionization rate arrays. These are used to store the
        # output of the raytracing module. TODO: python column density array is actually not needed ?
        coldensh_out_flat = np.ravel(np.zeros((N, N, N), dtype="float64"))
        phi_ion_flat = np.ravel(np.zeros((N, N, N), dtype="float64"))

        # Copy density field to GPU once at the beginning of timestep (!! do_all_sources assumes this !!)
        libasora.density_to_device(ndens_flat, N)
        printlog("Copied source data to device.", logfile, quiet)

    printlog(f"dr [Mpc]: {dr / 3.086e24:.3e}", logfile, quiet)
    printlog(
        f"Running on {NumSrc:n} source(s), total normalized ionizing flux: {src_flux.sum():.2e}",
        logfile,
        quiet,
    )
    printlog(
        f"Mean density (cgs): {ndens.mean():.3e}, Mean ionized fraction: {xh_av.mean():.3e}",
        logfile,
        quiet,
    )

    trt0 = time.time()
    printlog("Doing Raytracing...", logfile, quiet, " ")
    # Set rates to 0. When using ASORA, this is done internally by the library (directly on the GPU)
    if not use_gpu:
        phi_ion = np.zeros((N, N, N), order="F")
        phi_heat = np.zeros((N, N, N), order="F")
        coldensh_out = np.zeros((N, N, N), order="F")

    # Do the raytracing part for each source. This computes the cumulative ionization rate for each cell.
    if use_gpu:
        # Use GPU raytracing
        libasora.do_all_sources(
            R_max_LLS,
            coldensh_out_flat,
            sig,
            dr,
            ndens_flat,
            xh_av_flat,
            phi_ion_flat,
            NumSrc,
            N,
            minlogtau,
            dlogtau,
            NumTau,
        )
    else:
        # Use CPU raytracing with subbox optimization
        nsubbox, photonloss = libc2ray.raytracing.do_all_sources(
            src_flux,
            src_pos,
            max_subbox,
            subboxsize,
            coldensh_out,
            sig,
            dr,
            np.asfortranarray(ndens),
            xh_av,
            phi_ion,
            phi_heat,
            loss_fraction,
            photo_thin_table,
            photo_thick_table,
            heat_thin_table,
            heat_thick_table,
            minlogtau,
            dlogtau,
            R_max_LLS,
        )
    printlog(f"took {(time.time() - trt0): .1f} s.", logfile, quiet)

    # Since chemistry (ODE solving) is done on the CPU in Fortran, flattened CUDA arrays need to be reshaped
    if use_gpu:
        phi_ion = np.reshape(phi_ion_flat, (N, N, N))
    else:
        printlog(
            f"Average number of subboxes: {nsubbox / NumSrc:n}, Total photon loss: {photonloss:.3e}",
            logfile,
            quiet,
        )

    if stats and not use_gpu:
        return phi_ion, nsubbox, photonloss
    else:
        return phi_ion, phi_heat


# ==========
# Deprecated
# ==========
def do_all_sources(
    dr,
    src_flux,
    src_pos,
    r_RT,
    use_gpu,
    max_subbox,
    loss_fraction,
    ndens,
    xh_av,
    photo_thin_table,
    minlogtau,
    dlogtau,
    sig,
    stats=False,
):
    """Computes the global rates for all cells and all sources

    Warning: Calling this function with use_gpu = True assumes that the radiation
    tables have previously been copied to the GPU using photo_table_to_device()

    Parameters
    ----------
    dr : float
        Cell dimension in each direction in cm
    src_flux : 1D-array of shape (numsrc)
        Array containing the total ionizing flux of each source, normalized by S_star (1e48 by default)
    src_pos : 2D-array of shape (3,numsrc)
        Array containing the 3D grid position of each source, in Fortran indexing (from 1)
    r_RT : int
        Parameter which determines the size of the raytracing volume around each source:
        * When using CPU (cubic) RT, this sets the increment of the cubic region (subbox) that will be treated.
        Raytracing stops when either max_subbox is reached or when the photon loss is low enough. For example, if
        r_RT = 5, the size of the cube around the source will grow as 10^3, 20^3, ...
        * When using GPU (octahedral) RT with ASORA, this sets the size of the octahedron such that a sphere of
        radius r_RT fits inside the octahedron.

    use_gpu : bool
        Whether or not to use the GPU-accelerated ASORA library for raytracing.
    max_subbox : int
        Maximum subbox to raytrace when using CPU cubic raytracing. Has no effect when use_gpu is true
    loss_fraction : float
        Fraction of remaining photons below we stop ray-tracing (subbox technique). Has no effect when use_gpu is true
    ndens : 3D-array
        The hydrogen number density of each cell in cm^-3
    xh_av : 3D-array
        The ionized fraction of each cell to use for column density calculation. This is typically
        the time-averaged value over a timestep
    photo_thin_table : 1D-array
        Tabulated values of the integral ∫L_v*e^(-τ_v)/hv. When using GPU, this table needs to have been copied to the GPU
        in a separate (previous) step, using photo_table_to_device()
    minlogtau : float
        Base 10 log of the minimum value of the table in τ (excluding τ = 0)
    dlogtau : float
        Step size of the logτ-table
    sig : float
        Constant photoionization cross-section of hydrogen in cm^2.
    stats : bool
        When using CPU raytracing, return number of subboxes used and photon loss. Parameter has no effect
        when using GPU RT. Default is False

    Returns
    -------
    phi_ion : 3D-array
        Photoionization rate of each cell due to all sources
    nsubbox : int, only when stats=True and use_gpu=False
        Total number of subbox increments used across all sources
    photonloss : float, only when stats=True and use_gpu=False
        Flux of photons that leaves the subbox used for RT
    """
    # Allow a call with GPU only if 1. the asora library is present and 2. the GPU memory has been allocated using device_init()
    if use_gpu and not cuda_is_init():
        raise RuntimeError(
            "GPU not initialized. Please initialize it by calling device_init(N)"
        )

    # Set some constant sizes
    NumSrc = src_flux.shape[0]  # Number of sources
    N = xh_av.shape[0]  # Mesh size
    NumTau = photo_thin_table.shape[0]  # Number of radiation table points

    # When using GPU raytracing, data has to be reshaped & reformatted and copied to the device
    if use_gpu:
        # Format input data for the CUDA extension module (flat arrays, C-types,etc)
        xh_av_flat = np.ravel(xh_av).astype("float64", copy=True)
        ndens_flat = np.ravel(ndens).astype("float64", copy=True)
        srcpos_flat, normflux_flat = format_sources(src_pos, src_flux)

        # Initialize Flat Column density & ionization rate arrays. These are used to store the
        # output of the raytracing module. TODO: python column density array is actually not needed ?
        coldensh_out_flat = np.ravel(np.zeros((N, N, N), dtype="float64"))
        phi_ion_flat = np.ravel(np.zeros((N, N, N), dtype="float64"))

        # Copy density field to GPU once at the beginning of timestep (!! do_all_sources assumes this !!)
        libasora.density_to_device(ndens_flat, N)

    # Set rates to 0. When using ASORA, this is done internally by the library (directly on the GPU)
    else:
        phi_ion = np.zeros((N, N, N), order="F")
        coldensh_out = np.zeros((N, N, N), order="F")

    # Raytrace all sources
    if use_gpu:
        # Use GPU raytracing
        libasora.do_all_sources(
            srcpos_flat,
            normflux_flat,
            r_RT,
            coldensh_out_flat,
            sig,
            dr,
            ndens_flat,
            xh_av_flat,
            phi_ion_flat,
            NumSrc,
            N,
            minlogtau,
            dlogtau,
            NumTau,
        )
        phi_ion = np.reshape(
            phi_ion_flat, (N, N, N)
        )  # Need to reshape rates back to 3D for output
    else:
        # Use CPU raytracing with subbox optimization
        nsubbox, photonloss = libc2ray.raytracing.do_all_sources(
            src_flux,
            src_pos,
            max_subbox,
            r_RT,
            coldensh_out,
            sig,
            dr,
            ndens,
            xh_av,
            phi_ion,
            loss_fraction,
            photo_thin_table,
            minlogtau,
            dlogtau,
        )

    if stats and not use_gpu:
        return phi_ion, nsubbox, photonloss
    else:
        return phi_ion
