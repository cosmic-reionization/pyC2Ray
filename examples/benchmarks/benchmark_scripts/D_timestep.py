"""D. End-to-end timestep: GPU raytracing vs. CPU chemistry in pyC2Ray's evolve3D.

Runs pyc2ray.evolve.evolve3D (single GPU, no MPI) for a few consecutive
timesteps on a synthetic, production-like problem and times every raytracing
(asora.do_all_sources, GPU) and chemistry (libc2ray.chemistry.global_pass,
Fortran on one CPU core) call inside it. evolve3D iterates raytracing +
chemistry until the ionized fraction converges, so a timestep costs
n_iter x (raytracing + chemistry) plus the per-iteration host work (array
copies, convergence sums), reported as "other".

Problem: uniform hydrogen density 1.9e-4 cm^-3 (mean density at z=9), T=1e4 K,
initial ionized fraction 2e-4, box 100 cMpc at z=9 (physical cell size
100 Mpc / N / (1+z)), timestep 10 Myr, randomly placed sources with fluxes
log-uniform in 1e48-1e50 photons/s, black-body (T_eff = 5e4 K) photoionization
tables, chemistry constants as in the pyC2Ray parameter files.

Usage (on a GPU node):
  python benchmark_scripts/D_timestep.py [--mesh-sizes 100 256 --cases 15:100 15:10000 ...]
"""

import argparse
import os
import time

import astropy.constants as cst
import astropy.units as u
import numpy as np

import common
import pyc2ray.evolve as evolve
from pyc2ray.asora_core import device_close, device_init, photo_table_to_device
from pyc2ray.radiation.blackbody import BlackBodySource
from pyc2ray.radiation.common import make_tau_table

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--mesh-sizes", type=int, nargs="+", default=[100, 256])
p.add_argument("--cases", nargs="+", default=["15:100", "15:10000", "30:10000", "15:100000"],
               help="R:num_sources pairs, run for every mesh size")
p.add_argument("--batch-sizes", type=int, nargs="+", default=[8])
p.add_argument("--steps", type=int, default=3, help="consecutive timesteps per case")
args = p.parse_args()

# ----------------------------------------------------------------------------
# Physics (values of the pyC2Ray parameter files, e.g. validation/parameters_*.yml)
# ----------------------------------------------------------------------------
ZRED, BOX_CMPC, DT_MYR = 9.0, 100.0, 10.0
NDENS, TEMP, XH0 = 1.9e-4, 1e4, 2e-4
ETH0, ETHE1, FH0, XIH0 = 13.598, 54.416, 0.83, 1.0
EV2K = 1.0 / (cst.k_B * u.K).to("eV").value
EV2FR = (u.eV / cst.h).to("Hz").value
CHEM = {
    "bh00": 2.59e-13, "albpow": -0.7, "colh0": 1.3e-8 * FH0 * XIH0 / ETH0**2,
    "temph0": ETH0 * EV2K, "abu_c": 7.1e-7,
}
SIG = 6.30e-18
MINLOGTAU, MAXLOGTAU, NUMTAU = -20.0, 4.0, 10000
CONVERGENCE_FRACTION = 1e-4


class Timed:
    """Wraps a function and accumulates its wall time and number of calls."""

    def __init__(self, fn):
        self.fn, self.t, self.n = fn, 0.0, 0

    def __call__(self, *a, **kw):
        t0 = time.perf_counter()
        try:
            return self.fn(*a, **kw)
        finally:
            self.t += time.perf_counter() - t0
            self.n += 1


class Proxy:
    """Module stand-in that overrides some attributes and forwards the rest."""

    def __init__(self, module, **override):
        self._module, self.__dict__["_override"] = module, override

    def __getattr__(self, name):
        return self._override[name] if name in self._override else getattr(self._module, name)


raytracing = Timed(evolve.libasora.do_all_sources)
chemistry = Timed(evolve.libc2ray.chemistry.global_pass)
evolve.libasora = Proxy(evolve.libasora, do_all_sources=raytracing)
evolve.libc2ray = Proxy(evolve.libc2ray, chemistry=Proxy(evolve.libc2ray.chemistry, global_pass=chemistry))

tau, dlogtau = make_tau_table(MINLOGTAU, MAXLOGTAU, NUMTAU)
freq_min, freq_max = EV2FR * ETH0, 10 * EV2FR * ETHE1
radsource = BlackBodySource(5e4, False, freq_min, 2.8)
thin, thick = radsource.make_photo_table(tau, freq_min, freq_max, 1e48)


def run_case(N, nsrc, radius, batch, seed=918):
    rec = {"mesh_size": N, "num_sources": nsrc, "radius": radius, "R_max": float(radius), "batch_size": batch,
           "device_bytes_expected": common.asora_bytes(N, batch)}
    rng = np.random.default_rng(seed)
    src_pos = rng.integers(1, N + 1, size=(3, nsrc))  # Fortran indexing
    src_flux = 10 ** rng.uniform(0.0, 2.0, size=nsrc)  # in units of 1e48 photons/s
    shape = (N, N, N)
    ndens = np.full(shape, NDENS, order="F")
    temp = np.full(shape, TEMP, order="F")
    xh = np.full(shape, XH0, order="F")
    clump = np.ones(shape, order="F")  # clumping factor (a full cube, even if constant)
    dr = (BOX_CMPC * u.Mpc / N).cgs.value / (1 + ZRED)
    dt = (DT_MYR * u.Myr).to("s").value

    device_init(N, batch, 0, 1)
    try:
        photo_table_to_device(thin, thick)
        steps = []
        for step in range(args.steps):
            raytracing.t = raytracing.n = chemistry.t = chemistry.n = 0
            t0 = time.perf_counter()
            xh_new, phi, _ = evolve.evolve3D(
                dt=dt, dr=dr, src_flux=src_flux, src_pos=src_pos, use_gpu=True, max_subbox=0, subboxsize=0,
                loss_fraction=0.0, use_mpi=False, comm=None, rank=0, nprocs=1, temp=temp, ndens=ndens, xh=xh,
                clump=clump, photo_thin_table=thin, photo_thick_table=thick, minlogtau=MINLOGTAU, dlogtau=dlogtau,
                R_max_LLS=float(radius), convergence_fraction=CONVERGENCE_FRACTION, sig=SIG,
                logfile=os.devnull, quiet=True, **CHEM,
            )
            t_step = time.perf_counter() - t0
            valid = bool(np.all(np.isfinite(xh_new)) and np.all(np.isfinite(phi)) and np.any(phi > 0))
            steps.append({
                "t_step_s": t_step, "t_raytracing_s": raytracing.t, "t_chemistry_s": chemistry.t,
                "t_other_s": t_step - raytracing.t - chemistry.t, "iterations": raytracing.n,
                "mean_xHII": float(np.mean(xh_new)), "valid": valid,
            })
            print(f"N={N} src={nsrc} R={radius} batch={batch} step {step}: {t_step:.2f} s, {raytracing.n} iterations, "
                  f"raytracing {raytracing.t:.2f} s, chemistry {chemistry.t:.2f} s, <x_HII>={np.mean(xh_new):.3e}, "
                  f"valid={valid}", flush=True)
            xh = xh_new
    finally:
        device_close()
    rec["steps"] = steps
    rec["valid"] = all(s["valid"] for s in steps)
    tot = {k: sum(s[k] for s in steps) for k in ("t_step_s", "t_raytracing_s", "t_chemistry_s", "t_other_s", "iterations")}
    rec.update({
        "time_per_step_s": tot["t_step_s"] / len(steps),
        "time_per_iteration_s": tot["t_step_s"] / tot["iterations"],
        "raytracing_per_iteration_s": tot["t_raytracing_s"] / tot["iterations"],
        "chemistry_per_iteration_s": tot["t_chemistry_s"] / tot["iterations"],
        "iterations_per_step": tot["iterations"] / len(steps),
        "raytracing_fraction": tot["t_raytracing_s"] / tot["t_step_s"],
        "chemistry_fraction": tot["t_chemistry_s"] / tot["t_step_s"],
        "other_fraction": tot["t_other_s"] / tot["t_step_s"],
    })
    return rec


records = []
for N in args.mesh_sizes:
    for case in args.cases:
        radius, nsrc = case.split(":")
        for batch in args.batch_sizes:
            records.append(run_case(N, int(nsrc), radius, batch))
common.save_results("D_timestep", {**vars(args), "zred": ZRED, "box_cMpc": BOX_CMPC, "dt_Myr": DT_MYR,
                                   "ndens": NDENS, "temp": TEMP, "xh0": XH0, "num_tau": NUMTAU}, records)
