"""A (accuracy) and B2 (accuracy vs. cost): DISCO-DJ against pkdgrav3 from identical ICs.

pkdgrav3 writes Zel'dovich ICs (iLPT=1, bWriteIC) for the same seed and transfer function as
its reference run. Their displacement field psi(q) gives the linear density field
delta_lin = -div psi, which is passed to DISCO-DJ (with_external_ics(delta=...)); DISCO-DJ then
builds its own nLPT ICs from it, so both codes evolve the same linear field. The 1LPT
displacement of DISCO-DJ is checked against pkdgrav3's (r and amplitude ratio) before anything
else is compared.

For each DISCO-DJ configuration (stepper, n_steps, PM grid, ...) and each output redshift of the
pkdgrav3 run, both particle sets are CIC-assigned on the same grid and the record holds
P_dj(k)/P_pkd(k) and r(k) = P_x / sqrt(P_dj P_pkd), plus the run time of the DISCO-DJ run.

    python benchmark_scripts/compare_pkdgrav3.py --test B2 --pkd-ic <ic tipsy> \
        --pkd-snap 10:<file> --pkd-snap 5:<file> --config bullfrog:10 --config fastpm:10:res_pm=512
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import djbench  # noqa: E402  (common setup, metadata, P(k) file)

HEADER = onp.dtype([("t", ">f8"), ("n", ">u4"), ("ndim", ">u4"), ("nsph", ">u4"),
                    ("ndark", ">u4"), ("nstar", ">u4"), ("pad", ">u4")])
DARK = onp.dtype([("m", ">f4"), ("pos", ">f4", 3), ("vel", ">f4", 3), ("eps", ">f4"), ("phi", ">f4")])


def read_tipsy(fn):
    """pkdgrav3 standard (big-endian) tipsy, dark matter only. Positions in box units [-0.5, 0.5)."""
    with open(fn, "rb") as f:
        h = onp.fromfile(f, HEADER, 1)[0]
        p = onp.fromfile(f, DARK, h["ndark"])
    return float(h["t"]), p["pos"].astype(onp.float64)


def lattice_psi(pos_box, N, L):
    """Displacement psi(q) on the IC lattice (Mpc/h, shape (N,N,N,3)) and the lattice offset s (cells).

    pkdgrav3 writes the IC particles in lattice order with x varying fastest (index = x + N y + N^2 z),
    so the lattice site comes from the particle order. (Rounding positions to the nearest site fails once
    displacements exceed half a cell: max |psi| = 0.85 cells at 512^3, z = 150.)"""
    u = (pos_box + 0.5) * N
    i = onp.arange(N ** 3)
    q = onp.stack([i % N, (i // N) % N, i // (N * N)], axis=1)
    d = u - q
    d -= onp.round(d / N) * N  # periodic wrap
    s = float(onp.median(d))   # lattice offset (0.5: cell centres)
    d -= s
    assert onp.abs(d).max() < N / 4, "IC particles are not in lattice order"
    psi = onp.zeros((N, N, N, 3))
    psi[q[:, 0], q[:, 1], q[:, 2]] = d * (L / N)
    return psi, s


def div_fourier(psi, L):
    N = psi.shape[0]
    k = 2 * onp.pi * onp.fft.fftfreq(N, d=L / N)
    kz = 2 * onp.pi * onp.fft.rfftfreq(N, d=L / N)
    kx, ky, kzz = onp.meshgrid(k, k, kz, indexing="ij")
    out = sum(1j * kk * onp.fft.rfftn(psi[..., c]) for c, kk in enumerate((kx, ky, kzz)))
    return onp.fft.irfftn(out, s=(N, N, N), axes=(0, 1, 2))


def parse_config(s):
    """stepper:n_steps[:key=val,...] -> dict of run_nbody keyword arguments."""
    parts = s.split(":")
    cfg = {"stepper": parts[0], "n_steps": int(parts[1])}
    for kv in (parts[2].split(",") if len(parts) > 2 and parts[2] else []):
        k, v = kv.split("=")
        cfg[k] = v if k in ("time_var", "ic_method", "method") else (float(v) if "." in v else int(v))
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="A")
    ap.add_argument("--pkd-ic", required=True, help="pkdgrav3 Zel'dovich IC tipsy file (iLPT=1)")
    ap.add_argument("--pkd-snap", action="append", required=True, help="z:file, pkdgrav3 snapshot")
    ap.add_argument("--config", action="append", required=True, help="stepper:n_steps[:k=v,...]")
    ap.add_argument("--boxsize", type=float, default=100.0)
    ap.add_argument("--lpt-order", type=int, default=2)
    ap.add_argument("--grid", type=int, default=None, help="comparison grid (default: N)")
    ap.add_argument("--pk-bins", type=int, default=40)
    ap.add_argument("--precision", default="single")
    ap.add_argument("--label", default="", help="prefix of the record label (label = prefix + config)")
    a = ap.parse_args()

    if a.precision == "double":
        import jax
        jax.config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp
    from discodj import DiscoDJ

    L = a.boxsize
    out_path = djbench.BENCH_DIR / "benchmark_results" / os.environ.get("BENCH_SYSTEM", "local") / a.test / \
        f"{a.test}_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    meta = djbench.metadata()

    # --- linear field from pkdgrav3's ZA ICs
    a_ic, pos = read_tipsy(a.pkd_ic)
    N = round(len(pos) ** (1 / 3))
    grid = a.grid or N
    psi_pkd, s = lattice_psi(pos, N, L)
    del pos
    z_ini = 1 / a_ic - 1

    pk_path = djbench.linear_pk_file(Path(os.environ.get("TMPDIR", "/tmp")) / f"pk_lin_{os.getpid()}.txt")
    dj = DiscoDJ(dim=3, res=N, boxsize=L, cosmo=djbench.COSMO, precision=a.precision)
    dj = dj.with_timetables().with_linear_ps(transfer_function="from_file", filename=str(pk_path), fix_sigma8=True)
    D = lambda aa: float(dj.cosmo.Dplus(jnp.asarray(aa, dtype=dj.dtype)))
    delta0 = -div_fourier(psi_pkd, L) * D(1.0) / D(a_ic)  # linear delta extrapolated to a = 1

    def with_delta(d0):
        return dj.with_external_ics(delta=jnp.asarray(d0, dtype=dj.dtype)).with_lpt(n_order=a.lpt_order)

    # Check: DISCO-DJ's 1LPT displacement at a_ic must reproduce pkdgrav3's psi; fix the amplitude if not
    d1 = with_delta(delta0)
    psi_dj = onp.asarray(d1.evaluate_lpt_psi_at_a(a_ic, n_order=1)).reshape(N, N, N, 3)
    amp = float((psi_dj * psi_pkd).sum() / (psi_dj * psi_dj).sum())
    r_psi = float((psi_dj * psi_pkd).sum() / onp.sqrt((psi_dj ** 2).sum() * (psi_pkd ** 2).sum()))
    check = {"z_ini": z_ini, "N": N, "lattice_offset_cells": s, "psi_amp_ratio_pkd_over_dj": amp,
             "psi_r": r_psi, "rms_psi_pkd": float(onp.sqrt((psi_pkd ** 2).mean()))}
    print("IC check:", check, flush=True)
    if abs(amp - 1) > 1e-3:
        delta0 *= amp
        check["amplitude_corrected"] = True
    dj = with_delta(delta0)
    del psi_dj, psi_pkd, d1
    shift = s * L / N  # pkdgrav3 lattice = DISCO-DJ lattice + shift

    # --- pkdgrav3 reference fields
    snaps = []
    for zs in a.pkd_snap:
        z, fn = zs.split(":", 1)
        a_s, p = read_tipsy(fn)
        X = jnp.asarray(((p + 0.5) % 1.0) * L, dtype=dj.dtype)
        snaps.append({"z": float(z), "z_file": 1 / a_s - 1, "a": a_s,
                      "delta": jax.block_until_ready(dj.get_delta_from_pos(X, res=grid, worder=2))})
        del X, p

    for cs in a.config:
        cfg = parse_config(cs)
        n_steps = cfg.pop("n_steps")
        kw = dict(res_pm=N, ic_method="lpt") | cfg
        rec = {"test": a.test, "label": a.label + cs, "config": cs, "n_steps": n_steps, "kw": kw,
               "ic_check": check, "grid": grid, "z_ini": z_ini, "outputs": []}
        try:
            for sn in snaps:
                t0 = time.perf_counter()
                X, _, _ = jax.block_until_ready(dj.run_nbody(a_ini=a_ic, a_end=sn["a"], n_steps=n_steps, **kw))
                dt_first = time.perf_counter() - t0
                X = (X.reshape(-1, 3) + shift) % L
                d = dj.get_delta_from_pos(X, res=grid, worder=2)
                # NB: the 2nd return value is already r(k) = P_x / sqrt(P_1 P_2) (docstring says "Pkx")
                k, rk, p1, p2 = dj.evaluate_cross_power_spectrum(d, sn["delta"], bins=a.pk_bins, deconvolve=True,
                                                                 worder=2)
                k, rk, p1, p2 = (onp.asarray(v, dtype=onp.float64) for v in (k, rk, p1, p2))
                rec["outputs"].append({"z": sn["z"], "z_pkd_file": sn["z_file"], "t_run_incl_compile": dt_first,
                                       "k": k.tolist(), "P_dj": p1.tolist(), "P_pkd": p2.tolist(),
                                       "ratio": (p1 / p2).tolist(), "r": rk.tolist()})
                kny = onp.pi * grid / L
                sel = lambda kmax: k <= kmax
                o = rec["outputs"][-1]
                print(f"[{cs}] z={sn['z']:.1f}: r(k<1)={onp.min(onp.asarray(o['r'])[sel(1.0)]):.4f} "
                      f"r(k<kNy/2)={onp.min(onp.asarray(o['r'])[sel(kny / 2)]):.4f} "
                      f"ratio(k<1) in [{onp.min(onp.asarray(o['ratio'])[sel(1.0)]):.3f},"
                      f"{onp.max(onp.asarray(o['ratio'])[sel(1.0)]):.3f}] t={dt_first:.2f}s", flush=True)
            rec["status"] = "ok"
        except Exception as e:
            rec["status"] = "error"
            rec["error"] = f"{type(e).__name__}: {e}"[:3000]
            print(rec["error"], flush=True)
        djbench.append_record(out_path, rec, meta)
    print("->", out_path)


if __name__ == "__main__":
    main()
