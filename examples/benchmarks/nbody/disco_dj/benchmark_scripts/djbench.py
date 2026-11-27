"""Run one DISCO-DJ benchmark and append its record to the result file.

Sets up the common case (Giri+24 cosmology, L = 100 Mpc/h, z = 150 -> 5, same linear P(k)
as the pkdgrav3 benchmarks), runs the PM N-body on one GPU and writes

    benchmark_results/<system>/<test>/<test>_<slurm job id>.json

Several runs of one job go into the same file (appended to "records"). One run per process,
so the peak GPU memory reported by JAX's allocator belongs to that run alone.

Timings (all wall clock, after block_until_ready):
  setup     timetables, linear P(k), ICs (white noise + phi) and nLPT
  first     first run_nbody call: tracing + XLA compilation + execution
  run       repeated identical call: execution only (best of --repeat)
  compile   first - run
Output modes (--outputs):
  final     only the final state (B1, B2)
  loop      one run_nbody call per output interval, gridding + host copy of every output (B3, D)
  collect   one call with collect_all over the output times (B3; memory grows with #outputs)

Examples:
    python benchmark_scripts/djbench.py --test B1 --label N256 --res 256
    python benchmark_scripts/djbench.py --test B3 --label loop99 --res 256 --outputs loop
"""

import argparse
import datetime
import json
import os
import platform
import re
import socket
import subprocess
import time
import traceback
from pathlib import Path

import numpy as onp

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent
TF_FILE = HERE / "euclid_z0_transfer_combined.dat"  # same file as the pkdgrav3 benchmarks (#47)

# Common setup: Giri+24 (arXiv:2403.04838) as in #47. Omega_b only enters a fitted transfer
# function, not used here (P(k) is read from TF_FILE).
COSMO = dict(Omega_c=0.32 - 0.049, Omega_b=0.049, Omega_k=0.0, h=0.67, n_s=0.96, sigma8=0.83,
             w0=-1.0, wa=0.0)
MYR_PER_HUBBLE_TIME = 9778.0  # 1/H0 in Myr for h = 1


def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout
    except OSError:
        return ""


def _git(src):
    if not src or not Path(src).is_dir():
        return {}
    g = lambda *a: _run(["git", "-C", str(src), *a]).strip()
    return {"path": str(src), "commit": g("rev-parse", "--short=8", "HEAD"),
            "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def metadata():
    import jax
    env = os.environ.get
    smi = _run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"])
    cpu = re.search(r"Model name:\s*(.+)", _run(["lscpu"]))
    try:
        import jaxlib
        jaxlib_v = jaxlib.__version__
    except Exception:
        jaxlib_v = ""
    pkgs = {}
    for p in ("diffrax", "equinox", "numpy", "einops"):
        try:
            pkgs[p] = __import__(p).__version__
        except Exception:
            pass
    try:
        cuda_backend = jax.extend.backend.get_backend().platform_version
    except Exception:
        cuda_backend = ""
    return {
        "system": env("BENCH_SYSTEM") or socket.gethostname(),
        "build": env("BENCH_BUILD", ""),
        "hostname": socket.gethostname(),
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "slurm": {k: env("SLURM_" + k.upper(), "") for k in
                  ("job_id", "job_partition", "job_nodelist", "gpus_on_node", "cpus_per_task", "ntasks")},
        "slurm_job_id": env("SLURM_JOB_ID", ""),
        "gpu": [l.strip() for l in smi.splitlines() if l.strip()],
        "cpu": cpu.group(1).strip() if cpu else platform.processor(),
        "cpu_arch": platform.machine(),
        "python": platform.python_version(),
        "jax": jax.__version__, "jaxlib": jaxlib_v, "jax_backend": cuda_backend,
        "jax_devices": [str(d) for d in jax.devices()],
        "packages": pkgs,
        "xla_env": {k: v for k, v in os.environ.items() if k.startswith(("XLA_", "JAX_", "TF_"))},
        "discodj": _git(env("DISCODJ_SRC")),
        "benchmark_setup": _git(BENCH_DIR),  # pyC2Ray checkout holding these scripts
    }


def output_scale_factors(z_from=30.0, z_to=5.0, dt_myr=10.86, h=COSMO["h"], om=0.32):
    """Output times every dt_myr from z_from to z_to (99 outputs for the defaults), flat LCDM."""
    ol = 1.0 - om
    a = onp.geomspace(1.0 / (1 + z_from) * 0.5, 1.0, 200001)
    integrand = 1.0 / (a * onp.sqrt(om / a ** 3 + ol))
    t = onp.concatenate([[0.0], onp.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * onp.diff(a))])
    t *= MYR_PER_HUBBLE_TIME / h
    t0 = onp.interp(1 / (1 + z_from), a, t)
    t1 = onp.interp(1 / (1 + z_to), a, t)
    ts = onp.arange(t0, t1 + 1e-6, dt_myr)
    return onp.interp(ts, t, a)


def linear_pk_file(path, n_s=COSMO["n_s"]):
    """Two-column (k [h/Mpc], T(k)) transfer function -> (k, k^n_s T^2), normalised later to sigma8."""
    k, T = onp.loadtxt(TF_FILE, unpack=True)[:2]
    onp.savetxt(path, onp.c_[k, k ** n_s * T ** 2])
    return path


def append_record(out_path, rec, meta=None):
    """Append one record to a result file under an exclusive lock (several processes of one job may
    write the same file at the same time; an unlocked read-modify-write loses records)."""
    import fcntl
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(out_path) + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if out_path.exists():
            doc = json.loads(out_path.read_text())
        else:
            doc = {"metadata": meta if meta is not None else metadata(), "records": []}
        doc["records"].append(rec)
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(doc, indent=1))
        tmp.replace(out_path)
        fcntl.flock(lk, fcntl.LOCK_UN)


def peak_mem():
    import jax
    try:
        s = jax.devices()[0].memory_stats() or {}
        return {k: int(s[k]) for k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit") if k in s}
    except Exception:
        return {}


def block(x):
    import jax
    return jax.block_until_ready(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--res", type=int, default=256, help="particles per dimension")
    ap.add_argument("--res-pm", type=int, default=None, help="PM grid (default: res)")
    ap.add_argument("--boxsize", type=float, default=100.0)
    ap.add_argument("--z-ini", type=float, default=150.0)
    ap.add_argument("--z-end", type=float, default=5.0)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--time-var", default="D", help='"D", "a", "log_a", "superconft"')
    ap.add_argument("--stepper", default="bullfrog")
    ap.add_argument("--ic-method", default="lpt")
    ap.add_argument("--lpt-order", type=int, default=2)
    ap.add_argument("--worder", type=int, default=2)
    ap.add_argument("--antialias", type=int, default=0)
    ap.add_argument("--deconvolve", action="store_true")
    ap.add_argument("--n-resample", type=int, default=1)
    ap.add_argument("--grad-kernel-order", type=int, default=4)
    ap.add_argument("--method", default="pm", help='"pm", "treepm" (Tree-PM) or "nufftpm"')
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--precision", default="single", choices=["single", "double"])
    ap.add_argument("--seed", type=int, default=314159265)
    ap.add_argument("--grid", type=int, default=None, help="density grid for outputs/P(k) (default: res)")
    ap.add_argument("--outputs", default="final", choices=["final", "loop", "collect"])
    ap.add_argument("--n-outputs", type=int, default=None, help="use only the first n output times")
    ap.add_argument("--steps-per-output", type=int, default=1, help="loop/collect: steps per output interval")
    ap.add_argument("--repeat", type=int, default=1, help="timed repetitions after the first call")
    ap.add_argument("--save-delta", default=None, help="save the final density grid (.npy) here")
    ap.add_argument("--pk-bins", type=int, default=32)
    ap.add_argument("--write-dir", default=None, help="loop mode: write every output grid here (B3/D)")
    ap.add_argument("--grid-format", default="both", choices=["npy", "bin", "both"])
    ap.add_argument("--snap-z", type=float, nargs="*", default=[], help="loop mode: HDF5 particle snapshots")
    ap.add_argument("--pk-z", type=float, nargs="*", default=[], help="loop mode: record P(k) at these z")
    ap.add_argument("--jit-segments", action="store_true",
                    help="loop mode: compile one output interval once (scale factors traced)")
    ap.add_argument("--free-ics", action="store_true",
                    help="loop mode: drop the ICs and nLPT fields after the first segment (needs --repeat 0)")
    a = ap.parse_args()
    if a.free_ics and (a.repeat or a.outputs != "loop"):
        raise SystemExit("--free-ics needs --outputs loop and --repeat 0")

    if a.precision == "double":
        import jax
        jax.config.update("jax_enable_x64", True)
    import jax
    import jax.numpy as jnp
    from discodj import DiscoDJ

    res_pm = a.res_pm or a.res
    grid = a.grid or a.res
    a_ini, a_end = 1.0 / (1 + a.z_ini), 1.0 / (1 + a.z_end)
    rec = {"test": a.test, "label": a.label, "params": vars(a) | {"res_pm": res_pm, "grid": grid},
           "cosmology": COSMO, "status": "started", "timings": {}}
    out_path = BENCH_DIR / "benchmark_results" / os.environ.get("BENCH_SYSTEM", "local") / a.test / \
        f"{a.test}_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = rec["timings"]
    try:
        # --- setup: cosmology tables, linear P(k), ICs, nLPT
        t0 = time.perf_counter()
        pk_path = linear_pk_file(Path(os.environ.get("TMPDIR", "/tmp")) / f"pk_lin_{os.getpid()}.txt")
        dj = DiscoDJ(dim=3, res=a.res, boxsize=a.boxsize, cosmo=COSMO, precision=a.precision)
        dj = dj.with_timetables().with_linear_ps(transfer_function="from_file", filename=str(pk_path),
                                                 fix_sigma8=True)
        dj = dj.with_ics(seed=a.seed)
        block(dj._ics["fphi"])
        t["ics"] = time.perf_counter() - t0
        dj = dj.with_lpt(n_order=a.lpt_order)
        block(jax.tree_util.tree_leaves(dj._lpt))
        t["setup"] = time.perf_counter() - t0
        rec["mem_after_setup"] = peak_mem()

        kw = dict(stepper=a.stepper, method=a.method, res_pm=res_pm, antialias=a.antialias, worder=a.worder,
                  deconvolve=a.deconvolve, n_resample=a.n_resample, grad_kernel_order=a.grad_kernel_order,
                  ic_method=a.ic_method, chunk_size=a.chunk_size, time_var=a.time_var)

        def run_final():
            return block(dj.run_nbody(a_ini=a_ini, a_end=a_end, n_steps=a.n_steps, **kw))

        if a.outputs == "final":
            fn = run_final
        else:
            a_out = output_scale_factors()
            a_out = a_out[a_out <= a_end * (1 + 1e-9)]
            if a.n_outputs:
                a_out = a_out[:a.n_outputs]
            rec["a_outputs"] = a_out.tolist()
            spo = a.steps_per_output
            # steps from a_ini to the first output (in D, as many as n_steps would give over the full range)
            n_first = max(1, a.n_steps - spo * (len(a_out) - 1))

            wdir = Path(a.write_dir) if a.write_dir else None
            if wdir:  # pyC2Ray layout: <base>.<idx>.den.<grid>.{0,npy} + redshift_density.txt
                wdir.mkdir(parents=True, exist_ok=True)
                onp.savetxt(wdir / "redshift_density.txt", 1 / a_out - 1, fmt="%.6f")
            snap_idx = {int(onp.argmin(abs(1 / a_out - 1 - z))) for z in a.snap_z}
            pk_idx = {int(onp.argmin(abs(1 / a_out - 1 - z))) for z in a.pk_z}
            rec["io"] = {"grid_files": 0, "grid_bytes": 0, "snap_files": 0, "snap_bytes": 0}

            def write_output(i, delta, X, P, tl):
                """pyC2Ray density input (1+delta, float32) as .npy and/or pkdgrav3-style raw binary
                (tools21cm.Pkdgrav3data transposes on reading, so the raw file holds (1+delta).T),
                plus HDF5 particle snapshots at the requested redshifts."""
                t1 = time.perf_counter()
                z = 1 / a_out[i] - 1
                d1 = (1.0 + delta).astype(onp.float32)
                if a.grid_format in ("npy", "both"):
                    f = wdir / f"disco.{i:05d}.den.{grid}.npy"; onp.save(f, d1)
                    rec["io"]["grid_files"] += 1; rec["io"]["grid_bytes"] += f.stat().st_size
                if a.grid_format in ("bin", "both"):
                    f = wdir / f"disco.{i:05d}.den.{grid}.0"; onp.ascontiguousarray(d1.T).tofile(f)
                    rec["io"]["grid_files"] += 1; rec["io"]["grid_bytes"] += f.stat().st_size
                tl["io"] = tl.get("io", 0.0) + time.perf_counter() - t1
                if i in snap_idx:
                    t1 = time.perf_counter()
                    f = wdir / f"snap_{a.res}_z{z:.3f}.hdf5"
                    dj.save_as_hdf5(str(f), X, P, float(a_out[i]), compressed=False)
                    rec["io"]["snap_files"] += 1; rec["io"]["snap_bytes"] += f.stat().st_size
                    tl["snap"] = tl.get("snap", 0.0) + time.perf_counter() - t1

            # One output interval as a single compiled function of (state, a0, a1): without it every
            # run_nbody call is traced and compiled again (a0, a1 are new Python floats each time).
            kw_seg = kw | {"ic_method": "lpt"}

            # DiscoDJ keeps ICs/nLPT arrays in its pytree aux data, so it cannot be a jit argument, and
            # closing over the full object would embed those GB-sized arrays as constants: close over a
            # light copy without them (the segment starts from the particles anyway).
            d_light = dj.copy()
            d_light._lpt, d_light._ics = None, {}

            def _segment(X, P, a0, a1):
                d = d_light.with_external_ics(pos=X.reshape(-1, 3), vel=P.reshape(-1, 3))
                return d.run_nbody(a_ini=a0, a_end=a1, n_steps=spo, **kw_seg)[:2]
            segment_jit = jax.jit(_segment)

            if a.outputs == "loop":
                def fn():
                    tl = {"step": 0.0, "grid": 0.0, "host": 0.0}
                    fn.calls = getattr(fn, "calls", 0) + 1
                    X, P, _ = block(dj.run_nbody(a_ini=a_ini, a_end=float(a_out[0]), n_steps=n_first, **kw))
                    if a.free_ics:
                        # the ICs (fphi) and nLPT fields are not needed after the first segment; dropping
                        # them saves ~13 GB at 768^3. (update(lpt=None) is a no-op, hence the attributes.)
                        dj._lpt, dj._ics = None, {}
                    d = dj
                    for i in range(len(a_out)):
                        if i > 0:
                            t1 = time.perf_counter()
                            if a.jit_segments:
                                X, P = block(segment_jit(X, P, jnp.asarray(a_out[i - 1], dtype=dj.dtype),
                                                         jnp.asarray(a_out[i], dtype=dj.dtype)))
                            else:
                                d = d.with_external_ics(pos=X.reshape(-1, 3), vel=P.reshape(-1, 3))
                                X, P, _ = block(d.run_nbody(a_ini=float(a_out[i - 1]), a_end=float(a_out[i]),
                                                            n_steps=spo, **kw_seg))
                            tl["step"] += time.perf_counter() - t1
                        t1 = time.perf_counter()
                        delta = block(dj.get_delta_from_pos(X, res=grid, worder=2))
                        tl["grid"] += time.perf_counter() - t1
                        t1 = time.perf_counter()
                        dh = onp.asarray(delta)
                        tl["host"] += time.perf_counter() - t1
                        if wdir and fn.calls == 1:  # write on the first pass only
                            write_output(i, dh, X, P, tl)
                        if i in pk_idx:
                            k, pk, _ = dj.evaluate_power_spectrum(delta, bins=a.pk_bins, deconvolve=True, worder=2)
                            rec.setdefault("pk_outputs", {})[f"{1 / a_out[i] - 1:.3f}"] = {
                                "k": onp.asarray(k).tolist(), "pk": onp.asarray(pk).tolist()}
                    fn.last = tl
                    return X, P, a_out[-1]
            else:  # collect: one call, outputs at every scale factor of the array
                a_arr = onp.concatenate([onp.geomspace(a_ini, a_out[0], n_first + 1)[:-1], a_out])
                if spo > 1:
                    raise SystemExit("collect mode supports --steps-per-output 1 only")
                rec["n_steps_total"] = len(a_arr) - 1

                def fn():
                    out = block(dj.run_nbody(a_ini=float(a_arr[0]), a_end=float(a_arr[-1]), n_steps=None,
                                             **(kw | {"time_var": jnp.asarray(a_arr, dtype=dj.dtype)}),
                                             collect_all=True))
                    return out[0][-1], out[1][-1], out[2]

        t0 = time.perf_counter()
        X, P, _ = fn()
        t["first"] = time.perf_counter() - t0
        if getattr(fn, "last", None):
            t["first_split"] = fn.last
        runs = []
        for _ in range(a.repeat):
            del X, P
            t0 = time.perf_counter()
            X, P, _ = fn()
            runs.append(time.perf_counter() - t0)
            if getattr(fn, "last", None):
                t.setdefault("run_split", []).append(fn.last)
        if runs:
            t["run_all"] = runs
            t["run"] = min(runs)
            t["compile"] = t["first"] - t["run"]
        rec["mem"] = peak_mem()

        # --- final density field and P(k) (cheap; recorded for the accuracy tests)
        t0 = time.perf_counter()
        delta = block(dj.get_delta_from_pos(X, res=grid, worder=2))
        t["grid_final"] = time.perf_counter() - t0
        k, pk, _ = dj.evaluate_power_spectrum(delta, bins=a.pk_bins, deconvolve=True, worder=2)
        rec["pk"] = {"k": onp.asarray(k).tolist(), "pk": onp.asarray(pk).tolist(), "z": a.z_end}
        d = onp.asarray(delta)
        rec["delta_stats"] = {"mean": float(d.mean()), "std": float(d.std()), "min": float(d.min()),
                              "max": float(d.max()), "finite": bool(onp.isfinite(d).all())}
        rec["finite_pos"] = bool(jnp.isfinite(X).all())
        if a.save_delta:
            Path(a.save_delta).parent.mkdir(parents=True, exist_ok=True)
            onp.save(a.save_delta, d.astype(onp.float32))
            rec["saved_delta"] = a.save_delta
        rec["n_particles"] = a.res ** 3
        if "run" in t:
            rec["ns_per_particle_step"] = t["run"] / (a.res ** 3 * max(a.n_steps, 1)) * 1e9
        rec["status"] = "ok" if rec["finite_pos"] and rec["delta_stats"]["finite"] else "nonfinite"
    except Exception as e:  # record OOM and other failures as data points
        msg = f"{type(e).__name__}: {e}"
        rec["status"] = "oom" if ("RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()) else "error"
        rec["error"] = msg[:4000]
        rec["traceback"] = traceback.format_exc()[-4000:]
        rec["mem"] = peak_mem()

    append_record(out_path, rec)
    tm = {k: round(v, 3) for k, v in t.items() if isinstance(v, float)}
    print(f"[{a.test}/{a.label}] {rec['status']} timings={tm} "
          f"peak={rec.get('mem', {}).get('peak_bytes_in_use', 0) / 2**30:.2f} GiB -> {out_path}", flush=True)
    if rec["status"] not in ("ok",):
        print(rec.get("error", ""), flush=True)


if __name__ == "__main__":
    main()
