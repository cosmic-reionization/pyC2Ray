"""Shared helpers for the pyC2Ray ASORA benchmarks (issues #43, #44, #45).

Every benchmark script uses these to set up a synthetic problem (uniform density,
randomly placed point sources, as in pyC2Ray's tests/test_asora.py), time the
raytracing call, and write one JSON file per run under benchmark_results/, with
enough metadata (system, GPU, backend, code commit) to compare runs later.
"""

import ctypes
import datetime
import importlib.metadata
import json
import os
import platform
import shutil
import socket
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import astropy.constants as cst
import astropy.units as u
import numpy as np

from pyc2ray.load_extensions import load_asora
from pyc2ray.radiation.blackbody import BlackBodySource
from pyc2ray.radiation.common import make_tau_table

PROJECT_DIR = Path(__file__).resolve().parent.parent
# BENCH_RESULTS_DIR overrides the results location (e.g. for smoke tests)
RESULTS_DIR = Path(os.environ.get("BENCH_RESULTS_DIR", PROJECT_DIR / "benchmark_results"))

asora = load_asora()
if asora is None:
    raise RuntimeError("ASORA library (pyc2ray.lib.libasora) not available")


# ----------------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------------
def _run(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except Exception:
        return ""


def _asora_backend():
    """'hip' or 'cuda', from what the installed libasora is linked against."""
    out = _run(["ldd", asora.__file__])
    if "libamdhip64" in out:
        return "hip"
    if "libcudart" in out or "libcuda.so" in out:
        return "cuda"
    # nvcc links the CUDA runtime statically by default
    if shutil.which("nvidia-smi"):
        return "cuda"
    return "unknown"


def _gpu_name():
    if shutil.which("rocm-smi"):
        out = _run(["rocm-smi", "--showproductname", "--json"])
        try:
            cards = json.loads(out)
            return next(iter(cards.values())).get("Card Series", "")
        except Exception:
            pass
    if shutil.which("nvidia-smi"):
        return _run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).split("\n")[0]
    return ""


def _pyc2ray_source():
    """Source checkout pyc2ray was pip-installed from, and its git commit.

    Note: this is the checkout's HEAD when the benchmark runs, so rebuild after
    changing branches/commits.
    """
    try:
        dist = importlib.metadata.distribution("pyc2ray")
        url = json.loads(dist.read_text("direct_url.json"))["url"]
        src = url.removeprefix("file://")
    except Exception:
        return {"source": "", "commit": "", "branch": "", "dirty": None}
    commit = _run(["git", "-C", src, "rev-parse", "--short", "HEAD"]).strip()
    branch = _run(["git", "-C", src, "rev-parse", "--abbrev-ref", "HEAD"]).strip()
    dirty = bool(_run(["git", "-C", src, "status", "--porcelain", "--untracked-files=no"]).strip())
    return {"source": src, "commit": commit, "branch": branch, "dirty": dirty}


def _cpu_model():
    for line in _run(["lscpu"]).splitlines():
        if line.startswith("Model name:"):
            return line.split(":", 1)[1].strip()
    return platform.processor()


def _runtime_versions():
    """GPU runtime and driver versions as reported by the runtime libasora uses."""
    backend, rt = _runtime()
    if rt is None:
        return {}
    prefix = "hip" if backend == "hip" else "cuda"
    out = {}
    for name in ("RuntimeGetVersion", "DriverGetVersion"):
        v = ctypes.c_int()
        if getattr(rt, prefix + name)(ctypes.byref(v)) == 0:
            out[name.removesuffix("GetVersion").lower()] = v.value
    return out


def metadata():
    hostname = socket.gethostname()
    backend = _asora_backend()
    env = os.environ.get
    return {
        # Results folder name: set BENCH_SYSTEM in the job script (e.g. dardel, arrhenius)
        "system": env("BENCH_SYSTEM") or env("SLURM_CLUSTER_NAME") or hostname,
        # Build label for plots: BENCH_BUILD (e.g. hip-nvidia), default the detected backend
        "build": env("BENCH_BUILD") or backend,
        "hostname": hostname,
        "date": datetime.datetime.now().isoformat(timespec="seconds"),
        "slurm": {
            "job_id": env("SLURM_JOB_ID", ""),
            "cluster": env("SLURM_CLUSTER_NAME", ""),
            "partition": env("SLURM_JOB_PARTITION", ""),
            "nodelist": env("SLURM_JOB_NODELIST", ""),
            "num_nodes": env("SLURM_JOB_NUM_NODES", ""),
            "gpus_on_node": env("SLURM_GPUS_ON_NODE", ""),
        },
        "slurm_job_id": env("SLURM_JOB_ID", ""),
        "gpu": _gpu_name(),
        "visible_devices": env("ROCR_VISIBLE_DEVICES") or env("HIP_VISIBLE_DEVICES") or env("CUDA_VISIBLE_DEVICES", ""),
        "backend": backend,
        "runtime_version": _runtime_versions(),
        "rocm": env("ROCM_PATH", ""),
        "cuda": env("CUDA_HOME") or env("CUDA_PATH", ""),
        "cpu": _cpu_model(),
        "cpu_arch": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pyc2ray": _pyc2ray_source(),
    }


# ----------------------------------------------------------------------------
# GPU memory
# ----------------------------------------------------------------------------
_rt = None


def _runtime():
    """ctypes handle to the GPU runtime libasora is linked against."""
    global _rt
    if _rt is None:
        backend = _asora_backend()
        lib = {"hip": "libamdhip64", "cuda": "libcudart"}.get(backend)
        path = next(
            (
                line.split("=>")[1].split()[0]
                for line in _run(["ldd", asora.__file__]).splitlines()
                if lib and lib in line and "=>" in line
            ),
            None,
        )
        if path is None and backend == "cuda":
            # Statically linked runtime: load libcudart separately. Memory queries
            # are device-wide, so they still see ASORA's allocations.
            cuda_home = os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH") or ""
            for cand in ("libcudart.so", "libcudart.so.12", os.path.join(cuda_home, "lib64", "libcudart.so")):
                try:
                    ctypes.CDLL(cand)
                    path = cand
                    break
                except OSError:
                    continue
        _rt = (backend, ctypes.CDLL(path)) if path else (backend, None)
        if _rt[1] is not None:
            # Create the device context now, so its one-time allocation is not
            # attributed to the first problem's memory use
            (_rt[1].hipFree if backend == "hip" else _rt[1].cudaFree)(None)
    return _rt


def gpu_mem_info():
    """(free, total) device memory in bytes on the current device, or (None, None)."""
    backend, rt = _runtime()
    if rt is None:
        return None, None
    free, total = ctypes.c_size_t(), ctypes.c_size_t()
    fn = rt.hipMemGetInfo if backend == "hip" else rt.cudaMemGetInfo
    if fn(ctypes.byref(free), ctypes.byref(total)) != 0:
        return None, None
    return free.value, total.value


def asora_bytes(mesh_size, batch_size):
    """Device memory ASORA allocates in device_init: (3 + batch) grids of float64."""
    return (3 + batch_size) * mesh_size**3 * 8


def docs_memory_bytes(mesh_size, batch_size, R, num_sources):
    """GPU memory estimate from the pyC2Ray docs (tutorials/memory):
    ((3 N^3 + (sqrt(2) R)^3) M + 4 Nsrc) * 8 bytes, for comparison."""
    return ((3 * mesh_size**3 + (np.sqrt(2) * R) ** 3) * batch_size + 4 * num_sources) * 8


# ----------------------------------------------------------------------------
# Problem setup (follows tests/test_asora.py)
# ----------------------------------------------------------------------------
MINLOG_TAU, MAXLOG_TAU, NUM_TAU = -20.0, 4.0, 20000
SIGMA_HI = np.float64(6.30e-18)  # HI cross section at its ionizing frequency
BOX_PC = 50.0  # box size used to set the cell size dr


def box_radius(mesh_size):
    """Raytracing radius (cells) that covers the whole periodic box."""
    return np.sqrt(3) * mesh_size / 2


_warmed_up = False


@contextmanager
def asora_problem(mesh_size, batch_size, num_sources, R_max, seed=918, gpu_rank=0, num_gpus=1, src_slice=None):
    """Allocate ASORA, copy tables/density/sources to the device, and yield
    (args for asora.do_all_sources, dict of info). Frees the device on exit.

    src_slice=(start, stop) keeps only that part of the num_sources random
    sources (MPI source splitting, as in pyc2ray/evolve.py)."""
    global _warmed_up
    if not _warmed_up:
        # The first setup also makes one-time runtime allocations (code objects,
        # copy buffers): do it once on a tiny problem so that they are not
        # counted in the first measured problem's memory use
        _warmed_up = True
        with asora_problem(8, 1, 1, 2.0, seed, gpu_rank, num_gpus) as (warm_args, _):
            asora.do_all_sources(*warm_args)
    free_before, total = gpu_mem_info()
    asora.device_init(mesh_size, batch_size, gpu_rank, num_gpus)
    try:
        tau, dlogtau = make_tau_table(MINLOG_TAU, MAXLOG_TAU, NUM_TAU)
        freq_min, freq_max = (
            (13.598 * u.eV / cst.h).to("Hz").value,
            (54.416 * u.eV / cst.h).to("Hz").value,
        )
        radsource = BlackBodySource(1e5, False, freq_min, SIGMA_HI)
        thin, thick = radsource.make_photo_table(tau, freq_min, freq_max, 1e48)
        asora.photo_table_to_device(thin, thick, NUM_TAU)

        size = mesh_size**3
        coldensh_out = np.zeros(size, dtype=np.float64)
        phi_ion = np.zeros(size, dtype=np.float64)
        ndens = np.full(size, 1e-3, dtype=np.float64)
        xHII = np.full(size, 1e-4, dtype=np.float64)
        asora.density_to_device(ndens, mesh_size)

        rng = np.random.default_rng(seed)
        src_pos = rng.integers(0, mesh_size, size=(3 * num_sources), dtype=np.int32)
        norm_flux = rng.uniform(1e10, 1e14, size=num_sources).astype(np.float64)
        norm_flux *= 100.0 / 1e48
        if src_slice is not None:
            i0, i1 = src_slice
            src_pos = np.ascontiguousarray(src_pos[3 * i0 : 3 * i1])
            norm_flux = np.ascontiguousarray(norm_flux[i0:i1])
            num_sources = i1 - i0
        asora.source_data_to_device(src_pos, norm_flux, num_sources)

        dr = (BOX_PC * u.pc / mesh_size).cgs.value
        free_after, _ = gpu_mem_info()
        info = {
            "device_bytes_expected": asora_bytes(mesh_size, batch_size),
            "device_bytes_used": (free_before - free_after) if free_before is not None else None,
            "device_bytes_total": total,
            "device_bytes_free_before": free_before,
        }
        args = (
            float(R_max), coldensh_out, SIGMA_HI, dr, ndens, xHII, phi_ion,
            num_sources, mesh_size, MINLOG_TAU, dlogtau, NUM_TAU,
        )
        yield args, info
    finally:
        asora.device_close()


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------
def time_do_all_sources(args, warmup=1, repeats=5, max_seconds=120.0):
    """Time asora.do_all_sources(*args). Returns (list of wall times in s, valid).

    Stops repeating once max_seconds of timed calls have accumulated (at least
    one timed call). ASORA returns silently on GPU errors, so validity is
    checked on the output: phi_ion must be finite and not all zero.
    """
    phi_ion = args[6]
    for _ in range(warmup):
        asora.do_all_sources(*args)
    times = []
    for _ in range(repeats):
        phi_ion[:] = 0.0
        t0 = time.perf_counter()
        asora.do_all_sources(*args)
        times.append(time.perf_counter() - t0)
        if sum(times) > max_seconds:
            break
    valid = bool(np.all(np.isfinite(phi_ion)) and np.any(phi_ion > 0))
    return times, valid


# ----------------------------------------------------------------------------
# Results
# ----------------------------------------------------------------------------
def save_results(test, params, records):
    """Write benchmark_results/<system>/<test>/<test>_<jobid or timestamp>.json."""
    meta = metadata()
    tag = meta["slurm_job_id"] or datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / meta["system"] / test / f"{test}_{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    # Several runs of the same test in one job: never overwrite, add a counter
    k = 1
    while out.exists():
        k += 1
        out = out.with_name(f"{test}_{tag}_{k}.json")
    out.write_text(json.dumps({"test": test, "meta": meta, "params": params, "records": records}, indent=1))
    print(f"Results written to {out}")
    return out


# ----------------------------------------------------------------------------
# One measurement point
# ----------------------------------------------------------------------------
def measure_point(mesh_size, batch_size, num_sources, radius, warmup=1, repeats=5, max_seconds=120.0, **extra):
    """Time asora.do_all_sources for one configuration and return a record.

    radius is a number of cells or 'box'. Configurations that do not fit in
    device memory are returned with a 'skipped' reason instead of timings.
    Extra keyword arguments are stored in the record as-is.
    """
    R = box_radius(mesh_size) if radius == "box" else float(radius)
    rec = {"mesh_size": mesh_size, "radius": str(radius), "R_max": R, "num_sources": num_sources,
           "batch_size": batch_size, "device_bytes_expected": asora_bytes(mesh_size, batch_size), **extra}
    _, total = gpu_mem_info()
    if total is not None and rec["device_bytes_expected"] > 0.95 * total:
        rec["skipped"] = f"needs {rec['device_bytes_expected'] / 1e9:.1f} GB > 95% of {total / 1e9:.1f} GB"
        print(f"{_describe(rec)}: skipped ({rec['skipped']})", flush=True)
        return rec
    with asora_problem(mesh_size, batch_size, num_sources, R) as (args, info):
        times, valid = time_do_all_sources(args, warmup, repeats, max_seconds)
    t = float(np.median(times))
    rec.update(info)
    rec.update({"times_s": times, "time_median_s": t, "valid": valid,
                "time_per_source_s": t / num_sources, "sources_per_s": num_sources / t})
    used = info["device_bytes_used"]
    print(f"{_describe(rec)}: median {t:.4f} s over {len(times)} calls, valid={valid}, "
          f"device {used / 1e9 if used else float('nan'):.2f} GB", flush=True)
    return rec


def _describe(rec):
    return f"N={rec['mesh_size']} src={rec['num_sources']} R={rec['radius']} batch={rec['batch_size']}"


def add_common_args(parser, warmup=1, repeats=5, max_seconds=120.0):
    parser.add_argument("--warmup", type=int, default=warmup)
    parser.add_argument("--repeats", type=int, default=repeats)
    parser.add_argument("--max-seconds", type=float, default=max_seconds, help="time budget per point for repeats")
