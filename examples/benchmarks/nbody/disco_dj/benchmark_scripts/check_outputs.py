"""A: are the density grids written by djbench (--write-dir) readable as pyC2Ray input?

Reads every grid pair (disco.<i>.den.<n>.npy / .0, pyC2Ray naming) of a run directory
  - .npy as pyC2Ray does (overdensity = load - 1),
  - .bin with tools21cm.Pkdgrav3data.load_density_field (as pyC2Ray does for pkdgrav3 grids),
checks that both give the same delta (incl. orientation), that the last one equals the in-memory final
field saved by djbench (--save-delta), and that redshift_density.txt matches the number of files.
Appends the result to benchmark_results/<system>/A_io/A_io_<job>.json.

    python benchmark_scripts/check_outputs.py <write dir> <final delta .npy> [--grid N]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import djbench  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("write_dir")
    ap.add_argument("final_delta")
    ap.add_argument("--boxsize", type=float, default=100.0)
    a = ap.parse_args()
    import tools21cm as t2c

    wd = Path(a.write_dir)
    npys = sorted(wd.glob("disco.*.den.*.npy"))
    zs = onp.loadtxt(wd / "redshift_density.txt")
    rec = {"test": "A_io", "write_dir": str(wd), "n_npy": len(npys), "n_bin": len(list(wd.glob("disco.*.den.*.0"))),
           "n_redshifts": int(zs.size), "z_first": float(zs[0]), "z_last": float(zs[-1]),
           "tools21cm": getattr(t2c, "__version__", "?"), "checks": []}
    worst = 0.0
    t_npy = t_bin = 0.0
    for f in npys:
        t0 = time.perf_counter(); d_npy = onp.load(f).astype(onp.float64) - 1.0; t_npy += time.perf_counter() - t0
        n = d_npy.shape[0]
        fb = f.with_suffix(".0")
        t0 = time.perf_counter()
        d_bin = t2c.Pkdgrav3data(a.boxsize, n, Omega_m=0.32, verbose=False).load_density_field(str(fb))
        t_bin += time.perf_counter() - t0
        diff = float(abs(d_bin - d_npy).max())
        worst = max(worst, diff)
        rec["checks"].append({"file": f.name, "mean_npy": float(d_npy.mean()), "std": float(d_npy.std()),
                              "max_abs_diff_bin_vs_npy": diff})
    final = onp.load(a.final_delta).astype(onp.float64)
    last = onp.load(npys[-1]).astype(onp.float64) - 1.0
    rec["final_vs_last_output_max_abs_diff"] = float(abs(final - last).max())
    rec["final_vs_last_output_r"] = float((final * last).sum() / onp.sqrt((final ** 2).sum() * (last ** 2).sum()))
    rec["max_abs_diff_bin_vs_npy"] = worst
    rec["t_read_npy_per_file"] = t_npy / max(len(npys), 1)
    rec["t_read_bin_per_file"] = t_bin / max(len(npys), 1)
    rec["status"] = "ok" if (worst < 1e-4 and rec["final_vs_last_output_r"] > 0.999999
                             and len(npys) == zs.size) else "mismatch"

    out_path = djbench.BENCH_DIR / "benchmark_results" / os.environ.get("BENCH_SYSTEM", "local") / "A_io" / \
        f"A_io_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    djbench.append_record(out_path, rec)
    print({k: v for k, v in rec.items() if k != "checks"}, flush=True)


if __name__ == "__main__":
    main()
