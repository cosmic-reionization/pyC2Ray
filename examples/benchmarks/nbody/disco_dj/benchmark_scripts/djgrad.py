"""B6: cost of gradients through the N-body run (field-level inference use).

Loss L(w) = mean(delta_final(w)^2) of the final CIC density, as a function of the real-space white
noise w. Times (after compilation) and peak GPU memory of
  forward       L(w)
  grad_adjoint  dL/dw, reverse mode with DISCO-DJ's adjoint method (memory independent of n_steps)
  grad_backprop dL/dw, reverse mode storing the trajectory (adjoint_method=False)
  jvp           forward-mode directional derivative
One mode per process (peak memory per mode). Appends to benchmark_results/<system>/B6/B6_<job>.json.

    python benchmark_scripts/djgrad.py --res 128 --n-steps 10 --mode grad_adjoint
"""
import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as onp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import djbench  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=10)
    ap.add_argument("--z-ini", type=float, default=150.0)
    ap.add_argument("--z-end", type=float, default=5.0)
    ap.add_argument("--mode", required=True, choices=["forward", "grad_adjoint", "grad_backprop", "jvp"])
    ap.add_argument("--repeat", type=int, default=2)
    a = ap.parse_args()

    import jax
    import jax.numpy as jnp
    from discodj import DiscoDJ

    rec = {"test": "B6", "label": f"N{a.res}_s{a.n_steps}_{a.mode}", "params": vars(a), "timings": {}}
    out_path = djbench.BENCH_DIR / "benchmark_results" / os.environ.get("BENCH_SYSTEM", "local") / "B6" / \
        f"B6_{os.environ.get('SLURM_JOB_ID', 'local')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        pk_path = djbench.linear_pk_file(Path(os.environ.get("TMPDIR", "/tmp")) / f"pk_lin_{os.getpid()}.txt")
        dj0 = DiscoDJ(dim=3, res=a.res, boxsize=100.0, cosmo=djbench.COSMO)
        dj0 = dj0.with_timetables().with_linear_ps(transfer_function="from_file", filename=str(pk_path),
                                                   fix_sigma8=True)
        a_ini, a_end = 1 / (1 + a.z_ini), 1 / (1 + a.z_end)
        w = jax.random.normal(jax.random.PRNGKey(1), (a.res,) * 3, dtype=dj0.dtype)
        v = jax.random.normal(jax.random.PRNGKey(2), (a.res,) * 3, dtype=dj0.dtype)
        adjoint = a.mode != "grad_backprop"

        def loss(wn):
            d = dj0.with_ics(white_noise_space="real", white_noise_field=wn).with_lpt(n_order=2)
            X, _, _ = d.run_nbody(a_ini=a_ini, a_end=a_end, n_steps=a.n_steps, res_pm=a.res, stepper="bullfrog",
                                  adjoint_method=adjoint if a.mode.startswith("grad") else None)
            return jnp.mean(d.get_delta_from_pos(X, try_to_jit=False) ** 2)

        f = {"forward": jax.jit(loss),
             "grad_adjoint": jax.jit(jax.value_and_grad(loss)),
             "grad_backprop": jax.jit(jax.value_and_grad(loss)),
             "jvp": jax.jit(lambda wn: jax.jvp(loss, (wn,), (v,)))}[a.mode]
        t0 = time.perf_counter()
        out = jax.block_until_ready(f(w))
        rec["timings"]["first"] = time.perf_counter() - t0
        runs = []
        for _ in range(a.repeat):
            t0 = time.perf_counter()
            out = jax.block_until_ready(f(w))
            runs.append(time.perf_counter() - t0)
        rec["timings"]["run"] = min(runs)
        rec["timings"]["compile"] = rec["timings"]["first"] - rec["timings"]["run"]
        leaves = [onp.asarray(x) for x in jax.tree_util.tree_leaves(out)]
        rec["finite"] = bool(all(onp.isfinite(x).all() for x in leaves))
        rec["loss"] = float(leaves[0])
        if a.mode.startswith("grad"):
            rec["grad_rms"] = float(onp.sqrt((leaves[1] ** 2).mean()))
        rec["status"] = "ok" if rec["finite"] else "nonfinite"
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        rec["status"] = "oom" if ("RESOURCE_EXHAUSTED" in msg or "out of memory" in msg.lower()) else "error"
        rec["error"] = msg[:3000]
        rec["traceback"] = traceback.format_exc()[-3000:]
    rec["mem"] = djbench.peak_mem()
    djbench.append_record(out_path, rec)
    print(f"[B6/{rec['label']}] {rec['status']} {({k: round(v, 3) for k, v in rec['timings'].items()})} "
          f"peak={rec['mem'].get('peak_bytes_in_use', 0) / 2**30:.2f} GiB {rec.get('error', '')[:300]}", flush=True)


if __name__ == "__main__":
    main()
