#!/usr/bin/env python3
"""T1 with filtered coarse-graining: 4 x 4 block averages of the PDEBench fields instead of strided subsampling.

    python3 datasets/generators/gen_T1_blockavg.py \
        --src <PDEBench 2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5>

Same trajectories, time steps, sample order, split and target definition as ``datasets/T1_cns_vorticity.pkl``
(500 trajectories x 20 steps; sample ``20 i + t``: ``(rho, Vx, Vy, p)`` at step ``t`` -> vorticity at ``t + 1``; the
periodic central-difference curl with ``dx = 1/32``), but every 32 x 32 field is the mean over the 4 x 4 block of
the 128 x 128 field (an ideal box filter before subsampling) instead of every 4th point.  The coarse grid is still
uniform with spacing 1/32 (block centres), so the periodic complex of T1p applies unchanged.

Output ``datasets/v2/T1_cns_vorticity_blockavg.pkl`` with the keys of the T1 pickle (``points``, ``faces``,
``X_data (10000, 1024, 4)``, ``Y_data (10000, 1024, 1)``, ``grid_size``) plus ``coarse_graining`` and ``source``.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np

N_TRAJ, N_STEPS, GS, FINE = 500, 20, 32, 128


def block_mean(a: np.ndarray, f: int) -> np.ndarray:
    """``(..., n, n) -> (..., n/f, n/f)`` mean over f x f blocks."""
    s = a.shape
    return a.reshape(*s[:-2], s[-2] // f, f, s[-1] // f, f).mean(axis=(-3, -1))


def curl(vx: np.ndarray, vy: np.ndarray, dx: float) -> np.ndarray:
    """Periodic central-difference curl ``dVy/dx - dVx/dy`` of fields indexed ``[..., ix, iy]`` (as T1)."""
    return ((np.roll(vy, -1, -2) - np.roll(vy, 1, -2)) - (np.roll(vx, -1, -1) - np.roll(vx, 1, -1))) / (2 * dx)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--t1", default=os.path.join(os.path.dirname(__file__), "..", "T1_cns_vorticity.pkl"),
                    help="the T1 pickle (points/faces copied, strided-subsampling consistency check)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "v2",
                                                  "T1_cns_vorticity_blockavg.pkl"))
    ap.add_argument("--chunk", type=int, default=50)
    a = ap.parse_args()
    import h5py
    t0 = time.time()
    f = FINE // GS
    fields = ("density", "Vx", "Vy", "pressure")
    coarse = {k: np.empty((N_TRAJ, N_STEPS + 1, GS, GS), dtype=np.float64) for k in fields}
    strided_vx = np.empty((N_TRAJ, N_STEPS + 1, GS, GS), dtype=np.float32)
    with h5py.File(a.src, "r") as h:
        for s0 in range(0, N_TRAJ, a.chunk):
            s1 = min(N_TRAJ, s0 + a.chunk)
            for k in fields:
                raw = h[k][s0:s1, :N_STEPS + 1]                                     # (c, 21, 128, 128)
                coarse[k][s0:s1] = block_mean(raw.astype(np.float64), f)
                if k == "Vx":
                    strided_vx[s0:s1] = raw[:, :, ::f, ::f]
            print(f"  trajectories {s0}..{s1 - 1} read ({time.time() - t0:.0f}s)", flush=True)
    with open(a.t1, "rb") as fh:
        t1 = pickle.load(fh)
    # consistency: the strided subsample of the same trajectories is the T1 input (sample order 20 i + t)
    X1 = np.asarray(t1["X_data"], dtype=np.float32)
    err = np.abs(strided_vx[:, :N_STEPS].reshape(N_TRAJ * N_STEPS, GS * GS) - X1[..., 1]).max()
    print(f"  strided Vx vs T1 inputs: max abs diff {err:.3e}", flush=True)
    if err > 1e-6:
        raise RuntimeError("trajectory order differs from the T1 data set")
    X = np.stack([coarse[k][:, :N_STEPS] for k in fields], -1).reshape(N_TRAJ * N_STEPS, GS * GS, 4)
    W = curl(coarse["Vx"][:, 1:], coarse["Vy"][:, 1:], 1.0 / GS)                    # vorticity at t + 1
    Y = W.reshape(N_TRAJ * N_STEPS, GS * GS, 1)
    out = dict(points=np.asarray(t1["points"]), faces=np.asarray(t1["faces"]), X_data=X.astype(np.float32),
               Y_data=Y.astype(np.float32), n_nodes=GS * GS, n_samples=N_TRAJ * N_STEPS, grid_size=GS,
               task="cns_vorticity_prediction_blockavg", topology=t1.get("topology"),
               coarse_graining=f"{f}x{f} block averages of the {FINE}x{FINE} fields (T1: every {f}th point)",
               source=os.path.basename(a.src))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print(f"wrote {a.out}: X {X.shape}, Y {Y.shape} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
