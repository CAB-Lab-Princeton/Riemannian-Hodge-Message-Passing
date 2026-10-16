#!/usr/bin/env python3
"""T1 on graded, anisotropic periodic grids (task ``T1g``): the PDEBench trajectories of T1 sampled exactly at the
nodes of non-uniform tensor-product grids, target = the physical vorticity of the next step.

    python3 datasets/generators/gen_T1_graded.py \
        --src <PDEBench 2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5>

Same trajectories, time steps, sample order and split as ``datasets/T1_cns_vorticity.pkl`` (500 trajectories x 20
steps, sample ``20 i + t``; split 7000 / 1500 / 1500 by trajectory).  Every grid is a subset of the 128 x 128
simulation grid: node ``(i, j)`` sits at fine-grid point ``(kx_i, ky_j)``, i.e. at ``((kx_i + 1/2) / 128,
(ky_j + 1/2) / 128)`` on the periodic unit square (the cell centres of the PDEBench grid).  Inputs ``(rho, Vx, Vy, p)``
at step ``t`` are the simulation values at those points (no interpolation); the target is the vorticity
``dVy/dx - dVx/dy`` at step ``t + 1`` computed spectrally on the full 128 x 128 grid (the fields are periodic and
well resolved: second-order central differences on the fine grid differ by 0.56% in relative L2, the coarse 32 x 32
stencil of T1 by 6.6%) and sampled at the same points.  Nothing from the fine grid enters the inputs.

Grids, as spacings in fine cells (``s_i = kx_{i+1} - kx_i``, cyclic; values 1, 2, 4, 8; neighbour ratio <= 2):

* ``G1`` (training): ``sx = 1x8, 2x4, 4x3, 8x10, 4x3, 2x4`` from ``kx_0 = 0``; ``sy = 1x12, 2x2, 4x3, 8x11, 4x2, 2x2``
  from ``ky_0 = 52``.  One cluster of fine cells per axis at different places; cell aspect ratios 1/8 ... 8.
* ``G2`` (zero-shot): two clusters per axis, ``half = 1,1,2,2,2,4,4,4,8,8,8,8,4,4,2,2``, ``sx = half half`` from 20,
  ``sy = half half`` rolled by 4 from 90.
* ``U32``: uniform spacing 4 (the T1 / T1p points, but the fine-grid target).  ``U64``: uniform spacing 2 (64 x 64).
* ``G1rot``: ``G1`` rotated by +90 degrees about the origin of the torus, ``(x, y) -> (-y, x)``: node ``(i', j')`` of the
  rotated grid (numbered ``i' * 32 + j'`` in its own frame) is node ``(j', 31 - i')`` of ``G1``; scalars and the
  vorticity (a pseudo-scalar, invariant under rotations) are carried over, velocities rotated,
  ``(Vx', Vy') = (-Vy, Vx)``; the cells are the rotated cells of ``G1`` (same connectivity and orientation).

Cells: quads ``(i, j), (i+1, j), (i+1, j+1), (i, j+1)`` (counter-clockwise, indices mod n) and triangles from the
quads split along ``(i, j)-(i+1, j+1)`` (the ``'main'`` diagonal of ``CochainComplex.from_grid``).

Output ``datasets/v2/T1_cns_vorticity_graded.pkl``: ``grids`` (per grid: ``kx, ky, sx, sy, x, y, points (n, 2),
quads, triangles, node_spacing (n, 2)`` = mean of the two adjacent spacings per axis in fine cells, ``description``),
``train`` (``G1``: ``X_data (10000, 1024, 4)``, ``Y_data (10000, 1024, 1)``) and ``zero_shot`` (per grid, the 1500 test
samples ``8500 .. 9999``: ``X_data``, ``Y_data``, ``sample_ids``), ``target``, ``source``, ``split``.

``--zero-shot-samples val`` writes ``datasets/v2/T1_cns_vorticity_graded_val.pkl`` instead: ``grids`` and ``zero_shot``
with the 1500 validation samples ``7000 .. 8499`` (trajectories 350 .. 424) on every grid other than ``G1`` (the
validation samples on ``G1`` are part of ``train``), for decisions that must not look at the test trajectories.
"""
from __future__ import annotations

import argparse
import os
import pickle
import time

import numpy as np

N_TRAJ, N_STEPS, FINE = 500, 20, 128
N_TEST0 = 8500                      # first test sample (sequential 70 / 15 / 15 split of the 10000 samples)
N_VAL0 = 7000                       # first validation sample


def spacings_G1() -> tuple[list[int], int, list[int], int]:
    sx = [1] * 8 + [2] * 4 + [4] * 3 + [8] * 10 + [4] * 3 + [2] * 4
    sy = [1] * 12 + [2] * 2 + [4] * 3 + [8] * 11 + [4] * 2 + [2] * 2
    return sx, 0, sy, 52


def spacings_G2() -> tuple[list[int], int, list[int], int]:
    half = [1, 1, 2, 2, 2, 4, 4, 4, 8, 8, 8, 8, 4, 4, 2, 2]
    sx = half + half
    sy = list(np.roll(half + half, 4))
    return sx, 20, sy, 90


def check_spacings(s: list[int]) -> None:
    s = list(s)
    if sum(s) != FINE:
        raise ValueError(f"spacings sum to {sum(s)}, not {FINE}")
    for a, b in zip(s, s[1:] + s[:1]):
        if max(a, b) > 2 * min(a, b):
            raise ValueError(f"neighbour ratio > 2 in {s}")


def tensor_grid(sx, ox, sy, oy, name: str, description: str) -> dict:
    check_spacings(sx)
    check_spacings(sy)
    kx = (ox + np.concatenate([[0], np.cumsum(sx)[:-1]])) % FINE
    ky = (oy + np.concatenate([[0], np.cumsum(sy)[:-1]])) % FINE
    ordx, ordy = np.argsort(kx), np.argsort(ky)                 # number the nodes by increasing coordinate
    kx, ky = kx[ordx], ky[ordy]
    sx = np.diff(np.concatenate([kx, [kx[0] + FINE]]))
    sy = np.diff(np.concatenate([ky, [ky[0] + FINE]]))
    nx, ny = len(kx), len(ky)
    x, y = (kx + 0.5) / FINE, (ky + 0.5) / FINE
    gx, gy = np.meshgrid(x, y, indexing="ij")
    points = np.stack([gx.reshape(-1), gy.reshape(-1)], 1)
    vid = lambda i, j: (i % nx) * ny + (j % ny)                 # noqa: E731
    quads = np.array([[vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)]
                      for i in range(nx) for j in range(ny)], dtype=np.int64)
    tris = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    hx = 0.5 * (sx + np.roll(sx, 1))                            # node spacing: mean of the two adjacent intervals
    hy = 0.5 * (sy + np.roll(sy, 1))
    node_spacing = np.stack(np.meshgrid(hx, hy, indexing="ij"), -1).reshape(-1, 2)
    return dict(name=name, kx=kx, ky=ky, sx=sx, sy=sy, x=x, y=y, points=points, quads=quads, triangles=tris,
                node_spacing=node_spacing, node_perm=None, description=description)


def rotated(g: dict) -> dict:
    """``g`` rotated by +90 degrees, ``(x, y) -> (-y, x)`` mod 1, renumbered in its own frame (see module doc)."""
    nx, ny = len(g["kx"]), len(g["ky"])
    if nx != ny:
        raise ValueError("rotation needs a square node grid")
    n = nx
    ip, jp = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    perm = (jp * n + (n - 1 - ip)).reshape(-1)                  # new node i'*n + j' <- old node (j', n-1-i')
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    old = g["points"][perm]
    points = np.stack([np.mod(-old[:, 1], 1.0), old[:, 0]], 1)
    x, y = np.unique(points[:, 0]), np.unique(points[:, 1])
    gx, gy = np.meshgrid(x, y, indexing="ij")
    if not np.allclose(np.stack([gx.reshape(-1), gy.reshape(-1)], 1), points):
        raise RuntimeError("rotated grid is not numbered i' * n + j' in its own frame")
    out = dict(g)
    out.update(name=g["name"] + "rot", kx=(FINE - 1 - g["ky"])[::-1], ky=g["kx"], sx=np.roll(g["sy"][::-1], -1),
               sy=g["sx"].copy(), x=x, y=y, points=points, quads=inv[g["quads"]], triangles=inv[g["triangles"]],
               node_spacing=g["node_spacing"][perm][:, ::-1].copy(), node_perm=perm,
               description=g["description"] + ", rotated by +90 degrees ((x, y) -> (-y, x))")
    return out


def spectral_curl(vx: np.ndarray, vy: np.ndarray) -> np.ndarray:
    """``dVy/dx - dVx/dy`` of periodic fields ``[..., ix, iy]`` on the unit square (spacing 1/n)."""
    n = vx.shape[-1]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=1.0 / n)
    kx, ky = k[:, None], k[None, :]
    return np.real(np.fft.ifft2(1j * kx * np.fft.fft2(vy) - 1j * ky * np.fft.fft2(vx)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--t1", default=os.path.join(os.path.dirname(__file__), "..", "T1_cns_vorticity.pkl"),
                    help="the T1 pickle (trajectory-order consistency check)")
    ap.add_argument("--out", default=None, help="output pickle (default datasets/v2/T1_cns_vorticity_graded[_val].pkl)")
    ap.add_argument("--zero-shot-samples", default="test", choices=["test", "val"],
                    help="samples on the zero-shot grids: the test split (default; with the training data) or the "
                         "validation split (separate file without the training data)")
    ap.add_argument("--chunk", type=int, default=25)
    a = ap.parse_args()
    val = a.zero_shot_samples == "val"
    if a.out is None:
        a.out = os.path.join(os.path.dirname(__file__), "..", "v2",
                             "T1_cns_vorticity_graded_val.pkl" if val else "T1_cns_vorticity_graded.pkl")
    import h5py
    t0 = time.time()
    g1 = tensor_grid(*spacings_G1(), "G1", "graded training grid (one fine cluster per axis)")
    grids = {"G1": g1,
             "G2": tensor_grid(*spacings_G2(), "G2", "graded grid with two fine clusters per axis"),
             "U32": tensor_grid([4] * 32, 0, [4] * 32, 0, "U32", "uniform 32 x 32 (the T1 points)"),
             "U64": tensor_grid([2] * 64, 0, [2] * 64, 0, "U64", "uniform 64 x 64"),
             "G1rot": rotated(g1)}
    fields = ("density", "Vx", "Vy", "pressure")
    N = N_TRAJ * N_STEPS
    z0, z1 = (N_VAL0, N_TEST0) if val else (N_TEST0, N)                  # zero-shot sample range
    n_te = z1 - z0
    n_tr = 1 if val else N
    train_X = np.empty((n_tr, g1["points"].shape[0], 4), dtype=np.float32)
    train_Y = np.empty((n_tr, g1["points"].shape[0], 1), dtype=np.float32)
    zs = {k: dict(X=np.empty((n_te, g["points"].shape[0], 4), dtype=np.float32),
                  Y=np.empty((n_te, g["points"].shape[0], 1), dtype=np.float32)) for k, g in grids.items() if k != "G1"}
    strided_vx = np.empty((N, 32 * 32), dtype=np.float32)
    with h5py.File(a.src, "r") as h:
        for s0 in range(0, N_TRAJ, a.chunk):
            s1 = min(N_TRAJ, s0 + a.chunk)
            raw = {k: h[k][s0:s1, :N_STEPS + 1].astype(np.float64) for k in fields}      # (c, 21, 128, 128)
            W = spectral_curl(raw["Vx"][:, 1:], raw["Vy"][:, 1:])                      # vorticity at t + 1
            ids = (np.arange(s0, s1)[:, None] * N_STEPS + np.arange(N_STEPS)[None, :]).reshape(-1)
            strided_vx[ids] = raw["Vx"][:, :N_STEPS, ::4, ::4].reshape(len(ids), -1)
            for name, g in grids.items():
                base = g if g["node_perm"] is None else grids[name[:-3]]
                kxg, kyg = np.meshgrid(base["kx"], base["ky"], indexing="ij")
                kxg, kyg = kxg.reshape(-1), kyg.reshape(-1)
                Xg = np.stack([raw[k][:, :N_STEPS][..., kxg, kyg] for k in fields], -1).reshape(len(ids), -1, 4)
                Yg = W[..., kxg, kyg].reshape(len(ids), -1, 1)
                if g["node_perm"] is not None:                                         # rotated copy of the base
                    Xg, Yg = Xg[:, g["node_perm"]], Yg[:, g["node_perm"]]
                    Xg = np.stack([Xg[..., 0], -Xg[..., 2], Xg[..., 1], Xg[..., 3]], -1)
                if name == "G1":
                    if not val:
                        train_X[ids], train_Y[ids] = Xg, Yg
                else:
                    sel = (ids >= z0) & (ids < z1)
                    zs[name]["X"][ids[sel] - z0] = Xg[sel]
                    zs[name]["Y"][ids[sel] - z0] = Yg[sel]
            print(f"  trajectories {s0}..{s1 - 1} ({time.time() - t0:.0f}s)", flush=True)
    with open(a.t1, "rb") as fh:
        t1 = pickle.load(fh)
    err = float(np.abs(strided_vx - np.asarray(t1["X_data"], dtype=np.float32)[..., 1]).max())
    print(f"  strided Vx vs T1 inputs: max abs diff {err:.3e}", flush=True)
    if err > 1e-6:
        raise RuntimeError("trajectory order differs from the T1 data set")
    out = dict(grids=grids, train=None if val else dict(grid="G1", X_data=train_X, Y_data=train_Y),
               zero_shot={k: dict(grid=k, X_data=v["X"], Y_data=v["Y"], sample_ids=np.arange(z0, z1))
                          for k, v in zs.items()},
               zero_shot_split="validation (trajectories 350 .. 424)" if val else "test (trajectories 425 .. 499)",
               fields=("rho", "Vx", "Vy", "p"),
               target="vorticity dVy/dx - dVx/dy at step t + 1, spectral derivative on the 128 x 128 grid, sampled at "
                      "the grid nodes",
               split="sequential 7000 / 1500 / 1500 (by trajectory: 20 consecutive samples per trajectory)",
               source=os.path.basename(a.src), n_samples=N)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "wb") as fh:
        pickle.dump(out, fh, protocol=4)
    print(f"wrote {a.out} ({time.time() - t0:.0f}s): train X {train_X.shape}; zero-shot "
          + ", ".join(f"{k} {v['X'].shape}" for k, v in zs.items()), flush=True)


if __name__ == "__main__":
    main()
