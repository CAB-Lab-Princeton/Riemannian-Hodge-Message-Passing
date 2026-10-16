"""Nonlinear magnetostatics with grain-oriented and laminated (anisotropic) steel in electromagnets and transformers of
several layout families on one fixed lattice mesh (task family MXA, the anisotropic version of MXS).

Physics (2-D cross-section; SI units inside the solver; stored lengths in mm, B in T, current densities in A/mm^2)::

    curl H(B) = J,   B = curl(A e_z) = (dA/dy, -dA/dx)   in the box Omega,   A = 0 on the box boundary,

i.e. A minimises ``Pi(A) = int W(curl A) - int J A``.  The edge fluxes ``Phi_e = A(dst) - A(src)`` are an exactly
closed 1-cochain (``d1 Phi = 0``).

Steel law (energy density, rolling direction ``r = (cos phi, sin phi)`` per face, ``theta`` the angle between B and
``r``)::

    W(B) = w_e(|B|) cos^2 theta + w_h(|B|) sin^2 theta,      H = dW/dB = R(phi) diag(nu_1, nu_2) R(phi)^T B,

with ``w_e``, ``w_h`` the energies of two Froehlich-Kennelly curves ``B = mu_0 H + J_s H / (H_k + H)`` with the same
saturation polarisation ``J_s = 2.0 T`` and the initial relative permeabilities ``mu_e`` (easy, along the rolling
direction) and ``mu_h`` (hard, across).  The secant reluctivity is a symmetric tensor with principal axes along and
across the rolling direction; ``nu_1 = nu_e(|B|)`` for B along ``r`` and ``nu_2 = nu_h(|B|)`` for B across ``r``, and
in the linear regime ``nu = R(phi) diag(nu_e(0), nu_h(0)) R(phi)^T`` exactly.  The ratio ``nu_h / nu_e`` is
``mu_e / mu_h`` up to the knee and falls to 1 in deep saturation (both directions saturate).  This is the
energy-consistent form of the two-curve law ``nu = R diag(nu_e(|B|), nu_h(|B|)) R^T``: the latter is not monotone for
strong anisotropy (the symmetric part of its incremental reluctivity has eigenvalues down to -19 min(nu_e, nu_h) near
the knee at ratio 25), so it has no energy and the field problem need not have a unique solution.  The energy above
is strongly convex (smallest eigenvalue of the incremental reluctivity >= 0.97 min(nu_e, nu_h); ``selftest``), so the
discrete problem has a unique solution and damped Newton on the convex energy converges.

Materials (``phase``): 0 air, 1 copper (both ``nu_0``), 2 NO non-oriented steel (isotropic, mu_r 3000), 3 GO
grain-oriented steel (mu_e 20000, mu_h 800: ratio 25), 4 LAM laminated stack (mu_e 2500, mu_h 500: ratio 5).  The
models receive the phase indicators and the rolling direction of the GO / LAM faces, never the law.

Mesh (one fixed mesh for all layouts): equilateral lattice (spacing 8 mm, rows parallel to x; straight left / right
sides) over the device region [-168, 168] x [-118, 122] mm, graded far field (8 mm + 0.35 x distance, at most 64 mm)
to the box [-400, 400]^2 (A = 0), scipy Delaunay.  ``fine`` (4 mm) and ``coarse`` (16 mm) meshes of the same
construction.  Every part of the device is a per-face material indicator (staircase interfaces on the lattice).

Layout families (one per sample; dimensions, gaps, constructions, rolling directions, coils and currents drawn per
sample; dimensions in mm):

* ``EI`` -- three-limb E core + I armature (electromagnet / actuator): outer limbs 24-40, centre limb 32-72, windows
  32-64 wide and 48-104 high, yoke 24-44; armature 16-40 thick, width W - 16 .. W + 24, offset +-16, gaps 0-21 at
  the two outer limbs (tilted armature when they differ); windings on the centre limb (p 0.9) and on each outer limb
  (p 0.35), independent current densities U(-J_max, J_max).
* ``UI`` -- C (U) core + I armature: limbs 24-48, window 40-128 x 48-104, windings on either or both limbs.
* ``ST`` -- E-I transformer with magnetic shunts (closed core or gap <= 7 mm): primary (bottom) and secondary (top)
  winding on the centre limb, separated by shunt bars (p 0.8; 6-14 thick, attached to the centre or the outer limb with
  a 6-16 air gap); load current ``J_l`` in the primary, ``-J_l A_p / A_s`` in the secondary (ampere-turn balance) plus
  a magnetising current in the primary.
* ``EE`` (held out: never in the training set) -- two E cores facing each other (gapped inductor): centre-limb gap
  0-21, outer limbs touching (p 0.5) or gapped like the centre limb; winding on the centre limb across both windows,
  optional outer windings (|J| <= 6 A/mm^2: the full-height windings carry more ampere-turns).

Constructions of the cores: E / U lamination (one rolling direction along the limbs; armature along its length;
p 0.45), stacked strips with butt joints (limbs along the limbs, yokes along the yokes; p 0.2) or mitred outer corners
(p 0.15), or misaligned stacks (every piece a random rolling direction; p 0.2); in addition every limb is a
misaligned piece with probability 0.1.  Core material per sample (NO 0.25, GO 0.45, LAM 0.30), armature / shunt /
external pieces their own materials.  External steel objects: 0 / 1 / 2 randomly rotated rectangles (16-40 mm) with
random rolling directions beside the device.

Discretisation: P1 finite elements, element-wise constant material, rolling direction and current; consistent load
``F_i = sum_f J_f |f| / 3``; damped Newton on the convex energy (exact symmetric tangent, Armijo backtracking) from
the linear solve with the initial permeabilities, stopped at ``|R_free| <= 1e-10 |F_free|``; sparse direct solves.
Targets: ``B_node`` = area-weighted average of the element fields over the incident triangles (the fixed linear map of
the edge fluxes ``rhmp.tasks.maxwell.FluxToNodeField``).

Sub-commands::

    python3 -u datasets/generators/gen_maxwell_aniso.py data --workers 48   # MXA (5000) + ood15/fine/coarse/heldout
    python3 -u datasets/generators/gen_maxwell_aniso.py selftest            # law, convergence, Newton vs Picard
    python3 -u datasets/generators/gen_maxwell_aniso.py pilot --n 400       # saturation statistics per family
    python3 -u datasets/generators/gen_maxwell_aniso.py mesh --plot         # mesh statistics (+ png of layouts)

Outputs (``datasets/v2/``): ``MXA.pt`` (5000 samples of the families EI / UI / ST, ~1.9K nodes), ``MXA_ood15.pt``
(test samples 4250..4749 with all currents x1.5), ``MXA_fine.pt`` / ``MXA_coarse.pt`` (the same 500 test layouts on
the 4 mm / 16 mm meshes), ``MXA_heldout.pt`` (500 samples of the held-out family EE, base mesh), each with a ``.json``
summary.  Per file: ``pos (n0, 2)`` mm, ``faces (n2, 3)`` (counter-clockwise), ``boundary (n0,)``, ``lattice_face
(n2,)``, and per sample ``phase (N, n2)`` int8, ``phi (N, n2)`` float32 (rolling direction in [0, pi) of the GO / LAM
faces, 0 elsewhere), ``J_face (N, n2)`` float32 A/mm^2, ``family (N,)``, ``A (N, n0)`` float64 Wb/m, ``B_node (N, n0,
2)`` T, Newton statistics and the layout parameters.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay, cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_maxwell import AMM2, MM, MU0, NU0, P1Mesh, _TRI7, _structured_square, _tri_signed_area, node_field  # noqa

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------------------------------------------------
JS = 2.0                                                    # saturation polarisation of every steel (T)
PHASE_AIR, PHASE_COPPER, PHASE_NO, PHASE_GO, PHASE_LAM = 0, 1, 2, 3, 4
PHASE_NAMES = ("air", "copper", "NO", "GO", "LAM")
MATERIALS = {PHASE_NO: ("NO", 3000.0, 3000.0),              # phase: (name, mu_r easy, mu_r hard)
             PHASE_GO: ("GO", 20000.0, 800.0),
             PHASE_LAM: ("LAM", 2500.0, 500.0)}
ORIENTED = (PHASE_GO, PHASE_LAM)

SEED_A = 20261003
N_SAMPLES = 5000
N_EXTRA = 500
TEST_START = 4250
OOD_FACTOR = 1.5
FAMILIES = ("EI", "UI", "ST", "EE")
FAMILY_ID = {f: i for i, f in enumerate(FAMILIES)}
TRAIN_FAMILIES, TRAIN_P = ("EI", "UI", "ST"), (0.4, 0.3, 0.3)
HELDOUT_FAMILY = "EE"
J_MAX_G = 8.0                   # current density of the gapped devices (A/mm^2)
J_LOAD = 6.0                    # transformer load current density (A/mm^2)
J_MAG0, J_MAG1 = 0.5, 0.8       # transformer magnetising current density: U(-1, 1) x (J_MAG0 + J_MAG1 gap_mm)
J_MAX_EE = 6.0                  # current density of the held-out EE inductors (full-height windings; A/mm^2)

S_BASE = 8.0                    # lattice spacing of the base mesh (mm)
LAT_X = (-168.0, 168.0)
LAT_Y0, LAT_Y1 = -118.0, 122.0
BOX_A = (-400.0, 400.0, -400.0, 400.0)
G_FAR, H_FAR = 0.35, 64.0
SCALES_A = {"base": 1.0, "fine": 0.5, "coarse": 2.0}
MESH_SEEDS_A = {"base": 31, "fine": 32, "coarse": 33}
DEV_X = (-140.0, 140.0)         # device envelope (external pieces go between it and the lattice border)
DEV_Y = (-110.0, 114.0)
CLEAR = 4.0                     # coil clearance (mm)

NEWTON_TOL = 1e-10
NEWTON_MAX = 100
B_LIN = 1e-8                    # |B| below which the law is evaluated in its linear limit (T)


# ---------------------------------------------------------------------------------------------------------------------
# material law
# ---------------------------------------------------------------------------------------------------------------------
def fk_hk(mur: float) -> float:
    """Knee field ``H_k = J_s / (mu_0 (mu_r - 1))`` (A/m) of a Froehlich-Kennelly curve with initial ``mu_r``."""
    return JS / (MU0 * (mur - 1.0))


def fk_nu(B, hk):
    """Reluctivity ``H(B) / B`` (m/H) of the curve ``B = mu_0 H + J_s H / (H_k + H)``."""
    b = MU0 * hk + JS - B
    return 2.0 * hk / (b + np.sqrt(b * b + 4.0 * MU0 * hk * B))


def fk_dnu(B, hk):
    """``d nu / d B``."""
    b = MU0 * hk + JS - B
    s = np.sqrt(b * b + 4.0 * MU0 * hk * B)
    return -2.0 * hk / (b + s) ** 2 * (-1.0 + (2.0 * MU0 * hk - b) / s)


def fk_energy(B, hk):
    """``w(B) = int_0^B H dB = mu_0 H^2 / 2 + J_s H_k psi(H / H_k)``, ``psi(x) = x^2 / (1 + x) - x + log(1 + x)``
    (series for small x: no cancellation)."""
    H = fk_nu(B, hk) * B
    x = H / hk
    xs = np.minimum(x, 1e-3)
    series = xs * xs * (0.5 - xs * (2.0 / 3.0 - xs * (0.75 - xs * (0.8 - xs * 5.0 / 6.0))))
    xl = np.maximum(x, 1e-3)
    closed = xl * xl / (1.0 + xl) - xl + np.log1p(xl)
    return 0.5 * MU0 * H * H + JS * hk * np.where(x < 1e-3, series, closed)


def law(B: np.ndarray, phase: np.ndarray, phi: np.ndarray, hessian: bool = True, secant: bool = False) -> dict:
    """Field-dependent law of every face: ``H (n, 2)`` (A/m), energy density ``W (n,)`` (J/m^3), incremental
    reluctivity ``dH/dB (n, 2, 2)`` (symmetric) and, with ``secant=True``, the secant tensor ``nu (n, 2, 2)`` with
    ``H = nu B``."""
    n = len(B)
    out = dict(H=NU0 * B, W=0.5 * NU0 * (B * B).sum(1))
    eye = np.broadcast_to(NU0 * np.eye(2), (n, 2, 2))
    if hessian:
        out["dH"] = eye.copy()
    if secant:
        out["nu"] = eye.copy()
    for cls, (_, mue, muh) in MATERIALS.items():
        sel = np.where(phase == cls)[0]
        if not len(sel):
            continue
        hke, hkh = fk_hk(mue), fk_hk(muh)
        Bs, ph = B[sel], phi[sel]
        c, s = np.cos(ph), np.sin(ph)
        p = Bs[:, 0] * c + Bs[:, 1] * s                                    # component along the rolling direction
        q = -Bs[:, 0] * s + Bs[:, 1] * c                                   # component across
        b = np.hypot(p, q)
        lin = b < B_LIN
        bs = np.where(lin, 1.0, b)
        nue, nuh = fk_nu(b, hke), fk_nu(b, hkh)
        dnue, dnuh = fk_dnu(b, hke), fk_dnu(b, hkh)
        we, wh = fk_energy(b, hke), fk_energy(b, hkh)
        fp, fpp = nue * b, nue + b * dnue                                  # f = w_e: f' = H_e, f'' = dH_e / dB
        D, Dp, Dpp = wh - we, (nuh - nue) * b, (nuh - nue) + b * (dnuh - dnue)
        g = D / bs ** 2                                                    # W = f(b) + g(b) q^2
        gp = Dp / bs ** 2 - 2.0 * D / bs ** 3
        gpp = Dpp / bs ** 2 - 4.0 * Dp / bs ** 3 + 6.0 * D / bs ** 4
        A1 = fp + gp * q * q
        Wp = A1 * p / bs
        Wq = A1 * q / bs + 2.0 * g * q
        W = we + g * q * q
        nu0e, nu0h = fk_nu(0.0, hke), fk_nu(0.0, hkh)                      # linear limit
        Wp = np.where(lin, nu0e * p, Wp)
        Wq = np.where(lin, nu0h * q, Wq)
        W = np.where(lin, 0.5 * (nu0e * p * p + nu0h * q * q), W)
        out["H"][sel, 0] = Wp * c - Wq * s
        out["H"][sel, 1] = Wp * s + Wq * c
        out["W"][sel] = W
        Q = np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2)    # columns r, n
        if hessian:
            def rad(Fp, Fpp):
                return (Fpp * p * p / bs ** 2 + Fp * q * q / bs ** 3, Fpp * q * q / bs ** 2 + Fp * p * p / bs ** 3,
                        (Fpp - Fp / bs) * p * q / bs ** 2)
            f_pp, f_qq, f_pq = rad(fp, fpp)
            g_pp, g_qq, g_pq = rad(gp, gpp)
            Wpp = f_pp + q * q * g_pp
            Wqq = f_qq + q * q * g_qq + 4.0 * q * (gp * q / bs) + 2.0 * g
            Wpq = f_pq + q * q * g_pq + 2.0 * q * (gp * p / bs)
            Wpp, Wqq, Wpq = np.where(lin, nu0e, Wpp), np.where(lin, nu0h, Wqq), np.where(lin, 0.0, Wpq)
            Hl = np.stack([np.stack([Wpp, Wpq], -1), np.stack([Wpq, Wqq], -1)], -2)
            out["dH"][sel] = Q @ Hl @ Q.transpose(0, 2, 1)
        if secant:
            a = np.where(lin, nu0e, A1 / bs)
            a2 = np.where(lin, nu0h, A1 / bs + 2.0 * g)
            Sl = np.zeros((len(sel), 2, 2))
            Sl[:, 0, 0], Sl[:, 1, 1] = a, a2
            out["nu"][sel] = Q @ Sl @ Q.transpose(0, 2, 1)
    return out


def linear_tensor(phase: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Low-field reluctivity tensor ``R(phi) diag(nu_e(0), nu_h(0)) R(phi)^T`` per face ``(n, 2, 2)``."""
    return law(np.zeros((len(phase), 2)), phase, phi, hessian=True)["dH"]


def true_tensor(Bf: np.ndarray, phase: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Secant reluctivity tensor ``nu(B)`` per face (``H = nu B``)."""
    return law(Bf, phase, phi, hessian=False, secant=True)["nu"]


# ---------------------------------------------------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------------------------------------------------
def lattice_points(scale: float) -> tuple[np.ndarray, float]:
    """Equilateral lattice of the device region (rows parallel to x, odd rows shifted by s/2 and closed by points on
    the straight left / right sides).  Returns the points and the y of the top row."""
    s = S_BASE * scale
    h = s * math.sqrt(3.0) / 2.0
    rows, j = [], 0
    while LAT_Y0 + j * h <= LAT_Y1 + 1e-9:
        y = LAT_Y0 + j * h
        if j % 2 == 0:
            xs = np.arange(LAT_X[0], LAT_X[1] + 1e-9, s)
        else:
            xs = np.concatenate([[LAT_X[0]], np.arange(LAT_X[0] + 0.5 * s, LAT_X[1] - 0.5 * s + 1e-9, s), [LAT_X[1]]])
        rows.append(np.stack([xs, np.full(len(xs), y)], 1))
        j += 1
    return np.concatenate(rows), LAT_Y0 + (j - 1) * h


def _dist_rect(X: np.ndarray, r) -> np.ndarray:
    dx = np.maximum(np.maximum(r[0] - X[:, 0], X[:, 0] - r[1]), 0.0)
    dy = np.maximum(np.maximum(r[2] - X[:, 1], X[:, 1] - r[3]), 0.0)
    return np.hypot(dx, dy)


def far_size(X: np.ndarray, rect, scale: float) -> np.ndarray:
    return scale * np.minimum(S_BASE + G_FAR * _dist_rect(X, rect), H_FAR)


def build_lattice_mesh(scale: float = 1.0, seed: int = 31) -> dict:
    """The fixed mesh of the MXA layouts: the lattice of :func:`lattice_points` (spacing ``8 scale`` mm) inside the
    device region, graded far-field points (dart throwing with minimum spacing ``0.8 h``, none closer than ``0.75 h``
    to the lattice region) and graded points on the box, scipy Delaunay, counter-clockwise triangles.  Checks: disk
    topology, boundary = box vertices, every triangle inside the lattice region is an equilateral lattice triangle."""
    rng = np.random.default_rng([seed, int(round(scale * 1000))])
    lat, ytop = lattice_points(scale)
    rect = (LAT_X[0], LAT_X[1], LAT_Y0, ytop)
    s = S_BASE * scale
    # box boundary points (graded)
    bpts = []
    corners = np.array([[BOX_A[0], BOX_A[2]], [BOX_A[1], BOX_A[2]], [BOX_A[1], BOX_A[3]], [BOX_A[0], BOX_A[3]]])
    for k in range(4):
        a, b = corners[k], corners[(k + 1) % 4]
        L = float(np.linalg.norm(b - a))
        u = np.linspace(0.0, 1.0, 513)
        hh = far_size(a[None] + u[:, None] * (b - a)[None], rect, scale)
        Ncum = np.concatenate([[0.0], np.cumsum(0.5 * (1.0 / hh[1:] + 1.0 / hh[:-1]) * np.diff(u) * L)])
        n = max(1, int(round(Ncum[-1])))
        t = np.interp(np.arange(n) * Ncum[-1] / n, Ncum, u)
        bpts.append(a[None] + t[:, None] * (b - a)[None])
    bpts = np.concatenate(bpts)
    # far-field candidates with density ~ 1 / h^2
    area_box = (BOX_A[1] - BOX_A[0]) * (BOX_A[3] - BOX_A[2])
    n_cand = int(6.0 * 1.155 * area_box / s ** 2 / 3.0)
    C = np.stack([rng.uniform(BOX_A[0], BOX_A[1], n_cand), rng.uniform(BOX_A[2], BOX_A[3], n_cand)], 1)
    hc = far_size(C, rect, scale)
    keep = (rng.random(n_cand) < (s / hc) ** 2) & (_dist_rect(C, rect) > 0.75 * hc)
    keep &= (C[:, 0] > BOX_A[0] + 0.5 * hc) & (C[:, 0] < BOX_A[1] - 0.5 * hc) & \
            (C[:, 1] > BOX_A[2] + 0.5 * hc) & (C[:, 1] < BOX_A[3] - 0.5 * hc)
    C, hc = C[keep], hc[keep]
    order = rng.permutation(len(C))
    C, hc = C[order], hc[order]
    fixed = np.concatenate([lat, bpts])
    allp = np.concatenate([fixed, C])
    nbrs = cKDTree(allp).query_ball_point(C, r=0.8 * hc)
    nf = len(fixed)
    acc = np.zeros(len(allp), dtype=bool)
    acc[:nf] = True
    for i, nb in enumerate(nbrs):
        ii = nf + i
        acc[ii] = not any(q != ii and acc[q] for q in nb)
    X = np.concatenate([lat, bpts, C[acc[nf:]]])
    F = Delaunay(X).simplices.astype(np.int64)
    sa = _tri_signed_area(X, F)
    F[sa < 0] = F[sa < 0][:, [0, 2, 1]]
    F = F[np.abs(sa) > 1e-9 * s * s]
    n0 = len(X)
    n_lat = len(lat)
    cent = X[F].mean(1)
    in_rect = (cent[:, 0] > rect[0]) & (cent[:, 0] < rect[1]) & (cent[:, 1] > rect[2]) & (cent[:, 1] < rect[3])
    if not (F[in_rect] < n_lat).all():
        raise RuntimeError("a far-field point lies inside the lattice region")
    Q = X[F]
    lens = np.stack([np.linalg.norm(Q[:, (k + 1) % 3] - Q[:, k], axis=1) for k in range(3)], 1)
    eq = in_rect & (np.abs(lens - s).max(1) < 1e-6 * s)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    ek, cnt = np.unique(E[:, 0] * n0 + E[:, 1], return_counts=True)
    bverts = np.unique(np.stack([ek[cnt == 1] // n0, ek[cnt == 1] % n0], 1))
    bmask = np.zeros(n0, dtype=bool)
    bmask[bverts] = True
    on_box = (np.abs(X[:, 0] - BOX_A[0]) < 1e-9) | (np.abs(X[:, 0] - BOX_A[1]) < 1e-9) | \
             (np.abs(X[:, 1] - BOX_A[2]) < 1e-9) | (np.abs(X[:, 1] - BOX_A[3]) < 1e-9)
    if not np.array_equal(bmask, on_box):
        raise RuntimeError("boundary vertices of the triangulation differ from the box vertices")
    ang = []
    for k in range(3):
        a_, b_ = Q[:, (k + 1) % 3] - Q[:, k], Q[:, (k + 2) % 3] - Q[:, k]
        ang.append(np.degrees(np.arccos(np.clip((a_ * b_).sum(1) / np.linalg.norm(a_, axis=1) /
                                                np.linalg.norm(b_, axis=1), -1, 1))))
    ang = np.stack(ang, 1)
    used = np.unique(F)
    stats = dict(scale=scale, seed=seed, spacing_mm=s, n0=int(n0), n1=int(len(ek)), n2=int(len(F)),
                 euler=int(n0 - len(ek) + len(F)), n_lattice=int(n_lat), n_far=int(n0 - n_lat - len(bpts)),
                 n_box=int(len(bpts)), n_lattice_faces=int(in_rect.sum()), n_equilateral=int(eq.sum()),
                 unused=int(n0 - len(used)), nonmanifold=int((cnt > 2).sum()), min_angle=float(ang.min()),
                 min_angle_p1=float(np.percentile(ang.min(1), 1)), max_angle=float(ang.max()),
                 frac_obtuse=float((ang.max(1) > 90.0).mean()), lattice_rect=[float(v) for v in rect])
    if stats["euler"] != 1 or stats["unused"] or stats["nonmanifold"]:
        raise RuntimeError(f"invalid mesh: {stats}")
    return dict(pos=X, faces=F, boundary=on_box, lattice_face=in_rect, stats=stats)


# ---------------------------------------------------------------------------------------------------------------------
# layouts: pieces (shape, material, rolling direction) and coils (shape, current density)
# ---------------------------------------------------------------------------------------------------------------------
def _rect(x0, x1, y0, y1) -> dict:
    return dict(kind="rect", v=[float(x0), float(x1), float(y0), float(y1)])


def _rrect(cx, cy, a, b, th) -> dict:
    return dict(kind="rrect", v=[float(cx), float(cy), float(a), float(b), float(th)])


def _poly(P) -> dict:
    return dict(kind="poly", v=[[float(x), float(y)] for x, y in P])


def _inside(shape: dict, X: np.ndarray) -> np.ndarray:
    k, v = shape["kind"], shape["v"]
    if k == "rect":
        return (X[:, 0] > v[0]) & (X[:, 0] < v[1]) & (X[:, 1] > v[2]) & (X[:, 1] < v[3])
    if k == "rrect":
        d = X - np.array(v[:2])[None]
        u = d[:, 0] * math.cos(v[4]) + d[:, 1] * math.sin(v[4])
        w = -d[:, 0] * math.sin(v[4]) + d[:, 1] * math.cos(v[4])
        return (np.abs(u) < 0.5 * v[2]) & (np.abs(w) < 0.5 * v[3])
    from matplotlib.path import Path
    return Path(np.array(v)).contains_points(X)


def _material(rng, p=(0.25, 0.45, 0.30)) -> int:
    return int(rng.choice([PHASE_NO, PHASE_GO, PHASE_LAM], p=p))


def _construction(rng) -> str:
    return str(rng.choice(["lam", "strip_butt", "strip_mitre", "random"], p=[0.45, 0.2, 0.15, 0.2]))


def _core(rng, kind: str, xc: float, y0: float, wo: float, wc: float, ww: float, hw: float, ty: float, up: bool,
          cls: int, cons: str, limb_len: list | None = None) -> tuple[list, dict]:
    """Pieces of an E (``kind='E'``: yoke + three limbs) or U core (``'U'``: yoke + two limbs; ``wc`` unused).
    ``up=True``: yoke at the bottom (``y0`` = its bottom), limbs up to the pole faces; ``up=False``: mirrored
    (``y0`` = top of the yoke, limbs down).  ``limb_len``: per-limb lengths beyond the yoke (default ``hw``).
    Returns ``(pieces, geometry)``: pieces = list of (shape, material, rolling direction or None)."""
    W = 2 * wo + (wc if kind == "E" else 0.0) + (2 * ww if kind == "E" else ww)
    xl, xr = xc - W / 2, xc + W / 2
    if kind == "E":
        limbs = [(xl, xl + wo), (xc - wc / 2, xc + wc / 2), (xr - wo, xr)]
        windows = [(xl + wo, xc - wc / 2), (xc + wc / 2, xr - wo)]
    else:
        limbs = [(xl, xl + wo), (xr - wo, xr)]
        windows = [(xl + wo, xr - wo)]
    ll = limb_len or [hw] * len(limbs)
    if up:
        yk = (y0, y0 + ty)
        lys = [(yk[1], yk[1] + L) for L in ll]
    else:
        yk = (y0 - ty, y0)
        lys = [(yk[0] - L, yk[0]) for L in ll]
    lim_phi = math.pi / 2
    if cons == "lam":
        phis = [lim_phi] * (1 + len(limbs))
    elif cons in ("strip_butt", "strip_mitre"):
        phis = [0.0] + [lim_phi] * len(limbs)
    else:
        phis = list(rng.uniform(0.0, math.pi, 1 + len(limbs)))
    for j in range(len(limbs)):                                    # misaligned limb stacks
        if cons != "random" and rng.random() < 0.1:
            phis[1 + j] = float(rng.uniform(0.0, math.pi))
    pieces = [(_rect(xl, xr, yk[0], yk[1]), cls, phis[0])]
    for (a, b), (c0, c1), ph in zip(limbs, lys, phis[1:]):
        pieces.append((_rect(a, b, c0, c1), cls, ph))
    if cons == "strip_mitre":                                     # outer corners: limb-side triangle of the diagonal
        (a0, a1), (b0, b1) = limbs[0], limbs[-1]
        if up:
            tl = [(a0, yk[0]), (a1, yk[1]), (a0, yk[1])]
            tr = [(b1, yk[0]), (b1, yk[1]), (b0, yk[1])]
        else:
            tl = [(a0, yk[1]), (a0, yk[0]), (a1, yk[0])]
            tr = [(b1, yk[1]), (b0, yk[0]), (b1, yk[0])]
        pieces += [(_poly(tl), cls, phis[1]), (_poly(tr), cls, phis[-1])]
    geo = dict(W=W, xl=xl, xr=xr, limbs=limbs, windows=windows, yoke=yk, limb_y=lys,
               pole=(max(c1 for _, c1 in lys) if up else min(c0 for c0, _ in lys)))
    return pieces, geo


def _armature(rng, xa, wa, ta, y_left, y_right, x_left, x_right, cls, cons) -> tuple:
    """I armature whose bottom edge passes through ``(x_left, y_left)`` and ``(x_right, y_right)``."""
    th = math.atan2(y_right - y_left, x_right - x_left)
    yb = y_left + (xa - x_left) * math.tan(th)                     # bottom edge at x = xa
    cx, cy = xa - 0.5 * ta * math.sin(th), yb + 0.5 * ta * math.cos(th)
    phi = (th % math.pi) if cons != "random" else float(rng.uniform(0.0, math.pi))
    return (_rrect(cx, cy, wa, ta, th), cls, phi)


def _external(rng, n: int, bbox) -> list:
    """External steel objects: randomly rotated rectangles between the device and the lattice border."""
    out, tries = [], 0
    while len(out) < n and tries < 400:
        tries += 1
        a, b = rng.uniform(16.0, 40.0, 2)
        th = float(rng.uniform(0.0, math.pi))
        r = 0.5 * math.hypot(a, b)
        c = np.array([rng.uniform(LAT_X[0] + 10 + r, LAT_X[1] - 10 - r), rng.uniform(LAT_Y0 + 10 + r, LAT_Y1 - 14 - r)])
        if (bbox[0] - 10 - r < c[0] < bbox[1] + 10 + r) and (bbox[2] - 10 - r < c[1] < bbox[3] + 10 + r):
            continue
        if any(np.hypot(*(c - np.array(o[0]["v"][:2]))) < r + o[3] + 8.0 for o in out):
            continue
        out.append((_rrect(c[0], c[1], a, b, th), _material(rng, (1 / 3, 1 / 3, 1 / 3)), float(rng.uniform(0, math.pi)),
                    r))
    return [o[:3] for o in out]


def _coil_block(x0, x1, y0, y1, J) -> dict:
    return dict(shape=_rect(x0, x1, y0, y1), J=float(J))


def _limb_winding(rng, geo, j: int, J: float, y0: float, y1: float, width_frac=(0.3, 0.45), side_room=None) -> list:
    """Two coil blocks around limb ``j`` (go and return, +J on the left side) between heights ``y0..y1``."""
    a, b = geo["limbs"][j]
    rooms = []
    for side in (-1, 1):
        room = None
        for (w0, w1) in geo["windows"]:                            # window on this side of the limb
            if (side < 0 and abs(w1 - a) < 1e-9) or (side > 0 and abs(w0 - b) < 1e-9):
                room = w1 - w0
        rooms.append(room if room is not None else (side_room if side_room is not None else 48.0))
    cw = float(rng.uniform(*width_frac)) * min(rooms)              # equal blocks: zero net current per winding
    blocks = []
    for side, sgn in ((-1, 1.0), (1, -1.0)):
        x_edge = a if side < 0 else b
        x0, x1 = (x_edge - CLEAR - cw, x_edge - CLEAR) if side < 0 else (x_edge + CLEAR, x_edge + CLEAR + cw)
        blocks.append(_coil_block(x0, x1, y0, y1, sgn * J))
    return blocks


def _coil_span(rng, lo: float, hi: float, frac=(0.55, 0.9)) -> tuple[float, float]:
    L = hi - lo - 2 * CLEAR
    h = float(rng.uniform(*frac)) * L
    y0 = lo + CLEAR + float(rng.uniform(0.0, L - h))
    return y0, y0 + h


def _family_ei(rng, kind: str) -> dict:
    for _ in range(1000):
        if kind == "E":
            wo = rng.uniform(24, 40)
            wc = float(np.clip(wo * rng.uniform(1.3, 2.2), 32, 72))
            ww = rng.uniform(32, 64)
        else:
            wo = rng.uniform(24, 48)
            wc = 0.0
            ww = rng.uniform(40, 128)
        hw, ta = rng.uniform(48, 104), rng.uniform(16, 40)
        ty = float(np.clip(wo * rng.uniform(0.8, 1.25), 24, 44 if kind == "E" else 48))
        W = 2 * wo + wc + (2 * ww if kind == "E" else ww)
        if rng.random() < 0.6:
            gL, gR = rng.uniform(0, 21, 2)
        else:
            gL = gR = rng.uniform(0, 21)
        H = ty + hw + max(gL, gR) + ta
        if W <= 272 and H <= 214:
            break
    yb = -0.5 * (ty + hw + 0.5 * (gL + gR) + ta) + rng.uniform(-6, 6)
    cls, cons = _material(rng), _construction(rng)
    pieces, geo = _core(rng, kind, 0.0, yb, wo, wc, ww, hw, ty, True, cls, cons)
    wa, xa = W + rng.uniform(-16, 24), rng.uniform(-16, 16)
    cls_a = cls if rng.random() < 0.6 else _material(rng)
    xL, xR = 0.5 * sum(geo["limbs"][0]), 0.5 * sum(geo["limbs"][-1])
    pole = geo["pole"]
    pieces.append(_armature(rng, xa, wa, ta, pole + gL, pole + gR, xL, xR, cls_a, cons))
    coils, windings = [], []
    if kind == "E":
        want = [rng.random() < 0.35, rng.random() < 0.9, rng.random() < 0.35]
    else:
        want = [rng.random() < 0.7, rng.random() < 0.7]
    if not any(want):
        want[1 if kind == "E" else int(rng.integers(2))] = True
    shared = kind == "E" and (want[1] and (want[0] or want[2]))
    for j, on in enumerate(want):
        if not on:
            continue
        J = float(rng.uniform(-J_MAX_G, J_MAX_G))
        y0, y1 = _coil_span(rng, geo["yoke"][1], pole)
        frac = (0.3, 0.45) if (shared or kind == "U") else (0.35, 0.8)
        side_room = min(geo["xl"] - (LAT_X[0] + 12), 48.0)
        coils += _limb_winding(rng, geo, j, J, y0, y1, frac, side_room)
        windings.append(dict(limb=j, J=J))
    top = max(pole + gL, pole + gR) + ta
    bbox = (min(geo["xl"], xa - wa / 2) - 56, max(geo["xr"], xa + wa / 2) + 56, yb, top + 4)
    n_ext = int(rng.choice(3, p=[0.45, 0.35, 0.2]))
    pieces += _external(rng, n_ext, bbox)
    return dict(dims=dict(wo=wo, wc=wc, ww=ww, hw=hw, ty=ty, ta=ta, wa=wa, xa=xa, gL=gL, gR=gR, W=W, yb=yb),
                core_material=cls, armature_material=cls_a, construction=cons, pieces=pieces, coils=coils,
                windings=windings, n_external=n_ext)


def _family_st(rng) -> dict:
    for _ in range(1000):
        wo = rng.uniform(24, 40)
        wc = float(np.clip(wo * rng.uniform(1.3, 2.2), 32, 72))
        ww, hw, ta = rng.uniform(40, 64), rng.uniform(64, 104), rng.uniform(16, 40)
        ty = float(np.clip(wo * rng.uniform(0.8, 1.25), 24, 44))
        W = 2 * wo + wc + 2 * ww
        g = 0.0 if rng.random() < 0.5 else float(rng.uniform(0, 7))
        if W <= 272 and ty + hw + g + ta <= 214:
            break
    yb = -0.5 * (ty + hw + g + ta) + rng.uniform(-6, 6)
    cls, cons = _material(rng), _construction(rng)
    pieces, geo = _core(rng, "E", 0.0, yb, wo, wc, ww, hw, ty, True, cls, cons)
    wa, xa = W + rng.uniform(0, 16), rng.uniform(-4, 4)
    cls_a = cls if rng.random() < 0.7 else _material(rng)
    pole = geo["pole"]
    xL, xR = 0.5 * sum(geo["limbs"][0]), 0.5 * sum(geo["limbs"][-1])
    pieces.append(_armature(rng, xa, wa, ta, pole + g, pole + g, xL, xR, cls_a, cons))
    ys = geo["yoke"][1] + float(rng.uniform(0.4, 0.6)) * hw
    shunt = rng.random() < 0.8
    ts = float(rng.uniform(6, 14)) if shunt else float(rng.uniform(6, 10))
    sh = None
    if shunt:
        gs, att = float(rng.uniform(6, 16)), str(rng.choice(["centre", "outer"]))
        cls_s = _material(rng)
        phs = 0.0 if rng.random() < 0.8 else float(rng.uniform(0, math.pi))
        for (w0, w1), left in zip(geo["windows"], (True, False)):
            at_centre = (att == "centre")
            if left:
                x0, x1 = (w0 + gs, w1) if at_centre else (w0, w1 - gs)
            else:
                x0, x1 = (w0, w1 - gs) if at_centre else (w0 + gs, w1)
            pieces.append((_rect(x0, x1, ys, ys + ts), cls_s, phs))
        sh = dict(gap=gs, attached=att, material=cls_s, phi=phs)
    frac = (0.45, 0.8)
    Jl = float(rng.uniform(-J_LOAD, J_LOAD))
    Jm = float(rng.uniform(-1, 1)) * (J_MAG0 + J_MAG1 * g)
    yp0, yp1 = _coil_span(rng, geo["yoke"][1], ys, (0.6, 0.95))
    ys0, ys1 = _coil_span(rng, ys + ts, pole, (0.6, 0.95))
    cwp = [float(rng.uniform(*frac)) * ww] * 2                      # equal blocks: zero net current per winding
    cws = [float(rng.uniform(*frac)) * ww] * 2
    a, b = geo["limbs"][1]
    Ap = sum(cwp) * (yp1 - yp0)
    As = sum(cws) * (ys1 - ys0)
    Jp, Js_ = Jl + Jm, -Jl * Ap / As
    coils = [_coil_block(a - CLEAR - cwp[0], a - CLEAR, yp0, yp1, Jp),
             _coil_block(b + CLEAR, b + CLEAR + cwp[1], yp0, yp1, -Jp),
             _coil_block(a - CLEAR - cws[0], a - CLEAR, ys0, ys1, Js_),
             _coil_block(b + CLEAR, b + CLEAR + cws[1], ys0, ys1, -Js_)]
    top = pole + g + ta
    bbox = (min(geo["xl"], xa - wa / 2) - 20, max(geo["xr"], xa + wa / 2) + 20, yb, top + 4)
    n_ext = int(rng.choice(3, p=[0.45, 0.35, 0.2]))
    pieces += _external(rng, n_ext, bbox)
    return dict(dims=dict(wo=wo, wc=wc, ww=ww, hw=hw, ty=ty, ta=ta, wa=wa, xa=xa, gap=g, ys=ys, ts=ts, W=W, yb=yb),
                core_material=cls, armature_material=cls_a, construction=cons, shunt=sh, pieces=pieces, coils=coils,
                windings=[dict(name="primary", J=Jp, J_load=Jl, J_mag=Jm), dict(name="secondary", J=Js_)],
                n_external=n_ext)


def _family_ee(rng) -> dict:
    for _ in range(1000):
        wo = rng.uniform(24, 40)
        wc = float(np.clip(wo * rng.uniform(1.3, 2.2), 32, 72))
        ww = rng.uniform(32, 64)
        hl, hu = rng.uniform(32, 64, 2)
        tl, tu = rng.uniform(24, 40, 2)
        gc = float(rng.uniform(0, 21))
        go = 0.0 if rng.random() < 0.5 else gc
        W = 2 * wo + wc + 2 * ww
        gmax = max(gc, go)
        if W <= 272 and tl + hl + gmax + hu + tu <= 214:
            break
    yb = -0.5 * (tl + hl + gmax + hu + tu) + rng.uniform(-6, 6)
    cls, cons = _material(rng), _construction(rng)
    pieces, geo = _core(rng, "E", 0.0, yb, wo, wc, ww, hl, tl, True, cls, cons)
    pole = geo["pole"]
    y_ub = pole + gmax + hu                                       # bottom of the upper yoke
    cls_u = cls if rng.random() < 0.7 else _material(rng)
    cons_u = cons if rng.random() < 0.7 else _construction(rng)
    lens = [y_ub - (pole + go), y_ub - (pole + gc), y_ub - (pole + go)]
    pu, geo_u = _core(rng, "E", 0.0, y_ub + tu, wo, wc, ww, hu, tu, False, cls_u, cons_u, limb_len=lens)
    pieces += pu
    coils, windings = [], []
    want = [rng.random() < 0.3, True, rng.random() < 0.3]
    for j, on in enumerate(want):
        if not on:
            continue
        J = float(rng.uniform(-J_MAX_EE, J_MAX_EE))
        y0, y1 = _coil_span(rng, geo["yoke"][1], y_ub, (0.6, 0.92))
        frac = (0.3, 0.45) if (want[0] or want[2]) else (0.35, 0.8)
        coils += _limb_winding(rng, geo, j, J, y0, y1, frac, min(geo["xl"] - (LAT_X[0] + 12), 48.0))
        windings.append(dict(limb=j, J=J))
    bbox = (geo["xl"] - 56, geo["xr"] + 56, yb, y_ub + tu + 4)
    n_ext = int(rng.choice(3, p=[0.45, 0.35, 0.2]))
    pieces += _external(rng, n_ext, bbox)
    return dict(dims=dict(wo=wo, wc=wc, ww=ww, h_lower=hl, h_upper=hu, ty_lower=tl, ty_upper=tu, gap_centre=gc,
                          gap_outer=go, W=W, yb=yb), core_material=cls, upper_material=cls_u, construction=cons,
                upper_construction=cons_u, pieces=pieces, coils=coils, windings=windings, n_external=n_ext)


def draw_layout(idx: int, heldout: bool = False, seed: int = SEED_A) -> dict:
    """Layout and excitation of sample ``idx`` of the main set (families EI / UI / ST) or of the held-out set (EE)."""
    rng = np.random.default_rng([seed + (1 if heldout else 0), idx])
    fam = HELDOUT_FAMILY if heldout else str(rng.choice(TRAIN_FAMILIES, p=TRAIN_P))
    if fam == "EI":
        p = _family_ei(rng, "E")
    elif fam == "UI":
        p = _family_ei(rng, "U")
    elif fam == "ST":
        p = _family_st(rng)
    else:
        p = _family_ee(rng)
    p["family"] = fam
    p["idx"] = idx
    return p


def paint(mesh: dict, p: dict, factor: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-face phase (int8), rolling direction (radians in [0, pi), 0 for non-oriented faces) and current density
    (A/mm^2) of layout ``p`` on ``mesh`` (centroid tests; coils first, steel pieces painted over them in order).  The
    current density of a coil block is scaled so that its painted current equals the nominal one (block area times
    current density): every winding carries zero net current on every mesh."""
    X, F = mesh["pos"], mesh["faces"]
    cent = X[F].mean(1)
    phase = np.zeros(len(F), dtype=np.int8)
    phi = np.zeros(len(F))
    J = np.zeros(len(F))
    area = 0.5 * np.abs(_tri_signed_area(X, F))
    coil_id = np.full(len(F), -1)
    for k, c in enumerate(p["coils"]):
        sel = _inside(c["shape"], cent)
        phase[sel] = PHASE_COPPER
        coil_id[sel] = k
    for shape, cls, ph in p["pieces"]:
        sel = _inside(shape, cent)
        phase[sel] = cls
        phi[sel] = (ph % math.pi) if ph is not None else 0.0
        coil_id[sel] = -1
    for k, c in enumerate(p["coils"]):                    # painted current = nominal current of the block
        sel = coil_id == k
        if sel.any():
            v = c["shape"]["v"]
            J[sel] = factor * c["J"] * (v[1] - v[0]) * (v[3] - v[2]) / area[sel].sum()
    phi[~np.isin(phase, ORIENTED)] = 0.0
    return phase, phi, J


# ---------------------------------------------------------------------------------------------------------------------
# P1 finite elements + Newton
# ---------------------------------------------------------------------------------------------------------------------
def curl_basis(m: P1Mesh) -> np.ndarray:
    """``(n2, 3, 2)``: ``B_f = sum_i A_i Gb[f, i]`` with ``Gb = R(-90 deg) grad lambda_i`` (``B = (dA/dy, -dA/dx)``)."""
    return np.stack([m.G[:, :, 1], -m.G[:, :, 0]], -1)


def newton_solve_a(m: P1Mesh, phase: np.ndarray, phi: np.ndarray, J_face_amm2: np.ndarray | None,
                   tol: float = NEWTON_TOL, max_it: int = NEWTON_MAX, linear: bool = False,
                   load: np.ndarray | None = None) -> dict:
    """Damped Newton for the anisotropic nonlinear problem (energy ``Pi(A) = sum_f |f| W_f(B_f) - F.A``, exact
    symmetric tangent ``sum_f |f| Gb_f dH/dB Gb_f^T``, Armijo on the convex energy), from the linear solve with the
    low-field tensors.  ``linear=True``: that linear solve only.

    Returns:
        dict with ``A (n0,)`` Wb/m, ``Bf (n2, 2)`` T, ``it``, ``res`` (relative free residual), ``energy`` (J/m), the
        residual history and the number of line-search halvings.
    """
    Gb = curl_basis(m)
    Fv = m.load(J_face_amm2 * AMM2) if load is None else load
    fr = m.free
    nF = float(np.linalg.norm(Fv[fr])) or 1.0
    A = np.zeros(m.n0)

    def assemble(T):                                                   # sum_f |f| Gb T Gb^T
        return m.csr(np.einsum("fid,fde,fje->fij", Gb, T, Gb) * m.area[:, None, None])

    T0 = linear_tensor(phase, phi)
    K0 = assemble(T0)[fr][:, fr]
    A[fr] = spla.spsolve(K0.tocsc(), Fv[fr])
    if linear:
        Bf = np.einsum("fid,fi->fd", Gb, A[m.F])
        R = K0 @ A[fr] - Fv[fr]
        return dict(A=A, Bf=Bf, it=1, res=float(np.linalg.norm(R) / nF),
                    energy=float(0.5 * (m.area * np.einsum("fd,fde,fe->f", Bf, T0, Bf)).sum()), hist=[], halvings=0)

    def state(Avec, hessian=False):
        Bf = np.einsum("fid,fi->fd", Gb, Avec[m.F])
        L = law(Bf, phase, phi, hessian=hessian)
        e_mag, e_src = float((m.area * L["W"]).sum()), float(Fv[fr] @ Avec[fr])
        return e_mag - e_src, Bf, L, abs(e_mag) + abs(e_src)

    def residual(L):
        r = np.zeros(m.n0)
        np.add.at(r, m.F.ravel(), (np.einsum("fid,fd->fi", Gb, L["H"]) * m.area[:, None]).ravel())
        return (r - Fv)[fr]

    Pi, Bf, L, scale = state(A, hessian=True)
    hist, halvings, it, res = [], 0, 0, math.inf
    for it in range(1, max_it + 1):
        R = residual(L)
        res = float(np.linalg.norm(R) / nF)
        hist.append(res)
        if res <= tol:
            break
        Kt = assemble(L["dH"])[fr][:, fr]
        delta = spla.spsolve(Kt.tocsc(), -R)
        slope = float(R @ delta)
        alpha = 1.0
        while True:
            An = A.copy()
            An[fr] += alpha * delta
            Pn, Bn, Ln, sn = state(An)
            if Pn <= Pi + 1e-4 * alpha * slope + 1e-12 * max(scale, sn) or alpha < 1e-6:
                break
            alpha *= 0.5
            halvings += 1
        A, Pi, Bf, scale = An, Pn, Bn, sn
        L = law(Bf, phase, phi, hessian=True)
    else:
        res = float(np.linalg.norm(residual(L)) / nF)
    return dict(A=A, Bf=Bf, it=it, res=res, energy=float((m.area * L["W"]).sum()), hist=hist, halvings=halvings)


def picard_solve_a(m: P1Mesh, phase: np.ndarray, phi: np.ndarray, J_face_amm2: np.ndarray, omega: float, iters: int,
                   A_ref: np.ndarray | None = None) -> dict:
    """Secant iteration ``A_j = (1 - omega) A_{j-1} + omega K(nu(B_{j-1}))^{-1} F`` with the symmetric secant tensor
    (``H = nu(B) B``), from ``A_0 = 0``: an independent iteration for the same discrete problem (verification)."""
    Gb = curl_basis(m)
    Fv = m.load(J_face_amm2 * AMM2)
    fr = m.free
    A = np.zeros(m.n0)
    errs = []
    for _ in range(iters):
        Bf = np.einsum("fid,fi->fd", Gb, A[m.F])
        T = true_tensor(Bf, phase, phi)
        K = m.csr(np.einsum("fid,fde,fje->fij", Gb, T, Gb) * m.area[:, None, None])[fr][:, fr]
        An = np.zeros(m.n0)
        An[fr] = spla.spsolve(K.tocsc(), Fv[fr])
        A = (1.0 - omega) * A + omega * An
        if A_ref is not None:
            errs.append(float(np.linalg.norm(A - A_ref) / np.linalg.norm(A_ref)))
    return dict(A=A, errs=errs)


# ---------------------------------------------------------------------------------------------------------------------
# data sets
# ---------------------------------------------------------------------------------------------------------------------
_W: dict = {}


def _init_worker(mesh: dict) -> None:
    _W["mesh"] = mesh
    _W["m"] = P1Mesh(mesh["pos"], mesh["faces"], mesh["boundary"])


def _jsonable_layout(p: dict) -> dict:
    out = {k: v for k, v in p.items() if k not in ("pieces", "coils")}
    out["pieces"] = [dict(shape=s, material=PHASE_NAMES[c], phi=ph) for s, c, ph in p["pieces"]]
    out["coils"] = p["coils"]
    return out


def solve_layout(mesh: dict, m: P1Mesh, p: dict, factor: float = 1.0) -> dict:
    phase, phi, J = paint(mesh, p, factor)
    t0 = time.time()
    sol = newton_solve_a(m, phase, phi, J)
    Bn = node_field(m, sol["Bf"])
    Bmag = np.linalg.norm(sol["Bf"], axis=1)
    steel = phase >= PHASE_NO
    return dict(idx=p["idx"], family=FAMILY_ID[p["family"]], phase=phase, phi=phi.astype(np.float32),
                J_face=J.astype(np.float32), A=sol["A"], B_node=Bn.astype(np.float32), it=sol["it"], res=sol["res"],
                halvings=sol["halvings"], energy=sol["energy"], seconds=time.time() - t0,
                bmax_steel=float(Bmag[steel].max()) if steel.any() else 0.0, bmax=float(Bmag.max()),
                b_p99_steel=float(np.percentile(Bmag[steel], 99)) if steel.any() else 0.0,
                n_steel=int(steel.sum()), n_copper=int((phase == PHASE_COPPER).sum()), params=_jsonable_layout(p))


def _solve_job(args) -> dict:
    idx, factor, heldout = args
    return solve_layout(_W["mesh"], _W["m"], draw_layout(idx, heldout), factor)


def make_set(mesh: dict, ids: list[int], factor: float, heldout: bool, workers: int) -> list[dict]:
    with Pool(workers, initializer=_init_worker, initargs=(mesh,)) as pool:
        return pool.map(_solve_job, [(i, factor, heldout) for i in ids], chunksize=4)


def _law_meta() -> dict:
    return dict(law="W(B) = w_e(|B|) cos^2 theta + w_h(|B|) sin^2 theta (Froehlich-Kennelly energies, theta = angle "
                    "between B and the rolling direction)", Js=JS, mu0=MU0,
                materials={PHASE_NAMES[k]: dict(mu_easy=v[1], mu_hard=v[2], Hk_easy=fk_hk(v[1]), Hk_hard=fk_hk(v[2]))
                           for k, v in MATERIALS.items()},
                phase_names=list(PHASE_NAMES), units=dict(length="mm", B="T", J="A/mm^2", A="Wb/m (T m)", nu="m/H"))


def pack(mesh: dict, rows: list[dict], meta: dict) -> dict:
    import torch
    return dict(pos=torch.tensor(mesh["pos"]), faces=torch.tensor(mesh["faces"]),
                boundary=torch.tensor(mesh["boundary"]), lattice_face=torch.tensor(mesh["lattice_face"]),
                sample_ids=torch.tensor([r["idx"] for r in rows]), family=torch.tensor([r["family"] for r in rows]),
                phase=torch.tensor(np.stack([r["phase"] for r in rows])),
                phi=torch.tensor(np.stack([r["phi"] for r in rows])),
                J_face=torch.tensor(np.stack([r["J_face"] for r in rows])),
                A=torch.tensor(np.stack([r["A"] for r in rows])),
                B_node=torch.tensor(np.stack([r["B_node"] for r in rows])),
                newton_it=torch.tensor([r["it"] for r in rows]), newton_res=torch.tensor([r["res"] for r in rows]),
                energy=torch.tensor([r["energy"] for r in rows]),
                bmax_steel=torch.tensor([r["bmax_steel"] for r in rows]),
                b_p99_steel=torch.tensor([r["b_p99_steel"] for r in rows]),
                meta=dict(meta, params=[r["params"] for r in rows], mesh_stats=mesh["stats"], families=list(FAMILIES)))


def _summary(rows: list[dict], mesh: dict, extra: dict) -> dict:
    it = np.array([r["it"] for r in rows])
    res = np.array([r["res"] for r in rows])
    bp = np.array([r["b_p99_steel"] for r in rows])
    fam = np.array([r["family"] for r in rows])
    per_fam = {}
    for f, k in FAMILY_ID.items():
        sel = fam == k
        if sel.any():
            per_fam[f] = dict(n=int(sel.sum()), frac_bp99_gt_1p6=float((bp[sel] > 1.6).mean()),
                              frac_bp99_lt_1p2=float((bp[sel] < 1.2).mean()), bp99_median=float(np.median(bp[sel])))
    return dict(extra, N=len(rows), mesh=mesh["stats"], newton_it=dict(mean=float(it.mean()), max=int(it.max()),
                                                                       p50=float(np.median(it))),
                newton_res_max=float(res.max()), seconds_mean=float(np.mean([r["seconds"] for r in rows])),
                b_p99_steel=dict(p10=float(np.percentile(bp, 10)), p50=float(np.median(bp)),
                                 p90=float(np.percentile(bp, 90)), max=float(bp.max())),
                frac_bp99_gt_1p6=float((bp > 1.6).mean()), frac_bp99_gt_1p9=float((bp > 1.9).mean()),
                frac_bp99_lt_1p2=float((bp < 1.2).mean()), per_family=per_fam,
                halvings_total=int(sum(r["halvings"] for r in rows)),
                frac_oriented_steel=float(np.mean([np.isin(r["phase"], ORIENTED).sum() / max(r["n_steel"], 1)
                                                   for r in rows])))


def cmd_data(a) -> None:
    import torch
    out_dir = os.path.join(ROOT, "datasets", "v2")
    os.makedirs(out_dir, exist_ok=True)
    meshes = {}
    for name in ("base", "fine", "coarse"):
        t0 = time.time()
        meshes[name] = build_lattice_mesh(SCALES_A[name], MESH_SEEDS_A[name])
        print(f"mesh {name}: {meshes[name]['stats']} ({time.time() - t0:.1f}s)", flush=True)
    ext_ids = list(range(TEST_START, min(TEST_START + N_EXTRA, a.n))) if a.n > TEST_START else list(range(min(a.n, 50)))
    jobs = [("MXA", "base", list(range(a.n)), 1.0, False), ("MXA_ood15", "base", ext_ids, OOD_FACTOR, False),
            ("MXA_fine", "fine", ext_ids, 1.0, False), ("MXA_coarse", "coarse", ext_ids, 1.0, False),
            ("MXA_heldout", "base", list(range(N_EXTRA if a.n > TEST_START else min(a.n, 50))), 1.0, True)]
    for fname, mname, ids, factor, heldout in jobs:
        if a.only and fname not in a.only.split(","):
            continue
        t0 = time.time()
        rows = make_set(meshes[mname], ids, factor, heldout, a.workers)
        meta = dict(generator="datasets/generators/gen_maxwell_aniso.py", seed=SEED_A, mesh=mname,
                    scale=SCALES_A[mname], mesh_seed=MESH_SEEDS_A[mname], current_factor=factor, J_max=J_MAX_G,
                    J_max_EE=J_MAX_EE, J_load=J_LOAD, J_mag=[J_MAG0, J_MAG1], box=BOX_A, heldout=heldout,
                    families_present=sorted({FAMILIES[r["family"]] for r in rows}), **_law_meta(),
                    split="sequential 70/15/15 of the 5000 base samples; test100 = 4250..4349",
                    created=time.strftime("%Y-%m-%d %H:%M:%S"))
        torch.save(pack(meshes[mname], rows, meta), os.path.join(out_dir, fname + ".pt"))
        summ = _summary(rows, meshes[mname], dict(file=fname, factor=factor, mesh=mname, heldout=heldout,
                                                  seconds_total=time.time() - t0))
        with open(os.path.join(out_dir, fname + ".json"), "w") as f:
            json.dump(summ, f, indent=1)
        print(json.dumps(summ), flush=True)


def cmd_pilot(a) -> None:
    mesh = build_lattice_mesh(1.0, MESH_SEEDS_A["base"])
    print("mesh", json.dumps(mesh["stats"]), flush=True)
    t0 = time.time()
    rows = make_set(mesh, list(range(a.n)), 1.0, False, a.workers)
    rows += make_set(mesh, list(range(a.n // 3)), 1.0, True, a.workers)
    s = _summary(rows, mesh, dict(pilot=True, seconds_total=time.time() - t0))
    print(json.dumps(s, indent=1), flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------------------------------------------------
def law_checks(seed: int = 0) -> dict:
    """Gradient / Hessian consistency, convexity, the two principal curves, the linear limit, the secant tensor and
    the rotation covariance of the law."""
    rng = np.random.default_rng(seed)
    out = {}
    n = 4000
    for cls, (name, mue, muh) in MATERIALS.items():
        b = np.concatenate([10 ** rng.uniform(-6, 0.7, n // 2), rng.uniform(1.5, 2.6, n // 2)])
        th = rng.uniform(0, 2 * math.pi, n)
        B = np.stack([b * np.cos(th), b * np.sin(th)], 1)
        phase = np.full(n, cls, dtype=np.int8)
        phi = rng.uniform(0, math.pi, n)
        L = law(B, phase, phi, hessian=True, secant=True)
        d = 1e-7 * np.maximum(b, 1e-3)[:, None]
        Hfd = np.zeros_like(B)
        dHfd = np.zeros((n, 2, 2))
        for k in range(2):
            e = np.zeros_like(B)
            e[:, k] = d[:, 0]
            Wp, Wm = law(B + e, phase, phi, hessian=False)["W"], law(B - e, phase, phi, hessian=False)["W"]
            Hfd[:, k] = (Wp - Wm) / (2 * d[:, 0])
            Hp, Hm = law(B + e, phase, phi, hessian=False)["H"], law(B - e, phase, phi, hessian=False)["H"]
            dHfd[:, :, k] = (Hp - Hm) / (2 * d)
        Hn = np.linalg.norm(L["H"], axis=1)
        nmin = np.minimum(fk_nu(b, fk_hk(mue)), fk_nu(b, fk_hk(muh)))
        ev = np.linalg.eigvalsh(L["dH"])
        # rotation covariance: H(R B; phi + a) = R H(B; phi)
        a_ = rng.uniform(0, 2 * math.pi, n)
        Rm = np.stack([np.stack([np.cos(a_), -np.sin(a_)], -1), np.stack([np.sin(a_), np.cos(a_)], -1)], -2)
        Lr = law(np.einsum("nij,nj->ni", Rm, B), phase, (phi + a_) % math.pi, hessian=False)
        # principal curves
        bb = np.linspace(1e-3, 3.0, 300)
        ph0 = np.zeros(300)
        He = law(np.stack([bb, 0 * bb], 1), np.full(300, cls, np.int8), ph0, hessian=False)["H"][:, 0]
        Hh = law(np.stack([0 * bb, bb], 1), np.full(300, cls, np.int8), ph0, hessian=False)["H"][:, 1]
        out[name] = dict(
            H_vs_fd_energy_max_rel=float(np.max(np.linalg.norm(L["H"] - Hfd, axis=1) / Hn)),
            dH_vs_fd_H_max_rel=float(np.max(np.abs(L["dH"] - dHfd).max((1, 2)) / nmin)),
            dH_symmetric_max=float(np.abs(L["dH"] - L["dH"].transpose(0, 2, 1)).max() / NU0),
            min_eig_dH_over_min_nu=float((ev[:, 0] / nmin).min()),
            secant_reproduces_H_max_rel=float(np.max(np.linalg.norm(np.einsum("nij,nj->ni", L["nu"], B) - L["H"], axis=1)
                                                     / Hn)),
            rotation_covariance_max_rel=float(np.max(np.linalg.norm(Lr["H"] - np.einsum("nij,nj->ni", Rm, L["H"]), axis=1)
                                                     / Hn)),
            easy_curve_max_rel=float(np.max(np.abs(He - fk_nu(bb, fk_hk(mue)) * bb) / (fk_nu(bb, fk_hk(mue)) * bb))),
            hard_curve_max_rel=float(np.max(np.abs(Hh - fk_nu(bb, fk_hk(muh)) * bb) / (fk_nu(bb, fk_hk(muh)) * bb))),
            ratio_nu_hard_over_easy={str(x): float(fk_nu(x, fk_hk(muh)) / fk_nu(x, fk_hk(mue)))
                                     for x in (0.01, 1.0, 1.5, 1.8, 2.0, 2.2, 2.5)},
            B_easy_at_H={str(h): float(MU0 * h + JS * h / (fk_hk(mue) + h)) for h in (100, 1000, 10000, 100000)},
            B_hard_at_H={str(h): float(MU0 * h + JS * h / (fk_hk(muh) + h)) for h in (100, 1000, 10000, 100000)})
    # convexity on a dense (|B|, angle) grid
    bb = np.concatenate([np.linspace(1e-3, 3.0, 1500), np.linspace(3.0, 10.0, 100)])
    th = np.linspace(0, math.pi / 2, 91)
    Bg, Tg = np.meshgrid(bb, th, indexing="ij")
    B = np.stack([(Bg * np.cos(Tg)).ravel(), (Bg * np.sin(Tg)).ravel()], 1)
    for cls, (name, mue, muh) in MATERIALS.items():
        L = law(B, np.full(len(B), cls, np.int8), np.zeros(len(B)), hessian=True)
        b = np.linalg.norm(B, axis=1)
        out[name]["grid_min_eig_dH_over_min_nu"] = float((np.linalg.eigvalsh(L["dH"])[:, 0] /
                                                          np.minimum(fk_nu(b, fk_hk(mue)), fk_nu(b, fk_hk(muh)))).min())
    # the two-curve secant law nu = R diag(nu_e(|B|), nu_h(|B|)) R^T for comparison (not used): symmetric part of its
    # incremental reluctivity
    worst = {}
    for cls, (name, mue, muh) in MATERIALS.items():
        if mue == muh:
            continue
        hke, hkh = fk_hk(mue), fk_hk(muh)
        b = np.linalg.norm(B, axis=1)
        u = B / b[:, None]
        ne, nh, de, dh = fk_nu(b, hke), fk_nu(b, hkh), fk_dnu(b, hke), fk_dnu(b, hkh)
        K = np.zeros((len(b), 2, 2))
        Kp = np.zeros((len(b), 2, 2))
        K[:, 0, 0], K[:, 1, 1], Kp[:, 0, 0], Kp[:, 1, 1] = ne, nh, de, dh
        T = K + b[:, None, None] * np.einsum("nij,nj,nk->nik", Kp, u, u)
        S = 0.5 * (T + T.transpose(0, 2, 1))
        worst[name] = float((np.linalg.eigvalsh(S)[:, 0] / np.minimum(ne, nh)).min())
    out["two_curve_secant_law_min_eig_sym_over_min_nu"] = worst
    return out


def manufactured_convergence(ns=(8, 16, 32, 64, 128), L_mm: float = 200.0, B_max: float = 2.4,
                             phi: float = math.radians(30.0)) -> dict:
    """P1 + Newton convergence for ``A = A0 sin(pi x / L) sin(pi y / L)`` in GO steel everywhere (rolling direction
    ``phi``, ``|B|`` up to ``B_max``), ``J = curl H(curl A)`` by central differences, degree-5 quadrature load."""
    L = L_mm * MM
    A0 = B_max * L / math.pi
    k = math.pi / L

    def A_ex(x, y):
        return A0 * np.sin(k * x) * np.sin(k * y)

    def B_ex(x, y):
        return np.stack([A0 * k * np.sin(k * x) * np.cos(k * y), -A0 * k * np.cos(k * x) * np.sin(k * y)], -1)

    def H_ex(x, y):
        B = B_ex(x, y)
        sh = B.shape
        Bf = B.reshape(-1, 2)
        return law(Bf, np.full(len(Bf), PHASE_GO, np.int8), np.full(len(Bf), phi), hessian=False)["H"].reshape(sh)

    def J_ex(x, y, d=1e-7):
        return ((H_ex(x + d, y)[..., 1] - H_ex(x - d, y)[..., 1]) - (H_ex(x, y + d)[..., 0] -
                                                                     H_ex(x, y - d)[..., 0])) / (2 * d)

    lam, w = _TRI7
    rows = []
    for n in ns:
        pos, F, bnd = _structured_square(n, L_mm)
        m = P1Mesh(pos, F, bnd)
        Xq = np.einsum("qk,fkd->fqd", lam, m.X[F])
        Jq = J_ex(Xq[..., 0], Xq[..., 1])
        Fv = np.zeros(m.n0)
        for kk in range(3):
            np.add.at(Fv, F[:, kk], (m.area[:, None] * w[None] * Jq * lam[None, :, kk]).sum(1))
        sol = newton_solve_a(m, np.full(len(F), PHASE_GO, dtype=np.int8), np.full(len(F), phi), None, load=Fv, tol=1e-12)
        Ah = np.einsum("qk,fk->fq", lam, sol["A"][F])
        Aq = A_ex(Xq[..., 0], Xq[..., 1])
        eL2 = math.sqrt(float((m.area[:, None] * w[None] * (Ah - Aq) ** 2).sum()))
        nL2 = math.sqrt(float((m.area[:, None] * w[None] * Aq ** 2).sum()))
        Bq = B_ex(Xq[..., 0], Xq[..., 1])
        eB = math.sqrt(float((m.area[:, None] * w[None] * ((sol["Bf"][:, None, :] - Bq) ** 2).sum(-1)).sum()))
        nB = math.sqrt(float((m.area[:, None] * w[None] * (Bq ** 2).sum(-1)).sum()))
        rows.append(dict(n=n, h_mm=L_mm / n, n0=m.n0, newton_it=sol["it"], newton_res=sol["res"],
                         halvings=sol["halvings"], rel_L2_A=eL2 / nL2, rel_L2_B=eB / nB))
    for a_, b_ in zip(rows[:-1], rows[1:]):
        b_["rate_L2_A"] = math.log(a_["rel_L2_A"] / b_["rel_L2_A"]) / math.log(2.0)
        b_["rate_L2_B"] = math.log(a_["rel_L2_B"] / b_["rel_L2_B"]) / math.log(2.0)
    return dict(problem=f"A = A0 sin(pi x/L) sin(pi y/L), L = {L_mm} mm, |B| <= {B_max} T, GO steel everywhere, "
                        f"rolling direction {math.degrees(phi):.0f} deg", rows=rows)


def _energy(m: P1Mesh, phase, phi, Fv, A) -> float:
    Bf = np.einsum("fid,fi->fd", curl_basis(m), A[m.F])
    return float((m.area * law(Bf, phase, phi, hessian=False)["W"]).sum() - Fv[m.free] @ A[m.free])


def _refinement_job(args):
    scale, idx, heldout = args
    mesh = build_lattice_mesh(scale, 300 + int(scale * 100))
    m = P1Mesh(mesh["pos"], mesh["faces"], mesh["boundary"])
    p = draw_layout(idx, heldout)
    phase, phi, J = paint(mesh, p)
    sol = newton_solve_a(m, phase, phi, J)
    steel = phase >= PHASE_NO
    return dict(scale=scale, idx=idx, family=p["family"], n0=m.n0, energy=sol["energy"], newton_it=sol["it"],
                res=sol["res"], steel_area_mm2=float(m.area[steel].sum() / MM ** 2),
                b_p99_steel=float(np.percentile(np.linalg.norm(sol["Bf"][steel], axis=1), 99)))


def cmd_selftest(a) -> None:
    t0 = time.time()
    out = dict(law=law_checks())
    print("law", json.dumps(out["law"]), flush=True)
    out["manufactured"] = manufactured_convergence()
    for r in out["manufactured"]["rows"]:
        print("manufactured", json.dumps(r), flush=True)
    mesh = build_lattice_mesh(1.0, MESH_SEEDS_A["base"])
    m = P1Mesh(mesh["pos"], mesh["faces"], mesh["boundary"])
    rng = np.random.default_rng(1)
    rows = []
    picked = {}
    for idx in range(200):
        f = draw_layout(idx)["family"]
        picked.setdefault(f, []).append(idx)
    cases = [(i, False) for f in TRAIN_FAMILIES for i in picked[f][:2]] + [(0, True), (1, True)]
    for idx, ho in cases:
        p = draw_layout(idx, ho)
        phase, phi, J = paint(mesh, p)
        eps = 1e-4
        s_small = newton_solve_a(m, phase, phi, J * eps)
        s_lin = newton_solve_a(m, phase, phi, J, linear=True)
        sol = newton_solve_a(m, phase, phi, J)
        Fv = m.load(J * AMM2)
        E0 = _energy(m, phase, phi, Fv, sol["A"])
        worst = math.inf
        for _ in range(10):
            dA = np.zeros(m.n0)
            dA[m.free] = rng.normal(size=len(m.free))
            for rel in (1e-4, 1e-2, 1e-1):
                worst = min(worst, _energy(m, phase, phi, Fv, sol["A"] + dA * rel * np.abs(sol["A"]).max() /
                                           np.abs(dA).max()) - E0)
        # rotation: the same layout on the rotated mesh with rotated rolling directions gives the same potential
        th = math.radians(37.0)
        Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        mr = P1Mesh(mesh["pos"] @ Rm.T, mesh["faces"], mesh["boundary"])
        sol_r = newton_solve_a(mr, phase, np.where(np.isin(phase, ORIENTED), (phi + th) % math.pi, 0.0), J)
        steel = phase >= PHASE_NO
        row = dict(idx=idx, family=p["family"], newton_it=sol["it"], res=sol["res"], halvings=sol["halvings"],
                   b_p99_steel=float(np.percentile(np.linalg.norm(sol["Bf"][steel], axis=1), 99)),
                   linear_limit_rel_diff=float(np.linalg.norm(s_small["A"] / eps - s_lin["A"]) /
                                               np.linalg.norm(s_lin["A"])),
                   min_energy_increase=worst, energy=E0,
                   rotated_mesh_rel_diff_A=float(np.linalg.norm(sol_r["A"] - sol["A"]) / np.linalg.norm(sol["A"])),
                   nonlinear_vs_linear_rel_diff=float(np.linalg.norm(sol["A"] - s_lin["A"]) / np.linalg.norm(sol["A"])))
        lin_iso = newton_solve_a(m, np.where(phase >= PHASE_NO, PHASE_NO, phase).astype(np.int8), phi, J)
        row["isotropic_NO_vs_true_rel_diff_B"] = float(np.linalg.norm(lin_iso["Bf"] - sol["Bf"]) /
                                                       np.linalg.norm(sol["Bf"]))
        flip = newton_solve_a(m, phase, np.where(np.isin(phase, ORIENTED), (phi + 0.5 * math.pi) % math.pi, 0.0), J)
        row["rolling_directions_turned_90deg_rel_diff_B_node"] = float(
            np.linalg.norm(node_field(m, flip["Bf"]) - node_field(m, sol["Bf"])) / np.linalg.norm(node_field(m, sol["Bf"])))
        for om in (0.5, 0.25):
            r = picard_solve_a(m, phase, phi, J, om, 48, A_ref=sol["A"])
            row[f"picard_omega_{om}"] = {str(k): r["errs"][k - 1] for k in (1, 2, 4, 8, 16, 32, 48)}
        rows.append(row)
        print("sample", json.dumps(row), flush=True)
    out["samples"] = rows
    jobs = [(sc, idx, ho) for sc in (2.0, 1.0, 0.5, 0.25) for idx, ho in cases[::2]]
    with Pool(min(a.workers, len(jobs))) as pool:
        out["refinement"] = pool.map(_refinement_job, jobs)
    for r in out["refinement"]:
        print("refine", json.dumps(r), flush=True)
    out["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"selftest written to {a.out} ({out['seconds']:.0f}s)", flush=True)


def cmd_mesh(a) -> None:
    for name in ("base", "fine", "coarse"):
        mesh = build_lattice_mesh(SCALES_A[name], MESH_SEEDS_A[name])
        print(name, json.dumps(mesh["stats"]), flush=True)
        if a.plot and name == "base":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
            cmap = ListedColormap(["white", "#e8a33d", "#9aa5b1", "#3c78d8", "#6aa84f"])
            ids = [0, 1, 2, 3, 5, 8]
            fig, ax = plt.subplots(2, 4, figsize=(24, 11))
            for k, axi in enumerate(ax.ravel()):
                ho = k >= 6
                p = draw_layout(ids[k] if not ho else k - 6, heldout=ho)
                ph, phi, J = paint(mesh, p)
                axi.tripcolor(mesh["pos"][:, 0], mesh["pos"][:, 1], mesh["faces"], facecolors=ph.astype(float),
                              cmap=cmap, vmin=-0.5, vmax=4.5, edgecolors="k", linewidth=0.1)
                cent = mesh["pos"][mesh["faces"]].mean(1)
                o = np.isin(ph, ORIENTED) & (np.arange(len(ph)) % 3 == 0)
                axi.quiver(cent[o, 0], cent[o, 1], np.cos(phi[o]), np.sin(phi[o]), pivot="mid", headwidth=0,
                           headlength=0, headaxislength=0, scale=60, width=0.002)
                axi.set_xlim(-175, 175)
                axi.set_ylim(-125, 128)
                axi.set_aspect("equal")
                axi.set_title(f"{p['family']} #{p['idx']} {p.get('construction')}")
            fig.savefig(os.path.join(ROOT, "runs", "maxwell", "mxa_layouts.png"), dpi=110, bbox_inches="tight")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--n", type=int, default=N_SAMPLES)
    d.add_argument("--workers", type=int, default=32)
    d.add_argument("--only", default=None, help="comma-separated file names (MXA, MXA_ood15, MXA_fine, MXA_coarse, "
                                                "MXA_heldout)")
    m = sub.add_parser("mesh")
    m.add_argument("--plot", action="store_true")
    s = sub.add_parser("selftest")
    s.add_argument("--out", default=os.path.join(ROOT, "runs", "maxwell", "selftest_mxa.json"))
    s.add_argument("--workers", type=int, default=16)
    q = sub.add_parser("pilot")
    q.add_argument("--n", type=int, default=300)
    q.add_argument("--workers", type=int, default=32)
    a = p.parse_args(argv)
    os.makedirs(os.path.join(ROOT, "runs", "maxwell"), exist_ok=True)
    {"data": cmd_data, "mesh": cmd_mesh, "selftest": cmd_selftest, "pilot": cmd_pilot}[a.cmd](a)


if __name__ == "__main__":
    main()
