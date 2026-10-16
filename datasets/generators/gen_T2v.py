"""Reproduce the T2 torus advection-diffusion generator with its per-sample velocity (task T2v) and build the extra
evaluation sets.

The paper's T2 pickle (``datasets/T2_torus_advection_diffusion.pkl``) is the output of the paper's torus generator
(``generate_torus_fast.py``) run with 3000 samples, ``n_steps = 10``,
``diffusivity = 0.03``, ``dt = 0.02`` (the ``physics`` entry of the pickle), with the velocity removed.  The generator
draws a different velocity for every sample, ``v = alpha vt + beta vp`` with ``alpha ~ U(0.3, 1.5)``,
``beta ~ U(0, 0.8)``; its own output file (``torus_advdiff_dataset.pkl``) keeps it as ``V_data (N, n0, 3)``.

The scheme of the generator, per time step (float64, the stored mesh)::

    u <- (I - dt D L)^{-1} ( u - dt v . g(u) )

* ``L``: cotangent Laplacian ``sum_j w_ij (u_j - u_i)``, ``w_ij = (cot a + cot b) / 2`` (the stiffness matrix; no mass
  matrix, so the effective diffusivity at vertex i is ``D A_i`` with the dual area ``A_i``);
* ``g(u)_i``: unweighted mean over the faces incident to i of the P1 gradient of u on the face (vertex-averaged
  gradient); the advection term is the explicit, centred, advective (non-conservative) ``v_i . g_i``;
* ``vt``: unit toroidal field projected on the vertex tangent plane; ``vp``: the field
  ``(-sin p cos t, -sin p sin t, cos p)`` with ``p = arctan2(rho - 1, z)``, projected on the tangent plane and
  normalised (``rho - 1`` refers to the torus before the rescaling by its maximal radius 1.4).

This script reproduces the generator from the stored mesh (same numpy RNG call sequence and arithmetic, vectorised)
and

1. checks the reproduction against the stored arrays (``X``: initial conditions, ``V``: velocities, ``Y``: targets)
   and evaluates the descriptions of the scheme found in the paper (15 steps with D = 0.01; a lumped-mass system)
   against the stored targets;
2. records physical diagnostics: effective diffusivity, CFL number, divergence of the velocity fields, the sign
   pattern of ``vp`` relative to the poloidal direction, drift of ``int u dA`` and of ``sum_i u_i``;
3. writes ``datasets/v2/T2v.pt`` (+ ``T2v.json``) with

   * ``V (N, n0, 3)`` float32: the generator's velocity (copied from its own output file when available, else the
     regenerated one), ``alpha, beta (N,)``, the basis fields ``vt, vp (n0, 3)``;
   * ``alpha_sweep``: the 450 test samples (initial condition and ``beta`` unchanged) re-simulated with
     ``alpha`` in ``ALPHAS`` (training range 0.3..1.5): ``V (A, 450, n0, 3)``, ``Y (A, 450, n0)``;
   * ``rollout``: the 10-step (= stored target) and 20-step states of the 450 test samples, ``Y10``, ``Y20``;
   * ``meta``: the checks and diagnostics above.

Usage (CPU, about a minute):

    python3 -u datasets/generators/gen_T2v.py [--source PATH_TO_torus_advdiff_dataset.pkl] [--out datasets/v2/T2v.pt]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
from scipy.sparse import csr_matrix, diags, eye as speye
from scipy.sparse.linalg import factorized

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_SOURCE = os.path.join(ROOT, "datasets", "torus_advdiff_dataset.pkl")   # optional (see --source)
PHYSICS = dict(diffusivity=0.03, dt=0.02, n_steps=10)        # the stored pickle's 'physics' entry
ALPHAS = (1.5, 1.75, 2.0, 2.25, 2.5)                         # velocity-extrapolation sweep (training: 0.3..1.5)
SEED = 42


# ======================================================================================================================
# generator operators (generate_torus_fast.py, vectorised with the same per-element arithmetic)
# ======================================================================================================================
def cotan_laplacian(points: np.ndarray, faces: np.ndarray) -> csr_matrix:
    """``build_cotangent_laplacian`` of the generator: off-diagonal ``+w``, diagonal ``-sum w`` (float32 values,
    cosines clipped to +-0.999)."""
    n = len(points)
    i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
    vi, vj, vk = points[i], points[j], points[k]
    e_ij, e_jk, e_ki = vj - vi, vk - vj, vi - vk
    area = np.linalg.norm(np.cross(e_ij, -e_ki), axis=1) / 2.0
    ok = area >= 1e-12

    def cot(a, b):
        c = np.clip((a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) + 1e-12), -0.999, 0.999)
        return c / (np.sqrt(1 - c ** 2) + 1e-12)
    rows, cols, vals = [], [], []
    for a, b, cv in ((j, k, cot(e_ij, -e_ki)), (k, i, cot(e_jk, -e_ij)), (i, j, cot(e_ki, -e_jk))):
        a, b, w = a[ok], b[ok], (cv / 2.0)[ok]
        rows += [a, b, a, b]
        cols += [b, a, a, b]
        vals += [w, w, -w, -w]
    return csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))


def generator_operators(points: np.ndarray, faces: np.ndarray) -> dict:
    """Normals, Laplacian, face-gradient helpers and the two velocity basis fields of the generator."""
    n = len(points)
    fi, fj, fk = faces[:, 0], faces[:, 1], faces[:, 2]
    normals = np.zeros((n, 3))
    nm = np.cross(points[fj] - points[fi], points[fk] - points[fi])
    for v in (fi, fj, fk):
        np.add.at(normals, v, nm)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
    L = cotan_laplacian(points, faces)
    e1 = points[fj] - points[fi]
    e2 = points[fk] - points[fi]
    fn = np.cross(e1, e2)
    fa2 = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
    fn /= fa2
    cr_kj = np.cross(fn, points[fk] - points[fj])
    cr_ik = np.cross(fn, points[fi] - points[fk])
    cr_ji = np.cross(fn, points[fj] - points[fi])
    nc = np.zeros(n)
    np.add.at(nc, fi, 1)
    np.add.at(nc, fj, 1)
    np.add.at(nc, fk, 1)
    nc[nc == 0] = 1.0
    theta = np.arctan2(points[:, 1], points[:, 0])
    vt = np.stack([-np.sin(theta), np.cos(theta), np.zeros(n)], axis=1)          # float64 (as the generator)
    vt -= np.sum(vt * normals, 1, keepdims=True) * normals
    vt /= (np.linalg.norm(vt, axis=1, keepdims=True) + 1e-9)
    rxy = np.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    phi = np.arctan2(rxy - 1.0, points[:, 2])
    vp = np.stack([-np.sin(phi) * np.cos(theta), -np.sin(phi) * np.sin(theta), np.cos(phi)], axis=1)
    vp -= np.sum(vp * normals, 1, keepdims=True) * normals
    vp /= (np.linalg.norm(vp, axis=1, keepdims=True) + 1e-9)
    return dict(n=n, faces=faces, fi=fi, fj=fj, fk=fk, normals=normals, L=L, fa2=fa2, cr_kj=cr_kj, cr_ik=cr_ik,
                cr_ji=cr_ji, nc=nc, vt=vt, vp=vp)


def vertex_gradient(ops: dict, u: np.ndarray) -> np.ndarray:
    """Unweighted mean over incident faces of the P1 face gradient, ``(n,) -> (n, 3)`` (the generator's ``g``)."""
    fi, fj, fk = ops["fi"], ops["fj"], ops["fk"]
    gt = (u[fi][:, None] * ops["cr_kj"] + u[fj][:, None] * ops["cr_ik"] + u[fk][:, None] * ops["cr_ji"]) / ops["fa2"]
    g = np.zeros((ops["n"], 3))
    for d in range(3):
        np.add.at(g[:, d], fi, gt[:, d])
        np.add.at(g[:, d], fj, gt[:, d])
        np.add.at(g[:, d], fk, gt[:, d])
    return g / ops["nc"][:, None]


def draw_samples(points: np.ndarray, n_samples: int, seed: int = SEED) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Initial conditions and velocity coefficients in the generator's RNG call order.

    Returns:
        ``X (N, n)`` float64 (before the float32 storage), ``alpha (N,)``, ``beta (N,)``.
    """
    n = len(points)
    rng = np.random.RandomState(seed)
    X = np.zeros((n_samples, n))
    alpha = np.zeros(n_samples)
    beta = np.zeros(n_samples)
    for s in range(n_samples):
        nb = rng.randint(2, 6)
        u0 = np.zeros(n)
        for _ in range(nb):
            c = points[rng.randint(0, n)]
            sig = rng.uniform(0.1, 0.3)
            st = rng.uniform(-2.0, 2.0)
            u0 += st * np.exp(-np.linalg.norm(points - c, axis=1) ** 2 / (2 * sig ** 2))
        u0 -= u0.mean()
        X[s] = u0
        alpha[s], beta[s] = rng.uniform(0.3, 1.5), rng.uniform(0.0, 0.8)
    return X, alpha, beta


def velocity(ops: dict, a: float, b: float) -> np.ndarray:
    """``a vt + b vp`` with the generator's dtypes (``vt`` float64, ``vp`` float32, Python scalars)."""
    return float(a) * ops["vt"] + float(b) * ops["vp"]


def make_solver(ops: dict, D: float, dt: float, mass: np.ndarray | None = None):
    """``u -> (I - dt D L)^{-1} u`` (generator), or ``u -> (M - dt D L)^{-1} M u`` with a lumped mass ``M``."""
    n = ops["n"]
    if mass is None:
        return factorized((speye(n) - dt * D * ops["L"]).tocsc())
    M = diags(mass)
    solve = factorized((M - dt * D * ops["L"]).tocsc())
    return lambda r: solve(mass * r)


def simulate(ops: dict, u0: np.ndarray, vf: np.ndarray, n_steps: int, solve, dt: float,
             keep: tuple[int, ...] = ()) -> tuple[np.ndarray, dict]:
    """``n_steps`` IMEX steps of the generator from ``u0 (n,)`` with the velocity ``vf (n, 3)``; ``keep``: steps whose
    states are returned as well."""
    u = np.asarray(u0, dtype=np.float64)
    states = {}
    for t in range(1, n_steps + 1):
        g = vertex_gradient(ops, u)
        u = solve(u - dt * np.sum(vf * g, axis=1))
        if t in keep:
            states[t] = u.copy()
    return u, states


# ======================================================================================================================
# diagnostics
# ======================================================================================================================
def dual_areas(points: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Barycentric dual areas (1/3 of the incident triangle areas), float64."""
    P = points.astype(np.float64)
    a = 0.5 * np.linalg.norm(np.cross(P[faces[:, 1]] - P[faces[:, 0]], P[faces[:, 2]] - P[faces[:, 0]]), axis=1)
    A = np.zeros(len(P))
    for v in range(3):
        np.add.at(A, faces[:, v], a / 3.0)
    return A


def edges_of(faces: np.ndarray, n: int) -> np.ndarray:
    """Unique ``src < dst`` edges in lexicographic order."""
    e = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    e = np.sort(e, axis=1)
    key = np.unique(e[:, 0].astype(np.int64) * n + e[:, 1])
    return np.stack([key // n, key % n], 1)


def cotan_weights(points: np.ndarray, faces: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """``(cot a + cot b) / 2`` per edge (float64, unclipped)."""
    P = points.astype(np.float64)
    n = len(P)
    key = edges[:, 0] * n + edges[:, 1]
    out = np.zeros(len(edges))
    for a, b, c in ((0, 1, 2), (1, 2, 0), (2, 0, 1)):
        ia, ib, ic = faces[:, a], faces[:, b], faces[:, c]
        u, v = P[ia] - P[ic], P[ib] - P[ic]
        cot = (u * v).sum(1) / np.linalg.norm(np.cross(u, v), axis=1)
        pos = np.searchsorted(key, np.minimum(ia, ib) * n + np.maximum(ia, ib))
        np.add.at(out, pos, 0.5 * cot)
    return out


def divergence(points: np.ndarray, faces: np.ndarray, V: np.ndarray, A: np.ndarray) -> np.ndarray:
    """DEC divergence of a vertex vector field: ``-(d0^T star1 theta)_i / A_i``, ``theta`` the trapezoidal one-form."""
    P = points.astype(np.float64)
    n = len(P)
    e = edges_of(faces, n)
    th = 0.5 * ((V[e[:, 0]] + V[e[:, 1]]) * (P[e[:, 1]] - P[e[:, 0]])).sum(1)
    F = cotan_weights(points, faces, e) * th
    acc = np.zeros(n)
    np.add.at(acc, e[:, 1], F)          # (d0^T F)_i = inflow - outflow
    np.add.at(acc, e[:, 0], -F)
    return -acc / A


def scheme_diagnostics(points: np.ndarray, faces: np.ndarray, ops: dict, X: np.ndarray, Y: np.ndarray,
                       alpha: np.ndarray, beta: np.ndarray) -> dict:
    """Physical diagnostics of the stored data and of the generator's scheme."""
    P = points.astype(np.float64)
    n = len(P)
    A = dual_areas(points, faces)
    e = edges_of(faces, n)
    elen = np.linalg.norm(P[e[:, 1]] - P[e[:, 0]], axis=1)
    D, dt, T = PHYSICS["diffusivity"], PHYSICS["dt"], PHYSICS["n_steps"] * PHYSICS["dt"]
    vmax = np.sqrt(alpha ** 2 + beta ** 2).max()
    # poloidal direction of the rescaled torus (tube centre at rho = R / (R + r) = 1 / 1.4)
    rho = np.sqrt(P[:, 0] ** 2 + P[:, 1] ** 2)
    R_s = float(np.median(rho[np.abs(P[:, 2]) > 0.8 * np.abs(P[:, 2]).max()]))      # top/bottom circle radius
    ph = np.arctan2(P[:, 2], rho - R_s)
    er = np.stack([P[:, 0] / rho, P[:, 1] / rho, np.zeros(n)], 1)
    ez = np.array([0.0, 0.0, 1.0])
    e_phi = -np.sin(ph)[:, None] * er + np.cos(ph)[:, None] * ez[None]
    e_th = np.stack([-P[:, 1] / rho, P[:, 0] / rho, np.zeros(n)], 1)
    vp = ops["vp"].astype(np.float64)
    vt = ops["vt"].astype(np.float64)
    s = np.sign((vp * e_phi).sum(1))
    div_t = divergence(points, faces, vt, A)
    div_p = divergence(points, faces, vp, A)
    m_x = X @ A
    m_y = Y @ A
    l1 = np.abs(X) @ A
    return dict(
        n0=n, n1=len(e), n2=len(faces), structured_grid="59 x 29 (u_res=60, v_res=30 parametric torus, cleaned)",
        dual_area=dict(min=A.min(), median=float(np.median(A)), max=A.max(), total=A.sum()),
        edge_length=dict(min=elen.min(), median=float(np.median(elen)), max=elen.max()),
        tube_centre_radius_after_rescaling=R_s,
        effective_diffusivity_D_times_A=dict(min=D * A.min(), median=float(D * np.median(A)), max=D * A.max(),
                                             diffusion_length_sqrt_2DAT=float(np.sqrt(2 * D * np.median(A) * T))),
        nominal_diffusion_length_sqrt_2DT=float(np.sqrt(2 * D * T)),
        cfl_max_speed_dt_over_min_edge=float(vmax * dt / elen.min()),
        cfl_max_speed_dt_over_median_edge=float(vmax * dt / np.median(elen)),
        advection_distance_over_median_edge=float(vmax * T / np.median(elen)),
        vt_dot_e_theta=dict(min=float(np.abs((vt * e_th).sum(1)).min()), mean=float(np.abs((vt * e_th).sum(1)).mean())),
        vp_dot_e_phi_abs=dict(min=float(np.abs((vp * e_phi).sum(1)).min()), mean=float(np.abs((vp * e_phi).sum(1)).mean())),
        vp_sign_vs_poloidal=dict(plus=int((s > 0).sum()), minus=int((s < 0).sum()),
                                 note="vp = sign * e_phi: the 'poloidal' field reverses direction across the "
                                      "circles where vp is normal to the surface before the projection"),
        divergence_rms_vt=float(np.sqrt((A * div_t ** 2).sum() / A.sum())),
        divergence_rms_vp=float(np.sqrt((A * div_p ** 2).sum() / A.sum())),
        divergence_scale_1_over_median_edge=float(1.0 / np.median(elen)),
        mass_drift_area_weighted=dict(
            mean_abs_rel=float(np.mean(np.abs(m_y - m_x) / l1)), max_abs_rel=float(np.max(np.abs(m_y - m_x) / l1)),
            definition="|sum_i A_i (y_i - f_i)| / sum_i A_i |f_i|"),
        sum_drift_unweighted=dict(
            mean_abs_rel=float(np.mean(np.abs(Y.sum(1) - X.sum(1)) / np.abs(X).sum(1))),
            definition="|sum_i (y_i - f_i)| / sum_i |f_i| (the implicit step (I - dt D L) conserves sum_i u_i)"),
        amplitude_ratio_l2=dict(mean=float(np.mean(np.linalg.norm(Y, axis=1) / np.linalg.norm(X, axis=1))),
                                max=float(np.max(np.linalg.norm(Y, axis=1) / np.linalg.norm(X, axis=1)))),
    )


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300))


# ======================================================================================================================
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--t2", default=os.path.join(ROOT, "datasets", "T2_torus_advection_diffusion.pkl"))
    p.add_argument("--source", default=DEFAULT_SOURCE, help="the generator's own output file (with V_data)")
    p.add_argument("--out", default=os.path.join(ROOT, "datasets", "v2", "T2v.pt"))
    p.add_argument("--check-samples", type=int, default=3000, help="samples re-simulated for the target check")
    p.add_argument("--meta-only", default=None, metavar="JSON",
                   help="write the checks and diagnostics to this file and do not write --out")
    a = p.parse_args()
    t0 = time.time()
    d = pickle.load(open(a.t2, "rb"))
    points = np.asarray(d["points"])                 # float32, as the generator used them
    faces = np.asarray(d["faces"], dtype=np.int64)
    Xs = np.asarray(d["X_data"], dtype=np.float32)
    Ys = np.asarray(d["Y_data"], dtype=np.float32)
    N, n = Xs.shape
    phys = d.get("physics", {})
    if phys and {k: phys.get(k) for k in PHYSICS} != PHYSICS:
        raise RuntimeError(f"unexpected physics entry {phys}")
    src = None
    if a.source and os.path.exists(a.source):
        src = pickle.load(open(a.source, "rb"))
    meta: dict = dict(t2_file=a.t2, source_file=a.source if src is not None else None, physics=PHYSICS, seed=SEED,
                      alphas=list(ALPHAS))
    if src is not None:
        meta["source_vs_t2"] = {k: bool(np.array_equal(np.asarray(src[k]), np.asarray(d[k])))
                                for k in ("points", "faces", "X_data", "Y_data")}
    ops = generator_operators(points, faces)
    X64, alpha, beta = draw_samples(points, N)
    meta["X_regen_maxabs_vs_stored"] = float(np.abs(X64.astype(np.float32) - Xs).max())
    V = np.stack([velocity(ops, alpha[s], beta[s]) for s in range(N)]).astype(np.float32)
    if src is not None:
        Vs = np.asarray(src["V_data"], dtype=np.float32)
        meta["V_regen_maxabs_vs_stored"] = float(np.abs(V - Vs).max())
        meta["V_regen_bitwise_equal"] = bool(np.array_equal(V, Vs))
        V = Vs
    print(f"[gen_T2v] X regen max abs {meta['X_regen_maxabs_vs_stored']:.2e}; "
          f"V regen max abs {meta.get('V_regen_maxabs_vs_stored', float('nan')):.2e} ({time.time() - t0:.1f}s)",
          flush=True)
    D, dt, ns = PHYSICS["diffusivity"], PHYSICS["dt"], PHYSICS["n_steps"]
    solve = make_solver(ops, D, dt)
    nchk = min(a.check_samples, N)
    Yr = np.zeros((nchk, n))
    for s in range(nchk):
        Yr[s], _ = simulate(ops, X64[s], velocity(ops, alpha[s], beta[s]), ns, solve, dt)
    err = np.abs(Yr.astype(np.float32) - Ys[:nchk])
    meta["target_check"] = dict(samples=nchk, maxabs=float(err.max()), max_rel_l2=float(max(
        _rel(Yr[s], Ys[s]) for s in range(nchk))), target_maxabs=float(np.abs(Ys[:nchk]).max()),
        scheme="u <- (I - dt D L)^-1 (u - dt v . g(u)), D=0.03, dt=0.02, 10 steps (generate_torus_fast.py)")
    print(f"[gen_T2v] target check ({nchk} samples): max abs {err.max():.2e}, max rel l2 "
          f"{meta['target_check']['max_rel_l2']:.2e} ({time.time() - t0:.1f}s)", flush=True)
    # descriptions in the paper versus the stored targets (first 100 samples)
    A = dual_areas(points, faces)
    variants = {
        "paper_text_15_steps_nu_0.01": dict(D=0.01, steps=15, mass=None),
        "lumped_mass_D0.03_10_steps": dict(D=0.03, steps=10, mass=A),
        "lumped_mass_nu0.01_15_steps": dict(D=0.01, steps=15, mass=A),
        "no_diffusion_10_steps": dict(D=0.0, steps=10, mass=None),
        "same_scheme_dt_over_10_100_steps": dict(D=0.03, steps=100, mass=None, dt=dt / 10),
    }
    vcheck = {}
    for name, v in variants.items():
        dtv = v.get("dt", dt)
        sv = make_solver(ops, v["D"], dtv, v["mass"])
        rel = [_rel(simulate(ops, X64[s], velocity(ops, alpha[s], beta[s]), v["steps"], sv, dtv)[0], Ys[s])
               for s in range(100)]
        vcheck[name] = dict(mean_rel_l2=float(np.mean(rel)), max_rel_l2=float(np.max(rel)))
    meta["variant_check_first100"] = vcheck
    print(f"[gen_T2v] variants: {json.dumps(vcheck)}", flush=True)
    meta["diagnostics"] = scheme_diagnostics(points, faces, ops, Xs.astype(np.float64), Ys.astype(np.float64),
                                             alpha, beta)
    # extra evaluation sets on the test split (sequential 70/15/15)
    nt, nv = int(0.7 * N), int(0.15 * N)
    test = np.arange(nt + nv, N)
    Xt = Xs[test].astype(np.float64)
    sweep_V = np.zeros((len(ALPHAS), len(test), n, 3), dtype=np.float32)
    sweep_Y = np.zeros((len(ALPHAS), len(test), n), dtype=np.float32)
    growth = {}
    for ia, al in enumerate(ALPHAS):
        for j, s in enumerate(test):
            vf = velocity(ops, al, beta[s])
            sweep_V[ia, j] = vf
            sweep_Y[ia, j] = simulate(ops, Xt[j], vf, ns, solve, dt)[0]
        growth[str(al)] = float(np.mean(np.linalg.norm(sweep_Y[ia], axis=1) / np.linalg.norm(Xt, axis=1)))
    Y10 = np.zeros((len(test), n), dtype=np.float32)
    Y20 = np.zeros((len(test), n), dtype=np.float32)
    for j, s in enumerate(test):
        u20, st = simulate(ops, Xt[j], velocity(ops, alpha[s], beta[s]), 2 * ns, solve, dt, keep=(ns,))
        Y10[j], Y20[j] = st[ns], u20
    meta["alpha_sweep"] = dict(alphas=list(ALPHAS), test_index=[int(test[0]), int(test[-1])],
                               amplitude_ratio_l2_mean=growth,
                               note="initial conditions (stored float32 X) and beta of the test samples kept")
    meta["rollout"] = dict(Y10_maxabs_vs_stored=float(np.abs(Y10 - Ys[test]).max()),
                           Y20_amplitude_ratio_l2_mean=float(np.mean(np.linalg.norm(Y20, axis=1) /
                                                                      np.linalg.norm(Xt, axis=1))))
    meta["seconds"] = time.time() - t0
    if a.meta_only:
        with open(a.meta_only, "w") as f:
            json.dump(meta, f, indent=2, default=float)
        print(json.dumps(meta, indent=2, default=float))
        return
    import torch
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(dict(V=torch.from_numpy(np.ascontiguousarray(V)), alpha=torch.from_numpy(alpha),
                    beta=torch.from_numpy(beta), vt=torch.from_numpy(ops["vt"].astype(np.float32)),
                    vp=torch.from_numpy(ops["vp"].astype(np.float32)), test_index=torch.from_numpy(test),
                    alpha_sweep=dict(alphas=torch.tensor(ALPHAS, dtype=torch.float64),
                                     V=torch.from_numpy(sweep_V), Y=torch.from_numpy(sweep_Y)),
                    rollout=dict(Y10=torch.from_numpy(Y10), Y20=torch.from_numpy(Y20)), meta=meta), a.out)
    with open(a.out.replace(".pt", ".json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(json.dumps(meta, indent=2, default=float))


if __name__ == "__main__":
    main()
