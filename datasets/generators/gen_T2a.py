"""Anisotropic scalar transport on the torus (task T2a): field-aligned diffusion along helical field lines.

    d_t u + div(v u) = div(D grad u)    on the torus surface of T2,    t in [0, T],  T = 0.2

* **mesh**: the T2 torus (``pv.ParametricTorus(R=1, r=0.4, u_res=60, v_res=30)`` triangulated, cleaned, centred and
  divided by its maximal radius 1.4; 1711 vertices, a structured 59 x 29 grid; reproduced bitwise by
  :func:`torus_mesh`), and for the zero-shot mesh-transfer tests the same construction at twice (``u_res=119,
  v_res=59``: 6844 vertices) and half (``31, 16``: 450 vertices) the linear resolution.
* **initial condition**: the T2 initial conditions (2-5 Gaussian blobs, widths U[0.1, 0.3], amplitudes U[-2, 2], vertex
  mean removed; the T2 generator's ``RandomState(42)`` stream, replayed by :func:`draw_blobs`; on the T2 mesh they
  equal the stored ``X_data``).
* **velocity** (divergence free): ``v = alpha e_theta + beta (R / rho) e_phi + n x grad psi'`` with the T2 amplitudes
  ``alpha ~ U(0.3, 1.5)``, ``beta ~ U(0, 0.8)`` of every sample, the unit toroidal field ``e_theta``, the solenoidal
  poloidal field ``(R / rho) e_phi`` (``rho`` = distance to the symmetry axis, ``R`` the tube-centre radius; the T2
  generator's ``vp`` was a compressive field, see ``gen_T2v.py``) and a random solenoidal perturbation from the
  stream function ``psi' = sum_k a_k cos(m_k theta + n_k phi + p_k)`` with RMS speed ``eps ~ U(0, 0.3)``.
* **diffusion**: ``D = D_perp P + (D_par - D_perp) b b^T`` (tangent plane), ``b = cos(chi) e_theta + sin(chi) e_phi``
  a helical unit direction with pitch angle ``chi = chi0 + chi1 sin(phi + p1) + chi2 sin(theta + p2)`` (``chi0 ~
  U(-1.2, 1.2)``, ``chi1, chi2 ~ U(0, 0.3)``): transport along field lines on a toroidal flux surface;
  ``D_par = D0 sqrt(r)``, ``D_perp = D0 / sqrt(r)`` with the anisotropy ratio ``r ~ logU(1, 100)`` and the geometric
  mean ``D0 ~ logU(0.005, 0.02)`` per sample (parallel diffusion length up to 0.28 = 3.7 median edges over T).

Discretisation (float64, conservative, finite volumes on the barycentric dual cells = DEC with the Whitney mass):

* mass ``M`` = barycentric dual areas (lumped); diffusion ``K(D)`` = P1 / Whitney-Galerkin stiffness
  ``sum_f |f| grad(phi)^T D_f grad(phi)`` with ``D_f`` evaluated at the face barycentre (tangent to the face);
* advection: dual-edge fluxes from the stream function of ``v`` (``Psi = beta R theta - alpha r phi + psi'``, the
  flux through a dual edge is the difference of ``Psi`` at its two ends; multivalued parts by wrapped angle
  differences), so the discrete divergence is exactly zero; centred conservative flux
  ``(A u)_i = sum_j F_ij (u_i + u_j) / 2`` (skew-symmetric, ``1^T A = 0``);
* Crank-Nicolson ``(M + dt/2 (A + K)) u^{n+1} = (M - dt/2 (A + K)) u^n`` with ``N_SUB`` sub-steps over ``T``, chosen
  by refinement (relative change below 1e-3; ``meta['time_convergence']``).  Unconditionally stable and mass
  conserving to round-off (``meta['mass_drift_max']``); the isotropic limit ``K(D0 I) = D0 K_cot`` is checked.

Inputs (``rhmp.tasks.torus.load_T2a``): ``f``; the velocity (vertex vectors; RHMP: edge one-form + half-difference as
T2v); the anisotropy as descriptors only: per edge ``cos^2(t_e, b)`` and ``log(t_e^T D t_e)`` (means over the incident
faces), per face ``log D_par, log D_perp``; baselines: node features ``(f, Vx, Vy, Vz, bx, by, bz, log D_par,
log D_perp)``.  Target: ``u(T)``.

Output ``datasets/v2/T2a.pt`` (+ ``.json``): the base set (3000 samples, the T2 split) and, for the 450 test samples,
the zero-shot sets ``alpha`` (alpha in ``ALPHAS_X``), ``ratio`` (r in ``RATIOS_X``), ``rollout`` (``u(2T)``),
``fine`` / ``coarse`` (the same instances re-solved on the 6844- / 450-vertex tori).  ``--val-meshes PATH``: only
the 450 validation instances re-solved on the fine and coarse tori (mesh transfer judged on validation data;
``datasets/v2/T2a_valmesh.pt``).

    python3 -u datasets/generators/gen_T2a.py [--workers 16] [--check-only | --val-meshes PATH]
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import aniso_fields as AF  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
T_END = 0.2                                   # horizon of T2 (10 steps of 0.02)
N_SUB = 200                                   # Crank-Nicolson sub-steps over T (refinement study in meta)
ALPHAS_X = (2.0, 2.5)                         # zero-shot toroidal amplitudes (training 0.3 .. 1.5)
RATIOS_X = (300.0, 1000.0)                    # zero-shot anisotropy ratios (training 1 .. 100)
MESHES = {"base": (60, 30), "fine": (119, 59), "coarse": (31, 16)}
SEED_T2 = 42                                  # T2 generator stream (initial conditions, alpha, beta)
SEED_NEW = 2027                               # new per-sample fields (pitch, D, perturbation)
MODES = ((0, 1), (1, 0), (1, 1), (1, -1), (2, 1), (1, 2), (0, 2), (2, 0))      # (m, n) of the perturbation


# ======================================================================================================================
# mesh and torus geometry
# ======================================================================================================================
def torus_mesh(u_res: int, v_res: int) -> tuple[np.ndarray, np.ndarray]:
    """The T2 generator's torus (``create_torus_mesh`` of ``generate_torus_fast.py``): float32 points, int64 faces."""
    import pyvista as pv
    m = pv.PolyData(pv.ParametricTorus(ringradius=1.0, crosssectionradius=0.4, u_res=u_res, v_res=v_res))
    if not m.is_all_triangles:
        m = m.triangulate()
    m = m.clean()
    p = m.points.copy()
    f = m.faces.reshape((-1, 4))[:, 1:].astype(np.int64)
    p -= p.mean(axis=0)
    p /= (np.max(np.linalg.norm(p, axis=1)) + 1e-9)
    return np.asarray(p), f


def torus_radii(P: np.ndarray) -> tuple[float, float]:
    """Tube-centre radius ``R`` and tube radius ``r`` of points on a torus about the z axis: ``rho^2 + z^2 = 2 R rho +
    (r^2 - R^2)`` is linear in ``rho`` (exact least squares for points on the surface)."""
    P = np.asarray(P, dtype=np.float64)
    rho = np.hypot(P[:, 0], P[:, 1])
    A = np.stack([rho, np.ones_like(rho)], 1)
    c = np.linalg.lstsq(A, rho ** 2 + P[:, 2] ** 2, rcond=None)[0]
    R = c[0] / 2.0
    return float(R), float(np.sqrt(c[1] + R * R))


def frames(P: np.ndarray, R: float) -> dict:
    """Angles and the right-handed orthonormal frame ``(e_theta, e_phi, n)`` (n outward) at points ``P (M, 3)``."""
    P = np.asarray(P, dtype=np.float64)
    rho = np.hypot(P[:, 0], P[:, 1])
    th = np.arctan2(P[:, 1], P[:, 0])
    ph = np.arctan2(P[:, 2], rho - R)
    er = np.stack([np.cos(th), np.sin(th), np.zeros_like(th)], 1)
    et = np.stack([-np.sin(th), np.cos(th), np.zeros_like(th)], 1)
    ez = np.array([0.0, 0.0, 1.0])
    ep = -np.sin(ph)[:, None] * er + np.cos(ph)[:, None] * ez
    nn = np.cos(ph)[:, None] * er + np.sin(ph)[:, None] * ez
    return dict(rho=rho, th=th, ph=ph, et=et, ep=ep, n=nn)


def wrap(a: np.ndarray) -> np.ndarray:
    return (a + np.pi) % (2 * np.pi) - np.pi


# ======================================================================================================================
# per-sample fields
# ======================================================================================================================
def draw_blobs(n_samples: int, base_points: np.ndarray) -> list[dict]:
    """Blob parameters of the T2 initial conditions (the T2 generator's RNG call order; centres as 3-D points of the
    T2 mesh) and its ``alpha``, ``beta``."""
    n = len(base_points)
    rng = np.random.RandomState(SEED_T2)
    out = []
    for _ in range(n_samples):
        nb = rng.randint(2, 6)
        c, sig, st = [], [], []
        for _ in range(nb):
            c.append(base_points[rng.randint(0, n)])
            sig.append(rng.uniform(0.1, 0.3))
            st.append(rng.uniform(-2.0, 2.0))
        a, b = rng.uniform(0.3, 1.5), rng.uniform(0.0, 0.8)
        out.append(dict(c=np.stack(c), sig=np.array(sig), st=np.array(st), alpha=a, beta=b))
    return out


def initial_condition(blob: dict, points: np.ndarray) -> np.ndarray:
    """``sum_k st_k exp(-|x - c_k|^2 / (2 sig_k^2))`` minus its vertex mean, with the generator's float32 arithmetic
    (on the T2 mesh this is the stored ``X_data``)."""
    u0 = np.zeros(len(points))
    for c, sig, st in zip(blob["c"], blob["sig"], blob["st"]):
        u0 += st * np.exp(-np.linalg.norm(points - c, axis=1) ** 2 / (2 * sig ** 2))
    return u0 - u0.mean()


def draw_new(n_samples: int) -> list[dict]:
    """New per-sample parameters: pitch of the field lines, diffusivities, solenoidal perturbation."""
    rng = np.random.default_rng(SEED_NEW)
    out = []
    for _ in range(n_samples):
        chi = dict(c0=rng.uniform(-1.2, 1.2), c1=rng.uniform(0.0, 0.3), c2=rng.uniform(0.0, 0.3),
                   p1=rng.uniform(0, 2 * np.pi), p2=rng.uniform(0, 2 * np.pi))
        ratio = float(np.exp(rng.uniform(0.0, np.log(100.0))))
        D0 = float(np.exp(rng.uniform(np.log(0.005), np.log(0.02))))
        amp = rng.normal(size=len(MODES)) / np.array([m * m + k * k for m, k in MODES], dtype=np.float64)
        ph = rng.uniform(0, 2 * np.pi, len(MODES))
        eps = rng.uniform(0.0, 0.3)
        out.append(dict(chi=chi, ratio=ratio, D0=D0, amp=amp, mph=ph, eps=eps))
    return out


def stream_pert(q: dict, th: np.ndarray, ph: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``psi'`` and its angular derivatives at angles ``(th, ph)`` (times ``scale``)."""
    psi = np.zeros_like(th)
    dth = np.zeros_like(th)
    dph = np.zeros_like(th)
    for (m, k), a, p in zip(MODES, q["amp"], q["mph"]):
        arg = m * th + k * ph + p
        psi += a * np.cos(arg)
        dth -= a * m * np.sin(arg)
        dph -= a * k * np.sin(arg)
    return scale * psi, scale * dth, scale * dph


def velocity(F: dict, R: float, r: float, alpha: float, beta: float, q: dict, scale: float) -> np.ndarray:
    """``v = alpha e_theta + beta (R / rho) e_phi + (psi'_theta / rho) e_phi - (psi'_phi / r) e_theta``."""
    _, dth, dph = stream_pert(q, F["th"], F["ph"], scale)
    vt = alpha - dph / r
    vp = beta * R / F["rho"] + dth / F["rho"]
    return vt[:, None] * F["et"] + vp[:, None] * F["ep"]


def stream_total(F: dict, R: float, r: float, alpha: float, beta: float, q: dict, scale: float):
    """Single-valued part ``psi'`` and the coefficients of the multivalued parts of ``Psi = beta R theta - alpha r phi
    + psi'`` (``n x grad Psi = v``)."""
    psi, _, _ = stream_pert(q, F["th"], F["ph"], scale)
    return psi, beta * R, -alpha * r


def pert_scale(q: dict, Fv: dict, r: float) -> float:
    """Scale of ``psi'`` such that the RMS speed of the perturbation over the given points is ``q['eps']``."""
    if q["eps"] <= 0:
        return 0.0
    _, dth, dph = stream_pert(q, Fv["th"], Fv["ph"], 1.0)
    sp_ = np.sqrt(((dth / Fv["rho"]) ** 2 + (dph / r) ** 2).mean())
    return q["eps"] / max(sp_, 1e-300)


def pitch(q: dict, F: dict) -> np.ndarray:
    c = q["chi"]
    return c["c0"] + c["c1"] * np.sin(F["ph"] + c["p1"]) + c["c2"] * np.sin(F["th"] + c["p2"])


def direction(q: dict, F: dict) -> np.ndarray:
    chi = pitch(q, F)
    return np.cos(chi)[:, None] * F["et"] + np.sin(chi)[:, None] * F["ep"]


# ======================================================================================================================
# discretisation
# ======================================================================================================================
class MeshOps:
    """Geometry of one torus mesh shared by all samples: faces, edges, normals, barycentres, dual-edge ends."""

    def __init__(self, points: np.ndarray, faces: np.ndarray, R: float, r: float) -> None:
        self.P = np.asarray(points, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.n0 = len(self.P)
        self.R, self.r = R, r
        topo = AF.tri_topology(self.faces, self.n0)
        self.edges = topo["edges"]
        self.face_edges = topo["face_edges"]
        self.n1 = len(self.edges)
        V = self.P[self.faces]
        nf = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
        self.area = 0.5 * np.linalg.norm(nf, axis=1)
        self.nf = nf / (2 * self.area[:, None])
        self.bary = V.mean(1)
        self.Fv = frames(self.P, R)
        self.Ff = frames(self.bary, R)
        if np.mean((self.nf * self.Ff["n"]).sum(1)) < 0:          # orient face normals outward (frame normal)
            self.nf = -self.nf
        mass = np.zeros(self.n0)
        np.add.at(mass, self.faces.ravel(), np.repeat(self.area / 3.0, 3))
        self.mass = mass
        # incident faces of every edge (closed manifold: exactly two)
        ef = [[] for _ in range(self.n1)]
        for f in range(len(self.faces)):
            for j in range(3):
                ef[self.face_edges[f, j]].append(f)
        ef = np.array(ef, dtype=np.int64)
        if ef.shape != (self.n1, 2):
            raise ValueError("torus mesh: every edge must have exactly two faces")
        a, b = self.edges[:, 0], self.edges[:, 1]
        t = self.P[b] - self.P[a]
        self.elen = np.linalg.norm(t, axis=1)
        self.tu = t / self.elen[:, None]
        mid = 0.5 * (self.P[a] + self.P[b])
        Fm = frames(mid, R)
        side = ((self.bary[ef] - mid[:, None]) * np.cross(Fm["n"], self.tu)[:, None]).sum(-1)    # (n1, 2)
        first = np.where(side[:, 0] < side[:, 1], 0, 1)
        self.dual_P = ef[np.arange(self.n1), first]               # dual edge P -> Q: (P->Q) x n points src -> dst
        self.dual_Q = ef[np.arange(self.n1), 1 - first]
        self.dual_len = (np.linalg.norm(self.bary[self.dual_P] - mid, axis=1)
                         + np.linalg.norm(self.bary[self.dual_Q] - mid, axis=1))
        self.mid = mid
        self.Fm = Fm
        Kc, _, _ = AF.p1_stiffness(self.P, self.faces, np.repeat(np.eye(3)[None], len(self.faces), 0))
        self.K_iso = Kc

    def fluxes(self, alpha: float, beta: float, q: dict, scale: float) -> np.ndarray:
        """Dual-edge fluxes ``F_e`` (src -> dst) of ``v`` from its stream function (exactly divergence free)."""
        psi, cth, cph = stream_total(self.Ff, self.R, self.r, alpha, beta, q, scale)
        P, Q = self.dual_P, self.dual_Q
        th, ph = self.Ff["th"], self.Ff["ph"]
        return (psi[P] - psi[Q]) + cth * wrap(th[P] - th[Q]) + cph * wrap(ph[P] - ph[Q])

    def advection(self, F: np.ndarray) -> sp.csr_matrix:
        """``(A u)_i = sum_{e ∋ i} F_{i -> j} (u_i + u_j) / 2``."""
        a, b = self.edges[:, 0], self.edges[:, 1]
        h = 0.5 * F
        rows = np.concatenate([a, a, b, b])
        cols = np.concatenate([a, b, a, b])
        vals = np.concatenate([h, h, -h, -h])
        return sp.csr_matrix((vals, (rows, cols)), shape=(self.n0, self.n0))

    def face_tensors(self, bf: np.ndarray, Dpar: float, Dperp: float) -> np.ndarray:
        """``D_f = D_perp P_f + (D_par - D_perp) b_f b_f^T`` in the plane of each face (``b`` projected, normalised)."""
        n = self.nf
        bt = bf - (bf * n).sum(1, keepdims=True) * n
        bt /= np.linalg.norm(bt, axis=1, keepdims=True)
        Pf = np.eye(3)[None] - n[:, :, None] * n[:, None, :]
        return Dperp * Pf + (Dpar - Dperp) * bt[:, :, None] * bt[:, None, :], bt

    def edge_inputs(self, D: np.ndarray, bt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per edge: mean over the incident faces of ``cos^2(t_e, b_f)`` and ``log`` of the mean ``t_e^T D_f t_e``."""
        fe = self.face_edges                                         # (n2, 3)
        t = self.tu[fe]                                              # (n2, 3, 3)
        c2 = np.einsum("fjd,fd->fj", t, bt) ** 2
        p = np.einsum("fjd,fde,fje->fj", t, D, t)
        return AF.cell_edge_average(c2, fe, self.n1), np.log(AF.cell_edge_average(p, fe, self.n1))

    def stiffness(self, D: np.ndarray) -> sp.csr_matrix:
        K, _, _ = AF.p1_stiffness(self.P, self.faces, D)
        return K


def solve_sample(ops: MeshOps, u0: np.ndarray, A: sp.csr_matrix, K: sp.csr_matrix, n_sub: int, T: float,
                 keep: tuple[float, ...] = ()) -> tuple[np.ndarray, dict, float]:
    """Crank-Nicolson over ``[0, T]`` with ``n_sub`` steps; returns ``u(T)``, the states at the times in ``keep`` and
    the largest relative mass drift ``|sum M (u^n - u^0)| / sum M |u^0|``."""
    dt = T / n_sub
    M = sp.diags(ops.mass)
    L = (A + K).tocsr()
    lhs = splu((M + 0.5 * dt * L).tocsc())
    rhs_op = (M - 0.5 * dt * L).tocsr()
    u = np.asarray(u0, dtype=np.float64)
    m0, ref = float(ops.mass @ u), float(ops.mass @ np.abs(u)) + 1e-300
    drift = 0.0
    states = {}
    keep_steps = {int(round(k / dt)): k for k in keep}
    for s in range(1, n_sub + 1):
        u = lhs.solve(rhs_op @ u)
        drift = max(drift, abs(float(ops.mass @ u) - m0) / ref)
        if s in keep_steps:
            states[keep_steps[s]] = u.copy()
    return u, states, drift


def sample_fields(ops: MeshOps, blob: dict, q: dict, alpha: float | None = None, ratio: float | None = None,
                  scale: float | None = None) -> dict:
    """Velocity, direction, tensors, operators and inputs of one sample on one mesh."""
    a = blob["alpha"] if alpha is None else alpha
    b = blob["beta"]
    r = q["ratio"] if ratio is None else ratio
    Dpar, Dperp = q["D0"] * np.sqrt(r), q["D0"] / np.sqrt(r)
    V = velocity(ops.Fv, ops.R, ops.r, a, b, q, scale)
    F = ops.fluxes(a, b, q, scale)
    bv = direction(q, ops.Fv)
    bf = direction(q, ops.Ff)
    D, bt = ops.face_tensors(bf, Dpar, Dperp)
    c2, logp = ops.edge_inputs(D, bt)
    return dict(V=V, F=F, bv=bv, D=D, c2=c2, logp=logp, Dpar=Dpar, Dperp=Dperp, A=ops.advection(F),
                K=ops.stiffness(D))


# ======================================================================================================================
# workers
# ======================================================================================================================
_G: dict = {}


def _init(meshes: dict, blobs: list, news: list, scales: dict, n_sub: int) -> None:
    _G.update(meshes=meshes, blobs=blobs, news=news, scales=scales, n_sub=n_sub)


def _work(job: tuple) -> tuple:
    kind, i = job
    ops_b = _G["meshes"]["base"]
    blob, q = _G["blobs"][i], _G["news"][i]
    sc = _G["scales"][i]
    n_sub = _G["n_sub"]
    if kind in ("fine", "coarse"):
        ops = _G["meshes"][kind]
        fl = sample_fields(ops, blob, q, scale=sc)
        u0 = initial_condition(blob, ops.P.astype(np.float32))
        mult = 2 if kind == "fine" else 1
        uT, _, drift = solve_sample(ops, u0, fl["A"], fl["K"], mult * n_sub, T_END)
        return kind, i, dict(X=u0, Y=uT, V=fl["V"], bv=fl["bv"], c2=fl["c2"], logp=fl["logp"], drift=drift)
    u0 = _G["X"][i] if "X" in _G else initial_condition(blob, ops_b.P.astype(np.float32))
    if kind == "base":
        fl = sample_fields(ops_b, blob, q, scale=sc)
        uT, st, drift = solve_sample(ops_b, u0, fl["A"], fl["K"], n_sub, T_END)
        return kind, i, dict(Y=uT, V=fl["V"], bv=fl["bv"], c2=fl["c2"], logp=fl["logp"], drift=drift,
                             Dpar=fl["Dpar"], Dperp=fl["Dperp"])
    if kind == "rollout":
        fl = sample_fields(ops_b, blob, q, scale=sc)
        u2, st, drift = solve_sample(ops_b, u0, fl["A"], fl["K"], 2 * n_sub, 2 * T_END, keep=(T_END,))
        return kind, i, dict(Y2=u2, Y1=st[T_END], drift=drift)
    if kind.startswith("alpha"):
        al = float(kind[5:])
        fl = sample_fields(ops_b, blob, q, alpha=al, scale=sc)
        uT, _, drift = solve_sample(ops_b, u0, fl["A"], fl["K"], n_sub, T_END)
        return kind, i, dict(Y=uT, V=fl["V"], drift=drift)
    if kind.startswith("ratio"):
        ra = float(kind[5:])
        fl = sample_fields(ops_b, blob, q, ratio=ra, scale=sc)
        uT, _, drift = solve_sample(ops_b, u0, fl["A"], fl["K"], n_sub, T_END)
        return kind, i, dict(Y=uT, c2=fl["c2"], logp=fl["logp"], Dpar=fl["Dpar"], Dperp=fl["Dperp"], drift=drift)
    raise ValueError(kind)


# ======================================================================================================================
# checks
# ======================================================================================================================
def checks(meshes: dict, blobs: list, news: list, scales: dict, X: np.ndarray, n_check: int = 8) -> dict:
    """Time-step convergence, flux consistency, discrete divergence, isotropic limit, mass conservation, mesh
    convergence (base vs fine, interpolated by nearest base vertex of the fine mesh's coincident vertices)."""
    ops = meshes["base"]
    out: dict = {}
    # discrete divergence of the stream-function fluxes and consistency with the DEC flux star_1 (v.t)
    q, blob = news[0], blobs[0]
    F = ops.fluxes(blob["alpha"], blob["beta"], q, scales[0])
    div = np.zeros(ops.n0)
    np.add.at(div, ops.edges[:, 0], F)
    np.add.at(div, ops.edges[:, 1], -F)
    Fq = np.zeros(ops.n1)                                            # midpoint rule on the two dual segments
    for a_, b_ in ((ops.bary[ops.dual_P], ops.mid), (ops.mid, ops.bary[ops.dual_Q])):
        Fs = frames(0.5 * (a_ + b_), ops.R)
        Vs = velocity(Fs, ops.R, ops.r, blob["alpha"], blob["beta"], q, scales[0])
        Fq += (Vs * np.cross(b_ - a_, Fs["n"])).sum(1)               # v . (tau x n) |segment|
    out["flux"] = dict(div_max_rel=float(np.abs(div).max() / np.abs(F).max()),
                       quadrature_rel_l2=float(np.linalg.norm(F - Fq) / np.linalg.norm(Fq)))
    # isotropic limit: K(D0 I_tangent) = D0 K_cot
    Df, _ = ops.face_tensors(direction(q, ops.Ff), 1.0, 1.0)
    Kt = ops.stiffness(Df)
    out["isotropic_limit_rel"] = float(abs(Kt - ops.K_iso).max() / abs(ops.K_iso).max())
    # time-step convergence (relative l2 change when doubling the number of sub-steps)
    tc = {}
    for n_sub in (25, 50, 100, 200, 400, 800):
        tc[n_sub] = []
        for i in range(n_check):
            fl = sample_fields(ops, blobs[i], news[i], scale=scales[i])
            tc[n_sub].append(solve_sample(ops, X[i], fl["A"], fl["K"], n_sub, T_END)[0])
    keys = sorted(tc)
    out["time_convergence"] = {f"{a}->{b}": float(np.max([np.linalg.norm(tc[a][i] - tc[b][i]) /
                                                          np.linalg.norm(tc[b][i]) for i in range(n_check)]))
                               for a, b in zip(keys[:-1], keys[1:])}
    out["time_error_vs_800_at_N_SUB"] = float(np.max([np.linalg.norm(tc[N_SUB][i] - tc[800][i]) /
                                                      np.linalg.norm(tc[800][i]) for i in range(n_check)]))
    # mesh convergence: the fine torus contains the base vertices? (structured grids: every second vertex)
    fine = meshes["fine"]
    d2 = ((fine.P[:, None, :] - ops.P[None, :, :]) ** 2).sum(-1) if fine.n0 * ops.n0 < 2e7 else None
    if d2 is not None:
        nn = d2.argmin(0)                                            # fine vertex nearest to each base vertex
        dist = np.sqrt(d2[nn, np.arange(ops.n0)])
        errs = []
        for i in range(n_check):
            fb = sample_fields(ops, blobs[i], news[i], scale=scales[i])
            ub = solve_sample(ops, X[i], fb["A"], fb["K"], N_SUB, T_END)[0]
            ff = sample_fields(fine, blobs[i], news[i], scale=scales[i])
            uf = solve_sample(fine, initial_condition(blobs[i], fine.P.astype(np.float32)), ff["A"], ff["K"],
                              2 * N_SUB, T_END)[0]
            errs.append(float(np.linalg.norm(ub - uf[nn]) / np.linalg.norm(uf[nn])))
        out["mesh_convergence_base_vs_fine"] = dict(rel_l2_mean=float(np.mean(errs)), rel_l2_max=float(np.max(errs)),
                                                    coincident_vertex_dist_max=float(dist.max()))
    return out


# ======================================================================================================================
def _val_meshes(a, meshes: dict, blobs: list, news: list, scales: dict, meta: dict, N: int, t0: float) -> None:
    """The validation instances (the T2 split) re-solved on the fine and coarse tori, written to ``a.val_meshes``."""
    import torch
    nt, nv = int(0.7 * N), int(0.15 * N)
    val = list(range(nt, nt + nv))
    jobs = [("fine", i) for i in val] + [("coarse", i) for i in val]
    res: dict = {}
    with Pool(a.workers, initializer=_init, initargs=(meshes, blobs, news, scales, N_SUB)) as pool:
        for kind, i, r_ in pool.imap_unordered(_work, jobs, chunksize=4):
            res[(kind, i)] = r_
    f32 = lambda x: torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float32)))  # noqa: E731
    out = dict(val_index=torch.tensor(val))
    for name in ("fine", "coarse"):
        m = meshes[name]
        st = lambda key: f32(np.stack([res[(name, i)][key] for i in val]))  # noqa: E731
        out[name] = dict(points=f32(m.P), faces=torch.from_numpy(m.faces), edges=torch.from_numpy(m.edges),
                         X=st("X"), Y=st("Y"), V=st("V"), b=st("bv"), edge_cos2=st("c2"), edge_logp=st("logp"))
    meta = dict(meta, split="validation", val_index=[val[0], val[-1]],
                mass_drift_max=float(max(r_["drift"] for r_ in res.values())), seconds=time.time() - t0)
    out["meta"] = meta
    os.makedirs(os.path.dirname(os.path.abspath(a.val_meshes)), exist_ok=True)
    torch.save(out, a.val_meshes)
    with open(os.path.splitext(a.val_meshes)[0] + ".json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(json.dumps(meta, indent=2, default=float))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--t2", default=os.path.join(ROOT, "datasets", "T2_torus_advection_diffusion.pkl"))
    p.add_argument("--t2v", default=os.path.join(ROOT, "datasets", "v2", "T2v.pt"))
    p.add_argument("--out", default=os.path.join(ROOT, "datasets", "v2", "T2a.pt"))
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--val-meshes", default=None, metavar="PATH",
                   help="write only the validation instances re-solved on the fine and coarse tori to PATH (zero-shot "
                        "mesh transfer judged on validation data; the base-mesh data are those of T2a.pt)")
    a = p.parse_args()
    t0 = time.time()
    d = pickle.load(open(a.t2, "rb"))
    X = np.asarray(d["X_data"], dtype=np.float64)
    N = len(X)
    meshes = {}
    for name, (ur, vr) in MESHES.items():
        pts, fcs = torus_mesh(ur, vr)
        if name == "base":
            if not (np.array_equal(pts, np.asarray(d["points"])) and np.array_equal(fcs, np.asarray(d["faces"]))):
                raise RuntimeError("torus_mesh(60, 30) does not reproduce the T2 mesh")
            R, r = torus_radii(pts)
        meshes[name] = MeshOps(pts, fcs, R, r)
    blobs = draw_blobs(N, np.asarray(d["points"]))
    x_err = float(np.abs(np.stack([initial_condition(blobs[i], np.asarray(d["points"])) for i in range(8)])
                         .astype(np.float32) - X[:8].astype(np.float32)).max())
    news = draw_new(N)
    scales = {i: pert_scale(news[i], meshes["base"].Fv, r) for i in range(N)}
    meta = dict(T=T_END, n_sub=N_SUB, seeds=dict(t2=SEED_T2, new=SEED_NEW), R=R, r=r, alphas_x=list(ALPHAS_X),
                ratios_x=list(RATIOS_X), meshes={k: dict(n0=v.n0, n1=v.n1, n2=len(v.faces),
                                                         median_edge=float(np.median(v.elen)))
                                                 for k, v in meshes.items()},
                ic_regen_maxabs_vs_T2=x_err)
    if a.t2v and os.path.exists(a.t2v):
        import torch
        v2 = torch.load(a.t2v, map_location="cpu", weights_only=False)
        meta["alpha_beta_vs_T2v_maxabs"] = float(max(np.abs(v2["alpha"].numpy() - [b["alpha"] for b in blobs]).max(),
                                                     np.abs(v2["beta"].numpy() - [b["beta"] for b in blobs]).max()))
    print(f"[gen_T2a] meshes {meta['meshes']} R={R:.6f} r={r:.6f} ic err {x_err:.1e} ({time.time() - t0:.1f}s)",
          flush=True)
    if a.val_meshes:
        return _val_meshes(a, meshes, blobs, news, scales, meta, N, t0)
    meta["checks"] = checks(meshes, blobs, news, scales, X)
    print(f"[gen_T2a] checks {json.dumps(meta['checks'])} ({time.time() - t0:.1f}s)", flush=True)
    if a.check_only:
        print(json.dumps(meta, indent=2, default=float))
        return
    nt, nv = int(0.7 * N), int(0.15 * N)
    test = list(range(nt + nv, N))
    jobs = [("base", i) for i in range(N)] + [("rollout", i) for i in test]
    jobs += [(f"alpha{al}", i) for al in ALPHAS_X for i in test] + [(f"ratio{ra}", i) for ra in RATIOS_X for i in test]
    jobs += [("fine", i) for i in test] + [("coarse", i) for i in test]
    _G["X"] = X
    res: dict = {}
    with Pool(a.workers, initializer=_init, initargs=(meshes, blobs, news, scales, N_SUB)) as pool:
        for k, (kind, i, r_) in enumerate(pool.imap_unordered(_work, jobs, chunksize=4)):
            res[(kind, i)] = r_
            if (k + 1) % 500 == 0:
                print(f"  {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    import torch
    f32 = lambda x: torch.from_numpy(np.ascontiguousarray(np.asarray(x, dtype=np.float32)))  # noqa: E731

    def stack(kind, key, ids):
        return f32(np.stack([res[(kind, i)][key] for i in ids]))
    base_ids = list(range(N))
    drifts = [res[("base", i)]["drift"] for i in base_ids]
    blob_arr = dict(alpha=np.array([b["alpha"] for b in blobs]), beta=np.array([b["beta"] for b in blobs]))
    par = dict(ratio=np.array([q["ratio"] for q in news]), D0=np.array([q["D0"] for q in news]),
               eps=np.array([q["eps"] for q in news]), chi0=np.array([q["chi"]["c0"] for q in news]))
    out = dict(
        points=f32(meshes["base"].P), faces=torch.from_numpy(meshes["base"].faces), edges=torch.from_numpy(meshes["base"].edges),
        X=f32(X), Y=stack("base", "Y", base_ids), V=stack("base", "V", base_ids), b=stack("base", "bv", base_ids),
        edge_cos2=stack("base", "c2", base_ids), edge_logp=stack("base", "logp", base_ids),
        Dpar=torch.tensor([res[("base", i)]["Dpar"] for i in base_ids]),
        Dperp=torch.tensor([res[("base", i)]["Dperp"] for i in base_ids]),
        params={k: torch.from_numpy(v) for k, v in {**blob_arr, **par}.items()},
        test_index=torch.tensor(test),
        rollout=dict(Y1=stack("rollout", "Y1", test), Y2=stack("rollout", "Y2", test)),
        alpha={f"{al:g}": dict(Y=stack(f"alpha{al}", "Y", test), V=stack(f"alpha{al}", "V", test)) for al in ALPHAS_X},
        ratio={f"{ra:g}": dict(Y=stack(f"ratio{ra}", "Y", test), edge_cos2=stack(f"ratio{ra}", "c2", test),
                               edge_logp=stack(f"ratio{ra}", "logp", test),
                               Dpar=torch.tensor([res[(f"ratio{ra}", i)]["Dpar"] for i in test]),
                               Dperp=torch.tensor([res[(f"ratio{ra}", i)]["Dperp"] for i in test]))
               for ra in RATIOS_X},
    )
    for name in ("fine", "coarse"):
        m = meshes[name]
        out[name] = dict(points=f32(m.P), faces=torch.from_numpy(m.faces), edges=torch.from_numpy(m.edges),
                         X=stack(name, "X", test), Y=stack(name, "Y", test), V=stack(name, "V", test),
                         b=stack(name, "bv", test), edge_cos2=stack(name, "c2", test),
                         edge_logp=stack(name, "logp", test))
    allres = list(res.values())
    meta["mass_drift_max"] = float(max(r_["drift"] for r_ in allres))
    meta["mass_drift_base_max"] = float(max(drifts))
    Yb = out["Y"].double().numpy()
    meta["target_stats"] = dict(amplitude_ratio_l2_mean=float(np.mean(np.linalg.norm(Yb, axis=1) /
                                                                      np.linalg.norm(X, axis=1))),
                                rel_change_l2_mean=float(np.mean(np.linalg.norm(Yb - X, axis=1) /
                                                                 np.linalg.norm(X, axis=1))))
    meta["seconds"] = time.time() - t0
    out["meta"] = meta
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(out, a.out)
    with open(a.out.replace(".pt", ".json"), "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(json.dumps(meta, indent=2, default=float))


if __name__ == "__main__":
    main()
