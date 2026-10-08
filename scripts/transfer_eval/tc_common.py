"""Shared helpers of the zero-shot transfer evaluation (new meshes, new geometries, gauge transformations).

Evaluation only: no training, no write access to the library or to the run directories.  The library is imported
from the repository that contains this file (``RHMP_ROOT``), the runs are read from ``RHMP_RUNS`` (default
``results/`` of the repository, see :data:`SPEC`) and the transfer data sets from ``datasets/v2/transfer``
(``gen_planar.py``).

Contents
* meshes:   seeded random Delaunay meshes of the unit square (original recipe of the T5 / T6 / T7 generators), the
            anisotropic variant, ellipsoid surface meshes (original T3 recipe), mesh statistics;
* fields:   the T6 / T7 field generators of ``datasets/generators/gen_T{6,7}_native.py`` (torch CUDA RNG, seed 42,
            N = 10000) split into (a) the draw of the random parameters of all samples and (b) the evaluation of the fields
            of chosen samples on an arbitrary mesh; the T3 generator of the original data set;
* encodings: the v1 node encodings of an edge cochain (verified in ``scripts/frame_eval``) and the face -> node average;
* runs:     loading of a finished run as ``rhmp.train.eval_ckpt`` rebuilds it, evaluation with the library metrics.
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pickle
import sys
import time

import numpy as np
import torch

# the repository that contains this file (override with RHMP_ROOT); runs are read from RHMP_RUNS (default
# <root>/results, the layout of the shipped result directories: <task>/<model>_s<seed>/)
ROOT = os.path.abspath(os.path.expanduser(os.environ.get(
    "RHMP_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))))
for _p in (ROOT, os.path.join(ROOT, "datasets", "generators")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
DROOT = os.path.join(ROOT, "datasets")
RUNS = os.path.abspath(os.path.expanduser(os.environ.get("RHMP_RUNS", os.path.join(ROOT, "results"))))
EDGE_EPS = 1e-12                       # v1 encode_edges_to_nodes: unit direction = edge / (|edge| + 1e-12)
N_ALL = 10000                          # samples of the original T3 / T6 / T7 data sets
TEST100 = np.arange(8500, 8600)        # test100 of the paper: first 100 samples of the test split (70 / 15 / 15)

BASELINES = ["gcn", "gat", "schnet", "egnn", "mpsn", "sccnn", "gauge_cnn", "gem_cnn", "cw_net", "clifford_smpn",
             "deeponet"]
SPEC = {   # RHMP run and baseline directory (<base>/<model>_s42) of every task, relative to RUNS
    "T3": dict(rhmp="ellipsoid_flow/rhmp_s42", base="ellipsoid_flow"),
    "T6": dict(rhmp="wilson_loop/rhmp_s42", base="wilson_loop"),
    "T7": dict(rhmp="yang_mills/rhmp_s42", base="yang_mills"),
}
# additional RHMP runs to evaluate (task -> name -> run directory relative to RUNS); default: the run of SPEC
RHMP_RUNS: dict = {}
if os.environ.get("RHMP_RUNS_JSON"):                       # {task: {name: run dir}} override (relative to RUNS or absolute)
    import json as _json
    for _t, _r in _json.loads(os.environ["RHMP_RUNS_JSON"]).items():
        RHMP_RUNS[_t] = dict(RHMP_RUNS.get(_t, {}), **_r)


# ======================================================================================================================
# planar meshes
# ======================================================================================================================
def signed_areas(pts: np.ndarray, faces: np.ndarray) -> np.ndarray:
    a, b, c = pts[faces[:, 0]], pts[faces[:, 1]], pts[faces[:, 2]]
    return 0.5 * ((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))


def make_ccw(pts: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, int]:
    """Faces with counter-clockwise orientation (rows with negative signed area get two vertices swapped).
    Returns ``(faces, number of flipped faces)``."""
    flip = signed_areas(pts, faces) < 0
    return np.where(flip[:, None], faces[:, [0, 2, 1]], faces), int(flip.sum())


def graded_x(u: np.ndarray, contrast: float) -> np.ndarray:
    """Map uniform numbers ``u`` in [0, 1] to x coordinates with the density ``rho(x) ~ 1 + (contrast - 1) (1 - x)^2`` (inverse
    CDF on a fine grid): the point density is ``contrast`` times larger at x = 0 than at x = 1, so the local node spacing varies
    by a factor ``sqrt(contrast)`` along x (smooth, bounded grading)."""
    xs = np.linspace(0.0, 1.0, 200001)
    cdf = np.cumsum(1.0 + (contrast - 1.0) * (1.0 - xs) ** 2)
    cdf = (cdf - cdf[0]) / (cdf[-1] - cdf[0])
    return np.interp(u, cdf, xs)


def delaunay_cloud(seed: int, n_pts: int, stretch=None, grade: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Random Delaunay mesh of the unit square, recipe of ``gen_common.delaunay_1024`` (``np.random.seed(seed)``,
    ``rand(n_pts, 2)``, scipy Delaunay).  ``stretch = (sx, sy)``: the triangulation is the Delaunay triangulation of the
    cloud with coordinates multiplied by ``stretch`` (anisotropic neighbour structure); the returned points are the
    unstretched ones, so the domain stays the unit square.  ``grade = c``: the x coordinates are redistributed to the density
    ``1 + (c - 1)(1 - x)^2`` (graded towards x = 0, node spacing varying by sqrt(c)).  Returns ``(pts (n, 2) float64,
    faces (n2, 3))``."""
    from scipy.spatial import Delaunay
    np.random.seed(seed)
    pts = np.random.rand(n_pts, 2).astype(np.float64)
    if grade is not None:
        pts[:, 0] = graded_x(pts[:, 0], grade)
    q = pts if stretch is None else pts * np.asarray(stretch, dtype=np.float64)
    return pts, Delaunay(q).simplices.astype(np.int64)


def canonical_edges(faces: np.ndarray, n: int) -> np.ndarray:
    a, b = faces, np.roll(faces, -1, axis=1)
    keys = np.unique((np.minimum(a, b).astype(np.int64) * n + np.maximum(a, b)).ravel())
    return np.stack([keys // n, keys % n], axis=1).astype(np.int64)


def mesh_stats(pts: np.ndarray, faces: np.ndarray) -> dict:
    """Basic statistics of a triangle mesh (planar or surface): counts, angles (degrees), edge lengths, areas, aspect
    ratio ``R / (2 r)`` (1 = equilateral) of the triangles, Euler characteristic."""
    P = np.asarray(pts, dtype=np.float64)
    if P.shape[1] == 2:
        P = np.concatenate([P, np.zeros((len(P), 1))], 1)
    n = len(P)
    a, b, c = P[faces[:, 0]], P[faces[:, 1]], P[faces[:, 2]]

    def angle(u, v, w):
        e1, e2 = v - u, w - u
        cs = (e1 * e2).sum(1) / (np.linalg.norm(e1, axis=1) * np.linalg.norm(e2, axis=1))
        return np.degrees(np.arccos(np.clip(cs, -1.0, 1.0)))
    ang = np.stack([angle(a, b, c), angle(b, c, a), angle(c, a, b)], 1)
    edges = canonical_edges(faces, n)
    L = np.linalg.norm(P[edges[:, 1]] - P[edges[:, 0]], axis=1)
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    la, lb, lc = np.linalg.norm(b - c, axis=1), np.linalg.norm(c - a, axis=1), np.linalg.norm(a - b, axis=1)
    R = la * lb * lc / (4 * area)
    r = area / (0.5 * (la + lb + lc))
    asp = R / (2 * r)
    ev = P[edges[:, 1]] - P[edges[:, 0]]
    planar = np.asarray(pts).shape[1] == 2
    return {"n_nodes": int(n), "n_edges": int(len(edges)), "n_faces": int(len(faces)),
            "edge_median_abs_dx_over_dy": (float(np.median(np.abs(ev[:, 0])) / np.median(np.abs(ev[:, 1]))) if planar else None),
            "edge_rms_dx_over_dy": (float(np.sqrt((ev[:, 0] ** 2).mean() / (ev[:, 1] ** 2).mean())) if planar else None),
            "euler_characteristic": int(n - len(edges) + len(faces)),
            "min_angle_deg": float(ang.min()), "p1_min_angle_deg": float(np.percentile(ang.min(1), 1)),
            "median_min_angle_deg": float(np.median(ang.min(1))), "max_angle_deg": float(ang.max()),
            "frac_obtuse_faces": float((ang.max(1) > 90.0).mean()),
            "edge_len_min": float(L.min()), "edge_len_max": float(L.max()), "edge_len_mean": float(L.mean()),
            "edge_len_median": float(np.median(L)), "edge_len_max_over_min": float(L.max() / L.min()),
            "area_total": float(area.sum()), "face_area_min": float(area.min()), "face_area_max": float(area.max()),
            "face_area_mean": float(area.mean()), "aspect_median": float(np.median(asp)),
            "aspect_p99": float(np.percentile(asp, 99)), "aspect_max": float(asp.max())}


# ======================================================================================================================
# T6 / T7: the v1 generators (torch CUDA RNG) split into parameter draw and field evaluation
# ======================================================================================================================
class PlanarFields:
    """Random parameters of the T6 / T7 samples, drawn with the CUDA RNG exactly as ``gen_T6_native.py`` /
    ``gen_T7_native.py`` do (``torch.manual_seed(42)``, N = 10000 samples, same call sequence), and the evaluation of the
    fields of chosen samples on any planar mesh with the same elementwise float32 arithmetic.

    T6 (c = 1): ``theta = sum_5 d0[amp exp(-|x - c|^2 / 2 sigma^2)] + sum_{v < n_vort} s (-r_y dx + r_x dy) / (r^2 + 0.01)``
    (smooth pure-gauge bumps plus 1 to 3 softened point vortices, midpoint rule on the edges).
    T7 (c = 3): ``A^a = 0.1 N(0, 1) per edge + sum_4 d0[...] + sum_2 vortex terms`` for each Lie-algebra component a.

    The parameter draws of T6 do not depend on the mesh.  The T7 sequence starts with the white noise
    ``randn(N, n_edges, 3)`` of the ORIGINAL mesh, whose size fixes the RNG offset of all later draws; it is drawn here
    (``white0``, 366 MB) so that the parameters of the later draws are those of the original samples.  The CUDA kernel
    configuration of that large draw depends on the GPU (number of SMs); the original data were generated on the
    GPU recorded in ``datasets/v2/T7_native.json`` (an RTX PRO 6000 Blackwell), as this class is.
    """

    def __init__(self, task: str, device="cuda", n_edges_orig: int = 3050, keep_white: bool = True):
        assert task in ("T6", "T7")
        self.task, self.device = task, torch.device(device)
        assert self.device.type == "cuda", "the parameters of the original data come from the CUDA RNG stream"
        N, dev = N_ALL, self.device
        torch.manual_seed(42)
        if task == "T6":
            self.gauge = []
            for _ in range(5):
                cx = torch.rand(N, 1, device=dev)
                cy = torch.rand(N, 1, device=dev)
                sigma = torch.rand(N, 1, device=dev) * 0.3 + 0.1
                amp = torch.randn(N, 1, device=dev) * 1.0
                self.gauge.append((cx, cy, sigma, amp))
            n_vort = torch.randint(1, 4, (N,), device=dev)
            self.vort = []
            for v in range(3):
                mask = (n_vort > v).float()
                cx = torch.rand(N, 1, device=dev)
                cy = torch.rand(N, 1, device=dev)
                strength = (torch.randn(N, 1, device=dev) * 2.0) * mask.unsqueeze(-1)
                self.vort.append((cx, cy, strength))
            self.white0 = None
        else:
            w = torch.randn(N, n_edges_orig, 3, device=dev) * 0.1
            self.white0 = w if keep_white else None
            if not keep_white:
                del w
            self.gauge, self.vort = [], []
            for comp in range(3):
                for _ in range(4):
                    cx = torch.rand(N, 1, device=dev)
                    cy = torch.rand(N, 1, device=dev)
                    sigma = torch.rand(N, 1, device=dev) * 0.25 + 0.1
                    amp = torch.randn(N, 1, device=dev) * 0.5
                    self.gauge.append((comp, cx, cy, sigma, amp))
            for comp in range(3):
                for _ in range(2):
                    cx = torch.rand(N, 1, device=dev)
                    cy = torch.rand(N, 1, device=dev)
                    strength = torch.randn(N, 1, device=dev) * 0.3
                    self.vort.append((comp, cx, cy, strength))

    # ------------------------------------------------------------------------------------------------------------------
    def fields(self, idx, pts: np.ndarray, edges: np.ndarray, white=None) -> torch.Tensor:
        """Edge cochain of the samples ``idx`` on the mesh ``(pts, edges)``: ``(S, n_edges)`` (T6) or ``(S, n_edges, 3)``
        (T7) float32 on the generator's device, canonical orientation src < dst.  T7: ``white`` is the white-noise
        part ``(S, n_edges, 3)`` (default: the original draw ``white0[idx]``, valid on the original mesh only)."""
        dev = self.device
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=dev)
        S, n_edges = len(idx), len(edges)
        pts_t = torch.tensor(pts, dtype=torch.float32, device=dev)
        edges_t = torch.tensor(edges, device=dev)
        edge_dir = pts[edges[:, 1]] - pts[edges[:, 0]]
        dx_e = torch.tensor(edge_dir[:, 0], dtype=torch.float32, device=dev)
        dy_e = torch.tensor(edge_dir[:, 1], dtype=torch.float32, device=dev)
        if self.task == "T6":
            theta = torch.zeros(S, n_edges, device=dev)
            for cx, cy, sigma, amp in self.gauge:
                cx, cy, sigma, amp = cx[idx], cy[idx], sigma[idx], amp[idx]
                d2 = (pts_t[None, :, 0] - cx) ** 2 + (pts_t[None, :, 1] - cy) ** 2
                theta_node = amp * torch.exp(-d2 / (2 * sigma ** 2))
                theta += theta_node[:, edges_t[:, 1]] - theta_node[:, edges_t[:, 0]]
            mid_x = torch.tensor((pts[edges[:, 0], 0] + pts[edges[:, 1], 0]) / 2, dtype=torch.float32, device=dev)
            mid_y = torch.tensor((pts[edges[:, 0], 1] + pts[edges[:, 1], 1]) / 2, dtype=torch.float32, device=dev)
            for cx, cy, strength in self.vort:
                cx, cy, strength = cx[idx], cy[idx], strength[idx]
                rx = mid_x[None, :] - cx
                ry = mid_y[None, :] - cy
                r2 = rx ** 2 + ry ** 2 + 0.01
                theta += strength * (-ry * dx_e[None, :] + rx * dy_e[None, :]) / r2
                del rx, ry, r2
            return theta
        if white is None:
            if self.white0 is None or self.white0.shape[1] != n_edges:
                raise ValueError("T7: pass the white-noise part for a mesh other than the original")
            white = self.white0[idx]
        A = white.to(dev).clone()
        for comp, cx, cy, sigma, amp in self.gauge:
            cx, cy, sigma, amp = cx[idx], cy[idx], sigma[idx], amp[idx]
            d2 = (pts_t[None, :, 0] - cx) ** 2 + (pts_t[None, :, 1] - cy) ** 2
            theta = amp * torch.exp(-d2 / (2 * sigma ** 2))
            A[:, :, comp] += theta[:, edges_t[:, 1]] - theta[:, edges_t[:, 0]]
        edge_mid_x = (pts_t[edges_t[:, 0], 0] + pts_t[edges_t[:, 1], 0]) / 2
        edge_mid_y = (pts_t[edges_t[:, 0], 1] + pts_t[edges_t[:, 1], 1]) / 2
        for comp, cx, cy, strength in self.vort:
            cx, cy, strength = cx[idx], cy[idx], strength[idx]
            rx = edge_mid_x[None, :] - cx
            ry = edge_mid_y[None, :] - cy
            r2 = rx ** 2 + ry ** 2 + 0.01
            A[:, :, comp] += strength * (-ry * dx_e[None, :] + rx * dy_e[None, :]) / r2
        return A


def edge_face_data(faces: np.ndarray, n: int, device):
    """Canonical edges and oriented face -> edge incidence (``gen_common.edge_face_incidence``) as device tensors."""
    from gen_common import edge_face_incidence
    edges, fei, fes = edge_face_incidence(faces, n)
    return edges, torch.tensor(fei, device=device), torch.tensor(fes, device=device)


def face_values_T6(theta: torch.Tensor, fei: torch.Tensor, fes: torch.Tensor) -> torch.Tensor:
    """Plaquette flux ``d1 theta`` of every face in the face orientation (expression of ``gen_T6_native.py``)."""
    return (theta[:, fei[:, 0]] * fes[None, :, 0] + theta[:, fei[:, 1]] * fes[None, :, 1]
            + theta[:, fei[:, 2]] * fes[None, :, 2])


def face_values_T7(A: torch.Tensor, fei: torch.Tensor, fes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``F = dA + [A, A]`` of every face (expression of ``gen_T7_native.py``); also returns ``dA`` and ``[A, A]``."""
    A_f = [A[:, fei[:, k]] * fes[None, :, k:k + 1] for k in range(3)]
    dA = A_f[0] + A_f[1] + A_f[2]
    comm = (torch.cross(A_f[0], A_f[1], dim=-1) + torch.cross(A_f[1], A_f[2], dim=-1)
            + torch.cross(A_f[2], A_f[0], dim=-1))
    return dA + comm, dA, comm


# ======================================================================================================================
# v1 encodings (float64 recomputation, verified against the stored node encodings of the original data)
# ======================================================================================================================
def encode_edges_to_nodes(A: torch.Tensor, edges: np.ndarray, pts: np.ndarray, n: int) -> torch.Tensor:
    """Legacy node features of an edge cochain ``A (S, n1, c)`` (value on the canonical orientation src -> dst, src < dst):
    per node the mean over the incident edges of ``[A, A t_x, A t_y]``, ``t = (x_dst - x_src) / (|x_dst - x_src| + 1e-12)``
    -> ``(S, n, 3c)`` (blocks of ``c`` columns; isolated nodes: divisor 1).  Float64."""
    A = A.double().cpu()
    ev = pts[edges[:, 1]] - pts[edges[:, 0]]
    t = ev / (np.linalg.norm(ev, axis=1, keepdims=True) + EDGE_EPS)
    tx = torch.as_tensor(t[:, 0]).view(1, -1, 1)
    ty = torch.as_tensor(t[:, 1]).view(1, -1, 1)
    feats = torch.cat([A, A * tx, A * ty], dim=-1)
    src, dst = torch.as_tensor(edges[:, 0]), torch.as_tensor(edges[:, 1])
    out = torch.zeros(A.shape[0], n, feats.shape[-1], dtype=torch.float64)
    out.index_add_(1, src, feats)
    out.index_add_(1, dst, feats)
    cnt = torch.zeros(n, dtype=torch.float64)
    cnt.index_add_(0, src, torch.ones(len(src), dtype=torch.float64))
    cnt.index_add_(0, dst, torch.ones(len(dst), dtype=torch.float64))
    cnt[cnt == 0] = 1.0
    return out / cnt.view(1, -1, 1)


def faces_to_nodes(F: torch.Tensor, faces: np.ndarray, n: int) -> torch.Tensor:
    """Mean over the incident faces of a face cochain ``(S, n2, c)`` (v1 ``encode_faces_to_nodes``), float64."""
    F = F.double().cpu()
    out = torch.zeros(F.shape[0], n, F.shape[2], dtype=torch.float64)
    cnt = torch.zeros(n, dtype=torch.float64)
    for k in range(3):
        v = torch.as_tensor(faces[:, k])
        out.index_add_(1, v, F)
        cnt.index_add_(0, v, torch.ones(len(v), dtype=torch.float64))
    cnt[cnt == 0] = 1.0
    return out / cnt.view(1, -1, 1)


# ======================================================================================================================
# T3: ellipsoid surface flow
# ======================================================================================================================
T3_AXES = (1.4, 1.0, 0.7)       # original ellipsoid (a, b, c) before the normalisation to max |x| = 1
T3_RES = 35                      # vtkSphere theta / phi resolution of the original mesh (n_target 1200 -> 35)


def sphere_node_count(res: int) -> int:
    return res * (res - 2) + 2


def ellipsoid_mesh(axes=T3_AXES, res: int = T3_RES, scale: float | None = None):
    """Mesh of ``create_ellipsoid_mesh`` of the T3 generator: a ``pyvista.Sphere(radius=1, theta_resolution=res,
    phi_resolution=res)``, triangulated and cleaned, mapped to the semi-axes ``axes`` and divided by ``scale`` (default:
    the largest absolute coordinate of the mapped mesh, as the generator).  Returns ``(pts float64, faces int64,
    scale)``; ``pts.astype(float32)`` are the points stored in the pickle."""
    import pyvista as pv
    sph = pv.Sphere(radius=1.0, theta_resolution=res, phi_resolution=res).triangulate().clean()
    pts = sph.points.copy().astype(np.float64)
    pts[:, 0] *= axes[0]
    pts[:, 1] *= axes[1]
    pts[:, 2] *= axes[2]
    sc = float(np.max(np.abs(pts))) if scale is None else float(scale)
    pts /= sc
    faces = sph.faces.reshape((-1, 4))[:, 1:].astype(np.int64)
    return pts, faces, sc


def t3_draw_params(n_nodes: int = 1157, n_samples: int = N_ALL, seed: int = 42):
    """Parameters of the T3 samples, ``numpy.random.RandomState(seed)`` in the order of the generator: per sample the
    number of Gaussians ``randint(3, 7)``, then per Gaussian the centre node ``randint(0, n)``, ``sigma ~ U(0.15, 0.45)``,
    ``amp = U(1, 5) * (+-1)``.  Returns lists ``(node index, sigma, amp)`` per sample."""
    rng = np.random.RandomState(seed)
    out = []
    for _ in range(n_samples):
        nb = rng.randint(3, 7)
        src = []
        for _ in range(nb):
            ci = rng.randint(0, n_nodes)
            sig = rng.uniform(0.15, 0.45)
            amp = rng.uniform(1.0, 5.0) * rng.choice([-1, 1])
            src.append((int(ci), float(sig), float(amp)))
        out.append(src)
    return out


def t3_psi(src, pts: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """``psi = sum amp exp(-|x - c|^2 / (2 sigma^2))`` at the points ``pts`` for the sources ``src`` with centres
    ``centres[node index]`` (float64)."""
    psi = np.zeros(len(pts))
    for ci, sig, amp in src:
        c = centres[ci]
        psi += amp * np.exp(-np.linalg.norm(pts - c, axis=1) ** 2 / (2 * sig ** 2))
    return psi


class T3Target:
    """``v = n x grad(psi)`` at the vertices of a closed triangle surface, formulas of the T3 generator: per-face P1
    gradient ``(psi_i cr_kj + psi_j cr_ik + psi_k cr_ji) / (2 A)`` with ``cr = n_f x (opposite edge)``, arithmetic mean
    over the incident faces, ``v = n x g`` with the normalised sum of the face normals ``n`` at the vertex (each face
    normal weighted by its doubled area), tangential projection ``v -= (v.n) n``.  Float64."""

    def __init__(self, pts: np.ndarray, fcs: np.ndarray):
        self.n = len(pts)
        n = self.n
        normals = np.zeros((n, 3))
        nm = np.cross(pts[fcs[:, 1]] - pts[fcs[:, 0]], pts[fcs[:, 2]] - pts[fcs[:, 0]])
        for k in range(3):
            np.add.at(normals, fcs[:, k], nm)
        self.normals = normals / (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
        self.fi, self.fj, self.fk = fcs[:, 0], fcs[:, 1], fcs[:, 2]
        fn = np.cross(pts[self.fj] - pts[self.fi], pts[self.fk] - pts[self.fi])
        self.fa2 = np.linalg.norm(fn, axis=1, keepdims=True) + 1e-12
        fn = fn / self.fa2
        self.cr_kj = np.cross(fn, pts[self.fk] - pts[self.fj])
        self.cr_ik = np.cross(fn, pts[self.fi] - pts[self.fk])
        self.cr_ji = np.cross(fn, pts[self.fj] - pts[self.fi])
        nc = np.zeros(n)
        for f in (self.fi, self.fj, self.fk):
            np.add.at(nc, f, 1)
        nc[nc == 0] = 1
        self.nc = nc

    def __call__(self, psi: np.ndarray) -> np.ndarray:
        pi, pj, pk = psi[self.fi], psi[self.fj], psi[self.fk]
        gt = (pi[:, None] * self.cr_kj + pj[:, None] * self.cr_ik + pk[:, None] * self.cr_ji) / self.fa2
        grad = np.zeros((self.n, 3))
        for d in range(3):
            np.add.at(grad[:, d], self.fi, gt[:, d])
            np.add.at(grad[:, d], self.fj, gt[:, d])
            np.add.at(grad[:, d], self.fk, gt[:, d])
        grad /= self.nc[:, None]
        v = np.cross(self.normals, grad)
        v -= np.sum(v * self.normals, axis=1, keepdims=True) * self.normals
        return v


# ======================================================================================================================
# runs and evaluation (as ``scripts/frame_eval/frame_common.py``)
# ======================================================================================================================
def _empty():
    return torch.empty(0, dtype=torch.long)


def with_view(task, K, inputs, target, n: int, output_map="keep", **repl):
    """Test-only copy of ``task`` (``split = ([], [], arange(n))``) with new complex / inputs / targets."""
    om = task.output_map if isinstance(output_map, str) and output_map == "keep" else output_map
    return dataclasses.replace(task, K=K, inputs=inputs, target=target, split=(_empty(), _empty(), torch.arange(n)),
                               output_map=om, meta=dict(task.meta), **repl)


def load_task_for_run(conf: dict, device, extra_kw: dict | None = None):
    from rhmp.tasks import load_task
    a = conf["args"]
    kw = {"whitney": conf.get("model", {}).get("metric_type") == "tensor"}
    if a.get("abs_scale"):
        kw["abs_scale"] = True
    if a.get("no_output_map"):
        kw["output_map"] = False
    if extra_kw:
        kw.update(extra_kw)
    return load_task(a["task"], a.get("root"), native=a["native"], device=device, star=a.get("star", "cotan"), **kw)


def stats_match(task, conf) -> float:
    """Largest relative difference between the statistics of the loaded task and those saved in ``config.json``."""
    def rel(a, b_mean, b_std, is_std):
        a = a.cpu().double()
        b_mean = torch.tensor(b_mean, dtype=torch.float32).double()
        b_std = torch.tensor(b_std, dtype=torch.float32).double()
        if is_std:
            return float(((a - b_std).abs() / b_std).max())
        return float(((a - b_mean).abs() / torch.maximum(b_mean.abs(), b_std)).max())
    worst = 0.0
    xs = {int(k): v for k, v in conf["data"]["x_stats"].items()}
    for k, st in task.x_stats.items():
        worst = max(worst, rel(st.mean, xs[k]["mean"], xs[k]["std"], False), rel(st.std, xs[k]["mean"], xs[k]["std"], True))
    ys = conf["data"]["y_stats"]
    worst = max(worst, rel(task.y_stats.mean, ys["mean"], ys["std"], False), rel(task.y_stats.std, ys["mean"], ys["std"], True))
    return worst


@dataclasses.dataclass
class Run:
    task_name: str
    model_name: str
    run_dir: str
    conf: dict
    result: dict
    model: torch.nn.Module
    task: object                   # task as the run was trained / scored on (after prepare_task for baselines)
    task_raw: object               # task as loaded (before prepare_task)
    is_rhmp: bool
    ebs: int
    build: dict | None = None
    stats_rel_diff: float = 0.0

    @property
    def edge_protocol(self) -> bool:
        return bool(self.task.meta.get("v1_edge_protocol"))


def load_run(run_dir: str, task_name: str, model_name: str, device, task_cache: dict | None = None) -> Run:
    """Rebuild a finished run as ``rhmp.train.eval_ckpt`` does (same loader arguments as training)."""
    from rhmp.baselines.registry import SPECS, from_checkpoint, prepare_task
    from rhmp.model import RHMP
    conf = json.load(open(os.path.join(run_dir, "config.json")))
    result = json.load(open(os.path.join(run_dir, "result.json")))
    a = conf["args"]
    key = (a["task"], bool(a["native"]), str(device))
    if task_cache is not None and key in task_cache:
        task_raw = task_cache[key]
    else:
        task_raw = load_task_for_run(conf, device)
        if task_cache is not None:
            task_cache.clear()                     # keep one task in memory
            task_cache[key] = task_raw
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    trained = a.get("model") or "rhmp"
    is_rhmp = trained == "rhmp"
    task = task_raw
    if is_rhmp:
        model = RHMP.from_checkpoint(ck, map_location=device)
    else:
        task = prepare_task(trained, task_raw, ck.get("build"))
        model = from_checkpoint(ck, task, map_location=device)
        if not SPECS[trained].uses_output_map and task.output_map is not None:
            task = dataclasses.replace(task, output_map=None)
    model.eval()
    if hasattr(model, "record_diagnostics"):
        model.record_diagnostics = False
    diff = stats_match(task, conf)
    if diff > 1e-5:
        raise RuntimeError(f"{run_dir}: statistics of the loaded task differ from the run's by {diff:.3e} (relative)")
    ebs = a.get("eval_batch") or ((a.get("bs_meshes") or task.meta.get("batch_size", 8)) if task.variable_mesh
                                  else (a.get("batch") or task.meta.get("batch_size", 64)))
    return Run(task_name, trained, run_dir, conf, result, model, task, task_raw, is_rhmp, int(ebs),
               None if is_rhmp else ck.get("build"), diff)


@torch.no_grad()
def predict_view(run: Run, view, device, ebs: int | None = None):
    from rhmp.train import predict
    return predict(run.model, view, view.split[2], int(ebs or run.ebs), device)


def summarize_pred(view, pred, tgt) -> dict:
    from rhmp import metrics as M
    return M.summarize(pred, tgt, view.y_stats.as_tuple(), r2_std=view.meta.get("r2_std"), n_samples=None, prefix="",
                       center=view.target_kind != "cochain")


def eval_task_samples(run: Run, task, idx: torch.Tensor, device):
    """Library ``evaluate`` (as the trainer computes ``test100``)."""
    from rhmp.train import evaluate
    return evaluate(run.model, task, idx, run.ebs, device)


def raw_iso_r2(view, pred, tgt) -> float:
    """Isotropic R2 of node-vector targets in physical units, ``1 - sum |p - t|^2 / sum |t - mean_samples t|^2``."""
    st = view.y_stats
    p = st.denormalize(pred.double())
    t = st.denormalize(tgt.double())
    return float(1.0 - ((p - t) ** 2).sum() / ((t - t.mean(0, keepdim=True)) ** 2).sum())


KEYS = ("R2", "SSIM", "Pearson", "NRMSE", "MSE", "MAE", "R2_v1", "N")


def pick(m: dict) -> dict:
    return {k: (float(m[k]) if k != "N" else int(m[k])) for k in KEYS if k in m}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o
