"""Adapters for the original paper tasks T1, T2, T3, T5, T6, T7, T8 (+ T1q, T1p, T2v, T6f, T7f, T6_100K).

Legacy mode (``native=False``) reproduces v1's inputs/outputs and normalisation exactly (the v1 training script
``formal_benchmark.py``): node inputs ``X (N, n0, F)``, targets ``Y (N, n0, O)``, per-feature mean/std of the first
70 % (torch unbiased std; numpy biased std for T8), sequential 70/15/15 split.
Native mode puts inputs/outputs on the cells where the physics lives (see ``rhmp/tasks/__init__.py``).
"""
from __future__ import annotations

import json
import os
import pickle
import time
import warnings

import numpy as np
import torch
from torch import Tensor

from rhmp.data import (OutputMap, Stats, TaskData, cell_alignment, edge_alignment, feature_stats, holdout_split,
                       scale_stats, sequential_split)
from rhmp.tasks import TASK_DEFAULTS

V1_FILES = {
    "T1": ("T1_cns_vorticity.pkl", "T1_cns_vorticity"),
    "T2": ("T2_torus_advection_diffusion.pkl", "T2_torus_advection_diffusion"),
    "T3": ("T3_ellipsoid_coexact.pkl", "T3_ellipsoid_surface_flow"),
    "T5": ("T5_maxwell_poisson.pkl", "T5_maxwell_poisson"),
    "T6": ("T6_wilson_loop.pkl", "T6_wilson_loop"),
    "T7": ("T7_yang_mills_su2.pkl", "T7_yang_mills_su2"),
    "T8": ("T8_airfoil_pressure_persample.pkl", "T8_airfoil_pressure"),
    "T6_100K": ("T6_100K_wilson_loop.pkl", "T6_100K_wilson_loop"),
}
PAPER_EVAL_N = 100  # compute_all_metrics.py: first 100 samples of the test split


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def _read_pickle(droot: str, task: str) -> dict:
    path = os.path.join(droot, V1_FILES[task][0])
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} not found: download the v1 datasets with python3 datasets/download_v1.py "
                                f"(see datasets/README.md)")
    with open(path, "rb") as f:
        return pickle.load(f)


def _planar(points: np.ndarray) -> np.ndarray:
    """Drop an all-zero third coordinate (2-D meshes stored with z = 0)."""
    if points.shape[1] == 3 and np.abs(points[:, 2]).max() == 0.0:
        return points[:, :2]
    return points


def _xy(d: dict) -> tuple[Tensor, Tensor]:
    X = torch.as_tensor(np.asarray(d["X_data"], dtype=np.float32))
    Y = torch.as_tensor(np.asarray(d["Y_data"], dtype=np.float32))
    if X.dim() == 2:
        X = X.unsqueeze(-1)
    if Y.dim() == 2:
        Y = Y.unsqueeze(-1)
    return X, Y


def _complex(points: np.ndarray, faces: np.ndarray, star: str, device):
    from rhmp.complex import CochainComplex
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        K = CochainComplex.from_triangles(torch.as_tensor(points, dtype=torch.float64),
                                          torch.as_tensor(np.asarray(faces, dtype=np.int64)),
                                          star=star, device=device)
    if K.n[2] != len(faces):
        raise RuntimeError(f"complex dropped {len(faces) - K.n[2]} faces; face-indexed data would misalign")
    return K


def _paper_eval(N: int) -> Tensor:
    nt, nv = int(0.7 * N), int(0.15 * N)
    return torch.arange(nt + nv, min(nt + nv + PAPER_EVAL_N, N))


def _edge_one_form(V: Tensor, pos: Tensor, edges: Tensor, ev: Tensor | None = None) -> Tensor:
    """Midpoint-rule line integral of a node vector field along oriented edges.

    Args:
        V: ``(N, n0, D)`` node vectors; pos ``(n0, D)``; edges ``(n1, 2)`` (src, dst).
        ev: optional ``(n1, D)`` edge vectors ``x_dst - x_src`` (e.g. the minimal images of a periodic complex);
            default from ``pos``.
    Returns:
        ``(N, n1, 1)`` with ``V_e = (V_src + V_dst)/2 . (x_dst - x_src)``.
    """
    t = (pos[edges[:, 1]] - pos[edges[:, 0]] if ev is None else ev).to(V)
    Vm = 0.5 * (V[:, edges[:, 0]] + V[:, edges[:, 1]])
    return (Vm * t[None]).sum(-1, keepdim=True)


def _edge_half_difference(V: Tensor, pos: Tensor, edges: Tensor, ev: Tensor | None = None) -> Tensor:
    """Even companion of :func:`_edge_one_form`: ``(V_dst - V_src)/2 . (x_dst - x_src)``, ``(N, n1, 1)``.

    An edge flip swaps the endpoints and reverses the edge vector, so this column is unchanged (an even edge
    cochain).  Together with the midpoint one-form it keeps the tangential component of the node vectors at both
    endpoints of every edge; on a grid this is the whole node vector field.  ``ev`` as in :func:`_edge_one_form`.
    """
    t = (pos[edges[:, 1]] - pos[edges[:, 0]] if ev is None else ev).to(V)
    Vd = 0.5 * (V[:, edges[:, 1]] - V[:, edges[:, 0]])
    return (Vd * t[None]).sum(-1, keepdim=True)


def oriented_face_to_node(K, out_dim: int, area_ref: float | None = None) -> OutputMap:
    """Oriented face -> node average ``y_i = mean_{f ∋ i} sigma_f x_f`` for a planar 2-D complex.

    ``sigma_f = sign(signed area)`` (+1 for counter-clockwise faces): the physical orientation of the plane, which
    turns an odd face 2-cochain (flux / vorticity in the face's own orientation) into the node pseudo-scalar field of
    the v1 targets.  Model readout: ``cochain:2``.

    ``area_ref``: patch-area normalisation ``y_i = sum_{f ∋ i} sigma_f x_f / sum_{f ∋ i} (|f| / area_ref)``: the face
    values are read as integrals of a 2-form (e.g. circulations), and the node value is the integral over the patch of
    faces around the node divided by its area (in units of ``area_ref``), i.e. a density at the node.  On a uniform
    quad grid with cell area ``area_ref`` this is the mean; on graded grids it uses the true (minimal-image) areas.
    """
    from rhmp.ops import csr_from_coo
    F = K.cells[2].long().cpu()
    if (F < 0).any():
        raise ValueError("oriented_face_to_node: padded polygons not supported")
    if K.pos.shape[1] != 2:
        raise ValueError("oriented_face_to_node needs a planar complex with 2-D positions")
    Q = K.cell_positions(2).double().cpu()                      # (n2, m, 2) (unwrapped frames), shoelace signed area
    sa = 0.5 * (Q[..., 0] * Q.roll(-1, 1)[..., 1] - Q.roll(-1, 1)[..., 0] * Q[..., 1]).sum(1)
    sigma = torch.sign(sa)
    if (sigma == 0).any():
        raise ValueError("degenerate face in oriented_face_to_node")
    n0, n2, m = K.n[0], K.n[2], F.shape[1]
    rows = F.reshape(-1)
    cols = torch.arange(n2).repeat_interleave(m)
    if area_ref is None:
        cnt = torch.bincount(rows, minlength=n0).double()
        desc = "oriented face->node mean"
    else:
        cnt = torch.zeros(n0, dtype=torch.float64).index_add_(0, rows, (sa.abs() / float(area_ref)).repeat_interleave(m))
        desc = f"oriented face->node patch integral / patch area (area unit {area_ref:.6g})"
    vals = (sigma.repeat_interleave(m) / cnt[rows]).float()
    dev = K.device
    M = csr_from_coo(rows.to(dev), cols.to(dev), vals.to(dev), (n0, n2))
    MT = csr_from_coo(cols.to(dev), rows.to(dev), vals.to(dev), (n2, n0))
    ccw = int((sigma > 0).sum())
    return OutputMap("sparse", "cochain:2", out_dim, M=M, MT=MT,
                     description=f"{desc} (sigma=+1 for CCW; {ccw}/{n2} faces CCW)")


def direct_vector_map(K, model_readout: str = "grad") -> OutputMap:
    """Fixed edge -> vertex vector map ``v_i = sum_{e ∋ i} w_e t_e / deg_i`` (``t_e`` = unit vector src->dst, the same
    for both endpoints): the v1 edge-to-node average that *defines* the T5 targets (``w_e = -(d0 phi)_e``, the raw
    potential differences).  Composed with the ``grad`` readout (``w = d0 phi``) it represents the T5 targets exactly.
    """
    from rhmp.ops import csr_from_coo
    e = K.cells[1].long().cpu()
    ev = K.edge_vectors().double().cpu()
    t = ev / ev.norm(dim=1, keepdim=True)
    n0, n1, D = K.n[0], K.n[1], ev.shape[1]
    deg = torch.bincount(e.reshape(-1), minlength=n0).double().clamp(min=1)
    ends = torch.cat([e[:, 0], e[:, 1]])                               # vertex of each (edge, endpoint)
    eid = torch.arange(n1).repeat(2)
    rows = (ends[:, None] * D + torch.arange(D)[None, :]).reshape(-1)
    cols = eid[:, None].expand(-1, D).reshape(-1)
    vals = (t[eid] / deg[ends][:, None]).reshape(-1).float()
    dev = K.device
    M = csr_from_coo(rows.to(dev), cols.to(dev), vals.to(dev), (D * n0, n1))
    MT = csr_from_coo(cols.to(dev), rows.to(dev), vals.to(dev), (n1, D * n0))
    return OutputMap("direct_vector", model_readout, 1, M=M, MT=MT, D=D,
                     description="edge -> vertex vectors v_i = sum_e w_e t_e / deg_i (v1 average)")


def _t5_poisson_check(d: dict, K, omap: OutputMap, idx, n: int = 4) -> dict:
    """``omap(-d0 phi)`` vs the stored T5 target, ``phi`` = the generator's Poisson solve (cotan Laplacian with
    lumped areas, ``(L + 1e-6 I) phi = areas rho``, area-mean removed) of the stored ``rho``."""
    import scipy.sparse as sp
    from scipy.sparse.linalg import factorized
    P = np.asarray(d["points"], dtype=np.float64)[:, :2]
    Fc = np.asarray(d["faces"], dtype=np.int64)
    n0 = len(P)
    i, j, k = Fc[:, 0], Fc[:, 1], Fc[:, 2]
    eij, eki, ejk = P[j] - P[i], P[i] - P[k], P[k] - P[j]
    area = np.abs(eij[:, 0] * (-eki[:, 1]) - eij[:, 1] * (-eki[:, 0])) / 2
    ok = area >= 1e-12

    def cot2d(a, b):
        return (a[:, 0] * b[:, 0] + a[:, 1] * b[:, 1]) / (np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) + 1e-12)
    rows, cols, vals = [], [], []
    for a_, b_, cv in ((j, k, cot2d(eij, -eki)), (k, i, cot2d(ejk, -eij)), (i, j, cot2d(eki, -ejk))):
        a_, b_, w = a_[ok], b_[ok], cv[ok] / 2
        rows += [a_, b_, a_, b_]
        cols += [b_, a_, a_, b_]
        vals += [w, w, -w, -w]
    L = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n0, n0))
    areas = np.zeros(n0)
    for v in (i, j, k):
        np.add.at(areas, v[ok], area[ok] / 3)
    solve = factorized((L + 1e-6 * sp.eye(n0)).tocsc())
    e = K.cells[1].long().cpu().numpy()
    errs = []
    for s in list(idx)[:n]:
        rho = np.asarray(d["X_data"][s], dtype=np.float64).reshape(-1)
        phi = solve(areas * rho)
        phi -= np.sum(phi * areas) / np.sum(areas)
        w = torch.as_tensor(-(phi[e[:, 1]] - phi[e[:, 0]]), dtype=torch.float32)[:, None, None].to(K.device)
        v = omap(w)[:, 0].double().cpu().numpy()
        y = np.asarray(d["Y_data"][s], dtype=np.float64)
        errs.append(float(np.linalg.norm(v - y) / np.linalg.norm(y)))
    return {"max_rel_err_map_of_minus_d0_phi_vs_target": max(errs), "samples": len(errs)}


def surface_rotation(K, faces: np.ndarray, points: np.ndarray) -> OutputMap:
    """``v_i = n_i x g_i`` with the area-weighted vertex normals of the (consistently oriented) surface, exactly as
    the T3 generator builds its target ``v = n x grad psi``.  Model readout: ``node_vector`` (predicts ``g``)."""
    n = np.zeros_like(points)
    nm = np.cross(points[faces[:, 1]] - points[faces[:, 0]], points[faces[:, 2]] - points[faces[:, 0]])
    for k in range(3):
        np.add.at(n, faces[:, k], nm)
    n /= (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
    return OutputMap("cross_normal", "node_vector", 1, normals=torch.as_tensor(n, dtype=torch.float32).to(K.device),
                     description="v = n x g (rotation by +90 deg about the outward vertex normal)")


def stream_function_map(K, faces: np.ndarray, points: np.ndarray, model_readout: str = "grad") -> OutputMap:
    """Fixed edge -> vertex vector map of the T3 generator, ``v_i = n_i x mean_{f ∋ i} grad_f``.

    ``grad_f = (w_ij cr_ik + w_ik cr_ji) / (2 A_f)`` is the P1 gradient of face ``(i, j, k)`` written with the edge
    differences ``w_ab = phi_b - phi_a`` (``cr = n_f x`` opposite edge), the mean is the arithmetic mean over the
    incident faces and ``n_i`` the area-weighted vertex normal.  Composed with the ``grad`` readout (``w = d0 phi``) it
    represents the T3 targets exactly (``phi = psi``), on every mesh and in physical units.
    """
    from rhmp.ops import csr_from_coo
    P = np.asarray(points, dtype=np.float64)
    Fc = np.asarray(faces, dtype=np.int64)
    n0, n1 = int(K.n[0]), int(K.n[1])
    e = K.cells[1].long().cpu().numpy()
    key = {}
    for eid, (a, b) in enumerate(e):
        key[(int(a), int(b))] = (eid, 1.0)
        key[(int(b), int(a))] = (eid, -1.0)
    fi, fj, fk = Fc[:, 0], Fc[:, 1], Fc[:, 2]
    nm = np.cross(P[fj] - P[fi], P[fk] - P[fi])
    normals = np.zeros((n0, 3))
    for c in range(3):
        np.add.at(normals, Fc[:, c], nm)
    normals /= (np.linalg.norm(normals, axis=1, keepdims=True) + 1e-9)
    fa2 = np.linalg.norm(nm, axis=1, keepdims=True) + 1e-12
    fn = nm / fa2
    cr_ik = np.cross(fn, P[fi] - P[fk]) / fa2                       # coefficient of w_ij
    cr_ji = np.cross(fn, P[fj] - P[fi]) / fa2                       # coefficient of w_ik
    nc = np.bincount(Fc.reshape(-1), minlength=n0).astype(np.float64)
    nc[nc == 0] = 1.0
    rows, cols, vals = [], [], []
    for f in range(len(Fc)):
        i, j, k = int(fi[f]), int(fj[f]), int(fk[f])
        for (a, b), g in (((i, j), cr_ik[f]), ((i, k), cr_ji[f])):
            eid, sgn = key[(a, b)]
            for v in (i, j, k):
                u = np.cross(normals[v], g) * (sgn / nc[v])
                rows += [3 * v, 3 * v + 1, 3 * v + 2]
                cols += [eid] * 3
                vals += [u[0], u[1], u[2]]
    dev = K.device
    r, c = torch.as_tensor(rows), torch.as_tensor(cols)
    w = torch.as_tensor(vals, dtype=torch.float32)
    M = csr_from_coo(r.to(dev), c.to(dev), w.to(dev), (3 * n0, n1))
    MT = csr_from_coo(c.to(dev), r.to(dev), w.to(dev), (n1, 3 * n0))
    return OutputMap("direct_vector", model_readout, 1, M=M, MT=MT, D=3,
                     description="edge -> vertex vectors v_i = n_i x mean of the P1 face gradients (T3 generator)")


def _finish(name, K, inputs_raw: dict, target_raw: Tensor, *, target_degree, target_kind, in_dims, even_dims,
            connection_dims, x_kinds: dict, y_kind: str, spatial_dim, native, device, meta, split=None,
            unbiased=True, y_v1: Tensor | None = None, output_map: OutputMap | None = None) -> TaskData:
    """Normalise with train-part statistics and move everything to ``device``.

    ``x_kinds[k]`` / ``y_kind``: ``'v1'`` (mean/std), ``'scale'`` (odd cochains: mean 0, RMS per column),
    ``'isotropic'`` (vector fields: mean 0, one RMS for all components), ``'scale:<n>'`` (``'scale'`` for the first
    ``n`` columns, ``'v1'`` for the others).
    """
    N = target_raw.shape[0]
    split = split or sequential_split(N)
    tr = split[0]

    def stats(x, kind):
        if kind == "v1":
            return feature_stats(x[tr], unbiased=unbiased)
        if kind == "scale":
            return scale_stats(x[tr])
        if kind == "isotropic":
            return scale_stats(x[tr], isotropic=True)
        if kind.startswith("scale:"):
            n = int(kind.split(":")[1])
            a, b = scale_stats(x[tr][..., :n]), feature_stats(x[tr][..., n:], unbiased=unbiased)
            return Stats(torch.cat([a.mean, b.mean]), torch.cat([a.std, b.std]), kind)
        raise ValueError(kind)

    x_stats = {k: stats(v, x_kinds[k]) for k, v in inputs_raw.items()}
    y_stats = stats(target_raw, y_kind)
    inputs = {k: x_stats[k].normalize(v).to(device).contiguous() for k, v in inputs_raw.items()}
    target = y_stats.normalize(target_raw).to(device).contiguous()
    meta = dict(meta)
    v1_std = feature_stats((y_v1 if y_v1 is not None else target_raw)[tr], unbiased=unbiased).std
    meta.setdefault("r2_std", v1_std)
    meta.setdefault("paper_eval", _paper_eval(N))
    return TaskData(name=name, K=K, inputs=inputs, target=target, target_degree=target_degree,
                    target_kind=target_kind, in_dims=in_dims, even_dims=even_dims, connection_dims=connection_dims,
                    split=split, x_stats={k: s.to(device) for k, s in x_stats.items()}, y_stats=y_stats.to(device),
                    spatial_dim=spatial_dim, out_dim=int(target.shape[-1]), native=native, meta=meta,
                    output_map=output_map)


def _defaults(name: str) -> dict:
    d = TASK_DEFAULTS[name]
    return dict(v1_C=d["C"], batch_size=d["batch"], layers=d["layers"])


# ----------------------------------------------------------------------------------------------------------------
# T1 / T1q: CNS vorticity on the 32x32 grid
# ----------------------------------------------------------------------------------------------------------------
def load_T1(droot, *, native, device, star, quad=False, output_map=None, edge_even=False, node_readout=False,
            name=None, **_):
    """T1 CNS vorticity (32x32 grid; ``quad=True``: quad CW complex T1q).  Legacy: (rho,Vx,Vy,p) nodes -> vorticity
    nodes.  Native: (rho,p) nodes + velocity 1-form on edges; ``cochain:2`` readout + oriented face->node map.
    ``edge_even=True`` (task ``T1qe``): the even half-difference column of :func:`_edge_half_difference` is added to
    the edge inputs, so that the edge encoding of the node velocities loses nothing on the grid.
    ``node_readout=True`` (task ``T1qn``): the orientation enters on the input side.  The circulation of the velocity
    one-form around every face, divided by the signed face area (the mean vorticity of the face, independent of the
    face orientation), is an even face input, and the vorticity is read out at the vertices (``node_scalar``); no
    output map, so the readout is not restricted to face averages."""
    from rhmp.complex import CochainComplex
    d = _read_pickle(droot, "T1")
    pts = np.asarray(d["points"], dtype=np.float64)
    X, Y = _xy(d)
    gs = int(d.get("grid_size", 32))
    name = name or ("T1q" if quad else "T1")
    if quad:
        K = CochainComplex.from_grid((gs, gs), 1.0 / (gs - 1), cell="quad", star=star, device=device)
        # vertex id i*ny+j at (i h, j h) is the v1 ordering (meshgrid indexing='ij'); checked here
        if not torch.allclose(K.pos.double().cpu(), torch.as_tensor(pts), atol=1e-6):
            raise RuntimeError("from_grid vertex order does not match the T1 points")
    else:
        K = _complex(pts, d["faces"], star, device)
    meta = _defaults(name)
    meta["v1_name"] = V1_FILES["T1"][1]
    if not native:
        return _finish(name, K, {0: X}, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 4},
                       even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=2,
                       native=False, device=device, meta=meta)
    edges = K.cells[1].cpu()
    pts32 = torch.as_tensor(pts, dtype=torch.float32)
    V1form = _edge_one_form(X[..., 1:3], pts32, edges)
    meta["native_inputs"] = "node (rho, p); edge velocity 1-form (V_s+V_d)/2.(x_d-x_s)"
    e_dims, e_even = 1, {}
    if edge_even:
        if not native:
            raise ValueError("T1qe has no legacy variant (the even edge column is a native cochain input)")
        V1form = torch.cat([V1form, _edge_half_difference(X[..., 1:3], pts32, edges)], dim=-1)
        meta["native_inputs"] += "; even edge column (V_d-V_s)/2.(x_d-x_s)"
        e_dims, e_even = 2, {1: 1}
    if node_readout:
        d1 = K.d[1].to_dense().cpu().to(V1form.dtype)                               # (n2, n1)
        Q = torch.as_tensor(pts, dtype=torch.float64)[K.cells[2].long().cpu()]     # (n2, m, 2), shoelace signed area
        sa = 0.5 * (Q[..., 0] * Q.roll(-1, 1)[..., 1] - Q.roll(-1, 1)[..., 0] * Q[..., 1]).sum(1)
        if (sa == 0).any():
            raise ValueError("T1qn: degenerate face")
        W = ((V1form[..., 0] @ d1.T) / sa.to(V1form.dtype)).unsqueeze(-1)          # (N, n2, 1), even
        meta["native_inputs"] += "; even face column (d1 theta)_f / signed area"
        return _finish(name, K, {0: X[..., [0, 3]].contiguous(), 1: V1form, 2: W}, Y, target_degree=0,
                       target_kind="node_scalar", in_dims={0: 2, 1: e_dims, 2: 1}, even_dims={**e_even, 2: 1},
                       connection_dims={}, x_kinds={0: "v1", 1: "scale", 2: "v1"}, y_kind="v1", spatial_dim=2,
                       native=True, device=device, meta=meta)
    omap = oriented_face_to_node(K, 1) if output_map is not False else None     # vorticity = pseudo-scalar
    return _finish(name, K, {0: X[..., [0, 3]].contiguous(), 1: V1form}, Y, target_degree=0,
                   target_kind="node_scalar", in_dims={0: 2, 1: e_dims}, even_dims=e_even, connection_dims={},
                   x_kinds={0: "v1", 1: "scale"}, y_kind="v1", spatial_dim=2, native=True, device=device, meta=meta,
                   output_map=omap)


# ----------------------------------------------------------------------------------------------------------------
# T1p / T1pm: CNS vorticity on the periodic 32x32 grid (flat torus)
# ----------------------------------------------------------------------------------------------------------------
def load_T1p(droot, *, native, device, star, output_map=None, name="T1p", diagonal="main", cell_fields=False,
             filtered=False, vertex_logs=False, **_):
    """T1 on its periodic grid.  The PDEBench simulation (``2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic``) and the
    target (central differences with periodic wrap-around) are periodic, so the mesh is the flat torus
    ``CochainComplex.from_grid((32, 32), 1/32, periodic=True)`` (spacing 1/32 of the unit box, vertex order
    ``field[ix, iy].reshape(-1)`` as T1; wrap-around edges and faces, all cells congruent, no boundary).  Data,
    split, normalisation and target are those of T1.

    Native (the RHMP form): periodic quad complex with the T1qe inputs: vertex (rho, p); on every edge (wrap-around
    edges included, ``e`` the minimal-image edge vector) the velocity 1-form ``(V_s + V_d)/2 . e`` and its even
    half-difference ``(V_d - V_s)/2 . e``; readout ``cochain:2`` + oriented face->node mean.  ``cell_fields=True``
    (task ``T1pm``) adds ``log rho`` and ``log p`` averaged over the vertices of every edge and face as even columns
    (orientation-free material-like coefficients that the metric heads of degrees 1 and 2 see); ``vertex_logs=True``
    (task ``T1pml``) also appends ``log rho``, ``log p`` to the vertex inputs (legacy ``T1pml``: the v1 node inputs
    plus the same two log columns, a fairness control for the node baselines).
    Legacy (the baseline form): (rho, Vx, Vy, p) at the vertices of the periodic triangulated grid (``diagonal``,
    default ``'main'``: every square split along v00-v11, so all vertices are equivalent and cyclic shifts of the
    grid are automorphisms of the mesh) -> vorticity at the vertices.
    ``filtered=True`` (tasks ``T1pf`` / ``T1pmf``): the same trajectories coarse-grained by 4 x 4 block averages
    instead of strided subsampling (``datasets/v2/T1_cns_vorticity_blockavg.pkl``, generator
    ``datasets/generators/gen_T1_blockavg.py``; target: the same stencil on the filtered velocity).
    """
    from rhmp.complex import CochainComplex
    if filtered:
        path = os.path.join(droot, "v2", "T1_cns_vorticity_blockavg.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run python3 datasets/generators/gen_T1_blockavg.py "
                                    f"--src <PDEBench 2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5>")
        with open(path, "rb") as f:
            d = pickle.load(f)
    else:
        d = _read_pickle(droot, "T1")
    pts = np.asarray(d["points"], dtype=np.float64)
    X, Y = _xy(d)
    gs = int(d.get("grid_size", 32))
    cell = "quad" if native else "tri"
    K = CochainComplex.from_grid((gs, gs), 1.0 / gs, cell=cell, star=star, diagonal=None if native else diagonal,
                                 device=device, periodic=True)
    # vertex i*ny+j at (i h, j h): the T1 points are (i, j) / (gs - 1) on the non-periodic unit square
    if not torch.allclose(K.pos.double().cpu() * gs / (gs - 1), torch.as_tensor(pts), atol=1e-6):
        raise RuntimeError("periodic grid vertex order does not match the T1 points")
    meta = _defaults(name)
    meta["v1_name"] = V1_FILES["T1"][1]
    meta["mesh"] = (f"periodic {gs}x{gs} grid (flat torus, spacing 1/{gs}), "
                    + ("quads" if native else f"triangles (diagonal {diagonal!r})"))
    if not native:
        if cell_fields and not vertex_logs:
            raise ValueError(f"{name} has no legacy variant; the baselines use the legacy T1p")
        X0 = X
        if vertex_logs:                     # fairness control: the node baselines get the vertex logs of T1pml too
            X0 = torch.cat([X, torch.log(X[..., [0, 3]].double()).float()], -1)
            meta["legacy_inputs"] = "v1 node inputs (rho, Vx, Vy, p) + (log rho, log p)"
        return _finish(name, K, {0: X0.contiguous()}, Y, target_degree=0, target_kind="node_scalar",
                       in_dims={0: int(X0.shape[-1])}, even_dims={}, connection_dims={}, x_kinds={0: "v1"},
                       y_kind="v1", spatial_dim=2, native=False, device=device, meta=meta)
    edges = K.cells[1].cpu()
    ev = K.edge_vectors().cpu()
    pts32 = K.pos.float().cpu()
    E1 = torch.cat([_edge_one_form(X[..., 1:3], pts32, edges, ev),
                    _edge_half_difference(X[..., 1:3], pts32, edges, ev)], dim=-1)
    meta["native_inputs"] = ("node (rho, p); edge velocity 1-form (V_s+V_d)/2.e and even column (V_d-V_s)/2.e "
                             "(e = minimal-image edge vector)")
    inputs, in_dims, even_dims, kinds = {0: X[..., [0, 3]].contiguous(), 1: E1}, {0: 2, 1: 2}, {1: 1}, \
        {0: "v1", 1: "scale"}
    if cell_fields:
        L = torch.log(X[..., [0, 3]].double())                                          # (N, n0, 2) log rho, log p
        F2 = K.cells[2].long().cpu()
        Le = 0.5 * (L[:, edges[:, 0]] + L[:, edges[:, 1]])
        Lf = L[:, F2].mean(2)
        inputs[1] = torch.cat([E1, Le.float()], -1)
        inputs[2] = Lf.float().contiguous()
        in_dims, even_dims = {0: 2, 1: 4, 2: 2}, {1: 3, 2: 2}
        kinds = {0: "v1", 1: "scale:2", 2: "v1"}
        meta["native_inputs"] += "; even edge and face columns: vertex means of (log rho, log p)"
        if vertex_logs:
            inputs[0] = torch.cat([inputs[0], L.float()], -1).contiguous()
            in_dims[0] = 4
            meta["native_inputs"] += "; vertex (log rho, log p)"
    omap = oriented_face_to_node(K, 1) if output_map is not False else None     # vorticity = pseudo-scalar
    return _finish(name, K, inputs, Y, target_degree=0, target_kind="node_scalar", in_dims=in_dims,
                   even_dims=even_dims, connection_dims={}, x_kinds=kinds, y_kind="v1", spatial_dim=2, native=True,
                   device=device, meta=meta, output_map=omap)


# ----------------------------------------------------------------------------------------------------------------
# T1g: CNS vorticity on a graded, anisotropic periodic grid (+ zero-shot grids)
# ----------------------------------------------------------------------------------------------------------------
T1G_FILE = os.path.join("v2", "T1_cns_vorticity_graded.pkl")
T1G_ZERO_SHOT = ("U32", "G2", "U64", "G1rot")
T1G_AREA_REF = (1.0 / 32) ** 2          # cell area of the uniform 32 x 32 grid of the unit torus
_T1G_CACHE: dict = {}


def _t1g_data(droot: str) -> dict:
    path = os.path.join(droot, T1G_FILE)
    if path not in _T1G_CACHE:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run python3 datasets/generators/gen_T1_graded.py --src <PDEBench "
                                    f"2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5>")
        with open(path, "rb") as f:
            _T1G_CACHE.clear()
            _T1G_CACHE[path] = pickle.load(f)
    return _T1G_CACHE[path]


def _t1g_complex(g: dict, native: bool, star: str, device):
    from rhmp.complex import CochainComplex
    if native:
        return CochainComplex.from_polygons(g["points"], g["quads"], star=star, device=device, period=(1.0, 1.0))
    return CochainComplex.from_triangles(g["points"], g["triangles"], star=star, device=device, period=(1.0, 1.0))


def _t1g_abs_geometry(K, h: float) -> None:
    """In place: the scale-dependent descriptors of a planar complex (vertex dual area and mean edge length, edge length
    and star1, face area and star2) as ``log(x / x_ref)`` with the values of the uniform square grid of spacing ``h``
    (``h^2, h; h, 1; h^2, 1/h^2``) instead of ``log(x / median over this mesh)``: one length unit for every mesh of the
    task (the descriptors of the other columns are scale-free and unchanged)."""
    ell = K.edge_vectors().double().norm(dim=1)
    e = K.cells[1].long()
    tot = torch.zeros(K.n[0], dtype=ell.dtype, device=ell.device).index_add_(0, e[:, 0], ell).index_add_(0, e[:, 1], ell)
    cnt = torch.zeros_like(tot).index_add_(0, e[:, 0], torch.ones_like(ell)).index_add_(0, e[:, 1], torch.ones_like(ell))
    st = [v.double() for v in K.star]
    vals = {0: (st[0] / h ** 2, tot / cnt.clamp_min(1) / h), 1: (ell / h, st[1]), 2: (1.0 / (st[2] * h ** 2), st[2] * h ** 2)}
    for k, (a, b) in vals.items():
        if K.meta["geo_features"][k][:2] not in (("log_dual_measure", "log_mean_edge_length"), ("log_length", "log_star"),
                                                 ("log_area", "log_star")):
            raise RuntimeError(f"unexpected descriptor columns of degree {k}: {K.meta['geo_features'][k]}")
        g = K.geo[k].clone()
        g[:, 0], g[:, 1] = torch.log(a).to(g), torch.log(b).to(g)
        K.geo[k] = g
    K.meta["geo_reference"] = f"log(x / x_ref) of the uniform grid with spacing {h:.6g} (columns 0, 1)"


def _t1g_native_inputs(X: Tensor, K, cell_fields: bool, vertex_logs: bool) -> dict:
    """Native T1p / T1pm / T1pml inputs of node data ``X (N, n0, 4)`` on the complex ``K`` (minimal-image edges)."""
    edges, ev, pos = K.cells[1].cpu(), K.edge_vectors().cpu(), K.pos.float().cpu()
    E1 = torch.cat([_edge_one_form(X[..., 1:3], pos, edges, ev), _edge_half_difference(X[..., 1:3], pos, edges, ev)],
                   dim=-1)
    inputs = {0: X[..., [0, 3]].contiguous(), 1: E1}
    if cell_fields:
        L = torch.log(X[..., [0, 3]].double())
        F2 = K.cells[2].long().cpu()
        inputs[1] = torch.cat([E1, (0.5 * (L[:, edges[:, 0]] + L[:, edges[:, 1]])).float()], -1)
        inputs[2] = L[:, F2].mean(2).float().contiguous()
        if vertex_logs:
            inputs[0] = torch.cat([inputs[0], L.float()], -1).contiguous()
    return inputs


def load_T1g(droot, *, native, device, star, output_map=None, name="T1g", cell_fields=False, vertex_logs=False,
             area_map=True, abs_geo=False, **_):
    """T1 on a graded, anisotropic periodic grid (``datasets/v2/T1_cns_vorticity_graded.pkl``, generator
    ``datasets/generators/gen_T1_graded.py``): the T1 trajectories sampled exactly at the 32 x 32 nodes of the
    training grid ``G1`` (spacings of 1, 2, 4 and 8 cells of the 128 x 128 simulation grid, neighbour ratio <= 2, cell
    aspect ratios 1/8 ... 8), target = the vorticity at ``t + 1`` from the fine grid (spectral curl) at the same nodes.
    Periodic complexes with the true (minimal-image) geometry; split and protocol of T1.

    Native (RHMP): periodic quads with the T1p inputs (``cell_fields`` / ``vertex_logs``: T1pm / T1pml); readout
    ``cochain:2`` + oriented face->node map, by default (``area_map``) the patch-area normalised map of
    :func:`oriented_face_to_node` with ``area_ref = 1/32^2`` (the face values act as circulations; the mean map of
    T1p is the same map on the uniform grid); ``area_map=False``: the plain mean (ablation).  Legacy (baselines): node
    inputs ``(rho, Vx, Vy, p)`` (``vertex_logs``: + ``log rho, log p``) on the periodic graded grid triangulated along
    the ``(i, j)-(i+1, j+1)`` diagonals -> node vorticity.

    ``extra_tests['zs_<grid>']`` (zero-shot, scored at the end of training): the 100 paper-evaluation samples
    (``8500 .. 8599``) on the grids ``U32`` (uniform 32 x 32), ``G2`` (two fine clusters per axis), ``U64`` (uniform
    64 x 64) and ``G1rot`` (``G1`` rotated by 90 degrees with rotated velocities), each with its own complex, inputs
    built the same way, the training statistics and (native) its own output map.

    ``abs_geo`` (task ``T1gmla``, native): the scale-dependent geometry descriptors of every complex in one fixed length
    unit (:func:`_t1g_abs_geometry`, reference = the uniform 32 x 32 grid) instead of relative to the mesh's own median
    (the default of :mod:`rhmp.geometry`, which makes the descriptors of meshes with different median cell sizes, e.g.
    G1 (median edge 2 fine cells) and U32 (4), disagree about the same absolute cell size).
    """
    d = _t1g_data(droot)
    grids = d["grids"]
    X = torch.as_tensor(d["train"]["X_data"])
    Y = torch.as_tensor(d["train"]["Y_data"])
    K = _t1g_complex(grids["G1"], native, star, device)
    if abs_geo:
        if not native:
            raise ValueError(f"{name}: absolute descriptors are a native (RHMP) option")
        _t1g_abs_geometry(K, 1.0 / 32)
    meta = _defaults(name)
    meta["v1_name"] = "cns_vorticity_graded"
    meta["mesh"] = "periodic graded 32x32 grid G1 (" + ("quads" if native else "triangles, (i,j)-(i+1,j+1) diagonals") + ")"
    meta["target_definition"] = d["target"]
    meta["grids"] = {k: dict(description=g["description"], sx=[int(v) for v in g["sx"]], sy=[int(v) for v in g["sy"]],
                             n_nodes=int(len(g["points"]))) for k, g in grids.items()}
    meta["node_spacing"] = torch.as_tensor(grids["G1"]["node_spacing"], dtype=torch.float32)
    if not native:
        def legacy_inputs(Xr):
            if vertex_logs:
                return torch.cat([Xr, torch.log(Xr[..., [0, 3]].double()).float()], -1).contiguous()
            return Xr.contiguous()
        X0 = legacy_inputs(X)
        if vertex_logs:
            meta["legacy_inputs"] = "v1 node inputs (rho, Vx, Vy, p) + (log rho, log p)"
        td = _finish(name, K, {0: X0}, Y, target_degree=0, target_kind="node_scalar", in_dims={0: int(X0.shape[-1])},
                     even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=2, native=False,
                     device=device, meta=meta)
    else:
        inputs = _t1g_native_inputs(X, K, cell_fields, vertex_logs)
        in_dims = {k: int(v.shape[-1]) for k, v in inputs.items()}
        even_dims = {1: 3, 2: 2} if cell_fields else {1: 1}
        kinds = {0: "v1", 1: "scale:2" if cell_fields else "scale", 2: "v1"}
        meta["native_inputs"] = ("node (rho, p" + (", log rho, log p" if vertex_logs else "") + "); edge velocity "
                                 "1-form (V_s+V_d)/2.e and even column (V_d-V_s)/2.e (minimal-image e)"
                                 + ("; even edge and face columns: vertex means of (log rho, log p)" if cell_fields
                                    else ""))
        omap = oriented_face_to_node(K, 1, T1G_AREA_REF if area_map else None) if output_map is not False else None
        td = _finish(name, K, inputs, Y, target_degree=0, target_kind="node_scalar", in_dims=in_dims,
                     even_dims=even_dims, connection_dims={}, x_kinds={k: kinds[k] for k in inputs}, y_kind="v1",
                     spatial_dim=2, native=True, device=device, meta=meta, output_map=omap)
    # zero-shot grids: the paper-evaluation samples, normalised with the training statistics
    pe = td.meta["paper_eval"].cpu().numpy()
    for gname in T1G_ZERO_SHOT:
        z = d["zero_shot"][gname]
        pos = {int(s): j for j, s in enumerate(np.asarray(z["sample_ids"]).tolist())}
        sel = torch.as_tensor([pos[int(s)] for s in pe])
        Xz = torch.as_tensor(z["X_data"])[sel]
        Yz = torch.as_tensor(z["Y_data"])[sel]
        Kz = _t1g_complex(grids[gname], native, star, device)
        if abs_geo:
            _t1g_abs_geometry(Kz, 1.0 / 32)
        if native:
            raw = _t1g_native_inputs(Xz, Kz, cell_fields, vertex_logs)
            om = oriented_face_to_node(Kz, 1, T1G_AREA_REF if area_map else None) if output_map is not False else None
        else:
            raw, om = {0: legacy_inputs(Xz)}, None
        td.extra_tests[f"zs_{gname}"] = dict(
            K=Kz, inputs={k: td.x_stats[k].normalize(v.to(device)).contiguous() for k, v in raw.items()},
            target=td.y_stats.normalize(Yz.to(device)).contiguous(), output_map=om, grid=gname,
            sample_ids=torch.as_tensor(pe), node_spacing=torch.as_tensor(grids[gname]["node_spacing"],
                                                                          dtype=torch.float32))
    return td


# ----------------------------------------------------------------------------------------------------------------
# T2: torus advection-diffusion (scalar -> scalar)
# ----------------------------------------------------------------------------------------------------------------
def load_T2(droot, *, native, device, star, **_):
    """T2 torus advection-diffusion: scalar ``(N, 1711, 1)`` -> scalar at nodes (legacy == native)."""
    d = _read_pickle(droot, "T2")
    pts = np.asarray(d["points"], dtype=np.float64)
    X, Y = _xy(d)
    K = _complex(pts, d["faces"], star, device)
    meta = _defaults("T2")
    meta["v1_name"] = V1_FILES["T2"][1]
    return _finish("T2", K, {0: X}, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 1}, even_dims={},
                   connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=3, native=native,
                   device=device, meta=meta)


# ----------------------------------------------------------------------------------------------------------------
# T2v: torus advection-diffusion with the advecting velocity as an input (well-posed T2)
# ----------------------------------------------------------------------------------------------------------------
def t2v_file(droot: str) -> dict:
    """``datasets/v2/T2v.pt`` (``datasets/generators/gen_T2v.py``): per-sample velocities ``V (N, n0, 3)``, the
    coefficients ``alpha, beta``, the basis fields ``vt, vp`` and the extra evaluation sets (``alpha_sweep``,
    ``rollout``) of the test split."""
    path = os.path.join(droot, "v2", "T2v.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run python3 -u datasets/generators/gen_T2v.py (see "
                                f"datasets/README.md)")
    return torch.load(path, map_location="cpu", weights_only=False)


def t2v_inputs(X: Tensor, V: Tensor, K, native: bool) -> dict[int, Tensor]:
    """Raw (unnormalised) T2v inputs from the scalar ``X (N, n0, 1)`` and the node velocities ``V (N, n0, 3)``.

    Native: ``{0: X, 1: [theta, delta]}`` with the velocity one-form ``theta_e = (V_s + V_d)/2 . (x_d - x_s)`` (odd)
    and the half-difference ``delta_e = (V_d - V_s)/2 . (x_d - x_s)`` (even, :func:`_edge_half_difference`): together
    they give the tangential components ``V_s . t_e = theta - delta`` and ``V_d . t_e = theta + delta`` at both ends
    of every edge, from which the tangent velocity of every vertex is recovered exactly (least squares over its
    edges).  Legacy (node features of the baselines): ``{0: (f, Vx, Vy, Vz)}``.
    """
    if not native:
        return {0: torch.cat([X, V.to(X.dtype)], dim=-1)}
    edges = K.cells[1].cpu()
    pos = K.pos.detach().double().cpu()
    Vd = V.double()
    E = torch.cat([_edge_one_form(Vd, pos, edges), _edge_half_difference(Vd, pos, edges)], dim=-1).to(X.dtype)
    return {0: X, 1: E}


def _t2v_reconstruction_error(V: Tensor, K, n: int = 8) -> float:
    """Max relative error of the vertex velocities recovered from the native edge encoding (first ``n`` samples):
    least squares ``min_v sum_{e ∋ i} (v . t_e - p_{i,e})^2`` in the plane orthogonal to the vertex normal."""
    E = t2v_inputs(V[:n, :, :1], V[:n], K, True)[1].double()                # (n, n1, 2)
    e = K.cells[1].long().cpu()
    P = K.pos.detach().double().cpu()
    t = P[e[:, 1]] - P[e[:, 0]]                                              # (n1, 3)
    F = K.cells[2].long().cpu()
    nrm = torch.zeros_like(P).index_add_(0, F.reshape(-1), torch.cross(P[F[:, 1]] - P[F[:, 0]],
                                         P[F[:, 2]] - P[F[:, 0]], dim=1).repeat_interleave(3, 0))
    nrm = nrm / nrm.norm(dim=1, keepdim=True)
    ends = torch.cat([e[:, 0], e[:, 1]])
    proj = torch.cat([E[..., 0] - E[..., 1], E[..., 0] + E[..., 1]], dim=1)  # V_s . t, V_d . t   (n, 2 n1)
    tt = t.repeat(2, 1)
    M = torch.zeros(P.shape[0], 3, 3, dtype=torch.float64).index_add_(0, ends, tt[:, :, None] * tt[:, None, :])
    M = M + nrm[:, :, None] * nrm[:, None, :]                                 # pin the normal component to 0
    b = torch.zeros(n, P.shape[0], 3, dtype=torch.float64).index_add_(1, ends, proj[..., None] * tt[None])
    v = torch.linalg.solve(M[None], b.unsqueeze(-1)).squeeze(-1)
    Vt = V[:n].double() - (V[:n].double() * nrm).sum(-1, keepdim=True) * nrm
    return float(((v - Vt).norm(dim=-1).max() / Vt.norm(dim=-1).max()).item())


def load_T2v(droot, *, native, device, star, **_):
    """T2v: T2 with the advecting velocity given to the model (same mesh, samples and targets as T2).

    The T2 generator draws a new velocity ``v = alpha vt + beta vp`` for every sample (``gen_T2v.py``); without it
    the target is not a function of the input.  Native: scalar at the nodes and the velocity as an edge cochain
    (:func:`t2v_inputs`: odd one-form + even half-difference, ``in_dims {0: 1, 1: 2}``, ``even_dims {1: 1}``); the
    velocity one-form (column 0 of degree 1) is the natural transport field of ``RHMPConfig.transport``.  Legacy
    (node-input baselines): ``(f, Vx, Vy, Vz)`` at the nodes with per-feature statistics.  Target: ``u`` at the nodes.
    """
    d = _read_pickle(droot, "T2")
    pts = np.asarray(d["points"], dtype=np.float64)
    X, Y = _xy(d)
    K = _complex(pts, d["faces"], star, device)
    v2 = t2v_file(droot)
    V = v2["V"].float()
    if V.shape != (X.shape[0], X.shape[1], 3):
        raise RuntimeError(f"T2v: velocity shape {tuple(V.shape)} does not match the T2 data {tuple(X.shape)}")
    gm = v2.get("meta", {})
    meta = _defaults("T2v")
    meta["v1_name"] = V1_FILES["T2"][1]
    meta["velocity"] = dict(alpha=v2["alpha"].float(), beta=v2["beta"].float(),
                            note="v = alpha vt + beta vp per sample (gen_T2v.py)")
    meta["consistency"] = {"target_regeneration": gm.get("target_check"),
                           "V_bitwise_equal_to_generator_output": gm.get("V_regen_bitwise_equal"),
                           "edge_encoding_reconstruction_max_rel_err": _t2v_reconstruction_error(V, K)}
    if native:
        meta["native_inputs"] = ("node f; edge velocity 1-form (V_s+V_d)/2.(x_d-x_s) (odd) and half-difference "
                                 "(V_d-V_s)/2.(x_d-x_s) (even)")
        return _finish("T2v", K, t2v_inputs(X, V, K, True), Y, target_degree=0, target_kind="node_scalar",
                       in_dims={0: 1, 1: 2}, even_dims={1: 1}, connection_dims={},
                       x_kinds={0: "v1", 1: "scale"}, y_kind="v1", spatial_dim=3, native=True, device=device,
                       meta=meta)
    meta["legacy_inputs"] = "node (f, Vx, Vy, Vz)"
    return _finish("T2v", K, t2v_inputs(X, V, K, False), Y, target_degree=0, target_kind="node_scalar",
                   in_dims={0: 4}, even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=3,
                   native=False, device=device, meta=meta)


# ----------------------------------------------------------------------------------------------------------------
# T3: ellipsoid coexact flow (psi -> tangent vector field)
# ----------------------------------------------------------------------------------------------------------------
def load_T3(droot, *, native, device, star, output_map=None, stream=False, name="T3", **_):
    """T3 ellipsoid coexact flow psi -> v (node_vector).  Native: isotropic target scaling, the model predicts
    g = -n x v (true vector) and v = n x g is reported; legacy: v1 per-component normalisation, direct readout.
    ``stream=True`` (task ``T3g``): the model predicts a stream function, ``w = d0 phi`` on edges (``grad`` readout),
    and the fixed map of :func:`stream_function_map` turns it into the vertex field."""
    d = _read_pickle(droot, "T3")
    pts = np.asarray(d["points"], dtype=np.float64)
    X, Y = _xy(d)
    K = _complex(pts, d["faces"], star, device)
    meta = _defaults(name)
    meta["v1_name"] = V1_FILES["T3"][1]
    if stream:
        if not native:
            raise ValueError("T3g has no legacy variant")
        omap = stream_function_map(K, np.asarray(d["faces"], dtype=np.int64), pts)
        idx = _paper_eval(X.shape[0])[:4]
        psi = X[idx].permute(1, 0, 2).contiguous().to(K.device)                     # (n0, B, 1)
        v = omap(K.apply_d(0, psi)).permute(1, 0, 2).cpu()                          # (B, n0, 3)
        err = float((v - Y[idx]).norm() / Y[idx].norm())
        meta["consistency"] = {"rel_err_map_of_d0_psi_vs_target": err, "samples": int(len(idx))}
        if err > 1e-3:
            raise RuntimeError(f"T3g: output map does not reproduce the generator: {meta['consistency']}")
        meta["native_inputs"] = "node psi; readout grad (w = d0 phi on edges) + fixed generator edge->vertex map"
        return _finish(name, K, {0: X}, Y, target_degree=0, target_kind="node_vector", in_dims={0: 1},
                       even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="isotropic", spatial_dim=3,
                       native=True, device=device, meta=meta, output_map=omap)
    meta["vector_mode"] = "ls" if native else "direct"
    # v = n x grad(psi) is a pseudo-vector: the model predicts the true vector g = -n x v, reported as v = n x g
    omap = surface_rotation(K, np.asarray(d["faces"], dtype=np.int64), pts) \
        if (native and output_map is not False) else None
    return _finish("T3", K, {0: X}, Y, target_degree=0, target_kind="node_vector", in_dims={0: 1}, even_dims={},
                   connection_dims={}, x_kinds={0: "v1"}, y_kind="isotropic" if native else "v1", spatial_dim=3,
                   native=native, device=device, meta=meta, output_map=omap)


# ----------------------------------------------------------------------------------------------------------------
# T5: Maxwell / Poisson (rho -> E)
# ----------------------------------------------------------------------------------------------------------------
def load_T5(droot, *, native, device, star, grad=False, name="T5", **_):
    """T5 Maxwell/Poisson rho -> E.  Legacy: 2 node scalars (v1).  Native: node_vector (direct), isotropic scaling."""
    d = _read_pickle(droot, "T5")
    pts = _planar(np.asarray(d["points"], dtype=np.float64))
    X, Y = _xy(d)
    K = _complex(pts, d["faces"], star, device)
    meta = _defaults(name)
    meta["v1_name"] = V1_FILES["T5"][1]
    if grad:                                   # T5g: predict a potential, E_e = d0 phi, fixed v1 edge->node average
        if not native:
            raise ValueError("T5g has no legacy variant")
        omap = direct_vector_map(K, "grad")
        meta["consistency"] = _t5_poisson_check(d, K, omap, _paper_eval(X.shape[0]))
        if meta["consistency"]["max_rel_err_map_of_minus_d0_phi_vs_target"] > 1e-3:
            raise RuntimeError(f"T5g: output map does not reproduce the generator: {meta['consistency']}")
        meta["native_inputs"] = "node rho; readout grad (E = d0 phi on edges) + fixed v1 edge->node average"
        return _finish(name, K, {0: X}, Y, target_degree=0, target_kind="node_vector", in_dims={0: 1},
                       even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="isotropic", spatial_dim=2,
                       native=True, device=device, meta=meta, output_map=omap)
    if not native:
        return _finish("T5", K, {0: X}, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 1},
                       even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=2,
                       native=False, device=device, meta=meta)
    # the generator defines E_i = mean_{e ∋ i} E_e t_e, i.e. the 'direct' vector readout
    meta["vector_mode"] = "direct"
    return _finish("T5", K, {0: X}, Y, target_degree=0, target_kind="node_vector", in_dims={0: 1}, even_dims={},
                   connection_dims={}, x_kinds={0: "v1"}, y_kind="isotropic", spatial_dim=2, native=True,
                   device=device, meta=meta)


# ----------------------------------------------------------------------------------------------------------------
# T6 / T6f / T6_100K: U(1) Wilson loop
# ----------------------------------------------------------------------------------------------------------------
def _native_file(droot: str, task: str) -> dict:
    path = os.path.join(droot, "v2", f"{task}_native.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run python3 -u datasets/generators/gen_{task}_native.py (see datasets/README.md)")
    return torch.load(path, map_location="cpu", weights_only=False)


def load_T6(droot, *, native, device, star, face_target=False, name="T6", output_map=None, **_):
    """T6 U(1) Wilson loop.  Legacy: node encoding ``(N, n0, 3)``.  Native: theta on edges in exact gauge-connection
    mode; node target through ``cochain:2`` + oriented face->node map, or face target (``face_target``: T6f)."""
    d = _read_pickle(droot, "T6")
    pts = _planar(np.asarray(d["points"], dtype=np.float64))
    faces = np.asarray(d["faces"], dtype=np.int64)
    X, Y = _xy(d)
    K = _complex(pts, faces, star, device)
    meta = _defaults(name)
    meta["v1_name"] = V1_FILES["T6"][1]
    if not native:
        if face_target:
            raise ValueError("T6f has no legacy variant (face targets need the native edge inputs)")
        return _finish("T6", K, {0: X}, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 3},
                       even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=2,
                       native=False, device=device, meta=meta)
    nat = _native_file(droot, "T6")
    perm, sign = edge_alignment(K.cells[1], nat["edges"], K.n[0])
    theta = (nat["theta_edges"][:, perm] * sign[None]).unsqueeze(-1)         # (N, n1, 1), K orientation
    fperm, fsign = cell_alignment(K.cells[2], nat["faces"], K.n[0])
    plaq = (nat["plaq"][:, fperm] * fsign[None]).unsqueeze(-1)              # (N, n2, 1), K orientation
    # consistency: d1 theta == plaq (both in the orientation of the complex)
    th = theta[:16].permute(1, 0, 2).contiguous().to(K.device)
    err = (K.apply_d(1, th).permute(1, 0, 2).cpu() - plaq[:16]).abs().max().item()
    meta["consistency"] = {"d1_theta_minus_plaq_maxabs": err, "generator": nat["meta"]}
    if err > 1e-4:
        raise RuntimeError(f"T6 native alignment failed: max|d1 theta - plaq| = {err:.3e}")
    meta["native_inputs"] = "theta on edges (U(1) connection, exact gauge mode)"
    if face_target:
        return _finish(name, K, {1: theta}, plaq, target_degree=2, target_kind="cochain", in_dims={1: 1},
                       even_dims={}, connection_dims={1: 1}, x_kinds={1: "scale"}, y_kind="scale", spatial_dim=2,
                       native=True, device=device, meta=meta)
    omap = oriented_face_to_node(K, 1) if output_map is not False else None       # node flux = pseudo-scalar
    return _finish(name, K, {1: theta}, Y, target_degree=0, target_kind="node_scalar", in_dims={1: 1},
                   even_dims={}, connection_dims={1: 1}, x_kinds={1: "scale"}, y_kind="v1", spatial_dim=2,
                   native=True, device=device, meta=meta, output_map=omap)


def load_T6_100K(droot, *, native, device, star, **_):
    """T6 at 100K faces (legacy node encoding only; step benchmark)."""
    d = _read_pickle(droot, "T6_100K")
    pts = _planar(np.asarray(d["points"], dtype=np.float64))
    X, Y = _xy(d)
    K = _complex(pts, d["faces"], star, device)
    meta = _defaults("T6_100K")
    meta["v1_name"] = V1_FILES["T6_100K"][1]
    if native:
        meta["note"] = "no raw edge fields for T6_100K; legacy node encoding used"
    return _finish("T6_100K", K, {0: X}, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 3},
                   even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=2, native=False,
                   device=device, meta=meta)


# ----------------------------------------------------------------------------------------------------------------
# T7 / T7f: SU(2) Yang-Mills
# ----------------------------------------------------------------------------------------------------------------
def load_T7(droot, *, native, device, star, face_target=False, name="T7", connection=True, output_map=None, **_):
    """T7 SU(2) Yang-Mills.  Legacy: node encoding ``(N, n0, 9)``.  Native: A on edges (3 odd columns, connection +
    odd path); node target via ``cochain:2`` + oriented face->node map, or face target F (T7f)."""
    d = _read_pickle(droot, "T7")
    pts = _planar(np.asarray(d["points"], dtype=np.float64))
    faces = np.asarray(d["faces"], dtype=np.int64)
    X, Y = _xy(d)
    K = _complex(pts, faces, star, device)
    meta = _defaults(name)
    meta["v1_name"] = V1_FILES["T7"][1]
    if not native:
        if face_target:
            raise ValueError("T7f has no legacy variant")
        return _finish("T7", K, {0: X}, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 9},
                       even_dims={}, connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=2,
                       native=False, device=device, meta=meta)
    nat = _native_file(droot, "T7")
    perm, sign = edge_alignment(K.cells[1], nat["edges"], K.n[0])
    A = nat["A_edges"][:, perm] * sign[None, :, None]                        # (N, n1, 3)
    fperm, fsign = cell_alignment(K.cells[2], nat["faces"], K.n[0])
    F = nat["F_faces"][:, fperm] * fsign[None, :, None]                      # (N, n2, 3)
    del nat["A_edges"], nat["F_faces"]
    # consistency: the abelian part d1 A agrees with F up to the commutator (|dA| / |[A,A]| ~ 5.7)
    a = A[:8].permute(1, 0, 2).contiguous().to(K.device)
    dA = K.apply_d(1, a).permute(1, 0, 2).cpu()
    rel = ((dA - F[:8]).norm() / F[:8].norm()).item()
    meta["consistency"] = {"rel_norm_F_minus_dA": rel, "generator": nat["meta"]}
    if rel > 0.5:
        raise RuntimeError(f"T7 native alignment failed: |F - d1 A| / |F| = {rel:.3f}")
    meta["native_inputs"] = "su(2) connection A on edges (3 odd columns)"
    cdims = {1: 3} if connection else {}
    if face_target:
        return _finish(name, K, {1: A}, F, target_degree=2, target_kind="cochain", in_dims={1: 3}, even_dims={},
                       connection_dims=cdims, x_kinds={1: "scale"}, y_kind="scale", spatial_dim=2, native=True,
                       device=device, meta=meta)
    omap = oriented_face_to_node(K, 3) if output_map is not False else None       # node F = pseudo-scalars
    return _finish(name, K, {1: A}, Y, target_degree=0, target_kind="node_scalar", in_dims={1: 3}, even_dims={},
                   connection_dims=cdims, x_kinds={1: "scale"}, y_kind="v1", spatial_dim=2, native=True,
                   device=device, meta=meta, output_map=omap)


# ----------------------------------------------------------------------------------------------------------------
# T8: AirfRANS pressure, one mesh per sample
# ----------------------------------------------------------------------------------------------------------------
def complex_code_version() -> str:
    """Hash of the complex/geometry implementation (cached complexes are rebuilt when it changes)."""
    import hashlib
    import rhmp.complex
    import rhmp.geometry
    h = hashlib.sha1()
    for mod in (rhmp.complex, rhmp.geometry):
        with open(mod.__file__, "rb") as f:
            h.update(f.read())
    return h.hexdigest()[:16]


def _t8_complexes(droot: str, samples: list, star: str, cache: bool):
    """Build (or load from ``datasets/v2/T8_complexes_<star>.pt``) the CPU complexes of all T8 samples.

    The cache is keyed on the number of samples and on :func:`complex_code_version` (a hash of ``rhmp/complex.py``
    and ``rhmp/geometry.py``), so it is rebuilt automatically (about 8 s) whenever the complex implementation changes.
    """
    from rhmp.complex import CochainComplex
    path = os.path.join(droot, "v2", f"T8_complexes_{star}.pt")
    version = complex_code_version()
    if cache and os.path.exists(path):
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001 - unpickling an outdated class layout
            blob = {}
        if blob.get("n") == len(samples) and blob.get("version") == version:
            return blob["complexes"], dict(blob.get("info", {}), cache="hit", path=path)
    t0 = time.time()
    Ks = []
    for s in samples:
        pts = _planar(np.asarray(s["pts_3d"], dtype=np.float64))
        Ks.append(CochainComplex.from_triangles(torch.as_tensor(pts), torch.as_tensor(np.asarray(s["faces"])),
                                                star=star, device="cpu"))
    info = dict(build_seconds=time.time() - t0, n=len(Ks))
    if cache:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({"n": len(Ks), "version": version, "complexes": Ks, "info": info}, path)
        info = dict(info, cache="written", path=path, file_MB=os.path.getsize(path) / 1e6)
    return Ks, info


def _t8_inflow_forms(Ks, x_all):
    """Uniform inflow ``V = U (cos a, sin a)`` of every T8 sample as an odd edge 1-form ``theta_e = V . (x_d - x_s)``.

    ``x_pres`` columns are ``(sdf, U, a)`` with ``U`` (m/s) and the angle of attack ``a`` (radians, AirfRANS) constant
    per sample; the positions are in the airfoil frame (chord along +x), the frame ``a`` is measured in.  For a
    uniform field the midpoint rule ``(V_s + V_d)/2 . (x_d - x_s)`` is exact.

    Returns:
        list of ``(n1_i, 1)`` float32 tensors (edge order and orientation of the complexes).
    """
    out = []
    for K, x in zip(Ks, x_all):
        U, a = x[:, 1].double(), x[:, 2].double()
        if float(U.max() - U.min()) > 1e-4 * float(U.abs().max()) or float(a.max() - a.min()) > 1e-6:
            raise RuntimeError("T8: inlet velocity / angle of attack are not constant within a sample")
        V = torch.stack([U[0] * torch.cos(a[0]), U[0] * torch.sin(a[0])])
        ev = K.edge_vectors().double().cpu()
        out.append((ev @ V).float()[:, None])
    return out


def _t8_inflow_check(samples, n: int = 50) -> dict:
    """Consistency of the inflow convention with the stored flow field: far-field (top 10 % sdf) mean velocity of
    ``x_vort[:, :2]`` vs ``U (cos a, sin a)`` (a in radians) on the first ``n`` samples."""
    rel, cosang = [], []
    for s in samples[:n]:
        xp, xv = np.asarray(s["x_pres"], dtype=np.float64), np.asarray(s.get("x_vort", []), dtype=np.float64)
        if xv.ndim != 2 or xv.shape[1] < 2:
            return {"available": False}
        U, a = xp[0, 1], xp[0, 2]
        far = xp[:, 0] > np.quantile(xp[:, 0], 0.9)
        v = xv[far, :2].mean(0)
        d = np.array([np.cos(a), np.sin(a)])
        rel.append(float(np.linalg.norm(v - U * d) / U))
        cosang.append(float(v @ d / np.linalg.norm(v)))
    return {"available": True, "far_field_rel_err_median": float(np.median(rel)), "cos_angle_min": float(min(cosang))}


def load_T8(droot, *, native, device, star, cache=True, max_samples=None, whitney=True, inflow=False, name="T8",
            **_):
    """T8 AirfRANS pressure, one mesh per sample: list of complexes (cached), v1 70/30 split with the last 10 % of
    the train part as validation set, numpy-style (biased) normalisation from the first 700 samples as v1.

    ``inflow=True`` (task ``T8v``): the node inputs ``(sdf, U, a)`` are kept and the inflow velocity is added as an
    odd edge 1-form ``theta_e = V_inf . (x_d - x_s)`` (scale-normalised with the RMS of the first 700 samples, like
    the node statistics).  A reflection-invariant model cannot tell the pressure side from the suction side from the
    scalar angle ``a`` alone (a symmetric airfoil at ``a != 0`` has mirror-symmetric geometry but no mirror-symmetric
    pressure); the 1-form carries the flow direction equivariantly.
    """
    samples = _read_pickle(droot, "T8")
    N_all = len(samples)
    Ks, info = _t8_complexes(droot, samples, star, cache)
    x_all = [torch.as_tensor(np.asarray(s["x_pres"], dtype=np.float32)) for s in samples]
    y_all = [torch.as_tensor(np.asarray(s["y_pres"], dtype=np.float32)) for s in samples]
    x_all = [x[:, None] if x.dim() == 1 else x for x in x_all]
    y_all = [y[:, None] if y.dim() == 1 else y for y in y_all]
    for K, x in zip(Ks, x_all):
        if K.n[0] != x.shape[0]:
            raise RuntimeError("T8 complex / sample size mismatch")
    # v1 normalisation: numpy (biased) statistics over the first int(0.7 N) samples (formal_benchmark.py)
    n_v1_train = int(0.7 * N_all)
    x_stats = feature_stats(x_all[:n_v1_train], unbiased=False)
    y_stats = feature_stats(y_all[:n_v1_train], unbiased=False)
    e_all, e_stats, inflow_check = None, None, None
    if inflow:
        if not native:
            raise ValueError("T8v has no legacy variant (the inflow 1-form is a native cochain input)")
        e_all = _t8_inflow_forms(Ks, x_all)
        e_stats = scale_stats(e_all[:n_v1_train])
        inflow_check = _t8_inflow_check(samples)
        if inflow_check.get("available") and inflow_check["cos_angle_min"] < 0.99:
            raise RuntimeError(f"T8v: inflow direction disagrees with the stored flow field: {inflow_check}")
    split = holdout_split(N_all)
    idx_keep = None
    if max_samples is not None and max_samples < N_all:     # smoke tests: keep a few samples of every part
        per = max(1, max_samples // 3)
        idx_keep = torch.cat([s[:per] for s in split])
        remap = {int(i): j for j, i in enumerate(idx_keep.tolist())}
        split = tuple(torch.tensor([remap[int(i)] for i in s[:per].tolist()]) for s in split)
        Ks = [Ks[i] for i in idx_keep.tolist()]
        x_all = [x_all[i] for i in idx_keep.tolist()]
        y_all = [y_all[i] for i in idx_keep.tolist()]
        if e_all is not None:
            e_all = [e_all[i] for i in idx_keep.tolist()]
    if not whitney:                              # only tensor metrics use the Whitney blocks
        from rhmp.data import strip_whitney
        strip_whitney(Ks)
    from rhmp.data import pack_to            # one packed copy per dtype instead of ~40 per complex
    if e_all is None:
        raw_inputs = [{0: x_stats.normalize(x)} for x in x_all]
    else:
        raw_inputs = [{0: x_stats.normalize(x), 1: e_stats.normalize(e)} for x, e in zip(x_all, e_all)]
    Ks, inputs, target = pack_to((Ks, raw_inputs, [y_stats.normalize(y) for y in y_all]), device)
    meta = _defaults(name)
    meta.update(v1_name=V1_FILES["T8"][1], complexes=info, r2_std=y_stats.std,
                split_note="v1 70/30 (first 700 train, last 300 test); v2 selects on the last 10% of the train "
                           "part (630..699), v1 selected on the test part; normalisation from the first 700 (v1)",
                paper_eval=(torch.arange(850, 950) if idx_keep is None else split[2]))
    in_dims = {0: int(x_all[0].shape[-1])}
    x_stats_d = {0: x_stats.to(device)}
    if e_all is not None:
        in_dims[1] = 1
        x_stats_d[1] = e_stats.to(device)
        meta.update(native_inputs="node (sdf, U, angle of attack); edge inflow 1-form V_inf . (x_d - x_s)",
                    consistency={"inflow": inflow_check})
    return TaskData(name=name, K=Ks, inputs=inputs, target=target, target_degree=0, target_kind="node_scalar",
                    in_dims=in_dims, even_dims={}, connection_dims={}, split=split, x_stats=x_stats_d,
                    y_stats=y_stats.to(device), spatial_dim=2, out_dim=int(y_all[0].shape[-1]), native=native,
                    meta=meta)


# ----------------------------------------------------------------------------------------------------------------
# dispatcher
# ----------------------------------------------------------------------------------------------------------------
def load(name: str, droot: str, *, native: bool, device, star: str, **kw) -> TaskData:
    """Dispatch a paper task name to its loader."""
    if name == "T1":
        return load_T1(droot, native=native, device=device, star=star, **kw)
    if name == "T1q":
        return load_T1(droot, native=native, device=device, star=star, quad=True, **kw)
    if name == "T1qe":
        return load_T1(droot, native=native, device=device, star=star, quad=True, edge_even=True, name="T1qe", **kw)
    if name == "T1qn":
        return load_T1(droot, native=native, device=device, star=star, quad=True, edge_even=True, node_readout=True,
                       name="T1qn", **kw)
    if name == "T1p":
        return load_T1p(droot, native=native, device=device, star=star, **kw)
    if name == "T1pm":
        return load_T1p(droot, native=native, device=device, star=star, name="T1pm", cell_fields=True, **kw)
    if name == "T1pml":
        return load_T1p(droot, native=native, device=device, star=star, name="T1pml", cell_fields=True,
                        vertex_logs=True, **kw)
    if name == "T1pf":
        return load_T1p(droot, native=native, device=device, star=star, name="T1pf", filtered=True, **kw)
    if name == "T1pmf":
        return load_T1p(droot, native=native, device=device, star=star, name="T1pmf", cell_fields=True,
                        filtered=True, **kw)
    if name == "T1pmlf":
        return load_T1p(droot, native=native, device=device, star=star, name="T1pmlf", cell_fields=True,
                        vertex_logs=True, filtered=True, **kw)
    if name == "T1g":
        return load_T1g(droot, native=native, device=device, star=star, **kw)
    if name == "T1gml":
        return load_T1g(droot, native=native, device=device, star=star, name="T1gml", cell_fields=True,
                        vertex_logs=True, **kw)
    if name == "T1gmla":
        return load_T1g(droot, native=native, device=device, star=star, name="T1gmla", cell_fields=True,
                        vertex_logs=True, abs_geo=True, **kw)
    if name == "T1gml_mean":
        if not native:
            raise ValueError("T1gml_mean has no legacy variant (it changes the RHMP output map only)")
        return load_T1g(droot, native=native, device=device, star=star, name="T1gml_mean", cell_fields=True,
                        vertex_logs=True, area_map=False, **kw)
    if name == "T2":
        return load_T2(droot, native=native, device=device, star=star, **kw)
    if name == "T2v":
        return load_T2v(droot, native=native, device=device, star=star, **kw)
    if name == "T2a":
        from rhmp.tasks.torus import load_T2a
        return load_T2a(droot, native=native, device=device, star=star, **kw)
    if name == "T3":
        return load_T3(droot, native=native, device=device, star=star, **kw)
    if name == "T3g":
        return load_T3(droot, native=native, device=device, star=star, stream=True, name="T3g", **kw)
    if name == "T5":
        return load_T5(droot, native=native, device=device, star=star, **kw)
    if name == "T5g":
        return load_T5(droot, native=native, device=device, star=star, grad=True, name="T5g", **kw)
    if name == "T6":
        return load_T6(droot, native=native, device=device, star=star, **kw)
    if name == "T6f":
        return load_T6(droot, native=True, device=device, star=star, face_target=True, name="T6f", **kw)
    if name == "T7":
        return load_T7(droot, native=native, device=device, star=star, **kw)
    if name == "T7f":
        return load_T7(droot, native=True, device=device, star=star, face_target=True, name="T7f", **kw)
    if name == "T8":
        return load_T8(droot, native=native, device=device, star=star, **kw)
    if name == "T8v":
        return load_T8(droot, native=native, device=device, star=star, inflow=True, name="T8v", **kw)
    if name == "T6_100K":
        return load_T6_100K(droot, native=native, device=device, star=star, **kw)
    raise KeyError(name)
