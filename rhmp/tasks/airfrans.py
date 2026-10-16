"""AirfRANS task family (generator ``datasets/generators/gen_airfrans.py``; data from the AirfRANS RANS simulations).

All tasks keep the simulation order and the partition of the paper's T8 task: simulations 0..699 are the training
part (normalisation statistics; the loaders hold out 630..699 for model selection, as T8), 700..999 the test part, and
``meta['paper_eval']`` = simulations 850..949 ("test100", the paper's table).  Lengths are in chords, velocities in
m/s, pressures in Pa, vorticities in 1/s, the angle of attack in radians.

==================  =========================================================================================  ======
name                content                                                                                   mesh
==================  =========================================================================================  ======
T8_n500/2000/8000   test-only resolution views of T8: the same construction (uniform random CFD nodes,         T8res
                    Delaunay of the convex hull) with 500 / 2000 / 8000 nodes on the 300 test simulations
                    (nested point sets: the prefixes of T8's random permutations; n2000 reproduces the T8
                    test split exactly); x = (sdf, U, a), y = p; ``extra_tests['test100']``.
T8i                 T8 with the AFPi inputs (node (sdf, a), edge inflow 1-form + mean sdf, face mean sdf) and   T8
                    the AFPi output map; native only.
T8fr / T8xy /       T8 (same meshes, split, target and normalisation) with the frame that the gauge-CNN        T8
T8frxy              baselines see through their edge angles: ``fr`` = odd edge 1-forms (inflow
                    ``U (cos a dx + sin a dy)``, ``dx_e``, ``dy_e``), ``xy`` = chord-frame coordinates (x, y)
                    as two more node columns (all models).
AFV                 vorticity on conforming near-field meshes, Stokes target: the mean vorticity of the CFD    AFM
                    field over the star of every vertex, ``sum_{f ∋ i} sigma_f (d1 theta)_f / sum_{f ∋ i}
                    A_f`` with ``theta_e = int_e u.dl`` (accurate line integrals of the CFD velocity; Stokes:
                    ``(d1 theta)_f`` is the integral of the vorticity over triangle f).  native: node (sdf, U, a),
                    edge theta as a gauge connection (the model sees only d1 theta: exact invariance under
                    u -> u + grad lambda); readout ``cochain:2`` + the fixed area-weighted oriented face ->
                    vertex map.  legacy (node baselines): node (Vx, Vy, sdf, U, a) of the CFD.
AFVp                as AFV, target = the CFD point vorticity at the vertices (wall: the wall vorticity).       AFM
AFV_trap /          equal-information variants: theta = the trapezoid rule of the nodal CFD velocities (the     AFM
AFVp_trap           information the node baselines get); AFV_trap's target is then the star average of the
                    P1 curl of the nodal velocities (the discrete Stokes target of the T1 / v1 vorticity data).
AFU                 velocity flux 1-cochain ``F_e = int_e u.n ds`` (n = right normal of src -> dst; accurate    AFM
                    quadrature, closed per triangle to ~1e-4 for the incompressible CFD field).  native: node
                    (sdf, U, a, x, y), edge inflow 1-form; readout ``grad`` (F = d0 psi: exactly closed,
                    zero net flux out of every triangle).  legacy: node (sdf, U, a, x, y) -> node (Vx, Vy).
AFUv                native AFU inputs and ``grad`` readout, trained on the node velocities through the fixed   AFM
                    map F -> u (least-squares gradient of psi at every vertex, rotated: u = (psi_y, -psi_x);
                    u = 0 at wall vertices, the no-slip condition).
AFP / AFPfr /       pressure on the conforming meshes (the geometry-respecting counterpart of T8): node         AFM
AFPxy / AFPfrxy     (sdf, U, a); ``fr``: + odd edge 1-forms (inflow, dx, dy; native only); ``xy``: + chord-frame
                    coordinates as node columns (all models).
AFPi / AFPi0 /      pressure with the inflow in its covariant form (dynamic similarity).  native: node (sdf, a)  AFM
AFPiU / AFPiR /     [AFPi0: (sdf); AFPiU: (sdf, U, a); AFPiR: (sdf, a, log U)], edge inflow 1-form
AFPs                ``theta_e = u_inf . (x_dst - x_src)`` (odd; the amplitude column of ``--amplitude 1:1``) and the
                    mean sdf of the edge / face vertices as even edge / face columns (seen by the metric heads).
                    AFPs: no 1-form, node (U, sdf, a) with U as the amplitude column (``--amplitude 0:1``).  The
                    complexes carry ``meta['dirichlet'] = {0: outer-ring vertices}`` and ``meta['cohomology'] = {1:
                    generator of H^1 of the annulus}`` for projection layers.  Target and normalisation are those of
                    AFP (raw pressure, mean / std of simulations 0..699); AFPi / AFPi0 / AFPiR / AFPs add the fixed
                    output map ``y -> y - mean / std`` so that a model with the output factor ``s^2`` (``--amplitude``
                    ``--amplitude-power 2 --amplitude-energy``) is exactly homogeneous of degree two in the inflow in
                    physical units.  legacy (node baselines): node (sdf, U, a), i.e. AFP.
AFPpf               control: AFP with the Bernoulli pressure of the P1 potential flow with the Kutta condition    AFM
                    (:func:`potential_reference`: reference metric, no learning) as a fourth node column
                    ``(sdf, U, a, p_ref)`` for every model; target and normalisation of AFP.
AFPcp / AFPcpxy     controls: target = pressure coefficient ``p / (U^2 / 2)`` (mean / std normalisation); legacy   AFM
                    node (sdf, U, a) [AFPcpxy: + chord-frame (x, y)]; native: the AFPi inputs, no output map.
                    Predictions are converted to physical units by ``scripts/airfrans_pressure_eval.py``.
AFD                 derived diffusion task: P1 FEM solution of -div(kappa grad T) = f, T = 0 on the airfoil    AFM
                    and the outer ring, kappa = 1 + nu_t / nu of the simulation (the RANS eddy-viscosity
                    ratio, contrast up to ~1e3), f = random Gaussian sources.  The solution is computed by the
                    generator; only the meshes and kappa come from AirfRANS.  native: node f, even edge / face
                    log kappa (material columns for solver mode); legacy: node (f, mean log kappa).
==================  =========================================================================================  ======

The conforming tasks have ``extra_tests['fine']`` (the 300 test simulations on meshes with about 3.6x the nodes, same
construction with h halved, independent random nodes; zero-shot resolution transfer) and ``'fine_test100'``, and, when
``AFM_coarse.pt`` exists, ``'coarse'`` / ``'coarse_test100'`` (h doubled, about 0.3x the nodes).
"""
from __future__ import annotations

import functools
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from torch import Tensor

from rhmp.data import (Stats, TaskData, cell_alignment, edge_alignment, feature_stats, holdout_split, pack_to,
                       scale_stats)

__all__ = ["AIRFRANS_TASKS", "AIRFRANS_DEFAULTS", "airfrans_task_defaults", "StarAverageMap", "FluxVelocityMap",
           "OffsetMap", "wall_vertices", "afm_eval_source", "afp_output_map", "AFP_NATIVE", "load_t8_inflow",
           "potential_reference",
           "load_t8res", "load_t8_frame", "load_afm", "flux_divergence", "trapezoid_flux", "edge_flux_of_nodes"]

BUILD_THREADS = int(os.environ.get("RHMP_BUILD_THREADS", 4))
GPU_BUDGET_GB = float(os.environ.get("RHMP_GPU_BUDGET_GB", 4.0))
N_TRAIN_STATS = 700                                         # T8: statistics of the first 700 simulations
TEST100 = (850, 950)
KIND_VOLUME, KIND_WALL, KIND_OUTER = 0, 1, 2

AIRFRANS_DEFAULTS: dict[str, dict] = {
    "T8res": dict(C=256, layers=4, batch=8),
    "T8f": dict(C=256, layers=4, batch=8),
    "AF": dict(C=128, layers=4, batch=8),
}


def airfrans_task_defaults(name: str) -> dict:
    """Training defaults (``C``, ``layers``, ``batch`` = meshes per block-diagonal batch)."""
    if name.startswith("T8_n"):
        return dict(AIRFRANS_DEFAULTS["T8res"])
    if name.startswith("T8"):
        return dict(AIRFRANS_DEFAULTS["T8f"])
    return dict(AIRFRANS_DEFAULTS["AF"])


def _meta_defaults(name: str) -> dict:
    d = airfrans_task_defaults(name)
    return dict(v1_C=d["C"], batch_size=d["batch"], layers=d["layers"])


# ======================================================================================================================
# helpers
# ======================================================================================================================
def _load_blob(droot: str, base: str) -> dict:
    path = os.path.join(droot, "v2", base + ".pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run python3 -u datasets/generators/gen_airfrans.py (see its "
                                f"docstring; needs the AirfRANS data set)")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["_path"] = path
    return blob


def _sl(blob: dict, key: str, i: int, deg: int) -> Tensor:
    p = blob[f"ptr{deg}"]
    return blob[key][p[i]:p[i + 1]]


def _complex(pos: Tensor, faces: Tensor, star: str, strict: bool, whitney: bool):
    from rhmp.complex import CochainComplex
    K = CochainComplex.from_triangles(pos.double(), faces.long(), star=star, device="cpu")
    if strict and K.n[2] != faces.shape[0]:
        raise RuntimeError(f"the complex dropped {faces.shape[0] - K.n[2]} faces")
    if not whitney:
        K.whitney = {}
    return K


def _build(meshes: list, star: str, strict: bool, whitney: bool) -> list:
    """CPU complexes of ``[(pos, faces), ...]`` in a thread pool (bitwise identical to a serial build)."""
    if BUILD_THREADS <= 1 or len(meshes) < 8:
        return [_complex(p, f, star, strict, whitney) for p, f in meshes]
    nt = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with ThreadPoolExecutor(BUILD_THREADS) as ex:
            return list(ex.map(lambda m: _complex(m[0], m[1], star, strict, whitney), meshes))
    finally:
        torch.set_num_threads(nt)


def _nbytes(Ks: list) -> float:
    from rhmp.data import tensor_nbytes
    n = min(20, len(Ks))
    return sum(tensor_nbytes(K) for K in Ks[:n]) / max(n, 1) * len(Ks) / 1e9


def _subset(src: dict, idx) -> dict:
    idx = [int(i) for i in idx]
    return dict(K=[src["K"][i] for i in idx], inputs=[src["inputs"][i] for i in idx],
                target=[src["target"][i] for i in idx])


def _empty_split(N: int) -> tuple[Tensor, Tensor, Tensor]:
    e = torch.zeros(0, dtype=torch.long)
    return e, e.clone(), torch.arange(N)


def _cat_stats(parts: list[Stats], kind: str) -> Stats:
    return Stats(torch.cat([p.mean for p in parts]), torch.cat([p.std for p in parts]), kind)


def signed_areas(K) -> Tensor:
    """``(n2,)`` signed areas of the triangles of a planar complex in their stored orientation (float64)."""
    F = K.cells[2].long()
    P = K.pos.double()[F]
    return 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) -
                  (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))


# ======================================================================================================================
# fixed output maps that depend on the (batched) complex
# ======================================================================================================================
class StarAverageMap:
    """Oriented face 2-cochain -> vertex density: ``y_i = A_ref sum_{f ∋ i} sigma_f c_f / sum_{f ∋ i} |A_f|``.

    ``sigma_f`` is the sign of the signed area (+1 for counter-clockwise faces, the physical orientation of the plane).
    For a face circulation ``c = d1 theta`` this is the mean vorticity over the star of vertex i (Stokes; the interior
    edges of the star cancel).  ``A_ref`` is a fixed constant that only sets the units of the model output (the loaders
    use the median star area of the training meshes, so that predictions at initialisation scale like the targets).  Parameter free and linear; computed from the batch complex (block-diagonal batches of
    different meshes), so ``takes_complex = True`` (``rhmp.train.model_output`` passes the complex).
    """
    kind = "star_average"
    takes_complex = True

    def __init__(self, out_dim: int = 1, scale: float = 1.0):
        self.model_readout = "cochain:2"
        self.model_out_dim = int(out_dim)
        self.scale = float(scale)
        self.description = (f"oriented face -> vertex star average A_ref sum sigma_f c_f / sum |A_f| (Stokes density; "
                            f"A_ref = {self.scale:.4e})")

    @staticmethod
    def weights(K) -> tuple[Tensor, Tensor, Tensor]:
        sa = signed_areas(K)
        F = K.cells[2].long()
        den = torch.zeros(K.n[0], dtype=torch.float64, device=F.device).index_add_(
            0, F.reshape(-1), sa.abs().repeat_interleave(3))
        return F, torch.sign(sa), den

    def __call__(self, y: Tensor, K=None) -> Tensor:
        if K is None:
            raise ValueError("StarAverageMap needs the complex: call it as omap(y, K)")
        F, sig, den = self.weights(K)
        c = y * sig.to(y.dtype)[:, None, None]                                  # (n2, B, O)
        num = torch.zeros((K.n[0],) + tuple(y.shape[1:]), dtype=y.dtype, device=y.device)
        num = num.index_add(0, F.reshape(-1), c.repeat_interleave(3, dim=0))
        return num * (self.scale / den.clamp_min(1e-300)).to(y.dtype)[:, None, None]

    def to(self, device) -> "StarAverageMap":
        return self


class OffsetMap:
    """Fixed affine output map ``y -> scale * y + offset`` (constants of the task).  With ``offset = -mean / std`` of a
    mean / std normalised target, a model whose raw output is homogeneous in an input amplitude stays exactly
    homogeneous in physical units (``std * scale * y_model``) while the loss and the metrics use the task's normalised
    target; ``scale`` is a unit factor (the loaders choose it so that the amplitude factor of the model output is of
    order one on the training simulations)."""
    kind = "offset"
    takes_complex = False

    def __init__(self, offset: float, scale: float = 1.0, model_readout: str = "node_scalar", out_dim: int = 1):
        self.offset = float(offset)
        self.scale = float(scale)
        self.model_readout = model_readout
        self.model_out_dim = int(out_dim)
        self.description = (f"y -> {self.scale:.6g} y {self.offset:+.6f} in normalised target units (fixed unit factor "
                            f"and offset; homogeneous model output)")

    def __call__(self, y: Tensor, K=None) -> Tensor:
        return y * self.scale + self.offset

    def to(self, device) -> "OffsetMap":
        return self


def wall_vertices(K) -> Tensor:
    """``(n0,)`` bool: boundary vertices on the airfoil.  Every AirfRANS airfoil has its chord on [0, 1] x {0} and the
    outer boundary of the near-field meshes is at distance 0.5 from it, so the airfoil boundary vertices are the
    boundary vertices in the box (-0.25, 1.25) x (-0.35, 0.35) (checked against the generator's flags by the
    loader).  Works on block-diagonal batches."""
    x = K.pos[:, :2]
    return K.boundary[0].bool() & (x[:, 0] > -0.25) & (x[:, 0] < 1.25) & (x[:, 1].abs() < 0.35)


class FluxVelocityMap:
    """Edge flux 1-cochain ``F = d0 psi`` (``F_e = int_e u.n ds``, n = right normal) -> vertex velocities.

    At every vertex the gradient of the stream function is the least-squares fit ``g_i = argmin sum_{e ∋ i}
    (g . dx_e - dpsi_e)^2 / |dx_e|^2`` over the incident edges (``dpsi_e = +-F_e``, oriented away from i), and
    ``u_i = (g_y, -g_x)``; wall vertices (:func:`wall_vertices`) get ``u = 0`` (the no-slip condition).  ``scale`` is a
    fixed unit factor (a data-set constant).  Fixed and linear; per batch from the complex (``takes_complex = True``).
    Output ``(n0, B, 2)``.
    """
    kind = "flux_velocity"
    takes_complex = True

    def __init__(self, model_readout: str = "grad", scale: float = 1.0):
        self.model_readout = model_readout
        self.model_out_dim = 1
        self.scale = float(scale)
        self.description = (f"edge flux d0 psi -> vertex velocity (least-squares grad psi, rotated; u = 0 on the wall; "
                            f"fixed unit factor {self.scale:.4e})")

    @staticmethod
    def operator(K) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """``(rows, cols, vals)`` of the ``(2 n0, n1)`` matrix and the wall mask (float64)."""
        e = K.cells[1].long()
        P = K.pos.double()
        n0, n1 = K.n[0], K.n[1]
        dx = P[e[:, 1]] - P[e[:, 0]]                                             # (n1, 2) src -> dst
        w = 1.0 / (dx * dx).sum(1).clamp_min(1e-300)
        # for vertex src: dpsi = F (dst - src); for vertex dst: dx -> -dx, dpsi -> -F: the products are the same
        M = torch.zeros(n0, 2, 2, dtype=torch.float64, device=P.device)
        outer = w[:, None, None] * dx[:, :, None] * dx[:, None, :]
        M.index_add_(0, e[:, 0], outer)
        M.index_add_(0, e[:, 1], outer)
        tr = M.diagonal(dim1=1, dim2=2).sum(-1)
        Minv = torch.linalg.inv(M + 1e-12 * tr[:, None, None] * torch.eye(2, dtype=M.dtype, device=M.device))
        # g_i = Minv_i sum_e w_e dx_e F_e (both endpoints): coefficient of F_e in g_i is Minv_i w_e dx_e
        rows, cols, vals = [], [], []
        for end in (0, 1):
            i = e[:, end]
            cg = torch.einsum("eab,eb->ea", Minv[i], w[:, None] * dx)          # (n1, 2) coefficient of F_e in g_i
            cu = torch.stack([cg[:, 1], -cg[:, 0]], 1)                           # u = (g_y, -g_x)
            for comp in (0, 1):
                rows.append(2 * i + comp)
                cols.append(torch.arange(n1, device=P.device))
                vals.append(cu[:, comp])
        rows, cols, vals = torch.cat(rows), torch.cat(cols), torch.cat(vals)
        wall = wall_vertices(K).to(P.device)
        keep = ~wall[rows // 2]
        return rows[keep], cols[keep], vals[keep], wall

    def __call__(self, y: Tensor, K=None) -> Tensor:
        if K is None:
            raise ValueError("FluxVelocityMap needs the complex: call it as omap(y, K)")
        from rhmp.ops import csr_from_coo, spmm
        rows, cols, vals, _ = self.operator(K)
        n0, n1 = K.n[0], K.n[1]
        vals = vals * self.scale
        Mx = csr_from_coo(rows, cols, vals.to(y.dtype), (2 * n0, n1))
        MT = csr_from_coo(cols, rows, vals.to(y.dtype), (n1, 2 * n0))
        z = spmm(Mx, y.contiguous(), MT)                                         # (2 n0, B, 1)
        return z.view(n0, 2, y.shape[1]).permute(0, 2, 1).contiguous()

    def to(self, device) -> "FluxVelocityMap":
        return self


def trapezoid_flux(pos: Tensor, edges: Tensor, u: Tensor) -> Tensor:
    """``F_e = (u_src + u_dst)/2 . n_e |e|`` with ``n`` the right normal of ``src -> dst``: ``(n1,)`` (or ``(n1, B)``
    for ``u`` of shape ``(n0, B, 2)``)."""
    ev = (pos[edges[:, 1]] - pos[edges[:, 0]]).to(u.dtype)
    um = 0.5 * (u[edges[:, 0]] + u[edges[:, 1]])
    if um.dim() == 3:
        ev = ev[:, None, :]
    return um[..., 0] * ev[..., 1] - um[..., 1] * ev[..., 0]


edge_flux_of_nodes = trapezoid_flux


def flux_divergence(K, F: Tensor) -> Tensor:
    """Net outward flux of every triangle, ``(d1 F)_f * sigma_f`` (``(n2,)`` or ``(n2, B)``): zero for an exactly
    conservative (closed) flux cochain."""
    sa = signed_areas(K)
    x = F if F.dim() > 1 else F[:, None]
    out = K.apply_d(1, x.to(K.pos.dtype).contiguous()) * torch.sign(sa).to(K.pos.dtype)[:, None]
    return out if F.dim() > 1 else out[:, 0]


# ======================================================================================================================
# T8 resolution views (test only)
# ======================================================================================================================
def load_t8res(name: str, droot: str, *, n_pts: int, native: bool = True, device="cuda", star: str = "cotan",
               max_samples: int | None = None, whitney: bool = True, keep_on: str = "auto", forms: bool = False,
               xy: bool = False, **_) -> TaskData:
    """Test-only view ``T8_n<n_pts>`` (300 test simulations; ``extra_tests['test100']`` = sources 850..949).

    ``xy`` / ``forms``: the inputs of ``T8xy`` / ``T8fr`` (chord-frame coordinates as node columns; odd edge 1-forms),
    for zero-shot tests of models trained on those tasks.  Inputs / targets are normalised with the T8 statistics
    (stored by the generator; the frame columns with statistics of these meshes); ``rhmp.train --eval-ckpt``
    re-normalises with the statistics of the evaluated run anyway."""
    blob = _load_blob(droot, f"T8res_n{n_pts}")
    meta_b = blob["meta"]
    N = len(blob["ptr0"]) - 1
    ids = list(range(N if max_samples is None else min(N, max_samples)))
    t0 = time.time()
    Ks = _build([(_sl(blob, "pos", i, 0), _sl(blob, "faces", i, 2)) for i in ids], star, False, whitney)
    xs = [_sl(blob, "x", i, 0).float() for i in ids]
    ys = [_sl(blob, "y", i, 0).float() for i in ids]
    st = meta_b["t8_stats"]
    x_stats = Stats(torch.tensor(st["x_mean"]).float(), torch.tensor(st["x_std"]).float().clamp_min(1e-6), "v1-numpy")
    y_stats = Stats(torch.tensor(st["y_mean"]).float(), torch.tensor(st["y_std"]).float().clamp_min(1e-6), "v1-numpy")
    if xy:
        xs = [torch.cat([x, K.pos.float()[:, :2]], 1) for x, K in zip(xs, Ks)]
        pst = feature_stats([K.pos.float()[:, :2] for K in Ks], unbiased=False)
        x_stats = Stats(torch.cat([x_stats.mean, pst.mean]), torch.cat([x_stats.std, pst.std]), "v1-numpy")
    raw = [{0: x_stats.normalize(x)} for x in xs]
    e_stats = None
    if forms:
        es = []
        for K, x in zip(Ks, xs):
            U, a = x[0, 1].double(), x[0, 2].double()
            ev = K.edge_vectors().double().cpu()
            es.append(torch.stack([ev @ torch.stack([U * torch.cos(a), U * torch.sin(a)]), ev[:, 0], ev[:, 1]], 1).float())
        e_stats = scale_stats(es)
        for d, e in zip(raw, es):
            d[1] = e_stats.normalize(e)
    on = keep_on if keep_on != "auto" else (str(device) if _nbytes(Ks) < GPU_BUDGET_GB else "cpu")
    Ks, inputs, target = pack_to((Ks, raw, [y_stats.normalize(y) for y in ys]), on)
    main = dict(K=Ks, inputs=inputs, target=target)
    loc = [j for j in meta_b["test100_local"] if j < len(ids)]
    dropped = sum(int(_sl(blob, "faces", i, 2).shape[0] - K.n[2]) for i, K in zip(ids, Ks))
    meta = dict(_meta_defaults(name), source=blob["_path"], generator=meta_b.get("generator"), test_only=True,
                n_points=n_pts, source_ids=[meta_b["source_ids"][i] for i in ids], t8_verify=meta_b.get("t8_verify"),
                construction=meta_b.get("construction"), build_seconds=time.time() - t0, dropped_faces=dropped,
                r2_std=y_stats.std, note="test-only view; evaluate a trained T8 run with rhmp.train --eval-ckpt "
                                         "(re-normalises with the run's statistics); test100 = extra_tests['test100']")
    in_dims, xst = {0: int(xs[0].shape[-1])}, {0: x_stats.to(device)}
    if forms:
        in_dims[1], xst[1] = 3, e_stats.to(device)
    return TaskData(name=name, K=Ks, inputs=inputs, target=target, target_degree=0, target_kind="node_scalar",
                    in_dims=in_dims, even_dims={}, connection_dims={}, split=_empty_split(len(Ks)),
                    x_stats=xst, y_stats=y_stats.to(device), spatial_dim=2, out_dim=1,
                    native=native, meta=meta, extra_tests={"test100": _subset(main, loc)})


# ======================================================================================================================
# T8 with the frame (equal-information variants)
# ======================================================================================================================
def load_t8_frame(name: str, droot: str, *, forms: bool, xy: bool, native: bool = True, device="cuda",
                  star: str = "cotan", cache: bool = True, max_samples: int | None = None, whitney: bool = True,
                  **_) -> TaskData:
    """T8 (meshes, split, target, normalisation and test100 of ``rhmp.tasks.paper.load_T8``) with the chord frame.

    ``forms``: odd edge 1-forms ``[U (cos a dx + sin a dy), dx_e, dy_e]`` (native only; scale-normalised with the RMS
    of the first 700 samples).  ``xy``: chord-frame coordinates (x, y) appended to the node inputs (every model; in
    legacy mode the node inputs are ``(sdf, U, a, x, y)``)."""
    from rhmp.tasks import paper
    if forms and not native:
        raise ValueError(f"{name}: the frame 1-forms are native cochain inputs (no legacy variant)")
    samples = paper._read_pickle(droot, "T8")
    N_all = len(samples)
    Ks, info = paper._t8_complexes(droot, samples, star, cache)
    x_all = [torch.as_tensor(np.asarray(s["x_pres"], dtype=np.float32)) for s in samples]
    y_all = [torch.as_tensor(np.asarray(s["y_pres"], dtype=np.float32)).reshape(-1, 1) for s in samples]
    if xy:
        x_all = [torch.cat([x, K.pos.float()[:, :2]], 1) for x, K in zip(x_all, Ks)]
    n_tr = int(0.7 * N_all)
    x_stats = feature_stats(x_all[:n_tr], unbiased=False)
    y_stats = feature_stats(y_all[:n_tr], unbiased=False)
    e_all, e_stats = None, None
    if forms:
        e_all = []
        for K, x in zip(Ks, x_all):
            U, a = x[0, 1].double(), x[0, 2].double()
            ev = K.edge_vectors().double().cpu()
            e_all.append(torch.stack([ev @ torch.stack([U * torch.cos(a), U * torch.sin(a)]), ev[:, 0], ev[:, 1]],
                                     1).float())
        e_stats = scale_stats(e_all[:n_tr])
    split = holdout_split(N_all)
    idx_keep = None
    if max_samples is not None and max_samples < N_all:
        per = max(1, max_samples // 3)
        idx_keep = torch.cat([s_[:per] for s_ in split])
        remap = {int(i): j for j, i in enumerate(idx_keep.tolist())}
        split = tuple(torch.tensor([remap[int(i)] for i in s_[:per].tolist()]) for s_ in split)
        keep = idx_keep.tolist()
        Ks, x_all, y_all = [Ks[i] for i in keep], [x_all[i] for i in keep], [y_all[i] for i in keep]
        if e_all is not None:
            e_all = [e_all[i] for i in keep]
    if not whitney:
        from rhmp.data import strip_whitney
        strip_whitney(Ks)
    raw = [{0: x_stats.normalize(x)} if e_all is None else {0: x_stats.normalize(x), 1: e_stats.normalize(e)}
           for x, e in zip(x_all, e_all if e_all is not None else [None] * len(x_all))]
    Ks, inputs, target = pack_to((Ks, raw, [y_stats.normalize(y) for y in y_all]), device)
    in_dims = {0: int(x_all[0].shape[-1])}
    xs = {0: x_stats.to(device)}
    if e_all is not None:
        in_dims[1] = 3
        xs[1] = e_stats.to(device)
    meta = dict(_meta_defaults(name), complexes=info, r2_std=y_stats.std,
                split_note="T8 partition: 0..629 train, 630..699 validation, 700..999 test; statistics of 0..699",
                paper_eval=(torch.arange(*TEST100) if idx_keep is None else split[2]),
                native_inputs=("node (sdf, U, a" + (", x, y" if xy else "") + ")" +
                               ("; edge 1-forms U (cos a dx + sin a dy), dx, dy" if forms else "")))
    return TaskData(name=name, K=Ks, inputs=inputs, target=target, target_degree=0, target_kind="node_scalar",
                    in_dims=in_dims, even_dims={}, connection_dims={}, split=split, x_stats=xs,
                    y_stats=y_stats.to(device), spatial_dim=2, out_dim=1, native=native, meta=meta)


def load_t8_inflow(name: str, droot: str, *, native: bool = True, device="cuda", star: str = "cotan",
                   cache: bool = True, max_samples: int | None = None, whitney: bool = True, **_) -> TaskData:
    """``T8i``: T8 (meshes, split, target, normalisation and test100 of ``rhmp.tasks.paper.load_T8``) with the inputs of
    ``AFPi``: node (sdf, a), edge inflow 1-form ``theta_e = u_inf . (x_dst - x_src)`` (odd, the amplitude column) and
    the mean sdf of the edge / face vertices (even), and the fixed output map of :func:`afp_output_map`.  The T8 meshes
    are Delaunay triangulations of the convex hull of random CFD nodes: the airfoil is not a boundary of the complex,
    so there is no wall condition and no cohomology generator (projection layers fix the hull vertices).  Native
    only (the node baselines use ``T8``)."""
    from rhmp.tasks import paper
    if not native:
        raise ValueError(f"{name} is a native task (node baselines: T8)")
    samples = paper._read_pickle(droot, "T8")
    N_all = len(samples)
    Ks, info = paper._t8_complexes(droot, samples, star, cache)
    x_all = [torch.as_tensor(np.asarray(s["x_pres"], dtype=np.float32)) for s in samples]
    y_all = [torch.as_tensor(np.asarray(s["y_pres"], dtype=np.float32)).reshape(-1, 1) for s in samples]
    n_tr = int(0.7 * N_all)
    nodes, edges, faces, U2 = [], [], [], []
    for K, x in zip(Ks, x_all):
        U, a = x[0, 1].double(), x[0, 2].double()
        ev = K.edge_vectors().double().cpu()
        sdf = x[:, :1]
        e, f = K.cells[1].long().cpu(), K.cells[2].long().cpu()
        theta = (ev @ torch.stack([U * torch.cos(a), U * torch.sin(a)])).float()[:, None]
        nodes.append(torch.cat([sdf, x[:, 2:3]], 1))
        edges.append(torch.cat([theta, sdf[e].mean(1)], 1))
        faces.append(sdf[f].mean(1))
        U2.append(float(U) ** 2)
    x_stats = {0: feature_stats(nodes[:n_tr], unbiased=False),
               1: _mixed_stats(edges[:n_tr], [("scale", 1), ("v1", 1)]),
               2: feature_stats(faces[:n_tr], unbiased=False)}
    y_stats = feature_stats(y_all[:n_tr], unbiased=False)
    u2_ref = float(np.mean(U2[:n_tr]))
    split = holdout_split(N_all)
    idx_keep = None
    if max_samples is not None and max_samples < N_all:
        per = max(1, max_samples // 3)
        idx_keep = torch.cat([s_[:per] for s_ in split])
        remap = {int(i): j for j, i in enumerate(idx_keep.tolist())}
        split = tuple(torch.tensor([remap[int(i)] for i in s_[:per].tolist()]) for s_ in split)
        keep = idx_keep.tolist()
        Ks, y_all = [Ks[i] for i in keep], [y_all[i] for i in keep]
        nodes, edges, faces = [nodes[i] for i in keep], [edges[i] for i in keep], [faces[i] for i in keep]
    if not whitney:
        from rhmp.data import strip_whitney
        strip_whitney(Ks)
    raw = [{0: x_stats[0].normalize(n_), 1: x_stats[1].normalize(e_), 2: x_stats[2].normalize(f_)}
           for n_, e_, f_ in zip(nodes, edges, faces)]
    Ks, inputs, target = pack_to((Ks, raw, [y_stats.normalize(y) for y in y_all]), device)
    omap = OffsetMap(-float(y_stats.mean[0] / y_stats.std[0]), float(x_stats[1].std[0]) ** 2 / u2_ref)
    meta = dict(_meta_defaults(name), complexes=info, r2_std=y_stats.std, u2_ref=u2_ref,
                split_note="T8 partition: 0..629 train, 630..699 validation, 700..999 test; statistics of 0..699",
                paper_eval=(torch.arange(*TEST100) if idx_keep is None else split[2]),
                native_inputs="node (sdf, a); edge inflow 1-form (odd), mean sdf (even); face mean sdf (even)")
    return TaskData(name=name, K=Ks, inputs=inputs, target=target, target_degree=0, target_kind="node_scalar",
                    in_dims={0: 2, 1: 2, 2: 1}, even_dims={1: 1, 2: 1}, connection_dims={}, split=split,
                    x_stats={k: v.to(device) for k, v in x_stats.items()}, y_stats=y_stats.to(device), spatial_dim=2,
                    out_dim=1, native=native, meta=meta, output_map=omap)


# ======================================================================================================================
# conforming near-field meshes (AFM)
# ======================================================================================================================
def _rotation(deg: float) -> Tensor:
    t = float(np.deg2rad(deg))
    return torch.tensor([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]], dtype=torch.float64)


def potential_reference(K, r: dict) -> Tensor:
    """Bernoulli pressure ``(U^2 - |u|^2) / 2`` of the P1 potential flow around the airfoil with the Kutta condition,
    without any learning (the reference of ``scripts/airfrans_potential_reference.py`` computed with the library's
    operators in float64): the inflow 1-form is projected with the reference star (no flux through the wall,
    perturbation potential zero on the outer ring), the circulation along the harmonic cochain is fixed by the Kutta
    condition on the two wall edges at the trailing-edge vertex (the wall vertex of largest chord-frame x), and
    ``|u|^2`` is the Galerkin energy density at the vertices.  ``K`` must carry its Whitney blocks.  ``(n0,)``."""
    from rhmp.dec import harmonic_generators, vertex_energy
    from rhmp.layers import direct_projection, make_context
    f64 = torch.float64
    Kd = K.to(dtype=f64)
    ctx = make_context(Kd, scaling="dec", B=1, dtype=f64)
    V = r["rot"] @ torch.tensor([r["U"] * np.cos(r["aoa"]), r["U"] * np.sin(r["aoa"])], dtype=f64)
    th = Kd.edge_vectors() @ V
    gam = harmonic_generators(K, dtype=f64)[:, 0]
    free = (r["kind"] != KIND_OUTER).to(f64).view(-1, 1, 1)
    _, dal = direct_projection(ctx, torch.stack([th, gam], -1)[:, None, :], Kd.star[1][:, None], free)
    u0, gp = th - dal[:, 0, 0], gam - dal[:, 0, 1]
    e = K.cells[1].long()
    wall = r["kind"] == KIND_WALL
    te = torch.where(wall)[0][torch.argmax(r["pos0"][wall, 0])]
    we = wall[e[:, 0]] & wall[e[:, 1]] & K.boundary[1].bool() & ((e[:, 0] == te) | (e[:, 1] == te))
    W = we.to(f64) / Kd.edge_vectors().square().sum(1)
    u = u0 - (W * gp * u0).sum() / (W * gp * gp).sum() * gp
    q = vertex_energy(Kd, u[:, None], u[:, None])[:, 0]
    return 0.5 * (r["U"] ** 2 - q)


def _afm_raw(blob: dict, ids: list[int], star: str, whitney: bool, kind_check: bool = True, rotate: float = 0.0,
             topo: bool = False, reference: bool = False) -> tuple[list, list]:
    """CPU complexes and the per-sample raw fields aligned to the complexes (edge cochains in the complex's
    orientation).  ``rotate`` (degrees): the meshes are rotated rigidly about the origin (``r['rot']`` holds the
    rotation, ``r['pos0']`` the chord-frame positions).  ``topo``: the complexes get ``meta['dirichlet'] = {0: outer
    ring}`` and ``meta['cohomology'] = {1: harmonic generator of H^1}`` (:func:`rhmp.dec.harmonic_generators`)."""
    R = _rotation(rotate)
    meshes = [((_sl(blob, "pos", i, 0).double() @ R.T) if rotate else _sl(blob, "pos", i, 0), _sl(blob, "faces", i, 2))
              for i in ids]
    Ks = _build(meshes, star, True, whitney or reference)
    U_all, a_all = blob["meta"]["U"], blob["meta"]["aoa"]
    raws = []
    for i, K in zip(ids, Ks):
        n0 = K.n[0]
        gen_edges = _sl(blob, "edges", i, 1).long()
        perm, sign = edge_alignment(K.cells[1], gen_edges, n0)
        fperm, _ = cell_alignment(K.cells[2], _sl(blob, "faces", i, 2).long(), n0, oriented=False)
        r = dict(U=float(U_all[i]), aoa=float(a_all[i]), src=int(blob["meta"]["source_ids"][i]))
        for key in ("u", "p", "omega", "nut", "sdf"):
            r[key] = _sl(blob, key, i, 0).double()
        r["kind"] = _sl(blob, "kind", i, 0).long()
        for key in ("theta", "flux", "theta_trap", "flux_trap"):
            r[key] = _sl(blob, key, i, 1).double()[perm] * sign.double()
        r["nut_edge"] = _sl(blob, "nut_edge", i, 1).double()[perm]
        r["nut_face"] = _sl(blob, "nut_face", i, 2).double()[fperm]
        r["pos0"], r["rot"] = _sl(blob, "pos", i, 0).double(), R
        if kind_check and not rotate and not torch.equal(wall_vertices(K).cpu(), r["kind"] == KIND_WALL):
            raise RuntimeError(f"sample {i}: geometric wall test disagrees with the generator's vertex kinds")
        if topo:
            from rhmp.dec import harmonic_generators
            K.meta["dirichlet"] = {0: (r["kind"] == KIND_OUTER)}
            K.meta["cohomology"] = {1: harmonic_generators(K)}
        if reference:                                # no-learning physics reference as a node field
            r["p_ref"] = potential_reference(K, r)
            if not whitney:
                K.whitney = {}
        raws.append(r)
    return Ks, raws


def _afm_target(task: str, K, r: dict) -> Tensor:
    if task in ("AFV", "AFV_trap"):
        th = r["theta"] if task == "AFV" else r["theta_trap"]
        c = K.apply_d(1, th[:, None].to(K.pos.dtype)).double()          # (n2, 1) circulation, face orientation
        y = StarAverageMap(1, 1.0)(c[:, None, :], K)[:, 0]              # (n0, 1), physical units (1/s)
        return y.float()
    if task in ("AFVp", "AFVp_trap"):
        return r["omega"][:, None].float()
    if task == "AFU":
        return r["flux"][:, None].float()
    if task == "AFUv" or task == "AFUvel":
        u = r["u"].float().clone()
        return u
    if task == "AFD":
        return r["T"][:, None].float()
    if task.startswith("AFPcp"):                                 # pressure coefficient p / (U^2 / 2)
        return (r["p"] / (0.5 * r["U"] ** 2))[:, None].float()
    if task.startswith("AFP"):
        return r["p"][:, None].float()
    raise KeyError(task)


def _afm_inputs(task: str, K, r: dict, native: bool, nu: float) -> tuple[dict, dict, dict]:
    """Raw inputs ``{k: (n_k, F_k)}``, their normalisation kinds and the even column counts."""
    n0 = K.n[0]
    one = torch.ones(n0, 1, dtype=torch.float64)
    sdf = r["sdf"][:, None]
    Ua = torch.cat([one * r["U"], one * r["aoa"]], 1)
    pos = r.get("pos0", K.pos.double()[:, :2])                 # chord-frame coordinates (scalar inputs)
    if task in ("AFV", "AFVp", "AFV_trap", "AFVp_trap"):
        th = r["theta_trap"] if task.endswith("_trap") else r["theta"]
        if native:
            return {0: torch.cat([sdf, Ua], 1), 1: th[:, None]}, {0: "v1", 1: "scale"}, {}
        return {0: torch.cat([r["u"], sdf, Ua], 1)}, {0: "v1"}, {}
    if task in ("AFU", "AFUv"):
        node = torch.cat([sdf, Ua, pos], 1)
        if native:
            V = torch.stack([torch.tensor(r["U"] * np.cos(r["aoa"])), torch.tensor(r["U"] * np.sin(r["aoa"]))]).double()
            inflow = (K.edge_vectors().double().cpu() @ V)[:, None]
            return {0: node, 1: inflow}, {0: "v1", 1: "scale"}, {}
        return {0: node}, {0: "v1"}, {}
    if task in AFP_NATIVE:
        if not native:                                           # node baselines: (sdf, U, a) [+ chord-frame x, y]
            return {0: torch.cat([sdf, Ua] + ([r["pos0"]] if task.endswith("xy") else []), 1)}, {0: "v1"}, {}
        vcols, inflow = AFP_NATIVE[task]
        col = dict(sdf=sdf, a=one * r["aoa"], U=one * r["U"], logU=one * float(np.log(r["U"])))
        node = torch.cat([col[c] for c in vcols], 1)
        e, f = K.cells[1].long().cpu(), K.cells[2].long().cpu()
        sdf_e = sdf[e].mean(1)                                   # (n1, 1) even: mean over the end vertices
        sdf_f = sdf[f].mean(1)                                   # (n2, 1) even: mean over the face vertices
        k0 = [("scale", 1), ("v1", len(vcols) - 1)] if vcols[0] == "U" else "v1"
        if inflow:
            V = r["rot"] @ torch.tensor([r["U"] * np.cos(r["aoa"]), r["U"] * np.sin(r["aoa"])], dtype=torch.float64)
            theta = (K.edge_vectors().double().cpu() @ V)[:, None]
            return ({0: node, 1: torch.cat([theta, sdf_e], 1), 2: sdf_f},
                    {0: k0, 1: [("scale", 1), ("v1", 1)], 2: "v1"}, {1: 1, 2: 1})
        return {0: node, 1: sdf_e, 2: sdf_f}, {0: k0, 1: "v1", 2: "v1"}, {1: 1, 2: 1}
    if task == "AFPpf":                                          # every model: (sdf, U, a, reference pressure)
        return {0: torch.cat([sdf, Ua, r["p_ref"][:, None]], 1)}, {0: "v1"}, {}
    if task.startswith("AFP"):
        xy = task in ("AFPxy", "AFPfrxy")
        node = torch.cat([sdf, Ua] + ([pos] if xy else []), 1)
        if native and task in ("AFPfr", "AFPfrxy"):
            ev = K.edge_vectors().double().cpu() @ r["rot"]          # edge vectors in the chord frame
            V = torch.tensor([r["U"] * np.cos(r["aoa"]), r["U"] * np.sin(r["aoa"])], dtype=torch.float64)
            return {0: node, 1: torch.stack([ev @ V, ev[:, 0], ev[:, 1]], 1)}, {0: "v1", 1: "scale"}, {}
        return {0: node}, {0: "v1"}, {}
    if task == "AFD":
        lk_e = torch.log1p(r["nut_edge"] / nu)[:, None]                  # log kappa, kappa = 1 + nu_t / nu
        lk_f = torch.log1p(r["nut_face"] / nu)[:, None]
        f = r["f"][:, None]
        if native:
            return {0: f, 1: lk_e, 2: lk_f}, {0: "v1", 1: "v1", 2: "v1"}, {1: 1, 2: 1}
        e = K.cells[1].long().cpu()
        s = torch.zeros(n0, 1, dtype=torch.float64).index_add_(0, e[:, 0], lk_e).index_add_(0, e[:, 1], lk_e)
        c = torch.zeros(n0, dtype=torch.float64).index_add_(0, e[:, 0], torch.ones(len(e), dtype=torch.float64)) \
            .index_add_(0, e[:, 1], torch.ones(len(e), dtype=torch.float64))
        return {0: torch.cat([f, s / c.clamp_min(1)[:, None]], 1)}, {0: "v1"}, {}
    raise KeyError(task)


_AFM_SPEC = {
    # task: (target_degree, target_kind, readout (native), output map, connection dims (native))
    "AFV": (0, "node_scalar", None, "star", {1: 1}),
    "AFV_trap": (0, "node_scalar", None, "star", {1: 1}),
    "AFVp": (0, "node_scalar", None, "star", {1: 1}),
    "AFVp_trap": (0, "node_scalar", None, "star", {1: 1}),
    "AFU": (1, "cochain", "grad", None, {}),
    "AFUv": (0, "node_scalar", "grad", "flux_velocity", {}),
    "AFD": (0, "node_scalar", None, None, {}),
    "AFP": (0, "node_scalar", None, None, {}),
    "AFPfr": (0, "node_scalar", None, None, {}),
    "AFPxy": (0, "node_scalar", None, None, {}),
    "AFPfrxy": (0, "node_scalar", None, None, {}),
    "AFPi": (0, "node_scalar", None, "offset", {}),
    "AFPi0": (0, "node_scalar", None, "offset", {}),
    "AFPiU": (0, "node_scalar", None, None, {}),
    "AFPiR": (0, "node_scalar", None, "offset", {}),
    "AFPs": (0, "node_scalar", None, "offset", {}),
    "AFPcp": (0, "node_scalar", None, None, {}),
    "AFPcpxy": (0, "node_scalar", None, None, {}),
    "AFPpf": (0, "node_scalar", None, None, {}),
}

# native inputs of the covariant pressure tasks: (vertex columns, inflow 1-form on the edges)
AFP_NATIVE: dict[str, tuple] = {
    "AFPi": (("sdf", "a"), True), "AFPi0": (("sdf",), True), "AFPiU": (("sdf", "U", "a"), True),
    "AFPiR": (("sdf", "a", "logU"), True), "AFPs": (("U", "sdf", "a"), False),
    "AFPcp": (("sdf", "a"), True), "AFPcpxy": (("sdf", "a"), True),
}


def _mixed_stats(xs: list, kind) -> Stats:
    """Statistics of the columns of ``xs`` (list of ``(n_i, F)``): ``'scale'`` / ``'v1'`` for all columns, or a list of
    ``(kind, n_columns)`` groups in column order."""
    if isinstance(kind, str):
        return scale_stats(xs) if kind == "scale" else feature_stats(xs, unbiased=False)
    parts, lo = [], 0
    for kd, n in kind:
        if n > 0:
            parts.append(_mixed_stats([x[:, lo:lo + n] for x in xs], kd))
        lo += n
    return _cat_stats(parts, "+".join(kd for kd, n in kind if n > 0))


def load_afm(name: str, droot: str, *, task: str, native: bool = True, device="cuda", star: str = "cotan",
             fine: bool = True, max_samples: int | None = None, whitney: bool = True, keep_on: str = "auto",
             connection: bool = True, output_map=None, rotate: float = 0.0, **_) -> TaskData:
    """Tasks on the conforming near-field meshes (see the module docstring).

    Args:
        task: ``'AFV' | 'AFVp' | 'AFV_trap' | 'AFU' | 'AFUv' | 'AFD'``.
        connection: AFV*: the circulation 1-form enters as a gauge connection (only ``d1 theta`` reaches the model);
            ``False``: as an ordinary odd edge input.
        fine: also load ``extra_tests['fine']`` / ``['fine_test100']`` (4x meshes of the test simulations).
    """
    t_all = time.time()
    if task == "AFU" and not native:
        task_eff = "AFUvel"                         # node baselines: node velocities (fluxes for the evaluation)
    elif task == "AFUv" and not native:
        task_eff = "AFUvel"
    else:
        task_eff = task
    tdeg, tkind, readout, omap_kind, cdims = _AFM_SPEC[task]
    if task_eff == "AFUvel":
        tdeg, tkind, readout, omap_kind, cdims = 0, "node_scalar", None, None, {}
    if not native:
        readout, omap_kind, cdims = None, None, {}
    if output_map is False and omap_kind is not None:      # ablation: the model predicts the target directly
        readout = "node_scalar" if tkind == "node_scalar" else readout
        omap_kind = None
    if task.startswith("AFV") and not connection:
        cdims = {}
    blob = _load_blob(droot, "AFM")
    N = len(blob["ptr0"]) - 1
    nu = float(blob["meta"]["NU"])
    split = holdout_split(N)
    ids = list(range(N))
    if max_samples is not None and max_samples < N:
        per = max(1, max_samples // 3)
        parts = [s_[:per] for s_ in split]
        ids = torch.cat(parts).tolist()
        off = np.cumsum([0] + [len(p) for p in parts])
        split = tuple(torch.arange(off[j], off[j + 1]) for j in range(3))
    diff = _load_blob(droot, "AFD") if task == "AFD" else None
    topo = native and task in AFP_NATIVE                    # Dirichlet ring and cohomology generator (projection)
    Ks, raws = _afm_raw(blob, ids, star, whitney, rotate=rotate, topo=topo, reference=task == "AFPpf")
    if diff is not None:
        _attach_diffusion(diff, ids, raws)
    xs_raw, kinds, even = [], None, None
    ys_raw = []
    for K, r in zip(Ks, raws):
        x, kinds, even = _afm_inputs(task, K, r, native, nu)
        xs_raw.append(x)
        ys_raw.append(_afm_target(task_eff, K, r))
    degs = sorted(xs_raw[0])
    n_stat = [j for j, i in enumerate(ids) if i < N_TRAIN_STATS] or list(range(len(ids)))

    def stats(k):
        return _mixed_stats([xs_raw[j][k].float() for j in n_stat], kinds[k])
    x_stats = {k: stats(k) for k in degs}
    if tkind == "cochain" or (task_eff.startswith("AFV") and native):
        # odd targets, and the vorticity (a pseudo-scalar predicted through an oriented map of an odd face cochain):
        # scale-only statistics (a mean offset is not in the range of the orientation-equivariant model)
        y_stats = scale_stats([ys_raw[j] for j in n_stat])
    elif task_eff == "AFUv":                       # vertex velocities from fluxes: one isotropic scale, no offset
        y_stats = scale_stats([ys_raw[j] for j in n_stat], isotropic=True)
    else:
        y_stats = feature_stats([ys_raw[j] for j in n_stat], unbiased=False)
    on = keep_on if keep_on != "auto" else (str(device) if _nbytes(Ks) < GPU_BUDGET_GB else "cpu")
    norm_x = [{k: x_stats[k].normalize(x[k].float()) for k in degs} for x in xs_raw]
    norm_y = [y_stats.normalize(y) for y in ys_raw]
    Ks, inputs, target = pack_to((Ks, norm_x, norm_y), on)
    extra = {}
    for label in (("fine", "coarse") if fine else ()):
        base_name = "AFM_" + label
        if not os.path.exists(os.path.join(droot, "v2", base_name + ".pt")):
            continue
        fb = _load_blob(droot, base_name)
        if diff is not None and not os.path.exists(os.path.join(droot, "v2", "AFD_" + label + ".pt")):
            continue
        nf = len(fb["ptr0"]) - 1
        fids = list(range(nf if max_samples is None else min(nf, max_samples)))
        fK, fr = _afm_raw(fb, fids, star, whitney, rotate=rotate, topo=topo, reference=task == "AFPpf")
        if diff is not None:
            _attach_diffusion(_load_blob(droot, "AFD_" + label), fids, fr)
        fx, fy = [], []
        for K, r in zip(fK, fr):
            x, _, _ = _afm_inputs(task, K, r, native, nu)
            fx.append({k: x_stats[k].normalize(x[k].float()) for k in degs})
            fy.append(y_stats.normalize(_afm_target(task_eff, K, r)))
        src = dict(K=fK, inputs=fx, target=fy)
        extra[label] = src
        srcs = [int(s_) for s_ in fb["meta"]["source_ids"][:len(fids)]]
        loc = [j for j, s_ in enumerate(srcs) if TEST100[0] <= s_ < TEST100[1]]
        if loc:
            extra[label + "_test100"] = _subset(src, loc)
    omap = None
    u2_ref = float(np.mean([raws[j]["U"] ** 2 for j in n_stat]))        # mean squared inflow speed (training part)
    a_ref = float(np.median(np.concatenate([StarAverageMap.weights(Ks[j])[2].cpu().numpy() for j in n_stat[:100]])))
    if omap_kind == "star":
        omap = StarAverageMap(1, a_ref)
    elif omap_kind == "offset":                             # homogeneous model output in physical units
        omap = afp_output_map(task, x_stats, y_stats, u2_ref)
    elif omap_kind == "flux_velocity":
        # unit factor = RMS accurate flux / velocity scale of the training meshes (a length): the potential of the grad
        # readout then has the units of the flux-normalised AFU target
        f_rms = float(np.sqrt(np.mean(np.concatenate([raws[j]["flux"].numpy() ** 2 for j in n_stat]))))
        omap = FluxVelocityMap("grad", scale=f_rms / float(y_stats.std.mean()))
    in_dims = {k: int(xs_raw[0][k].shape[-1]) for k in degs}
    meta = dict(_meta_defaults(name), source=blob["_path"], generator=blob["meta"].get("generator"),
                r2_std=y_stats.std, star=star, afm_task=task_eff, nu=nu,
                split_note="T8 partition: 0..629 train, 630..699 validation, 700..999 test; statistics of 0..699",
                paper_eval=(torch.arange(*TEST100) if max_samples is None else split[2]),
                sample_ids=ids, load_seconds=time.time() - t_all, mean_star_area=a_ref,
                mesh_params=blob["meta"].get("params"), cp_target=task.startswith("AFPcp"), rotate=float(rotate),
                u2_ref=u2_ref)
    if task == "AFD" and native:
        meta["pde"] = dict(degree=0, source="inputs[0][...,0]", bc="dirichlet")
        meta["metric_ref"] = "1:0"
    if readout is not None and omap is None:
        meta["model_readout"] = readout
    return TaskData(name=name, K=Ks, inputs=inputs, target=target, target_degree=tdeg, target_kind=tkind,
                    in_dims=in_dims, even_dims=dict(even), connection_dims=dict(cdims), split=split,
                    x_stats={k: s_.to(device) for k, s_ in x_stats.items()}, y_stats=y_stats.to(device),
                    spatial_dim=2, out_dim=int(ys_raw[0].shape[-1]), native=native, meta=meta, extra_tests=extra,
                    output_map=omap, readout=(readout if (readout is not None and omap is None) else None))


def afp_output_map(task: str, x_stats: dict, y_stats: Stats, u2_ref: float):
    """The fixed output map of a covariant pressure task for given statistics (``None`` for tasks without one).

    Unit factor: the amplitude column (the inflow 1-form, or U for AFPs) is normalised by its RMS ``c`` over the
    training cells, so the output factor of the model is ``(U / c)^2``; with ``scale = c^2 / u2_ref`` (``u2_ref`` = the
    mean squared inflow speed of the training simulations) it becomes ``U^2 / u2_ref``, of order one."""
    if _AFM_SPEC.get(task, (None,) * 5)[3] != "offset":
        return None
    amp_deg = 1 if AFP_NATIVE[task][1] else 0
    return OffsetMap(-float(y_stats.mean[0] / y_stats.std[0]), float(x_stats[amp_deg].std[0]) ** 2 / float(u2_ref))


def afm_eval_source(task: str, droot: str, x_stats: dict, y_stats: Stats, *, base: str = "AFM", ids=None,
                    native: bool = True, star: str = "cotan", whitney: bool = False, rotate: float = 0.0) -> dict:
    """One evaluation set of a conforming-mesh task, built on the CPU with the statistics of a trained run.

    Args:
        task: AFM task name; droot: datasets directory; x_stats / y_stats: the run's statistics (CPU).
        base: ``'AFM'`` (ids = simulation indices), ``'AFM_fine'`` or ``'AFM_coarse'`` (ids = positions in that file).
        ids: samples of the file (default: the simulations 850..949, "test100").
        rotate: rigid rotation of the whole configuration in degrees (mesh and inflow; scalar inputs unchanged).
    Returns:
        dict with ``K``, ``inputs``, ``target`` (lists, normalised), ``kind`` (vertex kinds: 0 volume, 1 wall, 2 outer
        ring), ``pos0`` (chord-frame positions), ``U``, ``aoa``, ``src`` (simulation indices), and the raw CFD fields
        ``theta_cfd`` (accurate edge circulations of the velocity), ``p`` and ``sdf``.
    """
    blob = _load_blob(droot, base)
    srcs = [int(v) for v in blob["meta"]["source_ids"]]
    if ids is None:
        ids = [j for j, s_ in enumerate(srcs) if TEST100[0] <= s_ < TEST100[1]]
    ids = [int(i) for i in ids]
    nu = float(blob["meta"]["NU"])
    task_eff = "AFUvel" if (task in ("AFU", "AFUv") and not native) else task
    Ks, raws = _afm_raw(blob, ids, star, whitney, rotate=rotate, topo=native and task in AFP_NATIVE,
                        reference=task == "AFPpf")
    xs, ys = [], []
    for K, r in zip(Ks, raws):
        x, _, _ = _afm_inputs(task, K, r, native, nu)
        xs.append({k: x_stats[k].normalize(v.float()) for k, v in x.items()})
        ys.append(y_stats.normalize(_afm_target(task_eff, K, r)))
    return dict(K=Ks, inputs=xs, target=ys, kind=[r["kind"] for r in raws], pos0=[r["pos0"] for r in raws],
                U=[r["U"] for r in raws], aoa=[r["aoa"] for r in raws], src=[srcs[i] for i in ids],
                theta_cfd=[r["theta"] for r in raws], p=[r["p"] for r in raws], sdf=[r["sdf"] for r in raws])


def _attach_diffusion(blob: dict, ids: list[int], raws: list[dict]) -> None:
    for i, r in zip(ids, raws):
        if int(blob["meta"]["source_ids"][i]) != r["src"]:
            raise RuntimeError("AFD / AFM sample order differs")
        r["f"] = _sl(blob, "f", i, 0).double()
        r["T"] = _sl(blob, "T", i, 0).double()


# ======================================================================================================================
# registry
# ======================================================================================================================
def _loader(fn, **fixed):
    @functools.wraps(fn)
    def load(droot=None, **kw):
        from rhmp.tasks import resolve_root
        return fn(droot=resolve_root(droot), **{**fixed, **kw})
    return load


AIRFRANS_TASKS: dict = {
    **{f"T8_n{n}": _loader(load_t8res, name=f"T8_n{n}", n_pts=n) for n in (500, 2000, 8000)},
    **{f"T8{v}_n{n}": _loader(load_t8res, name=f"T8{v}_n{n}", n_pts=n, forms="fr" in v, xy="xy" in v)
       for v in ("fr", "xy", "frxy") for n in (500, 2000, 8000)},
    "T8fr": _loader(load_t8_frame, name="T8fr", forms=True, xy=False),
    "T8xy": _loader(load_t8_frame, name="T8xy", forms=False, xy=True),
    "T8frxy": _loader(load_t8_frame, name="T8frxy", forms=True, xy=True),
    "T8i": _loader(load_t8_inflow, name="T8i"),
    **{t: _loader(load_afm, name=t, task=t) for t in _AFM_SPEC},
    "AFVo": _loader(load_afm, name="AFVo", task="AFV", connection=False),
    "AFVpo": _loader(load_afm, name="AFVpo", task="AFVp", connection=False),
}
