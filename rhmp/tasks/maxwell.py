"""Nonlinear magnetostatics task family MXS (generator ``datasets/generators/gen_maxwell.py``).

Physics: 2-D cross-section of a three-limb EI-core electromagnet with saturable steel (Froehlich-Kennelly law,
``mu_r,init = 3000``, ``J_s = 2.0 T``), three windings with independent signed current densities (|J| <= 6 A/mm^2),
a movable and possibly tilted I-armature, random pole shoes and 0-2 external iron pieces, all on **one fixed mesh**
(2,085 nodes; the movable iron is a per-face material indicator, i.e. staircase interfaces on a regular lattice band).
Reference: P1 finite elements with damped Newton (``|R| <= 1e-10 |F|``) for ``-div(nu(|B|) grad A) = J``, ``A = 0`` on
the box, ``B = (dA/dy, -dA/dx)``.  Units: mm, T, A/mm^2.  Split: sequential 70/15/15 of 5000 samples (train
0..3499, validation 3500..4249, test 4250..4999), ``meta['paper_eval']`` = 4250..4349 ("test100").

==========  ========================================================================================================
name        content
==========  ========================================================================================================
MXS         target: vertex flux density ``B_i`` (T, 2 components) = area-weighted mean over the incident triangles of
            the element fields ``B_f``.  native (RHMP): vertex current density ``J_i`` (lumped: ``M J = F``, the
            consistent load of the element-wise current), even edge columns = phase fractions of the incident faces
            (air, copper, iron), even face columns = phase one-hot (material columns for solver mode); readout
            ``grad`` (``Phi = d0 phi``: edge fluxes of B, exactly closed, ``d1 Phi = 0``) followed by the fixed map
            :class:`FluxToNodeField` (``B_i = R_-90`` of the area-weighted mean element gradient).  legacy (node
            baselines): vertex ``(J_i, f_air, f_copper, f_iron)`` with ``f`` the phase fractions of the barycentric dual
            cell -> vertex ``(B_x, B_y)``.  Target scaling: one isotropic RMS (rotation-consistent) in both modes.
==========  ========================================================================================================

Secondary family ``MXE`` (:func:`load_mxe`): linear electrostatics of an electrical capacitance tomography sensor
(eight electrodes at random potentials as holes of the mesh, grounded screen, insulating pipe wall, oil / water / gas
flow pattern with permittivity contrast 80, space charge), target ``E = -grad phi`` at the vertices; same split,
extra sets and protocol; the solve layer takes the electrode potentials as inhomogeneous Dirichlet data
(``solve_bc='lift'``).

Anisotropic family ``MXA`` (:func:`load_mxa`, generator ``datasets/generators/gen_maxwell_aniso.py``): grain-oriented
(easy : hard permeability 25) and laminated (5) steel with a rolling direction per piece, non-oriented steel, E-I / U-I
electromagnets and shunt transformers with random dimensions, gaps, constructions and windings on one fixed lattice
mesh; models receive phase indicators and the rolling direction (RHMP: E(n)-invariant edge columns, baselines: frame
components), never the law; extra sets ``ood15``, ``fine``, ``coarse`` and ``heldout`` (a layout family absent from
training).

Extra test sets (``extra_tests``; list format, one complex per sample): ``ood15`` (test samples 4250..4749 with all
currents x1.5, base mesh), ``fine`` (the same 500 samples on a refined mesh, 7,953 nodes) and ``coarse`` (566 nodes);
each also as ``<name>_test100`` (its first 100 samples = the test100 instances).  No model ever sees the reluctivity:
the constitutive law has to be learned from the field data.
"""
from __future__ import annotations

import functools
import math
import os
import time
import weakref

import numpy as np
import torch
from torch import Tensor

from rhmp.data import Stats, TaskData, feature_stats, pack_to, scale_stats, sequential_split

__all__ = ["MXS_TASKS", "MXS_DEFAULTS", "mxs_task_defaults", "FluxToNodeField", "reluctivity", "load_mxs", "load_mxe",
           "load_mxa", "PHASE_NAMES", "ECT_PHASES", "MXA_PHASES", "face_onehot", "triangle_areas", "edge_directions",
           "rotate_frame_columns", "MU0", "NU0", "JS", "MUR_INIT", "HK"]

MU0 = 4e-7 * math.pi
NU0 = 1.0 / MU0
JS = 2.0                                  # saturation polarisation of the steel (T)
MUR_INIT = 3000.0                         # initial relative permeability
HK = JS / (MU0 * (MUR_INIT - 1.0))        # Froehlich-Kennelly knee field (A/m)
PHASE_NAMES = ("air", "copper", "iron")
MAP_LENGTH_MM = 100.0                     # unit length of FluxToNodeField (potential head output ~ O(1))
N_TRAIN = 3500
TEST100 = (4250, 4350)

MXS_DEFAULTS: dict[str, dict] = {"MXS": dict(C=128, layers=4, batch=64)}


def mxs_task_defaults(name: str) -> dict:
    """Training defaults (``C``, ``layers``, ``batch``)."""
    return dict(MXS_DEFAULTS["MXS"])


def reluctivity(B):
    """Reluctivity ``nu = H / B`` (m/H) of the steel (numpy or torch, ``B >= 0`` in T): the explicit inverse of the
    Froehlich-Kennelly law ``B = mu0 H + Js H / (Hk + H)`` (identical to the generator's ``nu_iron``)."""
    lib = torch if isinstance(B, Tensor) else np
    b = MU0 * HK + JS - B
    return 2.0 * HK / (b + lib.sqrt(b * b + 4.0 * MU0 * HK * B))


# ======================================================================================================================
# geometry helpers
# ======================================================================================================================
def triangle_areas(K) -> Tensor:
    """``(n2,)`` unsigned triangle areas (float64, units of ``K.pos`` squared)."""
    F = K.cells[2].long()
    P = K.pos.double()[F]
    return 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) -
                  (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])).abs()


def face_onehot(phase: Tensor) -> Tensor:
    """``(..., n2)`` int phase -> ``(..., n2, 3)`` float one-hot (air, copper, iron)."""
    return torch.nn.functional.one_hot(phase.long(), 3).to(torch.float32)


def _face_to_vertex(K, area: Tensor) -> tuple[Tensor, Tensor]:
    """CSR ``(n0, n2)`` with entries ``|f| / 3`` and the lumped (barycentric) vertex mass ``(n0,)`` (float64)."""
    from rhmp.ops import csr_from_coo
    F = K.cells[2].long()
    rows = F.reshape(-1)
    cols = torch.arange(F.shape[0], device=F.device).repeat_interleave(3)
    vals = (area / 3.0).repeat_interleave(3)
    mass = torch.zeros(K.n[0], dtype=torch.float64, device=F.device).index_add_(0, rows, vals)
    return csr_from_coo(rows, cols, vals, (K.n[0], F.shape[0])), mass


class FluxToNodeField:
    """Edge fluxes ``Phi = d0 A`` (flux of B through each edge, right normal of src -> dst) -> vertex fields
    ``B_i = sum_{f ∋ i} |f| R grad_f(A) / sum_{f ∋ i} |f|`` with ``R (gx, gy) = (gy, -gx)`` (``B = curl A e_z``), times the
    fixed unit length ``scale`` (mm).  ``field='grad'``: potential differences ``d0 phi`` -> ``E_i = -sum |f| grad_f(phi)
    / sum |f|`` (``E = -grad phi``, a true vector; task MXE).

    ``grad_f(A) = s_0 Phi_{e0} g_1 - s_2 Phi_{e2} g_2`` from the two face edges at local vertex 0 (``g_k`` the P1
    barycentric gradients, ``s_j`` the face-edge orientation signs of ``K.meta``): a fixed linear map that depends only on
    differences of A (exact for the generator's targets), on the geometry and on the orientation of the plane (B is a
    pseudo-vector), not on vertex labels or edge / face orientation conventions.  Computed per complex (block-diagonal
    batches of meshes, rotated or refined meshes); ``takes_complex = True``.  Called without a complex it uses the
    task's own mesh (``base``), so linear-readout fits work unchanged.  Output ``(n0, B, 2)``.
    """
    kind = "flux_to_node_field"
    takes_complex = True

    def __init__(self, base=None, scale: float = MAP_LENGTH_MM, field: str = "curl"):
        if field not in ("curl", "grad"):
            raise ValueError(f"field must be 'curl' or 'grad', got {field!r}")
        self.model_readout = "grad"
        self.model_out_dim = 1
        self.scale = float(scale)
        self.field = field
        self.base = base
        self._cache: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()
        self.description = (f"edge fluxes d0 A -> vertex B = R(-90 deg) of the area-weighted mean element gradient "
                            f"(P1), unit length {self.scale:g} mm" if field == "curl" else
                            f"potential differences d0 phi -> vertex E = -(area-weighted mean element gradient) (P1), "
                            f"unit length {self.scale:g} mm")

    @torch.no_grad()
    def matrix(self, K) -> tuple[Tensor, Tensor]:
        """CSR ``(2 n0, n1)`` (rows ``2 i + c``) and its transpose, float32 on ``K``'s device (cached per complex)."""
        try:
            hit = self._cache.get(K)
        except TypeError:
            hit = None
        if hit is not None:
            return hit
        from rhmp import geometry as geo
        from rhmp.ops import csr_from_coo
        F = K.cells[2].long()
        V = K.pos.double()[F]                                                 # (n2, 3, 2)
        g = geo.barycentric_gradients(V)                                     # (n2, 3, 2)
        area = triangle_areas(K)
        fe, fs = K.meta["face_edges"].long(), K.meta["face_edge_signs"].double()
        n0, n1 = K.n[0], K.n[1]
        S = torch.zeros(n0, dtype=torch.float64, device=F.device).index_add_(0, F.reshape(-1), area.repeat_interleave(3))
        if self.field == "curl":
            Rg1 = torch.stack([g[:, 1, 1], -g[:, 1, 0]], -1)                  # R g_1
            Rg2 = torch.stack([g[:, 2, 1], -g[:, 2, 0]], -1)                  # R g_2
        else:                                                                 # E = -grad phi
            Rg1, Rg2 = -g[:, 1], -g[:, 2]
        rows, cols, vals = [], [], []
        for k in range(3):                                                    # each vertex of each face
            i = F[:, k]
            w = (area / S[i])[:, None]
            for e, c_vec in ((fe[:, 0], fs[:, 0:1] * Rg1), (fe[:, 2], -fs[:, 2:3] * Rg2)):
                for c in range(2):
                    rows.append(2 * i + c)
                    cols.append(e)
                    vals.append(self.scale * w[:, 0] * c_vec[:, c])
        rows, cols, vals = torch.cat(rows), torch.cat(cols), torch.cat(vals)
        key = rows * n1 + cols                                                # sum duplicates
        uk, inv = torch.unique(key, return_inverse=True)
        v = torch.zeros(len(uk), dtype=torch.float64, device=F.device).index_add_(0, inv, vals)
        r, cl = uk // n1, uk % n1
        M = csr_from_coo(r, cl, v.float(), (2 * n0, n1))
        MT = csr_from_coo(cl, r, v.float(), (n1, 2 * n0))
        try:
            self._cache[K] = (M, MT)
        except TypeError:
            pass
        return M, MT

    def __call__(self, y: Tensor, K=None) -> Tensor:
        """``(n1, B, 1)`` edge values -> ``(n0, B, 2)``."""
        from rhmp.ops import spmm
        K = self.base if K is None else K
        if K is None:
            raise ValueError("FluxToNodeField needs the complex (call it as omap(y, K))")
        if y.shape[-1] != 1:
            raise ValueError("FluxToNodeField expects one edge channel")
        M, MT = self.matrix(K)
        z = spmm(M, y.contiguous(), MT)                                       # (2 n0, B, 1)
        return z.view(K.n[0], 2, y.shape[1]).permute(0, 2, 1).contiguous()

    def to(self, device) -> "FluxToNodeField":
        return self


# ======================================================================================================================
# loading
# ======================================================================================================================
def _load_blob(droot: str, base: str) -> dict:
    path = os.path.join(droot, "v2", base + ".pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run python3 -u datasets/generators/gen_maxwell.py data (see its "
                                f"docstring)")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    blob["_path"] = path
    return blob


def _complex(blob: dict, star: str, whitney: bool):
    from rhmp.complex import CochainComplex
    K = CochainComplex.from_triangles(blob["pos"].double(), blob["faces"].long(), star=star, device="cpu")
    if K.n[2] != blob["faces"].shape[0]:
        raise RuntimeError(f"the complex dropped {blob['faces'].shape[0] - K.n[2]} faces")
    if not whitney:
        K.whitney = {}
    return K


def _raw_fields(K, blob: dict, idx) -> dict:
    """Raw (physical) per-sample inputs and targets on complex ``K`` for the samples ``idx`` of ``blob``."""
    from rhmp.ops import spmm
    area = triangle_areas(K)
    P, mass = _face_to_vertex(K, area)
    region = blob["region"].long()
    sign, wind = blob["coil_sign"].double(), blob["coil_winding"].long()
    cur = blob["currents"][idx].double()                                    # (N, 3) A/mm^2
    J_face = torch.zeros(len(idx), K.n[2], dtype=torch.float64)
    for j in range(6):
        J_face[:, region == j + 1] = (sign[j] * cur[:, wind[j]])[:, None]
    J_node = (spmm(P, J_face.T.contiguous().unsqueeze(-1))[..., 0] / mass[:, None]).T     # (N, n0): M J = F
    oh = face_onehot(blob["phase"][idx])                                     # (N, n2, 3)
    n_inc = torch.zeros(K.n[1], dtype=torch.float64).index_add_(
        0, K.meta["face_edges"].long().reshape(-1), torch.ones(3 * K.n[2], dtype=torch.float64))
    from rhmp.ops import csr_from_coo
    fe = K.meta["face_edges"].long()
    Inc = csr_from_coo(fe.reshape(-1), torch.arange(K.n[2]).repeat_interleave(3),
                       (1.0 / n_inc[fe.reshape(-1)]).float(), (K.n[1], K.n[2]))
    N = len(idx)
    edge_frac = spmm(Inc, oh.permute(1, 0, 2).reshape(K.n[2], N * 3).contiguous()).view(K.n[1], N, 3).permute(1, 0, 2)
    dual = spmm(P, oh.permute(1, 0, 2).reshape(K.n[2], N * 3).contiguous()).view(K.n[0], N, 3).permute(1, 0, 2)
    dual = dual / mass.float()[None, :, None]
    return dict(J_node=J_node.float(), face=oh, edge=edge_frac.contiguous(), dual=dual.contiguous(),
                B=blob["B_node"][idx].float(), A=blob["A"][idx])


def _consistency(K, omap: FluxToNodeField, raw: dict, n: int = 16) -> dict:
    """``1e3 omap(d0 A) / scale`` vs the stored targets (float64 potentials in Wb/m; positions in mm, B in T)."""
    A = raw["A"][:n].double().T.contiguous().unsqueeze(-1)                  # (n0, n, 1)
    Kd = K.to(dtype=torch.float64)
    Phi = Kd.apply_d(0, A).float()
    y = omap(Phi, K).permute(1, 0, 2) * (1e3 / omap.scale)
    t = raw["B"][:n]
    return {"max_rel_err_map_of_d0A_vs_target": float(((y - t).norm() / t.norm()).item()), "samples": n}


def _pack(raw: dict, native: bool) -> tuple[dict, Tensor]:
    """Model inputs and target (physical units) of the native or legacy view."""
    if native:
        inputs = {0: raw["J_node"].unsqueeze(-1), 1: raw["edge"], 2: raw["face"]}
    else:
        inputs = {0: torch.cat([raw["J_node"].unsqueeze(-1), raw["dual"]], -1)}
    return inputs, raw["B"]


def load_mxs(name: str, droot: str, *, native: bool = True, device="cuda", star: str = "cotan", fine: bool = True,
             max_samples: int | None = None, whitney: bool = True, output_map=None, **_) -> TaskData:
    """Task ``MXS`` (see the module docstring).

    Args:
        native: RHMP inputs (vertex current, edge / face phase columns; ``grad`` readout + :class:`FluxToNodeField`)
            or the node inputs of the baselines.
        fine: also load the extra test sets (``ood15``, ``fine``, ``coarse`` and their ``*_test100`` subsets).
        max_samples: smoke tests: keep ``max_samples // 3`` samples of every split part.
        output_map: ``False`` drops the output map (the model must then predict the vertex field directly).
    """
    t0 = time.time()
    blob = _load_blob(droot, "MXS")
    N_all = int(blob["currents"].shape[0])
    split = sequential_split(N_all)
    ids = torch.arange(N_all)
    if max_samples is not None and max_samples < N_all:
        per = max(1, max_samples // 3)
        parts = [s_[:per] for s_ in split]
        ids = torch.cat(parts)
        off = np.cumsum([0] + [len(p) for p in parts])
        split = tuple(torch.arange(int(off[j]), int(off[j + 1])) for j in range(3))
    K = _complex(blob, star, whitney)
    raw = _raw_fields(K, blob, ids)
    inputs, target = _pack(raw, native)
    tr = split[0]
    if native:
        x_stats = {0: feature_stats(inputs[0][tr]), 1: feature_stats(inputs[1][tr]), 2: feature_stats(inputs[2][tr])}
        in_dims, even_dims = {0: 1, 1: 3, 2: 3}, {1: 3, 2: 3}
    else:
        x_stats = {0: feature_stats(inputs[0][tr])}
        in_dims, even_dims = {0: 4}, {}
    y_stats = scale_stats(target[tr], isotropic=True)
    omap = FluxToNodeField(base=K) if (native and output_map is not False) else None
    meta = dict(v1_C=MXS_DEFAULTS["MXS"]["C"], batch_size=MXS_DEFAULTS["MXS"]["batch"],
                layers=MXS_DEFAULTS["MXS"]["layers"], source=blob["_path"], generator=blob["meta"].get("generator"),
                r2_std=y_stats.std, law=blob["meta"].get("law"), mur_init=blob["meta"].get("mur_init"),
                Js=blob["meta"].get("Js"), phase_columns=list(PHASE_NAMES), mesh_stats=blob["meta"].get("mesh_stats"),
                paper_eval=(torch.arange(*TEST100) if max_samples is None else split[2]),
                split_note="sequential 70/15/15 of 5000: train 0..3499, validation 3500..4249, test 4250..4999",
                sample_ids=ids.tolist() if max_samples is not None else None,
                native_inputs=("vertex J (lumped); edge / face phase (air, copper, iron) as even columns" if native else
                               "vertex (J, dual-cell phase fractions air / copper / iron)"))
    if omap is not None:
        meta["consistency"] = _consistency(K, omap, raw)
        if meta["consistency"]["max_rel_err_map_of_d0A_vs_target"] > 1e-4:
            raise RuntimeError(f"MXS: the output map does not reproduce the stored targets: {meta['consistency']}")
    extra = {}
    if fine:
        for label in ("ood15", "fine", "coarse"):
            base = "MXS_" + label
            if not os.path.exists(os.path.join(droot, "v2", base + ".pt")):
                continue
            fb = _load_blob(droot, base)
            nf = int(fb["currents"].shape[0])
            fids = torch.arange(nf if max_samples is None else min(nf, max(1, max_samples // 3)))
            Kf = K if label == "ood15" else _complex(fb, star, whitney)
            rf = _raw_fields(Kf, fb, fids)
            xf, yf = _pack(rf, native)
            xf = {k: x_stats[k].normalize(v) for k, v in xf.items()}
            yf = y_stats.normalize(yf)
            src = dict(K=[Kf] * len(fids), inputs=[{k: v[i].contiguous() for k, v in xf.items()} for i in range(len(fids))],
                       target=[yf[i].contiguous() for i in range(len(fids))])
            extra[label] = src
            sids = fb["sample_ids"][:len(fids)].tolist()
            loc = [j for j, s_ in enumerate(sids) if TEST100[0] <= s_ < TEST100[1]]
            if loc:
                extra[label + "_test100"] = {key: [src[key][j] for j in loc] for key in ("K", "inputs", "target")}
            meta[f"{label}_mesh"] = fb["meta"].get("mesh_stats", {}).get("n0")
    inputs = {k: x_stats[k].normalize(v).contiguous() for k, v in inputs.items()}
    target = y_stats.normalize(target).contiguous()
    K_cpu = K
    K, inputs, target = pack_to((K, inputs, target), device)
    if omap is not None:
        omap.base = K
    moved_K = {id(K_cpu): K}                                                  # one device copy per distinct mesh
    for src in extra.values():
        Ks = src["K"][0]
        if id(Ks) not in moved_K:
            moved_K[id(Ks)] = pack_to(Ks, device)
        src["inputs"], src["target"] = pack_to((src["inputs"], src["target"]), device)
        src["K"] = [moved_K[id(Ks)]] * len(src["target"])
    meta["load_seconds"] = time.time() - t0
    return TaskData(name=name, K=K, inputs=inputs, target=target, target_degree=0, target_kind="node_vector",
                    in_dims=in_dims, even_dims=even_dims, connection_dims={}, split=split,
                    x_stats={k: s_.to(device) for k, s_ in x_stats.items()}, y_stats=y_stats.to(device),
                    spatial_dim=2, out_dim=2, native=native, meta=meta, extra_tests=extra, output_map=omap)


# ======================================================================================================================
# MXE: electrostatics of an electrical capacitance tomography sensor (secondary family)
# ======================================================================================================================
ECT_PHASES = ("air", "wall", "oil", "water", "gas")


def _raw_fields_ect(K, blob: dict, idx) -> dict:
    """Raw per-sample inputs / targets of MXE on ``K`` (vertex charge lumped from the face charges, 0 on the fixed
    vertices; electrode potentials on the electrode outlines)."""
    from rhmp.ops import csr_from_coo, spmm
    area = triangle_areas(K)
    P, mass = _face_to_vertex(K, area)
    q = blob["q_face"][idx].double()                                          # (N, n2) V/mm^2
    q_node = (spmm(P, q.T.contiguous().unsqueeze(-1))[..., 0] / mass[:, None]).T
    fixed = K.boundary[0].cpu()
    q_node[:, fixed] = 0.0
    el = blob["electrode"].long()
    V = blob["potentials"][idx].double()                                      # (N, 8)
    V_node = torch.zeros(len(idx), K.n[0], dtype=torch.float64)
    on = el >= 0
    V_node[:, on] = V[:, el[on]]
    oh = torch.nn.functional.one_hot(blob["phase"][idx].long(), len(ECT_PHASES)).to(torch.float32)
    fe = K.meta["face_edges"].long()
    n_inc = torch.zeros(K.n[1], dtype=torch.float64).index_add_(0, fe.reshape(-1), torch.ones(3 * K.n[2],
                                                                                            dtype=torch.float64))
    Inc = csr_from_coo(fe.reshape(-1), torch.arange(K.n[2]).repeat_interleave(3),
                       (1.0 / n_inc[fe.reshape(-1)]).float(), (K.n[1], K.n[2]))
    N, nP = len(idx), len(ECT_PHASES)
    flat = oh.permute(1, 0, 2).reshape(K.n[2], N * nP).contiguous()
    edge_frac = spmm(Inc, flat).view(K.n[1], N, nP).permute(1, 0, 2)
    dual = spmm(P, flat).view(K.n[0], N, nP).permute(1, 0, 2) / mass.float()[None, :, None]
    return dict(q=q_node.float(), V=V_node.float(), face=oh, edge=edge_frac.contiguous(), dual=dual.contiguous(),
                E=blob["E_node"][idx].float(), phi=blob["phi"][idx])


def _pack_ect(raw: dict, native: bool) -> tuple[dict, Tensor]:
    """Model inputs and target (physical units) of MXE."""
    vert = torch.stack([raw["q"], raw["V"]], -1)
    if native:
        return {0: vert, 1: raw["edge"], 2: raw["face"]}, raw["E"]
    return {0: torch.cat([vert, raw["dual"]], -1)}, raw["E"]


def load_mxe(name: str, droot: str, *, native: bool = True, device="cuda", star: str = "cotan", fine: bool = True,
             max_samples: int | None = None, whitney: bool = True, output_map=None, **_) -> TaskData:
    """Task ``MXE``: electrical capacitance tomography sensor (8 electrodes at random potentials on an insulating pipe,
    grounded screen, oil / water / gas flow pattern with permittivities 2.2 / 80 / 1, wall 3.5, triboelectric space
    charge), ``-div(eps grad phi) = rho / eps_0``, target ``E = -grad phi`` at the vertices (V/mm).

    native: vertex ``(q, V)`` (lumped space charge rho/eps_0, 0 on the fixed vertices; electrode potential on the
    electrode outlines, 0 elsewhere: the Dirichlet data), even edge / face phase columns (5 phases); readout ``grad``
    (``d0 phi``, exactly curl-free) + :class:`FluxToNodeField` with ``field='grad'``.  The electrode outlines and the
    screen are the mesh boundary (``K.boundary[0]``; electrodes are holes), i.e. the fixed cells of ``solve_bc='lift'``.
    legacy: vertex ``(q, V, dual-cell phase fractions)`` -> vertex ``(E_x, E_y)``.  Same split / extra sets as MXS.
    """
    t0 = time.time()
    blob = _load_blob(droot, "MXE")
    N_all = int(blob["potentials"].shape[0])
    split = sequential_split(N_all)
    ids = torch.arange(N_all)
    if max_samples is not None and max_samples < N_all:
        per = max(1, max_samples // 3)
        parts = [s_[:per] for s_ in split]
        ids = torch.cat(parts)
        off = np.cumsum([0] + [len(p) for p in parts])
        split = tuple(torch.arange(int(off[j]), int(off[j + 1])) for j in range(3))
    K = _complex(blob, star, whitney)
    if not torch.equal(K.boundary[0].cpu(), blob["boundary"].bool()):
        raise RuntimeError("MXE: the complex's boundary differs from the generator's (screen + electrode outlines)")
    raw = _raw_fields_ect(K, blob, ids)
    inputs, target = _pack_ect(raw, native)
    tr = split[0]
    x_stats = {k: feature_stats(v[tr]) for k, v in inputs.items()}
    if native:
        in_dims, even_dims = {0: 2, 1: 5, 2: 5}, {1: 5, 2: 5}
    else:
        in_dims, even_dims = {0: 7}, {}
    y_stats = scale_stats(target[tr], isotropic=True)
    omap = FluxToNodeField(base=K, field="grad") if (native and output_map is not False) else None
    meta = dict(v1_C=MXS_DEFAULTS["MXS"]["C"], batch_size=MXS_DEFAULTS["MXS"]["batch"],
                layers=MXS_DEFAULTS["MXS"]["layers"], source=blob["_path"], generator=blob["meta"].get("generator"),
                r2_std=y_stats.std, phase_columns=list(ECT_PHASES), eps=blob["meta"].get("eps"),
                mesh_stats=blob["meta"].get("mesh_stats"),
                paper_eval=(torch.arange(*TEST100) if max_samples is None else split[2]),
                split_note="sequential 70/15/15 of 5000: train 0..3499, validation 3500..4249, test 4250..4999",
                native_inputs=("vertex (q = rho/eps_0 lumped, electrode potential V); edge / face phase (air, wall, "
                               "oil, water, gas) as even columns" if native else
                               "vertex (q, V, dual-cell phase fractions)"))
    if omap is not None:
        phi = raw["phi"][:16].double().T.contiguous().unsqueeze(-1)
        Phi = K.to(dtype=torch.float64).apply_d(0, phi).float()
        y = omap(Phi, K).permute(1, 0, 2) / omap.scale
        t = raw["E"][:16]
        meta["consistency"] = {"max_rel_err_map_of_d0phi_vs_target": float(((y - t).norm() / t.norm()).item()),
                               "samples": 16}
        if meta["consistency"]["max_rel_err_map_of_d0phi_vs_target"] > 1e-4:
            raise RuntimeError(f"MXE: the output map does not reproduce the stored targets: {meta['consistency']}")
    extra = {}
    if fine:
        for label in ("ood15", "fine", "coarse"):
            base = "MXE_" + label
            if not os.path.exists(os.path.join(droot, "v2", base + ".pt")):
                continue
            fb = _load_blob(droot, base)
            nf = int(fb["potentials"].shape[0])
            fids = torch.arange(nf if max_samples is None else min(nf, max(1, max_samples // 3)))
            Kf = K if label == "ood15" else _complex(fb, star, whitney)
            xf, yf = _pack_ect(_raw_fields_ect(Kf, fb, fids), native)
            xf = {k: x_stats[k].normalize(v) for k, v in xf.items()}
            yf = y_stats.normalize(yf)
            src = dict(K=[Kf] * len(fids), inputs=[{k: v[i].contiguous() for k, v in xf.items()} for i in range(len(fids))],
                       target=[yf[i].contiguous() for i in range(len(fids))])
            extra[label] = src
            sids = fb["sample_ids"][:len(fids)].tolist()
            loc = [j for j, s_ in enumerate(sids) if TEST100[0] <= s_ < TEST100[1]]
            if loc:
                extra[label + "_test100"] = {key: [src[key][j] for j in loc] for key in ("K", "inputs", "target")}
    inputs = {k: x_stats[k].normalize(v).contiguous() for k, v in inputs.items()}
    target = y_stats.normalize(target).contiguous()
    K_cpu = K
    K, inputs, target = pack_to((K, inputs, target), device)
    if omap is not None:
        omap.base = K
    moved_K = {id(K_cpu): K}
    for src in extra.values():
        Ks = src["K"][0]
        if id(Ks) not in moved_K:
            moved_K[id(Ks)] = pack_to(Ks, device)
        src["inputs"], src["target"] = pack_to((src["inputs"], src["target"]), device)
        src["K"] = [moved_K[id(Ks)]] * len(src["target"])
    meta["load_seconds"] = time.time() - t0
    return TaskData(name=name, K=K, inputs=inputs, target=target, target_degree=0, target_kind="node_vector",
                    in_dims=in_dims, even_dims=even_dims, connection_dims={}, split=split,
                    x_stats={k: s_.to(device) for k, s_ in x_stats.items()}, y_stats=y_stats.to(device),
                    spatial_dim=2, out_dim=2, native=native, meta=meta, extra_tests=extra, output_map=omap)


# ======================================================================================================================
# registry
# ======================================================================================================================
# ======================================================================================================================
# MXA: anisotropic (grain-oriented / laminated) steel, several layout families, held-out family
# ======================================================================================================================
MXA_PHASES = ("air", "copper", "NO", "GO", "LAM")
MXA_ORIENTED = (3, 4)                       # phases with a rolling direction (GO, LAM)
MXA_EXTRA = ("ood15", "fine", "coarse", "heldout")
MXA_LEGACY_COLUMNS = ("J", "f_air", "f_copper", "f_NO", "f_GO", "f_LAM", "GO_cos2phi", "GO_sin2phi", "LAM_cos2phi",
                      "LAM_sin2phi")
MXA_FRAME_PAIRS = ((6, 7), (8, 9))          # legacy columns that rotate as (cos 2phi, sin 2phi) with the frame


def rotate_frame_columns(x: Tensor, stats, pairs, theta) -> Tensor:
    """Normalised node inputs ``x (..., F)`` of a configuration rotated by ``theta`` (radians; a number or a tensor
    broadcastable to ``x[..., 0]``): every column pair ``(i, j)`` in ``pairs`` holds ``(cos 2 phi, sin 2 phi)`` frame
    components (in physical units) and turns by ``2 theta``; the other columns are unchanged."""
    if not pairs:
        return x
    x = x.clone()
    mean, std = stats.mean.to(x), stats.std.to(x)
    th = torch.as_tensor(theta, dtype=x.dtype, device=x.device)
    c, s_ = torch.cos(2.0 * th), torch.sin(2.0 * th)
    for i, j in pairs:
        a = x[..., i] * std[i] + mean[i]
        b = x[..., j] * std[j] + mean[j]
        x[..., i] = (c * a - s_ * b - mean[i]) / std[i]
        x[..., j] = (s_ * a + c * b - mean[j]) / std[j]
    return x


def edge_directions(K) -> Tensor:
    """``(n1, 2)`` unit edge vectors (float64; the sign is irrelevant for the squared cosines)."""
    E = K.cells[1].long()
    v = K.pos.double()[E[:, 1]] - K.pos.double()[E[:, 0]]
    return v / v.norm(dim=1, keepdim=True)


def _raw_fields_mxa(K, blob: dict, idx, orientation: bool = True) -> dict:
    """Raw per-sample inputs / targets of MXA on ``K``: vertex current (lumped), phase one-hot, edge phase fractions
    and squared cosines between every edge and the rolling direction of its incident GO / LAM faces (means over those
    faces, 0 without such a face), dual-cell phase fractions and the frame components ``(cos 2phi, sin 2phi)`` of the
    GO / LAM rolling directions (dual-cell area weighted, 0 elsewhere)."""
    from rhmp.ops import csr_from_coo, spmm
    area = triangle_areas(K)
    P, mass = _face_to_vertex(K, area)
    J_face = blob["J_face"][idx].double()                                     # (N, n2) A/mm^2
    J_node = (spmm(P, J_face.T.contiguous().unsqueeze(-1))[..., 0] / mass[:, None]).T
    phase = blob["phase"][idx].long()
    phi = blob["phi"][idx].double()
    N, nP, n1, n2 = len(idx), len(MXA_PHASES), K.n[1], K.n[2]
    oh = torch.nn.functional.one_hot(phase, nP).to(torch.float32)              # (N, n2, 5)
    fe = K.meta["face_edges"].long()                                          # (n2, 3)
    n_inc = torch.zeros(n1, dtype=torch.float64).index_add_(0, fe.reshape(-1), torch.ones(3 * n2, dtype=torch.float64))
    Inc = csr_from_coo(fe.reshape(-1), torch.arange(n2).repeat_interleave(3), (1.0 / n_inc[fe.reshape(-1)]).float(),
                       (n1, n2))
    flat = oh.permute(1, 0, 2).reshape(n2, N * nP).contiguous()
    edge_frac = spmm(Inc, flat).view(n1, N, nP).permute(1, 0, 2)
    t = edge_directions(K)[fe]                                                # (n2, 3, 2)
    r = torch.stack([torch.cos(phi), torch.sin(phi)], -1)                    # (N, n2, 2)
    c2 = torch.einsum("fjd,nfd->nfj", t, r) ** 2                              # (N, n2, 3)
    cols = []
    for ph in MXA_ORIENTED:
        m = (phase == ph).double()                                            # (N, n2)
        num = torch.zeros(N, n1, dtype=torch.float64).index_add_(1, fe.reshape(-1), (c2 * m[..., None]).reshape(N, -1))
        den = torch.zeros(N, n1, dtype=torch.float64).index_add_(1, fe.reshape(-1),
                                                                  m[..., None].expand(-1, -1, 3).reshape(N, -1))
        cols.append(torch.where(den > 0, num / den.clamp_min(1.0), torch.zeros_like(num)))
    edge = torch.cat([edge_frac, torch.stack(cols, -1).float()], -1)          # (N, n1, 7)
    dual = spmm(P, flat).view(K.n[0], N, nP).permute(1, 0, 2) / mass.float()[None, :, None]
    frame = []
    for ph in MXA_ORIENTED:
        m = (phase == ph).double()
        for fn in (torch.cos, torch.sin):
            v = (fn(2.0 * phi) * m).T.contiguous().unsqueeze(-1)              # (n2, N, 1)
            frame.append((spmm(P, v)[..., 0] / mass[:, None]).T)
    frame = torch.stack(frame, -1).float()                                    # (N, n0, 4)
    if not orientation:
        edge, frame = edge[..., :nP], frame[..., :0]
    return dict(J_node=J_node.float(), face=oh, edge=edge.contiguous(), dual=dual.contiguous(), frame=frame,
                B=blob["B_node"][idx].float(), A=blob["A"][idx])


def _pack_mxa(raw: dict, native: bool) -> tuple[dict, Tensor]:
    if native:
        return {0: raw["J_node"].unsqueeze(-1), 1: raw["edge"], 2: raw["face"]}, raw["B"]
    return {0: torch.cat([raw["J_node"].unsqueeze(-1), raw["dual"], raw["frame"]], -1)}, raw["B"]


def load_mxa(name: str, droot: str, *, native: bool = True, device="cuda", star: str = "cotan", fine: bool = True,
             max_samples: int | None = None, whitney: bool = True, output_map=None, orientation: bool = True,
             **_) -> TaskData:
    """Task ``MXA``: nonlinear magnetostatics with grain-oriented (GO, easy : hard permeability 25) and laminated
    (LAM, 5) steel with a rolling direction per piece, non-oriented steel (NO), air and copper, in E-I and U-I
    electromagnets and shunt transformers of random dimensions, gaps, constructions and windings, all on one fixed
    lattice mesh (1,818 nodes; generator ``datasets/generators/gen_maxwell_aniso.py``).  Target: vertex B (T).

    native (RHMP): vertex current density (lumped), even edge columns ``(f_air, f_copper, f_NO, f_GO, f_LAM,
    cos^2(edge, rolling direction) over the incident GO faces, the same over the incident LAM faces)`` (E(n)-invariant:
    the edge and the rolling direction rotate together), even face columns = phase one-hot; readout ``grad`` +
    :class:`FluxToNodeField`.  legacy (node baselines): vertex ``MXA_LEGACY_COLUMNS`` (current, dual-cell phase
    fractions, dual-cell (cos 2phi, sin 2phi) of the GO and LAM rolling directions: frame components, which rotate
    with the configuration, ``meta['frame_pairs']``).  Isotropic target scaling.  Split: sequential 70/15/15 of 5000,
    test100 = 4250..4349.  Extra tests (list format): ``ood15`` (test samples 4250..4749 with all currents x1.5),
    ``fine`` / ``coarse`` (the same layouts on the 4 mm / 16 mm meshes, 7,255 / 459 nodes), ``heldout`` (500 samples
    of the EE family, which never occurs in training), each also as ``<name>_test100`` (the test100 layouts; for
    ``heldout`` its first 100 samples).  ``orientation=False`` (task ``MXA_noorient``, an ablation) drops the
    rolling-direction columns (native: the two cos^2 edge columns; legacy: the four frame columns).
    """
    t0 = time.time()
    blob = _load_blob(droot, "MXA")
    N_all = int(blob["J_face"].shape[0])
    split = sequential_split(N_all)
    ids = torch.arange(N_all)
    if max_samples is not None and max_samples < N_all:
        per = max(1, max_samples // 3)
        parts = [s_[:per] for s_ in split]
        ids = torch.cat(parts)
        off = np.cumsum([0] + [len(p) for p in parts])
        split = tuple(torch.arange(int(off[j]), int(off[j + 1])) for j in range(3))
    K = _complex(blob, star, whitney)
    raw = _raw_fields_mxa(K, blob, ids, orientation)
    inputs, target = _pack_mxa(raw, native)
    tr = split[0]
    x_stats = {k: feature_stats(v[tr]) for k, v in inputs.items()}
    n_edge, n_leg = (7, len(MXA_LEGACY_COLUMNS)) if orientation else (5, 6)
    if native:
        in_dims, even_dims = {0: 1, 1: n_edge, 2: 5}, {1: n_edge, 2: 5}
    else:
        in_dims, even_dims = {0: n_leg}, {}
    y_stats = scale_stats(target[tr], isotropic=True)
    omap = FluxToNodeField(base=K) if (native and output_map is not False) else None
    fam = blob["family"][ids]
    meta = dict(v1_C=MXS_DEFAULTS["MXS"]["C"], batch_size=MXS_DEFAULTS["MXS"]["batch"],
                layers=MXS_DEFAULTS["MXS"]["layers"], source=blob["_path"], generator=blob["meta"].get("generator"),
                r2_std=y_stats.std, law=blob["meta"].get("law"), materials=blob["meta"].get("materials"),
                Js=blob["meta"].get("Js"), phase_columns=list(MXA_PHASES), mesh_stats=blob["meta"].get("mesh_stats"),
                families=blob["meta"].get("families"),
                family_counts={f: int((fam == k).sum()) for k, f in enumerate(blob["meta"].get("families", []))},
                paper_eval=(torch.arange(*TEST100) if max_samples is None else split[2]),
                split_note="sequential 70/15/15 of 5000: train 0..3499, validation 3500..4249, test 4250..4999",
                sample_ids=ids.tolist() if max_samples is not None else None,
                edge_columns=["f_air", "f_copper", "f_NO", "f_GO", "f_LAM", "cos2_edge_rolling_GO",
                              "cos2_edge_rolling_LAM"][:n_edge], orientation=orientation,
                legacy_columns=list(MXA_LEGACY_COLUMNS[:n_leg]),
                frame_pairs=[list(p) for p in MXA_FRAME_PAIRS] if (orientation and not native) else [],
                native_inputs=("vertex J (lumped); edge phase fractions + cos^2(edge, rolling direction) of "
                                        "the incident GO / LAM faces; face phase one-hot (even columns)" if native else
                                        "vertex (J, dual-cell phase fractions, dual-cell (cos 2phi, sin 2phi) of the "
                                        "GO / LAM rolling directions)"))
    if omap is not None:
        meta["consistency"] = _consistency(K, omap, raw)
        if meta["consistency"]["max_rel_err_map_of_d0A_vs_target"] > 1e-4:
            raise RuntimeError(f"MXA: the output map does not reproduce the stored targets: {meta['consistency']}")
    extra = {}
    if fine:
        for label in MXA_EXTRA:
            base = "MXA_" + label
            if not os.path.exists(os.path.join(droot, "v2", base + ".pt")):
                continue
            fb = _load_blob(droot, base)
            nf = int(fb["J_face"].shape[0])
            fids = torch.arange(nf if max_samples is None else min(nf, max(1, max_samples // 3)))
            Kf = K if label in ("ood15", "heldout") else _complex(fb, star, whitney)
            rf = _raw_fields_mxa(Kf, fb, fids, orientation)
            xf, yf = _pack_mxa(rf, native)
            xf = {k: x_stats[k].normalize(v) for k, v in xf.items()}
            yf = y_stats.normalize(yf)
            src = dict(K=[Kf] * len(fids), inputs=[{k: v[i].contiguous() for k, v in xf.items()} for i in range(len(fids))],
                       target=[yf[i].contiguous() for i in range(len(fids))])
            extra[label] = src
            sids = fb["sample_ids"][:len(fids)].tolist()
            loc = (list(range(min(100, len(fids)))) if label == "heldout" else
                   [j for j, s_ in enumerate(sids) if TEST100[0] <= s_ < TEST100[1]])
            if loc:
                extra[label + "_test100"] = {key: [src[key][j] for j in loc] for key in ("K", "inputs", "target")}
            meta[f"{label}_mesh"] = fb["meta"].get("mesh_stats", {}).get("n0")
    inputs = {k: x_stats[k].normalize(v).contiguous() for k, v in inputs.items()}
    target = y_stats.normalize(target).contiguous()
    K_cpu = K
    K, inputs, target = pack_to((K, inputs, target), device)
    if omap is not None:
        omap.base = K
    moved_K = {id(K_cpu): K}
    for src in extra.values():
        Ks = src["K"][0]
        if id(Ks) not in moved_K:
            moved_K[id(Ks)] = pack_to(Ks, device)
        src["inputs"], src["target"] = pack_to((src["inputs"], src["target"]), device)
        src["K"] = [moved_K[id(Ks)]] * len(src["target"])
    meta["load_seconds"] = time.time() - t0
    return TaskData(name=name, K=K, inputs=inputs, target=target, target_degree=0, target_kind="node_vector",
                    in_dims=in_dims, even_dims=even_dims, connection_dims={}, split=split,
                    x_stats={k: s_.to(device) for k, s_ in x_stats.items()}, y_stats=y_stats.to(device),
                    spatial_dim=2, out_dim=2, native=native, meta=meta, extra_tests=extra, output_map=omap)


def _loader(fn, **fixed):
    @functools.wraps(fn)
    def load(droot=None, **kw):
        from rhmp.tasks import resolve_root
        return fn(droot=resolve_root(droot), **{**fixed, **kw})
    return load


MXS_TASKS: dict = {"MXS": _loader(load_mxs, name="MXS"), "MXE": _loader(load_mxe, name="MXE"),
                   "MXA": _loader(load_mxa, name="MXA"),
                   "MXA_noorient": _loader(load_mxa, name="MXA_noorient", orientation=False)}
