"""Anisotropic scalar transport on the torus (task T2a; generator ``datasets/generators/gen_T2a.py``).

``d_t u + div(v u) = div(D grad u)`` on the T2 torus with a divergence-free velocity (per-sample toroidal and poloidal
amplitudes of T2 plus a random solenoidal perturbation) and field-aligned diffusion ``D = D_perp P + (D_par - D_perp)
b b^T`` along helical field lines ``b`` (per-sample pitch, smoothly varying), anisotropy ratio ``D_par / D_perp`` in
1..100; the T2 initial conditions; ``u(T)``, ``T = 0.2``, from an accurate conservative reference solver (P1 tensor
diffusion, exactly divergence-free dual fluxes, Crank-Nicolson with 200 sub-steps).

Inputs:

* native (RHMP): degree 0 ``f``; degree 1 ``[theta, delta, cos^2(t_e, b), log(t_e^T D t_e)]`` (the velocity one-form
  ``(V_s + V_d)/2 . (x_d - x_s)`` (odd), its half-difference (even, see ``rhmp.tasks.paper.t2v_inputs``), and the two
  E(n)-invariant anisotropy descriptors (even; means over the incident faces)); degree 2 ``[log D_par, log D_perp]``
  (even, constant per sample).  Suggested: ``--metric-ref 1:3`` (the projected diffusivity as the fixed offset of the
  edge metric) and ``--material 1:2,2:2`` (the anisotropy columns reach only the metric heads), the full tensor metric,
  and the transport term with ``metric='star'`` (geometric advective flux).
* legacy (baselines): node features ``(f, Vx, Vy, Vz, bx, by, bz, log D_par, log D_perp)``.

Target ``u(T)`` at the vertices.  Same split as T2 (sequential 70/15/15).  The zero-shot sets of the test split
(velocity and anisotropy above the training ranges, roll-out to ``2T``, a 2x finer and a 2x coarser torus) are read
by ``scripts/eval_torus_aniso.py`` through :func:`t2a_file` / :func:`t2a_inputs`.
"""
from __future__ import annotations

import os

import torch
from torch import Tensor

from rhmp.tasks.paper import _defaults, _finish, t2v_inputs

__all__ = ["t2a_file", "t2a_inputs", "t2a_complex", "load_T2a"]


def t2a_file(droot: str) -> dict:
    """``datasets/v2/T2a.pt`` (``datasets/generators/gen_T2a.py``)."""
    path = os.path.join(droot, "v2", "T2a.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run python3 -u datasets/generators/gen_T2a.py (see datasets/README.md)")
    return torch.load(path, map_location="cpu", weights_only=False)


def t2a_complex(points: Tensor, faces: Tensor, edges: Tensor, star: str, device):
    """Complex of a torus mesh of the generator and the map of the generator's edge data onto its edges.

    Returns:
        ``(K, perm)``: even edge data of the generator ``x (N, n1)`` are ``x[:, perm]`` in the complex's edge order.
    """
    from rhmp.complex import CochainComplex
    from rhmp.data import edge_alignment
    K = CochainComplex.from_triangles(torch.as_tensor(points, dtype=torch.float64), torch.as_tensor(faces).long(),
                                      star=star, device=device)
    if K.n[2] != len(faces):
        raise RuntimeError("T2a: the complex dropped faces; face data would misalign")
    perm, _ = edge_alignment(K.cells[1], torch.as_tensor(edges), int(K.n[0]))
    return K, perm


def t2a_inputs(X: Tensor, V: Tensor, b: Tensor, cos2: Tensor, logp: Tensor, Dpar: Tensor, Dperp: Tensor, K,
               native: bool) -> dict[int, Tensor]:
    """Raw (unnormalised) T2a inputs.

    Args:
        X: ``(N, n0)`` initial field; V, b: ``(N, n0, 3)`` velocity and field direction at the vertices;
        cos2, logp: ``(N, n1)`` edge descriptors; Dpar, Dperp: ``(N,)``; K: the complex of the mesh.
        native: native cochain inputs (RHMP) or the legacy node features (baselines).
    """
    X = X.float().reshape(X.shape[0], -1, 1)
    N = X.shape[0]
    lpar, lperp = torch.log(Dpar.double()).float(), torch.log(Dperp.double()).float()
    if not native:
        n0 = X.shape[1]
        consts = torch.stack([lpar, lperp], -1)[:, None, :].expand(N, n0, 2)
        return {0: torch.cat([X, V.float(), b.float(), consts], -1).contiguous()}
    E = t2v_inputs(X, V.float(), K, True)[1]                                    # theta (odd), delta (even)
    E = torch.cat([E, cos2.float().unsqueeze(-1), logp.float().unsqueeze(-1)], -1)
    n2 = int(K.n[2])
    F = torch.stack([lpar, lperp], -1)[:, None, :].expand(N, n2, 2).contiguous()
    return {0: X.contiguous(), 1: E.contiguous(), 2: F}


def load_T2a(droot, *, native, device, star, **_):
    """T2a (see the module docstring): native ``in_dims {0: 1, 1: 4, 2: 2}``, ``even_dims {1: 3, 2: 2}``; legacy
    ``in_dims {0: 9}``; node target."""
    blob = t2a_file(droot)
    K, perm = t2a_complex(blob["points"], blob["faces"], blob["edges"], star, device)
    raw = t2a_inputs(blob["X"], blob["V"], blob["b"], blob["edge_cos2"][:, perm], blob["edge_logp"][:, perm],
                     blob["Dpar"], blob["Dperp"], K, native)
    Y = blob["Y"].float().unsqueeze(-1)
    meta = _defaults("T2a")
    gm = blob.get("meta", {})
    meta["generator"] = {k: gm.get(k) for k in ("T", "n_sub", "checks", "mass_drift_max", "target_stats") if k in gm}
    meta["parameters"] = {k: v.float() for k, v in blob["params"].items()}
    meta["metric_ref"] = "1:3"
    meta["material"] = "1:2,2:2"
    if native:
        meta["native_inputs"] = ("node f; edge velocity 1-form (odd), half-difference, cos^2(t_e, b), log t_e^T D t_e "
                                 "(even); face log D_par, log D_perp (even)")
        return _finish("T2a", K, raw, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 1, 1: 4, 2: 2},
                       even_dims={1: 3, 2: 2}, connection_dims={}, x_kinds={0: "v1", 1: "scale", 2: "v1"},
                       y_kind="v1", spatial_dim=3, native=True, device=device, meta=meta)
    meta["legacy_inputs"] = "node (f, Vx, Vy, Vz, bx, by, bz, log D_par, log D_perp)"
    return _finish("T2a", K, raw, Y, target_degree=0, target_kind="node_scalar", in_dims={0: 9}, even_dims={},
                   connection_dims={}, x_kinds={0: "v1"}, y_kind="v1", spatial_dim=3, native=False, device=device,
                   meta=meta)

