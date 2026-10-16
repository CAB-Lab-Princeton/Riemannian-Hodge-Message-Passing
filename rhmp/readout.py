"""Readouts from hidden cochains (DESIGN §3.5).

All readouts are E(n)-invariant (``node_vector``: E(n)-equivariant) and permutation-equivariant; they pick a
frame of the hidden channels, so O(C) is a property of the message-passing stack only.  Orientation behaviour:

* ``node_scalar``   ``MLP(x_0)`` -> ``(n_0, B, out_dim)`` (vertex scalars, even); ``cfg.readout_energy = h``: the MLP
  also sees, and the output adds a linear map of, ``h`` Galerkin energy densities ``<x_1 A_j, x_1 B_j>`` of projections
  of the edge features at the vertices (symmetric quadratic forms of the edge cochains in the reference metric, even;
  e.g. the Bernoulli pressure ``-|u|^2 / 2`` of a velocity 1-form).
* ``cochain:k``     bias-free linear map of the odd features ``x_k`` -> ``(n_k, B, out_dim)`` (odd for k >= 1:
  fluxes, circulations, curvatures); ``cfg.cochain_head='gated'``: odd gated head (linear x even gate);
  ``'context'``: gated head that also sees ``d_{k-1} x_{k-1}`` (odd) and the vertex mean of ``x_0`` (even gate input);
  ``cfg.readout_wedge`` (k = 2): plus face wedge products of the edge features (true 2-cochains, quadratic).
* ``even:k``        ``MLP([log1p(x_k^2), log1p(|x_k|^2/C), geo_k])`` (``x_0`` itself for k = 0) -> ``(n_k, B, out_dim)``
  (orientation-invariant cell scalars, e.g. magnitudes).
* ``grad``          ``E = d_0 phi`` with ``phi = MLP(x_0)`` (``readout_head='linear'``: ``phi = Lin_nobias(x_0)``): odd
  edge cochain ``(n_1, B, out_dim)``, exactly curl-free (``d_1 E = 0``) and invariant under ``phi -> phi + const``.
* ``curl``          ``F = d_1 a`` with ``a = Lin_nobias(x_1)``: odd face cochain ``(n_2, B, out_dim)``, exactly closed
  (``d_2 F = 0`` on volumes) and, in connection mode, exactly gauge invariant (every hidden feature is).
* ``div`` / ``div:k``  ``y = d_{k-1}^T b`` with ``b = Lin_nobias(x_k)`` (``k = K`` for ``div``): output on
  ``(k-1)``-cells ``(n_{k-1}, B, out_dim)``, exactly co-closed (``d_{k-2}^T y = 0`` for k >= 2); for ``k = 1`` it is
  a vertex divergence (even) whose sum over the vertices of every graph is exactly 0 (discrete Gauss law).
  The operators are purely combinatorial (no stars), so the constraints hold to rounding.
* ``mdiv`` / ``mdiv:k``  mass-weighted divergence ``y = S_{k-1}^{-1} d_{k-1}^T b`` with ``b = Lin_nobias(x_k)`` and
  ``S_{k-1} = exp(log_star_{k-1})`` the per-graph geometric-mean-normalised star of the output degree (outputs of
  order one).  For ``k = 1`` (``S_0`` = normalised lumped vertex mass) ``sum_i star0_i y_i = 0`` exactly per graph:
  exactly mass-conserving density increments on variable meshes.  Same orientation behaviour as ``div:k``.
* ``node_vector``   odd per-edge scalars ``w_e = Lin(x_1)_e`` (``out_dim`` vector fields) turned into vertex vectors,
  ``(n_0, B, out_dim * D)`` with ``D = K.pos.shape[1]``:

  - ``vector_mode='ls'`` (least squares, exact for constant fields when ``w_e = l_e t_e . v``)::

        M_i = sum_{e ∋ i} omega_e l_e^2 t_e t_e^T ,      b_i = sum_{e ∋ i} omega_e l_e w_e t_e
        v_i = (M_i^2 + mu_i^2 I)^{-1} M_i b_i ,         mu_i = 1e-4 * trace(M_i)

    i.e. the minimiser of ``sum_e omega_e (l_e t_e . v - w_e)^2`` computed as damped least squares on the normal
    equations ``M v = b``.  Unlike adding ``1e-6 * trace`` to ``M``, the damped form has *zero* response in
    rank-deficient directions (vertices whose edges are collinear, e.g. on straight polygon sides) instead of
    amplifying fp32 rounding by ``1e6`` there; its bias on well-posed vertices is ``O((mu / lambda_min)^2)``.
    The small ``D x D`` solves run in float64 with autocast disabled.  ``omega_e =
    softplus(MLP([log1p(x_1^2), log1p(|x_1|^2/C), geo_1])) > 0`` is a learned even weight (1 at initialisation).
    For surfaces in 3-D (top degree 2, ``D = 3``) the system is solved in the vertex tangent plane (orthonormal
    basis of the two smallest eigenvectors of ``sum_f a_f n_f n_f^T``, which does not depend on the face
    orientation), so the output is tangent and the solve stays well conditioned on curved surfaces (the full 3x3
    system is nearly singular in the normal direction there).
  - ``vector_mode='direct'`` (v1 style): ``v_i = sum_{e ∋ i} w_e t_e / deg_i``.

  Both are E(n)-equivariant (rotations, reflections, translations) and invariant to the edge-orientation
  convention (``w_e`` and ``t_e`` flip together).
"""
from __future__ import annotations

import math
import weakref
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from . import ops
from .layers import LayerContext, _msq, linear, mlp2, reference_energy, spmm_ad
from .lifting import GeoMLP, vertex_mean

__all__ = [
    "NodeScalarReadout", "CochainReadout", "EvenCellReadout", "NodeVectorReadout", "GradReadout", "CurlReadout",
    "DivReadout", "MassDivReadout", "build_readout", "parse_readout", "readout_degrees", "vector_geometry", "ls_solve",
    "vertex_mean", "face_wedge",
]

# Damping of the least-squares vector reconstruction, relative to trace(M): v = (M^2 + (LS_REG tr M)^2)^{-1} M b.
LS_REG = 1e-4
# softplus^{-1}(1): bias giving omega = 1 at initialisation.
_SOFTPLUS_INV_ONE = math.log(math.e - 1.0)


_READOUT_HELP = ("use 'node_scalar', 'node_vector', 'cochain:k', 'even:k', 'grad', 'curl', 'div', 'div:k', 'mdiv' or "
                 "'mdiv:k'")


def parse_readout(spec: str) -> tuple[str, int | None]:
    """Parse a readout name.

    Args:
        spec: ``'node_scalar' | 'node_vector' | 'cochain:k' | 'even:k' | 'grad' | 'curl' | 'div' | 'div:k' | 'mdiv' |
            'mdiv:k'``.

    Returns:
        ``(kind, k)``: ``k`` is the degree of the hidden features the readout reads (``None`` for ``'div'`` /
        ``'mdiv'``, meaning the top degree).
    """
    fixed = {"node_scalar": ("node_scalar", 0), "node_vector": ("node_vector", 1), "grad": ("grad", 0),
             "curl": ("curl", 1), "div": ("div", None), "mdiv": ("mdiv", None)}
    if spec in fixed:
        return fixed[spec]
    if ":" in spec:
        kind, _, k = spec.partition(":")
        if kind in ("cochain", "even", "div", "mdiv") and k.strip().isdigit():
            return kind, int(k)
    raise ValueError(f"unknown readout {spec!r}; {_READOUT_HELP}")


def readout_degrees(spec: str, top: int) -> tuple[int, int]:
    """``(feature_degree, output_degree)`` of a readout on a complex of top degree ``top``.

    Raises:
        ValueError: if the readout does not exist on such a complex.
    """
    kind, k = parse_readout(spec)
    if kind in ("div", "mdiv"):
        k = top if k is None else k
        if not 1 <= k <= top:
            raise ValueError(f"readout {spec!r} needs 1 <= k <= {top}")
        return k, k - 1
    if kind == "curl" and top < 2:
        raise ValueError("readout 'curl' needs faces (top degree >= 2)")
    if k > top:
        raise ValueError(f"readout {spec!r}: degree {k} exceeds the top degree {top}")
    out = {"node_scalar": 0, "node_vector": 0, "grad": 1, "curl": 2}.get(kind, k)
    return k, out


class NodeScalarReadout(nn.Module):
    """``MLP(x_0)``: ``Linear(C, C) -> SiLU -> Linear(C, out_dim)``, output ``(n_0, B, out_dim)``.

    ``energy = h > 0`` (energy readout, simplicial complexes): with ``a = x_1 A``, ``b = x_1 B`` (``A, B: C -> h``,
    bias-free) and the Galerkin energy densities at the vertices

        q_i = sum_{f ∋ i} a_f^T M_f b_f / (E_ref sum_{f ∋ i} |f|)        (h channels; ``rhmp.dec.vertex_energy``)

    (``M_f``: Whitney mass block of the top cell ``f``, i.e. the reference metric ``H_1``; ``E_ref``: the reference
    energy density of the graph, :func:`rhmp.layers.reference_energy`), the output is

        y = W_2 SiLU(W_1 x_0 + b_1 + V q) + b_2 + w q

    with ``V`` and ``w`` zero-initialised (the plain readout at initialisation).  ``energy_clip = c > 0`` replaces
    ``q`` by ``c tanh(q / c)`` (a bound for meshes whose edge lengths vary over orders of magnitude, where hidden edge
    features that do not scale with the edge length have huge energy densities).  ``q`` is the pointwise product ``u . v``
    for cochains of constant fields in units of the reference energy, so ``w q`` contains the Bernoulli form
    ``c - |u|^2`` of a velocity 1-form; it is even (a product of two odd quantities), E(n)- and relabelling-invariant and
    resolution consistent.  The stack is unchanged (O(C)-equivariant); the readout fixes the channel frame.
    """

    def __init__(self, C: int, out_dim: int, energy: int = 0, energy_clip: float = 0.0) -> None:
        super().__init__()
        self.fc1 = nn.Linear(C, C)
        self.fc2 = nn.Linear(C, out_dim)
        self.energy = int(energy)
        self.energy_clip = float(energy_clip)
        if self.energy:
            self.en_a = nn.Linear(C, self.energy, bias=False)
            self.en_b = nn.Linear(C, self.energy, bias=False)
            self.en_hidden = nn.Linear(self.energy, C, bias=False)
            self.en_out = nn.Linear(self.energy, out_dim, bias=False)
            nn.init.zeros_(self.en_hidden.weight)
            nn.init.zeros_(self.en_out.weight)

    def energies(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """The energy channels ``q`` ``(n_0, B, h)`` (float32 / float64, autocast disabled)."""
        from .dec import vertex_energy
        K = ctx.K
        x1 = x[1]
        with torch.autocast(device_type=x1.device.type, enabled=False):
            a = linear(self.en_a, x1.to(ctx.dtype))
            b = linear(self.en_b, x1.to(ctx.dtype))
            q = vertex_energy(K, a, b)
            q = q / ctx.to_cells(reference_energy(ctx), 0).unsqueeze(-1).to(q.dtype)
            c = getattr(self, "energy_clip", 0.0)
            return c * torch.tanh(q / c) if c > 0 else q

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``x[0]`` ``(n_0, B, C)`` -> ``(n_0, B, out_dim)``."""
        if not getattr(self, "energy", 0):
            return mlp2(x[0], self.fc1.weight, self.fc1.bias, self.fc2.weight, self.fc2.bias)
        q = self.energies(x, ctx)
        e = self.fc1.bias + linear(self.en_hidden, q)
        y = mlp2(x[0], self.fc1.weight, e.to(x[0].dtype), self.fc2.weight, self.fc2.bias)
        return y.to(q.dtype) + linear(self.en_out, q)


def face_wedge(K: Any, a: Tensor, b: Tensor) -> Tensor:
    """Wedge (cup) product of primal 1-cochains on the faces: ``(n_1, ..., H) x (n_1, ..., H) -> (n_2, ..., H)``.

    ``(a ^ b)_f = c_m sum_j (a_j b_{j+1} - a_{j+1} b_j)`` over the boundary edges of the m-gon ``f`` in cyclic order
    (``K.meta['face_edges']``), with ``a_j = s_j a(e_j)`` read along the face orientation (``s_j`` =
    ``K.meta['face_edge_signs']``) and ``c_m = 1 / (8 sin^2(pi / m))`` (1/6 for triangles, 1/4 for quadrilaterals).
    For triangles and convex quadrilaterals this is the integral over the face of the wedge of the Whitney
    interpolants of two constant 1-forms, i.e. exact for constant forms.  Metric-free and antisymmetric; a true
    2-cochain (sign flip with the face orientation, invariant under edge orientation flips and under relabelling).
    """
    if "face_edges" not in K.meta:
        raise ValueError("face_wedge needs K.meta['face_edges'] (faces of a surface or volume complex)")
    fe, fs = K.meta["face_edges"], K.meta["face_edge_signs"]
    n2, m = fe.shape
    if "face_lengths" in K.meta:
        flen = K.meta["face_lengths"].to(device=fe.device, dtype=torch.long)
    else:
        flen = torch.full((n2,), m, dtype=torch.long, device=fe.device)
    j = torch.arange(m, device=fe.device)
    nxt = torch.where(j[None, :] + 1 < flen[:, None], j[None, :] + 1, torch.zeros_like(fe))
    s = fs.to(a.dtype).view(n2, m, *([1] * (a.dim() - 1)))
    A, B_ = a[fe.clamp_min(0)] * s, b[fe.clamp_min(0)] * s                    # (n2, m, ..., H); padding -> 0
    rows = torch.arange(n2, device=fe.device)[:, None]
    w = (A * B_[rows, nxt] - A[rows, nxt] * B_).sum(1)
    c = 1.0 / (8.0 * torch.sin(math.pi / flen.to(torch.float64)) ** 2)
    return w * c.to(w.dtype).view(n2, *([1] * (w.dim() - 1)))


class CochainReadout(nn.Module):
    """Bias-free linear map of ``x_k`` (odd for k >= 1): ``(n_k, B, C) -> (n_k, B, out_dim)``.

    ``head='gated'``: odd gated head ``y = W_2 (u ⊙ (1 + e))`` with ``u = W_1 x_k`` (``C -> hidden``, bias-free, odd)
    and the even gate ``e = MLP([log1p(u^2), log1p(|x_k|^2/C), even_k], geo_k)`` (``hidden`` wide, last layer
    zero-initialised: the linear head ``W_2 W_1 x_k`` at initialisation); ``even_k`` are the even input columns of
    degree k that the lifting sees (material columns excluded).  Odd times even, so the output keeps the orientation
    parity of a cochain; the head adds a local nonlinearity across the hidden channels.

    ``n_hist > 1`` (``readout_layers='all'``): the odd part also reads the features of degree k after the lifting and
    after every earlier layer (``ctx.history``), ``u = W_1 x_k + sum_l W_l x_k^(l)``, so that every output channel has
    its own combination of the filters of the stack.

    ``head='context'`` (k >= 1): the gated head with the features of the lower degrees of the cell:
    ``u = W_1 x_k + W_c d_{k-1} x_{k-1}`` (odd) and the vertex mean ``v = mean_{i in cell} (W_0 x_0 + b_0)`` (even)
    as additional gate input, ``e = MLP([log1p(u^2), v, log1p(|x_k|^2/C), even_k], geo_k)``, so the output can combine
    the k-cochain with vertex quantities (densities, divergences) of the same cell; still odd times even.

    ``depth > 1`` (gated / context heads): further odd gated stages ``u' = z + M z`` (``M`` bias-free, zero-initialised),
    ``z' = u' ⊙ (1 + e'([log1p(u'^2), v, ...]))`` before ``y = W_2 z``; the head is linear at initialisation.  With
    ``spatial=True`` every further stage also mixes the neighbouring cells, ``u' = z + M z + N (A z)`` with the
    combinatorial Hodge Laplacian ``A = d_{k-1} d_{k-1}^T + d_k^T d_k`` of degree k (odd to odd; ``N`` zero-initialised).

    ``skip = (o_k, o_km1)`` (gated / context heads): ``u`` also gets ``S_k f_k^odd + S_{k-1} d_{k-1} f_{k-1}^odd``, bias-free
    linear maps of the odd input columns of degree k and of the coboundary of those of degree k-1 (``ctx.inputs``;
    ``o_k``, ``o_km1`` = ``(lo, hi)`` column ranges or ``None``): a linear path from the inputs to the output.

    ``wedge = h > 0`` (gated / context heads, k = 2): ``u`` also gets ``W (x_1 A ^ x_1 B)``, the face wedge products
    (:func:`face_wedge`) of ``h`` pairs of projections of the final edge features (``A``, ``B``: ``C -> h``, bias-free;
    ``W`` zero-initialised, so the head is unchanged at initialisation).  A true 2-cochain, quadratic in the edge
    features: e.g. the advection ``theta ^ delta(d theta)`` of the vorticity 2-form ``d theta`` by the velocity
    1-form ``theta``, which an odd head cannot form.
    """

    def __init__(self, C: int, out_dim: int, k: int, head: str = "linear", hidden: int = 16, geo_dim: int = 0,
                 even_dim: int = 0, n_hist: int = 1, depth: int = 1, skip: tuple | None = None,
                 spatial: bool = False, wedge: int = 0) -> None:
        super().__init__()
        if head not in ("linear", "gated", "context"):
            raise ValueError(f"cochain readout head must be 'linear', 'gated' or 'context', got {head!r}")
        if wedge and (head == "linear" or int(k) != 2):
            raise ValueError("readout_wedge needs the gated or context head of a cochain:2 readout")
        self.degree = int(k)
        self.head = "gated" if (head == "context" and self.degree == 0) else head
        self.even_dim = int(even_dim)
        self.n_hist = int(n_hist)                         # > 1: the features after the lifting and every layer
        if self.n_hist > 1:
            self.lin_hist = nn.ModuleList([nn.Linear(C, out_dim if self.head == "linear" else hidden, bias=False)
                                           for _ in range(self.n_hist - 1)])
        if self.head == "linear":
            self.lin = nn.Linear(C, out_dim, bias=False)
            return
        ctx_dim = hidden if self.head == "context" else 0
        self.lin1 = nn.Linear(C, hidden, bias=False)
        if self.head == "context":
            self.lin_c = nn.Linear(C, hidden, bias=False)
            self.lin0 = nn.Linear(C, hidden)
        self.gate = GeoMLP(geo_dim, hidden + ctx_dim + 1 + self.even_dim, hidden, hidden, zero_last=True)
        self.lin2 = nn.Linear(hidden, out_dim, bias=False)
        self.skip = skip
        if skip is not None:
            (a, b), (c, d) = (skip[0] or (0, 0)), (skip[1] or (0, 0))
            self.skip_k = nn.Linear(b - a, hidden, bias=False) if b > a else None
            self.skip_km1 = nn.Linear(d - c, hidden, bias=False) if (d > c and self.degree >= 1) else None
        self.depth = int(depth)
        if self.depth > 1:                                # further odd gated stages, identity at initialisation
            self.mix = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(self.depth - 1)])
            for lin in self.mix:
                nn.init.zeros_(lin.weight)
            self.gates = nn.ModuleList([GeoMLP(geo_dim, hidden + ctx_dim + 1 + self.even_dim, hidden, hidden,
                                               zero_last=True) for _ in range(self.depth - 1)])
        self.spatial = bool(spatial)
        if self.spatial:
            if self.depth < 2:
                raise ValueError("readout_spatial needs readout_depth >= 2 (the spatial terms live in the later stages)")
            self.spat = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(self.depth - 1)])
            for lin in self.spat:
                nn.init.zeros_(lin.weight)
        self.wedge = int(wedge)
        if self.wedge:                                    # quadratic face term, zero at initialisation
            self.wedge_a = nn.Linear(C, self.wedge, bias=False)
            self.wedge_b = nn.Linear(C, self.wedge, bias=False)
            self.lin_w = nn.Linear(self.wedge, hidden, bias=False)
            nn.init.zeros_(self.lin_w.weight)

    def _earlier(self, ctx: LayerContext, like: Tensor) -> Tensor | None:
        """``sum_l W_l x_k^(l)`` over the features before the last layer (multi-layer readout), else ``None``."""
        if self.n_hist == 1:
            return None
        hist = getattr(ctx, "history", None)
        if hist is None or len(hist) != self.n_hist:
            raise ValueError(f"multi-layer readout expects {self.n_hist} feature tensors in ctx.history (the lifted "
                             f"field and one per layer), got {None if hist is None else len(hist)}")
        out = linear(self.lin_hist[0], hist[0]).to(like.dtype)
        for lin, h in zip(self.lin_hist[1:], hist[1:-1]):
            out = out + linear(lin, h).to(like.dtype)
        return out

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """Readout on degree ``k``."""
        if self.head == "linear":
            y = linear(self.lin, x[self.degree])
            early = self._earlier(ctx, y)
            return y if early is None else y + early
        k, xk = self.degree, x[self.degree]
        u = linear(self.lin1, xk)
        early = self._earlier(ctx, u)
        if early is not None:
            u = u + early
        if self.skip is not None:                                    # linear path from the raw odd inputs
            inp = ctx.inputs
            if self.skip_k is not None:
                a, b = self.skip[0]
                u = u + linear(self.skip_k, inp[k][..., a:b].to(u.dtype))
            if self.skip_km1 is not None:
                c, d = self.skip[1]
                f = inp[k - 1][..., c:d].contiguous()
                u = u + linear(self.skip_km1, spmm_ad(ctx.K.d[k - 1], ctx.K.dT[k - 1], f).to(u.dtype))
        if getattr(self, "wedge", 0):
            x1 = x[1]
            u = u + linear(self.lin_w, face_wedge(ctx.K, linear(self.wedge_a, x1), linear(self.wedge_b, x1)).to(u.dtype))
        if self.head == "context":
            K = ctx.K
            u = u + linear(self.lin_c, spmm_ad(K.d[k - 1], K.dT[k - 1], x[k - 1]).to(u.dtype))
            feats = [torch.log1p(u.square()), vertex_mean(K, k, linear(self.lin0, x[0]).to(u.dtype))]
        else:
            feats = [torch.log1p(u.square())]
        feats.append(torch.log1p(_msq(xk)).unsqueeze(-1))
        if self.even_dim:
            feats.append(ctx.even[k][..., :self.even_dim].to(u.dtype))
        e = self.gate(torch.cat(feats, dim=-1), ctx.geo[k])
        z = torch.addcmul(u, u, e)
        for s_, (mix, gate) in enumerate(zip(getattr(self, "mix", ()), getattr(self, "gates", ()))):
            u2 = z + linear(mix, z)
            if self.spatial:
                u2 = u2 + linear(self.spat[s_], self._laplacian(ctx, z))
            z = torch.addcmul(u2, u2, gate(torch.cat([torch.log1p(u2.square())] + feats[1:], dim=-1), ctx.geo[k]))
        return linear(self.lin2, z)

    def _laplacian(self, ctx: LayerContext, z: Tensor) -> Tensor:
        """Combinatorial Hodge Laplacian of degree k applied to ``z`` ``(n_k, B, H)`` (odd to odd)."""
        K, k = ctx.K, self.degree
        out = None
        if k > 0:
            out = spmm_ad(K.d[k - 1], K.dT[k - 1], spmm_ad(K.dT[k - 1], K.d[k - 1], z.contiguous()))
        if k < ctx.top:
            t = spmm_ad(K.dT[k], K.d[k], spmm_ad(K.d[k], K.dT[k], z.contiguous()))
            out = t if out is None else out + t
        return out


class EvenCellReadout(nn.Module):
    """Orientation-invariant readout on degree ``k``: ``MLP([f(x_k), log1p(|x_k|^2/C), geo_k])``.

    ``f(x_k) = log1p(x_k^2)`` (per channel) for k >= 1 and ``f(x_0) = x_0``.
    """

    def __init__(self, C: int, out_dim: int, k: int, geo_dim: int) -> None:
        super().__init__()
        self.degree = int(k)
        self.mlp = GeoMLP(geo_dim, C + 1, C, out_dim)

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``(n_k, B, C) -> (n_k, B, out_dim)``."""
        xk = x[self.degree]
        f = xk if self.degree == 0 else torch.log1p(xk.square())
        feats = torch.cat([f, torch.log1p(_msq(xk)).unsqueeze(-1)], dim=-1)
        return self.mlp(feats, ctx.geo[self.degree])


# --------------------------------------------------------------------------------------------------------------
# vector reconstruction
# --------------------------------------------------------------------------------------------------------------
_VEC_CACHE: "weakref.WeakKeyDictionary[Any, dict]" = weakref.WeakKeyDictionary()


@torch.no_grad()
def _tangent_bases(K: Any) -> Tensor:
    """Orthonormal tangent bases of a surface in 3-D, ``(n_0, 3, 2)`` (float64).

    Two smallest eigenvectors of ``N_i = sum_{f ∋ i} a_f n_f n_f^T`` (area-weighted face normal tensor; invariant
    under face orientation flips).  Polygons use their fan-triangulated vector area.
    """
    faces = K.cells[2]
    P = K.pos.to(torch.float64)
    n0 = P.shape[0]
    valid = faces >= 0
    f = faces.clamp_min(0)
    p0 = P[f[:, 0]]
    va = torch.zeros(faces.shape[0], 3, dtype=P.dtype, device=P.device)
    for j in range(1, faces.shape[1] - 1):
        cr = torch.linalg.cross(P[f[:, j]] - p0, P[f[:, j + 1]] - p0)
        va = va + torch.where(valid[:, j + 1, None], cr, torch.zeros_like(cr))
    va = 0.5 * va
    area = va.norm(dim=-1)
    Nf = va[:, :, None] * va[:, None, :] / area.clamp_min(1e-300)[:, None, None]
    N = torch.zeros(n0, 3, 3, dtype=P.dtype, device=P.device)
    for j in range(faces.shape[1]):
        w = valid[:, j]
        N.index_add_(0, f[w, j], Nf[w])
    _, evecs = torch.linalg.eigh(N)
    return evecs[..., :2].contiguous()


@torch.no_grad()
def vector_geometry(K: Any, dtype: torch.dtype, top: int) -> dict[str, Tensor | None]:
    """Geometry of the vector readout, cached per complex (weakly referenced), device and dtype.

    Args:
        K: ``CochainComplex``.
        dtype: feature dtype.
        top: top degree of the complex.

    Returns:
        dict with ``ev`` ``(n_1, D)`` edge vectors, ``t`` unit edge vectors, ``eet`` ``(n_1, 1, D*D)`` = ``ev ev^T``,
        ``deg`` ``(n_0, 1, 1)`` vertex valences (>= 1), ``E`` ``(n_0, 3, 2)`` tangent bases (surfaces in 3-D) or
        ``None``.
    """
    try:
        store = _VEC_CACHE.setdefault(K, {})
    except TypeError:  # complex not weak-referenceable: no caching
        store = {}
    key = (str(K.pos.device), dtype, top)
    if key not in store:
        ev = K.edge_vectors().to(dtype)
        n1, D = ev.shape
        ell = ev.norm(dim=-1, keepdim=True)
        t = ev / ell.clamp_min(torch.finfo(dtype).tiny)
        eet = (ev[:, :, None] * ev[:, None, :]).reshape(n1, 1, D * D).contiguous()
        ones = torch.ones(n1, 1, 1, dtype=dtype, device=ev.device)
        deg = ops.spmm(K.dT_abs[0], ones).clamp_min(1.0)
        E = _tangent_bases(K).to(dtype) if (D == 3 and top == 2) else None
        store[key] = dict(ev=ev, t=t, eet=eet, deg=deg, E=E)
    return store[key]


def _solve_spd(A: Tensor, r: Tensor, ok: Tensor) -> Tensor:
    """Closed-form solve of ``A v = r`` for SPD ``A`` ``(..., d, d)`` and ``r`` ``(..., V, d)`` (``d <= 3``)."""
    d = A.shape[-1]
    if d == 1:
        den = torch.where(ok, A[..., 0, 0], torch.ones_like(A[..., 0, 0]))
        return r / den[..., None, None]
    if d == 2:
        a, bb, c, e = A[..., 0, 0], A[..., 0, 1], A[..., 1, 0], A[..., 1, 1]
        det = torch.where(ok, a * e - bb * c, torch.ones_like(a))[..., None]
        r0, r1 = r[..., 0], r[..., 1]
        return torch.stack([(e[..., None] * r0 - bb[..., None] * r1) / det,
                            (a[..., None] * r1 - c[..., None] * r0) / det], dim=-1)
    if d == 3:
        m = [[A[..., i, j][..., None] for j in range(3)] for i in range(3)]
        cof = [[m[1][1] * m[2][2] - m[1][2] * m[2][1], m[1][2] * m[2][0] - m[1][0] * m[2][2],
                m[1][0] * m[2][1] - m[1][1] * m[2][0]],
               [m[0][2] * m[2][1] - m[0][1] * m[2][2], m[0][0] * m[2][2] - m[0][2] * m[2][0],
                m[0][1] * m[2][0] - m[0][0] * m[2][1]],
               [m[0][1] * m[1][2] - m[0][2] * m[1][1], m[0][2] * m[1][0] - m[0][0] * m[1][2],
                m[0][0] * m[1][1] - m[0][1] * m[1][0]]]
        det = m[0][0] * cof[0][0] + m[0][1] * cof[0][1] + m[0][2] * cof[0][2]
        det = torch.where(ok[..., None], det, torch.ones_like(det))
        rr = [r[..., j] for j in range(3)]
        return torch.stack([(cof[0][i] * rr[0] + cof[1][i] * rr[1] + cof[2][i] * rr[2]) / det for i in range(3)], -1)
    eye = torch.eye(d, dtype=A.dtype, device=A.device)
    Ab = torch.where(ok[..., None, None], A, eye.expand_as(A))
    return torch.linalg.solve(Ab.unsqueeze(-3), r.unsqueeze(-1)).squeeze(-1)


def ls_solve(M: Tensor, b: Tensor, reg: float = LS_REG) -> Tensor:
    """Damped least squares ``v = (M^2 + (reg * tr M)^2 I)^{-1} M b`` per cell (closed form for ``d <= 3``).

    Computed in float64 with autocast disabled; zero response along null directions of ``M``.

    Args:
        M: ``(..., d, d)`` symmetric positive semi-definite.
        b: ``(..., V, d)`` right-hand sides.
        reg: relative damping ``mu / tr(M)``.

    Returns:
        ``(..., V, d)`` in the dtype of ``b``; zero where ``tr(M) = 0`` (isolated vertices).
    """
    out_dtype = b.dtype
    with torch.autocast(device_type=M.device.type, enabled=False):
        M64, b64 = M.double(), b.double()
        d = M64.shape[-1]
        tr = torch.diagonal(M64, dim1=-2, dim2=-1).sum(-1)
        ok = tr > 0
        eye = torch.eye(d, dtype=M64.dtype, device=M64.device)
        A = M64 @ M64 + ((reg * tr) ** 2)[..., None, None] * eye
        r = torch.einsum("...ij,...vj->...vi", M64, b64)
        v = _solve_spd(A, r, ok)
        v = torch.where(ok[..., None, None], v, torch.zeros_like(v))
    return v.to(out_dtype)


class NodeVectorReadout(nn.Module):
    """Vertex vector fields from odd edge scalars (see module docstring).

    Args:
        C: channels.
        n_fields: number of vector fields ``V`` (``cfg.out_dim``); output ``(n_0, B, V * D)``.
        geo_dim1: ``G_1``.
        mode: ``'ls' | 'direct'``.
    """

    def __init__(self, C: int, n_fields: int, geo_dim1: int, mode: str = "ls") -> None:
        super().__init__()
        if mode not in ("ls", "direct"):
            raise ValueError(f"vector_mode must be 'ls' or 'direct', got {mode!r}")
        self.mode = mode
        self.V = int(n_fields)
        self.w = nn.Linear(C, self.V, bias=False)
        if mode == "ls":
            self.omega = GeoMLP(geo_dim1, C + 1, 32, 1, zero_last=True)
            with torch.no_grad():
                self.omega.fc2.bias.fill_(_SOFTPLUS_INV_ONE)

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``x[1]`` ``(n_1, B, C)`` -> ``(n_0, B, V * D)``."""
        K = ctx.K
        x1 = x[1]
        dt = ctx.dtype
        n1, B, _ = x1.shape
        n0 = K.n[0]
        geom = vector_geometry(K, dt, ctx.top)
        ev = geom["ev"]
        D, V = ev.shape[1], self.V
        w = linear(self.w, x1).to(dt)                                           # (n1, B, V)  odd
        if self.mode == "direct":
            contrib = (w.unsqueeze(-1) * geom["t"].view(n1, 1, 1, D)).reshape(n1, B, V * D)
            return spmm_ad(K.dT_abs[0], K.d_abs[0], contrib) / geom["deg"]
        feats = torch.cat([torch.log1p(x1.square()), torch.log1p(_msq(x1)).unsqueeze(-1)], dim=-1)
        omega = F.softplus(self.omega(feats, ctx.geo[1]).to(dt))               # (n1, B, 1)  even, > 0
        with torch.autocast(device_type=x1.device.type, enabled=False):         # geometric part in full precision
            M = spmm_ad(K.dT_abs[0], K.d_abs[0], (omega * geom["eet"]).contiguous()).view(n0, B, D, D)
            rhs = ((omega * w).unsqueeze(-1) * ev.view(n1, 1, 1, D)).reshape(n1, B, V * D)
            b = spmm_ad(K.dT_abs[0], K.d_abs[0], rhs).view(n0, B, V, D)
            E = geom["E"]
            if E is not None:                                                   # surface in 3-D: tangent plane
                M2 = torch.einsum("nia,nbij,njc->nbac", E, M, E)
                b2 = torch.einsum("nia,nbvi->nbva", E, b)
                v = torch.einsum("nia,nbva->nbvi", E, ls_solve(M2, b2))
            else:
                v = ls_solve(M, b)
        return v.reshape(n0, B, V * D)


class GradReadout(nn.Module):
    """``E = d_0 phi``, ``phi = MLP(x_0)`` (``head='mlp'``) or ``phi = Lin_nobias(x_0)`` (``head='linear'``, linear
    models such as the solver preset): odd edge cochain ``(n_1, B, out_dim)`` with ``d_1 E = 0`` exactly."""

    def __init__(self, C: int, out_dim: int, head: str = "mlp") -> None:
        super().__init__()
        if head not in ("mlp", "linear"):
            raise ValueError(f"grad readout head must be 'mlp' or 'linear', got {head!r}")
        self.linear = head == "linear"
        if self.linear:
            self.lin = nn.Linear(C, out_dim, bias=False)
        else:
            self.phi = NodeScalarReadout(C, out_dim)

    def potential(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """The predicted vertex potential ``phi`` ``(n_0, B, out_dim)``."""
        if self.linear:
            return linear(self.lin, x[0]).to(ctx.dtype)
        return self.phi(x, ctx).to(ctx.dtype)

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``(n_1, B, out_dim)``."""
        K = ctx.K
        return spmm_ad(K.d[0], K.dT[0], self.potential(x, ctx))


class CurlReadout(nn.Module):
    """``F = d_1 a``, ``a = Lin_nobias(x_1)``: odd face cochain ``(n_2, B, out_dim)``, closed (``d_2 F = 0``)."""

    def __init__(self, C: int, out_dim: int) -> None:
        super().__init__()
        self.lin = nn.Linear(C, out_dim, bias=False)

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``(n_2, B, out_dim)``."""
        K = ctx.K
        return spmm_ad(K.d[1], K.dT[1], linear(self.lin, x[1]).to(ctx.dtype))


class DivReadout(nn.Module):
    """``y = d_{k-1}^T b``, ``b = Lin_nobias(x_k)``: ``(n_{k-1}, B, out_dim)``, co-closed (``d_{k-2}^T y = 0``); for
    ``k = 1`` a vertex divergence with zero sum per graph."""

    def __init__(self, C: int, out_dim: int, k: int) -> None:
        super().__init__()
        if k < 1:
            raise ValueError("div readout needs k >= 1")
        self.k = int(k)
        self.lin = nn.Linear(C, out_dim, bias=False)

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``(n_{k-1}, B, out_dim)``."""
        K = ctx.K
        return spmm_ad(K.dT[self.k - 1], K.d[self.k - 1], linear(self.lin, x[self.k]).to(ctx.dtype))


class MassDivReadout(DivReadout):
    """``y = S_{k-1}^{-1} d_{k-1}^T b``, ``b = Lin_nobias(x_k)``, with ``S`` the normalised star of the output degree
    (``(n_{k-1}, B, out_dim)``); for ``k = 1`` ``sum_i star0_i y_i = 0`` exactly per graph (mass conservation)."""

    def forward(self, x: list[Tensor], ctx: LayerContext) -> Tensor:
        """``(n_{k-1}, B, out_dim)``."""
        y = super().forward(x, ctx)
        return y * torch.exp(-ctx.log_star[self.k - 1]).to(y.dtype).view(-1, 1, 1)


def build_readout(cfg: Any, geo_dims: dict[int, int]) -> nn.Module:
    """Readout module selected by ``cfg.readout`` (``cfg.out_dim``, ``cfg.vector_mode``, ``cfg.C``).

    The module carries ``feature_degree`` (hidden degree it reads) and ``output_degree`` (cells of its output).
    """
    top = max(geo_dims)
    kind, _ = parse_readout(cfg.readout)
    fdeg, odeg = readout_degrees(cfg.readout, top)
    energy = int(getattr(cfg, "readout_energy", 0) or 0)
    if energy and kind != "node_scalar":
        raise ValueError("readout_energy needs the node_scalar readout")
    if kind == "node_scalar":
        mod = NodeScalarReadout(cfg.C, cfg.out_dim, energy, float(getattr(cfg, "readout_energy_clip", 0.0) or 0.0))
    elif kind == "node_vector":
        mod = NodeVectorReadout(cfg.C, cfg.out_dim, geo_dims[1], cfg.vector_mode)
    elif kind == "cochain":
        head = getattr(cfg, "cochain_head", "linear")
        E = int(cfg.even_dims.get(fdeg, 0)) - int(getattr(cfg, "material_dims", {}).get(fdeg, 0))
        n_hist = (len(cfg.layer_types) + 1) if getattr(cfg, "readout_layers", "last") == "all" else 1
        skip = None
        if getattr(cfg, "readout_skip", False):
            if head == "linear":
                raise ValueError("readout_skip needs the gated or context cochain head")
            from .lifting import input_layout

            def odd_range(deg: int):
                if deg < 0 or deg > top or deg == 0:            # vertex columns have no orientation (not odd)
                    return None
                lay = input_layout(cfg, deg)
                return (lay["odd_lo"], lay["odd_hi"]) if lay["O"] > 0 else None
            skip = (odd_range(fdeg), odd_range(fdeg - 1))
        mod = CochainReadout(cfg.C, cfg.out_dim, fdeg, head=head, hidden=getattr(cfg, "readout_hidden", 16),
                             geo_dim=geo_dims[fdeg], even_dim=E if head != "linear" else 0, n_hist=n_hist,
                             depth=getattr(cfg, "readout_depth", 1), skip=skip,
                             spatial=getattr(cfg, "readout_spatial", False),
                             wedge=getattr(cfg, "readout_wedge", 0))
    elif kind == "even":
        mod = EvenCellReadout(cfg.C, cfg.out_dim, fdeg, geo_dims[fdeg])
    elif kind == "grad":
        mod = GradReadout(cfg.C, cfg.out_dim, getattr(cfg, "readout_head", "mlp"))
    elif kind == "curl":
        mod = CurlReadout(cfg.C, cfg.out_dim)
    elif kind == "mdiv":
        mod = MassDivReadout(cfg.C, cfg.out_dim, fdeg)
    else:
        mod = DivReadout(cfg.C, cfg.out_dim, fdeg)
    mod.feature_degree = fdeg
    mod.output_degree = odeg
    mod.feature_degrees = [fdeg, 1] if energy else [fdeg]      # degrees the last layer has to update
    return mod
