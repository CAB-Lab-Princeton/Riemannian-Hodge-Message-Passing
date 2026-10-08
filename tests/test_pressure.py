"""Components added for the AirfRANS pressure task (all optional, defaults unchanged):

* closed / harmonic generators of the first cohomology of planar domains with holes (``rhmp.dec``) and the per-cell
  domain fields ``meta['dirichlet']`` / ``meta['cohomology']`` that travel through block-diagonal batches;
* Galerkin (Whitney) energies of 1-cochains, exact for constant fields (``rhmp.dec.whitney_energy``, ``vertex_energy``);
* projection layers (``layers=[..., 'project', ...]``): metric Hodge projection of the edge features, harmonic / Kutta
  correction, vertex potential; CG and direct sparse solvers with adjoint gradients;
* the energy readout (``readout_energy``) and the energy amplitude with degree-two homogeneity
  (``amplitude_energy``, ``amplitude_power``);
* the covariant pressure tasks of ``rhmp.tasks.airfrans`` (``AFPi`` ...; data tests are skipped without ``AFM.pt``).

Every variant keeps the invariants of the method: O(C) equivariance of the stack, E(n) invariance, relabelling and
orientation parity, batch independence.
"""
from __future__ import annotations

import dataclasses
import os

import numpy as np
import pytest
import torch

from conftest import random_orthogonal
from rhmp import RHMP, RHMPConfig
from rhmp.complex import CochainComplex
from rhmp.dec import (cohomology_generators, harmonic_generators, top_measure, vertex_energy, whitney_energy)
from rhmp.layers import (PhysicalHodge, RHMPLayer, dense_projection, dirichlet_free, direct_projection, make_context,
                         metric_projection, reference_energy)
from test_equivariance import _check_OC
from test_model import F64, build_model, make_inputs, rel

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "datasets", "v2")


# ----------------------------------------------------------------------------------------------------------------
# meshes
# ----------------------------------------------------------------------------------------------------------------
def annulus(nr: int = 5, nt: int = 18, r0: float = 1.0, r1: float = 2.3, ecc: float = 0.7, jitter: float = 0.2,
            seed: int = 0):
    """Triangulated elliptic annulus (CCW faces): ``pos (nr nt, 2)``, ``faces``, outer-ring mask, inner-ring ids in
    counter-clockwise order."""
    rng = np.random.default_rng(seed)
    rr, tt = np.linspace(r0, r1, nr), np.linspace(0.0, 2 * np.pi, nt, endpoint=False)
    R, T = np.meshgrid(rr, tt, indexing="ij")
    inner = np.zeros_like(R, dtype=bool)
    inner[1:-1] = True
    R = R + inner * jitter * (r1 - r0) / (nr - 1) * rng.uniform(-1, 1, R.shape)
    T = T + jitter * (2 * np.pi / nt) * rng.uniform(-1, 1, T.shape)
    pos = np.stack([(R * np.cos(T)).ravel(), (ecc * R * np.sin(T)).ravel()], 1)
    i, j = np.meshgrid(np.arange(nr - 1), np.arange(nt), indexing="ij")
    i, j = i.ravel(), j.ravel()
    jn = (j + 1) % nt
    v00, v10, v11, v01 = i * nt + j, (i + 1) * nt + j, (i + 1) * nt + jn, i * nt + jn
    faces = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])
    outer = np.zeros(nr * nt, dtype=bool)
    outer[(nr - 1) * nt:] = True
    return pos, faces, outer, np.arange(nt)


def plate_with_holes(nx: int = 13, ny: int = 9, holes=((2, 5, 2, 4), (8, 11, 4, 7))):
    """Rectangular grid with rectangular holes (index boxes ``i0 <= i < i1``, ``j0 <= j < j1`` of the cells)."""
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    pos = np.stack([ii.ravel() / (nx - 1) * 2.0, jj.ravel() / (ny - 1)], 1).astype(np.float64)
    faces = []
    for i in range(nx - 1):
        for j in range(ny - 1):
            if any(a <= i < b and c <= j < d for a, b, c, d in holes):
                continue
            v00, v10, v11, v01 = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
            faces += [[v00, v10, v11], [v00, v11, v01]]
    faces = np.array(faces)
    used = np.unique(faces)
    remap = -np.ones(len(pos), dtype=np.int64)
    remap[used] = np.arange(len(used))
    return pos[used], remap[faces]


def rhombus_domain(seed: int = 0):
    """Disk of radius 1.6 with a thin rhombic hole (half-axes 0.5 and 0.05: two sharp tips at (+-0.5, 0)), Delaunay
    triangulated: ``pos``, ``faces`` (CCW), outer mask, ids of the two tip vertices."""
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    corners = np.array([[0.5, 0.0], [0.0, 0.05], [-0.5, 0.0], [0.0, -0.05]])
    hole = np.concatenate([corners[i] + np.outer(np.linspace(0, 1, 16, endpoint=False), corners[(i + 1) % 4] - corners[i])
                           for i in range(4)])
    tt = np.linspace(0, 2 * np.pi, 32, endpoint=False)
    ring = 1.6 * np.stack([np.cos(tt), np.sin(tt)], 1)

    def inside(P):                                              # |x| / 0.5 + |y| / 0.05 < 1
        return np.abs(P[:, 0]) / 0.5 + np.abs(P[:, 1]) / 0.05 < 1.0
    pts = rng.uniform(-1.5, 1.5, size=(900, 2))
    pts = pts[(np.linalg.norm(pts, axis=1) < 1.5) & (np.abs(pts[:, 0]) / 0.56 + np.abs(pts[:, 1]) / 0.11 > 1.0)]
    pos = np.concatenate([hole, ring, pts])
    F = Delaunay(pos).simplices.astype(np.int64)
    P = pos[F]
    sa = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])
    F[sa < 0] = F[sa < 0][:, [0, 2, 1]]
    F = F[~inside(pos[F].mean(1))]
    outer = np.zeros(len(pos), dtype=bool)
    outer[len(hole):len(hole) + len(ring)] = True
    return pos, F, outer, np.array([0, 32])


def domain(pos, faces, outer, device="cpu", dtype=F64, star: str = "cotan") -> CochainComplex:
    """Annulus complex with the domain fields of a projection layer (outer ring fixed, harmonic generator)."""
    K = CochainComplex.from_triangles(pos, faces, star=star, device=device)
    K.meta["dirichlet"] = {0: torch.as_tensor(outer, device=device)}
    K.meta["cohomology"] = {1: harmonic_generators(K, dtype=dtype)}
    return K.to(dtype=dtype)


def loop_sum(K, c: torch.Tensor, loop: np.ndarray) -> torch.Tensor:
    """Sum of an edge cochain along the closed vertex loop ``loop`` (in its own direction)."""
    key = {(int(a), int(b)): k for k, (a, b) in enumerate(K.cells[1].tolist())}
    tot = torch.zeros(c.shape[1:], dtype=c.dtype)
    for a, b in zip(loop.tolist(), np.roll(loop, -1).tolist()):
        tot = tot + (c[key[(a, b)]] if (a, b) in key else -c[key[(b, a)]])
    return tot


def proj_cfg(**kw) -> RHMPConfig:
    base = dict(in_dims={0: 2, 1: 2, 2: 1}, even_dims={1: 1, 2: 1}, C=8, layers=["project", "poly"], gate="homog",
                solve_iters=4000, solve_tol=1e-13)
    base.update(kw)
    return RHMPConfig(**base)


# ----------------------------------------------------------------------------------------------------------------
# cohomology generators
# ----------------------------------------------------------------------------------------------------------------
def test_cohomology_generators_annulus_disk_two_holes():
    pos, faces, outer, inner = annulus()
    K = CochainComplex.from_triangles(pos, faces)
    c = cohomology_generators(K)
    assert c.shape == (K.n[1], 1) and set(c.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert float(K.apply_d(1, c).abs().max()) == 0.0                          # exactly closed
    assert abs(float(loop_sum(K, c, inner)[0])) == 1.0                        # period +-1 around the hole: not exact
    g = harmonic_generators(K, dtype=F64)
    Kd = K.to(dtype=F64)
    assert float(Kd.apply_d(1, g).abs().max()) < 1e-12
    assert float(Kd.apply_dT(0, (Kd.star[1][:, None] * g)).abs().max()) < 1e-12   # co-closed, natural on both loops
    assert abs(abs(float(loop_sum(K, g, inner)[0])) - 1.0) < 1e-12
    # a disk has no generator; a plate with two holes has two independent ones
    from conftest import mesh_plane_grid
    assert cohomology_generators(CochainComplex.from_triangles(*mesh_plane_grid())).shape[1] == 0
    K2 = CochainComplex.from_triangles(*plate_with_holes())
    assert K2.n[0] - K2.n[1] + K2.n[2] == -1                                   # Euler characteristic 1 - b1
    c2 = cohomology_generators(K2)
    assert c2.shape[1] == 2 and float(K2.apply_d(1, c2).abs().max()) == 0.0
    g2 = harmonic_generators(K2, dtype=F64)
    gram = g2.T @ (K2.star[1].double()[:, None] * g2)
    assert float(torch.linalg.det(gram)) > 1e-6 * float(gram.diagonal().prod())
    # inconsistent face orientations are refused
    bad = faces.copy()
    bad[::3] = bad[::3][:, ::-1]
    with pytest.raises(ValueError):
        cohomology_generators(CochainComplex.from_triangles(pos, bad))


def test_batch_carries_domain_fields(device):
    Ks = [domain(*annulus(seed=s, nt=16 + 2 * s)[:3], device=device, dtype=torch.float32) for s in range(2)]
    Ks.append(CochainComplex.from_triangles(*plate_with_holes(), device=device))
    Ks[2].meta["dirichlet"] = {0: Ks[2].boundary[0].clone()}
    Ks[2].meta["cohomology"] = {1: harmonic_generators(Ks[2])}
    Kb = CochainComplex.batch(Ks)
    assert Kb.meta["dirichlet"][0].shape == (Kb.n[0],) and Kb.meta["cohomology"][1].shape == (Kb.n[1], 2)
    off = np.cumsum([0] + [K.n[1] for K in Ks])
    for i, K in enumerate(Ks):
        part = Kb.meta["cohomology"][1][off[i]:off[i + 1]]
        r = K.meta["cohomology"][1].shape[1]
        assert torch.equal(part[:, :r], K.meta["cohomology"][1]) and float(part[:, r:].abs().sum()) == 0.0
    plain = CochainComplex.batch([Ks[0], CochainComplex.from_triangles(*annulus(seed=5)[:2], device=device)])
    assert "cohomology" not in plain.meta and "dirichlet" not in plain.meta


# ----------------------------------------------------------------------------------------------------------------
# Galerkin energies
# ----------------------------------------------------------------------------------------------------------------
def test_whitney_energy_exact_for_constant_fields(device):
    """Exact up to the float32 rounding of the stored Galerkin blocks (amplified by the squared aspect ratio of a
    cell: 1e-3 on the sliver-heavy Delaunay fixture, 1e-5 on well-shaped meshes)."""
    from conftest import make_complex
    for name, tol in (("grid", 2e-5), ("delaunay", 2e-3), ("tets", 2e-4)):
        K = make_complex(name, device).to(dtype=F64)
        D = K.pos.shape[1]
        g = torch.Generator().manual_seed(0)
        U = torch.randn(3, D, generator=g, dtype=F64).to(device)
        ev = K.edge_vectors().double()
        a = ev @ U.T                                                           # (n1, 3) cochains of constant fields
        meas = top_measure(K)
        e = whitney_energy(K, a[:, :2], a[:, 1:])                              # pairs (0, 1), (1, 2)
        expect = meas[:, None] * torch.stack([U[0] @ U[1], U[1] @ U[2]])
        assert rel(e, expect) < tol, (name, rel(e, expect))
        q = vertex_energy(K, a[:, :2], a[:, 1:])
        assert rel(q, torch.stack([U[0] @ U[1], U[1] @ U[2]]).expand_as(q)) < tol
        assert bool((whitney_energy(K, torch.randn(K.n[1], 4, generator=g, dtype=F64).to(device)) > 0).all())
    # the lumped vertex formula is not exact on an irregular mesh (reason for the Galerkin form)
    K = make_complex("delaunay", device).to(dtype=F64)
    a = K.edge_vectors().double() @ torch.tensor([1.0, 0.3], dtype=F64, device=device)
    lumped = K.apply_dT_abs(0, (K.star[1] * a * a)[:, None])[:, 0] / (2 * K.star[0])
    assert float((lumped / 1.09 - 1).abs().max()) > 0.05


def test_aspect_weighting_of_the_vertex_energy(device):
    from rhmp.dec import ASPECT_CAP, top_aspect
    eq = CochainComplex.from_triangles(np.array([[0.0, 0.0], [1.0, 0.0], [0.5, np.sqrt(3) / 2]]),
                                       np.array([[0, 1, 2]]), device=device)
    assert abs(float(top_aspect(eq)) - 2 / np.sqrt(3)) < 1e-6                  # float32 positions
    # a strip of well-shaped triangles with one vertex pulled onto its neighbour's edge: slivers of aspect ~ 1e3
    pos, faces, outer, _ = annulus(nr=4, nt=12, jitter=0.0)
    pos = pos.copy()
    pos[13] = 0.999 * pos[12] + 0.001 * pos[14]
    K = CochainComplex.from_triangles(pos, faces, device=device).to(dtype=F64)
    asp = top_aspect(K)
    assert float(asp.max()) > 10 * ASPECT_CAP and float(asp.median()) < 4
    g = torch.Generator().manual_seed(0)
    a = torch.randn(K.n[1], 2, generator=g, dtype=F64).to(device)              # O(1) edge values: not cochain-like
    q_cap, q_raw = vertex_energy(K, a), vertex_energy(K, a, aspect_cap=None)
    touched = torch.zeros(K.n[0], dtype=torch.bool, device=device)
    touched[K.cells[2][asp > ASPECT_CAP].reshape(-1)] = True
    assert float(q_raw[touched].abs().max()) > 20 * float(q_cap[touched].abs().max())   # the sliver blow-up is damped
    torch.testing.assert_close(q_cap[~touched], q_raw[~touched])                # other vertices: unchanged
    u = torch.tensor([0.7, -1.2], dtype=F64, device=device)                   # still exact for constant fields
    th = (K.edge_vectors().double() @ u)[:, None]
    assert rel(vertex_energy(K, th), (u @ u).expand(K.n[0], 1)) < 1e-3


def test_energy_amplitude_is_the_field_magnitude_on_any_mesh():
    """``s_E`` of the 1-form of a uniform field is ``|u|`` whatever the mesh; the RMS cochain value is not."""
    u = torch.tensor([3.0, -1.5], dtype=F64)
    vals = []
    for nr, nt in ((5, 18), (9, 40)):
        pos, faces, outer, _ = annulus(nr=nr, nt=nt)
        K = CochainComplex.from_triangles(pos, faces).to(dtype=F64)
        th = (K.edge_vectors().double() @ u)[:, None, None]
        cfg = RHMPConfig(in_dims={0: 1, 1: 1}, C=4, n_layers=1, amplitude_dims={1: 1}, amplitude_power=2,
                         amplitude_energy=True)
        m = RHMP(cfg, K.geo_dims).double()
        inp, ctx = m._prepare({0: torch.zeros(K.n[0], 1, 1, dtype=F64), 1: th}, K)
        vals.append((float(ctx.amplitude), float(th.square().mean().sqrt())))
        assert abs(float(reference_energy(ctx)) * float(th.square().mean()) / float(u @ u) - 1.0) < 1e-6
    assert abs(vals[0][0] / float(u.norm()) - 1) < 1e-6 and abs(vals[1][0] / float(u.norm()) - 1) < 1e-6
    assert abs(vals[0][1] / vals[1][1] - 1.0) > 0.3                           # RMS cochain values scale with h


# ----------------------------------------------------------------------------------------------------------------
# metric projection
# ----------------------------------------------------------------------------------------------------------------
def _projection_setup(device, B=2, C=3, seed=0):
    K = domain(*annulus(seed=seed)[:3], device=device)
    ctx = make_context(K, scaling="dec", B=B, dtype=F64)
    g = torch.Generator().manual_seed(seed)
    x1 = torch.randn(K.n[1], B, C, generator=g, dtype=F64).to(device)
    H = torch.exp(torch.randn(K.n[1], B, generator=g, dtype=F64)).to(device)
    return K, ctx, x1, H


def test_metric_projection_cg_and_direct(device):
    K, ctx, x1, H = _projection_setup(device)
    free = dirichlet_free(ctx, 0, "dirichlet")
    up = ("diag", ctx.blocks[(0, True)], H.unsqueeze(-1))
    al, dal, res, n_it = metric_projection(ctx, x1, up, free, 4000, 1e-13)
    al2, dal2 = direct_projection(ctx, x1, H, free)
    al3, _, _, _ = metric_projection(ctx, x1, up, free, 4000, 1e-13, "twolevel")
    assert rel(al, al2) < 1e-9 and rel(dal, dal2) < 1e-9 and rel(al3, al2) < 1e-9 and float(res.max()) < 1e-12
    assert rel(dal2, K.apply_d(0, al2)) < 1e-12
    u = x1 - dal2
    div = K.apply_dT(0, (H.unsqueeze(-1) * u).contiguous())                   # d0^T H u: zero on the free vertices
    assert float((div * free).abs().max()) < 1e-9 * float(div.abs().max())
    assert float((al2 * (1 - free)).abs().max()) == 0.0
    al4, dal4 = direct_projection(ctx, u, H, free)                             # idempotent
    assert float(dal4.abs().max()) < 1e-9
    f = torch.randn(K.n[0], 2, 3, generator=torch.Generator().manual_seed(1), dtype=F64).to(device) * free
    _, d5 = direct_projection(ctx, K.apply_d(0, f.contiguous()), H, free)      # exact forms vanishing on the ring
    assert rel(d5, K.apply_d(0, f.contiguous())) < 1e-9
    _, d6 = direct_projection(ctx, x1, 7.5 * H, free)                          # independent of the scale of H
    assert rel(d6, dal2) < 1e-10
    # no fixed vertices: the zero-mean potential of the pure natural problem, same d0 alpha from both solvers
    a7, d7 = direct_projection(ctx, x1, H, None)
    a8, d8, _, _ = metric_projection(ctx, x1, up, None, 6000, 1e-13)
    assert rel(d7, d8) < 1e-7 and float(a7.mean(0).abs().max()) < 1e-10 and float(a8.mean(0).abs().max()) < 1e-10
    # dense Cholesky solver: equal to the sparse direct solver, with and without fixed vertices
    a9, d9 = dense_projection(ctx, x1, up, free)
    assert rel(a9, al2) < 1e-9 and rel(d9, dal2) < 1e-9
    a10, d10 = dense_projection(ctx, x1, up, None)
    assert rel(d10, d7) < 1e-9 and rel(a10, a7) < 1e-9 and float(a10.mean(0).abs().max()) < 1e-10


def test_metric_projection_gradients():
    pos, faces, outer, _ = annulus(nr=3, nt=7)
    K = domain(pos, faces, outer)
    ctx = make_context(K, scaling="dec", B=1, dtype=F64)
    free = dirichlet_free(ctx, 0, "dirichlet")
    g = torch.Generator().manual_seed(0)
    x1 = torch.randn(K.n[1], 1, 2, generator=g, dtype=F64, requires_grad=True)
    lh = torch.randn(K.n[1], 1, generator=g, dtype=F64, requires_grad=True)
    w = torch.randn(K.n[0], 1, 2, generator=g, dtype=F64)
    wd = torch.randn(K.n[1], 1, 2, generator=g, dtype=F64)

    def direct(x, logh):
        a, da = direct_projection(ctx, x, torch.exp(logh), free)
        return (a * w).sum() + (da * wd).sum()

    def cg(x, logh):
        up = ("diag", ctx.blocks[(0, True)], torch.exp(logh).unsqueeze(-1))
        a, da, _, _ = metric_projection(ctx, x, up, free, 3000, 1e-14)
        return (a * w).sum() + (da * wd).sum()

    def dense(x, logh):
        up = ("diag", ctx.blocks[(0, True)], torch.exp(logh).unsqueeze(-1))
        a, da = dense_projection(ctx, x, up, free)
        return (a * w).sum() + (da * wd).sum()

    assert torch.autograd.gradcheck(direct, (x1, lh), eps=1e-6, atol=1e-6, rtol=1e-5)
    gd = torch.autograd.grad(direct(x1, lh), (x1, lh))
    gc = torch.autograd.grad(cg(x1, lh), (x1, lh))
    assert rel(gc[0], gd[0]) < 1e-8 and rel(gc[1], gd[1]) < 1e-8
    ge = torch.autograd.grad(dense(x1, lh), (x1, lh))                           # autograd through the factorisation
    assert rel(ge[0], gd[0]) < 1e-8 and rel(ge[1], gd[1]) < 1e-8


def test_potential_flow_and_kutta_projection(device):
    """With the reference metric the layer's solve is the P1 potential flow: no flux through the inner boundary, the
    perturbation vanishes on the ring; the Kutta step makes the result W-orthogonal to the harmonic cochain and does
    not depend on the sign / scale of the generator or on a multiple of it added to the input."""
    pos, faces, outer, inner = annulus(nr=7, nt=30, jitter=0.1)
    K = domain(pos, faces, outer, device=device)
    layer = RHMPLayer(2, K.geo_dims, {1: 1, 2: 1}, C=3, kind="project", gate="none", learn_metric=False,
                      solve_iters=4000, solve_tol=1e-13).to(device).double()
    with torch.no_grad():
        for p in layer.kutta.parameters():
            p.copy_(torch.randn(p.shape, generator=torch.Generator().manual_seed(3), dtype=F64) * 0.5)
    g = torch.Generator().manual_seed(0)
    B, C = 2, 3
    even = [None, torch.rand(K.n[1], B, 1, generator=g, dtype=F64).to(device),
            torch.rand(K.n[2], B, 1, generator=g, dtype=F64).to(device)]
    u_inf = torch.tensor([1.0, 0.4], dtype=F64, device=device)
    theta = (K.edge_vectors().double() @ u_inf)[:, None, None].expand(-1, B, C).contiguous()
    x0 = torch.zeros(K.n[0], B, C, dtype=F64, device=device)
    x2 = torch.zeros(K.n[2], B, C, dtype=F64, device=device)

    def run(Kc, x1):
        ctx = make_context(Kc, scaling="dec", B=B, dtype=F64, even=even)
        return layer([x0, x1, x2], ctx), ctx

    (y0, y1, _), ctx = run(K, theta)
    assert float(K.apply_d(1, y1).abs().max()) < 1e-10                         # closed: a potential flow + circulation
    star1 = K.star[1][:, None, None]
    div = K.apply_dT(0, (star1 * y1).contiguous())
    free = dirichlet_free(ctx, 0, "dirichlet")
    assert float((div * free).abs().max()) < 1e-9                              # incl. the wall: no flux through it
    # Kutta: W-orthogonal to the (projected) harmonic cochain
    gam = K.meta["cohomology"][1]
    e = K.cells[1].long()
    g0 = ctx.geo[0]
    geo = torch.cat([ctx.geo[1], 0.5 * (g0[e[:, 0]] + g0[e[:, 1]]), (g0[e[:, 0]] - g0[e[:, 1]]).abs()], -1)
    en = torch.exp(ctx.log_star[1])[:, None] * gam.square().sum(-1, keepdim=True).expand(-1, B)
    lg = torch.log((en + 1e-12 * en.mean(0) + 1e-30) / (en.mean(0) + 1e-30)).unsqueeze(-1)
    W = torch.exp(ctx.log_star[1])[:, None] * torch.exp(layer.kutta(geo, torch.cat([lg, even[1]], -1)))
    inner_prod = (W[..., None] * gam[:, None, :] * y1).sum(0)
    assert float(inner_prod.abs().max()) < 1e-9 * float((W[..., None] * gam[:, None, :].abs() * y1.abs()).sum(0).max())
    assert abs(float(loop_sum(K, y1[:, 0, 0].cpu(), inner))) > 1e-3            # ... with a non-zero circulation
    # invariances of the Kutta step
    (z0, z1, _), _ = run(K, theta + 0.7 * gam[:, None, :].expand(-1, B, C))
    assert rel(z1, y1) < 1e-9
    K2 = dataclasses.replace(K, meta={**K.meta, "cohomology": {1: -2.5 * gam}})
    (_, w1, _), _ = run(K2, theta)
    assert rel(w1, y1) < 1e-9
    # without generators (or with project_harmonic=False) the layer is the plain projection: zero circulation
    K3 = dataclasses.replace(K, meta={k: v for k, v in K.meta.items() if k != "cohomology"})
    (_, v1, _), _ = run(K3, theta)
    assert abs(float(loop_sum(K, v1[:, 0, 0].cpu(), inner))) < 1e-9
    # direct solver: same result
    layer.project_solver = "direct"
    (d0_, d1_, _), _ = run(K, theta)
    assert rel(d1_, y1) < 1e-8 and rel(d0_, y0) < 1e-8


# ----------------------------------------------------------------------------------------------------------------
# model-level symmetries with projection layers, the energy readout and the energy amplitude
# ----------------------------------------------------------------------------------------------------------------
FULL = dict(amplitude_dims={1: 1}, amplitude_power=2, amplitude_energy=True, readout_energy=3, lift_vertex=True,
            lift_context=4)
P_VARIANTS = {
    "project": dict(),
    "project_direct": dict(project_solver="direct"),
    "project_tensor": dict(metric_type="tensor"),
    "project_dense": dict(project_solver="dense"),
    "project_tensor_dense": dict(metric_type="tensor", project_solver="dense"),
    "project_untied_none": dict(tie_metrics=False, scaling="none"),
    "poly_project_poly": dict(layers=["poly", "project", "poly"]),
    "full": dict(FULL),
    "full_direct": dict(FULL, project_solver="direct"),
    "energy_only": dict(layers=["poly", "poly"], readout_energy=4),
}


@pytest.mark.parametrize("variant", list(P_VARIANTS))
def test_OC_equivariance_projection(variant, device):
    K = domain(*annulus()[:3], device=device)
    _check_OC(K, proj_cfg(**P_VARIANTS[variant]), F64, 1e-8)


@pytest.mark.parametrize("variant", ["project", "full", "full_direct", "energy_only"])
@pytest.mark.parametrize("reflect", [False, True])
def test_En_invariance_projection(variant, reflect, device):
    pos, faces, outer, _ = annulus()
    R = random_orthogonal(2, seed=4, reflect=reflect)
    faces2 = faces[:, ::-1].copy() if reflect else faces                       # keep the faces counter-clockwise
    K = domain(pos, faces, outer, device=device)
    K2 = domain(pos @ R.T + np.array([2.0, -1.0]), faces2, outer, device=device)
    cfg = proj_cfg(**P_VARIANTS[variant])
    m = build_model(cfg, K, dtype=F64)
    inp = make_inputs(cfg, K, dtype=F64)
    y = m(inp, K)
    if reflect:                                                                # same cells, opposite face orientation:
        idx = torch.arange(K.n[2], device=device)                              # odd face inputs change sign (none here)
        assert torch.equal(K2.cells[1], K.cells[1])
    y2 = m(inp, K2)
    # the Galerkin blocks of the energy readout / amplitude are stored in float32 (the plain projection: 1e-7 from
    # the float32 positions behind the generators)
    assert rel(y2, y) < (1e-7 if variant == "project" else 2e-6), rel(y2, y)


@pytest.mark.parametrize("variant", ["project", "full_direct", "energy_only"])
def test_vertex_relabelling_projection(variant, device):
    from test_equivariance import _cell_maps, _map_cells
    pos, faces, outer, _ = annulus()
    perm = np.random.default_rng(2).permutation(len(pos))
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    K = domain(pos, faces, outer, device=device)
    K2 = domain(pos[perm], inv[faces], outer[perm], device=device)
    maps = _cell_maps(K, K2, inv)
    cfg = proj_cfg(**P_VARIANTS[variant])
    m = build_model(cfg, K, dtype=F64)
    inp = make_inputs(cfg, K, dtype=F64)
    permt = torch.as_tensor(perm, device=device)
    inp2 = {0: inp[0][permt]}
    for k in (1, 2):
        inp2[k] = _map_cells(inp[k], *maps[k], n_odd=cfg.in_dims[k] - cfg.even_dims.get(k, 0))
    assert rel(m(inp2, K2), m(inp, K)[permt]) < 1e-7


@pytest.mark.parametrize("variant", ["full", "full_direct"])
def test_batch_of_meshes_equals_parts_projection(variant, device):
    Ks = [domain(*annulus(seed=s, nt=16 + 2 * s, nr=4 + s)[:3], device=device) for s in range(3)]
    Kb = CochainComplex.batch(Ks)
    cfg = proj_cfg(**P_VARIANTS[variant])
    m = build_model(cfg, Ks[0], dtype=F64)
    inps = [make_inputs(cfg, K, B=1, seed=s, dtype=F64) for s, K in enumerate(Ks)]
    yb = m({k: torch.cat([i[k] for i in inps]) for k in inps[0]}, Kb)
    ys = torch.cat([m(i, K) for i, K in zip(inps, Ks)])
    assert rel(yb, ys) < 1e-8


@pytest.mark.parametrize("metric", ["diag", "tensor"])
def test_dense_solver_equals_cg_in_the_model(metric, device):
    """The dense Cholesky solver gives the same model outputs as CG converged to round-off (diagonal and tensor
    metrics) and refuses block-diagonal batches of meshes."""
    pos, faces, outer, _ = annulus()
    K = domain(pos, faces, outer, device=device)
    cfg_cg = proj_cfg(metric_type=metric, solve_iters=6000, solve_tol=1e-13)
    cfg_de = proj_cfg(metric_type=metric, project_solver="dense")
    inp = make_inputs(cfg_cg, K, dtype=F64)
    y_cg = build_model(cfg_cg, K, dtype=F64)(inp, K)
    y_de = build_model(cfg_de, K, dtype=F64)(inp, K)
    assert rel(y_de, y_cg) < 1e-8, rel(y_de, y_cg)
    Kb = CochainComplex.batch([K, domain(*annulus(seed=1, nt=18)[:3], device=device)])
    with pytest.raises(NotImplementedError):
        build_model(cfg_de, K, dtype=F64)(make_inputs(cfg_cg, Kb, B=1, dtype=F64), Kb)


def test_kutta_corner_prior(device):
    """The corner indicator marks the boundary edges at sharp re-entrant corners only; with the prior the Kutta
    weights start there (a circulation at initialisation), without it they start uniform (none)."""
    from rhmp.layers import boundary_corner
    pos, faces, outer, tips = rhombus_domain()
    K = domain(pos, faces, outer, device=device)
    assert K.n[0] - K.n[1] + K.n[2] == 0 and K.meta["cohomology"][1].shape[1] == 1
    ctx = make_context(K, scaling="dec", B=1, dtype=F64)
    c = boundary_corner(ctx)
    e = K.cells[1].cpu().numpy()
    at_tip = torch.as_tensor(np.isin(e, tips).any(1) & K.boundary[1].cpu().numpy().astype(bool), device=device)
    assert int(at_tip.sum()) == 4 and float(c[at_tip].min()) > 0.7 and float(c[~at_tip].max()) == 0.0
    smooth = domain(*annulus()[:3], device=device)                            # smooth hole: no corner anywhere
    assert float(boundary_corner(make_context(smooth, scaling="dec", B=1, dtype=F64)).max()) == 0.0
    u_inf = torch.tensor([1.0, 0.25], dtype=F64, device=device)
    share = {}
    for corner in (0.0, 4.0):
        cfg = proj_cfg(**FULL, project_solver="direct", kutta_corner=corner, kutta_range=12.0)
        m = build_model(cfg, K, randomize=False, dtype=F64)
        assert (m.layers[0].kutta.corner is not None) == (corner > 0)
        inp = make_inputs(cfg, K, dtype=F64)
        inp[1][..., 0] = (K.edge_vectors().double() @ u_inf)[:, None]
        f = m.projection_fields(inp, K)
        wg = K.star[1] * torch.exp(f["kappa"][:, 0]) * f["gamma"][:, 0, 0] ** 2
        share[corner] = float(wg[at_tip].sum() / wg.sum())
        circ = float((f["Gamma"]).abs().max())
        assert (circ > 1e-3) == (corner > 0)                                    # uniform W: no circulation at init
    assert share[4.0] > 0.99 and share[0.0] < 0.5
    cfg = proj_cfg(**FULL, kutta_corner=4.0)                                    # symmetries with the prior
    _check_OC(K, cfg, F64, 1e-8)
    R = random_orthogonal(2, seed=4, reflect=True)
    K2 = domain(pos @ R.T + np.array([0.5, 2.0]), faces[:, ::-1].copy(), outer, device=device)
    m = build_model(cfg, K, dtype=F64)
    inp = make_inputs(cfg, K, dtype=F64)
    assert rel(m(inp, K2), m(inp, K)) < 2e-6


def test_projection_fields_accessor(device):
    """``RHMP.projection_fields``: the unit inflow through the first projection layer with the model's metrics."""
    pos, faces, outer, inner = annulus(nr=6, nt=24, jitter=0.1)
    K = domain(pos, faces, outer, device=device)
    cfg = proj_cfg(**FULL, project_solver="direct")
    m = build_model(cfg, K, dtype=F64)
    inp = make_inputs(cfg, K, dtype=F64)
    u_inf = torch.tensor([1.0, 0.3], dtype=F64, device=device)
    inp[1][..., 0] = (K.edge_vectors().double() @ u_inf)[:, None]             # amplitude column: a uniform inflow
    f = m.projection_fields(inp, K)
    assert f["u"].shape == (K.n[1], 2, 1) and f["kappa"].shape == (K.n[1], 2) and f["Gamma"].shape == (1, 2, 1, 1)
    assert float(K.apply_d(1, f["u"]).abs().max()) < 1e-9 * float(f["u"].abs().max()) + 1e-12   # closed
    W = K.star[1][:, None] * torch.exp(f["kappa"])                             # W-orthogonal to the harmonic cochain
    num = (W[..., None] * f["gamma"] * f["u"]).sum(0)
    assert float(num.abs().max()) < 1e-8 * float((W[..., None] * f["gamma"].abs() * f["u"].abs()).sum(0).max())
    circ = loop_sum(K, f["u"][:, 0, 0].cpu(), inner)
    circ0 = loop_sum(K, f["u_no_circulation"][:, 0, 0].cpu(), inner)
    assert abs(float(circ0)) < 1e-9 and abs(float(circ)) > 1e-4               # the Kutta step adds the circulation
    with pytest.raises(ValueError):
        build_model(proj_cfg(layers=["poly", "poly"]), K, dtype=F64).projection_fields(inp, K)


def test_degree_two_homogeneity_and_energy_readout_init(device):
    K = domain(*annulus()[:3], device=device)
    cfg = proj_cfg(**FULL)
    m = build_model(cfg, K, dtype=F64)
    inp = make_inputs(cfg, K, dtype=F64)
    y = m(inp, K)
    for lam in (0.3, 7.0):
        inp2 = dict(inp)
        inp2[1] = torch.cat([lam * inp[1][..., :1], inp[1][..., 1:]], -1)       # scale the amplitude column only
        assert rel(m(inp2, K), lam ** 2 * y) < 1e-9
    # zero-initialised energy weights: the plain readout with the same parameters
    cfg0 = proj_cfg(lift_vertex=True, lift_context=4)
    m0 = build_model(cfg0, K, randomize=False, dtype=F64)
    m1 = RHMP(dataclasses.replace(cfg0, readout_energy=5), K.geo_dims).to(device=device, dtype=F64)
    missing = m1.load_state_dict(m0.state_dict(), strict=False).missing_keys
    assert missing and all(k.startswith("readout.en_") for k in missing)
    inp0 = make_inputs(cfg0, K, dtype=F64)
    assert rel(m1(inp0, K), m0(inp0, K)) < 1e-12
    hid = [v for _, v in sorted(m1.hidden(inp0, K).items())]
    q = m1.readout.energies(hid, m1._prepare(inp0, K)[1])
    assert q.shape == (K.n[0], 2, 5) and bool(torch.isfinite(q).all())
    m1.readout.energy_clip = 0.5                                              # bounded energy channels
    qc = m1.readout.energies(hid, m1._prepare(inp0, K)[1])
    torch.testing.assert_close(qc, 0.5 * torch.tanh(q / 0.5))
    assert float(qc.abs().max()) <= 0.5
    with pytest.raises(ValueError):
        RHMPConfig(in_dims={0: 1, 1: 1}, readout="cochain:1", readout_energy=2)
    with pytest.raises(ValueError):
        RHMPConfig(in_dims={0: 1, 1: 1}, amplitude_energy=True)


def test_defaults_unchanged_and_configs_round_trip():
    cfg = RHMPConfig(in_dims={0: 2, 1: 1})
    assert (cfg.amplitude_power, cfg.amplitude_energy, cfg.readout_energy, cfg.project_solver) == (1, False, 0, "cg")
    old = {k: v for k, v in cfg.to_dict().items()
           if k not in ("amplitude_power", "amplitude_energy", "readout_energy", "project_harmonic", "kutta_range",
                        "project_solver", "readout_energy_clip", "kutta_corner")}
    assert RHMPConfig.from_dict(old) == cfg
    full = proj_cfg(**FULL, project_solver="direct", kutta_range=5.0, project_harmonic=False)
    assert RHMPConfig.from_dict(full.to_dict()) == full
    K = domain(*annulus()[:3], dtype=torch.float32)
    m = RHMP(full, K.geo_dims)
    assert m.layers[0].kutta is None and len(m.layers[0].gates) == 2 and set(m.layers[0].heads) == {"H1"}
    m2 = RHMP.from_checkpoint(m.to_checkpoint())
    inp = make_inputs(full, K)
    torch.testing.assert_close(m2(inp, K), m(inp, K))
    with pytest.raises(ValueError):
        RHMPConfig(in_dims={0: 1}, layers=["project"], scaling="jacobi").layer_types and \
            RHMP(RHMPConfig(in_dims={0: 1, 1: 1}, layers=["project"], scaling="jacobi"), K.geo_dims)


def test_trainer_flags_pressure():
    from rhmp.data import TaskData
    from rhmp.train import build_config, needs_whitney, parse_args
    K = domain(*annulus()[:3], dtype=torch.float32)
    td = TaskData(name="x", K=[K], inputs=[{0: torch.zeros(K.n[0], 2), 1: torch.zeros(K.n[1], 2),
                                            2: torch.zeros(K.n[2], 1)}],
                  target=[torch.zeros(K.n[0], 1)], target_degree=0, target_kind="node_scalar",
                  in_dims={0: 2, 1: 2, 2: 1}, even_dims={1: 1, 2: 1}, connection_dims={},
                  split=(torch.arange(1), torch.arange(0), torch.arange(0)), x_stats={}, y_stats=None, spatial_dim=2)
    a = parse_args(["--task", "AFPi", "--layers", "project,poly", "--amplitude", "1:1", "--amplitude-power", "2",
                    "--amplitude-energy", "--readout-energy", "8", "--project-solver", "direct", "--kutta-range", "6",
                    "--gate", "homog"])
    cfg = build_config(td, a)
    assert (cfg.layers, cfg.amplitude_dims, cfg.amplitude_power, cfg.amplitude_energy, cfg.readout_energy,
            cfg.project_solver, cfg.kutta_range, cfg.project_harmonic) == (
        ["project", "poly"], {1: 1}, 2, True, 8, "direct", 6.0, True)
    assert needs_whitney(cfg.to_dict()) and not needs_whitney(RHMPConfig(in_dims={0: 1}).to_dict())
    assert build_config(td, parse_args(["--task", "AFPi", "--layers", "project", "--no-kutta"])).project_harmonic is False
    assert build_config(td, parse_args(["--task", "AFPi"])) == RHMPConfig(
        in_dims={0: 2, 1: 2, 2: 1}, even_dims={1: 1, 2: 1}, C=128, n_layers=4)


# ----------------------------------------------------------------------------------------------------------------
# tasks
# ----------------------------------------------------------------------------------------------------------------
def test_offset_map_and_mixed_stats():
    from rhmp.tasks.airfrans import OffsetMap, _mixed_stats
    om = OffsetMap(-0.3, 2.0)
    y = torch.randn(5, 1, 1)
    torch.testing.assert_close(om(y), 2.0 * y - 0.3)
    assert om.model_readout == "node_scalar" and om.model_out_dim == 1 and om.to("cpu") is om
    xs = [torch.randn(20, 3) * torch.tensor([2.0, 1.0, 5.0]) + torch.tensor([0.0, 3.0, -1.0]) for _ in range(4)]
    st = _mixed_stats(xs, [("scale", 1), ("v1", 2)])
    x = torch.cat(xs)
    assert float(st.mean[0]) == 0.0 and abs(float(st.std[0]) - float(x[:, 0].square().mean().sqrt())) < 1e-5
    torch.testing.assert_close(st.mean[1:], x[:, 1:].mean(0), atol=1e-5, rtol=1e-5)
    assert _mixed_stats(xs, "scale").kind == "scale" and _mixed_stats(xs, "v1").mean.shape == (3,)


needs_afm = pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "AFM.pt")), reason="datasets/v2/AFM.pt missing")


@needs_afm
def test_afpi_task_homogeneity_and_rotation():
    """AFPi: inputs, domain fields, exact degree-two scaling in physical units, rigid rotations."""
    from rhmp.tasks import load_task, resolve_root
    from rhmp.tasks.airfrans import AFP_NATIVE, afm_eval_source
    from rhmp.data import mesh_minibatch
    task = load_task("AFPi", None, native=True, device="cpu", max_samples=6, fine=False, whitney=True)
    assert task.in_dims == {0: 2, 1: 2, 2: 1} and task.even_dims == {1: 1, 2: 1} and task.readout == "node_scalar"
    K = task.K[0]
    assert int(K.meta["dirichlet"][0].sum()) > 10 and K.meta["cohomology"][1].shape == (K.n[1], 1)
    assert float(K.apply_d(1, K.meta["cohomology"][1]).abs().max()) < 1e-5
    legacy = load_task("AFPi", None, native=False, device="cpu", max_samples=6, fine=False, whitney=False)
    ref = load_task("AFP", None, native=False, device="cpu", max_samples=6, fine=False, whitney=False)
    assert legacy.output_map is None and torch.equal(legacy.inputs[0][0], ref.inputs[0][0])
    torch.testing.assert_close(task.target[0], ref.target[0])                  # the same target and normalisation
    cfg = RHMPConfig(in_dims=task.in_dims, even_dims=task.even_dims, C=8, layers=["project", "poly"], gate="homog",
                     amplitude_dims={1: 1}, amplitude_power=2, amplitude_energy=True, readout_energy=4,
                     project_solver="direct")
    torch.manual_seed(0)
    m = RHMP(cfg, task.geo_dims)
    Kb, xb, yb = mesh_minibatch(task.K, task.inputs, task.target, [0, 1], device="cpu")
    om = task.output_map
    p = task.y_stats.denormalize(om(m(xb, Kb)))                                # physical units
    xb2 = dict(xb)
    xb2[1] = torch.cat([2.0 * xb[1][..., :1], xb[1][..., 1:]], -1)             # inflow speed doubled
    p2 = task.y_stats.denormalize(om(m(xb2, Kb)))
    assert rel(p2, 4.0 * p) < 1e-5
    # rigid rotation of the whole configuration: same normalised inputs (to round-off), same prediction
    ids = task.meta["sample_ids"][:2]
    xs = {k: v.to("cpu") for k, v in task.x_stats.items()}
    a = afm_eval_source("AFPi", resolve_root(None), xs, task.y_stats.to("cpu"), ids=ids, whitney=True)
    b = afm_eval_source("AFPi", resolve_root(None), xs, task.y_stats.to("cpu"), ids=ids, whitney=True, rotate=137.0)
    assert rel(b["inputs"][0][1], a["inputs"][0][1]) < 1e-4 and torch.equal(a["inputs"][0][0], b["inputs"][0][0])
    ya = m({k: v[:, None] for k, v in a["inputs"][0].items()}, a["K"][0])
    yr = m({k: v[:, None] for k, v in b["inputs"][0].items()}, b["K"][0])
    assert rel(yr, ya) < 1e-3
    assert set(AFP_NATIVE) >= {"AFPi", "AFPi0", "AFPiU", "AFPiR", "AFPs", "AFPcp"}
    cp = load_task("AFPcp", None, native=False, device="cpu", max_samples=6, fine=False, whitney=False)
    assert cp.meta["cp_target"] and abs(float(cp.y_stats.std)) < 5.0           # pressure coefficient units


@pytest.mark.skipif(not os.path.exists(os.path.join(ROOT, "datasets", "T8_airfoil_pressure_persample.pkl")),
                    reason="datasets/T8_airfoil_pressure_persample.pkl missing")
def test_t8i_task():
    """T8 with the covariant inflow: T8 targets and normalisation, AFPi inputs, exact U^2 scaling; no domain fields."""
    from rhmp.data import mesh_minibatch
    from rhmp.tasks import list_tasks, load_task, task_defaults
    assert all(t in list_tasks() for t in ("T8i", "AFPi", "AFPi0", "AFPiU", "AFPiR", "AFPs", "AFPcp", "AFPcpxy"))
    assert task_defaults("AFPi")["batch"] == 8
    task = load_task("T8i", None, native=True, device="cpu", max_samples=6, whitney=True)
    ref = load_task("T8", None, native=True, device="cpu", max_samples=6, whitney=False)
    assert task.in_dims == {0: 2, 1: 2, 2: 1} and task.even_dims == {1: 1, 2: 1}
    torch.testing.assert_close(task.target[0], ref.target[0])
    torch.testing.assert_close(task.inputs[0][0][:, 0], ref.inputs[0][0][:, 0])       # sdf column
    assert "cohomology" not in task.K[0].meta
    cfg = RHMPConfig(in_dims=task.in_dims, even_dims=task.even_dims, C=8, layers=["project", "poly"], gate="homog",
                     amplitude_dims={1: 1}, amplitude_power=2, amplitude_energy=True, readout_energy=4,
                     project_solver="direct")
    torch.manual_seed(0)
    m = RHMP(cfg, task.geo_dims)
    Kb, xb, yb = mesh_minibatch(task.K, task.inputs, task.target, [0, 1], device="cpu")
    p = task.y_stats.denormalize(task.output_map(m(xb, Kb)))
    xb2 = dict(xb)
    xb2[1] = torch.cat([3.0 * xb[1][..., :1], xb[1][..., 1:]], -1)
    assert rel(task.y_stats.denormalize(task.output_map(m(xb2, Kb))), 9.0 * p) < 1e-5
    with pytest.raises(ValueError):
        load_task("T8i", None, native=False, device="cpu", max_samples=6)


@needs_afm
def test_potential_reference_control_task():
    """AFPpf: the no-learning Kutta potential flow as a node input explains most of the pressure on its own."""
    from rhmp.tasks import load_task
    task = load_task("AFPpf", None, native=False, device="cpu", max_samples=9, fine=False, whitney=False)
    assert task.in_dims == {0: 4}
    xs, ys = task.x_stats[0], task.y_stats
    P = torch.cat([xs.denormalize(x[0])[:, 3] for x in task.inputs]).double()
    T = torch.cat([ys.denormalize(y)[:, 0] for y in task.target]).double()
    assert float(1 - ((P - T) ** 2).sum() / ((T - T.mean()) ** 2).sum()) > 0.85
    ref = load_task("AFP", None, native=False, device="cpu", max_samples=9, fine=False, whitney=False)
    torch.testing.assert_close(xs.denormalize(task.inputs[0][0])[:, :3], ref.x_stats[0].denormalize(ref.inputs[0][0]),
                               rtol=1e-5, atol=1e-5)
