"""Periodic (flat torus) complexes and the periodic NS vorticity task T1p.

* topology of ``CochainComplex.from_grid(..., periodic=True)``: ``d d = 0`` exactly, Euler characteristic 0, Betti
  numbers (1, 2, 1), no boundary, every edge has two cofaces;
* intrinsic geometry: minimal-image edge vectors, uniform stars, descriptors equal to those of the interior cells of a
  non-periodic grid, CCW faces in their unwrapped frames, vertex order ``field[ix, iy].reshape(-1)``;
* RHMP symmetries on the torus: O(C) equivariance, vertex relabelling, face orientation flips, cyclic shifts of the
  grid (translations of the torus), batch independence;
* baselines on the periodic triangulated grid: outputs commute with cyclic shifts (edge vectors from the complex);
* T1p / T1pm data (skipped without the T1 data set): the target is the periodic central-difference curl of the next
  sample's velocity, wrap-around edge inputs use minimal-image edge vectors.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from conftest import ROOT, relabel_mesh
from rhmp.complex import CochainComplex
from test_equivariance import _cell_maps, _check_OC, _map_cells, _out_info, _readouts
from test_model import F64, build_model, make_inputs, rel, small_cfg

GRIDS = [("quad", None), ("tri", "main"), ("tri", "anti"), ("tri", "alternate")]
T1_PATH = os.path.join(ROOT, "datasets", "T1_cns_vorticity.pkl")


def _grid(cell, diagonal, shape=(6, 5), spacing=(0.3, 0.2), device="cpu", **kw) -> CochainComplex:
    return CochainComplex.from_grid(shape, spacing, cell=cell, diagonal=diagonal, periodic=True, device=device, **kw)


def _shift_perm(nx: int, ny: int, a: int, b: int) -> np.ndarray:
    """``sigma[v]``: vertex ``(i, j)`` -> ``((i + a) % nx, (j + b) % ny)`` (a translation of the torus)."""
    i, j = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    return (((i + a) % nx) * ny + (j + b) % ny).reshape(-1)


# ----------------------------------------------------------------------------------------------------------------
# complex
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cell,diagonal", GRIDS)
@pytest.mark.parametrize("shape", [(5, 4), (6, 6), (3, 3)])
def test_periodic_grid_topology(cell, diagonal, shape, device):
    from rhmp.dec import harmonic_basis
    K = _grid(cell, diagonal, shape, device=device)
    nx, ny = shape
    n = nx * ny
    assert K.n == ([n, 2 * n, n] if cell == "quad" else [n, 3 * n, 2 * n])
    assert K.check_d2() == 0.0
    assert K.n[0] - K.n[1] + K.n[2] == 0                                 # Euler characteristic of the torus
    assert not any(bool(b.any()) for b in K.boundary)
    cof = torch.zeros(K.n[1], dtype=torch.float32, device=device).index_add_(
        0, K.d[1].col_indices(), K.d_abs[1].values())
    assert bool((cof == 2).all())                                        # every edge has exactly two cofaces
    d0, d1 = K.d[0].to_dense().double().cpu(), K.d[1].to_dense().double().cpu()
    r0, r1 = int(torch.linalg.matrix_rank(d0)), int(torch.linalg.matrix_rank(d1))
    assert (K.n[0] - r0, K.n[1] - r0 - r1, K.n[2] - r1) == (1, 2, 1)     # Betti numbers (1, 2, 1)
    assert harmonic_basis(K.to(device, dtype=F64), 1).shape[1] == 2
    assert K.meta["period"] == (nx * 0.3, ny * 0.2) and K.meta["grid_shape"] == shape


@pytest.mark.parametrize("cell,diagonal", GRIDS)
def test_periodic_vertex_order_orientation_and_edge_vectors(cell, diagonal):
    nx, ny, (sx, sy) = 6, 5, (0.3, 0.2)
    K = _grid(cell, diagonal)
    v = torch.arange(nx * ny)
    torch.testing.assert_close(K.pos, torch.stack([(v // ny) * sx, (v % ny) * sy], 1).float())
    Q = K.cell_positions(2).double()
    a, b = Q[:, 1] - Q[:, 0], Q[:, 2] - Q[:, 0]
    assert bool(((a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) > 0).all())   # CCW in the unwrapped frames, wrap faces too
    ev = K.edge_vectors().double()
    raw = (K.pos[K.cells[1][:, 1]] - K.pos[K.cells[1][:, 0]]).double()
    assert bool((raw.abs().max(1).values > 0.5).any())                  # there are wrap-around edges ...
    assert bool((ev[:, 0].abs() <= sx + 1e-6).all() and (ev[:, 1].abs() <= sy + 1e-6).all())   # ... and minimal
    ok = torch.isclose(ev.abs(), torch.tensor([sx, 0.0], dtype=F64)).all(1) | \
        torch.isclose(ev.abs(), torch.tensor([0.0, sy], dtype=F64)).all(1) | \
        torch.isclose(ev.abs(), torch.tensor([sx, sy], dtype=F64)).all(1)
    assert bool(ok.all())


@pytest.mark.parametrize("cell,diagonal", [("quad", None), ("tri", "main"), ("tri", "anti")])
def test_periodic_geometry_is_uniform_and_matches_interior(cell, diagonal):
    """All cells of the regular torus are congruent: stars and descriptors are constant per degree (and per edge
    direction) and equal those of the interior cells of a large non-periodic grid with the same spacing."""
    h = 0.1
    K = _grid(cell, diagonal, (8, 8), h)
    star_default = "barycentric" if cell == "quad" else "cotan"
    Kn = CochainComplex.from_grid((12, 12), h, cell=cell, diagonal=diagonal, star=star_default)
    for k in range(3):
        assert torch.allclose(K.star[k], K.star[k][:1].expand_as(K.star[k]), rtol=1e-5) or k == 1
    torch.testing.assert_close(K.star[0], torch.full_like(K.star[0], h * h))
    if cell == "quad":
        torch.testing.assert_close(K.star[1], torch.ones_like(K.star[1]))
        torch.testing.assert_close(K.star[2], torch.full_like(K.star[2], 1 / (h * h)))
    # interior of the non-periodic grid: vertex (5, 5), its edges and faces
    vi = 5 * 12 + 5
    torch.testing.assert_close(K.geo[0], Kn.geo[0][vi].expand_as(K.geo[0]), atol=1e-5, rtol=1e-5)
    inner_e = (~Kn.boundary[1]) & ~(Kn.boundary[0][Kn.cells[1][:, 0]] | Kn.boundary[0][Kn.cells[1][:, 1]])
    inner_f = ~Kn.boundary[2]
    for k, inner in ((1, inner_e), (2, inner_f)):
        ref = {tuple(np.round(r, 5)) for r in Kn.geo[k][inner].numpy()}
        got = {tuple(np.round(r, 5)) for r in K.geo[k].numpy()}
        assert got == ref, (k, got, ref)
        ref_s = set(np.round(Kn.star[k][inner].numpy() / float(Kn.star[k][inner].max()), 5))
        got_s = set(np.round(K.star[k].numpy() / float(K.star[k].max()), 5))
        assert got_s == ref_s


def test_periodic_rectangular_stars():
    sx, sy = 0.3, 0.2
    K = _grid("quad", None, (6, 5), (sx, sy))
    ev = K.edge_vectors()
    xe = ev[:, 0].abs() > 0
    torch.testing.assert_close(K.star[1][xe], torch.full_like(K.star[1][xe], sy / sx))
    torch.testing.assert_close(K.star[1][~xe], torch.full_like(K.star[1][~xe], sx / sy))
    torch.testing.assert_close(K.star[2], torch.full_like(K.star[2], 1 / (sx * sy)))


@pytest.mark.parametrize("cell,diagonal", GRIDS)
def test_periodic_builders_agree(cell, diagonal):
    """from_polygons / from_triangles with ``period`` reproduce from_grid(periodic=True) (also with validation)."""
    K = _grid(cell, diagonal)
    build = CochainComplex.from_polygons if cell == "quad" else CochainComplex.from_triangles
    for validate in (True, False):
        K2 = build(K.pos.double(), K.cells[2], star=K.meta["star_type"], period=K.meta["period"], validate=validate)
        assert K2.n == K.n and torch.equal(K2.cells[1], K.cells[1]) and torch.equal(K2.cells[2], K.cells[2])
        for k in range(3):
            torch.testing.assert_close(K2.star[k], K.star[k])
            torch.testing.assert_close(K2.geo[k], K.geo[k])
        torch.testing.assert_close(K2.edge_vectors(), K.edge_vectors())


def test_periodic_batch_to_and_whitney():
    Kq, Kt = _grid("quad", None), _grid("tri", "main", (4, 5), (0.25, 0.2))
    Kb = CochainComplex.batch([Kq, Kt])
    torch.testing.assert_close(Kb.edge_vectors(), torch.cat([Kq.edge_vectors(), Kt.edge_vectors()]))
    torch.testing.assert_close(Kb.cell_positions(2)[:Kq.n[2]], Kq.cell_positions(2)[:, :Kb.cell_positions(2).shape[1]])
    K64 = Kq.to("cpu", dtype=F64)
    torch.testing.assert_close(K64.edge_vectors().double(), Kq.edge_vectors().double())
    # Whitney blocks of the periodic triangles are those of congruent right triangles (P1 stiffness identity)
    from rhmp.dec import assemble_whitney_metric
    H1 = assemble_whitney_metric(Kt.to("cpu", dtype=F64), 1).to_dense()
    d0 = Kt.d[0].to_dense().double()
    S = d0.T @ H1 @ d0
    assert torch.allclose(S.sum(1), torch.zeros(Kt.n[0], dtype=F64), atol=1e-10)    # constants in the kernel
    assert torch.allclose(S.diagonal(), S.diagonal()[:1].expand(Kt.n[0]), rtol=1e-8)  # all vertices equivalent


def test_periodic_errors():
    with pytest.raises(ValueError, match=">= 3"):
        CochainComplex.from_grid((2, 4), periodic=True)
    with pytest.raises(ValueError, match="not periodic"):
        CochainComplex.from_grid((4, 4), cell="tri", diagonal="v1", periodic=True)
    assert CochainComplex.from_grid((4, 4), cell="tri", periodic=True).meta["diagonal"] == "main"


# ----------------------------------------------------------------------------------------------------------------
# RHMP symmetries on the torus
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("cell,diagonal", [("quad", None), ("tri", "main")])
def test_periodic_OC_equivariance(cell, diagonal, device):
    K = _grid(cell, diagonal, device=device).to(device, dtype=F64)
    _check_OC(K, small_cfg(), F64, 1e-10)


def _rebuild(K, pos, cells):
    build = CochainComplex.from_polygons if K.meta["cell_type"] == "quad" else CochainComplex.from_triangles
    return build(pos, cells, star=K.meta["star_type"], period=K.meta["period"], device=K.pos.device)


def _check_relabel(K, K2, perm, inv, readouts):
    maps = _cell_maps(K, K2, inv)
    permt = torch.as_tensor(perm, device=K.pos.device)
    for readout in readouts:
        cfg = small_cfg(readout=readout, out_dim=2)
        m = build_model(cfg, K)
        inp = make_inputs(cfg, K)
        inp2 = {0: inp[0][permt]}
        for k, v in inp.items():
            if k > 0:
                inp2[k] = _map_cells(v, *maps[k], n_odd=cfg.in_dims[k] - cfg.even_dims.get(k, 0))
        y, y2 = m(inp, K), m(inp2, K2)
        odeg, odd = _out_info(readout, K.dim)
        expect = y[permt] if odeg == 0 else _map_cells(y, *maps[odeg], n_odd=(y.shape[-1] if odd else 0))
        assert rel(y2, expect) < 1e-5, (readout, rel(y2, expect))
    return maps


@pytest.mark.parametrize("cell,diagonal", [("quad", None), ("tri", "main")])
def test_periodic_vertex_relabelling(cell, diagonal, device):
    """Random relabelling of the torus vertices (complex rebuilt with the period): permuted outputs."""
    K = _grid(cell, diagonal, device=device)
    pos, cells = K.pos.double().cpu().numpy(), K.cells[2].cpu().numpy()
    pos2, cells2, perm = relabel_mesh(pos, cells, seed=3)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    maps = _check_relabel(K, _rebuild(K, pos2, cells2), perm, inv, _readouts(2))
    assert any(bool((s < 0).any()) for _, s in maps.values())


@pytest.mark.parametrize("cell,diagonal", [("quad", None), ("tri", "main")])
def test_periodic_cyclic_shift_equivariance(cell, diagonal, device):
    """A cyclic shift of the grid is an automorphism of the torus: shifted inputs give shifted outputs."""
    nx, ny = 6, 5
    K = _grid(cell, diagonal, (nx, ny), device=device)
    sigma = _shift_perm(nx, ny, 2, 3)                                     # new id of old vertex v
    perm = np.argsort(sigma)                                              # old vertex of new vertex i
    _check_relabel(K, K, perm, sigma, ["node_scalar", "cochain:2", "cochain:1", "even:2", "grad", "div"])


@pytest.mark.parametrize("cell", ["quad", "tri"])
def test_periodic_face_orientation_flip(cell, device):
    K = _grid(cell, "main", device=device)
    faces = K.cells[2].cpu().numpy()
    flip = np.random.default_rng(7).random(len(faces)) < 0.4
    faces2 = faces.copy()
    faces2[flip] = faces[flip][:, ::-1]
    K2 = _rebuild(K, K.pos.double().cpu().numpy(), faces2)
    sign = torch.as_tensor(np.where(flip, -1.0, 1.0), dtype=torch.float32, device=device)
    idx = torch.arange(len(faces), device=device)
    for readout in _readouts(2):
        cfg = small_cfg(readout=readout, out_dim=2)
        m = build_model(cfg, K)
        inp = make_inputs(cfg, K)
        inp2 = {**inp, 2: _map_cells(inp[2], idx, sign, n_odd=cfg.in_dims[2] - cfg.even_dims.get(2, 0))}
        y, y2 = m(inp, K), m(inp2, K2)
        odeg, odd = _out_info(readout, 2)
        expect = _map_cells(y, idx, sign, n_odd=y.shape[-1]) if (odeg == 2 and odd) else y
        assert rel(y2, expect) < 1e-5, (readout, rel(y2, expect))


def test_periodic_batch_independence(device):
    K = _grid("quad", None, device=device)
    cfg = small_cfg(readout="cochain:2", n_layers=3)
    m = build_model(cfg, K)
    inp, other = make_inputs(cfg, K, B=16, seed=0), make_inputs(cfg, K, B=16, seed=1)
    alone = m({k: v[:, :1].contiguous() for k, v in inp.items()}, K)
    mixed = m({k: torch.cat([inp[k][:, :1], other[k][:, 1:]], 1) for k in inp}, K)
    assert (alone - mixed[:, :1]).abs().max() < 1e-6


# ----------------------------------------------------------------------------------------------------------------
# baselines on the periodic triangulated grid
# ----------------------------------------------------------------------------------------------------------------
SHIFT_MODELS = ["gem_cnn", "gauge_cnn", "egnn", "schnet", "gcn", "gat", "mpsn", "fno"]


@pytest.mark.parametrize("name", SHIFT_MODELS)
def test_baselines_commute_with_cyclic_shifts(name, device):
    """Node-input baselines on the periodic triangulated grid (legacy T1p layout: 4 node columns -> 1 node target):
    the output of a shifted input is the shifted output.  Needs the minimal-image edge vectors of the complex."""
    from rhmp.baselines import registry
    from rhmp.data import Stats, TaskData, sequential_split
    nx = ny = 8
    K = CochainComplex.from_grid((nx, ny), 1 / nx, cell="tri", diagonal="main", periodic=True, device=device)
    g = torch.Generator().manual_seed(0)
    N = 6
    td = TaskData(name="T1p", K=K, inputs={0: torch.randn(N, K.n[0], 4, generator=g).to(device)},
                  target=torch.randn(N, K.n[0], 1, generator=g).to(device), target_degree=0,
                  target_kind="node_scalar", in_dims={0: 4}, even_dims={}, connection_dims={},
                  split=sequential_split(N), x_stats={0: Stats(torch.zeros(4), torch.ones(4))},
                  y_stats=Stats(torch.zeros(1), torch.ones(1)), spatial_dim=2, out_dim=1, native=False,
                  meta={"v1_C": 8, "layers": 2, "batch_size": 4})
    opts = {"hidden": 8, "modes": 3} if name == "fno" else {"hidden": 8}
    if name == "mpsn":
        opts["head"] = "node"
    torch.manual_seed(0)
    model, _ = registry.build_model(name, td, None, cache_path=None, **opts)
    with torch.no_grad():                                                   # non-trivial zero-initialised weights
        for p in model.parameters():
            if bool((p == 0).all()):
                p.add_(0.1 * torch.randn_like(p))
    model.eval()
    sigma = torch.as_tensor(_shift_perm(nx, ny, 3, 5), device=device)
    perm = torch.argsort(sigma)                                             # new vertex i = old vertex perm[i]
    x = td.inputs[0][:3].transpose(0, 1).contiguous()
    with torch.no_grad():
        y = model({0: x}, K)
        ys = model({0: x[perm].contiguous()}, K)
    assert rel(ys, y[perm]) < 1e-5, (name, rel(ys, y[perm]))


def test_adapter_uses_complex_edge_vectors():
    from rhmp.baselines.adapters import V1ComplexAdapter
    K = CochainComplex.from_grid((8, 8), 1 / 8, cell="tri", diagonal="main", periodic=True)
    A = V1ComplexAdapter(K)
    assert A.periodic
    torch.testing.assert_close(A.edge_vectors(), K.edge_vectors())
    ln = A.edge_lengths()
    assert torch.allclose(ln.min(), torch.tensor(1 / 8)) and torch.allclose(ln.max(), torch.tensor(2 ** 0.5 / 8))
    assert bool((A.face_orientation() > 0).all())                           # all CCW in the unwrapped frames
    _, dist = A.directed_geometry(True)
    assert float(dist.max() / dist.min()) < 1.5


# ----------------------------------------------------------------------------------------------------------------
# T1p data
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(T1_PATH), reason="T1 dataset not available")
def test_T1p_task_data():
    from rhmp.tasks import load_task
    td = load_task("T1p", None, native=True, device="cpu")
    K = td.K
    assert K.n == [1024, 2048, 1024] and K.meta["period"] == (1.0, 1.0) and td.readout == "cochain:2"
    assert td.in_dims == {0: 2, 1: 2} and td.even_dims == {1: 1}
    # raw edge inputs: midpoint one-form with minimal-image edge vectors (wrap edges have length 1/32)
    import pickle
    with open(T1_PATH, "rb") as f:
        d = pickle.load(f)
    X = torch.as_tensor(np.asarray(d["X_data"][:40], dtype=np.float32))
    Y = np.asarray(d["Y_data"][:40], dtype=np.float64)[..., 0]
    raw = td.x_stats[1].denormalize(td.inputs[1][:40])
    e, ev = K.cells[1].long(), K.edge_vectors()
    assert torch.allclose(ev.norm(dim=1), torch.full((K.n[1],), 1 / 32), atol=1e-7)
    V = X[..., 1:3]
    th = (0.5 * (V[:, e[:, 0]] + V[:, e[:, 1]]) * ev).sum(-1)
    torch.testing.assert_close(raw[..., 0], th, atol=1e-6, rtol=1e-5)
    # the target is the periodic central-difference curl (dx = 1/32) of the next sample's velocity (20 steps per
    # trajectory)
    g = 32
    Vx, Vy = X[:, :, 1].reshape(-1, g, g).double().numpy(), X[:, :, 2].reshape(-1, g, g).double().numpy()
    curl = ((np.roll(Vy, -1, 1) - np.roll(Vy, 1, 1)) - (np.roll(Vx, -1, 2) - np.roll(Vx, 1, 2))) * (g / 2)
    curl = curl.reshape(len(X), -1)
    for s in range(39):
        if s % 20 != 19:
            assert np.abs(curl[s + 1] - Y[s]).max() <= 1e-4 * np.abs(Y[s]).max()
    tl = load_task("T1p", None, native=False, device="cpu")
    assert tl.K.n == [1024, 3072, 2048] and tl.K.meta["cell_type"] == "triangle" and tl.in_dims == {0: 4}
    torch.testing.assert_close(tl.target, td.target)


@pytest.mark.skipif(not os.path.exists(T1_PATH), reason="T1 dataset not available")
def test_T1pm_cell_fields():
    from rhmp.tasks import load_task
    td = load_task("T1pm", None, native=True, device="cpu")
    assert td.in_dims == {0: 2, 1: 4, 2: 2} and td.even_dims == {1: 3, 2: 2}
    x0 = td.x_stats[0].denormalize(td.inputs[0][:3])                       # (rho, p)
    x2 = td.x_stats[2].denormalize(td.inputs[2][:3])
    F = td.K.cells[2].long()
    torch.testing.assert_close(x2, torch.log(x0)[:, F].mean(2), atol=1e-4, rtol=1e-4)
    tl = load_task("T1pml", None, native=True, device="cpu")
    assert tl.in_dims == {0: 4, 1: 4, 2: 2} and tl.even_dims == {1: 3, 2: 2}
    v = tl.x_stats[0].denormalize(tl.inputs[0][:3])
    torch.testing.assert_close(v[..., 2:], torch.log(v[..., :2]), atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(tl.target, td.target)


BLOCKAVG_PATH = os.path.join(ROOT, "datasets", "v2", "T1_cns_vorticity_blockavg.pkl")


@pytest.mark.skipif(not os.path.exists(BLOCKAVG_PATH), reason="block-averaged T1 data set not generated")
def test_T1pf_block_averaged_data():
    """T1pf: same trajectories and sample order as T1, fields averaged over 4 x 4 blocks, target = the same periodic
    stencil on the filtered velocity of the next sample."""
    import pickle
    with open(BLOCKAVG_PATH, "rb") as f:
        d = pickle.load(f)
    X = np.asarray(d["X_data"][:40], dtype=np.float64)
    Y = np.asarray(d["Y_data"][:40], dtype=np.float64)[..., 0]
    g = 32
    Vx, Vy = X[:, :, 1].reshape(-1, g, g), X[:, :, 2].reshape(-1, g, g)
    curl = ((np.roll(Vy, -1, 1) - np.roll(Vy, 1, 1)) - (np.roll(Vx, -1, 2) - np.roll(Vx, 1, 2))) * (g / 2)
    curl = curl.reshape(40, -1)
    for s in range(39):
        if s % 20 != 19:
            assert np.abs(curl[s + 1] - Y[s]).max() <= 1e-4 * np.abs(Y[s]).max()
    from rhmp.tasks import load_task
    td = load_task("T1pf", None, native=False, device="cpu")
    assert td.K.n == [1024, 3072, 2048] and td.K.meta["period"] == (1.0, 1.0)
