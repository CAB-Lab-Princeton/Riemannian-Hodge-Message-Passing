"""Tests of the AirfRANS task family (``rhmp/tasks/airfrans.py``, ``datasets/generators/gen_airfrans.py``).

Without data (synthetic meshes):
* the fixed output maps: the star-average map is Stokes-exact (a linear velocity field gives its constant vorticity at
  every vertex, independent of face orientations and of batching); the flux -> velocity map reconstructs uniform
  flows from ``F = d0 psi`` and sets the wall velocity to zero; ``F = d0 psi`` is exactly closed;
* the trainer hook ``model_output`` passes the complex to maps with ``takes_complex``;
* the registry resolves the AirfRANS task names;
* the mesh builder of the generator produces a valid conforming annulus mesh around a NACA-type airfoil (synthetic
  "CFD" nodes) and the edge quadrature is exact for polynomial fields.
With the generated data sets (skipped when ``datasets/v2/T8res_n2000.pt`` / ``AFM.pt`` are absent): ``T8_n2000``
reproduces the T8 test samples, the AFV target is reproduced by the exact Stokes model, and the accurate edge fluxes
are closed per triangle.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GEN = os.path.join(ROOT, "datasets", "generators")
for p in (ROOT, GEN):
    if p not in sys.path:
        sys.path.insert(0, p)

from rhmp.complex import CochainComplex  # noqa: E402
from rhmp.tasks.airfrans import (FluxVelocityMap, StarAverageMap, flux_divergence, signed_areas,  # noqa: E402
                                 trapezoid_flux, wall_vertices)

DATA = os.path.join(ROOT, "datasets", "v2")


def _delaunay(n=150, seed=0, box=((0.0, 1.0), (0.0, 1.0)), flip_half=False):
    from scipy.spatial import Delaunay
    rng = np.random.default_rng(seed)
    (x0, x1), (y0, y1) = box
    pts = np.concatenate([np.stack([rng.uniform(x0, x1, n), rng.uniform(y0, y1, n)], 1),
                          [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]])
    F = Delaunay(pts).simplices.astype(np.int64)
    P = pts[F]
    sa = (P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) - (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])
    F[sa < 0] = F[sa < 0][:, [0, 2, 1]]
    if flip_half:
        F[::2] = F[::2][:, [0, 2, 1]]
    return CochainComplex.from_triangles(torch.as_tensor(pts), torch.as_tensor(F), star="cotan")


def _theta_linear(K, A):
    """Line integrals of the linear field u(x) = A x along the edges (exact by the trapezoid rule)."""
    P = K.pos.double()
    u = P @ torch.as_tensor(A, dtype=torch.float64).T
    e = K.cells[1].long()
    return (0.5 * (u[e[:, 0]] + u[e[:, 1]]) * (P[e[:, 1]] - P[e[:, 0]])).sum(1)


@pytest.mark.parametrize("flip", [False, True])
def test_star_average_map_is_stokes_exact(flip):
    K = _delaunay(flip_half=flip)
    A = np.array([[0.3, -1.7], [0.9, -0.2]])          # vorticity dv/dx - du/dy = 0.9 + 1.7 = 2.6
    th = _theta_linear(K, A)
    c = K.apply_d(1, th[:, None].float()).double()       # circulation in the faces' own orientation
    y = StarAverageMap(1, 1.0)(c[:, None, :], K)[:, 0, 0]
    assert torch.allclose(y, torch.full_like(y, 2.6), rtol=1e-4, atol=1e-4)
    # a fixed unit factor scales the output and nothing else
    y2 = StarAverageMap(1, 0.5)(c[:, None, :], K)[:, 0, 0]
    assert torch.allclose(y2, 0.5 * y, rtol=1e-12)


def test_star_average_map_batched_equals_per_mesh():
    Ks = [_delaunay(n=80, seed=s) for s in (1, 2)]
    Kb = CochainComplex.batch(Ks)
    gen = torch.Generator().manual_seed(0)
    cs = [torch.randn(K.n[2], 1, 1, generator=gen, dtype=torch.float64) for K in Ks]
    m = StarAverageMap(1, 0.37)
    yb = m(torch.cat(cs), Kb)
    ys = torch.cat([m(c, K) for c, K in zip(cs, Ks)])
    assert torch.allclose(yb, ys, rtol=1e-12, atol=1e-12)


def test_flux_velocity_map_uniform_flow_and_wall():
    # a mesh of [0, 1] x [-0.3, 0.3]: its boundary lies in the airfoil box, so every boundary vertex is a wall vertex
    K = _delaunay(n=200, seed=3, box=((0.0, 1.0), (-0.3, 0.3)))
    wall = wall_vertices(K)
    assert bool((wall == K.boundary[0].bool()).all())
    U = torch.tensor([2.0, -0.7], dtype=torch.float64)
    psi = K.pos.double() @ torch.tensor([-U[1], U[0]], dtype=torch.float64)   # u = (psi_y, -psi_x) = U
    e = K.cells[1].long()
    F = (psi[e[:, 1]] - psi[e[:, 0]])[:, None, None]
    u = FluxVelocityMap("grad", 1.0)(F, K)[:, 0]
    assert torch.allclose(u[~wall], U.expand(int((~wall).sum()), 2).to(u.dtype), atol=1e-4)
    assert float(u[wall].abs().max()) == 0.0
    # the same fluxes are the trapezoid fluxes of the uniform field (exact) and are closed per triangle
    Ft = trapezoid_flux(K.pos.double(), e, U.expand(K.n[0], 2))
    assert torch.allclose(Ft, F[:, 0, 0], atol=1e-6)
    assert float(flux_divergence(K, F[:, 0, 0].float()).abs().max()) < 1e-5


def test_grad_flux_closed_and_trapezoid_not():
    K = _delaunay(n=200, seed=4)
    P = K.pos.double()
    psi = torch.sin(3 * P[:, 0]) * torch.cos(2 * P[:, 1])
    e = K.cells[1].long()
    F = psi[e[:, 1]] - psi[e[:, 0]]
    assert float(flux_divergence(K, F.float()).abs().max()) < 1e-5            # d1 d0 = 0
    # trapezoid fluxes of the (divergence-free) velocity of psi are not exactly closed on a coarse mesh
    u = torch.stack([-2 * torch.sin(3 * P[:, 0]) * torch.sin(2 * P[:, 1]),
                     -3 * torch.cos(3 * P[:, 0]) * torch.cos(2 * P[:, 1])], 1)
    Ft = trapezoid_flux(P, e, u)
    assert float(flux_divergence(K, Ft.float()).abs().max()) > 1e-4


def test_model_output_passes_complex():
    from rhmp.data import TaskData
    from rhmp.train import model_output

    class Map:
        takes_complex = True
        model_readout = "cochain:2"
        model_out_dim = 1
        description = "test"

        def __call__(self, y, K=None):
            assert K is not None
            return y[: K.n[0]] * 0 + K.n[0]

    K = _delaunay(n=30, seed=5)
    td = TaskData(name="x", K=K, inputs={}, target=torch.zeros(1, K.n[0], 1), target_degree=0,
                  target_kind="node_scalar", in_dims={}, even_dims={}, connection_dims={},
                  split=(torch.arange(1), torch.arange(0), torch.arange(0)), x_stats={}, y_stats=None,
                  spatial_dim=2, output_map=Map())
    out = model_output(td, lambda xb, KK: torch.ones(KK.n[2], 1, 1), {}, K)
    assert float(out[0, 0, 0]) == K.n[0]


def test_registry_resolves_airfrans_names():
    from rhmp.tasks import list_tasks, task_defaults
    names = list_tasks()
    for t in ("T8_n500", "T8_n2000", "T8_n8000", "T8fr", "T8xy", "AFV", "AFVp", "AFU", "AFUv", "AFD", "AFP"):
        assert t in names
        assert task_defaults(t)["batch"] == 8


# ----------------------------------------------------------------------------------------------------------------
# generator: mesh builder and quadrature on synthetic data
# ----------------------------------------------------------------------------------------------------------------
def _naca0012(n=300):
    x = 0.5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    t = 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1036 * x ** 4)
    upper = np.stack([x[::-1], t[::-1]], 1)            # TE -> LE on the upper side
    lower = np.stack([x[1:-1], -t[1:-1]], 1)           # LE -> TE on the lower side
    P = np.concatenate([upper, lower])
    return P


def test_build_mesh_synthetic_airfoil():
    pytest.importorskip("matplotlib")
    import gen_airfrans as G
    from matplotlib.path import Path
    from scipy.spatial import cKDTree
    P = _naca0012()
    if G.signed_area_poly(P) < 0:
        P = P[::-1].copy()
    rng = np.random.default_rng(0)
    pts = np.stack([rng.uniform(-0.8, 1.8, 60000), rng.uniform(-0.8, 0.8, 60000)], 1)
    pts = pts[~Path(P).contains_points(pts)]
    seg = np.roll(P, -1, 0) - P
    up = np.concatenate([P[i] + np.outer(np.linspace(0, 1, 10, endpoint=False), seg[i]) for i in range(len(P))])
    sdf = cKDTree(up).query(pts)[0]
    pos = np.concatenate([P, pts])
    cfd = dict(pos=pos, sdf=np.concatenate([np.zeros(len(P)), sdf]),
               area=np.full(len(pos), 2.6 * 1.6 / len(pos)),
               surface=np.concatenate([np.ones(len(P), bool), np.zeros(len(pts), bool)]))
    m = G.build_mesh(cfd, P, np.arange(len(P)), np.random.default_rng(1), 1.0,
                     dict(h_wall=0.03, growth=0.35, density=1.0))
    st = m["stats"]
    assert st["euler"] == 0 and st["nonmanifold"] == 0 and st["unused"] == 0
    assert st["n_boundary_edges"] == st["n_wall"] + st["n_ring"]
    X, F = m["X"], m["faces"]
    assert (G.tri_signed_area(X, F) > 0).all()
    assert not Path(P).contains_points(X[F].mean(1)).any()
    K = CochainComplex.from_triangles(torch.as_tensor(X), torch.as_tensor(F), star="cotan")
    assert K.n[2] == len(F) and K.check_d2() == 0.0
    # the geometric wall test agrees with the builder's vertex kinds
    assert torch.equal(wall_vertices(K).cpu(), torch.as_tensor(m["kind"] == G.KIND_WALL))


def test_edge_quadrature_exact_for_polynomials():
    import gen_airfrans as G

    class FieldProbe:
        def __call__(self, X, names=("U",)):
            x, y = X[:, 0], X[:, 1]
            return {"U": np.stack([x ** 3 - y, x * y ** 2 + 2.0], 1)}

    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (12, 2))
    edges = np.array([[i, j] for i in range(12) for j in range(i + 1, 12)][:30])
    kind = np.zeros(12, dtype=np.int64)
    kind[:3] = G.KIND_WALL                                    # graded rule on edges with one wall end
    res = G.edge_integrals(X, edges, kind, FieldProbe(), dict(G.MESH))
    # reference: high-order Gauss-Legendre
    xg, wg = np.polynomial.legendre.leggauss(64)
    t, w = 0.5 * (xg + 1), 0.5 * wg
    for k, (a, b) in enumerate(edges):
        if kind[a] == G.KIND_WALL and kind[b] == G.KIND_WALL:
            assert res["theta"][k] == 0.0 and res["flux"][k] == 0.0
            continue
        Q = X[a][None] + t[:, None] * (X[b] - X[a])[None]
        U = FieldProbe()(Q)["U"]
        d = X[b] - X[a]
        th = float((U @ d * w).sum())
        fl = float(((U[:, 0] * d[1] - U[:, 1] * d[0]) * w).sum())
        assert abs(res["theta"][k] - th) < 1e-9 * max(1.0, abs(th))
        assert abs(res["flux"][k] - fl) < 1e-9 * max(1.0, abs(fl))


# ----------------------------------------------------------------------------------------------------------------
# with the generated data sets
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not (os.path.exists(os.path.join(DATA, "T8res_n2000.pt")) and
                         os.path.exists(os.path.join(ROOT, "datasets", "T8_airfoil_pressure_persample.pkl"))),
                    reason="T8 data / T8res_n2000.pt not generated")
def test_t8_n2000_reproduces_t8_test_samples():
    from rhmp.tasks import load_task
    v = load_task("T8_n2000", ROOT, device="cpu", max_samples=4, whitney=False)
    t8 = load_task("T8", ROOT, device="cpu", max_samples=12, whitney=False)
    for j in range(4):
        i = int(t8.split[2][j])
        assert torch.equal(v.K[j].pos, t8.K[i].pos)
        assert torch.allclose(v.y_stats.denormalize(v.target[j]), t8.y_stats.denormalize(t8.target[i]), atol=1e-3)
        assert torch.allclose(v.x_stats[0].denormalize(v.inputs[j][0]), t8.x_stats[0].denormalize(t8.inputs[i][0]),
                              atol=1e-4)


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "AFM.pt")), reason="AFM.pt not generated")
def test_afv_target_is_the_exact_stokes_map():
    from rhmp.data import mesh_minibatch
    from rhmp.tasks import load_task
    td = load_task("AFV", ROOT, device="cpu", max_samples=6, whitney=False, fine=False)
    gain = float(td.x_stats[1].std / (td.y_stats.std * td.output_map.scale))
    K, xb, yb = mesh_minibatch(td.K, td.inputs, td.target, list(range(len(td.K))), device="cpu")
    y = td.output_map(K.apply_d(1, xb[1].contiguous()) * gain, K)
    assert float((y - yb).abs().max() / yb.abs().max()) < 1e-5


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "AFM.pt")), reason="AFM.pt not generated")
def test_afm_accurate_fluxes_are_closed():
    from rhmp.tasks import load_task
    td = load_task("AFU", ROOT, device="cpu", max_samples=6, whitney=False, fine=False)
    for K, y in zip(td.K, td.target):
        F = td.y_stats.denormalize(y)[:, 0].double()
        d = flux_divergence(K, F.float()).double()
        ref = torch.sparse.mm(K.d_abs[1].float(), F.abs()[:, None].float()).double()[:, 0]
        rel = (d.abs() / ref.clamp_min(1e-30))
        assert float(rel.median()) < 1e-3 and float(torch.quantile(rel, 0.99)) < 5e-3
        assert float(signed_areas(K).min()) > 0
