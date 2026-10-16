"""Graded anisotropic periodic grids and the NS vorticity task T1g.

* patch-area normalised oriented face -> node map: the mean on a uniform grid, exact densities of 2-forms on graded
  periodic grids;
* graded periodic quad / triangle complexes built from positions (``from_polygons`` / ``from_triangles`` with a
  period): minimal-image edge vectors, rectangle areas, rectangular stars, CCW faces;
* baselines on other grids (zero-shot): FNO on a grid of another size, the v1 edge protocol carried to extra test sets;
* T1g data (skipped without ``datasets/v2/T1_cns_vorticity_graded.pkl``): grid spacings and grading, exact samples of
  the simulation (the uniform zero-shot grid holds the T1 inputs), the rotated grid is the rotated training grid,
  the fine-grid target differs from the coarse T1 stencil by a few percent, RHMP gives the same output on the rotated
  grid as on the training grid.
"""
from __future__ import annotations

import os
import pickle

import numpy as np
import pytest
import torch

from conftest import ROOT
from rhmp.complex import CochainComplex
from test_model import F64

T1G_PATH = os.path.join(ROOT, "datasets", "v2", "T1_cns_vorticity_graded.pkl")
T1_PATH = os.path.join(ROOT, "datasets", "T1_cns_vorticity.pkl")
needs_t1g = pytest.mark.skipif(not os.path.exists(T1G_PATH), reason="graded T1 data set not generated")


def _graded(sx, sy, cell="quad", device="cpu"):
    """Periodic tensor grid of the unit torus with spacings ``sx``, ``sy`` (relative), numbered ``i * ny + j``."""
    x = np.concatenate([[0.0], np.cumsum(sx)[:-1]]) / np.sum(sx)
    y = np.concatenate([[0.0], np.cumsum(sy)[:-1]]) / np.sum(sy)
    nx, ny = len(x), len(y)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    P = np.stack([gx.reshape(-1), gy.reshape(-1)], 1)
    vid = lambda i, j: (i % nx) * ny + (j % ny)                 # noqa: E731
    Q = np.array([[vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)] for i in range(nx) for j in range(ny)])
    if cell == "quad":
        return CochainComplex.from_polygons(P, Q, device=device, period=(1.0, 1.0)), P, Q
    T = np.concatenate([Q[:, [0, 1, 2]], Q[:, [0, 2, 3]]])
    return CochainComplex.from_triangles(P, T, device=device, period=(1.0, 1.0)), P, T


SX, SY = [1, 2, 4, 2, 1, 1], [4, 2, 1, 1, 2]


def test_graded_periodic_complex_geometry():
    K, P, Q = _graded(SX, SY)
    hx, hy = np.asarray(SX, float) / sum(SX), np.asarray(SY, float) / sum(SY)
    assert K.check_d2() == 0.0 and K.n[0] - K.n[1] + K.n[2] == 0
    ev = K.edge_vectors().double().cpu().numpy()
    e = K.cells[1].cpu().numpy()
    raw = P[e[:, 1]] - P[e[:, 0]]
    assert np.allclose(ev, raw - np.round(raw), atol=1e-12)                          # minimal images
    lengths = np.sort(np.linalg.norm(ev, axis=1))
    expect = np.sort(np.concatenate([np.repeat(hx, len(SY)), np.tile(hy, len(SX))]))
    assert np.allclose(lengths, expect, atol=1e-7)
    i, j = np.divmod(np.arange(K.n[2]), len(SY))
    assert np.allclose(1.0 / K.star[2].double().cpu().numpy(), hx[i] * hy[j], rtol=1e-6)   # rectangle areas
    assert abs(float((1.0 / K.star[2].double()).sum()) - 1.0) < 1e-6
    # rectangular (circumcentric = barycentric) star1: dual length / primal length
    st1 = K.star[1].double().cpu().numpy()
    a, b = e[:, 0], e[:, 1]
    horiz = np.abs(ev[:, 1]) < 1e-12
    ia, ja = np.divmod(a, len(SY))
    dual = np.where(horiz, 0.5 * (hy[ja] + hy[(ja - 1) % len(SY)]), 0.5 * (hx[ia] + hx[(ia - 1) % len(SX)]))
    assert np.allclose(st1, dual / np.linalg.norm(ev, axis=1), rtol=1e-5)
    Xc = K.cell_positions(2).double().cpu()
    sa = 0.5 * (Xc[..., 0] * Xc.roll(-1, 1)[..., 1] - Xc.roll(-1, 1)[..., 0] * Xc[..., 1]).sum(1)
    assert bool((sa > 0).all())                                                      # CCW in the unwrapped frames
    Kt, _, _ = _graded(SX, SY, cell="tri")
    assert Kt.n == [K.n[0], 3 * K.n[0], 2 * K.n[0]] and Kt.check_d2() == 0.0


def test_patch_area_output_map():
    from rhmp.tasks.paper import oriented_face_to_node
    Ku = CochainComplex.from_grid((6, 5), (0.3, 0.2), cell="quad", periodic=True)
    x = torch.randn(Ku.n[2], 3, 1, generator=torch.Generator().manual_seed(0))
    mean, area = oriented_face_to_node(Ku, 1), oriented_face_to_node(Ku, 1, area_ref=0.3 * 0.2)
    torch.testing.assert_close(area(x), mean(x))                                    # uniform grid: the mean
    K, _, _ = _graded(SX, SY)
    A = (1.0 / K.star[2]).float()
    w0 = 2.5                                                                        # constant density, CCW faces
    y = oriented_face_to_node(K, 1, area_ref=0.01)((w0 * A)[:, None, None])
    torch.testing.assert_close(y, torch.full_like(y, w0 * 0.01))
    yb = oriented_face_to_node(K, 1)((w0 * A)[:, None, None])                       # the plain mean is not
    assert float((yb - yb.mean()).abs().max()) > 1e-3 * float(yb.abs().mean())


def test_fno_runs_on_another_grid_size():
    from rhmp.baselines import registry
    from rhmp.data import TaskData, Stats
    K8 = CochainComplex.from_grid((8, 8), 1 / 8, cell="tri", periodic=True)
    K16 = CochainComplex.from_grid((16, 16), 1 / 16, cell="tri", periodic=True)
    N = 6
    g = torch.Generator().manual_seed(1)
    td = TaskData(name="T1g", K=K8, inputs={0: torch.randn(N, 64, 4, generator=g)}, target=torch.randn(N, 64, 1, generator=g),
                  target_degree=0, target_kind="node_scalar", in_dims={0: 4}, even_dims={}, connection_dims={},
                  split=(torch.arange(4), torch.arange(4, 5), torch.arange(5, 6)),
                  x_stats={0: Stats(torch.zeros(4), torch.ones(4))}, y_stats=Stats(torch.zeros(1), torch.ones(1)),
                  spatial_dim=2, native=False)
    m, info = registry.build_model("fno", td, 20000)
    y8 = m({0: td.inputs[0].transpose(0, 1).contiguous()}, K8)
    y16 = m({0: torch.randn(256, 2, 4, generator=g)}, K16)
    assert y8.shape == (64, N, 1) and y16.shape == (256, 2, 1)


def test_edge_protocol_carries_to_extra_tests():
    from rhmp.baselines.registry import prepare_task
    from rhmp.data import TaskData, Stats
    K, _, T = _graded(SX, SY, cell="tri")
    K2 = CochainComplex.from_grid((5, 4), 0.25, cell="tri", periodic=True)
    g = torch.Generator().manual_seed(2)
    n, n2 = K.n[0], K2.n[0]
    td = TaskData(name="T1g", K=K, inputs={0: torch.randn(6, n, 4, generator=g)}, target=torch.randn(6, n, 1, generator=g),
                  target_degree=0, target_kind="node_scalar", in_dims={0: 4}, even_dims={}, connection_dims={},
                  split=(torch.arange(4), torch.arange(4, 5), torch.arange(5, 6)),
                  x_stats={0: Stats(torch.zeros(4), torch.ones(4))}, y_stats=Stats(torch.zeros(1), torch.ones(1)),
                  spatial_dim=2, native=False)
    yz = torch.randn(3, n2, 1, generator=g)
    td.extra_tests["zs"] = dict(K=K2, inputs={0: torch.randn(3, n2, 4, generator=g)}, target=yz, output_map=None)
    pt = prepare_task("mpsn", td)
    e = K2.cells[1].long()
    torch.testing.assert_close(pt.extra_tests["zs"]["target"], 0.5 * (yz[:, e[:, 0]] + yz[:, e[:, 1]]))
    assert pt.target.shape[1] == K.n[1] and td.extra_tests["zs"]["target"] is yz      # original task untouched


# ----------------------------------------------------------------------------------------------------------------
# T1g data
# ----------------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def t1g():
    with open(T1G_PATH, "rb") as f:
        return pickle.load(f)


@needs_t1g
def test_T1g_grids(t1g):
    G = t1g["grids"]
    for name in ("G1", "G2", "U32", "U64", "G1rot"):
        g = G[name]
        for s in (g["sx"], g["sy"]):
            s = np.asarray(s)
            assert s.sum() == 128 and set(s.tolist()) <= {1, 2, 4, 8}
            assert (np.maximum(s, np.roll(s, 1)) <= 2 * np.minimum(s, np.roll(s, 1))).all()     # grading <= 2
        assert np.allclose(np.diff(np.concatenate([g["kx"], [g["kx"][0] + 128]])), g["sx"])
    ar = np.asarray(G["G1"]["sx"])[:, None] / np.asarray(G["G1"]["sy"])[None, :]
    assert ar.min() == 1 / 8 and ar.max() == 8
    assert not np.array_equal(G["G1"]["sx"], G["G2"]["sx"])
    assert set(np.asarray(G["U32"]["sx"]).tolist()) == {4} and set(np.asarray(G["U64"]["sx"]).tolist()) == {2}
    assert t1g["train"]["X_data"].shape == (10000, 1024, 4) and t1g["zero_shot"]["U64"]["X_data"].shape[1] == 4096


@needs_t1g
@pytest.mark.skipif(not os.path.exists(T1_PATH), reason="T1 dataset not available")
def test_T1g_exact_samples_and_fine_target(t1g):
    with open(T1_PATH, "rb") as f:
        t1 = pickle.load(f)
    z = t1g["zero_shot"]["U32"]
    ids = np.asarray(z["sample_ids"])
    X1 = np.asarray(t1["X_data"], dtype=np.float32)[ids]
    assert np.array_equal(z["X_data"], X1)                       # the uniform grid holds the T1 inputs exactly
    Y1 = np.asarray(t1["Y_data"], dtype=np.float32)[ids]
    rel = np.linalg.norm(z["Y_data"] - Y1) / np.linalg.norm(Y1)
    assert 0.02 < rel < 0.12, rel                                # fine-grid target vs the coarse T1 stencil


@needs_t1g
def test_T1g_rotated_grid(t1g):
    G = t1g["grids"]
    g, r = G["G1"], G["G1rot"]
    perm = np.asarray(r["node_perm"])
    q = np.stack([np.mod(-g["points"][perm, 1], 1.0), g["points"][perm, 0]], 1)
    assert np.allclose(q, r["points"])
    Xg = t1g["train"]["X_data"][8500:8510]
    Yg = t1g["train"]["Y_data"][8500:8510]
    Xr, Yr = t1g["zero_shot"]["G1rot"]["X_data"][:10], t1g["zero_shot"]["G1rot"]["Y_data"][:10]
    assert np.array_equal(Yr, Yg[:, perm])
    assert np.array_equal(Xr[..., [0, 3]], Xg[:, perm][..., [0, 3]])
    assert np.array_equal(Xr[..., 1], -Xg[:, perm][..., 2]) and np.array_equal(Xr[..., 2], Xg[:, perm][..., 1])


@needs_t1g
def test_T1g_rhmp_rotation_invariant(t1g):
    """An (untrained) RHMP gives on the rotated grid exactly the outputs of the training grid: the edge cochains,
    vertex scalars, geometry descriptors and the oriented output map are all invariant under the rotation (the node
    baselines read (Vx, Vy) as scalar features and are not)."""
    from rhmp import RHMP, RHMPConfig
    from rhmp.tasks import load_task
    from rhmp.train import predict
    td = load_task("T1gml", None, native=True, device="cpu")
    src = td.extra_tests["zs_G1rot"]
    pe = td.meta["paper_eval"][:4]
    torch.manual_seed(0)
    cfg = RHMPConfig(in_dims=td.in_dims, even_dims=td.even_dims, readout="cochain:2", C=8, n_layers=2, gate="homog",
                     cochain_head="context", lift_vertex=True, readout_wedge=2, readout_depth=2, readout_spatial=True)
    model = RHMP(cfg, td.K.geo_dims).eval()
    with torch.no_grad():
        p, _ = predict(model, td, pe, 4, "cpu")
        src4 = dict(src, inputs={k: v[:4] for k, v in src["inputs"].items()}, target=src["target"][:4])
        q, _ = predict(model, td, torch.arange(4), 4, "cpu", source=src4)
    perm = torch.as_tensor(t1g["grids"]["G1rot"]["node_perm"])
    torch.testing.assert_close(q, p[:, perm], rtol=1e-4, atol=1e-4 * float(p.abs().max()))


def test_gem_radial_option():
    """GEM-CNN with edge-length dependent kernels (``model_opts={'radial': True}``): GEM-CNN at initialisation,
    the lengths enter once the radial gains move, and checkpoints rebuild with the option."""
    from rhmp.baselines import registry
    from rhmp.data import TaskData, Stats
    K, _, _ = _graded(SX, SY, cell="tri")
    n = K.n[0]
    g = torch.Generator().manual_seed(3)
    td = TaskData(name="T1g", K=K, inputs={0: torch.randn(5, n, 4, generator=g)}, target=torch.randn(5, n, 1, generator=g),
                  target_degree=0, target_kind="node_scalar", in_dims={0: 4}, even_dims={}, connection_dims={},
                  split=(torch.arange(3), torch.arange(3, 4), torch.arange(4, 5)),
                  x_stats={0: Stats(torch.zeros(4), torch.ones(4))}, y_stats=Stats(torch.zeros(1), torch.ones(1)),
                  spatial_dim=2, native=False)
    torch.manual_seed(0)
    m0, _ = registry.build_model("gem_cnn", td, None, hidden=8, cache_path=None)
    torch.manual_seed(0)
    m1, _ = registry.build_model("gem_cnn", td, None, hidden=8, cache_path=None, radial=True)
    assert m1.build.get("radial") is True and "radial" not in m0.build
    m1.load_state_dict({**m1.state_dict(), **m0.state_dict()})
    x = {0: td.inputs[0].transpose(0, 1).contiguous()}
    with torch.no_grad():
        torch.testing.assert_close(m1(x, K), m0(x, K))
        m1.core.radial[2].weight.normal_()
        assert rel_change(m1(x, K), m0(x, K)) > 1e-3
    m2 = registry.from_checkpoint({"model": "gem_cnn", "build": m1.build, "state_dict": m1.state_dict()}, td)
    with torch.no_grad():
        torch.testing.assert_close(m2(x, K), m1(x, K))


def rel_change(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


@needs_t1g
def test_T1gmla_absolute_descriptors():
    """T1gmla: scale-dependent descriptors in one length unit (the uniform 32 x 32 grid): zero on U32, log(1/2) / log(1/4)
    for the lengths / areas of U64, unchanged scale-free columns; the default descriptors are relative to each mesh."""
    from rhmp.tasks import load_task
    rel_td = load_task("T1gml", None, native=True, device="cpu")
    abs_td = load_task("T1gmla", None, native=True, device="cpu")
    Ku, Ka = abs_td.extra_tests["zs_U32"]["K"], abs_td.extra_tests["zs_U64"]["K"]
    for k in range(3):
        assert float(Ku.geo[k][:, :2].abs().max()) < 1e-5
        torch.testing.assert_close(abs_td.K.geo[k][:, 2:], rel_td.K.geo[k][:, 2:])
    torch.testing.assert_close(Ka.geo[1][:, 0], torch.full_like(Ka.geo[1][:, 0], float(np.log(0.5))))
    torch.testing.assert_close(Ka.geo[2][:, 0], torch.full_like(Ka.geo[2][:, 0], float(np.log(0.25))))
    torch.testing.assert_close(rel_td.extra_tests["zs_U64"]["K"].geo[1][:, 0], torch.zeros(Ka.n[1]))
    ell = abs_td.K.edge_vectors().double().norm(dim=1)
    torch.testing.assert_close(abs_td.K.geo[1][:, 0].double(), torch.log(ell * 32), rtol=1e-5, atol=1e-5)
