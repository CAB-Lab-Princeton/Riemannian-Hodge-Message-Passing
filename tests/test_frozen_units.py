"""``RHMPConfig.frozen_units``: the per-mesh geometric normalisers of the training mesh, stored in the model.

* on the training mesh the output is identical to the default (bitwise), for every mesh type;
* the stored quantities follow their definition on another mesh: stars divided by the training geometric means,
  relative descriptors re-expressed against the training medians, block bounds times
  ``beta_unit(train) / beta_unit(mesh)``;
* a mesh scaled by a global factor now changes the output (the default model is invariant to the length unit);
* rotations / reflections, vertex relabelling, face orientation flips and block-diagonal batches keep their
  invariances; checkpoints carry the references; unsupported layer types are refused;
* ``frozen_descriptors=False`` freezes only the operator normalisation (stars, block bounds); ``'auto'`` freezes a
  descriptor column where the mesh's median lies in the training range of that column.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from conftest import _MESHES, make_complex, random_orthogonal, relabel_mesh
from rhmp import RHMP
from rhmp.complex import CochainComplex, batch_complexes
from rhmp.layers import make_context, unit_reference
from test_equivariance import _build, _cell_maps, _map_cells, _out_info
from test_model import F64, build_model, make_inputs, rel, small_cfg

FINAL = dict(gate="homog", lift_context=4, lift_vertex=True, cochain_head="context", readout_skip=True,
             readout_depth=2, readout_spatial=True, readout_wedge=2)


def _graded_quads(sx, sy, scale=1.0, device="cpu"):
    x = np.concatenate([[0.0], np.cumsum(sx)[:-1]]) / np.sum(sx)
    y = np.concatenate([[0.0], np.cumsum(sy)[:-1]]) / np.sum(sy)
    nx, ny = len(x), len(y)
    gx, gy = np.meshgrid(x, y, indexing="ij")
    P = np.stack([gx.reshape(-1), gy.reshape(-1)], 1) * scale
    vid = lambda i, j: (i % nx) * ny + (j % ny)                 # noqa: E731
    Q = np.array([[vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)] for i in range(nx) for j in range(ny)])
    return CochainComplex.from_polygons(P, Q, device=device, period=(scale, scale))


def _meshes(device):
    return [make_complex("delaunay", device), make_complex("mixed", device), make_complex("tets", device, n=30),
            CochainComplex.from_grid((6, 5), (0.3, 0.2), cell="quad", periodic=True, device=device),
            _graded_quads([1, 2, 4, 2, 1, 1], [4, 2, 1, 1, 2], device=device)]


def _cfg(K, **kw):
    opts = dict(FINAL) if K.dim == 2 else {k: v for k, v in FINAL.items() if k != "readout_wedge"}
    opts.update(kw)
    return small_cfg(dim=K.dim, readout=f"cochain:{K.dim}", **opts)


@pytest.mark.parametrize("dtype", [F64, torch.float32])
def test_identical_on_the_training_mesh(dtype, device):
    for K in _meshes(device):
        K = K.to(device, dtype=dtype)
        for cfg in (_cfg(K), small_cfg(dim=K.dim, readout="node_scalar")):
            m = build_model(cfg, K, dtype=dtype)
            inp = make_inputs(cfg, K, dtype=dtype)
            with torch.no_grad():
                y0 = m(inp, K)
                m.freeze_units(K)
                y1 = m(inp, K)
            if str(device) == "cpu":                                     # bitwise (GPU scatter-adds are not ordered)
                assert torch.equal(y0, y1), (K.meta.get("cell_type"), float((y0 - y1).abs().max()))
            else:
                assert rel(y1, y0) < 1e-6, (K.meta.get("cell_type"), rel(y1, y0))


def test_normalisers_follow_the_definition():
    Ka = _graded_quads([1, 2, 4, 2, 1, 1], [4, 2, 1, 1, 2]).to(dtype=F64)
    Kb = CochainComplex.from_grid((7, 6), (0.21, 0.13), cell="quad", periodic=True).to(dtype=F64)
    ref = unit_reference(Ka, "dec", F64)
    frozen = dict(ref, key=("test", 1))
    own, fr = make_context(Kb, scaling="dec", B=1, dtype=F64), make_context(Kb, scaling="dec", B=1, dtype=F64,
                                                                                frozen=frozen)
    rb = unit_reference(Kb, "dec", F64)
    from rhmp.geometry import GEO_RELATIVE_COLUMNS
    for k in range(3):
        torch.testing.assert_close(fr.log_star[k], torch.log(Kb.star[k].double()) - ref["log_gm"][k])
        cols = torch.zeros(Kb.geo_dim(k), dtype=torch.bool)
        cols[list(GEO_RELATIVE_COLUMNS[k])] = True
        shift = (rb["geo_log_median"][k] - ref["geo_log_median"][k])[cols]
        torch.testing.assert_close(fr.geo[k][:, cols], own.geo[k][:, cols] + shift)
        torch.testing.assert_close(fr.geo[k][:, ~cols], own.geo[k][:, ~cols])
    for key, sc in fr.beta_scale.items():
        # in Kb's own units beta_unit(Kb) = rb; with the frozen star means it changes by exp(+-(gm_b - gm_a)_k)
        k, up = key
        expo = (rb["log_gm"][k] - ref["log_gm"][k]) * (1.0 if up else -1.0)
        torch.testing.assert_close(sc.reshape(()), ref["beta_unit"][key] / (rb["beta_unit"][key] * torch.exp(-expo)))


def test_global_scale_changes_the_output(device):
    K = _graded_quads([1, 2, 4, 2, 1, 1], [4, 2, 1, 1, 2], device=device).to(dtype=F64)
    K3 = _graded_quads([1, 2, 4, 2, 1, 1], [4, 2, 1, 1, 2], scale=3.0, device=device).to(dtype=F64)
    cfg = _cfg(K)
    m = build_model(cfg, K, dtype=F64)
    inp = make_inputs(cfg, K, dtype=F64)
    with torch.no_grad():
        assert rel(m(inp, K3), m(inp, K)) < 1e-6                        # default: invariant to the length unit
        #                                                                 (descriptors are stored in float32)
        m.freeze_units(K)
        y, y3 = m(inp, K), m(inp, K3)
    assert rel(y3, y) > 1e-3                                            # frozen units: the unit is that of K


@pytest.mark.parametrize("reflect", [False, True])
def test_rigid_motions(reflect, device):
    pos, cells = _MESHES["delaunay"]()
    R = random_orthogonal(2, seed=5, reflect=reflect)
    K = _build("delaunay", pos, cells, device).to(dtype=F64)
    K2 = _build("delaunay", pos @ R.T + np.array([0.7, -1.3]), cells, device).to(dtype=F64)
    Kref = _build("delaunay", 1.4 * pos, cells, device).to(dtype=F64)      # another mesh as training reference
    cfg = _cfg(K)
    m = build_model(cfg, K, dtype=F64).freeze_units(Kref)
    inp = make_inputs(cfg, K, dtype=F64)
    with torch.no_grad():
        torch.testing.assert_close(m(inp, K2), m(inp, K), rtol=1e-9, atol=1e-10)


def test_relabelling_and_orientation_flips(device):
    for name in ("delaunay", "mixed"):
        pos, cells = _MESHES[name]()
        pos2, cells2, perm = relabel_mesh(pos, cells, seed=4)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.size)
        K, K2 = _build(name, pos, cells, device), _build(name, pos2, cells2, device)
        maps = _cell_maps(K, K2, inv)
        cfg = _cfg(K)
        m = build_model(cfg, K).freeze_units(_build(name, 0.8 * pos, cells, device))
        inp = make_inputs(cfg, K)
        inp2 = {0: inp[0][torch.as_tensor(perm, device=device)]}
        for k, v in inp.items():
            if k > 0:
                inp2[k] = _map_cells(v, *maps[k], n_odd=cfg.in_dims[k] - cfg.even_dims.get(k, 0))
        with torch.no_grad():
            y, y2 = m(inp, K), m(inp2, K2)
        odeg, odd = _out_info(cfg.readout, K.dim)
        assert rel(y2, _map_cells(y, *maps[odeg], n_odd=(y.shape[-1] if odd else 0))) < 1e-5, name
    pos, faces = _MESHES["grid"]()
    flip = np.random.default_rng(6).random(len(faces)) < 0.4
    faces2 = faces.copy()
    faces2[flip] = faces[flip][:, ::-1]
    K, K2 = _build("grid", pos, faces, device), _build("grid", pos, faces2, device)
    sign = torch.as_tensor(np.where(flip, -1.0, 1.0), dtype=torch.float32, device=device)
    idx = torch.arange(len(faces), device=device)
    cfg = _cfg(K)
    m = build_model(cfg, K).freeze_units(_build("grid", 1.3 * pos, faces, device))
    inp = make_inputs(cfg, K)
    inp2 = {**inp, 2: _map_cells(inp[2], idx, sign, n_odd=cfg.in_dims[2] - cfg.even_dims.get(2, 0))}
    with torch.no_grad():
        y, y2 = m(inp, K), m(inp2, K2)
    assert rel(y2, _map_cells(y, idx, sign, n_odd=y.shape[-1])) < 1e-5


def test_block_diagonal_batches(device):
    Ka = make_complex("delaunay", device).to(dtype=F64)
    Kb = _graded_quads([1, 2, 4, 2, 1, 1], [4, 2, 1, 1, 2], device=device).to(dtype=F64)
    Kc = CochainComplex.from_grid((5, 4), 0.25, cell="tri", periodic=True, device=device).to(dtype=F64)
    cfg = _cfg(Ka)
    m = build_model(cfg, Ka, dtype=F64).freeze_units(Kb)
    ia, ic = make_inputs(cfg, Ka, B=1, dtype=F64), make_inputs(cfg, Kc, B=1, dtype=F64, seed=3)
    Kab = batch_complexes([Ka, Kc])
    iab = {k: torch.cat([ia[k], ic[k]], 0) for k in ia}
    with torch.no_grad():
        yab = m(iab, Kab)
        ya, yc = m(ia, Ka), m(ic, Kc)
    n = Ka.n[2]
    torch.testing.assert_close(yab[:n], ya, rtol=1e-9, atol=1e-10)
    torch.testing.assert_close(yab[n:], yc, rtol=1e-9, atol=1e-10)


def test_operators_only(device):
    """``frozen_descriptors=False``: stars and block bounds in training units, descriptors relative to each mesh."""
    Ka = _graded_quads([1, 2, 4, 2, 1, 1], [4, 2, 1, 1, 2], device=device).to(dtype=F64)
    Kb = CochainComplex.from_grid((7, 6), (0.21, 0.13), cell="quad", periodic=True, device=device).to(dtype=F64)
    cfg = _cfg(Ka, frozen_descriptors=False)
    m = build_model(cfg, Ka, dtype=F64)
    inp, ib = make_inputs(cfg, Ka, dtype=F64), make_inputs(cfg, Kb, dtype=F64, seed=2)
    with torch.no_grad():
        y0 = m(inp, Ka)
        m.freeze_units(Ka)
        y1, yb = m(inp, Ka), m(ib, Kb)
        ops_only = m.frozen_reference()
        m.cfg.frozen_descriptors = True
        full = m.frozen_reference()
        yb_full = m(ib, Kb)
    if str(device) == "cpu":
        assert torch.equal(y0, y1)
    else:
        assert rel(y1, y0) < 1e-6
    assert ops_only["geo_log_median"] is None and ops_only["key"] != full["key"]
    own = make_context(Kb, scaling="dec", B=1, dtype=F64)
    co, cf = (make_context(Kb, scaling="dec", B=1, dtype=F64, frozen=fr) for fr in (ops_only, full))
    for k in range(3):
        assert torch.equal(co.geo[k], own.geo[k]) and torch.equal(co.log_star[k], cf.log_star[k])
    assert all(torch.equal(co.beta_scale[key], cf.beta_scale[key]) for key in cf.beta_scale)
    assert rel(yb, yb_full) > 1e-6                                       # the descriptor shift matters elsewhere


def test_auto_descriptors(device):
    """``frozen_descriptors='auto'``: identical on the training mesh; on a mesh whose medians lie in the training
    ranges it reproduces ``True``; on a uniformly refined mesh (medians below the ranges) the scale columns are those of
    ``False``; stars and block bounds are frozen in all three settings."""
    sx, sy = [1, 2, 4, 2, 1, 1], [4, 2, 1, 1, 2]
    Ka = _graded_quads(sx, sy, device=device).to(dtype=F64)
    inside = CochainComplex.from_grid((8, 8), 1 / 8, cell="quad", periodic=True, device=device).to(dtype=F64)
    fine = _graded_quads(np.repeat(np.array(sx) / 4, 4), np.repeat(np.array(sy) / 4, 4), device=device).to(dtype=F64)
    cfg = _cfg(Ka, frozen_descriptors="auto")
    m = build_model(cfg, Ka, dtype=F64)
    inp = make_inputs(cfg, Ka, dtype=F64)
    with torch.no_grad():
        y0 = m(inp, Ka)
        m.freeze_units(Ka)
        y1 = m(inp, Ka)
    assert torch.equal(y0, y1) if str(device) == "cpu" else rel(y1, y0) < 1e-6
    refs = {}
    for mode in (True, False, "auto"):
        m.cfg.frozen_descriptors = mode
        refs[mode] = m.frozen_reference()
    scale_cols = {0: (0, 1), 1: (0,), 2: (0, 1)}                          # log measures / lengths (2-D)
    for Kt, same in ((inside, True), (fine, False)):
        ctx = {mode: make_context(Kt, scaling="dec", B=1, dtype=F64, frozen=fr) for mode, fr in refs.items()}
        for k in range(3):
            cols = list(scale_cols[k]) if Kt is fine else slice(None)
            assert torch.equal(ctx["auto"].geo[k][:, cols], ctx[same].geo[k][:, cols]), (k, same)
            assert torch.equal(ctx["auto"].log_star[k], ctx[True].log_star[k])
            assert torch.equal(ctx["auto"].log_star[k], ctx[False].log_star[k])
        bs = ctx[False].beta_scale
        assert all(torch.equal(ctx["auto"].beta_scale[key], bs[key]) for key in bs)
    it = make_inputs(cfg, inside, dtype=F64, seed=3)
    with torch.no_grad():
        ys = {}
        for mode in (True, False, "auto"):
            m.cfg.frozen_descriptors = mode
            ys[mode] = m(it, inside)
    assert torch.equal(ys["auto"], ys[True]) if str(device) == "cpu" else rel(ys["auto"], ys[True]) < 1e-6
    assert rel(ys[True], ys[False]) > 1e-6
    with pytest.raises(ValueError, match="frozen_descriptors"):
        dataclasses.replace(cfg, frozen_descriptors="train")


def test_checkpoints_and_refusals(device):
    K = _graded_quads([1, 2, 4, 2, 1, 1], [4, 2, 1, 1, 2], device=device).to(dtype=F64)
    Ko = CochainComplex.from_grid((6, 6), 1 / 6, cell="quad", periodic=True, device=device).to(dtype=F64)
    cfg = _cfg(K, frozen_units=True)
    m = build_model(cfg, K, dtype=F64)
    inp = make_inputs(cfg, K, dtype=F64)
    with pytest.raises(ValueError, match="freeze_units"):
        m(inp, K)
    m.freeze_units(K)
    m2 = RHMP.from_checkpoint(m.to_checkpoint()).to(device=device, dtype=F64)
    io = make_inputs(cfg, Ko, dtype=F64, seed=2)
    with torch.no_grad():
        torch.testing.assert_close(m2(io, Ko), m(io, Ko))
    expect = {"frozen_log_gm", "frozen_beta"} | {f"frozen_geo_{s}{k}" for s in ("", "min_", "max_") for k in range(3)}
    assert set(k for k in m.state_dict() if k.startswith("frozen_")) == expect
    # tensor metrics, normalised resolvent layers and projection layers are covered by frozen_units
    # (tests/test_frozen_units_tensor.py; the projection solve is independent of the operator normalisation), so they
    # are accepted; solve layers, un-normalised resolvent layers and Jacobi scaling are refused
    for ok in (dict(layers=["poly", "resolvent"]), dict(metric_type="tensor"), dict(layers=["project", "poly"])):
        assert dataclasses.replace(cfg, **ok).frozen_units
    for bad in (dict(layers=["poly", "solve"]), dict(layers=["poly", "resolvent"], resolvent_normalize=False),
                dict(scaling="jacobi")):
        with pytest.raises(ValueError, match="frozen_units"):
            dataclasses.replace(cfg, **bad)
    # a projection first layer: frozen on the training mesh, the model is unchanged (the solve is normalisation-free)
    cfgp = dataclasses.replace(cfg, layers=["project", "poly"], project_solver="dense")
    mp = build_model(cfgp, K, dtype=F64)
    inp = make_inputs(cfgp, K, dtype=F64)
    mp.freeze_units(K)
    mq = build_model(dataclasses.replace(cfgp, frozen_units=False), K, dtype=F64)
    mq.load_state_dict({k: v for k, v in mp.state_dict().items() if not k.startswith("frozen_")}, strict=False)
    with torch.no_grad():
        torch.testing.assert_close(mp(inp, K), mq(inp, K))
