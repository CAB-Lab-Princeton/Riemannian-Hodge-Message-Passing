"""Optional upgrades of the message-passing layer and of the cochain readout (defaults unchanged):

* ``gate='homog'``: update ``x + gamma g(xi) m`` homogeneous of degree one in the field amplitude
  (``rhmp.layers.RadialGate``);
* ``term_gates=True``: per-term gates ``m = sum_j c_j g_j t_j`` from scale-invariant invariants
  (``rhmp.layers.TermGate``);
* ``cochain_head='gated'``: odd gated head of ``cochain:k`` readouts (``rhmp.readout.CochainReadout``).

Every variant keeps the invariants of the method: O(C) equivariance of the message-passing stack, E(n) invariance,
vertex relabelling and face orientation parity, batch independence (also on the periodic grid of T1p).  Specific
properties: identity at initialisation, scale invariance of the gate factors, homogeneity of degree one of a model
whose only nonlinearities are these gates, oddness of the gated head, unchanged defaults and checkpoints.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from conftest import _MESHES, make_complex, random_orthogonal, relabel_mesh
from rhmp import RHMP, RHMPConfig
from rhmp.complex import CochainComplex
from rhmp.layers import RadialGate, TermGate
from test_equivariance import _build, _cell_maps, _check_OC, _map_cells, _out_info, _readouts
from test_model import F64, build_model, make_inputs, rel, small_cfg

UP_VARIANTS = {
    "homog": dict(gate="homog"),
    "terms": dict(term_gates=True),
    "homog_terms": dict(gate="homog", term_gates=True),
    "homog_terms_jacobi": dict(gate="homog", term_gates=True, scaling="jacobi"),
    "homog_terms_p3_nocross": dict(gate="homog", term_gates=True, poly_order=3, cross=False),
    "homog_terms_untied": dict(gate="homog", term_gates=True, tie_metrics=False),
    "homog_terms_resolvent": dict(gate="homog", term_gates=True, layers=["poly", "resolvent", "poly"]),
}
READOUT_VARIANTS = {"gated_head": dict(cochain_head="gated"),
                    "gated_head_h4": dict(cochain_head="gated", readout_hidden=4),
                    "context_head": dict(cochain_head="context"),
                    "amplitude": dict(amplitude_dims={1: 2}),
                    "context_amplitude_homog": dict(cochain_head="context", amplitude_dims={1: 2, 2: 1}, gate="homog",
                                                    term_gates=True),
                    "lift_context": dict(lift_context=4, cochain_head="context"),
                    "lift_vertex": dict(lift_vertex=True, lift_context=4, cochain_head="context", gate="homog"),
                    "multilayer": dict(readout_layers="all", cochain_head="context", gate="homog", term_gates=True),
                    "multilayer_linear": dict(readout_layers="all"),
                    "deep_head": dict(cochain_head="context", readout_depth=2, readout_layers="all"),
                    "skip_head": dict(cochain_head="context", readout_skip=True, gate="homog"),
                    "spatial_head": dict(cochain_head="context", readout_depth=2, readout_spatial=True),
                    "wedge_head": dict(cochain_head="context", readout_wedge=3, readout_depth=2, readout_spatial=True,
                                       gate="homog", term_gates=True)}


def _usable(readouts, opts):
    """Readouts a variant supports (the multi-layer readout exists for cochain:k readouts only, the wedge head for
    cochain:2)."""
    if opts.get("readout_wedge"):
        return [r for r in readouts if r == "cochain:2"]
    if opts.get("readout_layers") == "all":
        return [r for r in readouts if r.startswith("cochain")]
    return list(readouts)


def _periodic_quads(device):
    return CochainComplex.from_grid((6, 5), (0.3, 0.2), cell="quad", periodic=True, device=device)


@pytest.mark.parametrize("variant", list(UP_VARIANTS))
def test_OC_equivariance_upgrades(variant, device):
    for K in (make_complex("delaunay", device), _periodic_quads(device), make_complex("tets", device, n=30)):
        _check_OC(K.to(device, dtype=F64), small_cfg(dim=K.dim, n_layers=2, **UP_VARIANTS[variant]), F64, 1e-10)


@pytest.mark.parametrize("variant", list(UP_VARIANTS) + list(READOUT_VARIANTS))
def test_vertex_relabelling_upgrades(variant, device):
    opts = {**UP_VARIANTS.get(variant, {}), **READOUT_VARIANTS.get(variant, {})}
    for name in ("delaunay", "mixed"):
        pos, cells = _MESHES[name]()
        pos2, cells2, perm = relabel_mesh(pos, cells, seed=2)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.size)
        K, K2 = _build(name, pos, cells, device), _build(name, pos2, cells2, device)
        maps = _cell_maps(K, K2, inv)
        permt = torch.as_tensor(perm, device=device)
        for readout in _usable(("node_scalar", "cochain:1", "cochain:2", "even:2", "grad", "div"), opts):
            cfg = small_cfg(readout=readout, out_dim=2, **opts)
            m = build_model(cfg, K)
            inp = make_inputs(cfg, K)
            inp2 = {0: inp[0][permt]}
            for k, v in inp.items():
                if k > 0:
                    inp2[k] = _map_cells(v, *maps[k], n_odd=cfg.in_dims[k] - cfg.even_dims.get(k, 0))
            y, y2 = m(inp, K), m(inp2, K2)
            odeg, odd = _out_info(readout, K.dim)
            expect = y[permt] if odeg == 0 else _map_cells(y, *maps[odeg], n_odd=(y.shape[-1] if odd else 0))
            assert rel(y2, expect) < 1e-5, (name, readout, rel(y2, expect))


@pytest.mark.parametrize("variant", list(UP_VARIANTS) + list(READOUT_VARIANTS))
def test_face_orientation_flip_upgrades(variant, device):
    opts = {**UP_VARIANTS.get(variant, {}), **READOUT_VARIANTS.get(variant, {})}
    pos, faces = _MESHES["grid"]()
    flip = np.random.default_rng(3).random(len(faces)) < 0.4
    faces2 = faces.copy()
    faces2[flip] = faces[flip][:, ::-1]
    K, K2 = _build("grid", pos, faces, device), _build("grid", pos, faces2, device)
    sign = torch.as_tensor(np.where(flip, -1.0, 1.0), dtype=torch.float32, device=device)
    idx = torch.arange(len(faces), device=device)
    for readout in _usable(_readouts(2), opts):
        cfg = small_cfg(readout=readout, out_dim=2, **opts)
        m = build_model(cfg, K)
        inp = make_inputs(cfg, K)
        inp2 = {**inp, 2: _map_cells(inp[2], idx, sign, n_odd=cfg.in_dims[2] - cfg.even_dims.get(2, 0))}
        y, y2 = m(inp, K), m(inp2, K2)
        odeg, odd = _out_info(readout, 2)
        expect = _map_cells(y, idx, sign, n_odd=y.shape[-1]) if (odeg == 2 and odd) else y
        assert rel(y2, expect) < 1e-5, (readout, rel(y2, expect))


@pytest.mark.parametrize("variant", ["homog_terms", "gated_head", "context_amplitude_homog", "multilayer",
                                     "wedge_head"])
@pytest.mark.parametrize("reflect", [False, True])
def test_En_invariance_upgrades(variant, reflect, device):
    opts = {**UP_VARIANTS.get(variant, {}), **READOUT_VARIANTS.get(variant, {})}
    pos, cells = _MESHES["delaunay"]()
    R = random_orthogonal(2, seed=4, reflect=reflect)
    K = _build("delaunay", pos, cells, device)
    K2 = _build("delaunay", pos @ R.T + np.array([2.0, -1.0]), cells, device)
    for readout in _usable(("node_scalar", "cochain:1", "cochain:2", "node_vector"), opts):
        cfg = small_cfg(readout=readout, out_dim=2, **opts)
        m = build_model(cfg, K)
        inp = make_inputs(cfg, K)
        y, y2 = m(inp, K), m(inp, K2)
        if readout == "node_vector":
            y = (y.view(*y.shape[:2], -1, 2) @ torch.as_tensor(R, dtype=y.dtype, device=device).T).reshape(y.shape)
        assert rel(y2, y) < (1e-4 if readout == "node_vector" else 1e-5), (readout, rel(y2, y))


@pytest.mark.parametrize("variant", ["homog_terms", "gated_head", "context_amplitude_homog", "multilayer",
                                     "wedge_head"])
def test_batch_independence_upgrades(variant, device):
    opts = {**UP_VARIANTS.get(variant, {}), **READOUT_VARIANTS.get(variant, {})}
    for K in (make_complex("delaunay", device), _periodic_quads(device)):
        cfg = small_cfg(readout="cochain:2", n_layers=3, **opts)
        m = build_model(cfg, K)
        inp, other = make_inputs(cfg, K, B=16, seed=0), make_inputs(cfg, K, B=16, seed=1)
        alone = m({k: v[:, :1].contiguous() for k, v in inp.items()}, K)
        mixed = m({k: torch.cat([inp[k][:, :1], other[k][:, 1:]], 1) for k in inp}, K)
        # fp32 GEMMs block differently for B = 1 and B = 16 (on GPUs also with other kernels): relative tolerance
        tol = 1e-6 if str(device) == "cpu" else 1e-5
        assert (alone - mixed[:, :1]).abs().max() < tol * max(1.0, float(alone.detach().abs().max()))


def test_homog_gate_identity_at_init_and_scale_invariance():
    g = torch.Generator().manual_seed(0)
    x, m = torch.randn(50, 3, 8, generator=g, dtype=F64), torch.randn(50, 3, 8, generator=g, dtype=F64)
    gate = RadialGate("homog").double()
    torch.testing.assert_close(gate(x, m), x + m)                           # g = 1, gamma = 1 at initialisation
    with torch.no_grad():
        for p in gate.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=F64))
    a = gate.homog_scale(x, m)
    for lam in (1e-1, 7.0, 300.0):
        torch.testing.assert_close(gate.homog_scale(lam * x, lam * m), a, rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(gate(lam * x, lam * m), lam * gate(x, m), rtol=1e-6, atol=1e-8)
    z = torch.zeros(4, 3, 8, dtype=torch.float32)                          # all-zero cells: finite, factor = gamma g(0)
    assert bool(torch.isfinite(RadialGate("homog")(z, z)).all()) and bool(torch.isfinite(TermGate(2)(z, [z, z])).all())
    tg = TermGate(3).double()
    terms = [torch.randn(50, 3, 8, generator=g, dtype=F64) for _ in range(3)]
    torch.testing.assert_close(tg(x, terms), torch.ones(50, 3, 3, dtype=F64))
    with torch.no_grad():
        for p in tg.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=F64))
    torch.testing.assert_close(tg(5.0 * x, [5.0 * t for t in terms]), tg(x, terms), rtol=1e-6, atol=1e-8)


def test_term_gates_at_init_equal_the_plain_polynomial_layer(device):
    """Zero-initialised term gates (g = 1) reproduce the fused polynomial layer with the same parameters."""
    K = make_complex("delaunay", device).to(dtype=F64)
    for opts in (dict(), dict(scaling="jacobi"), dict(poly_order=3, cross=False), dict(gate="homog")):
        cfg = small_cfg(n_layers=2, **opts)
        m0 = build_model(cfg, K, dtype=F64)
        m1 = RHMP(dataclasses.replace(cfg, term_gates=True), K.geo_dims).to(device=device, dtype=F64)
        missing = m1.load_state_dict(m0.state_dict(), strict=False).missing_keys
        assert missing and all(".term_gates." in k for k in missing)
        inp = make_inputs(cfg, K, dtype=F64)
        assert rel(m1.hidden(inp, K)[2], m0.hidden(inp, K)[2]) < 1e-12


def test_homogeneous_model_is_homogeneous_of_degree_one(device):
    """Linear lifting, fixed metric, homogeneous gates (term gates included) and a linear readout: the only
    nonlinearities are scale invariant, so ``y(lambda f) = lambda y(f)``.  With the bounded radial gate it is not."""
    K = _periodic_quads(device).to(dtype=F64)
    base = dict(in_dims={0: 2, 1: 2}, even_dims={}, C=8, n_layers=3, lifting="linear", learn_metric=False,
                readout="cochain:2")
    for opts, homogeneous in ((dict(gate="homog", term_gates=True), True), (dict(gate="homog"), True),
                              (dict(gate="norm"), False)):
        cfg = RHMPConfig(**base, **opts)
        m = build_model(cfg, K, dtype=F64)
        inp = make_inputs(cfg, K, dtype=F64)
        y = m(inp, K)
        for lam in (0.01, 30.0):
            err = rel(m({k: lam * v for k, v in inp.items()}, K), lam * y)
            assert (err < 1e-8) if homogeneous else (err > 1e-3), (opts, lam, err)


def test_gated_head_is_odd_and_linear_at_init(device):
    K = make_complex("grid", device).to(dtype=F64)
    cfg = small_cfg(readout="cochain:2", cochain_head="gated", readout_hidden=4)
    m = build_model(cfg, K, randomize=False, dtype=F64)
    ro = m.readout
    assert ro.head == "gated" and ro.even_dim == cfg.even_dims[2]
    x = [torch.randn(K.n[k], 3, cfg.C, dtype=F64, device=device) for k in range(3)]
    ctx = m._prepare(make_inputs(cfg, K, B=3, dtype=F64), K)[1]
    torch.testing.assert_close(ro(x, ctx), (x[2] @ ro.lin1.weight.T) @ ro.lin2.weight.T)   # linear at init
    with torch.no_grad():
        for p in ro.parameters():
            p.add_(torch.randn_like(p))
    xn = [v.clone() for v in x]
    xn[2] = -x[2]
    torch.testing.assert_close(ro(xn, ctx), -ro(x, ctx))                     # odd in the face features


def test_defaults_unchanged_and_old_configs_load():
    cfg = RHMPConfig(in_dims={0: 2, 1: 2}, even_dims={1: 1}, readout="cochain:2")
    assert cfg.gate == "norm" and not cfg.term_gates and cfg.cochain_head == "linear"
    old = {k: v for k, v in cfg.to_dict().items() if k not in ("term_gates", "cochain_head", "readout_hidden",
                                                                "amplitude_dims", "lift_context", "lift_vertex",
                                                                "readout_layers", "readout_depth",
                                                                "readout_skip", "readout_spatial")}
    assert RHMPConfig.from_dict(old) == cfg
    assert cfg.amplitude_dims == {} and cfg.lift_context == 0
    m = RHMP(cfg, {0: 5, 1: 6, 2: 5})
    keys = set(m.state_dict())
    assert not any(".term_gates." in k for k in keys) and "readout.lin.weight" in keys
    assert not any("ctx_proj" in k for k in keys)
    assert all(m.layers[0].gates[k].mlp[0].weight.shape == (16, 1) for k in range(3))
    with pytest.raises(ValueError):
        RHMPConfig(in_dims={0: 1}, gate="other")
    with pytest.raises(ValueError):
        RHMPConfig(in_dims={0: 1}, cochain_head="mlp")
    with pytest.raises(ValueError, match="readout_layers"):
        RHMPConfig(in_dims={0: 1}, readout="node_scalar", readout_layers="all")
    with pytest.raises(ValueError, match="term_gates"):
        RHMP(RHMPConfig(in_dims={0: 1}, term_gates=True, metric_type="tensor"), {0: 5, 1: 6, 2: 5})


def test_trainer_flags(tmp_path):
    from rhmp.data import TaskData
    from rhmp.train import build_config, parse_args, run
    K = CochainComplex.from_grid((5, 5), 0.25, cell="quad", periodic=True)
    g = torch.Generator().manual_seed(0)
    N = 16
    task = TaskData.from_arrays(K, {0: torch.randn(N, K.n[0], 2, generator=g),
                                    1: torch.randn(N, K.n[1], 2, generator=g)},
                                torch.randn(N, K.n[2], 1, generator=g), readout="cochain:2", even_dims={1: 1},
                                name="tiny", meta={"v1_C": 8, "layers": 2, "batch_size": 4})
    args = parse_args(["--task", "tiny", "--epochs", "2", "--out", str(tmp_path), "--device", "cpu", "--C", "8",
                       "--layers", "2", "--quiet", "--gate", "homog", "--term-gates", "--cochain-head", "gated",
                       "--readout-hidden", "4"])
    cfg = build_config(task, args)
    assert cfg.gate == "homog" and cfg.term_gates and cfg.cochain_head == "gated" and cfg.readout_hidden == 4
    res = run(args, task=task)
    assert res["complete"] and np.isfinite(res["test"]["R2"])
    cfg0 = build_config(task, parse_args(["--task", "tiny"]))
    assert cfg0.gate == "norm" and not cfg0.term_gates and cfg0.cochain_head == "linear"


@pytest.mark.parametrize("readout", ["cochain:2", "cochain:1", "node_scalar", "grad"])
def test_amplitude_normalisation_is_homogeneous(readout, device):
    """With ``amplitude_dims`` the model is exactly homogeneous of degree one in the field-amplitude columns
    (whatever the rest of the network does); the other columns are not scaled."""
    K = make_complex("delaunay", device).to(dtype=F64)
    cfg = small_cfg(readout=readout, out_dim=2, amplitude_dims={1: 2, 2: 1}, cochain_head="context")
    m = build_model(cfg, K, dtype=F64)
    inp = make_inputs(cfg, K, dtype=F64)
    y = m(inp, K)
    for lam in (0.03, 11.0):
        scaled = {0: inp[0], 1: lam * inp[1], 2: torch.cat([lam * inp[2][..., :1], inp[2][..., 1:]], -1)}
        assert rel(m(scaled, K), lam * y) < 1e-10, (readout, lam)
    other = {**inp, 0: 2.0 * inp[0]}                                       # vertex columns are not amplitude columns
    assert rel(m(other, K), y) > 1e-3


def test_amplitude_normalisation_block_diagonal(device):
    """Per-graph amplitudes: a block-diagonal batch equals its parts evaluated alone."""
    from conftest import make_batch
    Kb, parts = make_batch(device)
    cfg = small_cfg(readout="cochain:2", amplitude_dims={1: 2})
    m = build_model(cfg, parts[0])
    ins = [make_inputs(cfg, K, B=1, seed=i) for i, K in enumerate(parts)]
    yb = m({k: torch.cat([x[k] for x in ins], 0) for k in ins[0]}, Kb)
    ys = torch.cat([m(x, K) for x, K in zip(ins, parts)], 0)
    assert rel(yb, ys) < 1e-5


@pytest.mark.parametrize("readout", ["node_scalar", "cochain:2", "grad"])
def test_gauge_invariance_with_upgrades(readout, device):
    """Exact gauge invariance A -> A + d_0 lambda of connection inputs holds with the layer, lifting and readout
    upgrades switched on (the connection columns still enter only through d_1 A; lift_vertex excludes them)."""
    K = make_complex("sphere", device)
    cfg = RHMPConfig(in_dims={0: 1, 1: 3}, even_dims={0: 1, 1: 1}, connection_dims={1: 1}, C=8, n_layers=2,
                     readout=readout, gate="homog", term_gates=True, cochain_head="context", lift_context=4,
                     lift_vertex=True)
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K, B=2)
    lam = 3.0 * torch.randn(K.n[0], 2, 1, generator=torch.Generator().manual_seed(9)).to(device)
    inp2 = {**inp, 1: inp[1].clone()}
    inp2[1][..., :1] += K.apply_d(0, lam)
    assert rel(m(inp2, K), m(inp, K)) < 1e-5


def test_multilayer_readout_reads_every_layer(device):
    """readout_layers='all': the output depends on the features after the lifting and after every layer through
    separate maps; with those maps zeroed it equals the single-layer readout."""
    K = make_complex("grid", device).to(dtype=F64)
    cfg = small_cfg(readout="cochain:2", cochain_head="context", readout_layers="all", n_layers=3)
    m = build_model(cfg, K, dtype=F64)
    assert len(m.readout.lin_hist) == 3
    inp = make_inputs(cfg, K, dtype=F64)
    y = m(inp, K)
    m1 = RHMP(dataclasses.replace(cfg, readout_layers="last"), K.geo_dims).to(device=device, dtype=F64)
    m1.load_state_dict({k: v for k, v in m.state_dict().items() if "lin_hist" not in k})
    with torch.no_grad():
        for lin in m.readout.lin_hist:
            lin.weight.zero_()
    torch.testing.assert_close(m(inp, K), m1(inp, K))
    assert rel(y, m1(inp, K)) > 1e-3


def test_deep_head_is_linear_at_init_and_odd(device):
    K = make_complex("grid", device).to(dtype=F64)
    cfg = small_cfg(readout="cochain:2", cochain_head="context", readout_depth=3, readout_hidden=4)
    m = build_model(cfg, K, randomize=False, dtype=F64)
    ro = m.readout
    assert len(ro.mix) == 2 and len(ro.gates) == 2
    x = [torch.randn(K.n[k], 3, cfg.C, dtype=F64, device=device) for k in range(3)]
    ctx = m._prepare(make_inputs(cfg, K, B=3, dtype=F64), K)[1]
    u = x[2] @ ro.lin1.weight.T + K.apply_d(1, x[1].contiguous()) @ ro.lin_c.weight.T
    torch.testing.assert_close(ro(x, ctx), u @ ro.lin2.weight.T)            # linear at initialisation
    with torch.no_grad():
        for p in ro.parameters():
            p.add_(0.3 * torch.randn_like(p))
    xn = [v.clone() for v in x]
    xn[1], xn[2] = -x[1], -x[2]                                            # flipping every odd input of degree 1, 2
    torch.testing.assert_close(ro(xn, ctx), -ro(x, ctx))


def test_readout_skip_is_linear_in_the_inputs(device):
    """With the head gates at zero, readout_skip adds exactly S_2 f_2^odd + S_1 d_1 f_1^odd to the hidden part."""
    K = make_complex("grid", device).to(dtype=F64)
    cfg = small_cfg(readout="cochain:2", cochain_head="context", readout_skip=True)
    m = build_model(cfg, K, randomize=False, dtype=F64)
    ro = m.readout
    assert ro.skip == ((0, 1), (0, 2)) and ro.skip_k.in_features == 1 and ro.skip_km1.in_features == 2
    inp = make_inputs(cfg, K, B=3, dtype=F64)
    x = [torch.randn(K.n[k], 3, cfg.C, dtype=F64, device=device) for k in range(3)]
    ctx = m._prepare(inp, K)[1]
    u = (x[2] @ ro.lin1.weight.T + K.apply_d(1, x[1].contiguous()) @ ro.lin_c.weight.T
         + inp[2][..., :1] @ ro.skip_k.weight.T + K.apply_d(1, inp[1].contiguous()) @ ro.skip_km1.weight.T)
    torch.testing.assert_close(ro(x, ctx), u @ ro.lin2.weight.T)
    with pytest.raises(ValueError, match="readout_skip"):
        RHMP(small_cfg(readout="cochain:2", readout_skip=True), K.geo_dims)


@pytest.mark.parametrize("mesh", ["periodic_quads", "delaunay", "quads", "mixed", "tets"])
def test_face_wedge_exact_antisymmetric(mesh, device):
    """face_wedge: dx ^ dy = signed face area (face orientation) on triangles and convex quadrilaterals, also across
    the wrap-around of the periodic grid; antisymmetric; sign flip with the face orientation."""
    from rhmp.readout import face_wedge
    K = (_periodic_quads(device) if mesh == "periodic_quads" else
         make_complex(mesh, device, n=30) if mesh == "tets" else make_complex(mesh, device)).to(dtype=F64)
    ev = K.edge_vectors().to(F64)
    a, b = ev[:, :1].contiguous(), ev[:, 1:2].contiguous()
    w = face_wedge(K, a, b)[:, 0]
    X = K.cell_positions(2).to(F64)[..., :2]                                 # (n2, m, 2) local frames, padded
    m = K.meta["face_lengths"] if "face_lengths" in K.meta else torch.full((K.n[2],), 3, device=device)
    area = torch.zeros(K.n[2], dtype=F64, device=device)
    for j in range(X.shape[1]):
        nj = torch.where(torch.as_tensor(j + 1, device=device) < m, j + 1, 0)
        valid = j < m
        Xn = X[torch.arange(K.n[2], device=device), nj]
        area = area + torch.where(valid, 0.5 * (X[:, j, 0] * Xn[:, 1] - X[:, j, 1] * Xn[:, 0]), area.new_zeros(()))
    if mesh != "tets":                                                       # planar faces in the xy plane
        sel = m <= 4
        assert bool(sel.any())
        torch.testing.assert_close(w[sel], area[sel], rtol=1e-9, atol=1e-12)
    g = torch.Generator().manual_seed(0)
    p, q = (torch.randn(K.n[1], 2, 3, generator=g, dtype=F64).to(device) for _ in range(2))
    torch.testing.assert_close(face_wedge(K, p, q), -face_wedge(K, q, p))
    assert float(face_wedge(K, p, p).abs().max()) < 1e-12
    if mesh == "delaunay":                                                    # reversed faces: values flip sign
        pos, faces = _MESHES["delaunay"]()
        K2 = _build("delaunay", pos, faces[:, ::-1].copy(), device).to(dtype=F64)
        assert K2.n == K.n and [int(i) for i in K2.meta["face_index"]] == [int(i) for i in K.meta["face_index"]]
        e1 = {tuple(sorted(e)): (i, e) for i, e in enumerate(K.cells[1].tolist())}
        idx = torch.tensor([e1[tuple(sorted(e))][0] for e in K2.cells[1].tolist()], device=device)
        sgn = torch.tensor([1.0 if e == e1[tuple(sorted(e))][1] else -1.0 for e in K2.cells[1].tolist()], dtype=F64,
                           device=device)[:, None, None]
        torch.testing.assert_close(face_wedge(K2, p[idx] * sgn, q[idx] * sgn), -face_wedge(K, p, q))


def test_wedge_head_zero_at_init_and_quadratic(device):
    """readout_wedge: the head is unchanged at initialisation (zero output map of the wedge features), the wedge part
    is quadratic in the edge features, and the option is refused without a gated / context cochain:2 head."""
    K = make_complex("grid", device).to(dtype=F64)
    cfg = small_cfg(readout="cochain:2", cochain_head="context", readout_wedge=4, readout_hidden=4)
    m = build_model(cfg, K, randomize=False, dtype=F64)
    ro = m.readout
    x = [torch.randn(K.n[k], 3, cfg.C, dtype=F64, device=device) for k in range(3)]
    ctx = m._prepare(make_inputs(cfg, K, B=3, dtype=F64), K)[1]
    y0 = ro(x, ctx)
    with torch.no_grad():
        ro.wedge_a.weight.add_(torch.randn_like(ro.wedge_a.weight))
    torch.testing.assert_close(ro(x, ctx), y0)
    with torch.no_grad():
        ro.lin_w.weight.normal_()
        ro.lin2.weight.normal_()
    u = x[2] @ ro.lin1.weight.T + K.apply_d(1, x[1].contiguous()) @ ro.lin_c.weight.T
    xs = [x[0], 2.0 * x[1], x[2]]                                          # wedge part scales with the square
    from rhmp.readout import face_wedge
    wv = face_wedge(K, x[1] @ ro.wedge_a.weight.T, x[1] @ ro.wedge_b.weight.T) @ ro.lin_w.weight.T
    ref = lambda uu: uu @ ro.lin2.weight.T                                  # gates are zero at initialisation
    torch.testing.assert_close(ro(x, ctx), ref(u + wv))
    us = x[2] @ ro.lin1.weight.T + K.apply_d(1, xs[1].contiguous()) @ ro.lin_c.weight.T
    torch.testing.assert_close(ro(xs, ctx), ref(us + 4.0 * wv))
    for bad in (dict(readout="cochain:2", readout_wedge=2), dict(readout="cochain:1", cochain_head="context",
                                                                 readout_wedge=2)):
        with pytest.raises(ValueError, match="readout_wedge"):
            RHMP(small_cfg(**bad), K.geo_dims)
