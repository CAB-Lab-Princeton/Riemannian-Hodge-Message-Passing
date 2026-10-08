"""``RHMPConfig.frozen_units`` with the tensor metric, normalised resolvent layers and the transport term (and the
diagonal polynomial stack it extends): the per-mesh geometric normalisers of the training mesh, stored in the model.

* on the training mesh the output is identical to the default (bitwise on the CPU) for every supported variant;
* on another mesh the stars, descriptors and block bounds follow their definition, the tensor-metric bound is
  ``rho * beta_unit(train)`` and the transport fluxes use the frozen bound of the degree-0 up block;
* a mesh scaled by a global factor now changes the output (the default model is invariant to the length unit);
* rigid motions, vertex relabelling, face orientation flips and block-diagonal batches keep their invariances;
  checkpoints carry the references; the trainer flag stores them; unsupported configurations are refused.
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from conftest import _MESHES, make_batch, make_complex, random_orthogonal, relabel_mesh
from rhmp import RHMP
from rhmp.layers import TensorMetric, block_beta_unit, block_operands, make_context, tensor_operands, unit_reference
from rhmp.transport import TransportTerm
from test_equivariance import _build, _cell_maps, _map_cells, _out_info
from test_model import F64, build_model, make_inputs, rel, small_cfg
from test_numerics import _rand_tensor_params

T2A = dict(metric_type="tensor", layers=["poly", "resolvent"], resolvent_iters=30,
           transport=dict(inputs=[0], order=2, metric="star"))
VARIANTS = {
    "diag": {},
    "tensor": dict(metric_type="tensor"),
    "resolvent": dict(layers=["poly", "resolvent"], resolvent_iters=30),
    "transport": dict(transport=dict(inputs=[0], hidden=1, order=2)),
    "tensor_transport": dict(metric_type="tensor", transport=dict(inputs=[0], order=2)),
    "t2a": T2A,
}
TENSOR = ("tensor", "tensor_transport", "t2a")


def _cfg(K, variant, **kw):
    return small_cfg(dim=K.dim, **dict(VARIANTS[variant], **kw))


def _scaled(name, factor, device):
    pos, cells = _MESHES[name]()
    return _build(name, factor * pos, cells, device)


@pytest.mark.parametrize("dtype", [F64, torch.float32])
def test_identical_on_the_training_mesh(dtype, device):
    for name in ("delaunay", "sphere", "tets", "mixed"):
        K = make_complex(name, device).to(device, dtype=dtype)
        for variant in VARIANTS:
            if variant in TENSOR and name == "mixed":                    # the tensor metric needs simplices
                continue
            cfg = _cfg(K, variant)
            m = build_model(cfg, K, dtype=dtype)
            inp = make_inputs(cfg, K, dtype=dtype)
            with torch.no_grad():
                y0 = m(inp, K)
                m.freeze_units(K)
                y1 = m(inp, K)
            if str(device) == "cpu":                                     # bitwise (GPU scatter-adds are not ordered)
                assert torch.equal(y0, y1), (name, variant, float((y0 - y1).abs().max()))
            else:
                assert rel(y1, y0) < 1e-6, (name, variant, rel(y1, y0))


def test_normalisers_follow_the_definition(device):
    Ka = make_complex("delaunay", device).to(dtype=F64)
    Kb = _scaled("grid", 0.37, device).to(dtype=F64)
    ref = unit_reference(Ka, "dec", F64)
    frozen = dict(ref, key=("test", 1))
    own, fr = make_context(Kb, scaling="dec", B=2, dtype=F64), make_context(Kb, scaling="dec", B=2, dtype=F64,
                                                                                frozen=frozen)
    rb = unit_reference(Kb, "dec", F64)
    from rhmp.geometry import GEO_RELATIVE_COLUMNS
    for k in range(3):
        torch.testing.assert_close(fr.log_star[k], torch.log(Kb.star[k].double()) - ref["log_gm"][k].to(device))
        cols = torch.zeros(Kb.geo_dim(k), dtype=torch.bool, device=device)
        cols[list(GEO_RELATIVE_COLUMNS[k])] = True
        shift = (rb["geo_log_median"][k] - ref["geo_log_median"][k]).to(device)[cols]
        torch.testing.assert_close(fr.geo[k][:, cols], own.geo[k][:, cols] + shift)
        torch.testing.assert_close(fr.geo[k][:, ~cols], own.geo[k][:, ~cols])
    for key, sc in fr.beta_scale.items():
        # in Kb's own units beta_unit(Kb) = rb; with the frozen star means it changes by exp(+-(gm_b - gm_a)_k)
        k, up = key
        expo = (rb["log_gm"][k] - ref["log_gm"][k]) * (1.0 if up else -1.0)
        torch.testing.assert_close(sc.reshape(()).cpu(),
                                   (ref["beta_unit"][key] / (rb["beta_unit"][key] * torch.exp(-expo))).cpu())
    # diagonal blocks (and the transport fluxes, which use the degree-0 up block): beta(H) beta_unit(train) / beta_unit
    g = torch.Generator().manual_seed(0)
    logH = (fr.log_star[1].unsqueeze(1) + torch.randn(Kb.n[1], 2, generator=g, dtype=F64).to(device))
    plain = dataclasses.replace(fr, beta_scale=None)
    torch.testing.assert_close(block_operands(fr, 0, True, logH).beta,
                               block_operands(plain, 0, True, logH).beta * fr.beta_scale[(0, True)])
    phi = torch.randn(Kb.n[1], 2, 1, generator=g, dtype=F64).to(device)
    term = TransportTerm(dict(inputs=[0], metric="star")).to(device=device, dtype=F64)
    op = block_operands(fr, 0, True, fr.log_star[1].unsqueeze(1).expand(-1, 2))
    torch.testing.assert_close(term.fluxes(phi, fr, None), torch.exp(op.log_h).unsqueeze(-1) * phi)
    # tensor metric: rho * beta_unit(train) (in the frozen star units of this mesh), also for the cross-up term
    b, a = _rand_tensor_params(Kb, 1, 2, g, device, F64)
    tm = TensorMetric(fr, 1, b=b, a=a)
    rho = fr.cell_max(tm.rowsum_abs(), 1)
    bu_train = block_beta_unit(fr, 0) * fr.beta_scale[(0, True)]
    torch.testing.assert_close(bu_train.reshape(()), ref["beta_unit"][(0, True)].to(device))
    opt = tensor_operands(fr, 0, tm)
    torch.testing.assert_close(opt.beta, rho * bu_train)
    torch.testing.assert_close(opt.cross_scale.reshape(-1, 2)[:1], (1.0 / (torch.sqrt(bu_train) * rho)))
    own_t = tensor_operands(plain, 0, TensorMetric(plain, 1, b=b, a=a))
    torch.testing.assert_close(opt.beta, own_t.beta * fr.beta_scale[(0, True)])


@pytest.mark.parametrize("variant", ["diag", "t2a"])
def test_global_scale_changes_the_output(variant, device):
    K = make_complex("delaunay", device).to(dtype=F64)
    K3 = _scaled("delaunay", 3.0, device).to(dtype=F64)
    cfg = _cfg(K, variant)
    m = build_model(cfg, K, dtype=F64)
    inp = make_inputs(cfg, K, dtype=F64)
    with torch.no_grad():
        assert rel(m(inp, K3), m(inp, K)) < 1e-6                        # default: invariant to the length unit
        #                                                                 (descriptors are stored in float32)
        m.freeze_units(K)
        y, y3 = m(inp, K), m(inp, K3)
    assert rel(y3, y) > 1e-3                                            # frozen units: the unit is that of K


@pytest.mark.parametrize("reflect", [False, True])
@pytest.mark.parametrize("variant", ["diag", "t2a"])
def test_rigid_motions(reflect, variant, device):
    """Exactly in fp64 on the same complex with transformed positions (tensor metrics recompute the Galerkin element
    geometry from the positions: tolerance 1e-9, as in ``test_equivariance.py::test_En_equivariance_exact_fp64``), and
    with the complex rebuilt from the transformed positions (geometry stored in float32)."""
    pos, cells = _MESHES["delaunay"]()
    R = random_orthogonal(2, seed=5, reflect=reflect)
    shift = np.array([0.7, -1.3])
    K = _build("delaunay", pos, cells, device).to(dtype=F64)
    K2 = dataclasses.replace(K, pos=K.pos @ torch.as_tensor(R.T, dtype=F64, device=device)
                             + torch.as_tensor(shift, dtype=F64, device=device))
    K2.whitney = {km: {**W, "t": W["t"] @ torch.as_tensor(R.T, dtype=W["t"].dtype, device=device)}
                  for km, W in K.whitney.items()}
    Kref = _build("delaunay", 1.4 * pos, cells, device).to(dtype=F64)      # another mesh as training reference
    cfg = _cfg(K, variant)
    m = build_model(cfg, K, dtype=F64).freeze_units(Kref)
    inp = make_inputs(cfg, K, dtype=F64)
    tensor = variant in TENSOR
    with torch.no_grad():
        assert rel(m(inp, K2), m(inp, K)) < (1e-9 if tensor else 1e-10)
        Kr = _build("delaunay", pos @ R.T + shift, cells, device).to(dtype=F64)
        assert rel(m(inp, Kr), m(inp, K)) < (1e-4 if tensor else 1e-6)


def test_relabelling_and_orientation_flips(device):
    for name, variant in (("delaunay", "t2a"), ("sphere", "tensor_transport"), ("mixed", "transport")):
        pos, cells = _MESHES[name]()
        pos2, cells2, perm = relabel_mesh(pos, cells, seed=4)
        inv = np.empty_like(perm)
        inv[perm] = np.arange(perm.size)
        K, K2 = _build(name, pos, cells, device), _build(name, pos2, cells2, device)
        maps = _cell_maps(K, K2, inv)
        cfg = _cfg(K, variant)
        m = build_model(cfg, K).freeze_units(_build(name, 0.8 * pos, cells, device))
        inp = make_inputs(cfg, K)
        inp2 = {0: inp[0][torch.as_tensor(perm, device=device)]}
        for k, v in inp.items():
            if k > 0:
                inp2[k] = _map_cells(v, *maps[k], n_odd=cfg.in_dims[k] - cfg.even_dims.get(k, 0))
        with torch.no_grad():
            y, y2 = m(inp, K), m(inp2, K2)
        odeg, odd = _out_info(cfg.readout, K.dim)
        want = (y[torch.as_tensor(perm, device=device)] if odeg == 0
                else _map_cells(y, *maps[odeg], n_odd=(y.shape[-1] if odd else 0)))
        assert rel(y2, want) < 1e-5, name
    pos, faces = _MESHES["grid"]()
    flip = np.random.default_rng(6).random(len(faces)) < 0.4
    faces2 = faces.copy()
    faces2[flip] = faces[flip][:, ::-1]
    K, K2 = _build("grid", pos, faces, device), _build("grid", pos, faces2, device)
    sign = torch.as_tensor(np.where(flip, -1.0, 1.0), dtype=torch.float32, device=device)
    idx = torch.arange(len(faces), device=device)
    cfg = _cfg(K, "t2a", readout="cochain:2")
    m = build_model(cfg, K).freeze_units(_build("grid", 1.3 * pos, faces, device))
    inp = make_inputs(cfg, K)
    inp2 = {**inp, 2: _map_cells(inp[2], idx, sign, n_odd=cfg.in_dims[2] - cfg.even_dims.get(2, 0))}
    with torch.no_grad():
        y, y2 = m(inp, K), m(inp2, K2)
    assert rel(y2, _map_cells(y, idx, sign, n_odd=y.shape[-1])) < 1e-5


@pytest.mark.parametrize("variant", ["diag", "t2a"])
def test_block_diagonal_batches(variant, device):
    Kb, parts = make_batch(device)
    Kb, parts = Kb.to(dtype=F64), [P.to(dtype=F64) for P in parts]
    if variant in TENSOR and Kb.meta.get("cell_type") != "triangle":
        pytest.skip("the batch of conftest.make_batch is not simplicial")
    cfg = _cfg(Kb, variant)
    m = build_model(cfg, parts[0], dtype=F64).freeze_units(_scaled("delaunay", 1.7, device).to(dtype=F64))
    ins = [make_inputs(cfg, P, B=1, dtype=F64, seed=i) for i, P in enumerate(parts)]
    with torch.no_grad():
        yb = m({k: torch.cat([x[k] for x in ins], 0) for k in ins[0]}, Kb)
        ptr = Kb.meta["ptr"][0].tolist()
        for i, (P, x) in enumerate(zip(parts, ins)):
            torch.testing.assert_close(yb[ptr[i]:ptr[i + 1]], m(x, P), rtol=1e-9, atol=1e-10)


def test_checkpoints_and_refusals(device):
    K = make_complex("delaunay", device).to(dtype=F64)
    Ko = _scaled("grid", 0.6, device).to(dtype=F64)
    cfg = _cfg(K, "t2a", frozen_units=True)
    m = build_model(cfg, K, dtype=F64)
    inp = make_inputs(cfg, K, dtype=F64)
    with pytest.raises(ValueError, match="freeze_units"):
        m(inp, K)
    m.freeze_units(K)
    m2 = RHMP.from_checkpoint(m.to_checkpoint()).to(device=device, dtype=F64)
    io = make_inputs(cfg, Ko, dtype=F64, seed=2)
    with torch.no_grad():
        torch.testing.assert_close(m2(io, Ko), m(io, Ko))
    assert set(k for k in m.state_dict() if k.startswith("frozen_")) == (
        {"frozen_log_gm", "frozen_beta"} | {f"frozen_geo_{s}{k}" for s in ("", "min_", "max_") for k in range(3)})
    for bad in (dict(layers=["poly", "solve"]), dict(resolvent_normalize=False), dict(scaling="jacobi", transport={}),
                dict(transport=dict(inputs=[0], norm="sample"))):
        with pytest.raises(ValueError, match="frozen_units"):
            dataclasses.replace(cfg, **bad)


def test_trainer_frozen_units_flag(tmp_path):
    from test_train import _args, tiny_shared_task
    from rhmp.train import run
    torch.set_num_threads(2)
    res = run(_args(tmp_path, "--batch", "8", "--frozen-units", "--transport", '{"inputs": [0], "order": 2}'),
              task=tiny_shared_task())
    assert res["complete"] and res["config"]["frozen_units"]
    ck = torch.load(tmp_path / "best.pt", map_location="cpu", weights_only=False)
    assert {"frozen_log_gm", "frozen_beta"} <= set(ck["state_dict"])


# ----------------------------------------------------------------------------------------------------------------
# frozen_descriptors=False: operators only (training-mesh star means and block bounds, descriptors relative to the mesh)
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [F64, torch.float32])
def test_operators_only_identical_on_the_training_mesh(dtype, device):
    for name in ("delaunay", "sphere", "tets"):
        K = make_complex(name, device).to(device, dtype=dtype)
        for variant in ("diag", "t2a"):
            cfg = _cfg(K, variant, frozen_descriptors=False)
            m = build_model(cfg, K, dtype=dtype)
            inp = make_inputs(cfg, K, dtype=dtype)
            with torch.no_grad():
                y0 = m(inp, K)
                m.freeze_units(K)
                y1 = m(inp, K)
            if str(device) == "cpu":
                assert torch.equal(y0, y1), (name, variant, float((y0 - y1).abs().max()))
            else:
                assert rel(y1, y0) < 1e-6, (name, variant, rel(y1, y0))


def test_operators_only_differs_only_through_the_descriptors(device):
    """Same stars, blocks and bounds as frozen_descriptors=True; the descriptors are the mesh's own (the default ones);
    the two settings give different outputs on another mesh and the same on the training mesh."""
    Ka = make_complex("delaunay", device).to(dtype=F64)
    Kb = _scaled("grid", 0.37, device).to(dtype=F64)
    ref = unit_reference(Ka, "dec", F64)
    own = make_context(Kb, scaling="dec", B=1, dtype=F64)
    on = make_context(Kb, scaling="dec", B=1, dtype=F64, frozen=dict(ref, key=("t", 1)))
    off = make_context(Kb, scaling="dec", B=1, dtype=F64, frozen=dict(ref, geo_log_median=None, key=("t", 0)))
    from rhmp.geometry import GEO_RELATIVE_COLUMNS
    for k in range(3):
        assert torch.equal(off.log_star[k], on.log_star[k])
        assert torch.equal(off.geo[k], own.geo[k])
        cols = list(GEO_RELATIVE_COLUMNS[k])
        rest = [j for j in range(Kb.geo_dim(k)) if j not in cols]
        assert torch.equal(on.geo[k][:, rest], own.geo[k][:, rest])
        assert not torch.equal(on.geo[k][:, cols], own.geo[k][:, cols])
    assert on.beta_scale.keys() == off.beta_scale.keys()
    for key in on.beta_scale:
        assert torch.equal(on.beta_scale[key], off.beta_scale[key])
    for (k, up), g in on.blocks.items():
        assert torch.equal(g.r, off.blocks[(k, up)].r)
    cfg = _cfg(Ka, "t2a")
    m = build_model(cfg, Ka, dtype=F64).freeze_units(Ka)
    ia, ib = make_inputs(cfg, Ka, dtype=F64), make_inputs(cfg, Kb, dtype=F64, seed=1)
    with torch.no_grad():
        ya_on, yb_on = m(ia, Ka), m(ib, Kb)
        m.cfg.frozen_descriptors = False
        ya_off, yb_off = m(ia, Ka), m(ib, Kb)
    assert torch.equal(ya_on, ya_off) if str(device) == "cpu" else rel(ya_off, ya_on) < 1e-6
    assert rel(yb_off, yb_on) > 1e-3


def test_operators_only_checkpoints_refusals_and_trainer_flag(tmp_path, device):
    K = make_complex("delaunay", device).to(dtype=F64)
    Ko = _scaled("grid", 0.6, device).to(dtype=F64)
    cfg = _cfg(K, "t2a", frozen_units=True, frozen_descriptors=False)
    m = build_model(cfg, K, dtype=F64).freeze_units(K)
    m2 = RHMP.from_checkpoint(m.to_checkpoint()).to(device=device, dtype=F64)
    assert m2.cfg.frozen_descriptors is False
    io = make_inputs(cfg, Ko, dtype=F64, seed=2)
    with torch.no_grad():
        torch.testing.assert_close(m2(io, Ko), m(io, Ko))
    for bad in (dict(layers=["poly", "solve"]), dict(resolvent_normalize=False), dict(scaling="jacobi", transport={}),
                dict(transport=dict(inputs=[0], norm="sample"))):
        with pytest.raises(ValueError, match="frozen_units"):
            dataclasses.replace(cfg, **bad)
    from test_train import _args, tiny_shared_task
    from rhmp.train import run
    torch.set_num_threads(2)
    with pytest.raises(ValueError, match="need --frozen-units"):
        run(_args(tmp_path / "a", "--batch", "8", "--no-frozen-descriptors"), task=tiny_shared_task())
    res = run(_args(tmp_path / "b", "--batch", "8", "--frozen-units", "--no-frozen-descriptors"),
              task=tiny_shared_task())
    assert res["complete"] and res["config"]["frozen_units"] and res["config"]["frozen_descriptors"] is False


def test_auto_descriptors_follow_the_rule(device):
    """``frozen_descriptors='auto'`` (T2a configuration): identical on the training mesh; on another mesh every relative
    column follows ``True`` when the mesh's median in training units lies in the training range of the column and
    ``False`` otherwise; stars and block bounds as in the other settings; invalid values refused."""
    from rhmp.geometry import GEO_RELATIVE_COLUMNS
    Ka = make_complex("delaunay", device).to(dtype=F64)
    cfg = _cfg(Ka, "t2a", frozen_descriptors="auto")
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
    ra = unit_reference(Ka, "dec", F64)
    n_out = {}
    for factor in (1.05, 0.05):
        Kt = _scaled("delaunay", factor, device).to(dtype=F64)
        rt = unit_reference(Kt, "dec", F64)
        ctx = {mode: make_context(Kt, scaling="dec", B=1, dtype=F64, frozen=fr) for mode, fr in refs.items()}
        n_out[factor] = 0
        for k in range(3):
            for j in GEO_RELATIVE_COLUMNS[k]:
                sh = float(rt["geo_log_median"][k][j] - ra["geo_log_median"][k][j])
                inside = float(ra["geo_min"][k][j]) <= sh <= float(ra["geo_max"][k][j])
                n_out[factor] += not inside
                assert torch.equal(ctx["auto"].geo[k][:, j], ctx[True if inside else False].geo[k][:, j]), (factor, k, j)
            assert torch.equal(ctx["auto"].log_star[k], ctx[False].log_star[k])
        assert all(torch.equal(ctx["auto"].beta_scale[key], ctx[False].beta_scale[key]) for key in ctx[False].beta_scale)
    assert n_out[1.05] == 0 and n_out[0.05] > 0
    with torch.no_grad():
        it = make_inputs(cfg, _scaled("delaunay", 1.05, device).to(dtype=F64), dtype=F64, seed=3)
        Ki = _scaled("delaunay", 1.05, device).to(dtype=F64)
        ys = {}
        for mode in (True, "auto"):
            m.cfg.frozen_descriptors = mode
            ys[mode] = m(it, Ki)
    assert torch.equal(ys["auto"], ys[True]) if str(device) == "cpu" else rel(ys["auto"], ys[True]) < 1e-6
    with pytest.raises(ValueError, match="frozen_descriptors"):
        dataclasses.replace(cfg, frozen_descriptors="train")
