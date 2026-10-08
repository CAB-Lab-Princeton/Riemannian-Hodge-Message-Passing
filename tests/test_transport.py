"""Transport term (``rhmp.transport``, ``RHMPConfig.transport``): operator identities, mass conservation, norm bounds,
agreement with a hand-written upwind finite-volume step, the symmetries of the paper (O(C), vertex relabelling and
edge orientation, face orientation, rigid motions, batch independence), configuration / checkpoint handling and the
T2v edge encoding of a node velocity field.
"""
from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
import torch

from conftest import _MESHES, make_batch, make_complex, random_orthogonal, relabel_mesh
from rhmp import CochainComplex, RHMPConfig
from rhmp.layers import make_context
from rhmp.model import RHMP
from rhmp.transport import (TransportTerm, apply_transport, normalize_transport, transport_bound, transport_columns,
                            transport_weights)
from test_equivariance import _build, _cell_maps, _map_cells
from test_model import F64, build_model, loss_of, make_inputs, rel, small_cfg

TR = dict(inputs=[0], order=2)
TRANSPORT_VARIANTS = {
    "inputs": dict(transport=dict(inputs=[0])),
    "inputs_order3": dict(transport=dict(inputs=[0, 1], order=3)),
    "cons_upwind": dict(transport=dict(inputs=[1], forms=["cons", "upwind"])),
    "adv_only": dict(transport=dict(inputs=[0], forms=["adv"], order=2)),
    "sample_norm": dict(transport=dict(inputs=[0], order=2, norm="sample")),
    "hidden": dict(transport=dict(hidden=2, order=2)),
    "latent": dict(latent_dims={1: 2}, transport=dict(latent=2, inputs=[0])),
    "resolvent": dict(layers=["resolvent", "poly"], transport=TR),
    "scaling_none": dict(scaling="none", transport=TR),
    "tensor": dict(metric_type="tensor", transport=TR),
    "untied": dict(tie_metrics=False, transport=dict(inputs=[0], hidden=1)),
    "star_tensor": dict(metric_type="tensor", transport=dict(inputs=[0], hidden=1, order=2, metric="star")),
}


def _ctx(K, B=1, dtype=F64, scaling="dec"):
    return make_context(K, scaling=scaling, B=B, dtype=dtype)


def _dense(g, w, n0, dtype=F64, device="cpu"):
    """Dense matrix of ``x -> apply_transport(g, w, x)`` (columns = images of the unit vectors)."""
    eye = torch.eye(n0, dtype=dtype, device=device).unsqueeze(1).contiguous()          # (n0, 1, n0)
    return apply_transport(g, w, eye)[:, 0, :]


def _flux(K, seed=0, device="cpu"):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(K.n[1], 1, 1, generator=g, dtype=F64).to(device)


def _built(cfg: RHMPConfig, K, seed: int = 0, dtype=None) -> RHMP:
    m = build_model(cfg, K, seed=seed, dtype=dtype)
    return m.init_latents(K)


# ----------------------------------------------------------------------------------------------------------------
# operators
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", ["delaunay", "sphere", "torus", "nonmanifold", "quads"])
def test_operator_identities_and_mass_conservation(name, device):
    """``A = C^T``, ``C + A = diag(s^2 d_0^T F)``, ``U`` symmetric PSD with zero row sums in the physical frame;
    ``sum_i s_i^{-1} (C x)_i = sum_i s_i^{-1} (U x)_i = 0`` (mass ``sum star_i u_i``), ``A`` does not conserve."""
    K = make_complex(name, device).to(dtype=F64)
    ctx = _ctx(K)
    g = ctx.blocks[(0, True)]
    F = _flux(K, device=device)
    one = torch.ones(1, dtype=F64, device=device)
    Cm = _dense(g, transport_weights(F, {"cons": one}), K.n[0], device=device)
    Am = _dense(g, transport_weights(F, {"adv": one}), K.n[0], device=device)
    Um = _dense(g, transport_weights(F, {"upwind": one}), K.n[0], device=device)
    s = g.scale.view(-1)
    divF = K.apply_dT(0, F.contiguous()).view(-1)                                    # d_0^T F
    assert rel(Am, Cm.T) < 1e-12
    assert rel(Cm + Am, torch.diag(s * s * divF)) < 1e-12
    assert rel(Um, Um.T) < 1e-12 and float(torch.linalg.eigvalsh(Um).min()) > -1e-10
    mass = 1.0 / s                                                                    # sum_i s_i^{-1} y_i
    x = torch.randn(K.n[0], generator=torch.Generator().manual_seed(1), dtype=F64).to(device)
    assert abs(float(mass @ (Cm @ x))) < 1e-10 * float((Cm @ x).abs().sum())
    assert abs(float(mass @ (Um @ x))) < 1e-10 * float((Um @ x).abs().sum())
    assert abs(float(mass @ (Am @ x))) > 1e-6 * float((Am @ x).abs().sum())
    # U annihilates the constants of the physical variable (u = 1, i.e. x = s^-1)
    assert float((Um @ (1.0 / s)).abs().max()) < 1e-10


@pytest.mark.parametrize("name", ["sphere", "torus", "delaunay"])
def test_divergence_free_flux_gives_skew_transport(name, device):
    """``F = d_1^T psi`` is divergence free on the dual cells (``d_0^T d_1^T = 0``): ``C`` is skew-symmetric."""
    K = make_complex(name, device).to(dtype=F64)
    g = _ctx(K).blocks[(0, True)]
    psi = torch.randn(K.n[2], 1, 1, generator=torch.Generator().manual_seed(2), dtype=F64).to(device)
    F = K.apply_dT(1, psi)
    Cm = _dense(g, transport_weights(F, {"cons": torch.ones(1, dtype=F64, device=device)}), K.n[0], device=device)
    assert float((Cm + Cm.T).norm() / Cm.norm()) < 1e-12


@pytest.mark.parametrize("name", ["delaunay", "sphere", "sliver", "nonmanifold", "quads"])
@pytest.mark.parametrize("norm", ["metric", "sample"])
def test_norm_bounds(name, norm, device):
    """``metric``: ``||C||, ||A||, ||U|| <= max |phi| / 2`` with the Gershgorin-normalised layer metric (random ``H``
    over ``e^{+-4}``); ``sample``: ``<= 1`` per field.  Checked against dense singular values."""
    K = make_complex(name, device).to(dtype=F64)
    ctx = _ctx(K)
    gen = torch.Generator().manual_seed(3)
    logH1 = (ctx.log_star[1].unsqueeze(1) + 4.0 * (2 * torch.rand(K.n[1], 1, generator=gen, dtype=F64) - 1).to(device))
    phi = 3.0 * torch.randn(K.n[1], 1, 2, generator=gen, dtype=F64).to(device)
    term = TransportTerm(dict(inputs=[0, 1], norm=norm)).to(device=device, dtype=F64)
    F = term.fluxes(phi, ctx, logH1)
    g = ctx.blocks[(0, True)]
    rb = transport_bound(g, F.abs())                                                # (n0, 1, J)
    for j in range(2):
        bound = (phi[..., j].abs().max() / 2) if norm == "metric" else torch.tensor(1.0, dtype=F64)
        assert float(rb[..., j].max()) <= float(bound) * (1 + 1e-12)
        for form in ("cons", "adv", "upwind"):
            M = _dense(g, transport_weights(F[..., j:j + 1], {form: torch.ones(1, dtype=F64, device=device)}),
                       K.n[0], device=device)
            assert float(torch.linalg.matrix_norm(M, ord=2)) <= float(rb[..., j].max()) * (1 + 1e-10), (form, j)


def test_star_metric_uses_the_reference_star(device):
    """``metric='star'``: the fluxes are ``star_1 phi`` normalised like the up block at ``H = star``, whatever the layer's
    learned metric (the advective flux stays geometric when the learned metric carries a diffusion tensor)."""
    K = make_complex("delaunay", device).to(dtype=F64)
    ctx = _ctx(K)
    phi = torch.randn(K.n[1], 1, 1, generator=torch.Generator().manual_seed(5), dtype=F64).to(device)
    star = ctx.log_star[1].unsqueeze(1)
    learned = star + torch.randn(K.n[1], 1, generator=torch.Generator().manual_seed(6), dtype=F64).to(device)
    t_star = TransportTerm(dict(inputs=[0], metric="star")).to(device=device, dtype=F64)
    t_layer = TransportTerm(dict(inputs=[0])).to(device=device, dtype=F64)
    assert rel(t_star.fluxes(phi, ctx, learned), t_layer.fluxes(phi, ctx, star)) < 1e-14
    assert rel(t_star.fluxes(phi, ctx, learned), t_layer.fluxes(phi, ctx, learned)) > 1e-3


def test_metric_norm_keeps_field_magnitude(device):
    """``norm='metric'``: doubling the field doubles the operator (the speed is kept); ``'sample'``: invariant."""
    K = make_complex("sphere", device).to(dtype=F64)
    ctx = _ctx(K)
    logH1 = ctx.log_star[1].unsqueeze(1)
    phi = torch.randn(K.n[1], 1, 1, generator=torch.Generator().manual_seed(4), dtype=F64).to(device)
    for norm, factor in (("metric", 2.0), ("sample", 1.0)):
        term = TransportTerm(dict(inputs=[0], norm=norm)).to(device=device, dtype=F64)
        assert rel(term.fluxes(2.0 * phi, ctx, logH1), factor * term.fluxes(phi, ctx, logH1)) < 1e-12


def test_matches_handwritten_upwind_finite_volumes(device):
    """``C - U`` with ``F = star_1 phi`` is the first-order upwind finite-volume step of ``d_t u + div(u v) = 0`` on
    the barycentric dual cells (cotan dual-edge fluxes of the trapezoidal velocity one-form), written as a loop."""
    pos, faces = _MESHES["grid"](nx=9, ny=8, jitter=0.2, seed=3)
    K = CochainComplex.from_triangles(pos, faces, device=device).to(dtype=F64)
    P = K.pos.cpu().numpy()
    vel = np.stack([1.0 + 0.3 * P[:, 1], 0.5 - 0.8 * P[:, 0]], 1)                     # a smooth non-uniform field
    e = K.cells[1].cpu().numpy()
    phi = 0.5 * ((vel[e[:, 0]] + vel[e[:, 1]]) * (P[e[:, 1]] - P[e[:, 0]])).sum(1)
    star0, star1 = K.star[0].cpu().numpy(), K.star[1].cpu().numpy()
    u = np.random.default_rng(5).standard_normal(K.n[0])
    rate = np.zeros(K.n[0])
    for (a, b), f in zip(e, star1 * phi):                                             # flux a -> b through the dual edge
        flux = f * (u[a] if f > 0 else u[b])
        rate[b] += flux
        rate[a] -= flux
    rate /= star0
    ctx = _ctx(K)
    g = ctx.blocks[(0, True)]
    F = torch.as_tensor(star1 * phi, dtype=F64, device=device).view(-1, 1, 1)
    w = transport_weights(F, {"cons": torch.ones(1, dtype=F64, device=device),
                              "upwind": -torch.ones(1, dtype=F64, device=device)})
    s = g.scale.view(-1, 1, 1)
    x = torch.as_tensor(u, dtype=F64, device=device).view(-1, 1, 1) / s                # symmetric frame x = s^-1 u
    m = apply_transport(g, w, x.contiguous())
    gm0 = float(torch.exp(ctx.log_gm[0]).item())                                      # star normalisation constant
    out = (s * m).view(-1).cpu().numpy() / gm0
    assert np.abs(out - rate).max() <= 1e-10 * np.abs(rate).max()


# ----------------------------------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------------------------------
def test_config_validation_and_roundtrip(tmp_path, device):
    t = normalize_transport({"inputs": [0], "forms": "cons+upwind"})
    assert t == dict(inputs=[0], latent=0, hidden=0, forms=["cons", "upwind"], order=1, norm="metric", metric="layer")
    assert normalize_transport(None) == {} and normalize_transport({}) == {}
    for bad in ({"bogus": 1}, {"inputs": [0], "order": 0}, {"inputs": [0], "norm": "x"}, {"forms": ["cons"]},
                {"inputs": [0], "forms": ["sideways"]}, {"hidden": 3}, {"inputs": [0, 0]}, {"inputs": [0], "metric": "x"}):
        with pytest.raises(ValueError):
            normalize_transport(bad)
    base = dict(in_dims={0: 1, 1: 3}, even_dims={1: 1}, connection_dims={1: 1}, C=8, n_layers=2)
    for bad in ({"inputs": [0]}, {"inputs": [2]}, {"inputs": [5]}, {"latent": 1}):      # connection / even / range
        with pytest.raises(ValueError):
            RHMPConfig(**base, transport=bad)
    with pytest.raises(ValueError):
        RHMPConfig(**base, transport={"inputs": [1]}, scaling="jacobi")
    cfg = RHMPConfig(**base, latent_dims={1: 2}, transport={"inputs": [1], "latent": 2, "order": 2})
    assert transport_columns(cfg) == [1, 2, 3]                                       # latent inserted before even
    d = cfg.to_dict()
    assert RHMPConfig.from_dict(json.loads(json.dumps(d))) == cfg
    old = {k: v for k, v in d.items() if k != "transport"}                            # a config saved before
    assert RHMPConfig.from_dict(old).transport == {}
    K = make_complex("sphere", device)
    m = _built(cfg, K)
    inp = make_inputs(cfg, K)
    y = m(inp, K)
    path = tmp_path / "m.pt"
    torch.save(m.to_checkpoint(), path)
    m2 = RHMP.from_checkpoint(str(path), map_location=device)
    assert rel(m2(inp, K), y) < 1e-6


def test_default_off_reproduces_existing_models(device):
    """Without ``transport`` the state dict has no new entries; with zero coefficients the transport model equals
    the model without it (same weights)."""
    K = make_complex("delaunay", device)
    cfg0 = small_cfg(n_layers=3)
    m0 = build_model(cfg0, K)
    assert not any("transport" in k for k in m0.state_dict())
    m1 = RHMP(dataclasses.replace(cfg0, transport=dict(inputs=[0], hidden=2, order=3)), K.geo_dims).to(device)
    missing, unexpected = m1.load_state_dict(m0.state_dict(), strict=False)
    assert not unexpected and missing and all("transport" in k for k in missing)
    inp = make_inputs(cfg0, K)
    assert rel(m1(inp, K), m0(inp, K)) < 1e-7
    with pytest.raises(ValueError):
        RHMP(dataclasses.replace(cfg0, layers=["solve"], transport=dict(inputs=[0]), scaling="jacobi"), K.geo_dims)


# ----------------------------------------------------------------------------------------------------------------
# symmetries of the model with the transport term
# ----------------------------------------------------------------------------------------------------------------
def _check_OC(K, cfg, dtype, tol, seed=0):
    m = _built(cfg, K, seed=seed, dtype=dtype)
    inp = make_inputs(cfg, K, B=2, seed=seed, dtype=dtype)
    x = m.lift(inp, K)
    for reflect in (False, True):
        Q = torch.as_tensor(random_orthogonal(cfg.C, seed=seed + 1, reflect=reflect), dtype=dtype, device=K.pos.device)
        a = m.propagate({k: v @ Q for k, v in x.items()}, K, inp)
        b = m.propagate(x, K, inp)
        for k in b:
            assert rel(a[k], b[k] @ Q) <= tol, (k, reflect, rel(a[k], b[k] @ Q))
    m0 = _built(dataclasses.replace(cfg, transport={}), K, seed=seed, dtype=dtype)   # the term changes the result
    m0.load_state_dict(m.state_dict(), strict=False)
    assert rel(m0.propagate(x, K, inp)[0], m.propagate(x, K, inp)[0]) > 1e-4


@pytest.mark.parametrize("variant", list(TRANSPORT_VARIANTS))
def test_OC_equivariance_with_transport(variant, device):
    K = make_complex("delaunay", device).to(dtype=F64)
    _check_OC(K, small_cfg(n_layers=2, **TRANSPORT_VARIANTS[variant]), F64, 1e-10)


@pytest.mark.parametrize("name", ["sphere", "torus"])
def test_OC_equivariance_with_transport_fp32(name, device):
    K = make_complex(name, device)
    _check_OC(K, small_cfg(n_layers=3, transport=dict(inputs=[0], hidden=1, order=2)), torch.float32, 1e-5)


@pytest.mark.parametrize("name,variant", [("delaunay", "inputs_order3"), ("sphere", "hidden"), ("torus", "sample_norm"),
                                          ("nonmanifold", "cons_upwind"), ("sphere", "tensor"),
                                          ("delaunay", "resolvent"), ("mixed", "inputs")])
def test_vertex_relabelling_with_transport(name, variant, device):
    """Random vertex permutation, complex rebuilt (canonical edge orientations change): the transport fields flip
    with their edges and the outputs are permuted (odd outputs pick up the orientation signs)."""
    pos, cells = _MESHES[name]()
    pos2, cells2, perm = relabel_mesh(pos, cells, seed=1)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    K, K2 = _build(name, pos, cells, device), _build(name, pos2, cells2, device)
    maps = _cell_maps(K, K2, inv)
    assert bool((maps[1][1] < 0).any())
    permt = torch.as_tensor(perm, device=device)
    for readout in ("node_scalar", "cochain:1", "grad"):
        cfg = small_cfg(readout=readout, out_dim=2, **TRANSPORT_VARIANTS[variant])
        m = build_model(cfg, K)
        inp = make_inputs(cfg, K)
        inp2 = {0: inp[0][permt]}
        for k, v in inp.items():
            if k > 0:
                inp2[k] = _map_cells(v, *maps[k], n_odd=cfg.in_dims[k] - cfg.even_dims.get(k, 0))
        y, y2 = m(inp, K), m(inp2, K2)
        expect = y[permt] if readout == "node_scalar" else _map_cells(y, *maps[1], n_odd=y.shape[-1])
        assert rel(y2, expect) < 1e-5, (readout, rel(y2, expect))


def test_vertex_relabelling_with_latent_transport_field(device):
    """A latent edge column used as the transport field moves with the mesh like an odd input."""
    pos, cells = _MESHES["delaunay"]()
    pos2, cells2, perm = relabel_mesh(pos, cells, seed=1)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    K, K2 = _build("delaunay", pos, cells, device), _build("delaunay", pos2, cells2, device)
    maps = _cell_maps(K, K2, inv)
    permt = torch.as_tensor(perm, device=device)
    cfg = small_cfg(**TRANSPORT_VARIANTS["latent"])
    m = _built(cfg, K)
    m2 = build_model(cfg, K2, randomize=False)
    m2.load_state_dict(m.state_dict())
    with torch.no_grad():
        m2.latent["1"].copy_(_map_cells(m.latent["1"].unsqueeze(1), *maps[1], n_odd=99).squeeze(1))
    inp = make_inputs(cfg, K)
    inp2 = {0: inp[0][permt]}
    for k, v in inp.items():
        if k > 0:
            inp2[k] = _map_cells(v, *maps[k], n_odd=cfg.in_dims[k] - cfg.even_dims.get(k, 0))
    assert rel(m2(inp2, K2), m(inp, K)[permt]) < 1e-5


@pytest.mark.parametrize("name", ["grid", "sphere"])
def test_face_orientation_flip_with_transport(name, device):
    """Reversing some faces flips odd face inputs / outputs only (the transport term lives on vertices and edges)."""
    pos, faces = _MESHES[name]()
    flip = np.random.default_rng(7).random(len(faces)) < 0.4
    faces2 = faces.copy()
    faces2[flip] = faces[flip][:, ::-1]
    K, K2 = _build(name, pos, faces, device), _build(name, pos, faces2, device)
    sign = torch.as_tensor(np.where(flip, -1.0, 1.0), dtype=torch.float32, device=device)
    idx = torch.arange(len(faces), device=device)
    for readout in ("node_scalar", "cochain:2"):
        cfg = small_cfg(readout=readout, out_dim=2, transport=dict(inputs=[0], hidden=2, order=2))
        m = build_model(cfg, K)
        inp = make_inputs(cfg, K)
        inp2 = {**inp, 2: _map_cells(inp[2], idx, sign, n_odd=cfg.in_dims[2] - cfg.even_dims.get(2, 0))}
        y, y2 = m(inp, K), m(inp2, K2)
        expect = _map_cells(y, idx, sign, n_odd=y.shape[-1]) if readout == "cochain:2" else y
        assert rel(y2, expect) < 1e-5, (readout, rel(y2, expect))


@pytest.mark.parametrize("name", ["delaunay", "sphere", "torus"])
@pytest.mark.parametrize("reflect", [False, True])
def test_rigid_motion_invariance_with_transport(name, reflect, device):
    """Rotations (+ reflections) and translations of the positions leave the outputs unchanged (the velocity
    one-form is invariant, and only lengths, areas and angles enter): exactly in fp64 on the same complex, and in
    fp32 with the complex rebuilt from the transformed positions."""
    K = make_complex(name, device).to(dtype=F64)
    D = K.pos.shape[1]
    R = torch.as_tensor(random_orthogonal(D, seed=6, reflect=reflect), dtype=F64, device=device)
    t = torch.randn(D, dtype=F64, generator=torch.Generator().manual_seed(8)).to(device) * 3.0
    K2 = dataclasses.replace(K, pos=K.pos @ R.T + t)
    cfg = small_cfg(transport=dict(inputs=[0, 1], hidden=2, order=2))
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K, dtype=F64)
    assert rel(m(inp, K2), m(inp, K)) < 1e-10
    pos, cells = _MESHES[name]()
    Kf, Kf2 = _build(name, pos, cells, device), _build(name, pos @ R.cpu().numpy().T + t.cpu().numpy(), cells, device)
    mf = build_model(cfg, Kf)
    inpf = make_inputs(cfg, Kf)
    assert rel(mf(inpf, Kf2), mf(inpf, Kf)) < 1e-5


@pytest.mark.parametrize("variant", ["inputs_order3", "sample_norm", "hidden"])
def test_batch_independence_with_transport(variant, device):
    K = make_complex("delaunay", device)
    cfg = small_cfg(n_layers=3, **TRANSPORT_VARIANTS[variant])
    m = build_model(cfg, K)
    inp = make_inputs(cfg, K, B=16, seed=0)
    other = make_inputs(cfg, K, B=16, seed=1)
    alone = m({k: v[:, :1].contiguous() for k, v in inp.items()}, K)
    assert rel(m(inp, K)[:, :1], alone) < 1e-6
    assert rel(m({k: torch.cat([inp[k][:, :1], other[k][:, 1:]], 1) for k in inp}, K)[:, :1], alone) < 1e-6
    Kb, parts = make_batch(device)
    ins = [make_inputs(cfg, P, B=1, seed=i) for i, P in enumerate(parts)]
    yb = m({k: torch.cat([x[k] for x in ins], 0) for k in ins[0]}, Kb)
    ptr = Kb.meta["ptr"][0].tolist()
    for i, (P, x) in enumerate(zip(parts, ins)):
        assert rel(yb[ptr[i]:ptr[i + 1]], m(x, P)) < 1e-6


def test_transport_message_conserves_mass_in_the_model(device):
    """The vertex message of a conservative transport term (forms ``cons`` + ``upwind``) has zero mass
    ``sum_i star_i^{1/2} m_i`` per sample, also on a block-diagonal batch (per graph)."""
    for K in (make_complex("torus", device).to(dtype=F64), make_batch(device)[0].to(dtype=F64)):
        cfg = small_cfg(n_layers=1, transport=dict(inputs=[0], hidden=1, forms=["cons", "upwind"], order=3))
        m = build_model(cfg, K, dtype=F64)
        inp = make_inputs(cfg, K, B=3, dtype=F64)
        inp_p, ctx = m._prepare(inp, K)
        x = m.lifting(inp_p, ctx)
        layer = m.layers[0]
        from rhmp.layers import SharedCoboundaries
        q = SharedCoboundaries(x, ctx)
        logH_up, _ = layer.metrics(x, q, ctx, [1], [])
        msg = layer.transport(x, ctx, logH_up[1], None, q.up(0))
        w = (1.0 / ctx.blocks[(0, True)].scale).view(-1, 1, 1) * msg                     # star^{1/2} m
        tot = w.sum(-1)                                                                 # per channel summed
        per = tot.sum(0) if K.batch is None else torch.zeros(K.num_graphs, tot.shape[1], dtype=F64,
                                                               device=device).index_add_(0, K.batch[0], tot)
        assert float(per.detach().abs().max()) < 1e-10 * float(w.detach().abs().sum())


def test_gradients_reach_the_transport_coefficients(device):
    K = make_complex("sphere", device)
    cfg = small_cfg(transport=dict(inputs=[0], hidden=1, order=2))
    m = RHMP(cfg, K.geo_dims).to(device)                                              # zero-initialised coefficients
    loss_of(m(make_inputs(cfg, K), K)).backward()
    g = {n: p.grad for n, p in m.named_parameters() if "transport.coef" in n}
    assert g and all(v is not None and float(v.abs().max()) > 0 for v in g.values())


# ----------------------------------------------------------------------------------------------------------------
# T2v edge encoding and trainer flag
# ----------------------------------------------------------------------------------------------------------------
def test_t2v_edge_encoding_is_lossless_and_has_the_right_parity(device):
    """The odd one-form and the even half-difference determine the tangent vertex velocities exactly; under vertex
    relabelling the one-form flips with the edge orientation and the half-difference does not."""
    from rhmp.tasks.paper import _t2v_reconstruction_error, t2v_inputs
    pos, faces = _MESHES["torus"](n_major=14, n_minor=9)
    K = CochainComplex.from_triangles(pos, faces, device=device)
    P = K.pos.double().cpu()
    F = K.cells[2].long().cpu()
    nrm = torch.zeros_like(P).index_add_(0, F.reshape(-1), torch.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]],
                                                                        dim=1).repeat_interleave(3, 0))
    nrm = nrm / nrm.norm(dim=1, keepdim=True)
    V = torch.randn(3, K.n[0], 3, generator=torch.Generator().manual_seed(0), dtype=F64)
    V = (V - (V * nrm).sum(-1, keepdim=True) * nrm).float()
    assert _t2v_reconstruction_error(V, K, n=3) < 1e-5
    X = torch.randn(3, K.n[0], 1)
    E = t2v_inputs(X, V, K, True)[1]
    pos2, faces2, perm = relabel_mesh(pos, faces, seed=2)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(perm.size)
    K2 = CochainComplex.from_triangles(pos2, faces2, device=device)
    idx, sign = _cell_maps(K, K2, inv)[1]
    E2 = t2v_inputs(X[:, perm], V[:, perm], K2, True)[1]
    expect = torch.empty_like(E)
    expect[:, idx.cpu()] = E * torch.stack([sign.cpu(), torch.ones_like(sign.cpu())], -1)
    assert rel(E2, expect) < 1e-6
    assert t2v_inputs(X, V, K, False)[0].shape == (3, K.n[0], 4)


def test_trainer_transport_flag(tmp_path):
    from test_train import _args, tiny_shared_task
    from rhmp.train import run
    torch.set_num_threads(2)
    res = run(_args(tmp_path, "--batch", "8", "--transport", '{"inputs": [0], "hidden": 1, "order": 2}'),
              task=tiny_shared_task())
    assert res["complete"] and res["config"]["transport"]["order"] == 2
    assert any(k.endswith("transport_cons_absmax") for k in res["diagnostics"]["all"])
    again = run(_args(tmp_path, "--batch", "8", "--transport", '{"inputs": [0], "hidden": 1, "order": 2}'),
                task=tiny_shared_task())
    assert again["test"]["R2"] == res["test"]["R2"]                                   # resume / skip with the flag


def test_t2v_task_on_the_stored_data():
    """T2v (needs datasets/T2_torus_advection_diffusion.pkl and datasets/v2/T2v.pt): same targets and split as T2,
    native edge encoding lossless, legacy node features ``(f, Vx, Vy, Vz)``, generator reproduction recorded."""
    import os

    from rhmp.tasks import load_task, resolve_root
    droot = resolve_root(None)
    if not (os.path.exists(os.path.join(droot, "v2", "T2v.pt"))
            and os.path.exists(os.path.join(droot, "T2_torus_advection_diffusion.pkl"))):
        pytest.skip("T2 / T2v data not available")
    nat = load_task("T2v", native=True, device="cpu", whitney=False)
    leg = load_task("T2v", native=False, device="cpu", whitney=False)
    t2 = load_task("T2", native=True, device="cpu", whitney=False)
    assert nat.in_dims == {0: 1, 1: 2} and nat.even_dims == {1: 1} and leg.in_dims == {0: 4}
    for td in (nat, leg):
        assert torch.equal(td.target, t2.target) and all(torch.equal(a, b) for a, b in zip(td.split, t2.split))
        assert torch.allclose(td.inputs[0][..., :1], t2.inputs[0], rtol=0, atol=1e-5)
    c = nat.meta["consistency"]
    assert c["edge_encoding_reconstruction_max_rel_err"] < 1e-5 and c["target_regeneration"]["maxabs"] < 1e-6


def test_t2a_task_on_the_stored_data():
    """T2a (needs datasets/v2/T2a.pt): native and legacy layouts, generator edge data aligned with the complex, the
    reference's checks (mass conservation, time-step convergence, isotropic limit) recorded in the file."""
    import os

    from rhmp.tasks import load_task, resolve_root
    from rhmp.tasks.torus import t2a_file
    droot = resolve_root(None)
    if not os.path.exists(os.path.join(droot, "v2", "T2a.pt")):
        pytest.skip("T2a data not available")
    nat = load_task("T2a", native=True, device="cpu", whitney=False)
    leg = load_task("T2a", native=False, device="cpu", whitney=False)
    assert nat.in_dims == {0: 1, 1: 4, 2: 2} and nat.even_dims == {1: 3, 2: 2} and leg.in_dims == {0: 9}
    assert torch.equal(nat.target, leg.target) and all(torch.equal(a, b) for a, b in zip(nat.split, leg.split))
    blob = t2a_file(droot)
    raw = nat.x_stats[1].denormalize(nat.inputs[1][:4])                         # cos^2 in [0, 1] on every edge
    assert float(raw[..., 2].min()) >= -1e-6 and float(raw[..., 2].max()) <= 1 + 1e-6
    chk = blob["meta"]["checks"]
    assert blob["meta"]["mass_drift_max"] < 1e-12 and chk["time_error_vs_800_at_N_SUB"] < 1e-3
    assert chk["isotropic_limit_rel"] < 1e-12 and chk["flux"]["div_max_rel"] < 1e-12
