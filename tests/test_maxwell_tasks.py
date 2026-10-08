"""Tests of the nonlinear magnetostatics task family (``rhmp/tasks/maxwell.py``, ``datasets/generators/gen_maxwell.py``)
and of the fixed-point (Picard) solve option of solve layers (``RHMPConfig.solve_picard``).

Without data:
* the material law of the task module is the generator's (explicit inverse of the Froehlich-Kennelly law);
* the generator's conforming mesh is valid and its damped Newton solve converges to the energy minimiser;
* the fixed output map ``FluxToNodeField`` equals an independent P1 computation, is invariant under vertex
  relabelling and face orientation flips, equivariant under rotations, and works on block-diagonal batches;
* Picard layers: the default configuration is unchanged (same parameters); with a frozen metric the fixed-point
  iteration equals one solve; **oracle**: the true reluctivity of a Newton solution, supplied as a frozen face
  reference, reproduces the finite-element potential to solver tolerance, and the field-energy invariant is |B|^2;
  O(C) equivariance, E(n) invariance, vertex relabelling, face orientation flips and batch independence with random
  (learned) metric heads; implicit gradients through the iterations agree with finite differences; trainer smoke run
  and checkpoint round trip;
* ``solve_bc='lift'`` (inhomogeneous Dirichlet data): on the ECT electrostatics family the solve layer with the true
  permittivity reproduces the FEM potential and keeps the electrode potentials exactly;
* the v1 edge protocol transforms extra test sets; rotated test views keep a complex-dependent output map.
Anisotropic family MXA (``gen_maxwell_aniso.py``): the energy-based law (gradient, symmetric convex tangent, secant
tensor on the rolling direction, principal curves, linear limit, rotation covariance; the literal two-curve law is not
monotone), the lattice mesh, Newton at the energy minimum, rotation covariance of the discrete problem, Picard =
Newton when unsaturated, the oracle (the frozen secant operator is the model's Galerkin assembly of the gradient
metric), E(n)-invariant edge inputs and the rotation of the baselines' frame columns.
With the generated data (skipped when ``datasets/v2/MXS.pt`` / ``MXE.pt`` / ``MXA.pt`` are absent): the loaders'
shapes, the exactness of the output maps on the stored targets and the legacy node inputs.
"""
from __future__ import annotations

import math
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

import gen_maxwell as G  # noqa: E402
from conftest import make_batch, make_complex, random_orthogonal  # noqa: E402
from rhmp import RHMP, RHMPConfig  # noqa: E402
from rhmp.complex import CochainComplex  # noqa: E402
from rhmp.tasks.maxwell import FluxToNodeField, reluctivity  # noqa: E402
from test_model import F64, loss_of, randomize_, rel  # noqa: E402

DATA = os.path.join(ROOT, "datasets", "v2")


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------
def _square_problem(n: int = 10, L_mm: float = 100.0, J: float = 200.0, seed: int = 0):
    """Small nonlinear magnetostatic problem on a structured square: an iron block, a +J and a -J coil block, A = 0 on
    the boundary; solved by the generator's damped Newton.  Returns (P1Mesh, phase, J_face, solution, pos_mm, faces)."""
    pos, F, bnd = G._structured_square(n, L_mm)
    cent = pos[F].mean(1) / L_mm
    phase = np.zeros(len(F), dtype=np.int8)
    phase[(cent[:, 0] > 0.3) & (cent[:, 0] < 0.7) & (cent[:, 1] > 0.25) & (cent[:, 1] < 0.6)] = G.PHASE_IRON
    Jf = np.zeros(len(F))
    Jf[(cent[:, 0] < 0.3) & (cent[:, 0] > 0.1) & (cent[:, 1] > 0.3) & (cent[:, 1] < 0.7)] = J
    Jf[(cent[:, 0] > 0.7) & (cent[:, 0] < 0.9) & (cent[:, 1] > 0.3) & (cent[:, 1] < 0.7)] = -J
    m = G.P1Mesh(pos, F, bnd)
    sol = G.newton_solve(m, phase, Jf, tol=1e-13)
    return m, phase, Jf, sol, pos, F


def _identity_solver(m: RHMP) -> RHMP:
    """Unit lifting / readout / gate weights and lam -> 0: the solver-preset model is then exactly the solve."""
    with torch.no_grad():
        m.lifting.lin0.weight.fill_(1.0)
        m.readout.lin.weight.fill_(1.0)
        for layer in m.layers:
            for g in layer.gates:
                g.gain.fill_(1.0)
            for p in layer.log_lam.values():
                p.fill_(-40.0)
    return m


def _p1_grad(K, A: torch.Tensor) -> torch.Tensor:
    """Independent P1 element gradients ``(n2, B, 2)`` of vertex fields ``A (n0, B)`` (planar complexes)."""
    P = K.pos.double()[K.cells[2].long()]                                   # (n2, 3, 2)
    Fc = K.cells[2].long()
    J = torch.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], 2)              # columns e1, e2
    Jinv = torch.linalg.inv(J)                                              # rows: grad lambda_1, grad lambda_2
    d1 = A[Fc[:, 1]] - A[Fc[:, 0]]
    d2 = A[Fc[:, 2]] - A[Fc[:, 0]]
    return d1.unsqueeze(-1) * Jinv[:, None, 0, :] + d2.unsqueeze(-1) * Jinv[:, None, 1, :]


def _node_field_reference(K, A: torch.Tensor) -> torch.Tensor:
    """Area-weighted vertex mean of R(-90 deg) grad A, ``(n0, B, 2)`` (independent of FluxToNodeField)."""
    g = _p1_grad(K, A)
    B = torch.stack([g[..., 1], -g[..., 0]], -1)
    P = K.pos.double()[K.cells[2].long()]
    area = 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) -
                  (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0])).abs()
    num = torch.zeros(K.n[0], *B.shape[1:], dtype=F64, device=A.device)
    den = torch.zeros(K.n[0], dtype=F64, device=A.device)
    for k in range(3):
        v = K.cells[2][:, k].long()
        num.index_add_(0, v, B * area[:, None, None])
        den.index_add_(0, v, area)
    return num / den[:, None, None]


def _picard_cfg(metric_type: str = "tensor", J: int = 3, **kw) -> RHMPConfig:
    base = dict(in_dims={0: 1, 1: 2, 2: 2}, even_dims={1: 2, 2: 2}, material_dims={1: 2, 2: 2}, readout="grad",
                metric_type=metric_type, log_range=3.0, solve_picard=J, solve_iters=3000, solve_tol=1e-13, C=3)
    base.update(kw)
    return RHMPConfig.solver_preset(**base)


def _picard_inputs(K, B: int = 2, seed: int = 0, dtype=F64) -> dict:
    g = torch.Generator().manual_seed(seed)
    out = {0: torch.randn(K.n[0], B, 1, generator=g, dtype=F64),
           1: torch.rand(K.n[1], B, 2, generator=g, dtype=F64), 2: torch.rand(K.n[2], B, 2, generator=g, dtype=F64)}
    return {k: v.to(device=K.pos.device, dtype=dtype) for k, v in out.items()}


# ----------------------------------------------------------------------------------------------------------------
# generator
# ----------------------------------------------------------------------------------------------------------------
def test_law_is_the_generators():
    B = np.linspace(0.0, 4.0, 401)
    assert np.allclose(reluctivity(B), G.nu_iron(B), rtol=1e-14, atol=0.0)
    assert torch.allclose(reluctivity(torch.as_tensor(B)), torch.as_tensor(G.nu_iron(B)), rtol=1e-12)
    assert abs(G.NU0 / G.nu_iron(np.zeros(1))[0] - 3000.0) < 1e-6
    H = np.logspace(-1, 6, 200)
    assert np.allclose(G.h_of_b(G.b_of_h(H)), H, rtol=1e-10)              # explicit inverse
    assert np.all(np.diff(G.nu_iron(B)) > 0)                               # monotone saturation


def test_generator_mesh_and_newton():
    """Coarse device mesh: valid conforming triangulation (fixed regions exact, box boundary), staircase phases, a
    converged Newton solve that minimises the energy, flux linkage of zero-net-current windings."""
    mesh = G.build_mesh(2.0, 13)
    st = mesh["stats"]
    assert st["euler"] == 1 and st["unused"] == 0 and st["nonmanifold"] == 0 and st["min_angle"] > 10.0
    p = G.draw_params(11)
    ph = G.face_phase(mesh, p)
    assert (ph[mesh["region"] == G.REGION_CORE] == G.PHASE_IRON).all()
    assert (ph[(mesh["region"] >= 1) & (mesh["region"] <= 6)] == G.PHASE_COPPER).all()
    assert ((ph == G.PHASE_IRON) & (mesh["region"] == G.REGION_AIR)).sum() > 10        # movable iron present
    m = G.P1Mesh(mesh["pos"], mesh["faces"], mesh["boundary"])
    J = G.face_current(mesh, p["currents"])
    assert abs((J * m.area).sum()) < 1e-9 * np.abs(J * m.area).sum()                     # zero net current
    sol = G.newton_solve(m, ph, J)
    assert sol["res"] <= 1e-10 and sol["it"] <= 30
    Fv = m.load(J * G.AMM2)
    E0 = G._energy(m, ph, Fv, sol["A"])
    rng = np.random.default_rng(0)
    for _ in range(5):
        dA = np.zeros(m.n0)
        dA[m.free] = rng.normal(size=len(m.free)) * 1e-3 * np.abs(sol["A"]).max()
        assert G._energy(m, ph, Fv, sol["A"] + dA) > E0
    Bn = G.node_field(m, sol["Bf"])
    K = CochainComplex.from_triangles(torch.as_tensor(mesh["pos"]), torch.as_tensor(mesh["faces"]))
    ref = _node_field_reference(K, torch.as_tensor(sol["A"])[:, None] * 1e3)[:, 0]      # mm -> m
    assert rel(ref, torch.as_tensor(Bn)) < 1e-6                                          # K.pos is float32


def test_generator_linear_limit():
    m, phase, Jf, _, _, _ = _square_problem()
    small = G.newton_solve(m, phase, Jf * 1e-5, tol=1e-13)
    lin = G.newton_solve(m, phase, Jf, linear=True)
    assert np.linalg.norm(small["A"] * 1e5 - lin["A"]) / np.linalg.norm(lin["A"]) < 1e-5


# ----------------------------------------------------------------------------------------------------------------
# output map
# ----------------------------------------------------------------------------------------------------------------
def test_flux_to_node_field_exact_and_symmetric(device):
    K = make_complex("delaunay", device, n=80, seed=3)
    g = torch.Generator().manual_seed(0)
    A = torch.randn(K.n[0], 3, generator=g, dtype=F64).to(device)
    omap = FluxToNodeField(base=K, scale=1.0)
    Phi = K.to(dtype=F64).apply_d(0, A.unsqueeze(-1).contiguous()).float()
    for c in range(3):
        y = omap(Phi[:, c:c + 1].contiguous(), K)
        assert y.shape == (K.n[0], 1, 2)
        assert rel(y[:, 0], _node_field_reference(K, A[:, c:c + 1])[:, 0]) < 1e-5
    # vertex relabelling and face orientation flips leave the field unchanged (permuted)
    pos, faces = K.pos.double().cpu().numpy(), K.cells[2].cpu().numpy()
    rng = np.random.default_rng(1)
    perm = rng.permutation(K.n[0])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    f2 = inv[faces]
    f2[::2] = f2[::2][:, [0, 2, 1]]
    K2 = CochainComplex.from_triangles(torch.as_tensor(pos[perm]), torch.as_tensor(f2), device=device)
    A2 = A[torch.as_tensor(perm, device=device)]
    y2 = omap(K2.to(dtype=F64).apply_d(0, A2[:, :1].contiguous().unsqueeze(-1)).float(), K2)
    y1 = omap(Phi[:, :1].contiguous(), K)
    assert rel(y2, y1[torch.as_tensor(perm, device=device)]) < 1e-5
    # rotation equivariance (B is a pseudo-vector: proper rotations act as on vectors)
    Q = torch.as_tensor(random_orthogonal(2, seed=4), dtype=torch.float32, device=device)
    K3 = CochainComplex.from_triangles(K.pos.double() @ Q.double().T, K.cells[2], device=device)
    y3 = omap(Phi[:, :1].contiguous(), K3)
    assert rel(y3, y1 @ Q.T) < 1e-5
    # block-diagonal batches: per-part maps
    Kb, parts = make_batch(device)
    Ab = [torch.randn(P.n[0], 1, generator=g, dtype=F64).to(device) for P in parts]
    yb = omap(Kb.to(dtype=F64).apply_d(0, torch.cat(Ab).unsqueeze(-1).contiguous()).float(), Kb)
    ptr = Kb.meta["ptr"][0].tolist()
    for i, P in enumerate(parts):
        yi = omap(P.to(dtype=F64).apply_d(0, Ab[i].unsqueeze(-1).contiguous()).float(), P)
        assert rel(yb[ptr[i]:ptr[i + 1]], yi) < 1e-5


# ----------------------------------------------------------------------------------------------------------------
# Picard solve layers
# ----------------------------------------------------------------------------------------------------------------
def test_picard_default_is_unchanged():
    """``solve_picard = 0`` (the default) builds exactly the parameters of a solve layer as before; configurations
    saved without the new keys load."""
    cfg = _picard_cfg(J=0)
    d = cfg.to_dict()
    d.pop("solve_picard")
    d.pop("picard_relax")
    cfg_old = RHMPConfig.from_dict(d)
    assert cfg_old.solve_picard == 0 and cfg_old == cfg
    K = make_complex("delaunay")
    keys0 = set(RHMP(cfg_old, K.geo_dims).state_dict())
    keys3 = set(RHMP(_picard_cfg(J=3), K.geo_dims).state_dict())
    assert not any("picard" in k for k in keys0)
    assert keys3 - keys0 == {"layers.0.picard_logit", "layers.0.picard_log_e0"}
    with pytest.raises(ValueError):
        RHMPConfig(in_dims={0: 1}, solve_picard=2)                                     # no solve layer


@pytest.mark.parametrize("metric_type", ["tensor", "diag"])
def test_picard_learned_init(metric_type):
    """``picard_init='learned'`` adds one parameter (the uniform starting invariant).  At its initial value the
    iteration matches the zero start to O(softplus(-8)); a larger value changes the first metric; gradients reach it;
    the start is a uniform scalar, so the O(C) equivariance of the stack is kept."""
    K0 = make_complex("delaunay", n=60, seed=2)
    K = CochainComplex.from_triangles(K0.pos.double(), K0.cells[2]).to(dtype=F64)
    cfg0 = _picard_cfg(metric_type, J=2, lifting="mlp", readout="cochain:0", C=3)
    cfg1 = _picard_cfg(metric_type, J=2, lifting="mlp", readout="cochain:0", C=3, picard_init="learned")
    torch.manual_seed(0)
    m0 = randomize_(RHMP(cfg0, K.geo_dims), scale=0.4).to(dtype=F64)
    m1 = RHMP(cfg1, K.geo_dims).to(dtype=F64)
    extra = set(m1.state_dict()) - set(m0.state_dict())
    assert extra == {"layers.0.picard_f0_raw"} and RHMPConfig.from_dict(cfg1.to_dict()) == cfg1
    m1.load_state_dict({**m0.state_dict(), "layers.0.picard_f0_raw": m1.state_dict()["layers.0.picard_f0_raw"]})
    inp = _picard_inputs(K, B=2)
    y0, y1 = m0(inp, K), m1(inp, K)
    assert rel(y1, y0) < 1e-2
    with torch.no_grad():
        m1.layers[0].picard_f0_raw.fill_(1.0)
    y2 = m1(inp, K)
    assert rel(y2, y0) > 1e-3
    loss_of(y2).backward()
    assert m1.layers[0].picard_f0_raw.grad is not None and float(m1.layers[0].picard_f0_raw.grad.abs()) > 0
    with torch.no_grad():                                                              # O(C) of the stack
        x = m1.lift(inp, K)
        Q = torch.as_tensor(random_orthogonal(3, seed=4, reflect=True), dtype=F64)
        a_ = m1.propagate({k: v @ Q for k, v in x.items()}, K, inp)
        b_ = m1.propagate(x, K, inp)
    assert rel(a_[0], b_[0] @ Q) < 1e-9
    with pytest.raises(ValueError):
        RHMPConfig.solver_preset(in_dims={0: 1}, readout="grad", solve_picard=2, picard_init="warm")


@pytest.mark.parametrize("metric_type", ["tensor", "diag"])
def test_picard_frozen_metric_equals_one_solve(metric_type, device):
    """With a frozen (state-independent) metric every fixed-point iterate is the solve itself."""
    K = make_complex("delaunay", device, n=60, seed=2).to(dtype=F64)
    inp = _picard_inputs(K)
    ys = []
    for J in (0, 1, 4):
        cfg = _picard_cfg(metric_type, J=J, learn_metric=False, metric_reference={2: 1})
        m = _identity_solver(RHMP(cfg, K.geo_dims).to(device=device, dtype=F64))
        ys.append(m(inp, K))
    assert rel(ys[1], ys[0]) < 1e-10 and rel(ys[2], ys[0]) < 1e-10


def test_picard_oracle_true_reluctivity(device):
    """Oracle: the reluctivity of a converged Newton solution, supplied as a frozen face reference (tensor metric,
    ``learn_metric=False``), makes the (Picard) solve layer reproduce the nonlinear finite-element potential to solver
    tolerance (the secant system ``K(nu(B*)) A* = F`` holds at the solution; the frozen Galerkin blocks are stored in
    fp32, which limits the agreement to ~1e-8); the field-energy invariant of A* is |B|^2 of the element fields."""
    m, phase, Jf, sol, pos, F = _square_problem(n=8, L_mm=250.0, J=120.0)    # dyadic coordinates (exact in fp32)
    assert float(np.linalg.norm(sol["Bf"], axis=1)[phase == G.PHASE_IRON].max()) > 1.8      # saturated
    K = CochainComplex.from_triangles(torch.as_tensor(pos * 1e-3), torch.as_tensor(F), device=device).to(dtype=F64)
    area = torch.as_tensor(m.area, dtype=F64)
    Fv = torch.zeros(K.n[0], dtype=F64).index_add_(0, torch.as_tensor(F).reshape(-1),
                                                     (torch.as_tensor(Jf * G.AMM2) * area / 3).repeat_interleave(3))
    mass = torch.zeros(K.n[0], dtype=F64).index_add_(0, torch.as_tensor(F).reshape(-1), (area / 3).repeat_interleave(3))
    J_node = (Fv / mass).to(device)                                                       # M J = F
    lognu = torch.log(torch.as_tensor(sol["nu"], dtype=F64)).to(device)
    inp = {0: J_node.view(-1, 1, 1), 2: lognu.view(-1, 1, 1)}
    A_true = torch.as_tensor(sol["A"], dtype=F64, device=device)
    for J in (0, 3):
        cfg = RHMPConfig.solver_preset(in_dims={0: 1, 2: 1}, even_dims={2: 1}, material_dims={2: 1},
                                       metric_reference={2: 0}, learn_metric=False, metric_type="tensor", C=1,
                                       solve_iters=4000, solve_tol=1e-14, solve_picard=J)
        model = _identity_solver(RHMP(cfg, K.geo_dims).to(device=device, dtype=F64))
        y = model(inp, K)[:, 0, 0]
        assert rel(y, A_true) < 1e-7, (J, rel(y, A_true))          # ~1.5e-8: fp32 storage of the Galerkin blocks
    layer = model.layers[0]
    with torch.no_grad():
        layer.picard_log_e0.zero_()
        from rhmp.layers import make_context
        ctx = make_context(K, scaling="dec", B=1, dtype=F64)
        f_top, f_edge = layer.picard_energy_feature(A_true.view(-1, 1, 1), ctx)
    e = torch.expm1(2.0 * f_top[:, 0])
    B2 = torch.as_tensor((sol["Bf"] ** 2).sum(1), dtype=F64, device=device)
    assert rel(e, B2) < 1e-7                                       # fp32-stored unit Galerkin blocks G0
    assert f_edge.shape == (K.n[1], 1)


@pytest.mark.parametrize("metric_type", ["tensor", "diag"])
def test_picard_symmetries(metric_type, device):
    """Random (learned) heads: O(C) equivariance of the stack, E(n) invariance, vertex relabelling, face orientation
    flips and batch independence (fp64, tight CG)."""
    K0 = make_complex("delaunay", device, n=70, seed=5)
    K = CochainComplex.from_triangles(K0.pos.double(), K0.cells[2], device=device).to(dtype=F64)  # fp32-exact source
    cfg = _picard_cfg(metric_type, J=3, lifting="mlp", readout="cochain:0", C=4)
    torch.manual_seed(0)
    model = randomize_(RHMP(cfg, K.geo_dims), scale=0.4).to(device=device, dtype=F64)
    inp = _picard_inputs(K, B=3)
    y = model(inp, K)
    assert torch.isfinite(y).all() and float(y.abs().max()) > 0
    assert model.diagnostics["layer0.picard_delta"] > 1e-8                             # the metric does change
    # O(C) of the message-passing stack
    x = model.lift(inp, K)
    Q = torch.as_tensor(random_orthogonal(4, seed=1, reflect=True), dtype=F64, device=device)
    a = model.propagate({k: v @ Q for k, v in x.items()}, K, inp)
    b = model.propagate(x, K, inp)
    assert rel(a[0], b[0] @ Q) < 1e-9
    # E(n): rotate + reflect + translate the positions exactly in fp64 (the descriptors, stars and Whitney blocks are
    # invariant; the tensor metric recomputes its element geometry from the positions, as in test_equivariance.py)
    import dataclasses
    R = torch.as_tensor(random_orthogonal(2, seed=2, reflect=True), dtype=F64, device=device)
    K2 = dataclasses.replace(K, pos=K.pos @ R.T + 0.3)
    K2.whitney = {km: {**W, "t": W["t"] @ R.T} for km, W in K.whitney.items()}
    assert rel(model(inp, K2), y) < 1e-9
    # vertex relabelling + face orientation flips
    pos, faces = K.pos.double().cpu().numpy(), K.cells[2].cpu().numpy()
    rng = np.random.default_rng(3)
    perm = rng.permutation(K.n[0])
    inv = np.empty_like(perm)
    inv[perm] = np.arange(len(perm))
    f2 = inv[faces]
    f2[1::3] = f2[1::3][:, [0, 2, 1]]
    K3 = CochainComplex.from_triangles(torch.as_tensor(pos[perm]), torch.as_tensor(f2), device=device).to(dtype=F64)
    from rhmp.robustness import remap_complex
    _, maps = remap_complex(K, perm=torch.as_tensor(perm))
    e_idx, _ = maps[1]
    # faces: match by vertex sets
    from rhmp.data import cell_alignment
    fperm, _ = cell_alignment(torch.as_tensor(perm)[K3.cells[2].long().cpu()], K.cells[2].long().cpu(), K.n[0],
                              oriented=False)
    inp3 = {0: inp[0][torch.as_tensor(perm, device=device)], 1: inp[1][e_idx.to(device)],
            2: inp[2][fperm.to(device)]}
    assert rel(model(inp3, K3), y[torch.as_tensor(perm, device=device)]) < 1e-9
    # batch independence: one sample alone, and block-diagonal batches
    y1 = model({k: v[:, 1:2].contiguous() for k, v in inp.items()}, K)
    assert rel(y1, y[:, 1:2]) < 1e-9
    Kb, parts = make_batch(device)
    Kb, parts = Kb.to(dtype=F64), [P.to(dtype=F64) for P in parts]
    ins = [_picard_inputs(P, B=1, seed=i) for i, P in enumerate(parts)]
    yb = model({k: torch.cat([d[k] for d in ins], 0) for k in ins[0]}, Kb)
    ptr = Kb.meta["ptr"][0].tolist()
    for i, P in enumerate(parts):
        assert rel(yb[ptr[i]:ptr[i + 1]], model(ins[i], P)) < 1e-8


@pytest.mark.parametrize("metric_type", ["tensor", "diag"])
def test_metric_geo_off_ignores_descriptors(metric_type):
    """``metric_geo=False``: the metric heads see no geometric descriptors (narrower heads, default unchanged), so the
    learned metric does not change when the descriptors are perturbed (the lifting is linear in the solver preset)."""
    import dataclasses
    K = make_complex("delaunay", "cpu", n=50, seed=3).to(dtype=F64)
    cfg_on = _picard_cfg(metric_type, J=2, readout="cochain:0")
    cfg_off = dataclasses.replace(cfg_on, metric_geo=False)
    assert RHMPConfig.from_dict({k: v for k, v in cfg_on.to_dict().items() if k != "metric_geo"}).metric_geo is True
    n_on = RHMP(cfg_on, K.geo_dims).num_parameters()
    torch.manual_seed(0)
    m = randomize_(RHMP(cfg_off, K.geo_dims), scale=0.4).to(dtype=F64)
    assert m.num_parameters() < n_on
    inp = _picard_inputs(K, B=2)
    y = m(inp, K)
    K2 = dataclasses.replace(K, geo=[g + 0.7 * torch.randn_like(g) for g in K.geo])
    assert rel(m(inp, K2), y) < 1e-12


def test_picard_implicit_gradients_match_finite_differences():
    """Gradients through the weight-tied iterations (implicit CG adjoints, energy invariant, relaxation) agree with
    central finite differences (fp64)."""
    K = make_complex("delaunay", "cpu", n=40, seed=7).to(dtype=F64)
    cfg = _picard_cfg("tensor", J=3, readout="cochain:0", C=2)
    torch.manual_seed(1)
    model = randomize_(RHMP(cfg, K.geo_dims), scale=0.4).to(dtype=F64)
    inp = _picard_inputs(K, B=2, seed=3)
    params = [model.layers[0].picard_logit, model.layers[0].picard_log_e0, model.layers[0].theads["T1"].b2.weight]

    def f():
        return loss_of(model(inp, K))
    grads = torch.autograd.grad(f(), params)
    for p, g in zip(params, grads):
        idx = (0,) * p.dim()
        eps = 1e-5
        with torch.no_grad():
            p[idx] += eps
            fp = f().item()
            p[idx] -= 2 * eps
            fm = f().item()
            p[idx] += eps
        fd = (fp - fm) / (2 * eps)
        assert abs(float(g[idx]) - fd) <= 1e-5 * max(1.0, abs(fd)), (p.shape, float(g[idx]), fd)


def test_picard_trainer_and_checkpoint(tmp_path):
    """``rhmp.train --solver-mode --solve-picard`` on a tiny magnetostatic data set (generator FEM, grad readout and the
    flux -> vertex field map): runs, records the Picard diagnostics, and the checkpoint reproduces the model."""
    from rhmp.data import TaskData
    from rhmp.train import model_output, parse_args, run
    m, phase, Jf, sol, pos, F = _square_problem(n=8)
    K = CochainComplex.from_triangles(torch.as_tensor(pos), torch.as_tensor(F))
    rng = np.random.default_rng(0)
    N = 24
    oh = torch.nn.functional.one_hot(torch.as_tensor(phase).long(), 3).float()
    Js, Bs = [], []
    for i in range(N):
        s = float(rng.uniform(0.2, 1.5))
        so = G.newton_solve(m, phase, Jf * s)
        Fv = m.load(Jf * s)
        Js.append(torch.as_tensor(Fv / m.mass, dtype=torch.float32))
        Bs.append(torch.as_tensor(G.node_field(m, so["Bf"]), dtype=torch.float32))
    omap = FluxToNodeField(base=K)
    task = TaskData.from_arrays(K, {0: torch.stack(Js).unsqueeze(-1), 2: oh.expand(N, -1, -1).contiguous()},
                                torch.stack(Bs), readout="grad", even_dims={2: 3}, name="MXtiny", output_map=omap)
    task.target_kind, task.target_degree = "node_vector", 0
    argv = ["--task", "MXtiny", "--out", str(tmp_path / "run"), "--epochs", "2", "--device", "cpu", "--quiet",
            "--solver-mode", "--solve-picard", "2", "--metric-type", "tensor", "--material", "2:3", "--log-range", "4",
            "--batch", "8", "--solve-iters", "400"]
    res = run(parse_args(argv), task=task)
    assert res["complete"] and math.isfinite(res["test"]["R2"])
    assert res["config"]["solve_picard"] == 2 and "layer0.picard_omega" in res["diagnostics"]["all"]
    m2 = RHMP.from_checkpoint(tmp_path / "run" / "best.pt", map_location="cpu")
    assert m2.cfg.solve_picard == 2
    xb = {k: v.transpose(0, 1).contiguous() for k, v in task.inputs.items()}
    with torch.no_grad():
        y = model_output(task, m2, xb, K)
    assert y.shape == (K.n[0], N, 2) and torch.isfinite(y).all()


# ----------------------------------------------------------------------------------------------------------------
# inhomogeneous Dirichlet data (solve_bc='lift') on the ECT electrostatics family (MXE)
# ----------------------------------------------------------------------------------------------------------------
def _ect_problem(idx: int = 5):
    mesh = G.build_mesh_ect(2.0, 23)
    p = G.draw_params_ect(idx)
    sol = G.solve_ect(mesh, p)
    K = CochainComplex.from_triangles(torch.as_tensor(mesh["pos"]), torch.as_tensor(mesh["faces"]))
    area = torch.as_tensor(G.P1Mesh(mesh["pos"] / G.MM, mesh["faces"], mesh["boundary"]).area)
    F = torch.as_tensor(mesh["faces"])
    Fv = torch.zeros(K.n[0], dtype=F64).index_add_(0, F.reshape(-1), (torch.as_tensor(sol["q_face"]) * area / 3)
                                                     .repeat_interleave(3))
    mass = torch.zeros(K.n[0], dtype=F64).index_add_(0, F.reshape(-1), (area / 3).repeat_interleave(3))
    q = Fv / mass
    q[K.boundary[0]] = 0.0
    V = torch.zeros(K.n[0], dtype=F64)
    el = torch.as_tensor(mesh["electrode"])
    V[el >= 0] = torch.as_tensor(sol["V"])[el[el >= 0]]
    return mesh, sol, K, q, V


def test_lift_reproduces_inhomogeneous_dirichlet_fem(device):
    """ECT sensor (electrodes = holes at random potentials, grounded screen, permittivity contrast 80, space charge):
    the solve layer with ``solve_bc='lift'`` and the true permittivity as frozen face reference reproduces the FEM
    potential (lumped load) to the fp32 accuracy of the Galerkin blocks and keeps the Dirichlet values exactly."""
    mesh, sol, K, q, V = _ect_problem()
    assert K.boundary[0].sum() == int(mesh["boundary"].sum()) and np.isclose(G.ECT_EPS.max() / G.ECT_EPS.min(), 80.0)
    Kd = K.to(device=device, dtype=F64)
    inp = {0: torch.stack([q, V], -1).view(-1, 1, 2).to(device),
           2: torch.log(torch.as_tensor(G.ECT_EPS[sol["phase"]], dtype=F64)).view(-1, 1, 1).to(device)}
    cfg = RHMPConfig.solver_preset(in_dims={0: 2, 2: 1}, even_dims={2: 1}, material_dims={2: 1}, metric_reference={2: 0},
                                   learn_metric=False, metric_type="tensor", C=1, solve_iters=4000, solve_tol=1e-14,
                                   solve_bc="lift")
    model = _identity_solver(RHMP(cfg, Kd.geo_dims).to(device=device, dtype=F64))
    y = model(inp, Kd)[:, 0, 0]
    phi = torch.as_tensor(sol["phi"], dtype=F64, device=device)
    assert rel(y, phi) < 1e-6, rel(y, phi)
    bnd = Kd.boundary[0]
    assert float((y[bnd] - V.to(device)[bnd]).abs().max()) < 1e-14                    # Dirichlet values kept
    # with zero boundary data the lift is the homogeneous Dirichlet solve
    cfg0 = RHMPConfig.from_dict({**cfg.to_dict(), "solve_bc": "dirichlet"})
    m0 = _identity_solver(RHMP(cfg0, Kd.geo_dims).to(device=device, dtype=F64))
    inp0 = {0: torch.stack([q, torch.zeros_like(V)], -1).view(-1, 1, 2).to(device), 2: inp[2]}
    assert rel(model(inp0, Kd), m0(inp0, Kd)) < 1e-12


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "MXE.pt")), reason="datasets/v2/MXE.pt not generated")
def test_mxe_loader():
    from rhmp.tasks import load_task
    td = load_task("MXE", None, native=True, device="cpu", max_samples=30, whitney=False)
    assert td.readout == "grad" and td.target_kind == "node_vector" and td.in_dims == {0: 2, 1: 5, 2: 5}
    assert td.meta["consistency"]["max_rel_err_map_of_d0phi_vs_target"] < 1e-5
    lg = load_task("MXE", None, native=False, device="cpu", max_samples=30, fine=False)
    raw = lg.x_stats[0].denormalize(lg.inputs[0])
    assert torch.allclose(raw[..., 2:].sum(-1), torch.ones_like(raw[..., 0]), atol=1e-5)


def test_trainer_resume_on_device(tmp_path, device):
    """Resume from ``last.pt`` on the training device (the RNG states are loaded with ``map_location=device`` and must
    be restored as CPU byte tensors): an interrupted run continues and matches the uninterrupted one."""
    from rhmp.data import TaskData
    from rhmp.train import parse_args, run
    K = make_complex("delaunay", device, n=40, seed=2)
    g = torch.Generator().manual_seed(0)
    N = 24
    x = torch.randn(N, K.n[0], 1, generator=g).to(device)
    task_args = dict(readout="node_scalar", name="RES")

    def task():
        return TaskData.from_arrays(K, {0: x}, (x ** 2).contiguous(), **task_args)
    base = ["--task", "RES", "--epochs", "3", "--device", str(device), "--quiet", "--C", "8", "--layers", "2",
            "--batch", "8", "--no-tf32"]
    full = run(parse_args(base + ["--out", str(tmp_path / "full")]), task=task())
    part = run(parse_args(base + ["--out", str(tmp_path / "part"), "--stop-after", "1"]), task=task())
    assert part["stopped_after"] == 1
    res = run(parse_args(base + ["--out", str(tmp_path / "part")]), task=task())
    assert res["complete"] and abs(res["best_val_R2"] - full["best_val_R2"]) < 1e-4


# ----------------------------------------------------------------------------------------------------------------
# protocol hooks
# ----------------------------------------------------------------------------------------------------------------
def test_edge_protocol_transforms_extra_tests():
    from rhmp.baselines.registry import prepare_task
    from rhmp.data import TaskData
    K = make_complex("delaunay", n=40, seed=1)
    g = torch.Generator().manual_seed(0)
    N = 12
    task = TaskData.from_arrays(K, {0: torch.randn(N, K.n[0], 2, generator=g)}, torch.randn(N, K.n[0], 2, generator=g),
                                readout="node_vector", name="MXS")
    K2 = make_complex("delaunay", n=60, seed=2)
    task.extra_tests = {"fine": dict(K=[K2] * 3, inputs=[{0: torch.randn(K2.n[0], 2, generator=g)} for _ in range(3)],
                                     target=[torch.randn(K2.n[0], 2, generator=g) for _ in range(3)])}
    t = prepare_task("mpsn", task, {"head": "v1_edge"})
    assert t.target.shape == (N, K.n[1], 1) and t.target_degree == 1
    src, new = task.extra_tests["fine"], t.extra_tests["fine"]
    e = K2.cells[1].long()
    for y, ye in zip(src["target"], new["target"]):
        assert ye.shape == (K2.n[1], 1) and rel(ye, 0.5 * (y[e[:, 0], :1] + y[e[:, 1], :1])) < 1e-7


def test_rotated_view_keeps_complex_dependent_map():
    """``robustness.transformed_task`` with a rotation: the exact model (``d0`` of the true potential through the map)
    scores identically on the rotated view (the map is recomputed from the rotated complex)."""
    from rhmp.data import TaskData
    from rhmp.robustness import transformed_task
    from rhmp.train import model_output
    K = make_complex("delaunay", n=60, seed=4)
    g = torch.Generator().manual_seed(0)
    N = 5
    A = torch.randn(N, K.n[0], generator=g, dtype=F64)
    omap = FluxToNodeField(base=K)
    Phi = K.to(dtype=F64).apply_d(0, A.T.contiguous().unsqueeze(-1)).float()            # (n1, N, 1)
    target = omap(Phi, K).transpose(0, 1).contiguous()
    task = TaskData.from_arrays(K, {0: A.float().unsqueeze(-1)}, target, readout="grad", name="MX", output_map=omap,
                                stats="none")
    task.target_kind, task.target_degree = "node_vector", 0
    Q = random_orthogonal(2, seed=5)
    view = transformed_task(task, torch.arange(N), Q=Q, shift=np.zeros(2))

    def exact(xb, Kx):                                                                  # d0 of the input potential
        return Kx.to(dtype=F64).apply_d(0, xb[0].double().contiguous()).float()
    yv = model_output(view, exact, {0: view.inputs[0].transpose(0, 1).contiguous()}, view.K)
    assert rel(yv.transpose(0, 1), view.target) < 1e-5


# ----------------------------------------------------------------------------------------------------------------
# data set (optional)
# ----------------------------------------------------------------------------------------------------------------
@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "MXS.pt")), reason="datasets/v2/MXS.pt not generated")
def test_mxs_loader():
    from rhmp.tasks import load_task, task_defaults
    td = load_task("MXS", None, native=True, device="cpu", max_samples=30, whitney=False)
    assert td.readout == "grad" and td.target_kind == "node_vector" and td.out_dim == 2
    assert td.in_dims == {0: 1, 1: 3, 2: 3} and td.even_dims == {1: 3, 2: 3}
    assert td.meta["consistency"]["max_rel_err_map_of_d0A_vs_target"] < 1e-5
    assert {"ood15", "fine", "coarse"} <= set(td.extra_tests)
    assert task_defaults("MXS")["C"] == 128
    lg = load_task("MXS", None, native=False, device="cpu", max_samples=30, fine=False)
    raw = lg.x_stats[0].denormalize(lg.inputs[0])
    assert torch.allclose(raw[..., 1:].sum(-1), torch.ones_like(raw[..., 0]), atol=1e-5)  # phase fractions
    assert lg.output_map is None and lg.readout == "node_vector"


# ----------------------------------------------------------------------------------------------------------------
# anisotropic family MXA (generator gen_maxwell_aniso.py)
# ----------------------------------------------------------------------------------------------------------------
def _ga():
    import gen_maxwell_aniso as GA
    return GA


def test_mxa_law_consistency():
    """Energy-based anisotropic law: H = dW/dB, symmetric incremental reluctivity = dH/dB, strong convexity, the
    secant tensor (principal axes on the rolling direction) reproduces H, the two principal curves are the Froehlich
    curves, the linear limit is R diag(nu_e, nu_h) R^T and the law is covariant under rotations."""
    GA = _ga()
    rng = np.random.default_rng(0)
    n = 600
    for cls, (_, mue, muh) in GA.MATERIALS.items():
        b = np.concatenate([10 ** rng.uniform(-5, 0.5, n // 2), rng.uniform(1.6, 2.5, n // 2)])
        th = rng.uniform(0, 2 * math.pi, n)
        B = np.stack([b * np.cos(th), b * np.sin(th)], 1)
        ph = np.full(n, cls, np.int8)
        phi = rng.uniform(0, math.pi, n)
        L = GA.law(B, ph, phi, hessian=True, secant=True)
        d = 1e-7 * np.maximum(b, 1e-3)
        nmin = np.minimum(GA.fk_nu(b, GA.fk_hk(mue)), GA.fk_nu(b, GA.fk_hk(muh)))
        for k in range(2):
            e = np.zeros_like(B)
            e[:, k] = d
            Lp, Lm = GA.law(B + e, ph, phi, hessian=False), GA.law(B - e, ph, phi, hessian=False)
            assert np.max(np.abs((Lp["W"] - Lm["W"]) / (2 * d) - L["H"][:, k]) / np.linalg.norm(L["H"], axis=1)) < 1e-5
            assert np.max(np.abs((Lp["H"] - Lm["H"]) / (2 * d[:, None]) - L["dH"][:, :, k]).max(1) / nmin) < 1e-3
        assert np.abs(L["dH"] - L["dH"].transpose(0, 2, 1)).max() < 1e-9 * GA.NU0
        assert (np.linalg.eigvalsh(L["dH"])[:, 0] / nmin).min() > 0.95
        assert np.allclose(np.einsum("nij,nj->ni", L["nu"], B), L["H"], rtol=1e-12, atol=1e-12)
        w, V = np.linalg.eigh(L["nu"])
        r = np.stack([np.cos(phi), np.sin(phi)], 1)
        if mue != muh:                                                     # principal axes along / across r
            assert np.abs(np.abs((V[:, :, 0] * r).sum(1)) - 1.0).max() < 1e-9
        a = rng.uniform(0, 2 * math.pi, n)
        Rm = np.stack([np.stack([np.cos(a), -np.sin(a)], -1), np.stack([np.sin(a), np.cos(a)], -1)], -2)
        Lr = GA.law(np.einsum("nij,nj->ni", Rm, B), ph, (phi + a) % math.pi, hessian=False)
        assert np.allclose(Lr["H"], np.einsum("nij,nj->ni", Rm, L["H"]), rtol=1e-10, atol=1e-8)
        bb = np.linspace(1e-3, 3.0, 50)
        z = np.zeros(50)
        He = GA.law(np.stack([bb, z], 1), np.full(50, cls, np.int8), z, hessian=False)["H"][:, 0]
        Hh = GA.law(np.stack([z, bb], 1), np.full(50, cls, np.int8), z, hessian=False)["H"][:, 1]
        assert np.allclose(He, GA.fk_nu(bb, GA.fk_hk(mue)) * bb, rtol=1e-12)
        assert np.allclose(Hh, GA.fk_nu(bb, GA.fk_hk(muh)) * bb, rtol=1e-12)
        T0 = GA.linear_tensor(np.array([cls], np.int8), np.array([0.3]))[0]
        R = np.array([[math.cos(0.3), -math.sin(0.3)], [math.sin(0.3), math.cos(0.3)]])
        assert np.allclose(T0, R @ np.diag([GA.fk_nu(0.0, GA.fk_hk(mue)), GA.fk_nu(0.0, GA.fk_hk(muh))]) @ R.T,
                           rtol=1e-12)


def test_mxa_two_curve_secant_law_is_not_monotone():
    """Why the energy form: the literal law nu = R diag(nu_e(|B|), nu_h(|B|)) R^T of grain-oriented steel (ratio 25)
    has an incremental reluctivity with a negative-definite direction near the knee."""
    GA = _ga()
    _, mue, muh = GA.MATERIALS[GA.PHASE_GO]
    hke, hkh = GA.fk_hk(mue), GA.fk_hk(muh)
    worst = np.inf
    for b in np.linspace(1.7, 2.1, 81):
        for th in np.linspace(0, math.pi / 2, 46):
            u = np.array([math.cos(th), math.sin(th)])
            K = np.diag([GA.fk_nu(b, hke), GA.fk_nu(b, hkh)])
            Kp = np.diag([GA.fk_dnu(b, hke), GA.fk_dnu(b, hkh)])
            T = K + b * np.outer(Kp @ u, u)
            worst = min(worst, np.linalg.eigvalsh(0.5 * (T + T.T))[0])
    assert worst < 0.0


def test_mxa_mesh_newton_and_rotation():
    """Coarse lattice mesh: valid, equilateral lattice faces; a layout with oriented steel: zero net current per
    winding, converged Newton at the energy minimum; the rotated mesh with rotated rolling directions gives the same
    potential (exact rotation covariance of the discrete problem)."""
    GA = _ga()
    mesh = GA.build_lattice_mesh(2.0, 33)
    st = mesh["stats"]
    assert st["euler"] == 1 and st["unused"] == 0 and st["nonmanifold"] == 0
    assert st["n_equilateral"] >= 0.9 * st["n_lattice_faces"]
    m = G.P1Mesh(mesh["pos"], mesh["faces"], mesh["boundary"])
    idx = next(i for i in range(50) if GA.draw_layout(i)["core_material"] == GA.PHASE_GO)
    p = GA.draw_layout(idx)
    phase, phi, J = GA.paint(mesh, p)
    assert np.isin(phase, GA.ORIENTED).any() and (phase == GA.PHASE_COPPER).any()
    assert abs((J * m.area).sum()) < 1e-9 * np.abs(J * m.area).sum()                   # zero net current
    sol = GA.newton_solve_a(m, phase, phi, J)
    assert sol["res"] <= 1e-10 and sol["it"] <= 40
    Fv = m.load(J * GA.AMM2)
    E0 = GA._energy(m, phase, phi, Fv, sol["A"])
    rng = np.random.default_rng(1)
    for _ in range(5):
        dA = np.zeros(m.n0)
        dA[m.free] = rng.normal(size=len(m.free)) * 1e-3 * np.abs(sol["A"]).max()
        assert GA._energy(m, phase, phi, Fv, sol["A"] + dA) > E0
    th = 0.7
    Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    mr = G.P1Mesh(mesh["pos"] @ Rm.T, mesh["faces"], mesh["boundary"])
    sol_r = GA.newton_solve_a(mr, phase, np.where(np.isin(phase, GA.ORIENTED), (phi + th) % math.pi, 0.0), J)
    assert np.linalg.norm(sol_r["A"] - sol["A"]) / np.linalg.norm(sol["A"]) < 1e-10


def test_mxa_newton_matches_picard_and_model_operator():
    """Unsaturated excitation: the secant (Picard) iteration converges to the Newton solution.  Oracle: at the
    solution, the generator's secant stiffness equals ``d0^T H_1(T) d0`` with the model's own float64 Galerkin
    assembly of the gradient metric ``T = R^T nu(B) R`` (the frozen-state operator is in the model family), and that
    operator reproduces the load (the fixed point of the frozen true metric is the FEM solution)."""
    import scipy.sparse as sp
    from rhmp.tasks.aniso import _galerkin
    GA = _ga()
    mesh = GA.build_lattice_mesh(2.0, 33)
    m = G.P1Mesh(mesh["pos"], mesh["faces"], mesh["boundary"])
    idx = next(i for i in range(50) if GA.draw_layout(i)["core_material"] == GA.PHASE_GO)
    for factor in (0.1, 0.03, 0.01, 0.003):                       # unsaturated: p99 |B| in the steel < 0.5 T
        phase, phi, J = GA.paint(mesh, GA.draw_layout(idx), factor=factor)
        sol = GA.newton_solve_a(m, phase, phi, J, tol=1e-12)
        if np.percentile(np.linalg.norm(sol["Bf"][phase >= GA.PHASE_NO], axis=1), 99) < 0.5:
            break
    pic = GA.picard_solve_a(m, phase, phi, J, 0.5, 60)
    assert np.linalg.norm(pic["A"] - sol["A"]) / np.linalg.norm(sol["A"]) < 1e-8
    nu = GA.true_tensor(sol["Bf"], phase, phi)
    Rot = np.array([[0.0, 1.0], [-1.0, 0.0]])
    T = np.einsum("ji,fjk,kl->fil", Rot, nu, Rot)
    Kp1 = m.csr(np.einsum("fid,fde,fje->fij", m.G, T, m.G) * m.area[:, None, None])
    K = CochainComplex.from_triangles(torch.as_tensor(mesh["pos"]), torch.as_tensor(mesh["faces"]), device="cpu")
    assert torch.equal(torch.sort(K.cells[2].long(), 1)[0], torch.sort(torch.as_tensor(mesh["faces"]), 1)[0])
    d0 = K.d[0].to_dense().double().numpy()
    Km = d0.T @ _galerkin(K, 1, torch.as_tensor(T)).toarray() @ d0                     # K.pos is float32
    assert np.linalg.norm(Km - Kp1.toarray()) / np.linalg.norm(Kp1.toarray()) < 1e-6
    assert np.linalg.norm(Kp1.toarray() - Kp1.toarray().T) < 1e-12 * np.linalg.norm(Kp1.toarray())
    fr = m.free
    Fv = m.load(J * GA.AMM2)
    assert np.linalg.norm((Kp1 @ sol["A"] - Fv)[fr]) < 1e-10 * np.linalg.norm(Fv[fr])
    assert np.linalg.norm((sp.csr_matrix(Km) @ sol["A"] - Fv)[fr]) < 1e-5 * np.linalg.norm(Fv[fr])


def _mxa_blob(mesh, layouts, GA):
    m = G.P1Mesh(mesh["pos"], mesh["faces"], mesh["boundary"])
    rows = [GA.solve_layout(mesh, m, dict(p, idx=i)) for i, p in enumerate(layouts)]
    return dict(J_face=torch.tensor(np.stack([r["J_face"] for r in rows])),
                phase=torch.tensor(np.stack([r["phase"] for r in rows])),
                phi=torch.tensor(np.stack([r["phi"] for r in rows])),
                B_node=torch.tensor(np.stack([r["B_node"] for r in rows])),
                A=torch.tensor(np.stack([r["A"] for r in rows])))


def test_mxa_inputs_invariance_and_frame_columns():
    """RHMP edge columns (phase fractions, cos^2 between edge and rolling direction) are unchanged by a rotation of
    the configuration; the baselines' frame columns (cos 2phi, sin 2phi) turn by twice the angle, which is what
    ``rotate_frame_columns`` applies to normalised inputs."""
    from rhmp.data import feature_stats
    from rhmp.tasks.maxwell import _raw_fields_mxa, rotate_frame_columns
    GA = _ga()
    mesh = GA.build_lattice_mesh(2.0, 33)
    layouts = [GA.draw_layout(i) for i in range(3)]
    blob = _mxa_blob(mesh, layouts, GA)
    th = 0.9
    Rm = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
    K = CochainComplex.from_triangles(torch.as_tensor(mesh["pos"]), torch.as_tensor(mesh["faces"]), device="cpu")
    Kr = CochainComplex.from_triangles(torch.as_tensor(mesh["pos"] @ Rm.T), torch.as_tensor(mesh["faces"]),
                                       device="cpu")
    blob_r = dict(blob)
    oriented = torch.isin(blob["phase"].long(), torch.tensor(GA.ORIENTED))
    blob_r["phi"] = torch.where(oriented, (blob["phi"].double() + th) % math.pi, blob["phi"].double()).float()
    ids = torch.arange(3)
    raw, raw_r = _raw_fields_mxa(K, blob, ids), _raw_fields_mxa(Kr, blob_r, ids)
    assert (raw["edge"][..., 5:] > 0).any()
    assert torch.allclose(raw["edge"], raw_r["edge"], atol=2e-6)
    assert torch.allclose(raw["dual"], raw_r["dual"], atol=1e-6)
    x = torch.cat([raw["J_node"].unsqueeze(-1), raw["dual"], raw["frame"]], -1)
    xr = torch.cat([raw_r["J_node"].unsqueeze(-1), raw_r["dual"], raw_r["frame"]], -1)
    st = feature_stats(x)
    pairs = ((6, 7), (8, 9))
    got = st.denormalize(rotate_frame_columns(st.normalize(x), st, pairs, th))
    assert torch.allclose(got, xr, atol=2e-5)
    back = rotate_frame_columns(rotate_frame_columns(st.normalize(x), st, pairs, th), st, pairs, -th)
    assert torch.allclose(st.denormalize(back), x, atol=1e-5)                          # physical units


@pytest.mark.skipif(not os.path.exists(os.path.join(DATA, "MXA.pt")), reason="datasets/v2/MXA.pt not generated")
def test_mxa_loader():
    from rhmp.tasks import load_task
    td = load_task("MXA", None, native=True, device="cpu", max_samples=30, whitney=False)
    assert td.readout == "grad" and td.target_kind == "node_vector" and td.out_dim == 2
    assert td.in_dims == {0: 1, 1: 7, 2: 5} and td.even_dims == {1: 7, 2: 5}
    assert td.meta["consistency"]["max_rel_err_map_of_d0A_vs_target"] < 1e-5
    assert {"ood15", "fine", "coarse", "heldout", "heldout_test100"} <= set(td.extra_tests)
    no = load_task("MXA_noorient", None, native=True, device="cpu", max_samples=30, whitney=False, fine=False)
    assert no.in_dims == {0: 1, 1: 5, 2: 5}
    lg = load_task("MXA", None, native=False, device="cpu", max_samples=30, fine=False)
    assert lg.in_dims == {0: 10} and lg.meta["frame_pairs"] == [[6, 7], [8, 9]]
    raw = lg.x_stats[0].denormalize(lg.inputs[0])
    assert torch.allclose(raw[..., 1:6].sum(-1), torch.ones_like(raw[..., 0]), atol=1e-5)
    assert (raw[..., 6:8].norm(dim=-1) <= raw[..., 4] + 1e-5).all()                     # |GO frame| <= GO fraction
