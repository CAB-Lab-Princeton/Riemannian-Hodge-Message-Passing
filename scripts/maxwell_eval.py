"""Extra evaluations of a trained MXS / MXA / MXE run (``rhmp/tasks/maxwell.py``: nonlinear magnetostatics with
isotropic / anisotropic steel, ECT electrostatics): the zero-shot and rotation rows of the paper's magnetostatics tables
(collected into ``results/evaluations/maxwell_runs.json`` by ``scripts/maxwell_tables.py``).

    python3 -u scripts/maxwell_eval.py runs/maxwell/MXS/<run>_s42 [--dump] [--angles 30,90,137] [--recovery-only]

Writes ``<run>/maxwell_eval.json`` with

* ``test100``: the library metrics of the checkpoint on test100 (reproduces ``result.json``);
* ``divergence``: net flux of B out of every triangle on test100, relative to the flux through its edges
  (``|sum_e s_fe Phi_e| / sum_e |Phi_e|``), median / p90 / p99 / max, for the edge fluxes of the predicted vertex fields
  (trapezoid rule, ``Phi_e = (B_src + B_dst)/2 . n_e |e|``) and, for RHMP, of its native flux cochain ``Phi = d0 phi``
  (closed by construction) and of the reference vertex fields themselves;
* ``rotation``: test100 metrics with the mesh rotated about the centre of its bounding box (MXS (0, 30) mm, MXE (0, 0))
  by each angle (positions rotated, scalar inputs unchanged, target vectors rotated;
  ``rhmp.robustness.transformed_task``; MXA baselines: the frame components (cos 2phi, sin 2phi) of the rolling
  directions turn by twice the angle, the RHMP edge columns are invariant);
* ``timing``: inference seconds per sample (batch 64, synchronised) and the training-time numbers of ``result.json``;
* ``transfer``: the zero-shot sets (currents x1.5, refined and coarsened mesh, MXA: held-out layout family; their
  test100 instances) as trained and, for solve models, with more CG iterations per solve (a numerical setting of the
  solve, not a learned one);
* ``recovery`` (RHMP solve runs with a learned metric): MXS: the learned reluctivity of the final iteration against
  the true law ``nu(|B|)`` on the iron faces of 64 test samples (``<run>/recovery.npz``: per-face rows; binned curve in
  the json), correlation and least-squares slope of log nu; MXE: the learned permittivity per phase against the true
  values (relative to air); MXA: the learned tensor of the final iteration against the true secant reluctivity
  tensor of the steel faces (principal-direction error, eigenvalue ratio, the curves along and across the rolling
  direction against |B|; ``recovery.npz`` holds the per-face rows).

``--dump``: predictions and targets in physical units (T) with the mesh and the inputs (current density in A/mm^2,
face phases) for the test split and the extra test sets, ``<root>/preds/<run>_<set>.npz``.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rhmp import metrics as M  # noqa: E402
from rhmp.train import _jsonable, _renormalize, evaluate, make_batch, model_output, predict  # noqa: E402

DUMP_SETS = ("test", "ood15", "fine", "coarse", "heldout")
EVAL_BATCH = 64                 # batch of the metric passes (``--batch``; memory only, no effect on the numbers)


def mesh_centre(K) -> np.ndarray:
    """Centre of the bounding box of the mesh (the rotation centre of the frame-robustness test)."""
    P = K.pos.double().cpu().numpy()
    return 0.5 * (P.min(0) + P.max(0))


def load_run(run: str, device):
    """Model, task (renormalised with the run's statistics, prepared for the model's protocol) and metadata."""
    from rhmp.model import RHMP
    from rhmp.tasks import load_task
    conf = json.load(open(os.path.join(run, "config.json")))
    args = conf["args"]
    trained = args.get("model") or "rhmp"
    native = bool(args.get("native", True))
    mcfg = conf.get("model", {})
    need_w = trained == "rhmp" and (mcfg.get("metric_type") == "tensor" or bool(mcfg.get("solve_picard")))
    task = load_task(args.get("task", "MXS"), args.get("root"), native=native, device=device,
                     star=args.get("star", "cotan"), whitney=need_w)
    task = _renormalize(task, conf["data"])
    ck = torch.load(os.path.join(run, "best.pt"), map_location=device, weights_only=False)
    untransformed = task
    if trained == "rhmp":
        model = RHMP.from_checkpoint(ck, map_location=device)
    else:
        import dataclasses

        from rhmp.baselines.registry import SPECS, from_checkpoint, prepare_task
        task = prepare_task(trained, task, ck.get("build"))
        model = from_checkpoint(ck, task, map_location=device)
        if not SPECS[trained].uses_output_map and task.output_map is not None:
            task = dataclasses.replace(task, output_map=None)
    model.eval()
    return model, task, untransformed, conf, trained


def _edge_protocol(task) -> bool:
    return bool(task.meta.get("v1_edge_protocol"))


def rel_divergence(K, B: torch.Tensor) -> torch.Tensor:
    """Per-triangle relative net flux of vertex fields ``B (N, n0, 2)`` (trapezoid edge fluxes), ``(N, n2)``."""
    from rhmp.tasks.airfrans import trapezoid_flux
    e = K.cells[1].long()
    P = K.pos.double()
    Phi = trapezoid_flux(P, e, B.double().permute(1, 0, 2))                    # (n1, N)
    return cochain_divergence(K, Phi).T


def cochain_divergence(K, Phi: torch.Tensor) -> torch.Tensor:
    """``|d1 Phi| / (|d1| |Phi|)`` per triangle for edge cochains ``Phi (n1, N)`` (float64), ``(n2, N)``."""
    from rhmp.ops import spmm
    Phi = Phi.double().contiguous()
    num = spmm(K.d[1], Phi)
    den = spmm(K.d_abs[1], Phi.abs().contiguous())
    return num.abs() / den.clamp_min(1e-300)


def _stats(x: torch.Tensor) -> dict:
    x = x.reshape(-1).double().cpu()
    q = torch.quantile(x[:min(len(x), 2 ** 24)], torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64))
    return dict(median=float(q[0]), p90=float(q[1]), p99=float(q[2]), max=float(x.max()), mean=float(x.mean()))


def _raw(task, y: torch.Tensor) -> torch.Tensor:
    return task.y_stats.denormalize(y)


def divergence_block(model, task, trained: str, idx, device) -> dict:
    """Divergence residuals on the samples ``idx`` (see the module docstring)."""
    out = {}
    if _edge_protocol(task):
        return dict(note="v1 edge protocol: the model predicts an edge-averaged scalar, no vector field")
    pred, tgt = predict(model, task, idx, EVAL_BATCH, device)
    K = task.K
    out["pred_trapezoid"] = _stats(rel_divergence(K, _raw(task, pred)))
    out["target_trapezoid"] = _stats(rel_divergence(K, _raw(task, tgt)))
    if trained == "rhmp" and task.output_map is not None:                      # native flux cochain d0 phi
        res = []
        for b in torch.split(idx, EVAL_BATCH):
            _, xb, _ = make_batch(task, b.to(device), device)
            with torch.no_grad():
                Phi = model(xb, K)[..., 0]                                     # (n1, B) edge cochain
            res.append(cochain_divergence(K, Phi.double()).T)
        out["native_cochain"] = _stats(torch.cat(res))
    return out


def rotation_block(model, untransformed, trained: str, idx, device, angles) -> dict:
    """Metrics on rotated copies of the test samples ``idx``."""
    from rhmp.baselines.registry import SPECS, prepare_task
    from rhmp.robustness import transformed_task
    out = {}
    c = mesh_centre(untransformed.K)
    for a in angles:
        th = math.radians(a)
        Q = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        view = transformed_task(untransformed, idx, Q=Q, shift=c - Q @ c)
        pairs = untransformed.meta.get("frame_pairs") or []
        if pairs:                                                              # frame components rotate (MXA legacy)
            import dataclasses

            from rhmp.tasks.maxwell import rotate_frame_columns
            ins = dict(view.inputs)
            ins[0] = rotate_frame_columns(ins[0], untransformed.x_stats[0], pairs, th)
            view = dataclasses.replace(view, inputs=ins)
        if trained != "rhmp":
            import dataclasses
            ck_build = getattr(model, "build", None) or {}
            view = prepare_task(trained, view, ck_build)
            if not SPECS[trained].uses_output_map and view.output_map is not None:
                view = dataclasses.replace(view, output_map=None)
        m = evaluate(model, view, view.split[2], EVAL_BATCH, device)
        out[str(a)] = {k: m[k] for k in ("R2", "NRMSE", "SSIM", "Pearson", "MSE", "N")}
    return out


def transfer_block(model, task, trained: str, device, iters: int) -> dict:
    """Zero-shot sets (``*_test100``): metrics as trained and, for models with solve layers, with ``iters`` CG
    iterations per solve (the finer mesh needs more iterations for the same residual; a numerical setting, not a
    learned one).  Records the CG iterations / residuals of the last batch."""
    out = {}
    solves = [lay for lay in getattr(model, "layers", []) if getattr(lay, "kind", None) == "solve"]
    for name in ("ood15_test100", "fine_test100", "coarse_test100", "heldout_test100"):
        src = task.extra_tests.get(name)
        if src is None:
            continue
        idx = torch.arange(len(src["target"]))
        row = {}
        for label, it in (("as_trained", None), (f"iters{iters}", iters)):
            if it is not None and not solves:
                continue
            old = [(lay.solve_iters, lay.solve_tol) for lay in solves]
            if it is not None:
                for lay in solves:
                    lay.solve_iters, lay.solve_tol = it, min(lay.solve_tol, 1e-6)
            model.record_diagnostics = True
            m = evaluate(model, task, idx, EVAL_BATCH, device, source=src)
            diag = {k: v for k, v in getattr(model, "diagnostics", {}).items() if "solve_" in k}
            model.record_diagnostics = False
            for lay, (a_, b_) in zip(solves, old):
                lay.solve_iters, lay.solve_tol = a_, b_
            row[label] = {**{k: m[k] for k in ("R2", "NRMSE", "SSIM", "Pearson", "N")}, "solver": diag}
        out[name] = row
    return out


def timing_block(model, task, idx, device, result: dict) -> dict:
    batches = list(torch.split(idx, 64))
    _, xb, _ = make_batch(task, batches[0].to(device), device)
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda *_: None)
    with torch.no_grad():
        model_output(task, model, xb, task.K)                                   # warm-up
        sync(device)
        t0 = time.time()
        n = 0
        for b in batches:
            _, xb, _ = make_batch(task, b.to(device), device)
            model_output(task, model, xb, task.K)
            n += len(b)
        sync(device)
    out = dict(infer_s_per_sample=(time.time() - t0) / max(n, 1), params=result.get("params"),
               s_per_epoch=result.get("s_per_epoch"), wall_clock_s=result.get("wall_clock_s"),
               best_epoch=result.get("best_epoch"), peak_mem_GB=result.get("peak_mem_GB"))
    diag = (result.get("diagnostics") or {}).get("all") or {}
    out["solver"] = {k: v for k, v in diag.items() if any(s in k for s in ("solve_it", "solve_res", "picard", "lam"))}
    return out


def _learned_face_metric(model, layer, task, ids, device) -> tuple:
    """Learned log material per face of the solve layer (final Picard iteration, or the metric of the layer input),
    ``(n, n2)``, the tensor anisotropy ratio (or None) and a description."""
    from rhmp.layers import _sym_eigvals
    K = task.K
    model.record_diagnostics = True
    sig_all, lr_all = [], []
    for b in torch.split(ids, 32):
        _, xb, _ = make_batch(task, b.to(device), device)
        with torch.no_grad():
            model(xb, K)
        rec = layer.last_picard if layer.picard > 0 else None
        if not rec:
            li = [i for i, lay in enumerate(model.layers) if lay is layer][0]
            mf = model.metric_fields(xb, K)[li]
            (sig_all if 1 in mf["sigma"] else lr_all).append(mf["sigma"][1] if 1 in mf["sigma"] else mf["log_ratio"][1])
        elif "sigma" in rec:
            sig_all.append(rec["sigma"])
        else:
            lr_all.append(rec["log_ratio1"])
    if sig_all:
        sig = torch.cat(sig_all, 1).permute(1, 0, 2, 3).double()
        ev = _sym_eigvals(sig)
        return 0.5 * torch.log(ev[..., 0] * ev[..., 1]), ev[..., 1] / ev[..., 0], "tensor: half log-det of sigma_f"
    lr = torch.cat(lr_all, 1).T.double()
    fe = K.meta["face_edges"].long()
    return lr[:, fe].mean(-1), None, "diag: mean over the face edges of log(H_1 / star_1)"


def recovery_block_mxe(model, task, run: str, device, n: int = 64) -> dict:
    """Learned permittivity per phase (relative to air) vs the true values (MXE)."""
    from rhmp.tasks.maxwell import ECT_PHASES, _load_blob
    from rhmp.tasks import resolve_root
    layer = [lay for lay in model.layers if getattr(lay, "kind", None) == "solve"]
    if not layer or not layer[-1].learn_metric:
        return dict(note="no solve layer with a learned metric")
    layer = layer[-1]
    blob = _load_blob(resolve_root(None), "MXE")
    ids = task.meta["paper_eval"][:n]
    eps = torch.as_tensor(blob["meta"]["eps"], dtype=torch.float64)
    phase = blob["phase"][ids].long().to(device)
    learned, aniso, kind = _learned_face_metric(model, layer, task, ids, device)
    air = phase == 0
    ref = torch.stack([learned[i][air[i]].median() for i in range(len(ids))])
    y = (learned - ref[:, None]).cpu().numpy()
    x = torch.log(eps[phase.cpu()] / eps[0]).numpy()
    per = {}
    for k, name in enumerate(ECT_PHASES):
        sel = phase.cpu().numpy() == k
        if sel.sum():
            per[name] = dict(n=int(sel.sum()), true_log_eps=float(np.log(eps[k] / eps[0])),
                             learned_median=float(np.median(y[sel])), learned_q25=float(np.percentile(y[sel], 25)),
                             learned_q75=float(np.percentile(y[sel], 75)))
    xs, ys = x.reshape(-1), y.reshape(-1)
    np.savez_compressed(os.path.join(run, "recovery.npz"), phase=phase.cpu().numpy(), learned_log_eps_rel=y,
                        true_log_eps_rel=x)
    return dict(kind=kind, n_samples=int(len(ids)), per_phase=per, corr_log_eps=float(np.corrcoef(xs, ys)[0, 1]),
                slope_log_eps=float(np.polyfit(xs, ys, 1)[0]), rmse_log_eps=float(np.sqrt(np.mean((ys - xs) ** 2))))


def _learned_sigma(model, layer, task, ids, device):
    """Learned per-face tensors of the solve layer, ``(n, n2, 2, 2)`` float64 (final Picard iteration, or the metric of
    the layer input for a single solve), or ``None`` for a diagonal metric."""
    K = task.K
    model.record_diagnostics = True
    sig_all = []
    for b in torch.split(ids, 32):
        _, xb, _ = make_batch(task, b.to(device), device)
        with torch.no_grad():
            model(xb, K)
        rec = layer.last_picard if layer.picard > 0 else None
        if rec:
            if "sigma" not in rec:
                return None
            sig_all.append(rec["sigma"])
        else:
            li = [i for i, lay in enumerate(model.layers) if lay is layer][0]
            mf = model.metric_fields(xb, K)[li]
            if 1 not in mf["sigma"]:
                return None
            sig_all.append(mf["sigma"][1])
    model.record_diagnostics = False
    return torch.cat(sig_all, 1).permute(1, 0, 2, 3).double()


def _gen_aniso():
    gdir = os.path.join(ROOT, "datasets", "generators")
    if gdir not in sys.path:
        sys.path.insert(0, gdir)
    import gen_maxwell_aniso
    return gen_maxwell_aniso


def recovery_block_mxa(model, task, run: str, device, n: int = 64) -> dict:
    """Learned tensor of the final iteration vs the true secant reluctivity tensor on the steel faces (MXA).

    Both are compared as metrics of the potential gradient (``T = R^T nu R``, ``B = R grad A``), relative to air (the
    learned tensors divided by the per-sample median ``sqrt(det sigma)`` over the air faces: the scale of the metric is a
    gauge of the normalised model).  Along the rolling direction r the gradient metric is the reluctivity for B
    *across* r (hard), across r it is the one for B along r (easy).  On the equilateral lattice faces a head that sees
    the edge columns ``cos^2(edge, r)`` produces tensors that are symmetric about r, so their principal axes lie along
    and across r; the informative quantities are the order of the eigenvalues (major axis along r:
    ``frac_major_axis_along_rolling_direction``), the eigenvalue ratio and the two curves."""
    from rhmp import geometry as geo
    from rhmp.layers import _sym_eigvals
    from rhmp.tasks import resolve_root
    from rhmp.tasks.maxwell import NU0, _load_blob
    layer = [lay for lay in model.layers if getattr(lay, "kind", None) == "solve"]
    if not layer or not layer[-1].learn_metric:
        return dict(note="no solve layer with a learned metric")
    layer = layer[-1]
    ids = task.meta["paper_eval"][:n]
    sig = _learned_sigma(model, layer, task, ids, device)
    if sig is None:
        return dict(note="diagonal metric: no per-face tensor (a diagonal edge metric cannot represent a misaligned "
                         "anisotropic tensor)")
    gen = _gen_aniso()
    blob = _load_blob(resolve_root(None), "MXA")
    K = task.K
    F = K.cells[2].long()
    g = geo.barycentric_gradients(K.pos.double()[F].cpu() * 1e-3)              # (n2, 3, 2) 1/m
    A = blob["A"][ids].double()
    grad = torch.einsum("fkd,nfk->nfd", g, A[:, F.cpu()])                     # (n, n2, 2)
    Bf = torch.stack([grad[..., 1], -grad[..., 0]], -1).numpy()
    phase = blob["phase"][ids].long().numpy()
    phi = blob["phi"][ids].double().numpy()
    nn_, n2 = phase.shape
    nu = gen.true_tensor(Bf.reshape(-1, 2), phase.reshape(-1), phi.reshape(-1)).reshape(nn_, n2, 2, 2)
    Rot = np.array([[0.0, 1.0], [-1.0, 0.0]])
    T = np.einsum("ji,nfjk,kl->nfil", Rot, nu, Rot) / NU0                      # gradient metric relative to air
    S = sig.cpu().numpy()
    ev = _sym_eigvals(sig).cpu().numpy()                                        # ascending
    logdet_half = 0.5 * np.log(ev[..., 0] * ev[..., 1])
    air = phase == 0
    ref = np.array([np.median(logdet_half[i][air[i]]) for i in range(nn_)])
    S = S / np.exp(ref)[:, None, None, None]
    r = np.stack([np.cos(phi), np.sin(phi)], -1)
    q = np.stack([-np.sin(phi), np.cos(phi)], -1)
    l_hard = np.log(np.einsum("nfi,nfij,nfj->nf", r, S, r))
    l_easy = np.log(np.einsum("nfi,nfij,nfj->nf", q, S, q))
    t_hard = np.log(np.einsum("nfi,nfij,nfj->nf", r, T, r))
    t_easy = np.log(np.einsum("nfi,nfij,nfj->nf", q, T, q))
    w, V = np.linalg.eigh(S)                                                    # learned principal axes
    vmax = V[..., :, 1]
    ang = np.degrees(np.arccos(np.clip(np.abs((vmax * r).sum(-1)), 0.0, 1.0)))
    l_ratio = w[..., 1] / w[..., 0]
    wt = np.linalg.eigvalsh(T)
    t_ratio = wt[..., 1] / wt[..., 0]
    Bm = np.linalg.norm(Bf, axis=-1)
    out = dict(kind="tensor: learned sigma_f of the final iteration vs R^T nu(B) R", n_samples=int(nn_))
    edges = np.arange(0.0, 2.65, 0.1)
    for ph_id, name in ((3, "GO"), (4, "LAM")):
        sel = phase == ph_id
        if not sel.any():
            continue
        strong = sel & (t_ratio > 2.0)
        x = np.concatenate([t_easy[sel], t_hard[sel]])
        y = np.concatenate([l_easy[sel], l_hard[sel]])
        row = dict(n_faces=int(sel.sum()), principal_angle_err_deg_median=float(np.median(ang[strong])),
                   principal_angle_err_deg_p90=float(np.percentile(ang[strong], 90)),
                   frac_angle_err_lt_15deg=float((ang[strong] < 15.0).mean()),
                   frac_major_axis_along_rolling_direction=float((ang[strong] < 45.0).mean()),
                   ratio_true_median=float(np.median(t_ratio[sel])), ratio_learned_median=float(np.median(l_ratio[sel])),
                   corr_log_ratio=float(np.corrcoef(np.log(t_ratio[sel]), np.log(l_ratio[sel]))[0, 1]),
                   corr_log_nu_both_directions=float(np.corrcoef(x, y)[0, 1]),
                   slope_log_nu_both_directions=float(np.polyfit(x, y, 1)[0]),
                   corr_log_nu_easy=float(np.corrcoef(t_easy[sel], l_easy[sel])[0, 1]),
                   corr_log_nu_hard=float(np.corrcoef(t_hard[sel], l_hard[sel])[0, 1]))
        curve = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = sel & (Bm >= lo) & (Bm < hi)
            if b.sum() >= 20:
                curve[f"{lo:.2f}"] = dict(n=int(b.sum()), B_mid=float(0.5 * (lo + hi)),
                                          true_easy=float(np.median(t_easy[b])), learned_easy=float(np.median(l_easy[b])),
                                          learned_easy_q25=float(np.percentile(l_easy[b], 25)),
                                          learned_easy_q75=float(np.percentile(l_easy[b], 75)),
                                          true_hard=float(np.median(t_hard[b])), learned_hard=float(np.median(l_hard[b])),
                                          learned_hard_q25=float(np.percentile(l_hard[b], 25)),
                                          learned_hard_q75=float(np.percentile(l_hard[b], 75)),
                                          ratio_true=float(np.median(t_ratio[b])),
                                          ratio_learned=float(np.median(l_ratio[b])),
                                          angle_err_median=float(np.median(ang[b])))
        row["curve"] = curve
        out[name] = row
    sel = phase == 2
    if sel.any():
        lt, ll = 0.5 * np.log(np.linalg.det(T)), 0.5 * np.log(np.linalg.det(S))
        x, y = lt[sel], ll[sel]
        curve = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            b = sel & (Bm >= lo) & (Bm < hi)
            if b.sum() >= 20:
                curve[f"{lo:.2f}"] = dict(n=int(b.sum()), B_mid=float(0.5 * (lo + hi)), true=float(np.median(lt[b])),
                                          learned=float(np.median(ll[b])), learned_q25=float(np.percentile(ll[b], 25)),
                                          learned_q75=float(np.percentile(ll[b], 75)),
                                          ratio_learned=float(np.median(l_ratio[b])))
        out["NO"] = dict(n_faces=int(sel.sum()), corr_log_nu=float(np.corrcoef(x, y)[0, 1]),
                         slope_log_nu=float(np.polyfit(x, y, 1)[0]), ratio_learned_median=float(np.median(l_ratio[sel])),
                         curve=curve)
    out["air_ratio_learned_median"] = float(np.median(l_ratio[air]))
    steel = phase >= 2
    np.savez_compressed(os.path.join(run, "recovery.npz"), sample=np.nonzero(steel)[0], phase=phase[steel],
                        phi=phi[steel], B=Bm[steel], true_easy=t_easy[steel], true_hard=t_hard[steel],
                        learned_easy=l_easy[steel], learned_hard=l_hard[steel], angle_err_deg=ang[steel],
                        ratio_true=t_ratio[steel], ratio_learned=l_ratio[steel])
    return out


def recovery_block(model, task, run: str, device, n: int = 64) -> dict:
    """Learned reluctivity of the final Picard / solve iteration vs the true law on the iron faces."""
    if task.name == "MXE":
        return recovery_block_mxe(model, task, run, device, n)
    if task.name.startswith("MXA"):
        return recovery_block_mxa(model, task, run, device, n)
    from rhmp.tasks.maxwell import NU0, _load_blob, reluctivity, triangle_areas
    from rhmp.tasks import resolve_root
    layer = None
    for lay in model.layers:
        if getattr(lay, "kind", None) == "solve":
            layer = lay
    if layer is None or not layer.learn_metric:
        return dict(note="no solve layer with a learned metric")
    blob = _load_blob(resolve_root(None), "MXS")
    ids = task.meta["paper_eval"][:n]
    K = task.K
    F = K.cells[2].long()
    P = K.pos.double()[F] * 1e-3                                              # m
    from rhmp import geometry as geo
    g = geo.barycentric_gradients(P)                                           # (n2, 3, 2)
    A = blob["A"][ids].double().to(device)                                     # (n, n0) Wb/m
    grad = torch.einsum("fkd,nfk->nfd", g, A[:, F])                           # (n, n2, 2)
    Bmag = grad.norm(dim=-1)                                                   # (n, n2) T
    phase = blob["phase"][ids].to(device).long()
    rows, curve = [], {}
    model.record_diagnostics = True
    sig_all, lr_all, ef_all = [], [], []
    for b in torch.split(ids, 32):
        _, xb, _ = make_batch(task, b.to(device), device)
        with torch.no_grad():
            model(xb, K)
        rec = layer.last_picard if layer.picard > 0 else None
        if rec:
            ef_all.append(rec["energy_feature"])                               # invariant of y_{J-1} (n2, B)
        if rec is None or not rec:                                             # single solve: metric of the input
            mf = None
            for lay_i, lay in enumerate(model.layers):
                if lay is layer:
                    mf = model.metric_fields(xb, K)[lay_i]
            if 1 in mf["sigma"]:
                sig_all.append(mf["sigma"][1])
            else:
                lr_all.append(mf["log_ratio"][1])
        elif "sigma" in rec:
            sig_all.append(rec["sigma"])
        else:
            lr_all.append(rec["log_ratio1"])
    nu_true = torch.where(phase == 2, reluctivity(Bmag), torch.full_like(Bmag, NU0))
    out = dict(n_samples=int(len(ids)))
    if sig_all:
        from rhmp.layers import _sym_eigvals
        sig = torch.cat(sig_all, 1).permute(1, 0, 2, 3).double()              # (n, n2, 2, 2)
        ev = _sym_eigvals(sig)                                                 # closed form (huge batches of 2x2)
        log_iso = 0.5 * torch.log(ev[..., 0] * ev[..., 1])                     # half log-det
        aniso = ev[..., 1] / ev[..., 0]
        learned, kind = log_iso, "tensor: half log-det of sigma_f"
    else:
        lr = torch.cat(lr_all, 1).T.double()                                   # (n, n1) log(H1 / star1)
        fe = K.meta["face_edges"].long()
        learned = lr[:, fe].mean(-1)                                           # face mean of the edge metric
        aniso, kind = None, "diag: mean over the face edges of log(H_1 / star_1)"
    iron = phase == 2
    air = phase == 0
    air_ref = torch.stack([learned[i][air[i]].median() for i in range(len(ids))])     # per-sample gauge (air)
    y = (learned - air_ref[:, None])[iron].cpu().numpy()                       # learned log(nu / nu_air)
    x = torch.log(nu_true / NU0)[iron].cpu().numpy()                           # true log(nu / nu0) = -log mu_r
    bm = Bmag[iron].cpu().numpy()
    r = float(np.corrcoef(x, y)[0, 1])
    slope, icpt = (float(v) for v in np.polyfit(x, y, 1))
    out.update(kind=kind, n_iron_faces=int(iron.sum()), corr_log_nu=r, slope_log_nu=slope, intercept=icpt,
               rmse_log_nu=float(np.sqrt(np.mean((y - x) ** 2))),
               air_spread=float(torch.cat([learned[i][air[i]] - air_ref[i] for i in range(len(ids))]).std()))
    edges = np.arange(0.0, 2.65, 0.05)
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (bm >= lo) & (bm < hi)
        if sel.sum() >= 20:
            curve[f"{lo:.2f}"] = dict(n=int(sel.sum()), B_mid=float(0.5 * (lo + hi)),
                                      true_log_nu=float(np.median(x[sel])),
                                      learned_median=float(np.median(y[sel])),
                                      learned_q25=float(np.percentile(y[sel], 25)),
                                      learned_q75=float(np.percentile(y[sel], 75)))
    out["curve"] = curve
    bmod = np.zeros(0)
    if ef_all:                                  # the model's own field strength: |grad y_{J-1}|, calibrated to tesla
        e0 = float(torch.exp(layer.picard_log_e0.detach()))
        ef = torch.cat(ef_all, 1).T.double()                                   # (n, n2)
        braw = torch.sqrt(e0 * torch.expm1(2.0 * ef))[iron].cpu().numpy()
        ok = bm > 0.2
        kappa = float(np.median(bm[ok] / np.maximum(braw[ok], 1e-30)))
        bmod = kappa * braw
        out["model_field"] = dict(calibration_T_per_unit=kappa,
                                  corr_B_model_vs_true=float(np.corrcoef(bmod, bm)[0, 1]),
                                  median_rel_err=float(np.median(np.abs(bmod[ok] - bm[ok]) / bm[ok])))
        cm = {}
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (bmod >= lo) & (bmod < hi)
            if sel.sum() >= 20:
                cm[f"{lo:.2f}"] = dict(n=int(sel.sum()), B_mid=float(0.5 * (lo + hi)),
                                       true_log_nu_at_B=float(np.log(float(reluctivity(np.array([0.5 * (lo + hi)]))[0])
                                                                     / NU0)),
                                       learned_median=float(np.median(y[sel])),
                                       learned_q25=float(np.percentile(y[sel], 25)),
                                       learned_q75=float(np.percentile(y[sel], 75)))
        out["curve_model_field"] = cm
    np.savez_compressed(os.path.join(run, "recovery.npz"), B=bm, true_log_nu_rel=x, learned_log_nu_rel=y,
                        B_model=bmod, sample=np.repeat(np.arange(len(ids)), iron.sum(1).cpu().numpy()),
                        aniso=(aniso[iron].cpu().numpy() if aniso is not None else np.zeros(0)))
    return out


def dump_block(model, task, untransformed, run: str, trained: str, device) -> dict:
    """Predictions / targets in physical units with mesh and inputs, per set."""
    from rhmp.tasks.maxwell import _load_blob
    from rhmp.tasks import resolve_root
    root = os.path.dirname(os.path.abspath(run))
    out_dir = os.path.join(root, "preds")
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(os.path.abspath(run))
    files = {}
    if _edge_protocol(task):
        sets = [("test", None)]
    else:
        sets = [("test", None)] + [(s, task.extra_tests.get(s)) for s in DUMP_SETS[1:] if s in task.extra_tests]
    for label, src in sets:
        tname = untransformed.name
        if src is None:
            idx = task.split[2]
            pred, tgt = predict(model, task, idx, EVAL_BATCH, device)
            K = task.K
            blob = _load_blob(resolve_root(None), tname)
            ids = idx
        else:
            idx = torch.arange(len(src["target"]))
            pred, tgt = predict(model, task, idx, EVAL_BATCH, device, source=src)
            K = src["K"][0]
            blob = _load_blob(resolve_root(None), tname + "_" + label)
            ids = idx
        pred, tgt = _raw(task, pred), _raw(task, tgt)
        f = os.path.join(out_dir, f"{name}_{label}.npz")
        mxe = tname == "MXE"
        if mxe:
            extra_in = dict(potentials=blob["potentials"][ids].numpy(), q_face=blob["q_face"][ids].numpy(),
                            electrode=blob["electrode"].numpy(), region=blob["region"].numpy())
        elif "J_face" in blob:                                                  # MXA
            extra_in = dict(J_face=blob["J_face"][ids].numpy(), phi=blob["phi"][ids].numpy(),
                            family=blob["family"][ids].numpy())
        else:
            extra_in = dict(currents=blob["currents"][ids].numpy(), region=blob["region"].numpy())
        np.savez_compressed(f, pos_mm=K.pos.double().cpu().numpy(), faces=K.cells[2].cpu().numpy(),
                            edges=K.cells[1].cpu().numpy(), pred=pred.float().cpu().numpy(),
                            target=tgt.float().cpu().numpy(), phase=blob["phase"][ids].numpy(),
                            sample_ids=blob["sample_ids"][ids].numpy(),
                            units=np.array("mm, V, V/mm, V/mm^2" if mxe else "mm, T, A/mm^2"),
                            target_space=np.array("edge-averaged first component" if _edge_protocol(task) else
                                                  ("vertex E (V/mm)" if mxe else "vertex B (T)")), **extra_in)
        files[label] = f
    return files


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("run")
    p.add_argument("--dump", action="store_true")
    p.add_argument("--angles", default="30,90,137")
    p.add_argument("--recovery-only", action="store_true")
    p.add_argument("--transfer-iters", type=int, default=512,
                   help="CG iterations per solve for the zero-shot sets (second evaluation of solve models)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch", type=int, default=64, help="batch of the metric passes (memory only)")
    a = p.parse_args(argv)
    global EVAL_BATCH
    EVAL_BATCH = a.batch
    device = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    model, task, untransformed, conf, trained = load_run(a.run, device)
    model.record_diagnostics = False
    result = json.load(open(os.path.join(a.run, "result.json"))) if os.path.exists(
        os.path.join(a.run, "result.json")) else {}
    path = os.path.join(a.run, "maxwell_eval.json")
    out = json.load(open(path)) if os.path.exists(path) else {}
    out.update(run=a.run, model=trained, native=task.native, edge_protocol=_edge_protocol(task))
    idx = task.meta["paper_eval"]
    t0 = time.time()
    if trained == "rhmp":
        out["recovery"] = recovery_block(model, task, a.run, device)
        model.record_diagnostics = False
        print("recovery", json.dumps(_jsonable({k: v for k, v in out["recovery"].items() if not k.startswith("curve")})),
              flush=True)
    if not a.recovery_only:
        out["test100"] = evaluate(model, task, idx, EVAL_BATCH, device)
        print("test100", json.dumps(out["test100"]), flush=True)
        out["divergence"] = divergence_block(model, task, trained, idx, device)
        print("divergence", json.dumps(out["divergence"]), flush=True)
        angles = [float(v) for v in a.angles.split(",") if v]
        out["rotation"] = rotation_block(model, untransformed, trained, idx, device, angles)
        out["rotation"]["0"] = {k: out["test100"][k] for k in ("R2", "NRMSE", "SSIM", "Pearson", "MSE", "N")}
        print("rotation", json.dumps(out["rotation"]), flush=True)
        out["timing"] = timing_block(model, task, idx, device, result)
        print("timing", json.dumps(_jsonable(out["timing"])), flush=True)
        if not _edge_protocol(task):
            out["transfer"] = transfer_block(model, task, trained, device, a.transfer_iters)
            print("transfer", json.dumps(_jsonable(out["transfer"])), flush=True)
        if a.dump:
            out["dumps"] = dump_block(model, task, untransformed, a.run, trained, device)
    out["seconds"] = time.time() - t0
    with open(path, "w") as f:
        json.dump(_jsonable(out), f, indent=1)
    print(f"written {path} ({out['seconds']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
