#!/usr/bin/env python3
"""Zero-shot generalisation of trained T2a runs (anisotropic transport on the torus), next to the in-distribution
``test100`` numbers.

    python3 scripts/eval_torus_aniso.py --runs results/torus_transport/rhmp_s42 --out runs/eval_T2a.json [--recovery 16]

Reproduces the zero-shot torus table of the paper (``results/evaluations/torus_zero_shot.json``; with ``--frozen-auto``
also ``torus_zero_shot_frozen_auto.json``).

Every run is rebuilt from its best checkpoint as the trainer evaluates it (task loaded with the run's arguments, data
re-expressed in the run's normalisation, the v1 edge protocol of MPSN / SCCNN / Clifford-SMPN through
``rhmp.baselines.registry.prepare_task``) and evaluated with ``rhmp.metrics.summarize`` on:

* ``test100``: the paper-evaluation samples (recomputed; check against ``result.json``);
* ``alpha2``, ``alpha2.5``: the 450 test samples with the toroidal amplitude raised to 2 and 2.5 (training 0.3..1.5);
* ``ratio300``, ``ratio1000``: anisotropy ratio 300 and 1000 (training 1..100), same geometric-mean diffusivity;
* ``rollout``: the model applied twice (its prediction fed back as the initial field) against ``u(2T)``;
* ``fine``, ``coarse``: the same physical instances on the 2x finer (6844 vertices) and 2x coarser (450 vertices) torus;
* ``rot30``, ``rot90``, ``rot137`` (and ``rot360`` = identity check): the test100 configuration rotated about the axis
  ``u = (0.320686, 0.540963, -0.777508)`` through the centroid (``scripts/frame_torus.py``): positions, velocity and
  field direction rotate (the baselines' node vectors ``V``, ``b``); RHMP's velocity cochains, recomputed from the
  rotated positions (stored in float32) and vectors, agree with angle 0 to round-off; descriptors and all scalars
  are unchanged;
* ``mass``: ``|sum_i A_i (pred_i - f_i)| / sum_i A_i |f_i|`` on the test samples (the reference conserves it to 1e-14).
* ``val_base``, ``val_fine``, ``val_coarse`` (``--val-meshes``, default ``datasets/v2/T2a_valmesh.pt`` when present):
  the 450 validation instances on the training torus and re-solved on the fine and coarse tori, so that mesh
  transfer is not judged on test data only.

``--frozen-units`` (RHMP runs): ``test450`` and the mesh sets again with the normalisers of the training mesh
(``model.freeze_units(K_train)`` on the trained checkpoint, no retraining; ``RHMPConfig.frozen_units``), under
``rec['frozen_units']``; on the training mesh the output is unchanged (``test450_max_abs_dpred``).
``--frozen-operators``: the same with operators only (``RHMPConfig.frozen_descriptors=False``: training-mesh star means
and block bounds, descriptors relative to each mesh), under ``rec['frozen_units_operators']``.  ``fine_t100`` /
``coarse_t100``: the fine / coarse metrics on the 100 paper-evaluation instances of the test split.  ``--sets``
restricts the evaluation to the named sets.  ``--frozen-auto``: ``frozen_descriptors='auto'`` under
``rec['frozen_units_auto']``, with the decision of every relative descriptor column on the base, fine and coarse
tori (``columns``: the mesh's median in training units, the training range, inside or not).

``--recovery N`` (RHMP runs with the tensor metric): learned per-face tensors ``sigma_f`` of every layer
(``RHMP.metric_fields``) against the true diffusion tensors on the first ``N`` test samples: angle between the principal
directions (degrees; tangent plane of the face) and the correlation of the log eigenvalue ratios.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from frame_torus import AXIS_REF, rebuilt_complex, rot3, rotation_axis  # noqa: E402
from rhmp import metrics as M  # noqa: E402
from rhmp.tasks import load_task  # noqa: E402
from rhmp.tasks.torus import t2a_complex, t2a_file, t2a_inputs  # noqa: E402
from rhmp.train import _renormalize, evaluate, predict  # noqa: E402

KEYS = ("R2", "SSIM", "Pearson", "NRMSE", "MSE")
_TASKS: dict = {}


def load(run_dir: str, device):
    from rhmp.model import RHMP
    conf = json.load(open(os.path.join(run_dir, "config.json")))
    a = conf["args"]
    name = a.get("model") or "rhmp"
    tensor = conf.get("model", {}).get("metric_type") == "tensor" if name == "rhmp" else False
    key = (a["task"], bool(a["native"]), tensor)
    if key not in _TASKS:
        _TASKS.clear()
        _TASKS[key] = load_task(a["task"], a.get("root"), native=key[1], device=device, star=a.get("star", "cotan"),
                                whitney=tensor)
    base = _TASKS[key]
    task_raw = _renormalize(dataclasses.replace(base, inputs=dict(base.inputs), meta=dict(base.meta)), conf["data"])
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    if name == "rhmp":
        model, task = RHMP.from_checkpoint(ck, map_location=device), task_raw
    else:
        from rhmp.baselines.registry import SPECS, from_checkpoint, prepare_task
        task = prepare_task(name, task_raw, ck.get("build"))
        model = from_checkpoint(ck, task, map_location=device)
        if not SPECS[name].uses_output_map and task.output_map is not None:
            task = dataclasses.replace(task, output_map=None)
    model.eval()
    if hasattr(model, "record_diagnostics"):
        model.record_diagnostics = False
    ebs = int(a.get("eval_batch") or a.get("batch") or 64)
    return conf, name, model, task_raw, task, ck.get("build"), ebs, tensor


def view_of(name: str, task_raw, build, K, inputs: dict, target_node: torch.Tensor):
    """Test view (node targets; prepare_task for edge-protocol baselines) on complex ``K``."""
    from rhmp.baselines.registry import SPECS, prepare_task
    n = target_node.shape[0]
    v = dataclasses.replace(task_raw, K=K, inputs=inputs, target=target_node.contiguous(), meta=dict(task_raw.meta),
                            split=(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.arange(n)),
                            output_map=None, extra_tests={})
    if name != "rhmp":
        v = prepare_task(name, v, build)
        if not SPECS[name].uses_output_map and v.output_map is not None:
            v = dataclasses.replace(v, output_map=None)
    return v


def summarize(view, P, T) -> dict:
    r = M.summarize(P, T, view.y_stats.as_tuple(), r2_std=view.meta.get("r2_std"), center=True)
    return {k: float(r[k]) for k in KEYS}


def to_nodes(view, P: torch.Tensor, edge: bool) -> torch.Tensor:
    P = view.y_stats.denormalize(P)
    if not edge:
        return P
    K = view.K
    e = K.cells[1].long().to(P.device)
    n0 = int(K.n[0])
    acc = torch.zeros(P.shape[0], n0, P.shape[-1], dtype=P.dtype, device=P.device)
    acc.index_add_(1, e[:, 0], P).index_add_(1, e[:, 1], P)
    cnt = torch.bincount(e.reshape(-1), minlength=n0).to(P.dtype).clamp_min(1).view(1, -1, 1)
    return acc / cnt


def norm_inputs(task_raw, raw: dict, device) -> dict:
    return {k: task_raw.x_stats[k].normalize(v.to(device)).contiguous() for k, v in raw.items()}


def run_set(name, model, task_raw, build, ebs, K, raw, Ynode, device):
    """Normalised predictions and metrics of one zero-shot set (evaluation batch scaled down with the mesh size and
    capped at ``EVAL_BATCH_MAX``; predictions are per sample, so the batch size does not change them)."""
    tgt = task_raw.y_stats.normalize(Ynode.to(device).float().unsqueeze(-1))
    view = view_of(name, task_raw, build, K, norm_inputs(task_raw, raw, device), tgt)
    bs = max(1, min(ebs, EVAL_BATCH_MAX, int(ebs * task_raw.K.n[0] / K.n[0])))
    P, T = predict(model, view, view.split[2], bs, device)
    return view, P, T, summarize(view, P, T)


EVAL_BATCH_MAX = 16          # shared GPUs: small evaluation batches


@torch.no_grad()
def recovery(model, task_raw, blob, ids, device) -> dict:
    """Learned per-face tensors of every layer vs the true diffusion tensors (principal direction, log ratio)."""
    K = task_raw.K
    raw = t2a_inputs(blob["X"][ids], blob["V"][ids], blob["b"][ids], blob["edge_cos2"][ids][:, PERM["base"]],
                     blob["edge_logp"][ids][:, PERM["base"]], blob["Dpar"][ids], blob["Dperp"][ids], K, True)
    x = {k: v.transpose(0, 1).contiguous() for k, v in norm_inputs(task_raw, raw, device).items()}
    fields = model.metric_fields(x, K)
    F = K.cells[2].long()
    P = K.pos.double()
    nf = torch.cross(P[F[:, 1]] - P[F[:, 0]], P[F[:, 2]] - P[F[:, 0]], dim=1)
    nf = nf / nf.norm(dim=1, keepdim=True)
    bv = blob["b"][ids].double().to(device)                                       # (S, n0, 3)
    bf = bv[:, F].mean(2)                                                          # (S, n2, 3) face mean direction
    bf = bf - (bf * nf).sum(-1, keepdim=True) * nf
    bf = bf / bf.norm(dim=-1, keepdim=True)
    true_lr = torch.log(blob["Dpar"][ids].double() / blob["Dperp"][ids].double()).to(device)   # (S,)
    out = {}
    for li, fl in enumerate(fields):
        if 1 not in fl.get("sigma", {}):
            continue
        sig = fl["sigma"][1].double()                                              # (n2, S, 3, 3)
        Pn = torch.eye(3, dtype=torch.float64, device=device) - nf[:, :, None] * nf[:, None, :]
        st = Pn[:, None] @ sig @ Pn[:, None]                                       # tangent part
        ev, evec = torch.linalg.eigh(st.cpu())                                     # ascending; one eigenvalue ~ 0
        ev, evec = ev.to(device), evec.to(device)                                  # (cuSOLVER: batch too large)
        lmax, lmid = ev[..., 2], ev[..., 1]
        vmax = evec[..., :, 2]                                                     # (n2, S, 3)
        cosang = (vmax * bf.permute(1, 0, 2)).sum(-1).abs().clamp(max=1.0)
        ang = torch.rad2deg(torch.arccos(cosang))
        lr = torch.log(lmax / lmid.clamp_min(1e-30))                               # learned log ratio per face
        lr_s = lr.median(0).values                                                 # per sample
        c = float(np.corrcoef(lr_s.cpu().numpy(), true_lr.cpu().numpy())[0, 1]) if len(ids) > 2 else float("nan")
        out[f"layer{li}"] = dict(angle_deg_median=float(ang.median()), angle_deg_mean=float(ang.mean()),
                                 learned_log_ratio_median=float(lr.median()), true_log_ratio_median=float(true_lr.median()),
                                 corr_log_ratio_per_sample=c)
    return out


PERM: dict = {}


@torch.no_grad()
def _mesh_sets(name, model, task_raw, build, ebs, conf, blob, base, wants, device, pos_pe=None) -> tuple:
    """``test450`` and the mesh-transfer sets (``fine``, ``coarse``; with the validation file ``val_base``,
    ``val_fine``, ``val_coarse``) of the model in its current state: ``(sets, test450 view, test450 predictions)``;
    with ``pos_pe`` (positions of the paper-evaluation samples in the test arrays) also ``fine_t100`` / ``coarse_t100``,
    the same metrics on those 100 instances."""
    native = bool(task_raw.native)
    Kb = task_raw.K

    def raw_of(K, d):
        return t2a_inputs(d["X"], d["V"], d["b"], d["c2"], d["lp"], d["Dpar"], d["Dperp"], K, native)
    sets: dict = {}
    te = blob["test_index"].long()
    view, P1, _, sets["test450"] = run_set(name, model, task_raw, build, ebs, Kb, raw_of(Kb, base), blob["Y"][te],
                                           device)
    latent = name == "rhmp" and conf.get("model", {}).get("latent_dims")
    for mesh in ("fine", "coarse"):
        if not wants(mesh):
            continue
        if latent:
            sets[mesh] = None                                                       # latent fields are tied to one mesh
            continue
        d = blob[mesh]
        Km, pm = MESH_K[mesh]
        dm = dict(X=d["X"], V=d["V"], b=d["b"], c2=d["edge_cos2"][:, pm], lp=d["edge_logp"][:, pm],
                  Dpar=base["Dpar"], Dperp=base["Dperp"])
        vmesh, Pm, Tm, sets[mesh] = run_set(name, model, task_raw, build, ebs, Km, raw_of(Km, dm), d["Y"], device)
        if pos_pe is not None:
            sub = torch.as_tensor(pos_pe, device=Pm.device)
            sets[f"{mesh}_t100"] = summarize(vmesh, Pm[sub], Tm[sub])
    vm = VAL_MESHES.get("data")
    if vm is not None:
        va = vm["val_index"].long()
        if not torch.equal(task_raw.split[1].cpu(), va):
            raise RuntimeError("validation split differs from the generator's")
        pb = PERM["base"]
        vb = dict(X=blob["X"][va], V=blob["V"][va], b=blob["b"][va], c2=blob["edge_cos2"][va][:, pb],
                  lp=blob["edge_logp"][va][:, pb], Dpar=blob["Dpar"][va], Dperp=blob["Dperp"][va])
        if wants("val_base"):
            sets["val_base"] = run_set(name, model, task_raw, build, ebs, Kb, raw_of(Kb, vb), blob["Y"][va], device)[3]
        for mesh in ("fine", "coarse"):
            if not wants(f"val_{mesh}"):
                continue
            if latent:
                sets[f"val_{mesh}"] = None
                continue
            d = vm[mesh]
            Km, pm = MESH_K[mesh]
            dm = dict(X=d["X"], V=d["V"], b=d["b"], c2=d["edge_cos2"][:, pm], lp=d["edge_logp"][:, pm],
                      Dpar=vb["Dpar"], Dperp=vb["Dperp"])
            sets[f"val_{mesh}"] = run_set(name, model, task_raw, build, ebs, Km, raw_of(Km, dm), d["Y"], device)[3]
    return sets, view, P1


def auto_columns(model, meshes: dict) -> dict:
    """``frozen_descriptors='auto'`` decision of every relative descriptor column on every mesh: the mesh's median in
    training units (``log median - log median_train``), the training range and whether it lies inside (training
    units) or not (relative to the mesh)."""
    from rhmp.geometry import GEO_FEATURE_NAMES, GEO_RELATIVE_COLUMNS
    from rhmp.layers import unit_reference
    out = {}
    for name, K in meshes.items():
        r = unit_reference(K, model.cfg.scaling, torch.float64)
        out[name] = {}
        for k in range(int(K.dim) + 1):
            for j in GEO_RELATIVE_COLUMNS[k]:
                sh = float(r["geo_log_median"][k][j] - getattr(model, f"frozen_geo_{k}")[j].cpu())
                lo, hi = (float(getattr(model, f"frozen_geo_{b}_{k}")[j]) for b in ("min", "max"))
                out[name][f"{k}:{GEO_FEATURE_NAMES[k][j]}"] = dict(shift=sh, min=lo, max=hi, inside=lo <= sh <= hi)
    return out


@torch.no_grad()
def evaluate_run(run_dir: str, device, blob: dict, axis, n_rec: int, want: set | None = None,
                 frozen_units: bool = False, frozen_operators: bool = False, frozen_auto: bool = False) -> dict:
    t0 = time.time()
    conf, name, model, task_raw, task, build, ebs, tensor = load(run_dir, device)
    native = bool(task_raw.native)
    res_file = json.load(open(os.path.join(run_dir, "result.json")))
    te = blob["test_index"].long()
    if not torch.equal(task_raw.split[2].cpu(), te):
        raise RuntimeError("test split differs from the generator's")
    edge = bool(task.meta.get("v1_edge_protocol"))
    pe = task.meta["paper_eval"]

    def wants(k):
        return want is None or k in want
    rec = {"run": run_dir, "model": name, "native": native, "params": res_file.get("params"), "edge_protocol": edge,
           "seed": conf.get("args", {}).get("seed"), "stored_test100": {k: res_file["test100"][k] for k in KEYS},
           "best_val_R2": res_file.get("best_val_R2"), "s_per_epoch": res_file.get("s_per_epoch")}
    m0 = evaluate(model, task, pe, ebs, device)
    rec["test100"] = {k: float(m0[k]) for k in KEYS}
    rec["reproduction_max_abs_diff"] = max(abs(rec["test100"][k] - rec["stored_test100"][k])
                                           for k in ("R2", "SSIM", "Pearson", "NRMSE"))
    Kb = task_raw.K
    pb = PERM["base"]
    sel = te
    base = dict(X=blob["X"][sel], V=blob["V"][sel], b=blob["b"][sel], c2=blob["edge_cos2"][sel][:, pb],
                lp=blob["edge_logp"][sel][:, pb], Dpar=blob["Dpar"][sel], Dperp=blob["Dperp"][sel])

    def raw_of(K, d):
        return t2a_inputs(d["X"], d["V"], d["b"], d["c2"], d["lp"], d["Dpar"], d["Dperp"], K, native)
    # in-distribution test split (all 450) through the same path (identity check of the construction) and the
    # mesh-transfer sets
    pos_pe = [int(np.where(te.numpy() == int(i))[0][0]) for i in pe.tolist()]
    sets, view, P1 = _mesh_sets(name, model, task_raw, build, ebs, conf, blob, base, wants, device, pos_pe)
    Pnode = to_nodes(view, P1, edge)
    A = Kb.star[0].double().to(device)
    f = base["X"].double().to(device)
    mp = (Pnode[..., 0].double() * A).sum(1)
    mf = (f * A).sum(1)
    my = (blob["Y"][sel].double().to(device) * A).sum(1)
    ref = (f.abs() * A).sum(1)
    rec["mass"] = dict(pred_vs_input_mean=float(((mp - mf).abs() / ref).mean()),
                       target_vs_input_mean=float(((my - mf).abs() / ref).mean()),
                       pred_vs_target_mean=float(((mp - my).abs() / ref).mean()))
    for al, d in blob["alpha"].items():
        if wants(f"alpha{al}"):
            dd = dict(base, V=d["V"])
            sets[f"alpha{al}"] = run_set(name, model, task_raw, build, ebs, Kb, raw_of(Kb, dd), d["Y"], device)[3]
    for ra, d in blob["ratio"].items():
        if wants(f"ratio{ra}"):
            dd = dict(base, c2=d["edge_cos2"][:, pb], lp=d["edge_logp"][:, pb], Dpar=d["Dpar"], Dperp=d["Dperp"])
            sets[f"ratio{ra}"] = run_set(name, model, task_raw, build, ebs, Kb, raw_of(Kb, dd), d["Y"], device)[3]
    if wants("rollout"):
        dd = dict(base, X=Pnode[..., 0].float().cpu())                             # roll-out: feed the prediction back
        sets["rollout"] = run_set(name, model, task_raw, build, ebs, Kb, raw_of(Kb, dd), blob["rollout"]["Y2"],
                                  device)[3]
    # rotations of the test100 configuration (positions, velocity and direction vectors rotate)
    sub = {k: (v[pos_pe] if torch.is_tensor(v) else v) for k, v in base.items()}
    c0 = Kb.pos.double().cpu().numpy().mean(0)
    pred0 = None
    for deg in (0, 30, 90, 137, 360):
        if not wants("rot"):
            break
        Q = rot3(axis, deg)
        key = ("rot", deg)
        if key not in ROT_K:
            ROT_K[key] = rebuilt_complex(Kb, Q, c0)
        Qt = torch.as_tensor(Q, dtype=torch.float64)
        dr = dict(sub, V=(sub["V"].double() @ Qt.T).float(), b=(sub["b"].double() @ Qt.T).float())
        v, P, T, m = run_set(name, model, task_raw, build, ebs, ROT_K[key], raw_of(ROT_K[key], dr),
                             blob["Y"][te[pos_pe]], device)
        if pred0 is None:
            pred0 = P
        m["max_abs_dpred_vs_angle0"] = float((P.double() - pred0.double()).abs().max())
        sets[f"rot{deg}"] = m
    rec["sets"] = sets
    if tensor and n_rec > 0 and wants("recovery"):
        rec["recovery"] = recovery(model, task_raw, blob, te[:n_rec], device)
    if frozen_units and name == "rhmp":
        # the same model with the normalisers of its training mesh (no retraining): the output on the training mesh is
        # unchanged (test450 identity check), the mesh-transfer sets change
        model.freeze_units(task_raw.K)
        fsets, fview, fP = _mesh_sets(name, model, task_raw, build, ebs, conf, blob, base, wants, device, pos_pe)
        rec["frozen_units"] = dict(sets=fsets, test450_max_abs_dpred=float((fP.double() - P1.double()).abs().max()))
    if frozen_operators and name == "rhmp":
        # operators only (RHMPConfig.frozen_descriptors=False): training-mesh star means and block bounds, the
        # descriptors relative to each mesh
        model.cfg.frozen_descriptors = False
        model.freeze_units(task_raw.K)
        osets, oview, oP = _mesh_sets(name, model, task_raw, build, ebs, conf, blob, base, wants, device, pos_pe)
        rec["frozen_units_operators"] = dict(sets=osets,
                                             test450_max_abs_dpred=float((oP.double() - P1.double()).abs().max()))
    if frozen_auto and name == "rhmp":
        # frozen_descriptors='auto': per relative column training units while the mesh's median lies in the training
        # range, else relative to the mesh; star means and block bounds frozen
        model.cfg.frozen_descriptors = "auto"
        model.freeze_units(task_raw.K)
        asets, aview, aP = _mesh_sets(name, model, task_raw, build, ebs, conf, blob, base, wants, device, pos_pe)
        cols = auto_columns(model, {"base": task_raw.K, "fine": MESH_K["fine"][0], "coarse": MESH_K["coarse"][0]})
        rec["frozen_units_auto"] = dict(sets=asets, columns=cols,
                                        test450_max_abs_dpred=float((aP.double() - P1.double()).abs().max()))
    rec["seconds"] = time.time() - t0
    return rec


MESH_K: dict = {}
ROT_K: dict = {}
VAL_MESHES: dict = {}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--recovery", type=int, default=16)
    ap.add_argument("--sets", default=None, help="comma-separated subset of the sets (test100 and test450 always): "
                    "alpha2,alpha2.5,ratio300,ratio1000,rollout,fine,coarse,val_base,val_fine,val_coarse,rot,recovery")
    ap.add_argument("--val-meshes", default=None, metavar="PATH",
                    help="validation instances on the fine / coarse tori (gen_T2a.py --val-meshes; default "
                         "datasets/v2/T2a_valmesh.pt when present)")
    ap.add_argument("--frozen-units", action="store_true",
                    help="RHMP runs: the mesh sets again with the normalisers of the training mesh (frozen_units)")
    ap.add_argument("--frozen-operators", action="store_true",
                    help="RHMP runs: the mesh sets with frozen operators only (frozen_units, frozen_descriptors=False)")
    ap.add_argument("--frozen-auto", action="store_true",
                    help="RHMP runs: the mesh sets with frozen_units and frozen_descriptors='auto' (column decisions too)")
    a = ap.parse_args(argv)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    axis = rotation_axis(0)
    if np.abs(axis - AXIS_REF).max() > 1e-6:
        raise RuntimeError("rotation axis differs from the reference")
    from rhmp.tasks import resolve_root
    root = resolve_root(None)
    blob = t2a_file(root)
    _, PERM["base"] = t2a_complex(blob["points"], blob["faces"], blob["edges"], "cotan", "cpu")
    for mesh in ("fine", "coarse"):
        d = blob[mesh]
        MESH_K[mesh] = t2a_complex(d["points"], d["faces"], d["edges"], "cotan", device)
    vpath = a.val_meshes or os.path.join(root, "v2", "T2a_valmesh.pt")
    if a.val_meshes or os.path.exists(vpath):
        vm = torch.load(vpath, map_location="cpu", weights_only=False)
        for mesh in ("fine", "coarse"):                 # the same tori as the test sets (complexes and edge maps shared)
            for key in ("points", "faces", "edges"):
                if not torch.equal(vm[mesh][key], blob[mesh][key]):
                    raise RuntimeError(f"{vpath}: the {mesh} torus differs from the one of the test sets")
        VAL_MESHES["data"] = vm
    want = None if not a.sets else set(a.sets.split(","))
    results = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for rd in a.runs:
        rd = os.path.abspath(rd)
        if not os.path.exists(os.path.join(rd, "result.json")):
            continue
        r = evaluate_run(rd, device, blob, axis, a.recovery, want, a.frozen_units, a.frozen_operators, a.frozen_auto)
        results[os.path.basename(rd)] = r
        s = r["sets"]

        def g(k, s=s):
            return "-" if s.get(k) is None else f"{s[k]['R2']:.4f}"
        rot = (f"rot30 {g('rot30')} rot90 {g('rot90')} rot137 {g('rot137')} "
               f"(360 d {s['rot360']['max_abs_dpred_vs_angle0']:.1e}) | " if "rot360" in s else "")
        print(f"{os.path.basename(rd):18s} test100 {r['test100']['R2']:.4f} (d {r['reproduction_max_abs_diff']:.1e}) | "
              f"a2 {g('alpha2')} a2.5 {g('alpha2.5')} | r300 {g('ratio300')} r1000 {g('ratio1000')} | roll {g('rollout')} | "
              f"fine {g('fine')} ({g('fine_t100')}) coarse {g('coarse')} ({g('coarse_t100')}) | val {g('val_base')} "
              f"{g('val_fine')} {g('val_coarse')} | {rot}"
              f"mass {r['mass']['pred_vs_input_mean']:.2e} [{r['seconds']:.0f}s]", flush=True)
        for key in ("frozen_units", "frozen_units_operators", "frozen_units_auto"):
            if key in r:
                fs = r[key]["sets"]
                print(f"{'':18s} {key}: test450 {g('test450', fs)} (d {r[key]['test450_max_abs_dpred']:.1e}) | fine "
                      f"{g('fine', fs)} ({g('fine_t100', fs)}) coarse {g('coarse', fs)} ({g('coarse_t100', fs)}) | val "
                      f"{g('val_base', fs)} {g('val_fine', fs)} {g('val_coarse', fs)}", flush=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        if device.type == "cuda":
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
