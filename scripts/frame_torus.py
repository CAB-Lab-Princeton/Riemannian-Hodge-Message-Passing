#!/usr/bin/env python3
"""Frame robustness of trained T2v runs: test100 metrics on the original test set and on rigidly rotated copies.

    python3 scripts/frame_torus.py --runs runs/<T2v run directories> --out runs/frame_T2v.json

The rotation helpers of this file (``rotation_axis``, ``rot3``, ``rebuilt_complex``) are also used by
``scripts/eval_torus_aniso.py`` for the rotated evaluations of the paper's torus task (T2a).

Same construction as the frame evaluation of the other paper tasks (``scripts/frame_eval``): the physical configuration
is rotated by ``Q`` (Rodrigues) about the unit axis ``u`` drawn by ``rotation_axis(0)`` (``u = (0.320686, 0.540963,
-0.777508)``) through the vertex centroid, at the angles 0, 30, 90, 137 and 360 degrees (360 = identity check); no
retraining.  The vertex positions rotate (complex rebuilt from the rotated positions, same cell order and orientation);
the scalar fields ``f`` and ``u`` do not change; the velocity rotates, ``V -> V Q^T``.  RHMP reads the velocity as an
edge cochain, whose values ``(V_s +- V_d)/2 . (x_d - x_s)`` do not change (``rhmp.robustness.transformed_task``); the
baselines read the node features ``(f, Vx, Vy, Vz)``, regenerated from the rotated velocity and normalised with the
run's (original-frame) statistics.  MPSN / SCCNN / Clifford-SMPN keep their v1 edge protocol
(``rhmp.baselines.registry.prepare_task`` on the rotated view).  Metrics: ``rhmp.metrics.summarize`` on the 100
paper-evaluation samples, as ``rhmp.train.evaluate`` computes ``test100``; the recomputed angle-0 metrics are compared
with ``result.json``, and the largest change of any prediction relative to angle 0 is recorded (RHMP: round-off;
360 degrees: identity).
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rhmp import metrics as M  # noqa: E402
from rhmp.tasks import load_task  # noqa: E402
from rhmp.tasks.paper import t2v_file, t2v_inputs  # noqa: E402
from rhmp.train import _renormalize, evaluate, predict  # noqa: E402

ANGLES = (0, 30, 90, 137, 360)
AXIS_REF = np.array([0.320686, 0.540963, -0.777508])
KEYS = ("R2", "SSIM", "Pearson", "NRMSE", "MSE", "MAE")


def rotation_axis(seed: int = 0) -> np.ndarray:
    """First normalised ``default_rng(seed + j).normal(size=3)`` whose components all have modulus in [0.3, 0.8]."""
    for j in range(1000):
        u = np.random.default_rng(seed + j).normal(size=3)
        u = u / np.linalg.norm(u)
        if np.all(np.abs(u) >= 0.3) and np.all(np.abs(u) <= 0.8):
            return u
    raise RuntimeError("no admissible axis")


def rot3(axis: np.ndarray, deg: float) -> np.ndarray:
    a = math.radians(deg)
    u = np.asarray(axis, dtype=np.float64)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return math.cos(a) * np.eye(3) + math.sin(a) * ux + (1 - math.cos(a)) * np.outer(u, u)


def rebuilt_complex(K, Q: np.ndarray, c: np.ndarray):
    """Complex of the rotated mesh ``Q (x - c) + c``, checked to keep the cell order and orientation."""
    from rhmp.robustness import remap_complex
    K_new, maps = remap_complex(K, Q=Q, shift=c - Q @ c)
    for k, (idx, sgn) in maps.items():
        if not torch.equal(idx.cpu(), torch.arange(len(idx))) or (sgn is not None and not bool((sgn.cpu() == 1).all())):
            raise RuntimeError(f"rotated complex changed the cells of degree {k}")
    return K_new


_TASKS: dict = {}


def load(run_dir: str, device):
    from rhmp.model import RHMP
    conf = json.load(open(os.path.join(run_dir, "config.json")))
    a = conf["args"]
    name = a.get("model") or "rhmp"
    key = (a["task"], bool(a["native"]))
    if key not in _TASKS:
        _TASKS[key] = load_task(a["task"], a.get("root"), native=key[1], device=device, star=a.get("star", "cotan"),
                                whitney=False)
    task_raw = dataclasses.replace(_TASKS[key], inputs=dict(_TASKS[key].inputs), meta=dict(_TASKS[key].meta))
    task_raw = _renormalize(task_raw, conf["data"])                  # the run's statistics (identical up to round-off)
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
    ebs = a.get("eval_batch") or a.get("batch") or 64
    return conf, name, model, task_raw, task, ck.get("build"), int(ebs)


def summarize(view, pred, tgt) -> dict:
    r = M.summarize(pred, tgt, view.y_stats.as_tuple(), r2_std=view.meta.get("r2_std"), center=True)
    return {k: float(r[k]) for k in KEYS}


@torch.no_grad()
def evaluate_run(run_dir: str, device, axis, raw, cache) -> dict:
    from rhmp.baselines.registry import SPECS, prepare_task
    from rhmp.robustness import transformed_task
    t0 = time.time()
    conf, name, model, task_raw, task, build, ebs = load(run_dir, device)
    res = json.load(open(os.path.join(run_dir, "result.json")))
    pe = task.meta["paper_eval"]
    rec = {"run": run_dir, "model": name, "native": bool(task.native), "params": res.get("params"),
           "stored_test100": {k: res["test100"][k] for k in KEYS}}
    m0 = evaluate(model, task, pe, ebs, device)
    rec["recomputed_test100"] = {k: float(m0[k]) for k in KEYS}
    rec["reproduction_max_abs_diff"] = max(abs(rec["recomputed_test100"][k] - rec["stored_test100"][k])
                                           for k in ("R2", "SSIM", "Pearson", "NRMSE"))
    pred0, _ = predict(model, task, pe, ebs, device)
    K0 = task_raw.K
    c0 = K0.pos.double().cpu().numpy().mean(0)
    sel = pe.to(task_raw.target.device)
    out = {}
    for deg in ANGLES:
        Q = rot3(axis, deg)
        if name == "rhmp":
            view = transformed_task(task, pe, Q=Q, shift=c0 - Q @ c0)
        else:
            ck = (deg, "K")
            if ck not in cache:
                cache[ck] = rebuilt_complex(K0, Q, c0)
            Vr = raw["V"][pe.cpu()].double() @ torch.as_tensor(Q).T
            X = raw["X"][pe.cpu()]
            xin = t2v_inputs(X, Vr.float(), cache[ck], native=False)[0]
            xn = task_raw.x_stats[0].normalize(xin.to(sel.device)).contiguous()
            if deg == 0:
                d_in = float((xn - task_raw.inputs[0].index_select(0, sel)).abs().max())
            view = dataclasses.replace(task_raw, K=cache[ck], inputs={0: xn},
                                       target=task_raw.target.index_select(0, sel).contiguous(),
                                       split=(torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long),
                                              torch.arange(len(sel))), meta=dict(task_raw.meta), output_map=None)
            view = prepare_task(name, view, build)
            if not SPECS[name].uses_output_map and view.output_map is not None:
                view = dataclasses.replace(view, output_map=None)
        p, t = predict(model, view, view.split[2], ebs, device)
        m = summarize(view, p, t)
        m["max_abs_dpred_vs_angle0"] = float((p.double() - pred0.double()).abs().max())
        out[str(deg)] = m
    rec["angles"] = out
    if name != "rhmp":
        rec["regenerated_inputs_vs_loaded_maxabs"] = d_in
    rec["seconds"] = time.time() - t0
    return rec


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True                # as rhmp.train (stored results)
        torch.backends.cudnn.allow_tf32 = True
    axis = rotation_axis(0)
    if np.abs(axis - AXIS_REF).max() > 1e-6:
        raise RuntimeError(f"rotation axis {axis} differs from the reference {AXIS_REF}")
    import pickle

    from rhmp.tasks import resolve_root
    droot = resolve_root(None)
    with open(os.path.join(droot, "T2_torus_advection_diffusion.pkl"), "rb") as f:
        d = pickle.load(f)
    raw = dict(X=torch.as_tensor(np.asarray(d["X_data"], dtype=np.float32)).unsqueeze(-1), V=t2v_file(droot)["V"].float())
    results = {"axis": axis.tolist(), "angles_deg": list(ANGLES), "centre": "vertex centroid", "runs": {}}
    if os.path.exists(a.out):
        results = json.load(open(a.out))
    cache: dict = {}
    for rd in a.runs:
        rd = os.path.abspath(rd)
        if not os.path.exists(os.path.join(rd, "result.json")):
            continue
        r = evaluate_run(rd, device, axis, raw, cache)
        results["runs"][os.path.basename(rd)] = r
        print(f"{os.path.basename(rd):18s} stored R2 {r['stored_test100']['R2']:.5f} recomputed "
              f"{r['recomputed_test100']['R2']:.5f} | R2 " + " ".join(f"{k}:{v['R2']:.4f}" for k, v in r["angles"].items())
              + f" | max dpred 137: {r['angles']['137']['max_abs_dpred_vs_angle0']:.2e} 360: "
              f"{r['angles']['360']['max_abs_dpred_vs_angle0']:.2e} ({r['seconds']:.0f}s)", flush=True)
        with open(a.out, "w") as f:
            json.dump(results, f, indent=1)
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
