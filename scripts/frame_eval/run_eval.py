"""Frame dependence of trained models: test100 metrics of every run on the original and on rigidly rotated test meshes.

Reproduces the rigid-rotation rows of the gauge-field tasks of the paper (``results/evaluations/wilson_loop_rotations.json``,
``yang_mills_rotations.json``): every angle is scored with the per-mesh normalisers (default), with those of the
training mesh (``frozen_units``) and with the per-column rule ``frozen_descriptors='auto'`` (``frozen_auto``).

    cd scripts/frame_eval && CUDA_VISIBLE_DEVICES=0 python3 run_eval.py --device cuda --tf32 1 --tasks T6,T7 --models rhmp
    CUDA_VISIBLE_DEVICES=  python3 run_eval.py --device cpu  --tasks T3,T6,T7 --tag cpu_fp32     # all runs, CPU

Runs (read only): ``results/<task>/<model>_s42`` (see ``SPEC``; ``RHMP_RUNS`` selects another run root).  No training
and no write access to the run directories: results go to ``--out-dir`` (default ``runs/frame_eval_out``).

Rotation of the physical configuration: vertex positions are rotated about the mesh centroid
(planar meshes: in-plane rotation; T2 / T3 surfaces: rotation about one fixed unit axis through the centroid);
scalar fields are unchanged; frame-dependent quantities rotate with the mesh (T3 targets; the vector columns of the
legacy node encodings of T6 / T7, recomputed from the unchanged edge cochain on the rotated mesh; the nodal velocity
columns of the legacy AirfRANS inputs); cochain data on edges / faces are unchanged.  Predictions are compared with the
rotated targets by the library metrics (``rhmp.metrics.summarize``, as ``rhmp.train.evaluate`` computes ``test100``).

Native runs (RHMP) use ``rhmp.robustness.transformed_task`` (shared meshes) or the same construction with the task's output
map kept (AirfRANS, variable meshes: the library's variable-mesh branch drops the output map).  Baselines are scored on
legacy node-encoded views of the same rotated meshes, with the v1 edge protocol of mpsn / sccnn / clifford_smpn applied to
the rotated targets by the library's ``prepare_task``.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
import traceback

import numpy as np
import torch

from frame_common import (ANGLES, ROOT, Run, encode_edges_to_nodes, eval_task_samples, flat, jsonable, load_run,
                          max_pred_diff, mesh_centroid, pick, predict_view, rebuilt_complex, rotate_positions,
                          rotate_vector_columns, rotate_vector_target, rotation, rotation_axis, shift_about, summarize_pred, with_view)

RUNS = os.path.abspath(os.path.expanduser(os.environ.get("RHMP_RUNS", os.path.join(ROOT, "results"))))
DROOT = os.path.join(ROOT, "datasets")
BASELINES = ["gcn", "gat", "schnet", "egnn", "mpsn", "sccnn", "gauge_cnn", "gem_cnn", "cw_net", "clifford_smpn",
             "deeponet"]
SPEC = {   # RHMP run and baseline directory (<base>/<model>_s42) of every task, relative to RUNS
    "T3": dict(rhmp="ellipsoid_flow/rhmp_s42", base="ellipsoid_flow", load_kw={}),
    "T6": dict(rhmp="wilson_loop/rhmp_s42", base="wilson_loop", load_kw={}),
    "T7": dict(rhmp="yang_mills/rhmp_s42", base="yang_mills", load_kw={}),
}


# additional RHMP runs to evaluate (task -> name -> run directory relative to RUNS); default: the run of SPEC
RHMP_RUNS: dict = {}
if os.environ.get("RHMP_RUNS_JSON"):                       # {task: {name: run dir}} override (relative to RUNS or absolute)
    import json as _json
    for _t, _r in _json.loads(os.environ["RHMP_RUNS_JSON"]).items():
        RHMP_RUNS[_t] = dict(RHMP_RUNS.get(_t, {}), **_r)


# ======================================================================================================================
# per-task context (raw data needed to regenerate legacy inputs)
# ======================================================================================================================
class Ctx:
    def __init__(self, name: str, axis: np.ndarray, route: str = "regen"):
        self.name, self.axis, self.route = name, axis, route         # route: how legacy frame columns are rotated
        self.A = self.edges = self.pts = None
        self.blob = None
        self.krot: dict = {}                      # (angle, native) -> rotated complex(es), shared by the runs of a task
        if name in ("T6", "T7"):
            nat = torch.load(os.path.join(DROOT, "v2", f"{name}_native.pt"), map_location="cpu", weights_only=False)
            A = nat["theta_edges" if name == "T6" else "A_edges"]
            self.A = A.unsqueeze(-1) if A.dim() == 2 else A                       # (N, n1, c) float32
            self.edges = nat["edges"].numpy().astype(np.int64)
            self.pts = nat["points"].numpy().astype(np.float64)
            del nat
        if name == "AFV_trap":
            self.blob = torch.load(os.path.join(DROOT, "v2", "AFM.pt"), map_location="cpu", weights_only=False)


# ======================================================================================================================
# views
# ======================================================================================================================
def rotated_complexes(ctx: Ctx, task, deg: float):
    """Complex (shared mesh) or list of complexes (one per test100 simulation, each about its own vertex centroid) of the
    rotated meshes; cached per (angle, native flag) so that the runs of a task share them (and their operator caches)."""
    key = (deg, bool(task.native))
    if key not in ctx.krot:
        if task.variable_mesh:
            Ks = []
            for i in task.meta["paper_eval"].tolist():
                K = task.K[int(i)]
                Q = rotation(2, deg)
                Ks.append(rebuilt_complex(K, Q, shift_about(Q, mesh_centroid(K))))
            ctx.krot[key] = Ks
        else:
            Q = rotation(task.spatial_dim, deg, ctx.axis)
            ctx.krot[key] = rebuilt_complex(task.K, Q, shift_about(Q, mesh_centroid(task.K)))
    return ctx.krot[key]


def check_identity(x, y, what, tol, notes: dict):
    d = float((x - y).abs().max())
    notes[what] = max(notes.get(what, 0.0), d)
    if d > tol:
        raise RuntimeError(f"{what}: {d:.3e} > {tol:.1e}")


def legacy_planar_inputs(ctx: Ctx, run: Run, pe: torch.Tensor, Q: np.ndarray, c0: np.ndarray, notes: dict) -> torch.Tensor:
    """Normalised legacy node inputs ``(100, n0, 3c)`` of the rotated planar mesh (T6 / T7), regenerated from the
    unchanged edge cochain with the formula of ``frame_common.encode_edges_to_nodes`` and normalised with the run's
    (original-frame) statistics."""
    task = run.task_raw
    st = task.x_stats[0].to("cpu")
    if ctx.route == "rotate":                       # route 2: rotate the identified vector columns of the stored features
        Xs = st.denormalize(task.inputs[0].index_select(0, pe.to(task.target.device)).cpu().double())
        Xn = st.normalize(rotate_vector_columns(Xs, Q, Xs.shape[-1] // 3).float())
    else:                                           # route 1 (default): regenerate on the rotated mesh
        pts_r = rotate_positions(ctx.pts, Q, c0)
        X = encode_edges_to_nodes(ctx.A[pe.cpu()], ctx.edges, pts_r, len(ctx.pts))          # float64
        Xn = st.normalize(X.float())
    return Xn.to(task.target.device).contiguous()


def baseline_view(run: Run, ctx: Ctx, deg: float, notes: dict):
    from rhmp.baselines.registry import SPECS, prepare_task
    task = run.task_raw                                  # legacy node-encoded task (before the v1 edge protocol)
    pe = task.meta["paper_eval"]
    dim = task.spatial_dim
    Q = rotation(dim, deg, ctx.axis)
    c0 = mesh_centroid(task.K)
    K_new = rotated_complexes(ctx, task, deg)
    sel = pe.to(task.target.device)
    inputs = {k: v.index_select(0, sel) for k, v in task.inputs.items()}
    target = task.target.index_select(0, sel)
    if ctx.name == "T3":                                 # vector target rotates (inputs: scalar stream function)
        target = rotate_vector_target(task, target, Q)
    if ctx.name in ("T6", "T7"):                         # legacy columns: recomputed on the rotated mesh
        inputs[0] = legacy_planar_inputs(ctx, run, pe, Q, c0, notes)
        if deg == 0:
            check_identity(inputs[0], task.inputs[0].index_select(0, sel), "regenerated_inputs_vs_loaded_normalised_maxabs",
                           1e-4, notes)
    view = with_view(task, K_new, inputs, target.contiguous(), len(sel))
    view = prepare_task(run.model_name, view, run.build)                         # v1 edge protocol (mpsn/sccnn/clifford)
    if not SPECS[run.model_name].uses_output_map and view.output_map is not None:
        view = dataclasses.replace(view, output_map=None)
    return view


def afv_view(run: Run, ctx: Ctx, deg: float, notes: dict):
    """AirfRANS (variable meshes): per-mesh rotation about the vertex centroid.  Native: cochain data unchanged
    (theta on edges, node scalars sdf / U / aoa), the task's output map is kept.  Legacy: the nodal velocity columns
    are regenerated from the raw CFD field (float64) rotated by Q, then normalised with the run's statistics."""
    from rhmp.baselines.registry import SPECS
    task = run.task_raw
    ids = [int(i) for i in task.meta["paper_eval"].tolist()]
    Ks = rotated_complexes(ctx, task, deg)
    xs, ys = [], []
    blob = ctx.blob
    Q = rotation(2, deg)
    if not run.is_rhmp:
        p0 = blob["ptr0"]
        U_all, a_all = blob["meta"]["U"], blob["meta"]["aoa"]
    for i in ids:
        ys.append(task.target[i])
        if run.is_rhmp:
            xs.append(task.inputs[i])
            continue
        st = task.x_stats[0].to("cpu")
        Qt = torch.as_tensor(Q, dtype=torch.float64)
        if ctx.route == "rotate":                   # route 2: rotate the velocity columns of the stored normalised input
            x = st.denormalize(task.inputs[i][0].cpu().double())
            x = torch.cat([x[:, :2] @ Qt.T, x[:, 2:]], 1)
        else:                                       # route 1 (default): raw CFD field from the data set, rotated in float64
            sl = slice(int(p0[i]), int(p0[i + 1]))
            u = blob["u"][sl].double()
            sdf = blob["sdf"][sl].double()[:, None]
            one = torch.ones(u.shape[0], 1, dtype=torch.float64)
            x = torch.cat([u @ Qt.T, sdf, one * float(U_all[i]), one * float(a_all[i])], 1)
        xn = st.normalize(x.float())
        if deg == 0 and ctx.route != "rotate":
            check_identity(xn, task.inputs[i][0].cpu(), "regenerated_inputs_vs_loaded_normalised_maxabs", 0.0, notes)
        xs.append({0: xn})
    view = with_view(task, Ks, xs, ys, len(ids))
    if not run.is_rhmp and not SPECS[run.model_name].uses_output_map and view.output_map is not None:
        view = dataclasses.replace(view, output_map=None)
    return view


def rhmp_shared_view(run: Run, ctx: Ctx, deg: float):
    from rhmp.robustness import transformed_task
    task = run.task
    pe = task.meta["paper_eval"]
    Q = rotation(task.spatial_dim, deg, ctx.axis)
    c0 = mesh_centroid(task.K)
    return transformed_task(task, pe, Q=Q, shift=shift_about(Q, c0))


def make_view(run: Run, ctx: Ctx, deg: float, notes: dict):
    if ctx.name == "AFV_trap":
        return afv_view(run, ctx, deg, notes)
    if run.is_rhmp:
        return rhmp_shared_view(run, ctx, deg)
    return baseline_view(run, ctx, deg, notes)


# ======================================================================================================================
# evaluation of one run
# ======================================================================================================================
def raw_iso_r2(view, pred, tgt):
    """Isotropic R2 of node-vector targets in physical units: ``1 - sum |p - t|^2 / sum |t - mean_samples t|^2`` (invariant
    under a joint rotation of predictions and targets)."""
    st = view.y_stats
    p = st.denormalize(pred.double())
    t = st.denormalize(tgt.double())
    return float(1.0 - ((p - t) ** 2).sum() / ((t - t.mean(0, keepdim=True)) ** 2).sum())


def evaluate_run(run: Run, ctx: Ctx, device, angles) -> dict:
    t0 = time.time()
    task = run.task
    pe = task.meta["paper_eval"]
    rec = {"task": run.task_name, "model": run.model_name, "run_dir": run.run_dir, "params": run.result.get("params"),
           "native": bool(run.conf["args"]["native"]), "eval_batch": run.ebs,
           "target_protocol": (run.result.get("model_info") or {}).get("target_protocol"),
           "stored_test100": pick(run.result["test100"]), "complete": run.result.get("complete"),
           "stats_rel_diff_loaded_vs_saved": getattr(run, "stats_rel_diff", None)}
    # --- original test set, exactly as the trainer computes test100 -----------------------------------------------
    m0 = eval_task_samples(run, task, pe, device)
    rec["recomputed_test100"] = pick(m0)
    keys = ("R2", "SSIM", "Pearson", "NRMSE")
    rec["reproduction_abs_diff"] = {k: abs(rec["recomputed_test100"][k] - rec["stored_test100"][k]) for k in keys}
    rec["reproduction_max_abs_diff"] = max(rec["reproduction_abs_diff"].values())
    from rhmp.train import predict
    with torch.no_grad():
        pred0, tgt0 = predict(run.model, task, pe, run.ebs, device)
    vector = task.target_kind == "node_vector"
    angles_out = {}
    for deg in angles:
        notes: dict = {}
        view = make_view(run, ctx, deg, notes)
        with torch.no_grad():
            pred, tgt = predict_view(run, view, device)
        m = summarize_pred(view, pred, tgt)
        out = pick(m)
        out["max_abs_dpred_vs_angle0_not_rotated"] = max_pred_diff(pred, pred0)
        out["max_abs_pred0"] = float(flat(pred0).abs().max())
        if vector and not run.edge_protocol:
            Q = rotation(task.spatial_dim, deg, ctx.axis)
            st = view.y_stats
            pr = st.denormalize(flat(pred))
            p0r = st.denormalize(flat(pred0))
            out["max_abs_dpred_vs_Q_angle0_physical"] = float((pr - p0r @ torch.as_tensor(Q).T).abs().max())
            out["R2_iso_physical"] = raw_iso_r2(view, pred, tgt)
            # metrics of a perfectly equivariant prediction p(alpha) = Q p(0) against the rotated target: the part of the
            # change of the metrics that comes from the metric definitions (per-component normalisation, data range of the
            # flattened components) and not from the model
            Qt = torch.as_tensor(Q, dtype=torch.float64, device=pred0.device)
            p_phys = st.denormalize(pred0.double()) @ Qt.T
            p_id = ((p_phys - st.mean.to(p_phys)) / st.std.to(p_phys)).to(pred0.dtype)
            out["ideal_equivariant"] = pick(summarize_pred(view, p_id, tgt))
        if notes:
            out["notes"] = notes
        angles_out[str(deg)] = out
        del view, pred, tgt
    rec["angles"] = angles_out
    rec["seconds"] = time.time() - t0
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", default="T6,T7")
    ap.add_argument("--models", default=None, help="comma-separated subset (default: rhmp + all baselines present)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tf32", type=int, default=1, help="TF32 matmuls on CUDA (the trainer's default; stored results)")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mem-fraction", type=float, default=0.035, help="CUDA allocator cap (fraction of the GPU)")
    ap.add_argument("--angles", default=",".join(str(a) for a in ANGLES))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "runs", "frame_eval_out"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--route", default="regen", choices=["regen", "rotate"],
                    help="legacy frame columns of the baselines: regenerate on the rotated mesh (default) or rotate "
                         "the identified vector columns of the stored features (cross-check of the regeneration)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    device = torch.device(a.device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(a.tf32)
        torch.backends.cudnn.allow_tf32 = bool(a.tf32)
        torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    tag = a.tag or (f"{device.type}_" + ("tf32" if (device.type == "cuda" and a.tf32) else "fp32"))
    os.makedirs(a.out_dir, exist_ok=True)
    angles = [float(x) if "." in x else int(x) for x in a.angles.split(",")]
    axis, draw = rotation_axis()
    env = {"device": str(device), "tf32": bool(a.tf32) if device.type == "cuda" else False, "threads": a.threads,
           "torch": torch.__version__, "axis": axis.tolist(), "axis_draw_index": draw, "angles": angles,
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu", "tag": tag,
           "route": a.route}
    print(json.dumps(env), flush=True)
    for tname in a.tasks.split(","):
        spec = SPEC[tname]
        t_task = time.time()
        ctx = Ctx(tname, axis, a.route)
        rhmp_runs = RHMP_RUNS.get(tname, {"rhmp": spec["rhmp"]})
        models = list(rhmp_runs) + [m for m in BASELINES if os.path.isdir(os.path.join(RUNS, spec["base"], f"{m}_s42"))]
        models = [m for m in models if m in (a.models.split(",") if a.models else list(rhmp_runs))]
        missing = [m for m in BASELINES if not os.path.isdir(os.path.join(RUNS, spec["base"], f"{m}_s42"))]
        res = {"env": env, "task": tname, "missing_baselines": missing, "runs": {}}
        cache: dict = {}
        path = os.path.join(a.out_dir, f"results_{tname}_{tag}.json")
        for m in models:
            rdir = os.path.join(RUNS, rhmp_runs[m] if m in rhmp_runs else os.path.join(spec["base"], f"{m}_s42"))
            try:
                run = load_run(rdir, tname, "rhmp" if m in rhmp_runs else m, device, cache, spec["load_kw"])
                rec = evaluate_run(run, ctx, device, angles)
                if run.is_rhmp and hasattr(run.model, "freeze_units"):      # normalisers of the training mesh
                    run.model.freeze_units(run.task.K)
                    rec["frozen_units"] = evaluate_run(run, ctx, device, angles)
                    print(f"[{tname}/{m}] frozen_units R2(angle)="
                          f"{ {k: round(v['R2'], 5) for k, v in rec['frozen_units']['angles'].items()} }", flush=True)
                    if hasattr(run.model.cfg, "frozen_descriptors"):        # per-column rule 'auto'
                        run.model.cfg.frozen_descriptors = "auto"
                        rec["frozen_auto"] = evaluate_run(run, ctx, device, angles)
                        print(f"[{tname}/{m}] frozen_auto R2(angle)="
                              f"{ {k: round(v['R2'], 5) for k, v in rec['frozen_auto']['angles'].items()} }", flush=True)
                res["runs"][m] = rec
                r2 = {k: round(v["R2"], 5) for k, v in rec["angles"].items()}
                print(f"[{tname}/{m}] stored R2 {rec['stored_test100']['R2']:.6f} recomputed {rec['recomputed_test100']['R2']:.6f} "
                      f"(max|d|={rec['reproduction_max_abs_diff']:.1e}) R2(angle)={r2} [{rec['seconds']:.1f}s]", flush=True)
                del run
            except Exception as e:  # noqa: BLE001 - record and continue with the next run
                traceback.print_exc()
                res["runs"][m] = {"task": tname, "model": m, "error": repr(e)}
            json.dump(jsonable(res), open(path, "w"), indent=1)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        res["seconds"] = time.time() - t_task
        json.dump(jsonable(res), open(path, "w"), indent=1)
        print(f"[{tname}] done in {res['seconds']:.0f}s -> {path}", flush=True)


if __name__ == "__main__":
    main()
