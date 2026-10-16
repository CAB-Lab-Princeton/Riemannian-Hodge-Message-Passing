"""Evaluation of pressure runs on the conforming AirfRANS meshes (tasks AFP* of ``rhmp/tasks/airfrans.py``), all in
physical units, without retraining (the zero-shot, rotation, wall and lift entries of the paper's AirfRANS tables;
collected into ``results/evaluations/airfrans_pressure_runs.json`` by ``scripts/airfrans_pressure_collect.py``):

    python3 scripts/airfrans_pressure_eval.py RUN_DIR [RUN_DIR ...] [--dump] [--device cpu] [--rotations 30,90,137]

Per run (RHMP or a baseline trained with ``rhmp.train``) the best checkpoint is evaluated on the 100 paper-evaluation
simulations (850..949) of

* ``test100``: the training resolution (must reproduce ``result.json["test100"]``);
* ``fine`` / ``coarse``: the same simulations on meshes with about 3.6x / 0.3x the nodes (zero-shot);
* rotations: the whole configuration (mesh and inflow) rotated rigidly; the scalar inputs (sdf, U, angle of attack,
  chord-frame coordinates of the ``xy`` controls) are unchanged, the complexes are rebuilt from the rotated positions
  and the inflow 1-form of the covariant tasks from the rotated inflow;
* ``wall``: the vertices on the airfoil only (surface pressure), and the pressure lift coefficient
  ``C_l = -oint p n . e_L ds / (U^2 / 2)`` (``n`` the wall normal into the fluid, ``e_L`` normal to the inflow, chord 1)
  from the predicted and from the CFD wall pressure of the same mesh.

Runs trained on the pressure coefficient (``AFPcp*``) are converted with the dynamic pressure of each simulation,
``p = C_p U^2 / 2``, before any metric.  Metrics: SSIM / Pearson / NRMSE / R2 with the library definitions
(``rhmp.metrics``; R2 centred on the pooled mean).  Output: ``RUN_DIR/pressure_metrics.json``; ``--dump`` writes
``<task dir>/preds/<run>.npz`` (meshes, vertex kinds, predictions and targets in Pa for ``test100``, ``fine_test100``,
``coarse_test100``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def load_run(run_dir: str, device):
    """``(conf, model_name, model, task name, native, x_stats, y_stats, output map, cp_target)`` of a finished run."""
    from rhmp.data import Stats
    from rhmp.tasks import load_task, resolve_root
    from rhmp.tasks.airfrans import N_TRAIN_STATS, _load_blob, afp_output_map
    from rhmp.train import needs_whitney
    conf = json.load(open(os.path.join(run_dir, "config.json")))
    a = conf["args"]
    model_name = a.get("model") or "rhmp"
    task_name, native = a["task"], bool(a["native"])
    whitney = model_name == "rhmp" and needs_whitney(conf.get("model", {}))
    small = load_task(task_name, a.get("root"), native=native, device="cpu", star=a.get("star", "cotan"),
                      max_samples=9, fine=False, whitney=whitney)                      # the model's input layout only
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    if model_name == "rhmp":
        from rhmp.model import RHMP
        model = RHMP.from_checkpoint(ck, map_location=device)
    else:
        from rhmp.baselines.registry import from_checkpoint, prepare_task
        model = from_checkpoint(ck, prepare_task(model_name, small, ck.get("build")), map_location=device)
    model.eval()
    if hasattr(model, "record_diagnostics"):
        model.record_diagnostics = False
    d = conf["data"]
    x_stats = {int(k): Stats(torch.tensor(v["mean"]), torch.tensor(v["std"]), v.get("kind", "v1"))
               for k, v in d["x_stats"].items()}
    y_stats = Stats(torch.tensor(d["y_stats"]["mean"]), torch.tensor(d["y_stats"]["std"]), d["y_stats"].get("kind", "v1"))
    omap = None
    if model_name == "rhmp" and d.get("output_map"):
        u2 = d.get("meta", {}).get("u2_ref")
        if u2 is None:                                           # runs written before the value was stored
            U = np.asarray(_load_blob(resolve_root(a.get("root")), "AFM")["meta"]["U"][:N_TRAIN_STATS])
            u2 = float(np.mean(U ** 2))
        omap = afp_output_map(task_name, x_stats, y_stats, u2)
    return dict(conf=conf, model_name=model_name, model=model, task=task_name, native=native, x_stats=x_stats,
                y_stats=y_stats, omap=omap, cp=task_name.startswith("AFPcp"), whitney=whitney,
                star=a.get("star", "cotan"), root=a.get("root"))


@torch.no_grad()
def predict(run: dict, src: dict, device, batch: int = 8) -> tuple[list, list]:
    """Predictions and targets in Pa (lists of ``(n0_i,)`` float64 arrays)."""
    from rhmp.data import mesh_minibatch
    preds, tgts = [], []
    ys = run["y_stats"]
    for s in range(0, len(src["K"]), batch):
        ids = list(range(s, min(s + batch, len(src["K"]))))
        K, xb, yb = mesh_minibatch(src["K"], src["inputs"], src["target"], ids, device=device)
        y = run["model"](xb, K)
        if run["omap"] is not None:
            y = run["omap"](y)
        sizes = [int(src["target"][i].shape[0]) for i in ids]
        p = ys.denormalize(y[:, 0].cpu()).double().split(sizes, 0)
        t = ys.denormalize(yb[:, 0].cpu()).double().split(sizes, 0)
        for i, pi, ti in zip(ids, p, t):
            q = 0.5 * src["U"][i] ** 2 if run["cp"] else 1.0     # pressure-coefficient runs: back to Pa
            preds.append((pi[:, 0] * q).numpy())
            tgts.append((ti[:, 0] * q).numpy())
    return preds, tgts


def metrics(preds: list, tgts: list) -> dict:
    from rhmp import metrics as M
    p = [torch.as_tensor(x)[:, None] for x in preds]
    t = [torch.as_tensor(x)[:, None] for x in tgts]
    ssim, pear = M.ssim_pearson(p, t)
    return dict(SSIM=ssim, Pearson=pear, NRMSE=M.nrmse(p, t), R2=M.r2_score(p, t))


def wall_lift(src: dict, vals: list) -> np.ndarray:
    """Pressure lift coefficient of every simulation from vertex pressures (Pa): ``-sum_e p_e n_e l_e . e_L / (U^2/2)``
    over the wall edges, ``p_e`` the mean of the two end values, ``n_e`` the unit normal into the fluid (chord frame)."""
    out = []
    for K, kind, X, U, aoa, p in zip(src["K"], src["kind"], src["pos0"], src["U"], src["aoa"], vals):
        e = K.cells[1].long().cpu().numpy()
        kd = np.asarray(kind)
        X = np.asarray(X)
        wall = (kd[e[:, 0]] == 1) & (kd[e[:, 1]] == 1) & K.boundary[1].cpu().numpy().astype(bool)
        fe = K.meta["face_edges"].cpu().numpy()
        face_of = np.zeros(K.n[1], dtype=np.int64)
        face_of[fe.ravel()] = np.repeat(np.arange(K.n[2]), fe.shape[1])
        F = K.cells[2].long().cpu().numpy()
        we = np.where(wall)[0]
        a, b = X[e[we, 0]], X[e[we, 1]]
        t = b - a
        n = np.stack([t[:, 1], -t[:, 0]], 1)                                  # |n| = edge length
        to_fluid = X[F[face_of[we]]].mean(1) - 0.5 * (a + b)
        n = n * np.sign((n * to_fluid).sum(1))[:, None]
        pe = 0.5 * (p[e[we, 0]] + p[e[we, 1]])
        force = -(pe[:, None] * n).sum(0)
        eL = np.array([-np.sin(aoa), np.cos(aoa)])
        out.append(float(force @ eL) / (0.5 * U ** 2))
    return np.array(out)


def evaluate_run(run_dir: str, device, rotations: list[float], dump: bool, batch: int) -> dict:
    from rhmp.tasks import resolve_root
    from rhmp.tasks.airfrans import afm_eval_source
    run = load_run(run_dir, device)
    droot = resolve_root(run["root"])
    kw = dict(native=run["native"], star=run["star"], whitney=run["whitney"])

    def source(base, rotate=0.0):
        return afm_eval_source(run["task"], droot, run["x_stats"], run["y_stats"], base=base, rotate=rotate, **kw)

    res_path = os.path.join(run_dir, "result.json")
    R = json.load(open(res_path)) if os.path.exists(res_path) else {}
    out = dict(run=os.path.basename(os.path.abspath(run_dir)), task=run["task"], model=run["model_name"],
               params=run["conf"].get("params"), seed=run["conf"]["args"].get("seed"),
               best_val_R2=R.get("best_val_R2"), complete=bool(R.get("complete")),
               native=run["native"], cp_target=run["cp"])
    arrays = {}
    src = source("AFM")
    p0, t0 = predict(run, src, device, batch)
    out["test100"] = metrics(p0, t0)
    if R.get("test100") and not run["cp"]:                       # the trainer's numbers (same checkpoint and sets)
        out["test100_trainer_R2"] = R["test100"]["R2"]
        out["test100_matches_trainer"] = bool(abs(R["test100"]["R2"] - out["test100"]["R2"]) < 2e-4)
    wall = [np.asarray(k) == 1 for k in src["kind"]]
    pw, tw = [p[w] for p, w in zip(p0, wall)], [t[w] for t, w in zip(t0, wall)]
    out["wall"] = metrics(pw, tw)
    off = [~w for w in wall]
    out["off_wall"] = metrics([p[w] for p, w in zip(p0, off)], [t[w] for t, w in zip(t0, off)])
    cl_p, cl_t = wall_lift(src, p0), wall_lift(src, t0)
    out["lift"] = dict(mae=float(np.abs(cl_p - cl_t).mean()), rel_err_median=float(np.median(np.abs(cl_p - cl_t) /
                                                                                              np.abs(cl_t).clip(1e-6))),
                       R2=float(1 - ((cl_p - cl_t) ** 2).sum() / ((cl_t - cl_t.mean()) ** 2).sum()),
                       cl_true_mean_abs=float(np.abs(cl_t).mean()))

    def add_dump(name, s, p, t):
        Ks = s["K"]
        arrays[f"{name}_pos"] = np.concatenate([K.pos.cpu().numpy() for K in Ks])
        arrays[f"{name}_faces"] = np.concatenate([K.cells[2].cpu().numpy() for K in Ks])
        arrays[f"{name}_edges"] = np.concatenate([K.cells[1].cpu().numpy() for K in Ks])
        for k in range(3):
            arrays[f"{name}_ptr{k}"] = np.cumsum([0] + [K.n[k] for K in Ks])
        arrays[f"{name}_kind"] = np.concatenate([np.asarray(k).astype(np.int8) for k in s["kind"]])
        arrays[f"{name}_pred"] = np.concatenate(p).astype(np.float32)[:, None]
        arrays[f"{name}_target"] = np.concatenate(t).astype(np.float32)[:, None]
        arrays[f"{name}_src"] = np.asarray(s["src"])

    if dump:
        add_dump("test100", src, p0, t0)
    out["zero_shot"] = {}
    for label, base in (("fine", "AFM_fine"), ("coarse", "AFM_coarse")):
        if not os.path.exists(os.path.join(droot, "v2", base + ".pt")):
            continue
        s = source(base)
        p, t = predict(run, s, device, max(1, batch // 4) if label == "fine" else batch)
        out["zero_shot"][label] = metrics(p, t)
        w = [np.asarray(k) == 1 for k in s["kind"]]
        out["zero_shot"][label]["wall_R2"] = metrics([a[m] for a, m in zip(p, w)], [a[m] for a, m in zip(t, w)])["R2"]
        if dump:
            add_dump(label + "_test100", s, p, t)
        del s
    out["rotation"], out["rotation_max_rel_change"] = {}, {}
    scale = float(np.sqrt(np.mean(np.concatenate(t0) ** 2)))
    for deg in rotations:
        s = source("AFM", rotate=deg)
        p, t = predict(run, s, device, batch)
        key = str(int(deg)) if float(deg).is_integer() else str(deg)
        out["rotation"][key] = metrics(p, t)["R2"]
        out["rotation_max_rel_change"][key] = float(max(np.abs(a - b).max() for a, b in zip(p, p0)) / scale)
        del s
    with open(os.path.join(run_dir, "pressure_metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    if dump:
        pdir = os.path.join(os.path.dirname(os.path.abspath(run_dir)), "preds")
        os.makedirs(pdir, exist_ok=True)
        path = os.path.join(pdir, os.path.basename(os.path.abspath(run_dir)) + ".npz")
        np.savez_compressed(path, task=run["task"], model=run["model_name"],
                            units="lengths in chords; pressure Pa (kinematic, m^2/s^2); kind 0 volume, 1 wall, 2 ring",
                            **arrays)
        out["dump"] = path
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--device", default=None)
    ap.add_argument("--rotations", default="30,90,137")
    ap.add_argument("--dump", action="store_true")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--force", action="store_true", help="re-evaluate runs that already have pressure_metrics.json")
    a = ap.parse_args(argv)
    device = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    rots = [float(v) for v in a.rotations.split(",") if v]
    for r in a.runs:
        if not os.path.exists(os.path.join(r, "best.pt")):
            print(f"skip {r}: no best.pt", flush=True)
            continue
        if os.path.exists(os.path.join(r, "pressure_metrics.json")) and not a.force and not a.dump:
            print(f"skip {r}: evaluated", flush=True)
            continue
        res = evaluate_run(r, device, rots, a.dump, a.batch)
        print(json.dumps({k: res[k] for k in ("run", "task", "model", "test100", "wall", "lift", "zero_shot", "rotation",
                                              "rotation_max_rel_change") if k in res}, indent=1), flush=True)


if __name__ == "__main__":
    main()
