"""Tables of the MXS / MXA / MXE runs (``scripts/run_maxwell.sh``; the paper's ``results/evaluations/maxwell_runs.json``):
``results.json`` with the exact metrics of every
run and Markdown tables (MXA: one generalization table with the in-distribution test100 numbers, currents x1.5, the
held-out layout family, the refined / coarsened meshes and the rotated test meshes).

    python3 scripts/maxwell_tables.py --root runs/maxwell/MXA [--json results.json] [--md RESULTS.md]

Reads ``<root>/*_s42/result.json`` (trainer) and ``maxwell_eval.json`` (``scripts/maxwell_eval.py``).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re

NAMES = {
    "esolver_T": "RHMP solver, tensor metric, Dirichlet lift",
    "esolver_D": "RHMP solver, diagonal metric, Dirichlet lift",
    "esolver_T_nolift": "RHMP solver, tensor metric, homogeneous Dirichlet (no lift)",
    "esolver_T_frozen": "RHMP solver, frozen reference metric, lift",
    "solver_T_J4": "RHMP solver, tensor metric, Picard J=4",
    "solver_T_J8": "RHMP solver, tensor metric, Picard J=8",
    "solver_T_J2": "RHMP solver, tensor metric, Picard J=2",
    "solver_T_J2_nogeo": "RHMP solver, tensor, Picard J=2, heads without geometry",
    "solver_T_J4_nogeo": "RHMP solver, tensor, Picard J=4, heads without geometry",
    "solver_T_J0_nogeo": "RHMP solver, tensor, one solve, heads without geometry",
    "solver_D_J4": "RHMP solver, diagonal metric, Picard J=4",
    "solver_T_J0": "RHMP solver, tensor metric, one linear solve",
    "solver_T_frozen": "RHMP solver, frozen reference metric",
    "solver_D_J2_nogeo": "RHMP solver, diagonal metric, Picard J=2, heads without geometry",
    "solver_T_J2_nogeo_init": "RHMP solver, tensor, Picard J=2, learned starting state, heads without geometry",
    "solver_T_J2_noorient": "RHMP solver, tensor, Picard J=2, no rolling-direction input (ablation)",
    "stack_T_J2": "RHMP 3 poly + Picard solve (J=2), C=64",
    "stack_T_J3": "RHMP 3 poly + Picard solve (J=3), C=64",
    "poly4": "RHMP general four-layer (C=128)",
    "poly4_frozen": "RHMP general four-layer, frozen metric",
    "gcn": "GCN", "gat": "GAT", "schnet": "SchNet", "egnn": "EGNN", "mpsn": "MPSN (edge protocol)",
    "sccnn": "SCCNN (edge protocol)", "gauge_cnn": "GaugeEquivCNN", "gem_cnn": "GEM-CNN", "cw_net": "CW Net",
    "clifford_smpn": "Clifford-SMPN (edge protocol)", "deeponet": "DeepONet",
    "mpsn_node": "MPSN (node head)", "sccnn_node": "SCCNN (node head)", "clifford_smpn_node": "Clifford-SMPN (node head)",
    "gem_cnn_rotaug": "GEM-CNN, rotation-augmented", "gauge_cnn_rotaug": "GaugeEquivCNN, rotation-augmented",
    "mgn": "MeshGraphNet, 15 steps (reference, not a paper baseline)",
}
ORDER = list(NAMES)
EXTRA = ("ood15_test100", "fine_test100", "coarse_test100", "heldout_test100", "ood15", "fine", "coarse", "heldout")


def _f(x, nd=4):
    if x is None:
        return "-"
    if isinstance(x, str):
        return x
    if abs(x) < 1e-3 and x != 0:
        return f"{x:.2e}"
    return f"{x:.{nd}f}"


def collect(root: str) -> dict:
    runs = {}
    for d in sorted(glob.glob(os.path.join(root, "*_s42"))):
        key = os.path.basename(d)[:-4]
        rp = os.path.join(d, "result.json")
        if not os.path.exists(rp):
            continue
        r = json.load(open(rp))
        row = dict(run=d, name=NAMES.get(key, key), model=r.get("model"), params=r.get("params"),
                   complete=r.get("complete"), epochs=r.get("epochs_run"), best_epoch=r.get("best_epoch"),
                   val_R2=r.get("best_val_R2"), s_per_epoch=r.get("s_per_epoch"), wall_clock_s=r.get("wall_clock_s"),
                   test=r.get("test"), test100=r.get("test100"),
                   protocol=(r.get("model_info") or {}).get("target_protocol"),
                   width=(r.get("model_info") or {}).get("hidden", (r.get("config") or {}).get("C")),
                   config={k: (r.get("config") or {}).get(k) for k in ("layers", "C", "metric_type", "solve_picard",
                                                                        "log_range", "material_dims", "readout",
                                                                        "learn_metric")} if r.get("model") == "rhmp"
                   else None, args=r.get("args"))
        for k in EXTRA:
            if k in r:
                row[k] = r[k]
        ep = os.path.join(d, "maxwell_eval.json")
        if os.path.exists(ep):
            e = json.load(open(ep))
            for k in ("divergence", "rotation", "timing", "recovery", "transfer"):
                if k in e:
                    row[k] = e[k] if k != "recovery" else {kk: vv for kk, vv in e[k].items()
                                                            if not kk.startswith("curve")}
            for ck in ("curve", "curve_model_field"):
                if "recovery" in e and ck in e["recovery"]:
                    row["recovery_" + ck] = e["recovery"][ck]
        runs[key] = row
    return runs


def tables(runs: dict) -> str:
    keys = [k for k in ORDER if k in runs] + [k for k in runs if k not in ORDER]
    out = ["# MXS / MXA / MXE results", "",
           "test100 = samples 4250..4349; seed 42; 100 epochs unless noted.  Edge-protocol rows are edge-space metrics "
           "of the edge-averaged first component (not comparable with the vertex-field rows).", "",
           "## Main table (test100)", "",
           "| model | params | SSIM | Pearson | NRMSE | R2 | val R2 | s/epoch | epochs |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k in keys:
        r = runs[k]
        t = r.get("test100") or {}
        out.append(f"| {r['name']} | {r['params']:,} | {_f(t.get('SSIM'), 3)} | {_f(t.get('Pearson'), 3)} | "
                   f"{_f(t.get('NRMSE'))} | {_f(t.get('R2'))} | {_f(r.get('val_R2'))} | {_f(r.get('s_per_epoch'), 1)} | "
                   f"{r.get('epochs')}{'' if r.get('complete') else ' (incomplete)'} |")
    if any(runs[k].get("heldout_test100") for k in keys):
        out += ["", "## Generalization (zero-shot; R2 on 100 instances each)", "",
                "in-distribution = test100; currents x1.5 = the test100 layouts at 1.5x current; held-out family = the "
                "first 100 EE samples (layout family absent from training); refined / coarsened = the test100 layouts "
                "on the 4 mm / 16 mm meshes; rotated = test100 with the mesh rotated about the box centre (baselines: "
                "rolling-direction frame components rotated too).", "",
                "| model | params | in-distribution | currents x1.5 | held-out family | refined mesh | coarsened mesh | "
                "rot 30 | rot 90 | rot 137 |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for k in keys:
            r = runs[k]
            ro = r.get("rotation") or {}
            cells = [(r.get(s_) or {}).get("R2") for s_ in ("test100", "ood15_test100", "heldout_test100",
                                                           "fine_test100", "coarse_test100")]
            cells += [(ro.get(a) or {}).get("R2") for a in ("30.0", "90.0", "137.0")]
            out.append(f"| {r['name']} | {r['params']:,} | " + " | ".join(_f(c) for c in cells) + " |")
    out += ["", "## Saturation extrapolation and mesh transfer (zero-shot, R2 / NRMSE on the 100 test100 instances)", "",
            "| model | base mesh | sources x1.5 | refined mesh | coarsened mesh |", "|---|---|---|---|---|"]
    for k in keys:
        r = runs[k]
        cells = []
        for s in ("test100", "ood15_test100", "fine_test100", "coarse_test100"):
            m = r.get(s) or {}
            cells.append(f"{_f(m.get('R2'))} / {_f(m.get('NRMSE'))}" if m else "-")
        out.append(f"| {r['name']} | " + " | ".join(cells) + " |")
    rows_t = [(k, runs[k]["transfer"]) for k in keys if runs[k].get("transfer")]
    if any(any(len(v) > 1 for v in t.values()) for _, t in rows_t):
        out += ["", "## Zero-shot sets of solve models with more CG iterations per solve (R2; numerical setting)", "",
                "| model | set | as trained | more iterations | CG iterations (last batch) |", "|---|---|---:|---:|---|"]
        for k, t in rows_t:
            for name, row in t.items():
                if len(row) < 2:
                    continue
                lab = [x for x in row if x != "as_trained"][0]
                it = {kk.split(".")[-1]: vv for kk, vv in (row[lab].get("solver") or {}).items()}
                out.append(f"| {runs[k]['name']} | {name} | {_f(row['as_trained']['R2'])} | {_f(row[lab]['R2'])} | "
                           f"{', '.join(f'{a}={_f(b, 3)}' for a, b in it.items())} |")
    out += ["", "## Divergence residual of the predicted field (test100; relative net flux per triangle)", "",
            "| model | trapezoid fluxes of the vertex field: median / p99 | native flux cochain: max |",
            "|---|---|---|"]
    for k in keys:
        dv = runs[k].get("divergence") or {}
        if not dv:
            continue
        if "note" in dv:
            out.append(f"| {runs[k]['name']} | n/a (edge protocol) | - |")
            continue
        p = dv.get("pred_trapezoid", {})
        nat = dv.get("native_cochain", {})
        out.append(f"| {runs[k]['name']} | {_f(p.get('median'))} / {_f(p.get('p99'))} | "
                   f"{_f(nat.get('max')) if nat else '-'} |")
    ref = next((runs[k]["divergence"].get("target_trapezoid") for k in keys
                if (runs[k].get("divergence") or {}).get("target_trapezoid")), None)
    if ref:
        out.append(f"| reference vertex fields (targets) | {_f(ref.get('median'))} / {_f(ref.get('p99'))} | - |")
    out += ["", "## Frame robustness (test100 R2, mesh rotated about the box centre)", "",
            "| model | 0 deg | 30 deg | 90 deg | 137 deg | max abs dR2 |", "|---|---:|---:|---:|---:|---:|"]
    for k in keys:
        ro = runs[k].get("rotation") or {}
        if not ro:
            continue
        vals = [ro.get(a, {}).get("R2") for a in ("0", "30.0", "90.0", "137.0")]
        d = max(abs(v - vals[0]) for v in vals if v is not None) if vals[0] is not None else None
        out.append(f"| {runs[k]['name']} | " + " | ".join(_f(v) for v in vals) + f" | {_f(d)} |")
    rows_a = [(k, runs[k]["recovery"]) for k in keys if (runs[k].get("recovery") or {}).get("GO")]
    if rows_a:
        out += ["", "## Anisotropic constitutive recovery (learned tensor of the final iteration vs R^T nu(B) R)", "",
                "| model | steel | major axis along the rolling direction (fraction of faces) | principal-direction "
                "error (deg): median / p90 | eigenvalue ratio: true / learned (median) | corr log ratio | corr log nu "
                "(easy, hard) |", "|---|---|---:|---|---|---:|---|"]
        for k, rc in rows_a:
            for st in ("GO", "LAM"):
                v = rc.get(st)
                if not v:
                    continue
                out.append(f"| {runs[k]['name']} | {st} | {_f(v.get('frac_major_axis_along_rolling_direction'), 3)} | "
                           f"{_f(v['principal_angle_err_deg_median'], 1)} / "
                           f"{_f(v['principal_angle_err_deg_p90'], 1)} | {_f(v['ratio_true_median'], 2)} / "
                           f"{_f(v['ratio_learned_median'], 2)} | {_f(v['corr_log_ratio'], 3)} | "
                           f"{_f(v['corr_log_nu_easy'], 3)}, {_f(v['corr_log_nu_hard'], 3)} |")
            v = rc.get("NO")
            if v:
                out.append(f"| {runs[k]['name']} | NO | - | - | 1 / {_f(v['ratio_learned_median'], 2)} | - | "
                           f"{_f(v['corr_log_nu'], 3)} (slope {_f(v['slope_log_nu'], 2)}) |")
    out += ["", "## Constitutive recovery (learned reluctivity of the final iteration vs nu(|B|), iron faces)", "",
            "| model | kind | corr(log nu) | slope | RMSE(log nu) |", "|---|---|---:|---:|---:|"]
    for k in keys:
        rc = runs[k].get("recovery") or {}
        if "corr_log_nu" in rc:
            out.append(f"| {runs[k]['name']} | {rc.get('kind')} | {_f(rc.get('corr_log_nu'))} | "
                       f"{_f(rc.get('slope_log_nu'))} | {_f(rc.get('rmse_log_nu'))} |")
        elif "corr_log_eps" in rc:
            pp = "; ".join(f"{ph} {_f(v['learned_median'], 2)} (true {_f(v['true_log_eps'], 2)})"
                           for ph, v in (rc.get("per_phase") or {}).items())
            out.append(f"| {runs[k]['name']} | {rc.get('kind')}; log eps rel. air per phase: {pp} | "
                       f"{_f(rc.get('corr_log_eps'))} | {_f(rc.get('slope_log_eps'))} | {_f(rc.get('rmse_log_eps'))} |")
    out += ["", "## Timing", "", "| model | s/epoch | inference s/sample | solver |", "|---|---:|---:|---|"]
    for k in keys:
        tm = runs[k].get("timing") or {}
        sv = tm.get("solver") or {}
        svs = ", ".join(f"{kk.split('.')[-1]}={_f(vv, 3)}" for kk, vv in sv.items() if "clamp" not in kk)
        out.append(f"| {runs[k]['name']} | {_f(runs[k].get('s_per_epoch'), 1)} | {_f(tm.get('infer_s_per_sample'))} | "
                   f"{svs or '-'} |")
    return "\n".join(out) + "\n"


METRICS = ("SSIM", "Pearson", "NRMSE", "R2")
ZERO_SHOT = ("ood15_test100", "heldout_test100", "fine_test100", "coarse_test100", "ood15", "heldout", "fine", "coarse")


def _m(d) -> dict | None:
    return {k: d[k] for k in METRICS if k in d} if d else None


def paper_rows(roots: list[str], prefix: str = "runs/maxwell") -> list[dict]:
    """One record per run in the schema of the paper's ``results.json`` (run, task, mode, model, seed, params, best_val_R2,
    best_epoch, epochs_run, complete, s_per_epoch, peak_mem_GB, gpu, test100, test, zero_shot, rotation, options,
    config, final_train_loss, final_val_MSE); ``model`` is ``rhmp`` or the baseline's registry name, ``variant`` the run
    directory (``<variant>_s<seed>``)."""
    rows = []
    for root in roots:
        task_dir = os.path.basename(os.path.normpath(root))
        for d in sorted(glob.glob(os.path.join(root, "*_s[0-9]*"))):
            rp = os.path.join(d, "result.json")
            key = os.path.basename(d)
            if not re.fullmatch(r".+_s\d+", key):                        # aborted / renamed directories
                continue
            if not os.path.exists(rp):                                    # unfinished run: progress only
                hp, cp = os.path.join(d, "history.json"), os.path.join(d, "config.json")
                if os.path.exists(hp) and os.path.exists(cp):
                    h = json.load(open(hp))
                    c = json.load(open(cp))
                    if h:
                        best = max(h, key=lambda x: x.get("val_R2", -1e9))
                        rows.append(dict(run=f"{prefix}/{task_dir}/{key}", task=task_dir,
                                         model=(c.get("args") or {}).get("model") or "rhmp", variant=key.rsplit("_s", 1)[0],
                                         seed=(c.get("args") or {}).get("seed"), params=c.get("params"),
                                         best_val_R2=best.get("val_R2"), best_epoch=best.get("epoch"),
                                         epochs_run=len(h), complete=False))
                continue
            r = json.load(open(rp))
            args = r.get("args") or {}
            model = r.get("model") or args.get("model") or "rhmp"
            row = dict(run=f"{prefix}/{task_dir}/{key}", task=r.get("task", task_dir), mode=r.get("mode"), model=model,
                       variant=key.rsplit("_s", 1)[0], seed=r.get("seed"), params=r.get("params"),
                       best_val_R2=r.get("best_val_R2"), best_epoch=r.get("best_epoch"), epochs_run=r.get("epochs_run"),
                       complete=r.get("complete"), s_per_epoch=r.get("s_per_epoch"), peak_mem_GB=r.get("peak_mem_GB"),
                       gpu=r.get("gpu"), test100=_m(r.get("test100")), test=_m(r.get("test")))
            zs = {k: _m(r[k]) for k in ZERO_SHOT if r.get(k)}
            if zs:
                row["zero_shot"] = zs
            ep = os.path.join(d, "maxwell_eval.json")
            if os.path.exists(ep):
                e = json.load(open(ep))
                rot = e.get("rotation") or {}
                if rot:
                    row["rotation"] = {str(int(float(a_))): _m(v)
                                       for a_, v in sorted(rot.items(), key=lambda kv: float(kv[0]))}
            info = r.get("model_info") or {}
            if model == "rhmp":
                cfg = r.get("config") or {}
                row["options"] = {k: cfg.get(k) for k in ("layers", "C", "metric_type", "solve_picard", "metric_geo",
                                                           "material_dims", "log_range", "learn_metric", "solve_bc",
                                                           "solve_iters", "solve_tol") if k in cfg}
                row["config"] = cfg
            else:
                opts = {}
                if args.get("model_opts"):
                    opts["model_opts"] = args["model_opts"]
                if r.get("augmentation"):
                    opts["augmentation"] = r["augmentation"]
                if info.get("target_protocol"):
                    opts["target_protocol"] = info["target_protocol"]
                opts["hidden"] = info.get("hidden")
                row["options"] = opts
            hp = os.path.join(d, "history.json")
            if os.path.exists(hp):
                h = json.load(open(hp))
                if h:
                    row["final_train_loss"] = h[-1].get("train_loss")
                    row["final_val_MSE"] = h[-1].get("val_MSE")
            rows.append(row)
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default="runs/maxwell/MXS")
    p.add_argument("--json", default=None)
    p.add_argument("--md", default=None)
    p.add_argument("--paper-json", default=None, help="write the run records of --paper-roots in the paper's results "
                                                      "schema to this file and exit")
    p.add_argument("--paper-roots", default=None, help="comma-separated task roots (default: --root)")
    a = p.parse_args(argv)
    if a.paper_json:
        rows = paper_rows((a.paper_roots or a.root).split(","))
        with open(a.paper_json, "w") as f:
            json.dump(rows, f, indent=1)
        print(f"written {len(rows)} records to {a.paper_json}")
        return
    runs = collect(a.root)
    jpath = a.json or os.path.join(a.root, "results.json")
    with open(jpath, "w") as f:
        json.dump(runs, f, indent=1)
    md = tables(runs)
    mpath = a.md or os.path.join(a.root, "RESULTS.md")
    with open(mpath, "w") as f:
        f.write(md)
    print(md)
    print(f"written {jpath}, {mpath}")


if __name__ == "__main__":
    main()
