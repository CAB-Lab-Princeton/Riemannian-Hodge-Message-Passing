#!/usr/bin/env python3
"""Regenerate ``results/RESULTS.md`` (the tables of the paper) from the result files in ``results/``.

    python3 scripts/collect_results.py results --out results/RESULTS.md
    python3 scripts/collect_results.py runs --index-only            # index of your own runs (runs/<task>/<run>/result.json)

The script reads JSON files only (no torch, no GPU, no data sets).  Layout of ``results/``: one directory per task
(``ns_vorticity``, ``torus_transport``, ``ellipsoid_flow``, ``maxwell_magnetostatics``, ``wilson_loop``, ``yang_mills``,
``yang_mills_face``, ``airfrans_pressure``) with one run directory ``<model>_s<seed>`` per model and seed, each holding
the trainer's ``result.json`` (and ``config.json``; ``best.pt`` for the RHMP runs), plus ``results/evaluations/`` with
the zero-shot, rotation and bootstrap evaluations (``results/evaluations/README.md``).

Every number is ``result.json["test100"]`` (SSIM, Pearson, NRMSE, R2 on the first 100 test samples, the protocol of
the paper's tables) of the seed-42 run, or the mean and sample standard deviation over the seeds 42, 1 and 2.  The
zero-shot columns of the AirfRANS table come from ``results/evaluations/airfrans_pressure_runs.json``
(``zero_shot.coarse.R2`` / ``zero_shot.fine.R2``, written by ``scripts/airfrans_pressure_eval.py``).  R2 conventions
(``rhmp.metrics``, recorded as ``r2_definition``): centred ``1 - SS_res / SS_tot`` with the per-position mean over
samples for node targets; uncentred ``1 - SS_res / sum t^2`` for orientation-odd cochain targets (``yang_mills_face``,
marked with a dagger); MPSN, SCCNN and Clifford-SMPN are scored in edge space on the fixed-mesh tasks (their v1 edge
protocol).  ``-`` marks a missing run.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys
from typing import Any

DAGGER = "†"
METRICS = ("SSIM", "Pearson", "NRMSE", "R2")
MODELS = ["gcn", "gat", "schnet", "egnn", "mpsn", "sccnn", "gauge_cnn", "gem_cnn", "cw_net", "clifford_smpn",
          "deeponet", "fno", "rhmp"]
NAMES = {"gcn": "GCN", "gat": "GAT", "schnet": "SchNet", "egnn": "EGNN", "mpsn": "MPSN", "sccnn": "SCCNN",
         "gauge_cnn": "GaugeEquivCNN", "gem_cnn": "GEM-CNN", "cw_net": "CW Net", "clifford_smpn": "Clifford-SMPN",
         "deeponet": "DeepONet", "fno": "FNO", "rhmp": "RHMP", "mgn": "MeshGraphNet"}
FAMILY = {"gcn": "graph", "gat": "graph", "schnet": "graph", "egnn": "graph", "mpsn": "topological",
          "sccnn": "topological", "gauge_cnn": "geometric", "gem_cnn": "geometric", "cw_net": "geometric",
          "clifford_smpn": "geometric", "deeponet": "operator", "fno": "operator", "rhmp": ""}
TASKS = [  # directory, name in the paper, task codes (RHMP / baselines)
    ("ns_vorticity", "NS vorticity on a stretched periodic grid", "T1gml_mean / T1g"),
    ("torus_transport", "Anisotropic transport on a torus", "T2a"),
    ("ellipsoid_flow", "Ellipsoid surface flow", "T3g / T3"),
    ("maxwell_magnetostatics", "Nonlinear anisotropic magnetostatics", "MXA"),
    ("wilson_loop", "U(1) Wilson loop", "T6"),
    ("yang_mills", "SU(2) Yang-Mills", "T7"),
]
AIRFRANS = ("airfrans_pressure", "AirfRANS pressure on body-fitted meshes", "AFPi0 / AFP")
SEEDS = (42, 1, 2)


# ================================================================================================================
# helpers
# ================================================================================================================
def load(path: str) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def result(root: str, task_dir: str, model: str, seed: int) -> dict | None:
    return load(os.path.join(root, task_dir, f"{model}_s{seed}", "result.json"))


def t100(res: dict | None, key: str) -> float | None:
    v = (res or {}).get("test100")
    return v.get(key) if isinstance(v, dict) else None


def f(x: Any, nd: int = 4) -> str:
    """``nd`` decimals; six decimals for scores above 0.9999 and scientific notation below 1e-3 (as the paper)."""
    if x is None:
        return "-"
    x = float(x)
    if not math.isfinite(x):
        return "nan"
    if 0 < abs(x) < 1e-3:
        return f"{x:.2e}"
    if 0.9999 <= x < 1 and nd <= 4:
        return f"{x:.6f}"
    return f"{x:.{nd}f}"


def pm(vals: list[float], nd: int = 4) -> str:
    """mean ± sample std of the values of the three seeds (``-`` if none; no ± for a single seed)."""
    if not vals:
        return "-"
    m = statistics.fmean(vals)
    if len(vals) < 2:
        return f(m, nd) + " (1 seed)"
    s = statistics.stdev(vals)
    return f"{f(m, nd)} ± {s:.{nd}f}" if s >= 10 ** (-nd) else f"{f(m, nd)} ± {s:.1e}"


def fk(n: Any) -> str:
    if n is None:
        return "-"
    n = int(n)
    return f"{n / 1000:.1f}K" if n < 10_000 else f"{n / 1000:.0f}K"


def odd(res: dict | None) -> str:
    return DAGGER if res and "uncentred" in str(res.get("r2_definition", "")) else ""


def edge_protocol(res: dict | None) -> str:
    """``†`` for runs scored in edge space (the v1 edge protocol of MPSN / SCCNN / Clifford-SMPN)."""
    info = (res or {}).get("model_info") or {}
    return DAGGER if "edge" in str(info.get("target_protocol", "")).lower() else ""


def table(header: list[str], rows: list[list[Any]], align: str | None = None) -> str:
    align = align or ("l" + "r" * (len(header) - 1))
    sep = ["---:" if a == "r" else "---" for a in align]
    out = ["| " + " | ".join(header) + " |", "|" + "|".join(sep) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out) + "\n"


def models_present(root: str, task_dir: str) -> list[str]:
    return [m for m in MODELS if any(os.path.isdir(os.path.join(root, task_dir, f"{m}_s{s}")) for s in SEEDS)]


def best_marks(rows: list[dict], key: str, higher: bool) -> set:
    """Index of the best value of ``key`` (ties included)."""
    vals = [(r[key], i) for i, r in enumerate(rows) if r.get(key) is not None]
    if not vals:
        return set()
    best = max(v for v, _ in vals) if higher else min(v for v, _ in vals)
    return {i for v, i in vals if v == best}


# ================================================================================================================
# sections
# ================================================================================================================
def task_rows(root: str, task_dir: str, seed: int = 42) -> list[dict]:
    rows = []
    for m in models_present(root, task_dir):
        res = result(root, task_dir, m, seed)
        rows.append({"model": m, "res": res, "params": (res or {}).get("params"),
                     **{k: t100(res, k) for k in METRICS}})
    return rows


def sec_main(root: str) -> str:
    out = ["## 1. Fixed-mesh tasks (Table 2 of the paper)\n",
           "Seed 42, metrics on the first 100 test samples (`test100`).  SSIM and Pearson: higher is better; NRMSE: "
           "lower is better; R2 is the coefficient of determination of the normalised target.  Bold: best value "
           "of the column.  Parameter counts are the trainable parameters of each model (baseline widths matched "
           "to the RHMP budget of the task).\n"]
    for task_dir, name, codes in TASKS:
        rows = task_rows(root, task_dir)
        marks = {k: best_marks(rows, k, k != "NRMSE") for k in METRICS}
        lines = []
        for i, r in enumerate(rows):
            cells = [NAMES[r["model"]] + edge_protocol(r["res"]), FAMILY.get(r["model"], ""), fk(r["params"])]
            for k in METRICS:
                s = f(r[k])
                cells.append(f"**{s}**" if i in marks[k] else s)
            lines.append(cells)
        out += [f"### {name} (`{codes}`, `results/{task_dir}/`)\n",
                table(["model", "family", "params", "SSIM", "Pearson", "NRMSE", "R2"], lines, "llrrrrr")]
    out.append(DAGGER + ": the run predicts the edge-averaged target and is scored in edge space (the original "
               "evaluation protocol of MPSN, SCCNN and Clifford-SMPN on these tasks).\n")
    return "\n".join(out)


def sec_airfrans(root: str) -> str:
    task_dir, name, codes = AIRFRANS
    rows = task_rows(root, task_dir)
    ev = load(os.path.join(root, "evaluations", "airfrans_pressure_runs.json")) or []
    zs = {}
    for e in ev:
        if e.get("seed") == 42 and e.get("task") in ("AFP", "AFPi0") and e.get("run") == f"{e.get('model')}_s42":
            zs[e["model"]] = e.get("zero_shot") or {}
    for r in rows:
        z = zs.get(r["model"], {})
        r["coarse"] = (z.get("coarse") or {}).get("R2")
        r["fine"] = (z.get("fine") or {}).get("R2")
    keys = list(METRICS) + ["coarse", "fine"]
    marks = {k: best_marks(rows, k, k != "NRMSE") for k in keys}
    lines = []
    for i, r in enumerate(rows):
        cells = [NAMES[r["model"]] + edge_protocol(r["res"]), FAMILY.get(r["model"], ""), fk(r["params"])]
        for k in keys:
            s = f(r[k])
            cells.append(f"**{s}**" if i in marks[k] else s)
        lines.append(cells)
    return "\n".join([
        "## 2. AirfRANS pressure on body-fitted meshes (Table 3 of the paper)\n",
        f"`{codes}`, `results/{task_dir}/`; seed 42, 100 test simulations (850..949).  SSIM, Pearson, NRMSE and R2 at "
        "the training mesh size (about 1.9K nodes, `test100` of `result.json`); R2 (0.55K) and R2 (6.9K): the same "
        "simulations meshed with the spacing doubled and halved, evaluated without retraining "
        "(`results/evaluations/airfrans_pressure_runs.json`, `zero_shot.coarse` / `zero_shot.fine`).  The paper's "
        "Table 3 lists EGNN, MPSN, GaugeEquivCNN, GEM-CNN, CW Net, Clifford-SMPN and RHMP; the remaining baselines "
        "are in the appendix of the paper and here.\n",
        table(["model", "family", "params", "SSIM", "Pearson", "NRMSE", "R2", "R2 (0.55K)", "R2 (6.9K)"], lines,
              "llrrrrrrr")])


def sec_seeds(root: str) -> str:
    out = ["## 3. Cross-seed statistics (seeds 42, 1, 2)\n",
           "Mean ± sample standard deviation of the `test100` metrics over the three training seeds (the paper's "
           "cross-seed appendix reports the standard deviations).  Every run was trained from scratch with its seed; "
           "the test samples are the same.\n"]
    for task_dir, name, codes in TASKS + [AIRFRANS]:
        lines = []
        for m in models_present(root, task_dir):
            per = {k: [] for k in METRICS}
            n = 0
            for s in SEEDS:
                res = result(root, task_dir, m, s)
                if res is None or not isinstance(res.get("test100"), dict):
                    continue
                n += 1
                for k in METRICS:
                    per[k].append(float(res["test100"][k]))
            lines.append([NAMES[m], str(n)] + [pm(per[k]) for k in METRICS])
        out += [f"### {name} (`results/{task_dir}/`)\n",
                table(["model", "seeds", "SSIM", "Pearson", "NRMSE", "R2"], lines, "lrrrrr")]
    return "\n".join(out)


def sec_extra(root: str) -> str:
    res = result(root, "yang_mills_face", "rhmp", 42)
    rows = []
    if res:
        rows.append(["RHMP, face target (`T7f`)" + odd(res), fk(res.get("params"))] + [f(t100(res, k)) for k in METRICS])
    out = ["## 4. Further RHMP runs\n",
           f"`yang_mills_face`: the Yang-Mills model with the field strength predicted on the faces (task `T7f`; {DAGGER} "
           "uncentred R2 of the orientation-odd face cochain), seed 42.\n",
           table(["run", "params", "SSIM", "Pearson", "NRMSE", "R2"], rows, "lrrrrr"),
           "\nZero-shot transfer, rigid rotations, the test-set bootstrap and the Maxwell / AirfRANS ablations and "
           "controls are in `results/evaluations/` (see the README there).\n"]
    return "\n".join(out)


def sec_index(root: str) -> str:
    rows = []
    for p in sorted(glob.glob(os.path.join(root, "*", "*", "result.json"))):
        rel = os.path.relpath(os.path.dirname(p), root)
        res = load(p)
        if res is None:
            continue
        args = res.get("args") or {}
        rows.append([f"`{rel}`", str(res.get("task", args.get("task", ""))), str(res.get("model", "rhmp")),
                     str(res.get("mode", "")), str(res.get("seed", args.get("seed", ""))),
                     f(t100(res, "R2")) + odd(res), f((res.get("test") or {}).get("R2")), fk(res.get("params")),
                     str(res.get("epochs_run", "-")), f(res.get("s_per_epoch"), 1), f(res.get("peak_mem_GB"), 2),
                     str(res.get("gpu", "")).replace("NVIDIA ", "")])
    return "\n".join(["## Appendix: index of all runs\n",
                      "`test100 R2` on the first 100 test samples, `test R2` on the full test split; `s/epoch` and "
                      "`peak GB` as measured during training on the listed GPU.\n",
                      table(["run", "task", "model", "inputs", "seed", "test100 R2", "test R2", "params", "epochs",
                             "s/epoch", "peak GB", "GPU"], rows, "lllllrrrrrrl")])


def sec_paper_tables(root: str) -> str:
    """The layout of the paper's Table 2 (models x tasks, one table per metric) and Table 3, for the README."""
    out = []
    per_task = {td: {r["model"]: r for r in task_rows(root, td)} for td, _, _ in TASKS}
    short = {"ns_vorticity": "NS vorticity", "torus_transport": "torus transport", "ellipsoid_flow": "ellipsoid flow",
             "maxwell_magnetostatics": "magnetostatics", "wilson_loop": "Wilson loop", "yang_mills": "Yang-Mills"}
    models = [m for m in MODELS if any(m in per_task[td] for td, _, _ in TASKS)]
    for k in ("SSIM", "Pearson", "NRMSE"):
        lines = []
        marks = {td: best_marks([per_task[td].get(m, {}) for m in models], k, k != "NRMSE") for td, _, _ in TASKS}
        for i, m in enumerate(models):
            cells = [NAMES[m]]
            for td, _, _ in TASKS:
                r = per_task[td].get(m)
                s = f(r[k]) if r else "-"
                cells.append(f"**{s}**" if i in marks[td] else s)
            lines.append(cells)
        out += [f"**{k}**" + (" (lower is better)" if k == "NRMSE" else " (higher is better)") + "\n",
                table(["model"] + [short[td] for td, _, _ in TASKS], lines, "l" + "r" * len(TASKS))]
    return "\n".join(out)


# ================================================================================================================
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="results", help="results directory (or a runs/ directory)")
    ap.add_argument("--out", default=None, help="Markdown file to write (default: standard output)")
    ap.add_argument("--index-only", action="store_true", help="only the index of all result.json files")
    ap.add_argument("--paper-tables", action="store_true",
                    help="only the tables in the layout of the paper (models x tasks per metric; the README)")
    a = ap.parse_args(argv)
    if a.index_only:
        text = "# Runs\n\n" + sec_index(a.root)
    elif a.paper_tables:
        text = sec_paper_tables(a.root) + "\n" + sec_airfrans(a.root)
    else:
        text = "\n".join([
            "# Results\n",
            "Generated by `python3 scripts/collect_results.py results --out results/RESULTS.md` from the result "
            "files in `results/` (one directory per task and run; the layout and the conventions are described in "
            "the docstring of that script and in `results/README.md`).  The numbers are those of the tables of "
            "*Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame Equivariance* "
            "(Zheng & Allen-Blanchette, NeurIPS 2026).\n",
            sec_main(a.root), sec_airfrans(a.root), sec_seeds(a.root), sec_extra(a.root), sec_index(a.root)])
    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w") as fh:
            fh.write(text)
        print(f"wrote {a.out} ({len(text.splitlines())} lines)")
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    main()
