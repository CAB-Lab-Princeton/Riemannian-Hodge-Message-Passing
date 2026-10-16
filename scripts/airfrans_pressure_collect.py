"""Collect the AirfRANS pressure study into one JSON list and Markdown tables (the paper's
``results/evaluations/airfrans_pressure_runs.json``).

    python3 scripts/airfrans_pressure_collect.py [--root runs/airfrans] [--out runs/airfrans/results_pressure.json]
                                                  [--tasks AFP AFPi ...] [--md]

One record per run directory ``<root>/<task>/<name>_s<seed>`` with a ``result.json``: ``run, task, model, params, seed,
best_val_R2, complete, epochs_run, test100 {SSIM, Pearson, NRMSE, R2}`` (physical units; from
``pressure_metrics.json`` of ``scripts/airfrans_pressure_eval.py`` when present, else the trainer's
``result.json["test100"]``), ``zero_shot {fine, coarse}``, ``rotation {"30": R2, ...}``, ``wall``, ``lift`` and
``control`` (what differs from the main protocol; empty for the main rows).
"""
from __future__ import annotations

import argparse
import glob
import json
import os

MAIN_TASKS = ("AFP", "AFPi", "AFPi0", "AFPiU", "AFPiR", "AFPs", "AFPcp", "AFPxy", "AFPcpxy", "AFPpf", "T8i")
TASK_CONTROL = {
    "AFPcp": "trained on the pressure coefficient p / (U^2 / 2), scored in Pa",
    "AFPxy": "chord-frame positions (x, y) as extra node inputs",
    "AFPcpxy": "trained on the pressure coefficient, chord-frame positions as extra node inputs",
    "AFPpf": "Bernoulli pressure of the no-learning Kutta potential flow as an extra node input",
    "AFPiU": "ablation: no U^2 scaling (node inputs (sdf, U, a) + inflow 1-form, plain output)",
    "AFPs": "ablation: no inflow 1-form (node inputs (U, sdf, a), U^2 scaling from the scalar U)",
    "AFPiR": "variant: log U as an extra node input (weak Reynolds-number dependence)",
    "AFPi0": "variant: no angle-of-attack scalar (node input sdf only)",
    "T8i": "original T8 construction (2000 random CFD nodes, Delaunay of the convex hull)",
}
NAME_CONTROL = {
    "nosolve": "ablation: no projection layer (four polynomial layers)",
    "noenergy": "ablation: no energy readout",
    "nokutta": "ablation: projection layer without the harmonic (Kutta) correction",
    "normgate": "ablation: bounded radial gate instead of the homogeneous gate",
    "noscale": "ablation: no U^2 scaling",
    "noform": "ablation: no inflow 1-form",
}


def control_of(task: str, name: str) -> str:
    parts = [TASK_CONTROL[task]] if task in TASK_CONTROL else []
    for key, text in NAME_CONTROL.items():
        if key in name and text not in parts:
            parts.append(text)
    return "; ".join(parts)


def collect(root: str, tasks=None) -> list[dict]:
    rows = []
    for d in sorted(glob.glob(os.path.join(root, "*", "*_s*"))):
        task = os.path.basename(os.path.dirname(d))
        if task in ("preds", "triage", "T8res") or (tasks and task not in tasks):
            continue
        rp = os.path.join(d, "result.json")
        if not os.path.exists(rp):
            continue
        R = json.load(open(rp))
        name = os.path.basename(d)
        base, _, seed = name.rpartition("_s")
        mp = os.path.join(d, "pressure_metrics.json")
        P = json.load(open(mp)) if os.path.exists(mp) else {}
        keep = ("SSIM", "Pearson", "NRMSE", "R2")
        t100 = P.get("test100") or {k: R.get("test100", {}).get(k) for k in keep}
        row = dict(run=name, task=task, model=R.get("model", "rhmp"), params=R.get("params"), seed=int(seed),
                   best_val_R2=R.get("best_val_R2"), best_epoch=R.get("best_epoch"), complete=bool(R.get("complete")),
                   epochs_run=R.get("epochs_run"), test100={k: t100.get(k) for k in keep},
                   zero_shot={k: {m: v.get(m) for m in keep + ("wall_R2",)} for k, v in P.get("zero_shot", {}).items()},
                   rotation={k: v for k, v in P.get("rotation", {}).items()},
                   rotation_max_rel_change=P.get("rotation_max_rel_change", {}),
                   wall={k: P.get("wall", {}).get(k) for k in keep} if P.get("wall") else {},
                   lift=P.get("lift", {}), control=control_of(task, base), s_per_epoch=R.get("s_per_epoch"),
                   device=R.get("device"), gpu=R.get("gpu"))
        if not P:                                                 # no zero-shot evaluation yet: the trainer's sets
            for k in ("fine", "coarse"):
                m = R.get(k + "_test100")
                if m and "R2" in m:
                    row["zero_shot"][k] = {q: m.get(q) for q in keep}
        rows.append(row)
    return rows


def _f(v, fmt=".4f"):
    return "-" if v is None else format(v, fmt)


def markdown(rows: list[dict]) -> str:
    out = ["| task | run | params | val R2 | SSIM | Pearson | NRMSE | R2 | fine R2 | coarse R2 | rot 30 / 90 / 137 | wall R2 | "
           "lift MAE | control |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---|"]
    for r in sorted(rows, key=lambda r: (r["task"], -(r["test100"].get("R2") or -9))):
        t, z, rot = r["test100"], r["zero_shot"], r["rotation"]
        out.append(f"| {r['task']} | {r['run']}{'' if r['complete'] else ' (incomplete)'} | {r['params']} | "
                   f"{_f(r['best_val_R2'])} | {_f(t.get('SSIM'))} | {_f(t.get('Pearson'))} | {_f(t.get('NRMSE'))} | "
                   f"{_f(t.get('R2'))} | {_f(z.get('fine', {}).get('R2'))} | {_f(z.get('coarse', {}).get('R2'))} | "
                   f"{' / '.join(_f(rot.get(k), '.3f') for k in ('30', '90', '137'))} | "
                   f"{_f(r['wall'].get('R2') if r['wall'] else None)} | {_f(r['lift'].get('mae') if r['lift'] else None)} | "
                   f"{r['control']} |")
    return "\n".join(out)


def seed_table(rows: list[dict]) -> str:
    """Mean and standard deviation over the seeds of every (task, run name) with more than one complete seed."""
    import statistics
    groups: dict = {}
    for r in rows:
        if r["complete"]:
            groups.setdefault((r["task"], r["run"].rpartition("_s")[0]), []).append(r)
    out = ["| task | run | seeds | R2 mean | R2 std | R2 per seed | SSIM | Pearson | NRMSE | val R2 mean |",
           "|---|---|---|---:|---:|---|---:|---:|---:|---:|"]
    for (task, name), g in sorted(groups.items(), key=lambda kv: (kv[0][0], -statistics.mean(
            x["test100"]["R2"] for x in kv[1]))):
        if len(g) < 2:
            continue
        g = sorted(g, key=lambda x: (x["seed"] != 42, x["seed"]))
        r2 = [x["test100"]["R2"] for x in g]

        def mean(key):
            return statistics.mean(x["test100"][key] for x in g)
        out.append(f"| {task} | {name} | {', '.join(str(x['seed']) for x in g)} | {statistics.mean(r2):.4f} | "
                   f"{statistics.pstdev(r2):.4f} | {' / '.join(f'{v:.4f}' for v in r2)} | {mean('SSIM'):.4f} | "
                   f"{mean('Pearson'):.4f} | {mean('NRMSE'):.4f} | "
                   f"{statistics.mean(x['best_val_R2'] for x in g):.4f} |")
    return "\n".join(out)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="runs/airfrans")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tasks", nargs="*", default=list(MAIN_TASKS))
    ap.add_argument("--md", action="store_true")
    ap.add_argument("--all", action="store_true", help="include incomplete runs")
    a = ap.parse_args(argv)
    rows = collect(a.root, a.tasks)
    if not a.all:
        rows = [r for r in rows if r["complete"]]
    out = a.out or os.path.join(a.root, "results_pressure.json")
    with open(out, "w") as f:
        json.dump(rows, f, indent=1)
    print(f"{len(rows)} runs -> {out}")
    if a.md:
        print(markdown(rows))
        print()
        print(seed_table(rows))


if __name__ == "__main__":
    main()
