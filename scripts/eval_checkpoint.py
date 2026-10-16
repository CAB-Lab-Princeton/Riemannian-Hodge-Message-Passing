#!/usr/bin/env python3
"""Re-evaluate the checkpoint of a finished run and compare with its stored ``result.json``.

    python3 scripts/eval_checkpoint.py results/torus_transport/rhmp_s42 [RUN_DIR ...] [--extras] [--out FILE.json]

Reproduces the test metrics of a run directory (``config.json``, ``best.pt``, ``result.json``): the task is loaded as
the run loaded it, the model is rebuilt from ``best.pt`` (RHMP or a baseline), the data are expressed in the run's
normalisation, and the full test split (``test``) and the first 100 test samples of the paper tables (``test100``) are
scored with ``rhmp.train.evaluate``, i.e. exactly as at the end of training.  ``--extras`` also scores the task's extra
test sets (zero-shot grids / meshes stored in ``result.json`` under their own keys, e.g. ``fine``, ``coarse``,
``zs_U64``).  Every metric is printed next to the stored value with its relative difference; ``--out`` writes the
comparison as JSON.  Nothing is written into the run directory.

Run-to-run differences of the order of 1e-7 (relative) are expected on GPUs: CUDA atomics and TF32 kernels do not
round identically from one evaluation to the next, and runs trained on another GPU model deviate at the 1e-6 level.
Needs the data set of the task (datasets/README.md); uses a GPU when one is visible.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import ROOT, load_run  # noqa: E402
from rhmp.train import evaluate  # noqa: E402

KEYS = ("R2", "NRMSE", "SSIM", "Pearson")


def reproduce(run_dir: str, device, extras: bool) -> dict:
    stored = json.load(open(os.path.join(run_dir, "result.json")))
    model, view, conf, ebs, _ = load_run(run_dir, device)
    a = conf["args"]
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = not a.get("no_tf32", False)
        torch.backends.cudnn.allow_tf32 = not a.get("no_tf32", False)
    todo = [("test", view.split[2], None)]
    pe = view.meta.get("paper_eval")
    if pe is not None and len(pe):
        todo.append(("test100", pe, None))
    if extras:
        todo += [(name, torch.arange(len(src["target"])), src) for name, src in view.extra_tests.items()]
    out = {"run": os.path.relpath(run_dir, ROOT), "task": a["task"], "model": a.get("model") or "rhmp",
           "params": stored.get("params"), "sets": {}}
    worst = 0.0
    t0 = time.time()
    for name, idx, src in todo:
        if name not in stored or not isinstance(stored[name], dict) or "R2" not in stored[name]:
            continue
        with torch.no_grad():
            r = evaluate(model, view, idx, ebs, device, source=src)
        row = {}
        for k in KEYS:
            s, v = float(stored[name][k]), float(r[k])
            row[k] = {"stored": s, "now": v, "rel_diff": abs(v - s) / max(abs(s), 1e-30)}
        out["sets"][name] = row
        if name in ("test", "test100"):
            worst = max(worst, row["R2"]["rel_diff"], row["NRMSE"]["rel_diff"])
        print(f"  {name:14s} R2 {row['R2']['stored']:.10f} -> {row['R2']['now']:.10f} (rel {row['R2']['rel_diff']:.1e})   "
              f"NRMSE {row['NRMSE']['stored']:.8f} -> {row['NRMSE']['now']:.8f} (rel {row['NRMSE']['rel_diff']:.1e})",
              flush=True)
    out["seconds"] = time.time() - t0
    out["max_rel_diff_test_test100"] = worst
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run directories (config.json, best.pt, result.json)")
    ap.add_argument("--extras", action="store_true", help="also score the extra test sets of the task")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="JSON file for the comparison (default: print only)")
    a = ap.parse_args()
    device = torch.device(a.device)
    res = {"_torch": torch.__version__,
           "_device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"}
    for rd in a.runs:
        rd = os.path.abspath(rd)
        print(f"== {os.path.relpath(rd, ROOT)}", flush=True)
        res[os.path.relpath(rd, ROOT)] = reproduce(rd, device, a.extras)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)
        print(f"-> {a.out}")


if __name__ == "__main__":
    main()
