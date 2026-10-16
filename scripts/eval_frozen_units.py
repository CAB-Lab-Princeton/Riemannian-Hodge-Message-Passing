#!/usr/bin/env python3
"""Zero-shot transfer of trained RHMP runs to the T1g grids, with the per-mesh normalisers of each mesh (default),
with the normalisers of the training mesh (``frozen_units``: ``RHMPConfig.frozen_units``, applied to the trained
checkpoint without retraining: ``model.freeze_units(K_train)``) and with only the operator normalisation of the training
mesh (``frozen_operators``: star means and block bounds frozen, descriptors relative to each mesh;
``RHMPConfig.frozen_descriptors=False``) or with the per-column rule ``frozen_descriptors='auto'`` (``frozen_auto``;
the decision of every relative column on every grid is recorded in ``frozen_auto_columns``).

    python3 scripts/eval_frozen_units.py results/ns_vorticity/rhmp_s42 [RUN_DIR ...] [--out runs/ns_frozen_eval.json]
                                         [--modes default,frozen_units,frozen_operators,frozen_auto]

Reproduces the zero-shot NS table of the paper (``results/evaluations/ns_zero_shot.json``).

Baseline runs (T1g legacy) are scored the same way in the default mode only (``frozen_units`` is an RHMP option).
Modes not recomputed (``--modes``) keep their entries in ``--out``.

For every run and mode, on the grids ``G1`` (the training grid of T1g), ``U32``, ``G2``, ``U64`` and ``G1rot``:

* ``val``: the 1500 validation samples (7000 .. 8499, trajectories 350 .. 424; zero-shot grids from
  ``datasets/v2/T1_cns_vorticity_graded_val.pkl``, ``gen_T1_graded.py --zero-shot-samples val``);
* ``test100``: the 100 paper-evaluation samples (8500 .. 8599).

Inputs are built as the run's task builds them (native T1pml cochains with minimal-image edge vectors; the T1p runs
use the same layout) and normalised with the statistics stored in the run (``config.json['data']``); targets are the
fine-grid (spectral) vorticity of the graded data set and the output map is the run's (patch-area for ``T1gml``, mean
otherwise).  Metrics as ``rhmp.train.evaluate`` (R2 in normalised units, per-position mean).  A T1p run is trained on
the coarse-stencil target of T1, so on these grids its numbers include that difference of target definitions.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_common import ROOT, load_run  # noqa: E402
from rhmp.tasks import resolve_root  # noqa: E402
from rhmp.tasks.paper import (T1G_AREA_REF, _t1g_complex, _t1g_data, _t1g_native_inputs,  # noqa: E402
                              oriented_face_to_node)
from rhmp.train import evaluate  # noqa: E402

GRIDS = ("G1", "U32", "G2", "U64", "G1rot")
KEYS = ("R2", "SSIM", "Pearson", "NRMSE")


def _stats(d: dict):
    return (torch.tensor(d["mean"], dtype=torch.float32), torch.tensor(d["std"], dtype=torch.float32))


def sources(view, conf: dict, device, area_map: bool) -> dict:
    """``{(grid, split): source}`` with the run's normalisation (see module doc).  Baseline runs (legacy node inputs)
    get the triangulated grids, their node inputs (plus the vertex logs for ``T1gml``) and, under the v1 edge protocol,
    the edge-averaged targets."""
    droot = resolve_root(None)
    main = _t1g_data(droot)
    with open(os.path.join(droot, "v2", "T1_cns_vorticity_graded_val.pkl"), "rb") as f:
        valset = pickle.load(f)
    xs = {int(k): _stats(v) for k, v in conf["data"]["x_stats"].items()}
    ym, ysd = _stats(conf["data"]["y_stats"])
    native = bool(conf["args"].get("native", True)) and (conf["args"].get("model") or "rhmp") == "rhmp"
    edge_protocol = bool(view.meta.get("v1_edge_protocol"))
    out = {}
    for grid in GRIDS:
        Kz = _t1g_complex(main["grids"][grid], native, "cotan", device)
        om = oriented_face_to_node(Kz, 1, T1G_AREA_REF if area_map else None) if native else None
        for split in ("val", "test100"):
            if grid == "G1":
                lo, hi = (7000, 8500) if split == "val" else (8500, 8600)
                X = torch.as_tensor(main["train"]["X_data"][lo:hi])
                Y = torch.as_tensor(main["train"]["Y_data"][lo:hi])
            elif split == "val":
                X, Y = torch.as_tensor(valset["zero_shot"][grid]["X_data"]), torch.as_tensor(valset["zero_shot"][grid]["Y_data"])
            else:
                X = torch.as_tensor(main["zero_shot"][grid]["X_data"][:100])
                Y = torch.as_tensor(main["zero_shot"][grid]["Y_data"][:100])
            if native:
                raw = _t1g_native_inputs(X, Kz, cell_fields=True, vertex_logs=True)
            elif xs[0][0].numel() == 6:                            # legacy T1gml: node inputs + vertex logs
                raw = {0: torch.cat([X, torch.log(X[..., [0, 3]].double()).float()], -1)}
            else:
                raw = {0: X}
            if set(raw) != set(xs):
                raise ValueError(f"input degrees {sorted(raw)} do not match the run's {sorted(xs)}")
            inp = {k: ((v - xs[k][0]) / xs[k][1]).to(device).contiguous() for k, v in raw.items()}
            tgt = ((Y - ym) / ysd).to(device)
            if edge_protocol:                                       # MPSN / SCCNN / Clifford: 0.5 (y_src + y_dst)
                e = Kz.cells[1].long()
                tgt = 0.5 * (tgt.index_select(1, e[:, 0]) + tgt.index_select(1, e[:, 1]))
            out[(grid, split)] = dict(K=Kz, inputs=inp, target=tgt.contiguous(), output_map=om)
    return out


def auto_columns(model, srcs: dict) -> dict:
    """Per grid, the ``frozen_descriptors='auto'`` decision of every relative descriptor column: the grid's median in
    training-mesh units against the [min, max] of the column on the training mesh."""
    from rhmp.geometry import GEO_FEATURE_NAMES, GEO_RELATIVE_COLUMNS
    out = {}
    for (grid, _), src in srcs.items():
        if grid in out:
            continue
        lm, rows = src["K"].meta["geo_log_median"], []
        for k in range(model.top + 1):
            for c in GEO_RELATIVE_COLUMNS[k]:
                med = float(lm[k].reshape(-1)[c]) - float(getattr(model, f"frozen_geo_{k}")[c])
                lo, hi = (float(getattr(model, f"frozen_geo_{m}_{k}")[c]) for m in ("min", "max"))
                rows.append({"degree": k, "column": GEO_FEATURE_NAMES[k][c], "median_in_train_units": med,
                             "train_min": lo, "train_max": hi, "inside": lo <= med <= hi})
        out[grid] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", default=os.path.join(ROOT, "runs", "ns_frozen_eval.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--modes", default=None,
                    help="comma-separated subset of default,frozen_units,frozen_operators,frozen_auto")
    a = ap.parse_args()
    dev = torch.device(a.device)
    res = json.load(open(a.out)) if os.path.exists(a.out) else {}
    for rd in a.runs:
        rd = os.path.abspath(rd)
        key = os.path.relpath(rd, ROOT)
        model, view, conf, ebs, _ = load_run(rd, dev)
        task = conf["args"]["task"]
        is_rhmp = (conf["args"].get("model") or "rhmp") == "rhmp"
        area = is_rhmp and task.startswith("T1g") and task != "T1gml_mean"
        srcs = sources(view, conf, dev, area_map=area)
        row = dict(res.get(key, {}))
        row.update({"task": task, "model": conf["args"].get("model") or "rhmp",
                    "output_map": ("patch-area" if area else "mean") if is_rhmp else None,
                    "training_mesh": "G1" if task.startswith("T1g") else "uniform periodic 32 x 32 (T1p)"})
        modes = ("default", "frozen_units", "frozen_operators", "frozen_auto") if is_rhmp else ("default",)
        for mode in (m for m in modes if a.modes is None or m in a.modes.split(",")):
            if mode != "default":
                model.cfg.frozen_descriptors = {"frozen_units": True, "frozen_operators": False,
                                                "frozen_auto": "auto"}[mode]
                model.freeze_units(view.K)                       # the run's training complex
            if mode == "frozen_auto":
                row["frozen_auto_columns"] = auto_columns(model, srcs)
            out = {}
            for (grid, split), src in srcs.items():
                with torch.no_grad():
                    r = evaluate(model, view, torch.arange(len(src["target"])), ebs, dev, source=src)
                out.setdefault(grid, {})[split] = {k: float(r[k]) for k in KEYS}
                print(f"{key:40s} {mode:12s} {grid:6s} {split:7s} " + " ".join(f"{k} {r[k]:.4f}" for k in KEYS),
                      flush=True)
            row[mode] = out
        if "test100" in json.load(open(os.path.join(rd, "result.json"))):
            rj = json.load(open(os.path.join(rd, "result.json")))
            row["result_json"] = {k: {m: rj[k][m] for m in KEYS} for k in rj if k.startswith("zs_") or k == "test100"}
        res[key] = row
        json.dump(res, open(a.out, "w"), indent=1)
        del model
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
