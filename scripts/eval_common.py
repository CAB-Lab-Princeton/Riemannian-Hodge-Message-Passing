"""Shared helper of the evaluation scripts: rebuild a finished run (``config.json`` + ``best.pt``) for evaluation.

``load_run(run_dir, device)`` loads the task exactly as ``rhmp.train.run`` loaded it for the run (the arguments stored
in ``config.json``), rebuilds the model from ``best.pt`` (RHMP or a baseline of ``rhmp.baselines.registry``), re-expresses
the data in the normalisation of the training run (``rhmp.train._renormalize``, as ``--eval-ckpt`` does) and returns
``(model, task_view, config, eval_batch_size, renormalised)``.  Tasks are cached per process, so several runs of the
same task share one loaded data set.  The library is imported from the repository that contains this file (set
``RHMP_DATA_ROOT`` to read the data sets from another directory).
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault("RHMP_DATA_ROOT", ROOT)

from rhmp.tasks import load_task  # noqa: E402
from rhmp.train import _renormalize, needs_whitney  # noqa: E402

_TASK_CACHE: dict = {}


def load_for_run(conf: dict, device) -> object:
    """The task of a run, loaded with the run's arguments (``conf['args']``)."""
    a = conf["args"]
    model_name = a.get("model") or "rhmp"
    star = a.get("star", "cotan")
    if model_name != "rhmp":
        from rhmp.baselines.registry import SPECS
        need = SPECS[model_name].data_star if model_name in SPECS else None
        if need:
            star = need
    kw = {}
    if a.get("max_samples") is not None:
        kw["max_samples"] = a["max_samples"]
    if a.get("no_fine"):
        kw["fine"] = False
    if a.get("no_output_map"):
        kw["output_map"] = False
    if a.get("data_on", "auto") != "auto":
        kw["keep_on"] = str(device) if a["data_on"] == "device" else "cpu"
    if not (model_name == "rhmp" and needs_whitney(conf.get("model", {}))):
        kw["whitney"] = False               # Whitney blocks: tensor metrics / energy readouts / Picard layers only
    if a.get("abs_scale"):
        kw["abs_scale"] = True
    key = (a["task"], bool(a.get("native", True)), star, tuple(sorted(kw.items())))
    if key not in _TASK_CACHE:
        t0 = time.time()
        _TASK_CACHE[key] = load_task(a["task"], a.get("root"), native=bool(a.get("native", True)), device=device,
                                     star=star, **kw)
        print(f"    loaded task {key[0]} ({'native' if key[1] else 'legacy'}) in {time.time() - t0:.1f}s", flush=True)
    return _TASK_CACHE[key]


def task_view(task):
    """A shallow copy of a cached task whose mutable containers can be modified per run."""
    return dataclasses.replace(task, extra_tests={k: dict(v) for k, v in task.extra_tests.items()},
                               meta=dict(task.meta))


def same_stats(task, summary: dict) -> bool:
    """Whether the task's normalisation statistics equal those stored in a run's ``config.json['data']``."""
    def eq(st, d):
        return np.array_equal(st.mean.detach().cpu().numpy().astype(np.float32), np.asarray(d["mean"], np.float32)) \
            and np.array_equal(st.std.detach().cpu().numpy().astype(np.float32), np.asarray(d["std"], np.float32))
    try:
        if not eq(task.y_stats, summary["y_stats"]):
            return False
        xs = {int(k): v for k, v in summary["x_stats"].items()}
        return set(xs) == set(task.x_stats) and all(eq(task.x_stats[k], xs[k]) for k in xs)
    except (KeyError, TypeError):
        return False


def load_run(run_dir: str, device):
    """``(model, task view, config, eval batch size, renormalised)`` of a finished run (see the module docstring)."""
    from rhmp.model import RHMP
    conf = json.load(open(os.path.join(run_dir, "config.json")))
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    view = task_view(load_for_run(conf, device))
    trained = conf["args"].get("model") or "rhmp"
    if trained == "rhmp":
        model = RHMP.from_checkpoint(ck, map_location=device)
    else:
        from rhmp.baselines.registry import SPECS, from_checkpoint, prepare_task
        view = prepare_task(trained, view, ck.get("build"))
        model = from_checkpoint(ck, view, map_location=device)
        if not SPECS[trained].uses_output_map and view.output_map is not None:
            view = dataclasses.replace(view, output_map=None)
    model.eval()
    renorm = not same_stats(view, conf["data"])
    if renorm:
        view = _renormalize(view, conf["data"])
    if hasattr(model, "record_diagnostics"):
        model.record_diagnostics = True                  # as in the trainer's test evaluation
    a = conf["args"]
    if view.variable_mesh:
        bs = a.get("bs_meshes") or view.meta.get("batch_size", 8)
    else:
        bs = a.get("batch") or view.meta.get("batch_size", 64)
    ebs = a.get("eval_batch") or bs
    return model, view, conf, ebs, renorm
