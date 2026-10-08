"""Helpers for the frame-dependence evaluation of trained RHMP / baseline runs (evaluation only, no training).

Everything here is read-only with respect to the library: helper code lives in this file, nothing is monkey-patched.
The library is imported from the repository that contains this file (``RHMP_ROOT``).  Contents:

* rotations (planar about the mesh centroid; 3-D about one fixed unit axis drawn with a fixed seed);
* the legacy (v1) node encodings of the edge cochains of T6 / T7 (``[avg(c), avg*dx(c), avg*dy(c)]``), recomputed from
  the native edge cochain, and the rotation of the identified frame components;
* rotated test views of a task for the baselines (legacy node-encoded inputs) and, for the variable-mesh AirfRANS task,
  for the native model;
* loading of a finished run exactly as ``rhmp.train.eval_ckpt`` rebuilds it, and the evaluation of a view with the
  library metrics (``rhmp.metrics.summarize`` through ``rhmp.train.predict``).
"""
from __future__ import annotations

import dataclasses
import json
import math
import os
import pickle
import sys

import numpy as np
import torch

# the repository that contains this file (override with RHMP_ROOT); runs are read from RHMP_RUNS (default
# <root>/results, the layout of the shipped result directories: <task>/<model>_s<seed>/)
ROOT = os.path.abspath(os.path.expanduser(os.environ.get(
    "RHMP_ROOT", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

ANGLES = (0, 30, 90, 137, 360)            # degrees; 360 is the identity check
AXIS_SEED = 0                             # seed of the 3-D rotation axis (see ``rotation_axis``)
EDGE_EPS = 1e-12                          # v1 encode_edges_to_nodes: unit direction = edge / (|edge| + 1e-12)


# ======================================================================================================================
# rotations
# ======================================================================================================================
def rotation_axis(seed: int = AXIS_SEED) -> tuple[np.ndarray, int]:
    """Unit axis for the 3-D rotations: normalised Gaussian vectors from ``numpy.random.default_rng(seed + j)``,
    ``j = 0, 1, ...``; the first draw whose three components all have modulus in [0.3, 0.8] is used (so the axis is
    at least 36.9 degrees away from every coordinate axis).  Returns ``(axis, j)``."""
    for j in range(1000):
        u = np.random.default_rng(seed + j).normal(size=3)
        u = u / np.linalg.norm(u)
        if np.all(np.abs(u) >= 0.3) and np.all(np.abs(u) <= 0.8):
            return u, j
    raise RuntimeError("no admissible axis found")


def rot2(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rot3(axis: np.ndarray, deg: float) -> np.ndarray:
    """Rodrigues rotation matrix by ``deg`` degrees about the unit vector ``axis`` (right-handed)."""
    a = math.radians(deg)
    u = np.asarray(axis, dtype=np.float64)
    ux = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return math.cos(a) * np.eye(3) + math.sin(a) * ux + (1 - math.cos(a)) * np.outer(u, u)


def rotation(dim: int, deg: float, axis: np.ndarray | None = None) -> np.ndarray:
    return rot2(deg) if dim == 2 else rot3(axis, deg)


def shift_about(Q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Translation ``t`` such that ``x -> Q x + t`` is the rotation ``Q`` about the point ``c``."""
    return np.asarray(c, dtype=np.float64) - Q @ np.asarray(c, dtype=np.float64)


def mesh_centroid(K) -> np.ndarray:
    """Vertex centroid of a (single-mesh) complex, float64."""
    return K.pos.double().cpu().numpy().mean(0)


def rotate_positions(pos: np.ndarray, Q: np.ndarray, c: np.ndarray) -> np.ndarray:
    """``Q (x - c) + c`` for rows ``x`` of ``pos`` (float64)."""
    return (np.asarray(pos, dtype=np.float64) - c) @ Q.T + c


# ======================================================================================================================
# legacy (v1) node encodings of edge cochains (T6, T7)
# ======================================================================================================================
def encode_edges_to_nodes(A: torch.Tensor, edges: np.ndarray, pts: np.ndarray, n: int) -> torch.Tensor:
    """Legacy node features of an edge cochain, recomputed in float64.

    ``A`` is ``(N, n1, c)`` (value of the cochain on the edge in its canonical orientation src -> dst, src < dst);
    ``t_e = (x_dst - x_src) / (|x_dst - x_src| + 1e-12)`` is the unit edge direction.  Per node ``i`` with the incident
    edges ``E(i)`` and ``deg_i = |E(i)|`` (isolated nodes: 1):

        ``[ mean_E A ,  mean_E A t_x ,  mean_E A t_y ]``   ->   ``(N, n, 3c)``  (blocks of ``c`` columns).

    The first block is invariant under a rotation of the mesh; the two other blocks are the components of the in-plane
    vector ``mean_E A t`` and rotate as ``v -> Q v``.
    """
    A = A.double()
    ev = pts[edges[:, 1]] - pts[edges[:, 0]]
    t = ev / (np.linalg.norm(ev, axis=1, keepdims=True) + EDGE_EPS)
    tx = torch.as_tensor(t[:, 0]).view(1, -1, 1)
    ty = torch.as_tensor(t[:, 1]).view(1, -1, 1)
    feats = torch.cat([A, A * tx, A * ty], dim=-1)                                   # (N, n1, 3c)
    src = torch.as_tensor(edges[:, 0])
    dst = torch.as_tensor(edges[:, 1])
    out = torch.zeros(A.shape[0], n, feats.shape[-1], dtype=torch.float64)
    out.index_add_(1, src, feats)
    out.index_add_(1, dst, feats)
    cnt = torch.zeros(n, dtype=torch.float64)
    cnt.index_add_(0, src, torch.ones(len(src), dtype=torch.float64))
    cnt.index_add_(0, dst, torch.ones(len(dst), dtype=torch.float64))
    cnt[cnt == 0] = 1.0
    return out / cnt.view(1, -1, 1)


def rotate_vector_columns(X: torch.Tensor, Q: np.ndarray, c: int) -> torch.Tensor:
    """Rotate the in-plane vector columns of legacy features ``(N, n, 3c)``: for every channel ``a`` the pair
    ``(X[..., c + a], X[..., 2c + a])`` is transformed by ``Q``; the ``avg`` block ``X[..., :c]`` is left unchanged."""
    Qt = torch.as_tensor(Q, dtype=X.dtype)
    out = X.clone()
    vx, vy = X[..., c:2 * c], X[..., 2 * c:3 * c]
    out[..., c:2 * c] = Qt[0, 0] * vx + Qt[0, 1] * vy
    out[..., 2 * c:3 * c] = Qt[1, 0] * vx + Qt[1, 1] * vy
    return out


# ======================================================================================================================
# TaskData views
# ======================================================================================================================
def _empty():
    return torch.empty(0, dtype=torch.long)


def with_view(task, K, inputs, target, n: int, output_map="keep", **repl):
    """Test-only copy of ``task`` (``split = ([], [], arange(n))``) with new complexes / inputs / targets."""
    om = task.output_map if isinstance(output_map, str) and output_map == "keep" else output_map
    return dataclasses.replace(task, K=K, inputs=inputs, target=target, split=(_empty(), _empty(), torch.arange(n)),
                               output_map=om, meta=dict(task.meta), **repl)


def stats_apply(stats, x: torch.Tensor) -> torch.Tensor:
    return stats.normalize(x)


def rotate_vector_target(task, Y: torch.Tensor, Q: np.ndarray) -> torch.Tensor:
    """Rotate a normalised node-vector target ``(N, n, D)``: de-normalise with the task's statistics, ``v -> Q v`` (rows:
    ``Y Q^T``), normalise again (the statistics are those of the training run, i.e. of the original frame)."""
    st = task.y_stats
    raw = st.denormalize(Y.double())
    raw = raw @ torch.as_tensor(Q, dtype=torch.float64, device=raw.device).T
    mean, std = st.mean.to(raw), st.std.to(raw)
    return ((raw - mean) / std).to(Y.dtype)


def rebuilt_complex(K, Q: np.ndarray, shift: np.ndarray):
    """Complex of the rotated mesh (library ``rhmp.robustness.remap_complex`` with the identity relabelling); asserts that
    the rebuilt complex has the cell order of the original, so that data of every degree stay aligned."""
    from rhmp.robustness import remap_complex
    K_new, maps = remap_complex(K, Q=Q, shift=shift)
    for k, (idx, sgn) in maps.items():
        if not torch.equal(idx.cpu(), torch.arange(len(idx))):
            raise RuntimeError(f"rotated complex has a different cell order in degree {k}")
        if sgn is not None and not bool((sgn.cpu() == 1).all()):
            raise RuntimeError(f"rotated complex has a different cell orientation in degree {k}")
    return K_new


# ======================================================================================================================
# runs
# ======================================================================================================================
@dataclasses.dataclass
class Run:
    task_name: str
    model_name: str
    run_dir: str
    conf: dict
    result: dict
    model: torch.nn.Module
    task: object                   # task as the run was trained / scored on (after prepare_task for baselines)
    task_raw: object               # task as loaded (before prepare_task); edge-protocol views start from this
    is_rhmp: bool
    ebs: int
    build: dict | None = None      # checkpoint build options of a baseline (for prepare_task on rotated views)

    @property
    def edge_protocol(self) -> bool:
        """True for runs scored on the edge-averaged first target component (v1 protocol of mpsn / sccnn / clifford_smpn)."""
        return bool(self.task.meta.get("v1_edge_protocol"))


def load_task_for_run(conf: dict, device, extra_kw: dict | None = None):
    from rhmp.tasks import load_task
    a = conf["args"]
    kw = {"whitney": conf.get("model", {}).get("metric_type") == "tensor"}
    if a.get("abs_scale"):
        kw["abs_scale"] = True
    if a.get("no_output_map"):
        kw["output_map"] = False
    if extra_kw:
        kw.update(extra_kw)
    return load_task(a["task"], a.get("root"), native=a["native"], device=device, star=a.get("star", "cotan"), **kw)


def stats_match(task, conf) -> float:
    """Largest relative difference between the statistics of the loaded task and the ones saved in the run's
    ``config.json`` (mean differences relative to ``max(|mean|, std)``, std differences relative to ``std``).  Not exactly
    zero because the loader's float32 reductions depend on the thread count (summation order)."""
    def rel(a, b_mean, b_std, is_std):
        a = a.cpu().double()
        b_mean = torch.tensor(b_mean, dtype=torch.float32).double()
        b_std = torch.tensor(b_std, dtype=torch.float32).double()
        if is_std:
            return float(((a - b_std).abs() / b_std).max())
        return float(((a - b_mean).abs() / torch.maximum(b_mean.abs(), b_std)).max())
    worst = 0.0
    xs = {int(k): v for k, v in conf["data"]["x_stats"].items()}
    for k, st in task.x_stats.items():
        worst = max(worst, rel(st.mean, xs[k]["mean"], xs[k]["std"], False), rel(st.std, xs[k]["mean"], xs[k]["std"], True))
    ys = conf["data"]["y_stats"]
    worst = max(worst, rel(task.y_stats.mean, ys["mean"], ys["std"], False), rel(task.y_stats.std, ys["mean"], ys["std"], True))
    return worst


def load_run(run_dir: str, task_name: str, model_name: str, device, task_cache: dict | None = None,
             extra_kw: dict | None = None) -> Run:
    """Rebuild a finished run as ``rhmp.train.eval_ckpt`` does (same loader arguments as training; the statistics of
    the loaded task are compared with the saved ones instead of re-normalising)."""
    from rhmp.baselines.registry import SPECS, from_checkpoint, prepare_task
    from rhmp.model import RHMP
    conf = json.load(open(os.path.join(run_dir, "config.json")))
    result = json.load(open(os.path.join(run_dir, "result.json")))
    a = conf["args"]
    key = (a["task"], bool(a["native"]), str(device), json.dumps(extra_kw or {}, sort_keys=True))
    if task_cache is not None and key in task_cache:
        task_raw = task_cache[key]
    else:
        task_raw = load_task_for_run(conf, device, extra_kw)
        if task_cache is not None:
            task_cache.clear()                     # keep one task in memory
            task_cache[key] = task_raw
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location=device, weights_only=False)
    trained = a.get("model") or "rhmp"
    is_rhmp = trained == "rhmp"
    task = task_raw
    if is_rhmp:
        model = RHMP.from_checkpoint(ck, map_location=device)
    else:
        task = prepare_task(trained, task_raw, ck.get("build"))
        model = from_checkpoint(ck, task, map_location=device)
        if not SPECS[trained].uses_output_map and task.output_map is not None:
            task = dataclasses.replace(task, output_map=None)
    model.eval()
    if hasattr(model, "record_diagnostics"):
        model.record_diagnostics = False
    diff = stats_match(task, conf)          # saved statistics refer to the task after prepare_task (edge protocol)
    if diff > 1e-5:
        raise RuntimeError(f"{run_dir}: statistics of the loaded task differ from the run's by {diff:.3e} (relative)")
    ebs = a.get("eval_batch") or ((a.get("bs_meshes") or task.meta.get("batch_size", 8)) if task.variable_mesh
                                  else (a.get("batch") or task.meta.get("batch_size", 64)))
    run = Run(task_name, trained, run_dir, conf, result, model, task, task_raw, is_rhmp, int(ebs),
              None if is_rhmp else ck.get("build"))
    run.stats_rel_diff = diff
    return run


# ======================================================================================================================
# evaluation
# ======================================================================================================================
@torch.no_grad()
def predict_view(run: Run, view, device):
    """Normalised predictions and targets of ``view`` (the library ``predict``)."""
    from rhmp.train import predict
    return predict(run.model, view, view.split[2], run.ebs, device)


def summarize_pred(view, pred, tgt) -> dict:
    """The library metrics of ``rhmp.train.evaluate`` on given predictions."""
    from rhmp import metrics as M
    return M.summarize(pred, tgt, view.y_stats.as_tuple(), r2_std=view.meta.get("r2_std"), n_samples=None, prefix="",
                       center=view.target_kind != "cochain")


def eval_task_samples(run: Run, task, idx: torch.Tensor, device):
    """Library ``evaluate`` of the run's model on samples ``idx`` of ``task`` (as the trainer computes ``test100``)."""
    from rhmp.train import evaluate
    return evaluate(run.model, task, idx, run.ebs, device)


def flat(x) -> torch.Tensor:
    """Concatenate predictions (stacked tensor or ragged list) into a ``(M, d)`` double tensor."""
    if isinstance(x, torch.Tensor):
        return x.reshape(-1, x.shape[-1]).double().cpu()
    return torch.cat([t.reshape(-1, t.shape[-1]) for t in x], 0).double().cpu()


def max_pred_diff(a, b) -> float:
    return float((flat(a) - flat(b)).abs().max())


KEYS = ("R2", "SSIM", "Pearson", "NRMSE", "MSE", "MAE", "R2_v1", "N")


def pick(m: dict) -> dict:
    return {k: (float(m[k]) if k != "N" else int(m[k])) for k in KEYS if k in m}


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return str(o)
    return o
