"""Rotation-augmented training of a node baseline on MXS / MXA / MXE (legacy inputs), same protocol as ``rhmp.train``
otherwise.

    python3 -u scripts/maxwell_augment.py --model gem_cnn --param-budget 0.090709 --out runs/maxwell/MXS/gem_cnn_rotaug_s42

Every training sample gets its own random rotation of the mesh about the centre of its bounding box in every step: its
mesh is taken from a pool of 360 pre-rotated copies (1 degree apart; positions rotated, scalar inputs unchanged, the
target vectors rotated; on MXA the frame components (cos 2phi, sin 2phi) of the rolling directions turn by twice the
angle) and the 64 samples of a step form one block-diagonal batch.  Validation (model selection) and
the test metrics use the unrotated mesh, exactly as the plain run; ``scripts/maxwell_eval.py`` adds the rotated test
sets.  Writes ``config.json``, ``history.json``, ``best.pt``, ``last.pt`` and ``result.json`` in the trainer's format.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rhmp.data import concat_cells  # noqa: E402
from rhmp.train import _jsonable, _write_json, evaluate, seed_everything  # noqa: E402

def rotation(deg: float) -> np.ndarray:
    th = math.radians(deg)
    return np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--task", default="MXS")
    p.add_argument("--param-budget", type=float, required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--eta-min", type=float, default=1e-5)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--pool", type=int, default=360)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    from rhmp.baselines.registry import build_model
    from rhmp.complex import CochainComplex
    from rhmp.tasks import load_task
    device = torch.device(a.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, "result.json")
    if os.path.exists(res_path) and json.load(open(res_path)).get("complete"):
        print(f"skipping: {res_path} is complete", flush=True)
        return
    seed_everything(a.seed)
    task = load_task(a.task, None, native=False, device=device)
    budget = int(round(a.param_budget * 1e6))
    seed_everything(a.seed)
    model, info = build_model(a.model, task, budget)
    model = model.to(device)
    n_params = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)
    # pool of rotated meshes (same vertex / face order, positions rotated about the box centre)
    t0 = time.time()
    K0 = task.K
    P = K0.pos.double().cpu().numpy()
    centre = 0.5 * (P.min(0) + P.max(0))
    Fc = K0.cells[2].long().cpu()
    angles = np.arange(a.pool) * (360.0 / a.pool)
    pool, Qs = [], []
    for deg in angles:
        Q = rotation(deg)
        pool.append(CochainComplex.from_triangles(torch.as_tensor((P - centre) @ Q.T + centre), Fc, device=device))
        Qs.append(torch.as_tensor(Q, dtype=torch.float32, device=device))
    Qs = torch.stack(Qs)                                                          # (pool, 2, 2)
    print(f"pool of {a.pool} rotated meshes in {time.time() - t0:.1f}s; model {a.model} {n_params} params", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs, eta_min=a.eta_min)
    gen = torch.Generator().manual_seed(a.seed)
    tr, va, te = task.split
    X, Y = task.inputs[0], task.target                                           # (N, n0, F), (N, n0, 2)
    pairs = task.meta.get("frame_pairs") or []                                  # MXA: rolling-direction frame columns
    ang_rad = torch.as_tensor(np.radians(angles), dtype=X.dtype, device=device)
    sig = {"augment": "per-sample random rotation (pool of %d angles)" % a.pool}
    args_dict = dict(vars(a), native=False, model=a.model, root=None, star="cotan", augment=sig["augment"])
    _write_json(os.path.join(a.out, "config.json"), {"args": args_dict, "model": {"name": a.model, **info},
                                                     "params": n_params, "data": task.summary()})
    history, best_r2, best_ep = [], -float("inf"), 0
    t_fit = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        torch.cuda.synchronize(device)
        t1 = time.time()
        perm = tr[torch.randperm(len(tr), generator=gen)]
        loss_sum, n_steps = 0.0, 0
        for s in range(0, len(perm), a.batch):
            b = perm[s:s + a.batch]
            ang = torch.randint(0, a.pool, (len(b),), generator=gen)
            Kb = CochainComplex.batch([pool[int(i)] for i in ang])
            bd = b.to(device)
            xs = X.index_select(0, bd)
            if pairs:
                from rhmp.tasks.maxwell import rotate_frame_columns
                xs = rotate_frame_columns(xs, task.x_stats[0], pairs, ang_rad[ang.to(device)][:, None])
            xb = {0: concat_cells(list(xs))}
            yrot = torch.einsum("bij,bnj->bni", Qs[ang.to(device)], Y.index_select(0, bd))   # rotated targets
            yb = concat_cells(list(yrot))
            opt.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(xb, Kb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
            opt.step()
            loss_sum += float(loss.detach())
            n_steps += 1
        sched.step()
        torch.cuda.synchronize(device)
        t_train = time.time() - t1
        val = evaluate(model, task, va, a.batch, device)
        rec = {"epoch": ep, "lr": opt.param_groups[0]["lr"], "train_loss": loss_sum / max(n_steps, 1),
               **{f"val_{k}": v for k, v in val.items()}, "R2": val["R2"], "time": t_train,
               "peak_mem_GB": torch.cuda.max_memory_allocated(device) / 2 ** 30}
        history.append(rec)
        if val["R2"] > best_r2:
            best_r2, best_ep = val["R2"], ep
            torch.save({**model.to_checkpoint(), "epoch": ep, "val": val, "task": a.task, "native": False},
                       os.path.join(a.out, "best.pt"))
        torch.save({"model": model.to_checkpoint(), "epoch": ep, "history": history}, os.path.join(a.out, "last.pt"))
        _write_json(os.path.join(a.out, "history.json"), history)
        if ep % 5 == 0 or ep <= 2 or ep == a.epochs:
            print(f"  ep{ep:4d} loss={rec['train_loss']:.5f} val R2={val['R2']:.4f} ({t_train:.1f}s)", flush=True)
    ck = torch.load(os.path.join(a.out, "best.pt"), map_location=device, weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    test = evaluate(model, task, te, a.batch, device)
    result = {"task": a.task, "native": False, "mode": "legacy", "seed": a.seed, "model": a.model, "model_info": info,
              "params": n_params, "params_M": n_params / 1e6, "best_epoch": best_ep, "best_val_R2": best_r2,
              "test": test, "test_R2": test["R2"], "test100": evaluate(model, task, task.meta["paper_eval"], a.batch,
                                                                         device),
              "wall_clock_s": time.time() - t_fit, "epochs_run": len(history),
              "s_per_epoch": float(np.mean([h["time"] for h in history])),
              "peak_mem_GB": max(h["peak_mem_GB"] for h in history), "augmentation": sig["augment"],
              "args": args_dict, "complete": len(history) >= a.epochs}
    _write_json(res_path, _jsonable(result))
    print(f"[{a.task}/{a.model} rot-aug] best val R2={best_r2:.4f}@{best_ep} test100 R2={result['test100']['R2']:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
