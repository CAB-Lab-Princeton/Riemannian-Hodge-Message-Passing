"""What the projection layer of a trained pressure model has learned (tasks AFPi* of ``rhmp/tasks/airfrans.py``).

    python3 scripts/airfrans_pressure_analysis.py RUN_DIR [--device cpu] [--n 100]

On the test100 simulations the unit inflow 1-form is sent through the model's first projection layer with the metrics
the model computes (``RHMP.projection_fields``):

* Kutta metric: share of the ``W``-energy of the harmonic cochain carried by the two wall edges at the trailing-edge
  vertex, and the distance of the largest edge weight from the trailing edge;
* circulation of the projected flow around the airfoil against the circulation of the CFD velocity on the outer ring
  (both in units of ``U`` x chord): correlation, slope, median ratio; the same for the Kutta condition with the
  reference metric (``scripts/airfrans_potential_reference.py``) if its JSON is given;
* Bernoulli pressure of the projected flow, ``(U^2 / 2)(1 - |u|^2)`` with the Galerkin vertex energy, against the CFD
  pressure (R2 over all / wall vertices): the inviscid flow inside the trained model;
* the learned edge metric relative to the reference star (mean of ``log(H / star)`` minus its mean over the mesh) in
  bins of the wall distance of the edge.

Output: ``RUN_DIR/pressure_analysis.json``.
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
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main(argv=None) -> None:
    from airfrans_pressure_eval import load_run
    from rhmp.data import mesh_minibatch
    from rhmp.dec import vertex_energy
    from rhmp.tasks import resolve_root
    from rhmp.tasks.airfrans import afm_eval_source
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args(argv)
    device = torch.device(a.device)
    run = load_run(a.run, device)
    model = run["model"]
    src = afm_eval_source(run["task"], resolve_root(run["root"]), run["x_stats"], run["y_stats"], native=True,
                          star=run["star"], whitney=True)
    n = min(a.n, len(src["K"]))
    bins = [0.0, 0.004, 0.01, 0.02, 0.05, 0.1, 0.2, 0.6]
    rows, logh_sum, logh_cnt = [], np.zeros(len(bins) - 1), np.zeros(len(bins) - 1)
    P_b, T_b, P_w, T_w = [], [], [], []
    for i in range(n):
        K, xb, _ = mesh_minibatch(src["K"], src["inputs"], src["target"], [i], device=device)
        with torch.no_grad():
            f = model.projection_fields(xb, K)
        X = src["pos0"][i].numpy()
        kind = src["kind"][i].numpy()
        U, aoa = src["U"][i], src["aoa"][i]
        e = K.cells[1].long().cpu().numpy()
        ev = X[e[:, 1]] - X[e[:, 0]]
        unit = ev @ np.array([np.cos(aoa), np.sin(aoa)])                     # cochain of the unit inflow
        probe = f["probe"][:, 0, 0].double().cpu().numpy()
        k = float(np.sqrt((unit ** 2).sum() / (probe ** 2).sum()))
        u = f["u"][:, 0, 0].double().cpu().numpy() * k                       # projected flow for unit inflow speed
        fe = K.meta["face_edges"].cpu().numpy()
        fs = K.meta["face_edge_signs"].cpu().numpy()
        sgn = np.zeros(K.n[1])
        bnd = K.boundary[1].cpu().numpy().astype(bool)
        sgn[fe.ravel()] = fs.ravel()                                         # boundary edges: one face each
        ring = bnd & (kind[e[:, 0]] == 2) & (kind[e[:, 1]] == 2)
        wall_e = bnd & (kind[e[:, 0]] == 1) & (kind[e[:, 1]] == 1)
        circ_model = float((u * sgn)[ring].sum())                            # counter-clockwise, units U x chord
        circ_cfd = float((src["theta_cfd"][i].numpy() * sgn)[ring].sum() / U)
        row = dict(src=src["src"][i], U=U, aoa=aoa, circ_model=circ_model, circ_cfd=circ_cfd)
        if f["kappa"] is not None:
            wall_v = np.where(kind == 1)[0]
            te = wall_v[np.argmax(X[wall_v, 0])]
            te_e = wall_e & ((e[:, 0] == te) | (e[:, 1] == te))
            star1 = K.star[1].double().cpu().numpy()
            W = star1 * np.exp(f["kappa"][:, 0].double().cpu().numpy())
            wg = W * (f["gamma"][:, 0, :].double().cpu().numpy() ** 2).sum(-1)
            j = int(np.argmax(wg))
            mid = 0.5 * (X[e[j, 0]] + X[e[j, 1]])
            row.update(kutta_share_te=float(wg[te_e].sum() / wg.sum()), kutta_share_max=float(wg[j] / wg.sum()),
                       kutta_max_dist_te=float(np.linalg.norm(mid - X[te])),
                       kutta_share_within_0p02=float(wg[np.linalg.norm(0.5 * (X[e[:, 0]] + X[e[:, 1]]) - X[te],
                                                                       axis=1) < 0.02].sum() / wg.sum()),
                       kappa_te=float(f["kappa"][:, 0].double().cpu().numpy()[te_e].mean()),
                       kappa_median=float(np.median(f["kappa"][:, 0].double().cpu().numpy())))
        ut = torch.as_tensor(u, dtype=torch.float64)[:, None]
        q = vertex_energy(K.to("cpu", dtype=torch.float64), ut, ut)[:, 0].numpy()
        pb = 0.5 * U ** 2 * (1.0 - q)
        p = src["p"][i].numpy()
        P_b.append(pb), T_b.append(p), P_w.append(pb[kind == 1]), T_w.append(p[kind == 1])
        if f["log_H_ratio"] is not None:
            lh = f["log_H_ratio"][:, 0].double().cpu().numpy()
            lh = lh - lh.mean()
            se = src["sdf"][i].numpy()[e].mean(1)
            idx = np.digitize(se, bins) - 1
            for b in range(len(bins) - 1):
                logh_sum[b] += lh[idx == b].sum()
                logh_cnt[b] += (idx == b).sum()
        rows.append(row)

    def r2(pr, tg):
        pr, tg = np.concatenate(pr), np.concatenate(tg)
        return float(1 - ((pr - tg) ** 2).sum() / ((tg - tg.mean()) ** 2).sum())

    cm, cc = np.array([r["circ_model"] for r in rows]), np.array([r["circ_cfd"] for r in rows])
    out = dict(run=a.run, n=n,
               circulation=dict(corr=float(np.corrcoef(cm, cc)[0, 1]), slope=float((cm * cc).sum() / (cc * cc).sum()),
                                median_ratio=float(np.median(cm / cc)),
                                rel_err_median=float(np.median(np.abs(cm - cc) / np.abs(cc)))),
               bernoulli_of_projected_flow=dict(R2=r2(P_b, T_b), R2_wall=r2(P_w, T_w)),
               log_H_by_wall_distance={f"{bins[b]:g}-{bins[b + 1]:g}": float(logh_sum[b] / max(logh_cnt[b], 1))
                                       for b in range(len(bins) - 1)} if logh_cnt.sum() else None)
    if "kutta_share_te" in rows[0]:
        for key in ("kutta_share_te", "kutta_share_max", "kutta_max_dist_te", "kutta_share_within_0p02", "kappa_te",
                    "kappa_median"):
            v = np.array([r[key] for r in rows])
            out[key] = dict(mean=float(v.mean()), median=float(np.median(v)), min=float(v.min()), max=float(v.max()))
    params = {n_: float(p_) for n_, p_ in model.named_parameters() if ".proj." in n_}
    out["projection_weights"] = params
    out["samples"] = rows
    with open(os.path.join(a.run, "pressure_analysis.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "samples"}, indent=1))


if __name__ == "__main__":
    main()
