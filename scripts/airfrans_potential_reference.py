"""Physics reference for the AirfRANS pressure tasks (no learning): P1 potential flow on the conforming near-field
meshes (``datasets/v2/AFM*.pt``) with the Kutta condition, and the Bernoulli pressure.

    python3 scripts/airfrans_potential_reference.py [--set AFM] [--ids 850:950] [--out runs/airfrans/potential_ref.json]

Per mesh (lengths in chords, velocities in m/s, kinematic pressure in m^2/s^2 as stored):

* inflow 1-form ``theta_e = u_inf . (x_dst - x_src)``;
* perturbation potential ``phi = -(d0^T H d0)^-1 d0^T H theta`` with ``H`` = the cotan star (P1 stiffness), the natural
  condition at the wall (no penetration of the total flow) and ``phi = 0`` on the outer ring;
* circulation: the harmonic 1-cochain ``gamma`` of the annulus (closed, co-closed with the natural condition on both
  boundaries, period 1 counter-clockwise), amplitude from (a) the Kutta condition at the trailing edge (the tangential
  velocities on the two wall edges at the trailing-edge vertex cancel: ``<gamma, u>_W = 0`` with ``W`` supported on
  those two edges), (b) the circulation of the CFD field on the outer ring (oracle);
* Bernoulli: ``p_i = (U^2 - |u|^2_i) / 2`` with ``|u|^2_i`` the area-weighted mean of the (constant) squared P1
  velocities of the triangles around vertex ``i``.

Reported: R2 (centred on the pooled mean) / Pearson of the Bernoulli pressure against the CFD pressure over all vertices,
wall vertices and off-wall vertices, for no circulation, Kutta and oracle circulation, and the same with the CFD
velocities at the vertices (the reference of the task notes).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def mesh(blob: dict, i: int) -> dict:
    p0, p1, p2 = blob["ptr0"], blob["ptr1"], blob["ptr2"]
    sl0, sl1, sl2 = slice(int(p0[i]), int(p0[i + 1])), slice(int(p1[i]), int(p1[i + 1])), slice(int(p2[i]), int(p2[i + 1]))
    m = dict(X=blob["pos"][sl0].numpy().astype(np.float64), F=blob["faces"][sl2].numpy().astype(np.int64),
             E=blob["edges"][sl1].numpy().astype(np.int64), kind=blob["kind"][sl0].numpy(),
             u=blob["u"][sl0].numpy().astype(np.float64), p=blob["p"][sl0].numpy().astype(np.float64),
             sdf=blob["sdf"][sl0].numpy().astype(np.float64), theta_cfd=blob["theta"][sl1].numpy(),
             U=float(blob["meta"]["U"][i]), aoa=float(blob["meta"]["aoa"][i]))
    return m


def operators(m: dict) -> dict:
    X, F, E = m["X"], m["F"], m["E"]
    n0, n1, n2 = len(X), len(E), len(F)
    d0 = sp.csr_matrix((np.concatenate([-np.ones(n1), np.ones(n1)]),
                        (np.concatenate([np.arange(n1)] * 2), np.concatenate([E[:, 0], E[:, 1]]))), shape=(n1, n0))
    key = {(int(a), int(b)): j for j, (a, b) in enumerate(E)}
    fe = np.zeros((n2, 3), dtype=np.int64)
    fs = np.zeros((n2, 3))
    cot = np.zeros((n2, 3))
    for j in range(3):                                   # edge j joins local vertices j and j+1; opposite vertex j+2
        a, b, c = F[:, j], F[:, (j + 1) % 3], F[:, (j + 2) % 3]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        fe[:, j] = [key[(int(x), int(y))] for x, y in zip(lo, hi)]
        fs[:, j] = np.where(a < b, 1.0, -1.0)
        v1, v2 = X[a] - X[c], X[b] - X[c]
        cot[:, j] = (v1 * v2).sum(1) / np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    w = np.zeros(n1)
    np.add.at(w, fe.ravel(), 0.5 * cot.ravel())          # unclamped cotan star = P1 stiffness weights
    area = 0.5 * np.abs((X[F[:, 1], 0] - X[F[:, 0], 0]) * (X[F[:, 2], 1] - X[F[:, 0], 1]) -
                        (X[F[:, 1], 1] - X[F[:, 0], 1]) * (X[F[:, 2], 0] - X[F[:, 0], 0]))
    d1 = sp.csr_matrix((fs.ravel(), (np.repeat(np.arange(n2), 3), fe.ravel())), shape=(n2, n1))
    return dict(d0=d0, d1=d1, w=w, fe=fe, fs=fs, cot=cot, area=area, n=(n0, n1, n2))


def closed_generator(m: dict, op: dict) -> np.ndarray:
    """A closed, non-exact 1-cochain with period 1 around the airfoil (counter-clockwise): the coboundary of the
    polar angle around a point inside the airfoil, divided by 2 pi."""
    X, E = m["X"], m["E"]
    wall = m["kind"] == 1
    xc = np.array([0.5 * (X[wall, 0].min() + X[wall, 0].max()), 0.0])
    for attempt in range(40):
        ang = np.arctan2(X[:, 1] - xc[1], X[:, 0] - xc[0])
        c = ang[E[:, 1]] - ang[E[:, 0]]
        c = (c + np.pi) % (2 * np.pi) - np.pi
        if np.abs(op["d1"] @ c).max() < 1e-9:
            return c / (2 * np.pi)
        # the point is not inside the airfoil: move it to the centroid of a wall slice
        sel = wall & (np.abs(X[:, 0] - (0.2 + 0.015 * attempt)) < 0.02)
        xc = X[sel].mean(0)
    raise RuntimeError("no interior point found")


def solve_dirichlet(Kmat, rhs: np.ndarray, fixed: np.ndarray) -> np.ndarray:
    free = np.where(~fixed)[0]
    out = np.zeros(len(rhs))
    out[free] = spla.spsolve(Kmat[free][:, free].tocsc(), rhs[free])
    return out


def face_sq_velocity(u: np.ndarray, op: dict) -> np.ndarray:
    """|u_f|^2 of a closed 1-cochain: (1 / A_f) sum_e cot_e u_e^2 / 2 (exact for the P1 / Whitney interpolant)."""
    return (0.5 * op["cot"] * u[op["fe"]] ** 2).sum(1) / op["area"]


def vertex_mean(q_f: np.ndarray, m: dict, op: dict) -> np.ndarray:
    n0 = op["n"][0]
    num, den = np.zeros(n0), np.zeros(n0)
    for j in range(3):
        np.add.at(num, m["F"][:, j], q_f * op["area"])
        np.add.at(den, m["F"][:, j], op["area"])
    return num / den


def potential_flow(m: dict, clamp: float = 0.0) -> dict:
    op = operators(m)
    X, E, kind = m["X"], m["E"], m["kind"]
    d0, w = op["d0"], op["w"]
    if clamp > 0:                                        # the clamped cotan star of rhmp.geometry (kappa * median)
        w = np.maximum(w, clamp * np.median(w))
    uinf = m["U"] * np.array([np.cos(m["aoa"]), np.sin(m["aoa"])])
    theta = (X[E[:, 1]] - X[E[:, 0]]) @ uinf
    Kmat = (d0.T @ sp.diags(w) @ d0).tocsr()
    ring = kind == 2
    phi = solve_dirichlet(Kmat, -(d0.T @ (w * theta)), ring)
    u0 = theta + d0 @ phi
    c = closed_generator(m, op)
    pin = np.zeros(len(X), dtype=bool)
    pin[0] = True
    psi = solve_dirichlet(Kmat, d0.T @ (w * c), pin)                 # pure Neumann problem, one vertex pinned
    gam = c - d0 @ psi
    # wall loop: the boundary edges with two wall ends; trailing edge = wall vertex of largest x
    wall = kind == 1
    is_wall_edge = wall[E[:, 0]] & wall[E[:, 1]] & (np.asarray(np.abs(op["d1"]).sum(0)).ravel() == 1)
    te = np.where(wall)[0][np.argmax(X[wall, 0])]
    te_edges = np.where(is_wall_edge & ((E[:, 0] == te) | (E[:, 1] == te)))[0]
    ell = np.linalg.norm(X[E[:, 1]] - X[E[:, 0]], axis=1)
    W = np.zeros(len(E))
    W[te_edges] = 1.0 / ell[te_edges] ** 2
    G_kutta = (W * gam * u0).sum() / (W * gam * gam).sum()
    # CFD circulation on the outer ring (counter-clockwise) = sum over ring boundary edges of theta_cfd * gamma sign
    is_ring_edge = ring[E[:, 0]] & ring[E[:, 1]] & (np.asarray(np.abs(op["d1"]).sum(0)).ravel() == 1)
    sgn = np.sign(c[is_ring_edge])                                   # orientation of each ring edge relative to ccw
    circ_cfd = float((m["theta_cfd"][is_ring_edge] * sgn).sum())
    is_wall_adj = is_wall_edge
    out = dict(U=m["U"], aoa=m["aoa"], Gamma_kutta=float(-G_kutta), Gamma_cfd=circ_cfd, n_te_edges=int(len(te_edges)),
               closed=float(np.abs(op["d1"] @ gam).max()), period=float((gam[is_wall_adj] * np.sign(c[is_wall_adj])).sum()))
    fields = {}
    for name, G in (("nocirc", 0.0), ("kutta", G_kutta), ("oracle", -circ_cfd)):
        u = u0 - G * gam
        q = vertex_mean(face_sq_velocity(u, op), m, op)
        fields[name] = 0.5 * (m["U"] ** 2 - q)
    # diagonal-star vertex energy of the Kutta flow (the formula of the energy readout)
    u = u0 - G_kutta * gam
    star0 = np.zeros(len(X))
    for j in range(3):
        np.add.at(star0, m["F"][:, j], op["area"] / 3.0)
    qd = (np.abs(d0).T @ (w * u * u)) / (2.0 * star0)
    fields["kutta_diag"] = 0.5 * (m["U"] ** 2 - qd)
    fields["cfd_bernoulli"] = 0.5 * (m["U"] ** 2 - (m["u"] ** 2).sum(1))
    return dict(out, fields=fields, p=m["p"], kind=kind, sdf=m["sdf"])


def r2(p, t):
    p, t = np.concatenate(p), np.concatenate(t)
    return float(1.0 - ((p - t) ** 2).sum() / ((t - t.mean()) ** 2).sum())


def pearson(p, t):
    return float(np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(p, t)]))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--set", default="AFM")
    ap.add_argument("--ids", default="850:950", help="start:stop[:step] of the sample indices of the set")
    ap.add_argument("--out", default=None)
    ap.add_argument("--clamp", type=float, default=0.0, help="clamp the cotan star below at CLAMP * median (rhmp: 1e-2)")
    ap.add_argument("--fit-ids", default=None, help="fit p = a p_kutta + b on these samples and report the calibrated R2")
    a = ap.parse_args(argv)
    blob = torch.load(os.path.join(ROOT, "datasets", "v2", a.set + ".pt"), map_location="cpu", weights_only=False)
    ids = list(range(*[int(v) for v in a.ids.split(":")]))
    res = [potential_flow(mesh(blob, i), a.clamp) for i in ids]
    names = list(res[0]["fields"])
    summ = {"set": a.set, "ids": a.ids, "n": len(res)}
    for sel_name, sel in (("all", lambda r: np.ones(len(r["p"]), bool)), ("wall", lambda r: r["kind"] == 1),
                          ("off_wall", lambda r: r["kind"] != 1), ("sdf>0.02", lambda r: r["sdf"] > 0.02)):
        for nm in names:
            P = [r["fields"][nm][sel(r)] for r in res]
            T = [r["p"][sel(r)] for r in res]
            summ[f"{sel_name}/{nm}"] = dict(R2=r2(P, T), Pearson=pearson(P, T))
            # dynamic-pressure units: per-simulation Cp errors weighted equally
            Pc = [x / (0.5 * r["U"] ** 2) for x, r in zip(P, res)]
            Tc = [x / (0.5 * r["U"] ** 2) for x, r in zip(T, res)]
            summ[f"{sel_name}/{nm}"]["R2_cp"] = r2(Pc, Tc)
    gk = np.array([r["Gamma_kutta"] for r in res])
    gc = np.array([r["Gamma_cfd"] for r in res])
    summ["circulation"] = dict(corr=float(np.corrcoef(gk, gc)[0, 1]), slope=float((gk * gc).sum() / (gc * gc).sum()),
                               kutta_over_cfd_median=float(np.median(gk / gc)),
                               rel_err_median=float(np.median(np.abs(gk - gc) / np.abs(gc))))
    if a.fit_ids:                                        # one affine calibration of Cp fitted on other samples
        fit = [potential_flow(mesh(blob, i), a.clamp) for i in range(*[int(v) for v in a.fit_ids.split(":")])]
        x = np.concatenate([r["fields"]["kutta"] / (0.5 * r["U"] ** 2) for r in fit])
        y = np.concatenate([r["p"] / (0.5 * r["U"] ** 2) for r in fit])
        A = np.stack([x, np.ones_like(x)], 1)
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        P = [(coef[0] * r["fields"]["kutta"] / (0.5 * r["U"] ** 2) + coef[1]) * 0.5 * r["U"] ** 2 for r in res]
        summ["calibrated_kutta"] = dict(slope=float(coef[0]), offset_cp=float(coef[1]),
                                        R2=r2(P, [r["p"] for r in res]),
                                        R2_wall=r2([p_[r["kind"] == 1] for p_, r in zip(P, res)],
                                                   [r["p"][r["kind"] == 1] for r in res]))
    summ["checks"] = dict(closed_max=float(max(r["closed"] for r in res)),
                          period=[float(min(r["period"] for r in res)), float(max(r["period"] for r in res))],
                          te_edges=[int(min(r["n_te_edges"] for r in res)), int(max(r["n_te_edges"] for r in res))])
    print(json.dumps(summ, indent=1))
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        json.dump(summ, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
