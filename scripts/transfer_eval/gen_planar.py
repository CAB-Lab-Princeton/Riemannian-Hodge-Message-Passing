"""Test data of T6 / T7 on new planar meshes (zero-shot transfer evaluation).

    python3 scripts/transfer_eval/gen_planar.py --task T6       # -> datasets/v2/transfer/T6_<condition>.pt, stats_T6.json

Generates the test data of the zero-shot conditions of ``run_eval.py`` (CUDA GPU: the field generators replay the CUDA
random streams of the original data sets).

For each condition the fields of the 100 test samples (ids 8500..8599 of the original data set) are evaluated on the mesh of
the condition with the field generator of the original data (``tc_common.PlanarFields``: the random parameters of the
original samples, drawn with the CUDA RNG exactly as ``gen_T{6,7}_native.py``), and inputs / targets are built by the original
definitions:

* edge cochain ``theta`` (T6, ``(S, n1)``) / ``A`` (T7, ``(S, n1, 3)``), canonical edges src < dst, float32;
* face cochain ``plaq = d1 theta`` / ``F = dA + [A, A]`` in the (counter-clockwise) face orientation, float32;
* legacy node inputs ``[avg | avg t_x | avg t_y]`` of the edge cochain (float64 recomputation, ``tc_common.
  encode_edges_to_nodes``; verified against the stored ``X_data`` in ``regen_planar.py``);
* node targets: mean of the face cochain over the incident faces (float64 -> float32).

T7 has an iid white-noise part (0.1 N(0, 1) per edge and component) that depends on the edge set of the mesh: on the original
mesh it is the original draw, on every other mesh it is drawn anew (CPU generator, fixed seed per condition); all smooth
parts (Gaussian gauge bumps, vortices) are those of the original samples.

Conditions: ``orig_regen`` (original mesh, seed-42 Delaunay: the regeneration of the stored data), ``new`` (seed-7 random
Delaunay mesh, 1024 nodes), ``aniso`` (the points of ``new`` re-triangulated by the Delaunay triangulation of the cloud
stretched 3:1 in y, domain unchanged: triangles elongated along x), ``fine`` (seed-11 random Delaunay mesh, 4096 nodes);
supplementary ``new_s<k>`` / ``aniso_s<k>`` (k = 1, 2, 3: point seeds 101, 202, 303) for the variability between mesh
realisations and ``graded`` (seed 13: node density 9x larger at x = 0 than at x = 1, spacing varies by a factor 3).
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import time

import numpy as np
import torch

import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tc_common import (DROOT, N_ALL, TEST100, PlanarFields, delaunay_cloud, edge_face_data, encode_edges_to_nodes,
                       face_values_T6, face_values_T7, faces_to_nodes, jsonable, make_ccw, mesh_stats, signed_areas)

PKL = {"T6": "T6_wilson_loop.pkl", "T7": "T7_yang_mills_su2.pkl"}
# condition -> (points seed, number of points, stretch of the triangulation, white-noise seed for T7)
CONDITIONS = {
    "orig_regen": dict(seed=42, n_pts=1024, stretch=None, white_seed=None),
    "new": dict(seed=7, n_pts=1024, stretch=None, white_seed=7001),
    "aniso": dict(seed=7, n_pts=1024, stretch=(1.0, 3.0), white_seed=7002),
    "fine": dict(seed=11, n_pts=4096, stretch=None, white_seed=7003),
    # supplementary: three further random meshes and their anisotropic re-triangulations (mesh-realisation variability)
    "new_s1": dict(seed=101, n_pts=1024, stretch=None, white_seed=7011),
    "aniso_s1": dict(seed=101, n_pts=1024, stretch=(1.0, 3.0), white_seed=7012),
    "new_s2": dict(seed=202, n_pts=1024, stretch=None, white_seed=7021),
    "aniso_s2": dict(seed=202, n_pts=1024, stretch=(1.0, 3.0), white_seed=7022),
    "new_s3": dict(seed=303, n_pts=1024, stretch=None, white_seed=7031),
    "aniso_s3": dict(seed=303, n_pts=1024, stretch=(1.0, 3.0), white_seed=7032),
    # supplementary: strongly graded node density (density 9x larger at x = 0 than at x = 1: spacing varies by a factor 3)
    "graded": dict(seed=13, n_pts=1024, stretch=None, grade=9.0, white_seed=7041),
}


def rms(x) -> float:
    return float(torch.as_tensor(x).double().pow(2).mean().sqrt())


def build_condition(T: str, name: str, spec: dict, gen: PlanarFields, ref: dict, dev) -> dict:
    ids = TEST100
    if name == "orig_regen":
        pts = np.asarray(ref["points"], dtype=np.float64)[:, :2]
        faces = np.asarray(ref["faces"], dtype=np.int64)
        n_flipped = 0
    else:
        pts, faces = delaunay_cloud(spec["seed"], spec["n_pts"], spec["stretch"], spec.get("grade"))
        faces, n_flipped = make_ccw(pts, faces)
    n = len(pts)
    edges, fei, fes = edge_face_data(faces, n, dev)
    n_edges = len(edges)
    c = 1 if T == "T6" else 3
    white = None
    if T == "T7" and spec["white_seed"] is not None:
        g = torch.Generator().manual_seed(spec["white_seed"])
        white = torch.randn(len(ids), n_edges, 3, generator=g) * 0.1
    E = gen.fields(ids, pts, edges, white=white)
    if T == "T6":
        Fv = face_values_T6(E, fei, fes)
    else:
        Fv, dA, comm = face_values_T7(E, fei, fes)
    Ec, Fc = E.cpu(), Fv.cpu()
    X = encode_edges_to_nodes(Ec.reshape(len(ids), n_edges, c), edges, pts, n)
    Y = faces_to_nodes(Fc.reshape(len(ids), len(faces), c), faces, n)
    area = signed_areas(pts, faces)
    out = {"pts": torch.as_tensor(pts), "faces": torch.as_tensor(faces), "edges": torch.as_tensor(edges),
           ("theta" if T == "T6" else "A"): Ec, ("plaq" if T == "T6" else "F"): Fc,
           "X": X.float(), "Y": Y.float(), "sample_ids": torch.as_tensor(ids),
           "meta": {"condition": name, **spec, "n_flipped_to_ccw": n_flipped, "task": T}}
    if T == "T7":
        out["dA"] = dA.cpu()                    # abelian part of F (face orientation), for the consistency check d1 A = dA
    st = {"mesh": mesh_stats(pts, faces), "n_flipped_to_ccw": n_flipped, "spec": spec,
          "all_faces_ccw": bool((area > 0).all()),
          "edge_cochain_rms": rms(Ec), "face_cochain_rms": rms(Fc), "node_target_rms": rms(Y),
          "node_target_mean": float(Y.double().mean()), "node_input_rms": rms(X)}
    Lv = np.linalg.norm(pts[edges[:, 1]] - pts[edges[:, 0]], axis=1)
    mid_x = 0.5 * (pts[edges[:, 0], 0] + pts[edges[:, 1], 0])
    st["edge_len_median_x_below_0.1"] = float(np.median(Lv[mid_x < 0.1]))      # grading of the mesh along x
    st["edge_len_median_x_above_0.9"] = float(np.median(Lv[mid_x > 0.9]))
    a_t = torch.as_tensor(np.abs(area)).double()
    if T == "T6":
        st["face_flux_density_rms"] = rms(Fc.double() / a_t[None, :])
        # independent check of the data on every mesh: the plaquette flux of the discrete connection against the flux of the
        # continuum field, curl of the softened vortices (-r_y, r_x) s / (r^2 + 0.01) = 0.02 s / (r^2 + 0.01)^2 (the gauge bumps
        # are curl free), evaluated at the face centroids times the face areas; relative L2 error over all faces and samples
        cent = pts[faces].mean(1)
        ids_t = torch.as_tensor(ids, device=dev)
        B = np.zeros((len(ids), len(faces)))
        for cx, cy, strength in gen.vort:
            cx_, cy_, s_ = (t[ids_t].double().cpu().numpy()[:, 0] for t in (cx, cy, strength))
            r2c = (cent[None, :, 0] - cx_[:, None]) ** 2 + (cent[None, :, 1] - cy_[:, None]) ** 2
            B += 0.02 * s_[:, None] / (r2c + 0.01) ** 2
        flux_c = B * np.abs(area)[None, :]
        diff = Fc.double().numpy() - flux_c
        st["face_flux_vs_continuum_rel_L2"] = float(np.linalg.norm(diff) / np.linalg.norm(flux_c))
        # the long boundary slivers of the random Delaunay meshes (edges up to ~0.9) violate the midpoint rule of the generator and
        # dominate the error over all faces (also for the original data); the error on the well-shaped faces is the discretisation error
        P3 = np.concatenate([pts, np.zeros((len(pts), 1))], 1)
        a_, b_, c_ = P3[faces[:, 0]], P3[faces[:, 1]], P3[faces[:, 2]]
        la_, lb_, lc_ = np.linalg.norm(b_ - c_, axis=1), np.linalg.norm(c_ - a_, axis=1), np.linalg.norm(a_ - b_, axis=1)
        R_ = la_ * lb_ * lc_ / (4 * np.abs(area))                       # circumradius
        r_ = np.abs(area) / (0.5 * (la_ + lb_ + lc_))                   # inradius
        aspect = R_ / (2 * r_)                                          # R / (2 r), 1 for an equilateral triangle
        good = aspect <= 5.0
        st["face_flux_vs_continuum_rel_L2_aspect_le_5"] = float(np.linalg.norm(diff[:, good]) / np.linalg.norm(flux_c[:, good]))
        st["frac_faces_aspect_le_5"] = float(good.mean())
    else:
        st["face_flux_density_rms"] = rms(Fc.double() / a_t[None, :, None])
        st["dA_rms"] = rms(dA.cpu())
        st["commutator_rms"] = rms(comm.cpu())
        st["dA_over_comm_mean_norm"] = float(dA.norm(dim=-1).mean() / comm.norm(dim=-1).mean())
        st["white_noise_rms"] = rms(white if white is not None else gen.white0[torch.as_tensor(ids, device=dev)].cpu())
    return out, st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["T6", "T7"])
    ap.add_argument("--out-dir", default=os.path.join(DROOT, "v2", "transfer"))
    ap.add_argument("--mem-fraction", type=float, default=0.035)
    a = ap.parse_args()
    T = a.task
    dev = torch.device("cuda")
    torch.cuda.set_per_process_memory_fraction(a.mem_fraction)
    os.makedirs(a.out_dir, exist_ok=True)
    ref = pickle.load(open(os.path.join(DROOT, PKL[T]), "rb"))
    n_edges_orig = len(np.asarray(ref["edge_index"]))
    gen = PlanarFields(T, dev, n_edges_orig=n_edges_orig)
    stats = {"task": T, "device": torch.cuda.get_device_name(0), "torch": torch.__version__, "conditions": {}}
    # stored data of the original test100 (reference statistics)
    nat = torch.load(os.path.join(DROOT, "v2", f"{T}_native.pt"), map_location="cpu", weights_only=False)
    key_e, key_f = ("theta_edges", "plaq") if T == "T6" else ("A_edges", "F_faces")
    Ys = torch.as_tensor(np.asarray(ref["Y_data"]))
    pts0 = np.asarray(ref["points"], dtype=np.float64)[:, :2]
    area0 = np.abs(signed_areas(pts0, np.asarray(ref["faces"], dtype=np.int64)))
    Fs = nat[key_f][TEST100]
    stats["stored_original_test100"] = {
        "edge_cochain_rms": rms(nat[key_e][TEST100]), "face_cochain_rms": rms(Fs),
        "node_target_rms": rms(Ys[TEST100]), "node_target_mean": float(Ys[TEST100].double().mean()),
        "node_input_rms": rms(torch.as_tensor(np.asarray(ref["X_data"]))[TEST100]),
        "face_flux_density_rms": rms(Fs.double() / torch.as_tensor(area0).double().reshape((1, -1) + (1,) * (Fs.dim() - 2))),
        "node_target_rms_all_10000": rms(Ys), "node_target_rms_train_70pct": rms(Ys[:7000]),
        "edge_cochain_rms_all_10000": rms(nat[key_e])}
    for name, spec in CONDITIONS.items():
        t0 = time.time()
        out, st = build_condition(T, name, spec, gen, ref, dev)
        st["seconds"] = time.time() - t0
        stats["conditions"][name] = st
        torch.save(out, os.path.join(a.out_dir, f"{T}_{name}.pt"))
        print(f"[{T}/{name}] n0={st['mesh']['n_nodes']} n1={st['mesh']['n_edges']} n2={st['mesh']['n_faces']} "
              f"min angle {st['mesh']['min_angle_deg']:.3f} deg  target rms {st['node_target_rms']:.5f} "
              f"(flux density rms {st['face_flux_density_rms']:.4f}) [{st['seconds']:.1f}s]", flush=True)
    json.dump(jsonable(stats), open(os.path.join(a.out_dir, f"stats_{T}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
