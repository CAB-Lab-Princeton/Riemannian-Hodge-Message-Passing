"""AirfRANS-based data sets of the RHMP v2 AirfRANS task family (``rhmp/tasks/airfrans.py``).

Source data: the 1000 AirfRANS simulations (Bonnet et al., NeurIPS 2022 Datasets and Benchmarks; incompressible RANS,
k-omega SST, OpenFOAM), read with the ``airfrans`` package (``airfrans.Simulation(root, name)``).  Simulation order:
``sorted(os.listdir(root))[:1000]`` (the order of the paper's T8 generator); the paper's T8 partition is kept for every
task: simulations 0..699 train (the loaders hold out 630..699 for validation), 700..999 test, and the paper tables use
the test simulations 850..949 ("test100").  Lengths are in chords (chord = 1 m), velocities in m/s, the angle of
attack in radians, pressures in Pa (kinematic pressure times rho as stored by AirfRANS).

Sub-commands
------------
``t8res``  T8 resolution-transfer test sets.  The T8 generator draws, per simulation and in the order above, 2000 of
           the ~177K CFD nodes with ``numpy.random.RandomState(42).choice(n, 2000, replace=False)``, which is the
           prefix of ``RandomState.permutation(n)`` (numpy's legacy implementation); the stream is advanced
           sequentially over the 1000 simulations.  This command replays that stream, verifies that the prefix of
           length 2000 reproduces the stored T8 data exactly (positions, triangles, inputs, targets), and writes for
           the 300 test simulations the same construction with the prefixes of length 500 and 8000 (nested point sets:
           500 of the 2000 T8 points, and 6000 more): uniform random CFD nodes, scipy Delaunay triangulation of their
           convex hull (triangles also cross the airfoil), node inputs (sdf, U, angle of attack), target pressure.
           Output: ``datasets/v2/T8res_n{500,2000,8000}.pt`` (+ ``.json``).

``mesh``   Conforming near-field meshes with CFD fields (tasks AFV*, AFU*, AFP*, AFD), see :func:`build_mesh`.
           Output: ``datasets/v2/AFM.pt`` (1000 simulations), ``AFM_fine.pt`` and ``AFM_coarse.pt`` (the 300 test
           simulations with the size field halved / doubled: about 3.6x / 0.3x the nodes; same construction,
           independent random points).

``diffusion`` The derived diffusion problem on these meshes (``AFD*.pt``).

Usage (on the GPU host; CPU only, ~2-6 min with 32 workers)::

    export AIRFRANS_ROOT=/path/to/airfrans/Dataset
    python3 -u datasets/generators/gen_airfrans.py t8res --workers 32
    python3 -u datasets/generators/gen_airfrans.py mesh --workers 32
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import warnings
from multiprocessing import Pool

import numpy as np
from scipy.spatial import Delaunay

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
N_SIM = 1000
N_TRAIN_PAPER = 700               # T8: first 700 simulations train (normalisation statistics), last 300 test
T8_SEED = 42
T8_POINTS = 2000
TEST100 = (850, 950)              # source indices of the paper's 100 test meshes


def airfrans_root(arg: str | None) -> str:
    root = arg or os.environ.get("AIRFRANS_ROOT")
    if not root or not os.path.isdir(root):
        raise SystemExit("set --airfrans-root or AIRFRANS_ROOT to the AirfRANS 'Dataset' directory")
    return root


def sim_names(root: str) -> list[str]:
    """The simulation order of the paper's T8 generator (``sorted(os.listdir(root))[:1000]``)."""
    names = sorted(os.listdir(root))[:N_SIM]
    bad = [n for n in names if not os.path.isdir(os.path.join(root, n))]
    if bad or len(names) != N_SIM:
        raise RuntimeError(f"unexpected AirfRANS directory content: {bad[:3]} ({len(names)} entries)")
    return names


def load_sim(root: str, name: str):
    import airfrans as af
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return af.Simulation(root=root, name=name)


def pack(samples: list[dict], meta: dict, keys: list[str]) -> dict:
    """Flat concatenation with offsets ``ptr0`` / ``ptr1`` / ``ptr2`` (as ``gen_HP.py``)."""
    import torch
    out = {"meta": meta}
    for deg, nk in (("0", "n0"), ("1", "n1"), ("2", "n2")):
        if nk in samples[0]["stats"]:
            n = np.array([s["stats"][nk] for s in samples])
            out["ptr" + deg] = torch.from_numpy(np.concatenate([[0], np.cumsum(n)]).astype(np.int64))
    for k in keys:
        out[k] = torch.from_numpy(np.ascontiguousarray(np.concatenate([s[k] for s in samples], 0)))
    out["sample_stats"] = [s["stats"] for s in samples]
    return out


# ======================================================================================================================
# t8res: the T8 sampling at other resolutions
# ======================================================================================================================
def _count_points(args):
    root, name = args
    return int(np.asarray(load_sim(root, name).position).shape[0])


def t8_permutations(root: str, names: list[str], n_keep: int, workers: int) -> tuple[list[np.ndarray], list[int]]:
    """Prefixes (length ``n_keep``) of the per-simulation permutations of the T8 stream ``RandomState(42)``."""
    with Pool(workers) as pool:
        npts = pool.map(_count_points, [(root, n) for n in names], chunksize=4)
    rng = np.random.RandomState(T8_SEED)
    perms = []
    for n in npts:                          # RandomState.choice(n, k, replace=False) == permutation(n)[:k]
        perms.append(rng.permutation(n)[:n_keep].astype(np.int64))
    return perms, npts


def _t8_sample(args):
    """One T8-style sample (float32 positions, Delaunay of the convex hull, x = (sdf, U, a), y = p)."""
    root, name, idx, src = args
    s = load_sim(root, name)
    pos = np.array(s.position, dtype=np.float32)
    sdf = np.array(s.sdf, dtype=np.float32)
    pres = np.array(s.pressure, dtype=np.float32)
    n = len(idx)
    tri = Delaunay(pos[idx])
    faces = tri.simplices.astype(np.int64)
    x = np.column_stack([sdf[idx], np.full((n, 1), float(s.inlet_velocity), dtype=np.float32),
                         np.full((n, 1), float(s.angle_of_attack), dtype=np.float32)]).astype(np.float32)
    return dict(pos=pos[idx], faces=faces.astype(np.int32), x=x, y=pres[idx].reshape(n, 1).astype(np.float32),
                stats=dict(src=int(src), name=name, n0=n, n2=len(faces), n_cfd=int(len(pos))))


def _faces_equal(a: np.ndarray, b: np.ndarray) -> bool:
    if a.shape == b.shape and np.array_equal(a, b):
        return True
    if a.shape != b.shape:
        return False
    ka = np.sort(np.sort(a, 1).view([("", a.dtype)] * 3).ravel())
    kb = np.sort(np.sort(b, 1).view([("", b.dtype)] * 3).ravel())
    return bool(np.array_equal(ka, kb))


def cmd_t8res(a) -> None:
    import torch
    root = airfrans_root(a.airfrans_root)
    names = sim_names(root)
    t0 = time.time()
    n_keep = max(a.points)
    perms, npts = t8_permutations(root, names, max(n_keep, T8_POINTS), a.workers)
    print(f"permutations of {len(names)} simulations ({time.time() - t0:.1f}s); CFD nodes {min(npts)}..{max(npts)}",
          flush=True)
    out_dir = os.path.join(ROOT, "datasets", "v2")
    os.makedirs(out_dir, exist_ok=True)
    # 1) reproduce the stored T8 data (all 1000 simulations at 2000 points) and the T8 normalisation statistics
    with Pool(a.workers) as pool:
        t8 = pool.map(_t8_sample, [(root, names[i], perms[i][:T8_POINTS], i) for i in range(N_SIM)], chunksize=4)
    xs = np.concatenate([s["x"] for s in t8[:N_TRAIN_PAPER]]).astype(np.float64)
    ys = np.concatenate([s["y"] for s in t8[:N_TRAIN_PAPER]]).astype(np.float64)
    t8_stats = dict(x_mean=xs.mean(0).tolist(), x_std=xs.std(0).tolist(), y_mean=ys.mean(0).tolist(),
                    y_std=ys.std(0).tolist(), note="numpy (biased) statistics of simulations 0..699 at 2000 points "
                                                   "(the T8 loader's normalisation)")
    verify = {"checked": False}
    pkl = a.t8_pickle or os.path.join(ROOT, "datasets", "T8_airfoil_pressure_persample.pkl")
    if os.path.exists(pkl):
        with open(pkl, "rb") as f:
            ref = pickle.load(f)
        bad = {"pos": 0, "faces": 0, "x": 0, "y": 0}
        for i, (r, s) in enumerate(zip(ref, t8)):
            bad["pos"] += int(not np.array_equal(np.asarray(r["pts_3d"], dtype=np.float32)[:, :2], s["pos"]))
            bad["faces"] += int(not _faces_equal(np.asarray(r["faces"], dtype=np.int64), s["faces"].astype(np.int64)))
            bad["x"] += int(not np.array_equal(np.asarray(r["x_pres"], dtype=np.float32), s["x"]))
            bad["y"] += int(not np.array_equal(np.asarray(r["y_pres"], dtype=np.float32).reshape(-1, 1), s["y"]))
        verify = dict(checked=True, pickle=pkl, n=len(ref), mismatching_samples=bad,
                      exact=all(v == 0 for v in bad.values()))
        print("T8 reproduction:", json.dumps(verify), flush=True)
        if not verify["exact"]:
            raise SystemExit("the replayed T8 stream does not reproduce the stored T8 data")
    else:
        print(f"WARNING: {pkl} not found; the T8 reproduction check is skipped", flush=True)
    test_ids = list(range(N_TRAIN_PAPER, N_SIM))
    for n_pts in a.points:
        t1 = time.time()
        if n_pts == T8_POINTS:
            samples = [t8[i] for i in test_ids]
        else:
            with Pool(a.workers) as pool:
                samples = pool.map(_t8_sample, [(root, names[i], perms[i][:n_pts], i) for i in test_ids], chunksize=4)
        meta = dict(task="T8res", n_points=n_pts, source_ids=test_ids, names=[names[i] for i in test_ids],
                    test100_local=list(range(TEST100[0] - N_TRAIN_PAPER, TEST100[1] - N_TRAIN_PAPER)),
                    t8_stats=t8_stats, t8_verify=verify, generator="datasets/generators/gen_airfrans.py t8res",
                    construction="prefix of length n_points of the T8 permutation stream RandomState(42) (nested "
                                 "point sets across resolutions); scipy Delaunay of the convex hull; x = (sdf, U, a), "
                                 "y = p (Pa)", gen_seconds=time.time() - t1)
        path = os.path.join(out_dir, f"T8res_n{n_pts}.pt")
        torch.save(pack(samples, meta, ["pos", "faces", "x", "y"]), path)
        summ = dict({k: v for k, v in meta.items() if k not in ("source_ids", "names", "test100_local")}, file=path,
                    file_MB=os.path.getsize(path) / 1e6, n2_mean=float(np.mean([s["stats"]["n2"] for s in samples])))
        json.dump(summ, open(path.replace(".pt", ".json"), "w"), indent=2)
        print(f"wrote {path} ({summ['file_MB']:.1f} MB, {time.time() - t1:.1f}s)", flush=True)


# ======================================================================================================================
# mesh: conforming near-field meshes with CFD fields
# ======================================================================================================================
# Size field (lengths in chords): h(x) = min(h_wall + growth * max(d - d_min, 0), h_wall/2 + growth * |x - x_LE|,
# h_wall/2 + growth * |x - x_TE|), d = wall distance; the fine set halves h everywhere (4x the nodes).
MESH = dict(R=0.5, h_wall=0.0065, growth=0.25, dmin_fac=0.6, thin=0.45, density=1.35, quad=16, quad_wall=24)
KIND_VOLUME, KIND_WALL, KIND_OUTER = 0, 1, 2


def airfoil_loop(s, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The airfoil surface as a counter-clockwise closed polygon ``(m, 2)`` and the ids of its CFD surface nodes."""
    from scipy.spatial import cKDTree
    L = np.asarray(s.airfoil.lines).reshape(-1, 3)[:, 1:]
    adj: dict[int, list[int]] = {}
    for u, v in L:
        adj.setdefault(int(u), []).append(int(v))
        adj.setdefault(int(v), []).append(int(u))
    order, prev, cur = [int(L[0, 0])], None, int(L[0, 0])
    while True:
        nxt = [w for w in adj[cur] if w != prev][0]
        if nxt == order[0]:
            break
        order.append(nxt)
        prev, cur = cur, nxt
        if len(order) > len(L) + 1:
            raise RuntimeError("airfoil boundary is not a single closed loop")
    P = np.asarray(s.airfoil_position, dtype=np.float64)[order]
    if signed_area_poly(P) < 0:
        P = P[::-1].copy()
    surf = np.where(np.asarray(s.surface))[0]
    d, j = cKDTree(pos[surf]).query(P)
    if d.max() > 1e-9:
        raise RuntimeError("airfoil polygon and CFD surface nodes differ")
    return P, surf[j]


def signed_area_poly(P: np.ndarray) -> float:
    return float(0.5 * np.sum(P[:, 0] * np.roll(P[:, 1], -1) - np.roll(P[:, 0], -1) * P[:, 1]))


def tri_signed_area(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    Q = X[F]
    return 0.5 * ((Q[:, 1, 0] - Q[:, 0, 0]) * (Q[:, 2, 1] - Q[:, 0, 1]) -
                  (Q[:, 1, 1] - Q[:, 0, 1]) * (Q[:, 2, 0] - Q[:, 0, 0]))


def cfd_fields(s) -> dict:
    """CFD point data of one simulation: positions, sdf, velocity, pressure, nu_t, point vorticity, node areas."""
    g = s.internal
    pos = np.asarray(s.position, dtype=np.float64)
    u = np.asarray(s.velocity, dtype=np.float64)
    gg = g.copy()
    gg.point_data["U3"] = np.column_stack([u, np.zeros(len(u))])
    d = gg.compute_derivative(scalars="U3", gradient=False, vorticity=True)
    omega = np.asarray(d.point_data["vorticity"])[:, 2].astype(np.float64)       # dv/dx - du/dy
    cells = g.cells_dict[9]                                                         # VTK_QUAD
    area = np.zeros(len(pos))
    np.add.at(area, cells.ravel(), np.repeat(np.asarray(g.cell_data["Area"], dtype=np.float64) / 4.0, 4))
    return dict(grid=g, pos=pos, sdf=np.asarray(s.sdf, dtype=np.float64)[:, 0], u=u,
                p=np.asarray(s.pressure, dtype=np.float64)[:, 0], nut=np.asarray(s.nu_t, dtype=np.float64)[:, 0],
                omega=omega, area=area, surface=np.asarray(s.surface, dtype=bool), NU=float(s.NU),
                U=float(s.inlet_velocity), aoa=float(s.angle_of_attack))


class Probe:
    """Bilinear interpolation of the CFD point data (``vtkProbeFilter`` on the quad grid, one cached cell locator).
    Points inside the airfoil (outside the fluid) get ``u = 0`` and ``nu_t = 0`` (solid body at rest)."""

    def __init__(self, grid, poly: np.ndarray):
        import vtk
        from matplotlib.path import Path
        self.grid = grid
        self.loc = vtk.vtkStaticCellLocator()
        self.loc.SetDataSet(grid)
        self.loc.BuildLocator()
        self.path = Path(poly)
        self.z = float(np.asarray(grid.points)[0, 2])

    def __call__(self, X: np.ndarray, names=("U", "nut")) -> dict:
        import pyvista as pv
        import vtk
        pf = vtk.vtkProbeFilter()
        pf.SetSourceData(self.grid)
        pf.SetInputData(pv.PolyData(np.column_stack([X, np.full(len(X), self.z)])))
        pf.SetCellLocatorPrototype(self.loc)
        pf.Update()
        out = pv.wrap(pf.GetOutput())
        valid = np.asarray(out["vtkValidPointMask"]).astype(bool)
        res = {}
        for n in names:
            v = np.asarray(out[n], dtype=np.float64)
            v = v[:, :2] if n == "U" else v
            res[n] = v.copy()
        bad = ~valid
        if bad.any():
            solid = bad & self.path.contains_points(X)
            for n in names:
                res[n][solid] = 0.0
            other = bad & ~solid
            if other.any():
                raise RuntimeError(f"{int(other.sum())} probe points outside the CFD domain")
        return res


def size_field(X: np.ndarray, d: np.ndarray, x_le: np.ndarray, x_te: np.ndarray, h_w: float, g: float,
               d_min: float) -> np.ndarray:
    h1 = h_w + g * np.maximum(d - d_min, 0.0)
    h2 = 0.5 * h_w + g * np.linalg.norm(X - x_le, axis=1)
    h3 = 0.5 * h_w + g * np.linalg.norm(X - x_te, axis=1)
    return np.minimum(h1, np.minimum(h2, h3))


def _poly_point(P: np.ndarray, sarc: np.ndarray, per: float, sv: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Points at arc lengths ``sv`` on the closed polygon ``P`` (``sarc`` cumulative arc length at the vertices):
    positions, segment index, local parameter."""
    sv = np.mod(sv, per)
    k = np.clip(np.searchsorted(sarc, sv, side="right") - 1, 0, len(P) - 1)
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    t = (sv - sarc[k]) / np.maximum(seg[k], 1e-300)
    X = P[k] + t[:, None] * (np.roll(P, -1, 0)[k] - P[k])
    return X, k, t


def _offset_curve(P: np.ndarray, R: float, n_dir: int = 1440) -> np.ndarray:
    """The curve at distance ``R`` from the airfoil polygon, as rays from the mid-chord point (star-shaped)."""
    from scipy.spatial import cKDTree
    seg = np.roll(P, -1, 0) - P
    up = np.concatenate([P[i] + np.outer(np.linspace(0.0, 1.0, 16, endpoint=False), seg[i]) for i in range(len(P))])
    kd = cKDTree(up)
    c = np.array([0.5 * (P[:, 0].min() + P[:, 0].max()), P[:, 1].mean()])
    phi = np.linspace(0.0, 2 * np.pi, n_dir, endpoint=False)
    dirs = np.stack([np.cos(phi), np.sin(phi)], 1)
    lo, hi = np.zeros(n_dir), np.full(n_dir, R + 1.0)
    for _ in range(40):                              # bisection on dist(c + r dir, airfoil) = R
        mid = 0.5 * (lo + hi)
        d, _ = kd.query(c[None] + mid[:, None] * dirs)
        inside = d < R
        lo = np.where(inside, mid, lo)
        hi = np.where(inside, hi, mid)
    return c[None] + 0.5 * (lo + hi)[:, None] * dirs


def _stratified_arc(per: float, h_of_s, rng, s0: float = 0.0) -> np.ndarray:
    """Arc-length positions ``s0, s0 + h(s0) U(0.85, 1.15), ...`` around a closed curve of length ``per``."""
    out, sv = [s0], s0 + h_of_s(np.array([s0]))[0] * rng.uniform(0.85, 1.15)
    while sv < s0 + per - 0.5 * h_of_s(np.array([sv]))[0]:
        out.append(sv)
        sv += h_of_s(np.array([sv]))[0] * rng.uniform(0.85, 1.15)
    return np.mod(np.array(out), per)


def build_mesh(cfd: dict, P: np.ndarray, surf_ids: np.ndarray, rng, scale: float = 1.0, prm: dict | None = None):
    """Conforming near-field mesh of one simulation.

    Domain: the fluid within distance ``R`` (0.5 chords) of the airfoil, an annulus (one hole).  Nodes: (i) airfoil
    surface points at arc-length spacing h (positions on the CFD surface polygon; values interpolated along it; the
    trailing- and leading-edge vertices are always included), (ii) an outer ring at distance R (the CFD nodes nearest
    to an arc-length sampling of the offset curve), (iii) CFD nodes at wall distance d in [d_min, R - 0.6 h(R)),
    drawn independently with probability ``density * node_area * 1.1547 / h(x)^2`` (the density of an equilateral
    mesh of size h) and thinned to a minimum spacing ``thin * h(x)`` (dart throwing in random order).  scipy Delaunay
    triangulation; triangles inside the airfoil or outside the ring are removed; volume nodes that encroach on a
    boundary segment (inside its diametral circle) are removed until every wall and ring segment is a mesh edge.
    Triangles are counter-clockwise.  ``scale = 0.5`` halves h everywhere (the 4x set).

    Returns:
        dict with ``X (n0, 2)``, ``faces (n2, 3)``, ``kind (n0,)`` (0 volume, 1 wall, 2 outer ring), ``cfd_id (n0,)``
        (-1 for interpolated wall points), ``s_wall`` (arc length of wall points) and statistics.
    """
    from matplotlib.path import Path
    from scipy.spatial import Delaunay as _Del, cKDTree
    prm = dict(MESH, **(prm or {}))
    R, h_w, g = prm["R"], prm["h_wall"] * scale, prm["growth"] * scale
    d_min = prm["dmin_fac"] * h_w
    pos, sdf = cfd["pos"], cfd["sdf"]
    seg = np.linalg.norm(np.roll(P, -1, 0) - P, axis=1)
    sarc = np.concatenate([[0.0], np.cumsum(seg)])[:-1]
    per = float(seg.sum())
    ite = int(np.argmax(P[:, 0]))
    ile = int(np.argmax(np.linalg.norm(P - P[ite], axis=1)))
    x_te, x_le = P[ite], P[ile]

    def h_at(X, d):
        return size_field(X, d, x_le, x_te, h_w, g, d_min)

    # (i) wall points.  Near the trailing edge (arc length < te_sym from it) the points of the two surfaces are placed
    # at the same arc-length distances from the corner: in a wedge, a point of one surface is then outside the
    # diametral circle of every segment of the other surface, so the wall segments stay Delaunay edges.
    def h_wall_s(sv):
        X, _, _ = _poly_point(P, sarc, per, sv)
        return h_at(X, np.zeros(len(X)))
    te_sym = prm.get("te_sym", 0.1)
    sig, sv = [], h_wall_s(np.array([sarc[ite]]))[0] * rng.uniform(0.85, 1.15)
    while sv < te_sym:
        sig.append(sv)
        h_up = h_wall_s(np.array([sarc[ite] + sv]))[0]
        h_lo = h_wall_s(np.array([sarc[ite] - sv]))[0]
        sv += min(h_up, h_lo) * rng.uniform(0.85, 1.15)
    sig = np.array(sig)
    rest = _stratified_arc(per - 2 * sv, lambda q: h_wall_s(q + sarc[ite] + sv), rng, s0=0.0)
    s_w = np.concatenate([[sarc[ite]], sarc[ite] + sig, sarc[ite] - sig, sarc[ite] + sv + rest])
    s_w = np.mod(s_w, per)
    j = int(np.argmin(np.abs(((s_w - sarc[ile]) + per / 2) % per - per / 2)))
    s_w[j] = sarc[ile]                                                  # leading-edge vertex
    s_w = np.unique(s_w)
    Xw, _, _ = _poly_point(P, sarc, per, s_w)
    # (ii) outer ring: CFD nodes nearest to an arc-length sampling of the offset curve
    C = _offset_curve(P, R)
    cs = np.linalg.norm(np.roll(C, -1, 0) - C, axis=1)
    carc = np.concatenate([[0.0], np.cumsum(cs)])[:-1]
    cper = float(cs.sum())

    def h_ring_s(sv):
        X, _, _ = _poly_point(C, carc, cper, sv)
        return h_at(X, np.full(len(X), R))
    s_r = _stratified_arc(cper, h_ring_s, rng, s0=rng.uniform(0, cper))
    Xr_target, _, _ = _poly_point(C, carc, cper, s_r)
    near = np.where(np.abs(sdf - R) < 0.05)[0]
    _, jr = cKDTree(pos[near]).query(Xr_target)
    ring_ids = near[jr]
    _, first = np.unique(ring_ids, return_index=True)
    ring_ids = ring_ids[np.sort(first)]                                 # keep the curve order
    h_ring = float(np.median(h_ring_s(s_r)))
    # (iii) volume candidates
    cand = np.where((sdf >= d_min) & (sdf < R - 0.6 * h_ring) & ~cfd["surface"])[0]
    hc = h_at(pos[cand], sdf[cand])
    pc = np.clip(prm["density"] * cfd["area"][cand] * 1.1547 / hc ** 2, 0.0, 1.0)
    keep = rng.random(len(cand)) < pc
    cand, hc = cand[keep], hc[keep]
    order = rng.permutation(len(cand))
    cand, hc = cand[order], hc[order]
    fixed = np.concatenate([Xw, pos[ring_ids]])
    allp = np.concatenate([fixed, pos[cand]])
    tree = cKDTree(allp)
    nbrs = tree.query_ball_point(pos[cand], r=prm["thin"] * hc)
    nf = len(fixed)
    acc = np.zeros(len(allp), dtype=bool)
    acc[:nf] = True
    for i, nb in enumerate(nbrs):
        ii = nf + i
        ok = True
        for q in nb:
            if q != ii and acc[q]:
                ok = False
                break
        acc[ii] = ok
    vol_ids = cand[acc[nf:]]
    # triangulate, carve, recover boundary segments
    nw, nr = len(Xw), len(ring_ids)
    wall_path, ring_path = Path(P), Path(pos[ring_ids])
    for it in range(8):
        X = np.concatenate([Xw, pos[ring_ids], pos[vol_ids]])
        F = _Del(X).simplices.astype(np.int64)
        sa = tri_signed_area(X, F)
        F[sa < 0] = F[sa < 0][:, [0, 2, 1]]
        cent = X[F].mean(1)
        F = F[~wall_path.contains_points(cent) & ring_path.contains_points(cent)]
        n0 = len(X)
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        ek, cnt = np.unique(E[:, 0] * n0 + E[:, 1], return_counts=True)
        wseg = np.sort(np.stack([np.arange(nw), np.roll(np.arange(nw), -1)], 1), 1)
        rseg = np.sort(np.stack([nw + np.arange(nr), nw + np.roll(np.arange(nr), -1)], 1), 1)
        segs = np.concatenate([wseg, rseg])
        miss = ~np.isin(segs[:, 0] * n0 + segs[:, 1], ek)
        if not miss.any():
            break
        A, B = X[segs[miss, 0]], X[segs[miss, 1]]
        mid, rad = 0.5 * (A + B), 0.5 * np.linalg.norm(B - A, axis=1) * 1.05
        Xv = pos[vol_ids]
        bad = np.zeros(len(vol_ids), dtype=bool)
        for m_, r_ in zip(mid, rad):
            bad |= np.linalg.norm(Xv - m_, axis=1) < r_
        if not bad.any():
            raise RuntimeError(f"boundary segment missing without an encroaching volume node: {segs[miss].tolist()} "
                               f"(wall points 0..{nw - 1}, ring points {nw}..{nw + nr - 1})")
        vol_ids = vol_ids[~bad]
    else:
        raise RuntimeError("could not recover the boundary segments")
    # checks: a manifold annulus whose boundary edges are exactly the wall and ring segments
    used = np.unique(F)
    bnd_keys = ek[cnt == 1]
    stats = dict(n0=int(n0), n1=int(len(ek)), n2=int(len(F)), n_wall=int(nw), n_ring=int(nr),
                 n_volume=int(len(vol_ids)), euler=int(n0 - len(ek) + len(F)), boundary_iterations=it,
                 nonmanifold=int((cnt > 2).sum()), unused=int(n0 - len(used)), n_boundary_edges=int(len(bnd_keys)),
                 h_wall=h_w, growth=g, d_min=d_min, R=R)
    if stats["euler"] != 0 or stats["nonmanifold"] or stats["unused"] or len(bnd_keys) != nw + nr or \
            not np.isin(bnd_keys, segs[:, 0] * n0 + segs[:, 1]).all():
        raise RuntimeError(f"invalid near-field mesh: {stats}")
    ang = []
    Q = X[F]
    for k in range(3):
        a_, b_ = Q[:, (k + 1) % 3] - Q[:, k], Q[:, (k + 2) % 3] - Q[:, k]
        ang.append(np.degrees(np.arccos(np.clip((a_ * b_).sum(1) / np.linalg.norm(a_, axis=1) /
                                                np.linalg.norm(b_, axis=1), -1, 1))))
    minang = np.min(np.stack(ang, 1), 1)
    stats.update(min_angle=float(minang.min()), min_angle_p1=float(np.percentile(minang, 1)),
                 frac_min_angle_lt10=float((minang < 10).mean()))
    kind = np.concatenate([np.full(nw, KIND_WALL), np.full(nr, KIND_OUTER), np.full(len(vol_ids), KIND_VOLUME)])
    cfd_id = np.concatenate([np.full(nw, -1), ring_ids, vol_ids]).astype(np.int64)
    return dict(X=X, faces=F, kind=kind, cfd_id=cfd_id, s_wall=s_w, stats=stats, sarc=sarc, per=per)


def _quad_rule(Q: int, graded: bool) -> tuple[np.ndarray, np.ndarray]:
    x, w = np.polynomial.legendre.leggauss(Q)
    t, w = 0.5 * (x + 1.0), 0.5 * w
    if graded:                                     # s = t^3 clusters the nodes at s = 0 (the wall end)
        return t ** 3, w * 3.0 * t ** 2
    return t, w


def edge_integrals(X: np.ndarray, edges: np.ndarray, kind: np.ndarray, probe: Probe, prm: dict) -> dict:
    """Line integrals of the CFD velocity along the oriented edges ``src -> dst``:
    ``theta_e = int_e u . dl`` (circulation 1-form) and ``flux_e = int_e u . n ds`` with ``n`` the right normal of
    ``x_dst - x_src`` (outward flux of the triangle to the edge's right; ``flux = d0 psi`` for a stream function
    ``u = (psi_y, -psi_x)``).  Gauss-Legendre quadrature on the bilinear interpolant of the CFD point data
    (``prm['quad']`` nodes; ``prm['quad_wall']`` nodes graded towards the wall for edges with one wall end); edges
    with two wall ends lie on the no-slip, no-penetration wall: both integrals are 0."""
    a, b = X[edges[:, 0]], X[edges[:, 1]]
    wa, wb = kind[edges[:, 0]] == KIND_WALL, kind[edges[:, 1]] == KIND_WALL
    theta, flux = np.zeros(len(edges)), np.zeros(len(edges))
    groups = [(~wa & ~wb, False, False), (wa & ~wb, True, False), (~wa & wb, True, True)]
    for mask, graded, flip in groups:
        idx = np.where(mask)[0]
        if not len(idx):
            continue
        s_, w_ = _quad_rule(prm["quad_wall"] if graded else prm["quad"], graded)
        A, B = (b[idx], a[idx]) if flip else (a[idx], b[idx])           # quadrature parameter from the wall end
        pts = A[:, None, :] + s_[None, :, None] * (B - A)[:, None, :]
        U = probe(pts.reshape(-1, 2), names=("U",))["U"].reshape(len(idx), len(s_), 2)
        Ubar = (U * w_[None, :, None]).sum(1)                           # mean velocity along the edge
        t = b[idx] - a[idx]                                               # src -> dst
        theta[idx] = (Ubar * t).sum(1)
        flux[idx] = Ubar[:, 0] * t[:, 1] - Ubar[:, 1] * t[:, 0]
    return dict(theta=theta, flux=flux)


def mesh_sample(args):
    """All stored fields of one simulation's near-field mesh (see ``cmd_mesh``)."""
    root, name, src, scale, seed, prm = args
    from gen_common import mesh_edges
    t0 = time.time()
    s = load_sim(root, name)
    cfd = cfd_fields(s)
    P, surf_ids = airfoil_loop(s, cfd["pos"])
    errors = []
    for attempt in range(5):                     # a new random draw if the boundary recovery fails (rare)
        rng = np.random.default_rng([seed, src, int(round(1.0 / scale)), attempt])
        try:
            m = build_mesh(cfd, P, surf_ids, rng, scale, prm)
            break
        except RuntimeError as e:
            errors.append(str(e))
    else:
        raise RuntimeError(f"{name}: no valid mesh after 5 draws: {errors}")
    m["stats"]["attempt"] = attempt
    X, F, kind, cid = m["X"], m["faces"], m["kind"], m["cfd_id"]
    n0 = len(X)
    # node values: CFD point data, wall points by linear interpolation along the surface polygon (u = nu_t = 0)
    vals = {}
    for key in ("p", "omega", "nut", "sdf"):
        v = np.zeros(n0)
        v[cid >= 0] = cfd[key][cid[cid >= 0]]
        vals[key] = v
    u = np.zeros((n0, 2))
    u[cid >= 0] = cfd["u"][cid[cid >= 0]]
    wall = kind == KIND_WALL
    sp = np.concatenate([m["sarc"], [m["per"]]])
    for key in ("p", "omega"):
        f = cfd[key][surf_ids]
        vals[key][wall] = np.interp(m["s_wall"], sp, np.concatenate([f, f[:1]]))
    vals["sdf"][wall] = 0.0
    vals["nut"][wall] = 0.0
    edges = mesh_edges(F, n0)
    probe = Probe(cfd["grid"], P)
    prm_ = dict(MESH, **(prm or {}))
    ei = edge_integrals(X, edges, kind, probe, prm_)
    ev = X[edges[:, 1]] - X[edges[:, 0]]
    um = 0.5 * (u[edges[:, 0]] + u[edges[:, 1]])
    theta_trap = (um * ev).sum(1)
    flux_trap = um[:, 0] * ev[:, 1] - um[:, 1] * ev[:, 0]
    mids = 0.5 * (X[edges[:, 0]] + X[edges[:, 1]])
    cent = X[F].mean(1)
    pr = probe(np.concatenate([mids, cent]), names=("nut",))["nut"]
    nut_e, nut_f = np.maximum(pr[:len(mids)], 0.0), np.maximum(pr[len(mids):], 0.0)
    both_wall = (kind[edges[:, 0]] == KIND_WALL) & (kind[edges[:, 1]] == KIND_WALL)
    nut_e[both_wall] = 0.0
    # consistency diagnostics: closedness of the accurate fluxes (Gauss) and of the trapezoid fluxes, per triangle
    from gen_common import edge_face_incidence
    e2, fe_idx, fe_sign = edge_face_incidence(F, n0)
    assert np.array_equal(e2, edges)
    area = tri_signed_area(X, F)
    div_acc = (ei["flux"][fe_idx] * fe_sign).sum(1)
    div_trap = (flux_trap[fe_idx] * fe_sign).sum(1)
    fscale = np.abs(ei["flux"][fe_idx]).sum(1) + 1e-30
    stats = dict(m["stats"], src=int(src), name=name, scale=scale, n_cfd=int(len(cfd["pos"])),
                 U=cfd["U"], aoa=cfd["aoa"], NU=cfd["NU"], seconds=time.time() - t0,
                 div_acc_rel_median=float(np.median(np.abs(div_acc) / fscale)),
                 div_acc_rel_p99=float(np.percentile(np.abs(div_acc) / fscale, 99)),
                 div_trap_rel_median=float(np.median(np.abs(div_trap) / fscale)),
                 div_trap_rel_p99=float(np.percentile(np.abs(div_trap) / fscale, 99)),
                 area_min=float(area.min()))
    f32 = np.float32
    return dict(pos=X.astype(np.float64), faces=F.astype(np.int32), kind=kind.astype(np.int8),
                cfd_id=cid, u=u.astype(f32), p=vals["p"].astype(f32), omega=vals["omega"].astype(f32),
                nut=vals["nut"].astype(f32), sdf=vals["sdf"].astype(f32),
                edges=edges.astype(np.int32), theta=ei["theta"].astype(np.float64),
                flux=ei["flux"].astype(np.float64), theta_trap=theta_trap.astype(np.float64),
                flux_trap=flux_trap.astype(np.float64), nut_edge=nut_e.astype(f32), nut_face=nut_f.astype(f32),
                stats=dict(stats, n0=int(n0), n1=int(len(edges)), n2=int(len(F))))


MESH_KEYS = ["pos", "faces", "kind", "cfd_id", "u", "p", "omega", "nut", "sdf", "edges", "theta", "flux",
             "theta_trap", "flux_trap", "nut_edge", "nut_face"]


def cmd_mesh(a) -> None:
    import torch
    root = airfrans_root(a.airfrans_root)
    names = sim_names(root)
    out_dir = os.path.join(ROOT, "datasets", "v2")
    os.makedirs(out_dir, exist_ok=True)
    prm = dict(MESH)
    test_ids = list(range(N_TRAIN_PAPER, N_SIM)) if a.n is None else list(range(N_TRAIN_PAPER, N_TRAIN_PAPER + a.n))
    sets = [("AFM", list(range(N_SIM)) if a.n is None else list(range(a.n)), 1.0),
            ("AFM_fine", test_ids, 0.5),
            ("AFM_coarse", test_ids, 2.0)]
    for base, ids, scale in sets:
        if a.only and base not in a.only:
            continue
        t0 = time.time()
        with Pool(a.workers) as pool:
            samples = pool.map(mesh_sample, [(root, names[i], i, scale, a.seed, prm) for i in ids], chunksize=2)
        st = [s_["stats"] for s_ in samples]
        meta = dict(task="AFM", source_ids=ids, names=[names[i] for i in ids], scale=scale, seed=a.seed, params=prm,
                    generator="datasets/generators/gen_airfrans.py mesh", gen_seconds=time.time() - t0,
                    kinds={"volume": KIND_VOLUME, "wall": KIND_WALL, "outer": KIND_OUTER},
                    edge_order="unique src<dst pairs of the triangles, lexicographic",
                    units="lengths in chords, u in m/s, p in Pa, nu_t in m^2/s, omega in 1/s, angle of attack in rad",
                    theta="int_e u.dl along src->dst (Gauss-Legendre on the bilinear CFD interpolant)",
                    flux="int_e u.n ds, n = right normal of src->dst (= d0 psi for u = (psi_y, -psi_x))",
                    U=[s_["U"] for s_ in st], aoa=[s_["aoa"] for s_ in st], NU=st[0]["NU"])
        path = os.path.join(out_dir, base + ".pt")
        torch.save(pack(samples, meta, MESH_KEYS), path)
        summ = dict(file=path, file_MB=os.path.getsize(path) / 1e6, N=len(ids), scale=scale,
                    gen_seconds=meta["gen_seconds"])
        for k in ("n0", "n1", "n2", "n_wall", "n_ring", "n_volume", "min_angle", "min_angle_p1",
                  "frac_min_angle_lt10", "div_acc_rel_median", "div_acc_rel_p99", "div_trap_rel_median",
                  "div_trap_rel_p99", "boundary_iterations", "seconds"):
            v = np.array([s_[k] for s_ in st], dtype=np.float64)
            summ[k] = dict(min=float(v.min()), mean=float(v.mean()), max=float(v.max()))
        json.dump(dict(summ, params=prm), open(path.replace(".pt", ".json"), "w"), indent=2)
        print(json.dumps(summ, indent=1), flush=True)


# ======================================================================================================================
# diffusion: a derived scalar problem with the RANS eddy-viscosity ratio as the coefficient
# ======================================================================================================================
def _diffusion_source(base_pos: np.ndarray, base_faces: np.ndarray, rng) -> dict:
    """2-5 Gaussian sources with centres uniform over the near-field domain (a random point of a random base-mesh
    triangle, chosen with probability proportional to its area), widths U[0.03, 0.15] chords, amplitudes
    +-U[1, 3]."""
    A = np.abs(tri_signed_area(base_pos, base_faces))
    n_src = int(rng.integers(2, 6))
    tri = rng.choice(len(A), n_src, p=A / A.sum())
    r1, r2 = rng.random(n_src), rng.random(n_src)
    s1 = np.sqrt(r1)
    Q = base_pos[base_faces[tri]]
    c = (1 - s1)[:, None] * Q[:, 0] + (s1 * (1 - r2))[:, None] * Q[:, 1] + (s1 * r2)[:, None] * Q[:, 2]
    return dict(c=c, w=rng.uniform(0.03, 0.15, n_src), a=rng.uniform(1.0, 3.0, n_src) * rng.choice([-1.0, 1.0], n_src))


def _source_eval(S: dict, X: np.ndarray) -> np.ndarray:
    d2 = ((X[:, None, :] - S["c"][None]) ** 2).sum(-1)
    return (S["a"][None] * np.exp(-d2 / (2 * S["w"][None] ** 2))).sum(-1)


def _diffusion_solve(args):
    """P1 FEM of -div(kappa grad T) = f with kappa_T = 1 + nu_t(centroid) / nu per triangle, lumped-mass right-hand
    side, T = 0 on the airfoil and the outer ring (sparse direct solve)."""
    import scipy.sparse.linalg as spla
    from gen_common import p1_stiffness_2d
    pos, faces, kind, nut_face, nu, S = args
    kappa = 1.0 + np.maximum(nut_face, 0.0) / nu
    K, mass, _ = p1_stiffness_2d(pos, faces, kappa)
    f = _source_eval(S, pos)
    b = mass * f
    free = np.where(kind == KIND_VOLUME)[0]
    KII = K[free][:, free].tocsc()
    T = np.zeros(len(pos))
    T[free] = spla.spsolve(KII, b[free])
    res = float(np.linalg.norm(KII @ T[free] - b[free]) / max(np.linalg.norm(b[free]), 1e-30))
    return dict(f=f.astype(np.float32), T=T.astype(np.float32),
                stats=dict(n0=int(len(pos)), residual=res, kappa_max=float(kappa.max()), kappa_min=float(kappa.min()),
                           n_src=int(len(S["a"]))))


def cmd_diffusion(a) -> None:
    import torch
    out_dir = os.path.join(ROOT, "datasets", "v2")
    base = torch.load(os.path.join(out_dir, "AFM.pt"), map_location="cpu", weights_only=False)
    nu = float(base["meta"]["NU"])

    def sl(blob, key, i, deg):
        p = blob[f"ptr{deg}"]
        return blob[key][p[i]:p[i + 1]].numpy()
    src_of = {int(s_): i for i, s_ in enumerate(base["meta"]["source_ids"])}
    sources = {}
    for s_, i in src_of.items():                 # one source field per simulation, shared by the base and fine meshes
        rng = np.random.default_rng([a.seed, 7, s_])
        sources[s_] = _diffusion_source(sl(base, "pos", i, 0), sl(base, "faces", i, 2).astype(np.int64), rng)
    for name in ("AFM", "AFM_fine", "AFM_coarse"):
        if name != "AFM" and not os.path.exists(os.path.join(out_dir, name + ".pt")):
            continue
        if os.path.exists(os.path.join(out_dir, name.replace("AFM", "AFD") + ".pt")) and not a.force:
            print(f"{name.replace('AFM', 'AFD')}.pt exists (use --force to regenerate)", flush=True)
            continue
        blob = base if name == "AFM" else torch.load(os.path.join(out_dir, name + ".pt"), map_location="cpu",
                                                     weights_only=False)
        ids = [int(s_) for s_ in blob["meta"]["source_ids"]]
        jobs = [(sl(blob, "pos", i, 0), sl(blob, "faces", i, 2).astype(np.int64), sl(blob, "kind", i, 0),
                 sl(blob, "nut_face", i, 2).astype(np.float64), nu, sources[s_]) for i, s_ in enumerate(ids)]
        t0 = time.time()
        with Pool(a.workers) as pool:
            res = pool.map(_diffusion_solve, jobs, chunksize=4)
        out_name = name.replace("AFM", "AFD")
        meta = dict(task="AFD", mesh=name, source_ids=ids, seed=a.seed, nu=nu, gen_seconds=time.time() - t0,
                    generator="datasets/generators/gen_airfrans.py diffusion",
                    pde="-div(kappa grad T) = f, T = 0 on the airfoil and the outer ring; kappa = 1 + nu_t / nu "
                        "per triangle (nu_t at the centroid, bilinear CFD interpolant); P1 FEM, lumped mass",
                    sources=[{k: v.tolist() for k, v in sources[s_].items()} for s_ in ids])
        path = os.path.join(out_dir, out_name + ".pt")
        torch.save(pack(res, meta, ["f", "T"]), path)
        st = [r["stats"] for r in res]
        summ = dict(file=path, N=len(ids), gen_seconds=meta["gen_seconds"],
                    max_residual=float(max(s_["residual"] for s_ in st)),
                    kappa_max=float(max(s_["kappa_max"] for s_ in st)),
                    kappa_max_median=float(np.median([s_["kappa_max"] for s_ in st])),
                    T_std=float(np.std(np.concatenate([r["T"] for r in res]))))
        json.dump(summ, open(path.replace(".pt", ".json"), "w"), indent=2)
        print(json.dumps(summ), flush=True)


# ======================================================================================================================
# CLI
# ======================================================================================================================
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("t8res", help="T8 resolution-transfer test sets")
    p.add_argument("--points", type=int, nargs="+", default=[500, 2000, 8000])
    p.add_argument("--t8-pickle", default=None, help="stored T8 data for the reproduction check (default: datasets/)")
    q = sub.add_parser("mesh", help="conforming near-field meshes (AFM.pt, AFM_fine.pt)")
    q.add_argument("--seed", type=int, default=2026)
    q.add_argument("--n", type=int, default=None, help="only the first n simulations of each set (tests)")
    q.add_argument("--only", nargs="*", default=None, help="AFM, AFM_fine and/or AFM_coarse")
    d = sub.add_parser("diffusion", help="derived diffusion problem on the AFM meshes (AFD.pt, AFD_fine.pt)")
    d.add_argument("--seed", type=int, default=2026)
    d.add_argument("--force", action="store_true", help="regenerate existing AFD*.pt files")
    for r in (p, q, d):
        r.add_argument("--airfrans-root", default=None)
        r.add_argument("--workers", type=int, default=32)
    a = ap.parse_args(argv)
    if a.cmd == "t8res":
        cmd_t8res(a)
    elif a.cmd == "mesh":
        cmd_mesh(a)
    elif a.cmd == "diffusion":
        cmd_diffusion(a)


if __name__ == "__main__":
    main()
