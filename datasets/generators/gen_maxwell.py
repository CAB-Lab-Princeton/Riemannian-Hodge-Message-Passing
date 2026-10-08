"""Nonlinear magnetostatics of a three-limb EI-core electromagnet with saturable iron (task family MXS).

Physics (2-D cross-section; SI units inside the solver; stored lengths in mm, B in T, current densities in A/mm^2)::

    -div( nu(|B|) grad A ) = J        in the box Omega,      A = 0 on the box boundary,
    B = curl(A e_z) = (dA/dy, -dA/dx),     |B| = |grad A|,

A = A_z the magnetic vector potential (Wb/m), J = J_z the impressed current density of the windings.  The edge fluxes
``Phi_e = A(dst) - A(src)`` are the fluxes of B through the edges (right normal of src -> dst): an exactly closed
1-cochain, ``d1 Phi = 0`` (no net flux out of any triangle).

Materials: air and copper (``nu = nu_0 = 1 / mu_0``) and non-oriented electrical steel with the Froehlich-Kennelly law

    B(H) = mu_0 H + J_s H / (H_k + H),       J_s = 2.0 T,  mu_r,init = 3000  (H_k = J_s / (mu_0 (mu_r - 1)) = 530.7 A/m),

monotone and smooth, initial relative permeability 3000, knee at 1.5-1.8 T, saturation polarisation 2.0 T, slope
``mu_0`` in deep saturation (B = 1.65 T at 2.5 kA/m, 1.81 T at 5 kA/m, 2.04 T at 50 kA/m).  Its inverse is explicit:
``nu(B) = H / B = 2 H_k / (b + sqrt(b^2 + 4 mu_0 H_k B))``, ``b = mu_0 H_k + J_s - B``, increasing from
``1 / (3000 mu_0)`` to ``1 / mu_0``.  No model ever receives nu or mu: the models see the current density and the
material phase only.

Device (mm; one fixed mesh, conforming to every fixed interface; the movable parts are per-face material indicators,
i.e. staircase interfaces on the fixed mesh):

* E-core (fixed): back yoke [-128, 128] x [-88, -48]; outer limbs |x| in [96, 128], centre limb |x| <= 32, all up to the
  pole faces at y = 48; windows 64 x 96 mm.
* Three windings (fixed copper blocks, 26 x 78 mm each, y in [-42, 36]): left-limb winding, sides x in [-92, -66]
  (+J_a) and [-158, -132] (-J_a); centre-limb winding, sides [-62, -36] (+J_b) and [36, 62] (-J_b); right-limb
  winding, sides [66, 92] (+J_c) and [132, 158] (-J_c).  Every winding carries zero net current.
* I-armature (movable iron bar): bottom edge at heights 48 + g_L / 48 + g_R above the outer-limb centres
  (x = -/+ 112; g in [6, 24] mm, so the bar may be tilted by up to 4.6 deg), thickness t in [16, 40], width w in
  [176, 288], lateral offset x0 in [-24, 24].
* Pole shoes (movable iron): on each side of every limb tip, with probability 0.6, a shoe y in [40, 48] extending the
  limb by U(4, 16) mm.
* External iron pieces: 0 / 1 / 2 (probabilities 0.4 / 0.35 / 0.25) randomly rotated rectangles (sides 24-64 mm) at
  10-70 mm from the device (a steel object in the stray field).
* Box: [-320, 320] x [-290, 350] (A = 0, about 2x the device size).
* Currents: J_a, J_b, J_c ~ U(-J_max, J_max) independently, J_max = 6 A/mm^2 (NI up to 12.2 kA per winding): from the
  linear regime (small currents, large gaps) to deep saturation of the limbs, pole shoes and armature.

Discretisation: P1 finite elements on the fixed triangulation (scipy Delaunay of a variable-density point set with the
interfaces as recovered constrained segments; 4.5 mm in the gap / armature band, 8 mm in the core, up to 45 mm at the
box), element-wise constant material and current; consistent load ``F_i = sum_f J_f |f| / 3``; damped Newton on the
convex energy ``Pi(A) = sum_f |f| W_f(|grad A|) - F.A`` (``W(B) = int_0^B H dB``, closed form; Armijo backtracking),
started from the linear solve with the initial permeability, stopped at ``|R_free| <= 1e-10 |F_free|``; sparse direct
solves (SuperLU).  Targets: ``B_node`` = area-weighted average over the incident triangles of the element fields
``B_f`` (a fixed linear map of the edge fluxes, ``rhmp.tasks.maxwell.FluxToNodeField``).

Sub-commands::

    python3 -u datasets/generators/gen_maxwell.py data --workers 48          # MXS.pt (5000) + ood15 / fine / coarse
    python3 -u datasets/generators/gen_maxwell.py selftest                    # solver verification (json report)
    python3 -u datasets/generators/gen_maxwell.py mesh --plot                 # mesh statistics (+ png)

Outputs (``datasets/v2/``): ``MXS.pt`` (5000 samples on the base mesh, ~2.0K nodes), ``MXS_ood15.pt`` (test samples
4250..4749 with all currents x1.5), ``MXS_fine.pt`` / ``MXS_coarse.pt`` (the same 500 test samples on a refined (h/2)
and a coarsened (1.8 h) mesh of the same construction, independent random points), each with a ``.json`` summary.
Per file: ``pos (n0, 2)`` mm, ``faces (n2, 3)`` (counter-clockwise), ``region (n2,)`` (fixed region: 0 air, 1..6 coil
blocks, 7 E-core), ``coil_sign (6,)``/``coil_winding (6,)``, and per sample ``currents (N, 3)`` A/mm^2,
``phase (N, n2)`` int8 (0 air, 1 copper, 2 iron), ``A (N, n0)`` float64 Wb/m, ``B_node (N, n0, 2)`` T, Newton
statistics and the layout parameters.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial import Delaunay, cKDTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------------------------------------------------
# constants
# ---------------------------------------------------------------------------------------------------------------------
MU0 = 4e-7 * math.pi
NU0 = 1.0 / MU0
JS = 2.0                                # saturation polarisation (T)
MUR_INIT = 3000.0                       # initial relative permeability
HK = JS / (MU0 * (MUR_INIT - 1.0))      # Froehlich-Kennelly knee field (A/m)
MM = 1e-3                               # stored lengths are mm
AMM2 = 1e6                              # A/mm^2 -> A/m^2

SEED = 20261001
N_SAMPLES = 5000
J_MAX = 6.0                             # A/mm^2
N_EXTRA = 500                           # test samples re-solved on the transfer meshes / at 1.5x current
TEST_START = 4250                       # sequential 70/15/15 split of 5000: test = 4250..4999
OOD_FACTOR = 1.5
SCALES = {"base": 1.0, "fine": 0.5, "coarse": 2.0}
MESH_SEEDS = {"base": 11, "fine": 12, "coarse": 13}

BOX = (-320.0, 320.0, -290.0, 350.0)
CORE = np.array([[-128, -88], [128, -88], [128, 48], [96, 48], [96, -48], [32, -48], [32, 48], [-32, 48],
                 [-32, -48], [-96, -48], [-96, 48], [-128, 48]], dtype=np.float64)
COIL_Y = (-42.0, 36.0)
COIL_X = [(-158.0, -132.0), (-92.0, -66.0), (-62.0, -36.0), (36.0, 62.0), (66.0, 92.0), (132.0, 158.0)]
COIL_WINDING = np.array([0, 0, 1, 1, 2, 2])          # winding of each coil block: a (left), b (centre), c (right)
COIL_SIGN = np.array([-1.0, 1.0, 1.0, -1.0, 1.0, -1.0])
LIMBS = [(-128.0, -96.0), (-32.0, 32.0), (96.0, 128.0)]
POLE_Y = 48.0
SHOE_Y = (41.0, 48.0)
DEVICE_BOX = (-170.0, 170.0, -100.0, 125.0)
# gap / armature band: an equilateral lattice (spacing s = 4 mm x scale, rows parallel to the pole faces, one row on
# the pole faces) from two rows below the pole faces (pole-shoe rows, outside the limbs) to y ~ 80 mm, x in
# [-152, 152]; every limb corner is a lattice point for s in {2, 4, 8}
LATTICE_S = 4.0
LATTICE_X = (-152.0, 152.0)
LATTICE_TOP = 80.0
BAND = (-152.0, 152.0, 41.0, 80.0)
CORE_BOX = (-162.0, 162.0, -92.0, 122.0)
H_BAND, H_CORE, H_FAR = 4.0, 8.0, 45.0
G_BAND, G_CORE = 0.30, 0.25
REGION_AIR, REGION_CORE = 0, 7
PHASE_AIR, PHASE_COPPER, PHASE_IRON = 0, 1, 2

NEWTON_TOL = 1e-10
NEWTON_MAX = 80


# ---------------------------------------------------------------------------------------------------------------------
# material law
# ---------------------------------------------------------------------------------------------------------------------
def nu_iron(B: np.ndarray) -> np.ndarray:
    """Reluctivity ``H(B) / B`` of the Froehlich-Kennelly law (m/H), ``B >= 0`` in T."""
    b = MU0 * HK + JS - B
    s = np.sqrt(b * b + 4.0 * MU0 * HK * B)
    return 2.0 * HK / (b + s)


def dnu_iron(B: np.ndarray) -> np.ndarray:
    """``d nu / d B``."""
    b = MU0 * HK + JS - B
    s = np.sqrt(b * b + 4.0 * MU0 * HK * B)
    dD = -1.0 + (2.0 * MU0 * HK - b) / s
    return -2.0 * HK / (b + s) ** 2 * dD


def h_of_b(B: np.ndarray) -> np.ndarray:
    """``H(B)`` (A/m) of the iron law."""
    return nu_iron(B) * B


def b_of_h(H: np.ndarray) -> np.ndarray:
    """``B(H)`` (T) of the iron law."""
    return MU0 * H + JS * H / (HK + H)


def energy_iron(B: np.ndarray) -> np.ndarray:
    """Stored energy density ``W(B) = int_0^B H dB = B H - W*(H)`` (J/m^3), ``W*(H) = mu_0 H^2/2 + J_s (H - H_k
    log(1 + H/H_k))`` the co-energy."""
    H = h_of_b(B)
    return B * H - (0.5 * MU0 * H * H + JS * (H - HK * np.log1p(H / HK)))


# ---------------------------------------------------------------------------------------------------------------------
# mesh
# ---------------------------------------------------------------------------------------------------------------------
def _rect(x0, x1, y0, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], dtype=np.float64)


def _dist_box(X: np.ndarray, box) -> np.ndarray:
    dx = np.maximum(np.maximum(box[0] - X[:, 0], X[:, 0] - box[1]), 0.0)
    dy = np.maximum(np.maximum(box[2] - X[:, 1], X[:, 1] - box[3]), 0.0)
    return np.hypot(dx, dy)


def size_field(X: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Target element size (mm): fine in the gap / armature band, medium in the core, growing to the box."""
    h1 = H_BAND + G_BAND * _dist_box(X, BAND)
    h2 = H_CORE + G_CORE * _dist_box(X, CORE_BOX)
    return scale * np.minimum(np.minimum(h1, h2), H_FAR)


def fixed_polygons() -> list[np.ndarray]:
    """Closed polygons whose edges must be mesh edges: box, E-core, the six coil blocks."""
    polys = [_rect(BOX[0], BOX[1], BOX[2], BOX[3]), CORE.copy()]
    polys += [_rect(x0, x1, COIL_Y[0], COIL_Y[1]) for x0, x1 in COIL_X]
    return polys


def band_lattice(scale: float) -> np.ndarray:
    """Equilateral lattice of the gap / armature band (rows parallel to the pole faces, row 0 on them, down to the
    pole-shoe bottom); the rows below the pole faces skip the limb interiors."""
    s = LATTICE_S * scale
    dy = s * math.sqrt(3.0) / 2.0
    rows = []
    j = int(math.ceil((SHOE_Y[0] - POLE_Y) / dy - 1e-9))         # lowest row at or above the pole-shoe bottom
    while POLE_Y + j * dy <= LATTICE_TOP + 1e-9:
        y = POLE_Y + j * dy
        xs = np.arange(LATTICE_X[0] + (0.5 * s if j % 2 else 0.0), LATTICE_X[1] + 1e-9, s)
        if j < 0:
            inside = np.zeros(len(xs), dtype=bool)
            for a, b in LIMBS:
                inside |= (xs > a + 1e-9) & (xs < b - 1e-9)
            xs = xs[~inside]
        rows.append(np.stack([xs, np.full(len(xs), y)], 1))
        j += 1
    return np.concatenate(rows)


def _graded(a: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    """Interior points of the segment ``a b`` at spacing ~h(x) (graded; endpoints excluded)."""
    L = float(np.linalg.norm(b - a))
    u = np.linspace(0.0, 1.0, 257)
    h = size_field(a[None] + u[:, None] * (b - a)[None], scale)
    Ncum = np.concatenate([[0.0], np.cumsum(0.5 * (1.0 / h[1:] + 1.0 / h[:-1]) * np.diff(u) * L)])
    n = max(1, int(round(Ncum[-1])))
    if n == 1:
        return np.zeros((0, 2))
    t = np.interp(np.arange(1, n) * Ncum[-1] / n, Ncum, u)
    return a[None] + t[:, None] * (b - a)[None]


def _subdivide(polys: list[np.ndarray], scale: float, anchors: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Points on the polygon edges (vertices, the ``anchors`` lying on an edge, graded points between them) and the
    constrained segments between consecutive points.

    Returns:
        points ``(m, 2)``, segments ``(s, 2)`` (indices into points), polygon id of every segment, mask of the anchors
        used (``(len(anchors),)``).
    """
    pts, segs, pid = [], [], []
    used = np.zeros(len(anchors), dtype=bool)
    for p, P in enumerate(polys):
        loop = []
        for i in range(len(P)):
            a, b = P[i], P[(i + 1) % len(P)]
            ab = b - a
            L = float(np.linalg.norm(ab))
            t = ((anchors - a) @ ab) / (L * L) if len(anchors) else np.zeros(0)
            off = np.abs((anchors - a) @ np.array([-ab[1], ab[0]])) / L if len(anchors) else np.zeros(0)
            on = (off < 1e-7) & (t > 1e-9) & (t < 1 - 1e-9)
            used |= on
            ts = np.concatenate([[0.0], np.sort(t[on]), [1.0]])
            for k in range(len(ts) - 1):
                pa, pb = a + ts[k] * ab, a + ts[k + 1] * ab
                loop.append(pa)
                loop.extend(_graded(pa, pb, scale))
        base = len(pts)
        pts.extend(loop)
        m = len(loop)
        segs.extend([(base + k, base + (k + 1) % m) for k in range(m)])
        pid.extend([p] * m)
    return np.array(pts), np.array(segs, dtype=np.int64), np.array(pid), used


def _point_seg_dist(X: np.ndarray, A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Distance of every point ``X (n, 2)`` to the nearest segment ``A[j] B[j]`` (brute force in chunks)."""
    out = np.full(len(X), np.inf)
    AB = B - A
    L2 = np.maximum((AB ** 2).sum(1), 1e-300)
    for s in range(0, len(X), 4096):
        x = X[s:s + 4096, None, :]
        t = np.clip(((x - A[None]) * AB[None]).sum(-1) / L2[None], 0.0, 1.0)
        d = np.linalg.norm(x - (A[None] + t[..., None] * AB[None]), axis=-1)
        out[s:s + 4096] = d.min(1)
    return out


def _tri_signed_area(X: np.ndarray, F: np.ndarray) -> np.ndarray:
    P = X[F]
    return 0.5 * ((P[:, 1, 0] - P[:, 0, 0]) * (P[:, 2, 1] - P[:, 0, 1]) -
                  (P[:, 1, 1] - P[:, 0, 1]) * (P[:, 2, 0] - P[:, 0, 0]))


def _inside_poly(X: np.ndarray, P: np.ndarray) -> np.ndarray:
    from matplotlib.path import Path
    return Path(P).contains_points(X)


def build_mesh(scale: float = 1.0, seed: int = 11) -> dict:
    """The fixed conforming mesh of the device family (``scale`` multiplies the size field).

    Gap / armature band: the equilateral lattice of :func:`band_lattice` (spacing ``4 scale`` mm), so that the movable
    iron parts get regular staircase interfaces.  Interfaces: graded points (lattice points on an interface are kept
    as its points).  Other interior points: candidates drawn with density ~ 1 / h^2 and thinned to a minimum spacing
    ``0.8 h`` (dart throwing in random order); points closer than ``0.55 h`` to a fixed interface are dropped.  scipy
    Delaunay; interior points inside the diametral circle of a missing constrained segment are removed until every
    interface segment is a mesh edge.  Triangles counter-clockwise.

    Returns:
        dict with ``pos (n0, 2)`` mm, ``faces (n2, 3)``, ``region (n2,)``, ``boundary (n0,)`` (box vertices) and mesh
        statistics.
    """
    rng = np.random.default_rng([seed, int(round(scale * 1000))])
    polys = fixed_polygons()
    lat = band_lattice(scale)
    pp, segs, seg_poly, on_edge = _subdivide(polys, scale, lat)
    lat = lat[~on_edge]
    if len(lat):                                   # lattice points too close to an interface point are dropped
        dmin, _ = cKDTree(pp).query(lat)
        lat = lat[dmin > 0.3 * LATTICE_S * scale]
    fp = np.concatenate([pp, lat])
    if len(np.unique(np.round(fp, 9), axis=0)) != len(fp):
        raise RuntimeError("fixed points are duplicated")
    # candidates
    area_box = (BOX[1] - BOX[0]) * (BOX[3] - BOX[2])
    hmin = scale * H_BAND
    n_cand = int(6.0 * 1.155 * area_box / hmin ** 2 / 3.0)
    C = np.stack([rng.uniform(BOX[0], BOX[1], n_cand), rng.uniform(BOX[2], BOX[3], n_cand)], 1)
    hc = size_field(C, scale)
    keep = rng.random(n_cand) < (hmin / hc) ** 2
    C, hc = C[keep], hc[keep]
    dseg = _point_seg_dist(C, fp[segs[:, 0]], fp[segs[:, 1]])
    keep = dseg > 0.55 * hc
    C, hc = C[keep], hc[keep]
    order = rng.permutation(len(C))
    C, hc = C[order], hc[order]
    allp = np.concatenate([fp, C])
    tree = cKDTree(allp)
    nbrs = tree.query_ball_point(C, r=0.8 * hc)
    nf = len(fp)
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
    vol = C[acc[nf:]]
    # triangulate + recover the constrained segments
    n_split = 0
    for it in range(40):
        X = np.concatenate([pp, lat, vol])
        F = Delaunay(X).simplices.astype(np.int64)
        sa = _tri_signed_area(X, F)
        F[sa < 0] = F[sa < 0][:, [0, 2, 1]]
        F = F[np.abs(sa) > 1e-9]
        n0 = len(X)
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        ek = np.unique(E[:, 0] * n0 + E[:, 1])
        sk = np.sort(segs, 1)
        miss = ~np.isin(sk[:, 0] * n0 + sk[:, 1], ek)
        if not miss.any():
            break
        A_, B_ = X[segs[miss, 0]], X[segs[miss, 1]]
        mid, rad = 0.5 * (A_ + B_), 0.5 * np.linalg.norm(B_ - A_, axis=1) * 1.05
        bad = np.zeros(len(vol), dtype=bool)
        for m_, r_ in zip(mid, rad):
            bad |= np.linalg.norm(vol - m_, axis=1) < r_
        if bad.any():
            vol = vol[~bad]
            continue
        bad = np.zeros(len(lat), dtype=bool)
        for m_, r_ in zip(mid, rad):
            bad |= np.linalg.norm(lat - m_, axis=1) < r_
        if bad.any():
            lat = lat[~bad]
            continue
        # encroached by another interface (narrow clearances): split the missing segments at their midpoints
        new_idx = len(pp) + np.arange(int(miss.sum()))
        pp = np.concatenate([pp, mid])
        keep_s = segs[~miss]
        split = np.concatenate([np.stack([segs[miss, 0], new_idx], 1), np.stack([new_idx, segs[miss, 1]], 1)])
        seg_poly = np.concatenate([seg_poly[~miss], seg_poly[miss], seg_poly[miss]])
        segs = np.concatenate([keep_s, split])
        n_split += int(miss.sum())
    else:
        raise RuntimeError("could not recover the constrained segments")
    # regions (exact: the fixed interfaces are mesh edges)
    cent = X[F].mean(1)
    region = np.zeros(len(F), dtype=np.int8)
    for j, (x0, x1) in enumerate(COIL_X):
        inside = (cent[:, 0] > x0) & (cent[:, 0] < x1) & (cent[:, 1] > COIL_Y[0]) & (cent[:, 1] < COIL_Y[1])
        region[inside] = j + 1
    region[_inside_poly(cent, CORE)] = REGION_CORE
    # every vertex of a region face lies inside or on the region's polygon (no face straddles an interface)
    for rid, P in [(REGION_CORE, CORE)] + [(j + 1, _rect(x0, x1, COIL_Y[0], COIL_Y[1])) for j, (x0, x1) in
                                           enumerate(COIL_X)]:
        from matplotlib.path import Path
        path = Path(P)
        V = X[F[region == rid]].reshape(-1, 2)
        ok = path.contains_points(V, radius=1e-6) | path.contains_points(V, radius=-1e-6)
        if not ok.all():
            raise RuntimeError(f"region {rid}: a face straddles its interface")
    on_box = (np.abs(X[:, 0] - BOX[0]) < 1e-9) | (np.abs(X[:, 0] - BOX[1]) < 1e-9) | \
             (np.abs(X[:, 1] - BOX[2]) < 1e-9) | (np.abs(X[:, 1] - BOX[3]) < 1e-9)
    n0 = len(X)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    ek, cnt = np.unique(E[:, 0] * n0 + E[:, 1], return_counts=True)
    bverts = np.unique(np.stack([ek[cnt == 1] // n0, ek[cnt == 1] % n0], 1))
    bmask = np.zeros(n0, dtype=bool)
    bmask[bverts] = True
    if not np.array_equal(bmask, on_box):
        raise RuntimeError("boundary vertices of the triangulation differ from the box vertices")
    ang = []
    Q = X[F]
    for k in range(3):
        a_, b_ = Q[:, (k + 1) % 3] - Q[:, k], Q[:, (k + 2) % 3] - Q[:, k]
        ang.append(np.degrees(np.arccos(np.clip((a_ * b_).sum(1) / np.linalg.norm(a_, axis=1) /
                                                np.linalg.norm(b_, axis=1), -1, 1))))
    ang = np.stack(ang, 1)
    used = np.unique(F)
    stats = dict(scale=scale, seed=seed, n0=int(n0), n1=int(len(ek)), n2=int(len(F)), euler=int(n0 - len(ek) + len(F)),
                 n_interface_points=int(len(pp)), n_lattice=int(len(lat)), n_interior=int(len(vol)),
                 recovery_iterations=int(it), segments_split=int(n_split),
                 unused=int(n0 - len(used)), nonmanifold=int((cnt > 2).sum()), min_angle=float(ang.min()),
                 min_angle_p1=float(np.percentile(ang.min(1), 1)), max_angle=float(ang.max()),
                 frac_obtuse=float((ang.max(1) > 90.0).mean()), n_core_faces=int((region == REGION_CORE).sum()),
                 n_coil_faces=int(((region >= 1) & (region <= 6)).sum()),
                 n_band_nodes=int(((X[:, 0] > BAND[0]) & (X[:, 0] < BAND[1]) & (X[:, 1] > BAND[2]) &
                                   (X[:, 1] < BAND[3])).sum()))
    if stats["euler"] != 1 or stats["unused"] or stats["nonmanifold"]:
        raise RuntimeError(f"invalid mesh: {stats}")
    return dict(pos=X, faces=F, region=region, boundary=on_box, stats=stats)


# ---------------------------------------------------------------------------------------------------------------------
# per-sample layout
# ---------------------------------------------------------------------------------------------------------------------
def draw_params(idx: int, seed: int = SEED) -> dict:
    """Layout and excitation of sample ``idx`` (independent of the mesh)."""
    rng = np.random.default_rng([seed, idx])
    gL, gR = rng.uniform(6.0, 24.0, 2)
    p = dict(gL=float(gL), gR=float(gR), t=float(rng.uniform(16.0, 40.0)), w=float(rng.uniform(176.0, 288.0)),
             x0=float(rng.uniform(-24.0, 24.0)))
    shoes = []
    for (x0, x1) in LIMBS:
        for side in (0, 1):
            e = float(rng.uniform(4.0, 16.0)) if rng.random() < 0.6 else 0.0
            shoes.append(e)
    p["shoes"] = shoes                                           # (left limb L, R, centre L, R, right limb L, R)
    n_ext = int(rng.choice(3, p=[0.4, 0.35, 0.25]))
    ext = []
    tries = 0
    while len(ext) < n_ext and tries < 200:
        tries += 1
        a, b = rng.uniform(24.0, 64.0, 2)
        th = float(rng.uniform(0.0, math.pi))
        dist = float(rng.uniform(10.0, 70.0))
        side = int(rng.integers(4))
        r = 0.5 * math.hypot(a, b)
        if side == 0:      # left
            c = (DEVICE_BOX[0] - dist - r, float(rng.uniform(DEVICE_BOX[2], DEVICE_BOX[3])))
        elif side == 1:    # right
            c = (DEVICE_BOX[1] + dist + r, float(rng.uniform(DEVICE_BOX[2], DEVICE_BOX[3])))
        elif side == 2:    # below
            c = (float(rng.uniform(DEVICE_BOX[0], DEVICE_BOX[1])), DEVICE_BOX[2] - dist - r)
        else:              # above
            c = (float(rng.uniform(DEVICE_BOX[0], DEVICE_BOX[1])), DEVICE_BOX[3] + dist + r)
        if not (BOX[0] + 30 + r < c[0] < BOX[1] - 30 - r and BOX[2] + 30 + r < c[1] < BOX[3] - 30 - r):
            continue
        if any(math.hypot(c[0] - e["c"][0], c[1] - e["c"][1]) < r + e["r"] + 10.0 for e in ext):
            continue
        ext.append(dict(c=c, a=float(a), b=float(b), th=th, r=r))
    p["ext"] = [dict(c=list(e["c"]), a=e["a"], b=e["b"], th=e["th"]) for e in ext]
    p["currents"] = [float(v) for v in rng.uniform(-J_MAX, J_MAX, 3)]
    return p


def _in_rot_rect(X: np.ndarray, c, a: float, b: float, th: float) -> np.ndarray:
    d = X - np.asarray(c)[None]
    u = d[:, 0] * math.cos(th) + d[:, 1] * math.sin(th)
    v = -d[:, 0] * math.sin(th) + d[:, 1] * math.cos(th)
    return (np.abs(u) < 0.5 * a) & (np.abs(v) < 0.5 * b)


def armature_corners(p: dict) -> np.ndarray:
    """The four corners of the armature (counter-clockwise, mm)."""
    yl, yr = POLE_Y + p["gL"], POLE_Y + p["gR"]
    slope = (yr - yl) / 224.0
    th = math.atan(slope)
    t_vec = np.array([math.cos(th), math.sin(th)])
    n_vec = np.array([-math.sin(th), math.cos(th)])
    y0 = 0.5 * (yl + yr) + slope * p["x0"]                       # bottom edge at x = x0
    bc = np.array([p["x0"], y0])
    hw = 0.5 * p["w"]
    return np.stack([bc - hw * t_vec, bc + hw * t_vec, bc + hw * t_vec + p["t"] * n_vec, bc - hw * t_vec + p["t"] * n_vec])


def face_phase(mesh: dict, p: dict) -> np.ndarray:
    """Per-face material phase of sample ``p`` on ``mesh``: fixed regions exactly, movable iron by centroid test."""
    X, F, region = mesh["pos"], mesh["faces"], mesh["region"]
    cent = X[F].mean(1)
    phase = np.zeros(len(F), dtype=np.int8)
    phase[(region >= 1) & (region <= 6)] = PHASE_COPPER
    phase[region == REGION_CORE] = PHASE_IRON
    iron = _inside_poly(cent, armature_corners(p))
    k = 0
    for (x0, x1) in LIMBS:
        eL, eR = p["shoes"][k], p["shoes"][k + 1]
        k += 2
        if eL > 0:
            iron |= (cent[:, 0] > x0 - eL) & (cent[:, 0] < x0) & (cent[:, 1] > SHOE_Y[0]) & (cent[:, 1] < SHOE_Y[1])
        if eR > 0:
            iron |= (cent[:, 0] > x1) & (cent[:, 0] < x1 + eR) & (cent[:, 1] > SHOE_Y[0]) & (cent[:, 1] < SHOE_Y[1])
    for e in p["ext"]:
        iron |= _in_rot_rect(cent, e["c"], e["a"], e["b"], e["th"])
    movable = iron & (region == REGION_AIR)
    phase[movable] = PHASE_IRON
    return phase


def face_current(mesh: dict, currents) -> np.ndarray:
    """Current density per face (A/mm^2)."""
    region = mesh["region"]
    J = np.zeros(len(region))
    for j in range(6):
        J[region == j + 1] = COIL_SIGN[j] * currents[COIL_WINDING[j]]
    return J


# ---------------------------------------------------------------------------------------------------------------------
# P1 finite elements + Newton
# ---------------------------------------------------------------------------------------------------------------------
class P1Mesh:
    """Precomputed P1 geometry (SI units) and the CSR pattern of the stiffness matrix."""

    def __init__(self, pos_mm: np.ndarray, faces: np.ndarray, boundary: np.ndarray):
        X = pos_mm * MM
        self.X, self.F = X, faces
        self.n0, self.n2 = len(X), len(faces)
        P = X[faces]
        J = np.stack([P[:, 1] - P[:, 0], P[:, 2] - P[:, 0]], axis=2)              # columns e1, e2
        det = J[:, 0, 0] * J[:, 1, 1] - J[:, 0, 1] * J[:, 1, 0]
        self.area = 0.5 * np.abs(det)
        Jinv = np.linalg.inv(J)
        G = np.zeros((len(faces), 3, 2))
        G[:, 1:, :] = Jinv
        G[:, 0, :] = -G[:, 1, :] - G[:, 2, :]
        self.G = G                                                                 # (n2, 3, 2) grad lambda_i (1/m)
        rows = np.repeat(faces, 3, axis=1).ravel()
        cols = np.tile(faces, (1, 3)).ravel()
        Kpat = sp.csr_matrix((np.arange(len(rows), dtype=np.float64) + 1.0, (rows, cols)), shape=(self.n0, self.n0))
        # map every local entry to its CSR slot (duplicates sum)
        Kpat.sum_duplicates()
        coo_key = rows.astype(np.int64) * self.n0 + cols
        csr_rows = np.repeat(np.arange(self.n0), np.diff(Kpat.indptr))
        csr_key = csr_rows.astype(np.int64) * self.n0 + Kpat.indices
        self.slot = np.searchsorted(csr_key, coo_key)
        self.indptr, self.indices = Kpat.indptr.copy(), Kpat.indices.copy()
        self.free = np.where(~boundary)[0]
        self.mass = np.zeros(self.n0)
        np.add.at(self.mass, faces.ravel(), np.repeat(self.area / 3.0, 3))

    def csr(self, local: np.ndarray) -> sp.csr_matrix:
        data = np.bincount(self.slot, weights=local.ravel(), minlength=len(self.indices))
        return sp.csr_matrix((data, self.indices, self.indptr), shape=(self.n0, self.n0))

    def grad(self, A: np.ndarray) -> np.ndarray:
        """Element gradients ``(n2, 2)`` of the P1 field ``A (n0,)``."""
        return np.einsum("fid,fi->fd", self.G, A[self.F])

    def load(self, J_face_si: np.ndarray) -> np.ndarray:
        """Consistent load ``F_i = sum_f J_f |f| / 3``."""
        Fv = np.zeros(self.n0)
        np.add.at(Fv, self.F.ravel(), np.repeat(J_face_si * self.area / 3.0, 3))
        return Fv


def _material_eval(B: np.ndarray, iron: np.ndarray):
    """``nu``, ``d nu / d B`` and the energy density per face."""
    nu = np.full(B.shape, NU0)
    dnu = np.zeros(B.shape)
    W = 0.5 * NU0 * B * B
    if iron.any():
        Bi = B[iron]
        nu[iron] = nu_iron(Bi)
        dnu[iron] = dnu_iron(Bi)
        W[iron] = energy_iron(Bi)
    return nu, dnu, W


def newton_solve(m: P1Mesh, phase: np.ndarray, J_face_amm2: np.ndarray | None, tol: float = NEWTON_TOL,
                 max_it: int = NEWTON_MAX, linear: bool = False, load: np.ndarray | None = None) -> dict:
    """Damped Newton for ``-div(nu(|grad A|) grad A) = J``, ``A = 0`` on the box.

    ``linear=True``: one solve with the initial permeability of the iron (``nu = nu_iron(0)``).  ``load``: a
    precomputed load vector (SI) instead of the element-wise constant current density (verification).

    Returns:
        dict with ``A (n0,)`` Wb/m, ``Bf (n2, 2)`` T, ``nu (n2,)``, ``it``, ``res`` (relative free residual), ``energy``
        (J/m), the residual history and the number of line-search halvings.
    """
    iron = phase == PHASE_IRON
    Fv = m.load(J_face_amm2 * AMM2) if load is None else load
    fr = m.free
    nF = float(np.linalg.norm(Fv[fr])) or 1.0
    A = np.zeros(m.n0)
    GG = np.einsum("fid,fjd->fij", m.G, m.G) * m.area[:, None, None]             # (n2, 3, 3) |f| G G^T

    def assemble_secant(nu):
        return m.csr(GG * nu[:, None, None])

    nu0 = np.where(iron, nu_iron(np.zeros(1))[0], NU0)
    Ksec = assemble_secant(nu0)[fr][:, fr]
    A[fr] = spla.spsolve(Ksec.tocsc(), Fv[fr])
    if linear:
        g = m.grad(A)
        B = np.linalg.norm(g, axis=1)
        R = Ksec @ A[fr] - Fv[fr]
        return dict(A=A, Bf=np.stack([g[:, 1], -g[:, 0]], 1), nu=nu0, it=1, res=float(np.linalg.norm(R) / nF),
                    energy=float(0.5 * (m.area * nu0 * B * B).sum()), hist=[], halvings=0)

    def energy_and_grad(Avec):
        g = m.grad(Avec)
        B = np.linalg.norm(g, axis=1)
        nu, dnu, W = _material_eval(B, iron)
        e_mag, e_src = float((m.area * W).sum()), float(Fv[fr] @ Avec[fr])
        return e_mag - e_src, g, B, nu, dnu, abs(e_mag) + abs(e_src)

    Pi, g, B, nu, dnu, scale = energy_and_grad(A)
    hist, halvings = [], 0
    it = 0
    res = math.inf
    for it in range(1, max_it + 1):
        Ksec = assemble_secant(nu)
        R = (Ksec @ A - Fv)[fr]
        res = float(np.linalg.norm(R) / nF)
        hist.append(res)
        if res <= tol:
            break
        # tangent: |f| G [nu I + (dnu/dB / B) g g^T] G^T
        c = np.where(B > 0, dnu / np.maximum(B, 1e-300), 0.0)
        Gg = np.einsum("fid,fd->fi", m.G, g)                                       # (n2, 3)
        Kt_loc = GG * nu[:, None, None] + (m.area * c)[:, None, None] * Gg[:, :, None] * Gg[:, None, :]
        Kt = m.csr(Kt_loc)[fr][:, fr]
        delta = spla.spsolve(Kt.tocsc(), -R)
        slope = float(R @ delta)
        alpha = 1.0
        while True:              # Armijo on the convex energy, with a round-off allowance of the energy sums
            An = A.copy()
            An[fr] += alpha * delta
            Pn, gn, Bn, nun, dnun, sn = energy_and_grad(An)
            if Pn <= Pi + 1e-4 * alpha * slope + 1e-12 * max(scale, sn) or alpha < 1e-6:
                break
            alpha *= 0.5
            halvings += 1
        A, Pi, g, B, nu, dnu, scale = An, Pn, gn, Bn, nun, dnun, sn
    else:
        Ksec = assemble_secant(nu)
        res = float(np.linalg.norm((Ksec @ A - Fv)[fr]) / nF)
    _, _, W = _material_eval(B, iron)
    return dict(A=A, Bf=np.stack([g[:, 1], -g[:, 0]], 1), nu=nu, it=it, res=res, energy=float((m.area * W).sum()),
                hist=hist, halvings=halvings)


def picard_solve(m: P1Mesh, phase: np.ndarray, J_face_amm2: np.ndarray, omega: float, iters: int,
                 A_ref: np.ndarray | None = None) -> dict:
    """Secant (Picard / fixed-point) iteration ``A_j = (1 - omega) A_{j-1} + omega K(nu(B_{j-1}))^{-1} F`` from
    ``A_0 = 0``: an independent iteration for the same discrete problem (verification; the structure of the
    fixed-point solve layer of the RHMP model).  Returns the final iterate and, with ``A_ref``, the relative error
    after every iteration."""
    iron = phase == PHASE_IRON
    Fv = m.load(J_face_amm2 * AMM2)
    fr = m.free
    GG = np.einsum("fid,fjd->fij", m.G, m.G) * m.area[:, None, None]
    A = np.zeros(m.n0)
    errs = []
    for _ in range(iters):
        B = np.linalg.norm(m.grad(A), axis=1)
        nu, _, _ = _material_eval(B, iron)
        K = m.csr(GG * nu[:, None, None])[fr][:, fr]
        An = np.zeros(m.n0)
        An[fr] = spla.spsolve(K.tocsc(), Fv[fr])
        A = (1.0 - omega) * A + omega * An
        if A_ref is not None:
            errs.append(float(np.linalg.norm(A - A_ref) / np.linalg.norm(A_ref)))
    return dict(A=A, errs=errs)


def node_field(m: P1Mesh, Bf: np.ndarray) -> np.ndarray:
    """Area-weighted average of the element fields over the incident triangles, ``(n0, 2)``."""
    num = np.zeros((m.n0, 2))
    den = np.zeros(m.n0)
    for k in range(3):
        np.add.at(num, m.F[:, k], Bf * m.area[:, None])
        np.add.at(den, m.F[:, k], m.area)
    return num / den[:, None]


def flux_linkages(m: P1Mesh, mesh: dict, A: np.ndarray) -> list[float]:
    """Flux linkage per turn of each winding (Wb/m): mean of A over the + side minus the mean over the - side."""
    region = mesh["region"]
    Af = A[m.F].mean(1)
    out = []
    for w in range(3):
        val = 0.0
        for j in np.where(COIL_WINDING == w)[0]:
            sel = region == j + 1
            val += COIL_SIGN[j] * float((Af[sel] * m.area[sel]).sum() / m.area[sel].sum())
        out.append(val)
    return out


# ---------------------------------------------------------------------------------------------------------------------
# data sets
# ---------------------------------------------------------------------------------------------------------------------
_W: dict = {}


def _init_worker(mesh: dict) -> None:
    _W["mesh"] = mesh
    _W["m"] = P1Mesh(mesh["pos"], mesh["faces"], mesh["boundary"])


def _solve_sample(args) -> dict:
    idx, factor = args
    mesh, m = _W["mesh"], _W["m"]
    p = draw_params(idx)
    cur = [factor * c for c in p["currents"]]
    phase = face_phase(mesh, p)
    J = face_current(mesh, cur)
    t0 = time.time()
    sol = newton_solve(m, phase, J)
    Bn = node_field(m, sol["Bf"])
    Bmag = np.linalg.norm(sol["Bf"], axis=1)
    iron = phase == PHASE_IRON
    return dict(idx=idx, currents=np.array(cur), phase=phase, A=sol["A"], B_node=Bn.astype(np.float32),
                it=sol["it"], res=sol["res"], halvings=sol["halvings"], energy=sol["energy"],
                lam=np.array(flux_linkages(m, mesh, sol["A"])), seconds=time.time() - t0,
                bmax_iron=float(Bmag[iron].max()) if iron.any() else 0.0,
                bmax=float(Bmag.max()), b_p99_iron=float(np.percentile(Bmag[iron], 99)) if iron.any() else 0.0,
                n_iron_movable=int((iron & (mesh["region"] == REGION_AIR)).sum()), params=p)


def make_set(mesh: dict, ids: list[int], factor: float, workers: int) -> list[dict]:
    with Pool(workers, initializer=_init_worker, initargs=(mesh,)) as pool:
        return pool.map(_solve_sample, [(i, factor) for i in ids], chunksize=4)


def pack(mesh: dict, rows: list[dict], meta: dict) -> dict:
    import torch
    out = dict(pos=torch.tensor(mesh["pos"]), faces=torch.tensor(mesh["faces"]), region=torch.tensor(mesh["region"]),
               boundary=torch.tensor(mesh["boundary"]), coil_sign=torch.tensor(COIL_SIGN),
               coil_winding=torch.tensor(COIL_WINDING), sample_ids=torch.tensor([r["idx"] for r in rows]),
               currents=torch.tensor(np.stack([r["currents"] for r in rows])),
               phase=torch.tensor(np.stack([r["phase"] for r in rows])),
               A=torch.tensor(np.stack([r["A"] for r in rows])),
               B_node=torch.tensor(np.stack([r["B_node"] for r in rows])),
               newton_it=torch.tensor([r["it"] for r in rows]), newton_res=torch.tensor([r["res"] for r in rows]),
               energy=torch.tensor([r["energy"] for r in rows]), linkage=torch.tensor(np.stack([r["lam"] for r in rows])),
               bmax_iron=torch.tensor([r["bmax_iron"] for r in rows]),
               b_p99_iron=torch.tensor([r["b_p99_iron"] for r in rows]),
               meta=dict(meta, params=[r["params"] for r in rows], mesh_stats=mesh["stats"]))
    return out


def _summary(rows: list[dict], mesh: dict, extra: dict) -> dict:
    it = np.array([r["it"] for r in rows])
    res = np.array([r["res"] for r in rows])
    bm = np.array([r["bmax_iron"] for r in rows])
    bp = np.array([r["b_p99_iron"] for r in rows])
    return dict(extra, N=len(rows), mesh=mesh["stats"], newton_it=dict(mean=float(it.mean()), max=int(it.max()),
                                                                       p50=float(np.median(it))),
                newton_res_max=float(res.max()), seconds_mean=float(np.mean([r["seconds"] for r in rows])),
                bmax_iron=dict(p10=float(np.percentile(bm, 10)), p50=float(np.median(bm)),
                               p90=float(np.percentile(bm, 90)), max=float(bm.max())),
                frac_bmax_iron_gt_1p6=float((bm > 1.6).mean()), frac_bmax_iron_gt_1p9=float((bm > 1.9).mean()),
                frac_bp99_gt_1p6=float((bp > 1.6).mean()), frac_bmax_iron_lt_1p2=float((bm < 1.2).mean()),
                halvings_total=int(sum(r["halvings"] for r in rows)))


def _law_meta() -> dict:
    return dict(law="Froehlich-Kennelly: B = mu0 H + Js H / (Hk + H)", Js=JS, mur_init=MUR_INIT, Hk=HK, mu0=MU0,
                units=dict(length="mm", B="T", J="A/mm^2", A="Wb/m (T m)", nu="m/H"))


def cmd_data(a) -> None:
    import torch
    out_dir = os.path.join(ROOT, "datasets", "v2")
    os.makedirs(out_dir, exist_ok=True)
    meshes = {}
    for name in ("base", "fine", "coarse"):
        t0 = time.time()
        meshes[name] = build_mesh(SCALES[name], MESH_SEEDS[name])
        print(f"mesh {name}: {meshes[name]['stats']} ({time.time() - t0:.1f}s)", flush=True)
    jobs = [("MXS", "base", list(range(a.n)), 1.0)]
    ext_ids = list(range(TEST_START, min(TEST_START + N_EXTRA, a.n))) if a.n > TEST_START else list(range(min(a.n, 50)))
    jobs += [("MXS_ood15", "base", ext_ids, OOD_FACTOR), ("MXS_fine", "fine", ext_ids, 1.0),
             ("MXS_coarse", "coarse", ext_ids, 1.0)]
    for fname, mname, ids, factor in jobs:
        if a.only and fname not in a.only.split(","):
            continue
        t0 = time.time()
        rows = make_set(meshes[mname], ids, factor, a.workers)
        meta = dict(generator="datasets/generators/gen_maxwell.py", seed=SEED, mesh=mname, scale=SCALES[mname],
                    mesh_seed=MESH_SEEDS[mname], current_factor=factor, J_max=J_MAX, box=BOX, **_law_meta(),
                    split="sequential 70/15/15 of the 5000 base samples; test100 = 4250..4349",
                    created=time.strftime("%Y-%m-%d %H:%M:%S"))
        blob = pack(meshes[mname], rows, meta)
        torch.save(blob, os.path.join(out_dir, fname + ".pt"))
        summ = _summary(rows, meshes[mname], dict(file=fname, factor=factor, mesh=mname,
                                                  seconds_total=time.time() - t0))
        with open(os.path.join(out_dir, fname + ".json"), "w") as f:
            json.dump(summ, f, indent=1)
        print(json.dumps(summ), flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------------------------------------------------
_TRI7 = (np.array([[1 / 3, 1 / 3, 1 / 3],
                   [0.0597158717, 0.4701420641, 0.4701420641], [0.4701420641, 0.0597158717, 0.4701420641],
                   [0.4701420641, 0.4701420641, 0.0597158717],
                   [0.7974269853, 0.1012865073, 0.1012865073], [0.1012865073, 0.7974269853, 0.1012865073],
                   [0.1012865073, 0.1012865073, 0.7974269853]]),
         np.array([0.225] + [0.1323941527] * 3 + [0.1259391805] * 3))         # degree-5 rule (Dunavant)


def _structured_square(n: int, L_mm: float):
    xs = np.linspace(0.0, L_mm, n + 1)
    X, Y = np.meshgrid(xs, xs, indexing="ij")
    pos = np.stack([X.ravel(), Y.ravel()], 1)
    F = []
    for i in range(n):
        for j in range(n):
            a, b, c, d = i * (n + 1) + j, (i + 1) * (n + 1) + j, (i + 1) * (n + 1) + j + 1, i * (n + 1) + j + 1
            F += [[a, b, c], [a, c, d]] if (i + j) % 2 == 0 else [[a, b, d], [b, c, d]]
    bnd = (np.abs(pos[:, 0]) < 1e-12) | (np.abs(pos[:, 0] - L_mm) < 1e-12) | (np.abs(pos[:, 1]) < 1e-12) | \
          (np.abs(pos[:, 1] - L_mm) < 1e-12)
    return pos, np.array(F, dtype=np.int64), bnd


def manufactured_convergence(ns=(8, 16, 32, 64, 128), L_mm: float = 200.0, B_max: float = 2.4) -> dict:
    """P1 + Newton convergence for ``A = A0 sin(pi x / L) sin(pi y / L)`` in iron everywhere (``|B|`` up to
    ``B_max``, deep saturation), ``J = -div(nu(|grad A|) grad A)`` by central differences, degree-5 quadrature load."""
    L = L_mm * MM
    A0 = B_max * L / math.pi

    def A_ex(x, y):
        return A0 * np.sin(math.pi * x / L) * np.sin(math.pi * y / L)

    def grad_ex(x, y):
        k = math.pi / L
        return np.stack([A0 * k * np.cos(k * x) * np.sin(k * y), A0 * k * np.sin(k * x) * np.cos(k * y)], -1)

    def flux(x, y):
        g = grad_ex(x, y)
        return nu_iron(np.linalg.norm(g, axis=-1))[..., None] * g

    def J_ex(x, y, d=1e-7):
        return -((flux(x + d, y)[..., 0] - flux(x - d, y)[..., 0]) + (flux(x, y + d)[..., 1] -
                                                                       flux(x, y - d)[..., 1])) / (2 * d)

    lam, w = _TRI7
    rows = []
    for n in ns:
        pos, F, bnd = _structured_square(n, L_mm)
        m = P1Mesh(pos, F, bnd)
        Xq = np.einsum("qk,fkd->fqd", lam, m.X[F])                                   # (n2, 7, 2) SI
        Jq = J_ex(Xq[..., 0], Xq[..., 1])
        Fv = np.zeros(m.n0)
        for k in range(3):
            np.add.at(Fv, F[:, k], (m.area[:, None] * w[None] * Jq * lam[None, :, k]).sum(1))
        sol = newton_solve(m, np.full(len(F), PHASE_IRON, dtype=np.int8), None, load=Fv, tol=1e-12)
        Ah = np.einsum("qk,fk->fq", lam, sol["A"][F])
        eL2 = math.sqrt(float((m.area[:, None] * w[None] * (Ah - A_ex(Xq[..., 0], Xq[..., 1])) ** 2).sum()))
        nL2 = math.sqrt(float((m.area[:, None] * w[None] * A_ex(Xq[..., 0], Xq[..., 1]) ** 2).sum()))
        gh = m.grad(sol["A"])
        eH1 = math.sqrt(float((m.area[:, None] * w[None] * ((gh[:, None, :] - grad_ex(Xq[..., 0], Xq[..., 1])) ** 2)
                               .sum(-1)).sum()))
        nH1 = math.sqrt(float((m.area[:, None] * w[None] * (grad_ex(Xq[..., 0], Xq[..., 1]) ** 2).sum(-1)).sum()))
        rows.append(dict(n=n, h_mm=L_mm / n, n0=m.n0, newton_it=sol["it"], newton_res=sol["res"],
                         rel_L2_A=eL2 / nL2, rel_L2_B=eH1 / nH1))
    for a_, b_ in zip(rows[:-1], rows[1:]):
        b_["rate_L2_A"] = math.log(a_["rel_L2_A"] / b_["rel_L2_A"]) / math.log(2.0)
        b_["rate_L2_B"] = math.log(a_["rel_L2_B"] / b_["rel_L2_B"]) / math.log(2.0)
    return dict(problem=f"A = A0 sin(pi x/L) sin(pi y/L), L = {L_mm} mm, |B| <= {B_max} T, iron everywhere",
                rows=rows)


def law_checks() -> dict:
    H = np.logspace(-2, 7, 2001)
    B = b_of_h(H)
    Hr = h_of_b(B)
    Bs = np.linspace(1e-4, 4.0, 4001)
    nu = nu_iron(Bs)
    d = 1e-6
    dnu_fd = (nu_iron(Bs + d) - nu_iron(Bs - d)) / (2 * d)
    W_fd = (energy_iron(Bs + d) - energy_iron(Bs - d)) / (2 * d)
    return dict(roundtrip_max_rel=float(np.max(np.abs(Hr - H) / H)), nu_monotone=bool(np.all(np.diff(nu) > 0)),
                dH_dB_positive=bool(np.all(np.diff(h_of_b(Bs)) > 0)),
                dnu_vs_fd_max_rel=float(np.max(np.abs(dnu_iron(Bs) - dnu_fd) / np.abs(dnu_fd))),
                dW_vs_H_max_rel=float(np.max(np.abs(W_fd - h_of_b(Bs)) / h_of_b(Bs))),
                mur_at_B0=float(NU0 / nu_iron(np.array([0.0]))[0]), nu_limit_over_nu0=float(nu_iron(np.array([50.0]))[0] / NU0),
                B_at_H={str(int(h)): float(b_of_h(np.array([h]))[0]) for h in (100, 500, 1000, 2500, 5000, 10000, 50000,
                                                                                  100000)},
                mur_at_B={str(b): float(NU0 / nu_iron(np.array([b]))[0]) for b in (0.5, 1.0, 1.5, 1.8, 2.0, 2.2, 2.5)})


def _energy(m: P1Mesh, phase: np.ndarray, Fv: np.ndarray, A: np.ndarray) -> float:
    B = np.linalg.norm(m.grad(A), axis=1)
    _, _, W = _material_eval(B, phase == PHASE_IRON)
    return float((m.area * W).sum() - Fv[m.free] @ A[m.free])


def _probe(m: P1Mesh, Bf: np.ndarray, pts_mm: np.ndarray) -> np.ndarray:
    """Element field at points (the containing triangle)."""
    from matplotlib.tri import Triangulation, TrapezoidMapTriFinder
    tri = Triangulation(m.X[:, 0], m.X[:, 1], m.F)
    f = TrapezoidMapTriFinder(tri)(pts_mm[:, 0] * MM, pts_mm[:, 1] * MM)
    return np.where(f[:, None] >= 0, Bf[np.maximum(f, 0)], np.nan)


PROBES = np.array([[0.0, 0.0], [-112.0, 0.0], [112.0, 0.0], [0.0, -68.0], [0.0, 51.0], [-64.0, 0.0]])


def _refinement_job(args):
    scale, seed, idx = args
    mesh = build_mesh(scale, seed)
    m = P1Mesh(mesh["pos"], mesh["faces"], mesh["boundary"])
    if idx < 0:
        p = dict(gL=6.0, gR=6.0, t=0.0, w=0.0, x0=0.0, shoes=[0.0] * 6, ext=[], currents=[4.0, 4.0, 4.0])
    else:
        p = draw_params(idx)
    ph = face_phase(mesh, p)
    J = face_current(mesh, p["currents"])
    sol = newton_solve(m, ph, J)
    return dict(scale=scale, idx=idx, n0=m.n0, energy=sol["energy"], linkage=flux_linkages(m, mesh, sol["A"]),
                probes=_probe(m, sol["Bf"], PROBES).tolist(), newton_it=sol["it"], res=sol["res"],
                iron_area_mm2=float((m.area[ph == PHASE_IRON]).sum() / MM ** 2))


def cmd_selftest(a) -> None:
    t0 = time.time()
    out = dict(law=law_checks())
    print("law", json.dumps(out["law"]), flush=True)
    out["manufactured"] = manufactured_convergence()
    for r in out["manufactured"]["rows"]:
        print("manufactured", json.dumps(r), flush=True)
    mesh = build_mesh(1.0, MESH_SEEDS["base"])
    m = P1Mesh(mesh["pos"], mesh["faces"], mesh["boundary"])
    lin, mini, pic = [], [], []
    rng = np.random.default_rng(0)
    for idx in (0, 4, 11, 9, 3):
        p = draw_params(idx)
        ph = face_phase(mesh, p)
        J = face_current(mesh, p["currents"])
        eps = 1e-4
        s_small = newton_solve(m, ph, J * eps)
        s_lin = newton_solve(m, ph, J, linear=True)
        lin.append(dict(idx=idx, rel_diff=float(np.linalg.norm(s_small["A"] / eps - s_lin["A"]) /
                                                np.linalg.norm(s_lin["A"]))))
        sol = newton_solve(m, ph, J)
        Fv = m.load(J * AMM2)
        E0 = _energy(m, ph, Fv, sol["A"])
        worst = math.inf
        for _ in range(20):
            dA = np.zeros(m.n0)
            dA[m.free] = rng.normal(size=len(m.free))
            for rel in (1e-4, 1e-2, 1e-1):
                dA_s = dA * rel * np.abs(sol["A"]).max() / np.abs(dA).max()
                worst = min(worst, _energy(m, ph, Fv, sol["A"] + dA_s) - E0)
        mini.append(dict(idx=idx, energy=E0, min_energy_increase=worst))
        Bm = np.linalg.norm(sol["Bf"], axis=1)
        row = dict(idx=idx, b_p99_iron=float(np.percentile(Bm[ph == PHASE_IRON], 99)), newton_it=sol["it"])
        for om in (1.0, 0.5, 0.25):
            r = picard_solve(m, ph, J, om, 64, A_ref=sol["A"])
            row[f"omega_{om}"] = {str(k): r["errs"][k - 1] for k in (1, 2, 4, 8, 16, 32, 64)}
        pic.append(row)
        print("sample", idx, lin[-1], mini[-1], row, flush=True)
    out["linear_limit"] = lin
    out["energy_minimum"] = mini
    out["picard_vs_newton"] = pic
    jobs = [(sc, 100 + int(sc * 100), idx) for sc in (2.0, 1.0, 0.5, 0.25) for idx in (-1, 0, 4, 11, 3)]
    with Pool(min(a.workers, len(jobs))) as pool:
        ref = pool.map(_refinement_job, jobs)
    out["refinement"] = ref
    for r in ref:
        print("refine", json.dumps(r), flush=True)
    out["seconds"] = time.time() - t0
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"selftest written to {a.out} ({out['seconds']:.0f}s)", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# secondary family MXE: linear electrostatics of an electrical capacitance tomography (ECT) sensor
# ---------------------------------------------------------------------------------------------------------------------
# Geometry (mm): grounded circular screen r = 100; insulating pipe wall 60 < r < 66 (fixed); 8 electrodes on the outer
# pipe surface (annular sectors 66 < r < 70, 30 deg each, centred at k 45 deg): conductors, i.e. holes of the mesh with
# phi = V_k on their boundary; flow region r < 60: oil background with water droplets / gas bubbles (staircase).
# -div(eps_r grad phi) = q (q = rho / eps_0 in V/mm^2: triboelectric space charge), phi = 0 on the screen.
ECT_R = dict(screen=100.0, pipe=60.0, wall=66.0, el=70.0)
ECT_N_EL, ECT_SPAN = 8, 30.0
ECT_PHASES = ("air", "wall", "oil", "water", "gas")
ECT_EPS = np.array([1.0, 3.5, 2.2, 80.0, 1.0])                     # relative permittivity per phase
ECT_REGION = dict(air=0, wall=1, pipe=2)
ECT_SEED = 20261002
ECT_V_MAX = 1.0                                                      # electrode potentials U(-1, 1) V
ECT_Q_MAX = 0.01                                                     # space-charge density / eps_0, V/mm^2
ECT_SCALES = {"base": 1.0, "fine": 0.5, "coarse": 2.0}


def ect_size_field(X: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """2.5 mm around the pipe wall and the electrodes (58 < r < 72), 3.4 mm in the flow, growing to ~7.5 mm at the
    screen."""
    r = np.linalg.norm(X, axis=1)
    h = np.where(r < 58.0, np.minimum(3.4, 2.5 + 0.25 * (58.0 - r)),
                 np.where(r > 72.0, 2.5 + 0.18 * (r - 72.0), 2.5))
    return scale * h


def _ect_el_angles() -> list[tuple[float, float]]:
    return [(math.radians(45.0 * k - 0.5 * ECT_SPAN), math.radians(45.0 * k + 0.5 * ECT_SPAN)) for k in range(ECT_N_EL)]


def _arc_points(r: float, a0: float, a1: float, scale: float, endpoint: bool) -> np.ndarray:
    h = float(ect_size_field(np.array([[r, 0.0]]), scale)[0])
    n = max(1, int(round(r * abs(a1 - a0) / h)))
    t = np.linspace(a0, a1, n + 1) if endpoint else np.linspace(a0, a1, n + 1)[:-1]
    return np.stack([r * np.cos(t), r * np.sin(t)], 1)


def ect_in_electrode(X: np.ndarray, pad: float = 0.0) -> np.ndarray:
    """Points strictly inside an electrode sector."""
    r = np.linalg.norm(X, axis=1)
    th = np.arctan2(X[:, 1], X[:, 0])
    inside = np.zeros(len(X), dtype=bool)
    for a0, a1 in _ect_el_angles():
        d = np.angle(np.exp(1j * (th - 0.5 * (a0 + a1))))
        inside |= (np.abs(d) < 0.5 * (a1 - a0) - pad) & (r > ECT_R["wall"] + pad) & (r < ECT_R["el"] - pad)
    return inside


def build_mesh_ect(scale: float = 1.0, seed: int = 21) -> dict:
    """Conforming mesh of the ECT sensor: screen circle, pipe circles and electrode outlines are mesh edges, the
    electrodes are holes (their outlines and the screen are the mesh boundary).  Same point construction and segment
    recovery as :func:`build_mesh`.

    Returns:
        dict with ``pos``, ``faces`` (counter-clockwise), ``region (n2,)`` (0 outside, 1 wall, 2 pipe), ``boundary``,
        ``electrode (n0,)`` (electrode id of boundary vertices, -1 elsewhere / screen) and statistics.
    """
    rng = np.random.default_rng([seed, int(round(scale * 1000))])
    els = _ect_el_angles()
    pts: list = []
    key: dict = {}

    def add(P):
        idx = []
        for x in P:
            k_ = (round(float(x[0]), 7), round(float(x[1]), 7))
            if k_ not in key:
                key[k_] = len(pts)
                pts.append(np.array(x, dtype=np.float64))
            idx.append(key[k_])
        return idx

    segs = []

    def polyline(idx, closed):
        n = len(idx)
        for i in range(n if closed else n - 1):
            segs.append((idx[i], idx[(i + 1) % n]))
    polyline(add(_arc_points(ECT_R["screen"], 0.0, 2 * math.pi, scale, False)), True)
    polyline(add(_arc_points(ECT_R["pipe"], 0.0, 2 * math.pi, scale, False)), True)
    cuts = sorted([a for e in els for a in e])                               # wall circle with electrode corners
    wall = []
    for i, a0 in enumerate(cuts):
        a1 = cuts[(i + 1) % len(cuts)] + (2 * math.pi if i == len(cuts) - 1 else 0.0)
        wall.extend(_arc_points(ECT_R["wall"], a0, a1, scale, False))
    polyline(add(np.array(wall)), True)
    for a0, a1 in els:                                                       # electrode outlines (open polylines)
        side0 = np.linspace(ECT_R["wall"], ECT_R["el"], 3)
        P0 = np.stack([side0 * math.cos(a0), side0 * math.sin(a0)], 1)
        arc = _arc_points(ECT_R["el"], a0, a1, scale, True)
        P1 = np.stack([side0[::-1] * math.cos(a1), side0[::-1] * math.sin(a1)], 1)
        polyline(add(np.concatenate([P0[:-1], arc, P1[1:]])), False)
    pp = np.array(pts)
    segs = np.array(segs, dtype=np.int64)
    # interior candidates in the disk, outside the electrodes
    hmin = 2.5 * scale
    n_cand = int(6.0 * 1.155 * math.pi * ECT_R["screen"] ** 2 / hmin ** 2 / 3.0)
    C = rng.uniform(-ECT_R["screen"], ECT_R["screen"], (n_cand, 2))
    hc = ect_size_field(C, scale)
    keep = (np.linalg.norm(C, axis=1) < ECT_R["screen"] - 0.5 * hc) & ~ect_in_electrode(C) & \
        (rng.random(n_cand) < (hmin / hc) ** 2)
    C, hc = C[keep], hc[keep]
    dseg = _point_seg_dist(C, pp[segs[:, 0]], pp[segs[:, 1]])
    keep = dseg > 0.55 * hc
    C, hc = C[keep], hc[keep]
    order = rng.permutation(len(C))
    C, hc = C[order], hc[order]
    allp = np.concatenate([pp, C])
    nbrs = cKDTree(allp).query_ball_point(C, r=0.8 * hc)
    nf = len(pp)
    acc = np.zeros(len(allp), dtype=bool)
    acc[:nf] = True
    for i, nb in enumerate(nbrs):
        ii = nf + i
        acc[ii] = not any(q != ii and acc[q] for q in nb)
    vol = C[acc[nf:]]
    n_split = 0
    for it in range(40):
        X = np.concatenate([pp, vol])
        F = Delaunay(X).simplices.astype(np.int64)
        sa = _tri_signed_area(X, F)
        F[sa < 0] = F[sa < 0][:, [0, 2, 1]]
        F = F[np.abs(sa) > 1e-9]
        F = F[~ect_in_electrode(X[F].mean(1))]                               # electrodes are holes
        n0 = len(X)
        E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
        ek = np.unique(E[:, 0] * n0 + E[:, 1])
        sk = np.sort(segs, 1)
        miss = ~np.isin(sk[:, 0] * n0 + sk[:, 1], ek)
        if not miss.any():
            break
        A_, B_ = X[segs[miss, 0]], X[segs[miss, 1]]
        mid, rad = 0.5 * (A_ + B_), 0.5 * np.linalg.norm(B_ - A_, axis=1) * 1.05
        bad = np.zeros(len(vol), dtype=bool)
        for m_, r_ in zip(mid, rad):
            bad |= np.linalg.norm(vol - m_, axis=1) < r_
        if bad.any():
            vol = vol[~bad]
            continue
        new_idx = len(pp) + np.arange(int(miss.sum()))
        pp = np.concatenate([pp, mid])
        segs = np.concatenate([segs[~miss], np.stack([segs[miss, 0], new_idx], 1),
                               np.stack([new_idx, segs[miss, 1]], 1)])
        n_split += int(miss.sum())
    else:
        raise RuntimeError("could not recover the constrained segments")
    used = np.unique(F)
    if len(used) != len(X):                                                   # points inside the electrodes
        remap = -np.ones(len(X), dtype=np.int64)
        remap[used] = np.arange(len(used))
        X, F = X[used], remap[F]
    n0 = len(X)
    E = np.sort(np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), 1)
    ek, cnt = np.unique(E[:, 0] * n0 + E[:, 1], return_counts=True)
    bverts = np.unique(np.stack([ek[cnt == 1] // n0, ek[cnt == 1] % n0], 1))
    boundary = np.zeros(n0, dtype=bool)
    boundary[bverts] = True
    r = np.linalg.norm(X, axis=1)
    th = np.arctan2(X[:, 1], X[:, 0])
    electrode = -np.ones(n0, dtype=np.int64)
    for k, (a0, a1) in enumerate(els):
        d = np.angle(np.exp(1j * (th - 0.5 * (a0 + a1))))
        on = boundary & (np.abs(d) <= 0.5 * (a1 - a0) + 1e-9) & (r > ECT_R["wall"] - 1e-6) & (r < ECT_R["el"] + 1e-6)
        electrode[on] = k
    screen = boundary & (np.abs(r - ECT_R["screen"]) < 1e-6)
    if not np.all(screen | (electrode >= 0) | ~boundary):
        raise RuntimeError("boundary vertices that are neither on the screen nor on an electrode")
    rc = np.linalg.norm(X[F].mean(1), axis=1)
    region = np.where(rc < ECT_R["pipe"], ECT_REGION["pipe"],
                      np.where(rc < ECT_R["wall"], ECT_REGION["wall"], ECT_REGION["air"])).astype(np.int8)
    ang = []
    Q = X[F]
    for k in range(3):
        a_, b_ = Q[:, (k + 1) % 3] - Q[:, k], Q[:, (k + 2) % 3] - Q[:, k]
        ang.append(np.degrees(np.arccos(np.clip((a_ * b_).sum(1) / np.linalg.norm(a_, axis=1) /
                                                np.linalg.norm(b_, axis=1), -1, 1))))
    ang = np.stack(ang, 1)
    stats = dict(scale=scale, seed=seed, n0=int(n0), n1=int(len(ek)), n2=int(len(F)), euler=int(n0 - len(ek) + len(F)),
                 recovery_iterations=int(it), segments_split=int(n_split), nonmanifold=int((cnt > 2).sum()),
                 min_angle=float(ang.min()), min_angle_p1=float(np.percentile(ang.min(1), 1)),
                 n_boundary=int(boundary.sum()), n_electrode_vertices=int((electrode >= 0).sum()))
    if stats["euler"] != 1 - ECT_N_EL or stats["nonmanifold"]:
        raise RuntimeError(f"invalid ECT mesh: {stats}")
    return dict(pos=X, faces=F, region=region, boundary=boundary, electrode=electrode, stats=stats)


def draw_params_ect(idx: int, seed: int = ECT_SEED) -> dict:
    """Electrode potentials, flow pattern and space charge of ECT sample ``idx`` (mesh independent)."""
    rng = np.random.default_rng([seed, idx])
    bg = 4 if rng.random() < 0.25 else 2                                      # gas or oil background
    incl = []
    for _ in range(int(rng.integers(1, 7))):
        a = float(rng.uniform(4.0, 22.0))
        b = float(rng.uniform(0.4, 1.0)) * a
        rr = float(np.sqrt(rng.uniform(0.0, 1.0))) * (56.0 - a)
        ph = float(rng.uniform(0, 2 * math.pi))
        mat = (3 if rng.random() < 0.6 else 4) if bg == 2 else (3 if rng.random() < 0.6 else 2)
        incl.append(dict(c=[rr * math.cos(ph), rr * math.sin(ph)], a=a, b=b, th=float(rng.uniform(0, math.pi)),
                         phase=int(mat)))
    blobs = []
    for _ in range(int(rng.integers(0, 3))):
        rr = float(np.sqrt(rng.uniform(0.0, 1.0))) * 45.0
        ph = float(rng.uniform(0, 2 * math.pi))
        blobs.append(dict(c=[rr * math.cos(ph), rr * math.sin(ph)], w=float(rng.uniform(4.0, 12.0)),
                          q=float(rng.uniform(-ECT_Q_MAX, ECT_Q_MAX))))
    return dict(background=bg, inclusions=incl, blobs=blobs,
                potentials=[float(v) for v in rng.uniform(-ECT_V_MAX, ECT_V_MAX, ECT_N_EL)])


def face_phase_ect(mesh: dict, p: dict) -> np.ndarray:
    cent = mesh["pos"][mesh["faces"]].mean(1)
    region = mesh["region"]
    phase = np.where(region == ECT_REGION["air"], 0, np.where(region == ECT_REGION["wall"], 1, p["background"]))
    pipe = region == ECT_REGION["pipe"]
    for e in p["inclusions"]:
        d = cent - np.asarray(e["c"])[None]
        u = d[:, 0] * math.cos(e["th"]) + d[:, 1] * math.sin(e["th"])
        v = -d[:, 0] * math.sin(e["th"]) + d[:, 1] * math.cos(e["th"])
        phase[pipe & ((u / e["a"]) ** 2 + (v / e["b"]) ** 2 < 1.0)] = e["phase"]
    return phase.astype(np.int8)


def charge_ect(X: np.ndarray, p: dict) -> np.ndarray:
    q = np.zeros(len(X))
    for b in p["blobs"]:
        q += b["q"] * np.exp(-((X - np.asarray(b["c"])[None]) ** 2).sum(1) / (2.0 * b["w"] ** 2))
    return q


def solve_ect(mesh: dict, p: dict, factor: float = 1.0) -> dict:
    """P1 FEM in mm units: ``K_ij = sum_f eps_f |f| grad l_i . grad l_j``, consistent load of the face-centroid charge,
    Dirichlet data on the screen (0) and the electrodes (``V_k``); sparse direct solve."""
    X, F = mesh["pos"], mesh["faces"]
    m = P1Mesh(X / MM, F, mesh["boundary"])                                   # P1Mesh works in "SI" = here mm
    phase = face_phase_ect(mesh, p)
    eps = ECT_EPS[phase]
    GG = np.einsum("fid,fjd->fij", m.G, m.G) * m.area[:, None, None]
    Kmat = m.csr(GG * eps[:, None, None])
    q_face = factor * charge_ect(X[F].mean(1), p)
    Fv = m.load(q_face)
    g = np.zeros(m.n0)
    el = mesh["electrode"]
    V = factor * np.asarray(p["potentials"])
    g[el >= 0] = V[el[el >= 0]]
    fr = m.free
    rhs = Fv[fr] - Kmat[fr][:, ~np.isin(np.arange(m.n0), fr)] @ g[~np.isin(np.arange(m.n0), fr)]
    phi = g.copy()
    phi[fr] = spla.spsolve(Kmat[fr][:, fr].tocsc(), rhs)
    res = float(np.linalg.norm((Kmat @ phi - Fv)[fr]) / max(np.linalg.norm(rhs), 1e-300))
    grad = m.grad(phi)
    E_f = -grad
    return dict(phi=phi, E_node=node_field(m, E_f), E_face=E_f, phase=phase, q_face=q_face, V=V, res=res,
                energy=float(0.5 * (m.area * eps * (grad ** 2).sum(1)).sum()))


def _ect_job(args):
    idx, factor = args
    mesh = _W["mesh"]
    p = draw_params_ect(idx)
    sol = solve_ect(mesh, p, factor)
    return dict(idx=idx, phi=sol["phi"], E_node=sol["E_node"].astype(np.float32), phase=sol["phase"],
                q_face=sol["q_face"].astype(np.float32), V=sol["V"], res=sol["res"], energy=sol["energy"],
                emax=float(np.linalg.norm(sol["E_face"], axis=1).max()), params=p)


def _ect_init(mesh: dict) -> None:
    _W["mesh"] = mesh


def cmd_estat(a) -> None:
    """Data sets ``MXE.pt`` (5000 samples), ``MXE_ood15`` (test samples x1.5 potentials and charges), ``MXE_fine`` /
    ``MXE_coarse`` (test samples on the refined / coarsened mesh), with a verification json."""
    import torch
    out_dir = os.path.join(ROOT, "datasets", "v2")
    os.makedirs(out_dir, exist_ok=True)
    meshes = {name: build_mesh_ect(ECT_SCALES[name], 21 + i) for i, name in enumerate(("base", "fine", "coarse"))}
    for name, mesh in meshes.items():
        print(f"ECT mesh {name}: {mesh['stats']}", flush=True)
    ext = list(range(TEST_START, min(TEST_START + N_EXTRA, a.n))) if a.n > TEST_START else list(range(min(a.n, 50)))
    jobs = [("MXE", "base", list(range(a.n)), 1.0), ("MXE_ood15", "base", ext, OOD_FACTOR),
            ("MXE_fine", "fine", ext, 1.0), ("MXE_coarse", "coarse", ext, 1.0)]
    for fname, mname, ids, factor in jobs:
        mesh = meshes[mname]
        t0 = time.time()
        with Pool(a.workers, initializer=_ect_init, initargs=(mesh,)) as pool:
            rows = pool.map(_ect_job, [(i, factor) for i in ids], chunksize=8)
        blob = dict(pos=torch.tensor(mesh["pos"]), faces=torch.tensor(mesh["faces"]),
                    region=torch.tensor(mesh["region"]), boundary=torch.tensor(mesh["boundary"]),
                    electrode=torch.tensor(mesh["electrode"]), sample_ids=torch.tensor([r["idx"] for r in rows]),
                    potentials=torch.tensor(np.stack([r["V"] for r in rows])),
                    phase=torch.tensor(np.stack([r["phase"] for r in rows])),
                    q_face=torch.tensor(np.stack([r["q_face"] for r in rows])),
                    phi=torch.tensor(np.stack([r["phi"] for r in rows])),
                    E_node=torch.tensor(np.stack([r["E_node"] for r in rows])),
                    res=torch.tensor([r["res"] for r in rows]), energy=torch.tensor([r["energy"] for r in rows]),
                    meta=dict(generator="datasets/generators/gen_maxwell.py estat", seed=ECT_SEED, mesh=mname,
                              scale=ECT_SCALES[mname], current_factor=factor, phases=list(ECT_PHASES),
                              eps=ECT_EPS.tolist(), V_max=ECT_V_MAX, q_max=ECT_Q_MAX, mesh_stats=mesh["stats"],
                              units=dict(length="mm", phi="V", E="V/mm", q="V/mm^2 (rho / eps_0)"),
                              params=[r["params"] for r in rows], created=time.strftime("%Y-%m-%d %H:%M:%S")))
        torch.save(blob, os.path.join(out_dir, fname + ".pt"))
        em = np.array([r["emax"] for r in rows])
        summ = dict(file=fname, N=len(rows), mesh=mesh["stats"], res_max=float(max(r["res"] for r in rows)),
                    emax=dict(p10=float(np.percentile(em, 10)), p50=float(np.median(em)), p90=float(np.percentile(em, 90))),
                    seconds=time.time() - t0)
        with open(os.path.join(out_dir, fname + ".json"), "w") as f:
            json.dump(summ, f, indent=1)
        print(json.dumps(summ), flush=True)


def ect_selftest() -> dict:
    """Patch test (homogeneous dielectric, linear Dirichlet data on every boundary vertex: exact), linearity in the
    data, and mesh convergence of the field energy (capacitance-type quantity) for 3 samples."""
    mesh = build_mesh_ect(1.0, 21)
    X = mesh["pos"]
    m = P1Mesh(X / MM, mesh["faces"], mesh["boundary"])
    GG = np.einsum("fid,fjd->fij", m.G, m.G) * m.area[:, None, None]
    Kmat = m.csr(GG * 3.7)
    g = 0.3 * X[:, 0] - 0.7 * X[:, 1] + 0.1
    fr = m.free
    bd = ~np.isin(np.arange(m.n0), fr)
    phi = g.copy()
    phi[fr] = spla.spsolve(Kmat[fr][:, fr].tocsc(), -Kmat[fr][:, bd] @ g[bd])
    out = dict(patch_test_max_err=float(np.abs(phi - g).max()))
    p = draw_params_ect(5)
    s1, s2 = solve_ect(mesh, p, 1.0), solve_ect(mesh, p, 2.0)
    out["linearity_rel"] = float(np.linalg.norm(s2["phi"] - 2 * s1["phi"]) / np.linalg.norm(2 * s1["phi"]))
    conv = []
    for idx in (3, 7, 11):
        p = draw_params_ect(idx)
        row = dict(idx=idx)
        for sc in (2.0, 1.0, 0.5, 0.25):
            sol = solve_ect(build_mesh_ect(sc, 31), p)
            row[str(sc)] = dict(energy=sol["energy"], res=sol["res"])
        conv.append(row)
    out["refinement"] = conv
    return out


def cmd_mesh(a) -> None:
    for name in ("base", "fine", "coarse"):
        mesh = build_mesh(SCALES[name], MESH_SEEDS[name])
        print(name, json.dumps(mesh["stats"]), flush=True)
        if a.plot and name == "base":
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            p = draw_params(3)
            ph = face_phase(mesh, p)
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            for axi, lim in zip(ax, [BOX, (-180, 180, -110, 140)]):
                axi.tripcolor(mesh["pos"][:, 0], mesh["pos"][:, 1], mesh["faces"], facecolors=ph.astype(float),
                              cmap="Pastel1", edgecolors="k", linewidth=0.15)
                axi.set_xlim(lim[0], lim[1])
                axi.set_ylim(lim[2], lim[3])
                axi.set_aspect("equal")
            fig.savefig(os.path.join(ROOT, "runs", "maxwell", "mesh_base.png"), dpi=130, bbox_inches="tight")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("data")
    d.add_argument("--n", type=int, default=N_SAMPLES)
    d.add_argument("--workers", type=int, default=32)
    d.add_argument("--only", default=None, help="comma-separated file names (MXS, MXS_ood15, MXS_fine, MXS_coarse)")
    m = sub.add_parser("mesh")
    m.add_argument("--plot", action="store_true")
    s = sub.add_parser("selftest")
    s.add_argument("--out", default=os.path.join(ROOT, "runs", "maxwell", "selftest.json"))
    s.add_argument("--workers", type=int, default=16)
    e = sub.add_parser("estat", help="secondary family MXE (ECT electrostatics)")
    e.add_argument("--n", type=int, default=N_SAMPLES)
    e.add_argument("--workers", type=int, default=32)
    e.add_argument("--selftest", action="store_true")
    a = p.parse_args(argv)
    if a.cmd == "estat":
        if a.selftest:
            res = ect_selftest()
            os.makedirs(os.path.join(ROOT, "runs", "maxwell"), exist_ok=True)
            with open(os.path.join(ROOT, "runs", "maxwell", "selftest_ect.json"), "w") as f:
                json.dump(res, f, indent=1)
            print(json.dumps(res), flush=True)
        else:
            cmd_estat(a)
    elif a.cmd == "data":
        cmd_data(a)
    elif a.cmd == "mesh":
        os.makedirs(os.path.join(ROOT, "runs", "maxwell"), exist_ok=True)
        cmd_mesh(a)
    else:
        cmd_selftest(a)


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
