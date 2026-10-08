# Data sets

The data sets are **not** part of the repository.  The loaders (`rhmp.tasks.load_task`, `python -m rhmp.train --task
NAME`) look for them under `datasets/` of the checkout, under `$RHMP_DATA_ROOT` (the repository root or the datasets
directory itself), or under the `--root` / `root=` you pass.

```
datasets/
  README.md            this file
  download_v1.py       downloads the original pickles of the paper (and optionally the v1 checkpoints) from OSF
  generators/          seeded generators of the derived sets (numpy / scipy; T6 / T7 native also need torch + a GPU)
  *.pkl                original data sets (downloaded; git-ignored)
  v2/*.pt, v2/*.pkl    generated data sets (git-ignored), each .pt with a .json summary
  v2/transfer/*.pt     zero-shot test meshes of the gauge-field tasks (generated; git-ignored)
```

Every data set of the paper is obtained in one of two ways: the original pickles are downloaded (section 1), the
derived sets are generated from them, from PDEBench or from AirfRANS by the scripts in `generators/` (section 2).
`bash scripts/gen_datasets.sh` runs the generators of section 2 in order.

## 1. Original data sets (`datasets/*.pkl`)

These are the data sets released with the original code of the paper (version 1,
https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing), hosted on
[OSF](https://osf.io/35gcm/?view_only=0df302ed83154a8ba6bb1f8d8a1983ec) (1.1 GB archive):

```bash
python3 datasets/download_v1.py            # datasets/*.pkl
python3 datasets/download_v1.py --ckpt     # + the v1 checkpoints into checkpoints_v1/ (only for rhmp.train --eval-v1)
```

All shared-mesh sets are dicts with `points (n0, D)`, `faces (n2, 3)`, `X_data (N, n0, F)` (or `(N, n0)`) and
`Y_data (N, n0, O)`; the protocol is a sequential 70/15/15 split with per-feature statistics of the training part.

| file | size | points | faces | X_data | Y_data | content |
|---|---:|---|---|---|---|---|
| `T1_cns_vorticity.pkl` | 205 MB | (1024, 2) | (1922, 3) | (10000, 1024, 4) | (10000, 1024, 1) | regular 32x32 grid; (rho, Vx, Vy, p)_t -> vorticity_{t+1}; PDEBench compressible NS |
| `T2_torus_advection_diffusion.pkl` | 41 MB | (1711, 3) | (3422, 3) | (3000, 1711) | (3000, 1711) | scalar transport on a torus; the per-sample velocity is not stored (`v2/T2v.pt` restores it) |
| `T3_ellipsoid_coexact.pkl` | 185 MB | (1157, 3) | (2310, 3) | (10000, 1157, 1) | (10000, 1157, 3) | `v = n x grad(psi)`: tangent field from a stream function |
| `T5_maxwell_poisson.pkl` | 62 MB | (1024, 3) | (2027, 3) | (5000, 1024, 1) | (5000, 1024, 2) | charge density -> electric field at vertices (not used by the camera-ready tables) |
| `T6_wilson_loop.pkl` | 164 MB | (1024, 3) | (2027, 3) | (10000, 1024, 3) | (10000, 1024, 1) | U(1): node encoding (avg, avg dx, avg dy) of the edge angle; plaquette flux averaged to vertices |
| `T7_yang_mills_su2.pkl` | 492 MB | (1024, 3) | (2027, 3) | (10000, 1024, 9) | (10000, 1024, 3) | SU(2): `F = dA + [A, A]`, 3 components at vertices |
| `T8_airfoil_pressure_persample.pkl` | 208 MB | list of 1000 dicts | | | | `pts_3d (n, 3)`, `faces (m, 3)`, `x_pres (n, F)`, `y_pres (n, 1)`; about 2000 nodes each; AirfRANS |

The OSF archive contains these seven sets.  Two further files of the original repository, `T6_100K_wilson_loop.pkl`
(405 MB, the 100K-face scaling study) and `T7_small_yang_mills.pkl` (61 MB, the metric-parameterisation sweep), are
produced by the generators `gen_T6_100K_wilson_loop.py` and `gen_T7_small_yang_mills.py` of that repository.  The
third coordinate of the planar meshes (T5-T8) is 0.  T1 is derived from [PDEBench](https://github.com/pdebench/PDEBench)
(Takamoto et al., NeurIPS 2022) and T8 from [AirfRANS](https://github.com/Extrality/AirfRANS) (Bonnet et al., NeurIPS 2022).
The T3 pickle contains pyvista objects next to the numpy arrays: `pip install pyvista` before loading it (the loader
only uses the arrays).

## 2. Generated data sets (`datasets/v2/`)

Generated with the seeded scripts in `datasets/generators/` (outputs go to `datasets/v2/` by default; `--out` /
`--out-dir` to change it).  Every CPU generator is deterministic given its seed (`numpy.random.default_rng` /
`RandomState` with fixed seeds, float64 arithmetic, `scipy.sparse` direct solves); the summaries next to the files
(`*.json`) record the seeds, the solver residuals and the consistency checks.  The two GPU generators replay the
CUDA random streams of the original T6 / T7 generators and reproduce the original samples exactly (checked against
the stored node encodings); on a CPU, or on a GPU whose kernel launch configuration differs from the original one for
the first large draw of T7, the draws differ (the T6 draws do not depend on the device).

### The seven tasks of the paper

| task (code) | files | generator and command | needs | size |
|---|---|---|---|---:|
| NS vorticity on a stretched periodic grid (`T1gml_mean`; baselines `T1g`) | `T1_cns_vorticity_graded.pkl`, `T1_cns_vorticity_graded_val.pkl` | `python3 datasets/generators/gen_T1_graded.py --src <PDEBench file>` and `... --zero-shot-samples val` (about 45 s each) | `T1_cns_vorticity.pkl`, `h5py` and the PDEBench file `2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5` (the simulation T1 was sampled from) | 401 MB + 206 MB |
| Anisotropic transport on a torus (`T2a`) | `T2v.pt`, `T2a.pt`, `T2a_valmesh.pt` | `python3 datasets/generators/gen_T2v.py` (CPU, about a minute), `python3 datasets/generators/gen_T2a.py --workers 16` (30 s), `python3 datasets/generators/gen_T2a.py --val-meshes datasets/v2/T2a_valmesh.pt` (20 s) | `T2_torus_advection_diffusion.pkl`; `pyvista` for the finer / coarser tori | 123 MB, 521 MB, 176 MB |
| Ellipsoid surface flow (`T3g`; baselines `T3`) | none (the original pickle) | | `T3_ellipsoid_coexact.pkl`, `pyvista` | |
| Nonlinear anisotropic magnetostatics (`MXA`) | `MXA.pt`, `MXA_ood15.pt`, `MXA_fine.pt`, `MXA_coarse.pt`, `MXA_heldout.pt` | `python3 datasets/generators/gen_maxwell_aniso.py data --workers 32` (CPU, about 25 s with 40 workers); `... selftest` verifies the constitutive law and the solver | nothing (self-contained) | 300, 30, 118, 8, 30 MB |
| U(1) Wilson loop (`T6`) | `T6_native.pt` | `python3 datasets/generators/gen_T6_native.py` (CUDA GPU) | `T6_wilson_loop.pkl` | 194 MB |
| SU(2) Yang-Mills (`T7`, face target `T7f`) | `T7_native.pt` | `python3 datasets/generators/gen_T7_native.py` (CUDA GPU) | `T7_yang_mills_su2.pkl` | 581 MB |
| AirfRANS pressure on body-fitted meshes (`AFPi0`; baselines `AFP`) | `AFM.pt`, `AFM_fine.pt`, `AFM_coarse.pt` | `AIRFRANS_ROOT=/path/to/airfrans/Dataset python3 datasets/generators/gen_airfrans.py mesh --workers 32` (CPU, a few minutes) | the AirfRANS simulations (`pip install airfrans`, `airfrans.dataset.download(root, unzip=True)`: the `Dataset` directory), `pyvista` | 364, 406, 31 MB |

What the generated sets contain:

* **`T1_cns_vorticity_graded.pkl`** (`gen_T1_graded.py`): the 500 PDEBench trajectories x 20 steps of T1 (same sample
  order and split: 7000 / 1500 / 1500 by trajectory) sampled exactly at the nodes of non-uniform 32 x 32 periodic
  tensor-product grids that are subsets of the 128 x 128 simulation grid (spacings of 1, 2, 4 and 8 fine cells,
  neighbour ratio at most 2, cell aspect ratios 1/8 to 8); the target is the vorticity of the next step computed
  spectrally on the fine grid and sampled at the nodes.  Keys: `grids` (`G1` the training grid, `G2` a second
  stretched grid, `U32` / `U64` uniform grids with 32^2 / 64^2 nodes, `G1rot` = `G1` rotated by 90 degrees), `train`
  (all 10000 samples on `G1`) and `zero_shot` (the 1500 test samples on every other grid).  The `_val` file holds the
  zero-shot grids for the 1500 validation samples (for choices that must not look at the test trajectories).
* **`T2v.pt`** (`gen_T2v.py`): the per-sample velocity `v = alpha vt + beta vp` of the original T2 generator,
  reproduced from its random stream (the stored T2 targets are re-simulated to 2.4e-7), plus a velocity-extrapolation
  sweep and 20-step references of the test samples.
* **`T2a.pt`** (`gen_T2a.py`): anisotropic field-aligned transport `d_t u + div(v u) = div(D grad u)` on the T2 torus
  (3000 samples, the T2 split): divergence-free velocities, helical field directions, anisotropy ratios 1..100, exact
  finite-volume / DEC solves (mass conserved to 1e-14).  For the 450 test samples the zero-shot sets `alpha` (toroidal
  speed beyond the training range), `ratio` (anisotropy ratio 300 and 1000), `rollout` (`u(2T)`), `fine` and `coarse`
  (the same instances on tori with 6844 and 450 vertices).  `T2a_valmesh.pt`: the 450 validation instances on the
  fine and coarse tori.
* **`MXA*.pt`** (`gen_maxwell_aniso.py`): 2-D nonlinear magnetostatics `curl H(B) = J` with grain-oriented and
  laminated (anisotropic) steel in electromagnets and transformers of several layout families on one fixed lattice
  mesh, solved by P1 finite elements and damped Newton on a convex energy (5000 samples, split 3500 / 750 / 750,
  test100 = samples 4250..4349).  `_ood15`: coil currents x 1.5; `_fine` / `_coarse`: the test samples re-solved on
  meshes with half / twice the spacing (7255 / 459 nodes); `_heldout`: 500 samples of a layout family absent from
  training.
* **`T6_native.pt`, `T7_native.pt`** (`gen_T6_native.py`, `gen_T7_native.py`): the raw edge cochains of the
  original samples (the U(1) connection per edge and the plaquette flux per face; the su(2) connection and
  `F = dA + [A, A]`), which the original pickles do not store.  The node encodings recomputed from them reproduce the
  stored `X_data` / `Y_data` to 1e-7.
* **`AFM*.pt`** (`gen_airfrans.py mesh`): the 1000 AirfRANS simulations re-meshed on conforming (body-fitted)
  near-field meshes with about 1.9K nodes, with the signed distance, inlet velocity, angle of attack and the CFD
  pressure and velocity at the nodes; `AFM_fine.pt` / `AFM_coarse.pt`: the 300 test simulations with the mesh size
  halved / doubled (about 6.9K / 0.55K nodes).  Simulation order and split as the original T8 task (train 0..699,
  the loaders hold out 630..699 for validation, test 700..999, test100 = 850..949).
* **`transfer/T6_*.pt`, `transfer/T7_*.pt`** (`scripts/transfer_eval/gen_planar.py --task T6|T7`, CUDA GPU): the 100
  test samples of the gauge-field tasks evaluated on new random Delaunay meshes (another 1024-node mesh, its 3:1
  anisotropic re-triangulation, a 4096-node mesh, further mesh realisations and a graded mesh) for the zero-shot
  evaluation of `scripts/transfer_eval/run_eval.py` (232 MB).

### Further data sets of the library

The task suite of the library goes beyond the paper (`docs/TASK_SUITE.md`, `docs/TASK_SUITE_DETAILS.md`,
`docs/ANISO_TASKS.md`); `EXTRA=1`, `SUITE=1` and `ANISO=1` of `scripts/gen_datasets.sh` generate these sets:

| file(s) | generator | samples | size | tasks (`rhmp.tasks`) |
|---|---|---|---|---|
| `T1_cns_vorticity_blockavg.pkl` | `gen_T1_blockavg.py --src <PDEBench file>` (4 x 4 block averages instead of strided subsampling) | 10000 | 195 MB | `T1pf`, `T1pmf` |
| `HP_k{10,100,1000}.pt` (+ `_fine`) | `gen_HP.py --kappa K --n 5000 --n-fine 500` | 5000 (+ 500 at 4x resolution) | 0.62 GB (+ 0.25 GB) | `HP[flux\|fluxd\|fluxfem\|grad]_k<K>` |
| `HP_k10000.pt` | `gen_HP.py --kappa 10000 --n 1000 --n-fine 0` | 1000 | 0.12 GB | `HP_k10000` |
| `HP_k100_aniso.pt`, `HP_k100_aniso{10,100}.pt` (+ `_fine`) | `gen_HP.py --kappa 100 --aniso` / `--aniso-max R` | 5000 (+ 500) | 0.70-1.05 GB | `HP_k100_aniso[<R>]` |
| `TET_k100.pt` (+ `_fine`) | `gen_TET.py --kappa 100 --n 3000 --n-fine 200` | 3000 (+ 200) | 1.87 GB (+ 0.54 GB) | `TET[flux\|fluxfem\|grad]_k100` |
| `SURF.pt`, `SURF_geo.pt`, `SURF_topo.pt` | `gen_surf.py` | 5700 / 500 / 500 closed surfaces | 683 / 60 / 112 MB | `SURF`, `SURF_heat` (+ geometry / topology transfer) |
| `DYN.pt`, `DYNfix.pt` | `gen_dyn.py` | 600 / 500 trajectories x 101 states | 423 / 327 MB | `DYN*`, `DYNfix*` |
| `HP_qual_graded.pt`, `HP_qual_sliver.pt` | `gen_qual.py` (needs `HP_k100.pt`) | 100 test instances x 7 / 6 mesh levels | 92 / 79 MB | `HP_qual_*` |
| `AHP_r{10,100}.pt` (+ `_fine`) | `gen_aniso.py --tasks AHP` | 5000 (+ 500) | 0.74 GB (+ 0.30 GB) | `AHP_r<R>` |
| `ASURF_r{10,100}.pt` (+ `_fine`) | `gen_aniso.py --tasks ASURF` | 4000 (+ 400) | 1.34 GB (+ 0.53 GB) | `ASURF[_heat]_r<R>` |
| `ACURL_r{10,100}.pt` (+ `_fine`) | `gen_aniso.py --tasks ACURL` | 3000 (+ 200) | 3.16 GB (+ 0.91 GB) | `ACURL[b]_r<R>[_n1500]` |
| `ADARCY_r{10,100}.pt` (+ `_fine`) | `gen_aniso.py --tasks ADARCY` | 3000 (+ 200) | 3.61 GB (+ 1.07 GB) | `ADARCY[p]_r<R>[_n1500]` |
| `T8res_n{500,2000,8000}.pt` | `gen_airfrans.py t8res` | 300 test simulations | 7 / 29 / 115 MB | `T8[fr\|xy\|frxy]_n<N>` (test only) |
| `AFD.pt`, `AFD_fine.pt` | `gen_airfrans.py diffusion` (needs `AFM*.pt`) | 1000 / 300 | small | `AFD` |
| `MXS*.pt`, `MXE*.pt` | `gen_maxwell.py data` / `gen_maxwell.py estat` | 5000 / 500 / 500 / 500 | 189 / 19 / 72 / 5 MB; 230 / 23 / 86 / 7 MB | `MXS` (isotropic magnetostatics), `MXE` (electrostatics) |

The AirfRANS task family (`rhmp/tasks/airfrans.py`) also defines, on the `AFM` meshes and without new files, the
pressure tasks `AFP` (node inputs (sdf, U, a) -> p), `AFPxy` (+ chord-frame coordinates), the covariant variants
`AFPi` / `AFPi0` / `AFPiU` / `AFPiR` / `AFPs` (inflow as an edge 1-form, dynamic similarity through the output map),
the pressure-coefficient controls `AFPcp` / `AFPcpxy`, the potential-flow control `AFPpf` and `T8i` (the `AFPi` inputs
on the original T8 meshes); see the module docstring, `scripts/run_pressure.sh` and `docs/MATH.md` section 9c.

Equations, discretisations, splits and protocols: [docs/TASK_SUITE.md](../docs/TASK_SUITE.md),
[docs/TASK_SUITE_DETAILS.md](../docs/TASK_SUITE_DETAILS.md), [docs/ANISO_TASKS.md](../docs/ANISO_TASKS.md) and
[docs/DATASETS.md](../docs/DATASETS.md) (file formats).  Variable-mesh sets are packed as flat concatenations with
offsets `ptr0..ptr3` (float32 fields, int32 cells); every target is an exact sparse FEM / DEC / FEEC solve (residuals
at round-off, recorded in the `.json` summaries).

Host memory: HP/TET-size sets are built on the CPU (`OMP_NUM_THREADS=1` speeds up the many small operations on a busy
machine).  Tensor-metric runs keep the Whitney blocks of every complex (a 2.5K-node tetrahedral complex is about
45 MB with them), so the 3000-sample tetrahedral sets need about 140 GB of host RAM; the registered `_n1500` subsets
need about 70 GB.
