# Riemannian Hodge Message Passing (RHMP)

`rhmp` is a PyTorch library for learning physical fields on meshes: scalars, fluxes, gauge connections and pressures
on triangle meshes, curved surfaces, tetrahedral volumes and grids.  The mesh is turned into a cochain complex whose
coboundaries `d_k` (discrete gradient, curl, divergence) are fixed by oriented incidence, and the network learns
symmetric positive-definite cochain metrics `H_k` from local, coordinate-free geometric and material descriptors.
Every propagation step is a metric-weighted Hodge block `d_k^T H_{k+1} d_k` (or its resolvent, solve or projection),
so the differential identities (`d_{k+1} d_k = 0`, nonnegative energies, positive-semidefinite operators, exact
Abelian gauge invariance) hold by construction while geometry, material response and anisotropic coupling are learned.
The metric is invariant under orthogonal changes of the hidden feature basis, which makes the propagation
cochain-frame equivariant.  This repository is the code of *Learning Discrete Riemannian Metrics for Physical Fields
with Cochain-Frame Equivariance* (Zheng & Allen-Blanchette, NeurIPS 2026, spotlight; arXiv:2608.14556): the library,
the seven benchmark tasks, the baselines, the trained models of the paper's tables and the scripts that reproduce them.

## The seven tasks

| task (paper) | physics | task code (RHMP / baselines) | data (`datasets/README.md`) |
|---|---|---|---|
| NS vorticity on a stretched periodic grid | compressible Navier-Stokes fields of PDEBench sampled on a periodic 32x32 tensor-product grid whose spacings vary by a factor 8 along each axis; (rho, v, p) at step t -> vorticity at step t+1; zero-shot grids: a second stretched grid, uniform 32^2 and 64^2 grids, the rotated training grid | `T1gml_mean` / `T1g` | `T1_cns_vorticity.pkl` + `v2/T1_cns_vorticity_graded.pkl` |
| Anisotropic transport on a torus | advection-diffusion `d_t u + div(v u) = div(D grad u)` on a torus with a divergence-free velocity and field-aligned anisotropic diffusion (ratio 1..100); initial field -> field at `T`; zero-shot: faster flows, ratios 300 / 1000, two-step roll-out, finer / coarser tori, rotations | `T2a` | `T2_torus_advection_diffusion.pkl` + `v2/T2v.pt`, `v2/T2a.pt` |
| Ellipsoid surface flow | tangent vector field `v = n x grad(psi)` on an ellipsoid surface from its stream-function source; RHMP predicts the stream function and applies the exact map | `T3g` / `T3` | `T3_ellipsoid_coexact.pkl` |
| Nonlinear anisotropic magnetostatics | 2-D `curl H(B) = J` with grain-oriented and laminated steel (nonlinear, anisotropic constitutive law) in electromagnets and transformers of several layout families; currents and material layout -> vector potential; zero-shot: currents x 1.5, a held-out layout family, finer / coarser meshes, rotations | `MXA` | `v2/MXA*.pt` |
| U(1) Wilson loop | a U(1) connection on the edges of a random Delaunay mesh -> its plaquette curvature; zero-shot: new meshes, anisotropic and finer meshes, gauge transformations, rotations | `T6` | `T6_wilson_loop.pkl` + `v2/T6_native.pt` |
| SU(2) Yang-Mills | an su(2) connection on the edges -> the field strength `F = dA + [A, A]` (node target; `T7f`: the field strength on the faces) | `T7` (`T7f`) | `T7_yang_mills_su2.pkl` + `v2/T7_native.pt` |
| AirfRANS pressure on body-fitted meshes | the AirfRANS RANS simulations re-meshed on conforming near-field meshes (1.9K nodes, one mesh per airfoil); signed distance, inlet speed and angle of attack -> pressure; zero-shot: 0.55K- and 6.9K-node meshes of the same airfoils, rotations | `AFPi0` / `AFP` | `v2/AFM*.pt` (from the AirfRANS data) |

Baselines: GCN, GAT, SchNet, EGNN, MPSN, SCCNN, GaugeEquivCNN, GEM-CNN, CW Net, Clifford-SMPN, DeepONet (not on
AirfRANS) and FNO (NS vorticity only), all trained by the same trainer with their widths matched to the RHMP
parameter budget of the task (+20 % headroom).  Seed 42 gives the tables; seeds 1 and 2 give the cross-seed
statistics.  All runs train for 100 epochs and report `result.json["test100"]`: SSIM, Pearson, NRMSE and R2 on the
first 100 test samples.

## Install

```bash
pip install -e .                 # torch >= 2.4, numpy, scipy (validated with Python 3.12 and torch 2.10, CPU and CUDA)
pip install -e ".[viz,dev]"      # + matplotlib (figures, example 06) and pytest
pip install pyvista h5py airfrans   # data sets: the ellipsoid pickle and the torus / AirfRANS generators (pyvista),
                                    # the NS grid generator (h5py), the AirfRANS download (airfrans)
```

The library, the examples and most tests need no data set.

## Data

```bash
python3 datasets/download_v1.py          # the original pickles of the paper from OSF (1.1 GB) -> datasets/*.pkl
bash scripts/gen_datasets.sh             # the generated sets of the seven tasks -> datasets/v2/ (needs a GPU for T6 / T7)
SRC=/path/to/2D_CFD_Rand_M0.1_Eta0.01_Zeta0.01_periodic_128_Train.hdf5 \
AIRFRANS_ROOT=/path/to/airfrans/Dataset bash scripts/gen_datasets.sh   # + the NS grid (PDEBench) and AirfRANS sets
python3 scripts/transfer_eval/gen_planar.py --task T6   # zero-shot meshes of the gauge-field tasks (also --task T7)
```

[datasets/README.md](datasets/README.md) lists every file with its generator, inputs, seeds and size; every CPU
generator is deterministic given its seed, and the T6 / T7 regeneration reproduces the original samples with the
CUDA random number generator.

## Training the models of the paper

`scripts/train_paper_models.sh` holds the exact trainer arguments of every run in `results/` (they are the `args`
stored in each run's `config.json`):

```bash
bash scripts/train_paper_models.sh rhmp torus          # the RHMP model of one task  (ns torus ellipsoid maxwell wilson yang_mills yang_mills_face airfrans)
bash scripts/train_paper_models.sh baselines torus     # its baselines (or a subset: ... baselines torus gcn gem_cnn)
SEED=1 bash scripts/train_paper_models.sh all          # everything for another seed
DRY=1 bash scripts/train_paper_models.sh all           # print the commands only
```

The RHMP configurations (100 epochs, seed 42; `--out` chooses the run directory):

```bash
# NS vorticity, stretched grid: projection first layer, homogeneous gate, context head with face wedge products (88.8K parameters)
python3 -m rhmp.train --task T1gml_mean --C 102 --layers project,poly,poly,poly --gate homog --cochain-head context \
    --lift-context 16 --lift-vertex --readout-depth 3 --readout-spatial --readout-skip --readout-wedge 32 \
    --readout-hidden 32 --project-solver dense --epochs 100 --seed 42 --out runs/ns_vorticity/rhmp_s42
# torus transport: projection layer, full tensor metric, resolvent third layer, transport term (35.5K)
python3 -m rhmp.train --task T2a --layers project,poly,resolvent,poly --metric-type tensor --tensor-param full \
    --resolvent-iters 12 --metric-ref 1:3 --material 1:2,2:2 --transport '{"inputs":[0],"order":2,"metric":"star"}' \
    --log-range 5.0 --project-solver dense --epochs 100 --seed 42 --out runs/torus_transport/rhmp_s42
# ellipsoid flow: linear lifting and stream-function head (13.3K)
python3 -m rhmp.train --task T3g --lifting linear --readout-head linear --fit-readout --epochs 100 --seed 42 \
    --out runs/ellipsoid_flow/rhmp_s42
# magnetostatics: one physical solve layer with four Picard iterations and a tensor metric, material columns only (2,229 parameters)
python3 -m rhmp.train --task MXA --solver-mode --layers solve --C 4 --metric-type tensor --material 1:7,2:5 \
    --log-range 5.0 --solve-precond twolevel --solve-iters 128 --solve-tol 1e-05 --solve-picard 4 --no-metric-geo \
    --eval-batch 16 --epochs 100 --seed 42 --out runs/maxwell_magnetostatics/rhmp_s42
# Wilson loop: projection first layer (72.9K)
python3 -m rhmp.train --task T6 --layers project,poly,poly,poly --project-solver dense --epochs 100 --seed 42 \
    --out runs/wilson_loop/rhmp_s42
# Yang-Mills: the NS configuration with C = 120 (105.8K); --task T7f for the face target
python3 -m rhmp.train --task T7 --C 120 --layers project,poly,poly,poly --gate homog --cochain-head context \
    --lift-context 16 --lift-vertex --readout-depth 3 --readout-spatial --readout-skip --readout-wedge 32 \
    --readout-hidden 32 --project-solver dense --epochs 100 --seed 42 --out runs/yang_mills/rhmp_s42
# AirfRANS pressure: projection layer with the sparse-LU solver, energy readout, dynamic similarity (88.9K)
python3 -m rhmp.train --task AFPi0 --C 118 --layers project,poly,poly,poly --gate homog --lift-context 16 \
    --lift-vertex --amplitude 1:1 --amplitude-power 2 --amplitude-energy --readout-energy 16 \
    --project-solver direct --epochs 100 --seed 42 --out runs/airfrans_pressure/rhmp_s42
```

Baselines take the node inputs (`--legacy`) and a parameter budget in millions where the task sets one:

```bash
python3 -m rhmp.train --task T1g --legacy --model gem_cnn --param-budget 0.088868 --epochs 100 --seed 42 --out runs/ns_vorticity/gem_cnn_s42
python3 -m rhmp.train --task T2a --legacy --model gat --param-budget 0.0375 --epochs 100 --seed 42 --out runs/torus_transport/gat_s42   # mpsn: 0.036135
python3 -m rhmp.train --task T3  --legacy --model gcn --epochs 100 --seed 42 --out runs/ellipsoid_flow/gcn_s42           # T6, T7: the same form
python3 -m rhmp.train --task MXA --legacy --model mpsn --param-budget 0.091861 --eval-batch 16 --model-opts '{"head":"v1_edge"}' \
    --epochs 100 --seed 42 --out runs/maxwell_magnetostatics/mpsn_s42                                                 # sccnn, clifford_smpn: the same head
python3 -m rhmp.train --task AFP --legacy --model cw_net --param-budget 0.089813 --epochs 100 --seed 42 --out runs/airfrans_pressure/cw_net_s42
```

MPSN, SCCNN and Clifford-SMPN use their original edge protocol on the fixed-mesh tasks (edge-averaged target,
edge-space metrics; `docs/BASELINES.md`).  Training times of the RHMP runs on one RTX PRO 6000 with nothing else running (100 epochs): about
0.3 h (ellipsoid), 1.1 h (NS, Wilson loop, Yang-Mills, AirfRANS), 1.5 h (torus) and 2.5 h (magnetostatics); the
`wall_clock_s` field of a shipped `result.json` is the time of that run on a shared GPU and can be several times longer.
Runs resume from `last.pt` when interrupted and are skipped when their `result.json` is complete.  The Maxwell
ablations and the AirfRANS ablations / controls of the paper's appendix are launched by `scripts/run_maxwell.sh` and
`scripts/run_pressure.sh`; `scripts/run_baselines.sh` trains any baseline on any task.

## Evaluating a checkpoint, zero-shot transfer and rotations

Every RHMP run in `results/` ships its checkpoint (`best.pt`, 6 MB in total).  Re-evaluating one reproduces the
stored `test` / `test100` metrics (to about 1e-7 relative on the same GPU model):

```bash
python3 scripts/eval_checkpoint.py results/torus_transport/rhmp_s42                 # test + test100 vs result.json
python3 scripts/eval_checkpoint.py results/ns_vorticity/rhmp_s42 --extras           # + the zero-shot grids of the run
python3 -m rhmp.train --task T2a --eval-ckpt results/torus_transport/rhmp_s42 --out runs/eval   # the trainer's evaluation path
```

The zero-shot and rotation evaluations of the paper (`results/evaluations/`, README there):

```bash
python3 scripts/eval_frozen_units.py results/ns_vorticity/rhmp_s42 --out runs/ns_zero_shot.json           # NS: new grids, 90-degree rotation
python3 scripts/eval_torus_aniso.py --runs results/torus_transport/rhmp_s42 --out runs/torus_zero_shot.json  # torus: speeds, ratios, roll-out, meshes, rotations
python3 scripts/eval_torus_aniso.py --runs results/torus_transport/rhmp_s42 --frozen-auto --out runs/torus_zero_shot_frozen_auto.json
(cd scripts/transfer_eval && python3 run_eval.py --device cuda --tasks T6,T7 --models rhmp)                 # gauge fields: new meshes, gauge transformations
(cd scripts/frame_eval && python3 run_eval.py --device cuda --tasks T6,T7 --models rhmp)                    # gauge fields: rigid rotations
cp -r results/maxwell_magnetostatics/rhmp_s42 runs/mxa && python3 scripts/maxwell_eval.py runs/mxa          # magnetostatics: currents x 1.5, held-out family, meshes, rotations
cp -r results/airfrans_pressure/rhmp_s42 runs/afp && python3 scripts/airfrans_pressure_eval.py runs/afp     # AirfRANS: 0.55K / 6.9K-node meshes, rotations, wall, lift
```

Each script scores the stored `test100` first and then the transfer sets (the last two write their JSON next to the
run directory, hence the copy); the NS, torus and gauge-field scripts also evaluate the model with the per-mesh
normalisers frozen at the training mesh (`frozen_units`) and with the per-column rule `frozen_descriptors='auto'`
(`docs/MATH.md`, section 4b).  Symmetry tables of any run: `scripts/eval_robustness.py`.

## Results

Generated from `results/` by `python3 scripts/collect_results.py results --paper-tables`; the complete tables
(including R2, parameter counts, the cross-seed means and standard deviations of seeds 42, 1, 2 and an index of all
253 runs) are in [results/RESULTS.md](results/RESULTS.md).  Table 2 of the paper (six fixed-mesh tasks, seed 42,
first 100 test samples; MPSN, SCCNN and Clifford-SMPN scored in edge space):

**SSIM** (higher is better)

| model | NS vorticity | torus transport | ellipsoid flow | magnetostatics | Wilson loop | Yang-Mills |
|---|---:|---:|---:|---:|---:|---:|
| GCN | 0.6169 | 0.7775 | 0.2100 | 0.3560 | 0.7971 | 0.4407 |
| GAT | 0.5908 | 0.8627 | 0.2091 | 0.3924 | 0.9080 | 0.4720 |
| SchNet | 0.6245 | 0.7934 | 0.2090 | 0.3485 | 0.9784 | 0.5039 |
| EGNN | 0.6160 | 0.7797 | 0.2093 | 0.4089 | 0.9564 | 0.4321 |
| MPSN | 0.6167 | 0.7664 | 0.1523 | 0.4353 | 0.7896 | 0.5183 |
| SCCNN | 0.6102 | 0.7627 | 0.1562 | 0.4706 | 0.8558 | 0.5191 |
| GaugeEquivCNN | 0.9440 | 0.9396 | 0.9684 | 0.6060 | 0.9947 | 0.8262 |
| GEM-CNN | 0.9659 | 0.9605 | 0.9716 | 0.6100 | 0.9982 | 0.8680 |
| CW Net | 0.9212 | 0.9323 | 0.4827 | 0.5483 | 0.9745 | 0.6843 |
| Clifford-SMPN | 0.7371 | 0.8930 | 0.1669 | 0.4464 | 0.9090 | 0.5136 |
| DeepONet | 0.6248 | 0.2886 | 0.2422 | 0.3479 | 0.8417 | 0.4055 |
| FNO | 0.8717 | - | - | - | - | - |
| RHMP | **0.9944** | **0.9900** | **0.999998** | **0.9787** | **0.999940** | **0.999992** |

**Pearson** (higher is better)

| model | NS vorticity | torus transport | ellipsoid flow | magnetostatics | Wilson loop | Yang-Mills |
|---|---:|---:|---:|---:|---:|---:|
| GCN | 0.1591 | 0.7848 | 0.0196 | 0.1353 | 0.1112 | 0.1855 |
| GAT | 0.0949 | 0.8977 | 0.0334 | 0.2325 | 0.5708 | 0.2623 |
| SchNet | 0.0818 | 0.7842 | 0.0450 | 0.0468 | 0.8621 | 0.3281 |
| EGNN | 0.0920 | 0.7889 | 0.0459 | 0.2662 | 0.7647 | 0.1608 |
| MPSN | 0.1339 | 0.7865 | 0.0457 | 0.3113 | 0.2709 | 0.3371 |
| SCCNN | 0.0281 | 0.7850 | 0.0748 | 0.3723 | 0.4445 | 0.3385 |
| GaugeEquivCNN | 0.9003 | 0.9588 | 0.9741 | 0.5994 | 0.9485 | 0.7775 |
| GEM-CNN | 0.9409 | 0.9720 | 0.9740 | 0.6008 | 0.9756 | 0.8295 |
| CW Net | 0.8802 | 0.9341 | 0.5032 | 0.5098 | 0.8428 | 0.5965 |
| Clifford-SMPN | 0.5476 | 0.8942 | 0.0915 | 0.2880 | 0.6401 | 0.3581 |
| DeepONet | 0.2262 | 0.1309 | 0.1893 | 0.1217 | 0.2858 | 0.0284 |
| FNO | 0.8828 | - | - | - | - | - |
| RHMP | **0.9911** | **0.9909** | **0.999997** | **0.9893** | **0.999912** | **0.999990** |

**NRMSE** (lower is better)

| model | NS vorticity | torus transport | ellipsoid flow | magnetostatics | Wilson loop | Yang-Mills |
|---|---:|---:|---:|---:|---:|---:|
| GCN | 0.0350 | 0.0360 | 0.0613 | 0.0640 | 0.0149 | 0.0355 |
| GAT | 0.0356 | 0.0260 | 0.0614 | 0.0629 | 0.0102 | 0.0348 |
| SchNet | 0.0349 | 0.0360 | 0.0614 | 0.0642 | 0.0054 | 0.0340 |
| EGNN | 0.0350 | 0.0358 | 0.0614 | 0.0624 | 0.0072 | 0.0356 |
| MPSN | 0.0336 | 0.0362 | 0.0698 | 0.0579 | 0.0159 | 0.0331 |
| SCCNN | 0.0336 | 0.0361 | 0.0697 | 0.0559 | 0.0135 | 0.0330 |
| GaugeEquivCNN | 0.0126 | 0.0161 | 0.0141 | 0.0519 | 0.0027 | 0.0227 |
| GEM-CNN | 0.0096 | 0.0133 | 0.0139 | 0.0519 | 0.0016 | 0.0202 |
| CW Net | 0.0140 | 0.0207 | 0.0532 | 0.0561 | 0.0056 | 0.0290 |
| Clifford-SMPN | 0.0271 | 0.0258 | 0.0696 | 0.0576 | 0.0107 | 0.0331 |
| DeepONet | 0.0349 | 0.0628 | 0.0602 | 0.0642 | 0.0138 | 0.0362 |
| FNO | 0.0227 | - | - | - | - | - |
| RHMP | **0.0054** | **0.0076** | **1.45e-04** | **0.0150** | **8.93e-05** | **1.67e-04** |

Table 3 of the paper (AirfRANS pressure, seed 42, 100 test simulations; metrics at the 1.9K-node training size, R2
without retraining on the 0.55K- and 6.9K-node meshes of the same simulations):

| model | params | SSIM | Pearson | NRMSE | R2 | R2 (0.55K) | R2 (6.9K) |
|---|---:|---:|---:|---:|---:|---:|---:|
| GCN | 90K | 0.6891 | 0.1642 | 0.0227 | 0.0282 | 0.0268 | 0.0238 |
| GAT | 92K | 0.6174 | 0.1705 | 0.0233 | -0.0231 | -0.0401 | -0.0107 |
| SchNet | 96K | 0.6730 | 0.2327 | 0.0228 | 0.0229 | 0.0115 | 0.0204 |
| EGNN | 100K | 0.6975 | 0.2517 | 0.0226 | 0.0360 | 0.0170 | -0.0010 |
| MPSN | 96K | 0.6756 | 0.1492 | 0.0229 | 0.0152 | 0.0166 | 0.0130 |
| SCCNN | 94K | 0.6893 | 0.1652 | 0.0227 | 0.0297 | 0.0284 | 0.0266 |
| GaugeEquivCNN | 91K | 0.9213 | 0.8570 | 0.0098 | 0.8199 | 0.7568 | 0.6877 |
| GEM-CNN | 93K | 0.9468 | 0.9030 | 0.0077 | 0.8897 | 0.7277 | 0.6774 |
| CW Net | 99K | 0.6934 | 0.2195 | 0.0226 | 0.0407 | 0.0231 | -0.0148 |
| Clifford-SMPN | 96K | 0.6838 | 0.1822 | 0.0228 | 0.0265 | 0.0164 | 0.0201 |
| RHMP | 89K | **0.9983** | **0.9967** | **0.0016** | **0.9950** | **0.9688** | **0.9817** |

## Quickstart

```python
import numpy as np
import torch
from scipy.spatial import Delaunay
from rhmp import CochainComplex, RHMP, RHMPConfig

pts = np.random.default_rng(0).random((200, 2))                      # any triangle mesh (numpy or torch)
K = CochainComplex.from_triangles(pts, Delaunay(pts).simplices)      # exact d_k, reference stars, geometry
print(K)                                                             # cells per degree, star, device

cfg = RHMPConfig(in_dims={0: 1, 1: 1}, even_dims={1: 1},             # a vertex field + an even edge coefficient
                 C=32, n_layers=3, readout="node_scalar", out_dim=1)
model = RHMP(cfg, K.geo_dims)

B = 4                                                                # samples sharing the mesh
inputs = {0: torch.randn(K.n[0], B, 1), 1: torch.randn(K.n[1], B, 1)}   # {degree: (n_k, B, F_k)}
y = model(inputs, K)                                                 # (n_0, B, 1)

opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss = torch.nn.functional.mse_loss(y, torch.randn_like(y))          # one training step on a dummy target
opt.zero_grad(); loss.backward(); opt.step()
print(tuple(y.shape), f"loss {loss.item():.3f}", f"{model.num_parameters()} parameters")
```

A trained model is loaded with `RHMP.from_checkpoint("results/wilson_loop/rhmp_s42/best.pt", map_location="cpu")`
and applied to any mesh in the same way; `K.check_d2()` returns the (round-off) size of `d_1 d_0`.
Inputs and outputs are dictionaries keyed by degree (0 vertices, 1 edges, 2 faces, 3 cells).  Layer types are
`poly` (polynomial filters of the normalised Hodge operators), `resolvent` (a learned Green's function through a
batched conjugate-gradient solve), `solve` (a physical solve with Dirichlet / Neumann data, optionally with Picard
iterations for state-dependent metrics) and `project` (a metric Hodge projection of the edge features with a learned
harmonic correction; solvers `cg`, `direct` = sparse LU, `dense` = batched Cholesky); metrics are diagonal or full
tensors per cell.  [docs/TUTORIAL.md](docs/TUTORIAL.md) walks through the library, [docs/MATH.md](docs/MATH.md)
states the mathematics as implemented with the test of every guarantee, [docs/API.md](docs/API.md) (generated by
`docs/gen_api.py`) documents every public function, and `examples/` holds six runnable CPU examples (complex
construction, the DEC toolkit, training on heterogeneous diffusion, a custom task, variable meshes, reading out the
learned metric).

## Repository layout

| path | contents |
|---|---|
| `rhmp/` | the package: cochain complexes (`complex.py`), geometry, sparse operators, DEC toolkit (`dec.py`), metrics, layers (`layers.py`), transport term, lifting, readouts, the model (`model.py`), the trainer (`train.py`), tasks (`tasks/`: the paper tasks, `torus.py`, `maxwell.py`, `airfrans.py`, the synthetic / extension / anisotropy suites) and baselines (`baselines/`, with the original v1 implementation vendored in `baselines/v1/`) |
| `tests/` | the test suite |
| `examples/` | six runnable examples |
| `docs/` | TUTORIAL, MATH, THEORY, API (generated), DESIGN, TASK_SUITE, TASK_SUITE_DETAILS, ANISO_TASKS, BASELINES, DATASETS |
| `datasets/` | the downloader of the original pickles, the generators (`generators/`), [README](datasets/README.md) |
| `scripts/` | `train_paper_models.sh`, `eval_checkpoint.py`, the zero-shot / rotation evaluations (`eval_frozen_units.py`, `eval_torus_aniso.py`, `transfer_eval/`, `frame_eval/`, `maxwell_eval.py`, `airfrans_pressure_eval.py`), the Maxwell and AirfRANS study launchers (`run_maxwell.sh`, `run_pressure.sh`), `collect_results.py`, `sanitize_results.py`, data generation (`gen_datasets.sh`), and the robustness / transfer / metric-recovery tools of the library (`eval_robustness.py`, `eval_on.py`, `metric_recovery.py`, `t6_mesh_transfer.py`, `run_aniso.sh`, `run_baselines.sh`); every script starts with a comment that says what it reproduces |
| `results/` | the 253 runs of the paper (`<task>/<model>_s<seed>/result.json`, `config.json`; RHMP checkpoints), the evaluation files (`evaluations/`) and [RESULTS.md](results/RESULTS.md) |

## Tests

```bash
python3 -m pytest tests -q               # CPU; the CUDA tests run when a GPU is visible (about 7 minutes on an idle GPU)
```

The suite checks the exact structure (`d_{k+1} d_k = 0`, orientation and relabelling invariance, gauge invariance,
operator norms, batch independence), the numerics of every layer type and solver, the metrics against the original
evaluation code, checkpoint round trips, the trainer, the task loaders and the baselines.  Tests that need a data
set are skipped when the file is absent.

## Citation and licence

```
Zheng and Allen-Blanchette, "Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame
Equivariance", Advances in Neural Information Processing Systems (NeurIPS), 2026 (spotlight).  arXiv:2608.14556.
```

Metadata in [CITATION.cff](CITATION.cff).  Licence: see [LICENSE_NOTE.md](LICENSE_NOTE.md).  The vendored v1 code in
`rhmp/baselines/v1/` comes from the original repository, https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing.
