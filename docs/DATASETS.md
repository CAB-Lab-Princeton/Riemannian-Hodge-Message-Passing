# Dataset file formats

[datasets/README.md](../datasets/README.md) describes where the files come from and how to regenerate them.  The
loader is `rhmp.tasks.load_task(name, root, ...)` (paper tasks in `rhmp/tasks/paper.py`, HP / TET in `rhmp/tasks/synthetic.py`,
SURF / DYN / QUAL in `rhmp/tasks/suite.py`, the anisotropy suite in `rhmp/tasks/aniso.py`).

## Original data sets of the paper (`datasets/*.pkl`, released with the v1 code)

Shared-mesh sets are dicts with `points (n0, D)`, `faces (n2, 3)`, `X_data (N, n0, F)` (or `(N, n0)`) and
`Y_data (N, n0, O)`.  The v1 protocol uses a sequential 70/15/15 split and the per-feature mean / std of the training
part (`X[:nt].mean((0, 1))`, `.std((0, 1)).clamp(1e-6)`).

| file | points | faces | X_data | Y_data | notes |
|---|---|---|---|---|---|
| T1_cns_vorticity.pkl | (1024, 2) f64 | (1922, 3) | (10000, 1024, 4) | (10000, 1024, 1) | regular 32x32 grid triangulated; (rho, Vx, Vy, p)_t -> vorticity_{t+1}; PDEBench CNS |
| T2_torus_advection_diffusion.pkl | (1711, 3) | (3422, 3) | (3000, 1711) | (3000, 1711) | scalar transport on a torus; the per-sample velocity is not stored (see `v2/T2v.pt`) |
| T3_ellipsoid_coexact.pkl | (1157, 3) | (2310, 3) | (10000, 1157, 1) | (10000, 1157, 3) | v = n x grad(psi): tangent vector field from a stream function (node_vector readout) |
| T5_maxwell_poisson.pkl | (1024, 3) f64 | (2027, 3) | (5000, 1024, 1) | (5000, 1024, 2) | charge density -> E (2-D vector at nodes); v1 treated the 2 components as scalars |
| T6_wilson_loop.pkl | (1024, 3) f64 | (2027, 3) | (10000, 1024, 3) | (10000, 1024, 1) | U(1): X = node encoding (avg, avg dx, avg dy) of the edge angle theta; Y = plaquette flux averaged to nodes |
| T6_100K_wilson_loop.pkl | (50010, 3) | (99982, 3) | (500, 50010, 3) | (500, 50010, 1) | same physics, 100K faces |
| T7_yang_mills_su2.pkl | (1024, 3) f64 | (2027, 3) | (10000, 1024, 9) | (10000, 1024, 3) | SU(2): 3 channels x (avg, avg dx, avg dy); F = dA + [A, A] (3 components at nodes) |
| T7_small_yang_mills.pkl | (256, 3) | (494, 3) | (5000, 256, 9) | (5000, 256, 3) | small variant used for the metric-parameterisation sweep of the paper |
| T8_airfoil_pressure_persample.pkl | list of 1000 dicts | | | | keys `pts_3d (n, 3)`, `faces (m, 3)`, `x_pres (n, F)`, `y_pres (n, 1)`; about 2000 nodes each; AirfRANS; split 70/30 (first 700 train), no validation split in v1 |

T1 consists of 500 PDEBench trajectories x 20 consecutive steps (sample `20 i + t`, `t = 0..19`; the target of sample
`s` is exactly the periodic central-difference curl, `dx = 1/32`, of the velocity of sample `s + 1` within a
trajectory), strided subsampling 128 -> 32 of a periodic simulation.  Tasks `T1p` / `T1pm` put the same data on the
periodic grid (`CochainComplex.from_grid((32, 32), 1/32, periodic=True)`; vertex order unchanged); `T1pf` / `T1pmf`
use `datasets/v2/T1_cns_vorticity_blockavg.pkl` (same trajectories, 4 x 4 block averages instead of strided
subsampling; `datasets/generators/gen_T1_blockavg.py`, about 30 s from the PDEBench HDF5 file).

`T1g` / `T1gml` use `datasets/v2/T1_cns_vorticity_graded.pkl` (`datasets/generators/gen_T1_graded.py`, about 45 s):
the same trajectories sampled exactly at the nodes of graded, anisotropic periodic 32 x 32 grids (subsets of the
128 x 128 simulation points; spacings of 1, 2, 4 and 8 fine cells, neighbour ratio <= 2, cell aspect ratios 1/8 to 8),
target = the vorticity at `t + 1` computed spectrally on the 128 x 128 grid and sampled at the nodes (the coarse T1
stencil differs from it by 6.6 % in relative L2).  Keys: `grids` (`G1` training grid, `G2` second graded grid, `U32`,
`U64`, `G1rot` = `G1` rotated by 90 degrees; per grid fine indices, spacings, points, quads, triangles, node spacings),
`train` (`G1`, all 10000 samples) and `zero_shot` (the 1500 test samples on every other grid).
`datasets/v2/T1_cns_vorticity_graded_val.pkl` (`gen_T1_graded.py --zero-shot-samples val`) holds the same zero-shot
grids for the 1500 validation samples (trajectories 350 .. 424), for choices that must not look at the test
trajectories (`scripts/eval_frozen_units.py`).  Node order on every grid: node `i * n_y + j` is lattice point
`(kx[i], ky[j])` of the 128 x 128 simulation grid, at `((kx[i] + 1/2) / 128, (ky[j] + 1/2) / 128)`.

The third coordinate of the planar meshes (T5/T6/T7/T8) is 0; T1 points are 2-D.  The raw edge cochains of T6/T7 are
not stored in the pickles; `datasets/generators/gen_T6_native.py` / `gen_T7_native.py` regenerate them with the v1
generators' random streams (the node encodings are reproduced to 1e-7) into `datasets/v2/T6_native.pt` /
`T7_native.pt`.  The v1 checkpoints (seeds 42, 1, 2; `python3 datasets/download_v1.py --ckpt`) live under
`checkpoints_v1/{task}/{model}/best_model.pt` for `python -m rhmp.train --eval-v1`.

## v2 packed sets (`datasets/v2/*.pt`)

`torch.save`d dicts.  Variable-mesh sets store all samples as flat concatenations with int64 offsets `ptr0 ... ptr3`
(`N + 1` entries each; sample `i` owns rows `ptr_k[i]:ptr_k[i+1]` of every degree-k array), float32 fields, int32
cells, plus `meta` (generator arguments, seeds, residuals) and, for most sets, a `.json` summary next to the file.
Edges are the unique `src < dst` vertex pairs of the cells in lexicographic order, and edge quantities refer to the
orientation `src -> dst`; loaders align every stored cell order with the complex they build
(`rhmp.data.cell_alignment`, `edge_alignment`).  `*_fine.pt` files re-solve the first test-split instances on meshes
with 4x as many vertices (same physical fields): the zero-shot resolution-transfer sets.

| family | per-sample arrays (degree) | notes |
|---|---|---|
| T2v (`gen_T2v.py`, shared mesh of T2) | `V (N, n0, 3)` (float32, the generator's velocity), `alpha, beta (N,)`, `vt, vp (n0, 3)`, `test_index (450,)`, `alpha_sweep = {alphas, V (5, 450, n0, 3), Y (5, 450, n0)}`, `rollout = {Y10, Y20 (450, n0)}`, `meta` (reproduction checks, scheme diagnostics) | `v = alpha vt + beta vp`; the stored T2 targets are re-simulated to 2.4e-7 (`meta.target_check`) |
| T2a (`gen_T2a.py`, T2 torus) | `points, faces, edges`, `X, Y (N, n0)`, `V, b (N, n0, 3)`, `edge_cos2, edge_logp (N, n1)` (generator edge order; the loader aligns them), `Dpar, Dperp (N,)`, `params`, `test_index`, zero-shot sets `alpha`, `ratio`, `rollout`, `fine`, `coarse` (own meshes), `meta` (scheme checks); `T2a_valmesh.pt` (`--val-meshes`): `val_index`, the validation instances on the `fine` and `coarse` tori | `d_t u + div(v u) = div(D grad u)`, divergence-free `v`, field-aligned `D`; P1 tensor diffusion, exactly divergence-free dual fluxes, Crank-Nicolson (200 sub-steps); mass conserved to 1e-14 |
| HP (`gen_HP.py`) | `pos` (0), `faces` (2), `f`, `u`, `bnd` (0), `sigma_edge`, `sigma_proj_edge` / `sigma_dir_edge` (1), `logsigma_face`, `logdet_face`, `logratio_face` (2), `flux`, `flux_fem` (1); evaluation only: `theta_face`, `sigma_tensor_face` | `-div(sigma grad u) = f`, u = 0 on the boundary, P1 FEM with element-wise sigma, lumped mass |
| TET (`gen_TET.py`) | `pos`, `tets`, `f`, `u`, conductivity per tet, face fluxes | 3-D P1 FEM on random tetrahedral meshes |
| SURF (`gen_surf.py`) | `pos`, `faces`, `f`, `u` (screened Poisson), `heat` (heat flow), `family` | closed surfaces at a fixed physical resolution (median edge 0.07) |
| DYN (`gen_dyn.py`) | `pos`, `faces`, edge velocity 1-form, 101 states per trajectory | implicit DEC advection-diffusion with a divergence-free velocity |
| HP_qual (`gen_qual.py`) | HP_k100 test instances on graded and sliver meshes, `u` and the 4x-finer reference `u_ref`, per-level quality statistics | test only |
| MXS (`gen_maxwell.py`) | one shared mesh: `pos`, `faces`, `region` (fixed regions), `boundary`; per sample `currents (N, 3)`, `phase (N, n2)` int8 (air / copper / iron), `A (N, n0)` f64 (Wb/m), `B_node (N, n0, 2)` (T), Newton statistics, layout parameters in `meta` | nonlinear magnetostatics `-div(nu(|B|) grad A) = J`, P1 FEM + damped Newton; `_ood15` (currents x1.5), `_fine` / `_coarse` (other meshes) re-solve test samples 4250..4749 |
| MXA (`gen_maxwell_aniso.py`) | one shared lattice mesh: `pos`, `faces`, `boundary`, `lattice_face`; per sample `phase (N, n2)` int8 (air / copper / NO / GO / LAM), `phi (N, n2)` (rolling direction), `J_face (N, n2)` (A/mm^2), `family (N,)`, `A (N, n0)` f64, `B_node (N, n0, 2)` (T), Newton statistics, layouts in `meta` | anisotropic nonlinear magnetostatics (energy `w_e(|B|) cos^2 + w_h(|B|) sin^2` of the angle to the rolling direction), P1 FEM + damped Newton; `_ood15`, `_fine` / `_coarse` re-solve test samples 4250..4749, `_heldout` = 500 samples of a layout family absent from training |
| AFM (`gen_airfrans.py mesh`) | per simulation: conforming near-field mesh (`pos`, `faces`, boundary rings), signed distance, inlet velocity and angle of attack, node pressure and velocity targets; `AFM_fine.pt`: the same simulations on finer meshes | AirfRANS simulations re-meshed on body-fitted near-field meshes (1.9K nodes; 0.55K / 6.9K-node zero-shot meshes), see `rhmp/tasks/airfrans.py` |
| MXE (`gen_maxwell.py estat`) | one shared mesh with electrode holes: `pos`, `faces`, `region`, `electrode`; per sample `potentials (N, 8)`, `q_face`, `phase`, `phi`, `E_node (N, n0, 2)` | linear electrostatics `-div(eps grad phi) = q` with Dirichlet electrodes, P1 FEM; `_ood15` (potentials and charges x1.5), `_fine` / `_coarse` |
| AHP / ASURF / ACURL / ADARCY (`gen_aniso.py`) | `pos`, `faces` or `tets`, sources, targets (`u`, `heat`, `A`, `J`, `p`), edge projections `logproj_edge`, cell invariants (`logdet_*`, `logratio_*`), true tensors (`sigma_face`, `nu_tet`, `K_tet`) | exact P1 / Nédélec / RT0 solves; `ANISO_representability.json` holds the oracle analysis |

Full field lists, equations and sizes are in the module docstrings of the generators, in
[TASK_SUITE_DETAILS.md](TASK_SUITE_DETAILS.md) and in [ANISO_TASKS.md](ANISO_TASKS.md).  `T8_complexes_<star>.pt` is a cache of the T8 complexes written by the loader.
