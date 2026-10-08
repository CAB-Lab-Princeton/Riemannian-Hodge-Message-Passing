# Changelog

## 2.1.0 (October 2026): the tasks and components of the camera-ready paper

The release behind the tables of *Learning Discrete Riemannian Metrics for Physical Fields with Cochain-Frame
Equivariance* (NeurIPS 2026).  Every new option defaults to off: configurations, checkpoints and results of 2.0.0
load and evaluate unchanged.

### New tasks

* **NS vorticity on a stretched periodic grid** (`T1g`, `T1gml`, `T1gml_mean`; `datasets/generators/gen_T1_graded.py`):
  the PDEBench trajectories of T1 sampled on non-uniform periodic tensor-product grids, with the vorticity target
  computed spectrally on the fine simulation grid and zero-shot grids (a second stretched grid, uniform 32^2 / 64^2
  grids, the rotated training grid) evaluated at the end of every run; periodic grid complexes
  (`CochainComplex.from_grid(..., periodic=True)`); `T1p` / `T1pm` / `T1pml` put the original T1 data on the periodic grid.
* **Anisotropic transport on a torus** (`T2a`; `rhmp/tasks/torus.py`, `gen_T2a.py`): field-aligned anisotropic
  advection-diffusion with the velocity as an edge 1-form input, anisotropy descriptors on edges and faces, and
  zero-shot sets (speeds and anisotropy ratios beyond the training range, roll-out, finer and coarser tori);
  `T2v` restores the per-sample velocity of the original torus task (`gen_T2v.py`).
* **Nonlinear anisotropic magnetostatics** (`MXA`, `MXA_noorient`; `rhmp/tasks/maxwell.py`, `gen_maxwell_aniso.py`):
  grain-oriented and laminated steel in electromagnets and transformers of several layout families, exact
  finite-element targets, zero-shot sets (currents x 1.5, a held-out layout family, finer and coarser meshes); also
  the isotropic magnetostatics (`MXS`) and electrostatics (`MXE`) families of the same module.
* **AirfRANS pressure on body-fitted meshes** (`AFP*`, `T8i`; `rhmp/tasks/airfrans.py`, `gen_airfrans.py mesh`):
  conforming near-field meshes of the AirfRANS simulations with the inflow as an edge 1-form, the outer ring and the
  harmonic generator of the annulus as domain fields, zero-shot meshes with the spacing doubled and halved,
  pressure-coefficient and potential-flow controls.
* **Ellipsoid flow with the exact target structure** (`T3g`): the stream function is predicted and mapped to the
  tangent field by a fixed map; `--lifting linear`, `--readout-head linear`, `--fit-readout`.

### New components

* **Metric Hodge projection layer** (`layers=['project', ...]`; `rhmp.layers.metric_projection`,
  `direct_projection`, `dense_projection`, `KuttaHead`): the exact (gradient) part of the edge features is removed
  with the learned edge metric, the potential is passed to the vertices, and a harmonic correction with a learned
  Kutta metric sets the circulation; solvers `cg`
  (preconditioned CG with implicit gradients), `direct` (sparse LU on the CPU, adjoint gradients) and `dense`
  (batched Cholesky on the device, diagonal and tensor metrics); `RHMP.projection_fields` for analysis.
* **Picard solve layer** (`solve_picard`, `picard_relax`, `picard_init`; `metric_geo=False`): fixed-point iterations of
  the physical solve layer with a state-dependent metric (field-energy invariant of the previous iterate), and metric
  heads without geometric descriptors (a constitutive law without mesh-specific cues); `solve_bc='lift'` for
  inhomogeneous Dirichlet data.
* **Transport term** (`transport`; `rhmp.transport`): conservative, advective and upwind transport fluxes of the vertex
  features along an input or learned edge velocity, with the metric or per-sample normalisation.
* **Tensor metrics in projection and resolvent layers** (`metric_type='tensor'`, `tensor_param='full'`): the full SPD
  Galerkin metric now serves the projection and the normalised resolvent layers (and their frozen-unit evaluation).
* **Energy readout and dynamic similarity** (`readout_energy`, `amplitude_power`, `amplitude_energy`): node readouts
  with Galerkin energy densities of projected edge features; inputs divided by the field magnitude of an amplitude
  1-form and the output multiplied by its power (exact `U^2` scaling of pressures).
* **Wedge face readout** (`readout_wedge`, `cochain_head='context'`, `readout_depth`, `readout_spatial`,
  `readout_skip`): gated cochain heads with face wedge products of projected edge features, several odd gated stages
  with spatial mixing, and an input skip.
* **Lifting context and vertex invariants** (`lift_context`, `lift_vertex`): the lifting gates of the higher degrees see a
  projection of the lifted vertex features, and the vertex lifting sees the codifferential of the odd edge inputs.
* **Homogeneous gate** (`gate='homog'`): an update gate that is homogeneous of degree one in the field amplitude.
* **Frozen units** (`frozen_units`, `frozen_descriptors` in `True` / `False` / `'auto'`; `RHMP.freeze_units`): the
  per-mesh normalisers (star means, descriptor medians, block bounds) of the training mesh are stored in the model
  and used on every mesh; the `'auto'` rule keeps a descriptor column in training units while a mesh's median lies
  in the training range and relative to the mesh otherwise.  Applicable to trained checkpoints without retraining.
* Trainer: flags for all of the above (`--layers project,...`, `--project-solver`, `--solve-picard`, `--transport`,
  `--readout-energy`, `--amplitude-*`, `--readout-wedge`, `--cochain-head`, `--lift-context`, `--lift-vertex`,
  `--gate homog`, `--frozen-units`, `--frozen-descriptors`, ...), `rhmp.train.needs_whitney`; resuming on CUDA and the
  implicit PCG backward with a shared metric graph fixed.
* Baselines: FNO and the node baselines on the periodic grids; `rhmp.baselines.registry.prepare_task` also
  transforms the extra test sets of the new tasks (zero-shot grids on a shared complex, lists of meshes) under the v1
  edge protocol.
* Scripts: `scripts/train_paper_models.sh` (every run of the paper), `scripts/eval_checkpoint.py`, the zero-shot and
  rotation evaluations (`eval_frozen_units.py`, `eval_torus_aniso.py`, `transfer_eval/`, `frame_eval/`,
  `maxwell_eval.py`, `airfrans_pressure_eval.py`), `collect_results.py` for the camera-ready `results/` layout.
* `results/`: the 253 runs of the paper (RHMP and the baselines, seeds 42, 1, 2) with the RHMP checkpoints, the
  evaluation files and `RESULTS.md`.

## 2.0.0 (September 2026): first release of the `rhmp` library

Version 2 re-implements the model of the paper (v1 code: https://github.com/ContinuumCoder/Riemannian-Hodge-Message-Passing)
as a library.  The core idea is unchanged (fixed topology with `d_{k+1} d_k = 0`, geometry and material only through
SPD cochain metrics, O(C) cochain-frame equivariance, E(n) invariance, Abelian gauge invariance); the implementation,
the algorithms and the task suite are new.

### Library

* `rhmp.complex.CochainComplex`: builders (`from_triangles`, `from_polygons`, `from_tetrahedra`, `from_grid`),
  validation of degenerate and duplicate cells, block-diagonal batches (`CochainComplex.batch` / `from_list`,
  `K.graph_ids(k)`), pinned non-blocking device moves.
* `rhmp.geometry`: E(n)-invariant, scale-free descriptors; cotan / barycentric / unit reference stars.
* `rhmp.ops`: CSR `spmm` with a precomputed-transpose backward, Gershgorin bounds, segment operations.
* `rhmp.model.RHMP` / `RHMPConfig`: normalised metric Hodge blocks, half-operator cross terms, polynomial filters,
  radial gate, native inputs on any degree (`[connection | odd | even]` columns), readouts `node_scalar`,
  `node_vector` (damped least squares), `cochain:k`, `even:k` and the exact constraint readouts `grad`, `curl`, `div`,
  `mdiv`; diagnostics, `metric_fields`, checkpoints (`to_checkpoint` / `from_checkpoint`, also the trainer's
  `best.pt` / `last.pt`).
* `rhmp.train`: the trainer CLI with the protocol of the paper, resume, JSON histories, block-diagonal batching with a
  prefetch thread, `--eval-ckpt` (transfer evaluation with the training normalisation) and `--eval-v1` (v1
  checkpoints with the same split and metrics).
* `rhmp.metrics`: R2 / MSE / MAE / NRMSE / SSIM / Pearson matching the v1 evaluation code to 1e-6; uncentred R2 for
  orientation-odd cochain targets.
* `rhmp.tasks`: the paper tasks in native (cochain) and legacy (v1) form, T1q (quad CW complex), T6f / T7f (face
  targets), heterogeneous and anisotropic Poisson (`HP*`, with zero-shot 4x-resolution test sets), tetrahedral
  Poisson (`TET*`), the extension suite (SURF, DYN, HP_qual) and the anisotropy suite (AHP, ASURF, ACURL, ADARCY).
* Parameter-free orientation output maps for the pseudo-scalar targets of the paper (reproduced to 7e-8).

### Algorithms

* `rhmp.dec`: metric Hodge Laplacians, Hodge decomposition, harmonic bases, cohomology generators, batched
  O(C)-equivariant CG with unrolled or implicit gradients, Whitney / Galerkin metrics and energies, Whitney `sharp`
  and de Rham `flat`.
* Whitney tensor metric (`metric_type='tensor'`): per-cell SPD material tensors through the Galerkin star (P1
  stiffness at initialisation, Nedelec curl-curl on tetrahedra); `tensor_param='full'` covers all SPD tensors, the
  cone parameterisation (`'cone'`) generates M-matrices only.
* Resolvent layers, `metric_reference` (known material in physical log units), `learn_metric=False` (pure DEC prior),
  `latent_dims` (learned per-cell input columns for shared meshes), `abs_scale` (a global length-scale column).
* Material identification: `material_dims` (material columns reach only the metric heads), `solve` layers in physical
  units with Dirichlet / Neumann handling and a two-level preconditioner (`solve_precond='twolevel'`),
  `RHMPConfig.solver_preset` / trainer `--solver-mode`, `model.operator_residual` and the operator-identification loss
  `--aux-pde`.
* Baselines behind one registry and the same trainer: MeshGraphNet, the v1 baselines (GCN, GAT, SchNet, EGNN,
  GaugeEquivCNN, GEM-CNN, MPSN, SCCNN, CW Net, Clifford-SMPN, FNO, DeepONet) re-executed by vectorised cores on the
  vendored v1 modules (`rhmp.baselines.v1`), the v1 model `ours_v1`, and the controls `dec_fixed`, `unit_star`,
  `unit_fixed`; parameter matching with the rule of the paper.
* Evaluation scripts: robustness and symmetry tables, mesh / resolution transfer, metric recovery, rollouts.
