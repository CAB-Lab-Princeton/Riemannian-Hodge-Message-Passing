# RHMP v2 API reference

Rendered by `python3 docs/gen_api.py` from the docstrings of `rhmp` 2.1.0 (torch 2.11.0).
Do not edit by hand; regenerate after changing public docstrings.  Conventions used throughout: cochain
features have the layout `(n_k, B, C)` (cells of degree k, samples sharing the complex, channels); a
block-diagonal batch of different meshes has `B = 1` and `K.batch[k]` holds the graph id of every k-cell.
Each entry shows the signature, a one-line summary and, folded, the full docstring with argument and return
shapes.  Start with `CochainComplex`, `RHMPConfig`, `RHMP`, `rhmp.dec`, `TaskData` and `rhmp.train.run`;
the tutorial is `docs/TUTORIAL.md`, the mathematics `docs/MATH.md`.

## Contents

- [Package (`rhmp`)](#package-rhmp)
- [Cochain complexes (`rhmp.complex`)](#cochain-complexes-rhmpcomplex): `class CochainComplex`, `validate_triangles`, `batch_complexes`
- [Model (`rhmp.model`)](#model-rhmpmodel): `class RHMPConfig`, `class RHMP`
- [DEC toolkit (`rhmp.dec`)](#dec-toolkit-rhmpdec): `whitney_blocks`, `apply_whitney_metric`, `whitney_rowsum_abs`, `assemble_whitney_metric`, `cg_solve`, `cg_solve_implicit`, `hodge_laplacian`, `hodge_decompose`, `harmonic_basis`, `cohomology_generators`, `harmonic_generators`, `top_measure`, `top_aspect`, `whitney_energy`, ...
- [Geometry (`rhmp.geometry`)](#geometry-rhmpgeometry): `min_image`, `local_cell_positions`, `face_measures`, `surface_geometry`, `volume_geometry`, `barycentric_gradients`, `whitney1_values`, `whitney2_values_tet`, `galerkin_blocks`, `whitney_triangles`, `whitney_tets`
- [Sparse and segment operations (`rhmp.ops`)](#sparse-and-segment-operations-rhmpops): `spmm`, `sparse_csr`, `csr_from_coo`, `csr_with_values`, `gather_apply`, `scatter_apply`, `gershgorin_bound`, `beta_unit`, `broadcast_to_cells`, `segment_sum`, `segment_mean`, `segment_max`, `power_iteration_norm`, `to_nbc`, ...
- [Data containers (`rhmp.data`)](#data-containers-rhmpdata): `class Stats`, `class OutputMap`, `class TaskData`, `sequential_split`, `holdout_split`, `feature_stats`, `scale_stats`, `edge_alignment`, `cell_alignment`, `shared_minibatch`, `mesh_minibatch`, `iterate_indices`, `concat_cells`, `same_device`, ...
- [Metrics (`rhmp.metrics`)](#metrics-rhmpmetrics): `r2_score`, `mse`, `mae`, `nrmse`, `ssim_pearson`, `summarize`, `unnormalize`, `flatten_ragged`
- [Trainer (`rhmp.train`)](#trainer-rhmptrain): `parse_args`, `run`, `build_config`, `model_output`, `batch_loss`, `pde_residual`, `predict`, `evaluate`, `needs_whitney`, `main`
- [Task registry (`rhmp.tasks`)](#task-registry-rhmptasks): `load_task`, `list_tasks`, `task_defaults`, `resolve_root`
- [Torus transport task (`rhmp.tasks.torus`)](#torus-transport-task-rhmptaskstorus): `t2a_file`, `t2a_inputs`, `t2a_complex`, `load_T2a`
- [Magnetostatics tasks (`rhmp.tasks.maxwell`)](#magnetostatics-tasks-rhmptasksmaxwell): `mxs_task_defaults`, `class FluxToNodeField`, `reluctivity`, `load_mxs`, `load_mxe`, `load_mxa`, `face_onehot`, `triangle_areas`, `edge_directions`, `rotate_frame_columns`
- [AirfRANS tasks (`rhmp.tasks.airfrans`)](#airfrans-tasks-rhmptasksairfrans): `airfrans_task_defaults`, `class StarAverageMap`, `class FluxVelocityMap`, `class OffsetMap`, `wall_vertices`, `afm_eval_source`, `afp_output_map`, `load_t8_inflow`, `potential_reference`, `load_t8res`, `load_t8_frame`, `load_afm`, `flux_divergence`, `trapezoid_flux`, ...
- [Anisotropy task suite (`rhmp.tasks.aniso`)](#anisotropy-task-suite-rhmptasksaniso): `aniso_task_defaults`, `parse_name`, `load_aniso`, `structure_metrics`, `cell_tensors`
- [Extension task suite (`rhmp.tasks.suite`)](#extension-task-suite-rhmptaskssuite): `suite_task_defaults`, `load_surf`, `load_dyn`, `load_dynfix`, `load_qual`, `rollout_eval`, `surf_integral_error`, `mass_weighted_errors`, `mass_inverse_map`, `mass_div_readout_available`
- [Robustness (`rhmp.robustness`)](#robustness-rhmprobustness): `remap_complex`, `transformed_task`, `gauge_task`, `noise_task`, `random_rotation`, `robustness_table`
- [Baselines (`rhmp.baselines`)](#baselines-rhmpbaselines)
- [Baseline registry (`rhmp.baselines.registry`)](#baseline-registry-rhmpbaselinesregistry): `class ModelSpec`, `applicable`, `build_model`, `model_names`, `rhmp_param_count`, `default_mode`, `from_checkpoint`, `prepare_task`, `resolve_head`, `main`
- [Building blocks: layers (`rhmp.layers`)](#building-blocks-layers-rhmplayers): `class LayerContext`, `class BlockGeometry`, `make_context`, `class RadialGate`, `class RHMPLayer`, `class BlockOperands`, `block_operands`, `apply_L`, `apply_T`, `poly_block`, `poly_block_reference`, `spmm_ad`, `class SharedCoboundaries`, `mean_square`, ...
- [Building blocks: transport term (`rhmp.transport`)](#building-blocks-transport-term-rhmptransport): `normalize_transport`, `class TransportTerm`, `transport_columns`, `transport_weights`, `apply_transport`, `transport_bound`
- [Building blocks: metric heads (`rhmp.metric`)](#building-blocks-metric-heads-rhmpmetric): `class MetricHead`, `class TensorMetricHead`, `metric_feature_dim`
- [Building blocks: lifting (`rhmp.lifting`)](#building-blocks-lifting-rhmplifting): `class GeoMLP`, `class CochainLifting`, `input_layout`, `vertex_mean`
- [Building blocks: readouts (`rhmp.readout`)](#building-blocks-readouts-rhmpreadout): `class NodeScalarReadout`, `class CochainReadout`, `class EvenCellReadout`, `class NodeVectorReadout`, `class GradReadout`, `class CurlReadout`, `class DivReadout`, `class MassDivReadout`, `build_readout`, `parse_readout`, `readout_degrees`, `vector_geometry`, `ls_solve`, `face_wedge`
- [Registries](#registries-generated-from-the-code)
- [Trainer CLI](#trainer-cli-python--m-rhmptrain---help)

## Package: `rhmp`

Top-level exports (lazy imports).

<details><summary>module docstring</summary>

```text
RHMP v2: Riemannian Hodge Message Passing on cochain complexes.

Public API (names are resolved lazily, so that each one imports only its own sub-module)::

    from rhmp import RHMP, RHMPConfig, CochainComplex          # model, configuration, complexes
    from rhmp import cg_solve, hodge_decompose, sharp, flat    # DEC toolkit (rhmp.dec)
    from rhmp import load_task                                  # task registry (rhmp.tasks)
```

</details>

- `rhmp.RHMP`: re-export of `rhmp.model.RHMP` (documented there).
- `rhmp.RHMPConfig`: re-export of `rhmp.model.RHMPConfig` (documented there).
- `rhmp.CochainComplex`: re-export of `rhmp.complex.CochainComplex` (documented there).
- `rhmp.cg_solve`: re-export of `rhmp.dec.cg_solve` (documented there).
- `rhmp.hodge_decompose`: re-export of `rhmp.dec.hodge_decompose` (documented there).
- `rhmp.sharp`: re-export of `rhmp.dec.sharp` (documented there).
- `rhmp.flat`: re-export of `rhmp.dec.flat` (documented there).
- `rhmp.load_task`: re-export of `rhmp.tasks.load_task` (documented there).
### Constants

- `__version__`: '2.1.0'


## Cochain complexes: `rhmp.complex`

Meshes -> exact oriented incidence, stars, geometry, batching.

<details><summary>module docstring</summary>

```text
Cochain complexes with exact oriented incidence, reference Hodge stars and cached geometry.

A :class:`CochainComplex` stores, for top degree ``K`` (2 = surfaces, 3 = tetrahedral volumes):

* ``cells[k]``: ``cells[0] = arange(n0)[:, None]``, ``cells[1]`` edges ``(n1, 2)`` with ``src < dst``,
  ``cells[2]`` faces ``(n2, m)`` in the given orientation (``-1`` padded for polygons; canonical sorted
  vertex order for faces of tetrahedra), ``cells[3]`` tets ``(n3, 4)`` in the given order.
* ``d[k]``: coboundary ``(n_{k+1}, n_k)`` as CSR float32; ``dT[k]`` its separately built transpose;
  ``d_abs[k]``, ``dT_abs[k]`` the CSR of ``|d_k|``.  ``d_{k+1} d_k = 0`` holds exactly (integer
  arithmetic); :meth:`CochainComplex.check_d2` verifies it with a sparse product.
* ``star[k] > 0`` reference diagonal Hodge star, ``geo[k]`` E(n)-invariant descriptors
  (see :mod:`rhmp.geometry`), ``boundary[k]`` flags.
* ``batch[k]`` sample id of every cell for block-diagonal batches (``None`` for a single complex).

All construction is vectorised torch (no Python loops over cells) and runs on the target device.
Block-diagonal batches are built with ``CochainComplex.batch([K1, K2, ...])`` (class-level call; the
instance attribute ``K.batch`` holds the per-degree sample ids).
```

</details>

### class `CochainComplex`

```python
CochainComplex(
    dim: int,
    n: list[int],
    cells: list[Tensor],
    d: list[Tensor],
    dT: list[Tensor],
    d_abs: list[Tensor],
    dT_abs: list[Tensor],
    pos: Tensor,
    geo: list[Tensor],
    star: list[Tensor],
    boundary: list[Tensor],
    batch: list[Tensor] | None,
    num_graphs: int,
    meta: dict,
    whitney: dict = {},
)
```
Oriented cochain complex with CSR coboundaries, reference stars and E(n)-invariant geometry.

| field | type | default | description |
|---|---|---|---|
| `dim` | `int` | required | top degree. |
| `n` | `list[int]` | required | `[n_0, ..., n_K]` cell counts. |
| `cells` | `list[Tensor]` | required | `cells[k]` vertex ids of the k-cells (see module docstring). |
| `d` | `list[Tensor]` | required | CSR `d_k (n_{k+1}, n_k)`, `d_k^T`, `\|d_k\|`, `\|d_k\|^T`, k = 0..K-1. |
| `dT` | `list[Tensor]` | required | CSR `d_k (n_{k+1}, n_k)`, `d_k^T`, `\|d_k\|`, `\|d_k\|^T`, k = 0..K-1. |
| `d_abs` | `list[Tensor]` | required | CSR `d_k (n_{k+1}, n_k)`, `d_k^T`, `\|d_k\|`, `\|d_k\|^T`, k = 0..K-1. |
| `dT_abs` | `list[Tensor]` | required | CSR `d_k (n_{k+1}, n_k)`, `d_k^T`, `\|d_k\|`, `\|d_k\|^T`, k = 0..K-1. |
| `pos` | `Tensor` | required | `(n0, D)` float32 vertex positions (only used through edge vectors in vector readouts). |
| `geo` | `list[Tensor]` | required | `geo[k] (n_k, G_k)` float32 E(n)-invariant descriptors (names: `meta['geo_features'][k]`). |
| `star` | `list[Tensor]` | required | `star[k] (n_k,)` float32 reference Hodge star, > 0. |
| `boundary` | `list[Tensor]` | required | `boundary[k] (n_k,)` bool. |
| `batch` | `list[Tensor] \| None` | required | `batch[k] (n_k,)` int64 sample id per cell for block-diagonal batches, else `None`. |
| `num_graphs` | `int` | required | number of samples (meshes) in the complex. |
| `meta` | `dict` | required | extra info: `cell_type, star_type, geo_features, sizes, ptr, validation, face_index, geometry_stats, build_time_s`; `geo_log_median`: per degree `(num_graphs, G_k)` float64 log medians behind the relative descriptor columns (`rhmp.geometry.GEO_RELATIVE_COLUMNS`; 0 elsewhere); `face_edges (n2, m)` edge ids of every face in cyclic order (edge j joins vertex j and j+1 of `cells[2]`; -1 padded) with `face_edge_signs` (+-1, 0 pad); tets: `tet_faces (n3, 4)` (face i omits local vertex i) with `tet_face_signs`; polygons: `face_lengths (n2,)`; grids: `grid_shape`, `spacing`, `diagonal`; `beta_unit`: per-graph `(num_graphs,)` Gershgorin bounds of `S^{-1/2} A^T A S^{-1/2}` with keys `'up{k}_dec'` (A = d_k, S = star_k), `'up{k}_none'` (S = 1), `'down{k}_dec'` (A = d_{k-1}^T, S = 1/star_k), `'down{k}_none'`. Optional, set by a task and concatenated by `batch` when every part has them as dicts: `dirichlet = {k: (n_k,) bool}` (fixed cells of solve / projection layers; default: the boundary cells) and `cohomology = {k: (n_k, r)}` (closed k-cochains generating `H^k`, e.g. `rhmp.dec.cohomology_generators`; parts with fewer generators are padded with zero columns). |
| `whitney` | `dict` | `{}` | `{k: dict}` precomputed Whitney/Galerkin metric blocks of degree k (simplicial complexes only: k=1 for triangles, k=1 and 2 for tets; see `rhmp.geometry.whitney_triangles` / `rhmp.geometry.whitney_tets` and `rhmp.dec`), plus CSR `gather ((m_k n_top), n_k)` and `scatter (n_k, (m_k n_top))` maps between k-cells and per-top-cell slots (slot-major rows). |

<details><summary>docstring</summary>

```text
Fields (``K`` = top degree ``dim``):
    dim: top degree.
    n: ``[n_0, ..., n_K]`` cell counts.
    cells: ``cells[k]`` vertex ids of the k-cells (see module docstring).
    d, dT, d_abs, dT_abs: CSR ``d_k (n_{k+1}, n_k)``, ``d_k^T``, ``|d_k|``, ``|d_k|^T``, k = 0..K-1.
    pos: ``(n0, D)`` float32 vertex positions (only used through edge vectors in vector readouts).
    geo: ``geo[k] (n_k, G_k)`` float32 E(n)-invariant descriptors (names: ``meta['geo_features'][k]``).
    star: ``star[k] (n_k,)`` float32 reference Hodge star, > 0.
    boundary: ``boundary[k] (n_k,)`` bool.
    batch: ``batch[k] (n_k,)`` int64 sample id per cell for block-diagonal batches, else ``None``.
    num_graphs: number of samples (meshes) in the complex.
    whitney: ``{k: dict}`` precomputed Whitney/Galerkin metric blocks of degree k (simplicial complexes only:
        k=1 for triangles, k=1 and 2 for tets; see :func:`rhmp.geometry.whitney_triangles` /
        :func:`rhmp.geometry.whitney_tets` and :mod:`rhmp.dec`), plus CSR ``gather ((m_k n_top), n_k)`` and
        ``scatter (n_k, (m_k n_top))`` maps between k-cells and per-top-cell slots (slot-major rows).
    meta: extra info: ``cell_type, star_type, geo_features, sizes, ptr, validation, face_index,
        geometry_stats, build_time_s``; ``geo_log_median``: per degree ``(num_graphs, G_k)`` float64 log medians
        behind the relative descriptor columns (``rhmp.geometry.GEO_RELATIVE_COLUMNS``; 0 elsewhere);
        ``face_edges (n2, m)`` edge ids of every face in cyclic order
        (edge j joins vertex j and j+1 of ``cells[2]``; -1 padded) with ``face_edge_signs`` (+-1, 0 pad);
        tets: ``tet_faces (n3, 4)`` (face i omits local vertex i) with ``tet_face_signs``; polygons:
        ``face_lengths (n2,)``; grids: ``grid_shape``, ``spacing``, ``diagonal``; ``beta_unit``: per-graph
        ``(num_graphs,)`` Gershgorin bounds of ``S^{-1/2} A^T A S^{-1/2}`` with keys ``'up{k}_dec'``
        (A = d_k, S = star_k), ``'up{k}_none'`` (S = 1), ``'down{k}_dec'`` (A = d_{k-1}^T, S = 1/star_k),
        ``'down{k}_none'``.  Optional, set by a task and concatenated by ``batch`` when every part has them as
        dicts: ``dirichlet = {k: (n_k,) bool}`` (fixed cells of solve / projection layers; default: the boundary
        cells) and ``cohomology = {k: (n_k, r)}`` (closed k-cochains generating ``H^k``, e.g.
        :func:`rhmp.dec.cohomology_generators`; parts with fewer generators are padded with zero columns).
```

</details>

#### `CochainComplex.device` (property)

Device of all tensors of the complex.

#### `CochainComplex.geo_dims` (property)

`{k: G_k}` for all degrees (constructor argument of the model).

#### `CochainComplex.geo_dim`

```python
geo_dim(k: int) -> int
```
Number of descriptor columns `G_k` of degree `k`.

#### `CochainComplex.graph_ids`

```python
graph_ids(k: int) -> Tensor | None
```
Graph (sample) id of every k-cell of a block-diagonal batch, `(n_k,)` int64; `None` for a single complex.

<details><summary>docstring</summary>

```text
Use this instead of the instance attribute ``batch`` (whose name is shared with the class-level builder
``CochainComplex.batch([...])``).
```

</details>

#### classmethod `CochainComplex.from_list`

```python
from_list(complexes: list['CochainComplex']) -> CochainComplex
```
Block-diagonal batch of complexes (same as `CochainComplex.batch([...])` / `batch_complexes`).

<details><summary>docstring</summary>

```text
Args:
    complexes: non-empty list with equal ``dim``, position dimension, descriptor widths and device.
Returns:
    :class:`CochainComplex` with per-cell graph ids (``graph_ids(k)``), ``num_graphs`` and ``meta['ptr']``.
```

</details>

#### `CochainComplex.to`

```python
to(
    device=None,
    dtype: torch.dtype | None = None,
    non_blocking: bool | None = None,
) -> CochainComplex
```
Copy of the complex on `device` (CSR operators included).

<details><summary>docstring</summary>

```text
Host-to-GPU moves (CPU -> CUDA) use page-locked memory and asynchronous copies: the tensors of this (CPU)
complex are pinned once, in place (pinned CPU tensors behave like pageable ones, so later moves of the same
complex reuse them), and every copy is issued with ``non_blocking=True`` on the current CUDA stream.  No
device synchronisation is needed: kernels on that stream run after the copies, and PyTorch's caching host
allocator keeps the pinned buffers alive until the copies have completed.  (Pageable blocking copies would
stall the host behind the queued GPU work, about 5 ms per tensor on a busy GPU.)  Moves in all other directions
use plain copies.

Args:
    device: target device (default: current).
    dtype: optional floating dtype for ``d*``, ``pos``, ``geo``, ``star`` (e.g. ``torch.float64``
        for numerical tests); index and bool tensors are unchanged.
    non_blocking: ``None`` (default) = pinned asynchronous copies for CPU -> CUDA, plain copies otherwise;
        ``False`` forces the plain blocking copies.
Returns:
    a new :class:`CochainComplex`.
```

</details>

#### `CochainComplex.apply_d`

```python
apply_d(k: int, x: Tensor) -> Tensor
```
Coboundary `d_k x`: `(n_k, B, C) -> (n_{k+1}, B, C)` (any trailing shape; contiguous).

#### `CochainComplex.apply_dT`

```python
apply_dT(k: int, x: Tensor) -> Tensor
```
Transposed coboundary `d_k^T x`: `(n_{k+1}, B, C) -> (n_k, B, C)`.

#### `CochainComplex.apply_d_abs`

```python
apply_d_abs(k: int, x: Tensor) -> Tensor
```
`|d_k| x`: `(n_k, ...) -> (n_{k+1}, ...)` (Jacobi scalings, Gershgorin bounds).

#### `CochainComplex.apply_dT_abs`

```python
apply_dT_abs(k: int, x: Tensor) -> Tensor
```
`|d_k|^T x`: `(n_{k+1}, ...) -> (n_k, ...)`.

#### `CochainComplex.edge_vectors`

```python
edge_vectors() -> Tensor
```
`(n1, D)` edge vectors `pos[dst] - pos[src]` (E(n)-equivariant); on a periodic complex (`meta['period']`) their minimal images (block-diagonal batches with periodic parts store them in `meta['edge_vectors']`).

#### `CochainComplex.cell_positions`

```python
cell_positions(k: int) -> Tensor
```
Vertex positions of the k-cells, `(n_k, m, D)` (`-1` padded slots repeat the first vertex).

<details><summary>docstring</summary>

```text
On a periodic complex every cell is given in its own unwrapped frame (first vertex plus the minimal images of
the others, :func:`rhmp.geometry.local_cell_positions`); use this instead of ``pos[cells[k]]`` for anything
geometric (signed areas, barycentric gradients).
```

</details>

#### `CochainComplex.check_d2`

```python
check_d2() -> float
```
`max_k max |d_{k+1} d_k|` computed with sparse products (0.0 for a valid complex).

#### staticmethod `CochainComplex.from_triangles`

```python
from_triangles(
    pos,
    faces,
    *,
    star: str = 'cotan',
    validate: bool = True,
    device=None,
    period=None,
) -> CochainComplex
```
Triangle mesh (planar 2-D or surface in 3-D; boundaries and non-manifold edges allowed).

<details><summary>docstring</summary>

```text
Args:
    pos: ``(n0, 2|3)`` positions (numpy or torch, any float dtype).
    faces: ``(n2, 3)`` vertex ids (numpy or torch, int32/int64); orientation is kept.
    star: ``'cotan'`` (default) | ``'barycentric'`` | ``'unit'``.
    validate: drop degenerate/duplicate faces with a warning (``meta['validation']``,
        ``meta['face_index']`` maps kept faces to input rows).  With ``False`` faces with repeated
        vertices raise.
    device: target device (default: device of ``pos`` if a tensor, else CPU).
    period: ``None`` or the axis-aligned periods ``(L_x, L_y[, L_z])`` of a flat torus: faces may wrap around
        and all geometry is intrinsic to the torus (minimal-image edge vectors, every face measured in its
        own unwrapped frame; ``meta['period']``).
Returns:
    :class:`CochainComplex` with ``dim = 2``.
```

</details>

#### staticmethod `CochainComplex.from_polygons`

```python
from_polygons(
    pos,
    faces,
    *,
    star: str = 'barycentric',
    validate: bool = True,
    device=None,
    period=None,
) -> CochainComplex
```
Polygonal (CW) 2-complex: quads, mixed polygons, triangles.

<details><summary>docstring</summary>

```text
Args:
    pos: ``(n0, 2|3)`` positions.
    faces: ``list[list[int]]`` (cyclic vertex order = orientation) or ``(n2, M)`` integer array padded
        with ``-1`` (padding trailing).
    star: ``'barycentric'`` (default) | ``'cotan'`` (cotan terms on triangles, barycentric half dual
        edges on polygons = circumcentric on rectangles) | ``'unit'``.
    validate: drop polygons with < 3 vertices, repeated vertices, zero area, duplicates (warning).
    device: target device.
    period: ``None`` or the axis periods of a flat torus (periodic complex, see :meth:`from_triangles`).
Returns:
    :class:`CochainComplex` with ``dim = 2`` and ``cells[2]`` of shape ``(n2, M)`` (``-1`` padded);
    ``d[1]`` rows list the boundary edges in cyclic order with orientation signs.
```

</details>

#### staticmethod `CochainComplex.from_tetrahedra`

```python
from_tetrahedra(
    pos,
    tets,
    *,
    star: str = 'barycentric',
    validate: bool = True,
    device=None,
) -> CochainComplex
```
Tetrahedral 3-complex (degrees 0..3).

<details><summary>docstring</summary>

```text
Faces are the unique boundary triangles in canonical (sorted) vertex order; ``d_2`` carries the
orientation sign ``(-1)^i * parity`` of face i (omitting local vertex i) in each tet.

Args:
    pos: ``(n0, 3)`` positions.
    tets: ``(n3, 4)`` vertex ids (input order is the orientation).
    star: ``'barycentric'`` (default) | ``'unit'``.
    validate: drop tets with repeated vertices, zero volume, duplicates (warning).
    device: target device.
Returns:
    :class:`CochainComplex` with ``dim = 3``.
```

</details>

#### staticmethod `CochainComplex.from_grid`

```python
from_grid(
    shape: tuple[int, int],
    spacing=1.0,
    *,
    cell: str = 'quad',
    star: str | None = None,
    diagonal: str | None = None,
    device=None,
    periodic: bool = False,
) -> CochainComplex
```
Regular 2-D grid with `nx * ny` vertices, vertex id `i * ny + j` at `(i*sx, j*sy)`.

<details><summary>docstring</summary>

```text
The vertex numbering is the v1 convention (``meshgrid(xs, ys, indexing='ij')``, as in the v1 PDEBench
loader ``build_regular_grid`` used for T1), so per-vertex data stored as
``field[ix, iy].reshape(-1)`` lines up.

Args:
    shape: ``(nx, ny)`` vertices per axis (>= 2 each; >= 3 for ``periodic``).
    spacing: ``h`` or ``(sx, sy)``.  ``from_grid((32, 32), 1/31, cell='tri')`` reproduces T1.
    cell: ``'quad'`` (CCW quads ``[v00, v10, v11, v01]``) or ``'tri'``.
    star: default ``'cotan'`` for triangles, ``'barycentric'`` for quads (identical on rectangles).
    diagonal: triangle split for ``cell='tri'``: ``'v1'`` (default without ``periodic``) = scipy/qhull
        Delaunay of the ``linspace(0, 1, n)`` grid exactly as v1 built T1 (degenerate cocircular squares ->
        qhull-chosen diagonals; verified identical to the T1 faces with scipy 1.17), faces made
        CCW; ``'main'`` (v00-v11; default with ``periodic``), ``'anti'`` (v10-v01), ``'alternate'``
        (checkerboard) are deterministic and need no scipy.
    device: target device.
    periodic: flat torus of size ``(nx sx) x (ny sy)``: the cells of the squares ``(i, j)``, ``i < nx``,
        ``j < ny``, join vertex ``(i, j)`` to ``((i + 1) % nx, (j + 1) % ny)`` (wrap-around edges and faces),
        all cells are congruent and the geometry is intrinsic (minimal-image edge vectors, cells measured in
        their own unwrapped frames; ``meta['period'] = (nx sx, ny sy)``).  Euler characteristic 0, Betti
        numbers ``(1, 2, 1)``, no boundary.
Returns:
    :class:`CochainComplex` (``cell_type`` ``'quad'`` or ``'triangle'``) with ``meta['grid_shape']``.
```

</details>

#### staticmethod `CochainComplex.batch`

```python
batch(complexes: list[CochainComplex]) -> CochainComplex
```
Block-diagonal union of complexes (no re-triangulation; geometry is reused as is).

<details><summary>docstring</summary>

```text
Every part keeps its own per-mesh normalised descriptors and stars, so a batch is exactly the
disjoint union of its parts.  Parts may themselves be batches.

Args:
    complexes: non-empty list with equal ``dim``, position dimension, descriptor widths and device.
Returns:
    :class:`CochainComplex` with ``batch[k] (n_k,)`` sample ids, ``num_graphs = sum`` of parts,
    ``meta['sizes']`` (one ``[n_0..n_K]`` per graph) and ``meta['ptr'][k]`` (``(num_graphs+1,)`` offsets).
```

</details>

### `validate_triangles`

```python
validate_triangles(pos, faces) -> tuple[Tensor, dict]
```
Drop degenerate and duplicate triangles.

<details><summary>docstring</summary>

```text
A face is dropped if it repeats a vertex, if it is collinear up to round-off
(``2 * area < DEGENERATE_REL_MEASURE * longest_edge^2``) or if its vertex set already occurred
(first occurrence kept, whatever its orientation).  Order and orientation of kept faces are preserved.

Args:
    pos: ``(n0, D)`` positions (D = 2 or 3; numpy or torch).
    faces: ``(n2, 3)`` integer vertex ids (numpy or torch, int32/int64).
Returns:
    ``(faces_clean (n2', 3) int64, stats)`` with ``stats`` keys ``n_input, n_degenerate_index,
    n_degenerate_measure, n_duplicate, n_kept`` (ints) and ``kept_index`` (``(n2',)`` int64 indices
    of kept input faces).
Raises:
    ValueError: on malformed shapes or out-of-range vertex ids.
```

</details>

### `batch_complexes`

```python
batch_complexes(complexes: list[CochainComplex]) -> CochainComplex
```
Block-diagonal union of complexes (no re-triangulation; geometry is reused as is).

<details><summary>docstring</summary>

```text
Every part keeps its own per-mesh normalised descriptors and stars, so a batch is exactly the
disjoint union of its parts.  Parts may themselves be batches.

Args:
    complexes: non-empty list with equal ``dim``, position dimension, descriptor widths and device.
Returns:
    :class:`CochainComplex` with ``batch[k] (n_k,)`` sample ids, ``num_graphs = sum`` of parts,
    ``meta['sizes']`` (one ``[n_0..n_K]`` per graph) and ``meta['ptr'][k]`` (``(num_graphs+1,)`` offsets).
```

</details>


## Model: `rhmp.model`

Configuration and the RHMP network.

<details><summary>module docstring</summary>

```text
RHMP v2 model: configuration, lifting -> metric Hodge message passing -> readout (DESIGN §3, §4).

Inputs are a dict ``{k: (n_k, B, F_k)}`` of raw cochains on any degree (``cfg.in_dims``); the complex ``K`` is a
``rhmp.complex.CochainComplex`` shared by the ``B`` samples (or a block-diagonal batch of meshes with ``B = 1``).
The output of :meth:`RHMP.forward` is ``(n_out, B, out_dim)`` (``out_dim * D`` for ``node_vector``).

Checkpoints: ``{'cfg': cfg.to_dict(), 'geo_dims': {k: G_k}, 'state_dict': ...}`` (:meth:`RHMP.to_checkpoint`,
:meth:`RHMP.from_checkpoint`).  No parameter shape depends on the number of cells.
```

</details>

### class `RHMPConfig`

```python
RHMPConfig(
    in_dims: dict[int, int],
    even_dims: dict[int, int] = {},
    connection_dims: dict[int, int] = {},
    C: int = 128,
    n_layers: int = 4,
    poly_order: int = 2,
    metric_hidden: int = 32,
    log_range: float = 2.0,
    tie_metrics: bool = True,
    scaling: str = 'dec',
    cross: bool = True,
    gate: str = 'norm',
    identity_metric: bool = False,
    readout: str = 'node_scalar',
    out_dim: int = 1,
    vector_mode: str = 'ls',
    checkpoint_layers: bool = False,
    amp: bool = False,
    connection_odd: bool = False,
    lift_hidden: int = 64,
    fused: bool = True,
    layers: list[str] | None = None,
    resolvent_iters: int = 20,
    resolvent_grad: str = 'implicit',
    metric_type: str = 'diag',
    metric_reference: dict[int, int] = {},
    resolvent_warm_start: bool = True,
    learn_metric: bool = True,
    latent_dims: dict[int, int] = {},
    material_dims: dict[int, int] = {},
    lifting: str = 'mlp',
    resolvent_normalize: bool = True,
    solve_iters: int = 64,
    solve_tol: float = 1e-06,
    solve_bc: str = 'dirichlet',
    tensor_param: str = 'full',
    solve_precond: str = 'none',
    readout_head: str = 'mlp',
    term_gates: bool = False,
    cochain_head: str = 'linear',
    readout_hidden: int = 16,
    amplitude_dims: dict[int, int] = {},
    lift_context: int = 0,
    lift_vertex: bool = False,
    readout_layers: str = 'last',
    readout_depth: int = 1,
    readout_skip: bool = False,
    readout_spatial: bool = False,
    readout_wedge: int = 0,
    frozen_units: bool = False,
    frozen_descriptors: bool | str = True,
    amplitude_power: int = 1,
    amplitude_energy: bool = False,
    readout_energy: int = 0,
    readout_energy_clip: float = 0.0,
    project_harmonic: bool = True,
    kutta_range: float = 8.0,
    project_solver: str = 'cg',
    kutta_corner: float = 0.0,
    transport: dict = {},
    solve_picard: int = 0,
    picard_relax: float = 0.5,
    metric_geo: bool = True,
    picard_init: str = 'zero',
)
```
Hyper-parameters of `RHMP`.

| field | type | default | description |
|---|---|---|---|
| `in_dims` | `dict[int, int]` | required | `{k: F_k}` input width per degree (degrees absent or 0 have no inputs). |
| `even_dims` | `dict[int, int]` | `{}` | `{k: E_k}` number of even columns (listed last) of `inputs[k]`; they also feed the metric head of degree `k`. |
| `connection_dims` | `dict[int, int]` | `{}` | `{k: c_k}` number of gauge-connection columns (listed first) of `inputs[k]`; used only through `d_k A` (exact gauge invariance), `k < K`. |
| `C` | `int` | `128` | hidden channels. |
| `n_layers` | `int` | `4` | number of message-passing layers. |
| `poly_order` | `int` | `2` | order `P` of the scalar polynomial filters. |
| `metric_hidden` | `int` | `32` | hidden width of the metric MLPs. |
| `log_range` | `float` | `2.0` | `a`; `H/star` is confined to `[e^-a, e^a]`. |
| `tie_metrics` | `bool` | `True` | one metric per degree (up uses `H_m`, down uses `1/H_m`) vs. separate up/down heads. |
| `scaling` | `str` | `'dec'` | `'dec' \| 'jacobi' \| 'none'` operator scaling. |
| `cross` | `bool` | `True` | cross-degree transport terms. |
| `gate` | `str` | `'norm'` | `'norm'` (radial gate), `'relu'` (ablation), `'none'` (no nonlinearity; used by `solver_preset`) or `'homog'` (gate homogeneous of degree one in the field amplitude: `x + gamma g(xi) m` with scale-invariant invariants `xi`, see `rhmp.layers.RadialGate`). |
| `identity_metric` | `bool` | `False` | every metric `H = 1` (ablation of the learned metric, DESIGN §3.2): no learned correction and no reference star inside `H`. With `scaling='dec'` the blocks still carry the DEC stars of their own degree through the symmetric scaling `S` (`S_up = star_k`, `S_down = 1/star_k`), so the geometric (physics) prior survives at that level; the fully combinatorial variant builds the complex with `star='unit'` (all reference stars 1) or uses `scaling='none'`. The default model is the pure DEC operator (`H = star`) at initialisation. |
| `readout` | `str` | `'node_scalar'` | `'node_scalar' \| 'node_vector' \| 'cochain:k' \| 'even:k' \| 'grad' \| 'curl' \| 'div' \| 'div:k' \| 'mdiv' \| 'mdiv:k'` (see `rhmp.readout`; `grad`/`curl`/`div` satisfy `d d = 0` exactly). |
| `out_dim` | `int` | `1` | output channels (number of vector fields for `node_vector`). |
| `vector_mode` | `str` | `'ls'` | `'ls' \| 'direct'` (`node_vector` only). |
| `checkpoint_layers` | `bool` | `False` | activation checkpointing per layer (training memory). |
| `amp` | `bool` | `False` | run the dense MLPs under bf16 autocast on CUDA (sparse products stay fp32). |
| `connection_odd` | `bool` | `False` | connection columns also enter the ordinary odd path (non-Abelian connections). |
| `lift_hidden` | `int` | `64` | hidden width of the even lifting gates `e_k`. |
| `fused` | `bool` | `True` | fused polynomial-block kernels (`False` = plain autograd reference path, same maths). |
| `layers` | `list[str] \| None` | `None` | per-layer types (`'poly' \| 'resolvent' \| 'solve' \| 'project'`), e.g. `['poly', 'poly', 'resolvent', 'poly']` (DESIGN §9.2) or `['project', 'poly', 'poly', 'poly']`; `None` means `['poly'] * n_layers`. When given it defines the depth (`n_layers` is set to `len(layers)`). |
| `resolvent_iters` | `int` | `20` | CG iterations of resolvent layers (default 20). The solve acts identically on every channel and commutes with channel mixing (`M^{-1}(x Q) = (M^{-1} x) Q`), so no equivariant channel reduction lowers its cost; fewer iterations or warm starts do. |
| `resolvent_grad` | `str` | `'implicit'` | `'implicit'` (adjoint CG solve, memory independent of the iterations; default) or `'unrolled'` (autograd through the checkpointed iterations; memory grows with `resolvent_iters`). |
| `metric_type` | `str` | `'diag'` | `'diag'` (diagonal metrics, default) or `'tensor'` (Whitney/Galerkin material-tensor metric on the intermediate degrees for up blocks and cross-up terms, DESIGN §9.1). |
| `metric_reference` | `dict[int, int]` | `{}` | `{m: column}`: absolute index of an *even* column of `inputs[m]` holding a log-scale material reference `ref_m` (e.g. `log t^T Sigma t` on edges). It is a fixed offset of the log-metric, `H_m = star_m exp(ref_m) exp(a tanh(MLP))` (down blocks use the inverse), so the bounded correction is relative to the known coefficient. Tensor metrics scale the material tensor instead: `b_f` by `exp(mean of ref over the m-cells of f)` and, for the edge metric, `a_{f,j}` by `exp(ref of direction edge j)`. Ignored with `identity_metric`. |
| `resolvent_warm_start` | `bool` | `True` | start each resolvent layer's CG from the previous resolvent layer's solution of the same degree (same shape), solving only the correction (default True; exact implicit gradients). |
| `learn_metric` | `bool` | `True` | `False` removes all metric heads: `H_m = star_m exp(ref_m)` exactly (tied or untied, down blocks use the inverse; tensor metric `b = 1, a = 0` = the Galerkin/Whitney star, times the reference scaling). The model is then the pure DEC/FEEC physics prior with only scalar polynomial, cross, gate, lifting and readout parameters (the `dec_fixed` baseline). `identity_metric=True` takes precedence (`H = 1`). |
| `latent_dims` | `dict[int, int]` | `{}` | `{k: r}`: `r` learnable per-cell input columns `Z_k` `(n_k, r)` on degree `k` (a learned hidden field, e.g. a velocity 1-form the data do not provide), initialised `N(0, 0.1^2)` and inserted before the even columns of `inputs[k]` (layout `[connection \| odd \| latent \| even]`): odd cochains for `k >= 1` (they enter the odd path and the lifting gates, never the metric heads), vertex columns for `k = 0`. These are the only mesh-sized parameters, so they tie the model to *one* mesh (shared-mesh tasks; a different `n_k` or a block-diagonal batch raises). They are created by `model.init_latents(K)` (call it before building the optimizer) or lazily, with a warning, by the first forward; checkpoints store them. All symmetry statements hold conditional on the latent field. |
| `material_dims` | `dict[int, int]` | `{}` | `{k: m}`: the last `m` even columns of `inputs[k]` are *material* columns (e.g. `log sigma`): they feed *only* the metric heads (`psi` of the degree-k metric, the tensor-metric descriptors) and `metric_reference`, never the lifting, a gate or any feature path, so the only route from the material to the output is the metric `H`. |
| `lifting` | `str` | `'mlp'` | `'mlp'` (default) or `'linear'` (`x_0 = W f_0`, `x_k = Lin(d x_{k-1}) + Lin(f_k)`: no geometry, biases or even gates; linear in the field inputs). |
| `resolvent_normalize` | `bool` | `True` | `False`: resolvent layers use the un-normalised physical operator `(I + tau L^2 Delta_H)^{-1}` (identifies the metric's magnitude; see `RHMPLayer`). |
| `solve_iters` | `int` | `64` | `'solve'` layers (`layers=[..., 'solve', ...]`): CG budget (the CG count grows like the square root of the mesh condition number; warm starts reuse the previous solve layer) and boundary condition (`'dirichlet'`: `K.meta['dirichlet'][k]` or `K.boundary[k]` fixed; `'none'`: no fixed cells; `'neumann'`: no fixed cells and vertex solves return the zero-mean (lumped mass) solution of the pure Neumann problem, as in the T5 generator; `'lift'`: the fixed cells of `'dirichlet'` keep the layer input, i.e. inhomogeneous Dirichlet data such as electrode potentials). A solve layer computes `y_k = (Delta_H + lam / L^2)^{-1} x_k` with the un-normalised metric Hodge Laplacian `Delta_H` (FEM stiffness / lumped mass for k = 0) and `lam = softplus(log_lam)` (init 1e-3, `L^D` = domain measure). See `solver_preset`. |
| `solve_tol` | `float` | `1e-06` | `'solve'` layers (`layers=[..., 'solve', ...]`): CG budget (the CG count grows like the square root of the mesh condition number; warm starts reuse the previous solve layer) and boundary condition (`'dirichlet'`: `K.meta['dirichlet'][k]` or `K.boundary[k]` fixed; `'none'`: no fixed cells; `'neumann'`: no fixed cells and vertex solves return the zero-mean (lumped mass) solution of the pure Neumann problem, as in the T5 generator; `'lift'`: the fixed cells of `'dirichlet'` keep the layer input, i.e. inhomogeneous Dirichlet data such as electrode potentials). A solve layer computes `y_k = (Delta_H + lam / L^2)^{-1} x_k` with the un-normalised metric Hodge Laplacian `Delta_H` (FEM stiffness / lumped mass for k = 0) and `lam = softplus(log_lam)` (init 1e-3, `L^D` = domain measure). See `solver_preset`. |
| `solve_bc` | `str` | `'dirichlet'` | `'solve'` layers (`layers=[..., 'solve', ...]`): CG budget (the CG count grows like the square root of the mesh condition number; warm starts reuse the previous solve layer) and boundary condition (`'dirichlet'`: `K.meta['dirichlet'][k]` or `K.boundary[k]` fixed; `'none'`: no fixed cells; `'neumann'`: no fixed cells and vertex solves return the zero-mean (lumped mass) solution of the pure Neumann problem, as in the T5 generator; `'lift'`: the fixed cells of `'dirichlet'` keep the layer input, i.e. inhomogeneous Dirichlet data such as electrode potentials). A solve layer computes `y_k = (Delta_H + lam / L^2)^{-1} x_k` with the un-normalised metric Hodge Laplacian `Delta_H` (FEM stiffness / lumped mass for k = 0) and `lam = softplus(log_lam)` (init 1e-3, `L^D` = domain measure). See `solver_preset`. |
| `tensor_param` | `str` | `'full'` | tensor-metric parameterisation: `'full'` (default) `sigma_f = b_f expm(sum_j s_j t_j t_j^T)` with signed bounded `s` (every SPD tensor with bounded condition number, incl. anisotropy misaligned with the edges) or `'cone'` (`b_f I + sum_j a_j t_j t_j^T`, `a >= 0`: M-matrix stiffness only, the same class as the diagonal metric on triangles). Configurations saved without a `tensor_param` key (with `metric_type='tensor'`) load as `'cone'`, the parameterisation they were trained with. |
| `solve_precond` | `str` | `'none'` | CG preconditioner of solve layers: `'none'` (Jacobi, default) or `'twolevel'` (Jacobi plus an additive coarse correction on per-graph spatial aggregates of about `rhmp.layers.COARSE_AGG_SIZE` vertices, assembled per sample and graph; degree 0 only, other degrees keep Jacobi). Same solution, far fewer iterations on large meshes; O(C)-equivariant and per sample. |
| `readout_head` | `str` | `'mlp'` | `'mlp'` (default) or `'linear'`: the potential head of the `grad` readout (`phi = MLP(x_0)` or `phi = Lin_nobias(x_0)`); the solver preset uses `'linear'` so that the model stays linear in the field inputs. |
| `term_gates` | `bool` | `False` | per-term gates in polynomial layers (`rhmp.layers.TermGate`): `m_k = sum_j c_j g_j t_j` over the powers `L^p x_k` and the cross terms, with per-cell factors `g_j` from scale-invariant invariants of the term vectors (default `False`: the plain polynomial message). |
| `cochain_head` | `str` | `'linear'` | `'linear'` (default), `'gated'` or `'context'`: head of `cochain:k` readouts (`'gated'`: odd linear map times an even gate from invariants and the even inputs of degree k; `'context'`: the gated head with the coboundary of the degree `k-1` features in the odd part and the vertex mean of the vertex features in the gate; `rhmp.readout.CochainReadout`). |
| `readout_hidden` | `int` | `16` | hidden width of the gated cochain heads. |
| `amplitude_dims` | `dict[int, int]` | `{}` | `{k: n}`: the first `n` columns of `inputs[k]` scale with one field amplitude (e.g. the velocity 1-form and its even companion): they are divided by their per-sample RMS `s` (over all such columns and cells; per graph in block-diagonal batches) and the output is multiplied by `s`, so the model is exactly homogeneous of degree one in that amplitude (the other inputs, e.g. densities, are not scaled). `s` is invariant under orientation changes, relabelling, E(n) and O(C), and per sample. |
| `lift_context` | `int` | `0` | `p > 0`: the even lifting gates of degrees `k >= 1` also see the vertex mean over the cell of a projection `P x_0` (`C -> p`) of the lifted vertex features (default 0: off; `rhmp.lifting`). |
| `lift_vertex` | `bool` | `False` | the vertex MLP of the lifting also sees the DEC codifferential of the odd edge inputs and the vertex means of their squares and of the even edge inputs (default False; `rhmp.lifting`). |
| `readout_layers` | `str` | `'last'` | `'last'` (default) or `'all'`: `cochain:k` readouts read the features of degree k after the lifting and after every layer, each through its own linear map (`u = sum_l W_l x_k^(l)`), so that different output channels combine different filters of the stack (`rhmp.readout.CochainReadout`). |
| `readout_depth` | `int` | `1` | number of odd gated stages of the gated / context cochain heads (default 1). |
| `readout_skip` | `bool` | `False` | the gated / context head of `cochain:k` also reads the odd input columns of degree `k` and the coboundary `d_{k-1} f_{k-1}^odd` of the odd input columns of degree `k-1` through bias-free linear maps (a linear skip from the inputs, e.g. the circulation of the velocity 1-form on faces; default False). |
| `readout_spatial` | `bool` | `False` | the further stages of a deep gated / context head (`readout_depth >= 2`) also mix the neighbouring k-cells through the combinatorial Hodge Laplacian of degree k (default False). |
| `readout_wedge` | `int` | `0` | `h > 0`: the gated / context head of a `cochain:2` readout also reads `h` wedge products `(x_1 A_j) ^ (x_1 B_j)` of two projections of the final edge features (metric-free cup product of 1-cochains on the faces; a true 2-cochain, so the output keeps its orientation parity), e.g. the advection `theta ^ delta(d theta)` of a vorticity 2-form; default 0: off (`rhmp.readout.face_wedge`). |
| `frozen_units` | `bool` | `False` | the per-mesh geometric normalisers (geometric means of the reference stars, medians behind the relative descriptors, unit-metric bounds of the Hodge blocks) are those of the training mesh, stored in the model by `RHMP.freeze_units` (the trainer calls it on the training complex), instead of being recomputed on every mesh: on another mesh the stars are divided by the training geometric means, the descriptors become `log(x / median_train)` and each block bound is `beta(H) * beta_unit(train) / beta_unit(mesh)`; tensor-metric blocks use `rho * beta_unit(train)`, and resolvent layers and transport fluxes the same block bounds. On the training mesh the model is unchanged; it is no longer invariant to a global change of length unit (default False; diagonal and tensor metrics, 'dec' / 'none' scaling, polynomial and normalised resolvent layers, transport norm 'metric'). |
| `frozen_descriptors` | `bool \| str` | `True` | with `frozen_units`, also express the relative descriptors against the training medians (default True). False: only the operator normalisation is that of the training mesh (star means and block bounds); the descriptors stay `log(x / median_mesh)` on every mesh. `'auto'`: decided per relative column and mesh: training-mesh units if the mesh's median of the column, in training-mesh units, lies in the [min, max] of that column on the training mesh, else relative to the mesh's own median (star means and block bounds always in training-mesh units). Ignored without frozen references. |
| `amplitude_power` | `int` | `1` | the output is multiplied by `s^p` (default 1); `p = 2`: homogeneous of degree two in the amplitude columns (e.g. a pressure for a velocity 1-form: dynamic similarity); `p = 0`: the output is not rescaled (a nondimensional output, e.g. a pressure coefficient). |
| `amplitude_energy` | `bool` | `False` | the output factor `s` is the field magnitude of the amplitude 1-form columns of degree 1, `s_E = sqrt(sum_f theta_f^T M_f theta_f / sum_f \|f\|)` (Galerkin energy density over the domain; exactly `\|u\|` for the 1-form of a uniform field `u` on any simplicial mesh), instead of the RMS of the cochain values, which scales with the edge lengths; the inputs are still divided by the RMS. The same energy density (of the normalised columns) is the reference of energy readouts and projection layers (`rhmp.layers.reference_energy`). Needs `amplitude_dims[1] > 0` and Whitney blocks (default False). |
| `readout_energy` | `int` | `0` | `h > 0`: the `node_scalar` readout also reads `h` Galerkin energy densities `<x_1 A_j, x_1 B_j>` of projections of the final edge features at the vertices (symmetric quadratic forms in the reference metric; `rhmp.readout.NodeScalarReadout`); default 0: off. |
| `readout_energy_clip` | `float` | `0.0` | `c > 0`: the energy channels are bounded, `q -> c tanh(q / c)` (default 0: off). |
| `project_harmonic` | `bool` | `True` | `'project'` layers (`layers=[..., 'project', ...]`: metric Hodge projection of the edge features with the learned edge metric, `rhmp.layers.RHMPLayer._project_forward`) also remove the circulation along the cohomology generators `K.meta['cohomology'][1]` in a learned Kutta metric (default True; no effect on complexes without generators). |
| `kutta_range` | `float` | `8.0` | bound `a` of the log Kutta weights of projection layers (default 8). |
| `project_solver` | `str` | `'cg'` | solver of projection layers: `'cg'` (default; preconditioned CG with `solve_iters`, `solve_tol`, `solve_precond`) or `'direct'` (sparse LU per graph and sample on the CPU in float64 with adjoint gradients; exact, diagonal metrics only) or `'dense'` (batched dense Cholesky on the device in float64, one factorisation per sample; exact, diagonal and tensor metrics, one graph per batch: `rhmp.layers.dense_projection`). |
| `kutta_corner` | `float` | `0.0` | `s > 0`: prior of the Kutta weights at re-entrant corners of the boundary, `kappa = a tanh(MLP + s c_e)` with a learnable slope initialised at `s` and the corner indicator `c_e` of `rhmp.layers.boundary_corner` (default 0: the weights are learned from a uniform start). |
| `transport` | `dict` | `{}` | transport term of the vertices (`rhmp.transport`; empty = off, the default). Keys: `inputs` (odd columns of `inputs[1]` that are transport fields, e.g. a velocity one-form), `latent` (number of latent edge columns used as fields), `hidden` (0-2 O(C)-invariant odd contractions of the hidden state), `forms` (subset of `cons, adv, upwind`), `order` (power `P` of the transport operator per layer), `norm` (`'metric'`: fluxes `H_1 phi` with the Gershgorin-normalised metric, the field keeps its magnitude; `'sample'`: every field operator normalised per sample), `metric` (`'layer'`: the layer's learned `H_1`; `'star'`: the reference star). Each poly / resolvent layer adds `sum_p g_p T^p x_0` to the vertex message, `T = sum_j a_j C(F_j) + b_j A(F_j) + c_j U(F_j)`, with scalar coefficients (init 0), conservative `C`, advective `A = C^T` and upwind `U` operators of the fluxes `F_j = H_1 phi_j / beta`. Solve and projection layers carry no transport term. |
| `solve_picard` | `int` | `0` | number `J` of fixed-point (Picard) iterations of the vertex solve of `'solve'` layers with a state-dependent metric (0, the default: one solve). Iteration `j` predicts the vertex-block metric (`H_1`, diagonal or tensor) from the usual descriptors plus the field-energy invariant of the previous iterate (`y_0 = 0`) and solves with the layer input as right-hand side; `y_j = y_{j-1} + omega (S(H_j) x - y_{j-1})` for `j >= 2`, weight-tied, learnable `omega` (see `RHMPLayer`). For nonlinear constitutive laws (e.g. saturable magnetic materials, `nu(\|B\|)`). |
| `picard_relax` | `float` | `0.5` | initial relaxation `omega` in (0, 1) of the Picard iteration. |
| `metric_geo` | `bool` | `True` | `False`: the metric heads see no geometric descriptors (material columns, feature invariants and the Picard energy invariant only), so the learned metric is a function of material and state alone (a constitutive law without mesh-specific position cues). Default `True`. |
| `picard_init` | `str` | `'zero'` | `'zero'` (default: the first Picard iteration starts from `y_0 = 0`) or `'learned'` (a learnable uniform starting energy invariant; see `RHMPLayer`). |

<details><summary>docstring</summary>

```text
Attributes:
    in_dims: ``{k: F_k}`` input width per degree (degrees absent or 0 have no inputs).
    even_dims: ``{k: E_k}`` number of even columns (listed last) of ``inputs[k]``; they also feed the metric
        head of degree ``k``.
    connection_dims: ``{k: c_k}`` number of gauge-connection columns (listed first) of ``inputs[k]``; used only
        through ``d_k A`` (exact gauge invariance), ``k < K``.
    C: hidden channels.
    n_layers: number of message-passing layers.
    poly_order: order ``P`` of the scalar polynomial filters.
    metric_hidden: hidden width of the metric MLPs.
    log_range: ``a``; ``H/star`` is confined to ``[e^-a, e^a]``.
    tie_metrics: one metric per degree (up uses ``H_m``, down uses ``1/H_m``) vs. separate up/down heads.
    scaling: ``'dec' | 'jacobi' | 'none'`` operator scaling.
    cross: cross-degree transport terms.
    gate: ``'norm'`` (radial gate), ``'relu'`` (ablation), ``'none'`` (no nonlinearity; used by
        :meth:`solver_preset`) or ``'homog'`` (gate homogeneous of degree one in the field amplitude:
        ``x + gamma g(xi) m`` with scale-invariant invariants ``xi``, see ``rhmp.layers.RadialGate``).
    identity_metric: every metric ``H = 1`` (ablation of the learned metric, DESIGN §3.2): no learned correction
        and no reference star inside ``H``.  With ``scaling='dec'`` the blocks still carry the DEC stars of their
        own degree through the symmetric scaling ``S`` (``S_up = star_k``, ``S_down = 1/star_k``), so the geometric
        (physics) prior survives at that level; the fully combinatorial variant builds the complex with
        ``star='unit'`` (all reference stars 1) or uses ``scaling='none'``.  The default model is the pure DEC
        operator (``H = star``) at initialisation.
    readout: ``'node_scalar' | 'node_vector' | 'cochain:k' | 'even:k' | 'grad' | 'curl' | 'div' | 'div:k' |
        'mdiv' | 'mdiv:k'`` (see ``rhmp.readout``; ``grad``/``curl``/``div`` satisfy ``d d = 0`` exactly).
    out_dim: output channels (number of vector fields for ``node_vector``).
    vector_mode: ``'ls' | 'direct'`` (``node_vector`` only).
    checkpoint_layers: activation checkpointing per layer (training memory).
    amp: run the dense MLPs under bf16 autocast on CUDA (sparse products stay fp32).
    connection_odd: connection columns also enter the ordinary odd path (non-Abelian connections).
    lift_hidden: hidden width of the even lifting gates ``e_k``.
    fused: fused polynomial-block kernels (``False`` = plain autograd reference path, same maths).
    layers: per-layer types (``'poly' | 'resolvent' | 'solve' | 'project'``), e.g. ``['poly', 'poly', 'resolvent',
        'poly']`` (DESIGN §9.2) or ``['project', 'poly', 'poly', 'poly']``; ``None`` means
        ``['poly'] * n_layers``.  When given it defines the depth (``n_layers`` is set to ``len(layers)``).
    resolvent_iters: CG iterations of resolvent layers (default 20).  The solve acts identically on every
        channel and commutes with channel mixing (``M^{-1}(x Q) = (M^{-1} x) Q``), so no equivariant channel
        reduction lowers its cost; fewer iterations or warm starts do.
    resolvent_warm_start: start each resolvent layer's CG from the previous resolvent layer's solution of the same
        degree (same shape), solving only the correction (default True; exact implicit gradients).
    learn_metric: ``False`` removes all metric heads: ``H_m = star_m exp(ref_m)`` exactly (tied or untied, down
        blocks use the inverse; tensor metric ``b = 1, a = 0`` = the Galerkin/Whitney star, times the reference
        scaling).  The model is then the pure DEC/FEEC physics prior with only scalar polynomial, cross, gate,
        lifting and readout parameters (the ``dec_fixed`` baseline).  ``identity_metric=True`` takes precedence
        (``H = 1``).
    latent_dims: ``{k: r}``: ``r`` learnable per-cell input columns ``Z_k`` ``(n_k, r)`` on degree ``k`` (a learned
        hidden field, e.g. a velocity 1-form the data do not provide), initialised ``N(0, 0.1^2)`` and inserted
        before the even columns of ``inputs[k]`` (layout ``[connection | odd | latent | even]``): odd cochains for
        ``k >= 1`` (they enter the odd path and the lifting gates, never the metric heads), vertex columns for
        ``k = 0``.  These are the only mesh-sized parameters, so they tie the model to *one* mesh (shared-mesh
        tasks; a different ``n_k`` or a block-diagonal batch raises).  They are created by ``model.init_latents(K)``
        (call it before building the optimizer) or lazily, with a warning, by the first forward; checkpoints store
        them.
        All symmetry statements hold conditional on the latent field.
    material_dims: ``{k: m}``: the last ``m`` even columns of ``inputs[k]`` are *material* columns (e.g.
        ``log sigma``): they feed *only* the metric heads (``psi`` of the degree-k metric, the tensor-metric
        descriptors) and ``metric_reference``, never the lifting, a gate or any feature path, so the only route
        from the material to the output is the metric ``H``.
    lifting: ``'mlp'`` (default) or ``'linear'`` (``x_0 = W f_0``, ``x_k = Lin(d x_{k-1}) + Lin(f_k)``: no geometry,
        biases or even gates; linear in the field inputs).
    resolvent_normalize: ``False``: resolvent layers use the un-normalised physical operator
        ``(I + tau L^2 Delta_H)^{-1}`` (identifies the metric's magnitude; see ``RHMPLayer``).
    solve_iters, solve_tol, solve_bc: ``'solve'`` layers (``layers=[..., 'solve', ...]``): CG budget (the CG
        count grows like the square root of the mesh condition number; warm starts reuse the previous solve layer)
        and boundary condition (``'dirichlet'``: ``K.meta['dirichlet'][k]`` or ``K.boundary[k]`` fixed; ``'none'``:
        no fixed cells; ``'neumann'``: no fixed cells and vertex solves return the zero-mean (lumped mass)
        solution of the pure Neumann problem, as in the T5 generator; ``'lift'``: the fixed cells of
        ``'dirichlet'`` keep the layer input, i.e. inhomogeneous Dirichlet data such as electrode potentials).
        A solve layer computes ``y_k = (Delta_H + lam / L^2)^{-1} x_k`` with the un-normalised metric Hodge
        Laplacian ``Delta_H`` (FEM stiffness / lumped mass for k = 0) and ``lam = softplus(log_lam)`` (init 1e-3,
        ``L^D`` = domain measure).  See :meth:`solver_preset`.
    readout_head: ``'mlp'`` (default) or ``'linear'``: the potential head of the ``grad`` readout
        (``phi = MLP(x_0)`` or ``phi = Lin_nobias(x_0)``); the solver preset uses ``'linear'`` so that the model
        stays linear in the field inputs.
    term_gates: per-term gates in polynomial layers (``rhmp.layers.TermGate``): ``m_k = sum_j c_j g_j t_j`` over
        the powers ``L^p x_k`` and the cross terms, with per-cell factors ``g_j`` from scale-invariant
        invariants of the term vectors (default ``False``: the plain polynomial message).
    cochain_head: ``'linear'`` (default), ``'gated'`` or ``'context'``: head of ``cochain:k`` readouts
        (``'gated'``: odd linear map times an even gate from invariants and the even inputs of degree k;
        ``'context'``: the gated head with the coboundary of the degree ``k-1`` features in the odd part and the
        vertex mean of the vertex features in the gate; ``rhmp.readout.CochainReadout``).
    readout_hidden: hidden width of the gated cochain heads.
    lift_context: ``p > 0``: the even lifting gates of degrees ``k >= 1`` also see the vertex mean over the cell of a
        projection ``P x_0`` (``C -> p``) of the lifted vertex features (default 0: off; ``rhmp.lifting``).
    lift_vertex: the vertex MLP of the lifting also sees the DEC codifferential of the odd edge inputs and the
        vertex means of their squares and of the even edge inputs (default False; ``rhmp.lifting``).
    readout_depth: number of odd gated stages of the gated / context cochain heads (default 1).
    readout_spatial: the further stages of a deep gated / context head (``readout_depth >= 2``) also mix the
        neighbouring k-cells through the combinatorial Hodge Laplacian of degree k (default False).
    readout_skip: the gated / context head of ``cochain:k`` also reads the odd input columns of degree ``k`` and the
        coboundary ``d_{k-1} f_{k-1}^odd`` of the odd input columns of degree ``k-1`` through bias-free linear maps
        (a linear skip from the inputs, e.g. the circulation of the velocity 1-form on faces; default False).
    readout_wedge: ``h > 0``: the gated / context head of a ``cochain:2`` readout also reads ``h`` wedge products
        ``(x_1 A_j) ^ (x_1 B_j)`` of two projections of the final edge features (metric-free cup product of
        1-cochains on the faces; a true 2-cochain, so the output keeps its orientation parity), e.g. the
        advection ``theta ^ delta(d theta)`` of a vorticity 2-form; default 0: off (``rhmp.readout.face_wedge``).
    readout_layers: ``'last'`` (default) or ``'all'``: ``cochain:k`` readouts read the features of degree k after
        the lifting and after every layer, each through its own linear map (``u = sum_l W_l x_k^(l)``), so that
        different output channels combine different filters of the stack (``rhmp.readout.CochainReadout``).
    amplitude_dims: ``{k: n}``: the first ``n`` columns of ``inputs[k]`` scale with one field amplitude (e.g. the
        velocity 1-form and its even companion): they are divided by their per-sample RMS ``s`` (over all such
        columns and cells; per graph in block-diagonal batches) and the output is multiplied by ``s``, so the model
        is exactly homogeneous of degree one in that amplitude (the other inputs, e.g. densities, are not
        scaled).  ``s`` is invariant under orientation changes, relabelling, E(n) and O(C), and per sample.
    amplitude_power: the output is multiplied by ``s^p`` (default 1); ``p = 2``: homogeneous of degree two in the
        amplitude columns (e.g. a pressure for a velocity 1-form: dynamic similarity); ``p = 0``: the output is
        not rescaled (a nondimensional output, e.g. a pressure coefficient).
    amplitude_energy: the output factor ``s`` is the field magnitude of the amplitude 1-form columns of degree 1,
        ``s_E = sqrt(sum_f theta_f^T M_f theta_f / sum_f |f|)`` (Galerkin energy density over the domain; exactly
        ``|u|`` for the 1-form of a uniform field ``u`` on any simplicial mesh), instead of the RMS of the cochain
        values, which scales with the edge lengths; the inputs are still divided by the RMS.  The same energy
        density (of the normalised columns) is the reference of energy readouts and projection layers
        (``rhmp.layers.reference_energy``).  Needs ``amplitude_dims[1] > 0`` and Whitney blocks (default False).
    readout_energy: ``h > 0``: the ``node_scalar`` readout also reads ``h`` Galerkin energy densities
        ``<x_1 A_j, x_1 B_j>`` of projections of the final edge features at the vertices (symmetric quadratic forms
        in the reference metric; ``rhmp.readout.NodeScalarReadout``); default 0: off.
    readout_energy_clip: ``c > 0``: the energy channels are bounded, ``q -> c tanh(q / c)`` (default 0: off).
    project_harmonic: ``'project'`` layers (``layers=[..., 'project', ...]``: metric Hodge projection of the edge
        features with the learned edge metric, ``rhmp.layers.RHMPLayer._project_forward``) also remove the
        circulation along the cohomology generators ``K.meta['cohomology'][1]`` in a learned Kutta metric
        (default True; no effect on complexes without generators).
    kutta_range: bound ``a`` of the log Kutta weights of projection layers (default 8).
    kutta_corner: ``s > 0``: prior of the Kutta weights at re-entrant corners of the boundary,
        ``kappa = a tanh(MLP + s c_e)`` with a learnable slope initialised at ``s`` and the corner indicator
        ``c_e`` of ``rhmp.layers.boundary_corner`` (default 0: the weights are learned from a uniform start).
    project_solver: solver of projection layers: ``'cg'`` (default; preconditioned CG with ``solve_iters``,
        ``solve_tol``, ``solve_precond``) or ``'direct'`` (sparse LU per graph and sample on the CPU in float64 with
        adjoint gradients; exact, diagonal metrics only) or ``'dense'`` (batched dense Cholesky on the device in
        float64, one factorisation per sample; exact, diagonal and tensor metrics, one graph per batch:
        :func:`rhmp.layers.dense_projection`).
    solve_precond: CG preconditioner of solve layers: ``'none'`` (Jacobi, default) or ``'twolevel'`` (Jacobi plus
        an additive coarse correction on per-graph spatial aggregates of about ``rhmp.layers.COARSE_AGG_SIZE``
        vertices, assembled per sample and graph; degree 0 only, other degrees keep Jacobi).  Same solution, far
        fewer iterations on large meshes; O(C)-equivariant and per sample.
    tensor_param: tensor-metric parameterisation: ``'full'`` (default) ``sigma_f = b_f expm(sum_j s_j t_j t_j^T)``
        with signed bounded ``s`` (every SPD tensor with bounded condition number, incl. anisotropy misaligned with
        the edges) or ``'cone'`` (``b_f I + sum_j a_j t_j t_j^T``, ``a >= 0``: M-matrix stiffness only, the same
        class as the diagonal metric on triangles).  Configurations saved without a ``tensor_param`` key (with
        ``metric_type='tensor'``) load as ``'cone'``, the parameterisation they were trained with.
    resolvent_grad: ``'implicit'`` (adjoint CG solve, memory independent of the iterations; default) or
        ``'unrolled'`` (autograd through the checkpointed iterations; memory grows with ``resolvent_iters``).
    metric_type: ``'diag'`` (diagonal metrics, default) or ``'tensor'`` (Whitney/Galerkin material-tensor metric
        on the intermediate degrees for up blocks and cross-up terms, DESIGN §9.1).
    metric_reference: ``{m: column}``: absolute index of an *even* column of ``inputs[m]`` holding a log-scale
        material reference ``ref_m`` (e.g. ``log t^T Sigma t`` on edges).  It is a fixed offset of the log-metric,
        ``H_m = star_m exp(ref_m) exp(a tanh(MLP))`` (down blocks use the inverse), so the bounded correction is
        relative to the known coefficient.  Tensor metrics scale the material tensor instead: ``b_f`` by
        ``exp(mean of ref over the m-cells of f)`` and, for the edge metric, ``a_{f,j}`` by ``exp(ref of direction
        edge j)``.  Ignored with ``identity_metric``.
    transport: transport term of the vertices (``rhmp.transport``; empty = off, the default).  Keys: ``inputs``
        (odd columns of ``inputs[1]`` that are transport fields, e.g. a velocity one-form), ``latent`` (number of
        latent edge columns used as fields), ``hidden`` (0-2 O(C)-invariant odd contractions of the hidden state),
        ``forms`` (subset of ``cons, adv, upwind``), ``order`` (power ``P`` of the transport operator per layer),
        ``norm`` (``'metric'``: fluxes ``H_1 phi`` with the Gershgorin-normalised metric, the field keeps its
        magnitude; ``'sample'``: every field operator normalised per sample), ``metric`` (``'layer'``: the
        layer's learned ``H_1``; ``'star'``: the reference star).  Each poly / resolvent layer adds
        ``sum_p g_p T^p x_0`` to the vertex message, ``T = sum_j a_j C(F_j) + b_j A(F_j) + c_j U(F_j)``, with
        scalar coefficients (init 0), conservative ``C``, advective ``A = C^T`` and upwind ``U`` operators of the
        fluxes ``F_j = H_1 phi_j / beta``.  Solve and projection layers carry no transport term.
    frozen_units: the per-mesh geometric normalisers (geometric means of the reference stars, medians behind the
        relative descriptors, unit-metric bounds of the Hodge blocks) are those of the training mesh, stored in the
        model by :meth:`RHMP.freeze_units` (the trainer calls it on the training complex), instead of being
        recomputed on every mesh: on another mesh the stars are divided by the training geometric means, the
        descriptors become ``log(x / median_train)`` and each block bound is ``beta(H) * beta_unit(train) /
        beta_unit(mesh)``; tensor-metric blocks use ``rho * beta_unit(train)``, and resolvent layers and transport
        fluxes the same block bounds.  On the training mesh the model is unchanged; it is no longer invariant to a
        global change of length unit (default False; diagonal and tensor metrics, 'dec' / 'none' scaling,
        polynomial and normalised resolvent layers, transport norm 'metric').
    frozen_descriptors: with ``frozen_units``, also express the relative descriptors against the training medians
        (default True).  False: only the operator normalisation is that of the training mesh (star means and block
        bounds); the descriptors stay ``log(x / median_mesh)`` on every mesh.  ``'auto'``: decided per relative
        column and mesh: training-mesh units if the mesh's median of the column, in training-mesh units, lies in
        the [min, max] of that column on the training mesh, else relative to the mesh's own median (star means and
        block bounds always in training-mesh units).  Ignored without frozen references.
    solve_picard: number ``J`` of fixed-point (Picard) iterations of the vertex solve of ``'solve'`` layers with a
        state-dependent metric (0, the default: one solve).  Iteration ``j`` predicts the vertex-block metric
        (``H_1``, diagonal or tensor) from the usual descriptors plus the field-energy invariant of the previous
        iterate (``y_0 = 0``) and solves with the layer input as right-hand side; ``y_j = y_{j-1} + omega (S(H_j) x
        - y_{j-1})`` for ``j >= 2``, weight-tied, learnable ``omega`` (see ``RHMPLayer``).  For nonlinear
        constitutive laws (e.g. saturable magnetic materials, ``nu(|B|)``).
    picard_relax: initial relaxation ``omega`` in (0, 1) of the Picard iteration.
    metric_geo: ``False``: the metric heads see no geometric descriptors (material columns, feature invariants and
        the Picard energy invariant only), so the learned metric is a function of material and state alone (a
        constitutive law without mesh-specific position cues).  Default ``True``.
    picard_init: ``'zero'`` (default: the first Picard iteration starts from ``y_0 = 0``) or ``'learned'`` (a
        learnable uniform starting energy invariant; see ``RHMPLayer``).
```

</details>

#### classmethod `RHMPConfig.solver_preset`

```python
solver_preset(**kw: Any) -> RHMPConfig
```
A model that is linear in the field inputs and whose only nonlinearity is the metric map.

<details><summary>docstring</summary>

```text
``lifting='linear'``, ``layers=['solve']``, ``gate='none'`` (the solve layer returns ``y = (Delta_H +
lam / L^2)^{-1} x``), ``cross=False``, linear readout ``'cochain:0'`` (or any linear readout given in ``kw``),
``C=4``, ``readout_head='linear'`` (``grad``: ``E = d_0 Lin_nobias(x_0)``); everything can be overridden through
``kw`` (``in_dims`` is required).

For P1 finite-element data of ``-div(sigma grad u) = f`` (lumped-mass right-hand side,
homogeneous Dirichlet boundary) this class contains the exact discrete solution operator: with
``metric_type='tensor'`` (``H_1`` = Whitney/Galerkin star, ``b_f = sigma_f``: e.g. ``sigma`` as a material
column on the top degree with ``metric_reference`` and ``learn_metric=False``, or learned), and on triangle
meshes whose P1 stiffness is an M-matrix (non-negative cotan-type edge weights) also with the diagonal metric
(``H_1`` = the P1 edge weights).  With ``learn_metric=True`` the metric heads learn the material map.
```

</details>

#### `RHMPConfig.layer_types` (property)

Per-layer types (`['poly'] * n_layers` unless `layers` is given).

#### `RHMPConfig.to_dict`

```python
to_dict() -> dict[str, Any]
```
Plain (JSON- and `torch.save`-friendly) dict; `from_dict` inverts it.

#### classmethod `RHMPConfig.from_dict`

```python
from_dict(d: dict[str, Any]) -> RHMPConfig
```
Inverse of `to_dict` (accepts string degree keys, e.g. after a JSON round trip).

<details><summary>docstring</summary>

```text
Raises:
    ValueError: on unknown keys.
```

</details>

### class `RHMP`

```python
RHMP(cfg: RHMPConfig | dict, geo_dims: dict[int, int]) -> None
```
Bases: `Module`.

Riemannian Hodge Message Passing network (v2).

<details><summary>docstring</summary>

```text
Args:
    cfg: :class:`RHMPConfig` (or its dict).
    geo_dims: ``{k: G_k}`` descriptor widths of the complexes the model will see (``K.geo_dims``); the keys
        ``0..K`` also fix the top degree ``K``.
```

</details>

#### `RHMP.init_latents`

```python
init_latents(K: Any) -> RHMP
```
Create (or check) the learnable latent input fields of `cfg.latent_dims` for the mesh of `K`.

<details><summary>docstring</summary>

```text
Idempotent; call it before building the optimizer.  Latents are tied to one mesh.

Raises:
    ValueError: if ``K`` is a block-diagonal batch of several meshes, or if existing latents were sized for a
        different number of cells.
```

</details>

#### `RHMP.load_state_dict`

```python
load_state_dict(state_dict, strict: bool = True, assign: bool = False)
```
Copy parameters and buffers from `state_dict` into this module and its descendants.

<details><summary>docstring</summary>

```text
If :attr:`strict` is ``True``, then
the keys of :attr:`state_dict` must exactly match the keys returned
by this module's :meth:`~torch.nn.Module.state_dict` function.

.. warning::
    If :attr:`assign` is ``True`` the optimizer must be created after
    the call to :attr:`load_state_dict` unless
    :func:`~torch.__future__.get_swap_module_params_on_conversion` is ``True``.

Args:
    state_dict (dict): a dict containing parameters and
        persistent buffers.
    strict (bool, optional): whether to strictly enforce that the keys
        in :attr:`state_dict` match the keys returned by this module's
        :meth:`~torch.nn.Module.state_dict` function. Default: ``True``
    assign (bool, optional): When set to ``False``, the properties of the tensors
        in the current module are preserved whereas setting it to ``True`` preserves
        properties of the Tensors in the state dict. The only
        exception is the ``requires_grad`` field of :class:`~torch.nn.Parameter`
        for which the value from the module is preserved. Default: ``False``

Returns:
    ``NamedTuple`` with ``missing_keys`` and ``unexpected_keys`` fields:
        * ``missing_keys`` is a list of str containing any keys that are expected
            by this module but missing from the provided ``state_dict``.
        * ``unexpected_keys`` is a list of str containing the keys that are not
            expected by this module but present in the provided ``state_dict``.

Note:
    If a parameter or buffer is registered as ``None`` and its corresponding key
    exists in :attr:`state_dict`, :meth:`load_state_dict` will raise a
    ``RuntimeError``.
```

</details>

#### `RHMP.freeze_units`

```python
freeze_units(K: Any) -> RHMP
```
Store the per-mesh geometric normalisers of `K` (the training mesh, one complex) as buffers (`frozen_log_gm`, `frozen_geo_<k>`, `frozen_beta`; the descriptor ranges `frozen_geo_min_<k>` / `frozen_geo_max_<k>` for `frozen_descriptors='auto'`) and use them on every mesh from now on (`RHMPConfig.frozen_units`; `rhmp.layers.unit_reference`).  The output on `K` itself is unchanged, so a model trained without the option can be evaluated with it by calling this on its training mesh.

<details><summary>docstring</summary>

```text
Returns:
    ``self``.
```

</details>

#### `RHMP.frozen_reference`

```python
frozen_reference() -> dict | None
```
The stored training-mesh normalisers in the form of `rhmp.layers.make_context` (`frozen`), or `None` if `freeze_units` has not been called (and no checkpoint carried them).  Without `cfg.frozen_descriptors` the descriptor medians are left out (`geo_log_median=None`); with `'auto'` the descriptor ranges of the training mesh are added (`geo_range`).

#### staticmethod `RHMP.geo_dims_of`

```python
geo_dims_of(K: Any) -> dict[int, int]
```
`{k: G_k}` of a complex (constructor argument).

#### `RHMP.output_degree` (property)

Degree of the cells the output lives on (0 node readouts, 1 `grad`, 2 `curl`, k-1 `div:k`, ...).

#### `RHMP.feature_degree` (property)

Degree of the hidden cochains the readout reads.

#### `RHMP.num_parameters`

```python
num_parameters() -> int
```
Number of trainable scalars.

#### `RHMP.forward`

```python
forward(inputs: dict[int, Tensor], K: Any) -> Tensor
```
Predict.

<details><summary>docstring</summary>

```text
Args:
    inputs: ``{k: (n_k, B, F_k)}`` for every degree with ``cfg.in_dims[k] > 0``.
    K: ``CochainComplex`` on the model's device and dtype.

Returns:
    ``(n_out, B, out_dim)`` (``(n_0, B, out_dim * D)`` for ``node_vector``), in the model dtype.
```

</details>

#### `RHMP.lift`

```python
lift(inputs: dict[int, Tensor], K: Any) -> dict[int, Tensor]
```
Lifting only: `{k: (n_k, B, C)}`.

#### `RHMP.propagate`

```python
propagate(
    x: dict[int, Tensor],
    K: Any,
    inputs: dict[int, Tensor] | None = None,
) -> dict[int, Tensor]
```
Message-passing stack only (O(C)-equivariant): `{k: (n_k, B, C)} -> {k: (n_k, B, C)}`.

<details><summary>docstring</summary>

```text
Args:
    x: hidden cochains for every degree ``0..K``.
    K: complex.
    inputs: raw inputs (only their even columns are used, by the metric heads); required if the config
        has even columns.
```

</details>

#### `RHMP.metric_fields`

```python
metric_fields(inputs: dict[int, Tensor], K: Any) -> list[dict]
```
Learned metrics of every layer for these inputs (public accessor for analysis scripts).

<details><summary>docstring</summary>

```text
Returns:
    One dict per layer (see ``RHMPLayer.metric_fields``): ``{'log_ratio': {m: (n_m, B)}, 'phi': {m: (n_m, B)},
    'tensor': {km: (b (n_top, B), a (n_top, B, m))}, 'sigma': {km: (n_top, B, D, D)}}``.
```

</details>

#### `RHMP.projection_fields`

```python
projection_fields(
    inputs: dict[int, Tensor],
    K: Any,
    probe: Tensor | None = None,
    layer: int | None = None,
) -> dict
```
What a projection layer does to a probe 1-cochain, with the metrics it computes for these inputs (analysis: the learned potential flow, its circulation, the Kutta weights).

<details><summary>docstring</summary>

```text
Args:
    inputs, K: as for :meth:`forward`.
    probe: ``(n1, B, P)`` cochains (default: the amplitude columns of ``inputs[1]`` after the amplitude
        normalisation, i.e. the unit inflow 1-form).
    layer: index of the projection layer (default: the first one).
Returns:
    dict with ``u (n1, B, P)`` (probe projected, circulation removed), ``u_no_circulation`` (projection only),
    ``alpha (n0, B, P)``, ``Gamma (S, B, r, P)``, ``gamma (n1, B, r)`` (``None`` without generators),
    ``kappa (n1, B)`` (log Kutta weight relative to ``star1``), ``log_H_ratio (n1, B)`` (learned log edge
    metric relative to the normalised reference star; ``None`` for tensor metrics), ``probe`` and
    ``amplitude`` (the output amplitude ``(S, B)`` or ``None``).
```

</details>

#### `RHMP.operator_residual`

```python
operator_residual(
    inputs: dict[int, Tensor],
    K: Any,
    u: Tensor,
    f: Tensor,
    k: int = 0,
    layer: int = -1,
    bc: str | None = None,
) -> Tensor
```
Operator-identification residual `|| S_H u - M f ||^2 / || M f ||^2` per sample (auxiliary PDE loss).

<details><summary>docstring</summary>

```text
``S_H`` is the weak (FEM) metric Hodge stiffness of degree ``k`` built from the metrics that layer ``layer``
uses on these inputs (``d_0^T H_1 d_0`` for k = 0; ``+ star_k d_{k-1} H^down d_{k-1}^T star_k`` for k >= 1;
tensor metrics through the Whitney blocks), in physical units, and ``M = star_k`` the lumped mass.  Rows of
fixed cells are excluded under ``bc='dirichlet'`` (``K.meta['dirichlet'][k]`` or ``K.boundary[k]``; default
``cfg.solve_bc``).  For the diagonal metric the residual is linear in ``H``, so the loss is convex in the metric
(convex identification of the material from (u, f) pairs).

Args:
    inputs: model inputs (as for :meth:`forward`).
    K: complex (``scaling='dec'`` models).
    u: solution cochain ``(n_k, B)`` or ``(n_k, B, 1)`` (physical units, incl. its boundary values).
    f: source ``(n_k, B)`` or ``(n_k, B, 1)`` (physical units).
    k: degree of the equation.
    layer: index of the layer whose metric is used (default: the last layer).
    bc: ``'dirichlet' | 'none' | 'neumann'`` (default ``cfg.solve_bc``; only ``'dirichlet'`` excludes rows).

Returns:
    ``(1, B)`` (single complex) or ``(num_graphs, B)``; differentiable w.r.t. the model parameters.
```

</details>

#### `RHMP.hidden`

```python
hidden(inputs: dict[int, Tensor], K: Any) -> dict[int, Tensor]
```
Pre-readout features `{k: (n_k, B, C)}` (lifting + message passing).

#### `RHMP.fit_linear_readout_`

```python
fit_linear_readout_(
    inputs: dict[int, Tensor],
    K: Any,
    target: Tensor,
    output_map: Any = None,
    rtol: float = 1e-08,
) -> float
```
Least-squares weights of a linear readout from one batch (in place; minimum-norm, channels may be collinear).

<details><summary>docstring</summary>

```text
Linear readouts: ``cochain:k`` (``h_k W^T``), ``grad`` with ``readout_head='linear'`` (``d_0 (h_0 W^T)``),
``curl`` (``d_1 (h_1 W^T)``), ``div[:k]`` (``d_{k-1}^T (h_k W^T)``), ``mdiv[:k]`` (``S^{-1} d_{k-1}^T (h_k W^T)``);
the output is ``F W^T`` with the features ``F = Op(h)``.  With a (linear) task ``output_map`` the fit is made
in the target space through the composed map ``output_map o Op`` (one output column).  The solver class
(``solver_preset``) is exact up to the overall gain of lifting x readout; this initialises that gain from data
(the trainer does it for ``--solver-mode``).

Args:
    inputs, K: one batch (as for :meth:`forward`); target: model output units ``(n_out, B, out_dim)`` or, with
        ``output_map``, the task's target ``(n_t, B, O_t)``.
    output_map: optional ``rhmp.data.OutputMap`` applied to the model output.
    rtol: relative singular-value cut-off of the pseudo-inverse.

Returns:
    Relative residual of the fit.
```

</details>

#### `RHMP.diagnostics` (property)

Statistics of the last forward pass (with `record_diagnostics = True`), as python floats.

<details><summary>docstring</summary>

```text
One key scheme, ``layer{l}.<name>``:

* ``layer{l}.H{m}.<stat>`` for the degree-m metric (untied metrics: ``H{m}.up_<stat>`` / ``H{m}.down_<stat>``)
  with ``<stat>`` in ``mean, std, min, max`` (of the learned bounded part ``phi = log(H/(star exp(ref)))``),
  ``cond_learned`` (max over samples of ``max exp(phi) / min exp(phi)``: 1 at initialisation),
  ``cond_total`` (max over samples of ``max H / min H`` including the reference star),
  ``clamp_fraction`` (fraction of cells and samples with ``|tanh| > 0.99``) and ``sat`` (``|phi| > 0.95 a``);
* ``layer{l}.H{m}.tensor_<stat>`` for tensor metrics: ``logb_mean, aniso_mean, aniso_max`` (eigenvalue ratio of
  the learned ``sigma_f``, tangent plane on surfaces), ``clamp_b`` and ``s_absmax, clamp_s``
  (``tensor_param='full'``) or ``a_mean, a_max, clamp_a`` (``'cone'``);
* ``layer{l}.beta_up{k}`` / ``layer{l}.beta_down{k}``: mean Gershgorin normaliser of each block;
* resolvent layers: ``layer{l}.tau_up{k}`` / ``layer{l}.tau_down{k}`` and ``layer{l}.cg_res{k}`` (final relative
  CG residual, max over samples);
* solve layers: ``layer{l}.solve_res{k}`` (final relative CG residual, max over samples and graphs),
  ``layer{l}.solve_it{k}`` (CG iterations run) and ``layer{l}.lam{k}``.

Calling it synchronises the device.
```

</details>

#### `RHMP.to_checkpoint`

```python
to_checkpoint() -> dict[str, Any]
```
`{'cfg': cfg.to_dict(), 'geo_dims': {k: G_k}, 'state_dict': state_dict}` for `torch.save`.

#### classmethod `RHMP.from_checkpoint`

```python
from_checkpoint(
    ckpt: dict | str | os.PathLike,
    map_location: Any = None,
    strict: bool = True,
    weights_only: bool | None = None,
) -> RHMP
```
Rebuild a model from a checkpoint (dict or path) and return it.

<details><summary>docstring</summary>

```text
Accepted formats:

* :meth:`to_checkpoint` output ``{'cfg', 'geo_dims', 'state_dict'}``;
* the trainer's ``best.pt`` (the same keys plus epoch / validation / task metadata);
* the trainer's ``last.pt`` (the model under the ``'model'`` key next to optimizer, scheduler and RNG state).

Args:
    ckpt: checkpoint dict or path.
    map_location: device for loading (the returned model is moved there too).
    strict: ``load_state_dict`` strictness.
    weights_only: ``torch.load`` mode for paths.  ``None`` (default) first tries the safe ``weights_only=True``
        and falls back to full unpickling for files holding non-tensor state (e.g. the numpy RNG state of a
        trainer ``last.pt``): only load checkpoint files you trust.

Raises:
    ValueError: if the file does not contain an RHMP model.
```

</details>


## DEC toolkit: `rhmp.dec`

Galerkin metrics, CG solvers, Hodge Laplacians/decomposition, Whitney maps.

<details><summary>module docstring</summary>

```text
Discrete exterior calculus toolkit on cochain complexes (DESIGN §9.1, §9.3).

* Galerkin (Whitney) tensor metrics:
  :func:`whitney_blocks`, :func:`apply_whitney_metric`, :func:`whitney_rowsum_abs`, :func:`assemble_whitney_metric`.
  For a top cell f with material tensor ``sigma_f = b_f I + sum_j a_fj t_j t_j^T`` the degree-k metric block is
  ``M_f = b_f G0_f + sum_j a_fj Gk_fj`` (precomputed PSD blocks ``K.whitney[k]``); ``H_k = sum_f P_f^T M_f P_f``.
* Solvers: :func:`cg_solve` (batched CG on ``(n, B, C)``, per-sample Frobenius inner products -> O(C)-equivariant,
  unrolled autograd) and :func:`cg_solve_implicit` (adjoint-solve gradients).
* Metric Hodge Laplacians and decomposition: :func:`hodge_laplacian`, :func:`hodge_decompose`, :func:`harmonic_basis`;
  closed generators of the first cohomology of planar domains with holes: :func:`cohomology_generators`, and their
  harmonic representatives :func:`harmonic_generators`.
* Galerkin energies of 1-cochains: :func:`whitney_energy` (per top cell) and :func:`vertex_energy` (density at the
  vertices), exact for the cochains of constant fields.
* Whitney interpolation: :func:`sharp` (1-cochain -> vector field) and :func:`flat` (vector field -> 1-cochain).

Metric arguments ``H`` (Laplacians / decomposition) are indexed by degree (list, tuple or dict); an entry is a
diagonal metric ``(n_j,)`` or ``(n_j, B)``, a callable ``x -> H_j x`` (e.g. a Whitney metric), or ``None``
(-> ``K.star[j]``).  Inverse metrics are needed only on the "down" side and must be diagonal (lumped).
```

</details>

### `whitney_blocks`

```python
whitney_blocks(K, k: int, b: Tensor, a: Tensor | None) -> Tensor
```
Per-top-cell metric blocks `M_f = b_f G0_f + sum_j a_fj Gk_fj`.

<details><summary>docstring</summary>

```text
Computed with elementwise operations in a fixed order, so the block of sample ``b`` is bitwise independent of
the other samples.

Args:
    K: complex with ``K.whitney[k]``.
    k: metric degree (1: edges; 2: faces of tets).
    b: ``(n_top, B)`` or ``(n_top,)``, positive.
    a: ``(n_top, B, m)``, ``(n_top, m)`` or ``None``; nonnegative; ``m = K.whitney[k]['t'].shape[1]`` directions.
Returns:
    ``(n_top, B, m_k, m_k)`` blocks in the canonically oriented basis of ``K.whitney[k]['cells']``.
```

</details>

### `apply_whitney_metric`

```python
apply_whitney_metric(K, k: int, b: Tensor, a: Tensor | None, x: Tensor) -> Tensor
```
`H_k(b, a) x`: gather the `m_k` cochain values of every top cell, one `m_k x m_k` block product per (cell, sample), scatter back.  Differentiable in `b`, `a` and `x` (first order); cost `O(n_top m_k^2 B C)`; the forward result of every sample is bitwise independent of the other samples; only `x` and the blocks are kept for backward.

<details><summary>docstring</summary>

```text
Args:
    K: complex with ``K.whitney[k]``; k: metric degree.
    b: ``(n_top, B)`` (or ``(n_top, 1)`` / ``(n_top,)`` shared by all samples); a: ``(n_top, B, m)`` or ``None``.
    x: ``(n_k, B, C)`` contiguous.
Returns:
    ``(n_k, B, C)``.
```

</details>

### `whitney_rowsum_abs`

```python
whitney_rowsum_abs(K, k: int, b: Tensor, a: Tensor | None) -> Tensor
```
Row sums of `|H_k(b, a)|` bounded blockwise: `sum_f sum_j |M_f|_ij` scattered to the k-cells.

<details><summary>docstring</summary>

```text
An upper bound of the exact row sums (triangle inequality), hence valid in Gershgorin bounds; differentiable.

Args:
    K, k, b, a: as in :func:`whitney_blocks`.
Returns:
    ``(n_k, B)``.
```

</details>

### `assemble_whitney_metric`

```python
assemble_whitney_metric(K, k: int, b: Tensor | None = None, a: Tensor | None = None) -> Tensor
```
Assembled sparse `H_k` for one sample (diagnostics, tests, direct solvers).

<details><summary>docstring</summary>

```text
Args:
    K, k: as above.
    b: ``(n_top,)`` (default 1); a: ``(n_top, m)`` (default 0).
Returns:
    CSR ``(n_k, n_k)`` (dtype of ``b``, float32 by default).
```

</details>

### `cg_solve`

```python
cg_solve(
    matvec: Callable[[Tensor], Tensor],
    rhs: Tensor,
    iters: int = 32,
    tol: float = 1e-06,
    x0: Tensor | None = None,
    batch: Tensor | None = None,
    num_graphs: int = 1,
    early_exit: bool = True,
) -> tuple[Tensor, Tensor]
```
Batched conjugate gradients for symmetric positive (semi)definite systems `A x = rhs`.

<details><summary>docstring</summary>

```text
Every sample (column ``b``; and every graph of a block-diagonal batch when ``batch`` is given) runs its own CG
with Frobenius inner products over cells and channels, so the result for one sample never depends on the others,
and ``cg_solve(A, rhs @ R) == cg_solve(A, rhs) @ R`` for orthogonal channel mixings ``R`` (O(C)-equivariance) when
``matvec`` acts identically on every channel.  The system is solved for ``rhs / ||rhs||`` per sample (rescaled at
the end) and a sample is frozen once ``||r|| <= max(tol, 1e-12) ||rhs||``, so every division stays well scaled
and the unrolled backward is finite in fp32 and fp64.  Differentiable by unrolling.

Args:
    matvec: linear, symmetric PSD map ``(n, B, ...) -> (n, B, ...)`` (per sample).
    rhs: ``(n, B)`` or ``(n, B, C)``.
    iters: maximum number of iterations.
    tol: relative residual at which a sample is frozen (floored at ``1e-12``).
    x0: optional initial guess (same shape as ``rhs``).
    batch: optional ``(n,)`` graph id of every row (block-diagonal batch); num_graphs: number of graphs.
    early_exit: stop as soon as every sample is frozen (one host sync per iteration); ``False`` always runs
        ``iters`` iterations without host synchronisation (frozen samples simply stop changing).
Returns:
    ``(x, res)``: solution like ``rhs`` and relative residual norms ``res (T+1, S, B)`` with ``T <= iters`` the
    number of iterations run and ``S = num_graphs`` if ``batch`` is given else 1.
```

</details>

### `cg_solve_implicit`

```python
cg_solve_implicit(
    matvec: Callable[[Tensor], Tensor],
    rhs: Tensor,
    params: Sequence[Tensor] = (),
    iters: int = 32,
    tol: float = 1e-06,
    batch: Tensor | None = None,
    num_graphs: int = 1,
    early_exit: bool = True,
) -> tuple[Tensor, Tensor]
```
`cg_solve` with implicit (adjoint) gradients instead of unrolling: memory `O(1)` in `iters`.

<details><summary>docstring</summary>

```text
Backward solves ``A lam = dL/dx`` with the same CG and returns ``dL/drhs = lam`` and
``dL/dtheta = -lam^T (dA/dtheta) x`` for every tensor in ``params``.

Args:
    matvec: linear, symmetric PSD map (see :func:`cg_solve`), depending on the tensors in ``params``.
    rhs: ``(n, B)`` or ``(n, B, C)``.
    params: tensors used inside ``matvec`` that need gradients (e.g. the metric ``H``).
    iters, tol, batch, num_graphs, early_exit: as in :func:`cg_solve`.
Returns:
    ``(x, res)`` as in :func:`cg_solve` (``res`` is not differentiable).
```

</details>

### `hodge_laplacian`

```python
hodge_laplacian(K, k: int, H=None, weak: bool = True) -> Callable[[Tensor], Tensor]
```
Metric Hodge Laplacian of degree `k` as a function on `(n_k, B, ...)` cochains.

<details><summary>docstring</summary>

```text
``weak=True``: ``L_k = d_k^T H_{k+1} d_k + H_k d_{k-1} H_{k-1}^{-1} d_{k-1}^T H_k`` (symmetric PSD in the Euclidean
inner product; usable with :func:`cg_solve`).  ``weak=False``: ``Delta_k = H_k^{-1} L_k`` (self-adjoint in the
``H_k`` inner product; ``H_k`` must be diagonal).  ``H_{k+1}`` and ``H_k`` may be callables (Galerkin metrics);
``H_{k-1}`` must be diagonal.

Args:
    K: complex; k: degree; H: metrics by degree (default ``K.star``); weak: see above.
Returns:
    ``f(x) -> (n_k, B, ...)``.
```

</details>

### `hodge_decompose`

```python
hodge_decompose(
    K,
    k: int,
    x: Tensor,
    H=None,
    iters: int = 1000,
    tol: float = 1e-10,
) -> tuple[Tensor, Tensor, Tensor]
```
Hodge decomposition `x = d_{k-1} alpha + H_k^{-1} d_k^T gamma + h` (orthogonal in the `H_k` inner product).

<details><summary>docstring</summary>

```text
``alpha`` solves ``d_{k-1}^T H_k d_{k-1} alpha = d_{k-1}^T H_k x`` and ``gamma`` solves
``d_k H_k^{-1} d_k^T gamma = d_k x`` by :func:`cg_solve` (per sample, per graph for block-diagonal batches);
the harmonic part ``h`` is closed (``d_k h = 0``) and co-closed (``d_{k-1}^T H_k h = 0``).

Args:
    K: complex; k: degree; x: ``(n_k, B)`` or ``(n_k, B, C)``.
    H: metrics by degree; only the (diagonal) ``H_k`` matters (default ``K.star[k]``).
    iters, tol: CG controls (relative residual of the two normal equations).
Returns:
    ``(exact, coexact, harmonic)``, each like ``x``.
```

</details>

### `harmonic_basis`

```python
harmonic_basis(
    K,
    k: int,
    H=None,
    n_probe: int = 8,
    iters: int = 2000,
    tol: float = 1e-12,
    rank_tol: float = 1e-06,
    seed: int = 0,
) -> Tensor
```
`H_k`-orthonormal basis of the discrete harmonic k-cochains of a single complex.

<details><summary>docstring</summary>

```text
Random probes are projected onto the harmonic space by :func:`hodge_decompose` (the infinite-shift limit of
inverse iteration on ``L_k``), then orthonormalised in the ``H_k`` inner product with rank detection; the
dimension equals the k-th Betti number as long as it is ``<= n_probe``.

Args:
    K: single complex; k: degree; H: metrics by degree (diagonal ``H_k``, 1-D).
    n_probe: number of random probes (upper bound of the detectable dimension).
    iters, tol: CG controls; rank_tol: relative singular-value threshold; seed: probe seed.
Returns:
    ``(n_k, h)`` float64 basis with ``basis^T diag(H_k) basis = I``.
```

</details>

### `cohomology_generators`

```python
cohomology_generators(K) -> Tensor
```
Closed 1-cochains that generate the first cohomology of a planar 2-complex with holes (an oriented surface of genus 0 with boundary): `(n1, r)` float32 with entries in `{-1, 0, 1}`, `r` = number of boundary loops - 1.

<details><summary>docstring</summary>

```text
For every boundary loop except the longest one (the outer boundary of a planar domain) a shortest chain of faces
to the outer loop is found by a breadth-first search in the dual graph; the generator is the crossing cochain of
that chain, +-1 on the edges it crosses (signs from the face orientations) and 0 elsewhere.  It is exactly closed
(``d1 c = 0`` in integer arithmetic) and sums to +-1 around its loop, so it is not exact; the ``r`` cochains are a
basis of ``H^1`` (the first Betti number of a genus-0 surface with ``r + 1`` boundary loops).  Any metric-harmonic
representative follows by a projection (``c - d0 (d0^T H d0)^+ d0^T H c``).  The sign of a generator and the chain
depend on the cell numbering; the cohomology class does not, up to that sign.

Args:
    K: a single 2-complex with consistently oriented faces (``meta['face_edges']``).
Returns:
    ``(n1, r)`` on the device of ``K`` (``r = 0`` for a disk or a closed surface).
Raises:
    ValueError: for batches, non-manifold edges, inconsistent face orientations or disconnected domains.
```

</details>

### `harmonic_generators`

```python
harmonic_generators(K, H: Tensor | None = None, dtype: torch.dtype = torch.float32) -> Tensor
```
Harmonic representatives of `cohomology_generators`: `gamma = c - d0 psi` with `(d0^T H d0) psi = d0^T H c` on all vertices (natural condition on every boundary), so that `gamma` is closed (to round-off), co-closed (`d0^T H gamma = 0`: no flux through any boundary) and keeps the period +-1 of `c` around its hole: the circulation fields of the domain.  Sparse direct solves in float64 on the CPU.

<details><summary>docstring</summary>

```text
Args:
    K: a single planar 2-complex (see :func:`cohomology_generators`).
    H: diagonal edge metric ``(n1,)`` (default: the reference star ``K.star[1]``).
    dtype: dtype of the result.
Returns:
    ``(n1, r)`` on the device of ``K``.
```

</details>

### `top_measure`

```python
top_measure(K) -> Tensor
```
Areas (triangle complexes) or volumes (tetrahedral complexes) of the top cells, `(n_top,)` float64 (cells of a periodic complex in their own unwrapped frames).

### `top_aspect`

```python
top_aspect(K) -> Tensor
```
Aspect ratio `l_max^d / (d! |f|)` of the top cells (`l_max` the longest edge), `(n_top,)` float64.

### `whitney_energy`

```python
whitney_energy(K, a: Tensor, b: Tensor | None = None) -> Tensor
```
`int_f <a, b>` of the Whitney interpolants of two 1-cochains on every top cell, `e_f = a_f^T G0_f b_f` with the Galerkin mass blocks of `K.whitney[1]` (the reference metric `H_1` of the tensor-metric path).

<details><summary>docstring</summary>

```text
Symmetric, positive for ``b = a``, exact for the cochains of constant fields (``e_f = |f| u . v`` for
``a_e = u . (x_dst - x_src)``, ``b_e = v . (x_dst - x_src)``), invariant under edge orientation flips (both cochains
change sign), relabelling and E(n).

Args:
    K: triangle or tetrahedral complex with ``K.whitney[1]``; a, b: ``(n1, *tail)`` (``b = None``: ``b = a``).
Returns:
    ``(n_top, *tail)``.
```

</details>

### `vertex_energy`

```python
vertex_energy(
    K,
    a: Tensor,
    b: Tensor | None = None,
    measure: Tensor | None = None,
    aspect_cap: float | None = 32.0,
) -> Tensor
```
Galerkin energy density of two 1-cochains at the vertices: the `whitney_energy` of the top cells around vertex `i` divided by their measure, `q_i = sum_{f ∋ i} w_f a_f^T G0_f b_f / sum_{f ∋ i} w_f |f|`.

<details><summary>docstring</summary>

```text
The pointwise inner product ``u . v`` for the cochains of constant fields on any simplicial mesh (the lumped
formula ``sum_{e ∋ i} star1_e a_e b_e / (2 star0_i)`` is not, on graded meshes).  ``w_f = 1`` for cells with an
aspect ratio (:func:`top_aspect`) up to ``aspect_cap`` and ``(aspect_cap / aspect)^2`` beyond (slivers of
Delaunay triangulations of random points; still exact for constant fields; ``None``: no weighting).

Args:
    K: triangle or tetrahedral complex; a, b: ``(n1, *tail)``; measure: :func:`top_measure` if already known.
    aspect_cap: see above (default :data:`ASPECT_CAP`).
Returns:
    ``(n0, *tail)``.
```

</details>

### `sharp`

```python
sharp(K, x1: Tensor, at: str = 'vertex') -> Tensor
```
Whitney interpolation of a 1-cochain into a vector field (exact for constant and rigid-rotation fields).

<details><summary>docstring</summary>

```text
Args:
    K: triangle or tet complex; x1: ``(n1, *tail)`` edge cochain (canonical orientation).
    at: ``'vertex'`` (value of the Whitney field of every incident top cell at the vertex, averaged with the
        cells' areas/volumes as weights) or ``'cell'`` (value at the barycentre of every top cell).
Returns:
    ``(n0, *tail, D)`` or ``(n_top, *tail, D)``.
```

</details>

### `flat`

```python
flat(K, v: Tensor) -> Tensor
```
De Rham map of a vertex vector field: `x_e = (v_src + v_dst)/2 . (p_dst - p_src)` (trapezoidal rule, exact for fields that are linear along every edge).

<details><summary>docstring</summary>

```text
Args:
    K: complex; v: ``(n0, *tail, D)``.
Returns:
    ``(n1, *tail)``.
```

</details>

### Constants

- `ASPECT_CAP`: 32.0


## Geometry: `rhmp.geometry`

E(n)-invariant descriptors, reference Hodge stars, Whitney blocks.

<details><summary>module docstring</summary>

```text
E(n)-invariant geometry of cochain complexes: reference Hodge stars and per-cell descriptors.

Everything here is computed once per complex under ``torch.no_grad()`` in float64 (vectorised with
``index_add``/``scatter_reduce``; no loops over cells) and cached on the complex as float32.
Only intrinsic quantities (lengths, areas, volumes, angles) are used, so every output is invariant
under rotations, reflections and translations of ``pos``.  Scale quantities enter the descriptors
as ``log(x / median_x)`` with the median taken over *this* complex (mesh-transferable, DESIGN §3.1);
the log medians are returned as well (``geo_log_median``, columns :data:`GEO_RELATIVE_COLUMNS`), so that a
model can re-express the descriptors in the units of its training mesh (``RHMPConfig.frozen_units``).

Reference stars (DESIGN §3.1):
  * ``cotan``: ``star0`` barycentric dual area, ``star1 = sum_f w_fe`` with ``w_fe = cot(opposite
    angle)/2`` for triangular faces (``|c_f - m_e| / |e|``, the barycentric half dual edge ratio, for
    polygonal faces; this equals the circumcentric value on rectangles) clamped below at
    ``KAPPA * median``; ``star2 = 1/area``.  Surfaces only.
  * ``barycentric``: ``star1 = |dual edge| / |edge|`` (dual edge = polyline edge-midpoint -> face
    centroids); tets: ``star0`` dual volume, ``star1 = |dual face|/|edge|``, ``star2 = |dual
    edge|/|face|``, ``star3 = 1/vol``.
  * ``unit``: all ones.

Descriptor columns per degree are listed in :data:`GEO_FEATURE_NAMES` (identical for all cell
types so that a model can move between complexes; columns that do not apply are constant).
```

</details>

### `min_image`

```python
min_image(d: Tensor, period) -> Tensor
```
Minimal-image representative of position differences on a flat torus with axis-aligned periods.

<details><summary>docstring</summary>

```text
Args:
    d: ``(..., D)`` differences of vertex positions.
    period: ``(L_1, .., L_P)`` with ``P <= D`` (axes ``P..D-1`` are not periodic).
Returns:
    ``(..., D)``: ``d_i - L_i round(d_i / L_i)`` on the periodic axes (each in ``[-L_i/2, L_i/2]``), ``d_i``
    elsewhere.
```

</details>

### `local_cell_positions`

```python
local_cell_positions(P: Tensor, cells: Tensor, period=None) -> Tensor
```
Vertex positions of every cell, `(n, m, D)`; on a periodic complex in the cell's own (unwrapped) frame.

<details><summary>docstring</summary>

```text
Args:
    P: ``(n0, D)`` positions.
    cells: ``(n, m)`` vertex ids (``-1`` padded slots get the position of the first vertex).
    period: ``None`` or the axis periods of a flat torus (see :func:`min_image`): the first vertex of each cell
        is kept and the others are placed at their minimal images relative to it, so that every cell is a
        contiguous polygon / simplex (intrinsic geometry of the torus).
Returns:
    ``(n, m, D)`` in the dtype of ``P``.
```

</details>

### `face_measures`

```python
face_measures(P: Tensor, faces: Tensor, face_len: Tensor) -> tuple[Tensor, Tensor, Tensor]
```
Area, unit normal and area centroid of (possibly polygonal) faces via fan triangulation.

<details><summary>docstring</summary>

```text
Args:
    P: ``(n0, 3)`` float64 positions.
    faces: ``(n2, M)`` int64 vertex ids, ``-1`` padded (padding trailing).
    face_len: ``(n2,)`` number of valid vertices (>= 3).
Returns:
    area ``(n2,)``, normal ``(n2, 3)`` (zero for degenerate faces; orientation follows the vertex
    order), centroid ``(n2, 3)`` (vertex mean for triangles; signed-fan area centroid otherwise).
```

</details>

### `surface_geometry`

```python
surface_geometry(
    pos: Tensor,
    faces: Tensor,
    face_len: Tensor,
    edges: Tensor,
    corner: dict[str, Tensor],
    star: str = 'cotan',
    kappa: float = 0.01,
    period=None,
) -> dict
```
Stars, descriptors and boundary flags of a 2-complex (triangles and/or polygons).

<details><summary>docstring</summary>

```text
Args:
    pos: ``(n0, D)`` positions, D in {2, 3} (any float dtype; computed in float64).
    faces: ``(n2, M)`` int64 vertex ids (``-1`` padded), oriented as given.
    face_len: ``(n2,)`` int64 number of vertices per face.
    edges: ``(n1, 2)`` int64 canonical edges (src < dst).
    corner: corner/half-edge table, each ``(H,)`` with ``H = sum(face_len)``: ``face``, ``v`` (corner
        vertex), ``next``, ``prev`` (neighbouring vertices in the face), ``edge`` (edge id of the
        half-edge ``v -> next``), ``sign`` (+1 if ``v < next``), ``prev_idx`` / ``next_idx`` (corner index of
        ``prev`` / ``next`` in the same face; for triangles ``prev`` is the corner opposite the half-edge).
    star: ``'cotan' | 'barycentric' | 'unit'``.
    kappa: cotan clamp factor.
    period: ``None``, or the axis periods of a flat torus (periodic complex, see :func:`min_image`): edge
        vectors are minimal images and every face is measured in its own unwrapped frame
        (:func:`local_cell_positions`), so all quantities are the intrinsic ones of the torus.
Returns:
    dict with ``star`` (3 tensors ``(n_k,)`` float32 > 0), ``geo`` (3 tensors ``(n_k, G_k)`` float32),
    ``boundary`` (3 bool tensors) and ``stats`` (python numbers).
```

</details>

### `volume_geometry`

```python
volume_geometry(
    pos: Tensor,
    tets: Tensor,
    faces: Tensor,
    edges: Tensor,
    tet_face: Tensor,
    tet_edge: Tensor,
    face_edge: Tensor,
    face_corner: dict[str, Tensor],
    star: str = 'barycentric',
) -> dict
```
Stars, descriptors and boundary flags of a tetrahedral 3-complex.

<details><summary>docstring</summary>

```text
Args:
    pos: ``(n0, 3)`` positions.
    tets: ``(n3, 4)`` int64 vertex ids (input order = orientation).
    faces: ``(n2, 3)`` canonical (sorted) triangles.
    edges: ``(n1, 2)`` canonical edges.
    tet_face: ``(n3, 4)`` face id of the face opposite local vertex i.
    tet_edge: ``(n3, 6)`` edge id of local vertex pairs ``(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)``.
    face_edge: ``(n2, 3)`` edge ids of face edges ``(a,b),(b,c),(a,c)``.
    face_corner: corner table of the canonical faces (see :func:`surface_geometry`).
    star: ``'barycentric' | 'unit'``.
Returns:
    dict with ``star``/``geo``/``boundary`` lists of length 4 and ``stats``.
```

</details>

### `barycentric_gradients`

```python
barycentric_gradients(V: Tensor) -> Tensor
```
Gradients of the barycentric coordinates of simplices (in their affine hull; any ambient dim).

<details><summary>docstring</summary>

```text
Args:
    V: ``(n, d+1, D)`` vertex positions (float64 recommended), ``d <= D``.
Returns:
    ``(n, d+1, D)`` with ``grad(lambda_i) . (V_j - V_0) = delta_ij - delta_i0`` and zero normal part.
```

</details>

### `whitney1_values`

```python
whitney1_values(lam: Tensor, grads: Tensor, pairs: Tensor, signs: Tensor) -> Tensor
```
Whitney 1-form basis `w_ab = lam_a grad(lam_b) - lam_b grad(lam_a)` at quadrature points.

<details><summary>docstring</summary>

```text
Args:
    lam: ``(Q, d+1)`` barycentric coordinates of the quadrature points.
    grads: ``(n, d+1, D)`` barycentric gradients.
    pairs: ``(m, 2)`` local vertex pairs ``a -> b`` of the edges.
    signs: ``(n, m)`` +-1 orientation of ``a -> b`` relative to the complex's canonical edge.
Returns:
    ``(n, Q, D, m)`` values of the canonically oriented basis functions.
```

</details>

### `whitney2_values_tet`

```python
whitney2_values_tet(lam: Tensor, grads: Tensor, face_local: Tensor) -> Tensor
```
Whitney 2-form basis (vector proxy) of the 4 faces of tets at quadrature points.

<details><summary>docstring</summary>

```text
``w_abc = 2 (lam_a grad_b x grad_c + lam_b grad_c x grad_a + lam_c grad_a x grad_b)`` with ``(a, b, c)``
the face's vertices in canonical (sorted global id) order, so the basis is consistent with ``d_1``/``d_2``.

Args:
    lam: ``(Q, 4)``; grads: ``(n, 4, 3)``; face_local: ``(n, 4, 3)`` local vertex ids of face i in canonical order.
Returns:
    ``(n, Q, 3, 4)``.
```

</details>

### `galerkin_blocks`

```python
galerkin_blocks(W: Tensor, wq: Tensor, t: Tensor) -> tuple[Tensor, Tensor]
```
PSD blocks of the Galerkin star `int w_i . sigma w_j` for `sigma = b I + sum_k a_k t_k t_k^T`.

<details><summary>docstring</summary>

```text
Args:
    W: ``(n, Q, D, mk)`` basis values; wq: ``(n, Q)`` quadrature weights (measure included);
    t: ``(n, m, D)`` unit direction vectors.
Returns:
    ``G0 (n, mk, mk) = sum_q wq W_q^T W_q`` and ``Gk (n, m, mk, mk) = sum_q wq (W_q^T t_k)(W_q^T t_k)^T``.
```

</details>

### `whitney_triangles`

```python
whitney_triangles(pos: Tensor, faces: Tensor, face_edges: Tensor, face_edge_signs: Tensor) -> dict
```
Whitney 1-form Galerkin blocks of a triangle complex (2-D or surface in 3-D).

<details><summary>docstring</summary>

```text
Args:
    pos: ``(n0, D)``; faces: ``(n2, 3)`` (given orientation); face_edges/face_edge_signs: ``(n2, 3)`` edge id and
        orientation of local edge j (joining local vertices j and j+1).
Returns:
    dict (float32 tensors): ``cells (n2, 3)``, ``signs (n2, 3)``, ``dir_edges (n2, 3)``, ``t (n2, 3, D)`` unit
    vectors of the local edges, ``G0 (n2, 3, 3)``, ``Gk (n2, 3, 3, 3)``, ``rowsum_G0 (n2, 3)``,
    ``rowsum_Gk (n2, 3, 3)``.
```

</details>

### `whitney_tets`

```python
whitney_tets(pos: Tensor, tets: Tensor, tet_edge: Tensor, tet_faces: Tensor) -> tuple[dict, dict]
```
Whitney 1-form (edges, 6x6) and 2-form (faces, 4x4) Galerkin blocks of a tetrahedral complex.

<details><summary>docstring</summary>

```text
Both use the 6 edge directions of the tet as the tensor frame (their ``t t^T`` span Sym(3); the 4 face normals
do not) and the 4-point degree-2 quadrature rule.

Args:
    pos: ``(n0, 3)``; tets: ``(n3, 4)``; tet_edge: ``(n3, 6)`` edge ids of local pairs
        ``(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)``; tet_faces: ``(n3, 4)`` face ids (face i omits local vertex i).
Returns:
    ``(whitney1, whitney2)`` dicts as in :func:`whitney_triangles` (``cells`` = edges resp. faces, ``dir_edges``
    = ``tet_edge`` for both, ``t (n3, 6, 3)``).
```

</details>

### Constants

- `GEO_FEATURE_NAMES`: dict with keys {0, 1, 2, 3}
- `GEO_RELATIVE_COLUMNS`: dict with keys {0, 1, 2, 3}
- `STAR_TYPES`: ('cotan', 'barycentric', 'unit')
- `KAPPA`: 0.01
- `COT_CLAMP`: 10.0


## Sparse and segment operations: `rhmp.ops`

Spmm on (n, B, C), Gershgorin bounds, segment reductions.

<details><summary>module docstring</summary>

```text
Sparse, segment and diagnostic operations on cochain tensors.

Layout convention (DESIGN §2): cochain features are ``(n, B, C)`` contiguous tensors.  A sparse
operator ``A`` of shape ``(m, n)`` acts on them through a zero-copy ``(n, B*C)`` view, so one
sparse-dense product serves every sample and channel at once.

All sparse products run in fp32 (or fp64 in tests); bf16/fp16 inputs are cast up for the product
and the result is cast back.  Gradients of ``spmm`` use a *precomputed* transpose when one is given
(``AT``), so no ``.t()``/``.coalesce()`` happens during forward or backward.
```

</details>

### `spmm`

```python
spmm(A: Tensor, x: Tensor, AT: Tensor | None = None) -> Tensor
```
Sparse-dense product on the cochain layout.

<details><summary>docstring</summary>

```text
``x`` of shape ``(n, *tail)`` (typically ``(n, B, C)`` or ``(n, B)``) is viewed as
``(n, prod(tail))`` without copying, multiplied by ``A`` and viewed back.

Args:
    A: sparse CSR (or COO) ``(m, n)``, fp32 (fp64 allowed in tests).
    x: dense ``(n, *tail)``; must be contiguous.  bf16/fp16 inputs are computed in fp32 and cast
        back; an fp64 ``x`` upcasts ``A`` on the fly (exact for incidence matrices).
    AT: optional precomputed ``A^T`` (CSR ``(n, m)``).  When given, the backward pass uses it
        (no transposition at run time).  Always pass it in training code.
Returns:
    ``(m, *tail)`` tensor with the dtype of ``x``.
Raises:
    ValueError: on shape mismatch or non-contiguous ``x``.
```

</details>

### `sparse_csr`

```python
sparse_csr(crow: Tensor, col: Tensor, val: Tensor, shape: tuple[int, int]) -> Tensor
```
Wrap `(crow, col, val)` into a CSR tensor without invariant checks.

<details><summary>docstring</summary>

```text
Args:
    crow: ``(m+1,)`` int64 row pointer.
    col: ``(nnz,)`` int64 column indices (sorted within each row).
    val: ``(nnz,)`` values.
    shape: ``(m, n)``.
Returns:
    sparse CSR tensor ``(m, n)``.
```

</details>

### `csr_from_coo`

```python
csr_from_coo(row: Tensor, col: Tensor, val: Tensor, shape: tuple[int, int]) -> Tensor
```
Build a CSR matrix from COO triplets (vectorised; entries are sorted by `(row, col)`).

<details><summary>docstring</summary>

```text
Duplicate ``(row, col)`` pairs are *not* summed; callers guarantee uniqueness.

Args:
    row, col: ``(nnz,)`` int64 indices.
    val: ``(nnz,)`` values.
    shape: ``(m, n)``.
Returns:
    sparse CSR tensor ``(m, n)`` on the device of ``row``.
```

</details>

### `csr_with_values`

```python
csr_with_values(A: Tensor, val: Tensor) -> Tensor
```
Same sparsity pattern as CSR `A` with new values `val` (index tensors are shared).

<details><summary>docstring</summary>

```text
Args:
    A: CSR ``(m, n)``.
    val: ``(nnz,)``.
Returns:
    CSR ``(m, n)``.
```

</details>

### `gather_apply`

```python
gather_apply(col: Tensor, val: Tensor, x: Tensor) -> Tensor
```
`y = A x` for an operator whose rows all have `r` entries, via `index_select`.

<details><summary>docstring</summary>

```text
Args:
    col: ``(m, r)`` int64 column index of every row entry.
    val: ``(m, r)`` values (same dtype as ``x`` or castable).
    x: ``(n, *tail)``.
Returns:
    ``(m, *tail)``.
```

</details>

### `scatter_apply`

```python
scatter_apply(col: Tensor, val: Tensor, y: Tensor, n: int) -> Tensor
```
`x = A^T y` for the operator of `gather_apply`, via `index_add`.

<details><summary>docstring</summary>

```text
Args:
    col: ``(m, r)`` int64.
    val: ``(m, r)``.
    y: ``(m, *tail)``.
    n: number of columns of ``A``.
Returns:
    ``(n, *tail)``.
```

</details>

### `gershgorin_bound`

```python
gershgorin_bound(
    A_abs: Tensor,
    AT_abs: Tensor,
    h: Tensor,
    s_inv_sqrt: Tensor,
    batch: Tensor | None = None,
    num_graphs: int = 1,
) -> Tensor
```
Per-sample upper bound on `lambda_max(S^{-1/2} A^T diag(h) A S^{-1/2})`.

<details><summary>docstring</summary>

```text
Row sums of the entrywise absolute operator (Gershgorin):
``R_j = s_j^{-1/2} [ |A|^T ( h * (|A| s^{-1/2}) ) ]_j``, bound ``= max_j R_j`` per sample.
Exact upper bound for any ``h >= 0`` and ``s > 0``; differentiable in ``h`` and ``s``.

Args:
    A_abs: CSR ``|A|``, ``(m, n)``.  Up-block of degree k: ``|d_k|``; down-block: ``|d_{k-1}^T|``.
    AT_abs: CSR ``|A|^T``, ``(n, m)``.
    h: metric on the m-side cells, ``(m, B)`` or ``(m,)``; nonnegative.
    s_inv_sqrt: ``S^{-1/2}`` on the n-side cells, ``(n,)`` or ``(n, B)``; positive.
    batch: optional ``(n,)`` int64 sample id of each n-side cell (block-diagonal batch).
    num_graphs: number of samples in the block-diagonal batch.
Returns:
    ``(B,)`` when ``batch is None`` (shared mesh), else ``(num_graphs, B)``.
```

</details>

### `beta_unit`

```python
beta_unit(
    A_abs: Tensor,
    AT_abs: Tensor,
    s_inv_sqrt: Tensor | None = None,
    batch: Tensor | None = None,
    num_graphs: int = 1,
) -> Tensor
```
Gershgorin bound of the unit-metric operator `S^{-1/2} A^T A S^{-1/2}` (i.e. `h = 1`).

<details><summary>docstring</summary>

```text
For any SPD metric ``H`` (diagonal or Galerkin):
``lambda_max(S^{-1/2} A^T H A S^{-1/2}) <= lambda_max(H) beta_unit <= rowsum_max(|H|) beta_unit``,
so a tensor (Whitney) metric can be normalised with ``rowsum_max(|H|) * beta_unit`` (DESIGN §9.1).

Args:
    A_abs: CSR ``|A|`` ``(m, n)`` (may already carry a scaling folded into its values).
    AT_abs: CSR ``|A|^T`` ``(n, m)``.
    s_inv_sqrt: ``S^{-1/2}``: ``(n,)``, ``(n, B)`` or ``None`` (``S = 1``).
    batch, num_graphs: as in :func:`gershgorin_bound`.
Returns:
    ``(B,)`` (``B = 1`` unless ``s_inv_sqrt`` is ``(n, B)``) or ``(num_graphs, B)``.
```

</details>

### `broadcast_to_cells`

```python
broadcast_to_cells(bound: Tensor, batch: Tensor | None = None) -> Tensor
```
Expand a per-sample quantity to per-cell rows for broadcasting against `(n, B, ...)`.

<details><summary>docstring</summary>

```text
Args:
    bound: ``(B,)`` (shared mesh) or ``(num_graphs, B)`` (block-diagonal).
    batch: ``None`` or ``(n,)`` sample ids.
Returns:
    ``(1, B)`` if ``batch is None`` else ``(n, B)``.
```

</details>

### `segment_sum`

```python
segment_sum(x: Tensor, index: Tensor, num_segments: int) -> Tensor
```
Sum rows of `x` per segment.

<details><summary>docstring</summary>

```text
Args:
    x: ``(n, *tail)``.
    index: ``(n,)`` int64 segment id in ``[0, num_segments)``.
    num_segments: number of segments.
Returns:
    ``(num_segments, *tail)``.
```

</details>

### `segment_mean`

```python
segment_mean(x: Tensor, index: Tensor, num_segments: int) -> Tensor
```
Mean of rows of `x` per segment (empty segments give 0).

<details><summary>docstring</summary>

```text
Args:
    x: ``(n, *tail)``; index: ``(n,)``; num_segments: int.
Returns:
    ``(num_segments, *tail)``.
```

</details>

### `segment_max`

```python
segment_max(x: Tensor, index: Tensor, num_segments: int) -> Tensor
```
Max of rows of `x` per segment (empty segments give 0); differentiable.

<details><summary>docstring</summary>

```text
Args:
    x: ``(n, *tail)``; index: ``(n,)``; num_segments: int.
Returns:
    ``(num_segments, *tail)``.
```

</details>

### `power_iteration_norm`

```python
power_iteration_norm(
    matvec: Callable[[Tensor], Tensor],
    n: int,
    B: int,
    iters: int = 30,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
    seed: int = 0,
) -> Tensor
```
Per-sample largest-magnitude eigenvalue of a *symmetric* operator (diagnostics/tests).

<details><summary>docstring</summary>

```text
Args:
    matvec: maps ``(n, B, 1) -> (n, B, 1)``; column ``b`` must depend only on column ``b``.
    n: operator size.
    B: number of independent samples (columns).
    iters: number of iterations.
    device, dtype: of the iterate.
    seed: seed of the random start vector.
Returns:
    ``(B,)`` Rayleigh-quotient estimate of ``|lambda|_max`` (a lower bound that converges up).
```

</details>

### `to_nbc`

```python
to_nbc(x: Tensor) -> Tensor
```
`(B, n, C) -> (n, B, C)` contiguous (model layout).

### `to_bnc`

```python
to_bnc(x: Tensor) -> Tensor
```
`(n, B, C) -> (B, n, C)` contiguous (user layout).


## Data containers: `rhmp.data`

TaskData, normalisation, splits, minibatching, cochain alignment.

<details><summary>module docstring</summary>

```text
GPU-resident task containers, v1-compatible splits / normalisation, cochain alignment and minibatching.

``TaskData`` (DESIGN §4) holds everything a run needs, already normalised and on the target device:

* shared-mesh tasks (T1..T7, T6f, ...): ``K`` is one :class:`~rhmp.complex.CochainComplex`, ``inputs[k]`` is
  ``(N, n_k, F_k)`` and ``target`` is ``(N, n_t, out_dim)``.  A minibatch is converted to the model layout
  ``(n_k, B, F_k)`` by :func:`shared_minibatch`.
* variable-mesh tasks (T8, HP, TET): ``K`` is a list of complexes (one per sample), ``inputs`` a list of dicts of
  ``(n_k_i, F_k)`` tensors and ``target`` a list of ``(n_t_i, out_dim)`` tensors.  :func:`mesh_minibatch` builds a
  block-diagonal batch (``CochainComplex.batch``; ``B = 1``, cells concatenated, ``K.batch[k]`` sample ids).

Normalisation follows v1 (the v1 training script ``formal_benchmark.py``): per-feature mean/std over (samples,
cells) of the training part, ``std.clamp(1e-6)``; torch's unbiased std for the shared-mesh pickles, numpy's biased
std for T8.
Orientation-odd quantities (cochain inputs/targets on degree >= 1) and vector targets are only *scaled*
(mean 0, RMS scale) so that the normalisation commutes with orientation flips / rotations; see ``Stats``.
```

</details>

### class `Stats`

```python
Stats(mean: Tensor, std: Tensor, kind: str = 'v1')
```
Per-feature affine normalisation `x_n = (x - mean) / std`.

| field | type | default | description |
|---|---|---|---|
| `mean` | `Tensor` | required | `(F,)`. |
| `std` | `Tensor` | required | `(F,)` (>= 1e-6). |
| `kind` | `str` | `'v1'` | free-form description (e.g. `'v1'`, `'scale'`, `'isotropic'`). |

<details><summary>docstring</summary>

```text
Attributes:
    mean: ``(F,)``.
    std: ``(F,)`` (>= 1e-6).
    kind: free-form description (e.g. ``'v1'``, ``'scale'``, ``'isotropic'``).
```

</details>

#### `Stats.normalize`

```python
normalize(x: Tensor) -> Tensor
```
`(..., F) -> (..., F)`.

#### `Stats.denormalize`

```python
denormalize(x: Tensor) -> Tensor
```
`(..., F) -> (..., F)`.

#### `Stats.to`

```python
to(device) -> Stats
```
Copy on `device`.

#### `Stats.as_tuple`

```python
as_tuple() -> tuple[Tensor, Tensor]
```
`(mean, std)`.

#### `Stats.to_dict`

```python
to_dict() -> dict
```
JSON-friendly `{'mean', 'std', 'kind'}`.

### class `OutputMap`

```python
OutputMap(
    kind: str,
    model_readout: str,
    model_out_dim: int,
    M: Tensor | None = None,
    MT: Tensor | None = None,
    normals: Tensor | None = None,
    description: str = '',
    D: int = 0,
)
```
Fixed, parameter-free linear map from the model output to the task's target space.

| field | type | default | description |
|---|---|---|---|
| `kind` | `str` | required | `'sparse' \| 'cross_normal' \| 'direct_vector'`. |
| `model_readout` | `str` | required | readout and output width the model must be configured with. |
| `model_out_dim` | `int` | required | readout and output width the model must be configured with. |
| `M` | `Tensor \| None` | `None` | CSR `(n_t, n_m)` and its transpose (`sparse`). |
| `MT` | `Tensor \| None` | `None` | CSR `(n_t, n_m)` and its transpose (`sparse`). |
| `normals` | `Tensor \| None` | `None` | `(n0, 3)` unit vertex normals (`cross_normal`). |
| `description` | `str` | `''` | human-readable summary. |
| `D` | `int` | `0` |  |

<details><summary>docstring</summary>

```text
The v2 model is exactly equivariant under relabelling of cell orientations and invariant under reflections, so
it cannot output *pseudo*-scalars/vectors (vorticity, magnetic flux, rotated gradients ``n x grad psi``), which are
defined relative to the physical orientation of the domain.  The task supplies that orientation here:

* ``kind='sparse'``: ``y_t = M y_m`` (e.g. the oriented face -> node average
  ``y_i = mean_{f ∋ i} sigma_f x_f`` with ``sigma_f = +1`` for counter-clockwise faces, applied to a
  ``cochain:2`` readout);
* ``kind='cross_normal'``: ``v_i = n_i x g_i`` (rotation by +90 deg in the tangent plane of an oriented surface,
  applied to a ``node_vector`` readout);
* ``kind='direct_vector'``: edge scalars -> vertex vectors ``v_i = sum_{e ∋ i} w_e t_e / deg_i`` (``t_e`` unit
  src->dst vector; the v1 edge-to-node average), ``M`` of shape ``(D n0, n1)`` (row ``D i + c``), applied to a
  ``grad`` readout (``w = d0 phi``) for T5g.

Attributes:
    kind: ``'sparse' | 'cross_normal' | 'direct_vector'``.
    model_readout / model_out_dim: readout and output width the model must be configured with.
    M, MT: CSR ``(n_t, n_m)`` and its transpose (``sparse``).
    normals: ``(n0, 3)`` unit vertex normals (``cross_normal``).
    description: human-readable summary.
```

</details>

#### `OutputMap.to`

```python
to(device) -> OutputMap
```
Copy on `device`.

### class `TaskData`

```python
TaskData(
    name: str,
    K: Any,
    inputs: Any,
    target: Any,
    target_degree: int,
    target_kind: str,
    in_dims: dict[int, int],
    even_dims: dict[int, int],
    connection_dims: dict[int, int],
    split: tuple[Tensor, Tensor, Tensor],
    x_stats: dict[int, Stats],
    y_stats: Stats,
    spatial_dim: int,
    out_dim: int = 1,
    native: bool = True,
    meta: dict = {},
    extra_tests: dict = {},
    output_map: OutputMap | None = None,
    readout: str | None = None,
)
```
Everything a run needs, normalised and on `device` (DESIGN §4).

| field | type | default | description |
|---|---|---|---|
| `name` | `str` | required | task name (e.g. `'T6'`, `'T6f'`, `'HP_k100'`). |
| `K` | `Any` | required | shared complex, or list of complexes (variable meshes). |
| `inputs` | `Any` | required | `{k: (N, n_k, F_k)}` (shared) or list of `{k: (n_k_i, F_k)}` (variable meshes); normalised. |
| `target` | `Any` | required | `(N, n_t, out_dim)` or list of `(n_t_i, out_dim)`; normalised. |
| `target_degree` | `int` | required | degree of the output cells (0 for node targets). |
| `target_kind` | `str` | required | `'node_scalar' \| 'node_vector' \| 'cochain' \| 'even'`. |
| `in_dims` | `dict[int, int]` | required | model input configuration (DESIGN §3.4/3.6). |
| `even_dims` | `dict[int, int]` | required | model input configuration (DESIGN §3.4/3.6). |
| `connection_dims` | `dict[int, int]` | required | model input configuration (DESIGN §3.4/3.6). |
| `split` | `tuple[Tensor, Tensor, Tensor]` | required | `(train, val, test)` sample indices (int64, CPU). |
| `x_stats` | `dict[int, Stats]` | required | `{k: Stats}` input normalisation per degree. |
| `y_stats` | `Stats` | required | target `Stats` (`denormalize` gives physical units). |
| `spatial_dim` | `int` | required | embedding dimension of the mesh (2 or 3). |
| `out_dim` | `int` | `1` | number of output channels. |
| `native` | `bool` | `True` | native-cochain (True) or v1-legacy (False) inputs/outputs. |
| `meta` | `dict` | `{}` | task defaults and diagnostics: `v1_C`, `batch_size`, `readout`, `vector_mode`, `r2_std` (v1 per-feature target std for a v1-comparable R2), `paper_eval` (indices of the 100 test samples of the paper tables, `test100`), `consistency` checks, ... |
| `extra_tests` | `dict` | `{}` | additional evaluation sets normalised with the same statistics, e.g. `{'fine': {'K': [...], 'inputs': [...], 'target': [...]}}` (zero-shot resolution transfer). |
| `output_map` | `OutputMap \| None` | `None` | optional `OutputMap` applied to the model output before the loss/metrics; `None` means the model output is compared with `target` directly. |
| `readout` | `str \| None` | `None` | **the** readout the model is built with (single source of truth; `rhmp.train.build_config` uses it). `None` resolves once at construction to: the legacy `meta['model_readout']` if present, else `output_map.model_readout`, else `target_readout` (the readout that predicts `target` on its own cells). Examples: `'cochain:2'` + oriented face->node map for the pseudo-scalar node targets of T1/T6/T7, `'grad'` for the curl-free edge target of HPgrad. |

<details><summary>docstring</summary>

```text
Attributes:
    name: task name (e.g. ``'T6'``, ``'T6f'``, ``'HP_k100'``).
    K: shared complex, or list of complexes (variable meshes).
    inputs: ``{k: (N, n_k, F_k)}`` (shared) or list of ``{k: (n_k_i, F_k)}`` (variable meshes); normalised.
    target: ``(N, n_t, out_dim)`` or list of ``(n_t_i, out_dim)``; normalised.
    target_degree: degree of the output cells (0 for node targets).
    target_kind: ``'node_scalar' | 'node_vector' | 'cochain' | 'even'``.
    in_dims / even_dims / connection_dims: model input configuration (DESIGN §3.4/3.6).
    split: ``(train, val, test)`` sample indices (int64, CPU).
    x_stats: ``{k: Stats}`` input normalisation per degree.
    y_stats: target ``Stats`` (``denormalize`` gives physical units).
    spatial_dim: embedding dimension of the mesh (2 or 3).
    out_dim: number of output channels.
    native: native-cochain (True) or v1-legacy (False) inputs/outputs.
    meta: task defaults and diagnostics: ``v1_C``, ``batch_size``, ``readout``, ``vector_mode``,
        ``r2_std`` (v1 per-feature target std for a v1-comparable R2), ``paper_eval`` (indices of the 100 test
        samples of the paper tables, ``test100``), ``consistency`` checks, ...
    extra_tests: additional evaluation sets normalised with the same statistics, e.g.
        ``{'fine': {'K': [...], 'inputs': [...], 'target': [...]}}`` (zero-shot resolution transfer).
    output_map: optional :class:`OutputMap` applied to the model output before the loss/metrics; ``None``
        means the model output is compared with ``target`` directly.
    readout: **the** readout the model is built with (single source of truth; ``rhmp.train.build_config`` uses
        it).  ``None`` resolves once at construction to: the legacy ``meta['model_readout']`` if present, else
        ``output_map.model_readout``, else :attr:`target_readout` (the readout that predicts ``target`` on its
        own cells).  Examples: ``'cochain:2'`` + oriented face->node map for the pseudo-scalar node targets of
        T1/T6/T7, ``'grad'`` for the curl-free edge target of HPgrad.
```

</details>

#### classmethod `TaskData.from_arrays`

```python
from_arrays(
    K,
    inputs,
    target,
    *,
    readout: str,
    even_dims: dict | None = None,
    connection_dims: dict | None = None,
    split=(0.7, 0.15, 0.15),
    stats: str = 'auto',
    name: str = 'custom',
    output_map: OutputMap | None = None,
    meta: dict | None = None,
) -> TaskData
```
Build a normalised `TaskData` from plain (raw, unnormalised) tensors.

<details><summary>docstring</summary>

```text
Args:
    K: one :class:`~rhmp.complex.CochainComplex` shared by all samples, or a list with one complex per sample.
    inputs: ``{k: (N, n_k, F_k)}`` (shared mesh) or a list of ``{k: (n_k_i, F_k)}`` (variable meshes).  On
        degree ``k >= 1`` the columns are ``[connection | odd | even]`` as in ``RHMPConfig`` (even listed last).
    target: ``(N, n_t, O)`` or a list of ``(n_t_i, O)``.
    readout: the model readout (``node_scalar | node_vector | cochain:k | even:k | grad | curl | div[:k]``);
        it also fixes the target cells and kind.
    even_dims / connection_dims: ``{k: count}`` as in ``RHMPConfig``.
    split: ``(train, val, test)`` fractions (sequential, ``int(f * N)`` as v1) or three index tensors.
    stats: ``'auto'``: per column, odd cochain columns get ``scale_stats`` (mean 0, RMS; keeps orientation
        oddness), vertex and even columns ``feature_stats`` (mean/std); the target gets ``scale_stats`` for odd
        readouts (``cochain``, ``grad``, ``curl``, ``div``), isotropic scaling for ``node_vector`` and
        ``feature_stats`` otherwise; statistics from the training split.  ``'none'``: identity.
    name, output_map, meta: passed through (``meta`` gets ``r2_std`` and a default ``batch_size``).
Returns:
    TaskData (tensors stay on the device they were given on).
```

</details>

#### `TaskData.variable_mesh` (property)

True when every sample has its own complex.

#### `TaskData.num_samples` (property)

Number of samples `N`.

#### `TaskData.target_readout` (property)

Readout that predicts `target` on its own cells (`node_scalar | node_vector | cochain:k | even:k`).

#### `TaskData.model_readout` (property)

Deprecated alias of `readout`.

#### `TaskData.geo_dims` (property)

`{k: G_k}` of the task's complexes (model constructor argument).

#### `TaskData.summary`

```python
summary() -> dict
```
JSON-friendly description (shapes, split sizes, normalisation).

### `sequential_split`

```python
sequential_split(N: int, device=None) -> tuple[Tensor, Tensor, Tensor]
```
v1 split: first `int(0.7 N)` train, next `int(0.15 N)` val, rest test (no shuffling).

### `holdout_split`

```python
holdout_split(
    N: int,
    train_frac: float = 0.7,
    val_frac_of_train: float = 0.1,
    device=None,
) -> tuple[Tensor, Tensor, Tensor]
```
v1 T8 split (first 70 % train, last 30 % test) with the last 10 % of the train part held out for model selection (v1 selected on the test part; v2 does not).

### `feature_stats`

```python
feature_stats(x_train: Tensor | Sequence[Tensor], *, unbiased: bool = True) -> Stats
```
v1 per-feature statistics over all leading dims (samples x cells).

<details><summary>docstring</summary>

```text
Args:
    x_train: ``(N, n, F)`` tensor or list of ``(n_i, F)`` tensors (pooled).
    unbiased: torch default (``True``, v1 shared-mesh pickles) or numpy default (``False``, v1 T8).
Returns:
    ``Stats`` with ``mean, std`` of shape ``(F,)``, std clamped at 1e-6.
```

</details>

### `scale_stats`

```python
scale_stats(x_train: Tensor | Sequence[Tensor], *, isotropic: bool = False) -> Stats
```
Scale-only statistics for orientation-odd cochains and vector fields: `mean = 0`, `std = RMS`.

<details><summary>docstring</summary>

```text
Args:
    x_train: ``(N, n, F)`` or list of ``(n_i, F)``.
    isotropic: one common RMS for all ``F`` columns (vector components) instead of one per column.
Returns:
    ``Stats`` (``(F,)`` tensors).
```

</details>

### `edge_alignment`

```python
edge_alignment(K_edges: Tensor, src_edges: Tensor, n0: int) -> tuple[Tensor, Tensor]
```
Map the edges of a complex to the rows of another edge list.

<details><summary>docstring</summary>

```text
Args:
    K_edges: ``(n1, 2)`` edges of the complex (``src < dst``).
    src_edges: ``(n1, 2)`` edges in the data's order and orientation.
    n0: number of vertices.
Returns:
    ``(perm, sign)``: ``values_K = values_src[perm] * sign`` for an odd edge cochain (``sign = +1`` where the two
    orientations agree), ``values_src[perm]`` for even edge data.
Raises:
    ValueError: if the two edge sets differ.
```

</details>

### `cell_alignment`

```python
cell_alignment(
    K_cells: Tensor,
    src_cells: Tensor,
    n0: int,
    *,
    oriented: bool = True,
) -> tuple[Tensor, Tensor]
```
Map the k-cells (simplices, k>=2) of a complex to the rows of another cell list with the same vertex sets.

<details><summary>docstring</summary>

```text
Args:
    K_cells: ``(n_k, p)`` vertex tuples of the complex (``-1`` padding not supported).
    src_cells: ``(n_k, p)`` vertex tuples in the data's order/orientation.
    n0: number of vertices.
    oriented: also return the relative orientation sign.
Returns:
    ``(perm, sign)``: ``values_K = values_src[perm] * sign`` (odd) or ``values_src[perm]`` (even).
```

</details>

### `shared_minibatch`

```python
shared_minibatch(
    inputs: dict[int, Tensor],
    target: Tensor,
    idx: Tensor,
) -> tuple[dict[int, Tensor], Tensor]
```
Gather a shared-mesh minibatch in the model layout.

<details><summary>docstring</summary>

```text
Args:
    inputs: ``{k: (N, n_k, F_k)}``.
    target: ``(N, n_t, O)``.
    idx: ``(B,)`` sample indices.
Returns:
    ``({k: (n_k, B, F_k)}, (n_t, B, O))`` contiguous.
```

</details>

### `mesh_minibatch`

```python
mesh_minibatch(
    Ks: Sequence,
    inputs: Sequence[dict[int, Tensor]],
    target: Sequence[Tensor],
    idx,
    device=None,
    cache: dict | None = None,
)
```
Block-diagonal minibatch of variable meshes.

<details><summary>docstring</summary>

```text
Args:
    Ks: list of complexes (any device).
    inputs: list of ``{k: (n_k_i, F_k)}``.
    target: list of ``(n_t_i, O)``.
    idx: sample indices (tensor or list).
    device: device of the returned batch (defaults to the device of the first target).
    cache: optional dict; batched complexes are memoised by the tuple of indices (use for the fixed
        validation/test batches).
Returns:
    ``(K_batch, {k: (sum n_k, 1, F_k)}, (sum n_t, 1, O))``.
```

</details>

### `iterate_indices`

```python
iterate_indices(
    idx: Tensor,
    batch_size: int,
    *,
    shuffle: bool,
    generator: torch.Generator | None = None,
    drop_last: bool = False,
)
```
Yield index minibatches (CPU generator for device-independent determinism).

### `concat_cells`

```python
concat_cells(per_sample: Sequence[Tensor]) -> Tensor
```
Concatenate per-sample cell features `[(n_i, F)]` into the block-diagonal layout `(sum n_i, 1, F)`.

- `rhmp.data.to_nbc`: re-export of `rhmp.ops.to_nbc` (documented there).
### `same_device`

```python
same_device(a, b) -> bool
```
Device equality treating `cuda` and `cuda:<current>` as the same device.

### `tensor_nbytes`

```python
tensor_nbytes(o) -> int
```
Bytes held by the tensors inside `o` (tensors incl. sparse CSR, dicts, lists, dataclass fields of a complex).

### `strip_whitney`

```python
strip_whitney(obj) -> int
```
Drop the Whitney/Galerkin blocks (`K.whitney`, only used by tensor metrics) from a complex, a list of complexes, or every complex of a `TaskData` and its extra test sets.  Returns the number of bytes freed.

### `pack_to`

```python
pack_to(obj, device)
```
Move every tensor inside `obj` to `device` with one copy per dtype instead of one per tensor.

<details><summary>docstring</summary>

```text
``obj`` may be a tensor, a (nested) list/tuple/dict, or a dataclass such as a ``CochainComplex`` (sparse CSR
tensors included); the structure is rebuilt with views into a few contiguous device buffers.  Moving e.g. 5000
variable-mesh complexes (~200K small tensors) this way takes seconds even on a GPU that is time-sliced with other
processes, where per-tensor synchronous copies take tens of minutes.

Returns:
    an object of the same structure on ``device`` (non-tensor leaves are shared).
```

</details>

### `add_abs_scale`

```python
add_abs_scale(task: TaskData) -> TaskData
```
Append a constant *even* vertex column `log(median edge length)` (absolute length units) to `inputs[0]`.

<details><summary>docstring</summary>

```text
The v2 model is scale-free by design (descriptors ``log(x / median x)``, stars normalised per graph), so a task
whose physics has a fixed length scale (e.g. meshes of different sizes with a fixed diffusion length) must get the
absolute scale as an input.  The column is constant per mesh, standardised with the training split (a single
shared mesh gives an all-zero column), listed last and declared even (it also feeds the vertex metric head).
Extra test sets get the same column with the same statistics.  In place; returns ``task``.
```

</details>


## Metrics: `rhmp.metrics`

R2 / MSE / MAE / NRMSE / SSIM / Pearson (v1-compatible).

<details><summary>module docstring</summary>

```text
Evaluation metrics in torch (vectorised, any device), numerically matching the v1 numpy/torch code.

Reference implementations: the v1 evaluation script ``compute_all_metrics.py``, functions ``compute_r2_mse_mae``,
``compute_nrmse``, ``compute_ssim_pearson`` (vendored in ``rhmp.baselines.v1.metrics_v1``).
``tests/test_metrics.py`` checks agreement to 1e-6 on random data.

Conventions (identical to v1):

* ``R2``, ``MSE``, ``MAE`` are computed in the *normalised* target space (``(y - mean) / std``, per feature,
  statistics from the training split).  ``R2 = 1 - SS_res / max(SS_tot, 1e-8)`` with
  ``SS_tot = sum (t - t.mean(0))**2``: the mean is taken over samples, per (cell, feature) position.
* ``NRMSE``, ``SSIM`` and ``Pearson`` are computed on *unnormalised* values.  NRMSE = RMSE / range(target).
  SSIM / Pearson are per-sample (all cells and all output features of a sample flattened together) and then
  averaged over samples; SSIM uses a *global* data range (from all target samples) for C1/C2 and predictions are
  clipped to ``[t_min - 0.1 range, t_max + 0.1 range]``; Pearson is clipped to ``>= 0``.

Orientation-odd cochain targets (``cochain:k`` readouts: fluxes, curvatures) are scored with an *uncentred* R2,
``SS_tot = sum t**2`` (``center=False``): the mean of an odd quantity depends on the orientation convention of the
cells, so centring would make R2 depend on vertex labels.  The trainer selects this automatically.

Ragged data (variable meshes, e.g. T8/HP/TET block-diagonal batches) is supported by passing lists of per-sample
tensors ``[(n_i, d), ...]``.  For ragged data the R2 total sum of squares uses the pooled per-feature mean
(positions of different meshes do not correspond); for equal-size samples pass a stacked ``(N, n, d)`` tensor to get
exactly the v1 definition.

All reductions are done in float64.
```

</details>

### `r2_score`

```python
r2_score(pred: TensorOrList, target: TensorOrList, center: bool = True) -> float
```
Coefficient of determination, v1 definition.

<details><summary>docstring</summary>

```text
Args:
    pred, target: ``(N, n, d)`` (dense; SS_tot uses the per-position mean over samples, as v1) or lists of
        ``(n_i, d)`` (ragged; SS_tot uses the pooled per-feature mean).
    center: subtract the mean in SS_tot (v1).  ``False``: ``SS_tot = sum t**2`` (orientation-odd targets).
Returns:
    ``1 - SS_res / max(SS_tot, 1e-8)`` as a Python float.
```

</details>

### `mse`

```python
mse(pred: TensorOrList, target: TensorOrList) -> float
```
Mean squared error over all elements (float64 accumulation).

### `mae`

```python
mae(pred: TensorOrList, target: TensorOrList) -> float
```
Mean absolute error over all elements (float64 accumulation).

### `nrmse`

```python
nrmse(pred_raw: TensorOrList, target_raw: TensorOrList) -> float
```
`RMSE / max(range(target), 1e-8)` over all elements (unnormalised values).

### `ssim_pearson`

```python
ssim_pearson(
    pred_raw: TensorOrList,
    target_raw: TensorOrList,
    n_samples: int | None = None,
) -> tuple[float, float]
```
Per-sample global-range SSIM and clipped Pearson correlation, averaged over samples (v1 definition).

<details><summary>docstring</summary>

```text
Args:
    pred_raw, target_raw: unnormalised ``(N, ...)`` tensors, or lists of per-sample ``(n_i, ...)`` tensors.
    n_samples: evaluate only the first ``n_samples`` samples (the global data range and clipping bounds are
        still computed from *all* targets, exactly as v1).
Returns:
    ``(mean_ssim, mean_pearson)``.
```

</details>

### `summarize`

```python
summarize(
    pred: TensorOrList,
    target: TensorOrList,
    y_stats,
    *,
    r2_std: Tensor | None = None,
    n_samples: int | None = None,
    prefix: str = '',
    center: bool = True,
) -> dict[str, float]
```
All metrics for normalised predictions/targets.

<details><summary>docstring</summary>

```text
Args:
    pred, target: normalised model outputs and targets, ``(N, n, d)`` or lists of ``(n_i, d)``.
    y_stats: ``(mean, std)`` or ``{'mean','std'}`` with shape ``(d,)``: raw = normalised * std + mean.
    r2_std: optional per-feature std ``(d,)`` defining the v1 normalisation. When the training normalisation
        differs from v1's (e.g. isotropic scaling of vector targets), ``R2_v1`` recomputes R2 in the v1
        normalised space (only the std matters: the mean cancels in R2).
    n_samples: restrict every metric (including the SSIM data range) to the first ``n_samples`` samples, as the
        v1 function ``compute_all_metrics.evaluate_task`` does for the paper tables (100 test samples).
    prefix: prepended to every key.
    center: centred (v1) or uncentred (orientation-odd targets) R2, see :func:`r2_score`.
Returns:
    dict with ``R2, MSE, MAE`` (normalised space), ``NRMSE, SSIM, Pearson`` (raw space), optional ``R2_v1``,
    and ``N`` (number of samples evaluated).
```

</details>

### `unnormalize`

```python
unnormalize(x: TensorOrList, mean: Tensor, std: Tensor) -> TensorOrList
```
`x * std + mean` (broadcast over the last dim); works on tensors or lists of tensors.

### `flatten_ragged`

```python
flatten_ragged(x: TensorOrList) -> tuple[Tensor, Tensor, int]
```
Concatenate per-sample tensors.

<details><summary>docstring</summary>

```text
Args:
    x: ``(N, n, ...)`` tensor, or a list of ``N`` tensors ``(n_i, ...)`` with equal trailing shape.
Returns:
    ``(flat, seg, N)``: ``flat`` is ``(M, d)`` (trailing dims flattened into ``d``), ``seg`` is ``(M,)`` int64
    sample ids, ``N`` the number of samples.
```

</details>


## Trainer: `rhmp.train`

Training / evaluation loop and CLI.

<details><summary>module docstring</summary>

```text
Generic trainer / evaluator for RHMP v2 (v1 protocol by default).

    python3 -u -m rhmp.train --task T6 --native --epochs 100 --seed 42 --out runs/t6_native
    python3 -u -m rhmp.train --task T8 --epochs 2 --out runs/smoke_t8                  # block-diagonal batches of 8
    python3 -u -m rhmp.train --task T6 --legacy --eval-v1 checkpoints_v1/T6_wilson_loop/ours/best_model.pt                              --out runs/v1_t6                                             # v1 model, same split/metrics
    python3 -u -m rhmp.train --task HP_k100 --check-data                                  # load + print the data summary

Protocol (the v1 training script ``formal_benchmark.py``): Adam(lr 1e-3, weight decay 1e-5), cosine annealing per
epoch to 1e-5 over ``--epochs``, grad-norm clipping 1.0, batch 64 (shared meshes) / 8 meshes (variable meshes), MSE
on normalised targets, model selection by validation R2 (v1 formula), test metrics of the best model: R2/MSE/MAE
(normalised), NRMSE/SSIM/Pearson (unnormalised), on the full test split, on the first 100 test samples (the v1
paper tables, ``test100_*``) and on extra test sets (``fine_*``: zero-shot 4x resolution for HP/TET).

Data are GPU resident (no DataLoader).  Shared-mesh minibatches are gathered with ``index_select`` into the
``(n, B, C)`` layout; variable meshes are batched block-diagonally (``CochainComplex.batch``), with a background
thread preparing the next batch (complexes that do not fit the GPU budget stay on the CPU and are moved per batch).

Outputs in ``--out``: ``config.json``, ``history.json`` (per epoch: losses, val metrics, lr, train/eval seconds,
peak GPU memory, metric condition numbers), ``best.pt`` / ``last.pt`` (resume: re-run the same command),
``result.json`` (test metrics, wall-clock, parameters, diagnostics).  Seeding is deterministic (python, numpy,
torch, batch order from a CPU generator); CUDA atomics make runs close but not bitwise identical.
```

</details>

### `parse_args`

```python
parse_args(argv: list[str] | None = None) -> argparse.Namespace
```
Command-line interface (see module docstring).

### `run`

```python
run(args: argparse.Namespace, task: TaskData | None = None) -> dict
```
Train + evaluate (or `--eval-v1` / `--check-data`).  Returns the `result.json` content.

### `build_config`

```python
build_config(task: TaskData, args: argparse.Namespace)
```
RHMPConfig for `task` from the CLI arguments and the task defaults.

### `model_output`

```python
model_output(task: TaskData, fwd, xb: dict, K) -> Tensor
```
Model prediction in the task's target space (applies `task.output_map` when present; a map with `takes_complex = True` is called as `output_map(y, K)`, e.g. a per-mesh geometric map on block-diagonal batches of variable meshes).

### `batch_loss`

```python
batch_loss(model, task: TaskData, idx: Tensor, device=None, fwd=None) -> Tensor
```
MSE of one (block-diagonal) minibatch on normalised targets (pooled over all cells/samples/channels).

### `pde_residual`

```python
pde_residual(model, task: TaskData, xb: dict, yb: Tensor, K) -> Tensor | None
```
Relative weak-form PDE residual of the true solution with the model's metric, `(num_graphs|1, B)`.

<details><summary>docstring</summary>

```text
Uses ``task.meta['pde'] = {'degree': k, 'source': 'inputs[j][...,c]', 'bc': 'dirichlet'|'none'}``: the target
(denormalised) is ``u`` on the degree-k cells and the (denormalised) input column ``c`` of degree ``j`` is ``f``;
returns ``model.operator_residual(xb, K, u, f, k, bc=bc)`` or ``None`` when the task / model has no such data.
```

</details>

### `predict`

```python
predict(
    model,
    task: TaskData,
    idx: Tensor,
    batch_size: int,
    device,
    fwd=None,
    cache: dict | None = None,
    source: dict | None = None,
)
```
Normalised predictions and targets for samples `idx`.

<details><summary>docstring</summary>

```text
Args:
    source: optional extra test set ``{'K', 'inputs', 'target'}`` (lists) instead of the task's own data.
Returns:
    ``(pred, target)``: ``(N, n_t, O)`` tensors for shared meshes or equal-size variable meshes (stacked, so
    that R2 uses the v1 per-position mean), else lists of ``(n_t_i, O)`` tensors.
```

</details>

### `evaluate`

```python
evaluate(
    model,
    task: TaskData,
    idx: Tensor,
    batch_size: int,
    device,
    *,
    fwd=None,
    cache=None,
    source=None,
    n_samples: int | None = None,
    prefix: str = '',
) -> dict
```
`rhmp.metrics.summarize` of the predictions on `idx` (normalised pred/target + `task.y_stats`).

### `needs_whitney`

```python
needs_whitney(model_cfg: dict) -> bool
```
Whether a saved model configuration uses the Whitney / Galerkin blocks of the complexes (tensor metrics, the energy readout, the energy amplitude, the field-energy invariant of Picard solve layers); the loaders drop them otherwise.

### `main`

```python
main(argv: list[str] | None = None) -> None
```
CLI entry point (`python3 -m rhmp.train`).


## Task registry: `rhmp.tasks`

Load_task and the paper / synthetic tasks.

<details><summary>module docstring</summary>

```text
Task registry: ``load_task(name, root, native=True, device='cuda') -> TaskData``.

Paper tasks (v1 pickles in ``datasets/*.pkl``; v1 protocol: sequential 70/15/15 split, train-part normalisation):

======  =========================================================================================================
name    content (legacy = v1 exact inputs/outputs;  native = cochain inputs/outputs where the physics lives)
======  =========================================================================================================
T1      CNS vorticity, 32x32 grid (triangulated). legacy: (rho,Vx,Vy,p) at nodes. native: (rho,p) at nodes +
        velocity 1-form on edges ``V_e = (V_s+V_d)/2 . (x_d-x_s)``.  target: vorticity at nodes.
T1q     T1 on the quad CW complex ``CochainComplex.from_grid(cell='quad')`` (same data, node order matched).
T1qe    T1q with an additional even edge column (half-difference of the endpoint velocities along the edge).
T1qn    T1qe with the oriented face circulation as an even face input and a vertex readout (no output map).
T1p     T1 on its periodic grid (flat torus ``from_grid((32, 32), 1/32, periodic=True)``; the simulation and the
        target stencil are periodic).  native: periodic quads with the T1qe inputs (wrap-around edges included,
        minimal-image edge vectors).  legacy: v1 node inputs on the periodic triangulated grid (baselines).
T1pm    native T1p + log rho and log p averaged to edges and faces as even columns (seen by the metric heads).
T1pml   T1pm + (log rho, log p) as additional vertex inputs.
T1pf    T1p (T1pmf: T1pm, T1pmlf: T1pml) on the same trajectories coarse-grained by 4 x 4 block averages instead of
        strided subsampling (``datasets/v2/T1_cns_vorticity_blockavg.pkl``; aliasing analysis).
T1g     T1 on a graded, anisotropic periodic 32x32 grid (nodes = a non-uniform subset of the 128x128 simulation grid,
        spacings 1-8 fine cells, aspect ratios 1/8-8; ``datasets/v2/T1_cns_vorticity_graded.pkl``); target = the
        fine-grid (spectral) vorticity at t+1 at the nodes.  native: T1p inputs on periodic quads, patch-area
        normalised oriented face->node map; legacy: node inputs on the triangulated grid.  ``extra_tests``: zero-shot
        grids U32, G2, U64, G1rot (100 paper-evaluation samples each).  T1gml: T1pml inputs (legacy: + vertex logs);
        T1gml_mean: T1gml with the plain oriented face->node mean (native only); T1gmla: T1gml with the scale-dependent
        geometry descriptors in one fixed length unit instead of relative to each mesh's median (native only).
T2      torus advection-diffusion, scalar -> scalar at nodes (legacy == native).  The velocity, drawn anew for
        every sample by the generator, is not an input: the target is not a function of the input.
T2v     T2 with the velocity as an input (same mesh, samples, targets; ``datasets/v2/T2v.pt`` from
        ``gen_T2v.py``).  native: f at nodes + velocity one-form (odd) and half-difference (even) on edges.
        legacy: (f, Vx, Vy, Vz) at nodes.
T2a     anisotropic transport on the T2 torus (``rhmp/tasks/torus.py``, ``gen_T2a.py``): divergence-free velocity,
        field-aligned diffusion along helical field lines, ratio 1..100, accurate conservative reference.  native: f;
        edge velocity one-form + half-difference + cos^2(t_e, b) + log t_e^T D t_e; face log D_par, log D_perp.
        legacy: (f, Vx, Vy, Vz, bx, by, bz, log D_par, log D_perp) at nodes.
T3      ellipsoid coexact flow psi -> v = n x grad psi.  node_vector readout in both modes; legacy keeps the v1
        per-component target normalisation (and v1-style ``direct`` vector readout), native uses isotropic scaling
        (E(n)-consistent) and the least-squares readout.
T3g     T3 with the exact structure of its targets: readout ``grad`` (predict a stream function phi, w = d0 phi on
        edges) + the fixed edge->vertex map of the generator (P1 face gradients averaged to the vertices, v = n x g).
        Native only.
T5      Maxwell/Poisson rho -> E.  legacy: 2 node scalars (v1). native: node_vector (``direct`` readout = the
        generator's definition E_i = mean_e E_e t_e), isotropic target scaling.
T5g     T5 with the exact structure of its targets: readout ``grad`` (predict a potential phi, E_e = d0 phi on
        edges) + the fixed v1 edge->node average ``v_i = sum_e E_e t_e / deg_i`` (the generator's definition; the
        map applied to the generator's -d0 phi reproduces the stored targets to 3e-7).  Native only.
T6      U(1) Wilson loop. legacy: node encoding of theta. native: theta as an edge cochain in exact gauge
        connection mode (``connection_dims={1:1}``). target: plaquette flux averaged to nodes (v1 target).
T6f     native T6 inputs, target = plaquette flux ``d1 theta`` on faces (``cochain`` readout, degree 2).
T7      SU(2) Yang-Mills. legacy: node encoding of A (9 cols). native: A on edges (3 odd cols).  target F at nodes.
T7f     native T7 inputs, target = F = dA + [A,A] on faces (3 cols, ``cochain`` readout, degree 2).
T8      AirfRANS pressure, one mesh per sample (block-diagonal batches), v1 70/30 split with the last 10 % of the
        train part as validation set (v1 selected on the test part).  legacy == native (node scalars).
T8v     T8 + the inflow velocity as an odd edge 1-form ``V_inf . (x_d - x_s)`` (native only; node inputs, split,
        normalisation and block-diagonal batches as T8).  The scalar angle of attack alone cannot tell the pressure
        side from the suction side for a reflection-invariant model; the 1-form carries the flow direction.
T6_100K 100K-face Wilson loop (legacy only; used by the step benchmark).
======  =========================================================================================================

Orientation output maps (native T1/T1q/T3/T6/T7): vorticity, magnetic flux and ``n x grad psi`` are pseudo-scalars /
pseudo-vectors (defined w.r.t. the physical orientation), which the v2 model cannot output because it is exactly
equivariant under relabelling of cell orientations and reflection invariant.  The task then configures the model
with an orientation-free readout and applies a fixed ``rhmp.data.OutputMap`` that supplies the physical orientation:
T1/T1q/T6/T7 -> ``cochain:2`` + oriented face->node mean (sigma_f = +1 for CCW faces); T3 -> ``node_vector`` for the
true vector g, reported as v = n x g.  Targets and metrics stay exactly v1's.  ``load_task(..., output_map=False)``
disables it (ablation).  Legacy modes keep v1-like readouts (T1/T5 legacy inputs/targets are frame components, which
an isotropic equivariant model cannot relate to directions, so R2 stays near 0 there by construction).

Extension suite (``rhmp/tasks/suite.py``): SURF / SURF_heat (+ _geo / _topo transfer sets), DYN / DYNfix (+ _delta,
_cons, _cons_mass), HP_qual_graded / HP_qual_sliver (+ _ref) and the anisotropy tasks of ``rhmp/tasks/aniso.py``
(AHP, ASURF, ACURL, ADARCY) resolve through ``load_task`` as well (hook below; the paper tasks do not depend on those
modules).

AirfRANS tasks (``rhmp/tasks/airfrans.py``, generator ``datasets/generators/gen_airfrans.py``): T8 resolution views
``T8_n500`` / ``T8_n2000`` / ``T8_n8000`` (test only), T8 with the chord frame (``T8fr``, ``T8xy``, ``T8frxy``) and the
tasks on conforming near-field meshes (``AFV``, ``AFVp``, ``AFU``, ``AFUv``, ``AFD``, ...) resolve through the same hook.

Maxwell tasks (``rhmp/tasks/maxwell.py``, generator ``datasets/generators/gen_maxwell.py``): ``MXS``, nonlinear
magnetostatics of an EI-core electromagnet with saturable steel on one fixed mesh (vertex field B from currents and
material phases), ``MXA``, its anisotropic version (grain-oriented / laminated steel with rolling directions, several
layout families, a held-out family; generator ``gen_maxwell_aniso.py``; ablation ``MXA_noorient``), and ``MXE``,
linear electrostatics of an electrical capacitance tomography sensor (vertex field E from electrode potentials, space
charge and material phases).

Synthetic tasks (``datasets/v2/*.pt``, generators in ``datasets/generators/gen_*.py``), variable meshes:

* ``HP[<target>][_k<kappa>][_aniso[<R>]]`` (default kappa=100): 2-D hetero(-anisotropic) Poisson on variable meshes.
  Inputs: f (nodes); on edges (even) log sigma, or for anisotropic sets the log of the projected conductivity
  ``t^T Sigma t``; on faces (even) log sigma, or for the tensor sets ``_aniso<R>`` the invariants
  ``(log det Sigma, log anisotropy ratio)``.  ``_aniso10`` / ``_aniso100``: spatially varying SPD tensor field with
  a random principal direction and an eigenvalue ratio in [1, R]; ``_aniso``: legacy constant-ratio set.
  Targets (``<target>``): none -> u (nodes); ``flux`` -> ``-sigma_e d0 u`` (odd edge cochain, ``cochain:1``; its values
  scale with the edge length); ``fluxd`` -> flux density ``-sigma_e d0 u / |e|`` (resolution independent);
  ``fluxfem`` -> the conservative FEM flux ``-w_e d0 u`` (``w_e = -K_{src,dst}``; ``d0^T j = -M f`` exactly at
  interior nodes); ``grad`` -> ``E = -d0 u`` predicted through the exact readout ``grad`` (E = d0 phi, curl free).
  Extra test set ``fine``: first 500 test samples re-solved on meshes with 4x the nodes (zero-shot transfer).
* ``TET[<target>][_k<kappa>]``: 3-D tetrahedral Poisson; inputs f, log sigma on edges/faces/tets; same targets.
```

</details>

- `rhmp.tasks.TaskData`: re-export of `rhmp.data.TaskData` (documented there).
- `rhmp.tasks.Stats`: re-export of `rhmp.data.Stats` (documented there).
### `load_task`

```python
load_task(
    name: str,
    root: str | None = None,
    *,
    native: bool = True,
    device='cuda',
    star: str = 'cotan',
    **kw,
) -> TaskData
```
Load a task as a GPU-resident, normalised `TaskData`.

<details><summary>docstring</summary>

```text
Args:
    name: task name (see module docstring).
    root: repository root or datasets directory (``None``: ``$RHMP_DATA_ROOT`` if set, else this repository).
    native: native-cochain inputs/outputs (True) or v1 legacy inputs/outputs (False).
    device: target device of all tensors and complexes.
    star: reference Hodge star of the complexes (``cotan`` | ``barycentric`` | ``unit``; tetrahedral meshes
        use ``barycentric`` when ``cotan`` is requested).
    **kw: task-specific options (e.g. ``fine=False`` to skip the HP/TET resolution-transfer set,
        ``max_samples`` to truncate variable-mesh sets for smoke tests, ``cache=False``, ``whitney=False`` to
        drop the Whitney blocks that only tensor metrics use (41 % of a triangle complex), ``keep_on``,
        ``abs_scale=True`` to append the absolute mesh scale ``log(median edge length)`` as a constant even vertex
        column (the model is scale-free; use it when the physics has a fixed length scale), see
        :func:`rhmp.data.add_abs_scale`).
Returns:
    TaskData.
```

</details>

### `list_tasks`

```python
list_tasks() -> list[str]
```
Canonical task names (HP/TET accept further `_k<kappa>` / `_aniso` suffixes; extension-suite names (SURF*, DYN*, HP_qual_*, anisotropy tasks) and the AirfRANS tasks (`rhmp.tasks.airfrans`) are appended when their modules are importable).

### `task_defaults`

```python
task_defaults(name: str) -> dict
```
Training defaults (`C`, `layers`, `batch`) for a task name.

### `resolve_root`

```python
resolve_root(root: str | None) -> str
```
Return the directory that contains the v1 `*.pkl` files (and `v2/`).

<details><summary>docstring</summary>

```text
``root`` may be the repository root (containing ``datasets/``) or the datasets directory itself.  ``None`` means
the environment variable ``RHMP_DATA_ROOT`` if it is set (same two forms), else the repository this package
lives in.
```

</details>

### Constants

- `TASK_DEFAULTS`: dict with keys {'T1', 'T1q', 'T1qe', 'T1qn', 'T1p', 'T1pm', 'T1pml', 'T1pf', 'T1pmf', 'T1pmlf', 'T1g', 'T1gml', ...}


## Torus transport task: `rhmp.tasks.torus`

Anisotropic transport on the torus (T2a).

<details><summary>module docstring</summary>

```text
Anisotropic scalar transport on the torus (task T2a; generator ``datasets/generators/gen_T2a.py``).

``d_t u + div(v u) = div(D grad u)`` on the T2 torus with a divergence-free velocity (per-sample toroidal and poloidal
amplitudes of T2 plus a random solenoidal perturbation) and field-aligned diffusion ``D = D_perp P + (D_par - D_perp)
b b^T`` along helical field lines ``b`` (per-sample pitch, smoothly varying), anisotropy ratio ``D_par / D_perp`` in
1..100; the T2 initial conditions; ``u(T)``, ``T = 0.2``, from an accurate conservative reference solver (P1 tensor
diffusion, exactly divergence-free dual fluxes, Crank-Nicolson with 200 sub-steps).

Inputs:

* native (RHMP): degree 0 ``f``; degree 1 ``[theta, delta, cos^2(t_e, b), log(t_e^T D t_e)]`` (the velocity one-form
  ``(V_s + V_d)/2 . (x_d - x_s)`` (odd), its half-difference (even, see ``rhmp.tasks.paper.t2v_inputs``), and the two
  E(n)-invariant anisotropy descriptors (even; means over the incident faces)); degree 2 ``[log D_par, log D_perp]``
  (even, constant per sample).  Suggested: ``--metric-ref 1:3`` (the projected diffusivity as the fixed offset of the
  edge metric) and ``--material 1:2,2:2`` (the anisotropy columns reach only the metric heads), the full tensor metric,
  and the transport term with ``metric='star'`` (geometric advective flux).
* legacy (baselines): node features ``(f, Vx, Vy, Vz, bx, by, bz, log D_par, log D_perp)``.

Target ``u(T)`` at the vertices.  Same split as T2 (sequential 70/15/15).  The zero-shot sets of the test split
(velocity and anisotropy above the training ranges, roll-out to ``2T``, a 2x finer and a 2x coarser torus) are read
by ``scripts/eval_torus_aniso.py`` through :func:`t2a_file` / :func:`t2a_inputs`.
```

</details>

### `t2a_file`

```python
t2a_file(droot: str) -> dict
```
`datasets/v2/T2a.pt` (`datasets/generators/gen_T2a.py`).

### `t2a_inputs`

```python
t2a_inputs(
    X: Tensor,
    V: Tensor,
    b: Tensor,
    cos2: Tensor,
    logp: Tensor,
    Dpar: Tensor,
    Dperp: Tensor,
    K,
    native: bool,
) -> dict[int, Tensor]
```
Raw (unnormalised) T2a inputs.

<details><summary>docstring</summary>

```text
Args:
    X: ``(N, n0)`` initial field; V, b: ``(N, n0, 3)`` velocity and field direction at the vertices;
    cos2, logp: ``(N, n1)`` edge descriptors; Dpar, Dperp: ``(N,)``; K: the complex of the mesh.
    native: native cochain inputs (RHMP) or the legacy node features (baselines).
```

</details>

### `t2a_complex`

```python
t2a_complex(points: Tensor, faces: Tensor, edges: Tensor, star: str, device)
```
Complex of a torus mesh of the generator and the map of the generator's edge data onto its edges.

<details><summary>docstring</summary>

```text
Returns:
    ``(K, perm)``: even edge data of the generator ``x (N, n1)`` are ``x[:, perm]`` in the complex's edge order.
```

</details>

### `load_T2a`

```python
load_T2a(droot, *, native, device, star, **_)
```
T2a (see the module docstring): native `in_dims {0: 1, 1: 4, 2: 2}`, `even_dims {1: 3, 2: 2}`; legacy `in_dims {0: 9}`; node target.


## Magnetostatics tasks: `rhmp.tasks.maxwell`

Nonlinear (anisotropic) magnetostatics and electrostatics (MXS, MXA, MXE).

<details><summary>module docstring</summary>

```text
Nonlinear magnetostatics task family MXS (generator ``datasets/generators/gen_maxwell.py``).

Physics: 2-D cross-section of a three-limb EI-core electromagnet with saturable steel (Froehlich-Kennelly law,
``mu_r,init = 3000``, ``J_s = 2.0 T``), three windings with independent signed current densities (|J| <= 6 A/mm^2),
a movable and possibly tilted I-armature, random pole shoes and 0-2 external iron pieces, all on **one fixed mesh**
(2,085 nodes; the movable iron is a per-face material indicator, i.e. staircase interfaces on a regular lattice band).
Reference: P1 finite elements with damped Newton (``|R| <= 1e-10 |F|``) for ``-div(nu(|B|) grad A) = J``, ``A = 0`` on
the box, ``B = (dA/dy, -dA/dx)``.  Units: mm, T, A/mm^2.  Split: sequential 70/15/15 of 5000 samples (train
0..3499, validation 3500..4249, test 4250..4999), ``meta['paper_eval']`` = 4250..4349 ("test100").

==========  ========================================================================================================
name        content
==========  ========================================================================================================
MXS         target: vertex flux density ``B_i`` (T, 2 components) = area-weighted mean over the incident triangles of
            the element fields ``B_f``.  native (RHMP): vertex current density ``J_i`` (lumped: ``M J = F``, the
            consistent load of the element-wise current), even edge columns = phase fractions of the incident faces
            (air, copper, iron), even face columns = phase one-hot (material columns for solver mode); readout
            ``grad`` (``Phi = d0 phi``: edge fluxes of B, exactly closed, ``d1 Phi = 0``) followed by the fixed map
            :class:`FluxToNodeField` (``B_i = R_-90`` of the area-weighted mean element gradient).  legacy (node
            baselines): vertex ``(J_i, f_air, f_copper, f_iron)`` with ``f`` the phase fractions of the barycentric dual
            cell -> vertex ``(B_x, B_y)``.  Target scaling: one isotropic RMS (rotation-consistent) in both modes.
==========  ========================================================================================================

Secondary family ``MXE`` (:func:`load_mxe`): linear electrostatics of an electrical capacitance tomography sensor
(eight electrodes at random potentials as holes of the mesh, grounded screen, insulating pipe wall, oil / water / gas
flow pattern with permittivity contrast 80, space charge), target ``E = -grad phi`` at the vertices; same split,
extra sets and protocol; the solve layer takes the electrode potentials as inhomogeneous Dirichlet data
(``solve_bc='lift'``).

Anisotropic family ``MXA`` (:func:`load_mxa`, generator ``datasets/generators/gen_maxwell_aniso.py``): grain-oriented
(easy : hard permeability 25) and laminated (5) steel with a rolling direction per piece, non-oriented steel, E-I / U-I
electromagnets and shunt transformers with random dimensions, gaps, constructions and windings on one fixed lattice
mesh; models receive phase indicators and the rolling direction (RHMP: E(n)-invariant edge columns, baselines: frame
components), never the law; extra sets ``ood15``, ``fine``, ``coarse`` and ``heldout`` (a layout family absent from
training).

Extra test sets (``extra_tests``; list format, one complex per sample): ``ood15`` (test samples 4250..4749 with all
currents x1.5, base mesh), ``fine`` (the same 500 samples on a refined mesh, 7,953 nodes) and ``coarse`` (566 nodes);
each also as ``<name>_test100`` (its first 100 samples = the test100 instances).  No model ever sees the reluctivity:
the constitutive law has to be learned from the field data.
```

</details>

### `mxs_task_defaults`

```python
mxs_task_defaults(name: str) -> dict
```
Training defaults (`C`, `layers`, `batch`).

### class `FluxToNodeField`

```python
FluxToNodeField(base=None, scale: float = 100.0, field: str = 'curl')
```
Edge fluxes `Phi = d0 A` (flux of B through each edge, right normal of src -> dst) -> vertex fields `B_i = sum_{f ∋ i} |f| R grad_f(A) / sum_{f ∋ i} |f|` with `R (gx, gy) = (gy, -gx)` (`B = curl A e_z`), times the fixed unit length `scale` (mm).  `field='grad'`: potential differences `d0 phi` -> `E_i = -sum |f| grad_f(phi) / sum |f|` (`E = -grad phi`, a true vector; task MXE).

<details><summary>docstring</summary>

```text
``grad_f(A) = s_0 Phi_{e0} g_1 - s_2 Phi_{e2} g_2`` from the two face edges at local vertex 0 (``g_k`` the P1
barycentric gradients, ``s_j`` the face-edge orientation signs of ``K.meta``): a fixed linear map that depends only on
differences of A (exact for the generator's targets), on the geometry and on the orientation of the plane (B is a
pseudo-vector), not on vertex labels or edge / face orientation conventions.  Computed per complex (block-diagonal
batches of meshes, rotated or refined meshes); ``takes_complex = True``.  Called without a complex it uses the
task's own mesh (``base``), so linear-readout fits work unchanged.  Output ``(n0, B, 2)``.
```

</details>

#### `FluxToNodeField.matrix`

```python
matrix(K) -> tuple[Tensor, Tensor]
```
CSR `(2 n0, n1)` (rows `2 i + c`) and its transpose, float32 on `K`'s device (cached per complex).

#### `FluxToNodeField.to`

```python
to(device) -> FluxToNodeField
```
### `reluctivity`

```python
reluctivity(B)
```
Reluctivity `nu = H / B` (m/H) of the steel (numpy or torch, `B >= 0` in T): the explicit inverse of the Froehlich-Kennelly law `B = mu0 H + Js H / (Hk + H)` (identical to the generator's `nu_iron`).

### `load_mxs`

```python
load_mxs(
    name: str,
    droot: str,
    *,
    native: bool = True,
    device='cuda',
    star: str = 'cotan',
    fine: bool = True,
    max_samples: int | None = None,
    whitney: bool = True,
    output_map=None,
    **_,
) -> TaskData
```
Task `MXS` (see the module docstring).

<details><summary>docstring</summary>

```text
Args:
    native: RHMP inputs (vertex current, edge / face phase columns; ``grad`` readout + :class:`FluxToNodeField`)
        or the node inputs of the baselines.
    fine: also load the extra test sets (``ood15``, ``fine``, ``coarse`` and their ``*_test100`` subsets).
    max_samples: smoke tests: keep ``max_samples // 3`` samples of every split part.
    output_map: ``False`` drops the output map (the model must then predict the vertex field directly).
```

</details>

### `load_mxe`

```python
load_mxe(
    name: str,
    droot: str,
    *,
    native: bool = True,
    device='cuda',
    star: str = 'cotan',
    fine: bool = True,
    max_samples: int | None = None,
    whitney: bool = True,
    output_map=None,
    **_,
) -> TaskData
```
Task `MXE`: electrical capacitance tomography sensor (8 electrodes at random potentials on an insulating pipe, grounded screen, oil / water / gas flow pattern with permittivities 2.2 / 80 / 1, wall 3.5, triboelectric space charge), `-div(eps grad phi) = rho / eps_0`, target `E = -grad phi` at the vertices (V/mm).

<details><summary>docstring</summary>

```text
native: vertex ``(q, V)`` (lumped space charge rho/eps_0, 0 on the fixed vertices; electrode potential on the
electrode outlines, 0 elsewhere: the Dirichlet data), even edge / face phase columns (5 phases); readout ``grad``
(``d0 phi``, exactly curl-free) + :class:`FluxToNodeField` with ``field='grad'``.  The electrode outlines and the
screen are the mesh boundary (``K.boundary[0]``; electrodes are holes), i.e. the fixed cells of ``solve_bc='lift'``.
legacy: vertex ``(q, V, dual-cell phase fractions)`` -> vertex ``(E_x, E_y)``.  Same split / extra sets as MXS.
```

</details>

### `load_mxa`

```python
load_mxa(
    name: str,
    droot: str,
    *,
    native: bool = True,
    device='cuda',
    star: str = 'cotan',
    fine: bool = True,
    max_samples: int | None = None,
    whitney: bool = True,
    output_map=None,
    orientation: bool = True,
    **_,
) -> TaskData
```
Task `MXA`: nonlinear magnetostatics with grain-oriented (GO, easy : hard permeability 25) and laminated (LAM, 5) steel with a rolling direction per piece, non-oriented steel (NO), air and copper, in E-I and U-I electromagnets and shunt transformers of random dimensions, gaps, constructions and windings, all on one fixed lattice mesh (1,818 nodes; generator `datasets/generators/gen_maxwell_aniso.py`).  Target: vertex B (T).

<details><summary>docstring</summary>

```text
native (RHMP): vertex current density (lumped), even edge columns ``(f_air, f_copper, f_NO, f_GO, f_LAM,
cos^2(edge, rolling direction) over the incident GO faces, the same over the incident LAM faces)`` (E(n)-invariant:
the edge and the rolling direction rotate together), even face columns = phase one-hot; readout ``grad`` +
:class:`FluxToNodeField`.  legacy (node baselines): vertex ``MXA_LEGACY_COLUMNS`` (current, dual-cell phase
fractions, dual-cell (cos 2phi, sin 2phi) of the GO and LAM rolling directions: frame components, which rotate
with the configuration, ``meta['frame_pairs']``).  Isotropic target scaling.  Split: sequential 70/15/15 of 5000,
test100 = 4250..4349.  Extra tests (list format): ``ood15`` (test samples 4250..4749 with all currents x1.5),
``fine`` / ``coarse`` (the same layouts on the 4 mm / 16 mm meshes, 7,255 / 459 nodes), ``heldout`` (500 samples
of the EE family, which never occurs in training), each also as ``<name>_test100`` (the test100 layouts; for
``heldout`` its first 100 samples).  ``orientation=False`` (task ``MXA_noorient``, an ablation) drops the
rolling-direction columns (native: the two cos^2 edge columns; legacy: the four frame columns).
```

</details>

### `face_onehot`

```python
face_onehot(phase: Tensor) -> Tensor
```
`(..., n2)` int phase -> `(..., n2, 3)` float one-hot (air, copper, iron).

### `triangle_areas`

```python
triangle_areas(K) -> Tensor
```
`(n2,)` unsigned triangle areas (float64, units of `K.pos` squared).

### `edge_directions`

```python
edge_directions(K) -> Tensor
```
`(n1, 2)` unit edge vectors (float64; the sign is irrelevant for the squared cosines).

### `rotate_frame_columns`

```python
rotate_frame_columns(x: Tensor, stats, pairs, theta) -> Tensor
```
Normalised node inputs `x (..., F)` of a configuration rotated by `theta` (radians; a number or a tensor broadcastable to `x[..., 0]`): every column pair `(i, j)` in `pairs` holds `(cos 2 phi, sin 2 phi)` frame components (in physical units) and turns by `2 theta`; the other columns are unchanged.

### Constants

- `MXS_TASKS`: dict with keys {'MXS', 'MXE', 'MXA', 'MXA_noorient'}
- `MXS_DEFAULTS`: dict with keys {'MXS'}
- `PHASE_NAMES`: ('air', 'copper', 'iron')
- `ECT_PHASES`: ('air', 'wall', 'oil', 'water', 'gas')
- `MXA_PHASES`: ('air', 'copper', 'NO', 'GO', 'LAM')
- `MU0`: 1.2566370614359173e-06
- `NU0`: 795774.7154594767
- `JS`: 2.0
- `MUR_INIT`: 3000.0
- `HK`: 530.6933747645726


## AirfRANS tasks: `rhmp.tasks.airfrans`

Pressure / velocity tasks on body-fitted meshes (AFP*, AFV*, AFU*, AFD, T8*).

<details><summary>module docstring</summary>

```text
AirfRANS task family (generator ``datasets/generators/gen_airfrans.py``; data from the AirfRANS RANS simulations).

All tasks keep the simulation order and the partition of the paper's T8 task: simulations 0..699 are the training
part (normalisation statistics; the loaders hold out 630..699 for model selection, as T8), 700..999 the test part, and
``meta['paper_eval']`` = simulations 850..949 ("test100", the paper's table).  Lengths are in chords, velocities in
m/s, pressures in Pa, vorticities in 1/s, the angle of attack in radians.

==================  =========================================================================================  ======
name                content                                                                                   mesh
==================  =========================================================================================  ======
T8_n500/2000/8000   test-only resolution views of T8: the same construction (uniform random CFD nodes,         T8res
                    Delaunay of the convex hull) with 500 / 2000 / 8000 nodes on the 300 test simulations
                    (nested point sets: the prefixes of T8's random permutations; n2000 reproduces the T8
                    test split exactly); x = (sdf, U, a), y = p; ``extra_tests['test100']``.
T8i                 T8 with the AFPi inputs (node (sdf, a), edge inflow 1-form + mean sdf, face mean sdf) and   T8
                    the AFPi output map; native only.
T8fr / T8xy /       T8 (same meshes, split, target and normalisation) with the frame that the gauge-CNN        T8
T8frxy              baselines see through their edge angles: ``fr`` = odd edge 1-forms (inflow
                    ``U (cos a dx + sin a dy)``, ``dx_e``, ``dy_e``), ``xy`` = chord-frame coordinates (x, y)
                    as two more node columns (all models).
AFV                 vorticity on conforming near-field meshes, Stokes target: the mean vorticity of the CFD    AFM
                    field over the star of every vertex, ``sum_{f ∋ i} sigma_f (d1 theta)_f / sum_{f ∋ i}
                    A_f`` with ``theta_e = int_e u.dl`` (accurate line integrals of the CFD velocity; Stokes:
                    ``(d1 theta)_f`` is the integral of the vorticity over triangle f).  native: node (sdf, U, a),
                    edge theta as a gauge connection (the model sees only d1 theta: exact invariance under
                    u -> u + grad lambda); readout ``cochain:2`` + the fixed area-weighted oriented face ->
                    vertex map.  legacy (node baselines): node (Vx, Vy, sdf, U, a) of the CFD.
AFVp                as AFV, target = the CFD point vorticity at the vertices (wall: the wall vorticity).       AFM
AFV_trap /          equal-information variants: theta = the trapezoid rule of the nodal CFD velocities (the     AFM
AFVp_trap           information the node baselines get); AFV_trap's target is then the star average of the
                    P1 curl of the nodal velocities (the discrete Stokes target of the T1 / v1 vorticity data).
AFU                 velocity flux 1-cochain ``F_e = int_e u.n ds`` (n = right normal of src -> dst; accurate    AFM
                    quadrature, closed per triangle to ~1e-4 for the incompressible CFD field).  native: node
                    (sdf, U, a, x, y), edge inflow 1-form; readout ``grad`` (F = d0 psi: exactly closed,
                    zero net flux out of every triangle).  legacy: node (sdf, U, a, x, y) -> node (Vx, Vy).
AFUv                native AFU inputs and ``grad`` readout, trained on the node velocities through the fixed   AFM
                    map F -> u (least-squares gradient of psi at every vertex, rotated: u = (psi_y, -psi_x);
                    u = 0 at wall vertices, the no-slip condition).
AFP / AFPfr /       pressure on the conforming meshes (the geometry-respecting counterpart of T8): node         AFM
AFPxy / AFPfrxy     (sdf, U, a); ``fr``: + odd edge 1-forms (inflow, dx, dy; native only); ``xy``: + chord-frame
                    coordinates as node columns (all models).
AFPi / AFPi0 /      pressure with the inflow in its covariant form (dynamic similarity).  native: node (sdf, a)  AFM
AFPiU / AFPiR /     [AFPi0: (sdf); AFPiU: (sdf, U, a); AFPiR: (sdf, a, log U)], edge inflow 1-form
AFPs                ``theta_e = u_inf . (x_dst - x_src)`` (odd; the amplitude column of ``--amplitude 1:1``) and the
                    mean sdf of the edge / face vertices as even edge / face columns (seen by the metric heads).
                    AFPs: no 1-form, node (U, sdf, a) with U as the amplitude column (``--amplitude 0:1``).  The
                    complexes carry ``meta['dirichlet'] = {0: outer-ring vertices}`` and ``meta['cohomology'] = {1:
                    generator of H^1 of the annulus}`` for projection layers.  Target and normalisation are those of
                    AFP (raw pressure, mean / std of simulations 0..699); AFPi / AFPi0 / AFPiR / AFPs add the fixed
                    output map ``y -> y - mean / std`` so that a model with the output factor ``s^2`` (``--amplitude``
                    ``--amplitude-power 2 --amplitude-energy``) is exactly homogeneous of degree two in the inflow in
                    physical units.  legacy (node baselines): node (sdf, U, a), i.e. AFP.
AFPpf               control: AFP with the Bernoulli pressure of the P1 potential flow with the Kutta condition    AFM
                    (:func:`potential_reference`: reference metric, no learning) as a fourth node column
                    ``(sdf, U, a, p_ref)`` for every model; target and normalisation of AFP.
AFPcp / AFPcpxy     controls: target = pressure coefficient ``p / (U^2 / 2)`` (mean / std normalisation); legacy   AFM
                    node (sdf, U, a) [AFPcpxy: + chord-frame (x, y)]; native: the AFPi inputs, no output map.
                    Predictions are converted to physical units by ``scripts/airfrans_pressure_eval.py``.
AFD                 derived diffusion task: P1 FEM solution of -div(kappa grad T) = f, T = 0 on the airfoil    AFM
                    and the outer ring, kappa = 1 + nu_t / nu of the simulation (the RANS eddy-viscosity
                    ratio, contrast up to ~1e3), f = random Gaussian sources.  The solution is computed by the
                    generator; only the meshes and kappa come from AirfRANS.  native: node f, even edge / face
                    log kappa (material columns for solver mode); legacy: node (f, mean log kappa).
==================  =========================================================================================  ======

The conforming tasks have ``extra_tests['fine']`` (the 300 test simulations on meshes with about 3.6x the nodes, same
construction with h halved, independent random nodes; zero-shot resolution transfer) and ``'fine_test100'``, and, when
``AFM_coarse.pt`` exists, ``'coarse'`` / ``'coarse_test100'`` (h doubled, about 0.3x the nodes).
```

</details>

### `airfrans_task_defaults`

```python
airfrans_task_defaults(name: str) -> dict
```
Training defaults (`C`, `layers`, `batch` = meshes per block-diagonal batch).

### class `StarAverageMap`

```python
StarAverageMap(out_dim: int = 1, scale: float = 1.0)
```
Oriented face 2-cochain -> vertex density: `y_i = A_ref sum_{f ∋ i} sigma_f c_f / sum_{f ∋ i} |A_f|`.

<details><summary>docstring</summary>

```text
``sigma_f`` is the sign of the signed area (+1 for counter-clockwise faces, the physical orientation of the plane).
For a face circulation ``c = d1 theta`` this is the mean vorticity over the star of vertex i (Stokes; the interior
edges of the star cancel).  ``A_ref`` is a fixed constant that only sets the units of the model output (the loaders
use the median star area of the training meshes, so that predictions at initialisation scale like the targets).  Parameter free and linear; computed from the batch complex (block-diagonal batches of
different meshes), so ``takes_complex = True`` (``rhmp.train.model_output`` passes the complex).
```

</details>

#### staticmethod `StarAverageMap.weights`

```python
weights(K) -> tuple[Tensor, Tensor, Tensor]
```
#### `StarAverageMap.to`

```python
to(device) -> StarAverageMap
```
### class `FluxVelocityMap`

```python
FluxVelocityMap(model_readout: str = 'grad', scale: float = 1.0)
```
Edge flux 1-cochain `F = d0 psi` (`F_e = int_e u.n ds`, n = right normal) -> vertex velocities.

<details><summary>docstring</summary>

```text
At every vertex the gradient of the stream function is the least-squares fit ``g_i = argmin sum_{e ∋ i}
(g . dx_e - dpsi_e)^2 / |dx_e|^2`` over the incident edges (``dpsi_e = +-F_e``, oriented away from i), and
``u_i = (g_y, -g_x)``; wall vertices (:func:`wall_vertices`) get ``u = 0`` (the no-slip condition).  ``scale`` is a
fixed unit factor (a data-set constant).  Fixed and linear; per batch from the complex (``takes_complex = True``).
Output ``(n0, B, 2)``.
```

</details>

#### staticmethod `FluxVelocityMap.operator`

```python
operator(K) -> tuple[Tensor, Tensor, Tensor, Tensor]
```
`(rows, cols, vals)` of the `(2 n0, n1)` matrix and the wall mask (float64).

#### `FluxVelocityMap.to`

```python
to(device) -> FluxVelocityMap
```
### class `OffsetMap`

```python
OffsetMap(offset: float, scale: float = 1.0, model_readout: str = 'node_scalar', out_dim: int = 1)
```
Fixed affine output map `y -> scale * y + offset` (constants of the task).  With `offset = -mean / std` of a mean / std normalised target, a model whose raw output is homogeneous in an input amplitude stays exactly homogeneous in physical units (`std * scale * y_model`) while the loss and the metrics use the task's normalised target; `scale` is a unit factor (the loaders choose it so that the amplitude factor of the model output is of order one on the training simulations).

#### `OffsetMap.to`

```python
to(device) -> OffsetMap
```
### `wall_vertices`

```python
wall_vertices(K) -> Tensor
```
`(n0,)` bool: boundary vertices on the airfoil.  Every AirfRANS airfoil has its chord on [0, 1] x {0} and the outer boundary of the near-field meshes is at distance 0.5 from it, so the airfoil boundary vertices are the boundary vertices in the box (-0.25, 1.25) x (-0.35, 0.35) (checked against the generator's flags by the loader).  Works on block-diagonal batches.

### `afm_eval_source`

```python
afm_eval_source(
    task: str,
    droot: str,
    x_stats: dict,
    y_stats: Stats,
    *,
    base: str = 'AFM',
    ids=None,
    native: bool = True,
    star: str = 'cotan',
    whitney: bool = False,
    rotate: float = 0.0,
) -> dict
```
One evaluation set of a conforming-mesh task, built on the CPU with the statistics of a trained run.

<details><summary>docstring</summary>

```text
Args:
    task: AFM task name; droot: datasets directory; x_stats / y_stats: the run's statistics (CPU).
    base: ``'AFM'`` (ids = simulation indices), ``'AFM_fine'`` or ``'AFM_coarse'`` (ids = positions in that file).
    ids: samples of the file (default: the simulations 850..949, "test100").
    rotate: rigid rotation of the whole configuration in degrees (mesh and inflow; scalar inputs unchanged).
Returns:
    dict with ``K``, ``inputs``, ``target`` (lists, normalised), ``kind`` (vertex kinds: 0 volume, 1 wall, 2 outer
    ring), ``pos0`` (chord-frame positions), ``U``, ``aoa``, ``src`` (simulation indices), and the raw CFD fields
    ``theta_cfd`` (accurate edge circulations of the velocity), ``p`` and ``sdf``.
```

</details>

### `afp_output_map`

```python
afp_output_map(task: str, x_stats: dict, y_stats: Stats, u2_ref: float)
```
The fixed output map of a covariant pressure task for given statistics (`None` for tasks without one).

<details><summary>docstring</summary>

```text
Unit factor: the amplitude column (the inflow 1-form, or U for AFPs) is normalised by its RMS ``c`` over the
training cells, so the output factor of the model is ``(U / c)^2``; with ``scale = c^2 / u2_ref`` (``u2_ref`` = the
mean squared inflow speed of the training simulations) it becomes ``U^2 / u2_ref``, of order one.
```

</details>

### `load_t8_inflow`

```python
load_t8_inflow(
    name: str,
    droot: str,
    *,
    native: bool = True,
    device='cuda',
    star: str = 'cotan',
    cache: bool = True,
    max_samples: int | None = None,
    whitney: bool = True,
    **_,
) -> TaskData
```
`T8i`: T8 (meshes, split, target, normalisation and test100 of `rhmp.tasks.paper.load_T8`) with the inputs of `AFPi`: node (sdf, a), edge inflow 1-form `theta_e = u_inf . (x_dst - x_src)` (odd, the amplitude column) and the mean sdf of the edge / face vertices (even), and the fixed output map of `afp_output_map`.  The T8 meshes are Delaunay triangulations of the convex hull of random CFD nodes: the airfoil is not a boundary of the complex, so there is no wall condition and no cohomology generator (projection layers fix the hull vertices).  Native only (the node baselines use `T8`).

### `potential_reference`

```python
potential_reference(K, r: dict) -> Tensor
```
Bernoulli pressure `(U^2 - |u|^2) / 2` of the P1 potential flow around the airfoil with the Kutta condition, without any learning (the reference of `scripts/airfrans_potential_reference.py` computed with the library's operators in float64): the inflow 1-form is projected with the reference star (no flux through the wall, perturbation potential zero on the outer ring), the circulation along the harmonic cochain is fixed by the Kutta condition on the two wall edges at the trailing-edge vertex (the wall vertex of largest chord-frame x), and `|u|^2` is the Galerkin energy density at the vertices.  `K` must carry its Whitney blocks.  `(n0,)`.

### `load_t8res`

```python
load_t8res(
    name: str,
    droot: str,
    *,
    n_pts: int,
    native: bool = True,
    device='cuda',
    star: str = 'cotan',
    max_samples: int | None = None,
    whitney: bool = True,
    keep_on: str = 'auto',
    forms: bool = False,
    xy: bool = False,
    **_,
) -> TaskData
```
Test-only view `T8_n<n_pts>` (300 test simulations; `extra_tests['test100']` = sources 850..949).

<details><summary>docstring</summary>

```text
``xy`` / ``forms``: the inputs of ``T8xy`` / ``T8fr`` (chord-frame coordinates as node columns; odd edge 1-forms),
for zero-shot tests of models trained on those tasks.  Inputs / targets are normalised with the T8 statistics
(stored by the generator; the frame columns with statistics of these meshes); ``rhmp.train --eval-ckpt``
re-normalises with the statistics of the evaluated run anyway.
```

</details>

### `load_t8_frame`

```python
load_t8_frame(
    name: str,
    droot: str,
    *,
    forms: bool,
    xy: bool,
    native: bool = True,
    device='cuda',
    star: str = 'cotan',
    cache: bool = True,
    max_samples: int | None = None,
    whitney: bool = True,
    **_,
) -> TaskData
```
T8 (meshes, split, target, normalisation and test100 of `rhmp.tasks.paper.load_T8`) with the chord frame.

<details><summary>docstring</summary>

```text
``forms``: odd edge 1-forms ``[U (cos a dx + sin a dy), dx_e, dy_e]`` (native only; scale-normalised with the RMS
of the first 700 samples).  ``xy``: chord-frame coordinates (x, y) appended to the node inputs (every model; in
legacy mode the node inputs are ``(sdf, U, a, x, y)``).
```

</details>

### `load_afm`

```python
load_afm(
    name: str,
    droot: str,
    *,
    task: str,
    native: bool = True,
    device='cuda',
    star: str = 'cotan',
    fine: bool = True,
    max_samples: int | None = None,
    whitney: bool = True,
    keep_on: str = 'auto',
    connection: bool = True,
    output_map=None,
    rotate: float = 0.0,
    **_,
) -> TaskData
```
Tasks on the conforming near-field meshes (see the module docstring).

<details><summary>docstring</summary>

```text
Args:
    task: ``'AFV' | 'AFVp' | 'AFV_trap' | 'AFU' | 'AFUv' | 'AFD'``.
    connection: AFV*: the circulation 1-form enters as a gauge connection (only ``d1 theta`` reaches the model);
        ``False``: as an ordinary odd edge input.
    fine: also load ``extra_tests['fine']`` / ``['fine_test100']`` (4x meshes of the test simulations).
```

</details>

### `flux_divergence`

```python
flux_divergence(K, F: Tensor) -> Tensor
```
Net outward flux of every triangle, `(d1 F)_f * sigma_f` (`(n2,)` or `(n2, B)`): zero for an exactly conservative (closed) flux cochain.

### `trapezoid_flux`

```python
trapezoid_flux(pos: Tensor, edges: Tensor, u: Tensor) -> Tensor
```
`F_e = (u_src + u_dst)/2 . n_e |e|` with `n` the right normal of `src -> dst`: `(n1,)` (or `(n1, B)` for `u` of shape `(n0, B, 2)`).

### `edge_flux_of_nodes`

```python
edge_flux_of_nodes(pos: Tensor, edges: Tensor, u: Tensor) -> Tensor
```
`F_e = (u_src + u_dst)/2 . n_e |e|` with `n` the right normal of `src -> dst`: `(n1,)` (or `(n1, B)` for `u` of shape `(n0, B, 2)`).

### Constants

- `AIRFRANS_TASKS`: dict with keys {'T8_n500', 'T8_n2000', 'T8_n8000', 'T8fr_n500', 'T8fr_n2000', 'T8fr_n8000', 'T8xy_n500', 'T8xy_n2000', 'T8xy_n8000', 'T8frxy_n500', 'T8frxy_n2000', 'T8frxy_n8000', ...}
- `AIRFRANS_DEFAULTS`: dict with keys {'T8res', 'T8f', 'AF'}
- `AFP_NATIVE`: dict with keys {'AFPi', 'AFPi0', 'AFPiU', 'AFPiR', 'AFPs', 'AFPcp', 'AFPcpxy'}


## Anisotropy task suite: `rhmp.tasks.aniso`

AHP, ASURF, ACURL, ADARCY loaders and structure metrics.

<details><summary>module docstring</summary>

```text
Anisotropy task suite: PDEs whose operator is the Whitney/Galerkin Hodge star of a misaligned SPD material tensor
(generator ``datasets/generators/gen_aniso.py``; equations, discretisations, sizes and protocol in
``docs/ANISO_TASKS.md``).

===============  ===============  ================================================================  ====================
name             data file        inputs by degree (odd columns first, even last)                    target (readout)
===============  ===============  ================================================================  ====================
AHP_r<R>         AHP_r<R>         0: f | 1: log t^T S t (e) | 2: log det S, log ratio S (e)          u nodes
                                                                                                     (node_scalar)
ASURF_r<R>       ASURF_r<R>       as AHP, closed surfaces in 3-D (tangent-plane fibre tensor)       u (node_scalar)
ASURF_heat_r<R>  ASURF_r<R>       as ASURF                                                           heat flow u_T
                                                                                                     (node_scalar)
ACURL_r<R>       ACURL_r<R>       1: j (odd source 1-cochain), log t^T nu t (e) |                   A edges (cochain:1)
                                  2: log n^T nu n (e) | 3: log det nu, log ratio nu (e)
ACURLb_r<R>      ACURL_r<R>       as ACURL                                                           B = d1 A faces
                                                                                                     (curl: d2 B = 0)
ADARCY_r<R>      ADARCY_r<R>      0: f | 1: log t^T K t (e) | 2: log n^T K^-1 n (e) |               J face fluxes
                                  3: F_T = int_T f (odd), log det K, log ratio K (e)                (cochain:2)
ADARCYp_r<R>     ADARCY_r<R>      as ADARCY                                                          p nodes
                                                                                                     (node_scalar)
===============  ===============  ================================================================  ====================

``R`` in {10, 100} (maximal eigenvalue ratio; the ratio fades to 1 only near the zeros of the direction field).
Variable meshes (one complex per sample, block-diagonal batches), sequential 70/15/15 split, statistics from the
training part (``TaskData.from_arrays``: odd columns / odd targets scale-only, even and vertex columns mean/std),
``extra_tests['fine']`` = the first test samples re-solved on meshes with 4x the nodes (same physical instances),
ASURF also ``test_<family>``.  Triangle complexes use ``star`` (default ``cotan``), tetrahedral complexes
``barycentric`` when ``cotan`` is requested.  ``spatial_dim`` is 2 (AHP) or 3.

``meta['metric_ref']`` suggests the trainer's ``--metric-ref`` for the degree whose metric carries the material
(``1:0`` edges for the scalar problems, ``2:0`` faces for the 2-form problems), ``meta['recovery']`` tells
``scripts/metric_recovery.py`` which stored tensor each learned metric should be compared with, and
``meta['representability']`` summarises the generator's per-cell diagnostics.  :func:`structure_metrics` evaluates
the physics residuals of a model's predictions (discrete PDE residual, boundary conditions, conservation, d2 B = 0).
```

</details>

### `aniso_task_defaults`

```python
aniso_task_defaults(name: str) -> dict
```
Training defaults (`C`, `layers`, `batch` = meshes per block-diagonal batch).

### `parse_name`

```python
parse_name(name: str) -> tuple[str, int]
```
`'ACURLb_r100' -> ('ACURLb', 100)` (an optional `_n<N>` subset suffix is ignored here).

### `load_aniso`

```python
load_aniso(
    name: str,
    droot: str | None = None,
    *,
    native: bool = True,
    device='cuda',
    star: str = 'cotan',
    fine: bool = True,
    max_samples: int | None = None,
    keep_on: str = 'auto',
    n_samples: int | None = None,
    fine_max: int | None = None,
    whitney: bool = True,
    **_,
) -> TaskData
```
Load an anisotropy task (see the module table) as a normalised `TaskData`.

<details><summary>docstring</summary>

```text
Args:
    name: e.g. ``'ASURF_r100'``, ``'ACURLb_r10'``; the suffix ``_n<N>`` (registered for the tet tasks:
        ``_n1500``) uses only the first ``N`` samples of the file with the standard sequential 70/15/15 split,
        i.e. the data set that ``gen_aniso.py --n N`` would have produced (the 4x ``fine`` set is unchanged).
        A tet complex with Whitney blocks takes ~45 MB (~14 MB without, for diagonal metrics), so the 3000-sample
        tet sets need ~137 GB of host memory for tensor-metric runs; ``_n1500`` needs ~68 GB.
    droot: repository root or datasets directory (``None``: ``rhmp.tasks.resolve_root``).
    native: must be True (the cochain inputs/targets are the task definition).
    device: device of the statistics (and of the data when it fits ``rhmp.tasks.synthetic.GPU_BUDGET_GB``).
    star: reference star of triangle complexes (tets: ``barycentric`` when ``cotan`` is requested).
    fine: also build the 4x-finer ``fine`` extra test set.
    fine_max: at most this many ``fine`` samples (default: all for triangle tasks, 50 for tet tasks, whose 4x
        complexes take ~170 MB each with Whitney blocks).
    max_samples: smoke tests (first ``max_samples // 3`` samples of each split part; ``fine`` truncated too).
    keep_on: ``'auto' | device | 'cpu'``: where the complexes and tensors live.
    whitney: keep the Whitney/Galerkin blocks (needed by tensor metrics only); ``False`` drops them per complex
        while building, which keeps the host-memory peak low (``rhmp.tasks.load_task(..., whitney=False)``
        forwards the option).
```

</details>

### `structure_metrics`

```python
structure_metrics(
    model,
    task: TaskData,
    *,
    source: dict | None = None,
    idx=None,
    n: int = 32,
    batch_size: int = 4,
    device=None,
    targets: bool = True,
) -> dict
```
Physics residuals of a model's predictions on an anisotropy task (see `task.meta['structure']`).

<details><summary>docstring</summary>

```text
Args:
    model: called as ``model(inputs, K)`` (normalised output; the task's ``readout``).
    source: evaluation set ``{'K', 'inputs', 'target', 'source', 'sample_ids'}`` (default: the task's test split;
        for ``task.extra_tests['fine']`` pass that dict).
    idx: samples of ``source`` (default: the first ``n`` of the test split / of the set).
    targets: also report the residuals of the stored targets (round-off reference).
Returns:
    ``{metric: {'mean', 'max'}}`` for the predictions and ``target_<metric>`` for the stored targets.
```

</details>

### `cell_tensors`

```python
cell_tensors(blob: dict, i: int, key: str, K, top_key: str, inverse: bool = False) -> Tensor
```
Stored per-cell material tensors of sample `i` in the complex's top-cell order: `(n_top, D, D)` float64 (`inverse`: their inverses -- full-rank planar / volume tensors only; surface tensors are rank 2).

### Constants

- `ANISO_TASKS`: dict with keys {'AHP_r10', 'AHP_r100', 'ASURF_r10', 'ASURF_r100', 'ASURF_heat_r10', 'ASURF_heat_r100', 'ACURL_r10', 'ACURL_r100', 'ACURLb_r10', 'ACURLb_r100', 'ADARCY_r10', 'ADARCY_r100', ...}
- `ANISO_DEFAULTS`: dict with keys {'AHP', 'ASURF', 'ASURF_heat', 'ACURL', 'ACURLb', 'ADARCY', 'ADARCYp'}
- `SPECS`: dict with keys {'AHP', 'ASURF', 'ASURF_heat', 'ACURL', 'ACURLb', 'ADARCY', 'ADARCYp'}


## Extension task suite: `rhmp.tasks.suite`

SURF, DYN, QUAL loaders and rollouts.

<details><summary>module docstring</summary>

```text
Extension task suite: SURF (closed surfaces), DYN (advection-diffusion rollouts), QUAL (mesh-quality shift).

Datasets are produced by ``datasets/generators/gen_surf.py``, ``gen_dyn.py`` and ``gen_qual.py`` (details,
equations and sizes in ``docs/TASK_SUITE_DETAILS.md``).  Every loader returns a :class:`rhmp.data.TaskData` built
exactly like the HP/TET loaders of :mod:`rhmp.tasks.synthetic` (normalisation from the training part,
``node_scalar`` readout, complexes on the GPU when they fit ``GPU_BUDGET_GB``):

==================  ==================================================================================================
name                content
==================  ==================================================================================================
SURF                screened Poisson ``(M + eps L) u = M f`` on variable closed surfaces (ellipsoids, SH-perturbed
                    spheres, perturbed tori), one mesh per sample; input f (nodes), target u (nodes), spatial_dim 3.
                    Sequential 70/15/15 split.  Extra test sets (same normalisation): ``geo`` (superquadrics,
                    geometry transfer), ``topo`` (genus-2 double tori, topology transfer), and the per-family
                    in-distribution test subsets ``test_ellipsoid``, ``test_sphere_pert``, ``test_torus``.
SURF_heat           same inputs/meshes, target = heat flow ``exp(T Lap) f`` (T = 0.05, 10 implicit Euler steps).
SURF_geo, SURF_topo test-only views of the extra sets (split = (), (), all) for ``rhmp.train --eval-ckpt``;
SURF_heat_geo, ...  likewise for the heat target.
DYN                 one-step windows ``(u_t, theta) -> u_{t+1}`` of advection-diffusion trajectories on random
                    planar meshes (one mesh per trajectory); inputs u_t (nodes) and the velocity 1-form theta
                    (edges, odd); target u_{t+1} (nodes).  Split by trajectory (70/15/15); windows at every
                    ``T / windows`` steps.  ``task.rollout`` holds the full test trajectories for
                    :func:`rollout_eval`.
DYNfix              the same on one fixed mesh (shared-mesh TaskData, batches of 64 windows).
DYN_delta,          as DYN / DYNfix with the increment u_{t+1} - u_t as target (one-step persistence has R2 ~0.99 on
DYNfix_delta        the state target; rollouts add the predicted increment).
DYNfix_cons         exactly conservative variant on the fixed mesh: target du = u_{t+1} - u_t (scale-only normalisation,
                    loss in u-space) predicted through a fixed conservative output map: default a finite-volume update
                    du = (mean M / M) d0^T (l* . b) from a learned edge flux density b (readout ``cochain:1``), or
                    (``cons_param='div'``) du_i = (mean M / M_i) (d0^T b)_i (readout ``div:1``); in both cases
                    sum_i M_i du_i = 0 exactly for any weights.
DYN_cons            the same on variable meshes, with the model readout ``mdiv:1`` that divides the divergence by the
                    lumped mass inside the model (``rhmp.readout.MassDivReadout``).
DYN_cons_mass,      conservative variant with target M du (the mass change per dual cell) and the plain ``div:1``
DYNfix_cons_mass    readout, kept to evaluate checkpoints trained with it.  Ill-conditioned for rollouts: the
                    node-uniform loss on M du leaves errors ~1/M_i in du (M varies ~900:1 on the DYN meshes), which
                    diverge within a few autoregressive steps.
HP_qual_graded      HP_k100 physics on graded meshes (corner / circle refinement) and the original meshes, test only;
HP_qual_sliver      ... on sliver meshes (Delaunay of anisotropically scaled point clouds); one extra test set per
                    quality level (``base``, ``corner_r16``, ``circle_a30``, ``a8``, ...) -> accuracy-vs-quality curves.
HP_qual_*_ref       target = reference solution from the 4x-finer HP mesh interpolated to the nodes (instead of the
                    P1 solution on the quality-shifted mesh itself).
==================  ==================================================================================================

Test-only tasks use statistics of their own samples; ``python3 scripts/eval_on.py --run RUN --task HP_qual_sliver``
(or ``python3 -m rhmp.train --task HP_qual_sliver --eval-ckpt RUN``) re-expresses inputs/targets in the
normalisation of the training run before evaluating (``train._renormalize``).

Rollouts: ``rollout_eval(model, task, steps)`` feeds the model's (denormalised) prediction back as the next input and
reports R2 / NRMSE per horizon and the drift of the discrete mass ``sum_i M_i u_i`` (lumped mass of the generator).
```

</details>

### `suite_task_defaults`

```python
suite_task_defaults(name: str) -> dict
```
Training defaults (`C`, `layers`, `batch`) of a suite or anisotropy task.

### `load_surf`

```python
load_surf(
    name: str,
    droot: str | None = None,
    *,
    target: str = 'u',
    test_set: str | None = None,
    native: bool = True,
    device='cuda',
    star: str = 'cotan',
    fine: bool = True,
    extra: bool = True,
    max_samples: int | None = None,
    keep_on: str = 'auto',
    **_,
) -> TaskData
```
SURF / SURF_heat (`test_set=None`) or the test-only views `SURF[_heat]_geo` / `_topo`.

<details><summary>docstring</summary>

```text
Args:
    target: ``'u'`` (screened Poisson) or ``'heat'``.
    test_set: ``None`` (train/val/test on ``SURF.pt`` + extra test sets) or ``'geo'`` / ``'topo'``.
    fine / extra: ``False`` skips the extra test sets (``fine=False`` is what ``rhmp.train --no-fine`` passes).
    max_samples: smoke tests (first ``max_samples // 3`` samples of each split part; extra sets truncated to
        ``max_samples``).
```

</details>

### `load_dyn`

```python
load_dyn(
    name: str = 'DYN',
    droot: str | None = None,
    *,
    native: bool = True,
    device='cuda',
    star: str = 'cotan',
    windows=25,
    target: str = 'state',
    max_samples: int | None = None,
    keep_on: str = 'auto',
    **_,
) -> TaskData
```
DYN: one-step windows on random meshes (variable-mesh TaskData) + `task.rollout` (test trajectories).

<details><summary>docstring</summary>

```text
Args:
    windows: windows per trajectory (every ``T // windows`` steps; ``'all'`` = all ``T``).
    target: ``'state'`` (u_{t+1}, task ``DYN``), ``'delta'`` (u_{t+1} - u_t, task ``DYN_delta``; the rollout
        adds the predicted increment), ``'cons'`` (task ``DYN_cons``: du with scale-only normalisation and the
        mass-weighted divergence readout ``mdiv:1``, exactly conservative) or ``'cons_mass'`` (task
        ``DYN_cons_mass``: the mass change per dual cell ``M du`` with the plain ``div:1`` readout; an
        ill-conditioned conservative convention, kept to evaluate checkpoints trained with it).
    max_samples: smoke tests: about ``max_samples`` windows in total (whole trajectories, 1/3 per split part).
Inputs (native): ``{0: u_t (n0, 1), 1: theta (n1, 1)}`` (theta odd); legacy: ``{0: [u_t, v_x, v_y] (n0, 3)}``.
```

</details>

### `load_dynfix`

```python
load_dynfix(
    name: str = 'DYNfix',
    droot: str | None = None,
    *,
    native: bool = True,
    device='cuda',
    star: str = 'cotan',
    windows=25,
    target: str = 'state',
    max_samples: int | None = None,
    cons_param: str = 'flux',
    **_,
) -> TaskData
```
DYNfix: one-step windows on one fixed mesh (shared-mesh TaskData: `inputs[k]` is `(N, n_k, F_k)`).

<details><summary>docstring</summary>

```text
``target='delta'`` = task ``DYNfix_delta`` (increments); ``'cons'`` = ``DYNfix_cons``: du (scale-only
normalisation, loss in u-space) predicted through a fixed exactly-conservative output map, ``cons_param='flux'``
(default; :func:`flux_density_map`, readout ``cochain:1``: a learned edge flux density integrated over the dual
edges and divided by the dual area) or ``'div'`` (:func:`mass_inverse_map`, readout ``div:1``); ``'cons_mass'`` =
``DYNfix_cons_mass`` (see :func:`load_dyn`).
```

</details>

### `load_qual`

```python
load_qual(
    name: str,
    droot: str | None = None,
    *,
    kind: str = 'graded',
    target: str = 'u',
    native: bool = True,
    device='cuda',
    star: str = 'cotan',
    max_samples: int | None = None,
    keep_on: str = 'auto',
    **_,
) -> TaskData
```
HP_qual_graded / HP_qual_sliver (test only): HP_k100 physics on quality-shifted meshes.

<details><summary>docstring</summary>

```text
Built with the HP loader's own per-sample code (``rhmp.tasks.synthetic._build_set``: same inputs f / log sigma
on edges / log sigma on faces, same alignment), so an HP model can be evaluated unchanged.  ``split[2]`` = all
samples; ``extra_tests`` = one subset per quality level (``meta['levels']``).

Args:
    kind: ``'graded'`` or ``'sliver'``.
    target: ``'u'`` (P1 solution on the shifted mesh; the HP definition) or ``'ref'`` (4x-finer reference
        solution interpolated to the nodes).
    max_samples: keep the first ``max(1, max_samples // n_levels)`` base samples of every level.
```

</details>

### `rollout_eval`

```python
rollout_eval(
    model,
    task: TaskData,
    steps: int | None = None,
    *,
    device=None,
    batch_size: int | None = None,
    horizons=(1, 5, 10, 25, 50, 100),
    return_pred: bool = False,
    project_mass: bool = False,
) -> dict
```
Autoregressive rollouts of a one-step DYN/DYNfix model on the full test trajectories (`task.rollout`).

<details><summary>docstring</summary>

```text
Starting from the true ``u_0`` the model's denormalised prediction is fed back as the next input for ``steps``
steps (default: the stored trajectory length).  Metrics per horizon ``t = 1..steps``:

* ``R2``: v1 formula in the task's normalised target space (shared mesh: per-node mean over trajectories as for
  stacked tensors; variable meshes: pooled mean), ``NRMSE``: RMSE / range of the true fields at ``t``;
* ``mass_drift_mean`` / ``_max`` over trajectories of ``|sum M u_pred(t) - sum M u_0| / sum M |u_0|`` with the
  generator's lumped mass ``M`` (the true dynamics conserve it to round-off: ``true_mass_drift_max``);
* ``persistence_R2``: the trivial prediction ``u(t) = u_0`` for reference.

Args:
    model: a module called as ``model(inputs, K) -> (n0, B, 1)`` (normalised output).
    task: DYN or DYNfix :class:`TaskData` (native or legacy) with the ``rollout`` attribute.
    batch_size: trajectories per forward (default 16 on variable meshes, 64 on the shared mesh).
    return_pred: also return the predicted trajectories (list of ``(steps+1, n0)`` CPU tensors).
    project_mass: after every step shift each trajectory by a constant so that ``sum M u`` keeps its initial
        value (post-hoc exact conservation for any model, e.g. DYN / DYN_delta; no effect on the ``_cons``
        models, which conserve by construction).
Returns:
    dict with lists over horizons and ``summary`` = ``{'R2@h', 'NRMSE@h', 'mass_drift@h'}`` for ``h`` in
    ``horizons``, ``first_nonfinite_step`` (``None`` if the rollout stayed finite).
```

</details>

### `surf_integral_error`

```python
surf_integral_error(
    model,
    task: TaskData,
    *,
    source: dict | None = None,
    idx=None,
    batch_size: int = 8,
    device=None,
) -> dict
```
Structure metric of SURF / SURF_heat: relative error of the discrete integral `sum_i M_i u_i`.

<details><summary>docstring</summary>

```text
Both targets conserve it exactly (``1^T L = 0``: ``sum M u = sum M f`` for the screened Poisson solve and for every
implicit heat step), so ``e_i = |sum M u_pred - sum M f| / sum M |f|`` measures how far a prediction is from the
constraint (``M`` = lumped mass = ``K.star[0]``, float32).  The same quantity for the stored targets is returned as
a sanity reference (float32 round-off, ~1e-7).

Args:
    source: evaluation set ``{'K', 'inputs', 'target'}`` (default: the task itself, split ``idx`` = test split).
Returns:
    ``{'mean', 'max', 'median', 'target_max', 'N'}``.
```

</details>

### `mass_weighted_errors`

```python
mass_weighted_errors(task: TaskData, pred, *, source: dict | None = None, idx=None) -> dict
```
Mesh-density-independent errors of node predictions: `||u_pred - u||_M / ||u||_M` per sample, with the lumped mass `M = K.star[0]` (barycentric dual areas), in physical units.

<details><summary>docstring</summary>

```text
Node-pooled R2 / NRMSE weight every node equally, so on graded meshes (nodes clustered near a Dirichlet corner
where u ~ 0) they change with the mesh density even for a perfect solver; the mass-weighted error approximates
the continuous relative L2 error and is comparable across quality levels.

Args:
    pred: normalised predictions for the samples ``idx`` of ``source`` (as returned by ``rhmp.train.predict``:
        list of ``(n_i, 1)`` or a stacked ``(N, n, 1)`` tensor).
    source: evaluation set ``{'K', 'target'}`` (default: the task itself; ``idx`` default = test split).
Returns:
    ``{'relL2M_mean', 'relL2M_median', 'relL2M_pooled'}`` (pooled = sqrt(sum_i ||e_i||_M^2 / sum_i ||u_i||_M^2)).
```

</details>

### `mass_inverse_map`

```python
mass_inverse_map(mass: Tensor, device) -> OutputMap
```
Fixed output map `du_i = (mean M / M_i) z_i` on one mesh (model readout `div:1`, `z = d0^T b`).

<details><summary>docstring</summary>

```text
Makes a zero-sum divergence output an exactly conservative density increment (``sum_i M_i du_i = mean M *
sum_i z_i = 0``) while the loss is taken on ``du`` itself (node-uniform in u-space).
```

</details>

### `mass_div_readout_available`

```python
mass_div_readout_available() -> bool
```
True if the model provides the mass-weighted divergence readout `mdiv:1` (`y = star0^{-1} d0^T b`).

### Constants

- `SUITE_TASKS`: dict with keys {'SURF', 'SURF_heat', 'SURF_geo', 'SURF_topo', 'SURF_heat_geo', 'SURF_heat_topo', 'DYN', 'DYNfix', 'DYN_delta', 'DYNfix_delta', 'DYN_cons', 'DYNfix_cons', ...}
- `SUITE_DEFAULTS`: dict with keys {'SURF', 'DYN', 'DYNfix', 'HP_qual'}


## Robustness: `rhmp.robustness`

Symmetry and robustness transforms of a task.

<details><summary>module docstring</summary>

```text
Robustness transforms and the robustness table (exact symmetries, gauge noise, input noise, batch composition).

Library API (used by ``scripts/eval_robustness.py`` and ``tests/test_train.py``)::

    from rhmp.robustness import robustness_table, transformed_task, gauge_task, noise_task, random_rotation
    rows = robustness_table(model, task, te, relabel=True, flip_orient=True, rotate=True, gauge_noise=[0.1, 1],
                            noise=[0.01], batch_sizes=[1, 64], eval_batch=64, device="cuda")

Transforms (each returns a test-only :class:`~rhmp.data.TaskData` view with the task's statistics and readout):

* vertex relabelling / face-orientation flips / E(n) transforms -> :func:`transformed_task`: the complexes are rebuilt
  from the transformed mesh and the data of every degree are remapped (odd cochains with the orientation sign of the
  new canonical edges/faces, node vectors rotated, orientation output maps rebuilt from the new geometry);
* gauge noise ``theta -> theta + d0 lambda`` on connection inputs -> :func:`gauge_task`;
* Gaussian input noise on the normalised inputs -> :func:`noise_task`.

An exactly equivariant/invariant model gives the same metrics on every transformed view (up to the float32 rounding
of the rebuilt geometry, ~1e-7); ``tests/test_train.py::test_robustness_transforms_are_exact`` checks this.
```

</details>

### `remap_complex`

```python
remap_complex(K, *, perm=None, flip_frac=0.0, Q=None, shift=None, seed=0)
```
Transformed complex and per-degree data maps.

<details><summary>docstring</summary>

```text
Args:
    K: source complex (single mesh).
    perm: ``(n0,)`` new vertex i = old vertex ``perm[i]``.
    flip_frac: fraction of top cells whose orientation is reversed.
    Q, shift: E(n) transform ``x -> x Q^T + shift`` of the positions.
Returns:
    ``(K_new, maps)``, ``maps[k] = (idx, sign)``: ``new_k = old_k[idx] * sign`` (the sign only for odd data).
```

</details>

### `transformed_task`

```python
transformed_task(
    task: TaskData,
    idx: torch.Tensor,
    *,
    perm_seed=None,
    flip_frac=0.0,
    Q=None,
    shift=None,
    seed=0,
) -> TaskData
```
Test-split view of `task` under a mesh transform (one transform per mesh for variable-mesh tasks).

### `gauge_task`

```python
gauge_task(task: TaskData, idx: torch.Tensor, s: float, seed: int = 0) -> TaskData
```
Test-split view with gauge noise `theta -> theta + d0 lambda` on the connection columns (raw units), `lambda ~ N(0, s^2)` per node and sample.

### `noise_task`

```python
noise_task(task: TaskData, idx: torch.Tensor, s: float, seed: int = 0) -> TaskData
```
Test-split view with Gaussian noise of standard deviation `s` on all normalised inputs.

### `random_rotation`

```python
random_rotation(D: int, seed: int, proper: bool = True) -> np.ndarray
```
Random orthogonal `(D, D)` matrix with det +1 (`proper`) or -1.

### `robustness_table`

```python
robustness_table(
    model,
    task: TaskData,
    te: torch.Tensor,
    *,
    eval_batch: int,
    device,
    relabel: bool = False,
    flip_orient: bool = False,
    rotate: bool = False,
    reflect: bool = False,
    gauge_noise=(),
    noise=(),
    batch_sizes=(),
    extra: bool = True,
    seed: int = 0,
    log=<built-in function print>,
) -> list[dict]
```
Rows of the robustness table for `model` on the test samples `te` of `task`.

<details><summary>docstring</summary>

```text
Args:
    model: trained ``RHMP`` in eval mode (``record_diagnostics = False`` recommended).
    task: normalised task (statistics of the training run, see ``rhmp.train._renormalize``).
    te: test sample indices.
    eval_batch: evaluation batch size of the reference row ``base``.
    relabel, flip_orient, rotate, reflect: exact-symmetry rows (``reflect`` is skipped for tasks with an
        orientation output map, whose targets are pseudo-quantities, and for legacy frame-dependent inputs).
    gauge_noise: lambda stds of the gauge rows (tasks with connection inputs, shared meshes).
    noise: input-noise stds.
    batch_sizes: evaluation batch sizes (batch independence, with ``max_dpred`` vs the reference).
    extra: also evaluate the task's extra test sets (``fine``, quality splits...).
Returns:
    list of dicts ``{condition, R2, MSE, NRMSE, SSIM, Pearson, N, [max_dpred], [seconds], dR2}``.
```

</details>

### Constants

- `SYMMETRY_CONDITIONS`: ('relabel', 'flip-orient', 'rotate', 'reflect')


## Baselines: `rhmp.baselines`

Lazy entry points.

<details><summary>module docstring</summary>

```text
Baselines for the RHMP v2 task suite (same data, targets, loss, metrics and trainer as ``rhmp.RHMP``).

    from rhmp.baselines import build_model, applicable, MODELS
    ok, why = applicable("mgn", task)                       # task: rhmp.data.TaskData
    model, info = build_model("mgn", task, target_params)   # param-matched to the v2 model (v1 rule)
    y = model(inputs, K)                                    # {k: (n_k, B, F_k)}, complex -> (n_out, B, out_dim)

Modules:
    adapters      v1-compatible complex adapter (+ cached operators), input encoders, output heads
    v1_wrappers   uniform wrapper ``BaselineModel`` around the v1 baselines and the v1 model (``ours_v1``)
    mgn           MeshGraphNet (Pfaff et al., ICLR 2021), vectorised
    dec_fixed     fixed-metric RHMP controls (``dec_fixed``, ``unit_star``, ``unit_fixed``)
    param_match   v1 parameter-matching rule with a JSON cache
    registry      ``MODELS`` (name -> builder), ``applicable``, ``build_model``, benchmark CLI

Imports are lazy so that importing :mod:`rhmp.baselines` never pulls in the v1 tree.
```

</details>

- `rhmp.baselines.applicable`: re-export of `rhmp.baselines.registry.applicable` (documented there).
- `rhmp.baselines.build_model`: re-export of `rhmp.baselines.registry.build_model` (documented there).
- `rhmp.baselines.model_names`: re-export of `rhmp.baselines.registry.model_names` (documented there).
### Constants

- `MODELS`: dict with keys {'rhmp', 'dec_fixed', 'unit_star', 'unit_fixed', 'ours_v1', 'mgn', 'mgn_fast', 'gcn', 'gat', 'schnet', 'egnn', 'gauge_cnn', ...}
- `SPECS`: dict with keys {'rhmp', 'dec_fixed', 'unit_star', 'unit_fixed', 'ours_v1', 'mgn', 'mgn_fast', 'gcn', 'gat', 'schnet', 'egnn', 'gauge_cnn', ...}


## Baseline registry: `rhmp.baselines.registry`

Model specs, applicability, parameter matching.

<details><summary>module docstring</summary>

```text
Model registry: ``MODELS = {name: builder(task_data, target_params, **overrides)}``, applicability, budgets.

    from rhmp.baselines.registry import MODELS, SPECS, applicable, build_model, rhmp_param_count
    target = rhmp_param_count(task)                    # trainable parameters of the v2 model on this task
    ok, why = applicable("sccnn", task)
    model, info = build_model("sccnn", task, target)   # width matched by the v1 rule: params in [target, 1.2 target]

Families (see ``docs/BASELINES.md``):

* v2 and its physics-prior controls: ``rhmp`` (default), ``dec_fixed``, ``unit_star``, ``unit_fixed``
  (same architecture, not param-matched; they keep the task's orientation output map);
* v1 model: ``ours_v1`` (paper configuration; shared triangle meshes only);
* MeshGraphNet: ``mgn`` (15 processor steps, standard) and ``mgn_fast`` (8 steps), width param-matched;
* v1 graph baselines ``gcn gat schnet egnn`` and mesh/gauge baselines ``gauge_cnn gem_cnn``;
* v1 cell-complex baselines ``mpsn sccnn cw_net clifford_smpn``;
* operator baselines ``fno`` (regular grids: T1, T1q) and ``deeponet`` (node targets).

All non-v2 models predict the task target directly (the orientation output maps of native T1/T3/T6/T7 exist
because the v2 model is exactly orientation-equivariant and reflection-invariant; the baselines are not, so the
trainer drops ``task.output_map`` for them).  Benchmark CLI::

    python3 -m rhmp.baselines.registry --budgets T6 T7 HP_k100          # matched widths / params per model
    python3 -m rhmp.baselines.registry --bench T6 --models rhmp,mgn,mgn_fast --batch 64 --steps 20
```

</details>

### class `ModelSpec`

```python
ModelSpec(
    name: str,
    family: str,
    description: str,
    param_matched: bool = True,
    uses_output_map: bool = False,
    data_star: str | None = None,
    node_inputs: bool = True,
)
```
Registry metadata.

| field | type | default | description |
|---|---|---|---|
| `name` | `str` | required | registry name. |
| `family` | `str` | required | `'rhmp' \| 'v1_model' \| 'mgn' \| 'v1_graph' \| 'v1_mesh' \| 'v1_complex' \| 'operator'`. |
| `description` | `str` | required | one line. |
| `param_matched` | `bool` | `True` | width chosen by `rhmp.baselines.param_match.match_params`. |
| `uses_output_map` | `bool` | `False` | keeps the task's orientation output map (v2 family only). |
| `data_star` | `str \| None` | `None` | reference star the task must be loaded with (`None`: task default). |
| `node_inputs` | `bool` | `True` | the model reads node features (`NodeInputEncoder`); such models default to the legacy inputs on `LEGACY_TASKS`. |

<details><summary>docstring</summary>

```text
Attributes:
    name: registry name.
    family: ``'rhmp' | 'v1_model' | 'mgn' | 'v1_graph' | 'v1_mesh' | 'v1_complex' | 'operator'``.
    description: one line.
    param_matched: width chosen by :func:`rhmp.baselines.param_match.match_params`.
    uses_output_map: keeps the task's orientation output map (v2 family only).
    data_star: reference star the task must be loaded with (``None``: task default).
    node_inputs: the model reads node features (``NodeInputEncoder``); such models default to the legacy
        inputs on :data:`LEGACY_TASKS`.
```

</details>

### `applicable`

```python
applicable(name: str, td: Any) -> tuple[bool, str]
```
Whether model `name` can be trained on task `td` (`(True, '')` or `(False, reason)`).

### `build_model`

```python
build_model(
    name: str,
    td: Any,
    target_params: int | None = None,
    **overrides,
) -> tuple[nn.Module, dict]
```
Build model `name` for task `td`.

<details><summary>docstring</summary>

```text
Args:
    name: registry name.
    td: :class:`rhmp.data.TaskData` (complexes on the target device).
    target_params: parameter budget for param-matched models (default: the v2 model's count on ``td``).
    **overrides: builder options (``hidden``, ``n_layers``, ``amp``, ``args`` for the v2 family, ...).
Returns:
    ``(model, info)`` (model on the device of the task's complexes).
Raises:
    ValueError: if the model is not applicable to the task.
```

</details>

### `model_names`

```python
model_names() -> list[str]
```
All registered names.

### `rhmp_param_count`

```python
rhmp_param_count(td: Any, args: Any | None = None) -> int
```
Trainable parameters of the v2 model that `rhmp.train` builds for `td` (the default budget; for a task transformed by `prepare_task`, the v2 model of the original task).

### `default_mode`

```python
default_mode(name: str, task_name: str) -> str
```
`'legacy'` for node-input baselines on paper tasks with distinct legacy inputs, else `'native'`.

### `from_checkpoint`

```python
from_checkpoint(ck: dict, td: Any, map_location: Any = None) -> nn.Module
```
Rebuild a model saved with `to_checkpoint()` (baselines: `{'model', 'build', 'state_dict'}`).

### `prepare_task`

```python
prepare_task(name: str, td: Any, opts: dict | None = None) -> Any
```
Task as model `name` must be trained and scored on (identity except for the v1 edge protocol).

<details><summary>docstring</summary>

```text
v1 edge protocol (MPSN / SCCNN / Clifford-SMPN, ``head`` resolving to ``'v1_edge'``): the node target ``y``
becomes the edge-averaged target ``0.5 (y_src + y_dst)`` on the canonical edges (in normalised units, which equals
v1's raw-space average normalised with the node statistics), the target cells become the edges (``even:1``), and
- with ``v1_components='first'`` (default, as v1) - only the first target component is kept.  Metrics are then
edge-space metrics, exactly as the v1 evaluation script ``compute_all_metrics.py`` scored these models.

Args:
    name: registry name.
    td: :class:`rhmp.data.TaskData`.
    opts: builder options (``head``, ``v1_components``).
Returns:
    ``td`` itself, or a transformed copy (``meta['v1_edge_protocol']`` describes it, ``meta['untransformed']``
    holds the original task).
```

</details>

### `resolve_head`

```python
resolve_head(name: str, td: Any, head: str = 'auto') -> str | None
```
Head mode of MPSN / SCCNN / Clifford-SMPN on `td` (`None` for other models).

<details><summary>docstring</summary>

```text
``'auto'``: ``'v1_edge'`` on the shared-mesh paper tasks of :data:`V1_EDGE_TASKS` with node targets (v1's
protocol), ``'node'`` elsewhere.  A task already transformed by :func:`prepare_task` resolves to ``'v1_edge'``.
```

</details>

### `main`

```python
main(argv: list[str] | None = None) -> None
```
CLI entry point (`python3 -m rhmp.baselines.registry`; see the module docstring).

### Constants

- `MODELS`: dict with keys {'rhmp', 'dec_fixed', 'unit_star', 'unit_fixed', 'ours_v1', 'mgn', 'mgn_fast', 'gcn', 'gat', 'schnet', 'egnn', 'gauge_cnn', ...}
- `SPECS`: dict with keys {'rhmp', 'dec_fixed', 'unit_star', 'unit_fixed', 'ours_v1', 'mgn', 'mgn_fast', 'gcn', 'gat', 'schnet', 'egnn', 'gauge_cnn', ...}
- `LEGACY_SCALAR_COLUMNS`: dict with keys {'T6', 'T6_100K', 'T7'}
- `LEGACY_TASKS`: ('T1', 'T1p', 'T1pm', 'T1pml', 'T1pf', 'T1pmf', 'T1pmlf', 'T1g', 'T1gml', 'T2v', 'T2a', 'T3', 'T5', 'T6', 'T7', 'T6_1...
- `V1_EDGE_MODELS`: ('mpsn', 'sccnn', 'clifford_smpn')
- `V1_EDGE_TASKS`: ('T1', 'T1p', 'T1pm', 'T1pml', 'T1pf', 'T1pmf', 'T1pmlf', 'T1g', 'T1gml', 'T2', 'T2v', 'T2a', 'T3', 'T5', 'T6', 'T7')


## Building blocks: layers: `rhmp.layers`

Normalised metric Hodge operators, resolvent / solve / projection layers, RHMPLayer.

<details><summary>module docstring</summary>

```text
Normalised metric Hodge operators and the RHMP message-passing layer (DESIGN §3.3).

Notation (per degree ``k``, all feature tensors ``(n_k, B, C)``, per-cell/per-sample scalars ``(n, B)``):

* Up block of degree ``k`` (exists if ``k < K``): ``A = d_k`` ``(n_{k+1}, n_k)``, metric ``H = H_{k+1}``.
* Down block of degree ``k`` (exists if ``k > 0``): ``A = d_{k-1}^T`` ``(n_{k-1}, n_k)``, metric
  ``H = H^down_{k-1}`` (``= 1 / H_{k-1}`` when metrics are tied).
* Both blocks share one form.  With the diagonal scaling ``s = S^{-1/2}`` on degree ``k``::

      L x      = s ⊙ A^T ( h ⊙ A ( s ⊙ x ) ),        h = H / beta          (normalised operator)
      T y      = s ⊙ A^T ( sqrt(h) ⊙ y )                                   (half operator, L = T T^T)
      beta     = max_j s_j [ |A|^T ( H ⊙ |A| s ) ]_j   per sample (Gershgorin bound of s A^T H A s)

  hence ``||L|| <= 1`` and ``||T|| <= 1`` for every sample and every metric.  The cross-degree transport is
  ``cross_up(x_{k+1}) = T_up x_{k+1}`` and ``cross_down(x_{k-1}) = T_down x_{k-1}``.
* Scalings (``scaling=``): ``'dec'`` uses the reference star in the symmetrised frame ``x = star^{1/2} u``,
  i.e. ``S_up = star_k`` and ``S_down = 1 / star_k``; at initialisation (``H = star``) the blocks are the
  DEC Hodge Laplacian pieces ``star^{-1/2} d^T star d star^{-1/2}`` and ``star^{1/2} d star^{-1} d^T star^{1/2}``
  and the cross terms are the DEC coboundary / codifferential in that frame.  ``'jacobi'`` uses
  ``S = diag(A^T H A)`` per sample; ``'none'`` uses ``S = 1``.  All operators are invariant under a per-sample
  rescaling of ``S`` and of ``H``, so the stars are normalised per sample by their geometric mean (numerical
  hygiene only).

Layer update (all degrees synchronously from the layer input)::

    m_k = sum_p c^up_p L_up^p x_k + sum_p c^dn_p L_dn^p x_k + w_cu T_up x_{k+1} + w_cd T_dn x_{k-1}
          (+ the transport term sum_p g_p T^p x_0 on the vertices when configured, ``rhmp.transport``)
    x_k <- x_k + gamma_k * sigmoid(MLP(log1p(rms(m_k)))) * m_k / rms(m_k)      (radial norm gate)

Implementation notes (performance):

* For ``'dec'`` / ``'none'`` the constant scaling is folded into scaled copies of the CSR operators
  (``A diag(s)``, ``diag(s) A^T`` and their absolute values; same index arrays, new values), cached per complex,
  so no elementwise ``s ⊙ x`` pass is ever executed.  ``'jacobi'`` (per-sample ``s``) uses the explicit form.
* The first coboundary ``A (s ⊙ x)`` of every block is computed once per layer and shared by the block and by
  the metric invariants.
* The polynomial filter runs inside a fused autograd function (Horner-like recursion) that only keeps
  ``(A s)(s A^T) ... q`` for ``p >= 2`` for the backward and uses fused ``addcmul``/``vecdot`` passes.
```

</details>

### class `LayerContext`

```python
LayerContext(
    K: Any,
    top: int,
    B: int,
    dtype: torch.dtype,
    batch: list[Tensor] | None,
    num_graphs: int,
    log_star: list[Tensor],
    blocks: dict,
    bcount: list[Tensor | None],
    geo: list[Tensor],
    even: list[Tensor | None],
    scaling: str,
    record: bool = False,
    ref: list | None = None,
    resolvent_prev: dict | None = None,
    log_gm: list | None = None,
    log_vol: Tensor | None = None,
    beta_scale: dict | None = None,
    energy_ref: Tensor | None = None,
    transport_fields: Tensor | None = None,
)
```
Geometry-derived tensors shared by the lifting, all layers and the readout of one forward pass.

| field | type | default | description |
|---|---|---|---|
| `K` | `Any` | required | the `CochainComplex`. |
| `top` | `int` | required | top degree `K.dim`. |
| `B` | `int` | required | number of samples sharing the complex (1 for a block-diagonal batch of meshes). |
| `dtype` | `torch.dtype` | required | feature dtype (float32, or float64 in tests). |
| `batch` | `list[Tensor] \| None` | required | `K.batch` (list of `(n_k,)` int64 graph ids) or `None` for a single complex. |
| `num_graphs` | `int` | required | number of graphs in the complex. |
| `log_star` | `list[Tensor]` | required | `log` of the per-graph geometric-mean normalised reference star, `(n_k,)`. |
| `blocks` | `dict` | required | `{(k, is_up): BlockGeometry}` operators of every block (scaling folded in). |
| `bcount` | `list[Tensor \| None]` | required | number of boundary `(k-1)`-cells of each `k`-cell, `(n_k, 1, 1)` (`None` for k = 0). |
| `geo` | `list[Tensor]` | required | `K.geo[k]` in `dtype`, `(n_k, G_k)`. |
| `even` | `list[Tensor \| None]` | required | even raw inputs per degree `(n_k, B, E_k)` or `None`. |
| `scaling` | `str` | required | `'dec' \| 'jacobi' \| 'none'`. |
| `record` | `bool` | `False` | whether layers record diagnostics. |
| `ref` | `list \| None` | `None` | optional fixed log-metric references `(n_k, B)` per degree (`cfg.metric_reference`) or `None`. |
| `resolvent_prev` | `dict \| None` | `None` | `{k: (n_k, B, C)}` last resolvent-layer solution per degree (warm starts), or `None`. |
| `log_gm` | `list \| None` | `None` | `log` of the geometric mean of `star_k` per graph, `(1\|G,)` (restores physical units). |
| `log_vol` | `Tensor \| None` | `None` | `log` of the total dual measure `sum star_0` per graph, `(1\|G,)` (domain scale `L^D`). |
| `beta_scale` | `dict \| None` | `None` | `{(k, up): (1\|G, 1)}` factors `beta_unit(train) / beta_unit(mesh)` applied to the block bounds (`frozen_units`), else `None`. |
| `energy_ref` | `Tensor \| None` | `None` |  |
| `transport_fields` | `Tensor \| None` | `None` | input / latent transport fields of the transport term `(n_1, B, J)` or `None` (`rhmp.transport`). |

<details><summary>docstring</summary>

```text
Attributes:
    K: the ``CochainComplex``.
    top: top degree ``K.dim``.
    B: number of samples sharing the complex (1 for a block-diagonal batch of meshes).
    dtype: feature dtype (float32, or float64 in tests).
    batch: ``K.batch`` (list of ``(n_k,)`` int64 graph ids) or ``None`` for a single complex.
    num_graphs: number of graphs in the complex.
    log_star: ``log`` of the per-graph geometric-mean normalised reference star, ``(n_k,)``.
    blocks: ``{(k, is_up): BlockGeometry}`` operators of every block (scaling folded in).
    bcount: number of boundary ``(k-1)``-cells of each ``k``-cell, ``(n_k, 1, 1)`` (``None`` for k = 0).
    geo: ``K.geo[k]`` in ``dtype``, ``(n_k, G_k)``.
    even: even raw inputs per degree ``(n_k, B, E_k)`` or ``None``.
    scaling: ``'dec' | 'jacobi' | 'none'``.
    record: whether layers record diagnostics.
    ref: optional fixed log-metric references ``(n_k, B)`` per degree (``cfg.metric_reference``) or ``None``.
    resolvent_prev: ``{k: (n_k, B, C)}`` last resolvent-layer solution per degree (warm starts), or ``None``.
    log_gm: ``log`` of the geometric mean of ``star_k`` per graph, ``(1|G,)`` (restores physical units).
    log_vol: ``log`` of the total dual measure ``sum star_0`` per graph, ``(1|G,)`` (domain scale ``L^D``).
    beta_scale: ``{(k, up): (1|G, 1)}`` factors ``beta_unit(train) / beta_unit(mesh)`` applied to the block bounds
        (``frozen_units``), else ``None``.
    transport_fields: input / latent transport fields of the transport term ``(n_1, B, J)`` or ``None``
        (``rhmp.transport``).
```

</details>

#### `LayerContext.cell_mean`

```python
cell_mean(v: Tensor, k: int) -> Tensor
```
Per-sample mean over `k`-cells, broadcast back to cells.

<details><summary>docstring</summary>

```text
Args:
    v: ``(n_k, B)``.
    k: degree.

Returns:
    ``(1, B)`` (single complex) or ``(n_k, B)`` (block-diagonal batch).
```

</details>

#### `LayerContext.cell_max`

```python
cell_max(v: Tensor, k: int) -> Tensor
```
Per-sample maximum over `k`-cells.

<details><summary>docstring</summary>

```text
Args:
    v: ``(n_k, B)``.
    k: degree.

Returns:
    ``(1, B)`` (single complex) or ``(num_graphs, B)`` (block-diagonal batch).
```

</details>

#### `LayerContext.to_cells`

```python
to_cells(stat: Tensor, k: int) -> Tensor
```
Broadcast a per-sample statistic `(1|num_graphs, B)` to `k`-cells: `(1, B)` or `(n_k, B)`.

### class `BlockGeometry`

```python
BlockGeometry(
    A: Tensor,
    AT: Tensor,
    A_abs: Tensor,
    AT_abs: Tensor,
    r: Tensor | None,
    scale: Tensor | None,
    nb: int,
)
```
Operators of one block with the constant scaling folded in (cached per complex).

| field | type | default | description |
|---|---|---|---|
| `A` | `Tensor` | required | CSR `(m, n_k)`: `A diag(s)` for `'dec'`, the raw coboundary (or its transpose) otherwise. |
| `AT` | `Tensor` | required | CSR `(n_k, m)`: transpose of `A`. |
| `A_abs` | `Tensor` | required | CSR `\|A\|` (scaled like `A`). |
| `AT_abs` | `Tensor` | required | CSR `\|A\|^T`. |
| `r` | `Tensor \| None` | required | `\|A\| 1` `(m, 1, 1)` (Gershgorin helper; `= \|A_raw\| s`), `None` for Jacobi scaling. |
| `scale` | `Tensor \| None` | required | the folded scaling `s` `(n_k, 1, 1)` (`'dec'`) or `None`. |
| `nb` | `int` | required | degree of the neighbour cells (`k+1` for up blocks, `k-1` for down blocks). |

<details><summary>docstring</summary>

```text
Attributes:
    A: CSR ``(m, n_k)``: ``A diag(s)`` for ``'dec'``, the raw coboundary (or its transpose) otherwise.
    AT: CSR ``(n_k, m)``: transpose of ``A``.
    A_abs: CSR ``|A|`` (scaled like ``A``).
    AT_abs: CSR ``|A|^T``.
    r: ``|A| 1`` ``(m, 1, 1)`` (Gershgorin helper; ``= |A_raw| s``), ``None`` for Jacobi scaling.
    scale: the folded scaling ``s`` ``(n_k, 1, 1)`` (``'dec'``) or ``None``.
    nb: degree of the neighbour cells (``k+1`` for up blocks, ``k-1`` for down blocks).
```

</details>

### `make_context`

```python
make_context(
    K: Any,
    *,
    scaling: str,
    B: int,
    dtype: torch.dtype,
    even: list[Tensor | None] | None = None,
    record: bool = False,
    frozen: dict | None = None,
) -> LayerContext
```
Build the per-forward `LayerContext` (geometry only, no gradients; operators cached per complex).

<details><summary>docstring</summary>

```text
Args:
    K: ``CochainComplex`` (its float tensors must have dtype ``dtype``).
    scaling: ``'dec' | 'jacobi' | 'none'``.
    B: batch size of the features.
    dtype: feature dtype.
    even: even raw inputs per degree (``(n_k, B, E_k)`` or ``None``), length ``K.dim + 1``.
    record: record diagnostics in the layers.
    frozen: training-mesh normalisers (``frozen_units``; :func:`unit_reference` plus a hashable ``'key'``) or
        ``None`` (this complex's own normalisers, the default).

Returns:
    The context.
```

</details>

### class `RadialGate`

```python
RadialGate(mode: str = 'norm', hidden: int = 16) -> None
```
Bases: `Module`.

Radial norm gate with scalar-gain RMS normalisation and the residual update (O(C)-equivariant).

<details><summary>docstring</summary>

```text
``gate='norm'``: ``x + gamma * sigmoid(MLP(log1p(rms(m)))) * m / rms(m)`` with ``rms`` over channels per cell
(``MLP: 1 -> hidden -> SiLU -> 1``).  The gate is computed from the rms *before* normalisation, so the update
magnitude is a learned bounded function of the message magnitude and its direction is the message direction.

``gate='relu'`` (ablation, breaks O(C) and orientation symmetry): ``x + gamma * relu(m) / rms(relu(m))``.

``gate='none'`` (linear models, e.g. the solver preset): ``x + gamma * m`` (no nonlinearity, no normalisation).

``gate='homog'`` (homogeneous of degree one in the field amplitude): ``x + gamma * 2 sigmoid(MLP(xi)) * m`` with
the scale-invariant per-cell invariants ``xi = [log((|m|^2/C + eps) / (|x|^2/C + eps)), <x, m> / (C sqrt((|x|^2/C
+ eps)(|m|^2/C + eps)))]``, ``eps = HOMOG_EPS (|x|^2 + |m|^2) / C`` (relative size and alignment of the message;
``MLP: 2 -> hidden -> SiLU -> 1``, last layer zero-initialised, so the update is ``x + gamma m`` at
initialisation).  Scaling ``(x, m) -> (lambda x, lambda m)`` leaves the factor unchanged, so a stack of such layers
keeps the amplitude of its input instead of bounding every update by ``gamma``; with a constant factor it is a
linear filter.

Args:
    mode: ``'norm' | 'relu' | 'none' | 'homog'``.
    hidden: hidden width of the gate MLP.
```

</details>

#### `RadialGate.homog_scale`

```python
homog_scale(x: Tensor, m: Tensor) -> Tensor
```
Per-cell factor `gamma * 2 sigmoid(MLP(xi))` `(n, B, 1)` of the homogeneous gate (scale invariant).

#### `RadialGate.scale`

```python
scale(m: Tensor) -> Tensor
```
Per-cell factor `a` `(n, B, 1)` such that the update is `m ⊙ a`.

#### `RadialGate.forward`

```python
forward(x: Tensor, m: Tensor) -> Tensor
```
Residual update `x + m ⊙ a(m)`: `(n, B, C), (n, B, C) -> (n, B, C)`.

### class `RHMPLayer`

```python
RHMPLayer(
    top: int,
    geo_dims: dict[int, int],
    even_dims: dict[int, int],
    *,
    C: int,
    poly_order: int = 2,
    metric_hidden: int = 32,
    log_range: float = 2.0,
    tie_metrics: bool = True,
    scaling: str = 'dec',
    cross: bool = True,
    gate: str = 'norm',
    identity_metric: bool = False,
    fused: bool = True,
    kind: str = 'poly',
    resolvent_iters: int = 20,
    resolvent_grad: str = 'implicit',
    metric_type: str = 'diag',
    resolvent_warm_start: bool = True,
    learn_metric: bool = True,
    resolvent_normalize: bool = True,
    solve_iters: int = 64,
    solve_tol: float = 1e-06,
    solve_bc: str = 'dirichlet',
    tensor_param: str = 'full',
    solve_precond: str = 'none',
    term_gates: bool = False,
    project_harmonic: bool = True,
    kutta_range: float = 8.0,
    project_solver: str = 'cg',
    kutta_corner: float = 0.0,
    transport: dict | None = None,
    solve_picard: int = 0,
    picard_relax: float = 0.5,
    metric_geo: bool = True,
    picard_init: str = 'zero',
) -> None
```
Bases: `Module`.

One RHMP message-passing layer on a complex of top degree `top`.

<details><summary>docstring</summary>

```text
Args:
    top: top degree ``K``.
    geo_dims: ``{k: G_k}``.
    even_dims: ``{k: E_k}`` even raw input columns per degree (fed to the metric heads).
    C: channels.
    poly_order: ``P >= 1``.
    metric_hidden: hidden width of the metric MLPs.
    log_range: bound ``a`` of the log-metric.
    tie_metrics: one metric per degree (up uses ``H_m``, down uses ``1/H_m``) or separate up/down heads.
    scaling: ``'dec' | 'jacobi' | 'none'``.
    cross: cross-degree transport terms.
    gate: ``'norm' | 'relu' | 'none'``.
    identity_metric: every metric ``H = 1`` (ablation of the learned metric; with ``scaling='dec'`` the DEC stars
        of the block's own degree remain through the scaling ``S``; see ``RHMPConfig.identity_metric``).
    fused: use the fused polynomial kernels (``False`` = plain autograd reference path).
    kind: ``'poly'`` (polynomial filters), ``'resolvent'`` (self term ``(I + tau_up L_up + tau_dn L_dn)^{-1} x``
        by batched CG, DESIGN §9.2; learnable ``tau = exp(log_tau) <= TAU_MAX`` per degree and block, init 1),
        ``'solve'`` or ``'project'`` (metric Hodge projection of the edge features with the learned edge metric,
        harmonic / Kutta correction and vertex potential; :meth:`_project_forward`; uses ``solve_iters``,
        ``solve_tol``, ``solve_precond``).
    project_harmonic: projection layers remove the circulation along the cohomology generators of the complex
        (``K.meta['cohomology'][1]``) in the learned Kutta metric (default True; no effect without generators).
    kutta_range: bound of the log Kutta weights (``W / star1`` in ``[e^-a, e^a]``).
    project_solver: ``'cg'`` (preconditioned CG, :func:`metric_projection`) or ``'direct'`` (sparse LU per graph
        and sample on the CPU, :func:`direct_projection`; diagonal metrics).
    kutta_corner: initial slope of the corner prior of the Kutta weights (:class:`KuttaHead`; 0: off).
    resolvent_iters: CG iterations of resolvent layers.
    resolvent_grad: ``'implicit'`` (adjoint solve) or ``'unrolled'`` (checkpointed unrolled iterations).
    metric_type: ``'diag'`` or ``'tensor'`` (Whitney material-tensor metric for the up blocks and cross-up terms of
        the intermediate degrees, DESIGN §9.1).
    resolvent_warm_start: start the CG of a resolvent layer from the previous resolvent layer's solution of the
        same degree (stored in the forward context) when the shapes match.
    learn_metric: ``False`` removes the metric heads: ``H_m = star_m exp(ref_m)`` exactly (tensor metric:
        ``b = 1, a = 0``), the fixed DEC/FEEC operators.  ``identity_metric`` takes precedence.
    resolvent_normalize: ``False`` makes resolvent layers use the un-normalised physical operator
        ``(I + tau L^2 Delta_H)^{-1}`` (``L^D`` = domain measure) instead of ``(I + tau L_hat)^{-1}``: the metric's
        global magnitude then matters (the ``beta``-normalised ``L_hat`` is invariant to ``H -> c H``).
    solve_iters, solve_tol, solve_bc: ``kind='solve'``: CG budget and boundary condition (``'dirichlet'``: fixed
        ``K.meta['dirichlet'][k]`` or ``K.boundary[k]`` cells; ``'none'``; ``'neumann'``: no fixed cells and
        mean-free vertex sources; ``'lift'``: the fixed cells of ``'dirichlet'`` keep the layer input as
        inhomogeneous Dirichlet data, see :func:`physical_solve`).  A solve layer computes
        ``y_k = (Delta_H + lam / L^2)^{-1} x_k`` with the un-normalised metric Hodge Laplacian (physical units),
        ``lam = softplus(log_lam)`` (init 1e-3), and updates ``x_k <- gate(x_k, y_k - x_k [+ cross terms])``;
        with ``gate='none'`` (gain 1) this is ``x_k <- y_k``.
    tensor_param: ``'full'`` or ``'cone'`` parameterisation of the tensor-metric heads
        (:class:`rhmp.metric.TensorMetricHead`).
    solve_precond: ``'none'`` (Jacobi) or ``'twolevel'`` (Jacobi plus a coarse correction for vertex solves;
        :func:`physical_solve`).
    term_gates: polynomial layers with diagonal metrics only: every term of the message (each power ``L^p x``
        and each cross term) gets its own per-cell factor from scale-invariant invariants (:class:`TermGate`),
        ``m_k = sum_j c_j g_j t_j``; ``False`` (default) is the plain polynomial message.
    transport: transport specification (``rhmp.transport``; poly and resolvent layers): the vertex message
        gains ``sum_p g_p T^p x_0`` with ``T`` the dual-cell flux balance along edge transport fields and the
        layer's edge metric ``H_1``.  ``None`` / empty: off.
    solve_picard: ``kind='solve'``: number ``J`` of fixed-point (Picard) iterations of the vertex solve with a
        state-dependent metric (0: one solve with the metric of the layer input, the default).  Iteration ``j``
        predicts the metric of the vertex block (``H_1``: the diagonal edge head ``H1`` or the tensor head ``T1``)
        from the usual descriptors plus the field-energy invariant of the previous iterate
        (:meth:`picard_energy_feature`, ``y_0 = 0``), solves with the layer input as right-hand side and relaxes:
        ``y_1 = S(H_1) x``, ``y_j = y_{j-1} + omega (S(H_j) x - y_{j-1})``; weight-tied, ``omega = sigmoid(.)``
        learnable.  Other degrees of a Picard layer use one solve whose metric sees the energy invariant of the
        layer input ``x_0``.
    picard_relax: initial relaxation ``omega`` in (0, 1) of Picard layers.
    metric_geo: ``False`` removes the geometric descriptors from the inputs of the metric heads (material columns
        and feature invariants only): the learned metric is then a function of the material and of the state,
        i.e. a constitutive law, and cannot encode mesh-specific position cues.
    picard_init: state of the first Picard iteration: ``'zero'`` (``y_0 = 0``, the default: the first metric is
        the zero-field law) or ``'learned'`` (a learnable uniform energy invariant ``f_0 = softplus(.) >= 0``, a
        starting operating point, so that the zero-field law of the later iterations does not have to double as
        the initial guess; zero-initialised to ``f_0 ~ 3e-4``).
```

</details>

#### `RHMPLayer.metrics`

```python
metrics(
    x: list[Tensor],
    q: SharedCoboundaries,
    ctx: LayerContext,
    need_up: list[int] | None = None,
    need_dn: list[int] | None = None,
    extra: dict | None = None,
) -> tuple[list[Tensor | None], list[Tensor | None]]
```
Log-metrics of the blocks.

<details><summary>docstring</summary>

```text
Args:
    x: layer input features.
    q: shared coboundaries of ``x``.
    ctx: context.
    need_up: degrees ``m`` whose up-usage metric is needed (default: ``1..K``).
    need_dn: degrees ``m`` whose down-usage metric is needed (default: ``0..K-1``).
    extra: optional ``{m: (n_m, B)}`` invariants appended to the head features of degree ``m`` (Picard
        layers: the field-energy invariant on edges).

Returns:
    ``(logH_up, logH_dn)``: ``logH_up[m]`` ``(n_m, B)`` is used by the up block of degree ``m-1`` and
    ``logH_dn[m]`` by the down block of degree ``m+1`` (``None`` where not requested).
```

</details>

#### `RHMPLayer.tensor_metric`

```python
tensor_metric(
    km: int,
    x: list[Tensor],
    q: SharedCoboundaries,
    ctx: LayerContext,
    extra: tuple[Tensor, Tensor] | None = None,
) -> TensorMetric
```
The Galerkin material-tensor metric of degree `km` (`TensorMetric`); `extra = (on top cells (n_top, B), on edges (n_1, B))` invariants appended to the descriptors (Picard layers: field energy).

<details><summary>docstring</summary>

```text
Learned full tensors (``tensor_param='full'``): ``sigma_f = b_f expm(sum_j s_j t_j t_j^T)`` (any SPD tensor;
``s = log_range tanh(.)``, zero-init -> ``b_f I``) assembled into explicit Galerkin element matrices; learned cone
tensors: ``b_f I + sum_j a_j t_j t_j^T`` with ``a >= 0`` (``rhmp.dec`` blocks); fixed metrics
(``learn_metric=False``): ``b = 1, a = 0``.  Material references scale ``sigma_f`` (see
``RHMPConfig.metric_reference``).
```

</details>

#### `RHMPLayer.tensor_block`

```python
tensor_block(k: int, x: list[Tensor], q1: Tensor, tm: TensorMetric, ctx: LayerContext) -> Tensor
```
Up block of degree `k` with the Whitney tensor metric `H_{k+1}(b, a)` (DESIGN §9.1).

<details><summary>docstring</summary>

```text
``sum_p c_p L^p x_k + w T x_{k+1}`` with ``L = A^T H A / beta`` and ``T = A^T H / (sqrt(beta_unit) rho)``
(:func:`tensor_operands`; scaling folded into ``A``).  One metric application per ``L`` application: the
polynomial and the cross term share the final ``A^T H (.)``.
```

</details>

#### `RHMPLayer.block`

```python
block(
    up: bool,
    k: int,
    x: list[Tensor],
    q1: Tensor | None,
    log_H: Tensor,
    ctx: LayerContext,
) -> Tensor
```
Message of the up (`up=True`) or down block of degree `k`, including its cross term.

<details><summary>docstring</summary>

```text
Args:
    up: block kind.
    k: degree.
    x: layer input features per degree.
    q1: shared first coboundary ``A (s ⊙ x_k)`` (``None`` for Jacobi scaling).
    log_H: block metric ``(m, B)``.
    ctx: context.

Returns:
    ``(n_k, B, C)``.
```

</details>

#### `RHMPLayer.block_terms`

```python
block_terms(
    up: bool,
    k: int,
    x: list[Tensor],
    q1: Tensor | None,
    log_H: Tensor,
    ctx: LayerContext,
) -> tuple[list[Tensor], list[Tensor]]
```
The separate terms of `block` (for per-term gates): `[L x, .., L^P x, T x_nb]` and their scalar coefficients `[c_1, .., c_P, w]` (the cross term only with `cross`); each term `(n_k, B, C)`.

#### `RHMPLayer.forward`

```python
forward(x: list[Tensor], ctx: LayerContext, degrees: list[int] | None = None) -> list[Tensor]
```
One synchronous update of all degrees (or only of `degrees`).

<details><summary>docstring</summary>

```text
Args:
    x: ``[x_0, .., x_K]`` with ``x_k`` ``(n_k, B, C)``.
    ctx: forward context.
    degrees: degrees to update (default: all).  Features of other degrees are returned unchanged; the
        model uses this in its last layer, where only the readout degree matters.

Returns:
    Updated features, same shapes.
```

</details>

#### `RHMPLayer.metric_fields`

```python
metric_fields(x: list[Tensor], ctx: LayerContext) -> dict
```
The metrics this layer would use on input `x` (no update, no grad).

<details><summary>docstring</summary>

```text
Returns:
    ``{'log_ratio': {m: (n_m, B)}, 'phi': {m: (n_m, B)}, 'tensor': {km: (b, a)}, 'sigma': {km: (n_top, B, D,
    D)}}`` where ``log_ratio`` is
    ``log(H_m / star_m)`` of the up-usage (tied) metric including the reference offset (up to a per-sample
    constant: stars are normalised per sample), ``phi`` its learned bounded part, and ``tensor`` the material
    tensor parameters ``b`` ``(n_top, B)``, ``a`` ``(n_top, B, m)`` (after the reference scaling).  Untied
    metrics report the up heads for ``m >= 1``; for ``m = 0`` (down usage only) ``log_ratio = -log(H_down
    star_0)``, i.e. the equivalent tied metric (``phi = -phi_down``).
```

</details>

#### `RHMPLayer.tau`

```python
tau(key: str) -> Tensor
```
Resolvent time step `tau = exp(log_tau) <= TAU_MAX` of block `key` (`'up{k}'` / `'dn{k}'`).

#### `RHMPLayer.picard_energy_feature`

```python
picard_energy_feature(y: Tensor, ctx: LayerContext) -> tuple[Tensor, Tensor]
```
Field-energy invariant of a vertex field `y` `(n_0, B, C)` for Picard layers.

<details><summary>docstring</summary>

```text
``e_f = (1/C) sum_c (d_0 y_c)_f^T G_f (d_0 y_c)_f / |f|`` per top cell (``G_f`` = the Whitney/Galerkin star block
of the unit tensor, so ``e_f`` = the mean over channels of ``|grad y_c|^2`` of the P1 field in physical units),
returned as ``0.5 log(1 + e / e_0)`` on the top cells and its mean over the incident top cells on the edges
(``e_0 = exp(picard_log_e0)`` learnable).  O(C)-, E(n)-, orientation- and relabelling-invariant, per sample.

Returns:
    ``(on top cells (n_top, B), on edges (n_1, B))``.
```

</details>

### class `BlockOperands`

```python
BlockOperands(
    A: ForwardRef('Tensor'),
    AT: ForwardRef('Tensor'),
    s: ForwardRef('Tensor | None'),
    scale: ForwardRef('Tensor | None'),
    log_h: ForwardRef('Tensor'),
    beta: ForwardRef('Tensor'),
    nb: ForwardRef('int'),
)
```
Bases: `tuple`.

Normalised operands of one metric Hodge block (see module docstring).

<details><summary>docstring</summary>

```text
Attributes:
    A: CSR ``(m, n_k)`` used in the products (``A diag(s)`` when the scaling is folded in).
    AT: CSR ``(n_k, m)``, transpose of ``A``.
    s: scaling that still has to be applied explicitly (Jacobi: ``(n_k, B, 1)``), else ``None``.
    scale: the block scaling ``S^{-1/2}`` (``(n_k, 1, 1)``, ``(n_k, B, 1)``) or ``None`` (identity).
    log_h: ``log(H / beta)`` on the ``m`` neighbour cells, ``(m, B)``.
    beta: per-sample Gershgorin bound, ``(1, B)`` or ``(num_graphs, B)``.
    nb: degree of the neighbour cells (``k+1`` for up, ``k-1`` for down).
```

</details>

### `block_operands`

```python
block_operands(ctx: LayerContext, k: int, up: bool, log_H: Tensor) -> BlockOperands
```
Scaling, Gershgorin bound and normalised metric of the up (`up=True`) or down block of degree `k`.

<details><summary>docstring</summary>

```text
Args:
    ctx: forward context.
    k: degree of the block's cochains.
    up: up block (``A = d_k``, metric on ``(k+1)``-cells) or down block (``A = d_{k-1}^T``, metric on
        ``(k-1)``-cells).
    log_H: ``log H`` of the block metric, ``(m, B)``.

Returns:
    :class:`BlockOperands`.
```

</details>

### `apply_L`

```python
apply_L(op: BlockOperands, z: Tensor) -> Tensor
```
Normalised block operator `L z = s ⊙ A^T (h ⊙ A (s ⊙ z))`: `(n_k, B, C) -> (n_k, B, C)`.

### `apply_T`

```python
apply_T(op: BlockOperands, y: Tensor) -> Tensor
```
Half operator `T y = s ⊙ A^T (sqrt(h) ⊙ y)` (cross-degree transport): `(m, B, C) -> (n_k, B, C)`.

### `poly_block`

```python
poly_block(
    q1: Tensor,
    h: Tensor,
    c: Tensor,
    u: Tensor | None,
    X: Tensor | None,
    op: BlockOperands,
) -> Tensor
```
Fused polynomial block (see `_PolyBlockFn`); requires the scaling to be folded into `op.A`.

<details><summary>docstring</summary>

```text
Args:
    q1: ``A x`` (with the folded scaling), ``(m, B, C)``.
    h: ``exp(op.log_h)`` as ``(m, B, 1)``.
    c: polynomial coefficients ``(P,)``.
    u: ``w * sqrt(h)`` ``(m, B, 1)`` or ``None`` (no cross term).
    X: neighbour-degree features ``(m, B, C)`` (used iff ``u`` is given).
    op: block operands.

Returns:
    ``sum_p c_p L^p x + w T X``, ``(n_k, B, C)``.
```

</details>

### `poly_block_reference`

```python
poly_block_reference(
    x: Tensor,
    h: Tensor,
    c: Tensor,
    u: Tensor | None,
    X: Tensor | None,
    op: BlockOperands,
    q1: Tensor | None = None,
) -> Tensor
```
Plain-autograd reference of `poly_block` (also used for `scaling='jacobi'`).

<details><summary>docstring</summary>

```text
Args:
    x: block input ``(n_k, B, C)``.
    h, c, u, X, op: as in :func:`poly_block`.
    q1: optional precomputed ``A (s ⊙ x)``.

Returns:
    ``(n_k, B, C)``.
```

</details>

### `spmm_ad`

```python
spmm_ad(A: Tensor, AT: Tensor, x: Tensor) -> Tensor
```
Autograd-aware `A @ x` on `(n, B, C)` tensors (backward uses the precomputed transpose `AT`).

<details><summary>docstring</summary>

```text
Args:
    A: CSR ``(m, n)``.
    AT: CSR ``(n, m)``, the transpose of ``A`` (used for the backward; no run-time transposes).
    x: ``(n, B, C)`` (any trailing shape).

Returns:
    ``(m, B, C)``.
```

</details>

### class `SharedCoboundaries`

```python
SharedCoboundaries(x: list[Tensor], ctx: LayerContext) -> None
```
Lazily computed, memoised first coboundary products of one layer input.

<details><summary>docstring</summary>

```text
``up(k) = (d_k diag(s_up)) x_k`` ``(n_{k+1}, B, C)`` and ``dn(k) = (d_{k-1}^T diag(s_dn)) x_k``
``(n_{k-1}, B, C)`` (raw coboundaries for Jacobi scaling); each is the first sparse product of the
corresponding block and also feeds the metric invariants.
```

</details>

#### `SharedCoboundaries.up`

```python
up(k: int) -> Tensor
```
First coboundary of the up block of degree `k`.

#### `SharedCoboundaries.dn`

```python
dn(k: int) -> Tensor
```
First coboundary of the down block of degree `k`.

### `mean_square`

```python
mean_square(x: Tensor) -> Tensor
```
Per-cell mean square over channels `|x|^2 / C`: `(n, B, C) -> (n, B)`.

### `residual_scale`

```python
residual_scale(x: Tensor, m: Tensor, a: Tensor) -> Tensor
```
`x + m ⊙ a` for `x, m` `(n, B, C)` and a per-cell scale `a` `(n, B, 1)`.

### `row_linear`

```python
row_linear(x: Tensor, W: Tensor, b: Tensor | None = None) -> Tensor
```
`F.linear` for per-cell features `(n, B, in) -> (n, B, out)` with a fast weight gradient.

<details><summary>docstring</summary>

```text
Args:
    x: ``(..., in)``.
    W: ``(out, in)`` (may be a column slice of a larger weight).
    b: ``(out,)`` or ``None``.
```

</details>

### `linear`

```python
linear(mod: nn.Linear, x: Tensor) -> Tensor
```
Apply an `nn.Linear` module through `row_linear`.

### `mlp2`

```python
mlp2(x: Tensor, W1: Tensor, e: Tensor, W2: Tensor, b2: Tensor | None) -> Tensor
```
Memory-lean two-layer MLP `W2 silu(x W1^T + e) + b2` for per-cell features (see `_MLP2Fn`).

<details><summary>docstring</summary>

```text
Args:
    x: ``(..., in)``.
    W1: ``(hidden, in)`` (may be a column slice of a larger weight).
    e: additive pre-activation broadcastable to ``(..., hidden)`` (includes the first bias).
    W2: ``(out, hidden)``.
    b2: ``(out,)`` or ``None``.

Returns:
    ``(..., out)``.
```

</details>

### `resolvent_matvec`

```python
resolvent_matvec(
    y: Tensor,
    ops_up: tuple | None,
    ops_dn: tuple | None,
    h_up,
    h_dn,
    s_up,
    s_dn,
    tau_up,
    tau_dn,
) -> Tensor
```
`(I + tau_up L_up + tau_dn L_dn) y` with `tau` folded into the (normalised) metrics.

<details><summary>docstring</summary>

```text
Args:
    y: ``(n_k, B, C)``.
    ops_up, ops_dn: ``(A, AT)`` of the blocks or ``None``.
    h_up, h_dn: ``exp(log_h)`` ``(m, B, 1)``; s_up, s_dn: explicit scalings (Jacobi) or ``None``.
    tau_up, tau_dn: scalar tensors.
```

</details>

### `resolvent_apply`

```python
resolvent_apply(
    ctx: LayerContext,
    k: int,
    x: Tensor,
    op_up: BlockOperands | None,
    op_dn: BlockOperands | None,
    tau_up: Tensor | None,
    tau_dn: Tensor | None,
    iters: int,
    grad: str = 'implicit',
    x0: Tensor | None = None,
) -> tuple[Tensor, Tensor]
```
`y = (I + tau_up L_up + tau_dn L_dn)^{-1} x` for degree `k` by batched CG (`rhmp.dec.cg_solve`).

<details><summary>docstring</summary>

```text
Per-sample (per-graph) Frobenius inner products over (cells, channels) make the solve O(C)-equivariant and
batch-independent.  A fixed number of iterations runs without host synchronisation; samples freeze at a relative
residual of ``1e-12``.  ``cond <= 1 + tau_up + tau_dn`` since ``||L|| <= 1``.

Warm start: with an initial guess ``x0`` (e.g. the previous resolvent layer's solution) the correction
``delta = M^{-1} (x - M x0)`` is solved from zero and ``y = x0 + delta`` (``x0`` detached).  Autograd through the
right-hand side makes the implicit gradients exact: ``-lam^T (dM) (delta + x0) = -lam^T (dM) y``.

Channels: the solve acts identically on every channel and commutes with channel mixing, ``M^{-1}(x Q) =
(M^{-1} x) Q``, so there is no equivariant way to reduce its cost by mixing or compressing channels first; the
cost is ``iters`` matvecs on ``(n_k, B, C)`` (plus the adjoint solve in the backward).

Args:
    ctx: context.
    k: degree.
    x: ``(n_k, B, C)``.
    op_up, op_dn: normalised block operands (``None`` if the block does not exist).
    tau_up, tau_dn: scalar tensors ``> 0``.
    iters: CG iterations.
    grad: ``'implicit'`` (adjoint CG solve, memory independent of ``iters``) or ``'unrolled'`` (autograd through
        the iterations, recomputed in the backward via checkpointing; memory grows with ``iters``).
    x0: optional initial guess ``(n_k, B, C)`` (treated as a constant).

Returns:
    ``(y, rel_res)``: ``(n_k, B, C)`` and the per-sample relative residual ``|x - M y| / |x|`` ``(S, B)`` (no grad).
```

</details>

### class `TensorOperands`

```python
TensorOperands(
    A: ForwardRef('Tensor'),
    AT: ForwardRef('Tensor'),
    tm: ForwardRef("'TensorMetric'"),
    km: ForwardRef('int'),
    inv_beta: ForwardRef('Tensor'),
    cross_scale: ForwardRef('Tensor'),
    beta: ForwardRef('Tensor'),
)
```
Bases: `tuple`.

Normalised operands of an up block with a Whitney material-tensor metric (DESIGN §9.1).

<details><summary>docstring</summary>

```text
Attributes:
    A, AT: (scaled) operator ``d_k diag(s)`` and its transpose.
    tm: the :class:`TensorMetric` ``H_{k+1}``.
    km: metric degree ``k + 1``.
    inv_beta: ``1 / beta`` broadcast to the ``km``-cells ``(1|n_km, B, 1)``; ``beta = rho * beta_unit``.
    cross_scale: ``1 / (sqrt(beta_unit) rho)`` broadcast likewise (normalises the cross-up transport).
    beta: ``(S, B)``.
```

</details>

### `tensor_operands`

```python
tensor_operands(ctx: LayerContext, k: int, tm: TensorMetric) -> TensorOperands
```
Operands of the up block of degree `k` with the tensor metric `H_{k+1}` (`tm`).

<details><summary>docstring</summary>

```text
``rho = max_e rowsum|H|_e >= lambda_max(H)`` (per sample), ``beta_unit >= lambda_max(A^T A)``, hence
``||A^T H A / (rho beta_unit)|| <= 1`` and ``||A^T H / (sqrt(beta_unit) rho)|| <= 1``.  With ``frozen_units``
``beta_unit`` is the training mesh's (``ctx.beta_scale``), as for the diagonal blocks.
```

</details>

### `apply_L_tensor`

```python
apply_L_tensor(ctx: LayerContext, op: TensorOperands, z: Tensor) -> Tensor
```
`L z = A^T H A z / beta`: `(n_k, B, C) -> (n_k, B, C)`.

### `apply_T_tensor`

```python
apply_T_tensor(ctx: LayerContext, op: TensorOperands, y: Tensor) -> Tensor
```
Cross-up transport `T y = A^T H y / (sqrt(beta_unit) rho)`: `(n_{k+1}, B, C) -> (n_k, B, C)`.

### `block_beta_unit`

```python
block_beta_unit(ctx: LayerContext, k: int) -> Tensor
```
Gershgorin bound of `A^T A` for the (scaled) up-block operator of degree `k` (tensor metrics), cached.

<details><summary>docstring</summary>

```text
Returns:
    ``(1, 1)`` for a single complex or ``(num_graphs, 1)`` for a block-diagonal batch.
```

</details>

### `unit_reference`

```python
unit_reference(K: Any, scaling: str, dtype: torch.dtype) -> dict
```
The per-mesh geometric normalisers of one complex in its own units (what `RHMPConfig.frozen_units` stores): `log_gm` `(K+1,)` log geometric means of the reference stars, `geo_log_median` per degree `(G_k,)` (the medians behind the relative descriptors, 0 for scale-free columns), `geo_min` / `geo_max` per degree `(G_k,)` (range of every descriptor column over the complex) and `beta_unit` `{(k, up): ()}`, the unit-metric Gershgorin bounds of the scaled block operators.

<details><summary>docstring</summary>

```text
Raises:
    ValueError: for a block-diagonal batch of several meshes or a complex without ``meta['geo_log_median']``.
```

</details>

### class `PhysicalHodge`

```python
PhysicalHodge(ctx: LayerContext, up: tuple | None, dn: tuple | None) -> None
```
Un-normalised metric Hodge operator of degree `k` in the symmetric frame (physical units, no `beta`)::

<details><summary>docstring</summary>

```text
    A_sym = star_k^{-1/2} d_k^T H_{k+1} d_k star_k^{-1/2}  +  star_k^{1/2} d_{k-1} H^down_{k-1} d_{k-1}^T star_k^{1/2}

so that the metric Hodge Laplacian on cochain values is ``Delta_H = star_k^{-1/2} A_sym star_k^{1/2}`` and the
weak (FEM) stiffness is ``star_k^{1/2} A_sym star_k^{1/2}`` (``= d_0^T H_1 d_0`` for k = 0).  Built from the
cached DEC-scaled operators (``ctx.blocks``); the per-graph star normalisation constants are folded into the
effective metrics.  ``up`` is ``('diag', geometry, H (m, B, 1))`` or ``('tensor', geometry, TensorMetric)``; ``dn`` is
``('diag', geometry, H (m, B, 1))``.
```

</details>

#### `PhysicalHodge.apply`

```python
apply(z: Tensor) -> Tensor
```
`A_sym z`: `(n_k, B, C) -> (n_k, B, C)`.

#### `PhysicalHodge.diagonal`

```python
diagonal() -> Tensor
```
Diagonal of `A_sym` per cell and sample `(n_k, B, 1)` (exact for diagonal metrics; for tensor metrics the within-block couplings are dropped).  Used as a (detached) Jacobi preconditioner.

#### `PhysicalHodge.params`

```python
params() -> list[Tensor]
```
Tensors inside `apply` that require gradients (for implicit CG gradients).

### `physical_hodge`

```python
physical_hodge(
    ctx: LayerContext,
    k: int,
    logH_up: list,
    logH_dn: list,
    tensors: dict | None = None,
    w_up: Tensor | None = None,
    w_dn: Tensor | None = None,
) -> PhysicalHodge
```
`PhysicalHodge` of degree `k` from a layer's log-metrics (`logH_up[k+1]`, `logH_dn[k-1]`, normalised-star units) or its tensor metric `tensors[k+1]` (`TensorMetric`); optional scalar block weights.

### `physical_solve`

```python
physical_solve(
    ctx: LayerContext,
    k: int,
    x: Tensor,
    op: PhysicalHodge,
    shift: Tensor,
    free: Tensor | None,
    iters: int,
    tol: float,
    x0: Tensor | None = None,
    precond: str = 'none',
    mean_free: bool = False,
    lift: bool = False,
) -> tuple[Tensor, Tensor, Tensor, int]
```
`y = (Delta_H + shift)^{-1} x` on the free cells (`x = 0` imposed on fixed cells) by batched (P)CG with implicit gradients (`pcg_solve_implicit`, per-sample / per-graph inner products).

<details><summary>docstring</summary>

```text
Solved in the symmetric frame, ``y = star^{-1/2} (P (A_sym + shift) P)^{-1} P star^{1/2} x``; on the free cells this
is ``(K + shift M)^{-1} M x`` with the weak stiffness ``K`` and the lumped mass ``M = star_k``.  Jacobi
preconditioned (symmetric diagonal change of variables, per cell and sample), which removes the spread of the
physical metric (slivers, material contrast) from the CG condition number.

Args:
    ctx, k: context and degree; x: ``(n_k, B, C)`` (cochain values).
    op: the physical operator; shift: ``(1|n_k, 1|B, 1)`` identity shift (physical units).
    free: free-cell mask ``(n_k, 1, 1)`` or ``None``; iters, tol: CG budget; x0: optional warm start in the
        symmetric frame (detached).
    precond: ``'none'`` (Jacobi) or ``'twolevel'`` (Jacobi + additive coarse correction on spatial aggregates,
        :func:`two_level_preconditioner`; degree 0 only, where the near-kernel is the constants -- other degrees
        use Jacobi).
    mean_free: vertex solves with the natural boundary condition (``solve_bc='neumann'``): the lumped source
        ``M x`` is made compatible by removing its mean per graph, sample and channel (the ``eps -> 0`` limit of
        ``(K + eps I) y = M x``, as in the T5 generator), so the solution is the zero-mean (``1^T M y = 0``)
        solution of the pure Neumann problem instead of carrying a ``1/shift``-sized constant.
    lift: inhomogeneous Dirichlet data (``solve_bc='lift'``): the input on the fixed cells is the boundary value
        ``g`` (``y = x`` there) and on the free cells the source, ``y_free = (K + shift M)_ff^{-1} (M x - (K + shift
        M) g)_f``; the lift ``g`` enters through the right-hand side, so the implicit gradients stay exact.

Returns:
    ``(y, y_sym, rel_res, n_iter)``: solution ``(n_k, B, C)``, its symmetric-frame form (for warm starts), the final
    relative CG residual ``(S, B)`` and the number of CG iterations run.
```

</details>

### `dirichlet_free`

```python
dirichlet_free(ctx: LayerContext, k: int, bc: str) -> Tensor | None
```
Mask of the free k-cells `(n_k, 1, 1)` (1 free, 0 fixed) or `None` (`bc='none'` / `'neumann'`).

<details><summary>docstring</summary>

```text
Fixed cells: ``K.meta['dirichlet'][k]`` (bool ``(n_k,)``, list or dict per degree) when the task provides it,
else the boundary cells ``K.boundary[k]``.  ``bc='lift'`` has the same fixed cells (their values are the layer
input, see :func:`physical_solve`).
```

</details>

### `dyad_gram_inverse`

```python
dyad_gram_inverse(K: Any, km: int, dtype: torch.dtype) -> Tensor
```
Inverse Gram matrices of the edge dyads `t_j t_j^T` of every top cell, `(n_top, m, m)` (cached).

<details><summary>docstring</summary>

```text
``G_f[i, j] = <t_i t_i^T, t_j t_j^T>_F = (t_i . t_j)^2``; the dyads span Sym(2) on triangles (their tangent plane on
surfaces) and Sym(3) on tets, so ``G_f`` is invertible for non-degenerate cells.
```

</details>

### class `TensorMetric`

```python
TensorMetric(
    ctx: LayerContext,
    km: int,
    b: Tensor | None = None,
    a: Tensor | None = None,
    blocks: Tensor | None = None,
    sigma: Tensor | None = None,
) -> None
```
Galerkin material-tensor metric `H_km` of one forward pass (per top cell and sample).

<details><summary>docstring</summary>

```text
``cone``: coefficients ``(b, a)`` of the precomputed Whitney blocks (``rhmp.dec``; ``sigma = b I + sum a t t^T``).
``blocks``: explicit element matrices ``M_f (n_top, B, m, m)`` of full tensors ``sigma_f`` (:func:`galerkin_blocks`;
exact for any SPD tensor and well conditioned for any cell shape).  Both use the slot maps of ``K.whitney[km]``.
```

</details>

#### `TensorMetric.scale`

```python
scale(f: Tensor) -> TensorMetric
```
Metric multiplied by a positive per-top-cell factor `f` `(n_top|1, 1|B)`.

#### `TensorMetric.apply`

```python
apply(x: Tensor) -> Tensor
```
`H x`: `(n_km, B, C) -> (n_km, B, C)` (per-sample bitwise independent forward).

<details><summary>docstring</summary>

```text
The slot-major blocks are built once per grad mode (e.g. once for all CG iterations under ``no_grad`` and once
for the gradient pass) instead of on every application.
```

</details>

#### `TensorMetric.rowsum_abs`

```python
rowsum_abs() -> Tensor
```
Block-wise upper bound of the row sums of `|H|` `(n_km, B)` (Gershgorin; differentiable).

#### `TensorMetric.diagonal`

```python
diagonal() -> Tensor
```
Diagonal of `H` `(n_km, B)`.

#### `TensorMetric.params`

```python
params() -> list[Tensor]
```
Tensors inside `apply` that require gradients.

#### `TensorMetric.tensor`

```python
tensor() -> Tensor
```
`sigma_f` `(n_top, B, D, D)`.

#### `TensorMetric.as_dyads`

```python
as_dyads() -> tuple[Tensor, Tensor]
```
`(b, a)` with `sigma = b I + sum_j a_j t_j t_j^T` (reporting).

<details><summary>docstring</summary>

```text
Full tensors: the minimum-norm coordinates in the basis ``{I, t_j t_j^T}`` (pseudo-inverse in orthonormal
coordinates of Sym(D)); they stay bounded on needle-shaped cells, while on flat caps any dyad representation is
ill-conditioned -- prefer :meth:`tensor` (``metric_fields(...)['sigma']``) for per-cell material fields.
```

</details>

### `galerkin_geometry`

```python
galerkin_geometry(K: Any, km: int, dtype: torch.dtype) -> dict
```
Quadrature data of the Whitney `km`-form Galerkin star of every top cell (cached per complex).

<details><summary>docstring</summary>

```text
The basis functions are those of ``K.whitney[km]`` (same local slots and orientation signs; ``rhmp.geometry``),
evaluated in float64 from ``K.pos`` at the degree-2 rule (exact for ``int w_i . sigma_f w_j`` with constant
``sigma_f``): edges of triangles/surfaces (3 x 3 blocks), edges (6 x 6) and faces (4 x 4) of tets.

Returns:
    dict with ``W (n_top, Q, D, m)`` basis values and ``wq (n_top, Q)`` weights (cell measure included).
```

</details>

### `galerkin_blocks`

```python
galerkin_blocks(geom: dict, sigma: Tensor) -> Tensor
```
Galerkin star blocks `M_f[i, j] = int_f w_i . sigma_f w_j` for per-cell tensors (differentiable in sigma).

<details><summary>docstring</summary>

```text
Args:
    geom: :func:`galerkin_geometry`.
    sigma: ``(n_top, B, D, D)`` symmetric.

Returns:
    ``(n_top, B, m, m)`` symmetric blocks in the slot basis of ``K.whitney[km]``.
```

</details>

### class `TermGate`

```python
TermGate(n_terms: int, hidden: int = 16) -> None
```
Bases: `Module`.

Per-term gates of a polynomial layer (O(C)-equivariant, homogeneous of degree zero in the field amplitude).

<details><summary>docstring</summary>

```text
The message of degree ``k`` is ``m = sum_j c_j g_j t_j`` over its terms ``t_j`` (the powers ``L_up^p x``,
``L_dn^p x`` and the cross terms ``T_up x_{k+1}``, ``T_dn x_{k-1}``) with per-cell factors
``g = 2 sigmoid(MLP(xi))`` from the scale-invariant invariants of the term vectors relative to the layer input,
``xi_j = [log((|t_j|^2/C + eps_j) / (|x|^2/C + eps_j)), <x, t_j> / (C sqrt((|x|^2/C + eps_j)(|t_j|^2/C +
eps_j)))]`` with ``eps_j = HOMOG_EPS (|x|^2 + |t_j|^2) / C`` (``MLP: 2 J -> hidden -> SiLU -> J``, last layer
zero-initialised: ``g = 1``, the plain polynomial layer, at initialisation).  The factors are O(C)-invariant and
even, so the message stays an O(C)-equivariant odd cochain; the filter becomes cell- and configuration-dependent.

Args:
    n_terms: ``J``.
    hidden: hidden width of the MLP.
```

</details>

#### `TermGate.forward`

```python
forward(x: Tensor, terms: list[Tensor]) -> Tensor
```
Factors `(n, B, J)` for the terms `[(n, B, C)] * J` of the input `x` `(n, B, C)`.

### `metric_projection`

```python
metric_projection(
    ctx: LayerContext,
    x1: Tensor,
    up: tuple,
    free: Tensor | None,
    iters: int,
    tol: float,
    precond: str = 'none',
) -> tuple[Tensor, Tensor, Tensor, int]
```
Potential of the metric (Hodge / Leray) projection of edge cochains.

<details><summary>docstring</summary>

```text
``alpha = argmin_a ||x1 - d0 a||_H^2`` over the vertex potentials that vanish on the fixed vertices:
``(d0^T H d0) alpha = d0^T H x1`` on the free vertices, with the natural condition on the free part of the
boundary.  ``x1 - d0 alpha`` is the ``H``-orthogonal projection of ``x1`` onto the cochains that are co-closed on the
free vertices, ``d0^T H (x1 - d0 alpha) = 0`` there; for the 1-form of a uniform flow and ``H`` the reference star it
is the potential flow around the free boundary (no flux through it).  The result does not depend on the scale of
``H``.  Linear in ``x1`` and identical for every channel (O(C)-equivariant), per sample and per graph.  Jacobi (or
two-level) preconditioned CG with implicit gradients (:func:`pcg_solve_implicit`).

Args:
    ctx: context; x1: ``(n1, B, C)`` cochain values.
    up: the metric of the degree-0 up block, ``('diag', geometry, H (n1, B, 1))`` or ``('tensor', geometry,
        TensorMetric)`` with ``geometry = ctx.blocks[(0, True)]`` (as :class:`PhysicalHodge`).
    free: free-vertex mask ``(n0, 1, 1)`` or ``None`` (no fixed vertices: the zero-mean potential of every graph).
    iters, tol, precond: CG budget and preconditioner (``'none'``: Jacobi, ``'twolevel'``).

Returns:
    ``(alpha, d0 alpha, rel_res, n_iter)``: ``(n0, B, C)``, ``(n1, B, C)``, the final relative CG residual ``(S, B)``
    and the number of iterations run.
```

</details>

### class `KuttaHead`

```python
KuttaHead(
    geo0: int,
    geo1: int,
    even1: int = 0,
    hidden: int = 32,
    log_range: float = 8.0,
    corner: float = 0.0,
) -> None
```
Bases: `Module`.

Bounded log-weights `kappa = a tanh(MLP(.))` of the Kutta metric `W = star1 exp(kappa)` of a projection layer.

<details><summary>docstring</summary>

```text
Edge features: the descriptors ``geo_1``, the mean and the absolute difference over the two end vertices of the
vertex descriptors ``geo_0`` (a corner of the boundary shows up in the angle defect of its vertex), the even edge
inputs, and the log energy density of the projected harmonic cochains relative to its mean over the graph.  All of
them are invariant under E(n), O(C), relabelling and orientation changes.  ``MLP: P -> hidden -> SiLU -> 1`` with a
zero-initialised last layer (``W = star1`` at initialisation).

``corner = s > 0`` (optional prior, default off): ``kappa = a tanh(MLP(.) + s c_e)`` with a learnable slope ``s`` and
the corner indicator ``c_e in [0, 1]`` of the boundary edges at a re-entrant corner of the domain (:func:`corner
indicator <boundary_corner>`), where the potential flow is singular: the weights start concentrated at sharp
trailing edges instead of learning that from the data.

Args:
    geo0, geo1: descriptor widths of vertices and edges; even1: even edge input columns; hidden; log_range: ``a``.
    corner: initial slope of the corner prior (0: no prior).
```

</details>

#### `KuttaHead.forward`

```python
forward(geo: Tensor, feats: Tensor, corner: Tensor | None = None) -> Tensor
```
`geo (n1, geo1 + 2 geo0)`, `feats (n1, B, 1 + even1)`, `corner (n1,)` -> `kappa (n1, B)` in `[-a, a]`.

### `reference_energy`

```python
reference_energy(ctx: LayerContext) -> Tensor
```
Per-graph reference energy density of edge cochains `(1|G, 1|B)`: the value set by the model (`ctx.energy_ref`: the Galerkin energy density of the amplitude 1-form, `RHMPConfig.amplitude_energy`), else the energy density of a cochain of unit values under the diagonal reference star, `sum_e star1_e / sum_i star0_i`. A cochain of typical value 1 belongs to a field of magnitude `sqrt(E_ref)` (in inverse length units).

### `direct_projection`

```python
direct_projection(
    ctx: LayerContext,
    x1: Tensor,
    H: Tensor,
    free: Tensor | None,
) -> tuple[Tensor, Tensor]
```
The potential of `metric_projection` for a diagonal metric by sparse direct solves (one factorisation per graph and sample on the CPU, float64; exact to round-off, no iteration count, gradients by the adjoint method).

<details><summary>docstring</summary>

```text
Args:
    ctx: context; x1: ``(n1, B, C)``; H: ``(n1, B)`` positive; free: free-vertex mask ``(n0, 1, 1)`` or ``None``.
Returns:
    ``(alpha (n0, B, C), d0 alpha (n1, B, C))``; without fixed vertices the zero-mean potential of every graph.
```

</details>

### `dense_projection`

```python
dense_projection(
    ctx: LayerContext,
    x1: Tensor,
    up: tuple,
    free: Tensor | None,
) -> tuple[Tensor, Tensor]
```
The potential of `metric_projection` by batched dense Cholesky factorisations on the device in float64 (one factorisation per sample, exact to round-off, diagonal and tensor metrics; gradients by autograd through the factorisation).  The operator `s d0^T H d0 s` of the sample is materialised by applying it to the identity, so the solver is meant for complexes with one graph per batch and up to a few thousand vertices (the fixed meshes of the paper tasks); block-diagonal batches of meshes use `'direct'` or `'cg'`.

<details><summary>docstring</summary>

```text
Args:
    ctx: context; x1: ``(n1, B, C)``; up: ``('diag', geometry, H (n1, B, 1))`` or ``('tensor', geometry,
        TensorMetric)``; free: free-vertex mask ``(n0, 1, 1)`` or ``None``.
Returns:
    ``(alpha (n0, B, C), d0 alpha (n1, B, C))``; without fixed vertices the zero-mean potential.
```

</details>

### `direct_projection`

```python
direct_projection(
    ctx: LayerContext,
    x1: Tensor,
    H: Tensor,
    free: Tensor | None,
) -> tuple[Tensor, Tensor]
```
The potential of `metric_projection` for a diagonal metric by sparse direct solves (one factorisation per graph and sample on the CPU, float64; exact to round-off, no iteration count, gradients by the adjoint method).

<details><summary>docstring</summary>

```text
Args:
    ctx: context; x1: ``(n1, B, C)``; H: ``(n1, B)`` positive; free: free-vertex mask ``(n0, 1, 1)`` or ``None``.
Returns:
    ``(alpha (n0, B, C), d0 alpha (n1, B, C))``; without fixed vertices the zero-mean potential of every graph.
```

</details>

### `boundary_corner`

```python
boundary_corner(ctx: LayerContext) -> Tensor
```
Corner indicator of the edges, `(n1,)` in `[0, 1]`: for a boundary edge the larger of `clamp((-defect - pi/2) / (pi/2), 0, 1)` over its two end vertices (`defect = pi - interior angle sum` of a boundary vertex, the vertex descriptor `angle_defect`); 0 for interior edges and on smooth or convex parts of the boundary, close to 1 at a sharp re-entrant corner (interior angle towards 2 pi, e.g. a trailing edge seen from the fluid).  Intrinsic: E(n)-, relabelling- and orientation-invariant.

### Constants

- `TAU_MAX`: 100.0


## Building blocks: transport term: `rhmp.transport`

Advective transport fluxes of the vertex features.

<details><summary>module docstring</summary>

```text
Transport term: discrete Lie derivative of the vertex features along edge transport fields (``RHMPConfig.transport``).

Scalar transport ``d_t u + div(u v) = 0`` is generated by an operator that is not symmetric, so the symmetric positive
semi-definite Hodge blocks of the layer cannot represent it; without this term a model can only get a direction from
learned latent fields through the cross terms.  The transport term adds, on the vertices, the dual-cell flux balance
of the features along one or more *transport fields*.

A transport field ``phi`` is an odd (orientation-reversing), channel-invariant edge scalar: a velocity one-form given
as an input column of degree 1 (``'inputs'``), a latent edge column (``'latent'``), or an O(C)-invariant odd
contraction of the hidden state (``'hidden'``).  With the metric ``H_1`` of the layer (the one the up block of degree
0 uses) it defines the flux through the dual edges and three operators on the vertex features (symmetrised DEC frame
``x = star_0^{1/2} u``, ``s = star_0^{-1/2}`` the scaling of the degree-0 up block)::

    F      = (H_1 phi) / beta_up0                              flux through the dual edges (normalised metric)
    C(F) x = s ⊙ d_0^T ( F ⊙ |d_0| (s ⊙ x) ) / 2                conservative, centred: dual-cell flux balance
    A(F) x = s ⊙ |d_0|^T ( F ⊙ d_0 (s ⊙ x) ) / 2                advective, centred (A = C^T)
    U(F) x = s ⊙ d_0^T ( |F| ⊙ d_0 (s ⊙ x) ) / 2                upwind correction (graph Laplacian, weights |F| / 2)

The conservative upwind flux is ``C - U`` and the advective upwind form ``A + U``.  With the reference metric
``H_1 = star_1`` (cotan weights) ``star_1 phi`` is the DEC flux of ``v`` through the dual edges, ``C`` is the
finite-volume discretisation of ``-div(u v)`` on the dual cells and ``A`` that of ``v . grad u`` (both times the dual
area, mapped to the symmetric frame).  The message of degree 0 gains

    T   = sum_j ( a_j C(F_j) + b_j A(F_j) + c_j U(F_j) )           scalar coefficients per layer and field (init 0)
    m_0 += sum_{p=1..P} g_p T^p x_0,      g_1 = 1,  g_{p>1} learned (init 0)

Properties (tests in ``tests/test_transport.py``):

* the operator is linear in the hidden state and acts identically on every channel: O(C)-equivariant;
* ``phi`` odd and ``d_0`` odd: ``C``, ``A``, ``U`` are invariant under edge-orientation changes and vertex relabelling;
  only metrics, stars and incidence enter (E(n)-invariant);
* ``C`` and ``U`` conserve ``sum_i star_i^{1/2} x_i = sum_i star_i u_i`` exactly (``1^T d_0^T = 0``); ``A`` does not
  (it differs from ``-C`` by the divergence of the flux, ``C + A = diag(d_0^T F)`` in the physical frame);
* for a divergence-free flux (``d_0^T F = 0``) ``C`` is skew-symmetric (energy conserving);
* ``metric='star'`` replaces the learned ``H_1`` by the reference star ``star_1`` (the geometric DEC flux; for models
  whose learned metric carries a material such as an anisotropic diffusion tensor, which must not weight advection);
* norm bounds: ``||C(F)||, ||A(F)||, ||U(F)|| <= max_e |phi_e| / 2`` per sample (``norm='metric'``: the metric is
  normalised by the Gershgorin bound of the degree-0 up block, the field keeps its magnitude, which is the transport
  speed); ``norm='sample'`` divides every field operator by its own per-sample Gershgorin bound (``<= 1``, and
  invariant to the magnitude of the field).  Hidden-state fields pass through ``tanh`` (bounded, odd).

Other degrees: hidden cochains of degree ``k >= 1`` are orientation-odd, so an unsigned interpolation ``|d_k| x_k`` is
not a cochain and the construction above is specific to the (even) vertex features.  The transport of k-forms
(``L_v = d i_v + i_v d``) needs a discrete interior product with the flux (twisted) form of ``v``, which an edge
circulation does not determine; it is not implemented.
```

</details>

### `normalize_transport`

```python
normalize_transport(spec: Any) -> dict
```
Validated transport specification with defaults, or `{}` (transport off).

<details><summary>docstring</summary>

```text
Keys: ``inputs`` (list of odd column indices of ``inputs[1]``, default ``[]``), ``latent`` (number of latent edge
columns used as fields, the first ones of ``latent_dims[1]``, default 0), ``hidden`` (0, 1 or 2 hidden-state
fields, default 0), ``forms`` (subset of ``('cons', 'adv', 'upwind')``, default all), ``order`` (``P >= 1``,
default 1), ``norm`` (``'metric'`` (default) or ``'sample'``), ``metric`` (``'layer'`` (default): the layer's
learned edge metric ``H_1``; ``'star'``: the reference star ``star_1``, i.e. the geometric DEC flux, for models
whose learned metric carries a material such as an anisotropic diffusion tensor).

Raises:
    ValueError: on unknown keys or invalid values, or if no field is selected.
```

</details>

### class `TransportTerm`

```python
TransportTerm(spec: dict) -> None
```
Bases: `Module`.

Transport term of one layer (see the module docstring).

<details><summary>docstring</summary>

```text
Args:
    spec: transport specification (:func:`normalize_transport`; non-empty).  The input and latent fields are read
        from ``ctx.transport_fields`` (``(n_1, B, len(inputs) + latent)``, set by the model from the columns
        :func:`transport_columns`), the hidden fields are computed from the layer input.
```

</details>

#### `TransportTerm.extra_repr`

```python
extra_repr() -> str
```
Return the extra representation of the module.

<details><summary>docstring</summary>

```text
To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.
```

</details>

#### `TransportTerm.fields`

```python
fields(x: list[Tensor], ctx: Any) -> Tensor
```
Transport fields `phi` `(n_1, B, J)`: input / latent columns (`ctx.transport_fields`), then the hidden fields `tanh(<x_1, (|d_0| x_0) / 2> / C)` and `tanh(d_0 (|x_0|^2 / (2 C)))`.

#### `TransportTerm.fluxes`

```python
fluxes(phi: Tensor, ctx: Any, logH1: Tensor | None, tm: Any = None) -> Tensor
```
`F = (H_1 phi) / beta_up0` `(n_1, B, J)`; `norm='sample'` further divides every field by the per-sample Gershgorin bound of its operators.

#### `TransportTerm.forward`

```python
forward(
    x: list[Tensor],
    ctx: Any,
    logH1: Tensor | None,
    tm: Any = None,
    q: Tensor | None = None,
    diag: dict | None = None,
) -> Tensor
```
Transport message of the vertices `sum_p g_p T^p x_0` `(n_0, B, C)`.

<details><summary>docstring</summary>

```text
Args:
    x: layer input features.
    ctx: forward context (``scaling`` ``'dec'`` or ``'none'``).
    logH1: ``log H_1`` of the degree-0 up block ``(n_1, B)`` (diagonal metric), or ``None`` with ``tm``.
    tm: Whitney tensor metric of degree 1 (``rhmp.layers.TensorMetric``), or ``None``.
    q: optional shared first coboundary ``(d_0 diag(s)) x_0``.
    diag: optional dict receiving scalar diagnostics (``transport_absF_max``, ``transport_<form>_absmax``).
```

</details>

### `transport_columns`

```python
transport_columns(cfg: Any) -> list[int]
```
Columns of the effective `inputs[1]` (latent columns inserted, layout `[connection | odd | latent | even]`) that are transport fields: the requested odd input columns, then the first `transport['latent']` latent columns.

<details><summary>docstring</summary>

```text
Raises:
    ValueError: if a requested column is not an ordinary odd column (connection and even columns cannot carry a
        transport field: a connection would break the gauge invariance, an even column the orientation parity).
```

</details>

### `transport_weights`

```python
transport_weights(F: Tensor, coef: dict[str, Tensor]) -> dict[str, Tensor]
```
Edge weights of `T = sum_j a_j C(F_j) + b_j A(F_j) + c_j U(F_j)`: `{'cons': (n1, B, 1), ...}`.

<details><summary>docstring</summary>

```text
Args:
    F: fluxes ``(n_1, B, J)``.
    coef: ``{form: (J,)}`` coefficients (``a`` for ``'cons'``, ``b`` for ``'adv'``, ``c`` for ``'upwind'``).
```

</details>

### `apply_transport`

```python
apply_transport(g: Any, w: dict[str, Tensor], x: Tensor, q: Tensor | None = None) -> Tensor
```
`T x` for the edge weights `w` (`transport_weights`): `(n_0, B, C) -> (n_0, B, C)`.

<details><summary>docstring</summary>

```text
``C x = AT (w_C ⊙ A_abs x)``, ``U x = AT (w_U ⊙ A x)``, ``A x = AT_abs (w_A ⊙ A x)`` with the scaled operators of
the degree-0 up block (``A = d_0 diag(s)``, ``AT = diag(s) d_0^T``, absolute values likewise).

Args:
    g: ``BlockGeometry`` of the degree-0 up block.
    w: edge weights (missing forms are skipped).
    x: vertex features.
    q: optional precomputed ``A x`` (shared first coboundary of the layer).
```

</details>

### `transport_bound`

```python
transport_bound(g: Any, absF: Tensor) -> Tensor
```
Per-cell Gershgorin row bounds of `C(F)`, `A(F)` and `U(F)`: `s_i [ |d_0|^T (|F| ⊙ |d_0| s) ]_i / 2`.

<details><summary>docstring</summary>

```text
Args:
    g: the ``BlockGeometry`` of the degree-0 up block (scaling folded into ``A_abs``, ``AT_abs``; ``g.r = |d_0| s``).
    absF: ``|F|`` ``(n_1, B, J)``.

Returns:
    ``(n_0, B, J)``; the row sums and the column sums of ``|C|``, ``|A|``, ``|U|`` are bounded by these values.
```

</details>

### Constants

- `TRANSPORT_FORMS`: ('cons', 'adv', 'upwind')
- `TRANSPORT_NORMS`: ('metric', 'sample')
- `TRANSPORT_METRICS`: ('layer', 'star')
- `TRANSPORT_HIDDEN_MAX`: 2


## Building blocks: metric heads: `rhmp.metric`

Bounded log-metric and material-tensor heads.

<details><summary>module docstring</summary>

```text
Bounded log-metric heads (DESIGN §3.2).

For degree ``m`` the learned diagonal cochain metric is, per sample and per cell,

    H_m = ref_m * exp(phi_m),      phi_m = a * tanh(MLP_m(psi_m)),      a = log_range,

so that ``H_m / ref_m`` lies in ``[e^-a, e^a]``.  ``ref_m`` is the reference DEC Hodge star of the
complex (normalised per sample by its geometric mean inside the model, see ``layers.make_context``).
The last MLP layer is zero-initialised, hence ``H_m = ref_m`` (pure DEC) at initialisation.

``psi_m`` is built only from quantities that are invariant under the cochain-frame group O(C), under
E(n), under cell relabelling and under orientation changes, and it depends on one sample only:

* ``geo[m]``            ``(n_m, G_m)``   E(n)-invariant cell descriptors of the complex (broadcast over B),
* even raw inputs       ``(n_m, B, E_m)`` (e.g. a conductivity),
* feature invariants    ``(n_m, B, F_m)`` (built by the layer, see :func:`metric_feature_dim`).

The first linear layer is applied group-wise so that the ``geo`` part is computed once per cell and
broadcast over the batch (no ``(n_m, B, P_m)`` concatenation is ever materialised).
```

</details>

### class `MetricHead`

```python
MetricHead(
    geo_dim: int,
    even_dim: int,
    feat_dim: int,
    hidden: int = 32,
    log_range: float = 2.0,
) -> None
```
Bases: `Module`.

Bounded log-metric head `phi = a * tanh(MLP(psi))` for one degree.

<details><summary>docstring</summary>

```text
MLP: ``P -> hidden -> SiLU -> 1`` with ``P = geo_dim + even_dim + feat_dim``; last layer zero-init.

Args:
    geo_dim: ``G_m``, number of geometric descriptors of the degree.
    even_dim: ``E_m``, number of even raw input columns of the degree (0 if none).
    feat_dim: ``F_m``, number of feature invariants (:func:`metric_feature_dim`).
    hidden: hidden width (DESIGN default 32).
    log_range: ``a``; ``H/ref`` is confined to ``[e^-a, e^a]``.
```

</details>

#### `MetricHead.forward`

```python
forward(geo: Tensor, even: Tensor | None, feats: Tensor) -> Tensor
```
Bounded log-ratio `phi = log(H / ref)`.

<details><summary>docstring</summary>

```text
Args:
    geo: ``(n, G)`` geometric descriptors (ignored if ``G == 0``).
    even: ``(n, B, E)`` even raw inputs, or ``None`` if ``E == 0``.
    feats: ``(n, B, F)`` feature invariants.

Returns:
    ``phi``: ``(n, B)`` float32 (or the dtype of ``feats`` if it is float64), in ``[-a, a]``.
```

</details>

### class `TensorMetricHead`

```python
TensorMetricHead(
    in_top: int,
    in_edge: int,
    hidden: int = 32,
    log_range: float = 2.0,
    param: str = 'cone',
) -> None
```
Bases: `Module`.

Whitney-consistent material tensor per top cell (DESIGN §9.1).

<details><summary>docstring</summary>

```text
``param='full'`` (the ``RHMPConfig`` default): ``sigma_f = b_f expm(S_f)``, ``S_f = sum_j s_{f,j} t_j t_j^T`` with
signed ``s_{f,j} = a tanh(z_{f,j})`` (zero-init): every SPD tensor with bounded condition number (the edge dyads
span Sym(2) on triangles and Sym(3) on tets).  ``param='cone'`` (the default of this class and the
parameterisation of configurations saved without ``tensor_param``; the cone spanned by ``I`` and the edge dyads,
i.e. anisotropy along cell edges only):
``sigma_f = b_f I + sum_j a_{f,j} t_j t_j^T`` in the cell's own edge frame, with

    b_f     = exp(a * tanh(z_b))              in [e^-a, e^a]       z_b     = MLP_b(psi_f)
    a_{f,j} = e^a * tanh(max(z_{f,j}, 0))     in [0, e^a)          z_{f,j} = MLP_a([psi_f, psi_{e_j}])

where ``psi_f`` are the (O(C)-, E(n)-invariant, per-sample) descriptors of the top cell and ``psi_{e_j}`` those of
its direction edge ``j`` (so that ``a_{f,j}`` can align anisotropy with edge ``j``).  Both last layers are
zero-initialised: ``b = 1``, ``a = 0`` (the Galerkin/Whitney star of the reference metric) at initialisation;
the clamp passes gradients at 0 so ``a`` can grow from there.  The induced cochain metric is assembled by
``rhmp.dec.apply_whitney_metric``.

Args:
    in_top: width of ``psi_f``.
    in_edge: width of ``psi_{e_j}``.
    hidden: hidden width of both MLPs.
    log_range: ``a``.
    param: ``'full'`` or ``'cone'`` (see above).
```

</details>

#### `TensorMetricHead.forward`

```python
forward(psi_top: Tensor, psi_edge: Tensor) -> tuple[Tensor, Tensor]
```
Tensor parameters of every top cell.

<details><summary>docstring</summary>

```text
Args:
    psi_top: ``(n_top, B, in_top)``.
    psi_edge: ``(n_top, m, B, in_edge)`` descriptors of the ``m`` direction edges of each top cell.

Returns:
    ``(b, a)``: ``b`` ``(n_top, B)`` in ``[e^-a, e^a]`` and ``a`` ``(n_top, B, m)`` in ``[0, e^a)`` (float32 or
    the dtype of the inputs if float64).  With ``param='full'`` the second output is the signed log-tensor
    coordinate ``s`` ``(n_top, B, m)`` in ``(-a, a)`` (``sigma_f = b_f expm(sum_j s_j t_j t_j^T)``, assembled into
    explicit Galerkin blocks by ``rhmp.layers.galerkin_blocks``).
```

</details>

### `metric_feature_dim`

```python
metric_feature_dim(m: int, top: int) -> int
```
Number of feature invariants fed to the metric head of degree `m`.

<details><summary>docstring</summary>

```text
Layout (all computed per cell and per sample, see ``RHMPLayer._metric_features``)::

    log1p(|x_m|^2 / C)                                   always
    log1p(|q_m^-|^2 / C), asinh(<x_m, q_m^-> / C)       if m >= 1   (q_m^- = d_{m-1} S^-1/2 x_{m-1})
    log1p(|q_m^+|^2 / C)                                 if m < top  (q_m^+ = d_m^T S'^-1/2 x_{m+1})
    per-sample mean over m-cells of log1p(|x_m|^2 / C)   always

Args:
    m: cochain degree.
    top: top degree ``K`` of the complex.

Returns:
    ``F_m`` (3..5).
```

</details>


## Building blocks: lifting: `rhmp.lifting`

Raw inputs on any degree -> hidden cochains.

<details><summary>module docstring</summary>

```text
Cochain lifting: raw inputs on any degree -> hidden cochains (DESIGN §3.4, §3.6).

Input layout on degree ``k`` (``inputs[k]``: ``(n_k, B, F_k)``)::

    [ connection columns (c_k) | ordinary odd columns (O_k) | even columns (E_k) ]
      cfg.connection_dims[k]     F_k - c_k - E_k               cfg.even_dims[k]

Hidden cochains (all ``(n_k, B, C)``)::

    x_0 = MLP([f_0, geo_0])                                   (connection and material columns excluded)
    for k = 1..K:
        o_k = Lin_a(d_{k-1} x_{k-1}) + Lin_b(f_k^odd) + Lin_c(d_{k-1} A_{k-1})         (bias-free: odd)
        e_k = MLP([f_k^even, geo_k, log1p(|f_k^odd|^2), log1p(|d_{k-1} x_{k-1}|^2 / C),
                   mean over boundary (k-1)-cells of log1p(|x_{k-1}|^2 / C)])         (even, C-dim)
        x_k = o_k ⊙ (1 + e_k)                                                        (odd x even = odd)

With ``cfg.lift_context = p > 0`` the gates ``e_k`` (k >= 1) also see the vertex mean over the cell of a linear
projection ``P x_0`` (``C -> p``, even), i.e. the vertex quantities of the cell (densities, pressures) modulate the
lifted odd features channel by channel.  With ``cfg.lift_vertex`` the vertex MLP also sees even vertex invariants of
the edge inputs: the DEC codifferential (divergence) of the odd edge columns and the vertex means of their squares
and of the even edge columns.

*Material* columns (``cfg.material_dims[k]``: the last even columns) never enter the lifting: they only feed the
metric heads (and ``metric_reference``).  ``cfg.lifting='linear'`` drops the geometry, the biases and the even gates
(``x_0 = W f_0``, ``x_k = o_k``), so the lifting is linear in the field inputs.

``A_{k-1}`` are the connection columns of degree ``k-1``; they enter *only* through ``d_{k-1} A_{k-1}``, which
makes the network exactly invariant under ``A -> A + d_{k-2} lambda`` (``d d = 0``).  With
``cfg.connection_odd=True`` (non-Abelian connections) they additionally enter the ordinary odd path.
The last layer of every ``e_k`` MLP is zero-initialised (``x_k = o_k`` at initialisation).
```

</details>

### class `GeoMLP`

```python
GeoMLP(geo_dim: int, in_dim: int, hidden: int, out_dim: int, zero_last: bool = False) -> None
```
Bases: `Module`.

Two-layer MLP on `[feats, geo]` where the `geo` part of the first layer is computed once per cell.

<details><summary>docstring</summary>

```text
``y = fc2(SiLU(W_f feats + W_g geo + b))``.

Args:
    geo_dim: ``G`` (per-cell descriptors, broadcast over the batch); may be 0.
    in_dim: ``P`` (per-sample features); may be 0.
    hidden: hidden width.
    out_dim: output width.
    zero_last: zero-initialise the last layer.
```

</details>

#### `GeoMLP.forward`

```python
forward(feats: Tensor | None, geo: Tensor | None) -> Tensor
```
Evaluate the MLP.

<details><summary>docstring</summary>

```text
Args:
    feats: ``(n, B, P)`` (or ``None`` if ``P == 0``).
    geo: ``(n, G)`` (or ``None`` if ``G == 0``).

Returns:
    ``(n, B, out_dim)``, or ``(n, 1, out_dim)`` when ``P == 0``.
```

</details>

### class `CochainLifting`

```python
CochainLifting(cfg: Any, geo_dims: dict[int, int]) -> None
```
Bases: `Module`.

Lift raw inputs on any degree to hidden cochains `x_k` `(n_k, B, C)` (see module docstring).

<details><summary>docstring</summary>

```text
Args:
    cfg: ``RHMPConfig`` (uses ``in_dims, even_dims, material_dims, connection_dims, connection_odd, lifting, C,
        lift_hidden``).
    geo_dims: ``{k: G_k}``.
```

</details>

#### `CochainLifting.forward`

```python
forward(inputs: dict[int, Tensor], ctx: LayerContext) -> list[Tensor]
```
Lift.

<details><summary>docstring</summary>

```text
Args:
    inputs: ``{k: (n_k, B, F_k)}`` (validated by the model, already in ``ctx.dtype``).
    ctx: forward context.

Returns:
    ``[x_0, .., x_K]``, ``x_k``: ``(n_k, B, C)`` contiguous in ``ctx.dtype``.
```

</details>

### `input_layout`

```python
input_layout(cfg: Any, k: int) -> dict[str, int]
```
Column layout of `inputs[k]` for a config.

<details><summary>docstring</summary>

```text
Returns:
    ``dict(F, conn, E, M, E_lift, odd_lo, odd_hi, O)``: total width, number of connection columns, number of even
    columns, number of *material* columns (the last ``M`` even columns: metric heads / reference only), number of
    even columns seen by the lifting gates (``E - M``, columns ``[F - E, F - M)``), and the ``[odd_lo, odd_hi)``
    range of the columns entering the ordinary odd path (width ``O``).  For degree 0 the odd range is the range
    fed to the vertex map (odd/even is meaningless on vertices; material columns excluded).
```

</details>

### `vertex_mean`

```python
vertex_mean(K: Any, k: int, x0: Tensor) -> Tensor
```
Mean over the vertices of every k-cell: `(n_0, B, H) -> (n_k, B, H)` (`-1` padded slots skipped).


## Building blocks: readouts: `rhmp.readout`

Node / cochain / even / constraint / vector readouts.

<details><summary>module docstring</summary>

```text
Readouts from hidden cochains (DESIGN §3.5).

All readouts are E(n)-invariant (``node_vector``: E(n)-equivariant) and permutation-equivariant; they pick a
frame of the hidden channels, so O(C) is a property of the message-passing stack only.  Orientation behaviour:

* ``node_scalar``   ``MLP(x_0)`` -> ``(n_0, B, out_dim)`` (vertex scalars, even); ``cfg.readout_energy = h``: the MLP
  also sees, and the output adds a linear map of, ``h`` Galerkin energy densities ``<x_1 A_j, x_1 B_j>`` of projections
  of the edge features at the vertices (symmetric quadratic forms of the edge cochains in the reference metric, even;
  e.g. the Bernoulli pressure ``-|u|^2 / 2`` of a velocity 1-form).
* ``cochain:k``     bias-free linear map of the odd features ``x_k`` -> ``(n_k, B, out_dim)`` (odd for k >= 1:
  fluxes, circulations, curvatures); ``cfg.cochain_head='gated'``: odd gated head (linear x even gate);
  ``'context'``: gated head that also sees ``d_{k-1} x_{k-1}`` (odd) and the vertex mean of ``x_0`` (even gate input);
  ``cfg.readout_wedge`` (k = 2): plus face wedge products of the edge features (true 2-cochains, quadratic).
* ``even:k``        ``MLP([log1p(x_k^2), log1p(|x_k|^2/C), geo_k])`` (``x_0`` itself for k = 0) -> ``(n_k, B, out_dim)``
  (orientation-invariant cell scalars, e.g. magnitudes).
* ``grad``          ``E = d_0 phi`` with ``phi = MLP(x_0)`` (``readout_head='linear'``: ``phi = Lin_nobias(x_0)``): odd
  edge cochain ``(n_1, B, out_dim)``, exactly curl-free (``d_1 E = 0``) and invariant under ``phi -> phi + const``.
* ``curl``          ``F = d_1 a`` with ``a = Lin_nobias(x_1)``: odd face cochain ``(n_2, B, out_dim)``, exactly closed
  (``d_2 F = 0`` on volumes) and, in connection mode, exactly gauge invariant (every hidden feature is).
* ``div`` / ``div:k``  ``y = d_{k-1}^T b`` with ``b = Lin_nobias(x_k)`` (``k = K`` for ``div``): output on
  ``(k-1)``-cells ``(n_{k-1}, B, out_dim)``, exactly co-closed (``d_{k-2}^T y = 0`` for k >= 2); for ``k = 1`` it is
  a vertex divergence (even) whose sum over the vertices of every graph is exactly 0 (discrete Gauss law).
  The operators are purely combinatorial (no stars), so the constraints hold to rounding.
* ``mdiv`` / ``mdiv:k``  mass-weighted divergence ``y = S_{k-1}^{-1} d_{k-1}^T b`` with ``b = Lin_nobias(x_k)`` and
  ``S_{k-1} = exp(log_star_{k-1})`` the per-graph geometric-mean-normalised star of the output degree (outputs of
  order one).  For ``k = 1`` (``S_0`` = normalised lumped vertex mass) ``sum_i star0_i y_i = 0`` exactly per graph:
  exactly mass-conserving density increments on variable meshes.  Same orientation behaviour as ``div:k``.
* ``node_vector``   odd per-edge scalars ``w_e = Lin(x_1)_e`` (``out_dim`` vector fields) turned into vertex vectors,
  ``(n_0, B, out_dim * D)`` with ``D = K.pos.shape[1]``:

  - ``vector_mode='ls'`` (least squares, exact for constant fields when ``w_e = l_e t_e . v``)::

        M_i = sum_{e ∋ i} omega_e l_e^2 t_e t_e^T ,      b_i = sum_{e ∋ i} omega_e l_e w_e t_e
        v_i = (M_i^2 + mu_i^2 I)^{-1} M_i b_i ,         mu_i = 1e-4 * trace(M_i)

    i.e. the minimiser of ``sum_e omega_e (l_e t_e . v - w_e)^2`` computed as damped least squares on the normal
    equations ``M v = b``.  Unlike adding ``1e-6 * trace`` to ``M``, the damped form has *zero* response in
    rank-deficient directions (vertices whose edges are collinear, e.g. on straight polygon sides) instead of
    amplifying fp32 rounding by ``1e6`` there; its bias on well-posed vertices is ``O((mu / lambda_min)^2)``.
    The small ``D x D`` solves run in float64 with autocast disabled.  ``omega_e =
    softplus(MLP([log1p(x_1^2), log1p(|x_1|^2/C), geo_1])) > 0`` is a learned even weight (1 at initialisation).
    For surfaces in 3-D (top degree 2, ``D = 3``) the system is solved in the vertex tangent plane (orthonormal
    basis of the two smallest eigenvectors of ``sum_f a_f n_f n_f^T``, which does not depend on the face
    orientation), so the output is tangent and the solve stays well conditioned on curved surfaces (the full 3x3
    system is nearly singular in the normal direction there).
  - ``vector_mode='direct'`` (v1 style): ``v_i = sum_{e ∋ i} w_e t_e / deg_i``.

  Both are E(n)-equivariant (rotations, reflections, translations) and invariant to the edge-orientation
  convention (``w_e`` and ``t_e`` flip together).
```

</details>

### class `NodeScalarReadout`

```python
NodeScalarReadout(C: int, out_dim: int, energy: int = 0, energy_clip: float = 0.0) -> None
```
Bases: `Module`.

`MLP(x_0)`: `Linear(C, C) -> SiLU -> Linear(C, out_dim)`, output `(n_0, B, out_dim)`.

<details><summary>docstring</summary>

```text
``energy = h > 0`` (energy readout, simplicial complexes): with ``a = x_1 A``, ``b = x_1 B`` (``A, B: C -> h``,
bias-free) and the Galerkin energy densities at the vertices

    q_i = sum_{f ∋ i} a_f^T M_f b_f / (E_ref sum_{f ∋ i} |f|)        (h channels; ``rhmp.dec.vertex_energy``)

(``M_f``: Whitney mass block of the top cell ``f``, i.e. the reference metric ``H_1``; ``E_ref``: the reference
energy density of the graph, :func:`rhmp.layers.reference_energy`), the output is

    y = W_2 SiLU(W_1 x_0 + b_1 + V q) + b_2 + w q

with ``V`` and ``w`` zero-initialised (the plain readout at initialisation).  ``energy_clip = c > 0`` replaces
``q`` by ``c tanh(q / c)`` (a bound for meshes whose edge lengths vary over orders of magnitude, where hidden edge
features that do not scale with the edge length have huge energy densities).  ``q`` is the pointwise product ``u . v``
for cochains of constant fields in units of the reference energy, so ``w q`` contains the Bernoulli form
``c - |u|^2`` of a velocity 1-form; it is even (a product of two odd quantities), E(n)- and relabelling-invariant and
resolution consistent.  The stack is unchanged (O(C)-equivariant); the readout fixes the channel frame.
```

</details>

#### `NodeScalarReadout.energies`

```python
energies(x: list[Tensor], ctx: LayerContext) -> Tensor
```
The energy channels `q` `(n_0, B, h)` (float32 / float64, autocast disabled).

#### `NodeScalarReadout.forward`

```python
forward(x: list[Tensor], ctx: LayerContext) -> Tensor
```
`x[0]` `(n_0, B, C)` -> `(n_0, B, out_dim)`.

### class `CochainReadout`

```python
CochainReadout(
    C: int,
    out_dim: int,
    k: int,
    head: str = 'linear',
    hidden: int = 16,
    geo_dim: int = 0,
    even_dim: int = 0,
    n_hist: int = 1,
    depth: int = 1,
    skip: tuple | None = None,
    spatial: bool = False,
    wedge: int = 0,
) -> None
```
Bases: `Module`.

Bias-free linear map of `x_k` (odd for k >= 1): `(n_k, B, C) -> (n_k, B, out_dim)`.

<details><summary>docstring</summary>

```text
``head='gated'``: odd gated head ``y = W_2 (u ⊙ (1 + e))`` with ``u = W_1 x_k`` (``C -> hidden``, bias-free, odd)
and the even gate ``e = MLP([log1p(u^2), log1p(|x_k|^2/C), even_k], geo_k)`` (``hidden`` wide, last layer
zero-initialised: the linear head ``W_2 W_1 x_k`` at initialisation); ``even_k`` are the even input columns of
degree k that the lifting sees (material columns excluded).  Odd times even, so the output keeps the orientation
parity of a cochain; the head adds a local nonlinearity across the hidden channels.

``n_hist > 1`` (``readout_layers='all'``): the odd part also reads the features of degree k after the lifting and
after every earlier layer (``ctx.history``), ``u = W_1 x_k + sum_l W_l x_k^(l)``, so that every output channel has
its own combination of the filters of the stack.

``head='context'`` (k >= 1): the gated head with the features of the lower degrees of the cell:
``u = W_1 x_k + W_c d_{k-1} x_{k-1}`` (odd) and the vertex mean ``v = mean_{i in cell} (W_0 x_0 + b_0)`` (even)
as additional gate input, ``e = MLP([log1p(u^2), v, log1p(|x_k|^2/C), even_k], geo_k)``, so the output can combine
the k-cochain with vertex quantities (densities, divergences) of the same cell; still odd times even.

``depth > 1`` (gated / context heads): further odd gated stages ``u' = z + M z`` (``M`` bias-free, zero-initialised),
``z' = u' ⊙ (1 + e'([log1p(u'^2), v, ...]))`` before ``y = W_2 z``; the head is linear at initialisation.  With
``spatial=True`` every further stage also mixes the neighbouring cells, ``u' = z + M z + N (A z)`` with the
combinatorial Hodge Laplacian ``A = d_{k-1} d_{k-1}^T + d_k^T d_k`` of degree k (odd to odd; ``N`` zero-initialised).

``skip = (o_k, o_km1)`` (gated / context heads): ``u`` also gets ``S_k f_k^odd + S_{k-1} d_{k-1} f_{k-1}^odd``, bias-free
linear maps of the odd input columns of degree k and of the coboundary of those of degree k-1 (``ctx.inputs``;
``o_k``, ``o_km1`` = ``(lo, hi)`` column ranges or ``None``): a linear path from the inputs to the output.

``wedge = h > 0`` (gated / context heads, k = 2): ``u`` also gets ``W (x_1 A ^ x_1 B)``, the face wedge products
(:func:`face_wedge`) of ``h`` pairs of projections of the final edge features (``A``, ``B``: ``C -> h``, bias-free;
``W`` zero-initialised, so the head is unchanged at initialisation).  A true 2-cochain, quadratic in the edge
features: e.g. the advection ``theta ^ delta(d theta)`` of the vorticity 2-form ``d theta`` by the velocity
1-form ``theta``, which an odd head cannot form.
```

</details>

#### `CochainReadout.forward`

```python
forward(x: list[Tensor], ctx: LayerContext) -> Tensor
```
Readout on degree `k`.

### class `EvenCellReadout`

```python
EvenCellReadout(C: int, out_dim: int, k: int, geo_dim: int) -> None
```
Bases: `Module`.

Orientation-invariant readout on degree `k`: `MLP([f(x_k), log1p(|x_k|^2/C), geo_k])`.

<details><summary>docstring</summary>

```text
``f(x_k) = log1p(x_k^2)`` (per channel) for k >= 1 and ``f(x_0) = x_0``.
```

</details>

#### `EvenCellReadout.forward`

```python
forward(x: list[Tensor], ctx: LayerContext) -> Tensor
```
`(n_k, B, C) -> (n_k, B, out_dim)`.

### class `NodeVectorReadout`

```python
NodeVectorReadout(C: int, n_fields: int, geo_dim1: int, mode: str = 'ls') -> None
```
Bases: `Module`.

Vertex vector fields from odd edge scalars (see module docstring).

<details><summary>docstring</summary>

```text
Args:
    C: channels.
    n_fields: number of vector fields ``V`` (``cfg.out_dim``); output ``(n_0, B, V * D)``.
    geo_dim1: ``G_1``.
    mode: ``'ls' | 'direct'``.
```

</details>

#### `NodeVectorReadout.forward`

```python
forward(x: list[Tensor], ctx: LayerContext) -> Tensor
```
`x[1]` `(n_1, B, C)` -> `(n_0, B, V * D)`.

### class `GradReadout`

```python
GradReadout(C: int, out_dim: int, head: str = 'mlp') -> None
```
Bases: `Module`.

`E = d_0 phi`, `phi = MLP(x_0)` (`head='mlp'`) or `phi = Lin_nobias(x_0)` (`head='linear'`, linear models such as the solver preset): odd edge cochain `(n_1, B, out_dim)` with `d_1 E = 0` exactly.

#### `GradReadout.potential`

```python
potential(x: list[Tensor], ctx: LayerContext) -> Tensor
```
The predicted vertex potential `phi` `(n_0, B, out_dim)`.

#### `GradReadout.forward`

```python
forward(x: list[Tensor], ctx: LayerContext) -> Tensor
```
`(n_1, B, out_dim)`.

### class `CurlReadout`

```python
CurlReadout(C: int, out_dim: int) -> None
```
Bases: `Module`.

`F = d_1 a`, `a = Lin_nobias(x_1)`: odd face cochain `(n_2, B, out_dim)`, closed (`d_2 F = 0`).

#### `CurlReadout.forward`

```python
forward(x: list[Tensor], ctx: LayerContext) -> Tensor
```
`(n_2, B, out_dim)`.

### class `DivReadout`

```python
DivReadout(C: int, out_dim: int, k: int) -> None
```
Bases: `Module`.

`y = d_{k-1}^T b`, `b = Lin_nobias(x_k)`: `(n_{k-1}, B, out_dim)`, co-closed (`d_{k-2}^T y = 0`); for `k = 1` a vertex divergence with zero sum per graph.

#### `DivReadout.forward`

```python
forward(x: list[Tensor], ctx: LayerContext) -> Tensor
```
`(n_{k-1}, B, out_dim)`.

### class `MassDivReadout`

```python
MassDivReadout(C: int, out_dim: int, k: int) -> None
```
Bases: `DivReadout`.

`y = S_{k-1}^{-1} d_{k-1}^T b`, `b = Lin_nobias(x_k)`, with `S` the normalised star of the output degree (`(n_{k-1}, B, out_dim)`); for `k = 1` `sum_i star0_i y_i = 0` exactly per graph (mass conservation).

#### `MassDivReadout.forward`

```python
forward(x: list[Tensor], ctx: LayerContext) -> Tensor
```
`(n_{k-1}, B, out_dim)`.

### `build_readout`

```python
build_readout(cfg: Any, geo_dims: dict[int, int]) -> nn.Module
```
Readout module selected by `cfg.readout` (`cfg.out_dim`, `cfg.vector_mode`, `cfg.C`).

<details><summary>docstring</summary>

```text
The module carries ``feature_degree`` (hidden degree it reads) and ``output_degree`` (cells of its output).
```

</details>

### `parse_readout`

```python
parse_readout(spec: str) -> tuple[str, int | None]
```
Parse a readout name.

<details><summary>docstring</summary>

```text
Args:
    spec: ``'node_scalar' | 'node_vector' | 'cochain:k' | 'even:k' | 'grad' | 'curl' | 'div' | 'div:k' | 'mdiv' |
        'mdiv:k'``.

Returns:
    ``(kind, k)``: ``k`` is the degree of the hidden features the readout reads (``None`` for ``'div'`` /
    ``'mdiv'``, meaning the top degree).
```

</details>

### `readout_degrees`

```python
readout_degrees(spec: str, top: int) -> tuple[int, int]
```
`(feature_degree, output_degree)` of a readout on a complex of top degree `top`.

<details><summary>docstring</summary>

```text
Raises:
    ValueError: if the readout does not exist on such a complex.
```

</details>

### `vector_geometry`

```python
vector_geometry(K: Any, dtype: torch.dtype, top: int) -> dict[str, Tensor | None]
```
Geometry of the vector readout, cached per complex (weakly referenced), device and dtype.

<details><summary>docstring</summary>

```text
Args:
    K: ``CochainComplex``.
    dtype: feature dtype.
    top: top degree of the complex.

Returns:
    dict with ``ev`` ``(n_1, D)`` edge vectors, ``t`` unit edge vectors, ``eet`` ``(n_1, 1, D*D)`` = ``ev ev^T``,
    ``deg`` ``(n_0, 1, 1)`` vertex valences (>= 1), ``E`` ``(n_0, 3, 2)`` tangent bases (surfaces in 3-D) or
    ``None``.
```

</details>

### `ls_solve`

```python
ls_solve(M: Tensor, b: Tensor, reg: float = 0.0001) -> Tensor
```
Damped least squares `v = (M^2 + (reg * tr M)^2 I)^{-1} M b` per cell (closed form for `d <= 3`).

<details><summary>docstring</summary>

```text
Computed in float64 with autocast disabled; zero response along null directions of ``M``.

Args:
    M: ``(..., d, d)`` symmetric positive semi-definite.
    b: ``(..., V, d)`` right-hand sides.
    reg: relative damping ``mu / tr(M)``.

Returns:
    ``(..., V, d)`` in the dtype of ``b``; zero where ``tr(M) = 0`` (isolated vertices).
```

</details>

- `rhmp.readout.vertex_mean`: re-export of `rhmp.lifting.vertex_mean` (documented there).
### `face_wedge`

```python
face_wedge(K: Any, a: Tensor, b: Tensor) -> Tensor
```
Wedge (cup) product of primal 1-cochains on the faces: `(n_1, ..., H) x (n_1, ..., H) -> (n_2, ..., H)`.

<details><summary>docstring</summary>

```text
``(a ^ b)_f = c_m sum_j (a_j b_{j+1} - a_{j+1} b_j)`` over the boundary edges of the m-gon ``f`` in cyclic order
(``K.meta['face_edges']``), with ``a_j = s_j a(e_j)`` read along the face orientation (``s_j`` =
``K.meta['face_edge_signs']``) and ``c_m = 1 / (8 sin^2(pi / m))`` (1/6 for triangles, 1/4 for quadrilaterals).
For triangles and convex quadrilaterals this is the integral over the face of the wedge of the Whitney
interpolants of two constant 1-forms, i.e. exact for constant forms.  Metric-free and antisymmetric; a true
2-cochain (sign flip with the face orientation, invariant under edge orientation flips and under relabelling).
```

</details>


## Registries (generated from the code)

### Tasks: `rhmp.tasks.load_task(name, root, ...)`

Datasets are not part of the package: paper tasks read `datasets/*.pkl` (the original pickles), the other tasks read `datasets/v2/*.pt` produced by the generators in `datasets/generators/` (see `docs/DATASETS.md`, `docs/TASK_SUITE.md`).  HP/TET names accept the suffixes `_k<kappa>` and `_aniso[<R>]` (module docstring of `rhmp.tasks`).  Every name below is accepted by `load_task` and by `python -m rhmp.train --task NAME`.

| task | source | default C | default layers | default batch |
|---|---|---:|---:|---:|
| `T1` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T1q` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T1qe` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T1qn` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T1p` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T1pm` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T1pml` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T1pf` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T1pmf` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T1pmlf` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T1g` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T1gml` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T1gml_mean` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T1gmla` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T2` | paper (`rhmp.tasks.paper`) | 64 | 4 | 64 |
| `T2v` | paper (`rhmp.tasks.paper`) | 64 | 4 | 64 |
| `T2a` | torus (`rhmp.tasks.torus`) | 64 | 4 | 64 |
| `T3` | paper (`rhmp.tasks.paper`) | 64 | 4 | 64 |
| `T3g` | paper (`rhmp.tasks.paper`) | 64 | 4 | 64 |
| `T5` | paper (`rhmp.tasks.paper`) | 160 | 4 | 64 |
| `T5g` | paper (`rhmp.tasks.paper`) | 160 | 4 | 64 |
| `T6` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T6f` | paper (`rhmp.tasks.paper`) | 128 | 4 | 64 |
| `T7` | paper (`rhmp.tasks.paper`) | 160 | 4 | 64 |
| `T7f` | paper (`rhmp.tasks.paper`) | 160 | 4 | 64 |
| `T8` | paper (`rhmp.tasks.paper`) | 256 | 4 | 8 |
| `T8v` | paper (`rhmp.tasks.paper`) | 256 | 4 | 8 |
| `T6_100K` | paper (`rhmp.tasks.paper`) | 16 | 3 | 2 |
| `HP` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `HPflux` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `HPfluxd` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `HPfluxfem` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `HPgrad` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `HP_aniso10` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `HP_aniso100` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `HPflux_aniso100` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `HP_k10` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `HP_k1000` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `TET` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `TETflux` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `TETfluxfem` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `TETgrad` | synthetic (`rhmp.tasks.synthetic`) | 128 | 4 | 8 |
| `ACURL_r10` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ACURL_r100` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ACURL_r100_n1500` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ACURL_r10_n1500` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ACURLb_r10` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ACURLb_r100` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ACURLb_r100_n1500` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ACURLb_r10_n1500` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ADARCY_r10` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ADARCY_r100` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ADARCY_r100_n1500` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ADARCY_r10_n1500` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ADARCYp_r10` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ADARCYp_r100` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ADARCYp_r100_n1500` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ADARCYp_r10_n1500` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `AHP_r10` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `AHP_r100` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ASURF_heat_r10` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ASURF_heat_r100` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ASURF_r10` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `ASURF_r100` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `DYN` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 16 |
| `DYN_cons` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 16 |
| `DYN_cons_mass` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 16 |
| `DYN_delta` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 16 |
| `DYNfix` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 64 |
| `DYNfix_cons` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 64 |
| `DYNfix_cons_mass` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 64 |
| `DYNfix_delta` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 64 |
| `HP_qual_graded` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `HP_qual_graded_ref` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `HP_qual_sliver` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `HP_qual_sliver_ref` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `SURF` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `SURF_geo` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `SURF_heat` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `SURF_heat_geo` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `SURF_heat_topo` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `SURF_topo` | extension suite (`rhmp.tasks.suite`) | 128 | 4 | 8 |
| `T8_n500` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8_n2000` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8_n8000` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8fr_n500` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8fr_n2000` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8fr_n8000` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8xy_n500` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8xy_n2000` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8xy_n8000` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8frxy_n500` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8frxy_n2000` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8frxy_n8000` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8fr` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8xy` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8frxy` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `T8i` | AirfRANS (`rhmp.tasks.airfrans`) | 256 | 4 | 8 |
| `AFV` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFV_trap` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFVp` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFVp_trap` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFU` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFUv` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFD` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFP` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFPfr` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFPxy` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFPfrxy` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFPi` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFPi0` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFPiU` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFPiR` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFPs` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFPcp` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFPcpxy` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFPpf` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFVo` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `AFVpo` | AirfRANS (`rhmp.tasks.airfrans`) | 128 | 4 | 8 |
| `MXS` | magnetostatics (`rhmp.tasks.maxwell`) | 128 | 4 | 64 |
| `MXE` | magnetostatics (`rhmp.tasks.maxwell`) | 128 | 4 | 64 |
| `MXA` | magnetostatics (`rhmp.tasks.maxwell`) | 128 | 4 | 64 |
| `MXA_noorient` | magnetostatics (`rhmp.tasks.maxwell`) | 128 | 4 | 64 |

### Models: `python -m rhmp.train --model NAME` (`rhmp.baselines.registry.SPECS`)

Details, fairness rules and applicability: `docs/BASELINES.md`.  Baselines that wrap the v1 code (`ours_v1`, the v1 graph / mesh / complex baselines) use its vendored copy in `rhmp.baselines.v1`.

| model | family | description | param-matched | keeps task output map |
|---|---|---|:-:|:-:|
| `rhmp` | rhmp | RHMP v2: learned bounded metric around the DEC star | no | yes |
| `dec_fixed` | rhmp | v2 stack with frozen H = DEC star (geometry prior, no metric learning) | no | yes |
| `unit_star` | rhmp | v2 stack, star = 1 (combinatorial prior) + learned metric | no | yes |
| `unit_fixed` | rhmp | v2 stack, star = 1 and H = 1 frozen (combinatorial Hodge Laplacians) | no | yes |
| `ours_v1` | v1_model | v1 GaugeHodgeNetwork (paper config, per-cell metric bases) | no | no |
| `mgn` | mgn | MeshGraphNet, 15 processor steps (standard), width param-matched | yes | no |
| `mgn_fast` | mgn | MeshGraphNet, 8 processor steps, width param-matched | yes | no |
| `gcn` | v1_graph | GCN (Kipf & Welling 2017), topology only | yes | no |
| `gat` | v1_graph | GAT (Velickovic et al. 2018), 4 heads, LayerNorm, residual | yes | no |
| `schnet` | v1_graph | SchNet (Schuett et al. 2017), RBF distance filters, E(n)-invariant | yes | no |
| `egnn` | v1_graph | EGNN (Satorras et al. 2021) without coordinate updates, invariant inputs | yes | no |
| `gauge_cnn` | v1_mesh | gauge-equivariant mesh CNN (Cohen et al. 2019, simplified), edge-angle channel-pair transport | yes | no |
| `gem_cnn` | v1_mesh | GEM-CNN (de Haan et al. 2021, simplified): Fourier-angle kernels + transport | yes | no |
| `mpsn` | v1_complex | MPSN (Bodnar et al. 2021, simplified): per-degree adjacency message passing | yes | no |
| `sccnn` | v1_complex | SCCNN (Yang et al. 2022): Hodge-Laplacian polynomial filters per degree | yes | no |
| `cw_net` | v1_complex | CW Network (Bodnar et al. 2021): boundary/coboundary messages | yes | no |
| `clifford_smpn` | v1_complex | Clifford simplicial MP (Liu et al. 2024, simplified), Cl(2,0) multivectors on nodes/edges | yes | no |
| `fno` | operator | FNO (Li et al. 2021) on the 32x32 grid | yes | no |
| `deeponet` | operator | DeepONet (Lu et al. 2021): branch on the mesh mean, trunk on positions | yes | no |

## Trainer CLI: `python -m rhmp.train --help`

```text
usage: python -m rhmp.train [-h] --task TASK [--native | --legacy] [--root ROOT]
                  [--out OUT] [--device DEVICE] [--epochs EPOCHS]
                  [--seed SEED] [--model MODEL] [--param-budget PARAM_BUDGET]
                  [--model-opts MODEL_OPTS] [--C C] [--layers LAYERS]
                  [--metric-type {diag,tensor}]
                  [--resolvent-iters RESOLVENT_ITERS]
                  [--resolvent-grad {implicit,unrolled}]
                  [--resolvent-physical] [--latent K:R[,K:R]]
                  [--metric-ref K:COL[,K:COL]] [--material K:M[,K:M]]
                  [--solver-mode] [--solve-iters SOLVE_ITERS]
                  [--solve-tol SOLVE_TOL] [--solve-precond {none,twolevel}]
                  [--solve-bc {dirichlet,none,neumann,lift}]
                  [--solve-picard J] [--picard-relax W]
                  [--picard-init {zero,learned}] [--no-metric-geo]
                  [--tensor-param {full,cone}] [--aux-pde W]
                  [--no-learn-metric] [--transport JSON] [--poly POLY]
                  [--log-range LOG_RANGE] [--metric-hidden METRIC_HIDDEN]
                  [--scaling {dec,jacobi,none}] [--tie | --untie] [--no-cross]
                  [--gate {norm,relu,none,homog}] [--term-gates]
                  [--cochain-head {linear,gated,context}] [--lift-context P]
                  [--lift-vertex] [--readout-depth READOUT_DEPTH]
                  [--readout-spatial] [--readout-skip] [--frozen-units]
                  [--frozen-descriptors {train,mesh,auto}]
                  [--no-frozen-descriptors] [--readout-wedge H]
                  [--readout-layers {last,all}] [--amplitude K:N[,K:N]]
                  [--amplitude-power P] [--amplitude-energy]
                  [--readout-energy H] [--readout-energy-clip C] [--no-kutta]
                  [--kutta-range KUTTA_RANGE] [--kutta-corner S]
                  [--project-solver {cg,direct,dense}]
                  [--readout-hidden READOUT_HIDDEN] [--lifting {mlp,linear}]
                  [--readout-head {mlp,linear}] [--fit-readout]
                  [--identity-metric] [--star {cotan,barycentric,unit}]
                  [--vector-mode {ls,direct}] [--connection-odd {on,off}]
                  [--no-connection] [--no-output-map] [--batch BATCH]
                  [--bs-meshes BS_MESHES] [--eval-batch EVAL_BATCH] [--lr LR]
                  [--wd WD] [--eta-min ETA_MIN] [--clip CLIP] [--amp]
                  [--compile] [--compile-backend COMPILE_BACKEND]
                  [--ckpt-layers] [--no-tf32] [--max-samples MAX_SAMPLES]
                  [--max-train-batches MAX_TRAIN_BATCHES] [--no-fine]
                  [--abs-scale] [--data-on {auto,device,cpu}] [--no-resume]
                  [--force] [--check-data] [--eval-v1 CKPT]
                  [--eval-ckpt RUN_DIR] [--v1-eval-batch V1_EVAL_BATCH]
                  [--quiet]

Generic trainer / evaluator for RHMP v2 (v1 protocol by default).

    python3 -u -m rhmp.train --task T6 --native --epochs 100 --seed 42 --out runs/t6_native
    python3 -u -m rhmp.train --task T8 --epochs 2 --out runs/smoke_t8                  # block-diagonal batches of 8
    python3 -u -m rhmp.train --task T6 --legacy --eval-v1 checkpoints_v1/T6_wilson_loop/ours/best_model.pt                              --out runs/v1_t6                                             # v1 model, same split/metrics
    python3 -u -m rhmp.train --task HP_k100 --check-data                                  # load + print the data summary

Protocol (the v1 training script ``formal_benchmark.py``): Adam(lr 1e-3, weight decay 1e-5), cosine annealing per
epoch to 1e-5 over ``--epochs``, grad-norm clipping 1.0, batch 64 (shared meshes) / 8 meshes (variable meshes), MSE
on normalised targets, model selection by validation R2 (v1 formula), test metrics of the best model: R2/MSE/MAE
(normalised), NRMSE/SSIM/Pearson (unnormalised), on the full test split, on the first 100 test samples (the v1
paper tables, ``test100_*``) and on extra test sets (``fine_*``: zero-shot 4x resolution for HP/TET).

Data are GPU resident (no DataLoader).  Shared-mesh minibatches are gathered with ``index_select`` into the
``(n, B, C)`` layout; variable meshes are batched block-diagonally (``CochainComplex.batch``), with a background
thread preparing the next batch (complexes that do not fit the GPU budget stay on the CPU and are moved per batch).

Outputs in ``--out``: ``config.json``, ``history.json`` (per epoch: losses, val metrics, lr, train/eval seconds,
peak GPU memory, metric condition numbers), ``best.pt`` / ``last.pt`` (resume: re-run the same command),
``result.json`` (test metrics, wall-clock, parameters, diagnostics).  Seeding is deterministic (python, numpy,
torch, batch order from a CPU generator); CUDA atomics make runs close but not bitwise identical.

options:
  -h, --help            show this help message and exit
  --task TASK           T1 T1q T2 T3 T5 T6 T6f T7 T7f T8 T6_100K HP* TET* (see
                        rhmp.tasks)
  --native              native cochain I/O (default)
  --legacy              v1 inputs/outputs
  --root ROOT           repository root or datasets dir (default: this
                        repository)
  --out OUT             output directory (default runs/<task>_<mode>_s<seed>)
  --device DEVICE       cuda | cpu (default: cuda if available)
  --epochs EPOCHS
  --seed SEED
  --model MODEL         model from rhmp.baselines.registry: rhmp (v2,
                        default), dec_fixed, unit_star, unit_fixed, ours_v1,
                        mgn, mgn_fast, gcn, gat, schnet, egnn, gauge_cnn,
                        gem_cnn, mpsn, sccnn, cw_net, clifford_smpn, fno,
                        deeponet (docs/BASELINES.md)
  --param-budget PARAM_BUDGET
                        parameter budget (millions) of param-matched baselines
                        (default: the v2 model's count for the same task and
                        --C/--layers)
  --model-opts MODEL_OPTS
                        JSON dict of builder options for --model, e.g.
                        '{"n_layers": 8}' or '{"hidden": 64}'
  --C C                 hidden channels (default: v1 C of the task)
  --layers LAYERS       depth (e.g. 4; default: task default) or per-layer
                        types (poly | resolvent | solve | project), e.g.
                        poly,poly,resolvent,poly or project,poly,poly,poly
  --metric-type {diag,tensor}
                        diagonal metrics (default) or the Whitney/Galerkin
                        material-tensor metric (DESIGN 9.1)
  --resolvent-iters RESOLVENT_ITERS
                        CG iterations of resolvent layers
  --resolvent-grad {implicit,unrolled}
  --resolvent-physical  resolvent layers use the un-normalised physical
                        operator (I + tau L^2 Delta_H)^-1
                        (RHMPConfig.resolvent_normalize=False; needed for
                        tensor metrics in resolvent layers)
  --latent K:R[,K:R]    r learnable per-cell input columns on degree k
                        (RHMPConfig.latent_dims), e.g. 1:8; tied to one mesh:
                        shared-mesh tasks only
  --metric-ref K:COL[,K:COL]
                        metric reference input: the (even) input column COL of
                        degree K is a fixed log offset of the metric of degree
                        K (H = star exp(ref) exp(a tanh(.))), e.g. 1:0 = log
                        sigma_e on HP/TET edges
  --material K:M[,K:M]  the last M even input columns of degree K are material
                        columns: they reach only the metric heads and
                        --metric-ref, never the lifting / gates / features
                        (RHMPConfig.material_dims), e.g. 1:1,2:1 on HP/TET
  --solver-mode         RHMPConfig.solver_preset: linear lifting, one 'solve'
                        layer (y = (Delta_H + lam)^-1 x on the free cells,
                        physical units), no gates, no cross terms, linear
                        readout (node_scalar -> cochain:0), C = --C or 4: the
                        metric is the only nonlinearity (--layers may list
                        other types)
  --solve-iters SOLVE_ITERS
                        CG iterations of solve layers (default 64)
  --solve-tol SOLVE_TOL
                        relative CG residual at which solve layers stop early
                        (default 1e-6)
  --solve-precond {none,twolevel}
                        CG preconditioner of solve layers (default none =
                        Jacobi; twolevel = Jacobi + coarse correction on
                        spatial aggregates, vertex solves)
  --solve-bc {dirichlet,none,neumann,lift}
                        solve layers: fixed boundary cells (K.boundary /
                        K.meta['dirichlet']), none, neumann (no fixed cells,
                        zero-mean vertex solution of the pure Neumann problem,
                        e.g. T5/T5g), or lift (fixed cells keep the layer
                        input: inhomogeneous Dirichlet data, e.g. electrode
                        potentials)
  --solve-picard J      fixed-point (Picard) iterations of the vertex solve of
                        solve layers with a state-dependent metric (field-
                        energy invariant of the previous iterate;
                        RHMPConfig.solve_picard; default 0)
  --picard-relax W      initial relaxation omega in (0, 1) of the Picard
                        iteration (learnable; default 0.5)
  --picard-init {zero,learned}
                        state of the first Picard iteration: zero field
                        (default) or a learnable uniform energy invariant
                        (RHMPConfig.picard_init)
  --no-metric-geo       metric heads without geometric descriptors (material,
                        feature and Picard invariants only;
                        RHMPConfig.metric_geo=False)
  --tensor-param {full,cone}
                        tensor metrics: full SPD tensors b expm(sum s t t^T)
                        (default) or the edge cone b I + sum a t t^T with a >=
                        0 (the parameterisation of runs saved without this
                        option)
  --aux-pde W           add W * model.operator_residual(u, f) (relative weak-
                        form residual of the task's PDE with the learned
                        metric, true u and f) to the loss; tasks with
                        TaskData.meta['pde'] only (HP/TET potentials), ignored
                        otherwise
  --no-learn-metric     freeze the metric at the reference (H = star exp(ref);
                        RHMPConfig.learn_metric=False): the pure DEC/FEEC
                        physics prior without metric heads
  --transport JSON      transport term of the vertices (RHMPConfig.transport,
                        rhmp.transport), e.g. '{"inputs": [0], "order": 2}'
                        (input column 0 of degree 1 as the transport field),
                        '{"latent": 2, "hidden": 1}'; keys inputs, latent,
                        hidden, forms, order, norm, metric
  --poly POLY
  --log-range LOG_RANGE
  --metric-hidden METRIC_HIDDEN
  --scaling {dec,jacobi,none}
  --tie
  --untie
  --no-cross
  --gate {norm,relu,none,homog}
                        update gate: norm (bounded radial gate, default),
                        homog (homogeneous of degree one in the field
                        amplitude), relu (ablation), none (linear)
  --term-gates          per-term gates in polynomial layers (every power L^p x
                        and cross term gets a per-cell factor from scale-
                        invariant invariants; RHMPConfig.term_gates)
  --cochain-head {linear,gated,context}
                        head of cochain:k readouts: linear (default), gated
                        (odd linear map x even gate) or context (gated, with
                        the lower-degree features of the cell)
  --lift-context P      lifting gates of degrees >= 1 also see the vertex mean
                        of a P-dim projection of the lifted vertex features
                        (RHMPConfig.lift_context; default 0)
  --lift-vertex         the vertex lifting also sees the codifferential of the
                        odd edge inputs and vertex means of edge-input
                        invariants (RHMPConfig.lift_vertex)
  --readout-depth READOUT_DEPTH
                        odd gated stages of the gated / context cochain heads
                        (default 1)
  --readout-spatial     later stages of a deep cochain head (--readout-depth
                        >= 2) also mix neighbouring cells through the
                        combinatorial Hodge Laplacian
                        (RHMPConfig.readout_spatial)
  --readout-skip        gated / context cochain heads also read the odd inputs
                        of their degree and the coboundary of the odd inputs
                        one degree below (linear input skip;
                        RHMPConfig.readout_skip)
  --frozen-units        per-mesh geometric normalisers (star means, descriptor
                        medians, block bounds) of the training mesh, stored in
                        the model and used on every mesh
                        (RHMPConfig.frozen_units)
  --frozen-descriptors {train,mesh,auto}
                        with --frozen-units: relative descriptors in training-
                        mesh units ('train', the default), relative to each
                        mesh ('mesh': only star means and block bounds frozen)
                        or per column 'auto' (training units while the mesh's
                        median lies in the training range;
                        RHMPConfig.frozen_descriptors True / False / 'auto')
  --no-frozen-descriptors
                        same as --frozen-descriptors mesh
  --readout-wedge H     gated / context head of a cochain:2 readout also reads
                        H face wedge products of projections of the final edge
                        features (RHMPConfig.readout_wedge; default 0)
  --readout-layers {last,all}
                        cochain readouts read the features after the lifting
                        and after every layer (all) or after the last layer
                        only (last, default)
  --amplitude K:N[,K:N]
                        the first N input columns of degree K scale with one
                        field amplitude: they are divided by their per-sample
                        RMS and the output is multiplied by it
                        (RHMPConfig.amplitude_dims), e.g. 1:2 on T1p (velocity
                        1-form and its even companion)
  --amplitude-power P   the output is multiplied by s^P (default 1; 2:
                        homogeneous of degree two in the amplitude columns,
                        e.g. pressure for a velocity 1-form;
                        RHMPConfig.amplitude_power)
  --amplitude-energy    the output factor s is the field magnitude of the
                        amplitude 1-form (Galerkin energy density over the
                        domain, mesh independent) instead of the RMS cochain
                        value (RHMPConfig.amplitude_energy)
  --readout-energy H    node_scalar readout with H Galerkin energy densities
                        of projections of the edge features at the vertices
                        (RHMPConfig.readout_energy; default 0)
  --readout-energy-clip C
                        bound the energy channels: q -> C tanh(q / C)
                        (RHMPConfig.readout_energy_clip; default off)
  --no-kutta            'project' layers without the harmonic (Kutta)
                        correction (RHMPConfig.project_harmonic=False)
  --kutta-range KUTTA_RANGE
                        bound of the log Kutta weights (default 8)
  --kutta-corner S      corner prior of the Kutta weights with initial slope S
                        (RHMPConfig.kutta_corner; default off)
  --project-solver {cg,direct,dense}
                        solver of 'project' layers: cg (default), direct
                        (sparse LU on the CPU, diagonal metrics) or dense
                        (batched Cholesky on the device, diagonal and tensor
                        metrics, one graph per batch)
  --readout-hidden READOUT_HIDDEN
                        hidden width of the gated cochain head (16)
  --lifting {mlp,linear}
                        lifting of the field inputs (default mlp; linear: x_0
                        = W f_0, no geometry or even gates)
  --readout-head {mlp,linear}
                        potential head of the grad readout (default mlp;
                        linear: phi = Lin_nobias(x_0))
  --fit-readout         initialise a linear readout by least squares on the
                        first training batch (as --solver-mode does)
  --identity-metric
  --star {cotan,barycentric,unit}
  --vector-mode {ls,direct}
                        default: task specific
  --connection-odd {on,off}
                        connection columns also enter the odd path (default:
                        on for SU(2) T7/T7f, off otherwise)
  --no-connection       treat native connection inputs as plain odd inputs
  --no-output-map       disable the task's orientation output map (native
                        T1/T3/T6/T7): the model then predicts the pseudo-
                        scalar/vector target directly (not representable by an
                        equivariant model)
  --batch BATCH         samples per step on shared meshes (default 64)
  --bs-meshes BS_MESHES
                        meshes per block-diagonal batch (default 8)
  --eval-batch EVAL_BATCH
  --lr LR
  --wd WD
  --eta-min ETA_MIN
  --clip CLIP
  --amp                 bf16 autocast for the dense parts (sparse products
                        fp32)
  --compile             torch.compile the model
  --compile-backend COMPILE_BACKEND
                        torch.compile backend (inductor needs the Python
                        headers to build Triton's CUDA helpers; 'aot_eager'
                        works everywhere but gives no speed-up)
  --ckpt-layers         activation checkpointing per layer
  --no-tf32             disable TF32 dense matmuls on CUDA
  --max-samples MAX_SAMPLES
                        truncate the data set (smoke tests)
  --max-train-batches MAX_TRAIN_BATCHES
                        limit steps per epoch (smoke tests)
  --no-fine             skip the resolution-transfer test set (HP/TET)
  --abs-scale           append log(median edge length) as a constant even
                        vertex input (absolute mesh scale; the model is scale-
                        free by design)
  --data-on {auto,device,cpu}
                        where variable-mesh complexes live (auto: GPU if the
                        estimate is below rhmp.tasks.synthetic.GPU_BUDGET_GB,
                        env RHMP_GPU_BUDGET_GB)
  --no-resume           ignore an existing last.pt
  --force               re-run even if result.json exists
  --check-data          load the task, print its summary and exit
  --eval-v1 CKPT        evaluate a v1 checkpoint (legacy data)
  --eval-ckpt RUN_DIR   evaluate the best.pt of a finished v2 run on --task
                        (transfer/robustness sweeps); inputs and targets are
                        re-normalised with the statistics of the training run
  --v1-eval-batch V1_EVAL_BATCH
                        batch size for the v1 model (1 = per sample as
                        compute_all_metrics.py; v1 couples batches)
  --quiet
```

