# Design Document: Synthetic Seismic Generator, Unified HDF5 I/O with Explicit Normalization, and 1D/2D Quantum FWI via VQLS

**Date:** 2026-10-29
**Status:** Draft (Pending User Approval)
**Package:** `qpdesolver`

---

## 1. Intent and Why

`qpdesolver` currently implements gate-based quantum linear solvers (`HHLSolver`, `VQLSSolver`) and a 1D frequency-domain Helmholtz Full Waveform Inversion (`FWIProblem`, `FWIInversion`) backed by a group-based HDF5 schema (`src/qpdesolver/io/hdf5.py`). Separately, `src/qpdesolver/generator/` contains an untracked Devito utility (`devito_hdf5.py`), a tutorial notebook (`model_1_hdf5.ipynb`), and a 2D time-domain acoustic sample (`model_1.h5`, 101x101 grid, two layers at 1.5 and 2.5 km/s, 410 time steps).

The goal of this work is to unify these two worlds cleanly so researchers can:
1. Parameterize and generate new synthetic 2D seismic samples reproducibly through `qpdesolver.generator` (using Devito when installed, plus a pure-NumPy 2nd-order finite-difference fallback).
2. Read, write, inspect, and convert both `qpdesolver` native HDF5 datasets and `devito-seismic-hdf5` files through an optimized `qpdesolver.io` layer with **explicit, auditable unit/scale normalization** (critical for controlling matrix conditioning $\kappa(A)$ in variational quantum circuits).
3. Solve acoustic FWI in **both 1D and 2D** using `VQLSSolver` (and `ClassicalSolver` / `HHLSolver`), demonstrated end-to-end on `model_1.h5`.
4. Maintain strict repository hygiene: `.h5` files are never committed, commits are organized into isolated, well-documented units tracked with **Beads** (`bd`), and **no commit is ever executed without explicit user authorization**.

---

## 2. Goals, Non-Goals, and Anti-Goals

### Goals
- **Task 1 (`generator`):** Structure `src/qpdesolver/generator/` as a modular subpackage supporting reusable velocity model families (two-layer, multi-layer, Gaussian anomaly, salt/box inclusion), customizable source/receiver acquisition geometries, in-loop `SnapshotRecorder`, and dual simulation backends (`devito` primary + `numpy` 2D damped acoustic FD fallback). Keep backward-compatible re-exports in `qpdesolver.generator.devito_hdf5`.
- **Task 2 (`io`):** Optimize `src/qpdesolver/io/` using patterns from `model_1_hdf5.ipynb` and `devito_hdf5.py`: chunked compression (`gzip`/`lzf` + `shuffle`), optional `dtype` casting, lazy/sliced wavefield loading (`include_fields`, `snapshot_indices`), structured `summarize_hdf5`, and bidirectional interoperability between `devito-seismic-hdf5` and `qpdesolver` FWI models (1D and 2D). Expose an explicit normalization interface (`normalize: bool = False`, `unit_target: Optional[Literal["si", "kms", "dimensionless"]] = None`) that records exact scaling factors in `metadata["normalization"]`.
- **Task 3 (Git, Environment & Beads):**
  - Add `*.h5` and `*.hdf5` to `.gitignore`; untrack existing tracked `.h5` files (`git rm --cached`, keeping local files intact) when staging; move `model_1.h5` to `examples/data/model_1.h5` and `model_1_hdf5.ipynb` to `examples/notebooks/model_1_hdf5.ipynb`.
  - Provision `.venv` based on `/Users/micaelverissimo/Documents/pos-doc-activities/07-Oct-2026/.venv` and install `devito` while keeping `numpy<2.0` and `pennylane>=0.38.0,<0.39.0` intact.
  - Install and initialize **Beads** (`bd`) to manage independent, non-interfering tasks mapped to atomic commit boundaries, waiting for explicit user authorization before running any `git commit`.
- **Task 4 (1D/2D FWI + VQLS Example on `model_1.h5`):**
  - Extend `qpdesolver` with `Helmholtz2D` ($L_x \otimes I + I \otimes L_z - \omega^2 \operatorname{diag}(m)$ on $2^{n_x} \times 2^{n_z}$ grids), `VelocityModel2D`, `Acquisition2D`, `FWIDataset2D`, `FWIProblem2D`, generalized `FWIInversion`, and variational parameter warm-starting across FWI iterations.
  - Deliver `examples/11_fwi_vqls_model1.py` solving FWI with `VQLSSolver` on `model_1.h5` in both 1D (central depth profile) and 2D (subsampled $4 \times 4$ = 4-qubit grid), comparing against `ClassicalSolver`.

### Non-Goals
- 3D quantum Helmholtz/FWI solvers (statevector qubit count $(n_x + n_y + n_z)$ is outside target NISQ demonstration scope, though `export_devito_hdf5` and `SnapshotRecorder` preserve 2D/3D Devito support).
- Elastic, TTI, or viscoelastic adjoint FWI engines (inversion targets isotropic acoustic velocity $v(x,z)$; HDF5 I/O continues to support vector/tensor Devito fields).
- Time-domain quantum ODE time-marching for FWI.

### Anti-Goals
- **Zero `.h5` in commits:** Staging or committing any `.h5`/`.hdf5` artifact under any circumstance is a failure.
- **Zero unauthorized commits:** Executing `git commit` before the user explicitly authorizes it in chat is prohibited.
- **Zero silent normalization:** Modifying physical units or scaling arrays in `qpdesolver.io` when `normalize=False` is prohibited.
- **Zero regression on existing suite:** Breaking any existing 1D FWI, HHL, VQLS, trainability, or `17-Jul-2026` parity tests is prohibited.

---

## 3. Constraints

1. **Git Policy:** `.gitignore` must ignore `*.h5` and `*.hdf5`. Every proposed commit boundary is prepared and verified via `git status`, but `git commit` is held until explicit user approval.
2. **Python & Dependency Compatibility:**
   - Reference environment: `/Users/micaelverissimo/Documents/pos-doc-activities/07-Oct-2026/.venv` (`pennylane==0.38.0`, `pennylane-lightning==0.38.0`, `autoray==0.6.12`, `numpy==1.26.4`, `scipy==1.13.1`, `h5py==3.14.0`, `pytest==8.4.2`).
   - Workspace `.venv` must preserve `pennylane>=0.38.0,<0.39.0`, `autoray<0.7`, and `numpy>=1.24.0,<2.0.0`, while also installing `devito` (declared under optional dependency `[project.optional-dependencies] generator = ["devito>=4.8.0"]` in `pyproject.toml`, installed into `.venv` constrained by `numpy<2.0`). If the latest `devito` release requires Python >=3.10 while `07-Oct-2026/.venv` uses Python 3.9, `.venv` will be created via `uv venv --python 3.11` with the exact pinned versions from `07-Oct-2026/.venv` plus `devito` constrained to `numpy<2.0`.
3. **Quantum Register Sizing:** `LinearSystem` enforces $N = 2^n$. Consequently, `Helmholtz1D` requires $N_z = 2^{n_z}$ ($n_z \ge 1$) and `Helmholtz2D` requires $N_x = 2^{n_x}, N_z = 2^{n_z}$ ($n_x \ge 1, n_z \ge 1$) with total dimension $N = N_x N_z = 2^{n_x + n_z}$.
4. **Task Management via Beads (`bd`):** `beads` CLI (`bd`) is installed via Homebrew (`brew install beads`) and initialized so tasks and their dependency graph are tracked cleanly without cross-task interference. (Note: `.beads/` state will also remain uncommitted unless the user authorizes committing it or prefers ignoring `.beads/` in `.gitignore` — see Builder Calls).

---

## 4. Approaches Considered

1. **Approach A — Dual-Engine Generator + Unified I/O with Explicit Normalization + 1D/2D FWI with VQLS Warm-Start (Selected):**
   - Modularizes `generator` with Devito primary backend (installed in `.venv`) + NumPy 2D acoustic FD fallback; unifies `io` by upgrading `qpdesolver` group persistence and integrating `devito-seismic-hdf5` I/O + explicit normalization flags; extends `problems` and `applications/fwi` to 2D while keeping 1D intact.
   - *Trade-off:* Slightly broader engineering surface, but guarantees full Devito + PennyLane coexistence, deterministic unit tests even in minimal CI environments without C compilers, and clean 1D + 2D VQLS convergence.
2. **Approach B — Devito-Only Generator + Read-Only I/O Importer:**
   - Wraps Devito tutorials directly and adds a one-way importer to `io`.
   - *Rejected because:* Lacks write/read symmetry and chunked optimization in `qpdesolver.io`, and tightly couples all generator tests to JIT C compilation.
3. **Approach C — Replace `qpdesolver.io` Schema Globally with `devito-seismic-hdf5`:**
   - Drops `qpdesolver_type` groups in favor of the notebook's root-level HDF5 layout everywhere.
   - *Rejected because:* Breaks backward compatibility with `LinearSystem` CSR storage, trainability study results, existing examples, and unit tests.

---

## 5. Architecture and Component Specifications

### 5.1 Repository & Data Layout
- Move `src/qpdesolver/generator/model_1.h5` $\to$ `examples/data/model_1.h5` (ignored by Git via `*.h5`).
- Move `src/qpdesolver/generator/model_1_hdf5.ipynb` $\to$ `examples/notebooks/model_1_hdf5.ipynb` (updated to import from `qpdesolver.generator` and `qpdesolver.io`, writing output to `examples/data/`).
- Update `.gitignore` with `*.h5`, `*.hdf5`, and `.beads/` (so local Beads task tracking never pollutes code commits).

### 5.2 `src/qpdesolver/io/` — Optimized HDF5 I/O & Explicit Normalization
Files:
- `src/qpdesolver/io/schema.py`: Bump `SCHEMA_VERSION = "1.1.0"` (backward-compatible reader for `"1.0.0"`), add `"velocity_model_2d"` and `"fwi_dataset_2d"` to `SUPPORTED_TYPES`, and define `DEVITO_FORMAT = "devito-seismic-hdf5"`.
- `src/qpdesolver/io/normalization.py`:
  - `UnitTarget = Literal["si", "kms", "dimensionless"]`
  - `@dataclass(frozen=True) class NormalizationParams`:
    - `normalized: bool`
    - `source_units: str` (`"kms"` for Devito files where `vp` is km/s, `spacing` is m, `time` is ms; `"si"` when `vp` is m/s, `spacing` is m, `time` is s)
    - `target_units: str` (`"raw"`, `"si"`, `"kms"`, or `"dimensionless"`)
    - `velocity_scale: float` ($v_{\text{out}} = v_{\text{in}} \cdot \text{velocity\_scale}$)
    - `length_scale: float` ($h_{\text{out}} = h_{\text{in}} \cdot \text{length\_scale}$)
    - `time_scale: float` ($t_{\text{out}} = t_{\text{in}} \cdot \text{time\_scale}$)
    - `omega_scale: float` ($\omega_{\text{out}} = \omega_{\text{rad/s}} \cdot \text{omega\_scale}$)
    - `to_dict() -> Dict[str, Any]` and `denormalize_velocity(v: np.ndarray) -> np.ndarray` ($v / \text{velocity\_scale}$ or $v \cdot v_{\text{orig}}$).
  - Explicit contract on every loader/converter:
    - Signature includes `*, normalize: bool = False, unit_target: Optional[UnitTarget] = None, source_units: Literal["auto", "kms", "si"] = "auto"`.
    - If `normalize=False` and `unit_target is not None`: raise `ValueError("unit_target requires normalize=True")`.
    - If `normalize=False`: arrays and grid spacings are returned **verbatim** (`velocity_scale=1.0, length_scale=1.0`), `metadata["normalization"] = {"normalized": False, "target_units": "raw", ...}`.
    - If `normalize=True`: `unit_target` defaults to `"si"` if `None`:
      - `"si"`: velocity in `m/s` (`km/s * 1000`), spacing `h` in `m`, time in `s` (`ms * 1e-3`), angular frequency $\omega = 2\pi f_{\text{Hz}}$ in `rad/s`.
      - `"kms"`: velocity in `km/s`, spacing `h` in `km` (`m * 1e-3`), time in `s` (`ms * 1e-3`), $\omega = 2\pi f_{\text{Hz}}$ in `rad/s` (physically consistent since $(\text{km})^{-2}$ matches $(\text{rad/s})^2 / (\text{km/s})^2$).
      - `"dimensionless"`: first converts to consistent physical units (`si`), then scales by reference length $L_0 = h_{\text{eff}}$ (mean resampled grid spacing, so $\tilde{h}_x = h_x / L_0 = O(1)$) and reference velocity $v_0 = \operatorname{mean}(v_{\text{ref}})$ (using `initial_velocity` if present, else `velocity`), giving $\tilde{v} = v / v_0 = O(1)$ and $\tilde{\omega} = \omega L_0 / v_0$. This keeps Helmholtz diagonal entries $2/\tilde{h}^2 - \tilde{\omega}^2/\tilde{v}^2 = O(1)$, avoiding ill-conditioned matrices in VQLS.
- `src/qpdesolver/io/devito_io.py`:
  - `export_devito_hdf5(...)`: Evolves `devito_hdf5.export_hdf5` with snapshot-aligned chunking (`chunks=(1, *space_shape)` for `(nsnaps, *space)` datasets when `compression` is enabled), `shuffle=True` for `gzip`, explicit `@units` attributes (`"km/s"` on `vp`, `"m"` on `coords` and `spacing`, `"ms"` on `time`), `dtype` casting (`"float32"` default or `None`), and `strip_nbl=True`.
  - `load_devito_hdf5(path, *, include_fields: bool = True, snapshot_indices: Optional[Sequence[int]] = None, normalize: bool = False, unit_target: Optional[UnitTarget] = None, source_units: str = "auto") -> Dict[str, Any]`: Reads `devito-seismic-hdf5` without loading full 3D/4D wavefield stacks when `include_fields=False` or when `snapshot_indices` selects specific time slices.
  - `summarize_hdf5(path, *, show_attrs: bool = True, print_output: bool = True) -> str`: Works on both `devito-seismic-hdf5` and `qpdesolver` HDF5 files.
  - `extract_velocity_1d_from_devito(...) -> VelocityModel`: Extracts a vertical profile at `x_index` (defaults to source $x$-coordinate index if `geometry/src` exists, else midpoint $N_x // 2$), optionally resamples to `resample_to` ($2^n$), and applies explicit normalization if `normalize=True`.
  - `extract_velocity_2d_from_devito(...) -> VelocityModel2D`: Extracts the 2D `model/vp` array `(nx, nz)`, optionally resamples/smooths to `resample_shape=(target_nx, target_nz)`, and applies explicit normalization if `normalize=True`.
  - `extract_frequency_data_from_devito(path, frequencies_hz: Sequence[float], *, deconvolve_source: bool = True, reg_eps: float = 1e-6) -> Dict[str, Any]`: Applies `np.fft.rfft` along the time axis of `geometry/rec/data` (and `geometry/src/data`), interpolates complex spectra at `frequencies_hz`, and optionally deconvolves the Ricker wavelet spectrum $S(\omega) / (|S(\omega)|^2 + \epsilon)$ to match the unit-impulse RHS of Helmholtz.
  - `build_fwi_dataset_from_devito(...) -> Union[FWIDataset, FWIDataset2D]`: High-level builder creating a ready-to-invert 1D or 2D FWI dataset from a Devito `.h5` file:
    - Extracts `true_velocity` from `model/vp` onto the target power-of-2 grid; constructs `initial_velocity` via Gaussian smoothing (`initial_smoothing_sigma`) or constant background mean of top layer.
    - Maps physical source/receiver coordinates from `geometry/` onto nearest discrete indices of the resampled grid (deduplicating/clamping interior indices when `bc="dirichlet"`).
    - Supports `data_source="helmholtz_consistent"` (simulates exact Helmholtz observed data on the target coarse grid from the Devito-extracted `true_velocity` so coarse-grid quantum FWI is free of time-vs-coarse-space dispersion mismatch) **and** `data_source="devito_fft"` (uses `extract_frequency_data_from_devito` mapped onto target receivers).
- `src/qpdesolver/io/hdf5.py`:
  - Adds optional `compression: Optional[str] = "gzip"`, `compression_opts: int = 4`, `dtype: Optional[str] = None` to `save_linear_system`, `save_velocity_model`, `save_fwi_dataset`, `save_results`.
  - Adds `save_velocity_model_2d`, `load_velocity_model_2d`, `save_fwi_dataset_2d`, `load_fwi_dataset_2d`.
  - Adds `normalize` / `unit_target` support to `load_velocity_model*` and `load_fwi_dataset*`.

### 5.3 `src/qpdesolver/generator/` — Modular Synthetic Seismic Generator
Files:
- `src/qpdesolver/generator/models.py`:
  - Pure-NumPy velocity array builders (`float32`, `km/s` default):
    - `two_layer_velocity(shape=(101, 101), v_top=1.5, v_bottom=2.5, interface_z=0.5)`
    - `multi_layer_velocity(shape=(101, 101), velocities=(1.5, 2.0, 2.8), interfaces=(0.35, 0.70))`
    - `gaussian_anomaly_velocity(shape=(101, 101), v_bg=2.0, dv=0.5, center=(0.5, 0.5), sigma=(0.15, 0.15))`
    - `salt_inclusion_velocity(shape=(101, 101), v_top=1.6, v_bottom=2.6, v_salt=4.0, bounds_x=(0.3, 0.7), bounds_z=(0.45, 0.75))`
- `src/qpdesolver/generator/recorder.py`:
  - Houses `SnapshotRecorder`, `flatten_fields`, `strip_absorbing_layer` from `devito_hdf5.py`.
- `src/qpdesolver/generator/acoustic.py`:
  - `SeismicSurveySpec` dataclass (grid `shape`, `spacing`, `origin`, `nbl=10`, `space_order=2`, `t0=0.0`, `tn=1000.0`, `f0=0.010` kHz, `source_positions`, `receiver_positions`, `nsnaps=50`).
  - `generate_seismic_sample(output_path: Optional[Union[str, Path]], vp: np.ndarray, spec: SeismicSurveySpec, *, backend: Literal["auto", "devito", "numpy"] = "auto", attrs: Optional[Dict[str, Any]] = None, overwrite: bool = True) -> Dict[str, Any]`:
    - When `backend="devito"` (or `"auto"` with `devito` importable): builds Devito `Model`, `TimeAxis`, `RickerSource`, `Receiver`, `SnapshotRecorder`, and `Operator`, runs the time loop, and writes `output_path` via `export_devito_hdf5` (if `output_path` is provided) while returning the in-memory dictionary.
    - When `backend="numpy"`: runs an equivalent 2nd-order space / 2nd-order time damped acoustic finite-difference time loop in pure NumPy (CFL `critical_dt`, quadratic sponge damping layer `nbl`, Ricker wavelet injection, bilinear receiver interpolation, `nsnaps` strided recording) and writes the exact same `devito-seismic-hdf5` structure.
- `src/qpdesolver/generator/devito_hdf5.py`:
  - Thin compatibility re-export module preserving `SnapshotRecorder`, `export_hdf5`, `load_hdf5`, `summarize_hdf5`, `flatten_fields`, `strip_absorbing_layer`.

### 5.4 2D Helmholtz Discretization, 2D FWI & VQLS Warm-Starting
Files:
- `src/qpdesolver/problems/discretizations/helmholtz2d.py`:
  - `@REGISTRY_DISCRETIZATIONS.register("helmholtz2d") class Helmholtz2D(PDEDiscretization)`:
    - `__init__(self, nx: int, nz: int, hx: float = 1.0, hz: Optional[float] = None, bc: str = "dirichlet")`: validates both `nx` and `nz` are positive powers of 2; calls `super().__init__(n_points=nx * nz, h=hx, bc=bc)` and stores `nx, nz, hx, hz = hz or hx`.
    - `laplacian() -> np.ndarray`: $L_{2D} = \operatorname{kron}(L_x, I_{n_z}) + \operatorname{kron}(I_{n_x}, L_z)$ using 1D finite-difference Laplacians with `dirichlet`, `neumann`, or `periodic` BCs.
    - `operator(m=None, omega=1.0, velocity=None) -> np.ndarray`: accepts 2D `(nx, nz)` or flat `(nx*nz,)` slowness-squared `m` or `velocity`, returning complex Hermitian matrix $A = L_{2D} - \omega^2 \operatorname{diag}(m_{\text{flat}})$ of shape `(nx*nz, nx*nz)`.
    - `point_source_2d(ix: int, iz: int) -> np.ndarray`: delta source at flat index `ix * nz + iz`.
- `src/qpdesolver/applications/fwi/model.py`:
  - `VelocityModel2D`: 2D velocity `(nx, nz)`, `hx`, `hz`, `bc`, `metadata`, `.slowness_squared`, `.flat_velocity`, and `.resample(target_shape: Tuple[int, int], smooth_sigma: float = 0.0) -> VelocityModel2D` (optional Gaussian anti-aliasing filter before bilinear interpolation onto `(target_nx, target_nz)` preserving physical domain dimensions).
  - `Acquisition2D`: `sources: List[Tuple[int, int]]`, `receivers: List[Tuple[int, int]]`, `omegas: List[float]`.
  - `FWIDataset2D`: `acquisition: Acquisition2D`, `observed_data: np.ndarray` `(n_sources, n_omegas, n_receivers)`, `hx`, `hz`, `bc`, `true_velocity` `(nx, nz)`, `initial_velocity` `(nx, nz)`, `metadata`.
- `src/qpdesolver/applications/fwi/problem.py`:
  - Add optional `warm_start_solves: bool = False` to both `FWIProblem` and new `FWIProblem2D`:
    - When `warm_start_solves=True` and `self.solver` accepts `warm_start` (such as `VQLSSolver`), caches `res.metadata["optimal_params"]` per `(role, src_key, omega)` key and passes it on subsequent calls, accelerating variational convergence across FWI gradient evaluations.
  - `FWIProblem2D`: implements `solve_forward`, `solve_adjoint`, `forward_all`, `misfit`, `gradient_m` (returns `(nx, nz)` array), `gradient_v` (returns `(nx, nz)` array), and `simulate_dataset` using exact discrete adjoint-state calculus on `Helmholtz2D`:
    $$\frac{\partial J}{\partial m_{i_x, i_z}} = \sum_{s, \omega} \omega^2 \operatorname{Re}\!\left(\lambda_{s,\omega,(i_x,i_z)}^* \, u_{s,\omega,(i_x,i_z)}\right), \qquad \frac{\partial J}{\partial v} = -\frac{2}{v^3} \frac{\partial J}{\partial m}.$$
- `src/qpdesolver/applications/fwi/inversion.py`:
  - Extend `FWIObjective` and `FWIInversion` to seamlessly support both 1D (`FWIProblem` + `FWIDataset` + `VelocityModel`) and 2D (`FWIProblem2D` + `FWIDataset2D` + `VelocityModel2D`) by flattening optimization vector `x` to 1D for `Optimizer.minimize` and reshaping back to model shape when constructing `VelocityModel` / `VelocityModel2D`.
  - Note on `bounds`: adjust default `bounds` in `FWIInversion` to `Optional[Tuple[float, float]] = None` ONLY when normalized? **Important backward-compatibility rule:** Keep existing default `bounds=(500.0, 5000.0)` when `dataset` is unnormalized SI, or allow explicit `bounds=(...)` passed by caller (existing tests pass explicit `bounds` or SI velocities around `2000.0`, so keeping default `(500.0, 5000.0)` preserves 100% backward compatibility while callers using `"kms"` or `"dimensionless"` pass appropriate `bounds`, e.g., `(0.5, 3.0)`!).

### 5.5 Example: `examples/11_fwi_vqls_model1.py`
A standalone, reproducible example script that:
1. Locates `examples/data/model_1.h5` (and if absent on a fresh clone where `.h5` is gitignored, automatically generates it into `examples/data/model_1.h5` via `qpdesolver.generator`!).
2. Prints the HDF5 summary (`summarize_hdf5`) and demonstrates the explicit normalization flag (`normalize=True, unit_target="dimensionless"`), logging `v0`, `L0`, and condition number $\kappa(A)$ improvement.
3. **1D Quantum FWI (`N=4`, 2 qubits):** Extracts the central vertical velocity profile of `model_1.h5` (`1.5 -> 2.5 km/s`), builds a smoothed initial guess, and runs `FWIInversion` with `VQLSSolver(ansatz=RealAmplitudes(layers=3), optimizer=ScipyOptimizer("BFGS"), max_iter=80)` (with `warm_start_solves=True`) alongside a `ClassicalSolver` baseline. Reports initial/final misfit, wavefield fidelity, and denormalized velocity profile in `km/s`.
4. **2D Quantum FWI (`4 x 4 = 16`, 4 qubits):** Extracts and downsamples the 2D `model_1.h5` velocity model onto a $4 \times 4$ grid, configures surface sources/receivers, and runs 2D `FWIInversion` with `VQLSSolver` (warm-started) vs `ClassicalSolver`. Denormalizes the recovered 2D velocity back to `km/s`, logs misfit reduction and velocity RMSE, and optionally saves a comparison figure (`examples/data/model1_vqls_fwi.png`, or `reports/figures/`).

---

## 6. What Is Left to the Builder

- Exact choice of internal helper names, sponge damping profile coefficients in the NumPy fallback propagator (matching Devito's `initdamp` closely enough for stable absorbing boundaries), and `scipy.ndimage.gaussian_filter` boundary mode (`mode="nearest"`) during downsampling.
- Specific VQLS ansatz depth / max iterations in `examples/11_fwi_vqls_model1.py` tuned so the script finishes comfortably fast (< 30–60s on CPU) while achieving > 99% single-solve state fidelity and clear FWI misfit reduction in both 1D and 2D.
- Granular Beads (`bd`) issue breakdown and commit message drafting (strictly unstaged/uncommitted until user authorization).

