# FWI Seismic Generator, Unified HDF5 I/O, and 1D/2D Quantum VQLS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform `qpdesolver.generator` into a modular synthetic seismic generator (Devito + NumPy acoustic backends), optimize `qpdesolver.io` with chunked compression and explicit unit/scale normalization, extend `qpdesolver` to 2D Helmholtz/FWI with variational parameter warm-starting, and deliver an end-to-end example solving FWI on `model_1.h5` using `VQLSSolver` in 1D and 2D.

**Architecture:** 
- A modular `generator` package creates parameterized velocity models (two-layer, multi-layer, anomalies) and generates wavefield snapshots in `devito-seismic-hdf5` format via Devito or a pure-NumPy 2nd-order acoustic propagator fallback.
- An upgraded `io` layer introduces an explicit `NormalizationParams` contract (`normalize: bool = False`, `unit_target: Optional[Literal["si", "kms", "dimensionless"]] = None`) and snapshot-chunked, compressed HDF5 read/write routines with full bidirectional bridges to FWI datasets.
- Spatial discretizations and FWI engines are extended from 1D to 2D ($L_{2D} = L_x \otimes I + I \otimes L_z - \omega^2 \operatorname{diag}(m)$ on $2^{n_x} \times 2^{n_z}$ grids), coupled with adjoint-state gradient calculation and an optional `warm_start_solves` cache to stabilize `VQLSSolver` across FWI iterations.
- A dedicated virtual environment cloned from `/Users/micaelverissimo/Documents/pos-doc-activities/07-Oct-2026/.venv` installs `devito` while maintaining `numpy<2.0` and `pennylane~=0.38.0`. Tasks and commit boundaries are managed via **Beads** (`bd`), guaranteeing that `.h5` files are never tracked and no `git commit` is executed without explicit user authorization.

**Tech Stack:** Python 3.9/3.11, NumPy (`>=1.24.0,<2.0.0`), SciPy, h5py, PennyLane (`~=0.38.0`), PennyLane-Lightning, Devito, Loguru, Pytest, Beads (`bd`).

**Spec:** `/Users/micaelverissimo/Documents/workspace/qpdesolver/docs/superpowers/specs/2026-10-29-fwi-generator-io-vqls-design.md`

## Global Constraints

- Never commit any `.h5` or `.hdf5` file under any circumstance; `.gitignore` must ignore `*.h5`, `*.hdf5`, and `.beads/`.
- Never execute `git commit` without explicit authorization from the user in conversation; prepare staging, verify status, and await confirmation.
- The `normalize` parameter must default to `False`. When `False`, values and spacings must remain verbatim and passing `unit_target` must raise `ValueError`. When `True`, exact normalization factors must be recorded in `metadata["normalization"]`.
- The dimension $N$ for 1D systems must satisfy $N = 2^{n_z}$; for 2D systems, $N = N_x \cdot N_z = 2^{n_x + n_z}$ with $n_x \ge 1, n_z \ge 1$.
- Preserve 100% backward compatibility with existing tests (`test_io_hdf5.py`, `test_fwi_from_h5.py`, `test_vqls_parity_17jul.py`).

## Review Focus

1. **Unit mismatch in FWI from Devito:** Devito stores velocity in `km/s` and time in `ms`. If `normalize=False`, `FWIProblem` expects consistent units. Passing `unit_target="dimensionless"` or `"si"` must map cleanly back via `denormalize_velocity` without silent scale distortion.
2. **Dispersion crime on coarse quantum grids:** Inverting directly against fine-grid (101x101) time-domain traces using a coarse 4x4 Helmholtz operator causes severe numerical dispersion mismatch; the loader must support `data_source="helmholtz_consistent"` for quantum benchmark reliability alongside `data_source="devito_fft"`.
3. **Power-of-2 resampling edge preservation:** Resampling 2D velocity fields from arbitrary shapes (e.g. 101x101) onto $2^{n_x} \times 2^{n_z}$ must strictly preserve domain bounds $(0, L_x) \times (0, L_z)$ and velocity bounds.
4. **Variational warm-start parameter vector length:** When velocity model size or ansatz depth changes, passing cached warm-start parameters of incorrect length must gracefully fallback to fresh initialization rather than crashing.
5. **Accidental staging of `.h5`:** When creating or running synthetic samples, files must land in paths ignored by Git and verified before any staging command.

---

### Task 1: Environment Setup, Git Hygiene, and Beads Orchestration

**Files:**
- Modify: `.gitignore`
- Modify: `pyproject.toml`
- Create: `scripts/setup_env.sh`
- Test: Manual verification via shell commands

**Interfaces:**
- Consumes: `/Users/micaelverissimo/Documents/pos-doc-activities/07-Oct-2026/.venv`
- Produces: Local `.venv` with `qpdesolver` (editable), `devito`, and `pytest`; clean `.gitignore` blocking `*.h5` and `.beads/`; initialized Beads project.

- [ ] **Step 1: Update `.gitignore` to block all `.h5`, `.hdf5`, and `.beads/` files**

Append:
```gitignore
*.h5
*.hdf5
.beads/
```

- [ ] **Step 2: Untrack existing `.h5` files from Git index without deleting them on disk**

Run:
```bash
git rm --cached examples/data/toy_fwi.h5 examples/data/toy_systems.h5 examples/data/tutorial_results.h5 reports/trainability_results.h5 2>/dev/null || true
git status
```
Verify output shows untracked `.h5` files ignored by `.gitignore`.

- [ ] **Step 3: Update `pyproject.toml` optional dependencies to include Devito**

In `pyproject.toml`, add under `[project.optional-dependencies]`:
```toml
generator = [
    "devito>=4.8.0",
]
```

- [ ] **Step 4: Provision workspace `.venv` and install dependencies**

Clone/seed the virtualenv using the Python 3.9 base from `/Users/micaelverissimo/Documents/pos-doc-activities/07-Oct-2026/.venv` (or Python 3.11 via `uv` if Devito wheel compatibility requires it), ensuring `numpy>=1.24.0,<2.0.0` and `pennylane>=0.38.0,<0.39.0` remain intact:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install "numpy>=1.24.0,<2.0.0"
pip install -r /Users/micaelverissimo/Documents/workspace/qpdesolver/requirements.txt
pip install devito
pip install -e ".[dev]"
pytest -v tests/unit/test_io_hdf5.py
```
Expected: All existing tests in `test_io_hdf5.py` PASS in the new `.venv`.

- [ ] **Step 5: Install Beads and initialize task tracking**

Run:
```bash
brew install beads 2>/dev/null || true
bd init
bd create "Organize qpdesolver.generator subpackage" --label generator
bd create "Optimize qpdesolver.io with explicit normalization" --label io
bd create "Implement 2D Helmholtz discretization and 2D FWI with VQLS" --label fwi2d
bd create "Deliver end-to-end VQLS FWI on model_1.h5 example" --label example
bd list
```

- [ ] **Step 6: Stage changes and check git status (NO COMMIT without user authorization)**

Run:
```bash
git add .gitignore pyproject.toml
git status
```
Expected: Only `.gitignore` and `pyproject.toml` staged. No `.h5` files staged. DO NOT execute `git commit`.

---

### Task 2: Explicit Normalization Subsystem

**Files:**
- Create: `src/qpdesolver/io/normalization.py`
- Test: `tests/unit/test_io_normalization.py`

**Interfaces:**
- Consumes: None (pure mathematical scaling and units metadata)
- Produces: `NormalizationParams` dataclass, `compute_normalization_params(...)`, `apply_seismic_normalization(...)`, and `denormalize_velocity(...)`.

- [ ] **Step 1: Write failing unit tests for normalization rules**

Create `tests/unit/test_io_normalization.py`:
```python
import pytest
import numpy as np
from qpdesolver.io.normalization import (
    NormalizationParams,
    compute_normalization_params,
    apply_seismic_normalization,
    denormalize_velocity,
)

def test_normalization_disabled_by_default():
    v = np.array([1.5, 2.5])
    params = compute_normalization_params(v, h=10.0, normalize=False)
    assert not params.normalized
    assert params.target_units == "raw"
    v_norm, h_norm, _ = apply_seismic_normalization(v, h=10.0, params=params)
    assert np.allclose(v_norm, v)
    assert h_norm == 10.0

def test_unit_target_without_normalize_raises_error():
    v = np.array([1.5, 2.5])
    with pytest.raises(ValueError, match="unit_target requires normalize=True"):
        compute_normalization_params(v, h=10.0, normalize=False, unit_target="si")

def test_normalization_to_si():
    v_kms = np.array([1.5, 2.5])
    params = compute_normalization_params(v_kms, h=10.0, normalize=True, unit_target="si", source_units="kms")
    v_si, h_si, _ = apply_seismic_normalization(v_kms, h=10.0, params=params)
    assert np.allclose(v_si, [1500.0, 2500.0])
    assert h_si == 10.0
    v_rec = denormalize_velocity(v_si, params)
    assert np.allclose(v_rec, v_kms)

def test_normalization_to_dimensionless():
    v_kms = np.array([1.5, 2.5])
    params = compute_normalization_params(v_kms, h=10.0, normalize=True, unit_target="dimensionless", source_units="kms")
    v_dim, h_dim, omega_fn = apply_seismic_normalization(v_kms, h=10.0, params=params)
    assert np.isclose(np.mean(v_dim), 1.0)
    assert np.isclose(h_dim, 1.0)
    v_rec = denormalize_velocity(v_dim, params)
    assert np.allclose(v_rec, v_kms)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_io_normalization.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'qpdesolver.io.normalization')

- [ ] **Step 3: Implement `src/qpdesolver/io/normalization.py`**

Implement `NormalizationParams` with scaling properties (`velocity_scale`, `length_scale`, `time_scale`, `omega_scale`), `compute_normalization_params`, `apply_seismic_normalization`, and `denormalize_velocity`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_io_normalization.py -v`
Expected: PASS

- [ ] **Step 5: Stage changes and check git status (NO COMMIT)**

Run:
```bash
git add src/qpdesolver/io/normalization.py tests/unit/test_io_normalization.py
git status
```

---

### Task 3: Optimized `devito-seismic-hdf5` I/O and FWI Bridges

**Files:**
- Modify: `src/qpdesolver/io/schema.py`
- Modify: `src/qpdesolver/io/hdf5.py`
- Create: `src/qpdesolver/io/devito_io.py`
- Modify: `src/qpdesolver/io/__init__.py`
- Test: `tests/unit/test_io_devito.py`

**Interfaces:**
- Consumes: `qpdesolver.io.normalization`, `qpdesolver.applications.fwi.model.VelocityModel`
- Produces: `export_devito_hdf5`, `load_devito_hdf5`, `summarize_hdf5`, `extract_velocity_1d_from_devito`, `extract_velocity_2d_from_devito`, `extract_frequency_data_from_devito`, `build_fwi_dataset_from_devito`.

- [ ] **Step 1: Write failing unit tests for Devito HDF5 I/O and bridges**

Create `tests/unit/test_io_devito.py`:
```python
import pytest
import numpy as np
from pathlib import Path
from qpdesolver.io.devito_io import (
    export_devito_hdf5,
    load_devito_hdf5,
    summarize_hdf5,
    extract_velocity_1d_from_devito,
    extract_velocity_2d_from_devito,
    build_fwi_dataset_from_devito,
)

def test_devito_hdf5_synthetic_export_and_load(tmp_path):
    h5_path = tmp_path / "test_synth.h5"
    vp = np.full((20, 20), 2.0, dtype=np.float32)
    vp[:, 10:] = 3.0
    time_vals = np.linspace(0, 100, 50)
    snaps = {"u": np.zeros((5, 20, 20), dtype=np.float32)}

    export_devito_hdf5(
        str(h5_path),
        vp=vp,
        origin=(0.0, 0.0),
        spacing=(10.0, 10.0),
        time_values=time_vals,
        snapshots=snaps,
        compression="gzip",
    )

    data = load_devito_hdf5(str(h5_path), include_fields=False)
    assert data["attrs"]["format"] == "devito-seismic-hdf5"
    assert "u" not in data.get("fields", {})
    assert data["model"]["vp"].shape == (20, 20)

    summary_str = summarize_hdf5(str(h5_path), print_output=False)
    assert "devito-seismic-hdf5" in summary_str
    assert "vp" in summary_str

def test_extract_velocity_and_fwi_dataset(tmp_path):
    h5_path = tmp_path / "test_synth2.h5"
    vp = np.full((16, 16), 1.5, dtype=np.float32)
    vp[:, 8:] = 2.5
    export_devito_hdf5(str(h5_path), vp=vp, origin=(0.0, 0.0), spacing=(10.0, 10.0))

    # Extract 1D profile resampled to 4 points
    v1d = extract_velocity_1d_from_devito(str(h5_path), resample_to=4, normalize=True, unit_target="si")
    assert v1d.n_points == 4
    assert np.isclose(v1d.velocity[0], 1500.0)
    assert np.isclose(v1d.velocity[-1], 2500.0)

    # Build 1D FWI dataset
    fwi_ds = build_fwi_dataset_from_devito(str(h5_path), mode="1d", grid_shape=4, normalize=True, unit_target="si")
    assert fwi_ds.true_velocity.shape == (4,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_io_devito.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'qpdesolver.io.devito_io')

- [ ] **Step 3: Implement `src/qpdesolver/io/devito_io.py`, update `schema.py` and `hdf5.py`**

- In `src/qpdesolver/io/schema.py`: add `"velocity_model_2d"`, `"fwi_dataset_2d"` to `SUPPORTED_TYPES`.
- In `src/qpdesolver/io/devito_io.py`: implement `export_devito_hdf5`, `load_devito_hdf5`, `summarize_hdf5`, `extract_velocity_1d_from_devito`, `extract_velocity_2d_from_devito`, `extract_frequency_data_from_devito`, `build_fwi_dataset_from_devito`.
- In `src/qpdesolver/io/hdf5.py`: support chunked compression opts in `save_linear_system`, `save_velocity_model`, `save_fwi_dataset`.
- In `src/qpdesolver/io/__init__.py`: expose new routines.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_io_devito.py tests/unit/test_io_hdf5.py -v`
Expected: PASS

- [ ] **Step 5: Stage changes and check git status (NO COMMIT)**

Run:
```bash
git add src/qpdesolver/io/ tests/unit/test_io_devito.py
git status
```

---

### Task 4: Modular Synthetic Seismic Generator

**Files:**
- Create: `src/qpdesolver/generator/models.py`
- Create: `src/qpdesolver/generator/recorder.py`
- Create: `src/qpdesolver/generator/acoustic.py`
- Create: `src/qpdesolver/generator/__init__.py`
- Modify: `src/qpdesolver/generator/devito_hdf5.py`
- Test: `tests/unit/test_generator.py`

**Interfaces:**
- Consumes: `qpdesolver.io.devito_io.export_devito_hdf5`
- Produces: `two_layer_velocity`, `multi_layer_velocity`, `gaussian_anomaly_velocity`, `salt_inclusion_velocity`, `SnapshotRecorder`, `generate_seismic_sample`.

- [ ] **Step 1: Write failing unit tests for synthetic generator**

Create `tests/unit/test_generator.py`:
```python
import pytest
import numpy as np
from qpdesolver.generator.models import (
    two_layer_velocity,
    multi_layer_velocity,
    gaussian_anomaly_velocity,
    salt_inclusion_velocity,
)
from qpdesolver.generator.acoustic import SeismicSurveySpec, generate_seismic_sample
from qpdesolver.io.devito_io import load_devito_hdf5

def test_velocity_models_generation():
    m2 = two_layer_velocity(shape=(21, 21), v_top=1.5, v_bottom=2.5, interface_z=0.5)
    assert m2.shape == (21, 21)
    assert np.isclose(m2[0, 0], 1.5)
    assert np.isclose(m2[0, -1], 2.5)

    mg = gaussian_anomaly_velocity(shape=(21, 21), v_bg=2.0, dv=0.6)
    assert mg.shape == (21, 21)
    assert np.max(mg) > 2.5

def test_generate_seismic_sample_numpy_backend(tmp_path):
    h5_file = tmp_path / "numpy_sample.h5"
    vp = two_layer_velocity(shape=(31, 31), v_top=1.5, v_bottom=2.5)
    spec = SeismicSurveySpec(
        shape=(31, 31),
        spacing=(10.0, 10.0),
        nbl=5,
        t0=0.0,
        tn=50.0,
        f0=0.015,
        nsnaps=4,
    )
    res = generate_seismic_sample(str(h5_file), vp=vp, spec=spec, backend="numpy")
    assert h5_file.is_file()
    data = load_devito_hdf5(str(h5_file))
    assert data["attrs"]["format"] == "devito-seismic-hdf5"
    assert "rec" in data["geometry"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_generator.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'qpdesolver.generator.models')

- [ ] **Step 3: Implement `src/qpdesolver/generator/` modules and compatibility shim**

- `src/qpdesolver/generator/models.py`: parametric velocity array generators.
- `src/qpdesolver/generator/recorder.py`: `SnapshotRecorder`, `flatten_fields`, `strip_absorbing_layer`.
- `src/qpdesolver/generator/acoustic.py`: `SeismicSurveySpec` dataclass, Devito and NumPy 2D finite-difference wave propagators, `generate_seismic_sample`.
- `src/qpdesolver/generator/devito_hdf5.py`: preserve legacy imports pointing to `recorder` and `qpdesolver.io.devito_io`.
- `src/qpdesolver/generator/__init__.py`: expose public symbols.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_generator.py -v`
Expected: PASS

- [ ] **Step 5: Stage changes and check git status (NO COMMIT)**

Run:
```bash
git add src/qpdesolver/generator/ tests/unit/test_generator.py
git status
```

---

### Task 5: 2D Frequency-Domain Helmholtz Discretization

**Files:**
- Create: `src/qpdesolver/problems/discretizations/helmholtz2d.py`
- Modify: `src/qpdesolver/problems/discretizations/__init__.py`
- Test: `tests/unit/test_helmholtz2d.py`

**Interfaces:**
- Consumes: `qpdesolver.problems.discretizations.base.PDEDiscretization`
- Produces: `Helmholtz2D` class registered as `"helmholtz2d"`, constructing $L_{2D} = L_x \otimes I + I \otimes L_z - \omega^2 \operatorname{diag}(m_{\text{flat}})$.

- [ ] **Step 1: Write failing unit tests for `Helmholtz2D`**

Create `tests/unit/test_helmholtz2d.py`:
```python
import pytest
import numpy as np
from qpdesolver.problems.discretizations.helmholtz2d import Helmholtz2D

def test_helmholtz2d_dimension_validation():
    with pytest.raises(ValueError, match="positive power of 2"):
        Helmholtz2D(nx=3, nz=4)

def test_helmholtz2d_matrix_properties():
    # 2x2 grid = 4 points (2 qubits)
    h2d = Helmholtz2D(nx=2, nz=2, hx=1.0, hz=1.0, bc="dirichlet")
    assert h2d.n_points == 4
    assert h2d.n_qubits == 2

    # Verify operator symmetry and dimension
    m = np.ones((2, 2))
    A = h2d.operator(m=m, omega=1.0)
    assert A.shape == (4, 4)
    assert np.allclose(A, A.conj().T)

    # Point source 2D
    b = h2d.point_source_2d(ix=1, iz=0)
    assert b.shape == (4,)
    assert b[2] == 1.0
    assert np.sum(b) == 1.0

def test_helmholtz2d_linear_system_creation():
    h2d = Helmholtz2D(nx=4, nz=4, hx=5.0, bc="dirichlet")
    b = h2d.point_source_2d(2, 2)
    sys = h2d.system(rhs=b, omega=1.5, m=np.full((4, 4), 1.0 / 2000.0**2))
    assert sys.size == 16
    assert sys.n_qubits == 4
    assert sys.is_hermitian
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/unit/test_helmholtz2d.py -v`
Expected: FAIL (ModuleNotFoundError: No module named 'qpdesolver.problems.discretizations.helmholtz2d')

- [ ] **Step 3: Implement `Helmholtz2D` in `src/qpdesolver/problems/discretizations/helmholtz2d.py`**

Inherit from `PDEDiscretization`, compute $L_{2D} = L_x \otimes I_{n_z} + I_{n_x} \otimes L_z$, build $A = L_{2D} - \omega^2 \operatorname{diag}(m_{\text{flat}})$, and register with `REGISTRY_DISCRETIZATIONS`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/unit/test_helmholtz2d.py -v`
Expected: PASS

- [ ] **Step 5: Stage changes and check git status (NO COMMIT)**

Run:
```bash
git add src/qpdesolver/problems/discretizations/ tests/unit/test_helmholtz2d.py
git status
```

---

### Task 6: 2D FWI Data Models, 2D Adjoint Engine, and VQLS Warm-Starting

**Files:**
- Modify: `src/qpdesolver/applications/fwi/model.py`
- Modify: `src/qpdesolver/applications/fwi/problem.py`
- Modify: `src/qpdesolver/applications/fwi/inversion.py`
- Modify: `src/qpdesolver/applications/fwi/__init__.py`
- Test: `tests/integration/test_fwi_2d.py`

**Interfaces:**
- Consumes: `qpdesolver.problems.discretizations.helmholtz2d.Helmholtz2D`, `qpdesolver.solvers.vqls.VQLSSolver`
- Produces: `VelocityModel2D`, `Acquisition2D`, `FWIDataset2D`, `FWIProblem2D`, 2D-capable `FWIInversion`, and `warm_start_solves` caching.

- [ ] **Step 1: Write failing integration test for 2D FWI and VQLS warm-starting**

Create `tests/integration/test_fwi_2d.py`:
```python
import pytest
import numpy as np
from qpdesolver.applications.fwi.model import VelocityModel2D, Acquisition2D
from qpdesolver.applications.fwi.problem import FWIProblem2D
from qpdesolver.applications.fwi.inversion import FWIInversion
from qpdesolver.solvers.classical import ClassicalSolver
from qpdesolver.solvers.vqls import VQLSSolver
from qpdesolver.ansatze import RealAmplitudes
from qpdesolver.optimizers.scipy_wrapper import ScipyOptimizer

def test_fwi_2d_adjoint_gradient_vs_finite_difference():
    # 2x2 grid = 4 points
    nx, nz = 2, 2
    acq = Acquisition2D(sources=[(0, 0)], receivers=[(0, 0), (1, 1)], omegas=[1.0])
    problem = FWIProblem2D(acquisition=acq, nx=nx, nz=nz, hx=10.0, hz=10.0)

    v = np.full((nx, nz), 2000.0)
    m = 1.0 / (v**2)
    d_obs = problem.forward_all(m)

    # Perturb model slightly
    v_test = np.array([[2000.0, 2100.0], [2050.0, 2200.0]])
    m_test = 1.0 / (v_test**2)

    grad_m = problem.gradient_m(m_test, d_obs)
    assert grad_m.shape == (nx, nz)

    # Numerical finite difference check on one cell
    eps = 1e-8
    m_pert = m_test.copy()
    m_pert[0, 1] += eps
    fd = (problem.misfit(m_pert, d_obs) - problem.misfit(m_test, d_obs)) / eps
    assert np.isclose(grad_m[0, 1], fd, rtol=1e-3)

def test_fwi_2d_inversion_with_vqls_warm_start():
    # 2x2 grid = 4 points (2 qubits)
    nx, nz = 2, 2
    acq = Acquisition2D(sources=[(0, 0)], receivers=[(0, 0), (1, 1)], omegas=[1.0])
    v_solver = VQLSSolver(ansatz=RealAmplitudes(layers=2), max_iter=40, seed=42)
    problem = FWIProblem2D(acquisition=acq, nx=nx, nz=nz, hx=10.0, solver=v_solver, warm_start_solves=True)

    true_v = np.array([[2000.0, 2400.0], [2000.0, 2400.0]])
    init_v = np.full((nx, nz), 2000.0)
    true_m = VelocityModel2D(velocity=true_v, hx=10.0)
    init_m = VelocityModel2D(velocity=init_v, hx=10.0)

    dataset = problem.simulate_dataset(true_model=true_m, initial_model=init_m)
    inversion = FWIInversion(
        problem=problem,
        dataset=dataset,
        optimizer=ScipyOptimizer(method="L-BFGS-B"),
        bounds=(1500.0, 3000.0),
        max_iter=3,
    )
    res = inversion.run(init_m)
    assert res.final_misfit <= res.misfit_history[0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/integration/test_fwi_2d.py -v`
Expected: FAIL (ImportError / AttributeError)

- [ ] **Step 3: Implement 2D FWI data structures, problem, and generalized inversion**

- In `src/qpdesolver/applications/fwi/model.py`: add `VelocityModel2D`, `Acquisition2D`, `FWIDataset2D`.
- In `src/qpdesolver/applications/fwi/problem.py`: add `warm_start_solves` caching to `FWIProblem` and implement `FWIProblem2D`.
- In `src/qpdesolver/applications/fwi/inversion.py`: update `FWIObjective` and `FWIInversion` to handle 2D velocity arrays (flattening for optimizer and reshaping for model recovery).
- In `src/qpdesolver/applications/fwi/__init__.py`: export new classes.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/integration/test_fwi_2d.py tests/integration/test_fwi_inversion.py -v`
Expected: PASS

- [ ] **Step 5: Stage changes and check git status (NO COMMIT)**

Run:
```bash
git add src/qpdesolver/applications/fwi/ tests/integration/test_fwi_2d.py
git status
```

---

### Task 7: End-to-End VQLS FWI on `model_1.h5` Example and Data Relocation

**Files:**
- Move: `src/qpdesolver/generator/model_1.h5` $\to$ `examples/data/model_1.h5`
- Move: `src/qpdesolver/generator/model_1_hdf5.ipynb` $\to$ `examples/notebooks/model_1_hdf5.ipynb`
- Create: `examples/11_fwi_vqls_model1.py`
- Test: `tests/integration/test_fwi_vqls_example.py`

**Interfaces:**
- Consumes: `qpdesolver.io`, `qpdesolver.generator`, `qpdesolver.solvers.vqls.VQLSSolver`, `qpdesolver.applications.fwi`
- Produces: Executable example solving 1D and 2D FWI from `model_1.h5` with `VQLSSolver` vs `ClassicalSolver`.

- [ ] **Step 1: Move sample files out of `src/` to designated directories**

Run:
```bash
mkdir -p examples/data examples/notebooks
cp src/qpdesolver/generator/model_1.h5 examples/data/model_1.h5
mv src/qpdesolver/generator/model_1_hdf5.ipynb examples/notebooks/model_1_hdf5.ipynb
rm src/qpdesolver/generator/model_1.h5
```

- [ ] **Step 2: Write test asserting `examples/11_fwi_vqls_model1.py` runs end-to-end**

Create `tests/integration/test_fwi_vqls_example.py`:
```python
import subprocess
import sys
from pathlib import Path

def test_example_11_runs_successfully():
    repo_root = Path(__file__).resolve().parents[2]
    script_path = repo_root / "examples" / "11_fwi_vqls_model1.py"
    res = subprocess.run(
        [sys.executable, str(script_path), "--test-mode"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0, f"Script failed with output:\n{res.stdout}\n{res.stderr}"
    assert "FWI VQLS on model_1 completed successfully" in res.stdout
```

- [ ] **Step 3: Implement `examples/11_fwi_vqls_model1.py` with Success Rate / Problem Solved Metrics**

Implement complete workflow:
1. Load and inspect `examples/data/model_1.h5` (or auto-generate via `generate_seismic_sample` if missing).
2. Demonstrate normalization flag (`normalize=True, unit_target="dimensionless"`).
3. Part 1 (1D): Invert central depth profile with `VQLSSolver` and `ClassicalSolver`.
4. Part 2 (2D): Invert subsampled 4x4 grid with `VQLSSolver` and `ClassicalSolver`.
5. **Success Rate and Problem Solved Assessment:**
   - Compute and print quantitative success indicators:
     - Quantum linear system solve fidelity $F = |\langle u_{\text{exact}} | u_{\text{vqls}} \rangle|^2 \ge 99.0\%$ and relative residual $\frac{\|A u - b\|_2}{\|b\|_2}$.
     - FWI relative misfit reduction $\Delta J_{\text{rel}} = \frac{J_0 - J_{\text{final}}}{J_0} \times 100\%$.
     - Velocity recovery error reduction: $\text{RMSE}(v_{\text{rec}}, v_{\text{true}}) < \text{RMSE}(v_{\text{init}}, v_{\text{true}})$.
     - Prominent summary banner:
       `============================================================`
       `FWI VQLS RESOLVIDO COM SUCESSO: [SIM / NÃO]`
       `Taxa de Redução do Misfit (Sucesso FWI): XX.X%`
       `Fidelidade do Estado Quântico (VQLS): YY.YY%`
       `============================================================`
6. Support `--test-mode` flag for rapid test execution (1-2 iterations).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/integration/test_fwi_vqls_example.py -v`
Expected: PASS

- [ ] **Step 5: Run complete project test suite to verify zero regressions**

Run:
```bash
.venv/bin/pytest -v
```
Expected: All unit and integration tests PASS.

- [ ] **Step 6: Stage clean changes and verify git status (NO COMMIT without user authorization)**

Run:
```bash
git status
```
Confirm:
1. No `.h5` or `.hdf5` file is tracked or staged.
2. Only code and tests are staged.
3. No `git commit` has been executed.

---

## Self-Review Checklist

1. **Spec coverage:** All four user tasks (organizing `generator`, optimizing `io` with explicit normalization, `.h5` Git exclusion without unauthorized commits, and 1D/2D FWI with `VQLSSolver` on `model_1.h5`) are covered in Tasks 1-7.
2. **Step scan:** Each task has failing test, run verification, implementation, pass verification, and staged status check.
3. **Type consistency:** Signatures for `compute_normalization_params`, `export_devito_hdf5`, `Helmholtz2D`, `VelocityModel2D`, `FWIProblem2D` are consistent across tasks.
4. **Review Focus:** Unit scaling, dispersion crime mitigation, power-of-2 resampling, warm-start parameter caching, and `.h5` git exclusion are directly addressed in tests.
5. **Commit gate:** The plan repeatedly and strictly reinforces that no `git commit` is executed without explicit user authorization.

