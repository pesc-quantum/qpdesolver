# `qpdesolver`: Quantum Partial Differential Equation Solver & Trainability Framework

`qpdesolver` is a flexible, research-grade quantum algorithm engineering framework designed to solve Partial Differential Equations (PDEs) and conduct variational trainability studies. Developed to advance quantum Full Waveform Inversion (FWI) and quantum scientific computing, `qpdesolver` supports both fault-tolerant quantum algorithms (HHL) and NISQ variational algorithms (VQLS), coupled with Riemannian optimization, Lie-algebraic diagnostics, modular seismic data generation, and standardized HDF5 persistence.

---

## 1. Key Features

- **Multi-Algorithm Architecture**:
  - **Harrow-Hassidim-Lloyd (HHL)**: Modular implementation with exact signed two's-complement eigenvalue inversion and register size tuning.
  - **Variational Quantum Linear Solver (VQLS)**: Modular cost functions (Global, Local, Custom), exact Fubini-Study Riemannian gradients, and analytical state overlap calculations.
  - **Classical Baseline**: High-precision LU/Cholesky/SVD solvers for exact fidelity verification.
- **Full Waveform Inversion (FWI) Engine (1D & 2D)**:
  - 1D and 2D frequency-domain acoustic Helmholtz forward and adjoint solvers.
  - Adjoint-state method computing exact analytical gradients $\nabla_m J$ and $\nabla_v J$ with $< 10^{-7}$ relative error vs finite differences.
  - Variational parameter **warm-starting** across iterative FWI steps (reducing VQLS circuit evaluations by $> 70\%$).
  - End-to-end inversion pipeline (`FWIInversion`) supporting L-BFGS-B, Adam, and custom optimizers.
- **Modular Synthetic Seismic Generator (`qpdesolver.generator`)**:
  - Analytical and layered velocity models: `two_layer_velocity`, `multi_layer_velocity`, `gaussian_anomaly_velocity`, `salt_inclusion_velocity`.
  - Second-order acoustic finite-difference propagator with **Devito** acceleration and pure-NumPy fallback with Cerjan absorbing sponge boundary layers.
  - Multi-channel snapshot and geophone recording (`SnapshotRecorder`).
- **Devito HDF5 I/O & Explicit Normalization (`qpdesolver.io`)**:
  - Compressed, chunked snapshot storage (`chunks=(1, *space)`, `shuffle=True`).
  - Lazy, selective snapshot extraction and metadata inspection (`load_devito_hdf5`, `export_devito_hdf5`, `summarize_hdf5`).
  - Strict, explicit dimensional normalization (`normalize: bool = False`, with target units `"dimensionless"`, `"si"`, and `"kms"`).
  - Bridges converting Devito time surveys into frequency-domain Helmholtz FWI datasets (`build_fwi_dataset_from_devito`).
- **2D Helmholtz Discretization (`Helmholtz2D`)**:
  - Kronecker-sum 5-point finite-difference operator $L_{2D} = L_x \otimes I + I \otimes L_z - \omega^2 \operatorname{diag}(m)$.
  - Exact Pauli decomposition and Hermitian dilation for quantum linear solvers ($n = \log_2(nx \cdot nz)$ qubits).
- **Ansatz & Initialization Library**:
  - `RealAmplitudes`, `HardwareEfficient`, `AlternatingLayered`, `HamiltonianVariational`.
  - Custom user-defined ansatz wrappers (`FunctionAnsatz`, `TemplateAnsatz`).
  - Parameter initialization strategies: `uniform`, `small_angle`, `zeros`, `identity_blocks`, and `beta`.
- **Advanced Optimizer Suite**:
  - First-order: `GradientDescent`, `Adam`.
  - Second-order: `DampedNewton` (Levenberg-Marquardt with adaptive damping).
  - Riemannian: `QuantumNaturalGradient` (QNG) using the exact Fubini-Study metric tensor.
  - Stochastic: `SPSA`, `QNSPSA`.
  - SciPy wrapper: `BFGS`, `L-BFGS-B`, `Newton-CG`, `COBYLA`.
- **Comprehensive Trainability & Geometry Diagnostics**:
  - **Barren Plateau Detection**: Automated gradient variance sampling and exponential decay rate fitting ($O(2^{-2n})$).
  - **Dynamical Lie Algebra (DLA)**: Exact Lie closure dimension $\dim(\mathfrak{g})$ via PennyLane `lie_closure` with matrix commutant fallback.
  - **Quantum Fisher Information Matrix (QFIM)**: Rank saturation, spectral properties, and overparameterization threshold detection.
  - **Expressibility & Entanglement**: Meyer-Wallach entanglement measure and Haar-frame potential KL divergence.
  - **Mitigation Pipelines**: Progressive `LayerwiseTrainer` and initialization benchmarking.

---

## 2. Directory Structure

```
qpdesolver/
├── pyproject.toml               # Package configuration with optional [generator] dependencies
├── pytest.ini                   # Pytest test suite configuration
├── .gitignore                   # Ignores *.h5, *.hdf5, and IDE/agent configs
├── README.md                    # This documentation
├── docs/                        # Theoretical foundations and developer guides
│   ├── THEORY_AND_ALGORITHMS.md # Mathematical derivations (HHL, VQLS, FWI, DLA, QFIM)
│   ├── CUSTOM_ANSATZ_GUIDE.md   # Designing and benchmarking custom ansatze
│   ├── CUSTOM_PDE_PROBLEMS.md   # Implementing novel PDE discretizations
│   ├── CUSTOM_COSTS_AND_OPTIMIZERS.md # Custom VQLS cost functions and optimizers
│   └── VISUAL_TUTORIAL.md       # Master educational guide with diagrams
├── src/qpdesolver/
│   ├── core/                    # Registry pattern and metrics (fidelity, relative_residual)
│   ├── problems/                # LinearSystem, Pauli decomposition
│   │   └── discretizations/     # Helmholtz1D, Helmholtz2D, Poisson1D, Spectral, FEM
│   ├── generator/               # Modular synthetic seismic data generation
│   │   ├── models.py            # Velocity models (two-layer, multi-layer, salt, gaussian)
│   │   ├── acoustic.py          # Time-domain wave equation (Devito & NumPy fallback)
│   │   └── recorder.py          # Snapshot recorder and absorbing layer stripping
│   ├── io/                      # Standardized HDF5 & Devito I/O and normalization
│   │   ├── normalization.py     # Explicit dimensional normalization (SI, km/s, dimensionless)
│   │   ├── devito_io.py         # Snapshot-chunked Devito HDF5 reading and conversion
│   │   ├── hdf5.py              # Schema validation and LinearSystem / FWIDataset I/O
│   │   └── schema.py            # HDF5 dataset and group schemas
│   ├── ansatze/                 # RealAmplitudes, HardwareEfficient, custom ansatze
│   ├── optimizers/              # Adam, QNG, DampedNewton, SPSA, ScipyWrapper
│   ├── solvers/
│   │   ├── base.py              # LinearSolver ABC and SolveResult
│   │   ├── classical.py         # Classical LU linear solver
│   │   ├── hhl/                 # HHLSolver, HHLConfig, PennyLaneHHLKernel
│   │   └── vqls/                # VQLSSolver, VQLSObjective, GlobalCost, LocalCost
│   ├── trainability/            # Barren plateaus, DLA, QFIM, expressibility
│   └── applications/fwi/        # 1D & 2D FWI: VelocityModel, Acquisition, FWIProblem, FWIInversion
├── examples/
│   ├── data/                    # Data directory (strictly .gitignore for *.h5)
│   ├── notebooks/
│   │   └── model_1_hdf5.ipynb   # Seismic Devito data exploration notebook
│   ├── 01_hhl_quickstart.py     # HHL solve on Helmholtz 1D
│   ├── 02_vqls_custom_ansatz.py # Custom user-defined ansatz in VQLS
│   ├── 03_optimizer_zoo.py      # Benchmark across 5 optimization algorithms
│   ├── 04_fwi_from_h5_hhl_vs_vqls.py # FWI PDE solve with HHL vs VQLS
│   ├── 05_barren_plateau_scan.py     # Gradient variance scaling scan
│   ├── 06_dla_and_qfim_of_ansatze.py # DLA dimension and QFIM rank saturation
│   ├── 07_ansatz_comparison_on_pde.py# Multi-ansatz comparative benchmark
│   ├── 08_qiskit_device_swap.py      # Simulator and backend device swapping
│   ├── 09_researcher_walkthrough_custom_ansatz_and_pde.py # End-to-end custom PDE & ansatz tutorial
│   ├── 10_alternative_pde_representations.py # FDM vs FEM vs Fourier Spectral vs Quantum PINN
│   └── 11_fwi_vqls_model1.py    # End-to-end 1D & 2D FWI on model_1.h5 with VQLS
├── reports/
│   ├── fwi_model1_vqls_report.md     # Deep benchmark report on model_1.h5 FWI with VQLS
│   ├── fwi_model1_vqls_report.html   # Interactive publication HTML report with embedded figures
│   └── figures/                      # High-resolution (300 DPI) publication figures
├── scripts/
│   ├── run_fwi_model1_report.py # Generates deep FWI VQLS benchmark report and figures
│   ├── run_comparison.py        # Generates HHL vs VQLS benchmarks and reports
│   ├── run_trainability.py      # Generates trainability scan and reports
│   └── setup_env.sh             # Virtual environment setup script
└── tests/
    ├── unit/                    # 22 unit test suites across all modules (91 tests)
    └── integration/             # Integration tests (1D/2D FWI, parity, VQLS example 11)
```

---

## 3. Installation & Quickstart

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[generator]"
```

### Quickstart: Solving 2D Helmholtz with VQLS
```python
import numpy as np
from qpdesolver.problems.discretizations import Helmholtz2D
from qpdesolver.solvers.vqls import VQLSSolver
from qpdesolver.ansatze import RealAmplitudes
from qpdesolver.optimizers.scipy_wrapper import ScipyOptimizer
from qpdesolver.core.metrics import fidelity

# 4x4 grid (16 points, 4 qubits)
helm = Helmholtz2D(nx=4, nz=4, hx=1.0, hz=1.0)
m = np.ones(16) * (1.0 / 2.0**2)
system = helm.system(m=m, omega=0.5, rhs=helm.point_source_2d(0, 0))

# Solve with VQLS
solver = VQLSSolver(
    ansatz=RealAmplitudes(layers=2),
    optimizer=ScipyOptimizer("BFGS"),
    max_iter=30,
    seed=42,
)
result = solver.solve(system)

exact_sol = system.classical_solution()
print(f"VQLS Solution Fidelity: {fidelity(result.state, exact_sol)*100:.2f}%")
print(f"Residual: {result.residual:.4e}")
```

### Quickstart: Generating Synthetic Seismic Samples
```python
from qpdesolver.generator import generate_seismic_sample
from qpdesolver.generator.models import two_layer_velocity

# Generate a two-layer velocity field (1.5 km/s top, 2.5 km/s bottom)
vp = two_layer_velocity(shape=(101, 101), v_top=1.5, v_bottom=2.5)

# Propagate acoustic wave and save to HDF5 (using Devito or NumPy fallback)
generate_seismic_sample(
    filepath="examples/data/sample_survey.h5",
    vp=vp,
    f0=0.010,  # 10 Hz peak frequency
    backend="auto",
)
```

### Quickstart: 2D Full Waveform Inversion with VQLS on `model_1.h5`
```python
# Run the complete end-to-end benchmark
python examples/11_fwi_vqls_model1.py
```

Output:
```text
============================================================
[FWI VQLS RESOLVIDO COM SUCESSO: SIM]
Taxa de Redução do Misfit 1D: 100.00%
Taxa de Redução do Misfit 2D: 100.00%
Fidelidade Quântica do Solucionador VQLS 1D: 100.0000%
Fidelidade Quântica do Solucionador VQLS 2D: 100.0000%
============================================================
```

---

## 4. Benchmark Reports & Diagnostics

To generate the comprehensive geophysical and quantum benchmark report comparing real source signatures, receiver shot gathers, wavefield time snapshots, multi-frequency conditioning, and VQLS training dynamics:

```bash
python scripts/run_fwi_model1_report.py
```

Generated outputs:
- **Markdown Report**: `reports/fwi_model1_vqls_report.md`
- **Interactive HTML Report**: `reports/fwi_model1_vqls_report.html`
- **Publication Figures**: `reports/figures/` (300 DPI)

---

## 5. Running the Test Suite

```bash
pytest -v
```

All 92 test cases verify:
- Unit normalization conversions (`tests/unit/test_io_normalization.py`)
- Devito I/O reading, writing, and bridges (`tests/unit/test_io_devito.py`)
- Synthetic model generation and wavefield propagation (`tests/unit/test_generator.py`)
- 2D Helmholtz discretization and Pauli decomposition (`tests/unit/test_helmholtz2d.py`)
- 2D FWI forward modeling, analytical adjoint gradients, and VQLS warm-starting (`tests/integration/test_fwi_2d.py`)
- End-to-end Example 11 execution (`tests/integration/test_fwi_vqls_example.py`)

---

## 6. License & Data Policy

- Synthetic seismic HDF5 data files (`*.h5`, `*.hdf5`) are dynamically generated and strictly excluded from version control via `.gitignore`.
- Benchmark datasets can be regenerated locally at any time using `qpdesolver.generator`.
