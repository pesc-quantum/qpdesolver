# `qpdesolver`: Quantum Partial Differential Equation Solver & Trainability Framework

`qpdesolver` is a flexible, research-grade quantum algorithm engineering framework designed to solve Partial Differential Equations (PDEs) and conduct variational trainability studies. Developed to advance quantum Full Waveform Inversion (FWI) and quantum scientific computing, `qpdesolver` supports both fault-tolerant quantum algorithms (HHL) and NISQ variational algorithms (VQLS), coupled with Riemannian optimization, Lie-algebraic diagnostics, and standardized HDF5 persistence.

---

## 1. Key Features

- **Multi-Algorithm Architecture**:
  - **Harrow-Hassidim-Lloyd (HHL)**: Flexible, modular implementation with exact signed two's-complement eigenvalue inversion and register size tuning.
  - **Variational Quantum Linear Solver (VQLS)**: Modular cost functions (Global, Local, Custom), exact Fubini-Study Riemannian gradients, and analytical state overlap calculations.
  - **Classical Baseline**: High-precision LU/Cholesky/SVD solvers for exact fidelity verification.
- **Full Waveform Inversion (FWI) Engine**:
  - 1D frequency-domain acoustic Helmholtz forward and adjoint solvers.
  - Adjoint-state method computing exact gradients $\nabla_m J$ and $\nabla_v J$ with $< 10^{-5}$ relative error vs finite differences.
  - End-to-end inversion pipeline (`FWIInversion`) supporting L-BFGS-B, Adam, and custom optimizers.
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
- **Standardized HDF5 I/O**:
  - Versioned schema for `LinearSystem` (dense and CSR sparse), `VelocityModel`, `FWIDataset`, and nested study results.
- **Strict Legacy Parity**:
  - Bit-exact preservation of `17-Jul-2026` benchmark systems with 100.000% numerical fidelity.

---

## 2. Directory Structure

```
07-Oct-2026/
├── pyproject.toml               # Package build configuration (editable install)
├── requirements.txt             # Pinned environment dependencies (PennyLane 0.38, autoray<0.7)
├── pytest.ini                   # Pytest test suite configuration
├── PORTING.md                   # Detailed migration guide from 17-Jul-2026
├── README.md                    # This documenta├── docs/                        # Comprehensive academic and developer guides
│   ├── THEORY_AND_ALGORITHMS.md # Mathematical foundations (HHL, VQLS, FWI, DLA, QFIM)
│   ├── CUSTOM_ANSATZ_GUIDE.md   # Step-by-step tutorial on designing and benchmarking ansatze
│   ├── CUSTOM_PDE_PROBLEMS.md   # Step-by-step tutorial on implementing custom PDE discretizations
│   ├── CUSTOM_COSTS_AND_OPTIMIZERS.md # Guide for custom VQLS costs and Riemannian optimizers
│   ├── DISCRETIZATIONS_AND_ALTERNATIVE_STRATEGIES.md # Comprehensive guide on FDM vs FEM vs Spectral vs Q-PINN
│   └── VISUAL_TUTORIAL.md       # Master visual educational walkthrough linking schematics
├── src/qpdesolver/
│   ├── core/                    # Registry pattern and metrics (fidelity, residual)
│   ├── problems/                # LinearSystem, Pauli decomposition, Helmholtz1D, Poisson1D, Spectral, FEM
│   ├── io/                      # Standardized HDF5 read/write schema
│   ├── backends/                # PennyLaneBackend, QiskitBackend, CudaQBackend
│   ├── ansatze/                 # Ansatz library, adapters, and initializers
│   ├── optimizers/              # Adam, QNG, DampedNewton, SPSA, ScipyWrapper
│   ├── solvers/
│   │   ├── base.py              # LinearSolver ABC and SolveResult dataclass
│   │   ├── classical.py         # Classical LU linear solver
│   │   ├── hhl/                 # HHLSolver, HHLConfig, PennyLaneHHLKernel
│   │   └── vqls/                # VQLSSolver, VQLSObjective, GlobalCost, LocalCost
│   ├── trainability/            # Barren plateaus, DLA, QFIM, expressibility, mitigation
│   └── applications/fwi/        # VelocityModel, FWIDataset, FWIProblem, FWIInversion
├── examples/
│   ├── data/                    # Generated toy datasets (toy_fwi.h5, toy_systems.h5)
│   ├── 01_hhl_quickstart.py     # HHL solve on Helmholtz 1D
│   ├── 02_vqls_custom_ansatz.py # Custom user-defined ansatz in VQLS
│   ├── 03_optimizer_zoo.py      # Benchmark across 5 optimization algorithms
│   ├── 04_fwi_from_h5_hhl_vs_vqls.py # FWI PDE solve with HHL vs VQLS
│   ├── 05_barren_plateau_scan.py     # Gradient variance scaling scan
│   ├── 06_dla_and_qfim_of_ansatze.py # DLA dimension and QFIM rank saturation
│   ├── 07_ansatz_comparison_on_pde.py# Multi-ansatz comparative benchmark
│   ├── 08_qiskit_device_swap.py      # Simulator and backend device swapping
│   ├── 09_researcher_walkthrough_custom_ansatz_and_pde.py # End-to-end custom PDE & ansatz tutorial
│   └── 10_alternative_pde_representations.py # FDM vs FEM vs Fourier Spectral vs continuous Quantum PINN
├── reports/
├── reports/
│   ├── comparison_report.md     # Full quantitative report (HHL vs VQLS)
│   ├── comparison_report.html   # Rich dark-mode interactive HTML report
│   ├── trainability_report.md   # Theoretical & empirical trainability report
│   ├── trainability_report.html # Rich interactive HTML trainability report
│   ├── trainability_results.h5  # Structured experiment results in HDF5
│   └── figures/                 # High-resolution benchmark figures
├── scripts/
│   ├── run_comparison.py        # Generates HHL vs VQLS benchmarks and reports
│   └── run_trainability.py      # Generates trainability scan and reports
└── tests/
    ├── unit/                    # 17 unit tests across all subpackages
    └── integration/             # 6 integration tests (parity, FWI, trainability)
```

---

## 3. Installation & Quickstart

### Installation
From the `07-Oct-2026` folder:
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

### Quickstart: Solving Helmholtz 1D with HHL
```python
from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.solvers.hhl import HHLSolver, HHLConfig

# Build 2-qubit (4-point) Helmholtz problem
helm = Helmholtz1D(n_points=4, h=1.0, bc="dirichlet")
system = helm.system(rhs=helm.point_source(1), omega=1.0)

# Solve with HHL using 4 clock qubits
solver = HHLSolver(config=HHLConfig(n_clock=4))
result = solver.solve(system)

print(f"HHL Solution Fidelity: {result.fidelity*100:.2f}%")
print(f"Residual: {result.residual:.4e}")
```

### Quickstart: Solving Helmholtz 1D with VQLS & Local Cost
```python
from qpdesolver.solvers.vqls import VQLSSolver
from qpdesolver.solvers.vqls.costs import LocalCost
from qpdesolver.ansatze import RealAmplitudes
from qpdesolver.optimizers.first_order import Adam

solver = VQLSSolver(
    ansatz=RealAmplitudes(layers=2),
    cost=LocalCost(),
    optimizer=Adam(lr=0.05),
    max_iter=50,
)
result = solver.solve(system)
print(f"VQLS Solution Fidelity: {result.fidelity*100:.2f}%")
```

---

## 4. Full Waveform Inversion (FWI)

The FWI module integrates frequency-domain Helmholtz wavefield modeling with the adjoint state method:

```python
from qpdesolver.io import load_fwi_dataset
from qpdesolver.applications.fwi import FWIProblem, FWIInversion

# Load synthetic acquisition and survey data from HDF5
dataset = load_fwi_dataset("examples/data/toy_fwi.h5", group="fwi_synthetic_8pt")

# Setup FWI problem
problem = FWIProblem(acquisition=dataset.acquisition, n_points=8, h=dataset.h)

# Run iterative inversion starting from homogeneous background
inversion = FWIInversion(problem=problem, dataset=dataset, max_iter=15)
result = inversion.run()

print(f"Initial Misfit: {result.misfit_history[0]:.4e}")
print(f"Final Misfit:   {result.final_misfit:.4e}")
```

---

## 5. Running Tests & Benchmarks

Run the complete test suite:
```bash
pytest -v
```

Generate comparison and trainability reports:
```bash
python scripts/run_comparison.py
python scripts/run_trainability.py
```
Reports and figures will be available in `reports/`.

---

## 6. Academic Documentation & Developer Tutorials

For researchers and students working with `qpdesolver`, detailed mathematical and software tutorials are available in the [`docs/`](docs/) directory:

- **[Visual Walkthrough & Educational Guide (Illustrated)](docs/VISUAL_TUTORIAL.md)**:
  A highly visual, step-by-step educational guide with circuit diagrams, finite-difference stencil graphics, FWI adjoint flowcharts, and trainability manifold illustrations.
- **[Theoretical & Mathematical Foundations](docs/THEORY_AND_ALGORITHMS.md)**:
  Complete analytical derivations of HHL, VQLS (Global/Local costs), FWI Helmholtz adjoint state calculus, Barren Plateau scaling, Dynamical Lie Algebras $\dim(\mathfrak{g})$, and Quantum Fisher Information Metric (QFIM) rank.
- **[Creating Custom Ansatze Tutorial](docs/CUSTOM_ANSATZ_GUIDE.md)**:
  Step-by-step instructions on designing parameterized quantum circuits, defining layer slicing, Lie generator definitions for DLA, and integrating with Quantum Natural Gradient (QNG).
- **[Implementing Custom PDE Problems Tutorial](docs/CUSTOM_PDE_PROBLEMS.md)**:
  Guide on creating novel spatial discretizations (`PDEDiscretization`), assembling `LinearSystem` objects, Pauli decomposition, Hermitian dilation, and HDF5 persistence.
- **[Custom Cost Functions & Optimizers Guide](docs/CUSTOM_COSTS_AND_OPTIMIZERS.md)**:
  How to formulate regularized or localized VQLS loss functions and develop custom classical/Riemannian optimizers.
- **[End-to-End Researcher Walkthrough](examples/09_researcher_walkthrough_custom_ansatz_and_pde.py)**:
  A standalone executable script demonstrating the creation of a new PDE, a new ansatz, solving it with HHL and VQLS, computing DLA and QFIM, and saving to HDF5.
- **[Porting & Migration Guide](PORTING.md)**:
  Detailed comparison with `17-Jul-2026`, showcasing architectural refactoring and numerical parity proofs.
