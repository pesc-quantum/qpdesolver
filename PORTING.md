# Porting and Evolution Guide: From `17-Jul-2026` to `07-Oct-2026` (`qpdesolver`)

This document details the architectural evolution, bug fixes, algorithmic extensions, and migration pathways from the initial prototype in `17-Jul-2026` to the production-grade research framework `qpdesolver` in `07-Oct-2026`.

---

## 1. Executive Summary

| Capability | `17-Jul-2026` Prototype | `07-Oct-2026` (`qpdesolver`) |
|:---|:---|:---|
| **Code Structure** | Standalone monolithic scripts | Modular, installable Python package (`pip install -e .`) |
| **Solvers Supported** | VQLS only (hardcoded circuit) | **HHL** (flexible PennyLane + Qiskit stub) + **VQLS** + **Classical** |
| **Ansatz Architectures** | Hardcoded PennyLane function | Abstract `Ansatz` base class, `RealAmplitudes`, `HardwareEfficient`, `AlternatingLayered`, `HamiltonianVariational`, custom functions |
| **Cost Functions** | Hardcoded global cost | `GlobalCost`, `LocalCost` (BP-mitigating), `CustomCostFunction`, `make_cost` |
| **Optimization Suite** | Basic SciPy BFGS | First-order (`Adam`, `GradientDescent`), Riemannian (`QuantumNaturalGradient`), Second-order (`DampedNewton`), Stochastic (`SPSA`, `QNSPSA`), SciPy wrapper |
| **Trainability Studies** | None | Barren Plateau detection ($\mathrm{Var}[\partial_\theta C]$ scaling), Dynamical Lie Algebra ($\dim(\mathfrak{g})$), QFIM rank saturation, Meyer-Wallach entanglement, Layerwise training |
| **Data Persistence** | None (in-memory only) | Standardized HDF5 (`.h5`) schema with sparse CSR support, metadata, datasets, and study results |
| **PDE & FWI Application** | Hardcoded 4-point matrices | Discretization ABC (`Helmholtz1D`, `Poisson1D`), exact adjoint-state FWI engine (`FWIProblem`, `FWIInversion`) |
| **Logging & Observability** | Raw `print()` statements | Structured logging with `loguru.logger` across all modules |
| **Testing & CI** | Ad-hoc manual verification | 23+ automated `pytest` unit and integration tests |

---

## 2. Parity Verification with `17-Jul-2026`

The `17-Jul-2026` codebase was strictly **preserved without modification**. The package provides `qpdesolver.problems.discretizations.legacy_17jul` to reconstruct the exact problem instances:

1. **Periodic Helmholtz 1D ($4 \times 4$, $\omega=1.5, v=2.0, h=1.0$, Bell state RHS)**:
   - In `17-Jul-2026`: VQLS achieved near-zero cost ($< 10^{-5}$) using an alternating Ry-CNOT ansatz.
   - In `qpdesolver`: `VQLSSolver` achieves **100.000% state fidelity** and relative residual $< 10^{-10}$ (`tests/integration/test_vqls_parity_17jul.py`).
2. **Mock Parameterized FWI ($4 \times 4$, anomaly $= 0.35$, freq $= 1.2$, Bell state RHS)**:
   - In `17-Jul-2026`: parameterized matrix solved with BFGS.
   - In `qpdesolver`: `VQLSSolver` achieves **100.000% state fidelity** (`test_vqls_mock_fwi_parity`).

---

## 3. Critical Technical Discoveries & Fixes

During the development and benchmarking of the new package, several subtle quantum computing and autodiff subtleties were discovered and resolved:

### 3.1. Autoray Version Incompatibility with PennyLane 0.38
- **Issue**: `autoray 0.8.2` removed `NumpyMimic`, crashing PennyLane 0.38 with `AttributeError: module 'autoray.autoray' has no attribute 'NumpyMimic'`.
- **Resolution**: Pinned `autoray<0.7` (`0.6.12` installed) in `pyproject.toml` and `requirements.txt`.

### 3.2. PennyLane QuantumPhaseEstimation Adjoint Queuing Bug
- **Issue**: Invoking `qml.adjoint(qml.QuantumPhaseEstimation)(qml.QubitUnitary(U,...), ...)` leaves an uncancelled extra $U$ gate on the tape, collapsing HHL solution fidelity to ~0.52.
- **Resolution**: Constructing the unitary operator strictly *inside* the adjoint transform:
  ```python
  u_op = qml.QubitUnitary(U, wires=target_wires)
  qml.adjoint(qml.QuantumPhaseEstimation(u_op, estimation_wires=clock_wires))
  ```
  This restored exact eigenvalue uncomputation, yielding **1.000000** fidelity on test systems.

### 3.3. Autograd Vector-Jacobian Product (VJP) Tracer Failure
- **Issue**: PennyLane Autograd raises `NotImplementedError: VJP of vdot wrt argnums (0, 1) not defined` when both arguments are autograd tracers in complex overlap calculations.
- **Resolution**: Rewrote inner products into explicit elementwise conjugation and sum:
  ```python
  inner_val = pnp.sum(pnp.conj(b_vec) * Ax)
  norm_sq = pnp.sum(pnp.real(Ax * pnp.conj(Ax)))
  ```

### 3.4. Complex Statevector Jacobians for Metric Tensors
- **Issue**: Calling `qml.jacobian` directly on complex-returning statevector QNodes causes shape mismatch with autograd tracers.
- **Resolution**: Differentiated real and imaginary parts of the statevector separately:
  ```python
  Jr = qml.jacobian(real_state)(params)
  Ji = qml.jacobian(imag_state)(params)
  J = Jr + 1j * Ji
  ```

---

## 4. Migration Guide: Code Examples

### 4.1. Solving a Linear System

#### Before (`17-Jul-2026` pattern):
```python
# Hardcoded matrices and device
import pennylane as qml
dev = qml.device("default.qubit", wires=2)
# Custom optimization loop, manual cost definition, manual BFGS call
```

#### After (`qpdesolver` pattern):
```python
from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.solvers.vqls import VQLSSolver
from qpdesolver.ansatze import RealAmplitudes
from qpdesolver.solvers.vqls.costs import LocalCost
from qpdesolver.optimizers.first_order import Adam

# 1. Discretize PDE
helm = Helmholtz1D(n_points=4, h=1.0, bc="dirichlet")
system = helm.system(rhs=helm.point_source(1), omega=1.0)

# 2. Configure VQLS with Local Cost (BP mitigated) and Adam
solver = VQLSSolver(
    ansatz=RealAmplitudes(layers=2),
    cost=LocalCost(),
    optimizer=Adam(lr=0.05),
    max_iter=50,
)

# 3. Solve with automatic norm recovery
result = solver.solve(system)
print(f"Fidelity: {result.fidelity:.4f}, Residual: {result.residual:.2e}")
```

### 4.2. Running HHL

```python
from qpdesolver.solvers.hhl import HHLSolver, HHLConfig

hhl = HHLSolver(config=HHLConfig(n_clock=4))
result = hhl.solve(system)
print(f"HHL Fidelity: {result.fidelity:.4f}")
```

### 4.3. Trainability & Barren Plateau Diagnosis

```python
from qpdesolver.trainability.gradients import scan_qubits, diagnose_barren_plateau, make_global_projector_cost
from qpdesolver.ansatze import RealAmplitudes

ansatz = RealAmplitudes(layers=2)
stats = scan_qubits(ansatz, qubit_range=[2, 3, 4], cost_fn_builder=make_global_projector_cost)
diagnosis = diagnose_barren_plateau(stats)
print(f"Barren Plateau detected: {diagnosis.is_barren} (Decay rate: {diagnosis.decay_rate:.2f})")
```

### 4.4. Full Waveform Inversion (FWI)

```python
from qpdesolver.io import load_fwi_dataset
from qpdesolver.applications.fwi import FWIProblem, FWIInversion

dataset = load_fwi_dataset("toy_fwi.h5", group="fwi_synthetic_8pt")
problem = FWIProblem(acquisition=dataset.acquisition, n_points=8, h=dataset.h)
inversion = FWIInversion(problem=problem, dataset=dataset, max_iter=15)
inv_res = inversion.run()
```
