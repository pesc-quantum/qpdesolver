"""
Unit tests for HHL quantum kernel: exact spectrum regression, Trotterization, and auto-dilation.
"""
import pytest
import numpy as np
import scipy.linalg
from scipy.stats import ortho_group

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.solvers.hhl.config import HHLConfig
from qpdesolver.solvers.hhl.solver import HHLSolver
from qpdesolver.backends.pennylane_backend import PennyLaneBackend
from qpdesolver.core.metrics import fidelity

def test_hhl_exact_spectrum_regression():
    """
    REGRESSION TEST:
    A matrix with eigenvalues lying exactly on the QPE grid [-2, 1, 3, 4].
    Must achieve fidelity > 0.999999 and exact norm recovery.
    Verifies that bit-reversal, sign convention, and QPE adjoint queuing are exact.
    """
    Q = ortho_group.rvs(4, random_state=42)
    lam = np.array([-2.0, 1.0, 3.0, 4.0])
    A = Q @ np.diag(lam) @ Q.T
    b = np.array([1.0, 0.5, -0.2, 0.8])
    sys = LinearSystem(A=A, b=b, name="exact_spectrum_sys")

    n_c = 4
    # Spacing = 1.0 -> grid integer eigenvalues
    t = 2.0 * np.pi / (1.0 * (2**n_c))
    cfg = HHLConfig(
        n_clock=n_c,
        signed=True,
        evolution="exact",
        C=1.0,
        spectral_bounds=(1.0, 4.0),
    )
    # Monkey-patch or override time by passing spectral bounds that yield exact t
    # t = 2*pi * 0.5 * margin / lam_max => with margin=1.0, lam_max=2^(n_c-1) = 8: t = 2*pi*0.5*1.0/8 = 2*pi/16
    cfg.phase_margin = 1.0
    cfg.spectral_bounds = (1.0, 8.0)

    solver = HHLSolver(config=cfg, backend=PennyLaneBackend())
    result = solver.solve(sys)

    assert result.fidelity is not None
    assert result.fidelity > 0.99999, f"Expected fidelity ~ 1.0, got {result.fidelity}"
    assert result.residual < 1e-4

    # Native norm estimate should match classical norm
    exact_x = sys.classical_solution()
    native_norm = result.metadata["native_norm_estimate"]
    exact_norm = np.linalg.norm(exact_x)
    assert np.isclose(native_norm, exact_norm, rtol=1e-4)

def test_hhl_trotter_agreement():
    """Verifies that Trotterized HHL converges close to exact evolution."""
    A = np.array([
        [2.0, -0.5, 0.0, -0.5],
        [-0.5, 2.0, -0.5, 0.0],
        [0.0, -0.5, 2.0, -0.5],
        [-0.5, 0.0, -0.5, 2.0]
    ], dtype=complex)
    b = np.array([1.0, 0.0, 0.0, 1.0], dtype=complex)
    sys = LinearSystem(A=A, b=b)

    cfg_exact = HHLConfig(n_clock=5, evolution="exact", signed=True)
    cfg_trotter = HHLConfig(n_clock=5, evolution="trotter", trotter_steps=30, trotter_order=2, signed=True)

    solver_exact = HHLSolver(config=cfg_exact)
    solver_trotter = HHLSolver(config=cfg_trotter)

    res_exact = solver_exact.solve(sys)
    res_trotter = solver_trotter.solve(sys)

    # Overlap between Trotter state and exact state
    trot_fid = fidelity(res_exact.state, res_trotter.state)
    assert trot_fid > 0.99, f"Trotter fidelity vs exact state: {trot_fid}"

def test_hhl_auto_dilation():
    """Verifies solving a non-Hermitian 2x2 system via automatic Hermitian dilation."""
    A = np.array([[2.0, 1.0], [0.0, 2.0]], dtype=complex) # upper triangular, non-Hermitian
    b = np.array([1.0, 2.0], dtype=complex)
    sys = LinearSystem(A=A, b=b, name="non_herm_2x2")

    cfg = HHLConfig(n_clock=5, auto_dilate=True)
    solver = HHLSolver(config=cfg)
    result = solver.solve(sys)

    assert result.metadata["dilated"] is True
    assert result.fidelity > 0.95
    assert result.residual < 0.25
