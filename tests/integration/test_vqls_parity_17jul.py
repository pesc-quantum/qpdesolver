"""
Integration test: VQLS parity with 17-Jul-2026 systems and multi-optimizer tests.
"""
import pytest
import numpy as np

from qpdesolver.problems.discretizations.legacy_17jul import (
    get_legacy_17jul_helmholtz,
    get_legacy_17jul_mock_fwi,
)
from qpdesolver.ansatze.library import RealAmplitudes
from qpdesolver.solvers.vqls.solver import VQLSSolver
from qpdesolver.optimizers.scipy_wrapper import ScipyOptimizer
from qpdesolver.optimizers.first_order import Adam
from qpdesolver.optimizers.riemannian import QuantumNaturalGradient

def test_vqls_parity_17jul_helmholtz():
    """VQLS solving the 17-Jul Helmholtz system with RealAmplitudes(2) and BFGS."""
    sys_h = get_legacy_17jul_helmholtz(omega=1.5, velocity=2.0, h=1.0)
    ansatz = RealAmplitudes(layers=2)
    solver = VQLSSolver(ansatz=ansatz, optimizer=ScipyOptimizer("BFGS"), max_iter=100, seed=42)

    res = solver.solve(sys_h)

    # In 17-Jul: fidelity was 100.000%, cost was ~1e-13
    assert res.fidelity is not None
    assert res.fidelity > 0.999, f"Expected fidelity > 0.999, got {res.fidelity}"
    assert res.residual < 0.05

def test_vqls_parity_17jul_mock_fwi():
    """VQLS solving the 17-Jul mock FWI system."""
    sys_m = get_legacy_17jul_mock_fwi(velocity_anomaly=0.35, freq=1.2)
    ansatz = RealAmplitudes(layers=2)
    solver = VQLSSolver(ansatz=ansatz, optimizer=ScipyOptimizer("BFGS"), max_iter=100, seed=42)

    res = solver.solve(sys_m)

    assert res.fidelity is not None
    assert res.fidelity > 0.999, f"Expected fidelity > 0.999, got {res.fidelity}"

@pytest.mark.slow
def test_vqls_multi_optimizers_helmholtz():
    """Tests that Adam and QNG can optimize VQLS on the Helmholtz problem."""
    sys_h = get_legacy_17jul_helmholtz(omega=1.5, velocity=2.0, h=1.0)
    ansatz = RealAmplitudes(layers=2)

    # 1. Adam
    solver_adam = VQLSSolver(ansatz=ansatz, optimizer=Adam(lr=0.08), max_iter=120, seed=42)
    res_adam = solver_adam.solve(sys_h)
    assert res_adam.fidelity > 0.95

    # 2. QNG
    solver_qng = VQLSSolver(ansatz=ansatz, optimizer=QuantumNaturalGradient(lr=0.05), max_iter=80, seed=42)
    res_qng = solver_qng.solve(sys_h)
    assert res_qng.fidelity > 0.95
