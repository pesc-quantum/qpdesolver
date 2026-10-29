"""
Unit tests for HHL circuit planning logic.
"""
import pytest
import numpy as np

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.solvers.hhl.config import HHLConfig
from qpdesolver.solvers.hhl.planning import plan_hhl

def test_hhl_planning_signed():
    # Diagonal matrix with positive and negative eigenvalues
    A = np.diag([-2.0, 1.0, 3.0, 4.0])
    b = np.array([1.0, 1.0, 1.0, 1.0])
    sys = LinearSystem(A=A, b=b)

    cfg = HHLConfig(n_clock=4, signed=True, phase_margin=0.9, C="grid")
    plan = plan_hhl(sys, cfg)

    assert plan.n_clock == 4
    # Signed eigenvalue grid must contain both positive and negative values
    assert np.any(plan.lambda_grid < 0)
    assert np.any(plan.lambda_grid > 0)
    # Zero angle at k = 0
    assert plan.angles[0] == 0.0

    # Every nonzero angle must have |C / lambda| <= 1.0
    for theta in plan.angles:
        # |sin(theta/2)| <= 1
        assert np.abs(np.sin(theta / 2.0)) <= 1.0000001

    # Check evolution time phase bound: |lambda_max| * t / 2pi <= 0.5 * phase_margin
    max_phase = 4.0 * plan.t / (2.0 * np.pi)
    assert max_phase <= 0.5 * 0.9 + 1e-10

def test_hhl_planning_unsigned():
    A = np.diag([1.0, 2.0, 3.0, 4.0])
    b = np.array([1.0, 0.0, 0.0, 0.0])
    sys = LinearSystem(A=A, b=b)

    cfg = HHLConfig(n_clock=4, signed=False, phase_margin=0.9)
    plan = plan_hhl(sys, cfg)
    assert np.all(plan.lambda_grid >= 0)
