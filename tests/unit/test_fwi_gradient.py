"""Unit test verifying that FWI adjoint gradient matches finite-difference gradient."""

import numpy as np
import pytest

from qpdesolver.applications.fwi.model import VelocityModel, Acquisition
from qpdesolver.applications.fwi.problem import FWIProblem


def test_fwi_adjoint_gradient_m():
    """Verify adjoint state gradient w.r.t slowness squared m matches central finite differences."""
    n_points = 8
    h = 10.0
    # Two sources, two frequencies, multiple receivers
    acq = Acquisition(
        sources=[1, 3],
        receivers=[2, 4, 6],
        omegas=[0.5, 1.2],
    )
    problem = FWIProblem(acquisition=acq, n_points=n_points, h=h, bc="dirichlet")

    # True model with a perturbation
    v_true = 2000.0 * np.ones(n_points)
    v_true[3:5] = 2500.0
    true_model = VelocityModel(velocity=v_true, h=h)
    dataset = problem.simulate_dataset(true_model)

    # Background model to test gradient at
    v_test = 2100.0 * np.ones(n_points)
    m_test = 1.0 / (v_test**2)

    grad_adj = problem.gradient_m(m_test, dataset.observed_data)

    # Numerical finite differences w.r.t m
    eps = 1e-8
    grad_num = np.zeros(n_points)
    for i in range(n_points):
        m_plus = m_test.copy()
        m_plus[i] += eps
        j_plus = problem.misfit(m_plus, dataset.observed_data)

        m_minus = m_test.copy()
        m_minus[i] -= eps
        j_minus = problem.misfit(m_minus, dataset.observed_data)

        grad_num[i] = (j_plus - j_minus) / (2.0 * eps)

    # Adjoint gradient should match finite difference to high precision
    rel_err = np.linalg.norm(grad_adj - grad_num) / (np.linalg.norm(grad_num) + 1e-15)
    assert rel_err < 1e-5, f"Adjoint gradient relative error {rel_err:.2e} exceeds threshold."


def test_fwi_adjoint_gradient_v():
    """Verify adjoint gradient w.r.t velocity v matches central finite differences."""
    n_points = 8
    h = 10.0
    acq = Acquisition(
        sources=[2],
        receivers=[1, 3, 5],
        omegas=[0.8],
    )
    problem = FWIProblem(acquisition=acq, n_points=n_points, h=h, bc="dirichlet")

    v_true = 1800.0 * np.ones(n_points)
    v_true[4] = 2300.0
    true_model = VelocityModel(velocity=v_true, h=h)
    dataset = problem.simulate_dataset(true_model)

    v_test = 1900.0 * np.ones(n_points)
    grad_v_adj = problem.gradient_v(v_test, dataset.observed_data)

    # Numerical finite differences w.r.t v
    eps = 1e-3
    grad_v_num = np.zeros(n_points)
    for i in range(n_points):
        v_plus = v_test.copy()
        v_plus[i] += eps
        j_plus = problem.misfit(1.0 / (v_plus**2), dataset.observed_data)

        v_minus = v_test.copy()
        v_minus[i] -= eps
        j_minus = problem.misfit(1.0 / (v_minus**2), dataset.observed_data)

        grad_v_num[i] = (j_plus - j_minus) / (2.0 * eps)

    rel_err = np.linalg.norm(grad_v_adj - grad_v_num) / (np.linalg.norm(grad_v_num) + 1e-15)
    assert rel_err < 1e-5, f"Velocity gradient relative error {rel_err:.2e} exceeds threshold."
