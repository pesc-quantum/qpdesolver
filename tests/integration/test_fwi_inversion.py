"""Integration tests for FWI Inversion optimization engine."""

import numpy as np
import pytest

from qpdesolver.applications.fwi.model import VelocityModel, Acquisition
from qpdesolver.applications.fwi.problem import FWIProblem
from qpdesolver.applications.fwi.inversion import FWIInversion
from qpdesolver.optimizers.scipy_wrapper import ScipyOptimizer
from qpdesolver.optimizers.first_order import Adam


def test_fwi_inversion_lbfgs_convergence():
    """Verify L-BFGS-B minimizes misfit on a small 1D velocity anomaly."""
    n_points = 8
    h = 5.0
    acq = Acquisition(
        sources=[1, 4],
        receivers=[2, 3, 5, 6],
        omegas=[1.0],
    )
    problem = FWIProblem(acquisition=acq, n_points=n_points, h=h, bc="dirichlet")

    v_true = 2000.0 * np.ones(n_points)
    v_true[3:5] = 2400.0
    true_model = VelocityModel(velocity=v_true, h=h)

    v_init = 2000.0 * np.ones(n_points)
    init_model = VelocityModel(velocity=v_init, h=h)

    dataset = problem.simulate_dataset(true_model=true_model, initial_model=init_model)

    inv = FWIInversion(
        problem=problem,
        dataset=dataset,
        optimizer=ScipyOptimizer(method="L-BFGS-B"),
        parameterization="v",
        bounds=(1500.0, 3000.0),
        max_iter=12,
    )

    result = inv.run(init_model)

    assert result.final_misfit < result.misfit_history[0]
    # Misfit should decrease significantly (by at least 80%)
    rel_reduction = (result.misfit_history[0] - result.final_misfit) / result.misfit_history[0]
    assert rel_reduction > 0.80
    assert len(result.model_history) >= 1
    # Recovered velocity should show higher velocity around anomaly
    v_rec = result.final_velocity.velocity
    assert v_rec[3] > 2050.0 or v_rec[4] > 2050.0


def test_fwi_inversion_adam_optimizer():
    """Verify native Adam optimizer runs and decreases misfit on FWI."""
    n_points = 8
    h = 5.0
    acq = Acquisition(
        sources=[2],
        receivers=[1, 3, 5],
        omegas=[0.8],
    )
    problem = FWIProblem(acquisition=acq, n_points=n_points, h=h, bc="dirichlet")

    v_true = 1800.0 * np.ones(n_points)
    v_true[4] = 2200.0
    true_model = VelocityModel(velocity=v_true, h=h)

    v_init = 1800.0 * np.ones(n_points)
    init_model = VelocityModel(velocity=v_init, h=h)
    dataset = problem.simulate_dataset(true_model=true_model, initial_model=init_model)

    inv = FWIInversion(
        problem=problem,
        dataset=dataset,
        optimizer=Adam(lr=15.0),
        parameterization="v",
        bounds=(1200.0, 2800.0),
        max_iter=15,
    )

    result = inv.run(init_model)
    assert result.final_misfit <= result.misfit_history[0]
