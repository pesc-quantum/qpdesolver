"""Integration tests loading FWI problem from HDF5 file and performing inversion."""

from pathlib import Path
import numpy as np
import pytest

from qpdesolver.io import load_fwi_dataset
from qpdesolver.applications.fwi.problem import FWIProblem
from qpdesolver.applications.fwi.inversion import FWIInversion
from qpdesolver.optimizers.scipy_wrapper import ScipyOptimizer


def test_fwi_workflow_from_h5():
    """Load dataset from toy_fwi.h5, build problem, compute gradient and run inversion."""
    data_path = Path(__file__).resolve().parents[2] / "examples" / "data" / "toy_fwi.h5"
    if not data_path.is_file():
        pytest.skip("toy_fwi.h5 not generated yet")

    dataset = load_fwi_dataset(data_path, group="fwi_synthetic_8pt")
    assert dataset.observed_data.ndim == 3
    assert dataset.initial_velocity is not None
    assert dataset.true_velocity is not None

    n_points = len(dataset.initial_velocity)
    problem = FWIProblem(
        acquisition=dataset.acquisition,
        n_points=n_points,
        h=dataset.h,
        bc=dataset.bc,
    )

    # Initial misfit and gradient
    m_init = 1.0 / (dataset.initial_velocity**2)
    j0 = problem.misfit(m_init, dataset.observed_data)
    grad0 = problem.gradient_v(dataset.initial_velocity, dataset.observed_data)
    assert j0 > 0.0
    assert np.any(grad0 != 0.0)

    # Run 5 iterations of L-BFGS-B
    inv = FWIInversion(
        problem=problem,
        dataset=dataset,
        optimizer=ScipyOptimizer(method="L-BFGS-B"),
        parameterization="v",
        bounds=(1500.0, 3200.0),
        max_iter=5,
    )
    result = inv.run()

    assert result.final_misfit < j0
    assert len(result.misfit_history) >= 1
