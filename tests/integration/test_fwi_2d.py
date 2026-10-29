import pytest
import numpy as np
from qpdesolver.applications.fwi.model import VelocityModel, Acquisition, FWIDataset, VelocityModel2D, Acquisition2D, FWIDataset2D
from qpdesolver.applications.fwi.problem import FWIProblem, FWIProblem2D
from qpdesolver.applications.fwi.inversion import FWIInversion
from qpdesolver.solvers.classical import ClassicalSolver
from qpdesolver.solvers.vqls import VQLSSolver
from qpdesolver.ansatze import RealAmplitudes
from qpdesolver.optimizers.scipy_wrapper import ScipyOptimizer


def test_velocity_model_2d_properties_and_resample():
    v = np.array([[2000.0, 2200.0], [2100.0, 2300.0]])
    model = VelocityModel2D(velocity=v, hx=10.0, hz=12.0)

    assert model.shape == (2, 2)
    assert model.nx == 2
    assert model.nz == 2
    assert model.n_points == 4
    assert np.allclose(model.slowness_squared, 1.0 / (v**2))
    assert model.flat_velocity.shape == (4,)

    # Resample to (4, 4) with smoothing
    resampled = model.resample((4, 4), smooth_sigma=0.5)
    assert resampled.shape == (4, 4)
    assert resampled.n_points == 16
    assert np.all(resampled.velocity > 0.0)

    # Validation errors
    with pytest.raises(ValueError):
        VelocityModel2D(velocity=np.ones(4))  # 1D instead of 2D
    with pytest.raises(ValueError):
        VelocityModel2D(velocity=np.array([[-100.0, 2000.0], [2000.0, 2000.0]]))  # negative
    with pytest.raises(ValueError):
        VelocityModel2D(velocity=np.array([[np.nan, 2000.0], [2000.0, 2000.0]]))  # non-finite


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

    # Numerical finite difference check on one cell using central differences
    eps = 1e-8
    m_plus = m_test.copy()
    m_plus[0, 1] += eps
    m_minus = m_test.copy()
    m_minus[0, 1] -= eps
    fd = (problem.misfit(m_plus, d_obs) - problem.misfit(m_minus, d_obs)) / (2.0 * eps)
    assert np.isclose(grad_m[0, 1], fd, rtol=1e-3)

    # Check gradient w.r.t velocity v
    grad_v = problem.gradient_v(v_test, d_obs)
    assert grad_v.shape == (nx, nz)
    v_plus = v_test.copy()
    v_plus[0, 1] += 0.01
    v_minus = v_test.copy()
    v_minus[0, 1] -= 0.01
    fd_v = (problem.misfit(1.0 / (v_plus**2), d_obs) - problem.misfit(1.0 / (v_minus**2), d_obs)) / 0.02
    assert np.isclose(grad_v[0, 1], fd_v, rtol=1e-3)


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
    assert isinstance(res.final_velocity, VelocityModel2D)
    # Check that warm start cache was populated
    assert len(problem._warm_start_cache) > 0
    assert any(k[0] == "forward" for k in problem._warm_start_cache.keys())


def test_fwi_1d_warm_start_caching():
    # Test 1D FWIProblem with warm start enabled
    acq = Acquisition(sources=[1], receivers=[2, 3], omegas=[1.0])
    v_solver = VQLSSolver(ansatz=RealAmplitudes(layers=2), max_iter=15, seed=42)
    problem = FWIProblem(acquisition=acq, n_points=4, h=5.0, solver=v_solver, warm_start_solves=True)

    m = np.full(4, 1.0 / (2000.0**2))
    u = problem.solve_forward(m, 1.0, 1)
    assert ("forward", 1, 1.0) in problem._warm_start_cache
    assert len(problem._warm_start_cache[("forward", 1, 1.0)]) == v_solver.ansatz.num_params(2)
