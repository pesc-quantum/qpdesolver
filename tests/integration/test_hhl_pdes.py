"""
Integration tests: solving PDE linear systems with HHL.
"""
import pytest
import numpy as np

from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.problems.discretizations.poisson import Poisson1D
from qpdesolver.problems.discretizations.legacy_17jul import get_legacy_17jul_helmholtz
from qpdesolver.solvers.hhl.config import HHLConfig
from qpdesolver.solvers.hhl.solver import HHLSolver
from qpdesolver.io import save_results, load_results

def test_hhl_helmholtz_dirichlet_4points():
    hz = Helmholtz1D(n_points=4, h=1.0, bc="dirichlet")
    # Medium with two layers
    v = np.array([2.0, 2.0, 2.5, 2.5])
    m = 1.0 / (v**2)
    omega = 1.0
    sys = hz.system(rhs=hz.point_source(0), m=m, omega=omega)

    # 6 clock qubits yields ~0.999 fidelity
    cfg = HHLConfig(n_clock=6, signed=True, evolution="exact", phase_margin=0.9)
    solver = HHLSolver(config=cfg)
    result = solver.solve(sys)

    assert result.fidelity is not None
    assert result.fidelity > 0.98, f"Got fidelity {result.fidelity}"
    assert result.success_probability > 0.01

def test_hhl_legacy_17jul_helmholtz():
    sys = get_legacy_17jul_helmholtz(omega=1.5, velocity=2.0, h=1.0)
    cfg = HHLConfig(n_clock=6, signed=True, evolution="exact")
    solver = HHLSolver(config=cfg)
    result = solver.solve(sys)

    assert result.fidelity > 0.95
    assert result.success_probability > 0.01

def test_hhl_poisson_8points_and_hdf5_save(tmp_path):
    poisson = Poisson1D(n_points=8, h=1.0, bc="dirichlet")
    # Central delta source
    b = poisson.point_source(3)
    sys = poisson.system(rhs=b, kappa=1.0)

    cfg = HHLConfig(n_clock=6, signed=False, evolution="exact")
    solver = HHLSolver(config=cfg)
    result = solver.solve(sys)

    assert result.fidelity > 0.95

    # Test saving solve results to HDF5
    h5_file = tmp_path / "hhl_results.h5"
    save_results(h5_file, "poisson_hhl_run", result.to_dict(), config={"n_clock": 6})
    loaded_data, loaded_cfg = load_results(h5_file, "poisson_hhl_run")

    assert np.allclose(loaded_data["state"], result.state)
    assert np.isclose(loaded_data["fidelity"], result.fidelity)
    assert loaded_cfg["n_clock"] == 6
