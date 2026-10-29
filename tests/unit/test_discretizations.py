"""
Unit tests for PDE discretizations (Helmholtz, Poisson, 17-Jul legacy systems).
"""
import pytest
import numpy as np

from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.problems.discretizations.poisson import Poisson1D
from qpdesolver.problems.discretizations.legacy_17jul import (
    get_legacy_17jul_helmholtz,
    get_legacy_17jul_mock_fwi,
)
from qpdesolver.core.registry import REGISTRY_DISCRETIZATIONS

def test_registry_has_discretizations():
    assert "helmholtz1d" in REGISTRY_DISCRETIZATIONS
    assert "poisson1d" in REGISTRY_DISCRETIZATIONS

def test_helmholtz1d_dirichlet_matrix():
    h = 0.5
    h2 = h**2
    hz = Helmholtz1D(n_points=4, h=h, bc="dirichlet")
    m = np.array([1.0, 1.0, 2.0, 2.0])
    omega = 2.0
    A = hz.operator(m=m, omega=omega)

    # Expected Dirichlet 1D Laplacian
    # 2/h2 on diagonal, -1/h2 on off-diagonal
    diag_lap = 2.0 / h2
    off_lap = -1.0 / h2
    expected_A = np.array([
        [diag_lap - omega**2 * m[0], off_lap, 0.0, 0.0],
        [off_lap, diag_lap - omega**2 * m[1], off_lap, 0.0],
        [0.0, off_lap, diag_lap - omega**2 * m[2], off_lap],
        [0.0, 0.0, off_lap, diag_lap - omega**2 * m[3]],
    ], dtype=complex)

    assert np.allclose(A, expected_A)
    assert hz.dA_dm_diag(omega=omega) == -4.0

def test_poisson1d_analytical_spectrum():
    # Test analytical eigenvalues for constant kappa=1.0, Dirichlet BC
    N = 8
    h = 1.0
    poisson = Poisson1D(n_points=N, h=h, bc="dirichlet")
    A = poisson.operator(kappa=1.0)

    numerical_evals = np.sort(np.linalg.eigvalsh(A.real))
    analytical_evals = np.sort(poisson.analytical_eigenvalues())

    assert np.allclose(numerical_evals, analytical_evals, atol=1e-10)

def test_legacy_17jul_systems():
    # 1. 17-Jul periodic Helmholtz system
    sys_h = get_legacy_17jul_helmholtz(omega=1.5, velocity=2.0, h=1.0)
    assert sys_h.size == 4
    assert sys_h.n_qubits == 2
    assert sys_h.is_hermitian is True

    # Exact 17-Jul values check:
    # diag = 2.0 - 1.5^2 / 4.0 = 2.0 - 0.5625 = 1.4375
    # off-diag = -1.0
    # periodic corners = -1.0
    assert np.isclose(sys_h.A[0, 0], 1.4375)
    assert np.isclose(sys_h.A[0, 1], -1.0)
    assert np.isclose(sys_h.A[0, 3], -1.0)

    # 2. 17-Jul mock FWI system
    sys_m = get_legacy_17jul_mock_fwi(velocity_anomaly=0.35, freq=1.2)
    assert sys_m.size == 4
    assert sys_m.n_qubits == 2
    assert np.isclose(sys_m.A[0, 0], 1.5 + 0.2 * 0.35)
    assert np.isclose(sys_m.A[0, 1], 0.4 * 1.2)
    assert np.isclose(sys_m.A[0, 3], 0.1)


def test_spectral_and_finite_element_discretizations():
    from qpdesolver.problems.discretizations.spectral import FourierSpectral1D
    from qpdesolver.problems.discretizations.finite_element import FiniteElement1D

    # Test Fourier Spectral (modal and physical)
    spectral = FourierSpectral1D(n_points=4, L=2.0 * np.pi)
    L_modal = spectral.laplacian_modal()
    # Modal Laplacian is diagonal
    assert np.allclose(L_modal, np.diag(np.diag(L_modal)))
    A_modal = spectral.operator(omega=1.0, basis="modal")
    assert A_modal.shape == (4, 4)

    A_phys = spectral.operator(omega=1.0, basis="physical")
    assert A_phys.shape == (4, 4)
    # Physical pseudospectral operator should be symmetric for periodic domain
    assert np.allclose(A_phys, A_phys.T)

    # Test Finite Element Method (Galerkin)
    fem = FiniteElement1D(n_points=4, h=1.0, bc="dirichlet")
    K = fem.stiffness_matrix()
    M = fem.mass_matrix()
    assert K.shape == (4, 4)
    assert M.shape == (4, 4)
    assert np.allclose(K, K.T)
    assert np.allclose(M, M.T)

    sys_fem = fem.system(rhs=fem.point_source(1), omega=1.0)
    assert sys_fem.size == 4
    assert sys_fem.n_qubits == 2
    assert sys_fem.is_hermitian is True
