"""
Unit tests for 2D Helmholtz discretization (Helmholtz2D).
"""
import pytest
import numpy as np

from qpdesolver.problems.discretizations.helmholtz2d import Helmholtz2D
from qpdesolver.core.registry import REGISTRY_DISCRETIZATIONS


def test_helmholtz2d_dimension_validation():
    with pytest.raises(ValueError, match="positive power of 2"):
        Helmholtz2D(nx=3, nz=4)

    with pytest.raises(ValueError, match="positive power of 2"):
        Helmholtz2D(nx=4, nz=5)

    with pytest.raises(ValueError, match="positive power of 2"):
        Helmholtz2D(nx=0, nz=4)

    with pytest.raises(ValueError, match="positive power of 2"):
        Helmholtz2D(nx=4, nz=-2)

    with pytest.raises(ValueError, match="Unsupported boundary condition"):
        Helmholtz2D(nx=4, nz=4, bc="cauchy")


def test_helmholtz2d_matrix_properties():
    # 2x2 grid = 4 points (2 qubits)
    h2d = Helmholtz2D(nx=2, nz=2, hx=1.0, hz=1.0, bc="dirichlet")
    assert h2d.n_points == 4
    assert h2d.n_qubits == 2
    assert h2d.nx == 2
    assert h2d.nz == 2
    assert h2d.hx == 1.0
    assert h2d.hz == 1.0

    # Verify operator symmetry and dimension
    m = np.ones((2, 2))
    A = h2d.operator(m=m, omega=1.0)
    assert A.shape == (4, 4)
    assert np.allclose(A, A.conj().T)

    # Point source 2D
    b = h2d.point_source_2d(ix=1, iz=0)
    assert b.shape == (4,)
    assert b[2] == 1.0
    assert np.sum(b) == 1.0

    # Out-of-bounds point source checks
    with pytest.raises(IndexError, match="Source ix"):
        h2d.point_source_2d(ix=2, iz=0)

    with pytest.raises(IndexError, match="Source iz"):
        h2d.point_source_2d(ix=0, iz=-1)


def test_helmholtz2d_linear_system_creation():
    h2d = Helmholtz2D(nx=4, nz=4, hx=5.0, bc="dirichlet")
    b = h2d.point_source_2d(2, 2)
    sys = h2d.system(rhs=b, omega=1.5, m=np.full((4, 4), 1.0 / 2000.0**2))
    assert sys.size == 16
    assert sys.n_qubits == 4
    assert sys.is_hermitian
    assert sys.metadata["nx"] == 4
    assert sys.metadata["nz"] == 4
    assert sys.metadata["hx"] == 5.0
    assert sys.metadata["hz"] == 5.0


def test_helmholtz2d_laplacian_boundary_conditions():
    # Check Dirichlet 2x2
    h2d_dir = Helmholtz2D(nx=2, nz=2, hx=1.0, hz=1.0, bc="dirichlet")
    L_dir = h2d_dir.laplacian()
    assert L_dir.shape == (4, 4)
    assert np.allclose(L_dir, L_dir.T)
    # Stencil: 2/hx^2 + 2/hz^2 = 4 on diagonal, -1 on adjacent neighbors
    expected_L_dir = np.array([
        [4.0, -1.0, -1.0,  0.0],
        [-1.0, 4.0,  0.0, -1.0],
        [-1.0, 0.0,  4.0, -1.0],
        [ 0.0, -1.0, -1.0, 4.0],
    ])
    assert np.allclose(L_dir, expected_L_dir)

    # Check Periodic 4x4
    h2d_per = Helmholtz2D(nx=4, nz=4, hx=1.0, hz=1.0, bc="periodic")
    L_per = h2d_per.laplacian()
    assert L_per.shape == (16, 16)
    assert np.allclose(L_per, L_per.T)

    # Check Neumann 2x2
    h2d_neu = Helmholtz2D(nx=2, nz=2, hx=1.0, hz=1.0, bc="neumann")
    L_neu = h2d_neu.laplacian()
    assert L_neu.shape == (4, 4)
    assert np.allclose(L_neu, L_neu.T)


def test_helmholtz2d_slowness_and_velocity_inputs():
    h2d = Helmholtz2D(nx=2, nz=4, hx=2.0, hz=1.0)
    assert h2d.n_points == 8
    assert h2d.n_qubits == 3

    # 1. 2D array m
    m_2d = np.full((2, 4), 0.25)
    A1 = h2d.operator(m=m_2d, omega=2.0)
    assert A1.shape == (8, 8)

    # 2. 1D array m
    m_1d = np.full(8, 0.25)
    A2 = h2d.operator(m=m_1d, omega=2.0)
    assert np.allclose(A1, A2)

    # 3. Scalar m
    A3 = h2d.operator(m=0.25, omega=2.0)
    assert np.allclose(A1, A3)

    # 4. Velocity input
    A_vel = h2d.operator(velocity=2.0, omega=2.0)  # v=2.0 -> m=1/4=0.25
    assert np.allclose(A1, A_vel)

    # 5. Default velocity (v=2.0)
    A_def = h2d.operator(omega=2.0)
    assert np.allclose(A1, A_def)

    # 6. Derivative
    assert h2d.dA_dm_diag(omega=2.0) == -4.0

    # 7. Incompatible m shape
    with pytest.raises(ValueError, match="m shape"):
        h2d.operator(m=np.ones((3, 3)))


def test_helmholtz2d_row_major_ordering():
    nx, nz = 2, 4
    h2d = Helmholtz2D(nx=nx, nz=nz, hx=1.0, hz=1.0)
    m = np.zeros((nx, nz))
    # Place slowness anomaly only at (ix=1, iz=2) -> index 1 * 4 + 2 = 6
    m[1, 2] = 5.0
    omega = 3.0
    A = h2d.operator(m=m, omega=omega)
    L = h2d.laplacian()
    diff = L - A
    # The only non-zero diagonal element in L - A should be at index 6 with value omega^2 * 5.0 = 45.0
    diag_diff = np.diag(diff)
    expected_diag = np.zeros(8)
    expected_diag[6] = omega**2 * 5.0
    assert np.allclose(diag_diff, expected_diag)


def test_helmholtz2d_registry_and_export():
    assert "helmholtz2d" in REGISTRY_DISCRETIZATIONS
    assert REGISTRY_DISCRETIZATIONS.get("helmholtz2d") is Helmholtz2D

    from qpdesolver.problems.discretizations import Helmholtz2D as ExportedHelmholtz2D
    assert ExportedHelmholtz2D is Helmholtz2D
