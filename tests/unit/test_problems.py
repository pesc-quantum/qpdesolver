"""
Unit tests for LinearSystem and Pauli decomposition.
"""
import pytest
import numpy as np

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.problems.pauli import pauli_matrix, pauli_decompose
from qpdesolver.core.registry import Registry

def test_registry_operations():
    reg = Registry("test_reg")
    @reg.register("item_a")
    def create_a(val=10):
        return val * 2

    assert "item_a" in reg
    assert reg.available() == ["item_a"]
    assert reg.create("item_a", val=5) == 10

    with pytest.raises(ValueError, match="already registered"):
        @reg.register("item_a")
        def create_a_duplicate():
            pass

    with pytest.raises(KeyError, match="not found"):
        reg.get("nonexistent")

def test_pauli_matrix_properties():
    # Dimension and identities
    assert np.allclose(pauli_matrix("I"), np.eye(2))
    assert np.allclose(pauli_matrix("II"), np.eye(4))
    assert np.allclose(pauli_matrix("Z"), np.diag([1, -1]))
    
    # Hermitian and unitary
    for lbl in ["IX", "YZ", "ZY", "ZZ"]:
        p = pauli_matrix(lbl)
        assert np.allclose(p @ p.conj().T, np.eye(4))
        assert np.allclose(p, p.conj().T)

def test_pauli_decompose_roundtrip(rng):
    # Random 4x4 Hermitian matrix
    H_rand = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    H_herm = 0.5 * (H_rand + H_rand.conj().T)

    terms = pauli_decompose(H_herm)
    recon = np.zeros((4, 4), dtype=complex)
    for coeff, label in terms:
        recon += coeff * pauli_matrix(label)

    assert np.allclose(H_herm, recon, atol=1e-10)

    # Random 8x8 non-Hermitian matrix
    M_rand = rng.normal(size=(8, 8)) + 1j * rng.normal(size=(8, 8))
    terms_m = pauli_decompose(M_rand)
    recon_m = np.zeros((8, 8), dtype=complex)
    for coeff, label in terms_m:
        recon_m += coeff * pauli_matrix(label)

    assert np.allclose(M_rand, recon_m, atol=1e-10)

def test_linear_system_validations():
    # Valid 2-qubit system
    A = np.eye(4)
    b = np.array([1.0, 0.0, 0.0, 0.0])
    sys = LinearSystem(A=A, b=b, name="test_sys")

    assert sys.size == 4
    assert sys.n_qubits == 2
    assert sys.is_hermitian is True
    assert np.isclose(sys.condition_number(), 1.0)
    assert np.allclose(sys.classical_solution(), b)

    # Validation errors
    with pytest.raises(ValueError, match="square"):
        LinearSystem(A=np.ones((4, 3)), b=np.ones(4))
    with pytest.raises(ValueError, match="power of 2"):
        LinearSystem(A=np.eye(3), b=np.ones(3))
    with pytest.raises(ValueError, match="all zeros"):
        LinearSystem(A=np.eye(4), b=np.zeros(4))
    with pytest.raises(ValueError, match="non-finite"):
        LinearSystem(A=np.array([[np.nan, 0], [0, 1]]), b=np.ones(2))

def test_hermitian_dilation():
    # Non-Hermitian 2x2 matrix
    A = np.array([[1.0, 2.0], [0.0, 1.0]], dtype=complex)
    b = np.array([1.0, 3.0], dtype=complex)
    sys = LinearSystem(A=A, b=b, name="non_herm")
    assert sys.is_hermitian is False

    dil = sys.hermitian_dilation()
    assert dil.size == 4
    assert dil.n_qubits == 2
    assert dil.is_hermitian is True

    # Dilated solution check:
    # A_dil [0, x]^T = [b, 0]^T
    x_exact = sys.classical_solution()
    x_dil = dil.classical_solution()
    # The solution block corresponding to x is the second half: x_dil[2:]
    assert np.allclose(x_dil[2:], x_exact, atol=1e-10)
