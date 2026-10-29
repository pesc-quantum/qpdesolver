"""
Pure-NumPy Pauli decomposition and Kronecker tensor products.
No quantum framework dependencies.
Convention: wire 0 is the most significant bit (leftmost in Kronecker product).
"""
import itertools
from typing import List, Tuple
import numpy as np

# Elementary single-qubit Pauli matrices
_PAULI_1Q = {
    "I": np.array([[1.0, 0.0], [0.0, 1.0]], dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}

def pauli_matrix(label: str) -> np.ndarray:
    """
    Returns the 2^n x 2^n matrix representation of a Pauli string.
    Example: 'XZ' on 2 qubits returns X (wire 0) (x) Z (wire 1).
    """
    if not label:
        raise ValueError("Pauli label cannot be empty.")
    label = label.upper()
    mat = _PAULI_1Q[label[0]]
    for ch in label[1:]:
        mat = np.kron(mat, _PAULI_1Q[ch])
    return mat

def pauli_decompose(matrix: np.ndarray, tol: float = 1e-10) -> List[Tuple[complex, str]]:
    """
    Decomposes an (N, N) matrix (N = 2^n) into a linear combination of Pauli strings:
        M = sum_k c_k P_k
    where c_k = Tr(P_k^dagger M) / 2^n.
    Returns list of (coefficient, pauli_string) tuples with |coefficient| > tol.
    """
    mat = np.asarray(matrix, dtype=complex)
    rows, cols = mat.shape
    if rows != cols:
        raise ValueError(f"Matrix must be square, got shape {mat.shape}.")
    n_qubits = int(np.round(np.log2(rows)))
    if 2**n_qubits != rows:
        raise ValueError(f"Matrix dimension {rows} must be a power of 2.")

    dim = 2**n_qubits
    pauli_chars = ["I", "X", "Y", "Z"]
    terms: List[Tuple[complex, str]] = []

    for chars in itertools.product(pauli_chars, repeat=n_qubits):
        label = "".join(chars)
        p_mat = pauli_matrix(label)
        # For Pauli matrices, P^dagger = P
        coeff = np.trace(p_mat.conj().T @ mat) / dim
        if np.abs(coeff) > tol:
            terms.append((complex(coeff), label))

    return terms
