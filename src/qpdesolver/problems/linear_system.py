"""
LinearSystem data container and linear-algebra utilities.
Pure NumPy representation of a linear system A x = b.
"""
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional
import numpy as np

from qpdesolver.problems.pauli import pauli_decompose
from qpdesolver.core.metrics import condition_number

@dataclass(frozen=True)
class LinearSystem:
    """
    Represents a linear system of equations A x = b for quantum/classical solvers.

    Attributes:
        A: (N, N) matrix, where N = 2^n is a power of 2.
        b: (N,) right-hand side vector.
        name: human-readable name or identifier.
        metadata: optional dictionary with provenance, PDE parameters, physics info.
    """
    A: np.ndarray
    b: np.ndarray
    name: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Convert to numpy arrays if needed
        A_arr = np.asarray(self.A)
        b_arr = np.asarray(self.b)

        if A_arr.ndim != 2:
            raise ValueError(f"Matrix A must be 2D, got shape {A_arr.shape}.")
        if A_arr.shape[0] != A_arr.shape[1]:
            raise ValueError(f"Matrix A must be square, got shape {A_arr.shape}.")
        if b_arr.ndim != 1:
            raise ValueError(f"Vector b must be 1D, got shape {b_arr.shape}.")
        if A_arr.shape[0] != b_arr.shape[0]:
            raise ValueError(
                f"Dimension mismatch: A is {A_arr.shape[0]}x{A_arr.shape[1]}, but b has length {b_arr.shape[0]}."
            )

        n = A_arr.shape[0]
        n_qubits = int(np.round(np.log2(n)))
        if 2**n_qubits != n:
            raise ValueError(f"Dimension {n} is not a power of 2 (required for qubit registers).")

        if not np.all(np.isfinite(A_arr)):
            raise ValueError("Matrix A contains non-finite values (NaN or Inf).")
        if not np.all(np.isfinite(b_arr)):
            raise ValueError("Vector b contains non-finite values (NaN or Inf).")

        b_norm = float(np.linalg.norm(b_arr))
        if b_norm < 1e-15:
            raise ValueError("Right-hand side vector b cannot be all zeros.")

        # Object is frozen, so set via object.__setattr__
        object.__setattr__(self, "A", A_arr)
        object.__setattr__(self, "b", b_arr)

    @property
    def size(self) -> int:
        return self.A.shape[0]

    @property
    def n_qubits(self) -> int:
        return int(np.round(np.log2(self.size)))

    @property
    def b_norm(self) -> float:
        return float(np.linalg.norm(self.b))

    @property
    def b_state(self) -> np.ndarray:
        """Normalized statevector |b> = b / ||b||_2."""
        return self.b / self.b_norm

    @property
    def is_hermitian(self) -> bool:
        return bool(np.allclose(self.A, self.A.conj().T, atol=1e-10))

    def eigenvalue_bounds(self) -> Tuple[float, float]:
        """
        Returns (min_abs_lambda, max_abs_lambda) for the system matrix.
        If Hermitian, computes absolute values of eigenvalues.
        Otherwise, computes singular value bounds.
        """
        if self.is_hermitian:
            evals = np.linalg.eigvalsh(self.A)
            abs_evals = np.abs(evals)
            return float(np.min(abs_evals)), float(np.max(abs_evals))
        else:
            s = np.linalg.svd(self.A, compute_uv=False)
            return float(s[-1]), float(s[0])

    def condition_number(self) -> float:
        return condition_number(self.A)

    def pauli_terms(self, tol: float = 1e-10) -> List[Tuple[complex, str]]:
        """Decomposes A into a list of (coefficient, pauli_string) pairs."""
        return pauli_decompose(self.A, tol=tol)

    def hermitian_dilation(self) -> "LinearSystem":
        """
        Constructs the (2N x 2N) Hermitian dilated system:
            A_dil = [[0, A], [A^dagger, 0]]
            b_dil = [b, 0]
        Such that A_dil [0, x]^T = [A x, 0]^T = [b, 0]^T.
        """
        n = self.size
        A_dil = np.zeros((2 * n, 2 * n), dtype=complex)
        A_dil[:n, n:] = self.A
        A_dil[n:, :n] = self.A.conj().T

        b_dil = np.zeros(2 * n, dtype=complex)
        b_dil[:n] = self.b

        meta = dict(self.metadata)
        meta["dilated_from"] = self.name
        meta["original_size"] = n

        return LinearSystem(
            A=A_dil,
            b=b_dil,
            name=f"{self.name}_dilated",
            metadata=meta,
        )

    def classical_solution(self) -> np.ndarray:
        """Solves A x = b classically using np.linalg.solve."""
        return np.linalg.solve(self.A, self.b)
