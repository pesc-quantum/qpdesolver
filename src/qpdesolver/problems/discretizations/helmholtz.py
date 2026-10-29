"""
1D Helmholtz equation discretization:
    (-d^2/dx^2 - omega^2 * m(x)) u(x) = s(x)
where m(x) = 1 / v(x)^2 is the squared slowness.
"""
from typing import Optional, Union
import numpy as np

from qpdesolver.core.registry import REGISTRY_DISCRETIZATIONS
from qpdesolver.problems.discretizations.base import PDEDiscretization

@REGISTRY_DISCRETIZATIONS.register("helmholtz1d")
class Helmholtz1D(PDEDiscretization):
    """
    1D Finite-difference frequency-domain Helmholtz operator.
    Discrete Laplacian:
        (L u)_j = (-u_{j-1} + 2 u_j - u_{j+1}) / h^2
    Operator A = L - omega^2 * diag(m).
    """
    def laplacian(self) -> np.ndarray:
        """Constructs discrete negative 1D Laplacian L / h^2."""
        N = self.n_points
        h2 = self.h**2
        diag_main = 2.0 * np.ones(N) / h2
        diag_off = -1.0 * np.ones(N - 1) / h2

        L = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)
        if self.bc == "periodic" and N > 2:
            L[0, -1] = -1.0 / h2
            L[-1, 0] = -1.0 / h2
        elif self.bc == "neumann":
            # Zero derivative at boundary: forward/backward 1st order
            L[0, 0] = 1.0 / h2
            L[-1, -1] = 1.0 / h2
        # Dirichlet bc has zeros outside [0, N-1], which matches standard tridiagonal
        return L

    def operator(
        self,
        m: Optional[Union[np.ndarray, float]] = None,
        omega: float = 1.0,
        velocity: Optional[Union[np.ndarray, float]] = None,
        **kwargs
    ) -> np.ndarray:
        """
        Constructs A(m, omega) = L - omega^2 * diag(m).
        Can specify either squared slowness m or velocity v (where m = 1/v^2).
        """
        L = self.laplacian()
        N = self.n_points

        if m is not None:
            if np.isscalar(m):
                m_vec = np.full(N, float(m))
            else:
                m_vec = np.asarray(m, dtype=float)
        elif velocity is not None:
            if np.isscalar(velocity):
                v_vec = np.full(N, float(velocity))
            else:
                v_vec = np.asarray(velocity, dtype=float)
            m_vec = 1.0 / (v_vec**2)
        else:
            # Default homogeneous velocity v = 2.0
            m_vec = np.full(N, 1.0 / (2.0**2))

        if len(m_vec) != N:
            raise ValueError(f"m vector length {len(m_vec)} must match n_points {N}.")

        A = L - (float(omega)**2) * np.diag(m_vec)
        return A.astype(complex)

    def dA_dm_diag(self, omega: float) -> float:
        """
        Derivative of diagonal entry with respect to m_i:
            d A_ii / d m_i = -omega^2.
        Linear dependence makes adjoint state gradients straightforward.
        """
        return -float(omega)**2
