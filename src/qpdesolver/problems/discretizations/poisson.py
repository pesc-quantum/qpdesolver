"""
1D Poisson / diffusion equation discretization:
    -d/dx (kappa(x) du/dx) = f(x)
Strictly symmetric positive definite (SPD) under Dirichlet boundary conditions.
Ideal for theoretical benchmarks and analytical spectrum verification.
"""
from typing import Optional, Union
import numpy as np

from qpdesolver.core.registry import REGISTRY_DISCRETIZATIONS
from qpdesolver.problems.discretizations.base import PDEDiscretization

@REGISTRY_DISCRETIZATIONS.register("poisson1d")
class Poisson1D(PDEDiscretization):
    """
    1D Poisson operator with variable diffusion coefficient kappa(x).
    Standard finite differences:
        -(kappa_{j+1/2}(u_{j+1}-u_j) - kappa_{j-1/2}(u_j-u_{j-1})) / h^2
    """
    def operator(
        self,
        kappa: Optional[Union[np.ndarray, float]] = None,
        **kwargs
    ) -> np.ndarray:
        N = self.n_points
        h2 = self.h**2

        if kappa is None:
            k_vec = np.ones(N + 1, dtype=float)
        elif np.isscalar(kappa):
            k_vec = np.full(N + 1, float(kappa))
        else:
            k_arr = np.asarray(kappa, dtype=float)
            if len(k_arr) == N:
                # Interpolate to staggered grid interfaces
                k_vec = np.empty(N + 1, dtype=float)
                k_vec[0] = k_arr[0]
                k_vec[-1] = k_arr[-1]
                k_vec[1:-1] = 0.5 * (k_arr[:-1] + k_arr[1:])
            elif len(k_arr) == N + 1:
                k_vec = k_arr
            else:
                raise ValueError(f"kappa length {len(k_arr)} must be either {N} or {N+1}.")

        A = np.zeros((N, N), dtype=float)
        for i in range(N):
            A[i, i] = (k_vec[i] + k_vec[i + 1]) / h2
            if i > 0:
                A[i, i - 1] = -k_vec[i] / h2
            if i < N - 1:
                A[i, i + 1] = -k_vec[i + 1] / h2

        if self.bc == "periodic" and N > 2:
            A[0, -1] = -k_vec[0] / h2
            A[-1, 0] = -k_vec[0] / h2

        return A.astype(complex)

    def analytical_eigenvalues(self) -> np.ndarray:
        """
        Analytical eigenvalues for constant kappa=1, Dirichlet BC:
            lambda_k = 4/h^2 * sin^2(k * pi / (2 * (N + 1))), for k = 1, ..., N.
        """
        if self.bc != "dirichlet":
            raise NotImplementedError("Analytical eigenvalues provided only for Dirichlet BC.")
        N = self.n_points
        k = np.arange(1, N + 1)
        return 4.0 / (self.h**2) * (np.sin(k * np.pi / (2.0 * (N + 1))))**2
