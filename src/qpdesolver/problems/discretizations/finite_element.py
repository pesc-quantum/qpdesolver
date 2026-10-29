"""
1D Galerkin Finite Element Method (FEM) spatial discretization.
Transforms continuous weak variational formulations:
    int (du/dx * dv/dx) dx - omega^2 * int (m * u * v) dx = int (s * v) dx
into stiffness and mass matrices: (K - omega^2 * M) u = F.
"""
from typing import Optional, Union
import numpy as np

from qpdesolver.core.registry import REGISTRY_DISCRETIZATIONS
from qpdesolver.problems.discretizations.base import PDEDiscretization
from qpdesolver.problems.linear_system import LinearSystem

@REGISTRY_DISCRETIZATIONS.register("finite_element_1d")
class FiniteElement1D(PDEDiscretization):
    """
    1D Galerkin Finite Element formulation with piecewise linear (hat) basis functions.
    Constructs stiffness matrix K and mass matrix M:
        Stiffness K: int (d phi_i / dx * d phi_j / dx) dx
        Mass M:      int (phi_i * phi_j) dx
    Weak Helmholtz Operator:
        A = K - omega^2 * M
    """
    def stiffness_matrix(self) -> np.ndarray:
        """Assembles standard 1D stiffness matrix K."""
        N = self.n_points
        h = self.h
        diag_main = 2.0 / h * np.ones(N)
        diag_off = -1.0 / h * np.ones(N - 1)
        K = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)

        if self.bc == "periodic" and N > 2:
            K[0, -1] = -1.0 / h
            K[-1, 0] = -1.0 / h
        return K

    def mass_matrix(self) -> np.ndarray:
        """Assembles standard consistent 1D mass matrix M."""
        N = self.n_points
        h = self.h
        diag_main = (2.0 * h / 3.0) * np.ones(N)
        diag_off = (h / 6.0) * np.ones(N - 1)
        M = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)

        if self.bc == "periodic" and N > 2:
            M[0, -1] = h / 6.0
            M[-1, 0] = h / 6.0
        return M

    def operator(
        self,
        omega: float = 1.0,
        m: Optional[Union[np.ndarray, float]] = None,
    ) -> np.ndarray:
        K = self.stiffness_matrix()
        M = self.mass_matrix()
        m_val = float(m if m is not None else 1.0)
        return K - (omega**2) * m_val * M

    def system(
        self,
        rhs: np.ndarray,
        omega: float = 1.0,
        m: Optional[Union[np.ndarray, float]] = None,
        name: Optional[str] = None,
    ) -> LinearSystem:
        A = self.operator(omega=omega, m=m)
        # In FEM, right-hand side is the integrated load vector F = M * s
        M = self.mass_matrix()
        F_rhs = M @ np.asarray(rhs, dtype=float)

        sys_name = name or f"FEM1D_Helmholtz_omega={omega:.2f}"
        return LinearSystem(
            A=A,
            b=F_rhs,
            name=sys_name,
            metadata={
                "discretization": "finite_element_1d",
                "omega": omega,
                "n_points": self.n_points,
                "h": self.h,
                "bc": self.bc,
            },
        )
