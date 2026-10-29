"""
Fourier Pseudospectral and Modal representation of 1D differential operators.
Demonstrates non-finite-difference spectral expansions where derivatives
are diagonal in the Fourier basis (directly related to Quantum Fourier Transform QFT).
"""
from typing import Optional, Union
import numpy as np

from qpdesolver.core.registry import REGISTRY_DISCRETIZATIONS
from qpdesolver.problems.discretizations.base import PDEDiscretization
from qpdesolver.problems.linear_system import LinearSystem

@REGISTRY_DISCRETIZATIONS.register("fourier_spectral_1d")
class FourierSpectral1D(PDEDiscretization):
    """
    Fourier Pseudospectral differentiation for 1D periodic differential operators.
    In the Fourier basis (momentum space), the continuous Laplacian -d^2/dx^2 is exactly DIAGONAL:
        diag(k^2)
    In physical space, the differentiation matrix is dense:
        D_2 = F^dagger diag(-k^2) F
    where F is the Discrete Fourier Transform (DFT / QFT).
    """
    def __init__(self, n_points: int, L: float = 2.0 * np.pi):
        super().__init__(n_points=n_points, h=L / n_points, bc="periodic")
        self.L = float(L)

    def wavenumber_spectrum(self) -> np.ndarray:
        """Returns standard Fourier frequencies/wavenumbers k."""
        N = self.n_points
        # 2*pi/L * fftfreq
        return (2.0 * np.pi / self.L) * np.fft.fftfreq(N) * N

    def laplacian_modal(self) -> np.ndarray:
        """
        Laplacian in Fourier/modal representation (spectral space).
        Operator is exactly diagonal with entries k^2.
        """
        k = self.wavenumber_spectrum()
        return np.diag(k**2).astype(complex)

    def laplacian_physical(self) -> np.ndarray:
        """
        Pseudospectral negative Laplacian in physical coordinate space.
        Constructed via F^dagger diag(k^2) F.
        """
        N = self.n_points
        k = self.wavenumber_spectrum()
        F = np.fft.fft(np.eye(N)) / np.sqrt(N)
        D2 = F.conj().T @ np.diag(k**2) @ F
        return np.real_if_close(D2)

    def operator(
        self,
        omega: float = 1.0,
        m: Optional[Union[np.ndarray, float]] = None,
        basis: str = "physical",
    ) -> np.ndarray:
        """
        Constructs the Helmholtz operator A = -d^2/dx^2 - omega^2 * diag(m).
        basis: 'physical' (dense pseudospectral matrix in coordinate space)
               or 'modal' (diagonal Laplacian in Fourier momentum space).
        """
        if basis == "modal":
            L_op = self.laplacian_modal()
            if m is None or np.isscalar(m):
                m_val = float(m if m is not None else 1.0)
                return L_op - (omega**2) * m_val * np.eye(self.n_points)
            else:
                # In modal space, spatially-varying m(x) becomes a convolution matrix
                F = np.fft.fft(np.eye(self.n_points)) / np.sqrt(self.n_points)
                M_phys = np.diag(np.asarray(m, dtype=float))
                M_modal = F.conj().T @ M_phys @ F
                return L_op - (omega**2) * M_modal
        else:
            L_op = self.laplacian_physical()
            if m is None:
                m_diag = np.ones(self.n_points)
            elif np.isscalar(m):
                m_diag = float(m) * np.ones(self.n_points)
            else:
                m_diag = np.asarray(m, dtype=float)
            return L_op - (omega**2) * np.diag(m_diag)

    def system(
        self,
        rhs: np.ndarray,
        omega: float = 1.0,
        m: Optional[Union[np.ndarray, float]] = None,
        basis: str = "physical",
        name: Optional[str] = None,
    ) -> LinearSystem:
        A = self.operator(omega=omega, m=m, basis=basis)
        b_vec = np.asarray(rhs, dtype=complex)
        if basis == "modal":
            # Transform RHS into Fourier coefficients if specified
            b_vec = np.fft.fft(b_vec) / np.sqrt(self.n_points)

        sys_name = name or f"FourierSpectral_omega={omega:.2f}_{basis}"
        return LinearSystem(
            A=A,
            b=b_vec,
            name=sys_name,
            metadata={
                "discretization": "fourier_spectral",
                "basis": basis,
                "omega": omega,
                "n_points": self.n_points,
                "L": self.L,
            },
        )
