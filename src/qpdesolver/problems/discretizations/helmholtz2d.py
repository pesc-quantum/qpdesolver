"""
2D Helmholtz equation discretization:
    (-nabla^2 - omega^2 * m(x, z)) u(x, z) = s(x, z)
where m(x, z) = 1 / v(x, z)^2 is the squared slowness,
and nabla^2 = d^2/dx^2 + d^2/dz^2.
"""
from typing import Optional, Union, Dict, Any
import numpy as np

from qpdesolver.core.registry import REGISTRY_DISCRETIZATIONS
from qpdesolver.problems.discretizations.base import PDEDiscretization
from qpdesolver.problems.linear_system import LinearSystem


@REGISTRY_DISCRETIZATIONS.register("helmholtz2d")
class Helmholtz2D(PDEDiscretization):
    """
    2D Finite-difference frequency-domain Helmholtz operator on a uniform Cartesian grid:
        nx x nz grid points, where nx and nz must each be a positive power of 2.
    Row-major flattening:
        (ix, iz) -> ix * nz + iz.
    2D Laplacian:
        L_2D = L_x (x) I_nz + I_nx (x) L_z
    Operator:
        A(m, omega) = L_2D - omega^2 * diag(m_flat)
    """

    def __init__(
        self,
        nx: int,
        nz: int,
        hx: Optional[float] = None,
        hz: Optional[float] = None,
        bc: str = "dirichlet",
        h: Optional[float] = None,
    ):
        if nx < 2 or (nx & (nx - 1)) != 0:
            raise ValueError(f"nx must be a positive power of 2 >= 2, got {nx}.")
        if nz < 2 or (nz & (nz - 1)) != 0:
            raise ValueError(f"nz must be a positive power of 2 >= 2, got {nz}.")

        if hx is None:
            hx = 1.0 if h is None else float(h)
        else:
            hx = float(hx)

        if hz is None:
            hz = hx
        else:
            hz = float(hz)

        super().__init__(n_points=nx * nz, h=hx, bc=bc)
        self.nx = nx
        self.nz = nz
        self.hx = float(hx)
        self.hz = float(hz)

    @staticmethod
    def _laplacian_1d(n: int, h: float, bc: str) -> np.ndarray:
        """Constructs 1D discrete negative Laplacian for given boundary conditions."""
        h2 = h**2
        diag_main = 2.0 * np.ones(n) / h2
        diag_off = -1.0 * np.ones(n - 1) / h2

        L = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)
        if bc == "periodic" and n > 2:
            L[0, -1] = -1.0 / h2
            L[-1, 0] = -1.0 / h2
        elif bc == "neumann":
            # Zero derivative at boundary: forward/backward 1st order
            L[0, 0] = 1.0 / h2
            L[-1, -1] = 1.0 / h2
        return L

    def laplacian_x(self) -> np.ndarray:
        """Constructs 1D Laplacian along x-axis (nx x nx)."""
        return self._laplacian_1d(self.nx, self.hx, self.bc)

    def laplacian_z(self) -> np.ndarray:
        """Constructs 1D Laplacian along z-axis (nz x nz)."""
        return self._laplacian_1d(self.nz, self.hz, self.bc)

    def laplacian(self) -> np.ndarray:
        """
        Constructs discrete 2D negative Laplacian L_2D = L_x (x) I_nz + I_nx (x) L_z
        consistent with row-major flattening (ix, iz) -> ix * nz + iz.
        """
        Lx = self.laplacian_x()
        Lz = self.laplacian_z()
        Ix = np.eye(self.nx)
        Iz = np.eye(self.nz)
        return np.kron(Lx, Iz) + np.kron(Ix, Lz)

    def point_source_2d(self, ix: int, iz: int) -> np.ndarray:
        """
        Creates a single-point delta source vector at 2D grid index (ix, iz).
        Row-major flattening: idx = ix * nz + iz.
        """
        if not (0 <= ix < self.nx):
            raise IndexError(f"Source ix={ix} out of range [0, {self.nx}).")
        if not (0 <= iz < self.nz):
            raise IndexError(f"Source iz={iz} out of range [0, {self.nz}).")
        idx = ix * self.nz + iz
        return self.point_source(idx)

    def operator(
        self,
        m: Optional[Union[np.ndarray, float]] = None,
        omega: float = 1.0,
        velocity: Optional[Union[np.ndarray, float]] = None,
        **kwargs,
    ) -> np.ndarray:
        """
        Constructs A(m, omega) = L_2D - omega^2 * diag(m_flat).
        Can specify either squared slowness m or velocity v (where m = 1/v^2).
        m or velocity can be a 2D array (nx, nz), 1D array (nx * nz,), or a scalar.
        """
        L = self.laplacian()
        N = self.n_points

        if m is not None:
            if np.isscalar(m):
                m_flat = np.full(N, float(m))
            else:
                m_arr = np.asarray(m)
                if m_arr.shape == (self.nx, self.nz):
                    m_flat = m_arr.reshape(-1)
                elif m_arr.shape == (N,):
                    m_flat = m_arr
                else:
                    raise ValueError(
                        f"m shape {m_arr.shape} incompatible with grid ({self.nx}, {self.nz}) or size {N}."
                    )
        elif velocity is not None:
            if np.isscalar(velocity):
                v_flat = np.full(N, float(velocity))
            else:
                v_arr = np.asarray(velocity)
                if v_arr.shape == (self.nx, self.nz):
                    v_flat = v_arr.reshape(-1)
                elif v_arr.shape == (N,):
                    v_flat = v_arr
                else:
                    raise ValueError(
                        f"velocity shape {v_arr.shape} incompatible with grid ({self.nx}, {self.nz}) or size {N}."
                    )
            m_flat = 1.0 / (v_flat**2)
        else:
            # Default homogeneous velocity v = 2.0
            m_flat = np.full(N, 1.0 / (2.0**2))

        A = L - (float(omega) ** 2) * np.diag(m_flat)
        return A.astype(complex)

    def dA_dm_diag(self, omega: float) -> float:
        """
        Derivative of diagonal entry with respect to m_i:
            d A_ii / d m_i = -omega^2.
        Linear dependence makes adjoint state gradients straightforward.
        """
        return -float(omega) ** 2

    def system(self, rhs: np.ndarray, name: Optional[str] = None, **params) -> LinearSystem:
        """Builds a LinearSystem object given a right-hand side and operator parameters."""
        A = self.operator(**params)
        meta = {
            "discretization": self.__class__.__name__,
            "n_points": self.n_points,
            "nx": self.nx,
            "nz": self.nz,
            "hx": self.hx,
            "hz": self.hz,
            "bc": self.bc,
            "params": params,
        }
        sys_name = name or f"{self.__class__.__name__}_{self.nx}x{self.nz}_{self.bc}"
        return LinearSystem(A=A, b=rhs, name=sys_name, metadata=meta)
