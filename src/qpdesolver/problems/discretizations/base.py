"""
Abstract base class for PDE discretizations.
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import numpy as np

from qpdesolver.problems.linear_system import LinearSystem

class PDEDiscretization(ABC):
    """
    Base class for 1D/2D PDE spatial discretization on uniform grids.
    """
    def __init__(self, n_points: int, h: float = 1.0, bc: str = "dirichlet"):
        if n_points <= 0 or (n_points & (n_points - 1)) != 0:
            raise ValueError(f"n_points must be a positive power of 2, got {n_points}.")
        self.n_points = n_points
        self.h = float(h)
        self.bc = bc.lower()
        if self.bc not in ("dirichlet", "periodic", "neumann"):
            raise ValueError(f"Unsupported boundary condition '{bc}'. Supported: 'dirichlet', 'periodic', 'neumann'.")

    @property
    def n_qubits(self) -> int:
        return int(np.round(np.log2(self.n_points)))

    @abstractmethod
    def operator(self, **params) -> np.ndarray:
        """Constructs the discrete operator matrix A."""
        pass

    def point_source(self, idx: int) -> np.ndarray:
        """Creates a single-point delta source vector at index idx."""
        if not (0 <= idx < self.n_points):
            raise IndexError(f"Source index {idx} out of range [0, {self.n_points}).")
        b = np.zeros(self.n_points, dtype=float)
        b[idx] = 1.0
        return b

    def system(self, rhs: np.ndarray, name: Optional[str] = None, **params) -> LinearSystem:
        """Builds a LinearSystem object given a right-hand side and operator parameters."""
        A = self.operator(**params)
        meta = {
            "discretization": self.__class__.__name__,
            "n_points": self.n_points,
            "h": self.h,
            "bc": self.bc,
            "params": params,
        }
        sys_name = name or f"{self.__class__.__name__}_{self.n_points}_{self.bc}"
        return LinearSystem(A=A, b=rhs, name=sys_name, metadata=meta)
