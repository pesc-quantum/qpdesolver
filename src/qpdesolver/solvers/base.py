"""
Base classes and result containers for linear solvers.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import numpy as np

from qpdesolver.problems.linear_system import LinearSystem

@dataclass
class SolveResult:
    """
    Standard result returned by all linear solvers in qpdesolver.
    """
    x: np.ndarray
    state: np.ndarray
    residual: float
    fidelity: Optional[float] = None
    success_probability: Optional[float] = None
    history: List[float] = field(default_factory=list)
    n_circuit_evals: int = 0
    resources: Dict[str, Any] = field(default_factory=dict)
    wall_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Converts result into a serializable dictionary suitable for HDF5 persistence."""
        return {
            "x": self.x,
            "state": self.state,
            "residual": float(self.residual),
            "fidelity": float(self.fidelity) if self.fidelity is not None else None,
            "success_probability": (
                float(self.success_probability) if self.success_probability is not None else None
            ),
            "history": [float(h) for h in self.history],
            "n_circuit_evals": int(self.n_circuit_evals),
            "resources": self.resources,
            "wall_time": float(self.wall_time),
            "metadata": self.metadata,
        }

class LinearSolver(ABC):
    """
    Abstract interface for linear system solvers (classical, HHL, VQLS).
    """
    name: str = "linear_solver"

    @abstractmethod
    def solve(
        self,
        system: LinearSystem,
        *,
        warm_start: Optional[np.ndarray] = None,
    ) -> SolveResult:
        """Solves A x = b and returns a SolveResult."""
        pass
