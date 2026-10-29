"""
Abstract interfaces, protocols, and result dataclasses for optimizers.
Pure NumPy architecture: optimizers interact only with an Objective.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable, List, Dict, Any, Optional, Callable, FrozenSet
import numpy as np

@runtime_checkable
class Objective(Protocol):
    """
    Protocol defining mathematical capabilities of an optimization objective.
    """
    n_params: int

    def value(self, x: np.ndarray) -> float: ...

class CountingObjective:
    """
    Decorator wrapper that tracks the exact number of evaluations per capability.
    Essential for evaluating circuit evaluation counts in quantum algorithms.
    """
    def __init__(self, objective: Any):
        self.objective = objective
        self.n_params = objective.n_params
        self.counts: Dict[str, int] = {
            "value": 0,
            "grad": 0,
            "metric_tensor": 0,
            "hessian": 0,
            "state_overlap": 0,
        }

    def value(self, x: np.ndarray) -> float:
        self.counts["value"] += 1
        return self.objective.value(x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self.objective, "grad"):
            raise AttributeError("Wrapped objective does not support 'grad'.")
        self.counts["grad"] += 1
        return self.objective.grad(x)

    def metric_tensor(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self.objective, "metric_tensor"):
            raise AttributeError("Wrapped objective does not support 'metric_tensor'.")
        self.counts["metric_tensor"] += 1
        return self.objective.metric_tensor(x)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        if not hasattr(self.objective, "hessian"):
            raise AttributeError("Wrapped objective does not support 'hessian'.")
        self.counts["hessian"] += 1
        return self.objective.hessian(x)

    def state_overlap(self, x: np.ndarray, y: np.ndarray) -> float:
        if not hasattr(self.objective, "state_overlap"):
            raise AttributeError("Wrapped objective does not support 'state_overlap'.")
        self.counts["state_overlap"] += 1
        return self.objective.state_overlap(x, y)

@dataclass
class OptimizeResult:
    """
    Result returned by any Optimizer.
    """
    x: np.ndarray
    fun: float
    history: List[float]
    grad_norms: List[float] = field(default_factory=list)
    n_iter: int = 0
    n_evals: Dict[str, int] = field(default_factory=dict)
    converged: bool = False
    message: str = ""

class Optimizer(ABC):
    """
    Base optimizer interface.
    """
    name: str = "optimizer"
    requires: FrozenSet[str] = frozenset()

    def check_requirements(self, objective: Any) -> None:
        """Validates that the objective implements all capabilities required by this optimizer."""
        for req in self.requires:
            if not hasattr(objective, req):
                raise TypeError(
                    f"Optimizer '{self.name}' requires objective method '{req}', "
                    f"but {type(objective).__name__} does not implement it."
                )

    @abstractmethod
    def minimize(
        self,
        objective: Any,
        x0: np.ndarray,
        *,
        max_iter: int = 200,
        tol: float = 1e-10,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> OptimizeResult:
        """Minimizes the objective starting from x0."""
        pass
