"""
Euclidean first-order optimizers: GradientDescent and Adam.
"""
from typing import Optional, Callable, Any
import numpy as np

from qpdesolver.core.registry import REGISTRY_OPTIMIZERS
from qpdesolver.optimizers.base import Optimizer, OptimizeResult, CountingObjective

@REGISTRY_OPTIMIZERS.register("gd")
@REGISTRY_OPTIMIZERS.register("gradient_descent")
class GradientDescent(Optimizer):
    """
    Standard gradient descent with optional momentum.
    """
    name = "gradient_descent"
    requires = frozenset({"grad"})

    def __init__(self, lr: float = 0.1, momentum: float = 0.0):
        self.lr = float(lr)
        self.momentum = float(momentum)

    def minimize(
        self,
        objective: Any,
        x0: np.ndarray,
        *,
        max_iter: int = 200,
        tol: float = 1e-10,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> OptimizeResult:
        self.check_requirements(objective)
        counted = CountingObjective(objective)

        x = np.array(x0, dtype=float, copy=True)
        v = np.zeros_like(x)
        history = [counted.value(x)]
        grad_norms = []
        converged = False
        msg = "Maximum iterations reached."

        for it in range(max_iter):
            g = counted.grad(x)
            gnorm = float(np.linalg.norm(g))
            grad_norms.append(gnorm)

            if gnorm < tol:
                converged = True
                msg = f"Converged: gradient norm {gnorm:.3e} < tol {tol:.3e}."
                break

            v = self.momentum * v - self.lr * g
            x = x + v

            val = counted.value(x)
            history.append(val)
            if callback:
                callback(x)

        return OptimizeResult(
            x=x,
            fun=history[-1],
            history=history,
            grad_norms=grad_norms,
            n_iter=len(grad_norms),
            n_evals=counted.counts,
            converged=converged,
            message=msg,
        )

@REGISTRY_OPTIMIZERS.register("adam")
class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer.
    """
    name = "adam"
    requires = frozenset({"grad"})

    def __init__(
        self,
        lr: float = 0.01,
        b1: float = 0.9,
        b2: float = 0.999,
        eps: float = 1e-8,
    ):
        self.lr = float(lr)
        self.b1 = float(b1)
        self.b2 = float(b2)
        self.eps = float(eps)

    def minimize(
        self,
        objective: Any,
        x0: np.ndarray,
        *,
        max_iter: int = 200,
        tol: float = 1e-10,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> OptimizeResult:
        self.check_requirements(objective)
        counted = CountingObjective(objective)

        x = np.array(x0, dtype=float, copy=True)
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        history = [counted.value(x)]
        grad_norms = []
        converged = False
        msg = "Maximum iterations reached."

        for it in range(1, max_iter + 1):
            g = counted.grad(x)
            gnorm = float(np.linalg.norm(g))
            grad_norms.append(gnorm)

            if gnorm < tol:
                converged = True
                msg = f"Converged: gradient norm {gnorm:.3e} < tol {tol:.3e}."
                break

            m = self.b1 * m + (1.0 - self.b1) * g
            v = self.b2 * v + (1.0 - self.b2) * (g**2)

            m_hat = m / (1.0 - self.b1**it)
            v_hat = v / (1.0 - self.b2**it)

            x = x - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            val = counted.value(x)
            history.append(val)
            if callback:
                callback(x)

        return OptimizeResult(
            x=x,
            fun=history[-1],
            history=history,
            grad_norms=grad_norms,
            n_iter=len(grad_norms),
            n_evals=counted.counts,
            converged=converged,
            message=msg,
        )
