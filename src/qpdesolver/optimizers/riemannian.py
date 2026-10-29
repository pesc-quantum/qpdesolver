"""
Riemannian gradient flow: Quantum Natural Gradient (QNG) on the Fubini-Study state manifold.
"""
from typing import Optional, Callable, Any
import numpy as np

from qpdesolver.core.registry import REGISTRY_OPTIMIZERS
from qpdesolver.optimizers.base import Optimizer, OptimizeResult, CountingObjective

@REGISTRY_OPTIMIZERS.register("qng")
@REGISTRY_OPTIMIZERS.register("quantum_natural_gradient")
class QuantumNaturalGradient(Optimizer):
    """
    Quantum Natural Gradient (Stokes et al. 2020).
    Uses the Fubini-Study metric tensor g(theta):
        theta_{k+1} = theta_k - lr * (g(theta_k) + reg * I)^{-1} grad(theta_k)
    """
    name = "quantum_natural_gradient"
    requires = frozenset({"grad", "metric_tensor"})

    def __init__(self, lr: float = 0.1, reg: float = 1e-4, approx: str = "block-diag"):
        self.lr = float(lr)
        self.reg = float(reg)
        self.approx = approx

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

            metric = counted.metric_tensor(x)
            dim = len(x)
            reg_metric = metric + self.reg * np.eye(dim)

            # Solve regularized system: (g + reg * I) d = g
            try:
                nat_grad = np.linalg.solve(reg_metric, g)
            except np.linalg.LinAlgError:
                nat_grad = np.linalg.lstsq(reg_metric, g, rcond=1e-8)[0]

            x = x - self.lr * nat_grad

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
