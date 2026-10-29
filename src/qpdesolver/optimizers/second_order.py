"""
Second-order optimizers: Damped Newton / Levenberg-Marquardt with adaptive damping.
"""
from typing import Optional, Callable, Any
import numpy as np

from qpdesolver.core.registry import REGISTRY_OPTIMIZERS
from qpdesolver.optimizers.base import Optimizer, OptimizeResult, CountingObjective

@REGISTRY_OPTIMIZERS.register("damped_newton")
@REGISTRY_OPTIMIZERS.register("newton")
class DampedNewton(Optimizer):
    """
    Second-order Damped Newton optimizer with Levenberg-Marquardt adaptive damping.
        (H(x) + lambda * I) delta = -grad(x)
    Adapts lambda dynamically based on function decrease.
    """
    name = "damped_newton"
    requires = frozenset({"grad", "hessian"})

    def __init__(self, initial_damping: float = 1e-2, nu: float = 2.0):
        self.initial_damping = float(initial_damping)
        self.nu = float(nu)

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
        cur_val = counted.value(x)
        history = [cur_val]
        grad_norms = []
        converged = False
        msg = "Maximum iterations reached."
        damp = self.initial_damping

        for it in range(max_iter):
            g = counted.grad(x)
            gnorm = float(np.linalg.norm(g))
            grad_norms.append(gnorm)

            if gnorm < tol:
                converged = True
                msg = f"Converged: gradient norm {gnorm:.3e} < tol {tol:.3e}."
                break

            H = counted.hessian(x)
            dim = len(x)

            # Solve damped system: (H + damp * I) delta = -g
            try:
                delta = np.linalg.solve(H + damp * np.eye(dim), -g)
            except np.linalg.LinAlgError:
                delta = np.linalg.lstsq(H + damp * np.eye(dim), -g, rcond=1e-8)[0]

            x_cand = x + delta
            cand_val = counted.value(x_cand)

            if cand_val < cur_val:
                # Accept step and reduce damping
                x = x_cand
                cur_val = cand_val
                damp = max(damp / self.nu, 1e-8)
            else:
                # Reject step and increase damping
                damp = min(damp * self.nu, 1e8)

            history.append(cur_val)
            if callback:
                callback(x)

        return OptimizeResult(
            x=x,
            fun=cur_val,
            history=history,
            grad_norms=grad_norms,
            n_iter=len(grad_norms),
            n_evals=counted.counts,
            converged=converged,
            message=msg,
        )
