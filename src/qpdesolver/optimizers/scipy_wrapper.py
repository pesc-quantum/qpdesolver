"""
Wrapper for scipy.optimize.minimize algorithms (BFGS, L-BFGS-B, Newton-CG, etc.).
"""
from typing import Optional, Callable, Dict, Any
import numpy as np
import scipy.optimize

from qpdesolver.core.registry import REGISTRY_OPTIMIZERS
from qpdesolver.optimizers.base import Optimizer, OptimizeResult, CountingObjective

@REGISTRY_OPTIMIZERS.register("scipy")
@REGISTRY_OPTIMIZERS.register("bfgs")
@REGISTRY_OPTIMIZERS.register("lbfgs")
class ScipyOptimizer(Optimizer):
    """
    Wrapper around scipy.optimize.minimize.
    Automatically forwards gradient (jac) and Hessian if supported by the method and objective.
    """
    name = "scipy"

    def __init__(self, method: str = "BFGS", options: Optional[Dict[str, Any]] = None):
        self.method = method.upper() if method.lower() not in ("l-bfgs-b", "newton-cg") else method
        self.options = options or {}

    def minimize(
        self,
        objective: Any,
        x0: np.ndarray,
        *,
        max_iter: int = 200,
        tol: float = 1e-10,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> OptimizeResult:
        counted = CountingObjective(objective)
        x_init = np.array(x0, dtype=float, copy=True)

        history = [counted.value(x_init)]
        grad_norms = []

        def internal_callback(xk):
            val = counted.value(xk)
            history.append(val)
            if hasattr(objective, "grad"):
                try:
                    grad_norms.append(float(np.linalg.norm(counted.grad(xk))))
                except Exception:
                    pass
            if callback:
                callback(xk)

        jac_fn = None
        if hasattr(objective, "grad") and self.method not in ("COBYLA", "NELDER-MEAD"):
            jac_fn = counted.grad

        hess_fn = None
        if hasattr(objective, "hessian") and self.method in ("NEWTON-CG", "TRUST-EXACT", "TRUST-KRYLOV", "DOGLEG"):
            hess_fn = counted.hessian

        opts = dict(self.options)
        if "maxiter" not in opts:
            opts["maxiter"] = max_iter

        res = scipy.optimize.minimize(
            fun=counted.value,
            x0=x_init,
            method=self.method,
            jac=jac_fn,
            hess=hess_fn,
            tol=tol,
            callback=internal_callback,
            options=opts,
        )

        final_x = np.array(res.x, dtype=float)
        if len(history) == 1:
            history.append(float(res.fun))

        return OptimizeResult(
            x=final_x,
            fun=float(res.fun),
            history=history,
            grad_norms=grad_norms,
            n_iter=res.nit if hasattr(res, "nit") else len(history),
            n_evals=counted.counts,
            converged=bool(res.success),
            message=str(res.message),
        )
