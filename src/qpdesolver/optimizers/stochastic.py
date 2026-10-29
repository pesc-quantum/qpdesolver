"""
Shot-friendly stochastic optimizers: SPSA and Quantum Natural SPSA (QN-SPSA).
Evaluate constant number of circuits per step, well suited for noisy simulators and real hardware.
"""
from typing import Optional, Callable, Any
import numpy as np

from qpdesolver.core.registry import REGISTRY_OPTIMIZERS
from qpdesolver.optimizers.base import Optimizer, OptimizeResult, CountingObjective

@REGISTRY_OPTIMIZERS.register("spsa")
class SPSA(Optimizer):
    """
    Simultaneous Perturbation Stochastic Approximation (Spall 1992).
    Evaluates only 2 objective values per iteration, regardless of parameter dimension.
    """
    name = "spsa"

    def __init__(
        self,
        a: float = 0.2,
        c: float = 0.1,
        alpha: float = 0.602,
        gamma: float = 0.101,
        A: float = 10.0,
        seed: Optional[int] = None,
    ):
        self.a = float(a)
        self.c = float(c)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.A = float(A)
        self.seed = seed

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
        rng = np.random.default_rng(self.seed)

        x = np.array(x0, dtype=float, copy=True)
        dim = len(x)
        history = [counted.value(x)]
        grad_norms = []

        for k in range(1, max_iter + 1):
            ak = self.a / ((k + self.A) ** self.alpha)
            ck = self.c / (k ** self.gamma)

            # Rademacher perturbation (+1 or -1 with prob 0.5)
            delta = rng.choice([-1.0, 1.0], size=dim)

            val_plus = counted.value(x + ck * delta)
            val_minus = counted.value(x - ck * delta)

            ghat = (val_plus - val_minus) / (2.0 * ck * delta)
            gnorm = float(np.linalg.norm(ghat))
            grad_norms.append(gnorm)

            x = x - ak * ghat

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
            converged=False,
            message="SPSA iteration budget reached.",
        )

@REGISTRY_OPTIMIZERS.register("qnspsa")
class QNSPSA(Optimizer):
    """
    Quantum Natural SPSA (Gacon et al. 2021).
    Estimates the Fubini-Study metric tensor stochastically from state overlaps.
    """
    name = "qnspsa"
    requires = frozenset({"state_overlap"})

    def __init__(
        self,
        a: float = 0.2,
        c: float = 0.1,
        reg: float = 1e-4,
        seed: Optional[int] = None,
    ):
        self.a = float(a)
        self.c = float(c)
        self.reg = float(reg)
        self.seed = seed

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
        rng = np.random.default_rng(self.seed)

        x = np.array(x0, dtype=float, copy=True)
        dim = len(x)
        history = [counted.value(x)]
        grad_norms = []
        hbar = np.eye(dim)

        for k in range(1, max_iter + 1):
            ak = self.a / (k ** 0.602)
            ck = self.c / (k ** 0.101)

            # Gradient perturbation
            d1 = rng.choice([-1.0, 1.0], size=dim)
            val_plus = counted.value(x + ck * d1)
            val_minus = counted.value(x - ck * d1)
            ghat = (val_plus - val_minus) / (2.0 * ck * d1)
            grad_norms.append(float(np.linalg.norm(ghat)))

            # Metric perturbation: estimate QFIM from 2-point overlap
            d2 = rng.choice([-1.0, 1.0], size=dim)
            x_pert1 = x + ck * d1
            x_pert2 = x + ck * d2
            overlap = counted.state_overlap(x_pert1, x_pert2)
            # Fubini-Study distance d_FS^2 = 1 - |<psi1|psi2>|^2
            fid_loss = 1.0 - overlap
            delta_tensor = np.outer(d1, d2)
            h_sample = -0.5 * (fid_loss / (ck**2)) * delta_tensor
            h_sample = 0.5 * (h_sample + h_sample.T) # symmetrize

            # Running average of metric
            beta = 1.0 / k
            hbar = (1.0 - beta) * hbar + beta * h_sample
            h_reg = hbar + self.reg * np.eye(dim)

            try:
                step = np.linalg.solve(h_reg, ghat)
            except np.linalg.LinAlgError:
                step = ghat

            x = x - ak * step
            history.append(counted.value(x))
            if callback:
                callback(x)

        return OptimizeResult(
            x=x,
            fun=history[-1],
            history=history,
            grad_norms=grad_norms,
            n_iter=len(grad_norms),
            n_evals=counted.counts,
            converged=False,
            message="QN-SPSA iteration budget reached.",
        )
