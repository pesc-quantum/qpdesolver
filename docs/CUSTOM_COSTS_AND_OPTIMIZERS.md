# Tutorial: Custom Cost Functions & Optimizers in `qpdesolver`

This guide explains how to extend `qpdesolver` with novel cost functions for variational quantum algorithms and custom classical/Riemannian optimizers.

---

## 1. Creating Custom VQLS Cost Functions

Cost functions quantify the distance between the quantum candidate state $A |\psi(\theta)\rangle$ and the target state $|b\rangle$.

### 1.1. The `VQLSCost` Abstract Interface

All cost functions derive from [`VQLSCost`](../src/qpdesolver/solvers/vqls/costs.py):

```python
from abc import ABC, abstractmethod
from typing import Callable
import numpy as np

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.ansatze.base import Ansatz
from qpdesolver.backends.base import Backend

class VQLSCost(ABC):
    @abstractmethod
    def build(
        self,
        system: LinearSystem,
        ansatz: Ansatz,
        backend: Backend,
    ) -> Callable[[np.ndarray], float]:
        """Returns an autograd/autodiff-compatible callable: params -> scalar cost."""
        pass
```

### 1.2. Example: L2-Regularized Quantum Cost Function
Suppose you want to mitigate parameter drift or over-rotation by adding an L2 parameter penalty $\lambda \|\theta\|_2^2$:

```python
import pennylane as qml
from pennylane import numpy as pnp
from qpdesolver.solvers.vqls.costs import VQLSCost
from qpdesolver.core.registry import REGISTRY_COSTS

@REGISTRY_COSTS.register("l2_regularized_cost")
class L2RegularizedCost(VQLSCost):
    """
    Combines the standard Global VQLS cost with an L2 weight decay penalty:
        C(theta) = C_G(theta) + 0.5 * reg_lambda * ||theta||^2
    """
    def __init__(self, reg_lambda: float = 1e-3):
        self.reg_lambda = reg_lambda

    def build(self, system, ansatz, backend):
        dev = backend.device(wires=system.n_qubits)
        A_mat = pnp.array(system.A, dtype=complex)
        b_vec = pnp.array(system.b_normalized, dtype=complex)

        @qml.qnode(dev, diff_method="backprop")
        def state_circuit(params):
            ansatz.apply(params, wires=list(range(system.n_qubits)))
            return qml.state()

        def cost_fn(params):
            psi = state_circuit(params)
            A_psi = pnp.dot(A_mat, psi)

            # Complex overlaps without tracer conflicts
            norm_sq = pnp.sum(pnp.real(A_psi * pnp.conj(A_psi)))
            overlap = pnp.sum(pnp.conj(b_vec) * A_psi)
            overlap_sq = pnp.real(overlap * pnp.conj(overlap))

            c_global = 1.0 - (overlap_sq / (norm_sq + 1e-12))
            penalty = 0.5 * self.reg_lambda * pnp.sum(params**2)
            return c_global + penalty

        return cost_fn
```

### Using with `VQLSSolver`
```python
solver = VQLSSolver(
    ansatz=RealAmplitudes(layers=2),
    cost=L2RegularizedCost(reg_lambda=1e-4),
    max_iter=50,
)
```

---

## 2. Implementing Custom Optimizers

`qpdesolver` features an open, pure-NumPy optimization architecture where optimizers communicate with an [`Objective`](../src/qpdesolver/optimizers/base.py) protocol.

### 2.1. The `Objective` Protocol Capabilities
An `Objective` passed to an optimizer may provide:
- `objective.value(x)`: Evaluates cost function $f(x)$.
- `objective.grad(x)`: Computes analytical gradient vector $\nabla f(x)$.
- `objective.metric_tensor(x)`: Evaluates Fubini-Study Riemannian metric tensor $g(x)$.
- `objective.hessian(x)`: Computes the full second-order Hessian matrix $H(x)$.

### 2.2. Example: Barzilai-Borwein Two-Point Step-Size Gradient Descent
The Barzilai-Borwein (BB) method computes adaptive spectral step sizes $\alpha_k$ using two successive gradient evaluations without requiring line searches:

```python
from typing import Optional, Callable
import numpy as np

from qpdesolver.optimizers.base import Optimizer, OptimizeResult, CountingObjective
from qpdesolver.core.registry import REGISTRY_OPTIMIZERS

@REGISTRY_OPTIMIZERS.register("barzilai_borwein")
class BarzilaiBorweinOptimizer(Optimizer):
    """
    Two-Point Step-Size Gradient Method (Barzilai & Borwein, 1988).
    Adapts step sizes dynamically using curvature information from previous iterates.
    """
    name = "barzilai_borwein"
    requires = frozenset({"grad"})

    def __init__(self, initial_step: float = 0.05):
        self.initial_step = initial_step

    def minimize(
        self,
        objective,
        x0: np.ndarray,
        *,
        max_iter: int = 100,
        tol: float = 1e-7,
        callback: Optional[Callable[[np.ndarray], None]] = None,
    ) -> OptimizeResult:
        counted = CountingObjective(objective)
        x = np.array(x0, dtype=float, copy=True)
        g = counted.grad(x)

        history = [counted.value(x)]
        grad_norms = [float(np.linalg.norm(g))]
        alpha = self.initial_step
        converged = False

        for k in range(max_iter):
            if grad_norms[-1] < tol:
                converged = True
                break

            x_prev = x.copy()
            g_prev = g.copy()

            # Step update
            x = x - alpha * g
            history.append(counted.value(x))

            if callback:
                callback(x)

            g = counted.grad(x)
            gnorm = float(np.linalg.norm(g))
            grad_norms.append(gnorm)

            # BB curvature step-size update
            s = x - x_prev
            y = g - g_prev
            sy = np.dot(s, y)
            if sy > 1e-12:
                alpha = np.dot(s, s) / sy
            else:
                alpha = self.initial_step

        return OptimizeResult(
            x=x,
            fun=history[-1],
            history=history,
            grad_norms=grad_norms,
            n_iter=len(grad_norms),
            n_evals=counted.counts,
            converged=converged,
            message="Converged" if converged else "Max iterations reached",
        )
```

### Benchmarking the Custom Optimizer
Once registered, you can use your custom optimizer anywhere in the package:
```python
solver = VQLSSolver(
    ansatz=RealAmplitudes(layers=2),
    optimizer=BarzilaiBorweinOptimizer(initial_step=0.08),
    max_iter=60,
)
result = solver.solve(system)
print(f"BB Solver Fidelity: {result.fidelity * 100:.2f}% in {result.metadata['iterations']} iterations")
```
