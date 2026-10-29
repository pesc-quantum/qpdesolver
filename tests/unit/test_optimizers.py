"""
Unit tests for optimizers: first-order, SciPy, QNG, second-order, and stochastic.
"""
import pytest
import numpy as np

from qpdesolver.optimizers import (
    GradientDescent,
    Adam,
    ScipyOptimizer,
    QuantumNaturalGradient,
    DampedNewton,
    SPSA,
    CountingObjective,
)
from qpdesolver.core.registry import REGISTRY_OPTIMIZERS

class QuadraticObjective:
    """Quadratic objective f(x) = 1/2 x^T A x - b^T x with exact grad, metric, hessian."""
    def __init__(self, A: np.ndarray, b: np.ndarray):
        self.A = A
        self.b = b
        self.n_params = len(b)

    def value(self, x: np.ndarray) -> float:
        return float(0.5 * x @ self.A @ x - self.b @ x)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.A @ x - self.b

    def metric_tensor(self, x: np.ndarray) -> np.ndarray:
        # Standard Euclidean metric for this test
        return np.eye(self.n_params)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        return self.A

    def state_overlap(self, x: np.ndarray, y: np.ndarray) -> float:
        # Distance squared proxy
        d = np.linalg.norm(x - y)
        return float(np.exp(-0.5 * d**2))

def test_optimizers_registered():
    assert "gradient_descent" in REGISTRY_OPTIMIZERS
    assert "adam" in REGISTRY_OPTIMIZERS
    assert "bfgs" in REGISTRY_OPTIMIZERS
    assert "qng" in REGISTRY_OPTIMIZERS
    assert "damped_newton" in REGISTRY_OPTIMIZERS
    assert "spsa" in REGISTRY_OPTIMIZERS

def test_optimizers_on_quadratic():
    A = np.array([[3.0, 1.0], [1.0, 2.0]])
    b = np.array([2.0, -1.0])
    obj = QuadraticObjective(A, b)
    x_opt = np.linalg.solve(A, b)
    f_opt = obj.value(x_opt)

    x0 = np.array([5.0, -5.0])
    f0 = obj.value(x0)
    total_drop = f0 - f_opt

    optimizers = [
        ("GD", GradientDescent(lr=0.2), 100),
        ("Adam", Adam(lr=0.2), 100),
        ("BFGS", ScipyOptimizer("BFGS"), 30),
        ("QNG", QuantumNaturalGradient(lr=0.2), 100),
        ("Newton", DampedNewton(), 10),
    ]

    for name, opt, iters in optimizers:
        res = opt.minimize(obj, x0, max_iter=iters)
        achieved_drop = f0 - res.fun
        rel_drop = achieved_drop / total_drop
        assert rel_drop > 0.99, f"Optimizer {name} failed: relative drop {rel_drop:.4f}"

def test_damped_newton_few_iterations():
    A = np.array([[4.0, 1.0], [1.0, 3.0]])
    b = np.array([1.0, 2.0])
    obj = QuadraticObjective(A, b)
    x0 = np.array([10.0, -10.0])

    newton = DampedNewton()
    res = newton.minimize(obj, x0, max_iter=10)
    assert res.n_iter <= 7
    assert np.allclose(res.x, np.linalg.solve(A, b), atol=1e-5)

def test_optimizer_requires_validation():
    class MinimalObjective:
        n_params = 2
        def value(self, x): return float(np.sum(x**2))

    min_obj = MinimalObjective()
    qng = QuantumNaturalGradient()
    # Missing grad and metric_tensor
    with pytest.raises(TypeError, match="requires objective method"):
        qng.minimize(min_obj, np.array([1.0, 1.0]))

def test_counting_objective():
    A = np.eye(2)
    b = np.zeros(2)
    obj = QuadraticObjective(A, b)
    counted = CountingObjective(obj)

    counted.value(np.zeros(2))
    counted.grad(np.zeros(2))
    counted.hessian(np.zeros(2))

    assert counted.counts["value"] == 1
    assert counted.counts["grad"] == 1
    assert counted.counts["hessian"] == 1
    assert counted.counts["metric_tensor"] == 0
