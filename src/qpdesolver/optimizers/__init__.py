"""
Optimizers package: Euclidean 1st order, SciPy quasi-Newton, Riemannian QNG, Damped Newton, SPSA, QN-SPSA.
"""
from qpdesolver.optimizers.base import (
    Objective,
    CountingObjective,
    OptimizeResult,
    Optimizer,
)
from qpdesolver.optimizers.first_order import GradientDescent, Adam
from qpdesolver.optimizers.scipy_wrapper import ScipyOptimizer
from qpdesolver.optimizers.riemannian import QuantumNaturalGradient
from qpdesolver.optimizers.second_order import DampedNewton
from qpdesolver.optimizers.stochastic import SPSA, QNSPSA

__all__ = [
    "Objective",
    "CountingObjective",
    "OptimizeResult",
    "Optimizer",
    "GradientDescent",
    "Adam",
    "ScipyOptimizer",
    "QuantumNaturalGradient",
    "DampedNewton",
    "SPSA",
    "QNSPSA",
]
