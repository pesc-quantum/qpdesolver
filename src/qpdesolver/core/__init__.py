"""
Core utilities: registries and metrics.
"""
from qpdesolver.core.registry import (
    Registry,
    REGISTRY_ANSATZE,
    REGISTRY_OPTIMIZERS,
    REGISTRY_SOLVERS,
    REGISTRY_BACKENDS,
    REGISTRY_DISCRETIZATIONS,
    REGISTRY_INITIALIZERS,
    REGISTRY_COSTS,
)
from qpdesolver.core.metrics import fidelity, relative_residual, condition_number

__all__ = [
    "Registry",
    "REGISTRY_ANSATZE",
    "REGISTRY_OPTIMIZERS",
    "REGISTRY_SOLVERS",
    "REGISTRY_BACKENDS",
    "REGISTRY_DISCRETIZATIONS",
    "REGISTRY_INITIALIZERS",
    "REGISTRY_COSTS",
    "fidelity",
    "relative_residual",
    "condition_number",
]
