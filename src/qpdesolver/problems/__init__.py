"""
Problems module: LinearSystem, Pauli decomposition, and PDE discretizations.
"""
from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.problems.pauli import pauli_decompose, pauli_matrix
from qpdesolver.problems.discretizations import (
    PDEDiscretization,
    Helmholtz1D,
    Poisson1D,
    get_legacy_17jul_helmholtz,
    get_legacy_17jul_mock_fwi,
)

__all__ = [
    "LinearSystem",
    "pauli_decompose",
    "pauli_matrix",
    "PDEDiscretization",
    "Helmholtz1D",
    "Poisson1D",
    "get_legacy_17jul_helmholtz",
    "get_legacy_17jul_mock_fwi",
]
