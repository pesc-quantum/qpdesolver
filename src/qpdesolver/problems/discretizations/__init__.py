"""
Discretization modules for PDEs.
"""
from qpdesolver.problems.discretizations.base import PDEDiscretization
from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.problems.discretizations.helmholtz2d import Helmholtz2D
from qpdesolver.problems.discretizations.poisson import Poisson1D
from qpdesolver.problems.discretizations.spectral import FourierSpectral1D
from qpdesolver.problems.discretizations.finite_element import FiniteElement1D
from qpdesolver.problems.discretizations.legacy_17jul import (
    get_legacy_17jul_helmholtz,
    get_legacy_17jul_mock_fwi,
)

__all__ = [
    "PDEDiscretization",
    "Helmholtz1D",
    "Helmholtz2D",
    "Poisson1D",
    "FourierSpectral1D",
    "FiniteElement1D",
    "get_legacy_17jul_helmholtz",
    "get_legacy_17jul_mock_fwi",
]
