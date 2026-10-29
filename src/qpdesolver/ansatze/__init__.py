"""
Ansatze package: base class, library, adapters, and parameter initialization strategies.
"""
from qpdesolver.ansatze.base import Ansatz
from qpdesolver.ansatze.initialization import initialize_params
from qpdesolver.ansatze.library import (
    RealAmplitudes,
    HardwareEfficient,
    AlternatingLayered,
    HamiltonianVariational,
)
from qpdesolver.ansatze.adapters import (
    TemplateAnsatz,
    FunctionAnsatz,
    LegacyAnsatzAdapter,
)

__all__ = [
    "Ansatz",
    "initialize_params",
    "RealAmplitudes",
    "HardwareEfficient",
    "AlternatingLayered",
    "HamiltonianVariational",
    "TemplateAnsatz",
    "FunctionAnsatz",
    "LegacyAnsatzAdapter",
]
