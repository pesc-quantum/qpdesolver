"""
Full Waveform Inversion application package.
"""
from qpdesolver.applications.fwi.model import (
    VelocityModel,
    Acquisition,
    FWIDataset,
    VelocityModel2D,
    Acquisition2D,
    FWIDataset2D,
)
from qpdesolver.applications.fwi.problem import FWIProblem, FWIProblem2D
from qpdesolver.applications.fwi.inversion import FWIInversion, InversionResult

__all__ = [
    "VelocityModel",
    "Acquisition",
    "FWIDataset",
    "VelocityModel2D",
    "Acquisition2D",
    "FWIDataset2D",
    "FWIProblem",
    "FWIProblem2D",
    "FWIInversion",
    "InversionResult",
]
