"""
Full Waveform Inversion application package.
"""
from qpdesolver.applications.fwi.model import VelocityModel, Acquisition, FWIDataset
from qpdesolver.applications.fwi.problem import FWIProblem
from qpdesolver.applications.fwi.inversion import FWIInversion, InversionResult

__all__ = [
    "VelocityModel",
    "Acquisition",
    "FWIDataset",
    "FWIProblem",
    "FWIInversion",
    "InversionResult",
]
