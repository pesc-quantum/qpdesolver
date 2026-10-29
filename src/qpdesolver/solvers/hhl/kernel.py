"""
Abstract interface and output container for HHL quantum circuit execution kernels.
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.solvers.hhl.config import HHLConfig
from qpdesolver.solvers.hhl.planning import HHLPlan

@dataclass
class HHLRawOutput:
    """
    Raw output from the quantum circuit execution before classical postprocessing.
    """
    postselected: np.ndarray
    success_probability: float
    resources: Dict[str, Any]

class HHLKernel(ABC):
    """
    Abstract HHL circuit executor. Implementations exist for PennyLane, Qiskit, CUDA-Q.
    """
    @abstractmethod
    def run(self, system: LinearSystem, plan: HHLPlan, cfg: HHLConfig) -> HHLRawOutput:
        pass
