"""
Abstract base class for quantum execution backends.
"""
from abc import ABC, abstractmethod
from typing import Optional, Any, Callable

class Backend(ABC):
    """
    Abstract execution backend (PennyLane, Qiskit, CUDA-Q).
    """
    name: str = "base"
    shots: Optional[int] = None

    @property
    @abstractmethod
    def supports_statevector(self) -> bool:
        """Returns True if the backend supports direct statevector extraction."""
        pass

    @abstractmethod
    def hhl_kernel(self) -> Any:
        """Returns the HHL execution kernel for this backend."""
        pass
