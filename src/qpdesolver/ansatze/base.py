"""
Base class for parameterized quantum circuit ansatze.
"""
from abc import ABC, abstractmethod
from typing import Sequence, List, Optional, Any
import numpy as np

class Ansatz(ABC):
    """
    Abstract base class for variational ansatze.
    Compatible with PennyLane circuits and 17-Jul-2026 __call__ interface.
    """
    @abstractmethod
    def num_params(self, n_wires: int) -> int:
        """Returns the total number of variational parameters required for n_wires."""
        pass

    @abstractmethod
    def apply(self, params: Any, wires: Sequence[int]) -> None:
        """Applies the circuit operations onto the given wires in PennyLane."""
        pass

    def __call__(self, params: Any, wires: Sequence[int]) -> None:
        """Calls apply for backwards compatibility with 17-Jul vqls_core."""
        self.apply(params, wires)

    @property
    def n_layers(self) -> Optional[int]:
        """Number of repeating layers, if applicable."""
        return getattr(self, "layers", None)

    def with_layers(self, n_layers: int) -> "Ansatz":
        """Returns a copy of this ansatz configured with a different layer count."""
        raise NotImplementedError(f"Ansatz {self.__class__.__name__} does not support with_layers.")

    def layer_param_slices(self, n_wires: int) -> List[slice]:
        """Returns a list of slices partitioning flat params into layers."""
        raise NotImplementedError(f"Ansatz {self.__class__.__name__} does not define layer_param_slices.")

    def init_params(
        self,
        n_wires: int,
        rng: Optional[np.random.Generator] = None,
        strategy: str = "uniform",
        **kwargs
    ) -> np.ndarray:
        """Initializes parameters using a registered initialization strategy."""
        from qpdesolver.ansatze.initialization import initialize_params
        n_p = self.num_params(n_wires)
        return initialize_params(n_p, ansatz=self, n_wires=n_wires, rng=rng, strategy=strategy, **kwargs)
