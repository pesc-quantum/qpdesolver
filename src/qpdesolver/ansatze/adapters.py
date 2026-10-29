"""
Adapters for wrapping arbitrary callables, PennyLane templates, and legacy 17-Jul ansatze.
"""
from typing import Callable, Sequence, Any, Optional
import numpy as np

from qpdesolver.core.registry import REGISTRY_ANSATZE
from qpdesolver.ansatze.base import Ansatz

@REGISTRY_ANSATZE.register("template")
class TemplateAnsatz(Ansatz):
    """
    Wraps a PennyLane template (e.g. StronglyEntanglingLayers, SimplifiedTwoDesign)
    and flattens/unflattens its multi-dimensional parameter tensor automatically.
    """
    def __init__(self, template_cls: Any, n_layers: int = 2, **template_kwargs):
        self.template_cls = template_cls
        self.layers = int(n_layers)
        self.template_kwargs = template_kwargs

    def _get_shape(self, n_wires: int):
        if hasattr(self.template_cls, "shape"):
            return self.template_cls.shape(n_layers=self.layers, n_wires=n_wires)
        return (self.layers, n_wires, 3)

    def num_params(self, n_wires: int) -> int:
        shape = self._get_shape(n_wires)
        return int(np.prod(shape))

    def apply(self, params: Any, wires: Sequence[int]) -> None:
        shape = self._get_shape(len(wires))
        shaped_params = np.reshape(params, shape)
        self.template_cls(weights=shaped_params, wires=wires, **self.template_kwargs)

    def with_layers(self, n_layers: int) -> "TemplateAnsatz":
        return TemplateAnsatz(self.template_cls, n_layers=n_layers, **self.template_kwargs)

@REGISTRY_ANSATZE.register("function")
class FunctionAnsatz(Ansatz):
    """
    Wraps an arbitrary user function fn(params, wires) as an Ansatz.
    """
    def __init__(self, fn: Callable[[Any, Sequence[int]], None], num_params_fn: Callable[[int], int]):
        self.fn = fn
        self.num_params_fn = num_params_fn

    def num_params(self, n_wires: int) -> int:
        return self.num_params_fn(n_wires)

    def apply(self, params: Any, wires: Sequence[int]) -> None:
        self.fn(params, wires)

@REGISTRY_ANSATZE.register("legacy")
class LegacyAnsatzAdapter(Ansatz):
    """
    Wraps a 17-Jul-2026 vqls_core.base.Ansatz instance for seamless backwards compatibility.
    """
    def __init__(self, legacy_ansatz: Any):
        self.legacy_ansatz = legacy_ansatz
        self.layers = getattr(legacy_ansatz, "layers", None)

    def num_params(self, n_wires: int) -> int:
        return self.legacy_ansatz.num_params(n_wires)

    def apply(self, params: Any, wires: Sequence[int]) -> None:
        self.legacy_ansatz(params, wires)
