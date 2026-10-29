"""
Stub backend for native Qiskit execution.
To use Qiskit backends with PennyLane in the current version, use:
    PennyLaneBackend(device_name="qiskit.aer") or "qiskit.remote"
with the pennylane-qiskit plugin.
Direct native Qiskit kernel porting guide is documented in PORTING.md.
"""
from typing import Optional, Any
from qpdesolver.core.registry import REGISTRY_BACKENDS
from qpdesolver.backends.base import Backend

@REGISTRY_BACKENDS.register("qiskit")
class QiskitBackend(Backend):
    """
    Direct Qiskit backend stub.
    Raises NotImplementedError with clear instructions to PORTING.md.
    """
    name = "qiskit"

    def __init__(self, shots: Optional[int] = None):
        self.shots = shots

    @property
    def supports_statevector(self) -> bool:
        return self.shots is None

    def hhl_kernel(self) -> Any:
        raise NotImplementedError(
            "Native Qiskit HHL kernel is not implemented yet. "
            "Please refer to PORTING.md for implementation instructions, "
            "or use PennyLaneBackend(device_name='qiskit.aer') via pennylane-qiskit."
        )
