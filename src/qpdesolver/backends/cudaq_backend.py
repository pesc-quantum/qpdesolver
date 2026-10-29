"""
Stub backend for native NVIDIA CUDA-Q execution.
CUDA-Q requires Linux/Docker execution with GPU accelerators.
Direct native CUDA-Q kernel porting guide is documented in PORTING.md.
"""
from typing import Optional, Any
from qpdesolver.core.registry import REGISTRY_BACKENDS
from qpdesolver.backends.base import Backend

@REGISTRY_BACKENDS.register("cudaq")
class CudaQBackend(Backend):
    """
    NVIDIA CUDA-Q backend stub.
    Raises NotImplementedError with clear instructions to PORTING.md.
    """
    name = "cudaq"

    def __init__(self, target: str = "qpp-cpu", shots: Optional[int] = None):
        self.target = target
        self.shots = shots

    @property
    def supports_statevector(self) -> bool:
        return self.shots is None

    def hhl_kernel(self) -> Any:
        raise NotImplementedError(
            "Native CUDA-Q HHL kernel is not implemented yet. "
            "Please refer to PORTING.md for implementation instructions, "
            "or use PennyLaneBackend on macOS."
        )
