"""
Unit tests for execution backends.
"""
import pytest
from qpdesolver.backends.pennylane_backend import PennyLaneBackend
from qpdesolver.backends.qiskit_backend import QiskitBackend
from qpdesolver.backends.cudaq_backend import CudaQBackend
from qpdesolver.core.registry import REGISTRY_BACKENDS

def test_backends_registered():
    assert "pennylane" in REGISTRY_BACKENDS
    assert "qiskit" in REGISTRY_BACKENDS
    assert "cudaq" in REGISTRY_BACKENDS

def test_pennylane_backend_instantiation():
    backend = PennyLaneBackend(device_name="default.qubit")
    assert backend.supports_statevector is True

    op = backend.pauli_op("XZ", wires=[0, 1])
    assert op.name == "Prod" or hasattr(op, "matrix")

def test_stubs_raise_not_implemented():
    q_backend = QiskitBackend()
    with pytest.raises(NotImplementedError, match="PORTING.md"):
        q_backend.hhl_kernel()

    c_backend = CudaQBackend()
    with pytest.raises(NotImplementedError, match="PORTING.md"):
        c_backend.hhl_kernel()
