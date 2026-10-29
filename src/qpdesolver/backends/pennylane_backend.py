"""
PennyLane execution backend supporting default.qubit, lightning.qubit, and Qiskit plugin devices.
"""
from typing import Optional, Any, Callable, Sequence, Dict
import pennylane as qml

from qpdesolver.core.registry import REGISTRY_BACKENDS
from qpdesolver.backends.base import Backend

@REGISTRY_BACKENDS.register("pennylane")
class PennyLaneBackend(Backend):
    """
    PennyLane execution backend.
    """
    name = "pennylane"

    def __init__(
        self,
        device_name: str = "default.qubit",
        shots: Optional[int] = None,
        diff_method: Optional[str] = None,
        **device_kwargs
    ):
        self.device_name = device_name
        self.shots = shots
        if diff_method is not None:
            self.diff_method = diff_method
        elif "default.qubit" in device_name:
            self.diff_method = "backprop"
        else:
            self.diff_method = "parameter-shift"
        self.device_kwargs = device_kwargs

    @property
    def supports_statevector(self) -> bool:
        return self.shots is None

    def device(self, wires: Any) -> qml.Device:
        """Instantiates a PennyLane device with the configured wires and shots."""
        return qml.device(self.device_name, wires=wires, shots=self.shots, **self.device_kwargs)

    def qnode(self, fn: Callable, wires: Any, **qnode_kwargs) -> qml.QNode:
        """Constructs a QNode using this backend's device and diff_method."""
        dev = self.device(wires)
        kwargs = {"diff_method": self.diff_method}
        kwargs.update(qnode_kwargs)
        return qml.QNode(fn, dev, **kwargs)

    def pauli_op(self, label: str, wires: Sequence[int]) -> qml.operation.Operator:
        """
        Translates a Pauli string label (e.g. 'IXZ') into a PennyLane tensor operator.
        Wire 0 corresponds to label[0].
        """
        if len(label) != len(wires):
            raise ValueError(f"Label length {len(label)} must match number of wires {len(wires)}.")
        ops = []
        for ch, w in zip(label.upper(), wires):
            if ch == "I":
                ops.append(qml.Identity(w))
            elif ch == "X":
                ops.append(qml.PauliX(w))
            elif ch == "Y":
                ops.append(qml.PauliY(w))
            elif ch == "Z":
                ops.append(qml.PauliZ(w))
            else:
                raise ValueError(f"Invalid Pauli character '{ch}' in label '{label}'.")

        res = ops[0]
        for op in ops[1:]:
            res = res @ op
        return res

    def hhl_kernel(self) -> Any:
        from qpdesolver.solvers.hhl.pennylane_kernel import PennyLaneHHLKernel
        return PennyLaneHHLKernel(backend=self)
