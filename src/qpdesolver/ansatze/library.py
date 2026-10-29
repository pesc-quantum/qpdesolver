"""
Concrete ansatz implementations for variational solvers and trainability studies.
"""
from typing import Sequence, List, Tuple, Optional, Any
import numpy as np
import pennylane as qml

from qpdesolver.core.registry import REGISTRY_ANSATZE
from qpdesolver.ansatze.base import Ansatz

@REGISTRY_ANSATZE.register("realamplitudes")
class RealAmplitudes(Ansatz):
    """
    RealAmplitudes ansatz: alternating layers of RY rotations and linear CNOT entanglers.
    Strictly preserves real quantum state amplitudes.
    Parameter layout is bit-for-bit identical to 17-Jul-2026 RealAmplitudesAnsatz.
    """
    def __init__(self, layers: int = 2):
        self.layers = int(layers)

    def num_params(self, n_wires: int) -> int:
        return n_wires * (self.layers + 1)

    def apply(self, params: Any, wires: Sequence[int]) -> None:
        n_qubits = len(wires)
        idx = 0
        # Initial rotation layer
        for w in wires:
            qml.RY(params[idx], wires=w)
            idx += 1

        for _ in range(self.layers):
            # Entangling layer (Linear CNOTs)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[wires[i], wires[i + 1]])
            # Rotations
            for w in wires:
                qml.RY(params[idx], wires=w)
                idx += 1

    def with_layers(self, n_layers: int) -> "RealAmplitudes":
        return RealAmplitudes(layers=n_layers)

    def layer_param_slices(self, n_wires: int) -> List[slice]:
        slices = []
        for l in range(self.layers + 1):
            slices.append(slice(l * n_wires, (l + 1) * n_wires))
        return slices

@REGISTRY_ANSATZE.register("hardwareefficient")
class HardwareEfficient(Ansatz):
    """
    Generalized Hardware-Efficient Ansatz (HEA).
    Supports arbitrary single-qubit rotation families (e.g. RY, RZ, RX)
    and entanglement topologies ('linear', 'ring', 'full', 'none').
    """
    def __init__(
        self,
        layers: int = 2,
        rotations: Sequence[str] = ("RY", "RZ"),
        entangler: str = "linear",
        entangling_gate: str = "CNOT",
    ):
        self.layers = int(layers)
        self.rotations = tuple(r.upper() for r in rotations)
        self.entangler = entangler.lower()
        self.entangling_gate = entangling_gate.upper()

    @property
    def rot_per_qubit(self) -> int:
        return len(self.rotations)

    def num_params(self, n_wires: int) -> int:
        return n_wires * self.rot_per_qubit * (self.layers + 1)

    def _apply_rotation(self, op_name: str, theta: Any, wire: int) -> None:
        if op_name == "RX":
            qml.RX(theta, wires=wire)
        elif op_name == "RY":
            qml.RY(theta, wires=wire)
        elif op_name == "RZ":
            qml.RZ(theta, wires=wire)
        else:
            raise ValueError(f"Unsupported rotation gate '{op_name}'.")

    def _apply_entangler(self, w1: int, w2: int) -> None:
        if self.entangling_gate == "CNOT":
            qml.CNOT(wires=[w1, w2])
        elif self.entangling_gate == "CZ":
            qml.CZ(wires=[w1, w2])
        else:
            raise ValueError(f"Unsupported entangling gate '{self.entangling_gate}'.")

    def apply(self, params: Any, wires: Sequence[int]) -> None:
        n_qubits = len(wires)
        idx = 0

        def rotation_block():
            nonlocal idx
            for w in wires:
                for rot in self.rotations:
                    self._apply_rotation(rot, params[idx], w)
                    idx += 1

        rotation_block()

        for _ in range(self.layers):
            # Entangling layer
            if self.entangler == "linear":
                for i in range(n_qubits - 1):
                    self._apply_entangler(wires[i], wires[i + 1])
            elif self.entangler == "ring":
                for i in range(n_qubits):
                    self._apply_entangler(wires[i], wires[(i + 1) % n_qubits])
            elif self.entangler == "full":
                for i in range(n_qubits):
                    for j in range(i + 1, n_qubits):
                        self._apply_entangler(wires[i], wires[j])

            rotation_block()

    def with_layers(self, n_layers: int) -> "HardwareEfficient":
        return HardwareEfficient(
            layers=n_layers,
            rotations=self.rotations,
            entangler=self.entangler,
            entangling_gate=self.entangling_gate,
        )

    def layer_param_slices(self, n_wires: int) -> List[slice]:
        step = n_wires * self.rot_per_qubit
        return [slice(l * step, (l + 1) * step) for l in range(self.layers + 1)]

@REGISTRY_ANSATZE.register("alternatinglayered")
class AlternatingLayered(Ansatz):
    """
    Alternating brickwork layered ansatz with localized 2-qubit blocks.
    Known to mitigate barren plateaus at logarithmic depths (Cerezo et al. 2021).
    """
    def __init__(self, layers: int = 2, block_size: int = 2):
        self.layers = int(layers)
        self.block_size = int(block_size)

    def num_params(self, n_wires: int) -> int:
        # Each layer applies 2 rotations per active pair
        # Even layer: pairs (0,1), (2,3), ... -> n_wires params
        # Odd layer: pairs (1,2), (3,4), ... -> (n_wires - 2) params
        total = 0
        for l in range(self.layers):
            offset = l % 2
            n_pairs = (n_wires - offset) // 2
            total += n_pairs * 4 # 2 RY + 2 RZ per pair
        return max(total, 1)

    def apply(self, params: Any, wires: Sequence[int]) -> None:
        n_qubits = len(wires)
        idx = 0
        for l in range(self.layers):
            offset = l % 2
            for i in range(offset, n_qubits - 1, 2):
                w1, w2 = wires[i], wires[i + 1]
                qml.RY(params[idx], wires=w1); idx += 1
                qml.RY(params[idx], wires=w2); idx += 1
                qml.CNOT(wires=[w1, w2])
                qml.RZ(params[idx], wires=w1); idx += 1
                qml.RZ(params[idx], wires=w2); idx += 1

    def with_layers(self, n_layers: int) -> "AlternatingLayered":
        return AlternatingLayered(layers=n_layers, block_size=self.block_size)

@REGISTRY_ANSATZE.register("hamiltonianvariational")
class HamiltonianVariational(Ansatz):
    """
    Problem-inspired Hamiltonian Variational Ansatz:
        U(theta) = prod_{l=1}^L prod_{k=1}^K exp(-i * theta_{l,k} * H_k)
    where {H_k} are the Hamiltonian Pauli components.
    """
    def __init__(self, generators: Sequence[str], layers: int = 2):
        self.generators = list(generators)
        self.layers = int(layers)

    def num_params(self, n_wires: int) -> int:
        return len(self.generators) * self.layers

    def apply(self, params: Any, wires: Sequence[int]) -> None:
        idx = 0
        for _ in range(self.layers):
            for gen in self.generators:
                # Build Pauli observable
                ops = []
                for ch, w in zip(gen.upper(), wires):
                    if ch == "X": ops.append(qml.PauliX(w))
                    elif ch == "Y": ops.append(qml.PauliY(w))
                    elif ch == "Z": ops.append(qml.PauliZ(w))
                if ops:
                    obs = ops[0]
                    for op in ops[1:]: obs = obs @ op
                    qml.exp(obs, 1j * params[idx]) # or TrotterProduct
                idx += 1

    def with_layers(self, n_layers: int) -> "HamiltonianVariational":
        return HamiltonianVariational(generators=self.generators, layers=n_layers)
