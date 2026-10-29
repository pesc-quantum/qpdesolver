"""
Cost functions for Variational Quantum Linear Solver (VQLS).
Designed for open research and rapid prototyping:
- GlobalCost: inner product overlap cost.
- LocalCost: local projector overlap cost for barren plateau mitigation.
- CustomCostFunction: wraps an arbitrary user-defined loss formulation.
- Registry support via REGISTRY_COSTS.
"""
from abc import ABC, abstractmethod
from typing import Callable, Any, Union, Optional
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qpdesolver.core.registry import REGISTRY_COSTS
from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.ansatze.base import Ansatz
from qpdesolver.backends.pennylane_backend import PennyLaneBackend

class VQLSCost(ABC):
    """
    Abstract VQLS cost function builder.
    Researchers can subclass this to implement novel loss landscapes,
    regularization terms, local decompositions, or noise-mitigated observables.
    """
    requires_statevector: bool = True

    @abstractmethod
    def build(
        self,
        system: LinearSystem,
        ansatz: Ansatz,
        backend: PennyLaneBackend,
    ) -> Callable[[Any], Any]:
        """Constructs an autograd-differentiable cost function cost(params) -> float."""
        pass

@REGISTRY_COSTS.register("global")
@REGISTRY_COSTS.register("global_cost")
class GlobalCost(VQLSCost):
    """
    Global overlap inner-product cost function (Bravo-Prieto et al. 2023):
        C_G(theta) = 1 - |<b | A | x(theta)>|^2 / <x(theta) | A^dagger A | x(theta)>
    Differentiable via autograd backpropagation or parameter-shift.
    """
    def __init__(self, estimator: str = "statevector"):
        self.estimator = estimator
        self.requires_statevector = (estimator == "statevector")

    def build(
        self,
        system: LinearSystem,
        ansatz: Ansatz,
        backend: PennyLaneBackend,
    ) -> Callable[[Any], Any]:
        dev = backend.device(system.n_qubits)
        wires = list(range(system.n_qubits))

        @qml.qnode(dev, diff_method=backend.diff_method)
        def state_circuit(params):
            ansatz.apply(params, wires)
            return qml.state()

        A_mat = pnp.array(system.A, dtype=complex)
        b_vec = pnp.array(system.b_state, dtype=complex)

        def cost_fn(params):
            state = state_circuit(params)
            Ax = pnp.dot(A_mat, state)
            denom = pnp.sum(pnp.real(Ax * pnp.conj(Ax)))
            overlap = pnp.sum(pnp.conj(b_vec) * Ax)
            num = pnp.real(overlap * pnp.conj(overlap))
            return 1.0 - (num / (denom + 1e-15))

        return cost_fn

@REGISTRY_COSTS.register("local")
@REGISTRY_COSTS.register("local_cost")
class LocalCost(VQLSCost):
    """
    Local Hamiltonian overlap cost function:
        C_L(theta) = 1 - 1/n sum_j |<b | A^\dagger (Z_j (x) I) A | x(theta)>|^2
    Mitigates barren plateaus for n > 4 qubits by avoiding global projectors.
    """
    def __init__(self):
        self.requires_statevector = True

    def build(
        self,
        system: LinearSystem,
        ansatz: Ansatz,
        backend: PennyLaneBackend,
    ) -> Callable[[Any], Any]:
        dev = backend.device(system.n_qubits)
        wires = list(range(system.n_qubits))

        @qml.qnode(dev, diff_method=backend.diff_method)
        def state_circuit(params):
            ansatz.apply(params, wires)
            return qml.state()

        A_mat = pnp.array(system.A, dtype=complex)
        b_vec = pnp.array(system.b_state, dtype=complex)
        n_q = system.n_qubits

        def cost_fn(params):
            state = state_circuit(params)
            Ax = pnp.dot(A_mat, state)
            denom = pnp.sum(pnp.real(Ax * pnp.conj(Ax))) + 1e-15

            local_overlaps = []
            for j in range(n_q):
                Z_j = np.eye(2**n_q, dtype=complex)
                for bit in range(2**n_q):
                    if (bit >> (n_q - 1 - j)) & 1:
                        Z_j[bit, bit] = -1.0
                proj = 0.5 * (np.eye(2**n_q, dtype=complex) + Z_j)
                proj_mat = pnp.array(proj, dtype=complex)
                proj_Ax = pnp.dot(proj_mat, Ax)
                overlap = pnp.sum(pnp.conj(b_vec) * proj_Ax)
                local_overlaps.append(pnp.real(overlap * pnp.conj(overlap)))

            num = pnp.sum(pnp.stack(local_overlaps)) / n_q
            return 1.0 - (num / denom)

        return cost_fn

@REGISTRY_COSTS.register("custom")
class CustomCostFunction(VQLSCost):
    """
    Adapter allowing researchers to supply an arbitrary user-defined cost callable:
        fn(state, Ax, b, system, params) -> float
    or
        fn(params, state_circuit, system) -> float
    giving total freedom to test regularization, entropic penalties, or novel metrics.
    """
    def __init__(self, loss_fn: Callable[..., Any]):
        self.loss_fn = loss_fn
        self.requires_statevector = True

    def build(
        self,
        system: LinearSystem,
        ansatz: Ansatz,
        backend: PennyLaneBackend,
    ) -> Callable[[Any], Any]:
        dev = backend.device(system.n_qubits)
        wires = list(range(system.n_qubits))

        @qml.qnode(dev, diff_method=backend.diff_method)
        def state_circuit(params):
            ansatz.apply(params, wires)
            return qml.state()

        A_mat = pnp.array(system.A, dtype=complex)
        b_vec = pnp.array(system.b_state, dtype=complex)

        def cost_fn(params):
            state = state_circuit(params)
            Ax = pnp.dot(A_mat, state)
            # Try calling user loss with (state, Ax, b, system, params)
            try:
                return self.loss_fn(state, Ax, b_vec, system, params)
            except TypeError:
                return self.loss_fn(state, Ax, b_vec)

        return cost_fn

def make_cost(cost_spec: Union[str, VQLSCost, Callable], **kwargs) -> VQLSCost:
    """
    Convenience factory to create or adapt cost functions.
    Supports registry strings, VQLSCost instances, or arbitrary callables.
    """
    if isinstance(cost_spec, VQLSCost):
        return cost_spec
    if isinstance(cost_spec, str):
        return REGISTRY_COSTS.create(cost_spec, **kwargs)
    if callable(cost_spec):
        return CustomCostFunction(cost_spec)
    raise ValueError(f"Unsupported cost specification: {cost_spec}")
