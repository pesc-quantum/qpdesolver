"""
VQLSObjective implements the Objective protocol with autograd, QFIM metric tensor, and Hessians.
"""
from typing import Optional, Any
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.ansatze.base import Ansatz
from qpdesolver.solvers.vqls.costs import VQLSCost, GlobalCost
from qpdesolver.backends.pennylane_backend import PennyLaneBackend

class VQLSObjective:
    """
    Evaluates VQLS cost, analytical gradients, metric tensor, Hessian, and state overlaps.
    """
    def __init__(
        self,
        system: LinearSystem,
        ansatz: Ansatz,
        cost: Optional[VQLSCost] = None,
        backend: Optional[PennyLaneBackend] = None,
    ):
        self.system = system
        self.ansatz = ansatz
        self.cost = cost or GlobalCost()
        self.backend = backend or PennyLaneBackend()
        self.n_params = ansatz.num_params(system.n_qubits)

        self._cost_fn = self.cost.build(self.system, self.ansatz, self.backend)
        self._grad_fn = qml.grad(self._cost_fn)

        # Build dedicated statecircuit for metric_tensor and state_overlap
        dev = self.backend.device(system.n_qubits)
        wires = list(range(system.n_qubits))

        @qml.qnode(dev, diff_method=self.backend.diff_method)
        def state_circuit(params):
            self.ansatz.apply(params, wires)
            return qml.state()

        self._state_circuit = state_circuit

    def value(self, x: np.ndarray) -> float:
        px = pnp.array(x, requires_grad=True)
        return float(self._cost_fn(px))

    def grad(self, x: np.ndarray) -> np.ndarray:
        px = pnp.array(x, requires_grad=True)
        g = self._grad_fn(px)
        return np.array(g, dtype=float)

    def metric_tensor(self, x: np.ndarray, approx: str = "block-diag") -> np.ndarray:
        """Computes the Fubini-Study / quantum geometric tensor metric on the ansatz parameters."""
        px = pnp.array(x, requires_grad=True)
        # Compute real and imaginary Jacobians separately to handle complex statevectors
        def real_state(p):
            return pnp.real(self._state_circuit(p))

        def imag_state(p):
            return pnp.imag(self._state_circuit(p))

        jac_r = qml.jacobian(real_state)(px)
        jac_i = qml.jacobian(imag_state)(px)
        jac = np.array(jac_r, dtype=float) + 1j * np.array(jac_i, dtype=float)
        psi = np.array(self._state_circuit(px), dtype=complex)

        # Fubini-Study metric:
        # g_ij = Re( <d_i psi | d_j psi> - <d_i psi | psi> <psi | d_j psi> )
        overlap_jac = np.conj(jac.T) @ jac  # (n_params, n_params)
        psi_grad_overlap = np.conj(jac.T) @ psi  # (n_params,)
        fs_metric = np.real(overlap_jac - np.outer(psi_grad_overlap, np.conj(psi_grad_overlap)))
        # Symmetrize
        return 0.5 * (fs_metric + fs_metric.T)

    def hessian(self, x: np.ndarray) -> np.ndarray:
        """Computes exact Hessian via automatic differentiation."""
        px = pnp.array(x, requires_grad=True)
        hess = qml.jacobian(self._grad_fn)(px)
        return np.array(hess, dtype=float)

    def state_overlap(self, x: np.ndarray, y: np.ndarray) -> float:
        """Computes fidelity |<psi(x)|psi(y)>|^2 between two parameter states."""
        px = pnp.array(x, requires_grad=False)
        py = pnp.array(y, requires_grad=False)
        s1 = self._state_circuit(px)
        s2 = self._state_circuit(py)
        overlap = np.vdot(s1, s2)
        return float(np.abs(overlap)**2)

    def get_state(self, x: np.ndarray) -> np.ndarray:
        """Evaluates normalized statevector |x(theta)>."""
        px = pnp.array(x, requires_grad=False)
        return np.array(self._state_circuit(px))
