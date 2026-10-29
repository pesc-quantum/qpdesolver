"""
PennyLane implementation of the HHL quantum circuit kernel.
"""
from typing import Dict, Any, List
import numpy as np
import scipy.linalg
import pennylane as qml

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.solvers.hhl.config import HHLConfig
from qpdesolver.solvers.hhl.planning import HHLPlan
from qpdesolver.solvers.hhl.kernel import HHLKernel, HHLRawOutput
from qpdesolver.backends.pennylane_backend import PennyLaneBackend

class PennyLaneHHLKernel(HHLKernel):
    """
    Executes HHL on PennyLane devices.
    """
    def __init__(self, backend: PennyLaneBackend):
        self.backend = backend

    def _build_evolution_op(self, system: LinearSystem, plan: HHLPlan, cfg: HHLConfig, sys_w: List[int]):
        """Constructs the unitary evolution operation U = e^{i A t}."""
        if cfg.evolution == "exact":
            # Direct matrix exponential U = exp(i * A * t)
            U_mat = scipy.linalg.expm(1j * system.A * plan.t)
            return qml.QubitUnitary(U_mat, wires=sys_w)
        elif cfg.evolution == "trotter":
            # Trotterized Hamiltonian simulation
            terms = system.pauli_terms()
            coeffs = [c.real if np.isreal(c) else c for c, _ in terms]
            ops = [self.backend.pauli_op(lbl, sys_w) for _, lbl in terms]
            H = qml.dot(coeffs, ops)
            # TrotterProduct in PennyLane evaluates to exp(+i H time) when passed time=plan.t
            return qml.TrotterProduct(H, time=plan.t, n=cfg.trotter_steps, order=cfg.trotter_order)
        else:
            raise ValueError(f"Unknown evolution method '{cfg.evolution}'. Must be 'exact' or 'trotter'.")

    def run(self, system: LinearSystem, plan: HHLPlan, cfg: HHLConfig) -> HHLRawOutput:
        n_sys = system.n_qubits
        n_clk = plan.n_clock
        total_wires = n_sys + n_clk + 1

        sys_w = list(range(n_sys))
        clk_w = list(range(n_sys, n_sys + n_clk))
        anc_w = n_sys + n_clk

        dev = self.backend.device(total_wires)

        def circuit():
            # 1. State preparation for |b> on system register
            qml.StatePrep(system.b_state, wires=sys_w)

            # 2. Forward Quantum Phase Estimation
            u_op = self._build_evolution_op(system, plan, cfg, sys_w)
            qml.QuantumPhaseEstimation(u_op, estimation_wires=clk_w)

            # 3. Controlled-RY rotations on ancilla
            for k, theta in enumerate(plan.angles):
                if theta != 0.0:
                    bits = [int(ch) for ch in np.binary_repr(k, n_clk)]
                    qml.ctrl(qml.RY(theta, wires=anc_w), control=clk_w, control_values=bits)

            # 4. Inverse Quantum Phase Estimation (uncomputing)
            # CAUTION: construct op directly inside adjoint to prevent stray queueing
            u_op_inv = self._build_evolution_op(system, plan, cfg, sys_w)
            qml.adjoint(qml.QuantumPhaseEstimation(u_op_inv, estimation_wires=clk_w))

            return qml.state()

        qnode = qml.QNode(circuit, dev)

        # Inspect resources
        try:
            specs = qml.specs(qnode)()
            resources_data = {
                "num_wires": total_wires,
                "num_gates": specs["resources"].num_gates,
                "depth": specs["resources"].depth,
                "gate_types": dict(specs["resources"].gate_types),
            }
        except Exception:
            resources_data = {"num_wires": total_wires}

        # Execute circuit
        state = qnode()

        # Reshape to (2^n_sys, 2^n_clk, 2)
        # Clock register wire 0 is MSB in binary_repr
        tensor = state.reshape((2**n_sys, 2**n_clk, 2))
        # Postselect on: clock register all zeros (index 0) AND ancilla == 1 (index 1)
        postselected = tensor[:, 0, 1]
        p_success = float(np.vdot(postselected, postselected).real)

        return HHLRawOutput(
            postselected=postselected,
            success_probability=p_success,
            resources=resources_data,
        )
