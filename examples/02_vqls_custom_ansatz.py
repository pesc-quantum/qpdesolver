"""
Example 02: Solving a PDE with VQLS using a custom user-defined ansatz.
"""
import numpy as np
import pennylane as qml
from loguru import logger

from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.ansatze.adapters import FunctionAnsatz
from qpdesolver.solvers.vqls.solver import VQLSSolver
from qpdesolver.optimizers.scipy_wrapper import ScipyOptimizer

def main():
    logger.info("=== Example 02: VQLS with Custom User-Defined Ansatz ===")

    # 1. Physical Helmholtz problem
    hz = Helmholtz1D(n_points=4, h=1.0, bc="dirichlet")
    system = hz.system(rhs=hz.point_source(0), velocity=2.0, omega=1.0)
    logger.info(f"Target system: {system.name} ({system.n_qubits} qubits)")

    # 2. Define a custom ansatz not in PennyLane templates
    # Custom alternating rotations with ring CNOTs
    def my_custom_circuit(params, wires):
        # Layer 1: RX and RY
        for idx, w in enumerate(wires):
            qml.RX(params[2 * idx], wires=w)
            qml.RY(params[2 * idx + 1], wires=w)
        # Ring entanglement
        qml.broadcast(qml.CNOT, wires=wires, pattern="ring")
        # Layer 2: Final RY rotations
        offset = 2 * len(wires)
        for idx, w in enumerate(wires):
            qml.RY(params[offset + idx], wires=w)

    custom_ansatz = FunctionAnsatz(
        fn=my_custom_circuit,
        num_params_fn=lambda n_wires: 3 * n_wires, # 6 params for 2 qubits
    )

    # 3. Instantiate and run VQLS
    solver = VQLSSolver(
        ansatz=custom_ansatz,
        optimizer=ScipyOptimizer("BFGS"),
        max_iter=100,
        seed=42,
    )

    logger.info("Solving linear system with VQLS...")
    result = solver.solve(system)

    logger.success("VQLS Solve Completed!")
    logger.info(f"  Fidelity:          {result.fidelity * 100:.3f}%")
    logger.info(f"  Relative residual: {result.residual:.4e}")
    logger.info(f"  Final cost:        {result.metadata['final_cost']:.4e}")
    logger.info(f"  Circuit evals:     {result.n_circuit_evals}")

if __name__ == "__main__":
    main()
