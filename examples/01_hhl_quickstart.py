"""
Example 01: Quickstart solving a 1D Helmholtz PDE using the HHL algorithm.
"""
import numpy as np
from loguru import logger
from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.solvers.hhl.config import HHLConfig
from qpdesolver.solvers.hhl.solver import HHLSolver
from qpdesolver.backends.pennylane_backend import PennyLaneBackend

def main():
    logger.info("=== Example 01: HHL Solver Quickstart ===")
    
    # 1. Define physical problem: 1D Helmholtz equation on 4 grid points
    hz = Helmholtz1D(n_points=4, h=1.0, bc="dirichlet")
    v = np.array([2.0, 2.0, 2.5, 2.5])  # 2-layer velocity model
    omega = 1.0                         # angular frequency
    source = hz.point_source(0)         # point source injected at boundary
    system = hz.system(rhs=source, velocity=v, omega=omega, name="helmholtz_4pt")

    logger.info(f"Problem: {system.name}")
    logger.info(f"Grid size: {system.size} ({system.n_qubits} qubits)")
    logger.info(f"Condition number: kappa = {system.condition_number():.2f}")
    
    # 2. Configure HHL solver
    # 6 clock qubits -> 2^6 = 64 phase bins
    cfg = HHLConfig(
        n_clock=6,
        signed=True,          # signed QPE for indefinite Helmholtz operator
        evolution="exact",    # exact matrix exponential
        phase_margin=0.9,     # phase safety margin
    )
    backend = PennyLaneBackend(device_name="default.qubit")
    solver = HHLSolver(config=cfg, backend=backend)

    # 3. Solve
    logger.info("Running HHL simulation...")
    result = solver.solve(system)

    # 4. Display results
    logger.success("HHL Simulation Finished Successfully!")
    logger.info(f"  Fidelity with classical exact: {result.fidelity * 100:.3f}%")
    logger.info(f"  Relative residual:            {result.residual:.4e}")
    logger.info(f"  Success probability (ancilla):{result.success_probability * 100:.2f}%")
    logger.info(f"  Wall time:                     {result.wall_time:.3f} s")
    logger.info(f"  Circuit resources:             {result.resources}")
    logger.info(f"  Reconstructed solution x:     {np.round(result.x.real, 4)}")
    x_exact = system.classical_solution()
    logger.info(f"  Classical exact solution:      {np.round(x_exact.real, 4)}")

if __name__ == "__main__":
    main()
