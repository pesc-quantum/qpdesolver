"""
Example 03: Comparing different optimizers (BFGS, Adam, QNG, Damped Newton, SPSA) on VQLS.
"""
from loguru import logger
import numpy as np

from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.ansatze.library import RealAmplitudes
from qpdesolver.solvers.vqls.solver import VQLSSolver
from qpdesolver.optimizers import (
    ScipyOptimizer,
    Adam,
    QuantumNaturalGradient,
    DampedNewton,
    SPSA,
)

def main():
    logger.info("=== Example 03: VQLS Optimizer Zoo Benchmark ===")

    # Setup 4-point Helmholtz problem
    hz = Helmholtz1D(n_points=4, h=1.0, bc="dirichlet")
    system = hz.system(rhs=hz.point_source(0), velocity=2.0, omega=1.0)
    ansatz = RealAmplitudes(layers=2)

    optimizers = [
        ("BFGS (Quasi-Newton)", ScipyOptimizer("BFGS"), 60),
        ("Adam (Adaptive 1st-order)", Adam(lr=0.08), 100),
        ("QNG (Riemannian / Fubini-Study)", QuantumNaturalGradient(lr=0.15), 60),
        ("Damped Newton (2nd-order / LM)", DampedNewton(initial_damping=0.1), 30),
        ("SPSA (Shot-friendly stochastic)", SPSA(a=0.2, c=0.1, seed=42), 120),
    ]

    logger.info(f"{'Optimizer':<32} | {'Fidelity':<10} | {'Residual':<12} | {'Time (s)':<8}")
    logger.info("-" * 70)

    for name, opt, iters in optimizers:
        solver = VQLSSolver(ansatz=ansatz, optimizer=opt, max_iter=iters, seed=42)
        res = solver.solve(system)
        logger.info(
            f"{name:<32} | {res.fidelity*100:>8.2f}% | {res.residual:>12.4e} | {res.wall_time:>8.3f}"
        )

if __name__ == "__main__":
    main()
