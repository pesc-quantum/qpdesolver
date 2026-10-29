"""
Variational Quantum Linear Solver (VQLS) execution engine.
"""
import time
from typing import Optional, Dict, Any
import numpy as np

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.solvers.base import LinearSolver, SolveResult
from qpdesolver.solvers.norm_recovery import rescale_least_squares
from qpdesolver.ansatze.base import Ansatz
from qpdesolver.ansatze.library import RealAmplitudes
from qpdesolver.solvers.vqls.costs import VQLSCost, GlobalCost
from qpdesolver.solvers.vqls.objective import VQLSObjective
from qpdesolver.optimizers.base import Optimizer
from qpdesolver.optimizers.scipy_wrapper import ScipyOptimizer
from qpdesolver.backends.pennylane_backend import PennyLaneBackend
from qpdesolver.core.metrics import relative_residual, fidelity
from qpdesolver.core.registry import REGISTRY_SOLVERS

@REGISTRY_SOLVERS.register("vqls")
class VQLSSolver(LinearSolver):
    """
    Variational Quantum Linear Solver (VQLS) solver.
    """
    name = "vqls"

    def __init__(
        self,
        ansatz: Optional[Ansatz] = None,
        cost: Optional[VQLSCost] = None,
        optimizer: Optional[Optimizer] = None,
        backend: Optional[PennyLaneBackend] = None,
        max_iter: int = 100,
        init_strategy: str = "uniform",
        seed: int = 42,
    ):
        self.ansatz = ansatz or RealAmplitudes(layers=2)
        self.cost = cost or GlobalCost()
        self.optimizer = optimizer or ScipyOptimizer("BFGS")
        self.backend = backend or PennyLaneBackend()
        self.max_iter = max_iter
        self.init_strategy = init_strategy
        self.seed = seed

    def solve(
        self,
        system: LinearSystem,
        *,
        warm_start: Optional[np.ndarray] = None,
    ) -> SolveResult:
        t0 = time.perf_counter()

        objective = VQLSObjective(
            system=system,
            ansatz=self.ansatz,
            cost=self.cost,
            backend=self.backend,
        )

        n_p = self.ansatz.num_params(system.n_qubits)
        if warm_start is not None and len(warm_start) == n_p:
            x0 = np.array(warm_start, dtype=float, copy=True)
        else:
            rng = np.random.default_rng(self.seed)
            x0 = self.ansatz.init_params(system.n_qubits, rng=rng, strategy=self.init_strategy)

        # Execute optimization
        opt_res = self.optimizer.minimize(objective, x0, max_iter=self.max_iter)

        final_state = objective.get_state(opt_res.x)
        x_phys, alpha = rescale_least_squares(system.A, system.b, final_state)
        res = relative_residual(system.A, x_phys, system.b)

        # Classical solution for diagnostic fidelity
        x_exact = system.classical_solution()
        fid = fidelity(final_state, x_exact)

        wall_time = time.perf_counter() - t0

        meta = {
            "optimal_params": list(opt_res.x),
            "final_cost": float(opt_res.fun),
            "optimizer": self.optimizer.name,
            "converged": opt_res.converged,
            "message": opt_res.message,
            "eval_counts": opt_res.n_evals,
            "least_squares_alpha": alpha,
            "ansatz": self.ansatz.__class__.__name__,
        }

        # Estimate total circuit evaluations from counted calls
        total_circuits = sum(opt_res.n_evals.values())

        return SolveResult(
            x=x_phys,
            state=final_state,
            residual=res,
            fidelity=fid,
            success_probability=1.0,
            history=opt_res.history,
            n_circuit_evals=total_circuits,
            resources={"total_circuit_evals": total_circuits, "iterations": opt_res.n_iter},
            wall_time=wall_time,
            metadata=meta,
        )
