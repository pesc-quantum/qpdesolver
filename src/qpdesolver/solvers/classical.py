"""
Classical linear solver baseline using np.linalg.solve.
"""
import time
from typing import Optional
import numpy as np

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.solvers.base import LinearSolver, SolveResult
from qpdesolver.core.metrics import relative_residual, fidelity
from qpdesolver.core.registry import REGISTRY_SOLVERS

@REGISTRY_SOLVERS.register("classical")
class ClassicalSolver(LinearSolver):
    """
    Direct classical solver using numpy/scipy.
    """
    name = "classical"

    def solve(
        self,
        system: LinearSystem,
        *,
        warm_start: Optional[np.ndarray] = None,
    ) -> SolveResult:
        t0 = time.perf_counter()
        x_exact = system.classical_solution()
        wall_time = time.perf_counter() - t0

        norm_x = np.linalg.norm(x_exact)
        state = x_exact / norm_x if norm_x > 1e-15 else x_exact
        res = relative_residual(system.A, x_exact, system.b)
        fid = 1.0

        return SolveResult(
            x=x_exact,
            state=state,
            residual=res,
            fidelity=fid,
            success_probability=1.0,
            history=[res],
            n_circuit_evals=0,
            resources={"method": "np.linalg.solve"},
            wall_time=wall_time,
            metadata={"system_name": system.name, "condition_number": system.condition_number()},
        )
