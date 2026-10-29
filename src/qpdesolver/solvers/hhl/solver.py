"""
Harrow-Hassidim-Lloyd (HHL) solver implementation.
"""
import time
from typing import Optional, Dict, Any
import numpy as np

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.solvers.base import LinearSolver, SolveResult
from qpdesolver.solvers.norm_recovery import rescale_least_squares
from qpdesolver.solvers.hhl.config import HHLConfig
from qpdesolver.solvers.hhl.planning import plan_hhl
from qpdesolver.backends.base import Backend
from qpdesolver.backends.pennylane_backend import PennyLaneBackend
from qpdesolver.core.metrics import relative_residual, fidelity
from qpdesolver.core.registry import REGISTRY_SOLVERS

@REGISTRY_SOLVERS.register("hhl")
class HHLSolver(LinearSolver):
    """
    HHL quantum linear system solver.
    """
    name = "hhl"

    def __init__(
        self,
        config: Optional[HHLConfig] = None,
        backend: Optional[Backend] = None,
    ):
        self.config = config or HHLConfig()
        self.backend = backend or PennyLaneBackend()

    def solve(
        self,
        system: LinearSystem,
        *,
        warm_start: Optional[np.ndarray] = None,
    ) -> SolveResult:
        t0 = time.perf_counter()

        dilated = False
        if not system.is_hermitian:
            if not self.config.auto_dilate:
                raise ValueError(
                    f"System '{system.name}' is non-Hermitian and auto_dilate=False. "
                    "HHL requires Hermitian operator or auto-dilation."
                )
            sys_to_solve = system.hermitian_dilation()
            dilated = True
        else:
            sys_to_solve = system

        plan = plan_hhl(sys_to_solve, self.config)
        kernel = self.backend.hhl_kernel()
        raw = kernel.run(sys_to_solve, plan, self.config)

        if dilated:
            orig_n = system.size
            post_sub = raw.postselected[orig_n:]
            p_sub = float(np.vdot(post_sub, post_sub).real)
        else:
            post_sub = raw.postselected
            p_sub = raw.success_probability

        norm_post = np.linalg.norm(post_sub)
        if norm_post > 1e-15:
            state = post_sub / norm_post
        else:
            state = post_sub

        # Norm recovery via classical least squares
        x_phys, alpha = rescale_least_squares(system.A, system.b, state)
        res = relative_residual(system.A, x_phys, system.b)

        # Classical solution for diagnostic fidelity
        x_exact = system.classical_solution()
        fid = fidelity(state, x_exact)

        wall_time = time.perf_counter() - t0

        # Native HHL norm estimate
        native_norm_est = float(np.sqrt(max(p_sub, 0.0)) * system.b_norm / plan.C)

        meta = {
            "n_clock": plan.n_clock,
            "evolution": self.config.evolution,
            "signed": self.config.signed,
            "t": plan.t,
            "C": plan.C,
            "dilated": dilated,
            "least_squares_alpha": alpha,
            "native_norm_estimate": native_norm_est,
            "exact_norm": float(np.linalg.norm(x_exact)),
        }

        return SolveResult(
            x=x_phys,
            state=state,
            residual=res,
            fidelity=fid,
            success_probability=p_sub,
            history=[res],
            n_circuit_evals=1,
            resources=raw.resources,
            wall_time=wall_time,
            metadata=meta,
        )
