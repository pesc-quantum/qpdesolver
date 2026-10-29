"""
Full Waveform Inversion iterative optimization engine.
Couples FWIProblem objective and adjoint gradient with qpdesolver optimizers.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any
import time
import numpy as np
from loguru import logger

from qpdesolver.applications.fwi.model import VelocityModel, FWIDataset
from qpdesolver.applications.fwi.problem import FWIProblem
from qpdesolver.optimizers.base import Optimizer, Objective
from qpdesolver.optimizers.scipy_wrapper import ScipyOptimizer
from qpdesolver.optimizers.first_order import Adam, GradientDescent

@dataclass
class InversionResult:
    """Outcome of Full Waveform Inversion."""
    final_velocity: VelocityModel
    final_misfit: float
    misfit_history: List[float] = field(default_factory=list)
    model_history: List[np.ndarray] = field(default_factory=list)
    wall_time: float = 0.0
    n_evals: int = 0
    iterations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class FWIObjective(Objective):
    """
    Wraps FWIProblem misfit and adjoint gradient into a standard qpdesolver Objective.
    """
    def __init__(
        self,
        problem: FWIProblem,
        dataset: FWIDataset,
        parameterization: str = "v",
        bounds: Optional[Tuple[float, float]] = None,
    ):
        self.problem = problem
        self.dataset = dataset
        self.parameterization = parameterization.lower()
        self.bounds = bounds
        self.eval_count = 0

    @property
    def n_params(self) -> int:
        return self.problem.n_points

    def _project(self, x: np.ndarray) -> np.ndarray:
        if self.bounds is not None:
            return np.clip(x, self.bounds[0], self.bounds[1])
        return x

    def value(self, x: np.ndarray) -> float:
        x_proj = self._project(x)
        self.eval_count += 1
        if self.parameterization == "v":
            m = 1.0 / (x_proj**2)
        else:
            m = x_proj
        return self.problem.misfit(m, self.dataset.observed_data)

    def gradient(self, x: np.ndarray) -> np.ndarray:
        x_proj = self._project(x)
        if self.parameterization == "v":
            return self.problem.gradient_v(x_proj, self.dataset.observed_data)
        else:
            return self.problem.gradient_m(x_proj, self.dataset.observed_data)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.gradient(x)

class FWIInversion:
    """
    Orchestrates acoustic Full Waveform Inversion.
    """
    def __init__(
        self,
        problem: FWIProblem,
        dataset: FWIDataset,
        optimizer: Optional[Optimizer] = None,
        parameterization: str = "v",
        bounds: Optional[Tuple[float, float]] = (500.0, 5000.0),
        max_iter: int = 20,
    ):
        self.problem = problem
        self.dataset = dataset
        self.parameterization = parameterization.lower()
        self.bounds = bounds
        self.max_iter = max_iter

        if optimizer is None:
            self.optimizer = ScipyOptimizer(method="L-BFGS-B")
        else:
            self.optimizer = optimizer

    def run(self, initial_model: Optional[VelocityModel] = None) -> InversionResult:
        """
        Executes FWI starting from initial velocity model.
        """
        t0 = time.time()
        if initial_model is None:
            if self.dataset.initial_velocity is not None:
                initial_model = VelocityModel(
                    velocity=self.dataset.initial_velocity,
                    h=self.dataset.h,
                    bc=self.dataset.bc,
                )
            else:
                raise ValueError("Must provide initial_model or dataset with initial_velocity.")

        if initial_model.n_points != self.problem.n_points:
            initial_model = initial_model.resample(self.problem.n_points)

        x0 = (
            initial_model.velocity.copy()
            if self.parameterization == "v"
            else initial_model.slowness_squared.copy()
        )

        logger.info(
            f"Starting FWI Inversion: {self.problem.n_points} grid points, "
            f"{len(self.dataset.acquisition.sources)} sources, "
            f"{len(self.dataset.acquisition.omegas)} frequencies, "
            f"optimizer: {self.optimizer.name}"
        )

        obj = FWIObjective(
            problem=self.problem,
            dataset=self.dataset,
            parameterization=self.parameterization,
            bounds=self.bounds,
        )

        f0 = obj.value(x0)
        misfit_history: List[float] = [f0]
        model_history: List[np.ndarray] = [x0.copy()]

        def step_callback(x_k: np.ndarray):
            f_k = obj.value(x_k)
            misfit_history.append(float(f_k))
            model_history.append(x_k.copy())
            logger.debug(f"FWI step {len(misfit_history)-1}: misfit={f_k:.6e}")

        opt_res = self.optimizer.minimize(
            objective=obj,
            x0=x0,
            max_iter=self.max_iter,
            callback=step_callback,
        )

        x_final = opt_res.x
        if self.bounds is not None:
            x_final = np.clip(x_final, self.bounds[0], self.bounds[1])

        if self.parameterization == "v":
            v_final = x_final
        else:
            v_final = 1.0 / np.sqrt(np.maximum(x_final, 1e-12))

        final_model = VelocityModel(
            velocity=v_final,
            h=self.problem.h,
            bc=self.problem.bc,
        )
        final_misfit = obj.value(x_final)
        if len(misfit_history) == 1 or misfit_history[-1] != final_misfit:
            misfit_history.append(final_misfit)

        elapsed = time.time() - t0
        logger.info(
            f"FWI complete in {elapsed:.2f}s: initial misfit={misfit_history[0]:.6e}, "
            f"final misfit={final_misfit:.6e}"
        )

        return InversionResult(
            final_velocity=final_model,
            final_misfit=final_misfit,
            misfit_history=misfit_history,
            model_history=model_history,
            wall_time=elapsed,
            n_evals=obj.eval_count,
            iterations=opt_res.n_iter,
            metadata={"optimizer": self.optimizer.name, "converged": opt_res.converged},
        )
