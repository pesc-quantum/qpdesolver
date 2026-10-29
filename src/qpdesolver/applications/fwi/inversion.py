"""
Full Waveform Inversion iterative optimization engine.
Couples FWIProblem/FWIProblem2D objective and adjoint gradient with qpdesolver optimizers.
"""
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict, Any, Union
import time
import numpy as np
from loguru import logger

from qpdesolver.applications.fwi.model import VelocityModel, FWIDataset, VelocityModel2D, FWIDataset2D
from qpdesolver.applications.fwi.problem import FWIProblem, FWIProblem2D
from qpdesolver.optimizers.base import Optimizer, Objective
from qpdesolver.optimizers.scipy_wrapper import ScipyOptimizer
from qpdesolver.optimizers.first_order import Adam, GradientDescent

@dataclass
class InversionResult:
    """Outcome of Full Waveform Inversion."""
    final_velocity: Union[VelocityModel, VelocityModel2D]
    final_misfit: float
    misfit_history: List[float] = field(default_factory=list)
    model_history: List[np.ndarray] = field(default_factory=list)
    wall_time: float = 0.0
    n_evals: int = 0
    iterations: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

class FWIObjective(Objective):
    """
    Wraps FWIProblem or FWIProblem2D misfit and adjoint gradient into a standard qpdesolver Objective.
    """
    def __init__(
        self,
        problem: Union[FWIProblem, FWIProblem2D],
        dataset: Union[FWIDataset, FWIDataset2D],
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
            grad = self.problem.gradient_v(x_proj, self.dataset.observed_data)
        else:
            grad = self.problem.gradient_m(x_proj, self.dataset.observed_data)
        return np.asarray(grad, dtype=float).reshape(-1)

    def grad(self, x: np.ndarray) -> np.ndarray:
        return self.gradient(x)

class FWIInversion:
    """
    Orchestrates acoustic Full Waveform Inversion in 1D and 2D.
    """
    def __init__(
        self,
        problem: Union[FWIProblem, FWIProblem2D],
        dataset: Union[FWIDataset, FWIDataset2D],
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

    def run(self, initial_model: Optional[Union[VelocityModel, VelocityModel2D]] = None) -> InversionResult:
        """
        Executes FWI starting from initial velocity model.
        """
        t0 = time.time()
        is_2d = isinstance(self.problem, FWIProblem2D) or isinstance(initial_model, VelocityModel2D)

        if is_2d:
            if initial_model is None:
                if self.dataset.initial_velocity is not None:
                    initial_model = VelocityModel2D(
                        velocity=self.dataset.initial_velocity,
                        hx=getattr(self.dataset, "hx", 1.0),
                        hz=getattr(self.dataset, "hz", None),
                        bc=self.dataset.bc,
                    )
                else:
                    raise ValueError("Must provide initial_model or dataset with initial_velocity.")

            if initial_model.shape != (self.problem.nx, self.problem.nz):
                initial_model = initial_model.resample((self.problem.nx, self.problem.nz))

            x0 = (
                initial_model.flat_velocity.copy()
                if self.parameterization == "v"
                else initial_model.slowness_squared.reshape(-1).copy()
            )
        else:
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
        init_history_entry = (
            x0.copy().reshape(self.problem.nx, self.problem.nz) if is_2d else x0.copy()
        )
        model_history: List[np.ndarray] = [init_history_entry]

        def step_callback(x_k: np.ndarray):
            f_k = obj.value(x_k)
            misfit_history.append(float(f_k))
            step_entry = (
                x_k.copy().reshape(self.problem.nx, self.problem.nz) if is_2d else x_k.copy()
            )
            model_history.append(step_entry)
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

        if is_2d:
            final_model = VelocityModel2D(
                velocity=v_final.reshape(self.problem.nx, self.problem.nz),
                hx=self.problem.hx,
                hz=self.problem.hz,
                bc=self.problem.bc,
            )
        else:
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
