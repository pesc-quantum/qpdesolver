"""
Full Waveform Inversion (FWI) forward and adjoint problem modeling on 1D Helmholtz equations.
Uses exact adjoint state equations for calculating misfit gradients w.r.t slowness squared m and velocity v.
"""
from typing import Optional, Tuple, Dict, Any, List
import numpy as np
from loguru import logger

from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.solvers.base import LinearSolver
from qpdesolver.solvers.classical import ClassicalSolver
from qpdesolver.applications.fwi.model import VelocityModel, Acquisition, FWIDataset

class FWIProblem:
    """
    1D Frequency-Domain Acoustic Full Waveform Inversion Problem.
    Forward equation:
        (L - omega^2 diag(m)) u_{s, omega} = b_s
    Data projection:
        d_{s, omega} = R u_{s, omega}
    Misfit:
        J(m) = 0.5 * sum_{s, omega} || R u_{s, omega} - d_{s, omega}^{obs} ||_2^2
    Adjoint state equation:
        (L - omega^2 diag(m)) lambda_{s, omega} = R^T (R u_{s, omega} - d_{s, omega}^{obs})
    Gradient:
        dJ / dm_i = sum_{s, omega} omega^2 Re(lambda_{s, omega, i}^* u_{s, omega, i})
    """
    def __init__(
        self,
        acquisition: Acquisition,
        n_points: int,
        h: float = 1.0,
        bc: str = "dirichlet",
        solver: Optional[LinearSolver] = None,
    ):
        if n_points <= 0 or (n_points & (n_points - 1)) != 0:
            raise ValueError(f"n_points must be a positive power of 2, got {n_points}.")
        self.acquisition = acquisition
        self.n_points = n_points
        self.h = float(h)
        self.bc = bc
        self.solver = solver if solver is not None else ClassicalSolver()
        self.discretization = Helmholtz1D(n_points=self.n_points, h=self.h, bc=self.bc)

    def _build_system(self, m: np.ndarray, omega: float, rhs: np.ndarray) -> LinearSystem:
        """Constructs a LinearSystem for given slowness m, frequency omega, and source rhs."""
        A = self.discretization.operator(m=m, omega=omega)
        return LinearSystem(
            A=A,
            b=rhs,
            name=f"FWI_Helmholtz_omega={omega:.2f}",
            metadata={"is_hermitian": True},
        )

    def solve_forward(self, m: np.ndarray, omega: float, src_idx: int) -> np.ndarray:
        """
        Solves forward Helmholtz equation for a given source and frequency.
        Returns the wavefield solution u.
        """
        b = self.discretization.point_source(src_idx)
        system = self._build_system(m, omega, b)
        res = self.solver.solve(system)
        return res.x

    def solve_adjoint(self, m: np.ndarray, omega: float, residual_at_receivers: np.ndarray) -> np.ndarray:
        """
        Solves adjoint Helmholtz equation for backprojected data residual.
        Returns the adjoint wavefield lambda.
        """
        # Adjoint source: R^T delta_d
        b_adj = np.zeros(self.n_points, dtype=complex)
        for r_idx, r_val in zip(self.acquisition.receivers, residual_at_receivers):
            b_adj[r_idx] = r_val

        # Since A is real symmetric, A^\dagger = A
        # For complex rhs, we can solve real and imag parts or complex directly
        system_r = self._build_system(m, omega, np.real(b_adj))
        res_r = self.solver.solve(system_r)

        if np.iscomplexobj(b_adj) and np.any(np.imag(b_adj) != 0):
            system_i = self._build_system(m, omega, np.imag(b_adj))
            res_i = self.solver.solve(system_i)
            return res_r.x + 1j * res_i.x
        return res_r.x.astype(complex)

    def forward_all(self, m: np.ndarray) -> np.ndarray:
        """
        Computes synthetic wavefield data for all sources, omegas, and receivers.
        Returns array of shape (n_sources, n_omegas, n_receivers).
        """
        n_s = len(self.acquisition.sources)
        n_w = len(self.acquisition.omegas)
        n_r = len(self.acquisition.receivers)
        pred_data = np.zeros((n_s, n_w, n_r), dtype=complex)

        for s_i, src in enumerate(self.acquisition.sources):
            for w_i, omega in enumerate(self.acquisition.omegas):
                u = self.solve_forward(m, omega, src)
                pred_data[s_i, w_i, :] = u[self.acquisition.receivers]

        return pred_data

    def misfit(self, m: np.ndarray, observed_data: np.ndarray) -> float:
        """
        Computes data misfit J(m) = 0.5 * || d_pred(m) - d_obs ||_2^2.
        """
        pred_data = self.forward_all(m)
        diff = pred_data - observed_data
        return float(0.5 * np.sum(np.abs(diff)**2))

    def gradient_m(self, m: np.ndarray, observed_data: np.ndarray) -> np.ndarray:
        """
        Computes adjoint-state gradient of misfit w.r.t slowness squared m:
            dJ / dm_i = sum_{s, omega} omega^2 Re(lambda_{s, omega, i}^* u_{s, omega, i})
        """
        grad = np.zeros(self.n_points, dtype=float)

        for s_i, src in enumerate(self.acquisition.sources):
            for w_i, omega in enumerate(self.acquisition.omegas):
                u = self.solve_forward(m, omega, src)
                pred_r = u[self.acquisition.receivers]
                obs_r = observed_data[s_i, w_i, :]
                delta_d = pred_r - obs_r

                lambda_adj = self.solve_adjoint(m, omega, delta_d)
                # Gradient update: omega^2 * Re(lambda^* u)
                grad += (omega**2) * np.real(np.conj(lambda_adj) * u)

        return grad

    def gradient_v(self, v: np.ndarray, observed_data: np.ndarray) -> np.ndarray:
        """
        Computes gradient of misfit w.r.t velocity v using chain rule:
            m_i = 1 / v_i^2 => dm_i / dv_i = -2 / v_i^3
            dJ / dv_i = (dJ / dm_i) * (-2 / v_i^3)
        """
        m = 1.0 / (v**2)
        grad_m = self.gradient_m(m, observed_data)
        return grad_m * (-2.0 / (v**3))

    def simulate_dataset(
        self,
        true_model: VelocityModel,
        initial_model: Optional[VelocityModel] = None,
        noise_level: float = 0.0,
        seed: int = 42,
    ) -> FWIDataset:
        """
        Simulates noise-free or noisy observed data from true velocity model.
        """
        if true_model.n_points != self.n_points:
            true_model = true_model.resample(self.n_points)

        m_true = true_model.slowness_squared
        d_obs = self.forward_all(m_true)

        if noise_level > 0.0:
            rng = np.random.default_rng(seed)
            noise_r = rng.normal(0.0, noise_level, size=d_obs.shape)
            noise_i = rng.normal(0.0, noise_level, size=d_obs.shape)
            d_obs = d_obs + (noise_r + 1j * noise_i)

        init_v = None
        if initial_model is not None:
            if initial_model.n_points != self.n_points:
                initial_model = initial_model.resample(self.n_points)
            init_v = initial_model.velocity.copy()

        return FWIDataset(
            acquisition=self.acquisition,
            observed_data=d_obs,
            h=self.h,
            bc=self.bc,
            true_velocity=true_model.velocity.copy(),
            initial_velocity=init_v,
            metadata={"noise_level": noise_level},
        )
