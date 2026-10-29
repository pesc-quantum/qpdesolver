"""
Full Waveform Inversion (FWI) forward and adjoint problem modeling on 1D and 2D Helmholtz equations.
Uses exact adjoint state equations for calculating misfit gradients w.r.t slowness squared m and velocity v.
"""
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List, Union
import numpy as np
from loguru import logger

from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.problems.discretizations.helmholtz2d import Helmholtz2D
from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.solvers.base import LinearSolver, SolveResult
from qpdesolver.solvers.classical import ClassicalSolver
from qpdesolver.applications.fwi.model import (
    VelocityModel,
    Acquisition,
    FWIDataset,
    VelocityModel2D,
    Acquisition2D,
    FWIDataset2D,
)


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
        warm_start_solves: bool = False,
    ):
        if n_points <= 0 or (n_points & (n_points - 1)) != 0:
            raise ValueError(f"n_points must be a positive power of 2, got {n_points}.")
        self.acquisition = acquisition
        self.n_points = n_points
        self.h = float(h)
        self.bc = bc
        self.solver = solver if solver is not None else ClassicalSolver()
        self.warm_start_solves = warm_start_solves
        self._warm_start_cache: Dict[Tuple[str, Any, float], np.ndarray] = {}
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

    def _solve_system(self, system: LinearSystem, role: str, src_key: Any, omega: float) -> SolveResult:
        cache_key = (role, src_key, float(omega))
        warm_start_param = None
        if self.warm_start_solves:
            warm_start_param = self._warm_start_cache.get(cache_key)

        try:
            if warm_start_param is not None:
                res = self.solver.solve(system, warm_start=warm_start_param)
            else:
                res = self.solver.solve(system)
        except TypeError:
            res = self.solver.solve(system)

        if self.warm_start_solves and res.metadata and "optimal_params" in res.metadata:
            opt = res.metadata["optimal_params"]
            if opt is not None:
                self._warm_start_cache[cache_key] = np.array(opt, dtype=float)

        return res

    def solve_forward(self, m: np.ndarray, omega: float, src_idx: int) -> np.ndarray:
        """
        Solves forward Helmholtz equation for a given source and frequency.
        Returns the wavefield solution u.
        """
        b = self.discretization.point_source(src_idx)
        system = self._build_system(m, omega, b)
        res = self._solve_system(system, role="forward", src_key=src_idx, omega=omega)
        return res.x

    def solve_adjoint(
        self,
        m: np.ndarray,
        omega: float,
        residual_at_receivers: np.ndarray,
        src_idx: Any = 0,
    ) -> np.ndarray:
        """
        Solves adjoint Helmholtz equation for backprojected data residual.
        Returns the adjoint wavefield lambda.
        """
        # Adjoint source: R^T delta_d
        b_adj = np.zeros(self.n_points, dtype=complex)
        for r_idx, r_val in zip(self.acquisition.receivers, residual_at_receivers):
            b_adj[r_idx] = r_val

        # Since A is real symmetric, A^\dagger = A
        system_r = self._build_system(m, omega, np.real(b_adj))
        res_r = self._solve_system(system_r, role="adjoint_real", src_key=src_idx, omega=omega)

        if np.iscomplexobj(b_adj) and np.any(np.imag(b_adj) != 0):
            system_i = self._build_system(m, omega, np.imag(b_adj))
            res_i = self._solve_system(system_i, role="adjoint_imag", src_key=src_idx, omega=omega)
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

    def gradient_m(self, m: np.ndarray, observed_data: np.ndarray | FWIDataset) -> np.ndarray:
        """
        Computes adjoint-state gradient of misfit w.r.t slowness squared m:
            dJ / dm_i = sum_{s, omega} omega^2 Re(lambda_{s, omega, i}^* u_{s, omega, i})
        """
        obs_data = observed_data.observed_data if hasattr(observed_data, "observed_data") else observed_data
        grad = np.zeros(self.n_points, dtype=float)

        for s_i, src in enumerate(self.acquisition.sources):
            for w_i, omega in enumerate(self.acquisition.omegas):
                u = self.solve_forward(m, omega, src)
                pred_r = u[self.acquisition.receivers]
                obs_r = obs_data[s_i, w_i, :]
                delta_d = pred_r - obs_r

                lambda_adj = self.solve_adjoint(m, omega, delta_d, src_idx=src)
                # Gradient update: omega^2 * Re(lambda^* u)
                grad += (omega**2) * np.real(np.conj(lambda_adj) * u)

        return grad

    def gradient_v(self, v: np.ndarray | VelocityModel, observed_data: np.ndarray) -> np.ndarray:
        """
        Computes gradient of misfit w.r.t velocity v using chain rule:
            m_i = 1 / v_i^2 => dm_i / dv_i = -2 / v_i^3
            dJ / dv_i = (dJ / dm_i) * (-2 / v_i^3)
        """
        v_arr = np.asarray(v.velocity if hasattr(v, "velocity") else v)
        m = 1.0 / (v_arr**2)
        grad_m = self.gradient_m(m, observed_data)
        return grad_m * (-2.0 / (v_arr**3))

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


class FWIProblem2D:
    """
    2D Frequency-Domain Acoustic Full Waveform Inversion Problem.
    Forward equation:
        (L_2D - omega^2 diag(m)) u_{s, omega} = b_s
    Data projection:
        d_{s, omega} = R u_{s, omega}
    Misfit:
        J(m) = 0.5 * sum_{s, omega} || R u_{s, omega} - d_{s, omega}^{obs} ||_2^2
    Adjoint state equation:
        (L_2D - omega^2 diag(m)) lambda_{s, omega} = R^T (R u_{s, omega} - d_{s, omega}^{obs})
    Gradient:
        dJ / dm_{ix, iz} = sum_{s, omega} omega^2 Re(lambda_{s, omega, ix, iz}^* u_{s, omega, ix, iz})
    """
    def __init__(
        self,
        acquisition: Acquisition2D,
        nx: int,
        nz: int,
        hx: float = 1.0,
        hz: Optional[float] = None,
        bc: str = "dirichlet",
        solver: Optional[LinearSolver] = None,
        warm_start_solves: bool = False,
    ):
        if nx <= 0 or (nx & (nx - 1)) != 0:
            raise ValueError(f"nx must be a positive power of 2, got {nx}.")
        if nz <= 0 or (nz & (nz - 1)) != 0:
            raise ValueError(f"nz must be a positive power of 2, got {nz}.")
        self.acquisition = acquisition
        self.nx = int(nx)
        self.nz = int(nz)
        self.hx = float(hx)
        self.hz = float(hx) if hz is None else float(hz)
        self.bc = bc
        self.n_points = self.nx * self.nz
        self.solver = solver if solver is not None else ClassicalSolver()
        self.warm_start_solves = warm_start_solves
        self._warm_start_cache: Dict[Tuple[str, Any, float], np.ndarray] = {}
        self.discretization = Helmholtz2D(nx=self.nx, nz=self.nz, hx=self.hx, hz=self.hz, bc=self.bc)

    def _to_flat_index(self, pt: Union[int, Tuple[int, int], List[int]]) -> int:
        if isinstance(pt, (tuple, list)):
            return int(pt[0]) * self.nz + int(pt[1])
        return int(pt)

    def _to_tuple_index(self, pt: Union[int, Tuple[int, int], List[int]]) -> Tuple[int, int]:
        if isinstance(pt, (tuple, list)):
            return (int(pt[0]), int(pt[1]))
        idx = int(pt)
        return (idx // self.nz, idx % self.nz)

    def _build_system(self, m: np.ndarray, omega: float, rhs: np.ndarray) -> LinearSystem:
        """Constructs a LinearSystem for given slowness m, frequency omega, and source rhs."""
        A = self.discretization.operator(m=m, omega=omega)
        return LinearSystem(
            A=A,
            b=rhs,
            name=f"FWI2D_Helmholtz_omega={omega:.2f}",
            metadata={"is_hermitian": True},
        )

    def _solve_system(self, system: LinearSystem, role: str, src_key: Any, omega: float) -> SolveResult:
        cache_key = (role, src_key, float(omega))
        warm_start_param = None
        if self.warm_start_solves:
            warm_start_param = self._warm_start_cache.get(cache_key)

        try:
            if warm_start_param is not None:
                res = self.solver.solve(system, warm_start=warm_start_param)
            else:
                res = self.solver.solve(system)
        except TypeError:
            res = self.solver.solve(system)

        if self.warm_start_solves and res.metadata and "optimal_params" in res.metadata:
            opt = res.metadata["optimal_params"]
            if opt is not None:
                self._warm_start_cache[cache_key] = np.array(opt, dtype=float)

        return res

    def solve_forward(self, m: np.ndarray, omega: float, src_idx_or_tuple: Union[int, Tuple[int, int]]) -> np.ndarray:
        """
        Solves forward 2D Helmholtz equation for a given source and frequency.
        Returns the wavefield solution u with shape (nx, nz).
        """
        flat_idx = self._to_flat_index(src_idx_or_tuple)
        b = self.discretization.point_source(flat_idx)
        system = self._build_system(m, omega, b)
        src_key = self._to_tuple_index(src_idx_or_tuple)
        res = self._solve_system(system, role="forward", src_key=src_key, omega=omega)
        return res.x.reshape(self.nx, self.nz)

    def solve_adjoint(
        self,
        m: np.ndarray,
        omega: float,
        residual_at_receivers: np.ndarray,
        src_idx: Any = (0, 0),
    ) -> np.ndarray:
        """
        Solves adjoint 2D Helmholtz equation for backprojected data residual.
        Returns the adjoint wavefield lambda with shape (nx, nz).
        """
        b_adj = np.zeros(self.n_points, dtype=complex)
        for r, val in zip(self.acquisition.receivers, residual_at_receivers):
            r_flat = self._to_flat_index(r)
            b_adj[r_flat] += val

        src_key = self._to_tuple_index(src_idx) if isinstance(src_idx, (tuple, list, int, np.integer)) else src_idx

        system_r = self._build_system(m, omega, np.real(b_adj))
        res_r = self._solve_system(system_r, role="adjoint_real", src_key=src_key, omega=omega)

        if np.iscomplexobj(b_adj) and np.any(np.imag(b_adj) != 0):
            system_i = self._build_system(m, omega, np.imag(b_adj))
            res_i = self._solve_system(system_i, role="adjoint_imag", src_key=src_key, omega=omega)
            return (res_r.x + 1j * res_i.x).reshape(self.nx, self.nz)
        return res_r.x.astype(complex).reshape(self.nx, self.nz)

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
                for r_i, rec in enumerate(self.acquisition.receivers):
                    if isinstance(rec, (tuple, list)):
                        pred_data[s_i, w_i, r_i] = u[rec[0], rec[1]]
                    else:
                        pred_data[s_i, w_i, r_i] = u.ravel()[int(rec)]

        return pred_data

    def misfit(self, m: np.ndarray, observed_data: np.ndarray) -> float:
        """
        Computes data misfit J(m) = 0.5 * || d_pred(m) - d_obs ||_2^2.
        """
        pred_data = self.forward_all(m)
        diff = pred_data - observed_data
        return float(0.5 * np.sum(np.abs(diff)**2))

    def gradient_m(self, m: np.ndarray, observed_data: np.ndarray | FWIDataset2D) -> np.ndarray:
        """
        Computes adjoint-state gradient of misfit w.r.t slowness squared m:
            dJ / dm_{ix, iz} = sum_{s, omega} omega^2 Re(lambda_{s, omega, ix, iz}^* u_{s, omega, ix, iz})
        Returns array of shape (nx, nz).
        """
        obs_data = observed_data.observed_data if hasattr(observed_data, "observed_data") else observed_data
        grad = np.zeros((self.nx, self.nz), dtype=float)

        for s_i, src in enumerate(self.acquisition.sources):
            for w_i, omega in enumerate(self.acquisition.omegas):
                u = self.solve_forward(m, omega, src)
                pred_r = np.array([
                    u[rec[0], rec[1]] if isinstance(rec, (tuple, list)) else u.ravel()[int(rec)]
                    for rec in self.acquisition.receivers
                ], dtype=complex)
                obs_r = obs_data[s_i, w_i, :]
                delta_d = pred_r - obs_r

                lambda_adj = self.solve_adjoint(m, omega, delta_d, src_idx=src)
                grad += (omega**2) * np.real(np.conj(lambda_adj) * u)

        return grad

    def gradient_v(self, v: np.ndarray | VelocityModel2D, observed_data: np.ndarray) -> np.ndarray:
        """
        Computes gradient of misfit w.r.t velocity v using chain rule:
            dJ / dv = (dJ / dm) * (-2 / v^3)
        Returns shape matching input v: (nx, nz) or (n_points,).
        """
        v_arr = np.asarray(v.velocity if hasattr(v, "velocity") else v)
        m = 1.0 / (v_arr**2)
        grad_m = self.gradient_m(m, observed_data)
        if v_arr.shape == (self.nx, self.nz):
            return grad_m * (-2.0 / (v_arr**3))
        elif v_arr.shape == (self.n_points,):
            return (grad_m.reshape(-1)) * (-2.0 / (v_arr**3))
        else:
            raise ValueError(f"v shape {v_arr.shape} incompatible with ({self.nx}, {self.nz}).")

    def simulate_dataset(
        self,
        true_model: VelocityModel2D,
        initial_model: Optional[VelocityModel2D] = None,
        noise_level: float = 0.0,
        seed: int = 42,
    ) -> FWIDataset2D:
        """
        Simulates noise-free or noisy observed data from true 2D velocity model.
        """
        if true_model.shape != (self.nx, self.nz):
            true_model = true_model.resample((self.nx, self.nz))

        m_true = true_model.slowness_squared
        d_obs = self.forward_all(m_true)

        if noise_level > 0.0:
            rng = np.random.default_rng(seed)
            noise_r = rng.normal(0.0, noise_level, size=d_obs.shape)
            noise_i = rng.normal(0.0, noise_level, size=d_obs.shape)
            d_obs = d_obs + (noise_r + 1j * noise_i)

        init_v = None
        if initial_model is not None:
            if initial_model.shape != (self.nx, self.nz):
                initial_model = initial_model.resample((self.nx, self.nz))
            init_v = initial_model.velocity.copy()

        return FWIDataset2D(
            acquisition=self.acquisition,
            observed_data=d_obs,
            hx=self.hx,
            hz=self.hz,
            bc=self.bc,
            true_velocity=true_model.velocity.copy(),
            initial_velocity=init_v,
            metadata={"noise_level": noise_level},
        )
