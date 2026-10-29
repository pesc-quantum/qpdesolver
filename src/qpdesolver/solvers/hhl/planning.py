"""
HHL circuit planning: evolution time, eigenvalue grid mapping, and rotation angles.
Pure NumPy logic independent of quantum backends.
"""
from dataclasses import dataclass
from typing import Tuple, Union
import numpy as np

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.solvers.hhl.config import HHLConfig

@dataclass(frozen=True)
class HHLPlan:
    """
    Precomputed mathematical plan for an HHL circuit.
    """
    n_clock: int
    t: float
    lambda_grid: np.ndarray
    angles: np.ndarray
    C: float

def plan_hhl(system: LinearSystem, cfg: HHLConfig) -> HHLPlan:
    """
    Calculates evolution time t, discretized eigenvalue grid, and ancilla rotation angles.
    """
    if cfg.n_clock < 2:
        raise ValueError(f"n_clock must be at least 2, got {cfg.n_clock}.")

    if cfg.spectral_bounds is not None:
        lam_min, lam_max = cfg.spectral_bounds
    else:
        lam_min, lam_max = system.eigenvalue_bounds()

    if lam_max < 1e-12:
        raise ValueError("Maximum eigenvalue is too close to zero; system cannot be inverted.")

    two_n = 2**cfg.n_clock
    # Signed two's-complement range: [-2^{n-1}, 2^{n-1}-1] -> phases in [-0.5, 0.5)
    # To keep |lambda| * t / (2*pi) within range:
    max_phase_fraction = 0.5 if cfg.signed else 1.0
    t = 2.0 * np.pi * max_phase_fraction * cfg.phase_margin / lam_max

    k_indices = np.arange(two_n)
    if cfg.signed:
        signed_k = np.where(k_indices < two_n // 2, k_indices, k_indices - two_n)
    else:
        signed_k = k_indices

    # Discretized eigenvalue grid: lambda = 2*pi * (signed_k / 2^n) / t
    lam_grid = (2.0 * np.pi * signed_k) / (t * two_n)

    nonzero_mask = signed_k != 0
    min_grid_abs = float(np.min(np.abs(lam_grid[nonzero_mask])))

    if isinstance(cfg.C, (int, float)):
        C = float(cfg.C)
    elif cfg.C == "lambda_min":
        C = float(np.clip(lam_min, 0.0, min_grid_abs))
    elif cfg.C == "grid":
        C = min_grid_abs
    else:
        raise ValueError(f"Unknown HHL C selection '{cfg.C}'. Must be 'grid', 'lambda_min', or float.")

    angles = np.zeros(two_n, dtype=float)
    ratio = np.clip(C / lam_grid[nonzero_mask], -1.0, 1.0)
    angles[nonzero_mask] = 2.0 * np.arcsin(ratio)

    return HHLPlan(
        n_clock=cfg.n_clock,
        t=t,
        lambda_grid=lam_grid,
        angles=angles,
        C=C,
    )
