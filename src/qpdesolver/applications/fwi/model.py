"""
Data models for Full Waveform Inversion (FWI):
- VelocityModel: 1D subsurface acoustic velocity distribution.
- FWIDataset: multi-frequency, multi-shot acquisition and wavefield data.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import numpy as np

@dataclass
class VelocityModel:
    """
    1D Velocity model representation.
    """
    velocity: np.ndarray
    h: float = 1.0
    bc: str = "dirichlet"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        v = np.asarray(self.velocity, dtype=float)
        if v.ndim != 1:
            raise ValueError(f"Velocity profile must be 1D, got shape {v.shape}.")
        if np.any(v <= 0.0):
            raise ValueError("Velocity values must be strictly positive.")
        if not np.all(np.isfinite(v)):
            raise ValueError("Velocity values must be finite.")
        self.velocity = v

    @property
    def n_points(self) -> int:
        return len(self.velocity)

    @property
    def slowness_squared(self) -> np.ndarray:
        """m = 1 / v^2"""
        return 1.0 / (self.velocity**2)

    def resample(self, target_points: int) -> "VelocityModel":
        """
        Resamples velocity profile onto a new grid of size target_points (e.g. power of 2)
        using linear interpolation while strictly preserving domain endpoints.
        """
        if target_points <= 1:
            raise ValueError(f"target_points must be > 1, got {target_points}.")
        if target_points == self.n_points:
            return VelocityModel(velocity=self.velocity.copy(), h=self.h, bc=self.bc, metadata=dict(self.metadata))

        old_coords = np.linspace(0.0, 1.0, self.n_points)
        new_coords = np.linspace(0.0, 1.0, target_points)
        new_v = np.interp(new_coords, old_coords, self.velocity)
        new_h = self.h * (self.n_points - 1) / (target_points - 1)

        meta = dict(self.metadata)
        meta["resampled_from"] = self.n_points
        return VelocityModel(velocity=new_v, h=new_h, bc=self.bc, metadata=meta)

@dataclass
class Acquisition:
    """
    Seismic acquisition geometry.
    """
    sources: List[int]
    receivers: List[int]
    omegas: List[float]

@dataclass
class FWIDataset:
    """
    FWI observed dataset and survey configuration.
    observed_data shape: (n_sources, n_omegas, n_receivers) complex.
    """
    acquisition: Acquisition
    observed_data: np.ndarray
    h: float = 1.0
    bc: str = "dirichlet"
    true_velocity: Optional[np.ndarray] = None
    initial_velocity: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        d = np.asarray(self.observed_data, dtype=complex)
        expected_shape = (
            len(self.acquisition.sources),
            len(self.acquisition.omegas),
            len(self.acquisition.receivers),
        )
        if d.shape != expected_shape:
            raise ValueError(f"Observed data shape {d.shape} does not match acquisition {expected_shape}.")
        self.observed_data = d
