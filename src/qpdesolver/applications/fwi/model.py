"""
Data models for Full Waveform Inversion (FWI):
- VelocityModel: 1D subsurface acoustic velocity distribution.
- FWIDataset: multi-frequency, multi-shot acquisition and wavefield data.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
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


@dataclass
class VelocityModel2D:
    """
    2D Velocity model representation on a uniform Cartesian grid (nx, nz).
    """
    velocity: np.ndarray
    hx: float = 1.0
    hz: Optional[float] = None
    bc: str = "dirichlet"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        v = np.asarray(self.velocity, dtype=float)
        if v.ndim != 2:
            raise ValueError(f"Velocity profile must be 2D, got shape {v.shape}.")
        if np.any(v <= 0.0):
            raise ValueError("Velocity values must be strictly positive.")
        if not np.all(np.isfinite(v)):
            raise ValueError("Velocity values must be finite.")
        self.velocity = v
        self.hx = float(self.hx)
        self.hz = float(self.hx) if self.hz is None else float(self.hz)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.velocity.shape

    @property
    def nx(self) -> int:
        return self.velocity.shape[0]

    @property
    def nz(self) -> int:
        return self.velocity.shape[1]

    @property
    def n_points(self) -> int:
        return self.velocity.size

    @property
    def slowness_squared(self) -> np.ndarray:
        """m = 1 / v^2 (shape nx, nz)"""
        return 1.0 / (self.velocity**2)

    @property
    def flat_velocity(self) -> np.ndarray:
        return self.velocity.reshape(-1)

    def resample(self, target_shape: Tuple[int, int], smooth_sigma: float = 0.0) -> "VelocityModel2D":
        """
        Resamples 2D velocity profile onto a new grid of size target_shape = (target_nx, target_nz)
        using bilinear interpolation while strictly preserving domain corners,
        and optionally applies Gaussian smoothing.
        """
        target_nx, target_nz = target_shape
        if target_nx <= 1 or target_nz <= 1:
            raise ValueError(f"target_shape dimensions must be > 1, got {target_shape}.")

        if target_shape == self.shape and smooth_sigma <= 0.0:
            return VelocityModel2D(
                velocity=self.velocity.copy(),
                hx=self.hx,
                hz=self.hz,
                bc=self.bc,
                metadata=dict(self.metadata),
            )

        if target_shape == self.shape:
            new_v = self.velocity.copy()
            new_hx, new_hz = self.hx, self.hz
        else:
            from scipy.interpolate import RegularGridInterpolator
            old_x = np.linspace(0.0, 1.0, self.nx)
            old_z = np.linspace(0.0, 1.0, self.nz)
            interp = RegularGridInterpolator((old_x, old_z), self.velocity, bounds_error=False, fill_value=None)

            new_x = np.linspace(0.0, 1.0, target_nx)
            new_z = np.linspace(0.0, 1.0, target_nz)
            X, Z = np.meshgrid(new_x, new_z, indexing="ij")
            pts = np.stack([X.ravel(), Z.ravel()], axis=-1)
            new_v = interp(pts).reshape((target_nx, target_nz))

            new_hx = self.hx * (self.nx - 1) / (target_nx - 1)
            new_hz = self.hz * (self.nz - 1) / (target_nz - 1)

        if smooth_sigma > 0.0:
            import scipy.ndimage as ndi
            new_v = ndi.gaussian_filter(new_v, sigma=smooth_sigma, mode="nearest")

        meta = dict(self.metadata)
        meta["resampled_from"] = self.shape
        if smooth_sigma > 0.0:
            meta["smooth_sigma"] = smooth_sigma

        return VelocityModel2D(
            velocity=new_v,
            hx=new_hx,
            hz=new_hz,
            bc=self.bc,
            metadata=meta,
        )


@dataclass
class Acquisition2D:
    """
    2D Seismic acquisition geometry.
    sources: list of (ix, iz) coordinates
    receivers: list of (ix, iz) coordinates
    omegas: list of angular frequencies
    """
    sources: List[Tuple[int, int]]
    receivers: List[Tuple[int, int]]
    omegas: List[float]


@dataclass
class FWIDataset2D:
    """
    2D FWI observed dataset and survey configuration.
    observed_data shape: (n_sources, n_omegas, n_receivers) complex.
    """
    acquisition: Acquisition2D
    observed_data: np.ndarray
    hx: float = 1.0
    hz: Optional[float] = None
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
        self.hx = float(self.hx)
        self.hz = float(self.hx) if self.hz is None else float(self.hz)

