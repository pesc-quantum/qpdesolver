"""
Explicit normalization subsystem for seismic velocity models and PDE discretizations.

Provides uniform scaling and units metadata management for Devito HDF5 outputs,
FWI datasets, and linear systems.
"""

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Sequence, Tuple, Union
import numpy as np

UnitTarget = Literal["si", "kms", "dimensionless"]


@dataclass(frozen=True)
class NormalizationParams:
    """
    Metadata and scaling parameters for physical and dimensionless normalization.

    Attributes:
        normalized: Whether explicit normalization was applied.
        source_units: Source units of the input dataset ('kms', 'si', or 'raw').
        target_units: Target units after normalization ('raw', 'si', 'kms', or 'dimensionless').
        velocity_scale: Factor applied to velocity (v_out = v_in * velocity_scale).
        length_scale: Factor applied to grid spacing / length (h_out = h_in * length_scale).
        time_scale: Factor applied to time coordinates (t_out = t_in * time_scale).
        omega_scale: Factor applied to angular frequency (omega_out = omega * omega_scale).
        v0: Reference velocity used for scaling (if dimensionless), in source units.
        L0: Reference length used for scaling (if dimensionless), in source units.
    """

    normalized: bool = False
    source_units: str = "raw"
    target_units: str = "raw"
    velocity_scale: float = 1.0
    length_scale: float = 1.0
    time_scale: float = 1.0
    omega_scale: float = 1.0
    v0: Optional[float] = None
    L0: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert normalization parameters to a serializable dictionary."""
        return {
            "normalized": self.normalized,
            "source_units": self.source_units,
            "target_units": self.target_units,
            "velocity_scale": float(self.velocity_scale),
            "length_scale": float(self.length_scale),
            "time_scale": float(self.time_scale),
            "omega_scale": float(self.omega_scale),
            "v0": float(self.v0) if self.v0 is not None else None,
            "L0": float(self.L0) if self.L0 is not None else None,
        }

    def denormalize_velocity(
        self, v: Union[float, Sequence[float], np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Convert velocity back to source units."""
        return denormalize_velocity(v, self)

    def scale_omega(
        self, omega: Union[float, Sequence[float], np.ndarray]
    ) -> Union[float, np.ndarray]:
        """Scale angular frequency according to normalization parameters."""
        if isinstance(omega, (int, float, np.floating, np.integer)):
            return float(omega) * self.omega_scale
        return np.asarray(omega) * self.omega_scale


def compute_normalization_params(
    v: Union[float, Sequence[float], np.ndarray],
    h: Union[float, Sequence[float], np.ndarray] = 1.0,
    normalize: bool = False,
    unit_target: Optional[Union[str, UnitTarget]] = None,
    source_units: str = "auto",
) -> NormalizationParams:
    """
    Compute explicit scaling parameters for velocity, spatial grid, time, and frequency.

    Args:
        v: Velocity array or scalar.
        h: Spatial grid spacing (scalar float or tuple/list/array of spacings).
        normalize: If False, returns verbatim unity scaling. If True, computes
            scaling according to `unit_target`.
        unit_target: Target system of units ('si', 'kms', or 'dimensionless').
            Must be None when normalize=False. Defaults to 'si' when normalize=True.
        source_units: Source units of input ('auto', 'kms', or 'si').

    Returns:
        NormalizationParams dataclass instance.

    Raises:
        ValueError: If unit_target is provided when normalize=False, or if
            unit_target or source_units is invalid.
    """
    if not normalize:
        if unit_target is not None:
            raise ValueError("unit_target requires normalize=True")
        return NormalizationParams(
            normalized=False,
            source_units=source_units if source_units != "auto" else "raw",
            target_units="raw",
            velocity_scale=1.0,
            length_scale=1.0,
            time_scale=1.0,
            omega_scale=1.0,
            v0=None,
            L0=None,
        )

    if unit_target is None:
        unit_target = "si"

    valid_targets = {"si", "kms", "dimensionless"}
    if unit_target not in valid_targets:
        raise ValueError(
            f"Unsupported unit_target: '{unit_target}'. Must be one of {valid_targets}"
        )

    v_arr = np.asarray(v, dtype=float)
    if source_units == "auto":
        # Devito convention: velocities < 100.0 are in km/s (typically 1.0-8.0 km/s)
        # while velocities >= 100.0 are in m/s (typically 1000-8000 m/s).
        mean_v = float(np.mean(np.abs(v_arr)))
        detected_source = "kms" if mean_v < 100.0 else "si"
    elif source_units in {"kms", "si"}:
        detected_source = source_units
    else:
        raise ValueError(
            f"Unsupported source_units: '{source_units}'. Must be 'auto', 'kms', or 'si'"
        )

    h_arr = np.asarray(h, dtype=float)
    L0_src = float(np.mean(h_arr))
    v0_src = float(np.mean(v_arr))

    if v0_src == 0.0:
        raise ValueError("Reference velocity v0 cannot be zero.")

    # Calculate reference quantities in SI
    # Devito seismic standard:
    # 'kms': velocity in km/s, spacing h in meters, time in ms
    # 'si':  velocity in m/s,  spacing h in meters, time in s
    if detected_source == "kms":
        v0_si = v0_src * 1000.0
        L0_si = L0_src
    else:  # "si"
        v0_si = v0_src
        L0_si = L0_src

    if unit_target == "si":
        velocity_scale = 1000.0 if detected_source == "kms" else 1.0
        length_scale = 1.0
        time_scale = 1e-3 if detected_source == "kms" else 1.0
        omega_scale = 1.0
        v0 = v0_si
        L0 = L0_si
    elif unit_target == "kms":
        velocity_scale = 1.0 if detected_source == "kms" else 1e-3
        length_scale = 1e-3  # Spacing h from m to km
        time_scale = 1e-3 if detected_source == "kms" else 1.0
        omega_scale = 1.0
        v0 = v0_si / 1000.0
        L0 = L0_si / 1000.0
    elif unit_target == "dimensionless":
        # Scale coordinates by L0 and velocity by v0 so that v_dim = v / v0 and h_dim = h / L0
        velocity_scale = 1.0 / v0_src
        length_scale = 1.0 / L0_src
        time_scale = (1e-3 if detected_source == "kms" else 1.0) * (v0_si / L0_si)
        omega_scale = L0_si / v0_si
        v0 = v0_src
        L0 = L0_src

    return NormalizationParams(
        normalized=True,
        source_units=detected_source,
        target_units=unit_target,
        velocity_scale=velocity_scale,
        length_scale=length_scale,
        time_scale=time_scale,
        omega_scale=omega_scale,
        v0=v0,
        L0=L0,
    )


def apply_seismic_normalization(
    v: Union[float, Sequence[float], np.ndarray],
    h: Union[float, Sequence[float], np.ndarray],
    omega: Optional[Union[float, Sequence[float], np.ndarray, NormalizationParams]] = None,
    params: Optional[NormalizationParams] = None,
) -> Tuple[Any, Any, Any]:
    """
    Apply scaling parameters to velocity, grid spacing, and optional angular frequency.

    Args:
        v: Velocity array or scalar.
        h: Spatial grid spacing (scalar float or tuple/list/array of spacings).
        omega: Optional angular frequency (scalar or array) or NormalizationParams.
        params: NormalizationParams instance. If omitted, verbatim scaling is applied.

    Returns:
        A 3-tuple:
            (v_norm, h_norm, omega_scaled_or_fn)
            Where omega_scaled_or_fn is the scaled frequency (if omega was provided)
            or the callable params.scale_omega method.
    """
    if isinstance(omega, NormalizationParams) and params is None:
        params = omega
        omega = None

    if params is None:
        params = compute_normalization_params(v, h=h, normalize=False)

    # Scale velocity
    if isinstance(v, (int, float, np.floating, np.integer)):
        v_out = float(v) * params.velocity_scale
    else:
        v_out = np.asarray(v) * params.velocity_scale

    # Scale grid spacing
    if isinstance(h, (int, float, np.floating, np.integer)):
        h_out = float(h) * params.length_scale
    elif isinstance(h, tuple):
        h_out = tuple(float(x) * params.length_scale for x in h)
    elif isinstance(h, list):
        h_out = [float(x) * params.length_scale for x in h]
    else:
        h_out = np.asarray(h) * params.length_scale

    # Frequency output
    if omega is None:
        omega_out = params.scale_omega
    else:
        omega_out = params.scale_omega(omega)

    return v_out, h_out, omega_out


def denormalize_velocity(
    v: Union[float, Sequence[float], np.ndarray],
    params: NormalizationParams,
) -> Union[float, np.ndarray]:
    """
    Convert normalized velocity back to source units.

    Args:
        v: Normalized velocity array or scalar.
        params: NormalizationParams instance containing scaling factors.

    Returns:
        Denormalized velocity in source units.
    """
    if params.velocity_scale == 0.0:
        raise ValueError("velocity_scale cannot be zero for denormalization.")

    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v) / params.velocity_scale
    return np.asarray(v) / params.velocity_scale
