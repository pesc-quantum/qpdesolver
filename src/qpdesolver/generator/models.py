"""
Parametric 2D seismic velocity model generators.
"""
from __future__ import annotations

from typing import Tuple, Sequence, Union
import numpy as np


def two_layer_velocity(
    shape: Tuple[int, int] = (101, 101),
    v_top: float = 1.5,
    v_bottom: float = 2.5,
    interface_z: Union[float, int] = 0.5,
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Generate a 2D two-layer velocity model (nx, nz).

    Parameters:
        shape: (nx, nz) dimensions of the grid.
        v_top: Velocity of top layer (km/s).
        v_bottom: Velocity of bottom layer (km/s).
        interface_z: Depth of interface. If <= 1.0, treated as a fraction of nz;
                     otherwise treated as absolute grid index.
        dtype: Output array numpy dtype.

    Returns:
        np.ndarray of shape (nx, nz).
    """
    nx, nz = shape
    if 0.0 <= interface_z <= 1.0:
        cut = int(round(nz * interface_z))
    else:
        cut = int(round(interface_z))
    cut = max(0, min(nz, cut))

    v = np.full(shape, v_top, dtype=dtype)
    v[:, cut:] = v_bottom
    return v


def multi_layer_velocity(
    shape: Tuple[int, int] = (101, 101),
    velocities: Sequence[float] = (1.5, 2.0, 2.5),
    interfaces: Sequence[Union[float, int]] = (0.33, 0.66),
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Generate a horizontally stratified multi-layer 2D velocity model (nx, nz).

    Parameters:
        shape: (nx, nz) dimensions of the grid.
        velocities: Velocities for each layer from top to bottom (km/s).
        interfaces: Interface depths between layers. Length must be len(velocities) - 1.
                    Values <= 1.0 are treated as fractions of nz; > 1.0 as grid indices.
        dtype: Output array numpy dtype.

    Returns:
        np.ndarray of shape (nx, nz).
    """
    nx, nz = shape
    if len(interfaces) != len(velocities) - 1:
        raise ValueError(
            f"len(interfaces) ({len(interfaces)}) must be len(velocities) - 1 ({len(velocities) - 1})"
        )

    cut_indices = []
    for itf in interfaces:
        if 0.0 <= itf <= 1.0:
            idx = int(round(nz * itf))
        else:
            idx = int(round(itf))
        cut_indices.append(max(0, min(nz, idx)))

    v = np.empty(shape, dtype=dtype)
    prev = 0
    for vel, cut in zip(velocities[:-1], cut_indices):
        v[:, prev:cut] = vel
        prev = cut
    v[:, prev:] = velocities[-1]
    return v


def gaussian_anomaly_velocity(
    shape: Tuple[int, int] = (101, 101),
    v_bg: float = 2.0,
    dv: float = 0.6,
    center: Tuple[float, float] = (0.5, 0.5),
    sigma: Union[float, Tuple[float, float]] = (0.1, 0.1),
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Generate a 2D velocity model with a smooth Gaussian anomaly.

    Parameters:
        shape: (nx, nz) dimensions of the grid.
        v_bg: Background velocity (km/s).
        dv: Peak perturbation amplitude (+/- km/s).
        center: Anomaly center (cx, cz). If <= 1.0, relative to shape; else grid indices.
        sigma: Anomaly standard deviation (sx, sz) or scalar. If <= 1.0, relative to shape; else grid units.
        dtype: Output array numpy dtype.

    Returns:
        np.ndarray of shape (nx, nz).
    """
    nx, nz = shape
    cx = center[0] * (nx - 1) if 0.0 <= center[0] <= 1.0 else center[0]
    cz = center[1] * (nz - 1) if 0.0 <= center[1] <= 1.0 else center[1]

    if isinstance(sigma, (int, float)):
        sig_x = sigma * (nx - 1) if 0.0 < sigma <= 1.0 else float(sigma)
        sig_z = sigma * (nz - 1) if 0.0 < sigma <= 1.0 else float(sigma)
    else:
        sig_x = sigma[0] * (nx - 1) if 0.0 < sigma[0] <= 1.0 else float(sigma[0])
        sig_z = sigma[1] * (nz - 1) if 0.0 < sigma[1] <= 1.0 else float(sigma[1])

    sig_x = max(sig_x, 1e-6)
    sig_z = max(sig_z, 1e-6)

    x = np.arange(nx, dtype=np.float64)
    z = np.arange(nz, dtype=np.float64)
    X, Z = np.meshgrid(x, z, indexing="ij")

    gaussian = np.exp(-0.5 * (((X - cx) / sig_x) ** 2 + ((Z - cz) / sig_z) ** 2))
    v = v_bg + dv * gaussian
    return v.astype(dtype)


def salt_inclusion_velocity(
    shape: Tuple[int, int] = (101, 101),
    v_bg: float = 2.0,
    v_salt: float = 4.5,
    center: Tuple[float, float] = (0.5, 0.5),
    radius: Union[float, Tuple[float, float]] = (0.2, 0.15),
    geometry: str = "ellipse",
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """
    Generate a 2D velocity model containing a high-velocity salt body inclusion.

    Parameters:
        shape: (nx, nz) dimensions of the grid.
        v_bg: Background velocity (km/s).
        v_salt: Salt body velocity (km/s).
        center: Inclusion center (cx, cz). If <= 1.0, relative to shape; else grid indices.
        radius: Inclusion half-axes (rx, rz) or scalar. If <= 1.0, relative to shape; else grid units.
        geometry: Shape of inclusion ("ellipse" or "box").
        dtype: Output array numpy dtype.

    Returns:
        np.ndarray of shape (nx, nz).
    """
    nx, nz = shape
    cx = center[0] * (nx - 1) if 0.0 <= center[0] <= 1.0 else center[0]
    cz = center[1] * (nz - 1) if 0.0 <= center[1] <= 1.0 else center[1]

    if isinstance(radius, (int, float)):
        rx = radius * (nx - 1) if 0.0 < radius <= 1.0 else float(radius)
        rz = radius * (nz - 1) if 0.0 < radius <= 1.0 else float(radius)
    else:
        rx = radius[0] * (nx - 1) if 0.0 < radius[0] <= 1.0 else float(radius[0])
        rz = radius[1] * (nz - 1) if 0.0 < radius[1] <= 1.0 else float(radius[1])

    rx = max(rx, 1e-6)
    rz = max(rz, 1e-6)

    x = np.arange(nx, dtype=np.float64)
    z = np.arange(nz, dtype=np.float64)
    X, Z = np.meshgrid(x, z, indexing="ij")

    v = np.full(shape, v_bg, dtype=dtype)
    if geometry == "ellipse":
        mask = (((X - cx) / rx) ** 2 + ((Z - cz) / rz) ** 2) <= 1.0
    elif geometry == "box":
        mask = (np.abs(X - cx) <= rx) & (np.abs(Z - cz) <= rz)
    else:
        raise ValueError(f"Unknown geometry '{geometry}'. Choose 'ellipse' or 'box'.")

    v[mask] = v_salt
    return v

