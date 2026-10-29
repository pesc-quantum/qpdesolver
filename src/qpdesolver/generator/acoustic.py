"""
2D acoustic wave simulation and seismic survey generator.
Supports both Devito and pure NumPy/SciPy finite-difference wave propagators.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

from qpdesolver.io.devito_io import export_devito_hdf5


@dataclass
class SeismicSurveySpec:
    """
    Specification for a 2D synthetic seismic survey.

    Parameters:
        shape: (nx, nz) spatial grid dimensions.
        spacing: (dx, dz) physical grid spacing in meters.
        origin: (ox, oz) physical origin in meters.
        nbl: Number of absorbing boundary layers (padding cells).
        t0: Simulation start time in ms.
        tn: Simulation end time in ms.
        dt: Time step in ms. If None, derived from CFL stability condition.
        f0: Peak source frequency in kHz (e.g. 0.010 kHz = 10 Hz).
        nsnaps: Number of wavefield snapshots to record.
        space_order: Spatial discretization order (e.g. 2, 4).
        src_coords: Array of source physical coordinates, shape (nsrc, 2).
        rec_coords: Array of receiver physical coordinates, shape (nrec, 2).
        nrec: Number of receivers along the surface if rec_coords is None.
        strip_nbl: Whether to strip the absorbing layer from exported fields.
        dtype: Floating point precision ("float32" or "float64").
    """
    shape: Tuple[int, int] = (101, 101)
    spacing: Tuple[float, float] = (10.0, 10.0)
    origin: Tuple[float, float] = (0.0, 0.0)
    nbl: int = 10
    t0: float = 0.0
    tn: float = 1000.0
    dt: Optional[float] = None
    f0: float = 0.010
    nsnaps: int = 50
    space_order: int = 2
    src_coords: Optional[np.ndarray] = None
    rec_coords: Optional[np.ndarray] = None
    nrec: Optional[int] = None
    strip_nbl: bool = True
    dtype: str = "float32"


def _simulate_acoustic_numpy(
    filename: Union[str, Path],
    vp: np.ndarray,
    spec: SeismicSurveySpec,
    *,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
    overwrite: bool = True,
    attrs: Optional[Dict[str, Any]] = None,
) -> str:
    """Pure NumPy 2D acoustic finite-difference simulation."""
    nx, nz = spec.shape
    dx, dz = spec.spacing
    ox, oz = spec.origin
    nbl = int(spec.nbl)

    # Pad velocity domain with absorbing boundary if nbl > 0
    if nbl > 0:
        vp_pad = np.pad(vp, nbl, mode="edge")
    else:
        vp_pad = vp.copy()

    nx_pad, nz_pad = vp_pad.shape

    # Time stepping: CFL stability dt <= 0.5 * min(dx, dz) / max(vp)
    max_vp = float(np.max(vp))
    cfl_dt = 0.5 * min(dx, dz) / max_vp
    dt = float(spec.dt) if spec.dt is not None else float(cfl_dt)
    nt = int(math.ceil((spec.tn - spec.t0) / dt)) + 1
    time_values = np.linspace(spec.t0, spec.tn, nt, dtype=np.float64)

    # Ricker source wavelet
    f0 = spec.f0
    t0_ricker = 1.0 / f0
    tau = np.pi * f0 * (time_values - t0_ricker)
    src_signal = ((1.0 - 2.0 * tau**2) * np.exp(-tau**2)).astype(spec.dtype)

    # Source coordinate (default: top-center, 2 * dz depth)
    if spec.src_coords is not None:
        src_coords = np.asarray(spec.src_coords, dtype=np.float32)
    else:
        src_x = ox + (nx - 1) * dx * 0.5
        src_z = oz + 2.0 * dz
        src_coords = np.array([[src_x, src_z]], dtype=np.float32)

    # Nearest grid point for source injection
    s_i = int(round((src_coords[0, 0] - ox) / dx + nbl))
    s_j = int(round((src_coords[0, 1] - oz) / dz + nbl))
    s_i = max(1, min(nx_pad - 2, s_i))
    s_j = max(1, min(nz_pad - 2, s_j))

    # Receiver coordinates (default: line of receivers at 2 * dz depth)
    if spec.rec_coords is not None:
        rec_coords = np.asarray(spec.rec_coords, dtype=np.float32)
    else:
        nrec = spec.nrec if spec.nrec is not None else nx
        rec_x = np.linspace(ox, ox + (nx - 1) * dx, nrec)
        rec_z = np.full(nrec, oz + 2.0 * dz)
        rec_coords = np.column_stack([rec_x, rec_z]).astype(np.float32)

    n_receivers = len(rec_coords)
    rec_data = np.zeros((nt, n_receivers), dtype=spec.dtype)

    # Bilinear interpolation weights for receivers
    gx = (rec_coords[:, 0] - ox) / dx + nbl
    gz = (rec_coords[:, 1] - oz) / dz + nbl
    rec_i0 = np.clip(np.floor(gx).astype(int), 0, nx_pad - 2)
    rec_j0 = np.clip(np.floor(gz).astype(int), 0, nz_pad - 2)
    rec_wx = (gx - rec_i0).astype(spec.dtype)
    rec_wz = (gz - rec_j0).astype(spec.dtype)

    # Cerjan boundary absorbing sponge
    sponge = np.ones((nx_pad, nz_pad), dtype=np.float32)
    if nbl > 0:
        for i in range(nx_pad):
            for j in range(nz_pad):
                d_x = max(0, nbl - i, i - (nx_pad - 1 - nbl))
                d_z = max(0, nbl - j, j - (nz_pad - 1 - nbl))
                d = max(d_x, d_z)
                if d > 0:
                    sponge[i, j] = np.exp(-((0.015 * d) ** 2))

    # Snapshots recording schedule
    nsnaps_req = max(1, min(spec.nsnaps, nt))
    factor = max(1, int(math.ceil(nt / nsnaps_req)))
    snap_indices = np.arange(0, nt, factor)[:nsnaps_req]
    snap_set = set(snap_indices)
    snaps = np.zeros((len(snap_indices), nx_pad, nz_pad), dtype=spec.dtype)
    snap_ptr = 0

    # Finite difference state buffers
    u_prev = np.zeros((nx_pad, nz_pad), dtype=spec.dtype)
    u_curr = np.zeros((nx_pad, nz_pad), dtype=spec.dtype)
    u_next = np.zeros((nx_pad, nz_pad), dtype=spec.dtype)
    inv_dx2 = 1.0 / (dx * dx)
    inv_dz2 = 1.0 / (dz * dz)
    c2_dt2 = (dt * vp_pad) ** 2

    # Time integration loop
    for it in range(nt):
        # 2nd-order 5-point spatial Laplacian
        lap = np.zeros_like(u_curr)
        lap[1:-1, 1:-1] = (
            (u_curr[2:, 1:-1] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[:-2, 1:-1]) * inv_dx2
            + (u_curr[1:-1, 2:] - 2.0 * u_curr[1:-1, 1:-1] + u_curr[1:-1, :-2]) * inv_dz2
        )

        u_next = 2.0 * u_curr - u_prev + c2_dt2 * lap
        u_next[s_i, s_j] += c2_dt2[s_i, s_j] * src_signal[it] / (dx * dz)

        if nbl > 0:
            u_next *= sponge
            u_curr *= sponge

        # Bilinear receiver interpolation
        rec_data[it, :] = (
            (1.0 - rec_wx) * (1.0 - rec_wz) * u_curr[rec_i0, rec_j0]
            + rec_wx * (1.0 - rec_wz) * u_curr[rec_i0 + 1, rec_j0]
            + (1.0 - rec_wx) * rec_wz * u_curr[rec_i0, rec_j0 + 1]
            + rec_wx * rec_wz * u_curr[rec_i0 + 1, rec_j0 + 1]
        )

        if it in snap_set and snap_ptr < len(snap_indices):
            snaps[snap_ptr] = u_curr.copy()
            snap_ptr += 1

        u_prev[:] = u_curr
        u_curr[:] = u_next

    # Strip absorbing layer if requested
    if spec.strip_nbl and nbl > 0:
        vp_out = vp
        snaps_out = snaps[:, nbl:-nbl, nbl:-nbl]
    else:
        vp_out = vp_pad
        snaps_out = snaps

    extra_attrs = dict(attrs or {})
    extra_attrs["propagator_backend"] = "numpy"

    return export_devito_hdf5(
        filename=filename,
        vp=vp_out,
        origin=spec.origin,
        spacing=spec.spacing,
        time_values=time_values,
        snapshots={"u": snaps_out},
        sources={"src": {"coordinates": src_coords, "data": src_signal[:, None], "f0": spec.f0}},
        receivers={"rec": {"coordinates": rec_coords, "data": rec_data}},
        strip_nbl=False,
        dtype=spec.dtype,
        compression=compression,
        compression_opts=compression_opts,
        overwrite=overwrite,
        attrs=extra_attrs,
    )


def _simulate_acoustic_devito(
    filename: Union[str, Path],
    vp: np.ndarray,
    spec: SeismicSurveySpec,
    *,
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
    overwrite: bool = True,
    attrs: Optional[Dict[str, Any]] = None,
) -> str:
    """Devito 2D acoustic finite-difference simulation."""
    try:
        from examples.seismic import Model, TimeAxis, RickerSource, Receiver
        from devito import TimeFunction, Eq, solve, Operator
    except ImportError as exc:
        raise ImportError(
            "Devito backend requires 'devito' and 'examples.seismic'. "
            "Install devito or use backend='numpy'."
        ) from exc
    from qpdesolver.generator.recorder import SnapshotRecorder

    model = Model(
        vp=vp,
        origin=spec.origin,
        spacing=spec.spacing,
        shape=spec.shape,
        nbl=spec.nbl,
        space_order=spec.space_order,
        dtype=np.float32,
    )

    dt = spec.dt if spec.dt is not None else float(model.critical_dt)
    time_range = TimeAxis(start=spec.t0, stop=spec.tn, step=dt)

    # Source setup
    src = RickerSource(name="src", grid=model.grid, f0=spec.f0, npoint=1, time_range=time_range)
    if spec.src_coords is not None:
        src.coordinates.data[0, :] = spec.src_coords[0]
    else:
        src.coordinates.data[0, 0] = spec.origin[0] + (spec.shape[0] - 1) * spec.spacing[0] * 0.5
        src.coordinates.data[0, 1] = spec.origin[1] + 2.0 * spec.spacing[1]

    # Receiver setup
    nrec = spec.nrec if spec.nrec is not None else spec.shape[0]
    rec = Receiver(name="rec", grid=model.grid, npoint=nrec, time_range=time_range)
    if spec.rec_coords is not None:
        rec.coordinates.data[:, :] = spec.rec_coords
    else:
        rec.coordinates.data[:, 0] = np.linspace(
            spec.origin[0], spec.origin[0] + (spec.shape[0] - 1) * spec.spacing[0], num=nrec
        )
        rec.coordinates.data[:, 1] = spec.origin[1] + 2.0 * spec.spacing[1]

    # Wave equation
    u = TimeFunction(name="u", grid=model.grid, time_order=2, space_order=spec.space_order)
    pde = model.m * u.dt2 - u.laplace + model.damp * u.dt
    stencil = Eq(u.forward, solve(pde, u.forward))
    src_term = src.inject(field=u.forward, expr=src * dt**2 / model.m)
    rec_term = rec.interpolate(expr=u.forward)

    snap = SnapshotRecorder(u, nt=time_range.num, nsnaps=spec.nsnaps)
    op = Operator([stencil] + src_term + rec_term + snap.equations, subs=model.spacing_map)
    op(time=time_range.num - 1, dt=dt)

    extra_attrs = dict(attrs or {})
    extra_attrs["propagator_backend"] = "devito"

    return export_devito_hdf5(
        filename=filename,
        model=model,
        time_range=time_range,
        snapshots=snap,
        sources={"src": src},
        receivers={"rec": rec},
        strip_nbl=spec.strip_nbl,
        dtype=spec.dtype,
        compression=compression,
        compression_opts=compression_opts,
        overwrite=overwrite,
        attrs=extra_attrs,
    )


def generate_seismic_sample(
    filename: Union[str, Path],
    vp: np.ndarray,
    spec: Optional[SeismicSurveySpec] = None,
    *,
    backend: str = "auto",
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
    overwrite: bool = True,
    attrs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Generate synthetic seismic wavefield simulation data and export to HDF5.

    Parameters:
        filename: Destination HDF5 file path.
        vp: 2D velocity array of shape (nx, nz) in km/s.
        spec: SeismicSurveySpec surveying parameters. If None, default spec for vp.shape is used.
        backend: Wave propagator engine: "auto", "devito", or "numpy".
        compression: Dataset compression method ("gzip", "lzf", or None).
        compression_opts: Compression level (default 4).
        overwrite: Whether to overwrite existing destination file.
        attrs: Optional dictionary of root metadata attributes.

    Returns:
        str path of the exported HDF5 file.
    """
    vp = np.asarray(vp, dtype=np.float32)
    if vp.ndim != 2:
        raise ValueError(f"vp must be a 2D array, got shape {vp.shape}")

    if spec is None:
        spec = SeismicSurveySpec(shape=vp.shape)
    elif spec.shape != vp.shape:
        raise ValueError(f"vp shape {vp.shape} does not match spec.shape {spec.shape}")

    if backend == "auto":
        try:
            import devito
            import examples.seismic
            resolved_backend = "devito"
        except ImportError:
            resolved_backend = "numpy"
    elif backend in ("devito", "numpy"):
        resolved_backend = backend
    else:
        raise ValueError(f"Unknown backend '{backend}'. Supported backends: 'auto', 'devito', 'numpy'.")

    if resolved_backend == "devito":
        return _simulate_acoustic_devito(
            filename=filename,
            vp=vp,
            spec=spec,
            compression=compression,
            compression_opts=compression_opts,
            overwrite=overwrite,
            attrs=attrs,
        )
    else:
        return _simulate_acoustic_numpy(
            filename=filename,
            vp=vp,
            spec=spec,
            compression=compression,
            compression_opts=compression_opts,
            overwrite=overwrite,
            attrs=attrs,
        )

