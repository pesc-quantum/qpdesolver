"""
Optimized I/O routines for devito-seismic-hdf5 files and FWI bridges.
Provides snapshot-chunked, compressed export, selective lazy loading,
summarization, and bidirectional conversion to qpdesolver FWI datasets.
"""
from __future__ import annotations

import os
import math
import datetime as _dt
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Sequence, Union

import numpy as np
import h5py
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.ndimage import gaussian_filter

from qpdesolver.io.schema import DEVITO_FORMAT, SCHEMA_VERSION
from qpdesolver.io.normalization import (
    compute_normalization_params,
    apply_seismic_normalization,
    NormalizationParams,
)
from qpdesolver.applications.fwi.model import VelocityModel, FWIDataset, Acquisition

try:
    from qpdesolver.applications.fwi.model import VelocityModel2D, Acquisition2D, FWIDataset2D
except ImportError:
    VelocityModel2D = None  # type: ignore
    Acquisition2D = None  # type: ignore
    FWIDataset2D = None  # type: ignore


def _as_float_tuple(x: Any) -> Tuple[float, ...]:
    try:
        return tuple(float(v) for v in x)
    except TypeError:
        return (float(x),)


def _clean_h5_attr(v: Any) -> Any:
    if isinstance(v, bytes):
        return v.decode()
    if isinstance(v, np.ndarray) and v.dtype.kind in "SU":
        return [x.decode() if isinstance(x, bytes) else str(x) for x in v]
    return v


def export_devito_hdf5(
    filename: Union[str, Path],
    *,
    model: Optional[Any] = None,
    time_range: Optional[Any] = None,
    snapshots: Optional[Union[Any, Dict[str, np.ndarray]]] = None,
    fields: Optional[Any] = None,
    sources: Optional[Dict[str, Any]] = None,
    receivers: Optional[Dict[str, Any]] = None,
    vp: Optional[np.ndarray] = None,
    origin: Optional[Tuple[float, ...]] = None,
    spacing: Optional[Tuple[float, ...]] = None,
    time_values: Optional[Union[Sequence[float], np.ndarray]] = None,
    strip_nbl: bool = True,
    dtype: Optional[str] = "float32",
    compression: Optional[str] = "gzip",
    compression_opts: Optional[int] = 4,
    attrs: Optional[Dict[str, Any]] = None,
    overwrite: bool = True,
) -> str:
    """
    Export synthetic seismic simulation data to a self-describing devito-seismic-hdf5 file.

    Follows chunked storage layout: snapshots are chunked as (1, *space) when compressed,
    with shuffle=True for gzip. Physical units are explicitly tagged:
    - model/vp: "km/s"
    - coords/* and model spacing: "m"
    - time/*: "ms"
    """
    path_str = str(filename)
    if os.path.exists(path_str) and not overwrite:
        raise FileExistsError(f"File '{path_str}' already exists and overwrite=False.")

    cast = (lambda a: np.asarray(a).astype(dtype, copy=False)) if dtype else np.asarray
    nbl = int(getattr(model, "nbl", 0) or 0) if (model is not None and strip_nbl) else 0

    with h5py.File(path_str, "w") as h5:
        # 1. Root metadata
        h5.attrs["format"] = DEVITO_FORMAT
        h5.attrs["format_version"] = "1.0"
        h5.attrs["created"] = _dt.datetime.now().astimezone().isoformat()
        try:
            import devito
            h5.attrs["devito_version"] = devito.__version__
        except Exception:
            pass
        h5.attrs["numpy_version"] = np.__version__
        h5.attrs["absorbing_layer_stripped"] = bool(nbl)
        for k, v in (attrs or {}).items():
            h5.attrs[k] = v

        # 2. Model group
        shape = None
        spc = None
        org = None

        if model is not None:
            shape = tuple(int(s) for s in model.shape)
            spc = _as_float_tuple(model.spacing)
            org = _as_float_tuple(model.origin)
            domain_size = _as_float_tuple(getattr(model, "domain_size", [s * d for s, d in zip(shape, spc)]))
            crit_dt = float(getattr(model, "critical_dt", 1.0))
            so = getattr(model, "space_order", None)

            g = h5.create_group("model")
            g.attrs["shape"] = np.asarray(shape, dtype=np.int64)
            g.attrs["spacing"] = np.asarray(spc, dtype=np.float64)
            g.attrs["origin"] = np.asarray(org, dtype=np.float64)
            g.attrs["domain_size"] = np.asarray(domain_size, dtype=np.float64)
            g.attrs["nbl"] = nbl
            g.attrs["critical_dt"] = crit_dt
            g.attrs["dimensions"] = [d.name for d in model.grid.dimensions]
            if so is not None:
                g.attrs["space_order"] = int(so)

            try:
                params = model.physical_params()
            except Exception:
                params = {}
            for pname, pfunc in params.items():
                arr = cast(np.asarray(pfunc.data))
                if nbl:
                    from qpdesolver.generator.devito_hdf5 import strip_absorbing_layer
                    arr = strip_absorbing_layer(arr, nbl, ndim=arr.ndim)
                ds = g.create_dataset(pname, data=arr)
                if pname == "vp":
                    ds.attrs["units"] = "km/s"
                ds.attrs["description"] = f"model parameter {pname}"

        elif vp is not None:
            vp_arr = cast(vp)
            shape = tuple(int(s) for s in vp_arr.shape)
            spc = _as_float_tuple(spacing if spacing is not None else tuple(1.0 for _ in shape))
            org = _as_float_tuple(origin if origin is not None else tuple(0.0 for _ in shape))
            domain_size = tuple(float(s * d) for s, d in zip(shape, spc))

            g = h5.create_group("model")
            g.attrs["shape"] = np.asarray(shape, dtype=np.int64)
            g.attrs["spacing"] = np.asarray(spc, dtype=np.float64)
            g.attrs["origin"] = np.asarray(org, dtype=np.float64)
            g.attrs["domain_size"] = np.asarray(domain_size, dtype=np.float64)
            g.attrs["nbl"] = 0
            dim_names = ["x", "z"] if len(shape) == 2 else (["x", "y", "z"] if len(shape) == 3 else ["z"])
            g.attrs["dimensions"] = dim_names

            ds = g.create_dataset("vp", data=vp_arr)
            ds.attrs["units"] = "km/s"
            ds.attrs["description"] = "model parameter vp"

        if shape is not None and spc is not None and org is not None:
            c = h5.create_group("coords")
            dim_names = ["x", "z"] if len(shape) == 2 else (["x", "y", "z"] if len(shape) == 3 else ["z"])
            for i, d_name in enumerate(dim_names):
                axis_data = org[i] + np.arange(shape[i]) * spc[i]
                c_ds = c.create_dataset(d_name, data=np.asarray(axis_data, dtype=np.float64))
                c_ds.attrs["units"] = "m"
                c_ds.attrs["description"] = f"physical {d_name} axis"

        # 3. Time group
        tg = h5.create_group("time")
        tg.attrs["units"] = "ms"
        if time_range is not None:
            tg.attrs["t0"] = float(time_range.start)
            tg.attrs["tn"] = float(time_range.stop)
            tg.attrs["dt"] = float(time_range.step)
            tg.attrs["nt"] = int(time_range.num)
            time_ds = tg.create_dataset("time", data=np.asarray(time_range.time_values, dtype=np.float64))
            time_ds.attrs["units"] = "ms"
        elif time_values is not None:
            t_arr = np.asarray(time_values, dtype=np.float64)
            tg.attrs["t0"] = float(t_arr[0]) if len(t_arr) > 0 else 0.0
            tg.attrs["tn"] = float(t_arr[-1]) if len(t_arr) > 0 else 0.0
            tg.attrs["nt"] = len(t_arr)
            tg.attrs["dt"] = float(t_arr[1] - t_arr[0]) if len(t_arr) > 1 else 1.0
            time_ds = tg.create_dataset("time", data=t_arr)
            time_ds.attrs["units"] = "ms"

        # 4. Snapshots / Fields group with chunked storage (1, *space)
        if snapshots is not None:
            fg = h5.create_group("fields")
            if hasattr(snapshots, "arrays"):
                # SnapshotRecorder
                arrs = snapshots.arrays(nbl=nbl, dtype=dtype)
                tg.attrs["snapshot_factor"] = int(snapshots.factor)
                tg.attrs["snapshot_count"] = int(snapshots.nsnaps)
                tg.attrs["snapshot_record"] = snapshots.record
                tg.create_dataset("snapshot_index", data=snapshots.indices)
                if time_range is not None:
                    s_times = snapshots.times(time_range.step, time_range.start)
                    s_ds = tg.create_dataset("snapshot_time", data=s_times)
                    s_ds.attrs["units"] = "ms"
            else:
                arrs = {k: cast(v) for k, v in dict(snapshots).items()}
                if nbl:
                    from qpdesolver.generator.devito_hdf5 import strip_absorbing_layer
                    arrs = {k: strip_absorbing_layer(v, nbl, ndim=v.ndim - 1) for k, v in arrs.items()}
                if arrs:
                    tg.attrs["snapshot_count"] = int(len(next(iter(arrs.values()))))

            for name, arr in arrs.items():
                kw: Dict[str, Any] = {}
                if compression and arr.ndim > 1:
                    kw["compression"] = compression
                    if compression == "gzip":
                        if compression_opts is not None:
                            kw["compression_opts"] = compression_opts
                        kw["shuffle"] = True
                    # Snapshot chunking: (1, *space)
                    kw["chunks"] = (1, *arr.shape[1:])

                f_ds = fg.create_dataset(name, data=arr, **kw)
                f_ds.attrs["description"] = f"wavefield snapshots of {name}"
                f_ds.attrs["layout"] = "(snapshot, *space)"

        # 5. Geometry group: sources & receivers
        if sources or receivers:
            gg = h5.create_group("geometry")
            for kind, coll in (("source", sources or {}), ("receiver", receivers or {})):
                for name, obj in dict(coll).items():
                    sub = gg.create_group(name)
                    sub.attrs["kind"] = kind

                    if isinstance(obj, dict):
                        coords = obj.get("coordinates")
                        data = obj.get("data")
                        f0 = obj.get("f0")
                    else:
                        coords = getattr(getattr(obj, "coordinates", None), "data", None)
                        data = getattr(obj, "data", None)
                        f0 = getattr(obj, "f0", None)

                    if coords is not None:
                        c_ds = sub.create_dataset("coordinates", data=cast(coords))
                        c_ds.attrs["units"] = "m"
                        c_ds.attrs["layout"] = "(point, dim)"

                    if data is not None:
                        kw = {}
                        if compression and np.asarray(data).ndim > 0:
                            kw["compression"] = compression
                            if compression == "gzip" and compression_opts is not None:
                                kw["compression_opts"] = compression_opts
                        d_ds = sub.create_dataset("data", data=cast(data), **kw)
                        d_ds.attrs["layout"] = "(time, point)"
                        d_ds.attrs["description"] = f"{kind} traces"

                    if f0 is not None:
                        sub.attrs["f0"] = float(f0)

    return path_str


def load_devito_hdf5(
    filename: Union[str, Path],
    *,
    include_fields: bool = True,
    snapshot_indices: Optional[Sequence[int]] = None,
    normalize: bool = False,
    unit_target: Optional[str] = None,
    source_units: str = "auto",
) -> Dict[str, Any]:
    """
    Load a devito-seismic-hdf5 file into a nested Python dictionary.

    Parameters:
        filename: Path to the HDF5 file.
        include_fields: If False, skips loading wavefield arrays from 'fields' into memory.
        snapshot_indices: Optional sequence of integer snapshot indices to selectively load.
        normalize: Whether to apply explicit normalization.
        unit_target: Target unit system ('si', 'kms', 'dimensionless'). Must be None if normalize=False.
        source_units: Source units of input ('auto', 'kms', or 'si').
    """
    path_str = str(filename)
    if not os.path.isfile(path_str):
        raise FileNotFoundError(f"File not found: {path_str}")

    out: Dict[str, Any] = {}

    with h5py.File(path_str, "r") as h5:
        # Load root attrs
        out["attrs"] = {k: _clean_h5_attr(v) for k, v in h5.attrs.items()}

        def _walk_group(grp: h5py.Group, grp_path: str) -> Dict[str, Any]:
            res: Dict[str, Any] = {"attrs": {k: _clean_h5_attr(v) for k, v in grp.attrs.items()}}
            for key, item in grp.items():
                if isinstance(item, h5py.Group):
                    if key == "fields" and not include_fields:
                        res[key] = {"attrs": {k: _clean_h5_attr(v) for k, v in item.attrs.items()}}
                    else:
                        res[key] = _walk_group(item, f"{grp_path}/{key}")
                elif isinstance(item, h5py.Dataset):
                    if grp_path.endswith("/fields") or grp_path == "fields":
                        if not include_fields:
                            continue
                        if snapshot_indices is not None and item.ndim >= 1:
                            idx = list(snapshot_indices)
                            res[key] = item[idx]
                        else:
                            res[key] = item[()]
                    else:
                        res[key] = item[()]
            return res

        for key, item in h5.items():
            if isinstance(item, h5py.Group):
                if key == "fields" and not include_fields:
                    out[key] = {"attrs": {k: _clean_h5_attr(v) for k, v in item.attrs.items()}}
                else:
                    out[key] = _walk_group(item, key)
            elif isinstance(item, h5py.Dataset):
                out[key] = item[()]

    # Explicit normalization
    vp = out.get("model", {}).get("vp")
    spacing = out.get("model", {}).get("attrs", {}).get("spacing", (10.0, 10.0))
    h_eff = float(np.mean(spacing)) if hasattr(spacing, "__len__") else float(spacing)

    if not normalize:
        if unit_target is not None:
            raise ValueError("unit_target requires normalize=True")
        params = compute_normalization_params(vp if vp is not None else 1.0, h=h_eff, normalize=False)
        out["attrs"]["normalization"] = params.to_dict()
    else:
        src_u = source_units
        if src_u == "auto":
            # Check unit attribute on vp dataset
            vp_units = str(out.get("model", {}).get("attrs", {}).get("units", ""))
            if "km/s" in vp_units or (vp is not None and np.mean(vp) < 100.0):
                src_u = "kms"
            else:
                src_u = "si"

        params = compute_normalization_params(
            vp if vp is not None else 1.0,
            h=h_eff,
            normalize=True,
            unit_target=unit_target,
            source_units=src_u,
        )

        if vp is not None:
            out["model"]["vp"] = np.asarray(vp) * params.velocity_scale
        if "spacing" in out.get("model", {}).get("attrs", {}):
            out["model"]["attrs"]["spacing"] = np.asarray(out["model"]["attrs"]["spacing"]) * params.length_scale
        if "origin" in out.get("model", {}).get("attrs", {}):
            out["model"]["attrs"]["origin"] = np.asarray(out["model"]["attrs"]["origin"]) * params.length_scale
        if "coords" in out:
            for c_name, c_arr in out["coords"].items():
                if isinstance(c_arr, np.ndarray):
                    out["coords"][c_name] = c_arr * params.length_scale
        if "time" in out:
            for t_key in ("time", "snapshot_time"):
                if t_key in out["time"] and isinstance(out["time"][t_key], np.ndarray):
                    out["time"][t_key] = out["time"][t_key] * params.time_scale
        if "geometry" in out:
            for g_name, g_info in out["geometry"].items():
                if isinstance(g_info, dict) and "coordinates" in g_info:
                    g_info["coordinates"] = np.asarray(g_info["coordinates"]) * params.length_scale

        out["attrs"]["normalization"] = params.to_dict()

    return out


def summarize_hdf5(
    filename: Union[str, Path],
    show_attrs: bool = True,
    print_output: bool = True,
) -> str:
    """
    Summarize the structure of an HDF5 file (compatible with both devito-seismic-hdf5
    and qpdesolver formats).
    """
    path_str = str(filename)
    if not os.path.isfile(path_str):
        raise FileNotFoundError(f"File not found: {path_str}")

    size_mb = os.path.getsize(path_str) / (1024 ** 2)
    lines: List[str] = [f"{path_str}  ({size_mb:.2f} MB)"]

    with h5py.File(path_str, "r") as h5:
        if show_attrs and len(h5.attrs) > 0:
            for k, v in h5.attrs.items():
                lines.append(f"    @{k:<26} {_clean_h5_attr(v)}")

        def _visitor(name: str, obj: Any) -> None:
            depth = name.count("/")
            pad = "  " * (depth + 1)
            bname = name.split("/")[-1]
            if isinstance(obj, h5py.Group):
                lines.append(f"{pad}{bname}/")
            else:
                lines.append(f"{pad}{bname:<20} {str(obj.shape):<18} {obj.dtype}")
            if show_attrs and len(obj.attrs) > 0:
                for k, v in obj.attrs.items():
                    lines.append(f"{pad}    @{k:<22} {_clean_h5_attr(v)}")

        h5.visititems(_visitor)

    summary = "\n".join(lines)
    if print_output:
        print(summary)
    return summary


def extract_velocity_1d_from_devito(
    filename: Union[str, Path],
    *,
    x_index: Optional[int] = None,
    resample_to: Optional[int] = None,
    normalize: bool = False,
    unit_target: Optional[str] = None,
    source_units: str = "auto",
    bc: str = "dirichlet",
) -> VelocityModel:
    """
    Extract a 1D vertical depth velocity profile from a devito-seismic-hdf5 file.

    Parameters:
        filename: Path to the HDF5 file.
        x_index: Horizontal grid index to extract profile from. Defaults to source x
                 or midpoint Nx // 2.
        resample_to: Optional power-of-2 grid size to resample onto.
        normalize: Whether to normalize units.
        unit_target: Target unit system ('si', 'kms', 'dimensionless').
        source_units: Source units of input ('auto', 'kms', 'si').
        bc: Boundary condition for VelocityModel.
    """
    with h5py.File(str(filename), "r") as h5:
        if "model" not in h5 or "vp" not in h5["model"]:
            raise ValueError(f"HDF5 file '{filename}' does not contain model/vp dataset.")
        vp = np.array(h5["model"]["vp"])
        spacing = h5["model"].attrs.get("spacing", (10.0, 10.0))
        origin = h5["model"].attrs.get("origin", (0.0, 0.0))

        # Check geometry for source position if x_index is None
        if x_index is None and "geometry" in h5:
            for k in h5["geometry"].keys():
                if "src" in k and "coordinates" in h5["geometry"][k]:
                    src_coords = np.array(h5["geometry"][k]["coordinates"])
                    if src_coords.ndim >= 2 and src_coords.shape[1] >= 1:
                        x_phys = src_coords[0, 0]
                        org_x = origin[0] if hasattr(origin, "__len__") else 0.0
                        spc_x = spacing[0] if hasattr(spacing, "__len__") else 10.0
                        x_index = int(round((x_phys - org_x) / spc_x))
                        break

    if vp.ndim == 2:
        nx, nz = vp.shape
        if x_index is None:
            x_index = nx // 2
        x_index = max(0, min(nx - 1, x_index))
        v_profile = vp[x_index, :].astype(float)
        hz = float(spacing[1]) if hasattr(spacing, "__len__") and len(spacing) > 1 else float(spacing[0])
    elif vp.ndim == 1:
        v_profile = vp.astype(float)
        hz = float(spacing[0]) if hasattr(spacing, "__len__") else float(spacing)
    else:
        raise ValueError(f"Unsupported vp ndim: {vp.ndim}")

    params = compute_normalization_params(
        v_profile,
        h=hz,
        normalize=normalize,
        unit_target=unit_target,
        source_units=source_units,
    )
    v_norm, h_norm, _ = apply_seismic_normalization(v_profile, hz, params=params)

    meta = {
        "normalization": params.to_dict(),
        "source_path": str(filename),
        "x_index": x_index,
        "original_n_points": len(v_profile),
    }

    model = VelocityModel(velocity=v_norm, h=h_norm, bc=bc, metadata=meta)
    if resample_to is not None:
        model = model.resample(resample_to)
    return model


def extract_velocity_2d_from_devito(
    filename: Union[str, Path],
    *,
    resample_shape: Optional[Tuple[int, int]] = None,
    smooth_sigma: float = 0.0,
    normalize: bool = False,
    unit_target: Optional[str] = None,
    source_units: str = "auto",
    bc: str = "dirichlet",
) -> Any:
    """
    Extract a 2D velocity grid from a devito-seismic-hdf5 file.

    Parameters:
        filename: Path to the HDF5 file.
        resample_shape: Optional (target_nx, target_nz) grid shape.
        smooth_sigma: Optional Gaussian smoothing parameter before resampling.
        normalize: Whether to apply explicit normalization.
        unit_target: Target unit system ('si', 'kms', 'dimensionless').
        source_units: Source units of input ('auto', 'kms', 'si').
        bc: Boundary condition.
    """
    with h5py.File(str(filename), "r") as h5:
        if "model" not in h5 or "vp" not in h5["model"]:
            raise ValueError(f"HDF5 file '{filename}' does not contain model/vp dataset.")
        vp = np.array(h5["model"]["vp"], dtype=float)
        spacing = h5["model"].attrs.get("spacing", (10.0, 10.0))

    if vp.ndim != 2:
        raise ValueError(f"extract_velocity_2d_from_devito requires 2D model/vp, got shape {vp.shape}")

    hx = float(spacing[0])
    hz = float(spacing[1]) if len(spacing) > 1 else hx

    if smooth_sigma > 0.0:
        vp = gaussian_filter(vp, sigma=smooth_sigma, mode="nearest")

    orig_shape = vp.shape
    if resample_shape is not None:
        target_nx, target_nz = resample_shape
        orig_nx, orig_nz = orig_shape
        x_old = np.linspace(0.0, 1.0, orig_nx)
        z_old = np.linspace(0.0, 1.0, orig_nz)
        interp = RegularGridInterpolator((x_old, z_old), vp, bounds_error=False, fill_value=None)

        x_new = np.linspace(0.0, 1.0, target_nx)
        z_new = np.linspace(0.0, 1.0, target_nz)
        X_mesh, Z_mesh = np.meshgrid(x_new, z_new, indexing="ij")
        pts = np.stack([X_mesh.ravel(), Z_mesh.ravel()], axis=-1)
        vp = interp(pts).reshape((target_nx, target_nz))

        hx = hx * (orig_nx - 1) / (target_nx - 1) if target_nx > 1 else hx
        hz = hz * (orig_nz - 1) / (target_nz - 1) if target_nz > 1 else hz

    params = compute_normalization_params(
        vp,
        h=(hx + hz) / 2.0,
        normalize=normalize,
        unit_target=unit_target,
        source_units=source_units,
    )
    vp_norm = np.asarray(vp) * params.velocity_scale
    hx_norm = float(hx * params.length_scale)
    hz_norm = float(hz * params.length_scale)

    meta = {
        "normalization": params.to_dict(),
        "source_path": str(filename),
        "original_shape": orig_shape,
    }

    if VelocityModel2D is not None:
        return VelocityModel2D(velocity=vp_norm, hx=hx_norm, hz=hz_norm, bc=bc, metadata=meta)

    from dataclasses import dataclass, field
    @dataclass
    class _VelocityModel2DFallback:
        velocity: np.ndarray
        hx: float = 1.0
        hz: float = 1.0
        bc: str = "dirichlet"
        metadata: Dict[str, Any] = field(default_factory=dict)

        @property
        def slowness_squared(self) -> np.ndarray:
            return 1.0 / (self.velocity ** 2)

        @property
        def flat_velocity(self) -> np.ndarray:
            return self.velocity.reshape(-1)

        @property
        def shape(self) -> Tuple[int, ...]:
            return self.velocity.shape

    return _VelocityModel2DFallback(velocity=vp_norm, hx=hx_norm, hz=hz_norm, bc=bc, metadata=meta)


def extract_frequency_data_from_devito(
    filename: Union[str, Path],
    frequencies_hz: Sequence[float],
    *,
    deconvolve_source: bool = True,
    reg_eps: float = 1e-6,
) -> Dict[str, Any]:
    """
    Extract frequency-domain complex wavefield spectra from time-domain traces in
    geometry/rec/data and geometry/src/data.
    """
    with h5py.File(str(filename), "r") as h5:
        if "geometry" not in h5:
            raise ValueError(f"HDF5 file '{filename}' does not contain geometry group.")
        rec_grp = None
        src_grp = None
        for k in h5["geometry"].keys():
            if "rec" in k and "data" in h5["geometry"][k]:
                rec_grp = h5["geometry"][k]
            elif "src" in k and "data" in h5["geometry"][k]:
                src_grp = h5["geometry"][k]

        if rec_grp is None:
            raise ValueError(f"No receiver data found in '{filename}/geometry'.")

        rec_data = np.array(rec_grp["data"])  # (nt, nrec)
        src_data = np.array(src_grp["data"]) if src_grp is not None else None

        dt_ms = 1.0
        if "time" in h5:
            dt_ms = float(h5["time"].attrs.get("dt", 1.0))
        dt_s = dt_ms * 1e-3

    nt = rec_data.shape[0]
    fft_freqs = np.fft.rfftfreq(nt, d=dt_s)
    rec_fft = np.fft.rfft(rec_data, axis=0) * dt_s  # (n_fft, nrec)

    if deconvolve_source and src_data is not None:
        src_fft = np.fft.rfft(src_data, axis=0) * dt_s  # (n_fft, nsrc)
        src_spec = src_fft[:, 0] if src_fft.ndim > 1 else src_fft
        pow_s = np.abs(src_spec) ** 2
        max_p = np.max(pow_s) if np.max(pow_s) > 0 else 1.0
        deconv = np.conj(src_spec) / (pow_s + reg_eps * max_p)
        rec_fft = rec_fft * deconv[:, None]

    interp_r = interp1d(fft_freqs, np.real(rec_fft), axis=0, fill_value="extrapolate")
    interp_i = interp1d(fft_freqs, np.imag(rec_fft), axis=0, fill_value="extrapolate")
    data_freqs = interp_r(frequencies_hz) + 1j * interp_i(frequencies_hz)

    return {
        "frequencies_hz": np.asarray(frequencies_hz, dtype=float),
        "omegas": 2.0 * np.pi * np.asarray(frequencies_hz, dtype=float),
        "data": data_freqs,
        "dt": dt_s,
        "fft_freqs": fft_freqs,
    }


def build_fwi_dataset_from_devito(
    filename: Union[str, Path],
    *,
    mode: str = "1d",
    grid_shape: Union[int, Tuple[int, int]] = 4,
    data_source: str = "helmholtz_consistent",
    frequencies_hz: Optional[Sequence[float]] = None,
    normalize: bool = False,
    unit_target: Optional[str] = None,
    source_units: str = "auto",
    bc: str = "dirichlet",
    initial_smoothing_sigma: float = 0.0,
    v_init_factor: Optional[float] = None,
) -> Union[FWIDataset, Any]:
    """
    Construct a ready-to-invert FWIDataset (or 2D FWIDataset) from a devito-seismic-hdf5 file.

    Parameters:
        filename: Path to the HDF5 file.
        mode: Inversion mode, either "1d" or "2d".
        grid_shape: Target power-of-2 discrete grid points (int for 1D, (nx, nz) for 2D).
        data_source: Either "helmholtz_consistent" (simulates exact coarse Helmholtz data
                     avoiding dispersion crime) or "devito_fft" (interpolates FFT of time traces).
        frequencies_hz: Survey inversion frequencies in Hz.
        normalize: Whether to apply explicit normalization.
        unit_target: Target unit system ('si', 'kms', 'dimensionless').
        source_units: Input units ('auto', 'kms', 'si').
        bc: Boundary condition.
        initial_smoothing_sigma: Optional Gaussian smoothing sigma for initial velocity.
        v_init_factor: Optional scaling factor to construct initial velocity.
    """
    freqs = list(frequencies_hz) if frequencies_hz is not None else [5.0]

    if mode == "1d":
        n_points = grid_shape if isinstance(grid_shape, int) else grid_shape[0]
        v_true_model = extract_velocity_1d_from_devito(
            filename,
            resample_to=n_points,
            normalize=normalize,
            unit_target=unit_target,
            source_units=source_units,
            bc=bc,
        )
        true_v = v_true_model.velocity
        h = v_true_model.h

        # Build initial velocity guess
        if initial_smoothing_sigma > 0.0:
            from scipy.ndimage import gaussian_filter1d
            init_v = gaussian_filter1d(true_v, sigma=initial_smoothing_sigma, mode="nearest")
        elif v_init_factor is not None:
            init_v = true_v * v_init_factor
        else:
            # Default to top layer velocity
            init_v = np.full_like(true_v, true_v[0])

        init_model = VelocityModel(velocity=init_v, h=h, bc=bc, metadata=dict(v_true_model.metadata))

        # Determine frequency scale
        norm_info = v_true_model.metadata.get("normalization", {})
        omega_scale = float(norm_info.get("omega_scale", 1.0))
        omegas = [2.0 * np.pi * f * omega_scale for f in freqs]

        # Geometry discovery
        sources = [0]
        receivers = list(range(n_points))
        with h5py.File(str(filename), "r") as h5:
            if "geometry" in h5:
                for k in h5["geometry"].keys():
                    if "src" in k and "coordinates" in h5["geometry"][k]:
                        s_coords = np.array(h5["geometry"][k]["coordinates"])
                        if s_coords.ndim >= 2:
                            org_z = h5["model"].attrs.get("origin", (0.0, 0.0))[-1] if "model" in h5 else 0.0
                            length_scale = norm_info.get("length_scale", 1.0)
                            phys_h = h / max(length_scale, 1e-12)
                            s_idx = [int(round((c[-1] - org_z) / phys_h)) for c in s_coords]
                            sources = [max(0, min(n_points - 1, idx)) for idx in s_idx] or [0]
                    elif "rec" in k and "coordinates" in h5["geometry"][k]:
                        r_coords = np.array(h5["geometry"][k]["coordinates"])
                        if r_coords.ndim >= 2:
                            org_z = h5["model"].attrs.get("origin", (0.0, 0.0))[-1] if "model" in h5 else 0.0
                            length_scale = norm_info.get("length_scale", 1.0)
                            phys_h = h / max(length_scale, 1e-12)
                            r_idx = [int(round((c[-1] - org_z) / phys_h)) for c in r_coords]
                            receivers = [max(0, min(n_points - 1, idx)) for idx in r_idx] or list(range(n_points))

        acq = Acquisition(sources=sources, receivers=receivers, omegas=omegas)

        if data_source == "helmholtz_consistent":
            from qpdesolver.applications.fwi.problem import FWIProblem
            prob = FWIProblem(acquisition=acq, n_points=n_points, h=h, bc=bc)
            ds = prob.simulate_dataset(true_model=v_true_model, initial_model=init_model)
            ds.metadata["normalization"] = norm_info
            ds.metadata["data_source"] = data_source
            return ds

        elif data_source == "devito_fft":
            fdata = extract_frequency_data_from_devito(filename, frequencies_hz=freqs)
            f_data = fdata["data"]  # (n_freqs, n_rec)
            # Match acquisition shape (n_sources, n_omegas, n_receivers)
            n_s = len(sources)
            n_w = len(omegas)
            n_r = len(receivers)
            obs = np.zeros((n_s, n_w, n_r), dtype=complex)
            for s_i in range(n_s):
                for w_i in range(n_w):
                    obs[s_i, w_i, :] = f_data[w_i, :n_r]

            return FWIDataset(
                acquisition=acq,
                observed_data=obs,
                h=h,
                bc=bc,
                true_velocity=true_v,
                initial_velocity=init_v,
                metadata={"normalization": norm_info, "data_source": data_source},
            )
        else:
            raise ValueError(f"Unknown data_source '{data_source}'. Must be 'helmholtz_consistent' or 'devito_fft'.")

    elif mode == "2d":
        target_shape = grid_shape if isinstance(grid_shape, tuple) else (grid_shape, grid_shape)
        v2d = extract_velocity_2d_from_devito(
            filename,
            resample_shape=target_shape,
            normalize=normalize,
            unit_target=unit_target,
            source_units=source_units,
            bc=bc,
        )
        true_v = v2d.velocity
        hx = v2d.hx
        hz = v2d.hz
        norm_info = v2d.metadata.get("normalization", {})
        omega_scale = float(norm_info.get("omega_scale", 1.0))
        omegas = [2.0 * np.pi * f * omega_scale for f in freqs]

        if initial_smoothing_sigma > 0.0:
            init_v = gaussian_filter(true_v, sigma=initial_smoothing_sigma, mode="nearest")
        elif v_init_factor is not None:
            init_v = true_v * v_init_factor
        else:
            init_v = np.full_like(true_v, true_v[0, 0])

        sources_2d = [(0, 0)]
        receivers_2d = [(ix, 0) for ix in range(target_shape[0])]

        try:
            from qpdesolver.applications.fwi.model import Acquisition2D, FWIDataset2D
            acq2d = Acquisition2D(sources=sources_2d, receivers=receivers_2d, omegas=omegas)
            n_s = len(sources_2d)
            n_w = len(omegas)
            n_r = len(receivers_2d)
            obs = np.zeros((n_s, n_w, n_r), dtype=complex)
            return FWIDataset2D(
                acquisition=acq2d,
                observed_data=obs,
                hx=hx,
                hz=hz,
                bc=bc,
                true_velocity=true_v,
                initial_velocity=init_v,
                metadata={"normalization": norm_info, "data_source": data_source},
            )
        except (ImportError, AttributeError):
            from dataclasses import dataclass, field
            @dataclass
            class _Acq2DFallback:
                sources: List[Any]
                receivers: List[Any]
                omegas: List[float]
            @dataclass
            class _FWIDataset2DFallback:
                acquisition: _Acq2DFallback
                observed_data: np.ndarray
                hx: float = 1.0
                hz: float = 1.0
                bc: str = "dirichlet"
                true_velocity: Optional[np.ndarray] = None
                initial_velocity: Optional[np.ndarray] = None
                metadata: Dict[str, Any] = field(default_factory=dict)

            obs = np.zeros((len(sources_2d), len(omegas), len(receivers_2d)), dtype=complex)
            return _FWIDataset2DFallback(
                acquisition=_Acq2DFallback(sources=sources_2d, receivers=receivers_2d, omegas=omegas),
                observed_data=obs,
                hx=hx,
                hz=hz,
                bc=bc,
                true_velocity=true_v,
                initial_velocity=init_v,
                metadata={"normalization": norm_info, "data_source": data_source},
            )
    else:
        raise ValueError(f"Unsupported mode: '{mode}'. Must be '1d' or '2d'.")
