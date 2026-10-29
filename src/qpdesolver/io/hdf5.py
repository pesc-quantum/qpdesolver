"""
HDF5 I/O module for qpdesolver.
Provides robust read/write for LinearSystem, VelocityModel, FWIDataset, and nested study results.
"""
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import scipy.sparse as sp
import h5py

from qpdesolver.io.schema import H5SchemaError, SCHEMA_VERSION
from qpdesolver.io.normalization import (
    compute_normalization_params,
    apply_seismic_normalization,
    NormalizationParams,
)
from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.applications.fwi.model import VelocityModel, FWIDataset, Acquisition


def _ensure_group_writable(h5file: h5py.File, group_name: str, overwrite: bool) -> h5py.Group:
    """Helper to check or delete existing group based on overwrite flag."""
    if group_name in h5file:
        if not overwrite:
            raise H5SchemaError(f"Group '{group_name}' already exists in '{h5file.filename}' and overwrite is False.")
        del h5file[group_name]
    return h5file.create_group(group_name)


def _write_dataset(
    grp: h5py.Group,
    name: str,
    data: np.ndarray,
    *,
    compression: Optional[str] = None,
    compression_opts: Optional[int] = None,
    dtype: Optional[str] = None,
    chunks: Optional[Union[bool, Tuple[int, ...]]] = None,
    shuffle: bool = False,
    attrs: Optional[Dict[str, Any]] = None,
) -> h5py.Dataset:
    """Helper to write datasets with optional chunking, compression, and attribute tags."""
    arr = np.asarray(data)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    kw: Dict[str, Any] = {}
    if compression and arr.ndim > 0 and arr.size > 0:
        kw["compression"] = compression
        if compression == "gzip":
            if compression_opts is not None:
                kw["compression_opts"] = compression_opts
            if shuffle:
                kw["shuffle"] = True
        if chunks is not None:
            kw["chunks"] = chunks

    ds = grp.create_dataset(name, data=arr, **kw)
    if attrs:
        for k, v in attrs.items():
            if v is not None:
                ds.attrs[k] = v
    return ds


def list_groups(path: Union[str, Path], qpdesolver_type: Optional[str] = None) -> List[str]:
    """Lists group names in the HDF5 file, optionally filtered by qpdesolver_type attribute."""
    matched = []
    with h5py.File(path, "r") as f:
        for k in f.keys():
            grp = f[k]
            if isinstance(grp, h5py.Group):
                gtype = grp.attrs.get("qpdesolver_type", None)
                if qpdesolver_type is None or gtype == qpdesolver_type:
                    matched.append(k)
    return sorted(matched)


def save_linear_system(
    path: Union[str, Path],
    system: LinearSystem,
    group: str,
    *,
    sparse: bool = False,
    overwrite: bool = False,
    compression: Optional[str] = None,
    compression_opts: Optional[int] = None,
    dtype: Optional[str] = None,
) -> None:
    """Saves a LinearSystem to an HDF5 group with optional compression."""
    with h5py.File(path, "a") as f:
        grp = _ensure_group_writable(f, group, overwrite)
        grp.attrs["qpdesolver_type"] = "linear_system"
        grp.attrs["schema_version"] = SCHEMA_VERSION
        grp.attrs["name"] = system.name
        grp.attrs["metadata_json"] = json.dumps(system.metadata, default=str)

        _write_dataset(grp, "b", system.b, compression=compression, compression_opts=compression_opts, dtype=dtype)

        if sparse:
            csr = sp.csr_matrix(system.A)
            grp_a = grp.create_group("A")
            grp_a.attrs["format"] = "csr"
            grp_a.attrs["shape"] = csr.shape
            _write_dataset(grp_a, "data", csr.data, compression=compression, compression_opts=compression_opts, dtype=dtype)
            _write_dataset(grp_a, "indices", csr.indices, compression=compression, compression_opts=compression_opts)
            _write_dataset(grp_a, "indptr", csr.indptr, compression=compression, compression_opts=compression_opts)
        else:
            _write_dataset(grp, "A", system.A, compression=compression, compression_opts=compression_opts, dtype=dtype)


def load_linear_system(path: Union[str, Path], group: str) -> LinearSystem:
    """Loads and validates a LinearSystem from an HDF5 group."""
    with h5py.File(path, "r") as f:
        if group not in f:
            raise H5SchemaError(f"Group '{group}' not found in '{path}'.")
        grp = f[group]
        gtype = grp.attrs.get("qpdesolver_type", "")
        if gtype != "linear_system":
            raise H5SchemaError(f"Group '{group}' has invalid qpdesolver_type '{gtype}', expected 'linear_system'.")

        if "b" not in grp:
            raise H5SchemaError(f"Group '{group}' is missing dataset 'b'.")
        b = np.array(grp["b"])

        if "A" not in grp:
            raise H5SchemaError(f"Group '{group}' is missing 'A'.")

        if isinstance(grp["A"], h5py.Group):
            # Sparse CSR representation
            grp_a = grp["A"]
            fmt = grp_a.attrs.get("format", "")
            if fmt != "csr":
                raise H5SchemaError(f"Unsupported sparse format '{fmt}' in group '{group}/A'.")
            shape = tuple(grp_a.attrs["shape"])
            data = np.array(grp_a["data"])
            indices = np.array(grp_a["indices"])
            indptr = np.array(grp_a["indptr"])
            A = sp.csr_matrix((data, indices, indptr), shape=shape).toarray()
        else:
            A = np.array(grp["A"])

        name = grp.attrs.get("name", group)
        meta_json = grp.attrs.get("metadata_json", "{}")
        try:
            metadata = json.loads(meta_json)
        except Exception:
            metadata = {}

        try:
            return LinearSystem(A=A, b=b, name=name, metadata=metadata)
        except ValueError as e:
            raise H5SchemaError(f"Validation failed for system in '{group}': {str(e)}") from e


def save_velocity_model(
    path: Union[str, Path],
    model: VelocityModel,
    group: str,
    *,
    overwrite: bool = False,
    compression: Optional[str] = None,
    compression_opts: Optional[int] = None,
    dtype: Optional[str] = None,
) -> None:
    """Saves a VelocityModel to an HDF5 group."""
    with h5py.File(path, "a") as f:
        grp = _ensure_group_writable(f, group, overwrite)
        grp.attrs["qpdesolver_type"] = "velocity_model"
        grp.attrs["schema_version"] = SCHEMA_VERSION
        grp.attrs["h"] = float(model.h)
        grp.attrs["bc"] = model.bc
        grp.attrs["metadata_json"] = json.dumps(model.metadata, default=str)
        _write_dataset(
            grp,
            "velocity",
            model.velocity,
            compression=compression,
            compression_opts=compression_opts,
            dtype=dtype,
        )


def load_velocity_model(
    path: Union[str, Path],
    group: str,
    *,
    resample_to: Optional[int] = None,
    normalize: bool = False,
    unit_target: Optional[str] = None,
    source_units: str = "auto",
) -> VelocityModel:
    """Loads a VelocityModel, optionally resampling it to a specified number of grid points and normalizing."""
    with h5py.File(path, "r") as f:
        if group not in f:
            raise H5SchemaError(f"Group '{group}' not found in '{path}'.")
        grp = f[group]
        if grp.attrs.get("qpdesolver_type", "") != "velocity_model":
            raise H5SchemaError(f"Group '{group}' is not a velocity_model.")
        if "velocity" not in grp:
            raise H5SchemaError(f"Group '{group}' is missing dataset 'velocity'.")

        v = np.array(grp["velocity"])
        h = float(grp.attrs.get("h", 1.0))
        bc = grp.attrs.get("bc", "dirichlet")
        meta = json.loads(grp.attrs.get("metadata_json", "{}"))

    if not normalize:
        if unit_target is not None:
            raise ValueError("unit_target requires normalize=True")
        params = compute_normalization_params(v, h=h, normalize=False)
        meta["normalization"] = params.to_dict()
    else:
        params = compute_normalization_params(v, h=h, normalize=True, unit_target=unit_target, source_units=source_units)
        v, h, _ = apply_seismic_normalization(v, h, params=params)
        meta["normalization"] = params.to_dict()

    model = VelocityModel(velocity=v, h=h, bc=bc, metadata=meta)
    if resample_to is not None:
        model = model.resample(resample_to)
    return model


def save_velocity_model_2d(
    path: Union[str, Path],
    model: Any,
    group: str,
    *,
    overwrite: bool = False,
    compression: Optional[str] = None,
    compression_opts: Optional[int] = None,
    dtype: Optional[str] = None,
) -> None:
    """Saves a 2D VelocityModel to an HDF5 group."""
    with h5py.File(path, "a") as f:
        grp = _ensure_group_writable(f, group, overwrite)
        grp.attrs["qpdesolver_type"] = "velocity_model_2d"
        grp.attrs["schema_version"] = SCHEMA_VERSION
        hx = getattr(model, "hx", getattr(model, "h", 1.0))
        hz = getattr(model, "hz", hx)
        grp.attrs["hx"] = float(hx)
        grp.attrs["hz"] = float(hz)
        grp.attrs["bc"] = getattr(model, "bc", "dirichlet")
        grp.attrs["metadata_json"] = json.dumps(getattr(model, "metadata", {}), default=str)
        _write_dataset(
            grp,
            "velocity",
            np.asarray(model.velocity),
            compression=compression,
            compression_opts=compression_opts,
            dtype=dtype,
        )


def load_velocity_model_2d(
    path: Union[str, Path],
    group: str,
    *,
    resample_shape: Optional[Tuple[int, int]] = None,
    normalize: bool = False,
    unit_target: Optional[str] = None,
    source_units: str = "auto",
) -> Any:
    """Loads a 2D VelocityModel, optionally resampling and normalizing."""
    with h5py.File(path, "r") as f:
        if group not in f:
            raise H5SchemaError(f"Group '{group}' not found in '{path}'.")
        grp = f[group]
        if grp.attrs.get("qpdesolver_type", "") != "velocity_model_2d":
            raise H5SchemaError(f"Group '{group}' is not a velocity_model_2d.")
        if "velocity" not in grp:
            raise H5SchemaError(f"Group '{group}' is missing dataset 'velocity'.")

        v = np.array(grp["velocity"])
        hx = float(grp.attrs.get("hx", grp.attrs.get("h", 1.0)))
        hz = float(grp.attrs.get("hz", hx))
        bc = grp.attrs.get("bc", "dirichlet")
        meta = json.loads(grp.attrs.get("metadata_json", "{}"))

    if not normalize:
        if unit_target is not None:
            raise ValueError("unit_target requires normalize=True")
        params = compute_normalization_params(v, h=(hx + hz) / 2.0, normalize=False)
        meta["normalization"] = params.to_dict()
    else:
        params = compute_normalization_params(v, h=(hx + hz) / 2.0, normalize=True, unit_target=unit_target, source_units=source_units)
        v = np.asarray(v) * params.velocity_scale
        hx = float(hx * params.length_scale)
        hz = float(hz * params.length_scale)
        meta["normalization"] = params.to_dict()

    if resample_shape is not None:
        from scipy.interpolate import RegularGridInterpolator
        target_nx, target_nz = resample_shape
        orig_nx, orig_nz = v.shape
        x_old = np.linspace(0.0, 1.0, orig_nx)
        z_old = np.linspace(0.0, 1.0, orig_nz)
        interp = RegularGridInterpolator((x_old, z_old), v, bounds_error=False, fill_value=None)
        x_new = np.linspace(0.0, 1.0, target_nx)
        z_new = np.linspace(0.0, 1.0, target_nz)
        X_mesh, Z_mesh = np.meshgrid(x_new, z_new, indexing="ij")
        pts = np.stack([X_mesh.ravel(), Z_mesh.ravel()], axis=-1)
        v = interp(pts).reshape((target_nx, target_nz))
        hx = hx * (orig_nx - 1) / (target_nx - 1) if target_nx > 1 else hx
        hz = hz * (orig_nz - 1) / (target_nz - 1) if target_nz > 1 else hz
        meta["resampled_from"] = (orig_nx, orig_nz)

    try:
        from qpdesolver.applications.fwi.model import VelocityModel2D
        return VelocityModel2D(velocity=v, hx=hx, hz=hz, bc=bc, metadata=meta)
    except (ImportError, AttributeError):
        from dataclasses import dataclass, field
        @dataclass
        class _VelocityModel2DContainer:
            velocity: np.ndarray
            hx: float = 1.0
            hz: float = 1.0
            bc: str = "dirichlet"
            metadata: Dict[str, Any] = field(default_factory=dict)
        return _VelocityModel2DContainer(velocity=v, hx=hx, hz=hz, bc=bc, metadata=meta)


def save_fwi_dataset(
    path: Union[str, Path],
    dataset: FWIDataset,
    group: str,
    *,
    overwrite: bool = False,
    compression: Optional[str] = None,
    compression_opts: Optional[int] = None,
    dtype: Optional[str] = None,
) -> None:
    """Saves an FWIDataset to an HDF5 group."""
    with h5py.File(path, "a") as f:
        grp = _ensure_group_writable(f, group, overwrite)
        grp.attrs["qpdesolver_type"] = "fwi_dataset"
        grp.attrs["schema_version"] = SCHEMA_VERSION
        grp.attrs["h"] = float(dataset.h)
        grp.attrs["bc"] = dataset.bc
        grp.attrs["metadata_json"] = json.dumps(dataset.metadata, default=str)

        acq_grp = grp.create_group("acquisition")
        acq_grp.create_dataset("sources", data=np.array(dataset.acquisition.sources, dtype=int))
        acq_grp.create_dataset("receivers", data=np.array(dataset.acquisition.receivers, dtype=int))
        acq_grp.create_dataset("omegas", data=np.array(dataset.acquisition.omegas, dtype=float))

        _write_dataset(
            grp,
            "observed_data",
            dataset.observed_data,
            compression=compression,
            compression_opts=compression_opts,
            dtype=dtype,
        )
        if dataset.true_velocity is not None:
            _write_dataset(
                grp,
                "true_velocity",
                dataset.true_velocity,
                compression=compression,
                compression_opts=compression_opts,
                dtype=dtype,
            )
        if dataset.initial_velocity is not None:
            _write_dataset(
                grp,
                "initial_velocity",
                dataset.initial_velocity,
                compression=compression,
                compression_opts=compression_opts,
                dtype=dtype,
            )


def load_fwi_dataset(
    path: Union[str, Path],
    group: str,
    *,
    normalize: bool = False,
    unit_target: Optional[str] = None,
    source_units: str = "auto",
) -> FWIDataset:
    """Loads an FWIDataset from an HDF5 group, optionally normalizing."""
    with h5py.File(path, "r") as f:
        if group not in f:
            raise H5SchemaError(f"Group '{group}' not found in '{path}'.")
        grp = f[group]
        if grp.attrs.get("qpdesolver_type", "") != "fwi_dataset":
            raise H5SchemaError(f"Group '{group}' is not an fwi_dataset.")

        if "acquisition" not in grp or "observed_data" not in grp:
            raise H5SchemaError(f"Group '{group}' missing required acquisition or observed_data.")

        acq_grp = grp["acquisition"]
        sources = list(np.array(acq_grp["sources"]))
        receivers = list(np.array(acq_grp["receivers"]))
        omegas = list(np.array(acq_grp["omegas"]))

        obs = np.array(grp["observed_data"])
        h = float(grp.attrs.get("h", 1.0))
        bc = grp.attrs.get("bc", "dirichlet")
        meta = json.loads(grp.attrs.get("metadata_json", "{}"))

        true_v = np.array(grp["true_velocity"]) if "true_velocity" in grp else None
        init_v = np.array(grp["initial_velocity"]) if "initial_velocity" in grp else None

    if not normalize:
        if unit_target is not None:
            raise ValueError("unit_target requires normalize=True")
        ref_v = true_v if true_v is not None else (init_v if init_v is not None else 1.0)
        params = compute_normalization_params(ref_v, h=h, normalize=False)
        meta["normalization"] = params.to_dict()
    else:
        ref_v = init_v if init_v is not None else (true_v if true_v is not None else 1.0)
        params = compute_normalization_params(ref_v, h=h, normalize=True, unit_target=unit_target, source_units=source_units)
        if true_v is not None:
            true_v, _, _ = apply_seismic_normalization(true_v, h, params=params)
        if init_v is not None:
            init_v, _, _ = apply_seismic_normalization(init_v, h, params=params)
        h = float(h * params.length_scale)
        omegas = [float(params.scale_omega(w)) for w in omegas]
        meta["normalization"] = params.to_dict()

    acq = Acquisition(sources=sources, receivers=receivers, omegas=omegas)
    return FWIDataset(
        acquisition=acq,
        observed_data=obs,
        h=h,
        bc=bc,
        true_velocity=true_v,
        initial_velocity=init_v,
        metadata=meta,
    )


def save_fwi_dataset_2d(
    path: Union[str, Path],
    dataset: Any,
    group: str,
    *,
    overwrite: bool = False,
    compression: Optional[str] = None,
    compression_opts: Optional[int] = None,
    dtype: Optional[str] = None,
) -> None:
    """Saves a 2D FWIDataset to an HDF5 group."""
    with h5py.File(path, "a") as f:
        grp = _ensure_group_writable(f, group, overwrite)
        grp.attrs["qpdesolver_type"] = "fwi_dataset_2d"
        grp.attrs["schema_version"] = SCHEMA_VERSION
        hx = getattr(dataset, "hx", getattr(dataset, "h", 1.0))
        hz = getattr(dataset, "hz", hx)
        grp.attrs["hx"] = float(hx)
        grp.attrs["hz"] = float(hz)
        grp.attrs["bc"] = getattr(dataset, "bc", "dirichlet")
        grp.attrs["metadata_json"] = json.dumps(getattr(dataset, "metadata", {}), default=str)

        acq_grp = grp.create_group("acquisition")
        acq_grp.create_dataset("sources", data=np.array(dataset.acquisition.sources))
        acq_grp.create_dataset("receivers", data=np.array(dataset.acquisition.receivers))
        acq_grp.create_dataset("omegas", data=np.array(dataset.acquisition.omegas, dtype=float))

        _write_dataset(
            grp,
            "observed_data",
            dataset.observed_data,
            compression=compression,
            compression_opts=compression_opts,
            dtype=dtype,
        )
        if getattr(dataset, "true_velocity", None) is not None:
            _write_dataset(
                grp,
                "true_velocity",
                dataset.true_velocity,
                compression=compression,
                compression_opts=compression_opts,
                dtype=dtype,
            )
        if getattr(dataset, "initial_velocity", None) is not None:
            _write_dataset(
                grp,
                "initial_velocity",
                dataset.initial_velocity,
                compression=compression,
                compression_opts=compression_opts,
                dtype=dtype,
            )


def load_fwi_dataset_2d(
    path: Union[str, Path],
    group: str,
    *,
    normalize: bool = False,
    unit_target: Optional[str] = None,
    source_units: str = "auto",
) -> Any:
    """Loads a 2D FWIDataset from an HDF5 group."""
    with h5py.File(path, "r") as f:
        if group not in f:
            raise H5SchemaError(f"Group '{group}' not found in '{path}'.")
        grp = f[group]
        if grp.attrs.get("qpdesolver_type", "") != "fwi_dataset_2d":
            raise H5SchemaError(f"Group '{group}' is not an fwi_dataset_2d.")

        if "acquisition" not in grp or "observed_data" not in grp:
            raise H5SchemaError(f"Group '{group}' missing required acquisition or observed_data.")

        acq_grp = grp["acquisition"]
        sources = [tuple(s) if hasattr(s, "__len__") else int(s) for s in np.array(acq_grp["sources"])]
        receivers = [tuple(r) if hasattr(r, "__len__") else int(r) for r in np.array(acq_grp["receivers"])]
        omegas = list(np.array(acq_grp["omegas"]))

        obs = np.array(grp["observed_data"])
        hx = float(grp.attrs.get("hx", grp.attrs.get("h", 1.0)))
        hz = float(grp.attrs.get("hz", hx))
        bc = grp.attrs.get("bc", "dirichlet")
        meta = json.loads(grp.attrs.get("metadata_json", "{}"))

        true_v = np.array(grp["true_velocity"]) if "true_velocity" in grp else None
        init_v = np.array(grp["initial_velocity"]) if "initial_velocity" in grp else None

    if not normalize:
        if unit_target is not None:
            raise ValueError("unit_target requires normalize=True")
        ref_v = true_v if true_v is not None else (init_v if init_v is not None else 1.0)
        params = compute_normalization_params(ref_v, h=(hx + hz) / 2.0, normalize=False)
        meta["normalization"] = params.to_dict()
    else:
        ref_v = init_v if init_v is not None else (true_v if true_v is not None else 1.0)
        params = compute_normalization_params(ref_v, h=(hx + hz) / 2.0, normalize=True, unit_target=unit_target, source_units=source_units)
        if true_v is not None:
            true_v = np.asarray(true_v) * params.velocity_scale
        if init_v is not None:
            init_v = np.asarray(init_v) * params.velocity_scale
        hx = float(hx * params.length_scale)
        hz = float(hz * params.length_scale)
        omegas = [float(params.scale_omega(w)) for w in omegas]
        meta["normalization"] = params.to_dict()

    try:
        from qpdesolver.applications.fwi.model import Acquisition2D, FWIDataset2D
        acq = Acquisition2D(sources=sources, receivers=receivers, omegas=omegas)
        return FWIDataset2D(
            acquisition=acq,
            observed_data=obs,
            hx=hx,
            hz=hz,
            bc=bc,
            true_velocity=true_v,
            initial_velocity=init_v,
            metadata=meta,
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
        return _FWIDataset2DFallback(
            acquisition=_Acq2DFallback(sources=sources, receivers=receivers, omegas=omegas),
            observed_data=obs,
            hx=hx,
            hz=hz,
            bc=bc,
            true_velocity=true_v,
            initial_velocity=init_v,
            metadata=meta,
        )


def _write_dict_to_group(
    grp: h5py.Group,
    data: Dict[str, Any],
    compression: Optional[str] = None,
    compression_opts: Optional[int] = None,
    dtype: Optional[str] = None,
) -> None:
    for k, v in data.items():
        if isinstance(v, dict):
            sub = grp.create_group(k)
            _write_dict_to_group(sub, v, compression=compression, compression_opts=compression_opts, dtype=dtype)
        elif isinstance(v, np.ndarray):
            _write_dataset(grp, k, v, compression=compression, compression_opts=compression_opts, dtype=dtype)
        elif isinstance(v, (list, tuple)):
            arr = np.array(v)
            if arr.dtype.kind in ("i", "f", "c", "b"):
                _write_dataset(grp, k, arr, compression=compression, compression_opts=compression_opts, dtype=dtype)
            else:
                grp.attrs[k] = json.dumps(v, default=str)
        elif isinstance(v, (int, float, bool, np.number)):
            grp.attrs[k] = v
        elif isinstance(v, str):
            grp.attrs[k] = v
        elif v is None:
            grp.attrs[k] = "__NONE__"
        else:
            grp.attrs[k] = json.dumps(v, default=str)


def _read_dict_from_group(grp: h5py.Group) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in grp.attrs.items():
        if k in ("qpdesolver_type", "schema_version", "config_json"):
            continue
        if v == "__NONE__":
            out[k] = None
        elif isinstance(v, (np.bool_, bool)):
            out[k] = bool(v)
        elif isinstance(v, np.integer):
            out[k] = int(v)
        elif isinstance(v, np.floating):
            out[k] = float(v)
        else:
            out[k] = v
    for k in grp.keys():
        item = grp[k]
        if isinstance(item, h5py.Group):
            out[k] = _read_dict_from_group(item)
        elif isinstance(item, h5py.Dataset):
            out[k] = np.array(item)
    return out


def save_results(
    path: Union[str, Path],
    group: str,
    results: Dict[str, Any],
    config: Optional[Dict[str, Any]] = None,
    *,
    overwrite: bool = False,
    compression: Optional[str] = None,
    compression_opts: Optional[int] = None,
    dtype: Optional[str] = None,
) -> None:
    """Saves arbitrary study/solver results and configuration."""
    with h5py.File(path, "a") as f:
        grp = _ensure_group_writable(f, group, overwrite)
        grp.attrs["qpdesolver_type"] = "results"
        grp.attrs["schema_version"] = SCHEMA_VERSION
        if config is not None:
            grp.attrs["config_json"] = json.dumps(config, default=str)
        _write_dict_to_group(grp, results, compression=compression, compression_opts=compression_opts, dtype=dtype)


def load_results(path: Union[str, Path], group: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Loads study/solver results and configuration from an HDF5 group."""
    with h5py.File(path, "r") as f:
        if group not in f:
            raise H5SchemaError(f"Group '{group}' not found in '{path}'.")
        grp = f[group]
        if grp.attrs.get("qpdesolver_type", "") != "results":
            raise H5SchemaError(f"Group '{group}' is not of type 'results'.")
        cfg_str = grp.attrs.get("config_json", "{}")
        config = json.loads(cfg_str)
        results = _read_dict_from_group(grp)
    return results, config
