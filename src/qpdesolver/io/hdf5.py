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
from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.applications.fwi.model import VelocityModel, FWIDataset, Acquisition

def _ensure_group_writable(h5file: h5py.File, group_name: str, overwrite: bool) -> h5py.Group:
    """Helper to check or delete existing group based on overwrite flag."""
    if group_name in h5file:
        if not overwrite:
            raise H5SchemaError(f"Group '{group_name}' already exists in '{h5file.filename}' and overwrite is False.")
        del h5file[group_name]
    return h5file.create_group(group_name)

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
) -> None:
    """Saves a LinearSystem to an HDF5 group."""
    with h5py.File(path, "a") as f:
        grp = _ensure_group_writable(f, group, overwrite)
        grp.attrs["qpdesolver_type"] = "linear_system"
        grp.attrs["schema_version"] = SCHEMA_VERSION
        grp.attrs["name"] = system.name
        grp.attrs["metadata_json"] = json.dumps(system.metadata, default=str)

        grp.create_dataset("b", data=system.b)

        if sparse:
            csr = sp.csr_matrix(system.A)
            grp_a = grp.create_group("A")
            grp_a.attrs["format"] = "csr"
            grp_a.attrs["shape"] = csr.shape
            grp_a.create_dataset("data", data=csr.data)
            grp_a.create_dataset("indices", data=csr.indices)
            grp_a.create_dataset("indptr", data=csr.indptr)
        else:
            grp.create_dataset("A", data=system.A)

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
) -> None:
    """Saves a VelocityModel to an HDF5 group."""
    with h5py.File(path, "a") as f:
        grp = _ensure_group_writable(f, group, overwrite)
        grp.attrs["qpdesolver_type"] = "velocity_model"
        grp.attrs["schema_version"] = SCHEMA_VERSION
        grp.attrs["h"] = float(model.h)
        grp.attrs["bc"] = model.bc
        grp.attrs["metadata_json"] = json.dumps(model.metadata, default=str)
        grp.create_dataset("velocity", data=model.velocity)

def load_velocity_model(
    path: Union[str, Path],
    group: str,
    *,
    resample_to: Optional[int] = None,
) -> VelocityModel:
    """Loads a VelocityModel, optionally resampling it to a specified number of grid points."""
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

    model = VelocityModel(velocity=v, h=h, bc=bc, metadata=meta)
    if resample_to is not None:
        model = model.resample(resample_to)
    return model

def save_fwi_dataset(
    path: Union[str, Path],
    dataset: FWIDataset,
    group: str,
    *,
    overwrite: bool = False,
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

        grp.create_dataset("observed_data", data=dataset.observed_data)
        if dataset.true_velocity is not None:
            grp.create_dataset("true_velocity", data=dataset.true_velocity)
        if dataset.initial_velocity is not None:
            grp.create_dataset("initial_velocity", data=dataset.initial_velocity)

def load_fwi_dataset(path: Union[str, Path], group: str) -> FWIDataset:
    """Loads an FWIDataset from an HDF5 group."""
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
        acq = Acquisition(sources=sources, receivers=receivers, omegas=omegas)

        obs = np.array(grp["observed_data"])
        h = float(grp.attrs.get("h", 1.0))
        bc = grp.attrs.get("bc", "dirichlet")
        meta = json.loads(grp.attrs.get("metadata_json", "{}"))

        true_v = np.array(grp["true_velocity"]) if "true_velocity" in grp else None
        init_v = np.array(grp["initial_velocity"]) if "initial_velocity" in grp else None

    return FWIDataset(
        acquisition=acq,
        observed_data=obs,
        h=h,
        bc=bc,
        true_velocity=true_v,
        initial_velocity=init_v,
        metadata=meta,
    )

def _write_dict_to_group(grp: h5py.Group, data: Dict[str, Any]) -> None:
    for k, v in data.items():
        if isinstance(v, dict):
            sub = grp.create_group(k)
            _write_dict_to_group(sub, v)
        elif isinstance(v, np.ndarray):
            grp.create_dataset(k, data=v)
        elif isinstance(v, (list, tuple)):
            arr = np.array(v)
            if arr.dtype.kind in ("i", "f", "c", "b"):
                grp.create_dataset(k, data=arr)
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
) -> None:
    """Saves arbitrary study/solver results and configuration."""
    with h5py.File(path, "a") as f:
        grp = _ensure_group_writable(f, group, overwrite)
        grp.attrs["qpdesolver_type"] = "results"
        grp.attrs["schema_version"] = SCHEMA_VERSION
        if config is not None:
            grp.attrs["config_json"] = json.dumps(config, default=str)
        _write_dict_to_group(grp, results)

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
