"""
I/O subpackage for qpdesolver.
"""
from qpdesolver.io.schema import H5SchemaError, SCHEMA_VERSION, DEVITO_FORMAT
from qpdesolver.io.normalization import (
    NormalizationParams,
    compute_normalization_params,
    apply_seismic_normalization,
    denormalize_velocity,
)
from qpdesolver.io.hdf5 import (
    list_groups,
    save_linear_system,
    load_linear_system,
    save_velocity_model,
    load_velocity_model,
    save_velocity_model_2d,
    load_velocity_model_2d,
    save_fwi_dataset,
    load_fwi_dataset,
    save_fwi_dataset_2d,
    load_fwi_dataset_2d,
    save_results,
    load_results,
)
from qpdesolver.io.devito_io import (
    export_devito_hdf5,
    load_devito_hdf5,
    summarize_hdf5,
    extract_velocity_1d_from_devito,
    extract_velocity_2d_from_devito,
    extract_frequency_data_from_devito,
    build_fwi_dataset_from_devito,
)

__all__ = [
    "H5SchemaError",
    "SCHEMA_VERSION",
    "DEVITO_FORMAT",
    "NormalizationParams",
    "compute_normalization_params",
    "apply_seismic_normalization",
    "denormalize_velocity",
    "list_groups",
    "save_linear_system",
    "load_linear_system",
    "save_velocity_model",
    "load_velocity_model",
    "save_velocity_model_2d",
    "load_velocity_model_2d",
    "save_fwi_dataset",
    "load_fwi_dataset",
    "save_fwi_dataset_2d",
    "load_fwi_dataset_2d",
    "save_results",
    "load_results",
    "export_devito_hdf5",
    "load_devito_hdf5",
    "summarize_hdf5",
    "extract_velocity_1d_from_devito",
    "extract_velocity_2d_from_devito",
    "extract_frequency_data_from_devito",
    "build_fwi_dataset_from_devito",
]
