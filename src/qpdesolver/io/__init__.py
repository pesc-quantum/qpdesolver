"""
I/O subpackage for qpdesolver.
"""
from qpdesolver.io.schema import H5SchemaError, SCHEMA_VERSION
from qpdesolver.io.hdf5 import (
    list_groups,
    save_linear_system,
    load_linear_system,
    save_velocity_model,
    load_velocity_model,
    save_fwi_dataset,
    load_fwi_dataset,
    save_results,
    load_results,
)

__all__ = [
    "H5SchemaError",
    "SCHEMA_VERSION",
    "list_groups",
    "save_linear_system",
    "load_linear_system",
    "save_velocity_model",
    "load_velocity_model",
    "save_fwi_dataset",
    "load_fwi_dataset",
    "save_results",
    "load_results",
]
