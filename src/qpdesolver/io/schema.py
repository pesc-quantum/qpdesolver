"""
Schema definitions and validation utilities for HDF5 persistence.
"""
from typing import Dict, Any

class H5SchemaError(ValueError):
    """Raised when an HDF5 group violates the qpdesolver data layout schema."""
    pass

SCHEMA_VERSION = "1.1.0"
DEVITO_FORMAT = "devito-seismic-hdf5"

SUPPORTED_TYPES = (
    "linear_system",
    "velocity_model",
    "fwi_dataset",
    "results",
    "velocity_model_2d",
    "fwi_dataset_2d",
)
