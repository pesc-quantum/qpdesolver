"""
devito_hdf5 -- backward compatibility shim for Devito HDF5 I/O and snapshot recording.

Re-exports core functions from `qpdesolver.generator.recorder` and `qpdesolver.io.devito_io`.
"""
from __future__ import annotations

from qpdesolver.generator.recorder import (
    SnapshotRecorder,
    flatten_fields,
    strip_absorbing_layer,
)
from qpdesolver.io.devito_io import (
    export_devito_hdf5,
    export_devito_hdf5 as export_hdf5,
    load_devito_hdf5,
    load_devito_hdf5 as load_hdf5,
    summarize_hdf5,
)

__all__ = [
    "SnapshotRecorder",
    "export_hdf5",
    "load_hdf5",
    "summarize_hdf5",
    "flatten_fields",
    "strip_absorbing_layer",
    "export_devito_hdf5",
    "load_devito_hdf5",
]
