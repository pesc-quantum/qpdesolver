"""
Synthetic seismic model generation and wavefield simulation package.
"""
from qpdesolver.generator.models import (
    two_layer_velocity,
    multi_layer_velocity,
    gaussian_anomaly_velocity,
    salt_inclusion_velocity,
)
from qpdesolver.generator.recorder import (
    SnapshotRecorder,
    flatten_fields,
    strip_absorbing_layer,
)
from qpdesolver.generator.acoustic import (
    SeismicSurveySpec,
    generate_seismic_sample,
)
from qpdesolver.generator.devito_hdf5 import (
    export_hdf5,
    load_hdf5,
    summarize_hdf5,
)

__all__ = [
    "two_layer_velocity",
    "multi_layer_velocity",
    "gaussian_anomaly_velocity",
    "salt_inclusion_velocity",
    "SnapshotRecorder",
    "flatten_fields",
    "strip_absorbing_layer",
    "SeismicSurveySpec",
    "generate_seismic_sample",
    "export_hdf5",
    "load_hdf5",
    "summarize_hdf5",
]

