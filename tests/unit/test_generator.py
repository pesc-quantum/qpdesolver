"""
Unit tests for modular synthetic seismic generator.
"""
import pytest
import numpy as np

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
from qpdesolver.generator.acoustic import SeismicSurveySpec, generate_seismic_sample
from qpdesolver.io.devito_io import load_devito_hdf5


def test_velocity_models_generation():
    m2 = two_layer_velocity(shape=(21, 21), v_top=1.5, v_bottom=2.5, interface_z=0.5)
    assert m2.shape == (21, 21)
    assert np.isclose(m2[0, 0], 1.5)
    assert np.isclose(m2[0, -1], 2.5)

    mg = gaussian_anomaly_velocity(shape=(21, 21), v_bg=2.0, dv=0.6)
    assert mg.shape == (21, 21)
    assert np.max(mg) > 2.5

    # Multi-layer model
    ml = multi_layer_velocity(shape=(30, 30), velocities=[1.5, 2.0, 3.0], interfaces=[0.33, 0.66])
    assert ml.shape == (30, 30)
    assert np.isclose(ml[0, 0], 1.5)
    assert np.isclose(ml[0, 15], 2.0)
    assert np.isclose(ml[0, 25], 3.0)

    # Salt inclusion model
    ms = salt_inclusion_velocity(shape=(31, 31), v_bg=2.0, v_salt=4.5)
    assert ms.shape == (31, 31)
    assert np.max(ms) == 4.5
    assert np.min(ms) == 2.0


def test_generate_seismic_sample_numpy_backend(tmp_path):
    h5_file = tmp_path / "numpy_sample.h5"
    vp = two_layer_velocity(shape=(31, 31), v_top=1.5, v_bottom=2.5)
    spec = SeismicSurveySpec(
        shape=(31, 31),
        spacing=(10.0, 10.0),
        nbl=5,
        t0=0.0,
        tn=50.0,
        f0=0.015,
        nsnaps=4,
    )
    res = generate_seismic_sample(str(h5_file), vp=vp, spec=spec, backend="numpy")
    assert h5_file.is_file()
    data = load_devito_hdf5(str(h5_file))
    assert data["attrs"]["format"] == "devito-seismic-hdf5"
    assert "rec" in data["geometry"]
    assert "src" in data["geometry"]
    assert "fields" in data
    assert "u" in data["fields"]
    assert data["fields"]["u"].shape[0] == 4
    assert data["model"]["vp"].shape == (31, 31)


def test_generate_seismic_sample_devito_backend(tmp_path):
    pytest.importorskip("devito")
    h5_file = tmp_path / "devito_sample.h5"
    vp = two_layer_velocity(shape=(31, 31), v_top=1.5, v_bottom=2.5)
    spec = SeismicSurveySpec(
        shape=(31, 31),
        spacing=(10.0, 10.0),
        nbl=5,
        t0=0.0,
        tn=50.0,
        f0=0.015,
        nsnaps=4,
    )
    res = generate_seismic_sample(str(h5_file), vp=vp, spec=spec, backend="devito")
    assert h5_file.is_file()
    data = load_devito_hdf5(str(h5_file))
    assert data["attrs"]["format"] == "devito-seismic-hdf5"
    assert "rec" in data["geometry"]
    assert "fields" in data
    assert "u" in data["fields"]
    assert data["fields"]["u"].shape[0] == 4


def test_strip_absorbing_layer():
    arr = np.ones((10, 20, 20))
    stripped = strip_absorbing_layer(arr, nbl=2, ndim=2)
    assert stripped.shape == (10, 16, 16)


def test_devito_hdf5_backward_compatibility():
    from qpdesolver.generator.devito_hdf5 import (
        SnapshotRecorder as LegacyRecorder,
        export_hdf5,
        load_hdf5,
        summarize_hdf5,
        flatten_fields as legacy_flatten,
        strip_absorbing_layer as legacy_strip,
    )
    assert LegacyRecorder is SnapshotRecorder
    assert legacy_strip is strip_absorbing_layer
    assert legacy_flatten is flatten_fields
    assert callable(export_hdf5)
    assert callable(load_hdf5)
    assert callable(summarize_hdf5)

