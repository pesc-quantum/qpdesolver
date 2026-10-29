"""
Unit tests for optimized devito-seismic-hdf5 I/O routines, summarize utilities,
and FWI bridge converters.
"""
import pytest
import numpy as np
import h5py
from pathlib import Path

from qpdesolver.io.devito_io import (
    export_devito_hdf5,
    load_devito_hdf5,
    summarize_hdf5,
    extract_velocity_1d_from_devito,
    extract_velocity_2d_from_devito,
    extract_frequency_data_from_devito,
    build_fwi_dataset_from_devito,
)
from qpdesolver.io.hdf5 import (
    save_velocity_model_2d,
    load_velocity_model_2d,
    save_fwi_dataset_2d,
    load_fwi_dataset_2d,
)


def test_devito_hdf5_synthetic_export_and_load(tmp_path):
    h5_path = tmp_path / "test_synth.h5"
    vp = np.full((20, 20), 2.0, dtype=np.float32)
    vp[:, 10:] = 3.0
    time_vals = np.linspace(0, 100, 50)
    snaps = {"u": np.zeros((5, 20, 20), dtype=np.float32)}

    export_devito_hdf5(
        str(h5_path),
        vp=vp,
        origin=(0.0, 0.0),
        spacing=(10.0, 10.0),
        time_values=time_vals,
        snapshots=snaps,
        compression="gzip",
    )

    # Verify chunking and units in raw HDF5 file
    with h5py.File(str(h5_path), "r") as h5:
        assert h5["fields"]["u"].chunks == (1, 20, 20)
        assert h5["model"]["vp"].attrs.get("units") == "km/s"
        assert h5["coords"]["x"].attrs.get("units") == "m"
        assert h5["coords"]["z"].attrs.get("units") == "m"
        assert h5["time"].attrs.get("units") == "ms"

    data = load_devito_hdf5(str(h5_path), include_fields=False)
    assert data["attrs"]["format"] == "devito-seismic-hdf5"
    assert "u" not in data.get("fields", {})
    assert data["model"]["vp"].shape == (20, 20)

    summary_str = summarize_hdf5(str(h5_path), print_output=False)
    assert "devito-seismic-hdf5" in summary_str
    assert "vp" in summary_str


def test_extract_velocity_and_fwi_dataset(tmp_path):
    h5_path = tmp_path / "test_synth2.h5"
    vp = np.full((16, 16), 1.5, dtype=np.float32)
    vp[:, 8:] = 2.5
    export_devito_hdf5(str(h5_path), vp=vp, origin=(0.0, 0.0), spacing=(10.0, 10.0))

    # Extract 1D profile resampled to 4 points
    v1d = extract_velocity_1d_from_devito(str(h5_path), resample_to=4, normalize=True, unit_target="si")
    assert v1d.n_points == 4
    assert np.isclose(v1d.velocity[0], 1500.0)
    assert np.isclose(v1d.velocity[-1], 2500.0)

    # Build 1D FWI dataset (helmholtz_consistent)
    fwi_ds = build_fwi_dataset_from_devito(str(h5_path), mode="1d", grid_shape=4, normalize=True, unit_target="si")
    assert fwi_ds.true_velocity.shape == (4,)
    assert np.allclose(fwi_ds.true_velocity, v1d.velocity)

    # Build 2D FWI dataset
    fwi_ds_2d = build_fwi_dataset_from_devito(str(h5_path), mode="2d", grid_shape=(4, 4), normalize=True, unit_target="si")
    assert fwi_ds_2d.true_velocity.shape == (4, 4)


def test_devito_hdf5_selective_slice_and_normalization(tmp_path):
    h5_path = tmp_path / "test_slice_norm.h5"
    vp = np.full((16, 16), 2.0, dtype=np.float32)
    time_vals = np.linspace(0, 100, 20)
    snaps = {"u": np.arange(10 * 16 * 16, dtype=np.float32).reshape((10, 16, 16))}

    export_devito_hdf5(
        str(h5_path),
        vp=vp,
        origin=(0.0, 0.0),
        spacing=(10.0, 10.0),
        time_values=time_vals,
        snapshots=snaps,
        compression="gzip",
    )

    # Test selective slicing
    data_sliced = load_devito_hdf5(str(h5_path), include_fields=True, snapshot_indices=[1, 4])
    assert data_sliced["fields"]["u"].shape == (2, 16, 16)
    assert np.allclose(data_sliced["fields"]["u"][0], snaps["u"][1])
    assert np.allclose(data_sliced["fields"]["u"][1], snaps["u"][4])

    # Test unit_target without normalize raises ValueError
    with pytest.raises(ValueError, match="unit_target requires normalize=True"):
        load_devito_hdf5(str(h5_path), normalize=False, unit_target="si")

    # Test normalize=True to SI
    data_si = load_devito_hdf5(str(h5_path), normalize=True, unit_target="si", include_fields=False)
    assert np.isclose(data_si["model"]["vp"][0, 0], 2000.0)
    assert data_si["attrs"]["normalization"]["normalized"] is True
    assert data_si["attrs"]["normalization"]["target_units"] == "si"


def test_devito_frequency_extraction_and_2d_velocity(tmp_path):
    h5_path = tmp_path / "test_freq_2d.h5"
    vp = np.full((16, 16), 1.8, dtype=np.float32)
    vp[:, 8:] = 2.8
    nt = 100
    t_vals = np.linspace(0.0, 200.0, nt)  # dt = 2.0 ms
    rec_traces = np.sin(2.0 * np.pi * 0.01 * t_vals)[:, None] * np.ones((nt, 3))
    src_traces = np.sin(2.0 * np.pi * 0.01 * t_vals)[:, None] * np.ones((nt, 1))

    receivers = {
        "rec": {
            "coordinates": np.array([[10.0, 0.0], [50.0, 0.0], [90.0, 0.0]]),
            "data": rec_traces,
        }
    }
    sources = {
        "src": {
            "coordinates": np.array([[50.0, 0.0]]),
            "data": src_traces,
        }
    }

    export_devito_hdf5(
        str(h5_path),
        vp=vp,
        origin=(0.0, 0.0),
        spacing=(10.0, 10.0),
        time_values=t_vals,
        sources=sources,
        receivers=receivers,
    )

    # Test extract_frequency_data_from_devito
    freq_data = extract_frequency_data_from_devito(str(h5_path), frequencies_hz=[10.0])
    assert "data" in freq_data
    assert freq_data["frequencies_hz"][0] == 10.0
    assert freq_data["data"].shape[-1] == 3

    # Test build_fwi_dataset_from_devito with devito_fft
    fwi_fft = build_fwi_dataset_from_devito(str(h5_path), mode="1d", grid_shape=4, data_source="devito_fft", frequencies_hz=[10.0])
    assert fwi_fft.observed_data.shape == (1, 1, 3)

    # Test extract_velocity_2d_from_devito
    v2d = extract_velocity_2d_from_devito(str(h5_path), resample_shape=(8, 8), normalize=True, unit_target="si")
    assert v2d.velocity.shape == (8, 8)
    assert np.isclose(v2d.velocity[0, 0], 1800.0)
    assert np.isclose(v2d.velocity[0, -1], 2800.0)


def test_velocity_model_2d_and_fwi_dataset_2d_hdf5_roundtrip(tmp_path):
    h5_file = tmp_path / "test_2d_models.h5"
    vel = np.array([[1500.0, 2000.0], [1600.0, 2200.0]], dtype=float)

    class DummyModel2D:
        velocity = vel
        hx = 10.0
        hz = 12.0
        bc = "dirichlet"
        metadata = {"test": "2d"}

    save_velocity_model_2d(h5_file, DummyModel2D, "vel2d")
    loaded_vel2d = load_velocity_model_2d(h5_file, "vel2d")
    assert np.allclose(loaded_vel2d.velocity, vel)
    assert loaded_vel2d.hx == 10.0
    assert loaded_vel2d.hz == 12.0

    # 2D FWI dataset roundtrip
    class DummyAcq2D:
        sources = [(0, 0)]
        receivers = [(0, 0), (1, 1)]
        omegas = [1.0, 2.0]

    class DummyDataset2D:
        acquisition = DummyAcq2D
        observed_data = np.zeros((1, 2, 2), dtype=complex)
        hx = 10.0
        hz = 12.0
        bc = "dirichlet"
        true_velocity = vel
        initial_velocity = vel * 0.9
        metadata = {"tag": "2d_ds"}

    save_fwi_dataset_2d(h5_file, DummyDataset2D, "ds2d")
    loaded_ds2d = load_fwi_dataset_2d(h5_file, "ds2d")
    assert loaded_ds2d.observed_data.shape == (1, 2, 2)
    assert np.allclose(loaded_ds2d.true_velocity, vel)
