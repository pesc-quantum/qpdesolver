"""
Unit tests for HDF5 I/O routines and schema validation.
"""
import pytest
import numpy as np
import scipy.sparse as sp

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.applications.fwi.model import VelocityModel, FWIDataset, Acquisition
from qpdesolver.io import (
    H5SchemaError,
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

def test_linear_system_hdf5_roundtrip(tmp_path, rng):
    h5_file = tmp_path / "test_systems.h5"

    # 1. Real dense system
    A_real = rng.normal(size=(4, 4))
    A_real = A_real + 5.0 * np.eye(4) # ensure well conditioned
    b_real = np.array([1.0, 2.0, 3.0, 4.0])
    sys_real = LinearSystem(A=A_real, b=b_real, name="real_sys", metadata={"tag": "unit_test"})

    save_linear_system(h5_file, sys_real, "real_system")
    loaded_real = load_linear_system(h5_file, "real_system")

    assert loaded_real.name == "real_sys"
    assert loaded_real.metadata.get("tag") == "unit_test"
    assert np.allclose(loaded_real.A, A_real)
    assert np.allclose(loaded_real.b, b_real)

    # 2. Complex dense system
    A_cplx = A_real + 1j * rng.normal(size=(4, 4))
    b_cplx = b_real + 1j * np.array([0.1, -0.2, 0.3, -0.4])
    sys_cplx = LinearSystem(A=A_cplx, b=b_cplx, name="cplx_sys")

    save_linear_system(h5_file, sys_cplx, "complex_system")
    loaded_cplx = load_linear_system(h5_file, "complex_system")
    assert np.allclose(loaded_cplx.A, A_cplx)
    assert np.allclose(loaded_cplx.b, b_cplx)

    # 3. Sparse CSR system
    sys_sparse = LinearSystem(A=A_real, b=b_real, name="sparse_sys")
    save_linear_system(h5_file, sys_sparse, "sparse_system", sparse=True)
    loaded_sparse = load_linear_system(h5_file, "sparse_system")
    assert np.allclose(loaded_sparse.A, A_real)

    # list_groups
    groups = list_groups(h5_file, qpdesolver_type="linear_system")
    assert groups == ["complex_system", "real_system", "sparse_system"]

def test_velocity_model_hdf5_and_resampling(tmp_path):
    h5_file = tmp_path / "test_velocity.h5"

    # Linear profile of length 37 (non-power of 2)
    v_orig = np.linspace(1.5, 3.5, 37)
    model = VelocityModel(velocity=v_orig, h=10.0, bc="dirichlet", metadata={"profile": "linear"})

    save_velocity_model(h5_file, model, "vel_profile")
    loaded = load_velocity_model(h5_file, "vel_profile")
    assert loaded.n_points == 37
    assert np.allclose(loaded.velocity, v_orig)

    # Resample to 8 points (power of 2)
    resampled = load_velocity_model(h5_file, "vel_profile", resample_to=8)
    assert resampled.n_points == 8
    # Endpoints strictly preserved
    assert np.isclose(resampled.velocity[0], v_orig[0])
    assert np.isclose(resampled.velocity[-1], v_orig[-1])
    # Monotonicity preserved
    assert np.all(np.diff(resampled.velocity) > 0)

def test_fwi_dataset_hdf5_roundtrip(tmp_path, rng):
    h5_file = tmp_path / "test_fwi.h5"

    acq = Acquisition(sources=[0, 3], receivers=[0, 1, 2, 3], omegas=[1.0, 2.0])
    obs = rng.normal(size=(2, 2, 4)) + 1j * rng.normal(size=(2, 2, 4))
    true_v = np.array([2.0, 2.0, 2.5, 2.5])
    init_v = np.array([2.2, 2.2, 2.2, 2.2])

    dataset = FWIDataset(
        acquisition=acq,
        observed_data=obs,
        h=1.0,
        bc="dirichlet",
        true_velocity=true_v,
        initial_velocity=init_v,
        metadata={"survey": "toy1d"},
    )

    save_fwi_dataset(h5_file, dataset, "toy_survey")
    loaded = load_fwi_dataset(h5_file, "toy_survey")

    assert loaded.acquisition.sources == [0, 3]
    assert loaded.acquisition.receivers == [0, 1, 2, 3]
    assert loaded.acquisition.omegas == [1.0, 2.0]
    assert np.allclose(loaded.observed_data, obs)
    assert np.allclose(loaded.true_velocity, true_v)
    assert np.allclose(loaded.initial_velocity, init_v)

def test_results_hdf5_roundtrip(tmp_path):
    h5_file = tmp_path / "test_results.h5"

    config = {"n_clock": 6, "solver": "HHL", "ansatz": "HardwareEfficient"}
    data = {
        "fidelities": np.array([0.98, 0.99, 0.999]),
        "cost_history": [1.0, 0.5, 0.1, 0.001],
        "final_loss": 0.001,
        "converged": True,
        "sub_results": {
            "iterations": 45,
            "eigenvalues": np.array([0.5, 1.2, 2.8, 3.5]),
        }
    }

    save_results(h5_file, "run_01", data, config=config)
    loaded_data, loaded_config = load_results(h5_file, "run_01")

    assert loaded_config["n_clock"] == 6
    assert np.allclose(loaded_data["fidelities"], [0.98, 0.99, 0.999])
    assert np.isclose(loaded_data["final_loss"], 0.001)
    assert loaded_data["converged"] is True
    assert loaded_data["sub_results"]["iterations"] == 45
    assert np.allclose(loaded_data["sub_results"]["eigenvalues"], [0.5, 1.2, 2.8, 3.5])

def test_hdf5_schema_errors(tmp_path):
    h5_file = tmp_path / "error_cases.h5"

    A = np.eye(4)
    b = np.array([1.0, 0.0, 0.0, 0.0])
    sys = LinearSystem(A=A, b=b)

    # Overwrite collision
    save_linear_system(h5_file, sys, "test_grp")
    with pytest.raises(H5SchemaError, match="already exists"):
        save_linear_system(h5_file, sys, "test_grp", overwrite=False)

    # Overwrite = True succeeds
    save_linear_system(h5_file, sys, "test_grp", overwrite=True)

    # Corrupt group in h5: wrong type
    import h5py
    with h5py.File(h5_file, "a") as f:
        grp = f.create_group("corrupt_grp")
        grp.attrs["qpdesolver_type"] = "wrong_type"

    with pytest.raises(H5SchemaError, match="invalid qpdesolver_type"):
        load_linear_system(h5_file, "corrupt_grp")

    # Missing b dataset
    with h5py.File(h5_file, "a") as f:
        grp_no_b = f.create_group("no_b")
        grp_no_b.attrs["qpdesolver_type"] = "linear_system"
        grp_no_b.create_dataset("A", data=np.eye(4))

    with pytest.raises(H5SchemaError, match="missing dataset 'b'"):
        load_linear_system(h5_file, "no_b")
