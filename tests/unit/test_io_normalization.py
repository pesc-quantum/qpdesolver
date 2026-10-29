import pytest
import numpy as np
from qpdesolver.io.normalization import (
    NormalizationParams,
    compute_normalization_params,
    apply_seismic_normalization,
    denormalize_velocity,
)


def test_normalization_disabled_by_default():
    v = np.array([1.5, 2.5])
    params = compute_normalization_params(v, h=10.0, normalize=False)
    assert not params.normalized
    assert params.target_units == "raw"
    v_norm, h_norm, _ = apply_seismic_normalization(v, h=10.0, params=params)
    assert np.allclose(v_norm, v)
    assert h_norm == 10.0


def test_unit_target_without_normalize_raises_error():
    v = np.array([1.5, 2.5])
    with pytest.raises(ValueError, match="unit_target requires normalize=True"):
        compute_normalization_params(v, h=10.0, normalize=False, unit_target="si")


def test_normalization_to_si():
    v_kms = np.array([1.5, 2.5])
    params = compute_normalization_params(
        v_kms, h=10.0, normalize=True, unit_target="si", source_units="kms"
    )
    v_si, h_si, _ = apply_seismic_normalization(v_kms, h=10.0, params=params)
    assert np.allclose(v_si, [1500.0, 2500.0])
    assert h_si == 10.0
    v_rec = denormalize_velocity(v_si, params)
    assert np.allclose(v_rec, v_kms)


def test_normalization_to_dimensionless():
    v_kms = np.array([1.5, 2.5])
    params = compute_normalization_params(
        v_kms, h=10.0, normalize=True, unit_target="dimensionless", source_units="kms"
    )
    v_dim, h_dim, omega_fn = apply_seismic_normalization(v_kms, h=10.0, params=params)
    assert np.isclose(np.mean(v_dim), 1.0)
    assert np.isclose(h_dim, 1.0)
    v_rec = denormalize_velocity(v_dim, params)
    assert np.allclose(v_rec, v_kms)


def test_normalization_to_kms():
    v_si = np.array([1500.0, 2500.0])
    params = compute_normalization_params(
        v_si, h=10.0, normalize=True, unit_target="kms", source_units="si"
    )
    v_kms, h_kms, _ = apply_seismic_normalization(v_si, h=10.0, params=params)
    assert np.allclose(v_kms, [1.5, 2.5])
    assert np.isclose(h_kms, 0.01)  # 10 m -> 0.01 km
    v_rec = denormalize_velocity(v_kms, params)
    assert np.allclose(v_rec, v_si)


def test_tuple_grid_spacing():
    v_kms = np.array([1.5, 2.5])
    h_tuple = (10.0, 20.0)
    params = compute_normalization_params(
        v_kms, h=h_tuple, normalize=True, unit_target="si", source_units="kms"
    )
    v_si, h_si, _ = apply_seismic_normalization(v_kms, h=h_tuple, params=params)
    assert np.allclose(v_si, [1500.0, 2500.0])
    assert h_si == (10.0, 20.0)


def test_auto_source_units_detection():
    v_kms = np.array([1.5, 2.5])
    params_kms = compute_normalization_params(
        v_kms, h=10.0, normalize=True, unit_target="si", source_units="auto"
    )
    assert params_kms.source_units == "kms"
    assert params_kms.velocity_scale == 1000.0

    v_si = np.array([1500.0, 2500.0])
    params_si = compute_normalization_params(
        v_si, h=10.0, normalize=True, unit_target="si", source_units="auto"
    )
    assert params_si.source_units == "si"
    assert params_si.velocity_scale == 1.0


def test_invalid_unit_target():
    v = np.array([1.5, 2.5])
    with pytest.raises(ValueError, match="Unsupported unit_target"):
        compute_normalization_params(v, h=10.0, normalize=True, unit_target="invalid_target")


def test_invalid_source_units():
    v = np.array([1.5, 2.5])
    with pytest.raises(ValueError, match="Unsupported source_units"):
        compute_normalization_params(
            v, h=10.0, normalize=True, unit_target="si", source_units="invalid_source"
        )


def test_normalization_params_methods():
    v_kms = np.array([1.5, 2.5])
    params = compute_normalization_params(
        v_kms, h=10.0, normalize=True, unit_target="dimensionless", source_units="kms"
    )
    d = params.to_dict()
    assert d["normalized"] is True
    assert d["target_units"] == "dimensionless"
    assert "velocity_scale" in d
    assert "omega_scale" in d

    # Test scale_omega and method call
    omega = 2.0 * np.pi * 5.0
    omega_scaled = params.scale_omega(omega)
    assert np.isclose(omega_scaled, omega * params.omega_scale)

    v_dim = v_kms * params.velocity_scale
    v_rec = params.denormalize_velocity(v_dim)
    assert np.allclose(v_rec, v_kms)

    # Calling apply_seismic_normalization with omega passed directly
    v_norm, h_norm, omega_out = apply_seismic_normalization(
        v_kms, h=10.0, omega=omega, params=params
    )
    assert np.isclose(omega_out, omega_scaled)
