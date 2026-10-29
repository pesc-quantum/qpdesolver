"""
Unit tests for ansatze: RealAmplitudes parity with 17-Jul, HEA, adapters, and initialization.
"""
import pytest
import numpy as np
import pennylane as qml

from qpdesolver.ansatze import (
    RealAmplitudes,
    HardwareEfficient,
    TemplateAnsatz,
    FunctionAnsatz,
    LegacyAnsatzAdapter,
    initialize_params,
)
from qpdesolver.core.registry import REGISTRY_ANSATZE

def test_ansatze_registered():
    assert "realamplitudes" in REGISTRY_ANSATZE
    assert "hardwareefficient" in REGISTRY_ANSATZE

def test_realamplitudes_parity_with_17jul():
    """Verifies that RealAmplitudes produces the exact state layout as 17-Jul RealAmplitudesAnsatz."""
    import sys
    from pathlib import Path
    target = None
    for p in Path(__file__).resolve().parents:
        cand = p / "17-Jul-2026"
        if cand.is_dir() and (cand / "vqls_core").is_dir():
            target = cand
            break
    if target is None:
        pytest.skip("17-Jul-2026 directory not found for parity check.")
    sys.path.insert(0, str(target))
    from vqls_core.ansatze import RealAmplitudesAnsatz as LegacyRealAmplitudes

    n_qubits = 2
    layers = 2
    wires = list(range(n_qubits))
    dev = qml.device("default.qubit", wires=n_qubits)

    new_ansatz = RealAmplitudes(layers=layers)
    legacy_ansatz = LegacyRealAmplitudes(layers=layers)

    assert new_ansatz.num_params(n_qubits) == legacy_ansatz.num_params(n_qubits) == 6

    # Fixed parameters
    params = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])

    @qml.qnode(dev)
    def circuit_new():
        new_ansatz(params, wires)
        return qml.state()

    @qml.qnode(dev)
    def circuit_legacy():
        legacy_ansatz(params, wires)
        return qml.state()

    state_new = circuit_new()
    state_legacy = circuit_legacy()

    assert np.allclose(state_new, state_legacy, atol=1e-12)

def test_hardware_efficient_and_layer_slices():
    hea = HardwareEfficient(layers=3, rotations=("RY", "RZ"), entangler="linear")
    n_wires = 3
    # 3 wires * 2 rotations * (3 layers + 1) = 3 * 2 * 4 = 24
    assert hea.num_params(n_wires) == 24

    slices = hea.layer_param_slices(n_wires)
    assert len(slices) == 4
    assert slices[0] == slice(0, 6)
    assert slices[3] == slice(18, 24)

    hea_copy = hea.with_layers(5)
    assert hea_copy.layers == 5

def test_template_and_function_adapters():
    # TemplateAnsatz
    template = TemplateAnsatz(qml.StronglyEntanglingLayers, n_layers=2)
    n_params = template.num_params(2)
    assert n_params == 2 * 2 * 3 # 12

    # FunctionAnsatz
    def custom_circuit(p, wires):
        qml.RX(p[0], wires=wires[0])
        qml.RY(p[1], wires=wires[1])

    func_ansatz = FunctionAnsatz(custom_circuit, num_params_fn=lambda n: 2)
    assert func_ansatz.num_params(2) == 2

def test_initialization_strategies(rng):
    n_wires = 2
    ansatz = RealAmplitudes(layers=2)
    n_p = ansatz.num_params(n_wires)

    p_uni = initialize_params(n_p, rng=rng, strategy="uniform")
    assert len(p_uni) == n_p
    assert np.all((p_uni >= 0.0) & (p_uni <= 2 * np.pi))

    p_small = initialize_params(n_p, rng=rng, strategy="small_angle", sigma=0.01)
    assert np.all(np.abs(p_small) < 0.1)

    p_zeros = initialize_params(n_p, rng=rng, strategy="zeros")
    assert np.all(p_zeros == 0.0)

    p_ident = initialize_params(n_p, ansatz=ansatz, n_wires=n_wires, rng=rng, strategy="identity_blocks")
    assert len(p_ident) == n_p
