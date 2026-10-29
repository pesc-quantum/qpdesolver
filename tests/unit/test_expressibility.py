"""
Unit tests for Expressibility and Meyer-Wallach entanglement measure.
"""
import pytest
import numpy as np

from qpdesolver.trainability.expressibility import (
    meyer_wallach_measure,
    entangling_capability,
    expressibility,
)
from qpdesolver.ansatze.library import RealAmplitudes

def test_meyer_wallach_analytical_states():
    # 1. Product state |00> -> Q = 0
    s_prod = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    assert np.isclose(meyer_wallach_measure(s_prod, n_qubits=2), 0.0)

    # 2. Maximally entangled Bell state 1/sqrt(2)(|00> + |11>) -> Q = 1.0
    s_bell = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)], dtype=complex)
    assert np.isclose(meyer_wallach_measure(s_bell, n_qubits=2), 1.0)

def test_ansatz_expressibility_and_entanglement():
    ansatz = RealAmplitudes(layers=2)
    ent = entangling_capability(ansatz, n_qubits=2, n_samples=15)
    assert 0.0 < ent <= 1.0

    expr = expressibility(ansatz, n_qubits=2, n_pairs=50, bins=20)
    assert expr >= 0.0
