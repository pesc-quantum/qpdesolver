"""
Unit tests for Fubini-Study metric and Quantum Fisher Information Matrix (QFIM).
"""
import pytest
import numpy as np
import pennylane as qml

from qpdesolver.trainability.geometry import (
    fubini_study_metric,
    qfim,
    qfim_statistics,
)
from qpdesolver.ansatze.adapters import FunctionAnsatz
from qpdesolver.ansatze.library import RealAmplitudes

def test_fubini_study_single_qubit_ry():
    """For |psi(theta)> = RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1>, g = 1/4."""
    def single_ry(p, wires):
        qml.RY(p[0], wires=wires[0])

    ansatz = FunctionAnsatz(single_ry, num_params_fn=lambda n: 1)
    g = fubini_study_metric(ansatz, n_qubits=1, params=np.array([0.7]))
    assert g.shape == (1, 1)
    assert np.isclose(g[0, 0], 0.25, atol=1e-5)

def test_qfim_rank_properties():
    ansatz = RealAmplitudes(layers=2)
    stats = qfim_statistics(ansatz, n_qubits=2, n_samples=5, tol=1e-5)

    assert stats.n_params == 6
    assert stats.rank <= stats.max_rank_bound
    assert stats.effective_dimension > 0.0
