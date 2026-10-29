"""
Unit tests for gradient variance calculation and Barren Plateau diagnosis.
"""
import pytest
import numpy as np

from qpdesolver.ansatze.library import RealAmplitudes
from qpdesolver.trainability.gradients import (
    gradient_statistics,
    diagnose_barren_plateau,
    make_global_projector_cost,
    make_local_projector_cost,
    GradientStats,
)

def test_gradient_statistics_variance():
    ansatz = RealAmplitudes(layers=1)
    stats = gradient_statistics(
        ansatz=ansatz,
        n_qubits=2,
        cost_fn_builder=make_global_projector_cost,
        n_samples=25,
        seed=42,
    )

    assert stats.n_qubits == 2
    assert len(stats.variance) == ansatz.num_params(2)
    assert np.all(stats.variance >= 0.0)
    assert stats.total_variance > 0.0

def test_diagnose_barren_plateau_synthetic():
    # Synthetic exponential decay Var = 1.0 * exp(-0.7 * n)
    ns = [2, 3, 4, 5]
    synthetic_stats = [
        GradientStats(
            n_qubits=n,
            n_layers=2,
            mean=np.zeros(4),
            variance=np.full(4, np.exp(-0.7 * n)),
            cost_variance=0.1,
            samples=50,
        )
        for n in ns
    ]

    diag = diagnose_barren_plateau(synthetic_stats, threshold=0.3)
    assert diag.is_barren is True
    assert np.isclose(diag.decay_rate, 0.7, atol=0.05)
    assert diag.r2 > 0.99
