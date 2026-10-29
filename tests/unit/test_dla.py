"""
Unit tests for Dynamical Lie Algebra (DLA) computations.
"""
import pytest
import numpy as np

from qpdesolver.trainability.dla import dynamical_lie_algebra
from qpdesolver.ansatze.library import RealAmplitudes

def test_dla_single_qubit_su2():
    # Generators {X, Y} on 1 qubit must close to su(2) of dimension 3
    gens = ["X", "Y"]
    res = dynamical_lie_algebra(gens, n_qubits=1)
    assert res.dimension == 3
    assert res.is_full_su is True

def test_dla_tfim_generators():
    # Transverse-field Ising model generators {X0, X1, Z0Z1}
    gens = ["XI", "IX", "ZZ"]
    res = dynamical_lie_algebra(gens, n_qubits=2)
    # Known exact dimension = 6
    assert res.dimension == 6
    assert res.is_full_su is False # su(4) is 15

def test_dla_realamplitudes_ansatz():
    ansatz = RealAmplitudes(layers=2)
    res = dynamical_lie_algebra(ansatz, n_qubits=2)
    # Real orthogonal circuits stay in so(4), whose dimension is 6
    assert res.dimension <= 6
    assert res.variance_prediction > 0.0
