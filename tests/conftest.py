"""
Pytest configuration and shared fixtures for qpdesolver.
"""
import pytest
import numpy as np

@pytest.fixture
def rng():
    return np.random.default_rng(42)
