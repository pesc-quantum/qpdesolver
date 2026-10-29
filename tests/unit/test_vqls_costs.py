"""
Unit tests for VQLS cost functions, analytical gradients, and Fubini-Study metric.
"""
import pytest
import numpy as np
import pennylane as qml

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.ansatze.library import RealAmplitudes
from qpdesolver.ansatze.adapters import FunctionAnsatz
from qpdesolver.solvers.vqls.costs import GlobalCost, LocalCost
from qpdesolver.solvers.vqls.objective import VQLSObjective
from qpdesolver.backends.pennylane_backend import PennyLaneBackend

def test_vqls_cost_zero_at_exact_solution():
    """If ansatz prepares the exact normalized solution, cost must be zero."""
    A = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=complex)
    b = np.array([1.0, 0.0], dtype=complex)
    sys = LinearSystem(A=A, b=b)
    x_exact = sys.classical_solution()
    x_normed = x_exact / np.linalg.norm(x_exact)

    # Function ansatz that prepares the exact state directly when p=0
    def exact_prep(p, wires):
        qml.StatePrep(x_normed, wires=wires)

    ansatz = FunctionAnsatz(exact_prep, num_params_fn=lambda n: 1)
    backend = PennyLaneBackend()

    obj_global = VQLSObjective(sys, ansatz, cost=GlobalCost(), backend=backend)
    assert np.isclose(obj_global.value(np.array([0.0])), 0.0, atol=1e-10)

def test_vqls_gradient_matches_finite_differences():
    A = np.array([[3.0, 0.5], [0.5, 2.0]], dtype=complex)
    b = np.array([1.0, 2.0], dtype=complex)
    sys = LinearSystem(A=A, b=b)
    ansatz = RealAmplitudes(layers=1)

    obj = VQLSObjective(sys, ansatz)
    p0 = np.array([0.4, -0.6, 1.2, 0.8])

    # Analytical autograd gradient
    grad_auto = obj.grad(p0)

    # Central finite differences
    eps = 1e-6
    grad_fd = np.zeros_like(p0)
    for i in range(len(p0)):
        p_plus = p0.copy(); p_plus[i] += eps
        p_minus = p0.copy(); p_minus[i] -= eps
        grad_fd[i] = (obj.value(p_plus) - obj.value(p_minus)) / (2.0 * eps)

    assert np.allclose(grad_auto, grad_fd, atol=1e-5)

def test_vqls_metric_tensor():
    A = np.eye(4, dtype=complex)
    b = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    sys = LinearSystem(A=A, b=b)
    ansatz = RealAmplitudes(layers=1)

    obj = VQLSObjective(sys, ansatz)
    p0 = np.ones(ansatz.num_params(2)) * 0.5
    metric = obj.metric_tensor(p0)

    # Metric must be symmetric and positive semi-definite
    assert np.allclose(metric, metric.T)
    evals = np.linalg.eigvalsh(metric)
    assert np.all(evals >= -1e-10)

def test_custom_cost_function_flexibility():
    """Verifies that researchers can define arbitrary custom cost functions."""
    A = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=complex)
    b = np.array([1.0, 0.0], dtype=complex)
    sys = LinearSystem(A=A, b=b)
    ansatz = RealAmplitudes(layers=1)

    # Custom loss: standard residual + L2 parameter regularization penalty
    def my_research_loss(state, Ax, b_vec, system, params):
        residual_loss = 1.0 - (np.abs(np.vdot(b_vec, Ax))**2) / (np.vdot(Ax, Ax).real + 1e-12)
        l2_reg = 0.05 * np.sum(params**2)
        return residual_loss + l2_reg

    from qpdesolver.solvers.vqls.costs import CustomCostFunction
    custom_cost = CustomCostFunction(my_research_loss)
    obj = VQLSObjective(sys, ansatz, cost=custom_cost)

    p0 = np.array([0.5, 0.5, 0.5, 0.5])
    val = obj.value(p0)
    assert val > 0.05 * np.sum(p0**2)
