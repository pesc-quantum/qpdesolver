"""
Numerical metrics for evaluating solution quality and problem hardness.
"""
import numpy as np

def fidelity(state1: np.ndarray, state2: np.ndarray) -> float:
    """
    Computes quantum state fidelity |<psi|phi>|^2 between two normalized statevectors.
    """
    s1 = np.asarray(state1, dtype=complex)
    s2 = np.asarray(state2, dtype=complex)
    n1 = np.linalg.norm(s1)
    n2 = np.linalg.norm(s2)
    if n1 < 1e-15 or n2 < 1e-15:
        return 0.0
    u1 = s1 / n1
    u2 = s2 / n2
    overlap = np.vdot(u1, u2)
    return float(np.abs(overlap)**2)

def relative_residual(A: np.ndarray, x: np.ndarray, b: np.ndarray) -> float:
    """
    Computes relative residual ||A x - b||_2 / ||b||_2.
    """
    b_norm = np.linalg.norm(b)
    if b_norm < 1e-15:
        return float(np.linalg.norm(A @ x))
    res = np.linalg.norm(A @ x - b)
    return float(res / b_norm)

def condition_number(A: np.ndarray) -> float:
    """
    Computes spectral condition number kappa = sigma_max / sigma_min.
    """
    s = np.linalg.svd(A, compute_uv=False)
    s_min = s[-1]
    if s_min < 1e-15:
        return float("inf")
    return float(s[0] / s_min)
