"""
Norm and global phase recovery from normalized quantum statevectors.
Solves the 1D complex scalar least-squares problem:
    min_alpha || A (alpha * |x_hat>) - b ||_2^2
Analytical solution:
    alpha = <A |x_hat>, b> / <A |x_hat>, A |x_hat>>.
"""
from typing import Tuple
import numpy as np

def rescale_least_squares(
    A: np.ndarray,
    b: np.ndarray,
    state: np.ndarray,
) -> Tuple[np.ndarray, complex]:
    """
    Computes optimal complex scale factor alpha and physical solution vector x = alpha * state.

    Args:
        A: (N, N) matrix.
        b: (N,) right-hand side vector.
        state: (N,) normalized quantum statevector |x_hat>.

    Returns:
        (x, alpha): physical solution x and scalar alpha.
    """
    state_arr = np.asarray(state, dtype=complex)
    b_arr = np.asarray(b, dtype=complex)
    A_arr = np.asarray(A, dtype=complex)

    norm_s = np.linalg.norm(state_arr)
    if norm_s > 1e-15:
        state_normed = state_arr / norm_s
    else:
        state_normed = state_arr

    A_state = A_arr @ state_normed
    denom = np.vdot(A_state, A_state).real
    if denom < 1e-15:
        # Fallback if in nullspace
        alpha = complex(np.linalg.norm(b_arr) / max(norm_s, 1e-15))
    else:
        num = np.vdot(A_state, b_arr)
        alpha = complex(num / denom)

    x = alpha * state_normed
    return x, alpha
