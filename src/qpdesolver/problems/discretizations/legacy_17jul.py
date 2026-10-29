"""
Constructs the exact linear systems used in the 17-Jul-2026 experiments:
1. Helmholtz 1D periodic system (4 points, omega=1.5, v=2.0, h=1.0) with Bell state RHS.
2. Parameterized mock FWI system (4 points, velocity_anomaly=0.35, freq=1.2) with Bell state RHS.
"""
from typing import Tuple
import numpy as np

from qpdesolver.problems.linear_system import LinearSystem

def get_legacy_17jul_helmholtz(omega: float = 1.5, velocity: float = 2.0, h: float = 1.0) -> LinearSystem:
    """
    Reconstructs the 17-Jul periodic Helmholtz system:
        A = -Lap - (omega^2/velocity^2)*I
    where Lap is the periodic Laplacian:
        diag = -2/h^2, off-diag = 1/h^2.
    """
    diag_val = 2.0 / (h**2) - (omega**2) / (velocity**2)
    off_diag = -1.0 / (h**2)

    A_mat = np.array([
        [diag_val, off_diag, 0.0, off_diag],
        [off_diag, diag_val, off_diag, 0.0],
        [0.0, off_diag, diag_val, off_diag],
        [off_diag, 0.0, off_diag, diag_val]
    ], dtype=complex)

    # In 17-Jul: Bell state RHS |b> = 1/sqrt(2) (|00> + |11>) = [1/sqrt(2), 0, 0, 1/sqrt(2)]
    b_vec = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)], dtype=complex)

    return LinearSystem(
        A=A_mat,
        b=b_vec,
        name="17jul_helmholtz_periodic",
        metadata={"omega": omega, "velocity": velocity, "h": h, "source": "17-Jul-2026/helmholtz_1d.py"}
    )

def get_legacy_17jul_mock_fwi(velocity_anomaly: float = 0.35, freq: float = 1.2) -> LinearSystem:
    """
    Reconstructs the 17-Jul mock parameterized FWI system:
        A(anomaly, freq)
    """
    diag_val = 1.5 + 0.2 * velocity_anomaly
    cross_talk = 0.4 * freq

    A_mat = np.array([
        [diag_val, cross_talk, 0.0, 0.1],
        [cross_talk, diag_val, 0.1, 0.0],
        [0.0, 0.1, diag_val, cross_talk],
        [0.1, 0.0, cross_talk, diag_val]
    ], dtype=complex)

    b_vec = np.array([1.0 / np.sqrt(2.0), 0.0, 0.0, 1.0 / np.sqrt(2.0)], dtype=complex)

    return LinearSystem(
        A=A_mat,
        b=b_vec,
        name="17jul_mock_fwi",
        metadata={"velocity_anomaly": velocity_anomaly, "freq": freq, "source": "17-Jul-2026/mock_fwi_parameterized.py"}
    )
