"""
Configuration dataclass for Harrow-Hassidim-Lloyd (HHL) algorithm.
"""
from dataclasses import dataclass
from typing import Optional, Tuple, Union

@dataclass
class HHLConfig:
    """
    Configuration parameters for HHL solver.

    Attributes:
        n_clock: number of clock/ancilla qubits for Quantum Phase Estimation (QPE).
        evolution: method to implement e^{i A t}. 'exact' (dense expm) or 'trotter' (TrotterProduct).
        trotter_steps: number of Trotter steps if evolution='trotter'.
        trotter_order: order of Trotter formula (1 or 2).
        signed: if True, use two's-complement signed eigenvalue interpretation (essential for indefinite systems like Helmholtz).
        C: scale constant for eigenvalue inversion RY rotation (2 * arcsin(C / lambda)).
           'grid': smallest non-zero grid eigenvalue (guarantees |C/lambda| <= 1).
           'lambda_min': smallest actual matrix eigenvalue abs value.
           float: explicit numeric constant.
        spectral_bounds: optional explicit (lambda_min, lambda_max) bounds.
        phase_margin: safety fraction in [0, 1) to keep |lambda| * t / (2*pi) strictly inside (-0.5, 0.5).
        auto_dilate: if True, automatically perform Hermitian dilation if system is non-Hermitian.
    """
    n_clock: int = 6
    evolution: str = "exact"
    trotter_steps: int = 20
    trotter_order: int = 2
    signed: bool = True
    C: Union[str, float] = "grid"
    spectral_bounds: Optional[Tuple[float, float]] = None
    phase_margin: float = 0.90
    auto_dilate: bool = True
