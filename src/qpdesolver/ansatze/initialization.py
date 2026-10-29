"""
Parameter initialization strategies for variational circuits.
Includes Barren Plateau mitigation strategies:
- identity_blocks: pairs of parameter layers initialize to cancel out (Grant et al. 2019).
- small_angle: Gaussian N(0, sigma^2) centered at zero.
- uniform: Uniform in [low, high).
- zeros: Zero parameters.
- beta: Beta distribution.
"""
from typing import Optional, Any
import numpy as np

from qpdesolver.core.registry import REGISTRY_INITIALIZERS

@REGISTRY_INITIALIZERS.register("uniform")
def init_uniform(n_params: int, rng: np.random.Generator, low: float = 0.0, high: float = 2 * np.pi, **kwargs) -> np.ndarray:
    return rng.uniform(low, high, n_params)

@REGISTRY_INITIALIZERS.register("small_angle")
def init_small_angle(n_params: int, rng: np.random.Generator, sigma: float = 0.05, **kwargs) -> np.ndarray:
    return rng.normal(0.0, sigma, n_params)

@REGISTRY_INITIALIZERS.register("zeros")
def init_zeros(n_params: int, rng: np.random.Generator, **kwargs) -> np.ndarray:
    return np.zeros(n_params, dtype=float)

@REGISTRY_INITIALIZERS.register("beta")
def init_beta(n_params: int, rng: np.random.Generator, a: float = 2.0, b: float = 5.0, scale: float = 2 * np.pi, **kwargs) -> np.ndarray:
    return rng.beta(a, b, n_params) * scale

@REGISTRY_INITIALIZERS.register("identity_blocks")
def init_identity_blocks(
    n_params: int,
    rng: np.random.Generator,
    ansatz: Optional[Any] = None,
    n_wires: Optional[int] = None,
    **kwargs
) -> np.ndarray:
    """
    Initializes parameter blocks such that paired layers cancel, starting the ansatz near identity.
    For RealAmplitudes / HEA: layer 0 is initialized with random angles, and layer 1 with negative angles.
    """
    params = np.zeros(n_params, dtype=float)
    if ansatz is not None and n_wires is not None:
        try:
            slices = ansatz.layer_param_slices(n_wires)
            n_layers = len(slices)
            for i in range(0, n_layers, 2):
                s1 = slices[i]
                length1 = s1.stop - s1.start
                angles = rng.uniform(0.0, 2 * np.pi, length1)
                params[s1] = angles
                if i + 1 < n_layers:
                    s2 = slices[i + 1]
                    length2 = s2.stop - s2.start
                    min_len = min(length1, length2)
                    params[s2.start:s2.start + min_len] = -angles[:min_len]
            return params
        except Exception:
            pass

    # Fallback if structure is unknown: small angle
    return rng.normal(0.0, 0.01, n_params)

def initialize_params(
    n_params: int,
    ansatz: Optional[Any] = None,
    n_wires: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
    strategy: str = "uniform",
    **kwargs
) -> np.ndarray:
    """Dispatches parameter initialization via the registry."""
    if rng is None:
        rng = np.random.default_rng(42)
    init_fn = REGISTRY_INITIALIZERS.get(strategy)
    return init_fn(n_params, rng=rng, ansatz=ansatz, n_wires=n_wires, **kwargs)
