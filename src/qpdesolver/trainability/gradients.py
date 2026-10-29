"""
Gradient variance estimation and Barren Plateau (BP) detection.
Quantifies the exponential vanishing of variance: Var[dC/d_theta] ~ O(b^{-n}) as n increases.
"""
from dataclasses import dataclass
from typing import Sequence, List, Callable, Optional, Dict, Any
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qpdesolver.ansatze.base import Ansatz
from qpdesolver.backends.pennylane_backend import PennyLaneBackend

@dataclass
class GradientStats:
    """
    Empirical gradient statistics across random circuit parameterizations.
    """
    n_qubits: int
    n_layers: int
    mean: np.ndarray
    variance: np.ndarray
    cost_variance: float
    samples: int

    @property
    def total_variance(self) -> float:
        """Mean parameter gradient variance."""
        return float(np.mean(self.variance))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "n_qubits": self.n_qubits,
            "n_layers": self.n_layers,
            "mean": self.mean,
            "variance": self.variance,
            "total_variance": self.total_variance,
            "cost_variance": self.cost_variance,
            "samples": self.samples,
        }

@dataclass
class BPDiagnosis:
    """
    Quantitative diagnosis of Barren Plateau phenomenon.
    Fits Var ~ a * exp(-decay_rate * n).
    """
    decay_rate: float
    r2: float
    is_barren: bool
    reference_2_pow_minus_n: float

def gradient_statistics(
    ansatz: Ansatz,
    n_qubits: int,
    cost_fn_builder: Callable[[Any, Sequence[int]], Callable[[np.ndarray], float]],
    *,
    n_samples: int = 100,
    init_strategy: str = "uniform",
    seed: int = 42,
    backend: Optional[PennyLaneBackend] = None,
) -> GradientStats:
    """
    Samples gradient vectors of a parameterized circuit over random parameters.
    """
    b = backend or PennyLaneBackend()
    wires = list(range(n_qubits))
    cost_fn = cost_fn_builder(ansatz, wires)
    grad_fn = qml.grad(cost_fn)

    n_p = ansatz.num_params(n_qubits)
    rng = np.random.default_rng(seed)

    grads = []
    costs = []

    for _ in range(n_samples):
        p0 = ansatz.init_params(n_qubits, rng=rng, strategy=init_strategy)
        px = pnp.array(p0, requires_grad=True)
        g = grad_fn(px)
        c = cost_fn(px)
        grads.append(np.array(g, dtype=float))
        costs.append(float(c))

    grads_arr = np.array(grads) # (n_samples, n_p)
    costs_arr = np.array(costs)

    mean_grad = np.mean(grads_arr, axis=0)
    var_grad = np.var(grads_arr, axis=0)
    var_cost = float(np.var(costs_arr))

    layers = getattr(ansatz, "layers", 1)

    return GradientStats(
        n_qubits=n_qubits,
        n_layers=layers,
        mean=mean_grad,
        variance=var_grad,
        cost_variance=var_cost,
        samples=n_samples,
    )

def scan_gradient_variance(
    ansatz_factory: Callable[[int], Ansatz],
    qubits: Sequence[int],
    cost_fn_builder: Callable[[Any, Sequence[int]], Callable[[np.ndarray], float]],
    *,
    layers: int = 2,
    n_samples: int = 100,
    seed: int = 42,
) -> List[GradientStats]:
    """
    Sweeps qubit count n to evaluate scaling of gradient variance.
    """
    results = []
    for n in qubits:
        ansatz = ansatz_factory(layers)
        stats = gradient_statistics(
            ansatz=ansatz,
            n_qubits=n,
            cost_fn_builder=cost_fn_builder,
            n_samples=n_samples,
            seed=seed,
        )
        results.append(stats)
    return results

def diagnose_barren_plateau(stats_list: Sequence[GradientStats], threshold: float = 0.3) -> BPDiagnosis:
    """
    Fits exponential decay Var ~ A * exp(-decay_rate * n) on log-variances.
    If decay_rate > threshold and R^2 > 0.8, flags circuit as suffering from Barren Plateaus.
    """
    ns = np.array([s.n_qubits for s in stats_list], dtype=float)
    vars_val = np.array([max(s.total_variance, 1e-16) for s in stats_list], dtype=float)

    log_vars = np.log(vars_val)
    # Linear fit: log(Var) = -decay_rate * n + intercept
    p = np.polyfit(ns, log_vars, 1)
    decay_rate = -p[0]

    # R^2 correlation
    pred = np.polyval(p, ns)
    ss_tot = np.sum((log_vars - np.mean(log_vars))**2)
    ss_res = np.sum((log_vars - pred)**2)
    r2 = 1.0 - (ss_res / (ss_tot + 1e-12))

    is_barren = bool(decay_rate > threshold and r2 > 0.75)

    return BPDiagnosis(
        decay_rate=float(decay_rate),
        r2=float(r2),
        is_barren=is_barren,
        reference_2_pow_minus_n=float(np.log(2.0)), # ~0.693 for Haar 2^{-n}
    )

def make_global_projector_cost(ansatz: Ansatz, wires: Sequence[int]) -> Callable[[np.ndarray], float]:
    """
    Constructs textbook global projector cost: C = <0...0| U^\dagger |0...0><0...0| U |0...0> = |<0...0|psi>|^2.
    Well known to exhibit barren plateaus for all non-trivial ansatz depths (McClean et al. 2018).
    """
    n_q = len(wires)
    dev = qml.device("default.qubit", wires=n_q)

    @qml.qnode(dev)
    def circuit(params):
        ansatz.apply(params, wires)
        return qml.state()

    def cost(params):
        state = circuit(params)
        # Overlap with |0...0> (index 0)
        p0 = pnp.real(state[0] * pnp.conj(state[0]))
        return 1.0 - p0

    return cost

def make_local_projector_cost(ansatz: Ansatz, wires: Sequence[int]) -> Callable[[np.ndarray], float]:
    """
    Constructs local projector cost: C = 1 - 1/n sum_j <psi| |0><0|_j |psi>.
    Exhibits polynomial gradients for shallow architectures (Cerezo et al. 2021).
    """
    n_q = len(wires)
    dev = qml.device("default.qubit", wires=n_q)

    @qml.qnode(dev)
    def circuit(params):
        ansatz.apply(params, wires)
        # Measure local Z expectation values
        return [qml.expval(qml.PauliZ(w)) for w in wires]

    def cost(params):
        # <0|Z|0> = 1, <1|Z|1> = -1 -> projector P_0 = 1/2(I + Z)
        z_vals = circuit(params)
        local_p0 = [0.5 * (1.0 + z) for z in z_vals]
        avg_p0 = pnp.mean(pnp.stack(local_p0))
        return 1.0 - avg_p0

    return cost
