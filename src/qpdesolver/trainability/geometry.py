"""
Quantum geometry: Fubini-Study metric tensor and Quantum Fisher Information Matrix (QFIM).
Studies the Riemannian geometry of variational ansatz parameter spaces,
rank saturation, and overparameterization transitions (Larocca et al. 2023).
"""
from dataclasses import dataclass
from typing import Sequence, List, Dict, Any, Callable, Optional
import numpy as np
import pennylane as qml
from pennylane import numpy as pnp

from qpdesolver.ansatze.base import Ansatz

@dataclass
class QFIMStats:
    """
    Quantum Fisher Information Matrix spectral and rank properties.
    """
    rank: int
    n_params: int
    max_rank_bound: int
    effective_dimension: float
    eigenvalues: np.ndarray

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "n_params": self.n_params,
            "max_rank_bound": self.max_rank_bound,
            "effective_dimension": self.effective_dimension,
            "eigenvalues": self.eigenvalues,
        }

def fubini_study_metric(ansatz: Ansatz, n_qubits: int, params: np.ndarray) -> np.ndarray:
    """
    Computes the exact Fubini-Study metric tensor g_ij(theta):
        g_ij = Re( <d_i psi | d_j psi> - <d_i psi | psi> <psi | d_j psi> )
    """
    dev = qml.device("default.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev)
    def state_circuit(p):
        ansatz.apply(p, wires)
        return qml.state()

    px = pnp.array(params, requires_grad=True)

    def real_state(p): return pnp.real(state_circuit(p))
    def imag_state(p): return pnp.imag(state_circuit(p))

    jac_r = qml.jacobian(real_state)(px)
    jac_i = qml.jacobian(imag_state)(px)
    jac = np.array(jac_r, dtype=float) + 1j * np.array(jac_i, dtype=float)
    psi = np.array(state_circuit(px), dtype=complex)

    overlap_jac = np.conj(jac.T) @ jac
    psi_grad = np.conj(jac.T) @ psi
    fs = np.real(overlap_jac - np.outer(psi_grad, np.conj(psi_grad)))
    return 0.5 * (fs + fs.T)

def qfim(ansatz: Ansatz, n_qubits: int, params: np.ndarray) -> np.ndarray:
    """
    Computes Quantum Fisher Information Matrix QFIM = 4 * g_{FS}.
    """
    return 4.0 * fubini_study_metric(ansatz, n_qubits, params)

def qfim_statistics(
    ansatz: Ansatz,
    n_qubits: int,
    *,
    n_samples: int = 10,
    tol: float = 1e-6,
    seed: int = 42,
) -> QFIMStats:
    """
    Samples QFIM across parameter space to determine its typical rank and effective dimension.
    """
    rng = np.random.default_rng(seed)
    n_p = ansatz.num_params(n_qubits)
    ranks = []
    all_evals = []

    for _ in range(n_samples):
        p0 = ansatz.init_params(n_qubits, rng=rng, strategy="uniform")
        F = qfim(ansatz, n_qubits, p0)
        evals = np.linalg.eigvalsh(F)
        evals_pos = np.maximum(evals, 0.0)
        rank = int(np.sum(evals_pos > tol))
        ranks.append(rank)
        all_evals.append(evals_pos)

    avg_evals = np.mean(all_evals, axis=0)
    median_rank = int(np.median(ranks))

    # Upper bound on rank: min(n_params, 2 * (2^n - 1))
    bound = min(n_p, 2 * (2**n_qubits - 1))

    # Effective dimension (information-theoretic): sum(evals) / max(evals)
    max_e = np.max(avg_evals) if len(avg_evals) > 0 else 1.0
    eff_dim = float(np.sum(avg_evals) / max(max_e, 1e-12))

    return QFIMStats(
        rank=median_rank,
        n_params=n_p,
        max_rank_bound=bound,
        effective_dimension=eff_dim,
        eigenvalues=avg_evals,
    )

def overparameterization_scan(
    ansatz_factory: Callable[[int], Ansatz],
    n_qubits: int,
    layers_list: Sequence[int],
    *,
    n_samples: int = 5,
    seed: int = 42,
) -> List[QFIMStats]:
    """
    Evaluates QFIM rank across varying layer counts to detect overparameterization onset.
    Overparameterization occurs when rank saturates despite increasing layer count.
    """
    stats_list = []
    for l in layers_list:
        ansatz = ansatz_factory(l)
        stats = qfim_statistics(ansatz, n_qubits, n_samples=n_samples, seed=seed)
        stats_list.append(stats)
    return stats_list
