"""
Expressibility and Entangling Capability metrics (Sim et al. 2019, Meyer-Wallach 2002).
Quantifies how uniformly an ansatz explores Hilbert space and its capacity to generate entanglement.
"""
from typing import Sequence, Tuple
import numpy as np
import scipy.stats
import pennylane as qml

from qpdesolver.ansatze.base import Ansatz

def expressibility(
    ansatz: Ansatz,
    n_qubits: int,
    *,
    n_pairs: int = 1000,
    bins: int = 50,
    seed: int = 42,
) -> float:
    """
    Computes Kullback-Leibler divergence between the ansatz fidelity distribution
    and the Haar-distributed ensemble:
        D_KL( P_ansatz(F) || P_Haar(F) )
    where P_Haar(F) = (N - 1) (1 - F)^{N - 2}, with N = 2^n_qubits.
    Lower values indicate higher expressibility (closer to Haar random states).
    """
    rng = np.random.default_rng(seed)
    dev = qml.device("default.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev)
    def circuit(p):
        ansatz.apply(p, wires)
        return qml.state()

    fidelities = []
    for _ in range(n_pairs):
        p1 = ansatz.init_params(n_qubits, rng=rng, strategy="uniform")
        p2 = ansatz.init_params(n_qubits, rng=rng, strategy="uniform")
        s1 = circuit(p1)
        s2 = circuit(p2)
        fid = float(np.abs(np.vdot(s1, s2))**2)
        fidelities.append(np.clip(fid, 0.0, 1.0))

    fidelities = np.array(fidelities)

    # Histogram estimation
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    p_ansatz, _ = np.histogram(fidelities, bins=bin_edges, density=True)
    p_ansatz = p_ansatz / np.sum(p_ansatz) # normalize discrete probabilities

    # Theoretical Haar distribution: (N - 1) * (1 - F)^(N - 2)
    N = 2**n_qubits
    p_haar_cont = (N - 1) * ((1.0 - bin_centers)**(N - 2))
    p_haar = p_haar_cont / np.sum(p_haar_cont)

    # Regularize zero probabilities for KL divergence
    eps = 1e-10
    p_ansatz = np.clip(p_ansatz, eps, 1.0)
    p_haar = np.clip(p_haar, eps, 1.0)
    p_ansatz = p_ansatz / np.sum(p_ansatz)
    p_haar = p_haar / np.sum(p_haar)

    kl_div = float(scipy.stats.entropy(p_ansatz, p_haar))
    return kl_div

def meyer_wallach_measure(state: np.ndarray, n_qubits: int) -> float:
    """
    Computes the Meyer-Wallach entanglement measure Q(|psi>) in [0, 1]:
        Q = 2 (1 - 1/n sum_k Tr(rho_k^2))
    where rho_k is the 1-qubit reduced density matrix of qubit k.
    Q = 0 for product states, Q = 1 for maximally entangled states (e.g. Bell, GHZ).
    """
    if n_qubits < 2:
        return 0.0

    state_tensor = state.reshape([2] * n_qubits)
    purity_sum = 0.0

    for k in range(n_qubits):
        # Trace out all qubits except k
        axes_to_trace = [i for i in range(n_qubits) if i != k]
        # Partial trace via tensor contraction
        rho_k = np.tensordot(state_tensor, np.conj(state_tensor), axes=(axes_to_trace, axes_to_trace))
        purity = float(np.real(np.trace(rho_k @ rho_k)))
        purity_sum += purity

    q_measure = 2.0 * (1.0 - (purity_sum / n_qubits))
    return float(np.clip(q_measure, 0.0, 1.0))

def entangling_capability(
    ansatz: Ansatz,
    n_qubits: int,
    *,
    n_samples: int = 100,
    seed: int = 42,
) -> float:
    """
    Computes average Meyer-Wallach entanglement measure over random circuit parameters.
    """
    rng = np.random.default_rng(seed)
    dev = qml.device("default.qubit", wires=n_qubits)
    wires = list(range(n_qubits))

    @qml.qnode(dev)
    def circuit(p):
        ansatz.apply(p, wires)
        return qml.state()

    q_values = []
    for _ in range(n_samples):
        p = ansatz.init_params(n_qubits, rng=rng, strategy="uniform")
        s = circuit(p)
        q_val = meyer_wallach_measure(s, n_qubits)
        q_values.append(q_val)

    return float(np.mean(q_values))
