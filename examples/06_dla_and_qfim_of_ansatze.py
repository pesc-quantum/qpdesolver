"""
Example 06: Dynamical Lie Algebra and Quantum Fisher Information Metric (QFIM) analysis.
Examines the Lie closure dimension dim(g) and QFIM rank across ansatz architectures.
"""
from loguru import logger
from qpdesolver.ansatze import RealAmplitudes, HardwareEfficient
from qpdesolver.trainability.dla import dynamical_lie_algebra
from qpdesolver.trainability.geometry import qfim_statistics

def main():
    logger.info("Computing Dynamical Lie Algebra (DLA) dimension for 2 and 3 qubits...")
    ansatze = {
        "RealAmplitudes": RealAmplitudes(layers=2),
        "HardwareEfficient": HardwareEfficient(layers=2),
    }

    for name, ans in ansatze.items():
        for n_q in [2, 3]:
            dla = dynamical_lie_algebra(ans, n_qubits=n_q)
            full_su_dim = 4**n_q - 1
            logger.info(
                f"[{name}] {n_q} Qubits: dim(g) = {dla.dimension} "
                f"(Full su({2**n_q}) is {full_su_dim}, reached full: {dla.is_full_su})"
            )

    logger.info("Analyzing QFIM rank saturation for RealAmplitudes on 2 qubits...")
    for l in [1, 2, 3]:
        q_stat = qfim_statistics(RealAmplitudes(layers=l), n_qubits=2, n_samples=10, seed=42)
        logger.info(
            f"Layers={l} ({RealAmplitudes(layers=l).num_params(2)} params) | "
            f"QFIM Rank: {q_stat.rank} / {q_stat.max_rank_bound} | Effective Dim: {q_stat.effective_dimension:.2f}"
        )

if __name__ == "__main__":
    main()
