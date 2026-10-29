"""
Example 05: Barren Plateau Scan & Diagnosis.
Demonstrates sampling gradient variances across qubit counts and fitting
exponential decay rates to detect barren plateaus in variational circuits.
"""
from loguru import logger
from qpdesolver.ansatze import RealAmplitudes
from qpdesolver.trainability.gradients import (
    gradient_statistics,
    diagnose_barren_plateau,
    make_global_projector_cost,
    make_local_projector_cost,
)

def main():
    logger.info("Initializing Barren Plateau Scan (2 to 4 qubits)...")
    qubit_range = [2, 3, 4]
    ansatz = RealAmplitudes(layers=2)

    stats_global = []
    stats_local = []

    for n_q in qubit_range:
        logger.info(f"Sampling gradients for n_qubits={n_q}...")
        sg = gradient_statistics(ansatz, n_q, make_global_projector_cost, n_samples=25, seed=42)
        sl = gradient_statistics(ansatz, n_q, make_local_projector_cost, n_samples=25, seed=42)
        stats_global.append(sg)
        stats_local.append(sl)
        logger.info(
            f"Qubits={n_q} | Global Var: {sg.total_variance:.4e} | Local Var: {sl.total_variance:.4e}"
        )

    diag_g = diagnose_barren_plateau(stats_global)
    diag_l = diagnose_barren_plateau(stats_local)

    logger.info("--- Diagnostic Results ---")
    logger.info(f"Global Cost: Barren Plateau = {diag_g.is_barren} (Decay Rate: {diag_g.decay_rate:.3f}, R^2: {diag_g.r2:.3f})")
    logger.info(f"Local Cost:  Barren Plateau = {diag_l.is_barren} (Decay Rate: {diag_l.decay_rate:.3f}, R^2: {diag_l.r2:.3f})")

if __name__ == "__main__":
    main()
