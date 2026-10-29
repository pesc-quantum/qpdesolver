"""
Example 07: Direct Ansatz Comparison on a PDE Linear System.
Evaluates solution fidelity, wall time, and circuit evaluations across
different ansatz architectures solving the 1D Helmholtz equation.
"""
from loguru import logger
from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.ansatze import RealAmplitudes, HardwareEfficient
from qpdesolver.trainability.studies import AnsatzComparison

def main():
    logger.info("Initializing PDE Problem (Helmholtz 1D Dirichlet, 4 points, 2 qubits)...")
    helm = Helmholtz1D(n_points=4, h=1.0, bc="dirichlet")
    system = helm.system(rhs=helm.point_source(1), omega=1.0, name="Helmholtz_Comparison")

    ansatze = {
        "RealAmplitudes_L1": RealAmplitudes(layers=1),
        "RealAmplitudes_L2": RealAmplitudes(layers=2),
        "HardwareEfficient_L1": HardwareEfficient(layers=1),
        "HardwareEfficient_L2": HardwareEfficient(layers=2),
    }

    logger.info("Running comparative VQLS benchmark...")
    comparison = AnsatzComparison(
        system=system,
        ansatze=ansatze,
        max_iter=40,
        seeds=(42,),
    )
    result = comparison.run()

    logger.info("--- Benchmark Results Summary ---")
    for rec in result.records:
        logger.info(
            f"Ansatz: {rec['ansatz']:22s} | "
            f"Fidelity: {rec['fidelity']*100:.3f}% | "
            f"Residual: {rec['residual']:.4e} | "
            f"Time: {rec['wall_time']:.3f}s | "
            f"Params: {rec['num_params']}"
        )

if __name__ == "__main__":
    main()
