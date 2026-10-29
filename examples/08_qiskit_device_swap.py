"""
Example 08: Backend and Device Swapping.
Demonstrates configuring and switching PennyLane simulator backends (e.g. default.qubit, lightning.qubit)
and shows how Qiskit backend stubs integrate into the registry architecture.
"""
from loguru import logger
import pennylane as qml

from qpdesolver.core.registry import REGISTRY_BACKENDS
from qpdesolver.backends.pennylane_backend import PennyLaneBackend
from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.solvers.vqls import VQLSSolver
from qpdesolver.ansatze import RealAmplitudes
from qpdesolver.optimizers.scipy_wrapper import ScipyOptimizer

def main():
    logger.info("Listing registered backend providers:")
    for b_name in REGISTRY_BACKENDS.available():
        logger.info(f" - {b_name}: {REGISTRY_BACKENDS.get(b_name)}")

    # Setup small system
    helm = Helmholtz1D(n_points=4, h=1.0, bc="dirichlet")
    system = helm.system(rhs=helm.point_source(1), omega=1.0)

    # 1. Run on standard default.qubit with analytical gradient
    logger.info("Initializing PennyLaneBackend with 'default.qubit' (analytic backprop)...")
    backend_default = PennyLaneBackend(device_name="default.qubit")
    solver_default = VQLSSolver(
        ansatz=RealAmplitudes(layers=1),
        backend=backend_default,
        optimizer=ScipyOptimizer("BFGS"),
        max_iter=20,
        seed=42,
    )
    res_def = solver_default.solve(system)
    logger.info(f"default.qubit solve: Fidelity = {res_def.fidelity*100:.2f}%, Time = {res_def.wall_time:.3f}s")

    # 2. Check lightning availability and run with gradient-free or SPSA optimization
    lightning_avail = "lightning.qubit" in qml.devices.plugin_devices
    logger.info(f"lightning.qubit plugin available: {lightning_avail}")
    if lightning_avail:
        logger.info("Swapping backend to 'lightning.qubit' (gradient-free COBYLA optimizer)...")
        # lightning.qubit evaluates state vectors without backprop
        backend_lightning = PennyLaneBackend(device_name="lightning.qubit", diff_method=None)
        solver_lightning = VQLSSolver(
            ansatz=RealAmplitudes(layers=1),
            backend=backend_lightning,
            optimizer=ScipyOptimizer("COBYLA"),
            max_iter=30,
            seed=42,
        )
        res_light = solver_lightning.solve(system)
        logger.info(f"lightning.qubit solve: Fidelity = {res_light.fidelity*100:.2f}%, Time = {res_light.wall_time:.3f}s")

    logger.info("Backend device configuration and swapping demonstration completed!")

if __name__ == "__main__":
    main()
