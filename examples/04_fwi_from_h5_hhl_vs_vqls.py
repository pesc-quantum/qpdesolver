"""
Example 04: Loading FWI dataset from HDF5, solving forward Helmholtz PDE
with Classical, HHL, and VQLS solvers, and computing the adjoint state gradient.
"""
from pathlib import Path
import numpy as np
from loguru import logger

from qpdesolver.io import load_fwi_dataset
from qpdesolver.applications.fwi.problem import FWIProblem
from qpdesolver.solvers.classical import ClassicalSolver
from qpdesolver.solvers.hhl import HHLSolver, HHLConfig
from qpdesolver.solvers.vqls import VQLSSolver
from qpdesolver.ansatze import RealAmplitudes


def run_fwi_hhl_vs_vqls():
    h5_path = Path(__file__).resolve().parent / "data" / "toy_fwi.h5"
    if not h5_path.is_file():
        logger.error(f"Dataset {h5_path} not found. Run make_toy_datasets.py first.")
        return

    logger.info(f"Loading FWI dataset from {h5_path}...")
    dataset = load_fwi_dataset(h5_path, group="fwi_synthetic_8pt")

    # Resample to 4 points (2 qubits) for fast quantum solver demonstration
    n_points = 4
    init_model = dataset.initial_velocity[:n_points]
    logger.info(f"Target grid: {n_points} points (2 qubits). Initial velocity: {init_model}")

    # Build FWI problem on 4 points
    from qpdesolver.applications.fwi.model import Acquisition
    acq = Acquisition(sources=[1], receivers=[0, 1, 2, 3], omegas=[1.0])
    problem = FWIProblem(acquisition=acq, n_points=n_points, h=10.0, bc="dirichlet")

    m = 1.0 / (init_model**2)
    omega = acq.omegas[0]
    src = acq.sources[0]

    # 1. Classical forward solve
    logger.info("Solving forward Helmholtz PDE with ClassicalSolver...")
    problem.solver = ClassicalSolver()
    u_classical = problem.solve_forward(m, omega, src)
    logger.info(f"Classical wavefield solution u: {np.round(u_classical, 4)}")

    # 2. HHL forward solve
    logger.info("Solving forward Helmholtz PDE with HHLSolver (n_clock=4)...")
    hhl_solver = HHLSolver(config=HHLConfig(n_clock=4))
    problem.solver = hhl_solver
    u_hhl = problem.solve_forward(m, omega, src)
    hhl_fid = np.abs(np.vdot(u_classical, u_hhl)) ** 2 / (
        np.linalg.norm(u_classical) ** 2 * np.linalg.norm(u_hhl) ** 2
    )
    logger.info(f"HHL wavefield solution u: {np.round(u_hhl, 4)}")
    logger.info(f"HHL vs Classical wavefield fidelity: {hhl_fid * 100:.3f}%")

    # 3. VQLS forward solve
    logger.info("Solving forward Helmholtz PDE with VQLSSolver (RealAmplitudes, Adam)...")
    vqls_solver = VQLSSolver(
        ansatz=RealAmplitudes(layers=2),
        max_iter=40,
        seed=42,
    )
    problem.solver = vqls_solver
    u_vqls = problem.solve_forward(m, omega, src)
    vqls_fid = np.abs(np.vdot(u_classical, u_vqls)) ** 2 / (
        np.linalg.norm(u_classical) ** 2 * np.linalg.norm(u_vqls) ** 2
    )
    logger.info(f"VQLS wavefield solution u: {np.round(u_vqls, 4)}")
    logger.info(f"VQLS vs Classical wavefield fidelity: {vqls_fid * 100:.3f}%")

    # 4. Adjoint gradient computation
    logger.info("Computing adjoint-state gradient w.r.t velocity...")
    problem.solver = ClassicalSolver()
    d_obs = dataset.observed_data[:, :1, :n_points]
    grad_v = problem.gradient_v(init_model, d_obs)
    logger.info(f"Adjoint gradient dJ/dv: {grad_v}")
    logger.info("FWI HHL vs VQLS comparison completed successfully!")


if __name__ == "__main__":
    run_fwi_hhl_vs_vqls()
