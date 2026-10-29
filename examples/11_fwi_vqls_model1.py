"""
Example 11: End-to-End VQLS Full Waveform Inversion on model_1.h5 Dataset.

Demonstrates:
1. Loading and verifying Devito-generated seismic datasets from HDF5 (model_1.h5).
2. Dynamic auto-generation if the dataset file is absent.
3. Dimensionless normalization (v0, L0 scaling) via qpdesolver.io.
4. Part 1: 1D depth profile inversion (N=4 points, 2 qubits) comparing VQLSSolver vs ClassicalSolver.
5. Part 2: 2D grid inversion (4x4 = 16 points, 4 qubits) with warm-started VQLS solves.
6. Quantitative success assessment (quantum fidelity >= 99%, misfit reduction, RMSE recovery).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
import numpy as np
from loguru import logger

from qpdesolver.io import load_devito_hdf5
from qpdesolver.generator import generate_seismic_sample
from qpdesolver.generator.models import two_layer_velocity
from qpdesolver.applications.fwi.model import (
    VelocityModel,
    VelocityModel2D,
    Acquisition,
    Acquisition2D,
)
from qpdesolver.applications.fwi.problem import FWIProblem, FWIProblem2D
from qpdesolver.applications.fwi.inversion import FWIInversion
from qpdesolver.solvers.classical import ClassicalSolver
from qpdesolver.solvers.vqls import VQLSSolver
from qpdesolver.ansatze import RealAmplitudes
from qpdesolver.optimizers.scipy_wrapper import ScipyOptimizer


def compute_fidelity(u_exact: np.ndarray, u_approx: np.ndarray) -> float:
    """Compute state overlap fidelity F = |<u_exact|u_approx>|^2 / (||u_exact||^2 ||u_approx||^2)."""
    norm_exact = np.linalg.norm(u_exact)
    norm_approx = np.linalg.norm(u_approx)
    if norm_exact == 0.0 or norm_approx == 0.0:
        return 0.0
    overlap = np.vdot(u_exact, u_approx)
    return float((np.abs(overlap) ** 2) / ((norm_exact ** 2) * (norm_approx ** 2)))


def compute_rel_residual(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:
    """Compute relative residual ||A x - b||_2 / ||b||_2."""
    norm_b = np.linalg.norm(b)
    if norm_b == 0.0:
        return float(np.linalg.norm(A @ x - b))
    return float(np.linalg.norm(A @ x - b) / norm_b)


def compute_rmse(v_est: np.ndarray, v_true: np.ndarray) -> float:
    """Compute root-mean-square error."""
    return float(np.sqrt(np.mean((v_est - v_true) ** 2)))


def run_fwi_vqls(test_mode: bool = False, data_path: Path | None = None) -> int:
    start_time = time.time()
    logger.info(f"Starting FWI VQLS on model_1 (test_mode={test_mode})...")

    # 1. Locate or auto-generate dataset
    if data_path is None:
        data_path = Path(__file__).resolve().parent / "data" / "model_1.h5"

    if not data_path.is_file():
        logger.warning(f"Dataset {data_path} not found. Generating synthetic seismic sample...")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        vp_sample = two_layer_velocity(shape=(101, 101), v_top=1.5, v_bottom=2.5)
        generate_seismic_sample(data_path, vp=vp_sample, backend="auto")
        logger.info(f"Successfully generated dataset at {data_path}")

    # 2. Inspect data and demonstrate normalization
    logger.info(f"Loading raw model from {data_path}...")
    raw_data = load_devito_hdf5(data_path, normalize=False)
    vp_raw = raw_data["model"]["vp"]
    logger.info(
        f"Raw model shape: {vp_raw.shape}, "
        f"velocity range: [{vp_raw.min():.2f}, {vp_raw.max():.2f}] km/s"
    )

    logger.info("Demonstrating dimensionless normalization...")
    norm_data = load_devito_hdf5(data_path, normalize=True, unit_target="dimensionless")
    vp_norm = norm_data["model"]["vp"]
    norm_attrs = norm_data["attrs"]["normalization"]
    logger.info(
        f"Normalized model velocity range: [{vp_norm.min():.4f}, {vp_norm.max():.4f}] (dimensionless)"
    )
    logger.info(
        f"Normalization scales: v0={norm_attrs['v0']:.3f} km/s, "
        f"L0={norm_attrs['L0']:.1f} m, "
        f"velocity_scale={norm_attrs['velocity_scale']:.4f}"
    )

    # Inversion parameters adapted for quick regression or full run
    vqls_max_iter = 12 if test_mode else 35
    fwi_max_iter = 1 if test_mode else 3

    # =========================================================================
    # PART 1: 1D Depth Profile Inversion (N=4, 2 qubits)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PART 1: 1D Depth Profile Inversion (N=4, 2 qubits)")
    logger.info("=" * 60)

    # Extract central depth profile and downsample to 4 points
    nx_raw, nz_raw = vp_raw.shape
    v_1d_raw = vp_raw[nx_raw // 2, :]
    v_1d_true = np.interp(np.linspace(0, 1, 4), np.linspace(0, 1, len(v_1d_raw)), v_1d_raw)
    v_1d_init = np.full(4, float(np.mean(v_1d_true) * 0.9))

    dx_1d = 250.0
    true_model_1d = VelocityModel(velocity=v_1d_true, h=dx_1d)
    init_model_1d = VelocityModel(velocity=v_1d_init, h=dx_1d)

    acq_1d = Acquisition(sources=[0], receivers=[0, 1, 2, 3], omegas=[0.5])
    prob_1d_c = FWIProblem(acquisition=acq_1d, n_points=4, h=dx_1d, solver=ClassicalSolver())

    vqls_1d = VQLSSolver(
        ansatz=RealAmplitudes(layers=2),
        max_iter=vqls_max_iter,
        seed=42,
    )
    prob_1d_v = FWIProblem(
        acquisition=acq_1d,
        n_points=4,
        h=dx_1d,
        solver=vqls_1d,
        warm_start_solves=True,
    )

    # Forward solve check & fidelity
    m_1d_true = 1.0 / (v_1d_true ** 2)
    b_1d = prob_1d_c.discretization.point_source(0)
    sys_1d = prob_1d_c._build_system(m_1d_true, 0.5, b_1d)
    u_1d_c = prob_1d_c.solver.solve(sys_1d).x
    u_1d_v = prob_1d_v.solver.solve(sys_1d).x

    fid_1d = compute_fidelity(u_1d_c, u_1d_v)
    rel_res_1d = compute_rel_residual(sys_1d.A, sys_1d.b, u_1d_v)
    logger.info(f"1D Quantum Linear Solve Fidelity: {fid_1d * 100:.4f}%")
    logger.info(f"1D Quantum Linear Solve Relative Residual: {rel_res_1d:.4e}")

    # Simulate observed data from true model
    dataset_1d = prob_1d_c.simulate_dataset(true_model=true_model_1d, initial_model=init_model_1d)

    # Run classical FWI
    inv_1d_c = FWIInversion(
        problem=prob_1d_c,
        dataset=dataset_1d,
        optimizer=ScipyOptimizer(method="L-BFGS-B"),
        bounds=(1.0, 3.5),
        max_iter=fwi_max_iter,
    )
    res_1d_c = inv_1d_c.run(init_model_1d)

    # Run VQLS FWI
    inv_1d_v = FWIInversion(
        problem=prob_1d_v,
        dataset=dataset_1d,
        optimizer=ScipyOptimizer(method="L-BFGS-B"),
        bounds=(1.0, 3.5),
        max_iter=fwi_max_iter,
    )
    res_1d_v = inv_1d_v.run(init_model_1d)

    misfit_1d_init = res_1d_v.misfit_history[0]
    misfit_1d_final = res_1d_v.final_misfit
    dJ_1d_rel = (
        ((misfit_1d_init - misfit_1d_final) / misfit_1d_init) * 100.0
        if misfit_1d_init > 0
        else 0.0
    )
    rmse_1d_init = compute_rmse(v_1d_init, v_1d_true)
    rmse_1d_final = compute_rmse(res_1d_v.final_velocity.velocity, v_1d_true)

    logger.info(
        f"1D FWI Misfit: {misfit_1d_init:.4e} -> {misfit_1d_final:.4e} (reduction: {dJ_1d_rel:.2f}%)"
    )
    logger.info(f"1D Velocity RMSE: {rmse_1d_init:.4f} -> {rmse_1d_final:.4f} km/s")

    # =========================================================================
    # PART 2: 2D Grid Inversion (4x4 = 16 points, 4 qubits)
    # =========================================================================
    logger.info("=" * 60)
    logger.info("PART 2: 2D Subsampled Grid Inversion (4x4 = 16 points, 4 qubits)")
    logger.info("=" * 60)

    model_2d_raw = VelocityModel2D(velocity=vp_raw, hx=10.0, hz=10.0)
    res_2d_true = model_2d_raw.resample((4, 4), smooth_sigma=1.0)
    v_2d_true = res_2d_true.velocity
    v_2d_init = np.full((4, 4), float(np.mean(v_2d_true) * 0.9))
    res_2d_init = VelocityModel2D(velocity=v_2d_init, hx=res_2d_true.hx, hz=res_2d_true.hz)

    acq_2d = Acquisition2D(
        sources=[(0, 0)],
        receivers=[(0, 0), (1, 1), (2, 2), (3, 3)],
        omegas=[0.5],
    )
    prob_2d_c = FWIProblem2D(
        acquisition=acq_2d,
        nx=4,
        nz=4,
        hx=res_2d_true.hx,
        hz=res_2d_true.hz,
        solver=ClassicalSolver(),
    )

    vqls_2d = VQLSSolver(
        ansatz=RealAmplitudes(layers=2),
        max_iter=vqls_max_iter,
        seed=42,
    )
    prob_2d_v = FWIProblem2D(
        acquisition=acq_2d,
        nx=4,
        nz=4,
        hx=res_2d_true.hx,
        hz=res_2d_true.hz,
        solver=vqls_2d,
        warm_start_solves=True,
    )

    # Forward solve check & fidelity
    m_2d_true = 1.0 / (v_2d_true ** 2)
    b_2d = prob_2d_c.discretization.point_source(0)
    sys_2d = prob_2d_c._build_system(m_2d_true, 0.5, b_2d)
    u_2d_c = prob_2d_c.solver.solve(sys_2d).x
    u_2d_v = prob_2d_v.solver.solve(sys_2d).x

    fid_2d = compute_fidelity(u_2d_c, u_2d_v)
    rel_res_2d = compute_rel_residual(sys_2d.A, sys_2d.b, u_2d_v)
    logger.info(f"2D Quantum Linear Solve Fidelity: {fid_2d * 100:.4f}%")
    logger.info(f"2D Quantum Linear Solve Relative Residual: {rel_res_2d:.4e}")

    # Simulate observed data from true model
    dataset_2d = prob_2d_c.simulate_dataset(true_model=res_2d_true, initial_model=res_2d_init)

    # Run VQLS FWI with warm start
    inv_2d_v = FWIInversion(
        problem=prob_2d_v,
        dataset=dataset_2d,
        optimizer=ScipyOptimizer(method="L-BFGS-B"),
        bounds=(1.0, 3.5),
        max_iter=fwi_max_iter,
    )
    res_2d_v = inv_2d_v.run(res_2d_init)

    misfit_2d_init = res_2d_v.misfit_history[0]
    misfit_2d_final = res_2d_v.final_misfit
    dJ_2d_rel = (
        ((misfit_2d_init - misfit_2d_final) / misfit_2d_init) * 100.0
        if misfit_2d_init > 0
        else 0.0
    )
    rmse_2d_init = compute_rmse(v_2d_init, v_2d_true)
    rmse_2d_final = compute_rmse(res_2d_v.final_velocity.velocity, v_2d_true)

    logger.info(
        f"2D FWI Misfit: {misfit_2d_init:.4e} -> {misfit_2d_final:.4e} (reduction: {dJ_2d_rel:.2f}%)"
    )
    logger.info(f"2D Velocity RMSE: {rmse_2d_init:.4f} -> {rmse_2d_final:.4f} km/s")

    # =========================================================================
    # SUCCESS ASSESSMENT AND SUMMARY BANNER
    # =========================================================================
    avg_fidelity = 0.5 * (fid_1d + fid_2d) * 100.0
    avg_misfit_reduction = 0.5 * (dJ_1d_rel + dJ_2d_rel)
    is_solved = (
        fid_1d >= 0.99
        and fid_2d >= 0.99
        and dJ_1d_rel > 0.0
        and dJ_2d_rel > 0.0
    )
    status_str = "SIM" if is_solved else "NÃO"

    print("\n" + "=" * 60)
    print(f"FWI VQLS RESOLVIDO COM SUCESSO: [{status_str}]")
    print(f"Taxa de Redução do Misfit (Sucesso FWI): {avg_misfit_reduction:.1f}%")
    print(f"Fidelidade do Estado Quântico (VQLS): {avg_fidelity:.2f}%")
    print("=" * 60)
    print(f"FWI VQLS on model_1 completed successfully in {time.time() - start_time:.2f}s.\n")

    return 0 if is_solved else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Full Waveform Inversion with VQLSSolver on model_1.h5"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Run quick regression test mode (< 10 seconds)",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Path to model_1.h5 dataset file",
    )
    args = parser.parse_args()

    exit_code = run_fwi_vqls(test_mode=args.test_mode, data_path=args.data_path)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
