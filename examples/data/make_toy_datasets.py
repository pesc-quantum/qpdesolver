"""
Generate synthetic toy datasets for FWI benchmarks and PDE linear systems,
saving them in standard HDF5 format with qpdesolver schema metadata.
"""
from pathlib import Path
import numpy as np
from loguru import logger

from qpdesolver.applications.fwi.model import VelocityModel, Acquisition
from qpdesolver.applications.fwi.problem import FWIProblem
from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.problems.discretizations.poisson import Poisson1D
from qpdesolver.io import save_fwi_dataset, save_linear_system

def make_toy_datasets(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Generate toy FWI dataset
    fwi_h5_path = output_dir / "toy_fwi.h5"
    logger.info(f"Generating synthetic FWI dataset at {fwi_h5_path}...")

    n_points = 8
    h = 10.0
    acq = Acquisition(
        sources=[1, 4],
        receivers=[1, 2, 3, 4, 5, 6],
        omegas=[0.8, 1.5],
    )
    problem = FWIProblem(acquisition=acq, n_points=n_points, h=h, bc="dirichlet")

    # True model: 2000 m/s background with a 2600 m/s gaussian anomaly
    x = np.linspace(0, 1, n_points)
    v_true = 2000.0 + 600.0 * np.exp(-((x - 0.5) ** 2) / 0.04)
    true_model = VelocityModel(
        velocity=v_true,
        h=h,
        bc="dirichlet",
        metadata={"anomaly_type": "gaussian", "center": 0.5, "peak_velocity": 2600.0},
    )

    # Initial guess: homogeneous 2000 m/s
    v_init = 2000.0 * np.ones(n_points)
    init_model = VelocityModel(
        velocity=v_init,
        h=h,
        bc="dirichlet",
        metadata={"model_type": "homogeneous_background"},
    )

    dataset = problem.simulate_dataset(
        true_model=true_model,
        initial_model=init_model,
        noise_level=0.0,
    )

    save_fwi_dataset(fwi_h5_path, dataset, group="fwi_synthetic_8pt", overwrite=True)
    logger.info("Saved FWI dataset into group 'fwi_synthetic_8pt'.")

    # 2. Generate toy linear systems
    systems_h5_path = output_dir / "toy_systems.h5"
    logger.info(f"Generating PDE linear systems at {systems_h5_path}...")

    # Helmholtz 1D Dirichlet 4-point (2 qubits)
    helm_4 = Helmholtz1D(n_points=4, h=1.0, bc="dirichlet")
    sys_helm_4 = helm_4.system(rhs=helm_4.point_source(1), omega=1.0, name="helmholtz_dirichlet_4pt")
    save_linear_system(systems_h5_path, sys_helm_4, group="helmholtz_4pt", overwrite=True)

    # Helmholtz 1D Periodic 8-point (3 qubits)
    helm_8 = Helmholtz1D(n_points=8, h=1.0, bc="periodic")
    sys_helm_8 = helm_8.system(rhs=helm_8.point_source(2), omega=1.0, name="helmholtz_periodic_8pt")
    save_linear_system(systems_h5_path, sys_helm_8, group="helmholtz_8pt", overwrite=True)

    # Poisson 1D Dirichlet 4-point (2 qubits)
    poisson_4 = Poisson1D(n_points=4, h=1.0, bc="dirichlet")
    sys_poiss_4 = poisson_4.system(rhs=poisson_4.point_source(2), name="poisson_dirichlet_4pt")
    save_linear_system(systems_h5_path, sys_poiss_4, group="poisson_4pt", overwrite=True)

    logger.info(f"Successfully created toy datasets in {output_dir}")

if __name__ == "__main__":
    current_dir = Path(__file__).resolve().parent
    make_toy_datasets(current_dir)
