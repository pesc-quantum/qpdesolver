"""
Example 09: Researcher Walkthrough - Building a Custom PDE and Custom Ansatz.

This end-to-end tutorial demonstrates how an academic researcher can:
1. Formulate a new PDE discretization (1D Advection-Diffusion-Reaction).
2. Design and register a novel custom ansatz (Chained Ry-Rz-CZ).
3. Solve the custom PDE using both HHL and VQLS with automatic norm recovery.
4. Compute the Dynamical Lie Algebra (DLA) dimension and QFIM rank of the custom ansatz.
5. Persist the new PDE problem and results into standard HDF5.
"""
from typing import Sequence, List, Optional
from pathlib import Path
import numpy as np
import pennylane as qml
from loguru import logger

from qpdesolver.core.registry import REGISTRY_ANSATZE, REGISTRY_DISCRETIZATIONS
from qpdesolver.ansatze.base import Ansatz
from qpdesolver.problems.discretizations.base import PDEDiscretization
from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.solvers.classical import ClassicalSolver
from qpdesolver.solvers.hhl import HHLSolver, HHLConfig
from qpdesolver.solvers.vqls import VQLSSolver
from qpdesolver.solvers.vqls.costs import LocalCost
from qpdesolver.optimizers.first_order import Adam
from qpdesolver.trainability.dla import dynamical_lie_algebra
from qpdesolver.trainability.geometry import qfim_statistics
from qpdesolver.io import save_linear_system, save_results


# ==============================================================================
# Step 1: Define Custom Research PDE (1D Advection-Diffusion-Reaction)
# ==============================================================================
@REGISTRY_DISCRETIZATIONS.register("tutorial_adv_diff_react")
class AdvectionDiffusionReaction1D(PDEDiscretization):
    """
    1D Steady Advection-Diffusion-Reaction operator:
        -nu * d^2 u / dx^2 + c * du / dx + sigma * u = f(x)
    """
    def operator(self, nu: float = 1.0, c: float = 0.2, sigma: float = 0.05) -> np.ndarray:
        N = self.n_points
        h = self.h
        h2 = h**2

        diag_c = (2.0 * nu / h2) + sigma
        diag_sub = (-nu / h2) - (c / (2.0 * h))
        diag_sup = (-nu / h2) + (c / (2.0 * h))

        A = diag_c * np.eye(N) + diag_sup * np.eye(N, k=1) + diag_sub * np.eye(N, k=-1)
        if self.bc == "periodic" and N > 2:
            A[0, -1] = diag_sub
            A[-1, 0] = diag_sup
        return A

    def system(self, rhs: np.ndarray, nu: float = 1.0, c: float = 0.2, sigma: float = 0.05, name: Optional[str] = None) -> LinearSystem:
        A = self.operator(nu=nu, c=c, sigma=sigma)
        return LinearSystem(
            A=A,
            b=rhs,
            name=name or f"AdvDiffReact_N={self.n_points}",
            metadata={"nu": nu, "c": c, "sigma": sigma, "bc": self.bc, "h": self.h},
        )


# ==============================================================================
# Step 2: Define Custom Research Ansatz (Chained Ry-Rz-CZ)
# ==============================================================================
@REGISTRY_ANSATZE.register("tutorial_chained_ry_rz_cz")
class ChainedRyRzCZAnsatz(Ansatz):
    """
    Custom Research Ansatz:
    Alternating Ry-Rz single-qubit rotations with circular CZ entangling gates.
    """
    def __init__(self, layers: int = 2):
        self.name = "tutorial_chained_ry_rz_cz"
        self.layers = layers

    def num_params(self, n_qubits: int) -> int:
        return 2 * n_qubits * (self.layers + 1)

    def apply(self, params: Sequence[float], wires: Sequence[int]) -> None:
        n_q = len(wires)
        idx = 0

        # Initial single-qubit rotation layer
        for w in wires:
            qml.RY(params[idx], wires=w); idx += 1
            qml.RZ(params[idx], wires=w); idx += 1

        # Repeating entangling blocks
        for l in range(self.layers):
            for i in range(n_q):
                target = (i + 1) % n_q
                qml.CZ(wires=[wires[i], wires[target]])

            for w in wires:
                qml.RY(params[idx], wires=w); idx += 1
                qml.RZ(params[idx], wires=w); idx += 1

    def with_layers(self, layers: int) -> "ChainedRyRzCZAnsatz":
        return ChainedRyRzCZAnsatz(layers=layers)

    def generators(self, n_qubits: int) -> Optional[List[str]]:
        gen_list = []
        for i in range(n_qubits):
            y_str = list("I" * n_qubits); y_str[i] = "Y"; gen_list.append("".join(y_str))
            z_str = list("I" * n_qubits); z_str[i] = "Z"; gen_list.append("".join(z_str))
        return gen_list


# ==============================================================================
# Step 3: Execution and Comparison Pipeline
# ==============================================================================
def main():
    logger.info("==================================================================")
    logger.info("   qpdesolver: Custom PDE & Custom Ansatz Academic Walkthrough    ")
    logger.info("==================================================================")

    # 1. Build PDE system (4 points = 2 qubits)
    n_points = 4
    pde = AdvectionDiffusionReaction1D(n_points=n_points, h=0.25, bc="dirichlet")
    source_b = pde.point_source(1)  # Delta source at point 1
    system = pde.system(rhs=source_b, nu=1.0, c=0.2, sigma=0.05)

    logger.info(f"Assembled PDE: {system.name}")
    logger.info(f" - Grid Points: {system.size} (requires {system.n_qubits} qubits)")
    logger.info(f" - Condition Number kappa: {system.condition_number():.2f}")
    logger.info(f" - Is Hermitian? {system.is_hermitian}")

    # 2. Classical Reference Solve
    solver_classic = ClassicalSolver()
    res_classic = solver_classic.solve(system)
    logger.info(f"Classical Solution: {np.round(res_classic.x, 4)}")

    # 3. Solve with HHL (automatic Hermitian dilation if non-Hermitian)
    logger.info("Solving PDE with Harrow-Hassidim-Lloyd (HHL)...")
    solver_hhl = HHLSolver(config=HHLConfig(n_clock=4))
    res_hhl = solver_hhl.solve(system)
    logger.info(f"HHL State Fidelity: {res_hhl.fidelity * 100:.3f}%")
    logger.info(f"HHL Relative Residual: {res_hhl.residual:.3e}")

    # 4. Solve with VQLS using our Custom Ansatz & Local Cost
    custom_ansatz = ChainedRyRzCZAnsatz(layers=2)
    logger.info(f"Solving PDE with VQLS using custom ansatz: {custom_ansatz.name}...")
    solver_vqls = VQLSSolver(
        ansatz=custom_ansatz,
        cost=LocalCost(),
        optimizer=Adam(lr=0.06),
        max_iter=50,
        seed=42,
    )
    res_vqls = solver_vqls.solve(system)
    logger.info(f"VQLS State Fidelity: {res_vqls.fidelity * 100:.3f}%")
    logger.info(f"VQLS Relative Residual: {res_vqls.residual:.3e}")
    logger.info(f"VQLS Circuit Evals: {res_vqls.n_circuit_evals}")

    # 5. Academic Trainability & Geometric Analysis of Custom Ansatz
    logger.info("--- Analyzing Custom Ansatz Geometry ---")
    dla_res = dynamical_lie_algebra(custom_ansatz, n_qubits=2)
    logger.info(f"Dynamical Lie Algebra dim(g): {dla_res.dimension} (Full su(4)=15)")

    qfim_res = qfim_statistics(custom_ansatz, n_qubits=2, n_samples=10, seed=42)
    logger.info(f"QFIM Rank: {qfim_res.rank} / {qfim_res.max_rank_bound}")
    logger.info(f"Effective Dimension: {qfim_res.effective_dimension:.2f}")

    # 6. Save System and Experiment to HDF5
    h5_output = Path(__file__).resolve().parent / "data" / "tutorial_results.h5"
    h5_output.parent.mkdir(parents=True, exist_ok=True)
    save_linear_system(h5_output, system, group="custom_pde", overwrite=True)
    save_results(
        h5_output,
        group="vqls_custom_run",
        results={
            "fidelity": res_vqls.fidelity,
            "residual": res_vqls.residual,
            "solution": res_vqls.x,
            "dla_dim": dla_res.dimension,
            "qfim_rank": qfim_res.rank,
        },
        overwrite=True,
    )
    logger.info(f"Successfully persisted PDE and results into {h5_output}")

    # 7. Generate Didactic Visual Dashboard Plot
    logger.info("Generating researcher walkthrough didactic dashboard...")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5), dpi=220)

    # Panel A: Solution Vectors
    ax_sol = axes[0, 0]
    grid_x = np.arange(system.size)
    ax_sol.plot(grid_x, np.real(res_classic.x), "k-o", lw=2.5, markersize=8, label="Exact Classical (LU)")
    ax_sol.plot(grid_x, np.real(res_hhl.x), "b--s", lw=2, markersize=7, label=f"HHL (Fid: {res_hhl.fidelity*100:.1f}%)")
    ax_sol.plot(grid_x, np.real(res_vqls.x), "r-.^", lw=2, markersize=7, label=f"VQLS (Fid: {res_vqls.fidelity*100:.1f}%)")
    ax_sol.set_title("A. PDE Wavefield Solution Comparison", fontweight="bold")
    ax_sol.set_xlabel("Spatial Grid Index $j$")
    ax_sol.set_ylabel("Solution Amplitude $u_j$")
    ax_sol.set_xticks(grid_x)
    ax_sol.grid(True, linestyle="--", alpha=0.5)
    ax_sol.legend(frameon=True, fontsize=9)

    # Panel B: VQLS Convergence History
    ax_hist = axes[0, 1]
    if res_vqls.history:
        ax_hist.semilogy(res_vqls.history, "m-", lw=2, label="Local Cost $C_L(\\theta)$")
    ax_hist.set_title("B. VQLS Optimization Landscape", fontweight="bold")
    ax_hist.set_xlabel("Optimization Step")
    ax_hist.set_ylabel("Cost Value")
    ax_hist.grid(True, which="both", linestyle="--", alpha=0.5)
    ax_hist.legend(frameon=True, fontsize=9)

    # Panel C: QFIM Eigenvalue Spectrum
    ax_qfim = axes[1, 0]
    ax_qfim.bar(range(len(qfim_res.eigenvalues)), qfim_res.eigenvalues, color="#38bdf8", edgecolor="#0284c7")
    ax_qfim.axhline(1e-4, color="#ef4444", linestyle="--", lw=1.5, label=f"Threshold (Rank={qfim_res.rank})")
    ax_qfim.set_title("C. Custom Ansatz QFIM Spectrum", fontweight="bold")
    ax_qfim.set_xlabel("Eigenvalue Index")
    ax_qfim.set_ylabel("Eigenvalue Magnitude")
    ax_qfim.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax_qfim.legend(frameon=True, fontsize=9)

    # Panel D: DLA Dimension Bar Chart
    ax_dla = axes[1, 1]
    bars = ax_dla.bar(["Custom Chained Ry-Rz-CZ", r"Full $\mathfrak{su}(4)$"], [dla_res.dimension, 15],
                      color=["#818cf8", "#cbd5e1"], edgecolor="#475569")
    ax_dla.bar_label(bars, fmt="%d", padding=3, fontweight="bold")
    ax_dla.set_title(r"D. Dynamical Lie Algebra $\dim(\mathfrak{g})$", fontweight="bold")
    ax_dla.set_ylabel("Subspace Dimension")
    ax_dla.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    dashboard_path = Path(__file__).resolve().parent / "data" / "custom_walkthrough_dashboard.png"
    plt.savefig(dashboard_path)
    plt.close()
    logger.info(f"Saved didactic dashboard visualization to {dashboard_path}")
    logger.info("Researcher tutorial completed successfully!")


if __name__ == "__main__":
    main()
