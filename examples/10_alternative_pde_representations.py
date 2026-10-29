"""
Example 10: Alternative PDE Representations in Quantum Solvers.

Demonstrates:
1. Finite Difference (FDM) vs Galerkin Finite Element (FEM).
2. Fourier Modal / Spectral Representation (where Laplacian is exactly diagonal).
3. Grid-Free Continuous Quantum Function Representation (Quantum PINN / Quantum Coordinate Network).
"""
from pathlib import Path
import numpy as np
import pennylane as qml
from loguru import logger

from qpdesolver.problems.discretizations.helmholtz import Helmholtz1D
from qpdesolver.problems.discretizations.spectral import FourierSpectral1D
from qpdesolver.problems.discretizations.finite_element import FiniteElement1D
from qpdesolver.solvers.classical import ClassicalSolver
from qpdesolver.solvers.hhl import HHLSolver, HHLConfig
from qpdesolver.solvers.vqls import VQLSSolver
from qpdesolver.ansatze import RealAmplitudes
from qpdesolver.optimizers.first_order import Adam
from qpdesolver.optimizers.base import Objective


# ==============================================================================
# Part 1: Comparing Grid-Based Discretizations: FDM vs FEM
# ==============================================================================
def run_fdm_vs_fem_comparison():
    logger.info("--- Part 1: Finite Difference (FDM) vs Finite Element (FEM) ---")
    n_points = 4
    h = 1.0
    omega = 0.8

    # 1. Finite Difference Helmholtz (strong form)
    fdm = Helmholtz1D(n_points=n_points, h=h, bc="dirichlet")
    sys_fdm = fdm.system(rhs=fdm.point_source(1), omega=omega, name="FDM_Helmholtz")

    # 2. Galerkin Finite Element Helmholtz (weak form: (K - omega^2 M) u = F)
    fem = FiniteElement1D(n_points=n_points, h=h, bc="dirichlet")
    sys_fem = fem.system(rhs=fem.point_source(1), omega=omega, name="FEM_Helmholtz")

    # Solve both with Classical LU
    solver_c = ClassicalSolver()
    res_fdm = solver_c.solve(sys_fdm)
    res_fem = solver_c.solve(sys_fem)

    logger.info(f"FDM Operator Condition Number: {sys_fdm.condition_number():.2f}")
    logger.info(f"FEM Operator Condition Number: {sys_fem.condition_number():.2f}")
    logger.info(f"FDM Classical Solution u: {np.round(res_fdm.x, 4)}")
    logger.info(f"FEM Classical Solution u: {np.round(res_fem.x, 4)}")


# ==============================================================================
# Part 2: Modal Fourier Spectral Representation (Diagonal Laplacian)
# ==============================================================================
def run_fourier_spectral_modal():
    logger.info("--- Part 2: Modal Fourier Representation (Diagonal Momentum Space) ---")
    n_points = 4
    spectral = FourierSpectral1D(n_points=n_points, L=2.0 * np.pi)

    # In modal basis, the Laplacian is strictly DIAGONAL: diag(k^2)
    # Use non-integer omega=0.5 to prevent singular resonance (k^2 - omega^2 != 0)
    sys_modal = spectral.system(rhs=[0, 1, 0, 0], omega=0.5, basis="modal", name="Fourier_Modal")

    logger.info("Modal Helmholtz Operator Matrix A (Diagonal):")
    logger.info(f"\n{np.round(sys_modal.A, 3)}")
    logger.info(f"Is strictly diagonal? {np.allclose(sys_modal.A, np.diag(np.diag(sys_modal.A)))}")

    # Solving diagonal systems with HHL is highly efficient (no Trotter errors!)
    solver_hhl = HHLSolver(config=HHLConfig(n_clock=4))
    res_hhl = solver_hhl.solve(sys_modal)
    logger.info(f"HHL Modal Solution Fidelity: {res_hhl.fidelity * 100:.3f}%")


# ==============================================================================
# Part 3: Grid-Free Continuous Quantum Coordinate Representation (Quantum PINN)
# ==============================================================================
class QuantumPINNObjective(Objective):
    """
    Completely grid-free continuous coordinate encoding representation.
    Represents continuous field u_theta(x) = <0| U^dagger(x, theta) Z U(x, theta) |0>.
    Solves continuous boundary-value problem:
        -d^2 u / dx^2 + u(x) = f(x) on [0, pi], with u(0)=0, u(pi)=0.
    Exact analytical solution for f(x) = 2*sin(x): u(x) = sin(x).
    """
    def __init__(self, n_collocation: int = 15, seed: int = 42):
        self.dev = qml.device("default.qubit", wires=2)
        rng = np.random.default_rng(seed)
        # Random continuous collocation points in (0, pi)
        self.x_collocation = np.sort(rng.uniform(0.1, np.pi - 0.1, size=n_collocation))
        self.n_params = 6  # 2 encoding scaling params + 4 ansatz params

        @qml.qnode(self.dev, diff_method="backprop")
        def circuit(x_val, p):
            # 1. Continuous coordinate encoding S(x)
            qml.RY(p[0] * x_val, wires=0)
            qml.RZ(p[1] * x_val, wires=1)
            qml.CNOT(wires=[0, 1])
            # 2. Variational ansatz W(theta)
            qml.RY(p[2], wires=0)
            qml.RY(p[3], wires=1)
            qml.CNOT(wires=[0, 1])
            qml.RY(p[4], wires=0)
            qml.RY(p[5], wires=1)
            # Observable: PauliZ on wire 0
            return qml.expval(qml.PauliZ(0))

        self.u_qnode = circuit

    def predict_u(self, x_val: float, params: np.ndarray) -> float:
        return float(self.u_qnode(x_val, params))

    def value(self, params: np.ndarray) -> float:
        h_diff = 1e-3
        pde_loss = 0.0

        for x in self.x_collocation:
            # Continuous finite-difference differentiation of the quantum coordinate circuit
            u_c = self.u_qnode(x, params)
            u_plus = self.u_qnode(x + h_diff, params)
            u_minus = self.u_qnode(x - h_diff, params)

            # d^2 u / dx^2
            d2u_dx2 = (u_plus - 2.0 * u_c + u_minus) / (h_diff**2)
            # PDE: -d2u/dx2 + u = 2*sin(x)
            f_x = 2.0 * np.sin(x)
            residual = -d2u_dx2 + u_c - f_x
            pde_loss += residual**2

        pde_loss /= len(self.x_collocation)

        # Boundary condition penalties: u(0) = 0 and u(pi) = 0
        bc_0 = (self.u_qnode(0.0, params) - 0.0)**2
        bc_pi = (self.u_qnode(np.pi, params) - 0.0)**2
        bc_loss = 10.0 * (bc_0 + bc_pi)

        return float(pde_loss + bc_loss)

    def grad(self, params: np.ndarray) -> np.ndarray:
        # Central difference w.r.t parameters
        eps = 1e-4
        g = np.zeros_like(params)
        for i in range(len(params)):
            p_p = params.copy(); p_p[i] += eps
            p_m = params.copy(); p_m[i] -= eps
            g[i] = (self.value(p_p) - self.value(p_m)) / (2.0 * eps)
        return g


def run_continuous_quantum_pinn():
    logger.info("--- Part 3: Grid-Free Continuous Quantum PINN Representation ---")
    pinn_obj = QuantumPINNObjective(n_collocation=10, seed=42)

    # Optimize using qpdesolver's native Adam optimizer!
    opt = Adam(lr=0.08)
    p0 = np.array([1.0, 1.0, 0.5, 0.5, 0.5, 0.5])

    logger.info(f"Initial Continuous PINN Loss: {pinn_obj.value(p0):.4e}")
    res = opt.minimize(pinn_obj, p0, max_iter=40)
    logger.info(f"Optimized Continuous PINN Loss: {res.fun:.4e} in {res.n_iter} iterations")

    # Evaluate continuous solution against exact u(x) = sin(x)
    test_points = np.linspace(0, np.pi, 5)
    for x in test_points:
        u_pred = pinn_obj.predict_u(x, res.x)
        u_exact = np.sin(x)
        logger.info(f"x = {x:.2f} | Quantum u(x) = {u_pred:+.4f} | Exact u(x) = {u_exact:+.4f}")


def main():
    logger.info("==================================================================")
    logger.info("   qpdesolver: Alternative Discretizations & Strategies Tutorial ")
    logger.info("==================================================================")
    run_fdm_vs_fem_comparison()
    run_fourier_spectral_modal()
    run_continuous_quantum_pinn()
    logger.info("Tutorial completed successfully!")


if __name__ == "__main__":
    main()
