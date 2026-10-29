"""
Barren Plateau mitigation strategies:
- Layerwise training (Skolik et al. 2021): trains circuits progressively by growing layers.
- Initialization benchmarking (Grant et al. 2019): identity blocks and small angles.
"""
from typing import Dict, Any, List, Sequence, Callable, Optional
import numpy as np

from qpdesolver.problems.linear_system import LinearSystem
from qpdesolver.ansatze.base import Ansatz
from qpdesolver.solvers.vqls.solver import VQLSSolver
from qpdesolver.solvers.base import SolveResult

class LayerwiseTrainer:
    """
    Progressively grows circuit depth by training one layer at a time.
    Mitigates barren plateaus by keeping the active optimization dimension small.
    """
    def __init__(
        self,
        base_ansatz_factory: Callable[[int], Ansatz],
        max_layers: int = 3,
        optimizer: Optional[Any] = None,
        max_iter_per_layer: int = 40,
    ):
        self.base_ansatz_factory = base_ansatz_factory
        self.max_layers = max_layers
        self.optimizer = optimizer
        self.max_iter_per_layer = max_iter_per_layer

    def train(self, system: LinearSystem, seed: int = 42) -> SolveResult:
        current_params = None
        final_result = None

        for l in range(1, self.max_layers + 1):
            ansatz_l = self.base_ansatz_factory(l)
            n_p = ansatz_l.num_params(system.n_qubits)

            if current_params is None:
                warm = None
            else:
                # Expand params: reuse previous, initialize new layer near zero
                warm = np.zeros(n_p, dtype=float)
                warm[:len(current_params)] = current_params

            solver = VQLSSolver(
                ansatz=ansatz_l,
                optimizer=self.optimizer,
                max_iter=self.max_iter_per_layer,
                seed=seed,
            )
            final_result = solver.solve(system, warm_start=warm)
            current_params = np.array(final_result.metadata["optimal_params"])

        return final_result

def compare_initializations(
    system: LinearSystem,
    ansatz: Ansatz,
    strategies: Sequence[str] = ("identity_blocks", "small_angle", "uniform"),
    seeds: Sequence[int] = (42, 123),
    max_iter: int = 50,
) -> Dict[str, Dict[str, Any]]:
    """
    Runs VQLS comparing different parameter initialization strategies across multiple random seeds.
    """
    results: Dict[str, Dict[str, Any]] = {}

    for strat in strategies:
        fids = []
        costs = []
        for s in seeds:
            solver = VQLSSolver(
                ansatz=ansatz,
                init_strategy=strat,
                max_iter=max_iter,
                seed=s,
            )
            res = solver.solve(system)
            fids.append(res.fidelity)
            costs.append(res.metadata["final_cost"])

        results[strat] = {
            "mean_fidelity": float(np.mean(fids)),
            "std_fidelity": float(np.std(fids)),
            "mean_final_cost": float(np.mean(costs)),
            "all_fidelities": fids,
        }

    return results
